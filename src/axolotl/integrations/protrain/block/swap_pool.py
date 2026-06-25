"""Pinned-RAM activation pool for the SWAP block path.

One ``PinnedHostMemory`` region backs a variable-size slab: ``acquire(nbytes)``
carves an exactly-sized slice via a first-fit offset allocator, ``release`` returns
it and coalesces neighbours. A non-recomputed block's saved-for-backward set spans
tensors that differ by >100x in size (FFN intermediates vs norm/LoRA scratch);
fixed equal slots sized to the max wasted ~order-of-magnitude pinned RAM AND
under-counted the per-block tensor population (causing pool exhaustion that spilled
activations back onto the GPU and broke the steady-peak prediction). The slab packs
to the real aggregate instead.
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING

from axolotl.integrations.protrain.chunk.pinned_alloc import PinnedHostMemory
from axolotl.utils.logging import get_logger

if TYPE_CHECKING:
    import torch

LOG = get_logger(__name__)


# Retained for callers that size the pool by a per-block tensor-count heuristic.
DEFAULT_SLOTS_PER_BLOCK: int = 8

# close() drain timeout for in-flight borrows.
DEFAULT_CLOSE_DRAIN_TIMEOUT_S: float = 30.0

# Slab slice alignment (bytes). Keeps DMA targets aligned and bounds free-list
# churn from sub-cache-line splinters.
_SLAB_ALIGNMENT: int = 512


class ActivationSwapPool:
    """Variable-size pinned-host slab for activation swap.

    Backed by a single ``PinnedHostMemory`` region of ``capacity_bytes``.
    ``acquire(nbytes)`` returns ``(slot_id, view)`` where ``view`` is a 1-D uint8
    tensor of exactly ``nbytes`` carved from the region; the caller copies the
    activation into it on the swap stream. ``release(slot_id)`` frees the slice.

    Notes
    -----
    The pool is **stream-agnostic** — D2H/H2D copies happen on the SWAP wrapper's
    stream. Slot ownership is tracked Python-side only; CUDA never sees the
    free-list. Callers MUST synchronize the swap stream with their consumer before
    releasing a slot back to the pool.
    """

    def __init__(
        self,
        capacity_bytes: int,
        *,
        alignment: int = _SLAB_ALIGNMENT,
    ) -> None:
        """Allocate the backing pinned region and seed the free-list."""
        if capacity_bytes <= 0:
            raise ValueError(f"capacity_bytes must be positive, got {capacity_bytes}")
        if alignment <= 0:
            raise ValueError(f"alignment must be positive, got {alignment}")

        self.capacity_bytes = int(capacity_bytes)
        self._align = int(alignment)

        # One pinned region; borrow slot 0 once for the whole capacity and carve
        # sub-views from it. The pool's own free-list governs sub-allocation.
        self._pinned = PinnedHostMemory(n_buffer=1, S_chunk=self.capacity_bytes)
        self._region = self._pinned.buffer(0)

        self._closed = False
        self._closing = False
        # Free extents as (offset, size), kept sorted by offset and coalesced.
        self._free: list[tuple[int, int]] = [(0, self.capacity_bytes)]
        # slot_id -> (offset, aligned_size).
        self._alloc: dict[int, tuple[int, int]] = {}
        self._next_id: int = 0
        self._inflight: int = 0
        self._peak_used: int = 0
        # Lock against autograd worker / main thread races on free + inflight.
        self._lock = threading.Lock()

        LOG.debug(
            "ActivationSwapPool: capacity_bytes=%d alignment=%d precise=%s",
            self.capacity_bytes,
            self._align,
            self._pinned.is_precise_size,
        )

    def _round_up(self, n: int) -> int:
        a = self._align
        return ((int(n) + a - 1) // a) * a

    def acquire(self, nbytes: int) -> tuple[int, "torch.Tensor"]:
        """Reserve an ``nbytes`` slice; return (slot_id, pinned uint8 view of len nbytes)."""
        nbytes = int(nbytes)
        if nbytes <= 0:
            raise ValueError(f"nbytes must be positive, got {nbytes}")
        need = self._round_up(nbytes)
        with self._lock:
            if self._closed or self._closing:
                raise RuntimeError("ActivationSwapPool is closed")
            # First-fit over the coalesced free-list.
            idx = -1
            for i, (_off, sz) in enumerate(self._free):
                if sz >= need:
                    idx = i
                    break
            if idx < 0:
                raise RuntimeError(
                    f"ActivationSwapPool exhausted (need {need}B, "
                    f"capacity={self.capacity_bytes}B, in-flight={self._inflight}); "
                    "size the pool to the per-block saved aggregate x n_swap, or "
                    "verify the SWAP wrapper releases slots after backward."
                )
            off, sz = self._free[idx]
            if sz == need:
                self._free.pop(idx)
            else:
                self._free[idx] = (off + need, sz - need)
            slot_id = self._next_id
            self._next_id += 1
            self._alloc[slot_id] = (off, need)
            self._inflight += 1
            used = self.capacity_bytes - sum(s for _, s in self._free)
            if used > self._peak_used:
                self._peak_used = used
            # narrow() of a view never allocates; rollback path is defensive.
            try:
                view = self._region.narrow(0, off, nbytes)
            except BaseException:
                del self._alloc[slot_id]
                self._inflight -= 1
                self._free.append((off, need))
                self._coalesce_locked()
                raise
        return slot_id, view

    def view(self, slot_id: int) -> "torch.Tensor":
        """Return the pinned uint8 view of an allocated slot's full aligned extent.

        Used by the unpack path to re-read a slot mid-flight (between pack and
        release). Slicing back to the original ``nbytes`` is the caller's job.
        """
        with self._lock:
            if self._closed:
                raise RuntimeError("ActivationSwapPool is closed")
            ext = self._alloc.get(slot_id)
            if ext is None:
                raise KeyError(
                    f"ActivationSwapPool.view: slot {slot_id} is not allocated"
                )
            off, size = ext
            return self._region.narrow(0, off, size)

    def release(self, slot_id: int) -> None:
        """Return ``slot_id``'s slice to the free-list; pool does NOT issue stream syncs."""
        with self._lock:
            # release() proceeds during _closing so close()'s drain can converge.
            if self._closed:
                return
            ext = self._alloc.pop(slot_id, None)
            if ext is None:
                LOG.warning(
                    "ActivationSwapPool.release: unknown/double-released slot %d; ignored",
                    slot_id,
                )
                return
            self._free.append(ext)
            self._coalesce_locked()
            self._inflight -= 1

    def _coalesce_locked(self) -> None:
        """Merge adjacent free extents. Caller holds ``self._lock``."""
        self._free.sort()
        merged: list[tuple[int, int]] = []
        for off, sz in self._free:
            if merged and merged[-1][0] + merged[-1][1] == off:
                p_off, p_sz = merged[-1]
                merged[-1] = (p_off, p_sz + sz)
            else:
                merged.append((off, sz))
        self._free = merged

    @property
    def total_bytes(self) -> int:
        """Total pinned-host bytes held by the pool."""
        return self.capacity_bytes

    @property
    def peak_used_bytes(self) -> int:
        """High-water mark of simultaneously-allocated bytes (diagnostics/sizing)."""
        with self._lock:
            return self._peak_used

    @property
    def free_bytes(self) -> int:
        with self._lock:
            return sum(s for _, s in self._free)

    @property
    def inflight_count(self) -> int:
        with self._lock:
            return self._inflight

    def close(
        self,
        drain_timeout: float = DEFAULT_CLOSE_DRAIN_TIMEOUT_S,
        poll_interval: float = 0.01,
    ) -> None:
        """Free the pinned region. Three-phase: closing flag → drain inflight → free.  Idempotent."""
        with self._lock:
            if self._closed:
                return
            if self._closing:
                raise RuntimeError(
                    "ActivationSwapPool.close: close in progress on another thread"
                )
            self._closing = True
        try:
            # Drain in-flight slices; release() converges _inflight to 0.
            deadline = time.monotonic() + max(0.0, float(drain_timeout))
            while True:
                with self._lock:
                    inflight = self._inflight
                if inflight == 0:
                    break
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        f"ActivationSwapPool.close: timed out after "
                        f"{drain_timeout:.3f}s waiting for {inflight} "
                        "in-flight slice(s) to drain. Caller is missing "
                        "release() pairs or the swap stream has not "
                        "synchronized — retry close() after stragglers retire."
                    )
                time.sleep(max(0.0, float(poll_interval)))
            # Drain complete; drop the region borrow then free the pinned allocation.
            self._region = None  # type: ignore[assignment]
            self._pinned.release_buffer(0)
            self._pinned.close()
        except BaseException:
            with self._lock:
                self._closing = False
            raise
        with self._lock:
            self._closed = True
            self._free.clear()
            self._alloc.clear()
            self._inflight = 0

    def __del__(self) -> None:  # noqa: D401
        try:
            # Non-blocking cleanup; 30s drain would stall shutdown.
            self.close(drain_timeout=0)
        except Exception:  # noqa: BLE001 — destructor must not throw
            LOG.debug("ActivationSwapPool.__del__ cleanup skipped", exc_info=True)


__all__ = ["ActivationSwapPool", "DEFAULT_SLOTS_PER_BLOCK"]
