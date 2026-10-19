"""M3 tests for the BlockMode.OFFLOAD rollout (Option B).

Three tests covering the M3 exit criteria from
``BLOCK_MODE_OFFLOAD_DESIGN.md`` §7:

1. ``test_offload_mode_pre_backward_gather`` — invoking
   ``Scheduler.pre_block_backward`` for an OFFLOAD-mode block must
   leave the block's chunks GPU-resident, so the unpack hook hits the
   resident fast-path. End-to-end forward+backward through the
   scheduler matches a reference run within fp32 tolerance.

2. ``test_drain_deferred_offloads_at_end_of_iter`` — exercises the new
   ``ChunkManager.drain_deferred_offloads`` helper. With a live
   ``BackwardHandle`` outstanding and ``reduce_grads_and_offload``
   already run, the chunk sits in ``_deferred_offloads``; calling
   drain while refcount > 0 leaves it resident; dropping the handle
   then calling drain offloads it. This is the §3.3 defensive drain.

3. ``test_offload_mode_3iter_smoke`` — the M3 acceptance test: a
   tiny 2-block model with one OFFLOAD-wrapped block, looped
   forward+backward 3 iterations through the scheduler (NOT manual
   offload as in M2's roundtrip test). Compares grads against a
   reference each iter to validate the full scheduler integration.

All three tests require CUDA — the chunk-manager offload path is
defined in terms of GPU memory motion.
"""

from __future__ import annotations

from typing import cast

import pytest

from axolotl.integrations.protrain.types import (
    BlockId,
    BlockMode,
    BlockStrategyMap,
    ChunkId,
    ParamId,
)

# ---------------------------------------------------------------------------
# Helpers — same structural pattern as test_offload_mode_m2.py, copied
# locally for test-file isolation.
# ---------------------------------------------------------------------------


def _tiny_model(hidden: int = 64, n_layers: int = 2):
    """A tiny ``n_layers``-block model: embed -> transformer.h.[*] -> head.

    The block list lives at ``transformer.h`` so ``discover_blocks``
    (one of the ``_KNOWN_BLOCK_PATHS``) finds it without needing the
    attention-attribute heuristic. This is the GPT-2-style path; the
    M3 ``install_hooks`` test exercises that discovery.
    """
    import torch
    from torch import nn

    class _TransformerCore(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.h = nn.ModuleList(
                [nn.Linear(hidden, hidden, bias=False) for _ in range(n_layers)]
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            for layer in self.h:
                x = layer(x)
            return x

    class TinyTransformer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Linear(hidden, hidden, bias=False)
            self.transformer = _TransformerCore()
            self.head = nn.Linear(hidden, hidden, bias=False)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            x = self.embed(x)
            x = self.transformer(x)
            return self.head(x)

    torch.manual_seed(0)
    return TinyTransformer()


def _build_layout_for(model, S_chunk: int):
    """Build a ChunkLayout where each ``transformer.h.{i}`` linear is its own chunk."""
    from axolotl.integrations.protrain.chunk.layout import build_layout

    block_spans: dict[BlockId, list[ParamId]] = {}
    prefix = "transformer.h."
    for name, _ in model.named_parameters():
        if name.startswith(prefix):
            idx = int(name[len(prefix) :].split(".")[0])
            block_spans.setdefault(cast(BlockId, idx), []).append(cast(ParamId, name))

    exec_order = [cast(ParamId, n) for n, _ in model.named_parameters()]
    return build_layout(model, exec_order, S_chunk, block_spans)


def _build_chunk_manager(
    model, n_persist: int, S_chunk: int, n_buffer: int | None = None
):
    """Assemble a :class:`ChunkManager` for M3 tests."""
    import torch

    from axolotl.integrations.protrain.chunk.buffer_pool import BufferPool
    from axolotl.integrations.protrain.chunk.manager import ChunkManager
    from axolotl.integrations.protrain.chunk.pinned_alloc import PinnedHostMemory

    layout = _build_layout_for(model, S_chunk)
    if n_buffer is None:
        n_buffer = max(2, min(4, layout.N_chunk - n_persist))
    host = PinnedHostMemory(n_buffer=n_buffer, S_chunk=layout.S_chunk)
    pool = BufferPool(
        n_buffer=n_buffer,
        S_chunk=layout.S_chunk,
        pinned_host=host,
        device=torch.device("cuda"),
    )
    mgr = ChunkManager(
        model=model,
        layout=layout,
        n_persist=n_persist,
        buffer_pool=pool,
        cpu_optim=None,
        gpu_optim=None,
        device=torch.device("cuda"),
    )
    return mgr, layout, pool, host


def _build_scheduler(chunk_manager, layout, block_map: BlockStrategyMap):
    """Construct a Scheduler with default-ish bandwidth knobs for tests."""
    from axolotl.integrations.protrain.runtime.scheduler import Scheduler

    return Scheduler(
        chunk_manager=chunk_manager,
        block_map=block_map,
        layout=layout,
        # Bandwidths are stored but not consumed by the path we exercise.
        effective_h2d_bps=1e10,
        effective_d2h_bps=1e10,
    )


def _wrap_block_offload(model, layer_idx: int, manager):
    """Replace ``model.transformer.h[layer_idx]`` with an OffloadedBlock."""
    from axolotl.integrations.protrain.block.dispatcher import wrap_block
    from axolotl.integrations.protrain.block.offload import OffloadedBlock

    original = model.transformer.h[layer_idx]
    wrapped = wrap_block(original, BlockMode.OFFLOAD)
    assert isinstance(wrapped, OffloadedBlock)
    # Make the pack hook capture every saved tensor in our small model
    # (default 1 MiB threshold ≫ our linear weights).
    wrapped.SIZE_THRESHOLD_BYTES = 0
    # Note: attach_runtime is intentionally NOT called here. The tests
    # below either install hooks (which calls it) or attach manually
    # to validate specific paths.
    model.transformer.h[layer_idx] = wrapped
    return wrapped, original


def _h_block_chunks(layout, original_block, layer_idx: int) -> list[ChunkId]:
    """Return the sorted list of chunks covering ``transformer.h[layer_idx]``'s params."""
    h_param_names = {
        f"transformer.h.{layer_idx}.{n}" for n, _ in original_block.named_parameters()
    }
    return sorted(
        {
            layout.param_to_chunk[cast(ParamId, name)]
            for name in h_param_names
            if cast(ParamId, name) in layout.param_to_chunk
        }
    )


def _canonical_pname(name: str) -> str:
    """Strip the ``.block.`` segment introduced by OffloadedBlock wrapping."""
    return name.replace(".block.", ".")


# ---------------------------------------------------------------------------
# Test 1: pre_block_backward gathers the OFFLOAD chunk
# ---------------------------------------------------------------------------


@pytest.mark.gpu
def test_offload_mode_pre_backward_gather() -> None:
    """Manual pre_block_backward for an OFFLOAD block leaves the chunk resident.

    Builds a tiny 2-block model with block 0 wrapped as OFFLOAD,
    installs the scheduler+hooks, runs forward (chunk gets offloaded
    in the absence of forward-side hooks because we drive it here),
    manually invokes ``pre_block_backward(0)``, asserts the chunk is
    now resident again. Then runs backward and confirms grads match
    a reference forward+backward pair.
    """
    pytest.importorskip("torch")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA runtime")

    torch.cuda.empty_cache()
    device = torch.device("cuda")

    hidden = 32
    n_layers = 2

    # Reference model + identical-init ProTrain model.
    torch.manual_seed(123)
    ref_model = _tiny_model(hidden=hidden, n_layers=n_layers).to(device)
    torch.manual_seed(123)
    pt_model = _tiny_model(hidden=hidden, n_layers=n_layers).to(device)

    per_layer_bytes = hidden * hidden * 4
    S_chunk = per_layer_bytes + 1024

    mgr, layout, pool, host = _build_chunk_manager(
        pt_model, n_persist=1, S_chunk=S_chunk
    )
    mgr.materialize_offload()
    # Drop per-param post-grad hooks so backward leaves grads on
    # ``param.grad`` directly (we have no cpu_optim wired).
    mgr.uninstall()

    # Wrap h.0 in OffloadedBlock and wire it via attach_runtime
    # (mimics what ``install_hooks`` does in production).
    wrapped, original_h0 = _wrap_block_offload(pt_model, 0, mgr)

    # Build a block_map: block 0 is OFFLOAD, block 1 is NONE.
    block_map: BlockStrategyMap = {
        cast(BlockId, 0): BlockMode.OFFLOAD,
        cast(BlockId, 1): BlockMode.NONE,
    }
    scheduler = _build_scheduler(mgr, layout, block_map)
    wrapped.attach_runtime(mgr, scheduler)

    h0_chunks = _h_block_chunks(layout, original_h0, 0)
    assert h0_chunks, "test setup: expected h.0 to own at least one chunk"
    # The non-persistent set should include at least one h.0 chunk for
    # this test to be meaningful (otherwise OFFLOAD has nothing to do).
    non_persistent = sorted(mgr._non_persistent_ids)
    h0_non_persist = [cid for cid in h0_chunks if cid in non_persistent]
    assert h0_non_persist, (
        f"test setup: h.0 chunks {h0_chunks} should include at least one "
        f"non-persistent chunk (got non_persistent={non_persistent})"
    )

    torch.manual_seed(11)
    x = torch.randn(4, hidden, device=device)

    # --- Reference run -------------------------------------------------
    ref_model.zero_grad()
    ref_loss = ref_model(x).sum()
    ref_loss.backward()

    # --- ProTrain run --------------------------------------------------
    pt_model.zero_grad()
    # Gather every non-persistent chunk so forward sees its weights.
    for cid in non_persistent:
        mgr.gather(cid)
    pt_loss = pt_model(x)
    pt_loss = pt_loss.sum()

    # Simulate the post_block_forward offload that the real scheduler
    # would do for block 0 (the wrapper module has no forward-side
    # hooks installed in this test). After this call, the saved
    # tensors that aliased h.0's chunks are metadata-only and the
    # pool's free-list has the slots back (the BufferPool retains the
    # resident TAG for forward→backward reuse, but the slot is now
    # eligible for eviction by an unrelated ``acquire``).
    for cid in h0_non_persist:
        mgr.offload(cid)

    # Force a real eviction of h.0's chunk slot by acquiring every
    # other non-persistent chunk. This simulates the worst case where
    # forward→backward tag-preservation can't help: another block's
    # gather has reclaimed the slot, so pre_block_backward MUST
    # re-issue the H2D copy.
    other_non_persist = [cid for cid in non_persistent if cid not in h0_non_persist]
    for cid in other_non_persist:
        mgr.gather(cid)
    # After the cascade above, h.0's chunk should have been evicted by
    # the LRU/free-list policy (we only have ~4 buffer slots). Some
    # configurations may keep the tag — assert the *meaningful*
    # invariant via the manager's resident introspection rather than
    # the pool's tag-preserved state.
    pre_state = {cid: pool.lookup_resident(cid) is not None for cid in h0_non_persist}

    # --- The actual M3 assertion: pre_block_backward re-gathers -------
    scheduler.pre_block_backward(cast(BlockId, 0))
    for cid in h0_non_persist:
        assert pool.lookup_resident(cid) is not None, (
            f"after Scheduler.pre_block_backward(0), chunk {int(cid)} "
            "must be resident so the OFFLOAD unpack hook hits the "
            "fast-path. This is the §3.3 ordering invariant. "
            f"(pre-state resident: {pre_state[cid]})"
        )

    # Run backward; the unpack hook should hit the resident fast-path
    # (gather is idempotent on a resident chunk) and the gradient
    # kernels should consume views into the gathered pool buffer.
    pt_loss.backward()

    # --- Grad parity ---------------------------------------------------
    ref_named = dict(ref_model.named_parameters())
    for name, pp in pt_model.named_parameters():
        canon = _canonical_pname(name)
        rp = ref_named.get(canon)
        assert rp is not None, f"no ref param matches {name} (canon={canon})"
        if rp.grad is None and pp.grad is None:
            continue
        assert pp.grad is not None, f"pt grad missing for {name}"
        assert rp.grad is not None, f"ref grad missing for {canon}"
        assert torch.allclose(rp.grad, pp.grad, atol=1e-4, rtol=1e-4), (
            f"grad mismatch on {name}: "
            f"max_abs_diff={(rp.grad - pp.grad).abs().max().item():.6e}"
        )

    assert torch.allclose(ref_loss, pt_loss, atol=1e-4, rtol=1e-4), (
        f"loss mismatch ref={ref_loss.item():.6e} pt={pt_loss.item():.6e}"
    )

    # Cleanup.
    scheduler.drain()
    mgr.uninstall()
    host.close()
    del pool


# ---------------------------------------------------------------------------
# Test 2: drain_deferred_offloads end-of-iter helper
# ---------------------------------------------------------------------------


@pytest.mark.gpu
def test_drain_deferred_offloads_at_end_of_iter() -> None:
    """drain_deferred_offloads only offloads chunks whose refcount==0.

    Reproduces the §3.3 defensive-drain contract:

    * Hold a live ``BackwardHandle`` (refcount=1).
    * Call ``reduce_grads_and_offload(cid)`` -> chunk lands in
      ``_deferred_offloads`` (chunk still resident).
    * Call ``drain_deferred_offloads()`` while refcount > 0 -> NO
      offload happens, chunk still resident, set still contains cid.
    * Drop the handle (refcount=0). The handle's __del__ already
      drains via ``_release_backward_handle``, but we also assert
      that re-calling ``drain_deferred_offloads()`` after the drop
      is idempotent (no entries left).
    """
    pytest.importorskip("torch")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA runtime")

    torch.cuda.empty_cache()

    hidden = 64
    n_layers = 2
    model = _tiny_model(hidden=hidden, n_layers=n_layers).to("cuda")
    per_layer_bytes = hidden * hidden * 4
    S_chunk = per_layer_bytes + 4096

    mgr, layout, pool, host = _build_chunk_manager(model, n_persist=1, S_chunk=S_chunk)
    mgr.materialize_offload()

    non_persist = sorted(mgr._non_persistent_ids)
    assert non_persist, "test setup: need at least one non-persistent chunk"
    cid = non_persist[0]

    # --- Hold a live BackwardHandle -----------------------------------
    handle = mgr.gather_for_backward(cid)
    assert mgr._backward_refcount.get(cid, 0) == 1, (
        f"refcount==1 expected, got {mgr._backward_refcount.get(cid, 0)}"
    )
    assert pool.lookup_resident(cid) is not None, (
        "chunk must be resident after gather_for_backward"
    )

    # --- reduce_grads_and_offload defers ------------------------------
    # Synthesize zero grads so reduce has something to look at.
    for pid in layout.chunks[int(cid)]:
        param = mgr._params_by_id.get(pid)
        if param is None or not param.requires_grad:
            continue
        if param.data.numel() == 0:
            continue
        param.grad = torch.zeros_like(param.data)

    mgr.reduce_grads_and_offload(cid)
    assert cid in mgr._deferred_offloads, (
        "reduce_grads_and_offload should have queued a deferred offload "
        f"while refcount>0; _deferred_offloads={mgr._deferred_offloads}"
    )
    assert pool.lookup_resident(cid) is not None, (
        "chunk must remain resident while a BackwardHandle is alive"
    )

    # --- drain while refcount > 0 -> no-op on this chunk --------------
    drained = mgr.drain_deferred_offloads()
    assert drained == 0, (
        f"drain_deferred_offloads should not offload chunks with refcount>0; "
        f"reported {drained} drained"
    )
    assert cid in mgr._deferred_offloads, (
        "chunk should remain in _deferred_offloads while refcount>0; "
        f"_deferred_offloads={mgr._deferred_offloads}"
    )
    assert pool.lookup_resident(cid) is not None, (
        "chunk must still be resident; drain must not evict refcounted slots"
    )
    assert mgr._backward_refcount.get(cid, 0) == 1, (
        "drain must not perturb the refcount"
    )

    # --- Drop the handle: __del__ on release already drains -----------
    handle.release()
    assert cid not in mgr._backward_refcount, (
        "refcount entry should be popped to zero on release"
    )
    # The release() path drains through _release_backward_handle, so
    # _deferred_offloads is already empty at this point.
    assert cid not in mgr._deferred_offloads, (
        f"deferred offload should have drained on release; "
        f"_deferred_offloads={mgr._deferred_offloads}"
    )

    # --- drain again -> idempotent no-op ------------------------------
    drained2 = mgr.drain_deferred_offloads()
    assert drained2 == 0, (
        f"second drain should be a no-op (set already empty); reported {drained2}"
    )

    # Cleanup.
    mgr.uninstall()
    host.close()
    del pool


# ---------------------------------------------------------------------------
# Test 3: full scheduler-driven 3-iter smoke (M3 acceptance)
# ---------------------------------------------------------------------------


@pytest.mark.gpu
def test_offload_mode_3iter_smoke() -> None:
    """Tiny 2-block model with one OFFLOAD block trains via the SCHEDULER.

    The M3 acceptance test per §7: forward+backward through the actual
    install_hooks pipeline (not the M2 manual-offload simulation),
    looped 3 iterations. Reference grads match within fp32 tolerance.
    """
    pytest.importorskip("torch")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA runtime")

    torch.cuda.empty_cache()
    device = torch.device("cuda")

    hidden = 32
    n_layers = 2

    # Build reference + identical-init ProTrain models.
    torch.manual_seed(99)
    ref_model = _tiny_model(hidden=hidden, n_layers=n_layers).to(device)
    torch.manual_seed(99)
    pt_model = _tiny_model(hidden=hidden, n_layers=n_layers).to(device)

    for (rn, rp), (pn, pp) in zip(
        ref_model.named_parameters(),
        pt_model.named_parameters(),
        strict=True,
    ):
        assert rn == pn
        assert torch.equal(rp, pp), f"init mismatch on {rn}"

    per_layer_bytes = hidden * hidden * 4
    S_chunk = per_layer_bytes + 1024

    mgr, layout, pool, host = _build_chunk_manager(
        pt_model, n_persist=1, S_chunk=S_chunk
    )
    mgr.materialize_offload()
    # Drop CPU-grad hooks; we want grads on param.grad for parity check.
    mgr.uninstall()

    # Wrap h.0 in OffloadedBlock; h.1 stays unwrapped (NONE).
    wrapped_h0, _original_h0 = _wrap_block_offload(pt_model, 0, mgr)

    # block_map: block 0 is OFFLOAD, block 1 is NONE.
    block_map: BlockStrategyMap = {
        cast(BlockId, 0): BlockMode.OFFLOAD,
        cast(BlockId, 1): BlockMode.NONE,
    }
    scheduler = _build_scheduler(mgr, layout, block_map)

    # Install hooks against the ProTrain model. This is the new M3
    # path: install_hooks now wires OffloadedBlock to the runtime via
    # attach_runtime, mirroring the SwappedBlock path.
    from axolotl.integrations.protrain.runtime.hooks import (
        install_hooks,
        uninstall_hooks,
    )

    handles = install_hooks(
        model=pt_model,
        chunk_manager=mgr,
        block_map=block_map,
        scheduler=scheduler,
    )
    # Sanity-check that install_hooks invoked attach_runtime on h.0.
    assert wrapped_h0._chunk_manager is mgr, (
        "install_hooks should have wired chunk_manager onto OffloadedBlock"
    )
    assert wrapped_h0._scheduler is scheduler, (
        "install_hooks should have wired scheduler onto OffloadedBlock"
    )

    # Pre-warm: gather every non-persistent chunk before the first
    # forward. The forward-pre hooks will keep them resident on
    # successive iterations via the scheduler's lookahead.
    for cid in sorted(mgr._non_persistent_ids):
        mgr.gather(cid)

    torch.manual_seed(7)
    inputs = [torch.randn(4, hidden, device=device) for _ in range(3)]

    try:
        for iter_idx, x in enumerate(inputs):
            # Reference forward+backward.
            ref_model.zero_grad()
            ref_out = ref_model(x)
            ref_loss = ref_out.sum()
            ref_loss.backward()

            # ProTrain forward+backward through the scheduler hooks.
            pt_model.zero_grad()
            pt_out = pt_model(x)
            pt_loss = pt_out.sum()
            pt_loss.backward()

            # End-of-iter drain (incl. drain_deferred_offloads).
            scheduler.drain()
            # Re-warm any chunks the scheduler offloaded post-block;
            # for this CPU-optim-less test we keep the chunks resident
            # for the next iteration to skip the forward-side gather
            # plumbing (the focus of M3 is the BACKWARD path).
            for cid in sorted(mgr._non_persistent_ids):
                if pool.lookup_resident(cid) is None:
                    mgr.gather(cid)

            # --- Grad parity --------------------------------------
            ref_named = dict(ref_model.named_parameters())
            for name, pp in pt_model.named_parameters():
                canon = _canonical_pname(name)
                rp = ref_named.get(canon)
                assert rp is not None, (
                    f"iter {iter_idx}: no ref param matches {name} (canon={canon})"
                )
                if rp.grad is None and pp.grad is None:
                    continue
                assert pp.grad is not None, (
                    f"iter {iter_idx}: pt grad missing for {name}"
                )
                assert rp.grad is not None, (
                    f"iter {iter_idx}: ref grad missing for {canon}"
                )
                assert torch.allclose(rp.grad, pp.grad, atol=1e-4, rtol=1e-4), (
                    f"iter {iter_idx}: grad mismatch on {name}: "
                    f"max_abs_diff="
                    f"{(rp.grad - pp.grad).abs().max().item():.6e}"
                )

            assert torch.allclose(ref_loss, pt_loss, atol=1e-4, rtol=1e-4), (
                f"iter {iter_idx}: loss mismatch "
                f"ref={ref_loss.item():.6e} pt={pt_loss.item():.6e}"
            )

        # After the loop, _deferred_offloads should be empty (no
        # outstanding handles in steady state).
        assert not mgr._deferred_offloads, (
            f"after 3 iters drain, _deferred_offloads should be empty; "
            f"got {mgr._deferred_offloads}"
        )
    finally:
        uninstall_hooks(handles)
        mgr.uninstall()
        host.close()
        del pool
