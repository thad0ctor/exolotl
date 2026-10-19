"""Unit tests for the auto mode-selection logic (M7 follow-up).

Covers ``axolotl.integrations.protrain.api.model_wrapper._select_mode``
in isolation — no GPU, no profiler, no distributed init. Each test
builds a synthetic ``SearchResult`` / ``ChunkLayout`` / ``HardwareProfile``
and exercises one decision-tree branch:

* ``test_auto_picks_mode_a_when_fits`` — n_persist >= N_chunk → (True, False)
* ``test_auto_picks_mode_b_when_ram_sufficient`` — offload + plenty of CPU RAM → (False, False)
* ``test_auto_picks_mode_c_when_ram_tight`` — offload + RAM only fits sharded → (False, True)
* ``test_auto_raises_when_nothing_fits`` — offload + tiny RAM → RuntimeError
* ``test_explicit_flag_overrides_auto`` — auto_mode=False honours user flags

The M7 benchmark table (``DESIGN.md §Multi-GPU``) motivates the
"Mode A > Mode B > Mode C" preference ordering; these tests lock that
ordering in place so a future refactor of the selector can't silently
swap it.
"""

from __future__ import annotations

import pytest

from axolotl.integrations.protrain.api.model_wrapper import _select_mode
from axolotl.integrations.protrain.types import (
    BlockId,
    BlockMode,
    BlockStrategyMap,
    ChunkLayout,
    CostConfig,
    HardwareProfile,
    SearchResult,
)


def _mk_layout(*, s_chunk: int, n_chunk: int) -> ChunkLayout:
    """Build a minimal ChunkLayout — only S_chunk + N_chunk are read."""
    return ChunkLayout(
        S_chunk=s_chunk,
        N_chunk=n_chunk,
        chunks=tuple(() for _ in range(n_chunk)),
        param_to_chunk={},
        block_to_chunks={},
    )


def _mk_hw(*, gpu_count: int, zero3_shard: bool = False) -> HardwareProfile:
    return HardwareProfile(
        gpu_sku="RTX 3090",
        gpu_memory_bytes=24 * (1 << 30),
        gpu_count=gpu_count,
        pcie_h2d_bps=13e9,
        pcie_d2h_bps=13e9,
        has_nvlink=False,
        zero3_shard=zero3_shard,
    )


def _mk_search(*, n_persist: int, n_block: int = 4) -> SearchResult:
    """Build a minimal SearchResult with the n_persist we want to test."""
    cfg = CostConfig(
        n_persist=n_persist,
        n_buffer=2,
        n_swap=0,
        n_checkpoint=0,
    )
    block_map: BlockStrategyMap = {BlockId(i): BlockMode.NONE for i in range(n_block)}
    return SearchResult(
        cfg=cfg,
        block_map=block_map,
        predicted_peak_bytes=0,
        predicted_iter_s=0.0,
    )


def test_auto_picks_mode_a_when_fits() -> None:
    """n_persist >= N_chunk → Mode A (force_all_persistent=True)."""
    layout = _mk_layout(s_chunk=128 * (1 << 20), n_chunk=10)
    hw = _mk_hw(gpu_count=4)
    # Searcher placed every chunk on GPU — the definition of "fits".
    search = _mk_search(n_persist=10)

    # CPU RAM value is irrelevant on this branch (selector never
    # consults it when fitting on GPU). Pass 0 to prove that.
    force_persistent, zero3 = _select_mode(
        search_result=search,
        layout=layout,
        hw=hw,
        world_size=4,
        cpu_ram_per_rank_bytes=0,
        auto_mode=True,
        user_force_all_persistent=False,
        user_zero3_shard=None,
    )

    assert force_persistent is True
    assert zero3 is False


def test_auto_picks_mode_b_when_ram_sufficient() -> None:
    """Offload needed + RAM fits replicated → Mode B (not sharded)."""
    # 10 chunks of 128 MB each, n_persist=2 → 8 non-persistent chunks →
    # 1 GB total non-persistent bytes under replication.
    s_chunk = 128 * (1 << 20)
    n_chunk = 10
    layout = _mk_layout(s_chunk=s_chunk, n_chunk=n_chunk)
    hw = _mk_hw(gpu_count=4)
    search = _mk_search(n_persist=2)

    replicated_footprint = (n_chunk - 2) * s_chunk  # ~1 GB
    # Give each rank 4x the replicated footprint — well above the
    # Mode B threshold. Selector must prefer B (lower-latency) over C.
    cpu_ram_per_rank = 4 * replicated_footprint

    force_persistent, zero3 = _select_mode(
        search_result=search,
        layout=layout,
        hw=hw,
        world_size=4,
        cpu_ram_per_rank_bytes=cpu_ram_per_rank,
        auto_mode=True,
        user_force_all_persistent=False,
        user_zero3_shard=None,
    )

    assert force_persistent is False
    assert zero3 is False


def test_auto_picks_mode_c_when_ram_tight() -> None:
    """Offload needed + RAM fits only sharded → Mode C (zero3_shard=True)."""
    s_chunk = 128 * (1 << 20)
    n_chunk = 10
    layout = _mk_layout(s_chunk=s_chunk, n_chunk=n_chunk)
    hw = _mk_hw(gpu_count=4)
    search = _mk_search(n_persist=2)

    # Sharded footprint = replicated / 4 (ceiling div). Give the selector
    # just enough RAM to fit sharded but NOT replicated — the gap for
    # Mode C. Expected sharded bytes ~ 256 MB/rank; ~ 1 GB replicated.
    sharded_footprint = ((n_chunk - 2) * s_chunk + 3) // 4
    # 1.1x the sharded footprint — above the C threshold but well
    # below the B threshold (which is 4x larger).
    cpu_ram_per_rank = int(1.1 * sharded_footprint)
    # Sanity: make sure we're actually in the C window.
    assert cpu_ram_per_rank < (n_chunk - 2) * s_chunk, (
        "test setup error: RAM should be insufficient for replication"
    )

    force_persistent, zero3 = _select_mode(
        search_result=search,
        layout=layout,
        hw=hw,
        world_size=4,
        cpu_ram_per_rank_bytes=cpu_ram_per_rank,
        auto_mode=True,
        user_force_all_persistent=False,
        user_zero3_shard=None,
    )

    assert force_persistent is False
    assert zero3 is True


def test_auto_raises_when_nothing_fits() -> None:
    """Offload needed + RAM below sharded footprint → RuntimeError."""
    s_chunk = 128 * (1 << 20)
    n_chunk = 10
    layout = _mk_layout(s_chunk=s_chunk, n_chunk=n_chunk)
    hw = _mk_hw(gpu_count=4)
    search = _mk_search(n_persist=2)

    # Give the selector far less than the sharded footprint. The
    # workload truly doesn't fit on this node — raise.
    cpu_ram_per_rank = 1024  # 1 KB — well below any shard

    with pytest.raises(RuntimeError, match="does not fit"):
        _select_mode(
            search_result=search,
            layout=layout,
            hw=hw,
            world_size=4,
            cpu_ram_per_rank_bytes=cpu_ram_per_rank,
            auto_mode=True,
            user_force_all_persistent=False,
            user_zero3_shard=None,
        )


def test_explicit_flag_overrides_auto() -> None:
    """auto_mode=False → user flags are honoured verbatim.

    Key invariant: the selector must NOT second-guess explicit user
    intent when auto_mode is off. Set zero3_shard=True with
    auto_mode=False on a workload that fits in Mode A — the selector
    must still return (False, True).
    """
    # Model fits (n_persist == N_chunk) — under auto the selector
    # would pick Mode A. With auto_mode=False the user's zero3_shard
    # MUST win.
    layout = _mk_layout(s_chunk=128 * (1 << 20), n_chunk=10)
    hw = _mk_hw(gpu_count=4)
    search = _mk_search(n_persist=10)

    force_persistent, zero3 = _select_mode(
        search_result=search,
        layout=layout,
        hw=hw,
        world_size=4,
        cpu_ram_per_rank_bytes=10 * (1 << 30),  # plenty
        auto_mode=False,
        user_force_all_persistent=False,
        user_zero3_shard=True,
    )

    assert force_persistent is False
    assert zero3 is True

    # Also: auto_mode=False with force_all_persistent=True passes
    # through. (Even though n_persist < N_chunk could disagree with
    # the user's intent, auto_mode=False means "I know what I want".)
    search_tight = _mk_search(n_persist=1)
    force_persistent, zero3 = _select_mode(
        search_result=search_tight,
        layout=layout,
        hw=hw,
        world_size=4,
        cpu_ram_per_rank_bytes=10 * (1 << 30),
        auto_mode=False,
        user_force_all_persistent=True,
        user_zero3_shard=False,
    )
    assert force_persistent is True
    assert zero3 is False


def test_auto_single_rank_honours_searcher_n_persist() -> None:
    """world_size=1 → honour the searcher's offload decision.

    Single-rank has no multi-GPU mode to pick (zero3 is meaningless),
    but the selector must still respect ``n_persist < N_chunk`` from
    the searcher — forcing Mode A on a model that only fits with
    non-persistent chunks would OOM. Mode A is selected only when the
    searcher itself picked an all-persistent layout.
    """
    layout = _mk_layout(s_chunk=128 * (1 << 20), n_chunk=10)
    hw = _mk_hw(gpu_count=1)

    # Searcher wants offload (n_persist=1 < N_chunk=10): selector must
    # NOT force Mode A.
    search_offload = _mk_search(n_persist=1)
    force_persistent, zero3 = _select_mode(
        search_result=search_offload,
        layout=layout,
        hw=hw,
        world_size=1,
        cpu_ram_per_rank_bytes=0,
        auto_mode=True,
        user_force_all_persistent=False,
        user_zero3_shard=None,
    )
    assert force_persistent is False, (
        "single-rank with searcher n_persist < N_chunk must NOT force Mode A"
    )
    assert zero3 is False

    # Searcher wants all-persistent (n_persist=N_chunk): selector picks Mode A.
    search_all = _mk_search(n_persist=10)
    force_persistent, zero3 = _select_mode(
        search_result=search_all,
        layout=layout,
        hw=hw,
        world_size=1,
        cpu_ram_per_rank_bytes=0,
        auto_mode=True,
        user_force_all_persistent=False,
        user_zero3_shard=None,
    )
    assert force_persistent is True
    assert zero3 is False


def test_auto_mode_default_in_args() -> None:
    """``ProTrainArgs.protrain_auto_mode`` default must be True.

    This is the user-facing fix for the M7 footgun — flipping the
    default silently re-opens the ZeRO-3 performance trap.
    """
    from axolotl.integrations.protrain.args import ProTrainArgs

    field = ProTrainArgs.model_fields["protrain_auto_mode"]
    assert field.default is True, (
        f"protrain_auto_mode default is {field.default!r}, expected True. "
        "Flipping this re-opens the M7 ZeRO-3 footgun — see DESIGN.md "
        "§Multi-GPU."
    )
