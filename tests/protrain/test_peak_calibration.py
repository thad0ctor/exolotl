"""Symmetry guards for ``_calibrate_peak_with_actual_chunk_bytes``.

The post-search calibration must reverse out the SAME model-state
charge that :func:`cost.memory.model_state_present_bytes` added (full
optim state under full FT, fp16-only under LoRA-with-frozen-base) and
re-add a calibrated version using actual per-chunk bytes scaled by
the same per-factor breakdown. Pre-fix the calibration only reversed
out / re-added params at 1.0×, leaving the per-chunk full-state
multiplier hiding inside the residual ``f_bm`` and systematically
under-stating peak under full FT.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from axolotl.integrations.protrain.api.model_wrapper import (
    _apply_calibrated_runtime_peak_gate,
    _calibrate_peak_with_actual_chunk_bytes,
)
from axolotl.integrations.protrain.cost.memory import (
    ALPHA_FRAGMENTATION,
    gate_consistent_alpha,
)
from axolotl.integrations.protrain.types import (
    BlockId,
    ChunkId,
    ChunkLayout,
    CostConfig,
    HardwareProfile,
    ParamId,
    ProfilerTrace,
    SearchResult,
)


def _layout(*, n_chunk: int = 4, s_chunk: int = 1024) -> ChunkLayout:
    chunks = tuple((ParamId(f"p.{i}"),) for i in range(n_chunk))
    return ChunkLayout(
        S_chunk=s_chunk,
        N_chunk=n_chunk,
        chunks=chunks,
        param_to_chunk={ParamId(f"p.{i}"): ChunkId(i) for i in range(n_chunk)},
        block_to_chunks={
            BlockId(0): (ChunkId(0), ChunkId(1)),
            BlockId(1): (ChunkId(2),),
            # Chunk 3 left non-block — typical tail.
        },
    )


def _stub_chunk_manager(
    layout: ChunkLayout,
    persistent_ids: set[int],
    chunk_param_bytes: dict[int, int],
    trainable_ids: set[int] | None = None,
) -> SimpleNamespace:
    """Minimal stub matching what ``_calibrate_peak_with_actual_chunk_bytes`` reads.

    ``_chunk_bytes(layout, cm)`` walks ``cm.model.named_parameters()`` and
    sums ``numel * element_size`` per chunk. ParamIds are dotted (e.g.
    ``"p.0"``) and ``nn.Module.register_parameter`` rejects dotted
    names, so we stub ``named_parameters`` directly with the (name,
    Parameter) tuples we need.

    ``trainable_ids`` selects which chunks carry ``requires_grad=True`` so a
    test can model a frozen base (no trainable chunks) vs a trainable subset.
    Defaults to all chunks trainable (the historical behaviour).
    """
    params: list[tuple[str, nn.Parameter]] = []
    for cid, pids in enumerate(layout.chunks):
        target_bytes = chunk_param_bytes.get(cid, 0)
        is_trainable = trainable_ids is None or cid in trainable_ids
        for pid in pids:
            # fp32 = 4 bytes/element; ceil-div lands at-least the target
            # so the test math stays predictable.
            numel = max(1, (target_bytes + 3) // 4)
            param = nn.Parameter(
                torch.zeros(numel, dtype=torch.float32),
                requires_grad=is_trainable,
            )
            params.append((str(pid), param))

    model = SimpleNamespace(named_parameters=lambda: iter(params))
    return SimpleNamespace(
        model=model,
        _persistent_ids={ChunkId(cid) for cid in persistent_ids},
    )


def _trace(model_state_bytes: int) -> ProfilerTrace:
    return ProfilerTrace(
        op_order=(),
        intra_op_delta={},
        inter_op_delta={},
        activation_sizes={},
        model_state_bytes=model_state_bytes,
        pcie_h2d_bps=1.6e10,
        pcie_d2h_bps=1.6e10,
        nccl_gather_s={},
        nccl_reduce_s={},
        arch_hash="peak-calibration-test",
        bs=1,
        seq=1,
        sku="cpu",
        world=1,
    )


def test_calibrate_peak_scales_persistent_by_frozen_state_factor() -> None:
    """Calibrated persistent contribution scales with the FROZEN-state factor.

    The persistent term recovers resident frozen weights via ``frozen_factor``
    (= frozen_param_bytes / fp16_total). With an all-frozen base (no trainable
    chunks) the explicit trainable grad/optimizer term is 0, so doubling the
    frozen footprint doubles the persistent term. ``persistent_factor`` (which
    smeared trainable state into resident bytes) no longer drives this path.
    """
    layout = _layout(n_chunk=4, s_chunk=1024)  # fp16_total = 4096
    persistent_ids = {0, 1}
    chunk_bytes = {0: 800, 1: 700, 2: 500, 3: 600}  # actual_persistent = 1500
    # All chunks frozen so the explicit trainable term is 0 and we isolate
    # the frozen-factor scaling of the persistent body.
    cm = _stub_chunk_manager(layout, persistent_ids, chunk_bytes, trainable_ids=set())
    cfg = CostConfig(n_persist=2, n_buffer=1, n_swap=0, n_checkpoint=0)

    fp16_total = layout.N_chunk * layout.S_chunk  # 4096
    F_BM = 5000  # synthetic "fragmentation + activations + deltas" residual
    alpha = ALPHA_FRAGMENTATION  # 1.10
    buffer_factor = 2.0
    n_persist_eff = 2

    # actual_persistent = 1500; frozen footprint recomputed from the live
    # model (all chunks frozen). frozen total = sum of all four chunk bytes.
    actual_persistent = 1500
    frozen_total = 800 + 700 + 500 + 600  # 2600
    frozen_factor = max(1.0, frozen_total / fp16_total)  # 2600/4096 -> 1.0

    # original_peak's analytic charge still uses persistent_factor (from
    # trace.model_state_bytes); we double it across the two cases and confirm
    # the calibrated body tracks frozen_factor on the resident bytes.
    trace_a = _trace(model_state_bytes=fp16_total)  # persistent_factor 1.0
    cost_state_a = int(
        n_persist_eff * layout.S_chunk * 1.0
        + cfg.n_buffer * layout.S_chunk * buffer_factor
    )
    original_peak_a = int(alpha * (cost_state_a + F_BM))
    calibrated_a = _calibrate_peak_with_actual_chunk_bytes(
        original_peak=original_peak_a,
        layout=layout,
        chunk_manager=cm,
        cfg=cfg,
        trace=trace_a,
    )

    expected_a = int(
        1.05
        * (
            actual_persistent * frozen_factor
            + cfg.n_buffer * layout.S_chunk * buffer_factor
            + F_BM
        )
    )
    assert abs(calibrated_a - expected_a) <= 10, (
        f"calibrated_a={calibrated_a}, expected~{expected_a}"
    )


def test_calibrate_peak_reverse_out_uses_floored_4bit_alpha() -> None:
    """The f_bm reverse-out divides by the FLOORED alpha, not the raw 4-bit factor.

    The searcher builds ``original_peak`` with ``gate_consistent_alpha`` (sub-1.0
    4-bit factors floored to 1.0). Reversing it out with the raw 0.75/0.95 would
    over-recover ``f_bm`` and inflate the calibrated peak. This guards that the
    divisor matches the forward scaling.
    """
    layout = _layout(n_chunk=4, s_chunk=1024)  # fp16_total = 4096
    persistent_ids = {0, 1}
    chunk_bytes = {0: 800, 1: 700, 2: 500, 3: 600}  # actual_persistent = 1500
    # All-frozen so the explicit trainable term is 0 and we isolate the divisor.
    cm = _stub_chunk_manager(layout, persistent_ids, chunk_bytes, trainable_ids=set())
    cfg = CostConfig(n_persist=2, n_buffer=1, n_swap=0, n_checkpoint=0)

    # 4-bit, non-sharded -> alpha_fragmentation_for_cfg = 0.75 (Mode-A).
    hw_4bit = HardwareProfile(
        gpu_sku="test",
        gpu_memory_bytes=24 * (1 << 30),
        gpu_count=1,
        pcie_h2d_bps=13e9,
        pcie_d2h_bps=13e9,
        has_nvlink=False,
        dominant_param_bytes_per_element=0.5,
    )
    raw_alpha = 0.75
    assert gate_consistent_alpha(raw_alpha) == 1.0  # searcher floored it to 1.0

    fp16_total = layout.N_chunk * layout.S_chunk  # 4096
    buffer_factor = 2.0
    n_persist_eff = 2
    F_BM = 5000

    trace = _trace(model_state_bytes=fp16_total)  # persistent_factor 1.0
    cost_state = int(
        n_persist_eff * layout.S_chunk * 1.0
        + cfg.n_buffer * layout.S_chunk * buffer_factor
    )  # 4096
    # original_peak as the searcher builds it: floored alpha (1.0), not 0.75.
    original_peak = int(gate_consistent_alpha(raw_alpha) * (cost_state + F_BM))

    calibrated = _calibrate_peak_with_actual_chunk_bytes(
        original_peak=original_peak,
        layout=layout,
        chunk_manager=cm,
        cfg=cfg,
        trace=trace,
        hw=hw_4bit,
    )

    # Floored reverse-out recovers f_bm == F_BM; calibration_alpha = 1.0.
    actual_persistent = 1500
    frozen_factor = 1.0  # frozen_total 2600 / 4096 -> floored to 1.0
    expected = int(
        actual_persistent * frozen_factor
        + cfg.n_buffer * layout.S_chunk * buffer_factor
        + F_BM
    )  # 1500 + 2048 + 5000 = 8548
    assert abs(calibrated - expected) <= 10, (
        f"calibrated={calibrated} expected~{expected}"
    )

    # The buggy raw-0.75 divisor would over-recover f_bm and yield a larger peak.
    buggy_f_bm = int(original_peak / raw_alpha) - cost_state
    buggy_expected = int(
        actual_persistent + cfg.n_buffer * layout.S_chunk * buffer_factor + buggy_f_bm
    )
    assert calibrated < buggy_expected


def test_calibrate_peak_adds_optimizer_aware_trainable_state() -> None:
    """The calibrated peak adds an explicit, optimizer-aware trainable term.

    A frozen base (chunks 0/1) plus a small trainable adapter (chunks 2/3,
    requires_grad) must include the trainable grad + optimizer state. The term
    is larger for fp32 Adam (adamw_torch, 2+8 B/param) than bnb 8-bit
    (adamw_8bit, 2+2 B/param). Dropping it under-predicted and OOM'd the gate.
    """
    layout = _layout(n_chunk=4, s_chunk=1024)
    # Persistent = all four chunks (Mode A all-persistent). Trainable adapter
    # lives in chunks 2 and 3 (1024 bytes each fp32 → 256 numel each → 512
    # trainable numel total).
    cm = _stub_chunk_manager(
        layout,
        persistent_ids={0, 1, 2, 3},
        chunk_param_bytes={0: 1024, 1: 1024, 2: 1024, 3: 1024},
        trainable_ids={2, 3},
    )
    cfg = CostConfig(n_persist=4, n_buffer=0, n_swap=0, n_checkpoint=0)
    trace = _trace(model_state_bytes=layout.N_chunk * layout.S_chunk)
    original_peak = int(ALPHA_FRAGMENTATION * (4 * layout.S_chunk))

    calibrated_8bit = _calibrate_peak_with_actual_chunk_bytes(
        original_peak=original_peak,
        layout=layout,
        chunk_manager=cm,
        cfg=cfg,
        trace=trace,
        optimizer_name="adamw_8bit",
    )
    calibrated_fp32 = _calibrate_peak_with_actual_chunk_bytes(
        original_peak=original_peak,
        layout=layout,
        chunk_manager=cm,
        cfg=cfg,
        trace=trace,
        optimizer_name="adamw_torch",
    )

    trainable_numel = 512  # 2 chunks * (1024 // 4) fp32 numel
    # 8-bit Adam: grad 2 + optim 2 = 4 B/param. fp32 Adam: grad 2 + optim 8 = 10.
    state_8bit = trainable_numel * 4
    state_fp32 = trainable_numel * 10
    expected_diff = int(1.05 * (state_fp32 - state_8bit))

    assert calibrated_fp32 > calibrated_8bit, (
        f"fp32 Adam state must exceed 8-bit: fp32={calibrated_fp32}, "
        f"8bit={calibrated_8bit}"
    )
    assert abs((calibrated_fp32 - calibrated_8bit) - expected_diff) <= 5, (
        f"diff={calibrated_fp32 - calibrated_8bit}, expected~{expected_diff}"
    )

    # Absolute check: calibrated = 1.05 * (resident + trainable_state).
    # resident (all 4 chunks persistent) = 4 * 1024 = 4096; f_bm = 0 here.
    expected_8bit = int(1.05 * (4096 + state_8bit))
    assert abs(calibrated_8bit - expected_8bit) <= 5, (
        f"calibrated_8bit={calibrated_8bit}, expected~{expected_8bit}"
    )


def test_calibrate_peak_unknown_optimizer_is_conservative() -> None:
    """An unknown optimizer falls back to the fp32-Adam (largest) state size."""
    layout = _layout(n_chunk=4, s_chunk=1024)
    cm = _stub_chunk_manager(
        layout,
        persistent_ids={0, 1, 2, 3},
        chunk_param_bytes={0: 1024, 1: 1024, 2: 1024, 3: 1024},
        trainable_ids={2, 3},
    )
    cfg = CostConfig(n_persist=4, n_buffer=0, n_swap=0, n_checkpoint=0)
    trace = _trace(model_state_bytes=layout.N_chunk * layout.S_chunk)
    original_peak = int(ALPHA_FRAGMENTATION * (4 * layout.S_chunk))

    calibrated_unknown = _calibrate_peak_with_actual_chunk_bytes(
        original_peak=original_peak,
        layout=layout,
        chunk_manager=cm,
        cfg=cfg,
        trace=trace,
        optimizer_name=None,
    )
    calibrated_fp32 = _calibrate_peak_with_actual_chunk_bytes(
        original_peak=original_peak,
        layout=layout,
        chunk_manager=cm,
        cfg=cfg,
        trace=trace,
        optimizer_name="adamw_torch",
    )
    assert calibrated_unknown == calibrated_fp32


def test_calibrated_runtime_peak_gate_uses_replaced_prediction(
    caplog: pytest.LogCaptureFixture,
) -> None:
    layout = _layout(n_chunk=4, s_chunk=1024)
    cfg = CostConfig(n_persist=2, n_buffer=1, n_swap=0, n_checkpoint=0)
    cm = _stub_chunk_manager(
        layout,
        persistent_ids={0, 1},
        chunk_param_bytes={0: 4096, 1: 4096, 2: 1024, 3: 1024},
    )
    trace = _trace(model_state_bytes=layout.N_chunk * layout.S_chunk)

    raw_peak = int(ALPHA_FRAGMENTATION * (2 * layout.S_chunk + 2 * layout.S_chunk))
    raw_result = SearchResult(
        cfg=cfg,
        block_map={},
        predicted_peak_bytes=raw_peak,
        predicted_iter_s=1.25,
    )
    calibrated_peak = _calibrate_peak_with_actual_chunk_bytes(
        original_peak=raw_result.predicted_peak_bytes,
        layout=layout,
        chunk_manager=cm,
        cfg=cfg,
        trace=trace,
    )

    assert raw_result.predicted_peak_bytes < calibrated_peak
    raw_fitting_capacity = raw_result.predicted_peak_bytes + 1
    assert raw_result.predicted_peak_bytes <= raw_fitting_capacity < calibrated_peak

    with pytest.raises(RuntimeError, match="calibrated peak exceeds"):
        _apply_calibrated_runtime_peak_gate(
            raw_result,
            calibrated_peak_bytes=calibrated_peak,
            init_transient_peak_bytes=0,
            capacity_bytes=raw_fitting_capacity,
        )

    caplog.clear()
    with caplog.at_level(
        logging.INFO,
        logger="axolotl.integrations.protrain.api.model_wrapper",
    ):
        accepted = _apply_calibrated_runtime_peak_gate(
            raw_result,
            calibrated_peak_bytes=calibrated_peak,
            init_transient_peak_bytes=123,
            capacity_bytes=calibrated_peak,
        )

    assert accepted.predicted_peak_bytes == calibrated_peak
    assert accepted.predicted_init_transient_peak_bytes == 123
    assert accepted.predicted_iter_s == raw_result.predicted_iter_s
    assert any(
        "calibrated steady peak prediction" in record.getMessage()
        for record in caplog.records
    )
    assert not any(
        "raw search peak" in record.getMessage() for record in caplog.records
    )
