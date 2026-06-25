"""Diagnostic logging guards for ProTrain alpha calibration."""

from __future__ import annotations

import logging
from dataclasses import replace

import pytest

from axolotl.integrations.protrain.cost.memory import estimate_peak
from axolotl.integrations.protrain.cost.runtime import (
    _compose_t_iter_with_alpha_calibration,
)
from axolotl.integrations.protrain.types import (
    BlockId,
    BlockMode,
    ChunkLayout,
    CostConfig,
    HardwareProfile,
    ProfilerTrace,
)


def _trace(
    *,
    phase2_n_persist: int = 0,
    phase2_n_checkpoint: int = 0,
    phase2_fwd_s: float = 0.0,
    phase2_bwd_s: float = 0.0,
    phase2_step_s: float = 0.0,
    phase2_analytical_fwd_s: float = 0.0,
    phase2_analytical_bwd_s: float = 0.0,
    phase2_analytical_step_s: float = 0.0,
    phase2_per_comp_pred_iter_s: float = 0.0,
    phase2_iter_s: float = 0.0,
) -> ProfilerTrace:
    return ProfilerTrace(
        op_order=(),
        intra_op_delta={},
        inter_op_delta={},
        activation_sizes={BlockId(0): 0},
        model_state_bytes=0,
        pcie_h2d_bps=13e9,
        pcie_d2h_bps=13e9,
        nccl_gather_s={},
        nccl_reduce_s={},
        arch_hash="alpha-diagnostics-test",
        bs=1,
        seq=16,
        sku="test",
        world=1,
        phase2_n_persist=phase2_n_persist,
        phase2_n_checkpoint=phase2_n_checkpoint,
        phase2_fwd_s=phase2_fwd_s,
        phase2_bwd_s=phase2_bwd_s,
        phase2_step_s=phase2_step_s,
        phase2_analytical_fwd_s=phase2_analytical_fwd_s,
        phase2_analytical_bwd_s=phase2_analytical_bwd_s,
        phase2_analytical_step_s=phase2_analytical_step_s,
        phase2_per_comp_pred_iter_s=phase2_per_comp_pred_iter_s,
        phase2_iter_s=phase2_iter_s,
    )


def test_estimate_peak_debug_log_includes_mode_alpha(caplog: pytest.LogCaptureFixture):
    caplog.set_level(
        logging.DEBUG,
        logger="axolotl.integrations.protrain.cost.memory",
    )
    cfg = CostConfig(n_persist=2, n_buffer=1, n_swap=0, n_checkpoint=2)
    layout = ChunkLayout(
        S_chunk=1 << 20,
        N_chunk=4,
        chunks=((), (), (), ()),
        param_to_chunk={},
        block_to_chunks={BlockId(0): ()},
    )
    hw = HardwareProfile(
        gpu_sku="test",
        gpu_memory_bytes=24 * (1 << 30),
        gpu_count=1,
        pcie_h2d_bps=13e9,
        pcie_d2h_bps=13e9,
        has_nvlink=False,
        dominant_param_bytes_per_element=0.5,
    )

    hw_mode_c = replace(hw, zero3_shard=True)

    # Checkpointed Mode A (no ZeRO-3 sharding) keeps the 0.75 factor; only
    # Mode-C-with-CKPT escalates to 0.95.
    estimate_peak(cfg, _trace(), layout, {BlockId(0): BlockMode.CKPT}, hw)
    assert "estimate_peak:" in caplog.text
    assert "alpha=0.75" in caplog.text

    estimate_peak(cfg, _trace(), layout, {BlockId(0): BlockMode.CKPT}, hw_mode_c)
    assert "alpha=0.95" in caplog.text


def test_phase2_runtime_debug_log_includes_calibration_alphas(
    caplog: pytest.LogCaptureFixture,
):
    caplog.set_level(
        logging.DEBUG,
        logger="axolotl.integrations.protrain.cost.runtime",
    )
    cfg = CostConfig(n_persist=2, n_buffer=1, n_swap=0, n_checkpoint=2)
    trace = _trace(
        phase2_n_persist=2,
        phase2_n_checkpoint=2,
        phase2_fwd_s=1.2,
        phase2_bwd_s=1.5,
        phase2_step_s=1.1,
        phase2_analytical_fwd_s=1.0,
        phase2_analytical_bwd_s=1.0,
        phase2_analytical_step_s=1.0,
        phase2_per_comp_pred_iter_s=4.0,
        phase2_iter_s=5.0,
    )

    observed = _compose_t_iter_with_alpha_calibration(
        cfg=cfg,
        trace=trace,
        t_fwd=0.25,
        t_bwd=0.5,
        t_gpu_optim=0.1,
        t_cpu_optim=0.2,
        fwd_used_phase2_override=False,
        bwd_used_phase2_override=False,
    )

    assert observed > 0.0
    assert "estimate_runtime: phase-2 per-component alpha applied" in caplog.text
    assert "alpha_fwd=1.200" in caplog.text
    assert "alpha_bwd=1.500" in caplog.text
    assert "alpha_opt=1.100" in caplog.text
    assert "alpha_residual=1.250" in caplog.text
