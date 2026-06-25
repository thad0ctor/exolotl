"""Item 3 + Item 4 unit tests.

Item 3: Phase-2 re-pick quickstart predicate (opt-in via
``protrain_phase2_quickstart``). Exercises the pure-function helper
``_phase2_quickstart_should_skip`` so we don't have to spin up the full
GPU Phase-2 measurement path.

Item 4: Searcher's "all configs rejected" RuntimeError appends concrete
fix steps when the root cause is ``cpu_adam_bytes_per_sec=0``.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from pydantic import ValidationError
from torch import nn

from axolotl.integrations.protrain.api.model_wrapper import (
    _abort_phase2_bootstrap_runtime_without_restore,
    _downgrade_oversized_swap_to_checkpoint,
    _phase1_measured_iter_proxy_s,
    _phase2_quickstart_post_measurement_should_skip,
    _phase2_quickstart_should_skip,
    _phase2_quickstart_should_skip_unmeasured_upfront,
    _phase2_should_skip_resident_base_adapter,
    _profiler_trace_controls_for_model,
    _teardown_phase2_bootstrap_runtime,
)
from axolotl.integrations.protrain.args import ProTrainArgs
from axolotl.integrations.protrain.block.layout_rules import assign_modes
from axolotl.integrations.protrain.search import search
from axolotl.integrations.protrain.types import (
    BlockMode,
    CostConfig,
    SearchResult,
    WrappedModel,
)

# Reuse the synthetic builders + fixtures from test_cost_search; pytest
# only auto-injects fixtures defined in conftest.py or the importing
# module, so we redeclare local thin wrappers around the shared builders.
from tests.protrain.test_cost_search import (  # noqa: E402
    _make_hw,
    _make_layout,
    _make_trace,
)


@pytest.fixture
def toy_trace():
    return _make_trace()


@pytest.fixture
def toy_layout():
    return _make_layout()


@pytest.fixture
def toy_hw():
    return _make_hw()


# ---------------------------------------------------------------------------
# Item 3: Phase-2 quickstart predicate
# ---------------------------------------------------------------------------


def test_quickstart_disabled_never_skips_even_when_measurement_is_close():
    """Default (quickstart=False): re-pick always fires, even with tight match."""
    assert (
        _phase2_quickstart_should_skip(
            measured_iter_s=1.00,
            predicted_iter_s=1.00,
            quickstart=False,
            envelope=0.30,
        )
        is False
    )


def test_quickstart_enabled_skips_when_within_envelope():
    """quickstart=True + measurement within envelope -> skip re-pick."""
    # 10% error, envelope 30% -> skip.
    assert (
        _phase2_quickstart_should_skip(
            measured_iter_s=1.10,
            predicted_iter_s=1.00,
            quickstart=True,
            envelope=0.30,
        )
        is True
    )
    # Symmetric: measurement under prediction.
    assert (
        _phase2_quickstart_should_skip(
            measured_iter_s=0.85,
            predicted_iter_s=1.00,
            quickstart=True,
            envelope=0.30,
        )
        is True
    )


def test_quickstart_enabled_does_not_skip_when_outside_envelope():
    """quickstart=True + measurement outside envelope -> re-pick still fires."""
    # 50% over prediction, envelope 30% -> no skip.
    assert (
        _phase2_quickstart_should_skip(
            measured_iter_s=1.50,
            predicted_iter_s=1.00,
            quickstart=True,
            envelope=0.30,
        )
        is False
    )
    # At envelope boundary -> no skip (strict <).
    assert (
        _phase2_quickstart_should_skip(
            measured_iter_s=1.30,
            predicted_iter_s=1.00,
            quickstart=True,
            envelope=0.30,
        )
        is False
    )


def test_quickstart_refuses_to_skip_on_nonpositive_predictions():
    """Guard against div-by-zero / garbage measurements."""
    assert (
        _phase2_quickstart_should_skip(
            measured_iter_s=1.0,
            predicted_iter_s=0.0,
            quickstart=True,
            envelope=0.30,
        )
        is False
    )
    assert (
        _phase2_quickstart_should_skip(
            measured_iter_s=0.0,
            predicted_iter_s=1.0,
            quickstart=True,
            envelope=0.30,
        )
        is False
    )


def test_phase1_proxy_requires_backward_timing(toy_trace, toy_layout, toy_hw):
    """Without a measured backward wall, quickstart must not skip Phase-2."""
    cfg = CostConfig(n_persist=toy_layout.N_chunk, n_buffer=0, n_swap=0, n_checkpoint=0)
    block_map = assign_modes(
        cfg.n_swap,
        cfg.n_checkpoint,
        len(toy_trace.activation_sizes),
        n_offload=cfg.n_offload,
    )

    assert (
        _phase1_measured_iter_proxy_s(
            trace=toy_trace,
            cfg=cfg,
            layout=toy_layout,
            block_map=block_map,
            hw=toy_hw,
        )
        == 0.0
    )


def test_phase1_proxy_can_drive_upfront_quickstart(toy_trace, toy_layout, toy_hw):
    """The upfront quickstart path uses existing Phase-1 trace timings."""
    trace = replace(toy_trace, steady_bwd_wall_s=toy_trace.steady_fwd_wall_s * 2.0)
    cfg = CostConfig(n_persist=toy_layout.N_chunk, n_buffer=0, n_swap=0, n_checkpoint=0)
    block_map = assign_modes(
        cfg.n_swap,
        cfg.n_checkpoint,
        len(trace.activation_sizes),
        n_offload=cfg.n_offload,
    )

    proxy_s = _phase1_measured_iter_proxy_s(
        trace=trace,
        cfg=cfg,
        layout=toy_layout,
        block_map=block_map,
        hw=toy_hw,
    )

    assert proxy_s > trace.steady_fwd_wall_s + trace.steady_bwd_wall_s
    assert _phase2_quickstart_should_skip(
        measured_iter_s=proxy_s,
        predicted_iter_s=proxy_s * 1.10,
        quickstart=True,
        envelope=0.30,
    )


def test_quickstart_skips_unmeasured_on_demand_trace(toy_trace):
    """Large-model on-demand traces lack steady timing; opt-in quickstart keeps Phase-1."""
    trace = replace(
        toy_trace,
        hooked_fwd_wall_s=46.0,
        steady_fwd_wall_s=0.0,
        steady_bwd_wall_s=0.0,
    )

    assert _phase2_quickstart_should_skip_unmeasured_upfront(
        trace=trace,
        predicted_iter_s=6.7,
        quickstart=True,
    )
    assert not _phase2_quickstart_should_skip_unmeasured_upfront(
        trace=trace,
        predicted_iter_s=6.7,
        quickstart=False,
    )
    assert not _phase2_quickstart_should_skip_unmeasured_upfront(
        trace=trace,
        predicted_iter_s=0.0,
        quickstart=True,
    )


def test_quickstart_unmeasured_fallback_requires_absent_steady_timing(toy_trace):
    """If steady timing exists, use the normal measured-envelope predicate."""
    trace = replace(
        toy_trace,
        hooked_fwd_wall_s=46.0,
        steady_fwd_wall_s=1.0,
        steady_bwd_wall_s=2.0,
    )

    assert not _phase2_quickstart_should_skip_unmeasured_upfront(
        trace=trace,
        predicted_iter_s=6.7,
        quickstart=True,
    )


def test_post_measurement_quickstart_uses_original_prediction():
    """Regression: do not compare quickstart against the bootstrap prediction."""
    cfg = CostConfig(n_persist=0, n_buffer=2, n_swap=0, n_checkpoint=8)
    block_map = assign_modes(cfg.n_swap, cfg.n_checkpoint, 8)
    phase1_result = SearchResult(
        cfg=cfg,
        block_map=block_map,
        predicted_peak_bytes=1,
        predicted_iter_s=1.0,
    )

    assert _phase2_quickstart_post_measurement_should_skip(
        measured_iter_s=1.1,
        phase1_result=phase1_result,
        boot_cfg=cfg,
        boot_block_map=block_map,
        quickstart=True,
        envelope=0.30,
    )


def test_post_measurement_quickstart_refuses_bootstrap_mismatch():
    """Keeping the bootstrap runtime is not the same as keeping the Phase-1 pick."""
    phase1_cfg = CostConfig(n_persist=2, n_buffer=2, n_swap=0, n_checkpoint=4)
    phase1_block_map = assign_modes(phase1_cfg.n_swap, phase1_cfg.n_checkpoint, 8)
    phase1_result = SearchResult(
        cfg=phase1_cfg,
        block_map=phase1_block_map,
        predicted_peak_bytes=1,
        predicted_iter_s=1.0,
    )
    boot_cfg = CostConfig(n_persist=0, n_buffer=2, n_swap=0, n_checkpoint=8)
    boot_block_map = assign_modes(boot_cfg.n_swap, boot_cfg.n_checkpoint, 8)

    assert (
        _phase2_quickstart_post_measurement_should_skip(
            measured_iter_s=1.1,
            phase1_result=phase1_result,
            boot_cfg=boot_cfg,
            boot_block_map=boot_block_map,
            quickstart=True,
            envelope=0.30,
        )
        is False
    )


def test_phase2_skips_resident_base_adapter_training():
    """Frozen-base adapter runs skip the extra Phase-2 snapshot/restore pass."""

    class _AdapterModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.frozen_base = nn.Parameter(
                torch.empty(1000, device="meta", dtype=torch.uint8),
                requires_grad=False,
            )
            self.adapter = nn.Parameter(
                torch.empty(1, device="meta", dtype=torch.float16),
                requires_grad=True,
            )

    assert _phase2_should_skip_resident_base_adapter(
        _AdapterModel(),
        adapter="qlora",
        load_in_4bit=True,
        load_in_8bit=False,
    )


def test_resident_base_adapter_uses_forward_only_profiler_trace():
    """Frozen-base adapter traces avoid backward saved-tensor CPU spill."""

    class _AdapterModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.frozen_base = nn.Parameter(
                torch.empty(1000, device="meta", dtype=torch.uint8),
                requires_grad=False,
            )
            self.adapter = nn.Parameter(
                torch.empty(1, device="meta", dtype=torch.float16),
                requires_grad=True,
            )

    assert _profiler_trace_controls_for_model(
        _AdapterModel(),
        adapter="qlora",
        load_in_4bit=True,
        load_in_8bit=False,
    ) == (False, False)


def test_oversized_swap_pool_downgrades_to_checkpoint():
    """Runtime must not allocate a SWAP pool larger than the CPU budget."""

    class _Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(
                torch.empty(1024 * 1024, device="meta", dtype=torch.float16)
            )

    result = SearchResult(
        cfg=CostConfig(
            n_persist=0,
            n_buffer=1,
            n_swap=1,
            n_checkpoint=0,
            n_offload=0,
        ),
        block_map=assign_modes(n_swap=1, n_checkpoint=0, N_block=2),
        predicted_peak_bytes=0,
        predicted_iter_s=1.0,
    )

    adjusted = _downgrade_oversized_swap_to_checkpoint(
        blocks=[_Block(), _Block()],
        result=result,
        trace=_make_trace(
            n_block=2,
            activation_bytes_per_block=1 << 20,
            intra_delta_bytes=1 << 20,
        ),
        layout=_make_layout(n_block=2),
        hw=_make_hw(),
        cpu_capacity_bytes=8 << 20,
    )

    assert adjusted.cfg.n_swap == 0
    assert adjusted.cfg.n_checkpoint == 1
    assert list(adjusted.block_map.values()).count(BlockMode.CKPT) == 1
    assert BlockMode.SWAP not in adjusted.block_map.values()


def test_phase2_does_not_skip_full_finetune_training():
    """Full-FT still gets the standard Phase-2 calibration path."""

    class _FullFtModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(
                torch.empty(1000, device="meta", dtype=torch.float16),
                requires_grad=True,
            )

    assert not _phase2_should_skip_resident_base_adapter(
        _FullFtModel(),
        adapter=None,
        load_in_4bit=False,
        load_in_8bit=False,
    )
    assert _profiler_trace_controls_for_model(
        _FullFtModel(),
        adapter=None,
        load_in_4bit=False,
        load_in_8bit=False,
    ) == (True, True)


def test_phase2_abort_closes_without_restore_to_gpu():
    """Rollback-integrity abort must not take the fallback full-restore path."""
    calls: list[str] = []

    class _Handle:
        def remove(self) -> None:
            calls.append("remove")

    class _Scheduler:
        def close(self) -> None:
            calls.append("scheduler_close")

    class _ChunkManager:
        def restore_to_gpu(self) -> None:
            raise AssertionError("restore_to_gpu must not be called")

        def close(self) -> None:
            calls.append("chunk_manager_close")

    result = SearchResult(
        cfg=CostConfig(n_persist=0, n_buffer=0, n_swap=0, n_checkpoint=0),
        block_map=assign_modes(0, 0, 1),
        predicted_peak_bytes=0,
        predicted_iter_s=0.0,
    )
    wrapped = WrappedModel(
        module=nn.Sequential(),
        search_result=result,
        chunk_manager=_ChunkManager(),
        scheduler=_Scheduler(),
        _hook_handles=[_Handle()],
    )

    _abort_phase2_bootstrap_runtime_without_restore(
        model=wrapped.module,
        blocks=[],
        handles=[_Handle()],
        boot_wrapped=wrapped,
        context="test abort",
    )

    assert calls == ["remove", "scheduler_close", "chunk_manager_close"]


def test_phase2_restore_integrity_error_is_distinct():
    from axolotl.integrations.protrain.profiler.phase2 import (
        Phase2RestoreIntegrityError,
    )

    assert issubclass(Phase2RestoreIntegrityError, RuntimeError)


def test_phase2_teardown_closes_bootstrap_when_restore_raises():
    """The rebuild teardown closes wrapper resources even when restore fails."""
    pytest.importorskip("torch")
    from torch import nn

    calls: list[str] = []

    class _Handle:
        def remove(self) -> None:
            calls.append("remove")

    class _ChunkManager:
        def restore_to_gpu(self) -> None:
            calls.append("restore")
            raise RuntimeError("restore boom")

    class _BootWrapped:
        _hook_handles = [_Handle()]

        def close(self) -> None:
            calls.append("close")

    with pytest.raises(RuntimeError, match="restore boom"):
        _teardown_phase2_bootstrap_runtime(
            model=nn.Sequential(),
            blocks=[],
            handles=[_Handle()],
            chunk_manager=_ChunkManager(),
            boot_wrapped=_BootWrapped(),
            context="test phase-2 teardown",
        )

    assert calls == ["remove", "restore", "close"]


# ---------------------------------------------------------------------------
# Item 3: Pydantic args surface the new flags with the expected defaults
# ---------------------------------------------------------------------------


def test_protrain_args_quickstart_defaults():
    """The two new flags default to False / 0.30 and accept overrides."""
    args = ProTrainArgs.model_validate(
        {
            "plugins": ["axolotl.integrations.protrain.ProTrainPlugin"],
            "protrain_auto_memory": True,
            "base_model": "HuggingFaceTB/SmolLM2-135M",
        }
    )
    assert args.protrain_phase2_quickstart is False
    assert args.protrain_phase2_quickstart_envelope == pytest.approx(0.30)

    args_opt_in = ProTrainArgs.model_validate(
        {
            "plugins": ["axolotl.integrations.protrain.ProTrainPlugin"],
            "protrain_auto_memory": True,
            "base_model": "HuggingFaceTB/SmolLM2-135M",
            "protrain_phase2_quickstart": True,
            "protrain_phase2_quickstart_envelope": 0.15,
        }
    )
    assert args_opt_in.protrain_phase2_quickstart is True
    assert args_opt_in.protrain_phase2_quickstart_envelope == pytest.approx(0.15)


def test_protrain_args_quickstart_rejects_negative_envelope():
    """Negative quickstart envelopes fail validation instead of disabling the guard."""
    with pytest.raises(ValidationError):
        ProTrainArgs.model_validate(
            {
                "plugins": ["axolotl.integrations.protrain.ProTrainPlugin"],
                "protrain_auto_memory": True,
                "base_model": "HuggingFaceTB/SmolLM2-135M",
                "protrain_phase2_quickstart": True,
                "protrain_phase2_quickstart_envelope": -0.01,
            }
        )


# ---------------------------------------------------------------------------
# Item 4: searcher rejection message includes concrete fix steps
# ---------------------------------------------------------------------------


def test_search_rejection_message_includes_fix_steps_when_cpu_adam_zero(
    toy_trace, toy_layout, toy_hw
):
    """When ``cpu_adam_bytes_per_sec=0`` causes every offloaded config to be
    runtime-rejected, the searcher's RuntimeError must include the
    DS_SKIP_CUDA_CHECK / Mode A escape-hatch fix steps."""
    # Force the cpu_adam=0 + all-offload runtime-rejection path:
    #  - cpu_adam_bytes_per_sec=0 makes every n_persist<N_chunk config
    #    return inf from estimate_runtime (round-3 R15 contract).
    #  - capacity_bytes tight enough that all-persistent configs are
    #    capacity-rejected, leaving only offloaded (runtime-rejected) configs.
    hw_no_adam = replace(
        toy_hw,
        cpu_adam_bytes_per_sec=0.0,
        gpu_adam_bytes_per_sec=0.0,
    )

    # toy_layout: N_chunk=12, S_chunk=64MB. All-persistent footprint alone
    # exceeds ~768MB, so capacity=500MB rules out persistent configs while
    # leaving offloaded configs as the "feasible by capacity, infeasible by
    # runtime" set.
    MB = 1 << 20
    capacity_bytes = 500 * MB

    with pytest.raises(RuntimeError) as excinfo:
        search(toy_trace, toy_layout, capacity_bytes, hw_no_adam)

    msg = str(excinfo.value)
    # Diagnostic preserved.
    assert "cpu_adam_bytes_per_sec=0" in msg, msg
    # Concrete remediation steps must appear.
    assert "DS_SKIP_CUDA_CHECK" in msg, msg
    assert "Mode A" in msg, msg
    assert "pip install deepspeed" in msg, msg
    assert "CUDA_HOME" in msg, msg
    assert "protrain_force_all_persistent" in msg, msg


def test_search_rejection_message_unchanged_when_cpu_adam_positive(
    toy_trace, toy_layout, toy_hw
):
    """Non-zero cpu_adam keeps the original concise rejection message."""
    # capacity_bytes=0 still triggers the no-feasible-config branch, which
    # is a *different* failure mode (capacity gate, not runtime gate). The
    # new fix-step message must NOT leak into unrelated failure paths.
    with pytest.raises(RuntimeError) as excinfo:
        search(toy_trace, toy_layout, 0, toy_hw)
    msg = str(excinfo.value)
    assert "DS_SKIP_CUDA_CHECK" not in msg, msg
