"""Hook-less steady-state calibration tests for the ProTrain profiler.

Covers the TRACE_VERSION=4 additions: the profiler records both a HOOKED
forward wall-clock (with pre/post forward hooks on every nn.Module) AND
a STEADY-STATE forward wall-clock (measured before hooks are installed)
so the cost model can divide out the hook-dispatch overhead that
otherwise inflates ``t_fwd`` 2.5x on transformer-sized models.

Split into:
- GPU test (``test_trace_records_steady_wall_times``): end-to-end check
  that ``run_trace`` on a tiny GPT-2 populates both wall-time fields.
- CPU-only tests (``test_runtime_scale_applied``,
  ``test_scale_clamp_on_absurd_ratio``): synthetic ProfilerTrace builds
  + ``estimate_runtime`` calls, validating the scale plumbs through
  cost/runtime.py without needing a GPU.
"""

from __future__ import annotations

import math

import pytest

from axolotl.integrations.protrain.block.layout_rules import assign_modes
from axolotl.integrations.protrain.cost import estimate_runtime
from axolotl.integrations.protrain.types import (
    BlockId,
    ChunkId,
    ChunkLayout,
    CostConfig,
    HardwareProfile,
    OpId,
    OpRecord,
    ParamId,
    ProfilerTrace,
)

MB = 1 << 20
GB = 1 << 30


def _build_synthetic_trace(
    *,
    hooked_fwd_wall_s: float,
    steady_fwd_wall_s: float,
    n_block: int = 8,
    ops_per_block: int = 5,
    op_latency_s: float = 0.00002,  # 20 µs per op — keeps the total under 2x roofline
    activation_bytes_per_block: int = 32 * MB,
    model_state_bytes: int = 768 * MB,
) -> ProfilerTrace:
    """Minimal ProfilerTrace with configurable hook-scale fields."""
    op_order: list[OpRecord] = []
    op_latencies: dict[OpId, float] = {}
    intra_deltas: dict[OpId, int] = {}
    inter_deltas: dict[OpId, int] = {}
    op_id = 0
    for b in range(n_block):
        for k in range(ops_per_block):
            rec = OpRecord(
                op_id=OpId(op_id),
                module_path=f"block.{b}.op.{k}",
                qualified_name="aten::toy",
                shape_signature=((1,),),
                block_id=BlockId(b),
                is_forward=True,
            )
            op_order.append(rec)
            op_latencies[OpId(op_id)] = op_latency_s
            intra_deltas[OpId(op_id)] = 8 * MB
            inter_deltas[OpId(op_id)] = 2 * MB
            op_id += 1
    activation_sizes = {BlockId(b): activation_bytes_per_block for b in range(n_block)}
    return ProfilerTrace(
        op_order=tuple(op_order),
        intra_op_delta=intra_deltas,
        inter_op_delta=inter_deltas,
        activation_sizes=activation_sizes,
        model_state_bytes=model_state_bytes,
        pcie_h2d_bps=12e9,
        pcie_d2h_bps=12e9,
        nccl_gather_s={},
        nccl_reduce_s={},
        arch_hash="steady-test",
        bs=1,
        seq=128,
        sku="RTX 3090 (synthetic)",
        world=1,
        op_latencies=op_latencies,
        hooked_fwd_wall_s=hooked_fwd_wall_s,
        steady_fwd_wall_s=steady_fwd_wall_s,
        steady_bwd_wall_s=0.0,
    )


def _build_layout(
    n_chunk: int = 12, s_chunk: int = 64 * MB, n_block: int = 8
) -> ChunkLayout:
    chunks = tuple((ParamId(f"p.{i}"),) for i in range(n_chunk))
    return ChunkLayout(
        S_chunk=s_chunk,
        N_chunk=n_chunk,
        chunks=chunks,
        param_to_chunk={ParamId(f"p.{i}"): ChunkId(i) for i in range(n_chunk)},
        block_to_chunks={BlockId(b): (ChunkId(b % n_chunk),) for b in range(n_block)},
    )


def _build_hw() -> HardwareProfile:
    return HardwareProfile(
        gpu_sku="RTX 3090 (synthetic)",
        gpu_memory_bytes=24 * GB,
        gpu_count=1,
        pcie_h2d_bps=12e9,
        pcie_d2h_bps=12e9,
        has_nvlink=False,
        cpu_adam_bytes_per_sec=2e9,
        gpu_adam_bytes_per_sec=5e11,
    )


# ---------------------------------------------------------------------------
# GPU test — real ``run_trace`` against a tiny GPT-2
# ---------------------------------------------------------------------------


_TINY_MODEL_CANDIDATES = (
    "sshleifer/tiny-gpt2",
    "hf-internal-testing/tiny-random-gpt2",
)


def _load_tiny_gpt2():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    last_exc: Exception | None = None
    for name in _TINY_MODEL_CANDIDATES:
        try:
            tok = AutoTokenizer.from_pretrained(name)
            model = AutoModelForCausalLM.from_pretrained(name)
            return name, tok, model
        except Exception as exc:  # pragma: no cover - network-dependent
            last_exc = exc
            continue
    raise RuntimeError(f"no tiny-GPT2 checkpoint available: {last_exc}")


@pytest.mark.gpu
def test_trace_records_steady_wall_times(gpu_device):
    """``run_trace`` populates ``hooked_fwd_wall_s`` and ``steady_fwd_wall_s``.

    On any real transformer the hooked pass pays pre/post hook dispatch
    that the steady pass skips, so ``hooked >= steady`` must hold. Tiny
    GPT-2 has only a few dozen submodules so the inflation factor is
    small but the ordering invariant still holds.
    """
    pytest.importorskip("torch")
    pytest.importorskip("transformers")

    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    from axolotl.integrations.protrain.profiler import run_trace
    from axolotl.integrations.protrain.types import ProfilerConfig

    device = torch.device(f"cuda:{gpu_device}")
    _name, tok, model = _load_tiny_gpt2()
    model = model.to(device)

    bs, seq = 2, 64
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or "<|endoftext|>"
    enc = tok(
        ["hello world"] * bs,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=seq,
    )
    input_ids = enc["input_ids"].to(device)
    labels = input_ids.clone()
    batch = {"input_ids": input_ids, "labels": labels}

    cfg = ProfilerConfig(
        batch_size=bs,
        seq_len=seq,
        device=str(device),
        include_backward=True,
        on_demand=False,
    )

    trace = run_trace(model, batch, cfg)

    assert trace.hooked_fwd_wall_s > 0.0, (
        f"hooked_fwd_wall_s must be populated on GPU; got {trace.hooked_fwd_wall_s}"
    )
    assert trace.steady_fwd_wall_s > 0.0, (
        f"steady_fwd_wall_s must be populated on GPU; got {trace.steady_fwd_wall_s}"
    )
    # The hooked forward dispatches pre/post hooks on every submodule,
    # which strictly adds CPU work. Allow a small tolerance (1%) so that
    # on very small models where hook dispatch is negligible relative to
    # allocator jitter the test doesn't flake.
    assert trace.hooked_fwd_wall_s >= trace.steady_fwd_wall_s * 0.99, (
        f"hooked ({trace.hooked_fwd_wall_s:.6f}s) should be >= steady "
        f"({trace.steady_fwd_wall_s:.6f}s); ratio "
        f"{trace.steady_fwd_wall_s / trace.hooked_fwd_wall_s:.3f}"
    )
    # Backward was requested — steady_bwd_wall_s should be populated too.
    assert trace.steady_bwd_wall_s > 0.0, (
        f"steady_bwd_wall_s should be > 0 when include_backward=True; "
        f"got {trace.steady_bwd_wall_s}"
    )


@pytest.mark.gpu
def test_trace_records_per_block_peaks(gpu_device):
    """``run_trace`` populates ``steady_fwd_block_peak_bytes`` per block.

    The lightweight block-level hooks installed during the hook-less
    steady forward capture ``torch.cuda.max_memory_allocated`` after each
    block. Tiny GPT-2 has n_block>=2 transformer blocks; every block
    should have a recorded peak > 0.
    """
    pytest.importorskip("torch")
    pytest.importorskip("transformers")

    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    from axolotl.integrations.protrain.block.layout_rules import (
        discover_blocks,
        flatten_block_trees,
    )
    from axolotl.integrations.protrain.profiler import run_trace
    from axolotl.integrations.protrain.types import ProfilerConfig

    device = torch.device(f"cuda:{gpu_device}")
    _name, tok, model = _load_tiny_gpt2()
    model = model.to(device)

    n_block_expected = len(flatten_block_trees(discover_blocks(model)))
    assert n_block_expected >= 2, "tiny GPT-2 should have >=2 transformer blocks"

    bs, seq = 2, 64
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or "<|endoftext|>"
    enc = tok(
        ["hello world"] * bs,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=seq,
    )
    input_ids = enc["input_ids"].to(device)
    labels = input_ids.clone()
    batch = {"input_ids": input_ids, "labels": labels}

    cfg = ProfilerConfig(
        batch_size=bs,
        seq_len=seq,
        device=str(device),
        include_backward=True,
        on_demand=False,
    )
    trace = run_trace(model, batch, cfg)

    assert len(trace.steady_fwd_block_peak_bytes) == n_block_expected, (
        f"expected {n_block_expected} per-block peaks, got "
        f"{len(trace.steady_fwd_block_peak_bytes)}"
    )
    for bid, pk in trace.steady_fwd_block_peak_bytes.items():
        assert pk > 0, f"block {bid} peak bytes should be > 0, got {pk}"
    # Per-block max must not exceed the aggregate ``steady_fwd_peak_bytes``.
    max_block = max(trace.steady_fwd_block_peak_bytes.values())
    assert max_block <= trace.steady_fwd_peak_bytes, (
        f"per-block max ({max_block}) > aggregate peak "
        f"({trace.steady_fwd_peak_bytes}) — should be impossible"
    )


# ---------------------------------------------------------------------------
# CPU-only tests — synthetic traces, scale factor plumbs through cost model
# ---------------------------------------------------------------------------


def test_runtime_scale_applied():
    """Two traces with ratios 2.0x and 1.0x should give ~2x different t_fwd.

    Trace A: steady=1.0, hooked=1.0 -> scale = 1.0 (no correction).
    Trace B: steady=0.5, hooked=1.0 -> scale = 0.5 (halve the per-block sum).
    The forward-compute contribution in Trace B is half of Trace A, so the
    total iteration time should drop correspondingly (modulo communication
    and optimizer terms, which are identical between the two).
    """
    layout = _build_layout()
    hw = _build_hw()
    # All chunks persistent + no swap/ckpt keeps t_cpu_optim off the critical
    # path so the difference between A and B is dominated by t_fwd scaling.
    n_block = 8
    cfg = CostConfig(n_persist=layout.N_chunk, n_buffer=0, n_swap=0, n_checkpoint=0)
    block_map = assign_modes(0, 0, n_block)

    trace_a = _build_synthetic_trace(hooked_fwd_wall_s=1.0, steady_fwd_wall_s=1.0)
    trace_b = _build_synthetic_trace(hooked_fwd_wall_s=1.0, steady_fwd_wall_s=0.5)

    t_a = estimate_runtime(cfg, trace_a, layout, block_map, hw)
    t_b = estimate_runtime(cfg, trace_b, layout, block_map, hw)

    # Trace B scales its forward compute to 0.5x (and its derived
    # t_bwd = t_fwd * 2.0 also scales). Non-compute terms (comm, optim)
    # are identical. So t_b should be strictly less than t_a.
    assert t_b < t_a, (
        f"scale=0.5 trace should give smaller t_iter than scale=1.0; "
        f"t_a={t_a:.6f} t_b={t_b:.6f}"
    )
    # And the reduction should be roughly proportional to the scale
    # reduction — specifically, (t_a - t_b) should be on the order of
    # 0.5 * (t_fwd + t_bwd) = 0.5 * (t_fwd + 2 t_fwd) = 1.5 * t_fwd.
    # We have t_fwd ~= op_latencies * scale * N_block / N_chunk * ...
    # rather than reason precisely, assert a >1.4x ratio as a sanity
    # floor (t_a includes t_fwd + 2 t_fwd ~= 3 t_fwd of scale=1 budget
    # vs 1.5 t_fwd for scale=0.5).
    assert t_a / t_b >= 1.4, (
        f"t_a should be at least 1.4x t_b when hook-scale halves; ratio={t_a / t_b:.3f}"
    )


def test_scale_clamp_on_absurd_ratio():
    """hooked_fwd_wall_s < steady_fwd_wall_s is absurd — clamp to [0.3, 1.0].

    Synthetic trace where hooked=0.5 but steady=1.0 (raw ratio = 2.0 >
    _HOOK_SCALE_MAX). The cost model must refuse to amplify the per-block
    sum — it must fall through to the clamped-scale path (ratio clamped
    to 1.0) rather than using the steady measurement as the absolute
    total (which would propagate a bogus upward scaling).

    Validation: ``t_absurd`` must be finite, positive, and NOT larger
    than a trace where the hooked wall time matches the steady value
    (which is the no-correction baseline).
    """
    layout = _build_layout()
    hw = _build_hw()
    n_block = 8
    cfg = CostConfig(n_persist=layout.N_chunk, n_buffer=0, n_swap=0, n_checkpoint=0)
    block_map = assign_modes(0, 0, n_block)

    absurd_trace = _build_synthetic_trace(
        hooked_fwd_wall_s=0.5,
        steady_fwd_wall_s=1.0,
    )
    # Baseline: hooked and steady both 0.5 (the absurd trace's hooked
    # value), so the PRIMARY path fires and uses steady=0.5 as total.
    # The absurd trace, having steady > hooked, must fall through to
    # the SECONDARY path (clamp to 1.0) and NOT use its steady=1.0
    # value — its t_iter should be <= the baseline t_iter, never more.
    baseline_trace = _build_synthetic_trace(
        hooked_fwd_wall_s=0.5,
        steady_fwd_wall_s=0.5,
    )

    t_absurd = estimate_runtime(cfg, absurd_trace, layout, block_map, hw)
    t_baseline = estimate_runtime(cfg, baseline_trace, layout, block_map, hw)

    assert math.isfinite(t_absurd) and t_absurd > 0.0, (
        f"absurd-ratio trace must yield finite positive t_iter; got {t_absurd}"
    )
    # The clamp must prevent the absurd ratio from inflating t_fwd past
    # the baseline — if it used steady=1.0 as total, t_absurd would be
    # much larger than t_baseline (which uses steady=0.5).
    assert t_absurd <= t_baseline + 1e-6, (
        f"clamped absurd-ratio trace must not exceed baseline; "
        f"t_absurd={t_absurd:.6f} t_baseline={t_baseline:.6f}"
    )
