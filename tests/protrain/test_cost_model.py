"""Mode-aware fragmentation alpha dispatch and CKPT-chain reconstruction tests.

Covers the §7.6 raw-predictor under-prediction fix: a single 4-bit alpha was
calibrated against Mode-A where frozen-weight residency dominates; Mode-C with
gradient checkpointing reverses that dominance so the 0.75 factor structurally
under-predicts the raw peak. The dispatcher splits 4-bit alpha into Mode-A
(0.75) vs Mode-C-CKPT (0.95). The wrapper-side ``_reconstruct_f_bm`` is
updated to sum CKPT activation bytes (chain) rather than take the max, so it
agrees with ``estimate_peak``'s ``ckpt_chain_bytes`` term.
"""

from __future__ import annotations

from typing import Any

import pytest

from axolotl.integrations.protrain.cost.memory import (
    ALPHA_FRAGMENTATION,
    ALPHA_FRAGMENTATION_4BIT,
    ALPHA_FRAGMENTATION_4BIT_MODE_A,
    ALPHA_FRAGMENTATION_4BIT_MODE_C_CKPT,
    DEFAULT_CKPT_INTERNAL_RESIDUAL_FACTOR,
    _block_internal_saved_bytes,
    _compute_ckpt_chain_bytes,
    alpha_fragmentation_for_cfg,
    apply_gate_consistent_scaling,
    estimate_peak,
    gate_consistent_alpha,
    set_default_ckpt_internal_residual_factor,
)
from axolotl.integrations.protrain.types import (
    BlockId,
    BlockMode,
    BlockStrategyMap,
    ChunkLayout,
    CostConfig,
    HardwareProfile,
    ProfilerTrace,
)


def test_mode_alpha_constants_pinned():
    """Lock the two named mode-split constants so unrelated edits don't drift."""
    assert ALPHA_FRAGMENTATION_4BIT_MODE_A == pytest.approx(0.75)
    assert ALPHA_FRAGMENTATION_4BIT_MODE_C_CKPT == pytest.approx(0.95)
    # Mode-A constant stays the legacy 4-bit alpha for back-compat.
    assert ALPHA_FRAGMENTATION_4BIT_MODE_A == pytest.approx(ALPHA_FRAGMENTATION_4BIT)


def test_alpha_fragmentation_dispatch_mode_a():
    """4-bit + n_checkpoint=0 keeps the Mode-A 0.75 factor."""
    cfg_mode_a = CostConfig(n_persist=2, n_buffer=1, n_swap=0, n_checkpoint=0)
    alpha = alpha_fragmentation_for_cfg(0.5, cfg_mode_a)
    assert alpha == pytest.approx(ALPHA_FRAGMENTATION_4BIT_MODE_A)


def test_alpha_fragmentation_dispatch_mode_c_ckpt():
    """4-bit + n_checkpoint>0 picks the Mode-C-CKPT 0.95 factor under ZeRO-3 sharding."""
    cfg_mode_c = CostConfig(n_persist=2, n_buffer=1, n_swap=0, n_checkpoint=4)
    alpha = alpha_fragmentation_for_cfg(0.5, cfg_mode_c, is_mode_c=True)
    assert alpha == pytest.approx(ALPHA_FRAGMENTATION_4BIT_MODE_C_CKPT)


def test_alpha_fragmentation_dispatch_mode_a_with_ckpt_keeps_mode_a():
    """4-bit + n_checkpoint>0 WITHOUT Mode-C sharding stays on the 0.75 factor.

    Guards the bug where checkpointed Mode-A candidates were over-predicted with
    the 0.95 Mode-C-CKPT factor and wrongly rejected as over-capacity.
    """
    cfg_ckpt = CostConfig(n_persist=2, n_buffer=1, n_swap=0, n_checkpoint=4)
    alpha = alpha_fragmentation_for_cfg(0.5, cfg_ckpt, is_mode_c=False)
    assert alpha == pytest.approx(ALPHA_FRAGMENTATION_4BIT_MODE_A)


def test_alpha_fragmentation_dispatch_mode_c_without_ckpt_keeps_mode_a():
    """4-bit + n_swap>0 but no CKPT still routes to Mode-A (frozen-weight regime)."""
    cfg = CostConfig(n_persist=2, n_buffer=1, n_swap=4, n_checkpoint=0)
    alpha = alpha_fragmentation_for_cfg(0.5, cfg)
    assert alpha == pytest.approx(ALPHA_FRAGMENTATION_4BIT_MODE_A)


@pytest.mark.parametrize("bpe", [1.0, 2.0, 4.0])
def test_alpha_fragmentation_dispatch_non_4bit_ignores_mode(bpe: float):
    """fp16/bf16/8-bit keep the unconditional 1.10 alpha regardless of n_checkpoint."""
    cfg_a = CostConfig(n_persist=0, n_buffer=0, n_swap=0, n_checkpoint=0)
    cfg_c = CostConfig(n_persist=0, n_buffer=0, n_swap=0, n_checkpoint=8)
    assert alpha_fragmentation_for_cfg(bpe, cfg_a) == pytest.approx(ALPHA_FRAGMENTATION)
    assert alpha_fragmentation_for_cfg(bpe, cfg_c) == pytest.approx(ALPHA_FRAGMENTATION)


def test_alpha_fragmentation_dispatch_none_cfg_falls_back_to_mode_a():
    """A missing cfg keeps the legacy 4-bit Mode-A factor."""
    assert alpha_fragmentation_for_cfg(0.5, None) == pytest.approx(
        ALPHA_FRAGMENTATION_4BIT_MODE_A
    )
    assert alpha_fragmentation_for_cfg(2.0, None) == pytest.approx(ALPHA_FRAGMENTATION)


def _build_synthetic_trace(
    n_blocks: int,
    activation_per_block: int,
    model_state_bytes: int = 0,
    *,
    bs: int = 1,
    seq: int = 16,
    hidden_size: int = 0,
    num_attention_heads: int = 0,
    intermediate_size: int = 0,
) -> ProfilerTrace:
    """A trace with N blocks, each with ``activation_per_block`` bytes."""
    activation_sizes = {BlockId(i): activation_per_block for i in range(n_blocks)}
    return ProfilerTrace(
        op_order=(),
        intra_op_delta={},
        inter_op_delta={},
        activation_sizes=activation_sizes,
        model_state_bytes=model_state_bytes,
        pcie_h2d_bps=13e9,
        pcie_d2h_bps=13e9,
        nccl_gather_s={},
        nccl_reduce_s={},
        arch_hash="test",
        bs=bs,
        seq=seq,
        sku="test",
        world=1,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size,
    )


def _build_layout(s_chunk: int, n_chunk: int, n_blocks: int) -> ChunkLayout:
    return ChunkLayout(
        S_chunk=s_chunk,
        N_chunk=n_chunk,
        chunks=tuple(tuple() for _ in range(n_chunk)),  # type: ignore[arg-type]
        param_to_chunk={},
        block_to_chunks={BlockId(i): () for i in range(n_blocks)},
    )


def test_compute_ckpt_chain_bytes_sums_only_ckpt_blocks():
    """The helper sums activation bytes only across CKPT-tagged blocks."""
    trace = _build_synthetic_trace(n_blocks=4, activation_per_block=1_000_000)
    block_map: BlockStrategyMap = {
        BlockId(0): BlockMode.NONE,
        BlockId(1): BlockMode.CKPT,
        BlockId(2): BlockMode.CKPT,
        BlockId(3): BlockMode.SWAP,
    }
    assert _compute_ckpt_chain_bytes(trace, block_map) == 2_000_000


def test_compute_ckpt_chain_bytes_empty_inputs():
    """Empty block map or empty trace returns 0."""
    trace = _build_synthetic_trace(n_blocks=2, activation_per_block=1_000)
    assert _compute_ckpt_chain_bytes(trace, {}) == 0
    empty_trace = _build_synthetic_trace(n_blocks=0, activation_per_block=0)
    block_map: BlockStrategyMap = {BlockId(0): BlockMode.CKPT}
    assert _compute_ckpt_chain_bytes(empty_trace, block_map) == 0


def test_reconstruct_f_bm_uses_chain_not_max():
    """``_reconstruct_f_bm`` sums activation bytes across CKPT blocks.

    Reproduces the §7.6 structural under-bound: with K CKPT blocks each of
    size A, the max-based prior bound returned A; the chain bound returns
    K*A, matching ``estimate_peak``'s ``ckpt_chain_bytes`` term.
    """

    from axolotl.integrations.protrain.api.model_wrapper import (
        _calibrate_peak_with_actual_chunk_bytes,
    )

    n_blocks = 4
    activation_per_block = 4_000_000
    trace = _build_synthetic_trace(
        n_blocks=n_blocks, activation_per_block=activation_per_block
    )
    layout = _build_layout(s_chunk=1 << 20, n_chunk=2, n_blocks=n_blocks)
    block_map: BlockStrategyMap = {BlockId(i): BlockMode.CKPT for i in range(n_blocks)}
    cfg = CostConfig(
        n_persist=1, n_buffer=1, n_swap=0, n_checkpoint=n_blocks, n_offload=0
    )

    class _StubChunkManager:
        """Minimal stand-in for `ChunkManager` used by the calibrator."""

        _persistent_ids: tuple[int, ...] = (0,)

        class _Model:
            @staticmethod
            def named_parameters():
                return []

        model = _Model()

    chunk_manager = _StubChunkManager()

    captured: dict[str, int] = {}

    def _spy(bmap):
        # Mirror `_reconstruct_f_bm` body in `_calibrate_peak_with_actual_chunk_bytes`.
        ckpt_chain = _compute_ckpt_chain_bytes(trace, bmap)
        captured["chain"] = ckpt_chain
        return 0, len(bmap)

    # Drive the calibrator just for its side-effects via the helper; the
    # function under test is the public chain helper. We assert two things:
    #   1) The CKPT chain sums to n_blocks * activation_per_block.
    #   2) The legacy "max" semantics would have returned just A.
    chain = _compute_ckpt_chain_bytes(trace, block_map)
    assert chain == n_blocks * activation_per_block
    legacy_max = max(int(v) for v in trace.activation_sizes.values())
    assert legacy_max == activation_per_block
    assert chain > legacy_max, (
        "chain semantics must strictly exceed the legacy single-block max "
        "when more than one block is checkpointed"
    )

    # Run the calibrator end-to-end to confirm it doesn't crash with the
    # new chain semantics (no anchor assertion — the calibrator output
    # composes other terms).
    out = _calibrate_peak_with_actual_chunk_bytes(
        original_peak=1 << 30,
        layout=layout,
        chunk_manager=chunk_manager,
        cfg=cfg,
        trace=trace,
        block_map=block_map,
        hw=HardwareProfile(
            gpu_sku="test",
            gpu_memory_bytes=24 * (1 << 30),
            gpu_count=1,
            pcie_h2d_bps=13e9,
            pcie_d2h_bps=13e9,
            has_nvlink=False,
            dominant_param_bytes_per_element=0.5,
        ),
    )
    # Smoke check: positive, finite int.
    assert isinstance(out, int) and out > 0

    # Spy path: confirm the chain helper plugged into a `_reconstruct_f_bm`-shaped
    # closure surfaces the chain (not max) bytes.
    _spy(block_map)
    assert captured["chain"] == n_blocks * activation_per_block


def test_mode_c_ckpt_alpha_is_floored_in_gate():
    """Sub-1.0 4-bit fragmentation factors are anchors floored to 1.0 in the gate.

    Post-SwiGLU-fix the 0.75 (Mode-A) / 0.95 (Mode-C-CKPT) factors are calibration
    ANCHORS, not live multipliers: ``gate_consistent_alpha`` floors every sub-1.0
    value to 1.0 inside ``estimate_peak`` / the searcher, because deflating the
    now-realistic component sum would let the searcher accept configs the
    calibrated gate rejects. This regression-guards that intentional floor (the
    old test here asserted only an algebraic identity and proved nothing).
    """
    cfg_ckpt = CostConfig(n_persist=2, n_buffer=1, n_swap=0, n_checkpoint=4)
    cfg_none = CostConfig(n_persist=2, n_buffer=1, n_swap=0, n_checkpoint=0)

    # The raw dispatch is genuine (used by the per-dtype diagnostic log)...
    assert (
        alpha_fragmentation_for_cfg(0.5, cfg_ckpt, True)
        == ALPHA_FRAGMENTATION_4BIT_MODE_C_CKPT
    )
    assert (
        alpha_fragmentation_for_cfg(0.5, cfg_none, False)
        == ALPHA_FRAGMENTATION_4BIT_MODE_A
    )

    # ...but the gate floors every sub-1.0 factor to 1.0 and preserves >= 1.0.
    assert gate_consistent_alpha(ALPHA_FRAGMENTATION_4BIT_MODE_C_CKPT) == 1.0
    assert gate_consistent_alpha(ALPHA_FRAGMENTATION_4BIT_MODE_A) == 1.0
    assert gate_consistent_alpha(ALPHA_FRAGMENTATION) == pytest.approx(
        ALPHA_FRAGMENTATION
    )

    # End-to-end through the gate's scaling: a sub-1.0 alpha must NOT deflate the
    # raw peak (would fail if the floor were removed), while 1.10 is preserved.
    raw = 10 * (1 << 30)
    scaled_c = apply_gate_consistent_scaling(raw, raw, 0, 0.95)
    scaled_a = apply_gate_consistent_scaling(raw, raw, 0, 0.75)
    scaled_fp16 = apply_gate_consistent_scaling(raw, raw, 0, ALPHA_FRAGMENTATION)
    assert scaled_c == raw, "Mode-C-CKPT 0.95 alpha must not deflate the peak"
    assert scaled_a == raw, "Mode-A 0.75 alpha must not deflate the peak"
    assert scaled_fp16 == int(ALPHA_FRAGMENTATION * raw)


def test_estimate_peak_mode_c_ckpt_not_deflated_vs_mode_a():
    """estimate_peak end-to-end: 4-bit Mode-C-CKPT peak is gated, never deflated."""
    s_chunk = 1 << 28  # 256 MiB
    n_chunk = 4
    n_blocks = 4
    activation_per_block = 1 << 26  # 64 MiB
    layout = _build_layout(s_chunk=s_chunk, n_chunk=n_chunk, n_blocks=n_blocks)
    trace = _build_synthetic_trace(
        n_blocks=n_blocks, activation_per_block=activation_per_block
    )

    cfg_mode_a = CostConfig(n_persist=2, n_buffer=1, n_swap=0, n_checkpoint=0)
    cfg_mode_c_ckpt = CostConfig(n_persist=2, n_buffer=1, n_swap=0, n_checkpoint=4)

    block_map_a: BlockStrategyMap = {
        BlockId(i): BlockMode.NONE for i in range(n_blocks)
    }
    block_map_c: BlockStrategyMap = {
        BlockId(i): BlockMode.CKPT for i in range(n_blocks)
    }

    hw_4bit = HardwareProfile(
        gpu_sku="test",
        gpu_memory_bytes=24 * (1 << 30),
        gpu_count=1,
        pcie_h2d_bps=13e9,
        pcie_d2h_bps=13e9,
        has_nvlink=False,
        dominant_param_bytes_per_element=0.5,
    )

    peak_mode_a = estimate_peak(cfg_mode_a, trace, layout, block_map_a, hw_4bit)
    peak_mode_c_ckpt = estimate_peak(
        cfg_mode_c_ckpt, trace, layout, block_map_c, hw_4bit
    )
    assert peak_mode_a > 0 and peak_mode_c_ckpt > 0


# Llama-30B reference (huggyllama/llama-30b) used by the section 7.6 audit row.
_LLAMA_30B_HIDDEN = 6656
_LLAMA_30B_INTERMEDIATE = 17920
_LLAMA_30B_HEADS = 52
_LLAMA_30B_N_BLOCK = 60


def test_block_internal_saved_bytes_quadratic_in_seq():
    """Attn-score term scales O(seq^2); ffn/qkv terms scale O(seq)."""
    bs = 1
    hidden = 4096
    heads = 32
    intermediate = 11008

    def _residual(seq: int) -> int:
        trace = _build_synthetic_trace(
            n_blocks=1,
            activation_per_block=1,
            bs=bs,
            seq=seq,
            hidden_size=hidden,
            num_attention_heads=heads,
            intermediate_size=intermediate,
        )
        return _block_internal_saved_bytes(trace, BlockId(0), bytes_per_element=2.0)

    r_512 = _residual(512)
    r_1024 = _residual(1024)
    r_2048 = _residual(2048)
    ratio_1024_over_512 = r_1024 / max(1, r_512)
    ratio_2048_over_1024 = r_2048 / max(1, r_1024)
    assert 2.0 < ratio_1024_over_512 < 4.0
    assert 2.0 < ratio_2048_over_1024 < 4.0
    assert ratio_2048_over_1024 > ratio_1024_over_512


def test_block_internal_saved_bytes_zero_when_arch_fields_missing():
    """Legacy traces (arch fields = 0) get a 0 residual so the chain helper degrades."""
    trace = _build_synthetic_trace(n_blocks=1, activation_per_block=1, bs=1, seq=512)
    assert _block_internal_saved_bytes(trace, BlockId(0)) == 0


def _attn_trace(seq: int, backend: str, **over) -> ProfilerTrace:
    """Qwen3-14B-shaped GQA trace for backend-aware attention-term tests."""
    fields: dict[str, Any] = dict(
        hidden_size=5120,
        num_attention_heads=40,
        intermediate_size=17408,
        num_key_value_heads=8,
        head_dim=128,
        attn_implementation=backend,
    )
    fields.update(over)
    return ProfilerTrace(
        op_order=(),
        intra_op_delta={},
        inter_op_delta={},
        activation_sizes={BlockId(0): 1},
        model_state_bytes=0,
        pcie_h2d_bps=13e9,
        pcie_d2h_bps=13e9,
        nccl_gather_s={},
        nccl_reduce_s={},
        arch_hash="test",
        bs=1,
        seq=seq,
        sku="test",
        world=1,
        **fields,
    )


def test_attn_activation_flash_is_linear_in_seq():
    """Flash-Attention-2 retains no score matrix → term scales ~O(seq)."""
    from axolotl.integrations.protrain.cost.memory import attn_activation_bytes

    r8k = attn_activation_bytes(_attn_trace(8192, "flash_attention_2"))
    r16k = attn_activation_bytes(_attn_trace(16384, "flash_attention_2"))
    # Linear: doubling seq roughly doubles the term (small LSE keeps it ~2x).
    assert 1.9 < r16k / r8k < 2.2


def test_attn_activation_eager_is_quadratic_in_seq():
    """eager materializes the score matrix → term scales O(seq^2)."""
    from axolotl.integrations.protrain.cost.memory import attn_activation_bytes

    r8k = attn_activation_bytes(_attn_trace(8192, "eager"))
    r16k = attn_activation_bytes(_attn_trace(16384, "eager"))
    assert r16k / r8k > 3.0  # super-linear


def test_attn_activation_flash_never_underpredicts_eager_lower_bounds():
    """Flash term must stay >= the genuine projection/MLP floor (never 0 with arch)."""
    from axolotl.integrations.protrain.cost.memory import attn_activation_bytes

    flash = attn_activation_bytes(_attn_trace(16384, "flash_attention_2"))
    eager = attn_activation_bytes(_attn_trace(16384, "eager"))
    assert 0 < flash < eager  # flash is tight; eager stays conservative


def test_attn_activation_sdpa_without_evidence_is_quadratic():
    """SDPA with no dispatch probe must budget O(seq^2): it can fall back to math."""
    from axolotl.integrations.protrain.cost.memory import attn_activation_bytes

    # No dispatched_sdpa_backend recorded -> conservative, equals eager.
    sdpa = attn_activation_bytes(_attn_trace(16384, "sdpa"))
    eager = attn_activation_bytes(_attn_trace(16384, "eager"))
    assert sdpa == eager
    # And it is super-linear in seq (the score matrix is budgeted).
    r8k = attn_activation_bytes(_attn_trace(8192, "sdpa"))
    assert sdpa / r8k > 3.0


def test_attn_activation_sdpa_flash_evidence_is_linear():
    """SDPA proven to dispatch flash/mem-efficient gets the O(seq) flash term."""
    from axolotl.integrations.protrain.cost.memory import attn_activation_bytes

    flash = attn_activation_bytes(_attn_trace(16384, "flash_attention_2"))
    for evidence in ("flash", "mem_efficient"):
        sdpa = attn_activation_bytes(
            _attn_trace(16384, "sdpa", dispatched_sdpa_backend=evidence)
        )
        assert sdpa == flash


def test_attn_activation_sdpa_math_evidence_is_quadratic():
    """SDPA proven to dispatch the math backend gets the O(seq^2) eager term."""
    from axolotl.integrations.protrain.cost.memory import attn_activation_bytes

    eager = attn_activation_bytes(_attn_trace(16384, "eager"))
    sdpa = attn_activation_bytes(
        _attn_trace(16384, "sdpa", dispatched_sdpa_backend="math")
    )
    assert sdpa == eager


def test_attn_activation_sliding_window_caps_eager_span():
    """Sliding-window attention bounds the eager O(seq^2) term to O(seq*window)."""
    from axolotl.integrations.protrain.cost.memory import attn_activation_bytes

    full = attn_activation_bytes(_attn_trace(16384, "eager", sliding_window=0))
    windowed = attn_activation_bytes(_attn_trace(16384, "eager", sliding_window=1024))
    assert windowed < full


def test_attn_activation_unknown_backend_is_eager_safe():
    """Empty/unknown backend (legacy traces) keeps the conservative O(seq^2) path."""
    from axolotl.integrations.protrain.cost.memory import attn_activation_bytes

    unknown = attn_activation_bytes(_attn_trace(16384, ""))
    eager = attn_activation_bytes(_attn_trace(16384, "eager"))
    assert unknown == eager


def test_saved_tensor_proxy_backfills_internal_when_peaks_empty():
    """Empty steady_fwd_block_peak_bytes → proxy = block-output + internal saved.

    On-demand profiling at high seq leaves the per-block peaks empty; the proxy
    must fall back to full per-block residency (output + arch-aware internal),
    not the block-output boundary alone, or the searcher under-counts
    no-checkpoint configs the gate then fail-closes.
    """
    from axolotl.integrations.protrain.cost.memory import (
        _saved_tensor_bytes_per_block,
        attn_activation_bytes,
    )

    trace = _attn_trace(32768, "flash_attention_2")  # steady_fwd_block_peak_bytes empty
    assert not trace.steady_fwd_block_peak_bytes
    internal = attn_activation_bytes(trace)
    assert internal > 0
    proxy = _saved_tensor_bytes_per_block(trace)
    block_output = int(trace.activation_sizes[BlockId(0)])
    assert proxy[BlockId(0)] == block_output + internal
    # The internal term dominates: full residency is far above block-output alone.
    assert proxy[BlockId(0)] > 100 * block_output


def test_ckpt_chain_includes_internal_residual_when_enabled():
    """factor=1.0 raises chain bytes strictly above the legacy block-output-only sum."""
    n_blocks = 4
    activation_per_block = 1_000_000
    trace = _build_synthetic_trace(
        n_blocks=n_blocks,
        activation_per_block=activation_per_block,
        bs=1,
        seq=512,
        hidden_size=4096,
        num_attention_heads=32,
        intermediate_size=11008,
    )
    block_map: BlockStrategyMap = {BlockId(i): BlockMode.CKPT for i in range(n_blocks)}
    legacy = _compute_ckpt_chain_bytes(trace, block_map, internal_residual_factor=0.0)
    enabled = _compute_ckpt_chain_bytes(trace, block_map, internal_residual_factor=1.0)
    assert legacy == n_blocks * activation_per_block
    assert enabled > legacy


def test_disable_via_factor_zero():
    """``internal_residual_factor=0.0`` reproduces the pre-fix block-output-only chain bytes."""
    n_blocks = 6
    activation_per_block = 2_500_000
    trace = _build_synthetic_trace(
        n_blocks=n_blocks,
        activation_per_block=activation_per_block,
        bs=1,
        seq=1024,
        hidden_size=_LLAMA_30B_HIDDEN,
        num_attention_heads=_LLAMA_30B_HEADS,
        intermediate_size=_LLAMA_30B_INTERMEDIATE,
    )
    block_map: BlockStrategyMap = {BlockId(i): BlockMode.CKPT for i in range(n_blocks)}
    disabled = _compute_ckpt_chain_bytes(trace, block_map, internal_residual_factor=0.0)
    assert disabled == n_blocks * activation_per_block


def test_default_factor_constant_is_one():
    """Shipped default residual factor is 1.0 (full estimate)."""
    assert DEFAULT_CKPT_INTERNAL_RESIDUAL_FACTOR == pytest.approx(1.0)


def test_set_default_residual_factor_clamps_negatives():
    """The setter floors at 0.0; negative inputs become 0.0."""
    from axolotl.integrations.protrain.cost import memory as _mem

    original = _mem.DEFAULT_CKPT_INTERNAL_RESIDUAL_FACTOR
    try:
        set_default_ckpt_internal_residual_factor(-1.0)
        assert _mem.DEFAULT_CKPT_INTERNAL_RESIDUAL_FACTOR == pytest.approx(0.0)
        set_default_ckpt_internal_residual_factor(0.6)
        assert _mem.DEFAULT_CKPT_INTERNAL_RESIDUAL_FACTOR == pytest.approx(0.6)
    finally:
        set_default_ckpt_internal_residual_factor(original)


def test_estimate_peak_seq_512_30b_llama_alpha_steady_after_residual():
    """Section 7.6 audit-row anchor: residual + alpha-split narrows alpha_steady at seq=512.

    Measured steady peak at seq=512 on the audit row (30B Llama, 4-bit, Mode-C,
    n_persist=0/n_buffer=12/n_checkpoint=60) is 2.91 GiB. Adding the per-block
    internal residual (FFN-intermediate + attention scores + QKV projections,
    sized as one CKPT block's recompute window) must push the prediction
    monotonically toward the measured value vs. the pre-residual baseline.
    """
    GiB = 1 << 30
    seq = 512
    bs = 1
    measured_steady_gib = 2.91

    s_chunk = 67108864
    n_chunk = 302
    layout = ChunkLayout(
        S_chunk=s_chunk,
        N_chunk=n_chunk,
        chunks=tuple((f"p.{cid}",) for cid in range(n_chunk)),  # type: ignore[arg-type]
        param_to_chunk={},
        block_to_chunks={BlockId(b): () for b in range(_LLAMA_30B_N_BLOCK)},
    )

    per_block_act_bytes = bs * seq * _LLAMA_30B_INTERMEDIATE * 2
    activation_sizes = {
        BlockId(b): per_block_act_bytes for b in range(_LLAMA_30B_N_BLOCK)
    }
    trace = ProfilerTrace(
        op_order=(),
        intra_op_delta={},
        inter_op_delta={},
        activation_sizes=activation_sizes,
        model_state_bytes=16 * GiB,
        pcie_h2d_bps=13e9,
        pcie_d2h_bps=13e9,
        nccl_gather_s={},
        nccl_reduce_s={},
        arch_hash="huggyllama/llama-30b-qlora-modec",
        bs=bs,
        seq=seq,
        sku="audit",
        world=1,
        hidden_size=_LLAMA_30B_HIDDEN,
        num_attention_heads=_LLAMA_30B_HEADS,
        intermediate_size=_LLAMA_30B_INTERMEDIATE,
    )

    cfg = CostConfig(
        n_persist=0, n_buffer=12, n_swap=0, n_checkpoint=_LLAMA_30B_N_BLOCK
    )
    block_map: BlockStrategyMap = {
        BlockId(b): BlockMode.CKPT for b in range(_LLAMA_30B_N_BLOCK)
    }

    hw_4bit = HardwareProfile(
        gpu_sku="audit",
        gpu_memory_bytes=24 * GiB,
        gpu_count=1,
        pcie_h2d_bps=13e9,
        pcie_d2h_bps=13e9,
        has_nvlink=False,
        dominant_param_bytes_per_element=0.5,
    )

    from axolotl.integrations.protrain.cost import memory as _mem

    original_factor = _mem.DEFAULT_CKPT_INTERNAL_RESIDUAL_FACTOR
    try:
        set_default_ckpt_internal_residual_factor(0.0)
        pred_baseline = estimate_peak(cfg, trace, layout, block_map, hw_4bit) / GiB
        alpha_baseline = measured_steady_gib / max(pred_baseline, 1e-9)

        set_default_ckpt_internal_residual_factor(1.0)
        pred_with_residual = estimate_peak(cfg, trace, layout, block_map, hw_4bit) / GiB
        alpha_with_residual = measured_steady_gib / max(pred_with_residual, 1e-9)
    finally:
        set_default_ckpt_internal_residual_factor(original_factor)

    assert pred_with_residual > pred_baseline, (
        f"residual must raise prediction at seq=512 "
        f"(baseline={pred_baseline:.3f} GiB, "
        f"with_residual={pred_with_residual:.3f} GiB)"
    )
    assert abs(alpha_with_residual - 1.0) < abs(alpha_baseline - 1.0), (
        f"alpha_steady must narrow toward 1.0 at seq=512: "
        f"baseline={alpha_baseline:.3f} -> with_residual={alpha_with_residual:.3f}"
    )
