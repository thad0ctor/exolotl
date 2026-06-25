"""Peak-memory reconstruction for the ProTrain searcher."""

from __future__ import annotations

import logging
from collections import defaultdict

from axolotl.integrations.protrain.types import (
    BlockId,
    BlockMode,
    BlockStrategyMap,
    ChunkLayout,
    CostConfig,
    HardwareProfile,
    OpRecord,
    ProfilerTrace,
)
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


_STALE_TRACE_WARNING_EMITTED = False


def _saved_tensor_bytes_per_block(trace: ProfilerTrace) -> dict[BlockId, int]:
    """Per-block saved-for-backward bytes proxy from steady-state forward peaks."""
    deltas: dict[BlockId, int] = {}
    persisted_per_block_peak = getattr(trace, "steady_fwd_block_peak_bytes", None) or {}
    # Normalize key types so JSON/pickle round-trips still hit per_block_peak.get(BlockId).
    per_block_peak: dict[BlockId, int] = {
        BlockId(int(block_id)): int(peak_bytes)
        for block_id, peak_bytes in persisted_per_block_peak.items()
    }
    activation_sizes = trace.activation_sizes or {}
    # When the profiler skipped per-block peak capture — on-demand mode at high
    # seq leaves ``steady_fwd_block_peak_bytes`` empty (all steady-state fields
    # zero) — ``activation_sizes`` carries only the block-OUTPUT boundary
    # tensor. A NONE/OFFLOAD block recomputes nothing, so its FULL forward saved
    # set (Q/K/V/O projections, FA softmax-LSE, FFN intermediate) stays resident
    # across the backward window. The block-output proxy alone misses that
    # internal residency, so no-checkpoint configs look ~order-of-magnitude
    # cheaper than reality and the searcher picks n_checkpoint=0 the gate then
    # fail-closes at long seq. Backfill the internal term analytically (arch +
    # backend aware) so the fallback proxies full per-block residency.
    internal_saved = attn_activation_bytes(trace)
    # Enc-dec decoder blocks carry a second (cross-)attention module; add it for tree index > 0 (no-op decoder-only).
    tree_index_map = block_tree_index_map(trace)
    cross_saved = (
        attn_activation_bytes(trace) if _has_multiple_trees(tree_index_map) else 0
    )

    def _full_residency(bid: BlockId, sz: int) -> int:
        extra = cross_saved if tree_index_map.get(BlockId(int(bid)), 0) > 0 else 0
        return int(sz) + internal_saved + extra

    if not per_block_peak:
        return {
            BlockId(int(bid)): _full_residency(bid, sz)
            for bid, sz in activation_sizes.items()
        }

    # Sort by block id to walk in forward order. Keys are already
    # canonical ``BlockId`` after the normalization above, so sorted()
    # operates on int-equivalent NewType values without further coercion.
    sorted_bids = sorted(per_block_peak.keys())
    if not sorted_bids:
        return {
            BlockId(int(bid)): _full_residency(bid, sz)
            for bid, sz in activation_sizes.items()
        }

    forward_diffs: list[int] = []
    for prev_bid, cur_bid in zip(sorted_bids, sorted_bids[1:], strict=False):
        prev_peak = per_block_peak.get(prev_bid, 0)
        cur_peak = per_block_peak.get(cur_bid, 0)
        diff = cur_peak - prev_peak
        if diff > 0:
            forward_diffs.append(diff)
            # Forward difference attributes the cumulative-allocated rise
            # between prev and cur to the bytes prev_bid deposited.
            deltas[prev_bid] = diff

    # Last block has no successor to diff against. Use the median of the
    # observed forward diffs as a robust fallback. When ``forward_diffs``
    # is empty (single block, or every diff was non-positive — unusual
    # but possible if profiling captured a non-uniform steady state),
    # fall back to ``activation_sizes`` for every block.
    if forward_diffs:
        import statistics

        median_diff = int(statistics.median(forward_diffs))
        last_bid = sorted_bids[-1]
        if last_bid not in deltas:
            deltas[last_bid] = median_diff

    # Fill gaps for blocks missing or zero in the per-block peaks. Measured
    # forward diffs already include internal saved tensors, so a gap-filled
    # block must match that basis (output + internal), not output alone.
    for bid_raw, act_sz in activation_sizes.items():
        bid = BlockId(int(bid_raw))
        if bid not in deltas:
            deltas[bid] = _full_residency(bid, act_sz)

    return deltas


def _normalized_activation_sizes(trace: ProfilerTrace) -> dict[BlockId, int]:
    """Re-key ``trace.activation_sizes`` to canonical ``BlockId``.

    Persisted (JSON/pickle) traces stringify dict keys, so ``.get(BlockId)``
    lookups miss and undercount. Normalize once so all BlockId-keyed lookups
    hit the right entries.
    """
    return {
        BlockId(int(bid)): int(sz) for bid, sz in (trace.activation_sizes or {}).items()
    }


# Eq. 11 fragmentation factor; fp16/bf16/8-bit default. Per-dtype override in alpha_fragmentation_for_dtype.
ALPHA_FRAGMENTATION: float = 1.10

# bnb-4-bit alpha; calibrated against Mode-A peaks where frozen 4-bit weights
# dominate per-rank residency and the activation slice is small (empirical
# alpha_measured ~0.70 with cushion).
ALPHA_FRAGMENTATION_4BIT: float = 0.75
ALPHA_FRAGMENTATION_4BIT_MODE_A: float = ALPHA_FRAGMENTATION_4BIT

# Mode-C with gradient checkpointing reverses the dominance: activation churn
# is the dominant moving term, and the 0.75 Mode-A factor structurally
# under-predicts the raw peak (audit table Decision 1 shows alpha_steady
# = 1.43 / 1.25 / 1.08 at seq=512/1024/2048 on 30B-Llama Mode-C, i.e. raw
# predictor low by 8-31% at low seq). 0.95 narrows the gap without
# changing the wrapper-side calibration safety semantics.
# Calibration anchor only: gate_consistent_alpha floors it to 1.0 in estimate_peak; survives in the diagnostic log.
ALPHA_FRAGMENTATION_4BIT_MODE_C_CKPT: float = 0.95


def alpha_fragmentation_for_dtype(bytes_per_element: float) -> float:
    """Return ALPHA_FRAGMENTATION_4BIT for sub-byte dtypes, else ALPHA_FRAGMENTATION.

    Legacy dtype-only entry point; preserved so call sites that lack a
    ``CostConfig`` (e.g. ``predict_init_transient_peak_bytes``) keep their
    Mode-A calibration. Mode-aware call sites should use
    :func:`alpha_fragmentation_for_cfg`.
    """
    if bytes_per_element < 1.0:
        return ALPHA_FRAGMENTATION_4BIT_MODE_A
    return ALPHA_FRAGMENTATION


def alpha_fragmentation_for_cfg(
    bytes_per_element: float, cfg: CostConfig | None, is_mode_c: bool = False
) -> float:
    """Pick the fragmentation factor for ``(dtype, mode)``.

    Mode-A and Mode-C-without-CKPT keep the 0.75 factor (activation slice
    is small, frozen-weight residency dominates). Mode-C-with-CKPT uses
    0.95 because activation churn fragments differently. Non-4-bit dtypes
    keep the unconditional ``ALPHA_FRAGMENTATION`` (1.10) regardless of
    mode.

    ``is_mode_c`` reflects ZeRO-3 chunk sharding (``HardwareProfile.zero3_shard``);
    the 0.95 factor only applies under Mode C, so checkpointed Mode-A
    candidates are NOT over-predicted with it. Defaults to ``False``
    (Mode A) so callers without a hardware profile keep the 0.75 baseline.
    """
    if bytes_per_element >= 1.0:
        return ALPHA_FRAGMENTATION
    is_ckpt = bool(cfg is not None and int(getattr(cfg, "n_checkpoint", 0)) > 0)
    if is_ckpt and is_mode_c:
        # Floored to 1.0 by gate_consistent_alpha; raw value reaches only the diagnostic log.
        return ALPHA_FRAGMENTATION_4BIT_MODE_C_CKPT
    return ALPHA_FRAGMENTATION_4BIT_MODE_A


# Default multiplier for the per-block CKPT internal-residual proxy. Mutable so
# the plugin can apply the YAML knob (``protrain_ckpt_internal_residual_factor``)
# without threading it through every cost-model call site. 1.0 = include the full
# FFN-intermediate + attention-score + QKV proxy; 0.0 = disable the residual
# (reproduces the pre-fix block-output-only chain).
DEFAULT_CKPT_INTERNAL_RESIDUAL_FACTOR: float = 1.0


def set_default_ckpt_internal_residual_factor(factor: float) -> None:
    """Set the process-global default residual factor read by helpers when no explicit value is passed."""
    global DEFAULT_CKPT_INTERNAL_RESIDUAL_FACTOR
    DEFAULT_CKPT_INTERNAL_RESIDUAL_FACTOR = float(max(0.0, factor))


# Backends that stream the softmax (O(seq), no score matrix). "sdpa" is excluded — resolved per-trace in _attn_is_linear_mem.
_LINEAR_MEM_ATTN_BACKENDS: frozenset[str] = frozenset(
    {
        "flash_attention_2",
        "flash_attention_3",
        "flash_attn",
        "flash",
        "xformers",
        "mem_efficient",
    }
)


def _attn_is_linear_mem(trace: "ProfilerTrace", backend: str) -> bool:
    """True if attention streams the softmax (O(seq), no score matrix).

    Flash/mem-efficient backends are always linear. SDPA is linear ONLY when the
    warmup probe recorded a flash/mem-efficient dispatch; without that evidence
    (a math dispatch, or no probe at all) we conservatively budget O(seq^2),
    because SDPA can fall back to the math backend. Over-prediction is safe;
    under-prediction risks a runtime OOM.
    """
    if backend in _LINEAR_MEM_ATTN_BACKENDS:
        return True
    if backend == "sdpa":
        dispatched = str(getattr(trace, "dispatched_sdpa_backend", "") or "").lower()
        return dispatched in {"flash", "mem_efficient"}
    return False


# Gated-MLP (SwiGLU) saved-for-backward intermediates per block: gate_proj
# output, up_proj output, and the silu(gate)*up product — three seq*ffn_width
# tensors. See attn_activation_bytes for the empirical justification.
_GATED_MLP_SAVED_INTERMEDIATES: int = 3


def attn_activation_bytes(
    trace: ProfilerTrace,
    bytes_per_element: float = 2.0,
) -> int:
    """Per-block attention + MLP saved-for-backward bytes, backend-aware.

    ``trace.activation_sizes[bid]`` proxies only the BLOCK OUTPUT (the CKPT
    boundary tensor); this estimates the INTERNAL saved tensors the proxy
    misses for one transformer block, choosing the formula by attention
    MECHANISM so the cost model doesn't budget a score matrix the kernel never
    allocates.

    Backend cases (``trace.attn_implementation``):

    - Flash-Attention-2/3, xFormers, mem-efficient (and SDPA *proven* to
      dispatch flash/mem-efficient via the warmup probe) → O(seq): Q,K,V,O
      projections + the FA softmax-LSE working set. NO score matrix. KV is
      scaled by ``num_key_value_heads`` (GQA) so grouped-query models aren't
      over-counted.
    - eager / math / SDPA without flash evidence → O(seq^2): the full score
      matrix ``mbs * heads * seq^2`` is real and retained (conservative).
      Sliding-window attention caps the span to ``min(seq, window)`` so even
      eager degrades to O(seq * window). See ``_attn_is_linear_mem``.

    MLP term follows the MoE config: dense models use ``intermediate_size``;
    MoE models that expose ``num_experts_per_tok`` / ``moe_intermediate_size``
    use the active-expert width (``experts_per_tok * moe_intermediate``).

    Returns 0 when the trace lacks bs/seq/hidden (legacy traces) so the chain
    degrades to the block-output-only sum.
    """
    bs = int(getattr(trace, "bs", 0) or 0)
    seq = int(getattr(trace, "seq", 0) or 0)
    hidden = int(getattr(trace, "hidden_size", 0) or 0)
    heads = int(getattr(trace, "num_attention_heads", 0) or 0)
    intermediate = int(getattr(trace, "intermediate_size", 0) or 0)
    if bs <= 0 or seq <= 0 or hidden <= 0:
        return 0

    kv_heads = int(getattr(trace, "num_key_value_heads", 0) or 0) or max(heads, 1)
    head_dim = int(getattr(trace, "head_dim", 0) or 0)
    if head_dim <= 0:
        head_dim = hidden // max(heads, 1)
    window = int(getattr(trace, "sliding_window", 0) or 0)
    backend = str(getattr(trace, "attn_implementation", "") or "").lower()

    # Q,K,V projections with GQA: K,V have only kv_heads * head_dim columns.
    qkv_bytes = bs * seq * (max(heads, 1) + 2 * kv_heads) * head_dim * bytes_per_element
    # Attention output projection (retained input to o_proj).
    out_bytes = bs * seq * max(heads, 1) * head_dim * bytes_per_element

    if heads > 0 and not _attn_is_linear_mem(trace, backend):
        # eager / math / unproven-SDPA: full score matrix is materialized +
        # retained. Sliding window bounds the second seq factor to the span.
        span = min(seq, window) if window > 0 else seq
        attn_core_bytes = bs * heads * seq * span * bytes_per_element
    else:
        # Flash / xFormers / mem-efficient: no score matrix. The retained
        # working set is the projections + a small softmax-LSE (O(seq*heads),
        # fp32). 4 bytes/elem for the LSE regardless of compute dtype.
        lse_bytes = bs * max(heads, 1) * seq * 4
        attn_core_bytes = lse_bytes

    experts_per_tok = int(getattr(trace, "num_experts_per_tok", 0) or 0)
    moe_intermediate = int(getattr(trace, "moe_intermediate_size", 0) or 0)
    if experts_per_tok > 0 and moe_intermediate > 0:
        ffn_width = experts_per_tok * moe_intermediate
    else:
        ffn_width = intermediate
    # Gated MLP (SwiGLU — Llama/Qwen/Mistral/...) retains THREE seq*ffn_width
    # intermediates for backward: gate_proj output, up_proj output, and the
    # silu(gate)*up product. Counting a single intermediate under-counts the
    # per-block recompute working set ~2x at long seq — measured on Qwen3-14B,
    # the gate tracked actual max_active to +-0.1 GiB at <=8k but drifted -1.3
    # GiB at 16k (and worse at 32k) until this was corrected; recompute is O(seq)
    # (FA2, no score matrix), so the fix is this linear factor, not an O(seq^2)
    # term. Non-gated MLPs save fewer, so this stays conservative (never under).
    ffn_bytes = (
        bs * seq * ffn_width * _GATED_MLP_SAVED_INTERMEDIATES * bytes_per_element
        if ffn_width > 0
        else 0
    )

    return int(qkv_bytes + out_bytes + attn_core_bytes + ffn_bytes)


def _block_internal_saved_bytes(
    trace: ProfilerTrace,
    block_id: BlockId,
    bytes_per_element: float = 2.0,
) -> int:
    """Compat shim over :func:`attn_activation_bytes` (block_id reserved for heterogeneous-arch hooks)."""
    _ = block_id
    return attn_activation_bytes(trace, bytes_per_element=bytes_per_element)


def _compute_ckpt_chain_bytes(
    trace: ProfilerTrace,
    block_map: BlockStrategyMap | None,
    internal_residual_factor: float | None = None,
    bytes_per_element: float = 2.0,
) -> int:
    """Sum activation bytes across CKPT blocks plus one block's internal residual.

    CKPT blocks retain the block-input boundary tensor across the full
    backward window, so the chain is a sum over CKPT blocks rather than a
    single-block max. Used by both :func:`estimate_peak` and the wrapper-side
    ``_reconstruct_f_bm`` so the analytical and calibrated paths agree.

    Under non-reentrant CKPT, only one block's INTERNAL saved tensors
    (Q/K/V projections, attention scores, FFN intermediate) are live at a
    time — the current recompute window. This helper adds ONE block's
    worth of internal residual on top of the chain (a per-block max, not
    N_block worth) to avoid over-correcting at high seq where the
    attention-score term is O(seq^2). The block-output proxy on its own
    misses these tensors; the residual closes the gap at low seq where the
    chain term is small. ``internal_residual_factor`` (defaults to
    ``DEFAULT_CKPT_INTERNAL_RESIDUAL_FACTOR`` = 1.0) scales the
    contribution; 0.0 disables it (pre-fix behavior). See DESIGN.md
    Decision 1.
    """
    if not block_map or not trace.activation_sizes:
        return 0
    if internal_residual_factor is None:
        internal_residual_factor = DEFAULT_CKPT_INTERNAL_RESIDUAL_FACTOR
    total = 0
    max_residual = 0
    for bid_raw, act_sz in trace.activation_sizes.items():
        bid = BlockId(int(bid_raw))
        if block_map.get(bid, BlockMode.NONE) is BlockMode.CKPT:
            total += int(act_sz)
            if internal_residual_factor > 0.0:
                residual = _block_internal_saved_bytes(
                    trace, bid, bytes_per_element=bytes_per_element
                )
                if residual > max_residual:
                    max_residual = residual
    if internal_residual_factor > 0.0 and max_residual > 0:
        total += int(internal_residual_factor * max_residual)
    return total


def _group_ops_by_block(trace: ProfilerTrace) -> dict[BlockId, list[int]]:
    """Return ``{block_id -> [op_positions]}`` for forward ops only."""
    grouped: dict[BlockId, list[int]] = defaultdict(list)
    for i, op in enumerate(trace.op_order):
        if not op.is_forward:
            continue
        if op.block_id is None:
            continue
        grouped[op.block_id].append(i)
    return grouped


def _tree_index_for_path(module_path: str) -> int:
    """Best-effort tree-index inference from a module path."""
    normalized = module_path.removeprefix("base_model.model.").removeprefix(
        "base_model."
    )
    if normalized.startswith("encoder.") or normalized == "encoder":
        return 0
    if normalized.startswith("decoder.") or normalized == "decoder":
        return 1
    return 0


def block_tree_index_map(
    trace: ProfilerTrace,
) -> dict[BlockId, int]:
    """Map each ``BlockId`` to its forward-order tree index."""
    persisted = getattr(trace, "block_tree_index", None)
    if persisted:
        # Normalise JSON/pickle string keys back to BlockId.
        return {
            BlockId(int(block_id)): int(tree_index)
            for block_id, tree_index in persisted.items()
        }
    seen: dict[BlockId, int] = {}
    for op in trace.op_order:
        if not op.is_forward or op.block_id is None:
            continue
        if op.block_id in seen:
            continue
        seen[op.block_id] = _tree_index_for_path(op.module_path)
    return seen


def _has_multiple_trees(tree_index_map: dict[BlockId, int]) -> bool:
    """Return True iff at least two distinct tree indices are present."""
    if not tree_index_map:
        return False
    indices = set(tree_index_map.values())
    return len(indices) >= 2


def cross_attn_persist_bytes(
    trace: ProfilerTrace,
    block_map: BlockStrategyMap,
    tree_index_map: dict[BlockId, int],
) -> int:
    """Estimate cross-attention saved-state bytes that span trees."""
    if not _has_multiple_trees(tree_index_map):
        return 0
    encoder_bids = sorted(bid for bid, idx in tree_index_map.items() if idx == 0)
    if not encoder_bids:
        return 0
    last_enc_bid = encoder_bids[-1]
    last_enc_mode = block_map.get(last_enc_bid, BlockMode.NONE)
    if last_enc_mode is BlockMode.NONE or last_enc_mode is BlockMode.OFFLOAD:
        # Already counted in retained_none_bytes.
        return 0
    if last_enc_mode is BlockMode.CKPT:
        # ckpt_chain_bytes already covers residual.
        return 0
    return int(_normalized_activation_sizes(trace).get(last_enc_bid, 0))


def cross_attn_handoff_bytes(
    trace: ProfilerTrace,
    block_map: BlockStrategyMap,
    tree_index_map: dict[BlockId, int],
) -> int:
    """Return encoder-decoder handoff bytes regardless of encoder-last mode (cap-path use)."""
    if not _has_multiple_trees(tree_index_map):
        return 0
    encoder_bids = sorted(bid for bid, idx in tree_index_map.items() if idx == 0)
    if not encoder_bids:
        return 0
    last_enc_bid = encoder_bids[-1]
    last_enc_mode = block_map.get(last_enc_bid, BlockMode.NONE)
    # NONE/OFFLOAD already retain the full block bytes on GPU so the cap need not preserve them again.
    if last_enc_mode is BlockMode.NONE or last_enc_mode is BlockMode.OFFLOAD:
        return 0
    return int(_normalized_activation_sizes(trace).get(last_enc_bid, 0))


def op_cross_attn_surcharge(
    op: OpRecord,
    cross_attn_bytes: int,
    tree_index_map: dict[BlockId, int],
) -> int:
    """Per-op cross-attention surcharge during decoder forward."""
    if cross_attn_bytes <= 0 or op.block_id is None:
        return 0
    if tree_index_map.get(op.block_id, 0) > 0:
        return cross_attn_bytes
    return 0


def hot_iter_peak_cap(
    trace: ProfilerTrace,
    block_map: BlockStrategyMap,
    cfg: CostConfig | None = None,
    layout: ChunkLayout | None = None,
) -> int | None:
    """Measured ground-truth upper bound on the raw op-walk peak, or None."""
    if trace.steady_fwd_block_peak_bytes:
        forward_max_block_peak = max(trace.steady_fwd_block_peak_bytes.values())
        # Backward-aware: bwd holds grads + saved acts simultaneously; take max with bwd peaks.
        bwd_per_block = getattr(trace, "steady_bwd_block_peak_bytes", None) or {}
        if bwd_per_block:
            backward_max_block_peak = max(bwd_per_block.values())
            forward_max_block_peak = max(
                forward_max_block_peak, backward_max_block_peak
            )
        ckpt_recomp_bump = 0
        has_offload = False
        # Subtract CKPT/SWAP savings from the all-NONE ceiling so the cap shrinks with n_ckpt/n_swap.
        # Use per-block forward-peak deltas (saved-tensor proxy) not activation_sizes (output-only).
        saved_bytes_proxy = _saved_tensor_bytes_per_block(trace)
        # Encoder→decoder handoff: preserve cross-attn bytes when encoder-last is CKPT/SWAP.
        tree_index_map = block_tree_index_map(trace)
        cross_attn_bytes_for_cap = cross_attn_handoff_bytes(
            trace, block_map, tree_index_map
        )
        encoder_last_bid: BlockId | None = None
        if cross_attn_bytes_for_cap > 0:
            encoder_bids = sorted(
                bid for bid, idx in tree_index_map.items() if idx == 0
            )
            if encoder_bids:
                encoder_last_bid = encoder_bids[-1]
        ckpt_swap_savings = 0
        for bid_raw, act_sz in trace.activation_sizes.items():
            bid = BlockId(int(bid_raw))
            mode = block_map.get(bid, BlockMode.NONE)
            block_saved = int(saved_bytes_proxy.get(bid, act_sz))
            if (
                encoder_last_bid is not None
                and bid == encoder_last_bid
                and mode in (BlockMode.CKPT, BlockMode.SWAP)
            ):
                # Cross-attn output cannot be reclaimed on encoder-last under CKPT/SWAP.
                block_saved = max(0, block_saved - cross_attn_bytes_for_cap)
            if mode is BlockMode.CKPT:
                if act_sz > ckpt_recomp_bump:
                    ckpt_recomp_bump = act_sz
                ckpt_swap_savings += block_saved
            elif mode is BlockMode.SWAP:
                ckpt_swap_savings += block_saved
            elif mode is BlockMode.OFFLOAD:
                has_offload = True
        offload_bump = layout.S_chunk if (has_offload and layout is not None) else 0
        # Floor activation portion at zero; model-state floor preserved via add-back below.
        profile_time_model_state = (
            layout.N_chunk * layout.S_chunk if layout is not None else 0
        )
        # All-NONE ceiling = model_state + activation; only activation shrinks with CKPT/SWAP.
        all_none_activation_ceiling = max(
            0, forward_max_block_peak - profile_time_model_state
        )
        capped_savings = min(ckpt_swap_savings, all_none_activation_ceiling)
        return (
            profile_time_model_state
            + (all_none_activation_ceiling - capped_savings)
            + ckpt_recomp_bump
            + offload_bump
        )
    # Aggregate fallback; take max of fwd/bwd aggregates since bwd often exceeds fwd.
    bwd_aggregate = int(getattr(trace, "steady_bwd_peak_bytes", 0))
    aggregate_cap = max(int(trace.steady_fwd_peak_bytes), bwd_aggregate)
    if (
        aggregate_cap > 0
        and cfg is not None
        and cfg.n_checkpoint == 0
        and cfg.n_swap == 0
        and cfg.n_offload == 0
    ):
        return aggregate_cap
    return None


def apply_hot_iter_cap(
    raw_peak: int,
    model_state_present: int,
    measured_cap: int | None,
    layout: ChunkLayout,
) -> int:
    """Apply :func:`hot_iter_peak_cap` to ``raw_peak`` using layered decomposition."""
    if measured_cap is None:
        return raw_peak
    # Layered cap on activation portion only; model_state preserved.
    op_walk_portion = max(0, raw_peak - model_state_present)
    # Subtract profile-time params to recover activation ceiling; clamp at 0 for synthetic traces.
    profile_time_model_state = layout.N_chunk * layout.S_chunk
    measured_activation_cap = max(0, measured_cap - profile_time_model_state)
    if op_walk_portion > measured_activation_cap:
        op_walk_portion = measured_activation_cap
    # Reassemble: model_state_present is preserved through the cap.
    return model_state_present + op_walk_portion


# Pool sizing knobs mirrored from block.swap_pool.ActivationSwapPool; keep in sync.
SWAP_PREFETCH_DEPTH: int = 2
SWAP_SLOTS_PER_BLOCK: int = 8

# Per-block saved-for-backward tensors a non-recomputed (SWAP/NONE) block keeps
# resident on the host once swapped: block I/O + each linear's saved input + the
# FFN intermediates. The fused QLoRA stack (FLCE, fused RMSNorm/attn, LoRA
# kernels) collapses much of the naive set; measured on Qwen3-14B the surviving
# population is ~12*(seq*hidden) + ~5*(seq*intermediate) bytes/elem. Coefficients
# are deliberately conservative: the SWAP slab is sized from this, and
# UNDER-provisioning spills tensors back onto the GPU and breaks the steady-peak
# prediction, whereas over-provisioning only costs (abundant) pinned host RAM.
_SWAP_HIDDEN_TENSORS_PER_BLOCK: int = 12
_SWAP_INTERMEDIATE_TENSORS_PER_BLOCK: int = 5

# Headroom over n_swap * per-block aggregate for first-fit fragmentation + a few
# in-flight backward-prefetch slices.
_SWAP_POOL_HEADROOM: float = 1.25


def swap_block_saved_bytes(trace: ProfilerTrace, bytes_per_element: float = 2.0) -> int:
    """Host bytes one SWAP block's saved-for-backward set occupies, arch-aware."""
    seq = int(getattr(trace, "seq", 0) or 0)
    hidden = int(getattr(trace, "hidden_size", 0) or 0)
    if seq <= 0 or hidden <= 0:
        # Legacy trace without arch fields: block-output proxy x a count heuristic.
        outs = list((trace.activation_sizes or {}).values())
        base = max((int(v) for v in outs), default=0)
        return int(base * SWAP_SLOTS_PER_BLOCK)
    intermediate = int(getattr(trace, "intermediate_size", 0) or 0)
    experts_per_tok = int(getattr(trace, "num_experts_per_tok", 0) or 0)
    moe_intermediate = int(getattr(trace, "moe_intermediate_size", 0) or 0)
    if experts_per_tok > 0 and moe_intermediate > 0:
        intermediate = experts_per_tok * moe_intermediate
    hidden_bytes = _SWAP_HIDDEN_TENSORS_PER_BLOCK * seq * hidden
    inter_bytes = _SWAP_INTERMEDIATE_TENSORS_PER_BLOCK * seq * max(intermediate, 0)
    return int((hidden_bytes + inter_bytes) * bytes_per_element)


def swap_pool_capacity_bytes(
    trace: ProfilerTrace, n_swap: int, bytes_per_element: float = 2.0
) -> int:
    """Pinned-host capacity for the activation-swap slab: n_swap blocks coexist."""
    n = int(n_swap)
    if n <= 0:
        return 0
    per_block = swap_block_saved_bytes(trace, bytes_per_element=bytes_per_element)
    return int(n * per_block * _SWAP_POOL_HEADROOM)


def estimate_cpu_footprint(
    cfg: CostConfig,
    layout: ChunkLayout,
    hw: HardwareProfile,
    trace: ProfilerTrace | None = None,
) -> int:
    """Per-rank pinned CPU bytes held by non-persistent chunks + SWAP slots."""
    n_persist_eff = len(layout.effective_persistent_ids(cfg.n_persist))
    non_persist = max(0, layout.N_chunk - n_persist_eff)
    # trace.world is the real world_size; hw.gpu_count is local-only.
    if hw.zero3_shard:
        per_rank_divisor = trace.world if trace is not None else hw.gpu_count
    else:
        per_rank_divisor = 1
    per_rank_divisor = max(1, per_rank_divisor)
    per_chunk_sharded = (layout.S_chunk + per_rank_divisor - 1) // per_rank_divisor
    chunk_term = non_persist * per_chunk_sharded

    # Swap term rank-local: the variable-size slab holds all n_swap blocks'
    # saved-for-backward sets simultaneously (carved per-tensor, not equal slots).
    swap_term = 0
    if cfg.n_swap > 0 and trace is not None:
        swap_term = swap_pool_capacity_bytes(trace, cfg.n_swap)

    return chunk_term + swap_term


def model_state_present_bytes(
    cfg: CostConfig,
    layout: ChunkLayout,
    trace: ProfilerTrace,
) -> int:
    """Resident model-state bytes for the runtime's persistent + buffer set."""
    persistent_ids = layout.effective_persistent_ids(cfg.n_persist)
    n_persist_eff = len(persistent_ids)
    n_buffer = max(0, min(cfg.n_buffer, layout.N_chunk - n_persist_eff))

    fp16_total_bytes = layout.N_chunk * layout.S_chunk
    model_state_total = int(getattr(trace, "model_state_bytes", 0) or 0)
    if fp16_total_bytes > 0 and model_state_total > 0:
        # Clamp >= 1.0: aggregate includes fp16 params themselves.
        persistent_factor = max(1.0, model_state_total / fp16_total_bytes)
    else:
        # Process-scoped warn-once: search loop calls this O(candidates) times.
        global _STALE_TRACE_WARNING_EMITTED
        if not _STALE_TRACE_WARNING_EMITTED:
            LOG.warning(
                "model_state_present_bytes: trace.model_state_bytes is missing "
                "or zero (model_state_bytes=%d, fp16_total=%dB); falling back "
                "to the legacy n_persist*S_chunk multiplier. The peak estimate "
                "will UNDER-count full optimizer state — refresh the profiler "
                "trace cache (TRACE_VERSION bump) to restore Eq. 11 fidelity.",
                model_state_total,
                fp16_total_bytes,
            )
            _STALE_TRACE_WARNING_EMITTED = True
        persistent_factor = 1.0
    # Buffer slot during backward: fp16 params + fp16 grads.
    buffer_factor = 2.0
    return int(
        n_persist_eff * layout.S_chunk * persistent_factor
        + n_buffer * layout.S_chunk * buffer_factor
    )


def gate_consistent_alpha(alpha: float) -> float:
    """Floor ``alpha`` at 1.0 so the searcher never deflates a realistic sum.

    Mirrors the wrapper-side ``_calibrate_peak_with_actual_chunk_bytes``, whose
    ``calibration_alpha = max(1.0, ...)`` floors the fragmentation factor at 1.0
    — the 0.75 4-bit Mode-A factor was a phantom-compensator for an old
    O(seq^2) attention over-estimate, and applying it to a realistic component
    sum DEFLATES the peak (the searcher then accepts aggressive configs the
    calibrated gate rejects, fail-closing at runtime). Only the FLOOR is
    mirrored: the gate additionally caps at 1.05, but that cap is paired with
    its explicit additive component terms; ``estimate_peak``'s non-4-bit
    ``raw_peak`` carries the conservative 1.10 dtype alpha instead, so capping
    it would UNDER-predict relative to the established non-4-bit baseline.
    Floor-only keeps the fix safe (never deflates) and back-compatible.
    """
    return max(1.0, float(alpha))


def trainable_state_floor_bytes(trace: ProfilerTrace) -> int:
    """Trainable grad + optimizer-state bytes the searcher must always count.

    Returns the trace-recorded ``trainable_training_state_bytes`` (populated at
    profile time from the model-state footprint). This is a flat additive floor
    so the searcher counts the full trainable optimizer state regardless of
    ``n_persist`` — matching the calibrated gate, which adds the term
    explicitly. ``model_state_present_bytes`` smears the same bytes into
    ``persistent_factor``, which rounds toward 1.0 when a large frozen base
    dominates and under-counts the optimizer state. Legacy traces lacking the
    field (== 0) degrade to the smear (no floor), preserving back-compat.
    """
    return int(getattr(trace, "trainable_training_state_bytes", 0) or 0)


def apply_gate_consistent_scaling(
    raw_peak: int,
    model_state_present: int,
    trainable_floor: int,
    alpha: float,
) -> int:
    """Scale ``raw_peak`` the way the calibrated gate does.

    1. Floor/cap the fragmentation alpha at ``[1.0, 1.05]`` (no deflation).
    2. Add the trainable optimizer-state floor that ``model_state_present`` may
       have under-smeared, BUT only the portion not already resident: when
       ``model_state_present`` already exceeds the trainable floor (large
       ``n_persist`` recovers the full state through ``persistent_factor``), the
       additive top-up is zero, so we never double-count.

    Never under-predicts relative to the un-floored ``alpha * raw_peak``.
    """
    gate_alpha = gate_consistent_alpha(alpha)
    # Portion of the trainable optimizer state not yet captured by the resident
    # model-state term. Clamp at 0 so a fully-resident config adds nothing.
    trainable_topup = max(0, int(trainable_floor) - int(model_state_present))
    scaled = int(gate_alpha * (raw_peak + trainable_topup))
    return scaled


def estimate_peak(
    cfg: CostConfig,
    trace: ProfilerTrace,
    layout: ChunkLayout,
    block_map: BlockStrategyMap,
    hw: HardwareProfile,
) -> int:
    """Estimate steady-state peak GPU memory in bytes."""
    # Delegated so searcher inline fast-path and this validator share Eq. 11 accounting.
    model_state_present = model_state_present_bytes(cfg, layout, trace)

    # Persisted traces stringify keys; normalize once for all BlockId lookups.
    activation_sizes = _normalized_activation_sizes(trace)

    forward_ops_by_block = _group_ops_by_block(trace)
    tree_index_map = block_tree_index_map(trace)
    cross_attn_bytes = cross_attn_persist_bytes(trace, block_map, tree_index_map)
    # Block-internal saved tensors only; the block-input residual lives in ``ckpt_chain_bytes``.
    saved_bytes_proxy_for_op_walk = _saved_tensor_bytes_per_block(trace)

    ckpt_bump_op: dict[int, int] = {}
    # OFFLOAD bump fires at the last forward op (closest to the block's backward window).
    offload_bump_op: dict[int, int] = {}
    for block_id, op_idxs in forward_ops_by_block.items():
        if not op_idxs:
            continue
        mode = block_map.get(block_id, BlockMode.NONE)
        if mode is BlockMode.CKPT:
            ckpt_bump_op[op_idxs[0]] = int(block_id)
        elif mode is BlockMode.OFFLOAD:
            offload_bump_op[op_idxs[-1]] = int(block_id)

    retained_none_bytes = 0
    # CKPT chain term factored to ``_compute_ckpt_chain_bytes`` so the wrapper-side
    # ``_reconstruct_f_bm`` shares the same chain-sum semantics.
    ckpt_chain_bytes = _compute_ckpt_chain_bytes(trace, block_map)
    for bid, act_sz in activation_sizes.items():
        mode = block_map.get(bid, BlockMode.NONE)
        if mode is BlockMode.NONE or mode is BlockMode.OFFLOAD:
            # A NONE/OFFLOAD block recomputes nothing, so ALL its forward
            # saved-for-backward tensors (block input/output + FFN
            # intermediate + attention working set) stay resident across the
            # whole backward window. ``activation_sizes[bid]`` proxies only the
            # block-output boundary tensor; the saved-tensor proxy captures the
            # full per-block residency. Mirrors the calibrated gate's
            # ``_reconstruct_f_bm`` live-NONE term — using the block output
            # alone under-counts ~Σ(internal saved bytes) and let the searcher
            # accept n_checkpoint=0 configs the gate then fail-closes at long seq.
            retained_none_bytes += int(saved_bytes_proxy_for_op_walk.get(bid, act_sz))
        # SWAP: live only during the block's forward compute; assumed
        #       to overlap free GPU memory (§3.3). The CKPT-chain term
        #       does NOT apply because SWAP evicts the block-output
        #       tensor to the pinned-CPU swap pool (see swap_pool.py).

    # --- Op walk -------------------------------------------------------
    raw_peak = 0
    # Track activations that are "live as of op i". We build this
    # incrementally so ops inside a NONE block see that block's
    # activation bytes accumulate progressively (safer upper bound even
    # though the end-of-fwd sum already accounts for all of it). The
    # simplest correct accounting is:
    #
    #   live_at_op = retained_none_bytes_accumulated_up_to_block(op)
    #              + ckpt_bump_if_this_op_triggers
    #
    # We pre-compute the cumulative "NONE activations active by this
    # point in forward" by walking blocks in order.

    # Map op index -> cumulative NONE-activation bytes active at or
    # before this op. Blocks without a position in forward_ops_by_block
    # contribute no ordering, so we sort blocks by their first forward
    # op index.
    block_first_op = {bid: ops[0] for bid, ops in forward_ops_by_block.items() if ops}
    blocks_in_fwd_order = sorted(block_first_op.items(), key=lambda kv: kv[1])

    cumulative_none: list[tuple[int, int]] = []  # (first_op_idx, cumulative_bytes)
    running = 0
    for bid, first_idx in blocks_in_fwd_order:
        mode = block_map.get(bid, BlockMode.NONE)
        if mode is BlockMode.NONE or mode is BlockMode.OFFLOAD:
            # OFFLOAD retains forward activations on GPU (§3.3 lifecycle
            # table — "Forward activations: retained on GPU"). They join
            # the NONE running total so the live_none-at-op-i view sees
            # the same bytes as a NONE block would; the backward-window
            # chunk gather bump is a separate per-op bump landed via
            # ``offload_bump_op`` below. Use the saved-tensor proxy (full
            # per-block forward residency), not the block-output-only
            # ``activation_sizes``, so the live-NONE set matches the gate.
            running += int(
                saved_bytes_proxy_for_op_walk.get(bid, activation_sizes.get(bid, 0))
            )
        cumulative_none.append((first_idx, running))

    def _none_live_at(op_idx: int) -> int:
        """Cumulative NONE-block activation bytes at or before op_idx."""
        # Linear scan is fine; cumulative_none has at most N_block
        # entries (8-256 in realistic workloads).
        live = 0
        for first_idx, cum in cumulative_none:
            if first_idx <= op_idx:
                live = cum
            else:
                break
        return live

    for i, op in enumerate(trace.op_order):
        if not op.is_forward:
            continue

        intra = trace.intra_op_delta.get(op.op_id, 0)
        inter = trace.inter_op_delta.get(op.op_id, 0)
        live_none = _none_live_at(i)

        # CKPT recompute bump = internal saved-tensor delta; block-input residual already in ``ckpt_chain_bytes``.
        ckpt_extra = 0
        if i in ckpt_bump_op:
            bid = BlockId(ckpt_bump_op[i])
            block_act = activation_sizes.get(bid, 0)
            block_saved = int(saved_bytes_proxy_for_op_walk.get(bid, block_act))
            ckpt_extra = max(0, block_saved - block_act)

        # OFFLOAD bump = S_chunk gather only; activations already in live_none.
        offload_extra = 0
        if i in offload_bump_op:
            offload_extra = layout.S_chunk

        op_cross_attn = op_cross_attn_surcharge(op, cross_attn_bytes, tree_index_map)

        candidate = (
            model_state_present
            + live_none
            + ckpt_chain_bytes
            + ckpt_extra
            + offload_extra
            + op_cross_attn
            + intra
            + inter
        )
        if candidate > raw_peak:
            raw_peak = candidate

    # Degenerate trace (no forward ops): static estimate. ckpt_chain_bytes and retained_none_bytes are disjoint by construction so summing both does not double-count.
    if raw_peak == 0:
        raw_peak = model_state_present + retained_none_bytes + ckpt_chain_bytes

    # apply_hot_iter_cap bounds only the activation portion; preserves model-state through cap.
    measured_cap = hot_iter_peak_cap(trace, block_map, cfg, layout)
    raw_peak = apply_hot_iter_cap(raw_peak, model_state_present, measured_cap, layout)

    alpha = alpha_fragmentation_for_cfg(
        hw.dominant_param_bytes_per_element,
        cfg,
        bool(getattr(hw, "zero3_shard", False)),
    )
    trainable_floor = trainable_state_floor_bytes(trace)
    scaled = apply_gate_consistent_scaling(
        raw_peak, model_state_present, trainable_floor, alpha
    )
    if LOG.isEnabledFor(logging.DEBUG):
        LOG.debug(
            "estimate_peak: n_persist=%d n_buffer=%d n_swap=%d n_ckpt=%d n_offload=%d "
            "raw=%dB alpha=%.2f gate_alpha=%.2f trainable_floor=%dB -> %dB",
            cfg.n_persist,
            cfg.n_buffer,
            cfg.n_swap,
            cfg.n_checkpoint,
            cfg.n_offload,
            raw_peak,
            alpha,
            gate_consistent_alpha(alpha),
            trainable_floor,
            scaled,
        )
    return scaled


__all__ = [
    "ALPHA_FRAGMENTATION",
    "ALPHA_FRAGMENTATION_4BIT",
    "ALPHA_FRAGMENTATION_4BIT_MODE_A",
    "ALPHA_FRAGMENTATION_4BIT_MODE_C_CKPT",
    "DEFAULT_CKPT_INTERNAL_RESIDUAL_FACTOR",
    "alpha_fragmentation_for_cfg",
    "alpha_fragmentation_for_dtype",
    "apply_gate_consistent_scaling",
    "attn_activation_bytes",
    "gate_consistent_alpha",
    "_block_internal_saved_bytes",
    "_compute_ckpt_chain_bytes",
    "_saved_tensor_bytes_per_block",
    "block_tree_index_map",
    "cross_attn_handoff_bytes",
    "cross_attn_persist_bytes",
    "estimate_cpu_footprint",
    "estimate_peak",
    "hot_iter_peak_cap",
    "model_state_present_bytes",
    "op_cross_attn_surcharge",
    "set_default_ckpt_internal_residual_factor",
    "trainable_state_floor_bytes",
]
