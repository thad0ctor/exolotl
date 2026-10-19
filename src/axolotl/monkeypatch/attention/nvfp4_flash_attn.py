"""End-to-end native-NVFP4 attention for Qwen3.5 full-attention layers.

Replaces the full softmax-attention forward with a fused-producer NVFP4 pipeline:

  q_proj/k_proj/v_proj (+ q_norm/k_norm, unchanged bf16)
    -> fused RoPE -> NVFP4 pack of Q,K   (one pass, no standalone quant round-trip)
    -> v_proj output -> NVFP4 pack of V along the key axis
    -> packed-input native-NVFP4 flash forward (tl.dot_scaled, e2m1 + e4m3/group-16)
    -> attn_output * sigmoid(gate) -> o_proj

The fused producers eliminate the standalone Q/K pre-quant round-trips (the trap
that made the materialized-attention pipeline lose to bf16): the pack rides along
with the RoPE pass Q/K already pay. V can optionally be produced by a native-NVFP4
v_proj GEMM with a key-axis pack epilogue.

Qwen3.5 is HYBRID — only ``full_attention`` layers (head_dim 256 softmax GQA) are
patched; ``linear_attention`` layers are left untouched. The fast fused-producer
path is inference/forward-only; an explicitly enabled training path uses
``nvfp4_flash_attn_func`` with native NVFP4 backward. Opt-in: call
``patch_qwen3_5_nvfp4_attention(model)``; default OFF.

Per-key padding / non-causal-with-mask batches fall back to the model's original
forward (correctness over speed). The fast path serves dense causal prefill and
single-step / full decode.
"""

from __future__ import annotations

import os

import torch
from torch import nn

from axolotl.kernels.attn_nvfp4_flash import (
    _varlen_seq_arrays,
    nvfp4_flash_attention,
    nvfp4_flash_attention_packed,
    nvfp4_flash_attn_func,
)
from axolotl.kernels.nvfp4_fused_producers import (
    fused_rope_quant_qk,
    fused_rope_quant_qk_hp,
    fused_vproj_quant_v_keyaxis,
    prepack_vproj_weight,
    quant_v_keyaxis,
)
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

_BLOCK_N = 128

# Fuse the V NVFP4-quant into a native-NVFP4 v_proj GEMM epilogue (no bf16 V, no
# transpose copy, no standalone quant read). Makes v_proj itself FP4, so it trades a
# little parity (attn-op cos ~0.979 -> ~0.967) for ~1.4-1.5x on the V producer and
# +0.1-0.2x end-to-end (v_proj counted). Real-model logit cos stays >=0.998. OFF by
# default; flip via patch_qwen3_5_nvfp4_attention(model, fuse_vproj=True).
_FUSE_VPROJ = False
_CUSTOM_OP_ENV = "AXOLOTL_NVFP4_QWEN35_ATTENTION_CUSTOM_OP"

# NOTE (graph-break isolation REMOVED): the compiled training custom-op paths
# below used to be bracketed by torch._dynamo.graph_break() pairs — an old
# guard against an Inductor miscompile (param grads ~1e5 -> NaN) observed on
# an earlier torch/patch state, before the attention became an opaque
# differentiable custom op. A break anywhere inside the decoder-layer body
# makes dynamo skip the whole ``Qwen3_5TextModel.forward`` loop frame, so the
# decoder loop could never compile with the breaks in. The guard was verified
# STALE on torch 2.12/triton 3.7: (1) tiny-hybrid probes are BITWISE identical
# eager-vs-compiled under backend="aot_eager" and under Inductor with
# triton.codegen_upcast_to_fp32 disabled (the only compiled-vs-eager delta is
# Inductor keeping fused bf16 chains in fp32 — a precision difference, not a
# miscompile; FP4 quantizer bin flips amplify it on degenerate toy losses);
# (2) an 8-step 9B real-config A/B (32k long-doc packed LoRA) shows identical
# per-step losses and grad_norms within ~10% with and without the breaks.
# Packed-batch auto-gate default (see nvfp4_training.attention.packed_min_sample_len):
# 0 = gate OFF. With the fused RoPE+quant producers feeding the training forward
# pre-packed q/k (q_packs/k_packs) plus the varlen-tuned forward tiles, FP4 varlen
# beats FA2-varlen at every packed mean sample length measured (packed 8192, RTX
# PRO 6000: op-level mean-256 grad fwd 1.22x / fwd+bwd 1.22x at d256 h16/4,
# 1.51x / 1.32x at d128 h32/8; >=1.0x at means 128-2048), so short packs no
# longer need re-routing to the original forward.
_PACKED_MIN_SAMPLE_LEN_DEFAULT = 0

# Whether the installed sageattention fork's packed-operand forward accepts
# cu_seqlens (varlen): lets the NO-GRAD packed path reuse the fused producers
# (RoPE+quant one pass) instead of materializing roped Q/K and re-reading them
# in the kernel's own pre-quant. Detected once at import.
try:
    import inspect as _inspect

    _PACKED_FWD_VARLEN = (
        "cu_seqlens"
        in _inspect.signature(nvfp4_flash_attention_packed).parameters
    )
except (TypeError, ValueError):  # pragma: no cover - exotic callables
    _PACKED_FWD_VARLEN = False

# Whether the fork's TRAINING entry (nvfp4_flash_attn_func) accepts externally
# produced forward q/k packs (q_packs/k_packs): lets the packed-grad path use the
# fused RoPE+quant producers (one pass emits roped hp q/k for the backward AND
# the forward packs), eliminating the HF rope round-trip plus the standalone
# q/k forward pre-quant launches. Detected once at import.
try:
    _TRAIN_PREPACKED = (
        "q_packs" in _inspect.signature(nvfp4_flash_attn_func).parameters
    )
except (TypeError, ValueError):  # pragma: no cover - exotic callables
    _TRAIN_PREPACKED = False

# Whether the fork's TRAINING entry accepts the string backward_grad_dots mode
# ("bf16" / "fp4_rownorm" / "fp8_rownorm" / "fp4_legacy" / None=AUTO). Older
# hp-grad-dots forks only know the boolean backward_bf16_grad_dots alias.
try:
    _TRAIN_GRAD_DOTS = (
        "backward_grad_dots" in _inspect.signature(nvfp4_flash_attn_func).parameters
    )
except (TypeError, ValueError):  # pragma: no cover - exotic callables
    _TRAIN_GRAD_DOTS = False

# Whether the fork's TRAINING entry can write the HF attn_output layout
# directly (out_layout="zshd") and take the zshd gradient back without a
# transpose-fold copy. Eliminates one [Z,S,H*D]-sized copy in every
# checkpoint-replay forward AND one in every backward at long context.
try:
    _TRAIN_OUT_LAYOUT = (
        "out_layout" in _inspect.signature(nvfp4_flash_attn_func).parameters
    )
except (TypeError, ValueError):  # pragma: no cover - exotic callables
    _TRAIN_OUT_LAYOUT = False

# Fused RMSNorm+RoPE (the same compile-safe triton_op the fused_attn_kernel
# baseline uses for Qwen3.5). The patch previously re-EAGERIZED q_norm/k_norm
# (HF RMSNorm: fp32 cast + pow + mean + rsqrt + muls per call) on every
# checkpointed pass — measured as the dominant share of the patch's step
# overhead vs the fused-attention baseline at 32k (the attention KERNELS beat
# FA2 while the patch glue burned the win). With the fused op, norm+partial
# RoPE is one kernel per operand and the forward NVFP4 packs are produced from
# its output by the standalone strided quant (one extra read of roped Q/K, no
# eager norm chain).
try:
    from axolotl.kernels.gemma4_fused_rope import fused_rms_norm_rope

    _FUSED_NORM_ROPE = True
except ImportError:  # pragma: no cover - missing optional kernel dep
    _FUSED_NORM_ROPE = False


def _norm_eps(norm) -> float:
    eps = getattr(norm, "eps", None)
    if eps is None:
        eps = norm.variance_epsilon
    return eps


def _fused_norm_rope_ok(module: nn.Module) -> bool:
    """Once-per-module capability check for the fused RMSNorm+RoPE path."""
    cached = getattr(module, "_nvfp4_fnr_ok", None)
    if cached is not None:
        return cached
    ok = _FUSED_NORM_ROPE
    if ok:
        from axolotl.monkeypatch.models.qwen3_5.fused_attn import (
            _resolve_norm_module,
        )

        for norm in (module.q_norm, module.k_norm):
            n = _resolve_norm_module(norm)
            if getattr(n, "weight", None) is None:
                ok = False
                break
            try:
                _norm_eps(n)
            except AttributeError:
                ok = False
                break
    module._nvfp4_fnr_ok = ok
    return ok


def _fused_norm_rope(norm, x, cos, sin):
    """RMSNorm (unit-offset, Qwen3.5) + partial RoPE on ``x`` [Z, S, H, D]."""
    from axolotl.monkeypatch.models.qwen3_5.fused_attn import _resolve_norm_module

    n = _resolve_norm_module(norm)
    w = n.weight
    if w.device != x.device:  # accelerate CPU-staged params
        w = w.to(x.device, non_blocking=True)
    return fused_rms_norm_rope(x, w, cos, sin, eps=_norm_eps(n), unit_offset=True)


def _quant_packs_along_d(x: torch.Tensor):
    """NVFP4-pack ``x`` [Z, H, S, D] along D (the kernel q_packs/k_packs layout).

    The strided quant reads head-major views (e.g. ``transpose(1, 2)`` of a
    zshd fused-norm-rope output) without a fold copy when Z == 1 (reshape is a
    view); reshape handles the copy itself otherwise.
    """
    from axolotl.kernels.attn_nvfp4_flash import _quant_nvfp4

    z, h, s, d = x.shape
    return _quant_nvfp4(x.reshape(z * h, s, d))


def _grad_dots_kwargs(module) -> dict:
    """Backward grad-GEMM mode kwarg for the eager training entry.

    None (default) = kernel AUTO ("bf16" below an effective sequence of 4096 —
    the mean sample length for packed/varlen batches — else "fp4_rownorm"). On
    a legacy fork without the string modes the configured mode degrades to the
    boolean hp/legacy alias ("bf16" -> hp backward, rownorm/legacy -> legacy
    all-FP4 backward, None stays auto)."""
    mode = getattr(module, "_nvfp4_grad_dots", None)
    if _TRAIN_GRAD_DOTS:
        return {"backward_grad_dots": mode}
    return {"backward_bf16_grad_dots": None if mode is None else mode == "bf16"}


_GRAD_DOTS_MODES = ("bf16", "fp4_rownorm", "fp8_rownorm", "fp4_legacy")


def _resolve_grad_dots_alias(
    grad_dots: str | None, bf16_grad_dots: bool | None
) -> str | None:
    """Fold the DEPRECATED ``bf16_grad_dots`` boolean alias into ``grad_dots``
    (True -> "bf16", False -> "fp4_legacy"; both set -> error) and validate."""
    if bf16_grad_dots is not None:
        if grad_dots is not None:
            raise ValueError(
                "pass either grad_dots or the deprecated bf16_grad_dots alias, "
                "not both"
            )
        grad_dots = "bf16" if bf16_grad_dots else "fp4_legacy"
    if grad_dots is not None and grad_dots not in _GRAD_DOTS_MODES:
        raise ValueError(
            f"grad_dots must be one of {_GRAD_DOTS_MODES} or None (kernel "
            f"auto), got {grad_dots!r}"
        )
    return grad_dots


def _custom_op_enabled(module: nn.Module) -> bool:
    requested = getattr(module, "_nvfp4_compile_custom_op", False) or os.environ.get(
        _CUSTOM_OP_ENV, ""
    ).lower() in {"1", "true", "yes"}
    return bool(requested) and torch.compiler.is_compiling()


_DENSE_KIND_ATTR = "_nvfp4_dense_kind"
_TRIL_CACHE: dict[tuple, torch.Tensor] = {}


def _causal_tril(q_len: int, kv_len: int, device: torch.device) -> torch.Tensor:
    """Lower-triangular keep mask, cached by (shape, device) to avoid rebuilding the
    O(S^2) tensor on every classification."""
    key = (q_len, kv_len, device)
    t = _TRIL_CACHE.get(key)
    if t is None:
        t = torch.tril(torch.ones(q_len, kv_len, dtype=torch.bool, device=device))
        _TRIL_CACHE[key] = t
    return t


def _classify_dense_mask(
    attention_mask: torch.Tensor, q_len: int, kv_len: int
) -> str | None:
    m = attention_mask[..., :kv_len]
    if m.shape[-1] != kv_len:
        return None
    if m.dtype == torch.bool:
        # Boolean keep-mask (e.g. SDPA's create_causal_mask output).
        keep = m
    else:
        neg = torch.finfo(m.dtype).min
        keep = m > neg / 2
    if keep.shape[1] != 1:
        if not bool((keep == keep[:, :1]).all()):
            return None
        keep = keep[:, :1]
    keep = keep[:, 0]
    if bool(keep.all()):
        return "full"
    if q_len == kv_len:
        if bool((keep == _causal_tril(q_len, kv_len, keep.device)[None]).all()):
            return "causal"
    return None


def _mask_is_dense_causal_or_full(
    attention_mask: torch.Tensor | None, q_len: int, kv_len: int
) -> str | None:
    """Return 'causal' / 'full' if the mask is a dense causal/full mask, else None.

    None -> there is per-key padding (or a mask shape we can't densely represent),
    so the caller must fall back to the original forward.

    The dense 4D classification needs ``bool(...)`` reductions (each a device->host
    sync) plus an O(Z*S^2) full-mask compare. The mask tensor is shared across all
    decoder layers in a forward, so the result is cached on it (keyed by shape and
    ``_version``) — computed once per model-forward, not once per attention layer.
    """
    if attention_mask is None:
        return "causal" if q_len == kv_len else "full"
    if attention_mask.dim() != 4:
        return None
    ckey = (q_len, kv_len, attention_mask._version)
    cached = getattr(attention_mask, _DENSE_KIND_ATTR, None)
    if cached is not None and cached[0] == ckey:
        return cached[1]
    kind = _classify_dense_mask(attention_mask, q_len, kv_len)
    try:
        setattr(attention_mask, _DENSE_KIND_ATTR, (ckey, kind))
    except (AttributeError, RuntimeError):
        pass  # some tensor subclasses disallow setattr; correctness is unaffected
    return kind


# ---------------------------------------------------------------------------
# Packed-sequence (sample-packing / multipack) detection.
#
# Under axolotl multipack the batch arrives FLATTENED to one row (Z=1) with
# attention_mask=None and the sample boundaries encoded in position_ids (each
# sample's positions restart at 0). Without this detection the mask classifier
# returns "causal" and the FP4 kernel dense-attends the whole pack — orders of
# magnitude more work than varlen AND cross-sample attention contamination. The
# varlen kernel (cu_seqlens) runs block-diagonal causal attention instead.
#
# The cu_seqlens derivation needs host syncs (a .sum() reduction + nonzero), so
# like the mask classification it is cached ON the position_ids tensor itself
# (keyed by shape and ``_version``): the decoder layers all receive the SAME
# position_ids tensor object within one model forward, so this is computed once
# per step, not once per attention layer.
# ---------------------------------------------------------------------------
_PACKED_INFO_ATTR = "_nvfp4_packed_info"
_PACKED_FALLBACK_LOGGED = False
_PACKED_SAVE_PACKS_LOGGED = False
_PACKED_GATE_LOGGED = False


def _compute_packed_info(
    position_ids: torch.Tensor, q_len: int
) -> tuple[str | None, torch.Tensor | None, float | None]:
    """Classify position_ids: (None, None, None) -> not packed (plain dense batch);
    ("packed", cu_seqlens, mean_sample_len) -> Z=1 flattened pack the varlen
    kernel can serve (mean_sample_len = q_len / n_samples feeds the perf gate);
    ("fallback", None, None) -> packed but unsupported (e.g. Z>1) — the caller
    MUST fall back to the original forward (never dense-attend a packed batch)."""
    pos = position_ids[0] if position_ids.ndim == 3 else position_ids
    if pos.ndim == 1:
        pos = pos.unsqueeze(0)
    if pos.ndim != 2 or pos.shape[-1] != q_len:
        return (None, None, None)
    b = pos.shape[0]
    n_starts = int((pos == 0).sum())
    if n_starts <= b:
        return (None, None, None)  # at most one sample per row -> dense causal
    if b != 1 or not bool(pos[0, 0] == 0) or _varlen_seq_arrays is None:
        # Packed, but not the Z=1 flattened layout the varlen kernel requires
        # (or the installed sageattention fork predates varlen support).
        return ("fallback", None, None)
    from axolotl.monkeypatch.models.qwen3_5.modeling import get_cu_seqlens

    cu_seqlens = get_cu_seqlens(position_ids)
    # Mean sample length of the pack — pure tensor-metadata arithmetic (numel),
    # no device sync; cached with cu_seqlens so the per-step gate decision costs
    # nothing per layer.
    mean_len = q_len / max(int(cu_seqlens.numel()) - 1, 1)
    return ("packed", cu_seqlens, mean_len)


def _packed_position_ids_info(
    position_ids: torch.Tensor | None,
    q_len: int,
    compute=None,
    cache_attr: str = _PACKED_INFO_ATTR,
) -> tuple[str | None, torch.Tensor | None, float | None]:
    """Cached wrapper around ``_compute_packed_info`` (see block comment above).

    ``compute`` / ``cache_attr`` let model-specific patches (the Qwen3-VL one)
    plug in their own classification while reusing the once-per-step caching
    (``compute=None`` resolves to ``_compute_packed_info`` at call time so test
    monkeypatching of the module attribute keeps working).
    """
    if position_ids is None:
        return (None, None, None)
    if compute is None:
        compute = _compute_packed_info
    ckey = (q_len, tuple(position_ids.shape), position_ids._version)
    cached = getattr(position_ids, cache_attr, None)
    if cached is not None and cached[0] == ckey:
        return cached[1]
    info = compute(position_ids, q_len)
    try:
        setattr(position_ids, cache_attr, (ckey, info))
    except (AttributeError, RuntimeError):
        pass  # some tensor subclasses disallow setattr; correctness is unaffected
    return info


def _log_packed_fallback_once() -> None:
    global _PACKED_FALLBACK_LOGGED
    if not _PACKED_FALLBACK_LOGGED:
        _PACKED_FALLBACK_LOGGED = True
        LOG.warning(
            "nvfp4 attention: packed batch in an unsupported layout (multi-row "
            "batch or legacy sageattention fork without varlen); falling back to "
            "the model's original attention for packed batches"
        )


def _log_packed_gate_once(mean_len: float, min_len: int, use_fp4: bool) -> None:
    """One-time INFO stating which way the packed perf gate resolved."""
    global _PACKED_GATE_LOGGED
    if _PACKED_GATE_LOGGED:
        return
    _PACKED_GATE_LOGGED = True
    if use_fp4:
        LOG.info(
            "nvfp4 attention: packed-batch gate -> FP4 varlen attention "
            "(measured mean sample length %.1f >= packed_min_sample_len %d)",
            mean_len,
            min_len,
        )
    else:
        LOG.info(
            "nvfp4 attention: packed-batch gate -> original attention "
            "(measured mean sample length %.1f < packed_min_sample_len %d; "
            "set nvfp4_training.attention.packed_min_sample_len: 0 to force "
            "FP4 — with the fused-producer pre-packed training forward, FP4 "
            "varlen wins at all measured packed means, so a positive gate is "
            "mainly for legacy sageattention forks)",
            mean_len,
            min_len,
        )


def _log_packed_save_packs_ignored_once() -> None:
    global _PACKED_SAVE_PACKS_LOGGED
    if not _PACKED_SAVE_PACKS_LOGGED:
        _PACKED_SAVE_PACKS_LOGGED = True
        LOG.info(
            "nvfp4 attention: packed (varlen) batches are incompatible with the "
            "saved forward FP4 pack set; ignoring attention.backward.save_packs "
            "for packed batches (config still applies to dense batches; the "
            "configured grad_dots mode applies to packed batches too — every "
            "mode composes with varlen)"
        )


# ---------------------------------------------------------------------------
# Compile-safe packed-info hoist.
#
# ``_compute_packed_info`` needs data-dependent host reads (a ``.sum()`` device
# sync, ``aten.nonzero``) which graph-break under dynamo — and a single break
# inside the transformers decoder LOOP makes dynamo skip the whole
# ``Qwen3_5TextModel.forward`` frame (every layer runs eagerly). The
# classification only depends on position_ids CONTENT once per batch, so it is
# hoisted OUT of the per-layer forward: a dynamo-disabled pre-forward hook on
# the patched model resolves the routing ONCE per step (eagerly, before/around
# the compiled region) and threads the result to the attention layers as plain
# module attributes — a host string constant dynamo guards on plus a cu_seqlens
# tensor it lifts as a graph input. Inside the compiled loop there is no
# nonzero/item left; the custom op consumes cu_seqlens as a normal tensor.
#
# The hook resolves the FULL routing (packed/dense classification, the
# unsupported-layout fallback AND the packed_min_sample_len perf gate) so the
# compiled forward never touches ``mean_len`` (a per-batch float would
# recompile on every distinct value). Eager forwards keep the original
# in-forward classification (same math, same per-tensor cache the hook warms),
# so eager behavior is bit-identical with or without the hook.
# ---------------------------------------------------------------------------
_STEP_KIND_ATTR = "_nvfp4_step_kind"
_STEP_CU_ATTR = "_nvfp4_step_cu"
_STEP_MODULES_ATTR = "_nvfp4_attn_step_modules"


def _resolve_step_info(
    position_ids: torch.Tensor | None, q_len: int | None, min_len: int
) -> tuple[str, torch.Tensor | None]:
    """Once-per-step packed routing decision (host-side, eager).

    Returns ``("packed", cu_seqlens)`` (serve the FP4 varlen kernel),
    ``("dense", None)`` (plain dense causal/full routing) or ``("orig", None)``
    (the model's ORIGINAL forward must serve this batch: unsupported packed
    layout, or a positive perf gate routed the short-mean pack away).
    """
    if position_ids is None or q_len is None:
        return ("dense", None)
    pkind, cu_seqlens, mean_len = _packed_position_ids_info(position_ids, q_len)
    if pkind == "packed":
        if min_len > 0 and mean_len < min_len:
            _log_packed_gate_once(mean_len, min_len, use_fp4=False)
            return ("orig", None)
        _log_packed_gate_once(mean_len, min_len, use_fp4=True)
        return ("packed", cu_seqlens)
    if pkind == "fallback":
        _log_packed_fallback_once()
        return ("orig", None)
    return ("dense", None)


def _hook_q_len(args: tuple, kwargs: dict) -> int | None:
    """Best-effort sequence length of the incoming batch at the hook level."""
    for key in ("input_ids", "inputs_embeds", "hidden_states"):
        t = kwargs.get(key)
        if isinstance(t, torch.Tensor) and t.ndim >= 2:
            return t.shape[1]
    for a in args:
        if isinstance(a, torch.Tensor) and a.ndim >= 2:
            return a.shape[1]
    return None


def _step_info_pre_hook(model: nn.Module, args: tuple, kwargs: dict) -> None:
    modules = getattr(model, _STEP_MODULES_ATTR, ())
    if not modules:
        return
    position_ids = kwargs.get("position_ids")
    q_len = _hook_q_len(args, kwargs)
    if q_len is None and position_ids is not None:
        q_len = position_ids.shape[-1]
    min_len = getattr(
        modules[0], "_nvfp4_packed_min_sample_len", _PACKED_MIN_SAMPLE_LEN_DEFAULT
    )
    kind, cu_seqlens = _resolve_step_info(position_ids, q_len, min_len)
    for module in modules:
        setattr(module, _STEP_KIND_ATTR, kind)
        setattr(module, _STEP_CU_ATTR, cu_seqlens)


try:
    import torch._dynamo as _attn_dynamo

    # The hook body is data-dependent BY DESIGN (it is the hoist target); keep
    # dynamo out of it. When the patched model is compiled, the disabled call
    # splits the outer ``__call__`` frame once — the decoder LOOP frame below
    # it traces with zero breaks.
    _step_info_pre_hook_disabled = _attn_dynamo.disable(_step_info_pre_hook)
except Exception:  # pragma: no cover - dynamo always available with torch>=2
    _step_info_pre_hook_disabled = _step_info_pre_hook


def _install_step_info_hook(model: nn.Module) -> None:
    """Register the once-per-step packed-info hook on the patched model root."""
    modules = [m for m in model.modules() if getattr(m, "_nvfp4_patched", False)]
    if not modules:
        return
    setattr(model, _STEP_MODULES_ATTR, modules)
    if getattr(model, "_nvfp4_step_info_hooked", False):
        return
    model.register_forward_pre_hook(_step_info_pre_hook_disabled, with_kwargs=True)
    model._nvfp4_step_info_hooked = True


def _get_vproj_packed(module: nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
    """Lazily prepack v_proj and refresh if an optimizer updated the weight."""
    if type(module.v_proj) is not nn.Linear:
        raise TypeError("NVFP4 fused v_proj requires a plain nn.Linear v_proj")
    weight = module.v_proj.weight
    key = (weight.data_ptr(), weight._version, tuple(weight.shape), weight.dtype)
    cached = getattr(module, "_nvfp4_vproj_packed", None)
    if cached is None or cached[0] != key:
        wnv, wsc = prepack_vproj_weight(weight.detach())
        module._nvfp4_vproj_packed = (key, wnv, wsc)
        cached = module._nvfp4_vproj_packed
    return cached[1], cached[2]


def _can_fuse_vproj(module: nn.Module) -> bool:
    return type(module.v_proj) is nn.Linear


def _nvfp4_attention(
    module: nn.Module,
    query_states: torch.Tensor,  # [Z, H, S, D]  post q_norm (PRE-RoPE, or
    key_states: torch.Tensor,  # [Z, Hk, S, D]   already roped when roped=True)
    value_states: torch.Tensor,  # [Z, Hk, Skv, D]
    cos: torch.Tensor,  # [Z, S, rotary_dim]
    sin: torch.Tensor,
    scaling: float,
    causal: bool,
    hidden_states: torch.Tensor | None = None,  # [Z, S, hidden], for fused v_proj
    roped: bool = False,
) -> torch.Tensor:
    """Fused-producer NVFP4 attention. Returns ``[Z, S, H, D]`` (HF attn_output).

    ``roped=True``: q/k arrive already normed+roped (fused norm+rope path) and
    are packed by the standalone strided quant; otherwise the fused RoPE+quant
    producers rope them here.
    """
    z, h, s_q, d = query_states.shape
    hk = key_states.shape[1]
    s_kv = key_states.shape[2]

    if roped:
        qnv, qsc = _quant_packs_along_d(query_states)
        knv, ksc = _quant_packs_along_d(key_states)
    else:
        qnv, qsc = fused_rope_quant_qk(query_states, cos, sin)
        knv, ksc = fused_rope_quant_qk(key_states, cos, sin)
    if hidden_states is not None:
        # fuse the V quant into a native-NVFP4 v_proj GEMM epilogue (no bf16 V)
        wnv, wsc = _get_vproj_packed(module)
        vnv, vsc, _ = fused_vproj_quant_v_keyaxis(
            hidden_states, wnv, wsc, hk=hk, d=d, block_n=_BLOCK_N
        )
    else:
        vnv, vsc, _ = quant_v_keyaxis(value_states, block_n=_BLOCK_N)

    if _custom_op_enabled(module):
        from axolotl.kernels.attn_nvfp4_custom_op import (
            nvfp4_flash_attention_packed_custom_op,
        )

        # The compile custom op keeps the [Z, H, S, D] schema (its registered op /
        # fake are layout-fixed), so the transpose stays on the compiled path.
        out = nvfp4_flash_attention_packed_custom_op(
            qnv,
            qsc,
            knv,
            ksc,
            vnv,
            vsc,
            z=z,
            h=h,
            hk=hk,
            s_q=s_q,
            s_kv=s_kv,
            d=d,
            scaling=scaling,
            out_dtype=query_states.dtype,
            causal=causal,
            block_n=_BLOCK_N,
        )  # [Z, H, S, D]
        return out.transpose(1, 2)  # [Z, S, H, D]

    # Eager path: the kernel writes the [Z, S, H, D] HF attn_output layout directly,
    # so the per-layer transpose(1,2)+contiguous copy at the caller is eliminated.
    return nvfp4_flash_attention_packed(
        qnv,
        qsc,
        knv,
        ksc,
        vnv,
        vsc,
        z=z,
        h=h,
        hk=hk,
        s_q=s_q,
        s_kv=s_kv,
        d=d,
        scaling=scaling,
        out_dtype=query_states.dtype,
        causal=causal,
        block_n=_BLOCK_N,
        out_layout="zshd",
    )  # [Z, S, H, D]


# ---------------------------------------------------------------------------
# Shared-activation-pack FP4 for the full-attention block's hidden-consuming
# projections. q_proj (gated, -> 2*head_dim) and k_proj BOTH read hidden_states
# and contract the same K=hidden, so hidden is NVFP4-packed ONCE and reused for
# both FP4 GEMMs (the cross-op lever already used by the MLP gate/up and the GDN
# in_proj). o_proj (post-attention activation) gets its own FP4 GEMM. All
# parity-affecting (q/k/o go FP4) -> opt-in, default OFF, plain-nn.Linear only.
# ---------------------------------------------------------------------------
def _attn_proj_ok(module: nn.Module) -> bool:
    from axolotl.monkeypatch.attention.nvfp4_linear_attn import _is_plain_linear

    if not (
        _is_plain_linear(module.q_proj)
        and _is_plain_linear(module.k_proj)
        and _is_plain_linear(module.o_proj)
    ):
        return False
    return (
        module.q_proj.in_features % 16 == 0
        and module.k_proj.in_features % 16 == 0
        and module.o_proj.in_features % 16 == 0
    )


def _nvfp4_qk_proj(module, hidden_states):
    """FP4 q_proj + k_proj sharing ONE NVFP4 pack of hidden_states."""
    from axolotl.kernels.attn_nvfp4_flash import _quant_nvfp4
    from axolotl.monkeypatch.attention.nvfp4_linear_attn import (
        _gemm_from_packed_act,
        _get_packed_weight,
    )

    dtype = hidden_states.dtype
    k = hidden_states.shape[-1]
    lead = hidden_states.shape[:-1]
    x2d = hidden_states.reshape(-1, k)
    m = x2d.shape[0]
    anv, asc = _quant_nvfp4(x2d.unsqueeze(0))
    anv, asc = anv[0], asc[0]
    q_wnv, q_wsc = _get_packed_weight(module, "_qproj_packed", module.q_proj)
    k_wnv, k_wsc = _get_packed_weight(module, "_kproj_packed", module.k_proj)
    q_outf, k_outf = module.q_proj.out_features, module.k_proj.out_features
    q = _gemm_from_packed_act(anv, asc, q_wnv, q_wsc, m, q_outf, k, dtype)
    kk = _gemm_from_packed_act(anv, asc, k_wnv, k_wsc, m, k_outf, k, dtype)
    if module.q_proj.bias is not None:
        q = q + module.q_proj.bias
    if module.k_proj.bias is not None:
        kk = kk + module.k_proj.bias
    return q.reshape(*lead, q_outf), kk.reshape(*lead, k_outf)


def _nvfp4_o_proj(module, attn_output):
    """FP4 o_proj (+bias)."""
    from axolotl.kernels.nvfp4_linear import nvfp4_linear
    from axolotl.monkeypatch.attention.nvfp4_linear_attn import _get_packed_weight

    o_wnv, o_wsc = _get_packed_weight(module, "_oproj_packed", module.o_proj)
    out = nvfp4_linear(attn_output, o_wnv, o_wsc, module.o_proj.out_features)
    if module.o_proj.bias is not None:
        out = out + module.o_proj.bias
    return out


def _o_proj_fwd(module, attn_output):
    """o_proj through the fused LoRA kernel when attached (``apply_o``).

    The plain peft LoRA module forward round-trips the [Z, S, hidden] input
    bf16 -> fp32 (adapter dtype, ``_cast_input_dtype``) -> bf16 (autocast inside
    ``F.linear``) — two hidden-sized copies per call plus their ToCopyBackward0
    twins. ``apply_o`` (axolotl's LoRA kernel patch, same route the
    fused_attn_kernel baseline takes) keeps the input bf16 and casts only the
    skinny adapter weights. Falls back to the peft forward when unpatched.
    """
    apply_o = getattr(module, "apply_o", None)
    if apply_o is not None:
        return apply_o(attn_output)
    return module.o_proj(attn_output)


def make_nvfp4_forward(orig_forward):
    """Build a patched ``Qwen3_5Attention.forward`` that uses NVFP4 attention.

    Mirrors the stock forward (chunk q/gate, q_norm/k_norm, RoPE, attention, gate,
    o_proj). In no-grad mode it routes through the fused-producer NVFP4 path. If
    ``train_backward`` was enabled at patch time, grad-enabled dense prefill uses
    the differentiable native-NVFP4 attention function.
    """

    def forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask=None,
        past_key_values=None,
        **kwargs,
    ):
        grad_enabled = torch.is_grad_enabled()
        if kwargs.get("output_attentions"):
            return orig_forward(
                self,
                hidden_states,
                position_embeddings,
                attention_mask,
                past_key_values,
                **kwargs,
            )

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        q_len = input_shape[-1]

        # NVFP4 fast path: prefill only (no prior cached context) on a dense
        # causal/full mask. Cached decode reuses prior roped K/V from the cache,
        # which the fused-RoPE producer can't reconstruct, so it goes to stock.
        has_cache_context = (
            past_key_values is not None
            and past_key_values.get_seq_length(self.layer_idx) > 0
        )
        kind = None
        cu_seqlens = None
        if not has_cache_context:
            kind = _mask_is_dense_causal_or_full(attention_mask, q_len, q_len)
            if kind == "causal" and attention_mask is None:
                # Sample-packed (multipack) batches arrive mask-less with the
                # boundaries in position_ids — the decoder layer (stock and the
                # axolotl packing patch alike) forwards position_ids in kwargs.
                #
                # COMPILED path: the packed/dense/fallback routing (incl. the
                # perf gate) was hoisted to the model-level pre-forward hook
                # (once per step, eager — see _install_step_info_hook). The
                # data-dependent classification never runs inside the traced
                # decoder loop: ``step_kind`` is a host string dynamo guards
                # on and cu_seqlens a plain tensor input.
                step_kind = (
                    getattr(self, _STEP_KIND_ATTR, None)
                    if torch.compiler.is_compiling()
                    else None
                )
                if step_kind == "packed":
                    cu_seqlens = getattr(self, _STEP_CU_ATTR, None)
                    # defensive: never dense-attend a packed batch
                    kind = "packed" if cu_seqlens is not None else None
                elif step_kind == "orig":
                    kind = None  # fall to the original forward at the bottom
                elif step_kind is None:  # eager: in-forward classification
                    pkind, cu_seqlens, mean_len = _packed_position_ids_info(
                        kwargs.get("position_ids"), q_len
                    )
                    if pkind == "packed":
                        # PERF GATE (default OFF, min_len=0): with the fused
                        # producers feeding the training forward pre-packed q/k
                        # and the varlen-tuned tiles, FP4 varlen beats
                        # FA2-varlen at every measured packed mean (op-level
                        # mean-256 grad fwd 1.22x / fwd+bwd 1.22x at d256). A
                        # positive min_len still routes short-mean packs to the
                        # original forward (legacy-fork escape hatch). The
                        # decision input (mean_len) is cached with cu_seqlens
                        # on the position_ids tensor: per step, not per layer,
                        # and it applies identically to the grad / no-grad /
                        # compiled sub-paths below (the routing happens here,
                        # before any of them).
                        min_len = getattr(
                            self,
                            "_nvfp4_packed_min_sample_len",
                            _PACKED_MIN_SAMPLE_LEN_DEFAULT,
                        )
                        if min_len > 0 and mean_len < min_len:
                            _log_packed_gate_once(mean_len, min_len, use_fp4=False)
                            return orig_forward(
                                self,
                                hidden_states,
                                position_embeddings,
                                attention_mask,
                                past_key_values,
                                **kwargs,
                            )
                        _log_packed_gate_once(mean_len, min_len, use_fp4=True)
                        kind = "packed"
                    elif pkind == "fallback":
                        # Packed but in a layout varlen can't serve: NEVER
                        # dense-attend it (cross-sample contamination) —
                        # original forward.
                        _log_packed_fallback_once()
                        kind = None
                # step_kind == "dense": keep kind == "causal"

        # Shared-pack FP4 q/k_proj: no-grad prefill, opt-in, plain-Linear only.
        use_fp4_proj = (
            not grad_enabled
            and kind is not None
            and getattr(self, "_nvfp4_fuse_attn_proj", False)
            and getattr(self, "_nvfp4_attn_proj_ok", False)
        )
        # q/k feed QK^T->softmax (error amplified through exp); o_proj is a direct
        # readout. Separable sub-gates let q/k stay bf16 while o_proj goes FP4 (the
        # near-lossless subset). Both default ON when the main flag is on.
        use_fp4_qk = use_fp4_proj and getattr(self, "_nvfp4_fp4_qk", True)
        use_fp4_o = use_fp4_proj and getattr(self, "_nvfp4_fp4_o", True)
        v_full = None
        if use_fp4_qk:
            q_full, k_full = _nvfp4_qk_proj(self, hidden_states)
        elif hasattr(self, "_lora_batch_kernel"):
            # Fused LoRA QKV (``apply_qkv`` — the same route the
            # fused_attn_kernel baseline takes; ``_lora_batch_kernel`` is the
            # sentinel apply_lora_kernel_patches sets iff the FUSED kernel was
            # installed, not the plain fallback). The peft module forwards
            # instead round-trip the [Z, S, hidden] input bf16 -> fp32 (adapter
            # dtype) -> bf16 (autocast) per projection — with their backward
            # ToCopyBackward0 twins that is ~24 hidden-sized copies per
            # attention layer per checkpointed training step at 32k. matmul_lora
            # runs the SAME NVFP4 base GEMM (module threaded through the
            # W_quant slot) and the adapter GEMMs in the input dtype.
            q_full, k_full, v_full = self.apply_qkv(hidden_states)
        else:
            q_full = self.q_proj(hidden_states)
            k_full = self.k_proj(hidden_states)
        query_states, gate = torch.chunk(
            q_full.view(*input_shape, -1, self.head_dim * 2),
            2,
            dim=-1,
        )
        gate = gate.reshape(*input_shape, -1)

        def _v_states():
            v = v_full if v_full is not None else self.v_proj(hidden_states)
            return v.view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        key_pre = k_full.view(hidden_shape)
        # Fused RMSNorm + partial RoPE (one triton_op per operand — the same
        # kernel the fused_attn_kernel baseline uses). ``roped=True`` means
        # query_states/key_states below are ALREADY roped [Z, H, S, D] views;
        # the legacy path keeps eager norms + the rope(-quant) producers.
        roped = kind is not None and _fused_norm_rope_ok(self)
        if roped:
            query_states = _fused_norm_rope(
                self.q_norm, query_states.view(hidden_shape), cos, sin
            ).transpose(1, 2)
            key_states = _fused_norm_rope(self.k_norm, key_pre, cos, sin).transpose(
                1, 2
            )
        else:
            query_states = self.q_norm(query_states.view(hidden_shape)).transpose(
                1, 2
            )
            key_states = self.k_norm(key_pre).transpose(1, 2)

        if kind == "packed":
            # Varlen block-diagonal causal attention over the flattened pack:
            # each sample attends only within itself and per-block kv loops skip
            # cross-sample tiles (work scales with sum(s_i^2), not S^2). Same
            # handling for grad and no-grad (cheap, same kernel) so eval/prefill
            # of packed batches stays consistent with training.
            if grad_enabled and (
                not getattr(self, "_nvfp4_train_backward", False)
                or past_key_values is not None
            ):
                return orig_forward(
                    self,
                    hidden_states,
                    position_embeddings,
                    attention_mask,
                    past_key_values,
                    **kwargs,
                )
            if (
                not grad_enabled
                and past_key_values is None
                and _PACKED_FWD_VARLEN
                and not _custom_op_enabled(self)
            ):
                # No-grad packed prefill/eval, varlen-capable packed-operand
                # forward available: feed it from the FUSED PRODUCERS — RoPE +
                # NVFP4 pack of Q/K in one pass (no roped bf16 Q/K HBM
                # materialization + kernel pre-quant re-read), V key-axis packed
                # directly. The producers need no varlen awareness (quant is
                # per-token along D; V's key-axis groups match the in-kernel
                # quant layout exactly) — only the consumer takes cu_seqlens.
                # The TRAINING path below keeps standalone quant: the HP
                # backward needs high-precision roped q/k/v anyway.
                value_states = _v_states()
                if roped:
                    # q/k already roped by the fused norm+rope op; pack with
                    # the standalone strided quant (no eager rope round-trip).
                    qnv, qsc = _quant_packs_along_d(query_states)
                    knv, ksc = _quant_packs_along_d(key_states)
                else:
                    qnv, qsc = fused_rope_quant_qk(query_states, cos, sin)
                    knv, ksc = fused_rope_quant_qk(key_states, cos, sin)
                vnv, vsc, _ = quant_v_keyaxis(value_states, block_n=_BLOCK_N)
                z_p, h_p, _, d_p = query_states.shape
                attn_output = nvfp4_flash_attention_packed(
                    qnv,
                    qsc,
                    knv,
                    ksc,
                    vnv,
                    vsc,
                    z=z_p,
                    h=h_p,
                    hk=key_states.shape[1],
                    s_q=q_len,
                    s_kv=q_len,
                    d=d_p,
                    scaling=self.scaling,
                    out_dtype=query_states.dtype,
                    causal=True,
                    cu_seqlens=cu_seqlens,
                    block_n=_BLOCK_N,
                    out_layout="zshd",
                )  # [Z, S, H, D]
                attn_output = attn_output.reshape(*input_shape, -1).contiguous()
                attn_output = attn_output * torch.sigmoid(gate)
                if use_fp4_o:
                    return _nvfp4_o_proj(self, attn_output), None
                return _o_proj_fwd(self, attn_output), None

            from transformers.models.qwen3_5.modeling_qwen3_5 import (
                apply_rotary_pos_emb,
            )

            value_states = _v_states()
            # TRAINING fused producers: when the fork's training entry accepts
            # pre-packed q/k (q_packs/k_packs), one fused pass emits BOTH the
            # roped hp q/k (saved for the HP backward, grads flow through the
            # exact RoPE transpose) AND the forward NVFP4 packs — eliminating
            # the HF rope kernel round-trip and the standalone q/k forward
            # pre-quant launches. The packs are quantized from the dtype-rounded
            # hp values, so out/LSE/grads are bit-identical to feeding the same
            # roped hp through the no-packs path. grad_enabled here implies no
            # kv cache (cached grad batches returned to orig above).
            #
            # COMPILED (custom-op) path: the producers are raw Triton wrappers
            # dynamo cannot trace, so the pre-pack is skipped and the opaque op
            # runs its own internal pre-quant from the SAME dtype-rounded roped
            # hp — bit-identical out/grads, no graph break in the decoder loop.
            use_custom_op = _custom_op_enabled(self)
            prepack = grad_enabled and _TRAIN_PREPACKED and not use_custom_op
            q_packs = k_packs = None
            if roped:
                # fused norm+rope already produced roped hp q/k (grads flow
                # through its registered backward); just pack the forward
                # operands when the fork takes pre-packed q/k.
                query_roped, key_roped = query_states, key_states
                if prepack:
                    q_packs = _quant_packs_along_d(query_roped)
                    k_packs = _quant_packs_along_d(key_roped)
            elif prepack:
                qnv_t, qsc_t, query_roped = fused_rope_quant_qk_hp(
                    query_states, cos, sin
                )
                knv_t, ksc_t, key_roped = fused_rope_quant_qk_hp(
                    key_states, cos, sin
                )
                q_packs, k_packs = (qnv_t, qsc_t), (knv_t, ksc_t)
            else:
                query_roped, key_roped = apply_rotary_pos_emb(
                    query_states, key_states, cos, sin
                )
            if not grad_enabled and past_key_values is not None:
                past_key_values.update(key_roped, value_states, self.layer_idx)
            ng = query_roped.shape[1] // key_roped.shape[1]
            # Varlen is incompatible with the forward saved-pack set ONLY:
            # save_backward_packs is forced off for packed batches, while the
            # configured grad_dots mode passes through unchanged (every mode
            # composes with varlen; None = kernel AUTO resolves on the pack's
            # MEAN SAMPLE LENGTH — "bf16" below ~4k, else "fp4_rownorm").
            if getattr(self, "_nvfp4_save_backward_packs", False):
                _log_packed_save_packs_ignored_once()
            sr = getattr(self, "_nvfp4_stochastic_rounding", True)
            if use_custom_op:
                # Opaque differentiable custom op: Inductor compiles AROUND
                # the whole native-NVFP4 attention with no graph break, so the
                # decoder loop can compile as one graph (the legacy isolation
                # break pair was removed as a verified-stale guard — see the
                # NOTE next to _CUSTOM_OP_ENV).
                from axolotl.kernels.attn_nvfp4_custom_op import (
                    nvfp4_flash_attn_train_custom_op,
                )

                attn_output = nvfp4_flash_attn_train_custom_op(
                    query_roped,
                    key_roped,
                    value_states,
                    self.scaling,
                    causal=True,
                    num_key_value_groups=ng,
                    cu_seqlens=cu_seqlens,
                    stochastic_rounding=sr,
                    save_backward_packs=False,
                    backward_grad_dots=getattr(self, "_nvfp4_grad_dots", None),
                    out_layout="zshd",
                    q_packs=q_packs,
                    k_packs=k_packs,
                )  # [Z, S, H, D]
            elif grad_enabled:
                # kwargs only when prepacked (a legacy fork's signature lacks
                # q_packs; q_packs is always None there via _TRAIN_PREPACKED).
                prepacked_kwargs = (
                    {"q_packs": q_packs, "k_packs": k_packs}
                    if q_packs is not None
                    else {}
                )
                if _TRAIN_OUT_LAYOUT:
                    # the kernel writes [Z, S, H, D] directly and the backward
                    # consumes the zshd gradient via strided reads — no
                    # transpose-fold copy in either direction.
                    prepacked_kwargs["out_layout"] = "zshd"
                attn_output = nvfp4_flash_attn_func(
                    query_roped,
                    key_roped,
                    value_states,
                    self.scaling,
                    causal=True,
                    num_key_value_groups=ng,
                    cu_seqlens=cu_seqlens,
                    stochastic_rounding=sr,
                    save_backward_packs=False,
                    dkdv_scratch_bf16=getattr(
                        self, "_nvfp4_dkdv_scratch_bf16", False
                    ),
                    **_grad_dots_kwargs(self),
                    **prepacked_kwargs,
                )  # [Z, S, H, D] (zshd) or [Z, H, S, D] (legacy fork)
                if not _TRAIN_OUT_LAYOUT:
                    attn_output = attn_output.transpose(1, 2)  # -> [Z, S, H, D]
            else:
                # No-grad eval/prefill on a legacy fork (packed-operand forward
                # without cu_seqlens) or with a cache to fill: same varlen
                # kernel, forward-only entry with in-kernel quant.
                attn_output = nvfp4_flash_attention(
                    query_roped,
                    key_roped,
                    value_states,
                    self.scaling,
                    causal=True,
                    num_key_value_groups=ng,
                    cu_seqlens=cu_seqlens,
                    out_layout="zshd",
                )  # [Z, S, H, D]
            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            attn_output = attn_output * torch.sigmoid(gate)
            if use_fp4_o:
                return _nvfp4_o_proj(self, attn_output), None
            return _o_proj_fwd(self, attn_output), None

        if kind is not None:
            from transformers.models.qwen3_5.modeling_qwen3_5 import (
                apply_rotary_pos_emb,
            )

            if grad_enabled:
                if (
                    not getattr(self, "_nvfp4_train_backward", False)
                    or past_key_values is not None
                ):
                    return orig_forward(
                        self,
                        hidden_states,
                        position_embeddings,
                        attention_mask,
                        past_key_values,
                        **kwargs,
                    )
                # The graph breaks that used to bracket the FP4 attention
                # block here (an Inductor-miscompile guard from an earlier
                # torch/patch state) were verified STALE and removed so the
                # decoder loop can compile — see the NOTE next to
                # _CUSTOM_OP_ENV for the A/B evidence.
                use_custom_op = _custom_op_enabled(self)
                value_states = _v_states()
                save_packs = getattr(self, "_nvfp4_save_backward_packs", False)
                q_packs = k_packs = None
                if roped:
                    query_roped, key_roped = query_states, key_states
                    # pre-packed forward q/k (skips the kernel's own pre-quant
                    # launches); incompatible with the saved-pack set, which
                    # must derive from one fused HP read — and skipped on the
                    # COMPILED custom-op path (raw-Triton producers are not
                    # traceable; the opaque op pre-quants internally from the
                    # same dtype-rounded hp with bit-identical out/grads).
                    if _TRAIN_PREPACKED and not save_packs and not use_custom_op:
                        q_packs = _quant_packs_along_d(query_roped)
                        k_packs = _quant_packs_along_d(key_roped)
                else:
                    query_roped, key_roped = apply_rotary_pos_emb(
                        query_states, key_states, cos, sin
                    )
                sr = getattr(self, "_nvfp4_stochastic_rounding", True)
                grad_sr = sr and not getattr(
                    self, "_nvfp4_backward_rtn_grad_packs", False
                )
                ng = query_states.shape[1] // key_states.shape[1]
                if use_custom_op:
                    # Differentiable opaque custom op: Inductor compiles AROUND the
                    # whole native-NVFP4 attention (fwd + registered native-NVFP4
                    # bwd) instead of tracing tl.dot_scaled and silently falling the
                    # bwd subgraph back to eager. Same grads as nvfp4_flash_attn_func.
                    from axolotl.kernels.attn_nvfp4_custom_op import (
                        nvfp4_flash_attn_train_custom_op,
                    )

                    attn_output = nvfp4_flash_attn_train_custom_op(
                        query_roped,
                        key_roped,
                        value_states,
                        self.scaling,
                        causal=(kind == "causal"),
                        num_key_value_groups=ng,
                        stochastic_rounding=sr,
                        backward_p_dv_stochastic_rounding=grad_sr,
                        backward_dot_dv_stochastic_rounding=grad_sr,
                        backward_ds_dq_stochastic_rounding=grad_sr,
                        dkdv_scratch_bf16=getattr(
                            self, "_nvfp4_dkdv_scratch_bf16", False
                        ),
                        backward_grad_dots=getattr(self, "_nvfp4_grad_dots", None),
                        save_backward_packs=save_packs,
                        out_layout="zshd",
                        q_packs=q_packs,
                        k_packs=k_packs,
                    )
                else:
                    prepacked_kwargs = (
                        {"q_packs": q_packs, "k_packs": k_packs}
                        if q_packs is not None
                        else {}
                    )
                    if _TRAIN_OUT_LAYOUT:
                        prepacked_kwargs["out_layout"] = "zshd"
                    attn_output = nvfp4_flash_attn_func(
                        query_roped,
                        key_roped,
                        value_states,
                        self.scaling,
                        causal=(kind == "causal"),
                        num_key_value_groups=ng,
                        stochastic_rounding=sr,
                        backward_p_dv_stochastic_rounding=grad_sr,
                        backward_dot_dv_stochastic_rounding=grad_sr,
                        backward_ds_dq_stochastic_rounding=grad_sr,
                        save_backward_packs=save_packs,
                        dkdv_scratch_bf16=getattr(
                            self, "_nvfp4_dkdv_scratch_bf16", False
                        ),
                        **_grad_dots_kwargs(self),
                        **prepacked_kwargs,
                    )  # [Z, S, H, D] (zshd) or [Z, H, S, D] (legacy fork)
                    if not _TRAIN_OUT_LAYOUT:
                        attn_output = attn_output.transpose(1, 2)
            else:
                # Write roped K/V to the cache so subsequent decode steps see prefill
                # context, then run NVFP4 on the same roped K/V.
                fuse_v = (
                    getattr(self, "_nvfp4_fuse_vproj", _FUSE_VPROJ)
                    and past_key_values is None
                    and _can_fuse_vproj(self)
                )
                value_states = None
                if not fuse_v:
                    value_states = _v_states()
                if past_key_values is not None:
                    if roped:
                        k_roped = key_states
                    else:
                        k_roped, _ = apply_rotary_pos_emb(
                            key_states, key_states, cos, sin
                        )
                    past_key_values.update(k_roped, value_states, self.layer_idx)

                attn_output = _nvfp4_attention(
                    self,
                    query_states,
                    key_states,
                    value_states,
                    cos,
                    sin,
                    self.scaling,
                    causal=(kind == "causal"),
                    hidden_states=hidden_states if fuse_v else None,
                    roped=roped,
                )
            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            attn_output = attn_output * torch.sigmoid(gate)
            if use_fp4_o:
                return _nvfp4_o_proj(self, attn_output), None
            return _o_proj_fwd(self, attn_output), None

        return orig_forward(
            self,
            hidden_states,
            position_embeddings,
            attention_mask,
            past_key_values,
            **kwargs,
        )

    return forward


def patch_qwen3_5_nvfp4_attention(
    model: nn.Module,
    fuse_vproj: bool = _FUSE_VPROJ,
    fuse_attn_proj: bool = False,
    train_backward: bool = False,
    backward_rtn_grad_packs: bool = False,
    save_backward_packs: bool = False,
    dkdv_scratch_bf16: bool = False,
    grad_dots: str | None = None,
    bf16_grad_dots: bool | None = None,
    compile_custom_op: bool = False,
    stochastic_rounding: bool = True,
    packed_min_sample_len: int | None = None,
) -> int:
    """Patch every Qwen3.5 FULL-attention layer's forward to use NVFP4 attention.

    Leaves linear-attention layers untouched. Returns the count of patched layers.
    Idempotent: re-patching a model is a no-op on already-patched modules.

    ``fuse_vproj``: run v_proj as a native-NVFP4 GEMM with a fused key-axis pack
    epilogue (no bf16 V / transpose / standalone quant). Faster V producer at a
    small parity cost (v_proj goes FP4); only active on the no-grad cache-free
    prefill path and only for plain ``nn.Linear`` v_proj modules.

    ``grad_dots``: backward grad-GEMM mode ("bf16" / "fp4_rownorm" /
    "fp8_rownorm" / "fp4_legacy"); None (default) = kernel AUTO — "bf16" below
    an effective sequence of 4096 (the MEAN SAMPLE LENGTH for packed batches),
    else "fp4_rownorm" (measured 1.12x hp / 1.13x FA2 backward at s8192 d256
    with hp-equal convergence). ``bf16_grad_dots`` is the DEPRECATED boolean
    alias (True -> "bf16", False -> "fp4_legacy"); passing both raises.

    ``packed_min_sample_len``: packed-batch perf gate — packs whose MEAN sample
    length is below this keep the model's original (FA2-varlen) forward.
    Default 0 (gate off): with the fused RoPE+quant producers feeding the
    training forward pre-packed q/k plus the varlen-tuned tiles, FP4 varlen
    beats FA2-varlen at every measured packed mean (128-2048). A positive
    value re-routes short-mean packs to the original forward (useful on
    legacy sageattention forks without the pre-packed training entry); a very
    large value never uses FP4 for packed batches.
    """
    grad_dots = _resolve_grad_dots_alias(grad_dots, bf16_grad_dots)
    if packed_min_sample_len is None:
        packed_min_sample_len = _PACKED_MIN_SAMPLE_LEN_DEFAULT
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Attention

    # qwen3_5_moe uses its own full-attention class (Qwen3_5MoeAttention) for the
    # full_attention layers; its GatedDeltaNet/Vision attention are separate classes
    # and stay untouched. Same head_dim-256 gated softmax forward, so the wrapper fits.
    attn_classes: tuple = (Qwen3_5Attention,)
    try:
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
            Qwen3_5MoeAttention,
        )

        attn_classes = attn_classes + (Qwen3_5MoeAttention,)
    except ImportError:
        pass

    patched = 0
    seen_forward = None
    for module in model.modules():
        if isinstance(module, attn_classes):
            if getattr(module, "_nvfp4_patched", False):
                module._nvfp4_fuse_vproj = fuse_vproj
                module._nvfp4_train_backward = train_backward
                module._nvfp4_backward_rtn_grad_packs = backward_rtn_grad_packs
                module._nvfp4_save_backward_packs = save_backward_packs
                module._nvfp4_dkdv_scratch_bf16 = dkdv_scratch_bf16
                module._nvfp4_grad_dots = grad_dots
                module._nvfp4_compile_custom_op = compile_custom_op
                module._nvfp4_stochastic_rounding = stochastic_rounding
                module._nvfp4_packed_min_sample_len = packed_min_sample_len
                module._nvfp4_fuse_attn_proj = fuse_attn_proj
                module._nvfp4_attn_proj_ok = (
                    _attn_proj_ok(module) if fuse_attn_proj else False
                )
                _fused_norm_rope_ok(module)
                continue
            orig = type(module).forward
            if seen_forward is None:
                seen_forward = make_nvfp4_forward(orig)
            module.forward = seen_forward.__get__(module, type(module))
            module._nvfp4_patched = True
            module._nvfp4_fuse_vproj = fuse_vproj
            module._nvfp4_train_backward = train_backward
            module._nvfp4_backward_rtn_grad_packs = backward_rtn_grad_packs
            module._nvfp4_save_backward_packs = save_backward_packs
            module._nvfp4_dkdv_scratch_bf16 = dkdv_scratch_bf16
            module._nvfp4_grad_dots = grad_dots
            module._nvfp4_compile_custom_op = compile_custom_op
            module._nvfp4_stochastic_rounding = stochastic_rounding
            module._nvfp4_packed_min_sample_len = packed_min_sample_len
            module._nvfp4_fuse_attn_proj = fuse_attn_proj
            module._nvfp4_attn_proj_ok = (
                _attn_proj_ok(module) if fuse_attn_proj else False
            )
            # warm the fused norm+rope capability cache at patch time so the
            # lazy first-call setattr never happens inside a dynamo trace
            _fused_norm_rope_ok(module)
            patched += 1
    # Hoist the data-dependent packed-batch classification out of the compiled
    # decoder loop: resolved once per step in this (dynamo-disabled) model
    # pre-forward hook, threaded to the layers as module attributes.
    _install_step_info_hook(model)
    LOG.info(
        "nvfp4 attention: patched %d Qwen3.5 full-attention layers "
        "(fuse_vproj=%s, train_backward=%s, backward_rtn_grad_packs=%s, "
        "save_backward_packs=%s, dkdv_scratch_bf16=%s, grad_dots=%s, "
        "compile_custom_op=%s, packed_min_sample_len=%s)",
        patched,
        fuse_vproj,
        train_backward,
        backward_rtn_grad_packs,
        save_backward_packs,
        dkdv_scratch_bf16,
        grad_dots,
        compile_custom_op,
        packed_min_sample_len,
    )
    return patched
