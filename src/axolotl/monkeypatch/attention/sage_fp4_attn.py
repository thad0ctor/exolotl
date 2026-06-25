"""SageAttention-3 FP4 microscaling attention as an opt-in inference backend.

SageAttention-3 (arXiv 2505.11594) is an FP4 microscaling attention FORWARD for
Blackwell (sm_120): Q, K, V come in as fp16/bf16 and the QK^T and PV matmuls run in
NVFP4 with per-block scales, an FP4 analogue of FlashAttention. It is the natural
deployment for the FP4-attention-robust models this fork trains with Attn-QAT
(``fp4_attention_qat``): train fake-quant, serve real FP4.

This is INFERENCE ONLY — the Sage-3 kernel has no backward, so the registered
forward raises under ``torch.is_grad_enabled()`` rather than silently producing
wrong gradients. The kernel itself supports only dense causal/full attention with a
standard ``1/sqrt(head_dim)`` scale and ``head_dim < 256``; anything outside that
(per-key padding, sliding window, custom scaling, large head_dim) falls back to SDPA
so output stays correct — speed is sacrificed, never correctness.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

SAGE_FP4_IMPL_NAME = "sage_fp4"

_MAX_HEAD_DIM = 256  # Sage-3 kernel is undefined at/above this; SDPA handles it

_PER_BLOCK_MEAN = True
_ALLOW_FALLBACK = True

sageattn3_blackwell = None  # pylint: disable=invalid-name


def configure_sage_fp4(
    per_block_mean: bool = True, allow_fallback: bool = True
) -> None:
    """Set the runtime knobs read by ``sage_fp4_attention_forward``."""
    global _PER_BLOCK_MEAN, _ALLOW_FALLBACK  # pylint: disable=global-statement
    _PER_BLOCK_MEAN = per_block_mean
    _ALLOW_FALLBACK = allow_fallback


def _is_sage_fp4_available() -> bool:
    try:
        import sageattn3  # noqa: F401  # pylint: disable=unused-import

        return True
    except ImportError:
        return False


if _is_sage_fp4_available():
    from sageattn3 import sageattn3_blackwell


def _flash_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("flash_attn") is not None


def _fallback_kind(attention_mask: torch.Tensor | None, q_len: int, kv_len: int) -> str:
    """Classify the fallback for inputs the FP4 kernel can't serve.

    ``"causal"`` / ``"full"`` are flash-attention-eligible (dense, no per-key
    padding); ``"sdpa"`` is everything flash can't represent as a dense causal/full
    mask (per-key padding, sliding window, arbitrary additive bias)."""
    if attention_mask is None:
        return "causal" if q_len == kv_len else "full"
    if attention_mask.dim() != 4:
        return "sdpa"
    m = attention_mask[..., :kv_len]
    if m.shape[-1] != kv_len:
        return "sdpa"
    neg = torch.finfo(m.dtype).min
    keep = m > neg / 2
    if keep.shape[1] != 1:
        if not bool((keep == keep[:, :1]).all()):
            return "sdpa"
        keep = keep[:, :1]
    keep = keep[:, 0]
    if bool(keep.all()):
        return "full"
    if q_len == kv_len:
        causal = torch.tril(
            torch.ones(q_len, kv_len, dtype=torch.bool, device=keep.device)
        )[None]
        if bool((keep == causal).all()):
            return "causal"
    return "sdpa"


def _check_available() -> None:
    if sageattn3_blackwell is None:
        raise ImportError(
            "SageAttention-3 (sage_fp4) is not installed. Build it from source for "
            "your arch (sm_120): "
            "`cd SageAttention/sageattention3_blackwell && "
            'TORCH_CUDA_ARCH_LIST="12.0" python -m pip install --no-build-isolation .`'
        )


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """GQA broadcast: [B, n_kv, S, D] -> [B, n_kv*n_rep, S, D] (mirrors HF)."""
    batch, num_kv, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_kv, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_kv * n_rep, slen, head_dim)


def _mask_plan(
    attention_mask: torch.Tensor | None, q_len: int, kv_len: int
) -> str | None:
    """Classify ``attention_mask`` for the dense Sage-3 kernel.

    Returns ``"full"`` when every query attends every key (no masking — e.g. a
    single decode step, or a verified all-ones mask), else ``None`` to fall back to
    SDPA.

    NOTE: the FP4 *causal* kernel is numerically broken on Blackwell sm_120 (causal
    cosine-sim ~0.72 vs SDPA, confirmed SM-count-independent on RTX 5090 and RTX PRO
    6000 — Sage-3 was authored for non-causal diffusion attention). So causal
    (prefill) is intentionally NOT served by the FP4 kernel here; it returns None and
    routes to SDPA. Only the verified-correct full/non-causal path (cos ~0.98) uses
    FP4.
    """
    if attention_mask is None:
        return "full" if q_len == 1 else None  # square causal -> SDPA (see note)
    if attention_mask.dim() != 4:
        return None
    m = attention_mask[..., :kv_len]
    if m.shape[-1] != kv_len:
        return None

    neg = torch.finfo(m.dtype).min
    keep = m > neg / 2  # [Z, H_or_1, q_len, kv_len] boolean: True = attend
    if keep.shape[1] != 1:
        if not bool((keep == keep[:, :1]).all()):
            return None
        keep = keep[:, :1]
    keep = keep[:, 0]  # [Z, q_len, kv_len]

    if bool(keep.all()):
        return "full"
    return None


def sage_fp4_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    scaling: float | None = None,
    dropout: float = 0.0,
    is_causal: bool | None = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    """FP4 attention forward (inference) compatible with the HF AttentionInterface.

    Inputs are ``[B, H, S, D]``. Dispatches to ``sageattn3_blackwell`` for the dense
    causal/full standard-scale case and to SDPA otherwise; returns ``[B, S, H, D]``.
    """
    _check_available()

    if torch.is_grad_enabled() and (
        query.requires_grad or key.requires_grad or value.requires_grad
    ):
        raise RuntimeError(
            "sage_fp4 is an inference-only attention backend (the SageAttention-3 "
            "FP4 kernel has no backward). Use it for `axolotl inference`/eval or "
            "serving; for training use `fp4_attention_qat` (fake-quant, trainable)."
        )
    if kwargs.get("output_attentions") or kwargs.get("head_mask") is not None:
        raise NotImplementedError(
            "sage_fp4 does not support output_attentions=True or head_mask."
        )

    key = _repeat_kv(key, getattr(module, "num_key_value_groups", 1))
    value = _repeat_kv(value, getattr(module, "num_key_value_groups", 1))

    head_dim = query.shape[-1]
    q_len, kv_len = query.shape[-2], key.shape[-2]
    std_scale = 1.0 / math.sqrt(head_dim)
    nonstandard_scale = scaling is not None and abs(scaling - std_scale) > 1e-6

    plan = None
    if dropout == 0.0 and head_dim < _MAX_HEAD_DIM and not nonstandard_scale:
        plan = _mask_plan(attention_mask, q_len, kv_len)

    if plan is not None:
        if not getattr(sage_fp4_attention_forward, "_fp4_logged", False):
            LOG.info("sage_fp4: using SageAttention-3 FP4 kernel (plan=%s)", plan)
            sage_fp4_attention_forward._fp4_logged = True
        out = sageattn3_blackwell(
            query, key, value, is_causal=False, per_block_mean=_PER_BLOCK_MEAN
        )
        return out.transpose(1, 2).contiguous(), None

    if not _ALLOW_FALLBACK:
        raise RuntimeError(
            "sage_fp4: input cannot use the FP4 kernel (per-key padding, sliding "
            "window, custom softmax scale, head_dim>=%d, or dropout) and "
            "allow_fallback is false. Pad to a uniform length / use a model with "
            "standard scaling, or enable the fallback." % _MAX_HEAD_DIM
        )

    kind = _fallback_kind(attention_mask, q_len, kv_len)

    # Prefer flash attention (FA2 covers head_dim<=256, so it serves Qwen3.5 causal
    # prefill fast); SDPA only for masks flash can't represent densely (per-key
    # padding, sliding window, arbitrary additive bias).
    if kind in ("causal", "full") and dropout == 0.0 and _flash_available():
        from flash_attn import flash_attn_func

        if not getattr(sage_fp4_attention_forward, "_flash_logged", False):
            LOG.info("sage_fp4: falling back to flash_attn (kind=%s)", kind)
            sage_fp4_attention_forward._flash_logged = True
        out = flash_attn_func(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            dropout_p=dropout,
            softmax_scale=scaling,
            causal=(kind == "causal"),
        )
        return out, None  # flash returns [B, S, H, D]

    if not getattr(sage_fp4_attention_forward, "_sdpa_logged", False):
        LOG.info(
            "sage_fp4: falling back to SDPA (kind=%s; flash can't represent this "
            "mask densely or flash_attn is unavailable).",
            kind,
        )
        sage_fp4_attention_forward._sdpa_logged = True

    out = F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attention_mask[..., :kv_len] if attention_mask is not None else None,
        dropout_p=dropout,
        is_causal=(attention_mask is None and q_len == kv_len),
        scale=scaling,
    )
    return out.transpose(1, 2).contiguous(), None


def register_sage_fp4_attn() -> None:
    """Register ``sage_fp4`` in the transformers attention + mask registries.

    Uses ``eager_mask`` (a 4D additive causal+padding mask) so the forward can
    classify causal vs padded and route padded batches to SDPA — without it the
    registry returns ``None`` (bidirectional attention, future-token leakage).
    """
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, eager_mask
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    ALL_ATTENTION_FUNCTIONS.register(SAGE_FP4_IMPL_NAME, sage_fp4_attention_forward)
    if SAGE_FP4_IMPL_NAME not in ALL_MASK_ATTENTION_FUNCTIONS._global_mapping:
        ALL_MASK_ATTENTION_FUNCTIONS.register(SAGE_FP4_IMPL_NAME, eager_mask)


def patch_sage_fp4_attn() -> None:
    """Validate Sage-3 is importable; registration is done by register_sage_fp4_attn."""
    _check_available()
    LOG.info("SageAttention-3 (sage_fp4) validated successfully")
