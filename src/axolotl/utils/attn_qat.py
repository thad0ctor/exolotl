# mypy: disable-error-code="attr-defined"
"""FP4-attention quantization-aware training (Attn-QAT, arXiv 2603.00040).

QAT for FP4 attention: during SFT the attention operands (Q, K, V and the
softmax probabilities P) are FAKE-quantized to NVFP4 — quantized then immediately
dequantized, with all matmuls left in bf16 — so the model learns to tolerate real
FP4 attention at inference. This is an ACCURACY feature, not a speed one: a training
step stays ~bf16 cost/precision; the payoff is FP4-attention robustness for serving.

The fake-quant uses a straight-through estimator (STE): forward returns the
dequantized value, backward passes the gradient through unchanged. v1 is an EAGER
attention (P is materialized), which is correct for QAT — the STE makes the
backward exact without the paper's auxiliary high-precision O' reconstruction.

``nvfp4_qat_attention_forward`` dispatches to the fused FlashAttention-style Triton
kernel (``axolotl.kernels.attn_qat_flash``, linear memory, recomputes P with the O'
identity in its backward) when applicable — supported head_dim, Triton available,
dropout off, and a causal/full mask optionally combined with a per-key padding mask
(the common padded-SFT case) — and falls back to the eager path below for exotic
masks (sliding window / arbitrary per-query bias) or unsupported head_dims.
"""

from __future__ import annotations

import logging

import torch
from torch import nn

from axolotl.kernels.attn_qat_flash import (
    _supported as _fused_supported,
    fused_nvfp4_qat_attention as _fused_nvfp4_qat_attention,
)

LOG = logging.getLogger(__name__)

ATTN_QAT_IMPL_NAME = "nvfp4_qat"

_BLOCK_SIZE = 16  # NVFP4 block, taken along the last (head_dim) axis


def _nvfp4_quant_dequant(x: torch.Tensor) -> torch.Tensor:
    """NVFP4 fake-quant φ over the last axis in blocks of 16: per-block scale
    ``amax/6`` (e4m3), round to the E2M1 grid, dequantize back to ``x``'s dtype.

    Reuses torchao's ``NVFP4Tensor`` round-trip, which implements exactly this
    (amax/6 e4m3 block scale + round-to-nearest-even on the E2M1 grid). The
    softmax probabilities P are quantized along the key axis whose length is the
    sequence length and need not be a multiple of 16, so zero-pad the trailing
    block and slice it back — the zeros sit in their own partial block and do not
    perturb the real blocks' scales.
    """
    from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor

    orig_shape = x.shape
    orig_dtype = x.dtype
    n = orig_shape[-1]
    pad = (-n) % _BLOCK_SIZE
    x2d = x.reshape(-1, n)
    if pad:
        x2d = torch.nn.functional.pad(x2d, (0, pad))
    q = NVFP4Tensor.to_nvfp4(x2d.contiguous(), block_size=_BLOCK_SIZE)
    out = q.dequantize(orig_dtype)
    if pad:
        out = out[:, :n]
    return out.reshape(orig_shape)


def fake_quant_nvfp4(x: torch.Tensor) -> torch.Tensor:
    """Straight-through NVFP4 fake-quant: forward = quant→dequant, backward = id.

    ``x_fq = x + (φ(x) - x).detach()`` so the value seen downstream is the FP4
    round-trip while the gradient flows back as if no quantization happened.
    """
    return x + (_nvfp4_quant_dequant(x) - x).detach()


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """GQA broadcast: [B, n_kv, S, D] -> [B, n_kv*n_rep, S, D] (mirrors HF)."""
    batch, num_kv, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_kv, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_kv * n_rep, slen, head_dim)


def _fused_mask_plan(
    attention_mask: torch.Tensor | None, q_len: int, kv_len: int, dtype
) -> tuple[bool, torch.Tensor | None] | None:
    """Decide whether the fused kernel can serve ``attention_mask`` and, if so,
    how. Returns ``(causal, key_pad_bias)`` or ``None`` for unsupported masks.

    Supported: None (causal when square), plain causal, and causal-or-full plus a
    pure per-key padding mask (constant across query rows). The padding is returned
    as a ``[Z, kv_len]`` additive bias (0 keep, -inf pad). Anything else — sliding
    window, per-query bias, arbitrary additive bias — returns ``None`` so dispatch
    falls back to eager.

    ``key_pad_bias`` is None when there is no padding (pure causal), so the kernel
    skips the bias load entirely.
    """
    if attention_mask is None:
        return (True, None) if q_len == kv_len else None
    if attention_mask.dim() != 4:
        return None
    m = attention_mask[..., :kv_len]
    if m.shape[-1] != kv_len:
        return None

    neg = torch.finfo(m.dtype).min
    keep = m > neg / 2  # [Z, Hheads_or_1, q_len, kv_len] boolean

    # Must be identical across heads (HF padding/causal masks are head-broadcast).
    if keep.shape[1] != 1:
        if not bool((keep == keep[:, :1]).all()):
            return None
        keep = keep[:, :1]
    keep = keep[:, 0]  # [Z, q_len, kv_len]

    # Per-key padding = a key padded iff masked at the last query row, which (when
    # causal) attends to every key, or (when full) at any row. Validate the whole
    # mask equals (causal_component AND key_pad), so no other structure leaks.
    if q_len == kv_len:
        causal = torch.tril(
            torch.ones(q_len, kv_len, dtype=torch.bool, device=keep.device)
        )[None]
        key_pad = keep[:, -1, :]  # last row sees all keys causally
        recon = causal & key_pad[:, None, :]
        if bool((keep == recon).all()):
            return (True, _key_pad_bias(key_pad, neg_inf_dtype=dtype))
    # full (non-causal) attention with pure key padding, constant over query rows
    key_pad = keep.all(dim=1)  # [Z, kv_len]: key kept for every query row
    recon = key_pad[:, None, :].expand_as(keep)
    if bool((keep == recon).all()):
        return (False, _key_pad_bias(key_pad, neg_inf_dtype=dtype))
    return None


def _key_pad_bias(key_pad: torch.Tensor, neg_inf_dtype) -> torch.Tensor | None:
    """[Z, kv_len] boolean keep -> additive bias (0 keep, -inf pad), or None if
    every key is kept (no padding -> let the kernel skip the bias path)."""
    if bool(key_pad.all()):
        return None
    bias = torch.zeros(key_pad.shape, dtype=neg_inf_dtype, device=key_pad.device)
    bias.masked_fill_(~key_pad, float("-inf"))
    return bias


def nvfp4_qat_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    """Attention with NVFP4 fake-quant on Q, K, V and the softmax weights P.

    Uses the fused FlashAttention-style Triton kernel (linear memory) when the
    head_dim is supported, Triton is available, dropout is off, and the mask is a
    causal/full mask optionally with a per-key padding mask; otherwise falls back
    to the eager v1 below (materialized P) — same fake-quant, exotic-mask/head_dim
    safe. The eager path is structurally identical to HF's
    ``eager_attention_forward`` (GQA repeat, scaling, additive mask, fp32 softmax,
    dropout) with the four Algorithm-2 fake-quants inserted.
    """
    head_dim = query.shape[-1]
    q_len = query.shape[-2]
    kv_len = key.shape[-2]
    plan = None
    if dropout == 0.0 and _fused_supported(head_dim):
        plan = _fused_mask_plan(attention_mask, q_len, kv_len, query.dtype)
    if plan is not None:
        causal, key_pad_bias = plan
        if not getattr(nvfp4_qat_attention_forward, "_fused_logged", False):
            LOG.info(
                "fp4_attention_qat: using FUSED Triton flash kernel "
                "(causal=%s, key_padding=%s)",
                causal,
                key_pad_bias is not None,
            )
            nvfp4_qat_attention_forward._fused_logged = True
        out = _fused_nvfp4_qat_attention(
            query,
            key,
            value,
            scaling,
            causal,
            module.num_key_value_groups,
            key_pad_bias,
        )
        return out.transpose(1, 2).contiguous(), None

    if not getattr(nvfp4_qat_attention_forward, "_eager_logged", False):
        LOG.info("fp4_attention_qat: using EAGER materialized-P attention (v1)")
        nvfp4_qat_attention_forward._eager_logged = True

    key_states = _repeat_kv(key, module.num_key_value_groups)
    value_states = _repeat_kv(value, module.num_key_value_groups)

    q_fq = fake_quant_nvfp4(query)
    k_fq = fake_quant_nvfp4(key_states)
    v_fq = fake_quant_nvfp4(value_states)

    attn_weights = torch.matmul(q_fq, k_fq.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask[..., : k_fq.shape[-2]]

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        query.dtype
    )
    attn_weights = nn.functional.dropout(
        attn_weights, p=dropout, training=module.training
    )

    p_fq = fake_quant_nvfp4(attn_weights)

    attn_output = torch.matmul(p_fq, v_fq)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


def register_nvfp4_qat_attention() -> None:
    """Register the QAT attention in the transformers attention-function registry.

    Also registers the matching mask builder: ``create_causal_mask`` skips mask
    construction (returns None -> bidirectional attention, future-token leakage)
    for any impl absent from ``ALL_MASK_ATTENTION_FUNCTIONS``, so map this key to
    ``eager_mask``, which emits the additive 4D causal+padding mask this eager
    attention consumes.
    """
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, eager_mask
    from transformers.modeling_utils import AttentionInterface

    AttentionInterface.register(ATTN_QAT_IMPL_NAME, nvfp4_qat_attention_forward)
    if ATTN_QAT_IMPL_NAME not in ALL_MASK_ATTENTION_FUNCTIONS._global_mapping:
        ALL_MASK_ATTENTION_FUNCTIONS.register(ATTN_QAT_IMPL_NAME, eager_mask)


def apply_nvfp4_qat_attention(model: nn.Module) -> None:
    """Register + select NVFP4 fake-quant attention on ``model`` (opt-in).

    Registers the function, then routes the model's attention dispatch through it
    via ``set_attn_implementation`` (validates the key and propagates to every
    submodel). The model must follow the HF AttentionInterface dispatch pattern.
    """
    register_nvfp4_qat_attention()
    set_impl = getattr(model, "set_attn_implementation", None)
    if set_impl is None:
        raise RuntimeError(
            "fp4_attention_qat: model has no set_attn_implementation; this "
            "transformers version predates the AttentionInterface registry."
        )
    set_impl(ATTN_QAT_IMPL_NAME)
    LOG.info(
        "fp4_attention_qat: NVFP4 fake-quant attention active (impl=%s)",
        ATTN_QAT_IMPL_NAME,
    )
