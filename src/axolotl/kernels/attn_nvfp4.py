"""Forward-only NVFP4 attention: real FP4 QK^T and PV GEMMs on Blackwell.

Inference attention whose two matmuls run as native NVFP4 ``torch._scaled_mm``
GEMMs (e4m3 block scales, group-16, MSLK-swizzled scales). The whole design
hinges on quantize-ONCE: the per-call torchao/MSLK NVFP4 quantizer dominates at
attention shapes (a re-quant per GEMM was measured ~50x slower than bf16), so
each operand is quantized exactly once and the pre-quantized buffers are fed
straight to ``_scaled_mm`` via the fork's ``_mslk_scaled_mm`` seam.

Two-GEMM materialized attention (the [S,S] score matrix is built explicitly;
fine for inference at moderate seq):
  1. quantize Q (along D) and K (along D) once -> S = Q @ K^T via NVFP4.
  2. scale, causal mask, key-pad bias, softmax in fp32.
  3. quantize P (along the key axis) and V^T (along the key axis) once ->
     O = P @ V via NVFP4.

Layout rules forced by ``torch._scaled_mm`` (TN, packs 2 FP4/byte):
  * the contraction dim must be a multiple of 32 (logical), so QK contracts
    over D (256/128, already ok) and PV contracts over the key seq, which is
    zero-padded to 32. Zero score columns are masked to -inf (weight 0) and
    zero V rows contribute 0, so the padding is exact.
  * the B operand's trailing dim (= N, the GEMM output width) must be a
    multiple of 16. For QK that is the key seq (padded to 32); for PV that is
    the head dim (256/128, ok).

Forward only — no autograd. GQA is handled by repeat_kv on the (cheap) bf16
inputs before quantization.
"""

from __future__ import annotations

import torch

from axolotl.utils.nvfp4_training import (
    _GEMM_ALIGN,
    _mslk_quantize,
    _mslk_scaled_mm,
)

_NEG_INF = float("-inf")


def _pad_last(t: torch.Tensor, align: int) -> tuple[torch.Tensor, int]:
    """Zero-pad the last dim of ``t`` up to a multiple of ``align``."""
    n = t.shape[-1]
    rem = n % align
    if rem == 0:
        return t, n
    return torch.nn.functional.pad(t, (0, align - rem)), n


def _repeat_kv(t: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand [Z, Hk, S, D] -> [Z, Hk*n_rep, S, D] (GQA), matching HF repeat_kv."""
    if n_rep == 1:
        return t
    z, hk, s, d = t.shape
    return t[:, :, None, :, :].expand(z, hk, n_rep, s, d).reshape(z, hk * n_rep, s, d)


def nvfp4_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scaling: float,
    causal: bool = False,
    num_key_value_groups: int = 1,
    key_pad_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Native-NVFP4 inference attention, quantize-once.

    Args:
        query: ``[Z, H, S, D]`` high-precision (bf16/fp16/fp32).
        key/value: ``[Z, Hk, S_kv, D]`` (pre-repeat_kv GQA).
        scaling: softmax scale applied to the QK^T scores (e.g. ``1/sqrt(D)``).
        causal: apply a lower-triangular causal mask (assumes the standard
            bottom-right alignment when ``S != S_kv``).
        num_key_value_groups: ``H // Hk``; K/V are repeated to ``H`` heads.
        key_pad_bias: optional ``[Z, S_kv]`` additive bias on the key axis
            (e.g. ``0`` for real tokens, ``-inf`` for padding), broadcast over
            heads and query positions.

    Returns:
        ``[Z, H, S, D]`` attention output in ``query.dtype``.
    """
    z, h, s_q, d = query.shape
    out_dtype = query.dtype

    key = _repeat_kv(key, num_key_value_groups)
    value = _repeat_kv(value, num_key_value_groups)
    s_kv = key.shape[2]

    # _scaled_mm wants: QK contraction over D (mult of 32) and the score width
    # (= key seq) a mult of 16; pad the key seq to 32 to satisfy both GEMMs at
    # once (PV then contracts over this same padded key seq).
    d_pad = ((d + _GEMM_ALIGN - 1) // _GEMM_ALIGN) * _GEMM_ALIGN
    s_kv_pad = ((s_kv + _GEMM_ALIGN - 1) // _GEMM_ALIGN) * _GEMM_ALIGN

    # Flatten the batch*head loop into 2D GEMMs. Each (z,head) is independent.
    q2 = query.reshape(z * h, s_q, d)
    k2 = key.reshape(z * h, s_kv, d)
    v2 = value.reshape(z * h, s_kv, d)

    if d_pad != d:
        q2 = torch.nn.functional.pad(q2, (0, d_pad - d))
        k2 = torch.nn.functional.pad(k2, (0, d_pad - d))

    # _scaled_mm needs the QK B-operand (K) trailing dim = key seq a mult of 16,
    # and the PV contraction = key seq a mult of 32. Pad K/V seq to 32 ONCE; the
    # padded score columns are masked to -inf (softmax weight 0) and the padded V
    # rows contribute 0, so the result is exact.
    if s_kv_pad != s_kv:
        k2 = torch.nn.functional.pad(k2, (0, 0, 0, s_kv_pad - s_kv))
        v2 = torch.nn.functional.pad(v2, (0, 0, 0, s_kv_pad - s_kv))

    # Causal/padding masks built once (shared across heads), added in fp32 over
    # the padded key width; the [s_kv:s_kv_pad] tail is always masked.
    add_mask = torch.zeros(s_q, s_kv_pad, device=query.device, dtype=torch.float32)
    if s_kv_pad != s_kv:
        add_mask[:, s_kv:] = _NEG_INF
    if causal:
        offset = s_kv - s_q  # bottom-right alignment for cross-length causal
        qpos = torch.arange(s_q, device=query.device).unsqueeze(1)
        kpos = torch.arange(s_kv, device=query.device).unsqueeze(0)
        causal_mask = kpos > qpos + offset  # [s_q, s_kv] True where masked
        add_mask[:, :s_kv] = add_mask[:, :s_kv].masked_fill(causal_mask, _NEG_INF)

    bias_zh = None
    if key_pad_bias is not None:
        # [Z, S_kv] -> [Z, 1, S_kv_pad] broadcast over heads, expanded to z*h rows.
        bias_p = torch.nn.functional.pad(
            key_pad_bias.to(torch.float32), (0, s_kv_pad - s_kv)
        )
        bias_zh = (
            bias_p.reshape(z, 1, s_kv_pad)
            .expand(z, h, s_kv_pad)
            .reshape(z * h, 1, s_kv_pad)
        )

    outs = []
    for i in range(z * h):
        qi = q2[i]  # [s_q, d_pad]
        ki = k2[i]  # [s_kv_pad, d_pad]
        vi = v2[i]  # [s_kv_pad, d]

        # --- GEMM 1: scores = Q @ K^T (contraction D, quantize each once) ---
        qq, qsc, qinv = _mslk_quantize(qi.contiguous())
        kq, ksc, kinv = _mslk_quantize(ki.contiguous())
        scores = _mslk_scaled_mm(
            qq, qsc, qinv, kq, ksc, kinv, torch.float32
        )  # [s_q, s_kv_pad]
        scores = scores * scaling + add_mask
        if bias_zh is not None:
            scores = scores + bias_zh[i]

        probs = torch.softmax(scores, dim=-1).to(out_dtype)  # [s_q, s_kv_pad]

        # --- GEMM 2: out = P @ V (contraction = padded key seq) ---
        pq, psc, pinv = _mslk_quantize(probs.contiguous())
        # B operand is V^T ([D, s_kv_pad]) quantized along the (padded) key axis;
        # _mslk_scaled_mm computes P @ (V^T)^T = P @ V.
        vq, vsc, vinv = _mslk_quantize(vi.t().contiguous())
        oi = _mslk_scaled_mm(pq, psc, pinv, vq, vsc, vinv, out_dtype)  # [s_q, d]
        outs.append(oi)

    out = torch.stack(outs, dim=0).reshape(z, h, s_q, d)
    return out
