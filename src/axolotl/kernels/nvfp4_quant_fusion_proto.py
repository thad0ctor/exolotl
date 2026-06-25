"""Prototype: fuse the NVFP4 quant into the ops that PRODUCE attention operands.

Premise (materialized native-NVFP4 attention, e.g. attn_nvfp4.py): the standalone
per-operand quant (_mslk_quantize of Q, K, V, P) dominates — 4 quant launches per
(z,head), 64 launches for H16. Each launch re-reads a tensor that the producing op
(RoPE, softmax, v_proj) just wrote. This module emits the operand ALREADY
NVFP4-packed from the producing op's single pass, so the standalone quant launch
disappears.

All native NVFP4: e4m3 per-block (group-16) scale, MSLK-swizzled 128x4 scale tile,
hardware ``cvt.rn.satfinite.e2m1x2.f32`` packing — bit-compatible with
``mslk.triton_quantize_nvfp4`` so the output drops straight into ``_mslk_scaled_mm``.

The three fused producers, each validated (dequant cos / max-abs-err) against
"transform then _mslk_quantize" separately:
  1. fused_rope_quant_qk  — RoPE rotation + NVFP4 pack of rotated Q,K (block-16 on D)
  2. fused_softmax_quant_p — row softmax + NVFP4 pack of P            (block-16 on key)
  3. fused_vproj_epilogue_quant — identity epilogue + NVFP4 pack of V (block-16 on key)

The quant core (block amax -> e4m3 scale -> hw FP4 pack -> swizzled scale store) is
lifted from MSLK's _nvfp4_quantize_stacked_kernel; only the per-tile TRANSFORM differs.
Two-level scaling (per-tensor global_scale folded back at GEMM time) matches
``_mslk_quantize`` exactly so parity is apples-to-apples.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from mslk.quantize.triton.fp4_quantize import (
    convert_fp32_to_fp4_packed,
    nvfp4_scale_swizzle,
)

from axolotl.utils.nvfp4_training import _NVFP4_GLOBAL_AMAX

_BLOCK = 16
_E4M3_EPS = tl.constexpr(1.5258789e-05)
_F8E4M3_MAX = tl.constexpr(448.0)
_F4_MAX = tl.constexpr(6.0)


def _swizzled_scale_shape(m: int, k: int) -> tuple[int, int]:
    rounded_m = (m + 127) // 128 * 128
    n_blocks = k // _BLOCK
    rounded_k = (n_blocks + 3) // 4 * 4
    return rounded_m, rounded_k


@triton.jit
def _quant_emit(
    x_blocks,  # [M_PER_BLOCK, 4, 16] fp32, ALREADY-TRANSFORMED tile (one 64-col chunk)
    q_ptr,
    s_ptr,
    row_scale,  # global_scale scalar (= _NVFP4_GLOBAL_AMAX / amax)
    pid_m,
    pid_n,
    M,
    N,
    M_PER_BLOCK: tl.constexpr,
    NUM_N_BLOCKS: tl.constexpr,
):
    """NVFP4 block-quant + swizzled-scale store of a transformed [M_PER_BLOCK,64] tile.

    Lifted verbatim from MSLK's stacked-quant epilogue (steps 4-7) so the packed
    qdata and swizzled e4m3 scale are bit-identical to triton_quantize_nvfp4.
    """
    NUM_ELEM_PER_LAYOUT: tl.constexpr = 128 * 4

    offs_m = pid_m * M_PER_BLOCK + tl.arange(0, M_PER_BLOCK)[:, None]

    block_amax = tl.max(tl.abs(x_blocks), axis=2)  # [M_PER_BLOCK, 4]
    scales = tl.div_rn(block_amax, _F4_MAX) * row_scale
    scales = tl.clamp(scales, _E4M3_EPS, _F8E4M3_MAX)
    scales = scales.to(tl.float8e4nv)

    total_scale = tl.div_rn(row_scale, scales.to(tl.float32)[:, :, None])
    x_blocks = x_blocks * total_scale

    x_fp4x2 = convert_fp32_to_fp4_packed(x_blocks.reshape(M_PER_BLOCK, 32, 2).split())
    fp4_offs_m = pid_m * M_PER_BLOCK + tl.arange(0, M_PER_BLOCK)[:, None]
    fp4_offs_n = pid_n * 32 + tl.arange(0, 32)[None, :]
    fp4_mask = (fp4_offs_m < M) & (fp4_offs_n < N // 2)
    tl.store(q_ptr + fp4_offs_m * (N // 2) + fp4_offs_n, x_fp4x2, mask=fp4_mask)

    padded_r = (pid_m * M_PER_BLOCK + tl.arange(0, M_PER_BLOCK)).to(tl.int64)
    padded_r_2d = padded_r[:, None]
    layout_off = (
        padded_r_2d // 128
    ) * NUM_N_BLOCKS * NUM_ELEM_PER_LAYOUT + pid_n * NUM_ELEM_PER_LAYOUT
    offs_m_in_layout = (padded_r_2d % 128).to(tl.int32)
    scale_offs = layout_off + nvfp4_scale_swizzle(offs_m_in_layout)
    scale_store_offs_n = pid_n * 4 + tl.arange(0, 4)[None, :]
    store_mask = (offs_m < M) & (scale_store_offs_n < (N // 16))
    tl.store(s_ptr + scale_offs, scales.to(tl.uint8, bitcast=True), mask=store_mask)


# ----------------------------------------------------------------------------
# Fusion 1: RoPE -> NVFP4 Q,K  (rotation fused into the quant pass, block-16 on D)
# ----------------------------------------------------------------------------
@triton.jit
def _rope_quant_kernel(
    x_ptr,
    cos_ptr,
    sin_ptr,
    q_ptr,
    s_ptr,
    gscale,
    M,
    N,
    HALF,
    M_PER_BLOCK: tl.constexpr,
    NUM_N_BLOCKS: tl.constexpr,
):
    """RoPE (rotate_half convention) + NVFP4 pack, one [M_PER_BLOCK,64] tile.

    HF rotate_half: out = x*cos + rotate_half(x)*sin, where
    rotate_half(x) = cat(-x[half:], x[:half]). cos/sin are [M, N] (broadcast of the
    [S, D] tables over heads, materialized 2D for the proto — the real epilogue
    would index the [S,D] table by row's seq pos).
    """
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(0)
    offs_m = pid_m * M_PER_BLOCK + tl.arange(0, M_PER_BLOCK)[:, None]
    offs_n = pid_n * 64 + tl.arange(0, 64)[None, :]
    mask = (offs_m < M) & (offs_n < N)

    x = tl.load(x_ptr + offs_m * N + offs_n, mask=mask, other=0.0).to(tl.float32)
    cos = tl.load(cos_ptr + offs_m * N + offs_n, mask=mask, other=0.0).to(tl.float32)
    sin = tl.load(sin_ptr + offs_m * N + offs_n, mask=mask, other=0.0).to(tl.float32)

    # rotate_half: for cols in [0,HALF) partner is +HALF (negated); for [HALF,N) partner is -HALF.
    is_low = offs_n < HALF
    partner_n = tl.where(is_low, offs_n + HALF, offs_n - HALF)
    pmask = (offs_m < M) & (partner_n < N)
    xp = tl.load(x_ptr + offs_m * N + partner_n, mask=pmask, other=0.0).to(tl.float32)
    rot = tl.where(is_low, -xp, xp)
    x_rot = x * cos + rot * sin

    x_blocks = x_rot.reshape(M_PER_BLOCK, 4, 16)
    _quant_emit(
        x_blocks, q_ptr, s_ptr, gscale, pid_m, pid_n, M, N, M_PER_BLOCK, NUM_N_BLOCKS
    )


def fused_rope_quant(x2d, cos2d, sin2d, two_level: bool = False):
    """Rotate a [M, D] tensor (RoPE) and emit it NVFP4-packed in one pass.

    Returns (qdata [M, D/2] float4_e2m1fn_x2, scale [swizzled] e4m3, inv_gs) — the
    same triple as ``_mslk_quantize``, drop-in for ``_mslk_scaled_mm``.

    Default single-level (matches ``_mslk_quantize_sl``, the path the fork uses for
    forward activations): no per-tensor amax pass, so the fusion adds NOTHING beyond
    the rotation the op already does. ``two_level=True`` matches ``_mslk_quantize``
    (computes a global scale) for apples-to-apples parity against it.
    """
    M, N = x2d.shape
    assert N % 64 == 0
    if two_level:
        half = N // 2
        rot = torch.cat([-x2d[:, half:], x2d[:, :half]], dim=1)
        rotated = x2d.float() * cos2d.float() + rot.float() * sin2d.float()
        amax = torch.amax(torch.abs(rotated)).clamp(min=1e-12)
        gscale = (_NVFP4_GLOBAL_AMAX / amax).to(torch.float32)
    else:
        gscale = torch.tensor(1.0, dtype=torch.float32)
    rm, rk = _swizzled_scale_shape(M, N)
    q = x2d.new_empty(M, N // 2, dtype=torch.uint8)
    s = x2d.new_empty(rm, rk, dtype=torch.uint8)
    M_PER_BLOCK = min(triton.next_power_of_2(M), 128)
    grid = (triton.cdiv(N, 64), triton.cdiv(M, M_PER_BLOCK))
    _rope_quant_kernel[grid](
        x2d,
        cos2d,
        sin2d,
        q,
        s,
        float(gscale),
        M,
        N,
        N // 2,
        M_PER_BLOCK=M_PER_BLOCK,
        NUM_N_BLOCKS=triton.cdiv(N, 64),
    )
    return (
        q.view(torch.float4_e2m1fn_x2),
        s.view(torch.float8_e4m3fn),
        (1.0 / gscale).to(x2d.dtype),
    )


# ----------------------------------------------------------------------------
# Fusion 2: softmax -> NVFP4 P  (exp/normalize fused into quant, block-16 on key)
# ----------------------------------------------------------------------------
@triton.jit
def _softmax_quant_kernel(
    scores_ptr,  # [M, N] fp32 pre-softmax scores (scaled + masked)
    q_ptr,
    s_ptr,
    gscale,  # global scale of P (P in [0,1] -> amax<=1; host constant)
    M,
    N,
    BLOCK_N: tl.constexpr,  # next_pow2(N), the full key axis held in registers
    N_BLOCKS: tl.constexpr,  # N // 16  (number of group-16 blocks per row)
):
    """One program per ROW: row max + row sum + exp + NVFP4 pack of P, no gather.

    Softmax needs the whole row for its denominator, so the row lives in registers;
    the normalized row is then reshaped straight into group-16 blocks and packed.
    P >= 0 so amax <= 1; gscale is a host constant (no per-tensor amax pass needed).
    """
    row = tl.program_id(0)
    if row >= M:
        return
    offs = tl.arange(0, BLOCK_N)
    cmask = offs < N
    s = tl.load(scores_ptr + row * N + offs, mask=cmask, other=float("-inf")).to(
        tl.float32
    )
    row_max = tl.max(s, axis=0)
    e = tl.where(cmask, tl.exp(s - row_max), 0.0)
    p = e / tl.sum(e, axis=0)  # [BLOCK_N], zero past N

    NUM_ELEM_PER_LAYOUT: tl.constexpr = 128 * 4
    # reshape the row into [N_BLOCKS, 16] group blocks (BLOCK_N == N here: N is a
    # multiple of 64 so next_pow2 only differs when N isn't a power of 2 — guard via
    # the launch asserting N is the padded width).
    pb = p.reshape(N_BLOCKS, 16)
    block_amax = tl.max(tl.abs(pb), axis=1)  # [N_BLOCKS]
    scales = tl.clamp(tl.div_rn(block_amax, _F4_MAX) * gscale, _E4M3_EPS, _F8E4M3_MAX)
    scales = scales.to(tl.float8e4nv)
    total_scale = tl.div_rn(gscale, scales.to(tl.float32)[:, None])
    pb = pb * total_scale  # [N_BLOCKS, 16]

    # pack: [N_BLOCKS,16] -> [N_BLOCKS*8] bytes (2 fp4/byte), pairs along last dim.
    pairs = pb.reshape(N_BLOCKS * 8, 2).split()
    x_fp4x2 = convert_fp32_to_fp4_packed(pairs)  # [N_BLOCKS*8]
    q_offs = tl.arange(0, BLOCK_N // 2)
    tl.store(q_ptr + row * (N // 2) + q_offs, x_fp4x2, mask=q_offs < N // 2)

    # scale store: this row -> swizzled 128x4 layout, one scale per group block.
    padded_r = tl.full((1,), row, tl.int64)
    sb = tl.arange(0, N_BLOCKS)  # group-block index along N
    pid_n = sb // 4
    in4 = sb % 4
    layout_off = (padded_r // 128) * (
        tl.cdiv(N, 64)
    ) * NUM_ELEM_PER_LAYOUT + pid_n * NUM_ELEM_PER_LAYOUT
    offs_m_in_layout = (padded_r % 128).to(tl.int32)
    # swizzle for a single row m: sub_layout_off + (m//32)*4 + in4
    m = offs_m_in_layout
    sub_layout_off = (m % 32) * 16
    sub_row = m // 32
    scale_offs = layout_off + sub_layout_off + sub_row * 4 + in4
    tl.store(s_ptr + scale_offs, scales.to(tl.uint8, bitcast=True), mask=sb < N_BLOCKS)


def fused_softmax_quant(scores: torch.Tensor, out_dtype=torch.bfloat16):
    """Softmax(scores, dim=-1) and emit P NVFP4-packed in one pass. scores: [M, N]."""
    M, N = scores.shape
    assert N % 64 == 0, "N must be a multiple of 64 (already padded in attention)"
    assert N & (N - 1) == 0 or N % 16 == 0
    gscale = float(_NVFP4_GLOBAL_AMAX / 1.0)
    rm, rk = _swizzled_scale_shape(M, N)
    q = torch.empty(M, N // 2, dtype=torch.uint8, device=scores.device)
    s = torch.empty(rm, rk, dtype=torch.uint8, device=scores.device)
    BLOCK_N = triton.next_power_of_2(N)
    grid = (M,)
    _softmax_quant_kernel[grid](
        scores,
        q,
        s,
        gscale,
        M,
        N,
        BLOCK_N=BLOCK_N,
        N_BLOCKS=N // 16,
    )
    inv_gs = torch.tensor(1.0 / gscale, device=scores.device, dtype=out_dtype)
    return q.view(torch.float4_e2m1fn_x2), s.view(torch.float8_e4m3fn), inv_gs


# ----------------------------------------------------------------------------
# Fusion 3: v_proj epilogue -> NVFP4 V (identity transform; block-16 on key axis)
# ----------------------------------------------------------------------------
@triton.jit
def _identity_quant_kernel(
    x_ptr,
    q_ptr,
    s_ptr,
    gscale,
    M,
    N,
    M_PER_BLOCK: tl.constexpr,
    NUM_N_BLOCKS: tl.constexpr,
):
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(0)
    offs_m = pid_m * M_PER_BLOCK + tl.arange(0, M_PER_BLOCK)[:, None]
    offs_n = pid_n * 64 + tl.arange(0, 64)[None, :]
    mask = (offs_m < M) & (offs_n < N)
    x = tl.load(x_ptr + offs_m * N + offs_n, mask=mask, other=0.0).to(tl.float32)
    x_blocks = x.reshape(M_PER_BLOCK, 4, 16)
    _quant_emit(
        x_blocks, q_ptr, s_ptr, gscale, pid_m, pid_n, M, N, M_PER_BLOCK, NUM_N_BLOCKS
    )


def fused_vproj_quant(v2d: torch.Tensor, two_level: bool = False):
    """Emit a v_proj output [M, K] NVFP4-packed along K in one pass (epilogue stand-in).

    V gets no RoPE, so the v_proj GEMM epilogue can emit this directly. Here the
    transform is identity (the proto's job is to show the pack is free in the
    epilogue); in-tree it would replace the bf16 store at the end of v_proj.
    """
    M, N = v2d.shape
    assert N % 64 == 0
    if two_level:
        amax = torch.amax(torch.abs(v2d.float())).clamp(min=1e-12)
        gscale = (_NVFP4_GLOBAL_AMAX / amax).to(torch.float32)
    else:
        gscale = torch.tensor(1.0, dtype=torch.float32)
    rm, rk = _swizzled_scale_shape(M, N)
    q = v2d.new_empty(M, N // 2, dtype=torch.uint8)
    s = v2d.new_empty(rm, rk, dtype=torch.uint8)
    M_PER_BLOCK = min(triton.next_power_of_2(M), 128)
    grid = (triton.cdiv(N, 64), triton.cdiv(M, M_PER_BLOCK))
    _identity_quant_kernel[grid](
        v2d,
        q,
        s,
        float(gscale),
        M,
        N,
        M_PER_BLOCK=M_PER_BLOCK,
        NUM_N_BLOCKS=triton.cdiv(N, 64),
    )
    return (
        q.view(torch.float4_e2m1fn_x2),
        s.view(torch.float8_e4m3fn),
        (1.0 / gscale).to(v2d.dtype),
    )
