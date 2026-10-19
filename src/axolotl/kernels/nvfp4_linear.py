"""Native-NVFP4 linear GEMM via Triton ``tl.dot_scaled`` (e2m1 + e4m3 group-16).

Forward-only ``y = x @ W^T`` where both operands are packed to NVFP4 (the same
quantize-once layout the flash attention forward uses, see attn_nvfp4_flash):

  * packed operand : ``[rows, K//2]`` uint8  (2 e2m1 nibbles/byte, low first)
  * scale          : ``[rows, K//16]`` float8_e4m3fn  (row-major, NOT swizzled)

The weight is packed ONCE at patch time (``prepack_weight_nvfp4``) and cached; only
the activation is quantized per call (fused into the quant pre-pass). This targets
the large-K (K=hidden) projections of Qwen3.5's GatedDeltaNet (in_proj_qkv / in_proj_z
/ out_proj), where K=4096 makes the FP4 tensor-core path a clear win over bf16 and the
small per-call activation quant is amortized by the GEMM.

NVFP4 (e4m3 group-16), NOT MXFP4: passing a float8_e4m3fn scale to ``tl.dot_scaled``
emits ``mma.sync...kind::mxf4nvf4...ue4m3``; a uint8 scale would silently degrade to
MXFP4/e8m0. Single-level scaling (per-group amax -> e4m3 scale, no per-tensor global
scale) to match the flash forward's own pre-quant so parity is apples-to-apples.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from axolotl.kernels.attn_nvfp4_flash import _quant_nvfp4


@triton.jit
def _nvfp4_gemm_kernel(
    anv_ptr,
    asc_ptr,  # [M, K//2] uint8, [M, K//16] e4m3 (activation)
    wnv_ptr,
    wsc_ptr,  # [N, K//2] uint8, [N, K//16] e4m3 (weight, W[N,K])
    out_ptr,  # [M, N] out_dtype
    M,
    N,
    K,
    s_am,
    s_wn,
    s_om,  # row strides (col stride = 1)
    s_asc,
    s_wsc,  # scale row strides
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    KP2: tl.constexpr = BLOCK_K // 2
    KP16: tl.constexpr = BLOCK_K // 16
    mmask = offs_m < M
    nmask = offs_n < N
    for k0 in range(0, K, BLOCK_K):
        offk2 = k0 // 2 + tl.arange(0, KP2)
        offk16 = k0 // 16 + tl.arange(0, KP16)
        a = tl.load(
            anv_ptr + offs_m[:, None] * s_am + offk2[None, :],
            mask=mmask[:, None],
            other=0,
        )
        asc = tl.load(
            asc_ptr + offs_m[:, None] * s_asc + offk16[None, :],
            mask=mmask[:, None],
            other=0,
        ).to(tl.float8e4nv, bitcast=True)
        w = tl.load(
            wnv_ptr + offs_n[:, None] * s_wn + offk2[None, :],
            mask=nmask[:, None],
            other=0,
        )
        wsc = tl.load(
            wsc_ptr + offs_n[:, None] * s_wsc + offk16[None, :],
            mask=nmask[:, None],
            other=0,
        ).to(tl.float8e4nv, bitcast=True)
        acc = tl.dot_scaled(a, asc, "e2m1", w.T, wsc, "e2m1", acc=acc)

    tl.store(
        out_ptr + offs_m[:, None] * s_om + offs_n[None, :],
        acc.to(out_ptr.dtype.element_ty),
        mask=mmask[:, None] & nmask[None, :],
    )


def prepack_weight_nvfp4(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack an ``nn.Linear`` weight ``[N, K]`` to NVFP4 once (along K).

    Returns ``(wnv [N, K//2] uint8, wsc [N, K//16] e4m3)`` in the row-major
    tl.dot_scaled layout. K must be a multiple of 16.
    """
    n, k = weight.shape
    assert k % 16 == 0, "weight in_features (K) must be a multiple of 16"
    wnv, wsc = _quant_nvfp4(weight.unsqueeze(0).contiguous())
    return wnv[0], wsc[0]


def nvfp4_linear(
    x: torch.Tensor,
    wnv: torch.Tensor,
    wsc: torch.Tensor,
    n_out: int,
    block_m: int = 128,
    block_n: int = 128,
    block_k: int = 256,
    num_warps: int = 8,
    num_stages: int = 3,
) -> torch.Tensor:
    """``y = x @ W^T`` in native NVFP4. ``x`` is ``[..., K]`` (high precision),
    ``wnv``/``wsc`` are the pre-packed weight (``[N, K//2]`` / ``[N, K//16]``).

    The activation is quantized along K per call (single-level, group-16) then the
    GEMM runs as a ``tl.dot_scaled`` FP4 tensor-core op. Returns ``[..., N]`` in
    ``x.dtype``. K must be a multiple of 16.
    """
    out_dtype = x.dtype
    lead = x.shape[:-1]
    k = x.shape[-1]
    x2d = x.reshape(-1, k)
    m = x2d.shape[0]

    anv, asc = _quant_nvfp4(x2d.unsqueeze(0))
    anv = anv[0]
    asc = asc[0]

    out = torch.empty(m, n_out, device=x.device, dtype=out_dtype)
    grid = (triton.cdiv(m, block_m), triton.cdiv(n_out, block_n))
    _nvfp4_gemm_kernel[grid](
        anv.view(torch.uint8),
        asc.view(torch.uint8),
        wnv.view(torch.uint8),
        wsc.view(torch.uint8),
        out,
        m,
        n_out,
        k,
        anv.stride(0),
        wnv.stride(0),
        out.stride(0),
        asc.stride(0),
        wsc.stride(0),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out.reshape(*lead, n_out)
