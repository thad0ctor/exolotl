"""Fused NVFP4 producers in the ``tl.dot_scaled`` row-major layout.

These feed the packed-input path of ``attn_nvfp4_flash.nvfp4_flash_attention_packed``
so no standalone pre-quant HBM round-trip of Q/K/V is paid: the NVFP4 pack rides
along with the op that already touches the operand.

  * ``fused_rope_quant_qk``    — partial RoPE (Qwen3.5 rotate_half, ``rotary_dim`` of
    ``head_dim``) on Q,K + NVFP4 pack along head_dim, one pass. The pass-through
    tail (``head_dim - rotary_dim`` dims) is copied unrotated then quantized with
    the rotated head, so the whole head_dim is one contiguous packed row.
  * ``quant_v_keyaxis``        — V (no RoPE) packed along the KEY axis (V^T
    ``[.,D,Skv]`` layout the PV-GEMM consumes), via strided reads (no transpose
    copy). This is the v_proj epilogue stand-in; it reads V once and emits packed.

Layout (matches attn_nvfp4_flash._quant_nvfp4 / tl.dot_scaled, e4m3 group-16):
  * packed operand : ``[rows, K//2]`` uint8  (2 e2m1 nibbles/byte, low first)
  * scale          : ``[rows, K//16]`` float8_e4m3fn  (row-major, NOT swizzled)

The quant core (block amax -> e4m3 scale -> hw FP4 pack) is identical to
``attn_nvfp4_flash._quant_nvfp4_kernel``; only the per-tile transform differs (RoPE
for Q/K, identity for V). Single-level scaling (no per-tensor global amax), matching
what the flash kernel's own pre-quant uses, so parity is apples-to-apples.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl
from sageattention.nvfp4 import convert_fp32_to_fp4_packed

_E4M3_EPS = tl.constexpr(1.5258789e-05)
_F8E4M3_MAX = tl.constexpr(448.0)
_F4_MAX = tl.constexpr(6.0)


# ---------------------------------------------------------------------------
# Fused partial-RoPE -> NVFP4 pack of Q (or K), quantized along head_dim D.
# Grid: (Z*H, cdiv(S, BLOCK_R)). One program owns a [BLOCK_R, D] tile of one head.
# cos/sin are [Z, S, rotary_dim] (already mrope-resolved by the HF rotary module);
# they are broadcast over heads and indexed by the seq row.
# ---------------------------------------------------------------------------
@triton.jit
def _rope_quant_kernel(
    x_ptr,  # [Z*H, S, D]  high-precision (normed) Q or K
    cos_ptr,
    sin_ptr,  # [Z, S, ROT]  (ROT = rotary_dim)
    q_ptr,
    s_ptr,  # [Z*H, S, D//2] uint8, [Z*H, S, D//16] e4m3
    hp_ptr,  # [Z*H, S, D] roped high-precision out (STORE_HP only; dummy else)
    Z,
    H,
    S,
    s_xz,
    s_xh,
    s_xr,  # x: per-z, per-h, per-row(seq) strides; col(D) stride = 1
    s_cz,
    s_cr,  # cos/sin: per-z stride, per-row stride; col stride = 1
    s_qn,
    s_qr,  # q packed: per-(z*h) stride, per-row stride
    s_sn,
    s_sr,  # scale: per-(z*h) stride, per-row stride
    s_hn,
    s_hr,  # hp out: per-(z*h) stride, per-row stride
    D: tl.constexpr,
    ROT: tl.constexpr,
    HALF: tl.constexpr,
    STORE_HP: tl.constexpr,
    BLOCK_R: tl.constexpr,
    DP2: tl.constexpr,
    DP16: tl.constexpr,
    NG: tl.constexpr,
):
    pid_n = tl.program_id(0)  # z*h
    pid_r = tl.program_id(1)
    z = pid_n // H
    h = pid_n % H

    offs_r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
    rmask = offs_r < S
    offs_d = tl.arange(0, D)

    xbase = z * s_xz + h * s_xh
    x = tl.load(
        x_ptr + xbase + offs_r[:, None] * s_xr + offs_d[None, :],
        mask=rmask[:, None],
        other=0.0,
    ).to(tl.float32)

    # partial RoPE on the first ROT dims; tail [ROT, D) passes through unrotated.
    # rotate_half over the ROT block: low half partner is +HALF (negated), high -HALF.
    is_rot = offs_d < ROT
    is_low = offs_d < HALF
    partner = tl.where(is_low, offs_d + HALF, offs_d - HALF)
    xp = tl.load(
        x_ptr + xbase + offs_r[:, None] * s_xr + partner[None, :],
        mask=rmask[:, None] & (partner[None, :] < ROT),
        other=0.0,
    ).to(tl.float32)
    rot = tl.where(is_low, -xp, xp)

    cbase = z * s_cz
    # cos/sin only defined for d < ROT; clamp the col index, gate by is_rot.
    cd = tl.where(is_rot, offs_d, 0)
    cos = tl.load(
        cos_ptr + cbase + offs_r[:, None] * s_cr + cd[None, :],
        mask=rmask[:, None],
        other=0.0,
    ).to(tl.float32)
    sin = tl.load(
        sin_ptr + cbase + offs_r[:, None] * s_cr + cd[None, :],
        mask=rmask[:, None],
        other=0.0,
    ).to(tl.float32)

    x_rot = tl.where(is_rot[None, :], x * cos + rot * sin, x)

    if STORE_HP:
        # One extra store per element: the roped high-precision tile rides along
        # for the training path (the HP backward needs roped q/k saved).
        hp = x_rot.to(hp_ptr.dtype.element_ty)
        tl.store(
            hp_ptr + pid_n * s_hn + offs_r[:, None] * s_hr + offs_d[None, :],
            hp,
            mask=rmask[:, None],
        )
        # Pack from the SAME (dtype-rounded) values the backward will repack:
        # this makes the emitted packs bit-identical to ``_quant_nvfp4(hp)``,
        # so forward out/LSE/grads exactly match the standalone-quant path fed
        # the same hp tensor.
        x_rot = hp.to(tl.float32)

    # NVFP4 pack along D, group-16, single-level scale.
    xb = x_rot.reshape(BLOCK_R, NG, 16)
    amax = tl.max(tl.abs(xb), axis=2)
    sc = tl.clamp(amax / _F4_MAX, _E4M3_EPS, _F8E4M3_MAX).to(tl.float8e4nv)
    xn = xb / sc.to(tl.float32)[:, :, None]
    pairs = xn.reshape(BLOCK_R * DP2, 2).split()
    qpk = convert_fp32_to_fp4_packed(pairs).reshape(BLOCK_R, DP2)

    offs_qk = tl.arange(0, DP2)
    tl.store(
        q_ptr + pid_n * s_qn + offs_r[:, None] * s_qr + offs_qk[None, :],
        qpk,
        mask=rmask[:, None],
    )
    offs_sk = tl.arange(0, DP16)
    tl.store(
        s_ptr + pid_n * s_sn + offs_r[:, None] * s_sr + offs_sk[None, :],
        sc.to(tl.uint8, bitcast=True),
        mask=rmask[:, None],
    )


def fused_rope_quant_qk(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    store_hp: bool = False,
):
    """Apply Qwen3.5 partial RoPE to ``x`` and emit it NVFP4-packed (along D).

    Args:
        x: ``[Z, H, S, D]`` high precision (already q_norm/k_norm'd), D in {128,256}.
        cos, sin: ``[Z, S, rotary_dim]`` (mrope-resolved); rotary_dim <= D, even.
        store_hp: ALSO write the roped high-precision tensor (one extra store per
            element, same pass) — the training packed path needs roped hp q/k
            saved for the HP backward. The packs are then quantized from the
            dtype-rounded hp values, so they are bit-identical to
            ``_quant_nvfp4`` of the returned hp tensor.

    Returns:
        ``(packed uint8 [Z*H, S, D//2], scale float8_e4m3fn [Z*H, S, D//16])``
        in the row-major tl.dot_scaled layout, plus the roped hp tensor
        ``[Z, H, S, D]`` (x.dtype, contiguous) when ``store_hp``.
    """
    z, h, s, d = x.shape
    rot = cos.shape[-1]
    assert d % 16 == 0 and rot % 2 == 0 and rot <= d
    # Read x in whatever layout it arrives (typically a [Z,S,H,D].transpose(1,2)
    # view → non-contiguous, but D stays contiguous), so we DON'T pay a full bf16
    # copy here. The kernel only needs the head_dim (D) unit-stride; if some caller
    # passes a non-D-contiguous x, fall back to a copy (correctness over speed).
    if x.stride(3) != 1:
        x = x.contiguous()
    cos = cos.contiguous()
    sin = sin.contiguous()

    q = x.new_empty(z * h, s, d // 2, dtype=torch.uint8)
    sc = x.new_empty(z * h, s, d // 16, dtype=torch.uint8)
    hp = x.new_empty(z * h, s, d) if store_hp else q  # dummy ptr when not stored

    BLOCK_R = 64
    # Triton's default launch heuristic badly under-warps the d256 rope+quant
    # kernel (the big head_dim doubles every tile plane): on the 5090 (sm_120)
    # num_warps=8/num_stages=3 is ~1.7-1.8x over the default for both store_hp
    # paths. d128 is already near-optimal on the default heuristic.
    _nw = 8 if d >= 256 else None
    _ns = 3 if d >= 256 else None
    if os.environ.get("NVFP4_PROD_OVERRIDE"):  # sweep instrumentation

        def _env_pos_int(key, default):
            # parse a positive int from env; ignore missing/malformed/non-positive
            raw = os.environ.get(key)
            if raw is None:
                return default
            try:
                val = int(raw)
            except ValueError:
                return default
            return val if val > 0 else default

        BLOCK_R = _env_pos_int("NVFP4_PROD_BR", BLOCK_R)
        _nw = _env_pos_int("NVFP4_PROD_W", _nw)
        _ns = _env_pos_int("NVFP4_PROD_S", _ns)
    grid = (z * h, triton.cdiv(s, BLOCK_R))
    _extra = {}
    if _nw:
        _extra["num_warps"] = _nw
    if _ns:
        _extra["num_stages"] = _ns
    _rope_quant_kernel[grid](
        x,
        cos,
        sin,
        q,
        sc,
        hp,
        z,
        h,
        s,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        cos.stride(0),
        cos.stride(1),
        q.stride(0),
        q.stride(1),
        sc.stride(0),
        sc.stride(1),
        hp.stride(0) if store_hp else 0,
        hp.stride(1) if store_hp else 0,
        D=d,
        ROT=rot,
        HALF=rot // 2,
        STORE_HP=store_hp,
        BLOCK_R=BLOCK_R,
        DP2=d // 2,
        DP16=d // 16,
        NG=d // 16,
        **_extra,
    )
    if store_hp:
        return q, sc.view(torch.float8_e4m3fn), hp.view(z, h, s, d)
    return q, sc.view(torch.float8_e4m3fn)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class _FusedRopeQuantQK(torch.autograd.Function):
    """Differentiable fused RoPE+NVFP4-pack producer for the TRAINING path.

    Forward runs the one-pass fused kernel with ``store_hp=True`` (roped hp out
    + packs). Backward applies the exact RoPE transpose to the hp gradient
    (``dx = dy*cos - rotate_half(dy*sin)`` on the rotary dims, identity on the
    pass-through tail) — the same math autograd derives for the stock HF
    ``apply_rotary_pos_emb``. The packs are non-differentiable (uint8); the
    attention function returns its q/k grads w.r.t. the roped hp tensors.
    """

    @staticmethod
    def forward(ctx, x, cos, sin):
        qnv, qsc, hp = fused_rope_quant_qk(x, cos, sin, store_hp=True)
        ctx.save_for_backward(cos, sin)
        ctx.mark_non_differentiable(qnv, qsc)
        return qnv, qsc, hp

    @staticmethod
    def backward(ctx, _g_qnv, _g_qsc, g_hp):
        cos, sin = ctx.saved_tensors
        rot = cos.shape[-1]
        c = cos.unsqueeze(1)  # [Z, 1, S, ROT] broadcast over heads
        s = sin.unsqueeze(1)
        g_rot = g_hp[..., :rot]
        dx_rot = g_rot * c - _rotate_half(g_rot * s)
        if rot == g_hp.shape[-1]:
            return dx_rot, None, None
        return torch.cat((dx_rot, g_hp[..., rot:]), dim=-1), None, None


def fused_rope_quant_qk_hp(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Differentiable ``fused_rope_quant_qk(..., store_hp=True)``.

    Returns ``(qnv, qsc, hp_roped)``; gradients flow from ``hp_roped`` back to
    ``x`` through the exact RoPE transpose (see ``_FusedRopeQuantQK``).
    """
    qnv, qsc, hp = _FusedRopeQuantQK.apply(x, cos, sin)
    return qnv, qsc.view(torch.float8_e4m3fn), hp


def quant_v_keyaxis(
    value: torch.Tensor, block_n: int = 128
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Pack V (no RoPE) along the KEY axis -> V^T tl.dot_scaled layout.

    The v_proj epilogue stand-in: reads V ``[Z, Hk, Skv, D]`` once via strided reads
    (no transpose copy) and emits ``vnv [Z*Hk, D, Skv_pad//2]`` /
    ``vsc [Z*Hk, D, Skv_pad//16]`` (e4m3), key axis padded to a multiple of block_n.

    Returns (vnv, vsc, Skv_pad).
    """
    from axolotl.kernels.attn_nvfp4_flash import _next_mult, _quant_nvfp4

    z, hk, s_kv, d = value.shape
    v2 = value.reshape(z * hk, s_kv, d)
    s_kv_pad = _next_mult(s_kv, block_n)
    vnv, vsc = _quant_nvfp4(v2, transpose=True, k_pad=s_kv_pad)
    return vnv, vsc, s_kv_pad


# ---------------------------------------------------------------------------
# Fused v_proj GEMM with a key-axis NVFP4-pack epilogue. One program owns a
# [BLOCK_S, D] output tile of one (z, kv-head): it runs y = x @ Wv^T as a native
# NVFP4 tl.dot_scaled GEMM (x packed along K=hidden once by the caller, Wv prepacked
# per head), then packs the [BLOCK_S, D] result along the SEQ axis (group-16, the key
# axis the PV-GEMM contracts) and writes it transposed to vnv[zhk, D, S_pad//2] /
# vsc[zhk, D, S_pad//16]. This collapses {bf16 v_proj output + transpose + standalone
# key-axis quant read} into one pass: V never materializes in bf16 HBM and the quant
# is the GEMM's own epilogue. BLOCK_S must be a multiple of 16 (the pack group).
# Padded seq rows beyond S land in [S, S_pad): they get amax 0 -> eps scale / zero
# packed nibbles, contributing nothing to the eps-scaled PV columns.
# ---------------------------------------------------------------------------
@triton.jit
def _vproj_pack_keyaxis_kernel(
    xnv_ptr,
    xsc_ptr,  # [Z, S, K//2] uint8, [Z, S, K//16] e4m3  (activation)
    wnv_ptr,
    wsc_ptr,  # [HK*D, K//2] uint8, [HK*D, K//16] e4m3  (Wv, [HK*D,K])
    vnv_ptr,
    vsc_ptr,  # [Z*HK, D, S_pad//2] uint8, [Z*HK, D, S_pad//16] e4m3
    S,
    S_pad,
    K,
    sx_z,
    sx_s,  # x packed: per-z, per-row(seq) strides
    ssc_z,
    ssc_s,  # x scale: per-z, per-row strides
    sw_n,  # weight packed row stride (= K//2); scale row = K//16
    sv_d,
    svsc_d,  # vnv/vsc per-row(D) strides (= S_pad//2, S_pad//16)
    HK: tl.constexpr,
    D: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_K: tl.constexpr,
    KP2: tl.constexpr,
    KP16: tl.constexpr,
    SP2: tl.constexpr,
    SP16: tl.constexpr,  # BLOCK_S//2, BLOCK_S//16
):
    pid_z = tl.program_id(0)
    pid_hk = tl.program_id(1)
    pid_s = tl.program_id(2)
    zhk = pid_z * HK + pid_hk

    offs_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    smask = offs_s < S
    offs_d = tl.arange(0, D)  # this head's D output cols = W rows [hk*D, (hk+1)*D)
    wrow = pid_hk * D + offs_d

    acc = tl.zeros((BLOCK_S, D), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offk2 = k0 // 2 + tl.arange(0, KP2)
        offk16 = k0 // 16 + tl.arange(0, KP16)
        a = tl.load(
            xnv_ptr + pid_z * sx_z + offs_s[:, None] * sx_s + offk2[None, :],
            mask=smask[:, None],
            other=0,
        )
        asc = tl.load(
            xsc_ptr + pid_z * ssc_z + offs_s[:, None] * ssc_s + offk16[None, :],
            mask=smask[:, None],
            other=0,
        ).to(tl.float8e4nv, bitcast=True)
        w = tl.load(wnv_ptr + wrow[:, None] * sw_n + offk2[None, :])
        wsc = tl.load(
            wsc_ptr + wrow[:, None] * (K // 16) + offk16[None, :],
        ).to(tl.float8e4nv, bitcast=True)
        acc = tl.dot_scaled(a, asc, "e2m1", w.T, wsc, "e2m1", acc=acc)

    # zero the padded seq rows so they pack to amax 0 (eps scale, zero nibbles).
    acc = tl.where(smask[:, None], acc, 0.0)

    # pack along the SEQ axis (group-16): transpose the [BLOCK_S, D] tile to [D, BLOCK_S]
    # so groups of 16 run down the seq axis, matching the V^T key-axis layout.
    accT = tl.trans(acc)  # [D, BLOCK_S]
    NG: tl.constexpr = BLOCK_S // 16
    xb = accT.reshape(D, NG, 16)
    amax = tl.max(tl.abs(xb), axis=2)
    sc = tl.clamp(amax / _F4_MAX, _E4M3_EPS, _F8E4M3_MAX).to(tl.float8e4nv)
    xn = xb / sc.to(tl.float32)[:, :, None]
    pairs = xn.reshape(D * SP2, 2).split()
    qpk = convert_fp32_to_fp4_packed(pairs).reshape(D, SP2)

    offs_sp = pid_s * SP2 + tl.arange(0, SP2)
    tl.store(
        vnv_ptr + zhk * (D * sv_d) + offs_d[:, None] * sv_d + offs_sp[None, :],
        qpk,
    )
    offs_ssc = pid_s * SP16 + tl.arange(0, SP16)
    tl.store(
        vsc_ptr + zhk * (D * svsc_d) + offs_d[:, None] * svsc_d + offs_ssc[None, :],
        sc.to(tl.uint8, bitcast=True),
    )


def prepack_vproj_weight(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack a v_proj weight ``[HK*D, hidden]`` to NVFP4 once (along hidden=K).

    Returns ``(wnv [HK*D, K//2] uint8, wsc [HK*D, K//16] e4m3)`` in the row-major
    tl.dot_scaled layout. K (hidden) must be a multiple of 16.
    """
    from axolotl.kernels.attn_nvfp4_flash import _quant_nvfp4

    n, k = weight.shape
    assert k % 16 == 0
    wnv, wsc = _quant_nvfp4(weight.unsqueeze(0).contiguous())
    return wnv[0], wsc[0]


def fused_vproj_quant_v_keyaxis(
    hidden_states: torch.Tensor,  # [Z, S, hidden]
    wnv: torch.Tensor,  # [HK*D, hidden//2] uint8  (prepacked v_proj weight)
    wsc: torch.Tensor,  # [HK*D, hidden//16] e4m3
    hk: int,
    d: int,
    block_n: int = 128,
    block_s: int = 64,
    block_k: int = 256,
    num_warps: int = 8,
    num_stages: int = 3,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Fused native-NVFP4 v_proj GEMM emitting V already key-axis-packed (V^T layout).

    Computes ``V = hidden_states @ Wv^T`` in NVFP4 and packs the result along the
    SEQ (key) axis in the kernel's epilogue — no bf16 V, no transpose copy, no
    standalone quant read. Returns ``(vnv [Z*HK, D, S_pad//2], vsc [Z*HK, D,
    S_pad//16] e4m3, S_pad)`` matching ``quant_v_keyaxis`` (S_pad a multiple of
    ``block_n``). The activation is quantized along hidden once (single-level,
    group-16). ``block_s`` is the pack group span and must be a multiple of 16.
    """
    from axolotl.kernels.attn_nvfp4_flash import _next_mult, _quant_nvfp4

    z, s, k = hidden_states.shape
    assert k % 16 == 0 and block_s % 16 == 0
    s_pad = _next_mult(s, block_n)

    # quantize x along K once (shared across heads).
    xnv, xsc = _quant_nvfp4(hidden_states.reshape(z, s, k))
    xsc = xsc.view(torch.uint8)

    vnv = hidden_states.new_empty(z * hk, d, s_pad // 2, dtype=torch.uint8)
    vsc = hidden_states.new_zeros(z * hk, d, s_pad // 16, dtype=torch.uint8)

    grid = (z, hk, triton.cdiv(s, block_s))
    _vproj_pack_keyaxis_kernel[grid](
        xnv.view(torch.uint8),
        xsc,
        wnv.view(torch.uint8),
        wsc.view(torch.uint8),
        vnv,
        vsc,
        s,
        s_pad,
        k,
        xnv.stride(0),
        xnv.stride(1),
        xsc.stride(0),
        xsc.stride(1),
        wnv.stride(0),
        vnv.stride(1),
        vsc.stride(1),
        HK=hk,
        D=d,
        BLOCK_S=block_s,
        BLOCK_K=block_k,
        KP2=block_k // 2,
        KP16=block_k // 16,
        SP2=block_s // 2,
        SP16=block_s // 16,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return vnv, vsc.view(torch.float8_e4m3fn), s_pad
