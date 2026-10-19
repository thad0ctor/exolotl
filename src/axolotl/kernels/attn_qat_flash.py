# ruff: noqa: E741
"""Fused FlashAttention-style NVFP4 fake-quant attention (Attn-QAT, arXiv 2603.00040).

Long-context follow-up to the eager v1 in ``axolotl.utils.attn_qat``: instead of
materializing the full probability matrix P (O(seq^2) memory), this runs an online
softmax (FlashAttention) so memory stays linear in sequence length, while applying the
SAME NVFP4 fake-quant (block-16 along head_dim, amax/6 e4m3 block scale, E2M1
round-to-nearest-even, dequant; STE identity gradient) to Q, K, V and the attention
probabilities P. It is numerically equivalent to the eager v1 (differences are only
online-softmax reassociation plus FP4 rounding).

Kernel structure is adapted from the Triton tutorial ``06-fused-attention``
(https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html),
Tri Dao's FlashAttention-2 backward, with the paper's Algorithms 2 & 3 inserted:

  Forward (per Q-block i, streaming K/V-block j, online softmax):
    Qf,Kf,Vf = fake_quant(Q),fake_quant(K),fake_quant(V)   (block-16 over head_dim)
    S_ij = Qf_i . Kf_j^T * scale   (+ causal mask + per-key padding bias)
    online max/denom update
    P_ij  = exp(S_ij - m_i)
    Pf_ij = fake_quant(P_ij) over the KEY axis (block-16, trailing partial block padded)
    O_i  += Pf_ij . Vf_j        (low-precision result, returned)
    O'_i += P_ij  . Vf_j        (HIGH-precision P, kept only for the backward, eq. 9)

  Backward (Algorithm 3):
    recompute S/P with fake_quant so P matches the forward (paper's key point)
    D_i  = rowsum(dO_i * O'_i)   (eq. 9: P^T dP = dO^T O', uses the high-precision O')
    dS_i = P_i * dP_i - D_i * P_i   (eq. 8, high-precision softmax grad)
    accumulate dQ, dK, dV

Because every fake-quant is an STE (identity gradient), dQ/dK/dV equal the standard
attention gradient on the fake-quantized forward, i.e. they match the eager v1's
autograd gradients within FP4/reassociation tolerance.

NVFP4 fake-quant numerics here are bit-exact with the eager v1's torchao round-trip
(single-level scaling path of ``NVFP4Tensor.to_nvfp4`` + ``dequantize``): verified
block scale = clamp(amax/6, E4M3_EPS, 448) cast to e4m3, E2M1 round-half-to-even.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:  # pragma: no cover
    HAS_TRITON = False


_BLOCK_SIZE = 16  # NVFP4 block
_F4_MAX = 6.0  # E2M1 max magnitude
_E4M3_EPS = 0.015625  # smallest normal e4m3 = 2**-6 (torch.finfo(float8_e4m3fn).tiny)
_E4M3_MAX = 448.0


if HAS_TRITON:
    _BLK = tl.constexpr(16)
    _F4MAX = tl.constexpr(6.0)
    _EPS = tl.constexpr(0.015625)
    _EMAX = tl.constexpr(448.0)

    @triton.jit
    def _round_e2m1(x):
        """Round |x| in [0,6] to the E2M1 grid with round-half-to-even, keep sign.

        Grid: {0,0.5,1,1.5,2,3,4,6}. Even-mantissa (round-to on tie) values are
        {0,1,2,4}; odd ones {0.5,1.5,3,6}. Implemented branch-free with the same
        midpoints/tie resolution the bit-trick in torchao produces.
        """
        s = tl.where(x >= 0, 1.0, -1.0)
        a = tl.abs(x)
        # nearest grid point by midpoint thresholds
        r = tl.where(
            a < 0.25,
            0.0,
            tl.where(
                a < 0.75,
                0.5,
                tl.where(
                    a < 1.25,
                    1.0,
                    tl.where(
                        a < 1.75,
                        1.5,
                        tl.where(
                            a < 2.5,
                            2.0,
                            tl.where(a < 3.5, 3.0, tl.where(a < 5.0, 4.0, 6.0)),
                        ),
                    ),
                ),
            ),
        )
        # ties-to-even at the midpoints
        r = tl.where(a == 0.25, 0.0, r)
        r = tl.where(a == 0.75, 1.0, r)
        r = tl.where(a == 1.25, 1.0, r)
        r = tl.where(a == 1.75, 2.0, r)
        r = tl.where(a == 2.5, 2.0, r)
        r = tl.where(a == 3.5, 4.0, r)
        r = tl.where(a == 5.0, 4.0, r)
        return s * r

    @triton.jit
    def _fake_quant_blocks(x, BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr):
        """NVFP4 fake-quant of a [BLOCK_M, HEAD_DIM] tile over head_dim, block 16.

        HEAD_DIM must be a multiple of 16. Mirrors the eager single-level path:
        scale = clamp(amax/6, eps, 448) -> e4m3 -> f32; clamp(x/scale, -6, 6) ->
        E2M1 round -> * scale.
        """
        xf = x.to(tl.float32)
        nblk: tl.constexpr = HEAD_DIM // _BLK
        xb = tl.reshape(xf, (BLOCK_M, nblk, _BLK))
        amax = tl.max(tl.abs(xb), axis=2)
        scale = amax / _F4MAX
        scale = tl.clamp(scale, _EPS, _EMAX)
        scale = scale.to(tl.float8e4nv).to(tl.float32)
        scaled = xb / scale[:, :, None]
        scaled = tl.clamp(scaled, -_F4MAX, _F4MAX)
        q = _round_e2m1(scaled)
        deq = q * scale[:, :, None]
        return tl.reshape(deq, (BLOCK_M, HEAD_DIM))

    @triton.jit
    def _fake_quant_p(p, n_valid, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
        """NVFP4 fake-quant of a [BLOCK_M, BLOCK_N] probability tile over the KEY
        axis, block 16. BLOCK_N is a multiple of 16; ``n_valid`` columns are real,
        the rest are zero (already-masked padding) and form their own partial
        block so they do not perturb real blocks' scales -- matches the eager
        zero-pad-and-slice."""
        nblk: tl.constexpr = BLOCK_N // _BLK
        col = tl.arange(0, BLOCK_N)
        pv = tl.where(col[None, :] < n_valid, p, 0.0).to(tl.float32)
        pb = tl.reshape(pv, (BLOCK_M, nblk, _BLK))
        amax = tl.max(tl.abs(pb), axis=2)
        scale = amax / _F4MAX
        scale = tl.clamp(scale, _EPS, _EMAX)
        scale = scale.to(tl.float8e4nv).to(tl.float32)
        scaled = pb / scale[:, :, None]
        scaled = tl.clamp(scaled, -_F4MAX, _F4MAX)
        q = _round_e2m1(scaled)
        deq = q * scale[:, :, None]
        return tl.reshape(deq, (BLOCK_M, BLOCK_N))

    @triton.jit
    def _attn_qat_fwd(
        Q,
        K,
        V,
        sm_scale,
        B,
        O,
        Op,
        M,  # noqa: E741
        stride_qz,
        stride_qh,
        stride_qm,
        stride_qk,
        stride_kz,
        stride_kh,
        stride_kn,
        stride_kk,
        stride_vz,
        stride_vh,
        stride_vn,
        stride_vk,
        stride_oz,
        stride_oh,
        stride_om,
        stride_ok,
        stride_bz,
        Z,
        H,
        N_CTX,
        N_KV,
        HEAD_DIM: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        CAUSAL: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        GQA_GROUP: tl.constexpr,
    ):
        start_m = tl.program_id(0)
        off_hz = tl.program_id(1)
        off_z = off_hz // H
        off_h = off_hz % H
        off_hk = off_h // GQA_GROUP

        q_base = Q + off_z * stride_qz + off_h * stride_qh
        k_base = K + off_z * stride_kz + off_hk * stride_kh
        v_base = V + off_z * stride_vz + off_hk * stride_vh
        b_base = B + off_z * stride_bz

        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, HEAD_DIM)

        q_ptrs = q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
        m_mask = offs_m < N_CTX
        q = tl.load(q_ptrs, mask=m_mask[:, None], other=0.0)
        qf = _fake_quant_blocks(q, BLOCK_M, HEAD_DIM)

        if CAUSAL:
            hi = (start_m + 1) * BLOCK_M
            hi = tl.minimum(hi, N_KV)
        else:
            hi = N_KV

        # Pass 1: online softmax stats only (max + denom), no V -> logsumexp.
        # Needed so pass 2 can fake-quant the NORMALIZED P (eager quantizes P after
        # the softmax divide; fake-quanting unnormalized exp then dividing by l does
        # not equal that because the e4m3 block scale does not survive the divide).
        m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
        l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
        for start_n in range(0, hi, BLOCK_N):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            offs_n = start_n + tl.arange(0, BLOCK_N)
            n_mask = offs_n < N_KV
            k_ptrs = k_base + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
            k = tl.load(k_ptrs, mask=n_mask[:, None], other=0.0)
            kf = _fake_quant_blocks(k, BLOCK_N, HEAD_DIM)
            s = tl.dot(qf, tl.trans(kf)) * sm_scale
            if HAS_BIAS:
                bias = tl.load(b_base + offs_n, mask=n_mask, other=0.0)
                s = s + bias[None, :]
            s = tl.where(n_mask[None, :], s, -float("inf"))
            if CAUSAL:
                causal = offs_m[:, None] >= offs_n[None, :]
                s = tl.where(causal, s, -float("inf"))
            m_new = tl.maximum(m_i, tl.max(s, axis=1))
            m_safe = tl.where(m_new == -float("inf"), 0.0, m_new)
            alpha = tl.exp(m_i - m_safe)
            l_i = l_i * alpha + tl.sum(tl.exp(s - m_safe[:, None]), axis=1)
            m_i = m_new

        l_safe = tl.where(l_i == 0.0, 1.0, l_i)
        m_final = tl.where(m_i == -float("inf"), 0.0, m_i)
        lse = m_final + tl.log(l_safe)  # logsumexp; exp(s - lse) is the normalized P

        # Pass 2: normalized P = exp(s - lse), fake-quant it, build O (low-prec P)
        # and O' (high-prec P, for the backward).
        acc = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)
        accp = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)
        for start_n in range(0, hi, BLOCK_N):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            offs_n = start_n + tl.arange(0, BLOCK_N)
            n_mask = offs_n < N_KV
            k_ptrs = k_base + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
            v_ptrs = v_base + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk
            k = tl.load(k_ptrs, mask=n_mask[:, None], other=0.0)
            v = tl.load(v_ptrs, mask=n_mask[:, None], other=0.0)
            kf = _fake_quant_blocks(k, BLOCK_N, HEAD_DIM)
            vf = _fake_quant_blocks(v, BLOCK_N, HEAD_DIM)
            s = tl.dot(qf, tl.trans(kf)) * sm_scale
            if HAS_BIAS:
                bias = tl.load(b_base + offs_n, mask=n_mask, other=0.0)
                s = s + bias[None, :]
            s = tl.where(n_mask[None, :], s, -float("inf"))
            if CAUSAL:
                causal = offs_m[:, None] >= offs_n[None, :]
                s = tl.where(causal, s, -float("inf"))
            p = tl.exp(s - lse[:, None])  # normalized probabilities
            n_valid = tl.minimum(BLOCK_N, N_KV - start_n)
            pf = _fake_quant_p(p, n_valid, BLOCK_M, BLOCK_N)
            acc += tl.dot(pf.to(vf.dtype), vf)
            accp += tl.dot(p.to(vf.dtype), vf)

        m_i = lse

        o_base = O + off_z * stride_oz + off_h * stride_oh
        op_base = Op + off_z * stride_oz + off_h * stride_oh
        o_ptrs = o_base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
        op_ptrs = op_base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
        tl.store(o_ptrs, acc.to(O.dtype.element_ty), mask=m_mask[:, None])
        tl.store(op_ptrs, accp.to(O.dtype.element_ty), mask=m_mask[:, None])
        m_ptrs = M + off_hz * N_CTX + offs_m
        tl.store(m_ptrs, m_i, mask=m_mask)

    @triton.jit
    def _attn_qat_bwd(
        Q,
        K,
        V,
        sm_scale,
        B,
        DO,
        Op,
        M,
        DQ,
        DK,
        DV,
        stride_qz,
        stride_qh,
        stride_qm,
        stride_qk,
        stride_kz,
        stride_kh,
        stride_kn,
        stride_kk,
        stride_vz,
        stride_vh,
        stride_vn,
        stride_vk,
        stride_oz,
        stride_oh,
        stride_om,
        stride_ok,
        stride_dkz,
        stride_dkh,
        stride_dkn,
        stride_dkk,
        stride_bz,
        Z,
        H,
        N_CTX,
        N_KV,
        HEAD_DIM: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        CAUSAL: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        GQA_GROUP: tl.constexpr,
    ):
        # One program per (z,h,kv-block n): loop over q-blocks m, accumulate dK,dV
        # for this n and atomically scatter dQ. dQ/dK/dV use HIGH-precision softmax
        # grad (STE identity through every fake-quant) so they match eager autograd.
        start_n = tl.program_id(0)
        off_hz = tl.program_id(1)
        off_z = off_hz // H
        off_h = off_hz % H
        off_hk = off_h // GQA_GROUP

        q_base = Q + off_z * stride_qz + off_h * stride_qh
        k_base = K + off_z * stride_kz + off_hk * stride_kh
        v_base = V + off_z * stride_vz + off_hk * stride_vh
        do_base = DO + off_z * stride_oz + off_h * stride_oh
        op_base = Op + off_z * stride_oz + off_h * stride_oh
        dq_base = DQ + off_z * stride_qz + off_h * stride_qh
        dk_base = DK + off_z * stride_dkz + off_h * stride_dkh
        dv_base = DV + off_z * stride_dkz + off_h * stride_dkh

        offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, HEAD_DIM)
        n_mask = offs_n < N_KV
        if HAS_BIAS:
            bias = tl.load(B + off_z * stride_bz + offs_n, mask=n_mask, other=0.0)

        k_ptrs = k_base + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
        v_ptrs = v_base + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk
        k = tl.load(k_ptrs, mask=n_mask[:, None], other=0.0)
        v = tl.load(v_ptrs, mask=n_mask[:, None], other=0.0)
        kf = _fake_quant_blocks(k, BLOCK_N, HEAD_DIM)
        vf = _fake_quant_blocks(v, BLOCK_N, HEAD_DIM)

        dk = tl.zeros((BLOCK_N, HEAD_DIM), dtype=tl.float32)
        dv = tl.zeros((BLOCK_N, HEAD_DIM), dtype=tl.float32)

        if CAUSAL:
            lo = start_n * BLOCK_N
            lo = (lo // BLOCK_M) * BLOCK_M
        else:
            lo = 0

        for start_m in range(lo, N_CTX, BLOCK_M):
            offs_m = start_m + tl.arange(0, BLOCK_M)
            m_mask = offs_m < N_CTX

            q_ptrs = q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
            do_ptrs = (
                do_base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
            )
            op_ptrs = (
                op_base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
            )
            q = tl.load(q_ptrs, mask=m_mask[:, None], other=0.0)
            do = tl.load(do_ptrs, mask=m_mask[:, None], other=0.0).to(tl.float32)
            op = tl.load(op_ptrs, mask=m_mask[:, None], other=0.0).to(tl.float32)
            qf = _fake_quant_blocks(q, BLOCK_M, HEAD_DIM)
            m_i = tl.load(M + off_hz * N_CTX + offs_m, mask=m_mask, other=0.0)

            s = tl.dot(qf, tl.trans(kf)) * sm_scale
            if HAS_BIAS:
                s = s + bias[None, :]
            s = tl.where(n_mask[None, :], s, -float("inf"))
            if CAUSAL:
                causal = offs_m[:, None] >= offs_n[None, :]
                s = tl.where(causal, s, -float("inf"))
            p = tl.exp(s - m_i[:, None])  # high-precision P (eq. 8/9)
            p = tl.where(m_mask[:, None], p, 0.0)
            n_valid = tl.minimum(BLOCK_N, N_KV - start_n * BLOCK_N)
            pf = _fake_quant_p(p, n_valid, BLOCK_M, BLOCK_N)

            # D_i = rowsum(dO_i * O'_i) using high-precision O'   (eq. 9)
            d_i = tl.sum(do * op, axis=1)
            # dV flows through the fake-quant P (O = P_fq @ V_fq, STE -> id)
            dv += tl.dot(tl.trans(pf).to(vf.dtype), do.to(vf.dtype))
            dp = tl.dot(do.to(vf.dtype), tl.trans(vf))
            ds = p * (dp - d_i[:, None]) * sm_scale  # eq. 8
            ds = tl.where(n_mask[None, :], ds, 0.0)
            dk += tl.dot(tl.trans(ds).to(qf.dtype), qf)
            dq = tl.dot(ds.to(kf.dtype), kf)
            dq_ptrs = (
                dq_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
            )
            tl.atomic_add(dq_ptrs, dq, mask=m_mask[:, None])

        dk_ptrs = dk_base + offs_n[:, None] * stride_dkn + offs_d[None, :] * stride_dkk
        dv_ptrs = dv_base + offs_n[:, None] * stride_dkn + offs_d[None, :] * stride_dkk
        tl.store(dk_ptrs, dk.to(DK.dtype.element_ty), mask=n_mask[:, None])
        tl.store(dv_ptrs, dv.to(DV.dtype.element_ty), mask=n_mask[:, None])


def _supported(head_dim: int) -> bool:
    # power-of-2 (Triton arange/dot) AND multiple of 16 (NVFP4 block); 64/128/256.
    return (
        HAS_TRITON
        and head_dim % _BLOCK_SIZE == 0
        and head_dim & (head_dim - 1) == 0
        and 16 <= head_dim <= 256
    )


_NO_BIAS = torch.empty(0)


class _AttnQatFlash(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, sm_scale, causal, gqa_group, bias):
        # q: [Z,H,N_CTX,D]  k/v: [Z,Hk,N_KV,D]  bias: [Z,N_KV] additive or empty
        Z, H, N_CTX, D = q.shape
        N_KV = k.shape[2]
        BLOCK_M = 16
        BLOCK_N = 16
        has_bias = bias.numel() > 0
        b = bias if has_bias else q  # placeholder ptr when unused
        stride_bz = bias.stride(0) if has_bias else 0
        o = torch.empty_like(q)
        op = torch.empty_like(q)
        m = torch.empty((Z * H, N_CTX), device=q.device, dtype=torch.float32)
        grid = (triton.cdiv(N_CTX, BLOCK_M), Z * H)
        _attn_qat_fwd[grid](
            q,
            k,
            v,
            sm_scale,
            b,
            o,
            op,
            m,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            o.stride(0),
            o.stride(1),
            o.stride(2),
            o.stride(3),
            stride_bz,
            Z,
            H,
            N_CTX,
            N_KV,
            HEAD_DIM=D,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            CAUSAL=causal,
            HAS_BIAS=has_bias,
            GQA_GROUP=gqa_group,
            num_warps=4,
            num_stages=1,
        )
        ctx.save_for_backward(q, k, v, op, m, bias)
        ctx.sm_scale = sm_scale
        ctx.causal = causal
        ctx.gqa_group = gqa_group
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, op, m, bias = ctx.saved_tensors
        Z, H, N_CTX, D = q.shape
        Hk = k.shape[1]
        N_KV = k.shape[2]
        BLOCK_M = 16
        BLOCK_N = 16
        has_bias = bias.numel() > 0
        b = bias if has_bias else q
        stride_bz = bias.stride(0) if has_bias else 0
        do = do.contiguous()
        dq = torch.zeros_like(q, dtype=torch.float32)
        # dK/dV per query head, reduced over the GQA group afterwards.
        dk = torch.empty_like(q)
        dv = torch.empty_like(q)
        grid = (triton.cdiv(N_KV, BLOCK_N), Z * H)
        _attn_qat_bwd[grid](
            q,
            k,
            v,
            ctx.sm_scale,
            b,
            do,
            op,
            m,
            dq,
            dk,
            dv,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            do.stride(0),
            do.stride(1),
            do.stride(2),
            do.stride(3),
            dk.stride(0),
            dk.stride(1),
            dk.stride(2),
            dk.stride(3),
            stride_bz,
            Z,
            H,
            N_CTX,
            N_KV,
            HEAD_DIM=D,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            CAUSAL=ctx.causal,
            HAS_BIAS=has_bias,
            GQA_GROUP=ctx.gqa_group,
            num_warps=4,
            num_stages=1,
        )
        dq = dq.to(q.dtype)
        if ctx.gqa_group > 1:
            dk = dk.view(Z, Hk, ctx.gqa_group, N_KV, D).sum(2)
            dv = dv.view(Z, Hk, ctx.gqa_group, N_KV, D).sum(2)
        return dq, dk, dv, None, None, None, None


# See kernels/swiglu.py: run eager under torch.compile so the raw triton flash
# kernels don't leak into the compiled backward (decompose_triton_kernel_wrapper_
# functional). Inductor compiles around this op (graph break), like the native
# NVFP4 attention path.
@torch.compiler.disable
def fused_nvfp4_qat_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scaling: float,
    causal: bool,
    num_key_value_groups: int,
    key_pad_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fused NVFP4 fake-quant attention. q:[Z,H,S,D], k/v:[Z,Hk,S,D] (pre-repeat_kv).

    ``key_pad_bias`` is an optional [Z, N_KV] additive per-key bias (0 keep, -inf
    pad) applied to the scores before the online softmax, combined with the causal
    mask. Returns [Z,H,S,D]. Requires supported head_dim.
    """
    q = query.contiguous()
    k = key.contiguous()
    v = value.contiguous()
    if key_pad_bias is None:
        bias = _NO_BIAS.to(device=q.device, dtype=q.dtype)
    else:
        bias = key_pad_bias.contiguous().to(device=q.device, dtype=q.dtype)
    return _AttnQatFlash.apply(q, k, v, scaling, causal, num_key_value_groups, bias)
