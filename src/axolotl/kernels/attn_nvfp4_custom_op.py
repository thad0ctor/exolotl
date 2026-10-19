"""torch.compile custom-op boundary for native-NVFP4 attention."""

from __future__ import annotations

import torch

from axolotl.kernels.attn_nvfp4_flash import (
    _BWD_RN_DO_ZSHD,
    _gqa_reduce_cast_dkdv,
    _next_mult,
    _resolve_backward_grad_dots,
    _run_bwd,
    _run_bwd_hp,
    _varlen_seq_arrays,
    nvfp4_flash_attention,
    nvfp4_flash_attention_packed,
)

# Valid backward grad-GEMM modes (see the sage fork's BACKWARD GRAD-DOT MODES
# docstring). The custom-op schema has no Optional[str], so the empty string is
# the "None = kernel AUTO" sentinel everywhere a mode crosses the op boundary.
_GRAD_DOTS_MODES = ("bf16", "fp4_rownorm", "fp8_rownorm", "fp4_legacy")


def _resolve_grad_dots(
    mode_str: str, eff_seq: float, save_backward_packs: bool, ng: int
) -> str:
    """Resolve the op-boundary grad-dots string ("" = AUTO) to a concrete mode.

    The KERNEL owns the resolution policy (``_resolve_backward_grad_dots``:
    AUTO = "bf16" below ~4k effective sequence else "fp4_rownorm"; "bf16"
    clamped to "fp4_rownorm" against saved packs / GQA group > 8) — this just
    threads the string through with the same inputs the kernel would see. On a
    legacy fork (no resolver / no rownorm modes) it degrades to the old
    tri-state: hp ("bf16") whenever the packs-only saved set is absent, else
    the legacy all-FP4 backward.
    """
    mode = mode_str or None
    if _resolve_backward_grad_dots is not None:
        return _resolve_backward_grad_dots(mode, eff_seq, save_backward_packs, ng)
    if mode is None or mode in ("fp4_rownorm", "fp8_rownorm"):
        mode = "bf16"  # best legacy-fork stand-in for the rownorm arms
    if mode == "bf16" and (save_backward_packs or _run_bwd_hp is None):
        mode = "fp4_legacy"
    return mode


_DTYPE_TO_CODE = {
    torch.float16: 0,
    torch.bfloat16: 1,
    torch.float32: 2,
}
_CODE_TO_DTYPE = {code: dtype for dtype, code in _DTYPE_TO_CODE.items()}


def _dtype_to_code(dtype: torch.dtype) -> int:
    try:
        return _DTYPE_TO_CODE[dtype]
    except KeyError as exc:
        raise TypeError(f"unsupported NVFP4 attention output dtype: {dtype}") from exc


def _code_to_dtype(code: int) -> torch.dtype:
    try:
        return _CODE_TO_DTYPE[int(code)]
    except KeyError as exc:
        raise TypeError(
            f"unsupported NVFP4 attention output dtype code: {code}"
        ) from exc


def _empty_pack_outputs(tensor: torch.Tensor) -> tuple[torch.Tensor, ...]:
    return tuple(tensor.new_empty((0,), dtype=torch.uint8) for _ in range(10))


def _static_int(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _use_saved_train_packs(requested: bool, s_q, s_kv) -> bool:
    if not requested:
        return False
    s_q_i = _static_int(s_q)
    s_kv_i = _static_int(s_kv)
    if s_q_i is None or s_kv_i is None:
        return False
    return max(s_q_i, s_kv_i) <= 4096


@torch.library.custom_op("axolotl_nvfp4::flash_attention_packed", mutates_args=())
def _flash_attention_packed_op(
    qnv: torch.Tensor,
    qsc: torch.Tensor,
    knv: torch.Tensor,
    ksc: torch.Tensor,
    vnv: torch.Tensor,
    vsc: torch.Tensor,
    key_pad_bias: torch.Tensor,
    z: int,
    h: int,
    hk: int,
    s_q: int,
    s_kv: int,
    d: int,
    scaling: float,
    out_dtype_code: int,
    causal: bool,
    block_m: int,
    block_n: int,
    num_warps: int,
    num_stages: int,
) -> torch.Tensor:
    bias = None if key_pad_bias.numel() == 0 else key_pad_bias
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
        out_dtype=_code_to_dtype(out_dtype_code),
        causal=causal,
        key_pad_bias=bias,
        block_m=block_m,
        block_n=block_n,
        num_warps=num_warps,
        num_stages=num_stages,
    )


@_flash_attention_packed_op.register_fake
def _(
    qnv,
    qsc,
    knv,
    ksc,
    vnv,
    vsc,
    key_pad_bias,
    z: int,
    h: int,
    hk: int,
    s_q: int,
    s_kv: int,
    d: int,
    scaling: float,
    out_dtype_code: int,
    causal: bool,
    block_m: int,
    block_n: int,
    num_warps: int,
    num_stages: int,
):
    del qsc, knv, ksc, vnv, vsc, key_pad_bias
    del hk, s_kv, scaling, causal, block_m, block_n, num_warps, num_stages
    return qnv.new_empty((z, h, s_q, d), dtype=_code_to_dtype(out_dtype_code))


def nvfp4_flash_attention_packed_custom_op(
    qnv: torch.Tensor,
    qsc: torch.Tensor,
    knv: torch.Tensor,
    ksc: torch.Tensor,
    vnv: torch.Tensor,
    vsc: torch.Tensor,
    *,
    z: int,
    h: int,
    hk: int,
    s_q: int,
    s_kv: int,
    d: int,
    scaling: float,
    out_dtype: torch.dtype,
    causal: bool = False,
    key_pad_bias: torch.Tensor | None = None,
    block_m: int = 64,
    block_n: int = 128,
    num_warps: int = 8,
    num_stages: int = 3,
) -> torch.Tensor:
    bias = (
        qnv.new_empty((0,), dtype=torch.float32)
        if key_pad_bias is None
        else key_pad_bias
    )
    return _flash_attention_packed_op(
        qnv,
        qsc,
        knv,
        ksc,
        vnv,
        vsc,
        bias,
        z,
        h,
        hk,
        s_q,
        s_kv,
        d,
        scaling,
        _dtype_to_code(out_dtype),
        causal,
        block_m,
        block_n,
        num_warps,
        num_stages,
    )


# ---------------------------------------------------------------------------
# Differentiable training custom op.
#
# The forward custom op above is inference-only: it operates on already-packed
# NVFP4 operands and has no registered backward, so it can't carry gradients
# across the torch.compile boundary. This op wraps the FULL high-precision
# attention (internal pre-quant + native-NVFP4 flash forward) AND registers a
# native-NVFP4 backward via torch.library, so Inductor treats the whole attention
# as one opaque-but-DIFFERENTIABLE op: it compiles AROUND it (no attempt to trace
# tl.dot_scaled, so no InductorError + no silent eager fallback of the backward
# subgraph), while the registered backward computes the SAME native-NVFP4 grads as
# nvfp4_flash_attn_func.
#
# The forward returns ``out`` plus non-differentiable auxiliary outputs: per-row
# LSE, and optionally the forward Q/K/V FP4 packs needed by backward. The public
# wrapper still exposes a single differentiable tensor while backward can reuse
# those tensors across the opaque boundary.
# ---------------------------------------------------------------------------
@torch.library.custom_op("axolotl_nvfp4::flash_attention_train", mutates_args=())
def _flash_attention_train_op(
    query: torch.Tensor,  # [Z, H, Sq, D] high precision
    key: torch.Tensor,  # [Z, Hk, Skv, D]
    value: torch.Tensor,  # [Z, Hk, Skv, D]
    key_pad_bias: torch.Tensor,  # [Z, Skv] fp32 or empty
    cu_seqlens: torch.Tensor,  # [nseq+1] int varlen boundaries, or empty (dense)
    qnv_in: torch.Tensor,  # pre-packed fwd Q [Z*H, Sq, D//2] u8, or empty
    qsc_in: torch.Tensor,  # its e4m3 scale (as u8) [Z*H, Sq, D//16], or empty
    knv_in: torch.Tensor,  # pre-packed fwd K [Z*Hk, Skv, D//2] u8, or empty
    ksc_in: torch.Tensor,  # its e4m3 scale (as u8) [Z*Hk, Skv, D//16], or empty
    scaling: float,
    causal: bool,
    num_key_value_groups: int,
    sr: bool,
    backward_p_dv_sr: bool,
    backward_dot_dv_sr: bool,
    backward_ds_dq_sr: bool,
    dkdv_scratch_bf16: bool,
    grad_dots: str,
    save_backward_packs: bool,
    out_zshd: bool,
    block_m: int,
    block_n: int,
    num_warps: int,
    num_stages: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    del sr, backward_p_dv_sr, backward_dot_dv_sr, backward_ds_dq_sr
    del dkdv_scratch_bf16, grad_dots
    bias = None if key_pad_bias.numel() == 0 else key_pad_bias
    cu = None if cu_seqlens.numel() == 0 else cu_seqlens
    # Externally-produced forward q/k packs (fused RoPE+quant producers): empty
    # tensors are the "absent" sentinel (no Optional[Tensor] in op schemas, same
    # convention as key_pad_bias / cu_seqlens). They skip the matching forward
    # pre-quant only; backward still repacks from the saved hp q/k.
    q_packs = (
        None
        if qnv_in.numel() == 0
        else (qnv_in, qsc_in.view(torch.float8_e4m3fn))
    )
    k_packs = (
        None
        if knv_in.numel() == 0
        else (knv_in, ksc_in.view(torch.float8_e4m3fn))
    )
    # Varlen forces the HP backward, which is mutually exclusive with saved
    # packs — never produce packs for a varlen forward (mirrors the kernel).
    use_saved_packs = cu is None and _use_saved_train_packs(
        save_backward_packs, query.shape[2], key.shape[2]
    )
    if use_saved_packs:
        assert q_packs is None and k_packs is None, (
            "q_packs/k_packs are incompatible with save_backward_packs "
            "(rejected in the public wrapper)"
        )
        out, lse, packs = nvfp4_flash_attention(
            query,
            key,
            value,
            scaling,
            causal=causal,
            num_key_value_groups=num_key_value_groups,
            key_pad_bias=bias,
            block_m=block_m,
            block_n=block_n,
            num_warps=num_warps,
            num_stages=num_stages,
            return_lse=True,
            return_packs=True,
            out_layout="zshd" if out_zshd else "zhsd",
        )
    else:
        # kwargs only when prepacked (a legacy fork's nvfp4_flash_attention
        # lacks q_packs; the packs are always empty there).
        prepacked_kwargs = {}
        if q_packs is not None or k_packs is not None:
            prepacked_kwargs = {"q_packs": q_packs, "k_packs": k_packs}
        out, lse = nvfp4_flash_attention(
            query,
            key,
            value,
            scaling,
            causal=causal,
            num_key_value_groups=num_key_value_groups,
            key_pad_bias=bias,
            cu_seqlens=cu,
            block_m=block_m,
            block_n=block_n,
            num_warps=num_warps,
            num_stages=num_stages,
            return_lse=True,
            out_layout="zshd" if out_zshd else "zhsd",
            **prepacked_kwargs,
        )
        packs = _empty_pack_outputs(query)
    return (out, lse, *packs)


@_flash_attention_train_op.register_fake
def _(
    query,
    key,
    value,
    key_pad_bias,
    cu_seqlens,
    qnv_in,
    qsc_in,
    knv_in,
    ksc_in,
    scaling: float,
    causal: bool,
    num_key_value_groups: int,
    sr: bool,
    backward_p_dv_sr: bool,
    backward_dot_dv_sr: bool,
    backward_ds_dq_sr: bool,
    dkdv_scratch_bf16: bool,
    grad_dots: str,
    save_backward_packs: bool,
    out_zshd: bool,
    block_m: int,
    block_n: int,
    num_warps: int,
    num_stages: int,
):
    del key_pad_bias, scaling, causal, num_key_value_groups
    del qnv_in, qsc_in, knv_in, ksc_in
    del sr, backward_p_dv_sr, backward_dot_dv_sr, backward_ds_dq_sr
    del dkdv_scratch_bf16, grad_dots, num_warps, num_stages
    z, h, s_q, d = query.shape
    _, hk, s_kv, _ = key.shape
    out = query.new_empty((z, s_q, h, d) if out_zshd else (z, h, s_q, d))
    lse = query.new_empty((z * h, s_q), dtype=torch.float32)
    if cu_seqlens.numel() > 0 or not _use_saved_train_packs(
        save_backward_packs, s_q, s_kv
    ):
        return (out, lse, *_empty_pack_outputs(query))

    bwd_block_m = min(block_m, 64)
    s_q_bwd_pad = _next_mult(s_q, max(bwd_block_m, 32))
    s_kv_bwd_pad = _next_mult(s_kv, 64)
    return (
        out,
        lse,
        query.new_empty((z * h, s_q, d // 2), dtype=torch.uint8),
        query.new_empty((z * h, s_q, d // 16), dtype=torch.uint8),
        query.new_empty((z * h, d, s_q_bwd_pad // 2), dtype=torch.uint8),
        query.new_empty((z * h, d, s_q_bwd_pad // 16), dtype=torch.uint8),
        key.new_empty((z * hk, s_kv, d // 2), dtype=torch.uint8),
        key.new_empty((z * hk, s_kv, d // 16), dtype=torch.uint8),
        value.new_empty((z * hk, s_kv, d // 2), dtype=torch.uint8),
        value.new_empty((z * hk, s_kv, d // 16), dtype=torch.uint8),
        key.new_empty((z * hk, d, s_kv_bwd_pad // 2), dtype=torch.uint8),
        key.new_empty((z * hk, d, s_kv_bwd_pad // 16), dtype=torch.uint8),
    )


def _flash_attention_train_setup_context(ctx, inputs, output):
    (
        query,
        key,
        value,
        key_pad_bias,
        cu_seqlens,
        _qnv_in,  # fwd-only pre-packed operands: the backward repacks from hp
        _qsc_in,
        _knv_in,
        _ksc_in,
        scaling,
        causal,
        num_key_value_groups,
        sr,
        backward_p_dv_sr,
        backward_dot_dv_sr,
        backward_ds_dq_sr,
        dkdv_scratch_bf16,
        grad_dots,
        save_backward_packs,
        out_zshd,
        block_m,
        block_n,
        num_warps,
        num_stages,
    ) = inputs
    del num_key_value_groups
    out, lse, qnv, qsc, qtnv, qtsc, knv, ksc, vdnv, vdsc, ktnv, ktsc = output
    ctx.set_materialize_grads(False)
    ctx.mark_non_differentiable(
        lse, qnv, qsc, qtnv, qtsc, knv, ksc, vdnv, vdsc, ktnv, ktsc
    )
    use_saved_packs = save_backward_packs and qnv.numel() > 0
    if use_saved_packs:
        query_save = key_save = value_save = out
    else:
        query_save, key_save, value_save = query, key, value
    ctx.save_for_backward(
        query_save,
        key_save,
        value_save,
        out,
        key_pad_bias,
        cu_seqlens,
        lse,
        qnv,
        qsc,
        qtnv,
        qtsc,
        knv,
        ksc,
        vdnv,
        vdsc,
        ktnv,
        ktsc,
    )
    z, h, s_q, d = query.shape
    _, hk, s_kv, _ = key.shape
    ctx.dims = (z, h, hk, s_q, s_kv, d)
    ctx.scaling = scaling
    ctx.causal = causal
    ctx.sr = sr
    ctx.backward_p_dv_sr = backward_p_dv_sr
    ctx.backward_dot_dv_sr = backward_dot_dv_sr
    ctx.backward_ds_dq_sr = backward_ds_dq_sr
    ctx.dkdv_scratch_bf16 = dkdv_scratch_bf16
    # Grad-GEMM mode string ("" = kernel AUTO). Passed through UNRESOLVED: the
    # backward op resolves it via the kernel's own policy against the EFFECTIVE
    # saved-packs decision (the seq-length gate above) and the effective
    # sequence (s_kv, or the mean sample length under varlen) — deterministic
    # from inputs the bwd op already receives, so fwd/bwd always agree.
    ctx.grad_dots = grad_dots
    ctx.save_backward_packs = use_saved_packs
    ctx.out_zshd = bool(out_zshd)
    ctx.tiles = (block_m, block_n, num_warps, num_stages)


# The backward must ALSO be an opaque custom op. register_autograd makes the bwd
# part of the autograd graph, but under compiled-autograd torch.compile traces the
# bwd too — and _run_bwd's raw Triton kernels hit FakeTensor .data_ptr()/.stride()
# accesses (the "Cannot access data pointer of FakeTensor" InductorError). Wrapping
# _run_bwd in its own custom op + fake makes the whole backward opaque to Inductor.
@torch.library.custom_op("axolotl_nvfp4::flash_attention_train_bwd", mutates_args=())
def _flash_attention_train_bwd_op(
    grad_out: torch.Tensor,  # [Z, H, Sq, D]
    query: torch.Tensor,  # [Z, H, Sq, D]
    key: torch.Tensor,  # [Z, Hk, Skv, D]
    value: torch.Tensor,  # [Z, Hk, Skv, D]
    out: torch.Tensor,  # [Z, H, Sq, D]
    key_pad_bias: torch.Tensor,  # [Z, Skv] fp32 or empty
    cu_seqlens: torch.Tensor,  # [nseq+1] int varlen boundaries, or empty (dense)
    lse: torch.Tensor,  # [Z*H, Sq] fp32, forward softmax stats
    qnv: torch.Tensor,
    qsc: torch.Tensor,
    qtnv: torch.Tensor,
    qtsc: torch.Tensor,
    knv: torch.Tensor,
    ksc: torch.Tensor,
    vdnv: torch.Tensor,
    vdsc: torch.Tensor,
    ktnv: torch.Tensor,
    ktsc: torch.Tensor,
    z: int,
    h: int,
    hk: int,
    s_q: int,
    s_kv: int,
    d: int,
    save_backward_packs: bool,
    out_zshd: bool,
    scaling: float,
    causal: bool,
    sr: bool,
    backward_p_dv_sr: bool,
    backward_dot_dv_sr: bool,
    backward_ds_dq_sr: bool,
    dkdv_scratch_bf16: bool,
    grad_dots: str,
    block_m: int,
    block_n: int,
    num_warps: int,
    num_stages: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bias = None if key_pad_bias.numel() == 0 else key_pad_bias.to(torch.float32)
    seq_arrays = None
    varlen = cu_seqlens.numel() > 0
    if varlen:
        # The forward ran block-diagonal causal under these boundaries (the LSE
        # was computed under that mask); expand them again for the backward.
        # Every grad-dots mode composes with varlen on the new fork; saved
        # packs never do (the wrapper rejects and the fwd never produces them).
        if save_backward_packs:
            raise RuntimeError(
                "varlen (cu_seqlens) backward is incompatible with the saved "
                "forward FP4 pack set (save_backward_packs)"
            )
        if _varlen_seq_arrays is None:
            raise RuntimeError(
                "varlen (cu_seqlens) backward requires a sageattention fork "
                "with varlen support (_varlen_seq_arrays)"
            )
        seq_arrays = _varlen_seq_arrays(cu_seqlens, s_q, grad_out.device)
    # Resolve the mode HERE, with the kernel's own policy ("" = AUTO) — the
    # effective sequence is s_kv, or the MEAN SAMPLE LENGTH under varlen, and
    # save_backward_packs is the fwd's EFFECTIVE saved-packs decision.
    eff_seq = s_kv if not varlen else s_q / max(int(cu_seqlens.numel()) - 1, 1)
    mode = _resolve_grad_dots(grad_dots, eff_seq, save_backward_packs, h // hk)
    if mode == "fp8_rownorm" and save_backward_packs:
        # Mirrors the kernel assert (MXFP8 Q^T/K^T repack needs hp q/k, which
        # the packs-only saved set dropped). The wrapper rejects the REQUESTED
        # combination up front; this guards the resolved one.
        raise RuntimeError(
            "grad_dots='fp8_rownorm' is incompatible with save_backward_packs"
        )
    if varlen and mode != "bf16" and _resolve_backward_grad_dots is None:
        raise RuntimeError(
            "varlen (cu_seqlens) backward on a legacy sageattention fork is "
            "only implemented on the hp ('bf16') path"
        )
    if out_zshd and mode == "fp4_rownorm" and not _BWD_RN_DO_ZSHD:
        # LEGACY-FORK fold only: old fp4_rownorm asserts a contiguous
        # [Z*H,Sq,D] dO, so fold the zshd [Z,Sq,H,D] grad/out with one
        # transpose copy. Current forks (_BWD_RN_DO_ZSHD) take zshd directly
        # in every mode — two ~[Z*Sq*H*D] copies saved per backward.
        do = grad_out.permute(0, 2, 1, 3).reshape(z * h, s_q, d).contiguous()
        o = out.permute(0, 2, 1, 3).reshape(z * h, s_q, d).contiguous()
        zshd_bwd = False
    elif out_zshd:
        do = grad_out
        o = out
        zshd_bwd = True
    else:
        do = grad_out.reshape(z * h, s_q, d).contiguous()
        o = out.reshape(z * h, s_q, d).contiguous()
        zshd_bwd = False
    if save_backward_packs:
        # HP q/k/v were intentionally not saved. _run_bwd does not dereference them
        # when LSE plus all deterministic forward packs are supplied; it only needs
        # a same-device tensor for scratch allocations and dummy bias pointers.
        query_bwd = key_bwd = value_bwd = o
    else:
        query_bwd = query.reshape(z * h, s_q, d).contiguous()
        key_bwd = key.reshape(z * hk, s_kv, d).contiguous()
        value_bwd = value.reshape(z * hk, s_kv, d).contiguous()
    if mode == "bf16":
        # HP-grad-dots backward (the resolver already clamped "bf16" away from
        # saved packs / GQA group > 8): FP4 S/dP recomputes + bf16 grad GEMMs
        # with exact operands (the SR knobs and dkdv_scratch_bf16 are moot
        # here). dq/dk/dv come back FINAL — grad_out dtype and GQA dK/dV
        # already group-reduced in-kernel — so the reduce/cast epilogue below
        # is skipped.
        dq, dk, dv = _run_bwd_hp(
            query_bwd,
            key_bwd,
            value_bwd,
            do,
            o,
            bias,
            z,
            h,
            hk,
            s_q,
            s_kv,
            d,
            scaling,
            causal,
            lse.reshape(z * h, s_q).contiguous(),
            do_zshd=zshd_bwd,
            o_zshd=zshd_bwd,
            out_dtype=grad_out.dtype,
            seq_arrays=seq_arrays,
        )
        return (
            dq.reshape(z, h, s_q, d),
            dk.reshape(z, hk, s_kv, d),
            dv.reshape(z, hk, s_kv, d),
        )
    # FP4/FP8 grad-dot backward. ``grad_dots``/``seq_arrays`` exist only on the
    # new fork (the resolver's presence implies them); a legacy fork can only
    # reach here with mode "fp4_legacy" and seq_arrays None (raised above).
    mode_kwargs = {}
    if _resolve_backward_grad_dots is not None:
        mode_kwargs["grad_dots"] = mode
        if seq_arrays is not None:
            mode_kwargs["seq_arrays"] = seq_arrays
    dq, dk, dv = _run_bwd(
        query_bwd,
        key_bwd,
        value_bwd,
        do,
        o,
        bias,
        z,
        h,
        hk,
        s_q,
        s_kv,
        d,
        scaling,
        causal,
        sr,
        block_m,
        block_n,
        num_warps,
        num_stages,
        lse=lse.reshape(z * h, s_q).contiguous(),
        sr_p_dv=backward_p_dv_sr,
        sr_dot_dv=backward_dot_dv_sr,
        sr_ds_dq=backward_ds_dq_sr,
        dkdv_scratch_bf16=dkdv_scratch_bf16,
        qnv_saved=qnv if qnv.numel() else None,
        qsc_saved=qsc if qsc.numel() else None,
        qtnv_saved=qtnv if qtnv.numel() else None,
        qtsc_saved=qtsc if qtsc.numel() else None,
        knv_saved=knv if knv.numel() else None,
        ksc_saved=ksc if ksc.numel() else None,
        vnv_saved=vdnv if vdnv.numel() else None,
        vsc_saved=vdsc if vdsc.numel() else None,
        ktnv_saved=ktnv if ktnv.numel() else None,
        ktsc_saved=ktsc if ktsc.numel() else None,
        do_zshd=zshd_bwd,
        o_zshd=zshd_bwd,
        **mode_kwargs,
    )
    dq = dq.reshape(z, h, s_q, d).to(grad_out.dtype).contiguous()
    ng = h // hk
    if ng > 1:
        dk, dv = _gqa_reduce_cast_dkdv(dk, dv, z, h, hk, s_kv, d, grad_out.dtype)
    else:
        dk = dk.reshape(z, hk, s_kv, d).to(grad_out.dtype)
        dv = dv.reshape(z, hk, s_kv, d).to(grad_out.dtype)
    return dq.contiguous(), dk.contiguous(), dv.contiguous()


@_flash_attention_train_bwd_op.register_fake
def _(
    grad_out,
    query,
    key,
    value,
    out,
    key_pad_bias,
    cu_seqlens,
    lse,
    qnv,
    qsc,
    qtnv,
    qtsc,
    knv,
    ksc,
    vdnv,
    vdsc,
    ktnv,
    ktsc,
    z: int,
    h: int,
    hk: int,
    s_q: int,
    s_kv: int,
    d: int,
    save_backward_packs: bool,
    out_zshd: bool,
    scaling: float,
    causal: bool,
    sr: bool,
    backward_p_dv_sr: bool,
    backward_dot_dv_sr: bool,
    backward_ds_dq_sr: bool,
    dkdv_scratch_bf16: bool,
    grad_dots: str,
    block_m: int,
    block_n: int,
    num_warps: int,
    num_stages: int,
):
    del grad_out, out, key_pad_bias, cu_seqlens, lse
    del qnv, qsc, qtnv, qtsc, knv, ksc, vdnv, vdsc, ktnv, ktsc
    del scaling, causal, sr, save_backward_packs, out_zshd
    del backward_p_dv_sr, backward_dot_dv_sr, backward_ds_dq_sr, dkdv_scratch_bf16
    del grad_dots, block_m, block_n, num_warps, num_stages
    # The real op returns CONTIGUOUS dq/dk/dv (reshape / GQA-reduce empty), even when
    # the q/k/v inputs are non-contiguous roped views — the fake must match strides.
    # Both backward flavors return these final [Z,(H|Hk),S,D] grads (the HP-grad-dots
    # path allocates them directly; the legacy path reshapes/reduces to them).
    dq = query.new_empty((z, h, s_q, d))
    dk = key.new_empty((z, hk, s_kv, d))
    dv = value.new_empty((z, hk, s_kv, d))
    return dq, dk, dv


def _flash_attention_train_backward(
    ctx,
    grad_out,
    grad_lse,
    grad_qnv,
    grad_qsc,
    grad_qtnv,
    grad_qtsc,
    grad_knv,
    grad_ksc,
    grad_vdnv,
    grad_vdsc,
    grad_ktnv,
    grad_ktsc,
):
    del grad_lse, grad_qnv, grad_qsc, grad_qtnv, grad_qtsc
    del grad_knv, grad_ksc, grad_vdnv, grad_vdsc, grad_ktnv, grad_ktsc
    (
        query,
        key,
        value,
        out,
        key_pad_bias,
        cu_seqlens,
        lse,
        qnv,
        qsc,
        qtnv,
        qtsc,
        knv,
        ksc,
        vdnv,
        vdsc,
        ktnv,
        ktsc,
    ) = ctx.saved_tensors
    z, h, hk, s_q, s_kv, d = ctx.dims
    block_m, block_n, num_warps, num_stages = ctx.tiles
    dq, dk, dv = torch.ops.axolotl_nvfp4.flash_attention_train_bwd(
        grad_out.contiguous(),
        query,
        key,
        value,
        out,
        key_pad_bias,
        cu_seqlens,
        lse,
        qnv,
        qsc,
        qtnv,
        qtsc,
        knv,
        ksc,
        vdnv,
        vdsc,
        ktnv,
        ktsc,
        z,
        h,
        hk,
        s_q,
        s_kv,
        d,
        ctx.save_backward_packs,
        ctx.out_zshd,
        ctx.scaling,
        ctx.causal,
        ctx.sr,
        ctx.backward_p_dv_sr,
        ctx.backward_dot_dv_sr,
        ctx.backward_ds_dq_sr,
        ctx.dkdv_scratch_bf16,
        ctx.grad_dots,
        block_m,
        block_n,
        num_warps,
        num_stages,
    )
    # one grad slot per forward input (24 inputs)
    return (dq, dk, dv) + (None,) * 21


torch.library.register_autograd(
    "axolotl_nvfp4::flash_attention_train",
    _flash_attention_train_backward,
    setup_context=_flash_attention_train_setup_context,
)


def nvfp4_flash_attn_train_custom_op(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scaling: float,
    *,
    causal: bool = False,
    num_key_value_groups: int = 1,
    key_pad_bias: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
    stochastic_rounding: bool = True,
    backward_p_dv_stochastic_rounding: bool | None = None,
    backward_dot_dv_stochastic_rounding: bool | None = None,
    backward_ds_dq_stochastic_rounding: bool | None = None,
    dkdv_scratch_bf16: bool = False,
    backward_grad_dots: str | None = None,
    backward_bf16_grad_dots: bool | None = None,
    save_backward_packs: bool = False,
    out_layout: str = "zhsd",
    block_m: int = 64,
    block_n: int = 128,
    num_warps: int = 8,
    num_stages: int = 3,
    q_packs: tuple[torch.Tensor, torch.Tensor] | None = None,
    k_packs: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Differentiable native-NVFP4 attention as an opaque torch.compile custom op.

    Same forward/backward math as ``nvfp4_flash_attn_func`` but wrapped so Inductor
    compiles around it (no ``tl.dot_scaled`` trace, no eager fallback of the bwd
    subgraph). All SR knobs are op arguments (so they survive compile/serialization);
    the bwd reuses the forward LSE and, when requested, the deterministic forward
    FP4 packs across the opaque boundary for short/mid static sequence lengths.
    ``backward_grad_dots`` selects the backward grad-GEMM mode (the kernel's
    BACKWARD GRAD-DOT MODES): "bf16" (hp grad GEMMs with exact operands; clamped
    to "fp4_rownorm" against saved packs / GQA group > 8), "fp4_rownorm" (all-FP4
    with row-RMS-normalized grad packs; hp-equal quality, faster from ~4k seq),
    "fp8_rownorm" (MXFP8 grad operands; rejected with ``save_backward_packs``),
    "fp4_legacy" (the original all-FP4 backward; honors the SR knobs), or
    None = kernel AUTO ("bf16" below an effective sequence of 4096 — the mean
    sample length under varlen — else "fp4_rownorm"). The op schema has no
    Optional[str], so the EMPTY STRING is the "None/auto" sentinel across the
    boundary; resolution happens in the backward op via the kernel's own
    resolver, against the EFFECTIVE saved-packs decision (seq-length gated).
    ``backward_bf16_grad_dots`` (bool) is a DEPRECATED alias: True -> "bf16",
    False -> "fp4_legacy"; passing both raises.
    ``cu_seqlens`` (optional int ``[nseq+1]``, FA-varlen convention) runs PACKED
    sequences flattened into one Z=1 row as block-diagonal causal attention;
    requires ``causal=True`` and composes with every ``backward_grad_dots`` mode
    (``save_backward_packs`` is rejected, mirroring the kernel). The op schema
    has no Optional[Tensor], so an EMPTY tensor is the "no cu_seqlens" sentinel
    (same convention as ``key_pad_bias``).
    ``out_layout="zshd"`` writes the HF attention layout directly.
    ``q_packs`` / ``k_packs`` (optional ``(packed u8, e4m3 scale)`` pairs in the
    along-D ``_quant_nvfp4`` layout, e.g. from the fused RoPE+quant producers)
    skip the matching FORWARD pre-quant launches only; the backward repacks
    from the saved hp q/k as always. Threaded through the op schema as four
    tensors with EMPTY sentinels. Incompatible with ``save_backward_packs``.
    """
    if out_layout not in ("zhsd", "zshd"):
        raise ValueError("out_layout must be 'zhsd' or 'zshd'")
    if backward_bf16_grad_dots is not None:
        if backward_grad_dots is not None:
            raise ValueError(
                "pass either backward_grad_dots or the deprecated "
                "backward_bf16_grad_dots, not both"
            )
        backward_grad_dots = "bf16" if backward_bf16_grad_dots else "fp4_legacy"
    if backward_grad_dots is not None and backward_grad_dots not in _GRAD_DOTS_MODES:
        raise ValueError(
            f"backward_grad_dots must be one of {_GRAD_DOTS_MODES} or None "
            f"(auto), got {backward_grad_dots!r}"
        )
    if backward_grad_dots == "fp8_rownorm" and save_backward_packs:
        raise ValueError(
            "backward_grad_dots='fp8_rownorm' repacks Q^T/K^T as MXFP8; "
            "incompatible with the saved forward FP4 transposed packs "
            "(save_backward_packs)"
        )
    if (q_packs is not None or k_packs is not None) and save_backward_packs:
        raise ValueError(
            "q_packs/k_packs are incompatible with save_backward_packs "
            "(the saved pack set must come from the same fused HP read)"
        )
    if cu_seqlens is not None:
        if not causal:
            raise ValueError(
                "varlen (cu_seqlens) implements block-diagonal CAUSAL attention; "
                "pass causal=True"
            )
        if save_backward_packs:
            raise ValueError(
                "varlen (cu_seqlens) is incompatible with save_backward_packs "
                "(the forward saved-pack set is not varlen-aware); every "
                "backward_grad_dots mode itself composes with varlen"
            )
    bias = (
        query.new_empty((0,), dtype=torch.float32)
        if key_pad_bias is None
        else key_pad_bias
    )
    cu = (
        query.new_empty((0,), dtype=torch.int32)
        if cu_seqlens is None
        else cu_seqlens
    )
    empty_u8 = query.new_empty((0,), dtype=torch.uint8)
    qnv_in, qsc_in = (
        (q_packs[0].view(torch.uint8), q_packs[1].view(torch.uint8))
        if q_packs is not None
        else (empty_u8, empty_u8)
    )
    knv_in, ksc_in = (
        (k_packs[0].view(torch.uint8), k_packs[1].view(torch.uint8))
        if k_packs is not None
        else (empty_u8, empty_u8)
    )
    sr = stochastic_rounding
    p_dv = (
        sr
        if backward_p_dv_stochastic_rounding is None
        else backward_p_dv_stochastic_rounding
    )
    dot_dv = (
        sr
        if backward_dot_dv_stochastic_rounding is None
        else backward_dot_dv_stochastic_rounding
    )
    ds_dq = (
        sr
        if backward_ds_dq_stochastic_rounding is None
        else backward_ds_dq_stochastic_rounding
    )
    # custom-op schemas have no Optional[str]; the EMPTY STRING is the
    # "None = kernel AUTO" sentinel. Resolution happens in the backward op,
    # where the EFFECTIVE saved-packs decision (incl. the seq-length gate) and
    # the effective sequence (varlen mean sample length) are both known.
    grad_dots_str = backward_grad_dots or ""
    out, *_aux = torch.ops.axolotl_nvfp4.flash_attention_train(
        query,
        key,
        value,
        bias,
        cu,
        qnv_in,
        qsc_in,
        knv_in,
        ksc_in,
        scaling,
        causal,
        num_key_value_groups,
        sr,
        p_dv,
        dot_dv,
        ds_dq,
        dkdv_scratch_bf16,
        grad_dots_str,
        save_backward_packs,
        out_layout == "zshd",
        block_m,
        block_n,
        num_warps,
        num_stages,
    )
    return out
