"""Opaque torch custom ops wrapping the FLA GatedDeltaNet training kernels.

Why: with ``torch_compile: true`` the transformers Qwen3.5 decoder loop
(``Qwen3_5TextModel.forward``) can only compile if its body traces with ZERO
graph breaks — dynamo cannot resume from a break inside a for loop, so a single
break makes it skip the whole frame ("graph break in a for/while loop",
reported against modeling_qwen3_5.py:1235) and every decoder layer runs
eagerly. The FLA fast path breaks three ways:

- ``get_cu_seqlens`` needs ``aten.nonzero`` (data-dependent output shape);
- ``fla.ops.gated_delta_rule.chunk_gated_delta_rule`` is wrapped in
  ``@torch.compiler.disable``;
- ``fla.modules.convolution.causal_conv1d`` internals
  (``prepare_chunk_indices``) branch on tensor data (``.item()``).

Each wrapper below re-exposes the SAME FLA kernels (identical math, identical
saved tensors, no extra recompute) as opaque custom ops with fake impls, so the
whole linear-attention layer stays inside one dynamo graph: Inductor compiles
and fuses AROUND the opaque nodes instead of breaking on them. This follows the
``_build_fla_rmsnorm_gated_op`` precedent in ``modeling.py`` (backward is its
own opaque op because AOT autograd traces the backward graph too).

The ops take ``position_ids`` (not ``cu_seqlens``) and derive cu_seqlens
EAGERLY inside the opaque region: cu_seqlens has a data-dependent length, so
keeping it out of the traced graph means every fake impl is static-shaped — no
unbacked SymInts, no ``capture_dynamic_output_shape_ops`` (dynamo classifies
even a custom op with a dynamic-size fake as a graph-breaking "dynamic shape
operator" without that global flag). The nonzero is a [T]-element kernel, noise
next to the FLA kernels each op launches.

Training path only (no cache, no initial state): the decode/cache paths keep
calling FLA eagerly via the original entry points.
"""

from __future__ import annotations

import torch

__all__ = ["fla_ops_available"]

_OPS_BUILT = False
_OPS_BUILD_ERROR: str | None = None


def _cu_seqlens_from_position_ids(position_ids: torch.Tensor) -> torch.Tensor:
    """Same math as ``modeling.get_cu_seqlens`` (int32 starts + total len)."""
    if position_ids.ndim == 3:  # MRoPE [axes, B, T]: all axes share temporal pos
        position_ids = position_ids[0]
    pos = position_ids.reshape(-1)
    tensor_kwargs = {"dtype": torch.int32, "device": pos.device}
    indices_q = (pos == 0).nonzero().view(-1)
    return torch.cat(
        (
            indices_q.to(**tensor_kwargs),
            torch.tensor(pos.size(), **tensor_kwargs),
        )
    )


def _build_ops() -> None:
    """Register the custom ops once. Raises if FLA is unavailable."""
    global _OPS_BUILT
    if _OPS_BUILT:
        return

    from fla.modules.convolution import causal_conv1d_bwd, causal_conv1d_fwd
    from fla.modules.l2norm import l2norm_bwd, l2norm_fwd
    from fla.ops.gated_delta_rule.chunk import (
        chunk_gated_delta_rule_bwd,
        chunk_gated_delta_rule_fwd,
    )

    def _cu(position_ids: torch.Tensor | None) -> torch.Tensor | None:
        if position_ids is None:
            return None
        return _cu_seqlens_from_position_ids(position_ids)

    # ------------------------------------------------------------------
    # Varlen causal conv1d (training path: no residual/initial state/final
    # state). Forward and backward call the same FLA host functions that
    # CausalConv1dFunction dispatches to.
    # ------------------------------------------------------------------
    @torch.library.custom_op("axolotl_qwen3_5::gdn_conv", mutates_args=())
    def _gdn_conv(
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        activation: str | None,
        position_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        y, _ = causal_conv1d_fwd(
            x=x.contiguous(),
            weight=weight.contiguous(),
            bias=bias.contiguous() if bias is not None else None,
            residual=None,
            initial_state=None,
            output_final_state=False,
            activation=activation,
            cu_seqlens=_cu(position_ids),
        )
        return y

    @_gdn_conv.register_fake
    def _(x, weight, bias, activation, position_ids):
        return torch.empty(x.shape, dtype=x.dtype, device=x.device)

    @torch.library.custom_op("axolotl_qwen3_5::gdn_conv_bwd", mutates_args=())
    def _gdn_conv_bwd(
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        dy: torch.Tensor,
        activation: str | None,
        position_ids: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dx, dw, db, _, _ = causal_conv1d_bwd(
            x=x.contiguous(),
            dy=dy.contiguous(),
            dht=None,
            weight=weight.contiguous(),
            bias=bias.contiguous() if bias is not None else None,
            residual=None,
            initial_state=None,
            activation=activation,
            cu_seqlens=_cu(position_ids),
        )
        if db is None:
            db = weight.new_empty(0)
        return dx, dw, db

    @_gdn_conv_bwd.register_fake
    def _(x, weight, bias, dy, activation, position_ids):
        db_shape = bias.shape if bias is not None else (0,)
        db_dtype = bias.dtype if bias is not None else weight.dtype
        return (
            torch.empty(x.shape, dtype=x.dtype, device=x.device),
            torch.empty(weight.shape, dtype=weight.dtype, device=weight.device),
            torch.empty(db_shape, dtype=db_dtype, device=weight.device),
        )

    def _gdn_conv_setup(ctx, inputs, output):
        x, weight, bias, activation, position_ids = inputs
        ctx.activation = activation
        ctx.has_bias = bias is not None
        ctx.save_for_backward(x, weight, bias, position_ids)

    def _gdn_conv_backward(ctx, dy):
        x, weight, bias, position_ids = ctx.saved_tensors
        dx, dw, db = torch.ops.axolotl_qwen3_5.gdn_conv_bwd(
            x, weight, bias, dy, ctx.activation, position_ids
        )
        return dx, dw, (db if ctx.has_bias else None), None, None

    _gdn_conv.register_autograd(_gdn_conv_backward, setup_context=_gdn_conv_setup)

    # ------------------------------------------------------------------
    # chunk_gated_delta_rule, training path (initial_state=None,
    # output_final_state=False, use_qk_l2norm_in_kernel=True — the only
    # configuration Qwen3.5 uses in the no-cache forward). Mirrors
    # ChunkGatedDeltaRuleFunction 1:1: the forward also returns the tensors
    # the FLA autograd would save (post-l2norm q/k + rstds, cumsum'd g, A)
    # so the backward runs the native FLA backward with NO recompute.
    # ------------------------------------------------------------------
    @torch.library.custom_op("axolotl_qwen3_5::gdn_chunk", mutates_args=())
    def _gdn_chunk(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float,
        position_ids: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        # The eager call site does g.to(query.dtype) right at the kernel call;
        # doing the cast INSIDE the op keeps the op's g input float32, so the
        # f32 dg the FLA backward produces matches the input dtype exactly
        # (bit-identical grad flow to eager — no extra bf16 round-trip).
        g, beta = g.contiguous().to(q.dtype), beta.contiguous()
        qn, q_rstd = l2norm_fwd(q)
        kn, k_rstd = l2norm_fwd(k)
        g_cum, o, A, _ = chunk_gated_delta_rule_fwd(
            q=qn,
            k=kn,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=None,
            output_final_state=False,
            cu_seqlens=_cu(position_ids),
        )
        return o.to(q.dtype), qn, q_rstd, kn, k_rstd, g_cum, A

    @_gdn_chunk.register_fake
    def _(q, k, v, g, beta, scale, position_ids):
        def emp(shape, dtype):
            return torch.empty(shape, dtype=dtype, device=q.device)

        return (
            emp(v.shape, q.dtype),  # o
            emp(q.shape, q.dtype),  # qn
            emp(q.shape[:-1], torch.float32),  # q_rstd
            emp(k.shape, k.dtype),  # kn
            emp(k.shape[:-1], torch.float32),  # k_rstd
            emp(g.shape, torch.float32),  # g_cum (chunk_local_cumsum output_dtype)
            emp((*k.shape[:-1], 64), k.dtype),  # A (chunk 64, solve_tril -> k dtype)
        )

    @torch.library.custom_op("axolotl_qwen3_5::gdn_chunk_bwd", mutates_args=())
    def _gdn_chunk_bwd(
        qn: torch.Tensor,
        q_rstd: torch.Tensor,
        kn: torch.Tensor,
        k_rstd: torch.Tensor,
        v: torch.Tensor,
        g_cum: torch.Tensor,
        beta: torch.Tensor,
        A: torch.Tensor,
        do: torch.Tensor,
        scale: float,
        position_ids: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Mirror FLA's input_guard: the kernels hard-code contiguous strides,
        # and v arrives as a non-contiguous split/reshape VIEW of mixed_qkv
        # (the eager path saved input_guard's contiguous copy at forward time).
        qn, kn, v = qn.contiguous(), kn.contiguous(), v.contiguous()
        g_cum, beta, A = g_cum.contiguous(), beta.contiguous(), A.contiguous()
        dq, dk, dv, db, dg, _ = chunk_gated_delta_rule_bwd(
            q=qn,
            k=kn,
            v=v,
            g=g_cum,
            beta=beta,
            A=A,
            scale=scale,
            initial_state=None,
            do=do.contiguous(),
            dht=None,
            cu_seqlens=_cu(position_ids),
        )
        dq = l2norm_bwd(qn, q_rstd, dq)
        dk = l2norm_bwd(kn, k_rstd, dk)
        # same final casts as ChunkGatedDeltaRuleFunction.backward
        return (
            dq.to(qn.dtype),
            dk.to(kn.dtype),
            dv.to(v.dtype),
            dg.to(g_cum.dtype),
            db.to(beta.dtype),
        )

    @_gdn_chunk_bwd.register_fake
    def _(qn, q_rstd, kn, k_rstd, v, g_cum, beta, A, do, scale, position_ids):
        def emp(shape, dtype):
            return torch.empty(shape, dtype=dtype, device=qn.device)

        return (
            emp(qn.shape, qn.dtype),
            emp(kn.shape, kn.dtype),
            emp(v.shape, v.dtype),
            emp(g_cum.shape, g_cum.dtype),
            emp(beta.shape, beta.dtype),
        )

    def _gdn_chunk_setup(ctx, inputs, output):
        q, k, v, g, beta, scale, position_ids = inputs
        _, qn, q_rstd, kn, k_rstd, g_cum, A = output
        ctx.scale = scale
        ctx.save_for_backward(qn, q_rstd, kn, k_rstd, v, g_cum, beta, A, position_ids)

    def _gdn_chunk_backward(ctx, do, *unused_intermediate_grads):
        qn, q_rstd, kn, k_rstd, v, g_cum, beta, A, position_ids = ctx.saved_tensors
        dq, dk, dv, dg, db = torch.ops.axolotl_qwen3_5.gdn_chunk_bwd(
            qn, q_rstd, kn, k_rstd, v, g_cum, beta, A, do, ctx.scale, position_ids
        )
        # Bit-exact match with the eager path: there, the autograd engine casts
        # FLA's float32 dg to the g.to(query.dtype) cast node's bf16 before the
        # cast backward upcasts it again — i.e. eager dg lands on the bf16
        # grid. Reproduce that f32 -> bf16 -> f32 round-trip here.
        dg = dg.to(qn.dtype).to(g_cum.dtype)
        return dq, dk, dv, dg, db, None, None

    _gdn_chunk.register_autograd(_gdn_chunk_backward, setup_context=_gdn_chunk_setup)

    _OPS_BUILT = True


def fla_ops_available() -> bool:
    """Build the ops if needed; True when the FLA-backed custom ops exist."""
    global _OPS_BUILD_ERROR
    if _OPS_BUILT:
        return True
    if _OPS_BUILD_ERROR is not None:
        return False
    try:
        _build_ops()
    except Exception as exc:  # pragma: no cover - depends on fla install
        _OPS_BUILD_ERROR = str(exc)
        return False
    return True
