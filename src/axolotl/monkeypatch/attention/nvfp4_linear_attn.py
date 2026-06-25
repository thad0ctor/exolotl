"""Native-NVFP4 acceleration for Qwen3.5 LINEAR-attention (GatedDeltaNet) layers.

Qwen3.5 is HYBRID: 24 of 32 layers (9B) are ``linear_attention`` (GatedDeltaNet,
gated delta rule), the other 8 are full softmax attention. The full-attn layers are
handled by ``nvfp4_flash_attn``; this module handles the linear-attn layers.

Profiling a 9B GatedDeltaNet prefill (S=4096, idx-3 RTX PRO 6000) shows the time is
dominated by THREE large-K linear projections, not the delta-rule recurrence:

    in_proj_qkv  26%   (K=4096 -> N=8192)
    delta_rule   26%   (FLA chunked Triton, fp32, head_dim=128, small-K -> left bf16)
    out_proj     15%   (K=4096 -> N=4096)
    in_proj_z    15%   (K=4096 -> N=4096)
    conv/norm/ab  rest (memory-bound, small -> left bf16)

The three projections (~57% at S=4096, ~64% at S=8192) have K=4096 contraction — the
regime where native NVFP4 ``tl.dot_scaled`` beats bf16 (microbenched 1.7-2.3x incl.
activation quant on these exact shapes). The delta-rule kernel runs in fp32 with
chunk/head contractions of 64-128 (small-K, recurrent, numerically sensitive); FP4
there is both unlikely to win and parity-risky, so it is left untouched.

Novel cross-op fusion: ``in_proj_qkv`` and ``in_proj_z`` consume the SAME
``hidden_states`` and contract over the SAME K=4096. The activation is NVFP4-packed
ONCE and reused for both GEMMs (one quant pass feeds two FP4 tensor-core ops),
removing a redundant activation round-trip. ``in_proj_a/b`` (N=32, memory-bound) and
``out_proj`` (a different activation, post delta-rule) keep their own paths.

Forward/inference only (no autograd). Opt-in: call
``patch_qwen3_5_nvfp4_linear_attn(model)``; default OFF. Adapter-wrapped
projection modules are skipped; plain trainable weights are re-packed after
optimizer updates when no-grad eval next uses the fast path.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import triton
from torch import nn

from axolotl.kernels.attn_nvfp4_flash import _quant_nvfp4
from axolotl.kernels.nvfp4_linear import (
    _nvfp4_gemm_kernel,
    nvfp4_linear,
    prepack_weight_nvfp4,
)
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


def _is_plain_linear(module: nn.Module) -> bool:
    return type(module) is nn.Linear


def _get_packed_weight(
    owner: nn.Module, cache_attr: str, linear: nn.Linear
) -> tuple[torch.Tensor, torch.Tensor]:
    weight = linear.weight
    key = (weight.data_ptr(), weight._version, tuple(weight.shape), weight.dtype)
    cached = getattr(owner, cache_attr, None)
    if cached is None or cached[0] != key:
        wnv, wsc = prepack_weight_nvfp4(weight.detach())
        setattr(owner, cache_attr, (key, wnv, wsc))
        cached = getattr(owner, cache_attr)
    return cached[1], cached[2]


def _position_ids_are_dense_unpacked(
    position_ids: torch.Tensor | None, seq_len: int
) -> bool:
    if position_ids is None:
        return True
    if position_ids.ndim == 3:
        position_ids = position_ids[0]
    if position_ids.ndim != 2 or position_ids.shape[-1] != seq_len:
        return False
    ref = torch.arange(seq_len, device=position_ids.device, dtype=position_ids.dtype)
    return bool(torch.equal(position_ids, ref.expand_as(position_ids)))


def _call_orig_forward(
    orig_forward,
    module,
    hidden_states,
    cache_params,
    attention_mask,
    cache_position,
    position_ids,
    kwargs,
):
    if cache_position is None and position_ids is None and not kwargs:
        return orig_forward(module, hidden_states, cache_params, attention_mask)
    return orig_forward(
        module,
        hidden_states,
        cache_params=cache_params,
        attention_mask=attention_mask,
        cache_position=cache_position,
        position_ids=position_ids,
        **kwargs,
    )


def _gemm_from_packed_act(anv, asc, wnv, wsc, m, n_out, k, out_dtype):
    """FP4 GEMM reusing an ALREADY-packed activation (the shared-quant fast path).

    ``anv``/``asc`` are the packed activation ([m, k//2] / [m, k//16]); ``wnv``/``wsc``
    the pre-packed weight. Returns ``[m, n_out]``.
    """
    out = torch.empty(m, n_out, device=anv.device, dtype=out_dtype)
    block_m, block_n, block_k = 128, 128, 256
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
        num_warps=8,
        num_stages=3,
    )
    return out


def make_nvfp4_gdn_forward(orig_forward):
    """Patched ``Qwen3_5GatedDeltaNet.forward`` routing the three large-K
    projections through native NVFP4, with the in-proj activation shared.

    Falls back to ``orig_forward`` for: grad enabled, or any cached-state use
    (decode / chunked continuation) — the fast path serves dense prefill where the
    GEMM cost dominates and parity is straightforward.
    """

    def forward(
        self,
        hidden_states,
        cache_params=None,
        attention_mask=None,
        cache_position=None,
        position_ids=None,
        **kwargs,
    ):
        use_cache = cache_params is not None and cache_params.has_previous_state(
            self.layer_idx
        )
        if (
            torch.is_grad_enabled()
            or use_cache
            or kwargs
            or not _position_ids_are_dense_unpacked(
                position_ids, hidden_states.shape[1]
            )
        ):
            return _call_orig_forward(
                orig_forward,
                self,
                hidden_states,
                cache_params,
                attention_mask,
                cache_position,
                position_ids,
                kwargs,
            )

        from transformers.models.qwen3_5.modeling_qwen3_5 import (
            apply_mask_to_padding_states,
        )

        hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
        batch_size, seq_len, _ = hidden_states.shape

        # NOVEL FUSION: pack hidden_states to NVFP4 ONCE, feed both in_proj_qkv
        # and in_proj_z (same activation, same K contraction).
        k = hidden_states.shape[-1]
        hs2d = hidden_states.reshape(-1, k)
        m = hs2d.shape[0]
        anv, asc = _quant_nvfp4(hs2d.unsqueeze(0))
        anv = anv[0]
        asc = asc[0]
        out_dtype = hidden_states.dtype
        qkv_wnv, qkv_wsc = _get_packed_weight(self, "_qkv_packed", self.in_proj_qkv)
        z_wnv, z_wsc = _get_packed_weight(self, "_z_packed", self.in_proj_z)

        mixed_qkv = _gemm_from_packed_act(
            anv, asc, qkv_wnv, qkv_wsc, m, self.conv_dim, k, out_dtype
        ).reshape(batch_size, seq_len, self.conv_dim)
        z = _gemm_from_packed_act(
            anv, asc, z_wnv, z_wsc, m, self.value_dim, k, out_dtype
        ).reshape(batch_size, seq_len, -1, self.head_v_dim)

        mixed_qkv = mixed_qkv.transpose(1, 2)
        # a/b stay bf16 (N=num_v_heads, memory-bound)
        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)

        if cache_params is not None:
            new_conv_state = F.pad(
                mixed_qkv, (self.conv_kernel_size - mixed_qkv.shape[-1], 0)
            )
            cache_params.update_conv_state(new_conv_state, self.layer_idx)
        if self.causal_conv1d_fn is not None:
            mixed_qkv = self.causal_conv1d_fn(
                x=mixed_qkv,
                weight=self.conv1d.weight.squeeze(1),
                bias=self.conv1d.bias,
                activation=self.activation,
                seq_idx=None,
            )
        else:
            mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, : mixed_qkv.shape[-1]])

        mixed_qkv = mixed_qkv.transpose(1, 2)
        query, key, value = torch.split(
            mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1
        )
        query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        if self.num_v_heads // self.num_k_heads > 1:
            rep = self.num_v_heads // self.num_k_heads
            query = query.repeat_interleave(rep, dim=2)
            key = key.repeat_interleave(rep, dim=2)

        core_attn_out, last_recurrent_state = self.chunk_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=None,
            output_final_state=cache_params is not None,
            use_qk_l2norm_in_kernel=True,
        )
        if cache_params is not None:
            cache_params.update_recurrent_state(last_recurrent_state, self.layer_idx)

        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)

        # out_proj: separate activation (post delta-rule), its own NVFP4 path.
        out_wnv, out_wsc = _get_packed_weight(self, "_out_packed", self.out_proj)
        output = nvfp4_linear(core_attn_out, out_wnv, out_wsc, self.hidden_size)
        return output

    return forward


def patch_qwen3_5_nvfp4_linear_attn(model: nn.Module) -> int:
    """Patch every Qwen3.5 LINEAR-attention (GatedDeltaNet) layer to route its
    large-K projections through native NVFP4. Prepacks the three projection weights
    once. Returns the count of patched layers. Idempotent.
    """
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet

    patched = 0
    seen_forward = None
    for module in model.modules():
        if isinstance(module, Qwen3_5GatedDeltaNet):
            if getattr(module, "_nvfp4_patched", False):
                continue
            if not (
                _is_plain_linear(module.in_proj_qkv)
                and _is_plain_linear(module.in_proj_z)
                and _is_plain_linear(module.out_proj)
            ):
                continue
            orig = type(module).forward
            if seen_forward is None:
                seen_forward = make_nvfp4_gdn_forward(orig)
            module.forward = seen_forward.__get__(module, type(module))
            module._nvfp4_patched = True
            patched += 1
    LOG.info("nvfp4 linear-attn: patched %d Qwen3.5 GatedDeltaNet layers", patched)
    return patched
