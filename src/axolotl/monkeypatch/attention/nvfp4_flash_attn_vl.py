"""Native-NVFP4 attention for Qwen3-VL dense full-attention (LM) layers.

Qwen3-VL uses STANDARD (non-gated) softmax attention with multimodal mRoPE,
unlike Qwen3.5's gated DeltaNet-hybrid attention — so it needs its own forward
rather than the Qwen3.5 patch (which assumes q_proj -> head_dim*2 + sigmoid gate
and Qwen3.5's plain RoPE). This patches ``Qwen3VLTextAttention`` to run native
NVFP4 attention on the dense causal/full, cache-free PREFILL path in both grad
(training) and no-grad (eval / first generation token) modes — fed VL's
already-roped Q/K via the model's own ``apply_rotary_pos_emb`` (rope-agnostic op).
It falls back to the model's configured attention (flash_attention_2) for KV-cache
decode, non-dense / per-key-padded masks, and ``output_attentions``. head_dim 128
is supported by the kernel (d in {128, 256}). Vision attention
(Qwen3VLVisionBlock) is left untouched.

Sample-packed (multipack) batches route through the varlen (cu_seqlens) NVFP4
path like the Qwen3.5 patch — boundaries are derived from position-0 resets in
the flattened text-axis position_ids (the only form the HF Qwen3-VL text model
forwards to its attention layers; multi-axis mRoPE tensors never reach them),
validated conservatively, and gated by ``packed_min_sample_len``. Any ambiguous
boundary signal falls back to the original forward.

Opt-in via ``nvfp4_training.attention`` on a qwen3_vl model; default OFF.
"""

from __future__ import annotations

import torch
from torch import nn

from axolotl.kernels.attn_nvfp4_flash import nvfp4_flash_attn_func
from axolotl.monkeypatch.attention.nvfp4_flash_attn import (
    _PACKED_MIN_SAMPLE_LEN_DEFAULT,
    _compute_packed_info,
    _custom_op_enabled,
    _grad_dots_kwargs,
    _log_packed_gate_once,
    _log_packed_save_packs_ignored_once,
    _mask_is_dense_causal_or_full,
    _packed_position_ids_info,
    _resolve_grad_dots_alias,
)

# VL-specific packed-gate default: the VL packed path does not (yet) feed the
# pre-packed fused-producer q/k, and at Qwen3-VL's head_dim 128 FP4 varlen
# measured 8-27% SLOWER fwd+bwd than FA2-varlen even at long means — its value
# is numerical consistency with the FP4 forward, not speed. So unlike the text
# patch (default 0 = always FP4), VL keeps short/medium packs on FA2 unless the
# user sets packed_min_sample_len explicitly.
_VL_PACKED_MIN_SAMPLE_LEN_DEFAULT = 1024
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


# ---------------------------------------------------------------------------
# Packed-sequence (sample-packing / multipack) detection for Qwen3-VL.
#
# What the attention layer actually receives as ``position_ids`` (transformers
# 5.8.x, models/qwen3_vl/modeling_qwen3_vl.py, Qwen3VLTextModel.forward):
#
#   * user passes 2D ``[B, T]`` (axolotl text-pack collators do exactly this):
#     it is expanded to ``[4, B, T]`` with ALL axes identical, and the decoder
#     layers get ``text_position_ids = position_ids[0]`` — i.e. the original
#     packed ``[B, T]`` whose position-0 resets ARE the sample boundaries;
#   * user passes/model computes the 3-axis mRoPE ``[3, B, T]``
#     (``get_rope_index`` output for image/video batches): the text model sets
#     ``text_position_ids = None`` — the multi-axis tensor feeds ONLY the
#     rotary embedding, never the attention layers.
#
# So the vision-token mRoPE structure (temporal axis constant across an image
# run, gridded h/w axes — the thing that would fake or hide position-0 resets)
# can never reach this kwarg through the HF text model. The classification
# below still defends against non-standard callers handing a multi-axis tensor
# straight to the attention module, and validates the derived boundaries:
# anything ambiguous falls back to the model's original forward — a packed
# batch is NEVER dense-attended on a possibly-wrong block-diagonal mask.
# ---------------------------------------------------------------------------
_VL_PACKED_INFO_ATTR = "_nvfp4_vl_packed_info"
_VL_PACKED_AMBIGUOUS_LOGGED = False


def _log_vl_packed_ambiguous_once() -> None:
    global _VL_PACKED_AMBIGUOUS_LOGGED
    if not _VL_PACKED_AMBIGUOUS_LOGGED:
        _VL_PACKED_AMBIGUOUS_LOGGED = True
        LOG.warning(
            "nvfp4 attention (qwen3_vl): packed batch with an ambiguous "
            "boundary signal (multi-axis mRoPE position_ids, vision-style "
            "repeated/reordered positions, or a multi-row pack); falling back "
            "to the model's original attention for packed batches"
        )


def _compute_packed_info_vl(
    position_ids: torch.Tensor, q_len: int
) -> tuple[str | None, torch.Tensor | None, float | None]:
    """Qwen3-VL packed classification: like ``_compute_packed_info`` but with a
    conservative validity check on the derived boundaries.

    Returns ("packed", cu_seqlens, mean_sample_len) only when position_ids is
    (equivalent to) a flattened text-axis pack — every step is +1 or a reset to
    0, so each derived segment is EXACTLY arange(len): boundaries strictly
    increasing, every sample starting at position 0, [0, S) fully covered.
    Vision-token mRoPE runs (repeated temporal / gridded h-w positions) fail
    this and return ("fallback", None, None) -> original forward, never a
    wrong mask.
    """
    pos = position_ids
    if pos.ndim == 3:
        # The HF text model never forwards a multi-axis mRoPE tensor here (it
        # strips [3, B, T] to None and [4, B, T] to its text axis). A 3D tensor
        # means a non-standard caller: accept it only when every axis carries
        # the SAME positions (text-only degenerate case) — otherwise the
        # boundary signal is ambiguous.
        if not bool((pos == pos[:1]).all()):
            return ("fallback", None, None)
        pos = pos[0]
    if pos.ndim == 1:
        pos = pos.unsqueeze(0)
    if pos.ndim != 2 or pos.shape[-1] != q_len:
        return (None, None, None)
    info = _compute_packed_info(pos, q_len)
    if info[0] != "packed":
        return info
    # Validity: combined with pos[0, 0] == 0 (required by _compute_packed_info)
    # this forces every segment between 0-resets to be exactly arange(len).
    p = pos[0]
    if not bool(((p[1:] == p[:-1] + 1) | (p[1:] == 0)).all()):
        return ("fallback", None, None)
    return info


def _vl_packed_position_ids_info(
    position_ids: torch.Tensor | None, q_len: int
) -> tuple[str | None, torch.Tensor | None, float | None]:
    """Once-per-step cached VL packed classification (same caching pattern as
    the Qwen3.5 patch: all decoder layers see the same position_ids tensor)."""
    return _packed_position_ids_info(
        position_ids,
        q_len,
        compute=_compute_packed_info_vl,
        cache_attr=_VL_PACKED_INFO_ATTR,
    )


def make_nvfp4_vl_forward(orig_forward):
    """Build a patched ``Qwen3VLTextAttention.forward`` using NVFP4 attention.

    Runs NVFP4 on the dense causal/full, cache-free PREFILL path in BOTH grad
    (training) and no-grad (eval / first generation token) modes — mirroring the
    Qwen3.5 patch, so the two passes are numerically consistent (matters under
    gradient checkpointing). On the no-grad path it writes the roped K/V to the
    cache so subsequent decode steps see prefill context. Falls back to
    ``orig_forward`` (flash_attention_2) for: KV-cache decode (prior cached
    context — a dedicated FP4 decode kernel is a proven loser), per-key-padded /
    non-dense masks, ``output_attentions``, grad mode without a trained
    backward, and packed batches that are gated out (packed_min_sample_len) or
    whose boundary signal is ambiguous. The Qwen3.5-only inference fast paths
    (fuse_vproj / fp4_projections) are intentionally omitted here.
    """

    def forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask=None,
        past_key_values=None,
        **kwargs,
    ):
        if kwargs.get("output_attentions"):
            return orig_forward(
                self,
                hidden_states,
                position_embeddings,
                attention_mask,
                past_key_values,
                **kwargs,
            )

        grad_enabled = torch.is_grad_enabled()
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        q_len = input_shape[-1]

        # NVFP4 fast path is PREFILL only (no prior cached context) on a dense
        # causal/full mask. Cached decode reuses prior roped K/V and FP4 decode is
        # a proven loser, so it goes to stock attention.
        has_cache_context = (
            past_key_values is not None
            and past_key_values.get_seq_length(self.layer_idx) > 0
        )
        kind = None
        cu_seqlens = None
        if not has_cache_context:
            kind = _mask_is_dense_causal_or_full(attention_mask, q_len, q_len)
        if kind == "causal" and attention_mask is None:
            # Sample-packed (multipack) batch: boundaries live in position_ids.
            # The kwarg here is the TEXT axis (see the module block comment):
            # axolotl text packs arrive as flattened [1, T] per-sample-arange
            # positions, so position-0 resets ARE the boundaries. Anything the
            # validity check can't certify (multi-axis mRoPE, vision-style
            # repeats, multi-row packs) falls back to the original forward —
            # a packed batch is NEVER dense-attended.
            pkind, cu_seqlens, mean_len = _vl_packed_position_ids_info(
                kwargs.get("position_ids"), q_len
            )
            if pkind == "packed":
                # PERF GATE (same as the Qwen3.5 patch): short-mean packs leave
                # nothing quadratic for FP4 to win while its quant prologue is
                # linear in tokens — the original FA2-varlen forward is strictly
                # faster there.
                min_len = getattr(
                    self,
                    "_nvfp4_packed_min_sample_len",
                    _VL_PACKED_MIN_SAMPLE_LEN_DEFAULT,
                )
                if min_len > 0 and mean_len < min_len:
                    _log_packed_gate_once(mean_len, min_len, use_fp4=False)
                    kind = None
                else:
                    _log_packed_gate_once(mean_len, min_len, use_fp4=True)
                    kind = "packed"
            elif pkind == "fallback":
                _log_vl_packed_ambiguous_once()
                kind = None
        if kind is None:
            return orig_forward(
                self,
                hidden_states,
                position_embeddings,
                attention_mask,
                past_key_values,
                **kwargs,
            )
        # Grad mode needs the trained backward; a cache during grad is unexpected.
        if grad_enabled and (
            not getattr(self, "_nvfp4_train_backward", False)
            or past_key_values is not None
        ):
            return orig_forward(
                self,
                hidden_states,
                position_embeddings,
                attention_mask,
                past_key_values,
                **kwargs,
            )

        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            apply_rotary_pos_emb,
        )

        # Standard Qwen3-VL projections (no gate): q/k_norm on head_dim, then mRoPE.
        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(
            1, 2
        )
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(
            1, 2
        )
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_roped, key_roped = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # No-grad prefill: stash roped K/V so later decode steps see this context.
        if not grad_enabled and past_key_values is not None:
            past_key_values.update(key_roped, value_states, self.layer_idx)

        ng = query_states.shape[1] // key_states.shape[1]
        sr = getattr(self, "_nvfp4_stochastic_rounding", True)
        grad_sr = sr and not getattr(self, "_nvfp4_backward_rtn_grad_packs", False)
        save_packs = getattr(self, "_nvfp4_save_backward_packs", False)
        dkdv_bf16 = getattr(self, "_nvfp4_dkdv_scratch_bf16", False)

        if kind == "packed":
            # Varlen block-diagonal causal attention over the flattened pack
            # (work ~ sum(s_i^2), no cross-sample attention). Varlen is
            # incompatible with the forward saved-pack set ONLY: save_packs is
            # forced off for packed batches (one-time INFO; config still
            # applies to dense batches), while the configured grad_dots mode
            # passes through — every mode composes with varlen, and None =
            # kernel AUTO resolves on the pack's mean sample length. Same
            # handling as the Qwen3.5 patch.
            if save_packs:
                _log_packed_save_packs_ignored_once()
            if _custom_op_enabled(self):
                from axolotl.kernels.attn_nvfp4_custom_op import (
                    nvfp4_flash_attn_train_custom_op,
                )

                torch._dynamo.graph_break()
                attn_output = nvfp4_flash_attn_train_custom_op(
                    query_roped,
                    key_roped,
                    value_states,
                    self.scaling,
                    causal=True,
                    num_key_value_groups=ng,
                    cu_seqlens=cu_seqlens,
                    stochastic_rounding=sr,
                    save_backward_packs=False,
                    backward_grad_dots=getattr(self, "_nvfp4_grad_dots", None),
                    out_layout="zshd",
                )  # [Z, S, H, D]
                torch._dynamo.graph_break()
            else:
                attn_output = nvfp4_flash_attn_func(
                    query_roped,
                    key_roped,
                    value_states,
                    self.scaling,
                    causal=True,
                    num_key_value_groups=ng,
                    cu_seqlens=cu_seqlens,
                    stochastic_rounding=sr,
                    save_backward_packs=False,
                    **_grad_dots_kwargs(self),
                ).transpose(1, 2)  # [Z, H, S, D] -> [Z, S, H, D]
            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            return self.o_proj(attn_output), None

        # Same NVFP4 op for grad and no-grad (consistent under checkpointing). The
        # backward SR knobs are inert in no-grad. Under compile the opaque custom op
        # keeps Inductor from tracing tl.dot_scaled (decompose crash); the graph
        # breaks give it the eager boundary.
        if _custom_op_enabled(self):
            from axolotl.kernels.attn_nvfp4_custom_op import (
                nvfp4_flash_attn_train_custom_op,
            )

            torch._dynamo.graph_break()
            attn_output = nvfp4_flash_attn_train_custom_op(
                query_roped,
                key_roped,
                value_states,
                self.scaling,
                causal=(kind == "causal"),
                num_key_value_groups=ng,
                stochastic_rounding=sr,
                backward_p_dv_stochastic_rounding=grad_sr,
                backward_dot_dv_stochastic_rounding=grad_sr,
                backward_ds_dq_stochastic_rounding=grad_sr,
                dkdv_scratch_bf16=dkdv_bf16,
                backward_grad_dots=getattr(self, "_nvfp4_grad_dots", None),
                save_backward_packs=save_packs,
                out_layout="zshd",
            )  # [Z, S, H, D]
            torch._dynamo.graph_break()
        else:
            attn_output = nvfp4_flash_attn_func(
                query_roped,
                key_roped,
                value_states,
                self.scaling,
                causal=(kind == "causal"),
                num_key_value_groups=ng,
                stochastic_rounding=sr,
                backward_p_dv_stochastic_rounding=grad_sr,
                backward_dot_dv_stochastic_rounding=grad_sr,
                backward_ds_dq_stochastic_rounding=grad_sr,
                save_backward_packs=save_packs,
                dkdv_scratch_bf16=dkdv_bf16,
                **_grad_dots_kwargs(self),
            ).transpose(1, 2)  # [Z, H, S, D] -> [Z, S, H, D]

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        return self.o_proj(attn_output), None

    return forward


def _set_attrs(module, *, train_backward, backward_rtn_grad_packs, save_backward_packs,
               dkdv_scratch_bf16, grad_dots, compile_custom_op,
               stochastic_rounding, packed_min_sample_len):
    module._nvfp4_train_backward = train_backward
    module._nvfp4_backward_rtn_grad_packs = backward_rtn_grad_packs
    module._nvfp4_save_backward_packs = save_backward_packs
    module._nvfp4_dkdv_scratch_bf16 = dkdv_scratch_bf16
    module._nvfp4_grad_dots = grad_dots
    module._nvfp4_compile_custom_op = compile_custom_op
    module._nvfp4_stochastic_rounding = stochastic_rounding
    module._nvfp4_packed_min_sample_len = packed_min_sample_len


def patch_qwen3_vl_nvfp4_attention(
    model: nn.Module,
    *,
    train_backward: bool = False,
    backward_rtn_grad_packs: bool = False,
    save_backward_packs: bool = False,
    dkdv_scratch_bf16: bool = False,
    grad_dots: str | None = None,
    bf16_grad_dots: bool | None = None,
    compile_custom_op: bool = False,
    stochastic_rounding: bool = True,
    packed_min_sample_len: int | None = None,
) -> int:
    """Patch every Qwen3-VL LM full-attention layer's forward to use NVFP4.

    Leaves vision-encoder attention untouched. Idempotent. Returns patched count.

    ``grad_dots``: backward grad-GEMM mode ("bf16" / "fp4_rownorm" /
    "fp8_rownorm" / "fp4_legacy"); None (default) = kernel AUTO — "bf16" below
    an effective sequence of 4096 (the MEAN SAMPLE LENGTH for packed batches),
    else "fp4_rownorm". ``bf16_grad_dots`` is the DEPRECATED boolean alias
    (True -> "bf16", False -> "fp4_legacy"); passing both raises.

    ``packed_min_sample_len``: packed-batch (multipack) perf gate, identical to
    the Qwen3.5 patch — packs whose MEAN sample length is below this keep the
    model's original (FA2-varlen) forward; 0 disables the gate. NOTE: at
    Qwen3-VL's head_dim 128 the FP4 varlen fwd+bwd measured ~8-27% SLOWER than
    FA2-varlen even at long mean sample lengths (2k-8k; fwd-only is at parity
    at 2k) — unlike Qwen3.5's head_dim 256, where FP4 wins. The packed FP4 path
    here is for numerical-consistency / experimentation (this whole patch is
    already opt-in via allow_full_model); raise the gate to keep FA2 for speed.
    ``None`` resolves to the VL-specific default (1024 — see
    ``_VL_PACKED_MIN_SAMPLE_LEN_DEFAULT``), unlike the text patch whose auto
    default is 0.
    """
    grad_dots = _resolve_grad_dots_alias(grad_dots, bf16_grad_dots)
    if packed_min_sample_len is None:
        packed_min_sample_len = _VL_PACKED_MIN_SAMPLE_LEN_DEFAULT
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextAttention

    patched = 0
    seen_forward = None
    for module in model.modules():
        if isinstance(module, Qwen3VLTextAttention):
            kw = dict(
                train_backward=train_backward,
                backward_rtn_grad_packs=backward_rtn_grad_packs,
                save_backward_packs=save_backward_packs,
                dkdv_scratch_bf16=dkdv_scratch_bf16,
                grad_dots=grad_dots,
                compile_custom_op=compile_custom_op,
                stochastic_rounding=stochastic_rounding,
                packed_min_sample_len=packed_min_sample_len,
            )
            if getattr(module, "_nvfp4_patched", False):
                _set_attrs(module, **kw)
                continue
            orig = type(module).forward
            if seen_forward is None:
                seen_forward = make_nvfp4_vl_forward(orig)
            module.forward = seen_forward.__get__(module, type(module))
            module._nvfp4_patched = True
            _set_attrs(module, **kw)
            patched += 1
    LOG.info(
        "nvfp4 attention: patched %d Qwen3-VL full-attention layers "
        "(train_backward=%s, compile_custom_op=%s)",
        patched,
        train_backward,
        compile_custom_op,
    )
    return patched
