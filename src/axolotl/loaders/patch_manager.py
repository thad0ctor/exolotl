"""Patch manager class implementation to complement `axolotl.loaders.ModelLoader`.

Applies pre- and post-model load patches for various fixes and optimizations.
"""

from __future__ import annotations

import importlib.util
import os
import re
from functools import cached_property
from typing import TYPE_CHECKING

import addict
import torch
import transformers
from transformers import (
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from transformers.modeling_flash_attention_utils import is_flash_attn_available

from axolotl.loaders.utils import is_protrain_active
from axolotl.model_support import (
    ModelHookContext,
    ModelHookPhase,
    Unsupported,
    check_capability,
    get_model_support,
    get_model_support_for_cfg,
    resolve_model_support,
    run_model_support_hooks,
)
from axolotl.monkeypatch.multipack import (
    SUPPORTED_MULTIPACK_MODEL_TYPES,
    patch_for_multipack,
)
from axolotl.utils.dict import DictDefault
from axolotl.utils.logging import get_logger

if TYPE_CHECKING:
    from transformers import ProcessorMixin

LOG = get_logger(__name__)


def rl_trainer_builds_adapter(cfg) -> bool:
    """DPO, IPO and KTO hand TRL the LoRA config and TRL builds the PeftModel in the trainer."""
    from axolotl.utils.schemas.enums import RLType

    return bool(cfg.adapter) and cfg.rl in (RLType.DPO, RLType.IPO, RLType.KTO)


class PatchManager:
    """Manages the application of patches during the model loading process."""

    @staticmethod
    def apply_pre_config_load_patches(cfg: DictDefault):
        """
        Apply patches that must be set up before config loading.
        This is for patches that intercept remote code loading from HuggingFace,
        which needs to be in place before AutoConfig.from_pretrained() is called.

        Args:
            cfg: Configuration dictionary with model and training settings.
        """
        support = get_model_support_for_cfg(cfg)
        run_model_support_hooks(
            support,
            ModelHookPhase.BEFORE_CONFIG_LOAD,
            ModelHookContext(cfg=cfg, inference=cfg.inference),
        )

    @staticmethod
    def apply_pre_tokenizer_load_patches(cfg: DictDefault):
        """
        Apply patches that must be set up before tokenizer loading.
        This is for patches that intercept remote code loading from HuggingFace,
        which needs to be in place before AutoTokenizer.from_pretrained() is called.

        Args:
            cfg: Configuration dictionary with model and training settings.
        """
        support = get_model_support_for_cfg(cfg)
        run_model_support_hooks(
            support,
            ModelHookPhase.BEFORE_TOKENIZER_LOAD,
            ModelHookContext(cfg=cfg, inference=cfg.inference),
        )

    def __init__(
        self,
        cfg: DictDefault,
        model_config: PretrainedConfig | addict.Dict,
        inference: bool = False,
        tokenizer: PreTrainedTokenizerBase | None = None,
        processor: ProcessorMixin | None = None,
        reference_model: bool = False,
        train_dataset=None,
    ):
        """Initialize the `PatchManager`.

        Args:
            cfg: Configuration dictionary with model and training settings.
            model_config: Configuration object for the model.
            inference: Whether the model is being loaded for inference mode.
            train_dataset: Optional prepared (tokenized) training dataset. In the
                standard train path datasets are prepared before the model loads;
                threading the dataset here lets the NVFP4 residual calibration
                (`_nvfp4_calib_input_ids`) forward a real sample at swap time.
        """
        self.cfg = cfg
        self.model_config = model_config
        self.inference = inference
        self.tokenizer = tokenizer
        self.processor = processor
        self.reference_model = reference_model
        self.train_dataset = train_dataset

    def _hook_context(self, model: PreTrainedModel | None = None) -> ModelHookContext:
        return ModelHookContext(
            cfg=self.cfg,
            model_config=self.model_config,
            tokenizer=self.tokenizer,
            processor=self.processor,
            model=model,
            inference=self.inference,
            reference_model=self.reference_model,
        )

    @cached_property
    def has_flash_attn(self) -> bool:
        """Check if flash attention is installed."""
        return importlib.util.find_spec("flash_attn") is not None

    def apply_pre_model_load_patches(self):
        """Apply pre-model load patches based on config."""
        self._deactivate_hf_async_load()
        self._apply_parallel_weight_conversion()
        self._apply_torchao_patches()
        self._apply_transformers_patches()
        # self._apply_flex_attention_patches()
        self._apply_flash_attention_patches()
        self._apply_chunked_cross_entropy_patch()
        self._apply_sageattn_patches()
        self._apply_fsdp_patches()
        self._apply_adapter_patches()
        # Must precede fused-RoPE patches: re-parses ``Attention.forward``
        # via ``inspect.getsource``; the QKV regex misses on a patched body.
        self._apply_self_attention_lora_patch()
        self._apply_model_support_registrations()
        self._apply_model_support_pre_load_hook()
        self._apply_model_specific_patches()
        self._apply_fp8_patches()
        self._apply_flash_attention_peft_patches()
        self._apply_gradient_checkpointing_patches()
        self._patch_attention()
        self._apply_multipack_patches()
        self._apply_sdpa_varlen_patch()
        self._apply_large_head_attention_patch()
        self._patch_loss_llama()
        self._patch_llama_derived_model()
        self._apply_mistral_cross_entropy_patch()
        self._apply_fsdp2_bnb_patches()
        self._apply_patch_deepspeed_zero3()
        self._apply_voxtral_patches()
        self._apply_apertus_patches()
        self._apply_trl_vllm_patches()

    def apply_post_plugin_pre_model_load_patches(self):
        """Apply post plugin-pre_model_load load patches based on config."""
        self._apply_tiled_mlp(self.cfg.model_config_type)
        self._apply_moe_expert_quantization_patch()

    @staticmethod
    def _apply_torchao_patches():
        from axolotl.monkeypatch.torchao_optim import patch_torchao_optim_state_8bit

        patch_torchao_optim_state_8bit()

    def _apply_transformers_patches(self):
        from axolotl.monkeypatch.transformers.trainer_loss_calc import (
            patch_evaluation_loop,
            patch_maybe_log_save_evaluate,
        )
        from axolotl.monkeypatch.transformers_num_parameters import (
            patch_num_parameters,
        )

        patch_evaluation_loop()
        patch_maybe_log_save_evaluate()
        patch_num_parameters()

    def apply_post_model_build_patches(self, model: PreTrainedModel):
        """Apply patches right after model build, before post-load setup."""
        support = get_model_support(self.cfg.model_config_type)
        run_model_support_hooks(
            support,
            ModelHookPhase.AFTER_BASE_MODEL_BUILD,
            self._hook_context(model),
        )

        if self.cfg.model_config_type == "nemotron_h":
            # Runs after model build so a re-registration during checkpoint
            # construction cannot clobber the fix.
            self._fix_nemotron_h_conversion_mapping()

        # Gemma 4 hybrid attention runs here in post-build (NOT post-load):
        # the per-layer ``self_attn.config._attn_implementation="sdpa"``
        # override needs to walk the raw model tree, which is broken by
        # the post-load PEFT wrapping. The accompanying
        # ``patch_gemma4_hybrid_mask`` monkey-patch is module-level and
        # installation-time-independent, so both halves of the fix live
        # cleanly in the same call even though one is instance-scoped
        # and the other is module-scoped.
        self._apply_gemma_hybrid_attention(model)
        self._apply_gemma4_loss_kwargs()
        self._finalize_moe_expert_quantization(model)

    def _apply_model_support_registrations(self):
        support = get_model_support(self.cfg.model_config_type)
        if support is None:
            return
        registrations = resolve_model_support(support).registrations

        conversions_provider = registrations.weight_conversions
        conversions = (
            conversions_provider() if conversions_provider is not None else None
        )
        if conversions:
            from axolotl.utils.weight_conversions import register_weight_conversions

            for key, entries in conversions.items():
                entries = list(entries)
                register_weight_conversions(key, entries)
                self._warn_irreversible_weight_transforms(key, entries)

        patch_provider = registrations.patch_mappings
        patch_mapping = patch_provider() if patch_provider is not None else None
        if patch_mapping:
            from transformers.monkey_patching import register_patch_mapping

            register_patch_mapping(dict(patch_mapping), overwrite=True)

    @staticmethod
    def _warn_irreversible_weight_transforms(key: str, transforms: list) -> None:
        """``save_pretrained(save_original_format=True)`` reverses registered
        conversions; surface irreversible entries at registration instead of at
        save time."""
        for transform in transforms:
            problems = []
            if getattr(transform, "quantization_operation", None) is not None:
                problems.append("a quantization operation")
            for operation in getattr(transform, "operations", None) or ():
                try:
                    _ = operation.reverse_op
                except Exception:  # pylint: disable=broad-exception-caught
                    problems.append(f"no reverse for {type(operation).__name__}")
            if problems:
                LOG.warning(
                    "Weight conversion registered for %s cannot be reversed at save "
                    "time (%s); saving will fail or emit the converted (non-original) "
                    "checkpoint layout.",
                    key,
                    "; ".join(problems),
                )

    def _apply_model_support_pre_load_hook(self):
        support = get_model_support(self.cfg.model_config_type)
        if self.cfg.fused_attn_kernel:
            check_capability(support, "fused_attn_kernel", self.cfg.model_config_type)
        run_model_support_hooks(
            support,
            ModelHookPhase.BEFORE_MODEL_BUILD,
            self._hook_context(),
        )

    def apply_post_model_load_patches(self, model: PreTrainedModel):
        """Apply patches that require the model instance."""
        support = get_model_support(self.cfg.model_config_type)
        run_model_support_hooks(
            support,
            ModelHookPhase.AFTER_ADAPTER_LOAD,
            self._hook_context(model),
        )
        self._apply_llama_flash_attn_patches(model)
        self._apply_lora_kernel_patch(model)
        self._apply_scaling_softmax_patch(model)
        self._apply_fp8_attention_patches(model)
        self._apply_fp4_attention_qat(model)
        self._apply_nvfp4_training(model)
        self._apply_qwen3_5_native_nvfp4_patches(model)
        self._apply_tiled_mlp_post_load(model)
        self._apply_mrope_packing_patch(model)
        self._mark_nvfp4_ddp_ignore(model)

    def _apply_mrope_packing_patch(self, model: PreTrainedModel):
        # mRoPE text models only isolate packed samples given [4, B, L] position ids;
        # the packed collator emits (B, L), which transformers expands to 3 grids.
        if self.inference or not (self.cfg.processor_type or self.cfg.is_multimodal):
            return
        if not (self.cfg.sample_packing or self.cfg.eval_sample_packing):
            return
        from axolotl.monkeypatch.mrope_packing import patch_mrope_packing

        patch_mrope_packing(model)

    def _apply_gemma4_loss_kwargs(self):
        # Flip accepts_loss_kwargs True so the Trainer normalizes loss by
        # num_items_in_batch under grad accumulation (must run before trainer init).
        if self.cfg.model_config_type not in ("gemma4", "gemma4_unified"):
            return
        from axolotl.monkeypatch.gemma4_loss_kwargs import (
            patch_gemma4_accepts_loss_kwargs,
        )

        patch_gemma4_accepts_loss_kwargs()

    def _apply_gemma_hybrid_attention(self, model: PreTrainedModel):
        """Apply hybrid attention: FA2 for sliding window layers, SDPA for global layers.

        Gemma 4 has global (full_attention) layers with head_dim=512
        which exceeds flash attention's supported size. This patch loads the model
        with flash_attention_2 for the sliding window layers (head_dim=256), then
        gives each global layer a shallow-copied config with _attn_implementation="sdpa".

        We also install :func:`axolotl.monkeypatch.gemma4_hybrid_mask.patch_gemma4_hybrid_mask`
        which fixes the corresponding mask construction inside
        ``Gemma4TextModel.forward``. Without it, the per-layer SDPA config
        override is not enough — the forward still builds a 2D FA2-format mask
        at the model level and the SDPA layers crash at long context lengths
        with ``RuntimeError: The expanded size of the tensor ... must match``.
        """
        if not self.cfg.gemma4_hybrid_attn_impl:
            return

        import copy

        from axolotl.monkeypatch.attention.large_head import (
            resolve_large_head_policy,
            set_large_head_packed,
            set_large_head_policy,
        )
        from axolotl.monkeypatch.gemma4_hybrid_mask import (
            GLOBAL_PACKED_SDPA,
            patch_gemma4_hybrid_mask,
        )

        patch_gemma4_hybrid_mask()
        # Gemma-4 global layers reuse the generic large-head router. Default policy 'sdpa' (flash is
        # opt-in via large_head_attention / the deprecated flash_attn_d512), preserving prior default.
        set_large_head_policy(resolve_large_head_policy(self.cfg))
        set_large_head_packed(bool(self.cfg.sample_packing))

        # Navigate to the module that has 'layers' - varies by model structure:
        # Gemma4ForConditionalGeneration -> .model (Gemma4Model) -> .language_model (Gemma4TextModel) -> .layers
        # Gemma4ForCausalLM -> .model (Gemma4TextModel) -> .layers
        layers = None
        config_source = None
        for candidate in [model, getattr(model, "model", None)]:
            if candidate is None:
                continue
            # Check direct layers
            if hasattr(candidate, "layers"):
                layers = candidate.layers
                config_source = candidate
                break
            # Check language_model.layers (multimodal wrapper)
            lang_model = getattr(candidate, "language_model", None)
            if lang_model is not None and hasattr(lang_model, "layers"):
                layers = lang_model.layers
                config_source = lang_model
                break

        if layers is None:
            LOG.warning(
                "gemma4_hybrid_attn_impl: could not find decoder layers in model, skipping"
            )
            return

        config = getattr(config_source, "config", self.model_config)
        layer_types = getattr(config, "layer_types", None)
        if layer_types is None:
            LOG.warning(
                "gemma4_hybrid_attn_impl: model config has no 'layer_types', skipping. "
                "This feature requires a model with mixed sliding/global attention layers."
            )
            return

        patched_count = 0
        for layer_idx, layer in enumerate(layers):
            if layer_types[layer_idx] != "sliding_attention":
                # Global / full_attention layer (head_dim=512, FA2 can't serve it). Use the
                # packing-aware SDPA impl: it rebuilds the block-diagonal mask from position_ids so
                # the layer respects document boundaries under sample packing (plain "sdpa" gets a
                # None mask here and would attend across packed documents).
                attn_module = getattr(layer, "self_attn", None)
                if attn_module is not None and hasattr(attn_module, "config"):
                    sdpa_config = copy.copy(attn_module.config)
                    sdpa_config._attn_implementation = GLOBAL_PACKED_SDPA
                    attn_module.config = sdpa_config
                    patched_count += 1

        LOG.info(
            "gemma4_hybrid_attn_impl: patched %d global layers to use packing-aware SDPA "
            "(remaining %d sliding layers use flash_attention_2)",
            patched_count,
            len(layers) - patched_count,
        )

    def _apply_flash_attention_patches(self):
        """Apply patches related to Flash Attention."""
        if self.cfg.attn_implementation == "xformers":
            from axolotl.monkeypatch.attention import register_xformers_attn

            register_xformers_attn()

            if self.cfg.sample_packing:
                # Also patch FA2 slot for legacy code paths that use it directly
                from axolotl.monkeypatch.attention import patch_xformers_attn_over_fa2

                patch_xformers_attn_over_fa2()

        if self.cfg.attn_implementation == "sage":
            from axolotl.monkeypatch.attention import register_sage_attn

            register_sage_attn()

        if self.cfg.attn_implementation == "sage_fp4":
            from axolotl.monkeypatch.attention import register_sage_fp4_attn
            from axolotl.monkeypatch.attention.sage_fp4_attn import (
                configure_sage_fp4,
                patch_sage_fp4_attn,
            )

            if not self.inference:
                raise ValueError(
                    "attn_implementation: sage_fp4 is inference-only and cannot "
                    "be used by axolotl train. For Qwen3.5 native FP4 attention "
                    "training, keep a training-capable attention backend such as "
                    "flash_attention_2 and enable nvfp4_training with "
                    "attention.enabled plus attention.backward.enabled."
                )

            patch_sage_fp4_attn()
            sage_cfg = self.cfg.sage_fp4_attention
            configure_sage_fp4(
                per_block_mean=sage_cfg.per_block_mean if sage_cfg else True,
                allow_fallback=(sage_cfg.allow_fallback if sage_cfg else True),
            )
            register_sage_fp4_attn()

    def _apply_fp8_attention_patches(self, model):
        """Apply FP8 low-precision attention via torchao."""
        if self.cfg.attn_implementation == "fp8":
            from axolotl.monkeypatch.attention.fp8_attn import patch_fp8_attention

            patch_fp8_attention(model)

    def _apply_fp4_attention_qat(self, model: PreTrainedModel):
        """Route attention through the NVFP4 fake-quant QAT path (opt-in).

        Independent of nvfp4_training: works on a bf16 base. Routes through the
        fused FlashAttention-style Triton kernel (linear memory) when applicable
        and falls back to the eager materialized-P path otherwise, so it overrides
        any flash/sdpa request — warn if the user asked for one. Post-load so it
        sees the final model tree.
        """
        if not self.cfg.fp4_attention_qat:
            return

        from axolotl.utils.attn_qat import apply_nvfp4_qat_attention

        requested = self.cfg.attn_implementation
        if requested and requested not in ("eager", "nvfp4_qat"):
            LOG.warning(
                "fp4_attention_qat overrides attn_implementation=%s with the "
                "NVFP4 fake-quant attention (fused flash kernel, eager fallback).",
                requested,
            )
        apply_nvfp4_qat_attention(model)

    def _apply_nvfp4_training(self, model: PreTrainedModel):
        """Swap eligible linears for NVFP4-GEMM linears (Blackwell FP4 compute).

        Runs in post-load (after weights + any merge AND after PEFT wraps the
        model in ``ModelLoader._load_adapters``) so the swap sees real linear
        modules in their final tree position. FFT swaps raw ``nn.Linear``;
        adapter modes swap the FROZEN base_layer inside each ``lora.Linear``,
        which only exists once PEFT has wrapped the model — hence post-load.
        """
        nvfp4 = self.cfg.nvfp4_training
        if not (nvfp4 and nvfp4.enabled):
            return

        # In-process merge (legacy merge-lora) writes base_layer.weight.data +=
        # delta; the FP4 base modules expose weight read-only, so that write
        # would silently no-op. Keep the base in bf16 for merge — merge_and_unload
        # then merges into the real weight. (FP4 training is irrelevant at merge.)
        if self.cfg.merge_lora:
            return

        from axolotl.kernels.lora import set_nvfp4_shared_base_fprop
        from axolotl.utils.nvfp4_training import (
            NVFP4Recipe,
            convert_lora_base_to_nvfp4,
            convert_to_nvfp4_training,
        )

        recipe = NVFP4Recipe(
            stochastic_rounding=nvfp4.stochastic_rounding,
            hadamard=nvfp4.hadamard,
        )
        shared_lora_base_fprop = set_nvfp4_shared_base_fprop(
            getattr(nvfp4, "shared_lora_base_fprop", None)
        )
        exclude_modules = list(nvfp4.exclude_modules or [])
        if getattr(nvfp4, "quantize_lm_head", False):
            exclude_modules = self._nvfp4_unexclude_lm_head(model, exclude_modules)
        exclude = tuple(exclude_modules) + self._nvfp4_block_exclusions(
            model, nvfp4.skip_first_n_blocks or 0, nvfp4.skip_last_n_blocks or 0
        )

        adapter = self.cfg.adapter
        if shared_lora_base_fprop:
            LOG.info("NVFP4 LoRA shared base fprop enabled")

        # Transformer Engine backend: swap eligible linears to te.Linear; the
        # trainer wraps the step in te.fp8_autocast (set up below via a stored
        # recipe). FFT swaps raw nn.Linear; LoRA swaps the frozen base_layer
        # inside each lora.Linear (adapters stay high-precision).
        if getattr(nvfp4, "backend", "native") == "te":
            from axolotl.utils.nvfp4_training import (
                convert_lora_base_to_te_nvfp4,
                convert_to_te_nvfp4_training,
                te_nvfp4_available,
                te_nvfp4_recipe,
            )

            ok, reason = te_nvfp4_available()
            if not ok:
                raise RuntimeError(reason)
            # te keeps an HP (bf16) base and quantizes on the fly: it has no
            # FP4-storage path, so a requested storage/compute base_mode would
            # silently give no memory saving. Steer the user to backend: native.
            if adapter in ("lora", "qlora") and (
                nvfp4.quantize_base
                or getattr(nvfp4, "base_mode", None) in ("storage", "compute")
            ):
                LOG.warning(
                    "nvfp4_training.backend: te ignores base_mode/quantize_base "
                    "(te keeps a high-precision base, no FP4 storage saving). Use "
                    "backend: native for FP4-stored/compute LoRA bases."
                )
            if adapter in ("lora", "qlora"):
                count = convert_lora_base_to_te_nvfp4(model, recipe, exclude=exclude)
                empty_msg = (
                    "nvfp4_training(te) enabled but no eligible LoRA base layers "
                    "were swapped (is the model PEFT-wrapped?)"
                )
            elif adapter:
                raise ValueError(
                    f"nvfp4_training.backend: te supports full fine-tune or "
                    f"lora/qlora, not adapter={adapter}."
                )
            else:
                count = convert_to_te_nvfp4_training(model, recipe, exclude=exclude)
                empty_msg = "nvfp4_training(te) enabled but no nn.Linear swapped"
            model._te_nvfp4_recipe = te_nvfp4_recipe(recipe)
            if count == 0:
                LOG.warning(empty_msg)
            return

        if adapter in ("lora", "qlora"):
            # Resolve the base mode. Explicit base_mode wins; otherwise
            # qlora/quantize_base => FP4 storage, else default to FP4 compute
            # (pre-quantized base, fastest, the recommended LoRA path).
            base_mode = getattr(nvfp4, "base_mode", None)
            if base_mode is None:
                base_mode = (
                    "storage"
                    if (bool(nvfp4.quantize_base) or adapter == "qlora")
                    else "compute"
                )
            compute_base = base_mode == "compute"
            quantized_storage = base_mode == "storage"
            # The MSLK fast path wraps quant in registered custom ops; the torchao
            # fallback is pure-torch quant (nvfp4_quantize / _sr_dither) + an aten
            # _scaled_mm GEMM, which is also compile-safe (verified: a compiled
            # compute-base step with MSLK forced off completes with no graph breaks
            # beyond the model's own data-dependent ones). MSLK is just faster, so
            # surface it as info, not a warning.
            if compute_base and self.cfg.torch_compile:
                from axolotl.utils.nvfp4_training import _mslk_available

                if not _mslk_available():
                    LOG.info(
                        "nvfp4_training compute-base under torch_compile is using the "
                        "torchao fallback (MSLK not installed); this is compile-safe "
                        "but slower. Install MSLK for the faster custom-op quant path."
                    )
            # Both FP4 base modes need the NVFP4 all-gather hooks to shard under
            # FSDP2 (storage: one layout; compute: fprop+dgrad layouts).
            use_fsdp = (quantized_storage or compute_base) and bool(
                self.cfg.fsdp_config
            )
            # Frozen-base L2QER residual (default ON): the activation calibration
            # MUST run before the swap — the storage mode discards the bf16
            # master the quant error is computed from.
            base_residual_plan = self._nvfp4_prepare_base_residual(
                model, exclude, base_mode
            )
            count = convert_lora_base_to_nvfp4(
                model,
                recipe,
                quantized_storage=quantized_storage,
                compute_base=compute_base,
                fsdp=use_fsdp,
                exclude=exclude,
                base_residual=base_residual_plan,
            )
            empty_msg = (
                "nvfp4_training enabled but no eligible LoRA base layers were "
                "swapped (is the model PEFT-wrapped?)"
            )
        else:
            base_mode = getattr(nvfp4, "base_mode", None) or "compute"
            # base_residual corrects FROZEN FP4 stores; full fine-tune weights are
            # trainable (re-quantized every step), so a load-time residual would
            # go stale immediately — leave full-FT untouched.
            LOG.debug(
                "nvfp4 base_residual: full fine-tune keeps trainable bf16 masters "
                "(no frozen FP4 base store); base_residual does not apply."
            )
            count = convert_to_nvfp4_training(model, recipe, exclude=exclude)
            empty_msg = (
                "nvfp4_training enabled but no eligible nn.Linear layers were swapped"
            )
        if count == 0:
            LOG.warning(empty_msg)

        # lm_head / input-embedding / tied-shared-weight swaps (each opt-in). The
        # LoRA base converter only touches lora.Linear base_layers, so a frozen
        # lm_head/embedding that isn't a LoRA target is invisible to it and is
        # handled here.
        self._nvfp4_apply_tied_or_lm_head(model, recipe, base_mode)

        # Vision-tower encoder linears (multimodal, opt-in): frozen nn.Linear
        # under the vision module stay bf16 otherwise (not lora.Linear).
        if getattr(nvfp4, "quantize_vision_tower", False):
            from axolotl.utils.nvfp4_training import convert_vision_tower_to_nvfp4

            convert_vision_tower_to_nvfp4(model, recipe, base_mode=base_mode)

        # Fuse decoder RMSNorm + activation quant into one kernel so the base
        # linear reuses the norm's pre-quantized activation (native backend only;
        # the fused norm emits single-level FP4, matching the compute-base path).
        if getattr(nvfp4, "fuse_rmsnorm", True):
            from axolotl.utils.nvfp4_training import convert_norms_to_nvfp4_fused

            convert_norms_to_nvfp4_fused(model)

        if getattr(nvfp4, "fuse_qk_norm_rope", False):
            from axolotl.monkeypatch.models.qwen_fused_attn import patch_qwen_fused_attn

            patch_qwen_fused_attn(self.cfg.model_config_type)

        self._nvfp4_load_packed_sidecar(model)

        # Fused FP4 lm_head + cross-entropy: skip materializing the [M, vocab]
        # logits (memory win). Opt-in and only when the lm_head became an FP4
        # store above; falls back to the materialized CE path otherwise.
        #
        # When lm_head_distillation is enabled, its own forward patch already runs
        # the fused FP4 CE (then adds the KL term) and owns the head computation —
        # installing the plain fused-CE wrapper on top would shadow the KL term, so
        # skip it (the distillation forward calls fused_fp4_cross_entropy directly).
        _distill = getattr(nvfp4, "lm_head_distillation", None)
        _distill_on = _distill is not None and getattr(_distill, "enabled", False)
        if (
            getattr(nvfp4, "fused_fp4_cross_entropy", False)
            and getattr(nvfp4, "quantize_lm_head", False)
            and not _distill_on
        ):
            from axolotl.kernels.nvfp4_fused_ce import patch_model_fused_fp4_ce

            patch_model_fused_fp4_ce(
                model,
                fp4_matmul=True
                if getattr(nvfp4, "fused_fp4_cross_entropy_fp4_matmul", False)
                else None,
                vocab_block=getattr(nvfp4, "fused_ce_vocab_block", None),
            )

        if getattr(nvfp4, "fp8_lm_head", False):
            from axolotl.kernels.fp8_lm_head import patch_model_fp8_lm_head

            patch_model_fp8_lm_head(
                model,
                granularity=getattr(nvfp4, "fp8_lm_head_granularity", "rowwise"),
            )

        if getattr(nvfp4, "fp8_lm_head_cross_entropy", False):
            from axolotl.kernels.fp8_fused_ce import (
                patch_model_fp8_lm_head_cross_entropy,
            )

            patch_model_fp8_lm_head_cross_entropy(
                model,
                granularity=getattr(nvfp4, "fp8_lm_head_granularity", "rowwise"),
            )

        if getattr(nvfp4, "bf16_lm_head_cross_entropy", False):
            from axolotl.kernels.bf16_fused_ce import (
                patch_model_bf16_lm_head_cross_entropy,
            )

            patch_model_bf16_lm_head_cross_entropy(model)

    def _mark_nvfp4_ddp_ignore(self, model: PreTrainedModel):
        """Exclude NVFP4 frozen-base buffers from DDP's param/buffer sync.

        DDP NCCL-broadcasts module states across ranks (at init and, with
        broadcast_buffers, every step), but NCCL has no support for the packed
        ``Float4_e2m1fn_x2`` / fp8-scale dtypes the NVFP4 base stores — it raises
        "Input tensor data type is not supported for NCCL process group". Those
        buffers are frozen and bit-identical on every rank (deterministic quant),
        so they never need syncing; naming them in
        ``_ddp_params_and_buffers_to_ignore`` (read natively by DDP) skips them.
        """
        nvfp4 = self.cfg.nvfp4_training
        if not (nvfp4 and nvfp4.enabled):
            return

        exotic = {torch.float8_e4m3fn, torch.float8_e5m2}
        fp4 = getattr(torch, "float4_e2m1fn_x2", None)
        if fp4 is not None:
            exotic.add(fp4)

        ignore = [
            name
            for name, buf in model.named_buffers()
            if buf is not None
            and (type(buf).__name__ == "NVFP4Tensor" or buf.dtype in exotic)
        ]
        if not ignore:
            return

        existing = list(getattr(model, "_ddp_params_and_buffers_to_ignore", []))
        model._ddp_params_and_buffers_to_ignore = list(dict.fromkeys(existing + ignore))
        LOG.info("NVFP4: excluded %d FP4 base buffers from DDP sync", len(ignore))

    def _apply_qwen3_5_native_nvfp4_patches(self, model: PreTrainedModel):
        nvfp4 = self.cfg.nvfp4_training
        if not (nvfp4 and nvfp4.enabled):
            return
        attn = nvfp4.attention

        # Qwen3-VL: standard (non-gated) attention + multimodal mRoPE -> dedicated
        # VL patch (the Qwen3.5 patch assumes gated q + plain RoPE). Only the
        # attention path applies; VL has no DeltaNet linear_attn / Qwen3.5 MLP.
        if self.cfg.model_config_type == "qwen3_vl":
            if attn.enabled and not attn.allow_full_model:
                # Qwen3-VL is NON-hybrid: every LM layer is full softmax attention.
                # FP4 attention is ~0.97 cosine per layer, which compounds over all
                # ~36 layers (~0.97^36 -> ~0.3) and wrecks the forward (step-1 loss
                # ~5.3 vs ~1.4 without it). Refuse rather than silently train garbage.
                LOG.warning(
                    "nvfp4_training.attention.enabled is set on a non-hybrid Qwen3-VL "
                    "model, where FP4 attention would touch EVERY layer; the per-layer "
                    "FP4 error (~0.97 cos) compounds over all layers and destroys the "
                    "forward (step-1 loss ~5.3 vs ~1.4). Disabling native FP4 attention "
                    "(FP4 base weights/lm_head still apply). Set "
                    "nvfp4_training.attention.allow_full_model: true to override."
                )
                return
            if attn.enabled:
                LOG.warning(
                    "nvfp4_training.attention.allow_full_model is set: enabling native "
                    "FP4 attention on ALL Qwen3-VL layers. Expect a large forward "
                    "degradation (step-1 loss ~5.3 vs ~1.4) from compounded FP4 error."
                )
                from axolotl.monkeypatch.attention.nvfp4_flash_attn_vl import (
                    patch_qwen3_vl_nvfp4_attention,
                )

                patch_qwen3_vl_nvfp4_attention(
                    model,
                    train_backward=attn.backward.enabled,
                    backward_rtn_grad_packs=attn.backward.rtn_grad_packs,
                    save_backward_packs=attn.backward.save_packs,
                    dkdv_scratch_bf16=attn.backward.dkdv_scratch_bf16,
                    grad_dots=attn.backward.grad_dots,
                    compile_custom_op=bool(attn.backward.compile_custom_op),
                    stochastic_rounding=nvfp4.stochastic_rounding,
                    packed_min_sample_len=attn.packed_min_sample_len,
                )
            return

        if self.cfg.model_config_type not in ("qwen3_5", "qwen3_5_moe"):
            return

        if attn.enabled:
            from axolotl.monkeypatch.attention.nvfp4_flash_attn import (
                patch_qwen3_5_nvfp4_attention,
            )

            patch_qwen3_5_nvfp4_attention(
                model,
                fuse_vproj=(
                    self.inference if attn.fuse_vproj is None else attn.fuse_vproj
                ),
                fuse_attn_proj=attn.fp4_projections,
                train_backward=attn.backward.enabled,
                backward_rtn_grad_packs=attn.backward.rtn_grad_packs,
                save_backward_packs=attn.backward.save_packs,
                dkdv_scratch_bf16=attn.backward.dkdv_scratch_bf16,
                grad_dots=attn.backward.grad_dots,
                compile_custom_op=bool(attn.backward.compile_custom_op),
                stochastic_rounding=nvfp4.stochastic_rounding,
                packed_min_sample_len=attn.packed_min_sample_len,
            )

        if nvfp4.linear_attn:
            from axolotl.monkeypatch.attention.nvfp4_linear_attn import (
                patch_qwen3_5_nvfp4_linear_attn,
            )

            patch_qwen3_5_nvfp4_linear_attn(model)

        if nvfp4.mlp:
            from axolotl.monkeypatch.attention.nvfp4_mlp import (
                patch_qwen3_5_nvfp4_mlp,
            )

            patch_qwen3_5_nvfp4_mlp(model)

    def _nvfp4_load_packed_sidecar(self, model: PreTrainedModel):
        """Restore FP4-packed weights from a save_nvfp4 sidecar, if one exists.

        Looks in resume_from_checkpoint first (resume), then the base model dir
        (loading a save_nvfp4-exported model). No-op when no sidecar is present —
        the frozen base otherwise reconstructs deterministically from the bf16
        weights, so this only matters for save_nvfp4 exports / exact FP4 reload.
        """
        import os

        from axolotl.utils.nvfp4_training import (
            NVFP4_PACKED_SIDECAR,
            load_nvfp4_packed,
        )

        candidates = [self.cfg.resume_from_checkpoint, self.cfg.base_model]
        for cand in candidates:
            if not cand or not isinstance(cand, str):
                continue
            if os.path.isfile(os.path.join(cand, NVFP4_PACKED_SIDECAR)):
                load_nvfp4_packed(model, cand)
                return

    @staticmethod
    def _nvfp4_block_exclusions(
        model: PreTrainedModel, skip_first: int, skip_last: int
    ) -> tuple[str, ...]:
        """Translate skip_first/last_n_blocks into ``layers.<i>.`` name fragments.

        Block count is only known here (the model is built), so the block-range
        policy is resolved in the integration layer and passed to the swap as
        explicit ``exclude`` fragments.
        """
        if skip_first <= 0 and skip_last <= 0:
            return ()

        block_re = re.compile(r"(.*\blayers)\.(\d+)\.")
        prefixes: dict[str, set[int]] = {}
        for name, _ in model.named_modules():
            m = block_re.match(name)
            if m:
                prefixes.setdefault(m.group(1), set()).add(int(m.group(2)))

        fragments: list[str] = []
        for prefix, indices in prefixes.items():
            ordered = sorted(indices)
            skip = set(ordered[:skip_first])
            if skip_last > 0:
                skip |= set(ordered[len(ordered) - skip_last :])
            fragments.extend(f"{prefix}.{i}." for i in sorted(skip))
        return tuple(fragments)

    def _nvfp4_unexclude_lm_head(
        self, model: PreTrainedModel, exclude_modules: list[str]
    ) -> list[str]:
        """Drop ``lm_head`` from the NVFP4 exclusion so the converter swaps it.

        Tied embeddings are handled by the quantize-once path (see
        ``_nvfp4_apply_tied_or_lm_head``), so a FROZEN tied weight is allowed
        here; a TRAINABLE tied weight still raises (FP4-storing it would corrupt
        training). A fused-linear cross-entropy path (cut_cross_entropy) consumes
        the lm_head weight directly and bypasses the NVFP4 forward — raise. If the
        lm_head dims aren't FP4-swappable (%32), leave it excluded with a warning
        rather than crash. Only ``lm_head`` is removed; ``embed_tokens`` stays.
        """
        from axolotl.utils.nvfp4_training import _is_swappable

        if self._model_ties_embeddings(model):
            if self._tied_weight_trainable(model):
                raise RuntimeError(
                    "nvfp4_training.quantize_lm_head with tied embeddings requires a "
                    "FROZEN shared weight: the output and input embeddings share one "
                    "weight, and FP4-storing a TRAINABLE shared weight would corrupt "
                    "training. Freeze the embedding (e.g. use LoRA, which freezes the "
                    "base), or set quantize_lm_head: false."
                )
            # Frozen tied: the shared weight is quantized once and routed to both
            # roles in _nvfp4_apply_tied_or_lm_head; the name-fragment exclusion is
            # irrelevant for the tied path, so return unchanged.
            return exclude_modules

        nvfp4_cfg = self.cfg.nvfp4_training
        want_fused_ce = bool(getattr(nvfp4_cfg, "fused_fp4_cross_entropy", False))
        if self.cfg.cut_cross_entropy and not want_fused_ce:
            raise RuntimeError(
                "nvfp4_training.quantize_lm_head is incompatible with "
                "cut_cross_entropy: the fused linear cross-entropy kernel reads the "
                "lm_head weight directly to fuse the projection with the loss, which "
                "bypasses the NVFP4 lm_head forward (the FP4 head would be ignored, "
                "or the kernel would fail on the NVFP4 module's missing .weight). "
                "Disable one of them (cut_cross_entropy: false or "
                "quantize_lm_head: false), or set "
                "nvfp4_training.fused_fp4_cross_entropy: true to use the FP4-aware "
                "fused cross-entropy (reads the NVFP4-packed lm_head directly)."
            )

        out_emb = model.get_output_embeddings()
        if not isinstance(out_emb, torch.nn.Linear) or not _is_swappable(out_emb):
            in_f = getattr(out_emb, "in_features", "?")
            out_f = getattr(out_emb, "out_features", "?")
            LOG.warning(
                "nvfp4_training.quantize_lm_head: lm_head is not NVFP4-swappable "
                "(in=%s out=%s, both must be divisible by 32); keeping it in high "
                "precision.",
                in_f,
                out_f,
            )
            return exclude_modules

        without_lm_head = [m for m in exclude_modules if m != "lm_head"]
        if "lm_head" in exclude_modules:
            LOG.info(
                "nvfp4_training.quantize_lm_head: removing lm_head from the "
                "high-precision exclusion (it will be quantized to NVFP4)."
            )
        return without_lm_head

    @staticmethod
    def _nvfp4_swap_frozen_lm_head(
        model, recipe, base_mode: str, retain_teacher: bool = False
    ) -> None:
        """Swap a bare frozen lm_head (LoRA, not a target module) to NVFP4.

        Locates the output-embedding module by identity in the (possibly
        PEFT-wrapped) tree. If it's already an NVFP4 module (e.g. the user added
        lm_head to lora_target_modules and the LoRA converter handled it), this
        is a no-op.
        """
        import torch.nn as nn

        from axolotl.utils.nvfp4_training import swap_frozen_linear_to_nvfp4

        out_emb = model.get_output_embeddings()
        if not isinstance(out_emb, nn.Linear):
            return  # already swapped (NVFP4 module) or wrapped — nothing bare to do
        name = next(
            (n for n, m in model.named_modules() if m is out_emb),
            None,
        )
        if name is None:
            LOG.warning(
                "nvfp4_training.quantize_lm_head: could not locate the lm_head "
                "module in the model tree; leaving it in high precision."
            )
            return
        swap_frozen_linear_to_nvfp4(
            model, name, recipe, base_mode=base_mode, retain_teacher=retain_teacher
        )

    @staticmethod
    def _model_ties_embeddings(model: PreTrainedModel) -> bool:
        """Detect weight tying between the output and input embeddings.

        Checks both the config flag and weight identity — a model can tie via
        config or via a shared parameter object even if the flag is stale.
        """
        if getattr(getattr(model, "config", None), "tie_word_embeddings", False):
            return True
        try:
            out_emb = model.get_output_embeddings()
            in_emb = model.get_input_embeddings()
        except (AttributeError, NotImplementedError):
            return False
        out_w = getattr(out_emb, "weight", None)
        in_w = getattr(in_emb, "weight", None)
        return out_w is not None and in_w is not None and out_w is in_w

    @staticmethod
    def _tied_weight_trainable(model: PreTrainedModel) -> bool:
        """Whether the shared (tied) embedding weight requires grad.

        FP4-storing a trainable shared weight would corrupt training, so the
        quantize-once path is gated on this being False.
        """
        try:
            in_w = getattr(model.get_input_embeddings(), "weight", None)
        except (AttributeError, NotImplementedError):
            return False
        return bool(getattr(in_w, "requires_grad", False))

    def _nvfp4_apply_tied_or_lm_head(self, model, recipe, base_mode: str) -> None:
        """Route the tied / lm_head / embedding NVFP4 swaps post linear-conversion.

        Three independent flags (all OFF by default):
        - tied + quantize_lm_head (frozen): quantize the SHARED weight once and
          point both the embedding lookup and the lm_head GEMM at it.
        - quantize_lm_head (untied): swap the bare frozen lm_head.
        - quantize_embeddings: swap the frozen input embedding (also covers the
          tied case when quantize_lm_head is off — the shared weight is stored
          FP4 for the lookup, lm_head left HP).
        """
        import torch.nn as nn

        from axolotl.utils.nvfp4_training import (
            swap_frozen_embedding_to_nvfp4,
            swap_tied_embedding_and_lm_head_to_nvfp4,
        )

        nvfp4 = self.cfg.nvfp4_training
        want_lm_head = bool(getattr(nvfp4, "quantize_lm_head", False))
        want_embed = bool(getattr(nvfp4, "quantize_embeddings", False))
        if not (want_lm_head or want_embed):
            return

        # KL-distillation AND the low-rank residual both need the original bf16
        # head retained: distillation uses it as the teacher, the residual computes
        # the quant error E = W_bf16 - dequant(Q(W)) from it. Gated to
        # quantize_lm_head (the validator enforces that upstream).
        distill_cfg = getattr(nvfp4, "lm_head_distillation", None)
        residual_cfg = getattr(nvfp4, "lm_head_residual", None)
        retain_teacher = bool(
            want_lm_head
            and (
                (distill_cfg is not None and getattr(distill_cfg, "enabled", False))
                or (
                    residual_cfg is not None and getattr(residual_cfg, "enabled", False)
                )
            )
        )

        tied = self._model_ties_embeddings(model)

        if tied and want_lm_head:
            # Frozen-tied is guaranteed here (the trainable case raised in
            # _nvfp4_unexclude_lm_head). Quantize the shared weight once.
            in_name = self._module_name(model, model.get_input_embeddings())
            out_name = self._module_name(model, model.get_output_embeddings())
            if in_name and out_name:
                swap_tied_embedding_and_lm_head_to_nvfp4(
                    model, in_name, out_name, recipe, retain_teacher=retain_teacher
                )
            self._nvfp4_attach_residual(model, residual_cfg, retain_teacher)
            self._nvfp4_attach_distillation(model, distill_cfg, retain_teacher)
            return

        if want_lm_head:
            # The fused FP4 cross-entropy needs a row-sliceable (non-swizzled)
            # lm_head store; force the torchao storage class for it. Otherwise use
            # the requested base mode (compute/storage/hp).
            if bool(getattr(nvfp4, "fused_fp4_cross_entropy", False)):
                import torch.nn as _nn

                from axolotl.utils.nvfp4_training import swap_frozen_lm_head_tileable

                out_emb = model.get_output_embeddings()
                if isinstance(out_emb, _nn.Linear):
                    name = self._module_name(model, out_emb)
                    if name:
                        swap_frozen_lm_head_tileable(
                            model, name, recipe, retain_teacher=retain_teacher
                        )
            else:
                self._nvfp4_swap_frozen_lm_head(
                    model, recipe, base_mode, retain_teacher=retain_teacher
                )
            self._nvfp4_attach_residual(model, residual_cfg, retain_teacher)
            self._nvfp4_attach_distillation(model, distill_cfg, retain_teacher)

        if want_embed:
            in_emb = model.get_input_embeddings()
            if isinstance(in_emb, nn.Embedding):
                in_name = self._module_name(model, in_emb)
                if in_name:
                    swap_frozen_embedding_to_nvfp4(model, in_name)

    def _nvfp4_attach_residual(self, model, residual_cfg, retain_teacher: bool) -> None:
        """Compute + attach the low-rank bf16 residual to the FP4 lm_head.

        No-op unless lm_head_residual is enabled and the bf16 teacher weight was
        retained (used as E = W_bf16 - dequant(Q(W))). For the activation (L2QER)
        construction, capture calibration hidden states with one short forward over
        a few real batches; plain SVD needs no calibration.
        """
        if not (
            retain_teacher
            and residual_cfg is not None
            and getattr(residual_cfg, "enabled", False)
        ):
            return
        from axolotl.kernels.nvfp4_residual import ResidualConfig, attach_residual

        cfg = ResidualConfig(
            enabled=True,
            rank=int(getattr(residual_cfg, "rank", 32)),
            calibration=str(getattr(residual_cfg, "calibration", "activation")),
            calib_tokens=int(getattr(residual_cfg, "calib_tokens", 512)),
        )
        lm_head = model.get_output_embeddings()
        teacher_w = getattr(lm_head, "_distill_teacher_w", None)
        if teacher_w is None:
            LOG.warning(
                "lm_head_residual: bf16 teacher weight not retained; cannot build "
                "the residual. Skipping."
            )
            return

        calib_hidden = None
        if cfg.calibration == "activation":
            calib_hidden = self._nvfp4_capture_calib_hidden(model, cfg.calib_tokens)

        attach_residual(lm_head, teacher_w, cfg, calib_hidden)
        LOG.info(
            "NVFP4 lm_head_residual enabled (rank=%d, calibration=%s)",
            cfg.rank,
            cfg.calibration,
        )

    def _nvfp4_prepare_base_residual(self, model, exclude, base_mode: str):
        """Pre-swap activation calibration for the frozen-base L2QER residual.

        ``nvfp4_training.base_residual`` is ON by default: whenever the LoRA/QLoRA
        swap is about to freeze base linears on the FP4 grid, capture — with
        forward pre-hooks on every base linear that will be swapped — the
        per-input-channel running mean of x^2 over ONE short forward of the
        still-bf16 model (token source shared with the lm_head calibration).
        Must run BEFORE ``convert_lora_base_to_nvfp4``: the storage mode discards
        the bf16 master the quant error is computed from.

        Returns a ``BaseResidualPlan`` for the converter, or ``None`` (disabled).
        Per the measured finding (plain SVD is useless on the near-full-rank e2m1
        error), an UNAVAILABLE activation calibration disables the residual with
        a loud warning instead of silently degrading to plain SVD.
        """
        import torch.nn as nn

        nvfp4 = self.cfg.nvfp4_training
        br = getattr(nvfp4, "base_residual", None)
        # Missing block == schema default == enabled (the explicit product
        # decision: it should just happen when frozen FP4 base linears exist).
        enabled = getattr(br, "enabled", None) if br is not None else None
        if enabled is None:
            enabled = True
        if not enabled:
            LOG.debug("nvfp4 base_residual: disabled by config")
            return None
        if base_mode not in ("storage", "compute"):
            LOG.debug(
                "nvfp4 base_residual: base_mode=%s keeps a high-precision master "
                "(no frozen FP4 store to correct); skipping.",
                base_mode,
            )
            return None
        if not isinstance(model, nn.Module):
            return None

        from peft.tuners.lora import Linear as LoraLinear

        from axolotl.utils.nvfp4_training import BaseResidualPlan, _is_swappable

        rank = int(getattr(br, "rank", None) or 16) if br is not None else 16
        calibration = (
            str(getattr(br, "calibration", None) or "activation")
            if br is not None
            else "activation"
        )
        calib_tokens = (
            int(getattr(br, "calib_tokens", None) or 512) if br is not None else 512
        )

        # The linears the converter will swap: frozen, swappable lora.Linear
        # bases, minus the exclusions. lm_head is always excluded here — it has
        # its own lm_head_residual block (and lm_head-sized rows need the
        # streaming path the head machinery already covers).
        targets = []
        for name, module in model.named_modules():
            if not isinstance(module, LoraLinear):
                continue
            if any(frag in name for frag in exclude) or "lm_head" in name:
                continue
            base = module.base_layer
            if not isinstance(base, nn.Linear) or not _is_swappable(base):
                continue
            w = base.weight
            if w is None or w.is_meta or w.requires_grad:
                continue
            targets.append((name, base))
        if not targets:
            LOG.debug("nvfp4 base_residual: no eligible frozen base linears")
            return None
        if calibration != "activation":
            # Explicit ablation request: plain SVD per module (g=None).
            LOG.warning(
                "nvfp4 base_residual: calibration='svd' is an ablation — plain "
                "SVD recovers almost none of the near-full-rank e2m1 error."
            )
            return BaseResidualPlan(rank=rank, calibration=calibration, calib={})

        acc: dict = {}

        def _make_hook(name):
            def hook(_m, args):
                x = args[0] if args else None
                if not torch.is_tensor(x) or x.shape[-1] <= 0:
                    return
                h = x.detach().reshape(-1, x.shape[-1]).float()
                ent = acc.get(name)
                if ent is None:
                    acc[name] = [h.pow(2).sum(0), h.shape[0]]
                else:
                    ent[0] += h.pow(2).sum(0)
                    ent[1] += h.shape[0]

            return hook

        handles = [
            base.register_forward_pre_hook(_make_hook(name)) for name, base in targets
        ]
        n_tok = 0
        moved_from_cpu = False
        try:
            device = next(model.parameters()).device
            run_device = device
            if device.type == "cpu" and torch.cuda.is_available():
                # The NVFP4 LoRA loader keeps the bf16 model on CPU (the swap
                # streams it to the GPU layer by layer), but the calibration
                # forward has to run on the GPU: flash-attention / Triton
                # linear-attention kernels have no CPU path. Hop the model over
                # for the ONE short no-grad forward and hop back, so the
                # streaming swap still starts from CPU. The bf16 weights fit on
                # the GPU long before a training step would (no optimizer/grad/
                # activation state); if they don't, the OOM lands in the except
                # below and the residual loudly disables as before.
                run_device = torch.device("cuda")
                model.to(run_device)
                moved_from_cpu = True
            ids = self._nvfp4_calib_input_ids(model, calib_tokens, run_device)
            if ids is None:
                raise RuntimeError("no calibration input_ids available")
            n_tok = int(ids.numel())
            was_training = model.training
            model.eval()
            # The fused-LoRA kernel patches (apply_qkv/apply_o, the
            # GatedDeltaNet apply_<proj> helpers, the fused MLP instance
            # forward) call the Triton kernels with raw weights — the hooked
            # base nn.Linear modules never run under them, so the pre-hooks
            # above would capture nothing. Reroute through the plain module
            # forwards for this ONE short no-grad pass (numerics-equivalent;
            # perf is irrelevant here).
            from axolotl.monkeypatch.lora_kernels import plain_lora_module_forwards

            with torch.no_grad(), plain_lora_module_forwards(model):
                model(ids)
            if was_training:
                model.train()
        except Exception as exc:  # pylint: disable=broad-exception-caught
            LOG.warning(
                "nvfp4 base_residual: activation calibration unavailable (%s); "
                "DISABLING the base residual (no plain-SVD fallback — it is "
                "pointless on the near-full-rank e2m1 quant error).",
                exc,
            )
            return None
        finally:
            for handle in handles:
                handle.remove()
            if moved_from_cpu:
                model.to(torch.device("cpu"))
                torch.cuda.empty_cache()

        calib = {name: (s / max(count, 1)) for name, (s, count) in acc.items()}
        if not calib:
            LOG.warning(
                "nvfp4 base_residual: calibration forward captured no activations; "
                "DISABLING the base residual."
            )
            return None
        missing = [name for name, _ in targets if name not in calib]
        if missing:
            LOG.warning(
                "nvfp4 base_residual: %d/%d target linears captured no calibration "
                "activations (e.g. %s); their residuals will be skipped.",
                len(missing),
                len(targets),
                missing[0],
            )
        LOG.info(
            "nvfp4 base_residual: calibrated %d frozen base linears over %d tokens "
            "(rank=%d)",
            len(calib),
            n_tok,
            rank,
        )
        return BaseResidualPlan(rank=rank, calibration="activation", calib=calib)

    def _nvfp4_capture_calib_hidden(self, model, n_tokens: int):
        """Capture ``[N, H]`` final-layer hidden states feeding the lm_head.

        Forwards a short slice of the training dataset (or a synthetic token range
        if none is reachable) through the model with a forward PRE-hook on the
        lm_head module itself, capturing exactly the activations it consumes —
        immune to how the wrapper packages model outputs (``ModelOutput``
        subclasses without ``last_hidden_state``, e.g. the Qwen3.5 conditional
        generation output, broke the old output-spelunking path). Falls back to
        a forward hook on the model body when no output embedding module is
        exposed. Best-effort: returns None on any failure, and the residual
        falls back to plain SVD.
        """
        import torch

        try:
            device = next(model.parameters()).device
            # Build a small input_ids batch: prefer a real train sample, else a
            # deterministic token range (the second moment per hidden dim is robust
            # to the exact tokens — it is dominated by the layernorm output scale).
            ids = self._nvfp4_calib_input_ids(model, n_tokens, device)
            if ids is None:
                return None

            captured = {}
            get_out = getattr(model, "get_output_embeddings", None)
            lm_head = get_out() if callable(get_out) else None
            if lm_head is not None:
                # Pre-hook on the head: args[0] is the hidden-state input,
                # whatever the model's output wrapper looks like.
                def pre_hook(_m, args):
                    x = args[0] if args else None
                    if torch.is_tensor(x):
                        captured["h"] = x.detach()

                handle = lm_head.register_forward_pre_hook(pre_hook)
                fwd_target = model
            else:
                # Fallback: hook the model body output (legacy path for models
                # without an exposed output embedding module).
                base = getattr(model, "model", None) or getattr(
                    getattr(model, "base_model", model), "model", None
                )
                if base is None:
                    return None

                def hook(_m, _inp, out):
                    h = getattr(out, "last_hidden_state", None)
                    if h is None:
                        h = out[0] if isinstance(out, (tuple, list)) else out
                    if torch.is_tensor(h):
                        captured["h"] = h.detach()

                handle = base.register_forward_hook(hook)
                fwd_target = base

            was_training = model.training
            model.eval()
            try:
                with torch.no_grad():
                    fwd_target(ids)
            finally:
                handle.remove()
                if was_training:
                    model.train()
            h = captured.get("h")
            return None if h is None else h.reshape(-1, h.shape[-1])
        except Exception as e:  # best-effort calibration
            LOG.warning(
                "lm_head_residual: calibration capture failed (%s); plain SVD", e
            )
            return None

    def _nvfp4_calib_input_ids(self, model, n_tokens: int, device):
        """A ``[1, n]`` input_ids tensor for residual calibration, or None."""
        import torch

        ds = getattr(self, "train_dataset", None) or getattr(
            getattr(self, "cfg", None), "_train_dataset", None
        )
        try:
            if ds is not None and len(ds) > 0:
                ids = ds[0].get("input_ids")
                if ids is not None:
                    t = torch.as_tensor(ids, device=device).reshape(1, -1)[:, :n_tokens]
                    if t.shape[1] >= 2:
                        return t
        except Exception:
            pass
        # Fallback: a deterministic in-vocab token range.
        vocab = getattr(getattr(model, "config", None), "vocab_size", None)
        if not vocab:
            return None
        n = min(n_tokens, 256)
        return (torch.arange(n, device=device) % vocab).reshape(1, n)

    @staticmethod
    def _nvfp4_attach_distillation(model, distill_cfg, retain_teacher: bool) -> None:
        """Attach the KL-distillation state + forward patch to the FP4 head model.

        No-op unless distillation is enabled and the bf16 teacher was retained on
        the lm_head. The forward patch recomputes the (tiled) teacher logits and
        adds ``lambda * T^2 * KL(teacher || student)`` to the loss; it is gated on
        labels being present (training) and the head exposing both a retained
        teacher and a tile-able FP4 student store.
        """
        if not (
            retain_teacher
            and distill_cfg is not None
            and getattr(distill_cfg, "enabled", False)
        ):
            return
        from axolotl.kernels.nvfp4_distill import (
            DistillState,
            patch_model_distillation,
        )

        state = DistillState(
            enabled=True,
            lambda_=float(getattr(distill_cfg, "lambda_", 1.0)),
            temperature=float(getattr(distill_cfg, "temperature", 1.0)),
            top_k=getattr(distill_cfg, "top_k", None),
            cadence=int(getattr(distill_cfg, "cadence", 1)),
            teacher=str(getattr(distill_cfg, "teacher", "live")),
            vocab_block=int(getattr(distill_cfg, "teacher_vocab_block", 8192)),
        )
        patch_model_distillation(model, state)
        LOG.info(
            "NVFP4 lm_head_distillation enabled (lambda=%s, T=%s, top_k=%s, "
            "cadence=%s, teacher=%s)",
            state.lambda_,
            state.temperature,
            state.top_k,
            state.cadence,
            state.teacher,
        )

    @staticmethod
    def _module_name(model: PreTrainedModel, target) -> str | None:
        if target is None:
            return None
        return next((n for n, m in model.named_modules() if m is target), None)

    def _apply_chunked_cross_entropy_patch(self):
        if self.cfg.chunked_cross_entropy:
            from axolotl.monkeypatch.loss.chunked import patch_chunked_ce_loss_fn

            if self.cfg.chunked_cross_entropy_num_chunks:
                patch_chunked_ce_loss_fn(self.cfg.chunked_cross_entropy_num_chunks)
            else:
                patch_chunked_ce_loss_fn()

    def _apply_fsdp_patches(self):
        """Apply patches for FSDP configurations."""
        if self.cfg.fsdp_config:
            from axolotl.monkeypatch.accelerate.fsdp2 import (
                patch_initialize_missing_keys_for_fsdp,
            )

            patch_initialize_missing_keys_for_fsdp()

            # Only for adapter (frozen-base) runs: this patch leaves non-rank-0 base params on meta
            # (FSDP broadcasts rank-0's weights into them), which avoids world_size× CPU
            # materialization of large unrecognized-quantizer (NVFP4-modelopt) checkpoints. Those are
            # always trained with a frozen base, so no base optimizer state exists. For a FULL
            # fine-tune the base params DO carry optimizer state, and leaving them on meta deadlocks
            # the FSDP2 optimizer-state all-gather at checkpoint save (rank-0 real DTensors vs
            # non-rank-0 meta) — so fall back to the stock materialize-to-cpu path there.
            if self.cfg.fsdp_config.cpu_ram_efficient_loading and self.cfg.adapter:
                from axolotl.monkeypatch.accelerate.fsdp2 import (
                    patch_move_missing_keys_meta_for_fsdp,
                )

                patch_move_missing_keys_meta_for_fsdp()

        if self.cfg.context_parallel_size > 1 or self.cfg.fsdp_config:
            from axolotl.monkeypatch.accelerate.parallelism_config import (
                patch_parallelism_config,
            )

            patch_parallelism_config()
        if (
            self.cfg.fsdp_config
            and self.cfg.adapter
            and self.cfg.fsdp_config.activation_checkpointing
        ):
            from axolotl.monkeypatch.peft.state_dict import (
                patch_peft_checkpoint_wrapper_prefixes,
            )

            patch_peft_checkpoint_wrapper_prefixes()
        if self.cfg.fsdp_config:
            from axolotl.monkeypatch.accelerate.float8_fsdp import patch_float8_fsdp
            from axolotl.monkeypatch.accelerate.fsdp2 import (
                patch_accelerate_fsdp2,
                patch_tied_keys_for_meta_device,
            )

            patch_accelerate_fsdp2()
            from axolotl.monkeypatch.accelerate.fsdp2_checkpoint import (
                patch_fsdp2_full_checkpoint,
            )

            patch_fsdp2_full_checkpoint()
            # FSDP2 sharding for any torchao Float8Tensor weights (no-op without torchao)
            patch_float8_fsdp()
            if self.cfg.fsdp_config.cpu_ram_efficient_loading:
                patch_tied_keys_for_meta_device()
            if self.cfg.rl:
                from axolotl.monkeypatch.trainer.trl import patch_trl_prepare_fsdp2

                patch_trl_prepare_fsdp2()

    def _apply_adapter_patches(self):
        """Apply patches for adapter configurations."""
        from axolotl.loaders.model import should_skip_peft_embedding_upcast

        if should_skip_peft_embedding_upcast(self.cfg) or (
            self.cfg.adapter and is_protrain_active(self.cfg)
        ):
            from axolotl.loaders.utils import get_linear_embedding_layers
            from axolotl.monkeypatch.peft.utils import patch_peft_prep_code

            patch_peft_prep_code(
                get_linear_embedding_layers(self.cfg.model_config_type)
            )

    def _apply_flex_attention_patches(self):
        """Apply patches for flexible attention."""
        if self.cfg.attn_implementation == "flex_attention":
            from axolotl.monkeypatch.attention.flex_attn import (
                patch_flex_wrapper,
            )

            flex_attn_compile_kwargs = self.cfg.flex_attn_compile_kwargs or {}
            patch_flex_wrapper(**flex_attn_compile_kwargs)

    def _apply_sageattn_patches(self):
        """Apply patches for SageAttention."""
        if self.cfg.attn_implementation == "sage":
            from axolotl.monkeypatch.attention.sage_attn import patch_sageattn

            patch_sageattn()

    _FUSED_ATTN_KERNEL_SUPPORTED = (
        "qwen3",
        "qwen3_moe",
        "qwen3_vl",
        "qwen3_vl_text",
        "qwen3_5",
        "qwen3_5_text",
        "qwen3_5_moe",
        "qwen3_5_moe_text",
        "gemma4",
        "gemma4_text",
        "gemma4_unified",
        "gemma4_unified_text",
    )

    @staticmethod
    def _warn_if_fused_attn_unsupported(cfg):
        """Warn when ``fused_attn_kernel`` targets an unsupported
        ``model_config_type`` (derived post-schema by ``normalize_config()``)."""
        if not getattr(cfg, "fused_attn_kernel", False):
            return
        mct = getattr(cfg, "model_config_type", None)
        support = get_model_support(mct)
        check_capability(support, "fused_attn_kernel", mct)
        resolved = resolve_model_support(support)
        if resolved is not None and "fused_attn_kernel" in resolved.capabilities:
            return
        if mct and mct not in PatchManager._FUSED_ATTN_KERNEL_SUPPORTED:
            LOG.warning(
                "`fused_attn_kernel: true` is set but model_config_type=%r is not "
                "in the supported set %s. The flag is a silent no-op for this "
                "model. Remove the flag or use one of the supported model families.",
                mct,
                sorted(PatchManager._FUSED_ATTN_KERNEL_SUPPORTED),
            )

    def _apply_model_specific_patches(self):
        """Apply patches specific to model architectures."""
        self._warn_if_fused_attn_unsupported(self.cfg)

        if getattr(self.cfg, "use_kernels", None):
            from axolotl.monkeypatch.kernelize_fixes import patch_kernelize_fixes

            patch_kernelize_fixes()

        if (
            self.cfg.model_config_type == "llama4"
            and self.cfg.llama4_linearized_experts
        ):
            from axolotl.monkeypatch.models.llama4.modeling import (
                patch_llama4_linearized_modeling,
            )

            patch_llama4_linearized_modeling()

        # packed rows (multipack) and flattened batches both mark document
        # boundaries with position_ids restarting at 0; recurrent mixers need
        # that threaded into their kernels as seq_idx / cu_seqlens.
        packed_boundaries_needed = bool(
            self.cfg.sample_packing or getattr(self.cfg, "batch_flattening", False)
        )

        if self.cfg.model_config_type == "nemotron_h":
            from axolotl.monkeypatch.models.nemotron_h.modeling import (
                guard_nemotron_h_fused_scan,
                patch_nemotron_h_modeling_packing,
            )

            # The fused Mamba2 kernel applies out_proj itself, which a
            # quantized weight cannot serve; guard it for every run.
            guard_nemotron_h_fused_scan()

            if packed_boundaries_needed:
                patch_nemotron_h_modeling_packing(
                    kernels_enabled=bool(getattr(self.cfg, "use_kernels", False))
                )

        if self.cfg.model_config_type == "falcon_h1" and packed_boundaries_needed:
            from axolotl.monkeypatch.models.falcon_h1.modeling import (
                patch_falcon_h1_modeling_packing,
            )

            patch_falcon_h1_modeling_packing()

        if (
            self.cfg.model_config_type == "granitemoehybrid"
            and packed_boundaries_needed
        ):
            from axolotl.monkeypatch.models.granitemoehybrid.modeling import (
                patch_granitemoehybrid_modeling_packing,
            )

            patch_granitemoehybrid_modeling_packing()

        if packed_boundaries_needed:
            self._apply_linear_attention_packing_patches()
            self._apply_ssm_packing_patches()

        from axolotl.monkeypatch.models.vision_patch_embed_linear import (
            patch_vision_patch_embed_linear,
            unpatch_vision_patch_embed_linear,
        )

        if getattr(self.cfg, "vision_patch_embed_linear", None) is not False:
            patch_vision_patch_embed_linear(self.cfg.model_config_type)
        else:
            unpatch_vision_patch_embed_linear(self.cfg.model_config_type)

        # Patches requiring CUDA
        if torch.cuda.is_available():
            if (
                self.cfg.model_config_type in ["qwen3_5", "qwen3_5_moe"]
                and self.cfg.is_multimodal
                and self.cfg.attn_uses_flash_lib
            ):
                from axolotl.monkeypatch.models.qwen3_5.modeling import (
                    patch_qwen3_5_vlm_flash_attention,
                )

                patch_qwen3_5_vlm_flash_attention()

            if self.cfg.model_config_type in (
                "gemma4",
                "gemma4_text",
                "gemma4_unified",
                "gemma4_unified_text",
            ):
                # Shared-KV side channel when activation checkpointing (PR #3611).
                fsdp_cfg = self.cfg.fsdp_config
                needs_shared_kv_workaround = (not self.inference) and bool(
                    self.cfg.gradient_checkpointing
                    or self.cfg.activation_offloading
                    or (fsdp_cfg is not None and fsdp_cfg.activation_checkpointing)
                )
                if self.cfg.model_config_type in (
                    "gemma4_unified",
                    "gemma4_unified_text",
                ):
                    from axolotl.monkeypatch.models.gemma4_unified.fused_attn import (
                        patch_gemma4_unified_fused_attn as patch_fused_attn,
                    )
                else:
                    from axolotl.monkeypatch.models.gemma4.fused_attn import (
                        patch_gemma4_fused_attn as patch_fused_attn,
                    )
                patch_fused_attn(
                    install_shared_kv_workaround=needs_shared_kv_workaround
                )

            if self.cfg.fused_attn_kernel and self.cfg.model_config_type == "qwen3":
                from axolotl.monkeypatch.models.qwen3.fused_attn import (
                    patch_qwen3_fused_attn,
                )

                patch_qwen3_fused_attn()

            if self.cfg.fused_attn_kernel and self.cfg.model_config_type == "qwen3_moe":
                from axolotl.monkeypatch.models.qwen3_moe.fused_attn import (
                    patch_qwen3_moe_fused_attn,
                )

                patch_qwen3_moe_fused_attn()

            if self.cfg.fused_attn_kernel and self.cfg.model_config_type in (
                "qwen3_vl",
                "qwen3_vl_text",
            ):
                from axolotl.monkeypatch.models.qwen3_vl.fused_attn import (
                    patch_qwen3_vl_fused_attn,
                )

                patch_qwen3_vl_fused_attn()

            if self.cfg.fused_attn_kernel and self.cfg.model_config_type in (
                "qwen3_5",
                "qwen3_5_text",
            ):
                from axolotl.monkeypatch.models.qwen3_5.fused_attn import (
                    patch_qwen3_5_fused_attn,
                )

                patch_qwen3_5_fused_attn()

            if self.cfg.fused_attn_kernel and self.cfg.model_config_type in (
                "qwen3_5_moe",
                "qwen3_5_moe_text",
            ):
                from axolotl.monkeypatch.models.qwen3_5_moe.fused_attn import (
                    patch_qwen3_5_moe_fused_attn,
                )

                patch_qwen3_5_moe_fused_attn()

    # model types whose mixers accept seq_idx through kwargs but whose model
    # forward never derives it from position_ids
    _SEQ_IDX_INJECTED_MODELS = {
        "lfm2": ("transformers.models.lfm2.modeling_lfm2", "Lfm2Model"),
        "lfm2_moe": ("transformers.models.lfm2_moe.modeling_lfm2_moe", "Lfm2MoeModel"),
        "bamba": ("transformers.models.bamba.modeling_bamba", "BambaModel"),
    }

    _SSM_PACKING_PATCHES = {
        "mamba": "patch_mamba_modeling_packing",
        "mamba2": "patch_mamba2_modeling_packing",
        "falcon_mamba": "patch_falcon_mamba_modeling_packing",
    }

    def _apply_ssm_packing_patches(self):
        """Thread packed-document boundaries into the pure-SSM Mamba family."""
        if getattr(self.model_config, "mamba_backend", None) == "fla":
            return
        patch_name = self._SSM_PACKING_PATCHES.get(self.cfg.model_config_type)
        if patch_name is None:
            return

        from axolotl.monkeypatch.models.mamba import modeling as mamba_modeling

        getattr(mamba_modeling, patch_name)(
            kernels_enabled=bool(getattr(self.cfg, "use_kernels", False))
        )

    # hub-wrapped kernels that must take seq_idx for the injection to mean anything
    _SEQ_IDX_KERNELS = (
        "causal_conv1d_fn",
        "mamba2_split_conv1d_scan_combined",
        "mamba2_chunk_scan",
    )

    def _apply_linear_attention_packing_patches(self):
        """Thread packed-document boundaries into GatedDeltaNet / short-conv mixers.

        Applied whenever position_ids carry document boundaries (sample packing
        or batch flattening), and regardless of CUDA availability: the patched
        forwards raise if the varlen kernels are missing, which beats the stock
        forwards silently mixing state across documents.
        """
        model_type = self.cfg.model_config_type

        if model_type == "qwen3_next":
            from axolotl.monkeypatch.models.qwen3_next.modeling import (
                patch_qwen3_next_modeling_packing,
            )

            patch_qwen3_next_modeling_packing(
                torch_compile=bool(self.cfg.torch_compile),
            )

        if model_type in ("qwen3_5", "qwen3_5_text"):
            from axolotl.monkeypatch.models.qwen3_5.modeling import (
                patch_qwen3_5_modeling_packing,
            )

            patch_qwen3_5_modeling_packing(
                torch_compile=bool(self.cfg.torch_compile),
            )

        if model_type in ("qwen3_5_moe", "qwen3_5_moe_text"):
            from axolotl.monkeypatch.models.qwen3_5.modeling import (
                patch_qwen3_5_moe_modeling_packing,
            )

            patch_qwen3_5_moe_modeling_packing(
                torch_compile=bool(self.cfg.torch_compile),
            )

        if model_type in self._SEQ_IDX_INJECTED_MODELS:
            import importlib

            from axolotl.monkeypatch.models.mamba_utils import (
                patch_model_forward_seq_idx,
                require_seq_idx_kernels,
            )

            module_name, cls_name = self._SEQ_IDX_INJECTED_MODELS[model_type]
            module = importlib.import_module(module_name)
            require_seq_idx_kernels(
                module,
                self._SEQ_IDX_KERNELS,
                model_type,
                bool(getattr(self.cfg, "use_kernels", False)),
            )
            patch_model_forward_seq_idx(getattr(module, cls_name))
            LOG.info("Applied %s sample packing patch (seq_idx injection)", model_type)

    @staticmethod
    def _fix_nemotron_h_conversion_mapping():
        """Remove the spurious embedding→embeddings WeightRenaming from the
        nemotron_h checkpoint conversion mapping.

        transformers registers:
            WeightRenaming("embedding.weight", "embeddings.weight")
        to handle a legacy checkpoint variant. Released checkpoints already use
        ``backbone.embeddings.weight``, and the rename's reverse (applied on
        save) turns it back into ``backbone.embedding.weight`` when merging
        LoRA adapters into the base model.
        """
        try:
            from transformers.conversion_mapping import (
                WeightRenaming,
                get_checkpoint_conversion_mapping,
                register_checkpoint_conversion_mapping,
            )
        except ImportError:
            return

        mapping = get_checkpoint_conversion_mapping("nemotron_h")
        if mapping is None:
            return

        filtered = [
            entry
            for entry in mapping
            if not (
                isinstance(entry, WeightRenaming)
                and entry.source_patterns == ["embedding.weight"]
                and entry.target_patterns == ["embeddings.weight"]
            )
        ]
        if len(filtered) != len(mapping):
            register_checkpoint_conversion_mapping(
                "nemotron_h", filtered, overwrite=True
            )
            LOG.info(
                "Removed embedding→embeddings WeightRenaming from nemotron_h "
                "checkpoint conversion mapping"
            )

    def _apply_fp8_patches(self):
        """Apply patches for FP8 support."""
        if self.cfg.fp8:
            from axolotl.monkeypatch.accelerate.float8_moe_filter import (
                patch_fp8_exclude_moe_router,
            )
            from axolotl.monkeypatch.trainer_accelerator_args import (
                patch_create_accelerate_code_for_fp8,
            )
            from axolotl.utils.schemas.fp8 import resolve_fp8_recipe

            fp8_recipe = resolve_fp8_recipe(self.cfg.get("fp8_config"))
            patch_create_accelerate_code_for_fp8(
                enable_fsdp_float8_all_gather=self.cfg.fp8_enable_fsdp_float8_all_gather,
                fp8_recipe=fp8_recipe,
            )
            patch_fp8_exclude_moe_router()

    def _apply_flash_attention_peft_patches(self):
        """Apply patches for Flash Attention with PEFT."""
        if self.cfg.adapter:
            from axolotl.monkeypatch.transformers_fa_utils import (
                patch_fa_peft_integration,
            )

            patch_fa_peft_integration()

    def _apply_gradient_checkpointing_patches(self):
        """Apply patches for gradient checkpointing."""
        if (
            self.cfg.gradient_checkpointing
            and self.cfg.activation_offloading == "legacy"
        ):
            from axolotl.monkeypatch.gradient_checkpointing import (
                hf_grad_checkpoint_offload_wrapper,
            )

            transformers.modeling_utils.checkpoint = hf_grad_checkpoint_offload_wrapper
        elif (
            self.cfg.gradient_checkpointing
            and self.cfg.activation_offloading == "offload_disk"
        ):
            from axolotl.monkeypatch.gradient_checkpointing import (
                hf_grad_checkpoint_disk_offload_wrapper,
            )

            transformers.modeling_utils.checkpoint = (
                hf_grad_checkpoint_disk_offload_wrapper
            )

    def _apply_mistral_cross_entropy_patch(self):
        """Apply Mistral cross entropy patch if configured."""
        if (
            self.cfg.model_config_type == "mistral"
            and self.cfg.flash_attn_cross_entropy_loss
        ):
            from axolotl.monkeypatch.mistral_attn_hijack_flash import (
                patch_mistral_cross_entropy,
            )

            patch_mistral_cross_entropy()

    def _apply_self_attention_lora_patch(self):
        """Apply self-attention LoRA patches if configured."""
        if self.cfg.lora_qkv_kernel or self.cfg.lora_o_kernel:
            check_capability(
                get_model_support(self.cfg.model_config_type),
                "lora_kernels",
                self.cfg.model_config_type,
                feature="LoRA QKV/O kernels",
                hint="Set lora_qkv_kernel: false and lora_o_kernel: false.",
            )
            from axolotl.monkeypatch.lora_kernels import patch_self_attn_lora

            patch_self_attn_lora(self.cfg)

    def _apply_large_head_attention_patch(self):
        """Generic head_dim>256 capability for plain SDPA models. Gemma-4's hybrid path routes its
        globals through its own impl, so skip the generic sdpa wrapper there to avoid double-wiring."""
        from axolotl.monkeypatch.attention.large_head import (
            resolve_large_head_policy,
            set_large_head_packed,
            set_large_head_policy,
            unpatch_sdpa_large_head,
        )

        policy = resolve_large_head_policy(self.cfg)
        # Always (re)set the policy global from this run's config so a long-lived process can't
        # inherit a previous run's stale auto/triton_flash policy on an sdpa run.
        set_large_head_policy(policy)
        set_large_head_packed(bool(self.cfg.sample_packing))
        from axolotl.monkeypatch.attention.large_head import (
            patch_flex_large_head,
            unpatch_flex_large_head,
        )

        # FlexAttention: only BlockMasks carrying segment metadata (packed encoder/canvas global
        # layers) are routed; everything else stays on flex.
        if policy != "sdpa" and self.cfg.attn_implementation == "flex_attention":
            patch_flex_large_head()
        else:
            unpatch_flex_large_head()
        if policy == "sdpa" or self.cfg.gemma4_hybrid_attn_impl:
            unpatch_sdpa_large_head()
            return
        from axolotl.monkeypatch.attention.large_head import patch_sdpa_large_head

        patch_sdpa_large_head(policy)

    def _apply_sdpa_varlen_patch(self):
        """Route packed-row SDPA through cu_seqlens ``varlen_attn`` (no 4D mask).

        Auto-enabled for ``sdpa`` + ``sample_packing`` when the varlen kernel can serve
        the model (torch >= 2.10, head_dim <= 256, no sliding window). Opt in/out
        explicitly with ``sdpa_varlen``. When varlen is unavailable/unsuitable, stock SDPA
        stays and packing is still isolated via the dropped-mask block-diagonal path.
        """
        # False -> explicit opt-out; True -> explicit opt-in; None -> auto for sdpa packing.
        if self.cfg.sdpa_varlen is False:
            return
        explicit = self.cfg.sdpa_varlen is True
        auto = (
            self.cfg.sdpa_varlen is None
            and self.cfg.attn_implementation == "sdpa"
            and self.cfg.sample_packing
        )
        if not (explicit or auto):
            return

        support = get_model_support(self.cfg.model_config_type)
        resolved = resolve_model_support(support) if support is not None else None
        capability = resolved.capabilities.get("sdpa_varlen") if resolved else None
        if isinstance(capability, Unsupported):
            if explicit:
                LOG.warning(
                    "sdpa_varlen is not supported for model_type=%s.%s Keeping stock SDPA.",
                    self.cfg.model_config_type,
                    f" {capability.reason}" if capability.reason else "",
                )
            return

        from axolotl.monkeypatch.attention.sdpa_varlen import (
            _VARLEN_MAX_HEAD_DIM,
            patch_sdpa_varlen,
            varlen_available,
        )

        if not varlen_available():
            return  # torch < 2.10; block-diagonal packing path is correct

        # Every call would fall back anyway, and patching drops the shared 4D mask.
        half = self.cfg.torch_dtype in (torch.float16, torch.bfloat16)
        if not (half and torch.cuda.is_available()):
            if explicit:
                LOG.info(
                    "sdpa_varlen: varlen_attn needs CUDA fp16/bf16; keeping stock SDPA "
                    "(packing still isolated via the block-diagonal mask)."
                )
            return

        def _attr(name):
            mc = self.model_config
            return mc.get(name) if isinstance(mc, dict) else getattr(mc, name, None)

        head_dim = _attr("head_dim")
        if not head_dim and _attr("hidden_size") and _attr("num_attention_heads"):
            head_dim = _attr("hidden_size") // _attr("num_attention_heads")
        # MLA (DeepSeek-V3, Ling 3.0, ...) pairs wider query/key heads with a
        # shorter value head, which the single-head_dim varlen layout cannot express.
        qk_head_dim = _attr("qk_head_dim") or (
            (_attr("qk_nope_head_dim") or 0) + (_attr("qk_rope_head_dim") or 0)
        )
        v_head_dim = _attr("v_head_dim")
        mismatched_head_dims = bool(
            qk_head_dim and v_head_dim and qk_head_dim != v_head_dim
        )
        sliding = _attr("sliding_window")
        layer_types = _attr("layer_types")
        uses_sliding = bool(sliding) and (
            any(lt == "sliding_attention" for lt in layer_types)
            if layer_types
            else True
        )

        if (
            (head_dim and head_dim > _VARLEN_MAX_HEAD_DIM)
            or uses_sliding
            or mismatched_head_dims
        ):
            if explicit:
                LOG.info(
                    "sdpa_varlen: model has head_dim > %d, a sliding window, or "
                    "query/value heads of different width; keeping stock SDPA "
                    "(packing still isolated via the block-diagonal mask).",
                    _VARLEN_MAX_HEAD_DIM,
                )
            return

        patch_sdpa_varlen()

    def _apply_multipack_patches(self):
        """Apply multipack patches if necessary."""
        if (
            self.cfg.model_config_type in SUPPORTED_MULTIPACK_MODEL_TYPES
            and self.cfg.attn_supports_packing
            and self.cfg.sample_packing
        ):
            # Get automap config if it exists
            auto_map_config = None
            if isinstance(self.model_config, dict) and "auto_map" in self.model_config:
                auto_map_config = self.model_config["auto_map"]
            elif hasattr(self.model_config, "auto_map"):
                auto_map_config = self.model_config.auto_map

            # Determine if the model has remote code
            if auto_map_config is not None:
                has_remote_code = "AutoModelForCausalLM" in auto_map_config
            else:
                has_remote_code = False

            if has_remote_code:
                has_remote_code = bool(self.cfg.trust_remote_code)

            patch_for_multipack(
                self.cfg.model_config_type,
                model_name=self.cfg.base_model,
                has_remote_code=has_remote_code,
            )

        if self.cfg.sample_packing:
            from axolotl.monkeypatch.data.batch_dataset_fetcher import (
                apply_multipack_dataloader_patch,
            )

            LOG.info("Applying multipack dataloader patch for sample packing...")
            apply_multipack_dataloader_patch()

    def _apply_fsdp2_bnb_patches(self):
        """Apply FSDP2 BNB patches."""
        if self.cfg.fsdp_config and (self.cfg.load_in_4bit or self.cfg.load_in_8bit):
            from axolotl.monkeypatch.fsdp2_qlora import (
                apply_init_dtype_attrs_patch,
                apply_init_sharded_param_patch,
                apply_init_unsharded_param_patch,
                apply_linear8bitlt_save_patch,
            )

            apply_init_sharded_param_patch()
            apply_init_unsharded_param_patch()
            apply_init_dtype_attrs_patch()
            if self.cfg.load_in_8bit:
                apply_linear8bitlt_save_patch()

    def _deactivate_hf_async_load(self):
        """Load weights synchronously so they can be converted and not OOM."""
        if self.cfg.load_in_4bit or self.cfg.load_in_8bit:
            os.environ["HF_DEACTIVATE_ASYNC_LOAD"] = "1"

    def _apply_moe_expert_quantization_patch(self):
        """Patch transformers weight loading and PEFT for MoE expert quantization."""
        has_target_params = bool(getattr(self.cfg, "lora_target_parameters", None))

        if not self.cfg.quantize_moe_experts and not has_target_params:
            return

        from axolotl.loaders.nf4 import uses_staged_nf4
        from axolotl.monkeypatch.moe_quant import (
            patch_peft_target_parameters_matching,
        )

        if self.cfg.quantize_moe_experts and not uses_staged_nf4(self.cfg):
            from axolotl.monkeypatch.moe_quant import patch_moe_quantization_on_load

            patch_moe_quantization_on_load(self.cfg)

        patch_peft_target_parameters_matching()

    def _finalize_moe_expert_quantization(self, model: PreTrainedModel):
        """Log quantization results and set model flag for downstream use."""
        import torch

        model._moe_experts_quantized = False
        if self.cfg.quantize_moe_experts:
            from axolotl.monkeypatch.moe_quant import get_moe_quantized_count

            count = get_moe_quantized_count()
            if count > 0:
                import gc

                model._moe_experts_quantized = True
                LOG.info(
                    "Quantized %d MoE expert parameter(s) to %s during model loading",
                    count,
                    "4-bit" if self.cfg.load_in_4bit else "8-bit",
                )
                gc.collect()
                torch.cuda.empty_cache()

    def _apply_tiled_mlp(self, model_type: str):
        if self.cfg.tiled_mlp:
            from axolotl.monkeypatch.tiled_mlp import (
                patch_tiled_mlp,
            )

            patch_tiled_mlp(
                model_type,
                use_original_mlp=self.cfg.tiled_mlp_use_original_mlp,
                cfg_num_shards=self.cfg.tiled_mlp_num_shards,
                use_scattermoe=bool(self.cfg.use_scattermoe),
            )

    def _apply_tiled_mlp_post_load(self, model):
        """Re-wrap MoE block instances after kernels have installed their forward.

        Needed only when scattermoe-lora is active — ``model.kernelize()``
        binds ``HFScatterMoEGatedMLP.forward`` per instance, which shadows
        the class-level tiled patch. See
        :func:`axolotl.monkeypatch.tiled_mlp.patch_tiled_mlp_moe_instances`.
        """
        if not (self.cfg.tiled_mlp and self.cfg.use_scattermoe):
            return
        from axolotl.monkeypatch.tiled_mlp import patch_tiled_mlp_moe_instances

        patch_tiled_mlp_moe_instances(
            model,
            self.cfg.model_config_type,
            cfg_num_shards=self.cfg.tiled_mlp_num_shards,
        )

    def _apply_voxtral_patches(self):
        """Apply patches for Voxtral model."""
        if self.cfg.model_config_type == "voxtral":
            from axolotl.monkeypatch.models.voxtral.modeling import (
                patch_voxtral_conditional_generation_forward,
            )

            patch_voxtral_conditional_generation_forward()

    def _apply_parallel_weight_conversion(self):
        """Parallel stack/cat in transformers' expert-fusing converters; off with AXOLOTL_PARALLEL_CONVERT=0."""
        if os.environ.get("AXOLOTL_PARALLEL_CONVERT", "1") == "0":
            return
        from axolotl.monkeypatch.parallel_weight_conversion import (
            patch_parallel_weight_conversion,
        )

        patch_parallel_weight_conversion()

    def _patch_attention(self):
        """Apply attention-specific patches based on model type."""
        if not (
            self.cfg.attn_uses_flash_lib and hasattr(self.model_config, "model_type")
        ):
            return

        if self.model_config.model_type == "btlm":
            from axolotl.monkeypatch.btlm_attn_hijack_flash import (
                replace_btlm_attn_with_flash_attn,
            )

            replace_btlm_attn_with_flash_attn(self.cfg.base_model)

        if self.model_config.model_type == "stablelm_epoch" and self.cfg.sample_packing:
            from axolotl.monkeypatch.stablelm_attn_hijack_flash import (
                replace_stablelm_attn_with_flash_attn,
            )

            replace_stablelm_attn_with_flash_attn(self.cfg.base_model)

        if self.model_config.model_type in ("mistral3", "llava"):
            from axolotl.monkeypatch.models.pixtral.modeling_flash_attention_utils import (
                apply_patch_is_packed_sequence,
            )

            apply_patch_is_packed_sequence()

    def _patch_loss_llama(self):
        """Patch loss functions and other optimizations for LLaMA models."""
        if not self.cfg.is_llama_derived_model:
            return

        if self.cfg.flash_attn_cross_entropy and self.has_flash_attn:
            from axolotl.monkeypatch.llama_attn_hijack_flash import (
                patch_fa_llama_cross_entropy,
            )

            patch_fa_llama_cross_entropy()

    def _patch_llama_xformers_attention(self):
        """Apply xformers attention patches for LLaMA models."""
        from axolotl.monkeypatch.llama_attn_hijack_xformers import (
            hijack_llama_attention,
        )

        LOG.info("Patching with xformers attention...")
        hijack_llama_attention()

    def _patch_llama_derived_model(self):
        """Modify all llama derived models in one block."""
        if (
            self.cfg.is_llama_derived_model
            and self.cfg.attn_implementation == "xformers"
            and not (
                self.cfg.model_config_type in SUPPORTED_MULTIPACK_MODEL_TYPES
                and self.cfg.attn_supports_packing
                and self.cfg.sample_packing
            )
        ):
            self._patch_llama_xformers_attention()

    def _apply_llama_flash_attn_patches(self, model):
        """Apply LLaMA-specific flash attention patches."""

        if (
            self.model_config.model_type
            in ["llama", "llama4", "ernie4_5", "ernie4_5_moe"]
            and not self.cfg.trust_remote_code
            and not self.cfg.gptq
            and self.cfg.attn_uses_flash_lib
            and is_flash_attn_available()
            and not self.inference
        ):
            try:
                # TODO(MengqingCao): split these patches separately
                from axolotl.monkeypatch.llama_attn_hijack_flash import (
                    is_xformers_swiglu_available,
                    replace_llama_mlp_with_swiglu,
                )

                if self.cfg.flash_attn_fuse_mlp and is_xformers_swiglu_available():
                    LOG.info("Patching with SwiGLU...")
                    replace_llama_mlp_with_swiglu(model)
            except ImportError as e:
                LOG.warning(f"Flash Attention patches not applied: {e}")

    def _apply_lora_kernel_patch(self, model):
        """Apply LoRA kernel patches."""
        if (
            self.cfg.lora_mlp_kernel
            or self.cfg.lora_qkv_kernel
            or self.cfg.lora_o_kernel
        ):
            from peft import PeftModel

            from axolotl.monkeypatch.lora_kernels import apply_lora_kernel_patches

            if not isinstance(model, PeftModel) and rl_trainer_builds_adapter(self.cfg):
                # TRL wraps the adapter inside the trainer; the builder patches it then
                LOG.info(
                    "Deferring the fused LoRA kernel patch to the RL trainer build"
                )
                return
            apply_lora_kernel_patches(model=model, cfg=self.cfg)

    def _apply_patch_deepspeed_zero3(self):
        try:
            from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled

            from axolotl.monkeypatch.deepspeed_utils import apply_deepspeed_patches

            if self.cfg.activation_offloading is True and (
                is_deepspeed_zero3_enabled()
                or os.getenv("ACCELERATE_DEEPSPEED_ZERO_STAGE") == "3"
            ):
                apply_deepspeed_patches()
        except ImportError as e:
            LOG.warning(f"DeepSpeed patches not applied: {e}")

    def _apply_apertus_patches(self):
        """Apply patches for Apertus model."""
        if self.cfg.model_config_type == "apertus":
            from axolotl.monkeypatch.models.apertus.activation import (
                patch_apertus_xielu_activation,
            )

            patch_apertus_xielu_activation()

    def _apply_trl_vllm_patches(self):
        """Apply TRL vLLM patches: batched weight sync, NaN logprobs fix, scalar
        handling, and colocate engine kwargs."""
        if (
            self.cfg.rl
            and getattr(self.cfg, "trl", None)
            and getattr(self.cfg.trl, "use_vllm", False)
        ):
            from axolotl.monkeypatch.trainer.trl_vllm import (
                colocate_vllm_engine_kwargs,
                patch_trl_vllm,
                patch_vllm_colocate_engine_kwargs,
            )

            patch_trl_vllm()
            if getattr(self.cfg.trl, "vllm_mode", None) == "colocate":
                patch_vllm_colocate_engine_kwargs(
                    colocate_vllm_engine_kwargs(getattr(self.cfg, "vllm", None))
                )

    def _apply_scaling_softmax_patch(self, model: PreTrainedModel):
        """Apply Scaling Softmax (SSMax) patch.  Ref: https://arxiv.org/abs/2501.19399"""
        if self.cfg.scaling_softmax:
            from axolotl.monkeypatch.scaled_softmax_attn import (
                patch_scaled_softmax_attention,
            )

            patch_scaled_softmax_attention(
                scaling_factor_init=self.cfg.scaling_softmax_factor or 0.43,
                bias=self.cfg.scaling_softmax_bias or 0.0,
                model=model,
            )
