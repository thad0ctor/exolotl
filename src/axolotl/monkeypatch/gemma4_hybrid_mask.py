"""Hybrid attention mask fix for Gemma 4 (standard and unified).

Gemma 4 has full-attention (global) layers with ``head_dim=512`` which
exceeds flash-attention-2's supported size. Axolotl's hybrid-attention
patch in ``patch_manager._apply_gemma_hybrid_attention`` works around
this by forcing ``_attn_implementation="sdpa"`` on each global layer's
``self_attn.config``, leaving sliding-window layers on FA2.

The per-layer config override alone is insufficient, however:
``Gemma4TextModel.forward`` builds a single ``causal_mask_mapping`` dict
using the **model-level** config and passes the mapped mask to each
decoder layer. With FA2 still set at the model level, the ``full_attention``
entry in that mapping is a 2D mask (FA2 format), but SDPA needs a 4D mask.
The global layers then fail with::

    RuntimeError: The expanded size of the tensor (S) must match the existing
    size (B) at non-singleton dimension 2. Target sizes: [B, H, S, S]. Tensor
    sizes: [B, S]

...when the sequence length grows past roughly 7k tokens.

This module fixes the symptom by monkey-patching the mask-builder symbols in
the model's *module namespace* — NOT the originals in ``masking_utils``. Each
wrapper forces ``_attn_implementation="sdpa"`` on a shallow-copied config
before calling through, so the ``full_attention`` mask comes out
4D/SDPA-compatible. ``create_sliding_window_causal_mask`` is left alone, so
sliding-window layers continue to receive FA2-format masks.

Two forward paths exist, both must be covered:

* ``Gemma4TextModel.forward`` calls ``create_causal_mask`` directly.
* ``Gemma4Model.forward`` (multimodal) branches: the vision branch goes
  through ``create_causal_mask_mapping`` (covered transitively via
  module-level ``create_causal_mask``); the non-vision branch calls
  ``create_masks_for_generate``, whose dispatch lives in ``masking_utils``
  and is NOT reached by the ``create_causal_mask`` rebind — so it is wrapped
  explicitly via the ``_WRAPPED_SYMBOLS`` registry below.

``gemma4_unified`` reproduces the same mixed sliding/global architecture
(``global_head_dim=512``) in its own ``modeling_gemma4_unified`` namespace,
so both namespaces are patched when present.

Beyond the mask format, two further global-layer concerns are handled once a
Gemma 4 namespace is patched: ``_register_global_packed_sdpa`` rebuilds
block-diagonal masks from ``position_ids`` so head_dim=512 global layers
respect document boundaries under sample packing, and
``_patch_use_gqa_head_dim_guard`` keeps those layers on the memory-efficient
SDPA backend (head_dim>256 ``enable_gqa`` otherwise drops to the MATH kernel).

The patch is idempotent. Install once per process, before any Gemma 4
forward pass runs.
"""

from __future__ import annotations

import copy
import importlib
from typing import Any, Callable

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

_PATCH_APPLIED = False

# Attention-interface name for Gemma-4 global (full_attention) layers. They have head_dim=512, so FA2
# can't serve them and the model (FA2 at the top level) hands them no mask, relying on cu_seqlens that
# only the FA2 sliding layers consume. With sample packing that means the global layers would attend
# ACROSS document boundaries (pure causal). This impl rebuilds the block-diagonal-causal mask from
# position_ids so the globals respect doc boundaries, on the memory-efficient SDPA backend.
GLOBAL_PACKED_SDPA = "sdpa_global_packed"


# When set, the head_dim=512 global layers use the Triton flash_d512 kernel (fwd+bwd, varlen) instead
# of the SDPA efficient backend (~2x faster at head_dim 512). Set from cfg.flash_attn_d512.
def set_flash_d512(enabled: bool) -> None:
    """Backwards-compat shim: the head_dim>256 routing is now the generic large_head_attention
    capability. True -> 'auto' (flash only on packed rows, the proven win)."""
    from axolotl.monkeypatch.attention.large_head import set_large_head_policy

    set_large_head_policy("auto" if enabled else "sdpa")


def _packing_block_causal_mask(position_ids, dtype, device):
    """Block-diagonal causal additive mask [B,1,S,S] from packed position_ids (which reset to 0 at
    each document start). -inf across document boundaries and for non-causal positions, 0 elsewhere."""
    import torch

    if position_ids.dim() == 1:
        position_ids = position_ids[None]
    Bz, Sz = position_ids.shape
    doc = (position_ids == 0).cumsum(-1)  # [B,S] document index (1-based)
    same_doc = doc[:, :, None] == doc[:, None, :]  # [B,S,S]
    causal = torch.ones(Sz, Sz, dtype=torch.bool, device=device).tril()[None]  # [1,S,S]
    allow = same_doc & causal
    mask = torch.zeros(Bz, 1, Sz, Sz, dtype=dtype, device=device)
    mask.masked_fill_(~allow[:, None], torch.finfo(dtype).min)
    return mask


def _register_global_packed_sdpa() -> None:
    """Register the packing-aware global-layer attention impl (block-diagonal mask + efficient SDPA)."""
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    if GLOBAL_PACKED_SDPA in ALL_ATTENTION_FUNCTIONS.valid_keys():
        return

    def sdpa_global_packed_forward(module, query, key, value, attention_mask, **kwargs):
        # The top-level FA2 path leaves these layers maskless and carries packing only via cu_seqlens
        # (consumed by FA2, not SDPA). Rebuild the block-diagonal mask from position_ids so the global
        # layers don't cross document boundaries. Single-document rows stay maskless (is_causal).
        from axolotl.monkeypatch.attention.large_head import flash_d512_route

        position_ids = kwargs.get("position_ids")
        # The generic large-head router takes head_dim>256 packed rows through the Triton flash
        # kernel (per the large_head_attention policy); ~2.7x the 4D-mask SDPA, ~3x less memory than
        # nested-tensor SDPA. It declines (returns None) for single-doc/policy=sdpa -> SDPA below.
        if attention_mask is None:
            routed = flash_d512_route(
                module, query, key, value, kwargs.get("scaling"), position_ids
            )
            if routed is not None:
                return routed
        # Detect genuine (multi-document) packing: more doc-starts than batch rows.
        pid = None
        if attention_mask is None and position_ids is not None:
            p = position_ids if position_ids.dim() > 1 else position_ids[None]
            if int((p == 0).sum()) > p.shape[0]:
                pid = p
        # Packed without the kernel -> block-diagonal mask so globals respect doc boundaries.
        # Single-document -> mask stays None (SDPA is_causal, the fast path).
        if pid is not None:
            attention_mask = _packing_block_causal_mask(pid, query.dtype, query.device)
        return sdpa_attention_forward(
            module, query, key, value, attention_mask, **kwargs
        )

    ALL_ATTENTION_FUNCTIONS.register(GLOBAL_PACKED_SDPA, sdpa_global_packed_forward)
    LOG.info(
        "gemma4_hybrid_mask: registered '%s' (block-diagonal packing mask for head_dim=512 global "
        "layers so they respect document boundaries under sample packing)",
        GLOBAL_PACKED_SDPA,
    )


# MagicMock auto-spawns attributes, so ``hasattr(fn, "_axolotl_original")``
# alone would falsely identify mocks as wrappers. Marker fixes that.
_WRAPPER_MARKER = "_axolotl_gemma4_hybrid_mask_wrapper"


def _is_axolotl_wrapper(fn: Any) -> bool:
    return callable(fn) and getattr(fn, _WRAPPER_MARKER, False) is True


def _build_create_causal_mask_wrapper(original: Callable) -> Callable:
    """Force SDPA format for the full-attention mask.

    The global layers were patched to SDPA by ``_apply_gemma_hybrid_attention``,
    so their mask must be 4D. The original ``create_causal_mask`` dispatches on
    ``config._attn_implementation``; we shadow that with a local override on a
    shallow copy so the caller's config is left intact (the sliding-window
    factory still reads FA2 from it).
    """

    def hybrid_create_causal_mask(config: Any, *args: Any, **kwargs: Any):
        sdpa_config = copy.copy(config)
        sdpa_config._attn_implementation = "sdpa"
        return original(sdpa_config, *args, **kwargs)

    hybrid_create_causal_mask._axolotl_original = original  # type: ignore[attr-defined]
    setattr(hybrid_create_causal_mask, _WRAPPER_MARKER, True)
    return hybrid_create_causal_mask


def _build_create_masks_for_generate_wrapper(original: Callable) -> Callable:
    """Wrap ``create_masks_for_generate`` for the multimodal forward path.

    For Gemma 4 the original returns a dict keyed by layer pattern, each
    entry built from a single config — so an FA2 config produces a 2D
    ``full_attention`` mask the SDPA global layers crash on. We call the
    original twice (FA2 then SDPA) and keep ``full_attention`` from the
    SDPA result, everything else from the FA2 result. The doubled mask
    allocation (~128 MB at S=8192, B=2) is dwarfed by attention cost in
    the failure-mode batch shapes; if it ever matters, swap the second
    call for a direct ``LAYER_PATTERN_TO_MASK_FUNCTION_MAPPING["full_attention"]``
    invocation. Non-dict returns pass through (no hybrid layer types).
    """

    def hybrid_create_masks_for_generate(config: Any, *args: Any, **kwargs: Any):
        original_result = original(config, *args, **kwargs)

        if not isinstance(original_result, dict):
            return original_result
        if "full_attention" not in original_result:
            return original_result

        sdpa_config = copy.copy(config)
        sdpa_config._attn_implementation = "sdpa"
        sdpa_result = original(sdpa_config, *args, **kwargs)

        if not isinstance(sdpa_result, dict) or "full_attention" not in sdpa_result:
            # SDPA call took a different branch; bail out rather than guess.
            return original_result

        merged = dict(original_result)
        merged["full_attention"] = sdpa_result["full_attention"]
        return merged

    hybrid_create_masks_for_generate._axolotl_original = original  # type: ignore[attr-defined]
    setattr(hybrid_create_masks_for_generate, _WRAPPER_MARKER, True)
    return hybrid_create_masks_for_generate


# Symbols missing on older transformers versions are skipped at install.
_WRAPPED_SYMBOLS: dict[str, Callable[[Callable], Callable]] = {
    "create_causal_mask": _build_create_causal_mask_wrapper,
    "create_masks_for_generate": _build_create_masks_for_generate_wrapper,
}


# Each Gemma 4 variant fully redefines ``create_causal_mask`` in its own module
# namespace (gemma4_unified does NOT modular-import from gemma4), so both must be
# patched independently.
_TARGET_MODULES = (
    "transformers.models.gemma4.modeling_gemma4",
    "transformers.models.gemma4_unified.modeling_gemma4_unified",
)


def _patch_module(module: Any) -> tuple[list[str], list[str]]:
    """Wrap every known mask-builder symbol present in one module namespace.

    Returns ``(installed, covered)`` where ``installed`` is the symbols newly
    wrapped this call and ``covered`` is every symbol left in a wrapped state
    (newly wrapped *or* already our wrapper from a prior call). A symbol that is
    already our wrapper is left untouched — this keeps re-entry idempotent even
    if ``_PATCH_APPLIED`` was reset externally, without re-wrapping a wrapper.
    Missing symbols (older transformers) are skipped.
    """
    installed: list[str] = []
    covered: list[str] = []
    for symbol_name, build_wrapper in _WRAPPED_SYMBOLS.items():
        current = getattr(module, symbol_name, None)
        if current is None:
            continue
        if _is_axolotl_wrapper(current):
            covered.append(symbol_name)
            continue
        setattr(module, symbol_name, build_wrapper(current))
        installed.append(symbol_name)
        covered.append(symbol_name)
    return installed, covered


def _patch_use_gqa_head_dim_guard() -> bool:
    """Stop ``enable_gqa`` from forcing the MATH SDPA backend on large-head-dim layers.

    ``sdpa_attention_forward`` enables ``enable_gqa=True`` whenever ``attention_mask is None``
    (``use_gqa_in_sdpa``), with no head_dim check. But SDPA's flash/efficient GQA path only
    supports head_dim <= 256; at head_dim > 256 ``enable_gqa`` silently falls back to the MATH
    kernel, which materializes the full [H, S, S] scores. Repeating KV instead keeps the
    memory-efficient backend. For Gemma-4's head_dim=512 global layers this is ~2.9 GiB -> ~0.2 GiB
    per layer with identical math (repeat_kv == GQA).
    """
    try:
        import transformers.integrations.sdpa_attention as sdpa_mod
    except ImportError:
        return False
    original = sdpa_mod.use_gqa_in_sdpa
    if getattr(original, "_axolotl_head_dim_guarded", False):
        return True

    def use_gqa_in_sdpa_guarded(attention_mask, key):
        # head_dim > 256 -> enable_gqa drops to the MATH backend; force repeat_kv (efficient) instead.
        if key.shape[-1] > 256:
            return False
        return original(attention_mask, key)

    use_gqa_in_sdpa_guarded._axolotl_head_dim_guarded = True  # type: ignore[attr-defined]
    use_gqa_in_sdpa_guarded._axolotl_original = original  # type: ignore[attr-defined]
    sdpa_mod.use_gqa_in_sdpa = use_gqa_in_sdpa_guarded
    LOG.info(
        "gemma4_hybrid_mask: guarded use_gqa_in_sdpa (head_dim>256 -> repeat_kv, not enable_gqa) "
        "to keep the memory-efficient SDPA backend on head_dim=512 global layers"
    )
    return True


def patch_gemma4_hybrid_mask() -> bool:
    """Install the Gemma 4 hybrid-attention mask fix across all variants.

    Returns ``True`` if at least one namespace was patched, ``False`` if none
    of the target modules could be imported (e.g. transformers version predates
    Gemma 4) — in which case nothing is done and the caller can continue
    unaffected.
    """
    global _PATCH_APPLIED
    if _PATCH_APPLIED:
        return True

    patched_any = False
    for module_path in _TARGET_MODULES:
        try:
            module = importlib.import_module(module_path)
        except ImportError:
            LOG.debug(
                "gemma4_hybrid_mask: %s not importable, skipping. This is fine "
                "for non-Gemma4 training.",
                module_path,
            )
            continue
        installed, covered = _patch_module(module)
        if covered:
            patched_any = True
        if installed:
            LOG.info(
                "gemma4_hybrid_mask: patched %s symbols [%s] to force "
                "SDPA-format masks for full-attention layers",
                module_path,
                ", ".join(installed),
            )

    if not patched_any:
        LOG.warning(
            "gemma4_hybrid_mask: no expected mask-builder symbols found on any "
            "target module (looked for: %s). Transformers API may have changed; "
            "the hybrid attention bug fix is not active.",
            ", ".join(_WRAPPED_SYMBOLS),
        )
        return False

    # Only touch global SDPA state once we know a Gemma4 namespace was actually patched —
    # otherwise _PATCH_APPLIED stays False and unpatch() would skip cleaning these up.
    _patch_use_gqa_head_dim_guard()
    _register_global_packed_sdpa()
    _PATCH_APPLIED = True
    return True


def unpatch_gemma4_hybrid_mask() -> None:
    """Restore the original mask-builder symbols in every namespace. Tests."""
    global _PATCH_APPLIED
    if not _PATCH_APPLIED:
        return
    try:
        import transformers.integrations.sdpa_attention as sdpa_mod

        guarded = getattr(sdpa_mod, "use_gqa_in_sdpa", None)
        original = getattr(guarded, "_axolotl_original", None)
        if original is not None:
            sdpa_mod.use_gqa_in_sdpa = original
    except ImportError:
        pass
    for module_path in _TARGET_MODULES:
        try:
            module = importlib.import_module(module_path)
        except ImportError:
            continue
        for symbol_name in _WRAPPED_SYMBOLS:
            current = getattr(module, symbol_name, None)
            # Only restore if a downstream patch hasn't replaced our wrapper.
            if _is_axolotl_wrapper(current):
                setattr(module, symbol_name, current._axolotl_original)
    _PATCH_APPLIED = False
