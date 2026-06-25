"""Tests for the Gemma 4 hybrid-attention mask fix.

Pin that the patched mask builders pass an SDPA-overridden config to the
underlying factory, so full-attention global layers receive a 4D mask
instead of the 2D FA2 mask the model-level config would produce. Two
entry points must be covered: ``create_causal_mask`` (text-only path,
plus transitive coverage of ``create_causal_mask_mapping``) and
``create_masks_for_generate`` (multimodal non-vision branch — its
dispatch lives in ``masking_utils`` so the ``create_causal_mask`` rebind
doesn't reach it).
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytest.importorskip(
    "transformers.models.gemma4",
    reason="gemma4_hybrid_mask patch only matters when Gemma 4 is available",
)


@pytest.fixture
def restore_gemma4_module():
    """Snapshot patch-target symbols across all variants and restore after each test."""
    import importlib

    from transformers.models.gemma4 import modeling_gemma4

    from axolotl.monkeypatch import gemma4_hybrid_mask

    _MISSING = object()
    snapshots: dict = {}
    for module_path in gemma4_hybrid_mask._TARGET_MODULES:
        try:
            module = importlib.import_module(module_path)
        except ImportError:
            continue
        snapshots[module] = {
            name: getattr(module, name, _MISSING)
            for name in gemma4_hybrid_mask._WRAPPED_SYMBOLS
        }
    yield modeling_gemma4
    for module, names in snapshots.items():
        for name, original in names.items():
            if original is _MISSING:
                if hasattr(module, name):
                    delattr(module, name)
            else:
                setattr(module, name, original)
    gemma4_hybrid_mask._PATCH_APPLIED = False


# ---------------------------------------------------------------------------
# Unit tests: create_causal_mask wrapper
# ---------------------------------------------------------------------------


def test_patch_replaces_create_causal_mask(restore_gemma4_module):
    modeling_gemma4 = restore_gemma4_module
    from axolotl.monkeypatch.gemma4_hybrid_mask import patch_gemma4_hybrid_mask

    original = modeling_gemma4.create_causal_mask
    assert patch_gemma4_hybrid_mask() is True

    assert modeling_gemma4.create_causal_mask is not original
    assert modeling_gemma4.create_causal_mask._axolotl_original is original, (
        "patched wrapper must expose the original reference for teardown"
    )


def test_patch_is_idempotent(restore_gemma4_module):
    modeling_gemma4 = restore_gemma4_module
    from axolotl.monkeypatch.gemma4_hybrid_mask import patch_gemma4_hybrid_mask

    patch_gemma4_hybrid_mask()
    wrapper_first = modeling_gemma4.create_causal_mask

    # Second call must not re-wrap the already-wrapped function (which
    # would leak the original reference through a chain of wrappers).
    patch_gemma4_hybrid_mask()
    wrapper_second = modeling_gemma4.create_causal_mask

    assert wrapper_first is wrapper_second


def test_patched_mask_forces_sdpa_config(restore_gemma4_module):
    """Core invariant: caller's FA2 config must reach the factory as SDPA
    on a shallow copy, leaving the caller's config untouched."""
    modeling_gemma4 = restore_gemma4_module
    from axolotl.monkeypatch.gemma4_hybrid_mask import patch_gemma4_hybrid_mask

    # Mock must be installed before the patch so it gets captured as original.
    mock_factory = MagicMock(name="create_causal_mask", return_value="mask_4d")
    modeling_gemma4.create_causal_mask = mock_factory
    patch_gemma4_hybrid_mask()

    caller_config = SimpleNamespace(
        _attn_implementation="flash_attention_2",
        head_dim=512,
        some_other_attr="preserved",
    )
    result = modeling_gemma4.create_causal_mask(
        caller_config,
        inputs_embeds=None,
        attention_mask=None,
        past_key_values=None,
        position_ids=None,
    )

    assert result == "mask_4d"
    assert mock_factory.call_count == 1
    (passed_config, *_), passed_kwargs = mock_factory.call_args
    assert passed_config._attn_implementation == "sdpa"
    # Caller's config untouched — other builders share it and need FA2.
    assert caller_config._attn_implementation == "flash_attention_2"
    assert passed_config.head_dim == 512
    assert passed_config.some_other_attr == "preserved"


def test_patched_wrapper_passes_through_all_kwargs(restore_gemma4_module):
    """Forward args/kwargs unchanged so we survive transformers signature drift."""
    modeling_gemma4 = restore_gemma4_module
    from axolotl.monkeypatch.gemma4_hybrid_mask import patch_gemma4_hybrid_mask

    mock_factory = MagicMock(return_value="mask")
    modeling_gemma4.create_causal_mask = mock_factory
    patch_gemma4_hybrid_mask()

    caller_config = SimpleNamespace(_attn_implementation="flash_attention_2")
    modeling_gemma4.create_causal_mask(
        caller_config,
        "positional_arg",
        inputs_embeds="embeds",
        attention_mask="mask_2d",
        past_key_values="cache",
        position_ids="positions",
        or_mask_function="or_fn",
    )

    args, kwargs = mock_factory.call_args
    assert args[1] == "positional_arg"
    assert kwargs["inputs_embeds"] == "embeds"
    assert kwargs["attention_mask"] == "mask_2d"
    assert kwargs["past_key_values"] == "cache"
    assert kwargs["position_ids"] == "positions"
    assert kwargs["or_mask_function"] == "or_fn"


def test_unpatch_restores_original(restore_gemma4_module):
    modeling_gemma4 = restore_gemma4_module
    from axolotl.monkeypatch.gemma4_hybrid_mask import (
        patch_gemma4_hybrid_mask,
        unpatch_gemma4_hybrid_mask,
    )

    sentinel = MagicMock(name="original")
    modeling_gemma4.create_causal_mask = sentinel
    patch_gemma4_hybrid_mask()
    assert modeling_gemma4.create_causal_mask is not sentinel

    unpatch_gemma4_hybrid_mask()
    assert modeling_gemma4.create_causal_mask is sentinel


def test_unpatch_is_safe_without_prior_patch(restore_gemma4_module):
    from axolotl.monkeypatch.gemma4_hybrid_mask import unpatch_gemma4_hybrid_mask

    unpatch_gemma4_hybrid_mask()


def test_sliding_window_mask_builder_is_not_patched(restore_gemma4_module):
    """Sliding-window factory must stay unwrapped so sliding layers keep
    FA2 masks (and the FA2 speedup)."""
    modeling_gemma4 = restore_gemma4_module
    from axolotl.monkeypatch.gemma4_hybrid_mask import patch_gemma4_hybrid_mask

    if not hasattr(modeling_gemma4, "create_sliding_window_causal_mask"):
        pytest.skip("transformers version has no create_sliding_window_causal_mask")

    sliding_before = modeling_gemma4.create_sliding_window_causal_mask
    patch_gemma4_hybrid_mask()
    sliding_after = modeling_gemma4.create_sliding_window_causal_mask
    assert sliding_after is sliding_before


# ---------------------------------------------------------------------------
# Unit tests: create_masks_for_generate wrapper (multimodal forward path)
# ---------------------------------------------------------------------------


def test_patch_replaces_create_masks_for_generate(restore_gemma4_module):
    """Multimodal non-vision forward goes through this symbol; without a
    wrapper its ``full_attention`` entry is 2D and the SDPA layers crash."""
    modeling_gemma4 = restore_gemma4_module
    from axolotl.monkeypatch.gemma4_hybrid_mask import patch_gemma4_hybrid_mask

    if not hasattr(modeling_gemma4, "create_masks_for_generate"):
        pytest.skip(
            "transformers version has no modeling_gemma4.create_masks_for_generate"
        )

    original = modeling_gemma4.create_masks_for_generate
    assert patch_gemma4_hybrid_mask() is True

    assert modeling_gemma4.create_masks_for_generate is not original
    assert modeling_gemma4.create_masks_for_generate._axolotl_original is original, (
        "patched wrapper must expose the original reference for teardown"
    )


def test_create_masks_for_generate_wrapper_replaces_full_attention_only(
    restore_gemma4_module,
):
    """Wrapper rebuilds only ``full_attention`` from an SDPA call, keeps
    every other key from the FA2 call. Caller's config must not mutate."""
    modeling_gemma4 = restore_gemma4_module
    from axolotl.monkeypatch.gemma4_hybrid_mask import patch_gemma4_hybrid_mask

    if not hasattr(modeling_gemma4, "create_masks_for_generate"):
        pytest.skip("transformers version lacks create_masks_for_generate")

    sdpa_full_mask = "sdpa_full_4d_sentinel"
    fa2_full_mask = "fa2_full_2d_sentinel"
    fa2_sliding_mask = "fa2_sliding_2d_sentinel"
    sdpa_sliding_mask = "sdpa_sliding_4d_sentinel"

    def fake_original(config, *args, **kwargs):
        if config._attn_implementation == "sdpa":
            return {
                "full_attention": sdpa_full_mask,
                "sliding_attention": sdpa_sliding_mask,
            }
        return {
            "full_attention": fa2_full_mask,
            "sliding_attention": fa2_sliding_mask,
        }

    mock_original = MagicMock(side_effect=fake_original)
    modeling_gemma4.create_masks_for_generate = mock_original
    patch_gemma4_hybrid_mask()

    caller_config = SimpleNamespace(_attn_implementation="flash_attention_2")
    result = modeling_gemma4.create_masks_for_generate(
        caller_config,
        inputs_embeds=None,
        attention_mask=None,
        past_key_values=None,
        position_ids=None,
    )

    assert result == {
        "full_attention": sdpa_full_mask,
        "sliding_attention": fa2_sliding_mask,
    }
    assert mock_original.call_count == 2
    assert caller_config._attn_implementation == "flash_attention_2"


def test_create_masks_for_generate_wrapper_passes_through_non_dict(
    restore_gemma4_module,
):
    """Non-dict return (no ``layer_types``) is passed through; no SDPA shadow call."""
    modeling_gemma4 = restore_gemma4_module
    from axolotl.monkeypatch.gemma4_hybrid_mask import patch_gemma4_hybrid_mask

    if not hasattr(modeling_gemma4, "create_masks_for_generate"):
        pytest.skip("transformers version lacks create_masks_for_generate")

    sentinel = "single_tensor_passthrough"
    mock_original = MagicMock(return_value=sentinel)
    modeling_gemma4.create_masks_for_generate = mock_original
    patch_gemma4_hybrid_mask()

    caller_config = SimpleNamespace(_attn_implementation="flash_attention_2")
    result = modeling_gemma4.create_masks_for_generate(
        caller_config,
        inputs_embeds=None,
        attention_mask=None,
        past_key_values=None,
        position_ids=None,
    )

    assert result is sentinel
    assert mock_original.call_count == 1


def test_create_masks_for_generate_wrapper_passes_through_when_no_full_key(
    restore_gemma4_module,
):
    """No ``full_attention`` key (all-sliding model) → no second call."""
    modeling_gemma4 = restore_gemma4_module
    from axolotl.monkeypatch.gemma4_hybrid_mask import patch_gemma4_hybrid_mask

    if not hasattr(modeling_gemma4, "create_masks_for_generate"):
        pytest.skip("transformers version lacks create_masks_for_generate")

    fa2_only = {"sliding_attention": "fa2_sliding"}
    mock_original = MagicMock(return_value=fa2_only)
    modeling_gemma4.create_masks_for_generate = mock_original
    patch_gemma4_hybrid_mask()

    caller_config = SimpleNamespace(_attn_implementation="flash_attention_2")
    result = modeling_gemma4.create_masks_for_generate(
        caller_config,
        inputs_embeds=None,
        attention_mask=None,
        past_key_values=None,
        position_ids=None,
    )

    assert result == fa2_only
    assert mock_original.call_count == 1


def test_unpatch_restores_create_masks_for_generate(restore_gemma4_module):
    modeling_gemma4 = restore_gemma4_module
    from axolotl.monkeypatch.gemma4_hybrid_mask import (
        patch_gemma4_hybrid_mask,
        unpatch_gemma4_hybrid_mask,
    )

    if not hasattr(modeling_gemma4, "create_masks_for_generate"):
        pytest.skip("transformers version lacks create_masks_for_generate")

    sentinel = MagicMock(name="original_cmfg")
    modeling_gemma4.create_masks_for_generate = sentinel
    patch_gemma4_hybrid_mask()
    assert modeling_gemma4.create_masks_for_generate is not sentinel

    unpatch_gemma4_hybrid_mask()
    assert modeling_gemma4.create_masks_for_generate is sentinel


def test_create_masks_for_generate_wrapper_passes_through_when_sdpa_returns_non_dict(
    restore_gemma4_module,
):
    """Defensive bail-out: SDPA call returning non-dict → keep FA2 output."""
    modeling_gemma4 = restore_gemma4_module
    from axolotl.monkeypatch.gemma4_hybrid_mask import patch_gemma4_hybrid_mask

    if not hasattr(modeling_gemma4, "create_masks_for_generate"):
        pytest.skip("transformers version lacks create_masks_for_generate")

    fa2_dict = {"full_attention": "fa2_full", "sliding_attention": "fa2_sliding"}

    def fake_original(config, *args, **kwargs):
        if config._attn_implementation == "sdpa":
            return "unexpected_non_dict_return"
        return fa2_dict

    mock_original = MagicMock(side_effect=fake_original)
    modeling_gemma4.create_masks_for_generate = mock_original
    patch_gemma4_hybrid_mask()

    caller_config = SimpleNamespace(_attn_implementation="flash_attention_2")
    result = modeling_gemma4.create_masks_for_generate(
        caller_config,
        inputs_embeds=None,
        attention_mask=None,
        past_key_values=None,
        position_ids=None,
    )

    assert result == fa2_dict


# ---------------------------------------------------------------------------
# Patch lifecycle / state-management tests
# ---------------------------------------------------------------------------


def test_patch_skips_symbols_missing_on_older_transformers(restore_gemma4_module):
    """Missing symbol on older transformers → skip wrapper, still return True."""
    modeling_gemma4 = restore_gemma4_module
    from axolotl.monkeypatch import gemma4_hybrid_mask
    from axolotl.monkeypatch.gemma4_hybrid_mask import patch_gemma4_hybrid_mask

    if hasattr(modeling_gemma4, "create_masks_for_generate"):
        delattr(modeling_gemma4, "create_masks_for_generate")

    assert patch_gemma4_hybrid_mask() is True
    assert gemma4_hybrid_mask._is_axolotl_wrapper(modeling_gemma4.create_causal_mask)
    assert not hasattr(modeling_gemma4, "create_masks_for_generate")


def test_patch_returns_false_when_no_symbols_present(restore_gemma4_module):
    """All symbols missing → install reports failure so caller knows fix is inactive."""
    modeling_gemma4 = restore_gemma4_module
    from axolotl.monkeypatch.gemma4_hybrid_mask import patch_gemma4_hybrid_mask

    for name in ("create_causal_mask", "create_masks_for_generate"):
        if hasattr(modeling_gemma4, name):
            delattr(modeling_gemma4, name)

    assert patch_gemma4_hybrid_mask() is False


def test_patch_is_idempotent_when_flag_lost_but_wrapper_already_bound(
    restore_gemma4_module,
):
    """Flag reset but wrappers still bound → re-patch reports success and must
    not re-wrap an existing wrapper (no double-wrapping)."""
    from axolotl.monkeypatch import gemma4_hybrid_mask
    from axolotl.monkeypatch.gemma4_hybrid_mask import patch_gemma4_hybrid_mask

    modeling_gemma4 = restore_gemma4_module
    patch_gemma4_hybrid_mask()
    wrapped = modeling_gemma4.create_causal_mask
    assert gemma4_hybrid_mask._is_axolotl_wrapper(wrapped)

    # Simulate a fixture clearing the flag without restoring bindings.
    gemma4_hybrid_mask._PATCH_APPLIED = False
    assert patch_gemma4_hybrid_mask() is True

    rewrapped = modeling_gemma4.create_causal_mask
    assert rewrapped is wrapped, "re-patch must not wrap an existing wrapper"
    assert rewrapped._axolotl_original is wrapped._axolotl_original


def test_unpatch_leaves_third_party_rebindings_alone(restore_gemma4_module):
    """Downstream replacement of our wrapper must survive unpatch."""
    modeling_gemma4 = restore_gemma4_module
    from axolotl.monkeypatch.gemma4_hybrid_mask import (
        patch_gemma4_hybrid_mask,
        unpatch_gemma4_hybrid_mask,
    )

    patch_gemma4_hybrid_mask()
    third_party = MagicMock(name="downstream_replacement")
    modeling_gemma4.create_causal_mask = third_party

    unpatch_gemma4_hybrid_mask()

    assert modeling_gemma4.create_causal_mask is third_party


# ---------------------------------------------------------------------------
# Integration tests: tiny Gemma4TextModel (sliding + full_attention layers),
# end-to-end forward with padded mask, no real weights / CUDA needed.
# ---------------------------------------------------------------------------


def _build_tiny_gemma4_text_model():
    """Tiny randomly-initialized Gemma4TextModel with mixed layer types."""
    import torch
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextModel

    cfg = Gemma4TextConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=32,
        layer_types=["sliding_attention", "full_attention"],
        sliding_window=64,
        max_position_embeddings=2048,
        hidden_size_per_layer_input=16,
        vocab_size_per_layer_input=128,
    )
    cfg._attn_implementation = "sdpa"
    torch.manual_seed(42)
    model = Gemma4TextModel(cfg).eval()
    return model, cfg


def _apply_hybrid_attn_inline(model, cfg):
    """Replicate ``patch_manager._apply_gemma_hybrid_attention`` for tests."""
    import copy

    from axolotl.monkeypatch.gemma4_hybrid_mask import patch_gemma4_hybrid_mask

    for layer_idx, layer in enumerate(model.layers):
        if cfg.layer_types[layer_idx] != "sliding_attention":
            attn = getattr(layer, "self_attn", None)
            if attn is not None and hasattr(attn, "config"):
                sdpa_cfg = copy.copy(attn.config)
                sdpa_cfg._attn_implementation = "sdpa"
                attn.config = sdpa_cfg
    patch_gemma4_hybrid_mask()


def test_tiny_gemma4_long_context_forward_does_not_crash(restore_gemma4_module):
    """End-to-end: tiny model + hybrid patch survives padded forward at S=1024."""
    import torch

    model, cfg = _build_tiny_gemma4_text_model()
    _apply_hybrid_attn_inline(model, cfg)

    B, S = 2, 1024
    input_ids = torch.randint(0, cfg.vocab_size, (B, S))
    attn_mask = torch.ones(B, S, dtype=torch.long)
    # Padding required: without it SDPA short-circuits to is_causal=True
    # with mask=None and the materialized-4D-mask path goes unexercised.
    attn_mask[1, S // 2 :] = 0

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attn_mask)

    assert out.last_hidden_state.shape == (B, S, cfg.hidden_size)
    assert torch.isfinite(out.last_hidden_state).all()


def test_patched_create_causal_mask_returns_4d_for_real_config(
    restore_gemma4_module,
):
    """Real (un-mocked) factory through the wrapper returns a 4D mask
    even when the caller's config says FA2."""
    import torch
    from transformers.cache_utils import DynamicCache
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig

    from axolotl.monkeypatch.gemma4_hybrid_mask import patch_gemma4_hybrid_mask

    patch_gemma4_hybrid_mask()
    modeling_gemma4 = restore_gemma4_module

    cfg = Gemma4TextConfig(
        vocab_size=128,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=32,
        layer_types=["sliding_attention", "full_attention"],
        sliding_window=64,
        max_position_embeddings=2048,
        hidden_size_per_layer_input=16,
        vocab_size_per_layer_input=128,
    )
    cfg._attn_implementation = "flash_attention_2"

    B, S = 2, 1024
    inputs_embeds = torch.zeros((B, S, cfg.hidden_size), dtype=torch.float32)
    attention_mask = torch.ones((B, S), dtype=torch.long)
    attention_mask[1, S // 2 :] = 0
    position_ids = torch.arange(S).unsqueeze(0).expand(B, -1)
    past_key_values = DynamicCache(config=cfg)

    mask = modeling_gemma4.create_causal_mask(
        config=cfg,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        position_ids=position_ids,
    )

    assert mask is not None
    assert isinstance(mask, torch.Tensor)
    assert mask.dim() == 4, (
        f"expected 4D SDPA mask, got {mask.dim()}D shape={tuple(mask.shape)}"
    )
    assert mask.shape[0] == B
    assert mask.shape[-1] == S
    assert mask.shape[-2] == S
    assert cfg._attn_implementation == "flash_attention_2"


def test_patched_create_masks_for_generate_returns_4d_full_attention(
    restore_gemma4_module,
):
    """Real (un-mocked) factory through the wrapper: full_attention entry
    is 4D SDPA, sliding stays 2D FA2. This is the multimodal-CPT failure
    mode entry point."""
    import torch
    from transformers.cache_utils import DynamicCache
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig

    from axolotl.monkeypatch.gemma4_hybrid_mask import patch_gemma4_hybrid_mask

    patch_gemma4_hybrid_mask()
    modeling_gemma4 = restore_gemma4_module

    if not hasattr(modeling_gemma4, "create_masks_for_generate"):
        pytest.skip("transformers version lacks create_masks_for_generate")

    cfg = Gemma4TextConfig(
        vocab_size=128,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=32,
        layer_types=["sliding_attention", "full_attention"],
        sliding_window=64,
        max_position_embeddings=2048,
        hidden_size_per_layer_input=16,
        vocab_size_per_layer_input=128,
    )
    cfg._attn_implementation = "flash_attention_2"

    B, S = 2, 256
    inputs_embeds = torch.zeros((B, S, cfg.hidden_size), dtype=torch.float32)
    attention_mask = torch.ones((B, S), dtype=torch.long)
    attention_mask[1, S // 2 :] = 0
    position_ids = torch.arange(S).unsqueeze(0).expand(B, -1)
    past_key_values = DynamicCache(config=cfg)

    result = modeling_gemma4.create_masks_for_generate(
        config=cfg,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        position_ids=position_ids,
    )

    assert isinstance(result, dict), (
        f"expected a dict, got {type(result).__name__}"
    )
    assert "full_attention" in result
    full_attn_mask = result["full_attention"]
    assert isinstance(full_attn_mask, torch.Tensor)
    assert full_attn_mask.dim() == 4, (
        f"full_attention must be 4D, got {full_attn_mask.dim()}D "
        f"shape={tuple(full_attn_mask.shape)}"
    )

    if "sliding_attention" in result and result["sliding_attention"] is not None:
        sliding = result["sliding_attention"]
        if isinstance(sliding, torch.Tensor):
            assert sliding.dim() == 2, (
                f"sliding_attention must stay 2D FA2, got {sliding.dim()}D "
                f"shape={tuple(sliding.shape)}"
            )

    assert cfg._attn_implementation == "flash_attention_2"


def test_create_causal_mask_mapping_transitive_coverage(restore_gemma4_module):
    """``create_causal_mask_mapping`` calls module-level ``create_causal_mask``
    internally, so our rebind propagates transitively — no explicit wrapper
    needed. If a future transformers version inlines or aliases the call,
    this test fails and signals we need a direct wrapper."""
    import torch
    from transformers.cache_utils import DynamicCache
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig

    from axolotl.monkeypatch.gemma4_hybrid_mask import patch_gemma4_hybrid_mask

    modeling_gemma4 = restore_gemma4_module
    if not hasattr(modeling_gemma4, "create_causal_mask_mapping"):
        pytest.skip("transformers version lacks create_causal_mask_mapping")

    patch_gemma4_hybrid_mask()

    cfg = Gemma4TextConfig(
        vocab_size=128,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=32,
        layer_types=["sliding_attention", "full_attention"],
        sliding_window=64,
        max_position_embeddings=2048,
        hidden_size_per_layer_input=16,
        vocab_size_per_layer_input=128,
    )
    cfg._attn_implementation = "flash_attention_2"

    B, S = 2, 256
    inputs_embeds = torch.zeros((B, S, cfg.hidden_size), dtype=torch.float32)
    attention_mask = torch.ones((B, S), dtype=torch.long)
    attention_mask[1, S // 2 :] = 0
    position_ids = torch.arange(S).unsqueeze(0).expand(B, -1)
    past_key_values = DynamicCache(config=cfg)

    result = modeling_gemma4.create_causal_mask_mapping(
        config=cfg,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        position_ids=position_ids,
    )

    assert isinstance(result, dict)
    assert "full_attention" in result
    full_attn_mask = result["full_attention"]
    assert isinstance(full_attn_mask, torch.Tensor)
    assert full_attn_mask.dim() == 4, (
        f"expected 4D mask, got {full_attn_mask.dim()}D "
        f"shape={tuple(full_attn_mask.shape)}"
    )


def test_tiny_gemma4_consumes_dict_mask_from_create_masks_for_generate(
    restore_gemma4_module,
):
    """End-to-end multimodal contract: wrapper builds dict → text model
    short-circuits its own mask construction and consumes the dict directly.

    Uses ``Gemma4TextModel`` (not full ``Gemma4Model``) because the
    multimodal wrapper requires irrelevant audio/vision configs; the
    dict-passthrough invariant is what distinguishes patched/unpatched.
    All-SDPA at model level so this runs on CPU; the FA2-at-model-level
    shape correctness is covered by the unit-level test above."""
    import torch

    modeling_gemma4 = restore_gemma4_module
    if not hasattr(modeling_gemma4, "create_masks_for_generate"):
        pytest.skip("transformers version lacks create_masks_for_generate")

    model, cfg = _build_tiny_gemma4_text_model()
    _apply_hybrid_attn_inline(model, cfg)

    B, S = 2, 1024
    input_ids = torch.randint(0, cfg.vocab_size, (B, S))
    attn_mask_2d = torch.ones(B, S, dtype=torch.long)
    attn_mask_2d[1, S // 2 :] = 0

    inputs_embeds = model.embed_tokens(input_ids).detach()
    position_ids = torch.arange(S).unsqueeze(0).expand(B, -1)
    causal_mask_mapping = modeling_gemma4.create_masks_for_generate(
        cfg,
        inputs_embeds,
        attn_mask_2d,
        None,
        position_ids,
    )
    assert isinstance(causal_mask_mapping, dict)
    assert "full_attention" in causal_mask_mapping
    assert causal_mask_mapping["full_attention"].dim() == 4

    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            attention_mask=causal_mask_mapping,
            position_ids=position_ids,
        )

    assert out.last_hidden_state.shape == (B, S, cfg.hidden_size)
    assert torch.isfinite(out.last_hidden_state).all()


# ---------------------------------------------------------------------------
# Integration tests: tiny multimodal Gemma4Model (vision/audio towers None).
# ---------------------------------------------------------------------------


def _build_tiny_gemma4_multimodal_model():
    """Tiny multimodal ``Gemma4Model`` with vision/audio towers disabled
    (a valid Gemma4Config). Exercises the real ``Gemma4Model.forward``."""
    import torch
    from transformers.models.gemma4.configuration_gemma4 import (
        Gemma4Config,
        Gemma4TextConfig,
    )
    from transformers.models.gemma4.modeling_gemma4 import Gemma4Model

    text_cfg = Gemma4TextConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=32,
        layer_types=["sliding_attention", "full_attention"],
        sliding_window=64,
        max_position_embeddings=2048,
        hidden_size_per_layer_input=16,
        vocab_size_per_layer_input=128,
    )
    text_cfg._attn_implementation = "sdpa"
    cfg = Gemma4Config(
        text_config=text_cfg,
        vision_config=None,
        audio_config=None,
    )
    cfg._attn_implementation = "sdpa"
    torch.manual_seed(43)
    model = Gemma4Model(cfg).eval()
    return model, cfg


def _apply_hybrid_attn_inline_multimodal(model, cfg):
    """Mirror ``patch_manager._apply_gemma_hybrid_attention`` for the
    multimodal model — decoder layers live under ``language_model``."""
    import copy

    from axolotl.monkeypatch.gemma4_hybrid_mask import patch_gemma4_hybrid_mask

    layer_types = cfg.text_config.layer_types
    for layer_idx, layer in enumerate(model.language_model.layers):
        if layer_types[layer_idx] != "sliding_attention":
            attn = getattr(layer, "self_attn", None)
            if attn is not None and hasattr(attn, "config"):
                sdpa_cfg = copy.copy(attn.config)
                sdpa_cfg._attn_implementation = "sdpa"
                attn.config = sdpa_cfg
    patch_gemma4_hybrid_mask()


def test_tiny_gemma4_multimodal_forward_with_padding(restore_gemma4_module):
    """End-to-end multimodal-CPT bug fix: real Gemma4Model, hybrid attn,
    mbs=2 padded forward survives. Runs in SDPA on CPU; the FA2-at-model-
    level shape correctness is covered by the unit test above."""
    import torch

    model, cfg = _build_tiny_gemma4_multimodal_model()
    _apply_hybrid_attn_inline_multimodal(model, cfg)

    B, S = 2, 1024
    input_ids = torch.randint(0, cfg.text_config.vocab_size, (B, S))
    attn_mask = torch.ones(B, S, dtype=torch.long)
    attn_mask[1, S // 2 :] = 0

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attn_mask)

    assert out.last_hidden_state.shape == (B, S, cfg.text_config.hidden_size)
    assert torch.isfinite(out.last_hidden_state).all()


def test_tiny_gemma4_multimodal_forward_unpatched_baseline_works(
    restore_gemma4_module,
):
    """Baseline sanity: all-SDPA multimodal forward must run unpatched too,
    proving the surrounding tests don't get false signal from a broken
    test rig."""
    import torch

    model, cfg = _build_tiny_gemma4_multimodal_model()

    B, S = 2, 256
    input_ids = torch.randint(0, cfg.text_config.vocab_size, (B, S))
    attn_mask = torch.ones(B, S, dtype=torch.long)
    attn_mask[1, S // 2 :] = 0

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attn_mask)

    assert out.last_hidden_state.shape == (B, S, cfg.text_config.hidden_size)
    assert torch.isfinite(out.last_hidden_state).all()


@pytest.fixture
def restore_gemma4_hybrid_mask_both():
    """Snapshot ``create_causal_mask`` in BOTH the gemma4 and gemma4_unified
    namespaces (``patch_gemma4_hybrid_mask`` wraps each independently) and reset
    the module flag so the patch re-installs cleanly."""
    modeling_unified = pytest.importorskip(
        "transformers.models.gemma4_unified.modeling_gemma4_unified",
        reason="unified namespace coverage requires gemma4_unified",
    )
    from transformers.models.gemma4 import modeling_gemma4

    from axolotl.monkeypatch import gemma4_hybrid_mask

    saved_gemma4 = modeling_gemma4.create_causal_mask
    saved_unified = modeling_unified.create_causal_mask
    gemma4_hybrid_mask._PATCH_APPLIED = False
    try:
        yield modeling_unified
    finally:
        modeling_gemma4.create_causal_mask = saved_gemma4
        modeling_unified.create_causal_mask = saved_unified
        gemma4_hybrid_mask._PATCH_APPLIED = False


def test_patch_covers_unified_namespace(restore_gemma4_hybrid_mask_both):
    """The unified backbone redefines ``create_causal_mask`` in its own module,
    so the patch must wrap it too — not just ``modeling_gemma4``."""
    modeling_unified = restore_gemma4_hybrid_mask_both
    from axolotl.monkeypatch.gemma4_hybrid_mask import patch_gemma4_hybrid_mask

    original = modeling_unified.create_causal_mask
    assert patch_gemma4_hybrid_mask() is True

    assert modeling_unified.create_causal_mask is not original
    assert modeling_unified.create_causal_mask._axolotl_original is original
