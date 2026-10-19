"""Tests for ``assert_supported_peft_transformers_surface``.

Probes the startup-time guard that fails loud at config time if PEFT or
transformers drift away from the API surface ProTrain depends on
(``LoraLayer.adapter_layer_names`` and ``Trainer._load_from_checkpoint``).
"""

from __future__ import annotations

import pytest

from axolotl.integrations.protrain.check import (
    _trainer_load_from_checkpoint_signature_error,
    assert_supported_peft_transformers_surface,
)


def test_assert_passes_with_installed_versions() -> None:
    """Sanity: installed peft / transformers expose the expected surface."""
    assert_supported_peft_transformers_surface()


def test_assert_raises_on_missing_lora_layer_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Removing ``LoraLayer.adapter_layer_names`` must surface as RuntimeError."""
    from peft.tuners.lora import LoraLayer

    # The attribute is also defined on BaseTunerLayer (LoraLayer's MRO parent);
    # delete from every class in the MRO that owns it so hasattr returns False.
    for cls in LoraLayer.__mro__:
        if "adapter_layer_names" in cls.__dict__:
            monkeypatch.delattr(cls, "adapter_layer_names", raising=True)

    with pytest.raises(RuntimeError) as excinfo:
        assert_supported_peft_transformers_surface()

    msg = str(excinfo.value)
    assert "peft.tuners.lora.LoraLayer.adapter_layer_names" in msg
    assert "Validated upper bounds" in msg
    assert "peft=" in msg
    assert "transformers=" in msg


def test_assert_raises_on_missing_trainer_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Removing ``Trainer._load_from_checkpoint`` must surface as RuntimeError."""
    from transformers import Trainer

    monkeypatch.delattr(Trainer, "_load_from_checkpoint", raising=True)

    with pytest.raises(RuntimeError) as excinfo:
        assert_supported_peft_transformers_surface()

    msg = str(excinfo.value)
    assert "transformers.Trainer._load_from_checkpoint" in msg
    assert "Validated upper bounds" in msg


def test_trainer_load_signature_helper_accepts_expected_surface() -> None:
    """The supported HF Trainer resume surface must pass the signature guard."""

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        return None

    assert _trainer_load_from_checkpoint_signature_error(_load_from_checkpoint) is None


def test_assert_raises_on_incompatible_trainer_load_signature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Renaming the checkpoint argument must fail before training starts."""
    from transformers import Trainer

    def _load_from_checkpoint(self, checkpoint, model=None):
        return None

    monkeypatch.setattr(Trainer, "_load_from_checkpoint", _load_from_checkpoint)

    with pytest.raises(RuntimeError) as excinfo:
        assert_supported_peft_transformers_surface()

    msg = str(excinfo.value)
    assert "resume_from_checkpoint" in msg
    assert "signature is incompatible" in msg
    assert "transformers=" in msg
    assert "Validated upper bounds" in msg


def test_trainer_load_signature_helper_rejects_required_model() -> None:
    """The wrapper can only forward model safely when HF keeps it optional."""

    def _load_from_checkpoint(self, resume_from_checkpoint, model):
        return None

    msg = _trainer_load_from_checkpoint_signature_error(_load_from_checkpoint)

    assert msg is not None
    assert "model" in msg
    assert "optional" in msg
