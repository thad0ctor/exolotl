"""Adapter-aware profiler on-demand engagement tests."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from axolotl.integrations.protrain.profiler.trace import (
    _count_model_state_bytes,
    _model_state_footprint,
    _OnDemandMode,
    _select_on_demand_mode,
)
from axolotl.integrations.protrain.types import ProfilerConfig


class _MetaParamModel(nn.Module):
    def __init__(
        self,
        *,
        frozen_numel: int = 0,
        frozen_dtype: "torch.dtype" = torch.float16,
        trainable_numel: int = 0,
        trainable_dtype: "torch.dtype" = torch.float16,
    ) -> None:
        super().__init__()
        if frozen_numel:
            self.frozen_base = nn.Parameter(
                torch.empty(frozen_numel, device="meta", dtype=frozen_dtype),
                requires_grad=False,
            )
        if trainable_numel:
            self.adapter = nn.Parameter(
                torch.empty(trainable_numel, device="meta", dtype=trainable_dtype),
                requires_grad=True,
            )


def _cfg(
    *,
    adapter: str | None = None,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
) -> ProfilerConfig:
    return ProfilerConfig(
        batch_size=1,
        seq_len=1,
        device="cuda:0",
        adapter=adapter,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
    )


@pytest.mark.parametrize(
    ("cfg", "frozen_dtype"),
    [
        (_cfg(adapter="qlora", load_in_4bit=True), torch.uint8),
        (_cfg(adapter="lora", load_in_8bit=True), torch.uint8),
        (_cfg(adapter="lora"), torch.float16),
    ],
)
def test_adapter_training_uses_activation_only_when_frozen_base_exceeds_gate(
    cfg: ProfilerConfig, frozen_dtype: "torch.dtype"
) -> None:
    model = _MetaParamModel(
        frozen_numel=1000,
        frozen_dtype=frozen_dtype,
        trainable_numel=1,
    )
    footprint = _model_state_footprint(model)

    assert footprint.total_model_state_bytes > 0.60 * 1000
    assert footprint.trainable_training_state_bytes < 0.60 * 1000
    assert _select_on_demand_mode(cfg, footprint, 1000) == _OnDemandMode.ACTIVATION_ONLY


def test_tiny_trainable_fraction_uses_activation_only_without_cfg_adapter() -> None:
    model = _MetaParamModel(
        frozen_numel=1000,
        frozen_dtype=torch.uint8,
        trainable_numel=1,
    )
    footprint = _model_state_footprint(model)

    assert footprint.trainable_param_fraction < 0.05
    assert (
        _select_on_demand_mode(_cfg(), footprint, 1000) == _OnDemandMode.ACTIVATION_ONLY
    )


def test_full_finetune_still_engages_on_demand_when_state_exceeds_gate() -> None:
    model = _MetaParamModel(trainable_numel=1000)
    footprint = _model_state_footprint(model)

    assert footprint.frozen_param_bytes == 0
    assert footprint.total_model_state_bytes > 0.60 * 1000
    assert (
        _select_on_demand_mode(_cfg(), footprint, 1000)
        == _OnDemandMode.PARAM_AND_ACTIVATION
    )


def test_adapter_training_engages_when_trainable_state_exceeds_gate() -> None:
    model = _MetaParamModel(
        frozen_numel=1000,
        frozen_dtype=torch.uint8,
        trainable_numel=100,
    )
    footprint = _model_state_footprint(model)

    assert footprint.trainable_training_state_bytes > 0.60 * 1000
    assert (
        _select_on_demand_mode(_cfg(adapter="lora"), footprint, 1000)
        == _OnDemandMode.PARAM_AND_ACTIVATION
    )


def test_model_state_byte_count_preserves_existing_total_contract() -> None:
    model = _MetaParamModel(
        frozen_numel=1000,
        frozen_dtype=torch.uint8,
        trainable_numel=3,
    )
    footprint = _model_state_footprint(model)

    assert _count_model_state_bytes(model) == footprint.total_model_state_bytes
    assert footprint.frozen_param_bytes == 1000
    assert footprint.trainable_training_state_bytes == 3 * (4 + 12)
