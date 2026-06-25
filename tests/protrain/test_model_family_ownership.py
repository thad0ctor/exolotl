"""Synthetic model-family ownership guards for ProTrain layout planning."""

from __future__ import annotations

from typing import cast

from torch import nn

from axolotl.integrations.protrain.api.model_wrapper import _build_block_spans
from axolotl.integrations.protrain.chunk.layout import build_layout
from axolotl.integrations.protrain.types import ParamId


def _layout_for(model: nn.Module):
    _blocks, spans = _build_block_spans(model)
    exec_order = [cast(ParamId, name) for name, _ in model.named_parameters()]
    return spans, build_layout(model, exec_order, S_chunk=128, block_spans=spans)


class _TinyMoEBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = nn.Linear(8, 8, bias=False)
        self.experts = nn.ModuleList(
            [nn.Linear(8, 8, bias=False), nn.Linear(8, 8, bias=False)]
        )
        self.gate = nn.Linear(8, 2, bias=False)


class _TinyMoEModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_TinyMoEBlock(), _TinyMoEBlock()])
        self.lm_head = nn.Linear(8, 16, bias=False)


def test_moe_expert_params_are_block_owned_not_mandatory() -> None:
    spans, layout = _layout_for(_TinyMoEModel())
    expert_names = {
        cast(ParamId, name)
        for name, _ in _TinyMoEModel().named_parameters()
        if ".experts." in name
    }

    assert expert_names
    owned = set().union(*(set(params) for params in spans.values()))
    assert expert_names.issubset(owned)

    for name in expert_names:
        cid = layout.param_to_chunk[name]
        assert cid not in layout.mandatory_persistent, (
            f"{name} is inside a discovered transformer block and should be "
            "chunk-managed with that block, not pinned as non-block state"
        )
        assert any(cid in cids for cids in layout.block_to_chunks.values())


class _TinyVisualBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = nn.Linear(8, 8, bias=False)
        self.mlp = nn.Linear(8, 8, bias=False)


class _TinyVLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_TinyMoEBlock()])
        self.visual = nn.Module()
        self.visual.blocks = nn.ModuleList([_TinyVisualBlock()])
        self.visual.merger = nn.Linear(8, 8, bias=False)
        self.mm_projector = nn.Linear(8, 8, bias=False)
        self.multi_modal_projector = nn.Linear(8, 8, bias=False)


def test_vlm_visual_and_projector_params_are_explicitly_pinned() -> None:
    spans, layout = _layout_for(_TinyVLM())
    visual_or_projector_names = {
        cast(ParamId, name)
        for name, _ in _TinyVLM().named_parameters()
        if name.startswith("visual.")
        or name.startswith("mm_projector.")
        or name.startswith("multi_modal_projector.")
    }

    assert visual_or_projector_names
    assert len(spans) == 1
    assert all(
        not any(name in params for params in spans.values())
        for name in visual_or_projector_names
    )

    for name in visual_or_projector_names:
        cid = layout.param_to_chunk[name]
        assert cid in layout.mandatory_persistent, (
            f"{name} is outside the discovered language block tree and must be "
            "mandatory-persistent rather than silently offloaded without hooks"
        )
