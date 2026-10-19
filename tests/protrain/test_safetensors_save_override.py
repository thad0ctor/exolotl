"""Final-save guards for ProTrain full-FT + offload paths.

ProTrain's shape-preserving expand-placeholder shares scratch storage across
released chunks. safetensors detects this as "shared tensors" and refuses to
save the full model. LoRA-scope saves are unaffected because only adapter
weights are serialized.
"""

from __future__ import annotations

from collections import OrderedDict
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from axolotl.core.trainers.base import AxolotlTrainer
from axolotl.integrations.protrain.block import BlockMode, wrap_block
from axolotl.integrations.protrain.plugin import (
    _force_pickle_save_for_fullft_offload,
    normalize_saved_safetensors_for_save,
    normalize_state_dict_for_save,
    restore_fullft_offload_for_save,
)
from axolotl.train import _restore_protrain_fullft_offload_for_save


def _make_cfg_and_trainer(adapter):
    cfg = SimpleNamespace(adapter=adapter)
    args = SimpleNamespace(save_safetensors=True)
    trainer = SimpleNamespace(args=args)
    return cfg, trainer


def _make_wrapped(non_persistent_ids):
    chunk_manager = SimpleNamespace(_non_persistent_ids=non_persistent_ids)
    return SimpleNamespace(chunk_manager=chunk_manager)


def _make_wrapped_with_effective_offload(non_persistent_ids, n_persist, n_chunk):
    chunk_manager = SimpleNamespace(
        _non_persistent_ids=non_persistent_ids,
        layout=SimpleNamespace(N_chunk=n_chunk),
    )
    search_result = SimpleNamespace(cfg=SimpleNamespace(n_persist=n_persist))
    return SimpleNamespace(chunk_manager=chunk_manager, search_result=search_result)


class _RestoringChunkManager:
    def __init__(self, *, non_persistent_ids=None, restore_value=1234):
        self._non_persistent_ids = non_persistent_ids or {1}
        self.restore_value = restore_value
        self.calls = 0

    def restore_to_gpu(self):
        self.calls += 1
        return self.restore_value


@pytest.mark.parametrize("adapter", [None, ""])
def test_full_ft_with_offload_forces_pickle_save(adapter):
    """Full-FT (no adapter / empty-string adapter) + non-persistent chunks → flipped to False."""
    cfg, trainer = _make_cfg_and_trainer(adapter=adapter)
    wrapped = _make_wrapped(non_persistent_ids={1, 2, 3})

    _force_pickle_save_for_fullft_offload(cfg, trainer, wrapped)

    assert trainer.args.save_safetensors is False


def test_full_ft_with_effective_offload_forces_pickle_after_rebuild():
    """Rebuilt managers may expose effective offload via the picked cfg before ids are repopulated."""
    cfg, trainer = _make_cfg_and_trainer(adapter=None)
    wrapped = _make_wrapped_with_effective_offload(
        non_persistent_ids=set(),
        n_persist=2,
        n_chunk=5,
    )

    _force_pickle_save_for_fullft_offload(cfg, trainer, wrapped)

    assert trainer.args.save_safetensors is False


def test_no_offload_leaves_safetensors_alone():
    """All-persistent (n_persist == N_chunk) → no override."""
    cfg, trainer = _make_cfg_and_trainer(adapter=None)
    wrapped = _make_wrapped(non_persistent_ids=set())

    _force_pickle_save_for_fullft_offload(cfg, trainer, wrapped)

    assert trainer.args.save_safetensors is True


@pytest.mark.parametrize("adapter", ["lora", "qlora"])
def test_lora_scope_leaves_safetensors_alone(adapter):
    """LoRA / qLoRA adapter configured → override does NOT fire."""
    cfg, trainer = _make_cfg_and_trainer(adapter=adapter)
    wrapped = _make_wrapped(non_persistent_ids={1, 2, 3})

    _force_pickle_save_for_fullft_offload(cfg, trainer, wrapped)

    # LoRA-scope save isn't affected by the shared-tensor bug.
    assert trainer.args.save_safetensors is True


def test_existing_false_is_left_alone():
    """User-set save_safetensors=False → no-op (no double-flip, no warning)."""
    cfg, trainer = _make_cfg_and_trainer(adapter=None)
    trainer.args.save_safetensors = False
    wrapped = _make_wrapped(non_persistent_ids={1, 2, 3})

    _force_pickle_save_for_fullft_offload(cfg, trainer, wrapped)

    assert trainer.args.save_safetensors is False


def test_missing_chunk_manager_is_safe():
    """Wrapped object without chunk_manager (defensive) is a no-op."""
    cfg, trainer = _make_cfg_and_trainer(adapter=None)
    wrapped = SimpleNamespace(chunk_manager=None)

    _force_pickle_save_for_fullft_offload(cfg, trainer, wrapped)

    assert trainer.args.save_safetensors is True


def test_missing_trainer_args_is_safe():
    """Trainer without args attribute (test harness scenarios) is a no-op."""
    cfg = SimpleNamespace(adapter=None)
    trainer = SimpleNamespace()
    wrapped = _make_wrapped(non_persistent_ids={1, 2, 3})

    # Should not raise.
    _force_pickle_save_for_fullft_offload(cfg, trainer, wrapped)


@pytest.mark.parametrize("adapter", [None, ""])
def test_full_ft_offload_restore_runs_once_before_final_save(adapter):
    """Full-FT + offload restores params to standalone storage before save."""
    chunk_manager = _RestoringChunkManager()
    wrapped = SimpleNamespace(chunk_manager=chunk_manager)
    cfg = SimpleNamespace(adapter=adapter, _protrain_wrapped=wrapped)

    assert restore_fullft_offload_for_save(cfg) == 1234
    assert chunk_manager.calls == 1

    assert restore_fullft_offload_for_save(cfg) == 0
    assert chunk_manager.calls == 1


def test_full_ft_effective_offload_restore_after_rebuild():
    """Effective offload remains detectable even before ids are repopulated."""
    chunk_manager = _RestoringChunkManager(non_persistent_ids=set())
    chunk_manager.layout = SimpleNamespace(N_chunk=5)
    search_result = SimpleNamespace(cfg=SimpleNamespace(n_persist=2))
    wrapped = SimpleNamespace(chunk_manager=chunk_manager, search_result=search_result)
    cfg = SimpleNamespace(adapter=None, _protrain_wrapped=wrapped)

    assert restore_fullft_offload_for_save(cfg) == 1234
    assert chunk_manager.calls == 1


def test_full_ft_restore_unwraps_block_wrappers_before_final_save():
    """Final model saves should expose native HF keys, not ProTrain wrapper names."""
    chunk_manager = _RestoringChunkManager()
    block = nn.Linear(2, 2)
    model = nn.Sequential(wrap_block(block, BlockMode.CKPT))
    wrapped = SimpleNamespace(chunk_manager=chunk_manager, module=model)
    cfg = SimpleNamespace(adapter=None, _protrain_wrapped=wrapped)

    assert "0.block.weight" in dict(model.named_parameters())

    assert restore_fullft_offload_for_save(cfg) == 1234

    assert model[0] is block
    assert "0.weight" in dict(model.named_parameters())
    assert "0.block.weight" not in dict(model.named_parameters())


def test_full_ft_save_remaps_qwen35_visual_keys_to_native_prefix():
    cfg = SimpleNamespace(_protrain_wrapped=object())
    state_dict = OrderedDict(
        [
            ("model.language_model.visual.blocks.0.norm1.weight", object()),
            ("model.language_model.layers.0.input_layernorm.weight", object()),
        ]
    )

    remapped = normalize_state_dict_for_save(cfg, state_dict)

    assert "model.visual.blocks.0.norm1.weight" in remapped
    assert "model.language_model.visual.blocks.0.norm1.weight" not in remapped
    assert "model.language_model.layers.0.input_layernorm.weight" in remapped


def test_full_ft_save_remaps_inner_qwen35_visual_keys_before_save_pretrained_prefix():
    cfg = SimpleNamespace(_protrain_wrapped=object())
    state_dict = OrderedDict(
        [
            ("language_model.visual.blocks.0.norm1.weight", object()),
            ("language_model.layers.0.input_layernorm.weight", object()),
        ]
    )

    remapped = normalize_state_dict_for_save(cfg, state_dict)

    assert "visual.blocks.0.norm1.weight" in remapped
    assert "language_model.visual.blocks.0.norm1.weight" not in remapped
    assert "language_model.layers.0.input_layernorm.weight" in remapped


def test_full_ft_save_rejects_ambiguous_qwen35_visual_keys():
    cfg = SimpleNamespace(_protrain_wrapped=object())
    state_dict = OrderedDict(
        [
            ("model.visual.blocks.0.norm1.weight", object()),
            ("model.language_model.visual.blocks.0.norm1.weight", object()),
        ]
    )

    with pytest.raises(RuntimeError, match="contains both native Qwen3.5 visual"):
        normalize_state_dict_for_save(cfg, state_dict)


def test_full_ft_save_rewrites_saved_qwen35_visual_safetensors(tmp_path):
    from safetensors import safe_open
    from safetensors.torch import save_file

    cfg = SimpleNamespace(_protrain_wrapped=object())
    path = tmp_path / "model.safetensors"
    save_file(
        {
            "model.language_model.visual.blocks.0.norm1.weight": torch.ones(1),
            "model.language_model.layers.0.input_layernorm.weight": torch.zeros(1),
        },
        path,
    )

    assert normalize_saved_safetensors_for_save(cfg, tmp_path) == 1

    with safe_open(path, framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
    assert "model.visual.blocks.0.norm1.weight" in keys
    assert "model.language_model.visual.blocks.0.norm1.weight" not in keys
    assert "model.language_model.layers.0.input_layernorm.weight" in keys


def test_lora_offload_restore_is_skipped():
    """LoRA-scope save does not include base chunk placeholders."""
    chunk_manager = _RestoringChunkManager()
    wrapped = SimpleNamespace(chunk_manager=chunk_manager)
    cfg = SimpleNamespace(adapter="lora", _protrain_wrapped=wrapped)

    assert restore_fullft_offload_for_save(cfg) == 0
    assert chunk_manager.calls == 0


def test_train_save_hook_barriers_after_restore():
    """All ranks must finish sharded restore before rank 0 serializes."""
    events = []
    chunk_manager = _RestoringChunkManager()
    wrapped = SimpleNamespace(chunk_manager=chunk_manager)
    cfg = SimpleNamespace(adapter=None, _protrain_wrapped=wrapped)
    accelerator = SimpleNamespace(wait_for_everyone=lambda: events.append("barrier"))
    trainer = SimpleNamespace(accelerator=accelerator)

    _restore_protrain_fullft_offload_for_save(cfg, trainer)

    assert chunk_manager.calls == 1
    assert events == ["barrier"]


def test_train_save_hook_barriers_even_when_rank_has_nothing_to_restore():
    """Rank-local restore counts must not decide collective participation."""
    events = []
    chunk_manager = _RestoringChunkManager(restore_value=0)
    wrapped = SimpleNamespace(chunk_manager=chunk_manager)
    cfg = SimpleNamespace(adapter=None, _protrain_wrapped=wrapped)
    accelerator = SimpleNamespace(wait_for_everyone=lambda: events.append("barrier"))
    trainer = SimpleNamespace(accelerator=accelerator)

    _restore_protrain_fullft_offload_for_save(cfg, trainer)

    assert chunk_manager.calls == 1
    assert events == ["barrier"]


def test_trainer_terminal_save_barriers_even_when_rank_has_nothing_to_restore():
    events = []
    chunk_manager = _RestoringChunkManager(restore_value=0)
    cfg = SimpleNamespace(
        adapter=None,
        _protrain_wrapped=SimpleNamespace(chunk_manager=chunk_manager),
    )
    trainer = SimpleNamespace(
        axolotl_cfg=cfg,
        accelerator=SimpleNamespace(wait_for_everyone=lambda: events.append("barrier")),
        _is_protrain_terminal_checkpoint=lambda: True,
    )

    AxolotlTrainer._restore_protrain_fullft_offload_for_terminal_checkpoint(trainer)

    assert chunk_manager.calls == 1
    assert events == ["barrier"]


def test_full_ft_restore_latch_is_scoped_to_wrapped_instance():
    first = _RestoringChunkManager()
    second = _RestoringChunkManager()
    cfg = SimpleNamespace(
        adapter=None,
        _protrain_wrapped=SimpleNamespace(chunk_manager=first),
    )

    assert restore_fullft_offload_for_save(cfg) == 1234
    assert restore_fullft_offload_for_save(cfg) == 0

    cfg._protrain_wrapped = SimpleNamespace(chunk_manager=second)

    assert restore_fullft_offload_for_save(cfg) == 1234
    assert first.calls == 1
    assert second.calls == 1


def test_training_stop_marks_protrain_checkpoint_terminal():
    """Dataset exhaustion can make the final checkpoint arrive before max_steps."""
    trainer = SimpleNamespace(
        axolotl_cfg=SimpleNamespace(_protrain_wrapped=object()),
        control=SimpleNamespace(should_training_stop=True),
        state=SimpleNamespace(global_step=6, max_steps=100),
        args=SimpleNamespace(max_steps=100),
    )

    assert AxolotlTrainer._is_protrain_terminal_checkpoint(trainer) is True
