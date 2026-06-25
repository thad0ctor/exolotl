"""End-to-end cache build/load tests using a real multimodal processor."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from transformers import AutoProcessor

from axolotl.prompt_strategies.multimodal_pretrain import build_image_token_spec
from axolotl.utils.data.sft import (
    _load_or_build_mm_pretrain_cache,
    _processor_fingerprint,
)
from axolotl.utils.data.shared import (
    generate_pretraining_dataset_hash,
    get_prepared_dataset_path,
)
from axolotl.utils.dict import DictDefault

from tests.hf_offline_utils import enable_hf_offline

_SMOLVLM = "HuggingFaceTB/SmolVLM-500M-Instruct"


@pytest.fixture(scope="module", name="smolvlm_processor")
@enable_hf_offline
def fixture_smolvlm_processor(
    download_smolvlm_500m_instruct_model,  # noqa: ARG001
):
    return AutoProcessor.from_pretrained(_SMOLVLM)


def _write_jsonl(target: Path, image_paths: list[Path], image_token: str) -> Path:
    rows = [
        {"text": f"{image_token}\nrow {i}", "images": [str(p)]}
        for i, p in enumerate(image_paths)
    ]
    target.write_text("\n".join(json.dumps(r) for r in rows))
    return target


def _make_images(directory: Path, n: int) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    out = []
    for i in range(n):
        p = directory / f"img_{i}.png"
        arr = np.random.default_rng(i).integers(0, 255, (32, 32, 3)).astype("uint8")
        Image.fromarray(arr).save(p)
        out.append(p)
    return out


def _entry(jsonl_path: Path) -> DictDefault:
    return DictDefault(
        {
            "path": "json",
            "name": None,
            "split": "train",
            "data_files": str(jsonl_path),
            "ds_type": "json",
            "type": "multimodal_pretrain",
            "text_column": "text",
            "image_column": "images",
            "image_base_dir": None,
            "image_token": None,
            "multimodal": None,
            "skip": None,
            "trust_remote_code": False,
        }
    )


def _cfg(prepared_path: Path) -> DictDefault:
    return DictDefault(
        {
            "sequence_len": 1024,
            "eval_sequence_len": None,
            "tokenizer_config": _SMOLVLM,
            "dataset_prepared_path": str(prepared_path),
            "skip_prepare_dataset": False,
            "accelerator_config": None,
            "shuffle_merged_datasets": False,
            "streaming_multipack_buffer_size": 4,
            "sample_packing": False,
            "dataset_num_proc": None,
            "dataset_keep_in_memory": False,
            "added_tokens_overrides": None,
            "is_preprocess": False,
            "push_dataset_to_hub": None,
            "num_dataset_shards_to_save": None,
            "dataset_exact_deduplication": False,
            "dataset_processes": None,
            "seed": 42,
        }
    )


@pytest.fixture(name="mm_cache_env")
def fixture_mm_cache_env(tmp_path, smolvlm_processor):
    spec = build_image_token_spec(smolvlm_processor)
    images = _make_images(tmp_path / "imgs", n=8)
    jsonl = _write_jsonl(tmp_path / "train.jsonl", images, spec.image_token)
    cfg = _cfg(tmp_path / "prepared")
    entry = _entry(jsonl)
    return DictDefault(
        {
            "tmp": tmp_path,
            "jsonl": jsonl,
            "images": images,
            "cfg": cfg,
            "entry": entry,
            "processor": smolvlm_processor,
            "tokenizer": smolvlm_processor.tokenizer,
        }
    )


class TestCacheBuildAndReload:
    def test_first_call_builds_cache_to_disk(self, mm_cache_env):
        env = mm_cache_env
        ds = _load_or_build_mm_pretrain_cache(
            env.entry, env.cfg, env.tokenizer, env.processor
        )
        assert len(ds) == 8
        assert set(ds.column_names) >= {
            "input_ids",
            "labels",
            "attention_mask",
            "images",
            "_mm_text",
        }
        # Sanity: labels mirror input_ids pre-collator (collator masks image ids later).
        for ids, lbls in zip(ds[0:8]["input_ids"], ds[0:8]["labels"], strict=True):
            assert list(ids) == list(lbls)

        h = generate_pretraining_dataset_hash(
            env.cfg,
            env.entry,
            env.tokenizer.name_or_path,
            _processor_fingerprint(env.processor),
        )
        cache_dir = get_prepared_dataset_path(env.cfg, h)
        assert cache_dir.exists()
        assert any(cache_dir.glob("*.arrow"))

    def test_second_call_hits_cache_no_rebuild(self, mm_cache_env, monkeypatch):
        env = mm_cache_env
        _load_or_build_mm_pretrain_cache(
            env.entry, env.cfg, env.tokenizer, env.processor
        )

        # Sentinel: rebuild should not run on second call.
        from axolotl.utils.data import sft as sft_module

        rebuild_called = {"n": 0}

        original_build = sft_module._build_mm_pretrain_cache

        def spy_build(*args, **kwargs):
            rebuild_called["n"] += 1
            return original_build(*args, **kwargs)

        monkeypatch.setattr(sft_module, "_build_mm_pretrain_cache", spy_build)

        ds2 = _load_or_build_mm_pretrain_cache(
            env.entry, env.cfg, env.tokenizer, env.processor
        )
        assert rebuild_called["n"] == 0
        assert len(ds2) == 8

    def test_changing_data_files_invalidates_cache(self, mm_cache_env):
        env = mm_cache_env
        _load_or_build_mm_pretrain_cache(
            env.entry, env.cfg, env.tokenizer, env.processor
        )

        # New JSONL with different content yields different hash, so new cache dir.
        spec = build_image_token_spec(env.processor)
        more_images = _make_images(env.tmp / "imgs2", n=4)
        new_jsonl = _write_jsonl(
            env.tmp / "train_v2.jsonl", more_images, spec.image_token
        )
        new_entry = _entry(new_jsonl)

        ds_new = _load_or_build_mm_pretrain_cache(
            new_entry, env.cfg, env.tokenizer, env.processor
        )
        assert len(ds_new) == 4

        h1 = generate_pretraining_dataset_hash(
            env.cfg,
            env.entry,
            env.tokenizer.name_or_path,
            _processor_fingerprint(env.processor),
        )
        h2 = generate_pretraining_dataset_hash(
            env.cfg,
            new_entry,
            env.tokenizer.name_or_path,
            _processor_fingerprint(env.processor),
        )
        assert h1 != h2

    def test_with_format_torch_passes_through_mixed_columns(self, mm_cache_env):
        env = mm_cache_env
        ds = _load_or_build_mm_pretrain_cache(
            env.entry, env.cfg, env.tokenizer, env.processor
        )
        torch_ds = ds.with_format("torch")
        row = torch_ds[0]
        import torch

        assert isinstance(row["input_ids"], torch.Tensor)
        assert isinstance(row["labels"], torch.Tensor)
        assert isinstance(row["attention_mask"], torch.Tensor)
        assert isinstance(row["images"], list)
        assert isinstance(row["_mm_text"], str)
