"""Tests for the multimodal pretraining cache gating helpers."""

from types import SimpleNamespace

from axolotl.utils.data.sft import (
    _is_multimodal_pretrain_config,
    _mm_pretrain_cache_enabled,
    _processor_fingerprint,
)
from axolotl.utils.dict import DictDefault


def _mm_entry(**kwargs):
    base = {
        "path": "json",
        "name": None,
        "split": "train",
        "data_files": "/data/train.jsonl",
        "ds_type": "json",
        "type": "multimodal_pretrain",
        "text_column": "text",
        "image_column": "images",
        "image_base_dir": None,
        "image_token": None,
        "multimodal": None,
        "skip": None,
    }
    base.update(kwargs)
    return DictDefault(base)


def _cfg(**kwargs):
    base = {
        "dataset_prepared_path": "/tmp/prepared",
        "skip_prepare_dataset": False,
        "accelerator_config": None,
    }
    base.update(kwargs)
    return DictDefault(base)


class TestIsMultimodalPretrainConfig:
    def test_type_triggers(self):
        assert _is_multimodal_pretrain_config(_mm_entry(type="multimodal_pretrain"))

    def test_multimodal_flag_triggers(self):
        assert _is_multimodal_pretrain_config(
            _mm_entry(type="pretrain", multimodal=True)
        )

    def test_plain_pretrain_does_not_trigger(self):
        assert not _is_multimodal_pretrain_config(_mm_entry(type="pretrain"))


class TestMMPretrainCacheEnabled:
    def test_eval_path_disabled(self):
        assert not _mm_pretrain_cache_enabled(_cfg(), _mm_entry(), is_eval=True)

    def test_no_prepared_path_disabled(self):
        assert not _mm_pretrain_cache_enabled(
            _cfg(dataset_prepared_path=None), _mm_entry(), is_eval=False
        )

    def test_skip_prepare_dataset_disables(self):
        assert not _mm_pretrain_cache_enabled(
            _cfg(skip_prepare_dataset=True), _mm_entry(), is_eval=False
        )

    def test_dispatch_batches_disables(self):
        cfg = _cfg(accelerator_config=DictDefault({"dispatch_batches": True}))
        assert not _mm_pretrain_cache_enabled(cfg, _mm_entry(), is_eval=False)

    def test_non_mm_pretrain_disabled(self):
        assert not _mm_pretrain_cache_enabled(
            _cfg(), _mm_entry(type="pretrain", multimodal=None), is_eval=False
        )

    def test_happy_path_enabled(self):
        assert _mm_pretrain_cache_enabled(_cfg(), _mm_entry(), is_eval=False)

    def test_multimodal_flag_enables(self):
        assert _mm_pretrain_cache_enabled(
            _cfg(),
            _mm_entry(type="pretrain", multimodal=True),
            is_eval=False,
        )


class _FakeProc:
    pass


class TestProcessorFingerprint:
    def test_none_returns_none(self):
        assert _processor_fingerprint(None) is None

    def test_class_name_present(self):
        p = _FakeProc()
        fp = _processor_fingerprint(p)
        assert "_FakeProc" in fp

    def test_image_token_distinguishes(self):
        a = SimpleNamespace(image_token="<image>")
        b = SimpleNamespace(image_token="<|image_pad|>")
        assert _processor_fingerprint(a) != _processor_fingerprint(b)

    def test_boi_token_distinguishes(self):
        a = SimpleNamespace(boi_token="<start_of_image>")
        b = SimpleNamespace(boi_token="<image>")
        assert _processor_fingerprint(a) != _processor_fingerprint(b)

    def test_image_processor_settings_distinguish(self):
        a = SimpleNamespace(
            image_processor=SimpleNamespace(size={"shortest_edge": 336}, patch_size=14)
        )
        b = SimpleNamespace(
            image_processor=SimpleNamespace(size={"shortest_edge": 448}, patch_size=14)
        )
        assert _processor_fingerprint(a) != _processor_fingerprint(b)

    def test_same_class_same_settings_same_fingerprint(self):
        a = SimpleNamespace(image_token="<image>", image_seq_length=256)
        b = SimpleNamespace(image_token="<image>", image_seq_length=256)
        assert _processor_fingerprint(a) == _processor_fingerprint(b)
