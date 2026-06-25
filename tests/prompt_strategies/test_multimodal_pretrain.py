from __future__ import annotations

import pytest
from datasets import Dataset
from transformers import AutoProcessor

from axolotl.prompt_strategies.multimodal_pretrain import (
    _INCOMPATIBLE_PROCESSOR_REASONS,
    ImageTokenSpec,
    MultiModalPretrainDatasetWrappingStrategy,
    build_image_token_spec,
    check_processor_compatibility,
    load,
)
from axolotl.utils.data.utils import handle_long_seq_in_dataset
from axolotl.utils.dict import DictDefault

from tests.hf_offline_utils import enable_hf_offline

_SMOLVLM = "HuggingFaceTB/SmolVLM-500M-Instruct"


class _StubTokenizer:
    eos_token_id = 2
    pad_token_id = 0
    unk_token_id = 1
    all_special_tokens = ["<image>"]
    additional_special_tokens = ["<image>"]
    name_or_path = "stub-tokenizer"

    def get_added_vocab(self):
        return {"<image>": 42}

    def convert_tokens_to_ids(self, tok):
        return {"<image>": 42}.get(tok, self.unk_token_id)

    def __call__(self, text, add_special_tokens=True):
        ids = []
        for token in text.split():
            ids.append(42 if token == "<image>" else 100 + len(token))
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


class _StubProcessor:
    image_token = "<image>"

    def __init__(self):
        self.tokenizer = _StubTokenizer()


@pytest.fixture(scope="module", name="smolvlm_processor")
@enable_hf_offline
def fixture_smolvlm_processor(
    download_smolvlm_500m_instruct_model,  # pylint: disable=unused-argument
):
    return AutoProcessor.from_pretrained(_SMOLVLM)


def test_build_image_token_spec_autodetects_smolvlm(smolvlm_processor):
    spec = build_image_token_spec(smolvlm_processor)
    assert isinstance(spec, ImageTokenSpec)
    assert spec.image_token == "<image>"
    assert spec.image_token_id > 0
    assert spec.image_token_id in spec.image_family_token_ids


def test_build_image_token_spec_honors_override(smolvlm_processor):
    spec = build_image_token_spec(smolvlm_processor, override="<image>")
    assert spec.image_token == "<image>"


def test_build_image_token_spec_rejects_bad_override(smolvlm_processor):
    with pytest.raises(ValueError, match="not a registered special token"):
        build_image_token_spec(smolvlm_processor, override="<definitely-not-real>")


def test_build_image_token_spec_rejects_plain_word_override(smolvlm_processor):
    # Plain words BPE-tokenize but aren't placeholders.
    with pytest.raises(ValueError, match="not a registered special token"):
        build_image_token_spec(smolvlm_processor, override="image")


def test_build_image_token_spec_keeps_image_token_when_no_soft_token_in_name(
    smolvlm_processor,
):
    tok = smolvlm_processor.tokenizer
    image_id = tok.convert_tokens_to_ids("<image>")
    boi_id = tok.convert_tokens_to_ids("<fake_token_around_image>")
    assert boi_id != image_id, (
        "fixture assumption broken: SmolVLM tokenizer should map these to distinct ids"
    )

    class _FakeGemma4Like:
        image_token = "<image>"
        boi_token = "<fake_token_around_image>"
        tokenizer = tok

    spec = build_image_token_spec(_FakeGemma4Like())
    assert spec.image_token == "<image>"
    assert spec.image_token_id == image_id
    assert spec.image_token_id != boi_id


@pytest.mark.parametrize("cls_name", list(_INCOMPATIBLE_PROCESSOR_REASONS.keys()))
def test_check_processor_compatibility_rejects_incompatible(cls_name):
    fake = type(cls_name, (), {})()
    with pytest.raises(ValueError) as exc:
        check_processor_compatibility(fake)
    assert cls_name in str(exc.value)
    assert _INCOMPATIBLE_PROCESSOR_REASONS[cls_name] in str(exc.value)


def test_check_processor_compatibility_rejects_subclass():
    # MRO-name fallback must catch user-defined subclasses.
    class BaseMllama:
        pass

    BaseMllama.__name__ = "MllamaProcessor"

    class CustomUserProcessor(BaseMllama):
        pass

    CustomUserProcessor.__name__ = "CustomUserProcessor"

    with pytest.raises(ValueError, match="MllamaProcessor"):
        check_processor_compatibility(CustomUserProcessor())


def test_check_processor_compatibility_accepts_supported(smolvlm_processor):
    check_processor_compatibility(smolvlm_processor)


def test_load_returns_nonstreaming_dataset_strategy():
    processor = _StubProcessor()
    strategy = load(
        processor.tokenizer,
        DictDefault({"sequence_len": 128}),
        ds_cfg={"text_column": "caption", "image_column": "image_paths"},
        processor=processor,
    )
    assert isinstance(strategy, MultiModalPretrainDatasetWrappingStrategy)
    assert strategy.text_column == "caption"
    assert strategy.image_column == "image_paths"


def test_load_requires_processor_for_nonstreaming_strategy():
    tokenizer = _StubTokenizer()
    with pytest.raises(ValueError, match="requires a processor"):
        load(tokenizer, DictDefault({"sequence_len": 128}), ds_cfg={}, processor=None)


def test_load_rejects_processor_tokenizer_mismatch():
    processor = _StubProcessor()
    with pytest.raises(ValueError, match=r"processor\.tokenizer"):
        load(_StubTokenizer(), DictDefault({"sequence_len": 128}), processor=processor)


def test_nonstreaming_strategy_wraps_dataset_without_loading_pixels():
    processor = _StubProcessor()
    strategy = load(
        processor.tokenizer,
        DictDefault({"sequence_len": 128}),
        ds_cfg={"text_column": "caption", "image_column": "image_paths"},
        processor=processor,
    )
    dataset = Dataset.from_dict(
        {
            "caption": ["<image>\nfirst row", "text only row"],
            "image_paths": [["relative/a.png"], []],
            "metadata": ["dropped", "dropped"],
        }
    )

    wrapped = strategy.wrap_dataset(dataset, process_count=None)

    assert set(wrapped.column_names) == {
        "input_ids",
        "labels",
        "attention_mask",
        "images",
        "_mm_text",
    }
    assert wrapped[0]["images"] == ["relative/a.png"]
    assert wrapped[0]["_mm_text"] == "<image>\nfirst row"
    assert wrapped[1]["images"] == []
    assert wrapped[1]["labels"] == wrapped[1]["input_ids"]


def test_nonstreaming_strategy_defers_oversized_rows_to_standard_handler():
    processor = _StubProcessor()
    strategy = load(
        processor.tokenizer,
        DictDefault({"sequence_len": 2}),
        ds_cfg={"text_column": "caption", "image_column": "image_paths"},
        processor=processor,
    )
    dataset = Dataset.from_dict(
        {
            "caption": ["<image> too-long"],
            "image_paths": [["relative/a.png"]],
        }
    )

    wrapped = strategy.wrap_dataset(dataset, process_count=None)

    assert len(wrapped[0]["input_ids"]) > 2


def test_standard_length_handler_drops_nonstreaming_mm_oversized_rows():
    processor = _StubProcessor()
    strategy = load(
        processor.tokenizer,
        DictDefault({"sequence_len": 2}),
        ds_cfg={"text_column": "caption", "image_column": "image_paths"},
        processor=processor,
    )
    dataset = Dataset.from_dict(
        {
            "caption": ["<image> too-long"],
            "image_paths": [["relative/a.png"]],
        }
    )

    wrapped = strategy.wrap_dataset(dataset, process_count=None)
    filtered = handle_long_seq_in_dataset(
        wrapped,
        2,
        DictDefault(
            {
                "dataset_num_proc": None,
                "is_preprocess": False,
                "excess_length_strategy": "drop",
                "min_sample_len": 1,
            }
        ),
    )

    assert len(filtered) == 0
