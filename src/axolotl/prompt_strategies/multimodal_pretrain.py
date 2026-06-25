from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from datasets import Dataset, IterableDataset
from transformers import PreTrainedTokenizerBase, ProcessorMixin

from axolotl.prompt_tokenizers import DatasetWrappingStrategy
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


class MultiModalPretrainDatasetWrappingStrategy(DatasetWrappingStrategy):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        processor: ProcessorMixin,
        sequence_len: int,
        text_column: str = "text",
        image_column: str = "images",
        image_token: str | None = None,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.sequence_len = sequence_len
        self.text_column = text_column
        self.image_column = image_column
        self.image_token_spec = build_image_token_spec(processor, override=image_token)

    def _encode_batch(self, examples: dict[str, list]) -> dict[str, list]:
        return encode_multimodal_pretrain(
            examples,
            tokenizer=self.tokenizer,
            max_tokens=self.sequence_len,
            image_token=self.image_token_spec.image_token,
            image_token_id=self.image_token_spec.image_token_id,
            text_column=self.text_column,
            image_column=self.image_column,
            enforce_max_length=False,
        )

    def wrap_dataset(
        self,
        dataset,
        process_count: int | None = None,
        keep_in_memory: bool | None = False,
        **kwargs,
    ) -> Dataset | IterableDataset:
        if isinstance(dataset, Dataset):
            remove_columns = list(dataset.column_names)
        elif getattr(dataset, "features", None):
            remove_columns = list(dataset.features.keys())
        else:
            remove_columns = None

        map_kwargs: dict[str, Any] = {
            "batched": True,
            "remove_columns": remove_columns,
            "desc": "Tokenizing multimodal CPT dataset",
        }
        if isinstance(dataset, Dataset):
            if process_count:
                map_kwargs["num_proc"] = process_count
            if keep_in_memory is not None:
                map_kwargs["keep_in_memory"] = keep_in_memory

        return dataset.map(self._encode_batch, **map_kwargs)


def load(
    tokenizer,
    cfg,
    ds_cfg: Optional[dict[str, Any]] = None,
    processor: ProcessorMixin | None = None,
):
    ds_cfg = ds_cfg or {}
    if processor is None:
        raise ValueError(
            "Multimodal CPT (type: multimodal_pretrain) requires a processor. "
            "Set `processor_type: AutoProcessor` (or the concrete processor "
            "class) in your config."
        )
    check_processor_compatibility(processor)
    processor_tokenizer = getattr(processor, "tokenizer", None)
    if processor_tokenizer is not None and processor_tokenizer is not tokenizer:
        raise ValueError(
            "Multimodal CPT requires `tokenizer` to be `processor.tokenizer` "
            "so image placeholder ids stay aligned during encoding."
        )

    text_column = ds_cfg.get("text_column") or "text"
    image_column = ds_cfg.get("image_column") or "images"
    LOG.info(
        "multimodal CPT dataset path: text_column=%r image_column=%r",
        text_column,
        image_column,
    )
    return MultiModalPretrainDatasetWrappingStrategy(
        tokenizer=tokenizer,
        processor=processor,
        sequence_len=cfg.sequence_len,
        text_column=text_column,
        image_column=image_column,
        image_token=ds_cfg.get("image_token"),
    )


def encode_multimodal_pretrain(
    examples: dict[str, list],
    tokenizer: PreTrainedTokenizerBase,
    max_tokens: int,
    image_token: str,
    image_token_id: int,
    text_column: str = "text",
    image_column: str = "images",
    enforce_max_length: bool = True,
) -> dict[str, list]:
    texts: list[str] = examples[text_column]
    imgs_list: list[list[str]] = examples[image_column]

    if len(texts) != len(imgs_list):
        raise ValueError(
            f"encode_multimodal_pretrain: text column has {len(texts)} rows "
            f"but image column has {len(imgs_list)}"
        )

    input_ids: list[list[int]] = []
    labels: list[list[int]] = []
    attention_mask: list[list[int]] = []
    keep_images: list[list[str]] = []
    keep_text: list[str] = []

    for text, imgs in zip(texts, imgs_list, strict=True):
        if not isinstance(text, str):
            raise TypeError(
                f"encode_multimodal_pretrain: `{text_column}` must be str, "
                f"got {type(text).__name__}."
            )
        if imgs is None:
            imgs = []
        if not isinstance(imgs, (list, tuple)):
            raise ValueError(
                f"encode_multimodal_pretrain: row's `{image_column}` must be "
                f"a list; got {type(imgs).__name__}"
            )
        for j, ip in enumerate(imgs):
            if not isinstance(ip, str):
                raise TypeError(
                    f"encode_multimodal_pretrain: image {j} in row must be "
                    f"str, got {type(ip).__name__}."
                )
        # Avoid truncation before processor re-tokenization.
        enc = tokenizer(text, add_special_tokens=True)
        ids = list(enc["input_ids"]) + [tokenizer.eos_token_id]
        mask = list(enc["attention_mask"]) + [1]
        # Count by id; text.count can match <image> inside <image_soft_token>.
        n_placeholders = sum(1 for t in ids if t == image_token_id)
        if n_placeholders != len(imgs):
            raise ValueError(
                f"Multimodal CPT row has {n_placeholders} occurrence(s) of "
                f"{image_token!r} in text but {len(imgs)} image path(s). "
                f"Text and image count must match (one placeholder per image)."
            )
        if enforce_max_length and len(ids) > max_tokens:
            raise ValueError(
                f"Multimodal CPT row tokenizes to {len(ids)} tokens which "
                f"exceeds sequence_len={max_tokens}. Pre-chunk your text or "
                f"raise sequence_len (image patch expansion at the processor "
                f"may push the final length even higher)."
            )
        # Labels = ids; collator masks image-family ids after re-tokenization.
        input_ids.append(ids)
        labels.append(list(ids))
        attention_mask.append(mask)
        keep_images.append(list(imgs))
        keep_text.append(text)

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "images": keep_images,
        "_mm_text": keep_text,
    }


def _get_incompatible_processor_classes() -> tuple[type, ...]:
    import importlib

    classes: list[type] = []
    for mod_path, name in (
        ("transformers.models.mllama", "MllamaProcessor"),
        ("transformers.models.pixtral", "PixtralProcessor"),
        ("transformers.models.internvl", "InternVLProcessor"),
    ):
        try:
            mod = importlib.import_module(mod_path)
            cls = getattr(mod, name, None)
            if cls is not None:
                classes.append(cls)
        except ImportError:
            continue
    return tuple(classes)


_KNOWN_IMAGE_TOKEN_CANDIDATES: tuple[str, ...] = (
    "<image>",
    "<|image|>",
    "<|image_pad|>",
    "<image_soft_token>",
    "<start_of_image>",
    "[IMG]",
    "<IMG_CONTEXT>",
)

# Without masking these in labels, loss blows up ~10x on Qwen/SmolVLM.
_IMAGE_FAMILY_TOKEN_CANDIDATES: tuple[str, ...] = (
    "<image>",
    "<|image|>",
    "<|image_pad|>",
    "<image_soft_token>",
    "<start_of_image>",
    "<end_of_image>",
    "<|vision_start|>",
    "<|vision_end|>",
    "[IMG]",
    "[IMG_END]",
    "<IMG_CONTEXT>",
)

_INCOMPATIBLE_PROCESSOR_REASONS: dict[str, str] = {
    "MllamaProcessor": (
        "Llama-3.2-Vision (Mllama) uses cross-attention image injection, not "
        "in-stream placeholder tokens. Multimodal CPT is incompatible with "
        "this architecture; use chat-template SFT instead."
    ),
    "PixtralProcessor": (
        "Pixtral's tokenizer goes through mistral_common with a different "
        "API surface than AutoProcessor. Multimodal CPT not supported in v1; "
        "use chat-template SFT or Mistral-Small-3.1."
    ),
    "InternVLProcessor": (
        "InternVL ships a custom processing pipeline (AutoProcessor returns "
        "text-only); no pixel_values are produced. Multimodal CPT not "
        "supported in v1."
    ),
}
_INCOMPATIBLE_PROCESSOR_CLASSES = _get_incompatible_processor_classes()


@dataclass
class ImageTokenSpec:
    image_token: str
    image_token_id: int
    image_family_token_ids: set[int]


def build_image_token_spec(
    processor: ProcessorMixin, override: str | None = None
) -> ImageTokenSpec:
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise ValueError(
            "Processor has no `tokenizer` attribute — multimodal CPT "
            "requires a processor with a text tokenizer (e.g. one produced "
            "by AutoProcessor.from_pretrained for a VLM)."
        )

    def resolve_id(tok: str) -> int | None:
        tid = tokenizer.convert_tokens_to_ids(tok)
        unk = getattr(tokenizer, "unk_token_id", None)
        if tid is None or tid == unk:
            return None
        return tid

    known_special_tokens: set[str] = set()
    try:
        known_special_tokens |= set(tokenizer.get_added_vocab().keys())
    except Exception as exc:  # noqa: BLE001
        LOG.debug(
            "tokenizer.get_added_vocab() failed on %s: %s",
            type(tokenizer).__name__,
            exc,
        )
    known_special_tokens |= set(getattr(tokenizer, "all_special_tokens", None) or [])
    known_special_tokens |= set(
        getattr(tokenizer, "additional_special_tokens", None) or []
    )

    image_token: str | None = None
    image_token_id: int | None = None
    if override is not None:
        # Reject plain words that BPE-tokenize cleanly but aren't placeholders.
        if override not in known_special_tokens:
            raise ValueError(
                f"image_token override {override!r} is not a registered "
                f"special token on this tokenizer. Pick one of the model's "
                f"actual image tokens (e.g. '<image>', '<|image_pad|>', "
                f"'<start_of_image>'), or leave unset to autodetect."
            )
        image_token_id = resolve_id(override)
        if image_token_id is None:
            raise ValueError(
                f"image_token override {override!r} did not resolve to a "
                f"token id (unk). Remove the override to autodetect."
            )
        image_token = override
    else:
        proc_token = getattr(processor, "image_token", None)
        # Gemma-3-style only: `image_token` is the post-expansion soft token
        # (its name literally contains "soft_token"); the user-facing
        # placeholder is `boi_token`. Gemma-4 reverses this — `image_token`
        # IS the user-facing placeholder (`<|image|>`) and `boi_token`
        # (`<|image>`) is just a bracket marker, so don't blindly swap.
        boi_token = getattr(processor, "boi_token", None)
        if (
            boi_token
            and proc_token
            and boi_token != proc_token
            and boi_token in known_special_tokens
            and "soft_token" in proc_token
        ):
            proc_token = boi_token
        if proc_token is not None:
            image_token_id = resolve_id(proc_token)
            if image_token_id is not None:
                image_token = proc_token
        if image_token is None:
            for cand in _KNOWN_IMAGE_TOKEN_CANDIDATES:
                tid = resolve_id(cand)
                if tid is not None:
                    image_token = cand
                    image_token_id = tid
                    break
        if image_token is None:
            raise ValueError(
                "Could not autodetect the image placeholder token for this "
                "processor. Set `image_token: <token>` in the dataset config "
                "(e.g. '<image>' for LLaVA, '<|image_pad|>' for Qwen-VL, "
                "'<start_of_image>' for Gemma-3)."
            )

    # Filter to registered tokens so BPE-fallback ids don't get masked.
    family: set[int] = {image_token_id}  # type: ignore[arg-type]
    for cand in _IMAGE_FAMILY_TOKEN_CANDIDATES:
        if cand != image_token and cand not in known_special_tokens:
            continue
        tid = resolve_id(cand)
        if tid is not None:
            family.add(tid)
    return ImageTokenSpec(
        image_token=image_token,
        image_token_id=image_token_id,  # type: ignore[arg-type]
        image_family_token_ids=family,
    )


def check_processor_compatibility(processor: ProcessorMixin) -> None:
    if _INCOMPATIBLE_PROCESSOR_CLASSES and isinstance(
        processor, _INCOMPATIBLE_PROCESSOR_CLASSES
    ):
        for cls in _INCOMPATIBLE_PROCESSOR_CLASSES:
            if isinstance(processor, cls):
                raise ValueError(
                    f"Multimodal CPT is not supported for {cls.__name__}: "
                    f"{_INCOMPATIBLE_PROCESSOR_REASONS.get(cls.__name__, '')}"
                )
    # MRO-name fallback for test fakes and unimportable concrete classes.
    for base_cls in type(processor).__mro__:
        reason = _INCOMPATIBLE_PROCESSOR_REASONS.get(base_cls.__name__)
        if reason is not None:
            raise ValueError(
                f"Multimodal CPT is not supported for {base_cls.__name__}: {reason}"
            )
