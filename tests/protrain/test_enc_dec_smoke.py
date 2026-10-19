"""T5 encoder-decoder E2E smoke test for ProTrain — Item 9 cell B.

Item 8's ``batch_factory`` adds a ``seq2seq_lm`` factory and is covered
by ``test_batch_factory.py`` for shape contracts and CPU-only
forward+backward; this test drives a real encoder-decoder model
end-to-end through ``protrain_model_wrapper``.

Encoder-decoder support landed via ``discover_blocks``'s
``BlockTree`` return type:

- ``encoder.block`` and ``decoder.block`` are first-class dotted-path
  pairs in ``layout_rules._ENC_DEC_PATH_PAIRS``.
- ``discover_blocks`` returns ``list[BlockTree]`` — two entries for T5
  (encoder forward_order=0, decoder forward_order=1), one entry for
  causal-LM models. Consumers concatenate via ``flatten_block_trees``
  to recover the global block-id space.
- ``_looks_like_block`` recurses one level into ``T5Block.layer`` so
  the fallback heuristic also recognises T5-style nested attention
  modules.

The pre-flight check in this test still inspects ``discover_blocks``'s
output: it now succeeds on T5 and the test falls through to the full
wrap + 3-iter forward/backward/step path on the GPU.
"""

from __future__ import annotations

import math

import pytest


def _build_tiny_t5():
    """Construct a fresh-init tiny T5 — same shape as in test_batch_factory.

    Module-local helper so the skip path below can still import its
    way to the model when the discover_blocks check is being exercised.
    """
    from transformers import T5Config, T5ForConditionalGeneration

    cfg = T5Config(
        d_model=128,
        num_layers=2,
        num_decoder_layers=2,
        num_heads=4,
        d_ff=256,
        d_kv=32,
        vocab_size=128,
        decoder_start_token_id=0,
        pad_token_id=0,
    )
    return cfg, T5ForConditionalGeneration(cfg)


def test_protrain_enc_dec_smoke_t5() -> None:
    """T5-small enc-dec smoke: wrap + 3 iters; assert finite losses.

    Sequence:

    1. ``discover_blocks`` returns two ``BlockTree`` entries for T5
       (encoder forward_order=0, decoder forward_order=1). Both must
       be non-empty.
    2. ``protrain_model_wrapper`` wraps the model with Mode-A
       (force_all_persistent), then 3 forward+backward+step iters run
       on a fixed batch with finite loss assertions.
    """
    pytest.importorskip("torch")
    pytest.importorskip("transformers")

    import torch

    if not torch.cuda.is_available():
        pytest.skip("ProTrain enc-dec smoke requires CUDA.")

    from axolotl.integrations.protrain.block.layout_rules import (
        BlockTree,
        discover_blocks,
        flatten_block_trees,
    )
    from axolotl.integrations.protrain.profiler.batch_factory import (
        TASK_SEQ2SEQ_LM,
        detect_task_type,
    )

    cfg, model = _build_tiny_t5()

    # batch_factory must already classify this as seq2seq — that's the
    # part Item 8 covers and we re-assert it here so this test fails
    # loudly if a future refactor breaks task detection on T5.
    assert detect_task_type(model) == TASK_SEQ2SEQ_LM, (
        "T5ForConditionalGeneration must be detected as seq2seq_lm — "
        "the batch_factory path depends on it."
    )

    # discover_blocks now returns one BlockTree per transformer tree.
    # T5 surfaces two: encoder (forward_order=0) and decoder
    # (forward_order=1). Each BlockTree wraps a non-empty
    # nn.ModuleList of T5Block instances.
    trees = discover_blocks(model)
    assert isinstance(trees, list) and len(trees) == 2, (
        f"T5 should surface 2 BlockTrees (encoder+decoder); got {trees}"
    )
    assert all(isinstance(t, BlockTree) for t in trees)
    forward_orders = sorted(t.forward_order for t in trees)
    assert forward_orders == [0, 1], (
        f"T5 BlockTree forward_orders should be [0, 1]; got {forward_orders}"
    )
    flat_blocks = flatten_block_trees(trees)
    assert len(flat_blocks) == len(model.encoder.block) + len(model.decoder.block), (
        "flatten_block_trees should concatenate encoder + decoder blocks"
    )

    from axolotl.integrations.protrain.api import (
        protrain_model_wrapper,
        protrain_optimizer_wrapper,
    )
    from axolotl.integrations.protrain.types import HardwareProfile

    cfg.use_cache = False
    device = torch.device("cuda:0")
    model = model.to(device).to(dtype=torch.bfloat16)

    hw = HardwareProfile(
        gpu_sku=torch.cuda.get_device_name(0),
        gpu_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
        gpu_count=1,
        pcie_h2d_bps=13e9,
        pcie_d2h_bps=13e9,
        has_nvlink=False,
    )
    bs, seq = 2, 16
    wrapped = protrain_model_wrapper(
        model,
        model_config=cfg,
        hardware_profile=hw,
        batch_size=bs,
        seq_len=seq,
        capacity_bytes=20 * (1 << 30),
        force_all_persistent=True,
    )
    optim = protrain_optimizer_wrapper(wrapped, lr=1e-3)

    vocab = int(getattr(cfg, "vocab_size", 128))
    torch.manual_seed(0)
    input_ids = torch.randint(0, vocab, (bs, seq), device=device, dtype=torch.long)
    attention_mask = torch.ones((bs, seq), device=device, dtype=torch.long)
    labels = torch.randint(0, vocab, (bs, seq), device=device, dtype=torch.long)

    losses: list[float] = []
    for i in range(3):
        out = wrapped.module(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss_value = float(out.loss.detach())
        assert math.isfinite(loss_value), f"iter {i}: non-finite loss {loss_value}"
        out.loss.backward()
        optim.step()
        optim.zero_grad()
        losses.append(loss_value)

    print(f"\nProTrain enc-dec smoke (T5-tiny): losses={losses}")
