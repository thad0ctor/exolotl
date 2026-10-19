"""BERT sequence-classification E2E smoke test for ProTrain — Item 9 cell A.

Item 8's ``batch_factory`` adds a ``seq_classification`` factory and is
covered by ``test_batch_factory.py`` for shape contracts and CPU-only
forward+backward, but no test drives a real seq-cls model end-to-end
through the calibration profiler + ``protrain_model_wrapper`` on GPU.
The factory could subtly mis-shape labels and cache the wrong trace
without anyone noticing.

This test wraps a tiny BERT-shape model (``BertForSequenceClassification``,
2 hidden layers, 128 hidden, 4 heads, 2 labels — random init) with
ProTrain in Mode-A (``force_all_persistent=True``) on a single GPU and
runs three forward+backward+optimizer-step iterations on a fixed
synthetic batch.

Acceptance:

* Profiler trace exists post-wrap (cache hit OR miss path produced one).
* The wrapped model's forward returns finite logits of the expected
  ``(batch_size, num_labels)`` shape.
* Three training iterations complete without exception.
* All losses finite.

Mode-A is chosen because (a) this is a single-GPU smoke and Mode-C
requires a process group, and (b) it exercises the calibration profiler
path that builds the seq-cls batch via ``batch_factory.build_batch``.

Fast lane (no ``slow`` mark) — at this scale the wrap + 3 iters runs in
well under 30s on a single 3090.
"""

from __future__ import annotations

import math

import pytest


def test_protrain_seq_cls_smoke_bert() -> None:
    """Tiny BERT seq-cls: wrap + 3 training iters; finite logits + losses."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")

    import torch

    if not torch.cuda.is_available():
        pytest.skip("ProTrain seq-cls smoke requires CUDA.")

    from transformers import BertConfig, BertForSequenceClassification

    # Random-init tiny BERT — small enough that the profiler's
    # forward-only trace finishes in a few hundred ms on a 3090, but
    # large enough that the chunk pipeline has more than one chunk to
    # gather.
    cfg = BertConfig(
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=256,
        vocab_size=128,
        max_position_embeddings=64,
        num_labels=2,
        type_vocab_size=2,
    )
    # ``BertConfig`` does not expose ``use_cache``; the wrapper's
    # ``cfg.use_cache`` guard is a no-op here.
    model = BertForSequenceClassification(cfg).to(dtype=torch.bfloat16)

    device = torch.device("cuda:0")
    model = model.to(device)

    # Sanity: every param trainable (full FT — no LoRA).
    n_total = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n_trainable == n_total, (
        f"seq-cls smoke expects all params trainable; "
        f"trainable={n_trainable} total={n_total}"
    )

    # ProTrain wrap (Mode-A, single GPU, no offload).
    from axolotl.integrations.protrain.api import (
        protrain_model_wrapper,
        protrain_optimizer_wrapper,
    )
    from axolotl.integrations.protrain.api.model_wrapper import _arch_hash, _sku
    from axolotl.integrations.protrain.profiler.batch_factory import (
        TASK_SEQ_CLASSIFICATION,
        detect_task_type,
    )
    from axolotl.integrations.protrain.profiler.cache import (
        ProfilerCacheKey,
        load_cached_trace,
    )
    from axolotl.integrations.protrain.types import HardwareProfile

    # Pre-flight: detect_task_type must classify this as seq-cls so the
    # batch_factory uses ``seq_classification_batch_factory`` for the
    # profiler's dummy batch. Without this the profiler would fall back
    # to causal-LM and the trace would be useless for the seq-cls head.
    assert detect_task_type(model) == TASK_SEQ_CLASSIFICATION, (
        "BertForSequenceClassification must be detected as seq_classification "
        "for the calibration profiler to build the right dummy batch."
    )

    hw = HardwareProfile(
        gpu_sku=torch.cuda.get_device_name(0),
        gpu_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
        gpu_count=1,
        pcie_h2d_bps=13e9,
        pcie_d2h_bps=13e9,
        has_nvlink=False,
    )

    bs, seq = 2, 32

    # Capture the cache key BEFORE the wrap — ProTrain's CheckpointedBlock
    # wrapper inserts a ``.block.`` infix into ``named_parameters``, which
    # changes ``_arch_hash`` between pre-wrap (the lookup the profiler
    # uses) and post-wrap. Reading the key from post-wrap state would
    # always miss the cache regardless of whether the profiler actually
    # ran.
    pre_wrap_cache_key = ProfilerCacheKey(
        arch_hash=_arch_hash(model),
        bs=bs,
        seq=seq,
        sku=_sku(device),
        world=hw.gpu_count,
    )

    wrapped = protrain_model_wrapper(
        model,
        model_config=cfg,
        hardware_profile=hw,
        batch_size=bs,
        seq_len=seq,
        capacity_bytes=20 * (1 << 30),
        force_all_persistent=True,
    )

    # Acceptance #1: a profiler trace was produced and cached for this
    # model+batch shape. This is the smoke that the profiler ran
    # successfully against a non-causal-LM model.
    trace = load_cached_trace(pre_wrap_cache_key)
    assert trace is not None, (
        f"expected a cached profiler trace under key "
        f"{pre_wrap_cache_key.fingerprint()[:12]} post-wrap; "
        "calibration profiler may not have run for the seq-cls model"
    )
    assert len(trace.op_order) > 0, (
        "profiler trace has no ops — the forward pass against the seq-cls "
        f"batch never recorded anything (got: {trace})"
    )

    optim = protrain_optimizer_wrapper(wrapped, lr=1e-3)

    vocab = int(getattr(cfg, "vocab_size", 128))
    num_labels = int(getattr(cfg, "num_labels", 2))
    torch.manual_seed(0)
    input_ids = torch.randint(0, vocab, (bs, seq), device=device, dtype=torch.long)
    attention_mask = torch.ones((bs, seq), device=device, dtype=torch.long)
    labels = torch.randint(0, num_labels, (bs,), device=device, dtype=torch.long)

    # Acceptance #2: the wrapped model's forward returns finite logits
    # of the expected (B, num_labels) shape — proves the head is wired
    # correctly through ProTrain's hooks for the seq-cls head.
    out0 = wrapped.module(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )
    assert out0.logits.shape == (bs, num_labels), (
        f"expected logits shape ({bs}, {num_labels}); got {tuple(out0.logits.shape)}"
    )
    assert torch.isfinite(out0.logits).all(), (
        f"non-finite logits on first forward: {out0.logits}"
    )

    # Acceptance #3: three training iters complete without exception
    # and all losses are finite.
    losses: list[float] = []
    n_iters = 3
    for i in range(n_iters):
        out = wrapped.module(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = out.loss
        loss_value = float(loss.detach())
        assert math.isfinite(loss_value), (
            f"iter {i}: non-finite loss {loss_value}; losses so far={losses}"
        )
        loss.backward()
        optim.step()
        optim.zero_grad()
        losses.append(loss_value)

    print(f"\nProTrain seq-cls smoke (BERT-tiny): losses={losses}")

    assert all(math.isfinite(v) for v in losses), f"non-finite loss in {losses}"
