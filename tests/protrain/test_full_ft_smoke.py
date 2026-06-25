"""Full-finetune smoke test (no LoRA) for ProTrain — Item 9 cell B.

Every existing E2E ProTrain test wraps the model in LoRA before
``protrain_model_wrapper``. LoRA freezes >99% of the base parameters,
so the gradient pipeline only ever runs through ~1% of the chunks at
backward + optimizer-step time. Mode-B and Mode-C optimizer-state
sizing, the persistent-chunk grad-reduce coalesce, and the CPU/GPU
FusedAdam adapter pair could silently regress on full-fine-tune
workloads and no test would catch it.

This test exercises the full-FT path on a tiny SmolLM2-135M (a
Llama-architecture causal LM cached locally; falls back to a
fresh-init tiny Llama config when the cache is missing). The model
has every parameter trainable; ProTrain wraps it in Mode-A
(``force_all_persistent=True``) on a single GPU and runs three
training iterations. Acceptance:

* No crash, all losses finite.
* Loss decreases over the three iterations (final < first).

Mode-A is chosen rather than Mode-C because (a) this is a
single-GPU smoke and Mode-C requires a process group, and (b) the
"does the full-FT optimizer adapter pair drive every param" question
is the same in either mode — the gradient flows through every chunk
either way. The test is fast-lane (no ``slow`` mark) — at 135M params
the whole pipeline runs in well under 30s on a single 3090.
"""

from __future__ import annotations

import math

import pytest


def test_protrain_full_ft_smoke_smollm2() -> None:
    """SmolLM2-135M full-FT (no LoRA): three iters, finite losses, decreasing."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")

    import torch

    if not torch.cuda.is_available():
        pytest.skip("ProTrain full-FT smoke requires CUDA.")

    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        LlamaConfig,
        LlamaForCausalLM,
    )

    # Try the cached SmolLM2-135M first (Llama architecture, ~135M
    # params); fall back to a fresh-init tiny Llama if the HF cache is
    # cold or the host is offline. ``local_files_only=True`` keeps the
    # test deterministic — never reaches out to the hub mid-run.
    model: torch.nn.Module
    try:
        cfg = AutoConfig.from_pretrained(
            "HuggingFaceTB/SmolLM2-135M", local_files_only=True
        )
        cfg.use_cache = False
        model = AutoModelForCausalLM.from_pretrained(
            "HuggingFaceTB/SmolLM2-135M",
            local_files_only=True,
            torch_dtype=torch.bfloat16,
        )
    except Exception:
        # Fallback: fresh-init tiny Llama (same arch class as SmolLM2,
        # so ProTrain's block discovery via ``model.layers`` resolves
        # identically). Sized to match the smoke's "fast lane" intent —
        # 4 blocks, 256 hidden, total ~3M params.
        cfg = LlamaConfig(
            hidden_size=256,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=4,
            intermediate_size=512,
            vocab_size=1024,
            max_position_embeddings=128,
            rms_norm_eps=1e-5,
            use_cache=False,
        )
        model = LlamaForCausalLM(cfg).to(dtype=torch.bfloat16)

    device = torch.device("cuda:0")
    model = model.to(device)

    # Sanity: every param is trainable (no LoRA freeze).
    n_total = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n_trainable == n_total, (
        f"full-FT smoke expects every parameter trainable; "
        f"trainable={n_trainable} total={n_total}"
    )

    # ProTrain wrap (Mode-A: all chunks pinned on GPU, no offload).
    from axolotl.integrations.protrain.api import (
        protrain_model_wrapper,
        protrain_optimizer_wrapper,
    )
    from axolotl.integrations.protrain.types import HardwareProfile

    hw = HardwareProfile(
        gpu_sku=torch.cuda.get_device_name(0),
        gpu_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
        gpu_count=1,
        pcie_h2d_bps=13e9,
        pcie_d2h_bps=13e9,
        has_nvlink=False,
    )

    bs, seq = 1, 64
    wrapped = protrain_model_wrapper(
        model,
        model_config=cfg,
        hardware_profile=hw,
        batch_size=bs,
        seq_len=seq,
        capacity_bytes=20 * (1 << 30),
        force_all_persistent=True,
    )
    # 1e-3 LR — fresh-init or pretrained, both produce a visible loss
    # drop within three iters at this scale on bf16. The full-FT path
    # actually applies this LR to every param, so loss has to move; if
    # the optimizer adapter pair is silently a no-op the assertion at
    # the bottom catches it.
    optim = protrain_optimizer_wrapper(wrapped, lr=1e-3)

    vocab = int(getattr(cfg, "vocab_size", 1024))
    # Use the same input across iters so the only thing changing the
    # loss is parameter updates — makes the "loss decreases" check a
    # clean signal.
    torch.manual_seed(0)
    input_ids = torch.randint(0, vocab, (bs, seq), device=device, dtype=torch.long)
    labels = input_ids.clone()

    losses: list[float] = []
    n_iters = 3
    for i in range(n_iters):
        out = wrapped.module(input_ids=input_ids, labels=labels)
        loss = out.loss
        loss_value = float(loss.detach())
        assert math.isfinite(loss_value), (
            f"iter {i}: non-finite loss {loss_value}; losses so far={losses}"
        )
        loss.backward()
        optim.step()
        optim.zero_grad()
        losses.append(loss_value)

    print(f"\nProTrain full-FT smoke (SmolLM2-135M / tiny-Llama): losses={losses}")

    assert all(math.isfinite(v) for v in losses), f"non-finite loss in {losses}"
    assert losses[-1] < losses[0], (
        f"full-FT loss did not decrease over {n_iters} iters: {losses} — "
        f"the full-FT optimizer-adapter path may be inert (gradients not "
        f"reaching every param's chunk-state, or step never applied)"
    )
