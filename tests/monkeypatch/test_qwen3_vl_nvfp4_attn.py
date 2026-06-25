"""GPU parity tests for the Qwen3-VL native-NVFP4 attention monkeypatch.

Distinct from test_qwen3_vl_fused_attn.py (the bf16 fused-attn path): this covers
``patch_qwen3_vl_nvfp4_attention`` — the FP4 attention path. Requires Blackwell
(sm>=100) + the sageattention.nvfp4 kernel; head_dim is 128 (the kernel supports
d in {128, 256}, and 128 is Qwen3-VL's). Each test builds a fresh model and the
patch swaps the per-instance forward, so no class-level restore is needed.
"""

import pytest
import torch
import torch.nn.functional as F

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
    pytest.mark.skipif(
        not (
            torch.cuda.is_available()
            and torch.cuda.get_device_capability() >= (10, 0)
        ),
        reason="native NVFP4 attention requires sm>=100 (Blackwell)",
    ),
]

pytest.importorskip("transformers.models.qwen3_vl")
pytest.importorskip("sageattention.nvfp4")

HEAD_DIM = 128
N_HEADS = 4
N_KV = 2
HIDDEN = N_HEADS * HEAD_DIM
N_LAYERS = 2


def _build_model(seed: int = 0):
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel

    torch.manual_seed(seed)
    cfg = Qwen3VLTextConfig(
        vocab_size=128,
        hidden_size=HIDDEN,
        intermediate_size=256,
        num_hidden_layers=N_LAYERS,
        num_attention_heads=N_HEADS,
        num_key_value_heads=N_KV,
        head_dim=HEAD_DIM,
        max_position_embeddings=512,
        rms_norm_eps=1e-6,
        attention_dropout=0.0,
        pad_token_id=0,
    )
    cfg._attn_implementation = "sdpa"
    return Qwen3VLTextModel(cfg).cuda().to(torch.bfloat16).eval()


def _run_attention(model, layer_idx, hidden_states, position_ids):
    attn = model.layers[layer_idx].self_attn
    cos, sin = model.rotary_emb(hidden_states, position_ids)
    out, _ = attn(
        hidden_states=hidden_states,
        position_embeddings=(cos, sin),
        attention_mask=None,
    )
    return out


def _inputs(seq=256, seed=0):
    torch.manual_seed(seed)
    hs = torch.randn(1, seq, HIDDEN, device="cuda", dtype=torch.bfloat16)
    pos = torch.arange(seq, device="cuda").unsqueeze(0)
    return hs, pos


@pytest.mark.parametrize("layer_idx", [0, 1])
def test_nvfp4_vl_forward_parity_nograd(layer_idx):
    """No-grad (eval/prefill) path runs NVFP4 and matches SDPA within FP4 error."""
    from axolotl.monkeypatch.attention.nvfp4_flash_attn_vl import (
        patch_qwen3_vl_nvfp4_attention,
    )

    model = _build_model(seed=1)
    hs, pos = _inputs(seed=1)

    with torch.no_grad():
        ref = _run_attention(model, layer_idx, hs, pos)

    assert patch_qwen3_vl_nvfp4_attention(model, train_backward=True) == N_LAYERS
    with torch.no_grad():
        got = _run_attention(model, layer_idx, hs, pos)

    assert got.shape == ref.shape
    assert torch.isfinite(got).all()
    cos = F.cosine_similarity(ref.flatten().float(), got.flatten().float(), dim=0)
    assert cos > 0.9, f"layer {layer_idx} nvfp4-vl no-grad cos={cos:.4f} (FP4 lossy)"


# Qwen3-VL has this many full-attention LM layers; FP4's per-layer error compounds
# over ALL of them (the model is non-hybrid), unlike Qwen3.5 where only a few layers
# are FP4'd. cos^FULL_MODEL_LAYERS is the relevant model-level fidelity.
FULL_MODEL_LAYERS = 36
# Per-layer cosine required for the compounded product to stay >= 0.99 across the
# full model. At ~0.97-0.985 (measured) FP4 attention is WELL below this, which is
# why patching every layer wrecks the forward (step-1 loss ~5.3 vs ~1.4). The patch
# manager refuses FP4 attention on non-hybrid models for exactly this reason.
MODEL_SAFE_PER_LAYER_COS = 0.99 ** (1.0 / FULL_MODEL_LAYERS)


def test_nvfp4_vl_per_layer_error_is_not_model_safe():
    """Document the limitation: D=128 FP4 attention's per-layer cosine vs fp32 SDPA
    is too low to survive compounding over all 36 Qwen3-VL layers.

    This is NOT a kernel bug (the kernel matches an independent NVFP4 emulation) — it
    is inherent FP4 precision loss. If a future change makes the per-layer cosine
    model-safe, flip this assertion (and relax the patch-manager guard).
    """
    from axolotl.monkeypatch.attention.nvfp4_flash_attn_vl import (
        patch_qwen3_vl_nvfp4_attention,
    )

    # Realistic shape: longer seq, full head count.
    model = _build_model(seed=7)
    hs, pos = _inputs(seq=1024, seed=7)

    with torch.no_grad():
        ref = _run_attention(model, 0, hs, pos)
    patch_qwen3_vl_nvfp4_attention(model, train_backward=True)
    with torch.no_grad():
        got = _run_attention(model, 0, hs, pos)

    cos = F.cosine_similarity(
        ref.flatten().float(), got.flatten().float(), dim=0
    ).item()
    compounded = cos**FULL_MODEL_LAYERS
    assert cos < MODEL_SAFE_PER_LAYER_COS, (
        f"per-layer cos={cos:.4f} is now >= model-safe "
        f"{MODEL_SAFE_PER_LAYER_COS:.4f}; FP4 attention may now be safe on "
        "non-hybrid models — revisit the patch-manager guard."
    )
    # Compounded fidelity over the full model is poor: this is the documented limit.
    assert compounded < 0.9, f"compounded cos^{FULL_MODEL_LAYERS}={compounded:.4f}"


def test_nvfp4_vl_forward_parity_grad():
    """Grad (training) path runs NVFP4 and matches SDPA within FP4 error."""
    from axolotl.monkeypatch.attention.nvfp4_flash_attn_vl import (
        patch_qwen3_vl_nvfp4_attention,
    )

    model = _build_model(seed=2)
    hs, pos = _inputs(seed=2)

    with torch.no_grad():
        ref = _run_attention(model, 0, hs, pos)

    patch_qwen3_vl_nvfp4_attention(model, train_backward=True)
    got = _run_attention(model, 0, hs, pos)  # grad enabled

    assert torch.isfinite(got).all()
    cos = F.cosine_similarity(
        ref.flatten().float(), got.detach().flatten().float(), dim=0
    )
    assert cos > 0.9, f"nvfp4-vl grad cos={cos:.4f} (FP4 lossy)"


def test_nvfp4_vl_backward_grads_flow():
    """Backward through the NVFP4 path produces finite grads."""
    from axolotl.monkeypatch.attention.nvfp4_flash_attn_vl import (
        patch_qwen3_vl_nvfp4_attention,
    )

    model = _build_model(seed=3)
    patch_qwen3_vl_nvfp4_attention(model, train_backward=True)

    hs, pos = _inputs(seed=3)
    hs = hs.clone().requires_grad_(True)
    out = _run_attention(model, 0, hs, pos)
    # Use a strong (sum) signal — pow(2).mean() is ~1e-3 here and underflows to 0
    # in bf16 after the projection matmuls, masking a working backward.
    out.float().sum().backward()

    assert hs.grad is not None
    assert torch.isfinite(hs.grad).all()
    assert hs.grad.abs().sum() > 0


def test_patch_is_idempotent_and_counts_layers():
    from axolotl.monkeypatch.attention.nvfp4_flash_attn_vl import (
        patch_qwen3_vl_nvfp4_attention,
    )

    model = _build_model(seed=4)
    assert patch_qwen3_vl_nvfp4_attention(model, train_backward=True) == N_LAYERS
    # Re-patching an already-patched model adds nothing (updates attrs only).
    assert patch_qwen3_vl_nvfp4_attention(model, train_backward=True) == 0


def test_output_attentions_falls_back_to_stock():
    """output_attentions can't be served by the kernel -> stock forward."""
    from axolotl.monkeypatch.attention.nvfp4_flash_attn_vl import (
        patch_qwen3_vl_nvfp4_attention,
    )

    model = _build_model(seed=5)
    hs, pos = _inputs(seq=64, seed=5)
    cos, sin = model.rotary_emb(hs, pos)

    patch_qwen3_vl_nvfp4_attention(model, train_backward=True)
    attn = model.layers[0].self_attn
    with torch.no_grad():
        out, _weights = attn(
            hidden_states=hs,
            position_embeddings=(cos, sin),
            attention_mask=None,
            output_attentions=True,
        )
    # The NVFP4 path can't serve output_attentions; it must route to the stock
    # forward without erroring and still produce a finite output.
    assert out is not None and torch.isfinite(out).all()
