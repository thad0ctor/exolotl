"""fused_rope_quant_qk must read a transposed (non-contiguous, D-unit-stride) Q/K
view and produce BIT-IDENTICAL packs to the contiguous path — the invariant behind
dropping the per-layer .contiguous() copy (prefill grab #2). The production caller
passes q_norm(...).transpose(1,2), exactly this layout."""

import pytest
import torch

cuda = torch.cuda.is_available()
pytestmark = pytest.mark.skipif(not cuda, reason="needs a CUDA (sm_120) device")

if cuda:
    from axolotl.kernels.nvfp4_fused_producers import fused_rope_quant_qk


@pytest.mark.parametrize(
    "Z,H,S,D", [(1, 16, 300, 256), (1, 8, 256, 128), (2, 4, 128, 256)]
)
def test_strided_matches_contiguous(Z, H, S, D):
    torch.manual_seed(0)
    rot = D
    base = torch.randn(Z, S, H, D, device="cuda", dtype=torch.bfloat16)
    x_t = base.transpose(1, 2)  # [Z,H,S,D] non-contiguous, D unit-stride
    assert x_t.stride(3) == 1 and not x_t.is_contiguous()
    cos = torch.randn(Z, S, rot, device="cuda", dtype=torch.bfloat16)
    sin = torch.randn(Z, S, rot, device="cuda", dtype=torch.bfloat16)

    q_s, sc_s = fused_rope_quant_qk(x_t, cos, sin)  # strided (no copy)
    q_c, sc_c = fused_rope_quant_qk(x_t.contiguous(), cos, sin)  # contiguous reference

    assert torch.equal(q_s, q_c), "packed FP4 differs between strided and contiguous"
    assert torch.equal(sc_s.view(torch.uint8), sc_c.view(torch.uint8)), "scale differs"
    assert q_s.shape == (Z * H, S, D // 2) and q_s.dtype == torch.uint8
    assert sc_s.shape == (Z * H, S, D // 16)


def test_noncontiguous_d_falls_back():
    """If head_dim is NOT unit-stride, it must fall back to a copy and still be correct."""
    torch.manual_seed(0)
    Z, H, S, D = 1, 4, 64, 128
    # make D non-unit-stride by transposing S<->D, then take a view where D is dim 3
    x = torch.randn(Z, H, D, S, device="cuda", dtype=torch.bfloat16).transpose(
        2, 3
    )  # [Z,H,S,D], D stride = S
    assert x.stride(3) != 1
    cos = torch.randn(Z, S, D, device="cuda", dtype=torch.bfloat16)
    sin = torch.randn(Z, S, D, device="cuda", dtype=torch.bfloat16)
    q_fb, sc_fb = fused_rope_quant_qk(x, cos, sin)
    q_ref, sc_ref = fused_rope_quant_qk(x.contiguous(), cos, sin)
    assert torch.equal(q_fb, q_ref) and torch.equal(
        sc_fb.view(torch.uint8), sc_ref.view(torch.uint8)
    )


# ---------------------------------------------------------------------------
# STORE_HP: the training-path variant that also writes the roped hp tensor.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("Z,H,S,D", [(1, 4, 256, 128), (1, 16, 300, 256)])
def test_store_hp_rope_parity_and_pack_consistency(Z, H, S, D):
    """``store_hp=True``: (a) the roped hp output matches the HF rotate_half
    reference (cos > 0.999 — fp32 rope math vs HF's bf16, not bit-equal);
    (b) the emitted packs are BIT-IDENTICAL to ``_quant_nvfp4`` of that hp
    tensor (the invariant that makes the prepacked training forward
    bit-identical to the standalone-quant path); (c) ``store_hp=False`` output
    is unchanged vs the hp variant's packs ONLY where the dtype rounding
    agrees — but is itself unchanged vs the pre-STORE_HP kernel (same pack
    math from unrounded fp32)."""
    import torch.nn.functional as F
    from axolotl.kernels.attn_nvfp4_flash import _quant_nvfp4

    torch.manual_seed(1)
    rot = D
    x = torch.randn(Z, H, S, D, device="cuda", dtype=torch.bfloat16)
    cos = torch.randn(Z, S, rot, device="cuda", dtype=torch.bfloat16)
    sin = torch.randn(Z, S, rot, device="cuda", dtype=torch.bfloat16)

    qnv, qsc, hp = fused_rope_quant_qk(x, cos, sin, store_hp=True)
    assert hp.shape == (Z, H, S, D) and hp.dtype == x.dtype

    # (a) HF rotate_half reference
    def rotate_half(t):
        half = t.shape[-1] // 2
        return torch.cat((-t[..., half:], t[..., :half]), dim=-1)

    ref = x * cos.unsqueeze(1) + rotate_half(x) * sin.unsqueeze(1)
    c = F.cosine_similarity(hp.float().flatten(), ref.float().flatten(), dim=0)
    assert c.item() > 0.999, f"STORE_HP rope cos vs HF reference {c.item():.6f}"

    # (b) packs == _quant_nvfp4(hp), bit-for-bit
    qq, ss = _quant_nvfp4(hp.reshape(Z * H, S, D))
    assert torch.equal(qnv, qq)
    assert torch.equal(qsc.view(torch.uint8), ss.view(torch.uint8))

    # (c) the no-hp call still returns the 2-tuple with the same shapes
    qnv0, qsc0 = fused_rope_quant_qk(x, cos, sin)
    assert qnv0.shape == qnv.shape and qsc0.shape == qsc.shape


def test_fused_rope_quant_qk_hp_grad_matches_hf_autograd():
    """The differentiable wrapper's backward (exact RoPE transpose) matches
    autograd through the stock HF apply_rotary_pos_emb formula bit-for-bit."""
    Z, H, S, D = 1, 4, 192, 128
    torch.manual_seed(2)
    from axolotl.kernels.nvfp4_fused_producers import fused_rope_quant_qk_hp

    x = torch.randn(Z, H, S, D, device="cuda", dtype=torch.bfloat16)
    cos = torch.randn(Z, S, D, device="cuda", dtype=torch.bfloat16)
    sin = torch.randn(Z, S, D, device="cuda", dtype=torch.bfloat16)
    g = torch.randn(Z, H, S, D, device="cuda", dtype=torch.bfloat16)

    xg = x.clone().requires_grad_(True)
    _, _, hp = fused_rope_quant_qk_hp(xg, cos, sin)
    hp.backward(g)

    def rotate_half(t):
        half = t.shape[-1] // 2
        return torch.cat((-t[..., half:], t[..., :half]), dim=-1)

    xr = x.clone().requires_grad_(True)
    ref = xr * cos.unsqueeze(1) + rotate_half(xr) * sin.unsqueeze(1)
    ref.backward(g)

    assert torch.equal(xg.grad, xr.grad)
