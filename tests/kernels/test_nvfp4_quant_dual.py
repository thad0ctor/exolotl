"""Bit-exact correctness for the fused dual-layout NVFP4 pack.

``_quant_nvfp4_dual`` reads each ``[B, S, D]`` source ONCE and emits both the
along-D pack and the along-S (transpose) pack, replacing two separate
``_quant_nvfp4`` launches in the forward-for-backward path. The fused packs must
be byte-identical to the two separate calls (deterministic quant) — this gate
fails on any divergence in the packed nibbles OR the e4m3 block scales.
"""

import pytest
import torch

pytest.importorskip("triton", reason="triton required for fused kernels")
pytest.importorskip("mslk", reason="mslk fp4 packer required")

if not torch.cuda.is_available():
    pytest.skip("CUDA required for fused kernel tests", allow_module_level=True)

from axolotl.kernels.attn_nvfp4_flash import (  # noqa: E402
    _next_mult,
    _quant_nvfp4,
    _quant_nvfp4_dual,
)


def _eq_u8(a, b):
    return torch.equal(a.view(torch.uint8), b.view(torch.uint8))


@pytest.mark.parametrize("d", [128, 256])
@pytest.mark.parametrize("s", [300, 512, 1000, 2048])
@pytest.mark.parametrize("b", [1, 4])
@pytest.mark.parametrize("pad_mult", [64, 128])
def test_dual_pack_bit_exact(d, s, b, pad_mult):
    torch.manual_seed(1234)
    s_pad = _next_mult(s, pad_mult)
    x = torch.randn(b, s, d, device="cuda", dtype=torch.bfloat16)

    qa_ref, sa_ref = _quant_nvfp4(x)
    qb_ref, sb_ref = _quant_nvfp4(x, transpose=True, k_pad=s_pad)
    qa, sa, qb, sb = _quant_nvfp4_dual(x, s_pad)

    # layout A == along-D _quant_nvfp4(x)
    assert _eq_u8(qa, qa_ref), "layout-A packed nibbles diverged"
    assert _eq_u8(sa, sa_ref), "layout-A e4m3 scales diverged"
    # layout B == along-S _quant_nvfp4(x, transpose=True, k_pad=s_pad)
    assert _eq_u8(qb, qb_ref), "layout-B packed nibbles diverged"
    assert _eq_u8(sb, sb_ref), "layout-B e4m3 scales diverged"
