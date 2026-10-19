#!/usr/bin/env python
"""Correctness: the fp4_matmul fused-CE loss + grad_hidden must be identical
EAGER vs COMPILED (the re-engineered/compiled path must not change the math vs the
current eager fp4 path). Also report fp4 vs bf16-compute deltas (expected FP4 error).
"""
from __future__ import annotations
import os, sys
from pathlib import Path
import torch
import torch.nn as nn

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
import mslk  # noqa
assert not getattr(mslk, "_axo_stub", False)

from axolotl.utils.nvfp4_training import NVFP4FrozenBaseLinear, NVFP4Recipe
from axolotl.kernels.nvfp4_fused_ce import fused_fp4_cross_entropy

DEV, DT = "cuda", torch.bfloat16
assert "5090" in torch.cuda.get_device_name(0)


def run(fp4_mm, compiled, head, labels, h0):
    h = h0.clone().requires_grad_(True)
    fn = (lambda: fused_fp4_cross_entropy(h, head, labels, shift=False, fp4_matmul=fp4_mm))
    if compiled:
        import torch._dynamo as dynamo
        dynamo.reset()
        fn = torch.compile(fn)
    loss = fn()
    loss.backward()
    return loss.detach().float().item(), h.grad.detach().clone()


def main():
    M, V, H = 1024, 151936, 2048
    torch.manual_seed(0)
    W = torch.randn(V, H, device=DEV, dtype=DT) * 0.02
    lin = nn.Linear(H, V, bias=False).to(DEV, DT); lin.weight.data.copy_(W)
    head = NVFP4FrozenBaseLinear.from_linear(
        lin, NVFP4Recipe(stochastic_rounding=False, hadamard=False), fsdp=False).to(DEV)
    labels = torch.randint(0, V, (M,), device=DEV)
    h0 = torch.randn(M, H, device=DEV, dtype=DT)

    l_e, g_e = run(True, False, head, labels, h0)   # fp4 eager (current path)
    l_e2, g_e2 = run(True, False, head, labels, h0)  # fp4 eager AGAIN (nondeterminism floor)
    l_c, g_c = run(True, True, head, labels, h0)    # fp4 compiled (re-engineered)
    l_b, g_b = run(False, True, head, labels, h0)   # bf16-compute compiled

    de = (g_e - g_e2).abs().max().item()
    print(f"[floor] eager-vs-eager (same path, nondeterminism): "
          f"|dloss|={abs(l_e-l_e2):.2e} grad max|d|={de:.2e}")

    dloss = abs(l_e - l_c)
    dg = (g_e - g_c).abs().max().item()
    rel_g = (g_e - g_c).abs().max().item() / (g_e.abs().max().item() + 1e-12)
    print(f"fp4 EAGER loss   = {l_e:.8f}")
    print(f"fp4 COMPILED loss= {l_c:.8f}   |dloss|={dloss:.2e}")
    print(f"grad_hidden max|eager-compiled| = {dg:.2e}  rel={rel_g:.2e}")
    print(f"  -> {'BIT-EXACT' if dg==0 else 'within reduction-reorder noise' if rel_g<1e-2 else 'MISMATCH'}")
    print(f"\nfp4 vs bf16-compute (expected FP4 quant error):")
    print(f"  loss fp4={l_c:.6f} bf16c={l_b:.6f} |d|={abs(l_c-l_b):.2e}")
    print(f"  grad rel = {(g_c-g_b).abs().max().item()/(g_b.abs().max().item()+1e-12):.2e}")


if __name__ == "__main__":
    main()
