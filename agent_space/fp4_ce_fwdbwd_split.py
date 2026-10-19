#!/usr/bin/env python
"""Split fwd vs bwd for the compiled fp4 vs bf16-compute CE Function.

Times: fwd-only (no backward) and fwd+bwd, compiled, for both paths.
bwd = (fwd+bwd) - fwd. Pinpoints whether fp4's loss is in fwd or bwd.
The fp4 backward recomputes logits via fp4 scaled_mm AND dequants the tile for a
bf16 grad GEMM (dz @ w_tile) -- so it does ~2x the matmul work of forward.
"""
from __future__ import annotations
import os, sys, time
from pathlib import Path
import torch
import torch.nn as nn

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
import mslk  # noqa
assert not getattr(mslk, "_axo_stub", False)

from axolotl.utils.nvfp4_training import NVFP4FrozenBaseLinear, NVFP4Recipe
from axolotl.kernels.nvfp4_fused_ce import fused_fp4_cross_entropy, _VOCAB_BLOCK

DEV, DT = "cuda", torch.bfloat16
assert "5090" in torch.cuda.get_device_name(0)


def timed(fn, iters=40, warmup=15):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0


def main():
    M = int(os.environ.get("M", "1024"))
    V = int(os.environ.get("V", "151936"))
    H = int(os.environ.get("H", "2048"))
    torch.manual_seed(0)
    print(f"M={M} V={V} H={H} VOCAB_BLOCK={_VOCAB_BLOCK}\n", flush=True)

    W = torch.randn(V, H, device=DEV, dtype=DT) * 0.02
    lin = nn.Linear(H, V, bias=False).to(DEV, DT)
    lin.weight.data.copy_(W)
    head = NVFP4FrozenBaseLinear.from_linear(
        lin, NVFP4Recipe(stochastic_rounding=False, hadamard=False), fsdp=False).to(DEV)
    labels = torch.randint(0, V, (M,), device=DEV)
    h_const = torch.randn(M, H, device=DEV, dtype=DT)

    import torch._dynamo as dynamo
    dynamo.config.suppress_errors = True

    def fwd_only(fp4_mm):
        def step():
            h = h_const.clone().requires_grad_(True)
            return fused_fp4_cross_entropy(h, head, labels, shift=False, fp4_matmul=fp4_mm)
        return step

    def fwd_bwd(fp4_mm):
        def step():
            h = h_const.clone().requires_grad_(True)
            loss = fused_fp4_cross_entropy(h, head, labels, shift=False, fp4_matmul=fp4_mm)
            loss.backward()
            return h.grad
        return step

    for label, fp4_mm in [("bf16-compute", False), ("fp4_matmul", True)]:
        dynamo.reset()
        cf = torch.compile(fwd_only(fp4_mm))
        t_f = timed(cf)
        dynamo.reset()
        cfb = torch.compile(fwd_bwd(fp4_mm))
        t_fb = timed(cfb)
        print(f"{label:14s} fwd {t_f:7.3f}  fwd+bwd {t_fb:7.3f}  bwd~{t_fb - t_f:7.3f} ms")


if __name__ == "__main__":
    main()
