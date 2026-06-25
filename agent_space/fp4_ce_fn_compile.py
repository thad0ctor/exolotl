#!/usr/bin/env python
"""Does the autograd.Function actually get COMPILED (fused) or run eager-inside-graph?

Compile fused_fp4_cross_entropy end-to-end (fwd+bwd) for both:
  - fp4_matmul=False (bf16-compute, the OFF path that reportedly reaches parity)
  - fp4_matmul=True  (the scaled_mm path)
and time fwd+bwd. Also count actual scaled_mm / quant invocations via op logging,
and report graph breaks. If the Function body fuses, fp4 should beat bf16-compute
(per the dissect micro-bench: 1.62 vs 4.44 ms compiled). If it runs eager-in-graph,
fp4 stays at the eager ~7ms+ number.
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


def timed(fn, iters=50, warmup=20):
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
        lin, NVFP4Recipe(stochastic_rounding=False, hadamard=False), fsdp=False
    ).to(DEV)
    labels = torch.randint(0, V, (M,), device=DEV)

    import torch._dynamo as dynamo
    dynamo.config.suppress_errors = True

    def make_step(fp4_mm):
        def step():
            h = torch.randn(M, H, device=DEV, dtype=DT, requires_grad=True)
            loss = fused_fp4_cross_entropy(h, head, labels, shift=False,
                                           fp4_matmul=fp4_mm)
            loss.backward()
            return h.grad
        return step

    for label, fp4_mm in [("bf16-compute (OFF)", False), ("fp4_matmul (ON)", True)]:
        # eager
        t_eager = timed(make_step(fp4_mm), iters=30, warmup=10)
        # compiled
        dynamo.reset()
        from torch._dynamo.utils import counters
        counters.clear()
        cstep = torch.compile(make_step(fp4_mm))
        t_comp = timed(cstep, iters=30, warmup=15)
        gb = sum(counters.get("graph_break", {}).values())
        ug = counters.get("stats", {}).get("unique_graphs", "?")
        print(f"{label:22s} eager {t_eager:8.3f} ms  compiled {t_comp:8.3f} ms "
              f"[gb={gb} graphs={ug}]")


if __name__ == "__main__":
    main()
