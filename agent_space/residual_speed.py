#!/usr/bin/env python
"""Speed probe: permanent per-step cost of the low-rank residual on the FP4 head.

The residual is two small matmuls per head forward: (hidden[M,H] @ B.t()[H,k]) then
([M,k] @ A.t()[k,V]). It is paid at TRAIN and INFERENCE. We time:
  - base   : the FP4 lm_head GEMM alone (NVFP4FrozenBaseFunction)
  - +res_k : base + rank-k residual, for k in the sweep
and report ms/forward and the residual overhead %.

Run on a free RTX PRO 6000 (CUDA_VISIBLE_DEVICES=0/3), NOT GPU 6 (5090).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "agent_space"))


def _time(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters  # ms/iter


def run(args):
    import torch.nn as nn

    from axolotl.utils.nvfp4_training import NVFP4FrozenBaseLinear, NVFP4Recipe

    dev = "cuda"
    dt = torch.bfloat16
    V, H, M = args.vocab, args.hidden, args.tokens
    print(f"[speed] V={V} H={H} M={M} dtype={dt}", flush=True)

    lin = nn.Linear(H, V, bias=False).to(dev, dt)
    fp4 = NVFP4FrozenBaseLinear.from_linear(lin, NVFP4Recipe(), fsdp=False)
    x = torch.randn(M, H, device=dev, dtype=dt)

    base_ms = _time(lambda: fp4(x))
    print(f"  base FP4 GEMM            : {base_ms:.4f} ms", flush=True)

    for k in args.ranks:
        A = torch.randn(V, k, device=dev, dtype=dt)
        B = torch.randn(k, H, device=dev, dtype=dt)

        def step(A=A, B=B):
            o = fp4(x)
            return o + (x @ B.t()) @ A.t()

        ms = _time(step)
        ov = 100.0 * (ms - base_ms) / base_ms
        print(f"  +residual k={k:<3d}          : {ms:.4f} ms  (+{ov:.2f}% over base)", flush=True)


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--vocab", type=int, default=151936)
    p.add_argument("--hidden", type=int, default=4096)
    p.add_argument("--tokens", type=int, default=4096)
    p.add_argument("--ranks", type=int, nargs="*", default=[8, 16, 32, 64])
    return p.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
