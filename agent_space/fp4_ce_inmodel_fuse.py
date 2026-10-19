#!/usr/bin/env python
"""DECISIVE TEST: when fused_fp4_cross_entropy runs inside the COMPILED model
forward (the real axolotl path), does its tile loop fuse into the inductor graph,
or fall back to eager?

We build a tiny ForCausalLM-like module: a small transformer body (one linear) ->
hidden -> patched forward calls fused_fp4_cross_entropy. We compile the WHOLE
module (mirrors torch.compile(model)) and instrument:
  - is_compiling() seen inside the Function forward  (True => fused into graph)
  - number of inductor graphs produced
  - graph breaks
  - eager vs compiled step time of the fused-CE region

This isolates the model-wrapping effect (patched forward + CausalLMOutputWithPast)
from the full Qwen3 weight/optimizer load that OOMs under GPU contention.
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
import axolotl.kernels.nvfp4_fused_ce as fce
from axolotl.kernels.nvfp4_fused_ce import fused_fp4_cross_entropy, _VOCAB_BLOCK

DEV, DT = "cuda", torch.bfloat16
assert "5090" in torch.cuda.get_device_name(0)

# NOTE: no side-effecting instrumentation here -- a list.append inside the
# Function forward is itself untraceable and forces recompiles + eager fallback,
# poisoning the very thing we measure. We rely on the eager-vs-compiled speedup
# and graph/break counters instead.


def timed(fn, iters=30, warmup=15):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0


class TinyCausalLM(nn.Module):
    """Mirror the patched-forward structure: a body producing hidden, then the
    fused FP4 CE replacing lm_head+loss (exactly _make_fused_forward's shape)."""
    def __init__(self, head, H):
        super().__init__()
        self.body = nn.Linear(H, H, bias=False).to(DEV, DT)
        self.head = head

    def forward(self, x, labels):
        hidden = self.body(x)  # [M,H]
        loss = fused_fp4_cross_entropy(hidden, self.head, labels, shift=False,
                                       fp4_matmul=True)
        return loss


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
    x = torch.randn(M, H, device=DEV, dtype=DT)

    import torch._dynamo as dynamo
    dynamo.config.suppress_errors = True
    from torch._dynamo.utils import counters

    def step(m):
        m.body.weight.grad = None
        loss = m(x, labels)
        loss.backward()
        return loss

    for fp4_mm, tag in [(True, "fp4_matmul"), (False, "bf16-compute")]:
        # patch the module's fp4 flag via a fresh model each config
        class _M(TinyCausalLM):
            def forward(self, x, labels):
                hidden = self.body(x)
                return fused_fp4_cross_entropy(hidden, self.head, labels,
                                               shift=False, fp4_matmul=fp4_mm)
        model = _M(head, H)
        for p in model.parameters(): p.requires_grad_(True)
        t_eager = timed(lambda: step(model))
        dynamo.reset(); counters.clear()
        cmodel = torch.compile(model)
        t_comp = timed(lambda: step(cmodel))
        gb = sum(counters.get("graph_break", {}).values())
        ug = counters.get("stats", {}).get("unique_graphs", "?")
        print(f"{tag:14s} EAGER {t_eager:7.3f}  COMPILED {t_comp:7.3f} ms "
              f"[gb={gb} graphs={ug}]  speedup={t_eager/t_comp:.2f}x")


if __name__ == "__main__":
    main()
