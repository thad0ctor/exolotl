#!/usr/bin/env python
"""Dissect WHERE the fp4_matmul fused-CE head spends time, and whether a single
big (non-tiled) FP4 _scaled_mm changes the verdict.

Components measured on a fixed [M,H] hidden against a [V,H] NVFP4 store:
  A) activation quant once (mslk single-level)              -- already hoisted
  B) tiled _scaled_mm (per _VOCAB_BLOCK) producing logit tiles
  C) tiled fp32 logsumexp/gather (the CE accumulation)
  D) SINGLE big _scaled_mm -> [M,V] logits then one CE       -- non-tiled, more mem
  E) bf16: single cuBLAS [M,V] GEMM + one CE                 -- the baseline
  F) bf16-compute tiled (dequant tile -> bf16 mm + tiled CE) -- the OFF path

Reports eager and torch.compile timings, and the scaled_mm launch count.
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

from axolotl.utils.nvfp4_training import (
    NVFP4FrozenBaseLinear, NVFP4Recipe, _mslk_quantize_sl,
)
from axolotl.kernels.nvfp4_fused_ce import (
    _fp4_logits_tile, _weight_scale_for_scaled_mm, _VOCAB_BLOCK,
)

DEV = "cuda"
DT = torch.bfloat16
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
    print(f"M={M} V={V} H={H} VOCAB_BLOCK={_VOCAB_BLOCK} "
          f"ntiles={(V + _VOCAB_BLOCK - 1)//_VOCAB_BLOCK}\n", flush=True)

    W = torch.randn(V, H, device=DEV, dtype=DT) * 0.02
    lin = nn.Linear(H, V, bias=False).to(DEV, DT)
    lin.weight.data.copy_(W)
    head = NVFP4FrozenBaseLinear.from_linear(
        lin, NVFP4Recipe(stochastic_rounding=False, hadamard=False), fsdp=False
    ).to(DEV)
    store = head.w_q
    hidden = torch.randn(M, H, device=DEV, dtype=DT)
    scale = 1.0

    # ---- A) activation quant once ----
    def quant_once():
        return _mslk_quantize_sl(hidden)
    t_quant = timed(quant_once)
    hq, hsc = _mslk_quantize_sl(hidden)

    tiles = [(lo, min(lo + _VOCAB_BLOCK, V)) for lo in range(0, V, _VOCAB_BLOCK)]
    sliced = [store[lo:hi] for lo, hi in tiles]

    # ---- B) tiled scaled_mm only (no CE) ----
    def tiled_mm():
        out = []
        for tile in sliced:
            out.append(_fp4_logits_tile(hq, hsc, tile, DT))
        return out
    t_tiledmm = timed(tiled_mm)

    # ---- C) tiled scaled_mm + fp32 logsumexp/gather (the real fwd CE) ----
    labels = torch.randint(0, V, (M,), device=DEV)
    def tiled_full():
        rmax = torch.full((M,), float("-inf"), device=DEV, dtype=torch.float32)
        rsum = torch.zeros(M, device=DEV, dtype=torch.float32)
        llog = torch.zeros(M, device=DEV, dtype=torch.float32)
        for (lo, hi), tile in zip(tiles, sliced):
            logits = _fp4_logits_tile(hq, hsc, tile, DT).float() * scale
            tmax = logits.max(dim=1).values
            nmax = torch.maximum(rmax, tmax)
            rsum = rsum * torch.exp(rmax - nmax) + torch.exp(logits - nmax.unsqueeze(1)).sum(1)
            rmax = nmax
            in_t = (labels >= lo) & (labels < hi)
            cols = (labels - lo).clamp(0, hi - lo - 1)
            g = logits.gather(1, cols.unsqueeze(1)).squeeze(1)
            llog = torch.where(in_t, g, llog)
        return rmax + torch.log(rsum) - llog
    t_tiledfull = timed(tiled_full)

    # ---- D) SINGLE big scaled_mm -> [M,V] -> one CE ----
    big = store  # the whole [V,H] NVFP4 store; one scaled_mm
    def big_mm_ce():
        logits = _fp4_logits_tile(hq, hsc, big, DT).float() * scale  # [M,V]
        lse = torch.logsumexp(logits, dim=1)
        return lse - logits.gather(1, labels.unsqueeze(1)).squeeze(1)
    try:
        t_bigmm = timed(big_mm_ce)
    except Exception as e:
        t_bigmm = float("nan")
        print("  big_mm_ce FAILED:", type(e).__name__, e)

    # ---- E) bf16 baseline: single cuBLAS GEMM + CE ----
    Wb = store.dequantize(DT)
    def bf16_ce():
        logits = (hidden @ Wb.t()).float() * scale
        lse = torch.logsumexp(logits, dim=1)
        return lse - logits.gather(1, labels.unsqueeze(1)).squeeze(1)
    t_bf16 = timed(bf16_ce)

    # ---- F) bf16-compute tiled (the OFF path) ----
    deq_tiles = [t.dequantize(DT) for t in sliced]
    def bf16_tiled():
        rmax = torch.full((M,), float("-inf"), device=DEV, dtype=torch.float32)
        rsum = torch.zeros(M, device=DEV, dtype=torch.float32)
        llog = torch.zeros(M, device=DEV, dtype=torch.float32)
        for (lo, hi), wt in zip(tiles, deq_tiles):
            logits = (hidden @ wt.t()).float() * scale
            tmax = logits.max(dim=1).values
            nmax = torch.maximum(rmax, tmax)
            rsum = rsum * torch.exp(rmax - nmax) + torch.exp(logits - nmax.unsqueeze(1)).sum(1)
            rmax = nmax
            in_t = (labels >= lo) & (labels < hi)
            cols = (labels - lo).clamp(0, hi - lo - 1)
            g = logits.gather(1, cols.unsqueeze(1)).squeeze(1)
            llog = torch.where(in_t, g, llog)
        return rmax + torch.log(rsum) - llog
    t_bf16tiled = timed(bf16_tiled)

    print("EAGER (ms/call, fwd-only):")
    print(f"  A activation quant (once)          {t_quant:8.3f}")
    print(f"  B tiled scaled_mm only             {t_tiledmm:8.3f}")
    print(f"  C tiled scaled_mm + tiled CE       {t_tiledfull:8.3f}  <- fp4 fwd")
    print(f"  D SINGLE big scaled_mm + 1 CE      {t_bigmm:8.3f}")
    print(f"  E bf16 single GEMM + 1 CE          {t_bf16:8.3f}  <- baseline")
    print(f"  F bf16 tiled (OFF path)            {t_bf16tiled:8.3f}")
    print(f"\n  CE-overhead (C - B)                {t_tiledfull - t_tiledmm:8.3f}")
    print(f"  fp4-tiled vs bf16-single           {t_tiledfull/t_bf16:6.2f}x")
    print(f"  fp4-big   vs bf16-single           {t_bigmm/t_bf16:6.2f}x")
    print(f"  fp4-tiled vs fp4-big               {t_tiledfull/t_bigmm:6.2f}x")

    # ---- compiled variants of C, D, E, F ----
    import torch._dynamo as dynamo
    dynamo.config.suppress_errors = True
    print("\nCOMPILED (ms/call, fwd-only):")
    for name, fn in [("C tiled fp4", tiled_full), ("D big fp4", big_mm_ce),
                     ("E bf16 single", bf16_ce), ("F bf16 tiled", bf16_tiled)]:
        dynamo.reset()
        from torch._dynamo.utils import counters
        counters.clear()
        try:
            cfn = torch.compile(fn)
            t = timed(cfn, iters=50, warmup=25)
            gb = sum(counters.get("graph_break", {}).values())
            print(f"  {name:18s} {t:8.3f}  [graph_breaks={gb}]")
        except Exception as e:
            print(f"  {name:18s} FAILED {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
