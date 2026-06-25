#!/usr/bin/env python
"""Clean single-config full Qwen3 step: bf16 OR fp4, in one process, with graph
diagnostics. Avoids the cross-config contamination + lets us see graphs/breaks and
whether the BODY stays compiled. Pass config as argv[1] in {bf16,fp4}.
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

cfg = sys.argv[1] if len(sys.argv) > 1 else "fp4"
if cfg == "fp4":
    os.environ["AXOLOTL_NVFP4_FUSED_CE_FP4_MM"] = "1"

DEV, DT = "cuda", torch.bfloat16
assert "5090" in torch.cuda.get_device_name(0)
import torch._dynamo as dynamo
dynamo.config.suppress_errors = True
dynamo.config.accumulated_cache_size_limit = 256

from transformers import AutoModelForCausalLM, AutoTokenizer
from axolotl.utils.nvfp4_training import NVFP4FrozenBaseLinear, NVFP4Recipe
from axolotl.kernels.nvfp4_fused_ce import patch_model_fused_fp4_ce


def main():
    mid = "Qwen/Qwen3-1.7B"
    B, S = int(os.environ.get("AB_B", "2")), int(os.environ.get("AB_S", "512"))
    tok = AutoTokenizer.from_pretrained(mid)
    model = AutoModelForCausalLM.from_pretrained(mid, dtype=DT).to(DEV)
    V = model.get_output_embeddings().weight.shape[0]
    H = model.config.hidden_size

    txt = "Quantization reduces precision to save memory. " * 400
    ids = tok(txt, return_tensors="pt").input_ids[0]
    need = B * (S + 1)
    ids = ids.repeat(need // ids.numel() + 1)[:need].to(DEV).reshape(B, S + 1)
    inp, labels = ids[:, :-1].contiguous(), ids[:, 1:].contiguous()

    if cfg == "fp4":
        W = model.get_output_embeddings().weight.detach().clone()
        lin = nn.Linear(H, V, bias=False).to(DEV, DT); lin.weight.data.copy_(W)
        head = NVFP4FrozenBaseLinear.from_linear(
            lin, NVFP4Recipe(stochastic_rounding=False, hadamard=False), fsdp=False).to(DEV)
        model.set_output_embeddings(head); model.lm_head = head
        assert patch_model_fused_fp4_ce(model)

    model.train()

    # Proof the fp4 scaled_mm path fires: count native mslk quant calls during a
    # single EAGER step (before compile, so the counter side-effect can't poison
    # the compiled graph). bf16 -> 0, fp4 -> >0 quant + scaled_mm calls.
    if cfg == "fp4":
        import axolotl.utils.nvfp4_training as _nt
        n = {"q": 0, "mm": 0}
        _oq = _nt._mslk_quantize_sl_op._opoverload if hasattr(_nt._mslk_quantize_sl_op, "_opoverload") else None
        _osm = torch._scaled_mm
        def _wrap_mm(*a, **k):
            n["mm"] += 1
            return _osm(*a, **k)
        torch._scaled_mm = _wrap_mm
        with torch.no_grad():
            _ = model(input_ids=inp, labels=labels)
        torch._scaled_mm = _osm
        print(f"[proof] fp4 eager forward: torch._scaled_mm calls = {n['mm']} "
              f"(>0 confirms native fp4 GEMM path fired)")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5)
    cmodel = torch.compile(model)

    from torch._dynamo.utils import counters
    counters.clear()

    def step():
        opt.zero_grad(set_to_none=True)
        out = cmodel(input_ids=inp, labels=labels)
        out.loss.backward()
        opt.step()

    # warmup / compile
    t0 = time.perf_counter()
    for _ in range(10):
        step(); torch.cuda.synchronize()
    warm = time.perf_counter() - t0
    gb = sum(counters.get("graph_break", {}).values())
    ug = counters.get("stats", {}).get("unique_graphs", "?")

    samples = []
    for _ in range(25):
        torch.cuda.synchronize(); s = time.perf_counter()
        step(); torch.cuda.synchronize()
        samples.append((time.perf_counter() - s) * 1000)
    samples.sort()
    med = samples[len(samples)//2]
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"\nCONFIG={cfg}  full step median {med:.3f} ms  peak {peak:.2f} GB  "
          f"warmup {warm:.1f}s  [graph_breaks={gb} unique_graphs={ug}]")
    top = sorted(counters.get("graph_break", {}).items(), key=lambda kv: -kv[1])[:6]
    for r, c in top:
        print(f"   gb[{c}] {r[:110]}")


if __name__ == "__main__":
    main()
