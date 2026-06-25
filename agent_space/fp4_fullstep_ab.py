#!/usr/bin/env python
"""Interleaved A/B full-step driver: spawns a FRESH subprocess per (config, repeat)
so every measurement gets a clean GPU + clean dynamo cache (kills the cross-config
contamination in step_compare_compile_native.py, which runs all configs in one
process with a persistent class-level forward patch + shared dynamo cache).

Interleaves bf16/fp4 to average out clock-boost drift (per the documented
S512 cross-impl noise). Reports per-config median-of-medians.
"""
from __future__ import annotations
import json, os, statistics, subprocess, sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
PY = sys.executable
REPEATS = int(os.environ.get("AB_REPEATS", "3"))


def run_one(cfg):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(_HERE.parent / "src")
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = "6"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    p = subprocess.run([PY, str(_HERE / "fp4_fullstep_clean.py"), cfg],
                       capture_output=True, text=True, env=env, timeout=900)
    med = None
    for line in (p.stdout + p.stderr).splitlines():
        if line.startswith("CONFIG="):
            # CONFIG=fp4  full step median 156.783 ms ...
            import re
            m = re.search(r"median ([\d.]+) ms", line)
            gb = re.search(r"graph_breaks=(\d+)", line)
            ug = re.search(r"unique_graphs=(\d+)", line)
            med = (float(m.group(1)), int(gb.group(1)) if gb else -1,
                   int(ug.group(1)) if ug else -1, line.strip())
    if med is None:
        err = [l for l in (p.stdout+p.stderr).splitlines() if "Error" in l or "OOM" in l]
        return None, (err[-1] if err else "no result")
    return med, None


def main():
    results = {"bf16": [], "fp4": []}
    for r in range(REPEATS):
        for cfg in ("bf16", "fp4"):
            res, err = run_one(cfg)
            if res is None:
                print(f"repeat {r} {cfg}: FAILED {err}", flush=True)
                continue
            med, gb, ug, line = res
            results[cfg].append(med)
            print(f"repeat {r} {cfg:4s}: {med:8.3f} ms [gb={gb} graphs={ug}]", flush=True)

    print("\n=== SUMMARY (median of repeats) ===")
    out = {}
    for cfg, vals in results.items():
        if vals:
            out[cfg] = statistics.median(vals)
            print(f"  {cfg:4s}: median {out[cfg]:.2f} ms  (n={len(vals)}, all={[round(v,1) for v in vals]})")
    if "bf16" in out and "fp4" in out:
        delta = (out["fp4"] - out["bf16"]) / out["bf16"] * 100
        print(f"\n  fp4_matmul vs bf16: {delta:+.1f}% full step")
    Path(_HERE / "fp4_fullstep_ab.json").write_text(json.dumps(
        {"results": results, "summary": out}, indent=2))


if __name__ == "__main__":
    main()
