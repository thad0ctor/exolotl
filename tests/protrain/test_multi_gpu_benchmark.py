"""Shallow wrapper test around ``scripts/benchmark_multi_gpu.py``.

Runs the benchmark as a subprocess and sanity-checks mode-engagement
(not throughput targets — numbers are hardware-dependent). The default
test cadence skips this because the full benchmark takes ~2.5 minutes
wall-clock; users opt in by dropping the ``skip`` marker or running the
script directly.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


def _nvidia_smi_gpu_count() -> int:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
            timeout=10,
        ).decode("utf-8", errors="replace")
    except (
        FileNotFoundError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ):
        return 0
    return sum(1 for line in out.splitlines() if line.strip())


# Skipped by default — full benchmark takes ~2.5 min end-to-end and
# needs a 4-GPU rig. Set PROTRAIN_RUN_MULTI_GPU_BENCH=1 to opt in:
#   PROTRAIN_RUN_MULTI_GPU_BENCH=1 \
#       CUDA_VISIBLE_DEVICES=1,4,5,7 CUDA_DEVICE_ORDER=PCI_BUS_ID \
#       pytest tests/protrain/test_multi_gpu_benchmark.py -m slow
@pytest.mark.slow
@pytest.mark.gpu
def test_benchmark_multi_gpu_runs(tmp_path) -> None:
    if os.environ.get("PROTRAIN_RUN_MULTI_GPU_BENCH") != "1":
        pytest.skip(
            "PROTRAIN_RUN_MULTI_GPU_BENCH not set — full multi-GPU "
            "benchmark takes ~2.5 min and needs a 4-GPU rig. Set the "
            "env var to 1 to opt in."
        )
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("peft")

    gpu_count = _nvidia_smi_gpu_count()
    if gpu_count < 4:
        pytest.skip(f"requires >= 4 GPUs; nvidia-smi reports {gpu_count}")

    # Repo root — the script lives at scripts/benchmark_multi_gpu.py
    # and writes its results file to the same directory. To avoid
    # mutating the checked-in results file we run from a tmp_path
    # copy of the script; the JSON output file will land next to the
    # script (i.e. inside tmp_path).
    repo_root = Path(__file__).resolve().parents[2]
    src_script = repo_root / "scripts" / "benchmark_multi_gpu.py"
    assert src_script.exists(), f"missing benchmark script at {src_script}"

    script_copy = tmp_path / "benchmark_multi_gpu.py"
    script_copy.write_text(src_script.read_text())

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "1,4,5,7"
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    proc = subprocess.run(
        [sys.executable, str(script_copy)],
        env=env,
        cwd=str(tmp_path),
        check=False,
        capture_output=True,
        text=True,
        timeout=1200,
    )
    assert proc.returncode == 0, (
        f"benchmark exited {proc.returncode}\n"
        f"stdout tail:\n{proc.stdout[-3000:]}\n"
        f"stderr tail:\n{proc.stderr[-3000:]}"
    )

    json_path = tmp_path / "multi_gpu_benchmark_results.json"
    assert json_path.exists(), "benchmark did not write the JSON results file"
    payload = json.loads(json_path.read_text())
    summaries = {s["mode"]: s for s in payload["summaries"]}

    # (1) Every mode completed.
    for mode in ("single", "ddp", "replicated", "zero3"):
        assert mode in summaries, f"mode {mode!r} missing from benchmark output"
        assert summaries[mode]["throughput_samples_per_s"] > 0, (
            f"mode {mode!r} produced zero throughput"
        )

    # (2) Sharding actually saves CPU: ZeRO-3 per-rank CPU bytes
    # should be well below the replicated-mode footprint. Threshold
    # 0.4 gives headroom over the ideal 1/world_size = 0.25 to absorb
    # allocator / alignment overhead.
    z3_cpu = summaries["zero3"]["cpu_pinned_bytes_max"]
    rep_cpu = summaries["replicated"]["cpu_pinned_bytes_max"]
    assert rep_cpu > 0, "replicated mode reported zero CPU bytes — mode did not engage"
    assert z3_cpu <= 0.4 * rep_cpu, (
        f"ZeRO-3 CPU footprint {z3_cpu / 1e9:.3f} GB not <= 0.4 x replicated "
        f"{rep_cpu / 1e9:.3f} GB (sharding may not have engaged)"
    )

    # (3) DDP scaling invariant (M6 threshold): DDP throughput > 2.5x
    # single-rank. Same bar the test_protrain_4gpu_throughput_scaling
    # test asserts.
    single_tp = summaries["single"]["throughput_samples_per_s"]
    ddp_tp = summaries["ddp"]["throughput_samples_per_s"]
    assert ddp_tp > 2.5 * single_tp, (
        f"DDP throughput {ddp_tp:.2f} not > 2.5 x single-rank {single_tp:.2f}"
    )


# ---------------------------------------------------------------------------
# Lightweight JSON-validation tests
# ---------------------------------------------------------------------------
#
# Run on every CI invocation. Validate the LATEST checked-in benchmark
# results against the design-target scaling thresholds (DESIGN.md). When
# the JSON is missing — typically a fresh checkout — the tests skip
# rather than fail. Operators run ``scripts/benchmark_multi_gpu.py``
# periodically (after any Mode A/B/C path change) to refresh the JSON,
# and these tests certify the recorded numbers still meet the
# thresholds before the change is shipped.

_BENCH_JSON_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "multi_gpu_benchmark_results.json"
)

# The recorded thresholds below were calibrated on the canonical 4x RTX 3090
# rig listed in the JSON's ``workload.gpus`` field. A developer running the
# benchmark on a different GPU mix (different SKU, different count, mixed
# bandwidth) and committing the new JSON would silently regress these tests
# against thresholds that no longer apply. Skip the JSON-comparison tests
# unless the workload matches the canonical rig.
_CANONICAL_BENCH_GPUS = "1,4,5,7 (RTX 3090)"


def _load_summaries() -> dict[str, dict]:
    if not _BENCH_JSON_PATH.exists():
        pytest.skip(
            f"{_BENCH_JSON_PATH.name} not found — run "
            "`scripts/benchmark_multi_gpu.py` on a 4-GPU rig first to "
            "generate it (~150s)."
        )
    raw = json.loads(_BENCH_JSON_PATH.read_text())
    recorded_gpus = raw.get("workload", {}).get("gpus")
    if recorded_gpus != _CANONICAL_BENCH_GPUS:
        pytest.skip(
            f"benchmark JSON workload.gpus={recorded_gpus!r} != "
            f"canonical {_CANONICAL_BENCH_GPUS!r}; the recorded thresholds "
            "in this file were calibrated on the canonical 4x RTX 3090 rig "
            "and don't apply to other GPU mixes. Re-run the benchmark on "
            "the canonical rig to refresh the JSON, or adjust thresholds "
            "explicitly if you're changing the calibration target."
        )
    return {s["mode"]: s for s in raw.get("summaries", [])}


def test_recorded_mode_a_ddp_scaling_at_least_3x() -> None:
    """DDP composition: recorded throughput >= 3.0x single-rank.

    The plugin's job in Mode A is to NOT interfere with DDP's bucketed
    all-reduce. A regression here typically means the per-param
    all_reduce path is firing twice (DDP + chunk manager) — burning
    bandwidth for double-counted gradient sync.
    """
    summaries = _load_summaries()
    if "single" not in summaries or "ddp" not in summaries:
        pytest.skip("benchmark JSON missing 'single' or 'ddp' mode")
    baseline = summaries["single"]["throughput_samples_per_s"]
    ddp_tp = summaries["ddp"]["throughput_samples_per_s"]
    s = ddp_tp / baseline
    assert s >= 3.0, (
        f"recorded Mode A (DDP) scaling regressed: {s:.2f}x vs >=3.0x "
        "target. Re-run scripts/benchmark_multi_gpu.py and inspect the "
        "iter_times — the ``skip_internal_grad_reduce=True`` plumb-through "
        "may have broken."
    )


def test_recorded_mode_b_replicated_scaling_at_least_1_2x() -> None:
    """Replicated CPU offload: recorded throughput >= 1.2x single-rank."""
    summaries = _load_summaries()
    if "single" not in summaries or "replicated" not in summaries:
        pytest.skip("benchmark JSON missing 'single' or 'replicated' mode")
    baseline = summaries["single"]["throughput_samples_per_s"]
    rep_tp = summaries["replicated"]["throughput_samples_per_s"]
    s = rep_tp / baseline
    assert s >= 1.2, (
        f"recorded Mode B (replicated CPU offload) scaling regressed: "
        f"{s:.2f}x vs >=1.2x target."
    )


def test_recorded_mode_c_zero3_functional() -> None:
    """ZeRO-3 sharded must produce positive throughput (no scaling floor)."""
    summaries = _load_summaries()
    if "zero3" not in summaries:
        pytest.skip("benchmark JSON missing 'zero3' mode")
    tp = summaries["zero3"]["throughput_samples_per_s"]
    assert tp > 0.0, (
        f"Mode C (ZeRO-3 sharded) throughput non-positive: {tp}. "
        "The path failed to make forward progress."
    )


def test_recorded_pinned_cpu_drops_with_sharding() -> None:
    """Replicated/sharded pinned-CPU ratio >= 3x on 4 GPUs.

    Replicated holds each non-persistent chunk on every rank; sharded
    holds 1/world_size. On 4x 3090 the recorded ratio is ~4.0x. Below
    3x means sharding stopped partitioning chunks correctly.
    """
    summaries = _load_summaries()
    if "replicated" not in summaries or "zero3" not in summaries:
        pytest.skip("benchmark JSON missing 'replicated' or 'zero3' mode")
    rep_pinned = summaries["replicated"]["cpu_pinned_bytes_max"]
    z3_pinned = summaries["zero3"]["cpu_pinned_bytes_max"]
    if z3_pinned == 0:
        pytest.fail("zero3 pinned-CPU dropped to 0 — sharded chunks not allocated")
    ratio = rep_pinned / z3_pinned
    assert ratio >= 3.0, (
        f"replicated/sharded pinned-CPU ratio regressed: {ratio:.2f}x vs >=3.0x target."
    )
