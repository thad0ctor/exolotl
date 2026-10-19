from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import yaml

from axolotl.integrations.protrain.validation import runner


def test_parse_train_metrics_flags_nonfinite_values() -> None:
    text = "{'loss': '1.25', 'grad_norm': 'nan'}\n{'loss': 'inf'}"

    issues, metrics = runner.numerical_stability_issues(text, require_loss=True)

    assert metrics["losses"] == [1.25, math.inf]
    assert any("loss[1] is non-finite" in issue for issue in issues)
    assert any("grad_norm[0] is non-finite" in issue for issue in issues)


def test_single_gpu_recipe_writer_sets_resume_and_qlora(tmp_path: Path) -> None:
    dest = tmp_path / "resume.yml"
    output_dir = tmp_path / "out"
    checkpoint = output_dir / "checkpoint-50"

    runner.write_single_gpu_yaml(
        recipe_path=runner._DEFAULT_RECIPE,  # noqa: SLF001
        dest_path=dest,
        output_dir=output_dir,
        dataset_prepared_path=tmp_path / "prepared",
        max_steps=60,
        save_steps=60,
        resume_from_checkpoint=checkpoint,
    )

    cfg = yaml.safe_load(dest.read_text())
    assert cfg["adapter"] == "qlora"
    assert cfg["load_in_4bit"] is True
    assert cfg["dataset_prepared_path"] == str(tmp_path / "prepared")
    assert cfg["max_steps"] == 60
    assert cfg["save_steps"] == 60
    assert cfg["resume_from_checkpoint"] == str(checkpoint)
    assert cfg["plugins"] == ["axolotl.integrations.protrain.ProTrainPlugin"]
    assert cfg["protrain_auto_memory"] is True


def test_validation_module_dry_run_json(tmp_path: Path) -> None:
    cmd = [
        sys.executable,
        "-m",
        "axolotl.integrations.protrain.validation",
        "--suite",
        "single-gpu",
        "--dry-run",
        "--json",
        "--work-dir",
        str(tmp_path),
    ]

    proc = subprocess.run(  # nosec B603
        cmd,
        cwd=runner._repo_root(),  # noqa: SLF001
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload[0]["lane"] == "single-gpu"
    assert payload[0]["status"] == "DRY-RUN"
    rendered = " ".join(" ".join(cmd) for cmd in payload[0]["commands"])
    assert "axolotl.cli.train" in rendered


def test_validation_full_dry_run_covers_core_lanes(tmp_path: Path) -> None:
    cmd = [
        sys.executable,
        "-m",
        "axolotl.integrations.protrain.validation",
        "--suite",
        "full",
        "--dry-run",
        "--json",
        "--work-dir",
        str(tmp_path),
    ]

    proc = subprocess.run(  # nosec B603
        cmd,
        cwd=runner._repo_root(),  # noqa: SLF001
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    lanes = [result["lane"] for result in payload]
    assert lanes == [
        "cpu-core",
        "cpu-surface",
        "merge-surface",
        "cpu-full",
        "single-gpu-edge",
        "single-gpu",
        "two-gpu",
    ]
    coverage = "\n".join("\n".join(result["coverage"]) for result in payload)
    assert "Mode A/B/C selector" in coverage
    assert "LoRA merge math" in coverage
    assert "2-rank forced Mode C finite training" in coverage


def test_validation_maintainer_text_output_is_reviewable(tmp_path: Path) -> None:
    cmd = [
        sys.executable,
        "-m",
        "axolotl.integrations.protrain.validation",
        "--suite",
        "maintainer",
        "--dry-run",
        "--work-dir",
        str(tmp_path),
    ]

    proc = subprocess.run(  # nosec B603
        cmd,
        cwd=runner._repo_root(),  # noqa: SLF001
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )

    assert proc.returncode == 0, proc.stderr
    assert "[DRY-RUN] cpu-core" in proc.stdout
    assert "[DRY-RUN] cpu-surface" in proc.stdout
    assert "[DRY-RUN] merge-surface" in proc.stdout
    assert "ProTrain validation matrix" in proc.stdout
    assert "Lane" in proc.stdout
    assert "Hardware" in proc.stdout
    assert "covers:" in proc.stdout
    assert "merge-lora CLI dispatch" in proc.stdout


def test_validation_clean_removes_work_dir(tmp_path: Path) -> None:
    stale = tmp_path / "stale.txt"
    stale.write_text("old")

    cmd = [
        sys.executable,
        "-m",
        "axolotl.integrations.protrain.validation",
        "--clean",
        "--work-dir",
        str(tmp_path),
    ]

    proc = subprocess.run(  # nosec B603
        cmd,
        cwd=runner._repo_root(),  # noqa: SLF001
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )

    assert proc.returncode == 0, proc.stderr
    assert not stale.exists()
    assert not tmp_path.exists()
    assert "[PASS] cleanup" in proc.stdout


def test_validation_keep_cache_reports_reusable_cache(tmp_path: Path) -> None:
    work_dir = tmp_path / "run"
    cache_dir = tmp_path / "run-cache"
    cmd = [
        sys.executable,
        "-m",
        "axolotl.integrations.protrain.validation",
        "--suite",
        "cpu-core",
        "--dry-run",
        "--json",
        "--work-dir",
        str(work_dir),
        "--keep-cache",
    ]

    proc = subprocess.run(  # nosec B603
        cmd,
        cwd=runner._repo_root(),  # noqa: SLF001
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload[0]["metrics"]["cache_dir"] == str(cache_dir)
    assert (cache_dir / "hub").is_dir()
    assert (cache_dir / "datasets").is_dir()


def test_validation_single_gpu_dry_run_places_prepared_data_in_cache(
    tmp_path: Path,
) -> None:
    work_dir = tmp_path / "run"
    cache_dir = tmp_path / "cache"
    cmd = [
        sys.executable,
        "-m",
        "axolotl.integrations.protrain.validation",
        "--suite",
        "single-gpu",
        "--dry-run",
        "--work-dir",
        str(work_dir),
        "--cache-dir",
        str(cache_dir),
    ]

    proc = subprocess.run(  # nosec B603
        cmd,
        cwd=runner._repo_root(),  # noqa: SLF001
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )

    assert proc.returncode == 0, proc.stderr
    train_cfg = yaml.safe_load((work_dir / "single-gpu" / "train.yml").read_text())
    resume_cfg = yaml.safe_load((work_dir / "single-gpu" / "resume.yml").read_text())
    expected = str(cache_dir / "prepared" / "single-gpu")
    assert train_cfg["dataset_prepared_path"] == expected
    assert resume_cfg["dataset_prepared_path"] == expected


def test_validation_base_env_points_hf_caches_at_cache_dir(tmp_path: Path) -> None:
    env = runner._base_env(  # noqa: SLF001
        runner._repo_root(),
        "",
        tmp_path / "cache",  # noqa: SLF001
    )

    assert env["CUDA_VISIBLE_DEVICES"] == ""
    assert env["HF_HOME"] == str(tmp_path / "cache")
    assert env["HF_HUB_CACHE"] == str(tmp_path / "cache" / "hub")
    assert env["HF_DATASETS_CACHE"] == str(tmp_path / "cache" / "datasets")
