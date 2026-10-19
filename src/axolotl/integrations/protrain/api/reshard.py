"""Core reshard logic for ProTrain Mode-C optimizer state."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sys
from typing import Any

import torch

# Constants mirrored from api/checkpoint.py; avoid import to keep CLI loader light.
METADATA_FILENAME = "metadata.json"
GPU_OPTIM_FILENAME = "gpu_optim.pt"
CPU_OPTIM_DIRNAME = "cpu_optim"
SCHEMA_FORMAT_VERSION = 4
SUPPORTED_FORMAT_VERSIONS = {2, 3, 4}
SAVE_MODE_SHARDED = "sharded"
CHUNK_SHARD_FILE_RE = re.compile(r"^chunk_(\d+)_rank_(\d+)\.pt$")
GPU_OPTIM_RANK_FILE_RE = re.compile(r"^gpu_optim_rank_(\d+)\.pt$")
GPU_OPTIM_RANK_META_SUFFIX = ".meta.json"

_DTYPE_NAME_TO_TORCH: dict[str, torch.dtype] = {
    "torch.float16": torch.float16,
    "torch.bfloat16": torch.bfloat16,
    "torch.float32": torch.float32,
    "torch.float64": torch.float64,
    "torch.float": torch.float32,
    "torch.half": torch.float16,
    "torch.double": torch.float64,
}


# ---- Layout signature ------------------------------------------------------


def _layout_signature_from_fingerprint(fingerprint: dict[str, Any]) -> str:
    """SHA-256 over a layout fingerprint dict (must stay byte-compatible with api version)."""
    payload = json.dumps(fingerprint, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---- Per-region reshard ----------------------------------------------------


def _padded_region_bytes(region_bytes: int, elem_size: int, world_size: int) -> int:
    """Pad region_bytes so each rank owns a whole number of elements."""
    elem_count = (region_bytes + elem_size - 1) // elem_size
    padded_elems = ((elem_count + world_size - 1) // world_size) * world_size
    return padded_elems * elem_size


def _reshard_region_state(
    per_rank_tensors: list[torch.Tensor],
    *,
    region_bytes: int,
    elem_size: int,
    src_world: int,
    dst_world: int,
    region_bytes_padded_old: int | None = None,
    region_bytes_padded_new: int | None = None,
) -> list[torch.Tensor]:
    """Reshard one region's per-rank state tensor (e.g. ``exp_avg``) from
    ``src_world`` ranks to ``dst_world`` ranks.

    Inputs
    ------
    per_rank_tensors:
        List of length ``src_world`` of 1-D tensors, all with the same
        dtype and length ``shard_bytes_old / elem_size``.
    region_bytes:
        Un-padded valid bytes of the region (constant across world
        sizes).
    elem_size:
        Element size for the region's chunk-storage dtype, not necessarily
        the optimizer-state tensor dtype.
    region_bytes_padded_old / region_bytes_padded_new:
        If supplied (typically from the saved metadata), use these
        directly instead of recomputing — guards against any drift
        between the script's pad formula and the runtime's.

    Output
    ------
    List of length ``dst_world`` of 1-D tensors, all with the same dtype
    as the inputs and length ``shard_bytes_new / elem_size``.
    """
    if len(per_rank_tensors) != src_world:
        raise RuntimeError(
            f"reshard: expected {src_world} per-rank tensors, got "
            f"{len(per_rank_tensors)}"
        )
    dtype = per_rank_tensors[0].dtype
    for t in per_rank_tensors:
        if t.dtype != dtype:
            raise RuntimeError(
                f"reshard: per-rank tensors have inconsistent dtypes "
                f"({dtype} vs {t.dtype}) — refusing to mix"
            )

    if region_bytes_padded_old is None:
        region_bytes_padded_old = _padded_region_bytes(
            region_bytes, elem_size, src_world
        )
    if region_bytes_padded_new is None:
        region_bytes_padded_new = _padded_region_bytes(
            region_bytes, elem_size, dst_world
        )

    # Divisibility checks use the combined divisor (world * elem_size) to catch element truncation.
    if region_bytes % elem_size != 0:
        raise RuntimeError(
            f"reshard: region_bytes={region_bytes} is not divisible by "
            f"elem_size={elem_size}"
        )
    src_unit = src_world * elem_size
    dst_unit = dst_world * elem_size
    if region_bytes_padded_old % src_unit != 0:
        raise RuntimeError(
            f"reshard: region_bytes_padded_old={region_bytes_padded_old} "
            f"is not divisible by src_world * elem_size ({src_unit}); "
            f"cannot split into whole elements per rank"
        )
    if region_bytes_padded_new % dst_unit != 0:
        raise RuntimeError(
            f"reshard: region_bytes_padded_new={region_bytes_padded_new} "
            f"is not divisible by dst_world * elem_size ({dst_unit}); "
            f"cannot split into whole elements per rank"
        )

    expected_old_shard_numel = region_bytes_padded_old // src_unit
    for r, t in enumerate(per_rank_tensors):
        if t.numel() != expected_old_shard_numel:
            raise RuntimeError(
                f"reshard: per-rank tensor {r} has numel={t.numel()}, "
                f"expected {expected_old_shard_numel} "
                f"(region_bytes_padded={region_bytes_padded_old}, "
                f"elem_size={elem_size}, src_world={src_world})"
            )

    # Concatenate, carry only valid prefix forward; free old before allocating new.
    full_old = torch.cat(per_rank_tensors, dim=0).contiguous()
    valid_numel = region_bytes // elem_size
    valid_prefix = full_old[:valid_numel].clone()
    del full_old
    new_padded_numel = region_bytes_padded_new // elem_size
    full_new = torch.zeros(new_padded_numel, dtype=dtype)
    full_new[:valid_numel] = valid_prefix
    del valid_prefix

    new_shard_numel = region_bytes_padded_new // dst_unit
    out: list[torch.Tensor] = []
    for r in range(dst_world):
        start = r * new_shard_numel
        end = start + new_shard_numel
        # Clone so each slice owns its own storage (defensive for test inspection).
        out.append(full_new[start:end].clone())
    return out


# ---- Driver ---------------------------------------------------------------


def _read_metadata(src_dir: str) -> dict[str, Any]:
    meta_path = os.path.join(src_dir, METADATA_FILENAME)
    if not os.path.isfile(meta_path):
        raise RuntimeError(f"reshard: missing metadata at {meta_path!r}")
    with open(meta_path, encoding="utf-8") as f:
        return json.load(f)


def _validate_src_metadata(meta: dict[str, Any]) -> None:
    fmt = int(meta.get("format_version", 0))
    if fmt not in SUPPORTED_FORMAT_VERSIONS:
        raise RuntimeError(
            f"reshard: source format_version={fmt}, expected "
            f"one of {sorted(SUPPORTED_FORMAT_VERSIONS)}. "
            "Only Phase-2 sharded saves with layout_fingerprint are supported."
        )
    save_mode = meta.get("protrain_save_mode")
    if save_mode != SAVE_MODE_SHARDED:
        raise RuntimeError(
            f"reshard: source save_mode={save_mode!r}, expected "
            f"{SAVE_MODE_SHARDED!r}. Mode-B replicated saves do not need "
            "resharding (the load path tolerates world_size drift "
            "natively — see CHECKPOINT_DESIGN_PHASE2.md §4.1 Option B)."
        )
    if "regions_per_chunk" not in meta:
        raise RuntimeError(
            "reshard: source metadata missing 'regions_per_chunk'. The "
            "save predates Mode-C support or the file is corrupt."
        )
    if "layout_fingerprint" not in meta:
        raise RuntimeError(
            "reshard: source metadata missing 'layout_fingerprint'. The "
            "save predates the offline reshard support — re-save under a "
            "newer ProTrain build to capture the raw layout fields."
        )
    try:
        src_world = int(meta["protrain_world_size"])
    except Exception as exc:
        raise RuntimeError(
            "reshard: source metadata missing valid 'protrain_world_size'."
        ) from exc
    if src_world < 1:
        raise RuntimeError(
            f"reshard: invalid protrain_world_size={src_world}; expected >= 1."
        )


def _scan_src_chunks(src_dir: str, src_world: int) -> dict[int, list[str]]:
    """Return ``{chunk_id: [path_for_rank0, path_for_rank1, ...]}``."""
    cpu_dir = os.path.join(src_dir, CPU_OPTIM_DIRNAME)
    if not os.path.isdir(cpu_dir):
        return {}
    by_chunk: dict[int, dict[int, str]] = {}
    for name in sorted(os.listdir(cpu_dir)):
        m = CHUNK_SHARD_FILE_RE.match(name)
        if m is None:
            raise RuntimeError(
                f"reshard: unexpected file {name!r} in {cpu_dir!r} — "
                "Mode-C cpu_optim/ must contain only chunk_<N>_rank_<R>.pt"
            )
        cid = int(m.group(1))
        rank = int(m.group(2))
        if rank < 0 or rank >= src_world:
            raise RuntimeError(
                f"reshard: file {name!r} rank ordinal {rank} outside "
                f"[0, {src_world}) — corrupt source dir."
            )
        by_chunk.setdefault(cid, {})[rank] = os.path.join(cpu_dir, name)

    out: dict[int, list[str]] = {}
    for cid, by_rank in by_chunk.items():
        if set(by_rank.keys()) != set(range(src_world)):
            missing = set(range(src_world)) - set(by_rank.keys())
            raise RuntimeError(
                f"reshard: chunk {cid} missing per-rank shards for "
                f"ranks {sorted(missing)}"
            )
        out[cid] = [by_rank[r] for r in range(src_world)]
    return out


def _scan_gpu_rank_files(src_dir: str, src_world: int) -> list[str]:
    by_rank: dict[int, str] = {}
    for name in sorted(os.listdir(src_dir)):
        m = GPU_OPTIM_RANK_FILE_RE.match(name)
        if m is None:
            continue
        rank = int(m.group(1))
        if rank < 0 or rank >= src_world:
            raise RuntimeError(
                f"reshard: file {name!r} rank ordinal {rank} outside "
                f"[0, {src_world}) — corrupt source dir."
            )
        by_rank[rank] = os.path.join(src_dir, name)
    if not by_rank:
        return []
    if set(by_rank.keys()) != set(range(src_world)):
        missing = set(range(src_world)) - set(by_rank.keys())
        raise RuntimeError(
            "reshard: persistent gpu optimizer partition is missing "
            f"per-rank files for ranks {sorted(missing)}"
        )
    return [by_rank[r] for r in range(src_world)]


def _clone_state_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.clone()
    return value


def _param_group_metas(param_groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{k: v for k, v in group.items() if k != "params"} for group in param_groups]


def _local_param_count(param_groups: list[dict[str, Any]]) -> int:
    max_idx = -1
    for group in param_groups:
        for idx in group.get("params", []):
            max_idx = max(max_idx, int(idx))
    return max_idx + 1


def _persistent_param_names_from_metadata(metadata: dict[str, Any]) -> list[str] | None:
    fingerprint = metadata.get("layout_fingerprint")
    if not isinstance(fingerprint, dict):
        return None
    chunks = fingerprint.get("chunks")
    persistent_ids = metadata.get("protrain_persistent_ids")
    if chunks is None or persistent_ids is None:
        return None

    try:
        names: list[str] = []
        seen: set[str] = set()
        for cid in persistent_ids:
            for raw_name in chunks[int(cid)]:
                name = str(raw_name)
                if name in seen:
                    continue
                seen.add(name)
                names.append(name)
        return names
    except (IndexError, TypeError, ValueError):
        return None


def _load_gpu_rank_metas(src_paths: list[str]) -> list[dict[str, Any]]:
    metas: list[dict[str, Any]] = []
    missing: list[str] = []
    for path in src_paths:
        meta_path = path + GPU_OPTIM_RANK_META_SUFFIX
        if not os.path.isfile(meta_path):
            missing.append(os.path.basename(meta_path))
            continue
        with open(meta_path) as f:
            metas.append(json.load(f))
    if missing:
        raise RuntimeError(
            "reshard: persistent gpu optimizer cross-world reshard requires "
            "per-rank parameter-name sidecars written by the current save "
            "format; missing " + ", ".join(sorted(missing)) + ". Resume with "
            "the original world_size or re-save the checkpoint with a build "
            "that writes gpu_optim_rank_<R>.pt.meta.json."
        )
    return metas


def _rank_meta_group_names(meta: dict[str, Any], *, rank: int) -> list[list[str]]:
    if meta.get("format") != "protrain_gpu_optim_rank_param_names_v1":
        raise RuntimeError(
            f"reshard: gpu optimizer rank {rank} metadata has unsupported "
            f"format {meta.get('format')!r}"
        )
    groups = meta.get("param_groups_param_names")
    if not isinstance(groups, list):
        raise RuntimeError(
            f"reshard: gpu optimizer rank {rank} metadata is missing "
            "'param_groups_param_names'"
        )
    return [[str(name) for name in group] for group in groups]


def _rank_meta_key_to_name(meta: dict[str, Any], *, rank: int) -> dict[int, str]:
    raw = meta.get("param_names_by_state_index")
    if not isinstance(raw, dict):
        raise RuntimeError(
            f"reshard: gpu optimizer rank {rank} metadata is missing "
            "'param_names_by_state_index'"
        )
    return {int(key): str(value) for key, value in raw.items()}


def _reshard_persistent_gpu_optim(
    src_dir: str,
    dst_dir: str,
    *,
    src_world: int,
    dst_world: int,
    metadata: dict[str, Any],
) -> None:
    """Repartition round-robin persistent GPU optimizer state.

    Saved optimizer state_dict keys are post-param-group integer ids, not the
    original round-robin local order. The rank sidecars map those ids back to
    stable parameter names so state moves by parameter identity.
    """
    if metadata.get("protrain_persistent_huge_param_shards"):
        raise RuntimeError(
            "reshard: cross-world persistent optimizer reshard does not yet "
            "support huge-param within-shard saves "
            "('protrain_persistent_huge_param_shards' present). Re-save with "
            "a higher protrain_persistent_huge_param_threshold_bytes so "
            "persistent params use the round-robin whole-param path, or "
            "resume with the original world_size."
        )

    src_paths = _scan_gpu_rank_files(src_dir, src_world)
    if not src_paths:
        raise RuntimeError(
            "reshard: persistent gpu optimizer partition metadata is present, "
            "but no gpu_optim_rank_<R>.pt files were found. Resume with the "
            "original world_size or re-save the checkpoint with persistent "
            "optimizer rank files."
        )

    per_rank = [torch.load(p, map_location="cpu", weights_only=True) for p in src_paths]
    per_rank_meta = _load_gpu_rank_metas(src_paths)
    for rank, sd in enumerate(per_rank):
        if "state" not in sd or "param_groups" not in sd:
            raise RuntimeError(
                f"reshard: gpu_optim_rank_{rank}.pt missing 'state' or 'param_groups'"
            )

    group_metas = _param_group_metas(per_rank[0]["param_groups"])
    source_group_names: list[list[list[str]]] = []
    for rank, (sd, meta) in enumerate(zip(per_rank, per_rank_meta, strict=True)):
        if _param_group_metas(sd["param_groups"]) != group_metas:
            raise RuntimeError(
                "reshard: persistent gpu optimizer param-group hyperparams "
                f"differ between rank 0 and rank {rank}"
            )
        group_names = _rank_meta_group_names(meta, rank=rank)
        if len(group_names) != len(sd["param_groups"]):
            raise RuntimeError(
                f"reshard: gpu optimizer rank {rank} metadata has "
                f"{len(group_names)} param group(s), but state_dict has "
                f"{len(sd['param_groups'])}"
            )
        for group_idx, (names, group) in enumerate(
            zip(group_names, sd["param_groups"], strict=True)
        ):
            if len(names) != len(group.get("params", [])):
                raise RuntimeError(
                    f"reshard: gpu optimizer rank {rank} group {group_idx} "
                    "metadata size does not match state_dict param count"
                )
        source_group_names.append(group_names)

    local_counts = [_local_param_count(sd["param_groups"]) for sd in per_rank]
    total_params = sum(local_counts)
    param_names = _persistent_param_names_from_metadata(metadata)
    if param_names is None:
        raise RuntimeError(
            "reshard: persistent gpu optimizer cross-world reshard requires "
            "layout_fingerprint.chunks and protrain_persistent_ids metadata. "
            "Resume with the original world_size or re-save the checkpoint "
            "with current metadata."
        )
    if param_names is not None and len(param_names) != total_params:
        raise RuntimeError(
            "reshard: persistent parameter metadata length mismatch: "
            f"metadata has {len(param_names)} names, optimizer shards cover "
            f"{total_params} params"
        )
    expected_counts = [
        len(range(rank, total_params, src_world)) for rank in range(src_world)
    ]
    if local_counts != expected_counts:
        raise RuntimeError(
            "reshard: persistent gpu optimizer rank-local param counts "
            f"{local_counts} do not match round-robin counts {expected_counts} "
            f"for src_world={src_world}, total_params={total_params}"
        )

    name_to_global = {name: idx for idx, name in enumerate(param_names)}
    if len(name_to_global) != len(param_names):
        raise RuntimeError(
            "reshard: persistent parameter metadata contains duplicate names"
        )

    name_to_group: dict[str, int] = {}
    state_by_name: dict[str, dict[str, Any]] = {}
    for src_rank, (sd, meta, group_names) in enumerate(
        zip(per_rank, per_rank_meta, source_group_names, strict=True)
    ):
        rank_key_to_name = _rank_meta_key_to_name(meta, rank=src_rank)
        seen_rank_names: list[str] = []
        for group_idx, names in enumerate(group_names):
            for name in names:
                if name not in name_to_global:
                    raise RuntimeError(
                        "reshard: gpu optimizer rank "
                        f"{src_rank} metadata references unknown persistent "
                        f"parameter {name!r}"
                    )
                global_idx = name_to_global[name]
                if global_idx % src_world != src_rank:
                    raise RuntimeError(
                        "reshard: gpu optimizer rank "
                        f"{src_rank} metadata claims parameter {name!r} "
                        f"(global index {global_idx}), owned by rank "
                        f"{global_idx % src_world} under src_world={src_world}"
                    )
                seen_rank_names.append(name)
                prior = name_to_group.setdefault(name, group_idx)
                if prior != group_idx:
                    raise RuntimeError(
                        "reshard: persistent parameter "
                        f"{name!r} appears in multiple param groups "
                        f"({prior} and {group_idx})"
                    )

        expected_rank_names = param_names[src_rank::src_world]
        if len(seen_rank_names) != len(expected_rank_names) or set(
            seen_rank_names
        ) != set(expected_rank_names):
            raise RuntimeError(
                "reshard: gpu optimizer rank "
                f"{src_rank} metadata parameter coverage does not match the "
                "saved layout fingerprint"
            )

        for state_idx_raw, state in sd["state"].items():
            state_idx = int(state_idx_raw)
            if state_idx not in rank_key_to_name:
                raise RuntimeError(
                    "reshard: gpu optimizer rank "
                    f"{src_rank} state key {state_idx} is missing from "
                    "the parameter-name sidecar"
                )
            name = rank_key_to_name[state_idx]
            if name in state_by_name:
                raise RuntimeError(
                    f"reshard: duplicate persistent optimizer state for {name!r}"
                )
            state_by_name[name] = {
                key: _clone_state_value(value) for key, value in state.items()
            }

    if set(name_to_group) != set(param_names):
        missing = set(param_names) - set(name_to_group)
        raise RuntimeError(
            "reshard: persistent optimizer param-group coverage is incomplete; "
            f"missing names {sorted(missing)[:10]}"
        )

    for dst_rank in range(dst_world):
        owned_names = param_names[dst_rank::dst_world]
        new_name_to_key: dict[str, int] = {}
        new_group_names: list[list[str]] = []
        new_groups: list[dict[str, Any]] = []
        next_saved_key = 0
        for group_idx, meta in enumerate(group_metas):
            names = [name for name in owned_names if name_to_group[name] == group_idx]
            params: list[int] = []
            for name in names:
                new_name_to_key[name] = next_saved_key
                params.append(next_saved_key)
                next_saved_key += 1
            new_group_names.append(names)
            new_groups.append(dict(meta) | {"params": params})

        new_state: dict[int, dict[str, Any]] = {}
        for name in owned_names:
            state = state_by_name.get(name)
            if state is None:
                continue
            new_state[new_name_to_key[name]] = {
                key: _clone_state_value(value) for key, value in state.items()
            }

        dst_path = os.path.join(dst_dir, f"gpu_optim_rank_{dst_rank}.pt")
        torch.save({"state": new_state, "param_groups": new_groups}, dst_path)
        dst_meta = {
            "format": "protrain_gpu_optim_rank_param_names_v1",
            "rank": int(dst_rank),
            "persistent_param_names_full": param_names,
            "param_names_by_state_index": {
                str(idx): name for name, idx in sorted(new_name_to_key.items())
            },
            "param_groups_param_names": new_group_names,
        }
        with open(dst_path + GPU_OPTIM_RANK_META_SUFFIX, "w") as f:
            json.dump(dst_meta, f, indent=2, sort_keys=True)


def _region_expected_shard_numel(
    region_meta: dict[str, Any], *, world_size: int
) -> int:
    region_dtype = _DTYPE_NAME_TO_TORCH[region_meta["dtype"]]
    elem_size = region_dtype.itemsize
    region_bytes_padded = int(region_meta["region_bytes_padded"])
    unit = int(world_size) * elem_size
    if region_bytes_padded % unit != 0:
        raise RuntimeError(
            f"reshard: region_bytes_padded={region_bytes_padded} is not divisible "
            f"by world_size * elem_size ({unit})"
        )
    return region_bytes_padded // unit


def _map_cpu_state_keys_to_regions(
    *,
    cid: int,
    regs: list[dict[str, Any]],
    rank0_state: dict[Any, dict[str, Any]],
    src_world: int,
) -> dict[Any, int]:
    expected_by_region = {
        region_idx: _region_expected_shard_numel(region_meta, world_size=src_world)
        for region_idx, region_meta in enumerate(regs)
    }
    unused = set(expected_by_region)
    mapping: dict[Any, int] = {}

    for state_key in sorted(rank0_state):
        state_entry = rank0_state[state_key]
        moment = state_entry.get("exp_avg")
        if not isinstance(moment, torch.Tensor):
            raise RuntimeError(
                f"reshard: chunk {cid} state key {state_key!r} is missing tensor "
                "'exp_avg'; cannot map optimizer state to dtype regions"
            )
        numel = int(moment.numel())
        candidates = [
            region_idx
            for region_idx in sorted(unused)
            if expected_by_region[region_idx] == numel
        ]
        if len(candidates) != 1:
            raise RuntimeError(
                f"reshard: chunk {cid} state key {state_key!r} has numel={numel}, "
                f"but matches {len(candidates)} unused region(s). CPU optimizer "
                "state order cannot be inferred safely from this checkpoint; "
                "re-save with unique region shard sizes or resume with the "
                "original world_size."
            )
        region_idx = candidates[0]
        mapping[state_key] = region_idx
        unused.remove(region_idx)

    if unused:
        raise RuntimeError(
            f"reshard: chunk {cid} optimizer state covers {len(mapping)} region(s), "
            f"but metadata has unused region index/indices {sorted(unused)}"
        )
    return mapping


def reshard_mode_c_shards(
    src_dir: str,
    dst_dir: str,
    target_world_size: int,
    *,
    log_fn=None,
) -> None:
    """Read src_dir, write dst_dir at target_world_size ranks (refuses non-empty dst)."""
    if target_world_size < 1:
        raise ValueError(f"target_world_size must be >= 1 (got {target_world_size})")

    if log_fn is None:
        log_fn = lambda msg: print(msg, file=sys.stderr)  # noqa: E731

    meta = _read_metadata(src_dir)
    _validate_src_metadata(meta)

    src_world = int(meta["protrain_world_size"])
    if src_world == target_world_size:
        # No reshape; copy to emit a fresh dst_dir per the contract.
        log_fn(
            f"reshard: src_world == target_world == {src_world}; "
            "copying source directory verbatim"
        )
        if os.path.abspath(src_dir) == os.path.abspath(dst_dir):
            raise RuntimeError("reshard: dst_dir must differ from src_dir")
        if os.path.isdir(dst_dir) and os.listdir(dst_dir):
            raise RuntimeError(
                f"reshard: refusing to overwrite non-empty dst_dir {dst_dir!r}"
            )
        shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
        log_fn(f"reshard: copied {src_dir!r} to {dst_dir!r} (no reshape needed)")
        return

    log_fn(
        f"reshard: src={src_dir!r} dst={dst_dir!r} "
        f"src_world={src_world} target_world={target_world_size}"
    )

    if os.path.abspath(src_dir) == os.path.abspath(dst_dir):
        raise RuntimeError("reshard: dst_dir must differ from src_dir")
    if os.path.isdir(dst_dir) and os.listdir(dst_dir):
        raise RuntimeError(
            f"reshard: refusing to overwrite non-empty dst_dir {dst_dir!r}"
        )
    os.makedirs(dst_dir, exist_ok=True)
    cpu_dst_dir = os.path.join(dst_dir, CPU_OPTIM_DIRNAME)

    persistent_partition_active = (
        meta.get("protrain_persistent_partition_version") is not None
    )

    # gpu_optim.pt is rank-independent in Mode-C; partitioned saves need
    # rank-file reassignment to the target round-robin owner set.
    src_gpu = os.path.join(src_dir, GPU_OPTIM_FILENAME)
    if persistent_partition_active:
        _reshard_persistent_gpu_optim(
            src_dir,
            dst_dir,
            src_world=src_world,
            dst_world=target_world_size,
            metadata=meta,
        )
    elif os.path.isfile(src_gpu):
        shutil.copyfile(src_gpu, os.path.join(dst_dir, GPU_OPTIM_FILENAME))

    saved_regions: dict[str, list[dict[str, Any]]] = meta["regions_per_chunk"]

    # Only region_bytes_padded and shard_bytes change with world_size.
    new_regions: dict[str, list[dict[str, Any]]] = {}
    for cid_str, regs in saved_regions.items():
        new_list: list[dict[str, Any]] = []
        for r in regs:
            elem_size_int = _DTYPE_NAME_TO_TORCH[r["dtype"]].itemsize
            region_bytes = int(r["region_bytes"])
            new_padded = _padded_region_bytes(
                region_bytes, elem_size_int, target_world_size
            )
            new_shard_bytes = new_padded // target_world_size
            new_list.append(
                {
                    "chunk_offset": int(r["chunk_offset"]),
                    "region_bytes": region_bytes,
                    "region_bytes_padded": int(new_padded),
                    "shard_bytes": int(new_shard_bytes),
                    "dtype": r["dtype"],
                }
            )
        new_regions[cid_str] = new_list

    # Reshard each chunk's per-rank state files.
    chunk_paths = _scan_src_chunks(src_dir, src_world)
    if chunk_paths:
        os.makedirs(cpu_dst_dir, exist_ok=True)

    # Cross-check chunk ids in metadata and on disk.
    saved_cids = set(int(c) for c in saved_regions.keys())
    disk_cids = set(chunk_paths.keys())
    if saved_cids != disk_cids:
        raise RuntimeError(
            "reshard: regions_per_chunk chunk-ids "
            f"{sorted(saved_cids)} disagree with on-disk shard chunk-ids "
            f"{sorted(disk_cids)}"
        )

    for cid in sorted(chunk_paths.keys()):
        per_rank_paths = chunk_paths[cid]
        per_rank_state_dicts = [
            torch.load(p, map_location="cpu", weights_only=True) for p in per_rank_paths
        ]
        regs = saved_regions[str(cid)]

        # Each per-rank state_dict must have one state entry per region.
        rank0_state_keys = set(per_rank_state_dicts[0].get("state", {}).keys())
        if len(rank0_state_keys) != len(regs):
            raise RuntimeError(
                f"reshard: chunk {cid} rank 0 state has {len(rank0_state_keys)} "
                f"entries, expected {len(regs)} (one per region)"
            )
        for r_idx, sd in enumerate(per_rank_state_dicts):
            if "state" not in sd or "param_groups" not in sd:
                raise RuntimeError(
                    f"reshard: chunk {cid} rank {r_idx} state_dict missing "
                    "'state' or 'param_groups' key"
                )
            if set(sd["state"].keys()) != rank0_state_keys:
                raise RuntimeError(
                    f"reshard: chunk {cid} rank {r_idx} state has keys "
                    f"{sorted(sd['state'].keys())}, expected "
                    f"{sorted(rank0_state_keys)}"
                )
        state_key_to_region_idx = _map_cpu_state_keys_to_regions(
            cid=cid,
            regs=regs,
            rank0_state=per_rank_state_dicts[0]["state"],
            src_world=src_world,
        )

        # param_groups + step are rank-replicated; copy from rank-0.
        new_per_rank_states: list[dict[Any, dict[str, Any]]] = [
            {} for _ in range(target_world_size)
        ]
        for optim_state_key, region_idx in state_key_to_region_idx.items():
            region_meta = regs[region_idx]
            region_bytes = int(region_meta["region_bytes"])
            region_dtype = _DTYPE_NAME_TO_TORCH[region_meta["dtype"]]
            elem_size_int = region_dtype.itemsize
            saved_padded_old = int(region_meta["region_bytes_padded"])
            new_padded = new_regions[str(cid)][region_idx]["region_bytes_padded"]

            for state_key in ("exp_avg", "exp_avg_sq"):
                per_rank_inputs = [
                    sd["state"][optim_state_key][state_key]
                    for sd in per_rank_state_dicts
                ]
                per_rank_inputs = [t.flatten() for t in per_rank_inputs]
                new_slices = _reshard_region_state(
                    per_rank_inputs,
                    region_bytes=region_bytes,
                    elem_size=elem_size_int,
                    src_world=src_world,
                    dst_world=target_world_size,
                    region_bytes_padded_old=saved_padded_old,
                    region_bytes_padded_new=int(new_padded),
                )
                for r2, slice_ in enumerate(new_slices):
                    new_per_rank_states[r2].setdefault(optim_state_key, {})[
                        state_key
                    ] = slice_

            # Validate rank-replicated scalars before copying rank-0.
            rank0_region = per_rank_state_dicts[0]["state"][optim_state_key]
            for r_other in range(1, src_world):
                other_region = per_rank_state_dicts[r_other]["state"][optim_state_key]
                if set(other_region.keys()) != set(rank0_region.keys()):
                    raise ValueError(
                        f"reshard: chunk {cid} state key {optim_state_key!r} "
                        f"(region {region_idx}) state keys "
                        f"differ between rank 0 ({sorted(rank0_region.keys())}) "
                        f"and rank {r_other} ({sorted(other_region.keys())})"
                    )
            for k, v in rank0_region.items():
                if k in ("exp_avg", "exp_avg_sq"):
                    continue
                for r_other in range(1, src_world):
                    other_v = per_rank_state_dicts[r_other]["state"][optim_state_key][k]
                    if isinstance(v, torch.Tensor) or isinstance(other_v, torch.Tensor):
                        if not (
                            isinstance(v, torch.Tensor)
                            and isinstance(other_v, torch.Tensor)
                            and torch.equal(v, other_v)
                        ):
                            raise ValueError(
                                f"reshard: chunk {cid} state key "
                                f"{optim_state_key!r} (region {region_idx}) "
                                f"non-moment state key {k!r} differs between "
                                f"rank 0 and rank {r_other}"
                            )
                    elif v != other_v:
                        raise ValueError(
                            f"reshard: chunk {cid} state key {optim_state_key!r} "
                            f"(region {region_idx}) "
                            f"non-moment state key {k!r} differs between "
                            f"rank 0 ({v!r}) and rank {r_other} ({other_v!r})"
                        )
                for r2 in range(target_world_size):
                    val = v.clone() if isinstance(v, torch.Tensor) else v
                    new_per_rank_states[r2].setdefault(optim_state_key, {})[k] = val

        # Verify rank-replicated param_groups before reusing rank 0.
        param_groups = per_rank_state_dicts[0]["param_groups"]
        for r_other in range(1, src_world):
            other_pg = per_rank_state_dicts[r_other]["param_groups"]
            if len(other_pg) != len(param_groups):
                raise ValueError(
                    f"reshard: chunk {cid} param_groups length differs "
                    f"between rank 0 ({len(param_groups)}) and rank "
                    f"{r_other} ({len(other_pg)})"
                )
            for pg_idx, (pg0, pg_other) in enumerate(
                zip(param_groups, other_pg, strict=True)
            ):
                if set(pg0.keys()) != set(pg_other.keys()):
                    raise ValueError(
                        f"reshard: chunk {cid} param_groups[{pg_idx}] keys "
                        f"differ between rank 0 ({sorted(pg0.keys())}) and "
                        f"rank {r_other} ({sorted(pg_other.keys())})"
                    )
                for pk, pv in pg0.items():
                    pv_other = pg_other[pk]
                    if isinstance(pv, torch.Tensor) or isinstance(
                        pv_other, torch.Tensor
                    ):
                        if not (
                            isinstance(pv, torch.Tensor)
                            and isinstance(pv_other, torch.Tensor)
                            and torch.equal(pv, pv_other)
                        ):
                            raise ValueError(
                                f"reshard: chunk {cid} param_groups[{pg_idx}] "
                                f"key {pk!r} tensor differs between rank 0 and "
                                f"rank {r_other}"
                            )
                    elif pv != pv_other:
                        raise ValueError(
                            f"reshard: chunk {cid} param_groups[{pg_idx}] key "
                            f"{pk!r} differs between rank 0 ({pv!r}) and rank "
                            f"{r_other} ({pv_other!r})"
                        )

        # Write new per-rank shard files.
        for r2 in range(target_world_size):
            new_sd = {
                "state": new_per_rank_states[r2],
                "param_groups": param_groups,
            }
            out_path = os.path.join(cpu_dst_dir, f"chunk_{cid}_rank_{r2}.pt")
            torch.save(new_sd, out_path)

    # Recompute layout signature for the new world_size.
    fp = dict(meta["layout_fingerprint"])
    fp["world_size"] = int(target_world_size)
    new_signature = _layout_signature_from_fingerprint(fp)

    new_meta = dict(meta)
    new_meta["protrain_world_size"] = int(target_world_size)
    if new_meta.get("protrain_persistent_partition_version") is not None:
        new_meta["protrain_persistent_owner_world_size"] = int(target_world_size)
    new_meta["layout_fingerprint"] = fp
    new_meta["protrain_layout_signature"] = new_signature
    new_meta["regions_per_chunk"] = new_regions
    # Forensic field; loader ignores unknown keys. saving_rank preserved from original.
    new_meta["resharded_from_world_size"] = int(src_world)

    with open(os.path.join(dst_dir, METADATA_FILENAME), "w", encoding="utf-8") as f:
        json.dump(new_meta, f, indent=2, sort_keys=True)

    log_fn(
        f"reshard: wrote {dst_dir!r} "
        f"(chunks={len(chunk_paths)}, target_world={target_world_size})"
    )


__all__ = [
    "CHUNK_SHARD_FILE_RE",
    "CPU_OPTIM_DIRNAME",
    "GPU_OPTIM_FILENAME",
    "METADATA_FILENAME",
    "SAVE_MODE_SHARDED",
    "SCHEMA_FORMAT_VERSION",
    "_layout_signature_from_fingerprint",
    "_padded_region_bytes",
    "_reshard_region_state",
    "reshard_mode_c_shards",
]
