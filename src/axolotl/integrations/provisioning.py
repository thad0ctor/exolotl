"""Provision external plugin sources declared in an axolotl config.

A ``plugins:`` entry may be either a dotted class path (the original behaviour) or
a mapping describing where the plugin lives:

    plugins:
      - axolotl.integrations.liger.LigerPlugin          # dotted path, unchanged
      - cls: my_plugin.MyPlugin                          # external source
        source: https://github.com/org/repo.git          # git URL or local path
        ref: v1.2.0                                       # branch / tag / commit
        subdir: src                                       # dir within source on sys.path
        pip_install: editable                             # false | editable | requirements

Git/local sources are cloned (or resolved) into a self-ignoring cache directory and
added to ``sys.path`` so the axolotl checkout stays untouched. ``pip_install``
optionally installs the plugin and its dependencies into the active environment; the
axolotl install itself is never modified.

``provision_plugins`` rewrites ``cfg["plugins"]`` in place to a flat list of dotted
class paths, so every downstream consumer keeps seeing ``list[str]``.
"""

from __future__ import annotations

import hashlib
import importlib
import os
import subprocess  # nosec
import sys
from pathlib import Path
from urllib.parse import urlparse

from filelock import FileLock

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

DEFAULT_CACHE_DIRNAME = ".axolotl_plugins"
CACHE_DIR_ENV = "AXOLOTL_PLUGIN_CACHE_DIR"
_GIT_PREFIXES = ("http://", "https://", "git://", "ssh://", "git@")


def _looks_like_git(source: str) -> bool:
    return source.startswith(_GIT_PREFIXES) or source.endswith(".git")


def _resolve_cache_dir(cfg) -> Path:
    raw = os.environ.get(CACHE_DIR_ENV) or (
        cfg.get("plugin_cache_dir") if cfg else None
    )
    cache_dir = Path(raw).expanduser() if raw else Path.cwd() / DEFAULT_CACHE_DIRNAME
    cache_dir.mkdir(parents=True, exist_ok=True)
    # A self-contained ignore keeps a vanilla axolotl checkout clean without
    # touching the repo's root .gitignore.
    gitignore = cache_dir / ".gitignore"
    if not gitignore.exists():
        gitignore.write_text(
            "# Created by axolotl for cloned third-party plugins.\n*\n"
        )
    return cache_dir


def _source_key(source: str, ref: str | None) -> str:
    return hashlib.sha1(f"{source}@{ref or 'HEAD'}".encode()).hexdigest()[:10]  # nosec


def _clone_target(cache_dir: Path, source: str, ref: str | None) -> Path:
    name = Path(urlparse(source).path).name or "repo"
    if name.endswith(".git"):
        name = name[:-4]
    return cache_dir / f"{name}-{_source_key(source, ref)}"


def _run(cmd: list[str], cwd: Path | None = None, check: bool = True) -> None:
    try:
        subprocess.run(  # nosec
            cmd,
            cwd=str(cwd) if cwd else None,
            check=check,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        LOG.error(
            "Command failed (%s): %s\n%s", exc.returncode, " ".join(cmd), exc.output
        )
        raise


def _clone_or_update(source: str, ref: str | None, target: Path, update: bool) -> Path:
    if not target.exists():
        LOG.info("Cloning plugin source %s -> %s", source, target)
        _run(["git", "clone", source, str(target)])
    elif update:
        LOG.info("Updating plugin source at %s", target)
        _run(["git", "fetch", "--all", "--tags", "--prune"], cwd=target)
    else:
        LOG.info("Reusing cached plugin source at %s", target)
    if ref:
        # Also on cache reuse: heals a clone interrupted before the checkout, which
        # would otherwise stay pinned to the default branch.
        _run(["git", "checkout", ref], cwd=target)
    if update and _on_branch(target):
        # `git checkout` alone does not advance an already-checked-out branch, so
        # fast-forward it to the freshly fetched remote tip. Skipped when HEAD is
        # detached at an immutable tag/commit (nothing to advance).
        _run(["git", "merge", "--ff-only"], cwd=target, check=False)
    return target


def _on_branch(target: Path) -> bool:
    result = subprocess.run(  # nosec
        ["git", "symbolic-ref", "-q", "HEAD"],
        cwd=str(target),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def _resolve_local(source: str, base_dir: Path) -> Path:
    if source.startswith("file://"):
        source = urlparse(source).path
    path = Path(source).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Plugin source path does not exist: {path}")
    return path


def _pip_install(mode: str, repo: Path) -> None:
    if mode == "editable":
        if any(
            (repo / f).exists() for f in ("pyproject.toml", "setup.py", "setup.cfg")
        ):
            LOG.info("Installing plugin (editable): %s", repo)
            _run([sys.executable, "-m", "pip", "install", "-e", str(repo)])
            return
        LOG.warning(
            "%s is not a Python package; falling back to requirements install", repo
        )
        mode = "requirements"
    if mode == "requirements":
        req = repo / "requirements.txt"
        if req.exists():
            LOG.info("Installing plugin requirements: %s", req)
            _run([sys.executable, "-m", "pip", "install", "-r", str(req)])
        else:
            LOG.warning("No requirements.txt found in %s; nothing to install", repo)


def _resolve_subdir(root: Path, subdir: str | None) -> Path:
    if not subdir:
        return root
    root = root.resolve()
    path = (root / subdir).resolve()
    if path != root and not path.is_relative_to(root):
        raise ValueError(f"subdir {subdir!r} escapes the plugin source root {root}")
    return path


def _add_to_syspath(path: Path) -> None:
    p = str(path)
    if p not in sys.path:
        sys.path.insert(0, p)
        LOG.info("Added plugin path to sys.path: %s", p)


def _provision_one(spec, cache_dir: Path, base_dir: Path) -> Path:
    """Clone/resolve a single external plugin and wire it into the environment.

    Returns the directory the plugin package lives under (root + subdir), used both
    for sys.path injection and for class auto-discovery.
    """
    # Serialize clone + install across ranks/processes that share this cache.
    lock = FileLock(str(cache_dir / f"{_source_key(spec.source, spec.ref)}.lock"))
    with lock:
        if _looks_like_git(spec.source):
            root = _clone_or_update(
                spec.source,
                spec.ref,
                _clone_target(cache_dir, spec.source, spec.ref),
                spec.update,
            )
        else:
            root = _resolve_local(spec.source, base_dir)
        if spec.pip_install:
            _pip_install(spec.pip_install, root)
    search_path = _resolve_subdir(root, spec.subdir)
    # An editable install already exposes the package; sys.path injection is only
    # needed when we load the plugin straight from the source tree.
    if spec.pip_install != "editable":
        _add_to_syspath(search_path)
    return search_path


# Directories/modules that are never the plugin package; skip during discovery.
_DISCOVERY_SKIP_DIRS = {
    "tests",
    "test",
    "docs",
    "examples",
    "benchmarks",
    "build",
    "dist",
}
_DISCOVERY_SKIP_MODS = {"setup", "conftest"}


def _candidate_modules(search_dir: Path) -> list[str]:
    """Top-level importable names under search_dir, preferring packages over modules."""
    packages, modules = [], []
    for entry in sorted(search_dir.iterdir()):
        name = entry.name
        if name.startswith((".", "_")):
            continue
        if entry.is_dir() and (entry / "__init__.py").exists():
            if name not in _DISCOVERY_SKIP_DIRS:
                packages.append(name)
        elif entry.is_file() and name.endswith(".py"):
            mod = name[:-3]
            if mod not in _DISCOVERY_SKIP_MODS and not mod.startswith("test_"):
                modules.append(mod)
    # Loose modules (e.g. setup.py) are only imported when there's no real package.
    return packages or modules


def _discover_plugin_cls(search_dir: Path) -> str:
    """Find the single BasePlugin subclass exported by the package(s) under search_dir.

    Standard plugin layout: an importable package whose top-level ``__init__`` exports
    exactly one ``BasePlugin`` subclass. Raises if zero or several are found, so the
    user knows to set ``cls`` explicitly.
    """
    from axolotl.integrations.base import BasePlugin

    found: list[str] = []
    for modname in _candidate_modules(search_dir):
        try:
            module = importlib.import_module(modname)
        except Exception as exc:  # a non-plugin or dep-short module; skip it
            LOG.debug("Skipping %s during plugin discovery: %s", modname, exc)
            continue
        for attr, obj in vars(module).items():
            if (
                isinstance(obj, type)
                and issubclass(obj, BasePlugin)
                and obj is not BasePlugin
                and (
                    obj.__module__ == modname
                    or obj.__module__.startswith(modname + ".")
                )
            ):
                found.append(f"{modname}.{attr}")

    found = sorted(set(found))
    if len(found) == 1:
        return found[0]
    if not found:
        raise ValueError(
            f"No BasePlugin subclass found under {search_dir}. Export your plugin "
            "from the package's __init__, or set `cls` explicitly."
        )
    raise ValueError(
        f"Multiple plugin classes found ({', '.join(found)}); set `cls` to pick one."
    )


def provision_plugins(cfg) -> None:
    """Normalize ``cfg['plugins']`` to ``list[str]``, provisioning any external sources.

    Idempotent: a no-op when every entry is already a dotted path string.
    """
    plugins = cfg.get("plugins") if cfg else None
    if not plugins or all(isinstance(p, str) for p in plugins):
        return

    from axolotl.utils.schemas.config import PluginSpec

    cache_dir = _resolve_cache_dir(cfg)
    base_dir = Path.cwd()
    normalized: list[str] = []
    for entry in plugins:
        if isinstance(entry, str):
            normalized.append(entry)
            continue
        spec = entry if isinstance(entry, PluginSpec) else PluginSpec(**dict(entry))
        search_path = _provision_one(spec, cache_dir, base_dir) if spec.source else None
        if spec.cls:
            normalized.extend([spec.cls] if isinstance(spec.cls, str) else spec.cls)
        elif search_path is not None:
            normalized.append(_discover_plugin_cls(search_path))
        else:
            raise ValueError("plugin entry needs `cls` when no `source` is given")
    cfg["plugins"] = normalized
