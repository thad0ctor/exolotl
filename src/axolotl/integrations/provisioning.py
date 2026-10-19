"""Fetch and install external axolotl plugins.

This is the backend for ``axolotl plugins install`` and is only ever reached from
that command: it clones git sources, runs pip, and writes to the plugin cache.
Config load never touches this module -- it only verifies (see
``axolotl.integrations.verification``) that a declared plugin is already importable.

Resolution and provisioning are separate steps. ``resolve_install_spec`` turns
whatever the user typed into a concrete ``PluginSpec``; ``provision`` acts on that
spec and reports back what was installed, including the resolved commit SHA that
gets recorded in the manifest.
"""

from __future__ import annotations

import hashlib
import importlib
import re
import subprocess  # nosec
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal
from urllib.parse import urlparse

from filelock import FileLock

from axolotl.integrations.plugin_manifest import ensure_cache_dir
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

InstallMode = Literal["auto", "pip", "syspath"]

_GIT_PREFIXES = ("http://", "https://", "git://", "ssh://", "git@")
_PACKAGE_MARKERS = ("pyproject.toml", "setup.py", "setup.cfg")


class ProvisionAborted(RuntimeError):
    """The confirmation callback declined the install."""


@dataclass
class ProvisionPlan:
    """What ``provision`` is about to do, once the source has been fetched.

    Handed to the confirmation callback: everything below is known before any plugin
    code is executed (pip install, class discovery), so a caller can show the exact
    commit it resolved and let the user back out.
    """

    source: str
    ref: str | None
    subdir: str | None
    root: Path
    search_path: Path
    resolved_sha: str | None
    mode: Literal["pip", "syspath"]
    package_root: Path | None
    editable: bool
    requirements: Path | None


@dataclass
class ProvisionResult:
    """What ``provision`` actually did, in the shape the manifest needs."""

    source: str
    ref: str | None
    subdir: str | None
    root: Path
    search_path: Path
    resolved_sha: str | None
    mode: Literal["pip", "syspath"]
    syspath_entry: str | None
    cls: list[str]


def _looks_like_git(source: str) -> bool:
    return source.startswith(_GIT_PREFIXES) or source.endswith(".git")


def is_git_source(source: str) -> bool:
    """Whether a source is fetched with git, and so carries a moving-ref risk."""
    return _looks_like_git(source)


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


# `<transport>::<address>` invokes a git remote helper -- `ext::sh -c <payload>`
# runs an arbitrary program. `--` before the source stops option injection but not
# this, so reject the syntax outright. Anchored, so it never matches `https://`,
# `git@host:path`, or an IPv6 URL whose `::` sits inside brackets.
_REMOTE_HELPER_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*::")


def _check_source(source: str) -> None:
    if _REMOTE_HELPER_RE.match(source):
        raise ValueError(
            f"Refusing plugin source {source!r}: git remote-helper transports "
            "(`<name>::…`) can execute arbitrary programs."
        )


def _check_ref(ref: str) -> None:
    # `git checkout` has no way to say "this is a ref, not a flag", so a leading dash
    # would be parsed as an option (e.g. `--upload-pack=`, arbitrary execution).
    if ref.startswith("-"):
        raise ValueError(f"Invalid git ref {ref!r}: refs may not start with '-'.")


def _clone_or_update(source: str, ref: str | None, target: Path, update: bool) -> Path:
    _check_source(source)
    if ref:
        _check_ref(ref)
    if not target.exists():
        LOG.info("Cloning plugin source %s -> %s", source, target)
        _run(["git", "clone", "--", source, str(target)])
    elif update:
        LOG.info("Updating plugin source at %s", target)
        _run(["git", "fetch", "--all", "--tags", "--prune"], cwd=target)
    else:
        LOG.info("Reusing cached plugin source at %s", target)
    if ref:
        # Also on cache reuse: heals a clone interrupted before the checkout, which
        # would otherwise stay pinned to the default branch.
        _run(["git", "checkout", ref, "--"], cwd=target)
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


def _head_sha(target: Path) -> str | None:
    # Guarded so a plain directory does not report the SHA of an enclosing repo.
    if not (target / ".git").exists():
        return None
    result = subprocess.run(  # nosec
        ["git", "rev-parse", "HEAD"],
        cwd=str(target),
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    return result.stdout.strip() or None if result.returncode == 0 else None


def _resolve_local(source: str, base_dir: Path) -> Path:
    if source.startswith("file://"):
        source = urlparse(source).path
    path = Path(source).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Plugin source path does not exist: {path}")
    return path


def _resolve_subdir(root: Path, subdir: str | None) -> Path:
    if not subdir:
        return root
    root = root.resolve()
    path = (root / subdir).resolve()
    if path != root and not path.is_relative_to(root):
        raise ValueError(f"subdir {subdir!r} escapes the plugin source root {root}")
    return path


def _package_root(root: Path, search_path: Path) -> Path | None:
    """The directory pip should install, or None if this is not a Python package."""
    for candidate in (search_path, root):
        if any((candidate / marker).exists() for marker in _PACKAGE_MARKERS):
            return candidate
    return None


def _pip_install(package_root: Path, editable: bool) -> None:
    cmd = [sys.executable, "-m", "pip", "install"]
    if editable:
        cmd.append("-e")
    cmd.append(str(package_root))
    LOG.info("Installing plugin: %s", " ".join(cmd))
    _run(cmd)


def _pip_install_requirements(requirements: Path) -> None:
    LOG.info("Installing plugin requirements: %s", requirements)
    _run([sys.executable, "-m", "pip", "install", "-r", str(requirements)])


def _requirements_file(root: Path) -> Path | None:
    requirements = root / "requirements.txt"
    return requirements if requirements.exists() else None


def _add_to_syspath(path: Path) -> None:
    entry = str(path)
    if entry not in sys.path:
        # Append, never prepend: a plugin repo with a top-level `logging.py` (etc.)
        # must not shadow the stdlib or site-packages for the rest of the process.
        sys.path.append(entry)
        LOG.info("Added plugin path to sys.path: %s", entry)


def resolve_install_spec(
    source: str,
    ref: str | None = None,
    subdir: str | None = None,
    cls: str | list[str] | None = None,
):
    """Turn a user-supplied source into a concrete ``PluginSpec``.

    Git URLs and local paths resolve today; a name lookup against a plugin registry
    would slot in here as another branch returning the same ``PluginSpec``.
    """
    _check_source(source)
    if ref:
        _check_ref(ref)

    from axolotl.utils.schemas.config import PluginSpec

    if _looks_like_git(source):
        return PluginSpec(cls=cls, source=source, ref=ref, subdir=subdir)

    path = Path(source[len("file://") :] if source.startswith("file://") else source)
    if path.expanduser().exists() or any(sep in source for sep in ("/", "\\")):
        return PluginSpec(cls=cls, source=source, ref=ref, subdir=subdir)

    raise ValueError(
        f"Cannot resolve plugin source {source!r}: expected a git URL or a local path."
    )


def provision(
    spec,
    cache_dir: str | Path | None = None,
    base_dir: Path | None = None,
    mode: InstallMode = "auto",
    update: bool = False,
    confirm: Callable[[ProvisionPlan], bool] | None = None,
) -> ProvisionResult:
    """Fetch a plugin source and make it importable, reporting what was done.

    ``confirm`` is called with the resolved plan after the source is fetched and
    before any of its code runs; returning False aborts.
    """
    resolved_cache = ensure_cache_dir(cache_dir)
    base_dir = base_dir or Path.cwd()

    # Serialize clone + install across processes that share this cache.
    lock = FileLock(str(resolved_cache / f"{_source_key(spec.source, spec.ref)}.lock"))
    with lock:
        if _looks_like_git(spec.source):
            root = _clone_or_update(
                spec.source,
                spec.ref,
                _clone_target(resolved_cache, spec.source, spec.ref),
                update,
            )
            resolved_sha = _head_sha(root)
        else:
            root = _resolve_local(spec.source, base_dir)
            resolved_sha = _head_sha(root)

        search_path = _resolve_subdir(root, spec.subdir)
        package_root = _package_root(root, search_path)
        if mode == "pip" and package_root is None:
            raise ValueError(
                f"{root} has no {' / '.join(_PACKAGE_MARKERS)}; it cannot be pip "
                "installed. Use --mode syspath to load it from the source tree."
            )
        install_mode: Literal["pip", "syspath"] = (
            "pip" if mode != "syspath" and package_root is not None else "syspath"
        )
        # A local path is a plugin under development; track the tree, not a snapshot.
        editable = install_mode == "pip" and not _looks_like_git(spec.source)
        if editable:
            # An editable install tracks a mutable tree, so no commit describes it.
            resolved_sha = None
        requirements = _requirements_file(root) if install_mode == "syspath" else None

        plan = ProvisionPlan(
            source=spec.source,
            ref=spec.ref,
            subdir=spec.subdir,
            root=root,
            search_path=search_path,
            resolved_sha=resolved_sha,
            mode=install_mode,
            package_root=package_root,
            editable=editable,
            requirements=requirements,
        )
        if confirm is not None and not confirm(plan):
            raise ProvisionAborted(f"Install of {spec.source} was declined")

        if install_mode == "pip" and package_root is not None:
            _pip_install(package_root, editable=editable)
        elif requirements is not None:
            _pip_install_requirements(requirements)

    # Needed for class auto-discovery below, and for syspath mode it is the install.
    _add_to_syspath(search_path)
    importlib.invalidate_caches()

    if spec.cls:
        cls = [spec.cls] if isinstance(spec.cls, str) else list(spec.cls)
    else:
        cls = [_discover_plugin_cls(search_path, installed=install_mode == "pip")]

    return ProvisionResult(
        source=spec.source,
        ref=spec.ref,
        subdir=spec.subdir,
        root=root,
        search_path=search_path,
        resolved_sha=resolved_sha,
        mode=install_mode,
        syspath_entry=str(search_path) if install_mode == "syspath" else None,
        cls=cls,
    )


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
    found = packages or modules
    if not found and (search_dir / "src").is_dir():
        # src layout: the package name is the same wherever it is imported from.
        return _candidate_modules(search_dir / "src")
    return found


def _discover_plugin_cls(search_dir: Path, installed: bool = False) -> str:
    """Find the single BasePlugin subclass exported by the package(s) under search_dir.

    Standard plugin layout: an importable package whose top-level ``__init__`` exports
    exactly one ``BasePlugin`` subclass. Raises if zero or several are found, so the
    user knows to pass ``--cls`` explicitly. ``installed`` reports whether a pip
    install already happened, which changes what the user has to do next.
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
    if not found and installed:
        raise ValueError(
            f"The package at {search_dir} was installed, but no BasePlugin subclass "
            "could be discovered in it. Re-run with `--cls <dotted.path.To.Plugin>` "
            "to record which class to load."
        )
    if not found:
        raise ValueError(
            f"No BasePlugin subclass found under {search_dir}. Export your plugin "
            "from the package's __init__, or pass `--cls` explicitly."
        )
    raise ValueError(
        f"Multiple plugin classes found ({', '.join(found)}); pass `--cls` to pick one."
    )
