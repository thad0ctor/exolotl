"""Record of externally installed axolotl plugins.

``axolotl plugins install`` writes one manifest entry per plugin class it installs;
config load reads the manifest to learn where a plugin lives (its ``sys.path`` entry)
and which class a bare ``source:`` resolves to. The manifest is the only channel
between the two: config load never re-derives that information from the source tree.

Layout (``<cache_dir>/manifest.json``)::

    {
      "version": 1,
      "plugins": {
        "my_plugin.MyPlugin": {
          "source": "https://github.com/org/repo.git",
          "ref": "v1.2.0",
          "resolved_sha": "4f2a9c1...",
          "subdir": "src",
          "mode": "syspath",
          "syspath_entry": "/home/me/.cache/axolotl/plugins/repo-ab12cd34ef/src",
          "cls": ["my_plugin.MyPlugin"],
          "installed_at": "2026-07-19T12:00:00+00:00"
        }
      }
    }
"""

from __future__ import annotations

import importlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from filelock import FileLock

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

CACHE_DIR_ENV = "AXOLOTL_PLUGIN_CACHE_DIR"
XDG_CACHE_ENV = "XDG_CACHE_HOME"
DEFAULT_CACHE_SUBPATH = ("axolotl", "plugins")
MANIFEST_FILENAME = "manifest.json"
MANIFEST_VERSION = 1


def _empty_manifest() -> dict[str, Any]:
    return {"version": MANIFEST_VERSION, "plugins": {}}


def resolve_cache_dir(cache_dir: str | Path | None = None) -> Path:
    """Cache directory, without touching the filesystem.

    Per-user rather than per-project: a plugin is installed once and must resolve
    from whatever directory training is later launched in.
    """
    raw = cache_dir or os.environ.get(CACHE_DIR_ENV)
    if raw:
        # Absolute, so a relative `--cache-dir`/env var does not silently resolve
        # against a different cwd at train time than at install time.
        return Path(raw).expanduser().resolve()
    xdg = os.environ.get(XDG_CACHE_ENV)
    base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return base.joinpath(*DEFAULT_CACHE_SUBPATH)


def ensure_cache_dir(cache_dir: str | Path | None = None) -> Path:
    resolved = resolve_cache_dir(cache_dir)
    existed = resolved.exists()
    resolved.mkdir(parents=True, exist_ok=True)
    # A `*` ignore keeps clones out of a repo -- but only write it into a directory
    # we own. Pointed at an existing populated dir (e.g. `--cache-dir .`), it would
    # otherwise gitignore the user's whole project.
    gitignore = resolved / ".gitignore"
    owned = not existed or _is_our_cache(resolved)
    if owned and not gitignore.exists():
        gitignore.write_text(
            "# Created by axolotl for installed third-party plugins.\n*\n"
        )
    return resolved


def _is_our_cache(path: Path) -> bool:
    contents = {p.name for p in path.iterdir()}
    return (
        not contents
        or contents <= {MANIFEST_FILENAME, ".gitignore"}
        or all(
            name.endswith((".lock", ".tmp")) or (path / name / ".git").exists()
            for name in contents
        )
    )


def manifest_path(cache_dir: str | Path | None = None) -> Path:
    return resolve_cache_dir(cache_dir) / MANIFEST_FILENAME


def load_manifest(cache_dir: str | Path | None = None) -> dict[str, Any]:
    path = manifest_path(cache_dir)
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError:
        return _empty_manifest()
    except (OSError, ValueError) as exc:
        # ValueError covers both JSONDecodeError and a UnicodeDecodeError from a
        # truncated or non-UTF-8 file; either way, warn and carry on.
        LOG.warning("Ignoring unreadable plugin manifest at %s: %s", path, exc)
        return _empty_manifest()
    if not isinstance(data, dict) or not isinstance(data.get("plugins"), dict):
        LOG.warning("Ignoring malformed plugin manifest at %s", path)
        return _empty_manifest()

    version = data.get("version")
    if isinstance(version, int) and version > MANIFEST_VERSION:
        LOG.warning(
            "Plugin manifest at %s was written by a newer axolotl (version %s > %s); "
            "some entries may not be understood.",
            path,
            version,
            MANIFEST_VERSION,
        )

    plugins = {k: v for k, v in data["plugins"].items() if isinstance(v, dict)}
    if len(plugins) != len(data["plugins"]):
        LOG.warning("Ignoring malformed entries in the plugin manifest at %s", path)
    data["plugins"] = plugins
    return data


def entries(cache_dir: str | Path | None = None) -> list[dict[str, Any]]:
    """Manifest entries, each with its plugin key folded in as ``key``."""
    plugins = load_manifest(cache_dir).get("plugins", {})
    # key last: a hostile entry carrying its own "key" must not override the real one.
    return [{**value, "key": key} for key, value in sorted(plugins.items())]


def record_install(
    *,
    source: str,
    ref: str | None,
    resolved_sha: str | None,
    subdir: str | None,
    mode: str,
    syspath_entry: str | None,
    cls: list[str],
    cache_dir: str | Path | None = None,
) -> None:
    """Record one entry per installed class, keyed by its dotted path."""
    resolved = ensure_cache_dir(cache_dir)
    record = {
        "source": source,
        "ref": ref,
        "resolved_sha": resolved_sha,
        "subdir": subdir,
        "mode": mode,
        "syspath_entry": syspath_entry,
        "cls": list(cls),
        "installed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    # Multi-GPU launches share one cache; serialize read-modify-write.
    with FileLock(str(resolved / f"{MANIFEST_FILENAME}.lock")):
        manifest = load_manifest(resolved)
        manifest["version"] = MANIFEST_VERSION
        for cls_path in cls:
            manifest["plugins"][cls_path] = record
        # Readers (config load) take no lock, so swap the file in atomically rather
        # than letting them observe a half-written manifest.
        tmp = resolved / f"{MANIFEST_FILENAME}.tmp"
        try:
            tmp.write_text(json.dumps(manifest, indent=2) + "\n")
            os.replace(tmp, resolved / MANIFEST_FILENAME)
        finally:
            tmp.unlink(missing_ok=True)


def find_entries(spec, cache_dir: str | Path | None = None) -> list[dict[str, Any]]:
    """Manifest entries matching a config plugin spec, by class path then by source."""
    plugins = load_manifest(cache_dir).get("plugins", {})
    if spec.cls:
        keys = [spec.cls] if isinstance(spec.cls, str) else list(spec.cls)
        return [plugins[key] for key in keys if key in plugins]
    if not spec.source:
        return []

    matched: list[dict[str, Any]] = []
    for entry in plugins.values():
        if entry.get("source") != spec.source:
            continue
        # `ref` may be what the user asked for or the SHA the install resolved to.
        if spec.ref and spec.ref not in (entry.get("ref"), entry.get("resolved_sha")):
            continue
        if entry not in matched:
            matched.append(entry)
    return matched


def apply_syspath_entries(specs: Iterable, cache_dir: str | Path | None = None) -> None:
    """Put syspath-mode plugins recorded in the manifest back on ``sys.path``."""
    added = False
    for spec in specs:
        for entry in find_entries(spec, cache_dir):
            path = entry.get("syspath_entry")
            if path and path not in sys.path and Path(path).is_dir():
                # Append, never prepend: a plugin repo's top-level modules must not
                # shadow the stdlib or site-packages for the rest of the process.
                sys.path.append(path)
                added = True
                LOG.debug("Added plugin path to sys.path: %s", path)
    if added:
        # A path removed and re-added in one process keeps a stale finder otherwise.
        importlib.invalidate_caches()
