"""Shared helpers for the external-plugin tests.

The provisioning, verification, and CLI suites live in different test packages, so
these are imported rather than picked up from a conftest; the fixtures that wrap
them stay local to each module.
"""

import os
import shutil
import subprocess  # nosec
import sys
from types import SimpleNamespace

import pytest

from axolotl.integrations import provisioning

# Top-level module names the plugin tests write and import; unloaded after each test
# so a later test does not pick up an earlier one's module from `sys.modules`.
PLUGIN_TEST_MODULE_PREFIXES = {
    "cli_plugin",
    "dep_short_plugin",
    "disc_plugin",
    "disc_plugin2",
    "empty_pkg",
    "ext_plugin",
    "git_plugin",
    "installed_plugin",
    "marker",
    "multi_pkg",
    "multi_plugin",
    "nested_plugin",
    "pkg_plugin",
    "prepared_plugin",
    "req_plugin",
    "srclayout_plugin",
    "sub_disc_plugin",
}


def run_git(args, cwd):
    """Run git in a hermetic environment, skipping the test if git is unavailable."""
    git_bin = shutil.which("git")
    if not git_bin:
        pytest.skip("git executable not found in PATH")
    return subprocess.run(  # nosec
        [git_bin, *args],
        cwd=str(cwd),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env={
            "GIT_AUTHOR_NAME": "t",
            "GIT_AUTHOR_EMAIL": "t@t",
            "GIT_COMMITTER_NAME": "t",
            "GIT_COMMITTER_EMAIL": "t@t",
            "HOME": str(cwd),
            "PATH": os.environ.get("PATH", ""),
        },
    )


def isolate_default_cache(tmp_path, monkeypatch):
    """Point the per-user default cache inside tmp_path, without creating it."""
    monkeypatch.delenv("AXOLOTL_PLUGIN_CACHE_DIR", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    return tmp_path / "xdg" / "axolotl" / "plugins"


def restore_syspath():
    """Fixture body: undo whatever a test added to ``sys.path`` / ``sys.modules``."""
    before = list(sys.path)
    before_mods = set(sys.modules)
    yield
    sys.path[:] = before
    for name in list(sys.modules):
        if (
            name not in before_mods
            and name.split(".")[0] in PLUGIN_TEST_MODULE_PREFIXES
        ):
            sys.modules.pop(name, None)


def stub_pip(monkeypatch):
    """Fixture body: record every pip entry point instead of running pip."""
    calls = SimpleNamespace(packages=[], requirements=[])
    monkeypatch.setattr(
        provisioning,
        "_pip_install",
        lambda package_root, editable: calls.packages.append((package_root, editable)),
    )
    monkeypatch.setattr(
        provisioning,
        "_pip_install_requirements",
        calls.requirements.append,
    )
    return calls
