"""Tests for the `axolotl plugins` CLI group.

Git sources are real repositories created in `tmp_path`, and pip is always mocked.
"""

import json

import pytest
import yaml

from axolotl.cli.plugin_manager import plugins
from axolotl.integrations.plugin_manifest import manifest_path, record_install
from axolotl.integrations.verification import verify_plugins

from tests.plugin_test_utils import restore_syspath, run_git, stub_pip


@pytest.fixture
def cleanup_syspath():
    yield from restore_syspath()


@pytest.fixture
def no_pip(monkeypatch):
    return stub_pip(monkeypatch)


def _module_source(tmp_path, modname="cli_plugin"):
    root = tmp_path / "src"
    root.mkdir(parents=True, exist_ok=True)
    (root / f"{modname}.py").write_text("class CliPlugin:\n    pass\n")
    return root


def _package_source(tmp_path):
    root = tmp_path / "pkg_src"
    root.mkdir(parents=True, exist_ok=True)
    (root / "pyproject.toml").write_text("[project]\nname='x'\nversion='0'\n")
    return root


def _git_source(tmp_path, plugin_package=False):
    """A bare repo holding a plugin, plus the tag and SHA it can be installed at."""
    work = tmp_path / "work"
    work.mkdir(parents=True, exist_ok=True)
    if plugin_package:
        pkg = work / "cli_plugin"
        pkg.mkdir()
        (pkg / "plugin.py").write_text(
            "from axolotl.integrations.base import BasePlugin\n"
            "class CliPlugin(BasePlugin):\n    pass\n"
        )
        (pkg / "__init__.py").write_text(
            "from .plugin import CliPlugin\n__all__ = ['CliPlugin']\n"
        )
    else:
        (work / "cli_plugin.py").write_text("class CliPlugin:\n    pass\n")
    run_git(["init", "-q"], cwd=work)
    run_git(["add", "."], cwd=work)
    run_git(["commit", "-q", "-m", "init"], cwd=work)
    run_git(["tag", "v1"], cwd=work)
    sha = run_git(["rev-parse", "HEAD"], cwd=work).stdout.strip()

    bare = tmp_path / "repo.git"
    run_git(["clone", "-q", "--bare", str(work), str(bare)], cwd=tmp_path)
    return bare, sha


# --- plugins list -----------------------------------------------------------


def test_list_empty_manifest(cli_runner, tmp_path):
    cache = tmp_path / "cache"

    result = cli_runner.invoke(plugins, ["list", "--cache-dir", str(cache)])

    assert result.exit_code == 0
    assert "No external plugins recorded" in result.output
    assert str(manifest_path(cache)) in result.output


def test_list_populated_manifest(cli_runner, tmp_path):
    cache = tmp_path / "cache"
    record_install(
        source="https://github.com/org/repo.git",
        ref="v1.0.0",
        resolved_sha="0123456789abcdef0123456789abcdef01234567",
        subdir=None,
        mode="pip",
        syspath_entry=None,
        cls=["my_plugin.MyPlugin"],
        cache_dir=cache,
    )
    record_install(
        source=str(tmp_path / "local"),
        ref=None,
        resolved_sha=None,
        subdir="src",
        mode="syspath",
        syspath_entry=str(tmp_path / "local" / "src"),
        cls=["local_plugin.LocalPlugin"],
        cache_dir=cache,
    )

    result = cli_runner.invoke(plugins, ["list", "--cache-dir", str(cache)])

    assert result.exit_code == 0
    assert "CLASS" in result.output and "COMMIT" in result.output
    assert "my_plugin.MyPlugin" in result.output
    assert "https://github.com/org/repo.git" in result.output
    assert "01234567" in result.output  # short sha
    assert "local_plugin.LocalPlugin" in result.output
    assert "syspath" in result.output


def test_list_ignores_malformed_manifest_entries(cli_runner, tmp_path):
    cache = tmp_path / "cache"
    record_install(
        source="https://github.com/org/repo.git",
        ref=None,
        resolved_sha=None,
        subdir=None,
        mode="syspath",
        syspath_entry=str(tmp_path / "src"),
        cls=["my_plugin.MyPlugin"],
        cache_dir=cache,
    )
    manifest = json.loads(manifest_path(cache).read_text())
    manifest["plugins"]["broken.Plugin"] = "not a mapping"
    manifest_path(cache).write_text(json.dumps(manifest))

    result = cli_runner.invoke(plugins, ["list", "--cache-dir", str(cache)])

    assert result.exit_code == 0, result.output
    assert "my_plugin.MyPlugin" in result.output
    assert "broken.Plugin" not in result.output


# --- plugins install --------------------------------------------------------


def test_install_aborts_when_declined(cli_runner, tmp_path, no_pip, cleanup_syspath):
    cache = tmp_path / "cache"
    src = _module_source(tmp_path)

    result = cli_runner.invoke(
        plugins,
        [
            "install",
            str(src),
            "--cls",
            "cli_plugin.CliPlugin",
            "--cache-dir",
            str(cache),
        ],
        input="n\n",
    )

    assert result.exit_code == 1
    assert "Only install plugins you trust" in result.output
    assert "Aborted." in result.output
    assert "Installed." not in result.output
    assert not manifest_path(cache).exists()


def test_install_with_yes_records_manifest(
    cli_runner, tmp_path, no_pip, cleanup_syspath
):
    cache = tmp_path / "cache"
    src = _module_source(tmp_path)

    result = cli_runner.invoke(
        plugins,
        [
            "install",
            str(src),
            "--cls",
            "cli_plugin.CliPlugin",
            "--cache-dir",
            str(cache),
            "--yes",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Installed." in result.output
    assert "cls: cli_plugin.CliPlugin" in result.output
    assert f"source: {src}" in result.output
    assert not no_pip.packages

    manifest = json.loads(manifest_path(cache).read_text())
    entry = manifest["plugins"]["cli_plugin.CliPlugin"]
    assert entry["mode"] == "syspath"
    assert entry["syspath_entry"] == str(src)
    assert entry["source"] == str(src)


def test_install_warns_that_a_custom_cache_needs_the_env_var(
    cli_runner, tmp_path, monkeypatch, no_pip, cleanup_syspath
):
    # Installing somewhere training will not look is silent breakage later.
    cache = tmp_path / "cache"
    monkeypatch.delenv("AXOLOTL_PLUGIN_CACHE_DIR", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    src = _module_source(tmp_path)

    result = cli_runner.invoke(
        plugins,
        [
            "install",
            str(src),
            "--cls",
            "cli_plugin.CliPlugin",
            "--cache-dir",
            str(cache),
            "--yes",
        ],
    )

    assert result.exit_code == 0, result.output
    assert f"AXOLOTL_PLUGIN_CACHE_DIR={cache}" in result.output


def test_install_plan_shows_editable_pip_for_a_local_package(
    cli_runner, tmp_path, no_pip, cleanup_syspath
):
    cache = tmp_path / "cache"
    src = _package_source(tmp_path)

    result = cli_runner.invoke(
        plugins,
        [
            "install",
            str(src),
            "--cls",
            "cli_plugin.CliPlugin",
            "--cache-dir",
            str(cache),
            "--yes",
        ],
    )

    assert result.exit_code == 0, result.output
    assert f"install : pip install -e {src}" in result.output
    assert "commit  : (editable install; tracks your working tree)" in result.output
    assert no_pip.packages == [(src, True)]


def test_install_plan_shows_requirements(cli_runner, tmp_path, no_pip, cleanup_syspath):
    cache = tmp_path / "cache"
    src = _module_source(tmp_path)
    (src / "requirements.txt").write_text("# none\n")

    result = cli_runner.invoke(
        plugins,
        [
            "install",
            str(src),
            "--cls",
            "cli_plugin.CliPlugin",
            "--cache-dir",
            str(cache),
            "--yes",
        ],
    )

    assert result.exit_code == 0, result.output
    assert f"deps    : pip install -r {src / 'requirements.txt'}" in result.output
    assert no_pip.requirements == [src / "requirements.txt"]


def test_install_forced_syspath_mode(cli_runner, tmp_path, no_pip, cleanup_syspath):
    cache = tmp_path / "cache"
    src = _package_source(tmp_path)

    result = cli_runner.invoke(
        plugins,
        [
            "install",
            str(src),
            "--cls",
            "cli_plugin.CliPlugin",
            "--mode",
            "syspath",
            "--cache-dir",
            str(cache),
            "--yes",
        ],
    )

    assert result.exit_code == 0, result.output
    assert f"install : add {src} to sys.path" in result.output
    assert not no_pip.packages


def test_install_rejects_a_bare_name(cli_runner, tmp_path, no_pip):
    result = cli_runner.invoke(
        plugins,
        ["install", "definitely-not-a-path", "--cache-dir", str(tmp_path / "cache")],
    )

    assert result.exit_code != 0
    assert isinstance(result.exception, ValueError)


# --- moving-ref warning -----------------------------------------------------


def test_install_warns_for_a_moving_ref(cli_runner, tmp_path, no_pip, cleanup_syspath):
    bare, _ = _git_source(tmp_path)

    result = cli_runner.invoke(
        plugins,
        [
            "install",
            str(bare),
            "--ref",
            "v1",
            "--cls",
            "cli_plugin.CliPlugin",
            "--cache-dir",
            str(tmp_path / "cache"),
            "--yes",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "v1 is not a commit SHA" in result.output


def test_install_does_not_warn_for_a_pinned_sha(
    cli_runner, tmp_path, no_pip, cleanup_syspath
):
    bare, sha = _git_source(tmp_path)

    result = cli_runner.invoke(
        plugins,
        [
            "install",
            str(bare),
            "--ref",
            sha,
            "--cls",
            "cli_plugin.CliPlugin",
            "--cache-dir",
            str(tmp_path / "cache"),
            "--yes",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "not a commit SHA" not in result.output


def test_install_does_not_warn_for_a_local_path(
    cli_runner, tmp_path, no_pip, cleanup_syspath
):
    # Nothing moves under a local path, so there is no ref to pin.
    src = _module_source(tmp_path)

    result = cli_runner.invoke(
        plugins,
        [
            "install",
            str(src),
            "--cls",
            "cli_plugin.CliPlugin",
            "--cache-dir",
            str(tmp_path / "cache"),
            "--yes",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "not a commit SHA" not in result.output


# --- install -> config round trip -------------------------------------------


def test_install_snippet_round_trips_through_verify_plugins(
    cli_runner, tmp_path, monkeypatch, no_pip, cleanup_syspath
):
    # The YAML the install prints is what users paste into a config, so config load
    # has to accept it as-is — including the class it auto-discovered.
    cache = tmp_path / "cache"
    monkeypatch.setenv("AXOLOTL_PLUGIN_CACHE_DIR", str(cache))
    bare, sha = _git_source(tmp_path, plugin_package=True)

    result = cli_runner.invoke(plugins, ["install", str(bare), "--yes"])

    assert result.exit_code == 0, result.output
    snippet = result.output[result.output.index("plugins:") :]
    cfg = yaml.safe_load(snippet)
    assert cfg["plugins"] == [
        {"cls": "cli_plugin.CliPlugin", "source": str(bare), "ref": sha}
    ]

    verify_plugins(cfg)

    assert cfg["plugins"] == ["cli_plugin.CliPlugin"]
