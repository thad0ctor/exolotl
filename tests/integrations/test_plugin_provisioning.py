"""Tests for external plugin source provisioning."""

import shutil
import subprocess  # nosec
import sys

import pytest

from axolotl.integrations import provisioning
from axolotl.integrations.provisioning import (
    _looks_like_git,
    _pip_install,
    provision_plugins,
)
from axolotl.utils.schemas.config import PluginSpec


def _git(args, cwd):
    git_bin = shutil.which("git")
    if not git_bin:
        pytest.skip("git executable not found in PATH")
    subprocess.run(  # nosec
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
            "PATH": __import__("os").environ.get("PATH", ""),
        },
    )


def _make_plugin_module(root, modname="ext_plugin", cls="ExtPlugin"):
    root.mkdir(parents=True, exist_ok=True)
    (root / f"{modname}.py").write_text(f"class {cls}:\n    name = '{cls}'\n")
    return f"{modname}.{cls}"


def _make_plugin_package(root, pkg="disc_plugin", cls="DiscPlugin"):
    """A package whose __init__ exports a real BasePlugin subclass (for discovery)."""
    pkgdir = root / pkg
    pkgdir.mkdir(parents=True, exist_ok=True)
    (pkgdir / "plugin.py").write_text(
        "from axolotl.integrations.base import BasePlugin\n"
        f"class {cls}(BasePlugin):\n    pass\n"
    )
    (pkgdir / "__init__.py").write_text(
        f"from .plugin import {cls}\n__all__ = ['{cls}']\n"
    )
    return f"{pkg}.{cls}"


_TEST_MODULE_PREFIXES = {
    "ext_plugin",
    "nested_plugin",
    "git_plugin",
    "utils_ext_plugin",
    "disc_plugin",
    "disc_plugin2",
    "sub_disc_plugin",
    "empty_pkg",
    "marker",
    "multi_pkg",
}


def _make_multi_plugin_package(root, pkg="multi_pkg", classes=("Alpha", "Beta")):
    pkgdir = root / pkg
    pkgdir.mkdir(parents=True, exist_ok=True)
    body = "from axolotl.integrations.base import BasePlugin\n"
    for cls in classes:
        body += f"class {cls}(BasePlugin):\n    pass\n"
    (pkgdir / "__init__.py").write_text(body)
    return [f"{pkg}.{c}" for c in classes]


@pytest.fixture
def cleanup_syspath():
    before = list(sys.path)
    before_mods = set(sys.modules)
    yield
    sys.path[:] = before
    for name in list(sys.modules):
        if name not in before_mods and name.split(".")[0] in _TEST_MODULE_PREFIXES:
            sys.modules.pop(name, None)


def test_looks_like_git():
    assert _looks_like_git("https://github.com/a/b.git")
    assert _looks_like_git("git@github.com:a/b.git")
    assert _looks_like_git("/path/to/repo.git")
    assert not _looks_like_git("/path/to/local_dir")
    assert not _looks_like_git("./relative/dir")


def test_pip_install_normalization():
    assert PluginSpec(cls="a.B", pip_install=True).pip_install == "editable"
    assert PluginSpec(cls="a.B", pip_install=False).pip_install is False
    assert (
        PluginSpec(cls="a.B", pip_install="requirements").pip_install == "requirements"
    )
    assert PluginSpec(cls="a.B").pip_install is False


def test_string_entries_are_noop():
    cfg = {"plugins": ["pkg.A", "pkg.B"]}
    provision_plugins(cfg)
    assert cfg["plugins"] == ["pkg.A", "pkg.B"]


def test_empty_plugins_noop():
    cfg = {"plugins": None}
    provision_plugins(cfg)
    assert cfg["plugins"] is None


def test_local_source_path_injection(tmp_path, monkeypatch, cleanup_syspath):
    src = tmp_path / "plugin_src"
    cls = _make_plugin_module(src)
    monkeypatch.chdir(tmp_path)
    cfg = {
        "plugin_cache_dir": str(tmp_path / "cache"),
        "plugins": [{"cls": cls, "source": str(src)}],
    }
    provision_plugins(cfg)
    assert cfg["plugins"] == [cls]
    assert str(src) in sys.path
    import ext_plugin  # noqa: F401  importable now

    assert ext_plugin.ExtPlugin.name == "ExtPlugin"


def test_local_source_with_subdir(tmp_path, monkeypatch, cleanup_syspath):
    root = tmp_path / "repo"
    sub = root / "src"
    cls = _make_plugin_module(sub, modname="nested_plugin", cls="Nested")
    monkeypatch.chdir(tmp_path)
    cfg = {
        "plugin_cache_dir": str(tmp_path / "cache"),
        "plugins": [{"cls": cls, "source": str(root), "subdir": "src"}],
    }
    provision_plugins(cfg)
    assert str(sub) in sys.path
    assert str(root) not in sys.path


def test_discovery_no_cls(tmp_path, monkeypatch, cleanup_syspath):
    src = tmp_path / "repo"
    expected = _make_plugin_package(src, pkg="disc_plugin", cls="DiscPlugin")
    monkeypatch.chdir(tmp_path)
    cfg = {
        "plugin_cache_dir": str(tmp_path / "cache"),
        "plugins": [{"source": str(src)}],  # no cls -> auto-discover
    }
    provision_plugins(cfg)
    assert cfg["plugins"] == [expected]


def test_discovery_with_subdir_no_cls(tmp_path, monkeypatch, cleanup_syspath):
    root = tmp_path / "repo"
    expected = _make_plugin_package(root / "src", pkg="sub_disc_plugin", cls="SubDisc")
    monkeypatch.chdir(tmp_path)
    cfg = {
        "plugin_cache_dir": str(tmp_path / "cache"),
        "plugins": [{"source": str(root), "subdir": "src"}],
    }
    provision_plugins(cfg)
    assert cfg["plugins"] == [expected]


def test_discovery_multiple_classes_raises(tmp_path, monkeypatch, cleanup_syspath):
    src = tmp_path / "repo"
    _make_plugin_package(src, pkg="disc_plugin", cls="DiscA")
    _make_plugin_package(src, pkg="disc_plugin2", cls="DiscB")
    monkeypatch.chdir(tmp_path)
    cfg = {
        "plugin_cache_dir": str(tmp_path / "cache"),
        "plugins": [{"source": str(src)}],
    }
    with pytest.raises(ValueError, match="Multiple plugin classes"):
        provision_plugins(cfg)


def test_discovery_no_class_raises(tmp_path, monkeypatch, cleanup_syspath):
    src = tmp_path / "repo"
    pkg = src / "empty_pkg"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("X = 1\n")
    monkeypatch.chdir(tmp_path)
    cfg = {
        "plugin_cache_dir": str(tmp_path / "cache"),
        "plugins": [{"source": str(src)}],
    }
    with pytest.raises(ValueError, match="No BasePlugin subclass"):
        provision_plugins(cfg)


def test_discovery_skips_tests_dir(tmp_path, monkeypatch, cleanup_syspath):
    # A top-level `tests` package (even one with its own BasePlugin subclass) must be
    # skipped, so discovery still resolves the single real plugin rather than erroring.
    src = tmp_path / "repo"
    expected = _make_plugin_package(src, pkg="disc_plugin", cls="DiscPlugin")
    _make_plugin_package(src, pkg="tests", cls="TestHelperPlugin")  # must be ignored
    monkeypatch.chdir(tmp_path)
    cfg = {
        "plugin_cache_dir": str(tmp_path / "cache"),
        "plugins": [{"source": str(src)}],
    }
    provision_plugins(cfg)
    assert cfg["plugins"] == [expected]


def test_cls_list_one_source(tmp_path, monkeypatch, cleanup_syspath):
    # A list of cls loads several plugins from one source block.
    root = tmp_path / "repo"
    expected = _make_multi_plugin_package(
        root, pkg="multi_pkg", classes=("Alpha", "Beta")
    )
    monkeypatch.chdir(tmp_path)
    cfg = {
        "plugin_cache_dir": str(tmp_path / "cache"),
        "plugins": [{"source": str(root), "cls": expected}],
    }
    provision_plugins(cfg)
    assert cfg["plugins"] == expected


def test_cls_list_mixed_with_string_entries(tmp_path, monkeypatch, cleanup_syspath):
    root = tmp_path / "repo"
    expected = _make_multi_plugin_package(
        root, pkg="multi_pkg", classes=("Alpha", "Beta")
    )
    monkeypatch.chdir(tmp_path)
    cfg = {
        "plugin_cache_dir": str(tmp_path / "cache"),
        "plugins": ["builtin.Plugin", {"source": str(root), "cls": expected}],
    }
    provision_plugins(cfg)
    assert cfg["plugins"] == ["builtin.Plugin", *expected]


def test_empty_cls_list_falls_back_to_discovery(tmp_path, monkeypatch, cleanup_syspath):
    # cls: [] behaves like an omitted cls -> discover the single plugin.
    src = tmp_path / "repo"
    expected = _make_plugin_package(src, pkg="disc_plugin", cls="DiscPlugin")
    monkeypatch.chdir(tmp_path)
    cfg = {
        "plugin_cache_dir": str(tmp_path / "cache"),
        "plugins": [{"source": str(src), "cls": []}],
    }
    provision_plugins(cfg)
    assert cfg["plugins"] == [expected]


def test_pluginspec_requires_cls_or_source():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        PluginSpec()
    assert PluginSpec(cls="a.B").cls == "a.B"
    assert PluginSpec(cls=["a.B", "c.D"]).cls == ["a.B", "c.D"]
    assert PluginSpec(source="/x").source == "/x"


def test_subdir_escape_raises(tmp_path, monkeypatch, cleanup_syspath):
    src = tmp_path / "plugin_src"
    _make_plugin_module(src)
    monkeypatch.chdir(tmp_path)
    cfg = {
        "plugin_cache_dir": str(tmp_path / "cache"),
        "plugins": [{"cls": "x.Y", "source": str(src), "subdir": "../../etc"}],
    }
    with pytest.raises(ValueError):
        provision_plugins(cfg)


def test_absolute_subdir_escape_raises(tmp_path, monkeypatch, cleanup_syspath):
    src = tmp_path / "plugin_src"
    _make_plugin_module(src)
    monkeypatch.chdir(tmp_path)
    cfg = {
        "plugin_cache_dir": str(tmp_path / "cache"),
        "plugins": [{"cls": "x.Y", "source": str(src), "subdir": "/etc"}],
    }
    with pytest.raises(ValueError):
        provision_plugins(cfg)


def test_utils_config_prepare_plugins_provisions(
    tmp_path, monkeypatch, cleanup_syspath
):
    # The `axolotl.utils.config` entry point (used by tests/docs) must provision
    # external sources too, not just `axolotl.cli.config`.
    from axolotl.utils.config import prepare_plugins as utils_prepare_plugins
    from axolotl.utils.dict import DictDefault

    src = tmp_path / "plugin_src"
    cls = _make_plugin_module(src, modname="utils_ext_plugin", cls="UtilsPlugin")
    monkeypatch.chdir(tmp_path)
    cfg = DictDefault(
        {
            "plugin_cache_dir": str(tmp_path / "cache"),
            "plugins": [{"cls": cls, "source": str(src)}],
        }
    )
    utils_prepare_plugins(cfg)
    assert cfg["plugins"] == [cls]


def test_relative_local_source_missing_raises(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = {
        "plugin_cache_dir": str(tmp_path / "cache"),
        "plugins": [{"cls": "x.Y", "source": "./does-not-exist"}],
    }
    with pytest.raises(FileNotFoundError):
        provision_plugins(cfg)


def test_mixed_string_and_dict_entries(tmp_path, monkeypatch, cleanup_syspath):
    src = tmp_path / "plugin_src"
    cls = _make_plugin_module(src)
    monkeypatch.chdir(tmp_path)
    cfg = {
        "plugin_cache_dir": str(tmp_path / "cache"),
        "plugins": ["builtin.Plugin", {"cls": cls, "source": str(src)}],
    }
    provision_plugins(cfg)
    assert cfg["plugins"] == ["builtin.Plugin", cls]


def test_cache_dir_self_ignores(tmp_path, monkeypatch, cleanup_syspath):
    src = tmp_path / "plugin_src"
    cls = _make_plugin_module(src)
    monkeypatch.chdir(tmp_path)
    cache = tmp_path / "cache"
    cfg = {
        "plugin_cache_dir": str(cache),
        "plugins": [{"cls": cls, "source": str(src)}],
    }
    provision_plugins(cfg)
    assert (cache / ".gitignore").read_text().strip().endswith("*")


def test_env_overrides_cache_dir(tmp_path, monkeypatch, cleanup_syspath):
    src = tmp_path / "plugin_src"
    cls = _make_plugin_module(src)
    env_cache = tmp_path / "env_cache"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AXOLOTL_PLUGIN_CACHE_DIR", str(env_cache))
    cfg = {"plugins": [{"cls": cls, "source": str(src)}]}
    provision_plugins(cfg)
    assert env_cache.exists()


def test_git_clone_and_checkout(tmp_path, monkeypatch, cleanup_syspath):
    # Build a real local git repo (no network), then clone it via provisioning.
    work = tmp_path / "work"
    _make_plugin_module(work, modname="git_plugin", cls="GitPlugin")
    _git(["init", "-q"], cwd=work)
    _git(["add", "."], cwd=work)
    _git(["commit", "-q", "-m", "init"], cwd=work)
    _git(["tag", "v1"], cwd=work)

    bare = tmp_path / "repo.git"  # .git suffix -> treated as a git source
    _git(["clone", "-q", "--bare", str(work), str(bare)], cwd=tmp_path)

    monkeypatch.chdir(tmp_path)
    cfg = {
        "plugin_cache_dir": str(tmp_path / "cache"),
        "plugins": [{"cls": "git_plugin.GitPlugin", "source": str(bare), "ref": "v1"}],
    }
    provision_plugins(cfg)
    assert cfg["plugins"] == ["git_plugin.GitPlugin"]
    import git_plugin  # noqa: F401

    assert git_plugin.GitPlugin.name == "GitPlugin"

    # Second call reuses the cache (idempotent) without error.
    cfg2 = {
        "plugin_cache_dir": str(tmp_path / "cache"),
        "plugins": [{"cls": "git_plugin.GitPlugin", "source": str(bare), "ref": "v1"}],
    }
    provision_plugins(cfg2)
    assert cfg2["plugins"] == ["git_plugin.GitPlugin"]


def test_git_update_fast_forwards_default_branch(
    tmp_path, monkeypatch, cleanup_syspath
):
    # update: true must advance the cached clone to the latest commit on the
    # (moving) default branch — not silently stay at the originally cloned commit.
    work = tmp_path / "work"
    work.mkdir()
    (work / "marker.txt").write_text("v1\n")
    _git(["init", "-q"], cwd=work)
    _git(["add", "."], cwd=work)
    _git(["commit", "-q", "-m", "v1"], cwd=work)

    bare = tmp_path / "repo.git"
    _git(["clone", "-q", "--bare", str(work), str(bare)], cwd=tmp_path)

    monkeypatch.chdir(tmp_path)
    cache = tmp_path / "cache"
    cfg = {
        "plugin_cache_dir": str(cache),
        "plugins": [
            {"cls": "marker.X", "source": str(bare)}
        ],  # no ref -> default branch
    }
    provision_plugins(cfg)
    clone = next(
        p for p in cache.iterdir() if p.is_dir() and p.name.startswith("repo-")
    )
    assert (clone / "marker.txt").read_text().strip() == "v1"

    # advance the upstream default branch
    (work / "marker.txt").write_text("v2\n")
    _git(["commit", "-q", "-am", "v2"], cwd=work)
    _git(["push", "-q", str(bare), "HEAD"], cwd=work)

    # without update: still the old commit (cache reuse)
    provision_plugins(dict(cfg))
    assert (clone / "marker.txt").read_text().strip() == "v1"

    # with update: fast-forwards to the new commit
    cfg_update = {
        "plugin_cache_dir": str(cache),
        "plugins": [{"cls": "marker.X", "source": str(bare), "update": True}],
    }
    provision_plugins(cfg_update)
    assert (clone / "marker.txt").read_text().strip() == "v2"


def test_cached_clone_interrupted_before_checkout_heals(
    tmp_path, monkeypatch, cleanup_syspath
):
    # A provisioning run killed between `git clone` and `git checkout ref` leaves a
    # cached clone on the default branch; the next run must check out the ref rather
    # than reuse the wrong commit forever.
    from axolotl.integrations.provisioning import _clone_target

    work = tmp_path / "work"
    work.mkdir()
    (work / "marker.txt").write_text("v1\n")
    _git(["init", "-q"], cwd=work)
    _git(["add", "."], cwd=work)
    _git(["commit", "-q", "-m", "v1"], cwd=work)
    _git(["tag", "rel1"], cwd=work)
    (work / "marker.txt").write_text("v2\n")
    _git(["commit", "-q", "-am", "v2"], cwd=work)

    bare = tmp_path / "repo.git"
    _git(["clone", "-q", "--bare", str(work), str(bare)], cwd=tmp_path)

    monkeypatch.chdir(tmp_path)
    cache = tmp_path / "cache"
    cache.mkdir()
    # Simulate the interrupted run: clone exists at the ref-keyed target, but the
    # `git checkout rel1` never happened (HEAD is the default branch at v2).
    clone = _clone_target(cache, str(bare), "rel1")
    _git(["clone", "-q", str(bare), str(clone)], cwd=tmp_path)
    assert (clone / "marker.txt").read_text().strip() == "v2"

    cfg = {
        "plugin_cache_dir": str(cache),
        "plugins": [{"cls": "marker.X", "source": str(bare), "ref": "rel1"}],
    }
    provision_plugins(cfg)
    assert (clone / "marker.txt").read_text().strip() == "v1"


def test_git_update_with_tag_ref_stays_pinned(tmp_path, monkeypatch, cleanup_syspath):
    # update: true with an immutable tag ref leaves HEAD detached; provisioning
    # must not error and must stay pinned even as the branch advances past it.
    work = tmp_path / "work"
    work.mkdir()
    (work / "marker.txt").write_text("v1\n")
    _git(["init", "-q"], cwd=work)
    _git(["add", "."], cwd=work)
    _git(["commit", "-q", "-m", "v1"], cwd=work)
    _git(["tag", "rel1"], cwd=work)

    bare = tmp_path / "repo.git"
    _git(["clone", "-q", "--bare", str(work), str(bare)], cwd=tmp_path)

    # advance the default branch past the tag
    (work / "marker.txt").write_text("v2\n")
    _git(["commit", "-q", "-am", "v2"], cwd=work)
    _git(["push", "-q", str(bare), "HEAD"], cwd=work)

    monkeypatch.chdir(tmp_path)
    cache = tmp_path / "cache"
    cfg = {
        "plugin_cache_dir": str(cache),
        "plugins": [
            {"cls": "marker.X", "source": str(bare), "ref": "rel1", "update": True}
        ],
    }
    provision_plugins(cfg)  # must not raise despite detached HEAD + update
    clone = next(
        p for p in cache.iterdir() if p.is_dir() and p.name.startswith("repo-")
    )
    assert (clone / "marker.txt").read_text().strip() == "v1"


def test_pip_install_requirements_dispatch(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(provisioning, "_run", lambda cmd, cwd=None: calls.append(cmd))
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "requirements.txt").write_text("# none\n")
    _pip_install("requirements", repo)
    assert calls and calls[0][1:4] == ["-m", "pip", "install"]
    assert "-r" in calls[0]


def test_pip_install_editable_falls_back_without_package(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(provisioning, "_run", lambda cmd, cwd=None: calls.append(cmd))
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "requirements.txt").write_text("# none\n")
    # No pyproject/setup.* -> editable falls back to requirements.
    _pip_install("editable", repo)
    assert calls and "-r" in calls[0]


def test_pip_install_editable_uses_package(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(provisioning, "_run", lambda cmd, cwd=None: calls.append(cmd))
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "pyproject.toml").write_text("[project]\nname='x'\nversion='0'\n")
    _pip_install("editable", repo)
    assert calls and "-e" in calls[0]
