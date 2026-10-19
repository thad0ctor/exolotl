"""Tests for the `axolotl plugins install` provisioning backend.

Nothing here reaches the network: git sources are real repositories created in
`tmp_path`, and pip is always mocked.
"""

import sys

import pytest

from axolotl.integrations import provisioning
from axolotl.integrations.plugin_manifest import (
    entries,
    record_install,
    resolve_cache_dir,
)
from axolotl.integrations.provisioning import (
    ProvisionAborted,
    _clone_or_update,
    _clone_target,
    _looks_like_git,
    _pip_install,
    _pip_install_requirements,
    is_git_source,
    provision,
    resolve_install_spec,
)

from tests.plugin_test_utils import (
    isolate_default_cache,
    restore_syspath,
    run_git,
    stub_pip,
)


@pytest.fixture
def cleanup_syspath():
    yield from restore_syspath()


@pytest.fixture
def no_pip(monkeypatch):
    return stub_pip(monkeypatch)


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


def _make_multi_plugin_package(root, pkg="multi_pkg", classes=("Alpha", "Beta")):
    pkgdir = root / pkg
    pkgdir.mkdir(parents=True, exist_ok=True)
    body = "from axolotl.integrations.base import BasePlugin\n"
    for cls in classes:
        body += f"class {cls}(BasePlugin):\n    pass\n"
    (pkgdir / "__init__.py").write_text(body)
    return [f"{pkg}.{c}" for c in classes]


def _make_package_source(root, name="pkg_plugin"):
    root.mkdir(parents=True, exist_ok=True)
    (root / "pyproject.toml").write_text(f"[project]\nname='{name}'\nversion='0'\n")
    return root


# --- source resolution ------------------------------------------------------


def test_looks_like_git():
    assert _looks_like_git("https://github.com/a/b.git")
    assert _looks_like_git("git@github.com:a/b.git")
    assert _looks_like_git("/path/to/repo.git")
    assert not _looks_like_git("/path/to/local_dir")
    assert not _looks_like_git("./relative/dir")


def test_is_git_source_is_the_public_view_of_looks_like_git():
    # The CLI gates its moving-ref warning on this, so it must stay in step.
    for source in ("https://github.com/a/b.git", "git@h:a/b.git", "./dir", "/tmp/x"):
        assert is_git_source(source) == _looks_like_git(source)


def test_resolve_install_spec_git_url():
    spec = resolve_install_spec(
        "https://github.com/org/repo.git", ref="v1.0.0", subdir="src", cls="a.B"
    )
    assert spec.source == "https://github.com/org/repo.git"
    assert spec.ref == "v1.0.0"
    assert spec.subdir == "src"
    assert spec.cls == "a.B"


def test_resolve_install_spec_local_path(tmp_path):
    src = tmp_path / "plugin_src"
    src.mkdir()
    assert resolve_install_spec(str(src)).source == str(src)
    # A path-shaped source resolves even when it does not exist yet.
    assert resolve_install_spec("./nope/plugin").source == "./nope/plugin"


def test_resolve_install_spec_bare_name_raises():
    with pytest.raises(ValueError, match="expected a git URL or a local path"):
        resolve_install_spec("definitely-not-a-path")


# --- install mode selection -------------------------------------------------


def test_bare_module_dir_uses_syspath_mode(tmp_path, no_pip, cleanup_syspath):
    src = tmp_path / "plugin_src"
    cls = _make_plugin_module(src)
    spec = resolve_install_spec(str(src), cls=cls)

    result = provision(spec, cache_dir=tmp_path / "cache")

    assert result.mode == "syspath"
    assert result.syspath_entry == str(src)
    assert result.cls == [cls]
    assert not no_pip.packages
    assert str(src) in sys.path
    import ext_plugin  # noqa: F401  importable now

    assert ext_plugin.ExtPlugin.name == "ExtPlugin"


def test_package_dir_uses_pip_mode(tmp_path, no_pip, cleanup_syspath):
    src = _make_package_source(tmp_path / "repo")
    spec = resolve_install_spec(str(src), cls="pkg_plugin.PkgPlugin")

    result = provision(spec, cache_dir=tmp_path / "cache")

    assert result.mode == "pip"
    assert result.syspath_entry is None
    # A local source is someone's working tree, so it installs editable.
    assert no_pip.packages == [(src, True)]


def test_git_package_source_is_not_editable(tmp_path, no_pip, cleanup_syspath):
    work = tmp_path / "work"
    _make_package_source(work)
    run_git(["init", "-q"], cwd=work)
    run_git(["add", "."], cwd=work)
    run_git(["commit", "-q", "-m", "init"], cwd=work)
    bare = tmp_path / "repo.git"
    run_git(["clone", "-q", "--bare", str(work), str(bare)], cwd=tmp_path)

    spec = resolve_install_spec(str(bare), cls="pkg_plugin.PkgPlugin")
    result = provision(spec, cache_dir=tmp_path / "cache")

    assert result.mode == "pip"
    assert no_pip.packages and no_pip.packages[0][1] is False


def test_mode_syspath_skips_pip_for_a_package(tmp_path, no_pip, cleanup_syspath):
    src = _make_package_source(tmp_path / "repo")
    spec = resolve_install_spec(str(src), cls="pkg_plugin.PkgPlugin")

    result = provision(spec, cache_dir=tmp_path / "cache", mode="syspath")

    assert result.mode == "syspath"
    assert result.syspath_entry == str(src)
    assert not no_pip.packages


def test_mode_pip_on_non_package_raises(tmp_path, no_pip, cleanup_syspath):
    src = tmp_path / "plugin_src"
    cls = _make_plugin_module(src)
    spec = resolve_install_spec(str(src), cls=cls)

    with pytest.raises(ValueError, match="cannot be pip"):
        provision(spec, cache_dir=tmp_path / "cache", mode="pip")


def test_pip_install_command_editable(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(provisioning, "_run", lambda cmd, **kwargs: calls.append(cmd))
    repo = _make_package_source(tmp_path / "repo")

    _pip_install(repo, editable=True)

    assert calls == [[sys.executable, "-m", "pip", "install", "-e", str(repo)]]


def test_syspath_mode_installs_requirements(tmp_path, no_pip, cleanup_syspath):
    src = tmp_path / "plugin_src"
    cls = _make_plugin_module(src, modname="req_plugin", cls="ReqPlugin")
    (src / "requirements.txt").write_text("# none\n")
    plans = []

    def _accept(plan):
        plans.append(plan)
        return True

    result = provision(
        resolve_install_spec(str(src), cls=cls),
        cache_dir=tmp_path / "cache",
        confirm=_accept,
    )

    assert result.mode == "syspath"
    assert plans[0].requirements == src / "requirements.txt"
    assert no_pip.requirements == [src / "requirements.txt"]
    assert not no_pip.packages


def test_pip_mode_does_not_install_requirements(tmp_path, no_pip, cleanup_syspath):
    # pip resolves the package's own dependency metadata; a stray requirements.txt
    # next to it is not a second dependency list to apply.
    src = _make_package_source(tmp_path / "repo")
    (src / "requirements.txt").write_text("# none\n")

    provision(
        resolve_install_spec(str(src), cls="pkg_plugin.PkgPlugin"),
        cache_dir=tmp_path / "cache",
    )

    assert not no_pip.requirements


def test_pip_install_requirements_command(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(provisioning, "_run", lambda cmd, **kwargs: calls.append(cmd))
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("# none\n")

    _pip_install_requirements(requirements)

    assert calls == [[sys.executable, "-m", "pip", "install", "-r", str(requirements)]]


def test_pip_install_command_non_editable(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(provisioning, "_run", lambda cmd, **kwargs: calls.append(cmd))
    repo = _make_package_source(tmp_path / "repo")

    _pip_install(repo, editable=False)

    assert "-e" not in calls[0]
    assert calls[0][-1] == str(repo)


# --- subdir handling --------------------------------------------------------


def test_local_source_with_subdir(tmp_path, no_pip, cleanup_syspath):
    root = tmp_path / "repo"
    cls = _make_plugin_module(root / "src", modname="nested_plugin", cls="Nested")
    spec = resolve_install_spec(str(root), subdir="src", cls=cls)

    result = provision(spec, cache_dir=tmp_path / "cache")

    assert result.search_path == root / "src"
    assert str(root / "src") in sys.path
    assert str(root) not in sys.path


def test_subdir_escape_raises(tmp_path, no_pip, cleanup_syspath):
    src = tmp_path / "plugin_src"
    _make_plugin_module(src)
    spec = resolve_install_spec(str(src), subdir="../../etc", cls="x.Y")

    with pytest.raises(ValueError, match="escapes the plugin source root"):
        provision(spec, cache_dir=tmp_path / "cache")


def test_absolute_subdir_escape_raises(tmp_path, no_pip, cleanup_syspath):
    src = tmp_path / "plugin_src"
    _make_plugin_module(src)
    spec = resolve_install_spec(str(src), subdir="/etc", cls="x.Y")

    with pytest.raises(ValueError, match="escapes the plugin source root"):
        provision(spec, cache_dir=tmp_path / "cache")


def test_missing_local_source_raises(tmp_path, monkeypatch, no_pip):
    monkeypatch.chdir(tmp_path)
    spec = resolve_install_spec("./does-not-exist", cls="x.Y")

    with pytest.raises(FileNotFoundError):
        provision(spec, cache_dir=tmp_path / "cache")


# --- class discovery --------------------------------------------------------


def test_discovery_no_cls(tmp_path, no_pip, cleanup_syspath):
    src = tmp_path / "repo"
    expected = _make_plugin_package(src, pkg="disc_plugin", cls="DiscPlugin")

    result = provision(resolve_install_spec(str(src)), cache_dir=tmp_path / "cache")

    assert result.cls == [expected]


def test_discovery_with_subdir_no_cls(tmp_path, no_pip, cleanup_syspath):
    root = tmp_path / "repo"
    expected = _make_plugin_package(root / "src", pkg="sub_disc_plugin", cls="SubDisc")
    spec = resolve_install_spec(str(root), subdir="src")

    result = provision(spec, cache_dir=tmp_path / "cache")

    assert result.cls == [expected]


def test_discovery_multiple_classes_raises(tmp_path, no_pip, cleanup_syspath):
    src = tmp_path / "repo"
    _make_plugin_package(src, pkg="disc_plugin", cls="DiscA")
    _make_plugin_package(src, pkg="disc_plugin2", cls="DiscB")

    with pytest.raises(ValueError, match="Multiple plugin classes"):
        provision(resolve_install_spec(str(src)), cache_dir=tmp_path / "cache")


def test_discovery_no_class_raises(tmp_path, no_pip, cleanup_syspath):
    src = tmp_path / "repo"
    pkg = src / "empty_pkg"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("X = 1\n")

    with pytest.raises(ValueError, match="No BasePlugin subclass"):
        provision(resolve_install_spec(str(src)), cache_dir=tmp_path / "cache")


def test_discovery_skips_tests_dir(tmp_path, no_pip, cleanup_syspath):
    # A top-level `tests` package (even one with its own BasePlugin subclass) must be
    # skipped, so discovery still resolves the single real plugin rather than erroring.
    src = tmp_path / "repo"
    expected = _make_plugin_package(src, pkg="disc_plugin", cls="DiscPlugin")
    _make_plugin_package(src, pkg="tests", cls="TestHelperPlugin")  # must be ignored

    result = provision(resolve_install_spec(str(src)), cache_dir=tmp_path / "cache")

    assert result.cls == [expected]


def test_discovery_descends_into_src_layout(tmp_path, monkeypatch, cleanup_syspath):
    # `pyproject.toml` at the root, package under `src/`: the root itself holds no
    # importable name, so discovery has to look one level down.
    root = tmp_path / "repo"
    _make_package_source(root)
    expected = _make_plugin_package(
        root / "src", pkg="srclayout_plugin", cls="SrcLayoutPlugin"
    )
    # Stand in for the real pip install, which is what makes the package importable.
    monkeypatch.setattr(
        provisioning,
        "_pip_install",
        lambda package_root, editable: sys.path.insert(0, str(root / "src")),
    )

    result = provision(resolve_install_spec(str(root)), cache_dir=tmp_path / "cache")

    assert result.mode == "pip"
    assert result.cls == [expected]


def test_discovery_after_pip_install_says_the_package_was_installed(
    tmp_path, no_pip, cleanup_syspath
):
    src = _make_package_source(tmp_path / "repo")

    with pytest.raises(ValueError, match="was installed, but no BasePlugin subclass"):
        provision(resolve_install_spec(str(src)), cache_dir=tmp_path / "cache")


def test_cls_list_from_one_source(tmp_path, no_pip, cleanup_syspath):
    root = tmp_path / "repo"
    expected = _make_multi_plugin_package(root, classes=("Alpha", "Beta"))
    spec = resolve_install_spec(str(root), cls=expected)

    result = provision(spec, cache_dir=tmp_path / "cache")

    assert result.cls == expected


# --- git sources ------------------------------------------------------------


def test_git_clone_records_resolved_sha(tmp_path, no_pip, cleanup_syspath):
    work = tmp_path / "work"
    _make_plugin_module(work, modname="git_plugin", cls="GitPlugin")
    run_git(["init", "-q"], cwd=work)
    run_git(["add", "."], cwd=work)
    run_git(["commit", "-q", "-m", "init"], cwd=work)
    run_git(["tag", "v1"], cwd=work)
    expected_sha = run_git(["rev-parse", "HEAD"], cwd=work).stdout.strip()

    bare = tmp_path / "repo.git"  # .git suffix -> treated as a git source
    run_git(["clone", "-q", "--bare", str(work), str(bare)], cwd=tmp_path)

    cache = tmp_path / "cache"
    spec = resolve_install_spec(str(bare), ref="v1", cls="git_plugin.GitPlugin")
    result = provision(spec, cache_dir=cache)

    assert result.resolved_sha == expected_sha
    import git_plugin  # noqa: F401

    assert git_plugin.GitPlugin.name == "GitPlugin"

    record_install(
        source=result.source,
        ref=result.ref,
        resolved_sha=result.resolved_sha,
        subdir=result.subdir,
        mode=result.mode,
        syspath_entry=result.syspath_entry,
        cls=result.cls,
        cache_dir=cache,
    )
    recorded = entries(cache)
    assert [row["key"] for row in recorded] == ["git_plugin.GitPlugin"]
    assert recorded[0]["resolved_sha"] == expected_sha
    assert recorded[0]["ref"] == "v1"
    assert recorded[0]["mode"] == "syspath"


def test_option_like_ref_is_rejected():
    # `git checkout --upload-pack=...` would run an arbitrary program.
    with pytest.raises(ValueError, match="may not start with"):
        resolve_install_spec(
            "https://github.com/org/repo.git", ref="--upload-pack=touch /tmp/pwned"
        )


def test_option_like_ref_never_reaches_git(tmp_path, monkeypatch):
    ran = []
    monkeypatch.setattr(provisioning, "_run", lambda cmd, **kwargs: ran.append(cmd))
    target = tmp_path / "clone"

    with pytest.raises(ValueError, match="may not start with"):
        _clone_or_update("https://github.com/org/repo.git", "-b", target, update=False)

    assert not ran
    assert not target.exists()


def test_git_invocations_end_options_before_user_input(tmp_path, monkeypatch):
    ran = []
    monkeypatch.setattr(provisioning, "_run", lambda cmd, **kwargs: ran.append(cmd))

    _clone_or_update("--upload-pack=x.git", "v1", tmp_path / "clone", update=False)

    clone, checkout = ran
    # `--` makes git read the next argument as a repository, never as an option.
    assert clone == [
        "git",
        "clone",
        "--",
        "--upload-pack=x.git",
        str(tmp_path / "clone"),
    ]
    assert checkout == ["git", "checkout", "v1", "--"]


@pytest.mark.parametrize("bad", ["ext::sh -c id.git", "fd::17/repo.git"])
def test_remote_helper_sources_are_rejected(bad, tmp_path, monkeypatch):
    # `<name>::…` invokes a git remote helper (arbitrary program). `--` before the
    # source does not stop this, so it must be rejected before any git runs.
    monkeypatch.setattr(
        provisioning, "_run", lambda *a, **k: pytest.fail("git ran on a helper source")
    )
    with pytest.raises(ValueError, match="remote-helper"):
        resolve_install_spec(bad)
    with pytest.raises(ValueError, match="remote-helper"):
        _clone_or_update(bad, None, tmp_path / "c", update=False)


@pytest.mark.parametrize(
    "ok",
    [
        "https://github.com/o/r.git",
        "git@github.com:o/r.git",
        "https://[2001:db8::1]/r.git",  # IPv6 `::` is not remote-helper syntax
    ],
)
def test_normal_git_sources_are_accepted(ok):
    assert resolve_install_spec(ok).source == ok


def test_git_clone_reuses_cache(tmp_path, monkeypatch, no_pip, cleanup_syspath):
    work = tmp_path / "work"
    _make_plugin_module(work, modname="git_plugin", cls="GitPlugin")
    run_git(["init", "-q"], cwd=work)
    run_git(["add", "."], cwd=work)
    run_git(["commit", "-q", "-m", "init"], cwd=work)
    run_git(["tag", "v1"], cwd=work)

    bare = tmp_path / "repo.git"
    run_git(["clone", "-q", "--bare", str(work), str(bare)], cwd=tmp_path)

    cache = tmp_path / "cache"
    spec = resolve_install_spec(str(bare), ref="v1", cls="git_plugin.GitPlugin")
    first = provision(spec, cache_dir=cache)

    commands = []
    real_run = provisioning._run
    monkeypatch.setattr(
        provisioning,
        "_run",
        lambda cmd, **kwargs: (commands.append(cmd), real_run(cmd, **kwargs))[1],
    )
    second = provision(spec, cache_dir=cache)

    assert second.root == first.root
    assert not any("clone" in cmd for cmd in commands)


def test_cached_clone_interrupted_before_checkout_heals(
    tmp_path, no_pip, cleanup_syspath
):
    # A run killed between `git clone` and `git checkout ref` leaves a cached clone on
    # the default branch; the next run must check out the ref rather than reuse the
    # wrong commit forever.
    work = tmp_path / "work"
    work.mkdir()
    (work / "marker.txt").write_text("v1\n")
    run_git(["init", "-q"], cwd=work)
    run_git(["add", "."], cwd=work)
    run_git(["commit", "-q", "-m", "v1"], cwd=work)
    run_git(["tag", "rel1"], cwd=work)
    (work / "marker.txt").write_text("v2\n")
    run_git(["commit", "-q", "-am", "v2"], cwd=work)

    bare = tmp_path / "repo.git"
    run_git(["clone", "-q", "--bare", str(work), str(bare)], cwd=tmp_path)

    cache = tmp_path / "cache"
    cache.mkdir()
    clone = _clone_target(cache, str(bare), "rel1")
    run_git(["clone", "-q", str(bare), str(clone)], cwd=tmp_path)
    assert (clone / "marker.txt").read_text().strip() == "v2"

    spec = resolve_install_spec(str(bare), ref="rel1", cls="marker.X")
    provision(spec, cache_dir=cache)

    assert (clone / "marker.txt").read_text().strip() == "v1"


def test_git_update_fast_forwards_default_branch(tmp_path, no_pip, cleanup_syspath):
    # --update must advance the cached clone to the latest commit on the (moving)
    # default branch, not silently stay at the originally cloned commit.
    work = tmp_path / "work"
    work.mkdir()
    (work / "marker.txt").write_text("v1\n")
    run_git(["init", "-q"], cwd=work)
    run_git(["add", "."], cwd=work)
    run_git(["commit", "-q", "-m", "v1"], cwd=work)

    bare = tmp_path / "repo.git"
    run_git(["clone", "-q", "--bare", str(work), str(bare)], cwd=tmp_path)

    cache = tmp_path / "cache"
    spec = resolve_install_spec(str(bare), cls="marker.X")  # no ref -> default branch
    clone = provision(spec, cache_dir=cache).root
    assert (clone / "marker.txt").read_text().strip() == "v1"

    (work / "marker.txt").write_text("v2\n")
    run_git(["commit", "-q", "-am", "v2"], cwd=work)
    run_git(["push", "-q", str(bare), "HEAD"], cwd=work)

    provision(spec, cache_dir=cache)
    assert (clone / "marker.txt").read_text().strip() == "v1"

    provision(spec, cache_dir=cache, update=True)
    assert (clone / "marker.txt").read_text().strip() == "v2"


def test_git_update_with_tag_ref_stays_pinned(tmp_path, no_pip, cleanup_syspath):
    # --update with an immutable tag ref leaves HEAD detached; provisioning must not
    # error and must stay pinned even as the branch advances past it.
    work = tmp_path / "work"
    work.mkdir()
    (work / "marker.txt").write_text("v1\n")
    run_git(["init", "-q"], cwd=work)
    run_git(["add", "."], cwd=work)
    run_git(["commit", "-q", "-m", "v1"], cwd=work)
    run_git(["tag", "rel1"], cwd=work)

    bare = tmp_path / "repo.git"
    run_git(["clone", "-q", "--bare", str(work), str(bare)], cwd=tmp_path)

    (work / "marker.txt").write_text("v2\n")
    run_git(["commit", "-q", "-am", "v2"], cwd=work)
    run_git(["push", "-q", str(bare), "HEAD"], cwd=work)

    spec = resolve_install_spec(str(bare), ref="rel1", cls="marker.X")
    clone = provision(spec, cache_dir=tmp_path / "cache", update=True).root

    assert (clone / "marker.txt").read_text().strip() == "v1"


def test_local_dir_has_no_resolved_sha(tmp_path, no_pip, cleanup_syspath):
    # A plain directory must not report the SHA of an enclosing repository.
    src = tmp_path / "plugin_src"
    cls = _make_plugin_module(src)

    result = provision(
        resolve_install_spec(str(src), cls=cls), cache_dir=tmp_path / "cache"
    )

    assert result.resolved_sha is None


def test_editable_install_reports_no_sha(tmp_path, no_pip, cleanup_syspath):
    # The working tree is a local git checkout, but an editable install tracks
    # whatever it becomes, so no commit describes what is installed.
    work = _make_package_source(tmp_path / "work")
    run_git(["init", "-q"], cwd=work)
    run_git(["add", "."], cwd=work)
    run_git(["commit", "-q", "-m", "init"], cwd=work)
    plans = []

    def _accept(plan):
        plans.append(plan)
        return True

    result = provision(
        resolve_install_spec(str(work), cls="pkg_plugin.PkgPlugin"),
        cache_dir=tmp_path / "cache",
        confirm=_accept,
    )

    assert plans[0].editable is True
    assert plans[0].resolved_sha is None
    assert result.resolved_sha is None
    assert no_pip.packages == [(work, True)]


# --- confirmation and cache -------------------------------------------------


def test_confirmation_receives_plan_and_can_abort(tmp_path, no_pip, cleanup_syspath):
    src = tmp_path / "plugin_src"
    cls = _make_plugin_module(src)
    plans = []
    spec = resolve_install_spec(str(src), cls=cls)

    def _decline(plan):
        plans.append(plan)
        return False

    with pytest.raises(ProvisionAborted):
        provision(spec, cache_dir=tmp_path / "cache", confirm=_decline)

    assert len(plans) == 1
    assert plans[0].source == str(src)
    assert plans[0].mode == "syspath"
    assert plans[0].package_root is None
    assert plans[0].editable is False
    assert plans[0].requirements is None
    assert str(src) not in sys.path


def test_confirmation_runs_before_pip(tmp_path, no_pip, cleanup_syspath):
    src = _make_package_source(tmp_path / "repo")
    spec = resolve_install_spec(str(src), cls="pkg_plugin.PkgPlugin")

    with pytest.raises(ProvisionAborted):
        provision(spec, cache_dir=tmp_path / "cache", confirm=lambda plan: False)

    assert not no_pip.packages


def test_cache_dir_self_ignores(tmp_path, no_pip, cleanup_syspath):
    src = tmp_path / "plugin_src"
    cls = _make_plugin_module(src)
    cache = tmp_path / "cache"

    provision(resolve_install_spec(str(src), cls=cls), cache_dir=cache)

    assert (cache / ".gitignore").read_text().strip().endswith("*")


def test_env_overrides_cache_dir(tmp_path, monkeypatch, no_pip, cleanup_syspath):
    src = tmp_path / "plugin_src"
    cls = _make_plugin_module(src)
    env_cache = tmp_path / "env_cache"
    default_cache = isolate_default_cache(tmp_path, monkeypatch)
    monkeypatch.setenv("AXOLOTL_PLUGIN_CACHE_DIR", str(env_cache))

    provision(resolve_install_spec(str(src), cls=cls))

    assert env_cache.is_dir()
    assert not default_cache.exists()


def test_cache_dir_precedence(tmp_path, monkeypatch, no_pip, cleanup_syspath):
    src = tmp_path / "plugin_src"
    cls = _make_plugin_module(src)
    env_cache = tmp_path / "env_cache"
    explicit = tmp_path / "explicit_cache"
    default_cache = isolate_default_cache(tmp_path, monkeypatch)
    monkeypatch.setenv("AXOLOTL_PLUGIN_CACHE_DIR", str(env_cache))

    provision(resolve_install_spec(str(src), cls=cls), cache_dir=explicit)

    assert explicit.is_dir()
    assert not env_cache.exists()
    assert not default_cache.exists()


def test_default_cache_dir_is_per_user_not_per_project(tmp_path, monkeypatch):
    # A plugin is installed once and has to resolve from wherever training is
    # later launched, so the default must not depend on the current directory.
    default_cache = isolate_default_cache(tmp_path, monkeypatch)
    (tmp_path / "elsewhere").mkdir()
    monkeypatch.chdir(tmp_path)
    from_tmp = resolve_cache_dir()
    monkeypatch.chdir(tmp_path / "elsewhere")

    assert resolve_cache_dir() == from_tmp == default_cache

    monkeypatch.delenv("XDG_CACHE_HOME")
    assert resolve_cache_dir() == tmp_path / "home" / ".cache" / "axolotl" / "plugins"


def test_gitignore_not_written_into_a_populated_dir(tmp_path):
    from axolotl.integrations.plugin_manifest import ensure_cache_dir

    project = tmp_path / "myrepo"
    project.mkdir()
    (project / "train.py").write_text("x = 1\n")

    ensure_cache_dir(str(project))

    # `--cache-dir .` must not gitignore the user's whole project.
    assert not (project / ".gitignore").exists()
    # a fresh, empty cache dir still self-ignores
    empty = tmp_path / "fresh_cache"
    ensure_cache_dir(str(empty))
    assert (empty / ".gitignore").exists()
