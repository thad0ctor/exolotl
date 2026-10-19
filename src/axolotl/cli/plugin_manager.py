"""axolotl CLI for installing and managing external plugins."""

from __future__ import annotations

from typing import cast

import click

_FULL_SHA_LEN = 40
_HEX = set("0123456789abcdefABCDEF")


@click.group(name="plugins")
def plugins():
    """Install and manage external axolotl plugins."""


@plugins.command(name="install")
@click.argument("source", type=str)
@click.option("--ref", default=None, help="Git branch, tag, or commit to check out")
@click.option("--subdir", default=None, help="Subdirectory within the source to load")
@click.option(
    "--cls",
    "cls_path",
    default=None,
    multiple=True,
    help="Dotted path to the plugin class; repeatable. Auto-discovered when omitted",
)
@click.option(
    "--mode",
    type=click.Choice(["auto", "pip", "syspath"]),
    default="auto",
    help="auto pip installs a real package, else loads it from the source tree",
)
@click.option("--cache-dir", default=None, help="Where to keep fetched plugin sources")
@click.option("--update", is_flag=True, help="Re-fetch a source that is already cached")
@click.option("--yes", "-y", is_flag=True, help="Do not ask for confirmation")
def install(
    source: str,
    ref: str | None,
    subdir: str | None,
    cls_path: tuple[str, ...],
    mode: str,
    cache_dir: str | None,
    update: bool,
    yes: bool,
):
    """Install an external plugin from a git URL or a local path."""
    from axolotl.integrations.plugin_manifest import record_install, resolve_cache_dir
    from axolotl.integrations.provisioning import (
        InstallMode,
        ProvisionAborted,
        is_git_source,
        provision,
        resolve_install_spec,
    )

    spec = resolve_install_spec(
        source,
        ref=ref,
        subdir=subdir,
        cls=list(cls_path) if cls_path else None,
    )
    if is_git_source(source) and not _is_pinned(ref):
        click.secho(
            f"WARNING: {ref or 'the default branch'} is not a commit SHA. Branches "
            "and tags move, so what you install today is not guaranteed to be what "
            "you get tomorrow. Pin with --ref <40-char commit SHA>.",
            fg="yellow",
        )

    try:
        result = provision(
            spec,
            cache_dir=cache_dir,
            mode=cast(InstallMode, mode),
            update=update,
            confirm=lambda plan: _confirm(plan, yes),
        )
    except ProvisionAborted:
        click.echo("Aborted.")
        raise SystemExit(1) from None

    record_install(
        source=result.source,
        ref=result.ref,
        resolved_sha=result.resolved_sha,
        subdir=result.subdir,
        mode=result.mode,
        syspath_entry=result.syspath_entry,
        cls=result.cls,
        cache_dir=cache_dir,
    )

    click.secho("\nInstalled.", fg="green")
    click.echo("Add this to your config:\n")
    click.echo(_yaml_snippet(result))

    resolved_cache = resolve_cache_dir(cache_dir)
    if resolved_cache != resolve_cache_dir(None):
        click.secho(
            f"\nNOTE: this plugin was installed under {resolved_cache}, which is not "
            "the default cache. Training will not find it unless you export\n"
            f"  AXOLOTL_PLUGIN_CACHE_DIR={resolved_cache}",
            fg="yellow",
        )


@plugins.command(name="list")
@click.option("--cache-dir", default=None, help="Where fetched plugin sources are kept")
def list_plugins(cache_dir: str | None):
    """List externally installed plugins."""
    from axolotl.integrations.plugin_manifest import entries, manifest_path

    rows = entries(cache_dir)
    if not rows:
        click.echo(f"No external plugins recorded in {manifest_path(cache_dir)}")
        return

    header = ("CLASS", "SOURCE", "COMMIT", "MODE", "INSTALLED")

    def _cell(value) -> str:
        # A manifest can be hand-edited or shared; coerce so a non-string field
        # neither crashes `.ljust` nor injects newlines/ANSI into the table.
        return str(value if value is not None else "").replace("\n", " ")

    table = [
        (
            _cell(row.get("key")),
            _cell(row.get("source")),
            _cell(row.get("resolved_sha"))[:8],
            _cell(row.get("mode")),
            _cell(row.get("installed_at")),
        )
        for row in rows
    ]
    widths = [
        max(len(cell) for cell in column) for column in zip(header, *table, strict=True)
    ]
    for row in (header, *table):
        line = "  ".join(
            cell.ljust(width) for cell, width in zip(row, widths, strict=True)
        )
        click.echo(line.rstrip())


def _confirm(plan, yes: bool) -> bool:
    click.echo("\nAbout to install an external axolotl plugin:")
    click.echo(f"  source  : {plan.source}")
    click.echo(f"  ref     : {plan.ref or '(default branch)'}")
    if plan.editable:
        click.echo("  commit  : (editable install; tracks your working tree)")
    else:
        click.echo(f"  commit  : {plan.resolved_sha or '(not a git checkout)'}")
    if plan.subdir:
        click.echo(f"  subdir  : {plan.subdir}")
    click.echo(f"  fetched : {plan.root}")
    if plan.mode == "pip":
        editable = " -e" if plan.editable else ""
        click.echo(f"  install : pip install{editable} {plan.package_root}")
    else:
        click.echo(f"  install : add {plan.search_path} to sys.path")
    if plan.requirements:
        click.echo(f"  deps    : pip install -r {plan.requirements}")
    click.echo(
        "\nContinuing runs code from that source (its build backend, and an import "
        "of its modules to discover the plugin class). Only install plugins you trust."
    )
    return yes or click.confirm("Continue?")


def _is_pinned(ref: str | None) -> bool:
    return bool(ref and len(ref) == _FULL_SHA_LEN and set(ref) <= _HEX)


def _yaml_snippet(result) -> str:
    lines = ["plugins:"]
    if len(result.cls) == 1:
        lines.append(f"  - cls: {result.cls[0]}")
    else:
        lines.append("  - cls:")
        lines.extend(f"      - {cls_path}" for cls_path in result.cls)
    lines.append(f"    source: {result.source}")
    ref = result.resolved_sha or result.ref
    if ref:
        lines.append(f"    ref: {ref}")
    if result.subdir:
        lines.append(f"    subdir: {result.subdir}")
    return "\n".join(lines)
