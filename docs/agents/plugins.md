# Plugins & External Plugin Sources — Agent Reference

Plugins extend the training pipeline through hooks, loaded via the `plugins:` config key. Entries are either a dotted class path (built-in or already-importable) or a mapping referencing an **externally installed** plugin.

External plugins are a **two-step flow**: install explicitly, then reference from the config. Config load performs no fetch itself — axolotl never clones, pip installs, runs a subprocess, or hits the network at load time; it only verifies that what the config declares is already importable. (Verifying importability does `import` the plugin, so the plugin's own top-level code runs and may itself write `.pyc` files or do anything an import does — the guarantee is about axolotl, not about trusting the plugin.)

## 1. Install

```bash
axolotl plugins install <git-url|local-path> [options]
axolotl plugins install https://github.com/org/repo.git --ref 4f2a9c1e8b7d6a5c4b3a2918f7e6d5c4b3a29187
axolotl plugins install ./plugins/local_plugin --yes
axolotl plugins list          # what is installed, from the manifest
```

| Option | Meaning |
|--------|---------|
| `--ref` | Branch, tag, or commit to check out. For git sources, anything but a full 40-char SHA prints a "this ref moves" warning; local paths never warn. |
| `--subdir` | Subdirectory of the source holding the plugin package. |
| `--cls` | Dotted path to the plugin class; repeatable. Auto-discovered when omitted. |
| `--mode` | `auto` (default) pip installs a real package, else loads from the source tree; `pip` / `syspath` force one. `--mode pip` on a non-package errors. |
| `--cache-dir` | Where fetched sources + manifest live. |
| `--update` | Re-fetch an already-cached source. |
| `--yes` / `-y` | Skip the confirmation prompt. |

Local paths install editable (`pip install -e`), git sources do not. Source-tree (`syspath`) installs also `pip install -r requirements.txt` when the source root has one.

The command prints a plan (source, ref, resolved commit SHA, pip vs sys.path, deps) and asks for confirmation before any of the source's code runs. Declining exits 1 with `Aborted.`. On success it prints the YAML to paste into the config.

## 2. Reference from the config

```yaml
plugins:
  # Dotted class path — built-in or any package already on PYTHONPATH. Unchanged.
  - axolotl.integrations.liger.LigerPlugin

  # Externally installed plugin + the provenance it was installed from.
  - cls: my_plugin.MyPlugin
    source: https://github.com/org/repo.git
    ref: 4f2a9c1e8b7d6a5c4b3a2918f7e6d5c4b3a29187

  # Several classes installed from one source.
  - cls: [multi_pkg.AlphaPlugin, multi_pkg.BetaPlugin]
    source: https://github.com/org/multi-plugin.git

  # `cls` omitted — resolved from the manifest by source.
  - source: ./plugins/local_plugin
```

The string form is fully backward compatible. Mapping entries are normalized to dotted class paths before validation, so all downstream code still sees `list[str]`.

## PluginSpec Fields

A mapping entry is a **pointer to an already-installed plugin**, not an install instruction.

| Field | Required | Meaning |
|-------|----------|---------|
| `cls` | no¹ | Dotted path to the plugin class (`module.ClassName`) — a string, or a list for several classes from one `source`. Resolved from the manifest by `source` when omitted. |
| `source` | no¹ | Git URL or local path the plugin was installed from. Provenance only; never fetched at config load. Drives the install command shown in the error when the plugin is missing. |
| `ref` | no | Branch, tag, or commit it was installed at. Prefer a commit SHA. |
| `subdir` | no | Subdirectory of the source that held the package. |

¹ At least one of `cls` or `source` is required.

`pip_install:` and `update:` are **rejected** — they were install instructions, and configs no longer install. The error names the `axolotl plugins install` flag that replaced each.

If a declared plugin is not importable, config load raises `PluginNotInstalledError` with the exact `axolotl plugins install ...` command (including `--ref` / `--subdir` / `--cls`) to run. This also fires when the plugin imports but its own dependencies do not, and a `cls` naming a missing attribute reports the module and attribute — both would otherwise be swallowed by `PluginManager.register` and silently drop the plugin mid-run.

`ref:` matches either the ref recorded at install time or the commit SHA it resolved to, so the snippet the CLI prints (which uses the SHA) resolves against an install done without `--ref`.

## Auto-discovery convention

When `--cls` is omitted at install time, Axolotl imports the package(s) under the source (or its `subdir`) and uses the single `BasePlugin` subclass found. For a plugin to be discoverable: ship it as an importable package whose top-level `__init__.py` exports **exactly one** `BasePlugin` subclass. Discovery skips `tests/`, `docs/`, `examples/`, `build/`, `dist/`. Zero or multiple matches → the install fails and asks for `--cls`. Simplest discoverable layout (no `--subdir` needed):

```text
my-plugin/pyproject.toml
my-plugin/my_plugin/__init__.py   # from .plugin import MyPlugin; __all__ = ["MyPlugin"]
my-plugin/my_plugin/plugin.py     # class MyPlugin(BasePlugin): ...
```

## Cache Directory & Manifest

Per user, not per project — a plugin is installed once and must resolve from wherever training is later launched. Precedence: `--cache-dir` > `AXOLOTL_PLUGIN_CACHE_DIR` > `$XDG_CACHE_HOME/axolotl/plugins` > `~/.cache/axolotl/plugins`. There is **no config key** for this. A relative `--cache-dir`/env value is resolved to an absolute path, so prefer an absolute one to avoid surprises across working directories. A freshly created cache writes its own `.gitignore`; it will **not** add one to a directory that already holds other files, so pointing `--cache-dir` at a populated repo does not ignore your project.

Config load reads the cache but takes no flags, so `--cache-dir` only helps if training sees the same directory — the install prints a reminder to export `AXOLOTL_PLUGIN_CACHE_DIR` when it is non-default.

Every install is recorded in `<cache_dir>/manifest.json`: source, ref, resolved commit SHA, install mode, `sys.path` entry, and the classes provided. Config load reads it to resolve a bare `source:` and to restore `sys.path` for source-tree installs. Malformed entries are dropped with a warning rather than failing the run. Cloning and manifest writes are serialized with a file lock, so they are safe under multi-GPU / multi-process launches sharing one cache.

## CLI subcommands from plugins

A pip-installed plugin package can also contribute an `axolotl` subcommand via the `axolotl.cli_commands` entry point group (`[project.entry-points."axolotl.cli_commands"] my-command = "my_package.cli:my_command"`). Entry points are read from installed package metadata, so the package must be reinstalled after adding one.

## Writing a Plugin

Subclass `BasePlugin` (`src/axolotl/integrations/base.py`) and override only the hooks you need. Common ones:

| Hook | Purpose |
|------|---------|
| `register(cfg)` | Setup / validation when the plugin is loaded. |
| `get_input_args()` | Return a dotted path to a Pydantic model that adds config fields. |
| `pre_model_load(cfg)` | Monkey-patches / imports before the model is built. |
| `post_model_load(cfg, model)` | After model + adapters are loaded. |
| `get_trainer_cls(cfg)` | Provide a custom Trainer subclass. |
| `add_callbacks_post_trainer(cfg, trainer)` | Return a list of `TrainerCallback`s. |

```python
from axolotl.integrations.base import BasePlugin

class MyPlugin(BasePlugin):
    def register(self, cfg):
        ...
```

## Gotchas

- **External plugins must be self-contained.** This mechanism only loads plugins that hook in through `BasePlugin`. A plugin that requires edits to core axolotl files (trainers, builders, collators, schemas) cannot be loaded this way — those changes must be merged into axolotl itself.
- **`get_input_args()` paths are absolute imports.** If a plugin hardcodes `axolotl.integrations.<name>....`, it is designed to live inside the axolotl package, not as a standalone external source.
- **pip mode installs the directory holding `pyproject.toml`.** If a repo keeps it in a subdirectory, pass `--subdir`. A `src/` layout works without one: discovery descends into `src/` when the root holds no importable name.
- **Local paths install editable**, git sources do not — a local source is assumed to be your own working tree. An editable install records **no** commit SHA, because it tracks whatever the tree becomes.
- **Trust is at install time.** `axolotl plugins install` runs code from the source (build backend + an import for class discovery). Only install plugins you trust, and pin `--ref` to a full commit SHA.

See [docs/custom_integrations.qmd](../custom_integrations.qmd) for the full guide and built-in integration list.
