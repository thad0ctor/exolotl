# Plugins & External Plugin Sources — Agent Reference

Plugins extend the training pipeline through hooks, loaded via the `plugins:` config key. Entries are either a dotted class path (built-in or already-importable) or a mapping that points at an **external git repo / local directory** that axolotl provisions automatically.

## Declaring Plugins

```yaml
plugins:
  # Dotted class path — built-in or any package already on PYTHONPATH.
  - axolotl.integrations.liger.LigerPlugin

  # External source, `cls` auto-discovered from the package.
  - source: https://github.com/org/repo.git   # git URL or local path
    ref: v1.2.0                      # branch / tag / commit (SHA recommended)

  # Explicit class + options.
  - cls: my_plugin.MyPlugin          # dotted path to the plugin class
    source: https://github.com/org/repo.git
    subdir: src                      # dir within source added to sys.path
    pip_install: editable            # false | editable | requirements
    update: false                    # re-fetch a cached git source

  # Multiple plugins from one repo (cloned once).
  - source: https://github.com/org/multi-plugin.git
    cls: [multi_pkg.AlphaPlugin, multi_pkg.BetaPlugin]

  # Local directory (no clone — resolved + path-injected).
  - source: ./plugins/local_plugin
```

The string form is unchanged and fully backward compatible. Mapping entries are normalized to dotted class paths before validation, so all downstream code still sees `list[str]`.

## PluginSpec Fields

| Field | Required | Meaning |
|-------|----------|---------|
| `cls` | no¹ | Dotted path to the plugin class (`module.ClassName`) — a string, or a list to load several classes from one `source` (cloned once). Auto-discovered from `source` when omitted. |
| `source` | no¹ | Git URL or local path. Cloned if it ends in `.git` or starts with `http(s)://`/`git://`/`ssh://`/`git@`; otherwise treated as a local path. Omit if `cls` is already importable. |
| `ref` | no | Git branch, tag, or commit. Pin to a commit SHA for reproducibility. |
| `subdir` | no | Subdirectory of the source to add to `sys.path`. |
| `pip_install` | no | `false` (default) = path injection only; `editable` = `pip install -e` (falls back to `requirements` if not a package); `requirements` = `pip install -r requirements.txt`. |
| `update` | no | Re-fetch + re-checkout a cached git source. |

¹ At least one of `cls` or `source` is required.

`pip_install` only ever installs the plugin and its deps — it never modifies the axolotl install.

## Auto-discovery convention

When `cls` is omitted, Axolotl imports the package(s) under the source (or its `subdir`) and uses the single `BasePlugin` subclass found. For a plugin to be discoverable: ship it as an importable package whose top-level `__init__.py` exports **exactly one** `BasePlugin` subclass. Discovery skips `tests/`, `docs/`, `examples/`, `build/`, `dist/`. Zero or multiple matches → it raises and asks for an explicit `cls`. Simplest discoverable layout (no `subdir` needed):

```
my-plugin/pyproject.toml
my-plugin/my_plugin/__init__.py   # from .plugin import MyPlugin; __all__ = ["MyPlugin"]
my-plugin/my_plugin/plugin.py     # class MyPlugin(BasePlugin): ...
```

## Cache Directory

Cloned sources live in `./.axolotl_plugins/` by default. Override with the `plugin_cache_dir:` config key or `AXOLOTL_PLUGIN_CACHE_DIR` env var (e.g. `~/.cache/axolotl/plugins` to share across checkouts). The cache writes its own `.gitignore`, so the axolotl checkout stays clean.

Cloning is idempotent and serialized with a file lock, so it is safe under multi-GPU / multi-process launches sharing one cache.

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
- **`pip_install: editable` installs from the source root.** If a repo keeps `pyproject.toml` in a subdirectory, point `source:` at that subdirectory.
- **Trust:** cloning runs no repo code, but importing/registering a plugin does. Only point `source:` at trusted repos and pin `ref:` to a commit.

See [docs/custom_integrations.qmd](../custom_integrations.qmd) for the full guide and built-in integration list.
