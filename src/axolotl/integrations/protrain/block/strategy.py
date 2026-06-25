"""Strategy re-exports for the block manager.

Thin shim: `BlockMode` and `BlockStrategyMap` are owned by the shared
`types.py` data contract. This module re-exports them so callers inside
``block/`` can import a single local namespace without touching the types
module, and defines one local error type used by the dispatcher.

Paper reference: §3.1.2 — per-block activation strategy dispatcher.
"""

from __future__ import annotations

from axolotl.integrations.protrain.types import BlockMode, BlockStrategyMap


class StrategyError(RuntimeError):
    """Raised when a block-mode dispatch cannot produce a valid wrapper.

    Examples: unknown enum value, or attempting to unwrap a module
    that was never wrapped by the ProTrain dispatcher.
    """


__all__ = [
    "BlockMode",
    "BlockStrategyMap",
    "StrategyError",
]
