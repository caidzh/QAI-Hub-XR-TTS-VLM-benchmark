"""Flatten/unflatten transformer KV caches at graph boundaries."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def flatten_kv_cache(cache: Any) -> tuple[Any, ...]:
    """Return key/value tensors ordered by layer from current or legacy cache types."""

    if cache is None:
        return ()
    legacy_method = getattr(cache, "to_legacy_cache", None)
    legacy = legacy_method() if callable(legacy_method) else cache
    flattened: list[Any] = []
    for layer in legacy:
        if not isinstance(layer, tuple | list) or len(layer) < 2:
            raise TypeError(f"Unsupported cache layer: {type(layer).__name__}")
        flattened.extend((layer[0], layer[1]))
    return tuple(flattened)


def pairs_from_flat(flattened: Iterable[Any]) -> tuple[tuple[Any, Any], ...]:
    items = tuple(flattened)
    if len(items) % 2:
        raise ValueError("Flattened KV cache must contain alternating key/value tensors")
    return tuple((items[index], items[index + 1]) for index in range(0, len(items), 2))


def dynamic_cache_from_flat(flattened: Iterable[Any]) -> Any:
    try:
        from transformers.cache_utils import DynamicCache
    except ImportError as error:
        raise RuntimeError("Transformers is required to construct a DynamicCache") from error
    return DynamicCache.from_legacy_cache(pairs_from_flat(flattened))


def cache_tensor_names(num_layers: int, prefix: str = "past") -> list[str]:
    return [
        f"{prefix}_{kind}_{layer}"
        for layer in range(num_layers)
        for kind in ("key", "value")
    ]
