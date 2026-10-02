"""Canonical JSON: one exact text form for a value, so equal data always hashes equally.

Also holds small internal helpers for deep-freezing and thawing JSON-like data.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from ccr.errors import SerializationError

__all__ = ["canonical_json"]


def canonical_json(obj: object) -> str:
    """Return the canonical JSON text of ``obj``.

    Keys are sorted, separators are compact, non-ASCII is kept as is and NaN /
    infinity are rejected. Models are encoded through their ``to_dict()``.

    Raises:
        SerializationError: if ``obj`` contains anything that is not JSON data.
    """
    plain = _plain(obj)
    try:
        return json.dumps(
            plain, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        )
    except ValueError as e:
        raise SerializationError(f"value is not JSON-serializable: {e}") from e


def _plain(obj: object) -> Any:
    if obj is None or isinstance(obj, (str, bool, int, float)):
        return obj
    if isinstance(obj, Mapping):
        out: dict[str, Any] = {}
        for key, value in obj.items():
            if not isinstance(key, str):
                raise SerializationError(f"mapping keys must be str, got {type(key).__name__}")
            out[key] = _plain(value)
        return out
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        to_dict = getattr(obj, "to_dict", None)
        if callable(to_dict):
            return _plain(to_dict())
    raise SerializationError(f"cannot serialize object of type {type(obj).__name__}")


def freeze(obj: Any) -> Any:
    """Recursively turn mappings into ``MappingProxyType`` and lists into tuples."""
    if isinstance(obj, Mapping):
        return MappingProxyType({k: freeze(v) for k, v in obj.items()})
    if isinstance(obj, (list, tuple)):
        return tuple(freeze(v) for v in obj)
    return obj


def thaw(obj: Any) -> Any:
    """Inverse of :func:`freeze`: plain ``dict`` / ``list`` copies."""
    if isinstance(obj, Mapping):
        return {k: thaw(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [thaw(v) for v in obj]
    return obj


def hashable(obj: Any) -> Any:
    """A hashable stand-in for frozen data, consistent with ``==``."""
    if isinstance(obj, Mapping):
        return frozenset((k, hashable(v)) for k, v in obj.items())
    if isinstance(obj, (list, tuple)):
        return tuple(hashable(v) for v in obj)
    return obj
