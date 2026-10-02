"""Aggregators: the only swappable part of ccr.

Rule for every aggregator, built-in or custom: **aggregation is always per claim.**
Group the snapshot's pairs by claim (``snapshot.by_claim()``) and decide each claim
independently. An aggregator must be deterministic: the same snapshot gives the same
result.

Custom aggregators can be

* a class implementing the :class:`Aggregator` protocol (usually a
  :class:`BaseAggregator` subclass), registered with :func:`register_aggregator`;
* a plain function ``fn(snapshot) -> AggregationResult`` registered with the
  :func:`aggregator` decorator (function aggregators take no parameters);
* a plugin exposed through the ``ccr.aggregators`` entry-point group, whose value is a
  factory ``factory(**params) -> Aggregator`` (for example a ``BaseAggregator`` subclass).
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from importlib.metadata import EntryPoint, entry_points
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, TypeVar, runtime_checkable

from ccr.canonical import freeze, hashable
from ccr.errors import AggregatorError, AggregatorNotFoundError, CCRError

if TYPE_CHECKING:
    from ccr.snapshot import Snapshot

__all__ = [
    "AggregationResult",
    "Aggregator",
    "BaseAggregator",
    "MeanAggregator",
    "WeightedVoteAggregator",
    "aggregator",
    "get_aggregator",
    "list_aggregators",
    "register_aggregator",
]

ENTRY_POINT_GROUP = "ccr.aggregators"


@dataclass(frozen=True)
class AggregationResult:
    """What an aggregator returns.

    Attributes:
        result: claim -> outcome. Must be JSON-serializable.
        used_pair_ids: ids of the pairs that produced the result. ``None`` means every
            snapshot pair goes into the lineage.
        metadata: free-form, JSON-serializable extra information.
    """

    result: Mapping[str, Any]
    used_pair_ids: tuple[str, ...] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.result, Mapping):
            raise AggregatorError("AggregationResult.result must be a mapping")
        if not isinstance(self.metadata, Mapping):
            raise AggregatorError("AggregationResult.metadata must be a mapping")
        if self.used_pair_ids is not None:
            if isinstance(self.used_pair_ids, str):
                raise AggregatorError("used_pair_ids must be a sequence of ids, not a str")
            ids = tuple(self.used_pair_ids)
            if not all(isinstance(i, str) for i in ids):
                raise AggregatorError("used_pair_ids must contain only str ids")
            object.__setattr__(self, "used_pair_ids", ids)
        object.__setattr__(self, "result", freeze(self.result))
        object.__setattr__(self, "metadata", freeze(self.metadata))

    def __hash__(self) -> int:
        return hash((hashable(self.result), self.used_pair_ids, hashable(self.metadata)))


@runtime_checkable
class Aggregator(Protocol):
    """Anything with a ``name`` and an ``aggregate(snapshot)`` method."""

    name: str

    def aggregate(self, snapshot: Snapshot) -> AggregationResult: ...


class BaseAggregator:
    """Convenience base class with declared, validated parameters.

    Subclasses set ``name`` and ``PARAMS`` (allowed parameter names and their defaults)
    and implement :meth:`aggregate`. Unknown parameters raise :class:`AggregatorError`;
    subclasses validate values in ``__init__`` and raise :class:`AggregatorError` too.
    Merged parameters are available as ``self.params``.
    """

    name: str = "base"
    PARAMS: ClassVar[Mapping[str, Any]] = {}

    def __init__(self, **params: Any) -> None:
        unknown = sorted(set(params) - set(self.PARAMS))
        if unknown:
            raise AggregatorError(
                f"unknown parameter(s) for aggregator {self.name!r}: {', '.join(unknown)}"
                f" (allowed: {', '.join(sorted(self.PARAMS)) or 'none'})"
            )
        self.params: Mapping[str, Any] = freeze({**self.PARAMS, **params})

    def aggregate(self, snapshot: Snapshot) -> AggregationResult:
        raise NotImplementedError

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self.params.items())
        return f"{type(self).__name__}({args})"


def check_on_empty(value: object) -> str:
    if value not in ("none", "raise"):
        raise AggregatorError(f"on_empty must be 'none' or 'raise', got {value!r}")
    return str(value)


class _FunctionAggregator:
    """Wraps a plain ``fn(snapshot) -> AggregationResult`` as an aggregator."""

    def __init__(self, name: str, fn: Callable[[Snapshot], AggregationResult]) -> None:
        self.name = name
        self._fn = fn

    def aggregate(self, snapshot: Snapshot) -> AggregationResult:
        return self._fn(snapshot)

    def __repr__(self) -> str:
        return f"<function aggregator {self.name!r}>"


# --------------------------------------------------------------------------- registry

AggregatorFactory = Callable[..., Aggregator]

_registry: dict[str, AggregatorFactory] = {}
_entry_points: dict[str, EntryPoint] | None = None
_lock = threading.RLock()


def register_aggregator(name: str, factory: AggregatorFactory, *, replace: bool = False) -> None:
    """Register ``factory(**params) -> Aggregator`` under ``name``.

    A name that is already taken (registered, built-in, or offered by an entry-point
    plugin) raises :class:`AggregatorError` unless ``replace=True`` is passed.
    """
    if not isinstance(name, str) or not name:
        raise AggregatorError("aggregator name must be a non-empty str")
    if not callable(factory):
        raise AggregatorError("aggregator factory must be callable")
    with _lock:
        if not replace and (name in _registry or name in _discover()):
            raise AggregatorError(
                f"an aggregator named {name!r} is already registered;"
                " pass replace=True to replace it"
            )
        _registry[name] = factory


def _discover() -> dict[str, EntryPoint]:
    global _entry_points
    with _lock:
        if _entry_points is None:
            try:
                eps = entry_points(group=ENTRY_POINT_GROUP)
            except Exception:  # broken metadata must not break the library
                eps = ()  # type: ignore[assignment]
            _entry_points = {ep.name: ep for ep in eps}
        return _entry_points


def _factory_for(name: str) -> AggregatorFactory:
    with _lock:
        factory = _registry.get(name)
        if factory is not None:
            return factory
        ep = _discover().get(name)
        if ep is None:
            raise AggregatorNotFoundError(
                f"no aggregator named {name!r} (available: {', '.join(list_aggregators())})"
            )
        try:
            loaded = ep.load()
        except Exception as e:
            raise AggregatorError(f"failed to load aggregator plugin {name!r}: {e}") from e
        if not callable(loaded):
            raise AggregatorError(f"aggregator plugin {name!r} is not callable")
        _registry[name] = loaded
        return loaded  # type: ignore[no-any-return]


def get_aggregator(name: str, **params: Any) -> Aggregator:
    """Build the aggregator registered as ``name`` with ``params``."""
    factory = _factory_for(name)
    try:
        agg = factory(**params)
    except CCRError:
        raise
    except Exception as e:
        raise AggregatorError(f"could not create aggregator {name!r}: {e}") from e
    if isinstance(agg, type) or not isinstance(agg, Aggregator):
        raise AggregatorError(f"factory for {name!r} did not return an aggregator")
    return agg


def list_aggregators() -> tuple[str, ...]:
    """Names of all registered and discoverable aggregators, sorted."""
    with _lock:
        return tuple(sorted(set(_registry) | set(_discover())))


F = TypeVar("F", bound=Callable[["Snapshot"], AggregationResult])


def aggregator(name: str, *, replace: bool = False) -> Callable[[F], F]:
    """Decorator: register a plain ``fn(snapshot) -> AggregationResult`` as ``name``.

    Function aggregators accept no parameters; passing any raises :class:`AggregatorError`.
    A taken name raises :class:`AggregatorError` unless ``replace=True``.
    The function itself is returned unchanged.
    """

    def decorate(fn: F) -> F:
        instance = _FunctionAggregator(name, fn)

        def factory(**params: Any) -> Aggregator:
            if params:
                raise AggregatorError(f"function aggregator {name!r} accepts no parameters")
            return instance

        register_aggregator(name, factory, replace=replace)
        return fn

    return decorate


# Built-ins (imported last: they depend on the names defined above).
from ccr.aggregators.mean import MeanAggregator  # noqa: E402
from ccr.aggregators.weighted_vote import WeightedVoteAggregator  # noqa: E402

# Registered directly so that importing ccr does not trigger plugin discovery.
_registry[WeightedVoteAggregator.name] = WeightedVoteAggregator
_registry[MeanAggregator.name] = MeanAggregator
