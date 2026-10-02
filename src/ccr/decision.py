"""Decisions and their lineage (the exact pairs that produced them)."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ccr.canonical import canonical_json, freeze, hashable, thaw
from ccr.errors import LineageVerificationError, SerializationError

if TYPE_CHECKING:
    from ccr.models import Pair
    from ccr.snapshot import Snapshot

__all__ = ["Decision", "Lineage", "LineageEntry"]


def _as_mapping(d: object, model: str) -> Mapping[str, Any]:
    if not isinstance(d, Mapping):
        raise SerializationError(f"{model}.from_dict expects a mapping, got {type(d).__name__}")
    return d


def _require(d: Mapping[str, Any], key: str, model: str) -> Any:
    try:
        return d[key]
    except KeyError:
        raise SerializationError(f"{model}.from_dict: missing key {key!r}") from None


@dataclass(frozen=True)
class LineageEntry:
    """One pair that went into a decision."""

    pair_id: str
    origin: str
    content_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {"pair_id": self.pair_id, "origin": self.origin, "content_hash": self.content_hash}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> LineageEntry:
        d = _as_mapping(d, "LineageEntry")
        return cls(
            pair_id=_require(d, "pair_id", "LineageEntry"),
            origin=_require(d, "origin", "LineageEntry"),
            content_hash=_require(d, "content_hash", "LineageEntry"),
        )


@dataclass(frozen=True)
class Lineage:
    """The pairs behind a decision, in snapshot (id) order."""

    entries: tuple[LineageEntry, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "entries", tuple(self.entries))

    def pair_ids(self) -> tuple[str, ...]:
        return tuple(e.pair_id for e in self.entries)

    def by_origin(self) -> dict[str, tuple[str, ...]]:
        """Map each origin node to the ids of its pairs in the lineage (keys sorted)."""
        groups: dict[str, list[str]] = {}
        for e in self.entries:
            groups.setdefault(e.origin, []).append(e.pair_id)
        return {k: tuple(groups[k]) for k in sorted(groups)}

    def __len__(self) -> int:
        return len(self.entries)

    def to_dict(self) -> dict[str, Any]:
        return {"entries": [e.to_dict() for e in self.entries]}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Lineage:
        d = _as_mapping(d, "Lineage")
        return cls(tuple(LineageEntry.from_dict(e) for e in _require(d, "entries", "Lineage")))


@dataclass(frozen=True)
class Decision:
    """The outcome of one node aggregating one snapshot."""

    id: str
    node_id: str
    result: Mapping[str, Any]
    lineage: Lineage
    aggregator: str
    snapshot_digest: str
    metadata: Mapping[str, Any]
    created_at: float

    def __post_init__(self) -> None:
        if not isinstance(self.result, Mapping):
            raise ValueError("Decision.result must be a mapping")
        if not isinstance(self.metadata, Mapping):
            raise ValueError("Decision.metadata must be a mapping")
        if not isinstance(self.lineage, Lineage):
            raise ValueError("Decision.lineage must be a Lineage")
        if (
            not isinstance(self.created_at, (int, float))
            or isinstance(self.created_at, bool)
            or not math.isfinite(self.created_at)
        ):
            raise ValueError("Decision.created_at must be a finite real number")
        canonical_json(self.result)
        canonical_json(self.metadata)
        object.__setattr__(self, "result", freeze(self.result))
        object.__setattr__(self, "metadata", freeze(self.metadata))

    def __hash__(self) -> int:
        return hash((self.id, self.node_id, hashable(self.result), self.snapshot_digest))

    def trace(self, snapshot: Snapshot) -> tuple[Pair, ...]:
        """Return the full pairs of the lineage, in lineage order.

        Raises:
            LineageVerificationError: if a pair is missing or its content hash differs.
        """
        out: list[Pair] = []
        for e in self.lineage.entries:
            pair = snapshot.get(e.pair_id)
            if pair is None:
                raise LineageVerificationError(f"pair {e.pair_id!r} is not in the snapshot")
            if pair.content_hash != e.content_hash:
                raise LineageVerificationError(f"pair {e.pair_id!r} content hash differs")
            out.append(pair)
        return tuple(out)

    def verify(self, snapshot: Snapshot) -> bool:
        """True only if the snapshot digest and every lineage pair and hash match."""
        if snapshot.digest != self.snapshot_digest:
            return False
        try:
            self.trace(snapshot)
        except LineageVerificationError:
            return False
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "node_id": self.node_id,
            "result": thaw(self.result),
            "lineage": self.lineage.to_dict(),
            "aggregator": self.aggregator,
            "snapshot_digest": self.snapshot_digest,
            "metadata": thaw(self.metadata),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Decision:
        d = _as_mapping(d, "Decision")
        return cls(
            id=_require(d, "id", "Decision"),
            node_id=_require(d, "node_id", "Decision"),
            result=_require(d, "result", "Decision"),
            lineage=Lineage.from_dict(_require(d, "lineage", "Decision")),
            aggregator=_require(d, "aggregator", "Decision"),
            snapshot_digest=_require(d, "snapshot_digest", "Decision"),
            metadata=d.get("metadata", {}),
            created_at=_require(d, "created_at", "Decision"),
        )
