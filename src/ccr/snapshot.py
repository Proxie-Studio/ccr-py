"""Frozen, id-sorted views of an inbox."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable, Iterator

from ccr.canonical import canonical_json
from ccr.models import Pair

__all__ = ["Snapshot"]


class Snapshot:
    """An immutable, id-sorted set of pairs as seen by ``node_id``.

    Later inbox arrivals never change a snapshot.
    """

    __slots__ = ("_by_id", "_digest", "_node_id", "_pairs")

    def __init__(self, node_id: str, pairs: Iterable[Pair] = ()) -> None:
        if not isinstance(node_id, str) or not node_id:
            raise ValueError("Snapshot.node_id must be a non-empty str")
        ordered = tuple(sorted(pairs, key=lambda p: p.id))
        by_id: dict[str, Pair] = {}
        for p in ordered:
            if not isinstance(p, Pair):
                raise TypeError(f"Snapshot pairs must be Pair, got {type(p).__name__}")
            if p.id in by_id:
                raise ValueError(f"duplicate pair id in snapshot: {p.id!r}")
            by_id[p.id] = p
        self._node_id = node_id
        self._pairs = ordered
        self._by_id = by_id
        self._digest: str | None = None

    @property
    def node_id(self) -> str:
        return self._node_id

    @property
    def digest(self) -> str:
        """SHA-256 of the canonical JSON list of content hashes, in id order (cached)."""
        if self._digest is None:
            body = canonical_json([p.content_hash for p in self._pairs])
            self._digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
        return self._digest

    def pairs(self) -> tuple[Pair, ...]:
        return self._pairs

    def get(self, pair_id: str) -> Pair | None:
        return self._by_id.get(pair_id)

    def own(self) -> tuple[Pair, ...]:
        """Pairs created by this snapshot's node (``origin == node_id``)."""
        return tuple(p for p in self._pairs if p.origin == self._node_id)

    def others(self) -> tuple[Pair, ...]:
        return tuple(p for p in self._pairs if p.origin != self._node_id)

    def by_origin(self) -> dict[str, tuple[Pair, ...]]:
        return _group(self._pairs, lambda p: p.origin)

    def by_claim(self) -> dict[str, tuple[Pair, ...]]:
        return _group(self._pairs, lambda p: p.belief.claim)

    def filter(self, pair: Callable[[Pair], bool]) -> Snapshot:
        """A new snapshot with the pairs for which ``pair(p)`` is true. Keeps ``node_id``."""
        return Snapshot(self._node_id, (p for p in self._pairs if pair(p)))

    def __iter__(self) -> Iterator[Pair]:
        return iter(self._pairs)

    def __len__(self) -> int:
        return len(self._pairs)

    def __contains__(self, pair_id: object) -> bool:
        return isinstance(pair_id, str) and pair_id in self._by_id

    def __repr__(self) -> str:
        return f"Snapshot(node_id={self._node_id!r}, pairs={len(self._pairs)})"


def _group(pairs: tuple[Pair, ...], key: Callable[[Pair], str]) -> dict[str, tuple[Pair, ...]]:
    groups: dict[str, list[Pair]] = {}
    for p in pairs:
        groups.setdefault(key(p), []).append(p)
    return {k: tuple(groups[k]) for k in sorted(groups)}
