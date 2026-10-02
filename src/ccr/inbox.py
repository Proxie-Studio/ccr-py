"""A node's thread-safe, de-duplicating, id-sorted store of pairs."""

from __future__ import annotations

import bisect
import threading
from enum import Enum

from ccr.errors import InboxFullError, PairConflictError
from ccr.models import Pair
from ccr.snapshot import Snapshot

__all__ = ["AddResult", "Inbox"]

_CONFLICT_MODES = ("reject", "raise")


class AddResult(Enum):
    ADDED = "added"
    DUPLICATE = "duplicate"
    CONFLICT = "conflict"


class Inbox:
    """Holds pairs keyed by id, always id-sorted.

    * Same content (``content_hash``) seen before → ``DUPLICATE``, first observation kept.
    * Same id with different content → ``CONFLICT``, first-seen kept
      (or :class:`PairConflictError` when ``on_conflict="raise"``).
    """

    def __init__(self, max_size: int | None = None, on_conflict: str = "reject") -> None:
        if max_size is not None and (
            not isinstance(max_size, int) or isinstance(max_size, bool) or max_size < 0
        ):
            raise ValueError("max_size must be None or an int >= 0")
        if on_conflict not in _CONFLICT_MODES:
            raise ValueError(f"on_conflict must be one of {_CONFLICT_MODES}, got {on_conflict!r}")
        self.max_size = max_size
        self.on_conflict = on_conflict
        self._lock = threading.Lock()
        self._pairs: dict[str, Pair] = {}
        self._ids: list[str] = []
        self._hashes: set[str] = set()
        self._conflicts: list[tuple[Pair, Pair]] = []
        self._stats = {"added": 0, "duplicate": 0, "conflict": 0}

    def add(self, pair: Pair) -> AddResult:
        if not isinstance(pair, Pair):
            raise TypeError(f"Inbox.add expects a Pair, got {type(pair).__name__}")
        with self._lock:
            if pair.content_hash in self._hashes:  # first observation wins, regardless of id
                self._stats["duplicate"] += 1
                return AddResult.DUPLICATE
            existing = self._pairs.get(pair.id)
            if existing is not None:  # same id, different content
                self._conflicts.append((existing, pair))
                self._stats["conflict"] += 1
                if self.on_conflict == "raise":
                    raise PairConflictError(f"pair id {pair.id!r} already holds different content")
                return AddResult.CONFLICT  # first-seen wins
            if self.max_size is not None and len(self._pairs) >= self.max_size:
                raise InboxFullError(f"inbox is full (max_size={self.max_size})")
            bisect.insort(self._ids, pair.id)
            self._pairs[pair.id] = pair
            self._hashes.add(pair.content_hash)
            self._stats["added"] += 1
            return AddResult.ADDED

    def pairs(self) -> tuple[Pair, ...]:
        """All pairs, id-sorted."""
        with self._lock:
            return tuple(self._pairs[i] for i in self._ids)

    def get(self, pair_id: str) -> Pair | None:
        with self._lock:
            return self._pairs.get(pair_id)

    def snapshot(self, node_id: str) -> Snapshot:
        """A frozen copy of the current contents, taken under the inbox lock."""
        with self._lock:
            return Snapshot(node_id, tuple(self._pairs[i] for i in self._ids))

    @property
    def stats(self) -> dict[str, int]:
        """Counts of ``added`` / ``duplicate`` / ``conflict`` results (a copy)."""
        with self._lock:
            return dict(self._stats)

    @property
    def conflicts(self) -> tuple[tuple[Pair, Pair], ...]:
        """Every ``(existing, incoming)`` conflict seen, in arrival order."""
        with self._lock:
            return tuple(self._conflicts)

    def __len__(self) -> int:
        with self._lock:
            return len(self._pairs)

    def __contains__(self, pair_id: object) -> bool:
        with self._lock:
            return isinstance(pair_id, str) and pair_id in self._pairs
