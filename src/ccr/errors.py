"""Exception hierarchy for ccr.

Every error raised deliberately by the library subclasses :class:`CCRError`.
Invalid field values on models raise the standard :class:`ValueError`.
"""

from __future__ import annotations

__all__ = [
    "AggregatorError",
    "AggregatorNotFoundError",
    "CCRError",
    "DuplicateNodeError",
    "EmptySnapshotError",
    "InboxFullError",
    "LineageVerificationError",
    "NotConnectedError",
    "PairConflictError",
    "SerializationError",
    "UnknownNodeError",
]


class CCRError(Exception):
    """Base class for all ccr errors."""


class UnknownNodeError(CCRError):
    """A bus operation referenced a node id that is not registered."""


class DuplicateNodeError(CCRError):
    """A node id is already registered on the bus."""


class NotConnectedError(CCRError):
    """A node operation needs a bus but the node has none."""


class PairConflictError(CCRError):
    """A pair arrived with a known id but different content."""


class InboxFullError(CCRError):
    """The inbox reached its ``max_size``."""


class SerializationError(CCRError):
    """A value cannot be converted to canonical JSON, or a dict cannot be decoded."""


class AggregatorError(CCRError):
    """An aggregator was misconfigured or produced an invalid result."""


class AggregatorNotFoundError(AggregatorError):
    """No aggregator is registered under the requested name."""


class EmptySnapshotError(AggregatorError):
    """The aggregator found nothing usable and ``on_empty="raise"`` was set."""


class LineageVerificationError(CCRError):
    """A decision's lineage does not match the given snapshot."""
