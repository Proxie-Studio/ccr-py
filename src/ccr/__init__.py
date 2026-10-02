"""ccr — decentralised belief/evidence pairs, snapshots, aggregation and decision lineage.

Many nodes share pairs (a belief plus its evidence). Each node keeps its own inbox,
takes a frozen snapshot, runs a pluggable aggregator and produces its own decision with
a lineage. There is no central decision maker. Transport between devices is up to you:
implement :class:`BaseBus`.
"""

from ccr.aggregators import (
    AggregationResult,
    Aggregator,
    BaseAggregator,
    MeanAggregator,
    WeightedVoteAggregator,
    aggregator,
    get_aggregator,
    list_aggregators,
    register_aggregator,
)
from ccr.bus import AuditEntry, BaseBus, ExchangeBus
from ccr.canonical import canonical_json
from ccr.decision import Decision, Lineage, LineageEntry
from ccr.errors import (
    AggregatorError,
    AggregatorNotFoundError,
    CCRError,
    DuplicateNodeError,
    EmptySnapshotError,
    InboxFullError,
    LineageVerificationError,
    NotConnectedError,
    PairConflictError,
    SerializationError,
    UnknownNodeError,
)
from ccr.inbox import AddResult, Inbox
from ccr.models import Belief, Evidence, Message, Pair
from ccr.node import Node
from ccr.snapshot import Snapshot

__version__ = "1.0.0"

__all__ = [
    "AddResult",
    "AggregationResult",
    "Aggregator",
    "AggregatorError",
    "AggregatorNotFoundError",
    "AuditEntry",
    "BaseAggregator",
    "BaseBus",
    "Belief",
    "CCRError",
    "Decision",
    "DuplicateNodeError",
    "EmptySnapshotError",
    "Evidence",
    "ExchangeBus",
    "Inbox",
    "InboxFullError",
    "Lineage",
    "LineageEntry",
    "LineageVerificationError",
    "MeanAggregator",
    "Message",
    "Node",
    "NotConnectedError",
    "Pair",
    "PairConflictError",
    "SerializationError",
    "Snapshot",
    "UnknownNodeError",
    "WeightedVoteAggregator",
    "aggregator",
    "canonical_json",
    "get_aggregator",
    "list_aggregators",
    "register_aggregator",
]
