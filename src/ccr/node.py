"""A node: one device with its own inbox, aggregator and decisions."""

from __future__ import annotations

import time
import uuid
from collections import deque
from collections.abc import Callable, Mapping
from typing import Any

from ccr.aggregators import AggregationResult, Aggregator, get_aggregator
from ccr.bus import BaseBus, DeliveryStatus
from ccr.canonical import canonical_json
from ccr.decision import Decision, Lineage, LineageEntry
from ccr.errors import AggregatorError, CCRError, NotConnectedError, SerializationError
from ccr.inbox import AddResult, Inbox
from ccr.models import Belief, Evidence, Message, Pair
from ccr.snapshot import Snapshot

__all__ = ["Node"]

DecisionCallback = Callable[[Decision], None]

_STATUS: dict[AddResult, DeliveryStatus] = {
    AddResult.ADDED: "delivered",
    AddResult.DUPLICATE: "duplicate",
    AddResult.CONFLICT: "conflict",
}


def _resolve_aggregator(agg: str | Aggregator, params: Mapping[str, Any] | None) -> Aggregator:
    if isinstance(agg, str):
        return get_aggregator(agg, **dict(params or {}))
    if params:
        raise AggregatorError("aggregator_params can only be used with an aggregator name")
    if isinstance(agg, type):
        raise AggregatorError("pass an aggregator instance or a registered name, not a class")
    if not isinstance(agg, Aggregator) or not isinstance(agg.name, str):
        raise AggregatorError(f"{agg!r} is not an aggregator (needs .name and .aggregate)")
    return agg


class Node:
    """One device. Observes pairs, receives pairs from others and makes its own decisions.

    Args:
        node_id: device id, stamped as ``origin`` on every own pair.
        bus: if given, the node registers itself on it.
        aggregator: a registered aggregator name or an aggregator object.
        aggregator_params: parameters for a named aggregator.
        auto_decide: decide automatically after new pairs are added.
        auto_decide_inputs: decide after this many ADDED pairs (``None`` = every pair).
        publish_decisions: publish decisions on the bus by default.
        share_observations: publish own pairs on the bus.
        inbox_max_size, on_conflict: passed to the :class:`Inbox`.
        keep_decisions: how many made / received decisions to keep.
        clock: time source for ``created_at`` on own pairs and decisions.
    """

    def __init__(
        self,
        node_id: str,
        bus: BaseBus | None = None,
        aggregator: str | Aggregator = "weighted_vote",
        *,
        aggregator_params: Mapping[str, Any] | None = None,
        auto_decide: bool = False,
        auto_decide_inputs: int | None = None,
        publish_decisions: bool = True,
        share_observations: bool = True,
        inbox_max_size: int | None = None,
        on_conflict: str = "reject",
        keep_decisions: int = 100,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if not isinstance(node_id, str) or not node_id:
            raise ValueError("node_id must be a non-empty str")
        if auto_decide_inputs is not None and (
            not isinstance(auto_decide_inputs, int)
            or isinstance(auto_decide_inputs, bool)
            or auto_decide_inputs < 1
        ):
            raise ValueError("auto_decide_inputs must be None or an int >= 1")
        if (
            not isinstance(keep_decisions, int)
            or isinstance(keep_decisions, bool)
            or keep_decisions < 0
        ):
            raise ValueError("keep_decisions must be an int >= 0")

        self._node_id = node_id
        self._aggregator = _resolve_aggregator(aggregator, aggregator_params)
        self._inbox = Inbox(max_size=inbox_max_size, on_conflict=on_conflict)
        self.auto_decide = auto_decide
        self.auto_decide_inputs = auto_decide_inputs
        self.publish_decisions = publish_decisions
        self.share_observations = share_observations
        self.clock = clock
        self._decisions: deque[Decision] = deque(maxlen=keep_decisions)
        self._received: deque[Decision] = deque(maxlen=keep_decisions)
        self._made_cbs: list[DecisionCallback] = []
        self._received_cbs: list[DecisionCallback] = []
        self._new_inputs = 0
        self._bus: BaseBus | None = None
        if bus is not None:
            bus.register(self)
            self._bus = bus

    # ------------------------------------------------------------------ accessors

    @property
    def node_id(self) -> str:
        return self._node_id

    @property
    def bus(self) -> BaseBus | None:
        return self._bus

    @property
    def inbox(self) -> Inbox:
        return self._inbox

    @property
    def aggregator(self) -> Aggregator:
        return self._aggregator

    @property
    def decisions(self) -> tuple[Decision, ...]:
        """Decisions made by this node, oldest first (bounded by ``keep_decisions``)."""
        return tuple(self._decisions)

    @property
    def received_decisions(self) -> tuple[Decision, ...]:
        """Decisions received from other nodes, oldest first (bounded)."""
        return tuple(self._received)

    def snapshot(self) -> Snapshot:
        return self._inbox.snapshot(self._node_id)

    def __repr__(self) -> str:
        return f"Node({self._node_id!r}, aggregator={self._aggregator.name!r})"

    # ------------------------------------------------------------------ observing

    def observe(self, belief: Belief, evidence: Evidence | None = None) -> Pair:
        """Record an own pair (``origin = node_id``) and share it if ``share_observations``."""
        pair = Pair(
            origin=self._node_id,
            belief=belief,
            evidence=Evidence() if evidence is None else evidence,
            created_at=self.clock(),
        )
        self._ingest(pair)
        if self.share_observations and self._bus is not None:
            self._bus.publish(Message("pair", self._node_id, body=pair))
        return pair

    def observe_value(
        self,
        claim: str,
        value: Any,
        *,
        kind: str = "observation",
        source: str | None = None,
        dominance: float = 1.0,
        payload: Mapping[str, Any] | None = None,
        **extra: Any,
    ) -> Pair:
        """Shortcut for ``observe``. ``payload`` and ``**extra`` merge into the payload."""
        merged = dict(payload or {})
        clash = sorted(set(merged) & set(extra))
        if clash:
            raise ValueError(f"keys given both in payload and as keywords: {', '.join(clash)}")
        merged.update(extra)
        return self.observe(
            Belief(claim, value),
            Evidence(kind=kind, payload=merged, source=source, dominance=dominance),
        )

    def abstain(
        self, claim: str, *, source: str | None = None, payload: Mapping[str, Any] | None = None
    ) -> Pair:
        """Record "no opinion" on ``claim``.

        Pending client confirmation: abstaining does NOT withdraw earlier opinions.
        """
        return self.observe(
            Belief(claim, abstain=True),
            Evidence(kind="abstain", source=source, payload=dict(payload or {})),
        )

    # ------------------------------------------------------------------ sharing

    def _require_bus(self) -> BaseBus:
        if self._bus is None:
            raise NotConnectedError(f"node {self._node_id!r} has no bus")
        return self._bus

    def share(self, pair: Pair) -> int:
        """Publish ``pair`` to all other nodes; returns successful deliveries."""
        return self._require_bus().publish(Message("pair", self._node_id, body=pair))

    def send_to(self, pair: Pair, node_id: str) -> bool:
        """Send ``pair`` to one node."""
        bus = self._require_bus()
        return bus.send(Message("pair", self._node_id, body=pair, recipient=node_id), node_id)

    def receive(self, msg: Message) -> DeliveryStatus:
        """Called by the bus. Pairs go to the inbox; decisions are only recorded."""
        if isinstance(msg.body, Pair):
            return _STATUS[self._ingest(msg.body)]
        # Received decisions never enter the inbox, never feed the aggregator and never
        # trigger auto_decide.
        self._received.append(msg.body)
        for cb in list(self._received_cbs):
            cb(msg.body)
        return "delivered"

    def _ingest(self, pair: Pair) -> AddResult:
        res = self._inbox.add(pair)
        if res is AddResult.ADDED and self.auto_decide:
            self._new_inputs += 1
            if self._new_inputs >= (self.auto_decide_inputs or 1):
                self.decide()
        return res

    # ------------------------------------------------------------------ deciding

    def decide(self, snapshot: Snapshot | None = None, publish: bool | None = None) -> Decision:
        """Aggregate a snapshot (default: a fresh one), store the decision and publish it.

        ``publish=None`` follows ``publish_decisions`` and skips silently without a bus;
        ``publish=True`` without a bus raises :class:`NotConnectedError`.
        """
        do_publish = self.publish_decisions if publish is None else publish
        if publish and self._bus is None:
            raise NotConnectedError(f"node {self._node_id!r} has no bus to publish to")
        snap = self._inbox.snapshot(self._node_id) if snapshot is None else snapshot
        d = self._compute(snap, self._aggregator)
        self._decisions.append(d)
        self._new_inputs = 0
        for cb in list(self._made_cbs):
            cb(d)
        if do_publish and self._bus is not None:
            self._bus.publish(Message("decision", self._node_id, body=d))
        return d

    def replay(self, snapshot: Snapshot, aggregator: str | Aggregator | None = None) -> Decision:
        """Like :meth:`decide`, but not stored, not published and no callbacks."""
        agg = self._aggregator if aggregator is None else _resolve_aggregator(aggregator, None)
        return self._compute(snapshot, agg)

    def _compute(self, snap: Snapshot, agg: Aggregator) -> Decision:
        try:
            out = agg.aggregate(snap)
        except CCRError:
            raise
        except Exception as e:
            raise AggregatorError(f"aggregator {agg.name!r} failed: {e}") from e
        if not isinstance(out, AggregationResult):
            raise AggregatorError(
                f"aggregator {agg.name!r} returned {type(out).__name__}, not AggregationResult"
            )
        if out.used_pair_ids is None:
            used = snap.pairs()
        else:
            missing = [i for i in out.used_pair_ids if i not in snap]
            if missing:
                raise AggregatorError(f"used_pair_ids not in snapshot: {missing}")
            ids = set(out.used_pair_ids)
            used = tuple(p for p in snap.pairs() if p.id in ids)  # snapshot order
        try:
            canonical_json(out.result)
            canonical_json(out.metadata)
        except SerializationError as e:
            raise AggregatorError("aggregator result is not JSON-serializable") from e
        lineage = Lineage(tuple(LineageEntry(p.id, p.origin, p.content_hash) for p in used))
        return Decision(
            id=str(uuid.uuid4()),
            node_id=self._node_id,
            result=out.result,
            lineage=lineage,
            aggregator=agg.name,
            snapshot_digest=snap.digest,
            metadata=out.metadata,
            created_at=self.clock(),
        )

    def set_aggregator(self, agg: str | Aggregator, **params: Any) -> None:
        """Swap the aggregator at runtime (same rules as the constructor)."""
        self._aggregator = _resolve_aggregator(agg, params)

    # ------------------------------------------------------------------ callbacks

    def on_decision_made(self, cb: DecisionCallback) -> Callable[[], None]:
        """Call ``cb(decision)`` after each :meth:`decide`. Returns an unsubscribe function."""
        return _subscribe(self._made_cbs, cb)

    def on_decision_received(self, cb: DecisionCallback) -> Callable[[], None]:
        """Call ``cb(decision)`` for each received decision. Returns an unsubscribe function."""
        return _subscribe(self._received_cbs, cb)


def _subscribe(cbs: list[DecisionCallback], cb: DecisionCallback) -> Callable[[], None]:
    cbs.append(cb)

    def unsubscribe() -> None:
        if cb in cbs:
            cbs.remove(cb)

    return unsubscribe
