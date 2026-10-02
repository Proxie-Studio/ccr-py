from __future__ import annotations

import pytest

from ccr import (
    AggregatorError,
    Belief,
    Decision,
    Evidence,
    ExchangeBus,
    Message,
    Node,
    NotConnectedError,
    Pair,
    Snapshot,
    WeightedVoteAggregator,
)
from tests.conftest import make_pair


def test_observe_stamps_origin_and_clock() -> None:
    n = Node("dev", clock=lambda: 42.0)
    p = n.observe(Belief("c", 1))
    assert p.origin == "dev" and p.created_at == 42.0
    assert p.evidence == Evidence()
    assert n.decide().created_at == 42.0


def test_observe_publishes_only_when_sharing() -> None:
    bus = ExchangeBus()
    a = Node("a", bus)
    quiet = Node("q", bus, share_observations=False)
    b = Node("b", bus)
    pa = a.observe(Belief("c", 1))
    pq = quiet.observe(Belief("c", 2))
    assert pa.id in b.inbox and pa.id in quiet.inbox
    assert pq.id in quiet.inbox
    assert pq.id not in b.inbox and pq.id not in a.inbox
    # manual sharing still works
    assert quiet.share(pq) == 2
    assert pq.id in b.inbox
    pq2 = quiet.observe(Belief("c", 3))
    assert quiet.send_to(pq2, "a") is True
    assert pq2.id in a.inbox and pq2.id not in b.inbox


def test_share_without_bus() -> None:
    n = Node("n")
    p = n.observe(Belief("c", 1))  # no bus: silently not shared
    with pytest.raises(NotConnectedError):
        n.share(p)
    with pytest.raises(NotConnectedError):
        n.send_to(p, "x")


def test_observe_value_mapping() -> None:
    n = Node("n")
    p = n.observe_value(
        "temp", 21.5, kind="sensor", source="t1", dominance=0.7, payload={"unit": "C"}, raw=215
    )
    assert p.belief == Belief("temp", 21.5)
    assert p.evidence == Evidence(
        kind="sensor", source="t1", dominance=0.7, payload={"unit": "C", "raw": 215}
    )
    with pytest.raises(ValueError):
        n.observe_value("temp", 1, payload={"unit": "C"}, unit="F")


def test_abstain() -> None:
    n = Node("n")
    p = n.abstain("c", source="op", payload={"why": "unsure"})
    assert p.belief.abstain and p.belief.value is None
    assert p.evidence.kind == "abstain" and p.evidence.source == "op"
    assert p.evidence.payload == {"why": "unsure"}


def test_auto_decide_every_n_added_pairs() -> None:
    bus = ExchangeBus()
    other = Node("o", bus)
    n = Node("n", bus, auto_decide=True, auto_decide_inputs=3)
    n.observe(Belief("c", 1))  # own pair counts
    other.observe(Belief("c", 2))
    assert len(n.decisions) == 0
    other.observe(Belief("c", 3))
    assert len(n.decisions) == 1
    for v in range(4, 7):
        n.observe(Belief("c", v))
    assert len(n.decisions) == 2


def test_auto_decide_default_every_pair() -> None:
    n = Node("n", auto_decide=True)
    n.observe(Belief("c", 1))
    n.observe(Belief("c", 2))
    assert len(n.decisions) == 2


def test_duplicates_and_decisions_do_not_count() -> None:
    bus = ExchangeBus()
    other = Node("o", bus)
    n = Node("n", bus, auto_decide=True, auto_decide_inputs=2)
    p = other.observe(Belief("c", 1))
    other.share(p)  # duplicate
    other.decide()  # decision arrives at n
    assert len(n.decisions) == 0
    assert len(n.received_decisions) == 1
    other.observe(Belief("c", 2))
    assert len(n.decisions) == 1


def test_received_decisions_never_enter_inbox() -> None:
    bus = ExchangeBus()
    a = Node("a", bus)
    b = Node("b", bus, auto_decide=True)
    seen: list[Decision] = []
    b.on_decision_received(seen.append)
    a.inbox.add(make_pair(origin="z"))
    d = a.decide()
    assert b.received_decisions == (d,)
    assert seen == [d]
    assert len(b.inbox) == 0
    assert b.decisions == ()


def test_received_decisions_bounded() -> None:
    bus = ExchangeBus()
    a = Node("a", bus)
    b = Node("b", bus, keep_decisions=2)
    ds = [a.decide() for _ in range(4)]
    assert b.received_decisions == tuple(ds[-2:])


def test_callbacks_and_unsubscribe() -> None:
    n = Node("n")
    made: list[Decision] = []
    unsub = n.on_decision_made(made.append)
    d1 = n.decide()
    unsub()
    unsub()  # idempotent
    n.decide()
    assert made == [d1]


def test_replay_is_side_effect_free() -> None:
    bus = ExchangeBus()
    n = Node("n", bus)
    other = Node("o", bus)
    made: list[Decision] = []
    n.on_decision_made(made.append)
    n.observe(Belief("c", 1))
    d = n.replay(n.snapshot())
    assert d.result["c"]["value"] == 1
    assert n.decisions == () and made == [] and other.received_decisions == ()
    assert n.replay(n.snapshot(), "mean").aggregator == "mean"


def test_decide_uses_given_empty_snapshot() -> None:
    n = Node("n")
    n.observe(Belief("c", 1))
    empty = Snapshot("n")
    d = n.decide(empty)
    assert d.result == {}
    assert d.snapshot_digest == empty.digest
    assert len(d.lineage) == 0


def test_decide_publish_flags() -> None:
    n = Node("n")
    n.decide()  # publish=None, no bus: fine
    with pytest.raises(NotConnectedError):
        n.decide(publish=True)
    bus = ExchangeBus()
    a = Node("a", bus, publish_decisions=False)
    b = Node("b", bus)
    a.decide()
    assert b.received_decisions == ()
    a.decide(publish=True)
    assert len(b.received_decisions) == 1
    b.decide(publish=False)
    assert a.received_decisions == ()


def test_aggregator_resolution() -> None:
    n = Node("n", aggregator="weighted_vote", aggregator_params={"min_dominance": 0.5})
    assert isinstance(n.aggregator, WeightedVoteAggregator)
    assert n.aggregator.min_dominance == 0.5
    with pytest.raises(AggregatorError):
        Node("x", aggregator=WeightedVoteAggregator(), aggregator_params={"on_empty": "raise"})
    with pytest.raises(AggregatorError):
        Node("x", aggregator=WeightedVoteAggregator)  # type: ignore[arg-type]
    with pytest.raises(AggregatorError):
        Node("x", aggregator_params={"nope": 1})
    n.set_aggregator("mean", strict=False)
    assert n.aggregator.name == "mean"
    with pytest.raises(AggregatorError):
        n.set_aggregator(WeightedVoteAggregator(), min_dominance=1)


def test_receive_returns_status() -> None:
    n = Node("n")
    p = make_pair(origin="x")
    assert n.receive(Message("pair", "x", body=p)) == "delivered"
    assert n.receive(Message("pair", "x", body=Pair(p.origin, p.belief, p.evidence))) == "duplicate"
    clash = make_pair(origin="x", value="other", pair_id=p.id)
    assert n.receive(Message("pair", "x", body=clash)) == "conflict"


def test_constructor_validation() -> None:
    with pytest.raises(ValueError):
        Node("")
    with pytest.raises(ValueError):
        Node("n", auto_decide_inputs=0)
    with pytest.raises(ValueError):
        Node("n", on_conflict="bogus")
