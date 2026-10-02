from __future__ import annotations

import dataclasses

import pytest
from hypothesis import given

from ccr import (
    AggregationResult,
    AggregatorError,
    Belief,
    Decision,
    Evidence,
    LineageVerificationError,
    MeanAggregator,
    Node,
    Pair,
    Snapshot,
    WeightedVoteAggregator,
)
from tests.conftest import make_pair, pair_lists


class Fixed:
    """Aggregator returning a preset result."""

    name = "fixed"

    def __init__(self, out: AggregationResult) -> None:
        self.out = out

    def aggregate(self, snapshot: Snapshot) -> AggregationResult:
        return self.out


def three_node_snapshot() -> tuple[Node, Snapshot]:
    node = Node("c")
    node.inbox.add(make_pair(value="open", origin="a", dominance=0.9))
    node.inbox.add(make_pair(value="closed", origin="b", dominance=0.6))
    node.observe(Belief("c", "open"), Evidence(dominance=0.4))
    return node, node.snapshot()


def test_verify_true_on_original() -> None:
    node, snap = three_node_snapshot()
    d = node.decide()
    assert d.verify(snap)
    assert d.verify(node.snapshot())


def test_verify_false_after_change() -> None:
    node, _ = three_node_snapshot()
    d = node.decide()
    node.inbox.add(make_pair(value="new", origin="z"))
    assert not d.verify(node.snapshot())


def test_verify_false_on_altered_hash() -> None:
    node, snap = three_node_snapshot()
    d = node.decide()
    entries = list(d.lineage.entries)
    entries[0] = dataclasses.replace(entries[0], content_hash="0" * 64)
    bad = dataclasses.replace(d, lineage=dataclasses.replace(d.lineage, entries=tuple(entries)))
    assert not bad.verify(snap)
    with pytest.raises(LineageVerificationError):
        bad.trace(snap)


def test_verify_false_on_missing_pair() -> None:
    node, snap = three_node_snapshot()
    d = node.decide()
    assert not d.verify(snap.filter(lambda p: p.origin != "a"))
    with pytest.raises(LineageVerificationError):
        d.trace(snap.filter(lambda p: p.origin != "a"))


def test_trace_returns_exact_pairs() -> None:
    node, snap = three_node_snapshot()
    d = node.decide()
    assert d.trace(snap) == snap.pairs()
    assert all(a is b for a, b in zip(d.trace(snap), snap.pairs(), strict=True))


def test_by_origin() -> None:
    node, snap = three_node_snapshot()
    d = node.decide()
    expected = {o: tuple(p.id for p in ps) for o, ps in snap.by_origin().items()}
    assert d.lineage.by_origin() == expected
    assert list(d.lineage.by_origin()) == ["a", "b", "c"]


def test_lineage_in_snapshot_order_and_subset() -> None:
    node, snap = three_node_snapshot()
    ids = [p.id for p in snap]
    node.set_aggregator(Fixed(AggregationResult({}, used_pair_ids=(ids[2], ids[0]))))
    d = node.decide()
    assert d.lineage.pair_ids() == (ids[0], ids[2])


def test_bad_used_pair_ids() -> None:
    node, _ = three_node_snapshot()
    node.set_aggregator(Fixed(AggregationResult({}, used_pair_ids=("nope",))))
    with pytest.raises(AggregatorError):
        node.decide()


def test_non_json_result() -> None:
    node, _ = three_node_snapshot()
    node.set_aggregator(Fixed(AggregationResult({"x": object()})))
    with pytest.raises(AggregatorError):
        node.decide()
    node.set_aggregator(Fixed(AggregationResult({}, metadata={"x": float("nan")})))
    with pytest.raises(AggregatorError):
        node.decide()


def test_aggregator_exceptions_are_wrapped() -> None:
    class Boom:
        name = "boom"

        def aggregate(self, snapshot: Snapshot) -> AggregationResult:
            raise RuntimeError("kaboom")

    node = Node("n", aggregator=Boom())
    with pytest.raises(AggregatorError):
        node.decide()


def test_keep_decisions_bounds_history() -> None:
    node = Node("n", keep_decisions=3)
    ds = [node.decide() for _ in range(5)]
    assert node.decisions == tuple(ds[-3:])


def test_decision_round_trip() -> None:
    node, _ = three_node_snapshot()
    d = node.decide()
    again = Decision.from_dict(d.to_dict())
    assert again == d
    assert again.lineage == d.lineage


def test_decision_has_exact_fields() -> None:
    names = [f.name for f in dataclasses.fields(Decision)]
    assert names == [
        "id",
        "node_id",
        "result",
        "lineage",
        "aggregator",
        "snapshot_digest",
        "metadata",
        "created_at",
    ]


@given(pair_lists)
def test_determinism(pairs: list[Pair]) -> None:
    unique = list({p.id: p for p in pairs}.values())
    snap = Snapshot("n", unique)
    n1, n2 = Node("n"), Node("n")
    d1, d2 = n1.decide(snap), n2.replay(Snapshot("n", reversed(unique)), WeightedVoteAggregator())
    assert d1.result == d2.result
    assert d1.lineage == d2.lineage
    assert d1.snapshot_digest == d2.snapshot_digest
    m1 = n1.replay(snap, MeanAggregator(strict=False))
    m2 = n2.replay(snap, MeanAggregator(strict=False))
    assert m1.result == m2.result
    assert m1.lineage == m2.lineage
