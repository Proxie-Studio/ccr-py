"""__hash__ must agree with __eq__ for every hashable model: a == b implies hash(a) == hash(b)."""

from __future__ import annotations

from typing import Any

from hypothesis import given
from hypothesis import strategies as st

from ccr import (
    AggregationResult,
    AuditEntry,
    Belief,
    Decision,
    Evidence,
    Lineage,
    LineageEntry,
    Message,
    Pair,
)

# A small leaf pool so that equal-but-differently-typed values (1, 1.0, True) collide often.
leaves = st.sampled_from([0, 1, 0.0, 1.0, -0.0, True, False, None, "a", "b"])
json_values = st.recursive(
    leaves,
    lambda inner: st.one_of(
        st.lists(inner, max_size=3),
        st.dictionaries(st.sampled_from(["k", "j"]), inner, max_size=2),
    ),
    max_leaves=6,
)
payloads = st.dictionaries(st.sampled_from(["k", "j"]), json_values, max_size=2)
dominances = st.sampled_from([0, 0.0, 1, 1.0, 2])


def check(a: Any, b: Any) -> None:
    if a == b:
        assert hash(a) == hash(b), (a, b)


@given(st.sampled_from(["c", "d"]), json_values, json_values)
def test_belief(claim: str, v1: Any, v2: Any) -> None:
    check(Belief(claim, v1), Belief(claim, v2))


@given(payloads, payloads, dominances, dominances)
def test_evidence(p1: Any, p2: Any, d1: float, d2: float) -> None:
    check(Evidence(payload=p1, dominance=d1), Evidence(payload=p2, dominance=d2))


@given(json_values, json_values, payloads, payloads)
def test_pair_and_message(v1: Any, v2: Any, p1: Any, p2: Any) -> None:
    a = Pair("n", Belief("c", v1), Evidence(payload=p1), id="x", created_at=1.0)
    b = Pair("n", Belief("c", v2), Evidence(payload=p2), id="x", created_at=1)
    check(a, b)
    check(Message("pair", "n", body=a), Message("pair", "n", body=b))


@given(payloads, payloads, payloads, payloads)
def test_decision_and_result(r1: Any, r2: Any, m1: Any, m2: Any) -> None:
    lineage = Lineage((LineageEntry("p", "n", "h"),))
    a = Decision("d", "n", r1, lineage, "agg", "dg", m1, 1.0)
    b = Decision("d", "n", r2, lineage, "agg", "dg", m2, 1)
    check(a, b)
    check(Message("decision", "n", body=a), Message("decision", "n", body=b))
    check(AggregationResult(r1, ("p",), m1), AggregationResult(r2, ["p"], m2))  # type: ignore[arg-type]


def test_simple_models() -> None:
    check(LineageEntry("p", "n", "h"), LineageEntry("p", "n", "h"))
    check(Lineage((LineageEntry("p", "n", "h"),)), Lineage((LineageEntry("p", "n", "h"),)))
    check(
        AuditEntry(1.0, "pair", "a", "b", "i", "delivered"),
        AuditEntry(1, "pair", "a", "b", "i", "delivered"),
    )


def test_round_trip_equal_and_same_hash() -> None:
    p = Pair("n", Belief("c", {"k": [1, {"j": None}]}), Evidence(payload={"k": [True]}))
    q = Pair.from_dict(p.to_dict())
    assert p == q and hash(p) == hash(q)
    d = Decision("d", "n", {"c": {"v": [1]}}, Lineage(), "agg", "dg", {"m": [1]}, 1.0)
    e = Decision.from_dict(d.to_dict())
    assert d == e and hash(d) == hash(e)


def test_python_equality_is_looser_than_content_hash() -> None:
    """Documented: == follows Python (1 == 1.0 == True); content_hash is exact."""
    a, b = Belief("c", 1), Belief("c", 1.0)
    assert a == b and hash(a) == hash(b)
    pa = Pair("n", a, Evidence(), id="x", created_at=0.0)
    pb = Pair("n", b, Evidence(), id="x", created_at=0.0)
    assert pa == pb
    assert pa.content_hash != pb.content_hash
