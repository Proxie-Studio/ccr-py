from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from ccr import AddResult, Inbox, InboxFullError, Pair, PairConflictError, Snapshot
from tests.conftest import make_pair, pair_lists


def test_same_content_new_id_is_duplicate() -> None:
    inbox = Inbox()
    first = make_pair(pair_id="b")
    again = Pair(first.origin, first.belief, first.evidence, id="a")
    assert inbox.add(first) is AddResult.ADDED
    assert inbox.add(again) is AddResult.DUPLICATE
    assert inbox.pairs() == (first,)
    assert inbox.stats == {"added": 1, "duplicate": 1, "conflict": 0}


def test_same_id_different_content_is_conflict() -> None:
    inbox = Inbox()
    first = make_pair(value="x", pair_id="1")
    other = make_pair(value="y", pair_id="1")
    assert inbox.add(first) is AddResult.ADDED
    assert inbox.add(other) is AddResult.CONFLICT
    assert inbox.get("1") is first
    assert inbox.conflicts == ((first, other),)
    assert inbox.stats["conflict"] == 1


def test_conflict_raise_mode() -> None:
    inbox = Inbox(on_conflict="raise")
    inbox.add(make_pair(value="x", pair_id="1"))
    with pytest.raises(PairConflictError):
        inbox.add(make_pair(value="y", pair_id="1"))
    assert inbox.pairs()[0].belief.value == "x"


def test_bad_on_conflict() -> None:
    with pytest.raises(ValueError):
        Inbox(on_conflict="replace")


def test_always_id_sorted() -> None:
    inbox = Inbox()
    for i in ["m", "c", "x", "a"]:
        inbox.add(make_pair(value=i, pair_id=i))
    assert [p.id for p in inbox.pairs()] == ["a", "c", "m", "x"]


def test_max_size() -> None:
    inbox = Inbox(max_size=2)
    inbox.add(make_pair(value=1))
    inbox.add(make_pair(value=2))
    with pytest.raises(InboxFullError):
        inbox.add(make_pair(value=3))
    # duplicates are still reported when full
    assert inbox.add(make_pair(value=1)) is AddResult.DUPLICATE


@given(pair_lists)
def test_idempotence(pairs: list[Pair]) -> None:
    once, twice = Inbox(), Inbox()
    for p in pairs:
        once.add(p)
        twice.add(p)
        twice.add(p)
    assert once.pairs() == twice.pairs()
    assert once.snapshot("n").digest == twice.snapshot("n").digest


@given(pair_lists, st.randoms(use_true_random=False))
def test_order_invariance(pairs: list[Pair], rnd: object) -> None:
    # dedupe content first: with equal-content pairs, "first observation wins" is order
    # dependent by design.
    unique = list({p.content_hash: p for p in reversed(pairs)}.values())
    shuffled = list(unique)
    rnd.shuffle(shuffled)  # type: ignore[attr-defined]
    a, b = Inbox(), Inbox()
    for p in unique:
        a.add(p)
    for p in shuffled:
        b.add(p)
    assert a.pairs() == b.pairs()
    assert a.snapshot("n").digest == b.snapshot("n").digest


def test_snapshot_is_frozen() -> None:
    inbox = Inbox()
    inbox.add(make_pair(value=1))
    snap = inbox.snapshot("n1")
    digest = snap.digest
    inbox.add(make_pair(value=2))
    assert len(snap) == 1
    assert snap.digest == digest
    assert len(inbox.snapshot("n1")) == 2


def test_snapshot_views() -> None:
    p1 = make_pair("b", origin="n1", pair_id="1")
    p2 = make_pair("a", origin="n2", pair_id="2")
    p3 = make_pair("b", origin="n2", value="w", pair_id="3")
    snap = Snapshot("n1", [p3, p1, p2])
    assert snap.pairs() == (p1, p2, p3)
    assert snap.own() == (p1,)
    assert snap.others() == (p2, p3)
    assert list(snap.by_origin()) == ["n1", "n2"]
    assert snap.by_claim() == {"a": (p2,), "b": (p1, p3)}
    assert list(snap.by_claim()) == ["a", "b"]
    assert "1" in snap and "9" not in snap
    assert snap.get("2") is p2
    assert list(snap) == [p1, p2, p3]


def test_filter_keeps_node_id() -> None:
    snap = Snapshot("n1", [make_pair(origin="n1", value=1), make_pair(origin="n2", value=2)])
    filtered = snap.filter(pair=lambda p: p.belief.value == 1)
    assert filtered.node_id == "n1"
    assert len(filtered.own()) == 1
    assert len(snap.filter(lambda p: p.origin == "n2").own()) == 0


@given(pair_lists)
def test_same_pairs_same_digest(pairs: list[Pair]) -> None:
    unique = {p.id: p for p in pairs}.values()
    a = Snapshot("x", unique)
    b = Snapshot("y", reversed(list(unique)))
    assert a.digest == b.digest


def test_empty_snapshot_is_falsy_but_valid() -> None:
    snap = Snapshot("n")
    assert not snap
    assert snap.digest
