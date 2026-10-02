from __future__ import annotations

import dataclasses
import math

import pytest

from ccr import Belief, Evidence, Message, Pair, SerializationError, canonical_json
from tests.conftest import make_pair


def test_canonical_json_is_sorted_and_compact() -> None:
    assert canonical_json({"b": 1, "a": [1, 2.5, None, True]}) == '{"a":[1,2.5,null,true],"b":1}'
    assert canonical_json({"k": "é"}) == '{"k":"é"}'


@pytest.mark.parametrize("bad", [math.nan, math.inf, object(), {1: "x"}, {"s": {1, 2}}])
def test_canonical_json_rejects_non_json(bad: object) -> None:
    with pytest.raises(SerializationError):
        canonical_json(bad)


def test_models_are_frozen() -> None:
    p = make_pair()
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.origin = "x"  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.belief.value = "x"  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.evidence.dominance = 2.0  # type: ignore[misc]


def test_payload_is_deep_frozen() -> None:
    raw = {"a": {"b": [1, 2]}, "c": [{"d": 1}]}
    ev = Evidence(payload=raw)
    with pytest.raises(TypeError):
        ev.payload["x"] = 1  # type: ignore[index]
    with pytest.raises(TypeError):
        ev.payload["a"]["b"] = 3
    assert isinstance(ev.payload["a"]["b"], tuple)
    with pytest.raises(TypeError):
        ev.payload["c"][0]["d"] = 2
    # mutating the original dict does not leak in
    raw["a"]["b"].append(3)  # type: ignore[index]
    assert ev.payload["a"]["b"] == (1, 2)


def test_belief_value_is_deep_frozen() -> None:
    b = Belief("c", {"k": [1, 2]})
    with pytest.raises(TypeError):
        b.value["k"] = 1


@pytest.mark.parametrize("bad", [-0.1, math.nan, math.inf, -math.inf, True, False, "1", None])
def test_dominance_validation(bad: object) -> None:
    with pytest.raises(ValueError):
        Evidence(dominance=bad)  # type: ignore[arg-type]


def test_dominance_zero_and_int_accepted() -> None:
    assert Evidence(dominance=0).dominance == 0.0
    assert Evidence(dominance=2).dominance == 2.0
    assert isinstance(Evidence(dominance=2).dominance, float)


def test_abstain_with_value_raises() -> None:
    with pytest.raises(ValueError):
        Belief("c", "v", abstain=True)
    assert Belief("c", abstain=True).value is None


def test_belief_claim_must_be_non_empty() -> None:
    with pytest.raises(ValueError):
        Belief("")


def test_non_json_value_or_payload_raises() -> None:
    with pytest.raises(SerializationError):
        Belief("c", object())
    with pytest.raises(SerializationError):
        Belief("c", math.nan)
    with pytest.raises(SerializationError):
        Evidence(payload={"x": object()})
    with pytest.raises(SerializationError):
        Evidence(payload={"x": math.inf})


def test_round_trip_keeps_content_hash() -> None:
    p = make_pair("c", {"nested": [1, {"x": None}]}, payload={"a": [1, 2]}, source="cam")
    q = Pair.from_dict(p.to_dict())
    assert q == p
    assert q.content_hash == p.content_hash
    assert Belief.from_dict(p.belief.to_dict()) == p.belief
    assert Evidence.from_dict(p.evidence.to_dict()) == p.evidence


def test_message_round_trip() -> None:
    p = make_pair()
    m = Message("pair", "n1", body=p, recipient="n2")
    assert Message.from_dict(m.to_dict()) == m


def test_message_kind_must_match_body() -> None:
    with pytest.raises(ValueError):
        Message("decision", "n1", body=make_pair())


def test_content_hash_ignores_id_and_created_at() -> None:
    a = make_pair(pair_id="1")
    b = Pair(a.origin, a.belief, a.evidence, id="2", created_at=a.created_at + 10)
    assert a.content_hash == b.content_hash
    assert a.content_hash != make_pair(origin="other").content_hash


def test_pairs_are_hashable() -> None:
    p = make_pair(payload={"a": [1]})
    assert p in {p}
