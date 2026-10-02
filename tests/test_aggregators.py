from __future__ import annotations

import pytest

import ccr
from ccr import (
    AggregationResult,
    AggregatorError,
    AggregatorNotFoundError,
    BaseAggregator,
    EmptySnapshotError,
    MeanAggregator,
    Snapshot,
    WeightedVoteAggregator,
    get_aggregator,
    list_aggregators,
    register_aggregator,
)
from tests.conftest import make_pair


def snap(*pairs: object) -> Snapshot:
    return Snapshot("n1", pairs)  # type: ignore[arg-type]


# ----------------------------------------------------------------- weighted vote


def test_winner_and_score() -> None:
    s = snap(
        make_pair(value="open", dominance=0.9, origin="a"),
        make_pair(value="closed", dominance=0.6, origin="b"),
        make_pair(value="open", dominance=0.4, origin="c"),
    )
    r = WeightedVoteAggregator().aggregate(s).result["c"]
    assert r["value"] == "open"
    assert r["support"] == pytest.approx(1.3)
    assert r["total"] == pytest.approx(1.9)
    assert r["score"] == pytest.approx(1.3 / 1.9)
    assert (r["n_pairs"], r["n_abstain"]) == (3, 0)


def test_structured_values_vote_together() -> None:
    s = snap(
        make_pair(value={"b": 1, "a": [1]}, origin="a"),
        make_pair(value={"a": [1], "b": 1}, origin="b"),
        make_pair(value="other", origin="c"),
    )
    assert WeightedVoteAggregator().aggregate(s).result["c"]["value"] == {"a": (1,), "b": 1}


def test_tie() -> None:
    s = snap(make_pair(value="y", origin="a"), make_pair(value="x", origin="b"))
    r = WeightedVoteAggregator().aggregate(s).result["c"]
    assert r == {
        "value": None,
        "reason": "tie",
        "tied_values": ("x", "y"),  # frozen; to_dict() gives a list
        "total": 2.0,
        "n_pairs": 2,
        "n_abstain": 0,
    }


def test_tie_decodes_structured_values() -> None:
    s = snap(make_pair(value=[1, 2], origin="a"), make_pair(value={"k": 1}, origin="b"))
    r = WeightedVoteAggregator().aggregate(s).result["c"]
    assert r["reason"] == "tie"
    assert sorted(map(str, r["tied_values"])) == sorted(map(str, [(1, 2), {"k": 1}]))


def test_float_near_tie() -> None:
    s = snap(
        make_pair(value="x", dominance=0.1, origin="a"),
        make_pair(value="x", dominance=0.2, origin="b"),
        make_pair(value="y", dominance=0.3, origin="c"),
    )
    assert 0.1 + 0.2 != 0.3
    assert WeightedVoteAggregator().aggregate(s).result["c"]["reason"] == "tie"


def test_no_support() -> None:
    s = snap(make_pair(value="x", dominance=0, origin="a"), make_pair(value="y", dominance=0))
    r = WeightedVoteAggregator().aggregate(s).result["c"]
    assert r == {"value": None, "reason": "no_support", "total": 0.0, "n_pairs": 2, "n_abstain": 0}


def test_all_abstain() -> None:
    s = snap(make_pair(abstain=True, origin="a"), make_pair(abstain=True, origin="b"))
    out = WeightedVoteAggregator().aggregate(s)
    assert out.result["c"] == {
        "value": None,
        "reason": "abstain",
        "total": 0.0,
        "n_pairs": 2,
        "n_abstain": 2,
    }
    assert set(out.used_pair_ids or ()) == {p.id for p in s}


def test_mixed_abstain() -> None:
    ab = make_pair(abstain=True, origin="a")
    vote = make_pair(value="x", dominance=0.5, origin="b")
    out = WeightedVoteAggregator().aggregate(snap(ab, vote))
    r = out.result["c"]
    assert r["value"] == "x"
    assert r["total"] == 0.5 and r["score"] == 1.0
    assert r["n_abstain"] == 1 and r["n_pairs"] == 2
    assert ab.id in (out.used_pair_ids or ())


def test_min_dominance_skips_pairs() -> None:
    low = make_pair(value="y", dominance=0.1, origin="a")
    high = make_pair(value="x", dominance=0.5, origin="b")
    out = WeightedVoteAggregator(min_dominance=0.2).aggregate(snap(low, high))
    assert out.result["c"]["value"] == "x"
    assert out.result["c"]["n_pairs"] == 2
    assert out.used_pair_ids == (high.id,)


def test_claim_with_only_skipped_pairs_is_left_out() -> None:
    s = snap(make_pair("a", dominance=0.1), make_pair("b", dominance=1.0))
    assert list(WeightedVoteAggregator(min_dominance=0.5).aggregate(s).result) == ["b"]


def test_on_empty() -> None:
    assert WeightedVoteAggregator().aggregate(snap()).result == {}
    with pytest.raises(EmptySnapshotError):
        WeightedVoteAggregator(on_empty="raise").aggregate(snap())
    with pytest.raises(EmptySnapshotError):
        WeightedVoteAggregator(on_empty="raise", min_dominance=5).aggregate(snap(make_pair()))


def test_claims_aggregated_independently() -> None:
    s = snap(make_pair("a", "x"), make_pair("b", "y"), make_pair("b", "y", origin="n2"))
    r = WeightedVoteAggregator().aggregate(s).result
    assert r["a"]["value"] == "x" and r["a"]["n_pairs"] == 1
    assert r["b"]["value"] == "y" and r["b"]["n_pairs"] == 2


@pytest.mark.parametrize(
    "params",
    [{"bogus": 1}, {"on_empty": "maybe"}, {"min_dominance": -1}, {"min_dominance": True}],
)
def test_bad_params(params: dict[str, object]) -> None:
    with pytest.raises(AggregatorError):
        WeightedVoteAggregator(**params)
    with pytest.raises(AggregatorError):
        get_aggregator("weighted_vote", **params)


# ----------------------------------------------------------------- mean


def test_weighted_mean() -> None:
    s = snap(
        make_pair(value=10, dominance=1, origin="a"),
        make_pair(value=20.0, dominance=3, origin="b"),
        make_pair(abstain=True, origin="c"),
    )
    out = MeanAggregator().aggregate(s)
    assert out.result["c"] == {"value": 17.5, "total": 4.0, "n_pairs": 3, "n_abstain": 1}
    assert len(out.used_pair_ids or ()) == 3


@pytest.mark.parametrize("bad", ["10", True])
def test_mean_strict_rejects_non_numeric(bad: object) -> None:
    s = snap(make_pair(value=1, origin="a"), make_pair(value=bad, origin="b"))
    with pytest.raises(AggregatorError):
        MeanAggregator().aggregate(s)


def test_mean_non_strict_skips() -> None:
    good = make_pair(value=4, origin="a")
    s = snap(good, make_pair(value="x", origin="b"), make_pair(value=False, origin="c"))
    out = MeanAggregator(strict=False).aggregate(s)
    assert out.result["c"]["value"] == 4.0
    assert out.result["c"]["n_pairs"] == 3
    assert out.used_pair_ids == (good.id,)


def test_mean_shapes() -> None:
    zero = snap(make_pair(value=1, dominance=0))
    assert MeanAggregator().aggregate(zero).result["c"] == {
        "value": None,
        "reason": "no_support",
        "total": 0.0,
        "n_pairs": 1,
        "n_abstain": 0,
    }
    ab = snap(make_pair(abstain=True))
    assert MeanAggregator().aggregate(ab).result["c"]["reason"] == "abstain"
    assert MeanAggregator().aggregate(snap()).result == {}
    with pytest.raises(EmptySnapshotError):
        MeanAggregator(on_empty="raise").aggregate(snap())
    with pytest.raises(AggregatorError):
        MeanAggregator(strict="yes")


# ----------------------------------------------------------------- registry


def test_builtins_registered() -> None:
    assert {"weighted_vote", "mean"} <= set(list_aggregators())
    assert isinstance(get_aggregator("mean", strict=False), MeanAggregator)


def test_unknown_name() -> None:
    with pytest.raises(AggregatorNotFoundError):
        get_aggregator("does-not-exist")


def test_function_aggregator() -> None:
    @ccr.aggregator("test_fn_agg")
    def fn(snapshot: Snapshot) -> AggregationResult:
        return AggregationResult({"n": len(snapshot)})

    agg = get_aggregator("test_fn_agg")
    assert agg.name == "test_fn_agg"
    assert agg.aggregate(snap(make_pair())).result == {"n": 1}
    assert fn(snap()).result == {"n": 0}
    with pytest.raises(AggregatorError):
        get_aggregator("test_fn_agg", x=1)


def test_register_class_aggregator() -> None:
    class Count(BaseAggregator):
        name = "test_count"
        PARAMS = {"offset": 0}  # noqa: RUF012

        def aggregate(self, snapshot: Snapshot) -> AggregationResult:
            return AggregationResult({"n": len(snapshot) + self.params["offset"]})

    register_aggregator("test_count", Count)
    assert get_aggregator("test_count", offset=2).aggregate(snap()).result == {"n": 2}
    with pytest.raises(AggregatorError):
        get_aggregator("test_count", nope=1)


def test_aggregation_result_validation() -> None:
    with pytest.raises(AggregatorError):
        AggregationResult(["not", "a", "mapping"])  # type: ignore[arg-type]
    with pytest.raises(AggregatorError):
        AggregationResult({}, used_pair_ids="abc")  # type: ignore[arg-type]
    assert AggregationResult({}, used_pair_ids=["a"]).used_pair_ids == ("a",)  # type: ignore[arg-type]


# ----------------------------------------------------------------- registry: no silent replace


@pytest.mark.parametrize("name", ["weighted_vote", "mean"])
def test_builtins_cannot_be_replaced_silently(name: str) -> None:
    original = type(get_aggregator(name))

    with pytest.raises(AggregatorError, match="replace=True"):
        register_aggregator(name, WeightedVoteAggregator)
    with pytest.raises(AggregatorError, match="replace=True"):

        @ccr.aggregator(name)
        def impostor(snapshot: Snapshot) -> AggregationResult:
            return AggregationResult({})

    # still the original built-in after both attempts
    assert type(get_aggregator(name)) is original

    # explicit replacement is allowed
    @ccr.aggregator(name, replace=True)
    def explicit(snapshot: Snapshot) -> AggregationResult:
        return AggregationResult({"replaced": True})

    assert get_aggregator(name).aggregate(snap()).result == {"replaced": True}


def test_custom_name_cannot_be_registered_twice() -> None:
    register_aggregator("test_twice", WeightedVoteAggregator)
    with pytest.raises(AggregatorError):
        register_aggregator("test_twice", MeanAggregator)
    assert isinstance(get_aggregator("test_twice"), WeightedVoteAggregator)
    register_aggregator("test_twice", MeanAggregator, replace=True)
    assert isinstance(get_aggregator("test_twice"), MeanAggregator)


def test_entry_point_names_count_as_taken(monkeypatch: pytest.MonkeyPatch) -> None:
    import ccr.aggregators as mod

    monkeypatch.setattr(mod, "_entry_points", {"test_plugin": object()})
    assert "test_plugin" in list_aggregators()
    with pytest.raises(AggregatorError):
        register_aggregator("test_plugin", WeightedVoteAggregator)
    register_aggregator("test_plugin", WeightedVoteAggregator, replace=True)
    assert isinstance(get_aggregator("test_plugin"), WeightedVoteAggregator)


# ----------------------------------------------------------------- exact value comparison


def test_vote_compares_values_exactly() -> None:
    """1, 1.0 and True are different answers: values are compared by canonical JSON."""
    s = snap(
        make_pair(value=1, dominance=0.6, origin="a"),
        make_pair(value=1.0, dominance=0.5, origin="b"),
        make_pair(value="x", dominance=1.0, origin="c"),
    )
    r = WeightedVoteAggregator().aggregate(s).result["c"]
    # merged, 1 / 1.0 would total 1.1 and beat "x"; kept apart, "x" wins.
    assert r["value"] == "x"
    assert r["support"] == 1.0

    tie = (
        WeightedVoteAggregator()
        .aggregate(snap(make_pair(value=1, origin="a"), make_pair(value=1.0, origin="b")))
        .result["c"]
    )
    assert tie["reason"] == "tie"
    assert [type(v) for v in tie["tied_values"]] == [int, float]

    bools = (
        WeightedVoteAggregator()
        .aggregate(snap(make_pair(value=True, origin="a"), make_pair(value=1, origin="b")))
        .result["c"]
    )
    assert bools["reason"] == "tie"
    assert [type(v) for v in bools["tied_values"]] == [int, bool]
