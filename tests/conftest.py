from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from hypothesis import strategies as st

import ccr.aggregators as aggregators_mod
from ccr import Belief, Evidence, Pair


@pytest.fixture(autouse=True)
def _isolated_registry() -> Iterator[None]:
    """Each test sees the built-in registry only; registrations don't leak between tests."""
    saved = dict(aggregators_mod._registry)
    yield
    aggregators_mod._registry.clear()
    aggregators_mod._registry.update(saved)


def make_pair(
    claim: str = "c",
    value: Any = "v",
    *,
    origin: str = "n1",
    dominance: float = 1.0,
    abstain: bool = False,
    pair_id: str | None = None,
    **evidence: Any,
) -> Pair:
    belief = Belief(claim, None if abstain else value, abstain=abstain)
    ev = Evidence(dominance=dominance, **evidence)
    if pair_id is None:
        return Pair(origin, belief, ev)
    return Pair(origin, belief, ev, id=pair_id)


claims = st.sampled_from(["a", "b", "c"])
values = st.one_of(st.sampled_from(["x", "y", "z"]), st.integers(-3, 3))
dominances = st.sampled_from([0.0, 0.25, 0.5, 1.0, 2.0])
origins = st.sampled_from(["n1", "n2", "n3"])


@st.composite
def pairs_strategy(draw: st.DrawFn) -> Pair:
    abstain = draw(st.integers(0, 4)) == 0  # ~20% abstains
    return make_pair(
        draw(claims),
        draw(values),
        origin=draw(origins),
        dominance=draw(dominances),
        abstain=abstain,
        pair_id=draw(st.uuids()).hex,
    )


pair_lists = st.lists(pairs_strategy(), max_size=25)
