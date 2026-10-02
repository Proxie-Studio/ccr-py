"""Dominance-weighted vote per claim (the default aggregator)."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from typing import TYPE_CHECKING, Any, ClassVar

from ccr.aggregators import AggregationResult, BaseAggregator, check_on_empty
from ccr.canonical import canonical_json
from ccr.errors import AggregatorError, EmptySnapshotError

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ccr.models import Pair
    from ccr.snapshot import Snapshot

__all__ = ["WeightedVoteAggregator"]


class WeightedVoteAggregator(BaseAggregator):
    """For each claim, the value with the highest total dominance wins.

    Parameters:
        on_empty: ``"none"`` (return ``{}``) or ``"raise"`` (:class:`EmptySnapshotError`).
        min_dominance: voting pairs below this dominance are skipped (not in lineage).

    Decided claim: ``{"value", "support", "total", "score", "n_pairs", "n_abstain"}``.
    Undecided claim: ``value=None`` with ``reason`` ``"tie"``, ``"no_support"`` or
    ``"abstain"``.
    """

    name = "weighted_vote"
    PARAMS: ClassVar[Mapping[str, Any]] = {"on_empty": "none", "min_dominance": 0.0}

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self.on_empty = check_on_empty(self.params["on_empty"])
        md = self.params["min_dominance"]
        if (
            not isinstance(md, (int, float))
            or isinstance(md, bool)
            or not math.isfinite(md)
            or md < 0
        ):
            raise AggregatorError(f"min_dominance must be a finite number >= 0, got {md!r}")
        self.min_dominance = float(md)

    def aggregate(self, snapshot: Snapshot) -> AggregationResult:
        result: dict[str, Any] = {}
        used: list[Pair] = []
        for claim, pairs in snapshot.by_claim().items():  # always per claim
            # ABSTAIN (pending client confirmation): abstains are counted and kept in
            # lineage but never vote.
            abstains = [p for p in pairs if p.belief.abstain]
            voting = [
                p
                for p in pairs
                if not p.belief.abstain and p.evidence.dominance >= self.min_dominance
            ]
            used += abstains + voting
            base = {"n_pairs": len(pairs), "n_abstain": len(abstains)}

            if not voting:
                if abstains:
                    result[claim] = {"value": None, "reason": "abstain", "total": 0.0, **base}
                continue

            totals: defaultdict[str, float] = defaultdict(float)
            for p in voting:
                totals[canonical_json(p.belief.value)] += p.evidence.dominance
            total = sum(totals.values())
            if total == 0:
                result[claim] = {"value": None, "reason": "no_support", "total": 0.0, **base}
                continue

            top = max(totals.values())
            winners = sorted(v for v, w in totals.items() if math.isclose(w, top))
            if len(winners) > 1:
                result[claim] = {
                    "value": None,
                    "reason": "tie",
                    "tied_values": [json.loads(v) for v in winners],
                    "total": total,
                    **base,
                }
                continue

            result[claim] = {
                "value": json.loads(winners[0]),
                "support": top,
                "total": total,
                "score": top / total,
                **base,
            }

        if not result and self.on_empty == "raise":
            raise EmptySnapshotError("no usable pairs in snapshot")
        return AggregationResult(result, used_pair_ids=tuple(p.id for p in used))
