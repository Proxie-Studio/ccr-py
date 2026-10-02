"""Dominance-weighted mean per claim, for numeric values."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, ClassVar

from ccr.aggregators import AggregationResult, BaseAggregator, check_on_empty
from ccr.errors import AggregatorError, EmptySnapshotError

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ccr.models import Pair
    from ccr.snapshot import Snapshot

__all__ = ["MeanAggregator"]


def _is_numeric(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


class MeanAggregator(BaseAggregator):
    """For each claim, ``sum(value * dominance) / sum(dominance)``.

    Parameters:
        strict: ``True`` raises :class:`AggregatorError` on any non-numeric value;
            ``False`` skips non-numeric pairs (and leaves them out of the lineage).
        on_empty: ``"none"`` (return ``{}``) or ``"raise"`` (:class:`EmptySnapshotError`).

    Decided claim: ``{"value", "total", "n_pairs", "n_abstain"}``. Undecided claim:
    ``value=None`` with ``reason`` ``"no_support"`` or ``"abstain"``.
    """

    name = "mean"
    PARAMS: ClassVar[Mapping[str, Any]] = {"strict": True, "on_empty": "none"}

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        strict = self.params["strict"]
        if not isinstance(strict, bool):
            raise AggregatorError(f"strict must be a bool, got {strict!r}")
        self.strict = strict
        self.on_empty = check_on_empty(self.params["on_empty"])

    def aggregate(self, snapshot: Snapshot) -> AggregationResult:
        result: dict[str, Any] = {}
        used: list[Pair] = []
        for claim, pairs in snapshot.by_claim().items():  # always per claim
            abstains: list[Pair] = []  # ABSTAIN (pending client confirmation)
            numeric: list[Pair] = []
            for p in pairs:
                if p.belief.abstain:
                    abstains.append(p)
                elif _is_numeric(p.belief.value):
                    numeric.append(p)
                elif self.strict:
                    raise AggregatorError(
                        f"non-numeric value {p.belief.value!r} on claim {claim!r}"
                        f" (pair {p.id!r}); use strict=False to skip it"
                    )
            used += abstains + numeric
            base = {"n_pairs": len(pairs), "n_abstain": len(abstains)}

            if not numeric:
                if abstains:
                    result[claim] = {"value": None, "reason": "abstain", "total": 0.0, **base}
                continue

            total = math.fsum(p.evidence.dominance for p in numeric)
            if total == 0:
                result[claim] = {"value": None, "reason": "no_support", "total": 0.0, **base}
                continue

            mean = math.fsum(p.belief.value * p.evidence.dominance for p in numeric) / total
            result[claim] = {"value": mean, "total": total, **base}

        if not result and self.on_empty == "raise":
            raise EmptySnapshotError("no usable pairs in snapshot")
        return AggregationResult(result, used_pair_ids=tuple(p.id for p in used))
