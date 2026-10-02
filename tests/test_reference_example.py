"""The reference example from the spec (section 12), as an integration test."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import ccr
from ccr import AggregationResult, Belief, Evidence, ExchangeBus, Node, Snapshot


def test_reference_example(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)

    bus = ExchangeBus()
    a = Node("drone-a", bus)
    b = Node("drone-b", bus)
    c = Node("ground-c", bus)

    a.observe(Belief("bridge.status", "open"), Evidence(kind="camera", dominance=0.9))
    b.observe(Belief("bridge.status", "closed"), Evidence(kind="radar", dominance=0.6))
    c.observe(Belief("bridge.status", "open"), Evidence(kind="report", dominance=0.4))

    d = c.decide()
    r = d.result["bridge.status"]
    assert r["value"] == "open"
    assert r["support"] == pytest.approx(1.3)
    assert r["total"] == pytest.approx(1.9)
    assert r["score"] == pytest.approx(0.684, abs=1e-3)
    assert (r["n_pairs"], r["n_abstain"]) == (3, 0)
    by_origin = d.lineage.by_origin()
    assert list(by_origin) == ["drone-a", "drone-b", "ground-c"]
    assert all(len(ids) == 1 for ids in by_origin.values())

    @ccr.aggregator("veto_closed")
    def veto_closed(snapshot: Snapshot) -> AggregationResult:
        result, used = {}, []
        for claim, pairs in snapshot.by_claim().items():  # per claim
            vetoes = [
                p for p in pairs if p.belief.value == "closed" and p.evidence.dominance >= 0.5
            ]
            if vetoes:
                result[claim] = "closed"
                used += vetoes
            else:
                result[claim] = "open"
                used += pairs
        return AggregationResult(result, used_pair_ids=tuple(p.id for p in used))

    b.set_aggregator("veto_closed")
    vd = b.decide()
    assert vd.result == {"bridge.status": "closed"}
    assert vd.lineage.by_origin() == {"drone-b": tuple(p.id for p in b.snapshot().own())}

    assert d.verify(c.snapshot())
    bus.export_audit("audit.jsonl")
    rows = [json.loads(x) for x in (tmp_path / "audit.jsonl").read_text().splitlines()]
    # 3 pairs x 2 recipients + 2 decisions x 2 recipients
    assert len(rows) == 10
    assert all(row["status"] == "delivered" for row in rows)
