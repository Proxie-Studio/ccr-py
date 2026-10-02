from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from ccr import (
    Belief,
    DuplicateNodeError,
    ExchangeBus,
    Message,
    Node,
    UnknownNodeError,
)
from tests.conftest import make_pair


class Exploding:
    def __init__(self, node_id: str) -> None:
        self.node_id = node_id

    def receive(self, msg: Message) -> None:
        raise RuntimeError("boom")


def test_publish_skips_sender_and_send_reaches_one() -> None:
    bus = ExchangeBus()
    a, b, c = Node("a", bus), Node("b", bus), Node("c", bus)
    p = make_pair(origin="a")
    assert bus.publish(Message("pair", "a", body=p)) == 2
    assert p.id not in a.inbox and p.id in b.inbox and p.id in c.inbox

    q = make_pair(origin="a", value="other")
    assert bus.send(Message("pair", "a", body=q), "c") is True
    assert q.id in c.inbox and q.id not in b.inbox


def test_registration_errors() -> None:
    bus = ExchangeBus()
    Node("a", bus)
    with pytest.raises(DuplicateNodeError):
        Node("a", bus)
    with pytest.raises(UnknownNodeError):
        bus.send(Message("pair", "a", body=make_pair()), "zzz")
    with pytest.raises(UnknownNodeError):
        bus.unregister("zzz")
    bus.unregister("a")
    assert bus.node_ids == ()


def test_registration_order() -> None:
    bus = ExchangeBus()
    for n in ["z", "a", "m"]:
        Node(n, bus)
    assert bus.node_ids == ("z", "a", "m")
    bus.publish(Message("pair", "x", body=make_pair()))
    assert [e.recipient for e in bus.audit_log] == ["z", "a", "m"]


def test_audit_statuses() -> None:
    bus = ExchangeBus()
    a = Node("a", bus)
    b = Node("b", bus)
    p = a.observe(Belief("c", 1))
    b.share(p)  # back to a: duplicate content
    conflicting = make_pair(value=2, origin="x", pair_id=p.id)
    bus.send(Message("pair", "x", body=conflicting), "b")
    d = a.decide()
    log = bus.audit_log
    assert [(e.kind, e.sender, e.recipient, e.status) for e in log] == [
        ("pair", "a", "b", "delivered"),
        ("pair", "b", "a", "duplicate"),
        ("pair", "x", "b", "conflict"),
        ("decision", "a", "b", "delivered"),
    ]
    assert log[0].item_id == p.id and log[-1].item_id == d.id
    assert all(e.error is None for e in log)


def test_audit_disabled() -> None:
    bus = ExchangeBus(audit=False)
    a = Node("a", bus)
    Node("b", bus)
    a.observe(Belief("c", 1))
    assert bus.audit_log == ()


def test_on_delivery_error_log(caplog: pytest.LogCaptureFixture) -> None:
    bus = ExchangeBus()
    bus.register(Exploding("x"))
    b = Node("b", bus)
    with caplog.at_level(logging.WARNING):
        assert bus.publish(Message("pair", "a", body=make_pair())) == 1
    assert [e.status for e in bus.audit_log] == ["failed", "delivered"]
    assert "boom" in (bus.audit_log[0].error or "")
    assert "failed" in caplog.text
    assert len(b.inbox) == 1


def test_on_delivery_error_ignore(caplog: pytest.LogCaptureFixture) -> None:
    bus = ExchangeBus(on_delivery_error="ignore")
    bus.register(Exploding("x"))
    Node("b", bus)
    with caplog.at_level(logging.WARNING):
        assert bus.publish(Message("pair", "a", body=make_pair())) == 1
    assert [e.status for e in bus.audit_log] == ["failed", "delivered"]
    assert caplog.text == ""


def test_on_delivery_error_raise() -> None:
    bus = ExchangeBus(on_delivery_error="raise")
    bus.register(Exploding("x"))
    b = Node("b", bus)
    with pytest.raises(RuntimeError):
        bus.publish(Message("pair", "a", body=make_pair()))
    assert [e.status for e in bus.audit_log] == ["failed"]
    assert len(b.inbox) == 0
    with pytest.raises(ValueError):
        ExchangeBus(on_delivery_error="explode")


def test_export_audit(tmp_path: Path) -> None:
    bus = ExchangeBus()
    a = Node("a", bus)
    Node("b", bus)
    Node("c", bus)
    a.observe(Belief("c", "é"))
    path = tmp_path / "audit.jsonl"
    bus.export_audit(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    rows = [json.loads(line) for line in lines]
    assert {r["recipient"] for r in rows} == {"b", "c"}
    assert set(rows[0]) == {
        "timestamp",
        "kind",
        "sender",
        "recipient",
        "item_id",
        "status",
        "error",
    }
