"""Message buses.

:class:`BaseBus` is the extension point for real transports (Wi-Fi, MQTT, internet, ...):
ccr itself only does data and calculation. A transport implementation receives
:class:`~ccr.models.Message` objects (serialise them with ``msg.to_dict()`` and
:func:`~ccr.canonical.canonical_json`, rebuild with ``Message.from_dict``) and hands
incoming messages to the local node with ``node.receive(msg)``.

:class:`ExchangeBus` is in-process and synchronous, for tests and for running several
nodes in one program.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from ccr.canonical import canonical_json
from ccr.errors import DuplicateNodeError, UnknownNodeError
from ccr.inbox import AddResult
from ccr.models import Message

__all__ = ["AuditEntry", "BaseBus", "DeliveryStatus", "ExchangeBus", "Receiver"]

logger = logging.getLogger(__name__)

DeliveryStatus = Literal["delivered", "duplicate", "conflict", "failed"]

_FROM_ADD_RESULT: dict[AddResult, DeliveryStatus] = {
    AddResult.ADDED: "delivered",
    AddResult.DUPLICATE: "duplicate",
    AddResult.CONFLICT: "conflict",
}
_ERROR_MODES = ("log", "ignore", "raise")


class Receiver(Protocol):
    """What a bus delivers to: anything with a ``node_id`` and ``receive(msg)``."""

    @property
    def node_id(self) -> str: ...

    def receive(self, msg: Message) -> DeliveryStatus | AddResult | None: ...


class BaseBus(ABC):
    """Transport interface. Implement these four methods for a real network."""

    @abstractmethod
    def register(self, node: Receiver) -> None:
        """Attach ``node``. Raises :class:`DuplicateNodeError` if its id is taken."""

    @abstractmethod
    def unregister(self, node_id: str) -> None:
        """Detach a node. Raises :class:`UnknownNodeError` if it is not registered."""

    @abstractmethod
    def publish(self, msg: Message) -> int:
        """Deliver ``msg`` to every node except the sender; return successful deliveries."""

    @abstractmethod
    def send(self, msg: Message, to: str) -> bool:
        """Deliver ``msg`` to one node; return success. :class:`UnknownNodeError` if missing."""


@dataclass(frozen=True)
class AuditEntry:
    """One delivery attempt to one recipient."""

    timestamp: float
    kind: str
    sender: str
    recipient: str
    item_id: str
    status: DeliveryStatus
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "kind": self.kind,
            "sender": self.sender,
            "recipient": self.recipient,
            "item_id": self.item_id,
            "status": self.status,
            "error": self.error,
        }


class ExchangeBus(BaseBus):
    """In-process, synchronous bus. Delivers in registration order.

    Args:
        audit: record one :class:`AuditEntry` per recipient.
        on_delivery_error: what to do when ``node.receive`` raises —
            ``"log"`` (record ``failed``, log a warning, continue),
            ``"ignore"`` (record ``failed``, continue) or
            ``"raise"`` (record ``failed``, stop, re-raise).
    """

    def __init__(self, audit: bool = True, on_delivery_error: str = "log") -> None:
        if on_delivery_error not in _ERROR_MODES:
            raise ValueError(
                f"on_delivery_error must be one of {_ERROR_MODES}, got {on_delivery_error!r}"
            )
        self.audit = audit
        self.on_delivery_error = on_delivery_error
        self._nodes: dict[str, Receiver] = {}
        self._audit: list[AuditEntry] = []
        self._lock = threading.Lock()

    # -- registry

    def register(self, node: Receiver) -> None:
        with self._lock:
            if node.node_id in self._nodes:
                raise DuplicateNodeError(f"node id {node.node_id!r} is already registered")
            self._nodes[node.node_id] = node

    def unregister(self, node_id: str) -> None:
        with self._lock:
            if node_id not in self._nodes:
                raise UnknownNodeError(f"node id {node_id!r} is not registered")
            del self._nodes[node_id]

    @property
    def node_ids(self) -> tuple[str, ...]:
        """Registered node ids, in registration order."""
        with self._lock:
            return tuple(self._nodes)

    # -- delivery

    def publish(self, msg: Message) -> int:
        """Deliver to all registered nodes except the sender.

        A message with a ``recipient`` set is delivered to that node only.
        """
        if msg.recipient is not None:
            return int(self.send(msg, msg.recipient))
        with self._lock:
            targets = [n for nid, n in self._nodes.items() if nid != msg.sender]
        delivered = 0
        for node in targets:
            if self._deliver(msg, node):
                delivered += 1
        return delivered

    def send(self, msg: Message, to: str) -> bool:
        with self._lock:
            node = self._nodes.get(to)
        if node is None:
            raise UnknownNodeError(f"node id {to!r} is not registered")
        return self._deliver(msg, node)

    def _deliver(self, msg: Message, node: Receiver) -> bool:
        try:
            ret = node.receive(msg)
        except Exception as e:
            self._record(msg, node.node_id, "failed", f"{type(e).__name__}: {e}")
            if self.on_delivery_error == "raise":
                raise
            if self.on_delivery_error == "log":
                logger.warning(
                    "delivery of %s %s from %s to %s failed: %s",
                    msg.kind,
                    msg.item_id,
                    msg.sender,
                    node.node_id,
                    e,
                )
            return False
        if isinstance(ret, AddResult):
            status: DeliveryStatus = _FROM_ADD_RESULT[ret]
        elif ret in ("duplicate", "conflict"):
            status = ret
        else:
            status = "delivered"
        self._record(msg, node.node_id, status, None)
        return True

    # -- audit

    def _record(
        self, msg: Message, recipient: str, status: DeliveryStatus, error: str | None
    ) -> None:
        if not self.audit:
            return
        entry = AuditEntry(time.time(), msg.kind, msg.sender, recipient, msg.item_id, status, error)
        with self._lock:
            self._audit.append(entry)

    @property
    def audit_log(self) -> tuple[AuditEntry, ...]:
        with self._lock:
            return tuple(self._audit)

    def export_audit(self, path: str | os.PathLike[str]) -> None:
        """Write the audit log as JSON Lines (one canonical JSON object per line)."""
        entries = self.audit_log
        with open(path, "w", encoding="utf-8") as f:
            for e in entries:
                f.write(canonical_json(e.to_dict()))
                f.write("\n")
