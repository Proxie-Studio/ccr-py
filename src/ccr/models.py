"""Core immutable models: Belief, Evidence, Pair and Message."""

from __future__ import annotations

import hashlib
import math
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from ccr.canonical import canonical_json, freeze, hashable, thaw
from ccr.decision import Decision
from ccr.errors import SerializationError

__all__ = ["Belief", "Evidence", "Message", "Pair"]


def _is_real(x: object) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _as_mapping(d: object, model: str) -> Mapping[str, Any]:
    if not isinstance(d, Mapping):
        raise SerializationError(f"{model}.from_dict expects a mapping, got {type(d).__name__}")
    return d


def _require(d: Mapping[str, Any], key: str, model: str) -> Any:
    try:
        return d[key]
    except KeyError:
        raise SerializationError(f"{model}.from_dict: missing key {key!r}") from None


@dataclass(frozen=True)
class Belief:
    """A statement about a claim.

    ``abstain=True`` means "no opinion" on the claim and requires ``value is None``.
    ``value`` must be JSON data; it is deep-frozen (dicts become read-only, lists tuples).
    """

    claim: str
    value: Any = None
    # ABSTAIN (pending client confirmation): abstaining does NOT withdraw earlier opinions.
    abstain: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.claim, str) or not self.claim:
            raise ValueError("Belief.claim must be a non-empty str")
        if not isinstance(self.abstain, bool):
            raise ValueError("Belief.abstain must be a bool")
        if self.abstain and self.value is not None:
            raise ValueError("Belief.abstain=True requires value to be None")
        canonical_json(self.value)
        object.__setattr__(self, "value", freeze(self.value))

    def __hash__(self) -> int:
        return hash((self.claim, hashable(self.value), self.abstain))

    def to_dict(self) -> dict[str, Any]:
        return {"claim": self.claim, "value": thaw(self.value), "abstain": self.abstain}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Belief:
        d = _as_mapping(d, "Belief")
        return cls(
            claim=_require(d, "claim", "Belief"),
            value=d.get("value"),
            abstain=d.get("abstain", False),
        )


@dataclass(frozen=True)
class Evidence:
    """Support for a belief. ``dominance`` is the only thing that influences aggregation.

    ``dominance`` must be a finite real number ``>= 0``; zero means "recorded, but counts
    for nothing". ``payload`` must be JSON data and is deep-frozen.
    """

    kind: str = "observation"
    payload: Mapping[str, Any] = field(default_factory=dict)
    source: str | None = None
    dominance: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.kind, str) or not self.kind:
            raise ValueError("Evidence.kind must be a non-empty str")
        if not isinstance(self.payload, Mapping):
            raise ValueError("Evidence.payload must be a mapping")
        if self.source is not None and not isinstance(self.source, str):
            raise ValueError("Evidence.source must be a str or None")
        d = self.dominance
        if not _is_real(d) or not math.isfinite(d) or d < 0:
            raise ValueError(f"Evidence.dominance must be a finite real number >= 0, got {d!r}")
        canonical_json(self.payload)
        object.__setattr__(self, "payload", freeze(self.payload))
        object.__setattr__(self, "dominance", float(d))

    def __hash__(self) -> int:
        return hash((self.kind, hashable(self.payload), self.source, self.dominance))

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "payload": thaw(self.payload),
            "source": self.source,
            "dominance": self.dominance,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Evidence:
        d = _as_mapping(d, "Evidence")
        return cls(
            kind=d.get("kind", "observation"),
            payload=d.get("payload", {}),
            source=d.get("source"),
            dominance=d.get("dominance", 1.0),
        )


@dataclass(frozen=True)
class Pair:
    """A belief plus its evidence, created by the node ``origin``.

    ``content_hash`` covers ``origin``, ``belief`` and ``evidence`` only, so a
    re-observation of the same content (new ``id`` / ``created_at``) is a duplicate.
    """

    origin: str
    belief: Belief
    evidence: Evidence
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: float = field(default_factory=time.time)
    _content_hash: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.origin, str) or not self.origin:
            raise ValueError("Pair.origin must be a non-empty str")
        if not isinstance(self.belief, Belief):
            raise ValueError("Pair.belief must be a Belief")
        if not isinstance(self.evidence, Evidence):
            raise ValueError("Pair.evidence must be an Evidence")
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("Pair.id must be a non-empty str")
        if not _is_real(self.created_at) or not math.isfinite(self.created_at):
            raise ValueError("Pair.created_at must be a finite real number")
        body = canonical_json(
            {"origin": self.origin, "belief": self.belief, "evidence": self.evidence}
        )
        object.__setattr__(self, "_content_hash", hashlib.sha256(body.encode("utf-8")).hexdigest())

    @property
    def content_hash(self) -> str:
        return self._content_hash

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "origin": self.origin,
            "belief": self.belief.to_dict(),
            "evidence": self.evidence.to_dict(),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Pair:
        d = _as_mapping(d, "Pair")
        kwargs: dict[str, Any] = {}
        if "id" in d:
            kwargs["id"] = d["id"]
        if "created_at" in d:
            kwargs["created_at"] = d["created_at"]
        return cls(
            origin=_require(d, "origin", "Pair"),
            belief=Belief.from_dict(_require(d, "belief", "Pair")),
            evidence=Evidence.from_dict(_require(d, "evidence", "Pair")),
            **kwargs,
        )


@dataclass(frozen=True)
class Message:
    """An envelope carried by a bus. ``recipient=None`` means publish to all."""

    kind: Literal["pair", "decision"]
    sender: str
    body: Pair | Decision
    recipient: str | None = None

    def __post_init__(self) -> None:
        if self.kind == "pair":
            if not isinstance(self.body, Pair):
                raise ValueError("Message.kind='pair' requires a Pair body")
        elif self.kind == "decision":
            if not isinstance(self.body, Decision):
                raise ValueError("Message.kind='decision' requires a Decision body")
        else:
            raise ValueError(f"Message.kind must be 'pair' or 'decision', got {self.kind!r}")
        if not isinstance(self.sender, str) or not self.sender:
            raise ValueError("Message.sender must be a non-empty str")
        if self.recipient is not None and (
            not isinstance(self.recipient, str) or not self.recipient
        ):
            raise ValueError("Message.recipient must be a non-empty str or None")

    @property
    def item_id(self) -> str:
        """The id of the carried pair or decision."""
        return self.body.id

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "sender": self.sender,
            "recipient": self.recipient,
            "body": self.body.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Message:
        d = _as_mapping(d, "Message")
        kind = _require(d, "kind", "Message")
        raw = _require(d, "body", "Message")
        body: Pair | Decision
        if kind == "pair":
            body = Pair.from_dict(raw)
        elif kind == "decision":
            body = Decision.from_dict(raw)
        else:
            raise SerializationError(f"Message.from_dict: unknown kind {kind!r}")
        return cls(
            kind=kind,
            sender=_require(d, "sender", "Message"),
            body=body,
            recipient=d.get("recipient"),
        )
