#!/usr/bin/env python3
"""The queue/executor contract — the one module both sides are allowed to know about.

This file is the portability boundary. It deliberately contains no vendor concepts:
no Claude, no Anthropic, no MCP, no Jira client, no prompt text. An ActionRecord is
plain data that says *what should happen*; an Executor (see executors/base.py) decides
*how*. Swap every executor out and this module is untouched.

Rules for changing this file:
  * No field may name a vendor, product, or API.
  * `target_system` is an opaque string resolved to an executor by config, never a
    branch in core code.
  * Every field must survive a JSON round-trip, because the queue is a plain SQLite
    table and the archive has to outlive whatever tooling wrote it.

Provenance is not optional. The transcription pipeline that feeds this queue is known
to fabricate fluent, plausible text from low-signal audio, so every action must be
traceable back to the exact audio window that produced it and carry the measured
signal quality of that window. An action you cannot audit is an action you cannot
safely approve.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from enum import Enum
from typing import Any


SCHEMA_VERSION = 1


def _utc_now() -> str:
    """ISO-8601 UTC timestamp. Stored as text so SQLite and JSON agree."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Status(str, Enum):
    """Lifecycle of an action. Inherits from str so it JSON-serialises as its value.

    PENDING   extracted, awaiting human review
    APPROVED  human said yes; eligible for an executor to pick up
    REJECTED  human said no; terminal, kept for audit and for tuning the extractor
    EXECUTED  an executor completed it; terminal
    FAILED    an executor tried and failed; retryable back to APPROVED
    """

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXECUTED = "executed"
    FAILED = "failed"


class ActionType(str, Enum):
    """What shape of change this is, independent of which system it lands in."""

    CREATE = "create"
    UPDATE = "update"
    COMMENT = "comment"
    MESSAGE = "message"
    TASK = "task"


class ValidationError(ValueError):
    """An ActionRecord is structurally unusable. Raised, never logged-and-ignored."""


@dataclass(frozen=True)
class Provenance:
    """Where this action came from, and how much to trust it.

    speech_rumble_db is the measured difference between 300-3400 Hz speech-band energy
    and sub-300 Hz rumble in the source window. Empirically on this recorder:
    >= +4 dB transcribes reliably, +1 to +3 dB degrades proper nouns, below +1 dB the
    transcript is fabricated. It is stored per-action so a reviewer can see at a glance
    whether the text is likely to be real.
    """

    source_audio: str
    start_sec: float
    end_sec: float
    speech_rumble_db: float
    transcript_excerpt: str
    extractor: str
    glossary_terms_applied: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.source_audio:
            raise ValidationError("provenance.source_audio is required")
        if self.end_sec <= self.start_sec:
            raise ValidationError(
                f"provenance window is empty or inverted: "
                f"start={self.start_sec} end={self.end_sec}"
            )
        if not self.transcript_excerpt.strip():
            raise ValidationError("provenance.transcript_excerpt is required")
        if not self.extractor:
            raise ValidationError("provenance.extractor is required")

    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Provenance":
        known = {f.name for f in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ValidationError(f"unknown provenance fields: {sorted(unknown)}")
        data = dict(raw)
        if "glossary_terms_applied" in data:
            data["glossary_terms_applied"] = tuple(data["glossary_terms_applied"])
        return cls(**data)


@dataclass
class ActionRecord:
    """One proposed unit of work, reviewable by a human and executable by an adapter."""

    title: str
    body: str
    target_system: str
    action_type: ActionType
    confidence: float
    provenance: Provenance

    target_ref: str | None = None
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = field(default_factory=_utc_now)
    status: Status = Status.PENDING
    schema_version: int = SCHEMA_VERSION

    reviewed_at: str | None = None
    review_note: str | None = None
    executed_at: str | None = None
    executor: str | None = None
    execution_result: str | None = None

    def __post_init__(self) -> None:
        # Accept plain strings from JSON/SQLite and coerce to the enums.
        if isinstance(self.action_type, str):
            self.action_type = ActionType(self.action_type)
        if isinstance(self.status, str):
            self.status = Status(self.status)
        if isinstance(self.provenance, dict):
            self.provenance = Provenance.from_dict(self.provenance)

        if not self.title.strip():
            raise ValidationError("title is required")
        if not self.body.strip():
            raise ValidationError("body is required")
        if not self.target_system.strip():
            raise ValidationError("target_system is required")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValidationError(
                f"confidence must be in [0.0, 1.0], got {self.confidence}"
            )
        if self.schema_version != SCHEMA_VERSION:
            raise ValidationError(
                f"unsupported schema_version {self.schema_version}; "
                f"this build understands {SCHEMA_VERSION}"
            )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["action_type"] = self.action_type.value
        data["status"] = self.status.value
        data["provenance"]["glossary_terms_applied"] = list(
            self.provenance.glossary_terms_applied
        )
        return data

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ActionRecord":
        known = {f.name for f in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ValidationError(f"unknown action fields: {sorted(unknown)}")
        return cls(**raw)

    @classmethod
    def from_json(cls, text: str) -> "ActionRecord":
        try:
            raw = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValidationError(f"action is not valid JSON: {exc}") from exc
        if not isinstance(raw, dict):
            raise ValidationError(f"action must be a JSON object, got {type(raw).__name__}")
        return cls.from_dict(raw)
