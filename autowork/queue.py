#!/usr/bin/env python3
"""SQLite-backed review queue. The only thing that sits between extraction and action.

Design notes that are load-bearing, not stylistic:

* The canonical record is JSON in the `payload` column. Indexed columns beside it are
  denormalised copies used only for filtering. Reads always reconstruct from `payload`,
  so a schema change here never rewrites history and the archive stays self-describing
  long after this code is gone.

* Enqueue is idempotent on `dedupe_key` (source audio + normalised grounding quote). The
  pipeline is
  designed to be re-runnable after a failure, so re-processing a day must not duplicate
  actions. Re-enqueueing a known key returns the existing id and leaves its review state
  untouched -- an approval is never silently reset by a re-run.

* State transitions are checked against an explicit table. An invalid transition raises
  rather than being coerced, because every path out of PENDING is a decision someone is
  accountable for.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

from autowork.action import ActionRecord, Status
from autowork.extract import _normalise, source_key, titles_overlap

SCHEMA = """
CREATE TABLE IF NOT EXISTS actions (
    id             TEXT PRIMARY KEY,
    dedupe_key     TEXT NOT NULL UNIQUE,
    status         TEXT NOT NULL,
    target_system  TEXT NOT NULL,
    confidence     REAL NOT NULL,
    created_at     TEXT NOT NULL,
    source_audio   TEXT NOT NULL,
    payload        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_actions_status ON actions(status);
CREATE INDEX IF NOT EXISTS idx_actions_target ON actions(target_system);
CREATE INDEX IF NOT EXISTS idx_actions_source ON actions(source_audio);
"""

# Terminal states have no outgoing edges except the explicit FAILED -> APPROVED retry.
ALLOWED_TRANSITIONS: dict[Status, frozenset[Status]] = {
    Status.PENDING: frozenset({Status.APPROVED, Status.REJECTED, Status.DONE}),
    Status.APPROVED: frozenset({Status.EXECUTED, Status.FAILED, Status.REJECTED, Status.DONE}),
    Status.FAILED: frozenset({Status.APPROVED, Status.REJECTED, Status.DONE}),
    Status.REJECTED: frozenset(),
    Status.DONE: frozenset(),
    Status.EXECUTED: frozenset(),
}


class QueueError(RuntimeError):
    """Queue-level failure: unknown id, or an illegal state transition."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def dedupe_key(action: ActionRecord) -> str:
    """Stable identity: the same GROUNDING QUOTE from the same recording is the same
    action, whatever the model titled it.

    Titles proved useless as identity: measured, chunk overlap re-extracted one
    commitment as "Add quote quality threshold and top-ranked selection", "Add quote
    quality threshold and rank quotes" and "Set a quality threshold for quotes" -- three
    queue entries for one sentence someone spoke. The quote is the anchor the whole
    grounding design already trusts, so it is the identity too. Normalised the same way
    grounding normalises, so punctuation drift does not defeat it.

    Deliberately excludes title, body, confidence and the segment window: re-running
    with a better model or different chunking should change neither the identity nor an
    existing review decision.
    """
    p = action.provenance
    quote = _normalise(p.transcript_excerpt)
    raw = f"{p.source_audio}|{quote}"
    return sha256(raw.encode("utf-8")).hexdigest()


class ReviewQueue:
    """Use as a context manager so the connection is always closed."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def __enter__(self) -> ReviewQueue:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        self._conn.close()

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        try:
            with self._conn:
                yield self._conn
        except sqlite3.Error as exc:
            raise QueueError(f"sqlite failure on {self.db_path}: {exc}") from exc

    # ---- writes -------------------------------------------------------------

    def enqueue(self, action: ActionRecord) -> str:
        """Insert, or return the id of the existing action with the same dedupe_key."""
        key = dedupe_key(action)
        existing = self._conn.execute(
            "SELECT id FROM actions WHERE dedupe_key = ?", (key,)
        ).fetchone()
        if existing is not None:
            return str(existing["id"])

        with self._tx() as conn:
            conn.execute(
                "INSERT INTO actions (id, dedupe_key, status, target_system, "
                "confidence, created_at, source_audio, payload) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    action.id,
                    key,
                    action.status.value,
                    action.target_system,
                    action.confidence,
                    action.created_at,
                    action.provenance.source_audio,
                    action.to_json(),
                ),
            )
        return action.id

    def _transition(
        self,
        action_id: str,
        new_status: Status,
        **updates: str | None,
    ) -> ActionRecord:
        action = self.get(action_id)
        permitted = ALLOWED_TRANSITIONS[action.status]
        if new_status not in permitted:
            allowed = sorted(s.value for s in permitted)
            raise QueueError(
                f"illegal transition {action.status.value} -> {new_status.value} "
                f"for action {action_id}; permitted: "
                f"{allowed if allowed else 'none (terminal)'}"
            )

        action.status = new_status
        for field_name, value in updates.items():
            setattr(action, field_name, value)

        with self._tx() as conn:
            conn.execute(
                "UPDATE actions SET status = ?, payload = ? WHERE id = ?",
                (new_status.value, action.to_json(), action_id),
            )
        return action

    def approve(self, action_id: str, note: str | None = None) -> ActionRecord:
        return self._transition(
            action_id, Status.APPROVED, reviewed_at=_utc_now(), review_note=note
        )

    def reject(self, action_id: str, note: str) -> ActionRecord:
        if not note.strip():
            raise QueueError(
                "a rejection requires a note; it is the extractor's only feedback"
            )
        action = self._transition(
            action_id, Status.REJECTED, reviewed_at=_utc_now(), review_note=note
        )
        self._cascade_similar(action, Status.REJECTED, reviewed_at=_utc_now(),
                              review_note=f"same to-do as {action.id[:8]}: {note}")
        return action

    def mark_executed(self, action_id: str, executor: str, detail: str) -> ActionRecord:
        return self._transition(
            action_id,
            Status.EXECUTED,
            executed_at=_utc_now(),
            executor=executor,
            execution_result=detail,
        )

    def mark_failed(self, action_id: str, executor: str, detail: str) -> ActionRecord:
        return self._transition(
            action_id,
            Status.FAILED,
            executed_at=_utc_now(),
            executor=executor,
            execution_result=detail,
        )

    def retry(self, action_id: str) -> ActionRecord:
        return self._transition(action_id, Status.APPROVED)

    def mark_done(self, action_id: str, note: str | None = None) -> ActionRecord:
        """Human checked the work off. No executor runs. Drops off the morning digest.

        Same-recording title clones (different quotes, same to-do) are checked off
        with it. Otherwise marking one Dave Casal item done would leave its twin
        on tomorrow's email.
        """
        action = self._transition(
            action_id, Status.DONE, reviewed_at=_utc_now(), review_note=note
        )
        self._cascade_similar(
            action, Status.DONE, reviewed_at=_utc_now(),
            review_note=note or f"same to-do as {action.id[:8]}",
        )
        return action

    def _cascade_similar(self, action: ActionRecord, new_status: Status, **updates: str | None) -> None:
        src = source_key(action)
        for other in self.list_outstanding():
            if other.id == action.id:
                continue
            if source_key(other) != src:
                continue
            if not titles_overlap(other.title, action.title):
                continue
            if new_status not in ALLOWED_TRANSITIONS[other.status]:
                continue
            self._transition(other.id, new_status, **updates)

    def list_outstanding(self) -> list[ActionRecord]:
        """PENDING and APPROVED: still on the operator's plate.

        REJECTED, DONE, EXECUTED and FAILED are not. The morning digest is this list.
        """
        found: list[ActionRecord] = []
        for status in (Status.PENDING, Status.APPROVED):
            found.extend(self.list(status=status))
        return found

    # ---- reads --------------------------------------------------------------

    def get(self, action_id: str) -> ActionRecord:
        row = self._conn.execute(
            "SELECT payload FROM actions WHERE id = ?", (action_id,)
        ).fetchone()
        if row is None:
            raise QueueError(f"no action with id {action_id!r}")
        return ActionRecord.from_json(row["payload"])

    def list(
        self,
        *,
        status: Status | None = None,
        target_system: str | None = None,
        min_confidence: float | None = None,
        limit: int | None = None,
    ) -> list[ActionRecord]:
        clauses: list[str] = []
        params: list[object] = []
        if status is not None:
            clauses.append("status = ?")
            params.append(status.value)
        if target_system is not None:
            clauses.append("target_system = ?")
            params.append(target_system)
        if min_confidence is not None:
            clauses.append("confidence >= ?")
            params.append(min_confidence)

        sql = "SELECT payload FROM actions"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY confidence DESC, created_at ASC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)

        rows = self._conn.execute(sql, params).fetchall()
        return [ActionRecord.from_json(r["payload"]) for r in rows]

    def counts(self) -> dict[str, int]:
        rows = self._conn.execute(
            "SELECT status, COUNT(*) AS n FROM actions GROUP BY status"
        ).fetchall()
        return {r["status"]: r["n"] for r in rows}
