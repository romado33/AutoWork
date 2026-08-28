#!/usr/bin/env python3
"""Tests for the review CLI's non-interactive paths.

Run:
    python -m pytest tests/ -v

The interactive loop is not tested here (it needs a terminal); everything it relies on
is. The important cases are the ones where a mistake approves the WRONG action, since
approval is what makes an action eligible to be executed against a live system.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from autowork.action import ActionRecord, ActionType, Provenance, Status  # noqa: E402
from autowork.queue import QueueError, ReviewQueue  # noqa: E402
from review import main, resolve, summarise, wall_clock  # noqa: E402


def record(**overrides) -> ActionRecord:
    provenance_overrides = overrides.pop("provenance_overrides", {})
    provenance = dict(
        source_audio="D:/RECORD/R2026-08-25-13-23-54.MP3",
        start_sec=2100.0,
        end_sec=2220.0,
        speech_rumble_db=6.6,
        transcript_excerpt="I should take it up again with Dave Casale and get it finalized",
        extractor="gemma3:4b/grounded",
    )
    provenance.update(provenance_overrides)
    defaults = dict(
        title="Finalise the backlog tool with Dave Casale",
        body="Take it up again with Dave Casale.",
        target_system="backlog-tool",
        action_type=ActionType.UPDATE,
        confidence=0.82,
        provenance=Provenance(**provenance),  # type: ignore[arg-type]
    )
    defaults.update(overrides)
    return ActionRecord(**defaults)  # type: ignore[arg-type]


@pytest.fixture()
def queue(tmp_path):
    with ReviewQueue(tmp_path / "q.sqlite3") as q:
        yield q


# --- id resolution: getting this wrong approves the wrong action -----------------


def test_resolves_a_short_prefix(queue) -> None:
    """Happy path: --list prints 8 characters, so 8 characters must work."""
    action_id = queue.enqueue(record())
    assert resolve(queue, action_id[:8]) == action_id


def test_resolves_a_full_id(queue) -> None:
    action_id = queue.enqueue(record())
    assert resolve(queue, action_id) == action_id


def test_ambiguous_prefix_refuses_rather_than_guessing(queue) -> None:
    """Expected failure. Picking one would approve an action the user did not choose."""
    first = record(id="aaaa1111-0000-0000-0000-000000000001")
    second = record(
        id="aaaa1111-0000-0000-0000-000000000002", title="Something else",
        provenance_overrides={"transcript_excerpt": "an entirely different sentence was spoken here"},
    )
    queue.enqueue(first)
    queue.enqueue(second)

    with pytest.raises(QueueError, match="ambiguous"):
        resolve(queue, "aaaa1111")


def test_empty_id_is_refused(queue) -> None:
    """Expected failure: an empty string must not silently match or change mode."""
    queue.enqueue(record())
    with pytest.raises(QueueError, match="empty action id"):
        resolve(queue, "   ")


def test_unknown_prefix_raises(queue) -> None:
    """Expected failure."""
    with pytest.raises(QueueError, match="no action matching"):
        resolve(queue, "deadbeef")


# --- provenance display ----------------------------------------------------------


def test_wall_clock_is_derived_from_filename_plus_offset() -> None:
    """Happy path. With VOR off the recorder runs continuously, so filename start time
    plus offset is genuine wall-clock time.

    Filename says 13:23:54; the segment starts 2100s (35 minutes) in.
    """
    assert wall_clock(record()) == "2026-08-25 13:58"


def test_wall_clock_falls_back_rather_than_guessing() -> None:
    """Edge: an unparseable filename must not produce a plausible wrong time, because
    it would be correlated against a calendar."""
    shown = wall_clock(
        record(provenance_overrides={"source_audio": "some-other-recording.mp3"})
    )
    assert shown == "+35m"


def test_low_signal_actions_are_flagged_in_the_listing() -> None:
    """The listing must make unverifiable actions visible without opening each one."""
    risky = summarise(record(provenance_overrides={"speech_rumble_db": -0.9}))
    fine = summarise(record())

    assert "[UNVERIFIED AUDIO]" in risky
    assert "[UNVERIFIED AUDIO]" not in fine


def test_status_marks_distinguish_states(queue) -> None:
    """Edge: the listing is scanned, not read, so state must be visible at a glance."""
    action_id = queue.enqueue(record())
    assert summarise(queue.get(action_id)).startswith("?")
    assert summarise(queue.approve(action_id)).startswith("+")


# --- CLI entry points ------------------------------------------------------------


def test_list_exits_clean(tmp_path, capsys) -> None:
    db = tmp_path / "q.sqlite3"
    with ReviewQueue(db) as q:
        q.enqueue(record())

    assert main(["--queue", str(db), "--list"]) == 0
    assert "Finalise the backlog tool" in capsys.readouterr().out


def test_reject_without_a_note_is_refused(tmp_path) -> None:
    """Expected failure: the note is the extractor's only feedback signal."""
    db = tmp_path / "q.sqlite3"
    with ReviewQueue(db) as q:
        action_id = q.enqueue(record())

    assert main(["--queue", str(db), "--reject", action_id[:8]]) == 2

    with ReviewQueue(db) as q:
        assert q.get(action_id).status is Status.PENDING


def test_approve_marks_approved_but_does_not_execute(tmp_path, capsys) -> None:
    """The safety property of this tool: approving is a judgement, not an effect."""
    db = tmp_path / "q.sqlite3"
    with ReviewQueue(db) as q:
        action_id = q.enqueue(record())

    assert main(["--queue", str(db), "--approve", action_id[:8], "--note", "ok"]) == 0

    out = capsys.readouterr().out
    assert "Not executed" in out
    with ReviewQueue(db) as q:
        stored = q.get(action_id)
        assert stored.status is Status.APPROVED
        assert stored.review_note == "ok"
        assert stored.executed_at is None
        assert stored.executor is None


def test_done_checks_off_without_executing(tmp_path, capsys) -> None:
    """Morning-digest check-off: done is not an executor write."""
    db = tmp_path / "q.sqlite3"
    with ReviewQueue(db) as q:
        action_id = q.enqueue(record())

    assert main(["--queue", str(db), "--done", action_id[:8]]) == 0
    assert "No executor ran" in capsys.readouterr().out
    with ReviewQueue(db) as q:
        stored = q.get(action_id)
        assert stored.status is Status.DONE
        assert stored.executed_at is None
        assert stored.executor is None


def test_missing_queue_database_is_a_clear_error(tmp_path, capsys) -> None:
    """Expected failure: must not create an empty queue and report 'nothing pending'."""
    assert main(["--queue", str(tmp_path / "absent.sqlite3"), "--list"]) == 2
    assert "no queue database" in capsys.readouterr().err


def test_reject_source_via_cli_does_not_execute(tmp_path, capsys) -> None:
    """Operator move after 11:30: one command, whole recording, no executor."""
    db = tmp_path / "q.sqlite3"
    garbled = "R2026-08-27-11-30-55.MP3"
    with ReviewQueue(db) as q:
        first = q.enqueue(record(
            title="Separate two-page holes",
            provenance_overrides={
                "source_audio": garbled,
                "transcript_excerpt": "Separate two-page holes and then one-page duty.",
            },
        ))
        second = q.enqueue(record(
            title="Account for switching software",
            provenance_overrides={
                "source_audio": garbled,
                "transcript_excerpt": (
                    "Yeah, we'll definitely have the right you have to "
                    "account for switching software."
                ),
            },
        ))
        other = q.enqueue(record(
            title="Check Okta with Andrew",
            provenance_overrides={
                "source_audio": "R2026-08-26-09-04-45.MP3",
                "transcript_excerpt": "I'll check that it has the correct permissions anyway.",
            },
        ))

    assert main([
        "--queue", str(db),
        "--reject-source", garbled,
        "--note", "garbled restaurant audio, not a work conversation",
    ]) == 0
    out = capsys.readouterr().out
    assert "rejected 2" in out
    assert "Not executed" in out or "no executor" in out.lower()

    with ReviewQueue(db) as q:
        assert q.get(first).status is Status.REJECTED
        assert q.get(second).status is Status.REJECTED
        assert q.get(other).status is Status.PENDING
        assert q.get(first).executor is None


def test_reject_source_without_a_note_is_refused(tmp_path) -> None:
    db = tmp_path / "q.sqlite3"
    with ReviewQueue(db) as q:
        q.enqueue(record())

    assert main([
        "--queue", str(db),
        "--reject-source", "R2026-08-25-13-23-54.MP3",
    ]) == 2
    with ReviewQueue(db) as q:
        assert q.list_outstanding()


def test_illegal_transition_reports_and_exits_nonzero(tmp_path, capsys) -> None:
    """Expected failure: re-rejecting a terminal action."""
    db = tmp_path / "q.sqlite3"
    with ReviewQueue(db) as q:
        action_id = q.enqueue(record())
        q.reject(action_id, note="not real")

    assert main(["--queue", str(db), "--reject", action_id[:8], "--note", "again"]) == 1
    assert "illegal transition" in capsys.readouterr().err
