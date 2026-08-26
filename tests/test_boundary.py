#!/usr/bin/env python3
"""Tests for the queue/executor boundary.

Run:
    python -m pytest tests/ -v

Each area covers the happy path, one edge case, and the expected failure, per the
project testing standard. The safety-critical assertions are the ones proving an
unapproved action cannot reach an executor and that a re-run cannot duplicate work or
reset a review decision -- those are the properties the whole design exists to provide.
"""

from __future__ import annotations

import pytest

from autowork.action import (
    ActionRecord,
    ActionType,
    Provenance,
    Status,
    ValidationError,
)
from autowork.executors import (
    DryRunExecutor,
    ExecutionResult,
    Executor,
    ExecutorError,
    FileHandoffExecutor,
    Registry,
    dispatch,
)
from autowork.queue import QueueError, ReviewQueue


def make_provenance(**overrides: object) -> Provenance:
    defaults: dict[str, object] = {
        "source_audio": "RECORD/V2026-08-24-09-16-28.MP3",
        "start_sec": 2940.0,
        "end_sec": 3120.0,
        "speech_rumble_db": 1.6,
        "transcript_excerpt": "the Customer Intelligence Tool is supposed to match opportunities",
        "extractor": "phi4:latest/2026-08-25",
        "glossary_terms_applied": ("Claude", "Customer Intelligence Tool"),
    }
    defaults.update(overrides)
    return Provenance(**defaults)  # type: ignore[arg-type]


def make_action(**overrides: object) -> ActionRecord:
    defaults: dict[str, object] = {
        "title": "Fix opportunity matching in CIT",
        "body": "Claude is not reliably matching opportunities to existing entries.",
        "target_system": "backlog-tool",
        "action_type": ActionType.UPDATE,
        "confidence": 0.72,
        "provenance": make_provenance(),
    }
    defaults.update(overrides)
    return ActionRecord(**defaults)  # type: ignore[arg-type]


# --- ActionRecord ---------------------------------------------------------------


def test_action_survives_json_round_trip() -> None:
    """Happy path: the archive must be reconstructable from JSON alone."""
    original = make_action()
    restored = ActionRecord.from_json(original.to_json())

    assert restored.id == original.id
    assert restored.title == original.title
    assert restored.action_type is ActionType.UPDATE
    assert restored.status is Status.PENDING
    assert restored.provenance == original.provenance
    assert restored.provenance.glossary_terms_applied == ("Claude", "Customer Intelligence Tool")


def test_provenance_duration_is_derived_not_stored() -> None:
    """Edge: duration is computed, so it can never disagree with the window."""
    assert make_provenance(start_sec=10.0, end_sec=25.5).duration_sec == 15.5


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"confidence": 1.4}, "confidence"),
        ({"title": "   "}, "title"),
        ({"target_system": ""}, "target_system"),
        ({"schema_version": 99}, "schema_version"),
    ],
)
def test_action_rejects_invalid_fields(overrides: dict[str, object], expected: str) -> None:
    """Expected failure: bad data raises at construction, never reaches the queue."""
    with pytest.raises(ValidationError, match=expected):
        make_action(**overrides)


def test_provenance_rejects_inverted_window() -> None:
    """Expected failure: an empty window means the action is not auditable."""
    with pytest.raises(ValidationError, match="inverted"):
        make_provenance(start_sec=100.0, end_sec=100.0)


def test_action_rejects_unknown_fields() -> None:
    """Expected failure: a field this build does not understand is silent data loss."""
    raw = make_action().to_dict()
    raw["executed_by_robot"] = True
    with pytest.raises(ValidationError, match="unknown action fields"):
        ActionRecord.from_dict(raw)


# --- ReviewQueue ----------------------------------------------------------------


@pytest.fixture()
def queue(tmp_path) -> ReviewQueue:
    with ReviewQueue(tmp_path / "queue.sqlite3") as q:
        yield q


def test_enqueue_then_get(queue: ReviewQueue) -> None:
    """Happy path."""
    action = make_action()
    action_id = queue.enqueue(action)

    fetched = queue.get(action_id)
    assert fetched.title == action.title
    assert fetched.status is Status.PENDING
    assert queue.counts() == {"pending": 1}


def test_enqueue_is_idempotent_and_preserves_review(queue: ReviewQueue) -> None:
    """Edge: re-running a day must not duplicate, nor undo an approval."""
    first_id = queue.enqueue(make_action())
    queue.approve(first_id, note="looks right")

    # A second extraction pass over the same audio, with a differently-worded body
    # and a better confidence score, is still the same action.
    second_id = queue.enqueue(make_action(body="Reworded by a better model.", confidence=0.9))

    assert second_id == first_id
    assert len(queue.list()) == 1
    assert queue.get(first_id).status is Status.APPROVED
    assert queue.get(first_id).review_note == "looks right"


def test_list_filters_and_orders_by_confidence(queue: ReviewQueue) -> None:
    """Happy path: review highest-confidence first, filter out the noise."""
    queue.enqueue(make_action(
        title="low", confidence=0.2,
        provenance=make_provenance(transcript_excerpt="the first thing that was said about the mapping"),
    ))
    queue.enqueue(make_action(
        title="high", confidence=0.95,
        provenance=make_provenance(transcript_excerpt="a second and different commitment entirely"),
    ))
    queue.enqueue(make_action(
        title="other system", target_system="flare", confidence=0.8,
        provenance=make_provenance(transcript_excerpt="a third statement about the documentation"),
    ))

    assert [a.title for a in queue.list()] == ["high", "other system", "low"]
    assert [a.title for a in queue.list(target_system="flare")] == ["other system"]
    assert [a.title for a in queue.list(min_confidence=0.5)] == ["high", "other system"]


def test_rejection_requires_a_note(queue: ReviewQueue) -> None:
    """Expected failure: a rejection with no reason teaches the extractor nothing."""
    action_id = queue.enqueue(make_action())
    with pytest.raises(QueueError, match="requires a note"):
        queue.reject(action_id, note="  ")


def test_illegal_transition_raises(queue: ReviewQueue) -> None:
    """Expected failure: terminal means terminal."""
    action_id = queue.enqueue(make_action())
    queue.reject(action_id, note="hallucinated, no such conversation")

    with pytest.raises(QueueError, match="illegal transition"):
        queue.approve(action_id)


def test_failed_action_can_be_retried(queue: ReviewQueue) -> None:
    """Edge: a transient executor failure must not strand an approved action."""
    action_id = queue.enqueue(make_action())
    queue.approve(action_id)
    queue.mark_failed(action_id, executor="file_handoff", detail="disk full")

    assert queue.get(action_id).status is Status.FAILED
    assert queue.retry(action_id).status is Status.APPROVED


def test_get_unknown_id_raises(queue: ReviewQueue) -> None:
    """Expected failure."""
    with pytest.raises(QueueError, match="no action with id"):
        queue.get("does-not-exist")


# --- Executor boundary ----------------------------------------------------------


def test_dispatch_refuses_unapproved_action() -> None:
    """The safety property. An unreviewed action must never reach an adapter."""
    registry = Registry()
    registry.register("backlog-tool", DryRunExecutor())

    pending = make_action()
    assert pending.status is Status.PENDING

    with pytest.raises(ExecutorError, match="only 'approved' is dispatchable"):
        dispatch(pending, registry)


def test_dispatch_routes_to_registered_executor() -> None:
    """Happy path."""
    registry = Registry()
    registry.register("backlog-tool", DryRunExecutor())

    action = make_action(status=Status.APPROVED)
    result = dispatch(action, registry)

    assert result.ok
    assert "WOULD UPDATE in backlog-tool" in result.detail


def test_dry_run_warns_below_fabrication_threshold() -> None:
    """Edge: the reviewer must be told when the source audio cannot be trusted."""
    registry = Registry()
    registry.register("backlog-tool", DryRunExecutor())

    risky = make_action(
        status=Status.APPROVED,
        provenance=make_provenance(speech_rumble_db=-1.3),
    )
    result = dispatch(risky, registry)

    assert result.ok
    assert "fabrication threshold" in result.detail
    assert "-1.3 dB" in result.detail


def test_unroutable_target_raises() -> None:
    """Expected failure: an unknown system fails loudly, never silently no-ops."""
    registry = Registry()
    registry.register("backlog-tool", DryRunExecutor())

    orphan = make_action(status=Status.APPROVED, target_system="salesforce")
    with pytest.raises(ExecutorError, match="no executor registered"):
        dispatch(orphan, registry)


def test_registry_refuses_to_shadow_an_existing_route() -> None:
    """Expected failure: silently replacing a route is how writes go to the wrong place."""
    registry = Registry()
    registry.register("backlog-tool", DryRunExecutor())

    with pytest.raises(ExecutorError, match="refusing to shadow"):
        registry.register("backlog-tool", DryRunExecutor())


def test_adapter_exception_becomes_a_failure_result_not_a_crash() -> None:
    """Edge: an adapter blowing up must be recorded with context, and be retryable."""

    class ExplodingExecutor(Executor):
        @property
        def name(self) -> str:
            return "exploding"

        def can_handle(self, action: ActionRecord) -> bool:
            return True

        def execute(self, action: ActionRecord) -> ExecutionResult:
            raise ConnectionError("jira unreachable")

    registry = Registry()
    registry.register("jira", ExplodingExecutor())

    result = dispatch(make_action(status=Status.APPROVED, target_system="jira"), registry)

    assert not result.ok
    assert "exploding raised ConnectionError: jira unreachable" == result.detail


def test_file_handoff_writes_an_auditable_file(tmp_path) -> None:
    """Happy path for the executor that outlives access to every target system."""
    registry = Registry()
    registry.register("flare", FileHandoffExecutor(tmp_path / "handoff"))

    action = make_action(
        status=Status.APPROVED,
        target_system="flare",
        target_ref="Content/Topics/dispatch.htm",
        title="Clarify dispatch retry wording",
    )
    result = dispatch(action, registry)

    assert result.ok
    written = tmp_path / "handoff" / f"{action.id[:8]}-clarify-dispatch-retry-wording.md"
    assert written.exists()

    text = written.read_text(encoding="utf-8")
    assert "# Clarify dispatch retry wording" in text
    assert "Content/Topics/dispatch.htm" in text
    assert "+1.6 dB" in text
    assert "Customer Intelligence Tool" in text  # glossary terms recorded
    assert not list((tmp_path / "handoff").glob("*.tmp"))  # atomic write left no debris


def test_reworded_duplicates_collapse_by_grounding_quote(queue: ReviewQueue) -> None:
    """Regression from a real run: chunk overlap re-extracted one spoken commitment
    under three different titles, producing three queue entries. The grounding quote is
    the identity -- same quote, same action, whatever the model called it."""
    quote = "I think maybe there should be like a quality threshold assigned to each quote"
    first = queue.enqueue(make_action(
        title="Add quote quality threshold and top-ranked selection",
        provenance=make_provenance(transcript_excerpt=quote),
    ))
    second = queue.enqueue(make_action(
        title="Set a quality threshold for quotes",
        confidence=0.5,
        provenance=make_provenance(transcript_excerpt=quote),
    ))

    assert second == first
    assert len(queue.list()) == 1


def test_quote_identity_survives_punctuation_drift(queue: ReviewQueue) -> None:
    """Edge: the same quote reproduced with different punctuation is the same action."""
    first = queue.enqueue(make_action(
        provenance=make_provenance(transcript_excerpt="I'll take it up again, with Dave."),
    ))
    second = queue.enqueue(make_action(
        provenance=make_provenance(transcript_excerpt="I'll take it up again with Dave"),
    ))
    assert second == first
