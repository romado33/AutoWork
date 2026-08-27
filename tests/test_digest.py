#!/usr/bin/env python3
"""Morning digest: outstanding queue only, no model, no mixing into a call summary."""

from __future__ import annotations

from datetime import datetime

from autowork.action import ActionRecord, ActionType, Provenance, Status
from autowork.digest import apply_glossary, digest_subject, digest_text
from autowork.glossary import Glossary, Term
from autowork.queue import ReviewQueue


def _action(title: str, source: str, excerpt: str, status: Status = Status.PENDING) -> ActionRecord:
    return ActionRecord(
        title=title,
        body=title,
        target_system="backlog-tool",
        action_type=ActionType.TASK,
        confidence=0.9,
        status=status,
        provenance=Provenance(
            source_audio=source,
            start_sec=10.0, end_sec=20.0, speech_rumble_db=6.6,
            transcript_excerpt=excerpt,
            extractor="test",
        ),
    )


def test_digest_groups_by_recording_date_and_omits_done() -> None:
    actions = [
        _action("Okta permissions", "R2026-08-26-09-04-45.MP3", "I'll check permissions"),
        _action(
            "Finalize backlog tool",
            "R2026-08-25-13-23-54.MP3",
            "I should take it up again with Dave",
        ),
        _action(
            "Old item already done",
            "R2026-08-24-09-00-00.MP3",
            "this was finished",
            status=Status.DONE,
        ),
    ]
    # Digest renderer is given the outstanding list; DONE would not be in it.
    outstanding = [a for a in actions if a.status in (Status.PENDING, Status.APPROVED)]
    text = digest_text(outstanding)
    assert "2026-08-25" in text
    assert "2026-08-26" in text
    assert "Finalize backlog tool" in text
    assert "Okta permissions" in text
    assert "Old item already done" not in text
    assert "TO DO" not in text  # that heading is for call summaries


def test_digest_subject_counts_unreviewed() -> None:
    actions = [
        _action("a", "R2026-08-26-09-04-45.MP3", "first sentence here"),
        _action(
            "b", "R2026-08-25-13-23-54.MP3", "second different sentence",
            status=Status.APPROVED,
        ),
    ]
    subject = digest_subject(actions, when=datetime(2026, 8, 27, 7, 30))
    assert subject == "Outstanding actions 2026-08-27 (2: 1 unreviewed)"


def test_list_outstanding_skips_done_and_rejected(tmp_path) -> None:
    with ReviewQueue(tmp_path / "q.sqlite3") as queue:
        pending_id = queue.enqueue(_action(
            "still open", "R2026-08-26-09-04-45.MP3", "I'll check that"
        ))
        done_id = queue.enqueue(_action(
            "finished", "R2026-08-25-13-23-54.MP3", "a different spoken commitment"
        ))
        queue.mark_done(done_id)
        rejected_id = queue.enqueue(_action(
            "false", "R2026-08-24-09-00-00.MP3", "yet another distinct quote here"
        ))
        queue.reject(rejected_id, note="not a real commitment")
        outstanding = queue.list_outstanding()
        assert [a.id for a in outstanding] == [pending_id]
        assert outstanding[0].id == pending_id


def test_digest_rewrites_known_glossary_variants() -> None:
    """Queue titles are frozen on extract; the mail must still show the canonical name."""
    glossary = Glossary(terms=[
        Term(term="Dave Casal", tier="correct", variants=("Dave Cazal", "Cazal")),
    ])
    raw = _action(
        "Finalize backlog tool design with Dave Cazal",
        "R2026-08-25-13-23-54.MP3",
        "I should take it up again with Dave Cazal maybe",
    )
    shown = apply_glossary(raw, glossary)
    assert shown.title == "Finalize backlog tool design with Dave Casal"
    assert "Dave Casal" in shown.body
    assert "Dave Casal" in shown.provenance.transcript_excerpt
    assert "Cazal" not in shown.title
    assert raw.title.endswith("Cazal")  # stored record is untouched
    text = digest_text([shown])
    assert "Dave Casal" in text
    assert "Cazal" not in text


def test_digest_collapses_same_recording_title_clones() -> None:
    """The 25 Aug digest listed two Dave Casal backlog items for one commitment."""
    actions = [
        _action(
            "Finalize backlog tool design with Dave Casal",
            "R2026-08-25-13-23-54.MP3",
            "I should take it up again with Dave Casal maybe and just get it finalized",
        ),
        _action(
            "Finalize backlog tool with Dave Casal",
            "R2026-08-25-13-23-54.MP3",
            "But I mean it is working. It doesn't happen like that a lot making sure",
        ),
        _action(
            "Add quote quality threshold",
            "R2026-08-25-13-23-54.MP3",
            "there should be like a quality threshold assigned to each quote",
        ),
    ]
    text = digest_text(actions)
    numbered = [
        line for line in text.splitlines()
        if line[:3].endswith(". ") and line[0].isdigit()
    ]
    casal_items = [line for line in numbered if "Dave Casal" in line]
    assert len(casal_items) == 1
    assert "Finalize backlog tool design" in casal_items[0]
    assert any("Add quote quality threshold" in line for line in numbered)


def test_digest_includes_localhost_review_link() -> None:
    """The morning mail must be one click to the review UI on this PC."""
    from autowork.digest import REVIEW_URL, digest_html

    text = digest_text([_action(
        "Check permissions", "R2026-08-26-09-04-45.MP3", "I'll check that",
    )])
    html = digest_html([_action(
        "Check permissions", "R2026-08-26-09-04-45.MP3", "I'll check something else",
    )])
    assert REVIEW_URL in text
    assert f"href='{REVIEW_URL}'" in html or f'href="{REVIEW_URL}"' in html
    assert "Open the review queue" in html
