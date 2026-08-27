#!/usr/bin/env python3
"""Calendar matching: one overlapping event is a fact; two is a guess we refuse."""

from __future__ import annotations

from datetime import datetime, timedelta

from autowork.calendar_lookup import CalendarEvent, CalendarMatch, match_recording
from autowork.summarize import Summary, SummaryMeta


class FakeCalendar:
    def __init__(self, events: list[CalendarEvent]) -> None:
        self.events = events

    def events_overlapping(self, start: datetime, end: datetime) -> list[CalendarEvent]:
        return [e for e in self.events if e.start < end and e.end > start]


def test_single_overlapping_event_is_used() -> None:
    start = datetime(2026, 8, 26, 9, 4, 45)
    event = CalendarEvent(
        subject="Okta access review",
        start=datetime(2026, 8, 26, 9, 0),
        end=datetime(2026, 8, 26, 9, 30),
        attendees=("Rob Dods", "Andrew Smith", "Rob Dods"),
    )
    match = match_recording(start, 3.5, source=FakeCalendar([event]))
    assert match == CalendarMatch(
        subject="Okta access review",
        invitees=("Rob Dods", "Andrew Smith"),
        start=datetime(2026, 8, 26, 9, 0),
    )


def test_no_overlapping_event_omits() -> None:
    start = datetime(2026, 8, 26, 9, 4, 45)
    event = CalendarEvent(
        subject="Unrelated standup",
        start=datetime(2026, 8, 26, 14, 0),
        end=datetime(2026, 8, 26, 14, 30),
        attendees=("Rob Dods",),
    )
    assert match_recording(start, 3.5, source=FakeCalendar([event])) is None


def test_two_overlapping_events_are_not_guessed() -> None:
    """A wrong meeting on a work summary is worse than none."""
    start = datetime(2026, 8, 26, 9, 4, 45)
    events = [
        CalendarEvent("A", start, start + timedelta(minutes=30), ("Ann",)),
        CalendarEvent("B", start, start + timedelta(minutes=30), ("Bob",)),
    ]
    assert match_recording(start, 10, source=FakeCalendar(events)) is None


def test_invitees_render_as_calendar_not_spoken_names() -> None:
    meta = SummaryMeta(
        recorded_at=datetime(2026, 8, 26, 9, 4, 45),
        source_files=["R2026-08-26-09-04-45.MP3"],
        audio_minutes=3.5,
        speaker_count=2,
        meeting_title="Okta access review",
        invitees=["Rob Dods", "Andrew Smith"],
    )
    rendered = "\n".join(
        Summary(
            headline="h", topics=[], decisions=[], open_questions=[], meta=meta,
        ).meta_lines()
    )
    assert "Meeting       Okta access review" in rendered
    assert "Invitees      Rob Dods, Andrew Smith (from calendar)" in rendered
    assert "Came up" not in rendered
    assert "Participants  2 (distinct voices heard)" in rendered


def test_clarify_terms_render_in_text_and_markdown() -> None:
    summary = Summary(
        headline="h",
        topics=[],
        decisions=[],
        open_questions=[],
        clarify_terms=["Octo", "Cotera"],
    )
    text = summary.to_text()
    markdown = summary.to_markdown()
    assert "TERMS TO CLARIFY" in text
    assert "Octo" in text
    assert "## Terms to clarify" in markdown
    assert "- Cotera" in markdown
