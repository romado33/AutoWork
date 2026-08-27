#!/usr/bin/env python3
"""Conversation grouping: same Outlook event only, never a guessed topic."""

from __future__ import annotations

from datetime import datetime

from autowork.calendar_lookup import CalendarMatch
from autowork.conversations import group_conversations
from autowork.day import Contribution, isoformat


def _c(name: str, text: str) -> Contribution:
    stamp = datetime.strptime(name[1:20], "%Y-%m-%d-%H-%M-%S")
    return Contribution(
        source_file=name,
        keep=True,
        text=text,
        audio_minutes=15.0,
        speaker_count=2,
        recorded_at=isoformat(stamp),
    )


def test_no_calendar_means_one_email_per_recording() -> None:
    a = _c("R2026-08-26-09-04-45.MP3", "Okta access.")
    b = _c("R2026-08-26-13-00-00.MP3", "Also Okta, later.")
    groups = group_conversations([a, b], {a.source_file: None, b.source_file: None})
    assert len(groups) == 2
    assert groups[0].source_files == [a.source_file]
    assert groups[1].source_files == [b.source_file]


def test_same_outlook_event_groups_two_recordings() -> None:
    """The only 'same topic' signal: the same calendar event, derived."""
    a = _c("R2026-08-26-09-00-00.MP3", "First half of standup.")
    b = _c("R2026-08-26-09-20-00.MP3", "Second half of standup.")
    meeting = CalendarMatch(
        subject="Standup",
        start=datetime(2026, 8, 26, 9, 0),
        invitees=("Rob", "Andrew"),
    )
    groups = group_conversations(
        [a, b], {a.source_file: meeting, b.source_file: meeting}
    )
    assert len(groups) == 1
    assert groups[0].source_files == [a.source_file, b.source_file]
    assert groups[0].meta.meeting_title == "Standup"


def test_different_meetings_stay_separate_even_if_topics_sound_alike() -> None:
    """Yesterday's mapping call and today's Okta call both mentioned backlog.
    That must not become one email.
    """
    mapping = _c("R2026-08-25-13-23-54.MP3", "Entity mapping and the backlog tool.")
    okta = _c("R2026-08-26-09-04-45.MP3", "Okta and the backlog tool permissions.")
    mon = CalendarMatch("Mapping eval", start=datetime(2026, 8, 25, 13, 0))
    tue = CalendarMatch("Okta access", start=datetime(2026, 8, 26, 9, 0))
    groups = group_conversations(
        [mapping, okta],
        {mapping.source_file: mon, okta.source_file: tue},
    )
    assert len(groups) == 2
    assert groups[0].source_files == [mapping.source_file]
    assert groups[1].source_files == [okta.source_file]
