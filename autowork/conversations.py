"""Group recordings into conversations without asking a model whether topics match.

Two files become one email only when a derived fact says they are the same meeting:
the same Outlook event (start + subject). Consecutive recordings, similar-sounding
transcripts, or 'both mentioned the backlog tool' are not enough -- that is how
yesterday's mapping call leaked into this morning's Okta summary.

No calendar match: each recording is its own conversation.
"""

from __future__ import annotations

from dataclasses import dataclass

from autowork.calendar_lookup import CalendarMatch
from autowork.day import Contribution, combine_text, meta_from_contribs
from autowork.summarize import SummaryMeta


@dataclass
class Conversation:
    contribs: list[Contribution]
    calendar: CalendarMatch | None = None
    group_key: str = ""

    @property
    def source_files(self) -> list[str]:
        return [c.source_file for c in self.contribs]

    @property
    def text(self) -> str:
        return combine_text(self.contribs)

    @property
    def meta(self) -> SummaryMeta:
        meta = meta_from_contribs(self.contribs)
        if self.calendar is not None:
            if self.calendar.subject:
                meta.meeting_title = self.calendar.subject
            if self.calendar.invitees:
                meta.invitees = list(self.calendar.invitees)
        return meta


def group_conversations(
    contribs: list[Contribution],
    matches: dict[str, CalendarMatch | None],
) -> list[Conversation]:
    """Preserve first-seen order. Calendar-keyed groups absorb later files."""
    buckets: dict[str, list[Contribution]] = {}
    calendars: dict[str, CalendarMatch | None] = {}
    order: list[str] = []
    for contrib in contribs:
        match = matches.get(contrib.source_file)
        if match is not None and match.subject.strip():
            key = f"cal:{match.event_key}"
        else:
            key = f"file:{contrib.source_file}"
        if key not in buckets:
            order.append(key)
            buckets[key] = []
            calendars[key] = match
        buckets[key].append(contrib)
    return [
        Conversation(
            contribs=buckets[key],
            calendar=calendars[key],
            group_key=key,
        )
        for key in order
    ]
