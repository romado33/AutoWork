"""Derived meeting title and invitees from the local Outlook calendar.

Names on a summary must not be guessed from the transcript: a two-person call that
discussed four absent colleagues was once rendered as an attendee list. Calendar
data is a FACT about the invite, the same way the filename is a fact about the
date. It is labelled 'from calendar' and is NEVER mapped onto diarization labels
A/B -- those labels are per request and do not identify people.

No Microsoft Graph connector is configured in this project. Outlook on this
Windows machine is the source, via the desktop COM API, and only when it answers.
Zero overlapping events, two or more overlapping events, a missing Outlook, or
any COM error: omit the fields. A wrong meeting on a work summary is worse than
none.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Protocol

logger = logging.getLogger(__name__)


class CalendarError(RuntimeError):
    """Lookup failed. Callers catch this and omit calendar fields."""


@dataclass(frozen=True)
class CalendarEvent:
    subject: str
    start: datetime
    end: datetime
    attendees: tuple[str, ...] = ()


@dataclass(frozen=True)
class CalendarMatch:
    """At most one overlapping event. Ambiguous overlaps are not a match."""

    subject: str
    invitees: tuple[str, ...] = field(default_factory=tuple)
    start: datetime | None = None

    @property
    def event_key(self) -> str:
        """Identity for grouping recordings of the same meeting. Start plus
        subject, not the spoken topic: two calls that 'feel similar' must not
        merge because a model said so.
        """
        when = self.start.isoformat(timespec="minutes") if self.start else ""
        return f"{when}|{self.subject.strip().lower()}"


class EventSource(Protocol):
    def events_overlapping(self, start: datetime, end: datetime) -> list[CalendarEvent]:
        ...


def match_recording(
    start: datetime,
    duration_min: float,
    source: EventSource | None = None,
) -> CalendarMatch | None:
    """Return the single calendar event overlapping this recording window.

    duration_min is wall-clock coverage (file length or kept speech), used only
    to bound the window. Zero or negative duration still looks up a one-minute
    instant so a timestamp with no duration can match.
    """
    if duration_min < 0:
        duration_min = 0.0
    end = start + timedelta(minutes=max(duration_min, 1.0))
    backend = source if source is not None else OutlookSource()
    try:
        events = backend.events_overlapping(start, end)
    except CalendarError as exc:
        logger.info("calendar lookup skipped: %s", exc)
        return None
    except Exception as exc:  # COM, import, locale -- none of these may fail the day
        logger.info("calendar lookup skipped: %s", exc)
        return None

    overlapping = [
        ev for ev in events
        if ev.start < end and ev.end > start
    ]
    if len(overlapping) != 1:
        if len(overlapping) > 1:
            logger.info(
                "calendar omitted: %d overlapping events, not guessing",
                len(overlapping),
            )
        return None
    event = overlapping[0]
    invitees = tuple(dict.fromkeys(n.strip() for n in event.attendees if n.strip()))
    return CalendarMatch(
        subject=event.subject.strip(),
        invitees=invitees,
        start=event.start,
    )


class OutlookSource:
    """Default calendar of the logged-in Outlook profile. Optional: no pywin32,
    no Outlook, or a COM prompt the user dismisses all become 'omit'.
    """

    def events_overlapping(self, start: datetime, end: datetime) -> list[CalendarEvent]:
        try:
            import win32com.client  # type: ignore
        except ImportError as exc:
            raise CalendarError("pywin32 is not installed") from exc

        try:
            outlook = win32com.client.Dispatch("Outlook.Application")
            namespace = outlook.GetNamespace("MAPI")
            # 9 = olFolderCalendar
            folder = namespace.GetDefaultFolder(9)
            items = folder.Items
            items.IncludeRecurrences = True
            items.Sort("[Start]")
            # Outlook's Restrict filter is locale-sensitive. Walking a padded day
            # and filtering in Python is slower and does not invent a meeting.
            day = start.replace(hour=0, minute=0, second=0, microsecond=0)
            day_end = day + timedelta(days=1)
            restriction = (
                f"[Start] < '{_outlook_dt(day_end)}' AND "
                f"[End] > '{_outlook_dt(day)}'"
            )
            try:
                restricted = items.Restrict(restriction)
            except Exception:
                restricted = items

            found: list[CalendarEvent] = []
            for item in restricted:
                try:
                    ev_start = _as_datetime(item.Start)
                    ev_end = _as_datetime(item.End)
                except Exception:
                    continue
                if ev_start is None or ev_end is None:
                    continue
                if not (ev_start < end and ev_end > start):
                    continue
                found.append(
                    CalendarEvent(
                        subject=str(getattr(item, "Subject", "") or ""),
                        start=ev_start,
                        end=ev_end,
                        attendees=_attendee_names(item),
                    )
                )
            return found
        except CalendarError:
            raise
        except Exception as exc:
            raise CalendarError(str(exc)) from exc


def _outlook_dt(stamp: datetime) -> str:
    """US-style timestamp Outlook Restrict usually accepts on this machine."""
    return stamp.strftime("%m/%d/%Y %H:%M")


def _as_datetime(value: object) -> datetime | None:
    if isinstance(value, datetime):
        return value.replace(tzinfo=None)
    return None


def _attendee_names(item: object) -> tuple[str, ...]:
    names: list[str] = []
    recipients = getattr(item, "Recipients", None)
    if recipients is None:
        organizer = str(getattr(item, "Organizer", "") or "").strip()
        return (organizer,) if organizer else ()
    try:
        count = int(recipients.Count)
    except Exception:
        return ()
    for index in range(1, count + 1):
        try:
            name = str(recipients.Item(index).Name or "").strip()
        except Exception:
            continue
        if name:
            names.append(name)
    return tuple(dict.fromkeys(names))
