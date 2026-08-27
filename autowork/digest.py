"""Morning outstanding-work digest. No model, no new extraction.

The call-summary email covers one conversation. This covers the queue: every
PENDING or APPROVED item that has not been checked off as done. REJECTED, DONE
and EXECUTED stay out. Grouped by recording date so yesterday's mapping work is
visibly yesterday's, not mixed into this morning's Okta call.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from datetime import datetime

from autowork.action import ActionRecord, Status
from autowork.extract import collapse_same_recording
from autowork.glossary import Glossary
from autowork.summarize import SummaryMeta, _action_origin

OUTSTANDING = (Status.PENDING, Status.APPROVED)
REVIEW_URL = "http://127.0.0.1:8765/"


def apply_glossary(action: ActionRecord, glossary: Glossary) -> ActionRecord:
    """Display copy with known variants rewritten. Does not change queue identity.

    Titles and quotes are frozen on first extract. A later glossary entry (Cazal ->
    Casal) would otherwise keep appearing in every digest until the row is deleted.
    """
    title, _ = glossary.correct(action.title)
    body, _ = glossary.correct(action.body)
    excerpt, _ = glossary.correct(action.provenance.transcript_excerpt)
    return replace(
        action,
        title=title,
        body=body,
        provenance=replace(action.provenance, transcript_excerpt=excerpt),
    )


def apply_glossary_all(actions: list[ActionRecord], glossary: Glossary) -> list[ActionRecord]:
    return [apply_glossary(action, glossary) for action in actions]


def group_by_recording_date(actions: list[ActionRecord]) -> list[tuple[str, list[ActionRecord]]]:
    buckets: dict[str, list[ActionRecord]] = defaultdict(list)
    for action in actions:
        stamp = SummaryMeta.from_filename(action.provenance.source_audio).recorded_at
        key = stamp.strftime("%Y-%m-%d") if stamp else "unknown date"
        buckets[key].append(action)
    return [(day, buckets[day]) for day in sorted(buckets)]


def digest_subject(actions: list[ActionRecord], when: datetime | None = None) -> str:
    actions = collapse_same_recording(actions)
    day = (when or datetime.now()).strftime("%Y-%m-%d")
    n = len(actions)
    pending = sum(1 for a in actions if a.status is Status.PENDING)
    return f"Outstanding actions {day} ({n}: {pending} unreviewed)"


def digest_text(actions: list[ActionRecord]) -> str:
    if not actions:
        return "Nothing outstanding.\n"
    actions = collapse_same_recording(actions)
    lines = [
        "OUTSTANDING ACTIONS",
        "Still on your plate: pending review, or approved and not yet done.",
        "Check off in the browser:  " + REVIEW_URL,
        "  (this PC only; start it with scripts\\review-ui.bat if it is not running)",
        "Or:  scripts\\review.bat --done <id>",
        "",
    ]
    for day, group in group_by_recording_date(actions):
        lines += ["=" * 68, day, "=" * 68, ""]
        for index, action in enumerate(group, start=1):
            lines.append(f"{index}. [{action.status.value}] {action.title}")
            lines.append(f"   from: {_action_origin(action)}")
            lines.append(f"   {action.body}")
            lines.append(
                f"   target: {action.target_system} / {action.action_type.value} "
                f"/ confidence {action.confidence:.2f} / id {action.id[:8]}"
            )
            lines.append(f'   said: "{action.provenance.transcript_excerpt}"')
            lines.append("")
    lines += [
        "None of these have been executed. Review at " + REVIEW_URL,
        "  (or scripts\\review.bat).",
        "",
    ]
    return "\n".join(lines)


def digest_html(actions: list[ActionRecord]) -> str:
    import html as _html

    def esc(value: str) -> str:
        return _html.escape(str(value))

    if not actions:
        return "<p>Nothing outstanding.</p>"
    actions = collapse_same_recording(actions)
    parts = [
        "<h2 style='margin:0 0 8px'>Outstanding actions</h2>",
        "<p style='color:#555;font-size:13px'>Still on your plate: pending review, "
        "or approved and not yet done. "
        f"<a href='{REVIEW_URL}'>Open the review queue</a> "
        f"(<code>{REVIEW_URL}</code> — this PC only). "
        "Or <code>scripts\\review.bat --done &lt;id&gt;</code>.</p>",
    ]
    for day, group in group_by_recording_date(actions):
        parts.append(f"<h3>{esc(day)}</h3><ol>")
        for action in group:
            parts.append(
                f"<li style='margin-bottom:10px'>"
                f"<b>{esc(action.title)}</b> "
                f"<span style='color:#777;font-size:12px'>[{esc(action.status.value)}] "
                f"id {esc(action.id[:8])}</span><br>"
                f"<span style='color:#777;font-size:12px'>"
                f"From: {esc(_action_origin(action))}</span><br>"
                f"{esc(action.body)}<br>"
                f"<span style='color:#777;font-size:12px'>"
                f"{esc(action.target_system)} / {esc(action.action_type.value)} / "
                f"confidence {action.confidence:.2f}</span><br>"
                f"<i style='color:#555'>said: “{esc(action.provenance.transcript_excerpt)}”</i>"
                f"</li>"
            )
        parts.append("</ol>")
    parts.append(
        "<p style='color:#777;font-size:12px'>None of these have been executed. "
        f"<a href='{REVIEW_URL}'>Review queue</a> "
        "(or <code>scripts\\review.bat</code>).</p>"
    )
    return "\n".join(parts)
