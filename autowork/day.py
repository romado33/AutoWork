"""Per-recording contribution to a day's summary.

The email is a DAILY summary, not a summary of whichever files happened to arrive
in this USB plug-in. A second plug-in the same afternoon used to overwrite
summaries/YYYY-MM-DD.md with only the new recording. Each recording now writes a
sidecar; a later run rebuilds that date from every sidecar that passed the
relevance gate.

Sidecars live next to the transcript (gitignored with it). They are derived
facts plus the kept spoken text -- never model output -- so a rebuild cannot
invent a date or silently re-admit a dropped podcast.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from autowork.summarize import SummaryMeta

CONTRIB_SUFFIX = ".contrib.json"


class DayError(RuntimeError):
    """Sidecar read/write failed. Raised rather than pretending the day is empty."""


@dataclass
class Contribution:
    """What one recording contributes to its calendar date's summary."""

    source_file: str
    keep: bool
    text: str = ""
    audio_minutes: float = 0.0
    speaker_count: int = 0
    recorded_at: str | None = None
    excluded_because: str = ""
    # Kept by operator override (--force-keep), not by passing the gates. Excluded
    # from day rebuilds so a one-time exception stays one time: the 2026-08-27
    # personal recordings sat here as keep=True and would have been merged into a
    # later work summary by whichever recording shared their Outlook event.
    forced: bool = False

    @property
    def recorded_datetime(self) -> datetime | None:
        if not self.recorded_at:
            return None
        try:
            return datetime.fromisoformat(self.recorded_at)
        except ValueError:
            return None

    @property
    def date_key(self) -> str | None:
        stamp = self.recorded_datetime
        if stamp is None:
            return date_from_filename(self.source_file)
        return stamp.strftime("%Y-%m-%d")


def date_from_filename(name: str) -> str | None:
    """Recording date from the recorder's R/V timestamp, or None if unparseable."""
    stamp = SummaryMeta.from_filename(name).recorded_at
    if stamp is None:
        return None
    return stamp.strftime("%Y-%m-%d")


def isoformat(stamp: datetime | None) -> str | None:
    if stamp is None:
        return None
    return stamp.isoformat(timespec="seconds")


def contrib_path(transcript_dir: Path, source_file: str) -> Path:
    return transcript_dir / f"{Path(source_file).stem}{CONTRIB_SUFFIX}"


def save_contribution(transcript_dir: Path, contrib: Contribution) -> Path:
    """Atomic write so a crash cannot leave a half-JSON sidecar the next run would trust."""
    transcript_dir.mkdir(parents=True, exist_ok=True)
    path = contrib_path(transcript_dir, contrib.source_file)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(asdict(contrib), indent=2), encoding="utf-8")
    tmp.replace(path)
    return path


def load_contribution(path: Path) -> Contribution:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DayError(f"cannot read {path.name}: {exc}") from exc
    if not isinstance(raw, dict) or not raw.get("source_file"):
        raise DayError(f"{path.name} is not a contribution sidecar")
    return Contribution(
        source_file=str(raw["source_file"]),
        keep=bool(raw.get("keep")),
        text=str(raw.get("text") or ""),
        audio_minutes=float(raw.get("audio_minutes") or 0.0),
        speaker_count=int(raw.get("speaker_count") or 0),
        recorded_at=raw.get("recorded_at") or None,
        excluded_because=str(raw.get("excluded_because") or ""),
        # Absent in sidecars written before the override existed: an old keep is
        # an ordinary keep.
        forced=bool(raw.get("forced")),
    )


def load_kept_for_date(
    transcript_dir: Path, date_key: str, include_forced: bool = False
) -> list[Contribution]:
    """Every kept contribution for this calendar date, in filename order.

    Missing or unreadable sidecars are skipped with no contribution -- a corrupt
    file must not abort the rest of the day, and must not be treated as 'keep'.

    Operator-forced contributions are excluded unless asked for. Only the run that
    passed --force-keep may see them; every later run must rebuild the day from
    recordings that actually passed the gates.
    """
    if not transcript_dir.is_dir():
        return []
    kept: list[Contribution] = []
    for path in sorted(transcript_dir.glob(f"*{CONTRIB_SUFFIX}")):
        try:
            contrib = load_contribution(path)
        except DayError:
            continue
        if contrib.forced and not include_forced:
            continue
        if contrib.keep and contrib.date_key == date_key and contrib.text.strip():
            kept.append(contrib)
    return kept


def combine_text(contribs: list[Contribution]) -> str:
    return "\n\n".join(c.text.strip() for c in contribs if c.text.strip())


def meta_from_contribs(contribs: list[Contribution]) -> SummaryMeta:
    """Derived header for a rebuilt day. Speaker count is the max in any one
    recording, never a union -- diarization labels are per request.
    """
    stamps = [c.recorded_datetime for c in contribs if c.recorded_datetime]
    return SummaryMeta(
        recorded_at=min(stamps) if stamps else None,
        source_files=[c.source_file for c in contribs],
        audio_minutes=sum(c.audio_minutes for c in contribs),
        speaker_count=max((c.speaker_count for c in contribs), default=0),
    )


def clarify_markdown(terms: list[str]) -> str:
    if not terms:
        return ""
    lines = [
        "---",
        "",
        "## Terms to clarify",
        "",
        "Capitalised words in this transcript that the glossary does not know.",
        "Heuristic candidates -- skim and promote real names into `config/glossary.yml`.",
        "",
        *(f"- {word}" for word in terms),
        "",
    ]
    return "\n".join(lines)
