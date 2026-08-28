#!/usr/bin/env python3
"""Day-scoped summary sidecars: a second plug-in must not overwrite the morning."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from autowork.day import (
    Contribution,
    combine_text,
    contrib_path,
    date_from_filename,
    isoformat,
    load_contribution,
    load_kept_for_date,
    meta_from_contribs,
    save_contribution,
)
from extract_actions import parse_transcript  # noqa: E402


def contrib(
    name: str, text: str, keep: bool = True, speakers: int = 2, minutes: float = 10.0
) -> Contribution:
    stamp = datetime.strptime(name[1:20], "%Y-%m-%d-%H-%M-%S")
    return Contribution(
        source_file=name,
        keep=keep,
        text=text,
        audio_minutes=minutes,
        speaker_count=speakers,
        recorded_at=isoformat(stamp),
        excluded_because="" if keep else "ambient",
    )


def test_date_from_filename() -> None:
    assert date_from_filename("R2026-08-26-09-04-45.MP3") == "2026-08-26"
    assert date_from_filename("V2026-08-24-09-16-28.MP3") == "2026-08-24"
    assert date_from_filename("notes.mp3") is None


def test_second_recording_same_day_is_unioned(tmp_path: Path) -> None:
    """The failure this exists to stop: afternoon plug-in wiping the morning summary."""
    morning = contrib("R2026-08-26-09-04-45.MP3", "Morning: Okta access review with Andrew.")
    afternoon = contrib("R2026-08-26-16-10-00.MP3", "Afternoon: backlog tool follow-up.")
    save_contribution(tmp_path, morning)
    save_contribution(tmp_path, afternoon)

    kept = load_kept_for_date(tmp_path, "2026-08-26")
    assert [c.source_file for c in kept] == [
        "R2026-08-26-09-04-45.MP3",
        "R2026-08-26-16-10-00.MP3",
    ]
    combined = combine_text(kept)
    assert "Okta" in combined and "backlog" in combined

    meta = meta_from_contribs(kept)
    assert meta.date_only == "2026-08-26"
    assert meta.recorded_at == datetime(2026, 8, 26, 9, 4, 45)
    assert meta.audio_minutes == 20.0
    assert meta.speaker_count == 2


def test_excluded_recording_does_not_reenter_the_day(tmp_path: Path) -> None:
    meeting = contrib("R2026-08-25-13-23-54.MP3", "The mapping evaluation.")
    ambient = contrib(
        "R2026-08-25-10-52-46.MP3", "Beep beep. There it is man.", keep=False
    )
    save_contribution(tmp_path, meeting)
    save_contribution(tmp_path, ambient)

    kept = load_kept_for_date(tmp_path, "2026-08-25")
    assert [c.source_file for c in kept] == ["R2026-08-25-13-23-54.MP3"]
    assert "Beep beep" not in combine_text(kept)


def test_speaker_count_is_max_not_sum(tmp_path: Path) -> None:
    """Diarization labels are per request; summing would invent extra people."""
    a = contrib("R2026-08-26-09-00-00.MP3", "one", speakers=2, minutes=5)
    b = contrib("R2026-08-26-11-00-00.MP3", "two", speakers=2, minutes=5)
    save_contribution(tmp_path, a)
    save_contribution(tmp_path, b)
    assert meta_from_contribs(load_kept_for_date(tmp_path, "2026-08-26")).speaker_count == 2


def test_forced_contribution_stays_out_of_a_normal_rebuild(tmp_path: Path) -> None:
    """--force-keep is a ONE-TIME override, not a permanent keep.

    The 2026-08-27 11:30 and 12:31 personal recordings were forced through and
    their sidecars stayed on disk as keep=True with 60k characters of garbled
    speech. Any later recording sharing their Outlook event would have merged
    that text into a work summary email -- the exact content the relevance gate
    exists to keep out, arriving through the day rebuild.
    """
    work = contrib("R2026-08-27-14-00-00.MP3", "Okta access review with Andrew.")
    personal = contrib("R2026-08-27-11-30-55.MP3", "Are you guys all set for food?")
    personal.forced = True
    save_contribution(tmp_path, work)
    save_contribution(tmp_path, personal)

    kept = load_kept_for_date(tmp_path, "2026-08-27")
    assert [c.source_file for c in kept] == ["R2026-08-27-14-00-00.MP3"]
    assert "food" not in combine_text(kept)


def test_forced_contribution_is_visible_to_the_run_that_forced_it(tmp_path: Path) -> None:
    """The override run still has to be able to email its own summary."""
    personal = contrib("R2026-08-27-11-30-55.MP3", "Are you guys all set for food?")
    personal.forced = True
    save_contribution(tmp_path, personal)

    assert load_kept_for_date(tmp_path, "2026-08-27") == []
    forced = load_kept_for_date(tmp_path, "2026-08-27", include_forced=True)
    assert [c.source_file for c in forced] == ["R2026-08-27-11-30-55.MP3"]


def test_forced_survives_the_json_round_trip(tmp_path: Path) -> None:
    personal = contrib("R2026-08-27-12-31-02.MP3", "Personal catch-up.")
    personal.forced = True
    save_contribution(tmp_path, personal)
    reloaded = load_contribution(contrib_path(tmp_path, "R2026-08-27-12-31-02.MP3"))
    assert reloaded.forced is True


def test_sidecar_written_before_forced_existed_is_not_forced(tmp_path: Path) -> None:
    """Old sidecars have no 'forced' key; they must load as ordinary keeps."""
    path = tmp_path / "R2026-08-26-09-04-45.contrib.json"
    path.write_text(
        json.dumps(
            {
                "source_file": "R2026-08-26-09-04-45.MP3",
                "keep": True,
                "text": "Okta access review with Andrew.",
                "recorded_at": "2026-08-26T09:04:45",
            }
        ),
        encoding="utf-8",
    )
    assert load_contribution(path).forced is False
    assert len(load_kept_for_date(tmp_path, "2026-08-26")) == 1


def test_corrupt_sidecar_does_not_abort_the_day(tmp_path: Path) -> None:
    save_contribution(tmp_path, contrib("R2026-08-26-09-04-45.MP3", "kept speech"))
    (tmp_path / "R2026-08-26-10-00-00.contrib.json").write_text("{not json", encoding="utf-8")
    kept = load_kept_for_date(tmp_path, "2026-08-26")
    assert len(kept) == 1


def test_parse_transcript_ignores_clarify_footer(tmp_path: Path) -> None:
    """The glossary flywheel footer must not be treated as spoken text."""
    path = tmp_path / "R2026-08-26-09-04-45.md"
    path.write_text(
        "\n".join([
            "# Transcript — R2026-08-26-09-04-45.MP3",
            "",
            "## 690s – 900s  (210s)",
            "",
            "- Quality: **clean**, +7.7 dB speech-minus-rumble",
            "- Source offset: 690.0s – 900.0s",
            "",
            "Would you mind sharing your screen?",
            "",
            "---",
            "",
            "## Terms to clarify",
            "",
            "- Cotera",
            "- Octo",
            "",
        ]),
        encoding="utf-8",
    )
    segs = parse_transcript(path)
    assert len(segs) == 1
    assert "sharing your screen" in segs[0].text
    assert "Cotera" not in segs[0].text
    assert "Terms to clarify" not in segs[0].text


def test_parse_transcript_carries_speakers_and_unintelligible_flag(tmp_path: Path) -> None:
    """Cached re-runs must still see the 13-voice flag; the live path wrote it."""
    path = tmp_path / "R2026-08-27-11-30-55.md"
    path.write_text(
        "\n".join([
            "# Transcript — R2026-08-27-11-30-55.MP3",
            "",
            "## 885s – 1485s  (600s)",
            "",
            "- Quality: **clean**, +4.9 dB speech-minus-rumble",
            "- Source offset: 885s – 1485s",
            "- unintelligible: diarizer reported 12 voices in one upload",
            "- speakers: @, A, B, C, D, E, F, G, H, I, J, K, L",
            "",
            "Are you guys all set for food?",
            "",
        ]),
        encoding="utf-8",
    )
    segs = parse_transcript(path)
    assert len(segs) == 1
    assert segs[0].unintelligible_because.startswith("diarizer reported")
    labels = {t["speaker"] for t in segs[0].speakers}
    assert "@" in labels
    assert "L" in labels
    assert "Are you guys all set for food?" in segs[0].text
