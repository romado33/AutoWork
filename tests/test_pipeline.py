#!/usr/bin/env python3
"""Orchestrator failure modes, all three measured on the real pipeline.

A dry run that was not dry, a relevance gate that judged a cached recording
differently from a fresh one, and a single unreadable file taking the rest of the
day's audio down with it. Every test drives run_pipeline.main(); the cloud calls
(transcribe, classify, summarise, send) are stubbed, so nothing here bills or mails.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

import run_pipeline  # noqa: E402
from autowork.calendar_lookup import CalendarMatch  # noqa: E402
from autowork.day import contrib_path, load_contribution  # noqa: E402
from autowork.glossary import Glossary  # noqa: E402
from autowork.relevance import Verdict  # noqa: E402
from autowork.summarize import Summary  # noqa: E402

WORK = "We should move the Okta access review to Thursday so Andrew can join."
PERSONAL = "Are you guys all set for food? I think the kids want pizza again."


def write_transcript(path: Path, *chunks: str) -> Path:
    """A transcript in the shape render_transcript() writes and parse_transcript() reads."""
    lines = [f"# Transcript — {path.stem}.MP3", ""]
    start = 0
    for chunk in chunks:
        lines += [
            f"## {start}s – {start + 300}s  (300s)",
            "",
            "- Quality: **clean**, +7.7 dB speech-minus-rumble",
            f"- Source offset: {start}s – {start + 300}s",
            "",
            chunk,
            "",
        ]
        start += 300
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def stub_classify(text: str, *_args, **_kwargs) -> Verdict:
    """The pizza segment is personal; everything else is work."""
    if "pizza" in text:
        return Verdict(
            is_work=False,
            confidence=0.95,
            kind="personal_conversation",
            reason="family logistics",
        )
    return Verdict(
        is_work=True, confidence=0.95, kind="work_conversation", reason="access review"
    )


@pytest.fixture
def run(monkeypatch, tmp_path):
    """A pipeline whose paid, outbound and desktop calls are stubbed out.

    SUMMARY_TO in the operator's real .env is loaded by main(); send() is replaced
    so a test can never put a stubbed summary in someone's inbox. match_recording
    is replaced because it Dispatches to live Outlook over COM -- the suite was
    reaching into the operator's calendar and Windows was logging a fatal COM
    exception (0x80010001) for every run.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "not-a-real-key")
    monkeypatch.setattr(run_pipeline, "classify", stub_classify)
    monkeypatch.setattr(run_pipeline, "match_recording", lambda *a, **k: None)

    # Record what reaches the summariser. Asserting on the rendered summary file
    # instead would pass no matter what, because a stub summary never echoes its
    # input -- the text handed to the model is the thing under test.
    summarised: list[str] = []

    def stub_summarise(text: str, *_a, **_k) -> Summary:
        summarised.append(text)
        return Summary(headline="stub", topics=[], decisions=[], open_questions=[])

    monkeypatch.setattr(run_pipeline, "summarise", stub_summarise)
    sent: list = []
    monkeypatch.setattr(
        run_pipeline, "send", lambda *a, **k: sent.append(a) or "stubbed"
    )

    transcripts = tmp_path / "transcripts"
    transcripts.mkdir()

    def go(*argv: str) -> int:
        return run_pipeline.main(
            [
                "--transcripts",
                str(transcripts),
                "--queue",
                str(tmp_path / "queue.sqlite3"),
                *argv,
            ]
        )

    go.audio = tmp_path
    go.transcripts = transcripts
    go.queue = tmp_path / "queue.sqlite3"
    go.sent = sent
    go.summarised = summarised
    return go


def make_recording(run, stem: str, *chunks: str) -> Path:
    audio = run.audio / f"{stem}.MP3"
    audio.write_bytes(b"")
    write_transcript(run.transcripts / f"{stem}.md", *chunks)
    return audio


def test_dry_run_writes_nothing_for_an_already_transcribed_file(run, monkeypatch) -> None:
    """--dry-run promises "gate only, upload nothing".

    The check sat AFTER the cached-transcript branch, so any file with an existing
    .md still had its glossary rewritten, its sidecar written and its actions
    queued. An operator asking what a run would do got side effects instead.
    """
    audio = make_recording(run, "R2026-08-26-09-04-45", WORK)
    before = (run.transcripts / "R2026-08-26-09-04-45.md").read_text(encoding="utf-8")

    queued: list = []
    monkeypatch.setattr(
        run_pipeline, "extract_into_queue", lambda *a, **k: queued.append(a) or []
    )

    assert run("--files", str(audio), "--dry-run") == 0
    assert queued == []
    assert list(run.transcripts.glob("*.contrib.json")) == []
    assert not run.queue.exists()
    assert (run.transcripts / "R2026-08-26-09-04-45.md").read_text(
        encoding="utf-8"
    ) == before
    assert run.sent == []


def test_cached_relevance_is_per_segment_like_the_fresh_path(run) -> None:
    """One personal segment beside a work one must not drop the whole recording.

    The fresh path classified each segment; the cached path classified the joined
    text. The same recording was therefore keepable on first transcribe and
    droppable on re-run, and vice versa.
    """
    audio = make_recording(run, "R2026-08-26-09-04-45", WORK, PERSONAL)

    assert run("--files", str(audio), "--no-extract", "--no-email") == 0

    kept = load_contribution(contrib_path(run.transcripts, "R2026-08-26-09-04-45.MP3"))
    assert kept.keep
    assert "Okta access review" in kept.text
    assert "pizza" not in kept.text


def test_cached_recording_with_no_work_segment_is_excluded(run) -> None:
    audio = make_recording(run, "R2026-08-26-09-04-45", PERSONAL)

    assert run("--files", str(audio), "--no-extract", "--no-email") == 0

    kept = load_contribution(contrib_path(run.transcripts, "R2026-08-26-09-04-45.MP3"))
    assert not kept.keep
    assert kept.text == ""
    assert run.sent == []


def test_one_unreadable_recording_does_not_abort_the_others(run, monkeypatch) -> None:
    """A locked transcript or a busy queue used to end the whole batch.

    Only CloudTranscriptionError, RelevanceError, SummaryError and MailError were
    caught; an OSError from the glossary rewrite, the sidecar write or the SQLite
    queue escaped the loop and abandoned every recording after it.
    """
    bad = make_recording(run, "R2026-08-26-09-04-45", WORK)
    good = make_recording(run, "R2026-08-26-11-00-00", WORK)

    real_rewrite = Glossary.rewrite_file

    def flaky(self, path):
        if "09-04-45" in Path(path).name:
            raise OSError("transcript is locked by another process")
        return real_rewrite(self, path)

    monkeypatch.setattr(Glossary, "rewrite_file", flaky)

    assert run("--files", str(bad), str(good), "--no-extract", "--no-email") == 1

    assert not contrib_path(run.transcripts, "R2026-08-26-09-04-45.MP3").exists()
    survivor = load_contribution(
        contrib_path(run.transcripts, "R2026-08-26-11-00-00.MP3")
    )
    assert survivor.keep
    assert "Okta access review" in survivor.text


def test_force_keep_marks_its_sidecar_as_an_override(run) -> None:
    """So the next normal run cannot rebuild the personal text into a work email.

    The override run must still summarise its own recording, which is why the
    exclusion is a load-time filter the forcing run opts out of rather than a
    refusal to write the sidecar at all.
    """
    audio = make_recording(run, "R2026-08-27-11-30-55", PERSONAL)

    assert run("--files", str(audio), "--force-keep", "--no-email") == 0

    forced = load_contribution(contrib_path(run.transcripts, "R2026-08-27-11-30-55.MP3"))
    assert forced.keep
    assert forced.forced is True
    assert (run.audio / "summaries" / "2026-08-27-11-30.md").exists()


def test_a_forced_recording_does_not_return_on_a_later_normal_run(
    run, monkeypatch
) -> None:
    """The leak this exists to stop: 60k characters of garbled personal speech
    riding into a work summary because a later recording shared its Outlook event.

    Both recordings resolve to the SAME calendar event, which is the only way
    group_conversations merges two files. Without that the two sit in separate
    file-keyed buckets and the leak cannot be reproduced.
    """
    shared = CalendarMatch(
        subject="Weekly sync",
        invitees=("andrew@example.com",),
        start=datetime(2026, 8, 27, 11, 0),
    )
    monkeypatch.setattr(run_pipeline, "match_recording", lambda *a, **k: shared)

    personal = make_recording(run, "R2026-08-27-11-30-55", PERSONAL)
    assert run("--files", str(personal), "--force-keep", "--no-email") == 0
    assert "pizza" in run.summarised[0]

    work = make_recording(run, "R2026-08-27-14-00-00", WORK)
    assert run("--files", str(work), "--no-extract", "--no-email") == 0

    assert len(run.summarised) == 2
    assert "Okta access review" in run.summarised[1]
    assert "pizza" not in run.summarised[1]
