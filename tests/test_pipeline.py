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
    sent_kwargs: list = []

    def stub_send(*a, **k):
        sent.append(a)
        sent_kwargs.append(k)
        return "stubbed"

    monkeypatch.setattr(run_pipeline, "send", stub_send)

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
    go.sent_kwargs = sent_kwargs
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


def test_a_second_run_of_the_same_files_does_not_resend(run, monkeypatch) -> None:
    """--files is how an operator recovers a failed email. It must not also
    duplicate a successful one. --resend is the explicit override."""
    monkeypatch.setattr(
        run_pipeline,
        "summarise",
        lambda *a, **k: Summary(
            headline="Okta access review",
            topics=[{"label": "Okta", "summary": "Move the review to Thursday."}],
            decisions=[],
            open_questions=[],
        ),
    )
    audio = make_recording(run, "R2026-08-26-09-04-45", WORK)
    assert run("--files", str(audio), "--no-extract", "--to", "rob@example.com") == 0
    assert len(run.sent) == 1
    assert run("--files", str(audio), "--no-extract", "--to", "rob@example.com") == 0
    assert len(run.sent) == 1
    assert run(
        "--files", str(audio), "--no-extract", "--resend", "--to", "rob@example.com"
    ) == 0
    assert len(run.sent) == 2


def test_extract_of_one_file_overlaps_transcribe_of_the_next(run, monkeypatch) -> None:
    """Extraction is ~7s of CPU-bound API time after a multi-minute transcribe.
    Starting it while the next file uploads costs nothing and saves those 7s
    on every extra recording in a USB batch.
    """
    import time
    from autowork.transcribe_cloud import CloudSegment

    events: list[tuple[str, str, float]] = []

    def slow_transcribe(path, **_k):
        events.append(("t-start", path.name, time.monotonic()))
        time.sleep(0.12)
        events.append(("t-end", path.name, time.monotonic()))
        return [
            CloudSegment(
                source_audio=str(path),
                start_sec=0,
                end_sec=60,
                verdict="clean",
                speech_rumble_db=6.6,
                text=WORK,
            )
        ]

    def slow_extract(segments, path, queue_path):
        events.append(("e-start", path.name, time.monotonic()))
        time.sleep(0.12)
        events.append(("e-end", path.name, time.monotonic()))
        return []

    monkeypatch.setattr(run_pipeline, "transcribe_file_cloud", slow_transcribe)
    monkeypatch.setattr(run_pipeline, "extract_into_queue", slow_extract)

    first = run.audio / "R2026-08-26-09-04-45.MP3"
    second = run.audio / "R2026-08-26-11-00-00.MP3"
    first.write_bytes(b"")
    second.write_bytes(b"")
    assert run("--files", str(first), str(second), "--no-email") == 0

    e1_start = next(t for n, p, t in events if n == "e-start" and "09-04" in p)
    t2_start = next(t for n, p, t in events if n == "t-start" and "11-00" in p)
    e1_end = next(t for n, p, t in events if n == "e-end" and "09-04" in p)
    t2_end = next(t for n, p, t in events if n == "t-end" and "11-00" in p)
    assert e1_start < t2_end and t2_start < e1_end, (
        "extract of the first file ran only after the second transcription finished"
    )


def test_raw_transcript_is_spoken_text_not_gate_metadata() -> None:
    """The mailed .md transcript is for audit (quality, offsets). The operator
    asked for the words as well, without speech-minus-rumble wrapping them.
    """
    from autowork.gate import Verdict
    from autowork.transcribe import TranscribedSegment

    segs = [
        TranscribedSegment(
            source_audio="R2026-09-02-11-02-43.MP3",
            start_sec=72,
            end_sec=140,
            verdict=Verdict.CLEAN,
            speech_rumble_db=6.6,
            text="Work History is ready, needs a token.",
            speakers=[
                {"speaker": "A", "start": 72.0, "text": "Work History is ready, needs a token."},
                {"speaker": "B", "start": 91.0, "text": "And then AI text transformation."},
            ],
        )
    ]
    raw = run_pipeline.render_raw_transcript(Path("R2026-09-02-11-02-43.MP3"), segs)
    assert "Work History is ready, needs a token." in raw
    assert "And then AI text transformation." in raw
    assert "[01:12 A]" in raw
    assert "speech-minus-rumble" not in raw
    assert "**clean**" not in raw


def test_cached_run_writes_raw_transcript_and_attaches_it(run, monkeypatch) -> None:
    """A USB run already wrote transcripts/*.md; the operator still did not get
    the words in the inbox. Write stem.raw.txt and attach it to the summary mail.
    """
    monkeypatch.setattr(
        run_pipeline,
        "summarise",
        lambda *a, **k: Summary(
            headline="Okta access review",
            topics=[{"label": "Okta", "summary": "Move the review to Thursday."}],
            decisions=[],
            open_questions=[],
        ),
    )
    audio = make_recording(run, "R2026-08-26-09-04-45", WORK)
    assert run("--files", str(audio), "--no-extract", "--to", "rob@example.com") == 0
    raw = run.transcripts / "R2026-08-26-09-04-45.raw.txt"
    assert raw.is_file()
    text = raw.read_text(encoding="utf-8")
    assert WORK in text
    assert "speech-minus-rumble" not in text
    assert len(run.sent) == 1
    attached = run.sent_kwargs[0].get("attachments") or []
    assert [p.name for p in attached] == ["R2026-08-26-09-04-45.raw.txt"]


def test_operator_context_reaches_the_summariser(run, tmp_path, monkeypatch) -> None:
    """2026-09-02 Dan/Aurea: two feature tables were on screen. Without them as
    context the summary cannot map Whisper near-misses onto the numbered rows
    or keep per-row changes. The tables are a roster, not a second source of facts.
    """
    extras: list[str] = []

    def stub_summarise(text, *a, extra_instructions="", **k):
        extras.append(extra_instructions)
        return Summary(
            headline="stub",
            topics=[{"label": "Work History", "summary": "needs token"}],
            decisions=[],
            open_questions=[],
        )

    monkeypatch.setattr(run_pipeline, "summarise", stub_summarise)
    table = tmp_path / "features.md"
    table.write_text("| # | Feature |\n| 1 | Work History |\n", encoding="utf-8")
    audio = make_recording(run, "R2026-08-26-09-04-45", WORK)
    assert run(
        "--files", str(audio), "--context", str(table), "--no-extract", "--no-email"
    ) == 0
    assert extras, "summarise was not called"
    blob = extras[0]
    assert "Work History" in blob
    assert "Do not invent discussion" in blob
    assert "features.md" in blob


def test_html_context_drops_css_and_keeps_table_cells() -> None:
    html = """
    <html><head><style>.secret { background: magenta; }</style></head>
    <body>
      <h1>CSM Feature Shortlist</h1>
      <table>
        <tr><th>Feature</th><th>Status</th></tr>
        <tr><td>TrueContext Teamwork</td><td>Live</td></tr>
      </table>
    </body></html>
    """
    text = run_pipeline.html_to_visible_text(html)
    assert "TrueContext Teamwork" in text
    assert "Live" in text
    assert "magenta" not in text
    assert "background" not in text
