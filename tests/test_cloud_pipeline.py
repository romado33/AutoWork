#!/usr/bin/env python3
"""Tests for the cloud pipeline: ingest, upload splitting, mail config, summary shape.

Run:
    python -m pytest tests/ -v

No network and no API key required. Everything here is the logic that decides WHAT gets
uploaded, WHERE mail goes, and how a summary renders -- the parts a live run cannot
validate cheaply, because each live attempt costs money and minutes.
"""

from __future__ import annotations

import os

import pytest

from autowork.gate import Segment, Verdict
from autowork.mailer import MailConfig, MailError, build_message, default_recipient
from autowork.summarize import Summary
from autowork.transcribe_cloud import (
    MAX_UPLOAD_SEC,
    MIN_SEGMENT_SEC,
    split_for_upload,
)


def segment(start: float, end: float) -> Segment:
    return Segment(
        start_sec=start,
        end_sec=end,
        verdict=Verdict.CLEAN,
        mean_speech_db=-27.0,
        mean_delta_db=6.6,
        window_count=int((end - start) / 15),
    )


# --- upload splitting: this decides the bill and the failure blast radius ---------


def test_short_segment_uploads_whole() -> None:
    """Happy path."""
    assert split_for_upload(segment(0, 300)) == [(0, 300)]


def test_long_segment_is_split_by_duration_not_size() -> None:
    """The real 23-minute conversation. At 32 kbps it fits the size cap in one request,
    so only the duration cap prevents an 11-minute silent HTTP call whose failure would
    discard everything paid for so far."""
    windows = split_for_upload(segment(930, 2280))

    assert len(windows) == 3
    assert all(end - start <= MAX_UPLOAD_SEC for start, end in windows)
    # Contiguous and complete: no audio may be silently dropped between windows.
    assert windows[0][0] == 930
    assert windows[-1][1] == 2280
    for earlier, later in zip(windows, windows[1:]):
        assert earlier[1] == later[0]


def test_split_covers_the_whole_segment() -> None:
    """Edge: an awkward duration must not lose its tail."""
    windows = split_for_upload(segment(0, 1451))
    covered = sum(end - start for start, end in windows)
    assert covered == pytest.approx(1451)


def test_size_cap_still_binds_when_it_is_the_smaller_one() -> None:
    """Edge: with a tiny byte budget, size wins over duration."""
    windows = split_for_upload(segment(0, 3600), max_bytes=400_000)
    assert all(end - start <= 100 for start, end in windows)


def test_minimum_segment_is_above_a_gate_window() -> None:
    """A 15s gated window is one gate window wide. Uploading those cost two round trips
    and returned empty on real audio, so the minimum must exceed the window size."""
    assert MIN_SEGMENT_SEC > 15.0


# --- mail configuration: the credential mistakes that actually happen ------------


@pytest.fixture(autouse=True)
def clean_mail_env(monkeypatch):
    for name in (
        "SMTP_ADDRESS", "SMTP_APP_PASSWORD", "SMTP_PASSWORD", "SMTP_HOST",
        "SMTP_PORT", "GMAIL_EMAIL_ADDRESS", "GMAIL_ADDRESS", "GMAIL_APP_PASSWORD",
        "SUMMARY_TO",
    ):
        monkeypatch.delenv(name, raising=False)


def test_accepts_gmail_variable_names(monkeypatch) -> None:
    """GMAIL_* is what a person writes when setting up Gmail. Rejecting it because
    SMTP_* was documented is a pointless failure."""
    monkeypatch.setenv("GMAIL_EMAIL_ADDRESS", "romado33@gmail.com")
    monkeypatch.setenv("GMAIL_APP_PASSWORD", "abcdefghijklmnop")

    config = MailConfig.from_env()
    assert config.address == "romado33@gmail.com"
    assert config.app_password == "abcdefghijklmnop"


def test_strips_spaces_from_a_pasted_app_password(monkeypatch) -> None:
    """THE test. Google displays App Passwords as four space-separated groups; copying
    that verbatim gives 19 characters, and Gmail rejects it with an authentication
    error indistinguishable from a wrong password. This was hit for real."""
    monkeypatch.setenv("GMAIL_EMAIL_ADDRESS", "romado33@gmail.com")
    monkeypatch.setenv("GMAIL_APP_PASSWORD", "abcd efgh ijkl mnop")

    config = MailConfig.from_env()
    assert config.app_password == "abcdefghijklmnop"
    assert len(config.app_password) == 16


def test_smtp_names_win_over_gmail_names(monkeypatch) -> None:
    """Edge: an explicit SMTP_ setting is the more specific intent."""
    monkeypatch.setenv("SMTP_ADDRESS", "explicit@example.com")
    monkeypatch.setenv("GMAIL_EMAIL_ADDRESS", "fallback@gmail.com")
    monkeypatch.setenv("SMTP_APP_PASSWORD", "abcdefghijklmnop")

    assert MailConfig.from_env().address == "explicit@example.com"


def test_missing_credentials_name_both_accepted_forms(monkeypatch) -> None:
    """Expected failure: the error must say what to set, in either naming."""
    with pytest.raises(MailError, match="GMAIL_EMAIL_ADDRESS"):
        MailConfig.from_env()


def test_non_numeric_port_is_rejected(monkeypatch) -> None:
    """Expected failure."""
    monkeypatch.setenv("GMAIL_EMAIL_ADDRESS", "a@b.com")
    monkeypatch.setenv("GMAIL_APP_PASSWORD", "abcdefghijklmnop")
    monkeypatch.setenv("SMTP_PORT", "not-a-port")

    with pytest.raises(MailError, match="SMTP_PORT"):
        MailConfig.from_env()


def test_recipient_defaults_to_the_sender(monkeypatch) -> None:
    """You are mailing yourself; requiring SUMMARY_TO as well is needless ceremony."""
    monkeypatch.setenv("GMAIL_EMAIL_ADDRESS", "romado33@gmail.com")
    assert default_recipient() == "romado33@gmail.com"


def test_explicit_recipient_wins(monkeypatch) -> None:
    monkeypatch.setenv("GMAIL_EMAIL_ADDRESS", "sender@gmail.com")
    monkeypatch.setenv("SUMMARY_TO", "elsewhere@example.com")
    assert default_recipient() == "elsewhere@example.com"


def test_message_has_the_headers_a_mail_client_needs() -> None:
    message = build_message(
        sender="a@b.com", sender_name="AutoWork", recipient="c@d.com",
        subject="Work summary", text_body="body",
    )
    assert message["To"] == "c@d.com"
    assert message["Subject"] == "Work summary"
    assert "AutoWork" in message["From"]
    assert message["Date"]


def test_empty_recipient_is_refused() -> None:
    """Expected failure: a summary sent nowhere must not look like success."""
    from autowork.mailer import send

    config = MailConfig(address="a@b.com", app_password="abcdefghijklmnop")
    with pytest.raises(MailError, match="no recipient"):
        send("  ", "subject", "body", config=config)


# --- summary rendering -----------------------------------------------------------


def summary() -> Summary:
    return Summary(
        headline="Entity mapping weak globally, strong per-customer",
        topics=[{"label": "Mapping accuracy", "summary": "~28% exact, ~90% per customer"}],
        decisions=[{"decision": "Table the global-rules approach", "quote": "", "verified": True}],
        open_questions=["Where should confidence filtering live?"],
    )


def test_text_rendering_carries_every_section() -> None:
    """The mail body is plain text, so it must not depend on markdown to be readable."""
    text = summary().to_text()

    assert "Entity mapping weak globally" in text
    assert "~28% exact" in text
    assert "Table the global-rules approach" in text
    assert "Where should confidence filtering live?" in text


def test_markdown_rendering_carries_every_section() -> None:
    markdown = summary().to_markdown()
    for expected in ("## Discussed", "## Decided", "## Left open"):
        assert expected in markdown


def test_metadata_header_is_derived_not_generated() -> None:
    """Dates and durations are facts from the file. A model-invented timestamp on a
    work summary is worse than none, because it gets trusted and filed."""
    from autowork.summarize import SummaryMeta

    meta = SummaryMeta.from_filename("R2026-08-25-13-23-54.MP3")
    meta.audio_minutes = 23
    meta.speaker_count = 2
    rendered = "\n".join(
        Summary(
            headline="h", topics=[], decisions=[], open_questions=[], meta=meta,
        ).meta_lines()
    )

    assert "Tuesday 25 August 2026, 13:23" in rendered
    assert "23 min of speech" in rendered
    # The diarizer's voice count is the only real participant signal. A separate list
    # of names mentioned was removed at the operator's request: on a real two-person
    # call it listed four absent colleagues and read as an attendee list.
    assert "Participants  2 (distinct voices heard)" in rendered
    assert "Came up" not in rendered


def test_unparseable_filename_does_not_invent_a_date() -> None:
    """Expected failure mode handled: no date beats a wrong one."""
    from autowork.summarize import SummaryMeta

    meta = SummaryMeta.from_filename("some-other-file.mp3")
    assert meta.recorded_at is None
    assert "unknown" in meta.when


def test_actions_render_with_their_grounding_quote() -> None:
    """The to-do section is the detailed half; the quote is what makes it verifiable."""
    from autowork.action import ActionRecord, ActionType, Provenance

    action = ActionRecord(
        title="Finalise the backlog tool",
        body="Take it up again with Dave Casale.",
        target_system="backlog-tool",
        action_type=ActionType.UPDATE,
        confidence=0.82,
        provenance=Provenance(
            source_audio="R2026-08-25-13-23-54.MP3",
            start_sec=2100.0, end_sec=2220.0, speech_rumble_db=6.6,
            transcript_excerpt="I should take it up again with Dave Casale",
            extractor="openai:gpt-5.4-mini/grounded",
        ),
    )
    text = summary().to_text(actions=[action])

    assert "TO DO (1)" in text
    assert "Finalise the backlog tool" in text
    assert "I should take it up again with Dave Casale" in text
    assert "confidence 0.82" in text
    # The email must never imply anything was done.
    assert "have been executed" in text
    assert "review.bat" in text


def test_no_actions_means_no_todo_section() -> None:
    """Edge: an empty TO DO heading reads as a bug."""
    assert "TO DO" not in summary().to_text(actions=[])


def test_empty_sections_are_omitted_not_left_as_empty_headings() -> None:
    """Edge: an empty 'Decided' heading reads as a bug, and 'nothing was decided' is
    the common case in a working conversation."""
    sparse = Summary(headline="Quiet call", topics=[], decisions=[], open_questions=[])
    markdown = sparse.to_markdown()

    assert "Quiet call" in markdown
    assert "## Decided" not in markdown
    assert "## Left open" not in markdown


def test_from_dict_tolerates_missing_keys() -> None:
    """Edge: schema-enforced output should always be complete, but a defaulting parser
    means a schema change degrades rather than crashes the nightly run."""
    parsed = Summary.from_dict({"headline": "Only a headline"})

    assert parsed.headline == "Only a headline"
    assert parsed.topics == []
    assert parsed.open_questions == []


# --- concurrent transcription plumbing ---------------------------------------------


class _StubResponse:
    def __init__(self, text): self.text = text; self.segments = []


class _StubClient:
    """Answers transcription calls after a small sleep, recording concurrency."""

    def __init__(self):
        import threading
        self.lock = threading.Lock()
        self.active = 0
        self.peak = 0
        self.calls = []

        stub = self

        class _Transcriptions:
            def create(self, file, **kwargs):
                import time
                with stub.lock:
                    stub.active += 1
                    stub.peak = max(stub.peak, stub.active)
                time.sleep(0.05)
                name = getattr(file, "name", "?")
                with stub.lock:
                    stub.active -= 1
                    stub.calls.append(name)
                return _StubResponse(f"words from {name}")

        class _Audio:
            transcriptions = _Transcriptions()

        self.audio = _Audio()


def test_concurrent_uploads_preserve_chronological_order(tmp_path, monkeypatch) -> None:
    """Requests may complete in any order; the transcript must stay chronological.

    Also proves the pool genuinely overlaps requests -- if a refactor silently
    serialises it (an easy regression), peak concurrency drops to 1 and this fails.
    """
    import subprocess
    import autowork.transcribe_cloud as tc

    # A fake 40-minute recording: silence is fine, we stub the gate and the encoder.
    audio = tmp_path / "R2026-08-25-09-00-00.MP3"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
         "anullsrc=r=16000:cl=mono", "-t", "5", str(audio)],
        check=True, capture_output=True,
    )

    from autowork.gate import Segment, Verdict

    fake_segments = [
        Segment(start_sec=0, end_sec=1500, verdict=Verdict.CLEAN,
                mean_speech_db=-27, mean_delta_db=6.6, window_count=100),
        Segment(start_sec=1600, end_sec=2400, verdict=Verdict.CLEAN,
                mean_speech_db=-27, mean_delta_db=6.6, window_count=53),
    ]
    monkeypatch.setattr(tc, "measure", lambda *a, **k: [])
    monkeypatch.setattr(tc, "segments", lambda *a, **k: fake_segments)

    def fake_encode(source, start, end, dest):
        dest.write_bytes(b"mp3")
        return 3

    monkeypatch.setattr(tc, "encode_slice", fake_encode)

    stub = _StubClient()
    result = tc.transcribe_file_cloud(audio, client=stub)

    # 1500s splits into 3 windows of <=600s, plus 800s into 2: five uploads total.
    assert len(result) == 5
    starts = [seg.start_sec for seg in result]
    assert starts == sorted(starts), "transcript order must be chronological"
    assert stub.peak > 1, "uploads ran sequentially; the pool is not overlapping"


def test_transient_failure_is_retried_then_succeeds(tmp_path, monkeypatch) -> None:
    """A network blip must cost a retry, not a permanent hole in the transcript."""
    import autowork.transcribe_cloud as tc

    attempts = {"n": 0}

    def flaky(client, path, model):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise tc.CloudTranscriptionError("HTTP 429, slow down")
        return "recovered text", []

    monkeypatch.setattr(tc, "_transcribe_file", flaky)
    monkeypatch.setattr(tc.__dict__["_transcribe_with_retry"].__globals__["logger"], "warning", lambda *a, **k: None)
    # Shrink the backoff so the test is fast.
    import time as _time
    monkeypatch.setattr(_time, "sleep", lambda s: None)

    text, speakers = tc._transcribe_with_retry(None, tmp_path / "x.mp3", "m")
    assert text == "recovered text"
    assert attempts["n"] == 3


# --- ingest ledger: "new" must survive manual deletion of local copies -------------


@pytest.fixture()
def fake_device(tmp_path, monkeypatch):
    """A pretend recorder volume with two recordings on it."""
    import autowork.ingest as ing

    device = tmp_path / "device"
    (device / "RECORD").mkdir(parents=True)
    (device / "RECORD" / "R2026-08-25-09-00-00.MP3").write_bytes(b"a" * 200_000)
    (device / "RECORD" / "R2026-08-25-14-00-00.MP3").write_bytes(b"b" * 300_000)

    monkeypatch.setattr(ing, "find_device", lambda serial: device)
    return device


def test_first_ingest_copies_everything(fake_device, tmp_path) -> None:
    """Happy path."""
    from autowork.ingest import ingest

    result = ingest("AA986EA1", tmp_path / "audio")
    assert len(result.copied) == 2
    assert len(result.skipped) == 0


def test_second_ingest_copies_nothing(fake_device, tmp_path) -> None:
    """Idempotence."""
    from autowork.ingest import ingest

    dest = tmp_path / "audio"
    ingest("AA986EA1", dest)
    result = ingest("AA986EA1", dest)

    assert result.copied == []
    assert len(result.skipped) == 2


def test_manually_deleted_file_is_not_resurrected(fake_device, tmp_path) -> None:
    """THE test. The operator manages retention by deleting local copies by hand; the
    device never deletes. Without the ledger, the next run re-copies the file from the
    device and re-processes it at real cost."""
    from autowork.ingest import ingest

    dest = tmp_path / "audio"
    ingest("AA986EA1", dest)

    victim = dest / "R2026-08-25-09-00-00.MP3"
    victim.unlink()  # the manual retention delete

    result = ingest("AA986EA1", dest)
    assert result.copied == []
    assert not victim.exists(), "deleted recording came back from the device"


def test_grown_file_under_same_name_is_recopied(fake_device, tmp_path) -> None:
    """Edge: the recorder appends to the same file if recording resumes. A size change
    under a known name is NEW content and must not be skipped by the ledger."""
    from autowork.ingest import ingest

    dest = tmp_path / "audio"
    ingest("AA986EA1", dest)

    grown = fake_device / "RECORD" / "R2026-08-25-14-00-00.MP3"
    grown.write_bytes(b"b" * 500_000)

    result = ingest("AA986EA1", dest)
    assert [p.name for p in result.copied] == ["R2026-08-25-14-00-00.MP3"]
    assert (dest / "R2026-08-25-14-00-00.MP3").stat().st_size == 500_000


def test_pre_ledger_files_are_adopted(fake_device, tmp_path) -> None:
    """Edge: files ingested before the ledger existed must be adopted into it, so a
    later manual deletion stays deleted."""
    import shutil
    from autowork.ingest import LEDGER_NAME, ingest

    dest = tmp_path / "audio"
    dest.mkdir()
    # Simulate a pre-ledger ingest: file present, no ledger.
    shutil.copy2(fake_device / "RECORD" / "R2026-08-25-09-00-00.MP3", dest)

    ingest("AA986EA1", dest)
    assert (dest / LEDGER_NAME).is_file()

    (dest / "R2026-08-25-09-00-00.MP3").unlink()
    result = ingest("AA986EA1", dest)
    assert "R2026-08-25-09-00-00.MP3" not in [p.name for p in result.copied]


def test_corrupt_ledger_degrades_to_recopying_not_crashing(fake_device, tmp_path) -> None:
    """Expected failure mode handled: a mangled ledger must not brick ingest."""
    from autowork.ingest import LEDGER_NAME, ingest

    dest = tmp_path / "audio"
    ingest("AA986EA1", dest)
    (dest / LEDGER_NAME).write_text("{not json", encoding="utf-8")

    result = ingest("AA986EA1", dest)  # must not raise
    assert result.copied == []  # local copies still present, name+size check holds
