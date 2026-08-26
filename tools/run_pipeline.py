#!/usr/bin/env python3
"""End to end: ingest from the recorder, transcribe, summarise, extract, queue, email.

Usage:
    python tools/run_pipeline.py                      # full run, ingest from the device
    python tools/run_pipeline.py --files audio/*.MP3   # skip ingest, use these files
    python tools/run_pipeline.py --no-email            # everything but the send
    python tools/run_pipeline.py --no-diarize          # faster, no speaker labels
    python tools/run_pipeline.py --dry-run             # gate only, upload nothing

Configuration lives in .env, never in source or on the command line:

    OPENAI_API_KEY        required
    RECORDER_SERIAL       volume serial of the recorder (default AA986EA1)
    SMTP_ADDRESS          sending account, for the summary email
    SMTP_APP_PASSWORD     Google App Password, not the account password
    SUMMARY_TO            recipient

WHAT IS AUTOMATED AND WHAT IS NOT. Ingest, gating, transcription, summarising,
extraction and the email all run unattended: they are deterministic, gated, or
read-only. Action items land in the review queue as PENDING and are NEVER executed by
this script. Plugging in a USB stick must not be able to write to Jira.

Every stage is idempotent. A recording already transcribed is skipped, the queue dedupes
on source audio plus window plus title, and a re-run after a failure resumes rather than
duplicating. That matters because the expensive stage is billed per minute.

Exit codes: 0 all good · 1 one or more recordings failed · 2 bad arguments or config
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

from autowork.extract import ExtractionError, ExtractorConfig, extract_from_segment  # noqa: E402
from autowork.gate import GateConfig  # noqa: E402
from autowork.glossary import Glossary, GlossaryError  # noqa: E402
from autowork.ingest import IngestError, ingest  # noqa: E402
from autowork.llm import load_dotenv  # noqa: E402
from autowork.mailer import MailError, default_recipient, send  # noqa: E402
from autowork.prefilter import select  # noqa: E402
from autowork.action import Status  # noqa: E402
from autowork.queue import ReviewQueue  # noqa: E402
from autowork.relevance import RelevanceError, classify  # noqa: E402
from autowork.summarize import SummaryError, SummaryMeta, summarise  # noqa: E402
from autowork.transcribe import TranscribedSegment  # noqa: E402
from autowork.transcribe_cloud import (  # noqa: E402
    DEFAULT_MODEL,
    DIARIZE_MODEL,
    CloudTranscriptionError,
    transcribe_file_cloud,
)
from autowork.gate import Verdict  # noqa: E402

DEFAULT_SERIAL = "AA986EA1"


def render_transcript(path: Path, segs: list) -> str:
    """Markdown transcript, in the shape the downstream tools already parse."""
    lines = [
        f"# Transcript — {path.name}",
        "",
        f"- Segments kept: {len(segs)}",
        f"- Audio transcribed: {sum(s.duration_sec for s in segs):.0f}s",
        f"- Uploaded: {sum(s.uploaded_bytes for s in segs) / 1e6:.2f} MB",
        "",
        "Only audio passing the quality gate appears below. Rejected audio is not",
        "transcribed, because Whisper invents fluent text from low-signal input.",
        "",
    ]
    for seg in segs:
        lines += [
            f"## {seg.start_sec:.0f}s – {seg.end_sec:.0f}s  ({seg.duration_sec:.0f}s)",
            "",
            f"- Quality: **{seg.verdict}**, {seg.speech_rumble_db:+.1f} dB "
            f"speech-minus-rumble",
            f"- Source offset: {seg.start_sec:.0f}s – {seg.end_sec:.0f}s",
        ]
        if seg.glossary_applied:
            lines.append(f"- corrected: {', '.join(seg.glossary_applied)}")
        speakers = sorted({s["speaker"] for s in seg.speakers if s.get("speaker")})
        if speakers:
            lines.append(f"- speakers: {', '.join(speakers)}")
        lines += ["", seg.text or "_(no speech recognised)_", ""]

        if seg.speakers:
            lines += ["<details><summary>By speaker</summary>", ""]
            for turn in seg.speakers:
                stamp = turn.get("start") or 0.0
                lines.append(
                    f"- **{turn.get('speaker', '?')}** ({stamp:.0f}s): "
                    f"{turn.get('text', '').strip()}"
                )
            lines += ["", "</details>", ""]
    return "\n".join(lines)


def extract_into_queue(segments: list, path: Path, queue_path: str) -> list[str]:
    """Prefilter, extract grounded actions, and file them. Returns queued ids.

    One helper for both the fresh-transcription and cached-transcript paths, because
    the two drifted twice and each drift produced an email with an empty to-do list
    while the queue held real items. Accepts CloudSegment or TranscribedSegment; only
    the fields both carry are used.
    """
    joined = "\n\n".join(s.text for s in segments if s.text.strip())
    filtered = select(joined)
    if not filtered.passages:
        print("  no commitment language found; nothing queued")
        return []

    first, last = segments[0], segments[-1]
    verdict_value = getattr(first.verdict, "value", first.verdict)
    pseudo = TranscribedSegment(
        source_audio=str(path),
        start_sec=first.start_sec,
        end_sec=last.end_sec,
        verdict=Verdict.CLEAN if verdict_value == "clean" else Verdict.USABLE,
        speech_rumble_db=first.speech_rumble_db,
        text="\n\n".join(p.text for p in filtered.passages),
        glossary_applied=sorted({t for s in segments for t in s.glossary_applied}),
    )
    try:
        accepted, rejected = extract_from_segment(pseudo, ExtractorConfig())
    except ExtractionError as exc:
        print(f"  extraction failed: {exc}", file=sys.stderr)
        return []

    queued: list[str] = []
    fresh = 0
    with ReviewQueue(queue_path) as queue:
        # Containment check against what is ALREADY pending for this recording, not
        # just exact-quote identity. Measured: a re-run re-extracted the same spoken
        # commitments with quote spans a few words longer or shorter, and the queue
        # doubled (6 pending for 3 real actions). A new quote that contains or is
        # contained by an existing pending quote is the same moment heard again.
        from autowork.extract import quotes_overlap

        existing = [
            a.provenance.transcript_excerpt
            for a in queue.list(status=Status.PENDING)
            if a.provenance.source_audio == str(path)
        ]
        before = queue.counts().get("pending", 0)
        for record in accepted:
            quote = record.provenance.transcript_excerpt
            if any(quotes_overlap(quote, known) for known in existing):
                print(f"    = already pending (overlapping quote): {record.title[:60]}")
                continue
            queued.append(queue.enqueue(record))
            existing.append(quote)
        fresh = queue.counts().get("pending", 0) - before
    print(f"  queued {fresh} new action(s) "
          f"({len(accepted) - fresh} already known, {len(rejected)} rejected by grounding)")
    for record in accepted:
        print(f"    + {record.title}")
    return queued


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--files", nargs="*", help="skip ingest; process these files")
    parser.add_argument("--serial", default=None, help="recorder volume serial")
    parser.add_argument("--audio-dir", default="audio")
    parser.add_argument("--transcripts", default="transcripts")
    parser.add_argument("--queue", default="queue.sqlite3")
    parser.add_argument("--no-diarize", action="store_true")
    parser.add_argument("--no-email", action="store_true")
    parser.add_argument("--no-extract", action="store_true")
    parser.add_argument("--no-relevance", action="store_true",
                        help="skip the work-conversation check")
    parser.add_argument("--dry-run", action="store_true", help="gate only, no uploads")
    parser.add_argument("--to", default=None, help="summary recipient")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(message)s",
    )
    load_dotenv(PROJECT_ROOT / ".env")

    serial = args.serial or os.environ.get("RECORDER_SERIAL", DEFAULT_SERIAL)
    recipient = args.to or os.environ.get("SUMMARY_TO", "")
    audio_dir = Path(args.audio_dir)
    transcript_dir = Path(args.transcripts)
    transcript_dir.mkdir(parents=True, exist_ok=True)

    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set (put it in .env)", file=sys.stderr)
        return 2

    try:
        glossary = Glossary.load(PROJECT_ROOT / "config" / "glossary.yml")
    except GlossaryError as exc:
        print(f"glossary: {exc}", file=sys.stderr)
        return 2

    # --- 1. gather ------------------------------------------------------------
    if args.files:
        paths = [Path(p) for pattern in args.files for p in glob.glob(pattern)]
        print(f"using {len(paths)} file(s) from the command line")
    else:
        try:
            result = ingest(serial, audio_dir)
        except IngestError as exc:
            print(f"ingest: {exc}", file=sys.stderr)
            return 2
        print(f"ingest: {len(result.copied)} new, {len(result.skipped)} already present "
              f"({result.copied_bytes / 1e6:.1f} MB copied)")
        paths = result.copied

    if not paths:
        print("nothing new to process")
        return 0

    model = DEFAULT_MODEL if args.no_diarize else DIARIZE_MODEL
    gate = GateConfig()
    failures = 0
    all_text: list[str] = []
    queued_ids: list[str] = []
    meta_files: list[str] = []
    meta_minutes = 0.0
    meta_speaker_max = 0
    first_recorded = None

    # --- 2. per recording -----------------------------------------------------
    for path in sorted(paths):
        target = transcript_dir / f"{path.stem}.md"
        if target.exists():
            print(f"{path.name}: transcript exists, not re-transcribing")
            # Parse the transcript back into segments and use the SPOKEN TEXT, never
            # the raw markdown. Feeding the raw file to the summariser doubled the
            # input (the by-speaker section repeats the whole conversation in
            # differently-mangled form) and its artefacts leaked into a real summary
            # as a person called "C" and a tool called "COD/CLAW".
            from extract_actions import parse_transcript

            cached_segments = parse_transcript(target)
            if not cached_segments:
                print("  transcript holds no speech; skipping")
                continue
            cached = "\n\n".join(s.text for s in cached_segments)

            # Still classify it. Skipping the check on the cached path would let an
            # irrelevant recording into the summary on every subsequent run purely
            # because it had been transcribed once -- the expensive stage is skipped,
            # but the cheap safety check must not be.
            if not args.no_relevance:
                try:
                    cached_verdict = classify(cached)
                    if not cached_verdict.keep:
                        print(f"  excluded from the summary: {cached_verdict}")
                        continue
                except RelevanceError as exc:
                    print(f"  relevance check failed, keeping anyway: {exc}",
                          file=sys.stderr)
            all_text.append(cached)
            meta_files.append(path.name)
            cached_stamp = SummaryMeta.from_filename(path.name).recorded_at
            if cached_stamp and (first_recorded is None or cached_stamp < first_recorded):
                first_recorded = cached_stamp

            # Extract on the cached path too. Extraction is seconds and cents, and the
            # queue's quote-based dedupe makes re-extraction idempotent -- while
            # SKIPPING it here twice produced an email with an empty to-do list while
            # real items existed (once after a crash, once after a queue rebuild).
            # Only transcription is expensive enough to deserve a cache.
            if not args.no_extract:
                queued_ids += extract_into_queue(cached_segments, path, args.queue)
            continue

        print(f"\n{path.name}")
        if args.dry_run:
            from autowork.gate import measure, segments

            windows = measure(path, gate)
            keep = segments(windows, gate)
            kept = sum(s.duration_sec for s in keep)
            total = len(windows) * gate.window_sec
            print(f"  gate: would upload {kept / 60:.1f} of {total / 60:.1f} min "
                  f"({100 * kept / total if total else 0:.0f}%) in {len(keep)} segment(s)")
            continue

        try:
            segs = transcribe_file_cloud(
                path, model=model, gate=gate, glossary=glossary
            )
        except CloudTranscriptionError as exc:
            print(f"  transcription failed: {exc}", file=sys.stderr)
            failures += 1
            continue

        if not segs:
            print("  gate rejected everything; nothing uploaded, nothing to transcribe")
            continue

        # Relevance gate, PER SEGMENT. The audio gate proved there was speech-shaped
        # signal; this asks whether each stretch is a work conversation. Per segment
        # rather than per recording because both mixtures were real: a genuine meeting
        # with an hour of podcast appended (one verdict would drop the meeting or admit
        # the podcast), and ambient noise transcribed to plausible fragments. The full
        # transcript is still written either way -- it is already paid for, and a
        # dropped classification should be reviewable.
        kept_segs = list(segs)
        if not args.no_relevance:
            kept_segs = []
            for seg in segs:
                try:
                    verdict = classify(seg.text)
                except RelevanceError as exc:
                    # An unusable classifier must not silently discard the day.
                    print(f"  relevance check failed, keeping anyway: {exc}",
                          file=sys.stderr)
                    kept_segs.append(seg)
                    continue
                if verdict.keep:
                    kept_segs.append(seg)
                else:
                    print(f"  dropped {seg.start_sec:.0f}s-{seg.end_sec:.0f}s: {verdict}")
        joined_raw = "\n\n".join(s.text for s in kept_segs)

        target.write_text(render_transcript(path, segs), encoding="utf-8")
        uploaded = sum(s.uploaded_bytes for s in segs) / 1e6
        audio_min = sum(s.duration_sec for s in segs) / 60
        print(f"  transcribed {audio_min:.1f} min ({uploaded:.2f} MB uploaded) "
              f"-> {target}")

        # Everything downstream sees only the segments that survived relevance; the
        # dropped ones stay reviewable in the transcript file, which is already paid for.
        if not joined_raw.strip():
            print("  no work conversation in this recording; excluded from the summary")
            continue

        all_text.append(joined_raw)

        # Metadata reflects what the summary covers, not everything transcribed.
        meta_files.append(path.name)
        meta_minutes += sum(s.duration_sec for s in kept_segs) / 60
        # Speaker labels are PER REQUEST: chunk 1's "A" is not chunk 2's "A", so a
        # union across chunks inflates the count (measured: a two-person call reported
        # 4). The max within any single chunk is the defensible participant count.
        for s in kept_segs:
            distinct = {t["speaker"] for t in s.speakers if t.get("speaker")}
            meta_speaker_max = max(meta_speaker_max, len(distinct))
        stamp_from_name = SummaryMeta.from_filename(path.name).recorded_at
        if stamp_from_name and (first_recorded is None or stamp_from_name < first_recorded):
            first_recorded = stamp_from_name

        # --- 3. extract into the review queue --------------------------------
        if not args.no_extract:
            queued_ids += extract_into_queue(kept_segs, path, args.queue)

    if args.dry_run:
        print("\ndry run: nothing uploaded, nothing queued, no email sent")
        return 0

    # --- 4. summarise the day -------------------------------------------------
    combined = "\n\n".join(t for t in all_text if t.strip())
    if not combined.strip():
        print("\nno transcript text; no summary, no email")
        return 1 if failures else 0

    try:
        summary = summarise(combined, recorded_at=first_recorded)
    except SummaryError as exc:
        print(f"\nsummary failed: {exc}", file=sys.stderr)
        return 1

    # Factual header: every value derived from files and the diarizer, none generated.
    summary.meta = SummaryMeta(
        recorded_at=first_recorded,
        source_files=meta_files,
        audio_minutes=meta_minutes,
        speaker_count=meta_speaker_max,
    )

    # The email carries EVERY action still awaiting review, not only the ones this run
    # queued. Measured failure: a crashed run queued three items, the re-run used the
    # cached transcripts and so extracted nothing, and the email went out with no to-do
    # list while the queue silently held all three. The queue is the single source of
    # truth; the email is a view of it. Items you have already approved or rejected do
    # not reappear.
    with ReviewQueue(args.queue) as queue:
        actions = queue.list(status=Status.PENDING)

    print(f"\nsummary: {summary.headline}")
    print(f"  {summary.backend}, {summary.elapsed_sec:.1f}s, "
          f"{summary.input_tokens} in / {summary.output_tokens} out")

    stamp = datetime.now().strftime("%Y-%m-%d")
    summary_path = transcript_dir.parent / "summaries" / f"{stamp}.md"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(summary.to_markdown(actions=actions), encoding="utf-8")
    print(f"  saved {summary_path}")

    # --- 5. email -------------------------------------------------------------
    if args.no_email:
        print("  --no-email: not sending")
        return 1 if failures else 0
    recipient = recipient or default_recipient()
    if not recipient:
        print("  no recipient (set SUMMARY_TO in .env); not sending", file=sys.stderr)
        return 1 if failures else 0

    body = summary.to_text(actions=actions)
    subject = f"Work summary {summary.meta.date_only}: {summary.headline[:60]}"
    try:
        print("  " + send(recipient, subject, body,
                          html_body=summary.to_html(actions=actions)))
    except MailError as exc:
        print(f"  email failed: {exc}", file=sys.stderr)
        return 1

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
