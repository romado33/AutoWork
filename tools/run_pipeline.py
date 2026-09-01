#!/usr/bin/env python3
"""End to end: ingest from the recorder, transcribe, summarise, extract, queue, email.

Usage:
    python tools/run_pipeline.py                      # full run, ingest from the device
    python tools/run_pipeline.py --files audio/*.MP3   # skip ingest, use these files
    python tools/run_pipeline.py --no-email            # everything but the send
    python tools/run_pipeline.py --no-diarize          # faster, no speaker labels
    python tools/run_pipeline.py --dry-run             # gate only, upload nothing
    python tools/run_pipeline.py --resend              # email even if already sent

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

from autowork.extract import (  # noqa: E402
    ExtractionError,
    ExtractorConfig,
    extract_from_segment,
    quotes_overlap,
    source_key,
    titles_overlap,
)
from autowork.gate import GateConfig, Verdict  # noqa: E402
from autowork.glossary import Glossary, GlossaryError  # noqa: E402
from autowork.ingest import IngestError, ingest  # noqa: E402
from autowork.llm import load_dotenv  # noqa: E402
from autowork.mailer import MailError, default_recipient, send  # noqa: E402
from autowork.prefilter import select  # noqa: E402
from autowork.quality import assess, retain_intelligible, speaker_count_for_header  # noqa: E402
from autowork.action import Status  # noqa: E402
from autowork.queue import QueueError, ReviewQueue  # noqa: E402
from autowork.relevance import RelevanceError, classify  # noqa: E402
from autowork.summarize import SummaryError, SummaryMeta, summarise  # noqa: E402
from autowork.transcribe import TranscribedSegment  # noqa: E402
from autowork.transcribe_cloud import (  # noqa: E402
    DEFAULT_MODEL,
    DIARIZE_MODEL,
    CloudTranscriptionError,
    transcribe_file_cloud,
)
from autowork.day import (  # noqa: E402
    Contribution,
    DayError,
    clarify_markdown,
    date_from_filename,
    isoformat,
    load_kept_for_date,
    save_contribution,
)
from autowork.conversations import Conversation, group_conversations  # noqa: E402
from autowork.calendar_lookup import match_recording  # noqa: E402
from autowork.digest import apply_glossary_all  # noqa: E402
from autowork.sent import already_sent, conversation_key, mark_sent  # noqa: E402

DEFAULT_SERIAL = "AA986EA1"

# Operator --force-keep only. Does not change the default work/intelligibility gates.
FORCE_KEEP_PROMPT = """\
OPERATOR OVERRIDE (one-time, this recording only):
This is a personal conversation the operator chose to keep. Summarise what was
actually said as a personal catch-up, not as a work meeting. Do not invent
workstreams, product launches, satellite economics, or to-dos from garbled
fragments or background chatter. Omit unintelligible stretches. Empty topics
and an honest note are better than a fake meeting. Ignore the usual rule that
non-work content should be discarded — the operator already made that call.
"""


def render_transcript(path: Path, segs: list, glossary: Glossary | None = None) -> str:
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
        if getattr(seg, "unintelligible_because", ""):
            lines.append(f"- unintelligible: {seg.unintelligible_because}")
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

    if glossary is not None:
        combined = "\n\n".join(s.text for s in segs if getattr(s, "text", "").strip())
        unknown = glossary.unknown_proper_nouns(combined)
        footer = clarify_markdown(unknown)
        if footer:
            lines.append(footer.rstrip("\n"))
            lines.append("")
    return "\n".join(lines)


def speaker_max(segments: list) -> int:
    """Max believable distinct speakers in any one segment.

    Labels are per request, not a union. Implausible counts (13 on a 2-person
    recording, 2026-08-27) return 0 so the email omits the Participants line
    rather than printing a fake headcount.
    """
    return speaker_count_for_header(segments)


def note_contribution(
    transcript_dir: Path,
    path: Path,
    *,
    keep: bool,
    text: str,
    audio_minutes: float,
    speaker_count: int,
    excluded_because: str = "",
    forced: bool = False,
) -> str | None:
    """Persist this recording's day-summary contribution. Returns the date key."""
    stamp = SummaryMeta.from_filename(path.name).recorded_at
    save_contribution(
        transcript_dir,
        Contribution(
            source_file=path.name,
            keep=keep,
            text=text if keep else "",
            audio_minutes=audio_minutes if keep else 0.0,
            speaker_count=speaker_count if keep else 0,
            recorded_at=isoformat(stamp),
            excluded_because=excluded_because,
            forced=forced and keep,
        ),
    )
    return date_from_filename(path.name)


def keep_relevant(segments: list) -> tuple[list, list[tuple[object, object]]]:
    """Relevance gate, PER SEGMENT. Returns (kept, [(segment, verdict), ...]).

    One helper for both paths. The cached path used to classify the JOINED text
    instead, so a recording with one work segment beside one personal segment was
    keepable on first transcribe and droppable on re-run -- the verdict depended
    on which branch happened to run, not on what was said.

    A classifier that cannot answer keeps the segment: an unusable classifier must
    not be able to silently discard a real day.
    """
    kept: list = []
    dropped: list[tuple[object, object]] = []
    for seg in segments:
        try:
            verdict = classify(seg.text)
        except RelevanceError as exc:
            print(f"  relevance check failed, keeping anyway: {exc}", file=sys.stderr)
            kept.append(seg)
            continue
        if verdict.keep:
            kept.append(seg)
        else:
            dropped.append((seg, verdict))
    return kept, dropped


def extract_into_queue(segments: list, path: Path, queue_path: str) -> list[str]:
    """Prefilter, extract grounded actions, and file them. Returns queued ids.

    One helper for both the fresh-transcription and cached-transcript paths, because
    the two drifted twice and each drift produced an email with an empty to-do list
    while the queue held real items. Accepts CloudSegment or TranscribedSegment; only
    the fields both carry are used.
    """
    segments = [
        s for s in segments
        if not (getattr(s, "unintelligible_because", "") or "").strip()
    ]
    if not segments:
        print("  no intelligible segments; nothing queued")
        return []
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
        existing = [
            a
            for a in queue.list(status=Status.PENDING)
            if source_key(a) == path.name.lower()
        ]
        before = queue.counts().get("pending", 0)
        for record in accepted:
            quote = record.provenance.transcript_excerpt
            if any(quotes_overlap(quote, a.provenance.transcript_excerpt) for a in existing):
                print(f"    = already pending (overlapping quote): {record.title[:60]}")
                continue
            if any(titles_overlap(record.title, a.title) for a in existing):
                print(f"    = already pending (same to-do): {record.title[:60]}")
                continue
            queued.append(queue.enqueue(record))
            existing.append(record)
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
    parser.add_argument(
        "--force-keep",
        action="store_true",
        help="one-off operator override: summarise even if personal or garbled",
    )
    parser.add_argument(
        "--resend",
        action="store_true",
        help="email even if this conversation was sent already",
    )
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
        paths = list(result.copied)

        # Crash recovery: also pick up files recorded TODAY that have no transcript.
        # Measured gap: a run ingested a recording and then died; the ledger correctly
        # skipped the copy on the next run, so "copied this run" was empty and the
        # recording sat untranscribed forever. Restricted to today's date (from the
        # filename) so a fresh setup can never bulk-bill the months of old audio the
        # recorder keeps; older gaps are recovered explicitly with --files.
        today = datetime.now().strftime("%Y-%m-%d")
        for candidate in sorted(audio_dir.glob("*.MP3")) + sorted(audio_dir.glob("*.mp3")):
            stem = candidate.stem
            if (
                len(stem) >= 11
                and stem[1:11] == today
                and not (transcript_dir / f"{stem}.md").exists()
                and candidate not in paths
            ):
                print(f"recovering unprocessed recording from today: {candidate.name}")
                paths.append(candidate)

    if not paths:
        print("nothing new to process")
        return 0

    processed_names = {p.name for p in paths}

    model = DEFAULT_MODEL if args.no_diarize else DIARIZE_MODEL
    gate = GateConfig()
    failures = 0
    dates_touched: set[str] = set()

    def touch(date_key: str | None) -> None:
        if date_key:
            dates_touched.add(date_key)

    from concurrent.futures import ThreadPoolExecutor

    # One worker: SQLite does not enjoy two extractors writing at once. The overlap
    # we want is extract(file N) with transcribe(file N+1), which this gives us.
    extract_pool = ThreadPoolExecutor(max_workers=1)
    extract_futures: list = []

    def schedule_extract(segments: list, path: Path) -> None:
        extract_futures.append(
            extract_pool.submit(extract_into_queue, segments, path, args.queue)
        )

    # --- 2. per recording -----------------------------------------------------
    def process_recording(path: Path) -> None:
        """One recording: transcribe or reuse, gate, note the contribution, queue.

        Nested so it keeps this run's configuration in scope, and called inside a
        try/except so one locked transcript or busy queue skips that recording
        rather than abandoning every recording after it.
        """
        nonlocal failures
        target = transcript_dir / f"{path.stem}.md"

        if args.dry_run:
            # Checked BEFORE the cached branch. Sitting after it meant a dry run
            # still rewrote glossaries, wrote sidecars and queued actions for every
            # file that already had a transcript, which is not a dry run.
            print(f"\n{path.name}")
            if target.exists():
                print("  transcript exists; nothing would be uploaded")
                return
            from autowork.gate import measure, segments

            windows = measure(path, gate)
            keep = segments(windows, gate)
            kept = sum(s.duration_sec for s in keep)
            total = len(windows) * gate.window_sec
            print(f"  gate: would upload {kept / 60:.1f} of {total / 60:.1f} min "
                  f"({100 * kept / total if total else 0:.0f}%) in {len(keep)} segment(s)")
            return

        if target.exists():
            print(f"{path.name}: transcript exists, not re-transcribing")
            # Variants added after the first transcribe (Dave Cazal, OctoAdmin) live
            # in the glossary but were never applied to this file. Re-run the same
            # correct() pass the live transcriber uses; it is free and idempotent.
            applied = glossary.rewrite_file(target)
            if applied:
                print(f"  glossary corrected: {', '.join(applied)}")
            # Parse the transcript back into segments and use the SPOKEN TEXT, never
            # the raw markdown. Feeding the raw file to the summariser doubled the
            # input (the by-speaker section repeats the whole conversation in
            # differently-mangled form) and its artefacts leaked into a real summary
            # as a person called "C" and a tool called "COD/CLAW".
            from extract_actions import parse_transcript

            cached_segments = parse_transcript(target)
            if not cached_segments:
                print("  transcript holds no speech; skipping")
                touch(note_contribution(
                    transcript_dir, path, keep=False, text="",
                    audio_minutes=0, speaker_count=0,
                    excluded_because="transcript holds no speech",
                ))
                return
            if args.force_keep:
                print("  --force-keep: not dropping personal/unintelligible segments")
                cleaned_segs = []
                for seg in cached_segments:
                    cleaned, _ = assess(seg.text, getattr(seg, "speakers", None) or [])
                    if cleaned.strip():
                        seg.text = cleaned
                        cleaned_segs.append(seg)
                cached_segments = cleaned_segs
                if not cached_segments:
                    print("  excluded from the summary: nothing left after cleaning loops")
                    touch(note_contribution(
                        transcript_dir, path, keep=False, text="",
                        audio_minutes=0, speaker_count=0,
                        excluded_because="nothing left after cleaning loops",
                    ))
                    return
            else:
                cached_segments, dropped_q = retain_intelligible(cached_segments)
                for seg, reason in dropped_q:
                    print(
                        f"  dropped {seg.start_sec:.0f}s-{seg.end_sec:.0f}s: "
                        f"unintelligible ({reason})"
                    )
                if not cached_segments:
                    print("  excluded from the summary: no intelligible conversation")
                    touch(note_contribution(
                        transcript_dir, path, keep=False, text="",
                        audio_minutes=0, speaker_count=0,
                        excluded_because="no intelligible conversation",
                    ))
                    return

            # Still classify it. Skipping the check on the cached path would let an
            # irrelevant recording into the summary on every subsequent run purely
            # because it had been transcribed once -- the expensive stage is skipped,
            # but the cheap safety check must not be. Per segment, exactly as the
            # fresh path does it.
            if not args.no_relevance and not args.force_keep:
                cached_segments, dropped_rel = keep_relevant(cached_segments)
                for seg, verdict in dropped_rel:
                    print(f"  dropped {seg.start_sec:.0f}s-{seg.end_sec:.0f}s: {verdict}")
                if not cached_segments:
                    print("  excluded from the summary: no work conversation")
                    touch(note_contribution(
                        transcript_dir, path, keep=False, text="",
                        audio_minutes=0, speaker_count=0,
                        excluded_because="no work conversation",
                    ))
                    return

            cached = "\n\n".join(s.text for s in cached_segments if s.text.strip())
            touch(note_contribution(
                transcript_dir, path, keep=True, text=cached,
                audio_minutes=sum(s.duration_sec for s in cached_segments) / 60,
                speaker_count=speaker_max(cached_segments),
                forced=args.force_keep,
            ))

            # Extract on the cached path too. Extraction is seconds and cents, and the
            # queue's quote-based dedupe makes re-extraction idempotent -- while
            # SKIPPING it here twice produced an email with an empty to-do list while
            # real items existed (once after a crash, once after a queue rebuild).
            # Only transcription is expensive enough to deserve a cache.
            if not args.no_extract and not args.force_keep:
                schedule_extract(cached_segments, path)
            elif args.force_keep:
                print("  --force-keep: not extracting action items")
            return

        print(f"\n{path.name}")
        try:
            segs = transcribe_file_cloud(
                path, model=model, gate=gate, glossary=glossary
            )
        except CloudTranscriptionError as exc:
            print(f"  transcription failed: {exc}", file=sys.stderr)
            failures += 1
            return

        if not segs:
            print("  gate rejected everything; nothing uploaded, nothing to transcribe")
            touch(note_contribution(
                transcript_dir, path, keep=False, text="",
                audio_minutes=0, speaker_count=0,
                excluded_because="gate rejected everything",
            ))
            return

        if args.force_keep:
            print("  --force-keep: not dropping personal/unintelligible segments")
            for seg in segs:
                cleaned, _ = assess(seg.text, getattr(seg, "speakers", None) or [])
                if cleaned.strip():
                    seg.text = cleaned
            intelligible = [s for s in segs if s.text.strip()]
            dropped_q = []
        else:
            # Cheap intelligibility before the LLM relevance call. The 2026-08-27 11:30
            # recording measured CLEAN at +4.9 dB and the classifier kept it "despite
            # garbling" because work words appeared; 13 diarizer labels and a 59-copy
            # Whisper loop are not a meeting. Dropped segments stay in the transcript.
            intelligible, dropped_q = retain_intelligible(segs)
        for seg, reason in dropped_q:
            print(
                f"  dropped {seg.start_sec:.0f}s-{seg.end_sec:.0f}s: "
                f"unintelligible ({reason})"
            )

        # Relevance gate, PER SEGMENT, on what survived the cheap check.
        kept_segs = list(intelligible)
        if not args.no_relevance and not args.force_keep:
            kept_segs, dropped_rel = keep_relevant(intelligible)
            for seg, verdict in dropped_rel:
                print(f"  dropped {seg.start_sec:.0f}s-{seg.end_sec:.0f}s: {verdict}")
        joined_raw = "\n\n".join(s.text for s in kept_segs)

        target.write_text(render_transcript(path, segs, glossary), encoding="utf-8")
        uploaded = sum(s.uploaded_bytes for s in segs) / 1e6
        audio_min = sum(s.duration_sec for s in segs) / 60
        print(f"  transcribed {audio_min:.1f} min ({uploaded:.2f} MB uploaded) "
              f"-> {target}")

        # Everything downstream sees only the segments that survived relevance; the
        # dropped ones stay reviewable in the transcript file, which is already paid for.
        if not joined_raw.strip():
            print("  no intelligible work conversation; excluded from the summary")
            touch(note_contribution(
                transcript_dir, path, keep=False, text="",
                audio_minutes=0, speaker_count=0,
                excluded_because="no intelligible work conversation",
            ))
            return

        touch(note_contribution(
            transcript_dir, path, keep=True, text=joined_raw,
            audio_minutes=sum(s.duration_sec for s in kept_segs) / 60,
            speaker_count=speaker_max(kept_segs),
            forced=args.force_keep,
        ))

        # --- 3. extract into the review queue --------------------------------
        if not args.no_extract and not args.force_keep:
            schedule_extract(kept_segs, path)
        elif args.force_keep:
            print("  --force-keep: not extracting action items")

    try:
        for path in sorted(paths):
            try:
                process_recording(path)
            except (OSError, QueueError, DayError, GlossaryError) as exc:
                # A locked transcript, a full disk or a busy SQLite file is one bad
                # recording, not a reason to abandon the rest of the day's audio.
                print(f"{path.name}: skipped after an unexpected failure: {exc}",
                      file=sys.stderr)
                failures += 1
    finally:
        extract_pool.shutdown(wait=True)
    for fut in extract_futures:
        try:
            fut.result()
        except (OSError, QueueError, DayError) as exc:
            print(f"extraction skipped after an unexpected failure: {exc}",
                  file=sys.stderr)
            failures += 1

    if args.dry_run:
        print("\ndry run: nothing uploaded, nothing queued, no email sent")
        return 0

    # --- 4. one summary email per conversation --------------------------------
    # A conversation is one recording, or several that overlap the SAME Outlook
    # event. Same-sounding topics are not merged: that mixed a 25 Aug mapping
    # call into a 26 Aug Okta email. Outstanding items from other days belong
    # on the morning digest, not here.
    if not dates_touched:
        print("\nno transcript text; no summary, no email")
        return 1 if failures else 0

    with ReviewQueue(args.queue) as queue:
        pending = queue.list(status=Status.PENDING)

    recipient = recipient or default_recipient()
    summarised = 0
    all_contribs = []
    for date_key in sorted(dates_touched):
        # Only the run that forced a recording through may see its sidecar. Later
        # runs rebuild the day from recordings that actually passed the gates.
        all_contribs.extend(
            load_kept_for_date(
                transcript_dir, date_key, include_forced=args.force_keep
            )
        )

    matches: dict = {}
    for contrib in all_contribs:
        if contrib.recorded_datetime is None:
            matches[contrib.source_file] = None
            continue
        matched = match_recording(contrib.recorded_datetime, contrib.audio_minutes)
        matches[contrib.source_file] = matched
        if matched is not None:
            print(f"  calendar {contrib.source_file}: {matched.subject or '(no title)'} "
                  f"({len(matched.invitees)} invitees)")

    convos = group_conversations(all_contribs, matches)
    if args.force_keep and args.files:
        forced = [
            c for c in all_contribs
            if c.keep and c.source_file in processed_names
        ]
        if forced:
            convos = [Conversation(contribs=forced, group_key="force-keep")]

    for convo in convos:
        if not any(name in processed_names for name in convo.source_files):
            continue
        combined = convo.text
        if not combined.strip():
            continue
        try:
            extra = FORCE_KEEP_PROMPT if args.force_keep else ""
            summary = summarise(
                combined,
                recorded_at=convo.meta.recorded_at,
                glossary=glossary,
                require_intelligible=not args.force_keep,
                extra_instructions=extra,
            )
        except SummaryError as exc:
            print(f"\n{convo.source_files}: summary failed: {exc}", file=sys.stderr)
            failures += 1
            continue

        summary.meta = convo.meta
        summary.clarify_terms = glossary.unknown_proper_nouns(
            combined, min_occurrences=2
        )
        summary.apply_glossary(glossary)
        if args.force_keep:
            convo_actions = []
        else:
            convo_actions = apply_glossary_all(
                [
                    a for a in pending
                    if Path(a.provenance.source_audio).name in set(convo.source_files)
                ],
                glossary,
            )

        stamp = summary.meta.date_only
        time_bit = (
            summary.meta.recorded_at.strftime("%H-%M")
            if summary.meta.recorded_at else "unknown"
        )
        print(f"\nsummary {stamp} {time_bit}: {summary.headline}")
        print(f"  {summary.backend}, {summary.elapsed_sec:.1f}s, "
              f"{summary.input_tokens} in / {summary.output_tokens} out")
        print(f"  files: {', '.join(convo.source_files)}")

        summary_path = (
            transcript_dir.parent / "summaries" / f"{stamp}-{time_bit}.md"
        )
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            summary.to_markdown(actions=convo_actions), encoding="utf-8"
        )
        print(f"  saved {summary_path}")
        summarised += 1

        if (
            not args.force_keep
            and not summary.topics
            and not summary.decisions
            and not convo_actions
        ):
            print("  garbled or empty: not emailing a fake meeting")
            continue

        if args.no_email:
            print("  --no-email: not sending")
            continue
        if not recipient:
            print("  no recipient (set SUMMARY_TO in .env); not sending", file=sys.stderr)
            failures += 1
            continue

        if summary.meta.meeting_title:
            subject = (
                f"Meeting {stamp}: {summary.meta.meeting_title[:70]}"
            )
        else:
            subject = f"Call {stamp} {time_bit.replace('-', ':')}: {summary.headline[:50]}"
        sent_dir = transcript_dir.parent / "summaries"
        key = conversation_key(convo.source_files)
        if not args.resend and already_sent(sent_dir, key):
            print("  already emailed this conversation; pass --resend to send again")
            continue
        body = summary.to_text(actions=convo_actions)
        try:
            print("  " + send(recipient, subject, body,
                              html_body=summary.to_html(actions=convo_actions)))
        except MailError as exc:
            print(f"  email failed: {exc}", file=sys.stderr)
            failures += 1
            continue
        mark_sent(sent_dir, key, subject)

    if summarised == 0 and failures == 0:
        return 0
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
