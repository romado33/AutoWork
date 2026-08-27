#!/usr/bin/env python3
"""Read transcripts, extract grounded action items, and file them in the review queue.

Usage:
    python tools/extract_actions.py transcripts/R2026-08-25-13-23-54.md
    python tools/extract_actions.py transcripts/*.md --dry-run    # show, do not file

Extraction uses the configured cloud backend (see autowork/extract.py), never a
local model. Only prefiltered commitment-bearing passages are sent.

Configurable rather than hardcoded:
    AUTOWORK_QUEUE           queue database    (default: ./queue.sqlite3)

Everything filed lands as PENDING. Nothing is executed, and nothing is approved, by
this tool. Re-running on the same transcript is idempotent: the queue dedupes on source
audio, window and title, and will not reset a review decision you have already made.

Exit codes: 0 success · 1 one or more transcripts failed · 2 bad arguments
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from autowork.extract import (  # noqa: E402
    Candidate,
    ExtractionError,
    ExtractorConfig,
    extract_from_segment,
)
from autowork.gate import Verdict  # noqa: E402
from autowork.queue import ReviewQueue  # noqa: E402
from autowork.transcribe import TranscribedSegment  # noqa: E402

DEFAULT_QUEUE = PROJECT_ROOT / "queue.sqlite3"

_HEADING = re.compile(r"^##\s+\S+\s+.\s+\S+\s+\((\d+)s\)")
_QUALITY = re.compile(r"^- Quality: \*\*(\w+)\*\*, ([-+][\d.]+) dB")
_OFFSET = re.compile(r"^- Source offset: ([\d.]+)s\s*.\s*([\d.]+)s")
_CORRECTED = re.compile(r"corrected: ([^·\n]+)")
_SPEAKERS = re.compile(r"^- speakers:\s*(.+)$")
_UNINTELLIGIBLE = re.compile(r"^- unintelligible:\s*(.+)$")


def parse_transcript(path: Path) -> list[TranscribedSegment]:
    """Reconstruct segments from the markdown the transcriber wrote.

    Parsing our own output rather than re-deriving it keeps the two stages decoupled:
    you can hand-edit a transcript to fix a mis-transcribed name before extraction, and
    the correction flows through to the action items.
    """
    source = "unknown"
    for line in path.read_text(encoding="utf-8").splitlines()[:5]:
        if line.startswith("# Transcript"):
            source = line.split("—")[-1].strip() or path.stem

    segments: list[TranscribedSegment] = []
    current: dict[str, object] | None = None
    body: list[str] = []

    def flush() -> None:
        if current is None:
            return
        text = "\n".join(body).strip()
        if not text or text.startswith("_("):
            return
        segments.append(
            TranscribedSegment(
                source_audio=str(current["source"]),
                start_sec=float(current["start"]),
                end_sec=float(current["end"]),
                verdict=Verdict(current["verdict"]),
                speech_rumble_db=float(current["db"]),
                text=text,
                glossary_applied=list(current["glossary"]),  # type: ignore[arg-type]
                speakers=list(current["speakers"]),  # type: ignore[arg-type]
                unintelligible_because=str(current.get("unintelligible") or ""),
            )
        )

    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("## ") and _HEADING.match(line):
            flush()
            body = []
            current = {
                "source": source, "start": 0.0, "end": 0.0,
                "verdict": "usable", "db": 0.0, "glossary": [],
                "speakers": [], "unintelligible": "",
            }
            continue
        if line.startswith("---"):
            flush()
            current = None
            continue
        if current is None:
            continue

        if match := _QUALITY.match(line):
            current["verdict"], current["db"] = match.group(1), float(match.group(2))
        elif match := _OFFSET.match(line):
            current["start"], current["end"] = float(match.group(1)), float(match.group(2))
        elif line.startswith("- ") and (match := _CORRECTED.search(line)):
            current["glossary"] = [t.strip() for t in match.group(1).split(",")]
        elif match := _SPEAKERS.match(line):
            current["speakers"] = [
                {"speaker": label.strip()}
                for label in match.group(1).split(",")
                if label.strip()
            ]
        elif match := _UNINTELLIGIBLE.match(line):
            current["unintelligible"] = match.group(1).strip()
        elif not line.startswith("- "):
            body.append(line)

    flush()
    return segments


def report(accepted: list, rejected: list[Candidate], dry_run: bool) -> None:
    print(f"    {len(accepted)} accepted, {len(rejected)} rejected")
    for record in accepted:
        flag = " [UNVERIFIED AUDIO]" if record.provenance.speech_rumble_db < 1.0 else ""
        print(f"      + {record.title}{flag}")
        print(f"        {record.target_system} / {record.action_type.value} "
              f"/ confidence {record.confidence:.2f}")
        print(f"        \"{record.provenance.transcript_excerpt[:90]}\"")
    # Rejections are printed, not merely logged. A model proposing only garbage must
    # look different from a genuinely quiet day.
    for candidate in rejected:
        print(f"      - {candidate.title[:70]}  ({candidate.rejected_because})")
    if dry_run and accepted:
        print("    (dry run: nothing filed)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("transcripts", nargs="+")
    parser.add_argument("--model", default=os.environ.get("AUTOWORK_MODEL_EXTRACT", "gemma3:4b"))
    parser.add_argument("--ollama-url",
                        default=os.environ.get("AUTOWORK_OLLAMA_URL", "http://localhost:11434"))
    parser.add_argument("--queue", default=os.environ.get("AUTOWORK_QUEUE", DEFAULT_QUEUE))
    parser.add_argument("--owner", default="Rob")
    parser.add_argument("--min-confidence", type=float, default=0.3)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(message)s",
    )

    paths = [Path(p) for p in args.transcripts]
    if missing := [p for p in paths if not p.is_file()]:
        for path in missing:
            print(f"no such transcript: {path}", file=sys.stderr)
        return 2

    try:
        config = ExtractorConfig(
            model=args.model,
            ollama_url=args.ollama_url,
            owner_name=args.owner,
            min_confidence=args.min_confidence,
            timeout_sec=args.timeout,
        )
    except ExtractionError as exc:
        print(f"configuration: {exc}", file=sys.stderr)
        return 2

    queue = None if args.dry_run else ReviewQueue(args.queue)
    failures = 0
    filed = 0

    try:
        for path in paths:
            print(f"\n=== {path.name} ===")
            segments = parse_transcript(path)
            if not segments:
                print("    no transcribed segments found")
                continue

            for segment in segments:
                print(f"  {segment.start_sec:.0f}s-{segment.end_sec:.0f}s "
                      f"({segment.verdict.value}, {segment.speech_rumble_db:+.1f} dB, "
                      f"{len(segment.text)} chars)")
                try:
                    accepted, rejected = extract_from_segment(segment, config)
                except ExtractionError as exc:
                    print(f"    extraction failed: {exc}", file=sys.stderr)
                    failures += 1
                    continue

                report(accepted, rejected, args.dry_run)
                if queue is not None:
                    for record in accepted:
                        queue.enqueue(record)
                        filed += 1

        if queue is not None:
            print(f"\nfiled {filed} action(s); queue now: {queue.counts()}")
            print(f"queue database: {args.queue}")
    finally:
        if queue is not None:
            queue.close()

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
