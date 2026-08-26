#!/usr/bin/env python3
"""Turn transcripts into a short digest of the passages where someone committed to something.

Usage:
    python tools/digest.py transcripts/R2026-08-25-13-23-54.md
    python tools/digest.py transcripts/*.md --out digests/
    python tools/digest.py transcripts/*.md --context 2      # wider passages
    python tools/digest.py transcripts/*.md --stdout         # print, write nothing

No model is involved. This is deterministic pattern matching over the transcript, which
means: it runs in milliseconds, it cannot fabricate anything, and it stays entirely on
this machine. Measured on a real 23-minute conversation it reduced 19,350 characters to
4,598 -- about a two-minute read -- while retaining all three genuine action items.

WHY THIS EXISTS INSTEAD OF LOCAL LLM EXTRACTION: measured on this hardware, gemma3:4b
took 358s on a 2,600-character excerpt and found 1 of 3 real action items; qwen2.5:7b
took 493s and found none; phi4 never finished. A digest you read in two minutes beats
waiting eleven for a model to find a third of the items badly.

WHAT IT DOES NOT DO: it does not decide what the action is, who owns it, or where it
should go. It selects passages worth your attention and shows you when they were said.
You do the judging. That is a feature at this stage -- there is no fabrication surface
at all, because nothing is generated.

Configurable rather than hardcoded:
    AUTOWORK_DIGESTS   output directory (default: ./digests)

Exit codes: 0 success · 1 one or more transcripts failed · 2 bad arguments
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from autowork.prefilter import CONTEXT_SENTENCES, select  # noqa: E402

DEFAULT_OUT = PROJECT_ROOT / "digests"

_OFFSET = re.compile(r"^- Source offset: ([\d.]+)s")
_QUALITY = re.compile(r"^- Quality: \*\*(\w+)\*\*, ([-+][\d.]+) dB")

CUE_LABELS = {
    "first_person_commitment": "you committed",
    "request_of_someone": "someone asked",
    "shared_or_assigned": "we/you should",
    "follow_up": "follow-up",
    "deadline": "date mentioned",
}


def recording_start(source_name: str) -> datetime | None:
    """Parse the recorder's filename timestamp. None rather than a guess if it fails."""
    try:
        return datetime.strptime(Path(source_name).stem[1:20], "%Y-%m-%d-%H-%M-%S")
    except (ValueError, IndexError):
        return None


def parse_segments(path: Path) -> list[dict]:
    """Pull the transcript body and its offsets out of the markdown we wrote earlier."""
    source = path.stem
    segments: list[dict] = []
    current: dict | None = None
    body: list[str] = []

    def flush() -> None:
        if current is not None and (text := "\n".join(body).strip()):
            if not text.startswith("_("):
                current["text"] = text
                segments.append(current)

    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("# Transcript"):
            source = line.split("—")[-1].strip() or path.stem
        elif line.startswith("## ") and " – " in line:
            flush()
            body = []
            current = {"source": source, "start": 0.0, "quality": "", "db": 0.0}
        elif line.startswith("---"):
            flush()
            current = None
            body = []
        elif current is not None:
            if match := _OFFSET.match(line):
                current["start"] = float(match.group(1))
            elif match := _QUALITY.match(line):
                current["quality"], current["db"] = match.group(1), float(match.group(2))
            elif not line.startswith("- "):
                body.append(line)

    flush()
    return segments


def clock(start: datetime | None, offset_sec: float) -> str:
    if start is None:
        return f"+{int(offset_sec) // 60}m"
    return (start + timedelta(seconds=offset_sec)).strftime("%H:%M")


def render(path: Path, segments: list[dict], context: int) -> tuple[str, int, int, int]:
    """Build the digest. Returns (markdown, passages, original chars, kept chars)."""
    lines = [f"# Digest — {path.stem}", ""]
    total_original = total_kept = total_passages = 0
    blocks: list[str] = []

    for segment in segments:
        start = recording_start(str(segment["source"]))
        result = select(segment["text"], context)
        total_original += result.original_chars
        total_kept += result.selected_chars
        total_passages += len(result.passages)

        if not result.passages:
            continue

        # Sentence position within the segment maps to elapsed time only
        # approximately -- speech rate varies -- so this locates a passage for
        # listening, it does not timestamp it precisely.
        per_sentence = (
            segment["text"].count(".") and len(segment["text"]) / max(result.total_sentences, 1)
        )
        for passage in result.passages:
            approx_offset = float(segment["start"]) + (
                passage.first_sentence * per_sentence / 14.0 if per_sentence else 0
            )
            labels = ", ".join(CUE_LABELS.get(c, c) for c in passage.cues)
            blocks.append(
                f"### ~{clock(start, approx_offset)}  ·  {labels}\n\n{passage.text}\n"
            )

    if not blocks:
        lines += [
            "No commitment language found in this recording.",
            "",
            "That is a real answer, not a failure: most conversation is discussion.",
            "The full transcript is retained if you want to read it directly.",
            "",
        ]
    else:
        reduction = 100 * (1 - total_kept / total_original) if total_original else 0
        lines += [
            f"{total_passages} passage(s) worth a look, "
            f"{total_kept:,} of {total_original:,} characters "
            f"({reduction:.0f}% of the transcript skipped).",
            "",
            "Selected by commitment language, not by a model: nothing here is generated,",
            "summarised, or inferred. These are the speaker's own words. Passages a",
            "pattern did not match are not shown, so read the full transcript if",
            "something you expected is missing.",
            "",
            "---",
            "",
            *blocks,
        ]

    return "\n".join(lines), total_passages, total_original, total_kept


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("transcripts", nargs="+")
    parser.add_argument("--out", default=os.environ.get("AUTOWORK_DIGESTS", DEFAULT_OUT))
    parser.add_argument("--context", type=int, default=CONTEXT_SENTENCES,
                        help="sentences of context each side of a match")
    parser.add_argument("--stdout", action="store_true", help="print instead of writing")
    args = parser.parse_args(argv)

    if args.context < 0:
        print("--context must be >= 0", file=sys.stderr)
        return 2

    paths = [Path(p) for p in args.transcripts]
    if missing := [p for p in paths if not p.is_file()]:
        for path in missing:
            print(f"no such transcript: {path}", file=sys.stderr)
        return 2

    out_dir = Path(args.out)
    if not args.stdout:
        out_dir.mkdir(parents=True, exist_ok=True)

    failures = 0
    for path in paths:
        segments = parse_segments(path)
        if not segments:
            print(f"{path.name}: no transcript segments found", file=sys.stderr)
            failures += 1
            continue

        markdown, passages, original, kept = render(path, segments, args.context)
        if args.stdout:
            print(markdown)
        else:
            target = out_dir / f"{path.stem}-digest.md"
            target.write_text(markdown, encoding="utf-8")
            saved = 100 * (1 - kept / original) if original else 0
            print(f"{path.name}: {passages} passage(s), "
                  f"{kept:,}/{original:,} chars ({saved:.0f}% skipped) -> {target}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
