#!/usr/bin/env python3
"""Transcribe a day's recordings: gate, slice, Whisper, glossary correction.

Usage:
    python tools/transcribe_day.py D:/RECORD/R2026-08-25-13-23-54.MP3
    python tools/transcribe_day.py D:/RECORD/*.MP3 --out transcripts/
    python tools/transcribe_day.py day.mp3 --no-glossary          # A/B the glossary

Paths are configurable rather than hardcoded:
    AUTOWORK_SONA    path to sona.exe   (default: %LOCALAPPDATA%/Vibe/sona.exe)
    AUTOWORK_MODEL   path to ggml model (default: the Vibe large-v3-turbo download)
    AUTOWORK_OUT     output directory   (default: ./transcripts)

Transcripts are written as markdown, one file per recording, and are never deleted by
this tool -- retention is a manual decision. Re-running overwrites the file for a given
recording rather than appending, so the tool is idempotent.

Exit codes: 0 success · 1 one or more files failed · 2 bad arguments or missing input
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from autowork.gate import GateConfig, GateError  # noqa: E402
from autowork.glossary import Glossary, GlossaryError  # noqa: E402
from autowork.transcribe import (  # noqa: E402
    TranscribedSegment,
    TranscriberConfig,
    TranscriptionError,
    transcribe_file,
)

LOCALAPPDATA = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData/Local"))
DEFAULT_SONA = LOCALAPPDATA / "Vibe" / "sona.exe"
DEFAULT_MODEL = (
    LOCALAPPDATA / "github.com.thewh1teagle.vibe" / "ggml-large-v3-turbo.bin"
)
DEFAULT_GLOSSARY = PROJECT_ROOT / "config" / "glossary.yml"


def recording_start(path: Path) -> datetime | None:
    """Parse the recorder's filename timestamp, e.g. R2026-08-25-13-23-54.MP3.

    Returns None rather than guessing if the name does not match; a wrong wall-clock
    time on a transcript is worse than no wall-clock time, because it would be used to
    correlate against a calendar.
    """
    stem = path.stem
    if len(stem) < 20:
        return None
    try:
        return datetime.strptime(stem[1:20], "%Y-%m-%d-%H-%M-%S")
    except ValueError:
        return None


def clock(start: datetime | None, offset_sec: float) -> str:
    if start is None:
        return f"+{int(offset_sec) // 60:d}m{int(offset_sec) % 60:02d}s"
    return (start + timedelta(seconds=offset_sec)).strftime("%H:%M:%S")


def render(path: Path, segs: list[TranscribedSegment], glossary: Glossary | None) -> str:
    start = recording_start(path)
    header = [
        f"# Transcript — {path.name}",
        "",
        f"- Recorded: {start.isoformat(sep=' ') if start else 'unknown (filename not parsed)'}",
        f"- Segments kept: {len(segs)}",
        f"- Audio transcribed: {sum(s.duration_sec for s in segs):.0f}s",
        "",
        "Only audio passing the quality gate appears below. Rejected audio is not",
        "transcribed, because Whisper invents fluent text from low-signal input.",
        "",
    ]

    body: list[str] = []
    for seg in segs:
        flags = []
        if seg.prompt_used:
            flags.append("glossary prompt applied")
        if seg.glossary_applied:
            flags.append("corrected: " + ", ".join(seg.glossary_applied))
        if seg.repetition_flagged:
            flags.append("**REPETITION LOOP — treat as unverified**")

        body += [
            f"## {clock(start, seg.start_sec)} – {clock(start, seg.end_sec)}"
            f"  ({seg.duration_sec:.0f}s)",
            "",
            f"- Quality: **{seg.verdict.value}**, {seg.speech_rumble_db:+.1f} dB "
            f"speech-minus-rumble",
            f"- Source offset: {seg.start_sec:.0f}s – {seg.end_sec:.0f}s",
        ]
        if flags:
            body.append(f"- {' · '.join(flags)}")
        body += ["", seg.text or "_(no speech recognised)_", ""]

    footer: list[str] = []
    if glossary:
        combined = "\n".join(s.text for s in segs)
        unknown = glossary.unknown_proper_nouns(combined)
        if unknown:
            footer += [
                "---",
                "",
                "## Terms to clarify",
                "",
                "Capitalised words repeated in this transcript that the glossary does",
                "not know. This is a heuristic candidate list, not a judgement — skim it",
                "and promote the real ones into `config/glossary.yml`.",
                "",
                *(f"- {word}" for word in unknown),
                "",
            ]

    return "\n".join(header + body + footer)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("audio", nargs="+")
    parser.add_argument("--out", default=os.environ.get("AUTOWORK_OUT", "transcripts"))
    parser.add_argument("--sona", default=os.environ.get("AUTOWORK_SONA", DEFAULT_SONA))
    parser.add_argument("--model", default=os.environ.get("AUTOWORK_MODEL", DEFAULT_MODEL))
    parser.add_argument("--glossary", default=DEFAULT_GLOSSARY)
    parser.add_argument("--no-glossary", action="store_true")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--window-sec", type=float, default=GateConfig.window_sec)
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(message)s",
    )

    paths = [Path(a) for a in args.audio]
    missing = [p for p in paths if not p.is_file()]
    if missing:
        for p in missing:
            print(f"no such file: {p}", file=sys.stderr)
        return 2

    glossary: Glossary | None = None
    if not args.no_glossary:
        try:
            glossary = Glossary.load(args.glossary)
        except GlossaryError as exc:
            print(f"glossary: {exc}", file=sys.stderr)
            return 2

    try:
        config = TranscriberConfig(
            sona_path=Path(args.sona),
            model_path=Path(args.model),
            threads=args.threads,
            gate=GateConfig(window_sec=args.window_sec),
        )
    except (TranscriptionError, GateError) as exc:
        print(f"configuration: {exc}", file=sys.stderr)
        return 2

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    failures = 0
    for path in paths:
        try:
            segs = transcribe_file(path, config, glossary)
        except (TranscriptionError, GateError) as exc:
            print(f"error on {path.name}: {exc}", file=sys.stderr)
            failures += 1
            continue

        target = out_dir / f"{path.stem}.md"
        target.write_text(render(path, segs, glossary), encoding="utf-8")

        kept = sum(s.duration_sec for s in segs)
        flagged = sum(1 for s in segs if s.repetition_flagged)
        note = f"  ({flagged} flagged)" if flagged else ""
        print(f"{path.name}: {len(segs)} segment(s), {kept:.0f}s -> {target}{note}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
