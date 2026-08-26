#!/usr/bin/env python3
"""Report what a recording contains before spending an hour transcribing it.

Usage:
    python tools/scan_audio.py D:/RECORD/R2026-08-25-13-23-54.MP3
    python tools/scan_audio.py RECORD/*.MP3 --windows        # per-window detail
    python tools/scan_audio.py day.mp3 --window-sec 30 --delta-clean 5.0

Runs at roughly 3000x realtime, so screening a full workday takes seconds. Transcribing
that same day takes ~90 minutes, which is the entire point of scanning first.

Exit codes: 0 success · 1 runtime failure · 2 bad arguments or missing input
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from autowork.gate import (  # noqa: E402
    GateConfig,
    GateError,
    Verdict,
    measure,
    segments,
    summarise,
)

MARKS = {Verdict.CLEAN: "CLEAN ", Verdict.USABLE: "usable", Verdict.REJECT: "REJECT"}


def hhmmss(seconds: float) -> str:
    total = int(seconds)
    return f"{total // 3600:d}:{(total % 3600) // 60:02d}:{total % 60:02d}"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("audio", nargs="+", help="audio file(s) to scan")
    parser.add_argument("--windows", action="store_true", help="print every window")
    parser.add_argument("--window-sec", type=float, default=GateConfig.window_sec)
    parser.add_argument("--speech-floor", type=float, default=GateConfig.speech_floor_db)
    parser.add_argument("--delta-usable", type=float, default=GateConfig.delta_usable_db)
    parser.add_argument("--delta-clean", type=float, default=GateConfig.delta_clean_db)
    parser.add_argument("--bridge-gap", type=float, default=GateConfig.bridge_gap_sec)
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def scan_one(path: Path, config: GateConfig, show_windows: bool) -> float:
    """Print the report for one file. Returns transcribable seconds."""
    windows = measure(path, config)
    counts = summarise(windows, config)
    segs = segments(windows, config)

    total_sec = len(windows) * config.window_sec
    keep_sec = sum(s.duration_sec for s in segs)

    print(f"\n=== {path.name} ===")
    print(f"    duration {hhmmss(total_sec)}   windows {len(windows)} "
          f"@ {config.window_sec:g}s")
    print(f"    clean {counts['clean']}  usable {counts['usable']}  "
          f"reject {counts['reject']}")

    if show_windows:
        print()
        for w in windows:
            print(f"    {hhmmss(w.start_sec)}  speech {w.speech_db:7.1f} dB  "
                  f"delta {w.delta_db:+6.1f} dB  {MARKS[w.verdict(config)]}")

    if segs:
        print("\n    transcribable segments:")
        for s in segs:
            print(f"      {hhmmss(s.start_sec)} - {hhmmss(s.end_sec)} "
                  f"({s.duration_sec:6.0f}s)  {MARKS[s.verdict]}  "
                  f"speech {s.mean_speech_db:6.1f} dB  delta {s.mean_delta_db:+5.1f} dB")
    else:
        print("\n    no transcribable segments -- nothing here would produce real text")

    pct = (100.0 * keep_sec / total_sec) if total_sec else 0.0
    print(f"\n    -> transcribe {hhmmss(keep_sec)} of {hhmmss(total_sec)} ({pct:.0f}%)")
    return keep_sec


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(message)s",
    )

    try:
        config = GateConfig(
            window_sec=args.window_sec,
            speech_floor_db=args.speech_floor,
            delta_usable_db=args.delta_usable,
            delta_clean_db=args.delta_clean,
            bridge_gap_sec=args.bridge_gap,
        )
    except GateError as exc:
        print(f"bad gate configuration: {exc}", file=sys.stderr)
        return 2

    paths = [Path(a) for a in args.audio]
    missing = [p for p in paths if not p.is_file()]
    if missing:
        for p in missing:
            print(f"no such file: {p}", file=sys.stderr)
        return 2

    total_keep = 0.0
    failures = 0
    for path in paths:
        try:
            total_keep += scan_one(path, config, args.windows)
        except GateError as exc:
            print(f"error scanning {path}: {exc}", file=sys.stderr)
            failures += 1

    if len(paths) > 1:
        print(f"\n=== {len(paths)} files: {hhmmss(total_keep)} to transcribe ===")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
