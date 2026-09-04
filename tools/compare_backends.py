#!/usr/bin/env python3
"""Measure one cloud backend against another on the same transcript: latency and recall.

Usage:
    python tools/compare_backends.py transcripts/R2026-08-25-13-23-54.md
    python tools/compare_backends.py <transcript> --backends openai:gpt-5.4-mini anthropic:claude-sonnet-5
    python tools/compare_backends.py <transcript> --no-prefilter    # measure the full cost

Requires ANTHROPIC_API_KEY (or `ant auth login`) for any anthropic backend.

WHAT THIS MEASURES, and why each column matters:

    latency     wall-clock. The number that decides whether a full day is workable.
    accepted    candidates that survived quote-grounding against the transcript.
    rejected    candidates the grounding check refused, with reasons. A high count is
                not automatically bad -- it means the guard is working -- but a model
                that is mostly rejected is a model that mostly fabricates.
    recall      whether the KNOWN action items in this transcript were found. Speed is
                worthless without this, and it is the column local models fail on.

Ground truth is per-transcript and hand-specified below; extend KNOWN_ITEMS as you
review more recordings. Measuring recall against a guess is worse than not measuring it.

Exit codes: 0 success · 1 a backend failed · 2 bad arguments
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

from autowork.extract import (  # noqa: E402
    SYSTEM_PROMPT,
    ExtractorConfig,
    ground,
    parse_candidates,
)
from autowork.llm import LLMError, build_backend, load_dotenv  # noqa: E402
from autowork.prefilter import filtered_text  # noqa: E402
from extract_actions import parse_transcript  # noqa: E402

# Hand-verified action items per transcript, keyed by filename stem. Each value is a
# distinctive substring that must appear in an accepted candidate's grounding quote.
KNOWN_ITEMS: dict[str, tuple[str, ...]] = {
    "R2026-08-25-13-23-54": (
        "Dave Casale",
        "anyone reading the quote",
        "over like 0.9",
    ),
}

ACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "actions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "body": {"type": "string"},
                    "owner": {"type": "string", "enum": ["me", "other", "unclear"]},
                    "target_system": {"type": "string"},
                    "action_type": {
                        "type": "string",
                        "enum": ["create", "update", "comment", "message", "task"],
                    },
                    "confidence": {"type": "number"},
                    "quote": {"type": "string"},
                },
                "required": [
                    "title", "body", "owner", "target_system",
                    "action_type", "confidence", "quote",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["actions"],
    "additionalProperties": False,
}


def build_prompt(text: str, config: ExtractorConfig) -> str:
    systems = ", ".join(config.target_systems)
    return (
        f"{SYSTEM_PROMPT}\n"
        f"Allowed target_system values: {systems}\n"
        f"The recording's owner is {config.owner_name}.\n\n"
        f"TRANSCRIPT:\n{text}\n\nJSON:"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("transcript")
    parser.add_argument(
        "--backends",
        nargs="+",
        default=["anthropic:claude-sonnet-5", "openai:gpt-5.4-mini"],
    )
    parser.add_argument("--no-prefilter", action="store_true")
    parser.add_argument("--owner", default="Rob")
    args = parser.parse_args(argv)

    # Keys live in .env, never in the source or the command line.
    load_dotenv(PROJECT_ROOT / ".env")

    path = Path(args.transcript)
    if not path.is_file():
        print(f"no such transcript: {path}", file=sys.stderr)
        return 2

    segments = parse_transcript(path)
    if not segments:
        print(f"no transcript segments in {path.name}", file=sys.stderr)
        return 2

    raw = "\n\n".join(s.text for s in segments)
    text = raw if args.no_prefilter else filtered_text(raw)
    if not text.strip():
        print("prefilter selected nothing; use --no-prefilter to send everything")
        return 0

    config = ExtractorConfig(owner_name=args.owner)
    prompt = build_prompt(text, config)
    known = KNOWN_ITEMS.get(path.stem, ())

    print(f"{path.name}")
    print(f"  transcript      {len(raw):,} chars")
    print(f"  sent to model   {len(text):,} chars"
          f"{'' if args.no_prefilter else f' (prefiltered, {100 * (1 - len(text) / len(raw)):.0f}% skipped)'}")
    print(f"  prompt total    {len(prompt):,} chars")
    if known:
        print(f"  known items     {len(known)}")
    else:
        print("  known items     none recorded -- recall not measured for this file")
    print()

    failures = 0
    for spec in args.backends:
        try:
            backend = build_backend(spec)
        except LLMError as exc:
            print(f"{spec}: unavailable -- {exc}")
            failures += 1
            continue

        try:
            completion = backend.complete(prompt, schema=ACTION_SCHEMA)
            candidates = ground(parse_candidates(completion.text, config), text, config)
        except LLMError as exc:
            print(f"{completion_label(spec)}: FAILED after "
                  f"-- {str(exc)[:200]}")
            failures += 1
            continue
        except Exception as exc:  # noqa: BLE001 - report and keep comparing
            print(f"{completion_label(spec)}: unusable reply -- {str(exc)[:200]}")
            failures += 1
            continue

        accepted = [c for c in candidates if not c.rejected_because]
        rejected = [c for c in candidates if c.rejected_because]

        print(f"##### {completion.label}")
        print(f"  latency     {completion.elapsed_sec:.1f}s")
        if completion.input_tokens or completion.output_tokens:
            print(f"  tokens      {completion.input_tokens} in / "
                  f"{completion.output_tokens} out")
        print(f"  accepted    {len(accepted)}")
        print(f"  rejected    {len(rejected)}")

        if known:
            quotes = " ".join(c.quote for c in accepted)
            found = [item for item in known if item in quotes]
            print(f"  recall      {len(found)}/{len(known)} known items")
            for item in known:
                print(f"                {'FOUND' if item in quotes else 'MISSED'}  {item}")

        for candidate in accepted:
            print(f"    + {candidate.title[:66]}")
        for candidate in rejected:
            print(f"    - {candidate.title[:52]}  ({candidate.rejected_because})")
        print()

    return 1 if failures else 0


def completion_label(spec: str) -> str:
    return spec


if __name__ == "__main__":
    raise SystemExit(main())
