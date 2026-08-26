#!/usr/bin/env python3
"""Review pending action items: see them, approve them, reject them.

Usage:
    python tools/review.py                     # interactive review of PENDING items
    python tools/review.py --list              # show the queue, change nothing
    python tools/review.py --list --status approved
    python tools/review.py --approve <id> [--note "..."]
    python tools/review.py --reject  <id> --note "why"
    python tools/review.py --retry   <id>      # a FAILED action back to APPROVED
    python tools/review.py --show    <id>      # full detail for one action

Configurable rather than hardcoded:
    AUTOWORK_QUEUE   queue database (default: ./queue.sqlite3)

This tool never executes anything. Approving marks an action eligible for an executor;
a separate step runs it. That separation is deliberate -- approving is a judgement, and
executing is an effect, and conflating them means a mis-keystroke writes to Jira.

Every item is shown with the provenance needed to judge it: the audio file and offset,
the measured signal quality of that window, and the verbatim quote the extractor
grounded the action in. If the quote does not support the action, reject it. If the
signal quality is low, go and listen before approving.

Exit codes: 0 success · 1 runtime failure · 2 bad arguments
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from autowork.action import ActionRecord, Status  # noqa: E402
from autowork.queue import QueueError, ReviewQueue  # noqa: E402

DEFAULT_QUEUE = PROJECT_ROOT / "queue.sqlite3"

# Below this, the source audio measured in the range where Whisper fabricates. Such an
# action may be describing a conversation that never happened.
UNVERIFIED_DB = 1.0

STATUS_MARK = {
    Status.PENDING: "?",
    Status.APPROVED: "+",
    Status.REJECTED: "x",
    Status.EXECUTED: "*",
    Status.FAILED: "!",
}


def wall_clock(record: ActionRecord) -> str:
    """Absolute time of the moment this action came from, if the filename encodes it.

    Returns an offset instead of guessing when the name does not parse: a wrong
    wall-clock time is worse than none, because it gets correlated against a calendar.
    """
    stem = Path(record.provenance.source_audio).stem
    try:
        base = datetime.strptime(stem[1:20], "%Y-%m-%d-%H-%M-%S")
    except (ValueError, IndexError):
        return f"+{int(record.provenance.start_sec) // 60}m"
    return (base + timedelta(seconds=record.provenance.start_sec)).strftime(
        "%Y-%m-%d %H:%M"
    )


def summarise(record: ActionRecord) -> str:
    mark = STATUS_MARK.get(record.status, " ")
    flag = " [UNVERIFIED AUDIO]" if record.provenance.speech_rumble_db < UNVERIFIED_DB else ""
    return (
        f"{mark} {record.id[:8]}  {record.confidence:.2f}  "
        f"{record.target_system:<14} {record.title}{flag}"
    )


def show(record: ActionRecord) -> None:
    p = record.provenance
    print(f"\n{'=' * 72}")
    print(f"{record.title}")
    print(f"{'=' * 72}")
    print(f"  id          {record.id}")
    print(f"  status      {record.status.value}")
    print(f"  target      {record.target_system}"
          + (f" -> {record.target_ref}" if record.target_ref else ""))
    print(f"  action      {record.action_type.value}")
    print(f"  confidence  {record.confidence:.2f}")
    print(f"  when said   {wall_clock(record)}")
    print(f"  audio       {Path(p.source_audio).name} "
          f"[{p.start_sec:.0f}s-{p.end_sec:.0f}s]")

    quality = f"{p.speech_rumble_db:+.1f} dB speech-minus-rumble"
    if p.speech_rumble_db < UNVERIFIED_DB:
        quality += "  <-- BELOW the fabrication threshold; verify against the audio"
    print(f"  quality     {quality}")

    if p.glossary_terms_applied:
        print(f"  glossary    {', '.join(p.glossary_terms_applied)}")
    print(f"  extractor   {p.extractor}")

    print(f"\n  WHAT TO DO\n    {record.body}")
    print(f"\n  GROUNDING QUOTE (from the transcript)\n    \"{p.transcript_excerpt}\"")

    if record.review_note:
        print(f"\n  review note  {record.review_note}")
    if record.execution_result:
        print(f"\n  execution    [{record.executor}] {record.execution_result}")
    print()


def interactive(queue: ReviewQueue) -> int:
    """Walk the PENDING queue one item at a time."""
    pending = queue.list(status=Status.PENDING)
    if not pending:
        print("Nothing pending. Queue:", queue.counts() or "empty")
        return 0

    print(f"{len(pending)} pending action(s), highest confidence first.")
    print("For each: [a]pprove  [r]eject  [s]kip  [q]uit\n")

    approved = rejected = skipped = 0
    for index, record in enumerate(pending, start=1):
        show(record)
        print(f"  ---- {index} of {len(pending)} ----")

        while True:
            try:
                choice = input("  [a]pprove / [r]eject / [s]kip / [q]uit > ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\n  stopped.")
                choice = "q"

            if choice in {"a", "approve"}:
                note = input("  note (optional) > ").strip() or None
                queue.approve(record.id, note=note)
                approved += 1
                print("  APPROVED. Not executed -- run the executor separately.")
                break
            if choice in {"r", "reject"}:
                # A reason is required by the queue: it is the only feedback the
                # extractor ever gets about what it is getting wrong.
                note = ""
                while not note.strip():
                    note = input("  why? (required) > ")
                queue.reject(record.id, note=note)
                rejected += 1
                print("  rejected.")
                break
            if choice in {"s", "skip", ""}:
                skipped += 1
                break
            if choice in {"q", "quit"}:
                print(f"\napproved {approved}, rejected {rejected}, skipped {skipped}")
                print("queue:", queue.counts())
                return 0
            print("  unrecognised; enter a, r, s or q")

    print(f"\napproved {approved}, rejected {rejected}, skipped {skipped}")
    print("queue:", queue.counts())
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--queue", default=os.environ.get("AUTOWORK_QUEUE", DEFAULT_QUEUE))
    parser.add_argument("--list", action="store_true", help="list and exit")
    parser.add_argument("--status", choices=[s.value for s in Status],
                        help="filter --list by status (default: all)")
    parser.add_argument("--target", help="filter --list by target system")
    parser.add_argument("--show", metavar="ID", help="full detail for one action")
    parser.add_argument("--approve", metavar="ID")
    parser.add_argument("--reject", metavar="ID")
    parser.add_argument("--retry", metavar="ID")
    parser.add_argument("--note")
    args = parser.parse_args(argv)

    queue_path = Path(args.queue)
    if not queue_path.is_file():
        print(f"no queue database at {queue_path}. Run tools/extract_actions.py first.",
              file=sys.stderr)
        return 2

    with ReviewQueue(queue_path) as queue:
        try:
            if args.show is not None:
                show(queue.get(resolve(queue, args.show)))
                return 0

            if args.approve is not None:
                record = queue.approve(resolve(queue, args.approve), note=args.note)
                print(f"approved {record.id[:8]}: {record.title}")
                print("Not executed. Run the executor separately.")
                return 0

            if args.reject is not None:
                if not args.note:
                    print("--reject requires --note (it is the extractor's only "
                          "feedback)", file=sys.stderr)
                    return 2
                record = queue.reject(resolve(queue, args.reject), note=args.note)
                print(f"rejected {record.id[:8]}: {record.title}")
                return 0

            if args.retry is not None:
                record = queue.retry(resolve(queue, args.retry))
                print(f"{record.id[:8]} back to {record.status.value}")
                return 0

            if args.list:
                status = Status(args.status) if args.status else None
                records = queue.list(status=status, target_system=args.target)
                if not records:
                    print("nothing matches. Queue:", queue.counts() or "empty")
                    return 0
                print(f"{'':2}{'id':<9} {'conf':<5} {'target':<14} title")
                for record in records:
                    print(summarise(record))
                print(f"\n{len(records)} shown. Queue: {queue.counts()}")
                return 0

            return interactive(queue)

        except QueueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1


def resolve(queue: ReviewQueue, prefix: str) -> str:
    """Accept a short id prefix, as printed by --list, not just the full uuid.

    Refuses an ambiguous prefix rather than picking one, because the wrong action
    getting approved is exactly the failure this tool exists to prevent.
    """
    try:
        queue.get(prefix)
        return prefix
    except QueueError:
        pass

    if not prefix.strip():
        raise QueueError("an empty action id was given; pass an id from --list")

    matches = [r for r in queue.list() if r.id.startswith(prefix)]
    if not matches:
        raise QueueError(f"no action matching id {prefix!r}")
    if len(matches) > 1:
        ids = ", ".join(r.id[:12] for r in matches)
        raise QueueError(f"id {prefix!r} is ambiguous; matches {ids}")
    return matches[0].id


if __name__ == "__main__":
    raise SystemExit(main())
