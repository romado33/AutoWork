#!/usr/bin/env python3
"""Email the outstanding action queue. Intended for a weekday morning scheduled task.

Does not transcribe, summarise, or execute anything. Reads PENDING and APPROVED
items and mails them. Empty queue: exit 0, send nothing.

Usage:
    python tools/send_queue_digest.py
    python tools/send_queue_digest.py --no-email   # print only
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from autowork.digest import apply_glossary_all, digest_html, digest_subject, digest_text  # noqa: E402
from autowork.glossary import Glossary, GlossaryError  # noqa: E402
from autowork.llm import load_dotenv  # noqa: E402
from autowork.mailer import MailError, default_recipient, send  # noqa: E402
from autowork.queue import ReviewQueue  # noqa: E402

DEFAULT_QUEUE = PROJECT_ROOT / "queue.sqlite3"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--queue", default=os.environ.get("AUTOWORK_QUEUE", DEFAULT_QUEUE))
    parser.add_argument("--no-email", action="store_true")
    parser.add_argument("--to", default=None)
    args = parser.parse_args(argv)

    load_dotenv(PROJECT_ROOT / ".env")
    queue_path = Path(args.queue)
    if not queue_path.is_file():
        print(f"no queue at {queue_path}; nothing outstanding", file=sys.stderr)
        return 0

    with ReviewQueue(queue_path) as queue:
        actions = queue.list_outstanding()

    glossary_path = PROJECT_ROOT / "config" / "glossary.yml"
    try:
        glossary = Glossary.load(glossary_path)
        actions = apply_glossary_all(actions, glossary)
    except GlossaryError as exc:
        print(f"glossary: {exc}; sending names as stored", file=sys.stderr)

    if not actions:
        print("nothing outstanding; not sending")
        return 0

    subject = digest_subject(actions)
    text = digest_text(actions)
    print(text)
    if args.no_email:
        print("--no-email: not sending")
        return 0

    recipient = args.to or default_recipient()
    if not recipient:
        print("no recipient (set SUMMARY_TO in .env); not sending", file=sys.stderr)
        return 1
    try:
        print(send(recipient, subject, text, html_body=digest_html(actions)))
    except MailError as exc:
        print(f"email failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
