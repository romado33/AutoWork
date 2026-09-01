"""Idempotent record of which conversation emails have already gone out.

Transcription is keyed on the .md existing. Email was not keyed on anything:
re-running `python tools/run_pipeline.py --files …` after a successful send
duplicated the inbox. The key is the set of source files in the conversation,
so a later recording that joins the same Outlook event is a new send.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

LEDGER_NAME = ".sent.json"


def conversation_key(source_files: list[str] | tuple[str, ...]) -> str:
    """Stable identity for one email. Order of arrival must not change it."""
    return "|".join(sorted(Path(name).name.lower() for name in source_files if name))


def _path(ledger_dir: Path) -> Path:
    return Path(ledger_dir) / LEDGER_NAME


def _load(ledger_dir: Path) -> dict:
    path = _path(ledger_dir)
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        # A corrupt ledger must not swallow a real send: fail open, same as ingest.
        logger.warning("sent ledger %s unreadable (%s); treating as empty", path, exc)
        return {}
    if not isinstance(raw, dict):
        return {}
    return raw


def _save(ledger_dir: Path, data: dict) -> None:
    ledger_dir.mkdir(parents=True, exist_ok=True)
    path = _path(ledger_dir)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def already_sent(ledger_dir: Path, key: str) -> bool:
    if not key:
        return False
    return key in _load(ledger_dir)


def record_for(ledger_dir: Path, key: str) -> dict | None:
    entry = _load(ledger_dir).get(key)
    return entry if isinstance(entry, dict) else None


def mark_sent(ledger_dir: Path, key: str, subject: str) -> None:
    if not key:
        return
    data = _load(ledger_dir)
    data[key] = {
        "subject": subject,
        "sent_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    _save(ledger_dir, data)
