"""Durable send ledger so one conversation cannot be emailed twice.

USB plug-in is one email per Outlook conversation. Two processes can still
reach the mailer at once: a watcher volume-arrival and a manual `--files` run
both summarised R2026-09-21-09-45-57.MP3 at 13:42 and both mailed, because
`summaries/.sent.json` existed from earlier days and was never consulted.

Keyed by sorted source filenames (lowercase). `--resend` is the only override.

The exclusive lock is held from "may I send?" through the SMTP call so an
overlapping process cannot also decide to send. A check-then-write without the
lock is the 13:42 failure mode.
"""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)

LEDGER_NAME = ".sent.json"
LOCK_NAME = ".sent.lock"
CLAIM_TIMEOUT_SEC = 300


class SendLedgerError(RuntimeError):
    """Ledger lock or write failed. Raised rather than sending anyway."""


def ledger_key(source_files: list[str] | tuple[str, ...]) -> str:
    """Stable identity for one emailed conversation: sorted source filenames."""
    names = sorted({Path(name).name.lower() for name in source_files if name})
    if not names:
        raise SendLedgerError("cannot key a send with no source files")
    return "|".join(names)


class SendSlot:
    """Outcome of acquire(): either this process may send, or it must not."""

    def __init__(self, allowed: bool, ledger: SendLedger, key: str) -> None:
        self.allowed = allowed
        self._ledger = ledger
        self._key = key

    def record(self, subject: str) -> None:
        if not self.allowed:
            raise SendLedgerError("cannot record a send that was not claimed")
        self._ledger._write_entry(self._key, subject)


class SendLedger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.lock_path = self.path.with_name(LOCK_NAME)

    def already_sent(self, source_files: list[str] | tuple[str, ...]) -> bool:
        return self._entry(ledger_key(source_files)) is not None

    @contextmanager
    def acquire(
        self,
        source_files: list[str] | tuple[str, ...],
        *,
        resend: bool = False,
    ) -> Iterator[SendSlot]:
        """Yield a slot. The lock is held for the whole `with` body.

        Overlapping callers for the same sources: the second waits, then sees
        the first's ledger write and gets `allowed=False` (unless `--resend`).
        """
        key = ledger_key(source_files)
        with self._exclusive():
            already = self._entry(key)
            if already is not None and not resend:
                logger.info(
                    "already emailed %s at %s", key, already.get("sent_at")
                )
                yield SendSlot(False, self, key)
                return
            yield SendSlot(True, self, key)

    def _entry(self, key: str) -> dict | None:
        raw = self._load().get(key)
        return raw if isinstance(raw, dict) else None

    def _load(self) -> dict:
        if not self.path.is_file():
            return {}
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            # A corrupt ledger must not brick sending; worst case is a duplicate
            # of a conversation whose receipt we can no longer read.
            logger.warning(
                "send ledger %s unreadable (%s); treating as empty",
                self.path, exc,
            )
            return {}
        return raw if isinstance(raw, dict) else {}

    def _write_entry(self, key: str, subject: str) -> None:
        data = self._load()
        data[key] = {
            "sent_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "subject": subject[:200],
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, self.path)

    @contextmanager
    def _exclusive(self) -> Iterator[None]:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(self.lock_path), os.O_RDWR | os.O_CREAT)
        try:
            if os.lseek(fd, 0, os.SEEK_END) == 0:
                os.write(fd, b"0")
            os.lseek(fd, 0, os.SEEK_SET)
            _lock_fd(fd, timeout_sec=CLAIM_TIMEOUT_SEC)
            yield
        finally:
            try:
                os.lseek(fd, 0, os.SEEK_SET)
                _unlock_fd(fd)
            finally:
                os.close(fd)


def _lock_fd(fd: int, timeout_sec: float) -> None:
    """Exclusive lock with our own retry. msvcrt.LK_LOCK gives up after 10s."""
    deadline = time.monotonic() + timeout_sec
    while True:
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return
        except OSError:
            if time.monotonic() >= deadline:
                raise SendLedgerError(
                    f"timed out waiting {timeout_sec:.0f}s for the send ledger lock"
                ) from None
            time.sleep(0.05)


def _unlock_fd(fd: int) -> None:
    try:
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(fd, fcntl.LOCK_UN)
    except OSError:
        pass
