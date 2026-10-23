#!/usr/bin/env python3
"""Send ledger: one email per source-file set, even when two processes overlap.

The 2026-09-21 13:42 duplicate: watcher run-20260921-133711.log and the
manual run-20260921-aurea.log both summarised R2026-09-21-09-45-57.MP3 and
both mailed before either wrote summaries/.sent.json. A check-then-write
without a lock is that failure. These tests are two overlapping claims;
the second must not send.
"""

from __future__ import annotations

import time
from multiprocessing import Process
from pathlib import Path

from autowork.send_ledger import SendLedger, SendLedgerError, ledger_key


def _claim_and_maybe_send(ledger_path: str, result_path: str, hold: float) -> None:
    """Child process: claim the same sources, hold, record if allowed."""
    ledger = SendLedger(ledger_path)
    with ledger.acquire(["R2026-09-21-09-45-57.MP3"]) as slot:
        if not slot.allowed:
            Path(result_path).write_text("skipped", encoding="utf-8")
            return
        time.sleep(hold)
        slot.record("Call 2026-09-21 09:45")
        Path(result_path).write_text("sent", encoding="utf-8")


def test_ledger_key_is_sorted_lowercase_filenames() -> None:
    assert ledger_key(["B.MP3", "audio/a.mp3"]) == "a.mp3|b.mp3"
    assert ledger_key(["R2026-09-21-09-45-57.MP3"]) == "r2026-09-21-09-45-57.mp3"


def test_empty_sources_cannot_be_keyed() -> None:
    try:
        ledger_key([])
    except SendLedgerError as exc:
        assert "no source files" in str(exc)
    else:
        raise AssertionError("expected SendLedgerError")


def test_second_sequential_claim_does_not_send(tmp_path: Path) -> None:
    ledger = SendLedger(tmp_path / ".sent.json")
    sources = ["R2026-09-21-09-45-57.MP3"]
    with ledger.acquire(sources) as slot:
        assert slot.allowed
        slot.record("first")
    with ledger.acquire(sources) as slot:
        assert not slot.allowed
    assert ledger.already_sent(sources)


def test_resend_overrides_the_ledger(tmp_path: Path) -> None:
    ledger = SendLedger(tmp_path / ".sent.json")
    sources = ["R2026-09-21-09-45-57.MP3"]
    with ledger.acquire(sources) as slot:
        slot.record("first")
    with ledger.acquire(sources, resend=True) as slot:
        assert slot.allowed
        slot.record("second")
    assert "second" in (tmp_path / ".sent.json").read_text(encoding="utf-8")


def test_overlapping_claims_second_does_not_send(tmp_path: Path) -> None:
    """The 13:42 race: two processes, same sources, only one may send.

    Uses processes, not threads. Windows file locks are per-process, which is
    the real watcher-vs-manual-run case.
    """
    ledger_path = str(tmp_path / ".sent.json")
    a_out = tmp_path / "a.txt"
    b_out = tmp_path / "b.txt"
    first = Process(target=_claim_and_maybe_send, args=(ledger_path, str(a_out), 0.6))
    second = Process(target=_claim_and_maybe_send, args=(ledger_path, str(b_out), 0.05))
    first.start()
    time.sleep(0.15)
    second.start()
    first.join(timeout=10)
    second.join(timeout=10)
    assert first.exitcode == 0
    assert second.exitcode == 0
    outcomes = {a_out.read_text(encoding="utf-8"), b_out.read_text(encoding="utf-8")}
    assert outcomes == {"sent", "skipped"}
    assert SendLedger(ledger_path).already_sent(["R2026-09-21-09-45-57.MP3"])
