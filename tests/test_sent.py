#!/usr/bin/env python3
"""A conversation already emailed must not be mailed again on --files re-run.

There is no send ledger today: transcription is idempotent (the .md exists) but
email is not. Re-running the same cached files after a successful send duplicates
the inbox. The ledger key is the conversation's source files, so a later recording
that joins the same Outlook event is a new key and still mails.
"""

from __future__ import annotations

from autowork.sent import already_sent, conversation_key, mark_sent, record_for


def test_conversation_key_is_the_set_of_files_not_their_order() -> None:
    assert conversation_key(["b.MP3", "a.MP3"]) == conversation_key(["a.MP3", "b.MP3"])
    assert conversation_key(["a.MP3"]) != conversation_key(["a.MP3", "b.MP3"])


def test_marking_sent_is_remembered(tmp_path) -> None:
    key = conversation_key(["R2026-08-26-09-04-45.MP3"])
    assert not already_sent(tmp_path, key)
    mark_sent(tmp_path, key, subject="Call 2026-08-26 09:04: Okta")
    assert already_sent(tmp_path, key)


def test_a_joined_conversation_is_a_different_send(tmp_path) -> None:
    """Morning mailed file A; afternoon A+B sharing an Outlook event must still send."""
    mark_sent(tmp_path, conversation_key(["A.MP3"]), subject="morning")
    assert not already_sent(tmp_path, conversation_key(["A.MP3", "B.MP3"]))


def test_corrupt_ledger_does_not_block_a_send(tmp_path) -> None:
    (tmp_path / ".sent.json").write_text("{not json", encoding="utf-8")
    assert not already_sent(tmp_path, "anything")


def test_record_for_round_trips(tmp_path) -> None:
    key = conversation_key(["R2026-08-26-09-04-45.MP3"])
    mark_sent(tmp_path, key, subject="Call 2026-08-26")
    rec = record_for(tmp_path, key)
    assert rec is not None
    assert rec["subject"] == "Call 2026-08-26"
