#!/usr/bin/env python3
"""Local review UI: mark done without executing, localhost-only.

The morning digest is PENDING + APPROVED. The UI's job is to let a human move
an item to DONE (or REJECTED) so it stops appearing there. There is no execute
route on purpose.
"""

from __future__ import annotations

import json
import sys
import threading
from http.client import HTTPConnection
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from autowork.action import ActionRecord, ActionType, Provenance, Status  # noqa: E402
from autowork.digest import digest_text  # noqa: E402
from autowork.queue import QueueError, ReviewQueue  # noqa: E402
from review_ui import apply_action, make_server  # noqa: E402


def record(**overrides) -> ActionRecord:
    provenance_overrides = overrides.pop("provenance_overrides", {})
    provenance = dict(
        source_audio="D:/RECORD/R2026-08-25-13-23-54.MP3",
        start_sec=2100.0,
        end_sec=2220.0,
        speech_rumble_db=6.6,
        transcript_excerpt="I should take it up again with Dave Casale and get it finalized",
        extractor="openai:gpt-5.4-mini/grounded",
    )
    provenance.update(provenance_overrides)
    defaults = dict(
        title="Finalise the backlog tool with Dave Casale",
        body="Take it up again with Dave Casale.",
        target_system="backlog-tool",
        action_type=ActionType.UPDATE,
        confidence=0.82,
        provenance=Provenance(**provenance),  # type: ignore[arg-type]
    )
    defaults.update(overrides)
    return ActionRecord(**defaults)  # type: ignore[arg-type]


@pytest.fixture()
def db(tmp_path):
    path = tmp_path / "q.sqlite3"
    with ReviewQueue(path) as q:
        q.enqueue(record())
    return path


@pytest.fixture()
def server(db):
    httpd = make_server(db, port=0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield httpd
    httpd.shutdown()
    httpd.server_close()


def _request(httpd, method: str, path: str, body: dict | None = None,
             origin: str | None = None) -> tuple[int, dict | str]:
    host, port = httpd.server_address
    conn = HTTPConnection(host, port, timeout=5)
    headers = {}
    payload = None
    if origin is not None:
        headers["Origin"] = origin
    if body is not None:
        payload = json.dumps(body)
        headers["Content-Type"] = "application/json"
    conn.request(method, path, body=payload, headers=headers)
    res = conn.getresponse()
    raw = res.read().decode("utf-8")
    conn.close()
    if res.getheader("Content-Type", "").startswith("application/json"):
        return res.status, json.loads(raw)
    return res.status, raw


def test_page_is_served(server) -> None:
    status, body = _request(server, "GET", "/")
    assert status == 200
    assert "Mark done" in body
    assert "Nothing here is executed" in body


def test_outstanding_list_includes_pending(server) -> None:
    status, payload = _request(server, "GET", "/api/queue?view=outstanding")
    assert status == 200
    items = payload["groups"][0]["items"]
    assert items[0]["title"] == "Finalise the backlog tool with Dave Casale"
    assert items[0]["status"] == "pending"


def test_done_via_ui_drops_item_from_digest(server, db) -> None:
    """The contract the morning email depends on: DONE is not outstanding."""
    status, payload = _request(server, "GET", "/api/queue?view=outstanding")
    action_id = payload["groups"][0]["items"][0]["id"]

    status, result = _request(server, "POST", f"/api/item/{action_id}/done", {"note": ""})
    assert status == 200
    assert result["item"]["status"] == "done"

    with ReviewQueue(db) as q:
        stored = q.get(action_id)
        assert stored.status is Status.DONE
        assert stored.executed_at is None
        assert stored.executor is None
        outstanding = q.list_outstanding()
        assert outstanding == []
        assert "Finalise the backlog tool" not in digest_text(outstanding)


def test_reject_without_a_note_is_refused(server, db) -> None:
    status, payload = _request(server, "GET", "/api/queue?view=outstanding")
    action_id = payload["groups"][0]["items"][0]["id"]

    status, result = _request(server, "POST", f"/api/item/{action_id}/reject", {"note": ""})
    assert status == 400
    assert "note" in result["error"].lower()

    with ReviewQueue(db) as q:
        assert q.get(action_id).status is Status.PENDING


def test_approve_does_not_execute_and_stays_outstanding(server, db) -> None:
    status, payload = _request(server, "GET", "/api/queue?view=outstanding")
    action_id = payload["groups"][0]["items"][0]["id"]

    status, result = _request(server, "POST", f"/api/item/{action_id}/approve", {})
    assert status == 200
    assert result["item"]["status"] == "approved"

    with ReviewQueue(db) as q:
        stored = q.get(action_id)
        assert stored.status is Status.APPROVED
        assert stored.executor is None
        assert [a.id for a in q.list_outstanding()] == [action_id]


def test_there_is_no_execute_route(server, db) -> None:
    status, payload = _request(server, "GET", "/api/queue?view=outstanding")
    action_id = payload["groups"][0]["items"][0]["id"]
    status, result = _request(server, "POST", f"/api/item/{action_id}/execute", {})
    assert status == 404
    with ReviewQueue(db) as q:
        assert q.get(action_id).status is Status.PENDING


def test_remote_origin_is_refused(server) -> None:
    status, result = _request(
        server, "GET", "/api/queue", origin="http://evil.example"
    )
    assert status == 403
    assert "localhost" in result["error"]


def test_refuses_to_bind_a_public_interface() -> None:
    with pytest.raises(ValueError, match="localhost-only"):
        make_server(Path("queue.sqlite3"), host="0.0.0.0", port=0)


def test_apply_action_unknown_verb_is_refused(db) -> None:
    with ReviewQueue(db) as q:
        action_id = q.list()[0].id
        with pytest.raises(QueueError, match="unknown review action"):
            apply_action(q, action_id, "execute")
        assert q.get(action_id).status is Status.PENDING
