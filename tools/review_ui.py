#!/usr/bin/env python3
"""Local browser UI for the review queue.

Usage:
    python tools/review_ui.py              # bind 127.0.0.1:8765 and open a browser
    python tools/review_ui.py --port 9000
    python tools/review_ui.py --no-browser

This is a localhost page, not a network service. It never executes an action:
Done checks the item off the morning digest, Approve marks it eligible for a
future executor, Reject is extractor feedback. Plugging in a USB stick still
cannot write to Jira.

Configurable:
    AUTOWORK_QUEUE   queue database (default: ./queue.sqlite3)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

from autowork.action import ActionRecord, Status  # noqa: E402
from autowork.digest import group_by_recording_date  # noqa: E402
from autowork.extract import collapse_same_recording  # noqa: E402
from autowork.glossary import Glossary, GlossaryError  # noqa: E402
from autowork.queue import QueueError, ReviewQueue  # noqa: E402
from review import UNVERIFIED_DB, resolve, wall_clock  # noqa: E402

DEFAULT_QUEUE = PROJECT_ROOT / "queue.sqlite3"
DEFAULT_GLOSSARY = PROJECT_ROOT / "config" / "glossary.yml"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
LOCAL_HOSTS = frozenset({"127.0.0.1", "localhost"})

PAGE_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AutoWork review</title>
<style>
  :root {
    --bg: #f3efe6;
    --ink: #1c1915;
    --muted: #6b6258;
    --card: #fffcf7;
    --line: #e3d9c8;
    --done: #1f6b4a;
    --done-ink: #fff;
    --warn: #8a5a00;
    --warn-bg: #fff4d6;
    --reject: #8b2e2e;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; font: 16px/1.45 system-ui, Segoe UI, sans-serif;
    color: var(--ink); background: var(--bg);
  }
  header {
    background: #1c1915; color: #f3efe6; padding: 18px 24px 16px;
  }
  header h1 { margin: 0; font-size: 1.15rem; font-weight: 650; }
  header p { margin: 6px 0 0; color: #cbbfae; font-size: 0.9rem; }
  .glossary {
    display: flex; flex-wrap: wrap; gap: 8px; align-items: center;
    margin-top: 12px;
  }
  .glossary span { color: #cbbfae; font-size: 0.85rem; margin-right: 4px; }
  .glossary input {
    font: inherit; border: 1px solid #3d372f; background: #2a2620; color: #f3efe6;
    border-radius: 6px; padding: 6px 10px; width: 180px;
  }
  .glossary button {
    font: inherit; border-radius: 8px; padding: 6px 12px; cursor: pointer;
    border: 1px solid #cbbfae; background: transparent; color: #f3efe6;
  }
  main { max-width: 820px; margin: 0 auto; padding: 20px 16px 64px; }
  .tabs { display: flex; gap: 8px; margin-bottom: 18px; flex-wrap: wrap; align-items: center; }
  .tab {
    border: 1px solid var(--line); background: var(--card); color: var(--ink);
    border-radius: 999px; padding: 6px 14px; cursor: pointer; font: inherit;
  }
  .tab[aria-selected="true"] { background: #1c1915; color: #fff; border-color: #1c1915; }
  .counts { color: var(--muted); font-size: 0.9rem; margin-left: auto; }
  .day { font-size: 0.8rem; letter-spacing: 0.04em; text-transform: uppercase;
         color: var(--muted); margin: 22px 0 10px; }
  .card {
    background: var(--card); border: 1px solid var(--line); border-radius: 12px;
    padding: 16px 18px 14px; margin-bottom: 12px;
  }
  .card h2 { margin: 0 0 4px; font-size: 1.05rem; }
  .meta { color: var(--muted); font-size: 0.82rem; }
  .body { margin: 10px 0; }
  .quote {
    margin: 0; padding: 8px 12px; background: #f7f1e6; border-left: 3px solid #c4b496;
    color: #3d372f; font-style: italic;
  }
  .warn {
    background: var(--warn-bg); color: var(--warn); padding: 6px 10px;
    border-radius: 6px; font-size: 0.85rem; margin: 10px 0 0;
  }
  .badge {
    display: inline-block; font-size: 0.72rem; letter-spacing: 0.04em;
    text-transform: uppercase; border: 1px solid var(--line); border-radius: 999px;
    padding: 1px 8px; color: var(--muted); vertical-align: 2px;
  }
  .actions { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 14px; align-items: center; }
  .source-bar {
    display: flex; flex-wrap: wrap; gap: 8px; align-items: center;
    margin: 0 0 8px; padding: 8px 12px; background: #efe8db;
    border-radius: 8px; font-size: 0.85rem; color: var(--muted);
  }
  .source-bar .note { width: 100%; }
  button {
    font: inherit; border-radius: 8px; padding: 8px 14px; cursor: pointer; border: 1px solid var(--line);
    background: var(--card);
  }
  button.primary { background: var(--done); color: var(--done-ink); border-color: var(--done); }
  button.danger { color: var(--reject); border-color: #e2bcbc; }
  button:disabled { opacity: 0.55; cursor: default; }
  .note { width: 100%; margin-top: 10px; padding: 8px; font: inherit; border: 1px solid var(--line);
          border-radius: 8px; display: none; }
  .note.show { display: block; }
  .empty { color: var(--muted); padding: 32px 8px; }
  .error { color: var(--reject); margin: 12px 0; }
  footer { color: var(--muted); font-size: 0.8rem; margin-top: 28px; }
</style>
</head>
<body>
<header>
  <h1>AutoWork review</h1>
  <p>Mark done to drop an item from the morning to-do email. Nothing here is executed.</p>
  <form id="glossary-form" class="glossary">
    <span>Add a glossary name</span>
    <input id="g-term" placeholder="canonical (Okta)" autocomplete="off">
    <input id="g-variant" placeholder="heard as (Octo)" autocomplete="off">
    <button type="submit">Add</button>
  </form>
</header>
<main>
  <div class="tabs">
    <button class="tab" id="tab-todo" aria-selected="true">To do</button>
    <button class="tab" id="tab-closed">Closed</button>
    <span class="counts" id="counts"></span>
  </div>
  <div id="error" class="error" hidden></div>
  <div id="list"></div>
  <footer>
    Done is a check-off, not a write to Jira. Rejected and done items stay in the
    queue for audit. The CLI at <code>scripts\review.bat</code> still works.
  </footer>
</main>
<script>
const $ = (id) => document.getElementById(id);
let view = "outstanding";
let pendingReject = null;

function esc(s) {
  return String(s ?? "").replace(/[&<>"'`]/g, c => ({
    "&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;","`":"&#96;"
  }[c]));
}

async function load() {
  $("error").hidden = true;
  const res = await fetch("/api/queue?view=" + view);
  const data = await res.json();
  if (!res.ok) {
    $("error").hidden = false;
    $("error").textContent = data.error || ("HTTP " + res.status);
    return;
  }
  const c = data.counts || {};
  const outstanding = (c.pending || 0) + (c.approved || 0);
  $("counts").textContent = outstanding + " on the morning email";
  $("tab-todo").textContent = "To do (" + outstanding + ")";
  const closed = (c.done || 0) + (c.rejected || 0) + (c.executed || 0) + (c.failed || 0);
  $("tab-closed").textContent = "Closed (" + closed + ")";
  render(data.groups || []);
}

function sourceNoteId(src) {
  return "note-source-" + encodeURIComponent(src);
}

function render(groups) {
  const root = $("list");
  if (!groups.length) {
    root.innerHTML = view === "outstanding"
      ? "<div class='empty'>Nothing outstanding. The morning to-do email will be empty.</div>"
      : "<div class='empty'>Nothing closed yet.</div>";
    return;
  }
  root.innerHTML = groups.map(g => {
    const bySource = {};
    const order = [];
    for (const item of (g.items || [])) {
      const src = item.source || "unknown";
      if (!bySource[src]) { bySource[src] = []; order.push(src); }
      bySource[src].push(item);
    }
    return "<div class='day'>" + esc(g.date) + "</div>" + order.map(src => {
      const items = bySource[src];
      const nid = sourceNoteId(src);
      const bar = view === "outstanding"
        ? ("<div class='source-bar'><span>" + esc(src) + " · " + items.length +
           " item(s)</span>" +
           "<button class='danger' data-act='reject-source-toggle' data-source='" +
           esc(src) + "'>Reject this recording</button>" +
           "<textarea class='note' id='" + nid +
           "' rows='2' placeholder='Why reject this recording? Required.'></textarea></div>")
        : ("<div class='source-bar'><span>" + esc(src) + "</span></div>");
      return bar + items.map(card).join("");
    }).join("");
  }).join("");
}

function card(item) {
  const unverified = item.unverified
    ? "<div class='warn'>Audio below the fabrication threshold — listen before trusting this.</div>"
    : "";
  const buttons = view === "outstanding" ? (
    "<div class='actions'>" +
      "<button class='primary' data-act='done' data-id='" + esc(item.id) + "'>Mark done</button>" +
      "<button class='danger' data-act='reject-toggle' data-id='" + esc(item.id) + "'>Reject</button>" +
      "<button data-act='approve' data-id='" + esc(item.id) + "'>Approve (stays on to-do)</button>" +
    "</div>" +
    "<textarea class='note' id='note-" + esc(item.id) + "' rows='2' placeholder='Why reject? Required.'></textarea>"
  ) : "";
  return "<article class='card'>" +
    "<h2>" + esc(item.title) + " <span class='badge'>" + esc(item.status) + "</span></h2>" +
    "<div class='meta'>" + esc(item.when_said) + " · " + esc(item.source) +
      " · confidence " + esc(item.confidence) + " · id " + esc(item.short_id) + "</div>" +
    "<p class='body'>" + esc(item.body) + "</p>" +
    "<blockquote class='quote'>“" + esc(item.quote) + "”</blockquote>" +
    unverified + buttons +
    (item.review_note ? "<p class='meta'>Note: " + esc(item.review_note) + "</p>" : "") +
  "</article>";
}

$("tab-todo").onclick = () => { view = "outstanding"; $("tab-todo").setAttribute("aria-selected","true"); $("tab-closed").setAttribute("aria-selected","false"); load(); };
$("tab-closed").onclick = () => { view = "closed"; $("tab-closed").setAttribute("aria-selected","true"); $("tab-todo").setAttribute("aria-selected","false"); load(); };

document.addEventListener("click", async (ev) => {
  const btn = ev.target.closest("button[data-act]");
  if (!btn) return;
  const act = btn.getAttribute("data-act");
  if (act === "reject-toggle") {
    const id = btn.getAttribute("data-id");
    const box = $("note-" + id);
    const opening = !box.classList.contains("show");
    document.querySelectorAll(".note").forEach(n => n.classList.remove("show"));
    if (opening) {
      box.classList.add("show");
      btn.textContent = "Confirm reject";
      btn.setAttribute("data-act", "reject");
      box.focus();
    }
    return;
  }
  if (act === "reject-source-toggle") {
    const src = btn.getAttribute("data-source");
    const box = document.getElementById(sourceNoteId(src));
    const opening = !box.classList.contains("show");
    document.querySelectorAll(".note").forEach(n => n.classList.remove("show"));
    if (opening) {
      box.classList.add("show");
      btn.textContent = "Confirm reject this recording";
      btn.setAttribute("data-act", "reject-source");
      box.focus();
    }
    return;
  }
  btn.disabled = true;
  let res;
  if (act === "reject-source") {
    const src = btn.getAttribute("data-source");
    const note = (document.getElementById(sourceNoteId(src)) || {}).value || "";
    res = await fetch("/api/source/reject", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({source: src, note}),
    });
  } else {
    const id = btn.getAttribute("data-id");
    const note = ($("note-" + id) || {}).value || "";
    res = await fetch("/api/item/" + encodeURIComponent(id) + "/" + act, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({note}),
    });
  }
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    $("error").hidden = false;
    $("error").textContent = data.error || ("HTTP " + res.status);
    btn.disabled = false;
    return;
  }
  await load();
});

load();

$("glossary-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const term = $("g-term").value.trim();
  const variant = $("g-variant").value.trim();
  $("error").hidden = true;
  const res = await fetch("/api/glossary", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({term, variant}),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    $("error").hidden = false;
    $("error").textContent = data.error || ("HTTP " + res.status);
    return;
  }
  $("g-term").value = "";
  $("g-variant").value = "";
});
</script>
</body>
</html>
"""


def item_view(record: ActionRecord) -> dict:
    p = record.provenance
    return {
        "id": record.id,
        "short_id": record.id[:8],
        "status": record.status.value,
        "title": record.title,
        "body": record.body,
        "quote": p.transcript_excerpt,
        "source": Path(p.source_audio).name,
        "when_said": wall_clock(record),
        "confidence": f"{record.confidence:.2f}",
        "target": record.target_system,
        "action_type": record.action_type.value,
        "unverified": p.speech_rumble_db < UNVERIFIED_DB,
        "review_note": record.review_note,
    }


def grouped_views(records: list[ActionRecord]) -> list[dict]:
    groups = []
    for date, items in group_by_recording_date(records):
        groups.append({"date": date, "items": [item_view(a) for a in items]})
    return groups


def apply_action(
    queue: ReviewQueue, action_id: str, verb: str, note: str | None = None
) -> ActionRecord:
    """Mutate review state only. There is no execute verb on purpose."""
    resolved = resolve(queue, action_id)
    if verb == "done":
        return queue.mark_done(resolved, note=note or None)
    if verb == "approve":
        return queue.approve(resolved, note=note or None)
    if verb == "reject":
        return queue.reject(resolved, note=note or "")
    if verb == "retry":
        return queue.retry(resolved)
    raise QueueError(f"unknown review action {verb!r}")


def _is_local(handler: BaseHTTPRequestHandler) -> bool:
    host = (handler.headers.get("Host") or "").split(":")[0].strip().lower()
    if host not in LOCAL_HOSTS:
        return False
    origin = handler.headers.get("Origin")
    if origin:
        parsed = urlparse(origin)
        if parsed.hostname and parsed.hostname.lower() not in LOCAL_HOSTS:
            return False
    return True


class ReviewUIHandler(BaseHTTPRequestHandler):
    queue_path: Path = DEFAULT_QUEUE
    glossary_path: Path = DEFAULT_GLOSSARY

    def log_message(self, fmt: str, *args: object) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, payload: dict) -> None:
        self._send(code, json.dumps(payload).encode("utf-8"), "application/json; charset=utf-8")

    def _refuse_if_remote(self) -> bool:
        if _is_local(self):
            return False
        self._json(403, {"error": "this UI is localhost-only"})
        return True

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if self._refuse_if_remote():
            return
        parsed = urlparse(self.path)
        if parsed.path in {"/", "/index.html"}:
            self._send(200, PAGE_HTML.encode("utf-8"), "text/html; charset=utf-8")
            return
        if parsed.path == "/api/queue":
            view = "outstanding"
            if parsed.query:
                for part in parsed.query.split("&"):
                    if part.startswith("view="):
                        view = part.split("=", 1)[1]
            self._json(200, self._queue_payload(view))
            return
        self._json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if self._refuse_if_remote():
            return
        parsed = urlparse(self.path)
        parts = [p for p in parsed.path.split("/") if p]
        length = int(self.headers.get("Content-Length") or "0")
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            self._json(400, {"error": "body must be JSON"})
            return
        if not isinstance(payload, dict):
            self._json(400, {"error": "body must be a JSON object"})
            return
        note = payload.get("note")
        if not self.queue_path.is_file():
            self._json(404, {"error": f"no queue database at {self.queue_path}"})
            return

        if parts == ["api", "glossary"]:
            term = str(payload.get("term") or "")
            variant = str(payload.get("variant") or "")
            try:
                promoted = Glossary.promote(
                    self.glossary_path, term, variant=variant
                )
            except GlossaryError as exc:
                self._json(400, {"error": str(exc)})
                return
            self._json(200, {
                "ok": True,
                "term": promoted.term,
                "variants": list(promoted.variants),
            })
            return

        if parts == ["api", "source", "reject"]:
            source = str(payload.get("source") or "")
            try:
                with ReviewQueue(self.queue_path) as queue:
                    rejected = queue.reject_source(source, note=note or "")
            except QueueError as exc:
                self._json(400, {"error": str(exc)})
                return
            self._json(200, {
                "ok": True,
                "rejected": len(rejected),
                "ids": [item.id for item in rejected],
            })
            return

        if len(parts) != 4 or parts[0] != "api" or parts[1] != "item":
            self._json(404, {"error": "not found"})
            return
        action_id, verb = parts[2], parts[3]
        if verb not in {"done", "approve", "reject", "retry"}:
            self._json(404, {"error": "not found"})
            return
        try:
            with ReviewQueue(self.queue_path) as queue:
                record = apply_action(queue, action_id, verb, note=note)
        except QueueError as exc:
            self._json(400, {"error": str(exc)})
            return
        self._json(200, {"ok": True, "item": item_view(record)})

    def _queue_payload(self, view: str) -> dict:
        if not self.queue_path.is_file():
            return {
                "error": f"no queue database at {self.queue_path}",
                "groups": [],
                "counts": {},
            }
        with ReviewQueue(self.queue_path) as queue:
            counts = queue.counts()
            if view == "closed":
                records = [
                    a for a in queue.list()
                    if a.status not in (Status.PENDING, Status.APPROVED)
                ]
            else:
                records = collapse_same_recording(queue.list_outstanding())
            return {"groups": grouped_views(records), "counts": counts}


def make_server(
    queue_path: Path,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    glossary_path: Path | None = None,
) -> HTTPServer:
    if host not in LOCAL_HOSTS:
        raise ValueError(f"refusing to bind {host!r}; this UI is localhost-only")
    handler = type(
        "BoundReviewUIHandler",
        (ReviewUIHandler,),
        {
            "queue_path": Path(queue_path),
            "glossary_path": Path(glossary_path or DEFAULT_GLOSSARY),
        },
    )
    return HTTPServer((host, port), handler)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--queue", default=os.environ.get("AUTOWORK_QUEUE", DEFAULT_QUEUE))
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)

    try:
        server = make_server(Path(args.queue), host=args.host, port=args.port)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"error: could not bind {args.host}:{args.port}: {exc}", file=sys.stderr)
        return 1

    url = f"http://{args.host}:{server.server_port}/"
    print(f"Review queue UI on {url}", flush=True)
    print("Marking done drops an item from the morning to-do email. Nothing is executed.", flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
