#!/usr/bin/env python3
"""Executor that writes an approved action to a markdown file and stops there.

Why this exists as a first-class adapter rather than a fallback: it is the whole system
minus the last inch, and it is the version that keeps working when you no longer have
access to any of the target systems. Hand-off is a legitimate terminal state -- the value
of the pipeline is knowing what needs doing and having it written down accurately, not
necessarily having a robot do it.

It is also the safe default for a target you have an adapter for but do not yet trust:
point the config at file_handoff, read the output for a week, then switch.

Writes are atomic (temp file plus replace) so a crash mid-write cannot leave a half
file that a later run would append to.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

from autowork.action import ActionRecord
from autowork.executors.base import Executor, ExecutionResult

logger = logging.getLogger(__name__)

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def _slug(text: str, limit: int = 60) -> str:
    cleaned = _UNSAFE.sub("-", text.strip()).strip("-").lower()
    return (cleaned[:limit].rstrip("-") or "untitled")


class FileHandoffExecutor(Executor):
    """Renders one markdown file per action into `out_dir`."""

    def __init__(self, out_dir: str | Path) -> None:
        self.out_dir = Path(out_dir)

    @property
    def name(self) -> str:
        return "file_handoff"

    def can_handle(self, action: ActionRecord) -> bool:
        """Any action can be written down. The filesystem is the only dependency."""
        return True

    def _render(self, action: ActionRecord) -> str:
        p = action.provenance
        glossary = ", ".join(p.glossary_terms_applied) or "(none)"
        return "\n".join(
            [
                f"# {action.title}",
                "",
                f"- **Action**: {action.action_type.value}",
                f"- **Target**: {action.target_system}"
                + (f" -> `{action.target_ref}`" if action.target_ref else ""),
                f"- **Confidence**: {action.confidence:.2f}",
                f"- **Approved**: {action.reviewed_at or '(unknown)'}",
                f"- **Action id**: `{action.id}`",
                "",
                "## What to do",
                "",
                action.body.strip(),
                "",
                "## Provenance",
                "",
                f"- Source: `{p.source_audio}` "
                f"[{p.start_sec:.1f}s - {p.end_sec:.1f}s, {p.duration_sec:.1f}s]",
                f"- Signal: **{p.speech_rumble_db:+.1f} dB** speech-minus-rumble",
                f"- Extractor: {p.extractor}",
                f"- Glossary terms applied: {glossary}",
                "",
                "> Transcript excerpt (verify against the audio before acting):",
                "",
                "\n".join(
                    f"> {line}" for line in p.transcript_excerpt.strip().splitlines()
                ),
                "",
            ]
        )

    def execute(self, action: ActionRecord) -> ExecutionResult:
        try:
            self.out_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise RuntimeError(
                f"cannot create hand-off directory {self.out_dir}: {exc}"
            ) from exc

        target = self.out_dir / f"{action.id[:8]}-{_slug(action.title)}.md"
        tmp = target.with_suffix(".md.tmp")
        try:
            tmp.write_text(self._render(action), encoding="utf-8")
            os.replace(tmp, target)
        except OSError as exc:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"cannot write hand-off file {target}: {exc}") from exc

        logger.info("wrote hand-off for action %s to %s", action.id, target)
        return ExecutionResult(
            ok=True,
            detail=f"wrote hand-off markdown to {target}",
            external_ref=str(target),
        )
