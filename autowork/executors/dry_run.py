#!/usr/bin/env python3
"""Reference executor: reports what it would do and changes nothing.

This is not a stub. It is the executor you run every day until you trust the extractor,
and the one every new adapter is diffed against. It accepts any target_system on purpose
-- its whole job is to let you route an action to a system you have not written an
adapter for yet and still see the routing decision.

It also owns the low-confidence warning, so that logic is exercised constantly rather
than only in the adapters that touch live systems.
"""

from __future__ import annotations

import logging

from autowork.action import ActionRecord
from autowork.executors.base import Executor, ExecutionResult

logger = logging.getLogger(__name__)

# Below this speech-minus-rumble figure the source transcript is, on measured evidence
# from this recorder, likely to be fabricated rather than merely inaccurate.
FABRICATION_RISK_DB = 1.0


class DryRunExecutor(Executor):
    @property
    def name(self) -> str:
        return "dry_run"

    def can_handle(self, action: ActionRecord) -> bool:
        """Handles everything. See module docstring -- this is deliberate."""
        return True

    def execute(self, action: ActionRecord) -> ExecutionResult:
        p = action.provenance
        lines = [
            f"WOULD {action.action_type.value.upper()} in {action.target_system}",
            f"  target_ref : {action.target_ref or '(none)'}",
            f"  title      : {action.title}",
            f"  confidence : {action.confidence:.2f}",
            f"  source     : {p.source_audio} [{p.start_sec:.1f}s-{p.end_sec:.1f}s]",
            f"  signal     : {p.speech_rumble_db:+.1f} dB speech-minus-rumble",
        ]
        if p.speech_rumble_db < FABRICATION_RISK_DB:
            warning = (
                f"source audio at {p.speech_rumble_db:+.1f} dB is below the "
                f"{FABRICATION_RISK_DB:+.1f} dB fabrication threshold; "
                f"treat this text as unverified"
            )
            lines.append(f"  WARNING    : {warning}")
            logger.warning("action %s: %s", action.id, warning)

        detail = "\n".join(lines)
        logger.info("dry run for action %s in %s", action.id, action.target_system)
        return ExecutionResult(ok=True, detail=detail, external_ref=None)
