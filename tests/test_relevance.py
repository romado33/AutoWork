#!/usr/bin/env python3
"""Relevance keep/drop is the semantic nonsense veto. No network: we test the
verdict rule against the exact 2026-08-27 11:30 classifier output."""

from __future__ import annotations

from autowork.relevance import KEEP_THRESHOLD, PROMPT, RELEVANCE_SCHEMA, Verdict


def test_schema_requires_intelligible() -> None:
    """Without this field the model can hide nonsense under is_work=true."""
    assert "intelligible" in RELEVANCE_SCHEMA["required"]
    assert "intelligible" in RELEVANCE_SCHEMA["properties"]


def test_prompt_asks_intelligibility_before_work() -> None:
    """The 11:30 miss answered 'is it work' and skipped 'does it make sense'."""
    assert "TWO questions" in PROMPT
    assert "intelligible=false" in PROMPT
    assert "despite garbling" in PROMPT


def test_eleven_thirty_work_despite_garbling_must_drop() -> None:
    """Verbatim reason from logs/run-20260827-131248.log, KEEP at 0.92.

    The cheap loop/speaker checks may not catch a fluent salad chunk. The semantic
    veto has to, once the model admits it cannot follow the thread.
    """
    verdict = Verdict(
        is_work=True,
        confidence=0.92,
        kind="work_conversation",
        reason=(
            "Despite garbling, the transcript repeatedly discusses projects, "
            "contracts, staff, products, meetings, and business decisions."
        ),
        intelligible=False,
    )
    assert not verdict.keep


def test_kind_unintelligible_drops_even_if_labelled_work() -> None:
    """Inconsistent model output must not mail. Kind vetoes is_work."""
    verdict = Verdict(
        is_work=True,
        confidence=0.95,
        kind="unintelligible",
        reason="It's a messy but coherent discussion about jobs.",
        intelligible=True,
    )
    assert not verdict.keep


def test_rambling_real_work_conversation_is_kept() -> None:
    """Filler and false starts are normal speech, not a reason to drop the day."""
    verdict = Verdict(
        is_work=True,
        confidence=0.99,
        kind="work_conversation",
        reason="Two people discussing form-definition mapping accuracy with Claude.",
        intelligible=True,
    )
    assert verdict.keep


def test_uncertain_not_work_is_still_kept() -> None:
    """Low-confidence 'not work' must not discard a real conversation."""
    assert KEEP_THRESHOLD == 0.35
    verdict = Verdict(
        is_work=False,
        confidence=0.20,
        kind="other",
        reason="Too short to tell.",
        intelligible=True,
    )
    assert verdict.keep


def test_confident_podcast_is_dropped() -> None:
    verdict = Verdict(
        is_work=False,
        confidence=0.98,
        kind="media_playback",
        reason="A single fluent narrator introducing a series.",
        intelligible=True,
    )
    assert not verdict.keep
