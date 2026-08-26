#!/usr/bin/env python3
"""Tests for the commitment-language prefilter.

Run:
    python -m pytest tests/ -v

Two things matter here and they pull against each other: RECALL (a real commitment must
not be dropped, because the extractor never sees what the filter skips) and VOLUME (the
whole point is sending less to a slow model). The recall tests use verbatim sentences
from a real transcript whose action items are known.
"""

from __future__ import annotations

import pytest

from autowork.prefilter import (
    CUES,
    EXCLUSIONS,
    _COMPILED,
    FilterResult,
    filtered_text,
    match_cues,
    select,
    split_sentences,
)

# Verbatim from the 2026-08-25 recording. All three are genuine action items.
REAL_COMMITMENTS = [
    "I'm thinking I should take it up again with Dave Casale maybe and just get it "
    "finalized as far as how it works.",
    "Just making sure that the quote aligns to, like, anyone reading the quote could "
    "see how that quote is tied to the request.",
    "And then in the one in Salesforce, it should just be the top one or any that are "
    "over like 0.9 or something.",
]

# Also verbatim, and all pure discourse filler that fired the cues before exclusion.
REAL_FILLER = [
    "I'm going to be honest, I've been so slammed that I have not had a chance to read "
    "a lot of this yet.",
    "And overall, can you see, is this big enough for you to see?",
    "Let me just focus in.",
    "That is, uh, I am, I will say I'm a little bit concerned on like, some of the "
    "bottom ones.",
    "Um, here I go. I'll just go to our team.",
    "Well, I was gonna say maybe you could delineate by what's from a data destination.",
]


# --- the guard that matters most --------------------------------------------------


def test_patterns_kept_their_word_boundary_escapes() -> None:
    """A lost backslash turns \\b into chr(8), which matches nothing and raises nothing.

    This actually happened during development: every exclusion silently stopped working
    and the only symptom was a filter that kept too much. The module asserts this at
    import time; this test makes the failure legible if the assert is ever removed.
    """
    assert "\x08" not in EXCLUSIONS.pattern
    for name, compiled in _COMPILED.items():
        assert "\x08" not in compiled.pattern, f"{name} lost its escapes"
        assert "\\b" in compiled.pattern, f"{name} has no word boundaries"


# --- recall: the expensive kind of wrong ------------------------------------------


@pytest.mark.parametrize("sentence", REAL_COMMITMENTS, ids=["dave", "quote", "salesforce"])
def test_real_commitments_are_selected(sentence: str) -> None:
    """Happy path. A dropped commitment never reaches the queue and cannot be recovered
    except by re-running, so these are the tests that must not regress."""
    assert match_cues(sentence), f"real commitment not matched: {sentence[:60]}"


def test_impersonal_obligation_is_caught() -> None:
    """Regression: 'it should just be the top one' was dropped by an earlier version
    that only looked for first-person and we/you phrasing."""
    assert "shared_or_assigned" in match_cues(
        "it should just be the top one or any that are over like 0.9"
    )


# --- precision: the cheap kind of wrong -------------------------------------------


@pytest.mark.parametrize("sentence", REAL_FILLER)
def test_discourse_filler_is_excluded(sentence: str) -> None:
    """Every string here is real speech that fired a cue and is not a commitment."""
    assert match_cues(sentence) == (), f"filler was selected: {sentence[:60]}"


def test_weak_hypotheticals_are_not_cues() -> None:
    """Bare 'I can' / 'we could' / 'you could' are hypothetical in discussion, and the
    extractor prompt refuses hypotheticals anyway, so passing them buys nothing."""
    joined = " ".join(p for group in CUES.values() for p in group)
    for weak in (r"\bI can\b", r"\bwe could\b", r"\byou could\b"):
        assert weak not in joined


# --- selection and volume ---------------------------------------------------------


def test_selects_with_context_around_the_match() -> None:
    """Happy path: 'I'll do that' alone is useless; the neighbouring sentences carry
    the detail the extractor needs to write a usable body."""
    text = (
        "The mapping report came back yesterday. I'll send you the numbers. "
        "They cover about eight hundred submissions."
    )
    result = select(text, context_sentences=1)

    assert len(result.passages) == 1
    assert "mapping report" in result.passages[0].text
    assert "eight hundred submissions" in result.passages[0].text


def test_zero_context_selects_only_the_matching_sentence() -> None:
    """Edge."""
    text = "Nothing here. I'll send you the numbers. Nothing here either."
    result = select(text, context_sentences=0)

    assert len(result.passages) == 1
    assert result.passages[0].text == "I'll send you the numbers."


def test_overlapping_windows_are_merged() -> None:
    """Edge: a dense run of commitments must become one passage, not several
    near-duplicates -- duplicated context is volume paid for twice."""
    text = (
        "I'll do the first thing. I'll do the second thing. I'll do the third thing."
    )
    result = select(text, context_sentences=1)
    assert len(result.passages) == 1


def test_distant_matches_stay_separate() -> None:
    """Edge: unrelated moments must not be glued into one passage, which would invite
    the extractor to invent a connection between them."""
    filler = " ".join(f"This is sentence {i} about nothing." for i in range(10))
    text = f"I'll do the first thing. {filler} I'll do the second thing."
    result = select(text, context_sentences=1)
    assert len(result.passages) == 2


def test_transcript_with_no_commitments_selects_nothing() -> None:
    """Edge: 'nothing to do here' is a real answer and must not be an error."""
    text = (
        "The weather was pretty bad this morning. Traffic was slow on the bridge. "
        "That podcast about the trial was interesting."
    )
    result = select(text)

    assert result.passages == []
    assert result.matched_sentences == 0
    assert filtered_text(text) == ""


def test_empty_input_is_handled() -> None:
    """Edge: whitespace-only input must not crash. Reduction is 1.0 (nothing kept),
    which is correct; the 0.0 guard is for a genuinely empty FilterResult."""
    result = select("   ")
    assert result.passages == []
    assert result.matched_sentences == 0


def test_reduction_and_speedup_are_reported() -> None:
    """The numbers are the justification for the whole module, so they must be right."""
    filler = " ".join(f"This is sentence {i} about nothing at all." for i in range(40))
    result = select(f"{filler} I'll send you the numbers. {filler}", context_sentences=0)

    assert 0.9 < result.reduction < 1.0
    assert result.speedup > 10
    assert result.selected_chars < result.original_chars


def test_cue_counts_attribute_selections_to_categories() -> None:
    """Reporting which patterns fired is what makes this tunable from real misses."""
    text = "I'll send it tomorrow. Can you review the mapping? We should file a ticket."
    counts = select(text, context_sentences=0).cue_counts()

    assert counts["first_person_commitment"] >= 1
    assert counts["request_of_someone"] >= 1
    assert counts["shared_or_assigned"] >= 1


def test_filtered_text_separates_passages_blankline() -> None:
    """Passages must read as distinct moments, not one continuous conversation."""
    filler = " ".join(f"Sentence {i} of nothing." for i in range(10))
    text = f"I'll do the first thing. {filler} I'll do the second thing."
    assert "\n\n" in filtered_text(text, context_sentences=0)


def test_split_sentences_drops_blanks() -> None:
    assert split_sentences("One. Two.  Three.") == ["One.", "Two.", "Three."]


def test_empty_result_reduction_is_zero_not_a_crash() -> None:
    """Edge: division by zero on an empty transcript."""
    assert FilterResult().reduction == 0.0
    assert FilterResult().speedup == float("inf")
