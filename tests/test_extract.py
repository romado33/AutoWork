#!/usr/bin/env python3
"""Tests for the extractor's anti-fabrication logic.

Run:
    python -m pytest tests/ -v

No Ollama required: everything here tests the parsing, grounding and validation that
stands between a model's output and the review queue. That is deliberate -- the model
is swappable, but these checks are what make any model's output safe to file.

The grounding test is the important one. A model can invent an action; it cannot easily
invent an action whose supporting quote survives a check against the actual transcript.
"""

from __future__ import annotations

import json

import pytest

from autowork.action import ActionType
from autowork.extract import (
    MIN_QUOTE_WORDS,
    Candidate,
    ExtractionError,
    ExtractorConfig,
    _dedupe_by_quote,
    _dedupe_by_title,
    chunk_transcript,
    ground,
    parse_candidates,
    titles_overlap,
    to_action_record,
)
from autowork.gate import Verdict
from autowork.transcribe import TranscribedSegment

# A real passage from the 2026-08-25 recording.
TRANSCRIPT = (
    "I've kind of left the backlog tool off this week because Andrew was away. "
    "So I kind of left that in the state where it was, but I'm thinking I should "
    "take it up again with Dave Casale maybe and just get it finalized as far as "
    "how it works. But I mean, it is working. The only thing I was thinking about "
    "that maybe is worth exploring is making sure when we ask Claude about a feature "
    "or a customer need that it surfaces quotes as it usually does."
)


def config(**overrides) -> ExtractorConfig:
    defaults = dict(target_systems=("backlog-tool", "cit", "notes"), owner_name="Rob")
    defaults.update(overrides)
    return ExtractorConfig(**defaults)


def candidate(**overrides) -> Candidate:
    defaults = dict(
        title="Finalise the backlog tool with Dave Casale",
        body="Take the backlog tool up again and get it finalised.",
        owner="me",
        target_system="backlog-tool",
        action_type="task",
        confidence=0.8,
        quote="I should take it up again with Dave Casale",
    )
    defaults.update(overrides)
    return Candidate(**defaults)


def segment(text: str = TRANSCRIPT) -> TranscribedSegment:
    return TranscribedSegment(
        source_audio="D:/RECORD/R2026-08-25-13-23-54.MP3",
        start_sec=930.0,
        end_sec=2280.0,
        verdict=Verdict.CLEAN,
        speech_rumble_db=6.6,
        text=text,
        glossary_applied=["Claude", "forms"],
    )


# --- grounding: the anti-fabrication check ---------------------------------------


def test_grounded_candidate_is_accepted() -> None:
    """Happy path: a quote genuinely present in the transcript."""
    c = candidate()
    ground([c], TRANSCRIPT, config())
    assert c.rejected_because is None


def test_fabricated_quote_is_rejected() -> None:
    """THE test. A plausible action nobody actually said must not reach the queue."""
    c = candidate(
        title="Migrate the database to Postgres",
        quote="I'll migrate the database to Postgres by Friday",
    )
    ground([c], TRANSCRIPT, config())
    assert c.rejected_because == "quote not found in transcript (fabricated)"


def test_grounding_tolerates_punctuation_differences() -> None:
    """Edge: reproducing a quote without its comma is not fabrication."""
    c = candidate(quote="I should take it up again, with Dave Casale!")
    ground([c], TRANSCRIPT, config())
    assert c.rejected_because is None


def test_grounding_is_case_insensitive() -> None:
    """Edge."""
    c = candidate(quote="i SHOULD take IT up again with dave casale")
    ground([c], TRANSCRIPT, config())
    assert c.rejected_because is None


@pytest.mark.parametrize(
    "quote",
    [
        "Yeah",
        "Okay then",
        # Five words, and genuinely present in the transcript. It passed the original
        # five-word threshold and would have grounded an entirely fabricated action.
        "But I mean, it is",
    ],
)
def test_trivial_quotes_are_rejected(quote: str) -> None:
    """Expected failure: a quote that matches everything grounds nothing.

    "Yeah" appears in every transcript, so a substring check against it always passes
    and provides no evidence at all.
    """
    c = candidate(quote=quote)
    ground([c], TRANSCRIPT, config())
    assert c.rejected_because is not None
    assert "too short to verify" in c.rejected_because


def test_quote_at_exactly_the_word_threshold_is_accepted() -> None:
    """Edge: the boundary itself must not be rejected."""
    words = "I should take it up again with Dave".split()
    assert len(words) == MIN_QUOTE_WORDS
    c = candidate(quote=" ".join(words))
    ground([c], TRANSCRIPT, config())
    assert c.rejected_because is None


# --- vocabulary and validation ---------------------------------------------------


def test_invented_target_system_is_rejected() -> None:
    """Expected failure: the model must not choose a system to write to."""
    c = candidate(target_system="production-database")
    ground([c], TRANSCRIPT, config())
    assert c.rejected_because is not None
    assert "unknown target_system" in c.rejected_because


def test_invented_action_type_is_rejected() -> None:
    """Expected failure."""
    c = candidate(action_type="delete_everything")
    ground([c], TRANSCRIPT, config())
    assert c.rejected_because is not None
    assert "unknown action_type" in c.rejected_because


def test_low_confidence_is_rejected() -> None:
    """Expected failure."""
    c = candidate(confidence=0.1)
    ground([c], TRANSCRIPT, config(min_confidence=0.3))
    assert c.rejected_because is not None
    assert "below" in c.rejected_because


@pytest.mark.parametrize(("field", "expected"), [("title", "no title"), ("body", "no body")])
def test_empty_required_fields_are_rejected(field: str, expected: str) -> None:
    """Expected failure."""
    c = candidate(**{field: "   "})
    ground([c], TRANSCRIPT, config())
    assert c.rejected_because == expected


def test_nothing_is_silently_dropped() -> None:
    """Every candidate comes back, failures annotated. A model that proposes only
    garbage must look different from a quiet day with no action items."""
    candidates = [candidate(), candidate(quote="never said this anywhere at all")]
    returned = ground(candidates, TRANSCRIPT, config())

    assert len(returned) == 2
    assert sum(1 for c in returned if c.rejected_because) == 1


# --- parsing ---------------------------------------------------------------------


def test_parses_well_formed_reply() -> None:
    """Happy path."""
    raw = json.dumps({"actions": [{
        "title": "t", "body": "b", "owner": "me", "target_system": "cit",
        "action_type": "update", "confidence": 0.7, "quote": "q",
    }]})
    parsed = parse_candidates(raw, config())
    assert len(parsed) == 1
    assert parsed[0].confidence == 0.7


def test_empty_action_list_is_valid_not_an_error() -> None:
    """Edge: "nothing to do here" is the correct answer most of the time."""
    assert parse_candidates('{"actions": []}', config()) == []


def test_malformed_json_raises() -> None:
    """Expected failure: must not be mistaken for 'no actions found'."""
    with pytest.raises(ExtractionError, match="did not return valid JSON"):
        parse_candidates("here you go: {actions:", config())


def test_missing_actions_key_raises() -> None:
    """Expected failure."""
    with pytest.raises(ExtractionError, match="'actions' key"):
        parse_candidates('{"items": []}', config())


def test_one_bad_entry_does_not_discard_the_good_ones() -> None:
    """Edge: a single malformed entry must not lose a chunk's real findings."""
    raw = json.dumps({"actions": [
        "not an object",
        {"title": "good", "body": "b", "owner": "me", "target_system": "cit",
         "action_type": "task", "confidence": 0.9, "quote": "q"},
        {"title": "bad conf", "confidence": "very high"},
    ]})
    parsed = parse_candidates(raw, config())
    assert [c.title for c in parsed] == ["good"]


def test_out_of_range_confidence_is_clamped() -> None:
    """Edge: a model claiming 1.4 confidence should not break ActionRecord validation."""
    raw = json.dumps({"actions": [{
        "title": "t", "body": "b", "owner": "me", "target_system": "cit",
        "action_type": "task", "confidence": 1.4, "quote": "q",
    }]})
    assert parse_candidates(raw, config())[0].confidence == 1.0


# --- chunking --------------------------------------------------------------------


def test_short_transcript_is_one_chunk() -> None:
    assert chunk_transcript(TRANSCRIPT, size=10_000) == [TRANSCRIPT]


def test_empty_transcript_yields_no_chunks() -> None:
    """Edge: must not send an empty prompt to the model."""
    assert chunk_transcript("   ", size=1000) == []


def test_long_transcript_chunks_with_overlap() -> None:
    """Happy path: overlap means a commitment on a boundary appears in both chunks,
    and the queue's dedupe key collapses the duplicate rather than filing it twice."""
    text = " ".join(f"Sentence number {i} about the work." for i in range(400))
    chunks = chunk_transcript(text, size=2000, overlap=400)

    assert len(chunks) > 1
    assert all(len(c) <= 2600 for c in chunks)
    first_tail = chunks[0][-200:].split()
    assert any(word in chunks[1] for word in first_tail if len(word) > 4)


# --- record construction ---------------------------------------------------------


def test_record_carries_provenance_back_to_the_audio() -> None:
    """Happy path: an action must be traceable to the moment it came from."""
    record = to_action_record(candidate(), segment(), config())

    assert record.provenance.source_audio.endswith("R2026-08-25-13-23-54.MP3")
    assert record.provenance.start_sec == 930.0
    assert record.provenance.speech_rumble_db == 6.6
    assert record.provenance.glossary_terms_applied == ("Claude", "forms")
    assert record.action_type is ActionType.TASK


def test_attribution_to_someone_else_is_flagged_and_capped() -> None:
    """Speaker attribution is inferred from context, not diarization, which measured
    unusable on this audio. A record must not present that guess as certain."""
    record = to_action_record(candidate(owner="other", confidence=0.95), segment(), config())

    assert record.confidence <= 0.6
    assert "someone else" in record.body


def test_unclear_attribution_is_flagged() -> None:
    """Edge."""
    record = to_action_record(candidate(owner="unclear", confidence=0.9), segment(), config())
    assert "attribution unclear" in record.body


def test_overlong_title_is_truncated_not_rejected() -> None:
    """Edge: ActionRecord has no length limit, but a queue is unreadable with essays."""
    record = to_action_record(candidate(title="x" * 200), segment(), config())
    assert len(record.title) == 80


def test_chunk_overlap_duplicates_collapse_by_quote_containment() -> None:
    """Regression from a real run: the commitment straddling a chunk boundary was
    extracted twice with nested quote spans. Containment means the same spoken moment;
    the higher-confidence record survives."""
    short = to_action_record(
        candidate(confidence=0.88,
                  quote="only the top, I don't know, five would be used"),
        segment(), config(),
    )
    longer = to_action_record(
        candidate(confidence=0.90,
                  quote="only the top, I don't know, five would be used and then in the"),
        segment(), config(),
    )
    unrelated = to_action_record(
        candidate(title="Different action",
                  quote="an entirely different commitment about the docs"),
        segment(), config(),
    )

    survivors = _dedupe_by_quote([short, longer, unrelated])
    assert len(survivors) == 2
    quotes = [s.provenance.transcript_excerpt for s in survivors]
    assert any("and then in the" in q for q in quotes), "kept the higher-confidence one"
    assert any("different commitment" in q for q in quotes)


def test_casal_title_pair_is_the_same_todo() -> None:
    """Real digest duplicate: two titles, two quotes, one piece of work."""
    assert titles_overlap(
        "Finalize backlog tool design with Dave Casal",
        "Finalize backlog tool with Dave Casal",
    )


def test_distinct_commitments_do_not_collapse_by_title() -> None:
    """Andrew vs Joe, or quote-threshold vs Casal, must stay two items."""
    assert not titles_overlap(
        "Update Okta permissions for Andrew",
        "Update Okta permissions for Joe",
    )
    assert not titles_overlap(
        "Add quote quality threshold",
        "Set Salesforce quote confidence threshold",
    )
    assert not titles_overlap(
        "Finalize backlog tool with Dave Casal",
        "Check the backlog tool permissions",
    )


def test_same_run_title_clones_collapse_keeping_higher_confidence() -> None:
    keeper = to_action_record(
        candidate(
            title="Finalize backlog tool design with Dave Casal",
            confidence=0.96,
            quote="I should take it up again with Dave Casal maybe and just get it finalized",
        ),
        segment(), config(),
    )
    clone = to_action_record(
        candidate(
            title="Finalize backlog tool with Dave Casal",
            confidence=0.95,
            quote="But I mean it is working. It doesn't happen like that a lot. It's making sure",
        ),
        segment(), config(),
    )
    survivors = _dedupe_by_title([clone, keeper])
    assert len(survivors) == 1
    assert survivors[0].title.startswith("Finalize backlog tool design")
