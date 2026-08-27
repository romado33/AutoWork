#!/usr/bin/env python3
"""Cheap checks on a transcript: is this a conversation, or Whisper inventing one?

The audio gate asks whether the spectrum looks like speech. The relevance gate asks
whether the words are a work conversation. This module asks a third question, added
after a real failure on 2026-08-27:

    A 40-minute pocket recording of two people (plus maybe two background voices)
    was mailed as a 13-participant meeting about satellite-launch economics. The
    diarizer labelled one 10-minute upload A–L plus '@'. Whisper looped
    "I'm not sure if like the I'll start ahead and see if I can" 59 times with no
    period, and the summariser faithfully wrote up the fluent nonsense.

None of that is a question the audio gate can answer (the stretch measured CLEAN at
+4.9 dB), and the relevance classifier kept it "despite garbling" because work words
appeared. These checks are regex and counts. They run before any summariser or
extractor call.

Do not raise the audio-gate thresholds from this incident: those are calibrated
against labelled samples in tests/test_gate.py.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from autowork.transcribe import (
    excise_consecutive_phrase_runs,
    excise_repetition_loops,
    looks_like_repetition_loop,
)

logger = logging.getLogger(__name__)

# Pocket recorder, typically 2–4 people. A real 2-person call measured as 3–4 labels
# in one upload. 13 labels on 2026-08-27 11:30 was the diarizer fragmenting a noisy
# room, not a headcount. Above this, the count is not a fact we can print.
MAX_PLAUSIBLE_SPEAKERS = 6

# Consecutive copies of a 4+ word phrase. 20 is never a person; the 2026-08-27
# intra-sentence loop was x59. A 2-person meeting that then looped
# "I think it's a lot of people." ~150 times is sentence-level and is excised
# BEFORE this threshold is applied, so that conversation is not dropped.
ALWAYS_UNINTELLIGIBLE_PHRASE_RUN = 20

# Lower bar when the diarizer is already over-counting: a loop of 8+ copies plus
# 5+ voice labels in one upload is the decoder losing lock, not a 5-person standup
# that happened to repeat itself.
NOISY_ROOM_PHRASE_RUN = 8
NOISY_ROOM_SPEAKERS = 5

_VALID_SPEAKER = re.compile(r"^(?:[A-Z]|SPEAKER[_ ]?\d+)$", re.IGNORECASE)


@dataclass(frozen=True)
class Cleaning:
    text: str
    sentence_removed: int
    phrase_copies_removed: int
    longest_phrase_run: int


def valid_speaker_label(label: object) -> bool:
    """Diarizer labels we will count. '@' was emitted as a speaker on 2026-08-27."""
    if not isinstance(label, str):
        return False
    return bool(_VALID_SPEAKER.fullmatch(label.strip()))


def distinct_speaker_count(turns: list) -> int:
    labels = {
        str(turn.get("speaker", "")).strip()
        for turn in turns
        if isinstance(turn, dict) and valid_speaker_label(turn.get("speaker"))
    }
    return len(labels)


def speaker_count_for_header(segments: list) -> int:
    """Max valid labels in any one segment, or 0 if that max is not believable.

    Zero means the email omits the Participants line rather than printing 13.
    Labels are per request: never union across uploads.
    """
    n = 0
    for seg in segments:
        turns = getattr(seg, "speakers", None) or []
        n = max(n, distinct_speaker_count(turns))
    if n > MAX_PLAUSIBLE_SPEAKERS:
        return 0
    return n


def clean_hallucinated_text(text: str) -> Cleaning:
    """Sentence loops first, then intra-sentence n-gram runs."""
    after_sentences, sent_removed = excise_repetition_loops(text)
    cleaned, phrase_removed, longest = excise_consecutive_phrase_runs(after_sentences)
    return Cleaning(
        text=cleaned,
        sentence_removed=sent_removed,
        phrase_copies_removed=phrase_removed,
        longest_phrase_run=longest,
    )


def unintelligible_reason(
    text: str = "",
    turns: list | None = None,
    cleaning: Cleaning | None = None,
) -> str | None:
    """Why this segment must not be summarised or extracted, or None to keep it."""
    cleaning = cleaning or clean_hallucinated_text(text)
    n_speakers = distinct_speaker_count(turns or [])

    if n_speakers > MAX_PLAUSIBLE_SPEAKERS:
        return (
            f"diarizer reported {n_speakers} voices in one upload "
            f"(pocket recordings over-count; 13 on a 2-person recording, 2026-08-27)"
        )
    if looks_like_repetition_loop(cleaning.text):
        return "repetition loop still dominates after excision"
    if not cleaning.text.strip():
        return "nothing left after removing hallucination loops"
    if cleaning.longest_phrase_run >= ALWAYS_UNINTELLIGIBLE_PHRASE_RUN:
        return (
            f"Whisper looped a {cleaning.longest_phrase_run}-copy phrase with no "
            f"sentence boundary (decoder lost lock)"
        )
    if (
        cleaning.longest_phrase_run >= NOISY_ROOM_PHRASE_RUN
        and n_speakers >= NOISY_ROOM_SPEAKERS
    ):
        return (
            f"Whisper looped a phrase {cleaning.longest_phrase_run} times and the "
            f"diarizer split {n_speakers} voices"
        )
    return None


def assess(text: str, turns: list | None = None) -> tuple[str, str | None]:
    """Clean loops and decide whether the remainder is usable conversation.

    Returns (cleaned_text, reason_if_unintelligible).
    """
    cleaning = clean_hallucinated_text(text)
    reason = unintelligible_reason(turns=turns, cleaning=cleaning)
    if reason:
        logger.warning("unintelligible: %s", reason)
    elif cleaning.sentence_removed or cleaning.phrase_copies_removed:
        logger.warning(
            "removed %d looped sentence(s) and %d looped phrase-cop(ies); "
            "surrounding speech is kept",
            cleaning.sentence_removed,
            cleaning.phrase_copies_removed,
        )
    return cleaning.text, reason


def retain_intelligible(segments: list) -> tuple[list, list[tuple[object, str]]]:
    """Split segments into those safe to summarise/extract vs those to drop.

    Dropped segments stay in the transcript file (already paid for). Mutates
    ``.text`` in place when loops were excised so downstream never sees them.
    """
    kept: list = []
    dropped: list[tuple[object, str]] = []
    for seg in segments:
        turns = getattr(seg, "speakers", None) or []
        prior = (getattr(seg, "unintelligible_because", "") or "").strip()
        cleaned, reason = assess(seg.text, turns)
        if cleaned != seg.text:
            seg.text = cleaned
        reason = prior or reason
        if reason:
            if hasattr(seg, "unintelligible_because"):
                seg.unintelligible_because = reason
            dropped.append((seg, reason))
        else:
            kept.append(seg)
    return kept, dropped
