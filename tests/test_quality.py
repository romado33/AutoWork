#!/usr/bin/env python3
"""Cheap intelligibility checks, calibrated on the 2026-08-27 11:30 failure.

That recording was two people (plus maybe two background voices). The diarizer
labelled one 10-minute upload A–L plus '@', Whisper looped a 14-word phrase 59
times with no period, and the summariser mailed it as a 13-person satellite meeting.
"""

from __future__ import annotations

from autowork.quality import (
    ALWAYS_UNINTELLIGIBLE_PHRASE_RUN,
    MAX_PLAUSIBLE_SPEAKERS,
    assess,
    distinct_speaker_count,
    speaker_count_for_header,
    unintelligible_reason,
    valid_speaker_label,
)
from autowork.transcribe import excise_consecutive_phrase_runs, excise_repetition_loops

# Verbatim from transcripts/R2026-08-27-11-30-55.md (the intra-sentence loop).
START_AHEAD = "I'm not sure if like the I'll start ahead and see if I can"

# Verbatim grounding quotes that were queued from the same garbage transcript.
TWO_PAGE_HOLES = "Separate two-page holes and then one-page duty."
SWITCHING_SOFTWARE = (
    "Yeah, we'll definitely have the right you have to account for switching software."
)

# Real 2026-08-25 meeting speech, then the measured tail loop that must NOT drop it.
REAL_MEETING = (
    "So the problem was that it was a pretty small sample size overall as far as "
    "forms that had work history and it also had submissions along with them. "
    "I used the zipped folder that you gave me with the form definitions and then "
    "I used the OpenSearch API to look for submissions that went along with those."
)
TAIL_LOOP = "I think it's a lot of people. "


def _turns(*labels: str) -> list[dict]:
    return [{"speaker": label, "text": "x"} for label in labels]


def test_at_sign_is_not_a_speaker() -> None:
    """The 11:30 upload listed '@, A, B, ... L'. '@' is not a voice."""
    assert not valid_speaker_label("@")
    assert not valid_speaker_label("")
    assert valid_speaker_label("A")
    assert valid_speaker_label("L")
    assert distinct_speaker_count(_turns("@", "A", "B")) == 2


def test_thirteen_labels_are_not_a_participant_count() -> None:
    """Max-within-one-chunk was already the rule; 13 in one chunk is still garbage."""
    labels = ["@"] + [chr(ord("A") + i) for i in range(12)]
    assert len(labels) == 13

    class Seg:
        speakers = _turns(*labels)

    assert distinct_speaker_count(Seg.speakers) == 12
    assert 12 > MAX_PLAUSIBLE_SPEAKERS
    assert speaker_count_for_header([Seg]) == 0
    assert unintelligible_reason("Are you guys all set for food?", turns=Seg.speakers)


def test_two_person_call_is_not_flagged() -> None:
    """The 2026-08-25 mapping meeting diarized as A/B/C. That must still mail."""
    turns = _turns("A", "B", "C")
    cleaned, reason = assess(REAL_MEETING, turns)
    assert reason is None
    assert "OpenSearch" in cleaned
    assert speaker_count_for_header([type("S", (), {"speakers": turns})()]) == 3


def test_tail_loop_on_a_real_meeting_is_excised_not_dropped() -> None:
    """2026-08-25: excellent conversation, then 'I think it's a lot of people.' x150.

    Dropping the whole segment would discard the meeting. Excise the loop, keep
    the speech, two speakers.
    """
    text = REAL_MEETING + " " + (TAIL_LOOP * 150)
    cleaned, reason = assess(text, _turns("A", "B"))
    assert reason is None
    assert "OpenSearch" in cleaned
    assert cleaned.count("I think it's a lot of people.") <= 3


def test_intra_sentence_loop_from_1130_is_unintelligible() -> None:
    """No period, so sentence excision never fired. 59 copies measured; 20 is the floor."""
    text = " ".join([START_AHEAD] * ALWAYS_UNINTELLIGIBLE_PHRASE_RUN)
    cleaned, removed, longest = excise_consecutive_phrase_runs(text)
    assert longest >= ALWAYS_UNINTELLIGIBLE_PHRASE_RUN
    assert removed > 0
    assert cleaned.lower().count("start ahead") <= 3

    _, reason = assess(text, _turns("A", "B"))
    assert reason is not None
    assert "looped" in reason


def test_five_voices_plus_a_short_phrase_run_is_a_noisy_room() -> None:
    """2085s–2685s on 11:30 had 5 valid labels and the start-ahead loop."""
    text = " ".join([START_AHEAD] * 8)
    reason = unintelligible_reason(text, turns=_turns("A", "B", "C", "D", "E"))
    assert reason is not None


def test_queued_garbage_quotes_do_not_pass_on_their_own() -> None:
    """Those quotes are real substrings of the garbage; a 2-person utterance is fine.

    They become unintelligible only with the loop / speaker explosion around them.
    """
    for quote in (TWO_PAGE_HOLES, SWITCHING_SOFTWARE):
        assert unintelligible_reason(quote, turns=_turns("A", "B")) is None


def test_topic_salad_with_thirteen_speakers_is_dropped() -> None:
    """The satellite stretch looked like a meeting. The speaker count says it isn't."""
    salad = (
        "Yeah, I mean right now they can sign like a two point seven trillion "
        "dollar deal. So like Starlink is kind of a weird relationship. "
        "This guy's bringing me Nintendo and it's cool, this guy's bringing me "
        "PlayStation. We should go and line up one-to-one to reflect everything."
    )
    labels = [chr(ord("A") + i) for i in range(12)]
    _, reason = assess(salad, _turns(*labels))
    assert reason is not None
    assert "13" in reason or "12" in reason or "voices" in reason


def test_assess_refuses_the_1130_loop_without_a_model() -> None:
    """Cheap backstop: leftover salad must not reach the summariser as a meeting."""
    text = " ".join([START_AHEAD] * ALWAYS_UNINTELLIGIBLE_PHRASE_RUN)
    _, reason = assess(text, [])
    assert reason is not None
    assert "looped" in reason


def test_yeah_yeah_yeah_is_not_a_phrase_run() -> None:
    """Natural repetition is shorter than the 4-word minimum."""
    text = "Yeah yeah yeah yeah yeah. That is a fair point."
    cleaned, removed, longest = excise_consecutive_phrase_runs(text)
    assert removed == 0
    assert "fair point" in cleaned
    assert longest < ALWAYS_UNINTELLIGIBLE_PHRASE_RUN


def test_relevance_prompt_treats_topic_salad_as_unintelligible() -> None:
    """The 11:30 classifier kept the recording 'despite garbling'. The prompt must
    now name that failure so a future edit cannot silently drop the distinction."""
    from autowork.relevance import PROMPT

    assert "topic-salad" in PROMPT
    assert "despite garbling" in PROMPT
    assert "intelligible=false" in PROMPT


def test_sentence_excision_still_handles_the_measured_tail() -> None:
    """Phrase-run excision must not replace sentence excision; they stack."""
    before = "So yesterday he had started making epics with the backlog tool."
    after = "That's pretty much all I got, Rob, for today."
    text = f"{before} " + ("I think it's a lot of people. " * 150) + after
    cleaned, removed = excise_repetition_loops(text)
    assert removed > 140
    assert before in cleaned
    assert after in cleaned
