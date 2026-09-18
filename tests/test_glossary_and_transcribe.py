#!/usr/bin/env python3
"""Tests for the glossary tiers and the transcription safety checks.

Run:
    python -m pytest tests/ -v

The repetition-loop cases are drawn verbatim from measured Whisper output on real
low-signal audio. They are the exact strings the system must never file as fact.
"""

from __future__ import annotations

import textwrap

import pytest

from autowork.glossary import Glossary, GlossaryError, Term
from autowork.transcribe import (
    TranscriberConfig,
    TranscriptionError,
    looks_like_repetition_loop,
)


def write_glossary(tmp_path, body: str):
    path = tmp_path / "glossary.yml"
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return path


BASIC = """
    terms:
      - term: Claude
        tier: prompt
        variants: [Clark, Clode]
      - term: Customer Intelligence Tool
        tier: prompt
        variants: [personal intelligence tool]
      - term: MadCap Flare
        tier: correct
        variants: [Mad Cap Flair]
      - term: forms
        tier: correct
        variants: [forums]
    """


# --- loading ---------------------------------------------------------------------


def test_load_and_tier_split(tmp_path) -> None:
    """Happy path."""
    g = Glossary.load(write_glossary(tmp_path, BASIC))
    assert len(g.terms) == 4
    assert [t.term for t in g.prompt_terms] == ["Claude", "Customer Intelligence Tool"]


def test_missing_file_raises() -> None:
    """Expected failure."""
    with pytest.raises(GlossaryError, match="no glossary at"):
        Glossary.load("nope/glossary.yml")


def test_duplicate_term_raises(tmp_path) -> None:
    """Expected failure: two entries for one term make correction order-dependent."""
    path = write_glossary(tmp_path, """
        terms:
          - term: Claude
            tier: prompt
          - term: claude
            tier: correct
        """)
    with pytest.raises(GlossaryError, match="duplicate term"):
        Glossary.load(path)


def test_variant_colliding_with_a_canonical_term_raises(tmp_path) -> None:
    """Expected failure: this would rewrite a CORRECT word into a different one."""
    path = write_glossary(tmp_path, """
        terms:
          - term: forms
            tier: correct
            variants: [dispatch]
          - term: dispatch
            tier: correct
        """)
    with pytest.raises(GlossaryError, match="is itself a canonical term"):
        Glossary.load(path)


def test_bad_tier_raises(tmp_path) -> None:
    """Expected failure."""
    path = write_glossary(tmp_path, """
        terms:
          - term: Claude
            tier: shouty
        """)
    with pytest.raises(GlossaryError, match="expected 'prompt' or 'correct'"):
        Glossary.load(path)


# --- prompt tier -----------------------------------------------------------------


def test_prompt_contains_only_prompt_tier_terms(tmp_path) -> None:
    g = Glossary.load(write_glossary(tmp_path, BASIC))
    prompt = g.prompt_text()
    assert "Claude" in prompt
    assert "Customer Intelligence Tool" in prompt
    assert "MadCap Flare" not in prompt  # correct-tier only


def test_prompt_truncates_at_budget_and_keeps_file_order(tmp_path) -> None:
    """Edge: Whisper silently drops an over-long prompt, so we truncate deliberately."""
    g = Glossary(terms=[Term(term=f"Term{i:02d}", tier="prompt") for i in range(50)])
    prompt = g.prompt_text(budget=120)

    assert len(prompt) <= 130  # budget plus the closing punctuation
    assert "Term00" in prompt
    assert "Term49" not in prompt


def test_empty_prompt_tier_yields_empty_string() -> None:
    """Edge: no prompt-tier terms must not produce a dangling preamble."""
    assert Glossary(terms=[Term(term="thing", tier="correct")]).prompt_text() == ""


# --- correction tier -------------------------------------------------------------


def test_correction_fixes_measured_real_errors(tmp_path) -> None:
    """Happy path, using errors actually observed in this project's transcripts."""
    g = Glossary.load(write_glossary(tmp_path, BASIC))
    text = "I sent it to Clark and the personal intelligence tool checked the forums."
    fixed, applied = g.correct(text)

    assert "Claude" in fixed
    assert "Customer Intelligence Tool" in fixed
    assert "forms" in fixed
    assert "Clark" not in fixed
    assert set(applied) == {"Claude", "Customer Intelligence Tool", "forms"}


def test_correction_respects_word_boundaries(tmp_path) -> None:
    """Edge: a variant inside a longer word must not be rewritten."""
    g = Glossary.load(write_glossary(tmp_path, BASIC))
    fixed, applied = g.correct("Clarkson wrote the forums-adjacent doc.")

    assert "Clarkson" in fixed
    assert applied == ["forms"]  # "forums" matched; "Clark" inside "Clarkson" did not


def test_longer_variants_win_over_nested_shorter_ones(tmp_path) -> None:
    """Edge: multi-word variants must not be pre-empted by a shorter nested match."""
    path = write_glossary(tmp_path, """
        terms:
          - term: Customer Intelligence Tool
            tier: correct
            variants: [intelligence tool]
          - term: telemetry
            tier: correct
            variants: [tool]
        """)
    fixed, _ = Glossary.load(path).correct("the intelligence tool is fine")
    assert "Customer Intelligence Tool" in fixed


def test_correction_is_case_insensitive_and_normalises(tmp_path) -> None:
    g = Glossary.load(write_glossary(tmp_path, BASIC))
    fixed, _ = g.correct("i sent it to CLARK yesterday")
    assert "Claude" in fixed


def test_no_known_variants_leaves_text_untouched(tmp_path) -> None:
    g = Glossary.load(write_glossary(tmp_path, BASIC))
    original = "Nothing here needs any correction at all."
    fixed, applied = g.correct(original)
    assert fixed == original
    assert applied == []


def test_summary_hint_carries_context_and_variants(tmp_path) -> None:
    """The summariser already reads the whole call; this is how it learns Okta
    from an authorization discussion without a second billed pass."""
    g = Glossary.load(write_glossary(tmp_path, """
        terms:
          - term: Okta
            tier: correct
            about: [SSO, authorization]
            variants: [Octo, Octa]
          - term: Dan
            tier: prompt
            variants: []
        """))
    hint = g.summary_hint()
    assert "Okta" in hint
    assert "authorization" in hint
    assert "Octo" in hint
    assert "- Dan" not in hint  # no variants, no about: not a near-miss to teach


def test_fill_summary_prompt_injects_glossary() -> None:
    from autowork.summarize import fill_summary_prompt

    glossary = Glossary(terms=[
        Term(
            term="Okta", tier="correct",
            variants=("Octo",), about=("authorization",),
        ),
    ])
    filled = fill_summary_prompt(
        "Date {date}.\n{glossary}\n",
        date_text="Wednesday 26 August 2026",
        glossary=glossary,
    )
    assert "Wednesday 26 August 2026" in filled
    assert "{date}" not in filled
    assert "{glossary}" not in filled
    assert "Okta" in filled
    assert "authorization" in filled


def test_fill_summary_prompt_appends_when_placeholder_missing() -> None:
    """A pasted custom prompt must not silently drop the known-names block."""
    from autowork.summarize import fill_summary_prompt

    glossary = Glossary(terms=[
        Term(term="Okta", tier="correct", variants=("Octo",)),
    ])
    filled = fill_summary_prompt(
        "Just summarise this.",
        date_text="unknown",
        glossary=glossary,
    )
    assert "Just summarise this." in filled
    assert "Okta" in filled


def test_operator_summary_prompt_has_glossary_slot() -> None:
    from autowork.summarize import load_summary_prompt

    text = load_summary_prompt()
    assert "{glossary}" in text
    assert "near-miss" in text


def test_summary_prompt_drops_salad_and_keeps_named_follow_ups() -> None:
    """2026-09-03 12:10 mailed 'the one globe', CIT-for-the-other-one, and
    cell-tower records as real topics, then 'Research further' as a to-do.
    Two-sentence caps were dropping the useful half; salad was kept.
    """
    from autowork.summarize import load_summary_prompt

    text = load_summary_prompt()
    assert "at most six topics" not in text.lower()
    assert "at most two sentences" not in text.lower()
    assert "the one globe" in text
    assert "cell-tower" in text
    assert "Compress filler" in text or "not substance" in text
    assert "named object" in text.lower() or "who will do what" in text.lower()


def test_project_glossary_loads_okta_context() -> None:
    """The live glossary must teach the summariser that Octo-in-auth is Okta."""
    from pathlib import Path

    g = Glossary.load(Path(__file__).resolve().parent.parent / "config" / "glossary.yml")
    okta = next(t for t in g.terms if t.term == "Okta")
    assert "authorization" in okta.about
    assert "Octo" in okta.variants
    assert "Okta" in g.summary_hint()


# --- clarify loop ----------------------------------------------------------------


def test_rewrite_file_fixes_cached_transcript_and_is_idempotent(tmp_path) -> None:
    """Cached re-sends used to skip correct(), so Dave Cazal survived on disk."""
    g = Glossary.load(write_glossary(tmp_path, """
        terms:
          - term: Dave Casal
            tier: correct
            variants: [Dave Cazal, Cazal]
        """))
    path = tmp_path / "R2026-08-25-13-23-54.md"
    path.write_text(
        "I should take it up again with Dave Cazal maybe.\n",
        encoding="utf-8",
    )
    applied = g.rewrite_file(path)
    assert applied == ["Dave Casal"]
    text = path.read_text(encoding="utf-8")
    assert "Dave Casal" in text
    assert "Cazal" not in text
    assert g.rewrite_file(path) == []


# --- clarify loop ----------------------------------------------------------------


def test_unknown_proper_nouns_surfaces_repeated_unknown_names(tmp_path) -> None:
    """Happy path for the 'ask me to clarify' candidate list."""
    g = Glossary.load(write_glossary(tmp_path, BASIC))
    text = (
        "We looked at Cotera and then Cotera again. Claude was fine. "
        "Tesco came up once."
    )
    unknown = g.unknown_proper_nouns(text, min_occurrences=2)

    assert "Cotera" in unknown
    assert "Claude" not in unknown  # already known
    assert "Tesco" not in unknown  # only one occurrence


# --- repetition detection --------------------------------------------------------

MEASURED_HALLUCINATIONS = [
    "*Police* *Police* *Police* *Police* *Police* *Police*",
    "It's a miracle. It's a miracle. It's a miracle. It's a miracle. It's a miracle.",
    "I can read it later. I can read it later. I can read it later. I can read it later.",
    "*Evil music plays* *Evil music plays* *Evil music plays* *Evil music plays*",
    " ".join(["I'm going to go to the city of Estrella."] * 8),
    " ".join(["We're going to do a lot of things."] * 10),
]


@pytest.mark.parametrize("text", MEASURED_HALLUCINATIONS)
def test_detects_measured_hallucination_loops(text: str) -> None:
    """Every string here is real Whisper output from real low-signal audio."""
    assert looks_like_repetition_loop(text)


REAL_TRANSCRIPTS = [
    "Hey Dan. What's up Rob? How are you? Not too bad. How are you?",
    (
        "So the problem was that it was a pretty small sample size overall as far as "
        "forms that had work history and it also had submissions along with them. "
        "I used the zipped folder that you gave me with the form definitions."
    ),
    (
        "But in your example you need the customer to already have done a mapping. "
        "Correct, yeah. So I mean if you're just thinking about how to map when "
        "there's not an existing mapping, then that doesn't apply."
    ),
    "Yeah, yeah, yeah. That is, I will say I'm a little bit concerned.",
]


@pytest.mark.parametrize("text", REAL_TRANSCRIPTS)
def test_does_not_flag_real_conversation(text: str) -> None:
    """The detector must not fire on genuine speech, including natural repetition
    like 'Yeah, yeah, yeah' and a repeated question in a greeting."""
    assert not looks_like_repetition_loop(text)


def test_empty_text_is_not_a_loop() -> None:
    """Edge."""
    assert not looks_like_repetition_loop("")
    assert not looks_like_repetition_loop("   ")


# --- transcriber configuration ---------------------------------------------------


def test_config_rejects_missing_sona(tmp_path) -> None:
    """Expected failure: fail at construction, not half-way through a day's audio."""
    model = tmp_path / "model.bin"
    model.write_bytes(b"stub")
    with pytest.raises(TranscriptionError, match="sona not found"):
        TranscriberConfig(sona_path=tmp_path / "nope.exe", model_path=model)


def test_config_rejects_missing_model(tmp_path) -> None:
    """Expected failure."""
    sona = tmp_path / "sona.exe"
    sona.write_bytes(b"stub")
    with pytest.raises(TranscriptionError, match="whisper model not found"):
        TranscriberConfig(sona_path=sona, model_path=tmp_path / "nope.bin")


# --- loop excision ---------------------------------------------------------------


def test_excises_the_real_measured_tail_loop() -> None:
    """The 2026-08-25 transcript looped this ~150x after the conversation ended.

    Excision must keep the real speech on BOTH sides. Losing the sign-off, or the
    backlog-tool discussion before it, would be a worse outcome than the loop.
    """
    from autowork.transcribe import excise_repetition_loops

    before = "So yesterday he had started making epics with the backlog tool."
    after = "That's pretty much all I got, Rob, for today."
    text = f"{before} " + ("I think it's a lot of people. " * 150) + after

    cleaned, removed = excise_repetition_loops(text)

    assert removed > 140
    assert before in cleaned
    assert after in cleaned
    assert cleaned.count("I think it's a lot of people.") <= 3


def test_excision_keeps_genuine_consecutive_repetition_once() -> None:
    """Edge: a person really can say the same short thing twice. Keep it."""
    from autowork.transcribe import excise_repetition_loops

    text = "Yeah. Yeah. Yeah. That is a fair point."
    cleaned, removed = excise_repetition_loops(text)

    assert removed == 0
    assert "That is a fair point." in cleaned


def test_excision_only_collapses_consecutive_runs() -> None:
    """Edge: a phrase returned to later in a conversation is real speech, not a loop."""
    from autowork.transcribe import excise_repetition_loops

    text = (
        "I don't know. We looked at the mapping. I don't know. "
        "Then we checked the forms. I don't know. And that was that. I don't know."
    )
    cleaned, removed = excise_repetition_loops(text)

    assert removed == 0
    assert cleaned.count("I don't know.") == 4


def test_excision_leaves_clean_text_untouched() -> None:
    """Happy path: no loop, no change."""
    from autowork.transcribe import excise_repetition_loops

    text = "Hey Dan. What's up Rob? How are you? Not too bad."
    cleaned, removed = excise_repetition_loops(text)

    assert removed == 0
    assert cleaned == text


def test_a_wholly_fabricated_segment_still_trips_the_domination_check() -> None:
    """The two checks are complementary, not redundant.

    Excision removes a localised loop; the domination check is what catches a segment
    that is fabricated end to end, where excision leaves almost nothing behind.
    """
    from autowork.transcribe import excise_repetition_loops

    text = "*Police* *Police* *Police* *Police* *Police* *Police*"
    cleaned, _ = excise_repetition_loops(text)
    assert looks_like_repetition_loop(cleaned) or looks_like_repetition_loop(text)


# --- sound-event annotations -----------------------------------------------------


def test_annotation_only_segments_are_identified() -> None:
    """A segment can pass the audio gate and still contain no speech.

    Two 15s segments measured +3.2 and +3.9 dB on real audio and transcribed to
    "*Loud noise*" -- non-speech events, not conversation.
    """
    from autowork.transcribe import is_annotation_only

    assert is_annotation_only("*Loud noise*")
    assert is_annotation_only("  *Police* *Police*  ")
    assert not is_annotation_only("Hey Dan. What's up Rob?")
    assert not is_annotation_only("*Loud noise* and then he said something real.")
    assert not is_annotation_only("")


def test_clarify_list_ignores_sound_event_annotations(tmp_path) -> None:
    """Regression: a real run surfaced only 'Loud', from two '*Loud noise*' markers,
    crowding out the actual unknown names."""
    g = Glossary.load(write_glossary(tmp_path, BASIC))
    text = (
        "*Loud noise* *Loud noise* We talked about Cotera. "
        "Then Cotera came up again."
    )
    unknown = g.unknown_proper_nouns(text, min_occurrences=2)

    assert "Loud" not in unknown
    assert "Cotera" in unknown


def test_clarify_list_filters_interjections_but_keeps_mixed_case_names(tmp_path) -> None:
    """Regression from a real run.

    Two separate bugs made this list useless: a pattern requiring two lowercase letters
    after the capital missed "ViPON" and "FiveHands" entirely, and capitalised
    interjections crowded out whatever survived.
    """
    g = Glossary.load(write_glossary(tmp_path, BASIC))
    text = (
        "Yeah. I don't know if ViPON is a red herring. Hey, Callan was playing "
        "with it. Yep. And FiveHands, Calvin set that one up. Okay."
    )
    unknown = g.unknown_proper_nouns(text)

    assert "ViPON" in unknown
    assert "FiveHands" in unknown
    assert "Callan" in unknown
    assert "Calvin" in unknown
    for interjection in ("Yeah", "Hey", "Yep", "Okay"):
        assert interjection not in unknown


def test_sit_is_not_rewritten_to_cit() -> None:
    """2026-09-03 12:10: 'I have to sit around the other one' was rewritten to
    CIT, and the summary then invented a Customer Intelligence workstream from
    pocket-noise. sit is ordinary English; it must not be a CIT variant.
    """
    from pathlib import Path

    g = Glossary.load(Path(__file__).resolve().parent.parent / "config" / "glossary.yml")
    cit = next(t for t in g.terms if t.term == "CIT")
    assert "sit" not in {v.lower() for v in cit.variants}
    fixed, applied = g.correct(
        "So I have to sit around the other one. I don't know."
    )
    assert "sit around" in fixed
    assert "CIT" not in fixed
    assert "CIT" not in applied


def test_clarify_list_skips_months_god_and_known_observability_names() -> None:
    """2026-09-03 11:10 mailed Android and Sentry as terms to clarify, plus
    September/God/Cause from the same transcript. Sentry is a known system;
    month names and 'oh my God' are not names to promote.
    """
    from pathlib import Path

    g = Glossary.load(Path(__file__).resolve().parent.parent / "config" / "glossary.yml")
    text = (
        "Yeah, I think that was left over from when we were wondering whether it "
        "was going to be New Relic or Sentry or we weren't wondering but Claude "
        "was because it thought that there was an issue with Sentry data where it "
        "wasn't recording on Android what was doing on Android basically. "
        "I don't mean by mid-September. Oh my God. Cause I do have like I need "
        "my extra screen. Satish's Claude's latest status update."
    )
    unknown = g.unknown_proper_nouns(text)
    assert "Sentry" not in unknown
    assert "Android" not in unknown
    assert "September" not in unknown
    assert "God" not in unknown
    assert "Cause" not in unknown
    assert "Satish" not in unknown
