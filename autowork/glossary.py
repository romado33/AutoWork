#!/usr/bin/env python3
"""Domain glossary in two tiers, because Whisper's decode-time prompt is small.

TIER 1 -- prompt. Injected as `sona transcribe --prompt`. Biases decoding, so it can fix
a word Whisper never had a chance to hear correctly. Measured on real audio: without it
"Clark's just not good at that" and "a personal intelligence tool"; with it "Claude is
just not good at that" and "the Customer Intelligence Tool". Whisper caps this context
at 224 tokens, so the budget is tight and only frequent, reliably-mangled terms earn a
place.

TIER 2 -- correct. Applied to the finished transcript against the full term list. No
size limit. Cannot recover a word that was acoustically lost, but fixes near-misses
("forums" -> "forms") for free.

SAFETY: the prompt tier is only ever applied to segments the quality gate rated CLEAN.
On marginal audio, priming Whisper with this vocabulary made it hallucinate these very
terms and introduced a repetition loop it did not produce unprompted. That is worse
than a wrong word: it is a plausible wrong word drawn from your own domain. The gate
enforces this via Verdict.allows_glossary_prompt; nothing here should bypass it.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

# Whisper's prompt context is 224 tokens. We budget in characters because the exact BPE
# tokenizer is not available here; ~4 chars/token is the usual English approximation, so
# 700 chars is roughly 175 tokens and leaves comfortable margin for the framing sentence.
# Overshooting silently truncates the START of the prompt inside whisper.cpp, which
# would drop terms without telling us -- hence the conservative budget and the warning.
PROMPT_CHAR_BUDGET = 700

PROMPT_PREAMBLE = "A work conversation at TrueContext discussing"

_WORD_CHARS = r"[A-Za-z0-9]"

# Ordinary words that get capitalised mid-transcript, usually because Whisper started a
# new sentence where the speaker merely paused. They are never the names worth asking
# about, and left unfiltered they crowd out the ones that are. This is cheaper and more
# robust than trying to detect sentence boundaries in spontaneous speech, where the
# punctuation is Whisper's invention rather than the speaker's.
INTERJECTIONS = frozenset(
    {
        "hey", "yeah", "yep", "yes", "okay", "right", "well", "sorry", "mmm", "hmm",
        "oh", "ah", "uh", "um", "so", "but", "and", "then", "now", "like", "just",
        "actually", "maybe", "sure", "thanks", "correct", "exactly", "alright",
        "because", "what", "why", "how", "when", "where", "who", "the", "this",
        "that", "there", "they", "you", "your", "our", "not", "for", "with", "from",
    }
)


class GlossaryError(RuntimeError):
    """The glossary file is missing, malformed, or internally inconsistent."""


@dataclass(frozen=True)
class Term:
    term: str
    tier: str
    variants: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.term.strip():
            raise GlossaryError("a glossary entry has an empty term")
        if self.tier not in {"prompt", "correct"}:
            raise GlossaryError(
                f"term {self.term!r} has tier {self.tier!r}; "
                f"expected 'prompt' or 'correct'"
            )


@dataclass
class Glossary:
    terms: list[Term] = field(default_factory=list)

    @classmethod
    def load(cls, path: str | Path) -> Glossary:
        source = Path(path)
        if not source.is_file():
            raise GlossaryError(f"no glossary at {source}")
        try:
            raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            raise GlossaryError(f"{source} is not valid YAML: {exc}") from exc

        entries = raw.get("terms")
        if entries is None:
            raise GlossaryError(f"{source} has no 'terms' key")
        if not isinstance(entries, list):
            raise GlossaryError(f"{source}: 'terms' must be a list")

        terms: list[Term] = []
        seen: set[str] = set()
        for entry in entries:
            if not isinstance(entry, dict):
                raise GlossaryError(f"{source}: each term must be a mapping, got {entry!r}")
            name = str(entry.get("term", "")).strip()
            key = name.lower()
            if key in seen:
                raise GlossaryError(f"{source}: duplicate term {name!r}")
            seen.add(key)
            terms.append(
                Term(
                    term=name,
                    tier=str(entry.get("tier", "correct")),
                    variants=tuple(str(v) for v in (entry.get("variants") or [])),
                )
            )

        glossary = cls(terms=terms)
        glossary._check_variant_collisions()
        return glossary

    def _check_variant_collisions(self) -> None:
        """A variant that is also another entry's canonical term would rewrite a
        correct word into a different one. Fail loudly rather than corrupt transcripts."""
        canonical = {t.term.lower() for t in self.terms}
        for term in self.terms:
            for variant in term.variants:
                if variant.lower() in canonical and variant.lower() != term.term.lower():
                    raise GlossaryError(
                        f"variant {variant!r} of {term.term!r} is itself a canonical "
                        f"term; correcting it would corrupt correct transcripts"
                    )

    @property
    def prompt_terms(self) -> list[Term]:
        return [t for t in self.terms if t.tier == "prompt"]

    def prompt_text(self, budget: int = PROMPT_CHAR_BUDGET) -> str:
        """Build the decode-time prompt, in file order, stopping at the budget.

        File order is preserved deliberately: it is the operator's priority ordering,
        and silently reordering by some heuristic would make the truncation point
        unpredictable from reading the file.
        """
        names = [t.term for t in self.prompt_terms]
        if not names:
            return ""

        out: list[str] = []
        length = len(PROMPT_PREAMBLE) + 2
        for name in names:
            addition = len(name) + 2
            if length + addition > budget:
                logger.warning(
                    "glossary prompt budget (%d chars) reached after %d of %d prompt-tier "
                    "terms; %r onward were not included. Move less critical terms to the "
                    "correct tier.",
                    budget, len(out), len(names), name,
                )
                break
            out.append(name)
            length += addition

        return f"{PROMPT_PREAMBLE} {', '.join(out)}."

    def correct(self, text: str) -> tuple[str, list[str]]:
        """Replace known variants with their canonical term.

        Returns the corrected text and the canonical terms that were actually applied,
        which is recorded on the ActionRecord so a reviewer can see what was rewritten.

        Done in ONE pass with a single alternation, not variant by variant. Sequential
        replacement is subtly wrong: correcting "intelligence tool" to "Customer
        Intelligence Tool" and then applying a shorter variant "tool" rewrites the word
        inside the canonical text that was just written, yielding "Customer Intelligence
        telemetry". A single pass never rescans its own output, so each position in the
        text is rewritten at most once.

        Variants are ordered longest-first within the alternation, because Python's
        regex alternation prefers the leftmost listed alternative at a given position;
        that makes a multi-word variant win over a shorter one nested inside it.

        Matching is case-insensitive at word boundaries and the canonical spelling
        always wins, because the point is a consistent surface form downstream.
        """
        pairs = [
            (variant, term.term)
            for term in self.terms
            for variant in term.variants
            if variant.strip()
        ]
        if not pairs:
            return text, []
        pairs.sort(key=lambda pair: len(pair[0]), reverse=True)

        lookup = {variant.lower(): canonical for variant, canonical in pairs}
        alternation = "|".join(re.escape(variant) for variant, _ in pairs)
        pattern = re.compile(
            rf"(?<!{_WORD_CHARS})({alternation})(?!{_WORD_CHARS})",
            re.IGNORECASE,
        )

        applied: list[str] = []

        def _replace(match: re.Match[str]) -> str:
            canonical = lookup[match.group(1).lower()]
            if canonical not in applied:
                applied.append(canonical)
            return canonical

        return pattern.sub(_replace, text), applied

    def unknown_proper_nouns(self, text: str, min_occurrences: int = 1) -> list[str]:
        """Heuristic: capitalised words this glossary does not know, seen repeatedly.

        This is the cheap half of the "ask me to clarify" loop. It is a HEURISTIC, not a
        semantic judgement -- it finds candidates, it does not decide they are real. A
        name it surfaces may be an ordinary word Whisper capitalised mid-sentence, and a
        term it misses may still be wrong. Treat the output as a review list to skim,
        and promote the real entries into the glossary by hand.
        """
        known = {t.term.lower() for t in self.terms}
        for term in self.terms:
            known.update(v.lower() for v in term.variants)
            known.update(part.lower() for part in term.term.split())

        # Strip Whisper's sound-event annotations first. It writes non-speech events as
        # "*Loud noise*", "*Police*", "*Evil music plays*" -- capitalised words that are
        # not names at all. Left in, they crowd out the real candidates: a run over a
        # real transcript surfaced only "Loud", from two "*Loud noise*" markers.
        text = re.sub(r"\*[^*]*\*", " ", text)

        counts: dict[str, int] = {}
        # Skip sentence-initial words: capitalisation there carries no information.
        # The body allows interior capitals so mixed-case names are caught. An earlier
        # pattern required two lowercase letters after the capital, which silently
        # missed exactly the names most worth asking about: "ViPON", "FiveHands",
        # "OpenSearch". Those are the ones Whisper is least likely to have spelled right.
        for match in re.finditer(r"(?<![.!?]\s)(?<!^)\b([A-Z][A-Za-z]{2,})\b", text, re.M):
            word = match.group(1)
            if word.lower() in known or word.lower() in INTERJECTIONS:
                continue
            counts[word] = counts.get(word, 0) + 1

        return sorted(
            (w for w, n in counts.items() if n >= min_occurrences),
            key=lambda w: (-counts[w], w),
        )
