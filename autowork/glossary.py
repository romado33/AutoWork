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

import json
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
# Summariser hint: canonical names plus context. Larger than the Whisper prompt
# because this is a chat completion, not a 224-token decode prefix.
SUMMARY_HINT_BUDGET = 1600

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
    about: tuple[str, ...] = ()

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
            about_raw = entry.get("about") or []
            if isinstance(about_raw, str):
                about_raw = [about_raw]
            terms.append(
                Term(
                    term=name,
                    tier=str(entry.get("tier", "correct")),
                    variants=tuple(str(v) for v in (entry.get("variants") or [])),
                    about=tuple(str(v) for v in about_raw),
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

    def summary_hint(self, budget: int = SUMMARY_HINT_BUDGET) -> str:
        """Names and topic hints for the summariser. Not a second model pass.

        The regex correct() pass only rewrites listed variants. This list lets the
        summariser, which already reads the whole conversation, prefer Okta when
        the topic is authorization even if Whisper wrote a novel near-miss. It
        must not invent a product that is not in the transcript.
        """
        lines: list[str] = []
        for term in self.terms:
            if not term.variants and not term.about:
                continue
            detail: list[str] = []
            if term.about:
                detail.append(", ".join(term.about))
            if term.variants:
                detail.append("transcript may say " + ", ".join(term.variants))
            lines.append(f"- {term.term} ({'; '.join(detail)})")

        if not lines:
            return ""

        header = (
            "Known names. In headlines and topics, use the canonical spelling when "
            "the conversation is about that person or system, even if the transcript "
            "has a near-miss (an access/SSO/admin-portal discussion saying Octo or "
            "Octa is Okta). Do not introduce a name that is not discussed at all. "
            "Decision quotes stay verbatim, transcription errors included.\n"
        )
        kept: list[str] = []
        length = len(header)
        for line in lines:
            addition = len(line) + 1
            if length + addition > budget:
                logger.warning(
                    "glossary summary hint budget (%d chars) reached after %d of %d "
                    "terms; later entries were omitted",
                    budget, len(kept), len(lines),
                )
                break
            kept.append(line)
            length += addition
        return header + "\n".join(kept)

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

    def rewrite_file(self, path: str | Path) -> list[str]:
        """Apply correct() to a cached transcript on disk. No-op if already clean.

        Fresh transcription already runs correct() before write. Re-sends read the
        file as-is, so a variant added later (Dave Cazal, OctoAdmin) never reached
        the summariser. Rewriting here is the same pass, billed nothing.
        """
        source = Path(path)
        raw = source.read_text(encoding="utf-8")
        fixed, applied = self.correct(raw)
        if fixed != raw:
            source.write_text(fixed, encoding="utf-8")
            return applied
        return []

    @classmethod
    def promote(
        cls,
        path: str | Path,
        term: str,
        variant: str = "",
        about: list[str] | tuple[str, ...] | None = None,
    ) -> Term:
        """Add a name to the glossary file without rewriting the rest of it.

        Comments in glossary.yml explain measured WHY. A round-trip dump would
        wipe them. New terms are appended; a new variant of an existing term is
        spliced into that entry's variants list. Always correct-tier: the prompt
        budget is tight and a UI click must not evict Claude for a one-off name.
        """
        source = Path(path)
        canonical = term.strip()
        heard = variant.strip()
        if not canonical:
            raise GlossaryError("a term is required")
        if heard.lower() == canonical.lower():
            heard = ""

        glossary = cls.load(source)
        existing = next(
            (t for t in glossary.terms if t.term.lower() == canonical.lower()), None
        )
        if heard:
            other_canonical = {
                t.term.lower() for t in glossary.terms if t is not existing
            }
            if heard.lower() in other_canonical:
                raise GlossaryError(
                    f"variant {heard!r} of {canonical!r} is itself a canonical "
                    f"term; correcting it would corrupt correct transcripts"
                )

        if existing is not None:
            if not heard:
                return existing
            if heard.lower() in {v.lower() for v in existing.variants}:
                return existing
            _splice_variant(source, existing.term, heard)
            glossary = cls.load(source)
            return next(t for t in glossary.terms if t.term.lower() == existing.term.lower())

        _append_term(source, canonical, heard, tuple(about or ()))
        return cls.load(source).terms[-1]

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


def _yaml_string(value: str) -> str:
    """A YAML scalar that will not be interpreted as a boolean or a nested structure."""
    return json.dumps(value, ensure_ascii=False)


def _append_term(path: Path, term: str, variant: str, about: tuple[str, ...]) -> None:
    variants = f"[{_yaml_string(variant)}]" if variant else "[]"
    lines = [
        f"  - term: {_yaml_string(term)}",
        "    tier: correct",
        f"    variants: {variants}",
    ]
    if about:
        lines.append(
            "    about: [" + ", ".join(_yaml_string(a) for a in about) + "]"
        )
    raw = path.read_text(encoding="utf-8")
    path.write_text(raw.rstrip() + "\n" + "\n".join(lines) + "\n", encoding="utf-8")


def _splice_variant(path: Path, term: str, variant: str) -> None:
    """Insert one variant into an existing term's flow-style variants list."""
    raw = path.read_text(encoding="utf-8")
    term_re = re.compile(
        rf"^(\s*)- term:\s*(?:{_re_quote(term)})\s*$",
        re.M,
    )
    match = term_re.search(raw)
    if match is None:
        raise GlossaryError(f"cannot find term {term!r} in {path.name} to add a variant")
    start = match.end()
    next_term = re.search(r"^(\s*)- term:", raw[start:], re.M)
    block_end = start + next_term.start() if next_term else len(raw)
    block = raw[start:block_end]
    var_re = re.compile(r"^(\s*)variants:\s*\[(.*)\]", re.M)
    var_match = var_re.search(block)
    if var_match is None:
        # No variants line yet: insert one after the term line.
        insert = f"\n{match.group(1)}  variants: [{_yaml_string(variant)}]"
        path.write_text(raw[:start] + insert + raw[start:], encoding="utf-8")
        return
    inner = var_match.group(2).strip()
    current = yaml.safe_load(f"[{inner}]") if inner else []
    if not isinstance(current, list):
        raise GlossaryError(f"variants of {term!r} is not a list")
    if any(str(v).lower() == variant.lower() for v in current):
        return
    current.append(variant)
    rendered = "[" + ", ".join(_yaml_string(str(v)) for v in current) + "]"
    new_block = (
        block[: var_match.start()]
        + f"{var_match.group(1)}variants: {rendered}"
        + block[var_match.end() :]
    )
    path.write_text(raw[:start] + new_block + raw[block_end:], encoding="utf-8")


def _re_quote(term: str) -> str:
    escaped = re.escape(term)
    return rf'(?:"{escaped}"|\'{escaped}\'|{escaped})'
