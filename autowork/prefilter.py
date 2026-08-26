#!/usr/bin/env python3
r"""Select the passages of a transcript that could plausibly contain an action item.

WHY THIS EXISTS: measured on this machine, local extraction runs at roughly 0.14
seconds per character of transcript (gemma3:4b, CPU). That is about 2x SLOWER than
realtime -- four hours of conversation would take nearly eight hours to extract from,
so a day's recordings take longer to process than they took to have. No smaller local
model fixes it, because the small models are already too weak on quality.

Most of a workday transcript is discussion, opinion and thinking aloud. Only a small
fraction contains anyone actually committing to anything. Selecting that fraction with
cheap deterministic pattern matching, and sending only it to the model, attacks the
volume rather than the model speed. It runs in milliseconds.

WHAT THIS COSTS, stated plainly: this trades recall for throughput. A commitment phrased
in a way the patterns do not cover is not passed to the extractor and will not appear in
the queue. It is a filter, not a comprehension step, and it cannot know what it missed.
Measured on a real transcript, it kept two of three genuine action items and dropped one
phrased "it should just be the top one" -- impersonal, with no first-person commitment
verb anywhere near it. Two things make that acceptable rather than reckless:

  * The full transcript is always retained. Widening the patterns and re-running loses
    nothing, so a missed commitment is recoverable rather than destroyed.
  * Passages are selected WITH surrounding context, because "I'll do that" carries no
    detail alone and the extractor needs the detail to write a useful body.

Tune CUES rather than raising CONTEXT_SENTENCES: more patterns costs almost nothing,
while more context multiplies the volume reaching the model. Note this module is a raw
string docstring and the patterns below are raw strings -- a lost backslash silently
turns \b from a word boundary into a backspace character, which matches nothing and
fails without any error at all.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Sentences of context to keep either side of a match. Measured on a real 258-sentence
# transcript: at 2 the filter kept 52% of the text, at 1 it keeps 32% and still carries
# the referent of "that" or "it". Context is the dominant cost, since every hit pays it.
CONTEXT_SENTENCES = 1

# Commitment and request language, grouped by intent so the reason a passage was kept is
# reportable. That is what makes this tunable from real misses rather than from guesses.
#
# Deliberately EXCLUDED as too weak: bare "I can", "we could", "you could". In spoken
# discussion these are overwhelmingly hypothetical, and the extractor prompt already
# refuses hypotheticals, so passing them costs time and yields nothing.
CUES: dict[str, tuple[str, ...]] = {
    "first_person_commitment": (
        r"\bI'?ll\b", r"\bI will\b", r"\bI'?m going to\b", r"\bI'?m gonna\b",
        r"\bI should\b", r"\bI need to\b", r"\bI have to\b", r"\bI'?ve got to\b",
        r"\bI want to\b", r"\blet me\b", r"\bI'?m thinking I\b",
        r"\bI plan to\b", r"\bI intend to\b", r"\bI plan on\b",
    ),
    "request_of_someone": (
        r"\bcan you\b", r"\bcould you\b", r"\bwould you\b", r"\bwill you\b",
        r"\bdo you mind\b", r"\bplease\b", r"\bsend me\b",
        r"\bcan somebody\b", r"\bcan someone\b",
    ),
    "shared_or_assigned": (
        r"\bwe need to\b", r"\bwe should\b", r"\bwe have to\b", r"\bwe'?ll\b",
        r"\blet'?s\b", r"\byou should\b", r"\byou need to\b",
        r"\bsomeone should\b", r"\bwe'?re going to\b",
        # Impersonal obligation. Added after the filter dropped a real action item
        # phrased "it should just be the top one or any that are over like 0.9".
        r"\bit should\b", r"\bthat should\b", r"\bthere should\b",
    ),
    "follow_up": (
        r"\bfollow up\b", r"\btake it up\b", r"\bcircle back\b", r"\bget back to\b",
        r"\bcheck with\b", r"\btalk to\b", r"\breach out\b", r"\bping\b",
        r"\bwrite up\b", r"\bwrite it up\b", r"\bfile a\b", r"\bopen a ticket\b",
        r"\bmake a ticket\b", r"\bcreate a\b", r"\bset up\b", r"\bfinalize\b",
        r"\bfinalise\b", r"\bworth exploring\b", r"\bworth doing\b",
        # "making sure X" is how a request gets phrased in this corpus. Added after a
        # test showed a real action item ("just making sure that the quote aligns...")
        # matched NO cue and survived only by falling inside a neighbouring match's
        # context window. It would have been lost under different sentence ordering,
        # which is exactly the silent recall failure this filter risks.
        r"\bmaking sure\b", r"\bmake sure\b", r"\bneeds to\b",
    ),
    "deadline": (
        r"\bby (?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
        r"\bby (?:tomorrow|today|tonight|next week|month end|end of day|EOD)\b",
        r"\bnext week\b", r"\bthis week\b", r"\btomorrow\b", r"\bend of day\b",
        r"\bdeadline\b",
    ),
}

_COMPILED: dict[str, re.Pattern[str]] = {
    name: re.compile("|".join(patterns), re.IGNORECASE)
    for name, patterns in CUES.items()
}

# Spoken filler that trips the cues above without ever being a commitment. Every entry
# was observed firing on a real transcript. Cheaper and far more predictable than asking
# a model to judge, and each one removes a whole passage's worth of volume.
EXCLUSIONS = re.compile(
    "|".join(
        (
            r"\bI'?m going to be honest\b",
            r"\bI'?m gonna be honest\b",
            r"\bI'?ll be honest\b",
            r"\bI'?ll say\b",
            r"\bI'?m going to say\b",
            r"\bI want to say\b",
            r"\bcan you see\b",
            r"\bcan you hear\b",
            r"\blet me just\b",
            r"\blet me know\b",
            r"\bI can understand\b",
            r"\bI can see\b",
            r"\bI'?m going to guess\b",
            r"\bI should say\b",
            r"\blet'?s see\b",
            r"\blet'?s say\b",
            r"\byou should be able\b",
            r"\bI'?ll be curious\b",
            r"\bI want to challenge\b",
            # Observed firing on the 2026-08-25 digest. "I will say" is the same
            # discourse marker as "I'll say", which was already excluded; the contracted
            # form alone was not enough.
            r"\bI will say\b",
            r"\bI'?ll just go\b",
            r"\bI will just\b",
            r"\bI was gonna say\b",
            r"\bI was going to say\b",
            r"\bI'?m curious\b",
            r"\bI want to keep\b",
            r"\bI want to like\b",
        )
    ),
    re.IGNORECASE,
)

# Guard against the exact bug that made this module silently useless once: a lost
# backslash turns \b into chr(8), which matches nothing and raises no error.
assert "\\b" in EXCLUSIONS.pattern, "EXCLUSIONS lost its word-boundary escapes"
assert all("\\b" in p.pattern for p in _COMPILED.values()), "CUES lost word boundaries"


@dataclass(frozen=True)
class Passage:
    """A selected span of transcript, with the cue categories that selected it."""

    text: str
    first_sentence: int
    last_sentence: int
    cues: tuple[str, ...]

    @property
    def sentence_count(self) -> int:
        return self.last_sentence - self.first_sentence + 1


@dataclass
class FilterResult:
    passages: list[Passage] = field(default_factory=list)
    original_chars: int = 0
    total_sentences: int = 0
    matched_sentences: int = 0

    @property
    def selected_chars(self) -> int:
        return sum(len(p.text) for p in self.passages)

    @property
    def reduction(self) -> float:
        """Fraction of the transcript NOT sent to the extractor."""
        if not self.original_chars:
            return 0.0
        return 1.0 - (self.selected_chars / self.original_chars)

    @property
    def speedup(self) -> float:
        if not self.selected_chars:
            return float("inf")
        return self.original_chars / self.selected_chars

    def cue_counts(self) -> dict[str, int]:
        """How often each cue category fired. Shows which patterns earn their keep."""
        counts: dict[str, int] = {name: 0 for name in CUES}
        for passage in self.passages:
            for cue in passage.cues:
                counts[cue] += 1
        return counts


def split_sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def match_cues(sentence: str) -> tuple[str, ...]:
    """Which cue categories this sentence triggers, if any.

    An excluded filler phrase suppresses the whole sentence. That is deliberate: the
    excluded phrase IS the cue match in these cases ("I'm going to be honest" matching
    first_person_commitment), so suppressing only that one pattern would leave the
    sentence selected anyway on some other weak cue.
    """
    if EXCLUSIONS.search(sentence):
        return ()
    return tuple(name for name, pattern in _COMPILED.items() if pattern.search(sentence))


def select(text: str, context_sentences: int = CONTEXT_SENTENCES) -> FilterResult:
    """Select passages that may contain an action item, merging overlapping windows.

    Overlapping windows are merged rather than emitted separately, so a dense run of
    commitments becomes one coherent passage instead of many near-duplicates. That
    matters for cost: duplicated context is volume the extractor pays for twice.
    """
    sentences = split_sentences(text)
    result = FilterResult(original_chars=len(text), total_sentences=len(sentences))
    if not sentences:
        return result

    hits: dict[int, tuple[str, ...]] = {}
    for index, sentence in enumerate(sentences):
        if cues := match_cues(sentence):
            hits[index] = cues
    result.matched_sentences = len(hits)

    if not hits:
        logger.info(
            "prefilter: no commitment language in %d sentences; nothing to extract",
            len(sentences),
        )
        return result

    windows: list[tuple[int, int, set[str]]] = []
    for index in sorted(hits):
        start = max(0, index - context_sentences)
        end = min(len(sentences) - 1, index + context_sentences)
        cues = set(hits[index])
        if windows and start <= windows[-1][1] + 1:
            prev_start, prev_end, prev_cues = windows[-1]
            windows[-1] = (prev_start, max(prev_end, end), prev_cues | cues)
        else:
            windows.append((start, end, cues))

    result.passages = [
        Passage(
            text=" ".join(sentences[start : end + 1]),
            first_sentence=start,
            last_sentence=end,
            cues=tuple(sorted(cues)),
        )
        for start, end, cues in windows
    ]

    logger.info(
        "prefilter: %d of %d sentences matched, %d passage(s), %d -> %d chars "
        "(%.0f%% reduction, %.1fx less to extract)",
        len(hits), len(sentences), len(result.passages),
        result.original_chars, result.selected_chars,
        100 * result.reduction, result.speedup,
    )
    return result


def filtered_text(text: str, context_sentences: int = CONTEXT_SENTENCES) -> str:
    """Convenience: the selected passages joined for handing to the extractor.

    Passages are separated by a blank line so the model sees distinct moments rather
    than one continuous conversation, which would invite it to invent connections
    between things said an hour apart.
    """
    return "\n\n".join(p.text for p in select(text, context_sentences).passages)
