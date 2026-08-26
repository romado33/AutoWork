#!/usr/bin/env python3
"""Transcript -> candidate ActionRecords, via a cloud LLM backend.

Only the PREFILTERED passages are sent, never the whole transcript and never the audio:
measured, that is about 18% of a recording, pre-selected to commitment-bearing moments.

Local inference was tried and removed. gemma3:4b took 358s on a 2,600-character excerpt
and found 1 of 3 real action items; qwen2.5:7b took 493s and found none; phi4 never
finished. The same transcript through gpt-5.4-mini takes ~7s and finds all three.

THE CENTRAL PROBLEM: a language model asked to find action items in a rambling
conversation will find them whether or not they are there. Combined with a transcription
layer that already fabricates on bad audio, an eager extractor turns a misheard sentence
into a Jira ticket. Three defences, in order of strength:

  1. GROUNDING QUOTE (the strong one). Every candidate must carry a verbatim quote from
     the transcript. We then CHECK that the quote actually appears in the source text and
     discard any candidate whose quote does not. A model can invent an action; it has a
     much harder time inventing an action whose supporting quote survives a substring
     check against the transcript it was given.
  2. A prompt that permits returning nothing. Note it is deliberately NOT maximally
     strict: an earlier severe version found 1 of 3 real items because it refused all
     hedged phrasing. Grounding is the guard against invention, not prompt severity.
  3. A fixed target-system vocabulary. The model chooses from a configured list and
     anything else is rejected, so it cannot invent a system to write to.

Everything surviving all three still lands in the review queue as PENDING. None of this
is a substitute for a human reading it.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

from autowork.action import ActionRecord, ActionType, Provenance
from autowork.llm import Backend, LLMError, build_backend
from autowork.transcribe import TranscribedSegment

logger = logging.getLogger(__name__)

DEFAULT_BACKEND = "openai:gpt-5.4-mini"

# Characters per chunk. Kept well inside a small model's context so the transcript, the
# instructions and the JSON reply all fit with room to spare. Overlap carries a
# commitment that straddles a boundary into both chunks; the queue's dedupe key then
# collapses the duplicate rather than filing it twice.
CHUNK_CHARS = 6000
CHUNK_OVERLAP_CHARS = 600

# A quote must carry at least this many words to be checkable. "Yeah", "Okay" and
# "Correct" appear all over any transcript and would ground nothing -- a substring check
# against them always passes, so they provide no evidence at all. Counted in words
# rather than characters because the question is whether the quote is a meaningful
# fragment, and a character threshold rejects a real five-word commitment for being one
# character short.
#
# Set to 8 after testing found that "But I mean, it is" -- five words, and present
# verbatim in a real transcript -- passed at 5 and would have grounded a completely
# fabricated action. Common conversational filler is short; a quote long enough to be
# specific is much harder to pair with an invented action. The cost is that a genuinely
# terse commitment ("I'll update the doc") is rejected, which loses a real item rather
# than admitting a false one. That trade is deliberate: a missed action costs value, an
# invented one costs trust.
MIN_QUOTE_WORDS = 8

# Server-enforced reply shape. Lives beside the prompt on purpose: the two describe the
# same contract, and letting them drift is how a schema silently stops matching what the
# prompt asks for. `strict` schemas require every property listed in `required` and
# `additionalProperties: false`, so adding a field here means adding it in both places.
ACTION_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "actions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "body": {"type": "string"},
                    "owner": {"type": "string", "enum": ["me", "other", "unclear"]},
                    "target_system": {"type": "string"},
                    "action_type": {
                        "type": "string",
                        "enum": ["create", "update", "comment", "message", "task"],
                    },
                    "confidence": {"type": "number"},
                    "quote": {"type": "string"},
                },
                "required": [
                    "title", "body", "owner", "target_system",
                    "action_type", "confidence", "quote",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["actions"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """\
You extract action items from transcripts of spoken work conversations.

Return an empty list when there is genuinely nothing; that is a correct and common \
answer. But do not be so strict that you refuse real items for being tentatively \
phrased. An earlier version of this prompt demanded explicit commitments and, measured \
against a real transcript, found one of three genuine action items. The grounding check \
below is what guards against invention -- not prompt severity.

People rarely state commitments crisply out loud. "The only thing I was thinking about \
that maybe is worth exploring is making sure X" IS an action item even though it is \
hedged. Extract intent, not only explicit undertakings.

Extract:
  - something someone said they would do, or should do
  - something someone asked another person to do
  - a change to a system, process or document that a speaker argued for
  - a hedged or tentative proposal that names a specific concrete change

Do NOT extract:
  - pure analysis with no proposed change ("project was harder than site")
  - things already done ("I updated it yesterday")
  - meeting logistics ("can you see my screen")
  - anything where you cannot quote a specific sentence supporting it

The transcript comes from a pocket microphone and may contain transcription errors. \
If a passage is garbled, do not guess at what it meant. Skip it.

Return ONLY JSON of this exact shape:
{"actions": [{
  "title": "short imperative summary, under 80 characters",
  "body": "what specifically needs doing, and any detail stated in the conversation",
  "owner": "me" | "other" | "unclear",
  "target_system": one of the allowed systems listed below,
  "action_type": "create" | "update" | "comment" | "message" | "task",
  "confidence": a number from 0.0 to 1.0,
  "quote": "the VERBATIM sentence from the transcript that states this action"
}]}

The quote must be copied exactly, character for character, from the transcript. It is \
checked against the source. A candidate whose quote is not found verbatim is discarded.

"owner" is "me" if the speaker labelled as the recording's owner committed to it, \
"other" if someone else did, "unclear" if the transcript does not make it plain.
"""


class ExtractionError(RuntimeError):
    """Extraction failed. Raised rather than returning an empty list, because "no
    actions found" and "the model was unreachable" must never look the same."""


@dataclass(frozen=True)
class ExtractorConfig:
    backend: str = DEFAULT_BACKEND
    # The systems an action may target. A model choice outside this list is rejected,
    # so an incomplete list silently discards real action items: measured, a genuine
    # item was found and then thrown away because "salesforce" was missing here. Keep
    # it aligned with the systems actually discussed, and prefer "notes" as the
    # catch-all over leaving a real target off.
    target_systems: tuple[str, ...] = (
        "backlog-tool", "flare", "jira", "cit", "salesforce",
        "confluence", "slack", "email", "notes",
    )
    owner_name: str = "Rob"
    timeout_sec: int = 600
    min_confidence: float = 0.3
    chunk_chars: int = CHUNK_CHARS

    def __post_init__(self) -> None:
        if not self.target_systems:
            raise ExtractionError("at least one target_system must be configured")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ExtractionError(
                f"min_confidence must be in [0,1], got {self.min_confidence}"
            )
        if self.chunk_chars < 500:
            raise ExtractionError(f"chunk_chars too small: {self.chunk_chars}")


@dataclass
class Candidate:
    """A model-proposed action, before grounding and validation."""

    title: str
    body: str
    owner: str
    target_system: str
    action_type: str
    confidence: float
    quote: str
    rejected_because: str | None = field(default=None)


def _normalise(text: str) -> str:
    """Lowercase, strip punctuation and collapse whitespace, for quote matching.

    Matching on raw text would fail on trivial differences: the model reproduces a quote
    but drops a comma or normalises an ellipsis. Those are not fabrication. Inventing a
    sentence that survives this normalisation is still hard.
    """
    return " ".join(re.findall(r"[a-z0-9']+", text.lower()))


def chunk_transcript(
    text: str, size: int = CHUNK_CHARS, overlap: int = CHUNK_OVERLAP_CHARS
) -> list[str]:
    """Split on sentence boundaries, with overlap so a straddling commitment survives."""
    if len(text) <= size:
        return [text] if text.strip() else []

    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    current: list[str] = []
    length = 0

    for sentence in sentences:
        if length + len(sentence) > size and current:
            chunks.append(" ".join(current))
            # Carry the tail of this chunk into the next one.
            tail: list[str] = []
            tail_len = 0
            for prev in reversed(current):
                if tail_len + len(prev) > overlap:
                    break
                tail.insert(0, prev)
                tail_len += len(prev)
            current = tail
            length = tail_len
        current.append(sentence)
        length += len(sentence)

    if current:
        chunks.append(" ".join(current))
    return chunks


def parse_candidates(raw: str, config: ExtractorConfig) -> list[Candidate]:
    """Parse the model's JSON. Malformed entries are dropped individually, not fatally.

    One bad entry must not discard a whole chunk's real findings, but every drop is
    logged so a silently unhelpful model is visible rather than looking like a quiet day.
    """
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ExtractionError(f"model did not return valid JSON: {exc}") from exc

    if not isinstance(data, dict) or "actions" not in data:
        raise ExtractionError(
            f"expected an object with an 'actions' key, got {type(data).__name__}"
        )
    entries = data["actions"]
    if not isinstance(entries, list):
        raise ExtractionError(f"'actions' must be a list, got {type(entries).__name__}")

    out: list[Candidate] = []
    for entry in entries:
        if not isinstance(entry, dict):
            logger.warning("dropping non-object action entry: %r", entry)
            continue
        try:
            confidence = float(entry.get("confidence", 0.0))
        except (TypeError, ValueError):
            logger.warning("dropping entry with unparseable confidence: %r", entry)
            continue
        out.append(
            Candidate(
                title=str(entry.get("title", "")).strip(),
                body=str(entry.get("body", "")).strip(),
                owner=str(entry.get("owner", "unclear")).strip().lower(),
                target_system=str(entry.get("target_system", "")).strip(),
                action_type=str(entry.get("action_type", "task")).strip().lower(),
                confidence=max(0.0, min(1.0, confidence)),
                quote=str(entry.get("quote", "")).strip(),
            )
        )
    return out


def ground(candidates: list[Candidate], source_text: str, config: ExtractorConfig) -> list[Candidate]:
    """Mark candidates that fail grounding or validation. Nothing is silently dropped.

    Returns every candidate with `rejected_because` set on the failures, so a caller can
    report what was discarded. A high rejection rate is a signal about the model, and
    hiding it would make a badly-behaved extractor look like a quiet day.
    """
    haystack = _normalise(source_text)
    allowed_types = {t.value for t in ActionType}

    for candidate in candidates:
        # Stripped here rather than trusting the caller: ground() is the safety
        # boundary and must hold on its own, including when called directly.
        if not candidate.title.strip():
            candidate.rejected_because = "no title"
        elif not candidate.body.strip():
            candidate.rejected_because = "no body"
        elif len(_normalise(candidate.quote).split()) < MIN_QUOTE_WORDS:
            candidate.rejected_because = (
                f"quote too short to verify "
                f"({len(_normalise(candidate.quote).split())} words, "
                f"need {MIN_QUOTE_WORDS})"
            )
        elif _normalise(candidate.quote) not in haystack:
            candidate.rejected_because = "quote not found in transcript (fabricated)"
        elif candidate.target_system not in config.target_systems:
            candidate.rejected_because = (
                f"unknown target_system {candidate.target_system!r}"
            )
        elif candidate.action_type not in allowed_types:
            candidate.rejected_because = f"unknown action_type {candidate.action_type!r}"
        elif candidate.confidence < config.min_confidence:
            candidate.rejected_because = (
                f"confidence {candidate.confidence:.2f} below "
                f"{config.min_confidence:.2f}"
            )
    return candidates


def to_action_record(
    candidate: Candidate, segment: TranscribedSegment, config: ExtractorConfig
) -> ActionRecord:
    """Build the queue record, carrying provenance back to the audio."""
    owner_note = {
        "me": "",
        "other": "[someone else committed to this] ",
        "unclear": "[speaker attribution unclear] ",
    }.get(candidate.owner, "[speaker attribution unclear] ")

    # Attribution is inferred from conversational context, not diarization, which was
    # measured as unusable on this audio. Reflect that in the confidence rather than
    # presenting a guess as fact.
    confidence = candidate.confidence
    if candidate.owner != "me":
        confidence = min(confidence, 0.6)

    return ActionRecord(
        title=candidate.title[:80],
        body=f"{owner_note}{candidate.body}",
        target_system=candidate.target_system,
        action_type=ActionType(candidate.action_type),
        confidence=confidence,
        provenance=Provenance(
            source_audio=segment.source_audio,
            start_sec=segment.start_sec,
            end_sec=segment.end_sec,
            speech_rumble_db=segment.speech_rumble_db,
            transcript_excerpt=candidate.quote,
            extractor=f"{config.backend}/grounded",
            glossary_terms_applied=tuple(segment.glossary_applied),
        ),
    )


def extract_from_segment(
    segment: TranscribedSegment, config: ExtractorConfig
) -> tuple[list[ActionRecord], list[Candidate]]:
    """Extract grounded action records from one transcribed segment.

    Returns (accepted records, rejected candidates). The rejects are returned rather
    than logged away so the caller can show what the model proposed and why it was
    refused -- that is the feedback loop for deciding whether this model is good enough.
    """
    if not segment.text.strip():
        return [], []

    backend = build_backend(config.backend)
    systems = ", ".join(config.target_systems)
    accepted: list[ActionRecord] = []
    rejected: list[Candidate] = []

    for index, chunk in enumerate(chunk_transcript(segment.text, config.chunk_chars)):
        prompt = (
            f"{SYSTEM_PROMPT}\n"
            f"Allowed target_system values: {systems}\n"
            f"The recording's owner is {config.owner_name}.\n\n"
            f"TRANSCRIPT:\n{chunk}\n\n"
            f"JSON:"
        )
        logger.info(
            "extracting from chunk %d (%d chars) with %s",
            index + 1, len(chunk), backend.model,
        )
        try:
            completion = backend.complete(prompt, schema=ACTION_SCHEMA)
        except LLMError as exc:
            raise ExtractionError(f"backend {config.backend} failed: {exc}") from exc
        candidates = ground(
            parse_candidates(completion.text, config), chunk, config
        )

        for candidate in candidates:
            if candidate.rejected_because:
                logger.info(
                    "rejected %r: %s", candidate.title[:60], candidate.rejected_because
                )
                rejected.append(candidate)
            else:
                accepted.append(to_action_record(candidate, segment, config))

    return _dedupe_by_quote(accepted), rejected


def quotes_overlap(first: str, second: str, threshold: float = 0.75) -> bool:
    """Whether two grounding quotes are the same spoken moment.

    Containment alone missed a real duplicate pair whose spans overlapped heavily but
    started and ended at different words -- neither contained the other. Sequence
    similarity on the normalised text catches that; two DIFFERENT commitments sharing
    75% of their characters verbatim does not happen in practice, because quotes are
    literal transcript spans.
    """
    import difflib

    a, b = _normalise(first), _normalise(second)
    if not a or not b:
        return False
    if a in b or b in a:
        return True
    return difflib.SequenceMatcher(None, a, b).ratio() >= threshold


def _dedupe_by_quote(records: list[ActionRecord]) -> list[ActionRecord]:
    """Collapse records whose grounding quotes overlap by containment.

    Chunk overlap re-extracts the commitment that straddles the boundary, usually with
    slightly different quote spans ("...five would be used" vs "...five would be used
    and then in the"). A normalised quote contained in another's is the same spoken
    moment; keep the higher-confidence record, or the longer quote on a tie, because
    the longer quote gives the reviewer more context.
    """
    survivors: list[ActionRecord] = []
    for record in sorted(
        records,
        key=lambda r: (-r.confidence, -len(r.provenance.transcript_excerpt)),
    ):
        duplicate = any(
            quotes_overlap(
                record.provenance.transcript_excerpt,
                kept.provenance.transcript_excerpt,
            )
            for kept in survivors
        )
        if duplicate:
            logger.info("dropping chunk-overlap duplicate: %r", record.title[:60])
        else:
            survivors.append(record)
    return survivors
