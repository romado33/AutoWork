#!/usr/bin/env python3
"""Decide whether a transcript is a work conversation worth summarising.

THREE GATES, THREE DIFFERENT QUESTIONS.

  * The audio gate (gate.py) asks "is there speech-shaped signal here", from the
    spectrum, for free, before anything is uploaded.
  * Cheap quality (quality.py) asks "did Whisper/the diarizer obviously break", from
    loops and speaker-label explosion, with no model.
  * This module asks about MEANING. That is two judgements in one request, because
    they are not the same:

      1. Is this intelligible? Could a reader follow a single conversation?
      2. If so, is it a work conversation (vs podcast, commute chat, ...)?

The 2026-08-27 11:30 recording showed why (1) has to be able to veto (2). The
classifier returned work_conversation at 0.92 "despite garbling" because contracts
and Python appeared in the jumble, and `keep` trusted `is_work`. The audio gate had
passed at +4.9 dB. The mailed summary was satellite-launch economics that nobody
said. Work vocabulary inside nonsense is still nonsense.

Deliberately cheap: one small request per transcript, on text that already exists.
It cannot save transcription cost -- only the audio gate can do that. What it saves
is the summary and the action queue from being polluted.

`keep` still errs toward keeping when the doubt is "work vs personal". It does not
err toward keeping when the model says the text is unintelligible. Those error
costs are opposite: dropping a real conversation loses a day's actions; admitting
fluent garbage mails a fake meeting.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from autowork.llm import LLMError, build_backend

logger = logging.getLogger(__name__)

DEFAULT_BACKEND = "openai:gpt-5.4-mini"

# Below this, a "not work" verdict is still passed on. Set low on purpose: the cost
# of dropping a real conversation is losing a day's actions, while the cost of
# admitting an irrelevant one is a paragraph of noise in an email. This threshold
# does NOT apply to unintelligible text -- that is always dropped.
KEEP_THRESHOLD = 0.35

RELEVANCE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        # Separate from is_work_conversation on purpose. The 11:30 miss labelled
        # fluent salad as work because some sentences mentioned contracts.
        "intelligible": {"type": "boolean"},
        "is_work_conversation": {"type": "boolean"},
        "confidence": {"type": "number"},
        "kind": {
            "type": "string",
            "enum": [
                "work_conversation",
                "media_playback",
                "ambient_noise",
                "personal_conversation",
                "unintelligible",
                "other",
            ],
        },
        "reason": {"type": "string"},
    },
    "required": [
        "intelligible",
        "is_work_conversation",
        "confidence",
        "kind",
        "reason",
    ],
    "additionalProperties": False,
}

PROMPT = """\
You classify transcripts from a pocket voice recorder. The owner records their working \
day, so the recorder also captures commutes, media playing nearby, and stretches of \
handling noise that the transcription model renders as plausible-looking fragments -- \
sometimes fluent, grammatical English that nobody said.

Answer TWO questions, in order. Do not skip the first.

1. intelligible
   Could a person who was not in the room follow ONE conversation from this text?
   Real speech is rambling, hesitant, full of "um" and false starts. That IS \
intelligible. Set intelligible=true for that.
   Set intelligible=false when the text is nonsense, even if individual sentences \
look fine: topic-salad (satellites, then video games, then a backlog item, with no \
thread), a phrase repeating over and over, invented names with no discussion around \
them, fragments that do not follow each other, or anything you would honestly \
describe as "despite garbling", "jumble", or "no clear thread". Work-like words \
(contracts, Python, backlog, meetings) inside that salad do not make it intelligible. \
Measured miss: a two-person recording was classified as a work conversation \
"despite garbling" and mailed as a 13-person satellite meeting.

2. kind / is_work_conversation
   Only if intelligible is true. Otherwise kind MUST be unintelligible and \
is_work_conversation MUST be false.
     work_conversation      two or more people discussing work: systems, customers, \
projects, decisions, tasks.
     media_playback         a podcast, TV, radio or video. Tells: a narrator, an \
interviewer, production credits, a single voice reading fluent prose.
     ambient_noise          transcription of non-speech. Tells: disconnected \
fragments, onomatopoeia ("beep beep").
     personal_conversation  real speech between people, but not about work.
     unintelligible         not intelligible, per question 1.
     other                  none of the above.

Judge what the words ARE, not whether they are interesting. A dull work conversation \
is still a work conversation. A gripping podcast is still not one.

confidence is how sure you are of the classification, 0.0 to 1.0.
reason is one short sentence citing what decided it. If the honest reason would \
contain "despite garbling", you have already answered intelligible=false.

Return JSON only.
"""


class RelevanceError(RuntimeError):
    """Classification failed. Raised rather than defaulting, because silently treating
    an unknown as relevant defeats the gate and treating it as irrelevant loses a day."""


@dataclass(frozen=True)
class Verdict:
    is_work: bool
    confidence: float
    kind: str
    reason: str
    intelligible: bool = True

    @property
    def keep(self) -> bool:
        """Whether to pass this transcript downstream.

        Unintelligible text is never kept, even when the model also tagged it as
        work -- that combination was the 2026-08-27 11:30 mail. A low-confidence
        'not work' on text that IS intelligible is kept: an uncertain classifier
        should not be able to discard a real conversation.
        """
        if not self.intelligible or self.kind == "unintelligible":
            return False
        if self.is_work:
            return True
        return self.confidence < KEEP_THRESHOLD

    def __str__(self) -> str:
        clarity = "intelligible" if self.intelligible else "nonsensical"
        return (
            f"{self.kind} ({clarity}, confidence {self.confidence:.2f}): "
            f"{self.reason}"
        )


def classify(text: str, backend_spec: str = DEFAULT_BACKEND, sample_chars: int = 6000) -> Verdict:
    """Classify a transcript. Only a sample is sent; the character is clear early.

    Sampling from the START rather than the middle: a recording that opens as a
    conversation and drifts into a podcast should be judged on the conversation, and
    the reverse case is caught because media playback is recognisable from any window.
    """
    stripped = text.strip()
    if not stripped:
        raise RelevanceError("nothing to classify: the transcript is empty")

    backend = build_backend(backend_spec)
    prompt = f"{PROMPT}\nTRANSCRIPT:\n{stripped[:sample_chars]}\n\nJSON:"

    try:
        completion = backend.complete(prompt, schema=RELEVANCE_SCHEMA)
    except LLMError as exc:
        raise RelevanceError(f"classification failed: {exc}") from exc

    try:
        data = json.loads(completion.text)
    except json.JSONDecodeError as exc:
        raise RelevanceError(f"classifier returned invalid JSON: {exc}") from exc

    kind = str(data.get("kind", "other"))
    intelligible = data.get("intelligible")
    if intelligible is None:
        # Schema requires it; if a pasted backend drops the field, kind still vetoes.
        intelligible = kind not in {"unintelligible", "ambient_noise"}

    verdict = Verdict(
        is_work=bool(data.get("is_work_conversation")),
        confidence=max(0.0, min(1.0, float(data.get("confidence", 0.0)))),
        kind=kind,
        reason=str(data.get("reason", "")).strip(),
        intelligible=bool(intelligible) and kind != "unintelligible",
    )
    logger.info(
        "relevance: %s -> %s", "KEEP" if verdict.keep else "DROP", verdict
    )
    return verdict
