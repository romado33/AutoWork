#!/usr/bin/env python3
"""Decide whether a transcript is a work conversation worth summarising.

TWO GATES, TWO DIFFERENT QUESTIONS. The audio gate (gate.py) asks "is there speech-
shaped signal here" and answers it from the spectrum, for free, before anything is
uploaded. It cannot ask "is this a work conversation", because that is a question about
meaning and the audio does not contain the answer.

Both failure modes were observed on real recordings:

  * A 45-minute recording ended with an hour of true-crime podcast. Perfect audio,
    flawless transcription, and utterly irrelevant -- the audio gate loved it.
  * A 13-minute recording passed the audio gate at +4.3 dB and transcribed to
    "What something something for the little folks. For the grandmas. Oh, Poor Lector,
    Lector. Beep beep." Speech-shaped noise, no conversation.

Without this stage both flow into the daily summary as fact, and the second one is the
more dangerous because it is short, plausible-looking fragments rather than obviously
off-topic prose.

Deliberately cheap: one small request per transcript, on text that already exists. It
runs AFTER transcription because it needs the words, which is also why it cannot save
transcription cost -- only the audio gate can do that. What it saves is the summary and
the action queue from being polluted.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from autowork.llm import LLMError, build_backend

logger = logging.getLogger(__name__)

DEFAULT_BACKEND = "openai:gpt-5.4-mini"

# Below this, the transcript is not passed on. Set low on purpose: the cost of dropping
# a real conversation is losing a day's actions, while the cost of admitting an
# irrelevant one is a paragraph of noise in an email. Err toward keeping.
KEEP_THRESHOLD = 0.35

RELEVANCE_SCHEMA: dict = {
    "type": "object",
    "properties": {
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
    "required": ["is_work_conversation", "confidence", "kind", "reason"],
    "additionalProperties": False,
}

PROMPT = """\
You classify transcripts from a pocket voice recorder. The owner records their working \
day, so the recorder also captures commutes, media playing nearby, and stretches of \
handling noise that the transcription model renders as plausible-looking fragments.

Decide what this transcript actually is.

  work_conversation      two or more people discussing work: systems, customers, \
projects, decisions, tasks. Rambling, hesitant and full of filler is NORMAL for real \
speech -- do not mark it irrelevant for being messy.
  media_playback         a podcast, TV, radio or video. Tells: a narrator, an \
interviewer, production credits, a single voice reading fluent prose.
  ambient_noise          transcription of non-speech. Tells: disconnected fragments \
with no thread, onomatopoeia ("beep beep"), phrases that do not follow one another.
  personal_conversation  real speech between people, but not about work.
  unintelligible         too garbled to tell what it was.
  other                  none of the above.

Judge what the words ARE, not whether they are interesting. A dull work conversation is \
still a work conversation. A gripping podcast is still not one.

confidence is how sure you are of the classification, 0.0 to 1.0.
reason is one short sentence citing what decided it.

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

    @property
    def keep(self) -> bool:
        """Whether to pass this transcript downstream.

        A low-confidence 'not work' is kept: an uncertain classifier should not be able
        to discard a real conversation, and the reason is carried forward so a human
        reading the summary can see it was doubted.
        """
        if self.is_work:
            return True
        return self.confidence < KEEP_THRESHOLD

    def __str__(self) -> str:
        return f"{self.kind} (confidence {self.confidence:.2f}): {self.reason}"


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

    verdict = Verdict(
        is_work=bool(data.get("is_work_conversation")),
        confidence=max(0.0, min(1.0, float(data.get("confidence", 0.0)))),
        kind=str(data.get("kind", "other")),
        reason=str(data.get("reason", "")).strip(),
    )
    logger.info(
        "relevance: %s -> %s", "KEEP" if verdict.keep else "DROP", verdict
    )
    return verdict
