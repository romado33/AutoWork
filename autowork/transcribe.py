#!/usr/bin/env python3
"""Gate -> slice -> Whisper -> glossary correction. Produces auditable transcripts.

Order matters and is not negotiable:

  1. GATE the raw audio. Segments that fail are never sent to Whisper at all. This is
     what stops fabricated text entering the system, and it must run on unprocessed
     audio (see gate.py for why enhancement defeats the measurement).
  2. SLICE only the surviving segments out to temporary wav. A 45-minute file with one
     23-minute conversation transcribes in half the time, and the rejected audio cannot
     contribute hallucinations to the output because it is not present.
  3. TRANSCRIBE with the glossary prompt only on CLEAN segments. USABLE segments go
     through unprompted -- measured, priming marginal audio makes it hallucinate the
     primed vocabulary.
  4. CORRECT against the full glossary. Safe on any segment, since it only rewrites
     text that is already there.

Timestamps are absolute within the source file. With VOR off the recorder captures
continuously, so file start time plus offset is real wall-clock time. That is what makes
the provenance on an ActionRecord auditable: you can go back and listen to the moment.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from autowork.gate import GateConfig, GateError, Segment, Verdict, measure, segments
from autowork.glossary import Glossary

logger = logging.getLogger(__name__)

WHISPER_RATE_HZ = 16000

# A phrase repeating more than this many times is Whisper looping, not a person
# repeating themselves. Every measured hallucination took this form: "*Police*" x6,
# "city of Estrella" x14, "we're going to do a lot of things" x14, "It's a miracle" x6.
MAX_PHRASE_REPEATS = 3

# A loop must also DOMINATE the text, not merely occur. Real speech repeats common
# words constantly; a hallucination loop is nearly all repetition.
MIN_LOOP_COVERAGE = 0.5


class TranscriptionError(RuntimeError):
    """Transcription failed. Raised rather than returning empty text, because an empty
    transcript is indistinguishable from a silent segment and would be filed as fact."""


# Whisper writes non-speech events in asterisks: "*Loud noise*", "*Police*". A segment
# whose text is ONLY such annotations contains no speech, whatever the audio measured.
_ANNOTATION_ONLY = re.compile(r"^(?:\s*\*[^*]*\*\s*)+$")


def is_annotation_only(text: str) -> bool:
    """True if the text is nothing but Whisper sound-event markers."""
    return bool(text.strip()) and bool(_ANNOTATION_ONLY.match(text.strip()))


@dataclass
class TranscribedSegment:
    """One gated segment, transcribed and corrected."""

    source_audio: str
    start_sec: float
    end_sec: float
    verdict: Verdict
    speech_rumble_db: float
    text: str
    glossary_applied: list[str] = field(default_factory=list)
    prompt_used: bool = False
    repetition_flagged: bool = False

    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec


def looks_like_repetition_loop(
    text: str,
    max_repeats: int = MAX_PHRASE_REPEATS,
    min_coverage: float = MIN_LOOP_COVERAGE,
) -> bool:
    """Detect Whisper's hallucination signature: the same phrase over and over.

    This is the cheap second line of defence behind the audio gate. The gate should
    already have rejected the audio that causes this, but the signature is so reliable
    and so cheap to check that it is worth catching anything that slips through.

    TWO conditions, because repetition count alone is not enough. Real speech repeats
    words constantly -- "the" appears far more than three times in any real transcript,
    and "Yeah, yeah, yeah" is a normal thing for a person to say. What distinguishes a
    loop is that the repetition DOMINATES the text:

      1. a phrase repeats more than `max_repeats` times, AND
      2. those repeats cover at least `min_coverage` of all the words present.

    Punctuation is stripped before matching. Without that, "*Police* *Police* ..." --
    a real measured hallucination -- tokenises into punctuation-bearing words that never
    match each other, and the loop goes undetected.
    """
    words = re.findall(r"[a-z0-9']+", text.lower())
    total = len(words)
    if total < max_repeats + 1:
        return False

    # Size 1 is included deliberately: "*Police*" x6 is a single-word loop. The coverage
    # condition is what makes it safe to check, since a common word repeated in real
    # speech covers only a small fraction of the transcript.
    for size in range(1, 6):
        if total < size * (max_repeats + 1):
            continue
        counts: dict[str, int] = {}
        for i in range(total - size + 1):
            phrase = " ".join(words[i : i + size])
            counts[phrase] = counts.get(phrase, 0) + 1
        for phrase, count in counts.items():
            if count > max_repeats and (count * size) / total >= min_coverage:
                logger.debug(
                    "repetition loop: %r x%d covers %.0f%% of %d words",
                    phrase, count, 100 * count * size / total, total,
                )
                return True
    return False


def excise_repetition_loops(
    text: str, max_repeats: int = MAX_PHRASE_REPEATS
) -> tuple[str, int]:
    """Remove looped sentences, keeping the real speech around them.

    Flagging a whole segment because of a loop is too blunt. A 22-minute conversation
    measured CLEAN at +6.6 dB produced excellent text and then looped "I think it's a
    lot of people." roughly 150 times. Condemning the whole transcript for that would
    discard the conversation.

    IMPORTANT, learned the hard way: a loop is NOT necessarily trailing garbage after
    the speech ran out. Re-running that same audio with a slightly different decode
    prompt produced no loop at all and revealed roughly four minutes of real discussion
    that the loop had been covering. So excision is damage control, not a repair -- it
    salvages the text around a loop, but whatever the loop displaced is simply gone.
    When a segment loops, re-transcribing it is worth more than cleaning it up.

    Whisper's loops are always CONSECUTIVE repetitions of the same sentence, which is
    what makes them separable from real speech: a person may return to a phrase later,
    but they do not say it 150 times in a row. Only consecutive runs are collapsed, and
    a run is kept once rather than deleted outright, so nothing is silently vanished --
    if the person genuinely did repeat themselves, one instance survives.

    Returns the cleaned text and the number of sentences removed.
    """
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    if not parts:
        return text, 0

    kept: list[str] = []
    removed = 0
    run_key: str | None = None
    run_len = 0

    for part in parts:
        key = " ".join(re.findall(r"[a-z0-9']+", part.lower()))
        if key and key == run_key:
            run_len += 1
            if run_len > max_repeats:
                removed += 1
                continue
        else:
            run_key = key
            run_len = 1
        kept.append(part)

    return " ".join(kept), removed


def _slice_audio(source: Path, start_sec: float, end_sec: float, dest: Path) -> None:
    """Extract one segment to 16 kHz mono wav, which is what Whisper consumes."""
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-ss", f"{start_sec:.3f}", "-t", f"{end_sec - start_sec:.3f}",
        "-i", str(source), "-ac", "1", "-ar", str(WHISPER_RATE_HZ), str(dest),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except OSError as exc:
        raise TranscriptionError(f"could not run ffmpeg on {source}: {exc}") from exc
    if proc.returncode != 0 or not dest.is_file():
        raise TranscriptionError(
            f"ffmpeg failed slicing {source} "
            f"[{start_sec:.1f}s-{end_sec:.1f}s]: {proc.stderr.strip()[:300]}"
        )


def _run_whisper(
    sona: Path, model: Path, audio: Path, *, prompt: str | None, threads: int
) -> str:
    cmd = [
        str(sona), "transcribe", str(model), str(audio),
        "--language", "en", "--threads", str(threads), "--gpu-device=-1",
    ]
    if prompt:
        cmd += ["--prompt", prompt]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except OSError as exc:
        raise TranscriptionError(f"could not run {sona}: {exc}") from exc
    if proc.returncode != 0:
        raise TranscriptionError(
            f"sona failed on {audio.name}: {proc.stderr.strip()[:300]}"
        )
    return proc.stdout.strip()


@dataclass(frozen=True)
class TranscriberConfig:
    sona_path: Path
    model_path: Path
    threads: int = 16
    gate: GateConfig = field(default_factory=GateConfig)

    def __post_init__(self) -> None:
        if not Path(self.sona_path).is_file():
            raise TranscriptionError(f"sona not found at {self.sona_path}")
        if not Path(self.model_path).is_file():
            raise TranscriptionError(f"whisper model not found at {self.model_path}")
        if self.threads < 1:
            raise TranscriptionError(f"threads must be >= 1, got {self.threads}")


def transcribe_file(
    audio_path: str | Path,
    config: TranscriberConfig,
    glossary: Glossary | None = None,
) -> list[TranscribedSegment]:
    """Gate, slice, transcribe and correct one recording.

    Returns one TranscribedSegment per surviving gated segment. A file whose audio is
    entirely rejected returns an empty list -- correct behaviour, and distinguishable
    from failure because failure raises.
    """
    source = Path(audio_path)
    if shutil.which("ffmpeg") is None:
        raise TranscriptionError("ffmpeg not found on PATH")

    windows = measure(source, config.gate)
    keep: list[Segment] = segments(windows, config.gate)
    if not keep:
        logger.info("%s: no transcribable audio, skipping entirely", source.name)
        return []

    total = sum(s.duration_sec for s in keep)
    logger.info(
        "%s: transcribing %.0fs of %.0fs across %d segment(s)",
        source.name, total, len(windows) * config.gate.window_sec, len(keep),
    )

    prompt = glossary.prompt_text() if glossary else ""
    results: list[TranscribedSegment] = []

    with tempfile.TemporaryDirectory(prefix="autowork-") as tmpdir:
        workdir = Path(tmpdir)
        for index, segment in enumerate(keep):
            clip = workdir / f"seg{index:03d}.wav"
            _slice_audio(source, segment.start_sec, segment.end_sec, clip)

            use_prompt = bool(prompt) and segment.verdict.allows_glossary_prompt
            text = _run_whisper(
                Path(config.sona_path), Path(config.model_path), clip,
                prompt=prompt if use_prompt else None, threads=config.threads,
            )

            text, removed = excise_repetition_loops(text)
            if removed:
                logger.warning(
                    "%s [%.0fs-%.0fs]: removed %d looped sentence(s) despite the "
                    "segment passing the gate (%s, %+.1f dB). The surrounding speech "
                    "is kept.",
                    source.name, segment.start_sec, segment.end_sec, removed,
                    segment.verdict.value, segment.mean_delta_db,
                )

            applied: list[str] = []
            if glossary and text:
                text, applied = glossary.correct(text)

            if is_annotation_only(text):
                logger.info(
                    "%s [%.0fs-%.0fs]: passed the gate (%+.1f dB) but contains no "
                    "speech, only sound-event annotations (%s). Dropping.",
                    source.name, segment.start_sec, segment.end_sec,
                    segment.mean_delta_db, text.strip(),
                )
                continue

            # Checked AFTER excision: if a loop still dominates what remains, the
            # segment is not merely tainted at one end, it is substantially fabricated.
            looping = looks_like_repetition_loop(text)
            if looping:
                logger.warning(
                    "%s [%.0fs-%.0fs]: repetition still dominates after excision. "
                    "Treat this text as unverified.",
                    source.name, segment.start_sec, segment.end_sec,
                )

            results.append(
                TranscribedSegment(
                    source_audio=str(source),
                    start_sec=segment.start_sec,
                    end_sec=segment.end_sec,
                    verdict=segment.verdict,
                    speech_rumble_db=segment.mean_delta_db,
                    text=text,
                    glossary_applied=applied,
                    prompt_used=use_prompt,
                    repetition_flagged=looping,
                )
            )

    return results
