#!/usr/bin/env python3
"""Audio quality gate. Decides what is worth transcribing, and what would be invented.

This runs BEFORE Whisper, and it is the single most important safety component in the
pipeline. Whisper does not fail on unusable audio -- it produces fluent, grammatical,
confident English that nobody said. Measured examples from this recorder: "*Police*" x6,
"city of Estrella" x14, and "So, around Monday, we're going to do a lot of things",
which reads exactly like a real work commitment and is entirely fabricated. Downstream
stages cannot distinguish that from truth. This gate is where it gets stopped.

TWO CONDITIONS, BOTH REQUIRED
-----------------------------
Calibrated against 46 minutes of real recording (2026-08-25) with known outcomes:

  1. speech_db >= SPEECH_FLOOR_DB   Is there speech energy at all?
  2. delta_db  >= DELTA_USABLE_DB   Is that energy speech rather than rumble?

Either test alone produces false positives, in opposite directions, and both were
observed in the calibration file:

  * min 6:  delta +1.5 dB (passes 2) but speech -50.3 dB (fails 1). Transcribed to
            nothing at all -- there was no conversation, only room tone.
  * min 46: speech -29.1 dB (passes 1) but delta -0.9 dB (fails 2). Loud handling and
            movement noise. Transcribed to "*Painful music*".

MEASURE RAW AUDIO ONLY
----------------------
Never run this on enhanced or denoised audio. Demucs vocal isolation raised one clip's
delta from -1.3 to +2.7 dB while its transcript stayed worthless: it cut the rumble
denominator without recovering any speech numerator. The metric predicts transcription
success on unprocessed input, and any enhancement step can inflate it without adding a
single intelligible word.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)

# Calibrated thresholds. Overridable via GateConfig; these are the measured defaults.
# Calibrated against every sample with a known transcription outcome:
#
#   speech   delta   outcome                       verdict under these thresholds
#   -22.3    +4.1    excellent (podcast control)   CLEAN
#   -27.7    +6.8    excellent (2026-08-25 mtg)    CLEAN
#   -27.4    +1.6    partial, real content         USABLE
#   -34.6    +2.2    fabricated ("*Police*" x6)    REJECT  <- caught by the floor
#   -24.8    -1.3    fabricated                    REJECT  <- caught by the delta
#
# The floor was first set at -40 dB and that was wrong: it passed the -34.6 dB case,
# which fabricates. Both failing rows are rejected only because the two conditions
# catch different failure modes -- quiet-but-clean-looking, and loud-but-rumble.
SPEECH_FLOOR_DB = -32.0
DELTA_USABLE_DB = 1.0
DELTA_CLEAN_DB = 4.0

# Fraction of a segment's windows that must individually be CLEAN for the whole segment
# to count as CLEAN. See segments() for why this is neither 1.0 nor absent.
CLEAN_WINDOW_FRACTION = 0.8

# Speech intelligibility band. Consonant detail lives here and is what Whisper needs.
SPEECH_LOW_HZ = 300
SPEECH_HIGH_HZ = 3400

# Analysis sample rate. Whisper works at 16 kHz, so measuring there measures what it
# will actually receive, even when the source is 32 kHz.
ANALYSIS_RATE_HZ = 16000

# Silence in astats is reported as -inf; represent it as a finite floor.
SILENCE_DB = -120.0

_RMS_RE = re.compile(r"RMS_level=(-?[\d.]+|-?inf)")


class GateError(RuntimeError):
    """Measurement failed. Never swallowed -- an unmeasurable file must not be
    silently treated as passing, nor silently dropped."""


class Verdict(str, Enum):
    CLEAN = "clean"
    USABLE = "usable"
    REJECT = "reject"

    @property
    def transcribable(self) -> bool:
        return self is not Verdict.REJECT

    @property
    def allows_glossary_prompt(self) -> bool:
        """Whether the decode-time glossary prompt may be applied.

        Only CLEAN. Measured: priming Whisper with domain vocabulary on a low-signal
        segment made it hallucinate those very terms and introduced a repetition loop
        it did not produce unprompted. On marginal audio the prompt makes things worse,
        so the correction pass handles USABLE segments instead.
        """
        return self is Verdict.CLEAN


@dataclass(frozen=True)
class GateConfig:
    window_sec: float = 15.0
    speech_floor_db: float = SPEECH_FLOOR_DB
    delta_usable_db: float = DELTA_USABLE_DB
    delta_clean_db: float = DELTA_CLEAN_DB
    bridge_gap_sec: float = 30.0

    def __post_init__(self) -> None:
        if self.window_sec <= 0:
            raise GateError(f"window_sec must be positive, got {self.window_sec}")
        if self.delta_clean_db < self.delta_usable_db:
            raise GateError(
                f"delta_clean_db ({self.delta_clean_db}) must be >= "
                f"delta_usable_db ({self.delta_usable_db})"
            )


@dataclass(frozen=True)
class Window:
    """One measured slice of audio."""

    index: int
    start_sec: float
    end_sec: float
    speech_db: float
    rumble_db: float

    @property
    def delta_db(self) -> float:
        """Speech-band energy minus sub-300Hz rumble. The discriminator."""
        return self.speech_db - self.rumble_db

    def verdict(self, config: GateConfig) -> Verdict:
        if self.speech_db < config.speech_floor_db:
            return Verdict.REJECT
        if self.delta_db < config.delta_usable_db:
            return Verdict.REJECT
        if self.delta_db >= config.delta_clean_db:
            return Verdict.CLEAN
        return Verdict.USABLE


@dataclass(frozen=True)
class Segment:
    """A contiguous run of transcribable audio, ready to hand to Whisper."""

    start_sec: float
    end_sec: float
    verdict: Verdict
    mean_speech_db: float
    mean_delta_db: float
    window_count: int

    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec


def _run_ffmpeg_rms(path: Path, audio_filter: str, window_sec: float) -> list[float]:
    """Per-window RMS in dB for one frequency band, in a single ffmpeg pass."""
    if shutil.which("ffmpeg") is None:
        raise GateError("ffmpeg not found on PATH; the gate cannot measure audio")

    samples = int(ANALYSIS_RATE_HZ * window_sec)
    chain = (
        f"aresample={ANALYSIS_RATE_HZ},aformat=channel_layouts=mono,"
        f"{audio_filter},asetnsamples=n={samples},"
        f"astats=metadata=1:reset=1,"
        f"ametadata=print:key=lavfi.astats.Overall.RMS_level:file=-"
    )
    cmd = ["ffmpeg", "-hide_banner", "-nostats", "-i", str(path),
           "-af", chain, "-f", "null", "-"]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except OSError as exc:
        raise GateError(f"could not run ffmpeg on {path}: {exc}") from exc

    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()[-3:]
        raise GateError(
            f"ffmpeg failed on {path} (exit {proc.returncode}): {' | '.join(tail)}"
        )

    values: list[float] = []
    for match in _RMS_RE.finditer(proc.stdout):
        token = match.group(1)
        values.append(SILENCE_DB if token.endswith("inf") else float(token))

    if not values:
        raise GateError(
            f"no RMS measurements parsed from {path}; the file may be empty or "
            f"contain no audio stream"
        )
    return values


def measure(path: str | Path, config: GateConfig | None = None) -> list[Window]:
    """Measure a file window by window. Two ffmpeg passes regardless of length."""
    config = config or GateConfig()
    audio = Path(path)
    if not audio.is_file():
        raise GateError(f"no such audio file: {audio}")

    speech = _run_ffmpeg_rms(
        audio, f"highpass=f={SPEECH_LOW_HZ},lowpass=f={SPEECH_HIGH_HZ}", config.window_sec
    )
    rumble = _run_ffmpeg_rms(audio, f"lowpass=f={SPEECH_LOW_HZ}", config.window_sec)

    if len(speech) != len(rumble):
        # Both passes decode the same file with the same windowing, so a mismatch means
        # a decode problem rather than a rounding artefact. Trim, but say so loudly.
        logger.warning(
            "band pass length mismatch on %s (speech=%d rumble=%d); truncating to the "
            "shorter. Check the file for corruption.",
            audio, len(speech), len(rumble),
        )

    return [
        Window(
            index=i,
            start_sec=i * config.window_sec,
            end_sec=(i + 1) * config.window_sec,
            speech_db=s,
            rumble_db=r,
        )
        for i, (s, r) in enumerate(zip(speech, rumble))
    ]


def segments(windows: list[Window], config: GateConfig | None = None) -> list[Segment]:
    """Merge transcribable windows into contiguous segments.

    Gaps shorter than `bridge_gap_sec` are bridged rather than split, because a natural
    pause mid-conversation is not a reason to start a new transcription context -- and
    Whisper transcribes better with surrounding context than with fragments.

    A segment is CLEAN only if it is *overwhelmingly* clean: mean delta at or above the
    clean threshold AND at least CLEAN_WINDOW_FRACTION of its windows individually
    clean. Requiring every window to be clean was the first rule tried and it is too
    strict in practice -- over a 22-minute conversation a handful of marginal windows
    downgraded a run averaging +6.8 dB, needlessly withholding the glossary prompt from
    audio that had clearly earned it. Requiring only the mean would be too loose, since
    a few very strong windows could carry a mostly-marginal run. Both conditions
    together keep the prompt off material that has not earned it without punishing long
    real conversations for a few quiet moments.
    """
    config = config or GateConfig()
    runs: list[list[Window]] = []
    current: list[Window] = []
    gap_since_last = 0.0

    for window in windows:
        if window.verdict(config).transcribable:
            if current and gap_since_last > config.bridge_gap_sec:
                runs.append(current)
                current = []
            current.append(window)
            gap_since_last = 0.0
        else:
            gap_since_last += config.window_sec
    if current:
        runs.append(current)

    out: list[Segment] = []
    for run in runs:
        speech_avg = sum(w.speech_db for w in run) / len(run)
        delta_avg = sum(w.delta_db for w in run) / len(run)
        clean_count = sum(1 for w in run if w.verdict(config) is Verdict.CLEAN)
        overwhelmingly_clean = (
            delta_avg >= config.delta_clean_db
            and clean_count / len(run) >= CLEAN_WINDOW_FRACTION
        )
        verdict = Verdict.CLEAN if overwhelmingly_clean else Verdict.USABLE
        out.append(
            Segment(
                start_sec=run[0].start_sec,
                end_sec=run[-1].end_sec,
                verdict=verdict,
                mean_speech_db=speech_avg,
                mean_delta_db=delta_avg,
                window_count=len(run),
            )
        )
    return out


def summarise(windows: list[Window], config: GateConfig | None = None) -> dict[str, int]:
    """Window counts per verdict. Used for reporting what a day's audio contained."""
    config = config or GateConfig()
    counts = {v.value: 0 for v in Verdict}
    for window in windows:
        counts[window.verdict(config).value] += 1
    return counts
