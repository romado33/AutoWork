#!/usr/bin/env python3
"""Gate -> slice -> compress -> OpenAI transcription. Only pays for audio worth reading.

THE ORDER IS THE DESIGN. Three measured facts drive it:

  1. The GATE decides cost. Transcription is billed per minute of audio, so bytes are
     free and minutes are not. Today's 45-minute recording held 23 minutes of actual
     conversation; the rest was silence and pocket-handling noise. Uploading raw means
     paying roughly double.

  2. Ungated audio does not merely waste money, it POISONS the output. Whisper on
     low-signal audio returns fluent invention, not silence -- measured examples include
     "*Police*" x6 and "city of Estrella" x14. Those would flow into the summary and the
     action queue as fact. The gate is the only thing standing there.

  3. COMPRESSION is free quality-wise, within limits that were measured rather than
     assumed. On a 3-minute clip, transcripts from 16 kHz mono MP3 at 64k/32k/24k all
     landed 93-96% similar to the as-recorded 32 kHz stereo 128k baseline -- and NOT
     monotonically (64k scored below 32k), which means that band is transcription
     nondeterminism, not compression damage. Opus at 20k did genuinely degrade
     ("The board of the flight" for "They boarded a flight"), so it is not used.

     Mono is free: the device's two channels measured -91 dB apart, i.e. bit-identical
     duplicated mono from a single microphone. 16 kHz is free: Whisper resamples there
     internally, so anything higher is discarded by the model anyway.

     32 kbps is the default rather than 24 because the measurement was taken on GOOD
     audio; on marginal pocket audio compression artefacts compound with poor SNR, and
     the headroom is worth more than the extra 25% saving.
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from autowork.gate import GateConfig, GateError, Segment, measure, segments
from autowork.glossary import Glossary
from autowork.quality import assess

logger = logging.getLogger(__name__)

# OpenAI rejects requests above 25 MB. 20 MB leaves room for multipart overhead.
MAX_UPLOAD_BYTES = 20 * 1024 * 1024

TARGET_RATE_HZ = 16000
TARGET_BITRATE = "32k"

# Segments shorter than this are not uploaded. A 15-second fragment costs a round trip
# and rarely carries a usable action item; measured, the 15s leading segments of two
# real recordings were the recorder being picked up, held no speech, and returned empty.
MIN_SEGMENT_SEC = 30.0

# Maximum audio per upload. Well below the size limit on purpose: see split_for_upload
# for why duration matters more than bytes here.
MAX_UPLOAD_SEC = 600.0

DEFAULT_MODEL = "gpt-4o-transcribe"
DIARIZE_MODEL = "gpt-4o-transcribe-diarize"


class CloudTranscriptionError(RuntimeError):
    """Transcription failed. Raised rather than returning empty text, because an empty
    transcript is indistinguishable from a silent recording and would be filed as fact."""


@dataclass
class CloudSegment:
    """One gated segment, transcribed in the cloud."""

    source_audio: str
    start_sec: float
    end_sec: float
    verdict: str
    speech_rumble_db: float
    text: str
    speakers: list[dict] = field(default_factory=list)
    glossary_applied: list[str] = field(default_factory=list)
    uploaded_bytes: int = 0
    elapsed_sec: float = 0.0
    unintelligible_because: str = ""

    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec


def encode_slice(source: Path, start_sec: float, end_sec: float, dest: Path) -> int:
    """Extract one segment as 16 kHz mono MP3. Returns bytes written."""
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-ss", f"{start_sec:.3f}", "-t", f"{end_sec - start_sec:.3f}",
        "-i", str(source),
        "-ac", "1", "-ar", str(TARGET_RATE_HZ), "-b:a", TARGET_BITRATE,
        str(dest),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except OSError as exc:
        raise CloudTranscriptionError(f"cannot run ffmpeg: {exc}") from exc
    if proc.returncode != 0 or not dest.is_file():
        raise CloudTranscriptionError(
            f"ffmpeg failed slicing {source.name} "
            f"[{start_sec:.0f}s-{end_sec:.0f}s]: {proc.stderr.strip()[:250]}"
        )
    return dest.stat().st_size


def split_for_upload(
    segment: Segment,
    max_bytes: int = MAX_UPLOAD_BYTES,
    max_duration_sec: float = MAX_UPLOAD_SEC,
) -> list[tuple[float, float]]:
    """Split one gated segment into upload-sized windows.

    TWO caps, and the duration one usually binds. At 32 kbps a 20 MB budget is about 83
    minutes of audio, so size alone would send a 23-minute conversation as one request.
    That is a bad idea for reasons unrelated to size: a single long-lived HTTP request
    is fragile (a network blip loses the whole thing and all the money spent on it),
    it yields no progress until it completes, and diarization runs at roughly 2x
    realtime so a 23-minute upload is an 11-minute silent wait.

    Capping duration makes failures cheap and progress visible. Segments are cut on
    time rather than probed for size because a variable-bitrate surprise mid-upload is
    worse than a slightly conservative cut.
    """
    bytes_per_sec = 32_000 / 8  # 32 kbps
    max_sec = min(max_bytes / bytes_per_sec, max_duration_sec)
    duration = segment.end_sec - segment.start_sec

    if duration <= max_sec:
        return [(segment.start_sec, segment.end_sec)]

    windows: list[tuple[float, float]] = []
    cursor = segment.start_sec
    while cursor < segment.end_sec:
        end = min(cursor + max_sec, segment.end_sec)
        windows.append((cursor, end))
        cursor = end
    logger.info(
        "segment %.0fs-%.0fs split into %d uploads (%.0f min each max)",
        segment.start_sec, segment.end_sec, len(windows), max_sec / 60,
    )
    return windows


def _transcribe_with_retry(
    client, path: Path, model: str, attempts: int = 3
) -> tuple[str, list[dict]]:
    """Retry transient failures before giving up on a segment.

    Without this, one network blip or 429 permanently skips a segment: the pipeline is
    idempotent per RECORDING, so a re-run sees the transcript file and never retries the
    hole. Retrying twice with backoff costs nothing when things work; a deterministic
    400 pays two pointless fast-failing retries, which is an acceptable price for not
    special-casing error taxonomy.
    """
    import time as _time

    last: Exception | None = None
    for attempt in range(attempts):
        try:
            return _transcribe_file(client, path, model)
        except CloudTranscriptionError as exc:
            last = exc
            if attempt < attempts - 1:
                delay = 2 * (4**attempt)  # 2s, 8s
                logger.warning(
                    "attempt %d/%d failed for %s, retrying in %ds: %s",
                    attempt + 1, attempts, path.name, delay, str(exc)[:150],
                )
                _time.sleep(delay)
    raise last  # type: ignore[misc]


def _transcribe_file(client, path: Path, model: str) -> tuple[str, list[dict]]:
    """Send one file. Returns (text, speaker segments)."""
    kwargs: dict = {"model": model}
    if "diarize" in model:
        kwargs["response_format"] = "diarized_json"
        # Required by the diarization models: without it the request is rejected with
        # 400 invalid_value on `chunking_strategy`. "auto" lets the server pick the
        # segmentation, which is what we want -- our own gating already decided WHICH
        # audio to send, not how to split speech within it.
        kwargs["chunking_strategy"] = "auto"

    try:
        with path.open("rb") as handle:
            response = client.audio.transcriptions.create(file=handle, **kwargs)
    except Exception as exc:  # noqa: BLE001 - SDK raises several unrelated types
        raise CloudTranscriptionError(
            f"transcription of {path.name} failed with {model}: {str(exc)[:250]}"
        ) from exc

    speakers: list[dict] = []
    for raw in getattr(response, "segments", None) or []:
        speakers.append(
            {
                "speaker": getattr(raw, "speaker", None),
                "start": getattr(raw, "start", None),
                "end": getattr(raw, "end", None),
                "text": getattr(raw, "text", "") or "",
            }
        )

    text = (getattr(response, "text", None) or "").strip()
    if not text and speakers:
        # diarized_json may carry the words only in the segments.
        text = " ".join(s["text"].strip() for s in speakers if s["text"]).strip()

    # An empty result is NOT an error: it means this window held no speech, which the
    # gate cannot fully determine from level alone. Returning empty and letting the
    # caller skip the segment is the difference between losing 15 seconds and losing
    # the whole recording -- measured, an empty leading segment aborted two files and
    # threw away 22 good minutes between them.
    return text, speakers


def transcribe_file_cloud(
    audio_path: str | os.PathLike,
    *,
    model: str = DEFAULT_MODEL,
    gate: GateConfig | None = None,
    glossary: Glossary | None = None,
    client=None,
) -> list[CloudSegment]:
    """Gate a recording, then transcribe only the segments worth paying for."""
    import time

    source = Path(audio_path)
    if not source.is_file():
        raise CloudTranscriptionError(f"no such audio file: {source}")

    if client is None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise CloudTranscriptionError(
                "the openai SDK is not installed: pip install openai"
            ) from exc
        if not os.environ.get("OPENAI_API_KEY"):
            raise CloudTranscriptionError("OPENAI_API_KEY is not set")
        client = OpenAI()

    gate = gate or GateConfig()
    try:
        windows = measure(source, gate)
        keep = segments(windows, gate)
    except GateError as exc:
        raise CloudTranscriptionError(f"gating failed: {exc}") from exc

    if not keep:
        logger.info("%s: gate rejected everything; nothing uploaded", source.name)
        return []

    total_audio = sum(s.duration_sec for s in keep)
    logger.info(
        "%s: uploading %.0f min of %.0f min (%d segment(s))",
        source.name, total_audio / 60,
        len(windows) * gate.window_sec / 60, len(keep),
    )

    # Encode and upload OVERLAP. Diarization is ~2x realtime per request, so
    # sequential uploads make a 23-minute conversation an 11-minute wait; four
    # workers cut that to roughly the longest single chunk. Encoding every slice
    # first left the network idle for those few seconds of ffmpeg; submitting each
    # clip as soon as it exists overlaps the rest of the encodes with the first
    # uploads. Four, not more: enough to matter, small enough not to trip
    # per-minute rate limits, and each worker holds one clip's transcription in
    # memory at most.
    from concurrent.futures import Future, ThreadPoolExecutor

    def run_job(job) -> CloudSegment | None:
        _order, start, end, segment, clip, size = job
        started = time.monotonic()
        try:
            text, speakers = _transcribe_with_retry(client, clip, model)
        except CloudTranscriptionError as exc:
            # One bad segment must not cost the whole recording. Report it and
            # keep going; a partial transcript beats none, and the log shows
            # which window is missing.
            logger.warning("skipping %.0fs-%.0fs: %s", start, end, str(exc)[:200])
            return None
        elapsed = time.monotonic() - started

        if not text.strip():
            logger.info(
                "%.0fs-%.0fs passed the gate (%+.1f dB) but held no speech; "
                "skipping", start, end, segment.mean_delta_db,
            )
            return None

        text, unintelligible = assess(text, speakers)

        applied: list[str] = []
        if glossary:
            # Glossary.correct compiles fresh patterns per call and mutates
            # nothing shared, so calling it from worker threads is safe.
            text, applied = glossary.correct(text)

        logger.info(
            "transcribed %.0fs-%.0fs: %d chars from %.1f MB in %.1fs",
            start, end, len(text), size / 1e6, elapsed,
        )
        return CloudSegment(
            source_audio=str(source),
            start_sec=start,
            end_sec=end,
            verdict=segment.verdict.value,
            speech_rumble_db=segment.mean_delta_db,
            text=text,
            speakers=speakers,
            glossary_applied=applied,
            uploaded_bytes=size,
            elapsed_sec=elapsed,
            unintelligible_because=unintelligible or "",
        )

    futures: list[Future] = []
    with tempfile.TemporaryDirectory(prefix="autowork-cloud-") as tmp:
        workdir = Path(tmp)
        # Pool lives inside the tempdir so clips still exist when workers read them.
        with ThreadPoolExecutor(max_workers=4) as pool:
            order = 0
            for index, segment in enumerate(keep):
                if segment.duration_sec < MIN_SEGMENT_SEC:
                    logger.info(
                        "skipping %.0fs-%.0fs: %.0fs is below the %.0fs minimum",
                        segment.start_sec, segment.end_sec,
                        segment.duration_sec, MIN_SEGMENT_SEC,
                    )
                    continue
                for part, (start, end) in enumerate(split_for_upload(segment)):
                    clip = workdir / f"seg{index:03d}_{part:02d}.mp3"
                    size = encode_slice(source, start, end, clip)
                    if size > MAX_UPLOAD_BYTES:
                        raise CloudTranscriptionError(
                            f"{clip.name} is {size / 1e6:.1f} MB, above the "
                            f"{MAX_UPLOAD_BYTES / 1e6:.0f} MB upload budget"
                        )
                    job = (order, start, end, segment, clip, size)
                    futures.append(pool.submit(run_job, job))
                    order += 1
            # Submission order, not completion order: the transcript stays
            # chronological even though the network finishes chunks out of order.
            outcomes = [fut.result() for fut in futures]

    return [outcome for outcome in outcomes if outcome is not None]
