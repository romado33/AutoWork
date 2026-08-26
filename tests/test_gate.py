#!/usr/bin/env python3
"""Tests for the audio quality gate.

Run:
    python -m pytest tests/ -v

The calibration table below is the important part of this file. Every row is a real
measurement from a real recording whose transcription outcome is known, including two
that fabricated text. If a threshold change breaks one of these rows, it is letting
fabricated transcripts back into the pipeline -- do not adjust the row to match the
code, adjust the code or bring new measured evidence.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

from autowork.gate import (
    CLEAN_WINDOW_FRACTION,
    GateConfig,
    GateError,
    Verdict,
    Window,
    measure,
    segments,
    summarise,
)


def window(speech_db: float, rumble_db: float, index: int = 0) -> Window:
    return Window(
        index=index,
        start_sec=index * 15.0,
        end_sec=(index + 1) * 15.0,
        speech_db=speech_db,
        rumble_db=rumble_db,
    )


# (label, speech_db, delta_db, expected verdict)
CALIBRATION = [
    ("podcast control, transcribed excellently", -22.3, 4.1, Verdict.CLEAN),
    ("2026-08-25 meeting, transcribed excellently", -27.7, 6.8, Verdict.CLEAN),
    ("file1 min49, partial but real content", -27.4, 1.6, Verdict.USABLE),
    ("file1 min29, fabricated '*Police*' x6", -34.6, 2.2, Verdict.REJECT),
    ("file3 min32, fabricated despite being loud", -24.8, -1.3, Verdict.REJECT),
]


@pytest.mark.parametrize(
    ("label", "speech_db", "delta_db", "expected"),
    CALIBRATION,
    ids=[row[0] for row in CALIBRATION],
)
def test_calibration_against_known_outcomes(
    label: str, speech_db: float, delta_db: float, expected: Verdict
) -> None:
    """Every row is a measured sample whose transcript we have actually read."""
    w = window(speech_db=speech_db, rumble_db=speech_db - delta_db)
    assert w.verdict(GateConfig()) is expected, label


def test_the_two_conditions_catch_different_failure_modes() -> None:
    """The reason both conditions exist: neither alone rejects both failures."""
    config = GateConfig()
    quiet_but_positive_delta = window(speech_db=-34.6, rumble_db=-36.8)  # delta +2.2
    loud_but_rumble_dominated = window(speech_db=-24.8, rumble_db=-23.5)  # delta -1.3

    assert quiet_but_positive_delta.delta_db > config.delta_usable_db
    assert quiet_but_positive_delta.speech_db < config.speech_floor_db

    assert loud_but_rumble_dominated.speech_db > config.speech_floor_db
    assert loud_but_rumble_dominated.delta_db < config.delta_usable_db

    assert quiet_but_positive_delta.verdict(config) is Verdict.REJECT
    assert loud_but_rumble_dominated.verdict(config) is Verdict.REJECT


def test_delta_is_derived_from_the_two_bands() -> None:
    assert window(speech_db=-20.0, rumble_db=-26.0).delta_db == pytest.approx(6.0)


def test_only_clean_unlocks_the_glossary_prompt() -> None:
    """Measured: priming marginal audio with the glossary made it hallucinate those
    terms and introduced a repetition loop it did not produce unprompted."""
    assert Verdict.CLEAN.allows_glossary_prompt
    assert not Verdict.USABLE.allows_glossary_prompt
    assert not Verdict.REJECT.allows_glossary_prompt
    assert Verdict.USABLE.transcribable
    assert not Verdict.REJECT.transcribable


# --- configuration ---------------------------------------------------------------


def test_config_rejects_inverted_thresholds() -> None:
    """Expected failure: clean must not be easier to reach than usable."""
    with pytest.raises(GateError, match="delta_clean_db"):
        GateConfig(delta_usable_db=5.0, delta_clean_db=2.0)


def test_config_rejects_nonpositive_window() -> None:
    """Expected failure."""
    with pytest.raises(GateError, match="window_sec"):
        GateConfig(window_sec=0)


# --- segmentation ----------------------------------------------------------------


def test_segments_merge_contiguous_and_bridge_short_gaps() -> None:
    """Happy path: a natural pause mid-conversation must not split the segment."""
    config = GateConfig(window_sec=15.0, bridge_gap_sec=30.0)
    good = (-25.0, -30.0)  # delta +5.0, CLEAN
    dead = (-60.0, -61.0)  # below the floor, REJECT

    windows = [
        window(*good, index=0),
        window(*good, index=1),
        window(*dead, index=2),  # 15s gap, under the 30s bridge
        window(*good, index=3),
        window(*good, index=4),
    ]
    result = segments(windows, config)

    assert len(result) == 1
    assert result[0].start_sec == 0.0
    assert result[0].end_sec == 75.0
    assert result[0].verdict is Verdict.CLEAN


def test_segments_split_on_a_long_gap() -> None:
    """Edge: a gap longer than bridge_gap_sec is a real boundary."""
    config = GateConfig(window_sec=15.0, bridge_gap_sec=30.0)
    good = (-25.0, -30.0)
    dead = (-60.0, -61.0)

    windows = [
        window(*good, index=0),
        *[window(*dead, index=i) for i in range(1, 5)],  # 60s gap, over the bridge
        window(*good, index=5),
    ]
    result = segments(windows, config)

    assert len(result) == 2


def test_a_few_marginal_windows_do_not_downgrade_a_strong_segment() -> None:
    """Edge: the rule that 'all windows must be clean' was too strict in practice.

    A 22-minute real conversation averaging +6.8 dB was downgraded by a handful of
    quiet moments, needlessly withholding the glossary prompt.
    """
    config = GateConfig(window_sec=15.0)
    clean = (-25.0, -31.0)  # delta +6.0
    marginal = (-28.0, -30.0)  # delta +2.0, USABLE

    windows = [window(*clean, index=i) for i in range(9)]
    windows.append(window(*marginal, index=9))  # 90% clean, above the fraction

    result = segments(windows, config)
    assert len(result) == 1
    assert result[0].verdict is Verdict.CLEAN


def test_a_mostly_marginal_segment_stays_usable() -> None:
    """Edge, the other direction: strong windows must not carry a weak run to CLEAN."""
    config = GateConfig(window_sec=15.0)
    very_strong = (-22.0, -40.0)  # delta +18.0, pulls the mean up hard
    marginal = (-28.0, -30.0)  # delta +2.0

    windows = [window(*very_strong, index=0)]
    windows += [window(*marginal, index=i) for i in range(1, 6)]

    result = segments(windows, config)
    assert len(result) == 1
    clean_fraction = 1 / 6
    assert clean_fraction < CLEAN_WINDOW_FRACTION
    assert result[0].verdict is Verdict.USABLE


def test_no_transcribable_windows_yields_no_segments() -> None:
    """Edge: a silent hour must produce nothing, not an empty-but-present segment."""
    dead = [window(-60.0, -61.0, index=i) for i in range(10)]
    assert segments(dead, GateConfig()) == []


def test_summarise_counts_every_verdict() -> None:
    windows = [
        window(-25.0, -31.0, index=0),  # CLEAN
        window(-27.4, -29.0, index=1),  # USABLE
        window(-60.0, -61.0, index=2),  # REJECT
    ]
    assert summarise(windows, GateConfig()) == {"clean": 1, "usable": 1, "reject": 1}


# --- measurement -----------------------------------------------------------------


def test_measure_missing_file_raises() -> None:
    """Expected failure: an unmeasurable file must never be treated as passing."""
    with pytest.raises(GateError, match="no such audio file"):
        measure("does/not/exist.mp3")


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")
def test_measure_end_to_end_on_generated_audio(tmp_path) -> None:
    """Happy path against real ffmpeg: silence must be rejected, not invented over."""
    silent = tmp_path / "silence.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
         "anullsrc=r=16000:cl=mono", "-t", "30", str(silent)],
        check=True, capture_output=True,
    )

    windows = measure(silent, GateConfig(window_sec=15.0))

    assert len(windows) >= 2
    assert all(w.verdict(GateConfig()) is Verdict.REJECT for w in windows)
    assert segments(windows, GateConfig()) == []
