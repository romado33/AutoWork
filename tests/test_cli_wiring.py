#!/usr/bin/env python3
"""Tests that a CLI's argument wiring still matches the config object it builds.

Run:
    python -m pytest tests/ -v

No network and no API key required.

These exist because this kind of wiring rots invisibly. tools/extract_actions.py
passed `model` and `ollama_url` into ExtractorConfig, and went on passing them after
the move off local models removed both fields. The module still imported and the
suite stayed green, because every other caller imports only parse_transcript, which
never touches the config. The break surfaced only when a human ran the tool, and it
surfaced as TypeError -- not the ExtractionError the CLI catches and reports politely.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from autowork.extract import DEFAULT_BACKEND, ExtractorConfig  # noqa: E402
from extract_actions import build_parser, config_from_args  # noqa: E402


def test_default_arguments_build_a_valid_extractor_config() -> None:
    config = config_from_args(build_parser().parse_args(["transcript.md"]))
    assert isinstance(config, ExtractorConfig)
    assert config.backend == DEFAULT_BACKEND


def test_every_field_the_cli_sets_is_a_real_config_field() -> None:
    """The precise rot that broke this CLI: a kwarg the dataclass no longer has."""
    fields = {f.name for f in dataclasses.fields(ExtractorConfig)}
    config = config_from_args(build_parser().parse_args(["transcript.md"]))
    for name in ("backend", "owner_name", "min_confidence", "timeout_sec"):
        assert name in fields, f"{name} is no longer a field on ExtractorConfig"
        assert getattr(config, name) is not None


def test_explicit_backend_overrides_the_default() -> None:
    args = build_parser().parse_args(
        ["transcript.md", "--backend", "anthropic:claude-sonnet-5"]
    )
    assert config_from_args(args).backend == "anthropic:claude-sonnet-5"


def test_backend_comes_from_the_environment_when_unset(monkeypatch) -> None:
    monkeypatch.setenv("AUTOWORK_BACKEND_EXTRACT", "anthropic:claude-sonnet-5")
    # The default is read at parser-construction time, so build after setting it.
    args = build_parser().parse_args(["transcript.md"])
    assert config_from_args(args).backend == "anthropic:claude-sonnet-5"


def test_the_cli_no_longer_accepts_a_local_model_flag() -> None:
    """Local backends were evaluated and dropped; --ollama-url must not creep back."""
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            ["transcript.md", "--ollama-url", "http://localhost:11434"]
        )
