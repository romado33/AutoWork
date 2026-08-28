#!/usr/bin/env python3
"""The 7:30 digest must use the project venv, not a bare C:\\Python313.

A scheduled task pointing at a Python without openai/PyYAML fails at 7:30 with
an import error, not a mailer error. scripts\\setup.bat and scripts\\review.bat
already prefer .venv; the installer and watcher defaults must match.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INSTALLER = (ROOT / "scripts" / "Install-Watcher.ps1").read_text(encoding="utf-8")
WATCHER = (ROOT / "scripts" / "Watch-Recorder.ps1").read_text(encoding="utf-8")


def _param_block(text: str) -> str:
    start = text.index("param(")
    depth = 0
    for index, char in enumerate(text[start:], start):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return text[start:index + 1]
    raise AssertionError("unclosed param(")


def test_installer_prefers_venv_python_over_a_hardcoded_install() -> None:
    assert r".venv\Scripts\python.exe" in INSTALLER
    param_block = _param_block(INSTALLER)
    assert "C:\\Python313" not in param_block


def test_installer_logs_digest_via_cmd_redirection() -> None:
    """Same Windows gotcha as the watcher: Python logs on stderr."""
    assert "send_queue_digest.py" in INSTALLER
    assert "cmd /c" in INSTALLER
    assert "digest.log" in INSTALLER


def test_watcher_resolves_venv_when_python_is_omitted() -> None:
    assert r".venv\Scripts\python.exe" in WATCHER
    assert "C:\\Python313" not in _param_block(WATCHER)
