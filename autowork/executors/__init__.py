"""Executor adapters. Nothing here is auto-registered; wiring is explicit in config."""

from autowork.executors.base import (
    ExecutionResult,
    Executor,
    ExecutorError,
    Registry,
    dispatch,
)
from autowork.executors.dry_run import DryRunExecutor
from autowork.executors.file_handoff import FileHandoffExecutor

__all__ = [
    "DryRunExecutor",
    "ExecutionResult",
    "Executor",
    "ExecutorError",
    "FileHandoffExecutor",
    "Registry",
    "dispatch",
]
