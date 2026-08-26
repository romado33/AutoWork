#!/usr/bin/env python3
"""The Executor side of the boundary: the contract every adapter must satisfy.

Core code never branches on `target_system`. It asks the registry for an executor and
calls `execute`. That is the whole point of this file: adding a system, or removing one
because you no longer have access to it, is a config and adapter change, never a change
to the pipeline.

Two invariants are enforced here rather than in each adapter, because an adapter author
should not be able to opt out of them:

  1. Only APPROVED actions are ever dispatched. An adapter cannot see a PENDING action,
     so no adapter can accidentally act on unreviewed, possibly-fabricated text.
  2. An adapter that raises is reported as a failure with context attached, never
     swallowed. `dispatch` converts an exception into a FAILED ExecutionResult carrying
     the exception text, so the queue can retry it and a human can read why.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass

from autowork.action import ActionRecord, Status


class ExecutorError(RuntimeError):
    """Raised for boundary violations: unroutable target, or a mis-staged action."""


@dataclass(frozen=True)
class ExecutionResult:
    """Outcome of one execution attempt.

    external_ref is whatever handle the target system gave back (an issue key, a commit
    sha, a file path). Kept as an opaque string so this module stays vendor-neutral.
    """

    ok: bool
    detail: str
    external_ref: str | None = None

    def __post_init__(self) -> None:
        if not self.detail.strip():
            raise ValueError("ExecutionResult.detail is required, success or failure")


class Executor(abc.ABC):
    """Adapter that turns an approved ActionRecord into a real change.

    Implementations must be side-effect-free until `execute` is called, so that
    registry construction and routing can be tested without touching live systems.
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Stable identifier recorded on the action for audit. Not a display name."""

    @abc.abstractmethod
    def can_handle(self, action: ActionRecord) -> bool:
        """Whether this adapter is able to carry out `action`.

        Checked before dispatch so a misrouted action fails loudly at the boundary
        instead of half-way through a live write.
        """

    @abc.abstractmethod
    def execute(self, action: ActionRecord) -> ExecutionResult:
        """Carry out the action. Return a result; raise only on unexpected failure."""


class Registry:
    """Maps target_system -> Executor. Built from config by the caller, not by import.

    Nothing is auto-discovered. If you want an adapter wired in, you say so explicitly,
    which means reading the config tells you the complete list of systems this machine
    can write to.
    """

    def __init__(self) -> None:
        self._by_system: dict[str, Executor] = {}

    def register(self, target_system: str, executor: Executor) -> None:
        key = target_system.strip()
        if not key:
            raise ExecutorError("target_system must be a non-empty string")
        if key in self._by_system:
            raise ExecutorError(
                f"target_system {key!r} already routed to "
                f"{self._by_system[key].name!r}; refusing to shadow it"
            )
        self._by_system[key] = executor

    def executor_for(self, action: ActionRecord) -> Executor:
        executor = self._by_system.get(action.target_system)
        if executor is None:
            raise ExecutorError(
                f"no executor registered for target_system "
                f"{action.target_system!r} (registered: {sorted(self._by_system)})"
            )
        if not executor.can_handle(action):
            raise ExecutorError(
                f"executor {executor.name!r} is registered for "
                f"{action.target_system!r} but declined action {action.id}"
            )
        return executor

    @property
    def systems(self) -> tuple[str, ...]:
        return tuple(sorted(self._by_system))


def dispatch(action: ActionRecord, registry: Registry) -> ExecutionResult:
    """Route one approved action to its executor and normalise the outcome.

    Raises ExecutorError for boundary violations (unapproved action, no route), because
    those are bugs in the caller and must not be retried silently. Adapter failures are
    returned as ExecutionResult(ok=False) so the queue can mark them FAILED and retry.
    """
    if action.status is not Status.APPROVED:
        raise ExecutorError(
            f"refusing to dispatch action {action.id} with status "
            f"{action.status.value!r}; only {Status.APPROVED.value!r} is dispatchable"
        )

    executor = registry.executor_for(action)
    try:
        return executor.execute(action)
    except Exception as exc:  # noqa: BLE001 - re-raised as data, with context, below
        return ExecutionResult(
            ok=False,
            detail=f"{executor.name} raised {type(exc).__name__}: {exc}",
        )
