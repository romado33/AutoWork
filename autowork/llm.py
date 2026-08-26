#!/usr/bin/env python3
"""Interchangeable cloud LLM backends behind one interface: OpenAI or Claude.

The extractor does not know which one it is talking to. That is the point -- the choice
of provider is a config string, and the grounding, validation and queue logic downstream
is identical either way. It is also what keeps the project portable: no vendor is load-
bearing, and switching is one argument, not a refactor.

MEASURED on this machine (2,600-character excerpt containing 3 real action items):

    gemma3:4b   (local)   358s   found 1 of 3
    qwen2.5:7b  (local)   493s   found 0 of 3
    phi4        (local)   never finished (8,150 CPU-seconds, abandoned)

Local extraction runs roughly 2x SLOWER than realtime, so a day's conversation takes
longer to process than it took to have -- and the bigger local model was worse, not
better. Local inference was removed on that evidence. It is not a preference: the local
path cleared neither the latency nor the accuracy bar, and the same transcript through
gpt-5.4-mini took 7.2s and found all three items.

PRIVACY, stated precisely. A backend sends only what the prefilter selected -- commitment-bearing passages, roughly 18% of a
transcript -- never the full transcript, and never the audio. Paid API tiers are
contractually excluded from model training; free tiers generally are not, which is why
"free cloud" is the worst of the three options rather than the cheapest.
"""

from __future__ import annotations

import abc
import json
import logging
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Sonnet 5. Chosen explicitly by the operator; Haiku 4.5 is roughly half the price and
# worth measuring against this if cost ever matters more than quality here.
DEFAULT_CLOUD_MODEL = "claude-sonnet-5"

# OpenAI default. The mini tier is the right starting point: extraction is a simple,
# well-specified task, and the operator is paying per token from a personal account.
DEFAULT_OPENAI_MODEL = "gpt-5.4-mini"


class LLMError(RuntimeError):
    """The backend failed. Raised, never turned into an empty result, because
    "no action items found" and "the model was unreachable" must not look the same."""


@dataclass(frozen=True)
class Completion:
    """A backend's reply, plus what it cost, so the two can be compared honestly."""

    text: str
    backend: str
    model: str
    elapsed_sec: float
    input_tokens: int | None = None
    output_tokens: int | None = None

    @property
    def label(self) -> str:
        return f"{self.backend}:{self.model}"


class Backend(abc.ABC):
    """A prompt goes in, JSON text comes out. Nothing else is shared."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Stable identifier, recorded on the ActionRecord for audit."""

    @property
    @abc.abstractmethod
    def model(self) -> str: ...

    @abc.abstractmethod
    def complete(self, prompt: str, schema: dict | None = None) -> Completion:
        """Return the model's reply as JSON text.

        `schema` is a JSON Schema the reply must conform to. Backends that can enforce
        it server-side should; those that cannot must still ask for JSON, because the
        caller parses the result either way.
        """


class AnthropicBackend(Backend):
    """Claude API. Fast and accurate; sends the prefiltered passages off the machine.

    Uses server-enforced structured output (`output_config.format`), so the reply is
    guaranteed to be schema-valid JSON rather than something we hope parses. Adaptive
    thinking at low effort: extraction is a simple, well-specified task, and effort is
    what latency is most sensitive to.
    """

    def __init__(
        self,
        model: str = DEFAULT_CLOUD_MODEL,
        api_key: str | None = None,
        max_tokens: int = 8000,
        effort: str = "low",
    ) -> None:
        try:
            import anthropic
        except ImportError as exc:
            raise LLMError(
                "the anthropic SDK is not installed: pip install anthropic"
            ) from exc

        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise LLMError(
                "no Claude API credential found. Set ANTHROPIC_API_KEY, or run "
                "`ant auth login` and the SDK will pick up the profile automatically."
            )

        self._client = anthropic.Anthropic(api_key=key)
        self._model = model
        self.max_tokens = max_tokens
        self.effort = effort

    @property
    def name(self) -> str:
        return "anthropic"

    @property
    def model(self) -> str:
        return self._model

    def complete(self, prompt: str, schema: dict | None = None) -> Completion:
        import anthropic

        kwargs: dict = {
            "model": self._model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": self.effort},
        }
        if schema is not None:
            kwargs["output_config"]["format"] = {"type": "json_schema", "schema": schema}

        started = time.monotonic()
        try:
            response = self._client.messages.create(**kwargs)
        except anthropic.APIStatusError as exc:
            raise LLMError(
                f"Claude API returned {exc.status_code} for {self._model!r}: "
                f"{str(exc)[:300]}"
            ) from exc
        except anthropic.APIConnectionError as exc:
            raise LLMError(f"cannot reach the Claude API: {exc}") from exc

        # A safety refusal arrives as HTTP 200 with stop_reason "refusal", so checking
        # content without checking stop_reason first reads an empty response as "no
        # action items found".
        if getattr(response, "stop_reason", None) == "refusal":
            detail = getattr(response, "stop_details", None)
            raise LLMError(f"the model declined this request: {detail}")

        text = next(
            (block.text for block in response.content if block.type == "text"), ""
        )
        if not text.strip():
            raise LLMError(
                f"Claude returned no text block (stop_reason="
                f"{getattr(response, 'stop_reason', '?')})"
            )

        usage = getattr(response, "usage", None)
        return Completion(
            text=text,
            backend=self.name,
            model=self._model,
            elapsed_sec=time.monotonic() - started,
            input_tokens=getattr(usage, "input_tokens", None),
            output_tokens=getattr(usage, "output_tokens", None),
        )


class OpenAIBackend(Backend):
    """OpenAI chat completions with server-enforced JSON schema.

    Separate from AnthropicBackend rather than a shared "cloud" class on purpose: the
    two APIs differ in ways that matter (parameter names, how structured output is
    declared, whether temperature is accepted at all), and papering over that with
    conditionals is how subtle wrong-parameter bugs get in.

    The gpt-5 family rejects `temperature` outright, so it is only sent for older
    models. Determinism there comes from the schema constraint instead.
    """

    def __init__(
        self,
        model: str = "gpt-5.4-mini",
        api_key: str | None = None,
        max_tokens: int = 8000,
        reasoning_effort: str = "low",
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise LLMError("the openai SDK is not installed: pip install openai") from exc

        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise LLMError("no OpenAI credential found. Set OPENAI_API_KEY.")

        self._client = OpenAI(api_key=key)
        self._model = model
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort

    @property
    def name(self) -> str:
        return "openai"

    @property
    def model(self) -> str:
        return self._model

    @property
    def _is_reasoning_family(self) -> bool:
        return self._model.startswith(("gpt-5", "o1", "o3", "o4"))

    def complete(self, prompt: str, schema: dict | None = None) -> Completion:
        import openai

        kwargs: dict = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
        }

        # The reasoning family renamed the output cap and refuses temperature.
        if self._is_reasoning_family:
            kwargs["max_completion_tokens"] = self.max_tokens
            kwargs["reasoning_effort"] = self.reasoning_effort
        else:
            kwargs["max_tokens"] = self.max_tokens
            kwargs["temperature"] = 0

        if schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "actions", "strict": True, "schema": schema},
            }
        else:
            kwargs["response_format"] = {"type": "json_object"}

        started = time.monotonic()
        try:
            response = self._client.chat.completions.create(**kwargs)
        except openai.APIStatusError as exc:
            raise LLMError(
                f"OpenAI returned {exc.status_code} for {self._model!r}: {str(exc)[:300]}"
            ) from exc
        except openai.APIConnectionError as exc:
            raise LLMError(f"cannot reach the OpenAI API: {exc}") from exc

        choice = response.choices[0]
        # A truncated reply is invalid JSON that would look like a parse failure, so
        # name the real cause here instead.
        if choice.finish_reason == "length":
            raise LLMError(
                f"{self._model} hit the {self.max_tokens}-token output cap; "
                f"the reply is truncated and cannot be parsed"
            )

        text = choice.message.content or ""
        if not text.strip():
            raise LLMError(
                f"{self._model} returned no content (finish_reason="
                f"{choice.finish_reason})"
            )

        usage = response.usage
        return Completion(
            text=text,
            backend=self.name,
            model=self._model,
            elapsed_sec=time.monotonic() - started,
            input_tokens=getattr(usage, "prompt_tokens", None),
            output_tokens=getattr(usage, "completion_tokens", None),
        )


def load_dotenv(path: str | os.PathLike = ".env") -> None:
    """Load KEY=VALUE lines into the environment. Existing values are not overwritten.

    Deliberately minimal and dependency-free. Not overwriting means an explicitly
    exported key always beats a stale one left in a file.
    """
    from pathlib import Path

    source = Path(path)
    if not source.is_file():
        return
    for line in source.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        name = name.strip()
        if name and name not in os.environ:
            os.environ[name] = value.strip().strip('"').strip("'")


def build_backend(spec: str, **kwargs) -> Backend:
    """Build a backend from a "kind:model" string, e.g. "ollama:gemma3:4b".

    A string keeps the choice in config and out of the call sites, which is what makes
    local-versus-cloud a one-line change rather than a refactor.
    """
    kind, _, model = spec.partition(":")
    kind = kind.strip().lower()

    if kind in {"anthropic", "claude"}:
        return AnthropicBackend(model=model or DEFAULT_CLOUD_MODEL, **kwargs)
    if kind in {"openai", "gpt"}:
        return OpenAIBackend(model=model or DEFAULT_OPENAI_MODEL, **kwargs)
    raise LLMError(
        f"unknown backend {kind!r}; expected 'openai' or 'anthropic' "
        f"(got spec {spec!r})"
    )
