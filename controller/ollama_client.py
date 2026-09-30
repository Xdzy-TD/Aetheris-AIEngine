"""
Ollama HTTP Client — async interface to a local Ollama instance.

Wraps the Ollama REST API (``http://localhost:11434``) for:
- Chat completions with structured JSON / tool-calling output
- Model listing and pulling
- Health checks and connectivity probing

The client is the foundational layer for the LLM-driven planner.
When Ollama is unreachable, every method raises :class:`OllamaUnavailableError`
so the caller can fall back to deterministic fallback routing.

Environment variables:
    OLLAMA_BASE_URL   Base URL for the Ollama server (default: http://localhost:11434)
    AETHERIS_MODEL    Override the default model name
"""

from __future__ import annotations

import json
import os
from typing import Any, cast

import httpx
import structlog

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_BASE_URL = "http://localhost:11434"
_MODEL_PREFERENCE = [
    "qwen2.5:7b",
    "qwen3.8:27b",
    "llama3.1:8b",
    "mistral:7b",
    "llama3:8b",
    "qwen2.5:3b",
]
_DEFAULT_TIMEOUT = 120.0  # seconds — LLM inference can be slow


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class OllamaUnavailableError(ConnectionError):
    """Raised when the Ollama server is unreachable."""


class OllamaModelError(RuntimeError):
    """Raised when the requested model is not available."""


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class OllamaClient:
    """Async HTTP client for the Ollama REST API.

    Usage::

        client = OllamaClient()
        if await client.health_check():
            response = await client.chat(
                messages=[{"role": "user", "content": "Hello"}],
            )
    """

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        """
        Args:
            base_url: Ollama server URL.  Reads ``OLLAMA_BASE_URL`` env var.
            model:    Model name to use.  Reads ``AETHERIS_MODEL`` env var.
            timeout:  HTTP timeout in seconds.
        """
        self.base_url = (
            base_url
            or os.environ.get("OLLAMA_BASE_URL")
            or _DEFAULT_BASE_URL
        ).rstrip("/")

        self._configured_model = (
            model
            or os.environ.get("AETHERIS_MODEL")
        )
        self.timeout = timeout
        self._resolved_model: str | None = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def model(self) -> str:
        """The currently resolved model name."""
        if self._resolved_model:
            return self._resolved_model
        if self._configured_model:
            return self._configured_model
        return _MODEL_PREFERENCE[0]

    # ------------------------------------------------------------------
    # Health / discovery
    # ------------------------------------------------------------------

    async def health_check(self) -> bool:
        """Return True if the Ollama server is reachable."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self.base_url}/api/tags")
                return resp.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException, OSError):
            return False

    async def list_models(self) -> list[dict[str, Any]]:
        """Return a list of locally available models.

        Each entry has at least ``name``, ``size``, ``modified_at``.

        Raises:
            OllamaUnavailableError: If the server is unreachable.
        """
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{self.base_url}/api/tags")
                resp.raise_for_status()
                data = resp.json()
                return cast("list[dict[str, Any]]", data.get("models", []))
        except (httpx.ConnectError, httpx.TimeoutException, OSError) as exc:
            raise OllamaUnavailableError(
                f"Cannot reach Ollama at {self.base_url}: {exc}"
            ) from exc

    async def resolve_model(self) -> str:
        """Pick the best available model from the preference list.

        If ``AETHERIS_MODEL`` or ``model`` was explicitly set and is
        available, use that.  Otherwise walk the preference list.

        Returns:
            The resolved model name.

        Raises:
            OllamaModelError: If no suitable model is found.
            OllamaUnavailableError: If the server is unreachable.
        """
        models = await self.list_models()
        available_names = {m["name"] for m in models}
        # Also accept names without the tag (e.g. "qwen2.5" matches "qwen2.5:7b")
        available_short = {m["name"].split(":")[0] for m in models}

        # Check explicit configuration first
        if self._configured_model:
            if (
                self._configured_model in available_names
                or self._configured_model in available_short
            ):
                self._resolved_model = self._configured_model
                logger.info("ollama_model_resolved", model=self._resolved_model)
                return self._resolved_model

            # Try partial match: configured "qwen2.5" matches "qwen2.5:7b"
            for m in models:
                if m["name"].startswith(self._configured_model):
                    self._resolved_model = m["name"]
                    logger.info(
                        "ollama_model_partial_match",
                        configured=self._configured_model,
                        resolved=self._resolved_model,
                    )
                    return self._resolved_model

        # Walk the preference list
        for pref in _MODEL_PREFERENCE:
            if pref in available_names:
                self._resolved_model = pref
                logger.info("ollama_model_resolved", model=self._resolved_model)
                return self._resolved_model
            # Partial match
            for m in models:
                if m["name"].startswith(pref.split(":")[0]):
                    self._resolved_model = m["name"]
                    logger.info("ollama_model_resolved_partial", model=self._resolved_model)
                    return self._resolved_model

        # Last resort: use first available
        if models:
            self._resolved_model = models[0]["name"]
            logger.warning(
                "ollama_model_fallback_first_available",
                model=self._resolved_model,
            )
            return self._resolved_model

        raise OllamaModelError(
            "No models available on the Ollama server. "
            f"Pull one with: ollama pull {_MODEL_PREFERENCE[0]}"
        )

    async def pull_model(self, model_name: str | None = None) -> dict[str, Any]:
        """Trigger a model download on the Ollama server.

        Args:
            model_name: Model to pull (default: first from preference list).

        Returns:
            Status dict from Ollama.

        Raises:
            OllamaUnavailableError: If the server is unreachable.
        """
        name = model_name or _MODEL_PREFERENCE[0]
        logger.info("ollama_pulling_model", model=name)

        try:
            async with httpx.AsyncClient(timeout=600.0) as client:
                resp = await client.post(
                    f"{self.base_url}/api/pull",
                    json={"name": name, "stream": False},
                )
                resp.raise_for_status()
                return cast("dict[str, Any]", resp.json())
        except (httpx.ConnectError, httpx.TimeoutException, OSError) as exc:
            raise OllamaUnavailableError(
                f"Cannot reach Ollama at {self.base_url}: {exc}"
            ) from exc

    async def _ensure_model(self) -> str:
        """Resolve the model against what's actually pulled, once, and cache it.

        Without this, an unset ``model``/``AETHERIS_MODEL`` silently falls
        back to ``_MODEL_PREFERENCE[0]`` even if that model was never
        pulled — every chat call then 404s and the caller falls back to
        deterministic fallback routing with no clear error.
        """
        if self._resolved_model or self._configured_model:
            return self._resolved_model or cast(str, self._configured_model)
        try:
            return await self.resolve_model()
        except (OllamaModelError, OllamaUnavailableError):
            return _MODEL_PREFERENCE[0]

    # ------------------------------------------------------------------
    # Chat / completion
    # ------------------------------------------------------------------

    async def chat(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        format: str | dict | None = "json",
        temperature: float = 0.1,
        max_tokens: int = 2048,
        system: str | None = None,
    ) -> dict[str, Any]:
        """Send a chat completion request to Ollama.

        Args:
            messages:    Chat message list (role/content dicts).
            model:       Model override (default: resolved model).
            format:      Output format — ``"json"`` for structured output.
            temperature: Sampling temperature.
            max_tokens:  Maximum generation length.
            system:      System prompt (prepended as a system message).

        Returns:
            The full Ollama response dict containing at least
            ``message.content``.

        Raises:
            OllamaUnavailableError: If the server is unreachable.
        """
        model_name = model or await self._ensure_model()

        # Prepend system message if provided
        full_messages = list(messages)
        if system:
            full_messages.insert(0, {"role": "system", "content": system})

        payload: dict[str, Any] = {
            "model": model_name,
            "messages": full_messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if format is not None:
            payload["format"] = format

        logger.debug(
            "ollama_chat_request",
            model=model_name,
            n_messages=len(full_messages),
        )

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                )
                resp.raise_for_status()
                data = resp.json()

            logger.debug(
                "ollama_chat_response",
                model=data.get("model"),
                eval_count=data.get("eval_count"),
            )
            return cast("dict[str, Any]", data)

        except (httpx.ConnectError, httpx.TimeoutException, OSError) as exc:
            raise OllamaUnavailableError(
                f"Cannot reach Ollama at {self.base_url}: {exc}"
            ) from exc

    async def chat_json(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        temperature: float = 0.1,
        system: str | None = None,
    ) -> dict[str, Any]:
        """Chat with structured JSON output, parsed into a dict.

        Convenience wrapper around :meth:`chat` that extracts and parses
        the JSON content from the response.

        Returns:
            Parsed JSON dict from the model's response.

        Raises:
            json.JSONDecodeError: If the model's output is not valid JSON.
            OllamaUnavailableError: If the server is unreachable.
        """
        response = await self.chat(
            messages=messages,
            model=model,
            format="json",
            temperature=temperature,
            system=system,
        )

        content = response.get("message", {}).get("content", "{}")

        # Strip markdown fences if the model wraps output
        content = content.strip()
        if content.startswith("```"):
            lines = content.split("\n")
            # Remove first and last lines (fences)
            lines = [line for line in lines if not line.strip().startswith("```")]
            content = "\n".join(lines)

        return cast("dict[str, Any]", json.loads(content))

    # ------------------------------------------------------------------
    # Generate (non-chat)
    # ------------------------------------------------------------------

    async def generate(
        self,
        prompt: str,
        model: str | None = None,
        system: str | None = None,
        format: str | dict | None = None,
        temperature: float = 0.1,
    ) -> str:
        """Send a raw generate request (non-chat).

        Returns:
            The generated text.

        Raises:
            OllamaUnavailableError: If the server is unreachable.
        """
        model_name = model or await self._ensure_model()

        payload: dict[str, Any] = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if system:
            payload["system"] = system
        if format:
            payload["format"] = format

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                )
                resp.raise_for_status()
                data = resp.json()
                return cast("str", data.get("response", ""))

        except (httpx.ConnectError, httpx.TimeoutException, OSError) as exc:
            raise OllamaUnavailableError(
                f"Cannot reach Ollama at {self.base_url}: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    async def status(self) -> dict[str, Any]:
        """Return a status summary suitable for an API endpoint.

        Returns:
            Dict with connectivity, model, and available models info.
        """
        reachable = await self.health_check()
        result: dict[str, Any] = {
            "reachable": reachable,
            "base_url": self.base_url,
            "configured_model": self._configured_model,
            "resolved_model": self._resolved_model,
        }

        if reachable:
            try:
                models = await self.list_models()
                result["available_models"] = [m["name"] for m in models]
                result["model_count"] = len(models)
            except OllamaUnavailableError:
                result["available_models"] = []
                result["model_count"] = 0

        return result
