"""Local-only Ollama client for RouteSentinel.

Hard constraints (AGENTS.md):
- Never call paid/hosted APIs. Only a local Ollama server is used.
- Deterministic decoding: temperature=0, top_k=1, seed=42.
- Every call is cached on sha256(prompt + model digest + params) under
  ``outputs/.ollama_cache/``. Cache hits are served even when the server
  is down.
- If Ollama is unavailable, :meth:`OllamaClient.generate` raises
  :class:`LLMUnavailable` and the caller falls back silently to the
  deterministic template. No fake LLM output is ever produced.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

from sentinel.llm.ledger import count_tokens, log_cache_hit, log_call

MODEL_NAME = "qwen2.5:3b-instruct-q4_K_M"
TEMPERATURE = 0.0
TOP_K = 1
SEED = 42
REQUEST_TIMEOUT_S = 180
MAX_ATTEMPTS = 2  # initial try + one retry


class LLMUnavailable(Exception):
    """The local LLM cannot produce output.

    Raised when the ``ollama`` package is missing, the server is
    unreachable, or generation fails twice. Callers must fall back
    silently to deterministic templates.
    """


def default_cache_dir() -> Path:
    """Repo-level default on-disk cache directory."""
    return Path(__file__).resolve().parents[3] / "outputs" / ".ollama_cache"


def cache_key_for(prompt: str, model: str, params: dict[str, Any]) -> str:
    """sha256(prompt + model digest + params); params serialised with sorted keys."""
    digest = json.dumps({"model": model, "params": params}, sort_keys=True)
    return hashlib.sha256((prompt + "\n" + digest).encode("utf-8")).hexdigest()


class OllamaClient:
    """Thin deterministic wrapper around a local Ollama server."""

    def __init__(
        self,
        model: str = MODEL_NAME,
        temperature: float = TEMPERATURE,
        top_k: int = TOP_K,
        seed: int = SEED,
        cache_dir: str | Path | None = None,
        timeout: int = REQUEST_TIMEOUT_S,
        ledger_path: str | Path | None = None,
    ) -> None:
        self._model = model
        self._params: dict[str, Any] = {
            "temperature": temperature,
            "top_k": top_k,
            "seed": seed,
        }
        self._cache_dir = Path(cache_dir) if cache_dir is not None else default_cache_dir()
        self._timeout = timeout
        self._ledger_path = ledger_path

    @property
    def model(self) -> str:
        """Configured model name."""
        return self._model

    def effective_params(self, json_schema: dict[str, Any] | None = None) -> dict[str, Any]:
        """Decoding params plus the requested output format (part of the cache key)."""
        params = dict(self._params)
        if json_schema is not None:
            params["format"] = json_schema
        return params

    def cache_key(self, prompt: str, json_schema: dict[str, Any] | None = None) -> str:
        """Public cache-key helper (lets tests pre-seed the cache)."""
        return cache_key_for(prompt, self._model, self.effective_params(json_schema))

    def _read_cache(self, key: str) -> str | None:
        try:
            raw = json.loads((self._cache_dir / f"{key}.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        response = raw.get("response") if isinstance(raw, dict) else None
        return response if isinstance(response, str) and response else None

    def _write_cache(self, key: str, prompt: str, response: str) -> None:
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self._model,
            "params": self._params,
            "prompt": prompt,
            "response": response,
        }
        tmp = self._cache_dir / f"{key}.tmp"
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(self._cache_dir / f"{key}.json")

    def _chat_once(self, prompt: str, json_schema: dict[str, Any] | None) -> str:
        try:
            import ollama
        except ImportError as exc:
            raise LLMUnavailable(
                "ollama python client is not installed; local model unavailable"
            ) from exc
        options = {
            "temperature": self._params["temperature"],
            "top_k": self._params["top_k"],
            "seed": self._params["seed"],
        }
        request: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "options": options,
        }
        if json_schema is not None:
            request["format"] = json_schema
        try:
            chat = cast(Any, ollama).chat
            response = chat(timeout=self._timeout, **request)
        except TypeError:
            # Client versions without a timeout kwarg (e.g. ollama 0.6.x).
            response = cast(Any, ollama).chat(**request)
        if isinstance(response, dict):
            message = response.get("message", {})
            content = message.get("content", "") if isinstance(message, dict) else ""
        else:
            content = getattr(getattr(response, "message", None), "content", "") or ""
        if not isinstance(content, str) or not content.strip():
            raise ValueError("empty response from local model")
        return content.strip()

    def generate(self, prompt: str, json_schema: dict[str, Any] | None = None) -> str:
        """Generate text; cached on sha256(prompt + model digest + params), retry once.

        Raises :class:`LLMUnavailable` when the model cannot produce output;
        the caller falls back silently to the deterministic template.
        """
        key = self.cache_key(prompt, json_schema)
        cached = self._read_cache(key)
        if cached is not None:
            log_cache_hit(self._ledger_path)
            return cached
        last_error: Exception | None = None
        for _ in range(MAX_ATTEMPTS):
            try:
                text = self._chat_once(prompt, json_schema)
            except LLMUnavailable:
                raise
            except Exception as exc:  # transport/decoding errors: retry once, then give up
                last_error = exc
                continue
            self._write_cache(key, prompt, text)
            log_call(count_tokens(prompt), count_tokens(text), self._ledger_path)
            return text
        raise LLMUnavailable(f"local model {self._model!r} failed after retry: {last_error}")
