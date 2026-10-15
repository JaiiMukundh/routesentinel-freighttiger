"""Local-LLM package: Ollama client and token/call ledger."""

from sentinel.llm.client import LLMUnavailable, OllamaClient
from sentinel.llm.ledger import (
    count_tokens,
    log_cache_hit,
    log_call,
    log_fallback,
    read_summary,
    reset,
)

__all__ = [
    "LLMUnavailable",
    "OllamaClient",
    "count_tokens",
    "log_cache_hit",
    "log_call",
    "log_fallback",
    "read_summary",
    "reset",
]
