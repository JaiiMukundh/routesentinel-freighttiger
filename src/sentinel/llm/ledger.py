"""Token/call ledger for local-LLM usage.

Every LLM call (or cache hit, or template fallback) is recorded in
``outputs/token_cost_log.json`` with the keys
``calls``, ``input_tokens``, ``output_tokens``, ``cache_hits``,
``fallbacks``, ``fallback_rate`` and ``local_cost``.

The model always runs locally via Ollama, so ``local_cost`` is always 0.
Token counts use a whitespace split: an exact tokenizer is unnecessary for
a local model and this keeps counting deterministic without new deps.
No timestamps are stored (repo rule: no timestamps in outputs).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

LOCAL_COST = 0


def default_log_path() -> Path:
    """Repo-level default ledger file."""
    return Path(__file__).resolve().parents[3] / "outputs" / "token_cost_log.json"


def blank_summary() -> dict[str, Any]:
    """Fresh ledger contents."""
    return {
        "calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_hits": 0,
        "fallbacks": 0,
        "fallback_rate": 0.0,
        "local_cost": LOCAL_COST,
    }


def count_tokens(text: str) -> int:
    """Whitespace-split token estimate (local model; exact tokenizer not needed)."""
    return len(text.split())


def _resolve(path: str | Path | None) -> Path:
    return Path(path) if path is not None else default_log_path()


def read_summary(path: str | Path | None = None) -> dict[str, Any]:
    """Read the ledger; missing or corrupt files yield a blank summary."""
    resolved = _resolve(path)
    summary = blank_summary()
    try:
        raw = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return summary
    if not isinstance(raw, dict):
        return summary
    for key, blank in summary.items():
        value = raw.get(key, blank)
        if isinstance(blank, int) and isinstance(value, int):
            summary[key] = value
        elif isinstance(blank, float) and isinstance(value, (int, float)):
            summary[key] = float(value)
    summary["local_cost"] = LOCAL_COST
    return summary


def _write(summary: dict[str, Any], path: Path) -> dict[str, Any]:
    total = summary["calls"] + summary["cache_hits"] + summary["fallbacks"]
    summary["fallback_rate"] = round(summary["fallbacks"] / max(total, 1), 4)
    summary["local_cost"] = LOCAL_COST
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)
    return summary


def log_call(
    prompt_tokens: int,
    completion_tokens: int,
    path: str | Path | None = None,
) -> dict[str, Any]:
    """Record one completed (non-cached) model call."""
    resolved = _resolve(path)
    summary = read_summary(resolved)
    summary["calls"] += 1
    summary["input_tokens"] += prompt_tokens
    summary["output_tokens"] += completion_tokens
    return _write(summary, resolved)


def log_cache_hit(path: str | Path | None = None) -> dict[str, Any]:
    """Record one served-from-cache generation (no new tokens)."""
    resolved = _resolve(path)
    summary = read_summary(resolved)
    summary["cache_hits"] += 1
    return _write(summary, resolved)


def log_fallback(path: str | Path | None = None) -> dict[str, Any]:
    """Record one fallback to the deterministic template."""
    resolved = _resolve(path)
    summary = read_summary(resolved)
    summary["fallbacks"] += 1
    return _write(summary, resolved)


def reset(path: str | Path | None = None) -> dict[str, Any]:
    """Reset the ledger to blank (used by tests)."""
    resolved = _resolve(path)
    summary = blank_summary()
    return _write(summary, resolved)
