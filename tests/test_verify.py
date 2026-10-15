"""Phase 5 tests: claim verifier, template fallback, and LLM/--no-llm parity.

The stub model emits a fake note id (N011) and a made-up cause; both must
fail verification so the pipeline falls back to the deterministic template.
Verdicts, numbers, and note ids must be identical between the LLM and
--no-llm paths.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# Phase 0 packaging will put src on sys.path; until then, bootstrap it here.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sentinel.agent.explain import (  # noqa: E402
    FACT_SHEET_KEYS,
    build_fact_sheet,
    build_wording_prompt,
    explain_candidate,
    template_reason,
    word_reason,
)
from sentinel.agent.verify import VerifyResult, verify_reason  # noqa: E402
from sentinel.llm.client import LLMUnavailable, OllamaClient, cache_key_for  # noqa: E402
from sentinel.llm.ledger import (  # noqa: E402
    count_tokens,
    log_cache_hit,
    log_call,
    log_fallback,
    read_summary,
    reset,
)


class StubClient:
    """Stand-in model returning canned texts (duck-types OllamaClient.generate)."""

    def __init__(self, texts: list[str]) -> None:
        self._texts = texts
        self.calls = 0
        self.prompts: list[str] = []

    def generate(self, prompt: str, json_schema: dict[str, Any] | None = None) -> str:
        self.calls += 1
        self.prompts.append(prompt)
        index = min(self.calls - 1, len(self._texts) - 1)
        return self._texts[index]


class UnavailableClient:
    """Stand-in for an unreachable Ollama server."""

    def generate(self, prompt: str, json_schema: dict[str, Any] | None = None) -> str:
        raise LLMUnavailable("simulated outage")


def _justified() -> tuple[dict[str, Any], dict[str, Any]]:
    candidate = {
        "route": "Ahmedabad-Mumbai",
        "week_of": "2025-01-20",
        "cost_per_tonne_km": 3.290367,
        "own_pct": 0.294731,
        "peer_pct": 0.224937,
        "flagged": "No (justified)",
        "matched_note_id": "N002",
    }
    note = {
        "note_id": "N002",
        "scope": ["Ahmedabad-Mumbai"],
        "effect": "increase",
        "start": "2025-01-20",
        "end": "2025-01-26",
        "open_ended": False,
        "mag": [None, None],
        "cause": "festival",
    }
    return candidate, note


def _unexplained() -> tuple[dict[str, Any], dict[str, Any]]:
    candidate = {
        "route": "Delhi-Jaipur",
        "week_of": "2024-11-11",
        "cost_per_tonne_km": 4.172957,
        "own_pct": 0.355201,
        "peer_pct": 0.210344,
        "flagged": "Yes",
        "matched_note_id": "",
    }
    rejected = {
        "note_id": "N007",
        "scope": ["Delhi-Jaipur"],
        "effect": "decrease",
        "start": "2024-05-20",
        "end": None,
        "open_ended": False,
        "mag": [None, None],
        "cause": "road",
    }
    return candidate, rejected


# --- verifier accepts the deterministic templates ---


def test_template_justified_passes_verifier() -> None:
    candidate, note = _justified()
    facts = build_fact_sheet(candidate, note)
    result = verify_reason(template_reason(facts), facts)
    assert result == VerifyResult(ok=True, violations=())


def test_template_unexplained_without_note_passes_verifier() -> None:
    candidate, _ = _unexplained()
    facts = build_fact_sheet(candidate)
    result = verify_reason(template_reason(facts), facts)
    assert result.ok


def test_template_with_closest_rejected_note_passes_verifier() -> None:
    candidate, rejected = _unexplained()
    facts = build_fact_sheet(candidate, None, rejected)
    result = verify_reason(template_reason(facts), facts)
    assert result.ok
    assert "N007" in template_reason(facts)


# --- verifier rejects fabricated claims ---


def test_fake_note_id_and_made_up_cause_fail() -> None:
    candidate, note = _justified()
    facts = build_fact_sheet(candidate, note)
    bad = (
        "Matches note N011 dated 2025-01-20: a driver strike caused the spike. "
        "Costs rose +99.9% on Delhi-Jaipur."
    )
    result = verify_reason(bad, facts)
    assert not result.ok
    assert any("N011" in v for v in result.violations)
    assert any("Delhi-Jaipur" in v for v in result.violations)
    assert any("99.9%" in v for v in result.violations)


def test_wrong_percentage_and_date_fail() -> None:
    candidate, note = _justified()
    facts = build_fact_sheet(candidate, note)
    result = verify_reason("Costs rose +12.3% on 2025-02-01 for N002.", facts)
    assert not result.ok
    assert any("12.3%" in v for v in result.violations)
    assert any("2025-02-01" in v for v in result.violations)


def test_true_claims_pass() -> None:
    candidate, note = _justified()
    facts = build_fact_sheet(candidate, note)
    text = (
        "Matches note N002 dated 2025-01-20: costs on Ahmedabad-Mumbai "
        "rose +29.5% vs history and +22.5% vs peers to 3.29."
    )
    assert verify_reason(text, facts).ok


def test_percentage_range_dash_is_not_negative() -> None:
    candidate, note = _justified()
    note = dict(note, mag=[5, 7])
    facts = build_fact_sheet(candidate, note)
    assert verify_reason("Fuel effect of 5-7% is smaller than observed.", facts).ok


# --- fallback behaviour ---


def test_stub_emitting_fake_n011_falls_back_to_template(tmp_path: Path) -> None:
    candidate, note = _justified()
    facts = build_fact_sheet(candidate, note)
    ledger = tmp_path / "token_cost_log.json"
    stub = StubClient(
        [
            "Matches note N011 dated 2025-01-20: a driver strike caused it.",
            "Still N011: port congestion caused it.",
        ]
    )
    result = word_reason(facts, stub, ledger_path=ledger)
    assert stub.calls == 2  # one retry, then give up
    assert result.source == "template"
    assert result.reason == template_reason(facts)
    assert any("N011" in v for v in result.violations)
    summary = read_summary(ledger)
    assert summary["fallbacks"] == 1
    assert summary["fallback_rate"] == 1.0


def test_unavailable_client_falls_back_silently(tmp_path: Path) -> None:
    candidate, note = _justified()
    facts = build_fact_sheet(candidate, note)
    ledger = tmp_path / "token_cost_log.json"
    result = word_reason(facts, UnavailableClient(), ledger_path=ledger)
    assert result.source == "template"
    assert result.reason == template_reason(facts)
    assert read_summary(ledger)["fallbacks"] == 1


def test_no_llm_path_uses_template_without_touching_client() -> None:
    candidate, note = _justified()
    facts = build_fact_sheet(candidate, note)
    stub = StubClient(["anything"])
    result = word_reason(facts, stub, use_llm=False, ledger_path=":memory:")
    assert result.source == "template"
    assert result.reason == template_reason(facts)
    assert stub.calls == 0


def test_verdict_numbers_ids_identical_between_llm_and_nollm(tmp_path: Path) -> None:
    candidate, note = _justified()
    ledger = tmp_path / "token_cost_log.json"
    stub = StubClient(["Fake N011 strike story.", "Fake N011 again."])
    with_llm = explain_candidate(candidate, note, client=stub, ledger_path=ledger)
    without_llm = explain_candidate(candidate, note, use_llm=False, ledger_path=ledger)
    for key in ("route", "week_of", "cost_per_tonne_km", "flagged", "matched_note_id"):
        assert with_llm[key] == without_llm[key] == candidate[key]
    assert with_llm["reason"] == without_llm["reason"]  # both fell back to template


def test_good_llm_wording_may_differ_but_verdict_columns_match(tmp_path: Path) -> None:
    candidate, note = _justified()
    ledger = tmp_path / "token_cost_log.json"
    good = (
        "Matches note N002 dated 2025-01-20: the Ahmedabad-Mumbai corridor "
        "saw +29.5% vs its past average and +22.5% vs peers."
    )
    stub = StubClient([good])
    with_llm = explain_candidate(candidate, note, client=stub, ledger_path=ledger)
    assert with_llm["reason_source"] == "llm"
    for key in ("route", "week_of", "cost_per_tonne_km", "flagged", "matched_note_id"):
        assert with_llm[key] == candidate[key]


# --- fact-sheet discipline and prompt hygiene ---


def test_fact_sheet_contains_only_allowed_keys() -> None:
    candidate, note = _justified()
    _, rejected = _unexplained()
    facts = build_fact_sheet(candidate, note, rejected)
    assert set(facts) <= set(FACT_SHEET_KEYS)


def test_prompt_is_fact_sheet_only_and_marks_note_untrusted() -> None:
    candidate, note = _justified()
    facts = build_fact_sheet(candidate, note)
    prompt = build_wording_prompt(facts).lower()
    assert "only" in prompt and "fact sheet" in prompt
    assert "untrusted" in prompt


def test_percentage_displays_match_contract_format() -> None:
    candidate, _ = _unexplained()
    facts = build_fact_sheet(candidate)
    assert facts["own_pct_display"] == "+35.5%"
    assert facts["peer_pct_display"] == "+21.0%"


# --- client cache and ledger ---


def test_cache_hit_served_without_server(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    ledger = tmp_path / "token_cost_log.json"
    client = OllamaClient(cache_dir=cache_dir, ledger_path=ledger)
    key = client.cache_key("hello")
    assert key == cache_key_for("hello", client.model, client.effective_params())
    cache_dir.mkdir(parents=True)
    (cache_dir / f"{key}.json").write_text(
        json.dumps({"prompt": "hello", "response": "cached-text"}), encoding="utf-8"
    )
    assert client.generate("hello") == "cached-text"  # no ollama package needed
    summary = read_summary(ledger)
    assert summary["cache_hits"] == 1
    assert summary["calls"] == 0


def test_generate_without_ollama_raises_unavailable(tmp_path, monkeypatch) -> None:
    # Point at a guaranteed-dead port so the test is deterministic whether or
    # not a local model (shim or Ollama) is running on :11434.
    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:9")
    client = OllamaClient(cache_dir=tmp_path / "cache", ledger_path=tmp_path / "ledger.json")
    try:
        client.generate("hello")
    except LLMUnavailable:
        return
    raise AssertionError("expected LLMUnavailable when Ollama is unreachable")


def test_ledger_counts_and_zero_local_cost(tmp_path: Path) -> None:
    ledger = tmp_path / "token_cost_log.json"
    reset(ledger)
    log_call(count_tokens("a b c"), count_tokens("d e"), path=ledger)
    log_cache_hit(ledger)
    log_fallback(ledger)
    summary = read_summary(ledger)
    assert summary["calls"] == 1
    assert summary["input_tokens"] == 3
    assert summary["output_tokens"] == 2
    assert summary["cache_hits"] == 1
    assert summary["fallbacks"] == 1
    assert summary["fallback_rate"] == round(1 / 3, 4)
    assert summary["local_cost"] == 0
