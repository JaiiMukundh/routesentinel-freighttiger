"""Scripted Q&A tests (PLAN.md Phase 8 acceptance).

Covers the three required conversations:
1. "why did Chennai-Bangalore get pricier in March?" asks which year
   or answers 2025 citing N001/N009 plus the unexplained weeks.
2. "why did Kolkata-Bhubaneswar get pricier?" reports no flagged weeks.
3. "did N004 raise Mumbai-Pune costs?" cites N004's out-of-dataset text.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sentinel.qa.answer import answer_question, build_outputs  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def _ensure_outputs() -> None:
    csv_path = ROOT / "outputs" / "output.csv"
    db_path = ROOT / "outputs" / "findings.sqlite"
    if not csv_path.exists() or not db_path.exists():
        build_outputs(root=ROOT)


def test_chennai_bangalore_march() -> None:
    answer = answer_question("why did Chennai-Bangalore get pricier in March?", root=ROOT)
    assert ("2025" in answer and "N001" in answer and "N009" in answer) or (
        "which year" in answer.lower()
    )


def test_kolkata_bhubaneswar_no_flagged_weeks() -> None:
    answer = answer_question("why did Kolkata-Bhubaneswar get pricier?", root=ROOT)
    assert "No flagged weeks" in answer
    assert "Kolkata-Bhubaneswar" in answer


def test_n004_mumbai_pune() -> None:
    answer = answer_question("did N004 raise Mumbai-Pune costs?", root=ROOT)
    assert "N004" in answer
    assert "not part of this dataset" in answer
