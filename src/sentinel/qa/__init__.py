"""RouteSentinel Q&A package (Phase 8).

Rule-based question answering over ``outputs/findings.sqlite``.
The LLM never decides verdicts here: answers are composed only from
deterministic pipeline outputs, with every claim citing a row or note.
"""

from sentinel.qa.answer import answer_question, build_outputs

__all__ = ["answer_question", "build_outputs"]
