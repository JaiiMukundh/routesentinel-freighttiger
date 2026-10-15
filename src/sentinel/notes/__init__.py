"""Notes subpackage: deterministic extraction, indexing, retrieval."""

from sentinel.notes.extract import (
    NoteExtraction,
    extract_cause,
    extract_effect,
    extract_magnitude,
    extract_note,
    extract_scope,
    extract_window,
    load_notes,
)

__all__ = [
    "NoteExtraction",
    "extract_cause",
    "extract_effect",
    "extract_magnitude",
    "extract_note",
    "extract_scope",
    "extract_window",
    "load_notes",
]
