"""Hybrid BM25 + dense index over freight context notes.

Retrieval finds candidates; the evidence gate decides. Similarity scores
alone NEVER justify a candidate (AGENTS.md). This module only builds an
in-memory index and exposes exact (flat) search over all notes.

Backends (all deterministic, exact flat search, no paid APIs):
- BM25: ``rank_bm25.BM25Okapi`` when installed, else a built-in
  BM25-Okapi implementation with identical tokenization. Stable
  tie-break ``(-score, note_id)`` everywhere.
- Dense: fastembed ``bge-small`` embeddings when importable *and*
  loadable offline, else a TF-IDF cosine fallback (scikit-learn), else
  BM25-only. Missing optional deps only emit a warning, never crash.
"""

from __future__ import annotations

import importlib
import math
import re
import warnings
from dataclasses import dataclass, field
from typing import Any

TOKEN_RE = re.compile(r"[a-z0-9]+")

FASTEMBED_MODEL = "BAAI/bge-small-en"


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokenization shared by BM25 and fallbacks."""
    return TOKEN_RE.findall(text.lower())


class _BuiltinBM25:
    """Minimal BM25-Okapi (k1=1.5, b=0.75) over pre-tokenized docs.

    Used only when ``rank_bm25`` is not installed. Scores match the
    standard Okapi formula; ranking tie-breaks are applied by callers.
    """

    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self._k1 = k1
        self._b = b
        self._docs = corpus
        self._n = len(corpus)
        self._avgdl = sum(len(d) for d in corpus) / max(self._n, 1)
        df: dict[str, int] = {}
        for doc in corpus:
            for term in set(doc):
                df[term] = df.get(term, 0) + 1
        self._idf: dict[str, float] = {}
        for term, freq in df.items():
            # Standard Lucene-style Okapi idf, floored at 0.
            self._idf[term] = max(math.log((self._n - freq + 0.5) / (freq + 0.5) + 1.0), 0.0)

    def get_scores(self, query: list[str]) -> list[float]:
        scores = [0.0] * self._n
        for qi, doc in enumerate(self._docs):
            dl = len(doc)
            if dl == 0:
                continue
            norm = self._k1 * (1.0 - self._b + self._b * dl / self._avgdl)
            if self._avgdl == 0:
                norm = self._k1
            tf: dict[str, int] = {}
            for term in doc:
                tf[term] = tf.get(term, 0) + 1
            total = 0.0
            for term in query:
                f = tf.get(term, 0)
                if f == 0:
                    continue
                idf = self._idf.get(term, 0.0)
                total += idf * (f * (self._k1 + 1.0)) / (f + norm)
            scores[qi] = total
        return scores


def _load_bm25(corpus: list[list[str]]) -> tuple[Any, str]:
    """Return (bm25_object, backend_name), never raising on missing deps."""
    try:
        mod = importlib.import_module("rank_bm25")
        return mod.BM25Okapi(corpus), "rank_bm25"
    except ImportError:
        warnings.warn("rank_bm25 not installed; using built-in BM25 fallback", stacklevel=3)
        return _BuiltinBM25(corpus), "builtin"
    except Exception as exc:  # pragma: no cover - defensive
        warnings.warn(f"rank_bm25 failed ({exc}); using built-in BM25 fallback", stacklevel=3)
        return _BuiltinBM25(corpus), "builtin"


@dataclass
class NoteIndex:
    """In-memory hybrid index over notes. Built by :func:`build_index`."""

    note_ids: list[str] = field(default_factory=list)
    texts: list[str] = field(default_factory=list)
    tokenized: list[list[str]] = field(default_factory=list)
    bm25: Any = None
    bm25_backend: str = "builtin"
    dense_backend: str = "none"  # "fastembed" | "tfidf" | "none"
    dense_vectors: Any = None  # L2-normalized rows, aligned with note_ids
    dense_model: Any = None  # fastembed TextEmbedding instance, if used
    tfidf: Any = None  # fitted TfidfVectorizer, if dense_backend == "tfidf"

    def __len__(self) -> int:
        return len(self.note_ids)


def _l2_normalize_rows(matrix: Any) -> Any:
    import numpy as np

    arr = np.asarray(matrix, dtype=float)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return arr / norms


def _try_fastembed(texts: list[str]) -> tuple[Any, Any]:
    """Return (vectors, model) or (None, None) without ever raising."""
    try:
        mod = importlib.import_module("fastembed")
        model = mod.TextEmbedding(model_name=FASTEMBED_MODEL)
        import numpy as np

        vecs = np.asarray(list(model.embed(texts)), dtype=float)
        return _l2_normalize_rows(vecs), model
    except Exception as exc:
        warnings.warn(f"fastembed unavailable ({exc}); falling back to TF-IDF", stacklevel=4)
        return None, None


def _try_tfidf(texts: list[str]) -> tuple[Any, Any]:
    """Return (vectors, vectorizer) or (None, None) without ever raising."""
    try:
        mod = importlib.import_module("sklearn.feature_extraction.text")
        vec = mod.TfidfVectorizer(token_pattern=r"[a-z0-9]+", lowercase=True)
        mat = vec.fit_transform(texts).toarray()
        return _l2_normalize_rows(mat), vec
    except Exception as exc:  # pragma: no cover - defensive
        warnings.warn(f"TF-IDF unavailable ({exc}); dense search disabled", stacklevel=4)
        return None, None


def build_index(notes: list[dict[str, str]]) -> NoteIndex:
    """Build an in-memory hybrid index.

    Args:
        notes: list of ``{"id": ..., "text": ...}`` (``"note_id"`` also
            accepted as the id key). Sorted by id for determinism.
    """
    cleaned: list[tuple[str, str]] = []
    for note in notes:
        nid = str(note.get("id", note.get("note_id", "")))
        cleaned.append((nid, str(note.get("text", ""))))
    cleaned.sort(key=lambda pair: pair[0])
    note_ids = [nid for nid, _ in cleaned]
    texts = [text for _, text in cleaned]
    tokenized = [tokenize(t) for t in texts]

    bm25, bm25_backend = _load_bm25(tokenized)

    dense_vectors: Any = None
    dense_model: Any = None
    tfidf: Any = None
    dense_backend = "none"
    if texts:
        vecs, model = _try_fastembed(texts)
        if vecs is not None:
            dense_vectors, dense_model, dense_backend = vecs, model, "fastembed"
        else:
            tvecs, tvec = _try_tfidf(texts)
            if tvecs is not None:
                dense_vectors, tfidf, dense_backend = tvecs, tvec, "tfidf"
            else:
                warnings.warn("dense search disabled; hybrid = BM25-only", stacklevel=2)
    return NoteIndex(
        note_ids=note_ids,
        texts=texts,
        tokenized=tokenized,
        bm25=bm25,
        bm25_backend=bm25_backend,
        dense_backend=dense_backend,
        dense_vectors=dense_vectors,
        dense_model=dense_model,
        tfidf=tfidf,
    )


def _ranked(scores: list[float], note_ids: list[str], k: int) -> list[tuple[str, float]]:
    order = sorted(range(len(note_ids)), key=lambda i: (-scores[i], note_ids[i]))
    if k >= 0:
        order = order[:k]
    return [(note_ids[i], float(scores[i])) for i in order]


def full_bm25_scores(index: NoteIndex, query: str) -> list[float]:
    """Exact BM25 scores for every note, aligned with ``index.note_ids``."""
    if len(index) == 0:
        return []
    raw = index.bm25.get_scores(tokenize(query))
    return [float(s) for s in list(raw)]


def search_bm25(index: NoteIndex, query: str, k: int = -1) -> list[tuple[str, float]]:
    """BM25 ranking of ``(note_id, score)`` with ``(-score, note_id)`` order."""
    return _ranked(full_bm25_scores(index, query), index.note_ids, k)


def _embed_query(index: NoteIndex, query: str) -> Any | None:
    import numpy as np

    if index.dense_backend == "fastembed" and index.dense_model is not None:
        try:
            vec = np.asarray(list(index.dense_model.embed([query])), dtype=float).reshape(-1)
        except Exception as exc:
            warnings.warn(f"fastembed query failed ({exc}); scoring 0", stacklevel=3)
            return None
    elif index.dense_backend == "tfidf" and index.tfidf is not None:
        vec = np.asarray(index.tfidf.transform([query]).toarray(), dtype=float).reshape(-1)
    else:
        return None
    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        return None
    return vec / norm


def full_dense_scores(index: NoteIndex, query: str) -> list[float]:
    """Exact dense cosine scores for every note, aligned with ``note_ids``.

    Returns all zeros when dense search is disabled (BM25-only mode).
    """
    if len(index) == 0 or index.dense_vectors is None:
        return [0.0] * len(index)
    qvec = _embed_query(index, query)
    if qvec is None:
        return [0.0] * len(index)
    import numpy as np

    return [float(s) for s in (np.asarray(index.dense_vectors) @ qvec).tolist()]


def search_dense(index: NoteIndex, query: str, k: int = -1) -> list[tuple[str, float]]:
    """Dense ranking of ``(note_id, score)`` with ``(-score, note_id)`` order."""
    return _ranked(full_dense_scores(index, query), index.note_ids, k)
