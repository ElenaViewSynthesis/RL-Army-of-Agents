"""Local text embeddings for semantic recall (optional, best-effort).

Wraps a **local** sentence-transformer so the A2A system can embed notes and
queries with **zero budget** — no OpenAI/Gemini API. Default model:
``Alibaba-NLP/gte-Qwen2-1.5B-instruct``, which outputs **1536-dim** vectors
natively, matching the existing ``agent_responses.embedding vector(1536)`` column
(no migration).

**Optional + graceful.** ``sentence-transformers`` is an *optional* dependency
(the ``recall`` extra). If it isn't installed, or the model fails to load, every
function returns ``None`` — callers (``storage.save_response`` /
``storage.search_similar``) then simply skip embeddings. Persistence never breaks
because recall is unavailable.

**Instruct query asymmetry.** GTE *instruct* models expect a task instruction on
the **query** but embed **documents plainly**. ``embed(text, is_query=True)``
prepends the instruction; ``is_query=False`` (stored notes) does not. Using the
same model for both sides is required for meaningful distances.

Env knobs:
- ``RECALL_MODEL``            — model id (default gte-Qwen2-1.5B-instruct)
- ``RECALL_MODEL_REVISION``   — pin a specific commit (recommended for prod, since
  the model loads with ``trust_remote_code=True``)
- ``RECALL_QUERY_INSTRUCTION``— override the retrieval instruction text
"""

from __future__ import annotations

import logging
import os
import threading
from typing import List, Optional

logger = logging.getLogger("a2a_finance.embeddings")

EMBEDDING_DIM = 1536

_MODEL_NAME = os.getenv("RECALL_MODEL", "Alibaba-NLP/gte-Qwen2-1.5B-instruct")
_MODEL_REVISION = os.getenv("RECALL_MODEL_REVISION")  # pin a commit before prod
_QUERY_INSTRUCTION = os.getenv(
    "RECALL_QUERY_INSTRUCTION",
    "Given a search query, retrieve relevant financial research notes",
)

_model = None
_model_failed = False
_model_lock = threading.Lock()


def _detailed_query(text: str) -> str:
    """GTE instruct query format: an instruction prefix on the query only."""
    return f"Instruct: {_QUERY_INSTRUCTION}\nQuery: {text}"


def _get_model():
    """Lazily load the sentence-transformer once. None if unavailable.

    Any failure (extra not installed, model load error) disables embeddings for
    the process via ``_model_failed`` so we don't retry a broken setup per call.
    """
    global _model, _model_failed
    if _model is not None:
        return _model
    if _model_failed:
        return None
    with _model_lock:
        if _model is not None:
            return _model
        if _model_failed:
            return None
        try:
            from sentence_transformers import SentenceTransformer
        except Exception:
            logger.info("sentence-transformers not installed; recall disabled "
                        "(install the 'recall' extra to enable).")
            _model_failed = True
            return None
        try:
            kwargs = {"trust_remote_code": True}
            if _MODEL_REVISION:
                kwargs["revision"] = _MODEL_REVISION
            else:
                logger.warning(
                    "Loading %s with trust_remote_code=True and no pinned "
                    "revision; set RECALL_MODEL_REVISION to a fixed commit for "
                    "production (executes remote repo code).", _MODEL_NAME,
                )
            logger.info("Loading embedding model %s …", _MODEL_NAME)
            _model = SentenceTransformer(_MODEL_NAME, **kwargs)
            return _model
        except Exception as e:
            logger.warning("Embedding model load failed (recall disabled): %s", e)
            _model_failed = True
            return None


def available() -> bool:
    """True if the embedding model can be loaded (extra installed + loads)."""
    return _get_model() is not None


def embed(text: str, *, is_query: bool = False) -> Optional[List[float]]:
    """Embed ``text`` to a 1536-dim unit vector, or None on any failure.

    Args:
        text: The note (document) or search query to embed.
        is_query: True for a search query (adds the GTE instruction prefix);
            False for a stored note (embedded plainly). Must match between the
            write side and the query side.

    Returns:
        A list of 1536 floats (L2-normalized), or None if embeddings are
        unavailable, ``text`` is empty, or the output width is unexpected.
    """
    if not text:
        return None
    model = _get_model()
    if model is None:
        return None
    try:
        payload = _detailed_query(text) if is_query else text
        vec = model.encode(payload, normalize_embeddings=True)
        out = [float(x) for x in vec]
        if len(out) != EMBEDDING_DIM:
            logger.warning("Embedding width %d != %d; dropping.", len(out), EMBEDDING_DIM)
            return None
        return out
    except Exception as e:
        logger.warning("Embedding failed (skipped): %s", e)
        return None


def to_pgvector(vec: List[float]) -> str:
    """Format a vector as a pgvector string literal ``"[a,b,c]"`` for PostgREST."""
    return "[" + ",".join(repr(float(x)) for x in vec) + "]"
