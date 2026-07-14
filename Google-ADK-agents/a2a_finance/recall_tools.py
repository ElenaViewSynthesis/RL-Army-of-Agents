"""ADK tool for semantic recall of prior research notes.

Wraps ``storage.search_similar`` so an agent can retrieve past narrative notes
that are similar *in meaning* (not just same-ticker) and reconcile them into a
new note. Best-effort: returns an empty match list when persistence or the
embedding model is unavailable.
"""

from __future__ import annotations


def recall_similar_notes(query: str, subject: str = "", limit: int = 5) -> dict:
    """Return prior research notes semantically similar to the query.

    Use this before writing a note to check for and reconcile with relevant past
    analyses. Matches are ranked by similarity (1.0 = identical in meaning).

    Args:
        query: What to look for, e.g. the ticker plus the question ("NVDA
            valuation vs DCF") or a theme ("stretched balance sheets").
        subject: Optional exact filter on the note's subject (e.g. a ticker
            "NVDA" or commodity theme "OIL"). Empty = search across all subjects.
        limit: Maximum number of matches to return (default 5).

    Returns:
        ``{"matches": [{id, subject, text, rating, similarity}, ...]}`` — an
        empty list if nothing is stored, recall is unavailable, or no matches.
    """
    from a2a_finance.storage import search_similar

    return {"matches": search_similar(query, subject or None, limit)}
