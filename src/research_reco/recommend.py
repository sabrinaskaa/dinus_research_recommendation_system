# src/research_reco/recommend.py
from __future__ import annotations

from typing import Any, Dict, List, Optional
import math

from .bm25 import bm25_search, BM25Index


def _percentile(xs: List[float], p: float) -> float:
    """Simple percentile (0..1) without numpy."""
    if not xs:
        return 0.0
    xs2 = sorted(xs)
    k = (len(xs2) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return xs2[int(k)]
    d0 = xs2[f] * (c - k)
    d1 = xs2[c] * (k - f)
    return d0 + d1


def _auto_cutoff(scores: List[float]) -> float:
    """
    Dynamic cutoff so TopK doesn't force low-relevance junk.
    Strategy:
      - max_ratio = 0.65 of best score
      - p75 = 75th percentile
      - cutoff = max(max_ratio, p75)
    """
    if not scores:
        return float("inf")
    mx = max(scores)
    max_ratio = mx * 0.65
    p75 = _percentile(scores, 0.75)
    return max(max_ratio, p75)


def recommend_citations(
    bm25: BM25Index,
    query_tokens: List[str],
    top_k: int = 10,
    diversify: bool = True,
    original_query_tokens: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Return citation recommendations with automatic relevance cutoff.
    - top_k is the MAX cap, not forced output size.
    """
    # pull more candidates than needed
    candidate_k = max(30, top_k * 5)

    hits = bm25_search(bm25, query_tokens, top_k=candidate_k, include_meta=True)

    if not hits:
        return []

    # scores for cutoff
    scores = [float(h.get("score", 0.0)) for h in hits]
    cutoff = _auto_cutoff(scores)

    # apply cutoff
    filtered = [h for h in hits if float(h.get("score", 0.0)) >= cutoff]

    # fallback: at least 3 results if possible
    if len(filtered) < min(3, len(hits)):
        filtered = hits[: min(3, len(hits))]

    # now cap to top_k max
    filtered = filtered[:top_k]

    # optional diversify (very light)
    if diversify:
        # simple: prefer unique URLs/titles first
        seen = set()
        diversified = []
        for h in filtered:
            key = (h.get("url") or h.get("judul") or h.get("doc_id") or "")
            if key in seen:
                continue
            seen.add(key)
            diversified.append(h)
        # if too few, fill from remainder
        if len(diversified) < len(filtered):
            for h in filtered:
                if len(diversified) >= len(filtered):
                    break
                if h not in diversified:
                    diversified.append(h)
        filtered = diversified

    # attach score2 if you have rerank elsewhere (keep compatible)
    for h in filtered:
        if "score2" not in h:
            h["score2"] = h.get("score", 0.0)

    return filtered
