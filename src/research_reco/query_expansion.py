# src/research_reco/query_expansion.py
from __future__ import annotations
from typing import Dict, List, Any, Set, Tuple
import math


def expand_query_from_top_docs(
    docs_by_id: Dict[str, Dict[str, Any]],
    initial_hits: List[Dict[str, Any]],
    query_tokens: List[str],
    max_expand: int = 8,
    top_docs: int = 8,
) -> List[str]:
    """
    Expansion term mining:
    - take tokens from top retrieved docs
    - count terms (simple TF across top docs)
    - prefer terms that appear in multiple docs (more robust)
    - exclude original query tokens
    """
    qset: Set[str] = set(query_tokens)
    tf: Dict[str, int] = {}
    df: Dict[str, int] = {}

    used = 0
    for h in initial_hits[:top_docs]:
        doc_id = h.get("doc_id")
        if not doc_id:
            continue
        doc = docs_by_id.get(str(doc_id))
        if not doc:
            continue

        used += 1
        toks = doc.get("tokens", []) or []
        seen_in_doc = set()

        for t in toks:
            if not t or len(t) <= 2:
                continue
            if t in qset:
                continue
            tf[t] = tf.get(t, 0) + 1
            seen_in_doc.add(t)

        for t in seen_in_doc:
            df[t] = df.get(t, 0) + 1

    if used == 0:
        return query_tokens

    # score term by (tf * log(1+df)) to prefer terms repeated across docs
    scored: List[Tuple[str, float]] = []
    for t, c in tf.items():
        dfi = df.get(t, 1)
        score = float(c) * math.log(1.0 + dfi)
        scored.append((t, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    expansion = [t for t, _ in scored[:max_expand]]

    return query_tokens + expansion
