# src/research_reco/explain_bm25.py
from __future__ import annotations
from typing import Dict, List, Any, Tuple

def _get_attr(obj, names: List[str], default=None):
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
    return default

def bm25_term_contributions(index, query_tokens: List[str], doc_idx: int) -> List[Dict[str, Any]]:
    """
    Duck-typing: index minimal punya idf, doc_term_freqs, doc_lens, avgdl, k1, b.
    """
    idf: Dict[str, float] = _get_attr(index, ["idf"], {}) or {}
    doc_term_freqs = _get_attr(index, ["doc_term_freqs", "doc_tf", "term_freqs"], None)
    doc_lens = _get_attr(index, ["doc_lens", "doc_len", "doc_lengths"], None)
    avgdl = float(_get_attr(index, ["avgdl"], 1.0) or 1.0)
    k1 = float(_get_attr(index, ["k1"], 1.5) or 1.5)
    b = float(_get_attr(index, ["b"], 0.75) or 0.75)

    if doc_term_freqs is None or doc_lens is None:
        return []

    tf_map: Dict[str, int] = doc_term_freqs[doc_idx]
    dl: float = float(doc_lens[doc_idx])

    q = [t.lower() for t in query_tokens]
    uniq = []
    seen = set()
    for t in q:
        if t and t not in seen:
            uniq.append(t)
            seen.add(t)

    contribs = []
    denom_norm = k1 * (1 - b + b * (dl / avgdl))

    for term in uniq:
        tf = tf_map.get(term, 0)
        if tf <= 0:
            continue
        term_idf = float(idf.get(term, 0.0))
        score = term_idf * (tf * (k1 + 1)) / (tf + denom_norm)
        contribs.append({"term": term, "tf": tf, "idf": term_idf, "score": score})

    contribs.sort(key=lambda x: x["score"], reverse=True)
    return contribs

def explain_doc(index, query_tokens: List[str], doc_idx: int, top_terms: int = 6) -> Dict[str, Any]:
    contribs = bm25_term_contributions(index, query_tokens, doc_idx)
    top = contribs[:top_terms]
    return {
        "matched_terms": [x["term"] for x in top],
        "term_contributions": top,
    }
