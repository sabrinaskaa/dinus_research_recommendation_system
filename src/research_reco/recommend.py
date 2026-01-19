# src/research_reco/recommend.py
from __future__ import annotations

from typing import List, Dict, Any, Set, Optional
import re

from .bm25 import BM25Index, bm25_search


# -----------------------------
# Bonuses (your original logic)
# -----------------------------
def _keyword_overlap_bonus(query_tokens: Set[str], keyword: str | None) -> float:
    if not keyword:
        return 0.0
    kw_tokens = set([k.strip().lower() for k in keyword.replace(",", " ").split() if k.strip()])
    if not kw_tokens:
        return 0.0
    inter = len(query_tokens & kw_tokens)
    return 0.15 * inter


def _year_freshness_bonus(tanggal: str | None) -> float:
    """
    Small freshness bonus: papers after 2018 get +0.01 per year difference.
    """
    if not tanggal:
        return 0.0
    tgl = (tanggal or "").strip()
    if len(tgl) >= 4 and tgl[:4].isdigit():
        year = int(tgl[:4])
        return max(0.0, (year - 2018)) * 0.01
    return 0.0


# -----------------------------
# Snippet (explainability)
# -----------------------------
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"[a-zA-Z0-9_]+", re.UNICODE)


def _best_snippet(abstrak: str | None, query_tokens: List[str], max_len: int = 260) -> Dict[str, Any]:
    """
    Pick the sentence with the highest overlap with query tokens.
    If abstrak empty, returns empty snippet.
    """
    text = (abstrak or "").strip()
    if not text:
        return {"snippet": "", "score": 0, "matched": []}

    qset = set(t.lower() for t in query_tokens if t)
    sents = _SENT_SPLIT.split(text)

    best = ""
    best_score = -1
    best_matched: List[str] = []

    for s in sents:
        toks = _WORD_RE.findall(s.lower())
        tset = set(toks)
        matched = sorted(list(qset.intersection(tset)))
        score = len(matched)

        if score > best_score:
            best_score = score
            best = s.strip()
            best_matched = matched

    if len(best) > max_len:
        best = best[:max_len].rsplit(" ", 1)[0] + "..."

    return {"snippet": best, "score": max(0, best_score), "matched": best_matched}


# -----------------------------
# BM25 term explanation (safe / optional)
# -----------------------------
def _get_attr(obj: Any, names: List[str], default=None):
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
    return default


def _bm25_term_contributions(index: BM25Index, query_tokens: List[str], doc_idx: int) -> List[Dict[str, Any]]:
    """
    Compute per-term BM25 contributions if the index exposes:
      - idf: Dict[str,float]
      - doc_term_freqs / doc_tf: List[Dict[str,int]]
      - doc_lens: List[int]
      - avgdl: float
      - k1, b
    If not available -> return [] safely.
    """
    idf: Dict[str, float] = _get_attr(index, ["idf"], {}) or {}
    doc_term_freqs = _get_attr(index, ["doc_term_freqs", "doc_tf", "term_freqs"], None)
    doc_lens = _get_attr(index, ["doc_lens", "doc_len", "doc_lengths"], None)

    avgdl = float(_get_attr(index, ["avgdl"], 1.0) or 1.0)
    k1 = float(_get_attr(index, ["k1"], 1.5) or 1.5)
    b = float(_get_attr(index, ["b"], 0.75) or 0.75)

    if doc_term_freqs is None or doc_lens is None:
        return []

    if doc_idx < 0 or doc_idx >= len(doc_term_freqs):
        return []

    tf_map: Dict[str, int] = doc_term_freqs[doc_idx]
    dl: float = float(doc_lens[doc_idx])

    # unique query terms, preserve order
    uniq: List[str] = []
    seen = set()
    for t in query_tokens:
        t = (t or "").lower()
        if not t or t in seen:
            continue
        uniq.append(t)
        seen.add(t)

    denom_norm = k1 * (1 - b + b * (dl / (avgdl + 1e-9)))

    contribs: List[Dict[str, Any]] = []
    for term in uniq:
        tf = int(tf_map.get(term, 0))
        if tf <= 0:
            continue
        term_idf = float(idf.get(term, 0.0))
        score = term_idf * (tf * (k1 + 1)) / (tf + denom_norm)
        contribs.append({"term": term, "tf": tf, "idf": term_idf, "score": score})

    contribs.sort(key=lambda x: x["score"], reverse=True)
    return contribs


def _explain_bm25(index: BM25Index, query_tokens: List[str], item: Dict[str, Any], top_terms: int = 6) -> Dict[str, Any]:
    """
    Attach BM25 explain if doc_idx exists.
    bm25_search should ideally return doc_idx.
    If not, returns {}.
    """
    doc_idx = item.get("doc_idx", None)
    if doc_idx is None:
        # some implementations might use different key
        doc_idx = item.get("_doc_idx", None)
    if doc_idx is None:
        return {}

    try:
        doc_idx_int = int(doc_idx)
    except Exception:
        return {}

    contribs = _bm25_term_contributions(index, query_tokens, doc_idx_int)
    if not contribs:
        return {}

    top = contribs[:top_terms]
    return {
        "matched_terms": [x["term"] for x in top],
        "term_contributions": top,
    }


# -----------------------------
# Diversification (your logic, slightly safer tokenizing)
# -----------------------------
def _title_tokens(title: str) -> Set[str]:
    t = (title or "").lower().strip()
    if not t:
        return set()
    # more robust than split(): keep alnum/_ tokens
    return set(_WORD_RE.findall(t))


def recommend_citations(
    index: BM25Index,
    query_tokens: List[str],
    top_k: int = 10,
    diversify: bool = True,
) -> List[Dict[str, Any]]:
    """
    Sitasi: dari SEMUA corpus (index dibuild dari semua docs).
    Output enriched with:
      - score2 (reranked score)
      - snippet (best sentence from abstrak)
      - explain (bm25 term contributions) if doc_idx available
    """
    # take wider candidate pool for rerank + diversify
    docs_by_id = getattr(index, "docs_by_id", None)
    base = bm25_search(index, query_tokens, top_k=top_k * 4, include_meta=True)
    qset = set([t.lower() for t in query_tokens if t])

    # enrich + rerank
    for item in base:
        # bonuses
        bonus = 0.0
        bonus += _keyword_overlap_bonus(qset, item.get("keyword"))
        bonus += _year_freshness_bonus(item.get("tanggal"))

        item["score2"] = float(item.get("score", 0.0) + bonus)

        # snippet (human-readable)
        item["snippet"] = _best_snippet(item.get("abstrak"), query_tokens)

        # explain (if possible)
        expl = _explain_bm25(index, query_tokens, item, top_terms=6)
        if expl:
            item["explain"] = expl

    base.sort(key=lambda x: x.get("score2", 0.0), reverse=True)

    if not diversify:
        return base[:top_k]

    selected: List[Dict[str, Any]] = []
    seen_title_tokens: List[Set[str]] = []

    for item in base:
        tt = _title_tokens(item.get("judul") or "")

        if not tt:
            selected.append(item)
            if len(selected) >= top_k:
                break
            continue

        too_similar = False
        for prev in seen_title_tokens:
            jacc = len(tt & prev) / (len(tt | prev) + 1e-9)
            if jacc >= 0.65:
                too_similar = True
                break

        if not too_similar:
            selected.append(item)
            seen_title_tokens.append(tt)

        if len(selected) >= top_k:
            break

    return selected
