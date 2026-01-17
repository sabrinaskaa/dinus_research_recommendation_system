from __future__ import annotations
from typing import List, Dict, Any, Set
from .bm25 import BM25Index, bm25_search


def _keyword_overlap_bonus(query_tokens: Set[str], keyword: str | None) -> float:
    if not keyword:
        return 0.0
    kw_tokens = set([k.strip().lower() for k in keyword.replace(",", " ").split() if k.strip()])
    if not kw_tokens:
        return 0.0
    inter = len(query_tokens & kw_tokens)
    return 0.15 * inter


def recommend_citations(
    index: BM25Index,
    query_tokens: List[str],
    top_k: int = 10,
    diversify: bool = True,
) -> List[Dict[str, Any]]:
    """
    Sitasi: dari SEMUA corpus (index dibuild dari semua docs).
    """
    base = bm25_search(index, query_tokens, top_k=top_k * 4)
    qset = set(query_tokens)

    for item in base:
        bonus = 0.0
        bonus += _keyword_overlap_bonus(qset, item.get("keyword"))

        tgl = (item.get("tanggal") or "").strip()
        if len(tgl) >= 4 and tgl[:4].isdigit():
            year = int(tgl[:4])
            bonus += max(0.0, (year - 2018)) * 0.01

        item["score2"] = float(item["score"] + bonus)

    base.sort(key=lambda x: x["score2"], reverse=True)

    if not diversify:
        return base[:top_k]

    selected: List[Dict[str, Any]] = []
    seen_title_tokens: List[Set[str]] = []

    for item in base:
        title = (item.get("judul") or "").lower()
        tt = set([w for w in title.split() if w])

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
