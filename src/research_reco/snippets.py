# src/research_reco/snippets.py
from __future__ import annotations
import re
from typing import List, Optional, Dict, Any

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

def best_snippet(text: str, query_tokens: List[str], max_len: int = 260) -> Dict[str, Any]:
    """
    Ambil 1 kalimat (atau potongan) yang paling banyak mengandung query tokens.
    """
    if not text:
        return {"snippet": "", "score": 0, "matched": []}

    sents = _SENT_SPLIT.split(text.strip())
    qset = set(t.lower() for t in query_tokens if t)

    best = ""
    best_score = -1
    best_matched = []

    for s in sents:
        toks = re.findall(r"[a-zA-Z0-9_]+", s.lower())
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
