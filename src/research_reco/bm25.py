# src/research_reco/bm25.py
from __future__ import annotations

import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional


@dataclass
class BM25Index:
    k1: float
    b: float
    vocab_df: Dict[str, int]
    idf: Dict[str, float]
    doc_len: List[int]
    avgdl: float
    postings: Dict[str, List[Tuple[int, int]]]
    docs_meta: List[Dict[str, Any]]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Path) -> "BM25Index":
        with path.open("rb") as f:
            return pickle.load(f)

    def get_scores(self, query_tokens: List[str]) -> List[float]:
        """
        Return BM25 score for every doc (length = number of docs).
        Useful for debug/explain later.
        """
        N = len(self.docs_meta)
        if N == 0 or not query_tokens:
            return [0.0] * N

        # query term counts (not used in BM25 classic unless you want qtf weighting)
        q_terms: Dict[str, int] = {}
        for t in query_tokens:
            if not t:
                continue
            q_terms[t] = q_terms.get(t, 0) + 1

        scores = [0.0] * N

        for term in q_terms.keys():
            plist = self.postings.get(term)
            if not plist:
                continue

            idf = float(self.idf.get(term, 0.0))

            for doc_idx, tf in plist:
                dl = self.doc_len[doc_idx]
                denom = tf + self.k1 * (1 - self.b + self.b * (dl / (self.avgdl + 1e-9)))
                scores[doc_idx] += idf * (tf * (self.k1 + 1) / (denom + 1e-9))

        return scores


def build_bm25_index(
    docs_tokens: List[List[str]],
    docs_meta: List[Dict[str, Any]],
    k1: float = 1.5,
    b: float = 0.75,
) -> BM25Index:
    N = len(docs_tokens)

    vocab_df: Dict[str, int] = {}
    postings: Dict[str, List[Tuple[int, int]]] = {}

    doc_len: List[int] = [len(toks) for toks in docs_tokens]
    avgdl = (sum(doc_len) / N) if N else 0.0

    for i, toks in enumerate(docs_tokens):
        tf: Dict[str, int] = {}
        for t in toks:
            if not t:
                continue
            tf[t] = tf.get(t, 0) + 1

        for t, f in tf.items():
            vocab_df[t] = vocab_df.get(t, 0) + 1
            postings.setdefault(t, []).append((i, f))

    idf: Dict[str, float] = {}
    for t, df in vocab_df.items():
        idf[t] = math.log(1 + (N - df + 0.5) / (df + 0.5)) if N else 0.0

    if len(docs_meta) != N:
        raise ValueError(f"docs_meta length ({len(docs_meta)}) must match docs_tokens length ({N})")

    return BM25Index(
        k1=k1,
        b=b,
        vocab_df=vocab_df,
        idf=idf,
        doc_len=doc_len,
        avgdl=avgdl,
        postings=postings,
        docs_meta=docs_meta,
    )


def bm25_search(
    index: BM25Index,
    query_tokens: List[str],
    top_k: int = 10,
    include_meta: bool = True,
) -> List[Dict[str, Any]]:
    """
    Returns results with:
      - doc_idx (ALWAYS)
      - score
      - plus meta fields from docs_meta (if include_meta=True)
    """
    if not query_tokens:
        return []

    # Compute sparse scores only for docs hit by postings (faster)
    scores: Dict[int, float] = {}

    q_terms: Dict[str, int] = {}
    for t in query_tokens:
        if not t:
            continue
        q_terms[t] = q_terms.get(t, 0) + 1

    for term in q_terms.keys():
        plist = index.postings.get(term)
        if not plist:
            continue

        idf = float(index.idf.get(term, 0.0))

        for doc_idx, tf in plist:
            dl = index.doc_len[doc_idx]
            denom = tf + index.k1 * (1 - index.b + index.b * (dl / (index.avgdl + 1e-9)))
            score = idf * (tf * (index.k1 + 1) / (denom + 1e-9))
            scores[doc_idx] = scores.get(doc_idx, 0.0) + score

    if not scores:
        return []

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    out: List[Dict[str, Any]] = []
    for doc_idx, score in ranked:
        if include_meta:
            meta = index.docs_meta[doc_idx].copy()
        else:
            meta = {}

        # ALWAYS attach doc_idx and score
        meta["doc_idx"] = int(doc_idx)
        meta["score"] = float(score)

        # ensure doc_id exists (for eval/UI consistency)
        if "doc_id" not in meta or meta.get("doc_id") is None or str(meta.get("doc_id")).strip() == "":
            meta["doc_id"] = f"doc_{doc_idx}"

        out.append(meta)

    return out
