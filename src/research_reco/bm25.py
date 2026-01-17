from __future__ import annotations
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Any


@dataclass
class BM25Index:
    k1: float
    b: float
    vocab_df: Dict[str, int]                 # document frequency
    idf: Dict[str, float]
    doc_len: List[int]
    avgdl: float
    postings: Dict[str, List[Tuple[int, int]]]  # term -> list of (doc_idx, tf)
    docs_meta: List[Dict[str, Any]]          # meta per doc

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Path) -> "BM25Index":
        with path.open("rb") as f:
            return pickle.load(f)


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
            tf[t] = tf.get(t, 0) + 1
        for t, f in tf.items():
            vocab_df[t] = vocab_df.get(t, 0) + 1
            postings.setdefault(t, []).append((i, f))

    idf: Dict[str, float] = {}
    for t, df in vocab_df.items():
        idf[t] = math.log(1 + (N - df + 0.5) / (df + 0.5)) if N else 0.0

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


def bm25_search(index: BM25Index, query_tokens: List[str], top_k: int = 10) -> List[Dict[str, Any]]:
    if not query_tokens:
        return []

    scores: Dict[int, float] = {}
    q_terms: Dict[str, int] = {}
    for t in query_tokens:
        q_terms[t] = q_terms.get(t, 0) + 1

    for term in q_terms.keys():
        if term not in index.postings:
            continue
        idf = index.idf.get(term, 0.0)
        for doc_idx, tf in index.postings[term]:
            dl = index.doc_len[doc_idx]
            denom = tf + index.k1 * (1 - index.b + index.b * (dl / (index.avgdl + 1e-9)))
            score = idf * (tf * (index.k1 + 1) / (denom + 1e-9))
            scores[doc_idx] = scores.get(doc_idx, 0.0) + score

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    out: List[Dict[str, Any]] = []
    for doc_idx, score in ranked:
        meta = index.docs_meta[doc_idx].copy()
        meta["score"] = float(score)
        out.append(meta)
    return out
