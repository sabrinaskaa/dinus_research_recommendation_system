# src/research_reco/query_expansion.py
from __future__ import annotations

from typing import Dict, List, Any, Set, Tuple, Optional
import math


# Generic words that commonly appear across many papers.
# We do *not* want these in expansions.
GENERIC_TERMS: Set[str] = {
    # Indo
    "penelitian", "analisis", "sistem", "metode", "model", "algoritma",
    "menggunakan", "berbasis", "dengan", "untuk", "pada", "dalam", "terhadap",
    "pengembangan", "penerapan", "perancangan", "implementasi", "pengujian",
    "uji", "hasil", "kinerja", "tingkat", "meningkatkan", "akurasi",
    "studi", "kasus", "data", "dataset", "fitur", "seleksi", "optimasi",
    "evaluasi", "perbandingan", "rekomendasi", "pemilihan", "program",
    "jadi", "sesuai", "kemampuan", "universitas", "dian", "nuswantoro",
    # Drift-prone tokens that tend to appear in many unrelated academic titles
    # and were observed leaking into expansions (e.g., UI screenshot: "mahasiswa").
    "mahasiswa", "s1", "prodi", "teknik", "informatika", "jalur", "peminatan",
    "indeks", "index", "series", "tree",
    # English
    "study", "analysis", "system", "method", "methods", "model", "models",
    "algorithm", "algorithms", "approach", "using", "based", "data",
    "dataset", "feature", "features", "evaluation", "performance",
    "application", "implementation", "design",
}


def expand_query_from_top_docs(
    docs_by_id: Dict[str, Dict[str, Any]],
    initial_hits: List[Dict[str, Any]],
    query_tokens: List[str],
    max_expand: int = 8,
    top_docs: int = 8,
    *,
    idf: Optional[Dict[str, float]] = None,
    min_df_in_top_docs: int = 2,
    min_idf: float = 0.35,
) -> List[str]:

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
            if t in GENERIC_TERMS:
                continue
            tf[t] = tf.get(t, 0) + 1
            seen_in_doc.add(t)

        for t in seen_in_doc:
            df[t] = df.get(t, 0) + 1

    if used == 0:
        return query_tokens

    scored: List[Tuple[str, float]] = []
    for t, c in tf.items():
        dfi = df.get(t, 1)
        if dfi < min_df_in_top_docs:
            continue

        # If idf is available, filter out low-idf generic-ish terms.
        term_idf = float(idf.get(t, 0.0)) if idf else 0.0
        if idf is not None and term_idf < min_idf:
            continue

        # Score prefers:
        # - repeated terms across docs (df)
        # - frequent term in top-docs (tf)
        # - globally specific terms (idf)
        score = float(c) * math.log(1.0 + float(dfi))
        if idf is not None:
            score *= max(0.0, term_idf)

        scored.append((t, float(score)))

    if not scored:
        return query_tokens

    scored.sort(key=lambda x: x[1], reverse=True)
    expansion = [t for t, _ in scored[:max_expand]]

    return query_tokens + expansion
