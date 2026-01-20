# src/research_reco/supervisor_profiles.py
from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Any, Tuple


# Kata-kata super umum yang sering bikin rekomendasi dosen "nyasar".
# Ini bukan stopword bahasa Indonesia biasa, tapi "generic research words".
# Kita TURUNKAN bobotnya supaya query seperti "machine learning" tidak
# otomatis mengangkat dosen yang domainnya beda jauh.
GENERIC_TERMS = {
    # Indo
    "deteksi", "prediksi", "klasifikasi", "klaster", "pengembangan", "analisis",
    "sistem", "metode", "model", "algoritma", "pendekatan", "penerapan",
    "berbasis", "menggunakan", "dengan", "untuk", "pada", "dalam", "terhadap",
    "data", "dataset", "fitur", "seleksi", "optimasi", "evaluasi",
    # English
    "method", "methods", "model", "models", "algorithm", "algorithms",
    "approach", "system", "analysis", "data", "feature", "features",
    "classification", "prediction", "detection", "optimization", "evaluation",
    "machine", "learning", "deep", "neural", "network", "networks",
}


def _parse_year(tanggal: str | None) -> int:
    if not tanggal:
        return 0
    t = str(tanggal).strip()
    if len(t) >= 4 and t[:4].isdigit():
        try:
            return int(t[:4])
        except Exception:
            return 0
    return 0


def _safe_log1p(x: float) -> float:
    return math.log(1.0 + max(0.0, x))


def build_supervisor_profiles(processed_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Build supervisor profiles ONLY from source == "dosbing" and dosen != None.

    Treat each DOSEN as one "document" for DF/IDF across dosen.
    Vector weighting: (1 + log(1 + tf)) * idf

    Returns:
      {
        "idf": {term: idf, ...},
        "profiles": {
            "<dosen_name>": {
                "vector": {term: weight, ...},
                "norm": float,
                "top_terms": [t1..],
                "samples": [{doc_id, judul, tanggal, url}, ...],
                "pub_count": int
            }
        },
        "meta": {"dosen_count": int, "doc_count": int}
      }
    """
    # filter dosbing docs
    dosen_docs = [
        d for d in processed_docs
        if d.get("source") == "dosbing" and d.get("dosen")
    ]

    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for d in dosen_docs:
        grouped[str(d["dosen"])].append(d)

    # tf per dosen
    tf_dosen: Dict[str, Dict[str, int]] = {}
    for dosen, docs in grouped.items():
        tf: Dict[str, int] = defaultdict(int)
        for doc in docs:
            for t in doc.get("tokens", []) or []:
                if not t:
                    continue
                tf[t] += 1
        tf_dosen[dosen] = dict(tf)

    # df across dosen
    df: Dict[str, int] = defaultdict(int)
    for _, tf in tf_dosen.items():
        for term in tf.keys():
            df[term] += 1

    N = max(1, len(tf_dosen))
    idf: Dict[str, float] = {}
    for term, dfi in df.items():
        # BM25-style idf
        idf[term] = math.log(1 + (N - dfi + 0.5) / (dfi + 0.5))

    # Build profiles
    profiles: Dict[str, Any] = {}
    for dosen, tf in tf_dosen.items():
        vec: Dict[str, float] = {}
        for term, f in tf.items():
            vec[term] = (1.0 + _safe_log1p(f)) * float(idf.get(term, 0.0))

        norm = math.sqrt(sum(v * v for v in vec.values())) + 1e-9

        # top_terms: avoid extremely generic terms (idf too low)
        idf_cutoff = 0.15
        top_terms_scored = sorted(
            ((t, w) for t, w in vec.items() if idf.get(t, 0.0) >= idf_cutoff),
            key=lambda x: x[1],
            reverse=True
        )[:25]

        # fallback if filtering removed too much
        if len(top_terms_scored) < 8:
            top_terms_scored = sorted(vec.items(), key=lambda x: x[1], reverse=True)[:25]

        top_terms_only = [t for t, _ in top_terms_scored[:20]]

        pubs = grouped[dosen]
        pubs_sorted = sorted(pubs, key=lambda d: _parse_year(d.get("tanggal")), reverse=True)

        samples = []
        for p in pubs_sorted[:3]:
            samples.append({
                "doc_id": p.get("doc_id"),
                "judul": p.get("judul"),
                "tanggal": p.get("tanggal"),
                "url": p.get("url"),
            })

        profiles[dosen] = {
            "top_terms": top_terms_only,
            "vector": vec,
            "norm": float(norm),
            "samples": samples,
            "pub_count": int(len(pubs)),
        }

    return {
        "idf": idf,
        "profiles": profiles,
        "meta": {
            "dosen_count": int(len(profiles)),
            "doc_count": int(len(dosen_docs)),
        },
    }


def recommend_supervisors(
    profiles_obj: Dict[str, Any],
    query_tokens: List[str],
    top_k: int = 5,
    *,
    min_similarity: float = 0.08,
    rel_cutoff: float = 0.35,
    anchor_top_n: int = 2,
    anchor_idf_min: float = 0.65,
    generic_downweight: float = 0.35,
) -> List[Dict[str, Any]]:
    idf: Dict[str, float] = profiles_obj.get("idf", {}) or {}
    profiles: Dict[str, Any] = profiles_obj.get("profiles", {}) or {}

    # query tf
    qtf: Dict[str, int] = defaultdict(int)
    for t in query_tokens or []:
        if not t:
            continue
        qtf[t] += 1

    # Anchor terms: pilih term query paling spesifik (idf tinggi), non-generic.
    q_terms_unique = list(qtf.keys())
    anchor_candidates = [t for t in q_terms_unique if t in idf and t not in GENERIC_TERMS]
    anchor_candidates.sort(key=lambda t: float(idf.get(t, 0.0)), reverse=True)
    anchors = [
        t for t in anchor_candidates[: max(1, anchor_top_n)]
        if float(idf.get(t, 0.0)) >= anchor_idf_min
    ]

    # query vector
    qvec: Dict[str, float] = {}
    for term, f in qtf.items():
        if term in idf:
            w = (1.0 + _safe_log1p(f)) * float(idf.get(term, 0.0))
            # downweight kata generik, kecuali kalau dia anchor
            if term in GENERIC_TERMS and term not in anchors:
                w *= float(generic_downweight)
            qvec[term] = float(w)

    # fallback kalau qvec kosong
    if not qvec and qtf:
        for term, f in qtf.items():
            w = (1.0 + _safe_log1p(f)) * float(idf.get(term, 0.0))
            if w > 0:
                qvec[term] = float(w)

    qnorm = math.sqrt(sum(v * v for v in qvec.values())) + 1e-9
    qset = set(query_tokens or [])

    def cosine_and_evidence(vec: Dict[str, float], norm: float, top_evidence: int = 6) -> Tuple[float, List[Dict[str, Any]]]:
        dot = 0.0
        evidence = []
        for term, qw in qvec.items():
            dw = float(vec.get(term, 0.0))
            if dw <= 0:
                continue
            c = qw * dw
            dot += c
            evidence.append({"term": term, "q_w": float(qw), "d_w": float(dw), "contrib": float(c)})

        sim = dot / (qnorm * (float(norm) + 1e-9))
        evidence.sort(key=lambda x: x["contrib"], reverse=True)
        return float(sim), evidence[:top_evidence]

    scored: List[Dict[str, Any]] = []

    for dosen, info in profiles.items():
        vec = info.get("vector", {}) or {}
        norm = float(info.get("norm", 1.0))

        # Anchor gating: kalau query punya anchor spesifik, dosen wajib punya itu.
        if anchors:
            has_anchor = any(float(vec.get(a, 0.0)) > 0.0 for a in anchors)
            if not has_anchor:
                continue

        sim, evidence_terms = cosine_and_evidence(vec, norm, top_evidence=6)

        # quick matched terms for UI readability (overlap with top_terms)
        matched = [t for t in (info.get("top_terms", []) or []) if t in qset][:8]

        # small bonus for overlap readability (tidak boleh overpower)
        bonus = 0.03 * len(matched)

        # skip noisy
        if sim <= 0 and not matched:
            continue

        scored.append({
            "dosen": dosen,
            "score": float(sim + bonus),
            "similarity": float(sim),
            "matched_terms": matched,
            "evidence_terms": evidence_terms,
            "pub_count": int(info.get("pub_count", 0)),
            "samples": info.get("samples", []) or [],
            "top_terms": (info.get("top_terms", []) or [])[:10],
        })

    scored.sort(key=lambda x: x["score"], reverse=True)

    # Hard filter: buang similarity kecil banget (noise), tapi tetap relatif.
    if scored:
        best_sim = max(float(x.get("similarity", 0.0)) for x in scored) or 0.0
        cutoff = max(float(min_similarity), float(best_sim) * float(rel_cutoff))
        filtered = [x for x in scored if float(x.get("similarity", 0.0)) >= cutoff]
        if filtered:
            return filtered[:top_k]

    return scored[:top_k]
