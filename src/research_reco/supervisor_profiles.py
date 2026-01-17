from __future__ import annotations
import math
from collections import defaultdict
from typing import Dict, List, Any


def _parse_year(tanggal: str | None) -> int:
    if not tanggal:
        return 0
    t = tanggal.strip()
    if len(t) >= 4 and t[:4].isdigit():
        return int(t[:4])
    return 0


def build_supervisor_profiles(processed_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Profiles ONLY from source == "dosbing".
    Each dosen treated as one "document" for IDF across dosen.
    """
    dosen_docs = [d for d in processed_docs if d.get("source") == "dosbing" and d.get("dosen")]
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for d in dosen_docs:
        grouped[d["dosen"]].append(d)

    tf_dosen: Dict[str, Dict[str, int]] = {}
    for dosen, docs in grouped.items():
        tf: Dict[str, int] = defaultdict(int)
        for doc in docs:
            for t in doc.get("tokens", []):
                tf[t] += 1
        tf_dosen[dosen] = dict(tf)

    df: Dict[str, int] = defaultdict(int)
    for _, tf in tf_dosen.items():
        for term in tf.keys():
            df[term] += 1

    N = max(1, len(tf_dosen))
    idf: Dict[str, float] = {}
    for term, dfi in df.items():
        idf[term] = math.log(1 + (N - dfi + 0.5) / (dfi + 0.5))

    profiles: Dict[str, Any] = {}
    for dosen, tf in tf_dosen.items():
        vec: Dict[str, float] = {}
        for term, f in tf.items():
            vec[term] = (1.0 + math.log(1 + f)) * idf.get(term, 0.0)

        norm = math.sqrt(sum(v * v for v in vec.values())) + 1e-9
        top_terms = sorted(vec.items(), key=lambda x: x[1], reverse=True)[:20]
        top_terms_only = [t for t, _ in top_terms]

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
            "norm": norm,
            "samples": samples,
            "pub_count": len(pubs),
        }

    return {
        "idf": idf,
        "profiles": profiles,
    }


def recommend_supervisors(
    profiles_obj: Dict[str, Any],
    query_tokens: List[str],
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    idf: Dict[str, float] = profiles_obj.get("idf", {})
    profiles: Dict[str, Any] = profiles_obj.get("profiles", {})

    qtf: Dict[str, int] = defaultdict(int)
    for t in query_tokens:
        qtf[t] += 1

    qvec: Dict[str, float] = {}
    for term, f in qtf.items():
        if term in idf:
            qvec[term] = (1.0 + math.log(1 + f)) * idf.get(term, 0.0)

    qnorm = math.sqrt(sum(v * v for v in qvec.values())) + 1e-9

    def cosine(vec: Dict[str, float], norm: float) -> float:
        dot = 0.0
        for term, w in qvec.items():
            dot += w * vec.get(term, 0.0)
        return dot / (qnorm * (norm + 1e-9))

    scored = []
    qset = set(query_tokens)

    for dosen, info in profiles.items():
        sim = cosine(info["vector"], info["norm"])
        matched = [t for t in info.get("top_terms", []) if t in qset][:8]
        bonus = 0.03 * len(matched)

        scored.append({
            "dosen": dosen,
            "score": float(sim + bonus),
            "similarity": float(sim),
            "matched_terms": matched,
            "pub_count": info.get("pub_count", 0),
            "samples": info.get("samples", []),
            "top_terms": info.get("top_terms", [])[:10],
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]
