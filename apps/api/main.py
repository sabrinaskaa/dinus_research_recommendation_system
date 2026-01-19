# apps/api/main.py
from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

# allow import from /src without installing package
sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

from fastapi import FastAPI
from pydantic import BaseModel, Field

from research_reco.config import load_paths, load_settings
from research_reco.io_utils import read_jsonl, read_json
from research_reco.text_utils import load_stopwords, preprocess_text
from research_reco.bm25 import BM25Index, bm25_search
from research_reco.recommend import recommend_citations
from research_reco.supervisor_profiles import recommend_supervisors
from research_reco.query_expansion import expand_query_from_top_docs
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Research Reco API", version="1.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Load config
# -----------------------------
paths = load_paths()
settings = load_settings()

STOPWORDS = load_stopwords(
    paths.stopwords_file,
    use_sastrawi=settings.use_sastrawi_stopwords,
    use_domain=settings.use_domain_stopwords,
)

STEM_MODE = settings.stemming_mode  # "off" | "full" | "selective"
_processed_cache = read_jsonl(paths.processed_jsonl)
if _processed_cache:
    cand = str(_processed_cache[0].get("stemming_mode", STEM_MODE)).lower().strip()
    STEM_MODE = cand if cand in {"off", "full", "selective"} else "off"

# -----------------------------
# Assets (loaded on startup)
# -----------------------------
BM25: BM25Index | None = None
SUP_PROFILES: Dict[str, Any] | None = None
DOCS_BY_ID: Dict[str, Dict[str, Any]] = {}  # doc_id -> processed doc


class QueryReq(BaseModel):
    query: str = Field(..., min_length=2, description="User query / topic / idea")
    top_k: int = Field(10, ge=1, le=50, description="Number of results to return")


class PreprocessDebugReq(BaseModel):
    text: str = Field(..., min_length=1)
    stem_mode: Optional[str] = Field(None, description="Override: off|full|selective")
    use_sastrawi_stopwords: Optional[bool] = None
    use_domain_stopwords: Optional[bool] = None


@app.on_event("startup")
def _load_assets() -> None:
    global BM25, SUP_PROFILES, DOCS_BY_ID

    # Load BM25 index
    if paths.bm25_index_file.exists():
        BM25 = BM25Index.load(paths.bm25_index_file)

    # Load supervisor profiles
    if paths.supervisor_profiles_file.exists():
        SUP_PROFILES = read_json(paths.supervisor_profiles_file)

    # Load processed docs -> DOCS_BY_ID
    docs = read_jsonl(paths.processed_jsonl)
    DOCS_BY_ID = {}
    for d in docs:
        doc_id = d.get("doc_id")
        if doc_id:
            DOCS_BY_ID[str(doc_id)] = d
    if BM25 is not None:
        BM25.docs_by_id = DOCS_BY_ID  # type: ignore


# -----------------------------
# Helpers: explain/snippet
# -----------------------------
def _matched_terms(query_tokens: List[str], doc_tokens: List[str], limit: int = 12) -> List[str]:
    qset = set(query_tokens)
    dset = set(doc_tokens or [])
    matched = [t for t in query_tokens if t in dset]
    # keep order, unique
    seen = set()
    out = []
    for t in matched:
        if t not in seen:
            out.append(t)
            seen.add(t)
        if len(out) >= limit:
            break
    return out


def _make_snippet(text: str, matched: List[str], max_len: int = 240) -> str:
    """
    super-simple snippet:
    - find earliest occurrence of any matched term
    - cut around it
    """
    if not text:
        return ""
    if not matched:
        return (text[:max_len] + "…") if len(text) > max_len else text

    low = text.lower()
    best_pos = None
    best_term = None
    for t in matched:
        p = low.find(t.lower())
        if p != -1 and (best_pos is None or p < best_pos):
            best_pos = p
            best_term = t

    if best_pos is None:
        return (text[:max_len] + "…") if len(text) > max_len else text

    start = max(0, best_pos - 80)
    end = min(len(text), start + max_len)
    snippet = text[start:end]
    if start > 0:
        snippet = "…" + snippet
    if end < len(text):
        snippet = snippet + "…"
    return snippet


def _attach_explain(items: List[Dict[str, Any]], query_tokens: List[str]) -> List[Dict[str, Any]]:
    """
    Attach:
      - explain.matched_terms
      - snippet.snippet + snippet.matched
    Works because bm25_search now returns meta + tokens/abstrak if exist in docs_meta.
    """
    out = []
    for it in items:
        doc_tokens = it.get("tokens", []) or []
        matched = _matched_terms(query_tokens, doc_tokens, limit=12)

        text_src = (it.get("abstrak") or "") or (it.get("judul") or "")
        snippet = _make_snippet(text_src, matched, max_len=260)

        it2 = dict(it)
        it2["explain"] = {"matched_terms": matched}
        it2["snippet"] = {"snippet": snippet, "matched": matched}
        out.append(it2)
    return out


# -----------------------------
# Routes
# -----------------------------
@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "name": "DINUS Research Recommendation API",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "search_all": "/search",
            "recommend_citations": "/recommend/citations",
            "recommend_supervisors": "/recommend/supervisors",
            "demo_both": "/demo",
            "debug_preprocess": "/debug/preprocess",
            "suggest_keywords": "/suggest/keywords",
        },
        "preprocess": {
            "stemming_mode": STEM_MODE,
            "use_sastrawi_stopwords": settings.use_sastrawi_stopwords,
            "use_domain_stopwords": settings.use_domain_stopwords,
        },
    }


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "bm25_loaded": BM25 is not None,
        "profiles_loaded": SUP_PROFILES is not None,
        "processed_docs_loaded": len(DOCS_BY_ID),
        "preprocess": {
            "stemming_mode": STEM_MODE,
            "use_sastrawi_stopwords": settings.use_sastrawi_stopwords,
            "use_domain_stopwords": settings.use_domain_stopwords,
        },
        "paths": {
            "bm25_index_file": str(paths.bm25_index_file),
            "processed_jsonl": str(paths.processed_jsonl),
            "supervisor_profiles_file": str(paths.supervisor_profiles_file),
        },
    }


@app.post("/debug/preprocess")
def debug_preprocess(req: PreprocessDebugReq) -> Dict[str, Any]:
    stem_mode = STEM_MODE
    if req.stem_mode:
        cand = req.stem_mode.lower().strip()
        if cand in {"off", "full", "selective"}:
            stem_mode = cand

    use_sastrawi = settings.use_sastrawi_stopwords if req.use_sastrawi_stopwords is None else bool(req.use_sastrawi_stopwords)
    use_domain = settings.use_domain_stopwords if req.use_domain_stopwords is None else bool(req.use_domain_stopwords)

    stopwords = load_stopwords(paths.stopwords_file, use_sastrawi=use_sastrawi, use_domain=use_domain)
    tokens = preprocess_text(req.text, stopwords, stem_mode=stem_mode)

    return {
        "text": req.text,
        "stem_mode": stem_mode,
        "use_sastrawi_stopwords": use_sastrawi,
        "use_domain_stopwords": use_domain,
        "tokens": tokens,
        "token_count": len(tokens),
    }


@app.post("/search")
def search(req: QueryReq) -> Dict[str, Any]:
    if BM25 is None:
        return {"error": "BM25 index belum ada. Jalankan: python pipelines/indexing/build_bm25.py"}

    q_tokens = preprocess_text(req.query, STOPWORDS, stem_mode=STEM_MODE)

    # 1) initial retrieval (include_meta=True -> always has doc_idx, meta)
    initial = bm25_search(BM25, q_tokens, top_k=max(20, req.top_k * 2), include_meta=True)

    # 2) query expansion
    q_tokens2 = expand_query_from_top_docs(
        DOCS_BY_ID,
        initial_hits=initial,
        query_tokens=q_tokens,
        max_expand=8,
        top_docs=8,
    )

    # 3) final retrieval
    results = bm25_search(BM25, q_tokens2, top_k=req.top_k, include_meta=True)
    results = _attach_explain(results, q_tokens2)

    return {
        "query": req.query,
        "tokens": q_tokens,
        "expanded_tokens": q_tokens2,
        "stemming_mode": STEM_MODE,
        "results": results,
    }


@app.post("/recommend/citations")
def recommend_citation(req: QueryReq) -> Dict[str, Any]:
    if BM25 is None:
        return {"error": "BM25 index belum ada. Jalankan: python pipelines/indexing/build_bm25.py"}

    q_tokens = preprocess_text(req.query, STOPWORDS, stem_mode=STEM_MODE)

    initial = bm25_search(BM25, q_tokens, top_k=max(25, req.top_k * 3), include_meta=True)
    q_tokens2 = expand_query_from_top_docs(DOCS_BY_ID, initial, q_tokens, max_expand=8, top_docs=8)

    results = recommend_citations(BM25, q_tokens2, top_k=req.top_k, diversify=True)

    # ensure explain/snippet attached (recommend_citations returns meta+score2)
    results = _attach_explain(results, q_tokens2)

    return {
        "query": req.query,
        "tokens": q_tokens,
        "expanded_tokens": q_tokens2,
        "stemming_mode": STEM_MODE,
        "results": results,
    }


@app.post("/recommend/supervisors")
def recommend_supervisor(req: QueryReq) -> Dict[str, Any]:
    if SUP_PROFILES is None:
        return {"error": "Supervisor profiles belum ada. Jalankan: python pipelines/profiling/build_supervisor_profiles.py"}

    q_tokens = preprocess_text(req.query, STOPWORDS, stem_mode=STEM_MODE)
    results = recommend_supervisors(SUP_PROFILES, q_tokens, top_k=min(req.top_k, 20))

    return {
        "query": req.query,
        "tokens": q_tokens,
        "stemming_mode": STEM_MODE,
        "results": results,
    }


@app.post("/suggest/keywords")
def suggest_keywords(req: QueryReq) -> Dict[str, Any]:
    if BM25 is None:
        return {"error": "BM25 index belum ada. Jalankan: python pipelines/indexing/build_bm25.py"}

    q_tokens = preprocess_text(req.query, STOPWORDS, stem_mode=STEM_MODE)
    qset = set(q_tokens)

    hits = bm25_search(BM25, q_tokens, top_k=min(30, max(10, req.top_k * 3)), include_meta=True)

    freq: Dict[str, int] = {}
    used_docs: List[str] = []

    for h in hits[:12]:
        doc_id = h.get("doc_id")
        if not doc_id:
            continue
        doc_id = str(doc_id)
        doc = DOCS_BY_ID.get(doc_id)
        if not doc:
            continue

        used_docs.append(doc_id)
        toks = doc.get("tokens", []) or []
        for t in toks:
            if not t:
                continue
            if t in qset:
                continue
            if len(t) <= 2:
                continue
            freq[t] = freq.get(t, 0) + 1

    suggested = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:15]

    return {
        "query": req.query,
        "tokens": q_tokens,
        "stemming_mode": STEM_MODE,
        "suggested_keywords": [t for t, _ in suggested],
        "support": [{"term": t, "count": c} for t, c in suggested[:10]],
        "used_docs": used_docs[:8],
    }


@app.post("/demo")
def demo(req: QueryReq) -> Dict[str, Any]:
    """
    1 endpoint buat demo dosen:
      input -> keluar Sitasi + Dosen sekaligus (+ tokens).
    """
    out: Dict[str, Any] = {"query": req.query}

    # citations
    if BM25 is not None:
        q_tokens = preprocess_text(req.query, STOPWORDS, stem_mode=STEM_MODE)
        init = bm25_search(BM25, q_tokens, top_k=max(25, req.top_k * 3), include_meta=True)
        q_tokens2 = expand_query_from_top_docs(DOCS_BY_ID, init, q_tokens, max_expand=8, top_docs=8)

        cits = recommend_citations(BM25, q_tokens2, top_k=req.top_k, diversify=True)
        cits = _attach_explain(cits, q_tokens2)

        out["tokens"] = q_tokens
        out["expanded_tokens"] = q_tokens2
        out["citations"] = cits
    else:
        out["citations_error"] = "BM25 index belum ada."

    # supervisors
    if SUP_PROFILES is not None:
        q_tokens_s = preprocess_text(req.query, STOPWORDS, stem_mode=STEM_MODE)
        sups = recommend_supervisors(SUP_PROFILES, q_tokens_s, top_k=min(req.top_k, 20))
        out["supervisors"] = sups
    else:
        out["supervisors_error"] = "Supervisor profiles belum ada."

    out["stemming_mode"] = STEM_MODE
    return out
