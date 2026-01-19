# apps/api/main.py
from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple
import html
import re

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

# Extra lookup maps to make evidence/abstract resolution robust.
# In some cases, a BM25 result may not resolve cleanly to DOCS_BY_ID
# (e.g. when only doc_idx/url is present, or when the client caches older ids).
DOCS_BY_URL: Dict[str, Dict[str, Any]] = {}  # url -> processed doc
DOC_ID_BY_DOC_IDX: Dict[int, str] = {}  # doc_idx -> doc_id (from BM25.docs_meta)


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
    global BM25, SUP_PROFILES, DOCS_BY_ID, DOCS_BY_URL, DOC_ID_BY_DOC_IDX

    # Load BM25 index
    if paths.bm25_index_file.exists():
        BM25 = BM25Index.load(paths.bm25_index_file)

    # Load supervisor profiles
    if paths.supervisor_profiles_file.exists():
        SUP_PROFILES = read_json(paths.supervisor_profiles_file)

    # Load processed docs -> DOCS_BY_ID
    docs = read_jsonl(paths.processed_jsonl)
    DOCS_BY_ID = {}
    DOCS_BY_URL = {}
    for d in docs:
        doc_id = d.get("doc_id")
        if doc_id:
            DOCS_BY_ID[str(doc_id)] = d
        url = (d.get("url") or "").strip()
        if url:
            DOCS_BY_URL[url] = d
    if BM25 is not None:
        BM25.docs_by_id = DOCS_BY_ID  # type: ignore

        # Build doc_idx -> doc_id mapping from BM25 meta (stable even if client caches idx)
        DOC_ID_BY_DOC_IDX = {}
        try:
            meta_list = getattr(BM25, "docs_meta", None) or []
            for i, meta in enumerate(meta_list):
                did = (meta or {}).get("doc_id")
                if did:
                    DOC_ID_BY_DOC_IDX[int(i)] = str(did)
        except Exception:
            DOC_ID_BY_DOC_IDX = {}


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


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"[a-zA-Z0-9_]+", re.UNICODE)


def _best_sentence(text: str, terms: List[str], max_len: int = 360) -> str:
    """Pick the sentence with highest overlap with `terms` (case-insensitive)."""
    if not text:
        return ""
    tset = {t.lower() for t in terms if t}
    if not tset:
        # fallback: just truncate
        return (text[:max_len] + "…") if len(text) > max_len else text

    best = (0, "")
    for s in _SENT_SPLIT.split(text):
        s = (s or "").strip()
        if not s:
            continue
        words = {w.lower() for w in _WORD_RE.findall(s)}
        overlap = len(words & tset)
        if overlap > best[0]:
            best = (overlap, s)

    out = best[1] or text
    if len(out) > max_len:
        out = out[:max_len].rsplit(" ", 1)[0] + "…"
    return out


def _highlight_html(text: str, terms: List[str]) -> str:
    """Escape HTML then wrap matched terms with <b>...</b>."""
    if not text:
        return ""
    safe = html.escape(text)

    uniq: List[str] = []
    seen = set()
    for t in terms:
        t = (t or "").strip()
        if not t:
            continue
        tl = t.lower()
        if tl in seen:
            continue
        seen.add(tl)
        uniq.append(t)

    # Replace longer terms first to reduce nested matches.
    uniq.sort(key=lambda x: len(x), reverse=True)

    for t in uniq:
        pat = re.compile(rf"(?i)(?<![A-Za-z0-9_])({re.escape(t)})(?![A-Za-z0-9_])")
        safe = pat.sub(lambda m: f"<b>{m.group(1)}</b>", safe)

    return safe


def _lookup_full_doc(it: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Resolve a result item to the full processed doc.

    Primary key: doc_id
    Fallbacks:
      - doc_idx -> doc_id (BM25 meta)
      - url
    """
    # 1) doc_id
    doc_id = (it.get("doc_id") or "").__str__().strip()
    if doc_id:
        d = DOCS_BY_ID.get(doc_id)
        if isinstance(d, dict):
            return d

    # 2) doc_idx
    di = it.get("doc_idx", None)
    try:
        if di is not None:
            did = DOC_ID_BY_DOC_IDX.get(int(di))
            if did:
                d = DOCS_BY_ID.get(did)
                if isinstance(d, dict):
                    return d
    except Exception:
        pass

    # 3) url
    url = (it.get("url") or "").strip()
    if url:
        d = DOCS_BY_URL.get(url)
        if isinstance(d, dict):
            return d

    return None


def _attach_explain(
    items: List[Dict[str, Any]],
    highlight_terms: List[str],
) -> List[Dict[str, Any]]:
    """
    Attach:
      - explain.matched_terms
      - snippet.snippet + snippet.matched
    Works because bm25_search now returns meta + tokens/abstrak if exist in docs_meta.
    """
    out = []
    for it in items:
        full = _lookup_full_doc(it)

        doc_tokens = (it.get("tokens") or [])
        if (not doc_tokens) and isinstance(full, dict):
            doc_tokens = full.get("tokens") or []

        # Matched terms are based on *stemmed tokens* (BM25 space)
        matched = _matched_terms(highlight_terms, doc_tokens, limit=12)

        # Prefer full abstrak from processed docs.
        # Some retrieval outputs only carry meta fields, so relying on it.get('abstrak') can be empty.
        abstrak = ""
        if isinstance(full, dict):
            abstrak = full.get("abstrak") or ""
        if not abstrak:
            abstrak = (it.get("abstrak") or "")

        judul = ""
        if isinstance(full, dict):
            judul = full.get("judul") or ""
        if not judul:
            judul = (it.get("judul") or "")

        # Tokens should come from processed docs, to make matched_terms accurate.
        if (not doc_tokens) and isinstance(full, dict):
            doc_tokens = full.get("tokens") or []

        evidence_text = _best_sentence(abstrak, highlight_terms, max_len=360) if abstrak else judul
        evidence_html = _highlight_html(evidence_text, highlight_terms)

        it2 = dict(it)
        it2["explain"] = {"matched_terms": matched, "abstract_html": evidence_html, "abstract_text": evidence_text}
        # keep backward compatible snippet fields
        it2["snippet"] = {"snippet": evidence_text, "matched": matched, "html": evidence_html}
        # expose abstrak for UI if desired
        if abstrak and not it2.get("abstrak"):
            it2["abstrak"] = abstrak
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

    # Raw (non-stemmed) terms: helps highlight words in the *original* abstract
    # even when we use stemming for retrieval.
    raw_terms = [w.lower() for w in _WORD_RE.findall(req.query or "") if len(w) > 2]
    raw_terms = [w for w in raw_terms if w not in STOPWORDS]

    # 1) initial retrieval (include_meta=True -> always has doc_idx, meta)
    initial = bm25_search(BM25, q_tokens, top_k=max(20, req.top_k * 2), include_meta=True)

    # 2) query expansion
    q_tokens2 = expand_query_from_top_docs(
        DOCS_BY_ID,
        initial_hits=initial,
        query_tokens=q_tokens,
        max_expand=8,
        top_docs=8,
        idf=getattr(BM25, "idf", None),
    )

    # 3) final retrieval
    results = bm25_search(BM25, q_tokens2, top_k=req.top_k, include_meta=True)
    highlight_terms = list(dict.fromkeys((q_tokens or []) + raw_terms))
    results = _attach_explain(results, highlight_terms)

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
    raw_terms = [w.lower() for w in _WORD_RE.findall(req.query or "") if len(w) > 2]
    raw_terms = [w for w in raw_terms if w not in STOPWORDS]

    initial = bm25_search(BM25, q_tokens, top_k=max(25, req.top_k * 3), include_meta=True)
    q_tokens2 = expand_query_from_top_docs(
        DOCS_BY_ID,
        initial_hits=initial,
        query_tokens=q_tokens,
        max_expand=8,
        top_docs=8,
        idf=getattr(BM25, "idf", None),
    )

    # Pass both original tokens and expanded tokens so the recommender can
    # use anchors from the original query to avoid lexical drift.
    results = recommend_citations(BM25, q_tokens2, top_k=req.top_k, diversify=True, original_query_tokens=q_tokens)

    # ensure explain/snippet attached (recommend_citations returns meta+score2)
    highlight_terms = list(dict.fromkeys((q_tokens or []) + raw_terms))
    results = _attach_explain(results, highlight_terms)

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
        raw_terms = [w.lower() for w in _WORD_RE.findall(req.query or "") if len(w) > 2]
        raw_terms = [w for w in raw_terms if w not in STOPWORDS]
        init = bm25_search(BM25, q_tokens, top_k=max(25, req.top_k * 3), include_meta=True)
        q_tokens2 = expand_query_from_top_docs(
            DOCS_BY_ID,
            initial_hits=init,
            query_tokens=q_tokens,
            max_expand=8,
            top_docs=8,
            idf=getattr(BM25, "idf", None),
        )

        cits = recommend_citations(BM25, q_tokens2, top_k=req.top_k, diversify=True, original_query_tokens=q_tokens)
        highlight_terms = list(dict.fromkeys((q_tokens or []) + raw_terms))
        cits = _attach_explain(cits, highlight_terms)

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
