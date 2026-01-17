from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Dict

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

app = FastAPI(title="Research Reco API", version="1.0.0")

paths = load_paths()
settings = load_settings()

# Stopwords ON/OFF based on settings
STOPWORDS = load_stopwords(
    paths.stopwords_file,
    use_sastrawi=settings.use_sastrawi_stopwords,
    use_domain=settings.use_domain_stopwords,
)

# Stemming mode should match what was used at preprocessing time.
# Default: from settings; override: from processed_jsonl if present.
STEM_MODE = settings.stemming_mode  # "off" | "full" | "selective"
_processed_cache = read_jsonl(paths.processed_jsonl)
if _processed_cache:
    cand = str(_processed_cache[0].get("stemming_mode", STEM_MODE)).lower().strip()
    if cand in {"off", "full", "selective"}:
        STEM_MODE = cand
    else:
        STEM_MODE = "off"

# Optional: ensure stopword flags match processed file too (if you want strictness)
if _processed_cache:
    # If different, we keep runtime settings for stopwords (you can enforce strict if desired)
    pass

BM25: BM25Index | None = None
SUP_PROFILES: Dict[str, Any] | None = None


class QueryReq(BaseModel):
    query: str = Field(..., min_length=2, description="User query / topic / idea")
    top_k: int = Field(10, ge=1, le=50, description="Number of results to return")


@app.on_event("startup")
def _load_assets() -> None:
    global BM25, SUP_PROFILES
    if paths.bm25_index_file.exists():
        BM25 = BM25Index.load(paths.bm25_index_file)
    if paths.supervisor_profiles_file.exists():
        SUP_PROFILES = read_json(paths.supervisor_profiles_file)


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "bm25_loaded": BM25 is not None,
        "profiles_loaded": SUP_PROFILES is not None,
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


@app.post("/search")
def search(req: QueryReq) -> Dict[str, Any]:
    """
    Generic IR search over ALL corpora (UDINUS + DOSBING) because BM25 index was built from all docs.
    """
    if BM25 is None:
        return {
            "error": "BM25 index belum ada. Jalankan: python pipelines/indexing/build_bm25.py"
        }

    q_tokens = preprocess_text(req.query, STOPWORDS, stem_mode=STEM_MODE)
    results = bm25_search(BM25, q_tokens, top_k=req.top_k)

    return {
        "query": req.query,
        "tokens": q_tokens,
        "stemming_mode": STEM_MODE,
        "results": results,
    }


@app.post("/recommend/citations")
def recommend_citation(req: QueryReq) -> Dict[str, Any]:
    """
    Citation recommendation from ALL corpora.
    Uses BM25 + light rerank + light diversification (see research_reco/recommend.py).
    """
    if BM25 is None:
        return {
            "error": "BM25 index belum ada. Jalankan: python pipelines/indexing/build_bm25.py"
        }

    q_tokens = preprocess_text(req.query, STOPWORDS, stem_mode=STEM_MODE)
    results = recommend_citations(BM25, q_tokens, top_k=req.top_k, diversify=True)

    return {
        "query": req.query,
        "tokens": q_tokens,
        "stemming_mode": STEM_MODE,
        "results": results,
    }


@app.post("/recommend/supervisors")
def recommend_supervisor(req: QueryReq) -> Dict[str, Any]:
    """
    Supervisor recommendation ONLY from corpus with parent folder 'Publikasi Dosbing'
    because supervisor profiles are built only from source == 'dosbing'.
    """
    if SUP_PROFILES is None:
        return {
            "error": "Supervisor profiles belum ada. Jalankan: python pipelines/profiling/build_supervisor_profiles.py"
        }

    q_tokens = preprocess_text(req.query, STOPWORDS, stem_mode=STEM_MODE)
    results = recommend_supervisors(SUP_PROFILES, q_tokens, top_k=min(req.top_k, 20))

    return {
        "query": req.query,
        "tokens": q_tokens,
        "stemming_mode": STEM_MODE,
        "results": results,
    }
