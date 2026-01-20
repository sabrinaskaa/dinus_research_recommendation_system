from __future__ import annotations

import html
import json
import re
import sys
import urllib.parse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st


# =========================
# Make /src importable
# =========================
HERE = Path(__file__).resolve()
SRC_CANDIDATES = [
    HERE.parent / "src",
    HERE.parent.parent / "src",
    HERE.parent / "dinus_research_recommendation_system" / "src",
]
for p in SRC_CANDIDATES:
    if p.exists():
        sys.path.insert(0, str(p))
        break


try:
    from research_reco.config import load_paths, load_settings
    from research_reco.io_utils import read_json, read_jsonl
    from research_reco.text_utils import load_stopwords, preprocess_text
    from research_reco.bm25 import BM25Index, bm25_search
    from research_reco.query_expansion import expand_query_from_top_docs
    from research_reco.recommend import recommend_citations
    from research_reco.supervisor_profiles import recommend_supervisors
except Exception as e:
    st.error(
        "Gagal import modul project dari folder src/.\n\n"
        "Pastikan struktur repo kamu ada folder `src/research_reco` dan Streamlit dijalankan dari root repo.\n\n"
        f"Detail: {e}"
    )
    st.stop()



# =========================
# Config
# =========================
st.set_page_config(page_title="DINUS Research Recommendation", layout="wide")


# =========================
# CSS (Dark UI + tidy spacing + html buttons)
# =========================
CSS = """
<style>
:root{
  --bg:#070A12;
  --panel: rgba(255,255,255,0.04);
  --panel2: rgba(255,255,255,0.06);
  --border: rgba(255,255,255,0.12);
  --muted: rgba(255,255,255,0.65);
  --text: rgba(255,255,255,0.92);
  --pill: rgba(255,255,255,0.08);
  --kbd: rgba(255,255,255,0.10);
}

html, body, [class*="css"] {
  background: radial-gradient(1200px 600px at 10% 0%, rgba(100,120,255,0.10), transparent 60%),
              radial-gradient(900px 500px at 80% 10%, rgba(255,100,180,0.08), transparent 60%),
              var(--bg);
  color: var(--text);
}

.block-container{
  padding-top: 18px;
  max-width: 1100px;
}

/* Fix space above input (hide label) */
div[data-testid="stTextInput"] label { display:none !important; }
div[data-testid="stTextInput"] > div { margin-top:0 !important; padding-top:0 !important; }
div[data-testid="stTextInput"] .stTextInput { margin-top:0 !important; }

/* Streamlit default spacing sometimes too big */
div[data-testid="stVerticalBlock"] > div { padding-top: 0 !important; }

h1, h2, h3 { color: var(--text); }
.muted { color: var(--muted); }

.searchCard{
  border: 1px solid var(--border);
  background: var(--panel);
  border-radius: 18px;
  padding: 14px 14px 12px 14px;
  margin-bottom: 14px;
}

.kbd{
  display:inline-block;
  padding: 2px 8px;
  border: 1px solid var(--border);
  background: var(--kbd);
  border-radius: 999px;
  font-size: 12px;
  margin-right: 6px;
}

.pill{
  display:inline-block;
  padding: 3px 10px;
  border: 1px solid var(--border);
  background: var(--pill);
  border-radius: 999px;
  font-size: 12px;
  margin: 4px 6px 0 0;
}

.card{
  border: 1px solid var(--border);
  background: var(--panel2);
  border-radius: 18px;
  padding: 14px;
  margin-top: 12px;
}

.card a{
  color: rgba(160,190,255,0.95);
  text-decoration: none;
}
.card a:hover{ text-decoration: underline; }

.hr{
  height:1px; background: var(--border); margin: 10px 0;
}

.abstract-evidence{
  font-size: 13px;
  line-height: 1.55;
  color: rgba(255,255,255,0.84);
}
.abstract-evidence b{
  background: rgba(255,255,255,0.10);
  padding: 0 2px;
  border-radius: 4px;
}

.small{
  font-size: 12px;
  color: var(--muted);
}

/* Streamlit buttons consistent */
div.stButton > button{
  border: 1px solid var(--border) !important;
  background: rgba(255,255,255,0.06) !important;
  color: var(--text) !important;
  border-radius: 12px !important;
  padding: 10px 12px !important;
  height: 44px !important;
  font-weight: 650 !important;
  white-space: nowrap !important;
  font-size: 13px !important;
}
div.stButton > button:hover{
  background: rgba(255,255,255,0.10) !important;
}

/* Better checkbox color */
[data-testid="stCheckbox"] label { color: var(--text) !important; }

/* HTML action row (Buka, WA, Copy) */
.btnRow{
  display:flex;
  gap:10px;
  flex-wrap:wrap;
  margin-top: 10px;
  align-items: center;
}
.btn{
  display:inline-flex;
  align-items:center;
  justify-content:center;
  padding: 8px 12px;
  border-radius: 12px;
  border: 1px solid var(--border);
  background: rgba(255,255,255,0.06);
  color: var(--text);
  text-decoration:none;
  font-size: 13px;
  cursor:pointer;
  user-select:none;
}
.btn:hover{ background: rgba(255,255,255,0.10); }
.btn:active{ transform: translateY(1px); }

/* Inline action row spacing (for st.columns) */
.btnRowInline [data-testid="stHorizontalBlock"]{
  gap: 10px !important;
}

/* Make link_button look like our dark buttons */
a[data-testid="stLinkButton"]{
  border: 1px solid var(--border) !important;
  background: rgba(255,255,255,0.06) !important;
  color: var(--text) !important;
  border-radius: 12px !important;
  padding: 10px 12px !important;
  height: 44px !important;
  font-weight: 650 !important;
  white-space: nowrap !important;
  font-size: 13px !important;
  text-decoration: none !important;
  display: inline-flex !important;
  align-items: center !important;
  justify-content: center !important;
}
a[data-testid="stLinkButton"]:hover{
  background: rgba(255,255,255,0.10) !important;
}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)

# =========================
# Evidence + highlight helpers (adapted from API logic)
# =========================
_WORD_RE = re.compile(r"[A-Za-z0-9_]+", re.UNICODE)
_SENT_SPLIT = re.compile(r"(?<=[\.!\?])\s+")


def _matched_terms(query_terms: List[str], text: str) -> List[str]:
    if not query_terms or not text:
        return []
    t = text.lower()
    hits = []
    for w in query_terms:
        ww = str(w).lower().strip()
        if len(ww) < 2:
            continue
        if ww in t:
            hits.append(ww)
    # unique preserve order
    seen = set()
    out = []
    for x in hits:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _best_sentence(abstract: str, terms: List[str], max_chars: int = 320) -> str:
    if not abstract:
        return ""
    sents = _SENT_SPLIT.split(abstract)
    if not sents:
        sents = [abstract]

    best = ""
    best_score = -1
    for s in sents:
        s_clean = s.strip()
        if not s_clean:
            continue
        score = 0
        s_low = s_clean.lower()
        for t in terms:
            if t and str(t).lower() in s_low:
                score += 1
        # prefer medium length evidence
        score = score * 10 - abs(len(s_clean) - 220) / 50
        if score > best_score:
            best_score = score
            best = s_clean

    if len(best) > max_chars:
        best = best[: max_chars - 3].rstrip() + "..."
    return best


def _highlight_html(text: str, terms: List[str]) -> str:
    """HTML-escape then bold matched terms with safe word-ish boundary."""
    if not text:
        return ""
    out = html.escape(text)

    uniq = {t.strip() for t in terms if isinstance(t, str) and len(t.strip()) >= 2}
    for t in sorted(uniq, key=len, reverse=True):
        # boundary: not letter/digit/underscore around term
        pattern = re.compile(
            rf"(?<![A-Za-z0-9_])({re.escape(t)})(?![A-Za-z0-9_])",
            flags=re.IGNORECASE,
        )
        out = pattern.sub(r"<b>\1</b>", out)
    return out


def _lookup_full_doc(
    item: Dict[str, Any],
    docs_by_id: Dict[str, Dict[str, Any]],
    docs_by_url: Dict[str, Dict[str, Any]],
    doc_id_by_doc_idx: Dict[int, str],
) -> Dict[str, Any]:
    doc: Optional[Dict[str, Any]] = None

    doc_id = item.get("doc_id")
    if doc_id and str(doc_id) in docs_by_id:
        doc = docs_by_id[str(doc_id)]

    if doc is None and item.get("url"):
        u = str(item.get("url"))
        doc = docs_by_url.get(u)

    if doc is None and item.get("doc_idx") is not None:
        try:
            did = doc_id_by_doc_idx.get(int(item.get("doc_idx")))
            if did:
                doc = docs_by_id.get(did)
        except Exception:
            pass

    return doc or {}


def _attach_explain(
    results: List[Dict[str, Any]],
    highlight_terms: List[str],
    docs_by_id: Dict[str, Dict[str, Any]],
    docs_by_url: Dict[str, Dict[str, Any]],
    doc_id_by_doc_idx: Dict[int, str],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for it in results:
        doc = _lookup_full_doc(it, docs_by_id, docs_by_url, doc_id_by_doc_idx)
        abstract = (doc.get("abstrak") or doc.get("abstract") or it.get("abstrak") or it.get("abstract") or "")
        abstract = str(abstract)

        matched = _matched_terms(highlight_terms, abstract)
        evidence = _best_sentence(abstract, matched)
        html_ev = _highlight_html(evidence, matched)

        it2 = dict(it)
        it2["explain"] = {
            "matched_terms": matched,
            "abstract_html": html_ev,
        }

        # fallback fields (biar UI konsisten)
        if doc.get("judul") and not it2.get("judul"):
            it2["judul"] = doc.get("judul")
        if doc.get("tanggal") and not it2.get("tanggal"):
            it2["tanggal"] = doc.get("tanggal")
        if doc.get("url") and not it2.get("url"):
            it2["url"] = doc.get("url")
        if doc.get("source") and not it2.get("source"):
            it2["source"] = doc.get("source")

        out.append(it2)
    return out


# =========================
# Cache assets (BM25 index, docs, profiles, stopwords)
# =========================
@st.cache_resource(show_spinner=False)
def load_assets() -> Dict[str, Any]:
    # try to locate configs relative to current working dir
    # (Streamlit Cloud runs from repo root)
    paths_cfg = "configs/paths.yaml"
    if not Path(paths_cfg).exists():
        # common alt: nested folder
        nested = HERE.parent / "dinus_research_recommendation_system" / "configs" / "paths.yaml"
        if nested.exists():
            paths_cfg = str(nested)

    paths = load_paths(paths_cfg)
    settings = load_settings(paths_cfg)

    # stopwords
    stopwords = load_stopwords(
        paths.stopwords_file,
        use_sastrawi=settings.use_sastrawi_stopwords,
        use_domain=settings.use_domain_stopwords,
    )

    # docs
    docs_processed = read_jsonl(paths.processed_jsonl)
    docs_by_id = {str(d.get("doc_id")): d for d in docs_processed if d.get("doc_id") is not None}
    docs_by_url = {str(d.get("url")): d for d in docs_processed if d.get("url")}

    # detect stemming_mode from processed docs if present
    stem_mode = settings.stemming_mode
    if docs_processed:
        m = docs_processed[0].get("stemming_mode")
        if isinstance(m, str) and m.strip():
            stem_mode = m.strip().lower()

    # bm25
    bm25 = None
    doc_id_by_doc_idx: Dict[int, str] = {}
    if paths.bm25_index_file.exists():
        bm25 = BM25Index.load(paths.bm25_index_file)
        # attach docs for explain
        try:
            bm25.docs_by_id = docs_by_id
        except Exception:
            pass

        # map doc_idx -> doc_id (from docs_meta)
        try:
            for i, meta in enumerate(getattr(bm25, "docs_meta", []) or []):
                did = meta.get("doc_id") if isinstance(meta, dict) else None
                if did is not None:
                    doc_id_by_doc_idx[int(i)] = str(did)
        except Exception:
            doc_id_by_doc_idx = {}

    # supervisor profiles
    sup_profiles = None
    if paths.supervisor_profiles_file.exists():
        sup_profiles = read_json(paths.supervisor_profiles_file)

    return {
        "paths": paths,
        "settings": settings,
        "stopwords": stopwords,
        "stem_mode": stem_mode,
        "docs_by_id": docs_by_id,
        "docs_by_url": docs_by_url,
        "bm25": bm25,
        "doc_id_by_doc_idx": doc_id_by_doc_idx,
        "sup_profiles": sup_profiles,
    }


# =========================
# Utility helpers
# =========================
def wa_share_url(text: str) -> str:
    return "https://wa.me/?text=" + urllib.parse.quote(text)


def safe_text(x: Any) -> str:
    return "" if x is None else str(x)


def highlight_terms(text: str, terms: List[str]) -> str:
    # reuse the safe HTML highlighter used for evidence
    return _highlight_html(text, terms)


def format_citation_apa(item: Dict[str, Any]) -> str:
    authors = (
        item.get("peneliti")
        or item.get("penulis")
        or item.get("authors")
        or item.get("author")
        or "Unknown author"
    )
    year = str(item.get("tanggal", ""))[:4] if item.get("tanggal") else "n.d."
    title = (item.get("judul") or "Untitled").strip()
    source = (item.get("source") or "UDINUS").strip()
    url = (item.get("url") or "").strip()
    return f"{authors} ({year}). {title}. {source}.{(' ' + url) if url else ''}"


def format_citation_ieee(item: Dict[str, Any]) -> str:
    authors = (
        item.get("peneliti")
        or item.get("penulis")
        or item.get("authors")
        or item.get("author")
        or "Unknown author"
    )
    year = str(item.get("tanggal", ""))[:4] if item.get("tanggal") else "n.d."
    title = (item.get("judul") or "Untitled").strip()
    url = (item.get("url") or "").strip()
    online = f", [Online]. Available: {url}" if url else ""
    return f'{authors}, "{title}," {year}{online}.'


def year_of(item: Dict[str, Any]) -> int:
    t = item.get("tanggal")
    if not t:
        return 0
    try:
        return int(str(t)[:4])
    except Exception:
        return 0


def auto_cutoff_by_score(items: List[Dict[str, Any]], max_keep: int = 30) -> Tuple[List[Dict[str, Any]], float]:
    """Auto buang noise: keep hanya yang skornya masih masuk akal."""
    if not items:
        return [], 0.0

    ranked = sorted(items, key=lambda x: float(x.get("score", 0.0)), reverse=True)

    top_window = ranked[: min(len(ranked), max_keep)]
    scores = [float(x.get("score", 0.0)) for x in top_window]

    s_sorted = sorted(scores)
    mid = len(s_sorted) // 2
    median = s_sorted[mid] if len(s_sorted) % 2 else (s_sorted[mid - 1] + s_sorted[mid]) / 2.0
    abs_dev = [abs(s - median) for s in scores]
    ad_sorted = sorted(abs_dev)
    mad = ad_sorted[mid] if len(ad_sorted) % 2 else (ad_sorted[mid - 1] + ad_sorted[mid]) / 2.0

    k_anchor = min(10, len(scores)) - 1
    score_at_k = scores[k_anchor]

    thr = max(score_at_k, median - 0.5 * mad)
    thr = max(thr, scores[min(2, len(scores) - 1)] * 0.60)

    filtered = [it for it in ranked if float(it.get("score", 0.0)) >= thr]
    return filtered[:max_keep], float(thr)


 


# =========================
# Recommendation runners (local)
# =========================
def run_citations_local(query: str, topk: int = 80) -> List[Dict[str, Any]]:
    A = load_assets()
    bm25 = A["bm25"]
    if bm25 is None:
        st.error("BM25 index belum ada. Jalankan pipeline indexing dulu sampai file indexes/bm25/index.pkl kebentuk.")
        return []

    stopwords = A["stopwords"]
    stem_mode = A["stem_mode"]

    # tokens for retrieval
    q_tokens = preprocess_text(query, stopwords, stem_mode=stem_mode)

    # terms for highlight: pakai token query asli (tanpa stemming agresif) biar kayak 'IHSG' kebaca
    raw_terms = [w for w in _WORD_RE.findall(query) if len(w) >= 2]
    raw_terms_lower = []
    for w in raw_terms:
        wl = w.lower()
        if wl in stopwords:
            continue
        raw_terms_lower.append(w)

    # initial retrieve for expansion (lebih banyak dulu)
    initial = bm25_search(bm25, q_tokens, top_k=max(25, topk * 3), include_meta=True)

    # query expansion from top docs (nambah konteks tanpa jadi ngaco)
    try:
        q_tokens2 = expand_query_from_top_docs(
            A["docs_by_id"],
            initial,
            q_tokens,
            max_expand=8,
            top_docs=8,
            idf=getattr(bm25, "idf", None),
        )
    except Exception:
        q_tokens2 = q_tokens

    # final recommend
    results = recommend_citations(
        bm25,
        q_tokens2,
        top_k=topk,
        diversify=True,
        original_query_tokens=q_tokens,
    )

    # explain + highlight
    highlight_terms = []
    for t in (raw_terms_lower + (q_tokens or [])):
        tt = str(t).strip()
        if tt and tt not in highlight_terms:
            highlight_terms.append(tt)

    results = _attach_explain(
        results,
        highlight_terms,
        A["docs_by_id"],
        A["docs_by_url"],
        A["doc_id_by_doc_idx"],
    )

    return results


def run_dosbing_local(query: str, topk: int = 10) -> List[Dict[str, Any]]:
    A = load_assets()
    sup_profiles = A["sup_profiles"]
    if not sup_profiles:
        st.error("Supervisor profiles belum ada. Jalankan pipeline profiling supaya data/processed/profiles/supervisors.json kebentuk.")
        return []

    stopwords = A["stopwords"]
    stem_mode = A["stem_mode"]
    q_tokens = preprocess_text(query, stopwords, stem_mode=stem_mode)

    try:
        return recommend_supervisors(sup_profiles, q_tokens, top_k=topk)
    except Exception as e:
        st.error(f"Gagal rekomendasi dosbing: {e}")
        return []


# =========================
# UI Header
# =========================
st.markdown("<h1>DINUS Research Recommendation</h1>", unsafe_allow_html=True)

c1, c2 = st.columns([6, 1], vertical_alignment="center")
with c1:
    q = st.text_input(
        "Query",
        placeholder="Masukkan ide penelitian Anda...",
        key="query",
        label_visibility="collapsed",
    )
with c2:
    run = st.button("Cari", use_container_width=True)

c3, c4, c5 = st.columns([2.0, 2.0, 3.0], vertical_alignment="center")
with c3:
    show_dosbing = st.checkbox("Dosbing", value=True)
with c4:
    show_sitasi = st.checkbox("Sitasi", value=True)
with c5:
    sort_by = st.selectbox(
        "Urutkan",
        options=["Relevansi", "Tahun terbaru"],
        index=0,
        label_visibility="collapsed",
    )

st.markdown('</div>', unsafe_allow_html=True)


# =========================
# State
# =========================
if "results" not in st.session_state:
    st.session_state["results"] = {"dosbing": [], "sitasi": []}

if "cutoff_info" not in st.session_state:
    st.session_state["cutoff_info"] = {"thr": None}


# =========================
# Search action (local)
# =========================
if run and q.strip():
    with st.spinner("Sistem sedang mencari..."):
        dosbing_res: List[Dict[str, Any]] = []
        sitasi_res: List[Dict[str, Any]] = []

        if show_dosbing:
            dosbing_res = run_dosbing_local(q, topk=10)

        if show_sitasi:
            sitasi_res = run_citations_local(q, topk=80)

        st.session_state["results"] = {"dosbing": dosbing_res, "sitasi": sitasi_res}
        st.session_state["cutoff_info"] = {"thr": None}


dosbing: List[Dict[str, Any]] = st.session_state["results"].get("dosbing", [])
sitasi_raw: List[Dict[str, Any]] = st.session_state["results"].get("sitasi", [])


# =========================
# Sorting Sitasi + Auto cutoff
# =========================
sitasi_sorted = sitasi_raw[:]
if sort_by == "Tahun terbaru":
    sitasi_sorted = sorted(
        sitasi_sorted,
        key=lambda x: (year_of(x), float(x.get("score", 0.0))),
        reverse=True,
    )
else:
    sitasi_sorted = sorted(sitasi_sorted, key=lambda x: float(x.get("score", 0.0)), reverse=True)

sitasi, thr = auto_cutoff_by_score(sitasi_sorted, max_keep=30)
st.session_state["cutoff_info"] = {"thr": thr}


# =========================
# Render Dosbing first
# =========================
if show_dosbing:
    st.markdown(
        f"""
        <div style="display:flex; justify-content:space-between; align-items:baseline; margin-top:18px;">
          <div style="font-size:28px; font-weight:900; margin:0;">Rekomendasi Dosbing</div>
          <div class="small">{len(dosbing)} kandidat</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not dosbing and q.strip():
        st.markdown('<div class="muted">Tidak ada dosbing yang cocok.</div>', unsafe_allow_html=True)

    for item in dosbing:
        dosen = html.escape(item.get("dosen", "(Tanpa nama)"))
        score = float(item.get("score", 0.0))
        sim = item.get("similarity", None)
        pub_count = item.get("pub_count", 0)

        matched = item.get("matched_terms") or []
        samples = item.get("samples") or []

        chips = "".join([f'<span class="pill">{html.escape(str(t))}</span>' for t in matched[:10]])

        sample_html = ""
        if samples:
            lis = []
            for s in samples[:3]:
                j = html.escape(s.get("judul") or s.get("doc_id") or "Publikasi")
                u = s.get("url")
                tg = s.get("tanggal")
                if u:
                    li = f'<li><a href="{html.escape(u)}" target="_blank" rel="noreferrer">{j}</a>'
                else:
                    li = f"<li>{j}"
                if tg:
                    li += f' <span class="small">• {html.escape(str(tg))}</span>'
                li += "</li>"
                lis.append(li)
            sample_html = (
                '<div class="hr"></div>'
                '<div class="small" style="margin-bottom:6px;">Contoh publikasi dosen:</div>'
                f'<ul style="margin:0; padding-left:18px; font-size:13px;">{"".join(lis)}</ul>'
            )

        st.markdown(
            f"""
            <div class="card">
              <div style="display:flex; justify-content:space-between; gap:12px;">
                <div style="font-size:16px; font-weight:900;">{dosen}</div>
                <div class="small">
                  <span class="kbd">score</span> {score:.4f}
                  {"<span class='kbd'>sim</span> " + f"{float(sim):.4f}" if isinstance(sim, (float,int)) else ""}
                </div>
              </div>
              <div class="small" style="margin-top:6px;">Publikasi: {pub_count}</div>

              {"<div style='margin-top:10px;'><div class='small'>Cocok dengan:</div>" + chips + "</div>" if chips else ""}

              {sample_html}
            </div>
            """,
            unsafe_allow_html=True,
        )


# =========================
# Render Sitasi
# =========================
if show_sitasi:
    # Title without Streamlit heading anchors
    st.markdown(
        f"""
        <div style="display:flex; justify-content:space-between; align-items:baseline; margin-top:22px;">
          <div style="font-size:28px; font-weight:900; margin:0;">Rekomendasi Sitasi</div>
          <div class="small">{len(sitasi)} hasil</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not sitasi and q.strip():
        st.markdown('<div class="muted">Tidak ada sitasi yang cocok.</div>', unsafe_allow_html=True)

    for idx, item in enumerate(sitasi):
        judul_raw = item.get("judul") or "(Tanpa judul)"
        judul = html.escape(judul_raw)
        url = (item.get("url") or "").strip()
        score = float(item.get("score", 0.0))
        score2 = item.get("score2", None)
        tanggal = item.get("tanggal", None)
        source = item.get("source", None)

        explain = item.get("explain") or {}
        abs_html = safe_text(explain.get("abstract_html")).strip()
        matched_terms = explain.get("matched_terms") or item.get("matched_terms") or []

        # fallback: kalau evidence belum ada, highlight seluruh abstrak
        if not abs_html:
            raw_abs = item.get("abstrak") or item.get("abstract") or ""
            abs_html = highlight_terms(str(raw_abs), [str(t) for t in (matched_terms or [])])

        chips = "".join([f'<span class="pill">{html.escape(str(t))}</span>' for t in matched_terms[:10]])

        # Share text
        share_text = f"{judul_raw}"
        if tanggal:
            share_text += f" ({str(tanggal)[:4]})"
        if url:
            share_text += f"\n{url}"
        wa_url = wa_share_url(share_text)

        # Citations
        apa = format_citation_apa(item)
        ieee = format_citation_ieee(item)

        title_html = (
            f'<div style="font-size:16px; font-weight:900; line-height:1.25;">'
            + (f'<a href="{html.escape(url)}" target="_blank" rel="noreferrer">{judul}</a>' if url else judul)
            + "</div>"
        )

        meta_html = (
            f'<div class="small" style="margin-top:8px; line-height:1.6;">'
            f'<span class="kbd">score</span> {score:.4f} '
            + (f'<span class="kbd">score2</span> {float(score2):.4f} ' if isinstance(score2, (float,int)) else "")
            + (f"• {html.escape(str(tanggal))} " if tanggal else "")
            + (f"• {html.escape(str(source))} " if source else "")
            + "</div>"
        )

        abstrak_html = (
            '<div class="hr"></div>'
            '<div class="small">Abstrak (evidence):</div>'
            f'<div class="abstract-evidence">{abs_html}</div>'
        )

        if chips:
            abstrak_html += (
                "<div style='margin-top:10px;'>"
                "<div class='small'>Cocok dengan:</div>"
                f"{chips}"
                "</div>"
            )

        # open card (close after action row)
        st.markdown(f'<div class="card">{title_html}{meta_html}{abstrak_html}', unsafe_allow_html=True)

        # Action row: WhatsApp + Copy APA/IEEE (one line)
        st.markdown('<div class="btnRowInline">', unsafe_allow_html=True)
        b1, b2, b3 = st.columns([1.3, 1.3, 1.3], vertical_alignment="center")

        with b1:
            st.link_button("WhatsApp", wa_url, use_container_width=True)

        with b2:
            with st.popover("Copy APA", use_container_width=True):
                st.code(apa, language=None)

        with b3:
            with st.popover("Copy IEEE", use_container_width=True):
                st.code(ieee, language=None)

        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)  # close .card
