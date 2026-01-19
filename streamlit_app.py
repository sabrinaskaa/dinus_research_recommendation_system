# streamlit_app.py
import os
import json
import html
import re
import urllib.parse
from typing import Any, Dict, List, Tuple

import requests
import streamlit as st


# =========================
# Config
# =========================
st.set_page_config(
    page_title="DINUS Research Recommendation",
    layout="wide",
)

API_BASE = os.getenv("API_BASE_URL", "").strip()
if not API_BASE:
    API_BASE = os.getenv("NEXT_PUBLIC_API_BASE_URL", "").strip()
if not API_BASE:
    API_BASE = "http://localhost:8000"
API_BASE = API_BASE.rstrip("/")


# =========================
# CSS (Dark UI + spacing + tidy buttons)
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

/* ===== Fix space above input (label hidden) ===== */
div[data-testid="stTextInput"] label { display:none !important; }
div[data-testid="stTextInput"] > div { margin-top:0 !important; padding-top:0 !important; }
div[data-testid="stTextInput"] .stTextInput { margin-top:0 !important; }

/* streamlit default spacing sometimes too big */
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

/* Make Streamlit buttons consistent */
div.stButton > button, a[data-testid="stLinkButton"]{
  border: 1px solid var(--border) !important;
  background: rgba(255,255,255,0.06) !important;
  color: var(--text) !important;
  border-radius: 12px !important;
  padding: 10px 12px !important;
  height: 44px !important;
  font-weight: 650 !important;
  white-space: nowrap !important;
}
div.stButton > button{
  font-size: 13px !important;
}
a[data-testid="stLinkButton"]{
  font-size: 13px !important;
}
div.stButton > button:hover, a[data-testid="stLinkButton"]:hover{
  background: rgba(255,255,255,0.10) !important;
}

/* Better checkbox color */
[data-testid="stCheckbox"] label { color: var(--text) !important; }

/* Inline action row spacing */
.btnRowInline [data-testid="stHorizontalBlock"]{
  gap: 10px !important;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# =========================
# Helpers
# =========================
def api_post(path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{API_BASE}{path}"
    try:
        r = requests.post(url, json=payload, timeout=60)
    except requests.RequestException as e:
        raise RuntimeError(
            f"Gagal fetch ke API.\nURL: {url}\n"
            f"Pastikan backend jalan + API_BASE_URL benar.\nDetail: {e}"
        )
    if r.status_code >= 400:
        raise RuntimeError(f"API error {r.status_code}: {r.text[:800]}")
    return r.json()


def wa_share_url(text: str) -> str:
    return "https://wa.me/?text=" + urllib.parse.quote(text)


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


def highlight_terms(text: str, terms: List[str]) -> str:
    """Escape HTML then bold matched terms with safe "word-ish boundary"."""
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


def year_of(item: Dict[str, Any]) -> int:
    t = item.get("tanggal")
    if not t:
        return 0
    try:
        return int(str(t)[:4])
    except Exception:
        return 0


def auto_cutoff_by_score(items: List[Dict[str, Any]], max_keep: int = 30) -> Tuple[List[Dict[str, Any]], float]:
    """
    Keep only relevant results automatically using a robust score threshold.
    Strategy:
    - sort desc by score
    - compute median & MAD on top window
    - threshold = max(score_at_k, median - 0.5*MAD) (clamped)
    - keep items with score >= threshold, but at most max_keep.
    Returns (filtered, threshold).
    """
    if not items:
        return [], 0.0

    ranked = sorted(items, key=lambda x: float(x.get("score", 0.0)), reverse=True)

    # Always keep at least top 3 if exist
    top_window = ranked[: min(len(ranked), max_keep)]
    scores = [float(x.get("score", 0.0)) for x in top_window]

    # robust stats
    s_sorted = sorted(scores)
    mid = len(s_sorted) // 2
    median = s_sorted[mid] if len(s_sorted) % 2 else (s_sorted[mid - 1] + s_sorted[mid]) / 2.0
    abs_dev = [abs(s - median) for s in scores]
    ad_sorted = sorted(abs_dev)
    mad = ad_sorted[mid] if len(ad_sorted) % 2 else (ad_sorted[mid - 1] + ad_sorted[mid]) / 2.0

    # knee-ish anchor from topK default: use score at 10th (or last)
    k_anchor = min(10, len(scores)) - 1
    score_at_k = scores[k_anchor]

    thr = max(score_at_k, median - 0.5 * mad)
    # clamp to avoid insane cutoff if all low
    thr = max(thr, scores[min(2, len(scores) - 1)] * 0.60)

    filtered = [it for it in ranked if float(it.get("score", 0.0)) >= thr]
    return filtered[:max_keep], float(thr)


def safe_text(x: Any) -> str:
    return "" if x is None else str(x)


# =========================
# UI Header
# =========================
st.markdown("<h1>DINUS Research Recommendation</h1>", unsafe_allow_html=True)

# Search box (no weird top spacing)
with st.container():
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

    st.markdown("</div>", unsafe_allow_html=True)


# =========================
# State
# =========================
if "results" not in st.session_state:
    st.session_state["results"] = {"dosbing": [], "sitasi": []}

if "cutoff_info" not in st.session_state:
    st.session_state["cutoff_info"] = {"thr": None}


# =========================
# Search action
# =========================
if run and q.strip():
    with st.spinner("Sistem sedang mencari..."):
        dosbing_res = {"results": []}
        sitasi_res = {"results": []}

        if show_dosbing:
            dosbing_res = api_post("/recommend/supervisors", {"query": q, "topk": 10})

        if show_sitasi:
            # request more, but UI will auto-cutoff
            sitasi_res = api_post("/recommend/citations", {"query": q, "topk": 80})

        st.session_state["results"] = {
            "dosbing": dosbing_res.get("results", []),
            "sitasi": sitasi_res.get("results", []),
        }
        st.session_state["cutoff_info"] = {"thr": None}


dosbing: List[Dict[str, Any]] = st.session_state["results"].get("dosbing", [])
sitasi_raw: List[Dict[str, Any]] = st.session_state["results"].get("sitasi", [])


# =========================
# Sorting Sitasi
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

# Auto cutoff (avoid showing unrelated noise)
sitasi, thr = auto_cutoff_by_score(sitasi_sorted, max_keep=30)
st.session_state["cutoff_info"] = {"thr": thr}


# =========================
# Render Dosbing (first)
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

        if not abs_html:
            raw_abs = item.get("abstrak") or item.get("abstract") or ""
            abs_html = highlight_terms(raw_abs, matched_terms)

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

        st.markdown(f'<div class="card">{title_html}{meta_html}{abstrak_html}', unsafe_allow_html=True)

        # Action row: Buka sumber, WhatsApp, Copy APA, Copy IEEE (one line)
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

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)  # close .card
