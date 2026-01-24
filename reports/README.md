# UDINUS Research & Supervisor Recommendation System

Sistem rekomendasi untuk membantu mahasiswa UDINUS pada tahap awal penelitian (TA/Skripsi/Lomba) dengan dua fitur utama:

1. Rekomendasi sitasi/paper berbasis _Information Retrieval_ (BM25).
2. Rekomendasi dosen pembimbing berbasis profil publikasi (TF-IDF + Cosine Similarity).

Aplikasi disajikan melalui antarmuka **Streamlit** dan menjalankan inferensi secara lokal dengan memuat artefak hasil pipeline (dokumen terolah, indeks BM25, profil dosen).

## Struktur Proyek

- `data/`
  - `raw/` : data mentah (txt)
  - `interim/` : hasil parsing awal (JSONL)
  - `processed/` : dokumen siap indeks + profil dosen
- `pipelines/`
  - `ingest/` : parsing data raw → interim
  - `preprocess/` : preprocessing interim → processed
  - `indexing/` : build BM25 index
  - `profiling/` : build supervisor profiles
  - `eval/` : evaluasi retrieval
- `src/research_reco/` : modul inti (parser, preprocess, BM25, profil dosen)
- `streamlit_app.py` : aplikasi Streamlit (UI + inferensi lokal)
- `configs/` : konfigurasi path & settings
- `requirements.txt` : dependensi

## Instalasi

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate
```

## Instal Dependensi

```bash
pip install -r requirements.txt
```

## Parsing data raw → interim

```bash
python pipelines/ingest/parse_txt.py
```

## Preprocessing interim → processed

```bash
python pipelines/preprocess/build_processed.py
```

## Build indeks BM25

```bash
python pipelines/indexing/build_bm25.py
```

## Build profil dosen pembimbing

```bash
python pipelines/profiling/build_supervisor_profiles.py
```

## Run Application

```bash
streamlit run streamlit_app.py
```

## Link Deploy

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://dinusresearchrecommendationsystem.streamlit.app/)
