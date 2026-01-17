from pathlib import Path
import sys
import argparse

# allow running without installing package
sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

from research_reco.config import load_paths, load_settings
from research_reco.io_utils import read_jsonl, write_jsonl
from research_reco.text_utils import load_stopwords, preprocess_text, build_text_for_index


def main():
    ap = argparse.ArgumentParser(description="Build processed docs (tokens) from parsed_docs.jsonl")
    ap.add_argument(
        "--stem",
        choices=["off", "full", "selective"],
        help="Override stemming mode (off|full|selective). If omitted, uses configs/settings.yaml",
    )
    args = ap.parse_args()

    paths = load_paths()
    settings = load_settings()

    # Resolve stemming mode
    stem_mode = settings.stemming_mode  # already "off" if use_stemming false in settings
    if args.stem:
        stem_mode = args.stem.lower().strip()
    if stem_mode not in {"off", "full", "selective"}:
        stem_mode = "off"

    # Load stopwords based on ON/OFF settings
    stopwords = load_stopwords(
        paths.stopwords_file,
        use_sastrawi=settings.use_sastrawi_stopwords,
        use_domain=settings.use_domain_stopwords,
    )

    rows = read_jsonl(paths.parsed_jsonl)
    if not rows:
        raise RuntimeError(
            "parsed_docs.jsonl kosong. Jalankan pipelines/ingest/parse_txt.py dulu."
        )

    out = []
    for r in rows:
        text_for_index = build_text_for_index(
            r.get("judul"), r.get("keyword"), r.get("abstrak")
        )

        tokens = preprocess_text(
            text_for_index,
            stopwords=stopwords,
            stem_mode=stem_mode,
        )

        out.append(
            {
                "doc_id": r.get("doc_id"),
                "source": r.get("source"),
                "dosen": r.get("dosen"),
                "url": r.get("url"),
                "tanggal": r.get("tanggal"),
                "judul": r.get("judul"),
                "keyword": r.get("keyword"),
                "abstrak": r.get("abstrak"),
                "peneliti": r.get("peneliti"),
                "text_for_index": text_for_index,
                "tokens": tokens,
                # Save preprocessing config so API stays consistent
                "stemming_mode": stem_mode,
                "use_sastrawi_stopwords": bool(settings.use_sastrawi_stopwords),
                "use_domain_stopwords": bool(settings.use_domain_stopwords),
            }
        )

    write_jsonl(paths.processed_jsonl, out)
    print(
        f"[preprocess] wrote: {paths.processed_jsonl} docs={len(out)} "
        f"stemming_mode={stem_mode} "
        f"sastrawi_stopwords={settings.use_sastrawi_stopwords} "
        f"domain_stopwords={settings.use_domain_stopwords}"
    )


if __name__ == "__main__":
    main()
