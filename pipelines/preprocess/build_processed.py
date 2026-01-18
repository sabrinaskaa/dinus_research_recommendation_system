from __future__ import annotations

from pathlib import Path
import sys
import argparse

# allow running without installing package
sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

from research_reco.config import load_paths, load_settings
from research_reco.io_utils import read_jsonl, write_jsonl
from research_reco.text_utils import load_stopwords, preprocess_text

# Import build_text_for_index always; boosted is optional (fallback if missing)
from research_reco.text_utils import build_text_for_index

try:
    # If you've added boosting function, this import will work
    from research_reco.text_utils import build_boosted_text_for_index  # type: ignore
except Exception:
    build_boosted_text_for_index = None  # type: ignore


def main():
    ap = argparse.ArgumentParser(description="Build processed docs (tokens) from parsed_docs.jsonl")

    ap.add_argument(
        "--stem",
        choices=["off", "full", "selective"],
        help="Override stemming mode (off|full|selective). If omitted, uses configs/settings.yaml",
    )

    ap.add_argument(
        "--boost",
        choices=["off", "on"],
        default="on",
        help="Field boosting for indexing text (judul/keyword heavier). Default=on.",
    )
    ap.add_argument("--title_boost", type=int, default=2, help="Repetition factor for judul (default=2)")
    ap.add_argument("--keyword_boost", type=int, default=3, help="Repetition factor for keyword (default=3)")
    ap.add_argument("--abstract_boost", type=int, default=1, help="Repetition factor for abstrak (default=1)")

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
        raise RuntimeError("parsed_docs.jsonl kosong. Jalankan pipelines/ingest/parse_txt.py dulu.")

    use_boost = (args.boost == "on") and (build_boosted_text_for_index is not None)

    out = []
    for r in rows:
        judul = r.get("judul")
        keyword = r.get("keyword")
        abstrak = r.get("abstrak")

        # Build text_for_index with optional field boosting
        if use_boost:
            text_for_index = build_boosted_text_for_index(  # type: ignore
                judul,
                keyword,
                abstrak,
                title_boost=args.title_boost,
                keyword_boost=args.keyword_boost,
                abstract_boost=args.abstract_boost,
            )
        else:
            text_for_index = build_text_for_index(judul, keyword, abstrak)

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
                "judul": judul,
                "keyword": keyword,
                "abstrak": abstrak,
                "peneliti": r.get("peneliti"),
                "text_for_index": text_for_index,
                "tokens": tokens,

                # Save preprocessing config so API stays consistent
                "stemming_mode": stem_mode,
                "use_sastrawi_stopwords": bool(settings.use_sastrawi_stopwords),
                "use_domain_stopwords": bool(settings.use_domain_stopwords),

                # Save boosting config (so you can prove it in evaluation/report)
                "field_boosting": {
                    "enabled": bool(use_boost),
                    "title_boost": int(args.title_boost) if use_boost else 1,
                    "keyword_boost": int(args.keyword_boost) if use_boost else 1,
                    "abstract_boost": int(args.abstract_boost) if use_boost else 1,
                },
            }
        )

    write_jsonl(paths.processed_jsonl, out)
    print(
        f"[preprocess] wrote: {paths.processed_jsonl} docs={len(out)} "
        f"stemming_mode={stem_mode} "
        f"sastrawi_stopwords={settings.use_sastrawi_stopwords} "
        f"domain_stopwords={settings.use_domain_stopwords} "
        f"field_boosting={'on' if use_boost else 'off'}"
    )


if __name__ == "__main__":
    main()
