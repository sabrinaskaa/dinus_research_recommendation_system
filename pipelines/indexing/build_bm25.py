from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

from research_reco.config import load_paths
from research_reco.io_utils import read_jsonl
from research_reco.bm25 import build_bm25_index


def main():
    paths = load_paths()
    docs = read_jsonl(paths.processed_jsonl)
    if not docs:
        raise RuntimeError("docs.jsonl kosong. Jalankan preprocess dulu.")

    tokens = [d.get("tokens", []) for d in docs]
    meta = [{
        "doc_id": d.get("doc_id"),
        "source": d.get("source"),
        "dosen": d.get("dosen"),
        "judul": d.get("judul"),
        "keyword": d.get("keyword"),
        "tanggal": d.get("tanggal"),
        "url": d.get("url"),
        "peneliti": d.get("peneliti"),
    } for d in docs]

    index = build_bm25_index(tokens, meta)
    index.save(paths.bm25_index_file)

    print(f"[index] saved BM25 index to: {paths.bm25_index_file}")
    print(f"[index] docs: {len(meta)} vocab: {len(index.idf)} avgdl: {index.avgdl:.2f}")


if __name__ == "__main__":
    main()
