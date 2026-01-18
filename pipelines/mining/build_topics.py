# pipelines/mining/build_topics.py
from __future__ import annotations
from pathlib import Path
import json
import sys

sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

from research_reco.config import load_paths
from research_reco.io_utils import read_jsonl

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

def main():
    paths = load_paths()

    docs = read_jsonl(paths.processed_jsonl)
    if not docs:
        raise RuntimeError("docs.jsonl kosong. Jalankan preprocess dulu.")

    texts = [" ".join(d.get("tokens", [])) for d in docs]
    doc_ids = [d.get("doc_id") for d in docs]

    vectorizer = TfidfVectorizer(max_features=20000, min_df=2)
    X = vectorizer.fit_transform(texts)

    n_clusters = 8  # bisa lo ubah
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    labels = km.fit_predict(X)

    terms = vectorizer.get_feature_names_out()
    centers = km.cluster_centers_

    topics = []
    for c in range(n_clusters):
        # top terms cluster
        idxs = centers[c].argsort()[::-1][:15]
        top_terms = [terms[i] for i in idxs]

        members = [doc_ids[i] for i, lab in enumerate(labels) if lab == c][:30]
        topics.append({
            "cluster_id": c,
            "top_terms": top_terms,
            "sample_docs": members,
        })

    out_path = paths.data_processed / "topics.json"
    out_path.write_text(json.dumps(topics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[mining] wrote topics to: {out_path} clusters={n_clusters}")

if __name__ == "__main__":
    main()
