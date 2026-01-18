from __future__ import annotations
from pathlib import Path
import json
import sys

sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

from research_reco.config import load_paths, load_settings
from research_reco.io_utils import read_jsonl
from research_reco.bm25 import BM25Index, bm25_search
from research_reco.text_utils import load_stopwords, preprocess_text

def precision_at_k(ranked, relevant, k):
    if k <= 0: return 0.0
    hit = 0
    for d in ranked[:k]:
        if d in relevant:
            hit += 1
    return hit / k

def mrr_at_k(ranked, relevant, k):
    for i, d in enumerate(ranked[:k], start=1):
        if d in relevant:
            return 1.0 / i
    return 0.0

def ndcg_at_k(ranked, rel_map, k):
    # rel_map: doc_id -> relevance (0..3)
    def dcg(items):
        s = 0.0
        for i, d in enumerate(items, start=1):
            rel = float(rel_map.get(d, 0.0))
            if rel > 0:
                s += (2**rel - 1) / (math.log2(i + 1))
        return s

    import math
    dcg_val = dcg(ranked[:k])
    ideal = sorted(rel_map.items(), key=lambda x: x[1], reverse=True)
    ideal_ranked = [d for d, _ in ideal][:k]
    idcg = dcg(ideal_ranked)
    return 0.0 if idcg == 0 else dcg_val / idcg

def main():
    paths = load_paths()
    settings = load_settings()

    eval_file = Path("eval/queries.jsonl")
    if not eval_file.exists():
        raise RuntimeError("Buat eval/queries.jsonl dulu. Lihat template di eval/queries.example.jsonl")

    index = BM25Index.load(paths.bm25_index_file)

    stopwords = load_stopwords(
        paths.stopwords_file,
        use_sastrawi=settings.use_sastrawi_stopwords,
        use_domain=settings.use_domain_stopwords,
    )

    rows = [json.loads(x) for x in eval_file.read_text(encoding="utf-8").splitlines() if x.strip()]
    ks = [5, 10]

    agg = {f"P@{k}": 0.0 for k in ks}
    agg.update({f"MRR@{k}": 0.0 for k in ks})
    agg.update({f"nDCG@{k}": 0.0 for k in ks})
    n = 0

    for r in rows:
        q = r["query"]
        relevant = set(r.get("relevant_doc_ids", []))
        rel_map = r.get("relevance", {})  # optional graded
        stem_mode = r.get("stemming_mode", settings.stemming_mode)

        q_tokens = preprocess_text(q, stopwords=stopwords, stem_mode=stem_mode)
        res = bm25_search(index, q_tokens, top_k=50)
        ranked_doc_ids = [x["doc_id"] for x in res]

        for k in ks:
            agg[f"P@{k}"] += precision_at_k(ranked_doc_ids, relevant, k) if relevant else 0.0
            agg[f"MRR@{k}"] += mrr_at_k(ranked_doc_ids, relevant, k) if relevant else 0.0
            agg[f"nDCG@{k}"] += ndcg_at_k(ranked_doc_ids, rel_map, k) if rel_map else 0.0

        n += 1

    if n == 0:
        raise RuntimeError("eval/queries.jsonl kosong.")

    print("=== EVAL RESULTS ===")
    for k in ks:
        print(f"P@{k}:   {agg[f'P@{k}']/n:.4f}")
        print(f"MRR@{k}: {agg[f'MRR@{k}']/n:.4f}")
        print(f"nDCG@{k}:{agg[f'nDCG@{k}']/n:.4f}")
        print("")

if __name__ == "__main__":
    main()
