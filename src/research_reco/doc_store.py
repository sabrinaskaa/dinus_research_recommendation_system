from __future__ import annotations
from typing import Dict, Any, List
from pathlib import Path

from .io_utils import read_jsonl


def load_docs_by_id(processed_jsonl: Path) -> Dict[str, Dict[str, Any]]:
    rows: List[Dict[str, Any]] = read_jsonl(processed_jsonl)
    out: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        did = r.get("doc_id")
        if did:
            out[str(did)] = r
    return out


def compact_meta(doc: Dict[str, Any]) -> Dict[str, Any]:
    """
    Meta yang enak buat API/UI (nggak terlalu berat).
    tokens tetap ikut biar bisa keyword suggestion.
    """
    return {
        "doc_id": doc.get("doc_id"),
        "source": doc.get("source"),
        "dosen": doc.get("dosen"),
        "judul": doc.get("judul"),
        "keyword": doc.get("keyword"),
        "tanggal": doc.get("tanggal"),
        "url": doc.get("url"),
        "peneliti": doc.get("peneliti"),
        "abstrak": doc.get("abstrak"),
        "tokens": doc.get("tokens", []),
    }
