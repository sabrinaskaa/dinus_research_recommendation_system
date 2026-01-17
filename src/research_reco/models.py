from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any, List


@dataclass
class ParsedDoc:
    doc_id: str
    source: str              # "dosbing" | "udinus" | "unknown"
    dosen: Optional[str]     # only for dosbing source
    path: str                # original file path

    url: Optional[str] = None
    tanggal: Optional[str] = None
    judul: Optional[str] = None
    keyword: Optional[str] = None
    abstrak: Optional[str] = None
    peneliti: Optional[str] = None

    misc: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "source": self.source,
            "dosen": self.dosen,
            "path": self.path,
            "url": self.url,
            "tanggal": self.tanggal,
            "judul": self.judul,
            "keyword": self.keyword,
            "abstrak": self.abstrak,
            "peneliti": self.peneliti,
            "misc": self.misc,
        }


@dataclass
class ProcessedDoc:
    doc_id: str
    source: str
    dosen: Optional[str]
    url: Optional[str]
    tanggal: Optional[str]
    judul: Optional[str]
    keyword: Optional[str]
    abstrak: Optional[str]
    peneliti: Optional[str]
    text_for_index: str
    tokens: List[str]

    def meta(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "source": self.source,
            "dosen": self.dosen,
            "url": self.url,
            "tanggal": self.tanggal,
            "judul": self.judul,
            "keyword": self.keyword,
            "abstrak": self.abstrak,
            "peneliti": self.peneliti,
        }
