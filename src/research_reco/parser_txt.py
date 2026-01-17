from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, Tuple, List
import re
from .models import ParsedDoc

KNOWN_KEYS = {
    "url": "url",
    "tanggal": "tanggal",
    "judul": "judul",
    "keyword": "keyword",
    "abstrak": "abstrak",
    "peneliti": "peneliti",
    "authors": "peneliti",
    "author": "peneliti",
}

KEY_LINE_RE = re.compile(r"^\s*([A-Za-z_]+)\s*:\s*(.*)\s*$")


def infer_source_and_dosen(file_path: Path) -> Tuple[str, Optional[str]]:
    parts = list(file_path.parts)  # parts are strings already

    if "Publikasi Dosbing" in parts:
        idx = parts.index("Publikasi Dosbing")
        dosen = parts[idx + 1] if idx + 1 < len(parts) else None
        return "dosbing", dosen

    if "Publikasi UDINUS" in parts:
        return "udinus", None

    return "unknown", None


def parse_txt_file(path: Path) -> ParsedDoc:
    source, dosen = infer_source_and_dosen(path)
    doc_id = path.stem

    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()

    fields: Dict[str, str] = {}
    misc_lines: List[str] = []

    current_key: Optional[str] = None
    buffer: List[str] = []

    def flush():
        nonlocal current_key, buffer
        if current_key is not None:
            val = "\n".join(buffer).strip()
            if val:
                if current_key in fields:
                    fields[current_key] = (fields[current_key] + "\n" + val).strip()
                else:
                    fields[current_key] = val
        current_key = None
        buffer = []

    for raw in lines:
        line = raw.strip()

        # bare url line
        if (line.startswith("http://") or line.startswith("https://")) and "url" not in fields:
            fields["url"] = line
            continue

        m = KEY_LINE_RE.match(raw)
        if m:
            k_raw = m.group(1).strip().lower()
            v = m.group(2).strip()
            mapped = KNOWN_KEYS.get(k_raw)

            if mapped:
                flush()
                current_key = mapped
                buffer = [v] if v else []
                continue

        if current_key is None:
            if line:
                misc_lines.append(raw)
        else:
            buffer.append(raw)

    flush()

    return ParsedDoc(
        doc_id=doc_id,
        source=source,
        dosen=dosen,
        path=str(path),
        url=fields.get("url"),
        tanggal=fields.get("tanggal"),
        judul=fields.get("judul"),
        keyword=fields.get("keyword"),
        abstrak=fields.get("abstrak"),
        peneliti=fields.get("peneliti"),
        misc="\n".join(misc_lines).strip() if misc_lines else None,
    )


def collect_txt_files(root: Path) -> List[Path]:
    return [p for p in root.rglob("*.txt") if p.is_file()]
