from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

from research_reco.config import load_paths
from research_reco.parser_txt import collect_txt_files, parse_txt_file
from research_reco.io_utils import write_jsonl


def main():
    paths = load_paths()
    raw_root = paths.data_raw

    files = collect_txt_files(raw_root)
    print(f"[ingest] found {len(files)} txt files under {raw_root}")

    parsed = []
    for fp in files:
        doc = parse_txt_file(fp)
        parsed.append(doc.to_dict())

    write_jsonl(paths.parsed_jsonl, parsed)
    print(f"[ingest] wrote: {paths.parsed_jsonl}")


if __name__ == "__main__":
    main()
