from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

from research_reco.config import load_paths
from research_reco.io_utils import read_jsonl, write_json
from research_reco.supervisor_profiles import build_supervisor_profiles


def main():
    paths = load_paths()
    docs = read_jsonl(paths.processed_jsonl)
    if not docs:
        raise RuntimeError("docs.jsonl kosong. Jalankan preprocess dulu.")

    profiles = build_supervisor_profiles(docs)
    write_json(paths.supervisor_profiles_file, profiles)

    print(f"[profiling] saved supervisor profiles to: {paths.supervisor_profiles_file}")
    print(f"[profiling] dosen count: {len(profiles.get('profiles', {}))}")


if __name__ == "__main__":
    main()
