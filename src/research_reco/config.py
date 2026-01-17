from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import yaml

@dataclass(frozen=True)
class AppPaths:
    data_raw: Path
    data_interim: Path
    data_processed: Path
    profiles_dir: Path

    index_dir: Path
    bm25_index_file: Path

    parsed_jsonl: Path
    processed_jsonl: Path

    supervisor_profiles_file: Path
    stopwords_file: Path

    settings_file: Path


def load_paths(config_file: str = "configs/paths.yaml") -> AppPaths:
    cfg = yaml.safe_load(Path(config_file).read_text(encoding="utf-8"))

    def p(k: str) -> Path:
        return Path(cfg[k])

    paths = AppPaths(
        data_raw=p("data_raw"),
        data_interim=p("data_interim"),
        data_processed=p("data_processed"),
        profiles_dir=p("profiles_dir"),
        index_dir=p("index_dir"),
        bm25_index_file=p("bm25_index_file"),
        parsed_jsonl=p("parsed_jsonl"),
        processed_jsonl=p("processed_jsonl"),
        supervisor_profiles_file=p("supervisor_profiles_file"),
        stopwords_file=p("stopwords_file"),
        settings_file=p("settings_file"),
    )

    # ensure dirs exist
    paths.data_interim.mkdir(parents=True, exist_ok=True)
    paths.data_processed.mkdir(parents=True, exist_ok=True)
    paths.profiles_dir.mkdir(parents=True, exist_ok=True)
    paths.index_dir.mkdir(parents=True, exist_ok=True)

    return paths


@dataclass(frozen=True)
class AppSettings:
    use_sastrawi_stopwords: bool
    use_domain_stopwords: bool
    use_stemming: bool
    stemming_mode: str  # "off" | "full" | "selective"


def load_settings(config_file: str = "configs/paths.yaml") -> AppSettings:
    cfg = yaml.safe_load(Path(config_file).read_text(encoding="utf-8"))
    settings_path = Path(cfg.get("settings_file", "configs/settings.yaml"))
    s = yaml.safe_load(settings_path.read_text(encoding="utf-8"))

    mode = str(s.get("stemming_mode", "selective")).lower().strip()
    if mode not in {"off", "full", "selective"}:
        mode = "selective"

    use_stemming = bool(s.get("use_stemming", False))
    if not use_stemming:
        mode = "off"

    return AppSettings(
        use_sastrawi_stopwords=bool(s.get("use_sastrawi_stopwords", True)),
        use_domain_stopwords=bool(s.get("use_domain_stopwords", True)),
        use_stemming=use_stemming,
        stemming_mode=mode,
    )
