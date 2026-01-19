from __future__ import annotations
import re
from pathlib import Path
from typing import List, Set, Optional

try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
    _HAS_SASTRAWI = True
except ModuleNotFoundError:  # optional dependency
    StemmerFactory = None  # type: ignore
    StopWordRemoverFactory = None  # type: ignore
    _HAS_SASTRAWI = False

_WORD_RE = re.compile(r"[a-zA-Z0-9_]+", re.UNICODE)

_STEMMER = StemmerFactory().create_stemmer() if _HAS_SASTRAWI else None
_SASTRAWI_STOPWORDS = set(StopWordRemoverFactory().get_stop_words()) if _HAS_SASTRAWI else set()

_TECH_WHITELIST = {
    "ai","ml","nlp","cnn","rnn","lstm","gru","svm","knn","rf","xgboost","bert","gpt",
    "embedding","transformer","token","tokenizer","dataset","benchmark","accuracy",
    "precision","recall","f1","roc","auc","api","sql","mysql","nosql","json","xml",
    "http","https","tcp","udp","gpu","cpu","ram","iot","ui","ux","devops","docker",
    "kubernetes","k8s","linux","windows","android","ios","react","nextjs","node",
    "python","java","golang","rust","c","cpp","csharp","php","javascript","typescript"
}

def load_stopwords(custom_path: Path, use_sastrawi: bool, use_domain: bool) -> Set[str]:
    base: Set[str] = set()
    if use_sastrawi and _HAS_SASTRAWI:
        base |= set(_SASTRAWI_STOPWORDS)

    if use_domain and custom_path.exists():
        lines = [x.strip().lower() for x in custom_path.read_text(encoding="utf-8").splitlines()]
        base |= {x for x in lines if x and not x.startswith("#")}

    return base

def tokenize(text: str) -> List[str]:
    text = text.lower()
    return _WORD_RE.findall(text)

def stem_tokens_full(tokens: List[str]) -> List[str]:
    if _STEMMER is None:
        return list(tokens)
    return [_STEMMER.stem(t) for t in tokens]

def _looks_technical_or_english(token: str) -> bool:
    t = token
    if t in _TECH_WHITELIST:
        return True
    if len(t) <= 3:
        return True
    if "_" in t:
        return True
    has_alpha = any(c.isalpha() for c in t)
    has_digit = any(c.isdigit() for c in t)
    if has_alpha and has_digit:
        return True
    if t.endswith(("ing", "tion", "sion", "ment", "ness", "able", "ible", "ize", "ised", "ized")):
        return True
    return False

def _looks_indonesianish(token: str) -> bool:
    t = token
    if not t.isalpha():
        return False
    if _looks_technical_or_english(t):
        return False

    indo_prefixes = ("meng","meny","men","mem","me","peng","peny","pen","pem","di","ke","se","ber","ter","per")
    indo_suffixes = ("kan","i","an","nya","lah","kah","pun")
    if t.startswith(indo_prefixes) or t.endswith(indo_suffixes):
        return True

    return len(t) >= 5

def selective_stem(tokens: List[str]) -> List[str]:
    out: List[str] = []
    for t in tokens:
        if _looks_indonesianish(t):
            if _STEMMER is None:
                out.append(t)
            else:
                out.append(_STEMMER.stem(t))
        else:
            out.append(t)
    return out

def preprocess_text(
    text: str,
    stopwords: Set[str],
    stem_mode: str = "off",  # "off" | "full" | "selective"
) -> List[str]:
    toks = tokenize(text)
    toks = [t for t in toks if t not in stopwords and len(t) > 1]

    mode = (stem_mode or "off").lower().strip()
    if mode == "full":
        toks = stem_tokens_full(toks)
    elif mode == "selective":
        toks = selective_stem(toks)

    toks = [t for t in toks if t not in stopwords and len(t) > 1]
    return toks

def build_text_for_index(
    judul: Optional[str],
    keyword: Optional[str],
    abstrak: Optional[str],
) -> str:
    parts = []
    if judul: parts.append(judul)
    if keyword: parts.append(keyword)
    if abstrak: parts.append(abstrak)
    return " ".join(parts).strip()

def build_boosted_text_for_index(
    judul: Optional[str],
    keyword: Optional[str],
    abstrak: Optional[str],
    title_boost: int = 2,
    keyword_boost: int = 3,
    abstract_boost: int = 1,
) -> str:
    parts = []
    if judul:
        parts.extend([judul] * max(1, int(title_boost)))
    if keyword:
        parts.extend([keyword] * max(1, int(keyword_boost)))
    if abstrak:
        parts.extend([abstrak] * max(1, int(abstract_boost)))
    return " ".join([p for p in parts if p]).strip()
