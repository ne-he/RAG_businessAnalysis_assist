"""Document loaders — multi-format source ingestion.

Turns a raw filing on disk into clean text, regardless of format:

  * ``.html`` / ``.htm``  -> BeautifulSoup strip (drop script/style, collapse ws)
  * ``.pdf``              -> pypdf text extraction (lazy/optional dependency)
  * ``.txt``             -> read as-is

Each loaded document is paired with a metadata dict read from a sidecar
``<stem>.json`` (written by ``scripts/fetch_edgar.py``) so company / ticker /
fiscal_year / source_url travel with the text into chunking.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

_WS_RE = re.compile(r"[ \t]+")
_BLANKS_RE = re.compile(r"\n{3,}")


@dataclass
class LoadedDoc:
    text: str
    metadata: dict = field(default_factory=dict)


def clean_text(raw: str) -> str:
    """Normalize whitespace without destroying paragraph/line structure."""
    lines = [_WS_RE.sub(" ", ln).strip() for ln in raw.splitlines()]
    text = "\n".join(lines)
    return _BLANKS_RE.sub("\n\n", text).strip()


def load_html(path: Path) -> str:
    from bs4 import BeautifulSoup  # core dep, but import where used

    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    # get_text with newline separators keeps a coarse block structure
    return clean_text(soup.get_text("\n"))


def load_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader  # lazy, optional dep
    except ImportError as exc:  # pragma: no cover - env dependent
        raise RuntimeError(
            "PDF support needs pypdf. Install it: pip install pypdf"
        ) from exc

    reader = PdfReader(str(path))
    pages = [(page.extract_text() or "") for page in reader.pages]
    return clean_text("\n\n".join(pages))


def load_txt(path: Path) -> str:
    return clean_text(path.read_text(encoding="utf-8", errors="ignore"))


_LOADERS = {
    ".html": load_html,
    ".htm": load_html,
    ".pdf": load_pdf,
    ".txt": load_txt,
}


def _sidecar_metadata(path: Path) -> dict:
    meta_file = path.with_suffix(".json")
    if meta_file.exists():
        try:
            return json.loads(meta_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


def load_document(path: Path) -> LoadedDoc:
    loader = _LOADERS.get(path.suffix.lower())
    if loader is None:
        raise ValueError(f"Unsupported document type: {path.suffix} ({path.name})")
    return LoadedDoc(text=loader(path), metadata=_sidecar_metadata(path))


def load_corpus(corpus_root: Path) -> list[LoadedDoc]:
    """Load every supported document under ``corpus_root`` (skips sidecar .json)."""
    docs: list[LoadedDoc] = []
    for path in sorted(corpus_root.rglob("*")):
        if path.suffix.lower() in _LOADERS and path.is_file():
            docs.append(load_document(path))
    return docs
