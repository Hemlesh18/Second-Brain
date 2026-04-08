from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
from typing import Iterable

from bs4 import BeautifulSoup
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document


TEXT_EXTENSIONS = {
    ".md",
    ".txt",
    ".rst",
    ".org",
    ".html",
    ".htm",
    ".csv",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".java",
    ".go",
    ".rs",
    ".c",
    ".cpp",
    ".h",
    ".hpp",
    ".sh",
}


def _base_metadata(path: Path, source_type: str) -> dict:
    stat = path.stat()
    return {
        "source": str(path),
        "source_type": source_type,
        "filename": path.name,
        "extension": path.suffix.lower(),
        "ingested_at": datetime.now(timezone.utc).isoformat(),
        "ingested_ts": datetime.now(timezone.utc).timestamp(),
        "source_mtime": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        "source_mtime_ts": float(stat.st_mtime),
    }


def _load_text(path: Path) -> list[Document]:
    loader = TextLoader(str(path), autodetect_encoding=True)
    docs = loader.load()
    meta = _base_metadata(path, "text")
    for d in docs:
        d.metadata.update(meta)
    return docs


def _load_pdf(path: Path) -> list[Document]:
    loader = PyPDFLoader(str(path))
    docs = loader.load()
    for d in docs:
        base = _base_metadata(path, "pdf")
        base["page"] = d.metadata.get("page")
        d.metadata.update(base)
    return docs


def _parse_bookmark_html(path: Path) -> list[Document]:
    html = path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "html.parser")
    docs: list[Document] = []
    base = _base_metadata(path, "bookmarks")

    for a in soup.find_all("a"):
        href = a.get("href", "")
        title = a.get_text(" ", strip=True)
        if not href:
            continue
        content = f"Bookmark: {title}\nURL: {href}"
        doc = Document(page_content=content, metadata={**base, "url": href, "title": title})
        docs.append(doc)

    return docs


def _parse_bookmark_json(path: Path) -> list[Document]:
    raw = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    docs: list[Document] = []
    base = _base_metadata(path, "bookmarks")

    def walk(node: object) -> Iterable[dict]:
        if isinstance(node, dict):
            if "url" in node:
                yield node
            for value in node.values():
                yield from walk(value)
        elif isinstance(node, list):
            for item in node:
                yield from walk(item)

    for item in walk(raw):
        href = str(item.get("url", "")).strip()
        title = str(item.get("name", "") or item.get("title", "")).strip()
        if not href:
            continue
        content = f"Bookmark: {title}\nURL: {href}"
        doc = Document(page_content=content, metadata={**base, "url": href, "title": title})
        docs.append(doc)

    return docs


def _is_bookmark_file(path: Path) -> bool:
    name = path.name.lower()
    return "bookmark" in name and path.suffix.lower() in {".html", ".htm", ".json"}


def load_documents(paths: list[Path]) -> list[Document]:
    documents: list[Document] = []
    for path in paths:
        if not path.exists() or path.is_dir():
            continue

        suffix = path.suffix.lower()
        if suffix == ".pdf":
            documents.extend(_load_pdf(path))
            continue

        if _is_bookmark_file(path):
            if suffix in {".html", ".htm"}:
                documents.extend(_parse_bookmark_html(path))
            elif suffix == ".json":
                documents.extend(_parse_bookmark_json(path))
            continue

        if suffix in TEXT_EXTENSIONS:
            documents.extend(_load_text(path))

    return documents
