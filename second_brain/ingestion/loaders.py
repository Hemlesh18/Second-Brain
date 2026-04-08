"""Document loaders for various file formats.

Supported formats
-----------------
- Plain text / Markdown / reStructuredText (.txt, .md, .rst)
- PDF (.pdf)
- HTML bookmarks (.html, .htm)
- Source code files (.py, .js, .ts, ...)
- JSON bookmark exports
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, List

from bs4 import BeautifulSoup

from second_brain.config import (
    BOOKMARK_EXTENSIONS,
    CODE_EXTENSIONS,
    PDF_EXTENSIONS,
    TEXT_EXTENSIONS,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

class Document:
    """A loaded piece of content with metadata."""

    def __init__(self, content: str, metadata: dict = None) -> None:
        self.content = content
        self.metadata: dict = metadata or {}
        # Ensure ingestion timestamp is always present
        self.metadata.setdefault(
            "ingested_at", datetime.now(timezone.utc).isoformat()
        )

    def __repr__(self) -> str:  # pragma: no cover
        src = self.metadata.get("source", "unknown")
        return f"Document(source={src!r}, chars={len(self.content)})"


# ---------------------------------------------------------------------------
# Individual loaders
# ---------------------------------------------------------------------------

def load_text_file(path: Path) -> Document:
    """Load a plain-text, Markdown or RST file."""
    text = path.read_text(encoding="utf-8", errors="replace")
    return Document(
        content=text,
        metadata={
            "source": str(path),
            "type": "note",
            "filename": path.name,
            "modified_at": datetime.fromtimestamp(
                path.stat().st_mtime, tz=timezone.utc
            ).isoformat(),
        },
    )


def load_pdf(path: Path) -> Document:
    """Load a PDF file, extracting text from all pages."""
    try:
        from pypdf import PdfReader  # lazy import – optional dependency
    except ImportError:
        raise ImportError(
            "pypdf is required to load PDF files. Install it with: pip install pypdf"
        )

    reader = PdfReader(str(path))
    pages: List[str] = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(f"[Page {i + 1}]\n{text}")

    return Document(
        content="\n\n".join(pages),
        metadata={
            "source": str(path),
            "type": "pdf",
            "filename": path.name,
            "page_count": len(reader.pages),
            "modified_at": datetime.fromtimestamp(
                path.stat().st_mtime, tz=timezone.utc
            ).isoformat(),
        },
    )


def load_html_bookmarks(path: Path) -> List[Document]:
    """Load bookmarks from a Netscape-format HTML bookmarks file.

    Returns one Document per bookmark link found.
    """
    html = path.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(html, "html.parser")
    docs: List[Document] = []
    for a_tag in soup.find_all("a"):
        href = a_tag.get("href", "").strip()
        title = a_tag.get_text(strip=True)
        if not href:
            continue
        add_date = a_tag.get("add_date", "")
        docs.append(
            Document(
                content=f"Bookmark: {title}\nURL: {href}",
                metadata={
                    "source": str(path),
                    "type": "bookmark",
                    "url": href,
                    "title": title,
                    "add_date": add_date,
                },
            )
        )
    return docs


def load_json_bookmarks(path: Path) -> List[Document]:
    """Load bookmarks from a JSON file.

    Expected format: a list of objects with at least ``url`` and optionally
    ``title`` and ``description`` keys.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        data = [data]

    docs: List[Document] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        url = item.get("url", "")
        title = item.get("title", url)
        description = item.get("description", "")
        content_parts = [f"Bookmark: {title}", f"URL: {url}"]
        if description:
            content_parts.append(f"Description: {description}")
        docs.append(
            Document(
                content="\n".join(content_parts),
                metadata={
                    "source": str(path),
                    "type": "bookmark",
                    "url": url,
                    "title": title,
                },
            )
        )
    return docs


def load_code_file(path: Path) -> Document:
    """Load a source-code file."""
    text = path.read_text(encoding="utf-8", errors="replace")
    lang = path.suffix.lstrip(".")
    return Document(
        content=text,
        metadata={
            "source": str(path),
            "type": "code",
            "language": lang,
            "filename": path.name,
            "modified_at": datetime.fromtimestamp(
                path.stat().st_mtime, tz=timezone.utc
            ).isoformat(),
        },
    )


# ---------------------------------------------------------------------------
# Directory walker
# ---------------------------------------------------------------------------

def load_directory(directory: Path) -> Iterator[Document]:
    """Recursively load all supported files under *directory*.

    Yields Document objects (one per file for text/code/PDF, one per
    bookmark for HTML/JSON bookmark files).
    """
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        try:
            if suffix in TEXT_EXTENSIONS:
                yield load_text_file(path)
            elif suffix in PDF_EXTENSIONS:
                yield load_pdf(path)
            elif suffix in BOOKMARK_EXTENSIONS:
                yield from load_html_bookmarks(path)
            elif suffix == ".json":
                # Attempt JSON bookmarks; skip silently if format doesn't match
                try:
                    docs = load_json_bookmarks(path)
                    if docs:
                        yield from docs
                except (json.JSONDecodeError, KeyError):
                    logger.debug("Skipping non-bookmark JSON file: %s", path)
            elif suffix in CODE_EXTENSIONS:
                yield load_code_file(path)
            else:
                logger.debug("Unsupported file type, skipping: %s", path)
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Failed to load %s: %s", path, exc)


def load_path(path) -> List[Document]:
    """Load documents from a file or directory path.

    Parameters
    ----------
    path:
        A :class:`pathlib.Path` or string pointing to a file or directory.

    Returns
    -------
    list[Document]
        All documents loaded from the given path.
    """
    path = Path(path)
    if path.is_dir():
        return list(load_directory(path))
    suffix = path.suffix.lower()
    if suffix in TEXT_EXTENSIONS:
        return [load_text_file(path)]
    if suffix in PDF_EXTENSIONS:
        return [load_pdf(path)]
    if suffix in BOOKMARK_EXTENSIONS:
        return load_html_bookmarks(path)
    if suffix == ".json":
        return load_json_bookmarks(path)
    if suffix in CODE_EXTENSIONS:
        return [load_code_file(path)]
    raise ValueError(f"Unsupported file type: {path.suffix}")
