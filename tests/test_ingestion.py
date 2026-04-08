"""Tests for document loaders."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from second_brain.ingestion.loaders import (
    Document,
    load_code_file,
    load_directory,
    load_html_bookmarks,
    load_json_bookmarks,
    load_path,
    load_text_file,
)


# ---------------------------------------------------------------------------
# Document model
# ---------------------------------------------------------------------------

class TestDocument:
    def test_metadata_defaults(self):
        doc = Document(content="hello")
        assert "ingested_at" in doc.metadata

    def test_custom_metadata_preserved(self):
        doc = Document(content="hi", metadata={"source": "/foo.txt"})
        assert doc.metadata["source"] == "/foo.txt"
        assert "ingested_at" in doc.metadata

    def test_existing_ingested_at_not_overwritten(self):
        doc = Document(content="hi", metadata={"ingested_at": "2024-01-01T00:00:00+00:00"})
        assert doc.metadata["ingested_at"] == "2024-01-01T00:00:00+00:00"


# ---------------------------------------------------------------------------
# Text / Markdown loader
# ---------------------------------------------------------------------------

class TestLoadTextFile:
    def test_loads_content(self, tmp_path):
        p = tmp_path / "note.md"
        p.write_text("# My Note\n\nHello world.")
        doc = load_text_file(p)
        assert "Hello world" in doc.content

    def test_metadata_type_is_note(self, tmp_path):
        p = tmp_path / "note.txt"
        p.write_text("Some text")
        doc = load_text_file(p)
        assert doc.metadata["type"] == "note"
        assert doc.metadata["filename"] == "note.txt"

    def test_source_is_absolute_path(self, tmp_path):
        p = tmp_path / "test.md"
        p.write_text("content")
        doc = load_text_file(p)
        assert doc.metadata["source"] == str(p)


# ---------------------------------------------------------------------------
# Code loader
# ---------------------------------------------------------------------------

class TestLoadCodeFile:
    def test_loads_python(self, tmp_path):
        p = tmp_path / "script.py"
        p.write_text("def hello(): return 'world'")
        doc = load_code_file(p)
        assert "hello" in doc.content
        assert doc.metadata["type"] == "code"
        assert doc.metadata["language"] == "py"

    def test_loads_javascript(self, tmp_path):
        p = tmp_path / "app.js"
        p.write_text("console.log('hi')")
        doc = load_code_file(p)
        assert doc.metadata["language"] == "js"


# ---------------------------------------------------------------------------
# HTML bookmarks loader
# ---------------------------------------------------------------------------

class TestLoadHtmlBookmarks:
    def test_loads_netscape_bookmarks(self, tmp_path):
        html = textwrap.dedent("""\
            <!DOCTYPE NETSCAPE-Bookmark-file-1>
            <HTML><HEAD><TITLE>Bookmarks</TITLE></HEAD>
            <BODY>
            <DT><A HREF="https://example.com" ADD_DATE="1700000000">Example</A>
            <DT><A HREF="https://python.org">Python</A>
            </BODY></HTML>
        """)
        p = tmp_path / "bookmarks.html"
        p.write_text(html)
        docs = load_html_bookmarks(p)
        assert len(docs) == 2
        assert docs[0].metadata["url"] == "https://example.com"
        assert docs[0].metadata["type"] == "bookmark"
        assert "Example" in docs[0].content

    def test_skips_entries_without_href(self, tmp_path):
        html = "<HTML><BODY><DT><A>No href</A></BODY></HTML>"
        p = tmp_path / "bookmarks.html"
        p.write_text(html)
        docs = load_html_bookmarks(p)
        assert docs == []


# ---------------------------------------------------------------------------
# JSON bookmarks loader
# ---------------------------------------------------------------------------

class TestLoadJsonBookmarks:
    def test_loads_list(self, tmp_path):
        data = [
            {"url": "https://example.com", "title": "Example", "description": "A site"},
            {"url": "https://python.org", "title": "Python"},
        ]
        p = tmp_path / "bookmarks.json"
        p.write_text(json.dumps(data))
        docs = load_json_bookmarks(p)
        assert len(docs) == 2
        assert "Description: A site" in docs[0].content
        assert docs[1].metadata["url"] == "https://python.org"

    def test_single_object_wrapped(self, tmp_path):
        data = {"url": "https://example.com", "title": "Single"}
        p = tmp_path / "single.json"
        p.write_text(json.dumps(data))
        docs = load_json_bookmarks(p)
        assert len(docs) == 1


# ---------------------------------------------------------------------------
# load_directory
# ---------------------------------------------------------------------------

class TestLoadDirectory:
    def test_loads_mixed_files(self, tmp_path):
        (tmp_path / "note.md").write_text("# Hello")
        (tmp_path / "script.py").write_text("print('hi')")
        docs = list(load_directory(tmp_path))
        types = {d.metadata["type"] for d in docs}
        assert "note" in types
        assert "code" in types

    def test_raises_on_missing_directory(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            list(load_directory(tmp_path / "nonexistent"))

    def test_recurses_into_subdirectories(self, tmp_path):
        subdir = tmp_path / "sub"
        subdir.mkdir()
        (subdir / "deep.md").write_text("deep content")
        docs = list(load_directory(tmp_path))
        sources = [d.metadata["source"] for d in docs]
        assert any("deep.md" in s for s in sources)

    def test_unsupported_files_are_skipped(self, tmp_path):
        (tmp_path / "image.png").write_bytes(b"\x89PNG\r\n")
        docs = list(load_directory(tmp_path))
        assert docs == []


# ---------------------------------------------------------------------------
# load_path dispatch
# ---------------------------------------------------------------------------

class TestLoadPath:
    def test_dispatches_to_text_loader(self, tmp_path):
        p = tmp_path / "note.txt"
        p.write_text("hello")
        docs = load_path(p)
        assert len(docs) == 1
        assert docs[0].metadata["type"] == "note"

    def test_dispatches_to_code_loader(self, tmp_path):
        p = tmp_path / "main.py"
        p.write_text("pass")
        docs = load_path(p)
        assert docs[0].metadata["type"] == "code"

    def test_dispatches_to_directory(self, tmp_path):
        (tmp_path / "a.md").write_text("A")
        (tmp_path / "b.md").write_text("B")
        docs = load_path(tmp_path)
        assert len(docs) == 2

    def test_raises_on_unsupported_extension(self, tmp_path):
        p = tmp_path / "image.xyz"
        p.write_bytes(b"data")
        with pytest.raises(ValueError, match="Unsupported"):
            load_path(p)
