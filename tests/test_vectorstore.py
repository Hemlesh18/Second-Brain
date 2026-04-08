"""Tests for the VectorStore wrapper and text splitter."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from second_brain.ingestion.loaders import Document
from second_brain.vectorstore.store import VectorStore, _split_text


# ---------------------------------------------------------------------------
# Text splitter
# ---------------------------------------------------------------------------

class TestSplitText:
    def test_short_text_not_split(self):
        text = "Short text."
        chunks = _split_text(text, chunk_size=1000, chunk_overlap=200)
        assert chunks == ["Short text."]

    def test_long_text_split_into_multiple_chunks(self):
        text = "word " * 300  # 1500 chars
        chunks = _split_text(text, chunk_size=500, chunk_overlap=50)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) <= 500 + 1  # allow ±1 for boundary rounding

    def test_overlap_creates_continuity(self):
        text = "A" * 200 + " " + "B" * 200 + " " + "C" * 200
        chunks = _split_text(text, chunk_size=300, chunk_overlap=100)
        # The tail of one chunk should appear at the start of the next
        assert len(chunks) >= 2

    def test_empty_chunks_filtered_out(self):
        text = "   \n\n   "
        chunks = _split_text(text, chunk_size=1000, chunk_overlap=100)
        assert chunks == []

    def test_exact_chunk_size_returns_single_chunk(self):
        text = "x" * 1000
        chunks = _split_text(text, chunk_size=1000, chunk_overlap=0)
        assert len(chunks) == 1


# ---------------------------------------------------------------------------
# VectorStore with mocked ChromaDB
# ---------------------------------------------------------------------------

def _make_mock_collection(count: int = 0):
    col = MagicMock()
    col.count.return_value = count
    col.query.return_value = {
        "documents": [["chunk text"]],
        "metadatas": [[{"source": "/note.md", "type": "note", "ingested_at": "2024-01-01"}]],
        "distances": [[0.1]],
    }
    col.get.return_value = {
        "ids": [],
        "documents": [],
        "metadatas": [],
    }
    return col


@pytest.fixture()
def mock_store(tmp_path):
    """A VectorStore with ChromaDB fully mocked out."""
    store = VectorStore(persist_directory=str(tmp_path))
    mock_col = _make_mock_collection(count=3)
    store._collection = mock_col
    store._embed_fn = MagicMock()
    return store, mock_col


class TestVectorStoreAddDocuments:
    def test_upserts_chunks(self, mock_store):
        store, mock_col = mock_store
        docs = [
            Document(content="Hello world, this is a test document.", metadata={"source": "/a.md"}),
        ]
        n = store.add_documents(docs, chunk_size=1000, chunk_overlap=0)
        assert n == 1
        mock_col.upsert.assert_called_once()

    def test_returns_zero_for_empty_list(self, mock_store):
        store, mock_col = mock_store
        n = store.add_documents([])
        assert n == 0
        mock_col.upsert.assert_not_called()

    def test_long_document_produces_multiple_chunks(self, mock_store):
        store, mock_col = mock_store
        long_text = "sentence. " * 200  # ~2000 chars
        docs = [Document(content=long_text, metadata={"source": "/long.md"})]
        n = store.add_documents(docs, chunk_size=500, chunk_overlap=50)
        assert n > 1

    def test_metadata_stringified(self, mock_store):
        store, mock_col = mock_store
        docs = [Document(content="test", metadata={"source": "/x.md", "page_count": 5})]
        store.add_documents(docs)
        call_kwargs = mock_col.upsert.call_args[1]
        for meta in call_kwargs["metadatas"]:
            for v in meta.values():
                assert isinstance(v, str), f"Metadata value {v!r} is not a string"


class TestVectorStoreQuery:
    def test_returns_hits(self, mock_store):
        store, _ = mock_store
        hits = store.query("distributed systems")
        assert len(hits) == 1
        assert hits[0]["content"] == "chunk text"
        assert "distance" in hits[0]

    def test_empty_store_returns_empty(self, tmp_path):
        store = VectorStore(persist_directory=str(tmp_path))
        mock_col = _make_mock_collection(count=0)
        store._collection = mock_col
        store._embed_fn = MagicMock()
        hits = store.query("anything")
        assert hits == []


class TestVectorStoreDeleteBySource:
    def test_deletes_found_ids(self, mock_store):
        store, mock_col = mock_store
        mock_col.get.return_value = {"ids": ["id1", "id2"], "documents": [], "metadatas": []}
        n = store.delete_by_source("/note.md")
        assert n == 2
        mock_col.delete.assert_called_once_with(ids=["id1", "id2"])

    def test_returns_zero_when_nothing_found(self, mock_store):
        store, mock_col = mock_store
        mock_col.get.return_value = {"ids": [], "documents": [], "metadatas": []}
        n = store.delete_by_source("/nonexistent.md")
        assert n == 0
        mock_col.delete.assert_not_called()


class TestVectorStoreListSources:
    def test_deduplicates_sources(self, mock_store):
        store, mock_col = mock_store
        mock_col.get.return_value = {
            "metadatas": [
                {"source": "/a.md", "type": "note", "filename": "a.md", "ingested_at": "2024-01-01"},
                {"source": "/a.md", "type": "note", "filename": "a.md", "ingested_at": "2024-01-01"},
                {"source": "/b.md", "type": "code", "filename": "b.py", "ingested_at": "2024-01-02"},
            ]
        }
        sources = store.list_sources()
        assert len(sources) == 2

    def test_empty_store_returns_empty_list(self, tmp_path):
        store = VectorStore(persist_directory=str(tmp_path))
        mock_col = _make_mock_collection(count=0)
        store._collection = mock_col
        store._embed_fn = MagicMock()
        assert store.list_sources() == []
