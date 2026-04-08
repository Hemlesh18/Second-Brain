"""ChromaDB-backed vector store wrapper.

Responsibilities
----------------
- Embed documents with a local sentence-transformers model.
- Persist embeddings to disk via ChromaDB.
- Expose add / query / delete / list operations.
"""

from __future__ import annotations

import hashlib
import logging
from typing import List, Optional

from second_brain.config import BASE_DIR, EMBEDDING_MODEL, TOP_K
from second_brain.ingestion.loaders import Document

logger = logging.getLogger(__name__)

_COLLECTION_NAME = "second_brain"


def _doc_id(doc: Document) -> str:
    """Generate a stable ID for a document based on its source + content hash."""
    source = doc.metadata.get("source", "")
    digest = hashlib.sha256(doc.content.encode()).hexdigest()[:12]
    safe_source = source.replace("/", "_").replace("\\", "_")[-60:]
    return f"{safe_source}_{digest}"


class VectorStore:
    """Manages document embeddings and similarity search via ChromaDB."""

    def __init__(
        self,
        persist_directory: str = None,
        embedding_model: str = EMBEDDING_MODEL,
    ) -> None:
        """Initialise the vector store.

        Parameters
        ----------
        persist_directory:
            Path where ChromaDB will persist its data. Defaults to
            ``~/.second_brain/chroma_db``.
        embedding_model:
            Name of the sentence-transformers model used for embeddings.
        """
        self._persist_dir = persist_directory or str(BASE_DIR)
        self._embedding_model = embedding_model
        self._client = None
        self._collection = None
        self._embed_fn = None

    # ------------------------------------------------------------------
    # Lazy initialisation helpers
    # ------------------------------------------------------------------

    def _get_embed_fn(self):
        """Return (and cache) the ChromaDB embedding function."""
        if self._embed_fn is None:
            try:
                from chromadb.utils.embedding_functions import (
                    SentenceTransformerEmbeddingFunction,
                )
                self._embed_fn = SentenceTransformerEmbeddingFunction(
                    model_name=self._embedding_model
                )
            except ImportError as exc:
                raise ImportError(
                    "chromadb and sentence-transformers are required. "
                    "Install them with: pip install chromadb sentence-transformers"
                ) from exc
        return self._embed_fn

    def _get_collection(self):
        """Return (and cache) the ChromaDB collection."""
        if self._collection is None:
            try:
                import chromadb
            except ImportError as exc:
                raise ImportError(
                    "chromadb is required. Install it with: pip install chromadb"
                ) from exc

            self._client = chromadb.PersistentClient(path=self._persist_dir)
            self._collection = self._client.get_or_create_collection(
                name=_COLLECTION_NAME,
                embedding_function=self._get_embed_fn(),
            )
        return self._collection

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_documents(self, documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> int:
        """Split *documents* into chunks and add them to the store.

        Parameters
        ----------
        documents:
            Documents to ingest.
        chunk_size:
            Maximum number of characters per chunk.
        chunk_overlap:
            Character overlap between consecutive chunks.

        Returns
        -------
        int
            Number of chunks added (including those already present).
        """
        collection = self._get_collection()
        ids: List[str] = []
        texts: List[str] = []
        metas: List[dict] = []

        for doc in documents:
            chunks = _split_text(doc.content, chunk_size, chunk_overlap)
            for i, chunk in enumerate(chunks):
                chunk_id = f"{_doc_id(doc)}_c{i}"
                ids.append(chunk_id)
                texts.append(chunk)
                meta = {k: str(v) for k, v in doc.metadata.items()}
                meta["chunk_index"] = str(i)
                meta["total_chunks"] = str(len(chunks))
                metas.append(meta)

        if not ids:
            return 0

        # Upsert in batches to avoid large single requests
        batch_size = 100
        for start in range(0, len(ids), batch_size):
            collection.upsert(
                ids=ids[start : start + batch_size],
                documents=texts[start : start + batch_size],
                metadatas=metas[start : start + batch_size],
            )

        logger.info("Added %d chunks to vector store.", len(ids))
        return len(ids)

    def query(
        self,
        query_text: str,
        top_k: int = TOP_K,
        filter_metadata: Optional[dict] = None,
    ) -> List[dict]:
        """Return the *top_k* most relevant chunks for *query_text*.

        Parameters
        ----------
        query_text:
            The natural-language question or search string.
        top_k:
            Number of results to return.
        filter_metadata:
            Optional ChromaDB ``where`` filter dict.

        Returns
        -------
        list[dict]
            Each dict has ``content``, ``metadata``, and ``distance`` keys.
        """
        collection = self._get_collection()
        kwargs = {
            "query_texts": [query_text],
            "n_results": min(top_k, max(collection.count(), 1)),
        }
        if filter_metadata:
            kwargs["where"] = filter_metadata

        results = collection.query(**kwargs)
        hits = []
        docs_list = results.get("documents", [[]])[0]
        metas_list = results.get("metadatas", [[]])[0]
        distances_list = results.get("distances", [[]])[0]
        for doc_text, meta, dist in zip(docs_list, metas_list, distances_list):
            hits.append({"content": doc_text, "metadata": meta, "distance": dist})
        return hits

    def delete_by_source(self, source: str) -> int:
        """Delete all chunks whose ``source`` metadata field equals *source*.

        Returns the number of chunks deleted.
        """
        collection = self._get_collection()
        results = collection.get(where={"source": source})
        ids = results.get("ids", [])
        if ids:
            collection.delete(ids=ids)
        return len(ids)

    def list_sources(self) -> List[dict]:
        """Return a deduplicated list of ingested source documents.

        Returns
        -------
        list[dict]
            Each dict has ``source``, ``type``, ``filename``, and
            ``ingested_at`` keys (where available).
        """
        collection = self._get_collection()
        if collection.count() == 0:
            return []
        results = collection.get(include=["metadatas"])
        seen: dict = {}
        for meta in results.get("metadatas", []):
            src = meta.get("source", "")
            if src and src not in seen:
                seen[src] = {
                    "source": src,
                    "type": meta.get("type", ""),
                    "filename": meta.get("filename", ""),
                    "ingested_at": meta.get("ingested_at", ""),
                }
        return list(seen.values())

    def count(self) -> int:
        """Return the total number of chunks stored."""
        return self._get_collection().count()


# ---------------------------------------------------------------------------
# Text splitting helper
# ---------------------------------------------------------------------------

def _split_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Split *text* into overlapping chunks of at most *chunk_size* characters.

    Uses paragraph boundaries where possible to create more natural splits.
    """
    if len(text) <= chunk_size:
        return [text]

    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end >= len(text):
            chunks.append(text[start:])
            break
        # Try to split at a paragraph boundary first, then a newline, then
        # a space, falling back to a hard cut.
        for sep in ("\n\n", "\n", " "):
            split_pos = text.rfind(sep, start, end)
            if split_pos > start:
                end = split_pos + len(sep)
                break
        chunks.append(text[start:end])
        start = end - chunk_overlap
    return [c for c in chunks if c.strip()]
