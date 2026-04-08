"""Daily memory digest.

Generates a summary of documents ingested recently and surfaces noteworthy
patterns, helping you stay on top of what your Second Brain has learned.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from second_brain.config import LLM_MODEL, OLLAMA_BASE_URL
from second_brain.vectorstore.store import VectorStore

logger = logging.getLogger(__name__)

_DIGEST_PROMPT = """You are a personal knowledge assistant. The user has been
collecting notes and documents. Below is a summary of content indexed in the
last {period}. Create a helpful "Daily Memory Digest" that includes:

1. Key themes and topics encountered
2. Most important insights or facts
3. Suggested connections between ideas
4. Questions worth exploring further

Recent content:
{content}

Daily Memory Digest:"""

_CHANGE_PROMPT = """Below are excerpts from documents indexed in two different
time periods. Describe what changed or evolved between the two periods.

Earlier period content:
{earlier}

Recent content:
{recent}

What changed or evolved:"""


def _format_period(days: int) -> str:
    if days == 1:
        return "24 hours"
    if days <= 7:
        return f"{days} days"
    return f"{days // 7} week(s)"


class MemoryDigest:
    """Generates digest reports from recently ingested documents."""

    def __init__(
        self,
        vector_store: VectorStore = None,
        model: str = LLM_MODEL,
        ollama_base_url: str = OLLAMA_BASE_URL,
    ) -> None:
        self._store = vector_store or VectorStore()
        self._model = model
        self._ollama_base_url = ollama_base_url
        self._llm = None

    def _get_llm(self):
        if self._llm is None:
            try:
                from langchain_ollama import OllamaLLM
                self._llm = OllamaLLM(
                    model=self._model,
                    base_url=self._ollama_base_url,
                )
            except ImportError as exc:
                raise ImportError(
                    "langchain-ollama is required. "
                    "Install it with: pip install langchain-ollama"
                ) from exc
        return self._llm

    def _get_recent_chunks(self, days: int) -> List[dict]:
        """Return all chunks ingested within the last *days* days."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        cutoff_str = cutoff.isoformat()

        collection = self._store._get_collection()  # pylint: disable=protected-access
        if collection.count() == 0:
            return []

        results = collection.get(include=["documents", "metadatas"])
        chunks = []
        for doc_text, meta in zip(
            results.get("documents", []), results.get("metadatas", [])
        ):
            ingested_at = meta.get("ingested_at", "")
            if ingested_at >= cutoff_str:
                chunks.append({"content": doc_text, "metadata": meta})
        return chunks

    def generate_digest(self, days: int = 1) -> str:
        """Generate a memory digest for content ingested in the last *days* days.

        Parameters
        ----------
        days:
            Look-back window in days (default: 1 = today).

        Returns
        -------
        str
            The formatted digest text, or an informational message if no
            recent content was found.
        """
        chunks = self._get_recent_chunks(days)
        if not chunks:
            return (
                f"No new content was indexed in the last {_format_period(days)}. "
                "Try running `second-brain ingest <path>` to add documents."
            )

        # Aggregate content (truncate to ~4000 chars to stay within context)
        combined = "\n\n---\n\n".join(
            f"[{c['metadata'].get('source', 'unknown')}]\n{c['content']}"
            for c in chunks
        )
        if len(combined) > 4000:
            combined = combined[:4000] + "\n\n[...truncated for brevity...]"

        prompt = _DIGEST_PROMPT.format(
            period=_format_period(days), content=combined
        )
        llm = self._get_llm()
        response = llm.invoke(prompt)
        result = response.content if hasattr(response, "content") else str(response)
        return result.strip()

    def what_changed(self, earlier_days: int = 30, recent_days: int = 7) -> str:
        """Describe how your knowledge has evolved over time.

        Compares content from *earlier_days* ago against the last *recent_days*.

        Parameters
        ----------
        earlier_days:
            How many days ago to consider as "earlier" content.
        recent_days:
            Look-back window for "recent" content.

        Returns
        -------
        str
            A temporal comparison narrative.
        """
        recent_chunks = self._get_recent_chunks(recent_days)
        earlier_chunks = self._get_recent_chunks(earlier_days)
        # Remove recent from earlier
        recent_sources = {c["metadata"].get("source") for c in recent_chunks}
        older_chunks = [
            c for c in earlier_chunks
            if c["metadata"].get("source") not in recent_sources
        ]

        if not recent_chunks and not older_chunks:
            return "Not enough historical data to compare."

        def _summarise(chunks: List[dict], max_chars: int = 2000) -> str:
            text = "\n\n---\n\n".join(c["content"] for c in chunks)
            return text[:max_chars] + ("..." if len(text) > max_chars else "")

        prompt = _CHANGE_PROMPT.format(
            earlier=_summarise(older_chunks),
            recent=_summarise(recent_chunks),
        )
        llm = self._get_llm()
        response = llm.invoke(prompt)
        result = response.content if hasattr(response, "content") else str(response)
        return result.strip()
