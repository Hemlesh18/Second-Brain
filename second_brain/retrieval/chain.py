"""RAG (Retrieval-Augmented Generation) chain.

Wires together the vector store retriever and the local Ollama LLM to answer
questions about your indexed documents.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from second_brain.config import LLM_MODEL, OLLAMA_BASE_URL, TOP_K
from second_brain.vectorstore.store import VectorStore

logger = logging.getLogger(__name__)

_QA_PROMPT = """You are a helpful personal knowledge assistant. Use only the
context provided below to answer the question. If the context does not contain
enough information to answer, say so honestly.

Context:
{context}

Question: {question}

Answer:"""

_TEMPORAL_HINT = """The user's question may involve a specific time period.
Pay attention to dates and timestamps in the context when formulating your answer.

"""


def _is_temporal_query(question: str) -> bool:
    """Detect whether a question has a temporal component."""
    temporal_keywords = [
        "last year", "last month", "last week", "yesterday", "today",
        "this year", "this month", "recently", "when did", "what changed",
        "over time", "history", "before", "after", "since", "ago",
    ]
    question_lower = question.lower()
    return any(kw in question_lower for kw in temporal_keywords)


def _build_context(hits: List[dict]) -> str:
    """Format retrieved chunks into a numbered context string."""
    parts = []
    for i, hit in enumerate(hits, start=1):
        source = hit["metadata"].get("source", "unknown")
        ingested = hit["metadata"].get("ingested_at", "")
        header = f"[{i}] Source: {source}"
        if ingested:
            header += f" | Date: {ingested[:10]}"
        parts.append(f"{header}\n{hit['content']}")
    return "\n\n---\n\n".join(parts)


class RAGChain:
    """Retrieval-Augmented Generation chain backed by a local LLM."""

    def __init__(
        self,
        vector_store: VectorStore = None,
        model: str = LLM_MODEL,
        ollama_base_url: str = OLLAMA_BASE_URL,
        top_k: int = TOP_K,
    ) -> None:
        """Initialise the RAG chain.

        Parameters
        ----------
        vector_store:
            Pre-initialised :class:`VectorStore`. If ``None`` a new one is
            created with default settings.
        model:
            Ollama model name to use for generation.
        ollama_base_url:
            Base URL of the running Ollama server.
        top_k:
            Number of chunks to retrieve per query.
        """
        self._store = vector_store or VectorStore()
        self._model = model
        self._ollama_base_url = ollama_base_url
        self._top_k = top_k
        self._llm = None

    def _get_llm(self):
        """Return (and cache) the LangChain Ollama LLM instance."""
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

    def ask(
        self,
        question: str,
        filter_metadata: Optional[dict] = None,
    ) -> dict:
        """Answer *question* using retrieved context.

        Parameters
        ----------
        question:
            The user's natural-language question.
        filter_metadata:
            Optional metadata filter passed through to the vector store query.

        Returns
        -------
        dict
            A dict with keys ``answer`` (str) and ``sources`` (list[dict]).
        """
        hits = self._store.query(question, top_k=self._top_k, filter_metadata=filter_metadata)
        if not hits:
            return {
                "answer": "I couldn't find any relevant documents in your Second Brain.",
                "sources": [],
            }

        context = _build_context(hits)
        temporal_prefix = _TEMPORAL_HINT if _is_temporal_query(question) else ""
        prompt = temporal_prefix + _QA_PROMPT.format(
            context=context, question=question
        )

        llm = self._get_llm()
        response = llm.invoke(prompt)
        answer = response.content if hasattr(response, "content") else str(response)

        # Deduplicate sources
        sources = []
        seen_sources = set()
        for hit in hits:
            src = hit["metadata"].get("source", "")
            if src and src not in seen_sources:
                sources.append(hit["metadata"])
                seen_sources.add(src)

        return {"answer": answer.strip(), "sources": sources}
