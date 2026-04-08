"""Tests for the RAG chain and tagger."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from second_brain.ingestion.tagger import generate_summary, generate_tags
from second_brain.retrieval.chain import RAGChain, _build_context, _is_temporal_query


# ---------------------------------------------------------------------------
# Temporal query detection
# ---------------------------------------------------------------------------

class TestIsTemporalQuery:
    @pytest.mark.parametrize("question", [
        "What did I learn last year about ML?",
        "What changed over time in my notes?",
        "Show me recent activity.",
        "What did I read last week?",
        "When did I start learning Rust?",
    ])
    def test_detects_temporal_keywords(self, question):
        assert _is_temporal_query(question) is True

    @pytest.mark.parametrize("question", [
        "What is a neural network?",
        "Explain Raft consensus.",
        "How does Python GIL work?",
    ])
    def test_non_temporal_queries(self, question):
        assert _is_temporal_query(question) is False


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------

class TestBuildContext:
    def test_includes_source_and_content(self):
        hits = [
            {
                "content": "Raft uses leader election.",
                "metadata": {"source": "/notes/raft.md", "ingested_at": "2024-03-01T00:00:00"},
                "distance": 0.1,
            }
        ]
        ctx = _build_context(hits)
        assert "raft.md" in ctx
        assert "Raft uses leader election." in ctx
        assert "2024-03-01" in ctx

    def test_multiple_hits_separated_by_divider(self):
        hits = [
            {"content": "A", "metadata": {"source": "/a.md"}, "distance": 0.1},
            {"content": "B", "metadata": {"source": "/b.md"}, "distance": 0.2},
        ]
        ctx = _build_context(hits)
        assert "---" in ctx
        assert "A" in ctx
        assert "B" in ctx


# ---------------------------------------------------------------------------
# RAGChain.ask
# ---------------------------------------------------------------------------

def _make_rag_chain(hits=None, llm_response="The answer is 42."):
    mock_store = MagicMock()
    mock_store.count.return_value = 5
    mock_store.query.return_value = hits if hits is not None else [
        {
            "content": "Raft is a consensus algorithm.",
            "metadata": {"source": "/notes/raft.md", "type": "note", "ingested_at": "2024-01-01"},
            "distance": 0.05,
        }
    ]

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = llm_response

    chain = RAGChain(vector_store=mock_store)
    chain._llm = mock_llm
    return chain, mock_store, mock_llm


class TestRAGChainAsk:
    def test_returns_answer_and_sources(self):
        chain, _, _ = _make_rag_chain()
        result = chain.ask("What is Raft?")
        assert result["answer"] == "The answer is 42."
        assert len(result["sources"]) == 1
        assert result["sources"][0]["source"] == "/notes/raft.md"

    def test_empty_store_returns_no_docs_message(self):
        chain, mock_store, _ = _make_rag_chain(hits=[])
        result = chain.ask("anything")
        assert "couldn't find" in result["answer"].lower()

    def test_deduplicates_sources(self):
        hits = [
            {
                "content": "chunk 1",
                "metadata": {"source": "/notes/raft.md", "type": "note", "ingested_at": "2024-01-01"},
                "distance": 0.05,
            },
            {
                "content": "chunk 2",
                "metadata": {"source": "/notes/raft.md", "type": "note", "ingested_at": "2024-01-01"},
                "distance": 0.06,
            },
        ]
        chain, _, _ = _make_rag_chain(hits=hits)
        result = chain.ask("What is Raft?")
        assert len(result["sources"]) == 1

    def test_temporal_query_passes_through(self):
        chain, _, mock_llm = _make_rag_chain()
        chain.ask("What did I learn last year about distributed systems?")
        prompt_used = mock_llm.invoke.call_args[0][0]
        # Temporal hint should be prepended
        assert "time period" in prompt_used.lower() or "temporal" in prompt_used.lower()

    def test_filter_metadata_forwarded_to_store(self):
        chain, mock_store, _ = _make_rag_chain()
        chain.ask("How to sort?", filter_metadata={"type": "code"})
        call_kwargs = mock_store.query.call_args[1]
        assert call_kwargs.get("filter_metadata") == {"type": "code"}


# ---------------------------------------------------------------------------
# Tagger
# ---------------------------------------------------------------------------

class TestGenerateTags:
    def test_returns_list_of_tags(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "python, machine learning, algorithms"
        tags = generate_tags("Some text about ML", mock_llm)
        assert tags == ["python", "machine learning", "algorithms"]

    def test_handles_llm_response_with_content_attr(self):
        mock_llm = MagicMock()
        response = MagicMock()
        response.content = "ai, nlp"
        mock_llm.invoke.return_value = response
        tags = generate_tags("NLP text", mock_llm)
        assert "ai" in tags
        assert "nlp" in tags

    def test_returns_empty_on_llm_error(self):
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = RuntimeError("LLM unavailable")
        tags = generate_tags("some text", mock_llm)
        assert tags == []

    def test_truncates_long_text(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "tag1"
        long_text = "x" * 10000
        generate_tags(long_text, mock_llm)
        prompt = mock_llm.invoke.call_args[0][0]
        # The truncated text + "..." should appear, not the full 10000-char text
        assert len(prompt) < 10000


class TestGenerateSummary:
    def test_returns_summary_string(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "A great summary."
        summary = generate_summary("Some long document text...", mock_llm)
        assert summary == "A great summary."

    def test_returns_empty_on_error(self):
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = ConnectionError("no server")
        summary = generate_summary("text", mock_llm)
        assert summary == ""
