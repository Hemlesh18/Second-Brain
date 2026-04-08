"""Shared pytest fixtures."""

from __future__ import annotations

import pytest

from second_brain.ingestion.loaders import Document


@pytest.fixture()
def sample_document() -> Document:
    return Document(
        content="Distributed systems use consensus algorithms like Raft and Paxos.",
        metadata={"source": "/notes/dist_systems.md", "type": "note", "filename": "dist_systems.md"},
    )


@pytest.fixture()
def sample_documents() -> list:
    return [
        Document(
            content="Raft is a consensus algorithm designed for understandability.",
            metadata={"source": "/notes/raft.md", "type": "note"},
        ),
        Document(
            content="def merge_sort(arr): pass  # sort implementation",
            metadata={"source": "/code/sort.py", "type": "code", "language": "py"},
        ),
        Document(
            content="Bookmark: Python docs\nURL: https://docs.python.org",
            metadata={"source": "/bookmarks.html", "type": "bookmark", "url": "https://docs.python.org"},
        ),
    ]
