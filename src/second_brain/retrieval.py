from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from langchain_core.documents import Document

from .config import get_settings
from .vectorstore import get_vectorstore


def _to_timestamp(date_str: str | None) -> float | None:
    if not date_str:
        return None
    dt = datetime.fromisoformat(date_str)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def build_time_filter(since: str | None, until: str | None) -> dict[str, Any] | None:
    since_ts = _to_timestamp(since)
    until_ts = _to_timestamp(until)

    if since_ts is None and until_ts is None:
        return None

    if since_ts is not None and until_ts is not None:
        return {"$and": [{"source_mtime_ts": {"$gte": since_ts}}, {"source_mtime_ts": {"$lte": until_ts}}]}
    if since_ts is not None:
        return {"source_mtime_ts": {"$gte": since_ts}}
    return {"source_mtime_ts": {"$lte": until_ts}}


def retrieve(question: str, k: int | None = None, since: str | None = None, until: str | None = None) -> list[Document]:
    store = get_vectorstore()
    settings = get_settings()
    top_k = k or settings.top_k

    where = build_time_filter(since=since, until=until)
    return store.similarity_search(question, k=top_k, filter=where)
