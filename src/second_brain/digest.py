from __future__ import annotations

from datetime import datetime, timedelta, timezone

from .llm import get_chat_llm
from .vectorstore import get_vectorstore


def _day_bounds(day: str | None) -> tuple[float, float, str]:
    if day:
        base = datetime.fromisoformat(day)
    else:
        base = datetime.now(timezone.utc)

    if base.tzinfo is None:
        base = base.replace(tzinfo=timezone.utc)

    start = datetime(base.year, base.month, base.day, tzinfo=base.tzinfo)
    end = start + timedelta(days=1)
    return start.timestamp(), end.timestamp(), start.date().isoformat()


def daily_memory_digest(day: str | None = None, max_items: int = 50) -> str:
    start_ts, end_ts, day_iso = _day_bounds(day)
    store = get_vectorstore()

    data = store.get(
        where={
            "$and": [
                {"ingested_ts": {"$gte": start_ts}},
                {"ingested_ts": {"$lt": end_ts}},
            ]
        },
        include=["documents", "metadatas"],
        limit=max_items,
    )

    documents = data.get("documents", []) if data else []
    metadatas = data.get("metadatas", []) if data else []
    if not documents:
        return f"Aucune nouvelle mémoire ingérée le {day_iso}."

    lines = []
    for content, meta in zip(documents, metadatas):
        src = (meta or {}).get("source", "unknown")
        lines.append(f"Source: {src}\n{content[:1000]}")

    llm = get_chat_llm(temperature=0.2)
    prompt = (
        "Crée un digest quotidien concis en français: "
        "1) top apprentissages, 2) éléments actionnables, 3) angles à revisiter.\n\n"
        f"Date: {day_iso}\n\n"
        + "\n\n---\n\n".join(lines)
    )
    response = llm.invoke(prompt)
    return response.content if isinstance(response.content, str) else str(response.content)
