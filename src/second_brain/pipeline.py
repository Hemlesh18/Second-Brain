from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .llm import get_chat_llm
from .loaders import load_documents
from .vectorstore import get_vectorstore


TEXT_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=180,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def discover_files(inputs: list[Path]) -> list[Path]:
    files: list[Path] = []
    for p in inputs:
        if p.is_file():
            files.append(p)
            continue
        if p.is_dir():
            for child in p.rglob("*"):
                if child.is_file():
                    files.append(child)
    return files


def split_documents(docs: list[Document]) -> list[Document]:
    return TEXT_SPLITTER.split_documents(docs)


def _summarize_and_tag(content: str) -> tuple[str, list[str]]:
    llm = get_chat_llm(temperature=0.1)
    prompt = (
        "Tu es un assistant de knowledge management. "
        "Retourne STRICTEMENT un JSON avec les clés summary (string) et tags (tableau de 3 à 8 tags).\n\n"
        f"Contenu:\n{content[:8000]}"
    )
    response = llm.invoke(prompt)
    text = response.content if isinstance(response.content, str) else str(response.content)

    import json
    import re

    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        return (content[:300], ["untagged"])

    try:
        data = json.loads(match.group(0))
        summary = str(data.get("summary", "")).strip() or content[:300]
        tags = data.get("tags", ["untagged"])
        if not isinstance(tags, list):
            tags = ["untagged"]
        tags = [str(t).strip().lower() for t in tags if str(t).strip()]
        return summary, tags[:8] or ["untagged"]
    except Exception:
        return (content[:300], ["untagged"])


def build_enrichment_docs(docs: Iterable[Document]) -> list[Document]:
    enriched: list[Document] = []
    now_iso = datetime.now(timezone.utc).isoformat()
    now_ts = datetime.now(timezone.utc).timestamp()

    for d in docs:
        summary, tags = _summarize_and_tag(d.page_content)
        enriched.append(
            Document(
                page_content=f"Summary: {summary}\nTags: {', '.join(tags)}",
                metadata={
                    **d.metadata,
                    "doc_kind": "summary",
                    "tags": ",".join(tags),
                    "enriched_at": now_iso,
                    "enriched_ts": now_ts,
                },
            )
        )
    return enriched


def ingest(inputs: list[Path], enrich: bool = False) -> dict:
    files = discover_files(inputs)
    raw_docs = load_documents(files)
    chunks = split_documents(raw_docs)

    all_docs = list(chunks)
    if enrich and raw_docs:
        all_docs.extend(build_enrichment_docs(raw_docs))

    if not all_docs:
        return {"files": len(files), "raw_docs": 0, "chunks": 0, "stored": 0}

    store = get_vectorstore()
    store.add_documents(all_docs)

    return {
        "files": len(files),
        "raw_docs": len(raw_docs),
        "chunks": len(chunks),
        "stored": len(all_docs),
    }
