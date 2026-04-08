from __future__ import annotations

from langchain_chroma import Chroma

from .config import get_settings
from .llm import get_embedding_model


def get_vectorstore() -> Chroma:
    settings = get_settings()
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=settings.collection_name,
        persist_directory=str(settings.chroma_dir),
        embedding_function=get_embedding_model(),
    )
