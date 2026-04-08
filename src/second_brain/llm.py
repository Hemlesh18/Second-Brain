from __future__ import annotations

from langchain_ollama import ChatOllama, OllamaEmbeddings

from .config import get_settings


settings = get_settings()


def get_chat_llm(temperature: float = 0.2) -> ChatOllama:
    return ChatOllama(model=settings.chat_model, temperature=temperature)


def get_embedding_model() -> OllamaEmbeddings:
    return OllamaEmbeddings(model=settings.embed_model)
