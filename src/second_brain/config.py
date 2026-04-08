from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    chat_model: str = os.getenv("SB_CHAT_MODEL", "llama3.1:8b")
    embed_model: str = os.getenv("SB_EMBED_MODEL", "nomic-embed-text")
    chroma_dir: Path = Path(os.getenv("SB_CHROMA_DIR", "./data/chroma"))
    collection_name: str = os.getenv("SB_COLLECTION", "second_brain")
    top_k: int = int(os.getenv("SB_TOP_K", "6"))


def get_settings() -> Settings:
    return Settings()
