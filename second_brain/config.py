"""Second Brain – configuration."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Storage paths
# ---------------------------------------------------------------------------
BASE_DIR: Path = Path(os.getenv("CHROMA_DIR", str(Path.home() / ".second_brain" / "chroma_db")))

# ---------------------------------------------------------------------------
# LLM / embedding settings
# ---------------------------------------------------------------------------
LLM_MODEL: str = os.getenv("LLM_MODEL", "llama3")
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# ---------------------------------------------------------------------------
# Ingestion settings
# ---------------------------------------------------------------------------
NOTES_DIR: Path = Path(os.getenv("NOTES_DIR", str(Path.home() / "notes")))
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))

# ---------------------------------------------------------------------------
# Retrieval settings
# ---------------------------------------------------------------------------
TOP_K: int = int(os.getenv("TOP_K", "5"))

# ---------------------------------------------------------------------------
# Supported file extensions
# ---------------------------------------------------------------------------
TEXT_EXTENSIONS = {".txt", ".md", ".rst"}
CODE_EXTENSIONS = {".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".go", ".rs",
                   ".cpp", ".c", ".h", ".cs", ".rb", ".php", ".sh", ".yaml",
                   ".yml", ".toml", ".json"}
PDF_EXTENSIONS = {".pdf"}
BOOKMARK_EXTENSIONS = {".html", ".htm"}
