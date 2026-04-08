# 🧠 Second Brain

> **Your private, local-LLM-powered knowledge assistant.**  
> Index your notes, PDFs, bookmarks, and code — then ask questions in plain English.  
> Everything runs 100 % locally. No data ever leaves your machine.

---

## Features

| Feature | Description |
|---|---|
| **Multi-format ingestion** | Notes (`.txt`, `.md`), PDFs, HTML/JSON bookmarks, source code |
| **Local embeddings** | `sentence-transformers` — no API key required |
| **Local LLM** | Llama 3, Mistral, or any model served by Ollama |
| **Semantic search** | Powered by ChromaDB vector store |
| **Auto-tagging** | LLM-generated topic tags and document summaries |
| **Daily Memory Digest** | Summarise what you learned today / this week |
| **Temporal reasoning** | "What changed over the past month?" — compare knowledge over time |
| **Privacy-first** | All computation runs locally; data stored in `~/.second_brain/` |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| Vector DB | ChromaDB (persisted to disk) |
| LLM | Ollama (Llama 3, Mistral, ...) |
| Orchestration | LangChain |
| CLI | Click + Rich |

---

## Prerequisites

1. **Python 3.10+**
2. **Ollama** running locally:

   ```bash
   # macOS / Linux
   curl https://ollama.ai/install.sh | sh
   ollama serve &          # start the server
   ollama pull llama3      # download the default model (~4 GB)
   ```

---

## Installation

```bash
# Clone the repo
git clone https://github.com/Hemlesh18/Second-Brain.git
cd Second-Brain

# (Recommended) create a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# (Optional) install as a CLI tool
pip install -e .
```

### Configuration

Copy `.env.example` to `.env` and customise as needed:

```bash
cp .env.example .env
```

| Variable | Default | Description |
|---|---|---|
| `LLM_MODEL` | `llama3` | Ollama model name |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | sentence-transformers model |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `NOTES_DIR` | `~/notes` | Default directory scanned by `ingest` |
| `CHUNK_SIZE` | `1000` | Max characters per chunk |
| `CHUNK_OVERLAP` | `200` | Character overlap between chunks |
| `TOP_K` | `5` | Number of chunks retrieved per query |

---

## Usage

All commands are available via the `second-brain` CLI (or `python main.py`).

### Ingest documents

```bash
# Ingest a whole directory (recursively)
second-brain ingest ~/notes

# Ingest a single file
second-brain ingest ~/papers/distributed-systems.pdf

# Ingest with auto-tagging and summarisation (requires Ollama)
second-brain ingest ~/notes --tag
```

### Ask questions

```bash
# Ask anything about your indexed content
second-brain ask "What did I learn about distributed systems last year?"

# Filter by document type
second-brain ask "Show me sorting algorithms" --type code

# Use a different model
second-brain ask "Summarise my machine learning notes" --model mistral
```

### Daily Memory Digest

```bash
# Digest of content ingested today
second-brain digest

# Digest of the last 7 days
second-brain digest --days 7

# Show what changed between last month and last week
second-brain digest --changes
```

### Manage your index

```bash
# List all indexed documents
second-brain list

# List only PDFs
second-brain list --type pdf

# Delete a specific document
second-brain delete ~/papers/old-paper.pdf
```

---

## Project Structure

```
second_brain/
├── config.py              # Centralised configuration (env vars + defaults)
├── ingestion/
│   ├── loaders.py         # Document loaders (txt, md, pdf, bookmarks, code)
│   └── tagger.py          # LLM-powered auto-tagging & summarisation
├── vectorstore/
│   └── store.py           # ChromaDB wrapper (add / query / delete / list)
├── retrieval/
│   └── chain.py           # RAG chain with temporal reasoning support
├── memory/
│   └── digest.py          # Daily Memory Digest & temporal comparison
└── cli/
    └── main.py            # Click CLI (ingest, ask, digest, list, delete)
main.py                    # Entry point
tests/                     # pytest test suite
```

---

## Running Tests

The test suite uses mocks so it does **not** require Ollama or a real ChromaDB
instance:

```bash
pip install pytest pytest-mock
pytest tests/ -v
```

---

## Stretch Ideas (Roadmap)

- [ ] **Web UI** – A simple Gradio or Streamlit front end
- [ ] **Scheduled digest** – Cron job / systemd timer for a daily email/notification
- [ ] **Graph of ideas** – Build a knowledge graph from document relationships
- [ ] **Obsidian plugin** – Sync with an Obsidian vault
- [ ] **Multi-modal** – Index images (OCR) and audio (Whisper transcription)

---

## Privacy

All data is stored locally:
- Embeddings -> `~/.second_brain/chroma_db/`
- No telemetry, no API calls to external services

---

## License

MIT
