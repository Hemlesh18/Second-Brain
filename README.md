# Personal Second Brain (Local LLM)

Un "Second Brain" prive pour interroger tout ce que vous lisez et produisez:
- notes personnelles
- PDFs
- bookmarks
- code

Objectif: poser des questions comme:
- "Qu'est-ce que j'ai appris sur les systemes distribues l'an dernier ?"
- "Quels patterns reviennent dans mes notes d'architecture ?"
- "Comment ma comprehension de X a evolue dans le temps ?"

## Stack

- Embeddings + vector store: Chroma (local)
- LLM local: Ollama (Llama 3.1 / Mistral)
- Framework: LangChain

## Features

- Ingestion multi-sources (notes, PDF, bookmarks, code)
- Retrieval + reponse contextualisee
- Filtre temporel (`--since`, `--until`)
- Auto-tagging + summarization (`--enrich`)
- Daily memory digest
- Timeline thematique (ce qui change dans le temps)

## Prerequis

- Python 3.10+
- Ollama installe et lance localement

Telecharger des modeles (exemple):

```bash
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
cp .env.example .env
```

## Structure recommandee

```text
data/
	raw/          # vos notes/PDF/bookmarks/code a indexer
	chroma/       # base vectorielle locale
src/second_brain/
```

## Utilisation

Ingestion d'un dossier:

```bash
second-brain ingest ./data/raw
```

Ingestion avec enrichissement (resumes + tags auto):

```bash
second-brain ingest ./data/raw --enrich
```

Poser une question:

```bash
second-brain ask "Que sais-je sur les distributed systems ?"
```

Question avec filtre temporel:

```bash
second-brain ask "Qu'ai-je appris sur la replication ?" --since 2025-01-01 --until 2025-12-31
```

Timeline d'un sujet:

```bash
second-brain timeline "distributed systems"
```

Digest quotidien:

```bash
second-brain digest --day 2026-04-08
```

Statistiques du store:

```bash
second-brain stats
```

## Types de fichiers pris en charge

- Texte / notes: md, txt, rst, org, html, csv, json, yaml, toml
- Code: py, js, ts, tsx, jsx, java, go, rs, c, cpp, h, hpp, sh
- PDFs: .pdf
- Bookmarks:
	- HTML type export navigateur (Netscape bookmark file)
	- JSON de bookmarks (selon navigateur)

## Idees d'evolution

- Pipeline incremental avec detection de fichiers modifies
- Re-ranking (cross-encoder) pour ameliorer la precision
- Dashboard web (Streamlit/FastAPI + UI)
- Connecteurs Notion/Obsidian/GitHub/Gmail
- Memoire active avec rappels proactifs

## Depannage

- Erreur connexion modele:
	- verifier qu'Ollama tourne (`ollama list`)
	- verifier les variables `.env`
- Resultats faibles:
	- augmenter `--k`
	- ingerer plus de contexte
	- activer `--enrich`

## Confidentialite

Toutes les donnees restent locales si vous gardez:
- Ollama local
- Chroma local
- aucun appel API cloud