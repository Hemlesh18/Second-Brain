"""Click-based command-line interface for Second Brain."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from second_brain.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    LLM_MODEL,
    NOTES_DIR,
    TOP_K,
)
from second_brain.ingestion.loaders import load_path
from second_brain.ingestion.tagger import generate_summary, generate_tags
from second_brain.memory.digest import MemoryDigest
from second_brain.retrieval.chain import RAGChain
from second_brain.vectorstore.store import VectorStore

console = Console()
logging.basicConfig(level=logging.WARNING)


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------

@click.group()
@click.version_option(package_name="second-brain")
def cli() -> None:
    """Second Brain – your private, local-LLM-powered knowledge assistant."""


# ---------------------------------------------------------------------------
# ingest
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("path", default=str(NOTES_DIR), required=False)
@click.option("--chunk-size", default=CHUNK_SIZE, show_default=True,
              help="Maximum characters per chunk.")
@click.option("--chunk-overlap", default=CHUNK_OVERLAP, show_default=True,
              help="Character overlap between chunks.")
@click.option("--tag/--no-tag", default=False,
              help="Auto-tag and summarise documents using the LLM.")
@click.option("--model", default=LLM_MODEL, show_default=True,
              help="Ollama model used for tagging/summarisation.")
def ingest(path: str, chunk_size: int, chunk_overlap: int, tag: bool, model: str) -> None:
    """Ingest documents from PATH (file or directory) into the vector store.

    PATH defaults to the NOTES_DIR environment variable (~./notes if unset).
    """
    console.print(f"[bold cyan]Ingesting[/bold cyan] [yellow]{path}[/yellow] ...")
    try:
        docs = load_path(path)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)

    if not docs:
        console.print("[yellow]No documents found.[/yellow]")
        return

    store = VectorStore()

    if tag:
        from langchain_ollama import OllamaLLM  # noqa: F401 – guard import
        llm = OllamaLLM(model=model)
        for doc in docs:
            tags = generate_tags(doc.content, llm)
            summary = generate_summary(doc.content, llm)
            doc.metadata["tags"] = ", ".join(tags)
            doc.metadata["summary"] = summary

    n_chunks = store.add_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    console.print(
        f"[green]Done.[/green] Indexed [bold]{len(docs)}[/bold] document(s) "
        f"as [bold]{n_chunks}[/bold] chunk(s)."
    )


# ---------------------------------------------------------------------------
# ask
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("question")
@click.option("--top-k", default=TOP_K, show_default=True,
              help="Number of chunks to retrieve.")
@click.option("--model", default=LLM_MODEL, show_default=True,
              help="Ollama model used for generation.")
@click.option("--type", "doc_type", default=None,
              help="Filter by document type (note, pdf, code, bookmark).")
def ask(question: str, top_k: int, model: str, doc_type: str) -> None:
    """Ask a natural-language QUESTION about your indexed documents."""
    store = VectorStore()
    if store.count() == 0:
        console.print(
            "[yellow]Your Second Brain is empty. "
            "Run [bold]second-brain ingest <path>[/bold] first.[/yellow]"
        )
        return

    chain = RAGChain(vector_store=store, model=model, top_k=top_k)
    filter_meta = {"type": doc_type} if doc_type else None

    with console.status("[bold green]Thinking...[/bold green]"):
        result = chain.ask(question, filter_metadata=filter_meta)

    console.print(Panel(Markdown(result["answer"]), title="[bold]Answer[/bold]", border_style="green"))

    if result["sources"]:
        table = Table(title="Sources", show_header=True, header_style="bold magenta")
        table.add_column("Source", style="cyan", no_wrap=False)
        table.add_column("Type", style="yellow")
        table.add_column("Ingested", style="dim")
        for src in result["sources"]:
            table.add_row(
                src.get("source", ""),
                src.get("type", ""),
                src.get("ingested_at", "")[:10],
            )
        console.print(table)


# ---------------------------------------------------------------------------
# digest
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--days", default=1, show_default=True,
              help="Look-back window in days.")
@click.option("--model", default=LLM_MODEL, show_default=True,
              help="Ollama model used for summarisation.")
@click.option("--changes/--no-changes", default=False,
              help="Show what changed over the past month vs last week.")
def digest(days: int, model: str, changes: bool) -> None:
    """Generate a Daily Memory Digest from recently ingested content."""
    store = VectorStore()
    memory = MemoryDigest(vector_store=store, model=model)

    if changes:
        with console.status("[bold green]Analysing changes...[/bold green]"):
            result = memory.what_changed()
        console.print(Panel(Markdown(result), title="[bold]What Changed[/bold]", border_style="blue"))
    else:
        with console.status("[bold green]Generating digest...[/bold green]"):
            result = memory.generate_digest(days=days)
        console.print(Panel(Markdown(result), title=f"[bold]Memory Digest (last {days}d)[/bold]", border_style="cyan"))


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------

@cli.command("list")
@click.option("--type", "doc_type", default=None,
              help="Filter by type (note, pdf, code, bookmark).")
def list_docs(doc_type: str) -> None:
    """List all documents indexed in the Second Brain."""
    store = VectorStore()
    sources = store.list_sources()

    if doc_type:
        sources = [s for s in sources if s.get("type") == doc_type]

    if not sources:
        console.print("[yellow]No documents found.[/yellow]")
        return

    table = Table(title=f"Indexed Documents ({len(sources)})", show_header=True, header_style="bold magenta")
    table.add_column("#", style="dim", width=4)
    table.add_column("Source", style="cyan", no_wrap=False)
    table.add_column("Type", style="yellow")
    table.add_column("Ingested", style="dim")

    for i, src in enumerate(sources, start=1):
        table.add_row(
            str(i),
            src.get("source", ""),
            src.get("type", ""),
            src.get("ingested_at", "")[:10],
        )
    console.print(table)


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("source")
@click.option("--yes", is_flag=True, help="Skip confirmation prompt.")
def delete(source: str, yes: bool) -> None:
    """Delete all chunks for a document SOURCE from the vector store."""
    if not yes:
        click.confirm(f"Delete all chunks for '{source}'?", abort=True)
    store = VectorStore()
    n = store.delete_by_source(source)
    if n:
        console.print(f"[green]Deleted {n} chunk(s) for[/green] [cyan]{source}[/cyan].")
    else:
        console.print(f"[yellow]No chunks found for[/yellow] [cyan]{source}[/cyan].")
