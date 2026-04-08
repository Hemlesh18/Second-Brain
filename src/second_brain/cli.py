from __future__ import annotations

from pathlib import Path

import typer
from rich import print

from .digest import daily_memory_digest
from .pipeline import ingest
from .qa import answer_question
from .retrieval import retrieve
from .temporal import topic_timeline
from .vectorstore import get_vectorstore


app = typer.Typer(help="Personal Second Brain (local LLM + local vectors)")


@app.command()
def ingest_cmd(
    inputs: list[Path] = typer.Argument(..., help="Fichiers ou dossiers à indexer"),
    enrich: bool = typer.Option(False, "--enrich", help="Ajoute auto-tags et résumés"),
) -> None:
    stats = ingest(inputs, enrich=enrich)
    print("[green]Ingestion terminée[/green]")
    print(stats)


@app.command()
def ask(
    question: str = typer.Argument(..., help="Question à poser au Second Brain"),
    k: int = typer.Option(6, help="Nombre de passages récupérés"),
    since: str | None = typer.Option(None, help="Filtre date min ISO (YYYY-MM-DD)"),
    until: str | None = typer.Option(None, help="Filtre date max ISO (YYYY-MM-DD)"),
) -> None:
    docs = retrieve(question=question, k=k, since=since, until=until)
    if not docs:
        print("[yellow]Aucun document pertinent trouvé.[/yellow]")
        raise typer.Exit(0)

    answer = answer_question(question, docs)
    print(answer)


@app.command()
def timeline(topic: str = typer.Argument(..., help="Sujet à suivre dans le temps")) -> None:
    print(topic_timeline(topic))


@app.command()
def digest(day: str | None = typer.Option(None, help="Date ISO, ex: 2026-04-08")) -> None:
    print(daily_memory_digest(day=day))


@app.command()
def stats() -> None:
    store = get_vectorstore()
    count = store._collection.count()
    print({"chunks_in_store": count})


if __name__ == "__main__":
    app()
