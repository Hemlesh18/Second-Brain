from __future__ import annotations

from langchain_core.documents import Document

from .llm import get_chat_llm


def _format_context(docs: list[Document]) -> str:
    blocks = []
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page")
        when = d.metadata.get("source_mtime", "")
        ref = f"{src}" + (f" p.{page}" if page is not None else "")
        blocks.append(f"[{i}] {ref} ({when})\n{d.page_content}")
    return "\n\n".join(blocks)


def answer_question(question: str, docs: list[Document]) -> str:
    llm = get_chat_llm(temperature=0.1)
    prompt = (
        "Tu réponds en t'appuyant uniquement sur le contexte fourni. "
        "Si le contexte ne suffit pas, dis-le explicitement. "
        "Termine avec une section 'Sources' en listant les références [n].\n\n"
        f"Question:\n{question}\n\n"
        f"Contexte:\n{_format_context(docs)}"
    )
    response = llm.invoke(prompt)
    return response.content if isinstance(response.content, str) else str(response.content)
