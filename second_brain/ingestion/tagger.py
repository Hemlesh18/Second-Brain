"""Auto-tagging and summarization pipeline.

Uses the local LLM (via Ollama) to:
- Generate topic tags for a document
- Produce a concise summary
"""

from __future__ import annotations

import logging
from typing import List

logger = logging.getLogger(__name__)

_TAG_PROMPT = """Analyze the following text and return a comma-separated list of
relevant topic tags (3-7 tags). Only output the tags, nothing else.

Text:
{text}

Tags:"""

_SUMMARY_PROMPT = """Summarize the following text in 2-3 sentences, capturing the
key ideas. Only output the summary, nothing else.

Text:
{text}

Summary:"""


def _truncate(text: str, max_chars: int = 3000) -> str:
    """Truncate text to *max_chars* to avoid overloading the LLM context."""
    return text[:max_chars] + "..." if len(text) > max_chars else text


def generate_tags(text: str, llm_client) -> List[str]:
    """Generate topic tags for *text* using *llm_client*.

    Parameters
    ----------
    text:
        The document content to tag.
    llm_client:
        An object with an ``invoke`` method that accepts a prompt string and
        returns a string response (e.g. a LangChain LLM or a simple wrapper
        around the Ollama SDK).

    Returns
    -------
    list[str]
        A list of lowercase tag strings.
    """
    prompt = _TAG_PROMPT.format(text=_truncate(text))
    try:
        response = llm_client.invoke(prompt)
        raw = response.content if hasattr(response, "content") else str(response)
        tags = [t.strip().lower() for t in raw.split(",") if t.strip()]
        return tags
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("Tag generation failed: %s", exc)
        return []


def generate_summary(text: str, llm_client) -> str:
    """Generate a concise summary of *text* using *llm_client*.

    Parameters
    ----------
    text:
        The document content to summarise.
    llm_client:
        An object with an ``invoke`` method (same contract as for
        :func:`generate_tags`).

    Returns
    -------
    str
        A short summary string, or an empty string on failure.
    """
    prompt = _SUMMARY_PROMPT.format(text=_truncate(text))
    try:
        response = llm_client.invoke(prompt)
        raw = response.content if hasattr(response, "content") else str(response)
        return raw.strip()
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("Summary generation failed: %s", exc)
        return ""
