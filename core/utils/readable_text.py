"""The prose of a fetched page, without its menus.

A page converted from HTML keeps its navigation, its sidebars and its
footers as text, one short label per line. Readers that take a page's first
paragraph, or its first few thousand characters, then read the menu. This
keeps the runs of lines that read as prose, measured against the page's own
lines rather than a list of sites or labels.
"""

from __future__ import annotations

import re

__all__ = ["SENTENCE_END", "prose_paragraphs", "prose_of"]


SENTENCE_END = re.compile(r"[.!?][\"'”’)\]]*(?:\s|$)")


def prose_paragraphs(text: str) -> list[str]:
    """Runs of lines that read as prose: they end sentences and are longer than the page's median line.

    LIVE 2026-10-03: a fetched Wikipedia page opens with its navigation, one
    short label per line ("Jump to content", "Main menu", "Search"), and the
    passage taken from it was that menu; her repair answer said the link held
    only navigation. A page's own median line separates labels from prose,
    whatever the site.
    """
    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    if not lines:
        return []
    counts = sorted(len(line.split()) for line in lines)
    median = counts[len(counts) // 2]
    blocks: list[list[str]] = []
    for line in lines:
        if len(line.split()) > median and SENTENCE_END.search(line):
            if blocks and blocks[-1][-1] is not None:
                blocks[-1].append(line)
            else:
                blocks.append([line])
        elif blocks and blocks[-1][-1] is not None:
            blocks[-1].append(None)  # a break: the next prose line starts a new block
    return ["\n".join(line for line in block if line is not None) for block in blocks]


def prose_of(text: str) -> str:
    """The page's prose runs joined, or the text as it was when it has none."""
    runs = prose_paragraphs(text)
    return "\n\n".join(runs) if runs else str(text or "")
