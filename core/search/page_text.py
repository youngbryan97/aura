"""The readable text of an HTML page: scripts and styles out, block tags as line breaks, short scraps dropped."""
from __future__ import annotations

import html
import re

__all__ = ["_html_to_text"]


def _html_to_text(raw_html: str) -> str:
    text = re.sub(
        r"(?is)<(script|style|noscript|svg|iframe|canvas).*?>.*?</\1>",
        " ",
        raw_html,
    )
    text = re.sub(
        r"(?i)</?(article|section|div|p|li|ul|ol|table|tr|td|th|h[1-6]|blockquote|br)[^>]*>",
        "\n",
        text,
    )
    text = re.sub(r"(?is)<[^>]+>", " ", text)
    text = html.unescape(text)
    lines = [
        re.sub(r"\s+", " ", line).strip()
        for line in text.splitlines()
    ]
    lines = [line for line in lines if len(line) >= 3]
    return "\n".join(lines)
