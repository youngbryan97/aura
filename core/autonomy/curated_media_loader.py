"""core/autonomy/curated_media_loader.py
─────────────────────────────────────────
Parses ``aura/knowledge/bryan-curated-media.md`` into a typed corpus of
content items. Used by the (forthcoming) curiosity scheduler to pick what
to engage with next.

Narrow contract:
- ``load_corpus(path)`` returns ``list[ContentItem]``
- Each item has category, title, creator (or None), url (or None), description
- Parsing is permissive: any future bullet additions to the markdown that
  follow the existing format will be picked up automatically.

This module deliberately does NO fetching, NO state mutation, NO LLM calls.
It is a pure parser. Wiring into the autonomy pipeline lives elsewhere.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

DEFAULT_CORPUS_PATH = Path.home() / ".aura/live-source/aura/knowledge/bryan-curated-media.md"

_BULLET = re.compile(
    r"^- \*\*(?P<title>[^*]+?)\*\*\s*"
    r"(?:—\s*(?P<creator_or_url>[^—]+?))?"
    r"(?:\s*—\s*(?P<description>.+))?$"
)
_URL = re.compile(r"https?://\S+")


@dataclass(frozen=True)
class ContentItem:
    category: str
    title: str
    creator: Optional[str]
    url: Optional[str]
    description: str

    def has_direct_url(self) -> bool:
        return self.url is not None


def load_corpus(path: Path = DEFAULT_CORPUS_PATH) -> List[ContentItem]:
    """Parse the curated-media markdown into ContentItem records.

    Returns empty list if file is missing. Raises nothing on malformed
    bullets — they are skipped with a category-level marker so the parser
    never blocks on a single bad line.
    """
    if not path.exists():
        return []

    items: List[ContentItem] = []
    current_category: Optional[str] = None
    in_library = False

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.rstrip()

        # Detect entry into "The library" section; ignore preamble bullets
        if line.startswith("# The library"):
            in_library = True
            continue
        if not in_library:
            continue

        if line.startswith("## "):
            current_category = line[3:].strip()
            continue
        if line.startswith("---") or not line.startswith("- "):
            continue
        if current_category is None:
            continue

        match = _BULLET.match(line)
        if not match:
            continue

        title = match.group("title").strip()
        rest_a = (match.group("creator_or_url") or "").strip()
        description = (match.group("description") or "").strip()

        # rest_a may be either a creator name (films) or a URL (channels)
        url_match = _URL.search(rest_a)
        if url_match:
            url = url_match.group(0).rstrip(".,)")
            creator = None
        else:
            url = None
            creator = rest_a or None

        # Some entries put URL inside description (legacy format) — extract
        if url is None:
            d_url = _URL.search(description)
            if d_url:
                url = d_url.group(0).rstrip(".,)")

        items.append(
            ContentItem(
                category=current_category,
                title=title,
                creator=creator,
                url=url,
                description=description,
            )
        )

    return items


def categories(items: List[ContentItem]) -> List[str]:
    """Distinct categories in corpus order."""
    seen: List[str] = []
    for item in items:
        if item.category not in seen:
            seen.append(item.category)
    return seen
