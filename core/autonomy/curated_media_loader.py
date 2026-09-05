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

import os
import re
from dataclasses import dataclass
from pathlib import Path

from core.runtime.errors import record_degradation


def _default_corpus_path() -> Path:
    override = os.environ.get("AURA_CURATED_MEDIA_CORPUS_PATH")
    if override:
        return Path(override).expanduser()
    project_root = Path(os.environ.get("AURA_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
    return project_root.resolve() / "aura/knowledge/bryan-curated-media.md"


DEFAULT_CORPUS_PATH = _default_corpus_path()

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
    creator: str | None
    url: str | None
    description: str

    def has_direct_url(self) -> bool:
        return self.url is not None


@dataclass(frozen=True)
class CorpusParseReport:
    """What the parser accepted, and what it could not.

    CP126 (high): "Malformed bullets are silently dropped. The docstring
    promises a category-level marker, but unmatched bullets and entries
    before the exact library heading are skipped with no count, warning, or
    parse report."

    The docstring described a marker that does not exist. Skipping a bad
    line is the right behaviour — one typo should not empty the library —
    but a corpus that silently loses half its entries to a formatting change
    looks exactly like a corpus that is genuinely half that size, and the
    only symptom is Aura quietly never mentioning those films again.
    """

    total_bullets: int = 0
    parsed: int = 0
    unmatched: int = 0
    before_library_heading: int = 0
    uncategorised: int = 0
    samples: tuple[str, ...] = ()

    @property
    def dropped(self) -> int:
        """Library entries that were LOST, which preamble bullets are not.

        Measured against the real corpus, all twelve skipped bullets were
        instructional preamble before the "# The library" heading — prose
        that was never a library entry. Counting those as loss would make
        this report cry wolf on a healthy file, and a report that cries wolf
        gets muted, which is how the original silence returns.
        """
        return self.unmatched + self.uncategorised

    @property
    def complete(self) -> bool:
        return self.dropped == 0

    def to_dict(self) -> dict:
        return {
            "schema": "aura.curated_corpus_parse.v1",
            "total_bullets": self.total_bullets,
            "parsed": self.parsed,
            "dropped": self.dropped,
            "unmatched": self.unmatched,
            # Reported for completeness, deliberately NOT counted as loss.
            "before_library_heading": self.before_library_heading,
            "uncategorised": self.uncategorised,
            "complete": self.complete,
            "samples": list(self.samples),
        }


def load_corpus_with_report(
    path: Path | None = None,
) -> tuple[list[ContentItem], CorpusParseReport]:
    """Parse the curated-media markdown, reporting what was skipped.

    Skipping a malformed bullet is deliberate — one typo must not empty the
    library — but the skip is now counted and sampled, so a formatting change
    that quietly drops half the corpus is visible instead of looking like a
    smaller library.
    """
    path = Path(path).expanduser() if path is not None else _default_corpus_path()
    if not path.exists():
        return [], CorpusParseReport()

    items: list[ContentItem] = []
    current_category: str | None = None
    in_library = False
    total_bullets = 0
    unmatched = 0
    before_heading = 0
    uncategorised = 0
    samples: list[str] = []

    def _sample(line: str) -> None:
        """Only real losses are sampled; preamble is not a loss."""
        if len(samples) < 5:
            samples.append(line.strip()[:120])

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.rstrip()

        # Detect entry into "The library" section; ignore preamble bullets
        if line.startswith("# The library"):
            in_library = True
            continue
        if not in_library:
            if line.startswith("- "):
                total_bullets += 1
                before_heading += 1
            continue

        if line.startswith("## "):
            current_category = line[3:].strip()
            continue
        if line.startswith("---") or not line.startswith("- "):
            continue
        total_bullets += 1
        if current_category is None:
            uncategorised += 1
            _sample(line)
            continue

        match = _BULLET.match(line)
        if not match:
            unmatched += 1
            _sample(line)
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

    report = CorpusParseReport(
        total_bullets=total_bullets,
        parsed=len(items),
        unmatched=unmatched,
        before_library_heading=before_heading,
        uncategorised=uncategorised,
        samples=tuple(samples),
    )
    if not report.complete:
        record_degradation(
            "curated_media_loader",
            ValueError(
                f"curated corpus parsed {report.parsed}/{report.total_bullets} "
                f"bullets ({report.dropped} skipped)"
            ),
            severity="warning",
            action="loaded the parseable entries; the skipped ones are absent from the library",
            enforce_failure_policy=False,
        )
    return items, report


def load_corpus(path: Path | None = None) -> list[ContentItem]:
    """Parse the curated-media markdown into ContentItem records.

    Returns an empty list if the file is missing, and never raises on a
    malformed bullet. Callers wanting to know what was skipped should use
    :func:`load_corpus_with_report`.
    """
    items, _report = load_corpus_with_report(path)
    return items


def categories(items: list[ContentItem]) -> list[str]:
    """Distinct categories in corpus order."""
    seen: list[str] = []
    for item in items:
        if item.category not in seen:
            seen.append(item.category)
    return seen
