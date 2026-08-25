"""Shared lexical concepts for Aura's language substrate.

These are bounded semantic classes, not per-feature synonym lists. A visual
asset means the same thing to capability routing, desktop planning, and an OS
affordance, so those readers must consume one class rather than maintain
different spellings of it.
"""

from __future__ import annotations

import re
from functools import lru_cache

__all__ = [
    "OBJECT_CLASSES",
    "fold_noun",
    "mentions_object_class",
    "object_class_of",
    "object_class_pattern",
]


OBJECT_CLASSES: tuple[frozenset[str], ...] = (
    frozenset(
        {
            "code", "script", "snippet", "program", "programme", "python",
            "repl", "interpreter", "sandbox", "expression", "function",
            "test", "tests", "testcase", "testcases",
        }
    ),
    frozenset(
        {
            "web", "online", "internet", "google", "browser", "site",
            "website", "url", "page",
        }
    ),
    frozenset(
        {
            "image", "images", "picture", "pictures", "photo", "photos",
            "photograph", "photographs", "illustration", "illustrations",
            "artwork", "drawing", "painting", "diagram", "portrait",
            "portraits", "snapshot", "snapshots", "graphic", "graphics",
        }
    ),
    frozenset({"screen", "display", "desktop", "window", "monitor"}),
    frozenset(
        {
            "file", "files", "document", "folder", "directory", "path",
            "repo", "repository", "workspace", "filesystem",
        }
    ),
    frozenset({"memory", "memories", "recollection", "note", "notes"}),
    frozenset({"time", "clock", "date", "hour", "day"}),
    frozenset({"email", "mail", "message", "messages", "text", "dm"}),
    frozenset({"voice", "speech", "audio", "sound", "microphone"}),
    frozenset({"package", "library", "dependency", "module"}),
    # Engineering artifacts stay apart from visual references. A schematic is
    # computed from a model; routing it as a picture invites a plausible image
    # of a machine instead of an executable design.
    frozenset(
        {
            "schematic", "schematics", "blueprint", "blueprints", "assembly",
            "subassembly", "exploded", "cutaway", "cad", "bom", "bracket",
            "enclosure", "chassis", "linkage", "mechanism", "gearbox",
            "circuit", "wiring", "harness", "pcb", "manifold", "housing",
            "fixture", "jig",
        }
    ),
)

_WORD_RE = re.compile(r"[a-z0-9_+#]+(?:\.[a-z0-9_+#]+)*")


@lru_cache(maxsize=4096)
def fold_noun(word: str) -> str:
    """Return the conservative singular form used by concept matching."""
    lowered = str(word or "").strip().lower()
    if len(lowered) > 4 and lowered.endswith("ies"):
        return lowered[:-3] + "y"
    if len(lowered) > 3 and lowered.endswith("es") and lowered[-3] in "sxzh":
        return lowered[:-2]
    if len(lowered) > 3 and lowered.endswith("s") and not lowered.endswith("ss"):
        return lowered[:-1]
    return lowered


@lru_cache(maxsize=4096)
def object_class_of(word: str) -> frozenset[str]:
    """Return every lexical form in the bounded concept named by ``word``."""
    lowered = str(word or "").strip().lower()
    if not lowered:
        return frozenset()
    folded = fold_noun(lowered)
    for members in OBJECT_CLASSES:
        if lowered in members or folded in {fold_noun(member) for member in members}:
            return members
    return frozenset()


@lru_cache(maxsize=256)
def object_class_pattern(representative: str) -> str:
    """Regex alternation for one concept, longest lexical form first."""
    members = object_class_of(representative)
    terms = members or frozenset({str(representative or "").strip().lower()})
    escaped = [re.escape(term) for term in sorted(filter(None, terms), key=len, reverse=True)]
    if not escaped:
        return r"(?!)"
    return "(?:" + "|".join(escaped) + ")"


def mentions_object_class(text: str, representative: str) -> bool:
    """Whether ``text`` names any member of a bounded object concept."""
    members = {fold_noun(member) for member in object_class_of(representative)}
    if not members:
        return False
    return any(
        fold_noun(word) in members
        for word in _WORD_RE.findall(str(text or "").casefold())
    )
