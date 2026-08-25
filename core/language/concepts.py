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
    "extract_object_description",
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
_WORD_SPAN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_+#'\-]*")
_PHRASE_LEADERS = frozenset(
    {
        "a",
        "an",
        "any",
        "for",
        "me",
        "please",
        "some",
        "the",
        "us",
    }
)


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


def extract_object_description(
    text: str,
    representative: str,
    *,
    action_phrases: tuple[str, ...] = (),
    max_words: int = 10,
) -> str:
    """Return the noun phrase that describes a typed object mention.

    The object class supplies the head noun (``image``, ``photo``, ...).
    This function preserves the complete adjacent description in either
    common construction: ``image of a blue whale`` or ``a blue whale image``.
    Callers may provide the action phrases that introduce the object so words
    belonging to the request frame are not mistaken for modifiers.

    It is a bounded constituent reader, not a topic-specific vocabulary: the
    same code carries ``blue whale``, ``red panda``, and an unseen multiword
    subject without knowing any of those entities.
    """

    source = str(text or "")
    members = {fold_noun(member) for member in object_class_of(representative)}
    words = list(_WORD_SPAN_RE.finditer(source))
    if not source or not members or not words:
        return ""

    action_tokens = tuple(
        tuple(fold_noun(token) for token in _WORD_SPAN_RE.findall(phrase))
        for phrase in action_phrases
        if phrase
    )
    folded = [fold_noun(match.group(0)) for match in words]

    def _clause_left(index: int) -> int:
        left = 0
        for candidate in range(index - 1, -1, -1):
            gap = source[words[candidate].end() : words[candidate + 1].start()]
            if re.search(r"[.;!?\n]", gap):
                left = candidate + 1
                break
        return left

    def _clause_right(index: int) -> int:
        right = len(words)
        for candidate in range(index, len(words) - 1):
            gap = source[words[candidate].end() : words[candidate + 1].start()]
            if re.search(r"[.;!?\n]", gap):
                right = candidate + 1
                break
        return right

    for head_index, head in enumerate(folded):
        if head not in members:
            continue

        right = _clause_right(head_index)
        if head_index + 1 < right and folded[head_index + 1] == "of":
            start = head_index + 2
            if start >= right:
                continue
            end = min(right, start + max(1, max_words))
            return source[words[start].start() : words[end - 1].end()].strip()

        left = _clause_left(head_index)
        start = max(left, head_index - max(1, max_words))
        for action in action_tokens:
            width = len(action)
            if not width:
                continue
            for candidate in range(left, head_index - width + 1):
                if tuple(folded[candidate : candidate + width]) == action:
                    start = max(start, candidate + width)

        while start < head_index and folded[start] in _PHRASE_LEADERS:
            start += 1
        if start < head_index:
            return source[words[start].start() : words[head_index - 1].end()].strip()
    return ""
