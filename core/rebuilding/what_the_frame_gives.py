"""Features the frame already does, given to a program by code: their parts and their checks, nothing asked of her model.

The frame writes and reads the files people exchange documents as
(core/rebuilding/document_formats.js). A program that what is written about it,
or the person, says saves or opens those files gets a feature for each, wired
to the frame by code and checked by code, in place of a part her model would
have to get right: a file format is a fact, not a guess. A feature her model
listed for the same format is given the frame's part under its own name.

Which formats are meant is read from the words, as a person reads them: the
format named (".docx", "DOCX", "OpenDocument", "Rich Text Format", "Markdown",
"plain text") in a sentence about saving, exporting, opening or a file format.
"""
from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from core.rebuilding.checks_a_person_makes import Check
from core.rebuilding.the_program_as_built import Part
from core.rebuilding.what_a_program_does import Feature

__all__ = ["Given", "formats_named", "what_the_frame_gives"]

#: Each kind the frame writes, and the words that name it.
_NAMED_BY: dict[str, tuple[str, ...]] = {
    "docx": (r"\.?docx\b", r"office open xml", r"word documents?\b", r"\.doc\b"),
    "odt": (r"\.?odt\b", r"opendocument"),
    "rtf": (r"\.?rtf\b", r"rich text"),
    "html": (r"\.?html?\b", r"web pages?\b"),
    "md": (r"\bmarkdown\b", r"\.md\b"),
    "txt": (r"\.?txt\b", r"plain text"),
}

#: What a sentence must be about for a format named in it to be one the program saves or opens.
_ABOUT_FILES = re.compile(r"\b(sav\w*|export\w*|open\w*|import\w*|format\w*|file\w*|convert\w*|download\w*|writ\w*|read\w*)\b", re.I)

_SHOWN = {"docx": "Word Document (.docx)", "odt": "OpenDocument Text (.odt)", "rtf": "Rich Text (.rtf)",
          "html": "Web Page (.html)", "md": "Markdown (.md)", "txt": "Plain Text (.txt)"}


@dataclass
class Given:
    feature: Feature
    part: Part
    checks: list[Check]


def formats_named(texts: Sequence[str]) -> list[str]:
    """The kinds the frame writes that ``texts`` name in a sentence about files, in the frame's order."""
    found: set[str] = set()
    for text in texts:
        for sentence in re.split(r"(?<=[.!?;])\s+|\n", str(text or "")):
            if not _ABOUT_FILES.search(sentence):
                continue
            lowered = sentence.lower()
            found.update(kind for kind, ways in _NAMED_BY.items() if any(re.search(way, lowered) for way in ways))
    return [kind for kind in _NAMED_BY if kind in found]


def _for_kind(feature: Feature, kind: str) -> bool:
    words = f"{feature.name} {feature.how}".lower()
    return any(re.search(way, words) for way in _NAMED_BY[kind]) and bool(re.search(r"\b(sav|export|download|convert)", words))


def _part(label: str, kind: str) -> Part:
    code = f"app.command({{label: {json.dumps(label)}, menu: \"File\", run: () => app.formats.save({json.dumps(kind)})}});"
    return Part(label, code, [label])


def _check(label: str, kind: str) -> Check:
    return Check.model_validate({
        "feature": label,
        "rule": f"it saves the document as a .{kind} file holding what was written",
        "steps": [{"do": "type", "value": "Saved as a file"}, {"do": "click", "target": label}],
        "expect": [{"see": "download", "target": "Saved as a file", "value": f".{kind}"}],
    })


def what_the_frame_gives(features: list[Feature], sources: Sequence[Any], asked: str) -> tuple[list[Feature], dict[str, Given]]:
    """The features with one for saving each format named, and what the frame gives each such feature, by its name."""
    kinds = formats_named([asked, *(getattr(s, "text", "") for s in sources)])
    given: dict[str, Given] = {}
    features = list(features)
    for kind in kinds:
        mine = next((f for f in features if _for_kind(f, kind)), None)
        if mine is None:
            mine = Feature(name=f"Save as {_SHOWN[kind]}", how=f"File menu, Save as {_SHOWN[kind]}",
                           shows=f"a .{kind} file of the document is saved", place="File", weight=2)
            features.append(mine)
        given[mine.name] = Given(mine, _part(mine.name, kind), [_check(mine.name, kind)])
    return features, given
