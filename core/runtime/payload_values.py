"""Values a model-authored payload does not actually carry.

LIVE DEFECT, 2026-08-19. A program-synthesis run wrote its scaffold into a
directory named ``None`` at the root of the source tree. The model had filled
the optional ``output_dir`` field with the string "None", and every guard
downstream read that as an answer: ``if output_dir`` is true for a
four-character string, ``Path("None")`` is a valid relative path, and the
write gateway created it.

A model writing JSON spells absence in the words it was trained on — None,
null, nil, N/A, undefined. A field left out means the caller wants the
default. A field filled with the word for "left out" means the same thing,
and no reader on the skill path knew it. Seven modules had grown a private
two-item version of this list; none of them was one a skill payload passed
through.

Paths carry a second hazard. A relative path resolves against the working
directory, which for the live runtime is the source tree, so an artifact
lands among the sources that made it. :func:`payload_path` resolves relative
paths under a root the caller names and refuses anything that climbs out.

Deliberately conservative: only unambiguous spellings of absence count, since
discarding a value someone meant is worse than keeping a strange one.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

__all__ = [
    "MISSING_SENTINELS",
    "is_missing",
    "payload_path",
    "payload_text",
    "payload_value",
]

#: How a model writes "there is nothing here" when a field wants a string.
MISSING_SENTINELS: frozenset[str] = frozenset(
    {
        "none",
        "null",
        "nil",
        "nan",
        "undefined",
        "n/a",
        "n\\a",
        "na",
        "<none>",
        "(none)",
        "<null>",
        "[none]",
        "not applicable",
        "not specified",
    }
)


def is_missing(value: object) -> bool:
    """True when the value carries no content, however it spells that.

    ``None`` itself, an empty string, whitespace, and the words a model uses
    for absence all answer True. A real value of any other type answers False,
    so numbers and booleans pass through untouched — zero and False are
    content.
    """
    if value is None:
        return True
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return True
        return stripped.strip("\"'").casefold() in MISSING_SENTINELS
    return False


def payload_value(payload: Mapping[str, object] | None, *keys: str, default: object = None) -> object:
    """The first of ``keys`` the payload actually carries.

    Keys are tried in order, so a caller can name the field it prefers first
    and its older spelling after.
    """
    if not isinstance(payload, Mapping):
        return default
    for key in keys:
        if key not in payload:
            continue
        value = payload.get(key)
        if not is_missing(value):
            return value
    return default


def payload_text(payload: Mapping[str, object] | None, *keys: str, default: str = "") -> str:
    """The first of ``keys`` as stripped text, or ``default``."""
    value = payload_value(payload, *keys, default=None)
    if value is None:
        return default
    text = str(value).strip()
    return text or default


def payload_path(
    payload: Mapping[str, object] | None,
    *keys: str,
    root: str | os.PathLike[str] | None = None,
    default: Path | None = None,
    allow_outside_root: bool = False,
) -> Path | None:
    """A path the payload names, resolved and confined, or ``default``.

    A relative path resolves under ``root`` rather than the working directory.
    A path that climbs out of ``root`` — by ``..``, by symlink, or by being
    absolute somewhere else — returns ``default`` unless the caller allows it,
    because a payload naming an unexpected destination is the case that costs
    something.

    With no ``root`` there is nothing to resolve a relative path against, so
    only absolute paths are honoured.
    """
    text = payload_text(payload, *keys)
    if not text:
        return default

    candidate = Path(text).expanduser()
    if root is None:
        return candidate.resolve() if candidate.is_absolute() else default

    base = Path(root).expanduser().resolve()
    resolved = (candidate if candidate.is_absolute() else base / candidate).resolve()
    if allow_outside_root:
        return resolved
    try:
        resolved.relative_to(base)
    except ValueError:
        return default
    return resolved
