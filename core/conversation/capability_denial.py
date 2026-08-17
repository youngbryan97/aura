"""She should not deny a capability the registry says she has.

LIVE 2026-08-17: "how many python files live in core/introspection?" was
answered "I don't have file system access or the ability to count files in a
directory." Eight filesystem-capable skills are registered and enabled at that
moment — file_operation, computer_use, desktop_task, os_automation,
os_manipulation, run_code, code_repl, improve_own_code — and the same question
phrased "are in" instead of "live in" was answered exactly, with filenames.

A wrong denial is worse than a wrong attempt. It teaches the person that the
product cannot do something it can do, and they stop asking. That is the
failure Bryan named: "Last thing we want is for me or someone else to have her
try to do something only to find out she cant do it."

The check is against the REGISTRY, not against a list of things I believe she
can do. Skills register themselves; this reads what registered. When the
registry is empty or unreadable the answer is "no opinion" rather than an
assumption in either direction — claiming a capability she lacks would be the
same defect pointing the other way.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

__all__ = [
    "CapabilityDenial",
    "denied_registered_capabilities",
]

#: A denial of ability, not a refusal on principle. "I won't help with that"
#: and "I'd rather not" are choices and must not be overridden here.
_DENIAL_RE = re.compile(
    r"\b(?:"
    # Words sit between "have" and the noun in the sentence that motivated
    # this: "I don't have FILE SYSTEM access". Requiring them to be adjacent
    # made the check miss its own live case.
    r"i\s+(?:do\s+not|don'?t)\s+have\s+(?:\w+\s+){0,3}?"
    r"(?:ability|access|capability|means|way|permission)"
    r"|i\s+(?:can'?t|cannot|am\s+unable\s+to)\s+"
    r"|i\s+have\s+no\s+(?:access|ability|way|means)"
    r"|(?:that\s+)?(?:is|'s)\s+(?:not\s+something|beyond)\s+i\s+can"
    r")",
    re.IGNORECASE,
)

#: Subject → the concrete thing being denied. Each maps to whichever registered
#: skills could actually do it; the mapping is by capability, not by name, so a
#: renamed skill does not silently empty a row.
_DENIAL_SUBJECTS: tuple[tuple[Any, str, tuple[str, ...]], ...] = (
    (
        re.compile(
            r"\b(?:file\s?system|file\s+access|files?\b|directory|directories|folder)",
            re.IGNORECASE,
        ),
        "read the filesystem",
        ("file_operation", "computer_use", "desktop_task", "run_code", "code_repl"),
    ),
    (
        re.compile(r"\b(?:run|execute|write)\s+(?:code|python|a\s+script)", re.IGNORECASE),
        "run code",
        ("run_code", "code_repl", "improve_own_code"),
    ),
    (
        re.compile(r"\b(?:search\s+the\s+web|web\s+search|browse|internet|online)", re.IGNORECASE),
        "search the web",
        ("web_search", "sovereign_browser", "free_search", "grounded_search"),
    ),
    (
        re.compile(r"\b(?:screen|display|what.{0,12}on\s+(?:my|the)\s+screen)", re.IGNORECASE),
        "read the screen",
        ("computer_use", "desktop_task", "os_manipulation"),
    ),
    (
        re.compile(r"\bclipboard\b", re.IGNORECASE),
        "use the clipboard",
        ("os_automation", "computer_use", "desktop_task"),
    ),
)


@dataclass(frozen=True, slots=True)
class CapabilityDenial:
    """A denial the registry contradicts."""

    subject: str
    sentence: str
    skills: tuple[str, ...]


def _enabled_skill_names(engine: Any) -> set[str]:
    try:
        skills = getattr(engine, "skills", None) or {}
        names: set[str] = set()
        for name, meta in skills.items():
            if getattr(meta, "enabled", True):
                names.add(str(name))
        return names
    except (AttributeError, TypeError, ValueError):
        return set()


def denied_registered_capabilities(
    reply: Any, engine: Any = None
) -> tuple[CapabilityDenial, ...]:
    """Denials in this reply that the live registry contradicts."""

    text = str(reply or "")
    if not text.strip():
        return ()
    if engine is None:
        try:
            from core.capability_engine import CapabilityEngine

            engine = CapabilityEngine()
        except (ImportError, RuntimeError, TypeError, ValueError):
            return ()
    available = _enabled_skill_names(engine)
    if not available:
        return ()  # nothing registered: no opinion, rather than an assumption

    found: list[CapabilityDenial] = []
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", text):
        if not _DENIAL_RE.search(sentence):
            continue
        for pattern, subject, skills in _DENIAL_SUBJECTS:
            if not pattern.search(sentence):
                continue
            present = tuple(sorted(s for s in skills if s in available))
            if present:
                found.append(
                    CapabilityDenial(
                        subject=subject, sentence=sentence.strip(), skills=present
                    )
                )
                break
    return tuple(found)
