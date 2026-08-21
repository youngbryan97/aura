"""How many things she can do, and what they are called, read off the build.

LIVE 2026-08-19: "How many skills are registered in your capability engine
right now, and name three of them exactly as they're registered." The reply
was "39 .py files. I listed the directory rather than estimating" — an honest
account of a measurement of the wrong thing. Files in a source directory are
not entries in a registry; the registry held 82.

Counting the files was the closest thing to an answer available, because no
reading served the registry itself. This one does, and it counts every
register of capability rather than the skills alone, so the number cannot go
stale the next time a capability arrives that is not a skill.
"""

from __future__ import annotations

from typing import Any

from core.conversation.word_markers import names_any

__all__ = [
    "CAPABILITY_INVENTORY_HEADER",
    "asks_what_she_can_do",
    "capability_inventory_block",
]

CAPABILITY_INVENTORY_HEADER = "## EVERY CAPABILITY REGISTERED IN THIS BUILD"

#: Asking for the inventory: its size, its contents, or both.
_ASKS_INVENTORY_MARKERS = (
    "what can you do",
    "what are you able to do",
    "what are you capable of",
    "what do you do",
    "list your capabilities",
    "list your skills",
    "list your tools",
    "list out your",
    "what skills do you have",
    "what tools do you have",
    "what capabilities do you have",
    "how many skills",
    "how many tools",
    "how many capabilities",
    "how many things can you do",
    "name three of them",
    "your capability engine",
    "registered in your",
    "what's in your toolkit",
    "what is in your toolkit",
    "run through your capabilities",
    "inventory of your",
)

#: How many names to give when they ask for the list rather than the number.
_NAMES_SHOWN = 12


def asks_what_she_can_do(prompt: Any) -> bool:
    """True when the turn asks how many capabilities exist, or which."""
    text = str(prompt or "").strip().lower()
    if not text:
        return False
    return names_any(text, _ASKS_INVENTORY_MARKERS)


def capability_inventory_block(prompt: Any = "", engine: Any = None) -> str:
    """The count and the registered names, from the live registers."""
    if prompt and not asks_what_she_can_do(prompt):
        return ""
    try:
        from core.self.capability_sources import all_capabilities, registered_sources

        records = all_capabilities(engine)
    except (ImportError, RuntimeError, TypeError, ValueError):
        return ""
    if not records:
        return ""

    by_origin: dict[str, list[str]] = {}
    for name, record in sorted(records.items()):
        by_origin.setdefault(record.origin or "unknown", []).append(name)

    lines = [
        f"{len(records)} capabilities are registered in this process right now, "
        f"across {len(registered_sources())} registers:",
    ]
    for origin, names in sorted(by_origin.items()):
        shown = ", ".join(names[:_NAMES_SHOWN])
        remainder = len(names) - len(names[:_NAMES_SHOWN])
        tail = f", and {remainder} more" if remainder > 0 else ""
        lines.append(f"- {origin} ({len(names)}): {shown}{tail}")
    lines.append(
        "These are the registered names, exactly as they appear. Counting "
        "source files in a directory answers a different question and gives a "
        "different number."
    )
    return "\n".join(lines)
