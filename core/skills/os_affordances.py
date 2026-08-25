"""General macOS affordance knowledge.

Aura controls OS settings the way a power user does: she knows which
goal-states she can reach and reaches for the reliable mechanism, instead
of clicking through System Settings or having a bespoke method per
setting.

This module is KNOWLEDGE + ROUTING, not a second execution engine.
Execution belongs to the one canonical owner, ``OSSettingsAdapter``
(``core/capabilities/os_settings.py``) — booted at capability start, with
rollback and governed receipts. Each entry here declares, for one OS
goal-state:
  - how to recognize an objective asks for it (``extract``),
  - which adapter methods set and read it (``setter`` / ``getter``),
  - how to translate the canonical value into the adapter's argument,
  - how a read-back confirms the goal-state.

Adding a controllable setting is a registry entry — never a new method, a
new derivation branch, or a new verify branch. The derivation and the
``system_control`` executor that use this registry are domain-agnostic:
they never name "wallpaper".
"""
from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.language.concepts import extract_object_description, object_class_pattern
from core.runtime.os_automation_effects import extract_target_paths

_VISUAL_ASSET_RE = object_class_pattern("image")


@dataclass(frozen=True)
class OSAffordance:
    """One thing Aura knows how to do to the operating system, routed to
    the canonical OSSettingsAdapter for execution."""

    domain: str
    summary: str  # plain-words self-knowledge ("I can change the desktop wallpaper")
    value_kind: str  # "image" | "toggle" | "level"
    extract: Callable[[str], str | None]  # objective -> canonical value, or None
    getter: str  # OSSettingsAdapter method that reports the current state
    setter: str  # OSSettingsAdapter method that applies the change
    to_setter_arg: Callable[[str], Any]  # canonical value -> setter argument
    confirms: Callable[[str, str], bool]  # (read-back, canonical value) -> goal met?
    needs_image: bool = False  # value is a topic to fetch into an image file first


# ── value extractors ───────────────────────────────────────────────────

def _extract_wallpaper(text: str) -> str | None:
    """Topic for a wallpaper/background change.

    Covers both technical phrasing ("set wallpaper") and ordinary user
    phrasing ("make this eagle my background") while returning only the visual
    topic. The executor still goes through the generic ``system_control`` path.
    """
    setting_change = re.search(
        r"\b(?:change|set|update|make|use|apply)\b",
        text,
        flags=re.IGNORECASE,
    )
    setting_surface = re.search(
        r"\b(?:my\s+)?(?:desktop\s+)?(?:wallpaper|background)\b",
        text,
        flags=re.IGNORECASE,
    )
    local_paths = extract_target_paths(text, require_file_intent=False)
    if setting_change and setting_surface and local_paths:
        # A dot is ordinary path syntax, not a sentence boundary. The previous
        # combined span stopped at `.jpg`, which made every extension-bearing
        # local image invisible despite the path parser already recognizing it.
        return local_paths[0]

    image_topic_match = re.search(
        rf"\b(?:find|search|look\s+up)\b[^.;\n]{{0,80}}?\b{_VISUAL_ASSET_RE}\s+of\s+([^.;,\n]+)"
        rf"[^.;\n]{{0,100}}?\b(?:make|set|use)\b[^.;\n]{{0,50}}?\b(?:my\s+)?(?:wallpaper|desktop\s+background|background)\b",
        text,
        flags=re.IGNORECASE,
    )
    direct_match = re.search(
        r"\b(?:change|set|update|make)\b[^.;\n]{0,50}?\b(?:wallpaper|desktop\s+background|background)\b"
        r"(?:\s+(?:to|into|with|as)\b)?(?:\s+(?:a|an|the)\b)?\s*([^.;,\n]*)",
        text,
        flags=re.IGNORECASE,
    )

    def _clean(candidate: str) -> str | None:
        query = re.sub(r"\bfrom\s+(?:online|the\s+(?:internet|web))\b.*$", "", candidate, flags=re.IGNORECASE)
        query = re.sub(r"\b(?:and|then|also|please|online)\b.*$", "", query, flags=re.IGNORECASE)
        query = re.sub(rf"\b{_VISUAL_ASSET_RE}\b.*$", "", query, flags=re.IGNORECASE)
        query = re.sub(r"^(?:a|an|the|cool)\s+", "", query.strip(" ,?.!"), flags=re.IGNORECASE)
        query = query.strip(" ,?.!")[:120]
        if not query or query.lower() in {"it", "this", "that", "one"}:
            return None
        return query

    if image_topic_match:
        topic = _clean(image_topic_match.group(1))
        if topic:
            return topic
    described_object = extract_object_description(
        text,
        "image",
        action_phrases=("find", "search", "look up", "get", "download", "fetch"),
    )
    if described_object:
        topic = _clean(described_object)
        if topic:
            return topic
    if not direct_match:
        return None
    topic = _clean(direct_match.group(1))
    if topic:
        return topic

    # The target is a pronoun: "find an orca image online, set IT as my
    # wallpaper". `_clean` rightly refuses to treat "it" as a search topic, but
    # refusing is not the same as not knowing — the referent is in the same
    # sentence. Resolving it is the difference between the wallpaper leg
    # planning at all and being silently dropped from the objective, which is
    # what happened live: the chain reported success and the desktop picture
    # never changed.
    referent = re.search(
        rf"\b{_VISUAL_ASSET_RE}\s+of\s+(?:a|an|the)?\s*([^.;,\n]+)",
        text,
        flags=re.IGNORECASE,
    ) or re.search(
        rf"\b(?:a|an|some)\s+([A-Za-z][\w'-]{{2,40}})\s+{_VISUAL_ASSET_RE}\b",
        text,
        flags=re.IGNORECASE,
    )
    if referent:
        return _clean(referent.group(1))
    return None


def _extract_dark_mode(text: str) -> str | None:
    """'turn on dark mode' -> true; 'light mode'/'turn off dark mode' -> false."""
    lowered = str(text or "").lower()
    if re.search(r"\b(?:turn off|disable|exit|leave|switch off)\b[^.;\n]{0,16}\bdark mode\b", lowered) \
            or re.search(r"\b(?:light mode|turn on light)\b", lowered):
        return "false"
    if re.search(r"\bdark mode\b", lowered):
        return "true"
    return None


def _extract_volume(text: str) -> str | None:
    """'set the volume to 30(%)' -> '30' (clamped 0-100)."""
    match = re.search(
        r"\bvolume\b[^.;\n]{0,16}?(\d{1,3})\s*%?",
        text,
        flags=re.IGNORECASE,
    ) or re.search(
        r"(\d{1,3})\s*%?[^.;\n]{0,10}?\bvolume\b",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    return str(max(0, min(100, int(match.group(1)))))


# ── read-back confirmations ────────────────────────────────────────────

def _confirm_wallpaper(readback: str, value: str) -> bool:
    return bool(value) and Path(value).name.lower() in str(readback or "").lower()


def _confirm_dark_mode(readback: str, value: str) -> bool:
    want = "dark" if value == "true" else "light"
    return want in str(readback or "").lower()


def _confirm_volume(readback: str, value: str) -> bool:
    try:
        # macOS quantizes volume to 16 steps (~6.25% each), so the read-back
        # may differ from the request by up to one step.
        return abs(int(str(readback).strip()) - int(value)) <= 5
    except (TypeError, ValueError):
        return False


# ── the registry ───────────────────────────────────────────────────────

_AFFORDANCES: tuple[OSAffordance, ...] = (
    OSAffordance(
        domain="wallpaper",
        summary="Change the desktop wallpaper to an image (e.g. one found online).",
        value_kind="image",
        extract=_extract_wallpaper,
        needs_image=True,
        getter="get_wallpaper",
        setter="set_wallpaper",
        to_setter_arg=lambda value: value,  # resolved file path
        confirms=_confirm_wallpaper,
    ),
    OSAffordance(
        domain="dark_mode",
        summary="Turn macOS dark mode on or off.",
        value_kind="toggle",
        extract=_extract_dark_mode,
        getter="get_appearance_mode",
        setter="set_appearance_mode",
        to_setter_arg=lambda value: "dark" if value == "true" else "light",
        confirms=_confirm_dark_mode,
    ),
    OSAffordance(
        domain="volume",
        summary="Set the system output volume to a level (0-100).",
        value_kind="level",
        extract=_extract_volume,
        getter="get_volume",
        setter="set_volume",
        to_setter_arg=lambda value: int(value),
        confirms=_confirm_volume,
    ),
)

_BY_DOMAIN: dict[str, OSAffordance] = {a.domain: a for a in _AFFORDANCES}


def get_affordance(domain: str) -> OSAffordance | None:
    return _BY_DOMAIN.get(str(domain or "").strip().lower())


def detect_os_settings(objective: str) -> list[tuple[str, str]]:
    """Generic scan: which known OS settings does this objective request?

    Returns (domain, canonical_value) pairs. The loop never names a
    specific setting — every affordance is consulted the same way, so a
    new entry is recognized for free.
    """
    text = str(objective or "")
    found: list[tuple[str, str]] = []
    for affordance in _AFFORDANCES:
        value = affordance.extract(text)
        if value:
            found.append((affordance.domain, value))
    return found


def describe_affordances() -> list[str]:
    """Self-knowledge surface: the OS goal-states Aura knows how to reach."""
    return [a.summary for a in _AFFORDANCES]


def validate_value(affordance: OSAffordance, value: str) -> str | None:
    """Constrain the canonical value to its kind before it is used. Returns
    the canonical value, or None if invalid."""
    text = str(value or "").strip()
    if affordance.value_kind == "toggle":
        return text if text in {"true", "false"} else None
    if affordance.value_kind == "level":
        return str(max(0, min(100, int(text)))) if text.isdigit() else None
    if affordance.value_kind == "image":
        return text or None
    return text or None
