"""Everything she can do, gathered from every register that holds some of it.

Self-knowledge was sourced from the skill registry alone. She also answers
turns from deterministic readers — arithmetic, text operations, reading a named
file — which are not skills and were therefore invisible to her. Asked "can you
reverse a string for me", a capability she has, the honest-sounding answer came
back "nothing in the capability registry matches", because nothing did.

A register of capabilities declares itself here. Adding one teaches her about
that whole class at once, and nothing downstream has to learn its name.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

__all__ = [
    "CapabilityRecord",
    "register_source",
    "registered_sources",
    "all_capabilities",
    "sources_fingerprint",
]

_SOURCE_ERRORS = (ImportError, AttributeError, RuntimeError, TypeError, ValueError)


@dataclass(frozen=True)
class CapabilityRecord:
    """One capability, in the shape the lexicon reads."""

    name: str
    description: str = ""
    enabled: bool = True
    class_name: str = ""
    trigger_patterns: tuple[Any, ...] = field(default_factory=tuple)
    #: Which register this came from, so an answer can say where it looked.
    origin: str = ""


_SOURCES: dict[str, Callable[[Any], dict[str, CapabilityRecord]]] = {}


def register_source(
    name: str, loader: Callable[[Any], dict[str, CapabilityRecord]]
) -> None:
    """Declare a register of capabilities. `loader(engine)` returns its records."""
    _SOURCES[str(name)] = loader


def registered_sources() -> tuple[str, ...]:
    return tuple(_SOURCES)


def _skill_records(engine: Any = None) -> dict[str, CapabilityRecord]:
    if engine is None:
        try:
            from core.capability_engine import CapabilityEngine

            engine = CapabilityEngine()
        except _SOURCE_ERRORS:
            return {}
    skills = dict(getattr(engine, "skills", None) or {})
    records: dict[str, CapabilityRecord] = {}
    for name, meta in skills.items():
        records[str(name)] = CapabilityRecord(
            name=str(name),
            description=str(getattr(meta, "description", "") or ""),
            enabled=bool(getattr(meta, "enabled", True)),
            class_name=str(getattr(meta, "class_name", "") or ""),
            trigger_patterns=tuple(getattr(meta, "trigger_patterns", None) or ()),
            origin="skill_registry",
        )
    return records


def _reader_records(engine: Any = None) -> dict[str, CapabilityRecord]:
    """The deterministic readers, which answer without asking the model."""
    try:
        from core.conversation.turn_ownership import registered_readers
    except _SOURCE_ERRORS:
        return {}
    records: dict[str, CapabilityRecord] = {}
    for reader in registered_readers():
        records[str(reader.name)] = CapabilityRecord(
            name=str(reader.name),
            description=(
                f"Computed directly rather than generated: {reader.answers}."
            ),
            enabled=True,
            class_name=str(reader.function),
            # The vocabulary a reader answers to, taken from the reader itself.
            # A description written here would be a second place to keep the
            # truth, and the second place is the one that goes stale.
            trigger_patterns=reader.vocabulary(),
            origin="deterministic_readers",
        )
    return records


register_source("skill_registry", _skill_records)
register_source("deterministic_readers", _reader_records)


def all_capabilities(engine: Any = None) -> dict[str, CapabilityRecord]:
    """Every declared capability, from every declared register."""
    merged: dict[str, CapabilityRecord] = {}
    for loader in _SOURCES.values():
        try:
            found = loader(engine) or {}
        except _SOURCE_ERRORS:
            continue
        for name, record in found.items():
            # First register to claim a name keeps it: a skill and a reader
            # sharing a name are the same capability described twice, and the
            # skill carries the richer description.
            merged.setdefault(str(name), record)
    return merged


def sources_fingerprint(engine: Any = None) -> str:
    """Identity of the whole capability surface, so caches follow the build."""
    records = all_capabilities(engine)
    return "|".join(
        f"{name}:{int(bool(record.enabled))}" for name, record in sorted(records.items())
    )
