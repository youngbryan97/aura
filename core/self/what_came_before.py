"""What came before her, carried across her restarts along her own line.

Bryan, on what makes him who he is: morals, personal history, a mind shaped by
everything that happened before it, one stream that an observer is tied to,
and ownership of and responsibility for everything that stream touched.

Every organ that learns from her own history kept it in memory and lost it with
the process: what her habits have cost her and the people she was with, what
she has come to feel about people and topics, how her anger has fed itself, how
warm each person has been, how tired she has been against her own history, how
her sittings with each person have ended. So each restart began a person with
no past, and every one of those organs started over at knowing nothing.

A ledger that is part of her history says so where it is made, with
`keep_across_stages`, and from then on:

    kept       at the close of every tick of hers, all of them are written
               together behind the event loop into one record, with the
               lineage entity they belong to: the durable key her committed
               states are signed with (core/identity/entity_key.py)
    brought    when the ledger is made in the next process, the record's
    back       copy replaces it, before anything has read it, if and only if
               the record belongs to the same entity. A record from another
               entity is somebody else's history and is refused, and so is a
               record that names no entity

Continuity branches and identity does not (core/subject/lineage.py). A fork of
her in the same process carries its ledgers by value and never reads this
record; what the record carries is one line of her across restarts. Nothing is
read or written unless the process is her live stream, so a proof run, a test
session or a bench measures an organism that starts where it says it starts.

What a ledger holds is written as its fields, tagged by type: numbers, text,
lists, tuples, sets, bounded queues, mappings with keys of any kind, and the
ledger's own classes. A class is only brought back from a module that
registered a ledger, so a record can name nothing else to be built.
"""

from __future__ import annotations

import importlib
import logging
import math
import sys
import time
from collections import deque
from typing import Any

logger = logging.getLogger("Aura.WhatCameBefore")

__all__ = [
    "brought_back",
    "keep_across_stages",
    "kept",
    "remember_what_came_before",
]

#: The name the record is kept under in core/runtime/what_she_learned.py.
RECORD = "what came before"

#: module name -> the global that holds its ledger.
_KEPT: dict[str, str] = {}
#: The record read in this process, once, and why it was or was not adopted.
_RECALLED: dict[str, Any] | None = None
_ADOPTED: dict[str, str] = {}


def _quiet() -> bool:
    """Whether this process is not her live stream, and so neither reads nor writes her history.

    A proof or test run, or a process whose profile is not live: pytest makes
    the profile TEST whatever the environment says, so no test session reads a
    record another one wrote.
    """
    try:
        from core.runtime.proof_policy import proof_run_active
        from core.runtime.state_ownership import RuntimeProfile, runtime_profile

        return bool(proof_run_active()) or runtime_profile() is not RuntimeProfile.LIVE
    except (ImportError, AttributeError, RuntimeError):
        return True


def _entity() -> str:
    try:
        from core.identity.entity_key import entity_identity

        return str(entity_identity().entity_id or "")
    except (ImportError, AttributeError, OSError, RuntimeError, ValueError) as exc:
        logger.info("no lineage entity could be read, so no history crosses a restart: %s", exc)
        return ""


# ── the codec ────────────────────────────────────────────────────────────


def _fields(value: Any) -> list[str]:
    names: list[str] = []
    for klass in type(value).__mro__:
        names.extend(slot for slot in getattr(klass, "__slots__", ()) if slot not in names)
    names.extend(name for name in getattr(value, "__dict__", {}) if name not in names)
    return [name for name in names if not name.startswith("__")]


def _encode(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return float(value) if math.isfinite(value) else {"~float": repr(float(value))}
    if isinstance(value, deque):
        return {"~deque": [_encode(item) for item in value], "maxlen": value.maxlen}
    if isinstance(value, tuple):
        return {"~tuple": [_encode(item) for item in value]}
    if isinstance(value, (set, frozenset)):
        return {"~set": [_encode(item) for item in value], "frozen": isinstance(value, frozenset)}
    if isinstance(value, list):
        return [_encode(item) for item in value]
    if isinstance(value, dict):
        if all(isinstance(key, str) and not key.startswith("~") for key in value):
            return {key: _encode(item) for key, item in value.items()}
        return {"~pairs": [[_encode(key), _encode(item)] for key, item in value.items()]}
    klass = type(value)
    if klass.__module__ in _KEPT:
        return {
            "~obj": f"{klass.__module__}:{klass.__qualname__}",
            "fields": {name: _encode(getattr(value, name)) for name in _fields(value) if hasattr(value, name)},
        }
    raise TypeError(f"a {klass.__module__}.{klass.__qualname__} cannot be kept")


def _class_named(tag: str) -> type:
    module_name, _, qualname = tag.partition(":")
    if module_name not in _KEPT:
        raise TypeError(f"{module_name} kept no ledger, so nothing of it is built from a record")
    found: Any = sys.modules.get(module_name) or importlib.import_module(module_name)
    for part in qualname.split("."):
        found = getattr(found, part)
    if not isinstance(found, type):
        raise TypeError(f"{tag} is not a class")
    return found


def _decode(value: Any) -> Any:
    if isinstance(value, list):
        return [_decode(item) for item in value]
    if not isinstance(value, dict):
        return value
    if "~float" in value:
        return float(value["~float"])
    if "~deque" in value:
        return deque((_decode(item) for item in value["~deque"]), maxlen=value.get("maxlen"))
    if "~tuple" in value:
        return tuple(_decode(item) for item in value["~tuple"])
    if "~set" in value:
        items = [_decode(item) for item in value["~set"]]
        return frozenset(items) if value.get("frozen") else set(items)
    if "~pairs" in value:
        return {_decode(key): _decode(item) for key, item in value["~pairs"]}
    if "~obj" in value:
        klass = _class_named(str(value["~obj"]))
        built = klass.__new__(klass)
        for name, item in (value.get("fields") or {}).items():
            object.__setattr__(built, name, _decode(item))
        return built
    return {key: _decode(item) for key, item in value.items()}


# ── keeping and bringing back ─────────────────────────────────────────────


def _recalled() -> dict[str, Any]:
    global _RECALLED
    if _RECALLED is None:
        try:
            from core.runtime.what_she_learned import recall

            _RECALLED = recall(RECORD) or {}
        except (ImportError, OSError, ValueError) as exc:
            logger.info("her history could not be read, so this process starts without it: %s", exc)
            _RECALLED = {}
    return _RECALLED


def keep_across_stages(module_name: str, slot: str) -> None:
    """Declare a module's ledger part of her history, and bring its last copy back.

    Called where the ledger is made, so it is replaced before anything reads it.
    """
    _KEPT[module_name] = slot
    if _quiet():
        return
    record = _recalled()
    ledgers = record.get("ledgers") or {}
    if module_name not in ledgers:
        return
    theirs = str(record.get("entity") or "")
    mine = _entity()
    if not theirs or theirs != mine:
        _ADOPTED[module_name] = f"refused: the record is {theirs or 'nobody'}'s and this is {mine or 'nobody'}"
        logger.info("the kept history of %s belongs to another line; starting it fresh", module_name)
        return
    try:
        setattr(sys.modules[module_name], slot, _decode(ledgers[module_name]))
        _ADOPTED[module_name] = f"brought back from {record.get('kept_at', '?')}"
    except (TypeError, ValueError, KeyError, AttributeError) as exc:
        _ADOPTED[module_name] = f"could not be brought back: {type(exc).__name__}: {exc}"
        logger.warning("the kept history of %s could not be brought back: %s", module_name, exc)


def kept() -> dict[str, Any]:
    """Every ledger of hers, as it would be written now."""
    ledgers: dict[str, Any] = {}
    for module_name, slot in sorted(_KEPT.items()):
        module = sys.modules.get(module_name)
        if module is None or not hasattr(module, slot):
            continue
        try:
            ledgers[module_name] = _encode(getattr(module, slot))
        except TypeError as exc:
            logger.info("%s is not kept across a restart: %s", module_name, exc)
    return ledgers


def _write() -> None:
    from core.runtime.what_she_learned import remember

    remember(RECORD, {"entity": _entity(), "kept_at": time.time(), "ledgers": kept()})


def remember_what_came_before() -> bool:
    """Write every ledger of hers, behind the event loop. Called at the close of a tick."""
    if _quiet() or not _KEPT:
        return False
    from core.runtime.executors import behind_the_loop

    return behind_the_loop("what_came_before", _write)


def brought_back() -> dict[str, str]:
    """For each ledger whose kept copy was read this process, what became of it."""
    return dict(_ADOPTED)
