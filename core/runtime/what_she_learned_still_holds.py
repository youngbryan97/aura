"""What she has learned of each world she has been in, read again by the code as it stands: what no longer holds.

A change to how she learns or reads a world is tested where learning starts from nothing. What she has actually
learned holds every misreading of every visit, and the code reads it differently after the change. LIVE 23 November
2026, the first world she went back to after one such change (164f17a72) began, by her own reading of her record, with
her already lost, and she said so before acting once. Every test had passed.

Each part of a world's record is read by the module that keeps it. A module that can say what would make its part
unusable registers that here, by the part's name in the record (``replays``). ``what_no_longer_holds`` reads every kept
world through every registered reader that knows a part of it. A reader that raises on a record is a finding too: a
record the code can no longer read is a world she has forgotten without being told.
"""
from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from typing import Any

from core.runtime.what_she_learned import kept_worlds, recall

__all__ = ["readers", "replays", "what_no_longer_holds"]

logger = logging.getLogger("Aura.WhatSheLearnedStillHolds")

#: Given a world's name and its record, what of one part of it no longer holds under the code as it stands; [] where it
#: all does.
Replay = Callable[[str, dict[str, Any]], list[str]]

_REPLAYS: dict[str, Replay] = {}


def replays(part: str) -> Callable[[Replay], Replay]:
    """Registers how the part of a world's record kept under ``part`` is read again."""

    def keep(replay: Replay) -> Replay:
        _REPLAYS[part] = replay
        return replay

    return keep


def readers() -> list[str]:
    """The parts of a world's record that a reader has registered a replay for."""
    return sorted(_REPLAYS)


def what_no_longer_holds(worlds: Iterable[str] | None = None) -> tuple[dict[str, list[str]], int]:
    """For each kept world, what no longer holds of each part a reader knows; and how many worlds were read."""
    found: dict[str, list[str]] = {}
    read = 0
    for world in (kept_worlds() if worlds is None else list(worlds)):
        record = recall(world)
        known = [part for part in _REPLAYS if part in record]
        if not known:
            continue
        read += 1
        for part in known:
            try:
                wrong = _REPLAYS[part](world, record)
            except Exception as exc:  # noqa: BLE001 - a reader failing on her record is what this looks for
                wrong = [f"reading it raised {type(exc).__name__}: {exc}"]
            if wrong:
                found.setdefault(world, []).extend(f"{part}: {line}" for line in wrong)
    logger.info("what she learned, read again: %d world(s) read, %d with something that no longer holds", read,
                len(found))
    return found, read
