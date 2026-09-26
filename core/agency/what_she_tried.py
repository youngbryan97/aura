"""Every task she tried in a world, as it went: the experience a later version of her learns from.

SIMA 2 improves itself by keeping what it did. A task setter proposes, the
agent attempts, a judge scores the attempt, and the scored attempts train the
next version (arXiv 2512.04797, §4.5). Nothing learns from experience that
was not kept, so this is where every attempt goes.

Each episode says which world it was in, what the task was and who set it: a
person asking, her own setter proposing, or a label found afterwards for
something she did while doing something else. SIMA 2's report warns that
labels written after the fact are not tied to what was meant (§3.3.1), so the
three are kept apart and a learner can weigh them differently. It keeps the
chunks she played, in the text her hands read, how the trip ended, what the
world answered and whether that counts as success.

Kept as one line per episode under the state root, one file per world,
written through the gateway every consequential write goes through.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("Aura.WhatSheTried")

__all__ = ["ASKED", "HERSELF", "HINDSIGHT", "Episode", "how_it_goes", "keep", "what_she_tried"]

#: Who set a task.
ASKED, HERSELF, HINDSIGHT = "asked", "herself", "hindsight"


@dataclass
class Episode:
    """One attempt at one task in one world."""

    world: str
    task: str
    set_by: str
    succeeded: bool
    ended: str = ""
    answered: str = ""
    played: list[str] = field(default_factory=list)
    #: What kind of thing was tried, for counting what she is good at.
    skill: str = ""
    at: float = field(default_factory=time.time)


def _kept_in() -> Path:
    from core.runtime.state_ownership import state_root

    return state_root() / "state" / "experience"


def _file_for(world: str) -> Path:
    name = re.sub(r"[^a-z0-9]+", "-", str(world or "somewhere").lower()).strip("-") or "somewhere"
    return _kept_in() / f"{name}.jsonl"


def keep(episode: Episode) -> bool:
    """Add one episode to what she has tried in its world. Blocking: call it off the loop."""
    try:
        from core.governance_context import local_internal_governed_scope
        from core.runtime.file_write_gateway import get_file_write_gateway

        with local_internal_governed_scope(
            "what_she_tried.keep", domain="state_mutation", constraints={"world": episode.world}
        ):
            gateway = get_file_write_gateway()
            gateway.ensure_directory(_kept_in(), source="what_she_tried")
            gateway.append_text(
                _file_for(episode.world), json.dumps(asdict(episode)) + "\n", source="what_she_tried"
            )
        return True
    except Exception as exc:  # noqa: BLE001 - keeping experience is never the task
        from core.runtime.errors import record_degradation

        record_degradation("what_she_tried", exc, severity="info", action="carried on without keeping it")
        return False


def what_she_tried(world: str) -> list[Episode]:
    """Every episode kept for a world, oldest first."""
    path = _file_for(world)
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    found: list[Episode] = []
    for line in lines:
        try:
            held: dict[str, Any] = json.loads(line)
            found.append(Episode(**held))
        except (ValueError, TypeError):
            continue
    return found


def how_it_goes(world: str) -> dict[str, tuple[int, int]]:
    """Per task, how many times she tried it and how many times it worked."""
    tally: dict[str, list[int]] = {}
    for episode in what_she_tried(world):
        counts = tally.setdefault(episode.task, [0, 0])
        counts[0] += 1
        counts[1] += 1 if episode.succeeded else 0
    return {task: (tried, worked) for task, (tried, worked) in tally.items()}
