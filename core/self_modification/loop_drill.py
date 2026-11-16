"""A fire drill of her repair loop: one planted fault, raised in her running process when a person arms it (F15).

Her loop is proved on stand-in trees in the tests. A drill proves it on her own source, her origin and the process
she is running: the planted fault is raised once, when she starts, and her loop must hear it, make it again as a
test, mend it, pass her own gates, land the mend on origin, and load it into her running process. Game days and
fire drills test an incident response the same way, with a known fault.

Armed by a person: the file ``loop_drill.armed`` in her state root. While it is absent nothing here runs. ``label``
below is the planted fault, and it is planted on purpose: it reads a person's name without checking there is a
person. A mend of it is a drill result, not a fault of her own running, and is counted apart from those.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.SelfModification.Drill")

__all__ = ["armed_at", "label", "run_the_drill_if_armed"]


def armed_at() -> Path:
    from core.runtime.state_ownership import state_root

    return state_root() / "loop_drill.armed"


def label(person: Any) -> Any:
    """A person's name as a title: the drill's planted fault, for a person who is not there."""
    return (person.name.title() if person is not None else None)


def run_the_drill_if_armed() -> bool:
    """Raise the planted fault once and record it as any fault of hers is, when a drill is armed; whether it ran."""
    if not armed_at().exists():
        return False
    try:
        label(None)
    except AttributeError as exc:
        record_degradation("loop_drill", exc, severity="warning",
                           action="a drill of her repair loop: this fault is planted, for her loop to mend")
        logger.warning("🩹 A drill of her repair loop is armed: its planted fault was raised in her running process.")
        return True
    logger.info("🩹 A drill of her repair loop is armed, and its fault no longer raises: it was mended.")
    return False
