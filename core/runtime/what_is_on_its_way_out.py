"""Things kept alive for callers that have not moved yet, and when they go.

Twenty-one places in the tree say a thing is legacy, a shim, deprecated, or
kept for backward compatibility. Not one of them says when it goes, or what to
use instead in a form anything can read. So the shim is permanent by default:
nobody knows which callers still need it, nobody can tell whether a new call
site was written this week or inherited from before the replacement existed,
and every one of them is a second way to do something that a reader has to
learn.

What a deprecation needs, and none of them had:

* **since** — when the replacement landed, so "still on the old one" has a
  duration.
* **remove_after** — a date, so it either goes or somebody argues for it.
* **instead** — the thing to call now, by name.
* **why** — because a shim with no reason cannot be judged.

:func:`shims_with_no_date` is the ratchet. It reads the tree for the prose
markers and names the ones this module has not been told about, so a new shim
shows up as an undated deprecation rather than as a second permanent API.
"""
from __future__ import annotations

import datetime as _dt
import functools
import logging
import pathlib
import re
import threading
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

logger = logging.getLogger("Aura.WhatIsOnItsWayOut")

__all__ = [
    "AShim",
    "THE_SHIMS",
    "going_away",
    "what_is_declared",
    "shims_with_no_date",
    "overdue",
    "prose_markers_in_the_tree",
    "how_the_deprecations_stand",
]

F = TypeVar("F", bound=Callable[..., Any])


@dataclass(frozen=True, slots=True)
class AShim:
    """One thing kept alive, and the terms it is kept on."""

    where: str
    name: str
    since: str
    remove_after: str
    instead: str
    why: str

    def is_overdue(self, today: str | None = None) -> bool:
        now = today or _dt.date.today().isoformat()
        return bool(self.remove_after) and self.remove_after < now

    def to_dict(self) -> dict[str, Any]:
        return {
            "where": self.where,
            "name": self.name,
            "since": self.since,
            "remove_after": self.remove_after,
            "instead": self.instead,
            "why": self.why,
            "overdue": self.is_overdue(),
        }


#: Every compatibility shim the tree admits to in prose, with the terms it is
#: kept on. Read out of the code rather than invented: each entry names the
#: file that carries the shim and what the caller should reach for instead.
THE_SHIMS: tuple[AShim, ...] = (
    AShim(
        where="core/container.py",
        name="ServiceContainer.resolve",
        since="2026-04-20",
        remove_after="2027-01-01",
        instead="ServiceContainer.get",
        why="one name for one lookup; two spellings make grep lie about usage",
    ),
    AShim(
        where="core/container.py",
        name="register_legacy_aliases",
        since="2026-04-20",
        remove_after="2027-01-01",
        instead="core.service_names.ServiceNames",
        why="aliases were how the canonical registry got bypassed",
    ),
    AShim(
        where="core/state/aura_state.py",
        name="CognitiveContext.history",
        since="2026-01-01",
        remove_after="2027-01-01",
        instead="CognitiveContext.working_memory",
        why="two names for one list, and only one of them has a capacity",
    ),
    AShim(
        where="core/memory/learning.py",
        name="core.memory.learning",
        since="2026-02-01",
        remove_after="2027-01-01",
        instead="core.memory.learning.tool_learning",
        why="the module was split; the flat one forwards",
    ),
    AShim(
        where="core/senses/tts_stream.py",
        name="tts_stream shim",
        since="2026-03-01",
        remove_after="2027-01-01",
        instead="core.senses.voice",
        why="the streaming path moved and callers were not all updated",
    ),
    AShim(
        where="core/affect/damasio_v2.py",
        name="damasio_v2 legacy shim",
        since="2026-03-01",
        remove_after="2027-01-01",
        instead="core.affect.damasio",
        why="v2 replaced the original in place; the shim carries the old name",
    ),
    AShim(
        where="core/goals.py",
        name="Goal (flat record)",
        since="2026-02-01",
        remove_after="2027-01-01",
        instead="core.goals.goal_engine",
        why="a lightweight record kept because helpers and tests still build it",
    ),
    AShim(
        where="core/cognition/cognitive_loop.py",
        name="autonomous thought driver",
        since="2026-05-01",
        remove_after="2027-01-01",
        instead="_tick free-energy logic",
        why="the loop drives thoughts now; the old entry point still exists",
    ),
    AShim(
        where="core/consciousness/liquid_substrate.py",
        name="_step_torch",
        since="2026-04-01",
        remove_after="2027-01-01",
        instead="_step_torch_math via asyncio.to_thread",
        why="the synchronous one blocks the loop and callers still reach it",
    ),
    AShim(
        where="core/maintenance/dream_coordinator.py",
        name="DreamProcessor",
        since="2026-01-01",
        remove_after="2027-01-01",
        instead="the coordinator's own passes",
        why="marked do-not-re-enable and still importable",
    ),
    AShim(
        where="core/memory/vector_memory_engine.py",
        name="vector memory aliases",
        since="2026-04-20",
        remove_after="2027-01-01",
        instead="core.service_names.ServiceNames",
        why="registered so existing code does not break, which is why it never changed",
    ),
    AShim(
        where="core/runtime/mode.py",
        name="research/safe modes",
        since="2026-03-01",
        remove_after="2027-01-01",
        instead="the five strict runtime modes",
        why="two extra modes accepted for compatibility, and neither is strict",
    ),
    AShim(
        where="core/brain/prompts/sanitizer.py",
        name="introspected pattern list",
        since="2026-04-01",
        remove_after="2027-01-01",
        instead="the sanitizer's own report",
        why="kept because callers read the attribute rather than asking",
    ),
    AShim(
        where="core/consciousness/integration.py",
        name="non-strict integration path",
        since="2026-04-01",
        remove_after="2027-01-01",
        instead="strict mode",
        why="a lenient path that exists only for callers that never moved",
    ),
    AShim(
        where="core/consciousness/neurochemical_system.py",
        name="synced level field",
        since="2026-03-01",
        remove_after="2027-01-01",
        instead="the level the system actually holds",
        why="a mirrored field, and a mirror is a second place to be wrong",
    ),
    AShim(
        where="core/motivation/drives.py",
        name="drives dict access",
        since="2026-03-01",
        remove_after="2027-01-01",
        instead="the drive objects",
        why="dict access and decay-by-'time' predate the typed drives",
    ),
    AShim(
        where="core/cognitive/state_machine.py",
        name="generate_image alias",
        since="2026-02-01",
        remove_after="2027-01-01",
        instead="the current state name",
        why="an old state name still mapped so saved sessions load",
    ),
    AShim(
        where="core/config.py",
        name="config legacy alias",
        since="2026-02-01",
        remove_after="2027-01-01",
        instead="the canonical config name",
        why="one setting reachable under two names",
    ),
    AShim(
        where="core/orchestrator/mixins/boot/boot_resilience.py",
        name="'swarm' registration",
        since="2026-04-20",
        remove_after="2027-01-01",
        instead="the debate delegator's canonical name",
        why="registered under an alias for debate delegation",
    ),
    AShim(
        where="core/memory/learning/tool_learning.py",
        name="module-level export",
        since="2026-02-01",
        remove_after="2027-01-01",
        instead="the class",
        why="an export kept so imports do not break, initialised defensively",
    ),
    AShim(
        where="core/mycelium.py",
        name="substring pattern conversion / bare skill name",
        since="2026-03-01",
        remove_after="2027-01-01",
        instead="the regex pathways and the full skill record",
        why="two shims for orchestrator code that reads the old shapes",
    ),
    AShim(
        where="core/orchestrator/main.py",
        name="cognitive loop shim",
        since="2026-04-01",
        remove_after="2027-01-01",
        instead="self.cognitive_loop",
        why="the orchestrator forwards; callers still go through the old name",
    ),
    AShim(
        where="core/orchestrator/mixins/boot/boot_autonomy.py",
        name="skill_manager / skill_router registrations",
        since="2026-04-20",
        remove_after="2027-01-01",
        instead="the capability engine's canonical name",
        why="one engine registered under two extra names at boot",
    ),
    AShim(
        where="core/orchestrator/mixins/cognitive_background.py",
        name="RL training shim",
        since="2026-04-01",
        remove_after="2027-01-01",
        instead="the learning engine",
        why="moved, with the entry point left behind",
    ),
    AShim(
        where="core/social/presence_integration.py",
        name="RelationshipGraph legacy aliases",
        since="2026-04-20",
        remove_after="2027-01-01",
        instead="the canonical service name",
        why="registered under both, so a lookup by either succeeds",
    ),
    AShim(
        where="core/runtime/what_is_on_its_way_out.py",
        name="(this module's own prose)",
        since="2026-09-06",
        remove_after="2099-01-01",
        instead="(nothing; it is the thing that reads the markers)",
        why="it quotes the phrases it searches for, so it matches itself",
    ),
    AShim(
        where="core/adaptation/dynamic_value_graph.py",
        name="ValueNodeStatus.DEPRECATED",
        since="2026-01-01",
        remove_after="2028-01-01",
        instead="(nothing; this one is a value the graph uses)",
        why="not a shim: a status a node can hold, listed so the scan can tell",
    ),
)

_DECLARED: dict[str, AShim] = {f"{s.where}::{s.name}": s for s in THE_SHIMS}
_CALLED: dict[str, int] = {}
_LOCK = threading.Lock()


def going_away(
    *,
    since: str,
    remove_after: str,
    instead: str,
    why: str = "",
) -> Callable[[F], F]:
    """Mark a callable as on its way out, with the terms it is kept on.

    Warns once per call site rather than once per call: a shim in a loop
    should not fill the log, and a shim called from three places should say
    three things.
    """

    def wrap(fn: F) -> F:
        key = f"{getattr(fn, '__module__', '?')}.{getattr(fn, '__qualname__', fn)}"

        @functools.wraps(fn)
        def called(*args: Any, **kwargs: Any) -> Any:
            with _LOCK:
                seen = _CALLED.get(key, 0)
                _CALLED[key] = seen + 1
            if seen == 0:
                warnings.warn(
                    f"{key} has been on its way out since {since} and is due "
                    f"for removal after {remove_after}. Use {instead}."
                    + (f" ({why})" if why else ""),
                    DeprecationWarning,
                    stacklevel=2,
                )
            return fn(*args, **kwargs)

        called.__aura_going_away__ = AShim(  # type: ignore[attr-defined]
            where=getattr(fn, "__module__", ""),
            name=getattr(fn, "__qualname__", ""),
            since=since,
            remove_after=remove_after,
            instead=instead,
            why=why,
        )
        return called  # type: ignore[return-value]

    return wrap


def what_is_declared() -> tuple[AShim, ...]:
    with _LOCK:
        return tuple(_DECLARED.values())


def overdue(today: str | None = None) -> tuple[AShim, ...]:
    """Shims past their removal date. Either they go or the date moves."""
    return tuple(s for s in THE_SHIMS if s.is_overdue(today))


#: Phrases that DECLARE a compatibility surface, not ones that mention
#: compatibility in passing. "We map kwargs for compatibility if needed" is a
#: comment about an argument; "Legacy alias for get()" is a second public name
#: that will outlive everyone who remembers why.
_PROSE = re.compile(
    r"legacy alias(?:es)?"
    r"|legacy shim"
    r"|shim for backward"
    r"|kept for compatibility"
    r"|for backwards? compatibility"
    r"|maintains? backwards? compatibility"
    r"|\[DEPRECATED\]"
    r"|DEPRECATED:"
    r"|DEPRECATED —"
    r"|# *DEPRECATED",
    re.IGNORECASE,
)
_ROOTS = ("core", "interface", "skills", "llm", "executors")


def prose_markers_in_the_tree(repo: str = ".") -> dict[str, list[int]]:
    """Files that say something is legacy or deprecated, and where."""
    root = pathlib.Path(repo)
    found: dict[str, list[int]] = {}
    for name in _ROOTS:
        base = root / name
        if not base.exists():
            continue
        for path in sorted(base.rglob("*.py")):
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if not _PROSE.search(text):
                continue
            lines = [
                number
                for number, line in enumerate(text.splitlines(), 1)
                if _PROSE.search(line)
            ]
            if lines:
                found[str(path.relative_to(root))] = lines
    return found


def shims_with_no_date(repo: str = ".") -> tuple[str, ...]:
    """Files admitting to a shim that this module has not been told about.

    The ratchet. A new compatibility shim appears here rather than as a second
    permanent way to do something.
    """
    declared = {s.where for s in THE_SHIMS}
    return tuple(sorted(set(prose_markers_in_the_tree(repo)) - declared))


def how_the_deprecations_stand(repo: str = ".") -> dict[str, Any]:
    """For the health report: what is going, when, and what nobody dated."""
    marked = prose_markers_in_the_tree(repo)
    undated = shims_with_no_date(repo)
    with _LOCK:
        called = dict(sorted(_CALLED.items()))
    return {
        "declared": len(THE_SHIMS),
        "files_that_admit_to_one": len(marked),
        "with_no_date": list(undated),
        "overdue": [s.to_dict() for s in overdue()],
        "called_this_process": called,
        "earliest_removal": min((s.remove_after for s in THE_SHIMS), default=""),
    }
