"""What she is working on and how she has decided to go about it.

An approach held inside the loop that chose it is a gear in a box. Asked
mid-play what she is doing, she would answer from the model's guess while her
body worked from something else, and nothing outside that loop could be
affected by a decision she had actually made. The approach only means
anything if it is part of her — visible to the workspace that decides what
she is attending to, readable by whoever answers a question about the
present, and recorded so that adopting it and abandoning it are events she
can learn from.

So this is one small piece of live state with three obligations. It is
published when it changes, because a change of plan is worth attending to. It
is readable at any moment, because a question about what she is doing has to
be answered from what she is doing. And it is written to experience, because
an approach that keeps failing should be harder to adopt next time.

Nothing here is about any one task. A line taken, the reason for it, the
thing that would end it, and how that turned out is the shape of doing
anything on purpose.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.Doing")

_RECOVERABLE = (ImportError, AttributeError, RuntimeError, TypeError, ValueError, KeyError)

#: How long an undertaking nobody has touched still counts as current. A
#: process that died mid-task should not leave her answering questions about
#: work that stopped hours ago.
STALE_AFTER_S = 900.0


@dataclass
class Undertaking:
    """One thing she is doing, and the line she is taking through it."""

    goal: str
    approach: str = ""
    because: str = ""
    watching_for: str = ""
    alternatives: tuple[str, ...] = ()
    where: str = ""
    began_at: float = field(default_factory=time.time)
    changed_at: float = field(default_factory=time.time)
    changes: int = 0
    steps: int = 0
    #: Approaches she has already tried and left behind, in order.
    left_behind: tuple[str, ...] = ()

    def fresh(self) -> bool:
        return (time.time() - self.changed_at) < STALE_AFTER_S

    def as_lines(self) -> list[str]:
        """Her present tense, for whoever has to answer for it."""
        lines = [f"Working on: {self.goal}"]
        if self.where:
            lines.append(f"Where: {self.where}")
        if self.approach:
            said = f"How I am going about it: {self.approach}"
            if self.because:
                said = f"{said} (because {self.because})"
            lines.append(said)
        if self.watching_for:
            lines.append(f"What would make me change that: {self.watching_for}")
        if self.alternatives:
            lines.append(f"What I would do instead: {', '.join(self.alternatives)}")
        if self.left_behind:
            lines.append(f"Approaches I have already left behind: {'; '.join(self.left_behind)}")
        if self.steps:
            lines.append(f"Steps taken so far: {self.steps}")
        return lines


_current: Undertaking | None = None


def taking_on(goal: str, *, where: str = "") -> Undertaking:
    """She has started something."""
    global _current
    _current = Undertaking(goal=" ".join(str(goal or "").split()), where=str(where or ""))
    _publish(f"Starting: {_current.goal}", priority=0.75)
    return _current


def going_about_it(
    approach: str,
    *,
    because: str = "",
    watching_for: str = "",
    alternatives: tuple[str, ...] = (),
    spine: Any = None,
    lived: bool = True,
) -> None:
    """She has decided how to go about it, or decided differently.

    Recorded as a decision rather than a setting, because choosing a line and
    changing it are the two moments where what she does stops following from
    what she did before, and neither is visible in the moves themselves.
    """
    global _current
    if _current is None:
        _current = Undertaking(goal="")
    said = " ".join(str(approach or "").split())
    if not said:
        return
    changing = bool(_current.approach) and said != _current.approach
    if changing:
        _current.left_behind = (*_current.left_behind, _current.approach)[-6:]
        _current.changes += 1
    _current.approach = said
    _current.because = " ".join(str(because or "").split())
    _current.watching_for = " ".join(str(watching_for or "").split())
    _current.alternatives = tuple(alternatives)
    _current.changed_at = time.time()
    _publish(
        f"Changing approach: {said}" if changing else f"Approach: {said}",
        priority=0.85 if changing else 0.7,
    )
    _remember(changing=changing, spine=spine, lived=lived)


def a_step_taken() -> None:
    """One more move under the current approach."""
    if _current is not None:
        _current.steps += 1
        _current.changed_at = time.time()


def how_it_went(succeeded: bool, note: str = "", *, graph: Any = None) -> None:
    """How the approach she was taking turned out.

    Written where consequences live, so an approach that keeps failing is
    harder to reach for next time and one that works is easier — which is the
    only thing that makes holding an approach different from having a habit.
    """
    global _current
    if _current is None or not _current.approach:
        _current = None
        return
    try:
        if graph is None:
            from core.world_model.acg import acg as graph  # noqa: PLC0415
        graph.record_outcome(
            f"approach: {_current.approach}",
            f"{_current.goal} — {_current.where}".strip(" —"),
            note or ("it worked" if succeeded else "it did not work"),
            bool(succeeded),
        )
    except _RECOVERABLE as exc:
        record_degradation("what_she_is_doing", exc, action="record how an approach turned out")
    _current = None


def right_now() -> Undertaking | None:
    """What she is doing, if she is doing anything."""
    if _current is None or not _current.fresh():
        return None
    return _current


def as_lines() -> list[str]:
    """Her present tense, for whoever has to answer for it."""
    doing = right_now()
    return doing.as_lines() if doing is not None else []


def _remember(*, changing: bool, spine: Any, lived: bool) -> None:
    if _current is None:
        return
    try:
        from core.ontogeny.experience import (  # noqa: PLC0415
            Episode,
            Provenance,
            get_experience_spine,
        )

        store = spine if spine is not None else get_experience_spine()
        episode = Episode(
            provenance=Provenance.LIVE if lived else Provenance.TEST,
            control_point="agency.approach",
            features={"changes": float(_current.changes), "steps": float(_current.steps)},
            decision=_current.approach,
            options=(*_current.left_behind, _current.approach),
            decider="agency.what_she_is_doing",
            stakes=0.6 if changing else 0.4,
            context={"goal": _current.goal, "watching_for": _current.watching_for},
        )
        store.record(episode)
    except _RECOVERABLE as exc:
        record_degradation("what_she_is_doing", exc, severity="info", action="remember a change of plan")


def _publish(said: str, *, priority: float) -> None:
    """Offer it to the workspace, so the rest of her can be affected by it."""
    try:
        from core.consciousness.global_workspace import ContentType  # noqa: PLC0415
        from core.container import ServiceContainer  # noqa: PLC0415

        workspace = ServiceContainer.get("global_workspace", default=None)
        publish = getattr(workspace, "publish", None) if workspace else None
        if publish is None:
            return
        coroutine = publish(
            priority=priority,
            source="agency.what_she_is_doing",
            payload={
                "schema": "aura.intention.v1",
                "intention": {
                    "goal": getattr(_current, "goal", ""),
                    "approach": getattr(_current, "approach", ""),
                    "because": getattr(_current, "because", ""),
                    "watching_for": getattr(_current, "watching_for", ""),
                },
            },
            reason=said,
            content_type=ContentType.INTENTIONAL,
        )
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            coroutine.close()
            return
        task = loop.create_task(coroutine)
        task.add_done_callback(lambda done: done.exception())
    except _RECOVERABLE as exc:
        record_degradation("what_she_is_doing", exc, severity="info", action="say what she is doing")
