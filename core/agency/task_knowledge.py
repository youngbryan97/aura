"""Finding out how a thing is done, before and while doing it.

A goal loop that only ever reads the screen in front of it can play a game
badly forever. It has the board, the moves, and what its last few moves led
to, and none of that contains the thing a person would go and look up: that
this kind of task has a known way of being done well.

So a pursuit asks two questions it could not answer from the screen. Have I
done this before, and what happened? And if that is thin — what does anyone
know about doing this? The first comes from her own record, the second from
the search skills she already has.

What comes back is evidence, not instruction. It joins the goal, the reading
and the consequence history as another line the decision rests on, and it is
attributed, so a strategy that turns out to be wrong is traceable to where it
came from rather than folded invisibly into her judgement. Nothing here
phrases anything to steer an answer; a retrieved fact is a fact.

Looking things up is also something she should be able to say she is doing.
Findings go to the workspace, so the narrator can tell you what she read and
what she means to try before she tries it.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.TaskKnowledge")

#: How long a finding stays worth reusing inside one session. Long enough
#: that a run does not search twice for the same goal, short enough that a
#: strategy is re-checked when a task is returned to much later.
KNOWLEDGE_TTL_S = 1800.0
#: How many findings are worth carrying into a decision. Past a handful they
#: stop being evidence and start being a wall of text.
FINDINGS_KEPT = 4
#: Sentences longer than this are prose about a topic rather than a usable
#: statement of how to do it.
MAX_FINDING_CHARS = 240

#: Words that mark a sentence as saying HOW to do something, rather than
#: describing what the thing is. A search returns both, and only one of them
#: helps somebody who is mid-task.
ACTIONABLE = re.compile(
    r"\b(?:keep|hold|always|never|avoid|prefer|try to|make sure|start by|"
    r"the key is|best to|should|instead of|rather than|corner|order|priority|"
    r"strategy|technique|rule of thumb)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Finding:
    """One thing she learned about how to do this, and where it came from."""

    says: str
    source: str = ""

    def as_evidence(self) -> str:
        return f"Known about this task — {self.says} ({self.source})" if self.source else f"Known about this task — {self.says}"


@dataclass
class TaskKnowledge:
    """What she knows about doing this, and how she came to know it."""

    goal: str
    findings: list[Finding] = field(default_factory=list)
    from_memory: int = 0
    searched: str = ""
    #: True when this was looked up because the run had stopped getting
    #: anywhere, rather than at the start.
    stuck: bool = False
    at: float = field(default_factory=time.time)

    @property
    def known(self) -> bool:
        return bool(self.findings)

    def as_evidence(self) -> list[str]:
        return [finding.as_evidence() for finding in self.findings[:FINDINGS_KEPT]]

    def narrate(self) -> str:
        """What she found and what she means to do with it."""
        if not self.findings:
            if self.stuck:
                return f"This is not working and I could not find out why. I looked up: {self.searched}"
            return f"I could not find anything about how {self.goal[:80]} is usually done."
        lead = self.findings[0].says
        rest = len(self.findings) - 1
        tail = f", and {rest} other thing{'s' if rest > 1 else ''}" if rest > 0 else ""
        opening = "This stopped working, so I looked it up" if self.stuck else "I read that"
        joiner = ": " if self.stuck else " "
        return f"{opening}{joiner}{lead}{tail}. I am going to try that."


_remembered: dict[str, TaskKnowledge] = {}


def usable_sentences(text: str) -> list[str]:
    """The sentences in a body of text that say how to do something."""
    found: list[str] = []
    for raw in re.split(r"(?<=[.!?])\s+|\n+", str(text or "")):
        sentence = " ".join(raw.split()).strip(" -•*")
        if not sentence or len(sentence) > MAX_FINDING_CHARS or len(sentence) < 20:
            continue
        if ACTIONABLE.search(sentence):
            found.append(sentence)
    return found


def _from_her_own_record(goal: str, *, graph: Any = None) -> list[Finding]:
    """What her consequence graph already says about this kind of task."""
    try:
        if graph is None:
            from core.world_model.acg import acg as graph  # noqa: PLC0415
        rows = graph.query_consequences(goal) or []
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        record_degradation("task_knowledge", exc, severity="info", action="read her own record of this task")
        return []
    findings: list[Finding] = []
    for row in list(rows)[:FINDINGS_KEPT]:
        if not isinstance(row, dict):
            continue
        outcome = str(row.get("outcome") or "").strip()
        if outcome:
            worked = "worked" if row.get("success") else "did not work"
            findings.append(Finding(says=f"last time this {worked}: {outcome}"[:MAX_FINDING_CHARS], source="my own record"))
    return findings


async def _from_search(question: str, *, engine: Any = None) -> tuple[list[Finding], str]:
    """What anyone knows about doing this, through the search skills she has."""
    try:
        if engine is None:
            from core.container import get_container  # noqa: PLC0415
            from core.exceptions import ContainerError  # noqa: PLC0415

            try:
                engine = get_container().get("capability_engine")
            except (ContainerError, KeyError):
                engine = None
        if engine is None:
            return [], ""
        result = await engine.execute(
            "web_search",
            {"query": question, "num_results": 5},
            {"requested_via": "task_knowledge", "purpose": "how a task is done"},
        )
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError, TimeoutError) as exc:
        record_degradation("task_knowledge", exc, severity="info", action="looked up how a task is done")
        return [], ""

    text = ""
    if isinstance(result, dict):
        for key in ("summary", "answer", "content", "text", "result"):
            value = result.get(key)
            if isinstance(value, str) and value.strip():
                text = value
                break
        if not text:
            items = result.get("results") or result.get("items") or []
            if isinstance(items, list):
                text = " ".join(
                    str(item.get("snippet") or item.get("content") or item.get("title") or "")
                    for item in items
                    if isinstance(item, dict)
                )
    findings = [Finding(says=sentence, source="what I read") for sentence in usable_sentences(text)]
    return findings[:FINDINGS_KEPT], question


def how_is_this_done(goal: str) -> str:
    """The question to go and ask about a goal.

    Built from the goal itself rather than from a template per task type: the
    thing she wants to know is always the same shape, and the goal is the only
    part that varies.
    """
    stripped = " ".join(str(goal or "").split())[:120]
    return f"how to do this well: {stripped}" if stripped else ""


def why_is_this_stuck(goal: str, situation: str = "", history: Sequence[Any] = ()) -> str:
    """The question to ask when what she is doing has stopped working.

    A different question from :func:`how_is_this_done`, and the difference is
    the point. "How is this played" returns the beginner's answer she already
    has. What a stuck run needs is the answer to the position it is actually
    in — which moves it has tried, what they did, and what is in front of it.
    That is the question a person types when they are stuck, and it is the
    one that returns something they did not already know.

    Every part is measured: the moves come from the graded history, the
    situation from the last reading. Nothing is characterised or embellished.
    """
    stripped = " ".join(str(goal or "").split())[:100]
    if not stripped:
        return ""
    parts = [f"{stripped} stuck"]

    dead: list[str] = []
    for attempt in list(history)[-6:]:
        option = str(getattr(attempt, "option", "") or "")
        verdict = getattr(attempt, "verdict", None)
        if option and getattr(verdict, "held", True) is False and option not in dead:
            dead.append(option)
    if dead:
        parts.append(f"{' and '.join(dead)} do nothing")

    seen = _salient(situation)
    if seen:
        parts.append(f"showing {seen}")
    parts.append("what to do")
    return ", ".join(parts)[:200]


def _salient(situation: str) -> str:
    """The part of a reading worth putting in a question.

    A whole screen dump makes a search useless. What identifies a position is
    the values on it, so the numbers and short labels are kept and the prose
    around them is dropped.
    """
    text = " ".join(str(situation or "").split())
    if not text:
        return ""
    tokens = re.findall(r"\b[0-9]{1,6}\b", text)
    if tokens:
        # Most recent first would be arbitrary; the largest values are what
        # actually characterise a position.
        largest = sorted({int(token) for token in tokens}, reverse=True)[:6]
        return " ".join(str(value) for value in largest)
    return text[:60]


async def learn_about(
    goal: str,
    *,
    engine: Any = None,
    graph: Any = None,
    search: bool = True,
    remember: bool = True,
    situation: str = "",
    history: Sequence[Any] = (),
    because_stuck: bool = False,
) -> TaskKnowledge:
    """What she knows about doing this, from her own record and from looking.

    Her own record comes first because it is about her, doing this, here.
    Searching happens when that is thin — a run should not go to the network
    to be told something it already learned the hard way.
    """
    goal = str(goal or "").strip()
    knowledge = TaskKnowledge(goal=goal)
    if not goal:
        return knowledge

    held = _remembered.get(goal) if remember else None
    if held is not None and time.time() - held.at < KNOWLEDGE_TTL_S:
        return held

    mine = _from_her_own_record(goal, graph=graph)
    knowledge.findings.extend(mine)
    knowledge.from_memory = len(mine)

    # Being stuck asks a different question, and skips her own record.
    #
    # Her record is what got her here. The whole reason to look again is that
    # what she knows is not enough for the position she is in.
    if because_stuck:
        knowledge.findings.clear()
        knowledge.from_memory = 0
        knowledge.stuck = True

    if search and (because_stuck or len(knowledge.findings) < FINDINGS_KEPT):
        question = (
            why_is_this_stuck(goal, situation, history) if because_stuck else how_is_this_done(goal)
        )
        read, asked = await _from_search(question, engine=engine)
        knowledge.findings.extend(read)
        knowledge.searched = asked

    if remember:
        _remembered[goal] = knowledge
    _announce(knowledge)
    return knowledge


def forget_everything() -> None:
    """Drop what is held, so a later run learns again."""
    _remembered.clear()


def _announce(knowledge: TaskKnowledge) -> None:
    """Offer what she found to the workspace, so she can say it."""
    if not knowledge.known:
        return
    try:
        from core.consciousness.global_workspace import ContentType  # noqa: PLC0415
        from core.container import ServiceContainer  # noqa: PLC0415

        workspace = ServiceContainer.get("global_workspace", default=None)
        if workspace is None:
            return
        publish = getattr(workspace, "publish", None)
        if publish is None:
            return
        import asyncio  # noqa: PLC0415

        coroutine = publish(
            # Worth interrupting for: she is about to do something differently
            # from how she was doing it, and the person should hear why.
            priority=0.75,
            source="agency.learned",
            payload={
                "schema": "aura.decision.v1",
                "learned": {
                    "goal": knowledge.goal,
                    "findings": [finding.says for finding in knowledge.findings[:FINDINGS_KEPT]],
                    "from_memory": knowledge.from_memory,
                    "searched": knowledge.searched,
                },
            },
            reason=knowledge.narrate(),
            content_type=ContentType.MEMORIAL,
        )
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            coroutine.close()
            return
        task = loop.create_task(coroutine)
        task.add_done_callback(lambda done: done.exception())
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        record_degradation("task_knowledge", exc, severity="info", action="learned without saying so")


def stuck(history: Sequence[Any], *, run_of: int = 3) -> bool:
    """Whether the last few moves have stopped getting anywhere.

    A run of predictions that all broke means what she is doing is not
    working, whatever the reason. That is the moment to go and find out how
    the task is done rather than to keep pressing.
    """
    recent = list(history)[-run_of:]
    if len(recent) < run_of:
        return False
    return all(getattr(getattr(attempt, "verdict", None), "held", True) is False for attempt in recent)
