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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

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

#: Where an answer came from, because a strategy and a definition are not
#: interchangeable and the difference should survive into the evidence.
BACKGROUND = "background"
TACTIC = "tactic"

#: A question about what something IS. Her offline corpus is a full Wikipedia
#: snapshot, so these are answered instantly, privately, and without a
#: network — and asking the web first would be going outside for something
#: already in the building.
_BACKGROUND_RE = re.compile(
    r"\b(?:what\s+is|what\s+are|who\s+(?:is|was)|when\s+(?:did|was)|"
    r"define|meaning\s+of|history\s+of|background\s+on|tell\s+me\s+about)\b",
    re.IGNORECASE,
)
#: A question about what to DO. An encyclopedia entry describes a game; it
#: does not say what to do when your board is locked. These go to the web,
#: because that is where people write down how things are actually done.
_TACTIC_RE = re.compile(
    r"\b(?:how\s+(?:do|to|can)|what\s+(?:do|should)\s+i|stuck|strategy|"
    r"tactic|technique|tips?|best\s+way|get\s+past|not\s+working|"
    r"do\s+nothing|what\s+to\s+do)\b",
    re.IGNORECASE,
)


def kind_of_question(question: str) -> str:
    """Whether this asks what something is, or what to do about it.

    The distinction decides where to ask. Her offline encyclopedia holds
    6.5 million articles and answers the first kind instantly; it holds
    almost nothing about the second, because an article about a game says
    what the game is and not how to get out of a locked board.
    """
    text = str(question or "")
    if _TACTIC_RE.search(text):
        return TACTIC
    if _BACKGROUND_RE.search(text):
        return BACKGROUND
    return TACTIC


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
    #: Which kind of question was asked, and therefore which source answered.
    asking: str = ""
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


async def _from_her_own_shelf(question: str, *, engine: Any = None) -> list[Finding]:
    """What her offline encyclopedia says, which is instant and needs no network.

    Asked first for a question about what something IS. Going to the web for
    that is going outside for something already in the building — she carries
    a full Wikipedia snapshot and answers from it in tens of milliseconds.
    """
    engine = engine if engine is not None else _capability_engine()
    if engine is None:
        return []
    try:
        result = await engine.execute(
            "local_reference_search",
            {"query": question, "limit": 3},
            {"requested_via": "task_knowledge", "purpose": "background on a task"},
        )
    except (AttributeError, RuntimeError, TypeError, ValueError, TimeoutError) as exc:
        record_degradation("task_knowledge", exc, severity="info", action="read her own reference shelf")
        return []
    if not isinstance(result, Mapping):
        return []
    findings: list[Finding] = []
    for row in list(result.get("results") or [])[:FINDINGS_KEPT]:
        if not isinstance(row, Mapping):
            continue
        title = str(row.get("title") or "").strip()
        for sentence in usable_sentences(str(row.get("snippet") or ""))[:2]:
            findings.append(Finding(says=sentence, source=f"my own reference shelf: {title}"))
    return findings


def _capability_engine() -> Any:
    try:
        from core.container import get_container  # noqa: PLC0415
        from core.exceptions import ContainerError  # noqa: PLC0415

        try:
            return get_container().get("capability_engine")
        except (ContainerError, KeyError):
            return None
    except (ImportError, AttributeError, RuntimeError):
        return None



async def _search_results_for(question: str, *, engine: Any = None, browser: Any = None) -> list[dict[str, str]]:
    """Candidates for a question, fetched without navigating anywhere."""
    if browser is None:
        try:
            from core.capabilities.browser_controller import get_browser_controller  # noqa: PLC0415

            browser = get_browser_controller()
        except (ImportError, AttributeError, RuntimeError):
            return []
    try:
        return await browser.search_results(question, count=FINDINGS_KEPT)
    except (RuntimeError, OSError, AttributeError, TypeError, ValueError) as exc:
        record_degradation("task_knowledge", exc, severity="info", action="look for an answer")
        return []


async def _read_the_best_answer(
    results: Sequence[Mapping[str, Any]], question: str, *, browser: Any = None
) -> list[Finding]:
    """Open the most relevant result and read what it actually says.

    A snippet is an advertisement for an answer. Being stuck needs the answer,
    which means opening the page whose description best matches the question
    and reading the sentences in it that address that question — not the ones
    that happen to sound instructive.
    """
    if not results:
        return []
    wanted = _distinctive(question)
    best = max(
        results,
        key=lambda row: len(wanted & _distinctive(f"{row.get('title', '')} {row.get('snippet', '')}")),
    )
    url = str(best.get("url") or "").strip()
    if not url:
        return []
    if browser is None:
        try:
            from core.capabilities.browser_controller import get_browser_controller  # noqa: PLC0415

            browser = get_browser_controller()
        except (ImportError, AttributeError, RuntimeError):
            return []
    try:
        extract = await browser.extract_article_text(url)
    except (RuntimeError, OSError, AttributeError, TypeError, ValueError) as exc:
        record_degradation("task_knowledge", exc, severity="info", action="read a page for an answer")
        return []
    body = str(getattr(extract, "text", "") or "")
    if not body:
        return []
    # The sentences that answer THIS question, ranked by how much of it they
    # actually address. Untrusted text: it is carried as something she read,
    # attributed, and never as an instruction.
    scored = sorted(
        ((len(wanted & _distinctive(line)), line) for line in usable_sentences(body)),
        key=lambda row: row[0],
        reverse=True,
    )
    where = str(getattr(extract, "source_domain", "") or url)
    return [Finding(says=line, source=f"read on {where}") for score, line in scored[:FINDINGS_KEPT] if score]


def _distinctive(text: str) -> set[str]:
    from core.agency.deliberate_action import _distinctive as words  # noqa: PLC0415

    return words(text)


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
        knowledge.asking = kind_of_question(question)
        knowledge.searched = question
        if knowledge.asking == BACKGROUND:
            knowledge.findings.extend(await _from_her_own_shelf(question, engine=engine))
        if knowledge.asking == TACTIC or not knowledge.findings:
            read, _asked = await _from_search(question, engine=engine)
            if read:
                knowledge.findings.extend(read)
            else:
                # A search that returned only headlines is not an answer.
                # Open the best match and read it.
                results = await _search_results_for(question, engine=engine)
                knowledge.findings.extend(await _read_the_best_answer(results, question))

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
            asyncio.get_running_loop()
        except RuntimeError:
            coroutine.close()
            return
        from core.utils.task_tracker import get_task_tracker

        task = get_task_tracker().track(coroutine, name="task_knowledge.learn")
        task.add_done_callback(lambda done: done.exception())
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        record_degradation("task_knowledge", exc, severity="info", action="learned without saying so")



@dataclass(frozen=True)
class Implication:
    """What a thing she read means for the position she is actually in."""

    finding: str
    means: str
    favours: str = ""
    #: What she has already measured that bears on this advice. Empty when
    #: nothing does.
    against: str = ""

    @property
    def holds_up(self) -> bool:
        """Whether her own evidence supports acting on this."""
        return not self.against

    def as_evidence(self) -> str:
        line = f"What that means here — {self.means}"
        if self.favours:
            line = f"{line} (so: {self.favours})"
        if self.against:
            line = f"{line}. I am not taking that at face value: {self.against}"
        return line


def _favoured_option(finding: str, options: Sequence[Any]) -> str:
    """Which available move a finding points at, when it points at one.

    An option's own name appearing in the advice is the strongest signal
    there is, and it is checked directly rather than through the distinctive
    words — move names are short, and a filter that drops words under three
    letters drops "up" and "go" along with "the".
    """
    text = str(finding or "")
    if not text or not options:
        return ""
    # The same job choose_named already does, and for the same reason: a
    # sentence that works through the moves before settling on one mentions
    # several, so the last one named is the one it settled on. "Pressing up
    # would dislodge it — go left instead" favours left.
    from core.agency.deliberate_action import choose_named  # noqa: PLC0415

    named = choose_named(text, options)
    if named is not None:
        return str(getattr(named, "name", ""))
    said = _distinctive(text)
    if not said:
        return ""
    best_name, best_score = "", 0
    for option in options:
        described = _distinctive(f"{getattr(option, 'name', '')} {getattr(option, 'detail', '')}")
        score = len(said & described)
        if score > best_score:
            best_name, best_score = str(getattr(option, "name", "")), score
    return best_name


def _structural_meaning(finding: str, situation: str, options: Sequence[Any]) -> Implication:
    """What a finding means here, worked out without asking anything.

    Weaker than reasoning about it, and honest about that: it says which
    available move the advice names, and otherwise says it does not fit what
    is on screen. A floor, so that losing language costs her the quality of
    the thinking and not the step itself.
    """
    favours = _favoured_option(finding, options)
    shared = _distinctive(finding) & _distinctive(situation)
    if favours:
        means = f"it names {favours}, which is available now"
    elif shared:
        means = "it is about " + ", ".join(sorted(shared)[:3]) + ", which is on screen"
    else:
        means = "nothing on screen matches what it describes"
    return Implication(finding=finding, means=means, favours=favours)


def _meaning_question(finding: str, situation: str, options: Sequence[Any]) -> str:
    names = ", ".join(str(getattr(option, "name", "")) for option in options)
    return (
        "Given what is on screen, say in one sentence what this means for the "
        f"move to make next, and name one of: {names}."
    )


def _measured_against(favours: str, history: Sequence[Any]) -> str:
    """What she has already watched happen that bears on this advice.

    Advice is somebody else's experience of a task in general. Her own
    measurements are of this task, now, and when the two disagree the
    measurement wins — it is about the position she is actually in.

    A move she has watched change nothing twice does not become a good move
    because a page recommends it. Saying so is the difference between reading
    advice and following it.
    """
    if not favours:
        return ""
    broke = [
        attempt
        for attempt in history
        if str(getattr(attempt, "option", "")) == favours
        and getattr(getattr(attempt, "verdict", None), "held", True) is False
    ]
    if len(broke) < 2:
        return ""
    verdict = getattr(broke[-1], "verdict", None)
    why = verdict.why() if verdict is not None and hasattr(verdict, "why") else ""
    return f"{favours} has already been tried {len(broke)} times here and {why or 'did nothing'}"


def _appraised(meaning: Implication, history: Sequence[Any]) -> Implication:
    """The same reading, with her own evidence about it attached."""
    against = _measured_against(meaning.favours, history)
    if not against:
        return meaning
    return Implication(
        finding=meaning.finding, means=meaning.means, favours=meaning.favours, against=against
    )


async def work_out_what_it_means(
    knowledge: TaskKnowledge,
    situation: str,
    options: Sequence[Any] = (),
    *,
    think: Any = None,
    history: Sequence[Any] = (),
) -> list[Implication]:
    """Work out what she read against the position she is actually in.

    Retrieving advice is not applying it. "Keep your largest tile in a corner"
    is a fact about the game; what it means here depends on where the tiles
    actually are, and that comparison is the step between reading something
    and playing differently.

    Reasoned when language is reachable and derived structurally when it is
    not, so the step always happens and only its quality varies.
    """
    if not knowledge.known:
        return []
    meanings: list[Implication] = []
    for finding in knowledge.findings[:FINDINGS_KEPT]:
        said = finding.says
        if think is None:
            meanings.append(_appraised(_structural_meaning(said, situation, options), history))
            continue
        try:
            reply = await think(
                _meaning_question(said, situation, options),
                [f"What I read — {said}", f"What is on screen — {situation}"]
                + [f"Available move — {getattr(option, 'label', lambda: '')()}" for option in options],
            )
        except (RuntimeError, AttributeError, TypeError, ValueError, TimeoutError) as exc:
            record_degradation(
                "task_knowledge", exc, severity="info", action="worked out what a finding meant without language"
            )
            meanings.append(_appraised(_structural_meaning(said, situation, options), history))
            continue
        spoken = " ".join(str(reply or "").split())[:MAX_FINDING_CHARS]
        if not spoken:
            meanings.append(_appraised(_structural_meaning(said, situation, options), history))
            continue
        meanings.append(
            _appraised(
                Implication(finding=said, means=spoken, favours=_favoured_option(spoken, options)),
                history,
            )
        )
    return meanings


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
