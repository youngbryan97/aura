"""Choosing the next move, with a reason and a prediction attached.

Aura has two loops that drive a goal to a finish: :class:`GoalPursuitEngine`
executes a plan and replans when it stalls, and :meth:`FluidExecutor.pursue`
decides each move from what it can see. Both took the decision itself as an
injected callable, so the judgement lived outside her. This is that judgement.

A deliberation is not only a choice. It carries what she expects to be
different once the move lands, and :func:`confirm` checks that against the
next observation. A prediction that can be wrong is what separates acting
from flailing: the check resolves an episode in the experience spine, and a
broken expectation becomes evidence the next deliberation reads.

Nothing here knows about any particular environment. Options come from
whatever the caller can see and do, and an option carries its own expected
effect, so the reasoning step only ever picks from a closed set of moves that
are really available.
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from core.runtime.errors import record_degradation

logger = logging.getLogger(__name__)

#: How many past consequences of one action are worth reading before a move.
RECALL_DEPTH = 5
#: Confidence assigned when nothing has ever been recorded about an option.
UNTRIED_CONFIDENCE = 0.5
#: How much context one decision gets. Enough for the goal, the reading, the
#: moves, what she has learned and the last several outcomes; not enough for
#: a transcript of the whole run, which is what it grows into otherwise.
EVIDENCE_BUDGET_CHARS = 2400
#: Verbs that mean the words after them are the decision, in the shapes
#: people actually write them.
_DECIDING_VERB = (
    r"press(?:es|ed|ing)?|hit(?:s|ting)?|go(?:es|ing)?|mov(?:e|es|ed|ing)|"
    r"slid(?:e|es|ing)|choos(?:e|es|ing)|chose|pick(?:s|ed|ing)?|"
    r"tak(?:e|es|ing)|took|do(?:es|ing)?|play(?:s|ed|ing)?"
)
#: The shape a decision takes when it reaches the workspace, so anything
#: reading it there knows what it is looking at without guessing.
DECISION_SCHEMA = "aura.decision.v1"

ThinkFn = Callable[[str, Sequence[str]], Awaitable[str]]


class _NotAskedError(Exception):  # noqa: N818 - a control signal, not a failure
    """Raised when a caller deliberately skipped language for this decision.

    Not an error in any sense the name suggests, and named for the rule
    rather than against it: a caller that chose not to spend words on a
    routine move has not failed at anything.
    """


@dataclass(frozen=True)
class Expectation:
    """What should be observably different once a move lands.

    Three predicates cover any observation that reduces to text: the situation
    changed at all, some text is now present, some text is now gone. They are
    checked mechanically, so a prediction is never graded by the same faculty
    that made it.
    """

    changed: bool = True
    contains: tuple[str, ...] = ()
    absent: tuple[str, ...] = ()
    describes: str = ""
    #: A place something must be in, said the way a person says places — a
    #: corner, an edge, the middle. Only meaningful against a reading that
    #: kept its arrangement, and ignored against a flat one.
    at_place: str = ""
    keeping: tuple[str, ...] = ()

    def check(self, before: str, after: str) -> Verdict:
        missing: list[str] = []
        lingering: list[str] = []
        after_lower = after.lower()
        for needle in self.contains:
            if needle.lower() not in after_lower:
                missing.append(needle)
        for needle in self.absent:
            if needle.lower() in after_lower:
                lingering.append(needle)
        moved = before.strip() != after.strip()
        stuck = self.changed and not moved
        held = not (missing or lingering or stuck)
        return Verdict(
            held=held,
            observed_change=moved,
            missing=tuple(missing),
            lingering=tuple(lingering),
            stalled=stuck,
        )

    def check_in(self, before: Any, after: Any) -> Verdict:
        """Check the claim against a reading that kept its arrangement.

        The same claim, asked of the thing rather than of prose about it. A
        move can be checked for having moved something — not merely for the
        text differing, which a tile appearing somewhere else also does — and
        a claim about a place is a question with an answer.
        """
        from core.perception.what_is_there import holds_in  # noqa: PLC0415

        moved = before.as_text() != after.as_text()
        ok, why = holds_in(
            after,
            contains=self.contains,
            absent=self.absent,
            at_place=self.at_place,
            keeping=self.keeping,
        )
        stuck = self.changed and not moved
        missing = (why,) if not ok and "did not appear" in why else ()
        lingering = (why,) if not ok and "still there" in why else ()
        elsewhere = (why,) if not ok and not missing and not lingering else ()
        return Verdict(
            held=ok and not stuck,
            observed_change=moved,
            missing=missing or elsewhere,
            lingering=lingering,
            stalled=stuck,
        )

    def says_something(self) -> bool:
        """Whether this claim could be interestingly wrong.

        A claim that only says the view will differ is satisfied by almost any
        keystroke on almost any screen, so being right by it teaches nothing —
        and the length of her plans, the part of the screen she believes
        answers to her, and the record of what her moves lead to are all read
        off that verdict.
        """
        return bool(self.contains or self.absent or (self.at_place and self.keeping))

    def is_empty(self) -> bool:
        return not (self.contains or self.absent or self.changed)


@dataclass(frozen=True)
class Verdict:
    """Whether what she expected is what she got."""

    held: bool
    observed_change: bool
    missing: tuple[str, ...] = ()
    lingering: tuple[str, ...] = ()
    stalled: bool = False

    def why(self) -> str:
        if self.held:
            return "as expected"
        parts: list[str] = []
        if self.stalled:
            parts.append("nothing changed")
        if self.missing:
            parts.append("never appeared: " + ", ".join(self.missing))
        if self.lingering:
            parts.append("still there: " + ", ".join(self.lingering))
        return "; ".join(parts) or "expectation broken"


@dataclass(frozen=True)
class ActionOption:
    """One move that is really available right now.

    ``expectation`` belongs to the option rather than to the reasoning step.
    An affordance knows what it does; asking a language model to also invent
    its own success criterion lets it grade its own homework.
    """

    name: str
    params: Mapping[str, Any] = field(default_factory=dict)
    detail: str = ""
    expectation: Expectation = field(default_factory=Expectation)
    #: Whether taking this needs a reason in words.
    #:
    #: Most choices are fine to make from evidence — that is what keeps a
    #: loop fast, and the model is her language organ rather than her
    #: decision organ. Some are not: an option that throws away the work so
    #: far is not the kind of thing to reach for because it happened to rank
    #: highest with nothing else working. Measured live, a run with language
    #: out of reach began the task again three times in a hundred seconds,
    #: each time from a ranking, and each restart erased the record the
    #: ranking was built on.
    needs_words: bool = False

    def label(self) -> str:
        return f"{self.name}: {self.detail}" if self.detail else self.name


@dataclass(frozen=True)
class Attempt:
    """A move already made, and what it actually led to."""

    option: str
    expected: str
    verdict: Verdict
    #: Whether the situation moved TOWARD the goal, where that is measurable.
    #: None when there is nothing to measure it against.
    progressed: bool | None = None

    def as_evidence(self) -> str:
        if self.verdict.held and self.progressed:
            return f"{self.option} did what was expected ({self.expected}) and got closer."
        if self.verdict.held and self.progressed is False:
            return f"{self.option} changed things ({self.expected}) but got no closer."
        if self.verdict.held:
            return f"{self.option} did what was expected ({self.expected})."
        return f"{self.option} was expected to {self.expected}, but {self.verdict.why()}."


@dataclass
class Deliberation:
    """A chosen move, why it was chosen, and what it should lead to."""

    goal: str
    situation: str
    chosen: ActionOption | None
    rationale: str = ""
    confidence: float = UNTRIED_CONFIDENCE
    considered: tuple[str, ...] = ()
    recalled: tuple[str, ...] = ()
    episode_id: str | None = None
    reason: str = ""
    #: False when the choice was made without language — she acted, and could
    #: not put it in her own words because the model was not reachable.
    spoke: bool = True
    #: The moves after this one, when she named a sequence rather than a move.
    #: Empty when she named one thing, which is the ordinary case.
    then: tuple[ActionOption, ...] = ()
    #: What she claimed this move would do, when she said something specific.
    #: An option offers a default claim; a decision can carry a sharper one.
    expected: Expectation | None = None
    #: What KIND of situation this was, rather than which one. What the record
    #: is keyed on, so a position she has seen the like of before is
    #: recognised as one.
    shape: str = ""
    decided_at: float = field(default_factory=time.time)

    @property
    def reached(self) -> bool:
        return self.chosen is not None

    def narrate(self) -> str:
        """One line she can say out loud before the move lands.

        A choice made without language still gets a sentence, built from the
        decision rather than generated. She can say what she did and why
        while the organ that writes her sentences is reloading.
        """
        if self.chosen is None:
            return f"I have no move I can justify here — {self.reason}."
        if not self.spoke:
            expects = self.chosen.expectation.describes
            said = f"{self.chosen.label()} — {self.rationale}" if self.rationale else self.chosen.label()
            tail = f" I expect {expects}." if expects else ""
            return f"{said} (deciding without words for a moment).{tail}"
        expects = self.chosen.expectation.describes
        opening = f"{self.chosen.label()}"
        if self.rationale:
            opening = f"{opening} — {self.rationale}"
        return f"{opening}. I expect {expects}." if expects else f"{opening}."


def _how_alike(one: str, other: str) -> float:
    """How much two situations have in common, by what is named in them.

    Words rather than meaning, on purpose: this runs inside a loop that acts
    several times a second, and the things that make two screens alike — the
    values on them, the labels, the state — are named literally in what was
    read. Nothing here is about any one kind of screen.
    """
    here = {word for word in re.findall(r"[\w.]+", str(one or "").lower()) if len(word) > 1}
    there = {word for word in re.findall(r"[\w.]+", str(other or "").lower()) if len(word) > 1}
    if not here or not there:
        return 0.0
    return len(here & there) / float(len(here | there))


def recall_consequences(
    action: str, *, graph: Any = None, depth: int = RECALL_DEPTH, like: str = ""
) -> list[str]:
    """What happened the last few times this move was made.

    ``like`` is the situation she is in now. The same action has different
    consequences in different situations — that is what a situation IS — so
    recalling by action alone hands her the average of every board she has
    ever seen instead of what happened on boards like this one. With it, the
    few she is shown are the closest ones on record.
    """
    try:
        if graph is None:
            from core.world_model.acg import acg as graph  # noqa: PLC0415
        rows = graph.query_consequences(action) or []
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        record_degradation("deliberate_action", exc, action="recall consequences of a move")
        return []
    rows = list(rows)
    if like:
        rows.sort(
            key=lambda row: _how_alike(like, str(row.get("context", "")))
            if isinstance(row, Mapping)
            else 0.0,
            reverse=True,
        )
    lines: list[str] = []
    for row in rows[:depth]:
        if not isinstance(row, Mapping):
            continue
        outcome = str(row.get("outcome", "")).strip()
        if not outcome:
            continue
        went = "worked" if row.get("success") else "did not work"
        lines.append(f"{action} {went} before: {outcome}")
    return lines


#: Below this, the same move going different ways is noise rather than a
#: world that answers the same action in more than one way.
SPREAD_DEPTH = 4


def what_could_happen(action: str, *, graph: Any = None, depth: int = 12) -> str:
    """The different ways this move has gone before, when it has gone more than one.

    A single expected outcome is a bet that the world is deterministic. Where
    it is not — a board that deals a random tile, a page that loads
    differently, anyone else acting at the same time — the record already
    holds the answer: the same action, the same kind of situation, and more
    than one result. Naming the spread lets her decide over what could happen
    instead of over the one thing she expects, and lets her tell a move that
    is genuinely unreliable from one she has simply not tried much.

    Returns "" when the record has nothing to say, because a made-up spread
    is worse than none.
    """
    try:
        if graph is None:
            from core.world_model.acg import acg as graph  # noqa: PLC0415
        rows = list(graph.query_consequences(action) or [])[-depth:]
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        record_degradation("deliberate_action", exc, severity="info", action="weigh what a move could lead to")
        return ""
    if len(rows) < SPREAD_DEPTH:
        return ""
    worked = sum(1 for row in rows if isinstance(row, Mapping) and row.get("success"))
    if worked in (0, len(rows)):
        # It has always gone the same way. That is a prediction, not a spread.
        return ""
    return (
        f"{action} has gone more than one way here: it worked {worked} of the "
        f"last {len(rows)} times, so expect either."
    )


def confidence_from_history(recalled: Sequence[str]) -> float:
    """How much the record supports this move, with no record meaning no opinion."""
    if not recalled:
        return UNTRIED_CONFIDENCE
    grades = [line for line in recalled if " worked before" in line or " did not work before" in line]
    if not grades:
        return UNTRIED_CONFIDENCE
    worked = sum(1 for line in grades if " worked before" in line)
    settled = (worked + 1.0) / (len(grades) + 2.0)
    # A move the world answers in more than one way is not a move she knows,
    # however often it has worked. Confidence is pulled back toward no
    # opinion rather than reported as if the next time were the average.
    if any(" has gone more than one way here" in line for line in recalled):
        return (settled + UNTRIED_CONFIDENCE) / 2.0
    return settled


def _situation_evidence(
    goal: str,
    situation: str,
    options: Sequence[ActionOption],
    history: Sequence[Attempt],
    recalled: Sequence[str],
    knowledge: Sequence[str] = (),
) -> list[str]:
    """Facts the decision rests on. Every line is measured or retrieved.

    ``knowledge`` is what she found out about how this kind of task is done.
    It arrives attributed and sits beside the reading and the consequence
    history rather than above them, because a thing she read is evidence and
    not an instruction.
    """
    # In order of what a decision cannot do without.
    #
    # The goal, what is on screen, and what she can do are the decision. What
    # she has learned and what just happened inform it. What happened further
    # back informs it less, and there is more of it every cycle.
    must_have = [f"Goal: {goal}", f"What is visible now: {situation}"]
    must_have.extend(f"Available move — {option.label()}" for option in options)
    helpful = list(knowledge)
    helpful.extend(attempt.as_evidence() for attempt in reversed(list(history)))
    helpful.extend(recalled)
    return _within_budget(must_have, helpful)


def _within_budget(must_have: list[str], helpful: Sequence[str]) -> list[str]:
    """As much of the useful part as fits, newest first.

    Evidence accumulates: every graded move and every recalled consequence
    adds a line, and none of them ever leave. Measured live, a decision that
    started at 1100 characters was at 6565 by the middle of a run — the
    generation slowed with it until it timed out and she stopped playing.

    A decision needs a bounded amount of context to be a decision. What is
    dropped is the oldest, which is the part already accounted for by what
    came after it.
    """
    evidence = list(must_have)
    room = EVIDENCE_BUDGET_CHARS - sum(len(line) for line in evidence)
    for line in helpful:
        if room <= 0:
            break
        text = str(line or "")
        if not text:
            continue
        evidence.append(text)
        room -= len(text)
    return evidence


def _objective(goal: str, options: Sequence[ActionOption]) -> str:
    names = ", ".join(option.name for option in options)
    return f"Choose the next move toward this goal: {goal}. The available moves are: {names}."


def choose_named(reply: str, options: Sequence[ActionOption]) -> ActionOption | None:
    """Read a choice out of a reply by finding which option it names.

    A reply that works through the moves before settling on one mentions
    several, so the last one named wins: reasoning concludes at the end.
    """
    lowered = reply.lower()

    # A mention that follows a decision verb is the decision.
    #
    # "Last one named" is a good rule for a reply that works through the
    # options and settles, and a bad one for a sentence that names something
    # else afterwards. Measured live: "I'm going to press right because the
    # left column is full" was read as a decision to press left, and she
    # announced a move she had not made.
    decided: tuple[int, int, ActionOption] | None = None
    for option in options:
        name = re.escape(option.name.lower())
        for found in re.finditer(
            # Inflected, because people write "pressing right" as often as
            # "press right" — and a pattern that only matched the bare form
            # fell through to the old rule and picked a noun instead.
            rf"\b(?:{_DECIDING_VERB})\s+(?:the\s+)?{name}\b"
            rf"|\bi(?:'| a)?m\s+(?:going\s+to\s+)?(?:{_DECIDING_VERB})?\s*(?:the\s+)?{name}\b"
            rf"|\b{name}\s+it\s+is\b",
            lowered,
        ):
            ends = found.end()
            if decided is None or (ends, len(option.name)) > (decided[0], decided[1]):
                decided = (ends, len(option.name), option)
    if decided is not None:
        return decided[2]

    best: tuple[int, int, ActionOption] | None = None
    for option in options:
        name = option.name.lower()
        where = lowered.rfind(name)
        if where < 0:
            continue
        # Ranked by where the mention ENDS, then by how specific it is.
        #
        # One option's name can sit inside another's: "slow down" contains
        # "down". Ranking on where a mention starts picks the shorter one and
        # turns a decision about her own pacing into an arrow key. Both end
        # at the same place, so the longer name — the one that accounts for
        # more of what she actually said — wins.
        ends = where + len(name)
        if best is None or (ends, len(name)) > (best[0], best[1]):
            best = (ends, len(name), option)
    return best[2] if best else None


def _distinctive(text: str) -> set[str]:
    """The words in a phrase that carry what it is about."""
    # Function words only. An earlier version also dropped "play" and "game"
    # as too common, which removed the very words that separate a place to
    # play something from an article about it — the discrimination this exists
    # to make.
    common = {
        "the", "a", "an", "and", "or", "of", "to", "in", "on", "at", "for", "with",
        "it", "its", "this", "that", "you", "your", "get", "got", "go", "going",
        "keep", "until", "then", "so", "as", "up", "out", "one", "some", "any",
        "com", "org", "www", "https", "http",
    }
    return {
        word
        for word in re.findall(r"[a-z0-9]+", str(text or "").lower())
        if len(word) > 2 and word not in common
    }


def _describes_the_same_thing(wanted: str, option: ActionOption) -> float:
    """How much an option's own description matches what she is trying to do.

    The only way to tell a place from a page about the place, without asking
    anything: the words each option carries about itself, against the words
    of the goal. Measured live — with the resident model down, every search
    result looked identical and the encyclopedia article won on ordering.
    """
    goal_words = _distinctive(wanted)
    if not goal_words:
        return 0.0
    described = _distinctive(f"{option.name} {option.detail}")
    if not described:
        return 0.0
    return len(goal_words & described) / len(goal_words)


def _foresight_lines(foresight: dict[str, tuple[float, str]] | None) -> list[str]:
    """What she can see each move leading to, as evidence she can read."""
    if not foresight:
        return []
    ranked = sorted(foresight.items(), key=lambda row: row[1][0], reverse=True)
    return [
        f"{name} would leave: {said}" if said else f"{name} looks worth {score:.2f}"
        for name, (score, said) in ranked
    ]


def _best_ahead(
    foresight: dict[str, tuple[float, str]] | None, options: Sequence[ActionOption]
) -> ActionOption | None:
    """The move her own model of the world says leads somewhere best."""
    if not foresight:
        return None
    by_name = {option.name.lower(): option for option in options}
    ranked = sorted(foresight.items(), key=lambda row: row[1][0], reverse=True)
    for name, _scored in ranked:
        option = by_name.get(str(name).strip().lower())
        if option is not None and can_be_part_of_a_plan(option):
            return option
    return None


def choose_without_language(
    options: Sequence[ActionOption],
    history: Sequence[Attempt] = (),
    recalled: Sequence[str] = (),
    wanted: str = "",
    ranked: list[ActionOption] | None = None,
) -> tuple[ActionOption | None, str]:
    """Pick a move from evidence alone, with no language anywhere in it.

    The resident model is her language organ, not her decision organ.
    :mod:`core.cognition.pre_linguistic` says so as an invariant — actions can
    be dispatched even when the LLM is unavailable — and a goal loop that
    stops the moment a model is reloading has broken it. Measured live: a
    pursuit spent every cycle inside a forty-second model reload and ended
    having made no move.

    The policy is the one any evidence supports: prefer what has worked here,
    avoid what just did nothing, and otherwise try whatever has been tried
    least recently. It returns the reason it chose, because a decision nobody
    can explain is not better than no decision.
    """
    if not options:
        return None, "nothing is available to do"

    # Some things are not hers to do on a ranking alone.
    speakable = [option for option in options if not option.needs_words]
    if not speakable:
        return None, "everything available needs a reason I cannot put into words right now"
    options = speakable

    failed_recently: dict[str, int] = {}
    tried_at: dict[str, int] = {}
    for position, attempt in enumerate(history):
        tried_at[attempt.option] = position
        if not attempt.verdict.held:
            failed_recently[attempt.option] = failed_recently.get(attempt.option, 0) + 1

    scored: list[tuple[float, int, ActionOption]] = []
    matched: dict[str, float] = {}
    for option in options:
        lines = [line for line in recalled if line.startswith(option.name)]
        score = confidence_from_history(lines)
        # A move that just changed nothing is the worst thing to repeat.
        score -= 0.25 * failed_recently.get(option.name, 0)
        # What an option says about itself, against what she is trying to do.
        # Weighted to outrank the tie-breaks below without overriding a real
        # record of something working or failing.
        overlap = _describes_the_same_thing(wanted, option)
        matched[option.name] = overlap
        score += 0.4 * overlap
        # Among equals, the one left alone longest.
        staleness = -tried_at.get(option.name, -1)
        scored.append((score, staleness, option))

    scored.sort(key=lambda row: (row[0], row[1]), reverse=True)
    if ranked is not None:
        ranked.extend(option for _score, _staleness, option in scored)
    best_score, _staleness, best = scored[0]
    if failed_recently.get(best.name):
        why = f"{best.name} is the least bad of what is left"
    elif matched.get(best.name, 0.0) > 0.0 and not any(
        line.startswith(best.name) for line in recalled
    ):
        why = f"{best.name} is the one that describes what I am trying to do"
    elif best_score > UNTRIED_CONFIDENCE:
        why = f"{best.name} has worked here before"
    elif any(attempt.option == best.name for attempt in history):
        why = f"{best.name} is the one left alone longest"
    else:
        why = f"{best.name} has not been tried yet"
    return best, why



#: How many moves ahead one decision may commit to.
#:
#: Long enough that the fixed cost of looking and deciding is amortised over
#: several actions, short enough that she is never far from checking. A plan
#: is a bet on the world not changing under it, and the longer the bet the
#: worse it gets.
PLAN_AHEAD = 4


def can_be_part_of_a_plan(option: ActionOption) -> bool:
    """Whether this can follow another action without re-deciding.

    A plan is a sequence of things done to the world, one after the next. Two
    kinds of choice cannot sit in one: a decision about HOW she proceeds
    rather than what she does — pacing herself, changing her mind about the
    task — and anything that needs a reason in words.

    Told apart by what the option expects, which every option already states:
    an action on the world expects the world to be different afterwards, and
    a decision about herself does not. LIVE 2026-08-26: a plan built from
    everything available put "say less" between two moves, and her body tried
    to press it. She narrated "Say less did not land".
    """
    if option.needs_words:
        return False
    return bool(option.expectation.changed or option.expectation.contains)


def plan_without_language(
    ranked: Sequence[ActionOption], limit: int
) -> tuple[ActionOption, ...]:
    """A short sequence from a ranking, when she is deciding without words.

    Repeating one choice with no new information tells her nothing she does
    not already know, and the ranking behind a single choice already holds
    the rest of the answer: what the record supports, what just failed, what
    has been left alone longest. Taking the top few in order is the same
    judgement carried one step further, and every step of it is still checked
    against what actually happened.

    How far it runs is not decided here. It is her measured confidence, which
    is one move whenever her recent predictions have been breaking.
    """
    wanted = max(1, int(limit))
    if wanted <= 1:
        return ()
    seen: list[ActionOption] = []
    for option in ranked:
        if not can_be_part_of_a_plan(option):
            continue
        if option.name not in {chosen.name for chosen in seen}:
            seen.append(option)
        if len(seen) >= wanted:
            break
    return tuple(seen)


def how_far_to_commit(history: Sequence[Attempt], ceiling: int = PLAN_AHEAD) -> int:
    """How many moves ahead it is honest to commit to, given how it has gone.

    A plan is a bet that the world will not change under it, and in a world
    that generates new state after every action the bet decays with each
    step. So the length of a plan is not a setting — it is what her own
    recent predictions say about how predictable this is.

    Every recent prediction held: commit to the ceiling. Any of them broke:
    commit to less. Most of them broke: one move at a time, because that is
    what the evidence supports.
    """
    recent = [attempt for attempt in list(history)[-ceiling:] if attempt is not None]
    if not recent:
        # Nothing measured yet. One move, then find out.
        return 1
    held = sum(1 for attempt in recent if getattr(attempt.verdict, "held", False))
    share = held / len(recent)
    if share >= 1.0:
        return max(1, ceiling)
    if share >= 0.5:
        return max(1, ceiling // 2)
    return 1


def _mentions(
    text: str, by_name: dict, *, nominated: bool = False
) -> list[tuple[int, ActionOption]]:
    """Where each option is named, either anywhere or only where it is chosen."""
    found: list[tuple[int, ActionOption]] = []
    for name, option in by_name.items():
        escaped = re.escape(name)
        pattern = (
            rf"(?:(?:{_DECIDING_VERB})\s+(?:the\s+)?|\bthen\s+|(?:^|[.;:\n]|\d\s*[.)])\s*"
            rf"(?:\*\*)?)({escaped})\b"
            if nominated
            else rf"\b({escaped})\b"
        )
        for match in re.finditer(pattern, text, re.IGNORECASE):
            found.append((match.start(1), option))
    return found


#: What may sit between two moves and leave them one list.
_STILL_THE_LIST = re.compile(r"^[\s,;]*(?:then|and|,)?[\s,;]*$", re.IGNORECASE)


def _a_list_continues(
    text: str, nominated: list[tuple[int, ActionOption]], by_name: dict
) -> list[tuple[int, ActionOption]]:
    """A move that simply follows a nominated one is part of the same list.

    "up left up" nominates its first move by opening the sentence and its
    other two by following it. Without this the list is read as a plan of
    one, which is not what she said.
    """
    if not nominated:
        return nominated
    every = sorted(_mentions(text, by_name), key=lambda row: row[0])
    kept = {where for where, _ in nominated}
    for index, (where, _option) in enumerate(every):
        if where in kept or index == 0:
            continue
        before_where, before_option = every[index - 1]
        if before_where not in kept:
            continue
        gap = text[before_where + len(before_option.name) : where]
        if _STILL_THE_LIST.match(gap):
            kept.add(where)
    return [row for row in every if row[0] in kept]


def choose_sequence(
    reply: str, options: Sequence[ActionOption], limit: int = PLAN_AHEAD
) -> tuple[ActionOption, ...]:
    """An ordered plan read out of a reply, when it names one.

    A single choice is the special case of a plan of one. Reading several is
    what lets a fast loop stop paying to think and look between every action
    — a person playing a game does not re-read the board between the four
    keystrokes of a pattern they have already decided on.

    Only moves that are really available, only in the order they were named,
    and never more than the caller allows.
    """
    text = str(reply or "")
    if not text.strip() or not options:
        return ()
    by_name = {option.name.lower(): option for option in options}
    # A move she named as a move, not a direction word in prose.
    #
    # Every mention used to count, so a plan read out of "1. down (merge
    # 4+4=8) 2. right (shift tiles to the right edge to keep the left open)"
    # came out as down, right, LEFT, down — and she pressed a key her plan
    # never called for. LIVE 2026-08-26.
    #
    # Nominated means what it means anywhere else here: after a deciding
    # verb, after "then", or heading an item in a list. When she nominates
    # nothing — "up left up" — every mention counts, which is what a terse
    # answer means.
    found = _a_list_continues(text, _mentions(text, by_name, nominated=True), by_name)
    found = found or _mentions(text, by_name)
    if not found:
        return ()
    found.sort(key=lambda row: row[0])
    plan: list[ActionOption] = []
    for _where, option in found:
        # A name repeated back to back is one move said twice, not two moves.
        if plan and plan[-1] is option:
            continue
        if plan and not can_be_part_of_a_plan(option):
            # A decision about how she proceeds ends the plan rather than
            # sitting inside it. It stays available as a choice of its own.
            break
        plan.append(option)
        if len(plan) >= max(1, int(limit)):
            break
    return tuple(plan)


async def deliberate(
    goal: str,
    situation: str,
    options: Sequence[ActionOption],
    *,
    think: ThinkFn | None,
    knowledge: Sequence[str] = (),
    history: Sequence[Attempt] = (),
    stakes: float = 0.5,
    control_point: str = "agency.next_move",
    graph: Any = None,
    spine: Any = None,
    lived: bool = True,
    announce: bool = True,
    approach: str = "",
    foresight: dict[str, tuple[float, str]] | None = None,
    seeing: Any = None,
) -> Deliberation:
    """Pick the next move toward ``goal`` from what is available right now.

    ``think`` is her reasoning, wired by the caller to the amplifier or to
    deep deliberation. When it cannot be reached the deliberation comes back
    unreached rather than falling back to a fixed move: a loop that keeps
    acting without a reason to is the failure this organ exists to prevent.
    """
    options = list(options)
    if not options:
        return Deliberation(goal=goal, situation=situation, chosen=None, reason="nothing is available to do")

    # The kind of situation, where the reading kept its arrangement.
    #
    # Two positions that would be approached the same way have to look the
    # same to the record or nothing ever matches and nothing carries over.
    # Keyed on the reading itself — a truncated string of everything on the
    # screen — no two situations were ever alike, which is why forty runs of
    # experience amounted to nothing on the forty-first.
    shape = ""
    as_shape = getattr(seeing, "as_shape", None)
    if callable(as_shape):
        try:
            shape = str(as_shape() or "")
        except (AttributeError, TypeError, ValueError):
            shape = ""
    like = shape or situation

    recalled: list[str] = []
    for option in options:
        recalled.extend(recall_consequences(option.name, graph=graph, like=like))
        # Where the same move has gone more than one way, say so, so she is
        # deciding over what could happen and not over one expected future.
        spread = what_could_happen(option.name, graph=graph)
        if spread:
            recalled.append(spread)

    # What she can see coming, where she has worked out how this thing moves.
    #
    # A third way to reach a move, beside asking her language organ and
    # reading the record: applying what she knows about the world to each
    # option and looking at the result. Cheaper than words and more specific
    # than a memory, and it is the only route that can rule out a move that
    # would do nothing before she spends one on finding out.
    seen_coming = _foresight_lines(foresight)
    evidence = _situation_evidence(
        goal, situation, options, history, [*recalled, *seen_coming], knowledge
    )
    spoke = think is not None
    reply = ""
    try:
        if think is None:
            # No language this time by the caller's choice, not by failure.
            # A fast loop spends words where they change the answer.
            raise _NotAskedError
        reply = await think(_objective(goal, options), evidence)
    except _NotAskedError:
        spoke = False
    except (RuntimeError, AttributeError, TypeError, ValueError, TimeoutError) as exc:
        # Language being out of reach is not the same as having no judgement.
        #
        # The resident model is her language organ. A move can be chosen from
        # what the consequence graph and the last few attempts already say,
        # and that is what happens here — she acts, and cannot narrate it in
        # her own words until the model is back.
        # Logged, not recorded as a fault.
        #
        # Deciding without language is a designed path — the whole point of
        # being able to act while the model reloads — and recording each one
        # as a degradation made a working fallback look like a failing
        # subsystem: 31 of them in half an hour opened a runtime concern
        # incident about deliberate_action while she played perfectly well.
        logger.info("Deciding without language (%s): %s", type(exc).__name__, exc)
        spoke = False

    chosen = choose_named(reply or "", options) if spoke else None
    if chosen is None and spoke and str(reply or "").strip():
        # She answered and it named no move. Worth seeing: the run carries on
        # from evidence, so this is invisible otherwise, and a mind that is
        # answering but not being understood looks exactly like a mind that
        # is not answering.
        record_degradation(
            "deliberate_action",
            ValueError(f"no move named in: {' '.join(str(reply).split())[:160]}"),
            severity="info",
            action="chose from evidence because the reply named no available move",
        )
    ranking: list[ActionOption] = []
    # The judgement about all the options, whether or not she used words.
    #
    # This was worked out only when her words named nothing, so naming a move
    # cost her the ranking behind it — and with it every follow-on. Measured
    # live 2026-08-26: a cycle deciding without words committed to four moves
    # on one screen reading; the same cycle with words committed to one, and
    # the run spent its budget re-reading the board between moves it had
    # already made up its mind about.
    structural, why = choose_without_language(
        options,
        history,
        recalled,
        wanted=f"{goal}. {approach}".strip(". "),
        ranked=ranking,
    )
    if chosen is None:
        # The line she is taking counts here too.
        #
        # A standing approach that only reaches the decisions she puts into
        # words is not an approach, it is a remark. Most moves in a fast loop
        # are decided from evidence, and if her plan cannot reach those, her
        # plan cannot reach most of what she does.
        ahead = _best_ahead(foresight, options)
        if ahead is not None:
            structural = ahead
            why = str((foresight or {}).get(ahead.name, (0.0, ""))[1] or "").strip()
            why = why or "it is the best of the ways this could go"
        if structural is None:
            return Deliberation(
                goal=goal,
                situation=situation,
                chosen=None,
                reason=why,
                considered=tuple(option.name for option in options),
                recalled=tuple(recalled),
            )
        chosen = structural
        reply = why if not spoke else f"{(reply or '').strip()}\n{why}".strip()

    # How far ahead to commit, and to what.
    #
    # Committing is what makes her fast: looking at the screen and deciding
    # cost about one and a half seconds together, a keystroke about a tenth,
    # so a run that looks again between every move of a pattern it already
    # settled on plays at a fraction of the speed it could. Committing is
    # also a bet on the world not changing under her, which is why how far
    # she commits is her own measured confidence rather than a setting — and
    # why every step of it is still checked against what actually happened.
    #
    # Words are not required for it. A ranking is a judgement about all the
    # options, not only the winner, and carrying it one step further is the
    # same judgement rather than a new one.
    far = how_far_to_commit(history)
    if spoke:
        planned = choose_sequence(reply or "", options, far)
        if planned and len(planned) < far:
            # She named one move and is confident enough for several. The
            # rest is the ranking behind the one she named, which is the same
            # judgement carried forward rather than a new one — said out loud
            # as "same plan", and every step still checked against what happened.
            planned = plan_without_language([*planned, *ranking], far) or planned
    else:
        planned = plan_without_language([chosen, *ranking], far)
    logger.info(
        "committing %d ahead (spoke=%s, ranked %d): %s",
        far,
        spoke,
        len(ranking),
        [option.name for option in planned],
    )
    if planned and planned[0] is not chosen:
        # The plan and the settled choice disagree, so there is no plan: what
        # she concluded wins, and one move is a plan of one.
        planned = ()
    for_option = [line for line in recalled if line.startswith(chosen.name)]
    # What she claimed this move would do, when she claimed something.
    #
    # She predicts specifically and always has — "the two 4s in column 1 will
    # merge into an 8" — and none of it reached the check: the move carried
    # the option's own claim that the view would differ. Read here rather than
    # at any one call site, because a decision made anywhere deserves to be
    # graded on what it actually said.
    from core.agency.standing_strategy import claim_in  # noqa: PLC0415

    claimed = claim_in(reason_text := _reason_or_nothing(
        _rationale(reply, chosen, options), [*evidence, _objective(goal, options)], options
    ))
    deliberation = Deliberation(
        goal=goal,
        situation=situation,
        chosen=chosen,
        spoke=spoke,
        rationale=reason_text,
        expected=claimed if claimed.says_something() else None,
        shape=shape,
        then=planned[1:],
        confidence=confidence_from_history(for_option),
        considered=tuple(option.name for option in options),
        recalled=tuple(recalled),
    )
    deliberation.episode_id = _open_episode(
        deliberation, options, stakes=stakes, control_point=control_point, spine=spine, lived=lived
    )
    if announce:
        # A caller that reports the decision itself — at the moment the body
        # acts on it, with the reasoning attached — says so, and this stays
        # quiet rather than describing the same choice twice.
        _announce(deliberation, control_point)
    return deliberation


def _announce(deliberation: Deliberation, control_point: str) -> None:
    """Offer the decision to the global workspace.

    Offered rather than spoken. Whatever is acting carries straight on; a
    narrator, if one is running, hears it only if it wins the broadcast —
    which is the right test, because narrating what she is not attending to
    would not be self-awareness, it would be a log.

    The workspace is also what makes this general: a decision reaching it is
    available to every other faculty, not just to whatever wants to talk.
    """
    if deliberation.chosen is None:
        return
    try:
        from core.consciousness.global_workspace import ContentType  # noqa: PLC0415
        from core.container import ServiceContainer  # noqa: PLC0415

        workspace = ServiceContainer.get("global_workspace", default=None)
        if workspace is None:
            return
        payload = {
            "schema": DECISION_SCHEMA,
            "decision": {
                "control_point": control_point,
                "goal": deliberation.goal,
                "chose": deliberation.chosen.label(),
                "because": deliberation.rationale,
                "expected": deliberation.chosen.expectation.describes,
                "confidence": round(float(deliberation.confidence), 3),
                "spoke": deliberation.spoke,
                "considered": list(deliberation.considered),
                "episode_id": deliberation.episode_id,
            },
        }
        _offer_to_workspace(
            workspace,
            priority=_attention_for(deliberation),
            source=control_point,
            payload=payload,
            reason=deliberation.narrate(),
            content_type=ContentType.INTENTIONAL,
        )
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        record_degradation(
            "deliberate_action",
            exc,
            severity="info",
            action="decided without offering it to the workspace",
        )


def _attention_for(deliberation: Deliberation) -> float:
    """How much a decision deserves to be attended to.

    A move she is unsure of is worth more attention than one she is certain
    of, which is why this rises as confidence falls: the interesting decision
    is the one that might be wrong.
    """
    return max(0.0, min(1.0, 1.0 - float(deliberation.confidence) * 0.5))


def _offer_to_workspace(workspace: Any, **fields: Any) -> None:
    """Submit to the workspace from sync code without waiting on it."""
    publish = getattr(workspace, "publish", None)
    if publish is None:
        return
    coroutine = publish(**fields)
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        coroutine.close()
        return
    # Tracked, not raw. A fire-and-forget task nobody owns is invisible to
    # shutdown and to the runtime's own task census; the tracker gives it a
    # name and an owner while keeping the same non-blocking behaviour.
    from core.utils.task_tracker import get_task_tracker

    task = get_task_tracker().track(coroutine, name="deliberate_action.notice")
    # Deciding must not wait on being noticed.
    task.add_done_callback(lambda done: done.exception())




#: Words that mean a sentence is about what happens rather than about what
#: was asked. A reason either points at the situation or names a consequence.
_TALKS_ABOUT_CONSEQUENCE = re.compile(
    r"\b(?:because|since|so\s|keeps?|keeping|merge[sd]?|merging|avoid[s]?|corner|"
    r"space|room|clear|stuck|full|empty|safe|risk|worked|work[s]?|open|block(?:s|ed)?|"
    r"slide[sd]?|sliding|stack|column|row|edge|wall|combine[sd]?|before|tried)\b",
    re.IGNORECASE,
)


def _says_something_about_here(said: str, evidence: Sequence[str]) -> bool:
    """Whether a rationale is about the situation rather than the question.

    Positive evidence rather than echo-detection: a reason either names
    something on screen or says what a move would do. A sentence with
    neither is the task read back.
    """
    if _TALKS_ABOUT_CONSEQUENCE.search(said):
        return True
    seen = ""
    for line in evidence:
        text = str(line or "")
        if text.startswith("What is visible now:"):
            seen = text
            break
    if not seen:
        return True
    return bool(_distinctive(said) & _distinctive(seen))


def _is_just_the_options(said: str, options: Sequence[ActionOption]) -> bool:
    """Whether a rationale is the list of choices read back.

    "Available moves up/down/left/right" names every option and says nothing
    about any of them. A reason has to contain something the options do not.
    """
    words = _distinctive(said)
    if not words:
        return False
    names = set()
    for option in options:
        names |= _distinctive(f"{option.name} {option.detail}")
    names |= {"available", "move", "moves", "option", "options", "press"}
    return bool(words) and words <= names


def _reason_or_nothing(
    rationale: str, evidence: Sequence[str], options: Sequence[ActionOption] = ()
) -> str:
    """The rationale, unless it is the question or the options read back.

    A reply that repeats what it was given has not reasoned about it, and
    presenting an echo as a reason is worse than saying nothing: it looks
    like thinking. Measured live, a move was narrated as "Board: Up —
    Available moves up/down/left/right", which is the line she was handed.
    """
    said = " ".join(str(rationale or "").split())
    if not said:
        return ""
    if _is_just_the_options(said, options):
        return ""
    if not _says_something_about_here(said, evidence):
        # A restatement of the question. "Need choose next move among
        # up/down/left/right" names the task and says nothing about the board
        # in front of her — asked in different words rather than answered.
        return ""
    plain = said.lower().strip(" .,:;-")
    for line in evidence:
        given = " ".join(str(line or "").split()).lower()
        if not given:
            continue
        if plain and (plain in given or given.endswith(plain)):
            return ""
    return said


#: A conclusion this short carries no reason on its own — "Go up." — so the
#: sentence before it is kept as well.
_BARE_CONCLUSION_CHARS = 32


def _rationale(reply: str, chosen: ActionOption, options: Sequence[ActionOption] = ()) -> str:
    """The sentence in which she settled on this move, and why.

    A conclusion often follows its reason rather than containing it:
    "Keeping the corner matters most. Go up." Taking only the sentence with
    the move in it keeps "Go up" and throws away the thinking, which then
    reads as no reason at all.
    """
    lowered = (reply or "").lower()
    where = lowered.rfind(chosen.name.lower())
    if where < 0:
        return (reply or "").strip()
    start = max(lowered.rfind(".", 0, where), lowered.rfind("\n", 0, where)) + 1
    end = len(reply)
    for stop in (".", "\n"):
        found = reply.find(stop, where)
        if found >= 0:
            end = min(end, found + 1)
    settled = reply[start:end].strip()
    if len(settled) >= _BARE_CONCLUSION_CHARS or start <= 0:
        return settled if _is_about(settled, chosen, options) else ""
    before_start = max(
        lowered.rfind(".", 0, max(0, start - 1)), lowered.rfind("\n", 0, max(0, start - 1))
    ) + 1
    leading = reply[before_start:start].strip()
    said = f"{leading} {settled}".strip() if leading else settled
    return said if _is_about(said, chosen, options) else ""


def _is_about(said: str, chosen: ActionOption, options: Sequence[ActionOption]) -> bool:
    """Whether this sentence is her settling on THIS move rather than another.

    A move's name can sit inside a phrase about the board — "left" inside
    "the left side" — so the sentence found around it can be the one where
    she chose something else. Judged by the same reader that decides the
    move, because a sentence that reads as a decision to go down is not the
    reason she went left.

    LIVE 2026-08-26: "Going left — I choose down because the two 4s in
    column 1 will merge into an 8, consolidating the left side."
    """
    if not options:
        return True
    named = choose_named(said, options)
    return named is None or named.name == chosen.name


def _open_episode(
    deliberation: Deliberation,
    options: Sequence[ActionOption],
    *,
    stakes: float,
    control_point: str,
    spine: Any = None,
    lived: bool = True,
) -> str | None:
    """Record the decision now, with only what was true when it was made.

    ``lived`` is what keeps a rehearsal out of her history. The spine refuses
    anything but lived experience into a live store, but only if the producer
    says which it is, so a caller that is exercising the loop rather than
    living it must say so and the refusal happens structurally.
    """
    if deliberation.chosen is None:
        return None
    try:
        from core.ontogeny.experience import (  # noqa: PLC0415
            Episode,
            Provenance,
            get_experience_spine,
        )

        store = spine if spine is not None else get_experience_spine()
        episode = Episode(
            provenance=Provenance.LIVE if lived else Provenance.TEST,
            control_point=control_point,
            features={
                "options": float(len(options)),
                "confidence": float(deliberation.confidence),
                "recalled": float(len(deliberation.recalled)),
            },
            decision=deliberation.chosen.name,
            options=tuple(option.name for option in options),
            decider="agency.deliberate_action",
            stakes=float(stakes),
            context={"goal": deliberation.goal, "expected": deliberation.chosen.expectation.describes},
        )
        return store.record(episode)
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        record_degradation("deliberate_action", exc, action="open an episode for a move")
        return None


def _highest_in(text: str) -> int:
    values = [
        int(found.replace(",", ""))
        for found in re.findall(r"\b(\d[\d,]{0,9})\b", str(text or ""))
    ]
    return max(values) if values else 0


def made_progress(before: str, after: str, toward: str) -> bool | None:
    """Whether the situation moved toward the goal, where that is measurable.

    "It worked" has meant "the view is different", which is the honest test
    that an action HAPPENED and a poor test of whether it helped. A move that
    shuffles things scores exactly as well as one that builds, so a record
    built from it prefers whatever changes the screen most reliably rather
    than whatever gets anywhere. Measured live: she pressed the same
    direction over and over because it kept "working".

    Where the goal names a value, closer is computable — the largest value in
    front of her rose. Where it does not, there is nothing to measure and
    this says so rather than guessing.
    """
    wanted = str(toward or "").strip()
    if not re.fullmatch(r"\d[\d,]*", wanted):
        return None
    was, now = _highest_in(before), _highest_in(after)
    if not was and not now:
        return None
    return now > was


def confirm(
    deliberation: Deliberation,
    before: str,
    after: str,
    *,
    spine: Any = None,
    graph: Any = None,
    toward: str = "",
    seen_before: Any = None,
    seen_after: Any = None,
) -> Attempt:
    """Check the prediction against what actually happened, and record it.

    The episode is resolved from the measurement, so an action nobody watched
    stays unobserved instead of being scored as a failure.
    """
    if deliberation.chosen is None:
        return Attempt(option="", expected="", verdict=Verdict(held=False, observed_change=False))

    # What she claimed, when she claimed something, checked against the thing
    # rather than against prose about it.
    #
    # An option's own claim is that the view will differ, which is satisfied
    # by almost any keystroke on almost any screen. Live 2026-08-26: 17 of 20
    # such predictions held, and holding told her nothing, while the length of
    # her plans and the record of what her moves lead to both read that
    # verdict as though it were confidence.
    claim = deliberation.expected or deliberation.chosen.expectation
    if seen_before is not None and seen_after is not None:
        verdict = claim.check_in(seen_before, seen_after)
    else:
        verdict = claim.check(before, after)
    moved_on = made_progress(before, after, toward)
    attempt = Attempt(
        option=deliberation.chosen.name,
        expected=claim.describes or "the situation to change",
        verdict=verdict,
        progressed=moved_on,
    )
    logger.info(
        "%s: predicted %r, held=%s, closer=%s",
        attempt.option,
        attempt.expected[:60],
        verdict.held,
        moved_on,
    )
    _resolve_episode(deliberation, verdict, spine=spine)
    _record_consequence(deliberation, verdict, after, graph=graph, progressed=moved_on)
    _announce_outcome(deliberation, attempt, verdict)
    return attempt


def _announce_outcome(deliberation: Deliberation, attempt: Attempt, verdict: Verdict) -> None:
    """Offer what actually happened, so a broken prediction can be said too."""
    try:
        from core.consciousness.global_workspace import ContentType  # noqa: PLC0415
        from core.container import ServiceContainer  # noqa: PLC0415

        workspace = ServiceContainer.get("global_workspace", default=None)
        if workspace is None:
            return
        _offer_to_workspace(
            workspace,
            # A prediction that broke is the more interesting of the two, and
            # is the one worth interrupting for.
            priority=0.8 if not verdict.held else 0.3,
            source="agency.confirm",
            payload={
                "schema": DECISION_SCHEMA,
                "outcome": {
                    "chose": attempt.option,
                    "held": verdict.held,
                    "why": verdict.why(),
                    "episode_id": deliberation.episode_id,
                },
            },
            reason=attempt.as_evidence(),
            content_type=ContentType.META,
        )
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        record_degradation("deliberate_action", exc, severity="info", action="graded a move without announcing it")


def _resolve_episode(deliberation: Deliberation, verdict: Verdict, *, spine: Any = None) -> None:
    if not deliberation.episode_id:
        return
    try:
        from core.ontogeny.experience import Outcome, get_experience_spine  # noqa: PLC0415

        store = spine if spine is not None else get_experience_spine()
        outcome = Outcome.from_utility(
            1.0 if verdict.held else 0.0,
            "agency.deliberate_action.confirm",
            why=verdict.why(),
        )
        store.resolve(deliberation.episode_id, outcome)
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        record_degradation("deliberate_action", exc, action="resolve an episode for a move")


def _record_consequence(
    deliberation: Deliberation,
    verdict: Verdict,
    after: str,
    *,
    graph: Any = None,
    progressed: bool | None = None,
) -> None:
    if deliberation.chosen is None:
        return
    try:
        if graph is None:
            from core.world_model.acg import acg as graph  # noqa: PLC0415
        # Recorded as a success only when it also got somewhere, where that
        # is measurable. A move that changes the view without advancing the
        # goal is not a move worth reaching for again, and recording it as
        # one is how a record comes to prefer whatever reliably shuffles
        # things.
        worked = verdict.held if progressed is None else bool(verdict.held and progressed)
        said = verdict.why() if not verdict.held else (after[:400] or "as expected")
        if verdict.held and progressed is False:
            said = f"changed things but got no closer: {said}"
        graph.record_outcome(
            deliberation.chosen.name,
            deliberation.shape or deliberation.situation[:400],
            said,
            worked,
        )
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        record_degradation("deliberate_action", exc, action="record what a move led to")
