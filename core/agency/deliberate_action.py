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
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Mapping, Sequence

from core.runtime.errors import record_degradation

logger = logging.getLogger(__name__)

#: How many past consequences of one action are worth reading before a move.
RECALL_DEPTH = 5
#: Confidence assigned when nothing has ever been recorded about an option.
UNTRIED_CONFIDENCE = 0.5
#: The shape a decision takes when it reaches the workspace, so anything
#: reading it there knows what it is looking at without guessing.
DECISION_SCHEMA = "aura.decision.v1"

ThinkFn = Callable[[str, Sequence[str]], Awaitable[str]]


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

    def check(self, before: str, after: str) -> "Verdict":
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

    def label(self) -> str:
        return f"{self.name}: {self.detail}" if self.detail else self.name


@dataclass(frozen=True)
class Attempt:
    """A move already made, and what it actually led to."""

    option: str
    expected: str
    verdict: Verdict

    def as_evidence(self) -> str:
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


def recall_consequences(action: str, *, graph: Any = None, depth: int = RECALL_DEPTH) -> list[str]:
    """What happened the last few times this move was made."""
    try:
        if graph is None:
            from core.world_model.acg import acg as graph  # noqa: PLC0415
        rows = graph.query_consequences(action) or []
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        record_degradation("deliberate_action", exc, action="recall consequences of a move")
        return []
    lines: list[str] = []
    for row in list(rows)[:depth]:
        if not isinstance(row, Mapping):
            continue
        outcome = str(row.get("outcome", "")).strip()
        if not outcome:
            continue
        went = "worked" if row.get("success") else "did not work"
        lines.append(f"{action} {went} before: {outcome}")
    return lines


def confidence_from_history(recalled: Sequence[str]) -> float:
    """How much the record supports this move, with no record meaning no opinion."""
    if not recalled:
        return UNTRIED_CONFIDENCE
    worked = sum(1 for line in recalled if " worked before" in line)
    return (worked + 1.0) / (len(recalled) + 2.0)


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
    evidence = [f"Goal: {goal}", f"What is visible now: {situation}"]
    evidence.extend(knowledge)
    evidence.extend(f"Available move — {option.label()}" for option in options)
    evidence.extend(attempt.as_evidence() for attempt in history)
    evidence.extend(recalled)
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
    latest: tuple[int, ActionOption] | None = None
    for option in options:
        where = lowered.rfind(option.name.lower())
        if where < 0:
            continue
        if latest is None or where > latest[0]:
            latest = (where, option)
    return latest[1] if latest else None


def choose_without_language(
    options: Sequence[ActionOption],
    history: Sequence[Attempt] = (),
    recalled: Sequence[str] = (),
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

    failed_recently: dict[str, int] = {}
    tried_at: dict[str, int] = {}
    for position, attempt in enumerate(history):
        tried_at[attempt.option] = position
        if not attempt.verdict.held:
            failed_recently[attempt.option] = failed_recently.get(attempt.option, 0) + 1

    scored: list[tuple[float, int, ActionOption]] = []
    for option in options:
        lines = [line for line in recalled if line.startswith(option.name)]
        score = confidence_from_history(lines)
        # A move that just changed nothing is the worst thing to repeat.
        score -= 0.25 * failed_recently.get(option.name, 0)
        # Among equals, the one left alone longest.
        staleness = -tried_at.get(option.name, -1)
        scored.append((score, staleness, option))

    scored.sort(key=lambda row: (row[0], row[1]), reverse=True)
    best_score, _staleness, best = scored[0]
    if failed_recently.get(best.name):
        why = f"{best.name} is the least bad of what is left"
    elif best_score > UNTRIED_CONFIDENCE:
        why = f"{best.name} has worked here before"
    elif any(attempt.option == best.name for attempt in history):
        why = f"{best.name} is the one left alone longest"
    else:
        why = f"{best.name} has not been tried yet"
    return best, why


async def deliberate(
    goal: str,
    situation: str,
    options: Sequence[ActionOption],
    *,
    think: ThinkFn,
    knowledge: Sequence[str] = (),
    history: Sequence[Attempt] = (),
    stakes: float = 0.5,
    control_point: str = "agency.next_move",
    graph: Any = None,
    spine: Any = None,
    lived: bool = True,
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

    recalled: list[str] = []
    for option in options:
        recalled.extend(recall_consequences(option.name, graph=graph))

    evidence = _situation_evidence(goal, situation, options, history, recalled, knowledge)
    spoke = True
    reply = ""
    try:
        reply = await think(_objective(goal, options), evidence)
    except (RuntimeError, AttributeError, TypeError, ValueError, TimeoutError) as exc:
        # Language being out of reach is not the same as having no judgement.
        #
        # The resident model is her language organ. A move can be chosen from
        # what the consequence graph and the last few attempts already say,
        # and that is what happens here — she acts, and cannot narrate it in
        # her own words until the model is back.
        record_degradation(
            "deliberate_action",
            exc,
            severity="info",
            action="chose the next move without language",
        )
        spoke = False

    chosen = choose_named(reply or "", options) if spoke else None
    if chosen is None:
        structural, why = choose_without_language(options, history, recalled)
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

    for_option = [line for line in recalled if line.startswith(chosen.name)]
    deliberation = Deliberation(
        goal=goal,
        situation=situation,
        chosen=chosen,
        spoke=spoke,
        rationale=_rationale(reply, chosen),
        confidence=confidence_from_history(for_option),
        considered=tuple(option.name for option in options),
        recalled=tuple(recalled),
    )
    deliberation.episode_id = _open_episode(
        deliberation, options, stakes=stakes, control_point=control_point, spine=spine, lived=lived
    )
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
        loop = asyncio.get_running_loop()
    except RuntimeError:
        coroutine.close()
        return
    task = loop.create_task(coroutine)
    # Deciding must not wait on being noticed.
    task.add_done_callback(lambda done: done.exception())


def _rationale(reply: str, chosen: ActionOption) -> str:
    """The sentence in which she settled on this move."""
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
    return reply[start:end].strip()


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
        from core.ontogeny.experience import Episode, Provenance, get_experience_spine  # noqa: PLC0415

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


def confirm(
    deliberation: Deliberation,
    before: str,
    after: str,
    *,
    spine: Any = None,
    graph: Any = None,
) -> Attempt:
    """Check the prediction against what actually happened, and record it.

    The episode is resolved from the measurement, so an action nobody watched
    stays unobserved instead of being scored as a failure.
    """
    if deliberation.chosen is None:
        return Attempt(option="", expected="", verdict=Verdict(held=False, observed_change=False))

    verdict = deliberation.chosen.expectation.check(before, after)
    attempt = Attempt(
        option=deliberation.chosen.name,
        expected=deliberation.chosen.expectation.describes or "the situation to change",
        verdict=verdict,
    )
    _resolve_episode(deliberation, verdict, spine=spine)
    _record_consequence(deliberation, verdict, after, graph=graph)
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


def _record_consequence(deliberation: Deliberation, verdict: Verdict, after: str, *, graph: Any = None) -> None:
    if deliberation.chosen is None:
        return
    try:
        if graph is None:
            from core.world_model.acg import acg as graph  # noqa: PLC0415
        graph.record_outcome(
            deliberation.chosen.name,
            deliberation.situation[:400],
            verdict.why() if not verdict.held else (after[:400] or "as expected"),
            verdict.held,
        )
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        record_degradation("deliberate_action", exc, action="record what a move led to")
