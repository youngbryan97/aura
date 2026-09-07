"""The things she could do about herself, as entries rather than as line numbers.

`sequence_induction` used to widen its language down a fixed ladder: try a new
word, and if that returns nothing try an operation, and if that returns nothing
try a way of building, and so on for eight rungs. Every rung ran because the
one above it failed. Nothing in that is a decision, and the ladder itself is a
hand-written taxonomy of what development can be — the exact thing this
question is about.

Here the rungs are entries in a registry. Each says what it changes, what it
costs to try, and what it admits when it works, and those three are what a
value needs. The order they run in is a consequence of what they are worth
rather than of where they sit in a file, and
`where_a_split_disagrees_with_the_whole` is what makes that difference
checkable rather than a claim about style.

A new kind of developmental action needs no edit here. A developmental action
is a place a term can go plus a shape of term to look for, and both are values:
the destinations are the six things she can already revise, and a shape is a
term. `the_action_she_wrote` takes those two and gives back an action, so an
action she invents is admitted by the same call that admits one that was
written down.

What is deliberately not here
-----------------------------
A category of opportunity. There is no list of the kinds of thing that could be
wrong with her, because such a list is the taxonomy again with a different
name. What is wrong is read off the record as a number, and an action is worth
doing when the number says so.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

__all__ = [
    "ADevelopmentalAction",
    "WHAT_SHE_COULD_DO",
    "WHERE_A_TERM_CAN_GO",
    "forget_the_action",
    "the_action_she_wrote",
    "WHAT_THEY_HAVE_DONE",
    "WhatItHasDone",
    "how_wrong_she_was",
    "note_what_it_did",
    "the_actions_she_has",
    "what_it_has_done",
    "what_she_could_do",
]

logger = logging.getLogger("Aura.WhatSheCouldDoNext")


@dataclass(frozen=True, slots=True)
class ADevelopmentalAction:
    """One thing she could do about herself."""

    #: What it is called. Also the key, so admitting the same name twice
    #: replaces rather than duplicates.
    name: str
    #: What it changes about her. One of WHERE_A_TERM_CAN_GO, and the reason
    #: that set is closed is that it is the set of things a term can be
    #: installed as — which is a fact about the floor rather than a policy.
    over: str
    #: What it admits when it works. The key the record's estimators group by,
    #: so what a change of this kind has saved before is what it is estimated
    #: to save now.
    kind: str
    #: Doing it. Given whatever the caller is holding, gives back a note about
    #: what changed, or nothing where it changed nothing.
    do_it: Callable[..., Any]
    #: What trying costs, in candidates walked. Measured where the action
    #: knows, estimated from its own past where it does not.
    price: int = 0
    #: The term, where she wrote one. Nothing for the ones that were written
    #: down, and that difference is reported rather than hidden.
    written: Any = None
    #: Where it came from, for the trace.
    hers: bool = False
    #: What to judge it on, where the action knows better than the caller.
    #: Nothing means the caller's held-out families, which is the default and
    #: the right one for anything that changes the language.
    probe: Callable[[], Any] | None = None
    #: The most it may spend, in candidates. Zero means the ceiling decides,
    #: which is read off the family rather than set here.
    budget: int = 0
    #: Whether what came back counts. Nothing means anything truthy does.
    succeeded: Callable[[Any], bool] | None = None
    #: How to put it back. Nothing means the caller's snapshot does it.
    undo: Callable[[], None] | None = None
    #: shadow, canary, active or retired. A change starts in shadow and is
    #: promoted by evidence rather than by being installed.
    status: str = "active"
    #: Whether this action runs its own held-out test before returning.
    #:
    #: Letting go of a part and naming what two parts share both measure
    #: themselves on families they were not chosen for, and return None when
    #: the change did not pay. Anything that does not say so here is judged by
    #: the generic gate in _and_takes_itself_back, so "it returned a sentence"
    #: stops being the whole of the evidence for keeping a change.
    judges_itself: bool = False
    #: Whether it is about a case in hand or about her.
    #:
    #: Widening a language needs something the language could not say; there is
    #: nothing to widen it towards otherwise. Improving the order she tries
    #: things in needs no case at all — it is scored on the ones she has
    #: already lived. So the two kinds are eligible on different occasions, and
    #: an action asked to work with nothing to work on is ineligible rather
    #: than broken.
    needs_a_case: bool = True


#: The places a term can be installed. Not a taxonomy of development — a list
#: of the things in this codebase that hold a term and can be handed a
#: different one. Each has an installer, a lesion, and a persistence path, and
#: those three are what makes something a destination rather than a wish.
WHERE_A_TERM_CAN_GO: tuple[str, ...] = (
    "the words",
    "the ways of building words",
    "the ways of computing",
    "the shapes a rule can have",
    "the order she tries them in",
    "the proposer",
    "what a change is worth",
)


WHAT_SHE_COULD_DO: dict[str, ADevelopmentalAction] = {}


#: How each kept change was judged, counted since the process started.
#:
#: "held out" means a probe on families the change was not chosen for said it
#: paid. "unmeasured" means the change was kept because it reported doing
#: something and no probe could be built to check it — which is the honest
#: name for what every kept change used to be. "declined" and "did not pay"
#: are the two ways a change leaves nothing behind.
HOW_CHANGES_WERE_JUDGED: dict[str, int] = {
    "held out": 0,
    "unmeasured": 0,
    "judged itself": 0,
    #: An action that installs nothing. Asking for an example produces a
    #: question and changes nothing about her, so no probe has anything to
    #: weigh. Counted apart from "judged itself" because three actions that
    #: really run a held-out test and one that changes nothing read
    #: identically once they share a flag.
    "changes nothing": 0,
    "did not pay": 0,
    "declined": 0,
}


def how_changes_were_judged() -> dict[str, Any]:
    """The evidence behind every change kept this process.

    An external review put the missing invariant as: keeping a candidate
    should require E[capability | candidate] > E[capability | incumbent] under
    an evaluation the candidate was not optimised against. The generic layer
    did not require that — it kept anything that returned a sentence. It
    requires it now wherever a probe can be built, and where one cannot it
    says so instead of pretending, because a number that counts unmeasured
    keeps is what closing the gap is measured against.
    """
    kept = sum(
        HOW_CHANGES_WERE_JUDGED[one]
        for one in ("held out", "unmeasured", "judged itself", "changes nothing")
    )
    try:
        from core.cognition.what_a_change_measured_about_itself import (
            claiming_without_showing,
        )

        unshown = list(claiming_without_showing())
    except (ImportError, RuntimeError):
        unshown = []
    return {
        "schema": "aura.development.evidence.v1",
        "counts": dict(HOW_CHANGES_WERE_JUDGED),
        "kept": kept,
        "kept_without_evidence": HOW_CHANGES_WERE_JUDGED["unmeasured"],
        "share_with_evidence": (
            (kept - HOW_CHANGES_WERE_JUDGED["unmeasured"]) / kept if kept else 0.0
        ),
        #: Actions that took the self-judging opt-out and showed nothing.
        "claiming_without_showing": unshown,
    }


def _held_out_says_it_paid(name: str, before: Any, probe: Any) -> bool | None:
    """Did this change pay on families it was not chosen for?

    None where no probe could be built, which is a different answer from no
    and has to stay different: refusing every change nobody can measure would
    stop development on any faculty without a probe, and calling it evidence
    would be a lie.
    """
    if not probe:
        return None
    try:
        from core.cognition.what_she_does_about_herself import worth_keeping

        paid, _why = worth_keeping(before, probe)
    except (ImportError, RuntimeError, TypeError, ValueError, ZeroDivisionError) as exc:
        logger.debug("could not judge %s on held-out families: %s", name, exc)
        return None
    return bool(paid)


def _and_takes_itself_back(
    name: str, do_it: Callable[..., Any], *, judges_itself: bool = False
) -> Callable[..., Any]:
    """Wrap an action so a change is kept only on evidence, and undone otherwise.

    Two rules, and the second was missing.

    A change that reports it did nothing must have left nothing. That used to
    be each author's job. Naming what two parts share popped the head it
    added; letting go of a part did not put the part back, logged "she kept it
    after all", and returned None over a registry that no longer held it.

    And a change that reports it DID something must have paid for it. The
    generic layer kept anything returning a sentence, so "it returned a
    sentence" was the whole of the evidence for a self-authored mutation.
    Individual actions did better; the layer they all pass through did not.
    Now a held-out probe is taken before and after, and a change that does not
    pay on families it was not chosen for is put back exactly.

    Where no probe can be built the change is kept and counted as unmeasured.
    That is not a gate, and it is not pretending to be one — refusing every
    change nobody can measure would stop development on any faculty without a
    probe, so what happens instead is that the number is visible.
    """
    from functools import wraps

    @wraps(do_it)
    def acted(*args: Any, **kwargs: Any) -> Any:
        from core.cognition.what_she_can_take_back import only_if_it_pays

        before, probe = _how_things_stand()
        with only_if_it_pays(name) as trial:
            said = do_it(*args, **kwargs)
            if said is None:
                HOW_CHANGES_WERE_JUDGED["declined"] += 1
                return None
            if judges_itself:
                # The opt-out has to show its working. It was a boolean in a
                # table that nothing read except this line deciding to stop
                # asking, so the strongest thing the layer could say about a
                # self-judged change was that its author said it was fine.
                from core.cognition.what_a_change_measured_about_itself import (
                    note_a_claim,
                    the_evidence_in,
                )

                verdict = note_a_claim(name, said)
                measured = the_evidence_in(said)
                if measured is not None and not measured.paid:
                    HOW_CHANGES_WERE_JUDGED["did not pay"] += 1
                    logger.info(
                        "%s measured itself on %s and did not pay (%.4f -> %.4f); "
                        "putting it back",
                        name, ", ".join(measured.on), measured.before, measured.after,
                    )
                    return None
                HOW_CHANGES_WERE_JUDGED[verdict] += 1
                trial.keep(str(said))
                return said
            paid = _held_out_says_it_paid(name, before, probe)
            if paid is False:
                HOW_CHANGES_WERE_JUDGED["did not pay"] += 1
                logger.info(
                    "%s changed something and did not pay on held-out families; "
                    "putting it back", name
                )
                return None
            HOW_CHANGES_WERE_JUDGED["held out" if paid else "unmeasured"] += 1
            trial.keep(str(said))
        return said

    return acted


def _how_things_stand() -> tuple[Any, Any]:
    """The held-out probe, and how it stands before a change is made.

    Read before the action runs, because reading it afterwards reads a world
    the action has already changed — which is the same reason `retract` reads
    what rests on a head before removing it.
    """
    try:
        from core.cognition.what_she_does_about_herself import (
            _how_it_stands,  # noqa: PLC2701
            _probe,  # noqa: PLC2701
        )

        probe = _probe()
        return _how_it_stands(probe), probe
    except (ImportError, RuntimeError, TypeError, ValueError) as exc:
        logger.debug("no held-out probe available: %s", exc)
        return {}, ()


def what_she_could_do(
    name: str,
    *,
    over: str,
    kind: str,
    do_it: Callable[..., Any],
    price: int = 0,
    written: Any = None,
    hers: bool = False,
    needs_a_case: bool = True,
    probe: Callable[[], Any] | None = None,
    budget: int = 0,
    succeeded: Callable[[Any], bool] | None = None,
    undo: Callable[[], None] | None = None,
    status: str = "active",
    judges_itself: bool = False,
) -> ADevelopmentalAction:
    """Put an action in the registry. The one call, for hers and for ours."""
    if over not in WHERE_A_TERM_CAN_GO:
        raise ValueError(f"a term cannot go to {over!r}")
    made = ADevelopmentalAction(
        name=str(name),
        over=over,
        kind=str(kind),
        do_it=_and_takes_itself_back(
            str(name), do_it, judges_itself=bool(judges_itself)
        ),
        price=max(0, int(price)),
        written=written,
        hers=bool(hers),
        judges_itself=bool(judges_itself),
        needs_a_case=bool(needs_a_case),
        probe=probe,
        budget=max(0, int(budget)),
        succeeded=succeeded,
        undo=undo,
        status=str(status),
    )
    WHAT_SHE_COULD_DO[made.name] = made
    return made


def the_actions_she_has(
    *, with_a_case: bool = True
) -> tuple[ADevelopmentalAction, ...]:
    """Everything she could do here, in the order they were admitted.

    With no case in hand, only the ones that are about her.
    """
    return tuple(
        one
        for one in WHAT_SHE_COULD_DO.values()
        if with_a_case or not one.needs_a_case
    )


def the_action_she_wrote(
    name: str,
    *,
    over: str,
    look_for: Any,
    kind: str = "",
) -> ADevelopmentalAction:
    """An action she invented: a shape of term, and a place to put it.

    `look_for` is a term taking the situation and giving back the ENCODING of
    a candidate, the way `the_proposer_she_can_replace` does, so the action is
    a value all the way down and what comes back is a term rather than a
    running closure. The installer is whichever one holds `over`, and there is
    no second mechanism for it.
    """
    from core.cognition.the_floor_she_stands_on import Code, build, decode, run

    put_it = _WHERE_IT_GOES.get(over)
    if put_it is None:
        raise ValueError(f"nothing installs at {over!r}")

    # Built once, here, rather than on every call. Surface syntax handed in
    # raw made `run` refuse, the refusal was caught as "gave nothing", and the
    # action returned None forever — an operator she wrote that could not fire
    # and said nothing about why.
    looking = build(look_for)

    def do_it(situation: Any = None) -> Any:
        try:
            made = run(looking, fuel=200_000)
            if hasattr(made, "body"):
                made = run(made.body, (situation, *made.env), fuel=200_000)
            # A quoted term arrives as itself; anything else arrives written
            # down and has to read back as a term or it is not one.
            candidate = made if isinstance(made, Code) else decode(made)
        except Exception as exc:  # noqa: BLE001 - a refusal changes nothing
            logger.info("%s gave nothing: %s", name, exc)
            return None
        return put_it(candidate)

    return what_she_could_do(
        name,
        over=over,
        kind=kind or f"a term for {over}",
        do_it=do_it,
        written=look_for,
        hers=True,
    )


def forget_the_action(name: str) -> ADevelopmentalAction | None:
    """Take one out. The lesion."""
    return WHAT_SHE_COULD_DO.pop(str(name), None)


def _install_a_head(term: Any) -> Any:
    from core.cognition.a_way_of_computing_she_wrote import as_a_head
    from core.cognition.one_algebra import the_head_she_wrote

    return the_head_she_wrote(
        f"a way of computing ({len(WHAT_SHE_COULD_DO)})", 3, as_a_head(term)
    )


def _install_an_order(term: Any) -> Any:
    from core.cognition.the_order_she_tries_them_in import the_order_she_wrote

    return the_order_she_wrote(term)


def _install_a_proposer(term: Any) -> Any:
    from core.cognition.the_proposer_she_can_replace import the_proposer_she_wrote

    return the_proposer_she_wrote(term)


def _install_a_worth(term: Any) -> Any:
    from core.cognition.what_it_is_worth_doing import the_worth_she_wrote

    return the_worth_she_wrote(term)


def _install_a_word(term: Any) -> Any:
    """A word is a term over two numbers: how long the state is, and where.

    WHERE_FROM holds callables of (size, where). Closing the term over exactly
    those two is the whole of it — a word is not a kind of thing either.
    """
    from core.cognition.an_invented_kind import WHERE_FROM
    from core.cognition.the_floor_she_stands_on import A, N, build, run

    name = f"a word ({len(WHERE_FROM)})"
    # `build` passes a term through and compiles surface syntax, so an
    # installer takes whichever a caller has.
    body = build(term)

    def says(size: int, where: int) -> Any:
        # Applied, not seeded. The term takes its two numbers the way any term
        # takes anything; handing them to `run` as an environment instead
        # would work only for a term written against positions.
        return run(build(A(body, N(int(size)), N(int(where)))), fuel=200_000)

    WHERE_FROM[name] = says
    return says


def _install_a_way_of_building(term: Any) -> Any:
    """A way of building is a term with holes, which is what a maker is."""
    from core.cognition.an_invented_kind import WAYS_TO_BUILD
    from core.cognition.one_algebra import as_a_maker
    from core.cognition.the_floor_she_stands_on import build

    name = f"a way of building ({len(WAYS_TO_BUILD)})"
    made = as_a_maker(build(term))
    WAYS_TO_BUILD[name] = made
    return made


def _install_a_shape_for_a_rule(term: Any) -> Any:
    """A rule with no shape is its term plus where it has been held.

    Fitted at nothing and judged at nothing, because an installed term has
    been neither. Those fields are the record of what it survived, and
    writing anything into them here would be writing a result nobody ran.
    """
    from core.cognition.a_rule_with_no_shape import RULES_WITH_NO_SHAPE, Rule
    from core.cognition.the_floor_she_stands_on import build

    name = f"a shape for a rule ({len(RULES_WITH_NO_SHAPE)})"
    made = Rule(body=build(term))
    RULES_WITH_NO_SHAPE[name] = made
    return made


#: Which installer holds each destination. The reason `the_action_she_wrote`
#: needs no edit for a new action is that this table is about the substrate and
#: not about development.
#:
#: Three of the seven destinations had no installer, so an action SHE invented
#: could reach four of the places a hand-written one could reach, and asking
#: for one of the other three raised ValueError before anything ran. That is
#: the gap between "she can write a developmental operator" and "she can write
#: one that goes where the operator needs to go".
_WHERE_IT_GOES: dict[str, Callable[[Any], Any]] = {
    "the words": _install_a_word,
    "the ways of building words": _install_a_way_of_building,
    "the ways of computing": _install_a_head,
    "the shapes a rule can have": _install_a_shape_for_a_rule,
    "the order she tries them in": _install_an_order,
    "the proposer": _install_a_proposer,
    "what a change is worth": _install_a_worth,
}


@dataclass
class WhatItHasDone:
    """What an action has actually produced, kept beside it rather than in it.

    The posterior, and it is a count rather than a belief: how often this
    action was taken, how often the change was kept, and what the held-out
    families gained each time. Everything that calibrates an estimate against
    what happened reads this.

    Kept apart from the action because the action is a definition and this is a
    history, and a definition that changes every time it runs is neither.
    """

    taken: int = 0
    kept: int = 0
    gained: list[int] = field(default_factory=list)

    @property
    def how_often_it_pays(self) -> float:
        """Laplace, so no history means no certainty rather than none."""
        return (self.kept + 1) / (self.taken + 2)

    @property
    def what_it_gains(self) -> float:
        return sum(self.gained) / len(self.gained) if self.gained else 0.0

    def describes(self) -> str:
        return (
            f"kept {self.kept} of {self.taken}, "
            f"{self.what_it_gains:,.0f} each time"
        )


WHAT_THEY_HAVE_DONE: dict[str, WhatItHasDone] = {}


def what_it_has_done(name: str) -> WhatItHasDone:
    return WHAT_THEY_HAVE_DONE.setdefault(str(name), WhatItHasDone())


def note_what_it_did(name: str, *, kept: bool, gained: int = 0) -> WhatItHasDone:
    """Write down what happened, which is what makes the next estimate honest.

    An estimate never checked against an outcome is a rule for producing
    numbers, and a policy scored by one is optimising the number.
    """
    held = what_it_has_done(name)
    held.taken += 1
    if kept:
        held.kept += 1
        held.gained.append(int(gained))
        if len(held.gained) > 64:
            del held.gained[:-64]
    return held


def how_wrong_she_was() -> dict[str, dict[str, float]]:
    """What each action was estimated to gain, against what it gained.

    Calibration. A value that is never held against an outcome will drift, and
    a policy that maximises a drifting value is optimising the drift. What
    comes back is per action: how often it was expected to pay, how often it
    did, and the gap.
    """
    return {
        name: {
            "pays": round(held.how_often_it_pays, 3),
            "gains": round(held.what_it_gains, 1),
            "taken": held.taken,
        }
        for name, held in sorted(WHAT_THEY_HAVE_DONE.items())
    }
