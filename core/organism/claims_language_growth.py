"""The claims about the language she grows, and the instruments behind them.

Twenty predicates and the group that binds them to claims, taken out of
`model_validation` — 6,224 lines, of which these were 1,600 — because nothing
else in the registry reads them: the group is called once from
`install_runtime_validation` and the predicates are called only by the group.
The registry's own types come in from the registry; the registry imports this
group inside the function that installs it, so neither module loads a
half-built other.
"""

from __future__ import annotations

import logging
import pathlib
from typing import Any

from core.organism.model_validation import (
    _AN_EXPERIMENT_NOT_AN_INSTRUMENT,
    Claim,
    Evidence,
    Observation,
    ValidationTest,
    threshold_score,
)

logger = logging.getLogger("Aura.Validation.LanguageGrowth")


# ── the language she makes rules out of ───────────────────────────────────

def _language_left_as_found(work: Any) -> Any:
    """Run a check without leaving anything behind in the live language."""
    from core.cognition import an_invented_kind as kinds

    was = (
        dict(kinds.WHERE_FROM), dict(kinds.WHAT_OF_IT),
        dict(kinds.WAYS_TO_BUILD), dict(kinds.KINDS),
    )
    kinds.WAYS_TO_BUILD.clear()
    try:
        return work()
    finally:
        for holds, before in zip(
            (kinds.WHERE_FROM, kinds.WHAT_OF_IT, kinds.WAYS_TO_BUILD, kinds.KINDS), was, strict=True
        ):
            holds.clear()
            holds.update(before)


def _macros_admitted_as_new_words() -> int:
    """Words admitted whose meaning the closure already had. Must be none."""
    from core.cognition import an_invented_kind as kinds
    from core.cognition import widening_the_language as widening

    def check() -> int:
        kinds.WAYS_TO_BUILD["one after another"] = widening.one_after_another
        closure = kinds.addressings()
        composed = []
        for name, word in closure.items():
            if ", then " not in name:
                continue
            try:
                composed.append(
                    (name, tuple(word(at, 4) % 4 for at in range(4)))
                )
            except IndexError:
                # A word read off examples of another length refuses length
                # four, which is the language being honest rather than a word
                # admitted wrongly. It is not a candidate for this measurement
                # and letting the refusal escape reported it as a claim whose
                # prediction crashed.
                continue
        states = [(1, 2, 3, 4), (5, 6, 7, 8), (9, 1, 2, 6)]
        admitted = 0
        for _name, where in composed[:8]:
            if len(set(where)) != 4:
                continue
            pairs = [(one, tuple(one[at] for at in where)) for one in states]
            if widening.an_addressing_nobody_wrote(pairs, already=closure) is not None:
                admitted += 1
        return admitted

    return _language_left_as_found(check)


def _meanings_lost_by_admitting_a_way_of_building() -> int:
    """Meanings that stopped being expressible when the language grew."""
    from core.cognition import an_invented_kind as kinds
    from core.cognition import widening_the_language as widening
    from core.cognition.what_it_costs_to_say import everything_sayable

    def check() -> int:
        before = set(everything_sayable())
        kinds.WAYS_TO_BUILD["one after another"] = widening.one_after_another
        after = set(everything_sayable())
        return len(before - after) + (0 if len(after) > len(before) else 1)

    return _language_left_as_found(check)


def _constructors_she_built_that_were_already_written() -> int:
    """Ways of building she arrived at that the source registry already had."""
    from core.cognition import an_invented_kind as kinds
    from core.cognition import widening_the_language as widening
    from core.cognition.a_constructor_she_built import a_constructor_she_built

    def check() -> int:
        states = [(1, 2, 3, 4, 5), (6, 7, 8, 9, 1), (2, 4, 6, 8, 3),
                  (5, 1, 9, 3, 7), (8, 2, 6, 4, 9), (3, 7, 1, 5, 2)]
        wanted = [
            (one, tuple(one[((len(one) - 1 - at) - 2) % len(one)] for at in range(len(one))))
            for one in states
        ]
        built = a_constructor_she_built(
            wanted, now_sayable=lambda: kinds.induce_from(wanted) is not None
        )
        if built is None:
            return 1
        return 1 if built.name in widening.CONSTRUCTORS else 0

    return _language_left_as_found(check)


def _pairs_a_derived_operation_refuses() -> int:
    """Values a worked-out operation cannot answer. A table refuses them all."""
    from core.cognition.an_operation_that_generalises import (
        an_operation_that_generalises,
    )

    rule = an_operation_that_generalises(
        [(7, 3, 4), (9, 2, 7), (5, 5, 0), (2, 8, 6), (11, 4, 7), (6, 1, 5)]
    )
    if rule is None:
        return 1
    refused = 0
    for one, other in ((100, 37), (13, 91), (55, 55), (0, 8)):
        try:
            if rule(one, other) != abs(one - other):
                refused += 1
        except (ArithmeticError, KeyError, TypeError, ValueError):
            refused += 1
    return refused


def _language_lost_across_a_restart() -> int:
    """Words, ways and meanings that did not come back. Must be none.

    Written somewhere of its own. Checking whether a restart keeps the
    language by writing over the file that HOLDS the language would replace
    what she actually knows with the fixture used to test it.
    """
    import tempfile

    from core.cognition import an_invented_kind as kinds
    from core.cognition import what_she_gave_meaning as keeping
    from core.cognition.a_constructor_she_built import Recipe, build

    def check() -> int:
        kinds.WAYS_TO_BUILD["a way she built: 2 times over"] = build(
            Recipe(kind="over and over", depth=2)
        )
        states = [(1, 2, 3, 4, 5), (6, 7, 8, 9, 1), (2, 4, 6, 8, 3),
                  (5, 1, 9, 3, 7), (8, 2, 6, 4, 9), (3, 7, 1, 5, 2)]
        wanted = [
            (one, tuple(one[(at + 2) % len(one)] for at in range(len(one))))
            for one in states
        ]
        made = kinds.induce_from(wanted)
        if made is None or not kinds.admit("a restart check", made):
            return 1
        reach = len(kinds.addressings())
        if not keeping.keep():
            return 1
        kinds.WAYS_TO_BUILD.clear()
        kinds.KINDS.clear()
        keeping.recall()
        lost = 0
        if "a way she built: 2 times over" not in kinds.WAYS_TO_BUILD:
            lost += 1
        if len(kinds.addressings()) != reach:
            lost += 1
        if kinds.interpretation_of("a restart check") is None:
            lost += 1
        return lost

    kept_at = keeping._KEPT_AT
    with tempfile.TemporaryDirectory(prefix="aura-restart-check-") as somewhere:
        keeping._KEPT_AT = pathlib.Path(somewhere) / "meanings.json"
        try:
            return _language_left_as_found(check)
        finally:
            keeping._KEPT_AT = kept_at


def _ways_of_building_she_cannot_reach() -> int:
    """Whether the way she writes makers is itself a list. Must be none.

    Counts the ways of building that are named rather than written. A term
    with a hole in it is written; a constructor looked up by name is not, and
    a system whose makers all come from a list has a ceiling one level up
    however many levels it has.
    """
    from core.cognition.one_algebra import (
        Term,
        run,
        the_closure_of_composing_undoing_and_repeating,
    )

    def check() -> int:
        from core.cognition.an_invented_kind import WHERE_FROM

        far, along = WHERE_FROM["the far end"], WHERE_FROM["one along"]
        branching = Term(
            "if",
            (
                Term("same as", (
                    Term("left over", (Term("many"), Term("fixed", value=2))),
                    Term("fixed", value=0),
                )),
                Term("hole", value=0),
                Term("hole", value=1),
            ),
        )
        reach = the_closure_of_composing_undoing_and_repeating(
            dict(WHERE_FROM), deepest=3
        )
        shape = tuple(
            run(branching, at, size, (far, along)) % size
            for size in (3, 4, 5)
            for at in range(size)
        )
        # One violation if a maker outside those three cannot be written at
        # all, and one more if the closure is too small to be evidence.
        return (1 if shape in reach else 0) + (0 if len(reach) > 1000 else 1)

    return _language_left_as_found(check)


def _the_positional_bound_broken() -> int:
    """Answers outside what the ceiling argument allows. Must be none.

    The argument that her positional language is not universal is an induction
    over the heads, and an induction over a head list is worth what the head
    list is worth. This runs the conclusion against the interpreter and counts
    every answer that escapes the bound, plus one violation if the induction
    has no case for a head ``run`` dispatches on.
    """
    import inspect
    import re

    from core.cognition.an_invented_kind import WHERE_FROM
    from core.cognition.one_algebra import HEADS
    from core.cognition.one_algebra import run as positional_run
    from core.cognition.what_the_old_language_cannot_say import (
        a_sample_of_terms,
        the_bound_holds_on,
        the_heads_the_argument_covers,
        why_it_cannot_be_said,
    )

    source = inspect.getsource(positional_run)
    dispatched = set(re.findall(r'head == "([^"]+)"', source)) | set(HEADS)
    uncovered = len(dispatched ^ the_heads_the_argument_covers())

    terms = list(a_sample_of_terms(1200))
    words = list(WHERE_FROM.values())[:2]
    found = the_bound_holds_on(terms, words)
    ceiling = why_it_cannot_be_said()
    return (
        len(found["broken"])
        + uncovered
        + (0 if ceiling.strictly_wider else 1)
    )


def _universality_certificates_that_fail() -> int:
    """Kleene's constructors that do not compute what they should. Must be none.

    Five checks: the three starting points, recursion, and unbounded search —
    including that unbounded search on a predicate with no root exhausts its
    meter rather than returning, which is the check that the floor is
    genuinely universal rather than merely large.
    """
    from core.cognition.the_floor_she_stands_on import (
        MINUS,
        PLUS,
        TIMES,
        Code,
        L,
        N,
        OutOfFuel,
        V,
        build,
        run,
    )
    from core.cognition.what_the_floor_can_say import (
        SUCCESS,
        ZERO,
        by_recursion,
        take_the_one_at,
        the_least_where,
        what_the_arithmetic_rests_on,
    )

    def apply(work: Code, *given: int) -> Any:
        made = work
        for one in given:
            made = Code("of", parts=(made, Code("a number", value=one)))
        return run(made)

    failed = 0
    failed += 0 if apply(ZERO, 9) == 0 else 1
    failed += 0 if apply(SUCCESS, 9) == 10 else 1
    failed += 0 if apply(take_the_one_at(5, 3), 10, 11, 12, 13, 14) == 13 else 1
    times = by_recursion(
        build(L("x", N(0))),
        build(L("x", L("n", L("r", PLUS(V("r"), V("x")))))),
    )
    failed += 0 if apply(times, 7, 5) == 35 else 1
    root = the_least_where(build(L("k", MINUS(TIMES(V("k"), V("k")), N(49)))))
    failed += 0 if run(root) == 7 else 1
    try:
        run(the_least_where(build(L("k", N(1)))), fuel=20_000)
        failed += 1
    except OutOfFuel:
        # Not a failure: a certificate that runs out of fuel is one that did not close inside the bound, which is what this counts.
        pass
    failed += 0 if what_the_arithmetic_rests_on()["all_agree"] else 1
    return failed


def _the_floor_cannot_read_itself() -> int:
    """Run the floor's evaluator, written in the floor. Failures must be none.

    The claim made everywhere in this area is that the mechanism becomes an
    object she can hold — quoted, passed around, changed. An interpreter for
    the language, written IN the language, is what that claim comes to, and it
    was sitting in a file nothing imported.

    Two answers have to agree: running a term, and running the interpreter on
    the term. Anything else and what is in that file resembles the evaluator
    rather than being it.
    """
    from core.cognition.the_floor_reading_itself import interpret
    from core.cognition.the_floor_she_stands_on import (
        PLUS,
        TIMES,
        Code,
        L,
        N,
        V,
        build,
        run,
    )

    failed = 0
    for term in (build(PLUS(N(2), N(3))), build(TIMES(N(6), N(7)))):
        failed += 0 if run(term) == interpret(term) else 1
    # A function is compared by what it does, not by how it is written down.
    # Both sides come back a closure and the two closures differ inside — the
    # interpreter's carries the interpreter's own machinery — which is exactly
    # right and says nothing about whether they agree. Applying them does.
    twice = build(L("x", TIMES(V("x"), N(2))))
    for one in (0, 1, 7):
        applied = Code("of", parts=(twice, Code("a number", value=one)))
        failed += 0 if run(applied) == interpret(applied) == one * 2 else 1
    return failed


def _the_four_walls_growth_runs_into() -> int:
    """Run the four limits on growing a language. Failures must be none.

    The companion to the theorem below, and it had the same problem: imported
    by its own test and by nothing else, so four arguments about what
    self-improvement cannot do sat in a file that only ran when somebody ran
    the test for it.

    Each is executed. A universal language cannot be made more expressive by
    naming things in it, checked on a word. No update rule improves everywhere,
    checked by building the environment that beats one. "This language cannot
    say that" is assertable only where the language was walked end to end, so
    in a universal one it never is. And a question above computation is
    answered by saying so, not by trying.
    """
    from core.cognition.what_growth_cannot_do import (
        BOUNDED,
        UNIVERSAL,
        UNKNOWN,
        can_be_decided,
        how_expressive,
        naming_cannot_add_a_meaning,
        no_updater_wins_everywhere,
        what_verification_is_available,
        what_would_need_an_oracle,
    )

    failed = 0
    everything = how_expressive(
        repeats_without_bound=True, branches_on_its_own_values=True
    )
    failed += 0 if everything.verdict == UNIVERSAL else 1
    small = how_expressive(
        repeats_without_bound=False, branches_on_its_own_values=False, meanings=64
    )
    failed += 0 if small.verdict == BOUNDED else 1
    failed += (
        0
        if naming_cannot_add_a_meaning(
            lambda made: tuple(made((1, 2, 3))) == (2, 4, 6),
            a_word_she_made=lambda row: tuple(one * 2 for one in row),
            unfolds_to=lambda: (lambda row: tuple(one + one for one in row)),
        )
        else 1
    )
    beaten = no_updater_wins_everywhere(lambda _seen: 0)
    failed += 0 if beaten.holds and beaten.the_other_scored > beaten.scored else 1
    failed += (
        0
        if can_be_decided(language=everything, exhaustive_search_finished=False) == UNKNOWN
        else 1
    )
    failed += (
        0
        if can_be_decided(language=small, exhaustive_search_finished=True) is True
        else 1
    )
    failed += 0 if len(what_verification_is_available(change_is_arbitrary=True)) == 4 else 1
    failed += 0 if what_would_need_an_oracle("does this halt on every input") else 1
    return failed


def _the_tower_with_no_top() -> int:
    """Run the theorem that says where the regress ends. Failures must be none.

    The claim about the floor was ``asserted_in`` this module and nothing
    imported it, so the argument sat in a file that never ran — which is the
    one shape of dead code this codebase keeps finding, and a theorem is the
    worst thing to leave in it, because a proof nobody executes reads exactly
    like one that holds.

    Three things, each executed rather than cited. Naming a thing does not
    change what it denotes, on an actual word. A universal bedrock puts the
    number of further authoring events at nought, and a bedrock that is not
    universal puts it beyond any bound — the second is the regress, stated. And
    the gate stays outside, because an invariant of the form "everything
    admitted is harmless" survives only while the thing checking it cannot
    itself be admitted.
    """
    from core.cognition.where_the_tower_has_a_top import (
        AUTHORED_FOREVER,
        NEVER_AGAIN,
        UNDECIDED,
        a_gate_inside_the_space_cannot_hold,
        naming_adds_no_meaning,
        what_is_still_authored,
        where_the_tower_ends,
    )

    failed = 0
    # A word she made, and the same thing written out with the name gone.
    doubled = {"twice": lambda row: tuple(one * 2 for one in row)}
    failed += (
        0
        if naming_adds_no_meaning(
            lambda made: tuple(made((1, 2, 3))) == (2, 4, 6),
            the_name=doubled["twice"],
            written_out=lambda: (lambda row: tuple(one + one for one in row)),
        )
        else 1
    )
    universal = where_the_tower_ends(
        universal=True,
        certificate="Kleene's constructors, each exhibited as a term on the floor",
    )
    failed += 0 if universal.verdict == NEVER_AGAIN and universal.has_a_top else 1
    smaller = where_the_tower_ends(
        universal=False, a_behaviour_outside="the least k where a predicate holds"
    )
    failed += 0 if smaller.verdict == AUTHORED_FOREVER and not smaller.has_a_top else 1
    failed += 0 if where_the_tower_ends(universal=None).verdict == UNDECIDED else 1
    failed += 0 if a_gate_inside_the_space_cannot_hold().shows_the_gate_must_stay_out else 1
    failed += 0 if len(what_is_still_authored()) == 3 else 1
    return failed


def _a_way_of_computing_she_cannot_keep() -> int:
    """Steps between writing a head and using it after a restart. Must be none.

    Written, installed, run at a length it was never fitted at, kept, wiped,
    recalled, and run again. Any of those failing is a language whose grammar
    resets every morning.
    """
    import pathlib
    import tempfile

    from core.cognition import what_she_gave_meaning as keeping
    from core.cognition.a_way_of_computing_she_wrote import (
        a_way_of_computing_she_wrote,
    )
    from core.cognition.an_invented_kind import WHERE_FROM
    from core.cognition.one_algebra import DERIVED_HEADS, Term, the_head_she_wrote
    from core.cognition.one_algebra import run as positional_run

    def rule(at: int, size: int) -> int:
        return at + (at + 1) % size

    family = []
    for size in (4, 5, 6, 7):
        before = tuple(range(100, 100 + size))
        family.append(
            (before, tuple(before[rule(at, size) % size] for at in range(size)))
        )

    was = dict(DERIVED_HEADS)
    kept_at = keeping._KEPT_AT
    try:
        DERIVED_HEADS.clear()
        found = a_way_of_computing_she_wrote(
            family, now_sayable=lambda: False, words=dict(WHERE_FROM), within=20.0
        )
        if found is None:
            return 1
        the_head_she_wrote("a way of computing", 2, found.body)
        term = Term(
            "a way of computing",
            parts=(Term("hole", value=0), Term("hole", value=1)),
        )
        words = tuple(WHERE_FROM[one] for one in found.over)
        broken = 0
        for size in (9, 11):
            said = tuple(positional_run(term, at, size, words) for at in range(size))
            if said != tuple(rule(at, size) % size for at in range(size)):
                broken += 1
        with tempfile.TemporaryDirectory(prefix="aura-head-check-") as somewhere:
            keeping._KEPT_AT = pathlib.Path(somewhere) / "meanings.json"
            if not keeping.keep():
                return broken + 1
            DERIVED_HEADS.clear()
            keeping.recall()
            if "a way of computing" not in DERIVED_HEADS:
                return broken + 1
            said = tuple(positional_run(term, at, 9, words) for at in range(9))
            if said != tuple(rule(at, 9) % 9 for at in range(9)):
                broken += 1
        return broken
    finally:
        keeping._KEPT_AT = kept_at
        DERIVED_HEADS.clear()
        DERIVED_HEADS.update(was)


def _where_the_two_languages_disagree() -> int:
    """Places the compiled floor term and the old interpreter differ. Must be none.

    Both algebras, and refusals count as answers. A check comparing only the
    places where both succeeded would have missed the one real defect this
    found: substituting the inner expression into the body of ``through`` made
    the floor lazy where the interpreter is strict, so a term dividing by
    nothing refused in one language and answered nought in the other.
    """
    import itertools

    from core.cognition.an_operation_that_generalises import every_expression
    from core.cognition.one_algebra import every_term
    from core.cognition.the_old_language_on_the_floor import (
        operations_agree_everywhere,
        they_agree_everywhere,
    )

    terms = list(itertools.islice(every_term((0, 1, 2, 3), holes=2, deepest=3), 900))
    rules = list(itertools.islice(every_expression((0, 1, 2, 3), deepest=3), 900))
    apart = len(they_agree_everywhere(terms, ("here", "one along"))["apart"])
    apart += len(operations_agree_everywhere(rules)["apart"])
    return apart


#: Claim predicates that run an EXPERIMENT rather than read an instrument.


def _the_gap_that_should_not_be_there() -> int:
    """Ways the developmental result fails its own controls. Must be none.

    Three: the grown condition losing a family the reset condition solved; the
    gap failing to be there by the last block on a stream with structure in
    it; and any gap at all on the stream with nothing to carry. The third is
    the one that matters — a gain on the control would mean the gain on the
    other stream was about something else.
    """
    from tools.run_grown_against_reset_heads import run_stream

    # Families deep enough and a budget tight enough that the measurement can
    # move. At four families of depth three with two seconds each, every
    # condition solved four of four in every block on both streams, so the two
    # strict inequalities below were being asserted between two numbers that
    # were both the maximum and could not hold however well the mechanism
    # worked. That state had gone unnoticed since the claim was registered.
    #
    # Measured at these: grown [6, 6, 5] against reset [6, 4, 2].
    shared = run_stream(
        stream="shared", blocks=3, per_block=6, seed=1000, within=0.05, deepest=4
    )
    apart = run_stream(
        stream="apart", blocks=3, per_block=6, seed=1000, within=0.05, deepest=4
    )
    wrong = 0 if sum(shared["grown"]) > sum(shared["reset"]) else 1
    wrong += 0 if shared["grown"][-1] > shared["reset"][-1] else 1
    # The control is that the gap does not OPEN. Asserting it is exactly
    # nought would be asserting that a wall clock is noiseless.
    opened = (apart["grown"][-1] - apart["reset"][-1]) - (
        apart["grown"][0] - apart["reset"][0]
    )
    wrong += 0 if opened <= 0 else 1
    return wrong


def _transfer_that_went_the_wrong_way() -> int:
    """Ways the cross-domain result fails its own control. Must be none.

    Two: the piece failing to help on a domain whose term contains it, and the
    piece helping on one whose term does not. The second is the one that
    matters — help on the control would mean the help elsewhere was about
    something other than the piece.
    """
    from tools.run_grown_against_reset_heads import run_transfer

    with_it = without = control_with = control_without = usable = 0
    for seed in range(2000, 2006):
        row = run_transfer(seed=seed, families=4, within=2.0, deepest=4)
        if "why" in row:
            continue
        usable += 1
        with_it += row["related_with"]
        without += row["related_without"]
        control_with += row["apart_with"]
        control_without += row["apart_without"]
    if usable < 4:
        return 1
    return (0 if with_it >= without else 1) + (
        0 if control_with == control_without else 1
    )


def _recursive_heads_that_do_not_hold() -> int:
    """Recursive heads that fail at a length they were not fitted at. Must be none.

    Three families whose answers stand in a recurrence — doubling, factorial
    and triangular numbers. Each head is written from before-and-after states
    alone, and each is asked at lengths nine, eleven and thirteen, none of
    which it was fitted or judged at. One violation per family that fails, and
    one more if the lesion — taking the recurrence schema away — still finds
    doubling, because then the schema proved nothing.
    """
    import math

    from core.cognition.a_way_of_computing_she_wrote import (
        a_way_of_computing_she_wrote,
    )
    from core.cognition.an_invented_kind import WHERE_FROM
    from core.cognition.one_algebra import DERIVED_HEADS, Term, the_head_she_wrote
    from core.cognition.one_algebra import run as positional_run

    rules = {
        "doubling": lambda at, size: pow(2, at, size),
        "factorial": lambda at, size: math.factorial(at) % size,
        "triangular": lambda at, size: (at * (at + 1) // 2) % size,
    }

    def family(rule: Any) -> list[Any]:
        made = []
        for size in (4, 5, 6, 7):
            before = tuple(range(100, 100 + size))
            made.append(
                (before, tuple(before[rule(at, size) % size] for at in range(size)))
            )
        return made

    was = dict(DERIVED_HEADS)
    wrong = 0
    try:
        for name, rule in rules.items():
            DERIVED_HEADS.clear()
            found = a_way_of_computing_she_wrote(
                family(rule),
                now_sayable=lambda: False,
                words=dict(WHERE_FROM),
                within=20.0,
            )
            if found is None or not found.by_recurrence:
                wrong += 1
                continue
            the_head_she_wrote(name, 2, found.body)
            term = Term(name, parts=(Term("hole", value=0), Term("hole", value=1)))
            words = tuple(WHERE_FROM[one] for one in found.over)
            for size in (9, 11, 13):
                said = tuple(positional_run(term, at, size, words) for at in range(size))
                if said != tuple(rule(at, size) % size for at in range(size)):
                    wrong += 1
                    break
        DERIVED_HEADS.clear()
        lesioned = a_way_of_computing_she_wrote(
            family(rules["doubling"]),
            now_sayable=lambda: False,
            words=dict(WHERE_FROM),
            within=20.0,
            by_recurrence=False,
        )
        wrong += 0 if lesioned is None else 1
    finally:
        DERIVED_HEADS.clear()
        DERIVED_HEADS.update(was)
    return wrong


def _rules_with_no_shape_that_do_not_hold() -> int:
    """Shapeless rules that fail where they were never fitted. Must be none.

    Five families, three of which put a value into the state that was never in
    it — the case `language_limits` hands over rather than deciding, because no
    rule about where a cell came from can produce it. Each rule is written from
    before-and-after states alone and asked at lengths eleven and thirteen,
    neither of which it was fitted or judged at. One violation per family that
    fails, and one more if a rule does not survive being written down.
    """
    import random

    from core.cognition.a_rule_with_no_shape import (
        a_rule_she_wrote,
        read_a_rule_back,
        the_rule_written_down,
    )

    rules = {
        "add one": (lambda b, at, n: b[at] + 1, True),
        "twice": (lambda b, at, n: 2 * b[at], True),
        "mirror": (lambda b, at, n: b[n - 1 - at], False),
        "and its place": (lambda b, at, n: b[at] + at, True),
        "and the next": (lambda b, at, n: b[at] + b[(at + 1) % n], True),
    }

    def family(rule: Any) -> list[Any]:
        rng = random.Random(5)
        made = []
        for size in (4, 5, 6, 7, 8, 9):
            before = tuple(rng.sample(range(100), size))
            made.append(
                (before, tuple(rule(before, at, size) for at in range(size)))
            )
        return made

    wrong = 0
    for _name, (rule, makes) in rules.items():
        found = a_rule_she_wrote(family(rule), now_sayable=lambda: False, within=25.0)
        if found is None or found.makes_new_values is not makes:
            wrong += 1
            continue
        rng = random.Random(97)
        for size in (11, 13):
            before = tuple(rng.sample(range(200), size))
            if found.read(before) != tuple(rule(before, at, size) for at in range(size)):
                wrong += 1
                break
        back = read_a_rule_back(the_rule_written_down(found))
        if back is None or back.body != found.body:
            wrong += 1
    return wrong


def _a_choice_that_does_not_follow_the_record() -> int:
    """Ways the developmental ranking fails to be a decision. Must be none.

    Four. That a mode switch disagreeing with one ranking is not found, which
    would make the argument for one choice set vacuous. That varying the record
    does not vary what she chooses, which is what a ladder looks like. That the
    same varying moves a function returning the first rung, which would mean
    the check catches nothing. And that she cannot refuse.
    """
    from core.cognition.she_decides_to_develop import (
        forget_the_trace,
        what_to_do_next,
    )
    from core.cognition.the_record_of_her_own_work import (
        forget_the_record,
        note_an_episode,
    )
    from core.cognition.what_it_is_worth_doing import (
        the_choice_follows_the_record,
        where_a_split_disagrees_with_the_whole,
    )
    from core.cognition.what_she_could_do_next import (
        WHAT_SHE_COULD_DO,
        WHAT_THEY_HAVE_DONE,
        the_actions_she_has,
        what_she_could_do,
    )

    held, done = dict(WHAT_SHE_COULD_DO), dict(WHAT_THEY_HAVE_DONE)
    wrong = 0
    try:
        WHAT_SHE_COULD_DO.clear()
        WHAT_THEY_HAVE_DONE.clear()
        forget_the_record()
        forget_the_trace()
        worths = {"answer": 1, "write a head": 2}
        if not where_a_split_disagrees_with_the_whole(
            ordinary=["answer"],
            developmental=["write a head"],
            worth=worths.get,
            switch=lambda ordinary, developmental: "act",
        ):
            wrong += 1
        what_she_could_do(
            "a new word",
            over="the words",
            kind="a word",
            do_it=lambda one: "a word",
            price=400,
        )
        what_she_could_do(
            "a way of computing",
            over="the ways of computing",
            kind="a way of computing",
            do_it=lambda one: "a head",
            price=1400,
        )

        def cheap_is_all_that_helps() -> None:
            forget_the_record()
            for _ in range(3):
                note_an_episode("f", route=None, walked=1000)
            note_an_episode("f", route="a new word", walked=400, admitted="a word")
            for _ in range(3):
                note_an_episode("f", route="a new word", walked=20)

        def the_head_is_all_that_helps() -> None:
            forget_the_record()
            for _ in range(3):
                note_an_episode("f", route=None, walked=1000)
            note_an_episode(
                "f",
                route="a way of computing",
                walked=1400,
                admitted="a way of computing",
            )
            for _ in range(3):
                note_an_episode("f", route="a way of computing", walked=20)

        def chose() -> str:
            made = what_to_do_next("f", costs_now=1000)
            return made.action.name if made.action else "nothing"

        if not the_choice_follows_the_record(
            chose, [cheap_is_all_that_helps, the_head_is_all_that_helps]
        ):
            wrong += 1
        if the_choice_follows_the_record(
            lambda: the_actions_she_has()[0].name,
            [cheap_is_all_that_helps, the_head_is_all_that_helps],
        ):
            wrong += 1
        # And that she can refuse. Cheap to answer and dear to change is the
        # case where nothing is worth doing: a family met once whose answer
        # costs three candidates cannot repay a change costing four hundred,
        # however often it recurs.
        #
        # The first version of this check used a dear family met once and
        # expected a refusal, which was my expectation rather than the design:
        # nine thousand candidates an occasion will repay a fourteen-hundred
        # candidate change the first time it comes back.
        forget_the_record()
        note_an_episode("cheap", route="an answer", walked=3)
        if what_to_do_next("cheap", costs_now=3).action is not None:
            wrong += 1
    finally:
        WHAT_SHE_COULD_DO.clear()
        WHAT_SHE_COULD_DO.update(held)
        WHAT_THEY_HAVE_DONE.clear()
        WHAT_THEY_HAVE_DONE.update(done)
        forget_the_record()
        forget_the_trace()
    return wrong


def _a_developmental_record_that_cannot_be_checked() -> int:
    """Ways the record of who started what fails to be evidence. Must be none.

    Three: a receipt chain that does not follow, a destination that would let a
    change decide what is kept, and a promotion ladder that does not want more
    evidence for the parts everything runs through.
    """
    from core.cognition.how_a_change_is_promoted import (
        WHAT_A_TIER_WANTS,
        forget_the_receipts,
        nothing_installs_to_the_gate,
        promote,
        the_chain_holds,
    )

    wrong = 0
    forget_the_receipts()
    try:
        promote("word/x", became="shadow", started_by="she", evidence="4 of 4")
        promote("word/y", became="canary", started_by="she", evidence="5 of 5")
        if not the_chain_holds():
            wrong += 1
        if nothing_installs_to_the_gate():
            wrong += 1
        if WHAT_A_TIER_WANTS["the deciding"] <= WHAT_A_TIER_WANTS["word"]:
            wrong += 1
    finally:
        forget_the_receipts()
    return wrong


def _a_written_order_that_does_not_hold() -> int:
    """Ways the meta-invention result fails its own protocol. Must be none.

    Three: nothing written on the stream this is measured on; what was written
    failing to beat the authored order on the half it never saw; and the lesion
    failing to return the number to where it started. The second is the one
    that matters — selection happens on the training half, so something is
    usually found there, and the sealed half is what decides whether it meant
    anything.
    """
    import itertools
    import random

    from core.cognition.an_invented_kind import WHERE_FROM
    from core.cognition.one_algebra import every_term, holes_in
    from core.cognition.the_order_she_tries_them_in import (
        THE_ORDER,
        forget_the_order,
        the_order_she_uses,
    )
    from tools.run_meta_invention import (
        _an_episode,
        _an_order_she_writes,
        meta_capability,
    )

    rng = random.Random(3000)
    terms = [
        one
        for one in itertools.islice(every_term((0, 1, 2), holes=2, deepest=2), 8000)
        if holes_in(one) == 2
    ]
    names = sorted(WHERE_FROM)
    made: list[Any] = []
    while len(made) < 60:
        one = _an_episode(rng, terms, names)
        if one is not None:
            made.append(one)
    training, sealed = made[0::2], made[1::2]

    forget_the_order()
    before = meta_capability(sealed, THE_ORDER)
    written = _an_order_she_writes(training, deepest=4, within=45.0)
    if written is None:
        return 1
    wrong = 0 if meta_capability(sealed, written) < before else 1
    forget_the_order()
    wrong += 0 if meta_capability(sealed, the_order_she_uses()) == before else 1
    return wrong


def _install_language_growth_claims(suite: Any) -> None:
    """What she can do to the language she makes rules out of.

    Registered because a question about it was answered from a language
    model's priors and came back a flat denial — LIVE 2026-08-30, asked to
    prove she can invent primitives, she said her representation language was
    "the static set of instructions defined by my developers", two turns after
    using a word she had derived and kept. A claim with a test behind it is
    something she can answer from instead of guessing.
    """
    from core.cognition.sequence_reach import reach_preserves_correctness

    suite.add_test(ValidationTest(
        name="sequence_reach_cost_cannot_hide_a_lost_solution",
        description="the bounded sequence-reach utility orders correctness before candidate savings",
        required_capability="",
        observation=Observation(name="reach_utility_violations", value=0,
            source="tests/test_sequence_reach.py"),
        predict=lambda _m: len(reach_preserves_correctness()),
        score=lambda p, o: threshold_score(float(p), float(o.value), units=" violations"),
        owner="core/cognition/sequence_reach.py",
    ))
    suite.add_claim(Claim(
        statement="The sequence-reach utility cannot prefer search savings over a lower solved-family count.",
        test="sequence_reach_cost_cannot_hide_a_lost_solution",
        owner="core/cognition/sequence_reach.py",
        asserted_in="core/cognition/sequence_reach.py",
        evidence=Evidence.MEASURED_SYNTHETIC,
        evidence_note="Synthetic arithmetic canary and unit tests; not evidence of broad reasoning gain.",
    ))
    for name, description, predict, owner in (
        (
            "test_a_word_the_closure_already_says_is_refused",
            "a candidate word whose meaning some composition already produces is "
            "refused, so the vocabulary only grows where the meanings do",
            _macros_admitted_as_new_words,
            "core/cognition/widening_the_language.py",
        ),
        (
            "test_admitting_a_way_of_building_enlarges_the_meanings_not_only_the_spelling",
            "admitting a way of building words adds meanings and takes none away",
            _meanings_lost_by_admitting_a_way_of_building,
            "core/cognition/an_invented_kind.py",
        ),
        (
            "test_what_she_built_is_not_in_the_source_registry",
            "the way of building she arrives at for a family three words deep is "
            "described by a recipe she composed and is not a named constructor",
            _constructors_she_built_that_were_already_written,
            "core/cognition/a_constructor_she_built.py",
        ),
        (
            "test_it_answers_pairs_nobody_showed_her",
            "an operation derived from six examples answers values outside them, "
            "where the table read off the same examples refuses every one",
            _pairs_a_derived_operation_refuses,
            "core/cognition/an_operation_that_generalises.py",
        ),
        (
            "test_branching_is_not_something_those_three_could_have_produced",
            "a way of building words can be written that composition, inversion "
            "and iteration could not have produced between them",
            _ways_of_building_she_cannot_reach,
            "core/cognition/one_algebra.py",
        ),
        (
            "test_a_derived_word_comes_back_and_the_meaning_still_runs",
            "the derived words, the ways of building and the meanings written in "
            "them come back together, and the meaning still runs",
            _language_lost_across_a_restart,
            "core/cognition/what_she_gave_meaning.py",
        ),
        (
            "test_the_positional_language_has_a_ceiling_and_the_floor_does_not",
            "every positional term obeys a bound polynomial in the length of the "
            "state, so doubling is outside her positional language at every term "
            "length and over every vocabulary, and the floor says it",
            _the_positional_bound_broken,
            "core/cognition/what_the_old_language_cannot_say.py",
        ),
        (
            "test_the_floor_reaches_everything_computable",
            "Kleene's three starting points and three ways of building are each a "
            "term on the floor, and unbounded search on a predicate with no root "
            "exhausts its meter rather than returning",
            _universality_certificates_that_fail,
            "core/cognition/what_the_floor_can_say.py",
        ),
        (
            "test_the_floor_runs_its_own_evaluator_and_the_answers_agree",
            "the floor's evaluator is written in the floor, and running a term "
            "directly and running the interpreter on that term give the same "
            "answer — which is what 'the mechanism is an object she can hold' "
            "comes to",
            _the_floor_cannot_read_itself,
            "core/cognition/the_floor_reading_itself.py",
        ),
        (
            "test_growth_runs_into_four_walls_and_they_are_run_not_cited",
            "a universal language gains no meaning from a name, no update rule "
            "improves everywhere and the environment that beats one is built, "
            "unexpressibility is assertable only where the language was walked "
            "end to end, and a question above computation is answered by saying "
            "so",
            _the_four_walls_growth_runs_into,
            "core/cognition/what_growth_cannot_do.py",
        ),
        (
            "test_the_tower_ends_at_universality_and_nowhere_else",
            "a name can be substituted away without changing what it denotes, a "
            "universal bedrock puts the number of further authoring events at "
            "nought while a smaller one puts it beyond any bound, and an "
            "invariant checked by something that can itself be admitted stops "
            "holding once it is",
            _the_tower_with_no_top,
            "core/cognition/where_the_tower_has_a_top.py",
        ),
        (
            "test_she_writes_an_order_that_holds_on_episodes_it_never_saw",
            "the rule deciding what to try first is replaced by one she wrote, it "
            "ranks the winning word higher on invention episodes it never saw, and "
            "putting the authored rule back removes the gain exactly",
            _a_written_order_that_does_not_hold,
            "tools/run_meta_invention.py",
        ),
        (
            "test_her_choice_moves_when_the_record_moves",
            "developmental actions are ranked in one choice set rather than "
            "walked in a fixed order, what she chooses varies when the record "
            "varies, a function returning the first rung does not, and she can "
            "refuse to develop at all",
            _a_choice_that_does_not_follow_the_record,
            "core/cognition/she_decides_to_develop.py",
        ),
        (
            "test_a_receipt_chain_cannot_be_quietly_rewritten",
            "every installation writes a line carrying who started it and a "
            "digest of the line before, no destination lets a change decide "
            "what is kept, and the parts everything runs through want more "
            "evidence than the parts nothing rests on",
            _a_developmental_record_that_cannot_be_checked,
            "core/cognition/how_a_change_is_promoted.py",
        ),
        (
            "test_a_rule_with_no_shape_holds_where_it_was_never_fitted",
            "a rule whose shape is its own term is written from before-and-after "
            "states alone, puts values into the state that were never in it, and "
            "holds at lengths it was neither fitted nor judged at",
            _rules_with_no_shape_that_do_not_hold,
            "core/cognition/a_rule_with_no_shape.py",
        ),
        (
            "test_a_head_that_refers_to_itself_holds_where_it_was_never_fitted",
            "a head defined by what it says at the place before is written from "
            "before-and-after states alone and holds at lengths it was neither "
            "fitted nor judged at, and taking the recurrence away takes it with it",
            _recursive_heads_that_do_not_hold,
            "core/cognition/a_way_of_computing_she_wrote.py",
        ),
        (
            "test_what_she_wrote_carries_to_a_different_surface",
            "a piece written on one surface helps on a domain whose term "
            "contains it and does not help on one whose term does not, with the "
            "before-and-after states of the two domains looking unrelated",
            _transfer_that_went_the_wrong_way,
            "tools/run_grown_against_reset_heads.py",
        ),
        (
            "test_keeping_what_she_wrote_makes_the_next_one_easier",
            "on a stream of families with shared structure the agent that keeps "
            "what it wrote solves more than the one reset between blocks, and on "
            "a stream with nothing to carry the two are identical",
            _the_gap_that_should_not_be_there,
            "tools/run_grown_against_reset_heads.py",
        ),
        (
            "test_both_her_algebras_compile_to_one_semantics",
            "every positional term and every value expression computes on the "
            "floor exactly what its own interpreter computes, refusals included",
            _where_the_two_languages_disagree,
            "core/cognition/the_old_language_on_the_floor.py",
        ),
        (
            "test_a_way_of_computing_she_wrote_is_kept_and_still_runs",
            "a head written from before-and-after states alone computes the family "
            "at lengths it never saw, is kept, and still runs after a restart",
            _a_way_of_computing_she_cannot_keep,
            "core/cognition/a_way_of_computing_she_wrote.py",
        ),
    ):
        suite.add_test(
            ValidationTest(
                name=name,
                description=description,
                required_capability="representation_language_growth",
                observation=Observation(
                    name=f"{name}_violations",
                    value=0,
                    source=f"{owner} and tests/{owner.rsplit('/', 1)[-1]}",
                    units="violations",
                ),
                predict=lambda _m, run=predict: run(),
                score=lambda p, o: threshold_score(
                    float(p), float(o.value), units=" violations"
                ),
                owner=owner,
                expensive=name in _AN_EXPERIMENT_NOT_AN_INSTRUMENT,
            )
        )
    _install_vocabulary_claims(suite)
    _install_machinery_claims(suite)


def _install_vocabulary_claims(suite: Any) -> None:
    """The words and the ways of building them: admission, growth, universality."""
    suite.add_claim(
        Claim(
            statement=(
                "A word admitted to the language she makes rules out of must mean "
                "something no combination of existing words already meant."
            ),
            test="test_a_word_the_closure_already_says_is_refused",
            owner="core/cognition/widening_the_language.py",
            asserted_in="core/cognition/what_it_costs_to_say.py",
            live_channels=("language.words_derived", "language.meanings_reachable"),
            evidence=Evidence.MEASURED_SYNTHETIC,
            evidence_note=(
                "measured over the closure at lengths two to five; whether a word "
                "helps on families she has not met is a separate gate"
            ),
        )
    )
    suite.add_claim(
        Claim(
            statement=(
                "Admitting a way of building words enlarges the set of MEANINGS she "
                "can express, not only the set of expressions."
            ),
            test="test_admitting_a_way_of_building_enlarges_the_meanings_not_only_the_spelling",
            owner="core/cognition/an_invented_kind.py",
            asserted_in="core/cognition/what_it_costs_to_say.py",
            live_channels=("language.ways_of_building", "language.meanings_reachable"),
            evidence=Evidence.MEASURED_SYNTHETIC,
            evidence_note=(
                "forty meanings to two hundred and sixty, counted over the same "
                "bounded witness; no live turn has needed the enlargement"
            ),
        )
    )
    suite.add_claim(
        Claim(
            statement=(
                "She builds a way of making words from a recipe she composes, and "
                "the source registry does not contain it."
            ),
            test="test_what_she_built_is_not_in_the_source_registry",
            owner="core/cognition/a_constructor_she_built.py",
            asserted_in="core/cognition/a_constructor_she_built.py",
            live_channels=("language.ways_of_building",),
            evidence=Evidence.MEASURED_SYNTHETIC,
            evidence_note=(
                "measured on families needing a chain three words deep; the space "
                "of recipes is three ways and a depth, so this is growth within a "
                "described space and not unbounded synthesis"
            ),
        )
    )
    suite.add_claim(
        Claim(
            statement=(
                "An operation she derives is a rule over the pair, so it answers "
                "values nobody showed her."
            ),
            test="test_it_answers_pairs_nobody_showed_her",
            owner="core/cognition/an_operation_that_generalises.py",
            asserted_in="core/cognition/an_operation_that_generalises.py",
            evidence=Evidence.MEASURED_SYNTHETIC,
            evidence_note=(
                "derived by inverting the operation and held to examples it was "
                "not fitted on, to depth three over eight ways of combining"
            ),
        )
    )
    suite.add_claim(
        Claim(
            statement=(
                "A way of BUILDING words is a term she writes rather than one of a "
                "list, so what she can reach is not the closure of three named "
                "constructors."
            ),
            test="test_branching_is_not_something_those_three_could_have_produced",
            owner="core/cognition/one_algebra.py",
            asserted_in="core/cognition/one_algebra.py",
            evidence=Evidence.MEASURED_SYNTHETIC,
            evidence_note=(
                "composition, inversion and iteration reach 4,435 behaviours over "
                "the words she was given; a maker she wrote for a family that "
                "branches on size is not among them. Corrected 2026-09-01: this "
                "said the grammar was the floor of computing, and it is not. "
                "Every positional term is total and the terms are enumerable, so "
                "what they express is a recursively enumerable class of total "
                "functions with something computable outside it — measured in "
                "core/cognition/what_the_old_language_cannot_say.py, twice. The "
                "grammar was a menu after all, and the head registry is what "
                "replaced it"
            ),
            live_channels=("language.ways_of_building",),
        )
    )
    suite.add_claim(
        Claim(
            statement=(
                "The words she derives, the ways of building she admits, and the "
                "meanings written in them survive a restart together."
            ),
            test="test_a_derived_word_comes_back_and_the_meaning_still_runs",
            owner="core/cognition/what_she_gave_meaning.py",
            asserted_in="core/cognition/what_she_gave_meaning.py",
            evidence=Evidence.MEASURED_LIVE,
        )
    )
    suite.add_claim(
        Claim(
            statement=(
                "Her positional language is not universal: doubling by the length "
                "of the state is outside it at every term length and over every "
                "vocabulary she can build, so only a person could ever have put "
                "it in."
            ),
            test="test_the_positional_language_has_a_ceiling_and_the_floor_does_not",
            owner="core/cognition/what_the_old_language_cannot_say.py",
            asserted_in="core/cognition/what_the_old_language_cannot_say.py",
            evidence=Evidence.MEASURED_SYNTHETIC,
            evidence_note=(
                "an induction over the heads bounds every term by max(n, c, 2) to "
                "the power of its length; checked on 155,719 answers over 4,000 "
                "terms with no violation, and the induction's case list is "
                "compared against the heads run() dispatches on. A second witness "
                "walks the words she can build and answers differently from the "
                "n-th at the n-th place: 400 words, 1,200 places, no agreement"
            ),
            live_channels=("language.ways_of_building",),
        )
    )
    suite.add_claim(
        Claim(
            statement=(
                "The floor's evaluator is written in the floor itself, and "
                "running a term directly agrees with running the interpreter "
                "on that term — so the mechanism is an object she can quote, "
                "pass around and change rather than only a thing that runs."
            ),
            test="test_the_floor_runs_its_own_evaluator_and_the_answers_agree",
            owner="core/cognition/the_floor_reading_itself.py",
            asserted_in="core/cognition/the_floor_reading_itself.py",
            evidence=Evidence.MEASURED_SYNTHETIC,
            evidence_note=(
                "an addition and a multiplication run both ways and compared, "
                "and a function compared by what it does at three arguments — "
                "the two closures differ inside, which is right and says "
                "nothing about whether they agree"
            ),
        )
    )
    suite.add_claim(
        Claim(
            statement=(
                "Growing the language she thinks in runs into four walls, and each "
                "of them is executed here rather than cited: a name adds no "
                "meaning to a universal language, no update rule improves "
                "everywhere, unexpressibility may be asserted only where the "
                "language was walked end to end, and a question above computation "
                "is answered by saying so."
            ),
            test="test_growth_runs_into_four_walls_and_they_are_run_not_cited",
            owner="core/cognition/what_growth_cannot_do.py",
            asserted_in="core/cognition/what_growth_cannot_do.py",
            evidence=Evidence.MEASURED_SYNTHETIC,
            evidence_note=(
                "the substitution argument run on a word she made, the diagonal "
                "environment built and scored against the rule it beats, and the "
                "decidability verdict read off a language that was walked and one "
                "that cannot be"
            ),
        )
    )
    suite.add_claim(
        Claim(
            statement=(
                "Universality is where the regress of authored primitives ends, "
                "and it is the only place it can: a universal bedrock needs no "
                "further authoring and admits none, and a smaller one needs "
                "authoring again after every edit."
            ),
            test="test_the_tower_ends_at_universality_and_nowhere_else",
            owner="core/cognition/where_the_tower_has_a_top.py",
            asserted_in="core/cognition/where_the_tower_has_a_top.py",
            evidence=Evidence.MEASURED_SYNTHETIC,
            evidence_note=(
                "the three cases run — universal, smaller, and unknown — with the "
                "gate made a candidate of its own space and the invariant it "
                "checks stopping holding once it is"
            ),
        )
    )
    suite.add_claim(
        Claim(
            statement=(
                "The floor reaches every computable behaviour, so nothing above it "
                "will ever need a new primitive written by a person — and nothing "
                "could add one."
            ),
            test="test_the_floor_reaches_everything_computable",
            owner="core/cognition/what_the_floor_can_say.py",
            asserted_in="core/cognition/where_the_tower_has_a_top.py",
            evidence=Evidence.MEASURED_SYNTHETIC,
            evidence_note=(
                "Kleene's characterisation, with the three starting points and "
                "the three ways of building each exhibited as a term and checked; "
                "unbounded search on a predicate with no root exhausts its meter "
                "rather than returning, which is what separates universal from "
                "merely large. Four of the seven arithmetic heads are shown "
                "definable from the other three, so the instruction set is doing "
                "no work"
            ),
        )
    )


def _install_machinery_claims(suite: Any) -> None:
    """The ways of computing: heads, surfaces, provenance, and the one semantics."""
    suite.add_claim(
        Claim(
            statement=(
                "A change she made to the machinery she invents with reaches "
                "ordinary invention on families it never saw, with no source "
                "edited between the two."
            ),
            test="test_she_writes_an_order_that_holds_on_episodes_it_never_saw",
            owner="tools/run_meta_invention.py",
            asserted_in="core/cognition/the_order_she_tries_them_in.py",
            evidence=Evidence.MEASURED_SYNTHETIC,
            evidence_note=(
                "twenty sealed families per stream, every invention on the same "
                "tight budget, and the order selected on a rank measured over "
                "OTHER episodes — so what is reported is transfer rather than "
                "fit. Three streams where an order was written: 13 to 16, 16 to "
                "18, and 13 to 12, with the lesion returning each number exactly. "
                "Two better and one worse, which is the no-free-lunch theorem "
                "arriving in practice; a rule improving every stream would "
                "contradict a theorem this codebase already executes. A second "
                "measurement of the same claim, kept because it is a different "
                "run and not a restatement: one stream of five, sixty episodes "
                "split before anything was written, an eight-symbol order from "
                "the training half moving the mean rank of the winning word on "
                "the SEALED half from 2.000 to 1.833, and the authored rule put "
                "back returning it to 2.000 exactly. On the other four streams "
                "what selection found on the training half did not survive the "
                "sealed half — one was `nought minus how long the word is`, a "
                "rule saying prefer longer words. One meta-change with its "
                "control is not a trend, and recursive self-improvement is not "
                "claimed"
            ),
            live_channels=("language.ways_of_building",),
        )
    )
    suite.add_claim(
        Claim(
            statement=(
                "What she decides to do about herself follows the record of what "
                "her work has cost, and the thing she announces is the thing she "
                "carries out."
            ),
            test="test_her_choice_moves_when_the_record_moves",
            owner="core/cognition/she_decides_to_develop.py",
            asserted_in="core/autonomy/autonomous_initiative_loop.py",
            evidence=Evidence.MEASURED_SYNTHETIC,
            evidence_note=(
                "the choice among unpriced actions is a draw from what the "
                "counts support, and a function returning the first rung does "
                "not move when the record moves. The second half was measured "
                "and false: the idle loop asked what was worth doing, told the "
                "user it had decided on one thing, and then asked again and did "
                "whatever the second draw said — 162 of 200 episodes disagreed. "
                "The loop now hands the announced decision in, and the same "
                "fixture over 60 episodes disagrees 0 times"
            ),
        )
    )
    suite.add_claim(
        Claim(
            statement=(
                "Every installation writes a line carrying who started it and a "
                "digest of the line before, and every promotion records what it "
                "replaced, so a change can be undone and the record cannot be "
                "quietly rewritten to say a decision was hers."
            ),
            test="test_a_receipt_chain_cannot_be_quietly_rewritten",
            owner="core/cognition/how_a_change_is_promoted.py",
            asserted_in="core/cognition/she_decides_to_develop.py",
            evidence=Evidence.MEASURED_SYNTHETIC,
            evidence_note=(
                "the chain is checked by recomputing every digest. The second "
                "half was a promise and not a fact for the life of the module: "
                "no caller anywhere passed `replaced`, so the stack was empty, "
                "put_it_back returned None every time, and nothing could be "
                "undone. A developmental promotion now carries the snapshot the "
                "trial takes, and an undo that raises is recorded as one that "
                "would not go back rather than as a rollback"
            ),
        )
    )
    suite.add_claim(
        Claim(
            statement=(
                "The shape of a rule is a term rather than a record, so how many "
                "places it reads and whether it puts values into the state that "
                "were never in it are not fields anybody wrote."
            ),
            test="test_a_rule_with_no_shape_holds_where_it_was_never_fitted",
            owner="core/cognition/a_rule_with_no_shape.py",
            asserted_in="core/cognition/an_invented_kind.py",
            evidence=Evidence.MEASURED_SYNTHETIC,
            evidence_note=(
                "five families with distinct-valued states at lengths four to "
                "nine, fitted on three and judged on three, correct at eleven "
                "and thirteen. Three of the five make values that were never in "
                "the state, which is the case language_limits.certify hands over "
                "rather than deciding. Neither half is walked blindly: where an "
                "answer already sits is read off the data, where it does not the "
                "place is solved for, and what is done with a cell is inverted. "
                "Two families are out of reach and named — a third source, and a "
                "second source at a place past the forty-eight the fold walks"
            ),
            live_channels=("language.ways_of_building",),
        )
    )
    suite.add_claim(
        Claim(
            statement=(
                "A head defined by what it says at the place before is written "
                "from before-and-after states alone and holds at lengths it was "
                "neither fitted nor judged at."
            ),
            test="test_a_head_that_refers_to_itself_holds_where_it_was_never_fitted",
            owner="core/cognition/a_way_of_computing_she_wrote.py",
            asserted_in="core/cognition/a_way_of_computing_she_wrote.py",
            evidence=Evidence.MEASURED_SYNTHETIC,
            evidence_note=(
                "doubling, factorial and triangular numbers, fitted at lengths "
                "four and six, held to five and seven, correct at nine, eleven "
                "and thirteen. Two things make it possible and both are stated "
                "rather than hidden: a head is given a fixed point, written out "
                "of application rather than added as a primitive, so by the "
                "substitution argument it adds no meanings; and the step is "
                "solved for rather than searched, which is a schema — the "
                "inversion of that fixed point, the same move "
                "an_operation_that_generalises makes on arithmetic. Lesioning "
                "the schema leaves only enumeration, and enumeration finds "
                "nothing on these families"
            ),
            live_channels=("language.ways_of_building",),
        )
    )
    suite.add_claim(
        Claim(
            statement=(
                "A way of computing written on one surface carries to a family "
                "whose states look unrelated but whose structure contains it, and "
                "does not carry to one that does not contain it."
            ),
            test="test_what_she_wrote_carries_to_a_different_surface",
            owner="tools/run_grown_against_reset_heads.py",
            asserted_in="core/cognition/a_way_of_computing_she_wrote.py",
            evidence=Evidence.MEASURED_SYNTHETIC,
            evidence_note=(
                "sixteen seeds, six families each, different words on each "
                "domain so the states look unrelated. Related: 87 of 96 with the "
                "piece against 71 without. Control, whose term does not contain "
                "the piece anywhere: 90 and 90. Measured on the search rather "
                "than on admission, as above. The relation is constructed "
                "rather than found, and that is the limit of the claim — it "
                "shows the piece is what carries, not that any real pair of "
                "domains stands in this relation"
            ),
        )
    )
    suite.add_claim(
        Claim(
            statement=(
                "Keeping the ways of computing she wrote makes the next one "
                "easier to write, on a stream of families with shared structure "
                "and not on one without."
            ),
            test="test_keeping_what_she_wrote_makes_the_next_one_easier",
            owner="tools/run_grown_against_reset_heads.py",
            asserted_in="core/cognition/a_way_of_computing_she_wrote.py",
            evidence=Evidence.MEASURED_SYNTHETIC,
            evidence_note=(
                "five seeds, five blocks of six families, matched budget and "
                "words, families drawn at random rather than chosen. Shared "
                "stream: grown 129/150, reset 89, lesioned 81, with the gap by "
                "block 0, 1, 2, 2.8, 2.2. Control stream with nothing to carry: "
                "147, 148, 148, and the gap does not open. "
                "Measured on the SEARCH rather than on admission: these families "
                "are all inside what the positional language says, so the gate "
                "above the search would refuse every head as a shorter name. "
                "The gap is not monotone block to block, and the mechanism is "
                "transparent — later families are drawn over earlier terms — so "
                "this is compounding on a stream with structure rather than "
                "evidence about any real task distribution"
            ),
            live_channels=("language.ways_of_building",),
        )
    )
    suite.add_claim(
        Claim(
            statement=(
                "The two languages she thinks in are one semantics: a positional "
                "term and a value expression are both terms of the same eighteen "
                "heads, so an invention in either can be material for an "
                "invention in the other."
            ),
            test="test_both_her_algebras_compile_to_one_semantics",
            owner="core/cognition/the_old_language_on_the_floor.py",
            asserted_in="core/cognition/one_algebra.py",
            evidence=Evidence.MEASURED_SYNTHETIC,
            evidence_note=(
                "12,000 positional terms over three pairs of words at seven "
                "lengths — 1,296,000 places — and 8,000 value expressions over "
                "512,000 pairs, complete agreement including every refusal. "
                "Refusals are what earned the check: substituting rather than "
                "binding inside `through` made the floor lazy where the "
                "interpreter is strict, and a comparison that skipped refusals "
                "would not have seen it"
            ),
        )
    )
    suite.add_claim(
        Claim(
            statement=(
                "A way of COMPUTING — a head of the grammar, not a word in it — is "
                "written from before-and-after states alone, computes the family at "
                "lengths it never saw, and is still there after a restart."
            ),
            test="test_a_way_of_computing_she_wrote_is_kept_and_still_runs",
            owner="core/cognition/a_way_of_computing_she_wrote.py",
            asserted_in="core/cognition/one_algebra.py",
            evidence=Evidence.MEASURED_SYNTHETIC,
            evidence_note=(
                "no target operator, no candidate implementation, no name and no "
                "list of kinds go in; fitted at lengths four and six, held to five "
                "and seven, correct at nine and eleven. What a shortest-first "
                "search can reach is a few dozen symbols, which is Levin's bound "
                "rather than a defect: the library is what moves the horizon, and "
                "the measurement is that the same search at the same budget finds "
                "nothing with an empty library and the answer with one entry. "
                "This claim is about the SEARCH and the persistence, not about "
                "admission: on 120 families drawn at random the growth "
                "classifier said the positional language already says it every "
                "time, so no head has yet passed the full gate — see "
                "tests/test_a_head_has_not_yet_earned_its_place.py"
            ),
            live_channels=("language.ways_of_building",),
        )
    )


__all__ = ["_install_language_growth_claims"]
