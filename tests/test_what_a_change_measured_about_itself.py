"""An action that says it judges itself, and what it has to show for that."""
from __future__ import annotations

import pytest

from core.cognition.what_a_change_measured_about_itself import (
    ChangesNothing,
    WhatItMeasured,
    changes_nothing,
    claiming_without_showing,
    forget_everything,
    how_the_self_judged_stand,
    note_a_claim,
    the_evidence_in,
)
from core.cognition.what_she_could_do_next import (
    HOW_CHANGES_WERE_JUDGED,
    WHAT_SHE_COULD_DO,
    how_changes_were_judged,
    what_she_could_do,
)


@pytest.fixture(autouse=True)
def _clean():
    forget_everything()
    was = dict(HOW_CHANGES_WERE_JUDGED)
    for key in HOW_CHANGES_WERE_JUDGED:
        HOW_CHANGES_WERE_JUDGED[key] = 0
    yield
    HOW_CHANGES_WERE_JUDGED.update(was)
    forget_everything()


def _an_action(name: str, gives):
    WHAT_SHE_COULD_DO.pop(name, None)
    return what_she_could_do(
        name, over="the words", kind="k", do_it=lambda *a: gives, judges_itself=True
    )


def test_an_action_that_shows_its_working_is_judged_by_it() -> None:
    shown = WhatItMeasured(
        said="widened it",
        on=("f7", "f9"),
        before=0.40,
        after=0.62,
        why_it_counts="families it was not chosen for",
    )
    assert note_a_claim("a", shown) == "judged itself"
    assert claiming_without_showing() == ()


def test_an_action_that_shows_nothing_is_counted_as_unmeasured() -> None:
    """The opt-out was a boolean in a table and nothing checked the claim."""
    assert note_a_claim("b", "just a sentence") == "unmeasured"
    assert claiming_without_showing() == ("b",)


def test_evidence_with_no_families_is_not_evidence() -> None:
    empty = WhatItMeasured(said="did it", on=(), before=0, after=1, why_it_counts="")
    assert not empty.enough_to_be_believed
    assert note_a_claim("c", empty) == "unmeasured"


def test_an_action_that_installs_nothing_says_so_rather_than_judging_itself() -> None:
    """Three real claims hide behind one honest one once they share a flag."""
    assert note_a_claim("d", changes_nothing("asked a question")) == "changes nothing"
    assert claiming_without_showing() == ()
    said = how_the_self_judged_stand()
    assert said["counts"]["changes nothing"] == 1
    assert "judged itself" not in said["counts"]


def test_a_self_judged_change_that_did_not_pay_is_put_back() -> None:
    """An action that measured itself and lost must not be kept for returning
    an object."""
    regressed = WhatItMeasured(
        said="widened it", on=("f7",), before=0.6, after=0.4, why_it_counts="held out"
    )
    assert not regressed.paid
    assert not regressed  # the gate keeps what is truthy

    made = _an_action("regressed", regressed)
    assert made.do_it(None) is None
    assert HOW_CHANGES_WERE_JUDGED["did not pay"] == 1


def test_the_gate_counts_each_of_the_four_endings_apart() -> None:
    _an_action("shows working", WhatItMeasured(
        said="w", on=("f1",), before=0.1, after=0.9, why_it_counts="held out")
    ).do_it(None)
    _an_action("shows nothing", "a sentence").do_it(None)
    _an_action("asks", changes_nothing("asked")).do_it(None)
    _an_action("declines", None).do_it(None)

    counted = how_changes_were_judged()["counts"]
    assert counted["judged itself"] == 1
    assert counted["unmeasured"] == 1
    assert counted["changes nothing"] == 1
    assert counted["declined"] == 1


def test_the_report_names_who_is_claiming_without_showing() -> None:
    _an_action("shows nothing", "a sentence").do_it(None)
    said = how_changes_were_judged()
    assert said["claiming_without_showing"] == ["shows nothing"]
    assert said["kept_without_evidence"] == 1


def test_the_evidence_reader_only_reads_evidence() -> None:
    assert the_evidence_in("a sentence") is None
    assert the_evidence_in(changes_nothing("x")) is None
    shown = WhatItMeasured(said="s", on=("f",), before=0, after=1, why_it_counts="w")
    assert the_evidence_in(shown) is shown


def test_a_changes_nothing_carries_the_reason_it_installs_nothing() -> None:
    asked = changes_nothing("asked for an example", because="it installs nothing")
    assert isinstance(asked, ChangesNothing)
    assert str(asked) == "asked for an example"
    assert bool(asked)


def test_every_shipped_self_judging_action_shows_working_or_says_it_installs_nothing() -> None:
    """The four that take the opt-out, checked at the source rather than by
    running them: three of them decline in a bare process, so a test that ran
    them would pass while proving nothing."""
    import ast
    import pathlib

    source = pathlib.Path("core/cognition/what_she_does_about_herself.py").read_text()
    lines = source.splitlines(keepends=True)

    def body_of(name: str) -> str:
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.FunctionDef) and node.name == name:
                return "".join(lines[node.lineno - 1 : node.end_lineno])
        raise AssertionError(f"{name} is gone")

    for name in ("let_go", "one_name_for_both", "take_the_cause_that_pays"):
        shown = body_of(name)
        assert "_what_it_measured(" in shown or "WhatItMeasured(" in shown, name

    asking = body_of("ask_for_an_example")
    assert "changes_nothing(" in asking
    assert "WhatItMeasured(" not in asking, (
        "asking installs nothing; claiming to have measured it would be worse "
        "than claiming nothing"
    )


#: The three faculties that are about how she improves rather than about what
#: she knows. A change to one of these changes the machine that makes changes.
THE_META_FACULTIES = (
    "the order she tries them in",
    "the proposer",
    "what a change is worth",
)


def test_a_change_to_the_improvement_machinery_cannot_judge_itself() -> None:
    """The review's requirement, as an invariant rather than a habit.

    "Improvements to the improvement mechanism must be evaluated with the same
    rigor as ordinary improvements." An action that changes the order she
    tries things in, the proposer, or what a change is worth is an improvement
    to the machine that makes improvements, and letting one of those take the
    self-judging opt-out means the thing being changed is also the judge.

    It holds today. It is asserted so that it goes on holding.
    """
    from core.cognition.what_she_does_about_herself import (
        offer_what_she_can_do_about_what_she_is_made_of as offer,
    )

    offer()
    for module in (
        "core.cognition.she_improves_her_own_deciding",
        "core.cognition.an_action_she_writes_for_a_gap",
        "core.cognition.an_operator_she_invents",
        "core.cognition.does_improving_compound",
    ):
        __import__(module)
        for name in dir(__import__(module, fromlist=["x"])):
            if name.startswith("offer") or name.startswith("she_can"):
                found = getattr(__import__(module, fromlist=["x"]), name)
                if callable(found):
                    try:
                        found()
                    except Exception:  # noqa: BLE001 - registration is best effort
                        pass

    judging = sorted(
        name
        for name, action in WHAT_SHE_COULD_DO.items()
        if action.over in THE_META_FACULTIES and action.judges_itself
    )
    assert judging == [], (
        "these change how she improves and judge their own improvement: "
        + ", ".join(judging)
    )


def test_the_meta_faculties_are_places_a_term_can_actually_go() -> None:
    """A list of faculty names that drifts from the real one checks nothing."""
    from core.cognition.what_she_could_do_next import WHERE_A_TERM_CAN_GO

    for one in THE_META_FACULTIES:
        assert one in WHERE_A_TERM_CAN_GO, one
