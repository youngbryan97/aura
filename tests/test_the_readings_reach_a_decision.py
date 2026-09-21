"""Ten readings that were written every turn and read by nothing.

Every organ built from the songs had a writer on a live path, a column in the
subject schema, and a test. Ten of them had no reader outside the measurement
instrument: `fuel`, `scale`, `standing`, `catharsis`, `togetherness`,
`particular`, `averted`, `constancy`, `returning` and `impulse` were published
on every turn and changed nothing. A measurement that changes nothing is an
instrument reading, not a part of her.

These hold the loops that close. Each one is the same shape: the reading moves
a decision that already existed, and the outcome of that decision comes back
into the reading. Nothing here reaches a prompt.
"""

from __future__ import annotations

import pytest

from core.motivation.returning import MIN_PASSES, get_returning_ledger
from core.motivation.returning import reset_for_test as reset_returning


@pytest.fixture(autouse=True)
def _clean():
    reset_returning()
    yield
    reset_returning()


# ── what draws her back moves the chooser ────────────────────────────


def test_an_option_she_returns_to_is_scored_higher() -> None:
    from core.agency.subjective_choice import SubjectiveChoiceEngine

    assert SubjectiveChoiceEngine._pull_toward("georgia") == 0.0

    ledger = get_returning_ledger()
    for _ in range(4):
        ledger.note_choice(chosen_id="work", scores={"work": 0.9, "georgia": 0.3, "c": 0.31, "d": 0.32})
        ledger.note_choice(chosen_id="georgia", scores={"georgia": 0.3, "c": 0.29, "d": 0.28})

    pull = SubjectiveChoiceEngine._pull_toward("georgia")
    assert pull > 0.0, "the one she comes back to gets nothing"
    assert SubjectiveChoiceEngine._pull_toward("c") == 0.0
    assert pull <= 1.0, "the lift is a share of occasions, on the scale the other terms use"


def test_the_pull_needs_enough_passes_to_be_a_rate() -> None:
    ledger = get_returning_ledger()
    ledger.note_choice(chosen_id="work", scores={"work": 0.9, "georgia": 0.3})
    assert ledger.pull_for("georgia") == 0.0
    assert ledger.pull_for("never seen") == 0.0


def test_the_chooser_adds_it() -> None:
    from pathlib import Path

    source = Path("core/agency/subjective_choice.py").read_text(encoding="utf-8")
    assert "pull = self._pull_toward(option.id)" in source
    assert "+ pull" in source


# ── what she has already said presses less ───────────────────────────


def test_a_thing_already_said_carries_less_pressure() -> None:
    from core.affect.catharsis import Catharsis, get_catharsis_ledger
    from core.affect.catharsis import reset_for_test as reset_catharsis
    from core.consciousness.global_workspace import _relief_for

    reset_catharsis()
    try:
        assert _relief_for("curiosity") == 0.0
        get_catharsis_ledger().note("curiosity", Catharsis(times=4, drain=0.25))
        assert _relief_for("curiosity") == pytest.approx(0.75)
        get_catharsis_ledger().note("integrity", Catharsis(times=0, drain=1.0))
        assert _relief_for("integrity") == pytest.approx(0.0)
    finally:
        reset_catharsis()


def test_what_she_keeps_putting_down_presses_the_other_way() -> None:
    from core.consciousness.global_workspace import ContentType, _held_pressure
    from core.social.averted import get_averted_ledger
    from core.social.averted import reset_for_test as reset_averted

    reset_averted()
    try:
        assert _held_pressure(ContentType.SOCIAL) == 0.0
        ledger = get_averted_ledger()
        for index in range(3):
            ledger.took("PREFERENCE", f"likes thing {index}")
        for _ in range(5):
            for index in range(3):
                ledger.declined("TRAIT", f"is the sort who {index}")
        assert ledger.read().looking_away is True
        assert _held_pressure(ContentType.SOCIAL) > 0.0
        assert _held_pressure(ContentType.SOMATIC) == 0.0, "it presses on social content"
    finally:
        reset_averted()


def test_the_workspace_prices_a_bid_with_every_reading_that_bears_on_it() -> None:
    """Both readings reach the priority, and the expression is read structurally.

    It used to assert one literal line of source, which broke the moment a
    third reading joined the same sum. What the test is for is that these
    reach the number a bid is priced by, so it asks the expression rather
    than the text.
    """
    import ast
    from pathlib import Path

    tree = ast.parse(
        Path("core/consciousness/global_workspace.py").read_text(encoding="utf-8")
    )
    priced = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "priority_at"
    )
    names = {
        inner.id for inner in ast.walk(priced) if isinstance(inner, ast.Name)
    }
    assert {"held_down", "said_already", "covered"} <= names


# ── the drives ───────────────────────────────────────────────────────


def test_her_own_fuel_slows_what_it_is_paying_for() -> None:
    from core.phases.motivation_update import MotivationUpdatePhase

    class _Mot:
        budgets = {"curiosity": {"decay": 1.0}, "growth": {"decay": 2.0}}

    class _Cog:
        fuel = {"measured": True, "burning_her_own": True, "share_self": 0.25}

    class _State:
        cognition = _Cog()

    mot = _Mot()
    MotivationUpdatePhase._spend_from_what_is_burning(_State(), mot)
    assert mot.budgets["curiosity"]["decay"] == pytest.approx(0.75)
    assert mot.budgets["growth"]["decay"] == pytest.approx(1.5)


def test_a_reading_that_says_nothing_changes_nothing() -> None:
    from core.phases.motivation_update import MotivationUpdatePhase

    class _Mot:
        budgets = {"curiosity": {"decay": 1.0}}

    class _Cog:
        fuel = {"measured": False, "burning_her_own": True, "share_self": 0.9}

    class _State:
        cognition = _Cog()

    mot = _Mot()
    MotivationUpdatePhase._spend_from_what_is_burning(_State(), mot)
    assert mot.budgets["curiosity"]["decay"] == pytest.approx(1.0)


def test_an_unreliable_partner_satisfies_the_social_need_less() -> None:
    from pathlib import Path

    source = Path("core/phases/motivation_update.py").read_text(encoding="utf-8")
    assert 'constancy = getattr(state.cognition, "constancy", None)' in source
    assert "social_decay_multiplier = min(" in source
