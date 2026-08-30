"""A mind that reinvents the same property every morning has not learned it.

She can compose a property nobody wrote, prove it plays better and judge by it;
and she can induce the meaning of a kind of rule nobody wrote and have the
interpreter run it. Both lived in process memory, so both died when the process
did.

That is not a small omission. The point of being able to invent one was never
a single good measure — it was that what she works out is persistent, reusable,
composable and transferable, and three of those four fail if it does not
survive a restart.

What is kept is the RECIPE, never a pickled object: where a measure looks, what
it takes and how it combines; which two places a meaning reads from and what it
does with the pair. Both reconstruct exactly, both are readable by a person, and
neither can execute anything that was not already in the space she searches.
"""

from __future__ import annotations

import pytest

from core.agency.how_good_is_this import INVENTED, ON_TRIAL, forget, on_trial, promote
from core.agency.inventing_a_measure import Measure
from core.cognition.an_invented_kind import KINDS, admit, induce_from
from core.cognition.rule_ir import Node
from core.agency.what_she_invented import (
    forget_everything,
    keep,
    recall,
)
from core.cognition.what_she_gave_meaning import (
    forget_everything as forget_meanings,
    keep as keep_meanings,
    recall as recall_meanings,
)

A_MEASURE = Measure("neighbours", "the gap between them", "on average", True)
PAIRWISE = [((1, 5, 2, 9), (5, 5, 9, 9)), ((3, 1, 4, 1), (3, 3, 4, 4)),
            ((7, 2, 8, 6), (7, 7, 8, 8)), ((2, 6, 1, 3), (6, 6, 3, 3))]


@pytest.fixture(autouse=True)
def _a_clean_mind(tmp_path, monkeypatch):
    from core.agency import what_she_invented
    from core.cognition import what_she_gave_meaning

    monkeypatch.setattr(what_she_invented, "_KEPT_AT", tmp_path / "properties.json")
    monkeypatch.setattr(what_she_gave_meaning, "_KEPT_AT", tmp_path / "meanings.json")
    held, kinds, trials = dict(INVENTED), dict(KINDS), dict(ON_TRIAL)
    INVENTED.clear()
    KINDS.clear()
    ON_TRIAL.clear()
    yield
    INVENTED.clear()
    INVENTED.update(held)
    KINDS.clear()
    KINDS.update(kinds)
    ON_TRIAL.clear()
    ON_TRIAL.update(trials)


def _a_restart():
    """Everything she worked out, gone from memory but not from disk."""
    INVENTED.clear()
    KINDS.clear()
    ON_TRIAL.clear()


# ── a property she invented ──────────────────────────────────────────────

def test_a_property_she_invented_comes_back():
    promote(A_MEASURE, 0.4)
    assert keep() is True
    _a_restart()
    assert recall()["measures"] == 1
    assert A_MEASURE.name in INVENTED


def test_and_comes_back_worth_what_it_was_worth():
    from core.agency.how_good_is_this import AS_GOOD_A_GUESS_AS_ANY

    name = promote(A_MEASURE, 0.37)
    keep()
    _a_restart()
    recall()
    assert AS_GOOD_A_GUESS_AS_ANY[name] == pytest.approx(0.37)


def test_a_trial_interrupted_by_a_restart_carries_on():
    """A trial a reboot interrupted is not a trial that failed.

    Starting it again from nought would mean a property could never be judged
    at all on a machine that reboots.
    """
    name = on_trial(A_MEASURE, 0.4)
    ON_TRIAL[name].update({"before": 1.0, "from": 10.0, "to": 40.0, "seen": 12})
    keep()
    _a_restart()
    recall()
    assert ON_TRIAL[name]["seen"] == 12
    assert ON_TRIAL[name]["before"] == 1.0


# ── a meaning she induced ────────────────────────────────────────────────

def test_a_meaning_she_induced_comes_back():
    admit("pairwise", induce_from(PAIRWISE))
    assert keep_meanings() is True
    _a_restart()
    assert recall_meanings() == 1
    assert Node(kind="pairwise").apply((4, 9, 1, 2)) == (9, 9, 2, 2)


def test_a_written_node_carries_its_own_meaning():
    """So reading a node anywhere is enough to be able to run it."""
    admit("pairwise", induce_from(PAIRWISE))
    written = Node(kind="pairwise").to_json()
    assert "meaning" in written
    KINDS.clear()
    assert Node(kind="pairwise").apply((4, 9, 1, 2)) is None
    assert Node.from_json(written).apply((4, 9, 1, 2)) == (9, 9, 2, 2)


def test_a_node_of_a_kind_nobody_gave_a_meaning_carries_none():
    assert "meaning" not in Node(kind="then").to_json()


# ── and nothing is kept that was not worked out ──────────────────────────

def test_nothing_invented_writes_nothing():
    assert keep() is False
    assert keep_meanings() is False


def test_each_package_keeps_its_own():
    """The runtime may not reach agency or cognition, and does not have to.

    A package that describes how she judges a situation is where knowledge of
    how she judges a situation belongs; a package that holds what a rule means
    keeps its own meanings. The two are joined where reaching both is allowed,
    and the conductor asks for the join rather than importing across a
    boundary the layering gate exists to hold.
    """
    import inspect

    import core.agency.what_she_invented as properties
    import core.cognition.what_she_gave_meaning as meanings
    from core.runtime import autonomy_conductor

    assert properties.__name__.startswith("core.agency")
    assert meanings.__name__.startswith("core.cognition")
    conductor = inspect.getsource(autonomy_conductor)
    assert "core.agency.what_she_invented" not in conductor
    assert "core.cognition.what_she_gave_meaning" not in conductor
    assert "what_she_worked_out" in conductor


def test_a_recipe_that_cannot_be_read_is_skipped_rather_than_run(tmp_path, monkeypatch):
    from core.agency import what_she_invented
    from core.cognition import what_she_gave_meaning

    bad = tmp_path / "bad.json"
    bad.write_text('{"measures": [{"at": "nowhere"}]}')
    monkeypatch.setattr(what_she_invented, "_KEPT_AT", bad)
    assert recall() == {"measures": 0}

    worse = tmp_path / "worse.json"
    worse.write_text('{"x": {}}')
    monkeypatch.setattr(what_she_gave_meaning, "_KEPT_AT", worse)
    assert recall_meanings() == 0


def test_nothing_on_disk_recalls_nothing():
    assert recall() == {"measures": 0}
    assert recall_meanings() == 0


def test_what_was_worked_out_on_evidence_can_be_dropped():
    promote(A_MEASURE, 0.4)
    keep()
    assert forget_everything() is True
    _a_restart()
    assert recall()["measures"] == 0


# ── and the runtime does it without being asked ──────────────────────────

def test_the_conductor_puts_it_back_at_boot():
    import inspect

    from core.runtime.autonomy_conductor import AutonomyConductor

    source = inspect.getsource(AutonomyConductor.register_defaults)
    assert "worked_out.recall()" in source


def test_and_writes_it_down_as_it_goes():
    from core.runtime.autonomy_conductor import AutonomyConductor

    conductor = AutonomyConductor()
    conductor.register_defaults()
    assert "remember_what_she_invented" in (conductor.status().get("jobs") or {})
