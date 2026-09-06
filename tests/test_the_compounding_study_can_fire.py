"""Why generational compounding sat at depth 0, and what it took to move.

The study searches an operator space for a rule that scores better than the
one in force, and it reported depth 0 for the chain AND for its own null.
Measured rather than assumed: 824 candidate operators at depth 3, and every
single one scored exactly 5.0 on the searched half and exactly 0.0 on the held
out half. The cost was a constant with respect to the thing being varied, so
no search could ever have found anything.

Two causes, and each is a mechanism that could not fire.

The replay prices a chosen action by looking up what that action historically
cost — `cost_of[(family, action_name)]`. It built that index from
`Episode.route`, and route names the action only when the change was KEPT. In
a record of 512 episodes, not one route was an action name.

And the failure path never wrote the name down at all.
`note_an_episode(family, route=None, walked=...)` dropped it, though the
record has a field for exactly this. `Episode.tried`'s own docstring says why
it exists: "route names the action only when the change was kept, so a family
where everything she has was tried and nothing held was indistinguishable from
one she never tried". It was declared and never passed.
"""
from __future__ import annotations

import pytest

from core.cognition.does_improving_compound import _replay


@pytest.fixture
def _a_clean_record(monkeypatch, tmp_path):
    from core.cognition import the_record_of_her_own_work as record

    monkeypatch.setattr(record, "_KEPT_AT", tmp_path / "record.json")
    holder = record.the_record()
    monkeypatch.setattr(holder, "kept", [], raising=False)
    return holder


# ------------------------------------------------------ the name is written


def test_a_failed_action_now_records_what_was_tried():
    """The field was declared for this and the failure path did not pass it."""
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1]
        / "core" / "cognition" / "sequence_induction.py"
    ).read_text("utf-8")
    assert "tried=decided.action.name," in source


def test_an_episode_keeps_what_was_tried_apart_from_what_worked():
    from core.cognition.the_record_of_her_own_work import Episode

    failed = Episode(family="f", route=None, walked=7, tried="an operator she tried")
    worked = Episode(family="f", route="an operator that held", walked=3)

    assert failed.route is None
    assert failed.tried == "an operator she tried"
    # When nothing else is said, `tried` falls back to the route, so a kept
    # change is still findable by the same key.
    assert worked.tried in (None, "an operator that held")


# --------------------------------------------------- the price can be found


def _priced(episodes, family: str, picked: str) -> float:
    cost_of: dict[tuple[str, str | None], list[int]] = {}
    for one in episodes:
        cost_of.setdefault((one.family, one.route), []).append(one.walked)
        acted = getattr(one, "tried", None)
        if acted and acted != one.route:
            cost_of.setdefault((one.family, acted), []).append(one.walked)
    seen = cost_of.get((family, picked))
    return sum(seen) / len(seen) if seen else float("nan")


def test_an_action_that_was_tried_and_failed_can_now_be_priced():
    """Before, only an action that had once WORKED had a price."""
    from core.cognition.the_record_of_her_own_work import Episode

    episodes = [
        Episode(family="sequences", route=None, walked=40, tried="a costly operator"),
        Episode(family="sequences", route=None, walked=44, tried="a costly operator"),
        Episode(family="sequences", route="a cheap operator", walked=4),
    ]
    assert _priced(episodes, "sequences", "a costly operator") == 42.0
    assert _priced(episodes, "sequences", "a cheap operator") == 4.0


def test_two_operators_that_pick_differently_are_priced_differently():
    """The property the whole study rests on, and it did not hold."""
    from core.cognition.the_record_of_her_own_work import Episode

    kept = [
        Episode(family="sequences", route=None, walked=40, tried="expensive"),
        Episode(family="sequences", route="cheap", walked=4),
    ]
    cost_of: dict[tuple[str, str | None], list[int]] = {}
    for one in kept:
        cost_of.setdefault((one.family, one.route), []).append(one.walked)
        acted = getattr(one, "tried", None)
        if acted and acted != one.route:
            cost_of.setdefault((one.family, acted), []).append(one.walked)

    import core.cognition.she_decides_to_develop as deciding

    class ADecision:
        def __init__(self, name):
            self.action = type("An", (), {"name": name})()

    prices = []
    for name in ("expensive", "cheap"):
        original = deciding.what_to_do_next
        try:
            deciding.what_to_do_next = lambda *a, **k: ADecision(name)  # noqa: B023
            prices.append(_replay(["sequences"], cost_of, kept))
        finally:
            deciding.what_to_do_next = original

    assert prices[0] != prices[1], (
        "two operators choosing differently must cost differently, or no "
        "search over operators can find anything"
    )
    assert prices[0] > prices[1]


def test_an_action_nobody_has_tried_falls_back_to_the_family_average():
    """Which is right, and was the answer for every operator before."""
    from core.cognition.the_record_of_her_own_work import Episode

    kept = [Episode(family="sequences", route=None, walked=10)]
    cost_of = {("sequences", None): [10]}

    import core.cognition.she_decides_to_develop as deciding

    original = deciding.what_to_do_next
    try:
        deciding.what_to_do_next = lambda *a, **k: type(
            "D", (), {"action": type("A", (), {"name": "never tried"})()}
        )()
        assert _replay(["sequences"], cost_of, kept) == 10.0
    finally:
        deciding.what_to_do_next = original
