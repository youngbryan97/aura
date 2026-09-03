"""What works against what, learned rather than looked up.

Nobody playing Pokémon has the type chart. They have a hundred fights. And it
is not really about elements — it is that the thing in front of you belongs to
a KIND, your options belong to kinds, and some pairs go one way every time.

The tests are about failing builds, because that is the same shape: this kind
of failure yields to that remedy and not to the other one.
"""

from __future__ import annotations

from core.cognition.what_works_against_what import WhatBeatsWhat

REMEDIES = ["retry", "clear the cache", "bump the timeout", "pin the version"]


def _learned() -> WhatBeatsWhat:
    knows = WhatBeatsWhat()
    for _ in range(8):
        knows.it_went("retry", against="a flaky network", well=True)
        knows.it_went("clear the cache", against="a flaky network", well=False)
        knows.it_went("clear the cache", against="a stale artefact", well=True)
        knows.it_went("retry", against="a stale artefact", well=False)
    return knows


def test_the_same_act_goes_two_ways_on_what_it_is_against() -> None:
    knows = _learned()
    assert knows.how_it_goes("retry", against="a flaky network") > 0.8
    assert knows.how_it_goes("retry", against="a stale artefact") < 0.2


def test_it_orders_her_options_for_the_thing_in_front_of_her() -> None:
    knows = _learned()
    first, _, _ = knows.in_order(REMEDIES, against="a stale artefact")[0]
    assert first == "clear the cache"
    first, _, _ = knows.in_order(REMEDIES, against="a flaky network")[0]
    assert first == "retry"


def test_an_untried_act_is_an_experiment_and_not_a_bad_act() -> None:
    knows = _learned()
    assert set(knows.worth_finding_out(REMEDIES, against="a flaky network")) == {
        "bump the timeout",
        "pin the version",
    }
    # Held at the honest middle: above what has failed, below what has worked.
    assert 0.2 < knows.how_it_goes("bump the timeout", against="a flaky network") < 0.8


def test_the_same_number_from_three_and_from_thirty_is_not_the_same_claim() -> None:
    knows = WhatBeatsWhat()
    for _ in range(3):
        knows.it_went("a", against="k", well=True)
    for _ in range(30):
        knows.it_went("b", against="k", well=True)
    ordered = knows.in_order(["a", "b"], against="k")
    assert ordered[0][0] == "b", ordered
    assert ordered[0][2] == 30


def test_a_kind_she_has_never_met_says_so() -> None:
    assert "nothing yet" in WhatBeatsWhat().what_it_knows_about("something new")


def test_the_table_survives_the_process() -> None:
    knows = _learned()
    again = WhatBeatsWhat.from_memory(knows.as_memory())
    assert again.in_order(REMEDIES, against="a stale artefact")[0][0] == "clear the cache"
    assert WhatBeatsWhat.from_memory(None).against == {}
