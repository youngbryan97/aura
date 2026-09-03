"""Working out what is hidden from what nobody was able to produce.

The good Cluedo players track crosses, not ticks. A cross is not something
anybody said — it is what follows from somebody being unable to answer.

The tests are a failing request, because it is the same question: which
component could not have produced this.
"""

from __future__ import annotations

from core.cognition.what_nobody_could_show import WhatIsHidden


def _hunt() -> WhatIsHidden:
    return WhatIsHidden(
        candidates=("the cache", "the router", "the database", "the client"),
        parties=("the logs", "the traces", "the metrics"),
    )


def test_a_failure_to_produce_settles_several_at_once() -> None:
    """Which is why it is stronger than a sighting: a sighting settles one."""
    hunt = _hunt()
    hunt.could_not_produce("the logs", ["the cache", "the router"])
    assert "the logs" not in hunt.who_could_hold("the cache")
    assert "the logs" not in hunt.who_could_hold("the router")
    assert "the logs" in hunt.who_could_hold("the database")


def test_what_nobody_can_hold_is_the_answer() -> None:
    hunt = _hunt()
    # Nobody could produce the database. Everything else, somebody could.
    for who in ("the logs", "the traces", "the metrics"):
        hunt.could_not_produce(who, ["the database"])
    assert hunt.what_it_must_be() == ("the database",)
    assert "can only be the database" in hunt.describe()


def test_several_left_is_a_real_answer_and_not_a_failure() -> None:
    """It means the evidence narrows it to these and no further, and saying
    which would be making something up."""
    hunt = _hunt()
    for who in ("the logs", "the traces", "the metrics"):
        hunt.could_not_produce(who, ["the cache", "the router"])
    assert set(hunt.what_it_must_be()) == {"the cache", "the router"}
    assert "nothing says which" in hunt.describe()


def test_holding_one_of_several_settles_once_the_others_are_crossed_off() -> None:
    """Worth little on its own and a great deal later."""
    hunt = _hunt()
    hunt.produced_one_of("the traces", ["the cache", "the router"])
    hunt.could_not_produce("the traces", ["the cache"])
    # It held one of the two and cannot hold the cache, so it holds the
    # router — and therefore nobody else does.
    assert hunt.who_could_hold("the router") == ("the traces",)


def test_it_runs_to_a_standstill_rather_than_once() -> None:
    """Settling one group crosses things off, which can settle another."""
    hunt = WhatIsHidden(
        candidates=("a", "b", "c"), parties=("one", "two", "three")
    )
    hunt.produced_one_of("one", ["a", "b"])
    hunt.produced_one_of("two", ["b", "c"])
    hunt.could_not_produce("one", ["b"])
    # one holds a, so two cannot; two held b or c and b is still open to it.
    assert hunt.who_could_hold("a") == ("one",)


def test_asking_about_something_settled_learns_nothing() -> None:
    hunt = _hunt()
    for who in ("the logs", "the traces", "the metrics"):
        hunt.could_not_produce(who, ["the cache"])
    worth = dict(hunt.what_would_settle_it(list(hunt.candidates)))
    assert "the cache" not in worth, "already settled"
    assert worth.get("the database") == 3
