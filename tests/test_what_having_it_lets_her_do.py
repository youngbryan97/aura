"""Things that are impossible until she has something else.

Bare hands get wood; wood makes a pickaxe; a wooden pickaxe gets stone and
nothing harder. Nobody is told this. What they learn is that the thing between
them and the blue block is not skill or effort, it is an object.

The tests are a build pipeline, because that is the same shape: the step does
not fail because it is hard, it fails until the dependency is installed.
"""

from __future__ import annotations

from core.cognition.what_having_it_lets_her_do import WhatOpensWhat


def _watched() -> WhatOpensWhat:
    opens = WhatOpensWhat()
    # Running the tests works whenever the toolchain is there, and never
    # otherwise. She is always carrying the repo, which opens nothing.
    for holding, worked in (
        ({"repo"}, False),
        ({"repo"}, False),
        ({"repo", "toolchain"}, True),
        ({"repo", "toolchain"}, True),
        ({"repo", "toolchain", "network"}, True),
    ):
        opens.she_tried("run the tests", holding=holding, it_worked=worked)
    # Publishing needs the token as well as the toolchain.
    for holding, worked in (
        ({"repo", "toolchain"}, False),
        ({"repo", "toolchain"}, False),
        ({"repo", "toolchain", "token"}, True),
        ({"repo", "toolchain", "token"}, True),
    ):
        opens.she_tried("publish", holding=holding, it_worked=worked)
    # And linting needs the toolchain too, so it stands between her and two
    # things where the token stands between her and one.
    for holding, worked in (
        ({"repo"}, False),
        ({"repo"}, False),
        ({"repo", "toolchain"}, True),
        ({"repo", "toolchain"}, True),
    ):
        opens.she_tried("lint", holding=holding, it_worked=worked)
    return opens


def test_it_finds_the_thing_that_was_in_the_way() -> None:
    opens = _watched()
    assert opens.what_opens("run the tests") == ("toolchain",)


def test_a_thing_she_always_has_is_a_key_to_nothing() -> None:
    """It looks like a key to everything, which is how it gives itself away."""
    assert "repo" not in _watched().what_opens("run the tests")


def test_a_wall_becomes_an_errand() -> None:
    opens = _watched()
    assert opens.why_it_will_not_work("run the tests", holding={"repo"}) == (
        "toolchain",
    )
    assert opens.why_it_will_not_work(
        "run the tests", holding={"repo", "toolchain"}
    ) == ()


def test_what_to_fetch_first_is_what_opens_the_most() -> None:
    """Which comes out of counting rather than out of being told."""
    opens = _watched()
    first = opens.what_to_go_and_get(
        ["run the tests", "lint", "publish"], holding={"repo"}
    )
    assert first[0] == "toolchain", first
    assert "token" in first


def test_an_act_that_never_worked_promises_nothing() -> None:
    opens = WhatOpensWhat()
    for _ in range(4):
        opens.she_tried("mine the blue block", holding={"wooden pick"}, it_worked=False)
    assert opens.what_opens("mine the blue block") == ()
    assert opens.why_it_will_not_work("mine the blue block", holding=set()) == ()


def test_what_it_learned_survives_the_process() -> None:
    opens = _watched()
    again = WhatOpensWhat.from_memory(opens.as_memory())
    assert again.what_opens("run the tests") == ("toolchain",)
    assert WhatOpensWhat.from_memory("not a memory").what_opens("anything") == ()
