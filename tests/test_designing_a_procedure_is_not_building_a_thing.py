"""A sourdough question wrote an HTML file to disk.

LIVE, 2026-08-28: "Design me the experiment that would actually tell us, and
say what result would prove your friend wrong" was routed as a request to build
something, and something-s-off-with-my-sourdough.html was saved.

The pattern for "design me a ..." reads as far as the determiner and stops, so
"design me the experiment" and "design me the bracket" were the same sentence
to it. What is being designed decides.

A category rather than a list of nouns somebody thought of: what separates them
is that you cannot be handed one. A page, a bracket, a deck and a schematic all
exist afterwards and can be pointed at. An experiment, a protocol, a strategy
and a curriculum are things you DO, and designing one is describing it.
"""

from __future__ import annotations

import pytest

from core.intent.artifact_request import asks_for_an_artifact, names_an_artifact


@pytest.mark.parametrize(
    "said",
    [
        "Design me the experiment that would actually tell us, and say what "
        "result would prove your friend wrong.",
        "Design an experiment to test the starter.",
        "Sketch out the protocol for the trial.",
        "Draw up the procedure for onboarding.",
    ],
)
def test_designing_a_procedure_asks_for_no_thing(said):
    assert names_an_artifact(said) is False
    assert asks_for_an_artifact(said) is False


@pytest.mark.parametrize(
    "said",
    [
        "Design me a landing page for the product.",
        "design me a small underwater drone that can hold station at 50 m",
        "draw me a schematic of the cooling loop",
        "Build me a small web app, one self-contained file.",
        "Six slides, no fluff.",
        "sketch out how the gearbox would be laid out",
    ],
)
def test_designing_a_thing_still_does(said):
    assert names_an_artifact(said) is True
    assert asks_for_an_artifact(said) is True


def test_the_noun_is_read_and_not_just_the_determiner() -> None:
    """The two sentences differ only in what is being designed."""

    assert names_an_artifact("Design me the bracket that holds 200 kg.") is True
    assert names_an_artifact("Design me the experiment that settles it.") is False
