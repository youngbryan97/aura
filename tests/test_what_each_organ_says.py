"""Organs that know what they own, consume, promise, and do when they fail.

The blind comparison ended on a recommendation that is not "simplify Aura":

    You do not necessarily need fewer organs. You need organs that know
    exactly what they own, what they consume, what they promise and how
    failure propagates.

Aura has all four answers in four different places, so "does this organ know
what it is" had no answer. This asks all four of every package under core.

Where it stands: 120 organs, every one of them declares its edges, 32 answer
all four, 86 say nothing about what they promise.
"""
from __future__ import annotations

import json

import pytest

from core.verify.what_each_organ_says import (
    BASELINE,
    THE_FOUR_QUESTIONS,
    how_the_organs_answer,
    the_baseline,
    what_an_organ_says,
    which_organs_answer_nothing,
)


@pytest.fixture(scope="module")
def organs():
    return how_the_organs_answer()


# --------------------------------------------------------- the four answers


def test_the_four_questions_are_the_ones_the_review_asked():
    assert THE_FOUR_QUESTIONS == (
        "what it owns",
        "what it consumes",
        "what it promises",
        "how failure propagates",
    )


def test_every_organ_is_asked(organs):
    assert organs["organs"] > 100
    assert (
        organs["answer_all_four"] + organs["answer_some"] + organs["answer_nothing"]
        == organs["organs"]
    )


def test_nothing_is_a_complete_answer_to_what_it_consumes(organs):
    """Scoring "consumes nothing" as a gap would push every leaf to import something."""
    assert organs["who_does_not_say"]["what it consumes"] == 0
    assert organs["without_a_deps_file"] == []


def test_no_package_answers_none_of_the_four(organs):
    """A package that answers nothing is a folder, not an organ."""
    assert which_organs_answer_nothing() == []
    assert organs["answer_nothing"] == 0


# ------------------------------------------------------------ one organ


def test_an_organ_answers_from_its_own_files():
    """Nothing here is a claim about an organ the organ does not make."""
    from pathlib import Path

    where = Path(__file__).resolve().parents[1] / "core" / "runtime"
    said = what_an_organ_says(where)

    assert said["organ"] == "runtime"
    assert said["declares_its_edges"] is True
    assert said["promises"] > 0, "core/runtime declares protocols and promises"
    assert said["degrades"] > 0, "core/runtime records degradations"


def test_a_package_with_no_deps_file_says_so():
    from pathlib import Path

    said = what_an_organ_says(Path(__file__).resolve().parents[1] / "tests")
    assert said["declares_its_edges"] is False


# ------------------------------------------------------------- the ratchet


def test_the_number_answering_all_four_only_goes_up(organs):
    assert organs["answer_all_four"] >= the_baseline()["answer_all_four"]


@pytest.mark.parametrize("question", THE_FOUR_QUESTIONS)
def test_the_number_saying_nothing_only_goes_down(organs, question):
    held = the_baseline()["who_does_not_say"]
    assert organs["who_does_not_say"][question] <= held[question], (
        f"more organs stopped saying {question}"
    )


def test_the_baseline_is_readable_and_says_which_way_each_number_moves():
    held = json.loads(BASELINE.read_text("utf-8"))
    assert "only goes up" in held["note"]
    assert "only goes down" in held["note"]


def test_most_organs_still_say_nothing_about_what_they_promise(organs):
    """The honest state, pinned so closing it is visible.

    118 of 162. It was reported as 86 of 120 while the audit required an
    __init__.py and therefore never looked at 42 namespace packages — state,
    kernel, learning, ethics, health, organism, sovereign among them. A
    directory nobody wrote an __init__ for is not a random sample: it
    correlates with nobody having written its promises either.
    """
    assert organs["who_does_not_say"]["what it promises"] > 50


def test_the_audit_looks_at_namespace_packages_too(organs):
    """A package the audit cannot see is the one worth seeing."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "core"
    on_disk = {
        one.name
        for one in root.iterdir()
        if one.is_dir() and not one.name.startswith("__") and any(one.glob("*.py"))
    }
    assert organs["organs"] == len(on_disk)
    assert "state" in on_disk and "kernel" in on_disk


def test_the_count_is_in_the_health_report():
    from core.runtime.health_contract import runtime_health_report

    block = runtime_health_report()["integrity"]["what_each_organ_says"]
    assert set(block) >= {"organs", "answer_all_four", "who_does_not_say"}
