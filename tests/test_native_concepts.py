"""Concepts that are not words, and the ones that only look like it.

The end state worth wanting is internal representations optimised for
computation rather than communication, with language as one readout channel:
learned concepts that resist word-for-word translation and still take part in
reasoning.

The second half is the one that gets skipped. A latent that resists
translation is easy to produce — most of them do, and most of them are noise.
What makes one a concept is that using it beats using the best available word
on something she has to do. Untranslatable and useless is a residual, and
calling that ineffable is how a system acquires a private vocabulary that
means nothing.

Every case here has a known answer by construction.
"""

from __future__ import annotations

import random

import pytest

from core.cognition.native_concepts import (
    MIN_INSTANCES,
    Instance,
    Kind,
    assess,
    assess_all,
    medium_report,
)


def build(n: int = 80, seed: int = 11):
    """The outcome is an XOR of two features. No word in the lexicon names it."""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        first, second, noise = rng.random(), rng.random(), rng.random()
        out.append(
            Instance(
                features=(
                    first - 0.5,
                    second - 0.5,
                    noise - 0.5,
                    (first - 0.5) * (second - 0.5) * 4,
                ),
                label="heavy" if first > 0.5 else "light",
                outcome=(first > 0.5) != (second > 0.5),
            )
        )
    return out


SAYS_HEAVY = (1.0, 0.0, 0.0, 0.0)
THE_XOR_AXIS = (0.0, 0.0, 0.0, 1.0)
PURE_NOISE = (0.0, 0.0, 1.0, 0.0)


@pytest.fixture
def instances():
    return build()


# ── the three kinds ──────────────────────────────────────────────────────


def test_a_direction_a_word_already_names_is_verbal(instances):
    result = assess("says_heavy", SAYS_HEAVY, instances)
    assert result.kind is Kind.VERBAL
    assert result.translatability > 0.9
    assert result.nearest_word == "heavy"


def test_a_direction_no_word_names_that_decides_the_task_is_native(instances):
    result = assess("the_xor_axis", THE_XOR_AXIS, instances)
    assert result.kind is Kind.NATIVE
    assert result.translatability < 0.8
    assert result.participation > 0.2


def test_a_direction_no_word_names_that_decides_nothing_is_a_residual(instances):
    """The category most 'ineffable latent' claims actually belong to."""
    result = assess("pure_noise", PURE_NOISE, instances)
    assert result.kind is Kind.RESIDUAL
    assert result.translatability < 0.8
    assert result.participation < 0.1


def test_the_residual_verdict_is_reachable(instances):
    """An instrument whose disappointing outcome cannot occur agrees with
    whoever ran it."""
    kinds = {a.kind for a in assess_all(
        {"a": SAYS_HEAVY, "b": THE_XOR_AXIS, "c": PURE_NOISE}, instances
    )}
    assert kinds == {Kind.VERBAL, Kind.NATIVE, Kind.RESIDUAL}


# ── both measurements are needed ─────────────────────────────────────────


def test_untranslatable_alone_is_not_enough(instances):
    noise = assess("pure_noise", PURE_NOISE, instances)
    native = assess("the_xor_axis", THE_XOR_AXIS, instances)
    assert noise.translatability == pytest.approx(native.translatability, abs=0.15)
    assert noise.kind is not native.kind, (
        "two directions equally hard to translate, and only one is a concept"
    )


def test_participation_is_measured_on_the_task_not_asserted(instances):
    """Same direction, outcomes shuffled: it stops participating."""
    import random as _random

    rng = _random.Random(3)
    scrambled = [
        Instance(features=i.features, label=i.label, outcome=rng.random() < 0.5)
        for i in instances
    ]
    real = assess("the_xor_axis", THE_XOR_AXIS, instances)
    fake = assess("the_xor_axis", THE_XOR_AXIS, scrambled)
    assert real.kind is Kind.NATIVE
    assert fake.participation < real.participation


def test_a_word_that_anti_correlates_still_names_it(instances):
    """Saying the same thing inverted is still the word for it."""
    flipped = [
        Instance(features=tuple(-f for f in i.features), label=i.label, outcome=i.outcome)
        for i in instances
    ]
    assert assess("says_heavy", SAYS_HEAVY, flipped).kind is Kind.VERBAL


# ── the measurements themselves ──────────────────────────────────────────


def test_too_few_instances_cannot_tell_any_of_these_apart():
    result = assess("x", SAYS_HEAVY, build(n=MIN_INSTANCES - 1))
    assert result.kind is Kind.UNMEASURED


def test_the_split_is_at_the_median_not_at_zero():
    """A real concept that is off-centre otherwise separates nothing."""
    offset = [
        Instance(features=(f + 10.0,), label="w" if index % 2 else "", outcome=index % 2 == 0)
        for index, f in enumerate([-3, -2, -1, 0, 1, 2, 3, 4, 5, 6])
    ]
    result = assess("offset", (1.0,), offset)
    assert result.kind is not Kind.UNMEASURED
    assert 0.0 < result.translatability <= 1.0


def test_an_instance_with_no_outcome_does_not_count_toward_participation():
    ungraded = [
        Instance(features=i.features, label=i.label, outcome=None) for i in build()
    ]
    assert assess("the_xor_axis", THE_XOR_AXIS, ungraded).participation == 0.0


# ── the report ───────────────────────────────────────────────────────────


def test_the_report_counts_what_is_not_sayable(instances):
    report = medium_report(
        assess_all({"a": SAYS_HEAVY, "b": THE_XOR_AXIS, "c": PURE_NOISE}, instances)
    )
    assert report["native"] == 1
    assert report["verbal"] == 1
    assert report["residual"] == 1
    assert report["names"] == ["the_xor_axis"]


def test_no_natives_is_a_real_finding(instances):
    """Language with extra steps, which is what to expect at first."""
    report = medium_report(assess_all({"a": SAYS_HEAVY, "c": PURE_NOISE}, instances))
    assert report["native"] == 0
    assert report["native_fraction"] == 0.0


def test_residuals_do_not_count_toward_the_fraction(instances):
    report = medium_report(assess_all({"b": THE_XOR_AXIS, "c": PURE_NOISE}, instances))
    assert report["native_fraction"] == 1.0, (
        "a residual is not part of what she thinks in, so it does not dilute "
        "or inflate how much of it is unsayable"
    )
