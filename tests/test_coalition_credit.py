"""One-at-a-time lesioning cannot tell redundancy from irrelevance.

`core/verify/causal_influence.py` asks whether a faculty changed the output by
lesioning it alone. That is the right question with one blind spot, and the
blind spot is not small: two mechanisms that back each other up each read as
doing nothing when lesioned alone, and the pair is essential. A system with
thousands of parts and a one-at-a-time protocol will report most of them as
inert and will be wrong about which.

Every test here runs against a system whose structure is known by
construction, so the estimator can be checked rather than believed.
"""

from __future__ import annotations

import pytest

from core.verify.coalition_credit import (
    Role,
    attribute,
)


def duplicate_and_partner(active):
    """a and b are interchangeable; c and d only work together; e does nothing."""
    value = 0.0
    if "a" in active or "b" in active:
        value += 1.0
    if "c" in active and "d" in active:
        value += 1.0
    return value


@pytest.fixture
def known():
    return attribute(["a", "b", "c", "d", "e"], duplicate_and_partner)


def _credit(result, channel):
    return next(c for c in result.credits if c.channel == channel)


# ── the structure is recovered ───────────────────────────────────────────


def test_interchangeable_duplicates_are_found_redundant(known):
    for name in ("a", "b"):
        credit = _credit(known, name)
        assert credit.role is Role.REDUNDANT
        assert credit.leave_one_out == pytest.approx(0.0)
        assert credit.marginal == pytest.approx(0.5)


def test_a_pair_that_only_works_together_is_found_synergistic(known):
    for name in ("c", "d"):
        credit = _credit(known, name)
        assert credit.role is Role.SYNERGISTIC
        assert credit.leave_one_out == pytest.approx(1.0)
        assert credit.marginal == pytest.approx(0.5)


def test_a_mechanism_that_does_nothing_is_inert(known):
    credit = _credit(known, "e")
    assert credit.role is Role.INERT
    assert credit.marginal == pytest.approx(0.0)


def test_the_credits_sum_to_the_whole_systems_value(known):
    """Shapley's defining property, and a check the estimator is not drifting."""
    total = sum(c.marginal for c in known.credits)
    assert total == pytest.approx(duplicate_and_partner({"a", "b", "c", "d", "e"}))


# ── the finding this module exists for ───────────────────────────────────


def test_it_names_what_a_one_at_a_time_protocol_would_have_missed(known):
    hidden = {c.channel for c in known.hidden}
    assert hidden == {"a", "b"}, (
        "the redundant pair — each worthless to remove, both contributing — is "
        "exactly what the existing lesion protocol reports as inert"
    )


def test_an_independent_mechanism_agrees_with_the_one_at_a_time_verdict():
    def additive(active):
        return len(active) * 1.0

    result = attribute(["p", "q", "r"], additive)
    for credit in result.credits:
        assert credit.role is Role.INDEPENDENT
        assert credit.interaction == pytest.approx(0.0, abs=1e-9)


# ── the estimator ────────────────────────────────────────────────────────


def test_a_small_system_is_solved_exactly_rather_than_sampled(known):
    for credit in known.credits:
        assert credit.standard_error == 0.0, (
            "a system small enough to enumerate was sampled anyway"
        )


def test_sampling_recovers_the_exact_answer_on_a_larger_system():
    names = [f"m{i}" for i in range(12)]

    def system(active):
        value = float(len(active & {"m0", "m1", "m2"}))
        if {"m3", "m4"} <= active:
            value += 2.0
        return value

    result = attribute(names, system, permutations=400)
    for name in ("m0", "m1", "m2"):
        assert _credit(result, name).marginal == pytest.approx(1.0, abs=0.15)
    for name in ("m3", "m4"):
        assert _credit(result, name).marginal == pytest.approx(1.0, abs=0.2)
    assert _credit(result, "m11").marginal == pytest.approx(0.0, abs=0.15)


def test_the_answer_does_not_change_between_processes():
    """A credit assignment that moved with the seed would rank the seed."""
    names = [f"m{i}" for i in range(10)]

    def system(active):
        return float(len(active))

    first = attribute(names, system, permutations=64)
    second = attribute(names, system, permutations=64)
    assert [c.to_dict() for c in first.credits] == [c.to_dict() for c in second.credits]


def test_a_value_inside_the_noise_is_unmeasured_not_a_small_number():
    import random

    rng = random.Random(11)
    names = [f"m{i}" for i in range(9)]

    def noisy(active):
        return rng.gauss(0.0, 1.0)

    result = attribute(names, noisy, permutations=32)
    assert any(c.role is Role.UNMEASURED for c in result.credits), (
        "pure noise was reported as a set of small real contributions"
    )


def test_each_coalition_is_evaluated_once(known):
    """Trials are what they say, not what the permutations imply."""
    assert known.trials <= 2 ** 5


def test_no_channels_is_not_a_crash():
    result = attribute([], lambda active: 0.0)
    assert result.credits == () and result.trials == 0


# ── through the real lesion registry ─────────────────────────────────────


@pytest.fixture
def registered():
    from core.verify.lesion_registry import (
        LesionHandle,
        get_lesion_registry,
        reset_lesion_registry_for_test,
    )

    reset_lesion_registry_for_test()
    state = {"a": True, "b": True, "c": True, "d": True}
    for name in state:
        get_lesion_registry().register(
            LesionHandle(
                channel=name,
                lesion=(lambda n=name: state.__setitem__(n, False)),
                restore=(lambda n=name: state.__setitem__(n, True)),
                owner="test",
                neutral_description=f"{name} forced off",
                direct_actuation=True,
            ),
            replace=True,
        )
    yield state
    reset_lesion_registry_for_test()


def test_credit_runs_over_real_registered_lesions(registered):
    from core.verify.coalition_credit import attribute_registered

    def measure():
        value = 0.0
        if registered["a"] or registered["b"]:
            value += 1.0
        if registered["c"] and registered["d"]:
            value += 1.0
        return value

    result = attribute_registered(["a", "b", "c", "d"], measure)
    roles = {c.channel: c.role for c in result.credits}
    assert roles["a"] is Role.REDUNDANT and roles["b"] is Role.REDUNDANT
    assert roles["c"] is Role.SYNERGISTIC and roles["d"] is Role.SYNERGISTIC


def test_every_lesion_is_restored_after_a_run(registered):
    from core.verify.coalition_credit import attribute_registered

    attribute_registered(["a", "b", "c", "d"], lambda: float(sum(registered.values())))
    assert all(registered.values()), "a coalition run left a faculty lesioned"


def test_a_channel_with_no_registered_lesion_is_skipped_not_guessed(registered):
    from core.verify.coalition_credit import attribute_registered

    result = attribute_registered(["a", "b", "nobody_registered_this"], lambda: 1.0)
    assert {c.channel for c in result.credits} == {"a", "b"}


def test_no_registered_channels_returns_nothing_rather_than_a_verdict():
    from core.verify.coalition_credit import attribute_registered
    from core.verify.lesion_registry import reset_lesion_registry_for_test

    reset_lesion_registry_for_test()
    assert attribute_registered(["x"], lambda: 1.0).credits == ()
