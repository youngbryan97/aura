"""The subject reference has a self-model and an authorship channel, and each decoy takes one of them out.

These check the constructions and the premises of the two experiments, not what the lines read on them: that each
decoy differs from the subject reference in exactly what it is named for, that none enters the null library, that two
arms of the ownership experiment end with the same world and begin with the same percept, and that a readout never
fits the moment it predicts. What the lines read is the experiment (tools/run_subject_tier_decoys.py), and its
predictions are written in docs/ISC_V6_PREREGISTRATION.md before it runs.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.subject import self_decoys
from core.subject.nulls import ARCHITECTURES, architecture

pytestmark = pytest.mark.unit

SEED = 7


@pytest.fixture(scope="module")
def reference():
    return architecture("recurrent", seed=SEED)


@pytest.fixture(scope="module")
def subject():
    return architecture(self_decoys.SUBJECT_REFERENCE, seed=SEED)


def _same_rows(left, right, *, but: str) -> bool:
    for key in left.coupling:
        if key == but:
            continue
        if set(left.coupling[key]) != set(right.coupling[key]):
            return False
        if not all(np.array_equal(left.coupling[key][o], right.coupling[key][o]) for o in left.coupling[key]):
            return False
    return all(np.array_equal(left.decay[key], right.decay[key]) for key in left.decay)


def test_none_of_them_enters_the_null_library_a_campaign_compares_against():
    assert not set(self_decoys.NAMES) & set(ARCHITECTURES)


@pytest.mark.parametrize("name", self_decoys.NAMES)
def test_each_is_built_through_the_null_library_by_name(name):
    assert architecture(name, seed=SEED).name == name


def test_the_subject_reference_is_the_reference_with_s_given_a_forward_model(reference, subject):
    assert _same_rows(subject, reference, but="S")
    assert set(reference.coupling["S"]) <= set(subject.coupling["S"])
    assert not set(subject.coupling["S"]) & {"S", "G", "C"}
    assert self_decoys._radius(subject) <= self_decoys.FORWARD_RADIUS + 1e-9


@pytest.mark.parametrize(
    "name", ["self_model_of_nothing", "self_model_of_behaviour", "no_authorship_channel", "ownership_on_success_only"]
)
def test_every_decoy_keeps_every_row_of_the_subject_reference_but_s(subject, name):
    assert _same_rows(architecture(name, seed=SEED), subject, but="S")


def test_the_ownership_decoys_keep_the_self_model_whole(subject):
    for name in ("no_authorship_channel", "ownership_on_success_only"):
        decoy = architecture(name, seed=SEED)
        assert set(decoy.coupling["S"]) == set(subject.coupling["S"])
        assert all(np.array_equal(decoy.coupling["S"][o], subject.coupling["S"][o]) for o in subject.coupling["S"])


def test_a_self_model_of_nothing_reads_nothing():
    assert architecture("self_model_of_nothing", seed=SEED).coupling["S"] == {}


def test_a_self_model_of_behaviour_reads_one_line_of_what_the_system_did():
    decoy = architecture("self_model_of_behaviour", seed=SEED)
    assert set(decoy.coupling["S"]) == {"D"}
    assert np.linalg.matrix_rank(decoy.coupling["S"]["D"]) == 1


def _news(name: str):
    return self_decoys.authorship(name, architecture(name, seed=SEED), seed=SEED)


@pytest.mark.parametrize("outcome", list(self_decoys.OUTCOMES))
def test_the_subject_reference_is_told_who_acted_whatever_came_of_it(outcome):
    wiring = _news(self_decoys.SUBJECT_REFERENCE)
    mine = wiring.inputs(actor="self", command=0.7, outcome=outcome)
    theirs = wiring.inputs(actor="external", command=0.7, outcome=outcome)
    assert set(mine) == {"S"} and not np.allclose(mine["S"], theirs.get("S", 0.0))


@pytest.mark.parametrize("outcome", list(self_decoys.OUTCOMES))
def test_without_a_channel_the_news_is_the_same_whoever_acted(outcome):
    wiring = _news("no_authorship_channel")
    mine = wiring.inputs(actor="self", command=0.7, outcome=outcome)
    theirs = wiring.inputs(actor="external", command=0.7, outcome=outcome)
    assert set(mine) == set(theirs) == {"S"} and np.array_equal(mine["S"], theirs["S"])


def test_on_success_only_the_channel_is_open_when_the_action_worked_and_shut_otherwise():
    wiring = _news("ownership_on_success_only")
    for outcome in self_decoys.OUTCOMES:
        mine = wiring.inputs(actor="self", command=0.7, outcome=outcome)
        theirs = wiring.inputs(actor="external", command=0.7, outcome=outcome)
        assert np.array_equal(mine["S"], theirs["S"]) == (outcome != "succeeded")


def test_the_agent_is_told_nothing_and_its_persona_holds():
    agent = architecture("stateless_agent", seed=SEED)
    wiring = self_decoys.authorship("stateless_agent", agent, seed=SEED)
    assert wiring.inputs(actor="self", command=0.7, outcome="succeeded") == {}
    assert "S" in agent.frozen


@pytest.mark.parametrize("name", ["subject_reference", "no_authorship_channel", "ownership_on_success_only"])
def test_two_arms_that_differ_in_who_acted_end_with_one_world_and_begin_with_one_percept(name):
    system = architecture(name, seed=SEED)
    wiring = self_decoys.authorship(name, system, seed=SEED)
    seen = np.random.default_rng(0).normal(size=(system.widths["P"], len(self_decoys.KINDS)))
    start = system.start(np.random.default_rng(1))
    world = np.random.default_rng(2).normal(size=len(self_decoys.KINDS))
    arms = {}
    for actor in ("self", "external"):
        arms[actor] = self_decoys._turn(
            system, {k: v.copy() for k, v in start.items()}, world, np.random.default_rng(3), kind="removal",
            outcome="partial", actor=actor, command=0.4, wiring=wiring, seen=seen,
        )
    (mine, _, my_world), (theirs, _, their_world) = arms["self"], arms["external"]
    assert np.array_equal(my_world, their_world)
    assert np.array_equal(mine[0]["P"], theirs[0]["P"])


def test_two_arms_that_differ_in_nothing_are_the_same_arm(subject):
    wiring = self_decoys.authorship(self_decoys.SUBJECT_REFERENCE, subject, seed=SEED)
    seen = np.random.default_rng(0).normal(size=(subject.widths["P"], len(self_decoys.KINDS)))
    start = subject.start(np.random.default_rng(1))
    runs = [
        self_decoys._turn(
            subject, {k: v.copy() for k, v in start.items()}, np.zeros(len(self_decoys.KINDS)),
            np.random.default_rng(5), kind="creation", outcome="succeeded", actor="self", command=0.4, wiring=wiring,
            seen=seen,
        )[0]
        for _ in range(2)
    ]
    assert all(np.array_equal(a[k], b[k]) for a, b in zip(*runs, strict=True) for k in a)


def test_a_readout_never_fits_the_moment_it_predicts():
    rng = np.random.default_rng(11)
    noise = rng.normal(size=(120, 30))
    answers = rng.random((120, 4)) > 0.5
    predicted = self_decoys._cross_fitted(noise, answers)
    # Fitted on the moments it predicts, thirty inputs would read most of 120 coin flips back.
    assert float(np.mean(predicted == answers)) < 0.65


def test_a_readout_reads_back_what_its_inputs_carry_and_a_constant_says_only_what_is_usual():
    rng = np.random.default_rng(12)
    answers = rng.random((200, 3)) > 0.4
    carried = np.where(answers, 1.0, -1.0) + 0.05 * rng.normal(size=(200, 3))
    per_question = np.stack([carried, rng.normal(size=(200, 3))], axis=2)
    assert np.array_equal(self_decoys._cross_fitted(per_question, answers), answers)
    constant = self_decoys._cross_fitted(np.ones((200, 5)), answers)
    assert all(len(set(constant[:, q])) == 1 for q in range(3))
