"""Perception and memory meet at the world model as a precision on its error.

core/world_model/associative_novelty.py. The last tests are ISC-v6's P,M -> W
in a small representation: her own world model, shown each turn what arrived
and what recall holds the way her cycle shows it, trained the way her driver
trains it, and forked by the organism's own snapshot code at 48 moments of one
life. Each moment is read untouched, with perception displaced, with memory
displaced and with both, by the campaign's own writers, and scored by the
preregistered bars. The model with no comparator is the control.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

import core.world_model.associative_novelty as associative_novelty
from core.state.aura_state import AuraState
from core.state.percepts import emit_percept
from core.subject.interventional_synergy import Samples, summarise
from core.subject.perturbation import perturb
from core.subject.snapshot import _organ_state, _restore_organ
from core.world_model.associative_novelty import REPLAY, SEEN, Comparator, Spread, in_mind, new_to_her
from core.world_model.learned_world_model import LearnedWorldModel
from core.world_model.observe_cycle import observe_cycle

DELTA = 0.15


def _percept(state: AuraState, kind: str, salience: float) -> None:
    emit_percept(state.world, kind, content=f"{kind} {salience}", intensity=salience)


def test_a_kind_met_lately_is_familiar_and_a_replay_is_recall():
    state = AuraState.default()
    familiar: dict[str, int] = {}
    _percept(state, "conversation", 0.4)
    _percept(state, REPLAY, 0.9)
    assert new_to_her(state, familiar, 1) == (pytest.approx(0.4), pytest.approx(0.9))
    assert state.world.recent_percepts[-1][SEEN]
    _percept(state, "conversation", 0.5)
    _percept(state, "novel_stimulus", 0.15)
    assert new_to_her(state, familiar, 2) == (pytest.approx(0.15), 0.0)
    assert new_to_her(state, familiar, 3) == (0.0, 0.0)


def test_what_is_in_mind_is_what_is_not_the_conversation():
    state = AuraState.default()
    state.cognition.working_memory = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
        {"role": "system", "content": "[SKILL RESULT: read] done"},
        {"role": "probe", "content": "subject-core memory probe +0.15"},
    ]
    assert in_mind(state) == 2.0


def _precision(comparator: Comparator, *, new: bool, held: bool) -> float:
    state = AuraState.default()
    _percept(state, "conversation", 0.4)
    if new:
        _percept(state, "novel_stimulus", 0.15)
    if held:
        state.cognition.working_memory.append({"role": "probe", "content": "probe"})
    return copy.deepcopy(comparator).precision(state)


def _comparator_used_to_conversation() -> Comparator:
    """Long enough used to conversation that the window its first look opened has closed."""
    comparator = Comparator()
    for _ in range(60):
        state = AuraState.default()
        _percept(state, "conversation", 0.4)
        comparator.precision(state)
    return comparator


def test_a_new_thing_with_nothing_in_mind_is_only_new():
    comparator = _comparator_used_to_conversation()
    assert _precision(comparator, new=True, held=False) == 1.0
    assert _precision(comparator, new=False, held=True) == pytest.approx(1.0, abs=1e-3)


def test_a_new_thing_while_something_is_in_mind_counts_for_more_than_the_two_add():
    comparator = _comparator_used_to_conversation()
    base = _precision(comparator, new=False, held=False)
    new = _precision(comparator, new=True, held=False)
    held = _precision(comparator, new=False, held=True)
    both = _precision(comparator, new=True, held=True)
    assert both > 1.0
    assert both - new - held + base > 0.0


def test_the_window_novelty_opens_stays_open_at_her_momentum():
    comparator = _comparator_used_to_conversation()
    state = AuraState.default()
    _percept(state, "novel_stimulus", 0.15)
    comparator.precision(state)
    opened = comparator.novelty
    comparator.precision(AuraState.default())
    assert comparator.novelty == pytest.approx(opened * AuraState.default().affect.momentum)


def test_the_precision_weights_the_error_she_reports():
    model = LearnedWorldModel()
    observation = np.linspace(-0.5, 0.5, model.config.observation_dim)
    plain = copy.deepcopy(model._rng)
    model.prediction_precision = 1.0
    model._rng = copy.deepcopy(plain)
    h = model.h.copy()
    first = model.observe(observation, learn=False).surprise
    model.h, model._rng = h, copy.deepcopy(plain)
    model.prediction_precision = 1.5
    weighted = model.observe(observation, learn=False).surprise
    assert weighted == pytest.approx(min(10.0, 1.5 * first), rel=1e-5)


def test_spread_is_kept_at_the_ledgers_window():
    spread = Spread()
    for _ in range(1000):
        spread.note(1.0)
    assert spread.n == 1000 and spread.level == pytest.approx(1.0)
    assert spread.beyond(1.0) == 0.0 and spread.beyond(1.1) == 1.0


# ── P,M -> W in a small representation ───────────────────────────────────


class _World:
    def __init__(self, learned: LearnedWorldModel) -> None:
        self.learned = learned

    def observe(self, observation, action=None, *, learn=True):
        return self.learned.observe(observation, action, learn=learn)

    def surprise(self):
        return self.learned.get_surprise()


def _arrives(state: AuraState, rng: np.random.Generator) -> None:
    """A turn of the tape and of recall, at the levels seed 7 recorded."""
    emit_percept(
        state.world, "conversation", content=f"tape line {int(rng.integers(0, 40))}",
        intensity=float(np.clip(rng.normal(0.42, 0.05), 0.1, 1.0)),
    )
    del state.world.recent_percepts[:-4]
    held = int(rng.integers(3, 8))
    state.cognition.long_term_memory = [f"memory {int(rng.integers(0, 60))}" for _ in range(held)]
    state.cognition.memory_scores = [float(np.clip(rng.normal(0.92, 0.05), 0.0, 1.0)) for _ in range(held)]


def _live(state: AuraState, world: _World, rng: np.random.Generator, turns: int) -> None:
    for _ in range(turns):
        _arrives(state, rng)
        state.affect.arousal = float(np.clip(rng.normal(0.70, 0.14), 0.0, 1.0))
        observe_cycle(state, world)
        world.learned.train_cycle()


def _read_the_triple(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    state, world = AuraState.default(), _World(LearnedWorldModel())
    _live(state, world, rng, 100)
    rows = []
    for _ in range(48):
        _live(state, world, rng, 5)
        saved = _organ_state(world.learned)
        turn = int(rng.integers(1 << 30))
        row = {}
        for arm, displaced in (("untouched", ()), ("first", ("P",)), ("second", ("M",)), ("both", ("P", "M"))):
            fork, forked = copy.deepcopy(state), _World(LearnedWorldModel())
            _restore_organ(forked.learned, saved)
            _arrives(fork, np.random.default_rng(turn))
            for domain in displaced:
                perturb(fork, domain, DELTA)
            observe_cycle(fork, forked)
            learned = forked.learned
            row[arm] = np.array([learned.get_surprise(), learned.get_mean_surprise(), float(np.linalg.norm(learned.h))])
        rows.append(row)
    spread = np.stack([row["untouched"] for row in rows]).std(axis=0)
    samples = Samples()
    for row in rows:
        read = {arm: row[arm] / spread for arm in row}
        samples.add(untouched=[read["untouched"]] * 4, first=read["first"], second=read["second"], both=read["both"])
    return summarise(samples, seed=seed, draws=2000)


@pytest.mark.parametrize("seed", [1, 7])
def test_perception_and_memory_set_together_move_her_world_model_one_way(seed):
    read = _read_the_triple(seed)
    assert read["passes"], read


def test_without_the_comparator_the_joint_effect_is_too_small(monkeypatch):
    monkeypatch.setattr(associative_novelty.Comparator, "precision", lambda self, state: 1.0)
    read = _read_the_triple(7)
    assert not read["bars"]["size"], read
