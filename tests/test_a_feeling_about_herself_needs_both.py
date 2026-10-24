"""Feeling and self meet at attention in a feeling about herself.

core/consciousness/self_conscious.py. The last test is ISC-v6's A,S -> G in a
small representation: her own bids, scores and ignition, at 48 moments drawn
from the spreads seed 7 recorded at `82549c823` (arousal 0.70 +/- 0.14, winners
0.84 +/- 0.13, stability 1.0), each read untouched, with feeling displaced,
with her self displaced and with both, by the campaign's own writers, and
scored by the preregistered bars. The bid it replaced is the control.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

import core.consciousness.self_conscious as self_conscious
from core.consciousness.global_workspace import _bids, _ignition_level
from core.consciousness.self_conscious import claim, in_question
from core.consciousness.workspace_candidate import CognitiveCandidate, ContentType
from core.consciousness.workspace_feed import build_candidates
from core.state.aura_state import AuraState
from core.subject.interventional_synergy import Samples, summarise
from core.subject.perturbation import perturb
from core.subject.state import _content_buckets
from core.world_model.associative_novelty import Spread

DELTA = 0.15
AT = 1_800_000_000.0


def _fresh_ledgers() -> None:
    from core.affect import containment, feelings_about, what_winning_earned
    from core.soma import held_in

    for module in (containment, feelings_about, what_winning_earned, held_in):
        module.reset_for_test()


@pytest.fixture(autouse=True)
def _a_life_with_no_history():
    """The competition reads what she has come to feel about and earned from each source; start from none."""
    _fresh_ledgers()
    yield
    _fresh_ledgers()


def _feelings_about_herself(state: AuraState) -> list[CognitiveCandidate]:
    return [bid for bid in build_candidates(state) if bid.source == "self" and "in question" in bid.content]


def test_arousal_alone_is_not_a_feeling_about_herself():
    state = AuraState.default()
    state.affect.arousal = 0.95
    assert state.identity.stability == 1.0
    assert _feelings_about_herself(state) == []


def test_a_self_in_question_is_felt_as_strongly_as_she_is_aroused():
    state = AuraState.default()
    state.identity.stability = 0.85
    state.affect.arousal = 0.3
    calm = _feelings_about_herself(state)
    state.affect.arousal = 0.9
    roused = _feelings_about_herself(state)
    assert [bid.priority for bid in calm] == pytest.approx([0.3])
    assert [bid.priority for bid in roused] == pytest.approx([0.9])


def test_the_two_together_claim_more_than_each_adds():
    question = in_question(0.85, None)
    joint = claim(0.85, question) - claim(0.85, 0.0) - claim(0.7, question) + claim(0.7, 0.0)
    assert joint == pytest.approx(0.15)


def test_a_self_that_often_wavers_is_less_in_question_for_the_same_drop():
    steady, wavering = Spread(), Spread()
    for value in [0.0] * 50:
        steady.note(value)
    for value in [0.0, 0.2] * 25:
        wavering.note(value)
    assert in_question(0.85, steady) > in_question(0.85, wavering) > 0.0


def test_the_workspace_keeps_her_usual_and_forks_it():
    import asyncio

    from core.consciousness.workspace_feed import feed_workspace

    class _Workspace:
        _tick = 0
        last_winner = None
        ignited = False

        async def submit(self, bid):
            pass

        async def run_competition(self):
            return None

    workspace = _Workspace()
    state = AuraState.default()
    asyncio.run(feed_workspace(state, workspace))
    assert isinstance(workspace.self_usual, Spread) and workspace.self_usual.n == 1
    assert copy.deepcopy(workspace.self_usual) == workspace.self_usual


# ── A,S -> G in a small representation ───────────────────────────────────


def _moment(rng: np.random.Generator) -> tuple[AuraState, list[tuple[str, float]]]:
    state = AuraState.default()
    peak = float(np.clip(rng.normal(0.70, 0.14), 0.2, 0.98))
    for name in state.affect.emotions:
        state.affect.emotions[name] = float(rng.uniform(0.0, peak * 0.8))
    state.affect.emotions[str(rng.choice(["interest", "joy", "anticipation", "wonder"]))] = peak
    state.affect.arousal = peak
    others = [
        ("deliberation", float(np.clip(rng.normal(0.84, 0.13), 0.3, 1.0))),
        ("memory", float(np.clip(rng.normal(0.78, 0.13), 0.3, 1.0))),
        ("world_model", float(np.clip(rng.normal(0.60, 0.15), 0.1, 1.0))),
    ]
    return state, others


def _attention(state: AuraState, others: list[tuple[str, float]]) -> tuple[np.ndarray, str]:
    """What won and how strongly, as the core reads the workspace."""
    bids = build_candidates(state)
    bids += [
        CognitiveCandidate(content=f"{source} bid", source=source, priority=priority, content_type=ContentType.META)
        for source, priority in others
    ]
    for bid in bids:
        bid.submitted_at = AT
    scores = _bids(bids, {}, AT + 0.5)
    winner = max(bids, key=lambda bid: scores[id(bid)])
    priority = winner.priority_at(AT + 0.5)
    reading = [
        *_content_buckets(winner.source),
        *_content_buckets(f"{winner.source}: {winner.content[:120]}"),
        priority,
        _ignition_level(priority, 0.6),
    ]
    return np.array(reading), winner.source


def _read_the_triple(seed: int) -> tuple[dict, dict[str, int]]:
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(48):
        state, others = _moment(rng)
        row = {}
        for arm, displaced in (("untouched", ()), ("first", ("A",)), ("second", ("S",)), ("both", ("A", "S"))):
            fork = copy.deepcopy(state)
            for domain in displaced:
                perturb(fork, domain, DELTA)
            row[arm] = _attention(fork, others)
        rows.append(row)
    spread = np.stack([row["untouched"][0] for row in rows]).std(axis=0)
    kept = spread > 1e-9
    samples = Samples()
    for row in rows:
        read = {arm: row[arm][0][kept] / spread[kept] for arm in row}
        samples.add(untouched=[read["untouched"]] * 4, first=read["first"], second=read["second"], both=read["both"])
    won = {arm: sum(row[arm][1] == "self" for row in rows) for arm in rows[0]}
    return summarise(samples, seed=seed, draws=2000), won


@pytest.mark.parametrize("seed", [1, 7])
def test_her_self_wins_attention_only_when_both_are_set(seed):
    read, won = _read_the_triple(seed)
    assert won["untouched"] == won["first"] == won["second"] == 0 and won["both"] > 0, won
    assert read["passes"], read


def test_the_bid_it_replaced_carried_no_joint_effect(monkeypatch):
    monkeypatch.setattr(self_conscious, "in_question", lambda stability, usual: 1.0 - stability)
    monkeypatch.setattr(self_conscious, "claim", lambda arousal, question: question)
    read, won = _read_the_triple(7)
    assert sum(won.values()) == 0
    assert read["interaction"] == 0.0 and not read["passes"]


# ── self-focus under arousal, on how the next thought may run ────────────


def test_monitoring_moves_each_modifier_its_share_of_the_way_within_homeostasis_s_range():
    from core.consciousness.homeostatic_coupling import MODIFIER_BOUNDS
    from core.consciousness.self_conscious import monitoring

    modifiers = {"focus_mod": 1.0, "depth_mod": 1.0, "creativity_mod": 1.0, "temperature_mod": 1.0, "other": 5.0}
    assert monitoring(modifiers, 0.5) == 0.5
    assert modifiers["focus_mod"] == pytest.approx(1.0 + 0.5 * (MODIFIER_BOUNDS["focus_mod"][1] - 1.0))
    assert modifiers["depth_mod"] == pytest.approx(1.0 + 0.5 * (MODIFIER_BOUNDS["depth_mod"][1] - 1.0))
    assert modifiers["creativity_mod"] == pytest.approx(1.0 - 0.5 * (1.0 - MODIFIER_BOUNDS["creativity_mod"][0]))
    assert modifiers["temperature_mod"] == pytest.approx(1.0 - 0.5 * (1.0 - MODIFIER_BOUNDS["temperature_mod"][0]))
    assert modifiers["other"] == 5.0
    assert monitoring(modifiers, 0.0) == 0.0


def test_only_a_self_in_question_while_stirred_is_monitored():
    import asyncio

    from core.consciousness.workspace_feed import feed_workspace

    class _Workspace:
        _tick = 0
        last_winner = None
        ignited = False

        async def submit(self, bid):
            pass

        async def run_competition(self):
            return None

    def focus(stability: float, stirred_above: bool) -> float:
        state = AuraState.default()
        state.identity.stability = stability
        state.cognition.modifiers = {"focus_mod": 1.0}
        state.affect.markers["arousal_usual"] = {"n": 50, "level": 0.6, "spread": 0.05}
        state.affect.arousal = 0.8 if stirred_above else 0.6
        asyncio.run(feed_workspace(state, _Workspace()))
        return state.cognition.modifiers["focus_mod"]

    assert focus(1.0, True) == 1.0
    assert focus(0.85, False) == 1.0
    assert focus(0.85, True) > 1.0
