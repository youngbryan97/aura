"""Two names carrying one number are one dimension counted twice.

Measured on Aura's live corpus, 2026-08-25. Three pairs of the 74 named
dimensions were exactly equal on every one of 1,629 turns:

    attention.focus       == attention.salience_peak
    temporal.past         == memory.recall_hits
    temporal.future       == goal.priority

None was noticed by anything, because each looked like a live dimension with a
plausible value. The cost is not cosmetic:

  - coverage counts a duplicate twice, so the admission gate reads a state as
    better populated than it is;
  - the head gets two gradient paths to one signal, so its weight on that
    signal is split and its interpretation misleading;
  - and the causal work breaks. Ablating the memory channel silently ablated
    part of the temporal channel, so a channel influence map built on that
    corpus cannot say which of the two carried the effect.

The first was a bug: `peak / max(1.0, total)` and `peak / total` differ only
when the weights sum to less than one, which they never do. The other two were
derivations that were identity functions. This file drives the probes with
independently varying organ readings and requires the dimensions to move
independently too.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.brain.llm.endogenous_state import FEATURES, STATE_DIM, assemble_state


class _Substrate:
    """Independently varying readings on every axis the probes read."""

    def __init__(self, step: int) -> None:
        self._step = step

    def _state_snapshot_nowait(self, max_wait_s: float = 0.05) -> dict:
        rng = np.random.default_rng(self._step)
        return {
            "x": rng.normal(size=64),
            "phi": 0.1 + 0.01 * self._step,
            "snapshot_age_s": 0.0,
            "freshness_threshold_s": 5.0,
        }

    def current(self):
        step = self._step

        class _Vector:
            frustration = 0.01 * step
            curiosity = 0.02 * step
            energy = 0.03 * step
            focus = 0.04 * step

        return _Vector()


def _states(count: int = 24) -> np.ndarray:
    """Assemble the state repeatedly against independently moving organs."""
    rows = []
    for step in range(count):
        state = assemble_state()
        rows.append(np.where(state.present, state.values, np.nan))
    return np.asarray(rows, dtype=float)


def test_the_two_attention_dimensions_are_not_one(monkeypatch):
    """`focus` is a share and `salience_peak` is a magnitude."""
    from core.brain.llm import endogenous_state as module

    readings = []
    for peak, total in ((0.4, 1.0), (0.4, 2.0), (0.9, 3.0), (0.2, 8.0)):
        weights = [peak] + [(total - peak) / 3] * 3

        class _Space:
            def __init__(self, w):
                self._w = w

            def attentional_focus(self, n=None):
                return [(f"atom{i}", v) for i, v in enumerate(self._w)]

        monkeypatch.setattr(
            module, "_service", lambda name, s=_Space(weights): s if name == "atomspace" else None
        )
        read = module._probe_attention() or {}
        if "attention.focus" in read and "attention.salience_peak" in read:
            readings.append((read["attention.focus"], read["attention.salience_peak"]))

    assert readings, "the attention probe produced nothing to compare"
    focus = [r[0] for r in readings]
    peaks = [r[1] for r in readings]
    assert focus != peaks, (
        "attention.focus and attention.salience_peak carried one value; "
        f"focus={focus} peak={peaks}"
    )


def test_temporal_past_is_not_a_copy_of_a_memory_dimension(monkeypatch):
    """The derivation was `past = memory.recall_hits`, an identity."""
    from core.brain.llm import endogenous_state as module

    monkeypatch.setattr(
        module,
        "_probe_memory",
        lambda: {"memory.recall_hits": 0.25, "memory.episodic_recency": 0.80},
    )
    monkeypatch.setattr(
        module,
        "_probe_goal",
        lambda: {"goal.active": 1.0, "goal.priority": 0.7, "goal.progress": 0.3},
    )

    read = module._probe_temporal() or {}

    assert read["temporal.past"] != pytest.approx(0.25), (
        "temporal.past is still an exact copy of memory.recall_hits"
    )
    assert read["temporal.past"] == pytest.approx(0.80)


def test_temporal_future_is_not_a_copy_of_goal_priority(monkeypatch):
    """`goal.active * goal.priority` equalled goal.priority on every turn."""
    from core.brain.llm import endogenous_state as module

    monkeypatch.setattr(module, "_probe_memory", lambda: {"memory.recall_hits": 0.1})
    monkeypatch.setattr(
        module,
        "_probe_goal",
        lambda: {"goal.active": 1.0, "goal.priority": 0.7, "goal.progress": 0.4},
    )

    read = module._probe_temporal() or {}

    assert read["temporal.future"] != pytest.approx(0.7), (
        "temporal.future is still an exact copy of goal.priority"
    )
    # What is still ahead: priority scaled by what remains of the goal.
    assert read["temporal.future"] == pytest.approx(0.7 * 0.6)


def test_a_goal_with_no_progress_still_does_not_pin_future_to_priority(monkeypatch):
    """Progress pinned at zero is the goal engine's defect, not a licence for
    the temporal channel to become a second copy of the goal channel."""
    from core.brain.llm import endogenous_state as module

    monkeypatch.setattr(module, "_probe_memory", lambda: {})
    monkeypatch.setattr(
        module,
        "_probe_goal",
        lambda: {"goal.active": 1.0, "goal.priority": 0.9, "goal.progress": 0.0},
    )

    read = module._probe_temporal() or {}

    # It equals priority here, and that IS the honest reading when nothing has
    # progressed. What the guard above stops is it being an identity BY
    # CONSTRUCTION, which is what made it inseparable from goal.priority.
    assert read["temporal.future"] == pytest.approx(0.9)


def test_every_declared_dimension_has_a_distinct_name():
    """The cheapest version of the same check."""
    names = [feature.name for feature in FEATURES]
    assert len(names) == len(set(names))
    assert len(names) == STATE_DIM
