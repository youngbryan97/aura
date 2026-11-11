"""A surprise of her body's that keeps coming is felt less each time it comes.

LIVE 10 November 2026, 05:05 to 05:18 UTC: a test run beside her made the machine's load jitter. Each tick a body
channel's acceleration passed its line was a "novelty" at full size, 50 of them, and 29 CPU spikes were prediction
errors, while the earlier ones still moved her chemistry. Her norepinephrine reached 1.0, and through it her cortisol
the crisis line ("events still moving her cortisol: novelty 50x (15.00), prediction_error 29x (6.81)").
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.consciousness import embodied_interoception as module
from core.consciousness.neurochemical_system import (
    NeurochemicalSystem,
    _how_long_an_event_moves_cortisol,
)


@pytest.fixture
def clock(monkeypatch):
    now = [1_000.0]
    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: now[0], time=lambda: now[0]))
    return now


def _her_body() -> module.EmbodiedInteroception:
    body = module.EmbodiedInteroception()
    body._neurochemical_ref = NeurochemicalSystem()
    return body


def test_the_nth_surprise_of_a_kind_is_felt_as_a_1_over_n_share(clock):
    body = _her_body()
    assert [round(body._habituated("novelty", 0.3), 4) for _ in range(4)] == [0.3, 0.15, 0.1, 0.075]
    assert body._habituated("prediction_error", 0.4) == pytest.approx(0.4), "each kind habituates on its own"


def test_one_that_has_not_come_for_as_long_as_it_moves_her_is_felt_whole_again(clock):
    body = _her_body()
    for _ in range(5):
        body._habituated("novelty", 0.3)
    clock[0] += _how_long_an_event_moves_cortisol(body._neurochemical_ref) + 1.0
    assert body._habituated("novelty", 0.3) == pytest.approx(0.3)


def test_without_a_chemistry_to_be_moved_nothing_habituates(clock):
    body = module.EmbodiedInteroception()
    body._neurochemical_ref = SimpleNamespace(chemicals={})
    assert [body._habituated("novelty", 0.3) for _ in range(3)] == [0.3, 0.3, 0.3]


def test_her_body_hands_her_chemistry_the_habituated_novelty(clock):
    felt: list[float] = []

    class Recorded(NeurochemicalSystem):
        def on_novelty(self, amount: float = 0.3):
            felt.append(amount)
            return super().on_novelty(amount)

    body = module.EmbodiedInteroception()
    body._neurochemical_ref = Recorded()
    for _ in range(3):
        body.channels["metabolic_load"].acceleration = 0.5  # the jitter of a load that keeps changing pace
        body._trigger_neurochemical_events()
        clock[0] += 7.5  # one tick apart, as on 10 November: 50 in the 375 s an event moves her cortisol
    assert [round(a, 4) for a in felt] == [0.3, 0.15, 0.1]


def test_fifty_in_a_row_move_her_as_about_five_used_to(clock):
    body = _her_body()
    total = sum(body._habituated("novelty", 0.3) for _ in range(50))
    assert total < 0.3 * 5 < 0.3 * 50
