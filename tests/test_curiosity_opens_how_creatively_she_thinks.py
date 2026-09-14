"""Development moved what she felt and never how exploratory her thinking was.

The affect phase blends curiosity toward how unlike her ordinary life the moment
is, and the homeostatic coupling that sets how creatively the next thought may
run read engagement alone. So a moment of real novelty reached her curiosity and
stopped there: nothing developmental reached attention's modifiers. Creativity
now follows whichever is the stronger pull, engagement or curiosity, with the
same coefficient and range as before.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.consciousness.homeostatic_coupling import HomeostaticCoupling

DRIVES = {"energy": 0.8, "curiosity": 0.7, "persistence": 0.7}
SETTLED = {"valence": 0.2, "arousal": 0.5, "engagement": 0.4}


def _coupling() -> HomeostaticCoupling:
    return HomeostaticCoupling(orchestrator=None)


def test_curiosity_stronger_than_engagement_opens_creativity() -> None:
    coupling = _coupling()
    plain = coupling._compute_modifiers(DRIVES, SETTLED, 1.0)
    curious = coupling._compute_modifiers(DRIVES, {**SETTLED, "curiosity": 0.9}, 1.0)
    assert curious.creativity_mod > plain.creativity_mod
    assert curious.creativity_mod == pytest.approx(0.6 + 0.9 * 0.8)


def test_weaker_curiosity_leaves_creativity_where_engagement_puts_it() -> None:
    coupling = _coupling()
    modifiers = coupling._compute_modifiers(DRIVES, {**SETTLED, "curiosity": 0.1}, 1.0)
    assert modifiers.creativity_mod == pytest.approx(0.6 + 0.4 * 0.8)


def test_an_affect_reading_without_curiosity_is_the_old_formula() -> None:
    modifiers = _coupling()._compute_modifiers(DRIVES, SETTLED, 1.0)
    assert modifiers.creativity_mod == pytest.approx(0.6 + SETTLED["engagement"] * 0.8)


def test_the_settled_affect_carries_curiosity(monkeypatch) -> None:
    import core.container as container

    affect = SimpleNamespace(valence=0.1, arousal=0.4, engagement=0.3, curiosity=0.85)
    repository = SimpleNamespace(_current=SimpleNamespace(affect=affect))
    monkeypatch.setattr(
        container.ServiceContainer,
        "get",
        staticmethod(lambda name, default=None: repository if name == "state_repository" else default),
    )
    assert HomeostaticCoupling._settled_affect()["curiosity"] == pytest.approx(0.85)
