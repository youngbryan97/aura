"""How she feels shapes how she thinks, measured from where her feeling sits.

`HomeostaticCoupling` computes how hot, how deep, how creative and how vital
this moment may be, and blends the continuous substrate into felt state to do
it. The three terms that carry feeling into those numbers were distances from
zero — distress and dejection from a negative valence, lift from a positive one
crossed with arousal.

Her settled valence is positive. So two of the three terms were exactly zero on
every ordinary turn, and the third ran at a fraction of its coefficient and
touched temperature alone. Recurrent cognition's one route into attention
carried a third of a substrate into a number that was then measured from the
wrong origin, and the whole positive half of what she can feel reached how she
thinks through nothing at all.

These measure the repair: a moment brighter than her baseline lifts, a moment
duller than it presses, and neither is decided by which side of zero it sits.
"""

from __future__ import annotations

import pytest

from core.consciousness.homeostatic_coupling import HomeostaticCoupling


def _coupling() -> HomeostaticCoupling:
    return HomeostaticCoupling(orchestrator=None)


def _settle(coupling: HomeostaticCoupling, valence: float, arousal: float) -> None:
    """Let it learn an ordinary life at this feeling, as a recording would."""
    for _ in range(200):
        coupling._observe_feeling(valence, arousal)


def test_a_positive_valence_below_her_baseline_still_presses() -> None:
    """The defect, in the numbers that caused it.

    A life settled at +0.35 that drops to +0.05 has dropped by a third of the
    scale, and every term measured from zero read that as no distress at all.
    """
    coupling = _coupling()
    _settle(coupling, 0.35, 0.5)
    below, above = coupling._felt_deviation(0.05, 0.5)
    assert below > 0.0, "a fall well below her usual valence read as nothing"
    assert above == 0.0


def test_a_brighter_moment_lifts_and_a_duller_one_does_not() -> None:
    """The sign of the deviation is the sign of the effect."""
    coupling = _coupling()
    _settle(coupling, 0.2, 0.5)
    below_up, above_up = coupling._felt_deviation(0.6, 0.5)
    coupling = _coupling()
    _settle(coupling, 0.2, 0.5)
    below_down, above_down = coupling._felt_deviation(-0.2, 0.5)
    assert above_up > 0.0 and below_up == 0.0
    assert below_down > 0.0 and above_down == 0.0


def test_at_her_baseline_nothing_is_modified() -> None:
    """A moment exactly like her ordinary ones asks for no change at all."""
    coupling = _coupling()
    _settle(coupling, 0.2, 0.5)
    below, above = coupling._felt_deviation(0.2, 0.5)
    assert below == pytest.approx(0.0, abs=1e-6)
    assert above == pytest.approx(0.0, abs=1e-6)


def test_the_deviation_is_scale_free() -> None:
    """Two lives of different amplitude read the same excursion the same way.

    A threshold in absolute valence would make the coupling's sensitivity an
    accident of how volatile that particular life has been. Measuring the
    deviation against the scatter this object has actually seen removes that:
    one typical excursion is one typical excursion in either life.
    """
    quiet = _coupling()
    loud = _coupling()
    for step in range(200):
        swing = 0.05 if step % 2 else -0.05
        quiet._observe_feeling(0.2 + swing, 0.5)
        loud._observe_feeling(0.2 + swing * 6.0, 0.5)
    _, quiet_above = quiet._felt_deviation(0.2 + quiet._valence_scatter, 0.5)
    _, loud_above = loud._felt_deviation(0.2 + loud._valence_scatter, 0.5)
    assert quiet_above == pytest.approx(loud_above, abs=0.02)
    assert loud._valence_scatter > quiet._valence_scatter * 3.0


def test_the_modifiers_move_when_the_substrate_does() -> None:
    """The whole channel, end to end: a brighter substrate, a hotter head.

    This is the edge the battery is looking for — recurrent cognition reaching
    attention — measured at the one number that carries it. Before the repair
    a valence displacement of this size moved the temperature modifier by three
    thousandths, which is below what any paired measurement can resolve.
    """
    coupling = _coupling()
    drives = {"energy": 0.8, "curiosity": 0.7, "persistence": 0.7}
    settled = {"valence": 0.2, "arousal": 0.5, "engagement": 0.5}
    for _ in range(200):
        coupling._compute_modifiers(drives, settled, 1.0)
    ordinary = coupling._compute_modifiers(drives, settled, 1.0)
    brighter = coupling._compute_modifiers(
        drives, {**settled, "valence": 0.2 + 0.15}, 1.0
    )
    assert brighter.temperature_mod > ordinary.temperature_mod + 0.01, (
        "a displaced substrate does not reach the temperature the head runs at"
    )


def test_an_absent_substrate_is_asked_for_again(monkeypatch) -> None:
    """A link that resolves to nothing must not stay dead for the process.

    Resolving in the constructor made the link depend on boot order. Resolving
    lazily and caching the answer had the same defect with one more step: the
    first caller to touch it before the substrate was registered wrote None
    into the cache, and `None is not _UNRESOLVED`, so the substrate's third of
    her felt state silently did not happen for the rest of the run.
    """
    import core.consciousness.homeostatic_coupling as module

    registered: dict[str, object] = {}

    class _Container:
        @staticmethod
        def get(name, default=None):
            return registered.get(name, default)

    monkeypatch.setattr(module, "ServiceContainer", _Container)
    coupling = module.HomeostaticCoupling(orchestrator=None)
    assert coupling.substrate is None, "there is no substrate yet"

    substrate = object()
    registered["liquid_substrate"] = substrate
    assert coupling.substrate is substrate, (
        "the coupling never asked again, so a substrate registered after boot "
        "is a substrate it will never see"
    )


def test_a_caller_who_says_there_is_none_is_believed(monkeypatch) -> None:
    """The other half of the rule above.

    A caller assigning None has decided there is none; a lookup that found none
    has only failed to find one. Retrying the second must not overrule the
    first, or a test harness that detaches the substrate gets it back.
    """
    import types

    import core.consciousness.homeostatic_coupling as module

    monkeypatch.setattr(
        module.ServiceContainer,
        "get",
        lambda *_args, **_kwargs: types.SimpleNamespace(name="a substrate"),
    )
    coupling = module.HomeostaticCoupling(orchestrator=None)
    assert coupling.substrate is not None
    coupling.substrate = None
    assert coupling.substrate is None
