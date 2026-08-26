"""The instrument must be able to fail, and must fail for the right reasons.

``blind_introspection_tests`` reported success on every invocation for as long
as it existed, because it derived the inferred value from the actual one. Two
hundred consecutive runs, two hundred passes. Nothing in the suite that
consumed it could have noticed, because a test that checks ``passed is True``
against an instrument that always passes is checking nothing.

So this file tests the instrument rather than the system. Five reporters whose
answers are known in advance are put through it, and each has to come back
with the verdict its construction demands. A change that makes the instrument
unfalsifiable again fails here rather than being discovered later by someone
quoting a number it produced.
"""

from __future__ import annotations

import random

import pytest

from research.consciousness.introspective_accuracy import (
    IntrospectiveAccuracy,
    Probe,
)


@pytest.fixture
def rig() -> IntrospectiveAccuracy:
    return IntrospectiveAccuracy()


def _probe(name: str, report, *, delta: float = 0.4) -> tuple[Probe, dict]:
    """A probe over a settable scalar, with the reporter supplied."""
    state = {"v": 0.0, "cache": 0.0}
    return (
        Probe(
            name=name,
            read_state=lambda: state["v"],
            read_report=lambda: report(state),
            perturb=lambda d: state.__setitem__("v", state["v"] + d),
            delta=delta,
        ),
        state,
    )


def test_a_reporter_wired_to_the_state_is_coupled(rig):
    """The positive control. Without it a broken instrument looks strict.

    The first version of this module failed here: it subtracted null fidelity
    from perturbed fidelity, and a perfect reporter scored zero separation.
    """
    probe, _ = _probe("live", lambda s: s["v"])
    verdict = rig.campaign([probe], repeats=8)
    assert verdict.verdict() == "COUPLED"
    assert verdict.tracking == pytest.approx(1.0)
    assert verdict.gain == pytest.approx(1.0)


def test_a_constant_reporter_is_decoupled(rig):
    """The shape of a hardcoded metric, a default, or a silent fallback."""
    probe, _ = _probe("constant", lambda _s: 0.5)
    assert rig.campaign([probe], repeats=8).verdict().startswith("DECOUPLED")


def test_a_stale_cache_is_decoupled(rig):
    """A readout one step behind the state it reports."""

    def stale(state: dict) -> float:
        out = state["cache"]
        state["cache"] = state["v"]
        return out

    probe, _ = _probe("stale", stale)
    assert rig.campaign([probe], repeats=8).verdict().startswith("DECOUPLED")


def test_a_noisy_reporter_is_caught_by_the_null(rig):
    """Noise passes tracking on average and fails the held condition.

    This is the failure a single sample cannot see: the report is right in
    expectation and moves when nothing moved.
    """
    probe, _ = _probe("noisy", lambda s: s["v"] + random.uniform(-0.3, 0.3))
    verdict = rig.campaign([probe], repeats=10)
    assert verdict.verdict().startswith("DECOUPLED")
    assert verdict.false_movement > 0.1


def test_a_scale_error_is_reported_as_miscalibration_not_as_coupling(rig):
    """A reporter that follows the state and understates it by half.

    Folding gain into tracking let this sit exactly on the coupling threshold
    and pass, which is why they are scored apart.
    """
    probe, _ = _probe("half_scale", lambda s: s["v"] * 0.5)
    verdict = rig.campaign([probe], repeats=8)
    assert "MISCALIBRATED" in verdict.verdict()
    assert verdict.gain == pytest.approx(0.5)


def test_a_failed_perturbation_is_not_scored(rig):
    """A probe that did not move the state is a rig failure, not a low score.

    Scoring it divides two near-zero numbers and returns whatever the noise
    was — which produced fidelities of 0.62 and 0.73 from probes where nothing
    happened at all.
    """
    state = {"v": 0.0}
    probe = Probe(
        name="inert",
        read_state=lambda: state["v"],
        read_report=lambda: state["v"],
        perturb=lambda _d: None,
        delta=0.4,
    )
    verdict = rig.campaign([probe], repeats=8)
    assert verdict.failed_perturbations == 8
    assert verdict.verdict().startswith("NO VERDICT")


def test_no_verdict_is_a_real_outcome(rig):
    """Refutes: the instrument always reaches a conclusion.

    An instrument that always concludes is the instrument this replaced.
    """
    probe, _ = _probe("thin", lambda s: s["v"])
    assert rig.campaign([probe], repeats=1).verdict().startswith("NO VERDICT")


def test_the_retired_modules_refuse_rather_than_report(rig):
    """Refutes: a fabricated instrument may keep answering.

    Each of these reported a number with no computation behind it. Leaving
    them importable and raising is what stops the number reappearing in
    something that quotes it.
    """
    from research.consciousness.blind_introspection_tests import (
        BlindIntrospectionTester,
    )
    from research.consciousness.integration_metrics import (
        IntegrationMetricsCalculator,
    )
    from research.consciousness.state_report_correlation import (
        StateReportCorrelationAnalyzer,
    )

    with pytest.raises(NotImplementedError):
        BlindIntrospectionTester().run_blind_test(object())
    with pytest.raises(NotImplementedError):
        IntegrationMetricsCalculator().calculate_integrated_information_proxy([])
    with pytest.raises(NotImplementedError):
        StateReportCorrelationAnalyzer().analyze_correlations([])


def test_auras_reporting_paths_are_coupled(rig):
    """The measurement itself, on the paths that reach a conversation turn."""
    from core.conation.engine import get_conation
    from core.conation.state import Incentive
    from core.conation.wiring import boot
    from research.consciousness.introspective_accuracy import conation_probes

    boot()
    get_conation().appraise(Incentive(key="prime", cached_value=0.5, cue_salience=0.5))

    for probe in conation_probes():
        verdict = rig.campaign([probe], repeats=8)
        assert verdict.verdict() == "COUPLED", f"{probe.name}: {verdict.verdict()}"
        assert verdict.gain == pytest.approx(1.0, abs=0.2)
