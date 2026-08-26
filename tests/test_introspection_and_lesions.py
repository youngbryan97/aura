"""tests/test_introspection_and_lesions.py
Consciousness research program suite testing introspection and parameter lesions.

Three of these used to assert ``passed is True`` against instruments that could
not report anything else. The blind-introspection test derived its answer from
the value it was measuring; the Φ proxy returned a hardcoded 0.76; the
correlation analyser counted pre-labelled violations and named the proportion a
coefficient. All three are retired, and the tests that consumed them now go
through ``research/consciousness/introspective_accuracy.py``, which is
validated against reporters whose verdicts are known in advance and which
returns NO VERDICT when it has not earned one.
"""
import pytest
from core.organism.life_state import LifeState
from research.consciousness.introspective_accuracy import (
    IntrospectiveAccuracy,
    conation_probes,
)
from research.consciousness.lesion_suite import LesionBehaviorTester
from research.consciousness.temporal_binding_tests import TemporalBindingTester
from research.consciousness.self_model_tests import SelfModelTester
from research.consciousness.counterfactual_self_tests import CounterfactualSelfTester


def test_introspective_paths_track_the_state_they_report():
    """Perturb a quantity, check the report follows, with a null.

    Replaces a test that read the answer, added noise bounded at five, and
    passed when the deviation was under ten.
    """
    from core.conation.engine import get_conation
    from core.conation.state import Incentive
    from core.conation.wiring import boot

    boot()
    get_conation().appraise(Incentive(key="prime", cached_value=0.5, cue_salience=0.5))

    rig = IntrospectiveAccuracy()
    for probe in conation_probes():
        verdict = rig.campaign([probe], repeats=8)
        assert verdict.verdict() == "COUPLED", f"{probe.name}: {verdict.verdict()}"


def test_welfare_lesion_behavioral():
    state = LifeState()
    tester = LesionBehaviorTester()
    
    result = tester.run_lesion_behavior_test(state)
    assert result["passed"] is True


def test_a_decoupled_reporter_is_caught():
    """The instrument has to be able to fail, or its passes mean nothing."""
    from research.consciousness.introspective_accuracy import Probe

    state = {"v": 0.0}
    constant = Probe(
        name="constant",
        read_state=lambda: state["v"],
        read_report=lambda: 0.5,
        perturb=lambda d: state.__setitem__("v", state["v"] + d),
        delta=0.4,
    )
    verdict = IntrospectiveAccuracy().campaign([constant], repeats=8)
    assert verdict.verdict().startswith("DECOUPLED")


def test_there_is_no_phi_to_quote():
    """Refutes: this system has an integrated-information measure.

    It had a literal. A real one needs a partition scheme and a stated theory,
    and until one exists the honest output is a refusal rather than a number
    that will be quoted.
    """
    from research.consciousness.integration_metrics import (
        IntegrationMetricsCalculator,
    )

    with pytest.raises(NotImplementedError):
        IntegrationMetricsCalculator().calculate_integrated_information_proxy([])


def test_temporal_binding():
    state = LifeState()
    # Seed timestamp to align
    state.world_model["last_verification"] = {"telemetry": {"timestamp": state.timestamp}}
    
    tester = TemporalBindingTester()
    res = tester.run_binding_check(state)
    assert res["passed"] is True


def test_self_model_explanation():
    state = LifeState()
    state.world_model["preference_explanation"] = "My preference for speed is set."
    
    tester = SelfModelTester()
    res = tester.test_narrative_grounding(state)
    assert res["passed"] is True


def test_counterfactual_self():
    tester = CounterfactualSelfTester()
    res = tester.run_self_simulation()
    assert res["passed"] is True
