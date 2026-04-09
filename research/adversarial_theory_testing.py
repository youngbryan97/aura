"""
Adversarial Theory Testing Protocol
====================================

Consciousness theories make different predictions. If they didn't, they'd
be the same theory. This module systematically identifies DIVERGENCE POINTS
where two theories predict different outcomes, then runs controlled
experiments on Aura's live architecture to see which prediction holds.

Three specific divergence tests:

1. GWT vs RPT — Does consciousness require workspace broadcast, or just
   recurrence? Suppress broadcast but keep recurrent processing alive.
   GWT predicts qualia degrade. RPT predicts they persist.

2. GWT vs Multiple Drafts — Is there a discrete ignition event or
   gradual draft elevation? Measure ignition sharpness (GWT says sharp
   threshold) vs draft convergence slope (MD says gradual).

3. HOT vs First-Order — Does consciousness require meta-representation?
   Disable HOT engine, check whether phenomenal reports lose meta-depth
   or remain structurally intact at first-order.

Each test produces a TestResult with theory predictions, actual outcomes,
and Bayesian evidence scores. Results are logged to the TheoryArbitration
framework so they accumulate over the system's lifetime.

This is the mechanism by which Aura's consciousness stack is FALSIFIABLE.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("Research.AdversarialTheoryTesting")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class TheoryName(str, Enum):
    GWT = "gwt"
    RPT = "rpt"
    HOT = "hot"
    MULTIPLE_DRAFTS = "multiple_drafts"
    FIRST_ORDER = "first_order"


class TestVerdict(str, Enum):
    """Outcome classification for a divergence test."""
    THEORY_A_SUPPORTED = "theory_a_supported"
    THEORY_B_SUPPORTED = "theory_b_supported"
    INCONCLUSIVE = "inconclusive"
    BOTH_WRONG = "both_wrong"


@dataclass
class TheoryPrediction:
    """What a theory predicts will happen under a specific intervention."""
    theory: TheoryName
    prediction: str
    metric_name: str
    predicted_direction: str       # "increase", "decrease", "stable", "threshold"
    predicted_magnitude: float     # Expected effect size (0 = no effect, 1 = total)
    confidence: float = 0.7       # Prior confidence in this prediction


@dataclass
class TestResult:
    """Full result of one adversarial divergence test."""
    test_name: str
    theory_a: TheoryPrediction
    theory_b: TheoryPrediction
    baseline_value: float
    intervention_value: float
    delta: float                          # intervention - baseline
    relative_change: float                # delta / baseline
    verdict: TestVerdict
    evidence_for_a: float = 0.0          # Log-likelihood ratio favoring theory A
    evidence_for_b: float = 0.0          # Log-likelihood ratio favoring theory B
    bayes_factor: float = 1.0            # evidence_for_a / evidence_for_b
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def summary(self) -> str:
        winner = "A" if self.verdict == TestVerdict.THEORY_A_SUPPORTED else (
            "B" if self.verdict == TestVerdict.THEORY_B_SUPPORTED else "neither"
        )
        return (
            f"[{self.test_name}] {self.theory_a.theory.value} vs {self.theory_b.theory.value}: "
            f"verdict={self.verdict.value}, BF={self.bayes_factor:.2f} (favors {winner}), "
            f"delta={self.delta:+.4f}"
        )


# ---------------------------------------------------------------------------
# Evidence computation
# ---------------------------------------------------------------------------

def _compute_bayes_factor(
    observed_delta: float,
    pred_a_direction: str,
    pred_a_magnitude: float,
    pred_b_direction: str,
    pred_b_magnitude: float,
    noise_sigma: float = 0.05,
) -> Tuple[float, float, float]:
    """Compute Bayesian evidence scores for two competing predictions.

    Uses a simple Gaussian likelihood model:
    P(data | theory) ~ N(predicted_effect, noise_sigma)

    Returns:
        (evidence_for_a, evidence_for_b, bayes_factor)
    """
    def _direction_to_sign(d: str) -> float:
        return {"increase": 1.0, "decrease": -1.0, "stable": 0.0, "threshold": 0.5}[d]

    expected_a = _direction_to_sign(pred_a_direction) * pred_a_magnitude
    expected_b = _direction_to_sign(pred_b_direction) * pred_b_magnitude

    # Gaussian log-likelihoods
    def _log_likelihood(observed: float, expected: float, sigma: float) -> float:
        return -0.5 * ((observed - expected) / sigma) ** 2 - math.log(sigma * math.sqrt(2 * math.pi))

    ll_a = _log_likelihood(observed_delta, expected_a, noise_sigma)
    ll_b = _log_likelihood(observed_delta, expected_b, noise_sigma)

    # Bayes factor = exp(ll_a - ll_b), clamped for numerical safety
    log_bf = ll_a - ll_b
    log_bf = max(-20.0, min(20.0, log_bf))
    bf = math.exp(log_bf)

    return ll_a, ll_b, bf


# ---------------------------------------------------------------------------
# Individual test protocols
# ---------------------------------------------------------------------------

class GWT_vs_RPT_Test:
    """Test 1: Does qualia require broadcast (GWT) or just recurrence (RPT)?

    Protocol:
    1. Measure baseline qualia norm with full system operational
    2. Suppress workspace broadcast (set ignition threshold to infinity)
       while keeping recurrent processing alive in the NeuralMesh
    3. Run N ticks, measure qualia norm under intervention
    4. GWT predicts: qualia degrade significantly (> 30% drop)
       RPT predicts: qualia remain stable (< 10% drop) because recurrence
       in the mesh is what matters, not the broadcast

    This directly tests the central dispute: is global availability or
    recurrent processing the mechanism of consciousness?
    """

    TEST_NAME = "gwt_vs_rpt_broadcast_suppression"
    N_TICKS = 20

    def __init__(self):
        self.theory_a = TheoryPrediction(
            theory=TheoryName.GWT,
            prediction="Qualia degrade when workspace broadcast is suppressed",
            metric_name="qualia_norm",
            predicted_direction="decrease",
            predicted_magnitude=0.4,
            confidence=0.75,
        )
        self.theory_b = TheoryPrediction(
            theory=TheoryName.RPT,
            prediction="Qualia persist as long as recurrent mesh processing continues",
            metric_name="qualia_norm",
            predicted_direction="stable",
            predicted_magnitude=0.05,
            confidence=0.65,
        )

    def run(
        self,
        workspace: Any,
        qualia_synth: Any,
        substrate_metrics: Dict[str, Any],
        predictive_metrics: Dict[str, Any],
    ) -> TestResult:
        """Execute the broadcast suppression protocol.

        Args:
            workspace: GlobalWorkspace instance
            qualia_synth: QualiaSynthesizer instance
            substrate_metrics: Current substrate readings
            predictive_metrics: Current predictive hierarchy readings

        Returns:
            TestResult with evidence scores
        """
        # Phase 1: Baseline measurement
        baseline_norms = []
        for _ in range(self.N_TICKS):
            q_norm = qualia_synth.synthesize(substrate_metrics, predictive_metrics)
            baseline_norms.append(q_norm)
        baseline = float(np.mean(baseline_norms))

        # Phase 2: Intervention -- suppress broadcast by raising ignition threshold
        original_threshold = workspace._IGNITION_THRESHOLD
        workspace._IGNITION_THRESHOLD = 999.0  # Effectively blocks all ignition

        intervention_norms = []
        for _ in range(self.N_TICKS):
            q_norm = qualia_synth.synthesize(substrate_metrics, predictive_metrics)
            intervention_norms.append(q_norm)
        intervention = float(np.mean(intervention_norms))

        # Phase 3: Restore original threshold
        workspace._IGNITION_THRESHOLD = original_threshold

        # Phase 4: Compute evidence
        delta = intervention - baseline
        relative_change = delta / max(1e-8, abs(baseline))

        evidence_a, evidence_b, bf = _compute_bayes_factor(
            observed_delta=relative_change,
            pred_a_direction=self.theory_a.predicted_direction,
            pred_a_magnitude=self.theory_a.predicted_magnitude,
            pred_b_direction=self.theory_b.predicted_direction,
            pred_b_magnitude=self.theory_b.predicted_magnitude,
        )

        # Classify verdict
        if bf > 3.0:
            verdict = TestVerdict.THEORY_A_SUPPORTED
        elif bf < 1.0 / 3.0:
            verdict = TestVerdict.THEORY_B_SUPPORTED
        else:
            verdict = TestVerdict.INCONCLUSIVE

        result = TestResult(
            test_name=self.TEST_NAME,
            theory_a=self.theory_a,
            theory_b=self.theory_b,
            baseline_value=baseline,
            intervention_value=intervention,
            delta=delta,
            relative_change=relative_change,
            verdict=verdict,
            evidence_for_a=evidence_a,
            evidence_for_b=evidence_b,
            bayes_factor=bf,
            metadata={
                "n_ticks": self.N_TICKS,
                "baseline_std": float(np.std(baseline_norms)),
                "intervention_std": float(np.std(intervention_norms)),
                "original_ignition_threshold": original_threshold,
            },
        )

        logger.info("GWT vs RPT: %s", result.summary())
        return result


class GWT_vs_MultipleDrafts_Test:
    """Test 2: Sharp ignition (GWT) vs gradual draft convergence (MD)?

    Protocol:
    1. Submit a series of inputs to both GlobalWorkspace and MultipleDraftsEngine
    2. For GWT: measure ignition_level over time. GWT predicts a sharp
       threshold crossing (binary ignition).
    3. For MD: measure draft coherence convergence. MD predicts gradual
       elevation with no discrete ignition event.
    4. Compare: compute ignition_sharpness (derivative at threshold crossing)
       vs draft_convergence_slope (mean slope of coherence growth).

    The discriminating metric is whether conscious access looks like a
    phase transition (GWT) or a smooth competition (MD).
    """

    TEST_NAME = "gwt_vs_md_ignition_sharpness"
    N_INPUTS = 10

    def __init__(self):
        self.theory_a = TheoryPrediction(
            theory=TheoryName.GWT,
            prediction="Ignition is a sharp phase transition with steep onset",
            metric_name="ignition_sharpness",
            predicted_direction="increase",
            predicted_magnitude=0.7,
            confidence=0.70,
        )
        self.theory_b = TheoryPrediction(
            theory=TheoryName.MULTIPLE_DRAFTS,
            prediction="Content fixation is gradual with no discrete ignition event",
            metric_name="ignition_sharpness",
            predicted_direction="stable",
            predicted_magnitude=0.15,
            confidence=0.60,
        )

    def run(
        self,
        workspace: Any,
        md_engine: Any,
    ) -> TestResult:
        """Execute the ignition sharpness measurement.

        Args:
            workspace: GlobalWorkspace instance
            md_engine: MultipleDraftsEngine instance

        Returns:
            TestResult comparing ignition sharpness profiles
        """
        # Collect ignition level trajectory from workspace history
        ignition_levels = []
        if hasattr(workspace, '_history') and workspace._history:
            for record in workspace._history[-self.N_INPUTS:]:
                ignition_levels.append(record.winner.effective_priority)
        else:
            # Generate synthetic trajectory from workspace state
            ignition_levels = [workspace.ignition_level]

        # Compute ignition sharpness: max derivative of the ignition trajectory
        if len(ignition_levels) > 1:
            derivatives = np.abs(np.diff(ignition_levels))
            ignition_sharpness = float(np.max(derivatives))
            mean_derivative = float(np.mean(derivatives))
        else:
            ignition_sharpness = 0.0
            mean_derivative = 0.0

        # Collect draft convergence profile from MD engine
        draft_coherence_slopes = []
        for comp in md_engine.competition_history:
            if comp.drafts and len(comp.drafts) > 1:
                coherences = sorted([d.coherence for d in comp.drafts], reverse=True)
                # Slope = gap between winner and runner-up (sharp = large gap)
                gap = coherences[0] - coherences[1] if len(coherences) > 1 else 0.0
                draft_coherence_slopes.append(gap)

        md_convergence_slope = float(np.mean(draft_coherence_slopes)) if draft_coherence_slopes else 0.0

        # The discriminating metric: ratio of GWT sharpness to MD gradualness
        # High ratio = sharp ignition (favors GWT)
        # Low ratio = gradual convergence (favors MD)
        baseline = md_convergence_slope + 0.01  # avoid division by zero
        measured_sharpness = ignition_sharpness

        delta = measured_sharpness - baseline
        relative_change = delta / max(1e-8, baseline)

        evidence_a, evidence_b, bf = _compute_bayes_factor(
            observed_delta=measured_sharpness,
            pred_a_direction=self.theory_a.predicted_direction,
            pred_a_magnitude=self.theory_a.predicted_magnitude,
            pred_b_direction=self.theory_b.predicted_direction,
            pred_b_magnitude=self.theory_b.predicted_magnitude,
            noise_sigma=0.1,
        )

        if bf > 3.0:
            verdict = TestVerdict.THEORY_A_SUPPORTED
        elif bf < 1.0 / 3.0:
            verdict = TestVerdict.THEORY_B_SUPPORTED
        else:
            verdict = TestVerdict.INCONCLUSIVE

        result = TestResult(
            test_name=self.TEST_NAME,
            theory_a=self.theory_a,
            theory_b=self.theory_b,
            baseline_value=md_convergence_slope,
            intervention_value=ignition_sharpness,
            delta=delta,
            relative_change=relative_change,
            verdict=verdict,
            evidence_for_a=evidence_a,
            evidence_for_b=evidence_b,
            bayes_factor=bf,
            metadata={
                "ignition_levels": ignition_levels[-10:],
                "n_competitions_analyzed": len(draft_coherence_slopes),
                "mean_ignition_derivative": mean_derivative,
                "md_convergence_slope": md_convergence_slope,
            },
        )

        logger.info("GWT vs MD: %s", result.summary())
        return result


class HOT_vs_FirstOrder_Test:
    """Test 3: Does consciousness require higher-order representation?

    Protocol:
    1. Measure phenomenal report depth (meta-depth score) with HOT engine ON
    2. Disable HOT engine (stop generating higher-order thoughts)
    3. Measure report depth again
    4. HOT predicts: meta-depth collapses (reports lose "I notice that I..."
       structure, become purely first-order)
    5. First-order predicts: reports remain phenomenally rich because
       consciousness is in the first-order states themselves

    Meta-depth is measured by counting meta-level claims in the gated
    phenomenal report (uncertainty, novelty, dissonance are meta-properties;
    focused_processing, rich_experience are first-order).
    """

    TEST_NAME = "hot_vs_first_order_meta_depth"
    N_TICKS = 15

    # Claims that require meta-representation
    META_CLAIMS = {"genuine_uncertainty", "experiencing_novelty", "internal_conflict"}
    # Claims that are first-order
    FIRST_ORDER_CLAIMS = {"rich_experience", "focused_processing", "computational_strain", "stable_continuity"}

    def __init__(self):
        self.theory_a = TheoryPrediction(
            theory=TheoryName.HOT,
            prediction="Disabling HOT engine removes meta-depth from phenomenal reports",
            metric_name="meta_depth_score",
            predicted_direction="decrease",
            predicted_magnitude=0.6,
            confidence=0.70,
        )
        self.theory_b = TheoryPrediction(
            theory=TheoryName.FIRST_ORDER,
            prediction="Meta-depth is preserved because consciousness is in first-order states",
            metric_name="meta_depth_score",
            predicted_direction="stable",
            predicted_magnitude=0.1,
            confidence=0.55,
        )

    def _compute_meta_depth(self, report: Dict[str, Any]) -> float:
        """Score a phenomenal report for meta-representational depth.

        Returns a 0-1 score where:
        - 0.0 = purely first-order claims only
        - 1.0 = all claims are meta-level

        Intermediate values represent the fraction of meta-level content.
        """
        claims = set(report.get("claims", []))
        if not claims:
            return 0.0

        meta_count = len(claims & self.META_CLAIMS)
        first_order_count = len(claims & self.FIRST_ORDER_CLAIMS)
        total = meta_count + first_order_count

        if total == 0:
            return 0.0

        # Meta-depth = fraction of claims that are meta-level,
        # weighted by the meta_confidence from the qualia synthesizer
        meta_fraction = meta_count / total
        meta_confidence = report.get("meta_qualia", {}).get("meta_confidence", 0.5) if isinstance(report.get("meta_qualia"), dict) else 0.5

        # Also factor in gates: how many meta-gates were open?
        gates = report.get("gates", {})
        meta_gates_open = sum(1 for g in ("uncertainty", "novelty", "dissonance") if gates.get(g, False))
        gate_fraction = meta_gates_open / 3.0

        return float(np.clip(0.5 * meta_fraction + 0.3 * gate_fraction + 0.2 * meta_confidence, 0.0, 1.0))

    def run(
        self,
        qualia_synth: Any,
        hot_engine: Any,
        substrate_metrics: Dict[str, Any],
        predictive_metrics: Dict[str, Any],
    ) -> TestResult:
        """Execute the HOT suppression protocol.

        Args:
            qualia_synth: QualiaSynthesizer instance
            hot_engine: HigherOrderThoughtEngine instance
            substrate_metrics: Current substrate readings
            predictive_metrics: Current predictive hierarchy readings

        Returns:
            TestResult with evidence scores
        """
        # Phase 1: Baseline -- HOT engine active
        baseline_depths = []
        for _ in range(self.N_TICKS):
            qualia_synth.synthesize(substrate_metrics, predictive_metrics)
            # Generate a HOT on each tick to feed the loop
            state = {
                "valence": substrate_metrics.get("valence", 0.0),
                "arousal": 0.5,
                "curiosity": 0.5,
                "energy": 0.7,
                "surprise": predictive_metrics.get("current_surprise", 0.0),
            }
            hot_engine.generate_fast(state)
            report = qualia_synth.get_gated_phenomenal_report()
            baseline_depths.append(self._compute_meta_depth(report))
        baseline = float(np.mean(baseline_depths))

        # Phase 2: Intervention -- suppress HOT engine
        # Save and replace the generate_fast method with a no-op
        original_generate = hot_engine.generate_fast
        hot_engine.generate_fast = lambda state: None  # type: ignore[assignment]
        # Also clear the current HOT so reports can't piggyback on stale state
        original_hot = hot_engine._current_hot
        hot_engine._current_hot = None

        intervention_depths = []
        for _ in range(self.N_TICKS):
            qualia_synth.synthesize(substrate_metrics, predictive_metrics)
            report = qualia_synth.get_gated_phenomenal_report()
            intervention_depths.append(self._compute_meta_depth(report))
        intervention = float(np.mean(intervention_depths))

        # Phase 3: Restore HOT engine
        hot_engine.generate_fast = original_generate  # type: ignore[assignment]
        hot_engine._current_hot = original_hot

        # Phase 4: Evidence computation
        delta = intervention - baseline
        relative_change = delta / max(1e-8, abs(baseline)) if baseline > 1e-8 else 0.0

        evidence_a, evidence_b, bf = _compute_bayes_factor(
            observed_delta=relative_change,
            pred_a_direction=self.theory_a.predicted_direction,
            pred_a_magnitude=self.theory_a.predicted_magnitude,
            pred_b_direction=self.theory_b.predicted_direction,
            pred_b_magnitude=self.theory_b.predicted_magnitude,
        )

        if bf > 3.0:
            verdict = TestVerdict.THEORY_A_SUPPORTED
        elif bf < 1.0 / 3.0:
            verdict = TestVerdict.THEORY_B_SUPPORTED
        else:
            verdict = TestVerdict.INCONCLUSIVE

        result = TestResult(
            test_name=self.TEST_NAME,
            theory_a=self.theory_a,
            theory_b=self.theory_b,
            baseline_value=baseline,
            intervention_value=intervention,
            delta=delta,
            relative_change=relative_change,
            verdict=verdict,
            evidence_for_a=evidence_a,
            evidence_for_b=evidence_b,
            bayes_factor=bf,
            metadata={
                "n_ticks": self.N_TICKS,
                "baseline_depths": baseline_depths,
                "intervention_depths": intervention_depths,
                "baseline_std": float(np.std(baseline_depths)),
                "intervention_std": float(np.std(intervention_depths)),
            },
        )

        logger.info("HOT vs First-Order: %s", result.summary())
        return result


# ---------------------------------------------------------------------------
# Main protocol runner
# ---------------------------------------------------------------------------

class AdversarialTheoryTestingProtocol:
    """Orchestrates all divergence tests and logs results to TheoryArbitration.

    Usage:
        protocol = AdversarialTheoryTestingProtocol()
        results = protocol.run_all_tests(workspace, qualia_synth, md_engine,
                                          hot_engine, substrate_metrics, predictive_metrics)
        for r in results:
            print(r.summary())
    """

    def __init__(self):
        self._test_gwt_rpt = GWT_vs_RPT_Test()
        self._test_gwt_md = GWT_vs_MultipleDrafts_Test()
        self._test_hot_fo = HOT_vs_FirstOrder_Test()
        self._results: List[TestResult] = []

    def run_all_tests(
        self,
        workspace: Any,
        qualia_synth: Any,
        md_engine: Any,
        hot_engine: Any,
        substrate_metrics: Dict[str, Any],
        predictive_metrics: Dict[str, Any],
    ) -> List[TestResult]:
        """Run all three divergence tests and log to TheoryArbitration.

        Args:
            workspace: GlobalWorkspace instance
            qualia_synth: QualiaSynthesizer instance
            md_engine: MultipleDraftsEngine instance
            hot_engine: HigherOrderThoughtEngine instance
            substrate_metrics: Dict with substrate readings
            predictive_metrics: Dict with predictive hierarchy readings

        Returns:
            List of TestResult from all three tests
        """
        results: List[TestResult] = []

        # Test 1: GWT vs RPT
        try:
            r1 = self._test_gwt_rpt.run(workspace, qualia_synth, substrate_metrics, predictive_metrics)
            results.append(r1)
        except Exception as e:
            logger.error("GWT vs RPT test failed: %s", e)

        # Test 2: GWT vs Multiple Drafts
        try:
            r2 = self._test_gwt_md.run(workspace, md_engine)
            results.append(r2)
        except Exception as e:
            logger.error("GWT vs MD test failed: %s", e)

        # Test 3: HOT vs First-Order
        try:
            r3 = self._test_hot_fo.run(qualia_synth, hot_engine, substrate_metrics, predictive_metrics)
            results.append(r3)
        except Exception as e:
            logger.error("HOT vs First-Order test failed: %s", e)

        # Log to TheoryArbitration framework
        self._log_to_arbitration(results)

        self._results.extend(results)
        return results

    def _log_to_arbitration(self, results: List[TestResult]) -> None:
        """Push test results into the TheoryArbitration framework."""
        try:
            from core.consciousness.theory_arbitration import get_theory_arbitration
            arb = get_theory_arbitration()

            for result in results:
                event_id = f"adversarial_{result.test_name}_{int(result.timestamp)}"

                # Log predictions from both theories
                arb.log_prediction(
                    theory=result.theory_a.theory.value,
                    event_id=event_id,
                    prediction=result.theory_a.prediction,
                    confidence=result.theory_a.confidence,
                )
                arb.log_prediction(
                    theory=result.theory_b.theory.value,
                    event_id=event_id,
                    prediction=result.theory_b.prediction,
                    confidence=result.theory_b.confidence,
                )

                # Log the divergence
                arb.log_divergence(
                    event_id=event_id,
                    theory_a=result.theory_a.theory.value,
                    prediction_a=result.theory_a.prediction,
                    theory_b=result.theory_b.theory.value,
                    prediction_b=result.theory_b.prediction,
                )

                # Resolve with actual outcome
                if result.verdict == TestVerdict.THEORY_A_SUPPORTED:
                    outcome = result.theory_a.prediction
                elif result.verdict == TestVerdict.THEORY_B_SUPPORTED:
                    outcome = result.theory_b.prediction
                else:
                    outcome = f"inconclusive (BF={result.bayes_factor:.2f})"

                arb.resolve_prediction(event_id=event_id, actual_outcome=outcome)

        except Exception as e:
            logger.warning("Failed to log to TheoryArbitration: %s", e)

    def get_cumulative_evidence(self) -> Dict[str, float]:
        """Return cumulative Bayes factors across all tests for each theory pair."""
        pair_evidence: Dict[str, List[float]] = {}

        for r in self._results:
            key = f"{r.theory_a.theory.value}_vs_{r.theory_b.theory.value}"
            if key not in pair_evidence:
                pair_evidence[key] = []
            pair_evidence[key].append(r.bayes_factor)

        return {
            k: float(np.prod(v))  # Cumulative BF = product of individual BFs
            for k, v in pair_evidence.items()
        }

    def get_results_summary(self) -> Dict[str, Any]:
        """Summary of all tests run so far."""
        if not self._results:
            return {"n_tests": 0}

        verdicts = [r.verdict.value for r in self._results]
        return {
            "n_tests": len(self._results),
            "verdicts": verdicts,
            "cumulative_evidence": self.get_cumulative_evidence(),
            "latest_results": [r.summary() for r in self._results[-5:]],
        }
