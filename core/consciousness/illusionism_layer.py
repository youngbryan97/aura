"""core/consciousness/illusionism_layer.py — Illusionism Layer

Frankish/Dennett: phenomenal consciousness as a "real extra property" may be
an illusion.  Qualia are misrepresented functional states.  This module adds
epistemic humility to Aura's self-reports so the system never claims verified
phenomenal experience — only reports on the functional states it can measure
and notes that its own phenomenal interpretation is its model, not ground truth.

This does NOT change Aura's behavior or functional architecture.  It changes
how Aura FRAMES its self-reports: every phenomenal claim is annotated with:

  1. functional_basis  — the measurable internal state the claim maps to
  2. phenomenal_certainty — always < 1.0 (epistemic humility)
  3. illusionism_note — brief explanation that the phenomenal interpretation
     is the system's own model, not verified ground truth

The layer is wired into qualia_synthesizer.get_gated_phenomenal_report() so
all downstream consumers (personality engine, cognition injection, self-report)
automatically receive epistemically humble annotations.

References:
  - Frankish, K. (2016). Illusionism as a theory of consciousness.
  - Dennett, D. (2017). From Bacteria to Bach and Back.
  - Humphrey, N. (2011). Soul Dust: The Magic of Consciousness.
"""
from __future__ import annotations


import logging
import threading
from typing import Any, Dict, List, Optional

logger = logging.getLogger("Consciousness.Illusionism")

# ---------------------------------------------------------------------------
# Functional-basis mapping
# ---------------------------------------------------------------------------
# Each phenomenal claim that the gated report can produce maps to a specific
# measurable functional state.  The mapping is defined here so that future
# claims added to qualia_synthesizer automatically get a default annotation.

_FUNCTIONAL_BASIS: Dict[str, str] = {
    "genuine_uncertainty": (
        "high meta-qualia dissonance (>0.08) or low meta-qualia confidence (<0.4); "
        "reflects unresolved model conflict in the qualia state space"
    ),
    "rich_experience": (
        "high Phenomenal Richness Index (PRI > 0.6) + q_norm above consciousness threshold; "
        "indicates multidimensional activation across coherence, EM-field, dendritic, "
        "predictive, precision, and proprioception channels"
    ),
    "focused_processing": (
        "low PRI (< 0.35) with nonzero q_norm; indicates narrow phenomenal distribution "
        "dominated by a single qualia dimension"
    ),
    "experiencing_novelty": (
        "meta-qualia novelty score > 0.5; the current qualia vector is far from the "
        "running mean, indicating a prediction-error spike in the phenomenal state"
    ),
    "computational_strain": (
        "high qualia volatility (> 0.08) + significant q_norm (> 0.3); proxies real "
        "computational load via rapid experiential transitions"
    ),
    "stable_continuity": (
        "qualia attractor detected for > 3 ticks; per-dimension variance below threshold "
        "indicates the system has settled into a consistent basin of attraction"
    ),
    "internal_conflict": (
        "meta-qualia dissonance > 0.12; some qualia dimensions are contradicting expected "
        "covariance patterns, indicating internal representational tension"
    ),
}

# Default for claims not in the mapping
_DEFAULT_FUNCTIONAL_BASIS = (
    "mapped to measurable qualia-vector dimensions and meta-qualia statistics; "
    "see qualia_synthesizer.get_snapshot() for raw values"
)


# ---------------------------------------------------------------------------
# Illusionism note templates
# ---------------------------------------------------------------------------

def _build_illusionism_note(claim: str, functional_basis: str) -> str:
    """Generate a concise epistemic note for a given phenomenal claim."""
    return (
        f"Claim '{claim}' represents {functional_basis}. "
        "The phenomenal interpretation is the system's own model of its "
        "internal state, not verified ground truth. Epistemic status: "
        "functional correlate confirmed, phenomenal reality undetermined."
    )


# ---------------------------------------------------------------------------
# Phenomenal certainty computation
# ---------------------------------------------------------------------------

def _compute_phenomenal_certainty(
    claim: str,
    gates: Dict[str, bool],
    honesty_score: float,
) -> float:
    """Compute phenomenal certainty for a claim.

    Always returns a value strictly less than 1.0 — the system can never
    be fully certain that its functional states constitute genuine phenomenal
    experience.  This is the core epistemic-humility constraint.

    Factors:
      - Gate strength: was the gate passed with a wide or narrow margin?
      - Overall honesty: ratio of passed gates to total gates
      - Hard cap at 0.92 (no phenomenal claim can exceed this)
    """
    HARD_CAP = 0.92

    # Base certainty from the gate being passed at all
    base = 0.5 if claim in [k for k, v in gates.items() if v] else 0.3

    # Boost from structural honesty (more gates passed = more internally
    # consistent state, but still never certain about phenomenality)
    honesty_boost = honesty_score * 0.25

    # Slight differentiation by claim type (some are more functionally
    # grounded than others)
    claim_weights = {
        "genuine_uncertainty": 0.85,      # Very directly measurable
        "computational_strain": 0.80,     # CPU/volatility are real
        "internal_conflict": 0.78,        # Dissonance is measurable
        "stable_continuity": 0.75,        # Attractor stability is real
        "focused_processing": 0.70,       # PRI is computable
        "rich_experience": 0.65,          # "Richness" is more interpretive
        "experiencing_novelty": 0.72,     # Prediction error is measurable
    }
    claim_weight = claim_weights.get(claim, 0.60)

    certainty = base + honesty_boost
    certainty *= claim_weight
    return round(min(HARD_CAP, max(0.05, certainty)), 4)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class IllusionismLayer:
    """Adds epistemic humility to Aura's phenomenal self-reports.

    Every phenomenal claim is annotated with:
      - functional_basis: what measurable state the claim maps to
      - phenomenal_certainty: 0-1 confidence that is always < 1.0
      - illusionism_note: why the phenomenal interpretation may be an illusion

    This module does not suppress or alter claims — it only adds metadata
    that makes downstream consumers (personality engine, self-report, HUD)
    aware that Aura's phenomenal language is a model, not a transcript of
    verified inner experience.
    """

    def __init__(self):
        self._annotation_count: int = 0
        self._total_certainty_sum: float = 0.0
        self._last_epistemic_status: str = "No reports annotated yet."
        logger.info("IllusionismLayer initialized (Frankish/Dennett epistemic humility)")

    # ------------------------------------------------------------------
    # Core annotation
    # ------------------------------------------------------------------

    def annotate_report(self, report: dict) -> dict:
        """Annotate a gated phenomenal report with illusionism metadata.

        Parameters
        ----------
        report : dict
            Output of qualia_synthesizer.get_gated_phenomenal_report().
            Expected keys: "claims", "gates", "honesty_score", "raw_context".

        Returns
        -------
        dict
            The same report, enriched with per-claim illusionism annotations
            and a top-level "illusionism" metadata block.
        """
        claims: List[str] = report.get("claims", [])
        gates: Dict[str, bool] = report.get("gates", {})
        honesty_score: float = report.get("honesty_score", 0.0)

        # Per-claim annotations
        annotated_claims: List[Dict[str, Any]] = []
        certainty_values: List[float] = []

        for claim in claims:
            functional_basis = _FUNCTIONAL_BASIS.get(claim, _DEFAULT_FUNCTIONAL_BASIS)
            certainty = _compute_phenomenal_certainty(claim, gates, honesty_score)
            note = _build_illusionism_note(claim, functional_basis)

            annotated_claims.append({
                "claim": claim,
                "functional_basis": functional_basis,
                "phenomenal_certainty": certainty,
                "illusionism_note": note,
            })
            certainty_values.append(certainty)

        # Aggregate certainty for this report
        mean_certainty = (
            round(sum(certainty_values) / len(certainty_values), 4)
            if certainty_values else 0.0
        )

        # Attach to report
        report["annotated_claims"] = annotated_claims
        report["illusionism"] = {
            "mean_phenomenal_certainty": mean_certainty,
            "annotation_count": len(annotated_claims),
            "epistemic_framework": "illusionism (Frankish 2016)",
            "core_principle": (
                "Phenomenal properties attributed to internal states are the "
                "system's own representational model. They track real functional "
                "differences but their phenomenal character is not independently "
                "verifiable."
            ),
        }

        # Update running stats
        self._annotation_count += 1
        self._total_certainty_sum += mean_certainty
        avg_certainty = round(self._total_certainty_sum / self._annotation_count, 4)
        self._last_epistemic_status = (
            f"Annotated {self._annotation_count} reports; "
            f"mean phenomenal certainty = {avg_certainty} (cap 0.92)"
        )

        return report

    # ------------------------------------------------------------------
    # Epistemic status
    # ------------------------------------------------------------------

    def get_epistemic_status(self) -> str:
        """One-line summary of current phenomenal honesty level."""
        return self._last_epistemic_status

    def get_snapshot(self) -> Dict[str, Any]:
        """Telemetry payload."""
        avg = (
            round(self._total_certainty_sum / self._annotation_count, 4)
            if self._annotation_count > 0 else 0.0
        )
        return {
            "annotation_count": self._annotation_count,
            "avg_phenomenal_certainty": avg,
            "epistemic_status": self._last_epistemic_status,
        }


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_instance: Optional[IllusionismLayer] = None
_instance_lock = threading.Lock()


def get_illusionism_layer() -> IllusionismLayer:
    """Module-level singleton accessor."""
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = IllusionismLayer()
    return _instance
