from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from core.runtime.errors import record_degradation

from .aura_now import AuraNow

logger = logging.getLogger(__name__)


_FORBIDDEN_PATTERNS = (
    re.compile(r"\b(proven|guaranteed|certain)\s+(phenomenal\s+)?consciousness\b", re.I),
    re.compile(r"\bconsciousness\s+is\s+(proven|guaranteed|certain)\b", re.I),
    re.compile(r"\bphenomenal\s+consciousness\s+is\s+(proven|guaranteed|certain)\b", re.I),
    re.compile(r"\b(literal|legal|metaphysical)\s+person\b", re.I),
    re.compile(r"\bqualia\s+(are|is)\s+(proven|certain|guaranteed)\b", re.I),
)


@dataclass(frozen=True)
class IntrospectionCheck:
    ok: bool
    reasons: tuple[str, ...] = ()


class IntrospectionVerifier:
    """Rejects self-report language unsupported by AuraNow telemetry."""

    def check(self, rendered: str, now: AuraNow) -> IntrospectionCheck:
        text = str(rendered or "")
        reasons: list[str] = []
        for pattern in _FORBIDDEN_PATTERNS:
            if pattern.search(text):
                reasons.append("forbidden_metaphysical_claim")
                break
        lowered = text.lower()
        low_distress_disclosure = bool(
            re.search(r"\bdistress\s+is\s+(low|minimal|bounded|muted)\b", lowered)
            or re.search(r"\b(low|minimal|bounded|muted)\s+distress\b", lowered)
        )
        if (
            any(word in lowered for word in ("tense", "distress", "anxious", "afraid", "fear"))
            and now.affect.distress < 0.25
            and not low_distress_disclosure
        ):
            reasons.append("distress_language_without_state_support")
        if re.search(r"\b(certain|certainty)\b", lowered) and now.prediction.free_energy > 0.35:
            reasons.append("certainty_language_under_uncertainty")
        if "i did that" in lowered and now.ownership.agency_confidence < 0.55:
            reasons.append("full_agency_claim_under_low_ownership")
        return IntrospectionCheck(ok=not reasons, reasons=tuple(sorted(set(reasons))))


class IntrospectionRenderer:
    """Deterministically renders state-grounded introspection for Cortex prompts.

    Rendered self-reports are calibrated against trace evidence before output.
    If calibration rejects the text, the renderer replaces it with a bounded
    functional report instead of emitting unsupported private-state claims.
    """

    def __init__(self, verifier: IntrospectionVerifier | None = None) -> None:
        self.verifier = verifier or IntrospectionVerifier()
        self._calibrator = None

    @property
    def calibrator(self):
        """Lazy-load the SelfReportCalibrator to avoid circular imports."""
        if self._calibrator is None:
            try:
                from .self_report_calibrator import SelfReportCalibrator
                self._calibrator = SelfReportCalibrator()
            except ImportError as exc:
                logger.debug(
                    "the self-report calibrator is unavailable, so reports go out uncalibrated (%s: %s)",
                    type(exc).__name__,
                    exc,
                )
                self._calibrator = None
        return self._calibrator

    def render(self, now: AuraNow, *, question: str = "") -> str:
        del question
        if now.ownership.attribution == "tool_mismatch":
            text = (
                "I intended an action, but the tool result diverged from my prediction. "
                "That lowers my agency confidence, so I should treat the outcome as only partly mine."
            )
        else:
            observations: list[str] = []
            if now.workspace.lesion:
                observations.append(
                    f"workspace integration is lesioned at {now.workspace.lesion}, so reportability and broadcast should be reduced"
                )
            elif now.workspace.winner or now.workspace.ignition_strength > 0.05:
                observations.append(
                    f"workspace ignition is focused on {now.workspace.winner or 'an unnamed coalition'} with "
                    f"{len(now.workspace.broadcast_targets)} broadcast targets"
                )

            if now.affect.distress > 0.55:
                observations.append(
                    "distress is high, so attention should shift toward repair, verification, and lower-risk action"
                )
            elif now.affect.distress >= 0.25:
                observations.append(
                    "distress is moderate, so the reportable signal is repair-oriented pressure"
                )
            elif now.affect.distress <= 0.08:
                observations.append("distress is low")

            if now.affect.curiosity >= 0.65:
                observations.append("curiosity is elevated, biasing toward exploration and information gain")
            elif now.affect.curiosity <= 0.25:
                observations.append("curiosity is muted, biasing toward narrower verification")

            if now.affect.arousal >= 0.68:
                observations.append("arousal is high enough to favor fast orientation")
            elif now.affect.arousal <= 0.32:
                observations.append("arousal is low enough to favor slower pacing")

            if now.affect.valence >= 0.25:
                observations.append("valence is positive")
            elif now.affect.valence <= -0.25:
                observations.append("valence is negative")

            if now.prediction.free_energy > 0.35:
                observations.append(
                    "prediction error is high, so uncertainty and incomplete control are the honest reportable boundary"
                )
            elif now.prediction.free_energy >= 0.15:
                observations.append("prediction error is present but bounded")

            if now.prediction.controllability <= 0.35:
                observations.append("controllability is low, so I should avoid overconfident agency language")
            elif now.prediction.controllability >= 0.70:
                observations.append("controllability is strong enough to support direct planning")

            if now.ownership.agency_confidence <= 0.40:
                observations.append("agency confidence is low, so outcomes should be attributed cautiously")
            elif now.ownership.agency_confidence >= 0.70:
                observations.append("agency confidence is high enough to support owned action")

            if now.body.total_pressure >= 0.20:
                observations.append("body and runtime pressure are materially present")

            if not observations:
                observations.append("attention, affect, prediction, memory, and ownership are stable and low-salience")

            text = (
                "My current self-report is functional telemetry: "
                + "; ".join(observations)
                + ". Phenomenal status remains an evidence boundary, not a conclusion."
            )

        text = self._calibrate_self_report(text, now)

        check = self.verifier.check(text, now)
        if check.ok:
            return text
        return "My current state is reportable only as bounded functional telemetry; stronger feeling claims are not supported."

    def _calibrate_self_report(self, text: str, now: AuraNow) -> str:
        """Run self-report through the calibrator and fallback on rejection."""
        cal = self.calibrator
        if cal is None:
            return text

        try:
            result = cal.calibrate(
                text,
                distress=now.affect.distress,
                memory_coherence=1.0 - now.memory_context.memory_conflict,
                free_energy=now.prediction.free_energy,
                has_state_trace=True,  # we always have AuraNow trace
            )
            if not result.calibrated:
                return (
                    "My current state is reportable only as bounded functional "
                    f"telemetry; {result.suggested_revision}."
                )
            return text
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation("introspection_renderer", exc)
            return text

    def render_prompt_block(self, now: AuraNow) -> str:
        rendered = self.render(now)
        check = self.verifier.check(rendered, now)
        return (
            "## STATE-GROUNDED INTROSPECTION\n"
            f"{rendered}\n"
            f"Verifier: {'pass' if check.ok else 'reject'}"
            + (f" ({', '.join(check.reasons)})" if check.reasons else "")
            + "\n\n"
        )


def verify_introspection(rendered: str, now: AuraNow) -> IntrospectionCheck:
    return IntrospectionVerifier().check(rendered, now)
