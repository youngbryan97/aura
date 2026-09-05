"""core/epistemics/confidence_calibrator.py — Confidence Calibrator."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from core.epistemics.source_ranker import get_source_ranker

if TYPE_CHECKING:
    pass

logger = logging.getLogger("Aura.ConfidenceCalibrator")


class ConfidenceCalibrator:
    """Calibrates confidence scores based on source trust, evidence count, and logical disputes."""

    @staticmethod
    def calibrate(subject: Any) -> float | None:
        """Calibrate a ClaimGraph in-place or return confidence for one claim."""
        if not hasattr(subject, "nodes"):
            return ConfidenceCalibrator._calibrate_claim(subject)
        graph = subject
        ranker = get_source_ranker()
        
        for node in graph.nodes.values():
            # 1. Average source reliability
            source_scores = [ranker.get_reliability(src) for src in node.sources]
            avg_source_reliability = sum(source_scores) / len(source_scores) if source_scores else 0.5

            # 2. Support evidence factor (slight boost for multiple supporting evidences)
            support_boost = min(0.15, len(node.supporting_evidence) * 0.03)

            # 3. Contradiction penalty
            contradiction_penalty = 0.0
            for contradiction_id in node.contradiction_links:
                if contradiction_id in graph.nodes:
                    other = graph.nodes[contradiction_id]
                    # Penalty is proportional to the other claim's confidence and freshness
                    contradiction_penalty += 0.25 * other.confidence * other.freshness
            
            contradiction_penalty = min(0.70, contradiction_penalty)

            # 4. Final calculation
            base = avg_source_reliability + support_boost
            calibrated = base * (1.0 - contradiction_penalty)
            
            old_conf = node.confidence
            node.confidence = max(0.01, min(0.99, calibrated))
            
            logger.debug(
                "Calibrated claim %s: %.2f -> %.2f (penalty=%.2f)",
                node.claim_id, old_conf, node.confidence, contradiction_penalty
            )
        logger.info("Truth and Epistemics Calibration Pass completed.")
        return None

    @staticmethod
    def _calibrate_claim(claim: Any) -> float:
        """Return calibrated confidence for a TruthEngine Claim dataclass."""
        ranker = get_source_ranker()
        sources = list(getattr(claim, "sources", []) or [])
        source_scores = [ranker.get_reliability(src) for src in sources]
        avg_source_reliability = sum(source_scores) / len(source_scores) if source_scores else 0.5

        metadata = getattr(claim, "metadata", {}) or {}
        supporting_evidence = list(metadata.get("supporting_evidence", []) or [])
        supporting_claims = list(getattr(claim, "supporting_claims", []) or [])
        support_boost = min(0.15, (len(supporting_evidence) + len(supporting_claims)) * 0.03)

        contradiction_links = list(getattr(claim, "contradiction_links", []) or [])
        contradiction_penalty = min(0.70, 0.20 * len(contradiction_links))
        verification_count = int(getattr(claim, "verification_count", 0) or 0)
        verification_boost = min(0.12, verification_count * 0.04)

        calibrated = (avg_source_reliability + support_boost + verification_boost) * (1.0 - contradiction_penalty)
        return max(0.01, min(0.99, calibrated))
