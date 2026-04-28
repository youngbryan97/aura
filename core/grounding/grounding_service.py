"""GroundingService — orchestrates kernel + network + prediction ledger.

This is the surface a skill calls.  Two operations:

    learn_from_example(symbol, raw, confirmed)
        encode the example, update the concept prototype, link the
        symbol to the concept, write a GroundingEvent.

    predict_symbol_applies(symbol, raw)
        encode the new observation, score it against the best
        candidate concept for ``symbol``, register a prediction in the
        F2 ledger so the prediction can later be resolved with truth.

The service deliberately *uses the F2 prediction ledger* rather than
inventing a parallel one — that means every grounding prediction is
auditable through the same Brier/calibration machinery the rest of
Aura already trusts.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from core.grounding.grounding_kernel import GroundingKernel, GroundingObservation
from core.grounding.semiotic_network import SemioticNetwork
from core.grounding.types import GroundingEvent, new_id
from core.runtime.prediction_ledger import PredictionLedger


class GroundingService:
    def __init__(
        self,
        data_dir: Path,
        *,
        feature_dim: int = 128,
        ledger: Optional[PredictionLedger] = None,
    ):
        data_dir = Path(data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        self.network = SemioticNetwork(data_dir / "semiotic_network.json")
        self.kernel = GroundingKernel(feature_dim=feature_dim)
        self.ledger = ledger or PredictionLedger(data_dir / "grounding_predictions.db")
        # Register the default text method on the network.
        self.network.register_method(self.kernel.default_text_method())

    # ------------------------------------------------------------------
    def learn_from_example(
        self,
        *,
        symbol: str,
        raw: str,
        modality: str = "text",
        confirmed: bool = True,
        source: str = "user",
    ) -> Dict[str, Any]:
        obs = GroundingObservation(
            symbol=symbol,
            modality=modality,
            raw=raw,
            source=source,
            label_confirmed=confirmed,
        )
        evidence = self.kernel.encode(obs)
        self.network.add_evidence(evidence)

        method_id = self.kernel.default_text_method().method_id
        concept = self.network.create_or_update_concept(
            label=symbol,
            kind="textual" if modality == "text" else "perceptual",
            features=evidence.features,
            method_id=method_id,
            positive=confirmed,
        )
        link = self.network.link_symbol(
            symbol=symbol,
            concept_id=concept.concept_id,
            source=source,
            delta=0.10 if confirmed else -0.15,
        )

        event = GroundingEvent(
            event_id=new_id("grounding_event"),
            symbol=symbol,
            concept_id=concept.concept_id,
            evidence_id=evidence.evidence_id,
            prediction=True,
            observed=confirmed,
            reward=1.0 if confirmed else -1.0,
            confidence_before=max(0.0, concept.confidence - 0.05),
            confidence_after=concept.confidence,
            source=source,
        )
        self.network.record_event(event)
        return {
            "symbol": symbol,
            "concept_id": concept.concept_id,
            "evidence_id": evidence.evidence_id,
            "link_strength": link.strength,
            "confidence": concept.confidence,
            "reward": event.reward,
        }

    def predict_symbol_applies(
        self,
        *,
        symbol: str,
        raw: str,
        modality: str = "text",
    ) -> Dict[str, Any]:
        obs = GroundingObservation(symbol=symbol, modality=modality, raw=raw)
        evidence = self.kernel.encode(obs)
        self.network.add_evidence(evidence)

        candidates = self.network.concepts_for_symbol(symbol)
        if not candidates:
            return {
                "prediction_id": "",
                "applies": False,
                "confidence": 0.0,
                "reason": "unknown_symbol",
                "evidence_id": evidence.evidence_id,
            }

        concept, link = candidates[0]
        score = self.network.score_evidence_for_concept(
            evidence.evidence_id, concept.concept_id
        )
        # Map cosine [-1, 1] to a probability-shaped scalar in [0, 1],
        # then weight by the link strength * concept confidence.
        confidence = (
            max(0.0, min(1.0, (score + 1.0) / 2.0))
            * link.strength
            * concept.confidence
        )
        # Threshold tuned for the hash-token feature encoder; with
        # higher-quality embeddings (CLIP, etc.) this can be raised.
        applies = confidence >= 0.20

        prediction_id = self.ledger.register(
            belief=symbol,
            modality=modality,
            action="ground_predict",
            expected={"applies": True},
            prior_prob=confidence,
            agent_state={"concept_id": concept.concept_id, "link_id": link.link_id},
        )
        return {
            "prediction_id": prediction_id,
            "applies": applies,
            "confidence": confidence,
            "concept_id": concept.concept_id,
            "evidence_id": evidence.evidence_id,
            "link_strength": link.strength,
        }

    def confirm_prediction(self, prediction_id: str, *, applies: bool) -> Dict[str, Any]:
        record = self.ledger.resolve(
            prediction_id,
            observed={"applies": applies},
            observed_truth=applies,
        )
        return {
            "prediction_id": prediction_id,
            "brier": record.brier,
            "error": record.error,
            "resolved": record.resolved,
        }
