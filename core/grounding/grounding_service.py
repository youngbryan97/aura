"""GroundingService — orchestrates kernel + network + prediction ledger.

This is the surface a skill calls.  Three operations now form a closed
learning loop:

    learn_from_example(symbol, raw, confirmed)
        encode the example, update the concept prototype, link the
        symbol to the concept, write a GroundingEvent.

    predict_symbol_applies(symbol, raw)
        encode the new observation, score it against the best
        candidate concept for ``symbol``, register a prediction in the
        F2 ledger so the prediction can later be resolved with truth.

    confirm_prediction(prediction_id, applies)
        resolve the prediction in the F2 ledger, derive the semantic
        reward, ask the SemanticWeightGovernor whether plasticity is
        allowed, run the plastic adapter update, and emit a
        SemanticWeightUpdateReceipt with full provenance.

When a plastic adapter is wired in, prediction features pass through
``adapter.adapt_features`` *before* similarity scoring, so confirmed
positive feedback measurably improves future prediction confidence on
related held-out examples.  This is the closed loop:

    prediction -> confirmation -> reward -> governor -> adapter update
        -> different features next prediction -> different similarity score.

Without an adapter+governor pair the service still works in
"observe-only" mode: predictions land in the ledger and lessons are
recorded, but no weight update happens.  Tests exercise both modes.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from core.grounding.grounding_kernel import GroundingKernel, GroundingObservation
from core.grounding.semiotic_network import SemioticNetwork
from core.grounding.types import GroundingEvent, new_id
from core.plasticity.plastic_adapter import GroundingPlasticAdapter
from core.plasticity.semantic_weight_governor import (
    PlasticityDecision,
    SemanticWeightGovernor,
)
from core.runtime.prediction_ledger import PredictionLedger
from core.runtime.receipts import (
    SemanticWeightUpdateReceipt,
    get_receipt_store,
)


class GroundingService:
    PLASTIC_MODULE_NAME = "grounding_plastic_adapter"

    def __init__(
        self,
        data_dir: Path,
        *,
        feature_dim: int = 128,
        ledger: Optional[PredictionLedger] = None,
        plastic_adapter: Optional[GroundingPlasticAdapter] = None,
        governor: Optional[SemanticWeightGovernor] = None,
        emit_receipts: bool = True,
    ):
        data_dir = Path(data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        self.network = SemioticNetwork(data_dir / "semiotic_network.json")
        self.kernel = GroundingKernel(feature_dim=feature_dim)
        self.ledger = ledger or PredictionLedger(data_dir / "grounding_predictions.db")
        self.plastic_adapter = plastic_adapter
        self.governor = governor
        self.emit_receipts = bool(emit_receipts)
        # Pending predictions: prediction_id -> context needed to apply a
        # plastic update on confirmation.
        self._pending: Dict[str, Dict[str, Any]] = {}
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
        # Run features through the plastic adapter when one is wired —
        # this is what makes the closed loop visible to the prediction.
        scoring_features = list(evidence.features)
        if self.plastic_adapter is not None:
            scoring_features = self.plastic_adapter.adapt_features(scoring_features)
        evidence.features = scoring_features
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

        # Stash everything the confirmation step needs to drive the
        # plastic update.  Features here are post-adapter so the layer
        # update flows through the same pathway it shaped.
        self._pending[prediction_id] = {
            "evidence_id": evidence.evidence_id,
            "concept_id": concept.concept_id,
            "link_id": link.link_id,
            "predicted_applies": applies,
            "predicted_confidence": confidence,
            "features": scoring_features,
        }

        return {
            "prediction_id": prediction_id,
            "applies": applies,
            "confidence": confidence,
            "concept_id": concept.concept_id,
            "evidence_id": evidence.evidence_id,
            "link_strength": link.strength,
        }

    def confirm_prediction(
        self,
        prediction_id: str,
        *,
        applies: bool,
        vitality: float = 1.0,
        curiosity: float = 0.5,
        arousal: float = 0.5,
        free_energy: float = 0.0,
    ) -> Dict[str, Any]:
        """Resolve, compute reward, route through governor, update adapter.

        The vitality/curiosity/arousal/free_energy parameters are the
        signals the SemanticWeightGovernor uses to decide modulation.
        Real Aura wires these to the live affect/substrate snapshots;
        tests pass them in directly.
        """
        record = self.ledger.resolve(
            prediction_id,
            observed={"applies": applies},
            observed_truth=applies,
        )

        ctx = self._pending.pop(prediction_id, None)
        out: Dict[str, Any] = {
            "prediction_id": prediction_id,
            "brier": record.brier,
            "error": record.error,
            "resolved": record.resolved,
            "weight_update": None,
            "weight_update_reason": None,
        }

        if ctx is None or self.plastic_adapter is None or self.governor is None:
            # No adapter+governor pair (or we lost track of context):
            # observe-only mode.  Loop still produces a Brier; no
            # weight update is applied.
            return out

        predicted_applies = bool(ctx.get("predicted_applies"))
        correct = predicted_applies == applies
        # Reward signed by correctness, magnitude scaled by confidence
        # so confidently-right > marginally-right and confidently-wrong
        # produces a stronger correction than uncertain-wrong.
        confidence = float(ctx.get("predicted_confidence", 0.5))
        reward = (1.0 if correct else -1.0) * max(0.1, confidence)

        decision = self.governor.decide(
            module_name=self.PLASTIC_MODULE_NAME,
            reward=reward,
            vitality=vitality,
            curiosity=curiosity,
            arousal=arousal,
            free_energy=free_energy,
        )

        # Defence in depth: even if the local governor allowed, refuse
        # any update whose target is not in the Will-side allow-list.
        try:
            from core.will import is_plastic_target_allowed

            target_allowed = is_plastic_target_allowed(self.PLASTIC_MODULE_NAME)
        except Exception:
            target_allowed = True  # be lenient if Will isn't importable

        update_info: Optional[Dict[str, Any]] = None
        if decision.allowed and target_allowed:
            update_info = self.plastic_adapter.update_from_reward(
                reward=reward, modulation=decision.modulation
            )

        out["weight_update"] = update_info
        out["weight_update_reason"] = (
            decision.reason if not decision.allowed else (
                "target_denied_by_will_policy" if not target_allowed else "applied"
            )
        )

        if self.emit_receipts:
            try:
                store = get_receipt_store()
                store.emit(
                    SemanticWeightUpdateReceipt(
                        cause=f"confirm_prediction:{prediction_id}",
                        module=self.PLASTIC_MODULE_NAME,
                        prediction_id=prediction_id,
                        concept_id=str(ctx.get("concept_id") or ""),
                        evidence_id=str(ctx.get("evidence_id") or ""),
                        reward=reward,
                        modulation=float(decision.modulation),
                        delta_norm=float(
                            (update_info or {}).get("delta_norm", 0.0)
                        ),
                        hebb_norm=float((update_info or {}).get("hebb_norm", 0.0)),
                        allowed=bool(decision.allowed and target_allowed),
                    )
                )
            except Exception:
                # Receipt failure must not break the learning loop.
                pass

        return out

    # ------------------------------------------------------------------
    # introspection helpers
    # ------------------------------------------------------------------
    def pending_prediction_ids(self) -> List[str]:
        return list(self._pending.keys())

    def has_plastic_loop(self) -> bool:
        return self.plastic_adapter is not None and self.governor is not None
