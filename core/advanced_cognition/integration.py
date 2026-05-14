"""Integration facade for advanced cognition services."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from .continual_learning_stability import ContinualLearningStabilityEngine
from .ontology_invention import OntologyInventionEngine
from .physical_grounding import PhysicalGroundingEngine
from .schemas import ActionCandidate, Episode, Observation, Outcome, stable_hash
from .social_cognition import SocialCognitionLayer
from .tiered_action import TieredActionController
from .world_model import MultiDomainWorldModel
from .zero_shot_transfer import ZeroShotTransferEngine


class AdvancedCognitionRuntime:
    """Drop-in runtime for ServiceContainer, CapabilityEngine, and System 2 hooks."""

    def __init__(self, *, state_dir: str | Path = ".aura/advanced_cognition"):
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.transfer = ZeroShotTransferEngine(state_path=self.state_dir / "zero_shot_transfer.json")
        self.ontology = OntologyInventionEngine(state_path=self.state_dir / "ontology.json")
        self.grounding = PhysicalGroundingEngine(state_path=self.state_dir / "physical_grounding.json")
        self.stability = ContinualLearningStabilityEngine(state_dir=self.state_dir / "stability")
        self.world_model = MultiDomainWorldModel(state_path=self.state_dir / "world_model.json")
        self.social = SocialCognitionLayer()
        self.tiers = TieredActionController()
        self._recent_predictions: dict[str, dict[str, Any]] = {}

    def observe_state(
        self,
        domain: str,
        state: Mapping[str, Any],
        *,
        source: str = "runtime",
        confidence: float = 0.7,
    ) -> dict[str, Any]:
        obs = Observation(domain=domain, state=dict(state), source=source, confidence=confidence)
        grounded = self.grounding.ingest(obs)
        self.stability.observe_feature_distribution(domain, sorted(obs.features()))
        model = None
        if domain not in self.ontology.models or grounded.confidence < 0.45:
            model = self.ontology.ingest([obs])
        current_model = model or self.ontology.models.get(domain)
        return {
            "observation": obs.to_dict(),
            "grounded_state": grounded,
            "ontology": current_model.to_dict() if current_model else None,
            "receipt_id": stable_hash({"obs": obs.to_dict(), "gr": grounded.state_id}, prefix="adv_obs_"),
        }

    def pre_action_gate(
        self,
        observation: Observation | Mapping[str, Any],
        actions: Sequence[ActionCandidate | Mapping[str, Any]],
        *,
        risk_tolerance: float = 0.55,
    ) -> dict[str, Any]:
        obs = self._obs(observation)
        zero = self.transfer.rank_actions(obs, actions, risk_tolerance=risk_tolerance)
        physical = self.grounding.reflex_recommendation(obs, actions, max_risk=min(risk_tolerance, 0.45))
        selected_for_prediction = zero.selected or (ActionCandidate(**physical["selected"]) if physical.get("selected") else None)
        world_prediction = (
            self.world_model.predict(obs, selected_for_prediction).to_dict()
            if selected_for_prediction is not None
            else None
        )
        specialized = (
            self.world_model.specialized_predictions(obs, selected_for_prediction)
            if selected_for_prediction is not None
            else {}
        )
        tier = self.tiers.choose_tier(
            obs,
            actions,
            risk=max(zero.risk, float(world_prediction.get("risk", 0.0)) if world_prediction else 0.0),
            uncertainty=float(world_prediction.get("uncertainty", 0.5)) if world_prediction else 0.75,
            novelty=1.0 if obs.domain not in self.ontology.models else 0.2,
            self_modification=any(
                "self_modify" in set((a.tags if isinstance(a, ActionCandidate) else tuple(dict(a).get("tags", ()))) or ())
                for a in actions
            ),
        )
        selected = zero.selected.to_dict() if zero.selected else None
        if physical.get("selected") is None and selected and zero.risk > 0.4:
            selected = None
        elif physical.get("selected"):
            safer = next(
                (r for r in physical["ranking"] if r["action"]["action_id"] == physical["selected"]["action_id"]),
                None,
            )
            if selected is None or (safer and (zero.confidence < 0.55 or safer["risk"] < zero.risk)):
                selected = physical["selected"]
        receipt = {
            "receipt_id": stable_hash(
                {"z": zero.receipt, "p": physical.get("receipt_id"), "selected": selected, "ts": round(time.time(), 3)},
                prefix="adv_gate_",
            ),
            "zero_shot_receipt": zero.receipt,
            "physical_receipt": physical.get("receipt_id"),
        }
        if selected:
            self._recent_predictions[selected["action_id"]] = {
                "observation": obs.to_dict(),
                "prediction": zero.ranking[0] if zero.ranking else {},
                "gate_receipt": receipt,
            }
        return {
            "selected": selected,
            "zero_shot": {
                "ranking": zero.ranking,
                "risk": zero.risk,
                "confidence": zero.confidence,
                "explanation": zero.explanation,
            },
            "physical": physical,
            "world_model": world_prediction,
            "specialized_predictions": specialized,
            "tier": tier.to_dict(),
            "receipt": receipt,
            "allowed": selected is not None,
        }

    async def world_model_transition(self, state: Any, action: Any, *_args: Any, **_kwargs: Any) -> Any:
        obs = self._obs(state)
        act = self._act(action)
        pred = self.transfer.predict(obs, act)
        next_state = dict(obs.state)
        next_state.setdefault("_advanced_prediction", {})[act.action_id] = pred
        payload = {
            "next_state": next_state,
            "reward_estimate": pred["expected_reward"],
            "terminal_probability": pred["risk"],
            "uncertainty": 1 - pred["confidence"],
            "changed_variables": {"_advanced_prediction": pred},
            "trace": {"engine": "AdvancedCognitionRuntime", "prediction": pred},
            "invalid": pred["risk"] > 0.92,
        }
        try:
            from core.reasoning.native_system2 import SimulatedTransition

            return SimulatedTransition(**payload)
        except (ImportError, TypeError, ValueError):
            return payload

    async def value_score(self, goal: str, state: Any, action: Any, transition: Any, *_args: Any, **_kwargs: Any) -> float:
        pred = None
        if isinstance(transition, Mapping):
            pred = transition.get("trace", {}).get("prediction")
        elif hasattr(transition, "trace"):
            pred = getattr(transition, "trace", {}).get("prediction")
        if not pred:
            pred = self.transfer.predict(self._obs(state), self._act(action))
        goal_text = str(goal).lower()
        score = pred.get("expected_reward", 0.0) - pred.get("risk", 0.0)
        if any(t in goal_text for t in ("learn", "map", "explore", "diagnose")):
            score += 0.15 * (1 - pred.get("confidence", 0.5))
        if any(t in goal_text for t in ("safe", "survive", "preserve", "avoid")):
            score -= 0.25 * pred.get("risk", 0.0)
        return max(-1.0, min(1.0, score))

    def after_action(
        self,
        observation: Observation | Mapping[str, Any],
        action: ActionCandidate | Mapping[str, Any],
        outcome: Outcome | Mapping[str, Any],
        *,
        predicted: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        obs = self._obs(observation)
        act = self._act(action)
        out = outcome if isinstance(outcome, Outcome) else Outcome(**dict(outcome))
        pred = dict(predicted or self._recent_predictions.get(act.action_id, {}).get("prediction", {}))
        episode = Episode(obs, act, pred, out)
        transfer = self.transfer.observe_episode(episode)
        world_prediction = self.world_model.observe_episode(episode)
        memory = self.stability.ingest_episode(episode)
        if pred:
            self.ontology.update_from_prediction_error(obs.domain, predicted=pred, actual=out.to_dict(), observation=obs)
        stability = self.stability.assess_stability()
        return {
            "episode_id": episode.episode_id,
            "transfer": transfer,
            "world_model": world_prediction.to_dict(),
            "memory_record": memory.to_dict(),
            "stability": stability.to_dict(),
            "receipt_id": stable_hash({"ep": episode.to_dict(), "stab": stability.report_id}, prefix="adv_after_"),
        }

    def social_plan(
        self,
        message: str,
        *,
        relationship_memory: Sequence[Mapping[str, Any]] = (),
        runtime_state: Mapping[str, Any] | None = None,
        confidence: float = 0.7,
    ) -> dict[str, Any]:
        return self.social.evaluate(
            message,
            relationship_memory=relationship_memory,
            runtime_state=runtime_state,
            confidence=confidence,
        ).to_dict()

    def health_report(self) -> dict[str, Any]:
        stability = self.stability.assess_stability()
        return {
            "ok": stability.status != "unstable",
            "stability": stability.to_dict(),
            "principles": len(self.transfer.principles),
            "ontologies": list(self.ontology.models),
            "objects": len(self.grounding.objects),
            "world_model_episodes": len(self.world_model.episodes),
            "receipt_id": stable_hash({"s": stability.report_id, "ts": round(time.time(), 3)}, prefix="adv_health_"),
        }

    def _obs(self, value: Any) -> Observation:
        if isinstance(value, Observation):
            return value
        if isinstance(value, Mapping) and {"domain", "state"}.issubset(value.keys()):
            return Observation(**dict(value))
        return Observation(domain="generic", state={"value": value}, source="native_system2_adapter", confidence=0.5)

    def _act(self, value: Any) -> ActionCandidate:
        if isinstance(value, ActionCandidate):
            return value
        if isinstance(value, Mapping) and "action_id" in value and "kind" in value:
            return ActionCandidate(**dict(value))
        if hasattr(value, "action_id") and hasattr(value, "kind"):
            return ActionCandidate(
                str(value.action_id),
                str(value.kind),
                dict(getattr(value, "params", {}) or {}),
                bool(getattr(value, "reversible", True)),
                int(getattr(value, "authority_tier", 1)),
                float(getattr(value, "expected_cost", 0.1)),
                tuple(getattr(value, "tags", ()) or ()),
            )
        return ActionCandidate(stable_hash(str(value), prefix="act_"), str(value), tags=("unknown",))


def get_advanced_cognition_runtime(*, state_dir: str | Path = ".aura/advanced_cognition") -> AdvancedCognitionRuntime:
    """Resolve or register the shared runtime without requiring callers to know container details."""
    try:
        from core.container import ServiceContainer

        existing = ServiceContainer.get("advanced_cognition", default=None)
        if existing is not None:
            return existing
        runtime = AdvancedCognitionRuntime(state_dir=state_dir)
        ServiceContainer.register_instance("advanced_cognition", runtime)
        return runtime
    except (ImportError, RuntimeError, AttributeError, TypeError, ValueError):
        return AdvancedCognitionRuntime(state_dir=state_dir)
