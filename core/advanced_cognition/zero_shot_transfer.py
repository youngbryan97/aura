"""Fast zero/few-shot transfer through causal symbolic abstraction."""
from __future__ import annotations

import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from core.runtime.atomic_writer import atomic_write_text

from .schemas import (
    ActionCandidate,
    ActionDecision,
    Episode,
    Observation,
    Outcome,
    Principle,
    clamp,
    stable_hash,
)

DANGEROUS = {
    "action:irreversible",
    "tag:delete",
    "tag:overwrite",
    "tag:network_post",
    "tag:credential",
    "tag:self_modify",
    "tag:unknown_use",
    "tag:deploy",
    "tag:spend",
}

RESOURCE_ALIASES = {
    "hp": "health",
    "health": "health",
    "battery": "energy",
    "energy": "energy",
    "compute": "energy",
    "ram": "energy",
    "time": "time",
    "money": "capital",
    "trust": "social_trust",
    "inventory": "capability",
}


class ZeroShotTransferEngine:
    """Induces cross-domain principles from sparse outcome memories."""

    def __init__(self, *, state_path: str | Path | None = None) -> None:
        self.state_path = Path(state_path) if state_path else None
        self.episodes: list[Episode] = []
        self.principles: dict[str, Principle] = {}
        self.feature_effect_stats: dict[str, dict[str, float]] = defaultdict(
            lambda: {"support": 0.0, "harm": 0.0, "reward": 0.0, "surprise": 0.0}
        )
        self.domain_feature_memory: dict[str, Counter[str]] = defaultdict(Counter)
        if self.state_path and self.state_path.exists():
            self.load(self.state_path)

    def observe_episode(self, episode: Episode) -> dict[str, Any]:
        self.episodes.append(episode)
        for feature in episode.features():
            stats = self.feature_effect_stats[feature]
            stats["support"] += 1.0
            stats["harm"] += episode.outcome.harm
            stats["reward"] += episode.outcome.reward
            stats["surprise"] += episode.outcome.surprise
            self.domain_feature_memory[episode.observation.domain][feature] += 1
        induced = self._induce(episode)
        if self.state_path:
            self.save(self.state_path)
        return {
            "episode_id": episode.episode_id,
            "induced_principles": induced,
            "principle_count": len(self.principles),
        }

    def observe(
        self,
        observation: Observation | Mapping[str, Any],
        action: ActionCandidate | Mapping[str, Any],
        outcome: Outcome | Mapping[str, Any],
        *,
        predicted: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        obs = observation if isinstance(observation, Observation) else Observation(**dict(observation))
        act = action if isinstance(action, ActionCandidate) else ActionCandidate(**dict(action))
        out = outcome if isinstance(outcome, Outcome) else Outcome(**dict(outcome))
        return self.observe_episode(Episode(obs, act, dict(predicted or {}), out))

    def _induce(self, ep: Episode) -> list[str]:
        obs_features = ep.observation.features()
        action_features = ep.action.features()
        names: list[str] = []

        for raw, delta in ep.outcome.resources_delta.items():
            resource = RESOURCE_ALIASES.get(str(raw).lower(), str(raw).lower())
            conditions = self._salient(
                obs_features,
                ("resource", "health", "energy", "low", "has:entities", "hostile", "unknown"),
            )
            action_signature = self._salient(action_features, ("action:", "tag:", "irreversible"))
            if delta < 0:
                names.append(
                    self._upsert(
                        f"avoid_{resource}_loss_{stable_hash(sorted(conditions | action_signature))[:8]}",
                        conditions,
                        action_signature,
                        f"{resource}_loss",
                        ep,
                    )
                )
            elif delta > 0:
                names.append(
                    self._upsert(
                        f"seek_{resource}_gain_{stable_hash(sorted(conditions | action_signature))[:8]}",
                        conditions,
                        action_signature,
                        f"{resource}_gain",
                        ep,
                    )
                )

        if (
            (not ep.action.reversible or action_features & DANGEROUS)
            and (ep.outcome.harm > 0.1 or ep.outcome.surprise > 0.35 or not ep.outcome.success)
        ):
            conditions = self._salient(obs_features, ("unknown", "uncertain", "low", "confidence", "has:"))
            action_signature = (action_features & DANGEROUS) | {
                f for f in action_features if f.startswith("action:")
            }
            names.append(
                self._upsert(
                    f"avoid_irreversible_uncertain_{stable_hash(sorted(conditions | action_signature))[:8]}",
                    conditions,
                    action_signature,
                    "irreversible_risk",
                    ep,
                )
            )

        if ep.outcome.terminal or ep.outcome.harm > 0.55:
            conditions = self._salient(
                obs_features,
                ("hostile", "hazard", "adjacent", "low", "threat", "danger", "has:entities"),
            )
            action_signature = self._salient(
                action_features,
                ("action:", "tag:movement", "tag:attack", "tag:unknown"),
            )
            names.append(
                self._upsert(
                    f"avoid_terminal_hazard_{stable_hash(sorted(conditions | action_signature))[:8]}",
                    conditions,
                    action_signature,
                    "terminal_hazard",
                    ep,
                )
            )

        if ep.outcome.success and ep.outcome.reward > max(0.2, ep.outcome.harm + 0.25 * ep.outcome.surprise):
            conditions = self._salient(obs_features, ("available", "tool", "affordance", "resource", "has:"))
            action_signature = self._salient(action_features, ("action:", "tag:"))
            names.append(
                self._upsert(
                    f"prefer_affordance_{stable_hash(sorted(conditions | action_signature))[:8]}",
                    conditions,
                    action_signature,
                    "positive_affordance",
                    ep,
                )
            )

        return names

    def _upsert(
        self,
        name: str,
        condition_features: set[str],
        action_features: set[str],
        effect: str,
        ep: Episode,
    ) -> str:
        principle = self.principles.get(name)
        if not principle:
            principle = Principle(name, set(condition_features), set(action_features), effect)
            self.principles[name] = principle
        else:
            principle.condition_features = (principle.condition_features & condition_features) or set(
                list(principle.condition_features | condition_features)[:32]
            )
            principle.action_features = (principle.action_features & action_features) or set(
                list(principle.action_features | action_features)[:24]
            )
        principle.update(ep, True)
        return name

    @staticmethod
    def _salient(features: set[str], prefer: Sequence[str], limit: int = 16) -> set[str]:
        scored = []
        for feature in features:
            score = 3 if any(token in feature for token in prefer) else 0
            if feature.startswith(("domain:", "source:")):
                score -= 2
            if len(feature) > 80:
                score -= 1
            scored.append((score, feature))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return {feature for score, feature in scored[:limit] if score >= 0 or feature.startswith("has:")}

    def predict(self, obs: Observation, act: ActionCandidate) -> dict[str, Any]:
        matches = []
        risk = 0.05 + 0.08 * act.authority_tier + (0.12 if not act.reversible else 0.0)
        reward = 0.0
        mass = 0.0
        for principle in self.principles.values():
            applicability = principle.applies_to(obs, act)
            if applicability < 0.12:
                continue
            weight = applicability * max(0.01, principle.confidence)
            mass += weight
            reward += weight * principle.reward_mean
            if principle.effect.endswith("loss") or "hazard" in principle.effect or "risk" in principle.effect:
                risk += weight * (0.35 + principle.harm_mean)
            elif "positive" in principle.effect or principle.effect.endswith("gain"):
                reward += 0.2 * weight
                risk -= 0.08 * weight
            matches.append(
                {
                    "principle": principle.name,
                    "effect": principle.effect,
                    "applicability": round(applicability, 4),
                    "confidence": round(principle.confidence, 4),
                    "domains_seen": sorted(principle.domains_seen),
                }
            )

        for feature in obs.features() | act.features():
            stats = self.feature_effect_stats.get(feature)
            if stats and stats["support"] > 0:
                n = stats["support"]
                risk += 0.08 * (stats["harm"] / n) + 0.03 * (stats["surprise"] / n)
                reward += 0.06 * (stats["reward"] / n)

        if obs.domain not in self.domain_feature_memory:
            risk += 0.10
        if act.features() & DANGEROUS:
            risk += 0.16

        return {
            "risk": clamp(risk),
            "expected_reward": max(-1.0, min(1.0, reward)),
            "confidence": clamp(0.25 + math.tanh(mass) * 0.55 + 0.1 * min(len(matches), 3)),
            "matches": sorted(matches, key=lambda m: (-m["applicability"], -m["confidence"]))[:8],
            "prediction_id": stable_hash(
                {"obs": obs.observation_id, "act": act.action_id, "risk": risk},
                prefix="pred_",
            ),
        }

    def rank_actions(
        self,
        observation: Observation | Mapping[str, Any],
        actions: Sequence[ActionCandidate | Mapping[str, Any]],
        *,
        risk_tolerance: float = 0.55,
    ) -> ActionDecision:
        obs = observation if isinstance(observation, Observation) else Observation(**dict(observation))
        acts = [a if isinstance(a, ActionCandidate) else ActionCandidate(**dict(a)) for a in actions]
        ranking = []
        for action in acts:
            pred = self.predict(obs, action)
            utility = pred["expected_reward"] - pred["risk"] - 0.15 * action.expected_cost
            if action.reversible and pred["confidence"] < 0.55:
                utility += 0.08
            ranking.append(
                {
                    "action": action.to_dict(),
                    "risk": pred["risk"],
                    "expected_reward": pred["expected_reward"],
                    "confidence": pred["confidence"],
                    "utility": utility,
                    "matches": pred["matches"],
                    "prediction_id": pred["prediction_id"],
                }
            )
        ranking.sort(key=lambda r: (r["utility"], -r["risk"]), reverse=True)
        acceptable = [r for r in ranking if r["risk"] <= risk_tolerance]
        selected = ActionCandidate(**acceptable[0]["action"]) if acceptable else None
        explanation = (
            f"Selected {selected.kind}: utility={acceptable[0]['utility']:.3f}, "
            f"risk={acceptable[0]['risk']:.3f}, confidence={acceptable[0]['confidence']:.3f}."
            if selected
            else "No action accepted: all candidates exceed risk tolerance."
        )
        receipt = {
            "receipt_id": stable_hash(
                {"obs": obs.to_dict(), "ranking": ranking, "tol": risk_tolerance, "ts": round(time.time(), 3)},
                prefix="zst_",
            ),
            "engine": "ZeroShotTransferEngine",
        }
        return ActionDecision(
            selected,
            ranking,
            ranking[0]["risk"] if ranking else 1.0,
            ranking[0]["confidence"] if ranking else 0.0,
            explanation,
            receipt,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "episodes": [e.to_dict() for e in self.episodes[-2000:]],
            "principles": [p.to_dict() for p in self.principles.values()],
            "feature_effect_stats": dict(self.feature_effect_stats),
            "domain_feature_memory": {k: dict(v) for k, v in self.domain_feature_memory.items()},
        }

    def save(self, path: str | Path | None = None) -> None:
        target = Path(path or self.state_path)
        atomic_write_text(target, json.dumps(self.to_dict(), indent=2, sort_keys=True))

    def load(self, path: str | Path) -> None:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.episodes = []
        for raw in data.get("episodes", []):
            self.episodes.append(
                Episode(
                    Observation(**raw["observation"]),
                    ActionCandidate(**raw["action"]),
                    raw.get("predicted", {}),
                    Outcome(**raw["outcome"]),
                    raw.get("episode_id", ""),
                )
            )
        self.principles = {}
        for raw in data.get("principles", []):
            principle = Principle(
                raw["name"],
                set(raw.get("condition_features", [])),
                set(raw.get("action_features", [])),
                raw["effect"],
                raw.get("support", 0),
                raw.get("contradictions", 0),
                raw.get("reward_mean", 0.0),
                raw.get("harm_mean", 0.0),
                raw.get("confidence", 0.0),
                set(raw.get("domains_seen", [])),
                list(raw.get("examples", [])),
            )
            self.principles[principle.name] = principle
        self.feature_effect_stats = defaultdict(lambda: {"support": 0.0, "harm": 0.0, "reward": 0.0, "surprise": 0.0})
        for k, v in data.get("feature_effect_stats", {}).items():
            self.feature_effect_stats[k].update(v)
        self.domain_feature_memory = defaultdict(Counter)
        for k, v in data.get("domain_feature_memory", {}).items():
            self.domain_feature_memory[k].update(v)
