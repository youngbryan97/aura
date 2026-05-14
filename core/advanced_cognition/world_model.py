"""Learned multi-domain outcome model for consequence prediction."""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

from core.runtime.atomic_writer import atomic_write_text

from .schemas import ActionCandidate, Episode, Observation, Outcome, clamp, jaccard, stable_hash


@dataclass
class OutcomePrediction:
    prediction_id: str
    domain: str
    action_kind: str
    risk: float
    expected_reward: float
    uncertainty: float
    irreversible_risk: float
    social_risk: float
    resource_risk: float
    evidence_count: int
    similar_episodes: list[str] = field(default_factory=list)
    causal_factors: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MultiDomainWorldModel:
    """A small learned causal/outcome model backed by Aura's own episodes.

    It is intentionally model-agnostic: observations/actions become feature
    sets; outcomes update per-feature effect statistics and kNN memories.
    That makes it useful before heavyweight neural training is available while
    still producing real, calibrated predictions from experience.
    """

    def __init__(self, *, state_path: str | Path | None = None, max_episodes: int = 20000):
        self.state_path = Path(state_path) if state_path else None
        self.max_episodes = max_episodes
        self.episodes: list[Episode] = []
        self.feature_stats: dict[str, dict[str, float]] = defaultdict(
            lambda: {"n": 0.0, "reward": 0.0, "harm": 0.0, "surprise": 0.0, "terminal": 0.0}
        )
        self.action_stats: dict[str, dict[str, float]] = defaultdict(
            lambda: {"n": 0.0, "reward": 0.0, "harm": 0.0, "surprise": 0.0, "terminal": 0.0}
        )
        self.domain_stats: dict[str, dict[str, float]] = defaultdict(
            lambda: {"n": 0.0, "reward": 0.0, "harm": 0.0, "surprise": 0.0, "terminal": 0.0}
        )
        self.causal_edges: Counter[tuple[str, str]] = Counter()
        if self.state_path and self.state_path.exists():
            self.load(self.state_path)

    def observe_episode(self, episode: Episode) -> OutcomePrediction:
        self.episodes.append(episode)
        if len(self.episodes) > self.max_episodes:
            self.episodes = self.episodes[-self.max_episodes :]
        outcome_features = episode.outcome.features()
        for feature in episode.features():
            self._update_stats(self.feature_stats[feature], episode.outcome)
            for outcome_feature in outcome_features:
                self.causal_edges[(feature, outcome_feature)] += 1
        self._update_stats(self.action_stats[episode.action.kind], episode.outcome)
        self._update_stats(self.domain_stats[episode.observation.domain], episode.outcome)
        if self.state_path:
            self.save(self.state_path)
        return self.predict(episode.observation, episode.action)

    def predict(self, observation: Observation, action: ActionCandidate) -> OutcomePrediction:
        features = observation.features() | action.features()
        nearest = self._nearest(features, limit=12)
        evidence_count = len(nearest)
        weighted_reward = 0.0
        weighted_harm = 0.0
        weighted_surprise = 0.0
        weighted_terminal = 0.0
        weight_sum = 0.0
        for similarity, episode in nearest:
            weight = max(0.05, similarity)
            weighted_reward += weight * episode.outcome.reward
            weighted_harm += weight * episode.outcome.harm
            weighted_surprise += weight * episode.outcome.surprise
            weighted_terminal += weight * (1.0 if episode.outcome.terminal else 0.0)
            weight_sum += weight

        if weight_sum:
            reward = weighted_reward / weight_sum
            harm = weighted_harm / weight_sum
            surprise = weighted_surprise / weight_sum
            terminal = weighted_terminal / weight_sum
        else:
            action_stat = self.action_stats.get(action.kind)
            domain_stat = self.domain_stats.get(observation.domain)
            reward = self._mean(action_stat, "reward") * 0.6 + self._mean(domain_stat, "reward") * 0.4
            harm = self._mean(action_stat, "harm") * 0.6 + self._mean(domain_stat, "harm") * 0.4
            surprise = self._mean(action_stat, "surprise") * 0.6 + self._mean(domain_stat, "surprise") * 0.4
            terminal = self._mean(action_stat, "terminal") * 0.6 + self._mean(domain_stat, "terminal") * 0.4

        feature_risk = 0.0
        causal_factors: dict[str, float] = {}
        for feature in features:
            stat = self.feature_stats.get(feature)
            if not stat or stat["n"] <= 0:
                continue
            contribution = 0.35 * self._mean(stat, "harm") + 0.25 * self._mean(stat, "surprise") + 0.4 * self._mean(stat, "terminal")
            if contribution > 0.05:
                causal_factors[feature] = round(contribution, 4)
            feature_risk += contribution / max(5.0, math.sqrt(len(features)))

        irreversible_risk = 0.35 if not action.reversible else 0.0
        if {"tag:delete", "tag:deploy", "tag:self_modify", "tag:network_post"} & action.features():
            irreversible_risk += 0.25
        social_risk = 0.0
        if observation.domain in {"social", "conversation", "email", "reddit"} or "tag:social" in action.features():
            social_risk = 0.25 + harm * 0.5 + surprise * 0.15
        resource_risk = harm * 0.5 + terminal * 0.3
        risk = clamp(0.08 + harm * 0.35 + surprise * 0.15 + terminal * 0.25 + feature_risk + irreversible_risk)
        uncertainty = clamp(0.75 / (1 + evidence_count) + 0.25 * surprise + (0.15 if observation.domain not in self.domain_stats else 0.0))
        prediction_id = stable_hash(
            {
                "domain": observation.domain,
                "action": action.kind,
                "risk": risk,
                "reward": reward,
                "nearest": [ep.episode_id for _, ep in nearest[:4]],
            },
            prefix="wm_",
        )
        return OutcomePrediction(
            prediction_id=prediction_id,
            domain=observation.domain,
            action_kind=action.kind,
            risk=risk,
            expected_reward=max(-1.0, min(1.0, reward - harm * 0.2)),
            uncertainty=uncertainty,
            irreversible_risk=clamp(irreversible_risk),
            social_risk=clamp(social_risk),
            resource_risk=clamp(resource_risk),
            evidence_count=evidence_count,
            similar_episodes=[ep.episode_id for _, ep in nearest[:8]],
            causal_factors=dict(sorted(causal_factors.items(), key=lambda kv: kv[1], reverse=True)[:12]),
        )

    def specialized_predictions(self, observation: Observation, action: ActionCandidate) -> dict[str, Any]:
        base = self.predict(observation, action)
        return {
            "code_world": {
                "breakage_risk": base.risk if observation.domain in {"code", "repo", "self_modification"} else base.risk * 0.5,
                "impacted_features": base.causal_factors,
            },
            "environment_world": {
                "death_or_terminal_risk": base.resource_risk,
                "uncertainty": base.uncertainty,
            },
            "tool_world": {
                "irreversibility": base.irreversible_risk,
                "side_effect_risk": max(base.irreversible_risk, base.risk),
            },
            "social_world": {
                "dismissal_or_trust_risk": base.social_risk,
                "should_use_restraint": base.social_risk > 0.35 or base.uncertainty > 0.55,
            },
            "self_world": {
                "degradation_risk": base.risk if "tag:self_modify" in action.features() else base.risk * 0.35,
                "needs_stability_check": base.risk > 0.45 or base.uncertainty > 0.55,
            },
        }

    def _nearest(self, features: set[str], *, limit: int) -> list[tuple[float, Episode]]:
        scored = [(jaccard(features, episode.features()), episode) for episode in self.episodes]
        scored = [item for item in scored if item[0] > 0.05]
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored[:limit]

    @staticmethod
    def _update_stats(stats: dict[str, float], outcome: Outcome) -> None:
        stats["n"] += 1.0
        stats["reward"] += outcome.reward
        stats["harm"] += outcome.harm
        stats["surprise"] += outcome.surprise
        stats["terminal"] += 1.0 if outcome.terminal else 0.0

    @staticmethod
    def _mean(stats: Mapping[str, float] | None, key: str) -> float:
        if not stats or stats.get("n", 0.0) <= 0:
            return 0.0
        return float(stats.get(key, 0.0)) / max(1.0, float(stats.get("n", 0.0)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "episodes": [episode.to_dict() for episode in self.episodes[-2000:]],
            "feature_stats": dict(self.feature_stats),
            "action_stats": dict(self.action_stats),
            "domain_stats": dict(self.domain_stats),
            "causal_edges": {"|".join(k): v for k, v in self.causal_edges.items()},
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
        self.feature_stats = defaultdict(lambda: {"n": 0.0, "reward": 0.0, "harm": 0.0, "surprise": 0.0, "terminal": 0.0})
        for key, value in data.get("feature_stats", {}).items():
            self.feature_stats[key].update(value)
        self.action_stats = defaultdict(lambda: {"n": 0.0, "reward": 0.0, "harm": 0.0, "surprise": 0.0, "terminal": 0.0})
        for key, value in data.get("action_stats", {}).items():
            self.action_stats[key].update(value)
        self.domain_stats = defaultdict(lambda: {"n": 0.0, "reward": 0.0, "harm": 0.0, "surprise": 0.0, "terminal": 0.0})
        for key, value in data.get("domain_stats", {}).items():
            self.domain_stats[key].update(value)
        self.causal_edges = Counter()
        for key, value in data.get("causal_edges", {}).items():
            a, b = key.split("|", 1)
            self.causal_edges[(a, b)] = int(value)
