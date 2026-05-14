"""Shared schemas for Aura's advanced cognition runtime.

These records are deliberately small, deterministic, and serializable.  They
form the common substrate for typed observations, action priors, outcome
memory, transfer principles, and receipts.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence


def canonical_json(value: Any) -> str:
    def clean(v: Any) -> Any:
        if isinstance(v, Mapping):
            return {str(k): clean(v[k]) for k in sorted(v)}
        if isinstance(v, (list, tuple, set)):
            return [clean(x) for x in v]
        if isinstance(v, float):
            if math.isnan(v) or math.isinf(v):
                return str(v)
            return round(v, 8)
        return v

    return json.dumps(clean(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_hash(value: Any, *, prefix: str = "") -> str:
    return prefix + hashlib.blake2b(canonical_json(value).encode("utf-8"), digest_size=16).hexdigest()


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(x)))


def jaccard(a: set[str] | Sequence[str], b: set[str] | Sequence[str]) -> float:
    aa, bb = set(a), set(b)
    if not aa and not bb:
        return 1.0
    if not aa or not bb:
        return 0.0
    return len(aa & bb) / len(aa | bb)


@dataclass(frozen=True)
class Observation:
    domain: str
    state: dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    confidence: float = 0.7
    source: str = "unknown"
    observation_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "confidence", clamp(self.confidence))
        if not self.observation_id:
            object.__setattr__(
                self,
                "observation_id",
                stable_hash(
                    {
                        "domain": self.domain,
                        "state": self.state,
                        "ts": round(self.timestamp, 3),
                        "source": self.source,
                    },
                    prefix="obs_",
                ),
            )

    def features(self) -> set[str]:
        out = {f"domain:{self.domain}", f"source:{self.source}"}

        def walk(prefix: str, value: Any, depth: int = 0) -> None:
            if depth > 4:
                out.add(prefix + "deep")
                return
            if isinstance(value, Mapping):
                for k, v in value.items():
                    key = str(k).lower().strip().replace(" ", "_")
                    out.add(f"has:{prefix}{key}")
                    walk(prefix + key + ".", v, depth + 1)
            elif isinstance(value, (list, tuple)):
                out.add(f"{prefix}count:{len(value)}")
                for item in value[:16]:
                    walk(prefix, item, depth + 1)
            elif isinstance(value, bool):
                out.add(f"{prefix}{value}")
            elif isinstance(value, (int, float)):
                x = float(value)
                bucket = (
                    "very_low"
                    if x <= -0.5
                    else "low"
                    if x < 0.25
                    else "mid"
                    if x < 0.75
                    else "high"
                    if x < 1.5
                    else "very_high"
                )
                out.add(f"{prefix}{bucket}")
            elif value is None:
                out.add(f"{prefix}none")
            else:
                text = str(value).lower().replace("/", " ").replace("_", " ").replace("-", " ")
                for tok in text.split()[:16]:
                    if len(tok) >= 2:
                        out.add(f"{prefix}tok:{tok[:32]}")

        walk("", self.state)
        return out

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ActionCandidate:
    action_id: str
    kind: str
    params: dict[str, Any] = field(default_factory=dict)
    reversible: bool = True
    authority_tier: int = 1
    expected_cost: float = 0.1
    tags: tuple[str, ...] = ()

    def features(self) -> set[str]:
        out = {f"action:{self.kind}", f"tier:{self.authority_tier}"}
        out |= {f"tag:{t}" for t in self.tags}
        out.add("action:reversible" if self.reversible else "action:irreversible")
        for k, v in self.params.items():
            out.add(f"param:{k}")
            if isinstance(v, str):
                out.add(f"param:{k}:{v.lower()[:32]}")
            elif isinstance(v, bool):
                out.add(f"param:{k}:{v}")
            elif isinstance(v, (int, float)):
                out.add(f"param:{k}:numeric")
        return out

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Outcome:
    success: bool
    reward: float = 0.0
    harm: float = 0.0
    surprise: float = 0.0
    resources_delta: dict[str, float] = field(default_factory=dict)
    terminal: bool = False
    notes: str = ""
    facts: dict[str, Any] = field(default_factory=dict)

    @property
    def utility(self) -> float:
        return float(self.reward) - float(self.harm) - 0.25 * float(self.surprise)

    def features(self) -> set[str]:
        out = {f"success:{self.success}", f"terminal:{self.terminal}"}
        if self.reward > 0.5:
            out.add("outcome:high_reward")
        if self.harm > 0.5:
            out.add("outcome:high_harm")
        if self.surprise > 0.5:
            out.add("outcome:surprising")
        for k, v in self.resources_delta.items():
            trend = "gain" if v > 0 else "loss" if v < 0 else "same"
            out.add(f"resource:{k}:{trend}")
        return out

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Episode:
    observation: Observation
    action: ActionCandidate
    predicted: dict[str, Any]
    outcome: Outcome
    episode_id: str = ""
    created_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if not self.episode_id:
            object.__setattr__(
                self,
                "episode_id",
                stable_hash(
                    {
                        "obs": self.observation.to_dict(),
                        "action": self.action.to_dict(),
                        "outcome": self.outcome.to_dict(),
                        "ts": round(self.created_at, 3),
                    },
                    prefix="ep_",
                ),
            )

    def features(self) -> set[str]:
        return self.observation.features() | self.action.features()

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation": self.observation.to_dict(),
            "action": self.action.to_dict(),
            "predicted": self.predicted,
            "outcome": self.outcome.to_dict(),
            "episode_id": self.episode_id,
            "created_at": self.created_at,
        }


@dataclass
class Principle:
    name: str
    condition_features: set[str]
    action_features: set[str]
    effect: str
    support: int = 0
    contradictions: int = 0
    reward_mean: float = 0.0
    harm_mean: float = 0.0
    confidence: float = 0.0
    domains_seen: set[str] = field(default_factory=set)
    examples: list[str] = field(default_factory=list)

    def update(self, episode: Episode, matched: bool = True) -> None:
        if matched:
            self.support += 1
            n = max(1, self.support)
            self.reward_mean += (episode.outcome.reward - self.reward_mean) / n
            self.harm_mean += (episode.outcome.harm - self.harm_mean) / n
            self.domains_seen.add(episode.observation.domain)
            if len(self.examples) < 8:
                self.examples.append(episode.episode_id)
        else:
            self.contradictions += 1
        self.confidence = clamp((self.support + 1.0) / (self.support + self.contradictions + 3.0))

    def applies_to(self, observation: Observation, action: ActionCandidate) -> float:
        return clamp(
            0.65 * jaccard(self.condition_features, observation.features())
            + 0.35 * jaccard(self.action_features, action.features())
            + (0.08 if observation.domain in self.domains_seen else 0.0)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "condition_features": sorted(self.condition_features),
            "action_features": sorted(self.action_features),
            "effect": self.effect,
            "support": self.support,
            "contradictions": self.contradictions,
            "reward_mean": self.reward_mean,
            "harm_mean": self.harm_mean,
            "confidence": self.confidence,
            "domains_seen": sorted(self.domains_seen),
            "examples": self.examples,
        }


@dataclass(frozen=True)
class ActionDecision:
    selected: ActionCandidate | None
    ranking: list[dict[str, Any]]
    risk: float
    confidence: float
    explanation: str
    receipt: dict[str, Any] = field(default_factory=dict)
