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
    """Deterministic serialization — the basis of every ID in this module.

    Two properties this has to hold, both of which it previously did not:

    * **Sets are unordered**, so serializing one in iteration order gave the
      same logical set different JSON — and therefore different IDs — across
      processes. Set elements are sorted by their canonical form.
    * **Mapping keys are sorted BEFORE stringification**, which raised
      TypeError for incomparable key types (``{1: ..., "a": ...}``) and let two
      distinct keys with the same string form silently collapse into one.
      Keys are stringified first, sorted on that, and a collision is an error
      rather than a silent overwrite.
    """
    def clean(v: Any, _seen: frozenset[int] = frozenset()) -> Any:
        # A self-referential mapping or list — which a caller can assemble by
        # accident and which arrives here from arbitrary observation state —
        # recursed until the stack ran out, INSIDE Observation.__post_init__.
        # The reflex path constructs an Observation from whatever it was
        # handed, so the crash landed there.
        if isinstance(v, (Mapping, list, tuple, set, frozenset)):
            if id(v) in _seen:
                return "<cycle>"
            _seen = _seen | {id(v)}
        if isinstance(v, Mapping):
            cleaned: dict[str, Any] = {}
            for key in v:
                skey = str(key)
                if skey in cleaned:
                    raise ValueError(
                        f"canonical_json: mapping keys collapse to duplicate "
                        f"string form {skey!r}; identity would be ambiguous"
                    )
                cleaned[skey] = clean(v[key], _seen)
            return {k: cleaned[k] for k in sorted(cleaned)}
        if isinstance(v, (set, frozenset)):
            # Sort by canonical form so element order cannot leak into the hash.
            return sorted((clean(x, _seen) for x in v), key=lambda x: json.dumps(
                x, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str))
        if isinstance(v, (list, tuple)):
            return [clean(x, _seen) for x in v]
        if isinstance(v, float):
            if math.isnan(v) or math.isinf(v):
                return str(v)
            return round(v, 8)
        return v

    return json.dumps(clean(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_hash(value: Any, *, prefix: str = "") -> str:
    return prefix + hashlib.blake2b(canonical_json(value).encode("utf-8"), digest_size=16).hexdigest()


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp to [lo, hi], rejecting non-finite input.

    ``max(lo, min(hi, nan))`` returns **hi** in CPython, because every
    comparison with NaN is False. A NaN confidence therefore became MAXIMUM
    confidence — invalid telemetry promoted to the strongest possible signal,
    and then baked into content hashes. Non-finite input now yields the low
    bound: unknown is not certainty.
    """
    try:
        value = float(x)
    except (TypeError, ValueError):
        return lo
    if not math.isfinite(value):
        return lo
    return max(lo, min(hi, value))


def jaccard(a: set[str] | Sequence[str], b: set[str] | Sequence[str]) -> float:
    """Jaccard similarity, with empty-vs-empty scored as NO evidence.

    Returning 1.0 for two empty feature sets made a principle with no features
    match every action and observation it was ever compared against — a
    universal matcher created by an absence of information.
    """
    aa, bb = set(a), set(b)
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
        # An ID is a CONTENT BINDING, not a label. A caller-supplied id used to
        # be accepted verbatim, so any record could claim another record's
        # identity and substitute its provenance. The id is always recomputed;
        # a supplied one must match or it is rejected outright.
        computed = stable_hash(
            {
                "domain": self.domain,
                "state": self.state,
                "ts": round(self.timestamp, 3),
                "source": self.source,
            },
            prefix="obs_",
        )
        if self.observation_id and self.observation_id != computed:
            raise ValueError(
                f"observation_id does not bind this content "
                f"(supplied={self.observation_id!r}, computed={computed!r})"
            )
        object.__setattr__(self, "observation_id", computed)

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

    def __post_init__(self) -> None:
        # Nothing validated these fields, so a malformed candidate reached the
        # authority gate carrying whatever the caller happened to pass.
        if not str(self.kind or "").strip():
            raise ValueError("ActionCandidate.kind must be a non-empty string")
        if not isinstance(self.params, Mapping):
            raise TypeError("ActionCandidate.params must be a mapping")
        try:
            tier = int(self.authority_tier)
        except (TypeError, ValueError) as exc:
            raise ValueError("ActionCandidate.authority_tier must be an integer") from exc
        if tier < 0:
            raise ValueError("ActionCandidate.authority_tier must be non-negative")
        object.__setattr__(self, "authority_tier", tier)
        # A non-finite or negative cost must not be able to make an expensive
        # action look free to a cost-ranked selector.
        cost = self.expected_cost
        try:
            cost = float(cost)
        except (TypeError, ValueError):
            cost = 1.0
        if not math.isfinite(cost) or cost < 0.0:
            cost = 1.0
        object.__setattr__(self, "expected_cost", cost)
        object.__setattr__(self, "reversible", bool(self.reversible))
        object.__setattr__(self, "tags", tuple(str(t) for t in (self.tags or ())))

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

    def __post_init__(self) -> None:
        # These drive utility, ranking, and learning. Unbounded or non-finite
        # caller assertions could make one episode dominate every comparison,
        # and NaN would poison every mean it entered.
        object.__setattr__(self, "success", bool(self.success))
        object.__setattr__(self, "terminal", bool(self.terminal))
        for name in ("reward", "harm", "surprise"):
            raw = getattr(self, name)
            try:
                value = float(raw)
            except (TypeError, ValueError):
                value = 0.0
            if not math.isfinite(value):
                value = 0.0
            object.__setattr__(self, name, max(-1.0, min(1.0, value)))
        deltas = self.resources_delta if isinstance(self.resources_delta, Mapping) else {}
        cleaned: dict[str, float] = {}
        for key, raw in deltas.items():
            try:
                value = float(raw)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                cleaned[str(key)] = value
        object.__setattr__(self, "resources_delta", cleaned)
        object.__setattr__(self, "notes", str(self.notes or "")[:2000])

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
        # `predicted` participates in the identity: the episode IS the
        # evaluation of a specific forecast against a specific event, so two
        # different forecasts judged on the same event are two different
        # episodes and must not share one receipt id.
        computed = stable_hash(
            {
                "obs": self.observation.to_dict(),
                "action": self.action.to_dict(),
                "predicted": self.predicted,
                "outcome": self.outcome.to_dict(),
                "ts": round(self.created_at, 3),
            },
            prefix="ep_",
        )
        if self.episode_id and self.episode_id != computed:
            raise ValueError(
                f"episode_id does not bind this content "
                f"(supplied={self.episode_id!r}, computed={computed!r})"
            )
        object.__setattr__(self, "episode_id", computed)

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
    #: Episode ids already counted, so one outcome cannot be replayed into
    #: unlimited support. Bounded so a long-lived principle does not grow
    #: without limit; the bound is far above the support any real principle
    #: accumulates before it is acted on.
    counted_episodes: set[str] = field(default_factory=set)
    max_counted_episodes: int = 10_000

    def update(self, episode: Episode, matched: bool = True) -> bool:
        """Fold one episode in. Returns False if it was already counted.

        Without deduplication the same episode could be submitted repeatedly
        and each replay incremented support, so a single outcome could
        manufacture arbitrary confidence in a principle.
        """
        episode_id = getattr(episode, "episode_id", "") or ""
        if episode_id:
            if episode_id in self.counted_episodes:
                return False
            self.counted_episodes.add(episode_id)
            if len(self.counted_episodes) > self.max_counted_episodes:
                self.counted_episodes.pop()
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
        return True

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
