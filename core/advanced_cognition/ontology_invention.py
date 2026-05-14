"""OOD ontology invention for unfamiliar environments."""
from __future__ import annotations

import itertools
import json
import re
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from core.runtime.atomic_writer import atomic_write_text

from .schemas import Observation, clamp, jaccard, stable_hash

TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{1,48}|[@#$%&*!?<>/\-+=]+")


@dataclass
class OntologyEntityType:
    name: str
    evidence_keys: set[str] = field(default_factory=set)
    examples: list[str] = field(default_factory=list)
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "evidence_keys": sorted(self.evidence_keys),
            "examples": self.examples,
            "confidence": self.confidence,
        }


@dataclass
class OntologyRelation:
    source_type: str
    relation: str
    target_type: str
    support: int = 0
    confidence: float = 0.0
    evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AffordanceHypothesis:
    name: str
    preconditions: set[str]
    action_kind: str
    expected_effect: str
    support: int = 0
    confidence: float = 0.0
    tests: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "preconditions": sorted(self.preconditions),
            "action_kind": self.action_kind,
            "expected_effect": self.expected_effect,
            "support": self.support,
            "confidence": self.confidence,
            "tests": self.tests,
        }


@dataclass
class ExperimentProposal:
    name: str
    purpose: str
    minimal_action: dict[str, Any]
    expected_information_gain: float
    safety_notes: str
    reversible: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class OntologyModel:
    model_id: str
    domain: str
    entity_types: dict[str, OntologyEntityType]
    variables: dict[str, dict[str, Any]]
    relations: list[OntologyRelation]
    affordances: list[AffordanceHypothesis]
    hidden_state_hypotheses: list[dict[str, Any]]
    experiments: list[ExperimentProposal]
    confidence: float
    invented_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "domain": self.domain,
            "entity_types": {k: v.to_dict() for k, v in self.entity_types.items()},
            "variables": self.variables,
            "relations": [r.to_dict() for r in self.relations],
            "affordances": [a.to_dict() for a in self.affordances],
            "hidden_state_hypotheses": self.hidden_state_hypotheses,
            "experiments": [e.to_dict() for e in self.experiments],
            "confidence": self.confidence,
            "invented_at": self.invented_at,
        }


class OntologyInventionEngine:
    """Invents provisional ontologies and reversible experiments."""

    def __init__(self, *, state_path: str | Path | None = None):
        self.state_path = Path(state_path) if state_path else None
        self.models: dict[str, OntologyModel] = {}
        self.residuals: dict[str, list[dict[str, Any]]] = defaultdict(list)
        if self.state_path and self.state_path.exists():
            self.load(self.state_path)

    def ingest(self, observations: Sequence[Observation | Mapping[str, Any]]) -> OntologyModel:
        obs = [o if isinstance(o, Observation) else Observation(**dict(o)) for o in observations]
        if not obs:
            raise ValueError("ingest requires observations")
        domain = Counter(o.domain for o in obs).most_common(1)[0][0]
        entities = self._entities(obs)
        variables = self._variables(obs)
        relations = self._relations(obs, entities)
        affordances = self._affordances(obs)
        hidden = self._hidden(variables)
        experiments = self._experiments(affordances, hidden)
        confidence = self._score(obs, entities, variables, relations, affordances)
        model_id = stable_hash(
            {
                "domain": domain,
                "entities": {k: v.to_dict() for k, v in entities.items()},
                "vars": variables,
                "rels": [r.to_dict() for r in relations],
            },
            prefix="ont_",
        )
        model = OntologyModel(
            model_id,
            domain,
            entities,
            variables,
            relations,
            affordances,
            hidden,
            experiments,
            confidence,
        )
        self.models[domain] = model
        if self.state_path:
            self.save(self.state_path)
        return model

    def update_from_prediction_error(
        self,
        domain: str,
        *,
        predicted: Mapping[str, Any],
        actual: Mapping[str, Any],
        observation: Observation | Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        obs = observation if isinstance(observation, Observation) else (Observation(**dict(observation)) if observation else None)
        diff = {
            str(k): {"predicted": predicted.get(k), "actual": actual.get(k)}
            for k in set(predicted) | set(actual)
            if predicted.get(k) != actual.get(k)
        }
        residual = {
            "residual_id": stable_hash({"d": domain, "p": predicted, "a": actual, "ts": time.time()}, prefix="res_"),
            "domain": domain,
            "diff": diff,
            "observation": obs.to_dict() if obs else None,
            "severity": min(1.0, len(diff) / 10),
            "created_at": time.time(),
        }
        self.residuals[domain].append(residual)
        revision = None
        if obs and (len(self.residuals[domain]) >= 3 or residual["severity"] >= 0.4):
            revision = self.ingest([obs]).to_dict()
        return {"residual": residual, "revision": revision}

    def _entities(self, obs: list[Observation]) -> dict[str, OntologyEntityType]:
        out: dict[str, OntologyEntityType] = {}
        for observation in obs:
            for path, value in self._flat(observation.state):
                candidates: list[str] = []
                key = path.split(".")[-1].lower()
                if isinstance(value, Mapping):
                    candidates.append(str(value.get("type") or value.get("kind") or key.rstrip("s")))
                elif isinstance(value, list):
                    candidates.append(key.rstrip("s") or "item")
                    candidates.extend(
                        str(i.get("type") or i.get("kind") or key.rstrip("s"))
                        for i in value[:16]
                        if isinstance(i, Mapping)
                    )
                elif isinstance(value, str) and key in {"type", "kind", "role", "class"}:
                    candidates.append(value)
                for candidate in candidates:
                    name = re.sub(r"[^a-zA-Z0-9_]+", "_", candidate.lower()).strip("_")[:48]
                    if len(name) < 2:
                        continue
                    entity = out.setdefault(name, OntologyEntityType(name))
                    entity.evidence_keys.add(path)
                    if len(entity.examples) < 8:
                        entity.examples.append(stable_hash({"p": path, "v": str(value)[:120]}, prefix="ex_"))
                    entity.confidence = clamp(0.25 + 0.1 * len(entity.evidence_keys) + 0.05 * len(entity.examples))
        return out or {"unknown_entity": OntologyEntityType("unknown_entity", {"root"}, ["ex_root"], 0.2)}

    def _variables(self, obs: list[Observation]) -> dict[str, dict[str, Any]]:
        values: dict[str, list[Any]] = defaultdict(list)
        for observation in obs:
            for path, value in self._flat(observation.state):
                if isinstance(value, (int, float, bool, str)) or value is None:
                    values[path].append(value)
        out: dict[str, dict[str, Any]] = {}
        for path, items in values.items():
            if all(isinstance(x, (int, float, bool)) for x in items if x is not None):
                nums = [float(x) for x in items if x is not None]
                mean = sum(nums) / len(nums) if nums else 0.0
                variance = sum((x - mean) ** 2 for x in nums) / len(nums) if nums else 0.0
                out[path] = {
                    "kind": "numeric",
                    "stats": {
                        "count": len(items),
                        "min": min(nums) if nums else 0,
                        "max": max(nums) if nums else 0,
                        "mean": mean,
                        "variance": variance,
                    },
                    "confidence": clamp(0.2 + len(items) / max(8, len(obs) * 2)),
                }
            else:
                counts = Counter(str(x).lower() for x in items if x is not None)
                out[path] = {
                    "kind": "categorical",
                    "stats": {"count": len(items), "top_values": counts.most_common(8), "unique": len(counts)},
                    "confidence": clamp(0.2 + len(items) / max(8, len(obs) * 2)),
                }
        return out

    def _relations(
        self,
        obs: list[Observation],
        entities: dict[str, OntologyEntityType],
    ) -> list[OntologyRelation]:
        names = list(entities) or ["unknown_entity"]
        counts: Counter[tuple[str, str, str]] = Counter()
        evidence: dict[tuple[str, str, str], list[str]] = defaultdict(list)
        for observation in obs:
            features = sorted(observation.features())[:80]
            for a, b in itertools.combinations(features, 2):
                joined = a + b
                relation = (
                    "near"
                    if "adjacent" in joined
                    else "contains"
                    if "inventory" in joined or "contains" in joined
                    else "threatens"
                    if any(w in joined for w in ("hostile", "threat", "danger"))
                    else "affects_resource"
                    if any(w in joined for w in ("resource", "health", "energy"))
                    else None
                )
                if relation:
                    source = self._closest(a, names)
                    target = self._closest(b, names)
                    key = (source, relation, target)
                    counts[key] += 1
                    if len(evidence[key]) < 5:
                        evidence[key].append(observation.observation_id)
        return [
            OntologyRelation(source, relation, target, count, clamp(0.25 + count / max(4, len(obs))), evidence[(source, relation, target)])
            for (source, relation, target), count in counts.most_common(24)
        ]

    def _affordances(self, obs: list[Observation]) -> list[AffordanceHypothesis]:
        words: Counter[str] = Counter()
        for observation in obs:
            for feature in observation.features():
                if any(w in feature for w in ("available", "button", "link", "door", "tool", "prompt", "modal")):
                    words[feature] += 1
        affordances = []
        for feature, count in words.most_common(16):
            action = (
                "resolve_prompt"
                if "prompt" in feature or "modal" in feature
                else "activate_affordance"
                if "button" in feature or "link" in feature or "door" in feature
                else "probe"
            )
            affordances.append(
                AffordanceHypothesis(
                    "afford_" + stable_hash(feature)[:8],
                    {feature},
                    action,
                    "information_gain" if action == "probe" else "state_change",
                    count,
                    clamp(0.2 + count / max(5, len(obs))),
                    [f"Use reversible {action}; compare semantic diff."],
                )
            )
        return affordances or [
            AffordanceHypothesis(
                "safe_probe",
                {"unknown_environment"},
                "observe_or_probe",
                "information_gain",
                len(obs),
                0.25,
                ["Run observe/no-op before irreversible action."],
            )
        ]

    def _hidden(self, variables: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
        hidden = []
        for path, spec in variables.items():
            stats = spec.get("stats", {})
            if spec.get("kind") == "numeric" and stats.get("variance", 0) > 0.2:
                hidden.append(
                    {
                        "name": "latent_driver_for_" + path.replace(".", "_"),
                        "reason": "high numeric variance",
                        "observable_proxy": path,
                        "confidence": clamp(0.25 + stats.get("variance", 0) / 2),
                    }
                )
            if spec.get("kind") == "categorical" and stats.get("unique", 0) > 4:
                hidden.append(
                    {
                        "name": "latent_mode_for_" + path.replace(".", "_"),
                        "reason": "many categories suggest modes",
                        "observable_proxy": path,
                        "confidence": 0.35,
                    }
                )
        return hidden or [{"name": "unknown_ruleset", "reason": "insufficient evidence", "observable_proxy": "prediction_error", "confidence": 0.3}]

    def _experiments(
        self,
        affordances: list[AffordanceHypothesis],
        hidden: list[dict[str, Any]],
    ) -> list[ExperimentProposal]:
        experiments = []
        for affordance in affordances[:8]:
            experiments.append(
                ExperimentProposal(
                    "test_" + affordance.name,
                    f"Validate {affordance.action_kind}->{affordance.expected_effect}",
                    {"kind": affordance.action_kind, "params": {"probe": True}, "reversible": True},
                    clamp(0.35 + (1 - affordance.confidence) * 0.45),
                    "Reversible/no-op first; require gate if side effects appear.",
                )
            )
        for hypothesis in hidden[:6]:
            experiments.append(
                ExperimentProposal(
                    "resolve_" + hypothesis["name"],
                    "Reduce hidden-state uncertainty: " + hypothesis["reason"],
                    {"kind": "observe_compare", "params": {"proxy": hypothesis["observable_proxy"]}, "reversible": True},
                    clamp(0.45 + hypothesis.get("confidence", 0.3) * 0.3),
                    "Observation-only.",
                )
            )
        return sorted(experiments, key=lambda e: e.expected_information_gain, reverse=True)[:12]

    def _score(
        self,
        obs: list[Observation],
        entities: dict[str, OntologyEntityType],
        variables: dict[str, dict[str, Any]],
        relations: list[OntologyRelation],
        affordances: list[AffordanceHypothesis],
    ) -> float:
        structure = len(entities) + 0.5 * len(variables) + 0.7 * len(relations) + 0.6 * len(affordances)
        evidence = sum(e.confidence for e in entities.values()) + sum(v.get("confidence", 0) for v in variables.values())
        complexity_penalty = max(0.0, structure - max(6, len(obs) * 4)) * 0.03
        return clamp(0.15 + evidence / max(4, structure + 1) - complexity_penalty)

    @staticmethod
    def _flat(value: Any, prefix: str = "") -> list[tuple[str, Any]]:
        out = []
        if isinstance(value, Mapping):
            for key, child in value.items():
                path = f"{prefix}.{key}" if prefix else str(key)
                out.append((path, child))
                out += OntologyInventionEngine._flat(child, path)
        elif isinstance(value, list):
            out.append((prefix or "list", value))
            for i, child in enumerate(value[:24]):
                out += OntologyInventionEngine._flat(child, f"{prefix}[{i}]")
        else:
            out.append((prefix or "value", value))
        return out

    @staticmethod
    def _closest(feature: str, names: list[str]) -> str:
        tokens = set(TOKEN_RE.findall(feature.lower()))
        return max(names, key=lambda e: jaccard(tokens, set(e.split("_")))) if names else "unknown_entity"

    def save(self, path: str | Path | None = None) -> None:
        target = Path(path or self.state_path)
        payload = {"models": {k: v.to_dict() for k, v in self.models.items()}, "residuals": dict(self.residuals)}
        atomic_write_text(target, json.dumps(payload, indent=2, sort_keys=True))

    def load(self, path: str | Path) -> None:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.residuals = defaultdict(list, {k: list(v) for k, v in data.get("residuals", {}).items()})
