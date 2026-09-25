"""Source-disjoint concept discovery, revision, and contextual reuse.

Perception and knowledge stores may propose distinctions. Only independent
outcomes can validate one. This layer uses OntologyDiscovery for the experiment
and ConceptRegistry for cross-substrate identity; it defines neither another
language nor another truth store.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from functools import wraps
from pathlib import Path
from typing import Any

from core.brain.ontology_discovery import (
    CandidateLaw,
    Observation,
    OntologyDiscovery,
    Predicate,
    base_rate,
    candidate_predicates,
    score_split,
)
from core.cognition.concept_handle import BindingMethod, ConceptRegistry, Substrate
from core.evidence.packet import EvidenceKind, EvidencePacket
from core.language.contextual_usage import MeaningFeedback, UsageEvent
from core.runtime.lockdep import checked_lock, checked_thread_semaphore
from core.runtime.state_ownership import state_root

_MAX_FEATURES = 64
_MAX_CASES = 4096
_MAX_PROPOSALS = 256
_MAX_PRIMITIVES = 256
_MAX_PREDICTIONS = 4096
_MAX_USAGE = 2048
_MAX_BRANCHES = 128
_CHANNELS = frozenset({"observation", "corpus", "memory", "model", "mutation", "composition"})


def _serialized(method):
    @wraps(method)
    def call(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)
    return call


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str,
                                   separators=(",", ":")).encode("utf-8")).hexdigest()


def transition_features(
    before: Mapping[str, Any], after: Mapping[str, Any], *, action: str = ""
) -> dict[str, Any]:
    """Expose change and invariance without naming the concept in advance."""
    keys = sorted(set(before) | set(after))[:_MAX_FEATURES]
    features: dict[str, Any] = {}
    for key in keys:
        old, new = before.get(key), after.get(key)
        if type(old) not in (bool, int, float, str, type(None)) or type(new) not in (
            bool, int, float, str, type(None)
        ):
            continue
        features[f"present:{key}"] = key in after
        features[f"changed:{key}"] = old != new
        if key in before and key in after:
            features[f"invariant:{key}"] = old == new
        if type(old) in (int, float) and type(new) in (int, float):
            features[f"delta:{key}"] = float(new) - float(old)
        if type(new) in (bool, str):
            features[f"value:{key}"] = new
    if action:
        features["action"] = action
    return features


@dataclass(frozen=True, slots=True)
class SemanticCase:
    """One independent observation; a model prediction is never a witness."""

    source_id: str
    context_id: str
    outcome_name: str
    outcome: bool
    features: Mapping[str, Any]
    observed_at: float = field(default_factory=time.time)
    origin: str = "observation"
    intervention: str = ""
    candidate_dependencies: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if (not self.source_id or not self.context_id or not self.outcome_name
                or type(self.outcome) is not bool or not self.features
                or len(self.features) > _MAX_FEATURES
                or any(not isinstance(key, str) or not key or
                       type(value) not in (bool, int, float, str, type(None))
                       for key, value in self.features.items())):
            raise ValueError("semantic case needs identified, bounded observations")
        if any(type(value) in (int, float) and not math.isfinite(value)
               for value in self.features.values()):
            raise ValueError("semantic case has a nonfinite measurement")

    @property
    def identity(self) -> str:
        return _digest({"source": self.source_id, "context": self.context_id,
                        "outcome": self.outcome_name, "features": dict(self.features),
                        "intervention": self.intervention})

    @property
    def independent(self) -> bool:
        return self.origin == "observation" and not self.candidate_dependencies

    def to_dict(self) -> dict[str, Any]:
        return {"source_id": self.source_id, "context_id": self.context_id,
                "outcome_name": self.outcome_name, "outcome": self.outcome,
                "features": dict(self.features), "observed_at": self.observed_at,
                "origin": self.origin, "intervention": self.intervention,
                "candidate_dependencies": list(self.candidate_dependencies)}


@dataclass(frozen=True, slots=True)
class SemanticProposal:
    """A testable reading, including where it came from and what it assumes."""

    law: CandidateLaw
    channel: str
    reference: str
    context_id: str = ""
    dependencies: tuple[str, ...] = ()
    kind: str = "theoretical"
    assumptions: tuple[str, ...] = ()
    exposure_sources: tuple[str, ...] = ()
    expected_outcome: bool = True
    exposure_audited: bool = False

    def __post_init__(self) -> None:
        if (self.channel not in _CHANNELS or not self.reference
                or not self.law.predicates or len(self.law.predicates) > 3
                or self.kind not in {"theoretical", "strange"}
                or (self.kind == "strange" and not self.assumptions)
                or type(self.expected_outcome) is not bool):
            raise ValueError("semantic proposal lacks bounded origin or predicates")

    @property
    def identity(self) -> str:
        return _digest({"law": self.law.describe(), "channel": self.channel,
                        "reference": self.reference, "context": self.context_id,
                        "dependencies": self.dependencies, "kind": self.kind,
                        "assumptions": self.assumptions,
                        "exposure_sources": self.exposure_sources,
                        "expected_outcome": self.expected_outcome,
                        "exposure_audited": self.exposure_audited})


@dataclass(frozen=True, slots=True)
class PredictionCommitment:
    """A consequence fixed before a direct observation of the named source."""

    identity: str
    proposal_id: str
    source_id: str
    context_id: str
    outcome_name: str
    features_sha256: str
    expected_outcome: bool
    causal_path: tuple[str, ...]
    committed_at: float
    status: str = "pending"
    observed_outcome: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"identity": self.identity, "proposal_id": self.proposal_id,
                "source_id": self.source_id, "context_id": self.context_id,
                "outcome_name": self.outcome_name,
                "features_sha256": self.features_sha256,
                "expected_outcome": self.expected_outcome,
                "causal_path": list(self.causal_path),
                "committed_at": self.committed_at, "status": self.status,
                "observed_outcome": self.observed_outcome}


@dataclass(frozen=True, slots=True)
class SemanticSituation:
    """A possible observation with no claimed outcome yet."""

    source_id: str
    context_id: str
    features: Mapping[str, Any]
    cost: float = 1.0

    def __post_init__(self) -> None:
        if (not self.source_id or not self.context_id or not self.features
                or len(self.features) > _MAX_FEATURES or not math.isfinite(self.cost)
                or self.cost <= 0):
            raise ValueError("a proposed experiment needs a bounded situation and cost")


@dataclass(frozen=True, slots=True)
class SemanticPrimitive:
    """A measured reusable distinction with its source and counterevidence."""

    identity: str
    law: CandidateLaw
    receipt: Mapping[str, Any]
    source_ids: tuple[str, ...]
    contexts: tuple[str, ...]
    dependencies: tuple[str, ...] = ()
    status: str = "measured"
    revisions: tuple[str, ...] = ()
    validation_sources: tuple[str, ...] = ()
    concept_ids: tuple[str, ...] = ()
    trial_receipt_sha256: str = ""
    replication_receipts: tuple[str, ...] = ()

    @property
    def sparse_vector(self) -> tuple[str, ...]:
        return tuple(sorted(_digest(predicate.describe())[:16] for predicate in self.law.predicates))

    @property
    def precision(self) -> float:
        return float(self.receipt["evidence"]["transfer_precision"])

    def to_dict(self) -> dict[str, Any]:
        return {"identity": self.identity, "law": _law_dict(self.law),
                "receipt": dict(self.receipt), "source_ids": list(self.source_ids),
                "contexts": list(self.contexts), "dependencies": list(self.dependencies),
                "status": self.status, "revisions": list(self.revisions),
                "validation_sources": list(self.validation_sources),
                "concept_ids": list(self.concept_ids),
                "trial_receipt_sha256": self.trial_receipt_sha256,
                "replication_receipts": list(self.replication_receipts)}


def _law_dict(law: CandidateLaw) -> dict[str, Any]:
    return {"outcome_name": law.outcome_name,
            "predicates": [{"feature": item.feature, "op": item.op, "value": item.value}
                           for item in law.predicates]}


def _law_from_dict(payload: Mapping[str, Any]) -> CandidateLaw:
    return CandidateLaw(tuple(Predicate(**item) for item in payload["predicates"]),
                        payload["outcome_name"])


class SemanticDevelopment:
    """One bounded hypothesis cycle over direct evidence and existing organs."""

    def __init__(self, *, registry: ConceptRegistry | None = None,
                 concept_engine: Any = None, state_path: Path | None = None,
                 min_support: int = 8) -> None:
        self.registry = registry or ConceptRegistry()
        self.concept_engine = concept_engine
        self.state_path = Path(state_path) if state_path is not None else (
            state_root() / "data" / "cognition" / "semantic_development.json")
        self.min_support = min_support
        self._lock = checked_lock("core.cognition.semantic_development.state", reentrant=True)
        self._save_lane = checked_thread_semaphore(
            "core.cognition.semantic_development.save", budget_s=120.0)
        self.cases: dict[str, SemanticCase] = {}
        self.usage_events: dict[str, UsageEvent] = {}
        self.meaning_feedback: dict[str, MeaningFeedback] = {}
        self._usage_observed_count = 0
        self._case_slots: dict[tuple[str, str, str], str] = {}
        self.proposals: dict[str, SemanticProposal] = {}
        self.primitives: dict[str, SemanticPrimitive] = {}
        self.predictions: dict[str, PredictionCommitment] = {}
        self._pending_slots: dict[tuple[str, str, str], set[str]] = defaultdict(set)
        self._observed_count = 0
        self.trials: list[dict[str, Any]] = []
        self._validation_sources: set[str] = set()
        self._revision_cases: dict[str, list[SemanticCase]] = {}
        self.failed: list[dict[str, Any]] = []
        if self.state_path.exists():
            self.load()

    @_serialized
    def observe(self, case: SemanticCase) -> bool:
        """Retain a measured event once; a simulation stays proposal-only."""
        if not case.independent:
            raise ValueError("only direct observations enter semantic evidence")
        slot = (case.source_id, case.context_id, case.outcome_name)
        existing_id = self._case_slots.get(slot)
        if existing_id is not None:
            existing = self.cases[existing_id]
            if (existing.outcome, dict(existing.features), existing.intervention) != (
                    case.outcome, dict(case.features), case.intervention
            ):
                raise ValueError("one semantic observation slot changed")
            return False
        prior = self.cases.get(case.identity)
        if prior is not None:
            if prior != case:
                raise ValueError("one semantic evidence identity changed")
            return False
        if len(self.cases) >= _MAX_CASES:
            oldest = min(self.cases, key=lambda key: self.cases[key].observed_at)
            evicted = self.cases[oldest]
            self._case_slots.pop((evicted.source_id, evicted.context_id,
                                  evicted.outcome_name), None)
            del self.cases[oldest]
        self.cases[case.identity] = case
        self._case_slots[slot] = case.identity
        self._observed_count += 1
        for identity in self._pending_slots.pop(slot, set()):
            prediction = self.predictions[identity]
            status = ("invalidated_features" if prediction.features_sha256 != _digest(
                dict(case.features)) else "matched" if prediction.expected_outcome == case.outcome
                else "contradicted")
            self.predictions[identity] = replace(
                prediction, status=status, observed_outcome=case.outcome)
        return True

    @property
    def observation_count(self) -> int:
        with self._lock:
            return self._observed_count

    @_serialized
    def observe_usage(self, event: UsageEvent) -> bool:
        """Retain exposure without treating co-use or delivery as a definition."""
        prior = self.usage_events.get(event.source_id)
        if prior is not None:
            retry = (replace(event, observed_at=prior.observed_at)
                     if not event.cues and not prior.cues else event)
            if prior != retry:
                raise ValueError("one usage source changed its observation")
            return False
        if len(self.usage_events) >= _MAX_USAGE:
            oldest = min(self.usage_events,
                         key=lambda source: self.usage_events[source].observed_at)
            del self.usage_events[oldest]
            self.meaning_feedback = {source: feedback
                                     for source, feedback in self.meaning_feedback.items()
                                     if feedback.usage_source_id != oldest}
        self.usage_events[event.source_id] = event
        self._usage_observed_count += 1
        return True

    @property
    def usage_observation_count(self) -> int:
        with self._lock:
            return self._usage_observed_count

    @_serialized
    def observe_meaning_feedback(self, feedback: MeaningFeedback) -> bool:
        """Keep the correction separate from the use it interprets."""
        event = self.usage_events.get(feedback.usage_source_id)
        if event is None or feedback.term.casefold() not in (
                *event.terms, *event.referents):
            raise ValueError("meaning feedback has no retained matching usage")
        prior = self.meaning_feedback.get(feedback.source_id)
        if prior is not None:
            if prior != feedback:
                raise ValueError("one meaning feedback source changed its claim")
            return False
        if len(self.meaning_feedback) >= _MAX_USAGE:
            oldest = min(self.meaning_feedback,
                         key=lambda source: self.meaning_feedback[source].observed_at)
            del self.meaning_feedback[oldest]
        self.meaning_feedback[feedback.source_id] = feedback
        return True

    def _labeled_usage(self, term: str) -> tuple[tuple[UsageEvent, str], ...]:
        labels: dict[str, set[str]] = defaultdict(set)
        for feedback in self.meaning_feedback.values():
            if feedback.term.casefold() == term:
                labels[feedback.usage_source_id].add(feedback.sense)
        return tuple((self.usage_events[source], next(iter(senses)))
                     for source, senses in labels.items()
                     if len(senses) == 1 and source in self.usage_events)

    @staticmethod
    def _feature_measured(event: UsageEvent, feature: str, value: Any,
                          features: Mapping[str, Any]) -> bool:
        if feature.startswith(("cue:", "context:")):
            return feature in features
        if event.original_token_count > len(event.terms):
            if feature.startswith("co:"):
                return feature in features
            if feature in {"usage:spoken", "usage:frequency"}:
                return bool(value)
        return True

    @_serialized
    def usage_associations(
        self, term: str, *, setting: str = "", community: str = "", limit: int = 16,
    ) -> dict[str, Any]:
        """Compare observed co-use with its local base rate, not taxonomy."""
        if not 1 <= limit <= 64:
            raise ValueError("association limit is out of range")
        key = term.casefold()
        scoped = [event for event in self.usage_events.values()
                  if (not setting or event.setting == setting)
                  and (not community or event.community == community)]
        exposed = [event for event in scoped
                   if key in event.terms or key in event.referents]
        if not exposed:
            status = ("unmeasured_due_to_sampling" if any(
                event.original_token_count > len(event.terms) for event in scoped)
                else "unexposed")
            return {"status": status, "term": key, "observed_sources": 0,
                    "associations": (), "serving_authority": False}
        background = Counter(neighbor for event in scoped
                             for neighbor in set(event.terms))
        neighbors = Counter(neighbor for event in exposed
                            for neighbor in set(event.terms) - {key})
        associations = []
        for neighbor, support in neighbors.items():
            base = background[neighbor] / len(scoped)
            conditional = support / len(exposed)
            associations.append({"term": neighbor, "sources": support,
                                 "conditional_frequency": conditional,
                                 "background_frequency": base,
                                 "lift": conditional / base if base else 0.0,
                                 "relation": "observed_co_use"})
        associations.sort(key=lambda row: (-row["sources"], -row["lift"], row["term"]))
        cues = Counter((cue.channel, cue.name, str(cue.value))
                       for event in exposed for cue in event.cues)
        stretched = sum(key in event.stretched_terms for event in exposed)
        indirect = sum(key in event.referents and key not in event.terms
                       for event in exposed)
        return {"status": "observed_association", "term": key,
                "observed_sources": len(exposed),
                "partially_sampled_sources": sum(
                    event.original_token_count > len(event.terms) for event in exposed),
                "associations": tuple(associations[:limit]),
                "delivery_cues": tuple({"channel": channel, "name": name,
                                         "value": value, "sources": count}
                                        for (channel, name, value), count in cues.most_common(limit)),
                "stretched_sources": stretched, "indirect_sources": indirect,
                "serving_authority": False}

    @_serialized
    def contextual_senses(self, term: str, situation: UsageEvent, *,
                          excluded_sources: tuple[str, ...] = ()) -> dict[str, Any]:
        """Rank grounded readings by similar prior use; absence stays absence."""
        key = term.casefold()
        if key not in situation.terms and key not in situation.referents:
            raise ValueError("the situation does not contain the concept")
        excluded = {*excluded_sources, situation.source_id}
        labeled = [(event, sense) for event, sense in self._labeled_usage(key)
                   if event.source_id not in excluded]
        if not labeled:
            return {"status": "unexposed_to_grounded_sense", "term": key,
                    "candidates": (), "serving_authority": False}
        query = situation.features_for(key)
        grouped: dict[str, list[UsageEvent]] = defaultdict(list)
        for event, sense in labeled:
            grouped[sense].append(event)
        features_by_source = {
            event.source_id: event.features_for(key)
            for event, _sense in labeled
        }
        query_features = frozenset(
            (name, value) for name, value in query.items()
            if name != "usage:sampled"
            and self._feature_measured(situation, name, value, query))
        local = sum((not situation.setting or event.setting == situation.setting)
                    and (not situation.community or event.community == situation.community)
                    for event, _sense in labeled)
        candidates = []
        for sense, examples in grouped.items():
            positive_sources = {event.source_id for event in examples}
            negatives = [event for event, _label in labeled
                         if event.source_id not in positive_sources]
            score = math.log((len(examples) + 1) / (len(labeled) + len(grouped)))
            discriminators = []
            for feature in query_features:
                name, value = feature
                measured_positive = [features_by_source[event.source_id]
                                     for event in examples
                                     if self._feature_measured(
                                         event, name, value,
                                         features_by_source[event.source_id])]
                measured_negative = [features_by_source[event.source_id]
                                     for event in negatives
                                     if self._feature_measured(
                                         event, name, value,
                                         features_by_source[event.source_id])]
                if not measured_positive or not measured_negative:
                    continue
                positive = sum(item.get(name) == value for item in measured_positive)
                negative = sum(item.get(name) == value for item in measured_negative)
                contribution = math.log(
                    ((positive + 1) / (len(measured_positive) + 2)) /
                    ((negative + 1) / (len(measured_negative) + 2)))
                score += contribution
                if contribution > 0:
                    discriminators.append((feature[0], contribution))
            candidates.append({"sense": sense,
                               "independent_sources": len({event.source_id for event in examples}),
                               "evidence_score": score,
                               "supporting_features": tuple(name for name, _ in sorted(
                                   discriminators, key=lambda item: (-item[1], item[0]))[:8]),
                               "relation": "contrastive_abductive_context_fit"})
        candidates.sort(key=lambda row: (-row["evidence_score"],
                                         -row["independent_sources"], row["sense"]))
        tied = (len(candidates) > 1 and math.isclose(
            candidates[0]["evidence_score"], candidates[1]["evidence_score"],
            abs_tol=1e-9))
        status = ("unmeasured_context_transfer" if not local else
                  "no_discriminating_evidence" if tied else "ranked_hypotheses")
        return {"status": status,
                "term": key, "local_grounded_sources": local,
                "candidates": tuple(candidates), "serving_authority": False}

    @_serialized
    def evaluate_contextual_senses(self, term: str,
                                   heldout_sources: tuple[str, ...]) -> dict[str, Any]:
        """Measure interpretations with every held-out source excluded from fitting."""
        if not 1 <= len(heldout_sources) <= 256 or len(set(heldout_sources)) != len(
                heldout_sources):
            raise ValueError("sense evaluation needs distinct bounded held-out sources")
        labeled = dict((event.source_id, (event, sense))
                       for event, sense in self._labeled_usage(term.casefold()))
        if any(source not in labeled for source in heldout_sources):
            raise ValueError("held-out source lacks unambiguous attributed feedback")
        training_labels = Counter(sense for source, (_event, sense) in labeled.items()
                                  if source not in heldout_sources)
        if not training_labels:
            return {"status": "unmeasured_no_training_exposure", "n": 0,
                    "serving_authority": False}
        majority = max(training_labels, key=lambda sense: (training_labels[sense], sense))
        correct = baseline_correct = answered = 0
        by_community: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
        for source in heldout_sources:
            event, sense = labeled[source]
            result = self.contextual_senses(
                term, event, excluded_sources=heldout_sources)
            predicted = (result["candidates"][0]["sense"]
                         if result["status"] == "ranked_hypotheses" else None)
            answer = int(predicted is not None)
            hit = int(predicted == sense)
            answered += answer
            correct += hit
            baseline_correct += int(majority == sense)
            row = by_community[event.community or "unspecified"]
            row[0] += 1
            row[1] += answer
            row[2] += hit
        return {"status": "measured_development_only", "n": len(heldout_sources),
                "answered": answered, "correct": correct,
                "majority_baseline_correct": baseline_correct,
                "heldout_sources": heldout_sources,
                "by_community": {name: {"n": counts[0], "answered": counts[1],
                                        "correct": counts[2]}
                                 for name, counts in sorted(by_community.items())},
                "serving_authority": False}

    @_serialized
    def discriminating_usage_observations(self, term: str, situation: UsageEvent,
                                          *, limit: int = 8) -> dict[str, Any]:
        """Identify measured distinctions worth checking in an unresolved setting."""
        if not 1 <= limit <= 32:
            raise ValueError("discriminating observation limit is out of range")
        key = term.casefold()
        labeled = [(event, sense) for event, sense in self._labeled_usage(key)
                   if event.source_id != situation.source_id]
        if len({sense for _event, sense in labeled}) < 2:
            return {"status": "insufficient_rivals", "observations": (),
                    "serving_authority": False}
        grouped: dict[str, list[UsageEvent]] = defaultdict(list)
        for event, sense in labeled:
            grouped[sense].append(event)
        observed_names = set(situation.features_for(key))
        candidates: dict[tuple[str, Any], dict[str, float]] = defaultdict(dict)
        for sense, examples in grouped.items():
            feature_sets = [set(event.features_for(key).items()) for event in examples]
            frequencies = Counter(feature for features in feature_sets for feature in features)
            for feature, count in frequencies.items():
                candidates[feature][sense] = count / len(examples)
        ranked = []
        for (name, value), rates in candidates.items():
            if (name in observed_names or name.startswith("usage:")
                    or name == "context:speaker"):
                continue
            if name.startswith(("cue:", "context:")) and any(
                any(name not in event.features_for(key) for event in examples)
                for examples in grouped.values()
            ):
                continue
            if name.startswith("co:") and any(
                event.original_token_count > len(event.terms)
                for examples in grouped.values() for event in examples
            ):
                continue
            complete = {sense: rates.get(sense, 0.0) for sense in grouped}
            separation = max(complete.values()) - min(complete.values())
            if separation <= 0:
                continue
            ranked.append({"feature": name, "value": value,
                           "separation": separation,
                           "observed_rates": complete,
                           "status": "proposed_observation_not_evidence"})
        ranked.sort(key=lambda row: (-row["separation"], row["feature"], str(row["value"])))
        return {"status": "candidate_observations" if ranked else
                "no_discriminating_observation", "term": key,
                "observations": tuple(ranked[:limit]),
                "serving_authority": False}

    @_serialized
    def associative_analogy(self, left: str, right: str, *, setting: str = "",
                            community: str = "") -> dict[str, Any]:
        """Compare use-neighborhoods while keeping kind-of claims separate."""
        first = self.usage_associations(left, setting=setting, community=community)
        second = self.usage_associations(right, setting=setting, community=community)
        if (first["status"] != "observed_association" or
                second["status"] != "observed_association"):
            return {"status": "unmeasured" if "unmeasured_due_to_sampling" in (
                first["status"], second["status"]) else "unexposed",
                "shared": (), "serving_authority": False}
        left_neighbors = {row["term"] for row in first["associations"]}
        right_neighbors = {row["term"] for row in second["associations"]}
        shared = left_neighbors & right_neighbors
        union = left_neighbors | right_neighbors
        return {"status": "association_analogy", "left": left.casefold(),
                "right": right.casefold(), "shared": tuple(sorted(shared)),
                "jaccard": len(shared) / len(union) if union else 0.0,
                "relation": "shared_observed_use_not_taxonomy",
                "serving_authority": False}

    @_serialized
    def grounded_sense_cases(self, term: str, sense: str) -> tuple[SemanticCase, ...]:
        """Expose feedback as testable cases, not as an admitted definition."""
        key = term.casefold()
        return tuple(SemanticCase(
            event.source_id, event.context_id, f"usage_sense:{key}:{sense}",
            label == sense, event.features_for(key), event.observed_at,
            intervention="meaning_feedback")
            for event, label in self._labeled_usage(key))

    @_serialized
    def relation_evidence(self, left: str, right: str, *, atomspace: Any = None) -> dict[str, Any]:
        """Report story association apart from an explicitly stored kind-of edge."""
        from core.knowledge.atomspace import INHERITANCE, Link, concept, get_atomspace

        if atomspace is None:
            atomspace = get_atomspace()
        inheritance = atomspace.get_tv(Link(INHERITANCE, (concept(left), concept(right))))
        co_used = sum(left.casefold() in event.terms and right.casefold() in event.terms
                      for event in self.usage_events.values())
        return {"left": left.casefold(), "right": right.casefold(),
                "co_use_sources": co_used,
                "co_use_relation": "narrative_or_situational_association" if co_used
                else "unmeasured",
                "taxonomic_relation": ({"strength": inheritance.strength,
                                        "confidence": inheritance.confidence}
                                       if inheritance is not None else None),
                "serving_authority": False}

    @_serialized
    def eligible_proposals(
        self, outcome_name: str, features: Mapping[str, Any], *, limit: int = 16
    ) -> tuple[str, ...]:
        """Snapshot hypotheses that make a prospective claim here."""
        if not 1 <= limit <= 64:
            raise ValueError("prospective hypothesis limit is out of range")
        return tuple(identity for identity, proposal in reversed(tuple(self.proposals.items()))
                     if proposal.law.outcome_name == outcome_name
                     and proposal.kind == "theoretical" and proposal.law.holds(features))[:limit]

    @_serialized
    def propose(self, proposal: SemanticProposal) -> str:
        if len(self.proposals) >= _MAX_PROPOSALS and proposal.identity not in self.proposals:
            oldest = next(iter(self.proposals))
            del self.proposals[oldest]
        self.proposals[proposal.identity] = proposal
        return proposal.identity

    @_serialized
    def mutate(self, outcome_name: str, *, seed: int, limit: int = 16) -> tuple[str, ...]:
        """Explore bounded random combinations; none becomes a belief here."""
        if limit < 0 or limit > 64:
            raise ValueError("semantic mutation budget is out of range")
        fit_cases = self._cohorts(outcome_name)[0]
        fit = [Observation(case.features, case.outcome, case.observed_at, case.source_id)
               for case in fit_cases]
        atoms = candidate_predicates(fit)
        if len(atoms) < 2:
            return ()
        rng = random.Random(seed)
        names = []
        for index in range(limit):
            left, right = rng.sample(atoms, 2)
            if left.feature == right.feature:
                continue
            law = CandidateLaw((left, right), outcome_name)
            names.append(self.propose(SemanticProposal(
                law, "mutation", f"seed:{seed}:draw:{index}",
                exposure_sources=tuple(sorted({case.source_id for case in fit_cases})),
                exposure_audited=True)))
        return tuple(dict.fromkeys(names))

    @_serialized
    def propose_from_fit(self, outcome_name: str, *, limit: int = 8) -> tuple[str, ...]:
        """Freeze promising fit rules before their future outcome is observed."""
        if not 1 <= limit <= 12:
            raise ValueError("fit proposal budget is out of range")
        fit_cases = self._cohorts(outcome_name)[0]
        if len({case.source_id for case in fit_cases}) < self.min_support:
            return ()
        fit = [Observation(case.features, case.outcome, case.observed_at, case.source_id)
               for case in fit_cases]
        engine = OntologyDiscovery(outcome_name=outcome_name, min_support=self.min_support)
        laws, _ = engine._beam_search(fit, candidate_predicates(fit))
        exposure = tuple(sorted({case.source_id for case in fit_cases}))
        reference = "fit:" + _digest(exposure)
        return tuple(self.propose(SemanticProposal(
            law, "observation", reference, exposure_sources=exposure,
            exposure_audited=True)) for law in laws[:limit])

    @_serialized
    def theorize(
        self, law: CandidateLaw, *, channel: str, reference: str,
        assumptions: tuple[str, ...] = (), context_id: str = "",
        exposure_sources: tuple[str, ...] = (), expected_outcome: bool = True,
        exposure_audited: bool = False,
    ) -> str:
        """Keep a possible relation separate from certified knowledge."""
        return self.propose(SemanticProposal(
            law, channel, reference, context_id, (),
            "strange" if assumptions else "theoretical", assumptions,
            exposure_sources, expected_outcome, exposure_audited))

    @_serialized
    def explore(
        self, features: Mapping[str, Any], *, assumptions: tuple[str, ...] = (),
        max_steps: int = 4,
    ) -> dict[str, Any]:
        """Chain hypothetical consequences without manufacturing observations."""
        if not 1 <= max_steps <= 8:
            raise ValueError("theoretical exploration depth is out of range")
        frontier: list[tuple[dict[str, Any], tuple[str, ...]]] = [(dict(features), ())]
        paths: list[dict[str, Any]] = []
        contradictions: list[dict[str, Any]] = []
        available = set(assumptions)
        for _ in range(max_steps):
            next_frontier = []
            for state, lineage in frontier:
                for proposal in tuple(self.proposals.values()):
                    if (proposal.identity in lineage or
                            (proposal.kind == "strange" and
                             not set(proposal.assumptions) <= available) or
                            not proposal.law.holds(state)):
                        continue
                    effect = "hypothesis:" + proposal.law.outcome_name
                    if effect in state:
                        if state[effect] != proposal.expected_outcome:
                            contradictions.append({"proposal": proposal.identity,
                                                   "against": list(lineage), "effect": effect})
                        continue
                    branch = (*lineage, proposal.identity)
                    branch_state = {**state, effect: proposal.expected_outcome}
                    paths.append({"proposal": proposal.identity, "effect": effect,
                                  "expected_outcome": proposal.expected_outcome,
                                  "causal_path": list(branch),
                                  "assumptions": list(proposal.assumptions),
                                  "status": "counterfactual_only"})
                    next_frontier.append((branch_state, branch))
                    if len(paths) >= _MAX_BRANCHES:
                        break
                if len(paths) >= _MAX_BRANCHES:
                    break
            if not next_frontier or len(paths) >= _MAX_BRANCHES:
                break
            frontier = next_frontier[:_MAX_BRANCHES]
        return {"status": "counterfactual_only", "paths": paths,
                "contradictions": contradictions,
                "derived_features": sorted({path["effect"] for path in paths}),
                "serving_authority": False}

    @_serialized
    def commit_prediction(
        self, proposal_id: str, *, source_id: str, context_id: str,
        features: Mapping[str, Any], assumptions: tuple[str, ...] = (),
    ) -> str:
        """Pre-register a consequence for a source not yet observed.

        Alternate-world premises must already be measured and hold for this
        source. Merely imagining a premise never counts as a real-world trial.
        """
        proposal = self.proposals[proposal_id]
        path = self._prospective_path(
            proposal_id, source_id=source_id, context_id=context_id,
            features=features, assumptions=assumptions)
        slot = (source_id, context_id, proposal.law.outcome_name)
        for existing in self.predictions.values():
            if (existing.proposal_id, existing.source_id, existing.context_id,
                    existing.outcome_name) == (proposal_id, source_id, context_id,
                                              proposal.law.outcome_name):
                if existing.features_sha256 != _digest(dict(features)):
                    raise ValueError("a committed prediction cannot change its features")
                return existing.identity
        identity = _digest({"proposal": proposal_id, "source": source_id,
                            "context": context_id, "outcome": proposal.law.outcome_name,
                            "features": dict(features), "path": path["causal_path"]})
        if len(self.predictions) >= _MAX_PREDICTIONS:
            oldest = next(iter(self.predictions))
            removed = self.predictions.pop(oldest)
            self._pending_slots[(removed.source_id, removed.context_id,
                                 removed.outcome_name)].discard(oldest)
        self.predictions[identity] = PredictionCommitment(
            identity, proposal_id, source_id, context_id, proposal.law.outcome_name,
            _digest(dict(features)), proposal.expected_outcome,
            tuple(path["causal_path"]), time.time())
        self._pending_slots[slot].add(identity)
        return identity

    def _prospective_path(
        self, proposal_id: str, *, source_id: str, context_id: str,
        features: Mapping[str, Any], assumptions: tuple[str, ...],
    ) -> dict[str, Any]:
        proposal = self.proposals[proposal_id]
        if not source_id or not context_id or source_id in proposal.exposure_sources:
            raise ValueError("prediction requires a new identified, unexposed source")
        slot = (source_id, context_id, proposal.law.outcome_name)
        if slot in self._case_slots:
            raise ValueError("prediction must precede the direct observation")
        if (proposal.law.holds(features) and all(
            not predicate.feature.startswith("hypothesis:")
            for predicate in proposal.law.predicates
        )):
            path = {"proposal": proposal_id, "causal_path": [proposal_id]}
        else:
            scenario = self.explore(features, assumptions=assumptions)
            paths = [path for path in scenario["paths"] if path["proposal"] == proposal_id]
            if not paths:
                raise ValueError("proposal predicts no consequence in this situation")
            path = min(paths, key=lambda item: (len(item["causal_path"]), item["causal_path"]))
        for dependency_id in path["causal_path"]:
            dependency = self.proposals[dependency_id]
            if source_id in dependency.exposure_sources:
                raise ValueError("causal path was exposed to the test source")
            if dependency.kind != "strange":
                continue
            if not set(dependency.assumptions) <= set(assumptions):
                raise ValueError("alternate-world assumptions are not supplied")
            for premise in dependency.assumptions:
                primitive = self.primitives.get(premise)
                prior_slot = self._case_slots.get((source_id, context_id,
                                                   primitive.law.outcome_name)) if primitive else None
                prior_case = self.cases.get(prior_slot) if prior_slot else None
                if (primitive is None or primitive.status != "measured"
                        or not primitive.law.holds(features)
                        or source_id in primitive.source_ids
                        or prior_case is None or not prior_case.outcome
                        or not prior_case.independent
                        or not primitive.law.holds(prior_case.features)):
                    raise ValueError("alternate-world premise is not independently grounded")
        return path

    @_serialized
    def choose_experiment(
        self, proposal_ids: Sequence[str], situations: Sequence[SemanticSituation],
        *, assumptions: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        """Reuse Aura's experiment chooser on rival prospective consequences."""
        from core.cognition.the_experiment_that_settles_it import what_to_try

        if (not 2 <= len(proposal_ids) <= 16 or len(set(proposal_ids)) != len(proposal_ids)
                or not 1 <= len(situations) <= 32):
            raise ValueError("experiment comparison requires bounded rivals and situations")
        rivals = {identity: self.proposals[identity] for identity in proposal_ids}
        if len({item.law.outcome_name for item in rivals.values()}) != 1:
            raise ValueError("rival theories must predict the same measured outcome")

        def predicts(proposal: SemanticProposal, situation: SemanticSituation) -> bool | None:
            try:
                self._prospective_path(
                    proposal.identity, source_id=situation.source_id,
                    context_id=situation.context_id, features=situation.features,
                    assumptions=assumptions)
            except ValueError:
                return None
            return proposal.expected_outcome

        chosen = what_to_try(
            rivals, situations, predicts=predicts,
            plausibility=lambda _name, _item: 1.0, costs=lambda act: act.cost)
        if chosen is None:
            return {"status": "no_discriminating_observation", "serving_authority": False}
        return {"status": "proposed_experiment", "source_id": chosen.do.source_id,
                "context_id": chosen.do.context_id, "expected": chosen.expects,
                "silent": chosen.silent, "information_bits": chosen.settles,
                "cost": chosen.costs, "serving_authority": False}

    @_serialized
    def compare_predictions(self, left_id: str, right_id: str) -> dict[str, Any]:
        """Compare rival hypotheses on the same prospective, independent cases."""
        if left_id == right_id or left_id not in self.proposals or right_id not in self.proposals:
            raise ValueError("comparison needs two retained proposals")
        by_proposal: dict[str, dict[tuple[str, str, str], PredictionCommitment]] = {}
        for proposal_id in (left_id, right_id):
            by_proposal[proposal_id] = {
                (trial.source_id, trial.context_id, trial.outcome_name): trial
                for trial in self.predictions.values() if trial.proposal_id == proposal_id
                and trial.status in {"matched", "contradicted"}}
        shared = set(by_proposal[left_id]) & set(by_proposal[right_id])
        wins = [0, 0]
        sources = set()
        for slot in shared:
            left, right = by_proposal[left_id][slot], by_proposal[right_id][slot]
            if (left.expected_outcome == right.expected_outcome
                    or left.features_sha256 != right.features_sha256
                    or slot[0] in sources):
                continue
            sources.add(slot[0])
            wins[0 if left.status == "matched" else 1] += 1
        n = sum(wins)
        p_value = sum(math.comb(n, index) for index in range(max(wins), n + 1)) / 2**n if n else 1.0
        favored = None
        if n >= self.min_support and p_value <= 0.05:
            favored = left_id if wins[0] > wins[1] else right_id
        return {"status": "favored_candidate" if favored else "unresolved",
                "favored": favored, "matched_sources": n,
                "left_wins": wins[0], "right_wins": wins[1],
                "one_sided_p_value": p_value, "serving_authority": False}

    @_serialized
    def ground_strange(self, proposal_id: str) -> str | None:
        """Move a strange conjecture to the test queue after its premises hold."""
        proposal = self.proposals[proposal_id]
        if proposal.kind != "strange":
            raise ValueError("only alternate-world proposals need grounding")
        measured = [case for case in self.cases.values() if case.independent]
        for identity in proposal.assumptions:
            primitive = self.primitives.get(identity)
            if primitive is None or primitive.status != "measured":
                return None
            sources = {case.source_id for case in measured
                       if case.outcome_name == primitive.law.outcome_name
                       and case.outcome and primitive.law.holds(case.features)}
            if len(sources) < self.min_support:
                return None
        return self.propose(SemanticProposal(
            proposal.law, proposal.channel, proposal.reference, proposal.context_id,
            (*proposal.dependencies, *proposal.assumptions), "theoretical", (),
            tuple(dict.fromkeys((*proposal.exposure_sources, *(
                source for identity in proposal.assumptions
                for source in self.primitives[identity].source_ids)))),
            proposal.expected_outcome,
            proposal.exposure_audited))

    @_serialized
    def compose(self, left: str, right: str, *, outcome_name: str) -> str:
        """A higher concept remains a proposal until separately tested."""
        first, second = self.primitives[left], self.primitives[right]
        predicates = tuple(dict.fromkeys((*first.law.predicates, *second.law.predicates)))
        if len(predicates) > 3:
            raise ValueError("composition exceeds the tested predicate grammar")
        return self.propose(SemanticProposal(
            CandidateLaw(predicates, outcome_name), "composition", f"{left}:{right}",
            dependencies=(left, right),
            exposure_sources=tuple(dict.fromkeys((*first.source_ids, *second.source_ids))),
            exposure_audited=True))

    def _cohorts(self, outcome_name: str) -> tuple[tuple[SemanticCase, ...], ...]:
        measured = [case for case in self.cases.values()
                    if case.outcome_name == outcome_name and case.independent
                    and case.source_id not in self._validation_sources]
        by_source: dict[str, list[SemanticCase]] = defaultdict(list)
        for case in measured:
            by_source[case.source_id].append(case)
        sources = sorted(by_source, key=lambda name: (
            min(case.observed_at for case in by_source[name]), name))
        first, second = len(sources) // 2, len(sources) * 3 // 4
        return tuple(tuple(case for name in names for case in by_source[name])
                     for names in (sources[:first], sources[first:second], sources[second:]))

    @_serialized
    def fit_feature_inventory(self, outcome_name: str) -> dict[str, tuple[Any, ...]]:
        """Feature vocabulary and observed values from fit sources only."""
        inventory: dict[str, set[Any]] = defaultdict(set)
        for case in self._cohorts(outcome_name)[0]:
            for name, value in case.features.items():
                if value is not None and len(inventory[name]) < 16:
                    inventory[name].add(value)
        return {name: tuple(sorted(values, key=repr)) for name, values in inventory.items()}

    @_serialized
    def test(self, outcome_name: str, *, cohorts: tuple[Sequence[SemanticCase],
             Sequence[SemanticCase], Sequence[SemanticCase]] | None = None) -> dict[str, Any]:
        """Certify a useful distinction, or retain why no proposal survived."""
        groups = cohorts or self._cohorts(outcome_name)
        if len(groups) != 3 or any(not group for group in groups):
            return {"status": "unmeasured", "reason": "three source cohorts are required"}
        if any(case.outcome_name != outcome_name or not case.independent
               for group in groups for case in group):
            raise ValueError("semantic tests require direct, outcome-matched observations")
        held_sources = {case.source_id for group in groups[1:] for case in group}
        all_sources = {case.source_id for group in groups for case in group}
        if all_sources & self._validation_sources:
            raise ValueError("a semantic validation source cannot be spent twice")
        trial_index = len(self.trials) + 1
        trial_alpha = 0.05 / (trial_index * (trial_index + 1))
        observations = tuple(tuple(Observation(case.features, case.outcome,
            case.observed_at, case.source_id) for case in group) for group in groups)
        proposals = tuple(proposal.law for proposal in self.proposals.values()
                          if proposal.law.outcome_name == outcome_name
                          and proposal.kind == "theoretical"
                          and proposal.exposure_audited
                          and proposal.expected_outcome
                          and not held_sources.intersection(proposal.exposure_sources))
        engine = OntologyDiscovery(outcome_name=outcome_name, min_support=self.min_support,
                                   max_p_value=trial_alpha)
        result = engine.discover_partitioned(*observations, proposals=proposals)
        cohort_ids = tuple(tuple(sorted({case.source_id for case in group})) for group in groups)
        receipt = {"schema": "aura.semantic_development_trial.v1",
                   "outcome": outcome_name,
                   "trial_index": trial_index, "familywise_alpha": trial_alpha,
                   "cohort_sha256": [_digest(ids) for ids in cohort_ids],
                   "candidates_considered": result.candidates_considered,
                   "reason": result.refusal, "discovered": result.discovered.to_dict()
                   if result.discovered else None}
        receipt["receipt_sha256"] = _digest(receipt)
        self.trials.append({"outcome": outcome_name, "trial_index": trial_index,
                            "validation_sources": sorted(held_sources),
                            "receipt_sha256": receipt["receipt_sha256"]})
        self._validation_sources.update(held_sources)
        if result.discovered is None:
            self.failed.append({"scope": outcome_name, "receipt": receipt})
            self.failed = self.failed[-_MAX_PROPOSALS:]
            return {"status": "rejected", "receipt": receipt}
        law = result.discovered.law
        matching = tuple(proposal for proposal in self.proposals.values()
                         if proposal.law == law and proposal.exposure_audited
                         and not held_sources.intersection(proposal.exposure_sources))
        dependencies = tuple(dict.fromkeys(dep for proposal in matching
                                           for dep in proposal.dependencies))
        identity = _digest({"law": _law_dict(law), "parents": dependencies})
        prior_primitive = self.primitives.get(identity)
        if prior_primitive is not None:
            if prior_primitive.status != "measured":
                self.failed.append({"scope": identity, "reason": "revoked_law_reappeared",
                                    "receipt_sha256": receipt["receipt_sha256"]})
                self.failed = self.failed[-_MAX_PROPOSALS:]
                return {"status": "recovery_candidate", "receipt": receipt,
                        "serving_authority": False}
            replicated = replace(
                prior_primitive,
                source_ids=tuple(dict.fromkeys((*prior_primitive.source_ids,
                                                *(source for group in cohort_ids
                                                  for source in group)))),
                validation_sources=tuple(dict.fromkeys(
                    (*prior_primitive.validation_sources, *cohort_ids[1], *cohort_ids[2]))),
                contexts=tuple(sorted(set(prior_primitive.contexts) |
                                      {case.context_id for group in groups for case in group
                                       if law.holds(case.features)})),
                replication_receipts=(*prior_primitive.replication_receipts,
                                      receipt["receipt_sha256"]),
            )
            self.primitives[identity] = replicated
            return {"status": "replicated", "primitive": replicated.to_dict(),
                    "receipt": receipt}
        primitive = SemanticPrimitive(
            identity, law, result.discovered.to_dict(),
            tuple(dict.fromkeys(source for group in cohort_ids for source in group)),
            tuple(sorted({case.context_id for group in groups for case in group
                          if law.holds(case.features)})), dependencies,
            validation_sources=tuple((*cohort_ids[1], *cohort_ids[2])),
            trial_receipt_sha256=receipt["receipt_sha256"])
        concept_ids = []
        if self.concept_engine is not None:
            for concept in self.concept_engine.concepts():
                try:
                    self.concept_engine.certify(concept["concept_id"], result.discovered)
                except ValueError:
                    continue
                concept_ids.append(concept["concept_id"])
        primitive = replace(primitive, concept_ids=tuple(concept_ids))
        if len(self.primitives) >= _MAX_PRIMITIVES and identity not in self.primitives:
            oldest = next(iter(self.primitives))
            del self.primitives[oldest]
        self.primitives[identity] = primitive
        self._bind_primitive(primitive)
        return {"status": "measured", "primitive": primitive.to_dict(), "receipt": receipt}

    def _bind_primitive(self, primitive: SemanticPrimitive) -> None:
        witnesses = frozenset(f"semantic:{source}" for source in primitive.validation_sources)
        packet = EvidencePacket(strength=primitive.precision, mass=float(len(witnesses)),
            kind=EvidenceKind.OBSERVATION, sources=witnesses, subject=primitive.identity,
            produced_by="semantic_development")
        self.registry.bind(primitive.law.name(), Substrate.FORMED, primitive.identity,
                           method=BindingMethod.MEASURED,
                           confidence=packet.confidence, evidence=packet,
                           detail={"receipt_sha256": primitive.trial_receipt_sha256})

    @_serialized
    def retest(self, identity: str, fresh: Sequence[SemanticCase]) -> dict[str, Any]:
        """Reopen a compiled distinction when independent future data defeats it."""
        primitive = self.primitives[identity]
        new_sources = [case.source_id for case in fresh]
        if (not fresh or any(case.outcome_name != primitive.law.outcome_name
                             or not case.independent or case.source_id in primitive.source_ids
                             or case.source_id in self._validation_sources
                             for case in fresh) or len(set(new_sources)) != len(new_sources)
                or primitive.status != "measured"):
            raise ValueError("semantic revision needs fresh independent sources")
        accumulated = [*self._revision_cases.get(identity, ()), *fresh]
        observations = tuple(Observation(case.features, case.outcome, case.observed_at,
                                         case.source_id) for case in accumulated)
        score = score_split(primitive.law, observations)
        rate = base_rate(observations)
        lift = score.lift(rate)
        support_sources = {case.source_id for case in accumulated
                           if primitive.law.holds(case.features)}
        status = "inconclusive"
        look_index = len(primitive.revisions) + 1
        look_alpha = 0.05 / (look_index * (look_index + 1))
        if len(support_sources) >= self.min_support and 0.0 < rate < 1.0:
            if lift > 1.0:
                status = "retained"
            else:
                positives = sum(case.outcome for case in accumulated)
                population = len(accumulated)
                denominator = math.comb(population, score.support)
                lower_tail = sum(
                    math.comb(positives, hits) *
                    math.comb(population - positives, score.support - hits)
                    for hits in range(score.hits + 1)
                    if hits <= positives and score.support - hits <= population - positives
                ) / denominator
                if lower_tail <= look_alpha:
                    status = "revoked"
        receipt = _digest({"primitive": identity, "new_sources": sorted({case.source_id
                          for case in fresh}), "accumulated_sources": sorted({case.source_id
                          for case in accumulated}), "support": score.support,
                          "hits": score.hits, "lift": lift, "status": status,
                          "look_index": look_index, "familywise_alpha": look_alpha})
        self._revision_cases[identity] = accumulated
        self._validation_sources.update(new_sources)
        self.primitives[identity] = replace(
            primitive, status="revoked" if status == "revoked" else primitive.status,
            source_ids=tuple(dict.fromkeys((*primitive.source_ids,
                                            *(case.source_id for case in fresh)))),
            revisions=(*primitive.revisions, receipt))
        if status == "revoked":
            self.registry.unbind(Substrate.FORMED, identity)
            if self.concept_engine is not None:
                for concept_id in primitive.concept_ids:
                    self.concept_engine.revoke_certification(
                        concept_id, primitive.receipt["provenance"],
                        counterevidence=receipt)
            self.failed.append({"scope": identity, "reason": "fresh_counterevidence",
                                "receipt_sha256": receipt})
            self.failed = self.failed[-_MAX_PROPOSALS:]
        return {"status": status, "support": score.support, "hits": score.hits,
                "lift": lift, "look_index": look_index,
                "familywise_alpha": look_alpha, "receipt_sha256": receipt}

    @_serialized
    def related(self, left: str, right: str) -> dict[str, Any]:
        """Shared support proposes a relation without silently merging concepts."""
        a, b = self.primitives[left], self.primitives[right]
        overlap = set(a.sparse_vector) & set(b.sparse_vector)
        union = set(a.sparse_vector) | set(b.sparse_vector)
        shared_parents = set(a.dependencies) & set(b.dependencies)
        return {"left": left, "right": right, "shared_predicates": len(overlap),
                "shared_parents": len(shared_parents),
                "similarity": len(overlap) / len(union) if union else 0.0,
                "relation_status": "hypothesis" if overlap or shared_parents else "unmeasured"}

    @_serialized
    def infer(self, outcome_name: str, features: Mapping[str, Any], *,
              context_id: str = "") -> dict[str, Any]:
        """Use compiled evidence quickly while preserving context and alternatives."""
        matches = [primitive for primitive in self.primitives.values()
                   if primitive.status == "measured" and primitive.law.outcome_name == outcome_name
                   and primitive.law.holds(features)]
        matches.sort(key=lambda primitive: (-primitive.precision,
                                            len(primitive.law.predicates), primitive.identity))
        return {"status": "candidate" if matches else "unresolved",
                "outcome": outcome_name,
                "alternatives": [{"primitive": item.identity,
                                  "precision": item.precision,
                                  "context_status": ("measured" if context_id in item.contexts
                                                     else "unmeasured_transfer"),
                                  "receipt_sha256": item.receipt["provenance"]}
                                 for item in matches],
                "serving_authority": False}

    def save(self) -> None:
        """Persist the bounded graph without promoting an untested proposal."""
        from core.governance_context import local_internal_governed_scope
        from core.runtime.file_write_gateway import get_file_write_gateway

        if not self._save_lane.acquire():
            raise TimeoutError("semantic development state writer is still busy")
        try:
            with self._lock:
                body = {"schema": "aura.semantic_development.v1",
                        "cases": [case.to_dict() for case in self.cases.values()],
                        "usage_events": [event.to_dict() for event in self.usage_events.values()],
                        "usage_observed_count": self._usage_observed_count,
                        "meaning_feedback": [item.to_dict()
                                             for item in self.meaning_feedback.values()],
                        "proposals": [{"law": _law_dict(item.law), "channel": item.channel,
                                       "reference": item.reference, "context_id": item.context_id,
                                       "dependencies": list(item.dependencies), "kind": item.kind,
                                       "assumptions": list(item.assumptions),
                                       "exposure_sources": list(item.exposure_sources),
                                       "expected_outcome": item.expected_outcome,
                                       "exposure_audited": item.exposure_audited}
                                      for item in self.proposals.values()],
                        "primitives": [item.to_dict() for item in self.primitives.values()],
                        "predictions": [item.to_dict() for item in self.predictions.values()],
                        "observed_count": self._observed_count,
                        "trials": list(self.trials),
                        "revision_cases": {identity: [case.to_dict() for case in cases]
                                           for identity, cases in self._revision_cases.items()},
                        "failed": list(self.failed)}
            payload = json.dumps({**body, "receipt_sha256": _digest(body)}, sort_keys=True)
            with local_internal_governed_scope("semantic_development.save", domain="state_mutation"):
                gateway = get_file_write_gateway()
                gateway.ensure_directory(self.state_path.parent, source="semantic_development")
                gateway.write_text(self.state_path, payload, source="semantic_development")
        finally:
            self._save_lane.release()

    @_serialized
    def load(self) -> None:
        payload = json.loads(self.state_path.read_text())
        body = {key: value for key, value in payload.items() if key != "receipt_sha256"}
        if body.get("schema") != "aura.semantic_development.v1" or _digest(body) != payload.get(
            "receipt_sha256"
        ):
            raise ValueError("semantic development state failed integrity validation")
        self.cases = {case.identity: case for case in (
            SemanticCase(**{**raw, "candidate_dependencies": tuple(
                raw.get("candidate_dependencies", ()))}) for raw in body["cases"])}
        self.usage_events = {event.source_id: event for event in (
            UsageEvent.from_dict(raw) for raw in body.get("usage_events", ())) }
        self._usage_observed_count = max(len(self.usage_events),
                                         int(body.get("usage_observed_count", 0)))
        self.meaning_feedback = {item.source_id: item for item in (
            MeaningFeedback(**raw) for raw in body.get("meaning_feedback", ())) }
        self._case_slots = {
            (case.source_id, case.context_id, case.outcome_name): case.identity
            for case in self.cases.values()}
        self._observed_count = max(len(self.cases), int(body.get("observed_count", 0)))
        self.trials = list(body.get("trials", ()))
        self._validation_sources = {
            source for trial in self.trials for source in trial["validation_sources"]}
        self._revision_cases = {
            identity: [SemanticCase(**{**raw, "candidate_dependencies": tuple(
                raw.get("candidate_dependencies", ()))}) for raw in cases]
            for identity, cases in body.get("revision_cases", {}).items()}
        self._validation_sources.update(case.source_id for cases in self._revision_cases.values()
                                        for case in cases)
        self.proposals = {item.identity: item for item in (
            SemanticProposal(_law_from_dict(raw["law"]), raw["channel"], raw["reference"],
                             raw.get("context_id", ""), tuple(raw.get("dependencies", ())),
                             raw.get("kind", "theoretical"), tuple(raw.get("assumptions", ())),
                             tuple(raw.get("exposure_sources", ())),
                             raw.get("expected_outcome", True),
                             raw.get("exposure_audited", False))
            for raw in body["proposals"])}
        self.primitives = {item.identity: item for item in (
            SemanticPrimitive(raw["identity"], _law_from_dict(raw["law"]), raw["receipt"],
                              tuple(raw["source_ids"]), tuple(raw["contexts"]),
                              tuple(raw.get("dependencies", ())), raw["status"],
                              tuple(raw.get("revisions", ())),
                              tuple(raw.get("validation_sources", ())),
                              tuple(raw.get("concept_ids", ())),
                              raw.get("trial_receipt_sha256", ""),
                              tuple(raw.get("replication_receipts", ())))
            for raw in body["primitives"])}
        for primitive in self.primitives.values():
            if (primitive.status == "measured" and primitive.validation_sources
                    and primitive.trial_receipt_sha256):
                self._bind_primitive(primitive)
            elif primitive.status == "revoked":
                self.registry.unbind(Substrate.FORMED, primitive.identity)
                if self.concept_engine is not None and primitive.revisions:
                    for concept_id in primitive.concept_ids:
                        self.concept_engine.revoke_certification(
                            concept_id, primitive.receipt["provenance"],
                            counterevidence=primitive.revisions[-1])
        self.predictions = {item.identity: item for item in (
            PredictionCommitment(
                raw["identity"], raw["proposal_id"], raw["source_id"],
                raw["context_id"], raw["outcome_name"], raw["features_sha256"],
                raw["expected_outcome"], tuple(raw["causal_path"]),
                raw["committed_at"], raw["status"], raw["observed_outcome"])
            for raw in body.get("predictions", ()))}
        self._pending_slots = defaultdict(set)
        for prediction in self.predictions.values():
            if prediction.status == "pending":
                self._pending_slots[(prediction.source_id, prediction.context_id,
                                     prediction.outcome_name)].add(prediction.identity)
        self.failed = list(body["failed"])


__all__ = ["PredictionCommitment", "SemanticCase", "SemanticDevelopment", "SemanticPrimitive", "SemanticProposal", "SemanticSituation",
           "transition_features"]
