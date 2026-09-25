"""core/brain/ontology_discovery.py — the discovery step OntologyGenesis lacked.

CP126 c0a3c26e found that "autonomous formation of cognitive laws" was a loop
which logged, slept sixty seconds, and reached a comment saying the real logic
went there. `core/brain/ontology_genesis.py` was made to say so rather than
dress it up: `DISCOVERY_IMPLEMENTED = False`, and a note naming the four things
a real step owes — a candidate, an experiment, a verifier result, and a written
discovered result.

This is that step. It induces a **cognitive law** — a named conjunctive
predicate over observable runtime features that raises the probability of an
outcome — from the runtime's own recorded episodes, and it admits one only
after the law survives evidence it could have failed.

    anomaly       an outcome whose base rate leaves something to explain
    hypothesis    a conjunction found by beam search on the training split
    experiment    lift measured on a held-out split the search never saw
    verifier      an exact conditional null corrected for candidate search,
                  plus a per-conjunct ablation
    integration   the law enters the shared heuristic pool that
                  curiosity_explorer, dreamer_v2 and dream_skill already read
    transfer      lift re-measured on a third split, later in time
    retention     persisted with the evidence and a provenance hash

What this is honest about
-------------------------
The representational primitive a run invents is a conjunction over features it
was given — a new predicate the system did not have, with a name, a definition
and measured support. It is not an arbitrary new data structure, and inventing
the feature vocabulary itself is a further step this does not take. Saying so
here is the point: the module it replaces claimed the larger thing and did
nothing.

Every acceptance rule is a way for a run to end with no discovery. A discovery
loop that cannot come back empty is not measuring anything, and returning None
is the common case by design.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import random
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

logger = logging.getLogger("Aura.OntologyDiscovery")

#: Conjunction width. Beyond three conjuncts a rule found on a few hundred
#: episodes is describing the training split, and the held-out check starts
#: rejecting everything — the search cost rises while the yield falls.
MAX_CONJUNCTS = 3

#: How many episodes a rule must fire on, per split, before its precision is
#: worth reading. Below this a single episode moves the estimate by more than
#: the effect being claimed.
MIN_SUPPORT = 8

#: Permutations behind the p-value. 999 gives a smallest attainable p of 0.001,
#: which is the resolution the acceptance threshold below needs.
NULL_PERMUTATIONS = 999

#: A law must beat this share of its own null. Fixed before any run rather than
#: chosen after seeing one.
MAX_P_VALUE = 0.01

#: Held-out lift must exceed this. 1.0 is "the rule tells you nothing you did
#: not already know from the base rate".
MIN_HELDOUT_LIFT = 1.25

#: Quantiles that become numeric thresholds. Coarse on purpose: a threshold
#: fitted finely to the training split is the first thing the held-out check
#: throws away.
NUMERIC_QUANTILES = (0.25, 0.5, 0.75)


@dataclass(frozen=True)
class Observation:
    """One episode: what was observable, and what happened."""

    features: Mapping[str, Any]
    outcome: bool
    at: float = 0.0
    source_id: str = ""


@dataclass(frozen=True)
class Predicate:
    """One conjunct. `op` is one of >=, <, ==."""

    feature: str
    op: str
    value: Any

    def holds(self, features: Mapping[str, Any]) -> bool:
        if self.feature not in features:
            return False
        actual = features[self.feature]
        try:
            if self.op == ">=":
                return float(actual) >= float(self.value)
            if self.op == "<":
                return float(actual) < float(self.value)
            if self.op == "==":
                return actual == self.value
        # not a failure: two values that will not compare do not satisfy the condition,
        # and False is the refusing direction.
        except (TypeError, ValueError):
            return False
        return False

    def describe(self) -> str:
        if self.op == "==":
            return f"{self.feature} == {self.value!r}"
        return f"{self.feature} {self.op} {self.value:g}"


@dataclass(frozen=True)
class CandidateLaw:
    """A conjunction of predicates, and the outcome it claims to raise."""

    predicates: tuple[Predicate, ...]
    outcome_name: str

    def holds(self, features: Mapping[str, Any]) -> bool:
        return all(predicate.holds(features) for predicate in self.predicates)

    def describe(self) -> str:
        antecedent = " and ".join(p.describe() for p in self.predicates)
        return f"when {antecedent}, expect {self.outcome_name}"

    def name(self) -> str:
        digest = hashlib.sha256(self.describe().encode("utf-8")).hexdigest()[:10]
        return f"law_{digest}"


@dataclass(frozen=True)
class SplitScore:
    support: int
    hits: int

    @property
    def precision(self) -> float:
        return self.hits / self.support if self.support else 0.0

    def lift(self, base_rate: float) -> float:
        if base_rate <= 0.0:
            return 0.0
        return self.precision / base_rate


@dataclass(frozen=True)
class LawEvidence:
    """Everything an auditor needs to accept or reject the law."""

    base_rate: float
    train: SplitScore
    heldout: SplitScore
    transfer: SplitScore
    heldout_lift: float
    transfer_lift: float
    p_value: float
    permutations: int
    #: Held-out lift lost by dropping each conjunct. A conjunct that costs
    #: nothing is not part of the law and is pruned before this is recorded.
    ablation: dict[str, float] = field(default_factory=dict)
    validation_method: str = "permutation"
    familywise_candidates: int = 1
    transfer_p_value: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["train_precision"] = self.train.precision
        payload["heldout_precision"] = self.heldout.precision
        payload["transfer_precision"] = self.transfer.precision
        return payload


@dataclass(frozen=True)
class DiscoveredLaw:
    law: CandidateLaw
    evidence: LawEvidence
    discovered_at: float
    observation_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.law.name(),
            "rule": self.law.describe(),
            "outcome": self.law.outcome_name,
            "predicates": [p.describe() for p in self.law.predicates],
            "evidence": self.evidence.to_dict(),
            "discovered_at": self.discovered_at,
            "observation_count": self.observation_count,
            "provenance": self.provenance(),
        }

    def provenance(self) -> str:
        payload = json.dumps(
            {
                "rule": self.law.describe(),
                "evidence": self.evidence.to_dict(),
                "observation_count": self.observation_count,
            },
            sort_keys=True,
            default=str,
        )
        return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class DiscoveryOutcome:
    """The result of one cycle, including every way it can be nothing."""

    discovered: DiscoveredLaw | None
    refusal: str = ""
    candidates_considered: int = 0

    @property
    def found(self) -> bool:
        return self.discovered is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "found": self.found,
            "refusal": self.refusal,
            "candidates_considered": self.candidates_considered,
            "law": self.discovered.to_dict() if self.discovered else None,
        }


def _quantile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = q * (len(sorted_values) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return sorted_values[int(position)]
    weight = position - low
    return sorted_values[low] * (1 - weight) + sorted_values[high] * weight


def candidate_predicates(observations: Sequence[Observation]) -> list[Predicate]:
    """The predicate vocabulary this training split supports.

    Thresholds come from the split's own distribution rather than from
    constants, so a feature whose scale nobody anticipated still yields usable
    cut points, and a feature that never varies yields none.
    """
    numeric: dict[str, list[float]] = {}
    categorical: dict[str, set[Any]] = {}
    for observation in observations:
        for key, value in observation.features.items():
            if isinstance(value, bool):
                categorical.setdefault(key, set()).add(value)
            elif isinstance(value, (int, float)):
                numeric.setdefault(key, []).append(float(value))
            elif isinstance(value, str):
                categorical.setdefault(key, set()).add(value)

    predicates: list[Predicate] = []
    for key, values in sorted(numeric.items()):
        ordered = sorted(values)
        if ordered[0] == ordered[-1]:
            continue  # constant feature carries no information
        seen: set[float] = set()
        for q in NUMERIC_QUANTILES:
            threshold = round(_quantile(ordered, q), 6)
            if threshold in seen or threshold <= ordered[0]:
                continue
            seen.add(threshold)
            predicates.append(Predicate(key, ">=", threshold))
            predicates.append(Predicate(key, "<", threshold))
    for key, values in sorted(categorical.items(), key=lambda item: item[0]):
        if len(values) < 2:
            continue
        for value in sorted(values, key=repr):
            predicates.append(Predicate(key, "==", value))
    return predicates


def score_split(law: CandidateLaw, observations: Sequence[Observation]) -> SplitScore:
    support = 0
    hits = 0
    for observation in observations:
        if law.holds(observation.features):
            support += 1
            if observation.outcome:
                hits += 1
    return SplitScore(support=support, hits=hits)


def base_rate(observations: Sequence[Observation]) -> float:
    if not observations:
        return 0.0
    return sum(1 for o in observations if o.outcome) / len(observations)


def exact_conditional_p_value(law: CandidateLaw, observations: Sequence[Observation]) -> float:
    """Exact upper tail with the observed positive count and rule support fixed."""
    population = len(observations)
    positives = sum(item.outcome for item in observations)
    score = score_split(law, observations)
    if population == 0 or score.support == 0:
        return 1.0
    denominator = math.comb(population, score.support)
    return sum(
        math.comb(positives, hits)
        * math.comb(population - positives, score.support - hits)
        for hits in range(score.hits, min(score.support, positives) + 1)
        if score.support - hits <= population - positives
    ) / denominator


class OntologyDiscovery:
    """Induce a cognitive law, or come back with the reason there is none."""

    def __init__(
        self,
        *,
        outcome_name: str = "degradation",
        max_conjuncts: int = MAX_CONJUNCTS,
        min_support: int = MIN_SUPPORT,
        permutations: int = NULL_PERMUTATIONS,
        max_p_value: float = MAX_P_VALUE,
        min_heldout_lift: float = MIN_HELDOUT_LIFT,
        seed: int = 20260815,
    ) -> None:
        self.outcome_name = outcome_name
        self.max_conjuncts = max(1, int(max_conjuncts))
        self.min_support = max(1, int(min_support))
        self.permutations = max(1, int(permutations))
        self.max_p_value = float(max_p_value)
        self.min_heldout_lift = float(min_heldout_lift)
        self.seed = int(seed)

    # -- splits ----------------------------------------------------------
    def split(
        self, observations: Sequence[Observation]
    ) -> tuple[list[Observation], list[Observation], list[Observation]]:
        """Train, held-out and transfer, split by time rather than at random.

        A random split leaks: episodes from the same minute land on both sides
        and a rule about that minute scores as a rule about the runtime. The
        transfer split is strictly later than the other two, so retention is
        measured against a future the search could not see.
        """
        ordered = sorted(observations, key=lambda o: (o.at, id(o)))
        n = len(ordered)
        first = int(n * 0.5)
        second = int(n * 0.75)
        return ordered[:first], ordered[first:second], ordered[second:]

    # -- search ----------------------------------------------------------
    def _beam_search(
        self, train: Sequence[Observation], predicates: Sequence[Predicate]
    ) -> tuple[list[CandidateLaw], int]:
        rate = base_rate(train)
        if rate <= 0.0 or rate >= 1.0:
            return [], 0

        scored: list[tuple[float, CandidateLaw]] = []
        considered = 0
        frontier: list[tuple[Predicate, ...]] = [()]
        for _ in range(self.max_conjuncts):
            next_frontier: list[tuple[float, tuple[Predicate, ...]]] = []
            for prefix in frontier:
                for predicate in predicates:
                    if predicate in prefix:
                        continue
                    if any(p.feature == predicate.feature for p in prefix):
                        continue  # one conjunct per feature keeps rules readable
                    conjunction = (*prefix, predicate)
                    law = CandidateLaw(conjunction, self.outcome_name)
                    considered += 1
                    score = score_split(law, train)
                    if score.support < self.min_support:
                        continue
                    lift = score.lift(rate)
                    if lift <= 1.0:
                        continue
                    scored.append((lift, law))
                    next_frontier.append((lift, conjunction))
            next_frontier.sort(key=lambda item: (-item[0], len(item[1])))
            frontier = [conjunction for _, conjunction in next_frontier[:12]]
            if not frontier:
                break

        scored.sort(key=lambda item: (-item[0], len(item[1].predicates)))
        return [law for _, law in scored[:12]], considered

    # -- verifier --------------------------------------------------------
    def permutation_p_value(
        self, law: CandidateLaw, heldout: Sequence[Observation]
    ) -> float:
        """Share of shuffled label sets whose lift matches or beats the real one.

        The null this needs is "the same rule, on the same episodes, with the
        outcomes detached". Shuffling the labels keeps the rule's support and
        the outcome's base rate exactly as observed and destroys only the
        association between them, which is the thing being claimed.
        """
        rate = base_rate(heldout)
        observed = score_split(law, heldout).lift(rate)
        if observed <= 0.0:
            return 1.0

        firing = [law.holds(o.features) for o in heldout]
        labels = [o.outcome for o in heldout]
        support = sum(1 for f in firing if f)
        if support == 0:
            return 1.0

        rng = random.Random(self.seed)
        shuffled = list(labels)
        at_least_as_extreme = 0
        for _ in range(self.permutations):
            rng.shuffle(shuffled)
            hits = sum(1 for fires, label in zip(firing, shuffled) if fires and label)
            lift = (hits / support) / rate if rate > 0 else 0.0
            if lift >= observed:
                at_least_as_extreme += 1
        # +1 on both sides: the observed arrangement is itself one of the
        # arrangements under the null, and omitting it can return p = 0, which
        # claims more resolution than the permutation count supports.
        return (at_least_as_extreme + 1) / (self.permutations + 1)

    def prune(
        self, law: CandidateLaw, heldout: Sequence[Observation]
    ) -> tuple[CandidateLaw, dict[str, float]]:
        """Drop conjuncts on the supplied fit split, and report the rest.

        A conjunct whose removal does not lower fit lift is decoration. This
        must run before held-out scoring, or validation becomes another fit.
        """
        rate = base_rate(heldout)
        predicates = list(law.predicates)
        improved = True
        while improved and len(predicates) > 1:
            improved = False
            current = CandidateLaw(tuple(predicates), self.outcome_name)
            current_score = score_split(current, heldout)
            current_lift = current_score.lift(rate)
            for index in range(len(predicates)):
                reduced = predicates[:index] + predicates[index + 1 :]
                candidate = CandidateLaw(tuple(reduced), self.outcome_name)
                score = score_split(candidate, heldout)
                if score.support < self.min_support:
                    continue
                if score.lift(rate) >= current_lift:
                    predicates = reduced
                    improved = True
                    break

        final = CandidateLaw(tuple(predicates), self.outcome_name)
        final_lift = score_split(final, heldout).lift(rate)
        ablation: dict[str, float] = {}
        for index, predicate in enumerate(predicates):
            reduced = predicates[:index] + predicates[index + 1 :]
            if not reduced:
                ablation[predicate.describe()] = final_lift
                continue
            candidate = CandidateLaw(tuple(reduced), self.outcome_name)
            ablation[predicate.describe()] = round(
                final_lift - score_split(candidate, heldout).lift(rate), 6
            )
        return final, ablation

    # -- the cycle -------------------------------------------------------
    def discover(self, observations: Iterable[Observation]) -> DiscoveryOutcome:
        episodes = list(observations)
        if len(episodes) < self.min_support * 6:
            return DiscoveryOutcome(
                None,
                refusal=(
                    f"{len(episodes)} episode(s) is below the "
                    f"{self.min_support * 6} needed for three usable splits"
                ),
            )

        train, heldout, transfer = self.split(episodes)
        return self._discover_splits(train, heldout, transfer)

    def discover_partitioned(
        self,
        train: Sequence[Observation],
        heldout: Sequence[Observation],
        transfer: Sequence[Observation],
        *,
        proposals: Sequence[CandidateLaw] = (),
    ) -> DiscoveryOutcome:
        """Test proposals and learned rules on independent source cohorts."""
        groups = []
        for split in (train, heldout, transfer):
            ids = {item.source_id for item in split}
            if not split or "" in ids:
                raise ValueError("partitioned discovery needs identified sources in every split")
            if len(ids) != len(split):
                raise ValueError("partitioned discovery needs one independent row per source")
            groups.append(ids)
        if any(groups[left] & groups[right] for left, right in ((0, 1), (0, 2), (1, 2))):
            raise ValueError("partitioned discovery shares source identities across splits")
        if any(not isinstance(law, CandidateLaw) or law.outcome_name != self.outcome_name
               or not law.predicates for law in proposals):
            raise ValueError("proposed laws must name the measured outcome")
        return self._discover_splits(train, heldout, transfer, proposals=proposals)

    def _discover_splits(
        self,
        train: Sequence[Observation],
        heldout: Sequence[Observation],
        transfer: Sequence[Observation],
        *,
        proposals: Sequence[CandidateLaw] = (),
    ) -> DiscoveryOutcome:
        episodes = [*train, *heldout, *transfer]
        if any(len(split) < self.min_support for split in (train, heldout, transfer)):
            return DiscoveryOutcome(None, refusal="a discovery split has too few observations")
        overall_rate = base_rate(episodes)
        if overall_rate <= 0.0:
            return DiscoveryOutcome(None, refusal="no episode had the outcome")
        if overall_rate >= 1.0:
            return DiscoveryOutcome(
                None, refusal="every episode had the outcome; nothing to explain"
            )

        predicates = candidate_predicates(train)
        if not predicates and not proposals:
            return DiscoveryOutcome(
                None, refusal="no feature varied enough to form a predicate"
            )

        candidates, considered = self._beam_search(train, predicates)
        seen = {law.describe() for law in candidates}
        for law in proposals:
            if law.describe() not in seen:
                candidates.append(law)
                seen.add(law.describe())
                considered += 1
        if not candidates:
            return DiscoveryOutcome(
                None,
                refusal="no conjunction beat the base rate with enough support on train",
                candidates_considered=considered,
            )

        heldout_rate = base_rate(heldout)
        transfer_rate = base_rate(transfer)
        rejected: list[str] = []

        for candidate in candidates:
            # Pruning on held-out data makes the validation split part of fit.
            # The rule must be frozen before its first held-out observation.
            pruned, ablation = self.prune(candidate, train)
            heldout_score = score_split(pruned, heldout)
            if heldout_score.support < self.min_support:
                rejected.append(f"{pruned.describe()}: held-out support {heldout_score.support}")
                continue
            heldout_lift = heldout_score.lift(heldout_rate)
            if heldout_lift < self.min_heldout_lift:
                rejected.append(f"{pruned.describe()}: held-out lift {heldout_lift:.2f}")
                continue

            p_value = min(1.0, exact_conditional_p_value(pruned, heldout)
                          * len(candidates))
            if p_value > self.max_p_value:
                rejected.append(f"{pruned.describe()}: p={p_value:.3f}")
                continue

            transfer_score = score_split(pruned, transfer)
            transfer_lift = transfer_score.lift(transfer_rate)
            transfer_p_value = min(
                1.0, exact_conditional_p_value(pruned, transfer) * len(candidates)
            )
            if (transfer_score.support < self.min_support or transfer_lift <= 1.0
                    or transfer_p_value > self.max_p_value):
                rejected.append(
                    f"{pruned.describe()}: did not transfer "
                    f"(support {transfer_score.support}, lift {transfer_lift:.2f}, "
                    f"p={transfer_p_value:.3f})"
                )
                continue

            evidence = LawEvidence(
                base_rate=round(overall_rate, 6),
                train=score_split(pruned, train),
                heldout=heldout_score,
                transfer=transfer_score,
                heldout_lift=round(heldout_lift, 6),
                transfer_lift=round(transfer_lift, 6),
                p_value=round(p_value, 6),
                permutations=0,
                ablation=ablation,
                validation_method="exact_conditional_bonferroni",
                familywise_candidates=len(candidates),
                transfer_p_value=round(transfer_p_value, 6),
            )
            discovered = DiscoveredLaw(
                law=pruned,
                evidence=evidence,
                discovered_at=time.time(),
                observation_count=len(episodes),
            )
            logger.info(
                "🔬 OntologyGenesis discovered: %s (held-out lift %.2f, p=%.3f, "
                "transfer lift %.2f)",
                pruned.describe(),
                heldout_lift,
                p_value,
                transfer_lift,
            )
            return DiscoveryOutcome(discovered, candidates_considered=considered)

        return DiscoveryOutcome(
            None,
            refusal="; ".join(rejected[:4]) or "every candidate failed validation",
            candidates_considered=considered,
        )


__all__ = [
    "CandidateLaw",
    "DiscoveredLaw",
    "DiscoveryOutcome",
    "LawEvidence",
    "MAX_CONJUNCTS",
    "MAX_P_VALUE",
    "MIN_HELDOUT_LIFT",
    "MIN_SUPPORT",
    "NULL_PERMUTATIONS",
    "Observation",
    "OntologyDiscovery",
    "Predicate",
    "SplitScore",
    "base_rate",
    "candidate_predicates",
    "exact_conditional_p_value",
    "score_split",
]
