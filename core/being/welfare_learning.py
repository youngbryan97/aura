"""core/being/welfare_learning.py — Temporal Credit Assignment for Welfare.

Connects actions to LATER state changes, not just immediate reward.
If a shortcut causes later memory confusion, the shortcut becomes aversive.
If verification prevents contradiction, verification becomes preferred.

Design:
  - Maintains a causal ledger: action → delayed welfare delta
  - Uses exponential decay credit assignment (recent actions get more credit)
  - Updates the WelfareState aversion memory based on learned associations
  - Tracks which domains/actions reliably improve or harm welfare
"""
from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field

from core.being.welfare_transaction import WelfareTransaction

logger = logging.getLogger("Aura.WelfareLearning")


def _clip(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(v)))


@dataclass
class CausalAssociation:
    """A learned link between an action domain and a welfare outcome."""
    domain: str
    welfare_dimension: str          # e.g. "distress", "truth_integrity"
    direction: float                # positive = helps, negative = harms
    strength: float = 0.0           # 0-1, how strong the association is
    sample_count: int = 0
    last_updated: float = field(default_factory=time.time)


@dataclass
class WelfareLedgerEntry:
    """A timestamped record of welfare state for credit assignment."""
    timestamp: float
    welfare_score: float
    distress: float
    confidence: float
    integrity_guard: float
    action_domain: str = ""
    action_id: str = ""


class WelfareLearning:
    """Temporal credit assignment: learns which actions help or harm welfare.

    Usage:
        learner = WelfareLearning.get()
        learner.record_welfare_snapshot(current_welfare, recent_action_domain)
        learner.update_associations()
        harm_score = learner.predicted_harm("self_modification")
        benefit_score = learner.predicted_benefit("reflection")
    """

    _instance: WelfareLearning | None = None

    # Learning parameters
    CREDIT_DECAY = 0.85              # discount per step for credit assignment
    LEARNING_RATE = 0.08             # how fast associations update
    MIN_SAMPLES = 3                  # min observations before strong association
    ASSOCIATION_DECAY = 0.995        # slow decay of unused associations

    def __init__(self) -> None:
        self._ledger: deque[WelfareLedgerEntry] = deque(maxlen=500)
        self._associations: dict[str, dict[str, CausalAssociation]] = defaultdict(dict)
        self._domain_outcome_history: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=100)
        )
        self._last_processed_ledger_timestamp = 0.0
        self._learned_transaction_ids: set[str] = set()
        self._lesioned = False

    @classmethod
    def get(cls) -> WelfareLearning:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        cls._instance = None

    def record_welfare_snapshot(
        self,
        welfare_score: float,
        distress: float,
        confidence: float,
        integrity_guard: float,
        action_domain: str = "",
        action_id: str = "",
    ) -> None:
        """Record current welfare state for later credit assignment."""
        self._ledger.append(WelfareLedgerEntry(
            timestamp=time.time(),
            welfare_score=welfare_score,
            distress=distress,
            confidence=confidence,
            integrity_guard=integrity_guard,
            action_domain=action_domain,
            action_id=action_id,
        ))

    def update_associations(self) -> int:
        """Scan ledger for temporal patterns and update causal associations.

        Returns number of associations updated.
        """
        if self._lesioned or len(self._ledger) < 4:
            return 0

        entries = list(self._ledger)
        updated = 0

        newest_processed = self._last_processed_ledger_timestamp
        for i in range(1, len(entries)):
            current = entries[i]
            if current.timestamp <= self._last_processed_ledger_timestamp:
                continue
            welfare_delta = current.welfare_score - entries[i - 1].welfare_score
            distress_delta = current.distress - entries[i - 1].distress

            # Assign credit to recent actions with exponential decay
            credit = 1.0
            for j in range(i - 1, max(-1, i - 6), -1):
                past = entries[j]
                if past.action_domain:
                    self._update_single_association(
                        domain=past.action_domain,
                        dimension="welfare_score",
                        delta=welfare_delta * credit,
                    )
                    self._update_single_association(
                        domain=past.action_domain,
                        dimension="distress",
                        delta=distress_delta * credit,
                    )
                    updated += 1
                credit *= self.CREDIT_DECAY
            newest_processed = max(newest_processed, current.timestamp)

        self._last_processed_ledger_timestamp = newest_processed
        return updated

    def _update_single_association(
        self, domain: str, dimension: str, delta: float
    ) -> None:
        """Update a single causal association with new evidence."""
        key = dimension
        if key not in self._associations[domain]:
            self._associations[domain][key] = CausalAssociation(
                domain=domain,
                welfare_dimension=dimension,
                direction=0.0,
                strength=0.0,
            )

        assoc = self._associations[domain][key]
        # EMA update
        assoc.direction = (
            assoc.direction * (1 - self.LEARNING_RATE)
            + delta * self.LEARNING_RATE
        )
        assoc.strength = _clip(
            assoc.strength + 0.01  # strength grows with evidence
        )
        assoc.sample_count += 1
        assoc.last_updated = time.time()

        # Update domain outcome history
        self._domain_outcome_history[domain].append(delta)

    def learn_from_transactions(self) -> int:
        """Pull recent transactions and learn from their outcomes."""
        records = WelfareTransaction.recent_records(50)
        learned = 0

        for record in records:
            if record.tx_id in self._learned_transaction_ids:
                continue
            self._learned_transaction_ids.add(record.tx_id)
            w_delta = record.welfare_delta
            welfare_change = w_delta.get("welfare_score", 0.0)
            distress_change = w_delta.get("distress", 0.0)

            if abs(welfare_change) > 0.01 or abs(distress_change) > 0.01:
                self._update_single_association(
                    domain=record.domain,
                    dimension="welfare_score",
                    delta=welfare_change,
                )
                self._update_single_association(
                    domain=record.domain,
                    dimension="distress",
                    delta=distress_change,
                )
                learned += 1

            if not record.integrity_preserved:
                self._update_single_association(
                    domain=record.domain,
                    dimension="integrity",
                    delta=-0.2,  # integrity violations are strongly negative
                )
                learned += 1

            if not record.truth_preserved:
                self._update_single_association(
                    domain=record.domain,
                    dimension="truth",
                    delta=-0.25,
                )
                learned += 1

        return learned

    def predicted_harm(self, domain: str) -> float:
        """Predicted harm for an action in this domain (0 = safe, 1 = very harmful)."""
        if domain not in self._associations:
            return 0.0

        harm = 0.0
        for dim, assoc in self._associations[domain].items():
            if assoc.sample_count >= self.MIN_SAMPLES:
                if dim == "distress" and assoc.direction > 0:
                    harm += assoc.direction * assoc.strength
                elif dim == "welfare_score" and assoc.direction < 0:
                    harm += abs(assoc.direction) * assoc.strength
                elif dim in ("integrity", "truth") and assoc.direction < 0:
                    harm += abs(assoc.direction) * assoc.strength * 1.5  # extra weight

        return _clip(harm)

    def predicted_benefit(self, domain: str) -> float:
        """Predicted benefit for an action in this domain (0 = neutral, 1 = very beneficial)."""
        if domain not in self._associations:
            return 0.0

        benefit = 0.0
        for dim, assoc in self._associations[domain].items():
            if assoc.sample_count >= self.MIN_SAMPLES:
                if dim == "welfare_score" and assoc.direction > 0:
                    benefit += assoc.direction * assoc.strength
                elif dim == "distress" and assoc.direction < 0:
                    benefit += abs(assoc.direction) * assoc.strength

        return _clip(benefit)

    def should_avoid(self, domain: str) -> bool:
        """Whether this domain has accumulated enough negative evidence to avoid."""
        return self.predicted_harm(domain) > 0.4

    def should_prefer(self, domain: str) -> bool:
        """Whether this domain has accumulated enough positive evidence to prefer."""
        return self.predicted_benefit(domain) > 0.3

    def get_associations(self, domain: str) -> dict[str, CausalAssociation]:
        """Return all learned associations for a domain."""
        return dict(self._associations.get(domain, {}))

    def all_domains(self) -> list[str]:
        return list(self._associations.keys())

    def domain_summary(self) -> dict[str, dict[str, float]]:
        """Summary of learned harm/benefit per domain."""
        result = {}
        for domain in self._associations:
            result[domain] = {
                "predicted_harm": round(self.predicted_harm(domain), 4),
                "predicted_benefit": round(self.predicted_benefit(domain), 4),
                "should_avoid": self.should_avoid(domain),
                "should_prefer": self.should_prefer(domain),
            }
        return result

    def lesion(self) -> None:
        self._lesioned = True

    def restore(self) -> None:
        self._lesioned = False

    @property
    def is_lesioned(self) -> bool:
        return self._lesioned
