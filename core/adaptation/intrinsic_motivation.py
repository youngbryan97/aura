"""core/adaptation/intrinsic_motivation.py -- Intrinsic Motivation Signals
==========================================================================
Computes intrinsic motivation signals that feed into the DynamicValueGraph
as evidence for value evolution. Two types of intrinsic motivation:

1. Competence-based (δC_g): measures rate of competence improvement per
   goal. Actions that produce steep competence improvement get intrinsic
   reward. Based on Oudeyer & Kaplan's "Intrinsic Motivation Systems for
   Autonomous Mental Development" (2007).

2. Novelty-based: maintains a density model of experienced states using
   a sliding-window kernel density estimator. Novel states (low density)
   produce intrinsic reward. Based on Bellemare et al.'s count-based
   exploration (2016), adapted for continuous state spaces.

These signals feed into the DynamicValueGraph as EvidenceType entries.
When a cluster of experiences consistently yields high intrinsic reward
but maps to no existing drive, a candidate value is proposed.

Gate: No new value influences behavior until it passes through the
Candidate → Sandbox → Evidence → Adoption pipeline in the DVG.
"""
from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from core.memory.retention_policy import working_history_retention_policy

logger = logging.getLogger("Aura.IntrinsicMotivation")


@dataclass
class CompetenceRecord:
    """Tracks competence for a specific goal over time."""
    goal_name: str
    # Competence history: (timestamp, competence_score)
    history: deque[tuple[float, float]] = field(
        default_factory=lambda: deque(maxlen=200)
    )
    # Cumulative attempts and successes
    attempts: int = 0
    successes: int = 0

    def current_competence(self) -> float:
        """Current competence estimate (success rate, smoothed)."""
        if self.attempts == 0:
            return 0.0
        return self.successes / self.attempts

    def competence_derivative(self, window: int = 20) -> float:
        """Rate of competence change (δC_g).

        Positive = getting better, negative = getting worse.
        """
        if len(self.history) < 3:
            return 0.0
        recent = list(self.history)[-window:]
        if len(recent) < 3:
            return 0.0
        # Linear regression slope
        t_vals = np.array([r[0] for r in recent], dtype=np.float64)
        c_vals = np.array([r[1] for r in recent], dtype=np.float64)
        t_vals -= t_vals[0]  # Normalize to start at 0
        t_range = t_vals[-1] - t_vals[0]
        if t_range < 1e-6:
            return 0.0
        t_norm = t_vals / t_range
        t_mean = np.mean(t_norm)
        c_mean = np.mean(c_vals)
        denom = np.sum((t_norm - t_mean) ** 2)
        if denom < 1e-10:
            return 0.0
        return float(np.sum((t_norm - t_mean) * (c_vals - c_mean)) / denom)

    def record(self, success: bool) -> None:
        """Record an attempt and its outcome."""
        self.attempts += 1
        if success:
            self.successes += 1
        self.history.append((time.time(), self.current_competence()))


@dataclass
class NoveltyConfig:
    """Configuration for the novelty density model."""
    max_states: int = 5000         # Max states in the archive
    bandwidth: float = 0.5         # KDE bandwidth
    novelty_bonus_scale: float = 1.0  # Scale factor for novelty reward
    min_novelty: float = 0.01      # Below this, state is not novel


@dataclass
class IntrinsicReward:
    """An intrinsic reward signal."""
    source: str       # "competence" or "novelty"
    goal_name: str    # Which goal/state this relates to
    reward: float     # The intrinsic reward value
    details: dict[str, float]
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "goal_name": self.goal_name,
            "reward": round(self.reward, 6),
            "details": {k: round(v, 6) for k, v in self.details.items()},
            "timestamp": self.timestamp,
        }


class CompetenceMotivation:
    """Competence-based intrinsic motivation: δC_g per goal.

    Actions that produce steep competence improvement get high
    intrinsic reward. This drives the system to practice skills
    where it's improving fastest (the "learning progress" signal).
    """

    def __init__(self) -> None:
        self._goals: dict[str, CompetenceRecord] = {}
        self._reward_history: deque[IntrinsicReward] = deque(
            maxlen=working_history_retention_policy("AURA_COMPETENCE_REWARD_HISTORY_MAX").max_items
        )

    def record_attempt(self, goal_name: str, success: bool) -> IntrinsicReward:
        """Record a goal attempt and compute competence-based reward.

        Args:
            goal_name: Name of the goal being pursued.
            success: Whether the attempt succeeded.

        Returns:
            IntrinsicReward with competence derivative as reward signal.
        """
        if goal_name not in self._goals:
            self._goals[goal_name] = CompetenceRecord(goal_name=goal_name)

        record = self._goals[goal_name]
        record.record(success)

        # Reward POSITIVE learning progress only. abs(δC) paid out exactly the
        # same for competence collapsing as for competence improving, so a goal
        # Aura was rapidly getting worse at looked maximally worth practising —
        # the opposite of the documented intent, and a reward signal that grows
        # as performance degrades.
        delta_c = record.competence_derivative()
        reward = max(0.0, delta_c)

        ir = IntrinsicReward(
            source="competence",
            goal_name=goal_name,
            reward=reward,
            details={
                "competence": record.current_competence(),
                "delta_c": delta_c,
                "attempts": float(record.attempts),
                "successes": float(record.successes),
            },
        )
        self._reward_history.append(ir)
        return ir

    def get_most_improving_goals(self, top_k: int = 5) -> list[tuple[str, float]]:
        """Return goals with highest learning progress."""
        deltas = []
        for name, record in self._goals.items():
            delta = record.competence_derivative()
            # "Most improving", not "most changing": a goal in freefall is not
            # an improving one.
            deltas.append((name, max(0.0, delta)))
        deltas.sort(key=lambda x: x[1], reverse=True)
        return deltas[:top_k]

    def get_status(self) -> dict[str, Any]:
        return {
            "n_goals": len(self._goals),
            "goals": {
                name: {
                    "competence": round(r.current_competence(), 4),
                    "delta_c": round(r.competence_derivative(), 6),
                    "attempts": r.attempts,
                }
                for name, r in self._goals.items()
            },
            "total_rewards": len(self._reward_history),
        }


class NoveltyMotivation:
    """Novelty-based intrinsic motivation via state density estimation.

    Maintains a density model of experienced states. States that are
    far from any previously seen state produce high novelty reward,
    encouraging exploration of new regions of state space.

    Uses a simple sliding-window approach with L2 distance.
    """

    def __init__(self, config: NoveltyConfig | None = None) -> None:
        self._config = config or NoveltyConfig()
        self._state_archive: deque[np.ndarray] = deque(
            maxlen=self._config.max_states
        )
        self._reward_history: deque[IntrinsicReward] = deque(
            maxlen=working_history_retention_policy("AURA_NOVELTY_REWARD_HISTORY_MAX").max_items
        )
        self._dim: int | None = None

    def compute_novelty(self, state: np.ndarray) -> float:
        """Compute novelty of a state relative to the archive.

        Uses negative log pseudo-count: novelty = 1 / (count + 1)
        where count is approximated by kernel density.

        Args:
            state: State vector to evaluate.

        Returns:
            Novelty score (higher = more novel).
        """
        if len(self._state_archive) == 0:
            return 1.0

        state = np.asarray(state, dtype=np.float64).ravel()
        if state.size == 0 or not np.all(np.isfinite(state)):
            return 0.0

        # Only compare against archived states of the SAME width. The archive
        # never enforced a dimension, so a later state of a different length
        # made the array construction ragged or the subtraction raise — after
        # the caller had already been told novelty was being measured.
        same_dim = [row for row in self._state_archive if len(row) == state.size]
        if not same_dim:
            return 1.0
        archive = np.asarray(same_dim, dtype=np.float64)
        distances = np.linalg.norm(archive - state[np.newaxis, :], axis=1)

        # Kernel density estimate (Gaussian kernel). Each kernel is in (0, 1],
        # so the mean density is too: 1.0 means "identical to everything in the
        # archive", near 0 means "far from all of it".
        h = max(1e-6, float(self._config.bandwidth))
        density = float(np.mean(np.exp(-0.5 * (distances / h) ** 2)))

        # Novelty is the complement of density. The old formula was
        # min(1, 0.1/density): with even a handful of archived states, density
        # is small enough that this clamps to 1.0 — an EXACT repeat of a seen
        # state still scored maximally novel. The complement is bounded,
        # monotone, and actually discriminates.
        novelty = max(0.0, min(1.0, 1.0 - density))

        # min_novelty was declared and then never used. Honour it: below the
        # floor the state is not novel at all, rather than faintly novel.
        if novelty < float(self._config.min_novelty):
            return 0.0

        return float(novelty)

    def observe_and_reward(
        self, state: np.ndarray, context_name: str = "exploration"
    ) -> IntrinsicReward:
        """Observe a state, compute novelty reward, and archive it.

        Args:
            state: State vector to observe.
            context_name: Name/label for this observation context.

        Returns:
            IntrinsicReward with novelty score.
        """
        state = np.asarray(state, dtype=np.float64).ravel()

        if self._dim is None:
            self._dim = len(state)

        novelty = self.compute_novelty(state)
        reward = novelty * self._config.novelty_bonus_scale

        # Archive the state
        self._state_archive.append(state.copy())

        ir = IntrinsicReward(
            source="novelty",
            goal_name=context_name,
            reward=reward,
            details={
                "novelty": novelty,
                "archive_size": float(len(self._state_archive)),
                "state_norm": float(np.linalg.norm(state)),
            },
        )
        self._reward_history.append(ir)
        return ir

    def get_status(self) -> dict[str, Any]:
        return {
            "archive_size": len(self._state_archive),
            "max_states": self._config.max_states,
            "bandwidth": self._config.bandwidth,
            "total_rewards": len(self._reward_history),
            "state_dim": self._dim,
        }


class IntrinsicMotivationEngine:
    """Unified intrinsic motivation engine combining competence and novelty.

    Aggregates signals from both motivation types and feeds them into
    the DynamicValueGraph as evidence for value evolution.

    Also detects clusters of high-IM experiences that map to no existing
    drive, triggering candidate value proposals.
    """

    def __init__(self, novelty_config: NoveltyConfig | None = None) -> None:
        self.competence = CompetenceMotivation()
        self.novelty = NoveltyMotivation(novelty_config)

        # Track high-reward clusters for value proposal
        self._high_reward_clusters: dict[str, list[IntrinsicReward]] = defaultdict(list)
        self._proposal_threshold: int = 10  # N high-reward events → propose
        self._reward_threshold: float = 0.3  # Reward above this = "high"
        #: Per-context cap on retained proposal evidence. These lists were
        #: never capped, expired, or cleared, so a long-lived process grew them
        #: without bound.
        self._max_cluster_evidence: int = 200
        #: How far into each reward history has already been submitted as value
        #: evidence. Without this the whole history was replayed on every feed.
        self._fed_competence: int = 0
        self._fed_novelty: int = 0
        #: Values already submitted as proposals, so one cluster proposes once.
        self._proposed_values: set = set()

    def record_competence(self, goal_name: str, success: bool) -> IntrinsicReward:
        """Record a competence attempt and get reward."""
        reward = self.competence.record_attempt(goal_name, success)
        self._track_for_proposals(reward)
        return reward

    def record_novelty(
        self, state: np.ndarray, context: str = "exploration"
    ) -> IntrinsicReward:
        """Record a novel state observation and get reward."""
        reward = self.novelty.observe_and_reward(state, context)
        self._track_for_proposals(reward)
        return reward

    def _track_for_proposals(self, reward: IntrinsicReward) -> None:
        """Track high-reward experiences for potential value proposals."""
        if reward.reward >= self._reward_threshold:
            cluster = self._high_reward_clusters[reward.goal_name]
            cluster.append(reward)
            # Bounded, newest-wins. An unbounded per-context list grew forever
            # in a long-lived process; the proposal only ever reads a count and
            # a mean, both of which the recent window represents faithfully.
            if len(cluster) > self._max_cluster_evidence:
                del cluster[: len(cluster) - self._max_cluster_evidence]

    def check_value_proposals(
        self, existing_drives: list[str] | None = None
    ) -> list[dict[str, Any]]:
        """Check if any reward clusters should trigger a new value proposal.

        Args:
            existing_drives: List of existing value/drive names to check
                against. Clusters that map to existing drives are ignored.

        Returns:
            List of proposed new values with evidence.
        """
        existing = set(existing_drives or [])
        proposals = []

        for context_name, rewards in self._high_reward_clusters.items():
            if len(rewards) < self._proposal_threshold:
                continue

            # Skip if this already maps to an existing drive
            if context_name in existing:
                continue

            # Compute aggregate signal
            mean_reward = float(np.mean([r.reward for r in rewards]))
            sources = set(r.source for r in rewards)

            proposal = {
                "proposed_value_name": context_name,
                "evidence_count": len(rewards),
                "mean_reward": round(mean_reward, 4),
                "sources": list(sources),
                "first_seen": rewards[0].timestamp,
                "last_seen": rewards[-1].timestamp,
                "reasoning": (
                    f"Cluster of {len(rewards)} high-reward experiences "
                    f"(mean={mean_reward:.3f}) from {', '.join(sources)} "
                    f"around '{context_name}' with no matching existing drive"
                ),
            }
            proposals.append(proposal)

        return proposals

    def feed_to_value_graph(self) -> int:
        """Feed accumulated intrinsic rewards to the DynamicValueGraph.

        Returns:
            Number of evidence entries submitted.
        """
        try:
            from core.adaptation.dynamic_value_graph import (
                EvidenceType,
                ValueEvidence,
                get_dynamic_value_graph,
            )
        except ImportError:
            return 0

        graph = get_dynamic_value_graph()
        count = 0

        # Only rewards observed SINCE THE LAST FEED are submitted. Without a
        # cursor every call replayed the entire history, so the same
        # observation was counted again on every feed — inflating DVG evidence
        # counts and apparent diversity without a single new observation.
        competence_history = list(self.competence._reward_history)
        novelty_history = list(self.novelty._reward_history)

        # Feed competence rewards
        for reward in competence_history[self._fed_competence:]:
            if reward.reward > 0.01:  # Only meaningful signals
                graph.record_evidence(ValueEvidence(
                    # NOT free-energy reduction: nothing here computes a
                    # free-energy quantity. Competence reward is derived from
                    # recorded attempt outcomes, which is what OUTCOME_QUALITY
                    # means.
                    evidence_type=EvidenceType.OUTCOME_QUALITY,
                    value_name=reward.goal_name,
                    signal=min(1.0, reward.reward),
                    confidence=min(1.0, reward.details.get("attempts", 1) / 10.0),
                    source="intrinsic_motivation.competence",
                    context=f"δC={reward.details.get('delta_c', 0):.4f}",
                ))
                count += 1
        self._fed_competence = len(competence_history)

        # Feed novelty rewards
        for reward in novelty_history[self._fed_novelty:]:
            if reward.reward > 0.01:
                graph.record_evidence(ValueEvidence(
                    # Exploration pull, not free-energy reduction.
                    evidence_type=EvidenceType.ENGAGEMENT,
                    value_name=reward.goal_name,
                    signal=min(1.0, reward.reward),
                    confidence=0.5,  # Novelty is inherently uncertain
                    source="intrinsic_motivation.novelty",
                    context=f"novelty={reward.details.get('novelty', 0):.4f}",
                ))
                count += 1
        self._fed_novelty = len(novelty_history)

        # Check for value proposals. A value is proposed ONCE: without this the
        # same cluster was resubmitted on every feed, so a single proposal
        # accumulated unbounded supporting "evidence" it had never earned.
        existing = list(graph.get_adopted_values().keys())
        proposals = [
            proposal
            for proposal in self.check_value_proposals(existing)
            if proposal["proposed_value_name"] not in self._proposed_values
        ]
        for proposal in proposals:
            self._proposed_values.add(proposal["proposed_value_name"])
            graph.record_evidence(ValueEvidence(
                evidence_type=EvidenceType.SELF_REPORT,
                value_name=proposal["proposed_value_name"],
                signal=proposal["mean_reward"],
                confidence=min(1.0, proposal["evidence_count"] / 20.0),
                source="intrinsic_motivation.proposal",
                context=proposal["reasoning"][:200],
            ))
            count += 1
            logger.info(
                "Value proposal submitted: '%s' (%d evidence, reward=%.3f)",
                proposal["proposed_value_name"],
                proposal["evidence_count"],
                proposal["mean_reward"],
            )

        return count

    def get_status(self) -> dict[str, Any]:
        return {
            "competence": self.competence.get_status(),
            "novelty": self.novelty.get_status(),
            "high_reward_clusters": {
                k: len(v) for k, v in self._high_reward_clusters.items()
            },
        }
