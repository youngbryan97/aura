"""core/agency/decision_preference_learner.py — choices that author future preferences.

The InitiativeArbiter chooses among competing initiatives by a weighted score over eight
dimensions (urgency, novelty, identity_relevance, tension_resolution, expected_value,
resource_cost, social_appropriateness, continuity). Those weights came from defaults +
identity + momentary drive boosts — but nothing ever updated them from *how her past
choices actually turned out*. So her preferences were given to her; they were not authored
by her own lived consequences.

This closes that loop. It is the functional core of self-authored decision-making: when a
choice she made turns out well, the dimensions that *distinguished* that choice are
reinforced; when it turns out badly, they are attenuated. Over time the relative weighting
of her values is shaped by her own history — the same way a person comes to learn what they
actually care about by living with the results of what they chose.

Honest boundaries, enforced in code (not prose):
* It learns the *weighting* of her existing decision dimensions. It never invents a new
  value or dimension — the value space is fixed (the #47 governance bound:
  composition within designed drives, not unbounded genesis).
* Multipliers are clamped to ``[W_MIN, W_MAX]`` and drift is rate-limited, so a single
  good or bad outcome cannot capture her, and the system cannot run away.
* Credit assignment is contextual-bandit-flavoured and fully deterministic given the
  rewards — measurable and auditable without any LLM in the loop.

This is machinery: persistent weights that move from measured outcomes and feed straight
back into which initiative she picks. Not a prompt.
"""
from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.runtime.atomic_writer import atomic_write_text_behind
from core.runtime.errors import record_degradation
from core.runtime.state_ownership import state_root

logger = logging.getLogger("Aura.DecisionPreferenceLearner")

# The fixed decision dimensions (mirrors InitiativeArbiter.DIMENSION_NAMES). The learner
# may reweight these; it may never add to them.
DIMENSIONS = (
    "urgency",
    "novelty",
    "identity_relevance",
    "tension_resolution",
    "expected_value",
    "resource_cost",
    "social_appropriateness",
    "continuity",
)

W_MIN = 0.5          # a learned multiplier can never silence a dimension entirely
W_MAX = 1.5          # nor let one dominate
LEARNING_RATE = 0.08
MAX_STEP = 0.06      # per-resolution drift cap on any single multiplier
DECAY_TO_NEUTRAL = 0.0005  # slow pull back toward 1.0 so stale learning fades


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


@dataclass
class PendingChoice:
    choice_id: str
    chosen_scores: dict[str, float]   # the winning initiative's per-dimension scores
    pool_mean: dict[str, float]       # mean score per dimension across the option pool
    goal: str
    weights_used: dict[str, float]
    receipt_id: str | None = None
    created_at: float = field(default_factory=lambda: time.time())


class DecisionPreferenceLearner:
    """Learns per-dimension weight multipliers from the outcomes of chosen initiatives."""

    SERVICE_NAME = "decision_preference_learner"

    def __init__(self, state_path: str | Path | None = None) -> None:
        self._lock = threading.RLock()
        self._multipliers: dict[str, float] = {d: 1.0 for d in DIMENSIONS}
        self._pending: dict[str, PendingChoice] = {}
        self._resolved_count = 0
        self._reward_history: deque[float] = deque(maxlen=200)
        if state_path is None:
            try:
                from core.config import config

                state_path = Path(config.paths.data_dir) / "cognitive" / "decision_preferences.json"
            except (ImportError, AttributeError, RuntimeError) as exc:
                record_degradation("decision_preference_learner", exc, severity="debug")
                state_path = state_root() / "data" / "cognitive" / "decision_preferences.json"
        self._state_path = Path(state_path)
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        self._load()

    # ── feed-forward: the weights her choices are actually made with ─────────
    def effective_weights(self, base_weights: dict[str, float]) -> dict[str, float]:
        """Apply the learned multipliers to the arbiter's base weights.

        Dimensions she has learned actually serve her are weighted up; ones that have
        repeatedly disappointed are weighted down — within the clamped band.
        """
        with self._lock:
            return {
                dim: float(base_weights.get(dim, 0.5)) * self._multipliers.get(dim, 1.0)
                for dim in base_weights
            }

    def multipliers(self) -> dict[str, float]:
        with self._lock:
            return dict(self._multipliers)

    # ── capture: record a choice as it is made ───────────────────────────────
    def record_choice(
        self,
        *,
        chosen_scores: dict[str, float],
        pool_scores: list[dict[str, float]],
        goal: str = "",
        weights_used: dict[str, float] | None = None,
        expected_value: float = 0.5,
    ) -> str:
        """Register a made choice and open an outcome receipt for it.

        ``pool_scores`` is every option's per-dimension scores, so credit assignment can
        later ask "which dimensions did the *chosen* option score unusually high on,
        relative to the alternatives?" — those are the dimensions the choice expressed.
        """
        choice_id = f"choice-{uuid.uuid4().hex[:10]}"
        pool = pool_scores or [chosen_scores]
        pool_mean = {
            dim: sum(float(s.get(dim, 0.0)) for s in pool) / max(1, len(pool))
            for dim in DIMENSIONS
        }
        pending = PendingChoice(
            choice_id=choice_id,
            chosen_scores={d: float(chosen_scores.get(d, 0.0)) for d in DIMENSIONS},
            pool_mean=pool_mean,
            goal=str(goal or "")[:160],
            weights_used=dict(weights_used or {}),
        )
        pending.receipt_id = self._open_receipt(pending, expected_value)
        with self._lock:
            self._pending[choice_id] = pending
            # Bound memory: forget the oldest unresolved choices.
            if len(self._pending) > 256:
                oldest = sorted(self._pending.values(), key=lambda c: c.created_at)[:64]
                for c in oldest:
                    self._pending.pop(c.choice_id, None)
        return choice_id

    def _open_receipt(self, pending: PendingChoice, expected_value: float) -> str | None:
        try:
            from core.cognition.outcome_ledger import CreditSource, get_outcome_ledger

            top_dim = max(pending.chosen_scores, key=pending.chosen_scores.get) if pending.chosen_scores else "expected_value"
            return get_outcome_ledger().open(
                action=f"decision:{pending.goal or 'initiative'}",
                expected=float(expected_value),
                sources=[CreditSource("decision_dimension", top_dim, 1.0)],
                category="decision",
                context={"choice_id": pending.choice_id},
            )
        except (ImportError, AttributeError, RuntimeError, OSError, ValueError, TypeError) as exc:
            record_degradation("decision_preference_learner", exc, severity="debug")
            return None

    # ── learn: resolve a choice's outcome and update preferences ─────────────
    def resolve_choice(self, choice_id: str, reward: float) -> dict[str, float]:
        """Fold a choice's outcome into the learned weights.

        ``reward`` ∈ [-1, 1]: +1 the choice served her values, -1 it worked against them.
        Credit goes to the dimensions on which the chosen option *differed* from the pool:
        those are what the choice actually expressed. Reinforce them on a positive outcome,
        attenuate them on a negative one — bounded and rate-limited.
        """
        reward = _clamp(float(reward), -1.0, 1.0)
        with self._lock:
            pending = self._pending.pop(choice_id, None)
            if pending is None:
                return dict(self._multipliers)
            for dim in DIMENSIONS:
                # Signed salience: how much this dimension distinguished the choice.
                distinctiveness = pending.chosen_scores[dim] - pending.pool_mean.get(dim, 0.0)
                delta = LEARNING_RATE * reward * distinctiveness
                delta = _clamp(delta, -MAX_STEP, MAX_STEP)
                updated = self._multipliers[dim] + delta
                # Slow decay back toward neutral so old lessons fade rather than calcify.
                updated += (1.0 - updated) * DECAY_TO_NEUTRAL
                self._multipliers[dim] = _clamp(updated, W_MIN, W_MAX)
            self._resolved_count += 1
            self._reward_history.append(reward)
            self._save()
            multipliers = dict(self._multipliers)

        # Resolve the provenance receipt outside the lock.
        if pending.receipt_id:
            try:
                from core.cognition.outcome_ledger import get_outcome_ledger

                # observed maps reward [-1,1] → [0,1] so the ledger's own credit math agrees.
                get_outcome_ledger().resolve(
                    pending.receipt_id, 0.5 + 0.5 * reward, note="decision_preference_learner"
                )
            except (ImportError, AttributeError, RuntimeError, OSError, ValueError, TypeError) as exc:
                record_degradation("decision_preference_learner", exc, severity="debug")
        logger.info(
            "🧭 [DecisionPreference] resolved %s reward=%+.2f → multipliers updated (%d total)",
            choice_id, reward, self._resolved_count,
        )
        return multipliers

    # ── persistence ──────────────────────────────────────────────────────────
    def _save(self) -> None:
        try:
            payload = {
                "multipliers": self._multipliers,
                "resolved_count": self._resolved_count,
                "saved_at": time.time(),
            }
            # Behind the loop: a choice is resolved from the affect phase, and
            # this save's fsync would otherwise sit on the event loop.
            atomic_write_text_behind(
                self._state_path,
                json.dumps(payload, indent=2),
                encoding="utf-8",
            )
        except (OSError, TypeError, ValueError) as exc:
            record_degradation("decision_preference_learner", exc, severity="debug")

    def _load(self) -> None:
        if not self._state_path.exists():
            return
        try:
            data = json.loads(self._state_path.read_text(encoding="utf-8"))
            stored = data.get("multipliers", {})
            for dim in DIMENSIONS:
                if dim in stored:
                    self._multipliers[dim] = _clamp(float(stored[dim]), W_MIN, W_MAX)
            self._resolved_count = int(data.get("resolved_count", 0) or 0)
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            record_degradation("decision_preference_learner", exc, severity="debug")

    # ── introspection ─────────────────────────────────────────────────────────
    def stats(self) -> dict[str, Any]:
        with self._lock:
            mean_reward = (
                sum(self._reward_history) / len(self._reward_history)
                if self._reward_history else 0.0
            )
            return {
                "service": self.SERVICE_NAME,
                "multipliers": dict(self._multipliers),
                "resolved_count": self._resolved_count,
                "pending_choices": len(self._pending),
                "mean_recent_reward": round(mean_reward, 4),
                "state_path": str(self._state_path),
            }


_engine: DecisionPreferenceLearner | None = None
_engine_lock = threading.Lock()


def get_decision_preference_learner() -> DecisionPreferenceLearner:
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = DecisionPreferenceLearner()
                _register_in_container(_engine)
    return _engine


def _register_in_container(engine: DecisionPreferenceLearner) -> None:
    try:
        from core.container import ServiceContainer

        if not ServiceContainer.has(DecisionPreferenceLearner.SERVICE_NAME):
            reg = getattr(ServiceContainer, "register_instance", None)
            if callable(reg):
                reg(DecisionPreferenceLearner.SERVICE_NAME, engine,
                    required=False, registered_by="decision_preference_learner")
    except (ImportError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
        record_degradation("decision_preference_learner_register", exc, severity="debug")


def reset_decision_preference_learner_for_test() -> None:
    global _engine
    _engine = None
