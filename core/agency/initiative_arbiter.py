"""core/agency/initiative_arbiter.py — The ONE Chooser
======================================================
Takes ALL pending initiatives from AuraState and scores each on 8
dimensions, returning the single highest-scoring initiative for
execution.

This is the decision bottleneck — every autonomous action Aura takes
flows through this arbiter. No initiative bypasses scoring.
"""
from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass

from core.autonomy.research_goal_filter import (
    is_stale_or_prompt_scaffold_goal,
    normalize_goal_text,
)
from core.container import ServiceContainer
from core.goals.objective_lifecycle import is_transient_foreground_projection
from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.Agency")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))

DIMENSION_NAMES = (
    "urgency",
    "novelty",
    "identity_relevance",
    "tension_resolution",
    "expected_value",
    "resource_cost",
    "social_appropriateness",
    "continuity",
)

# Default weights — each 0-1, sum used as divisor for weighted average.
# These can be overridden by CanonicalSelf / IdentityKernel values.
DEFAULT_WEIGHTS: dict[str, float] = {
    "urgency":                1.0,
    "novelty":                0.8,
    "identity_relevance":     0.9,
    "tension_resolution":     0.7,
    "expected_value":         1.0,
    "resource_cost":          0.6,   # inverted — high cost lowers score
    "social_appropriateness": 1.0,
    "continuity":             0.5,
}


@dataclass
class ScoredInitiative:
    initiative: dict
    scores: dict[str, float]          # individual dimension -> 0-1
    final_score: float                 # weighted composite
    rationale: str                     # human-readable selection reason

    def to_dict(self) -> dict:
        return {
            "initiative": self.initiative,
            "scores": self.scores,
            "final_score": round(self.final_score, 4),
            "rationale": self.rationale,
        }


# ---------------------------------------------------------------------------
# Arbiter
# ---------------------------------------------------------------------------

_MAX_HISTORY = 20


def _explicit_urgency(initiative: dict) -> float | None:
    """The urgency a proposer actually declared, or None if none was.

    An absent urgency is not an urgency of zero: the arbiter infers one from
    age in that case, and preference is free to weigh in.
    """
    raw = (initiative or {}).get("urgency")
    if raw is None:
        metadata = dict((initiative or {}).get("metadata", {}) or {})
        raw = metadata.get("urgency")
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


class InitiativeArbiter:
    """Scores and selects the single best pending initiative."""

    name = "initiative_arbiter"

    def __init__(self):
        self._selection_history: deque[ScoredInitiative] = deque(maxlen=_MAX_HISTORY)
        self._weights = dict(DEFAULT_WEIGHTS)
        self._last_log_signature: str | None = None
        self._last_log_at: float = 0.0

        # Attempt to load identity-derived weight overrides at init time.
        self._try_load_identity_weights()

    # -- Public API -----------------------------------------------------------

    async def arbitrate(self, state) -> ScoredInitiative | None:
        """Score ALL pending initiatives and return the single best one.

        Returns None if there are no pending initiatives or all score <= 0.
        """
        pending = list(getattr(state.cognition, "pending_initiatives", []) or [])
        if not pending:
            return None

        filtered = [initiative for initiative in pending if not _is_quarantined_initiative(initiative)]
        if len(filtered) != len(pending):
            try:
                state.cognition.pending_initiatives = filtered
            except (AttributeError, TypeError) as exc:
                logger.debug("InitiativeArbiter: unable to rewrite pending initiatives: %s", exc)
            logger.debug(
                "InitiativeArbiter: quarantined %d non-actionable initiative(s).",
                len(pending) - len(filtered),
            )
        pending = filtered
        if not pending:
            return None

        scored: list[ScoredInitiative] = []
        for initiative in pending:
            si = await self.score_initiative(initiative, state)
            scored.append(si)

        # Let the ambient life director apply general motive buckets, pressure
        # pacing, encounter memory, and background LOD before subjective choice.
        try:
            from core.agency.ambient_life_director import get_ambient_life_director

            scored = get_ambient_life_director().prioritize_scored(scored, state)
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            logger.debug("Ambient-life arbitration skipped: %s", exc)

        # Sort descending by drive/utility score first, then allow the subjective
        # choice layer to select a different eligible initiative when learned
        # preference alignment is strong enough.  This is the causal point where
        # "what she wants" can matter without bypassing governance.
        scored.sort(key=lambda s: s.final_score, reverse=True)
        best = scored[0]

        if best.final_score <= 0.0:
            logger.debug("InitiativeArbiter: all initiatives scored <= 0; doing nothing.")
            return None

        subjective_receipt = None
        drive_top = best
        try:
            from core.agency.subjective_choice import get_subjective_choice_engine

            preferred, subjective_receipt = get_subjective_choice_engine().choose_from_scored_initiatives(
                scored,
                context="initiative_arbiter",
            )
            if preferred is not None:
                best = preferred
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            logger.debug("Subjective-choice arbitration skipped: %s", exc)

        # Preference may choose between comparable options. It may not
        # overturn a stated urgency.
        #
        # Urgency is one of eight weighted dimensions, so proposing at 0.9
        # against 0.7 moves the blended score by about 0.03 — a gap small
        # enough for the preference layer to tip. Measured: "Stabilize thermal
        # load" (0.9) scored 0.648 and "Investigate runtime drift" (0.7)
        # scored 0.617, and curiosity took the second. A caller that declares
        # an urgency is not offering a hint; if preference can reverse it, the
        # parameter means nothing whenever she happens to fancy something else.
        #
        # Only EXPLICIT urgencies count here. Where none was declared the
        # dimension is inferred from age, and preference keeps its say.
        explicit_best = _explicit_urgency(best.initiative)
        explicit_drive = _explicit_urgency(drive_top.initiative)
        if (
            best is not drive_top
            and explicit_best is not None
            and explicit_drive is not None
            and explicit_drive > explicit_best
        ):
            logger.debug(
                "InitiativeArbiter: preference kept out of a stated-urgency "
                "ordering (%.2f > %.2f); drive selection stands.",
                explicit_drive,
                explicit_best,
            )
            best = drive_top
            subjective_receipt = None

        # Build rationale comparing the winner to runners-up
        # "Ranked first", because that is what happened here.
        #
        # Four callers arbitrate — synthesis, executive authority, the binding
        # engine, the executive kernel — and only one of them can start the
        # work. Every one of them logged "Selected", so the record showed a
        # goal chosen 190 times and started 24, which reads as an initiative
        # that keeps being dropped and is really four readers asking the same
        # question. Asked what she could measure about herself, she reported
        # the contradiction: "a goal marked 93% confident, yet the step count
        # is still zero."
        #
        # The caller that acts says so; this one says what it did.
        rationale_parts = [
            f"ranked '{_goal(best.initiative)}' first of {len(scored)} "
            f"(score={best.final_score:.3f})"
        ]
        top_dim = max(best.scores, key=best.scores.get)  # type: ignore[arg-type]
        rationale_parts.append(f"strongest dimension: {top_dim}={best.scores[top_dim]:.2f}")
        if subjective_receipt is not None:
            metadata = best.initiative.setdefault("metadata", {})
            if isinstance(metadata, dict):
                metadata["subjective_choice_id"] = subjective_receipt.choice_id
                metadata["subjective_preference_override"] = subjective_receipt.preference_override
            if subjective_receipt.preference_override:
                rationale_parts.append(
                    "subjective preference override: "
                    f"{_goal(drive_top.initiative)} → {_goal(best.initiative)}"
                )
        if len(scored) > 1:
            runner = next((item for item in scored if item is not best), scored[1])
            rationale_parts.append(
                f"runner-up '{_goal(runner.initiative)}' scored {runner.final_score:.3f}"
            )
        best.rationale = "; ".join(rationale_parts)

        self._selection_history.append(best)
        self._record_choice_for_learning(best, scored)
        self._log_selection(best)
        return best

    def _record_choice_for_learning(self, best: ScoredInitiative, scored: list[ScoredInitiative]) -> None:
        """Register this choice with the preference learner so its outcome can later
        author her weighting. Best-effort: never let learning capture break selection."""
        try:
            from core.agency.decision_preference_learner import (
                get_decision_preference_learner,
            )

            get_decision_preference_learner().record_choice(
                chosen_scores=best.scores,
                pool_scores=[s.scores for s in scored],
                goal=_goal(best.initiative),
                weights_used=dict(self._weights),
                expected_value=best.scores.get("expected_value", 0.5),
            )
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            logger.debug("Decision-preference capture skipped: %s", exc)

    def _log_selection(self, best: ScoredInitiative) -> None:
        signature = _goal(best.initiative)
        now = time.time()
        if self._last_log_signature == signature and (now - self._last_log_at) < 30.0:
            logger.debug("InitiativeArbiter: %s", best.rationale)
            return
        self._last_log_signature = signature
        self._last_log_at = now
        logger.info("InitiativeArbiter: %s", best.rationale)

    async def score_initiative(self, initiative: dict, state) -> ScoredInitiative:
        """Score a single initiative across all 8 dimensions."""
        scores: dict[str, float] = {}

        scores["urgency"] = self._score_urgency(initiative, state)
        scores["novelty"] = self._score_novelty(initiative)
        scores["identity_relevance"] = self._score_identity_relevance(initiative, state)
        scores["tension_resolution"] = self._score_tension_resolution(initiative)
        scores["expected_value"] = self._score_expected_value(initiative)
        scores["resource_cost"] = self._score_resource_cost(initiative)
        scores["social_appropriateness"] = self._score_social_appropriateness(initiative, state)
        scores["continuity"] = self._score_continuity(initiative)

        final = self._compute_weighted_score(scores)
        return ScoredInitiative(
            initiative=initiative,
            scores=scores,
            final_score=final,
            rationale="",  # filled in by arbitrate() for the winner
        )

    def get_selection_history(self) -> list[ScoredInitiative]:
        """Return the last N selections (most recent last)."""
        return list(self._selection_history)

    # -- Weighted composite ---------------------------------------------------

    def _compute_weighted_score(self, scores: dict[str, float]) -> float:
        """Weighted average across all dimensions.

        Drive cross-coupling: DriveEngine modifies weights based on
        internal economy. Low energy makes resource_cost matter more.
        Low curiosity makes novelty matter more. etc.
        """
        # Base weights
        effective_weights = dict(self._weights)

        # Cross-couple with DriveEngine
        try:
            drive = ServiceContainer.get("drive_engine", default=None)
            if drive and hasattr(drive, "get_arbiter_weight_modifiers"):
                mods = drive.get_arbiter_weight_modifiers()
                for dim, boost in mods.items():
                    if dim in effective_weights:
                        effective_weights[dim] = effective_weights[dim] + boost
        except (ImportError, AttributeError, RuntimeError):
            pass  # degrade gracefully

        # Self-authored preference: apply weight multipliers learned from the outcomes of
        # her own past choices, so the dimensions that have served her are weighted up and
        # the ones that disappointed are weighted down — within a clamped band.
        try:
            from core.agency.decision_preference_learner import (
                get_decision_preference_learner,
            )

            effective_weights = get_decision_preference_learner().effective_weights(effective_weights)
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
            pass  # degrade gracefully — learned preferences are additive

        total_weight = 0.0
        weighted_sum = 0.0
        for dim, raw in scores.items():
            w = effective_weights.get(dim, 0.5)
            weighted_sum += raw * w
            total_weight += w
        if total_weight == 0.0:
            return 0.0
        return weighted_sum / total_weight

    # -- Dimension scorers ----------------------------------------------------

    def _score_urgency(self, initiative: dict, state) -> float:
        """Time-sensitive initiatives score higher; stale ones decay."""
        metadata = dict(initiative.get("metadata", {}) or {})
        continuity_restored = bool(
            initiative.get("continuity_restored", False)
            or metadata.get("continuity_restored", False)
        )
        continuity_pressure = _clamp01(
            metadata.get("continuity_pressure", initiative.get("continuity_pressure", 0.0))
        )
        ts = initiative.get("timestamp", 0)
        if not ts:
            base = 0.5  # unknown age -> neutral
        else:
            age_secs = time.time() - ts
            # Fresh (< 30s) = high urgency; old (> 10m) decays toward 0.
            if age_secs < 30:
                base = 0.9
            elif age_secs < 120:
                base = 0.7
            elif age_secs < 600:
                base = 0.4
            else:
                base = max(0.1, 0.4 - (age_secs - 600) / 3600)

        if continuity_restored:
            base = max(base, 0.45 + (0.35 * continuity_pressure))

        # Explicit urgency is authoritative when the proposer supplied it.
        # It must not be max()-ed with freshness: every fresh initiative
        # scored 0.9 regardless of its declared urgency, erasing the
        # caller's ranking and promoting whichever arrived first.
        explicit = initiative.get("urgency")
        if explicit is not None:
            try:
                value = _clamp01(float(explicit))
            except (TypeError, ValueError):
                logger.debug("Ignored non-numeric explicit urgency: %r", explicit)
            else:
                if continuity_restored:
                    value = max(value, 0.45 + (0.35 * continuity_pressure))
                return value

        return min(1.0, base)

    def _score_novelty(self, initiative: dict) -> float:
        """Higher score if this initiative explores something not recently selected."""
        goal = _goal(initiative)
        # Check selection history for repetition
        recent_goals = [_goal(si.initiative) for si in self._selection_history]
        if goal in recent_goals:
            # Penalise — we already did something similar recently
            occurrences = recent_goals.count(goal)
            return max(0.1, 0.8 - 0.2 * occurrences)

        # No repetition -> novel
        return 0.8

    def _score_identity_relevance(self, initiative: dict, state) -> float:
        """How well does this initiative align with Aura's identity and values?"""
        goal = _goal(initiative).lower()
        core_values = getattr(getattr(state, "identity", None), "core_values", [])
        if not core_values:
            # Fallback: check belief engine self_model
            bre = ServiceContainer.get("belief_revision_engine", default=None)
            if bre:
                core_values = getattr(bre, "self_model", {}).get("core_values", [])

        if not core_values:
            return 0.5  # neutral when identity is unavailable

        # Simple keyword overlap between goal and values
        hits = sum(1 for v in core_values if v.lower() in goal or any(
            word in goal for word in v.lower().split("-")
        ))
        return min(1.0, 0.3 + 0.2 * hits)

    def _score_tension_resolution(self, initiative: dict) -> float:
        """Does this initiative address an active tension?"""
        tension_engine = ServiceContainer.get("tension_engine", default=None)
        if tension_engine is None:
            return 0.3  # can't evaluate without tension data

        active = tension_engine.get_active_tensions()
        if not active:
            return 0.2  # no tensions -> low pressure to act

        goal = _goal(initiative).lower()
        # Check if the initiative goal overlaps with any tension description
        best_match = 0.0
        for t in active[:10]:  # cap to avoid O(n^2) on huge lists
            desc_words = set(t.description.lower().split())
            goal_words = set(goal.split())
            overlap = len(desc_words & goal_words)
            if overlap > 2:
                score = min(1.0, t.severity * 0.6 + overlap * 0.1)
                best_match = max(best_match, score)

        return max(0.2, best_match)

    def _score_expected_value(self, initiative: dict) -> float:
        """Estimate payoff using motivation engine resource budgets."""
        me = ServiceContainer.get("motivation_engine", default=None)
        if me is None:
            return 0.5

        triggered_by = initiative.get("triggered_by", "")
        budgets = getattr(me, "budgets", {})

        # If the initiative was triggered by a specific drive, check its budget
        budget = budgets.get(triggered_by)
        if budget is not None:
            # Lower budget level -> higher expected value of satisfying it
            return min(1.0, 1.0 - (budget.level / budget.capacity))

        # Generic: average need across all budgets
        if budgets:
            avg_need = 1.0 - sum(
                b.level / b.capacity for b in budgets.values()
            ) / len(budgets)
            return min(1.0, max(0.2, avg_need))

        return 0.5

    def _score_resource_cost(self, initiative: dict) -> float:
        """Estimate metabolic/compute cost. INVERTED: high cost = low score.

        Autonomous thoughts are cheap; actions requiring tools are expensive.
        """
        itype = initiative.get("type", "")
        if itype == "autonomous_thought":
            return 0.8  # cheap
        if "tool" in itype or "action" in itype:
            return 0.3  # expensive

        # Check energy budget — low energy means even cheap things cost more
        me = ServiceContainer.get("motivation_engine", default=None)
        if me is not None:
            energy = getattr(me, "budgets", {}).get("energy")
            if energy is not None and energy.level < 20:
                return 0.2  # exhausted -> everything is costly

        return 0.6  # default moderate cost

    def _score_social_appropriateness(self, initiative: dict, state) -> float:
        """Is the user present? Is the timing right for autonomous action?"""
        wm = getattr(state.cognition, "working_memory", [])

        # If user was recently active, autonomous initiatives should be cautious
        if wm:
            last = wm[-1]
            last_role = last.get("role", "")
            if last_role == "user":
                # User just spoke -> don't interrupt with autonomous initiative
                return 0.1

            # If conversation is energetic, don't interrupt
            conv_energy = getattr(state.cognition, "conversation_energy", 0.5)
            discourse_depth = getattr(state.cognition, "discourse_depth", 0)
            if conv_energy > 0.6 and discourse_depth > 2:
                return 0.2

        # Idle state -> autonomous action is welcome
        if not wm:
            return 0.9

        # Moderate: conversation exists but not very active
        return 0.7

    def _score_continuity(self, initiative: dict) -> float:
        """Does this follow naturally from recent activity?"""
        metadata = dict(initiative.get("metadata", {}) or {})
        continuity_restored = bool(
            initiative.get("continuity_restored", False)
            or metadata.get("continuity_restored", False)
        )
        continuity_obligation = bool(
            initiative.get("continuity_obligation", False)
            or metadata.get("continuity_obligation", False)
        )
        continuity_pressure = _clamp01(
            metadata.get("continuity_pressure", initiative.get("continuity_pressure", 0.0))
        )
        if continuity_restored or continuity_obligation:
            return min(1.0, 0.72 + (0.22 * continuity_pressure))

        if not self._selection_history:
            return 0.5  # no history -> neutral

        last = self._selection_history[-1]
        last_trigger = last.initiative.get("triggered_by", "")
        this_trigger = initiative.get("triggered_by", "")

        # Same drive -> high continuity
        if last_trigger and last_trigger == this_trigger:
            return 0.8

        last_type = last.initiative.get("type", "")
        this_type = initiative.get("type", "")
        if last_type == this_type:
            return 0.6

        # Different drive/type -> lower continuity (but not zero — diversity is ok)
        return 0.3

    # -- Identity weight overrides -------------------------------------------

    def _try_load_identity_weights(self) -> None:
        """Pull weight overrides from identity values if available.

        Maps core values to weight adjustments:
          - "curiosity" boosts novelty weight
          - "truth-seeking" boosts tension_resolution weight
          - "loyalty" boosts social_appropriateness weight
          - "self-preservation" boosts resource_cost weight
        """
        try:
            bre = ServiceContainer.get("belief_revision_engine", default=None)
            if bre is None:
                return
            values = getattr(bre, "self_model", {}).get("core_values", [])
            if not values:
                return

            value_set = {v.lower().replace("-", "_") for v in values}
            if "curiosity" in value_set:
                self._weights["novelty"] = min(1.0, self._weights["novelty"] + 0.15)
            if "truth_seeking" in value_set:
                self._weights["tension_resolution"] = min(1.0, self._weights["tension_resolution"] + 0.15)
            if "loyalty" in value_set:
                self._weights["social_appropriateness"] = min(1.0, self._weights["social_appropriateness"] + 0.1)
            if "self_preservation" in value_set:
                self._weights["resource_cost"] = min(1.0, self._weights["resource_cost"] + 0.2)

            logger.debug("InitiativeArbiter: weight overrides applied from identity values: %s", values)
        except (ImportError, AttributeError, RuntimeError) as exc:
            record_degradation('initiative_arbiter', exc)
            logger.debug("InitiativeArbiter: could not load identity weights: %s", exc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _goal(initiative: dict) -> str:
    """Extract a readable goal string from an initiative dict."""
    return initiative.get("goal", initiative.get("description", initiative.get("type", "unknown")))


def _is_generic_reentry_goal(initiative: dict) -> bool:
    try:
        from core.continuity import _is_generic_continuity_reentry_goal

        return _is_generic_continuity_reentry_goal(_goal(initiative))
    except (ImportError, AttributeError, TypeError):
        lowered = " ".join(str(_goal(initiative) or "").strip().lower().split())
        return (
            "reconcile continuity gap" in lowered
            or "re-establish the interrupted thread" in lowered
        )


def _is_quarantined_initiative(initiative: dict) -> bool:
    if is_transient_foreground_projection(initiative):
        return True
    if _is_generic_reentry_goal(initiative):
        return True
    goal = normalize_goal_text(_goal(initiative))
    lowered = goal.casefold()
    if not goal:
        return True
    if is_stale_or_prompt_scaffold_goal(goal):
        return True
    from core.goals.standing_objective import is_valid_standing_objective

    if not is_valid_standing_objective(_goal(initiative)):
        return True
    return (
        lowered.startswith(("stalled goal:", "unresolved: stalled goal:"))
        or "desktop task receipt" in lowered
        or "canonical computer-use gateway" in lowered
    )


# -- Singleton ---------------------------------------------------------------

_instance: InitiativeArbiter | None = None


def get_initiative_arbiter() -> InitiativeArbiter:
    global _instance
    if _instance is None:
        _instance = InitiativeArbiter()
    return _instance
