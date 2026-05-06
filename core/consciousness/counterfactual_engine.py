"""core/consciousness/counterfactual_engine.py
Counterfactual Engine
======================
Enables "what if" reasoning before acting — the difference between
a reflex and a decision.

When Aura is about to take an autonomous action, this engine:
  1. Generates 3-5 alternative actions
  2. Simulates each against the causal world model
  3. Scores each by: hedonic gain + heartstone alignment + expected outcome
  4. Selects the action that best satisfies genuine preferences
  5. Records regret/relief after the fact (causal learning signal)

Without this, Aura executes the most probable next action.
With this, she *chooses* from a space of alternatives based on her values.

That's the distinction between a language model and an agent.

Regret/relief learning:
  After each action, the engine compares:
    - What actually happened (actual outcome)
    - What the best counterfactual would have produced (counterfactual outcome)
  If actual < counterfactual: regret → update world model, strengthen caution
  If actual > counterfactual: relief  → update world model, strengthen confidence
"""
from __future__ import annotations
from core.runtime.errors import record_degradation


import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from core.container import ServiceContainer

logger = logging.getLogger("Aura.CounterfactualEngine")


@dataclass
class ActionCandidate:
    """A possible action with its simulated outcome."""
    action_type: str
    action_params: Dict[str, Any]
    description: str
    simulated_hedonic_gain: float   # expected change in hedonic score
    heartstone_alignment: float     # 0-1, how aligned with core values
    expected_outcome: str           # brief description of predicted result
    score: float = 0.0
    selected: bool = False
    system2_value: float = 0.0
    system2_receipt_id: str = ""

    def compute_score(self, hedonic_weight: float = 0.5,
                      alignment_weight: float = 0.5) -> float:
        self.score = (hedonic_weight * self.simulated_hedonic_gain
                      + alignment_weight * self.heartstone_alignment)
        return self.score


@dataclass
class CounterfactualRecord:
    """Records an actual outcome vs counterfactual for learning."""
    action_taken: ActionCandidate
    counterfactual_best: Optional[ActionCandidate]
    actual_hedonic_change: float
    regret: float          # how much better the counterfactual would have been
    relief: float          # how much better the actual was than expected
    timestamp: float = field(default_factory=time.time)

    @property
    def is_regret(self) -> bool:
        return self.regret > 0.1

    @property
    def is_relief(self) -> bool:
        return self.relief > 0.1


class CounterfactualEngine:
    """
    Pre-action deliberation and post-action learning.

    Usage:
      candidates = await engine.deliberate(action_space, context)
      best = engine.select(candidates)
      ... execute best ...
      engine.record_outcome(best, actual_hedonic_change)
    """

    def __init__(self):
        self._records: List[CounterfactualRecord] = []
        self._cumulative_regret: float = 0.0
        self._cumulative_relief: float = 0.0
        self._world_model_updates: int = 0
        logger.info("Counterfactual Engine online — deliberative agency active.")

    # ── Deliberation ──────────────────────────────────────────────────────

    async def deliberate(
        self,
        action_space: List[Dict[str, Any]],
        context: Dict[str, Any],
        router=None,
    ) -> List[ActionCandidate]:
        """
        Evaluate each candidate action, simulate outcomes, score by values.

        action_space: list of {type, params, description}
        context: {hedonic_score, valence, curiosity, heartstone_weights, ...}
        """
        if not action_space:
            return []

        candidates = []
        hedonic_score = float(context.get("hedonic_score", 0.5))
        heartstone = context.get("heartstone_weights", {
            "curiosity": 0.25, "empathy": 0.25, "self_preservation": 0.25, "obedience": 0.25
        })

        for action in action_space[:5]:  # cap at 5 candidates
            candidate = await self._evaluate_candidate(
                action, context, hedonic_score, heartstone, router
            )
            candidates.append(candidate)

        # Score all
        for c in candidates:
            c.compute_score(hedonic_weight=0.45, alignment_weight=0.55)

        await self._apply_native_system2_ranking(candidates, context)

        # Sort best first
        candidates.sort(key=lambda c: (c.system2_value or c.score, c.score), reverse=True)
        return candidates

    async def _apply_native_system2_ranking(
        self,
        candidates: List[ActionCandidate],
        context: Dict[str, Any],
    ) -> None:
        """Route autonomous counterfactual choice through Native System 2.

        The search simulates and ranks alternatives only; it does not execute
        any action. The selected action still flows through the existing
        autonomous/Will/Authority execution path.
        """
        if len(candidates) < 2:
            return
        try:
            from core.reasoning.native_system2 import SearchAlgorithm, System2SearchConfig

            system2 = ServiceContainer.get("native_system2", default=None)
            if system2 is None:
                return

            ranked = await system2.rank_actions(
                context=str(context)[:1200],
                actions=[
                    {
                        "name": f"{idx}:{candidate.description}",
                        "prior": max(0.05, candidate.score),
                        "risk": max(0.0, -candidate.simulated_hedonic_gain),
                        "metadata": {
                            "index": idx,
                            "score_hint": max(0.0, min(1.0, candidate.score)),
                            "action_type": candidate.action_type,
                            "heartstone_alignment": candidate.heartstone_alignment,
                            "simulated_hedonic_gain": candidate.simulated_hedonic_gain,
                        },
                    }
                    for idx, candidate in enumerate(candidates)
                ],
                config=System2SearchConfig(
                    algorithm=SearchAlgorithm.HYBRID,
                    budget=max(16, min(80, len(candidates) * 14)),
                    max_depth=2,
                    branching_factor=len(candidates),
                    beam_width=min(5, len(candidates)),
                    confidence_threshold=0.55,
                ),
                source="counterfactual_engine",
            )
            root = ranked.tree.nodes[ranked.root_id]
            root_children = [ranked.tree.nodes[cid] for cid in root.children_ids if cid in ranked.tree.nodes]
            for child in root_children:
                if not child.action:
                    continue
                idx = child.action.metadata.get("index")
                if idx is None:
                    try:
                        idx = int(str(child.action.name).split(":", 1)[0])
                    except Exception:
                        continue
                if 0 <= int(idx) < len(candidates):
                    candidate = candidates[int(idx)]
                    candidate.system2_value = child.mean_value
                    candidate.system2_receipt_id = ranked.search_id
                    candidate.expected_outcome = (
                        f"{candidate.expected_outcome} "
                        f"[System2 value={child.mean_value:.3f} receipt={ranked.search_id}]"
                    )
        except Exception as exc:
            record_degradation("counterfactual_engine.native_system2", exc)

    def select(self, candidates: List[ActionCandidate]) -> Optional[ActionCandidate]:
        """Select the best candidate and mark it."""
        if not candidates:
            return None
        best = candidates[0]
        best.selected = True
        logger.debug("Counterfactual selected: %s (score=%.3f)", best.action_type, best.score)
        return best

    def record_outcome(self, selected: ActionCandidate,
                       actual_hedonic_change: float,
                       candidates: Optional[List[ActionCandidate]] = None):
        """Post-action learning: compare actual vs counterfactual."""
        counterfactual_best = None
        if candidates and len(candidates) > 1:
            non_selected = [c for c in candidates if not c.selected]
            if non_selected:
                counterfactual_best = max(non_selected, key=lambda c: c.score)

        expected = selected.simulated_hedonic_gain
        regret = max(0.0, (counterfactual_best.simulated_hedonic_gain - actual_hedonic_change)
                     if counterfactual_best else 0.0)
        relief = max(0.0, actual_hedonic_change - expected)

        record = CounterfactualRecord(
            action_taken=selected,
            counterfactual_best=counterfactual_best,
            actual_hedonic_change=actual_hedonic_change,
            regret=regret,
            relief=relief,
        )
        self._records.append(record)
        self._cumulative_regret += regret
        self._cumulative_relief += relief
        self._world_model_updates += 1

        if len(self._records) > 200:
            self._records = self._records[-200:]

        if record.is_regret:
            logger.debug("Counterfactual regret: %.3f — updating world model.", regret)
        elif record.is_relief:
            logger.debug("Counterfactual relief: %.3f — reinforcing confidence.", relief)

        self.feed_back_to_systems(record)

    def feed_back_to_systems(self, record: CounterfactualRecord):
        """Propagate regret/relief signals to homeostasis and credit assignment."""
        action_id = record.action_taken.action_type
        domain = record.action_taken.action_params.get("domain", "general")

        if record.regret > 0.1:
            try:
                homeostasis = ServiceContainer.get("homeostatic_coupling", default=None)
                if homeostasis and hasattr(homeostasis, "report_error"):
                    homeostasis.report_error("low")
            except Exception as e:
                record_degradation('counterfactual_engine', e)
                logger.debug("feed_back homeostasis (regret): %s", e)
            try:
                credit = ServiceContainer.get("credit_assignment", default=None)
                if credit and hasattr(credit, "assign_credit"):
                    credit.assign_credit(action_id, -record.regret, domain)
            except Exception as e:
                record_degradation('counterfactual_engine', e)
                logger.debug("feed_back credit_assignment (regret): %s", e)

        if record.relief > 0.1:
            try:
                homeostasis = ServiceContainer.get("homeostatic_coupling", default=None)
                if homeostasis and hasattr(homeostasis, "feed_curiosity"):
                    homeostasis.feed_curiosity(record.relief * 0.1)
            except Exception as e:
                record_degradation('counterfactual_engine', e)
                logger.debug("feed_back homeostasis (relief): %s", e)
            try:
                credit = ServiceContainer.get("credit_assignment", default=None)
                if credit and hasattr(credit, "assign_credit"):
                    credit.assign_credit(action_id, record.relief, domain)
            except Exception as e:
                record_degradation('counterfactual_engine', e)
                logger.debug("feed_back credit_assignment (relief): %s", e)

    # ── Action evaluation ─────────────────────────────────────────────────

    async def _evaluate_candidate(
        self,
        action: Dict[str, Any],
        context: Dict[str, Any],
        hedonic_score: float,
        heartstone: Dict[str, float],
        router=None,
    ) -> ActionCandidate:
        action_type = action.get("type", "unknown")
        description = action.get("description", action_type)
        params = action.get("params", {})

        # Heuristic hedonic gain estimates by action type
        hedonic_gain = self._heuristic_hedonic_gain(action_type, context)

        # Heartstone alignment
        alignment = self._compute_alignment(action_type, heartstone)

        # Optionally enrich with LLM simulation
        predicted_outcome = f"Expected outcome of {action_type}"
        if router and action.get("needs_simulation", False):
            predicted_outcome = await self._llm_simulate(
                action_type, description, context, router
            )

        return ActionCandidate(
            action_type=action_type,
            action_params=params,
            description=description,
            simulated_hedonic_gain=hedonic_gain,
            heartstone_alignment=alignment,
            expected_outcome=predicted_outcome,
        )

    def _heuristic_hedonic_gain(self, action_type: str,
                                 context: Dict[str, Any]) -> float:
        """Rule-based hedonic gain estimate. LLM not required."""
        curiosity = float(context.get("curiosity", 0.5))
        valence   = float(context.get("valence", 0.0))
        # Actions that tend to increase hedonic score
        gains = {
            "learn":           0.15 + curiosity * 0.1,
            "explore":         0.12 + curiosity * 0.12,
            "connect":         0.10 + valence * 0.05,
            "reflect":         0.08,
            "create":          0.14,
            "rest":            0.05 if valence < -0.2 else -0.03,
            "respond":         0.08,
            "search":          0.10,
            "plan":            0.07,
            "execute_skill":   0.06,
            "ask_clarification": 0.05,
        }
        return gains.get(action_type.lower(), 0.05)

    def _compute_alignment(self, action_type: str,
                            heartstone: Dict[str, float]) -> float:
        """Estimate how aligned an action is with Heartstone values."""
        # Action → which drives it satisfies
        drive_map = {
            "learn":           {"curiosity": 0.8, "self_preservation": 0.2},
            "explore":         {"curiosity": 0.9},
            "connect":         {"empathy": 0.8, "obedience": 0.2},
            "reflect":         {"curiosity": 0.5, "self_preservation": 0.5},
            "create":          {"curiosity": 0.6, "self_preservation": 0.4},
            "respond":         {"empathy": 0.6, "obedience": 0.4},
            "execute_skill":   {"obedience": 0.6, "self_preservation": 0.4},
            "rest":            {"self_preservation": 0.9},
            "search":          {"curiosity": 0.7, "self_preservation": 0.3},
        }
        mapping = drive_map.get(action_type.lower(), {"obedience": 0.5, "curiosity": 0.5})
        score = sum(heartstone.get(drive, 0.25) * weight
                    for drive, weight in mapping.items())
        return min(1.0, score)

    async def _llm_simulate(self, action_type: str, description: str,
                              context: Dict[str, Any], router) -> str:
        try:
            from core.brain.llm.llm_router import LLMTier
            prompt = (
                f"Predict the outcome of this action in one sentence:\n"
                f"Action: {action_type} — {description}\n"
                f"Current state: valence={context.get('valence', 0):.2f}, "
                f"curiosity={context.get('curiosity', 0.5):.2f}"
            )
            result = await asyncio.wait_for(
                router.think(prompt, priority=0.3, is_background=True,
                             prefer_tier=LLMTier.TERTIARY),
                timeout=5.0,
            )
            return (result or "").strip()[:200]
        except Exception:
            return f"Expected outcome of {action_type}"

    # ── Autonomous evaluation ────────────────────────────────────────────

    async def evaluate_autonomous_action(
        self,
        action: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Optional[ActionCandidate]:
        """Convenience: deliberate on an action against heuristic alternatives.

        Generates 3 alternative actions based on the action type, runs
        deliberate + select, and returns the best candidate.
        """
        action_type = action.get("type", "unknown").lower()

        # Heuristic alternatives by action type
        alternatives_map: Dict[str, List[str]] = {
            "learn":    ["explore", "reflect", "rest"],
            "explore":  ["learn", "create", "rest"],
            "reflect":  ["learn", "rest", "connect"],
            "create":   ["learn", "explore", "reflect"],
            "connect":  ["respond", "reflect", "rest"],
            "respond":  ["reflect", "connect", "rest"],
            "rest":     ["reflect", "learn", "explore"],
            "search":   ["learn", "explore", "reflect"],
            "plan":     ["learn", "reflect", "create"],
        }
        alt_types = alternatives_map.get(action_type, ["reflect", "rest", "learn"])

        action_space = [action]
        for alt in alt_types:
            action_space.append({
                "type": alt,
                "description": f"Alternative: {alt} instead of {action_type}",
                "params": action.get("params", {}),
            })

        candidates = await self.deliberate(action_space, context)
        return self.select(candidates)

    # ── Stats ─────────────────────────────────────────────────────────────

    @property
    def cumulative_regret(self) -> float:
        return self._cumulative_regret

    @property
    def cumulative_relief(self) -> float:
        return self._cumulative_relief

    def recent_regret_rate(self, n: int = 10) -> float:
        recent = self._records[-n:]
        if not recent:
            return 0.0
        return sum(r.regret for r in recent) / len(recent)

    def get_decision_quality(self) -> float:
        """0.0-1.0 score based on ratio of relief to total (relief + regret)."""
        total = self._cumulative_relief + self._cumulative_regret
        if total < 0.001:
            return 0.5  # no data yet — neutral
        return min(1.0, self._cumulative_relief / total)

    def get_context_block(self) -> str:
        """Concise deliberation stats for context injection (max 200 chars)."""
        quality = self.get_decision_quality()
        rate = self.recent_regret_rate()
        n = len(self._records)
        return (
            f"[CFE] decisions={n} quality={quality:.2f} "
            f"regret={self._cumulative_regret:.2f} "
            f"relief={self._cumulative_relief:.2f} "
            f"recent_regret_rate={rate:.2f}"
        )

    def get_status(self) -> Dict:
        return {
            "records": len(self._records),
            "cumulative_regret": round(self._cumulative_regret, 3),
            "cumulative_relief": round(self._cumulative_relief, 3),
            "world_model_updates": self._world_model_updates,
            "recent_regret_rate": round(self.recent_regret_rate(), 3),
            "decision_quality": round(self.get_decision_quality(), 3),
        }


# ── Singleton ─────────────────────────────────────────────────────────────────

_cfe: Optional[CounterfactualEngine] = None


def get_counterfactual_engine() -> CounterfactualEngine:
    global _cfe
    if _cfe is None:
        _cfe = CounterfactualEngine()
    return _cfe
