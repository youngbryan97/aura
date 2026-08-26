"""Unified world model — one query surface over Aura's complementary world-model facets.

The critique's item #2 reads "collapse the many world-model classes into one." Tracing them,
the four are *not* four copies of the same thing — they are four complementary facets that a
destructive merge would impoverish:

    LearnedWorldModel       (core/world_model/learned_world_model)   forward dynamics: a VRNN
                            that predicts the next observation and reports surprise; supports
                            imagined rollouts.
    CausalWorldModel        (core/brain/causal_world_model)          a causal graph: do-style
                            interventions, counterfactual simulation, preventative analysis.
    MultiDomainWorldModel   (core/advanced_cognition/world_model)    experience-grounded
                            outcome prediction (reward/harm/surprise) from past episodes.
    LearnedMCTSPlanner      (core/cognition/mcts_world_model)        multi-step lookahead that
                            plans *over* the learned dynamics model.
    HowItMoves              (core/perception/how_it_moves)           rules worked out by
                            WATCHING HERSELF ACT: hypotheses about how a laid-out thing answers
                            to being pushed, scored against her own (before, action, after)
                            triples. Where the learned facet works on observation vectors and
                            answers "how surprised am I", this works on typed states and answers
                            "what would this look like if I did that" — the question a loop
                            acting on a screen actually has.

So the honest consolidation is a single coherent entry point that *routes each kind of question
to the facet that answers it best* — "what happens next / how surprised am I" → dynamics, "if I
intervene, what changes / what prevents X" → causal, "what's the likely outcome of this action"
→ outcome model, "plan a sequence" → MCTS. That removes the real problem the critique names
(four overlapping import surfaces and no single mental model) while preserving every capability.

Facets are lazy and independently fault-isolated: a missing or broken facet degrades that one
query path (returning ``None``), never the whole model. ``query(intent, ...)`` is the headline
single surface; the typed methods are there for callers that already know which facet they want.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Sequence

from core.runtime.errors import record_degradation

logger = logging.getLogger("WorldModel.Unified")


class UnifiedWorldModel:
    """A composite facade: one interface, four specialist world-model facets behind it."""

    def __init__(
        self, *, learned: Any = None, causal: Any = None, outcome: Any = None, rules: Any = None
    ) -> None:
        # Pre-injected facets (used by tests) are treated as already-resolved.
        self._learned = learned
        self._causal = causal
        self._outcome = outcome
        self._rules = rules
        self._failed: Dict[str, bool] = {}

    # ── lazy, fault-isolated facet resolution ─────────────────────────────

    @property
    def learned(self) -> Any:
        """Forward-dynamics model (VRNN): next-state prediction, surprise, imagined rollouts."""
        if self._learned is None and not self._failed.get("learned"):
            try:
                from core.world_model.learned_world_model import get_learned_world_model
                self._learned = get_learned_world_model()
            except (ImportError, AttributeError, RuntimeError, OSError, ValueError, TypeError) as exc:
                record_degradation("unified_world_model", exc, severity="debug",
                                   action="learned (forward-dynamics) facet unavailable")
                self._failed["learned"] = True
        return self._learned

    @property
    def rules(self) -> Any:
        """Rules worked out by watching herself act: typed state in, typed state out.

        Kept here rather than beside the loop that uses it because a question
        about what a world would do is a world-model question, and there is
        one surface for those. A caller that watches its own acts feeds this;
        a caller that does not gets nothing from it, honestly.

        Handed in rather than reached for. The rules are worked out from a
        typed reading of a screen, which lives above this layer, and a world
        model that imported one would be a world model that knew what kind of
        world it was in. It also has to be per-run: a rule that held on one
        thing is a guess about the next one.
        """
        return self._rules

    @property
    def causal(self) -> Any:
        """Causal graph: interventions, counterfactuals, preventative analysis."""
        if self._causal is None and not self._failed.get("causal"):
            try:
                from core.brain.causal_world_model import CausalWorldModel
                self._causal = CausalWorldModel()
            except (ImportError, AttributeError, RuntimeError, OSError, ValueError, TypeError) as exc:
                record_degradation("unified_world_model", exc, severity="debug",
                                   action="causal facet unavailable")
                self._failed["causal"] = True
        return self._causal

    @property
    def outcome(self) -> Any:
        """Experience-grounded outcome model: reward/harm/surprise for an (observation, action)."""
        if self._outcome is None and not self._failed.get("outcome"):
            try:
                from core.advanced_cognition.world_model import MultiDomainWorldModel
                self._outcome = MultiDomainWorldModel()
            except (ImportError, AttributeError, RuntimeError, OSError, ValueError, TypeError) as exc:
                record_degradation("unified_world_model", exc, severity="debug",
                                   action="outcome facet unavailable")
                self._failed["outcome"] = True
        return self._outcome

    # ── forward dynamics (LearnedWorldModel) ──────────────────────────────

    def observe(self, observation: Any, action: Any = None, *, learn: bool = True) -> Optional[Dict[str, Any]]:
        """Feed an observation through the dynamics model; returns the prediction (incl. surprise)."""
        m = self.learned
        if m is None:
            return None
        try:
            pred = m.observe(observation, action, learn=learn)
            return pred.to_dict() if hasattr(pred, "to_dict") else {"prediction": pred}
        except (AttributeError, RuntimeError, OSError, ValueError, TypeError) as exc:
            record_degradation("unified_world_model", exc, severity="debug", action="observe failed")
            return None

    def surprise(self) -> Optional[float]:
        """Current running surprise (prediction error) from the dynamics model."""
        m = self.learned
        if m is None:
            return None
        try:
            return float(m.get_surprise())
        except (AttributeError, RuntimeError, OSError, ValueError, TypeError) as exc:
            record_degradation("unified_world_model", exc, severity="debug")
            return None

    def watched(self, before: Any, action: Any, after: Any) -> bool:
        """One of her own acts, and what it did — the only thing that teaches a rule."""
        m = self.rules
        if m is None:
            return False
        try:
            m.watched(before, str(action), after)
            return True
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation("unified_world_model", exc, severity="debug",
                               action="watched an act without learning from it")
            return False

    def imagine(self, observation: Any, action_sequence: Sequence[Any]) -> Optional[List[Dict[str, Any]]]:
        """Roll a sequence of actions forward, through whichever facet can answer.

        A typed state — something laid out in rows and columns, that can be
        asked where things are — goes to the rules she worked out by watching.
        An observation vector goes to the learned dynamics. Neither is a
        fallback for the other: they answer different questions about
        different kinds of observation, and asking the wrong one gets nothing.
        """
        if hasattr(observation, "cells") and hasattr(observation, "as_text"):
            return self._imagine_typed(observation, action_sequence)
        m = self.learned
        if m is None:
            return None
        try:
            traj = m.imagine(observation, list(action_sequence))
            return [p.to_dict() if hasattr(p, "to_dict") else {"prediction": p} for p in traj]
        except (AttributeError, RuntimeError, OSError, ValueError, TypeError) as exc:
            record_degradation("unified_world_model", exc, severity="debug", action="imagine failed")
            return None

    def _imagine_typed(
        self, state: Any, action_sequence: Sequence[Any]
    ) -> Optional[List[Dict[str, Any]]]:
        """Roll a sequence forward over a typed state, one act at a time."""
        m = self.rules
        if m is None:
            return None
        trajectory: List[Dict[str, Any]] = []
        here = state
        for action in action_sequence:
            here = m.expect(here, str(action))
            if here is None:
                break
            trajectory.append({"action": str(action), "state": here, "prediction": here.as_text()})
        return trajectory or None

    # ── outcome prediction (MultiDomainWorldModel) ────────────────────────

    def predict_outcome(
        self,
        domain: str,
        state: Dict[str, Any],
        *,
        action_kind: str,
        action_params: Optional[Dict[str, Any]] = None,
        reversible: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """Predict reward/harm/surprise for an action in a state, from past experience.

        Accepts plain dicts and builds the schema objects internally, so callers don't have to
        import the outcome model's schemas just to ask "what will probably happen if I do this?"
        """
        m = self.outcome
        if m is None:
            return None
        try:
            from core.advanced_cognition.schemas import ActionCandidate, Observation
            obs = Observation(domain=domain, state=dict(state), source="unified_world_model")
            act = ActionCandidate(
                action_id=f"{action_kind}:{domain}", kind=action_kind,
                params=dict(action_params or {}), reversible=reversible,
            )
            pred = m.predict(obs, act)
            return pred.to_dict() if hasattr(pred, "to_dict") else {"prediction": pred}
        except (ImportError, AttributeError, RuntimeError, OSError, ValueError, TypeError) as exc:
            record_degradation("unified_world_model", exc, severity="debug",
                               action="predict_outcome failed")
            return None

    def observe_episode(self, episode: Any) -> Optional[Dict[str, Any]]:
        """Fold a completed episode into the experience-grounded outcome model."""
        m = self.outcome
        if m is None:
            return None
        try:
            pred = m.observe_episode(episode)
            return pred.to_dict() if hasattr(pred, "to_dict") else None
        except (AttributeError, RuntimeError, OSError, ValueError, TypeError) as exc:
            record_degradation("unified_world_model", exc, severity="debug")
            return None

    # ── causal reasoning (CausalWorldModel) ───────────────────────────────

    def observe_causal(self, source: str, target: str, correlation: float) -> bool:
        """Record an observed source→target association into the causal graph."""
        m = self.causal
        if m is None:
            return False
        try:
            m.add_observation(source, target, correlation)
            return True
        except (AttributeError, RuntimeError, OSError, ValueError, TypeError) as exc:
            record_degradation("unified_world_model", exc, severity="debug")
            return False

    def causal_effects(self, source: str) -> Optional[List[Any]]:
        """What does ``source`` causally push on, and how strongly?"""
        m = self.causal
        if m is None:
            return None
        try:
            return m.predict_effects(source)
        except (AttributeError, RuntimeError, OSError, ValueError, TypeError) as exc:
            record_degradation("unified_world_model", exc, severity="debug")
            return None

    def counterfactual(self, do_interventions: Dict[str, float], steps: int = 3) -> Optional[Dict[str, float]]:
        """Simulate ``do(interventions)`` forward through the causal graph."""
        m = self.causal
        if m is None:
            return None
        try:
            return m.simulate_counterfactual(do_interventions, steps=steps)
        except (AttributeError, RuntimeError, OSError, ValueError, TypeError) as exc:
            record_degradation("unified_world_model", exc, severity="debug",
                               action="counterfactual failed")
            return None

    def preventative_actions(self, undesirable_node: str) -> Optional[List[Any]]:
        """Which upstream causes, suppressed, most reduce an undesirable outcome?"""
        m = self.causal
        if m is None:
            return None
        try:
            return m.analyze_preventative_actions(undesirable_node)
        except (AttributeError, RuntimeError, OSError, ValueError, TypeError) as exc:
            record_degradation("unified_world_model", exc, severity="debug")
            return None

    # ── planning (LearnedMCTSPlanner over the dynamics model) ─────────────

    def plan(
        self,
        current_observation: Any,
        action_space: List[Any],
        value_scorer: Callable[[Any], float],
        *,
        num_simulations: int = 100,
        max_depth: int = 20,
    ) -> Optional[Dict[str, Any]]:
        """Multi-step lookahead via MCTS over the learned dynamics model."""
        m = self.learned
        if m is None:
            return None
        try:
            from core.cognition.mcts_world_model import LearnedMCTSPlanner
            planner = LearnedMCTSPlanner(
                world_model=m, action_space=action_space, value_scorer=value_scorer,
                num_simulations=num_simulations, max_depth=max_depth,
            )
            best_action, info = planner.plan(current_observation)
            return {"best_action": best_action, "info": info}
        except (ImportError, AttributeError, RuntimeError, OSError, ValueError, TypeError) as exc:
            record_degradation("unified_world_model", exc, severity="debug", action="plan failed")
            return None

    # ── the single dispatch surface ───────────────────────────────────────

    def query(self, intent: str, **kwargs: Any) -> Dict[str, Any]:
        """One entry point: route a world-model question to the facet that answers it.

        Intents: ``observe``, ``surprise``, ``imagine``, ``predict_outcome``, ``observe_causal``,
        ``causal_effects``, ``counterfactual``, ``prevent``, ``plan``. Always returns a dict with
        the routed facet, availability, and the result (``None`` if that facet is unavailable).
        """
        routes: Dict[str, tuple] = {
            "observe": ("learned", self.observe),
            "surprise": ("learned", lambda **k: self.surprise()),
            "imagine": ("learned", self.imagine),
            "watched": ("rules", self.watched),
            "predict_outcome": ("outcome", self.predict_outcome),
            "observe_episode": ("outcome", self.observe_episode),
            "observe_causal": ("causal", self.observe_causal),
            "causal_effects": ("causal", self.causal_effects),
            "counterfactual": ("causal", self.counterfactual),
            "prevent": ("causal", self.preventative_actions),
            "plan": ("learned", self.plan),
        }
        route = routes.get(intent)
        if route is None:
            return {"intent": intent, "facet": None, "available": False,
                    "result": None, "error": "unknown_intent"}
        facet_name, fn = route
        try:
            result = fn(**kwargs)
        except TypeError as exc:
            return {"intent": intent, "facet": facet_name, "available": True,
                    "result": None, "error": f"bad_args:{exc}"}
        return {
            "intent": intent,
            "facet": facet_name,
            "available": getattr(self, facet_name) is not None,
            "result": result,
        }

    # ── readout ───────────────────────────────────────────────────────────

    def status(self) -> Dict[str, Any]:
        """Which facets are live, and each one's own status where it exposes one."""
        out: Dict[str, Any] = {"module": "UnifiedWorldModel", "facets": {}}
        for name in ("learned", "causal", "outcome"):
            facet = getattr(self, name)
            entry: Dict[str, Any] = {"available": facet is not None}
            if facet is not None:
                for status_attr in ("get_status", "status", "to_dict"):
                    if hasattr(facet, status_attr):
                        try:
                            entry["detail"] = getattr(facet, status_attr)()
                        except (AttributeError, RuntimeError, OSError, ValueError, TypeError) as exc:
                            record_degradation("unified_world_model", exc, severity="debug")
                        break
            out["facets"][name] = entry
        return out


_instance: Optional[UnifiedWorldModel] = None


def get_unified_world_model() -> UnifiedWorldModel:
    global _instance
    if _instance is None:
        _instance = UnifiedWorldModel()
    return _instance
