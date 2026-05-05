"""Post-run analysis: causal trace, failure hypotheses, and lessons.

This module is environment-agnostic. It receives structured frames, receipts,
and semantic outcomes and produces a structured postmortem document.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PostmortemReport:
    """Structured failure analysis after an environment run terminates."""
    run_id: str
    environment_id: str
    mode: str
    terminal_reason: str  # "death", "crash", "timeout", "success", "contamination"
    death_message: str = ""
    final_parsed_state: dict[str, Any] = field(default_factory=dict)
    last_n_actions: list[dict[str, Any]] = field(default_factory=list)
    last_n_observations: list[str] = field(default_factory=list)
    ranked_candidates_at_death: list[dict[str, Any]] = field(default_factory=list)
    receipt_chain: list[str] = field(default_factory=list)
    resource_trends: dict[str, list[float]] = field(default_factory=dict)
    semantic_outcomes: list[str] = field(default_factory=list)
    prediction_errors: list[dict[str, Any]] = field(default_factory=list)
    known_hazards: list[str] = field(default_factory=list)
    avoidable_failure_hypotheses: list[str] = field(default_factory=list)
    recommended_regressions: list[str] = field(default_factory=list)
    total_steps: int = 0
    duration_s: float = 0.0
    created_at: float = field(default_factory=time.time)


class PostmortemGenerator:
    """Generates a PostmortemReport from a terminated run's frame history."""

    def __init__(self, lookback: int = 20):
        self.lookback = lookback

    def generate(
        self,
        *,
        run_id: str,
        environment_id: str,
        mode: str,
        terminal_reason: str,
        frames: list,
        started_at: float,
    ) -> PostmortemReport:
        last_frames = frames[-self.lookback:] if frames else []

        last_actions = []
        last_observations = []
        receipt_ids = []
        semantic_outcomes = []
        prediction_errors = []

        for f in last_frames:
            if f.action_intent:
                last_actions.append({
                    "name": f.action_intent.name,
                    "risk": f.action_intent.risk,
                    "parameters": f.action_intent.parameters,
                })
            last_observations.append(f.observation.text if hasattr(f.observation, "text") else "")
            if f.receipt:
                receipt_ids.append(f.receipt.receipt_id)
            if f.outcome_assessment:
                semantic_outcomes.extend(f.outcome_assessment.observed_events)

        death_message = ""
        final_state = {}
        if last_frames:
            final = last_frames[-1]
            if final.parsed_state:
                final_state = final.parsed_state.self_state
            # Check for death indicators
            obs_text = final.observation.text if hasattr(final.observation, "text") else ""
            if any(p in obs_text for p in ("You die", "DYWYPI", "Do you want your possessions identified")):
                death_message = obs_text

        # Generate avoidable failure hypotheses
        hypotheses = []
        if death_message:
            if any(a.get("risk") in ("risky", "irreversible") for a in last_actions[-3:]):
                hypotheses.append("Final actions included risky/irreversible intents near death")
            if any("blocked_by_wall" in o or "position_unchanged" in o for o in semantic_outcomes[-5:]):
                hypotheses.append("Repeated ineffective actions preceded terminal state")

        return PostmortemReport(
            run_id=run_id,
            environment_id=environment_id,
            mode=mode,
            terminal_reason=terminal_reason,
            death_message=death_message,
            final_parsed_state=final_state,
            last_n_actions=last_actions,
            last_n_observations=last_observations,
            receipt_chain=receipt_ids,
            semantic_outcomes=semantic_outcomes,
            prediction_errors=prediction_errors,
            known_hazards=[],
            avoidable_failure_hypotheses=hypotheses,
            recommended_regressions=[],
            total_steps=len(frames),
            duration_s=time.time() - started_at,
        )


__all__ = ["PostmortemReport", "PostmortemGenerator"]
