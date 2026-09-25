"""Live, outcome-grounded intake for the shared semantic-development cycle.

The capability engine owns the execution receipt. This adapter sees only
pre-execution metadata and the skill's returned status; it cannot treat model
text, proposed plans, or tool summaries as observed task success.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import Any

from core.cognition.concept_formation import get_concept_formation_engine
from core.cognition.concept_handle import get_concept_registry
from core.cognition.semantic_development import SemanticCase, SemanticDevelopment
from core.runtime.lockdep import checked_lock

logger = logging.getLogger("Aura.SemanticDevelopment")
_OUTCOME = "skill_returned_ok"
_SERVICE: SemanticDevelopment | None = None
_SERVICE_LOCK = checked_lock("core.cognition.semantic_runtime.singleton")
_SAVE_EVERY = 16
_DISCOVER_EVERY = 32


@dataclass(frozen=True, slots=True)
class SkillTrial:
    """One proposed action and the features available before its result."""

    source_id: str
    context_id: str
    features: dict[str, Any]
    prediction_ids: tuple[str, ...]


def get_semantic_development() -> SemanticDevelopment:
    global _SERVICE
    if _SERVICE is not None:
        return _SERVICE
    with _SERVICE_LOCK:
        if _SERVICE is None:
            _SERVICE = SemanticDevelopment(
                registry=get_concept_registry(),
                concept_engine=get_concept_formation_engine(),
            )
        return _SERVICE


def prepare_skill_trial(
    skill_name: str, params: dict[str, Any], context: dict[str, Any]
) -> SkillTrial:
    """Commit compatible hypotheses before the skill is actually invoked."""
    service = get_semantic_development()
    source_id = "skill:" + str(uuid.uuid4())
    context_id = "skill:" + skill_name
    features = {
        "skill": skill_name,
        "origin": str(context.get("origin") or "unknown")[:64],
        "effect_scope": str(context.get("effect_scope") or "unknown")[:64],
        "parameter_count": min(len(params), 64),
    }
    for name, value in sorted(params.items())[:16]:
        if isinstance(name, str) and name.isidentifier() and len(name) <= 48:
            features["parameter_type:" + name] = type(value).__name__
    predicted = []
    for proposal_id in service.eligible_proposals(_OUTCOME, features):
        try:
            predicted.append(service.commit_prediction(
                proposal_id, source_id=source_id, context_id=context_id,
                features=features))
        except ValueError as exc:
            logger.debug("Skill hypothesis had no eligible prospective path: %s", exc)
    if predicted:
        service.save()
    return SkillTrial(source_id, context_id, features, tuple(predicted))


def complete_skill_trial(trial: SkillTrial, result: dict[str, Any]) -> None:
    """Compare commitments to the returned result and retain the observation."""
    service = get_semantic_development()
    case = SemanticCase(
        trial.source_id, trial.context_id, _OUTCOME,
        result.get("ok") is True, trial.features,
    )
    if not service.observe(case):
        return
    count = service.observation_count
    if count >= 3 * service.min_support and count % _DISCOVER_EVERY == 0:
        finding = service.test(_OUTCOME)
        logger.info("Skill-outcome semantic trial: %s", finding["status"])
        proposed = service.propose_from_fit(_OUTCOME)
        logger.debug("Fit-derived skill hypotheses prepared: %d", len(proposed))
    if trial.prediction_ids or count % _SAVE_EVERY == 0:
        service.save()


__all__ = ["SkillTrial", "complete_skill_trial", "get_semantic_development",
           "prepare_skill_trial"]
