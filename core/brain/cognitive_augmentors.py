"""The optional organs a turn may pass through, and what each one learned.

An augmentor is allowed to fail. That is the whole point of registering them
separately: spiking inference, entity memory, the imagination workspace, the
bicameral advisory and the situation frame each improve a turn when they work
and must leave it untouched when they do not. Each one that runs gets its
outcome recorded, so an augmentor that never helps can be seen rather than
assumed.
"""
from __future__ import annotations

from typing import Any

from core.brain import advisory_passes
from core.runtime.errors import record_degradation
from core.state.aura_state import AuraState


class _RunsItsAugmentors:
    """Lifted whole from CognitiveEngine; see cognitive_engine.py."""

    def register_augmentor(self, augmentor: Any) -> bool:
        """Register a cognitive augmentor (e.g. SovereignWebAugmentor).

        This accepted any object at all: no declared name, no callable
        contract, no bound on what it returns — and its output goes into the
        prompt. The admission below is small on purpose (this is an in-process
        extension point, not a plugin marketplace) but it is a contract rather
        than a shrug, and a refusal is recorded instead of silently producing
        an augmentor that raises AttributeError on every turn.
        """
        # Imported here rather than at module level: the module these
        # came from imports this one to build the class. A call-time
        # import also still sees a test's patch of the original.
        from .cognitive_engine import (
            logger,
        )

        getter = getattr(augmentor, "get_augmentation", None)
        if not callable(getter):
            record_degradation(
                "cognitive_engine",
                TypeError(
                    f"augmentor {type(augmentor).__name__} has no callable get_augmentation"
                ),
                severity="warning",
                action="refused to register an augmentor with no augmentation contract",
            )
            return False
        if augmentor in self._augmentors:
            return True
        self._augmentors.append(augmentor)
        self._augmentor_registry_receipt = [
            type(existing).__name__ for existing in self._augmentors
        ]
        logger.info("🧠 CognitiveEngine: Registered augmentor %s", type(augmentor).__name__)
        return True

    def augmentor_registry_receipt(self) -> list[str]:
        """Which augmentors may contribute to a turn."""
        return list(getattr(self, "_augmentor_registry_receipt", []) or [])

    @classmethod
    def _bounded_augmentation(cls, raw: Any) -> Any:
        """Bound and neutralize what an augmentor contributes to the prompt.

        Augmentor output is not one of Aura's own measurements — it is
        whatever a registered object returned — and it reaches the prompt. It
        is bounded, and its text cannot forge contract structure.
        """
        from core.brain.living_mind_context import neutralize_learned_text

        if isinstance(raw, str):
            return neutralize_learned_text(raw)[: cls._AUGMENTATION_CHAR_LIMIT]
        if isinstance(raw, dict):
            return {
                str(key)[:120]: cls._bounded_augmentation(value)
                for key, value in list(raw.items())[:32]
            }
        if isinstance(raw, (list, tuple)):
            return [cls._bounded_augmentation(item) for item in list(raw)[:32]]
        return raw

    def _apply_spiking_active_inference(
        self,
        state: AuraState,
        objective: str,
        origin: str,
        context: dict[str, Any] | None,
        *,
        is_background: bool,
    ) -> dict[str, Any] | None:
        return advisory_passes.apply_spiking_active_inference(
            state, objective, origin, context, is_background=is_background
        )

    def _apply_entity_memory(
        self,
        state: AuraState,
        objective: str,
        context: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Make what Aura knows about the people/places/things in play causal.

        This runs on the live path so recognising an entity actually changes
        retrieval depth, retrieval targeting, and affect before the answer is
        generated — see core/memory/entity_memory_bridge.py, which owns the
        effects. Failure here is never fatal: the turn proceeds without the
        entity context, which is exactly how it behaved before this existed.
        """
        from .cognitive_engine import (
            _COGNITIVE_ENGINE_RECOVERABLE_ERRORS,
            logger,
        )

        merged_context = dict(context or {})
        try:
            from core.memory.entity_memory_bridge import apply_entity_context

            # source="user": the objective is what the interlocutor asked, so
            # it may introduce entities. Aura's own generated text is never
            # passed here — a name she invented must not become a permanent
            # member of her world that later mentions then "confirm".
            summary = apply_entity_context(
                state, objective, merged_context,
                source="user",
                evidence_id=str(merged_context.get("evidence_id") or ""),
            )
            if summary.get("entities"):
                merged_context["entity_memory"] = (
                    summary.get("context", {}).get("entity_memory")
                    or merged_context.get("entity_memory")
                )
                state.cognition.modifiers["entity_memory_effects"] = list(
                    summary.get("effects", [])
                )
                logger.debug(
                    "🧠 Entity memory: %d entity(ies) in play, %d effect(s).",
                    len(summary["entities"]), len(summary.get("effects", [])),
                )
        except _COGNITIVE_ENGINE_RECOVERABLE_ERRORS as exc:
            record_degradation(
                "cognitive_engine",
                exc,
                severity="warning",
                action="continued cognitive cycle without entity memory context",
            )
            logger.debug("Entity memory context skipped: %s", exc)
        return merged_context

    def _apply_imagination_workspace(
        self,
        state: AuraState,
        objective: str,
        origin: str,
        context: dict[str, Any] | None,
        *,
        is_background: bool,
    ) -> dict[str, Any] | None:
        return advisory_passes.apply_imagination_workspace(
            state, objective, origin, context, is_background=is_background
        )

    def _apply_bicameral_advisory(
        self,
        state: AuraState,
        objective: str,
        origin: str,
        context: dict[str, Any] | None,
        *,
        is_background: bool,
    ) -> dict[str, Any] | None:
        return advisory_passes.apply_bicameral_advisory(
            state, objective, origin, context, is_background=is_background
        )

    def _apply_cognitive_situation_frame(
        self,
        state: AuraState,
        objective: str,
        origin: str,
        context: dict[str, Any] | None,
        *,
        is_background: bool,
    ) -> dict[str, Any] | None:
        return advisory_passes.apply_cognitive_situation_frame(
            state, objective, origin, context, is_background=is_background
        )

    def _learn_spiking_active_inference_outcome(
        self,
        context: dict[str, Any] | None,
        *,
        outcome: str,
        reward: float,
    ) -> dict[str, Any] | None:
        return advisory_passes.learn_spiking_active_inference_outcome(
            context, outcome=outcome, reward=reward
        )

    def _learn_imagination_workspace_outcome(
        self,
        context: dict[str, Any] | None,
        *,
        outcome: str,
        reward: float,
        evidence_basis: str = "",
        evidence_id: str = "",
    ) -> dict[str, Any] | None:
        return advisory_passes.learn_imagination_workspace_outcome(
            context,
            outcome=outcome,
            reward=reward,
            evidence_basis=evidence_basis,
            evidence_id=evidence_id,
        )

    def _learn_bicameral_advisory_outcome(
        self,
        context: dict[str, Any] | None,
        *,
        outcome: str,
        reward: float,
    ) -> dict[str, Any] | None:
        return advisory_passes.learn_bicameral_advisory_outcome(
            context, outcome=outcome, reward=reward
        )
