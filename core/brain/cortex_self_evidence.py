"""Typed, evidence-bound facts about Aura's currently resident cortex.

This is an introspection reader, not a response policy. It resolves the one
validated active-cortex authority and the independently verified qualified
semantic-tissue authority, then emits compact assertions for the existing
language substrate. Missing evidence remains explicitly unmeasured.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class CortexSelfEvidence:
    resident_label: str
    model_type: str
    total_parameters: int
    native_context_tokens: int
    served_context_tokens: int
    promotion_verdict: str
    identity_behavior_changed: bool | None
    component_states: tuple[tuple[str, str], ...]
    semantic_active: bool
    semantic_verdict: str
    semantic_task_count: int
    semantic_exact_by_arm: tuple[tuple[str, int], ...]
    semantic_gain_count: int
    semantic_regression_count: int
    semantic_p_value: float | None

    def assertions(self) -> tuple[str, ...]:
        """Render privacy-safe facts for the operational language substrate."""

        lines = [
            "Resident cortex: "
            f"{self.resident_label}, {self.model_type}, "
            f"{self.total_parameters:,} parameters; native context "
            f"{self.native_context_tokens:,} tokens and currently qualified serving "
            f"context {self.served_context_tokens:,} tokens.",
        ]
        if self.promotion_verdict:
            behavior = (
                " The migration authority records that model-generation identity "
                "behavior changed."
                if self.identity_behavior_changed is True
                else ""
            )
            lines.append(
                f"Cortex promotion evaluation: {self.promotion_verdict}.{behavior}"
            )
        if self.component_states:
            dispositions = ", ".join(
                f"{name}={state}" for name, state in self.component_states
            )
            lines.append(f"Cortex migration components: {dispositions}.")
        if self.semantic_active and self.semantic_task_count > 0:
            arms = dict(self.semantic_exact_by_arm)
            lines.append(
                "Measured bounded recurrent semantic tissue: "
                f"{self.semantic_verdict}; treatment "
                f"{arms.get('treatment', 0)}/{self.semantic_task_count}, ordinary "
                f"decode {arms.get('ordinary_base', 0)}/{self.semantic_task_count}, "
                f"gain {self.semantic_gain_count}, regressions "
                f"{self.semantic_regression_count}"
                + (
                    f", paired exact p={self.semantic_p_value:.3g}."
                    if self.semantic_p_value is not None
                    else "."
                )
            )
        lines.append(
            "Cortex comparison boundary: no paired evidence currently attributes "
            "differences in conversational style, association speed, broad reasoning, "
            "knowledge, or subjective experience to the model swap; those differences "
            "are unmeasured, not observations."
        )
        return tuple(lines)


def _component_states(value: Any) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, dict):
        return ()
    states: list[tuple[str, str]] = []
    for name, component in sorted(value.items()):
        if not isinstance(component, dict):
            continue
        status = str(component.get("status") or "").strip()
        if status:
            states.append((str(name), status))
    return tuple(states)


def resolve_cortex_self_evidence() -> CortexSelfEvidence | None:
    """Resolve current cortex facts only from validated runtime authorities."""

    from core.brain.llm.model_registry import (
        get_active_cortex_spec,
        resident_model_identity,
    )
    from core.brain.llm.semantic_neural_serving import (
        semantic_neural_serving_status,
    )

    spec = get_active_cortex_spec()
    if spec is None or not spec.exact_identity:
        return None
    identity = resident_model_identity()
    migration = spec.migration_contract() or {}
    evaluation = spec.evaluation() or {}
    semantic_status = semantic_neural_serving_status(spec.model_path)
    semantic_receipt = (
        semantic_status.get("receipt") if semantic_status.get("active") is True else {}
    )
    qualification = (
        semantic_receipt.get("qualification")
        if isinstance(semantic_receipt, dict)
        else {}
    )
    if not isinstance(qualification, dict):
        qualification = {}
    exact_by_arm = qualification.get("independent_exact_by_arm")
    exact_items = (
        tuple(sorted((str(name), int(score)) for name, score in exact_by_arm.items()))
        if isinstance(exact_by_arm, dict)
        else ()
    )
    p_value = qualification.get("paired_one_sided_exact_p")
    return CortexSelfEvidence(
        resident_label=str(identity.get("label") or spec.size_class or "resident"),
        model_type=str(identity.get("model_type") or "unknown"),
        total_parameters=int(identity.get("total_parameters") or 0),
        native_context_tokens=int(identity.get("native_context_window") or 0),
        served_context_tokens=int(identity.get("served_context_tokens") or 0),
        promotion_verdict=str(evaluation.get("verdict") or ""),
        identity_behavior_changed=(
            evaluation.get("identity_behavior_changed")
            if isinstance(evaluation.get("identity_behavior_changed"), bool)
            else None
        ),
        component_states=_component_states(migration.get("components")),
        semantic_active=semantic_status.get("active") is True,
        semantic_verdict=str(qualification.get("verdict") or ""),
        semantic_task_count=int(qualification.get("task_count") or 0),
        semantic_exact_by_arm=exact_items,
        semantic_gain_count=int(qualification.get("gain_count") or 0),
        semantic_regression_count=int(qualification.get("regression_count") or 0),
        semantic_p_value=float(p_value) if isinstance(p_value, (int, float)) else None,
    )


__all__ = ["CortexSelfEvidence", "resolve_cortex_self_evidence"]
