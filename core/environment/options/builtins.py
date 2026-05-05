"""Built-in general options shared across environment families."""
from __future__ import annotations

from .base import Option


def builtin_options() -> dict[str, Option]:
    names = {
        "RESOLVE_MODAL": ("resolve active modal state", ["modal_active"], ["modal_cleared"]),
        "OBSERVE_MORE": ("gather more evidence", ["uncertainty_high"], ["uncertainty_reduced"]),
        "INSPECT_OBJECT": ("inspect object safely", ["object_visible"], ["object_understood"]),
        "STABILIZE_RESOURCE": ("recover a critical resource", ["resource_critical"], ["resource_stable"]),
        "RETREAT_FROM_HAZARD": ("increase distance from hazard", ["hazard_near"], ["risk_reduced"]),
        "NAVIGATE_TO_GOAL": ("navigate to known goal", ["path_known"], ["at_goal"]),
        "EXPLORE_FRONTIER": ("explore an unknown frontier", ["stable", "frontier_known"], ["new_information"]),
        "USE_KNOWN_SAFE_AFFORDANCE": ("use a validated affordance", ["affordance_known"], ["expected_effect_observed"]),
        "RUN_DIAGNOSTIC": ("run a reversible diagnostic", ["diagnostic_available"], ["diagnosis_known"]),
        "RECOVER_FROM_LOOP": ("break repeated failure loop", ["loop_detected"], ["loop_broken"]),
        "BACKTRACK": ("return to a prior stable context", ["prior_context_known"], ["context_stable"]),
        "SUMMARIZE_CONTEXT": ("compress context under budget pressure", ["context_large"], ["context_compressed"]),
        "SAVE_CHECKPOINT": ("save reversible checkpoint where allowed", ["snapshot_supported"], ["checkpoint_saved"]),
        "ROLLBACK_LAST_CHANGE": ("rollback a reversible change", ["rollback_available"], ["rollback_complete"]),
    }
    return {
        name: Option(
            name=name,
            description=description,
            initiation_conditions=initiation,
            termination_conditions=termination,
            expected_effects=termination,
            failure_conditions=["precondition_failed", "budget_exhausted"],
            risk_tags=set(),
            policy_name=name.lower(),
            max_steps=5 if name in {"RESOLVE_MODAL", "RECOVER_FROM_LOOP"} else 50,
            cooldown_steps=2 if name in {"RECOVER_FROM_LOOP", "ROLLBACK_LAST_CHANGE"} else 0,
        )
        for name, (description, initiation, termination) in names.items()
    }


__all__ = ["builtin_options"]
