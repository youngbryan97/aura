from __future__ import annotations

from typing import Any

from core.runtime import background_policy

_LOW_VALUE_BACKGROUND_PREFIXES = (
    "[identity refresh:",
    "researching ",
)

_LOW_VALUE_BACKGROUND_MARKERS = (
    "cognitive baseline tick",
    "background cognitive state",
    "quietly consolidating memory",
    "seek novel stimulation",
    "initiating social engagement",
    "silent auto-fix",
)

_SYNTHETIC_BACKGROUND_NOISE_MARKERS = (
    "task exception",
    "traceback",
    "future: <task finished",
    "database is locked",
    "full shm snapshot overflow",
    "runtime error",
    "warning:",
)


def normalize_objective_text(objective: Any) -> str:
    return " ".join(str(objective or "").lower().split())


def is_low_value_background_objective(objective: Any) -> bool:
    normalized = normalize_objective_text(objective)
    if not normalized:
        return False
    if any(normalized.startswith(prefix) for prefix in _LOW_VALUE_BACKGROUND_PREFIXES):
        return True
    return any(marker in normalized for marker in _LOW_VALUE_BACKGROUND_MARKERS)


def looks_like_background_noise(objective: Any) -> bool:
    normalized = normalize_objective_text(objective)
    if not normalized:
        return True
    return any(marker in normalized for marker in _SYNTHETIC_BACKGROUND_NOISE_MARKERS)


def background_response_suppression_reason(
    objective: Any,
    *,
    orchestrator: Any = None,
    include_synthetic_noise: bool = True,
) -> str:
    if is_low_value_background_objective(objective):
        return "low_value_autonomous_objective"
    if include_synthetic_noise and looks_like_background_noise(objective):
        return "synthetic_noise"

    return (
        background_policy.background_activity_reason(
            orchestrator,
            profile=background_policy.THOUGHT_BACKGROUND_POLICY,
        )
        or ""
    )


def clear_background_generation(state: Any, objective: Any) -> None:
    state.cognition.last_response = ""
    if is_low_value_background_objective(objective):
        state.cognition.current_objective = ""

        # ── Drive satisfaction on objective completion ──
        # Close the homeostatic loop: completing a goal satisfies the drive
        try:
            from core.container import ServiceContainer
            drive = ServiceContainer.get("drive_engine", default=None)
            if drive:
                import asyncio
                # Satisfy competence (completed a task) and curiosity (learned something)
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(drive.satisfy("competence", 10.0))
                    loop.create_task(drive.satisfy("curiosity", 5.0))
                except RuntimeError:
                    pass  # no event loop — skip
        except Exception:
            pass
