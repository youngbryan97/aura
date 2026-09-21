"""What a turn does to the state on the way in and on the way out, whichever path runs it.

A turn in the desktop runtime enters through `KernelInterface.process`, which
puts the message into working memory and tells the body somebody is there, and
runs through `AuraKernel.tick`, which clears the last turn's leftovers before
the phases and stamps who acted after them. The subject-core campaign runs the
same phases from `core/subject/driver.py` and ran nothing around them. So in
every campaign the partner's message never reached working memory: every
organ that reads the conversation read an empty one, and 109 of 315 columns
stayed constant for all 2,400 turns of the 21 September run, most of them the
readings the song organs write. The last action's source was never stamped, so
four columns of deliberation were constant too.

Each step here was lifted out of the path that already ran it, unchanged, and
that path now calls it. The campaign calls the same functions, so the two
cannot drift apart again. Nothing here writes to disk: persistence, the vault
commit and the trace stay in the tick, because a write that lands in one arm of
a paired intervention and not the other is a difference nothing thought.

Names from `aura_kernel` and `kernel_interface` are imported at call time. Both
modules import this one, and a test that patches a name on either has to reach
the code that reads it.
"""

from __future__ import annotations

import time
from typing import Any

__all__ = [
    "USER_ORIGINS",
    "admit_message",
    "bind_objective",
    "clear_last_turn",
    "finish_foreground",
    "note_presence",
    "objective_to_bind",
    "stamp_closure",
]

#: Origins that mean a person sent the message. Moved here from
#: kernel_interface, which reads it from here.
USER_ORIGINS = frozenset(
    {"user", "voice", "admin", "api", "gui", "ws", "websocket", "direct", "external", "test"}
)


def note_presence(message: str, origin: str, *, conversation_id: str) -> None:
    """Somebody is here: the idle clock, the subcortical stimulus and the taste loop.

    Only for a message a person sent. Each step failing is recorded and the
    turn goes on, as it did in `KernelInterface.process`.
    """
    from core.kernel.kernel_interface import _emit_kernel_fault, resolve_orchestrator

    if origin not in USER_ORIGINS:
        return
    # Update the orchestrator's user-interaction timestamp so idle
    # detectors (substrate decay, sleep triggers, proactive presence)
    # know the user is present. Without this, the kernel path bypasses
    # the orchestrator entirely and the system thinks it's been idle
    # for the entire session.
    try:
        orch = resolve_orchestrator()
        if orch is not None:
            orch._last_user_interaction_time = time.time()
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        _emit_kernel_fault(
            exc,
            action="continued kernel processing without orchestrator idle timestamp update",
            severity="warning",
            stage="process.touch_orchestrator",
        )
    # Signal the subcortical core that a stimulus has arrived.
    # This raises arousal, opens the thalamic gate, and restores
    # full mesh/substrate gain for the duration of user interaction.
    try:
        from core.consciousness.subcortical_core import get_subcortical_core

        get_subcortical_core().receive_stimulus(intensity=1.0, source=origin)
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        _emit_kernel_fault(
            exc,
            action="continued kernel processing without subcortical stimulus side-effect",
            severity="warning",
            stage="process.subcortical_stimulus",
        )
    # Taste loop: read this message as Bryan's reaction to the response we sent
    # last turn, and nudge the personalized TasteModel (inference-time alignment).
    try:
        from core.brain.conversation_outcome import register_reaction

        # Same key the amplifier recorded under, so a reaction is
        # matched to the response it is actually replying to.
        register_reaction(message, conversation_id=conversation_id)
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        _emit_kernel_fault(
            exc,
            action="continued kernel processing without taste-loop reaction update",
            severity="warning",
            stage="process.taste_reaction",
        )


def admit_message(state: Any, message: str, origin: str) -> None:
    """Put the message into working memory as the person's turn, once.

    Raises what it met, as the inline version did, so the caller's handoff
    guard still decides what a broken state means.
    """
    cognition = getattr(state, "cognition", None)
    if cognition is None:
        raise AttributeError("kernel state has no cognition")
    wm = getattr(cognition, "working_memory", None)
    if wm is None:
        wm = []
        cognition.working_memory = wm
    if not isinstance(wm, list):
        raise TypeError("kernel working_memory must be a list")
    # Avoid duplicating if routing phase or upstream already added it.
    last = wm[-1] if wm else {}
    if not isinstance(last, dict):
        last = {}
    if last.get("role") != "user" or last.get("content") != message:
        wm.append(
            {
                "role": "user",
                "content": message,
                "timestamp": time.time(),
                "origin": origin,
            }
        )


def clear_last_turn(state: Any, objective: str, turn_origin: str) -> None:
    """Objective hygiene and the last turn's transient modifiers, before any phase.

    Objective/initiative hygiene is a tick-lifecycle operation. It must happen
    before phase provenance starts; doing it inside AuraState.derive() falsely
    attributed the cleanup to whichever phase happened to derive next. The
    per-turn prompt and runtime modifiers are cleared so stale tool results,
    open social-thread directives, recovery flags or proof contracts cannot
    leak into an unrelated turn.
    """
    from core.kernel.aura_kernel import _record_kernel_degradation

    state.prepare_tick_boundary()
    try:
        from core.runtime.proof_policy import (
            clear_transient_response_modifiers,
            is_proof_repair_prompt,
            proof_persistent_objective,
            proof_run_active,
        )

        proof_active = proof_run_active(origin=turn_origin)
        bound_proof_objective = proof_persistent_objective(
            objective,
            origin=turn_origin,
        )
        clear_transient_response_modifiers(
            state.response_modifiers,
            strict=proof_active,
        )
        if proof_active:
            state.response_modifiers["proof_turn_objective"] = bound_proof_objective
            if is_proof_repair_prompt(objective, origin=turn_origin):
                state.response_modifiers["proof_repair_turn"] = True
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        _record_kernel_degradation(
            exc,
            action="continued tick after transient response-modifier scrub failed",
            severity="error",
        )
        for _stale_key in (
            "last_skill_run",
            "last_skill_ok",
            "last_skill_result_payload",
            "matched_skills",
            "intent_type",
            "precomputed_grounded_reply",
            "last_task_outcome",
            "last_task_id",
            "auto_browse_urls",
            "conversational_dynamics",
            "conv_dynamics_state",
            "response_contract",
        ):
            state.response_modifiers.pop(_stale_key, None)


def objective_to_bind(objective: str, turn_origin: str) -> str:
    """The objective as the tick binds it: a proof run's persistent objective, or the one given."""
    from core.kernel.aura_kernel import _record_kernel_degradation

    try:
        from core.runtime.proof_policy import proof_persistent_objective

        return proof_persistent_objective(
            objective,
            origin=turn_origin,
        )
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        _record_kernel_degradation(
            exc,
            action="continued tick after proof objective binding normalization failed",
            severity="error",
        )
        return objective


def bind_objective(state: Any, bound_objective: str) -> None:
    """The turn's objective, recorded with the executive that owns bindings."""
    from core.kernel.aura_kernel import get_executive_authority

    state.cognition.current_objective = bound_objective
    get_executive_authority().record_objective_binding(
        state,
        bound_objective,
        source="aura_kernel.tick",
        mode="unitary_tick",
        reason="kernel_tick_bound",
    )


def stamp_closure(state: Any, tick_id: Any) -> None:
    """Who acted this turn and what the executive decided, stamped into the state.

    Every committed state is self-documenting about the decision chain.
    """
    from core.kernel.aura_kernel import _record_kernel_degradation, logger

    try:
        state.cognition.last_kernel_cycle_id = tick_id
        state.cognition.last_action_source = state.cognition.current_origin or "kernel"

        from core.executive.executive_core import get_executive_core

        _exec = get_executive_core()
        if _exec is not None:
            _exec_stats = _exec.get_stats() if hasattr(_exec, "get_stats") else {}
            state.cognition.kernel_decision_count = int(_exec_stats.get("approved", 0) or 0)
            state.cognition.kernel_veto_count = int(_exec_stats.get("rejected", 0) or 0)
            _recent = _exec_stats.get("recent_decisions", []) or []
            state.cognition.last_veto_reasons = [
                str(d.get("reason", ""))
                for d in _recent
                if isinstance(d, dict) and d.get("outcome") == "rejected"
            ][-5:]
    except (ImportError, AttributeError, RuntimeError) as _cc_err:
        _record_kernel_degradation(
            _cc_err,
            action="continued tick without constitutional closure state stamp",
            severity="error",
        )
        logger.error("Constitutional closure stamp failed: %s", _cc_err, exc_info=True)


def finish_foreground(state: Any, *, objective: str, turn_origin: str) -> None:
    """Close a foreground objective as a live turn rather than a durable goal."""
    from core.goals.objective_lifecycle import finalize_foreground_turn_state
    from core.kernel.aura_kernel import ServiceContainer, logger

    receipt = finalize_foreground_turn_state(
        state,
        objective=objective,
        origin=turn_origin,
    )
    closure = ServiceContainer.get("executive_closure", default=None)
    if closure is not None and hasattr(closure, "complete_foreground_turn"):
        closure.complete_foreground_turn(objective, turn_origin)
    if receipt.get("preserved_background"):
        logger.debug(
            "Kernel: preserved a post-turn background objective after closing %s.",
            receipt.get("objective_digest") or "foreground turn",
        )
