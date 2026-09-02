"""interface/routes/chat_desktop_objective_gates.py — the execution chokepoints.

Two gates lifted out of ``_api_chat_turn``, where they were closures over the
turn's state. Their captured names are parameters and their bodies are
unchanged — statement for statement and docstring word for docstring word,
checked before the move.

They matter for a reason the size of the file has nothing to do with: every
reply exit routes through the chokepoint, so an objective cannot be executed
twice and a lane cannot claim it did something it did not. That is easier to
test as a function than as a closure inside a 4,000-line handler.

``chat.py`` keeps thin forwarders of the original names, so the call sites
inside the handler read as they did.
"""

from __future__ import annotations

import logging
from typing import Any

from core.runtime.errors import record_degradation
from interface.routes import chat_capability_inventory as _chat_capability_inventory
from interface.routes import chat_desktop_objective as _chat_desktop_objective
from interface.routes import chat_desktop_repair as _chat_desktop_repair
from interface.routes import chat_preflight as _chat_preflight
from interface.routes.chat_common import _CHAT_RECOVERABLE_ERRORS

logger = logging.getLogger("Aura.Chat")

__all__ = [
    "_lifted_apply_desktop_objective_chokepoint",
    "_lifted_run_desktop_objective_tracked",
]


async def _lifted_run_desktop_objective_tracked(
    message: str, *, cognitive_reply: str,
    _desktop_exec_state: Any,
    conversation_only_surface: Any,
    pending_exchange_id: Any,
) -> dict[str, Any] | None:
    """Single execution gate for desktop objectives.

    EVERY caller routes here so the step receipts always land in
    _desktop_exec_state (the reply doors attach them to the wire)
    and the chokepoint cannot double-execute an objective another
    lane already ran. Visible-demo rounds 3-5: the pre-freeform
    desktop lane called the executor directly, so the doors saw
    attempted=False/result=None and served receipt-less replies.
    """
    if conversation_only_surface:
        _desktop_exec_state["attempted"] = True
        return {
            "ok": False,
            "status": "paired_device_action_scope_denied",
            "response": (
                "This paired device is scoped to conversation and read-only world viewing. "
                "Desktop, file, tool, and control actions require the owner surface."
            ),
        }
    _desktop_exec_state["attempted"] = True
    executed = await _chat_desktop_objective._execute_desktop_objective_from_chat(
        message, cognitive_reply=cognitive_reply
    )
    if isinstance(executed, dict):
        _desktop_exec_state["result"] = executed.get("result")
        try:
            from core.conversation.action_episode import (
                action_episode_from_execution,
            )

            episode_authority_proven = not bool(executed.get("ok"))
            episode_authority_reason = "governed_executor_reported_failure"
            if bool(executed.get("ok")):
                episode_result = executed.get("result")
                if isinstance(episode_result, dict):
                    (
                        episode_authority_proven,
                        episode_authority_reason,
                    ) = _chat_desktop_objective._verified_desktop_task_result(
                        episode_result
                    )
                else:
                    episode_authority_proven = False
                    episode_authority_reason = "desktop_result_missing"
            action_episode = action_episode_from_execution(
                message,
                executed,
                capability="desktop_task",
                authority_kind="governed_action_episode",
                authority_proven=episode_authority_proven,
                authority_reason=episode_authority_reason,
            )
            if action_episode is not None:
                _desktop_exec_state["action_episode"] = action_episode.to_dict()
                await _chat_preflight._attach_logged_exchange_metadata(
                    pending_exchange_id,
                    {"action_episode": action_episode.to_dict()},
                )
        except _CHAT_RECOVERABLE_ERRORS as episode_exc:
            record_degradation("chat.action_episode", episode_exc)
    return executed


async def _lifted_apply_desktop_objective_chokepoint(
    final_text: str, status: str,
    *,
    _desktop_exec_state: Any,
    _semantic_user_message: Any,
    conversation_only_surface: Any,
    is_benchmark: Any,
    _run_desktop_objective_tracked: Any,
) -> tuple[str, str]:
    """Execute-or-stay-honest gate shared by EVERY reply exit.

    Round-10 live proof: the kernel/deep lane exits through its
    own response build and served a confabulated 'I created the
    folder' (with a fabricated 60-trillion-parameter self-claim)
    while no tool ever dispatched — the chokepoint guarded only
    the fastpath door. Both doors now pass through here.
    """
    if conversation_only_surface and _chat_preflight._looks_like_desktop_objective(
        _semantic_user_message
    ):
        return (
            "This paired device is scoped to conversation and read-only world viewing. "
            "Desktop, file, tool, and control actions require the owner surface.",
            "paired_device_action_scope_denied",
        )
    if (
        is_benchmark
        or _desktop_exec_state["attempted"]
        or str(status or "").startswith(
            (
                "live_proof",
                "desktop_objective",
                "file_operation",
                "web_interlocutor",
                "program_dna",
            )
        )
        or _chat_desktop_objective._blocks_consequential_desktop_execution(_semantic_user_message)
        or _chat_capability_inventory._looks_like_program_dna_execution_request(_semantic_user_message)
        or not _chat_preflight._looks_like_desktop_objective(_semantic_user_message)
    ):
        return final_text, status
    try:
        _executed = await _run_desktop_objective_tracked(
            _semantic_user_message,
            cognitive_reply=final_text,
        )
    except _CHAT_RECOVERABLE_ERRORS as _exec_exc:
        record_degradation("chat", _exec_exc)
        _executed = None
    if isinstance(_executed, dict) and _executed.get("response"):
        return (
            _chat_desktop_repair._apply_aura_voice_shaping(
                str(_executed.get("response") or "")
            ).strip()
            or final_text,
            str(_executed.get("status") or "desktop_objective"),
        )
    return final_text, status
