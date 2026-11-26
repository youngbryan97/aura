"""How a page fetched for a turn is bound to the answer contract.

Lifted whole out of `response_generation_unitary`, which imports them straight back: every
caller and every patch that names them there still finds them. What they
take from that module is imported at CALL time, for the same reason.
"""
from __future__ import annotations

from typing import Any


def _bind_fetched_content_to_the_contract(
    *,
    auto_browse_urls: Any,
    contract: Any,
    fetched_content_parts: Any,
    is_user_facing: Any,
    new_state: Any,
    objective: Any,
    self: Any,
) -> Any:
    """Bind fetched page content into the turn's contract.

    Moved out of ``UnitaryResponsePhase.execute`` by tools/extract_seam.py, which
    checks the body against the original token for token before
    writing. It reads 6 name(s) from the turn and hands back
    1.
    """
    from .response_generation_unitary import (
        _RESPONSE_RECOVERABLE_ERRORS,
        _record_response_degradation,
        build_response_contract,
        get_task_tracker,
        logger,
        stamp_grounding,
    )

    if fetched_content_parts:
        # Inject fetched content into working memory as a grounded context message
        fetched_block = "\n\n---\n\n".join(fetched_content_parts)
        new_state.cognition.working_memory.append(
            # Stamped, so the inference gate can tell evidence THIS
            # runtime gathered from text that merely looks like it.
            stamp_grounding(
                {
                    "role": "system",
                    "content": f"[FETCHED PAGE CONTENT]\n{fetched_block}",
                    "metadata": {
                        "type": "skill_result",
                        "skill": "sovereign_browser",
                        "ok": True,
                    },
                }
            )
        )
        # Also inject as a skill modifier so the LLM system prompt can reference it
        new_state.response_modifiers["last_skill_run"] = "sovereign_browser"
        new_state.response_modifiers["last_skill_ok"] = True
        new_state.response_modifiers["last_skill_turn_marker"] = new_state.response_modifiers.get("evidence_turn_marker")
        new_state.response_modifiers["last_skill_objective_hash"] = (
            self._objective_fingerprint(objective)
        )
        new_state.response_modifiers["last_skill_result_payload"] = {
            "ok": True,
            "content": fetched_block[:250000],
            "title": fetched_content_parts[0].split("\n")[0]
            if fetched_content_parts
            else "",
            "source": str(auto_browse_urls[0])[:1200]
            if auto_browse_urls
            else "",
        }
        # Rebuild contract now that tool evidence is available
        contract = build_response_contract(
            new_state, objective, is_user_facing=is_user_facing
        )
        new_state.response_modifiers["response_contract"] = contract.to_dict()

        # ── Background Knowledge Formalization ────────────────
        # Fire-and-forget: distill fetched content into the
        # KnowledgeGraph without blocking the user response.
        try:
            from core.learning.formalizer import formalize_content

            page_title = (
                fetched_content_parts[0].split("\n")[0] if fetched_content_parts else ""
            )
            page_url = str(auto_browse_urls[0]) if auto_browse_urls else ""
            get_task_tracker().create_task(
                formalize_content(
                    content=fetched_block[:60000],
                    source_title=page_title,
                    source_url=page_url,
                )
            )
            logger.info(
                "Background formalization task spawned for '%s'", page_title[:60]
            )
        except _RESPONSE_RECOVERABLE_ERRORS as formal_exc:
            _record_response_degradation(
                formal_exc,
                "UnitaryResponse: formalization task spawn skipped: %s",
                action="returned grounded page response without background formalization task",
            )
            logger.debug("Formalization task spawn skipped: %s", formal_exc)
    return contract


