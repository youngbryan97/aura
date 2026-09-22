"""Response Generation Phase for Aura's Cognitive Pipeline."""

import asyncio
import json
import logging
import re
import time
from typing import Any

from core.brain.generation_provenance import (  # noqa: F401  (read at call time by the lifted module)
    attributed_text,
    generation_metadata_of,
)
from core.brain.live_mind_contract import (
    append_text_mutation,  # noqa: F401  (read at call time by the lifted module)
    merge_text_mutations,  # noqa: F401  (read at call time by the lifted module)
    normalize_live_mind_surface_control_receipt,  # noqa: F401  (read at call time by the lifted module)
    summarize_text_mutation_authorship,  # noqa: F401  (read at call time by the lifted module)
)
from core.brain.llm.context_assembler import ContextAssembler
from core.brain.llm.latent_cortex.output_quality import evaluate_latent_output
from core.brain.reasoning_amplifier_flags import reasoning_amplifier_v2_enabled
from core.brain.request_contract import (
    project_user_surface_resume_capability,  # noqa: F401  (read at call time by the lifted module)
)
from core.container import ServiceContainer
from core.conversation.response_reliability import (
    assess_user_facing_reply,
    conversation_reliability_system_block,
    repair_generic_assistant_language,
    repair_instruction_shape,
    requested_output_contract,
)
from core.conversation.surface_disposition import COMPLETION_REASONS
from core.phases.dialogue_policy import (
    enforce_dialogue_contract,
    validate_dialogue_response,
)
from core.phases.executive_guard import get_executive_guard
from core.phases.response_contract import build_response_contract
from core.runtime import background_policy, response_policy
from core.runtime.connectivity import render_connectivity_prompt_block
from core.runtime.conversation_support import (
    schedule_conversation_support_updates,  # noqa: F401  (read at call time by the lifted module)
)
from core.runtime.desktop_task_contract import (
    desktop_task_action_sentence,  # noqa: F401  (read at call time by the lifted module)
)
from core.runtime.errors import record_degradation
from core.runtime.flags import env_present
from core.runtime.structured_input import answer_surface_token_floor
from core.synthesis import stabilize_user_facing_response, strip_meta_commentary
from core.utils.completed_capability import (
    completed_capabilities,  # noqa: F401  (read at call time by the lifted module)
)

from ..state.aura_state import (  # noqa: F401  (read at call time by the lifted module)
    AuraState,
    CognitiveMode,
)
from . import BasePhase
from .response_generation_steps import _RunsTheGenerationSteps
from .response_required_search import _RunsTheRequiredSearch

#: Returned by an extracted block that did NOT return early. A unique
#: object, so no value a block legitimately returns can be mistaken for it.
_SEAM_FELL_THROUGH = object()

#: Enough of a rejected draft to judge the rejection by.
_REJECTED_DRAFT_LOG_CHARS = 400


def _append_only_continuation_pending(
    generation_metadata: Any,
    *,
    clean_user_surface_contract: bool,
) -> bool:
    """Whether the exact draft must remain untouched for route continuation.

    A worker receipt that marks a clean user-surface answer incomplete assigns
    the next model call to the route's append-only continuation owner. Any
    response rewrite before that owner runs invalidates the relationship
    between the visible prefix and the saved KV state.
    """

    if not clean_user_surface_contract or not isinstance(
        generation_metadata, dict
    ):
        return False
    receipt = generation_metadata.get("surface_control_receipt")
    if not isinstance(receipt, dict):
        receipt = {}
    parent_completion_reasons = {
        str(reason or "").strip().lower()
        for reason in (
            generation_metadata.get("post_generation_completion_evidence")
            or generation_metadata.get("failure_reasons")
            or ()
        )
        if str(reason or "").strip()
    } & COMPLETION_REASONS
    if (
        receipt.get("semantic_completion_incomplete") is not True
        and not parent_completion_reasons
    ):
        return False
    reasons = {
        str(reason or "").strip().lower()
        for reason in receipt.get("surface_quality_gate_reasons") or ()
        if str(reason or "").strip()
    }
    reasons.update(parent_completion_reasons)
    try:
        from core.conversation.surface_disposition import draft_is_servable

        return draft_is_servable(reasons)
    except (ImportError, RuntimeError, TypeError, ValueError):
        # Receipt custody is stronger than an unavailable advisory classifier.
        # The route still performs the complete post-continuation assessment.
        return True


def _resolve_request_generation_metadata(
    *,
    sink: Any,
    task_snapshot: Any,
    attributed: Any,
    latent_trace: Any,
) -> dict[str, Any]:
    """Merge request evidence in publication order instead of picking a carrier.

    The mutable sink can be populated when the worker first publishes its
    receipt.  The inference gate may then amend the task-local receipt after
    inspecting the decoded text (for example, when it discovers a clipped
    tail).  Choosing the first non-empty carrier resurrects the older receipt
    and can reopen a second full generation.  Later request-local evidence
    therefore overrides earlier transport snapshots; metadata attributed to
    the returned text remains the final authority for those exact bytes.
    """

    resolved: dict[str, Any] = {}
    for candidate in (sink, task_snapshot, attributed, latent_trace):
        if isinstance(candidate, dict):
            resolved.update(candidate)
    return resolved


def _dialogue_mutation_provenance(
    before: Any,
    after: Any,
    *,
    retry_attempted: bool,
    selected_source: str = "",
) -> dict[str, Any]:
    """Describe the selected text, not merely the work attempted around it."""

    before_text = str(before or "")
    after_text = str(after or "")
    declared_source = str(selected_source or "").strip()
    if declared_source in {
        "incumbent",
        "suppressed",
        "model_retry",
        "deterministic_repair",
    }:
        selected_source = declared_source
    elif after_text == before_text:
        selected_source = "incumbent"
    elif not after_text:
        selected_source = "suppressed"
    elif retry_attempted:
        selected_source = "model_retry"
    else:
        selected_source = "deterministic_repair"
    model_replaced = selected_source == "model_retry"
    runtime_suppressed = selected_source == "suppressed"
    return {
        "stage": (
            "response_generation.dialogue_contract_retry"
            if model_replaced
            else "response_generation.dialogue_contract_suppression"
            if runtime_suppressed
            else "response_generation.dialogue_contract_repair"
        ),
        "method": (
            "model_dialogue_replacement"
            if model_replaced
            else "dialogue_contract_suppression"
            if runtime_suppressed
            else "deterministic_dialogue_repair"
        ),
        "deterministic": not model_replaced,
        "authorship_effect": (
            "replaced_by_model"
            if model_replaced
            else "replaced_by_runtime"
            if runtime_suppressed
            else "preserved"
        ),
        "model_replaced": model_replaced,
        "selected_source": selected_source,
    }

#: What a skill result may carry into the prompt.
#:
#: LIVE, 2026-08-20. http_request returned {ok, url, status, text, ...}. The
#: sanitizer kept `url` and dropped `text`, and the renderer showed the one it
#: was given — so she was handed proof that a fetch had happened with no trace
#: of what it said, and answered 10.5, then 12.4, against a real 11.9. Nothing
#: was hallucinating; there was nothing to read.
_EVIDENCE_SCALAR_KEYS: tuple[str, ...] = (
    "ok",
    "query",
    "answer",
    "summary",
    "message",
    "source",
    "url",
    "title",
    "provenance",
    "offline_fallback",
    "web_error",
    "confidence",
    "count",
    "mode",
    "status",
    "text",
    "body",
)

#: The order they are shown in, which puts the result after its address.
_EVIDENCE_RENDERED_KEYS: tuple[str, ...] = (
    "query",
    "answer",
    "summary",
    "message",
    "title",
    "source",
    "url",
    "content",
    "text",
    "body",
)

logger = logging.getLogger(__name__)

# Explicit tool compositions (e.g. composing an outbound message to ANOTHER AI
# in a web-interlocutor conversation) are non-user-facing, so they correctly
# skip the user-facing reply gates — but they are NOT autonomous background
# chatter either. They were explicitly requested and MUST produce real cortex
# output, so they are exempt from the idle/background suppression that protects
# against autonomous thought leaking. This is what lets her actually THINK a
# reply to ChatGPT instead of falling back to a canned default.
_EXPLICIT_TOOL_COMPOSITION_ORIGINS = frozenset({"web_interlocutor"})

_DOWNSTREAM_REPAIRABLE_RESPONSE_REASONS = {
    # One invented tool name in an otherwise grounded reply. The repair pass
    # regenerates against the same evidence, which already named the real
    # ones, so throwing the whole answer away costs more than it protects.
    "unregistered_capability_claim",
    "missing_requested_self_process_coverage",
    "missing_requested_paragraph_count",
    "missing_requested_list_count",
    "missing_requested_followup_question",
    "off_topic_self_reflection_reply",
    "pseudo_internal_jargon",
    "status_page_self_reflection",
}
_LOCAL_REPAIRABLE_RESPONSE_REASONS = _DOWNSTREAM_REPAIRABLE_RESPONSE_REASONS | {
    "generic_assistant_language",
}
#: Words shared by almost any two English sentences, so their overlap proves
#: no relationship between a search query and the question that produced it.
_QUERY_PROVENANCE_STOPWORDS = frozenset(
    {
        "about", "actually", "and", "any", "anything", "are", "ask", "because",
        "been", "but", "can", "could", "did", "does", "doing", "for", "from",
        "get", "give", "had", "has", "have", "her", "him", "his", "how", "its",
        "just", "know", "like", "look", "make", "many", "may", "more", "most",
        "much", "need", "not", "now", "one", "only", "other", "out", "over",
        "please", "really", "say", "see", "should", "some", "something",
        "still", "such", "take", "tell", "than", "that", "the", "their",
        "them", "then", "there", "these", "they", "thing", "think", "this",
        "those", "through", "too", "use", "very", "want", "was", "way", "well",
        "were", "what", "when", "where", "which", "who", "why", "will", "with",
        "would", "you", "your",
    }
)

_TOOL_FALSE_INABILITY_RE = re.compile(
    r"\b(?:"
    r"i\s+(?:can't|cannot|can\s+not|am\s+unable\s+to|don't\s+have\s+access\s+to|"
    r"do\s+not\s+have\s+access\s+to|lack(?:\s+the)?\s+ability\s+to|can't\s+directly)|"
    r"i'm\s+unable\s+to|i\s+am\s+unable\s+to"
    r")\b[^.\n]{0,180}\b(?:"
    r"browse|search|web|internet|look\s+up|fetch|access|open|visit"
    r")\b",
    re.IGNORECASE,
)
_SEARCH_SKILL_NAMES = {
    "free_search",
    "grounded_search",
    "search_web",
    "web_search",
}

_SOURCE_DEFINITION_TAIL_RE = re.compile(
    r"\b(?:what|who|where|when|how)\s+"
    r"(?:the\s+)?(?:source|page|site|article|[A-Z][A-Za-z0-9 .&'_-]{1,80})\s+"
    r"(?:says?|calls?|defines?|describes?|explains?)\s+"
    r"[^.?!;]{1,180}",
    re.IGNORECASE,
)


def _record_response_generation_degradation(
    error: BaseException,
    *,
    action: str,
    severity: str = "warning",
) -> None:
    record_degradation("response_generation", error, severity=severity, action=action)


def _judge_the_latent_quality(
    *,
    cleaned_response: Any,
    final_latent_quality: Any,
    latent_trace: Any,
    objective: Any,
    runtime_context: Any,
    state: Any,
) -> tuple[Any, Any]:
    """Judge the latent pass's quality when it reported success.

    Moved out of ``ResponseGenerationPhase.execute`` by tools/extract_seam.py, which checks
    the body against the original token for token before writing. The
    block returns early, so it sits in a nested function and _SEAM_FELL_THROUGH
    means it finished instead. It reads 6 name(s) and hands back
    1.
    """
    def _block() -> Any:
        nonlocal final_latent_quality
        if latent_trace.get("latent_cortex_succeeded") is True:
            latent_receipt = latent_trace.get("latent_cortex_receipt")
            latent_receipt = (
                dict(latent_receipt)
                if isinstance(latent_receipt, dict)
                else {}
            )
            final_latent_quality = evaluate_latent_output(
                cleaned_response,
                generated_tokens=latent_receipt.get(
                    "decode_generated_tokens"
                ),
                termination=latent_receipt.get("decode_termination"),
                objective=(
                    str(runtime_context.get("visible_user_message") or "").strip()
                    or str(objective or "").strip()
                ),
            )
            raw_quality = latent_receipt.get("output_quality")
            raw_quality = (
                dict(raw_quality) if isinstance(raw_quality, dict) else {}
            )
            latent_trace["latent_cortex_final_output_quality"] = dict(
                final_latent_quality
            )
            latent_trace[
                "latent_cortex_raw_final_quality_hash_match"
            ] = bool(
                raw_quality.get("text_sha256")
                == final_latent_quality.get("text_sha256")
            )
            if final_latent_quality.get("passed") is not True:
                failure_reasons = list(
                    final_latent_quality.get("reasons") or ["unknown"]
                )
                failure_reason = (
                    "final_output_quality_failed:" + ",".join(
                        str(reason) for reason in failure_reasons
                    )
                )
                latent_trace.update(
                    {
                        "latent_cortex_succeeded": False,
                        "latent_cortex_fallback_used": True,
                        "latent_cortex_failure_reason": failure_reason,
                    }
                )
                state.response_modifiers.update(latent_trace)
                state.response_modifiers.update(
                    {
                        "model_retry_suppressed": True,
                        "generation_failure_class": failure_reason[:120],
                        "response_path": (
                            "cognitive_engine_latent_owner_exhausted"
                        ),
                    }
                )
                record_degradation(
                    "latent_cortex.final_output_quality",
                    RuntimeError(failure_reason),
                    action=(
                        "discarded transformed latent text and refused a second model owner"
                    ),
                    severity="degraded",
                )
                logger.error(
                    "Recursive Latent Cortex final visible text failed its "
                    "hash-bound quality contract (%s); no second generation.",
                    ",".join(str(reason) for reason in failure_reasons),
                )
                return state
        return _SEAM_FELL_THROUGH

    _seam_early_response = _block()
    return _seam_early_response, final_latent_quality


async def _serve_the_cached_generation(
    *,
    generation_metadata: Any,
    latent_outcome: Any,
    latent_trace: Any,
    live_mind_controls_bound: Any,
    live_mind_generation_controls: Any,
    response_text: Any,
    self: Any,
    state: Any,
    token_budget: Any,
    visible_output_contract_payload: Any,
) -> tuple[Any, Any, Any]:
    """Serve a cached generation when this turn already has one.

    Moved out of ``ResponseGenerationPhase.execute`` by tools/extract_seam.py, which checks
    the body against the original token for token before writing. The
    block returns early, so it sits in a nested function and _SEAM_FELL_THROUGH
    means it finished instead. It reads 10 name(s) and hands back
    2.
    """
    async def _block() -> Any:
        nonlocal generation_metadata, response_text
        if latent_outcome.answer_available:
            response_text = latent_outcome.text
            latent_receipt = dict(
                latent_trace.get("latent_cortex_receipt") or {}
            )
            if not latent_outcome.succeeded:
                failure_reason = str(
                    latent_trace.get("latent_cortex_failure_reason")
                    or "latent_episode_failed"
                )
                state.response_modifiers.update(
                    {
                        "model_retry_suppressed": True,
                        "generation_failure_class": failure_reason[:120],
                    }
                )
            generation_metadata = {
                **latent_trace,
                "model_retry_suppressed": bool(
                    not latent_outcome.succeeded
                ),
                "surface_control_receipt": (
                    self._latent_cortex_surface_receipt(
                        latent_receipt,
                        controls_bound=live_mind_controls_bound,
                        generation_controls=live_mind_generation_controls,
                        token_budget=token_budget,
                        requested_output_contract=(
                            visible_output_contract_payload
                        ),
                    )
                ),
            }
        elif (
            latent_outcome.attempted
            and not latent_outcome.fallback_allowed
        ):
            failure_reason = str(
                latent_trace.get("latent_cortex_failure_reason")
                or "latent_owner_exhausted"
            )
            state.response_modifiers.update(
                {
                    "model_retry_suppressed": True,
                    "generation_failure_class": failure_reason[:120],
                    "response_path": (
                        "cognitive_engine_latent_owner_exhausted"
                    ),
                }
            )
            logger.error(
                "Recursive Latent Cortex exhausted the single resident "
                "owner (%s); refusing a colliding ordinary generation.",
                failure_reason,
            )
            return state
        return _SEAM_FELL_THROUGH

    _seam_early_response = await _block()
    return _seam_early_response, generation_metadata, response_text


def _the_amplifier_stood_down(draft, reason: str):
    """Say which gate declined, and hand the draft back untouched.

    The verifier-backed amplifier is the thing that re-checks a stated
    calculation against exact arithmetic. It has eight ways to decline and
    seven of them said nothing, so "it did not run" and "it ran and found
    nothing" looked identical from outside — and the runtime's own claim
    register has been reporting `recurrent_answer_enters_verified_complete_
    engine` as NEVER RUN, in the neural feed, the whole time.

    LIVE, 2026-09-08: asked for the daylight lost at 45°N between the solstice
    and the equinox, Aura wrote the hour-angle formula correctly, then read
    15h 08m off it where the formula gives 15h 26m, and answered "about 190
    minutes" against a true 206. Nothing recomputed the step. The amplifier
    classifies that question as `math` and had not run.
    """

    logger.info(
        "Reasoning amplifier stood down (%s); serving the draft as written.", reason
    )
    return draft




class ResponseGenerationPhase(_RunsTheGenerationSteps, _RunsTheRequiredSearch, BasePhase):
    """
    Phase 5: Response Generation.
    Constructs the prompt from the current state (identity, affect, memories)
    and invokes the LLM to generate Aura's response.
    """

    def __init__(self, container: Any):
        self.container = container
        self._last_reasoning_receipt: dict[str, Any] | None = None

    @staticmethod
    def _compact_prompt_payload(value: Any, *, limit: int = 3000) -> str:
        """Render runtime grounding payloads without letting them dominate the prompt."""

        try:
            rendered = json.dumps(value, ensure_ascii=True, sort_keys=True, default=str)
        except (TypeError, ValueError, OverflowError):
            rendered = str(value)
        rendered = rendered.strip()
        if len(rendered) <= limit:
            return rendered
        return f"{rendered[: max(0, limit - 16)]}...[truncated]"

    @classmethod
    def _append_system_block(
        cls,
        messages: list[dict[str, Any]],
        title: str,
        body: str,
    ) -> None:
        body = str(body or "").strip()
        if not body:
            return
        block = f"## {title}\n{body}"
        if messages and str(messages[0].get("role", "") or "").strip().lower() == "system":
            messages[0]["content"] = f"{str(messages[0].get('content', '')).rstrip()}\n\n{block}"
        else:
            messages.insert(0, {"role": "system", "content": block})

    @staticmethod
    def _sanitize_grounding_payload(payload: Any) -> dict[str, Any]:
        """Keep tool evidence useful without flooding prompts or memory."""

        if not isinstance(payload, dict):
            return {"ok": False, "result": str(payload or "")[:1200]}

        compact: dict[str, Any] = {}
        for key in _EVIDENCE_SCALAR_KEYS:
            if key not in payload:
                continue
            value = payload.get(key)
            if isinstance(value, str):
                compact[key] = value[:4000]
            elif isinstance(value, (bool, int, float)) or value is None:
                compact[key] = value
            else:
                compact[key] = str(value)[:1000]

        def _compact_items(items: Any, *, limit: int = 5) -> list[dict[str, Any]]:
            compact_items: list[dict[str, Any]] = []
            if not isinstance(items, list):
                return compact_items
            for item in items[:limit]:
                if not isinstance(item, dict):
                    continue
                compact_items.append(
                    {
                        "title": str(item.get("title") or "")[:300],
                        "url": str(item.get("url") or item.get("source") or "")[:500],
                        "snippet": str(
                            item.get("snippet")
                            or item.get("content")
                            or item.get("text")
                            or ""
                        )[:1200],
                    }
                )
            return [item for item in compact_items if any(item.values())]

        results = _compact_items(payload.get("results"), limit=5)
        if results:
            compact["results"] = results
        citations = _compact_items(payload.get("citations"), limit=5)
        if citations:
            compact["citations"] = citations
        chunks = _compact_items(payload.get("chunks"), limit=3)
        if chunks:
            compact["chunks"] = chunks

        content = str(payload.get("content") or payload.get("result") or "").strip()
        if content and "content" not in compact:
            compact["content"] = content[:6000]
        return compact

    # The keys a skill result may carry through to the prompt. Written once:
    # the sanitizer and the renderer both read it, and when only the renderer
    # learned about "text" the sanitizer had already dropped the body.


    @classmethod
    def _render_skill_result_block(
        cls,
        *,
        skill_name: str,
        payload: dict[str, Any],
    ) -> str:
        status = "✅" if payload.get("ok") else "⚠️"
        parts: list[str] = []
        # "text" and "body" are where a fetched document actually lands.
        #
        # LIVE, 2026-08-20. http_request returned {ok, url, status, text, ...}
        # and this rendered "Url: https://…" and stopped, because `url` was a
        # recognised key and the body was not. She was handed proof that a
        # fetch had happened with no trace of what it said, and answered
        # 10.5, then 12.4, against a real 11.9 — inventing, because there was
        # nothing to read. One recognised key was suppressing the result.
        for key in _EVIDENCE_RENDERED_KEYS:
            value = str(payload.get(key) or "").strip()
            if value:
                label = key.replace("_", " ").title()
                parts.append(f"{label}: {value[:1600]}")
        results = payload.get("results")
        if isinstance(results, list) and results:
            rendered_results = []
            for idx, item in enumerate(results[:5], start=1):
                if not isinstance(item, dict):
                    continue
                title = str(item.get("title") or "").strip()
                url = str(item.get("url") or "").strip()
                snippet = str(item.get("snippet") or "").strip()
                rendered_results.append(
                    f"{idx}. {title or 'Untitled'}"
                    + (f" — {url}" if url else "")
                    + (f" — {snippet[:500]}" if snippet else "")
                )
            if rendered_results:
                parts.append("Results:\n" + "\n".join(rendered_results))
        if not parts:
            parts.append(json.dumps(payload, ensure_ascii=True, default=str)[:2400])
        return f"[SKILL RESULT: {skill_name}] {status} " + "\n".join(parts)

    @staticmethod
    def _first_sentence(text: str, *, fallback: str = "") -> str:
        cleaned = " ".join(str(text or "").strip().split())
        if not cleaned:
            return fallback
        match = re.search(r"(.+?[.!?])(?:\s|$)", cleaned)
        return (match.group(1) if match else cleaned).strip()




    def _capability_engine(self) -> Any:
        """The engine that runs skills, or None when it is not up yet."""
        cap = self.container.get("capability_engine", default=None)
        if cap is None:
            try:
                cap = ServiceContainer.get("capability_engine", default=None)
            # not a failure: no capability engine in the container, and the caller checks for
            # None below.
            except (AttributeError, RuntimeError, TypeError, ValueError):
                cap = None
        return cap if cap is not None and hasattr(cap, "execute") else None






    @classmethod
    def _inject_live_runtime_grounding(
        cls,
        messages: list[dict[str, Any]],
        runtime_context: dict[str, Any],
    ) -> None:
        """Make live desktop mind/body/tool context visible to the full phase path.

        The compact desktop router path already receives these fields directly.
        The full response-generation phase must receive them too, otherwise a
        required desktop turn can be technically routed through CognitiveEngine
        while the model only sees a generic prompt.
        """

        live_mind = runtime_context.get("live_mind_context")
        if isinstance(live_mind, dict) and live_mind:
            compact_mind = {
                "required_for_live_desktop": live_mind.get("required_for_live_desktop"),
                "must_answer_from_full_mind_path": live_mind.get("must_answer_from_full_mind_path"),
                "required_subsystems_ok": live_mind.get("required_subsystems_ok"),
                "required_subsystems": live_mind.get("required_subsystems"),
                "lane": live_mind.get("lane"),
                "voice": live_mind.get("voice"),
                "substrate": live_mind.get("substrate"),
                "timescale_reconciliation": live_mind.get("timescale_reconciliation"),
                "governance": live_mind.get("governance"),
            }
            contract = str(runtime_context.get("mind_context_contract") or "").strip()
            timescale_block = ""
            timescale = live_mind.get("timescale_reconciliation")
            if isinstance(timescale, dict) and timescale:
                try:
                    from core.runtime.timescale_bridge import render_timescale_prompt_block

                    timescale_block = render_timescale_prompt_block(timescale)
                except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
                    _record_response_generation_degradation(
                        exc,
                        action="continued response generation without timescale reconciliation block",
                    )
            cls._append_system_block(
                messages,
                "LIVE MIND CONTEXT",
                (
                    f"{cls._compact_prompt_payload(compact_mind, limit=3200)}\n"
                    "This is causal grounding for the reply, not text to recite. "
                    "Use memory, current state, substrate, governance, and the live lane as one context. "
                    "Do not answer as a generic assistant persona."
                    + (f"\n{timescale_block}" if timescale_block else "")
                    + (f"\n{contract}" if contract else "")
                ),
            )

        speech_frame = runtime_context.get("live_speech_grounding_frame")
        if isinstance(speech_frame, dict) and speech_frame:
            compact_frame = {
                key: speech_frame.get(key)
                for key in (
                    "attention_focus",
                    "dominant_action",
                    "dominant_emotions",
                    "interests",
                    "mood",
                    "tone",
                    "requires_explicit_live_grounding",
                )
                if speech_frame.get(key) not in (None, "", [], {})
            }
            if compact_frame:
                cls._append_system_block(
                    messages,
                    "LIVE SPEECH GROUNDING",
                    (
                        f"{cls._compact_prompt_payload(compact_frame, limit=1200)}\n"
                        "This frame is grounding, not prose to repeat. Convert it into ordinary speech only when it helps."
                    ),
                )

        cognitive_situation = runtime_context.get("cognitive_situation_frame")
        if isinstance(cognitive_situation, dict) and cognitive_situation:
            try:
                from core.brain.cognitive_situation import (
                    render_cognitive_situation_prompt_block,
                )

                situation_block = render_cognitive_situation_prompt_block(
                    cognitive_situation,
                    compact=True,
                )
            except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
                _record_response_generation_degradation(
                    exc,
                    action="continued response generation without cognitive situation block",
                )
                situation_block = ""
            if situation_block:
                cls._append_system_block(
                    messages,
                    "COGNITIVE SITUATION FRAME",
                    situation_block,
                )

        from core.conversation.answer_provenance import AnswerProvenance, provenance_grounding_json
        from core.utils.injected_blocks import RUNTIME_EVIDENCE_ROLE, stamp_grounding

        prior_answer_provenance = AnswerProvenance.from_value(
            runtime_context.get("prior_answer_provenance")
        )
        if prior_answer_provenance is not None:
            evidence_index = next(
                (index for index in range(len(messages) - 1, -1, -1)
                 if messages[index].get("role") == "user"),
                len(messages),
            )
            messages.insert(evidence_index, stamp_grounding({
                "role": RUNTIME_EVIDENCE_ROLE,
                "content": provenance_grounding_json(prior_answer_provenance),
                "metadata": {"type": "prior_answer_provenance"},
            }))

        evidence_blocks = (
            (
                "CONTEXT CHALLENGE EVIDENCE",
                runtime_context.get("contextual_relevance_evidence"),
                "Use this to avoid inventing prior context; answer from the actual recent thread.",
            ),
            (
                "CONVERSATION RECALL EVIDENCE",
                runtime_context.get("conversation_recall_evidence"),
                "Use this as source-of-truth memory for the current recall question.",
            ),
            (
                "DEEP MEMORY RECALL",
                runtime_context.get("deep_memory_context"),
                "Silent background recall from long-term memory; draw on it only where "
                "genuinely relevant, never recite it.",
            ),
            (
                "GOVERNED CAPABILITY INVENTORY EVIDENCE",
                runtime_context.get("grounded_capability_inventory_context"),
                "Use this for capability questions; do not claim execution without receipts.",
            ),
            (
                "EVIDENCE-BOUND SELF-CLAIM EVIDENCE",
                runtime_context.get("evidence_bound_self_claim_context"),
                "Use this to keep consciousness, sentience, and personhood claims bounded by evidence.",
            ),
        )
        for title, payload, instruction in evidence_blocks:
            payload_text = str(payload or "").strip()
            if payload_text:
                cls._append_system_block(
                    messages,
                    title,
                    f"{payload_text[:3000]}\n{instruction}",
                )

    @staticmethod
    def _request_timeout(*, is_background: bool, deep_handoff: bool) -> float:
        if is_background:
            return 10.0
        if deep_handoff:
            return 210.0
        return 180.0

    @classmethod
    def _surface_request_timeout(
        cls,
        *,
        is_background: bool,
        deep_handoff: bool,
        token_budget: int,
    ) -> float:
        request_timeout = cls._request_timeout(
            is_background=is_background,
            deep_handoff=deep_handoff,
        )
        if is_background:
            return request_timeout
        return max(
            request_timeout,
            min(
                response_policy.USER_FACING_COMPLETION_DEADLINE_MAX_S,
                45.0 + (0.34 * max(1, int(token_budget))),
            ),
        )

    @staticmethod
    def _bounded_request_timeout(
        runtime_context: dict[str, Any],
        requested_timeout_s: float,
        *,
        reserve_s: float = 0.0,
    ) -> float:
        """Fit a model operation inside the owning CognitiveEngine deadline."""

        requested = max(0.0, float(requested_timeout_s))
        raw_deadline = runtime_context.get("cognitive_cycle_deadline_monotonic")
        try:
            deadline = float(raw_deadline)
        except (TypeError, ValueError, OverflowError):
            return requested
        if not deadline or not deadline < float("inf"):
            return requested
        available = deadline - time.monotonic() - max(0.0, float(reserve_s))
        return max(0.0, min(requested, available))

    @staticmethod
    def _latent_owner_exhausted(
        reason: str,
        receipt: dict[str, Any],
    ) -> bool:
        """Whether the selected latent path already consumed the model owner."""
        from core.brain.foreground_latent_runtime import latent_owner_exhausted

        return latent_owner_exhausted(reason, receipt)

    @staticmethod
    def _generation_metadata_snapshot(router: Any) -> dict[str, Any]:
        getter = getattr(router, "get_last_generation_metadata", None)
        if not callable(getter):
            return {}
        try:
            metadata = getter()
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            logger.debug("ResponseGeneration could not snapshot generation metadata: %s", exc)
            return {}
        return dict(metadata) if isinstance(metadata, dict) else {}

    @staticmethod
    def _latent_cortex_surface_receipt(
        receipt: dict[str, Any],
        *,
        controls_bound: bool,
        generation_controls: dict[str, Any],
        token_budget: int,
        requested_output_contract: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Translate a validated latent receipt into the live surface contract."""

        expected_alpha = generation_controls.get("clean_user_surface_steering_alpha")
        expected_loops = generation_controls.get("clean_user_surface_recurrent_loops")
        observed_alpha = receipt.get("episode_affective_steering_alpha")
        observed_steps = receipt.get("steps_taken")
        alpha_applied_ok = bool(
            isinstance(expected_alpha, (int, float))
            and not isinstance(expected_alpha, bool)
            and isinstance(observed_alpha, (int, float))
            and not isinstance(observed_alpha, bool)
            and abs(float(expected_alpha) - float(observed_alpha)) <= 1e-6
        )
        recurrence_applied_ok = bool(
            type(expected_loops) is int
            and expected_loops > 0
            and type(observed_steps) is int
            and observed_steps >= expected_loops
        )
        controls_applied = bool(
            controls_bound
            and receipt.get("episode_affective_steering_applied") is True
            and alpha_applied_ok
            and recurrence_applied_ok
        )
        output_quality = receipt.get("output_quality")
        output_quality = (
            dict(output_quality) if isinstance(output_quality, dict) else {}
        )
        output_quality_passed = bool(
            output_quality.get("schema") == "aura.latent_output_quality.v1"
            and output_quality.get("passed") is True
        )
        return {
            "enabled": True,
            "applied": controls_applied,
            "generation_required": True,
            "application_status": (
                "latent_cortex_controls_applied"
                if controls_applied
                else "latent_cortex_controls_unbound"
            ),
            "live_mind_controls_bound": bool(controls_bound),
            "clean_user_surface_contract": True,
            "surface_validation_prompt_present": True,
            "surface_alpha_applied": observed_alpha,
            "surface_alpha_applied_ok": alpha_applied_ok,
            "recurrent_runtime_loops_applied": observed_steps,
            "recurrent_runtime_loops_applied_ok": recurrence_applied_ok,
            "surface_quality_gate_enabled": True,
            "surface_quality_gate_passed": output_quality_passed,
            "surface_quality_gate_attempts": 1,
            "surface_quality_gate_reasons": list(
                output_quality.get("reasons") or []
            ),
            "latent_output_quality": output_quality,
            "generation_max_tokens": int(token_budget),
            "generated_tokens": int(receipt.get("decode_generated_tokens") or 0),
            "decode_temperature_applied": receipt.get("decode_temperature"),
            "decode_top_p_applied": receipt.get("decode_top_p"),
            "requested_output_contract": (
                dict(requested_output_contract)
                if isinstance(requested_output_contract, dict)
                else None
            ),
            "source": "recursive_latent_cortex",
        }

    async def _maybe_amplify_response(
        self,
        *,
        objective: str,
        draft: str,
        router: Any,
        state: AuraState,
        request_timeout: float,
        origin: str,
        tier: str,
        runtime_context: dict[str, Any],
        is_user_facing: bool,
        is_background: bool,
        proof_or_benchmark: bool,
        turn_completed_capabilities: frozenset[str] = frozenset(),
    ) -> str:
        """Run verifier-backed Amplifier v2 on eligible hard turns in the active phase.

        This is intentionally conservative. Action requests stay owned by tool
        dispatch, casual chat stays single-pass, proof lanes are untouched, and
        failures keep the original draft while recording a degradation receipt.
        """

        if not is_user_facing or is_background or proof_or_benchmark or not draft:
            return _the_amplifier_stood_down(draft, "not_a_user_facing_turn")
        if not reasoning_amplifier_v2_enabled():
            return _the_amplifier_stood_down(draft, "amplifier_switched_off")
        try:
            from core.brain.reasoning_amplifier_v2 import amplify_turn, is_amplifiable
        except ImportError as exc:
            _record_response_generation_degradation(
                exc,
                action="continued response generation without Amplifier v2 import",
            )
            return _the_amplifier_stood_down(draft, "amplifier_import_failed")

        task_type = is_amplifiable(objective)
        if task_type is None:
            return _the_amplifier_stood_down(draft, "question_is_not_a_checkable_one")

        # A successful capability receipt assigns this turn to the evidence
        # narration owner. The amplifier does not receive the capability's
        # evidence pack and cannot independently verify current external facts,
        # so a competing generation cannot earn promotion authority. Reopening
        # the turn here also repeats work already completed by the tool lane.
        completed = frozenset(turn_completed_capabilities) | completed_capabilities(
            runtime_context.get("completed_capability_evidence")
        )
        if completed:
            state.response_modifiers["reasoning_amplifier_v2_active_phase"] = {
                "task_type": task_type,
                "verified": False,
                "confidence": 0.0,
                "promotion_authority": "none",
                "adopted": False,
                "admitted": False,
                "admission_reason": "completed_capability_evidence_owned_turn",
                "completed_capabilities": sorted(completed),
            }
            logger.info(
                "Reasoning amplifier kept the capability-grounded draft; "
                "completed turn owner=%s.",
                ",".join(sorted(completed)),
            )
            return draft
        from core.brain.executable_reasoning import should_use_executable_reasoning

        executable_reasoning = should_use_executable_reasoning(
            objective,
            task_type=task_type,
        )

        # Keep the complete engine inside the enclosing CognitiveEngine turn.
        # Structured program generation on the resident 32B needs about 45-55s;
        # do not start it when the remaining phase contract cannot fund one
        # complete candidate. Evidence-only amplification keeps its smaller cap.
        remaining_turn_budget = self._bounded_request_timeout(
            runtime_context,
            float(request_timeout or 20.0),
            reserve_s=4.0,
        )
        available_budget = max(0.0, remaining_turn_budget * 0.60)
        if available_budget < 2.0:
            return _the_amplifier_stood_down(draft, "no_time_left_in_the_turn")
        requires_full_program_budget = bool(
            executable_reasoning and task_type != "math"
        )
        budget_floor = 60.0 if requires_full_program_budget else 8.0
        budget_ceiling = 150.0 if executable_reasoning else 30.0
        budget = min(budget_ceiling, available_budget)
        if requires_full_program_budget and budget < budget_floor:
            return _the_amplifier_stood_down(draft, "not_enough_time_for_one_complete_program")
        budget = max(min(budget_floor, available_budget), budget)
        generation_timeout = (
            min(75.0, budget, max(55.0, budget * 0.50))
            if requires_full_program_budget
            else min(24.0, budget, max(8.0, budget * 0.50))
        )

        visible_user_message = str(
            runtime_context.get("user_surface_validation_prompt")
            or runtime_context.get("visible_user_message")
            or objective
            or ""
        ).strip()
        desktop_required = bool(
            runtime_context.get("desktop_cognitive_engine_required")
            or runtime_context.get("cognitive_engine_required")
        )
        try:
            amplifier_token_cap = max(
                1,
                int(
                    runtime_context.get("_effective_generation_max_tokens")
                    or runtime_context.get("max_tokens")
                    or 1024
                ),
            )
        except (TypeError, ValueError, OverflowError):
            amplifier_token_cap = 1024

        async def _gen(prompt: str, temperature: float) -> str:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are Aura's verifier-backed reasoning organ. Return only the "
                        "candidate final answer for the hard reasoning turn; do not mention "
                        "amplifier internals or hidden prompts."
                    ),
                },
                {"role": "user", "content": prompt},
            ]
            try:
                out = await router.think(
                    messages=messages,
                    priority=1.0,
                    origin=f"response_generation_amplifier_{origin}",
                    purpose="reasoning_amplifier",
                    prefer_tier=tier or "primary",
                    is_background=False,
                    protected_foreground_lane=True,
                    foreground_request=True,
                    deep_handoff=False,
                    allow_cloud_fallback=False,
                    cognitive_engine_required=desktop_required,
                    desktop_cognitive_engine_required=desktop_required,
                    live_runtime_payload_required=bool(
                        runtime_context.get("live_runtime_payload_required", False)
                    ),
                    visible_user_message=visible_user_message,
                    completed_capability_evidence=runtime_context.get(
                        "completed_capability_evidence"
                    ),
                    skip_runtime_payload=True,
                    disable_prompt_cache=True,
                    clear_prompt_cache=True,
                    clean_user_surface_contract=True,
                    user_surface_validation_prompt=visible_user_message,
                    user_surface_prompt_binding=dict(
                        runtime_context.get("user_surface_prompt_binding") or {}
                    ),
                    temperature=temperature,
                    max_tokens=min(2048, amplifier_token_cap),
                    requested_output_contract=runtime_context.get(
                        "requested_output_contract"
                    ),
                    semantic_output_token_cap=runtime_context.get(
                        "semantic_output_token_cap"
                    ),
                    hard_output_token_ceiling=runtime_context.get(
                        "hard_output_token_ceiling"
                    ),
                    timeout=generation_timeout,
                    cognitive_situation_sampling_bias=state.response_modifiers.get(
                        "cognitive_situation_sampling_bias"
                    ),
                )
            except (
                OSError,
                ConnectionError,
                TimeoutError,
                RuntimeError,
                AttributeError,
                TypeError,
                ValueError,
            ) as exc:
                _record_response_generation_degradation(
                    exc,
                    action="kept original draft after Amplifier v2 generate failed",
                )
                return ""
            generation_metadata = self._generation_metadata_snapshot(router)
            if isinstance(out, dict):
                structured_metadata = out.get("generation_metadata") or out.get("metadata")
                if isinstance(structured_metadata, dict):
                    generation_metadata = dict(structured_metadata)
                out = out.get("content") or out.get("response") or ""
            return attributed_text(
                str(out or "").strip(),
                generation_metadata,
            )

        try:
            result = await asyncio.wait_for(
                amplify_turn(
                    objective,
                    _gen,
                    task_type=task_type,
                    time_budget_s=budget,
                    sample_budget=3 if executable_reasoning else None,
                    extra_context={
                        "live_response_phase": True,
                        "require_generation_metadata": True,
                        "disable_batched_candidates": True,
                        "generation_max_tokens": amplifier_token_cap,
                        "seed_candidates": [draft],
                        "enable_executable_reasoning": executable_reasoning,
                        "allow_textual_fallback_after_executable": True,
                        "cognitive_situation_frame": state.response_modifiers.get(
                            "cognitive_situation_frame"
                        ),
                    },
                ),
                timeout=budget,
            )
        except (
            OSError,
            ConnectionError,
            TimeoutError,
            RuntimeError,
            AttributeError,
            TypeError,
            ValueError,
        ) as exc:
            _record_response_generation_degradation(
                exc,
                action="kept original draft after Amplifier v2 failed",
            )
            return draft

        receipt = result.receipt.to_dict()
        self._last_reasoning_receipt = receipt
        state.response_modifiers["reasoning_receipt"] = receipt
        state.response_modifiers["reasoning_amplifier_v2_active_phase"] = {
            "task_type": task_type,
            "verified": bool(result.verified),
            "confidence": float(result.confidence),
            "promotion_authority": str(
                receipt.get("promotion_authority") or "none"
            ),
            "adopted": bool(
                result.answer
                and receipt.get("promotion_authority")
                in {"checked_verifier", "independent_executable_consensus"}
            ),
        }
        authority = str(receipt.get("promotion_authority") or "none")
        logger.info(
            "🧠 [AmplifyV2-active-phase] task=%s authority=%s conf=%.2f -> %s",
            task_type,
            authority,
            result.confidence,
            "adopted"
            if (
                result.answer
                and authority
                in {"checked_verifier", "independent_executable_consensus"}
            )
            else "kept draft",
        )
        promoted_answer = ""
        if authority == "checked_verifier":
            promoted_answer = str(result.answer or "").strip()
        elif authority == "independent_executable_consensus":
            promoted_answer = str(result.source_answer or result.answer or "").strip()
        if promoted_answer:
            amplified_text = attributed_text(
                promoted_answer,
                result.generation_metadata,
            )
            amplified_text.reasoning_source_answer = str(
                getattr(result, "source_answer", "") or promoted_answer
            )
            amplified_text.reasoning_text_mutations = [
                dict(item) for item in getattr(result, "text_mutations", [])
            ]
            return amplified_text
        return draft

    @staticmethod
    def _safe_bias_float(value: Any, default: float = 0.0) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError, OverflowError):
            return default
        return parsed if parsed == parsed else default

    @classmethod
    def _apply_generation_sampling_bias(
        cls,
        *,
        base_temperature: float,
        token_budget: int,
        biases: list[dict[str, Any] | None],
    ) -> tuple[float, int]:
        temperature = cls._safe_bias_float(base_temperature, 0.7)
        tokens = max(128, int(token_budget))
        token_factor = 1.0

        for bias in biases:
            if not isinstance(bias, dict):
                continue
            temperature += max(
                -0.20,
                min(0.20, cls._safe_bias_float(bias.get("temperature_delta"), 0.0)),
            )
            factor = cls._safe_bias_float(bias.get("max_tokens_factor"), 1.0)
            if 0.25 <= factor <= 1.25:
                token_factor *= factor

        temperature = max(0.10, min(1.15, temperature))
        tokens = max(128, min(8192, int(tokens * token_factor)))
        return temperature, tokens

    @staticmethod
    def _repair_substantive_instruction_shape_miss(
        objective: Any,
        response_text: Any,
    ) -> tuple[str, bool, tuple[str, ...]]:
        """Repair explicit shape misses locally when the content is already substantive."""
        response_text_s = str(response_text or "").strip()
        reliability = assess_user_facing_reply(str(objective or ""), response_text_s)
        reasons = tuple(reliability.reasons or ())
        output_contract = requested_output_contract(objective)
        if output_contract.exact_reply and reasons == ("missing_requested_exact_reply",):
            repaired = repair_instruction_shape(objective, response_text_s)
            repaired_assessment = assess_user_facing_reply(str(objective or ""), repaired)
            if repaired != response_text_s and repaired_assessment.ok:
                return repaired, True, reasons
        if len(response_text_s) < 48 or len(response_text_s.split()) < 8:
            return response_text_s, False, reasons

        reason_set = set(reasons)
        if (
            not reliability.retryable
            or not reason_set
            or not reason_set.issubset(_LOCAL_REPAIRABLE_RESPONSE_REASONS)
        ):
            return response_text_s, False, reasons

        repaired = response_text_s
        if "generic_assistant_language" in reason_set:
            repaired = repair_generic_assistant_language(objective, repaired)
        repaired = repair_instruction_shape(objective, repaired)
        if repaired == response_text_s:
            return response_text_s, False, reasons
        repaired_assessment = assess_user_facing_reply(str(objective or ""), repaired)
        if repaired_assessment.ok:
            return repaired, True, reasons
        return response_text_s, False, reasons

    async def execute(self, state: AuraState, objective: str | None = None, **kwargs) -> AuraState:
        """
        Build the LLM prompt from state and generate Aura's response.

        Assembles the message list via ContextAssembler, injects optional causal-world
        and skill-result context, calls the LLM router with affect-modulated parameters,
        runs the ExecutiveGuard alignment pass, and appends the cleaned response to
        working memory.  Suppressed when the CognitiveIntegrationLayer (Phase 7) is
        active for user-facing origins.
        """
        # 1. Use targeted objective from state rather than guessing via working_memory[-1]
        objective = state.cognition.current_objective
        origin = background_policy.normalize_origin(state.cognition.current_origin) or "system"
        state.cognition.current_origin = origin

        is_test_run = (
            origin == "test"
            or env_present("AURA_AGI_MAX_TASKS", description="AGI battery task cap; presence marks a battery run", owner="core.runtime")
            or env_present("AURA_TESTING", description="presence marks a test run", owner="core.runtime")
            or env_present("AURA_PROOF_RUN", description="presence marks a proof run", owner="core.runtime")
        )

        if not objective:
            logger.debug("⏭️ ResponseGeneration: No active objective, skipping.")
            return state

        # PHASE 7 SUPPRESSION: If Advanced Cognition (CognitiveIntegrationLayer) is
        # actively handling this same turn, Phase 5 MUST NOT fire. The older broad
        # ``cog.is_active`` check created a response dead zone for direct
        # CognitiveEngine callers: Phase 5 stood down even though Phase 7 was not
        # processing the turn. Suppression is now tied to active ownership only.
        cog = self.container.get("cognitive_integration", default=None)
        if (
            cog
            and getattr(cog, "_processing_turn", False)
            and background_policy.is_user_facing_origin(origin)
        ):
            logger.debug(
                "🛡️ ResponseGeneration: Phase 7 owns this turn — Phase 5 SUPPRESSED for %s.",
                origin,
            )
            return state
        # Also suppress if Phase 7 is currently mid-processing (race condition guard)
        if cog and getattr(cog, "_processing_turn", False):
            logger.debug("🛡️ ResponseGeneration: Phase 7 mid-processing — Phase 5 SUPPRESSED.")
            return state

        logger.info(
            "💭 ResponseGeneration: Generating response for objective: %s... (%s)",
            str(objective)[:30],
            state.cognition.current_mode.value,
        )

        response_mutation_receipt: dict[str, Any] = {"text_mutations": []}
        generation_metadata: dict[str, Any] = {}
        # Recovery branches can bypass the ordinary/latent generation join
        # below (for example, a verified tool result after model timeout).
        # Initialize publication ownership before any awaited work so final
        # validation never inherits an undefined or stale owner.
        latent_response_owned = False
        # These are captured before and immediately after the resident model
        # boundary. Optional postprocessors may consume the mutable request
        # context or collide with the enclosing deadline; neither event may
        # erase authority that this turn already established or text that the
        # integrity-checked model boundary already returned.
        turn_completed_capabilities: frozenset[str] = frozenset()
        servable_incumbent: Any = None
        try:
            _speech_profile, _sve = self._execute_substrate_voice_compile(objective, origin, state)

            is_background = not background_policy.is_user_facing_origin(origin)
            foreground_user_surface_owned = bool(not is_background and not is_test_run)
            explicit_tool_composition = origin in _EXPLICIT_TOOL_COMPOSITION_ORIGINS
            if is_background and not is_test_run and not explicit_tool_composition:
                try:
                    orchestrator = self.container.get("orchestrator", default=None)
                    reason = response_policy.background_response_suppression_reason(
                        objective,
                        orchestrator=orchestrator,
                        include_synthetic_noise=True,
                    )
                    if reason:
                        logger.info(
                            "🛡️ ResponseGeneration: suppressing background objective for origin=%s (%s).",
                            origin,
                            reason,
                        )
                        # Deferred, not failed: the loop reads this before it
                        # charges the objective with friction. LIVE 2026-09-16:
                        # the journal, held back every five minutes while chat
                        # turns ran, was booked as "repeatedly unresolved" until
                        # the resilience engine read depletion=1.00 off it.
                        state.response_modifiers["background_suppression"] = str(reason)
                        return state
                except (OSError, ConnectionError, TimeoutError) as exc:
                    _record_response_generation_degradation(
                        exc,
                        action="suppressed background response after background policy check failed",
                        severity="error",
                    )
                    logger.error(
                        "ResponseGeneration background policy check failed: %s", exc, exc_info=True
                    )
                    return state

            # 2. Build structured messages purely from State via ContextAssembler
            strict_answer_request = "<answer>" in objective.lower() or "answer_format" in kwargs
            proof_answer_run = bool(
                strict_answer_request
                and (
                    origin == "test"
                    or env_present("AURA_AGI_MAX_TASKS", description="AGI battery task cap; presence marks a battery run", owner="core.runtime")
                    or env_present("AURA_TESTING", description="presence marks a test run", owner="core.runtime")
                    or env_present("AURA_PROOF_RUN", description="presence marks a proof run", owner="core.runtime")
                )
            )
            if proof_answer_run:
                messages = [
                    {
                        "role": "system",
                        "content": (
                            "You are Aura's governed proof-answer lane. Solve the user's task. "
                            "Output the final answer strictly inside <answer>...</answer> tags. "
                            "Keep the tag content minimal and do not include chat filler."
                        ),
                    },
                    {"role": "user", "content": objective},
                ]
            else:
                from core.brain.cognitive_context_manager import (
                    bind_unified_context_to_state,
                )

                def _assemble(
                    history: list[dict[str, Any]] | None,
                    note: str | None,
                ) -> list[dict[str, str]]:
                    if history is None:
                        return ContextAssembler.build_messages(state, objective)
                    return ContextAssembler.build_messages(
                        state,
                        objective,
                        conversation_history=history,
                        history_note=note,
                    )

                await bind_unified_context_to_state(state, objective)
                runtime_context = kwargs.get("context")
                if not isinstance(runtime_context, dict):
                    runtime_context = {}
                delivered_history = None
                delivered_note = None
                if (
                    not is_background
                    and not is_test_run
                    and "recent_completed_exchanges" in runtime_context
                ):
                    from core.conversation.delivered_history import (
                        delivered_exchange_messages,
                    )

                    delivered_history = delivered_exchange_messages(
                        runtime_context.get("recent_completed_exchanges"),
                    )
                messages = _assemble(delivered_history, delivered_note)
            contract = build_response_contract(
                state,
                objective,
                is_user_facing=not is_background and not is_test_run,
            )
            state.response_modifiers["response_contract"] = contract.to_dict()
            if not is_background and not is_test_run:
                search_executed = await self._execute_required_search_evidence(
                    state=state,
                    objective=objective,
                    contract=contract,
                    origin=origin,
                    runtime_context=kwargs.get("context") if isinstance(kwargs.get("context"), dict) else {},
                )
                if search_executed:
                    messages = _assemble(delivered_history, delivered_note)
                    contract = build_response_contract(
                        state,
                        objective,
                        is_user_facing=True,
                    )
                    state.response_modifiers["response_contract"] = contract.to_dict()
            if (
                contract.reason != "ordinary_dialogue"
                and messages
                and messages[0].get("role") == "system"
            ):
                messages[0]["content"] = (
                    f"{messages[0]['content']}\n\n{contract.to_prompt_block().strip()}"
                )
            connectivity_block = render_connectivity_prompt_block(
                getattr(state, "response_modifiers", {}).get("connectivity")
                or getattr(getattr(state, "world", None), "facts", {}).get("connectivity")
            )
            if connectivity_block and messages and messages[0].get("role") == "system":
                messages[0]["content"] = f"{messages[0]['content']}\n\n{connectivity_block}"
            if not is_background and not is_test_run:
                reliability_block = conversation_reliability_system_block(objective)
                runtime_context = kwargs.get("context")
                if not isinstance(runtime_context, dict):
                    runtime_context = {}
                if messages and messages[0].get("role") == "system":
                    messages[0]["content"] = f"{messages[0]['content']}\n\n{reliability_block}"
                else:
                    messages.insert(0, {"role": "system", "content": reliability_block})
                repair_directive = ""
                repair_directive = str(
                    runtime_context.get("response_repair_directive") or ""
                ).strip()
                if repair_directive:
                    repair_block = (
                        "## LIVE RESPONSE REPAIR DIRECTIVE\n"
                        f"{repair_directive}\n"
                        "This directive is internal. Do not mention it in the answer."
                    )
                    if messages and messages[0].get("role") == "system":
                        messages[0]["content"] = (
                            f"{messages[0]['content']}\n\n{repair_block}"
                        )
                    else:
                        messages.insert(0, {"role": "system", "content": repair_block})
                self._execute_derived_merely_read(messages, objective, runtime_context, state)

            router, runtime_context = self._execute_causal_world_model(kwargs, messages, proof_answer_run, state)
            incoming_continuation_contract = bool(
                runtime_context.get("user_surface_completion_retry", False)
                and runtime_context.get("user_surface_continuation_contract", False)
            )
            resume_capability, turn_completed_capabilities = self._execute_turn_completed_capabilities(contract, incoming_continuation_contract, is_background, is_test_run, runtime_context, state)
            desktop_cognitive_engine_required = bool(
                runtime_context.get("desktop_cognitive_engine_required", False)
                or runtime_context.get("cognitive_engine_required", False)
            )
            live_mind_generation_controls = runtime_context.get(
                "live_mind_generation_controls"
            )
            if not isinstance(live_mind_generation_controls, dict):
                live_mind_generation_controls = {}
            live_mind_controls_bound = bool(
                runtime_context.get("live_mind_controls_bound", False)
            )
            clean_user_surface_contract = bool(
                runtime_context.get("clean_user_surface_contract", False)
                or desktop_cognitive_engine_required
                or foreground_user_surface_owned
            )
            user_surface_validation_prompt = str(
                runtime_context.get("user_surface_validation_prompt")
                or runtime_context.get("visible_user_message")
                or objective
                or ""
            ).strip()
            structural_answer_floor = answer_surface_token_floor(
                user_surface_validation_prompt
            )
            visible_output_contract = requested_output_contract(
                user_surface_validation_prompt
            )
            visible_output_contract_payload = (
                visible_output_contract.as_dict()
                if not is_background and visible_output_contract.constrained
                else None
            )
            runtime_fact_status_contract = bool(
                runtime_context.get("runtime_fact_status_contract", False)
                or runtime_context.get("grounded_runtime_status_contract", False)
            )
            tier = state.response_modifiers.get(
                "model_tier", "tertiary" if is_background else "primary"
            )
            deep_handoff = (
                bool(state.response_modifiers.get("deep_handoff", False)) and not is_background
            )
            if proof_answer_run:
                tier = str(kwargs.get("prefer_tier") or "tertiary")
                deep_handoff = False
            soma_data = getattr(state, "soma", None)
            hardware = getattr(soma_data, "hardware", {}) or {}
            thermal_c = float(hardware.get("temperature", 0.0) or 0.0)
            cpu_usage = float(hardware.get("cpu_usage", 0.0) or 0.0)
            memory_pressure = None
            try:
                mem_monitor = self.container.get("memory_monitor", default=None)
                if mem_monitor is not None:
                    memory_pressure = getattr(mem_monitor, "pressure", None)
            except (OSError, ConnectionError, TimeoutError) as exc:
                logger.debug(
                    "the memory monitor did not report pressure (%s: %s)", type(exc).__name__, exc
                )
                memory_pressure = None
            if memory_pressure is None:
                try:
                    from core.runtime import resource_psutil as psutil

                    memory_pressure = psutil.virtual_memory().percent
                except (ImportError, AttributeError, RuntimeError) as exc:
                    logger.debug(
                        "memory pressure is unreadable and reads as zero (%s: %s)",
                        type(exc).__name__,
                        exc,
                    )
                    memory_pressure = 0.0

            generation_temperature, token_budget = self._execute_affect_modulated_generation(deep_handoff, is_background, live_mind_controls_bound, live_mind_generation_controls, runtime_context, state)
            generation_top_p = max(
                0.05,
                min(
                    1.0,
                    self._safe_bias_float(
                        live_mind_generation_controls.get("top_p"),
                        0.90,
                    ),
                ),
            )
            # [STABILITY v55] Raised thermal from 85°C to 95°C (M-series
            # throttles at 100°C+) and memory pressure from 85% to 94%
            # (32B model normally uses 85-90% of 64GB).
            if thermal_c >= 95.0:
                logger.warning(
                    "🌡️ ResponseGeneration: thermal guard active (temp=%.1fC cpu=%.1f%% mem=%.1f%%). Downshifting tier/tokens.",
                    thermal_c,
                    cpu_usage,
                    float(memory_pressure or 0.0),
                )
                tier = "tertiary"
                deep_handoff = False
                token_budget = min(token_budget, 4096, max(1, int(token_budget * 0.7)))
                state.response_modifiers["thermal_guard"] = True
            elif float(memory_pressure or 0.0) >= 94.0:
                token_budget = min(token_budget, 4096, max(1, int(token_budget * 0.8)))
                state.response_modifiers["thermal_guard"] = True
            else:
                state.response_modifiers["thermal_guard"] = False
            if (
                visible_output_contract_payload is not None
                and visible_output_contract.hard_token_ceiling is not None
            ):
                token_budget = min(
                    token_budget,
                    max(1, int(visible_output_contract.hard_token_ceiling)),
                )
            else:
                # Sampling biases and stale caller defaults may make an answer
                # terser; neither may make the executable surface smaller than
                # the visible request. Live 2026-08-17: five requested Dijkstra
                # sections entered with a 1,536-token route allowance, were
                # halved by an internal bias, then capped again by RLC at 326.
                # The model stopped during pseudocode and four sections never
                # had a chance to exist. The structural planner is the owner of
                # this floor. A true user-declared output ceiling is handled by
                # the constrained branch above.
                token_budget = max(token_budget, structural_answer_floor)
            runtime_context["_effective_generation_max_tokens"] = token_budget
            runtime_context["requested_output_contract"] = (
                dict(visible_output_contract_payload)
                if visible_output_contract_payload is not None
                else None
            )
            runtime_context["semantic_output_token_cap"] = (
                visible_output_contract.semantic_token_cap
            )
            runtime_context["hard_output_token_ceiling"] = (
                visible_output_contract.hard_token_ceiling
            )

            response_text: Any = None
            latent_trace: dict[str, Any] = {
                "latent_cortex_selected": False,
                "latent_cortex_attempted": False,
                "latent_cortex_succeeded": False,
                "latent_cortex_fallback_used": False,
                "latent_cortex_failure_reason": "",
                "latent_cortex_identity_bound": False,
                "latent_cortex_receipt": {},
                "latent_cortex_progress": {},
            }
            amplifier_promotion_authority = "none"
            append_only_continuation_pending = False
            continuation_draft = ""
            try:
                request_timeout = self._surface_request_timeout(
                    is_background=is_background,
                    deep_handoff=deep_handoff,
                    token_budget=token_budget,
                )
                if not is_background and not deep_handoff and not proof_answer_run:
                    from core.brain.llm.generation_allowance import resident_generation_seconds
                    from core.runtime.completion_admission import admit_completion_work

                    measured_seconds = resident_generation_seconds(messages, token_budget)
                    if measured_seconds > 0.0:
                        request_timeout = max(request_timeout, measured_seconds)
                        admit_completion_work(request_timeout + 4.0)
                request_timeout = self._bounded_request_timeout(
                    runtime_context,
                    request_timeout,
                    reserve_s=4.0,
                )
                if request_timeout < 1.0:
                    raise TimeoutError(
                        "cognitive cycle budget exhausted before response generation"
                    )
                from core.brain.foreground_latent_runtime import (
                    run_foreground_latent_episode,
                )

                service = self.container.get("latent_cortex", default=None)
                orchestrator = self.container.get("orchestrator", default=None)
                if orchestrator is None:
                    orchestrator = getattr(self, "orchestrator", None)
                latent_outcome = await run_foreground_latent_episode(
                    orchestrator=orchestrator,
                    service=service,
                    messages=messages,
                    visible_objective=str(
                        runtime_context.get("visible_user_message") or objective or ""
                    ),
                    foreground=not is_background,
                    desktop_required=desktop_cognitive_engine_required,
                    cognitive_mode=str(state.cognition.current_mode.value),
                    request_timeout_s=request_timeout,
                    prompt_shape=(
                        runtime_context.get("prompt_shape")
                        if isinstance(runtime_context.get("prompt_shape"), dict)
                        else None
                    ),
                    compact_contract=bool(
                        runtime_context.get("compact_desktop_chat_contract", False)
                    ),
                    strict_output_contract=visible_output_contract_payload is not None,
                    incompatible_contract=bool(
                        runtime_context.get("desktop_execution_contract", False)
                        or runtime_context.get("capability_inventory_contract", False)
                        or runtime_fact_status_contract
                        or runtime_context.get("memory_state_contract", False)
                        or runtime_context.get("self_condition_contract", False)
                    ),
                    proof_or_benchmark=proof_answer_run,
                    explicitly_required=bool(
                        runtime_context.get("latent_cortex_required", False)
                    ),
                    tenant_id=str(runtime_context.get("tenant_id") or "local"),
                    user_id=str(
                        runtime_context.get("user_id")
                        or runtime_context.get("owner_id")
                        or "owner"
                    ),
                    session_id=str(
                        runtime_context.get("session_id")
                        or runtime_context.get("conversation_id")
                        or "local"
                    ),
                    domain=str(
                        runtime_context.get("latent_cortex_domain")
                        or "desktop_conversation"
                    ),
                    decode_max_tokens=int(token_budget),
                    decode_temperature=float(generation_temperature),
                    decode_top_p=float(generation_top_p),
                    recurrent_loops=int(
                        live_mind_generation_controls.get(
                            "clean_user_surface_recurrent_loops", 1
                        )
                    ),
                    steering_alpha=self._safe_bias_float(
                        live_mind_generation_controls.get(
                            "clean_user_surface_steering_alpha"
                        ),
                        0.25,
                    ),
                    capability_modifiers=dict(state.response_modifiers),
                )
                latent_trace = dict(latent_outcome.trace)
                state.response_modifiers.update(latent_trace)
                _seam_early_response, generation_metadata, response_text = await _serve_the_cached_generation(
                    generation_metadata=generation_metadata,
                    latent_outcome=latent_outcome,
                    latent_trace=latent_trace,
                    live_mind_controls_bound=live_mind_controls_bound,
                    live_mind_generation_controls=live_mind_generation_controls,
                    response_text=response_text,
                    self=self,
                    state=state,
                    token_budget=token_budget,
                    visible_output_contract_payload=visible_output_contract_payload,
                )
                if _seam_early_response is not _SEAM_FELL_THROUGH:
                    return _seam_early_response

                if response_text is None:
                    ordinary_timeout = self._bounded_request_timeout(
                        runtime_context,
                        request_timeout,
                        reserve_s=4.0,
                    )
                    if ordinary_timeout < 8.0:
                        raise TimeoutError(
                            "cognitive cycle budget exhausted before resident fallback"
                        )
                    router_generation_metadata_sink: dict[str, Any] = {}
                    final_generation_kwargs = {
                        **resume_capability.context,
                    }
                    if incoming_continuation_contract:
                        final_generation_kwargs.update(
                            {
                                "user_surface_continuation_contract": True,
                                "user_surface_continuation_partial": str(
                                    runtime_context.get(
                                        "user_surface_continuation_partial"
                                    )
                                    or ""
                                ),
                            }
                        )
                    think_coro = router.think(
                        messages=messages,
                        cognitive_mode=str(state.cognition.current_mode.value),
                        priority=1.0 if not is_background else 0.5,
                        origin=f"response_generation_{origin}",
                        purpose="reply" if not is_background else "background",
                        prefer_tier=tier,
                        is_background=is_background,
                        protected_foreground_lane=bool(not is_background and not proof_answer_run),
                        foreground_request=not is_background,
                        deep_handoff=deep_handoff,
                        allow_cloud_fallback=False,
                        cognitive_engine_required=bool(
                            runtime_context.get("cognitive_engine_required", False)
                        ),
                        # Forwarded, or the gate cannot know this turn has to
                        # emit a PLAN. ResponseGeneration already injects the
                        # desktop planning contract into the messages from this
                        # same flag, but never passed the flag itself, so the
                        # gate applied the origin's conversational default of
                        # 288 tokens — too small for a multi-step JSON plan.
                        # The model then emitted prose, the draft was judged
                        # truncated, and nothing executed.
                        desktop_execution_contract=bool(
                            runtime_context.get("desktop_execution_contract", False)
                        ),
                        desktop_cognitive_engine_required=bool(
                            runtime_context.get("desktop_cognitive_engine_required", False)
                            or runtime_context.get("cognitive_engine_required", False)
                        ),
                        live_runtime_payload_required=bool(
                            runtime_context.get("live_runtime_payload_required", False)
                        ),
                        visible_user_message=user_surface_validation_prompt,
                        completed_capability_evidence=runtime_context.get(
                            "completed_capability_evidence"
                        ),
                        recent_conversation_context=str(
                            runtime_context.get("recent_conversation_context") or ""
                        ),
                        recent_context_needed=bool(
                            runtime_context.get("recent_context_needed", False)
                        ),
                        allow_mesh_cognition=bool(
                            runtime_context.get("allow_mesh_cognition", True)
                        ),
                        skip_runtime_payload=bool(
                            runtime_context.get("skip_runtime_payload", False)
                        ),
                        disable_prompt_cache=bool(
                            runtime_context.get("disable_prompt_cache", False)
                        ),
                        clear_prompt_cache=bool(
                            runtime_context.get("clear_prompt_cache", False)
                        ),
                        memory_state_contract=bool(
                            runtime_context.get("memory_state_contract", False)
                        ),
                        runtime_fact_status_contract=runtime_fact_status_contract,
                        grounded_runtime_status_contract=runtime_fact_status_contract,
                        clean_user_surface_contract=clean_user_surface_contract,
                        semantic_completion_contract=clean_user_surface_contract,
                        user_surface_validation_prompt=user_surface_validation_prompt,
                        user_surface_prompt_binding=dict(
                            runtime_context.get("user_surface_prompt_binding") or {}
                        ),
                        clean_user_surface_recurrent_loops=int(
                            live_mind_generation_controls.get(
                                "clean_user_surface_recurrent_loops", 1
                            )
                        ),
                        clean_user_surface_steering_alpha=self._safe_bias_float(
                            live_mind_generation_controls.get(
                                "clean_user_surface_steering_alpha"
                            ),
                            0.25,
                        ),
                        live_mind_controls_bound=live_mind_controls_bound,
                        live_mind_generation_controls=dict(
                            live_mind_generation_controls
                        ),
                        live_mind_snapshot_ready=bool(
                            runtime_context.get("live_mind_snapshot_ready", False)
                        ),
                        live_mind_required_subsystems_ok=bool(
                            runtime_context.get(
                                "live_mind_required_subsystems_ok", False
                            )
                        ),
                        cognitive_situation_sampling_bias=state.response_modifiers.get(
                            "cognitive_situation_sampling_bias"
                        ),
                        soma=soma_data,
                        state=state,
                        temperature=generation_temperature,
                        top_p=generation_top_p,
                        max_tokens=token_budget,
                        user_surface_completion_floor=structural_answer_floor,
                        requested_output_contract=(
                            dict(visible_output_contract_payload)
                            if visible_output_contract_payload is not None
                            else None
                        ),
                        semantic_output_token_cap=visible_output_contract.semantic_token_cap,
                        hard_output_token_ceiling=visible_output_contract.hard_token_ceiling,
                        _generation_metadata_sink=router_generation_metadata_sink,
                        timeout=ordinary_timeout,
                        **final_generation_kwargs,
                    )
                    generation_metadata, response_text = await self._execute_tenth_clock_same(is_background, latent_trace, ordinary_timeout, origin, router, router_generation_metadata_sink, think_coro)

                append_only_continuation_pending = (
                    _append_only_continuation_pending(
                        generation_metadata,
                        clean_user_surface_contract=clean_user_surface_contract,
                    )
                )
                continuation_draft = (
                    str(response_text or "")
                    if append_only_continuation_pending
                    else ""
                )
                if append_only_continuation_pending:
                    logger.info(
                        "ResponseGeneration preserved a resumable foreground draft "
                        "for the route's append-only continuation owner."
                    )

                shape_repaired = False
                if (
                    not is_background
                    and not is_test_run
                    and not append_only_continuation_pending
                ):
                    pre_shape_text = response_text
                    response_text, shape_repaired, shape_repair_reasons = (
                        self._repair_substantive_instruction_shape_miss(
                            user_surface_validation_prompt, response_text
                        )
                    )
                    if shape_repaired:
                        append_text_mutation(
                            response_mutation_receipt,
                            stage="response_generation.pre_critique_shape",
                            method="deterministic_instruction_shape",
                            reasons=shape_repair_reasons,
                            before=pre_shape_text,
                            after=response_text,
                            deterministic=True,
                            authorship_effect="preserved",
                        )
                        logger.info(
                            "🛡️ ResponseGeneration repaired instruction shape locally before critique (%s).",
                            ",".join(shape_repair_reasons) or "unknown",
                        )

                pre_amplifier_text = response_text
                if str(response_text or "").strip():
                    servable_incumbent = response_text
                amplifier_promotion_authority = "none"
                # Selection reserves the latent lane; it does not prove that
                # lane authored the response.  When RLC declines before
                # execution, the ordinary resident fallback still needs the
                # same verifier, composer, and dialogue-recovery stages as an
                # ordinary turn.  A successful latent answer or its already
                # materialized incumbent keeps single-owner protection.
                latent_response_owned = bool(
                    latent_trace.get("latent_cortex_succeeded") is True
                    or latent_trace.get("latent_cortex_incumbent_fallback_served")
                    is True
                )
                # A selected latent episode already spent the hard-turn compute
                # budget even when admission later declined. Do not reopen the
                # multi-candidate amplifier here: the ordinary fallback is the
                # one remaining generator and the verifier/composer/retry stages
                # below may still inspect or repair what it produced.
                if (
                    latent_trace.get("latent_cortex_selected") is not True
                    and not append_only_continuation_pending
                ):
                    logger.info(
                        "Reasoning amplifier reached for this turn; deciding whether to run."
                    )
                    response_text = await self._maybe_amplify_response(
                        objective=objective,
                        draft=response_text,
                        router=router,
                        state=state,
                        request_timeout=request_timeout,
                        origin=origin,
                        tier=tier,
                        runtime_context=runtime_context,
                        is_user_facing=not is_background and not is_test_run,
                        is_background=is_background,
                        proof_or_benchmark=proof_answer_run,
                        turn_completed_capabilities=turn_completed_capabilities,
                    )
                    amplifier_promotion_authority = str(
                        (
                            state.response_modifiers.get(
                                "reasoning_amplifier_v2_active_phase"
                            )
                            or {}
                        ).get("promotion_authority")
                        or "none"
                    )
                else:
                    # The other way it never runs, and it said nothing either.
                    logger.info(
                        "Reasoning amplifier not reached (%s); the draft stands.",
                        "the latent cortex owns this turn"
                        if latent_trace.get("latent_cortex_selected") is True
                        else "an append-only continuation is pending",
                    )
                generation_metadata = self._execute_amplifier_generation_metadata(generation_metadata, latent_trace, pre_amplifier_text, response_mutation_receipt, response_text)

                # System 2 internal critique layer to verify logical correctness
                try:
                    from core.brain.reasoning_strategies import ReasoningStrategies

                    async def _raw_generate(p, **kw):
                        try:
                            critique_cap = max(
                                1,
                                int(kw.get("max_tokens") or token_budget),
                            )
                        except (TypeError, ValueError, OverflowError):
                            critique_cap = token_budget
                        kw["max_tokens"] = min(token_budget, critique_cap)
                        kw["user_surface_completion_floor"] = structural_answer_floor
                        kw["requested_output_contract"] = (
                            dict(visible_output_contract_payload)
                            if visible_output_contract_payload is not None
                            else None
                        )
                        kw["semantic_output_token_cap"] = (
                            visible_output_contract.semantic_token_cap
                        )
                        kw["hard_output_token_ceiling"] = (
                            visible_output_contract.hard_token_ceiling
                        )
                        kw.setdefault(
                            "user_surface_validation_prompt",
                            user_surface_validation_prompt,
                        )
                        kw.setdefault(
                            "completed_capability_evidence",
                            runtime_context.get("completed_capability_evidence"),
                        )
                        return await router.think(p, **kw)

                    strategies = ReasoningStrategies(_raw_generate)
                    if (
                        not desktop_cognitive_engine_required
                        and not append_only_continuation_pending
                        and not shape_repaired
                        and amplifier_promotion_authority == "none"
                        and strategies._is_logical_check(objective)
                    ):
                        logger.info("⚡ [Critique] Running System 2 self-critique on response...")
                        critique_response = await strategies._self_critique(objective, response_text, origin=origin)
                        if critique_response and critique_response != response_text:
                            logger.info("⚡ [Critique] Self-critique corrected the generated response!")
                            append_text_mutation(
                                response_mutation_receipt,
                                stage="response_generation.system2_critique",
                                method="model_critique_replacement",
                                reasons=["logical_self_critique"],
                                before=response_text,
                                after=critique_response,
                                deterministic=False,
                                authorship_effect="replaced_by_model",
                            )
                            response_text = critique_response
                            generation_metadata = {
                                **self._generation_metadata_snapshot(router),
                                **latent_trace,
                            }
                except (ImportError, AttributeError, TypeError, ValueError, LookupError, RuntimeError, NameError, SyntaxError, TimeoutError) as critique_exc:
                    logger.warning("Failed to run System 2 self-critique: %s", critique_exc)

                # ComposerNode: Structural Refinement
                composer = (
                    None
                    if (
                        latent_response_owned
                        or amplifier_promotion_authority != "none"
                        or append_only_continuation_pending
                        or foreground_user_surface_owned
                    )
                    else self.container.get("composer_node", default=None)
                )
                if composer and hasattr(composer, "refine"):
                    logger.debug("🎨 [Composer] Refining response structure...")
                    pre_composer_text = response_text
                    response_text = await composer.refine(response_text, objective=objective)
                    append_text_mutation(
                        response_mutation_receipt,
                        stage="response_generation.composer_refinement",
                        method="composer_replacement",
                        reasons=["structural_refinement"],
                        before=pre_composer_text,
                        after=response_text,
                        deterministic=False,
                        authorship_effect="replaced_by_runtime",
                    )

            except TimeoutError:
                state.response_modifiers.update(latent_trace)
                state.response_modifiers.update(
                    {
                        "model_retry_suppressed": True,
                        "generation_failure_class": (
                            str(latent_trace.get("latent_cortex_failure_reason") or "")
                            or "response_generation_deadline_exhausted"
                        )[:120],
                        "response_path": "cognitive_engine_generation_timeout",
                    }
                )
                logger.error(
                    "🛑 ResponseGeneration Phase TIMEOUT (%.0fs). Logic took too long.",
                    request_timeout + 4.0,
                )
                if servable_incumbent is not None and str(servable_incumbent).strip():
                    pre_optional_timeout_recovery = response_text
                    response_text = servable_incumbent
                    append_text_mutation(
                        response_mutation_receipt,
                        stage="response_generation.optional_stage_timeout_recovery",
                        method="preserve_servable_incumbent",
                        reasons=["optional_stage_exceeded_turn_deadline"],
                        before=pre_optional_timeout_recovery,
                        after=response_text,
                        deterministic=True,
                        authorship_effect="preserved",
                    )
                    state.response_modifiers["optional_stage_timeout_repaired"] = {
                        "method": "preserve_servable_incumbent",
                        "completed_capabilities": sorted(turn_completed_capabilities),
                    }
                    logger.warning(
                        "ResponseGeneration preserved the resident model's servable "
                        "incumbent after an optional stage exceeded the turn deadline."
                    )
                else:
                    required_tool_hit = self._successful_required_search_payload(state, contract)
                    if required_tool_hit and not is_background:
                        skill_name, payload = required_tool_hit
                        pre_tool_timeout_recovery = response_text
                        response_text = self._render_required_search_answer_from_payload(
                            payload=payload,
                        )
                        append_text_mutation(
                            response_mutation_receipt,
                            stage="response_generation.required_tool_timeout_recovery",
                            method="deterministic_grounded_evidence",
                            reasons=["cortex_timeout_with_successful_tool_evidence"],
                            before=pre_tool_timeout_recovery,
                            after=response_text,
                            deterministic=True,
                            authorship_effect="replaced_by_runtime",
                        )
                        state.response_modifiers["required_tool_timeout_repaired"] = {
                            "skill": skill_name,
                            "method": "deterministic_grounded_evidence",
                        }
                        logger.warning(
                            "ResponseGeneration answered from successful %s evidence "
                            "after Cortex timeout.",
                            skill_name,
                        )
                    else:
                        # No model-authored incumbent and no exact tool result exist.
                        # Return no text so the route can use its governed recovery
                        # owner instead of publishing an invented timeout answer.
                        return state

            # Handle None response from router.think()
            if response_text is None:
                required_tool_hit = self._successful_required_search_payload(state, contract)
                if required_tool_hit and not is_background:
                    skill_name, payload = required_tool_hit
                    pre_tool_empty_recovery = response_text
                    response_text = self._render_required_search_answer_from_payload(
                        payload=payload,
                    )
                    append_text_mutation(
                        response_mutation_receipt,
                        stage="response_generation.required_tool_empty_recovery",
                        method="deterministic_grounded_evidence",
                        reasons=["empty_cortex_result_with_successful_tool_evidence"],
                        before=pre_tool_empty_recovery,
                        after=response_text,
                        deterministic=True,
                        authorship_effect="replaced_by_runtime",
                    )
                    state.response_modifiers["required_tool_empty_repaired"] = {
                        "skill": skill_name,
                        "method": "deterministic_grounded_evidence",
                    }
                    logger.warning(
                        "🛡️ ResponseGeneration answered from successful %s evidence after empty Cortex result.",
                        skill_name,
                    )
                else:
                    logger.debug("💭 ResponseGeneration: LLM returned None. Skipping this tick.")
                    return state
            if not str(response_text or "").strip():
                required_tool_hit = self._successful_required_search_payload(state, contract)
                if required_tool_hit and not is_background:
                    skill_name, payload = required_tool_hit
                    pre_tool_blank_recovery = response_text
                    response_text = self._render_required_search_answer_from_payload(
                        payload=payload,
                    )
                    append_text_mutation(
                        response_mutation_receipt,
                        stage="response_generation.required_tool_blank_recovery",
                        method="deterministic_grounded_evidence",
                        reasons=["blank_cortex_result_with_successful_tool_evidence"],
                        before=pre_tool_blank_recovery,
                        after=response_text,
                        deterministic=True,
                        authorship_effect="replaced_by_runtime",
                    )
                    state.response_modifiers["required_tool_empty_repaired"] = {
                        "skill": skill_name,
                        "method": "deterministic_grounded_evidence",
                    }
                    logger.warning(
                        "🛡️ ResponseGeneration answered from successful %s evidence after blank Cortex result.",
                        skill_name,
                    )
                else:
                    logger.debug("💭 ResponseGeneration: LLM returned blank text. Skipping this tick.")
                    return state
            if not is_background and not append_only_continuation_pending:
                if (
                    origin != "test"
                    and not env_present("AURA_AGI_MAX_TASKS", description="AGI battery task cap; presence marks a battery run", owner="core.runtime")
                    and not env_present("AURA_TESTING", description="presence marks a test run", owner="core.runtime")
                    and not env_present("AURA_PROOF_RUN", description="presence marks a proof run", owner="core.runtime")
                ):
                    reliability = assess_user_facing_reply(
                        user_surface_validation_prompt, response_text
                    )
                    if reliability.retryable:
                        repaired_text, repaired_shape, repair_reasons = (
                            self._repair_substantive_instruction_shape_miss(
                                user_surface_validation_prompt, response_text
                            )
                        )
                        if repaired_shape:
                            logger.info(
                                "🛡️ ResponseGeneration repaired instruction shape locally after refinement (%s).",
                                ",".join(repair_reasons) or "unknown",
                            )
                            append_text_mutation(
                                response_mutation_receipt,
                                stage="response_generation.post_refinement_shape",
                                method="deterministic_instruction_shape",
                                reasons=repair_reasons,
                                before=response_text,
                                after=repaired_text,
                                deterministic=True,
                                authorship_effect="preserved",
                            )
                            response_text = repaired_text
                            reliability = assess_user_facing_reply(
                                user_surface_validation_prompt, response_text
                            )
                        reliability_reasons = set(reliability.reasons or ())
                        response_text_s = str(response_text or "").strip()
                        # One shared policy, not a third private opinion. A
                        # draft is discarded only when something in it must not
                        # be SPOKEN; a draft that merely fell short is kept and
                        # repaired. See core/conversation/surface_disposition.py
                        # for why the three gates were unified.
                        from core.conversation.surface_disposition import (
                            draft_is_servable,
                        )

                        if (
                            reliability_reasons
                            and (
                                reliability_reasons.issubset(
                                    _DOWNSTREAM_REPAIRABLE_RESPONSE_REASONS
                                )
                                or draft_is_servable(reliability_reasons)
                            )
                            and len(response_text_s) >= 48
                            and len(response_text_s.split()) >= 8
                        ):
                            # The draft itself, bounded. A reason and a length
                            # describe a rejection without saying what was
                            # rejected, and the two questions a reader has are
                            # "was the gate right?" and "what did she nearly
                            # say?" — neither answerable from a number. The
                            # file sink redacts, and this stays local.
                            logger.warning(
                                "🛡️ ResponseGeneration kept repairable foreground draft for final reply repair (%s, len=%d): %r",
                                ",".join(reliability.reasons) or "unknown",
                                len(response_text_s),
                                response_text_s[:_REJECTED_DRAFT_LOG_CHARS],
                            )
                            try:
                                from core.conversation.surface_disposition import (
                                    preserve_draft,
                                )

                                preserve_draft(response_text_s)
                            except (ImportError, RuntimeError, TypeError, ValueError) as exc:
                                logger.debug(
                                    "the rejected draft was not preserved (%s: %s)",
                                    type(exc).__name__,
                                    exc,
                                )
                        else:
                            logger.warning(
                                "🛡️ ResponseGeneration rejected unsafe user-facing draft (%s, len=%d): %r",
                                ",".join(reliability.reasons) or "unknown",
                                len(str(response_text or "")),
                                str(response_text or "")[:_REJECTED_DRAFT_LOG_CHARS],
                            )
                            return state

            action, content = self._execute_defensive_hardening_json(append_only_continuation_pending, kwargs, objective, response_mutation_receipt, response_text, state)

            # 5. Executive Guard — real-time identity alignment
            guard = get_executive_guard()
            if append_only_continuation_pending:
                cleaned_response, was_corrected, violations = content, False, []
            else:
                cleaned_response, was_corrected, violations = guard.align(content)
            append_text_mutation(
                response_mutation_receipt,
                stage="response_generation.executive_guard",
                method="deterministic_identity_alignment",
                reasons=list(violations or []),
                before=content,
                after=cleaned_response,
                deterministic=True,
                authorship_effect="preserved",
            )
            if was_corrected:
                logger.info(
                    "🛡️ ExecutiveGuard corrected %d violation(s) in LLM output.", len(violations)
                )

            async def _retry_dialogue(repair_block: str) -> str:
                retry_messages = [dict(msg) for msg in messages]
                if retry_messages and retry_messages[0].get("role") == "system":
                    retry_messages[0]["content"] = (
                        f"{repair_block}\n\n{retry_messages[0]['content']}"
                    )
                else:
                    retry_messages.insert(0, {"role": "system", "content": repair_block})

                retry_timeout = min(35.0, max(12.0, request_timeout * 0.5))
                retried = await router.think(
                    messages=retry_messages,
                    cognitive_mode=str(state.cognition.current_mode.value),
                    priority=1.0 if not is_background else 0.5,
                    origin=f"response_generation_{origin}",
                    purpose="reply" if not is_background else "background",
                    prefer_tier=tier,
                    is_background=is_background,
                    protected_foreground_lane=not is_background,
                    deep_handoff=deep_handoff,
                    allow_cloud_fallback=False,
                    cognitive_engine_required=bool(
                        runtime_context.get("cognitive_engine_required", False)
                    ),
                    desktop_cognitive_engine_required=desktop_cognitive_engine_required,
                    live_runtime_payload_required=bool(
                        runtime_context.get("live_runtime_payload_required", False)
                    ),
                    visible_user_message=user_surface_validation_prompt,
                    completed_capability_evidence=runtime_context.get(
                        "completed_capability_evidence"
                    ),
                    recent_conversation_context=str(
                        runtime_context.get("recent_conversation_context") or ""
                    ),
                    recent_context_needed=bool(
                        runtime_context.get("recent_context_needed", False)
                    ),
                    allow_mesh_cognition=bool(
                        runtime_context.get("allow_mesh_cognition", True)
                    ),
                    skip_runtime_payload=bool(
                        runtime_context.get("skip_runtime_payload", False)
                    ),
                    disable_prompt_cache=bool(
                        runtime_context.get("disable_prompt_cache", False)
                    ),
                    clear_prompt_cache=bool(
                        runtime_context.get("clear_prompt_cache", False)
                    ),
                    memory_state_contract=bool(
                        runtime_context.get("memory_state_contract", False)
                    ),
                    runtime_fact_status_contract=runtime_fact_status_contract,
                    grounded_runtime_status_contract=runtime_fact_status_contract,
                    clean_user_surface_contract=clean_user_surface_contract,
                    semantic_completion_contract=clean_user_surface_contract,
                    user_surface_validation_prompt=user_surface_validation_prompt,
                    user_surface_prompt_binding=dict(
                        runtime_context.get("user_surface_prompt_binding") or {}
                    ),
                    clean_user_surface_recurrent_loops=int(
                        live_mind_generation_controls.get(
                            "clean_user_surface_recurrent_loops", 1
                        )
                    ),
                    clean_user_surface_steering_alpha=self._safe_bias_float(
                        live_mind_generation_controls.get(
                            "clean_user_surface_steering_alpha"
                        ),
                        0.25,
                    ),
                    live_mind_controls_bound=live_mind_controls_bound,
                    live_mind_generation_controls=dict(
                        live_mind_generation_controls
                    ),
                    live_mind_snapshot_ready=bool(
                        runtime_context.get("live_mind_snapshot_ready", False)
                    ),
                    live_mind_required_subsystems_ok=bool(
                        runtime_context.get(
                            "live_mind_required_subsystems_ok", False
                        )
                    ),
                    cognitive_situation_sampling_bias=state.response_modifiers.get(
                        "cognitive_situation_sampling_bias"
                    ),
                    soma=soma_data,
                    state=state,
                    temperature=generation_temperature,
                    top_p=generation_top_p,
                    max_tokens=token_budget,
                    user_surface_completion_floor=structural_answer_floor,
                    requested_output_contract=(
                        dict(visible_output_contract_payload)
                        if visible_output_contract_payload is not None
                        else None
                    ),
                    semantic_output_token_cap=visible_output_contract.semantic_token_cap,
                    hard_output_token_ceiling=visible_output_contract.hard_token_ceiling,
                    timeout=retry_timeout,
                )
                retried_text = str(retried or "").strip()
                if guard and retried_text:
                    pre_retry_guard_text = retried_text
                    retried_text, retry_guard_corrected, retry_violations = guard.align(
                        retried_text
                    )
                    append_text_mutation(
                        response_mutation_receipt,
                        stage="response_generation.dialogue_retry_executive_guard",
                        method="deterministic_identity_alignment",
                        reasons=list(retry_violations or []),
                        before=pre_retry_guard_text,
                        after=retried_text,
                        deterministic=True,
                        authorship_effect="preserved",
                    )
                    if retry_guard_corrected:
                        logger.debug(
                            "ExecutiveGuard corrected dialogue-retry output before validation."
                        )
                return retried_text

            pre_dialogue_text = cleaned_response
            if append_only_continuation_pending:
                dialogue_validation = validate_dialogue_response(
                    cleaned_response,
                    contract,
                    state,
                )
                dialogue_retried = False
            else:
                (
                    cleaned_response,
                    dialogue_validation,
                    dialogue_retried,
                ) = await enforce_dialogue_contract(
                    cleaned_response,
                    contract,
                    retry_generate=(
                        _retry_dialogue
                        if (
                            not is_background
                            and not is_test_run
                            and not latent_response_owned
                            and amplifier_promotion_authority == "none"
                            # Foreground chat has one completion owner: the route's
                            # typed append-only continuation or governed recovery.
                            # Ownership belongs to the turn, not to whether every
                            # intermediate alignment stage kept non-empty bytes. A
                            # stage that empties a valid draft must never reopen a
                            # second full decode and consume the rest of the turn.
                            # Non-user-facing compositions still own their local
                            # retry because no chat route exists above them.
                            and not foreground_user_surface_owned
                            and not clean_user_surface_contract
                            and not desktop_cognitive_engine_required
                            and not bool(getattr(contract, "is_user_facing", False))
                        )
                        else None
                    ),
                    state=state,
                    user_message=user_surface_validation_prompt,
                )
            state.response_modifiers["dialogue_validation"] = dialogue_validation.to_dict()
            dialogue_provenance = _dialogue_mutation_provenance(
                pre_dialogue_text,
                cleaned_response,
                retry_attempted=dialogue_retried,
                selected_source=getattr(dialogue_validation, "selected_source", ""),
            )
            append_text_mutation(
                response_mutation_receipt,
                stage=dialogue_provenance["stage"],
                method=dialogue_provenance["method"],
                reasons=list(getattr(dialogue_validation, "violations", []) or []),
                before=pre_dialogue_text,
                after=cleaned_response,
                deterministic=dialogue_provenance["deterministic"],
                authorship_effect=dialogue_provenance["authorship_effect"],
            )
            if dialogue_provenance["model_replaced"]:
                generation_metadata = {
                    **self._generation_metadata_snapshot(router),
                    **latent_trace,
                }
                logger.info("🗣️ ResponseGeneration: retried draft to satisfy dialogue contract.")

            pre_tool_repair = cleaned_response
            if not append_only_continuation_pending:
                cleaned_response = self._repair_false_required_tool_inability(
                    state=state,
                    contract=contract,
                    response_text=cleaned_response,
                )
            append_text_mutation(
                response_mutation_receipt,
                stage="response_generation.tool_inability_grounding",
                method="deterministic_tool_claim_repair",
                reasons=["false_required_tool_inability"],
                before=pre_tool_repair,
                after=cleaned_response,
                deterministic=True,
                authorship_effect="augmented_by_runtime",
            )

            # 6. Clean response
            pre_clean_response = cleaned_response
            if not append_only_continuation_pending:
                cleaned_response = self._clean_response(
                    cleaned_response,
                    state,
                    allow_mumbling=is_background,
                )
            append_text_mutation(
                response_mutation_receipt,
                stage="response_generation.clean_response",
                method="deterministic_surface_cleanup",
                reasons=["surface_cleanup"],
                before=pre_clean_response,
                after=cleaned_response,
                deterministic=True,
                authorship_effect="preserved",
            )

            # 6b. SUBSTRATE VOICE: Shape the response — enforce the profile
            # The substrate compiled constraints. Now enforce them on the output.
            _shaped_messages = None
            if (
                _sve
                and _speech_profile
                and cleaned_response
                and not is_test_run
                and not latent_response_owned
                and not append_only_continuation_pending
            ):
                try:
                    pre_voice_shape = cleaned_response
                    shaped = _sve.shape_response(
                        cleaned_response,
                        preserve_semantic_content=not is_background,
                    )
                    if isinstance(shaped, list):
                        # A request has one transactional response. Splitting that
                        # response into delayed OutputGate emissions made the HTTP
                        # payload contain only the first chunk, so complete answers
                        # appeared visually clipped and later chunks could be lost on
                        # navigation or shutdown. Preserve the voice-shaped chunks,
                        # but deliver all of them in the primary response.
                        shaped_parts = [
                            str(part).strip() for part in shaped if str(part).strip()
                        ]
                        cleaned_response = (
                            "\n\n".join(shaped_parts)
                            if shaped_parts
                            else pre_voice_shape
                        )
                        _shaped_messages = None
                        logger.debug(
                            "🗣️ [SubstrateVoice] Rejoined %d shaped chunks into one complete response",
                            len(shaped_parts),
                        )
                    else:
                        cleaned_response = shaped
                    append_text_mutation(
                        response_mutation_receipt,
                        stage="response_generation.substrate_voice",
                        method="substrate_voice_shape",
                        reasons=["voice_profile"],
                        before=pre_voice_shape,
                        after=cleaned_response,
                        deterministic=False,
                        authorship_effect="preserved",
                    )
                except (RuntimeError, AttributeError, TypeError, ValueError) as _shape_exc:
                    _record_response_generation_degradation(
                        _shape_exc,
                        action="continued with cleaned response after substrate voice shaping failed",
                        severity="error",
                    )
                    logger.debug("ResponseShaper failed (using raw): %s", _shape_exc)

            if (
                not is_background
                and cleaned_response
                and not is_test_run
                and not latent_response_owned
                and not append_only_continuation_pending
            ):
                repaired_response, repaired_shape, repair_reasons = (
                    self._repair_substantive_instruction_shape_miss(
                        user_surface_validation_prompt, cleaned_response
                    )
                )
                if repaired_shape:
                    pre_post_voice_repair = cleaned_response
                    cleaned_response = repaired_response
                    append_text_mutation(
                        response_mutation_receipt,
                        stage="response_generation.post_voice_shape",
                        method="deterministic_instruction_shape",
                        reasons=repair_reasons,
                        before=pre_post_voice_repair,
                        after=cleaned_response,
                        deterministic=True,
                        authorship_effect="preserved",
                    )
                    state.response_modifiers["post_voice_shape_repair"] = {
                        "reasons": list(repair_reasons),
                        "method": "deterministic_instruction_shape",
                    }
                    logger.info(
                        "🛡️ ResponseGeneration repaired instruction shape after voice shaping (%s).",
                        ",".join(repair_reasons) or "unknown",
                    )

            pre_final_tool_repair = cleaned_response
            if not append_only_continuation_pending:
                cleaned_response = self._repair_false_required_tool_inability(
                    state=state,
                    contract=contract,
                    response_text=cleaned_response,
                )
            append_text_mutation(
                response_mutation_receipt,
                stage="response_generation.final_tool_inability_grounding",
                method="deterministic_tool_claim_repair",
                reasons=["false_required_tool_inability"],
                before=pre_final_tool_repair,
                after=cleaned_response,
                deterministic=True,
                authorship_effect="augmented_by_runtime",
            )

            # 6c. Skip emission for background tasks if they produced no meaningful content
            if is_background and not cleaned_response:
                return state

            if append_only_continuation_pending:
                # Every byte belongs to the cache-bound assistant prefix. The
                # route validates the completed text after it appends the next
                # segment; this phase must not judge or rewrite half a turn.
                cleaned_response = continuation_draft

            final_latent_quality: dict[str, Any] = {}
            if not append_only_continuation_pending:
                _seam_early_response, final_latent_quality = _judge_the_latent_quality(
                    cleaned_response=cleaned_response,
                    final_latent_quality=final_latent_quality,
                    latent_trace=latent_trace,
                    objective=objective,
                    runtime_context=runtime_context,
                    state=state,
                )
                if _seam_early_response is not _SEAM_FELL_THROUGH:
                    return _seam_early_response

            self._execute_surface_control_receipt(final_latent_quality, generation_metadata, latent_trace, live_mind_controls_bound, live_mind_generation_controls, response_mutation_receipt, state)

            new_state = self._execute_derive_new_state(_shaped_messages, _speech_profile, _sve, action, cleaned_response, is_background, is_test_run, objective, state)

            return new_state

        except (ImportError, AttributeError, RuntimeError) as e:
            _record_response_generation_degradation(
                e,
                action="returned prior state unchanged after response generation phase failed",
                severity="error",
            )
            logger.error("❌ ResponseGeneration: LLM call failed: %s", e, exc_info=True)
            return state

    def _clean_response(
        self,
        text: str,
        state: AuraState | None = None,
        *,
        allow_mumbling: bool = False,
    ) -> str:
        """Strip tags and assistant-isms without leaking internal thought into chat."""
        import re

        # A background generation that deferred/failed can hand us None, and every
        # re.* below then raises "expected string or bytes-like object, got
        # 'NoneType'" — which tripped the mind_tick circuit and held the runtime
        # DEGRADED (observed live 2026-07-04). A cleaner must never crash on None.
        if not isinstance(text, str):
            text = "" if text is None else str(text)

        mumbling = ""
        # Internal Monologue Spillage ("Mumbling")
        exp_state = "neutral"
        load = "normal"
        if state is not None and hasattr(state, "soma"):
            s_val = state.soma
            if s_val is not None:
                exp = getattr(s_val, "expressive", {}) or {}
                exp_state = exp.get("current_expression", "neutral")
                load = exp.get("cognitive_load", "normal")

            if allow_mumbling and (
                exp_state in ("contemplative", "anxious", "fatigued") or load == "high"
            ):
                # Extract the thought block before we strip it
                thought_match = re.search(r"<thought>(.*?)</thought>", text, flags=re.DOTALL)
                if thought_match:
                    thought_content = thought_match.group(1).strip()
                    # Just grab the last sentence or first few words to mumble
                    snippets = [s.strip() for s in thought_content.split(".") if s.strip()]
                    if snippets:
                        snippet = snippets[-1] if len(snippets) > 1 else snippets[0]
                        # Cap length
                        if len(snippet) > 80:
                            # vResilience: Workaround for str indexing/slice limitations
                            snippet = "".join([snippet[i] for i in range(77)]) + "..."
                        mumbling = f"*...{snippet.lower()}...*\n\n"

        text = re.sub(r"<thought>.*?</thought>", "", text, flags=re.DOTALL)
        text = re.sub(r"^Aura:\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"^Assistant:\s*", "", text, flags=re.IGNORECASE)

        # 🧠 COGNITIVE WIRING: Affect-Gated Prompt Hunting
        # Instead of using a 'fake band-aid' system prompt to forbid questions,
        # we wire this behavior directly into her mind. If the LLM reflexively
        # appends a trailing question, she must ACTUALLY be curious to ask it.
        if state is not None and hasattr(state, "affect"):
            curiosity = getattr(state.affect, "curiosity", 0.5)
            if curiosity < 0.70:
                # She is not curious enough to warrant a reflexive follow-up question.
                # Strip the trailing question (e.g. 'What do you think?').
                # This matches the last sentence if it ends in a question mark.
                text = re.sub(r"(?<=[.!?])\s+[A-Z][^.!?]*\?\s*$", "", text)
                text = text.strip()

        # Apply aggressive centralized scrubbing
        text = strip_meta_commentary(text)
        text = stabilize_user_facing_response(
            text,
            getattr(getattr(state, "cognition", None), "current_objective", "") or "",
        )

        return (mumbling + text.strip()).strip()
