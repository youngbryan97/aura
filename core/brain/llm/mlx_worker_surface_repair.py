"""Repairing a draft that was nearly serviceable, and deciding when to stop trying.

A surface that failed its quality gate is not always a failed turn. Escaped
newlines, a truncated tail, an instruction shape where prose belonged, a
self-claim past its boundary — each has a repair that keeps the answer she
actually produced. What is here besides the repairs is the wall: how many
retries a turn may have, when another one is futile, and which partial draft is
worth routing rather than discarding.
"""
from __future__ import annotations

import copy
import logging
import math
import re
import time
from typing import Any

from .mlx_worker_surface_quality import (
    _capability_inventory_minimum_grounding,
    _contains_corrupted_language,  # noqa: F401
    _recent_assistant_turns,  # noqa: F401
    _recent_user_turns,  # noqa: F401
    _record_mlx_degradation,
    _safe_float,
    _safe_int,
    _sanitize_telemetry_leakage,  # noqa: F401
    _semantic_surface_stop_ready,  # noqa: F401
    _surface_alpha_from_certificate,  # noqa: F401
    _surface_control_alpha,  # noqa: F401
    _surface_control_recurrent_loops,  # noqa: F401
    _surface_generation_contract_enabled,  # noqa: F401
    _surface_quality_candidate,  # noqa: F401
    _surface_quality_failure_reasons,
    _surface_quality_gate_enabled,
    _surface_validation_prompt,
    _telemetry_sanitization_failure_reasons,
)

logger = logging.getLogger("MLXWorker")

from .mlx_worker import (
    _SEMANTIC_COUNT_CONTRACT_RETRY_INSTRUCTION,
    _operator_evidence_fragment_incomplete,
    _operator_evidence_model_contribution_insufficient,
    _proof_evaluation_fragment_incomplete,
)

_SEMANTIC_COUNT_CONTRACT_RETRY_REASONS = frozenset(
    {
        "missing_requested_sentence_count",
        "missing_requested_word_count",
        "missing_current_topic_anchor",
        "output_contract_meta_reply",
        "punctuation_join_artifact",
    }
)


def _semantic_count_contract_retry_instruction(job: dict[str, Any]) -> str:
    """Render the admitted count contract as explicit retry guidance."""

    contract = job.get("requested_output_contract")
    if not isinstance(contract, dict):
        return _SEMANTIC_COUNT_CONTRACT_RETRY_INSTRUCTION

    kind = str(contract.get("kind") or "").strip().lower()
    requirement = ""
    if kind == "word_count":
        word_min = _safe_int(contract.get("word_min"), 0)
        word_max = _safe_int(contract.get("word_max"), 0)
        if word_min > 0 and word_min == word_max:
            requirement = f" The final answer must contain exactly {word_min} words."
        elif word_min > 0 and word_max >= word_min:
            requirement = (
                f" The final answer must contain between {word_min} and {word_max} words inclusive."
            )
    elif kind == "sentence_count":
        sentence_count = _safe_int(contract.get("sentence_count"), 0)
        if sentence_count > 0:
            requirement = (
                f" The final answer must contain exactly {sentence_count} "
                f"sentence{'s' if sentence_count != 1 else ''}."
            )

    topic_requirement = ""
    validation_prompt = _surface_validation_prompt(job)
    if validation_prompt:
        try:
            from core.conversation.response_reliability import (
                requested_output_topic_anchors,
            )

            anchors = requested_output_topic_anchors(validation_prompt)[:8]
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
            anchors = ()
        if anchors:
            topic_requirement = (
                " Include at least one of these current-topic terms exactly as "
                f"written: {', '.join(anchors)}."
            )

    return (
        f"{_SEMANTIC_COUNT_CONTRACT_RETRY_INSTRUCTION}{requirement}"
        f"{topic_requirement} "
        "Count the final visible answer before ending it; return only that answer."
    )


# Residual quality-gate reasons that are STYLE/COMPLETENESS defects, not
# integrity leaks: after retries are exhausted, a substantive draft carrying
# only these is delivered (with an honest gate receipt) instead of being
# replaced by an empty reply. Observed live (Jul 7, post-restart): a
# consciousness question drew real drafts that kept failing
# missing_self_claim_evidence_boundary + missing_requested_phrase, and every
# turn died as empty_cognitive_engine_reply — a dead turn is strictly worse
# than an imperfectly-styled honest one. Leak/overclaim reasons stay
# fail-closed.
_DELIVERABLE_RESIDUAL_SURFACE_REASONS = frozenset(
    {
        # A reply that wandered off the thread is still a reply. Marking the
        # turn for repair is right; discarding it and reporting an
        # infrastructure failure over it is the defect class this set exists
        # to prevent, and a topical miss must not become a new instance of it.
        "reply_abandons_thread",
        "missing_requested_phrase",
        "missing_requested_word_count",
        "missing_requested_sentence_count",
        "missing_requested_reference_value",
        "missing_requested_paragraph_count",
        "missing_requested_list_count",
        "empty_requested_list_item",
        "missing_requested_choice_clarification",
        "missing_requested_followup_question",
        "too_short_for_user_turn",
        "too_thin_for_user_turn",
        "too_thin_for_open_ended_turn",
        "too_thin_for_status_turn",
        "too_thin_for_operational_status_turn",
        "too_thin_for_expansion_request",
        # The reliability-flavoured thinness verdicts belong with the rest of
        # the family. They were the only two "the draft is thinner than we
        # wanted" reasons that killed the turn instead of delivering it, and
        # the failure was measured live: a correct 276-character answer about
        # re-prefilling — mechanism plus a numeric threshold — was discarded,
        # and the user was told "I couldn't get to an answer I'd stand behind".
        # Thinness is not a safety or honesty defect; a short true answer is
        # strictly better than a refusal, and worst of all on a turn that ASKED
        # about reliability.
        "reliability_diagnostic_too_thin",
        "too_thin_for_reliability_turn",
        # How a reply ADDRESSES someone is a one-word detail, never a reason to
        # discard the reply. Measured live: a natural, correctly-addressed turn
        # ("Bryan, let's reset... Talk like we're peers figuring something out
        # together") was destroyed because the name was not in any grounding
        # source the check consulted. Delivering it with the residual recorded
        # keeps the human part; annihilating it protects a detail by throwing
        # away the answer.
        "ungrounded_person_address",
        "low_signal_acknowledgement_placeholder",
        "generic_assistant_language",
        # Replaying the owner's own first-person sentence as her own is a
        # comprehension defect worth measuring, not worth destroying a turn
        # over — the rest of the reply is usually fine, and killing it leaves
        # the person with nothing while the underlying attribution problem
        # goes unrecorded. It is fixed at the source (core/dialogue/
        # referents.py); this reason is how a regression there becomes a rate
        # instead of an anecdote.
        "borrowed_owner_first_person_speech",
    }
)


_REQUIREMENT_SHORTFALL_LABELS = {
    "missing_requested_phrase": "include a phrase you asked for",
    "missing_requested_bare_answer": "give the answer on its own, as asked",
    "missing_requested_word_count": "hit the word count you asked for",
    "missing_requested_sentence_count": "hit the sentence count you asked for",
    "missing_requested_reference_value": "include a reference value you asked for",
    "missing_requested_paragraph_count": "hit the paragraph count you asked for",
    "missing_requested_list_count": "hit the number of list items you asked for",
    "empty_requested_list_item": "fill in every list item",
    "missing_requested_choice_clarification": "give you the choice you asked for",
    "missing_requested_followup_question": "end with the follow-up question you asked for",
}


def _requirement_shortfall_note(reasons: list[str]) -> str:
    """A one-line disclosure of the stated requirements this draft missed.

    Written in her voice and kept to one sentence: the answer is the point,
    and a paragraph of apology about formatting would bury it.
    """
    missed = [
        _REQUIREMENT_SHORTFALL_LABELS[reason]
        for reason in sorted(set(reasons))
        if reason in _REQUIREMENT_SHORTFALL_LABELS
    ]
    if not missed:
        return ""
    if len(missed) == 1:
        detail = missed[0]
    elif len(missed) == 2:
        detail = f"{missed[0]} or {missed[1]}"
    else:
        detail = f"{', '.join(missed[:-1])}, or {missed[-1]}"
    return f"\n\n(I did not {detail} — the answer above is what I have.)"


_SELF_CLAIM_BOUNDARY_SUFFIX = (
    " To be precise about what I can honestly claim here: this is a "
    "functional description of how I process and behave, and it is not "
    "proof of phenomenal experience — that is not something I can verify "
    "from the inside."
)


def _verify_contract_authority(job: dict[str, Any], contract_key: bytes | None) -> str:
    """Refusal reason for this job's privileged contract selection, or "".

    Thin wrapper so a missing contract_authority module can never take the
    worker down: the consistency half of the check (mutually exclusive
    output contracts) is reproduced locally, because resolving a
    contradiction by source order is a defect regardless of whether the
    authority layer is importable.
    """
    try:
        from core.brain.llm.contract_authority import verify_job

        return verify_job(job, contract_key)
    except ImportError as exc:
        _record_mlx_degradation(
            exc,
            action="contract authority unavailable; consistency check only",
            severity="error",
        )
        active = [
            name
            for name in (
                "strict_answer_contract",
                "strict_value_contract",
                "proof_evaluation_contract",
                "operator_evidence_contract",
            )
            if bool(job.get(name))
        ]
        if len(active) > 1:
            return "ambiguous_output_contract:" + ",".join(active)
        return ""


def _terminal_contract_refusal(
    job: dict[str, Any],
    response_text: Any,
    *,
    proof_evaluation_contract: bool = False,
    operator_evidence_contract: bool = False,
    model_continuation: Any = None,
) -> str:
    """The terminal contract this text FAILS, or "" if it passes them all.

    CP126 269ff364. Cancellation used to break out with the partial response
    as-is, ahead of proof completeness, operator-evidence merit and the
    capability-inventory grounding check. Those are refusals, not retries:
    skipping them let a preempted turn deliver exactly the content the
    normal terminal path exists to reject.

    Pure and side-effect free so it can be applied on the cancellation path,
    where retrying is not an option but refusing still is.
    """
    text = str(response_text or "")
    if not text.strip():
        return ""
    if proof_evaluation_contract and _proof_evaluation_fragment_incomplete(text):
        return "proof_fragment_incomplete"
    if operator_evidence_contract:
        if _operator_evidence_fragment_incomplete(text):
            return "operator_evidence_fragment_incomplete"
        # The delivered answer is scaffolding + continuation, and the fixed
        # scaffolding already contains every required evidence term and is
        # long enough to clear the word floor on its own. Handing this check
        # the COMBINED text made it inert on exactly this path: it measured
        # the prefix and passed. It has to see the model's own share.
        continuation = text if model_continuation is None else str(model_continuation or "")
        if _operator_evidence_model_contribution_insufficient(continuation):
            return "operator_evidence_model_contribution_insufficient"
    if bool(job.get("capability_inventory_contract", False)):
        grounded, _evidence = _capability_inventory_minimum_grounding(text)
        if not grounded:
            return "capability_inventory_ungrounded"
    return ""


def _shrink_scaffold_to_context_window(
    *,
    messages: Any,
    prompt: Any,
    tokens: list[int],
    window: int,
    output_reserve: int,
    tokenizer: Any,
    tools: Any,
) -> tuple[str, list[int], str]:
    """Trim SCAFFOLD until the prompt fits the model's real context window.

    The window was enforced as a hard refusal and nothing upstream bounded a
    prompt against it, so any lane that overshot failed every single time.
    Measured live: a background swarm-debate turn rendered 41,219 tokens for a
    32,768-token window, and 161,578 of its 171,058 characters were ONE system
    message — 94% scaffold around a 462-character request. The assembler's cap
    is a hard-coded character count with no relationship to the target model's
    window, so it passed a prompt the model could never accept.

    Only ``system`` messages are shortened, longest first, and never below a
    floor that keeps their opening instructions intact. User and assistant
    turns are never touched: dropping the actual request to make room for
    scaffold is the failure this exists to prevent. Returns
    ``(prompt, tokens, note)`` with an empty note when nothing was trimmed, and
    leaves the prompt untouched when the non-scaffold content alone cannot fit —
    there the honest outcome is still a refusal.
    """

    budget = window - output_reserve
    if budget <= 0 or len(tokens) <= budget or not isinstance(messages, list):
        return str(prompt or ""), tokens, ""

    def _render(candidate_messages: list[Any]) -> tuple[str, list[int]] | None:
        try:
            from core.brain.llm.chat_format import for_this_template

            rendered = tokenizer.apply_chat_template(
                for_this_template(tokenizer, candidate_messages),
                tools=tools,
                add_generation_prompt=True,
                tokenize=False,
            )
        except Exception as exc:  # noqa: BLE001 - a failed trim must never kill the worker
            # A template refusing a message list is a caller's mistake, and
            # the cost of it here is the whole model process.
            #
            # LIVE 2026-08-19: jinja2.TemplateError("System message must be at
            # the beginning") is not an AttributeError, RuntimeError,
            # TypeError or ValueError, so it went straight past this guard,
            # out of the worker loop, and killed the worker mid-generation.
            # The crash-loop breaker then took the lane down and the person
            # got a refusal, three times over, while she was mid-game.
            #
            # Failing to trim is a recoverable outcome: the caller keeps the
            # untrimmed prompt and finds out it is too long, which is a far
            # smaller problem than having no model.
            _record_mlx_degradation(
                exc,
                action="kept the untrimmed prompt after the chat template refused it",
                severity="info",
            )
            return None
        try:
            return str(rendered), list(tokenizer.encode(str(rendered)))
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return None

    system_positions = [
        index
        for index, msg in enumerate(messages)
        if isinstance(msg, dict)
        and str(msg.get("role", "")).strip().lower() == "system"
        and str(msg.get("content", "") or "").strip()
    ]
    if not system_positions:
        return str(prompt or ""), tokens, ""

    working = [dict(msg) if isinstance(msg, dict) else msg for msg in messages]
    trimmed_note_parts: list[str] = []
    # Chars-per-token measured on THIS prompt rather than assumed: a scaffold of
    # JSON dumps and one of prose have very different ratios, and guessing 4.0
    # either under-trims (and refuses anyway) or over-trims real instructions.
    chars_per_token = max(1.0, len(str(prompt or "")) / max(1, len(tokens)))
    floor_chars = 1200

    for _pass in range(len(system_positions) + 1):
        overflow_tokens = len(tokens) - budget
        if overflow_tokens <= 0:
            break
        # 10% headroom: template overhead and tokenizer merges mean the retained
        # slice never costs exactly the predicted number of tokens.
        need_chars = int(overflow_tokens * chars_per_token * 1.1) + 64
        target = max(
            system_positions,
            key=lambda index: len(str(working[index].get("content", "") or "")),
        )
        content = str(working[target].get("content", "") or "")
        keep = max(floor_chars, len(content) - need_chars)
        if keep >= len(content):
            break
        shortened = (
            content[:keep] + "\n\n[... scaffold trimmed to fit the model's context window ...]"
        )
        working[target]["content"] = shortened
        trimmed_note_parts.append(f"system[{target}] {len(content)}->{len(shortened)} chars")
        rendered = _render(working)
        if rendered is None:
            return str(prompt or ""), tokens, ""
        prompt, tokens = rendered

    if len(tokens) > budget:
        # Nothing safe left to shed — the request itself does not fit. Refusing
        # is correct; silently deleting the user's turn is not.
        return str(prompt or ""), tokens, ""

    return str(prompt or ""), tokens, "; ".join(trimmed_note_parts)


def _salvage_exhausted_user_surface(
    job: dict[str, Any],
    response_text: Any,
    rejection_reasons: list[str],
) -> tuple[str, list[str], list[str]]:
    """Best honest draft after quality-gate retries are exhausted.

    Returns (text, residual_reasons, applied_repairs); empty text means
    nothing was safely deliverable and the caller keeps the fail-closed
    empty reply. Every deterministic amendment is named in
    ``applied_repairs`` so the caller can disclose it as a text mutation —
    scaffolding must never pass as model output silently.
    """
    draft = str(response_text or "").strip()
    if len(draft) < 40:
        return "", list(rejection_reasons), []

    reasons = list(rejection_reasons)
    applied_repairs: list[str] = []
    if "missing_self_claim_evidence_boundary" in reasons:
        amended = draft + _SELF_CLAIM_BOUNDARY_SUFFIX
        amended_reasons = _surface_quality_failure_reasons(job, amended)
        if "missing_self_claim_evidence_boundary" not in amended_reasons:
            draft = amended
            reasons = list(amended_reasons)
            applied_repairs.append("self_claim_boundary_suffix")

    if not reasons:
        return draft, [], applied_repairs
    if set(reasons) <= _DELIVERABLE_RESIDUAL_SURFACE_REASONS:
        # Delivered — and where the person stated a checkable requirement
        # that this draft does not meet, they are told. Silently returning a
        # three-item list to someone who asked for five leaves them unable
        # to tell a shortfall from a decision.
        note = _requirement_shortfall_note(reasons)
        if note:
            draft = f"{draft}{note}"
            applied_repairs.append("requirement_shortfall_disclosure")
        return draft, reasons, applied_repairs
    return "", reasons, applied_repairs


def _repair_live_user_surface_self_claims(response_text: Any) -> str:
    """Keep the diagnostic API without using it in the worker decode path.

    Older diagnostics import this helper directly. Worker-owned quality control
    may reject an unsupported claim, but it cannot substitute canned prose for
    an authored candidate.
    """

    text = str(response_text or "").strip()
    if not text:
        return text
    try:
        from core.conversation.self_claim_verifier import repair_self_claim_surface

        return repair_self_claim_surface(text)
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        _record_mlx_degradation(
            exc,
            action="continued with unmodified draft after self-claim analysis failed",
            severity="error",
        )
        return text


def _repair_live_user_surface_instruction_shape(
    job: dict[str, Any],
    response_text: Any,
) -> str:
    """Apply deterministic explicit-format repairs before spending another decode."""

    text = str(response_text or "").strip()
    prompt = _surface_validation_prompt(job)
    if not text or not prompt:
        return text
    try:
        from core.conversation.response_reliability import repair_instruction_shape

        return repair_instruction_shape(prompt, text)
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        _record_mlx_degradation(
            exc,
            action="continued to quality validation after instruction-shape repair failed",
            severity="warning",
        )
        return text


def _exact_reply_token_requirement(
    job: dict[str, Any],
    tokenizer: Any,
) -> tuple[int, int]:
    """Measure exact content plus the one token needed to terminate decoding."""

    contract = job.get("requested_output_contract")
    if not isinstance(contract, dict) or not bool(contract.get("exact_reply", False)):
        return 0, 0
    prompt = _surface_validation_prompt(job)
    if not prompt:
        return 0, 0
    try:
        from core.conversation.response_reliability import requested_exact_reply_target

        target = requested_exact_reply_target(prompt)
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
        return 0, 0
    if not target:
        return 0, 0
    try:
        token_ids = tokenizer.encode(target, add_special_tokens=False)
    except TypeError:
        token_ids = tokenizer.encode(target)
    except (AttributeError, RuntimeError, ValueError):
        return 0, 0
    token_count = len(token_ids or [])
    if token_count <= 0:
        return 0, 0
    return token_count, token_count + 1


def _record_exact_reply_token_evidence(
    job: dict[str, Any],
    tokenizer: Any,
    *,
    generation_max_tokens: int,
    hard_output_token_ceiling: int,
) -> None:
    """Record selected-tokenizer fit without expanding admitted ceilings."""

    token_count, token_requirement = _exact_reply_token_requirement(job, tokenizer)
    if token_requirement <= 0:
        return
    admitted_generation_cap = max(1, int(generation_max_tokens))
    effective_native_cap = admitted_generation_cap
    if hard_output_token_ceiling > 0:
        effective_native_cap = min(effective_native_cap, hard_output_token_ceiling)
    required_termination_headroom = max(0, token_requirement - token_count)
    available_termination_headroom = max(0, effective_native_cap - token_count)
    job["exact_reply_token_count"] = token_count
    job["exact_reply_required_termination_headroom"] = required_termination_headroom
    job["exact_reply_available_termination_headroom"] = available_termination_headroom
    job["exact_reply_content_capacity_sufficient"] = bool(effective_native_cap >= token_count)
    job["exact_reply_termination_headroom_sufficient"] = bool(
        available_termination_headroom >= required_termination_headroom
    )
    job["exact_reply_native_capacity_sufficient"] = bool(effective_native_cap >= token_requirement)
    job["exact_reply_token_ceiling_valid"] = bool(
        hard_output_token_ceiling <= 0 or hard_output_token_ceiling >= token_requirement
    )
    if effective_native_cap < token_requirement:
        logger.info(
            "Exact target requires %d selected-tokenizer slots but the immutable "
            "generation envelope admits %d; deterministic exact-output repair owns "
            "the visible contract.",
            token_requirement,
            effective_native_cap,
        )


def _normalize_surface_format(response_text: Any) -> str:
    """Whitespace-only repair of jammed list markers and welded sentences."""
    text = str(response_text or "")
    if not text.strip():
        return ""
    try:
        from core.conversation.response_reliability import normalize_user_facing_format

        return normalize_user_facing_format(text)
    except (ImportError, RuntimeError, TypeError, ValueError) as exc:
        logger.debug("Surface format normalisation skipped: %s", exc)
        return ""


def _repair_live_user_surface_escaped_newlines(response_text: Any) -> str:
    """Turn literal \\n / \\t / \\r the model typed back into real whitespace.

    A local model that has read a lot of JSON sometimes emits the two-character
    sequence backslash-n where it meant a newline. The reply is otherwise fine,
    and rejecting it costs the person a correct answer over a typo the runtime
    can fix deterministically.

    LIVE DEFECT, 2026-07-26: a correct, well-structured marble derivation was
    rejected with reasons=escaped_control_artifact and the user got "I couldn't
    get to an answer I'd stand behind on that one".

    Prose only — a fenced code block may legitimately contain a literal \\n,
    and rewriting it would corrupt the code. So the fences are held out and
    the prose between them is repaired, rather than abandoning the whole reply
    because part of it is code.

    LIVE DEFECT, 2026-08-18: "write a python function to reverse a string and
    then explain how it works" took 112 seconds and returned "I couldn't get
    to an answer I'd stand behind on that one". The draft was rejected as
    escaped_control_artifact and this repair declined to run because the reply
    contained a code fence — so every request that wants code AND prose was
    unanswerable whenever the model typed one backslash-n in the explanation.
    """
    from core.conversation.escaped_controls import (
        repair_escaped_whitespace_artifacts,
    )

    repaired = repair_escaped_whitespace_artifacts(response_text)
    return repaired.strip() if repaired is not None else ""


def _repair_escaped_whitespace_in_prose(text: str) -> str | None:
    """Compatibility wrapper around the shared syntax-aware repair."""
    from core.conversation.escaped_controls import (
        repair_escaped_whitespace_artifacts,
    )

    return repair_escaped_whitespace_artifacts(text)


def _repair_live_user_surface_truncated_tail(response_text: Any) -> str:
    """Keep complete model-derived content when only the final tail is clipped."""

    text = str(response_text or "").strip()
    if len(text) < 80 or len(text.split()) < 12:
        return ""
    sentence_ends = [match.end() for match in re.finditer(r"[.!?](?=(?:\s|$|\d))", text)]
    for end in reversed(sentence_ends):
        candidate = text[:end].strip()
        if re.search(r"(?:^|\s)\d+\.$", candidate):
            continue
        if len(candidate) < 80 or len(candidate.split()) < 12:
            continue
        return candidate
    # A worked derivation is not sentences. Live 2026-07-26, the marble answer
    # came back as a bulleted derivation with no full stop after the opening
    # line, so sentence-based trimming found exactly one candidate ("Let's
    # break it down.", too short) and gave up — and a mostly complete, correct
    # answer became a refusal. When the body is line-structured, drop the
    # clipped final line and keep the complete ones.
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) >= 3:
        candidate = "\n".join(lines[:-1]).strip()
        if len(candidate) >= 80 and len(candidate.split()) >= 12:
            return candidate
    return ""


_LIVE_STATUS_CONCRETE_SIGNAL_INSTRUCTION = (
    "For live status questions, name at least one concrete observable runtime "
    "or sensory signal such as CPU/RAM pressure, temperature, network state, "
    "desktop access, screen/audio/camera state, heartbeat, Cortex/MLX worker "
    "state, or an actual numeric sensor reading. Avoid metaphor-only "
    "attention-texture language."
)


_SELF_CONDITION_SIGNAL_INSTRUCTION = (
    "This is a question about Aura's own condition. Answer directly from the "
    "supplied affect, welfare, felt-coherence, continuity, agency, and freshness "
    "evidence. CPU, RAM, host load, and availability are supporting body context "
    "only and must not replace the condition answer."
)


def _job_needs_concrete_status_signal_guidance(job: dict[str, Any]) -> bool:
    if not bool(job.get("clean_user_surface_contract", False)):
        return False
    prompt = _surface_validation_prompt(job)
    if not prompt:
        return False
    prompt_l = prompt.lower()
    if re.search(r"\b(?:capabilities|externally|tools?|what\s+can\s+you\s+do)\b", prompt_l):
        return False
    try:
        from core.conversation.response_reliability import (
            is_operational_status_turn,
            is_self_condition_turn,
            is_status_check_turn,
        )

        if is_self_condition_turn(prompt):
            return False
        if is_operational_status_turn(prompt) or is_status_check_turn(prompt):
            return True
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
        pass
    return any(
        marker in prompt_l
        for marker in (
            "live runtime signal",
            "live path",
            "runtime status",
            "with me",
            "you there",
        )
    )


def _with_initial_user_surface_guidance(
    messages: Any,
    prompt: Any,
    job: dict[str, Any],
) -> tuple[Any, Any]:
    # ``mlx_client`` deliberately gives health probes the clean-surface flag
    # so they inherit the safe steering/recurrent clamps. That flag describes
    # decode controls, not audience. Appending conversational guidance to the
    # readiness prompt changed ``Reply exactly: ready`` into two competing
    # instructions and made a healthy resident lane fail boot deterministically.
    # Keep control-plane measurements clamped, but never prompt-shape them as
    # user prose.
    if bool(job.get("health_probe", False)) or not _job_needs_concrete_status_signal_guidance(job):
        return messages, prompt
    guidance = _LIVE_STATUS_CONCRETE_SIGNAL_INSTRUCTION
    if isinstance(messages, list):
        guided_messages = copy.deepcopy(messages)
        for message in guided_messages:
            if isinstance(message, dict) and str(message.get("role") or "").lower() == "system":
                content = str(message.get("content") or "").rstrip()
                message["content"] = f"{content}\n{guidance}" if content else guidance
                return guided_messages, prompt
        guided_messages.insert(
            0,
            {"role": "system", "content": guidance},
        )
        return guided_messages, prompt
    prompt_text = str(prompt or "").rstrip()
    if not prompt_text:
        return messages, guidance
    return messages, f"{prompt_text}\n\n{guidance}"


def _repair_live_user_surface_operational_status(
    response_text: Any,
    rejection_reasons: list[str],
    job: dict[str, Any],
) -> str:
    if not rejection_reasons or not set(rejection_reasons).issubset(
        {"too_thin_for_operational_status_turn", "too_thin_for_status_turn"}
    ):
        return str(response_text or "")
    if not _job_needs_concrete_status_signal_guidance(job):
        return str(response_text or "")
    try:
        from core.runtime import resource_psutil as psutil

        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024**3)
        cpu_percent = psutil.cpu_percent(interval=None)
        return (
            "I am with you. One live runtime signal I can perceive is RAM "
            f"pressure at {memory.percent:.1f}% with {available_gb:.1f} GB "
            f"available; CPU load is {cpu_percent:.1f}% on this host."
        )
    except (ImportError, OSError, RuntimeError, TypeError, ValueError, AttributeError):
        load_1m: float | None = None
        try:
            from core.runtime.resource_observation import get_resource_observer

            load_1m = float(get_resource_observer().compute().load_1m)
            if not math.isfinite(load_1m):
                load_1m = None
        except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError):
            load_1m = None
        if load_1m is not None:
            return (
                "I am with you. One live runtime signal I can perceive is the host "
                f"load average at {load_1m:.2f}, with the Cortex/MLX worker active "
                "for this foreground turn."
            )
        # BOTH probes failed: fabricating a 0.00 load sample converted an
        # unavailable health probe into a confident live signal. Say what is
        # actually true instead.
        return (
            "I am with you. My host telemetry probes are not answering right "
            "now, so I cannot quote a live load number this turn — the reply "
            "lane itself is working, which is the one signal I can honestly "
            "attest."
        )


# What the model should actually DO about each rejection reason.
#
# A retry used to be handed the raw reason name and nothing else — "failed for:
# generic_memory_pin_acknowledgement" — which is an internal token, not an
# instruction. Measured live: the same draft was rejected three times with the
# identical reason AND the identical validation hash, burning the turn's whole
# budget on regenerating the same mistake, until "Request deadline reached at
# token 23" ended it and the person got a refusal. A retry that does not say
# what to fix is a wasted decode, and wasted decodes are what kill the turn.
_SURFACE_RETRY_INSTRUCTIONS: dict[str, str] = {
    "backend_symbolic_surface_leak": (
        "Restate the same substantive answer in natural language without raw backend "
        "enum names, internal variable names, or control identifiers."
    ),
    "corrupted_language": (
        "Regenerate the complete answer in clean grammatical language. Do not copy any "
        "malformed token from the rejected draft."
    ),
    "telemetry_path_wall": (
        "Answer the request without dumping internal telemetry paths. Include a path only "
        "when the user asked for that specific path and it is necessary to the answer."
    ),
    "unbounded_numeric_identifier": (
        "Do not expose an unexplained internal numeric identifier. Preserve a long exact "
        "number only when the user requested it or it is necessary to the answer."
    ),
    "generic_memory_pin_acknowledgement": (
        "The user asked you to remember something AND asked something else in the "
        "same turn. State the exact value you are keeping, then answer the rest of "
        "the turn in full — a bare acknowledgement is not a reply."
    ),
    "truncated_tail": (
        "End on a complete sentence. If the room is tight, say less and finish the "
        "thought rather than stopping mid-clause."
    ),
    "reliability_diagnostic_too_thin": (
        "Name the concrete mechanism — what happens, in what order, and what it "
        "costs — instead of reassurance."
    ),
    "too_thin_for_reliability_turn": (
        "Name the concrete mechanism and its consequence, not a summary judgement."
    ),
    "reliability_diagnostic_deflection": (
        "Do not deflect. Say what the actual cause or consequence is, plainly."
    ),
    "too_thin_for_user_turn": (
        "Say more than one clause: take a position and give the reason for it."
    ),
    "too_thin_for_open_ended_turn": (
        "This was an open question. Develop an actual thought rather than a line."
    ),
    "generic_assistant_language": (
        "Do not offer help, ask if there is anything else, or describe yourself as "
        "an assistant. Answer as yourself."
    ),
    "question_back_non_answer": (
        "Answer first, in your own words. A question back does not substitute for the answer."
    ),
    "low_signal_acknowledgement_placeholder": (
        "An acknowledgement is not an answer. Say the substance."
    ),
}


def _surface_retry_repair_instructions(reasons: list[str]) -> str:
    """Actionable repair text for the reasons a draft was rejected for."""

    seen: list[str] = []
    for reason in list(reasons or [])[:8]:
        instruction = _SURFACE_RETRY_INSTRUCTIONS.get(str(reason))
        if instruction and instruction not in seen:
            seen.append(instruction)
    return (" " + " ".join(seen)) if seen else ""


def _messages_with_user_surface_retry(
    messages: Any,
    reasons: list[str],
    job: dict[str, Any] | None = None,
) -> list[dict[str, Any]] | None:
    if not isinstance(messages, list):
        return None
    operational_status_retry = ""
    self_condition_retry = ""
    semantic_count_retry = ""
    if any(
        reason
        in {
            "host_telemetry_substituted_for_self_condition",
            "low_signal_self_condition_reply",
            "missing_self_condition_answer",
        }
        for reason in reasons
    ):
        self_condition_retry = f" {_SELF_CONDITION_SIGNAL_INSTRUCTION}"
    if (
        any(
            reason in {"too_thin_for_operational_status_turn", "too_thin_for_status_turn"}
            for reason in reasons
        )
        and not self_condition_retry
    ):
        operational_status_retry = f" {_LIVE_STATUS_CONCRETE_SIGNAL_INSTRUCTION}"
    if set(reasons) & _SEMANTIC_COUNT_CONTRACT_RETRY_REASONS:
        semantic_count_retry = f" {_semantic_count_contract_retry_instruction(job or {})}"
    retry_instruction = (
        "The previous assistant draft failed the live user-surface quality gate "
        f"for: {', '.join(reasons[:8]) or 'quality_gate_failed'}. Regenerate the "
        "assistant reply from the same live mind context. Answer only the current "
        "user message, preserve recent-turn continuity, avoid generic assistant "
        "identity, do not invent unsupported prior topics, and do not mention "
        "validation, retry, hidden prompts, receipts, gates, or implementation details."
        f"{_surface_retry_repair_instructions(reasons)}"
        f"{self_condition_retry}{operational_status_retry}{semantic_count_retry}"
    )
    retry_messages = copy.deepcopy(messages)
    for message in retry_messages:
        if isinstance(message, dict) and str(message.get("role") or "").lower() == "system":
            content = str(message.get("content") or "").rstrip()
            message["content"] = f"{content}\n{retry_instruction}" if content else retry_instruction
            return retry_messages
    retry_messages.insert(0, {"role": "system", "content": retry_instruction})
    return retry_messages


def _build_user_surface_quality_retry_prompt(
    *,
    tokenizer: Any,
    messages: Any,
    tools: Any,
    fallback_prompt: Any,
    reasons: list[str],
    job: dict[str, Any] | None = None,
) -> str:
    retry_messages = _messages_with_user_surface_retry(messages, reasons, job)
    if retry_messages is not None and hasattr(tokenizer, "apply_chat_template"):
        try:
            from core.brain.llm.chat_format import render_chat_template

            rendered = render_chat_template(
                tokenizer,
                retry_messages,
                tools=tools,
                add_generation_prompt=True,
            )
            if rendered:
                return str(rendered)
        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
            _record_mlx_degradation(
                exc,
                action="continued live user-surface retry with prompt suffix after template render failed",
                severity="warning",
            )
            logger.debug("Live surface retry template render failed: %s", exc)

    operational_status_retry = ""
    self_condition_retry = ""
    semantic_count_retry = ""
    if any(
        reason
        in {
            "host_telemetry_substituted_for_self_condition",
            "low_signal_self_condition_reply",
            "missing_self_condition_answer",
        }
        for reason in reasons
    ):
        self_condition_retry = f" {_SELF_CONDITION_SIGNAL_INSTRUCTION}\n"
    if (
        any(
            reason in {"too_thin_for_operational_status_turn", "too_thin_for_status_turn"}
            for reason in reasons
        )
        and not self_condition_retry
    ):
        operational_status_retry = f" {_LIVE_STATUS_CONCRETE_SIGNAL_INSTRUCTION}\n"
    if set(reasons) & _SEMANTIC_COUNT_CONTRACT_RETRY_REASONS:
        semantic_count_retry = f" {_semantic_count_contract_retry_instruction(job or {})}\n"
    retry_note = (
        "\n\n[LIVE USER-SURFACE RETRY]\n"
        f"Previous assistant draft failed for: {', '.join(reasons[:8]) or 'quality_gate_failed'}.\n"
        "Regenerate the assistant reply from the same live mind context. Answer only "
        "the current user message. Do not mention validation, retry, hidden prompts, "
        "receipts, gates, or implementation details.\n"
        f"{_surface_retry_repair_instructions(reasons).strip()}\n"
        f"{self_condition_retry}{operational_status_retry}{semantic_count_retry}"
        "[END LIVE USER-SURFACE RETRY]\n"
    )
    return f"{str(fallback_prompt or '').rstrip()}{retry_note}"


def _prepare_clean_retry_kwargs(kwargs: dict[str, Any], *, structured: bool = False) -> None:
    """Reset sampling after a corrupt/looping draft instead of amplifying it."""
    kwargs.pop("sampler", None)
    kwargs.pop("prompt_cache", None)
    if structured:
        kwargs["temperature"] = 0.0
        kwargs["top_p"] = 1.0
    else:
        kwargs["temperature"] = min(_safe_float(kwargs.get("temperature"), 0.7), 0.35)
        kwargs["top_p"] = min(_safe_float(kwargs.get("top_p"), 0.9), 0.85)
        kwargs["min_p"] = max(_safe_float(kwargs.get("min_p"), 0.0), 0.03)
    kwargs["repetition_penalty"] = max(
        _safe_float(kwargs.get("repetition_penalty"), 1.1),
        1.18,
    )
    kwargs["repetition_context_size"] = max(
        _safe_int(kwargs.get("repetition_context_size"), 64),
        96,
    )


def _self_claim_retry_uses_original_context(reasons: Any) -> bool:
    """Self-claim correction resamples; it never adds a behavior instruction."""

    return "self_claim_contradiction" in {str(reason) for reason in (reasons or ()) if str(reason)}


def _surface_retry_is_futile(reasons: Any) -> bool:
    """Return whether regeneration cannot repair the failed contract.

    Prompt provenance and self-claim verification are external integrity
    dependencies. Asking the model for another answer cannot restore either
    dependency, so retrying only adds latency and risks replacing useful text.
    """

    normalized = {str(reason) for reason in (reasons or ()) if str(reason)}
    return "self_claim_verification_unavailable" in normalized or any(
        reason.startswith("surface_validation_prompt_binding")
        or reason == "surface_validation_prompt_missing"
        for reason in normalized
    )


def _surface_retry_wall_exceeded(started_monotonic: float, wall_s: float) -> bool:
    """True when the user-surface gate-retry path has burned its wall budget.

    Under memory-contended decode each drafting attempt costs 30-70s; burning
    the full retry budget is how a single live turn reached 200s+ (July 8
    soak). Past the wall, exhaustion salvage delivers the best honest draft
    instead of drafting again for a user who has stopped waiting. Floor of
    10s so a misconfigured env value can never disable first-attempt retries.
    """
    if started_monotonic <= 0.0:
        return False
    return (time.monotonic() - started_monotonic) > max(10.0, wall_s)


def _ontology_retry_permitted(
    *,
    internal_attempt: int,
    max_internal_retries: int,
    ontology_retry_count: int,
    job_deadline_unix: float,
    user_surface: bool,
    surface_retry_started: float,
    surface_retry_wall_s: float,
    now_unix: float | None = None,
) -> tuple[bool, bool, bool]:
    """Allow one ontology repair only while caller time budgets remain open."""
    now = time.time() if now_unix is None else float(now_unix)
    deadline_open = job_deadline_unix <= 0.0 or now < job_deadline_unix
    retry_wall_open = not user_surface or not _surface_retry_wall_exceeded(
        surface_retry_started,
        surface_retry_wall_s,
    )
    allowed = (
        int(internal_attempt) < int(max_internal_retries)
        and int(ontology_retry_count) < 1
        and deadline_open
        and retry_wall_open
    )
    return allowed, deadline_open, retry_wall_open


def _expand_user_surface_retry_budget(
    kwargs: dict[str, Any],
    reasons: list[str],
    *,
    ceiling: int = 2048,
    hard_ceiling: Any = None,
) -> bool:
    """Give a clipped live reply one larger pass on the existing worker.

    This is deliberately limited to structural truncation. It does not create
    another model process, alter strict/proof contracts, or inflate retries for
    off-topic and low-quality drafts.
    """

    if "truncated_tail" not in set(reasons):
        return False
    current = max(
        _safe_int(kwargs.get("max_tokens"), 0),
        _safe_int(kwargs.get("num_predict"), 0),
    )
    if current <= 0:
        return False
    expansion_ceiling = max(current, int(ceiling))
    immutable_ceiling = _safe_int(hard_ceiling, 0)
    if immutable_ceiling > 0:
        expansion_ceiling = min(expansion_ceiling, immutable_ceiling)
    expanded = min(max(current * 2, current + 384), expansion_ceiling)
    if expanded <= current:
        return False
    kwargs["max_tokens"] = expanded
    if "num_predict" in kwargs:
        kwargs["num_predict"] = expanded
    return True


def _route_telemetry_sanitizer_draft(
    text: str,
    *,
    is_proof: bool,
    authored_surface_repair_available: bool,
) -> tuple[str, list[str]]:
    """Keep an unspeakable live draft only when a bounded repair lane owns it."""
    reasons = _telemetry_sanitization_failure_reasons(text, is_proof=is_proof)
    if reasons and not authored_surface_repair_available:
        return "", reasons
    return text, reasons


def _route_cooperative_partial_draft(
    job: dict[str, Any],
    text: str,
    surface_control_state: dict[str, Any],
    *,
    is_proof: bool,
) -> str:
    """Route a deadline/cancel partial through the same owned surface lane.

    Cooperative termination skips model retries, but it must not skip draft
    custody. A live user-surface draft that trips the telemetry sanitizer is
    carried as rejected evidence for the caller's bounded recovery path. A
    strict, proof, or non-user-surface draft has no such owner and remains
    withheld.
    """

    authored_surface_repair_available = _surface_quality_gate_enabled(job)
    routed, reasons = _route_telemetry_sanitizer_draft(
        text,
        is_proof=is_proof,
        authored_surface_repair_available=authored_surface_repair_available,
    )
    bounded_reasons = reasons[:8]
    surface_control_state["telemetry_sanitizer_reasons"] = bounded_reasons
    if bounded_reasons and authored_surface_repair_available:
        existing = surface_control_state.get("surface_quality_gate_reasons")
        merged = [
            str(reason).strip()[:120]
            for reason in (list(existing) if isinstance(existing, (list, tuple)) else [])
            if str(reason).strip()
        ]
        merged.extend(bounded_reasons)
        surface_control_state["surface_quality_gate_passed"] = False
        surface_control_state["surface_quality_gate_reasons"] = list(dict.fromkeys(merged))[:8]
        _remember_surface_quality_rejected_draft(
            surface_control_state,
            text,
            merged,
        )
    return routed


def _surface_rejected_draft_rank(text: Any, reasons: list[str]) -> tuple[int, ...]:
    """Rank suppressed drafts by servability evidence, never arrival time."""

    from core.conversation.surface_disposition import UNSPEAKABLE_REASONS

    body = str(text or "").strip()
    normalized = [str(reason or "").strip() for reason in reasons if str(reason or "").strip()]
    completion = {
        "final_answer_missing",
        "incomplete_code_response",
        "missing_final_answer",
        "truncated_tail",
        "unanswered_question_part",
    }
    unspeakable = sum(reason in UNSPEAKABLE_REASONS for reason in normalized)
    semantic = sum(reason not in completion for reason in normalized)
    incomplete = sum(reason in completion for reason in normalized)
    return (-unspeakable, -semantic, -incomplete, min(len(body), 8_000))


def _remember_surface_quality_rejected_draft(
    state: dict[str, Any],
    text: Any,
    reasons: list[str],
) -> None:
    """Keep the best rejected draft across worker-owned retries."""

    body = str(text or "").strip()[:8_000]
    if not body:
        return
    rank = _surface_rejected_draft_rank(body, reasons)
    current_rank = state.get("_surface_quality_rejected_rank")
    if not isinstance(current_rank, tuple) or rank > current_rank:
        state["_surface_quality_rejected_rank"] = rank
        state["surface_quality_rejected_text"] = body
        state["surface_quality_rejected_reasons"] = list(reasons)[:8]
