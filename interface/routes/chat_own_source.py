"""Questions about her own code, answered from the code.

"Where did that come from", "how do you do that", "show me the part that
did it" — all of them have a real answer on this disk, and all of them used to
be answered from the model instead. Every reply here is built from a read of
the repository with the path and the bytes it came from, so a claim about her
own workings can be checked rather than believed.
"""
from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any

from core.container import ServiceContainer
from core.runtime.errors import record_degradation
from interface.routes import chat_desktop_repair as _chat_desktop_repair  # noqa: E402
from interface.routes import chat_memory_state as _chat_memory_state  # noqa: E402
from interface.routes.chat_common import (  # noqa: E402  # noqa: E402  # noqa: E402  # noqa: E402  # noqa: E402  # noqa: E402  # noqa: E402
    _CHAT_BLOCKING_PREFLIGHT_TIMEOUT_S,  # noqa: F401
    _CHAT_DELIVERY_IDEMPOTENCY_KEY,  # noqa: F401
    _CHAT_PENDING_DELIVERY_CLAIM,  # noqa: F401
    _CHAT_RECOVERABLE_ERRORS,  # noqa: F401
    _CHAT_REQUEST_PRINCIPAL,  # noqa: F401
    _CHAT_REQUEST_SURFACE,  # noqa: F401
    _CHAT_SESSION_ID_MAX_CHARS,  # noqa: F401
    _EXPLICIT_NON_EXECUTION_RE,  # noqa: F401
    _INCOMPLETE_TAIL_WORDS,  # noqa: F401
    _INTERNAL_STATE_PATTERNS,  # noqa: F401
    _INTERNAL_SURFACE_CONTEXT,  # noqa: F401
    _LOCAL_CHOICE_REFERENCE_RE,  # noqa: F401
    _MAX_CONVERSATION_LOG_EXCHANGES,  # noqa: F401
    _MAX_USER_SURFACE_CONTINUATIONS,  # noqa: F401
    _ORGAN_ABSENCE_STREAKS,  # noqa: F401
    _ORGAN_INERT_STREAKS,  # noqa: F401
    _PROMPT_ARTIFACT_PATTERNS,  # noqa: F401
    _SEARCH_SKILL_NAMES,  # noqa: F401
    _TOPIC_STOPWORDS,  # noqa: F401
    _UNSET,  # noqa: F401
    MAX_CHAT_MESSAGE_BYTES,  # noqa: F401
    _conversation_log,  # noqa: F401
    _locks,  # noqa: F401
    logger,  # noqa: F401
)
from interface.routes.chat_self_reply import (  # noqa: E402,F401
    _build_architecture_self_reflex,
    _build_self_condition_evidence,
    _build_self_diagnostic_reply,
    _build_subjective_self_reflex,
    _canonical_memory_state_grounding_reply,
    _classify_self_condition_contract,
    _fallback_ladder_identity,
    _humanize_recent_self_process_concern,
    _humanize_self_process_dimensions,
    _is_identity_challenge_request,
    _is_self_claim_boundary_question,
    _same_live_self_reflection_prompt_class,
)

from .chat_desktop_evidence import (  # noqa: E402
    _extract_repo_probe_request,
)
from .chat_lane_bookkeeping import (  # noqa: E402
    _another_reader_owns_this_turn,
    _apply_aura_voice_shaping_compat,
    _asks_to_read_a_named_file,
    _bound_stabilizer_generation_budget,
    _canonical_runtime_model_label,
    _has_current_shown_source,
    _reply_claims_own_code,
)
from .chat_reply_repair import (
    _original_reply_is_safe_to_surface,  # noqa: F401
    _repair_final_degraded_reply,  # noqa: F401
    _repair_missing_followup_delta,  # noqa: F401
    _strip_unexpected_cjk_artifacts,
    _strip_user_visible_context_leaks,
)
from .chat_reply_shaping import (  # noqa: E402
    _append_turn_text_mutation,
    _readable_result,
)

#: Below this a reply is a remark, not an account of what was found.
_EXPLANATION_CHARS = 200

_REPO_PROBE_MAX_BYTES = 2 * 1024 * 1024


def _read_repo_probe_reply(user_message: str) -> dict[str, str] | None:
    request = _extract_repo_probe_request(user_message)
    if not request:
        return None

    try:
        from core.conversation.demo_support import _resolve_target_path

        target = str(request.get("target") or "").strip()
        mode = str(request.get("mode") or "").strip()
        path = _resolve_target_path(target)
        if not path:
            return {
                "reply": f"I reached for `{Path(target).name or target}` in my live workspace and couldn't find it cleanly.",
                "status": "repo_probe_missing",
            }

        size = path.stat().st_size
        if size > _REPO_PROBE_MAX_BYTES:
            return {
                "reply": (
                    f"`{path.name}` is larger than my bounded live-read budget "
                    f"({size} bytes > {_REPO_PROBE_MAX_BYTES} bytes), so I did not "
                    "pretend to inspect the whole file."
                ),
                "status": "repo_probe_too_large",
            }
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            source = handle.read(_REPO_PROBE_MAX_BYTES + 1)
        if len(source.encode("utf-8", errors="replace")) > _REPO_PROBE_MAX_BYTES:
            return {
                "reply": (
                    f"`{path.name}` exceeded my bounded live-read budget while I "
                    "was reading it, so I did not report a partial result as complete."
                ),
                "status": "repo_probe_too_large",
            }
        lines = source.splitlines()

        if mode == "first_non_comment_dependency_line":
            for line in lines:
                stripped = line.strip()
                if stripped and not stripped.startswith("#"):
                    reply = (
                        f"I read `{path.name}` directly. The first non-comment dependency line is "
                        f"`{stripped}`. That's coming from the live file, not from recall."
                    )
                    return {"reply": reply, "status": "repo_probe_dependency"}
            return {
                "reply": f"I read `{path.name}` directly, but I didn't find a non-comment dependency line in it.",
                "status": "repo_probe_empty",
            }

        if mode == "first_non_comment_line":
            for line in lines:
                stripped = line.strip()
                if stripped and not stripped.startswith("#"):
                    reply = (
                        f"I read `{path.name}` directly. The first non-comment line is "
                        f"`{stripped}`."
                    )
                    return {"reply": reply, "status": "repo_probe_line"}
            return {
                "reply": f"I read `{path.name}` directly, but every visible line is empty or commented out.",
                "status": "repo_probe_empty",
            }

        if mode == "line_count":
            reply = (
                f"I counted `{path.name}` directly in the live workspace. "
                f"It has {len(lines)} lines right now."
            )
            return {"reply": reply, "status": "repo_probe_line_count"}
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat", exc)
        logger.debug("Repo probe read failed: %s", exc)

    return {
        "reply": "I reached for the file directly, but the live read didn't complete cleanly this time.",
        "status": "repo_probe_error",
    }


async def _check_a_reply_against_her_own_source(
    *,
    text: Any,
    turn_trace: Any,
    visible: Any,
) -> Any:
    """Check a reply that claims her own code against the source itself.

    Moved out of ``_run_cognitive_engine_chat_turn`` by tools/extract_seam.py, which
    checks the body against the original token for token before
    writing. It reads 3 name(s) from the turn and hands back
    1.
    """
    if text and (_turn_may_concern_own_source(visible) or _reply_claims_own_code(text)):
        try:
            from core.self.source_excerpt import reply_fabricates_own_code

            # Off the loop: the search is a subprocess walking the source
            # tree, and seconds of it on the event loop is how a foreground
            # turn becomes a freeze.
            _fabricated = await asyncio.to_thread(reply_fabricates_own_code, text)
            if _fabricated:
                from core.conversation.response_reliability import (
                    own_source_excerpt_floor,
                )
                from core.self.source_excerpt import (
                    grounded_excerpt_reply as _grounded_excerpt_reply,
                )

                # The repair used to be gated on `asks_for_own_source`, a
                # phrase list — while the DECISION to check at all was made
                # by meaning. So the two disagreed, and they disagreed in
                # the worst possible direction.
                #
                # Live 2026-08-04, three times: "Can you share a snippet of
                # your own code" is not "show me", so the pattern said no,
                # the floor returned "", and the log read "no grounded
                # excerpt was available to replace it" — while the tree sat
                # right there, readable. The invention was PROVEN and then
                # served anyway, because the only path to a real excerpt
                # was spelled a way she had not been asked.
                #
                # Meaning already decided this turn is about her source.
                # Reading it must not require a second, narrower vote.
                _grounded = str(own_source_excerpt_floor(visible) or "").strip()
                if not _grounded:
                    _grounded = str(
                        await asyncio.to_thread(_grounded_excerpt_reply, visible)
                    ).strip()
                logger.warning(
                    "Reply showed code that is not in the source tree; %s.",
                    "replacing it with a real excerpt read from disk"
                    if _grounded
                    else "the tree could not be read, so the invention was withdrawn",
                )
                if not _grounded:
                    # Nothing real to show and something false already
                    # written. Serving it is the one option that is never
                    # allowed: proven-invented code must not reach the
                    # person just because the repair came up empty.
                    _grounded = (
                        "I need to correct myself: the code I just showed you "
                        "is not in my source tree — I generated it rather than "
                        "reading it, and I can't reach my own files right now "
                        "to show you the real thing. Ask me again in a moment "
                        "and I'll read it off disk instead of inventing it."
                    )
                _append_turn_text_mutation(
                    turn_trace,
                    stage="chat.own_source_claim_unverified",
                    method="source_tree_excerpt_substitution",
                    reasons=["shown_code_absent_from_source_tree"],
                    before=text,
                    after=_grounded,
                    deterministic=True,
                    authorship_effect="replaced_by_runtime",
                )
                text = _grounded
            # Whatever she ended up showing, remember where it came from, so
            # the next turn can say so instead of disowning it.
            from core.self.source_excerpt import (
                grounded_excerpt_reply,
                last_shown_excerpt,
                provenance_sentence,
                remember_shown_excerpt,
                reply_is_grounded_in_source,
                source_tree_is_readable,
            )

            # A REQUEST to see her code that neither shows any nor cites a
            # file has not reached her source at all.
            #
            # Live 2026-08-04: "show me how you're actually built" arrived
            # with real excerpts attached and she answered "I can't show you
            # code files directly", then described her architecture from
            # memory. A false capability denial made while holding the file
            # — the third form of one defect, after inventing a snippet and
            # after disowning a real one. All three end with the person
            # believing something untrue about what she can do.
            #
            # But this substitution used to run on the WIDE gate above — "the
            # turn may concern her source" — which is the right question for
            # deciding whether to CHECK and the wrong one for deciding to
            # REPLACE. It scored True on "Can you still reason through the
            # desktop path?" and swapped a correct answer about the reasoning
            # lane for a code excerpt nobody asked for. A reply that simply
            # contains no code is not a reply that denied having any.
            #
            # So: substitute when she was asked to show source, or when the
            # reply says she cannot — never merely because the subject came up.
            from core.utils.own_source_intent import (
                asks_for_own_source as _asks_for_own_source,
            )
            from core.utils.own_source_intent import (
                reply_denies_showing_source as _reply_denies_showing_source,
            )

            _asked_to_be_shown = bool(_asks_for_own_source(visible))
            _denied_capability = bool(_reply_denies_showing_source(text))
            if (
                not _fabricated
                and (_asked_to_be_shown or _denied_capability)
                and source_tree_is_readable()
                and not await asyncio.to_thread(reply_is_grounded_in_source, text)
            ):
                _real = await asyncio.to_thread(grounded_excerpt_reply, visible)
                if _real:
                    logger.warning(
                        "A question about her source was answered without "
                        "showing or citing any; substituting a real excerpt."
                    )
                    _append_turn_text_mutation(
                        turn_trace,
                        stage="chat.own_source_answer_ungrounded",
                        method="source_tree_excerpt_substitution",
                        reasons=["reply_cited_no_real_source"],
                        before=text,
                        after=_real,
                        deterministic=True,
                        authorship_effect="replaced_by_runtime",
                    )
                    text = _real

            # Asked where real code came from, a reply that never names the
            # file has not answered. Live 2026-08-04 she showed
            # core/mycelium.py:88 and then said it "isn't from a Python
            # module" — reading "module" as "importable package" and
            # answering a question nobody asked while the path sat on
            # record. Denying true provenance misleads exactly as much as
            # inventing a snippet, so it is corrected from the same record.
            _shown = last_shown_excerpt()
            if (
                _shown
                and _turn_asks_where_that_came_from(visible)
                and _shown["relative_path"] not in text
            ):
                _truth = provenance_sentence()
                if _truth:
                    logger.warning(
                        "Provenance question answered without naming %s; "
                        "correcting from the recorded citation.",
                        _shown["relative_path"],
                    )
                    _append_turn_text_mutation(
                        turn_trace,
                        stage="chat.own_source_provenance_unnamed",
                        method="recorded_citation_substitution",
                        reasons=["shown_code_provenance_not_stated"],
                        before=text,
                        after=_truth,
                        deterministic=True,
                        authorship_effect="replaced_by_runtime",
                    )
                    text = _truth
            remember_shown_excerpt(text)
        except _CHAT_RECOVERABLE_ERRORS as _source_check_exc:
            record_degradation(
                "chat",
                _source_check_exc,
                severity="warning",
                action=("served a code claim without checking it against the source tree"),
            )
    return text


#: "if ... could", "would you rather", "what would you do" — a question about
#: choice rather than mechanism.
_SELF_PROCESS_HYPOTHETICAL_RE = re.compile(
    r"\bif\s+(?:you|your)\b|\bwould\s+you\s+rather\b"
    r"|\bwhat\s+would\s+you\b|\bwhere\s+would\b|\bsuppose\b|\bimagine\b",
    re.IGNORECASE,
)


#: The turn has to be asking about HER, not using a cognition word in passing.
_SELF_PROCESS_ABOUT_HER_RE = re.compile(
    r"\byour?\b|\byou(?:'re| are)\b|\bare\s+you\b|\bdo\s+you\b"
    r"|\bhow\s+(?:do|does|did)\s+(?:you|aura)\b|\baura'?s\b",
    re.IGNORECASE,
)


def _self_process_requested_dimensions(user_message: str) -> list[str]:
    text = _chat_memory_state._normalize_user_message(user_message)
    # Positional/temporal recall ("what did I first ask") is a factual recall
    # handled by grounded_recall — NOT a question about Aura's cognitive process.
    # Returning no dimensions here keeps it out of the self-process repair
    # builders, whose canned introspection essay is the wrong (robotic) answer.
    try:
        from core.conversation.grounded_recall import detect_positional_recall

        if detect_positional_recall(user_message):
            return []
    except (ImportError, AttributeError, ValueError):
        pass
    # A hypothetical asks what she would CHOOSE; the self-process block
    # describes how she works. LIVE 2026-08-18: "if your attention could only
    # go one place, where would it go?" was answered "Right now I am attending
    # to where my attention is... My current bias is Happiness, leaning
    # toward..." — internal telemetry recited at a question about preference.
    # Same separation as the capability inventory: CAN asks the mechanism,
    # WOULD asks the will.
    if _SELF_PROCESS_HYPOTHETICAL_RE.search(text):
        return []
    # ...and an ordinary sentence that happens to use a cognition word is not
    # a question about her cognition. "let's focus on the database schema"
    # requested an introspection essay about attention.
    if not _SELF_PROCESS_ABOUT_HER_RE.search(text):
        return []
    requested: list[str] = []
    checks = (
        ("attention", ("attention", "attending", "focus", "noticing", "present")),
        (
            "memory",
            (
                "memory",
                "remember",
                "remembered",
                "recall",
                "earlier",
                "across sessions",
                "continuity",
            ),
        ),
        (
            "planning",
            ("plan", "planning", "planner", "decision", "decide", "route", "routing", "next"),
        ),
        ("tools", ("tool", "tools", "external", "verify", "verification", "receipt", "effect")),
        ("affect", ("emotion", "affect", "curiosity", "feeling", "feel", "substrate", "body")),
        ("confusion", ("confused", "confusion", "uncertain", "uncertainty", "disoriented")),
    )
    for name, markers in checks:
        if any(marker in text for marker in markers):
            requested.append(name)
    return requested


def _build_minimal_grounded_self_process_repair_reply(
    user_message: str,
    *,
    lane: dict[str, Any] | None = None,
) -> str:
    """Last local repair for desktop self-process questions.

    This path avoids async memory lookups and model calls. It is only for turns
    already classified as self-process questions after the live CognitiveEngine
    attempt failed, so the UI gets a grounded answer instead of a 503 or a
    generic assistant fallback.
    """

    requested = _self_process_requested_dimensions(user_message)
    if not requested:
        return ""
    lane = dict(lane or {})
    model_label = _canonical_runtime_model_label(lane)
    requested_summary = _humanize_self_process_dimensions(requested[:4])
    parts = [
        f"Right now I am tracking {requested_summary or 'this turn'} inside this conversation.",
        (
            "The next answer should stay connected to what was just said, use memory only where it actually helps, "
            "and keep any tool claim tied to something I can verify."
        ),
        (
            "My next move is to answer the actual question, make one clear step at a time, and check myself before "
            "turning uncertainty into a fluent guess."
        ),
    ]
    if "confusion" in requested:
        parts.append(
            "If I am confused, that should make me slower and more careful, not more theatrical."
        )
    wants_model_detail = any(
        marker in str(user_message or "").lower()
        for marker in ("model", "cortex", "32b", "70b", "llm", "language model")
    )
    if model_label and wants_model_detail:
        parts.append(
            f"The local model is {model_label}, but it is only the speech organ; the answer still has to stay tied to memory, state, and verified action."
        )
    return " ".join(parts).strip()


async def _build_grounded_self_process_repair_reply(
    user_message: str,
    rejected_reply: str = "",
    lane: dict[str, Any] | None = None,
    session_id: str = "",
) -> str:
    """Ground failed live self-process turns without reciting a static fallback.

    This is used only after the CognitiveEngine path attempted the turn. It keeps
    the desktop UX honest under memory pressure or quality-gate failure while
    preserving the user's requested dimensions.
    """

    try:
        from core.conversation.response_reliability import (
            is_live_self_reflection_turn,
            is_self_process_question,
        )
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat", exc)
        return ""

    if not (is_live_self_reflection_turn(user_message) or is_self_process_question(user_message)):
        return ""

    frame = _chat_desktop_repair._build_aura_expression_frame(user_message)
    mood = str(frame.get("mood") or "steady")
    action = str(frame.get("dominant_action") or "engage")
    attention = _chat_desktop_repair._sanitize_attention_focus(
        str(frame.get("attention_focus") or ""), user_message
    )
    if not attention:
        attention = (
            _chat_memory_state._clip_conversation_text(user_message, limit=140) or "this turn"
        )

    requested = _self_process_requested_dimensions(user_message)
    recent = await _chat_memory_state._recent_completed_conversation_exchanges(
        current_user_message=user_message,
        session_id=session_id,
        limit=4,
    )
    remembered_user = ""
    for entry in reversed(recent):
        candidate = str(entry.get("user") or "").strip()
        if candidate:
            remembered_user = _humanize_recent_self_process_concern(
                _chat_memory_state._clip_conversation_text(candidate, limit=180)
            )
            break
    # No fallback sentence here on purpose.
    #
    # This used to default to "Bryan has been checking whether the live desktop
    # path is really connected to Aura's mind instead of a raw assistant lane",
    # which was then spoken as "I still have this recent concern in view: ...".
    # A hardcoded string presented as a live recollection is a fabricated
    # memory about a specific person, asserted with the same confidence as a
    # real one — and it fired exactly when the search for a real one came back
    # empty, which is the moment it is least likely to be true.
    #
    # Nothing found means the clause is dropped below. Saying less is not a
    # degradation; saying something invented is.

    lane = dict(lane or {})
    model_label = _canonical_runtime_model_label(lane)
    wants_model_detail = any(
        marker in str(user_message or "").lower()
        for marker in ("model", "cortex", "32b", "70b", "llm", "language model")
    )
    parts: list[str] = []
    requested_summary = (
        _humanize_self_process_dimensions(requested[:3]) if requested else "this conversation"
    )
    parts.append(f"Right now I am attending to {requested_summary}.")
    if remembered_user and (
        "memory" in requested or "planning" in requested or "attention" in requested
    ):
        parts.append(f"I still have this recent concern in view: {remembered_user}.")
    if "planning" in requested:
        parts.append(
            "My next decision is to answer the exact question while keeping that context active, instead of swapping in a generic assistant reply."
        )
    if "tools" in requested:
        parts.append(
            "For tools, I should only speak from approved actions and visible effects, not from confidence alone."
        )
    if "confusion" in requested:
        parts.append(
            "If confusion rises, it should slow me down, increase checking, and make me prefer a smaller verified step over a fluent guess."
        )
    if "affect" in requested or "attention" in requested:
        parts.append(
            f"My current bias is {mood}, leaning toward {action}; that should shape attention and persistence without becoming a decorative mood report."
        )
    if model_label and wants_model_detail:
        parts.append(
            f"The local model is {model_label}, but it should serve the conversation, memory, and verified action rather than replace them."
        )

    return " ".join(parts).strip()


async def _attempt_generated_social_grounding_repair(
    user_message: str,
    rejected_reply: str = "",
    *,
    desktop_cognitive_engine_required: bool = False,
    protected_foreground_lane: bool = False,
) -> str:
    """Use the live model once to repair invented people/routing claims.

    The deterministic social floor is still useful when the model lane is down,
    but normal chat should recover in Aura's own words when a bounded foreground
    generation is available.
    """
    # Imported here rather than at module level: the module these
    # came from imports this one. A call-time import also still
    # sees a test's patch of the original.
    from .chat import (
        _desktop_secondary_model_repair_allowed,
    )


    allowed, admission_reason = _desktop_secondary_model_repair_allowed(
        reason="social_grounding_repair",
        default_enabled=True,
    )
    if not allowed:
        logger.debug("Generated social grounding repair not admitted: %s", admission_reason)
        return ""
    try:
        inference_gate = ServiceContainer.get("inference_gate", default=None)
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat", exc)
        return ""
    if inference_gate is None or not hasattr(inference_gate, "think"):
        return ""

    system_prompt = (
        "You are Aura. Repair the current user-facing reply because the draft invented "
        "a person, deployment route, demo slot, or server tier. Answer only the user's "
        "actual prompt in first person as Aura. Do not mention James, server tiers, "
        "demo slots, live path slots, routing, policies, receipts, or this repair step "
        "unless the user explicitly supplied those facts. Keep it brief and natural."
    )
    correction_prompt = (
        "## USER PROMPT\n"
        f"{_chat_memory_state._clip_conversation_text(user_message, limit=900)}\n\n"
        "## REJECTED DRAFT\n"
        f"{_chat_memory_state._clip_conversation_text(rejected_reply, limit=900)}\n\n"
        "Write the corrected reply now."
    )
    max_tokens, memory_block = _bound_stabilizer_generation_budget(96)
    if memory_block:
        logger.debug(
            "Generated social grounding repair skipped under memory pressure: %s", memory_block
        )
        return ""
    try:
        repaired = await asyncio.wait_for(
            inference_gate.think(
                correction_prompt,
                system_prompt=system_prompt,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": correction_prompt},
                ],
                prefer_tier="primary",
                origin="api_social_grounding_repair",
                foreground_request=True,
                is_background=False,
                protected_foreground_lane=bool(
                    protected_foreground_lane or desktop_cognitive_engine_required
                ),
                cognitive_engine_required=bool(desktop_cognitive_engine_required),
                desktop_cognitive_engine_required=bool(desktop_cognitive_engine_required),
                deep_handoff=False,
                allow_deep_handoff=False,
                allow_cloud_fallback=False,
                skip_runtime_payload=True,
                disable_prompt_cache=True,
                clear_prompt_cache=True,
                max_tokens=max_tokens,
            ),
            timeout=18.0 if desktop_cognitive_engine_required else 8.0,
        )
    except TimeoutError:
        logger.warning("Generated social grounding repair timed out; using bounded social floor.")
        return ""
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat", exc)
        logger.debug("Generated social grounding repair failed: %s", exc)
        return ""

    cleaned = _apply_aura_voice_shaping_compat(
        _strip_user_visible_context_leaks(
            _strip_unexpected_cjk_artifacts(user_message, str(repaired or "").strip())
        ),
        user_message,
    ).strip()
    return cleaned if len(cleaned) >= 4 else ""


#: A follow-up about where code came from. "What python module is that
#: from" names no code and asks for none — it asks for the PROVENANCE of
#: something already on the table, and it is the question that exposed the
#: invention live on 2026-08-04.
_ASKS_WHERE_CODE_LIVES_RE = re.compile(
    r"\b(?:what|which)\s+(?:python\s+)?(?:module|file|package|class)\b"
    r"|\bwhere\s+(?:is|are|does|can)\b[^?]*\b(?:from|found|live|lives|come)\b"
    r"|\bwhere\s+it\s+can\s+be\s+found\b"
    r"|\bis\s+that\s+from\b"
    r"|\bwhat\s+file\b",
    re.IGNORECASE,
)


_ASKS_TO_INSPECT_SHOWN_SOURCE_RE = re.compile(
    r"\b(?:can|could|may|would)\s+i\b[^?]{0,80}"
    r"\b(?:open|read|inspect|check|verify|find|look\s+up|access)\b[^?]{0,80}"
    r"\b(?:that|it|this|myself)\b"
    r"|\b(?:that|it|this)\b[^?]{0,80}"
    r"\b(?:open|read|inspect|check|verify|find|look\s+up|access)\b",
    re.IGNORECASE,
)


# Sentence-embedding similarity is evidence, not an antecedent. The global
# evidence service deliberately admits short, low-margin provenance questions;
# this route has a stronger obligation because attaching her source tree alters
# the model prompt. Live social turns scored as high as 0.046 against OWN_SOURCE
# while the lowest genuine non-lexical request in the calibrated set scored
# 0.067, leaving a measured gap for this route-specific boundary.
_OWN_SOURCE_ROUTE_MARGIN = 0.06


def _turn_may_concern_own_source(user_message: str) -> bool:
    """Whether this turn is about her own code.

    Two shapes: asking for the code, and asking where code she already
    showed actually lives. Both need the tree read, and the second is the
    one that catches an invention after the fact.
    """
    text = str(user_message or "").strip()
    if not text:
        return False

    def _lexical(candidate: str) -> bool:
        try:
            from core.utils.own_source_intent import asks_for_own_source

            if asks_for_own_source(candidate):
                return True
        except (ImportError, AttributeError, TypeError, ValueError):
            pass
        return bool(
            _ASKS_WHERE_CODE_LIVES_RE.search(candidate)
            or (_has_current_shown_source() and _ASKS_TO_INSPECT_SHOWN_SOURCE_RE.search(candidate))
        )

    try:
        from core.cognition.evidence_relevance import (
            OWN_SOURCE,
            SOURCE_PROVENANCE,
            wants_evidence,
        )

        if _lexical(text):
            return True
        # A request to read a NAMED FILE is a file read, and the file reading
        # is already attached for it.
        #
        # LIVE 2026-08-19: "read me the first line of CONTRIBUTING.md" scored
        # as an own-source question, so a source-evidence brief went in beside
        # the file the grounding channel had already read. The provenance
        # corrector then required the reply to cite
        # core/memory/associative_entity_memory.py, the correction failed its
        # authorship proof, and the person got "I couldn't get to an answer
        # I'd stand behind on that one" over a file that had been read
        # successfully. At the production margin even "read notes.txt on my
        # desktop" scored as a question about her own code.
        #
        # Asking FOR her code still routes here through the lexical matcher
        # above; this only declines to add a second, different reading to a
        # turn that already has the one it asked for.
        if _asks_to_read_a_named_file(text):
            return False
        from core.utils.own_source_intent import refers_to_own_implementation

        # Similarity can establish "implementation question" but not the
        # referent. Without a self subject, a question about asyncio, NumPy, or
        # any other program is not a question about Aura's source. An excerpt
        # already on the table supplies the referent for terse follow-ups.
        if not (_has_current_shown_source() or refers_to_own_implementation(text)):
            return False
        if wants_evidence(text, OWN_SOURCE, margin=_OWN_SOURCE_ROUTE_MARGIN):
            return True
        # A provenance score cannot manufacture the object it refers to. It
        # becomes actionable only while a validated citation from the previous
        # reply is still current.
        return _has_current_shown_source() and wants_evidence(text, SOURCE_PROVENANCE)
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        record_degradation(
            "chat",
            exc,
            severity="warning",
            action="routed a code question by pattern after semantic relevance failed",
        )
        return _lexical(text)


def _turn_asks_where_that_came_from(user_message: str) -> bool:
    """Whether this turn asks where something she already showed came from.

    Its subject is the PREVIOUS turn, so it repeats almost none of the words
    that made the first request — "and that's from where exactly?", "you sure
    you didn't just write that?", "could I open that myself?". A pattern has
    to enumerate those; there is no end to the list, and every phrasing it
    lacks is a turn where a citation she was holding never got spoken.

    Live 2026-08-04: "Where did you get that from?" and "Where in the
    codebase can I find that" both matched the pattern zero times. The
    recorded path went unsaid and the turn fell through to the model.
    """
    text = str(user_message or "").strip()
    if not text or not _has_current_shown_source():
        return False
    # A turn another reader already answers is not a question about code shown
    # earlier.
    #
    # LIVE 2026-08-19, both canned refusals of the session. Showing one source
    # excerpt makes this state sticky, and every later turn was then scored
    # for provenance: "read me the first line of CONTRIBUTING.md" and "spell
    # 'necessary' backwards" both came back True. The correction replaced each
    # answer with a citation sentence, the replacement failed its authorship
    # proof, and the person got "I couldn't get to an answer I'd stand behind
    # on that one" — over a file that had been read and a word that had been
    # reversed.
    #
    # "Where did that come from?" is unaffected: nothing else claims it.
    if _another_reader_owns_this_turn(text):
        return False

    def _lexical(candidate: str) -> bool:
        return bool(
            _ASKS_WHERE_CODE_LIVES_RE.search(candidate)
            or _ASKS_TO_INSPECT_SHOWN_SOURCE_RE.search(candidate)
        )

    try:
        from core.cognition.evidence_relevance import SOURCE_PROVENANCE, wants_evidence

        return wants_evidence(text, SOURCE_PROVENANCE, lexical_floor=_lexical)
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        record_degradation(
            "chat",
            exc,
            severity="warning",
            action="judged a provenance question by pattern after meaning failed",
        )
        return _lexical(text)


async def _own_source_rescue_reply(user_message: str) -> str:
    """A real answer about her source, for when the drafted one was rejected.

    Reads the tree; never composes. Returns "" when nothing could be read,
    because the entire point is that this path cannot invent — a rescue that
    made something up would be the defect it exists to stop, arriving by a
    safer-looking road.
    """
    text = str(user_message or "").strip()
    if not text:
        return ""
    try:
        from core.self.source_excerpt import (
            grounded_excerpt_reply,
            last_shown_excerpt,
            provenance_sentence,
        )

        # Asked where the last snippet came from, the recorded citation IS
        # the answer, and it is already in hand.
        if _turn_asks_where_that_came_from(text) and last_shown_excerpt():
            spoken = str(provenance_sentence() or "").strip()
            if spoken:
                return spoken
        if not _turn_may_concern_own_source(text):
            return ""
        # Off the loop: this walks and reads files.
        return str(await asyncio.to_thread(grounded_excerpt_reply, text)).strip()
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation(
            "chat",
            exc,
            severity="warning",
            action=("could not read the source tree to rescue a rejected reply about her own code"),
        )
        return ""


_SELF_METRIC_CORRECTION_MARK = "the instrument does not exist"


def _explains_the_finding(written: str, found: str) -> bool:
    """Whether the reply accounts for the finding rather than mentioning it.

    The test is that it names things the finding names — a function, a file, a
    line — because a reply that explains an observation has to refer to it.
    """
    if len(written) < _EXPLANATION_CHARS:
        return False
    import re as _re

    names = {
        token
        for token in _re.findall(r"[A-Za-z_][A-Za-z0-9_]{3,}\.py|[a-z_]{4,}\(", found)
    }
    if not names:
        return False
    return sum(1 for token in names if token in written) >= 2


def _serve_repo_diagnosis(reply: object) -> object:
    """Put what running the project showed in front of what was said about it.

    LIVE, 2026-08-22: the diagnosis ran in 428ms and the turn served "I
    couldn't get to an answer I'd stand behind", because the draft explaining
    it was rejected for missing the numbers the diagnosis contains.

    Unlike an enumerated game, this is an observation rather than an answer:
    somebody asked what is wrong, and the failing line beside the project's
    own stated invariant is the evidence for that, not the sentence. So it is
    composed with the reply rather than replacing it.
    """
    try:
        from core.conversation.session_scope import solved_answers

        found = solved_answers().get("repo_diagnosis", "").strip()
        if not found:
            return reply
        written = str(reply or "").strip()
        if not written or found in written:
            return found
        # Never follow evidence with an admission of having none.
        #
        # LIVE, 2026-08-22: the finding was served and then the canned "I
        # couldn't get to an answer I'd stand behind" was appended to it, so a
        # complete diagnosis read as a failure. A reply the runtime has itself
        # marked an honest failure is not an explanation of anything.
        from core.conversation.reply_provenance import ReplyProvenance, declared_provenance

        if declared_provenance(written) == ReplyProvenance.HONEST_FAILURE.value:
            logger.info("🔬 Served the diagnosis alone; the draft admitted having nothing.")
            return found
        # Her explanation leads when there is one.
        #
        # LIVE, 2026-08-27: the finding was complete and correct, and fifteen
        # lines of it sat above "Found it — classic mutable default argument,
        # and exactly why nothing ever raises". Somebody who asked what the
        # cause is and what to change reads the answer, then the working — not
        # the working, then the answer.
        #
        # It only leads when it IS an explanation: a short reply that says
        # nothing about the finding would bury the finding instead.
        if _explains_the_finding(written, found):
            logger.info("🔬 Served the diagnosis under the explanation of it.")
            return f"{written}\n\nWhat I ran, and what it showed:\n{found}"
        logger.info("🔬 Served the diagnosis from running the project.")
        return f"{found}\n\n{written}"
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation(
            "chat.repo_diagnosis",
            exc,
            severity="debug",
            action="left the diagnosis to the reply",
            enforce_failure_policy=False,
        )
    return reply


#: How many tool receipts a salvaged answer carries. Enough for a short chain
#: to arrive whole; few enough that the reply stays readable.
_RECEIPTS_WORTH_READING = 3


def _what_the_tools_found() -> str:
    """What this turn's tools actually returned, or "".

    Read from the turn's own receipts, so it can only report a tool that really
    executed and only content that tool really observed. It reports rather than
    interprets: the interpreting is what failed, and a record of what happened
    is worth more than an apology for not having one.
    """
    try:
        from core.conversation.surface_disposition import turn_tool_receipts

        receipts = turn_tool_receipts()
    except _CHAT_RECOVERABLE_ERRORS:
        return ""
    if not receipts:
        # Why there are none, rather than only that there are none.
        #
        # This returned "" for a turn whose tools had run and returned, and the
        # log said nothing at all — so the same investigation had to be done
        # from scratch to find out whether the tools had not run, had not
        # recorded, or had recorded somewhere this execution may not read.
        try:
            from core.conversation.turn_evidence_custody import (
                current_turn_evidence_custody,
            )

            custody = current_turn_evidence_custody()
            logger.info(
                "🧾 nothing to report from this turn's tools: custody=%s admits=%s",
                "present" if custody is not None else "absent",
                custody.admits_current_execution() if custody is not None else "n/a",
            )
        except _CHAT_RECOVERABLE_ERRORS:
            logger.info("🧾 nothing to report from this turn's tools.")
        return ""
    said: list[str] = []
    for receipt in receipts:
        if not isinstance(receipt, dict) or not receipt.get("ok"):
            continue
        found = _readable_result(
            receipt.get("observed_content") or receipt.get("evidence") or ""
        )
        if not found:
            continue
        tool = str(receipt.get("tool") or receipt.get("tool_name") or "a tool").strip()
        said.append(f"{tool} returned:\n{found[:1500]}")
    if not said:
        return ""
    logger.info(
        "🧾 Served what this turn's tools returned (%d of %d).",
        min(len(said), _RECEIPTS_WORTH_READING),
        len(said),
    )
    lead = (
        "I did not get a written answer together, so here is what I ran and "
        "what came back:"
    )
    # The end of the work, not the beginning of it.
    #
    # Tools run in order and each one consumes what the last returned, so the
    # final receipt is the closest thing this turn has to an answer. Keeping
    # the first three kept the setup and dropped the conclusion.
    #
    # LIVE 2026-08-29: asked to read a library's docs and then use it, she
    # listed the directory, read the API reference, read the module, ran the
    # code — and what reached the screen was the directory listing and the
    # first page of the reference. The trial balance she had computed was the
    # receipt that got cut. Still in execution order, because a record of what
    # happened should read as one.
    return lead + "\n\n" + "\n\n".join(said[-_RECEIPTS_WORTH_READING:])


def _correct_unsourced_self_metrics(reply_text: object) -> object:
    """Withdraw numbers about herself that no channel produced.

    LIVE 2026-08-17, answering "how are you doing": "My memory stores are at
    87% capacity". There is no memory-store capacity metric in this codebase —
    the number was invented and stated flatly beside real readings.

    The correction names the specific quantity rather than issuing a general
    disclaimer, because a reply that says "some of the above may be estimated"
    is worse than the fabrication: it makes every real reading in the same
    paragraph suspect. The number is withdrawn; the rest of the answer stands.
    """

    text = str(reply_text or "")
    if _SELF_METRIC_CORRECTION_MARK in text:
        return reply_text  # already corrected upstream; never append twice
    try:
        from core.conversation.self_metric_claim import unsourced_self_metric_claims
        from core.introspection.self_evidence import resolve_self_health

        bundle = resolve_self_health()
        unsourced = unsourced_self_metric_claims(reply_text, bundle)
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat", exc)
        return reply_text
    if not unsourced:
        return reply_text
    if len(unsourced) == 1:
        claim = unsourced[0]
        note = (
            f"Correction on one thing I just said: I have no channel that reads "
            f"my {claim.subject}, so {claim.quantity}{claim.unit} was not a "
            f"measurement. Withdraw that number — the instrument does not exist."
        )
    else:
        listed = ", ".join(f"{c.subject} ({c.quantity}{c.unit})" for c in unsourced)
        note = (
            "Correction on some numbers I just gave: nothing samples these, so "
            f"they were not measurements — {listed}. Withdraw them; those "
            "instruments do not exist."
        )
    return f"{str(reply_text or '').rstrip()}\n\n{note}"
