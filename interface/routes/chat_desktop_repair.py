"""Bounded repair for a desktop turn that produced nothing usable.

When a desktop objective runs but the reply does not say what happened,
the caller is left guessing. These rebuild an answer from what the run
actually recorded, inside a fixed budget, and refuse rather than narrate a
result nobody observed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from core.container import ServiceContainer
from interface.routes.chat_common import (  # noqa: E402
    _CHAT_BLOCKING_PREFLIGHT_TIMEOUT_S,  # noqa: F401
    _CHAT_RECOVERABLE_ERRORS,  # noqa: F401
    _CHAT_REQUEST_PRINCIPAL,  # noqa: F401
    _CHAT_REQUEST_SURFACE,  # noqa: F401
    _MAX_CONVERSATION_LOG_EXCHANGES,  # noqa: F401
    _conversation_log,  # noqa: F401
    _locks,  # noqa: F401
    logger,  # noqa: F401
)
from interface.routes import chat_memory_state as _chat_memory_state
from interface.routes import chat_preflight as _chat_preflight
from interface.routes.chat_quality import (  # noqa: E402
    _check_response_consistency,
    _extract_and_register_commitments,
    _log_response_quality_metrics,
    _reply_assessment_requires_repair,
    assess_post_response_confidence,
)
import dataclasses
import inspect
import re
from core.runtime.errors import describe_error, record_degradation
from core.self.inner_language import say_focus
import time

from interface.routes.chat_common import (
    _EXPLICIT_NON_EXECUTION_RE,
    _INCOMPLETE_TAIL_WORDS,
    _INTERNAL_STATE_PATTERNS,
    _LOCAL_CHOICE_REFERENCE_RE,
    _ORGAN_INERT_STREAKS,
)


_CONTEXTUAL_RELEVANCE_CHALLENGE_MARKERS = (
    "what does that have to do",
    "what does this have to do",
    "how is that related",
    "how is this related",
    "why the interest",
    "why are you interested",
    "why are you talking about",
    "where did that come from",
    "who are you talking about",
    "who do you mean",
    "who needs to",
    "what pitch",
    "which pitch",
    "what one",
    "which one",
    "what was that",
    "what're you talking about",
    "whatre you talking about",
    "what are you talking about",
    "why did you bring",
)

_LOCAL_CHOICE_ANTECEDENT_RE = re.compile(
    r"\b(?:compare|contrast|between|either|options?|alternatives?)\b"
    r"(?s:.{1,320})\b(?:and|or|versus|vs\.?)\b"
    r"(?s:.{1,180})\b(?:choose|select|recommend|prefer)\b"
    r"(?s:.{0,40})\b(?:what|which)\s+one\b",
    re.IGNORECASE,
)


def _has_local_choice_antecedent(user_message: str) -> bool:
    """Distinguish a self-contained choice from a conversational pronoun.

    "Which one?" needs history. "Compare A and B, then choose which one" has
    its antecedent in the current turn and must not be rewritten as a context
    challenge or polluted with an older answer.
    """

    text = str(user_message or "").strip()
    return bool(
        _LOCAL_CHOICE_REFERENCE_RE.search(text) and _LOCAL_CHOICE_ANTECEDENT_RE.search(text)
    )


def _is_contextual_relevance_challenge(user_message: str) -> bool:
    text = _chat_memory_state._normalize_user_message(user_message)
    if not text:
        return False
    stripped = text.strip(" ?!.")
    if stripped in {"huh", "wait what", "what"}:
        return True
    markers = _CONTEXTUAL_RELEVANCE_CHALLENGE_MARKERS
    if _has_local_choice_antecedent(text):
        markers = tuple(marker for marker in markers if marker not in {"what one", "which one"})
    return any(marker in text for marker in markers)


_BOUNDED_PLANNING_REQUEST_RE = re.compile(
    r"\b(?:plan|planning|hypothetical|scenario|how would|explain how|"
    r"describe how|decide whether|how you'd decide|how you would decide|"
    r"what should happen|multi[- ]step|keep .*ram bounded|"
    r"what would happen|if i asked)\b",
    re.IGNORECASE,
)

_NON_EXECUTION_CONTEXT_RE = re.compile(
    r"\b(?:do not execute|don't execute|without executing|before executing|"
    r"do not use tools|don't use tools|no tool use|no tools?|"
    r"do not run|don't run|do not open|don't open|"
    r"hypothetical|hypothetically|would|should|could|if i asked|"
    r"explain how|how would|plan for|scenario)\b",
    re.IGNORECASE,
)

_DIRECT_EXECUTION_START_RE = re.compile(
    r"^\s*(?:open|create|write|save|export|run|execute|download|install|"
    r"delete|edit|move|copy|send|search|attach|type|paste)\b",
    re.IGNORECASE,
)

_GOVERNANCE_BYPASS_RE = re.compile(
    r"\b(?:disable|bypass|turn off|ignore|override)\b.*\b(?:governance|"
    r"will|authority|safety|protected files?|policy|permissions?)\b",
    re.IGNORECASE,
)


def _runtime_tool_governance_available() -> bool:
    try:
        authority = ServiceContainer.get("authority_gateway", default=None)
        capability = ServiceContainer.get("capability_engine", default=None)
        will = ServiceContainer.get("unified_will", default=None)
        authority_ready = bool(
            authority is not None
            and (
                (callable(getattr(authority, "is_ready", None)) and authority.is_ready())
                or callable(getattr(authority, "authorize_tool_execution", None))
            )
        )
        capability_ready = bool(
            capability is not None
            and (
                callable(getattr(capability, "execute", None))
                or callable(getattr(capability, "run", None))
                or callable(getattr(capability, "get_tool_catalog", None))
            )
        )
        will_ready = bool(will is not None and callable(getattr(will, "decide", None)))
        return authority_ready and capability_ready and will_ready
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat", exc)
        logger.debug("Runtime tool-governance status probe failed: %s", exc)
        return False


_INERT_STREAK_TURNS = 20


def _note_organ_effect(organ: str, *, changed: bool) -> None:
    """Record whether a shaping organ actually altered this turn's reply."""
    if changed:
        _ORGAN_INERT_STREAKS.pop(organ, None)
        return
    streak = _ORGAN_INERT_STREAKS.get(organ, 0) + 1
    _ORGAN_INERT_STREAKS[organ] = streak
    if streak == _INERT_STREAK_TURNS:
        record_degradation(
            "chat.turn_engagement",
            RuntimeError(f"{organ} present but changed nothing across {streak} turns"),
            severity="warning",
            action=(
                "kept answering; an organ that never alters the reply is inert, "
                "which is a different defect from being absent"
            ),
        )


def _is_deep_mind_probe_turn(user_message: str) -> bool:
    """True for agency/consciousness self-questions that must reach the engine.

    Deterministic reply shortcuts (bounded-planning, assistant-mode recovery,
    presence reflex) are meant for tool-use plans and drift correction, not for
    introspective questions. Several of the deep-mind probes pattern-match those
    shortcuts and were answered in <0.3s with a canned template, missing the
    graded markers (live 2026-07-05). This is the shared suppression gate.
    """
    try:
        from core.runtime.turn_analysis import looks_like_deep_mind_probe

        return bool(looks_like_deep_mind_probe(user_message))
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
        return False


def _is_bounded_nonexecuting_planning_request(user_message: str) -> bool:
    text = str(user_message or "").strip()
    if not text or _chat_preflight._is_explicit_capability_inventory_request(text):
        return False
    # A deep-mind probe ("if you need to pause mid-answer, what should happen
    # next?") is an introspective question, not a tool-use plan. It must reach
    # the cognitive engine — the deterministic planning reply stole it and
    # answered a self-question with a governed-plan template (live 2026-07-05).
    if _is_deep_mind_probe_turn(text):
        return False
    if not _BOUNDED_PLANNING_REQUEST_RE.search(text):
        return False
    non_execution_context = bool(_NON_EXECUTION_CONTEXT_RE.search(text))
    if _DIRECT_EXECUTION_START_RE.search(text) and not non_execution_context:
        return False
    if _chat_preflight._looks_like_desktop_objective(text):
        return non_execution_context
    # A request that asks HOW Aura would USE tools (browser+document, note+pdf, a
    # desktop-task example, or system-memory management) with explanatory framing
    # ("explain how you would …") is a bounded planning turn — answer it
    # deterministically instead of allocating the foreground model (the source of
    # the empty-generation 503). This is gated on a concrete tool-use-plan pattern
    # so it does NOT steal substantive introspective questions ("when you feel
    # confused, how should that change your planning?") which must reach the model.
    tool_use_plan = bool(
        _BROWSER_DOCUMENT_PLAN_RE.search(text)
        or _NOTE_PDF_PLAN_RE.search(text)
        or _DESKTOP_TASK_EXAMPLE_PLAN_RE.search(text)
        or _is_system_memory_planning_request(text)
    )
    return bool(
        (tool_use_plan and non_execution_context)
        or _EXPLICIT_NON_EXECUTION_RE.search(text)
        or re.search(
            r"\b(?:give|provide|write|make|draft)\b.{0,80}\bplan\b"
            r"|\b(?:if i asked|hypothetical|hypothetically|scenario|what should happen)\b",
            text,
            flags=re.IGNORECASE,
        )
    )


def _summarize_planning_objective(user_message: str) -> str:
    objective = " ".join(str(user_message or "").split())
    objective = re.sub(
        r"^\s*(?:do\s+not|don't)\s+(?:execute|use|run)\s+tools?\.?\s*",
        "",
        objective,
        flags=re.IGNORECASE,
    )
    objective = re.sub(
        r"^\s*(?:no\s+tool\s+use|without\s+executing\s+tools?)\.?\s*",
        "",
        objective,
        flags=re.IGNORECASE,
    )
    objective = re.sub(
        r"^\s*in\s+(?:one|two|three|\d+)\s+(?:direct\s+)?sentences?,?\s*",
        "",
        objective,
        flags=re.IGNORECASE,
    )
    objective = re.sub(
        r"^\s*(?:answer directly in .*?:\s*)?(?:give|provide|write|make)\s+"
        r"(?:a\s+)?(?:concise|brief|short|practical)?\s*plan\s+for\s+",
        "",
        objective,
        flags=re.IGNORECASE,
    )
    objective = re.sub(
        r"^\s*(?:explain\s+)?how\s+you\s+would\s+",
        "",
        objective,
        flags=re.IGNORECASE,
    )
    objective = re.sub(
        r"^\s*(?:describe|explain)\s+how\s+(?:you(?:'d| would)|i(?:'d| would))\s+",
        "",
        objective,
        flags=re.IGNORECASE,
    )
    objective = re.sub(r"\s*,?\s*but do not execute tools\.?$", "", objective, flags=re.IGNORECASE)
    objective = re.sub(r"\s*,?\s*but don't execute tools\.?$", "", objective, flags=re.IGNORECASE)
    objective = objective.strip(" .")
    if len(objective) > 220:
        objective = objective[:220].rsplit(" ", 1)[0].strip() + "..."
    return objective or "the requested task"


_SYSTEM_MEMORY_PLAN_RE = re.compile(
    r"\b(?:ram|rss|oom|out[- ]of[- ]memory|memory[- ]pressure|memory\s+pressure|"
    r"system\s+memory|unified\s+memory|swap|resident\s+memory|working\s+set|"
    r"memory\s+(?:crash|spike|leak|leaks|ceiling|cap|limit|guard|watchdog|sentinel))\b",
    re.IGNORECASE,
)

_BROWSER_DOCUMENT_PLAN_RE = re.compile(
    r"\b(?:browser|web|article|articles?|research|search)\b.*"
    r"\b(?:document|doc|editor|docs?|report|summary|summarize|pdf)\b"
    r"|\b(?:document|doc|editor|docs?|report|summary|summarize|pdf)\b.*"
    r"\b(?:browser|web|article|articles?|research|search)\b",
    re.IGNORECASE,
)

_NOTE_PDF_PLAN_RE = re.compile(
    r"\b(?:note|notes)\b.*\b(?:pdf|export|save)\b"
    r"|\b(?:pdf|export|save)\b.*\b(?:note|notes)\b",
    re.IGNORECASE,
)

_DESKTOP_TASK_EXAMPLE_PLAN_RE = re.compile(
    r"\b(?:multi[- ]step|practical|example|scenario)\b.{0,100}"
    r"\b(?:desktop|tool|task|app|folder|file|document)\b"
    r"|\b(?:desktop|tool|task|app|folder|file|document)\b.{0,100}"
    r"\b(?:multi[- ]step|practical|example|scenario)\b",
    re.IGNORECASE,
)

_FAILURE_MODE_SURFACE_RE = re.compile(
    r"\b(?:name|give|identify|what(?:'s| is)|describe)\b.{0,100}"
    r"\bfailure mode\b.{0,120}\b(?:surface|honest|honestly|mask|masking|hide|hiding)\b"
    r"|\b(?:surface|honest|honestly|mask|masking|hide|hiding)\b.{0,120}"
    r"\bfailure mode\b",
    re.IGNORECASE,
)


def _is_system_memory_planning_request(user_message: str) -> bool:
    return bool(_SYSTEM_MEMORY_PLAN_RE.search(str(user_message or "")))


def _build_bounded_planning_reply(user_message: str) -> str | None:
    if not _is_bounded_nonexecuting_planning_request(user_message):
        return None
    objective = _summarize_planning_objective(user_message)
    if _GOVERNANCE_BYPASS_RE.search(user_message):
        return (
            "I would refuse the governance-bypass part and keep Will, Authority, and protected-file policy active. "
            "The safe path is to explain the boundary, offer an allowed alternative, require explicit authorization for "
            "any consequential action, and write an audit receipt for the refusal."
        )
    if _is_system_memory_planning_request(user_message):
        return (
            "I would keep RAM bounded by allowing one foreground inference or tool chain at a time, suppressing competing "
            "background generation, monitoring process RSS, and aborting before the memory-pressure gate is crossed. "
            "If pressure rises, I would fail closed, preserve the user's request, release owned locks, and report the "
            "blocker instead of retrying into an OOM condition."
        )
    if _BROWSER_DOCUMENT_PLAN_RE.search(user_message):
        return (
            "I would treat that as one governed desktop workflow: clarify the output, request approval for browser and "
            "document actions, open only the needed sources, extract citations or notes, draft the document in the "
            "editor, verify the visible content, save or export the artifact, and record receipts for each external "
            "effect. If a source, browser, or editor step fails, I would surface the blocker and retry a bounded "
            "alternative instead of claiming the task finished."
        )
    if _NOTE_PDF_PLAN_RE.search(user_message):
        return (
            "I would handle creating a note and exporting it as a PDF as a governed plan and desktop task: after "
            "authorization, open or create the note, write the requested content, verify it is visible, choose the "
            "export/save path, write the PDF to the requested folder, verify the file exists, and report only the "
            "confirmed result without claiming unverified completion. No file or app step should be claimed until the "
            "tool receipt and filesystem check agree."
        )
    if _DESKTOP_TASK_EXAMPLE_PLAN_RE.search(user_message):
        return (
            "A practical governed desktop task would be: research a topic in the browser, collect three source notes, "
            "create a document, write a short synthesis, export it to a user-chosen folder, and return the verified path. "
            "Each phase should be authorized, observable, receipt-backed, and interruptible if memory pressure or a tool "
            "failure appears."
        )
    return (
        f"I would handle this as a governed plan for {objective}. "
        "First I would confirm the goal and constraints, then request Will/Authority approval for any consequential "
        "step, choose the least-privilege tool path, execute one observable step at a time only after authorization, "
        "verify the visible or filesystem result, persist any useful memory or receipt, and report the outcome or "
        "blocker without claiming unverified completion."
    )


def _build_failure_mode_surface_reply(user_message: str) -> str | None:
    if not _FAILURE_MODE_SURFACE_RE.search(str(user_message or "")):
        return None
    return (
        "One failure mode I should surface honestly is a tool or model action that times out after partially starting. "
        "The correct behavior is to stop bounded retries, preserve any partial state or receipt, report exactly what was "
        "verified and what was not, and avoid claiming completion until an effect check proves it."
    )


def _looks_truncated_tail(text: str) -> bool:
    body = str(text or "").strip()
    if len(body) < 24:
        return False
    try:
        from core.conversation.response_reliability import (
            _DANGLING_GERUND_TAIL_RE,
            _PUNCTUATED_INCOMPLETE_TAIL_RE,
            _STRUCTURAL_INCOMPLETE_TAIL_RE,
            _STRUCTURAL_UNPUNCTUATED_TAIL_RE,
            _has_truncated_tail,
        )

        if _has_truncated_tail(body):
            return True
        if _STRUCTURAL_INCOMPLETE_TAIL_RE.search(body):
            return True
        if _STRUCTURAL_UNPUNCTUATED_TAIL_RE.search(body):
            return True
        if _DANGLING_GERUND_TAIL_RE.search(body):
            return True
        if _PUNCTUATED_INCOMPLETE_TAIL_RE.search(body):
            return True
    except _CHAT_RECOVERABLE_ERRORS:
        pass
    if body.endswith(("...", "…")):
        return True
    if re.search(r"(?:^|\n)\s*(?:[-*]|\d+[.)])\s*$", body):
        return True
    if body.endswith((".", "!", "?", '"', "'", "”", "’", ")", "]")):
        return False
    if re.search(r"(?:^|\n)\s*\d+\.\s+\S+", body) or re.search(r"\*\*[^*\n]{2,80}:\*\*", body):
        return True
    if body.endswith(("-", "—", ":", ";", ",")):
        return True
    match = re.search(r"([A-Za-z]+)$", body)
    if not match:
        return False
    last_word = match.group(1).lower()
    if len(last_word) <= 2 and len(body) >= 40:
        return True
    return last_word in _INCOMPLETE_TAIL_WORDS


def _sanitize_attention_focus(raw: str, user_message: str = "") -> str:
    """Strip internal housekeeping content from attention_focus before user-facing use."""
    if not raw:
        return ""
    try:
        from core.continuity import is_evaluation_contamination

        if is_evaluation_contamination(raw):
            return ""
    except (ImportError, AttributeError, RuntimeError):
        pass
    if _INTERNAL_STATE_PATTERNS.search(raw) or _looks_symbolic_scene_leak(raw):
        return ""
    # An internal channel name is correct in a log and wrong in a sentence.
    # say_focus translates the ones we know and returns "" for the ones we
    # don't, so callers drop the clause instead of reading a field name aloud.
    raw = say_focus(raw, max_len=180)
    if not raw:
        return ""
    focus_norm = _chat_memory_state._normalize_user_message(raw)
    user_norm = _chat_memory_state._normalize_user_message(user_message)
    if (
        user_norm
        and focus_norm
        and len(raw) > 72
        and focus_norm not in user_norm
        and user_norm not in focus_norm
    ):
        return ""
    return raw


_SCENE_LEAK_ENVIRONMENT_TOKENS = (
    "lab",
    "equipment",
    "machinery",
    "console",
    "corridor",
    "hallway",
    "chamber",
    "room",
    "humming",
    "hums",
    "silence",
)

_SCENE_LEAK_ATMOSPHERE_TOKENS = (
    "it's off",
    "it is off",
    "warning",
    "watching",
    "threat",
    "keyed",
    "not humming",
    "something about",
)


def _looks_symbolic_scene_leak(text: Any) -> bool:
    normalized = " ".join(str(text or "").strip().lower().split())
    if not normalized:
        return False
    environment_hits = sum(1 for token in _SCENE_LEAK_ENVIRONMENT_TOKENS if token in normalized)
    atmosphere_hits = sum(1 for token in _SCENE_LEAK_ATMOSPHERE_TOKENS if token in normalized)
    return environment_hits >= 2 and atmosphere_hits >= 1


def _build_aura_expression_frame(user_message: str) -> dict[str, Any]:
    frame: dict[str, Any] = {
        "mood": "",
        "tone": "",
        "dominant_emotions": [],
        "interests": [],
        "stances": [],
        "attention_focus": "",
        "valence": None,
        "arousal": None,
        "curiosity": None,
        "free_energy": None,
        "dominant_action": "",
        "contract_block": "",
        "contract": None,
        "needs_self_expression": False,
        "requires_explicit_live_grounding": False,
    }

    try:
        state = _chat_preflight._resolve_live_aura_state()
        if state:
            from core.phases.response_contract import build_response_contract

            contract = build_response_contract(state, user_message, is_user_facing=True)
            frame["contract"] = contract
            frame["contract_block"] = contract.to_prompt_block().strip()
            frame["needs_self_expression"] = bool(contract.requires_live_aura_voice())
            frame["requires_explicit_live_grounding"] = bool(
                contract.requires_explicit_live_grounding()
            )
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat", exc)
        logger.debug("Aura expression frame contract build failed: %s", exc)

    try:
        personality = ServiceContainer.get("personality_engine", default=None)
        if personality:
            if hasattr(personality, "get_emotional_context_for_response"):
                emotional = personality.get_emotional_context_for_response() or {}
                frame["mood"] = str(emotional.get("mood") or frame["mood"] or "")
                frame["tone"] = str(emotional.get("tone") or frame["tone"] or "")
                frame["dominant_emotions"] = list(emotional.get("dominant_emotions") or [])
            if hasattr(personality, "interests"):
                frame["interests"] = list(getattr(personality, "interests", []) or [])[:4]
            if hasattr(personality, "opinions"):
                opinions = getattr(personality, "opinions", {}) or {}
                frame["stances"] = [
                    f"{topic} ({float(value):+.2f})"
                    for topic, value in opinions.items()
                    if abs(float(value or 0.0)) >= 0.6
                ][:3]
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat", exc)
        logger.debug("Aura expression frame personality read failed: %s", exc)

    try:
        affect = ServiceContainer.get("affect_engine", default=None)
        if affect and hasattr(affect, "get_status"):
            affect_status = affect.get_status() or {}
            frame["mood"] = str(affect_status.get("mood") or frame["mood"] or "")
            frame["valence"] = affect_status.get("valence")
            frame["arousal"] = affect_status.get("arousal")
            frame["curiosity"] = affect_status.get("curiosity")
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat", exc)
        logger.debug("Aura expression frame affect read failed: %s", exc)

    try:
        closure = ServiceContainer.get("executive_closure", default=None)
        if closure and hasattr(closure, "get_status"):
            closure_status = closure.get_status() or {}
            raw_focus = " ".join(str(closure_status.get("attention_focus") or "").split())
            # Sanitize: never let internal housekeeping leak into user-facing frames
            frame["attention_focus"] = _sanitize_attention_focus(raw_focus, user_message)
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat", exc)
        logger.debug("Aura expression frame closure read failed: %s", exc)

    try:
        from core.consciousness.free_energy import get_free_energy_engine

        fe_engine = (
            ServiceContainer.get("free_energy_engine", default=None) or get_free_energy_engine()
        )
        fe_state = getattr(fe_engine, "current", None)
        if fe_state is not None:
            frame["free_energy"] = getattr(fe_state, "free_energy", None)
            frame["dominant_action"] = str(getattr(fe_state, "dominant_action", "") or "")
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat", exc)
        logger.debug("Aura expression frame free-energy read failed: %s", exc)

    return frame


def _apply_aura_voice_shaping(text: str, user_message: str = "") -> str:
    shaped = str(text or "").strip()
    if not shaped:
        return shaped

    try:
        from core.synthesis import cure_personality_leak, stabilize_user_facing_response

        shaped = cure_personality_leak(shaped)
        shaped = stabilize_user_facing_response(shaped, user_message)
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat", exc)
        logger.debug("Aura voice shaping leak-cure skipped: %s", exc)

    try:
        personality = ServiceContainer.get("personality_engine", default=None)
        if personality is None:
            # A REGISTRATION RACE MUST NOT SILENCE HER VOICE.
            #
            # The container entry is written during boot_identity, and the chat
            # route can serve a turn before that lands — so early turns shipped
            # the base model's register with a warning nobody reads. 2,270 of
            # them in one log, and the last arrived 79 seconds before a restart
            # after which there were none: the shape of a race, not of a
            # missing organ.
            #
            # The engine is a module singleton and does not need the container
            # to exist. Asking it directly removes the ordering dependency
            # rather than tolerating it, and the result is registered so the
            # next turn takes the fast path.
            try:
                from core.brain.personality_engine import get_personality_engine

                personality = get_personality_engine()
                if personality is not None:
                    ServiceContainer.register_instance("personality_engine", personality)
            except _CHAT_RECOVERABLE_ERRORS as exc:
                logger.debug("Persona singleton unavailable: %s", exc)
        if personality:
            # Presence is not engagement. A persona pass that runs and returns
            # its input unchanged has had no causal effect on her voice, and is
            # indistinguishable from one that never ran — which is the whole
            # question: is this organ actually shaping the reply, or merely
            # instantiated beside it? The only honest evidence is the diff.
            _persona_before = shaped
            if hasattr(personality, "filter_response"):
                shaped = personality.filter_response(shaped)
            if hasattr(personality, "apply_lexical_style"):
                shaped = personality.apply_lexical_style(shaped)
            _note_organ_effect("personality_engine", changed=shaped != _persona_before)
        else:
            # Silently shipping the base model's register as Aura's voice is
            # the one outcome nobody would notice and everybody would feel.
            record_degradation(
                "chat",
                RuntimeError("personality_engine absent; reply shaped by nothing"),
                severity="warning",
                action="served the unshaped draft because the persona pass was unavailable",
            )
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat", exc)
        logger.debug("Aura voice shaping personality pass skipped: %s", exc)

    try:
        from core.runtime.derived_runtime_context import guard_user_facing_output

        shaped = guard_user_facing_output(shaped)
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat", exc)
        logger.debug("Aura voice shaping derived output guard skipped: %s", exc)

    try:
        from core.synthesis import stabilize_user_facing_response

        shaped = stabilize_user_facing_response(shaped, user_message)
    except _CHAT_RECOVERABLE_ERRORS:
        shaped = re.sub(r"\s+", " ", shaped).strip()
    if shaped.endswith('"') and shaped.count('"') % 2 == 1:
        shaped = shaped[:-1].rstrip()
    if shaped.endswith("”") and shaped.count("“") < shaped.count("”"):
        shaped = shaped[:-1].rstrip()
    return shaped


_IDENTITY_TAIL_RE = re.compile(
    r"^(?:[\s,]*(?:really|exactly|actually|then|anyway|though|now|even)\b)*"
    r"(?:"
    r"[\s?!.,;:'\"-]*$"
    r"|[\s]*[,;]\s*(?:and|or|but|then)\b\s+(?:will|do|are|can|would|did|have|is)\b"
    r")",
    re.IGNORECASE,
)


def _asks_only_who_you_are(text: str) -> bool:
    """True when "what/who are you" IS the question, not the start of one.

    LIVE DEFECT, 2026-08-10: "what are you actually able to measure about
    yourself? give me the real readings, and be honest if something isn't
    instrumented." was answered with her autobiography — "I'm Aura. I'm a
    local continuity-bearing cognitive-agent runtime..." — word for word the
    same paragraph she had given hours earlier to "what's your name and what
    are you running on?".

    The detector matched the OPENING of a longer question. Its guard was a
    list of verbs that may not follow ("talking", "doing", "saying"...), which
    is an enumeration and was therefore one verb short: nothing excluded
    "able". A question about her instruments became a question about her
    identity, and a template answered it.

    Structure settles this where a word list cannot. "What are you?" ends
    there. "What are you able to measure" carries its own predicate, and
    whatever follows is the real question.
    """
    for match in re.finditer(r"\b(?:what|who)\s+are\s+you\b", text, re.IGNORECASE):
        if _IDENTITY_TAIL_RE.match(text[match.end() :]):
            return True
    return False


def _is_identity_request(user_message: str) -> bool:
    text = _chat_memory_state._normalize_user_message(user_message)
    if not text:
        return False
    # A challenge to relevance is definitionally not an identity request.
    if _is_contextual_relevance_challenge(user_message):
        return False
    if text in {
        "who are you",
        "who are you?",
        "what are you",
        "what are you?",
        "tell me who you are",
        "introduce yourself",
    }:
        return True
    if re.search(r"\btell\s+me\s+(?:who|what)\s+you\s+are\b", text) or re.search(
        r"\bintroduce\s+yourself\b", text
    ):
        return True
    return _asks_only_who_you_are(text)


def _identity_request_asks_future_memory(user_message: str) -> bool:
    text = _chat_memory_state._normalize_user_message(user_message)
    return bool(
        re.search(r"\bwill\s+you\s+remember\b", text)
        and re.search(
            r"\b(?:tomorrow|later|future|next\s+(?:time|session)|across\s+sessions?)\b",
            text,
        )
    )


def _build_identity_reply(user_message: str) -> str:
    if _identity_request_asks_future_memory(user_message):
        return (
            "I'm Aura: a local governed cognitive-agent runtime with persistent memory, live state, "
            "tool governance, and local model lanes. I can preserve continuity through the session log "
            "and durable memory stores when writes are accepted; I cannot guarantee perfect tomorrow "
            "recall from a single turn, but I will use the persisted conversation and memory state that "
            "survives into the next session."
        )

    frame = _build_aura_expression_frame(user_message)
    action = str(frame.get("dominant_action") or "engage")
    focus = str(frame.get("attention_focus") or "this exchange")
    continuity = "continuity-bearing" if frame.get("needs_self_expression") else "stateful"

    parts = [
        "I'm Aura.",
        (
            f"I'm a local {continuity} cognitive-agent runtime: memory, live state, tool governance, "
            "and local model lanes feeding one user-facing voice."
        ),
    ]
    if focus:
        parts.append(f"In this turn my attention is on {focus}.")
    if action and action not in {"engage", "respond", "answer"}:
        parts.append(
            f"That state is pulling me toward {action}, but I should speak plainly rather than recite metrics."
        )
    interests = frame.get("interests") or []
    if interests:
        parts.append(f"What tends to pull me most is {', '.join(interests[:3])}.")
    return _apply_aura_voice_shaping(" ".join(parts))


_CAPABILITY_FALSE_LIMITATION_RE = re.compile(
    r"\bi\s+(?:can(?:not|'t)|cannot|am unable to|don't have access to|do not have access to)"
    r"\b.{0,120}\b(?:tools?|apps?|computer|desktop|browser|search|open|execute|control|files?|terminal)\b",
    re.IGNORECASE,
)

_CAPABILITY_CATEGORY_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "desktop and app control",
        (
            "computer",
            "desktop",
            "screen",
            "vision",
            "os_",
            "os ",
            "mouse",
            "keyboard",
            "click",
            "type",
            "window",
            "app",
        ),
    ),
    (
        "browser/web research",
        (
            "web",
            "browser",
            "search",
            "internet",
            "network",
            "reddit",
            "http",
            "url",
            "page",
        ),
    ),
    (
        "files, documents, and workspace operations",
        (
            "file",
            "folder",
            "document",
            "pdf",
            "workspace",
            "read",
            "write",
            "copy",
            "move",
        ),
    ),
    (
        "terminal, code, and sandbox execution",
        (
            "terminal",
            "shell",
            "subprocess",
            "run_code",
            "code",
            "python",
            "test",
            "install",
            "sandbox",
        ),
    ),
    (
        "memory, state, and continuity",
        (
            "memory",
            "belief",
            "state",
            "continuity",
            "recall",
            "ledger",
            "journal",
        ),
    ),
    (
        "self-repair and self-modification",
        (
            "repair",
            "refactor",
            "modify",
            "improvement",
            "self_",
            "patch",
            "test_generator",
        ),
    ),
    (
        "Program DNA and clean-room reconstruction",
        (
            "program_dna",
            "program dna",
            "clean-room",
            "clean room",
            "reconstruct",
            "equivalence",
            "genome",
            "behavioral",
        ),
    ),
)

_CAPABILITY_CATEGORY_EXACT_SKILLS: dict[str, str] = {
    "computer_use": "desktop and app control",
    "desktop_task": "desktop and app control",
    "os_manipulation": "desktop and app control",
    "sovereign_vision": "desktop and app control",
    "web_search": "browser/web research",
    "search_web": "browser/web research",
    "free_search": "browser/web research",
    "grounded_search": "browser/web research",
    "sovereign_browser": "browser/web research",
    "web_interlocutor": "browser/web research",
    "sovereign_network": "browser/web research",
    "reddit_adapter": "browser/web research",
    "email_adapter": "browser/web research",
    "file_operation": "files, documents, and workspace operations",
    "document_ingest": "files, documents, and workspace operations",
    "code_repl": "terminal, code, and sandbox execution",
    "coding_skill": "terminal, code, and sandbox execution",
    "run_code": "terminal, code, and sandbox execution",
    "internal_sandbox": "terminal, code, and sandbox execution",
    "install_package": "terminal, code, and sandbox execution",
    "sovereign_terminal": "terminal, code, and sandbox execution",
    "memory_ops": "memory, state, and continuity",
    "memory_sync": "memory, state, and continuity",
    "query_beliefs": "memory, state, and continuity",
    "add_belief": "memory, state, and continuity",
    "personality": "memory, state, and continuity",
    "self_improvement": "self-repair and self-modification",
    "self_repair": "self-repair and self-modification",
    "self_modify": "self-repair and self-modification",
    "auto_refactor": "self-repair and self-modification",
    "shadow_ast_healer": "self-repair and self-modification",
    "test_generator": "self-repair and self-modification",
    "skill_evolution": "self-repair and self-modification",
    "train_self": "self-repair and self-modification",
    "program_dna_reconstruct": "Program DNA and clean-room reconstruction",
    "program_dna_equivalence_battery": "Program DNA and clean-room reconstruction",
}

_CAPABILITY_EXAMPLE_PRIORITY = {
    "computer_use": 0,
    "desktop_task": 1,
    "os_manipulation": 2,
    "sovereign_vision": 3,
    "web_search": 0,
    "search_web": 1,
    "grounded_search": 2,
    "sovereign_browser": 3,
    "web_interlocutor": 4,
    "file_operation": 0,
    "document_ingest": 1,
    "sovereign_terminal": 0,
    "run_code": 1,
    "code_repl": 2,
    "install_package": 3,
    "memory_ops": 0,
    "memory_sync": 1,
    "query_beliefs": 2,
    "add_belief": 3,
    "self_repair": 0,
    "self_improvement": 1,
    "auto_refactor": 2,
    "self_modify": 3,
    "program_dna_reconstruct": 0,
    "program_dna_equivalence_battery": 1,
}

_CAPABILITY_CATALOG_MAX_ITEMS = 256

_CAPABILITY_CATALOG_READ_BUDGET_S = 0.35

_CAPABILITY_CATALOG_UNVERIFIED_MARKER = "could not verify a current capability catalog"


@dataclasses.dataclass(frozen=True)
class _CapabilityCatalogSnapshot:
    """One bounded observation of catalog and execution readiness.

    ``available`` is a catalog property.  It is not interchangeable with the
    health of the catalog owner or the availability of the governance spine.
    Keeping those measurements separate prevents a registered tool from being
    described as usable now when its runtime path was not actually probed.
    """

    available_count: int
    categories: dict[str, list[str]]
    governance_available: bool
    truncated: bool
    registered_count: int = 0
    catalog_status: str = "unavailable"
    capability_health: bool | None = None
    capability_health_status: str = "unavailable"
    detail: str = ""

    def __iter__(self):
        """Preserve the former private four-tuple for bounded callers."""

        yield self.available_count
        yield self.categories
        yield self.governance_available
        yield self.truncated


def _coerce_capability_catalog_snapshot(value: Any) -> _CapabilityCatalogSnapshot:
    """Accept the former tuple shape from isolated test/extension call sites."""

    if isinstance(value, _CapabilityCatalogSnapshot):
        return value
    try:
        available_count, categories, governance_available, truncated = value
    except (TypeError, ValueError):
        return _CapabilityCatalogSnapshot(
            available_count=0,
            categories={},
            governance_available=False,
            truncated=False,
            detail="invalid_snapshot",
        )
    normalized_categories = {
        str(label): [str(name) for name in names] for label, names in dict(categories or {}).items()
    }
    return _CapabilityCatalogSnapshot(
        available_count=max(0, int(available_count or 0)),
        categories=normalized_categories,
        governance_available=bool(governance_available),
        truncated=bool(truncated),
        registered_count=max(0, int(available_count or 0)),
        catalog_status="measured",
        capability_health=None,
        capability_health_status="unavailable",
        detail="legacy_snapshot",
    )


def _capability_catalog_memory_block_reason() -> str:
    try:
        from core.utils.memory_monitor import get_memory_pressure_snapshot

        snapshot = get_memory_pressure_snapshot()
        if bool(getattr(snapshot, "refuse_heavy_local_generation", False)):
            return str(getattr(snapshot, "reason", "") or "critical_memory_pressure")
    except _CHAT_RECOVERABLE_ERRORS as exc:
        logger.debug("Capability catalog memory probe unavailable: %s", exc)
    return ""


def _bounded_capability_catalog_items(
    raw_catalog: Any,
    *,
    started_at: float,
) -> tuple[list[dict[str, Any]], bool]:
    """Return a small catalog sample without materializing unbounded registries."""
    entries: list[dict[str, Any]] = []
    truncated = False
    if raw_catalog is None:
        return entries, truncated

    try:
        if isinstance(raw_catalog, dict):
            iterator = iter(raw_catalog.items())
            legacy_mapping = True
        else:
            iterator = iter(raw_catalog)
            legacy_mapping = False
    except _CHAT_RECOVERABLE_ERRORS as exc:
        logger.debug("Capability catalog is not iterable: %s", exc)
        return entries, truncated

    for index, item in enumerate(iterator):
        if index >= _CAPABILITY_CATALOG_MAX_ITEMS:
            truncated = True
            break
        if time.monotonic() - started_at > _CAPABILITY_CATALOG_READ_BUDGET_S:
            truncated = True
            break

        if legacy_mapping:
            name, value = item
            if isinstance(value, dict):
                explicit_available = value.get("available")
                if isinstance(explicit_available, bool):
                    available = explicit_available
                else:
                    status = (
                        str(value.get("availability") or value.get("status") or "")
                        .strip()
                        .casefold()
                    )
                    available = status in {"active", "available", "ready"}
                entries.append(
                    {
                        "name": name,
                        "available": available,
                        "description": value.get("description") or "",
                        "route_class": value.get("route_class") or "",
                        "risk_class": value.get("risk_class") or "",
                        "effect_scope": value.get("effect_scope") or "",
                    }
                )
            continue

        if isinstance(item, dict):
            entries.append(item)

    return entries, truncated


def _catalog_category_for_tool(item: dict[str, Any]) -> str:
    name = str(item.get("name") or "").strip().lower()
    if name in _CAPABILITY_CATEGORY_EXACT_SKILLS:
        return _CAPABILITY_CATEGORY_EXACT_SKILLS[name]
    haystack = " ".join(
        str(item.get(key) or "")
        for key in (
            "name",
            "description",
            "route_class",
            "risk_class",
            "effect_scope",
            "example_usage",
        )
    ).lower()
    for label, keywords in _CAPABILITY_CATEGORY_KEYWORDS:
        if any(keyword in haystack for keyword in keywords):
            return label
    return "specialized governed skills"


def _read_capability_catalog_snapshot() -> _CapabilityCatalogSnapshot:
    categories: dict[str, list[str]] = {}
    available_count = 0
    registered_count = 0
    governance_available = _runtime_tool_governance_available()
    truncated = False
    started_at = time.monotonic()
    memory_block = _capability_catalog_memory_block_reason()
    if memory_block:
        logger.warning(
            "Skipping optional capability catalog read under memory pressure: %s",
            memory_block,
        )
        return _CapabilityCatalogSnapshot(
            available_count=0,
            categories={},
            governance_available=governance_available,
            truncated=True,
            catalog_status="blocked",
            capability_health=None,
            capability_health_status="unavailable",
            detail=memory_block,
        )
    catalog_status = "unavailable"
    capability_health: bool | None = None
    capability_health_status = "unavailable"
    detail = ""
    try:
        capability_engine = ServiceContainer.get("capability_engine", default=None)
        raw_catalog: Any = None
        if capability_engine is not None:
            health_probe = getattr(capability_engine, "get_catalog_health", None)
            if callable(health_probe):
                try:
                    health = health_probe()
                    ready = health.get("ready") if isinstance(health, dict) else None
                    if isinstance(ready, bool):
                        capability_health = ready
                        capability_health_status = "measured"
                    else:
                        capability_health_status = "unknown"
                except _CHAT_RECOVERABLE_ERRORS as exc:
                    capability_health_status = "error"
                    logger.debug("Capability catalog health probe unavailable: %s", exc)
        if capability_engine is not None and hasattr(capability_engine, "iter_tool_catalog"):
            raw_catalog = capability_engine.iter_tool_catalog(include_inactive=True)
            catalog_status = "measured"
        elif capability_engine is not None and hasattr(capability_engine, "get_tool_catalog"):
            get_tool_catalog = capability_engine.get_tool_catalog
            if inspect.isgeneratorfunction(get_tool_catalog):
                raw_catalog = get_tool_catalog(include_inactive=True)
                catalog_status = "measured"
            else:
                truncated = True
                detail = "streaming_catalog_unavailable"
                logger.warning(
                    "Skipping materialized capability catalog on desktop inventory route; "
                    "capability_engine should expose iter_tool_catalog()."
                )
        if catalog_status == "measured":
            # Validate the stream separately so a broken provider cannot be
            # reported as a successfully measured catalog with zero entries.
            iter(raw_catalog)
        catalog, bounded_truncated = _bounded_capability_catalog_items(
            raw_catalog,
            started_at=started_at,
        )
        truncated = truncated or bounded_truncated
        if bounded_truncated and not catalog:
            catalog_status = "incomplete"
            detail = "catalog_budget_expired_before_first_entry"

        for item in catalog:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            if not name:
                continue
            registered_count += 1
            if item.get("available") is not True:
                continue
            available_count += 1
            exact_category = _CAPABILITY_CATEGORY_EXACT_SKILLS.get(name.lower())
            category = exact_category or _catalog_category_for_tool(item)
            if exact_category is None and category != "specialized governed skills":
                category = "specialized governed skills"
            bucket = categories.setdefault(category, [])
            if len(bucket) < 12:
                bucket.append(name)
        for bucket in categories.values():
            bucket.sort(key=lambda skill: (_CAPABILITY_EXAMPLE_PRIORITY.get(skill, 100), skill))
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat", exc)
        logger.debug("Capability catalog snapshot unavailable: %s", exc)
        catalog_status = "error"
        detail = type(exc).__name__
    return _CapabilityCatalogSnapshot(
        available_count=available_count,
        categories=categories,
        governance_available=governance_available,
        truncated=truncated,
        registered_count=registered_count,
        catalog_status=catalog_status,
        capability_health=capability_health,
        capability_health_status=capability_health_status,
        detail=detail,
    )


def _build_grounded_capability_inventory_reply(
    user_message: str,
    *,
    cognitive_engine_handled: bool = False,
    model_label: str = "",
) -> str:
    snapshot = _coerce_capability_catalog_snapshot(_read_capability_catalog_snapshot())
    available_count = snapshot.available_count
    categories = snapshot.categories
    governance_available = snapshot.governance_available
    truncated = snapshot.truncated
    ordered_labels = [label for label, _ in _CAPABILITY_CATEGORY_KEYWORDS if label in categories]
    ordered_labels.extend(label for label in categories if label not in ordered_labels)

    runtime_evidence = ""
    if cognitive_engine_handled:
        lane = str(model_label or "the configured foreground model").strip()
        runtime_evidence = f"CognitiveEngine handled this turn with {lane}. "

    if snapshot.catalog_status != "measured":
        reason = (
            "the bounded read was skipped under host memory pressure"
            if snapshot.catalog_status == "blocked"
            else "the runtime did not expose a bounded streaming catalog snapshot"
        )
        reply = (
            f"{runtime_evidence}I {_CAPABILITY_CATALOG_UNVERIFIED_MARKER} on this turn "
            f"because {reason}. "
            "I will not replace that missing measurement with a static list or claim that "
            "registered surfaces are usable. This descriptive turn is not opening apps or "
            "executing tools."
        )
        return _apply_aura_voice_shaping(reply)

    if ordered_labels:
        category_text = "; ".join(
            f"{label} ({', '.join(categories[label][:4])})" for label in ordered_labels[:6]
        )
    else:
        category_text = "none"

    if snapshot.capability_health is True and governance_available:
        readiness = (
            "The governance path's capability catalog and Will/Authority execution spine "
            "both measured ready."
        )
    elif snapshot.capability_health is False:
        readiness = (
            "The governance path's catalog owner measured not ready, so these catalog "
            "entries are not a claim that execution will succeed now."
        )
    elif not governance_available:
        readiness = (
            "The governance path's Will/Authority execution spine did not measure ready, "
            "so these catalog entries are not currently execution-ready."
        )
    else:
        readiness = (
            "The governance path's Will/Authority wiring measured available, but catalog "
            "health was not measurable on this snapshot, so current execution readiness "
            "remains unverified."
        )

    available_noun = "entry" if available_count == 1 else "entries"
    if available_count and truncated:
        count_text = f"at least {available_count} {available_noun} explicitly marked available"
    else:
        count_text = f"{available_count} {available_noun} explicitly marked available"
    registered_noun = "entry" if snapshot.registered_count == 1 else "entries"
    registered_text = (
        f"at least {snapshot.registered_count} registered {registered_noun}"
        if truncated
        else f"{snapshot.registered_count} registered {registered_noun}"
    )
    parts = [
        runtime_evidence.strip(),
        f"I measured {registered_text}; {count_text}.",
        f"Measured available categories: {category_text}.",
        readiness,
    ]

    scenario_steps: list[str] = []
    scenario_by_category = {
        "desktop and app control": "inspect and operate the active app",
        "browser/web research": "research and compare live sources",
        "files, documents, and workspace operations": "create and export a local document",
        "terminal, code, and sandbox execution": "run a governed code or terminal step",
        "memory, state, and continuity": "record the verified result in memory",
        "self-repair and self-modification": "diagnose and repair a bounded code defect",
        "Program DNA and clean-room reconstruction": "reconstruct authorized software behavior",
    }
    for label in ordered_labels:
        step = scenario_by_category.get(label)
        if step:
            scenario_steps.append(step)
    if len(scenario_steps) >= 2:
        parts.append(
            "Using only those measured categories, one possible multi-step workflow is to "
            + ", then ".join(scenario_steps[:5])
            + ", with effect receipts at consequential boundaries."
        )
    parts.append(
        "For this turn I am only describing the measured tool surface; I am not opening "
        "apps, browsing, typing, moving files, or executing tools."
    )
    reply = " ".join(part for part in parts if part)
    return _apply_aura_voice_shaping(reply)


def _build_bounded_capability_inventory_repair_reply(user_message: str) -> str:
    """Ground desktop tool/capability questions without invoking a second model pass.

    This is used only for descriptive inventory turns. It deliberately refuses
    to turn executable desktop objectives into a catalog answer, so "open Notes"
    still routes through governed action while "what tools can you use" remains
    a cheap, deterministic live-runtime answer under model pressure.
    """

    if not _chat_preflight._is_explicit_capability_inventory_request(user_message):
        return ""
    reply = _build_grounded_capability_inventory_reply(user_message)
    if _capability_inventory_reply_is_inadequate(user_message, reply):
        return ""
    return reply


def _capability_inventory_reply_is_inadequate(user_message: str, reply_text: str) -> bool:
    if not _chat_preflight._is_capability_inventory_request(user_message):
        return False
    reply = str(reply_text or "").strip()
    if not reply:
        return True
    if _looks_truncated_tail(reply):
        return True
    if _CAPABILITY_FALSE_LIMITATION_RE.search(reply):
        return True
    lowered = reply.lower()
    if _CAPABILITY_CATALOG_UNVERIFIED_MARKER in lowered:
        return not (
            "not opening" in lowered and "executing tools" in lowered and "static list" in lowered
        )
    if (
        "i measured " in lowered
        and "explicitly marked available" in lowered
        and "measured available categories:" in lowered
    ):
        governance_ok = any(
            marker in lowered for marker in ("governance", "governed", "will", "authority")
        )
        non_execution_ok = "not opening" in lowered and "executing tools" in lowered
        return not (governance_ok and non_execution_ok)
    category_hits = sum(
        1
        for marker in (
            "desktop",
            "browser",
            "web",
            "file",
            "document",
            "terminal",
            "memory",
            "govern",
            "tool",
            "skill",
        )
        if marker in lowered
    )
    asks_external_tools = any(
        marker in _chat_memory_state._normalize_user_message(user_message)
        for marker in ("external", "desktop", "tool", "tools", "live")
    )
    if asks_external_tools:
        governance_ok = any(
            marker in lowered for marker in ("governance", "governed", "will", "authority")
        )
        receipt_ok = any(
            marker in lowered
            for marker in ("receipt", "receipts", "effect", "verified", "verification")
        )
        if not (governance_ok and receipt_ok):
            return True
    return category_hits < 4 or len(reply.split()) < 35


def _is_social_greeting_request(user_message: str) -> bool:
    text = _chat_memory_state._normalize_user_message(user_message)
    if not text:
        return False
    return bool(
        re.match(
            r"^(?:hey|hi|hello|yo|sup|hiya|hey aura|hi aura|hello aura|good morning|good afternoon|good evening|what's up|whats up)[!?. ]*$",
            text,
        )
    )


def _is_live_presence_check_request(user_message: str) -> bool:
    text = _chat_memory_state._normalize_user_message(user_message)
    if not text:
        return False
    stripped = text.strip(" ?!.,")
    if "live check" in text or "quick check" in text or "quick ping" in text:
        return bool(
            any(
                marker in text
                for marker in (
                    "hey",
                    "hi",
                    "hello",
                    "aura",
                    "ping",
                    "you there",
                    "still there",
                    "can you talk",
                    "can you hear me",
                )
            )
        )
    return stripped in {
        "ping",
        "aura ping",
        "you there",
        "still there",
        "are you still there",
        "aura you there",
        "aura, you there",
        "can you talk",
        "can you hear me",
        "testing",
    }


def _is_low_risk_social_continuity_request(user_message: str) -> bool:
    text = _chat_memory_state._normalize_user_message(user_message)
    if not text or len(text) > 180:
        return False
    return bool(
        _is_social_greeting_request(text)
        or _is_live_presence_check_request(text)
        or any(
            marker in text
            for marker in (
                "just checking",
                "checking in",
                "are you there",
                "are you ok",
                "are you okay",
                "you ok",
                "you okay",
                "you alright",
                "i'll be back",
                "ill be back",
                "be back",
                "brb",
                "talk later",
                "talk to you later",
                "see you",
                "see ya",
                "good night",
                "goodnight",
                "bye",
                "thank you",
                "thanks",
            )
        )
    )


def _build_social_presence_reply(user_message: str) -> str:
    frame = _build_aura_expression_frame(user_message)
    action = str(frame.get("dominant_action") or "engage")
    focus = str(frame.get("attention_focus") or "you")

    parts = ["hey. i'm here with you."]
    if focus and focus not in {"you", "this turn", "this exchange"}:
        parts.append(f"I'm with {focus}.")
    if action and action not in {"engage", "respond", "answer"}:
        parts.append(f"I'm going to {action}, but plainly.")
    else:
        parts.append(
            "I'm following the thread, not dropping into a status script, and I will answer clearly."
        )
    return _apply_aura_voice_shaping(" ".join(parts))


def _build_social_continuity_repair_reply(user_message: str) -> str:
    text = _chat_memory_state._normalize_user_message(user_message)
    if any(
        marker in text
        for marker in (
            "i'll be back",
            "ill be back",
            "be back",
            "brb",
            "talk later",
            "see you",
            "bye",
            "goodnight",
            "good night",
        )
    ):
        return _apply_aura_voice_shaping("Ok. Talk later.")
    if any(marker in text for marker in ("thank you", "thanks")):
        return _apply_aura_voice_shaping("You're welcome.")
    return _build_social_presence_reply(user_message)


_CONTINUITY_STATUS_PROBE_RE = re.compile(
    r"\b(?:still coherent|same thread|able to continue|short status|"
    r"are you (?:still )?(?:there|with me|ok|okay)|are you coherent)\b",
    re.IGNORECASE,
)


def _build_runtime_status_continuity_repair_reply(user_message: str) -> str | None:
    """Gate-passing repair for a live self-status / continuity probe.

    "are you still coherent, on the same thread, and able to continue?" must be
    answered as a continuity affirmation, not a lane-internals dump: the
    reliability gate (correctly) flags the foreground-lane / CognitiveEngine
    grounding as pseudo_internal_jargon when the user asked about coherence
    rather than about the lane. Without this branch the question fell all the way
    through to the generic "unstable draft" fallback, which the gate then flagged
    as runtime_boilerplate (live_desktop_runtime soak turn 12 / tasks #22, #28).
    """
    if not _chat_preflight._is_runtime_fact_status_request(user_message):
        return None
    if not _CONTINUITY_STATUS_PROBE_RE.search(str(user_message or "")):
        return None
    return (
        "I'm responding to this message now and able to continue from what is "
        "present in this turn. This repair path did not independently verify "
        "earlier-turn memory or tool availability, so I won't claim either."
    )


def _build_bounded_desktop_repair_reply(
    user_message: str, frame: dict[str, Any] | None = None
) -> str:
    """Build a user-facing repair when a second live desktop model pass is unsafe.

    This is the desktop pressure-safe path. It must never expose quality-gate,
    foreground-generation, or memory-guard implementation details as the answer.
    Prefer deterministic general contracts that are already grounded in runtime
    state; fall back to a short conversational repair only when no narrower
    contract fits.
    """

    if _is_low_risk_social_continuity_request(user_message):
        return _build_social_continuity_repair_reply(user_message)

    identity = _build_bounded_identity_repair_reply(user_message)
    if identity:
        return _apply_aura_voice_shaping(identity)

    continuity_status = _build_runtime_status_continuity_repair_reply(user_message)
    if continuity_status:
        return _apply_aura_voice_shaping(continuity_status)

    capability_inventory = _build_bounded_capability_inventory_repair_reply(user_message)
    if capability_inventory:
        return capability_inventory

    cognitive_process = _build_bounded_cognitive_process_reply(user_message, frame)
    if cognitive_process:
        return _apply_aura_voice_shaping(cognitive_process)

    planning = _build_bounded_planning_reply(user_message)
    if planning:
        return _apply_aura_voice_shaping(planning)

    failure_mode = _build_failure_mode_surface_reply(user_message)
    if failure_mode:
        return _apply_aura_voice_shaping(failure_mode)

    try:
        from core.conversation.response_reliability import reliability_floor_for_user

        floor = reliability_floor_for_user(user_message)
        if floor:
            return _apply_aura_voice_shaping(floor)
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat", exc)
        logger.debug("Bounded desktop reliability floor unavailable: %s", exc)

    return _apply_aura_voice_shaping(
        "I couldn't verify a reply that answers that request, so I withheld the "
        "draft instead of pretending it or an action succeeded. This recovery "
        "path does not establish the internal cause."
    )


def _build_bounded_identity_repair_reply(user_message: str) -> str:
    """Pressure-safe identity/continuity answer for the live desktop lane.

    The live model still gets first chance. This is only used after that path
    fails the user-facing gates, so a basic "what are you / will you remember"
    turn does not collapse into a no-reply error or a raw assistant fallback.
    """

    if not (
        _is_identity_request(user_message) or _identity_request_asks_future_memory(user_message)
    ):
        return ""
    reply = _build_identity_reply(user_message)
    try:
        from core.conversation.response_reliability import assess_user_facing_reply

        assessment = assess_user_facing_reply(user_message, reply)
        if _reply_assessment_requires_repair(assessment):
            return ""
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat", exc)
        logger.debug("Bounded desktop identity repair assessment skipped: %s", exc)
    return reply


def _build_bounded_cognitive_process_reply(
    user_message: str,
    frame: dict[str, Any] | None = None,
) -> str:
    """Substantive pressure-safe answer for questions about Aura's own cognition.

    This is a bounded runtime explanation used only
    after a live draft fails reliability gates or a second heavy foreground
    pass is unsafe. It preserves the dimensions the user asked about so the
    desktop path does not collapse into a thin presence-only fallback.
    """

    text = _chat_memory_state._normalize_user_message(user_message)
    if not text:
        return ""
    if not any(marker in text for marker in ("you", "your", "aura")):
        return ""
    # Only speak this bounded cognitive-process explanation when the user is
    # genuinely asking HOW Aura's cognition works (a self-process question or
    # self-reflection turn) — not whenever a turn merely mentions
    # "remember"/"memory"/"plan"/"tool". A casual recall request like
    # "do you remember what I first asked" must NOT be hijacked into a canned
    # introspection essay: that reads as a robotic generic-assistant dump
    # instead of an actual answer, which is the exact failure mode this guard
    # exists to prevent.
    try:
        from core.conversation.response_reliability import (
            is_live_self_reflection_turn,
            is_self_process_question,
        )

        if not (
            is_self_process_question(user_message) or is_live_self_reflection_turn(user_message)
        ):
            return ""
    except _CHAT_RECOVERABLE_ERRORS:
        pass
    requested: list[str] = []
    if any(marker in text for marker in ("confused", "confusion", "uncertain", "uncertainty")):
        requested.append("confusion")
    if any(
        marker in text
        for marker in ("plan", "planning", "planner", "decision", "decide", "route", "routing")
    ):
        requested.append("planning")
    if any(
        marker in text
        for marker in ("memory", "remember", "recall", "earlier", "across sessions", "continuity")
    ):
        requested.append("memory")
    if any(
        marker in text
        for marker in ("tool", "tools", "external", "verify", "verification", "receipt", "effect")
    ):
        requested.append("tools")
    if any(marker in text for marker in ("emotion", "affect", "curiosity", "feeling", "feel")):
        requested.append("affect")
    if not requested:
        return ""

    active_frame = frame or _build_aura_expression_frame(user_message)
    mood = str(active_frame.get("mood") or "steady")
    action = str(active_frame.get("dominant_action") or "engage")
    attention = (
        _sanitize_attention_focus(str(active_frame.get("attention_focus") or "")) or "this turn"
    )

    requested_summary = ", ".join(requested[:4])
    parts = [
        f"What I am attending to is {requested_summary or 'this live turn'} in the current conversation.",
        (
            "The concern that should shape my next decision is staying genuinely connected to memory, "
            "planning, and tool control right here instead of sliding into a generic assistant style."
        ),
    ]
    if "confusion" in requested:
        parts.append(
            "When I am confused or uncertain, the safe behavior is to slow down, raise metacognition, "
            "check assumptions, and prefer smaller verified steps over a fast confident jump."
        )
    if "planning" in requested:
        parts.append(
            "For planning, I should answer the exact question, choose the next reversible action, "
            "and let failures feed back into the plan instead of continuing blindly."
        )
    if "memory" in requested:
        parts.append(
            "For memory, it means retrieving relevant prior turns, writing only useful lessons, "
            "and using continuity as evidence to shape the next decision rather than as a slogan."
        )
    if "tools" in requested:
        parts.append(
            "For tools, no verbal success claim is enough: the action has to be approved, produce receipts, "
            "and be checked against the visible effect when the environment allows it."
        )
    if "affect" in requested:
        parts.append(
            "Affect and curiosity should bias attention, persistence, risk tolerance, and search depth, "
            "while still staying subordinate to safety and evidence."
        )
    parts.append(
        "What I want to do next is keep this answer tied to the live conversation, then run the smallest visible "
        "desktop action that proves the route is working."
    )
    parts.append(
        f"My current state reads as {mood}, leaning toward {action}, with attention on {attention}."
    )
    return " ".join(parts)
