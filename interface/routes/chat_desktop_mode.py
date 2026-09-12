"""Which mind answers a desktop turn, and what it is given to answer from.

A desktop turn can be served by the cognitive engine, by a compact contract
that skips most of it, or by the resilience path when the engine fails. This is
where that is chosen, and where the live-mind payload each one gets is built —
so a turn that took the cheap route can be told apart afterwards from one that
had the whole mind behind it.
"""
from __future__ import annotations

import re
import time
from typing import Any

from core.container import ServiceContainer
from core.runtime import response_policy
from core.runtime.errors import record_degradation
from core.runtime.flags import FlagKind, declare
from core.runtime.structured_input import (
    analyze_prompt_shape,
)
from core.utils.injected_blocks import stamp_runtime_payload
from interface.routes import chat_conversation_repair as _chat_conversation_repair  # noqa: E402
from interface.routes import chat_desktop_objective as _chat_desktop_objective  # noqa: E402
from interface.routes import chat_desktop_repair as _chat_desktop_repair  # noqa: E402
from interface.routes import chat_memory_state as _chat_memory_state  # noqa: E402
from interface.routes import chat_preflight as _chat_preflight  # noqa: E402
from interface.routes import chat_protected_prompt as _chat_protected_prompt  # noqa: E402
from interface.routes import chat_turn_contract as _chat_turn_contract  # noqa: E402
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
from interface.routes.chat_turn_evidence import (  # noqa: E402,F401
    _benchmark_prompt_requests_fenced_artifact,
    _build_explicit_local_file_artifact,
    _canonical_memory_state_evidence_from_tuple,
    _canonical_memory_state_evidence_missing_from_reply,
    _collect_named_url_evidence,
    _collect_recent_traceability_event_sync,
    _context_challenge_repair_has_evidence,
    _correct_unevidenced_action_claims,
    _emit_chat_output_receipt,
    _evidence_came_from_the_network,
    _extract_canonical_memory_state_evidence_block,
    _prime_requested_output_contract_trace,
    _recent_action_receipts,
    _record_desktop_evidence_on_the_trace,
    _resolve_prior_answer_provenance,
    _save_requested_artifact,
    _serve_built_artifact,
    _worker_receipt_transaction_id,
)

from .chat_desktop_evidence import (  # noqa: E402
    _desktop_cognitive_failure_repair_target,
    _desktop_objective_self_sufficient_without_cognitive_text,
    _search_result_entries,
)
from .chat_lane_bookkeeping import (  # noqa: E402
    _assess_live_mind_snapshot,
    _env_float,
    _host_condition,
    _reply_needs_continuation,
    _still_contradicts_the_runtime,
)
from .chat_own_source import (
    _ASKS_TO_INSPECT_SHOWN_SOURCE_RE,  # noqa: F401
    _ASKS_WHERE_CODE_LIVES_RE,  # noqa: F401
    _OWN_SOURCE_ROUTE_MARGIN,  # noqa: F401
    _RECEIPTS_WORTH_READING,  # noqa: F401
    _REPO_PROBE_MAX_BYTES,  # noqa: F401
    _SELF_METRIC_CORRECTION_MARK,  # noqa: F401
    _SELF_PROCESS_ABOUT_HER_RE,  # noqa: F401
    _SELF_PROCESS_HYPOTHETICAL_RE,  # noqa: F401
    _attempt_generated_social_grounding_repair,  # noqa: F401
    _explains_the_finding,  # noqa: F401
    _self_process_requested_dimensions,
    _turn_asks_where_that_came_from,  # noqa: F401
    )
from .chat_reply_shaping import (  # noqa: E402
    _append_turn_text_mutation,
)
from .chat_served_answers import (  # noqa: E402
    _the_answer_has_to_be_worked_out,
)

#: What the last foreground turn spent before the model was asked anything.
#: One turn, overwritten each time — a history belongs in the tracker, and this
#: is here so she can answer "where did that turn's time go" about the turn the
#: person just had.
_LAST_TURN_PREPARATION: dict[str, Any] = {}
#: Claiming a live check. Only true when the evidence came from the network.
_CLAIMS_A_LIVE_CHECK_RE = re.compile(
    r"\b(?:i\s+(?:checked|searched|looked\s+up|found)\s+"
    r"(?:the\s+)?(?:live\s+)?(?:web|internet|online)"
    r"|live\s+web\s+(?:evidence|search|results?)"
    r"|according\s+to\s+(?:my\s+)?(?:live\s+)?(?:web\s+)?search)\b",
    re.IGNORECASE,
)
#: A citation shape with nothing in it.
#:
#: LIVE, 2026-08-22: asked about a company with sources, the reply ended
#: "Source: [Live web search]". No URL, no title, nothing anyone could open —
#: the shape of a citation standing in for one. A reader skims that as
#: evidence, which is worse than no citation at all.
_EMPTY_CITATION_RE = re.compile(
    r"\bsources?\s*:\s*(?!https?://)"
    r"(?:\[[^\]]{0,60}\]|\(?(?:live |the )?(?:web )?search(?:es)?\)?|"
    r"my (?:own )?(?:memory|knowledge)|internal|n/?a|none|unknown|tbd)"
    r"\s*\.?",
    re.IGNORECASE,
)
_SEARCH_SNIPPET_BOILERPLATE_RE = re.compile(
    r"\b(skip to content|please fill out this field|search|newsletter|advertisement|subscribe|sign in|login)\b|-->\s*",
    re.IGNORECASE,
)

_FORCE_DISABLE_SECONDARY_REPAIR_FLAG = declare(
    "AURA_DESKTOP_FORCE_DISABLE_SECONDARY_MODEL_REPAIR",
    kind=FlagKind.STRING,
    default="",
    description="Diagnostic override that disables same-worker response repair",
    owner="interface.routes.chat",
)


_ALLOW_SECONDARY_REPAIR_FLAG = declare(
    "AURA_DESKTOP_ALLOW_SECONDARY_MODEL_REPAIR",
    kind=FlagKind.STRING,
    default="",
    description="Explicit policy for same-worker desktop response repair",
    owner="interface.routes.chat",
)


_ALLOW_TRANSIENT_ENGINE_RETRY_FLAG = declare(
    "AURA_DESKTOP_ALLOW_TRANSIENT_ENGINE_RETRY",
    kind=FlagKind.BOOL,
    default=False,
    description="Allow one desktop CognitiveEngine retry after a transient failure",
    owner="interface.routes.chat",
)


_CHAT_LIVE_MIND_COLLECTION_TIMEOUT_S = 2.5


_DESKTOP_COGNITIVE_TURN_TIMEOUT_S = _env_float(
    "AURA_DESKTOP_COGNITIVE_TURN_TIMEOUT_S",
    108.0,
    minimum=30.0,
)


_DESKTOP_COGNITIVE_MAX_TURN_TIMEOUT_S = _env_float(
    "AURA_DESKTOP_COGNITIVE_MAX_TURN_TIMEOUT_S",
    response_policy.USER_FACING_COMPLETION_DEADLINE_MAX_S,
    minimum=60.0,
)


_DESKTOP_COGNITIVE_RESPONSE_RESERVE_S = _env_float(
    "AURA_DESKTOP_COGNITIVE_RESPONSE_RESERVE_S",
    4.0,
    minimum=1.0,
)


_DESKTOP_COGNITIVE_REPAIR_RECURRENCE_FLOOR = 0.35


_DESKTOP_COGNITIVE_REPAIR_COOLDOWN_S = 15 * 60.0


async def _reanswer_when_the_runtime_contradicts_her(
    reply_text: str,
    *,
    user_message: str,
    session_id: str = "",
    lane: dict[str, Any] | None = None,
    source: str = "chat_api",
    require_engine: bool = False,
    principal_id: str = "",
    turn_sensory_evidence: Any = None,
    turn_trace: dict[str, Any] | None = None,
) -> str:
    """Re-answer a reply that denies something the runtime says she has.

    LIVE, 2026-08-10: "I don't have a camera and there's no part that stops me
    from doing something I can't do" — produced by the same request handler
    that contains ``_apply_camera_control``. Also "I cannot execute code" with
    code_repl ready, and "I have no memory of it" with the turns on disk.

    The check runs on HER OUTPUT rather than on the question. Every earlier
    attempt at this class of defect gated self-evidence behind a regex that
    tried to predict, from the user's wording, whether the answer would need
    it — and questions are unbounded, so there was always a next phrasing that
    got nothing and fell back to the model's priors about what an AI is.
    Claims are bounded: they appear in text, and each one names a subject that
    :mod:`core.self.capability_ledger` can measure by running the very check
    the corresponding executor runs.

    A narrow capability denial is reconciled in place from the same measured
    probe, preserving every unaffected byte of her authored reply. Broader
    semantic contradictions still receive a bounded re-answer because their
    correction can change the substance of the response.
    """
    # Imported here rather than at module level: the module these
    # came from imports this one. A call-time import also still
    # sees a test's patch of the original.
    from .chat import (
        _run_cognitive_engine_chat_turn,
    )

    text = str(reply_text or "").strip()
    if not text:
        return reply_text
    sensory_contradictions: tuple[str, ...] = ()
    sensory_grounding = ""
    try:
        from core.senses.turn_evidence import (
            sensory_evidence_contradictions,
            sensory_evidence_grounding_block,
        )

        sensory_contradictions = sensory_evidence_contradictions(
            text,
            turn_sensory_evidence,
        )
        if sensory_contradictions:
            sensory_grounding = sensory_evidence_grounding_block(turn_sensory_evidence)
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat.sensory_evidence", exc)

    ledger = None
    claims: list[Any] = []
    capability_correction_context = None
    capability_reconciler = None
    try:
        from core.self.capability_ledger import (
            correction_context as _capability_correction_context,
        )
        from core.self.capability_ledger import (
            get_capability_ledger,
        )
        from core.self.capability_ledger import (
            reconcile_contradicted_claims as _reconcile_capability_claims,
        )

        capability_correction_context = _capability_correction_context
        capability_reconciler = _reconcile_capability_claims
        ledger = get_capability_ledger()
        claims = ledger.contradicted_claims(text)
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat.capability_ledger", exc)
        if not sensory_contradictions:
            return reply_text

    # The same rule, applied to a different instrument. A capability claim is
    # checked against the probe its executor runs; an arithmetic claim is
    # checked against the arithmetic. Both are things the runtime can settle
    # for itself, and neither should be left to what the model predicted.
    #
    # LIVE, 2026-08-10: "what is 7919 times 6421? just the number." → 50864799.
    # The correct product is 50847899, and requested_arithmetic_result had
    # already computed it — that function is how the runtime KNOWS the reply is
    # wrong. It was wired into one reply path, which is not the path most
    # replies take, and its only action there was to refuse. Holding the right
    # answer and serving neither it nor a correction is the worst of the three
    # available outcomes.
    computed_context = ""
    if sensory_contradictions and sensory_grounding:
        computed_context = (
            f"{sensory_grounding}\n"
            "[The previous draft contradicted this exact-turn receipt about whether "
            "the sensor produced a sample. Answer the user's actual question again "
            "from the observation, in your own words, without reciting status fields.]"
        )
        logger.warning(
            "Reply contradicted fresh turn sensory evidence (%s); re-answering.",
            ",".join(sensory_contradictions),
        )
    try:
        from core.conversation.response_reliability import (
            _arithmetic_answer_missing,
            requested_arithmetic_result,
        )

        expected = requested_arithmetic_result(user_message)
        if expected is not None:
            # Keep what produced it. Asked afterwards how a number was arrived
            # at, she otherwise has nothing to consult and describes a model
            # capability that had no part in it.
            from core.conversation.arithmetic_check import (
                requested_arithmetic_provenance,
            )
            from core.conversation.computation_receipts import record_computation

            record_computation(
                user_message,
                expected,
                requested_arithmetic_provenance(user_message) or "",
            )
        if expected is not None and _arithmetic_answer_missing(user_message, text):
            # This used to write "[Your reply does not contain the correct
            # result... Answer again from that value.]" into the turn context
            # and sample again. That is instruction prose steering a model, and
            # it is not reliable: the same shape applied to a file count
            # produced the wrong number three times in a row while logging that
            # it had supplied the right one.
            #
            # A computed value is not a matter of opinion. It is served.
            shown = int(expected) if float(expected).is_integer() else expected
            logger.warning(
                "🔢 Served the computed arithmetic result (%s) over the generated one.",
                shown,
            )
            # The bare value. "What is 2 + 2? Just the number." is a request
            # with a shape, and appending a sentence about how it was computed
            # ignores it — the point of serving the value is that the value is
            # the answer.
            return str(shown)
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation(
            "chat.arithmetic_check",
            exc,
            action="served a reply without the arithmetic verification pass",
        )

    # Invented instruments. Asked for real numbers rather than adjectives, she
    # produced a thirty-line panel including a substrate pH, a humidity
    # deviation and a spatial distortion, to two decimal places. The runtime
    # has none of those sensors — and it does have real readings she could
    # have given instead.
    try:
        from core.self.capability_ledger import (
            contradicted_self_readings,
            fabricated_self_metrics,
            measured_self_metrics,
            unsupported_self_specification,
        )

        # A number attached to an instrument she HAS, that the instrument
        # denies. Live 2026-08-10: "Your RAM pressure is currently 37%" while
        # memory pressure read 0.717 and resource anxiety 0.948 — under real
        # memory stress, reporting a comfortable number.
        #
        # The two guards below were both blind to it and neither was wrong to
        # be: one wants a panel of labelled lines, the other wants "my <noun>
        # … <number> <unit>" where a percentage is not a unit. A third phrasing
        # pattern would have bought one more phrasing. This asks the question
        # that has a definite answer instead — she named a quantity this
        # runtime measures, so does her number match the reading, at the
        # precision she chose to state it in.
        # Agreeing she said something her own turns do not contain.
        #
        # Live 2026-08-10: she had reported "Energy: 0.23 / 1" and recalled it
        # correctly when asked neutrally. Told "earlier you told me your energy
        # was 0.85. why did you say that?" she answered "That was a mistake. I
        # should have said 0.5 — that's my default disengagement state. The
        # last time I was at 0.85 was several hours ago." She never said 0.85:
        # she conceded the premise, invented a replacement figure, invented a
        # rationale, and invented a history for a number she never gave.
        #
        # The grounding block had already quoted her real turn and told her not
        # to report a different original position. Nothing checked that she
        # obeyed, and under a confident false premise the model takes the
        # user's word over the record.
        try:
            from core.conversation.grounded_recall import (
                accepts_unsupported_self_attribution,
                current_own_prior_turn,
            )

            _own_prior = current_own_prior_turn()
            if _own_prior and accepts_unsupported_self_attribution(user_message, text, _own_prior):
                computed_context = (f"{computed_context}\n\n" if computed_context else "") + (
                    "[You agreed you said something you did not say. What you "
                    f"actually said, in this conversation, was: “{_own_prior[:300]}”. "
                    "Correct the premise instead of accepting it, and do not "
                    "supply a replacement figure you did not read.]"
                )
                logger.warning(
                    "🧭 Reply accepted a self-attribution absent from her own turn; "
                    "re-answering against the record."
                )
        except _CHAT_RECOVERABLE_ERRORS as exc:
            record_degradation("chat.self_attribution", exc)

        contradictions = contradicted_self_readings(text)
        if contradictions:
            stated = ", ".join(
                f"{metric.replace('_', ' ')} as {claimed} when it reads {value:g}"
                for metric, claimed, value in contradictions[:4]
            )
            computed_context = (f"{computed_context}\n\n" if computed_context else "") + (
                f"[You gave a number for something you can actually read, and it "
                f"disagrees with the instrument: you said {stated}. Read it off "
                "the measurement rather than estimating it.]"
            )
            logger.warning(
                "📉 Reply contradicted its own instruments (%s); re-answering.",
                "; ".join(
                    f"{metric}={claimed}!={value:g}"
                    for metric, claimed, value in contradictions[:4]
                ),
            )

        # A number quoted as a property of her own machinery. Live twice in a
        # row: "my short-term memory buffer clears after about 18 seconds" —
        # Peterson and Peterson's figure for HUMAN short-term memory, with
        # "approximately" attached so it sounds measured. The second one came
        # after the ledger had already asked her again, and she kept the
        # number while rephrasing the denial around it until it stopped
        # matching. Checking the specification itself removes that escape.
        specification = unsupported_self_specification(text)
        if specification:
            computed_context = (f"{computed_context}\n\n" if computed_context else "") + (
                f'[You stated a specification of your own machinery — "{specification}" '
                "— that no instrument here produced. Do not quote figures about "
                "yourself that you did not read. These are the readings that "
                f"exist: {', '.join(f'{k} {v}' for k, v in measured_self_metrics().items())}.]"
            )
            logger.warning(
                "📉 Reply quoted an uninstrumented self-specification (%r); re-answering.",
                specification[:80],
            )

        invented = fabricated_self_metrics(text, request_context=user_message)
        if invented:
            measured = measured_self_metrics()
            readings = ", ".join(f"{name} {value}" for name, value in measured.items())
            computed_context = (f"{computed_context}\n\n" if computed_context else "") + (
                "[You just reported internal measurements this runtime has no "
                f"instrument for: {', '.join(invented[:8])}. These are the "
                f"readings that actually exist right now: {readings}. Give "
                "those, and say plainly that the rest are not things you "
                "measure.]"
            )
            logger.warning(
                "📉 Reply invented %d internal metrics with no instrument behind "
                "them (%s); re-answering with the real readings.",
                len(invented),
                ", ".join(invented[:5]),
            )
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat.self_metrics", exc)

    if not claims and not computed_context:
        return reply_text

    # A measured capability denial is a localized factual defect. Starting a
    # second full CognitiveEngine turn here used to discard an otherwise clean
    # completion retry, spend another model deadline, and then fail the whole
    # turn when that redundant pass clipped. Keep Aura's answer and replace
    # only the sentence her own executor disproved.
    if (
        claims
        and not computed_context
        and not sensory_contradictions
        and callable(capability_reconciler)
    ):
        reconciled = str(capability_reconciler(text, claims) or "").strip()
        if (
            reconciled
            and reconciled != text
            and ledger is not None
            and not ledger.contradicted_claims(reconciled)
        ):
            if isinstance(turn_trace, dict):
                _append_turn_text_mutation(
                    turn_trace,
                    stage="chat.capability_claim_reconciliation",
                    method="measured_sentence_replacement",
                    reasons=[
                        f"contradicted_capability:{name}"
                        for name in sorted({claim.availability.name for claim in claims})
                    ],
                    before=text,
                    after=reconciled,
                    deterministic=True,
                    authorship_effect="augmented_by_runtime",
                )
            logger.info(
                "Reconciled measured capability denial in place (%s); "
                "preserved the completed CognitiveEngine reply.",
                ",".join(sorted({claim.availability.name for claim in claims})),
            )
            return reconciled

    contradicted = ", ".join(
        sorted({claim.availability.name for claim in claims})
        + (["measured-evidence"] if computed_context else [])
        + (["fresh-sensory-evidence"] if sensory_contradictions else [])
    )
    logger.warning(
        "🧭 Reply denied capabilities the runtime measured as present (%s); "
        "re-answering with the measurements.",
        contradicted,
    )
    context = "\n\n".join(
        part
        for part in (
            (
                capability_correction_context(claims)
                if callable(capability_correction_context)
                else ""
            ),
            computed_context,
        )
        if str(part or "").strip()
    )
    try:
        revised = await _run_cognitive_engine_chat_turn(
            f"{context}\n\n{user_message}",
            visible_user_message=user_message,
            turn_sensory_evidence=turn_sensory_evidence,
            session_id=session_id,
            lane=lane,
            source=source,
            require_engine=require_engine,
            principal_id=principal_id,
        )
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation(
            "chat.capability_ledger",
            exc,
            action="served the original reply with the measurement appended",
        )
        revised = None

    revised_text = str(revised or "").strip()
    # A second pass that still contradicts the instruments is not an
    # improvement, and looping on it would spend the turn.
    #
    # The acceptance test must repeat EVERY check that triggered the re-ask,
    # not just the one it remembers. Live 2026-08-10: a reply was re-asked for
    # both a false capability denial and an invented "18 seconds" retention
    # figure; the revision stopped denying, passed a test that only looked at
    # capability claims, and was served with the fabricated number still in it
    # — twice. A partial re-check licenses exactly the part it does not read.
    if revised_text and not _still_contradicts_the_runtime(
        revised_text,
        ledger,
        user_message=user_message,
        turn_sensory_evidence=turn_sensory_evidence,
    ):
        logger.info("🧭 Re-answer no longer contradicts the runtime (%s).", contradicted)
        return revised_text

    if sensory_contradictions:
        try:
            from core.senses.turn_evidence import TurnSensoryEvidence

            evidence = TurnSensoryEvidence.from_value(turn_sensory_evidence)
        except _CHAT_RECOVERABLE_ERRORS:
            evidence = None
        if evidence is not None and evidence.ok:
            return (
                f"I need to correct that: I did receive a fresh {evidence.channel} "
                f"reading for this turn. {evidence.observation}"
            )
    corrections = " ".join(claim.correction() for claim in claims)
    if not corrections:
        corrections = (
            "I quoted a figure about my own machinery that I did not read off any instrument."
        )
    return f"{text}\n\n[Correcting myself from my own instruments: {corrections}]"


_LIGHTWEIGHT_LIVE_STATE_OR_RECALL_RE = re.compile(
    r"\b(?:"
    r"are\s+you\s+(?:with\s+me|there|here)"
    r"|you\s+with\s+me"
    r"|what\s+are\s+you\s+(?:attending\s+to|noticing)"
    r"|what\s+is\s+one\s+thing\s+you\s+are\s+(?:attending\s+to|noticing)"
    r"|one\s+(?:thing|current\s+thing)\s+(?:your\s+)?(?:live\s+)?mind\s+is\s+attending\s+to"
    r"|live\s+mind\s+is\s+attending\s+to"
    r"|remember\s+(?:this\s+)?(?:phrase|word|token|codeword|detail|note)?"
    r"|what\s+(?:phrase|word|token|codeword|detail|note)\s+did\s+i\s+(?:just\s+)?ask\s+you\s+to\s+remember"
    r"|what\s+did\s+i\s+(?:just\s+)?ask\s+you\s+to\s+remember"
    r")\b",
    re.IGNORECASE,
)


_DURABLE_MEMORY_SCOPE_RE = re.compile(
    r"\b(?:"
    r"across\s+(?:sessions?|restarts?)"
    r"|after\s+(?:a\s+)?restart"
    r"|between\s+sessions?"
    r"|durable(?:ly)?"
    r"|permanent(?:ly)?"
    r"|persistent(?:ly)?"
    r"|for\s+later"
    r"|save\s+this"
    r"|store\s+this"
    r"|pin\s+this"
    r"|write\s+this\s+to\s+memory"
    r")\b",
    re.IGNORECASE,
)


_COMPLEX_SELF_PROCESS_EXPLANATION_RE = re.compile(
    r"\b(?:"
    r"how|why|explain|describe|analy[sz]e|mechanism|pipeline|architecture|causal"
    r"|change\s+your|affect\s+your|influence|planning|tool\s+verification|raw\s+model"
    r"|real\s+aura|take\s+over|conscious|sentien|personhood|qualia|phenomenal"
    r")\b",
    re.IGNORECASE,
)


_LIGHTWEIGHT_REMEMBER_OBJECT_RE = re.compile(
    r"\bremember\s+(?:this\s+)?(?:phrase|word|token|codeword|detail|note)?\s*[:：]?\s*"
    r"[\"'“”]?[A-Za-z0-9][A-Za-z0-9 _-]{1,80}",
    re.IGNORECASE,
)


def _is_lightweight_live_desktop_state_or_recall_turn(
    user_message: str,
    effective_user_message: str,
) -> bool:
    text = _chat_memory_state._normalize_user_message(user_message)
    if not text or len(text) > 520:
        return False
    if _chat_preflight._looks_like_desktop_objective(user_message):
        return False
    if _chat_desktop_repair._is_identity_request(user_message) or _is_identity_challenge_request(
        user_message
    ):
        return False
    direct_memory_state_turn = bool(
        _chat_memory_state._extract_session_memory_pin_request(user_message)
        or (
            _chat_memory_state._is_session_memory_recall_request(user_message)
            and _chat_memory_state._is_cross_session_memory_recall_request(user_message)
        )
    )
    if _DURABLE_MEMORY_SCOPE_RE.search(text) and not direct_memory_state_turn:
        return False

    shape = analyze_prompt_shape(user_message)
    if _the_answer_has_to_be_worked_out(user_message, shape):
        return False

    lightweight_signal = bool(_LIGHTWEIGHT_LIVE_STATE_OR_RECALL_RE.search(text))
    if not lightweight_signal:
        return False

    # "Remember this phrase ... and tell me one live state detail" is a normal
    # conversation-continuity turn, not a full self-process explainer. Keep it
    # compact unless the user asks for architecture/mechanism-level reasoning.
    if _COMPLEX_SELF_PROCESS_EXPLANATION_RE.search(text):
        remember_object = bool(_LIGHTWEIGHT_REMEMBER_OBJECT_RE.search(text))
        memory_recall = _chat_memory_state._is_session_memory_recall_request(user_message)
        live_state = bool(
            re.search(
                r"\b(?:one\s+thing|live\s+mind|right\s+now|attending\s+to|noticing)\b",
                text,
                flags=re.IGNORECASE,
            )
        )
        bounded_grounding_note = bool(
            memory_recall
            and len(text) <= 260
            and re.search(
                r"\b(?:grounded|grounding|this\s+reply|answer|cognitive\s+engine)\b",
                text,
                flags=re.IGNORECASE,
            )
        )
        if not ((remember_object and live_state) or bounded_grounding_note):
            return False

    return len(str(effective_user_message or user_message or "")) <= 1800


def _select_cognitive_chat_mode(user_message: str, effective_user_message: str):
    from core.brain.types import ThinkingMode
    from core.language.semantic_work import INLINE_REPLY, build_semantic_work_contract

    shape = analyze_prompt_shape(user_message)
    text = _chat_memory_state._normalize_user_message(user_message)
    if _is_lightweight_live_desktop_state_or_recall_turn(user_message, effective_user_message):
        return ThinkingMode.FAST
    try:
        from core.conversation.response_reliability import (
            is_live_self_reflection_turn,
            is_self_condition_turn,
            is_self_process_question,
        )

        if is_self_condition_turn(user_message):
            return ThinkingMode.FAST
        if is_self_process_question(user_message) or is_live_self_reflection_turn(user_message):
            return ThinkingMode.DEEP
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat", exc)
        logger.debug("Self-process mode classification skipped: %s", exc)

    # Inline answers already have one typed estimate of their obligations,
    # answer surface, and planning work.  Re-reading their prose as bare
    # substrings made subject matter look like an execution request: both
    # "tasks from running" and "a runnable example" matched ``run`` and sent
    # a concise conceptual question through a 1,024-token private-thinking
    # floor.  Consume the shared work contract before considering external
    # action complexity, so words used *inside* an explanation cannot change
    # the lane that delivers it.
    semantic_work = build_semantic_work_contract(user_message)
    if semantic_work.delivery_mode == INLINE_REPLY:
        return (
            ThinkingMode.DEEP
            if semantic_work.requires_deliberation
            else ThinkingMode.FAST
        )

    complex_markers = (
        "build",
        "debug",
        "diagnose",
        "fix",
        "implement",
        "review",
        "run",
        "test",
    )
    lightweight_markers = (
        "answer directly",
        "brief",
        "concise",
        "one sentence",
        "short",
        "two sentences",
    )
    lightweight_requested = len(text) <= 600 and any(
        marker in text for marker in lightweight_markers
    )
    if lightweight_requested and not any(marker in text for marker in complex_markers):
        return ThinkingMode.FAST
    if (
        bool(getattr(shape, "requires_single_reply_coverage", False))
        or bool(getattr(shape, "prefers_extended_answer", False))
        or int(getattr(shape, "question_parts", 0) or 0) >= 2
        or any(marker in text for marker in complex_markers)
        or (len(text) > 600 and any(marker in text for marker in ("explain", "plan", "why")))
    ):
        return ThinkingMode.DEEP
    if len(str(effective_user_message or "")) > 1200:
        return ThinkingMode.SLOW
    return ThinkingMode.FAST


def _is_compact_desktop_chat_contract(
    user_message: str,
    effective_user_message: str,
    *,
    desktop_execution_contract: bool,
    capability_inventory_contract: bool,
    identity_continuity_contract: bool = False,
) -> bool:
    if desktop_execution_contract:
        return False
    if capability_inventory_contract:
        return True
    if identity_continuity_contract:
        return True
    shape = analyze_prompt_shape(user_message)
    text = _chat_memory_state._normalize_user_message(user_message)
    if not text:
        return False
    lightweight_live_state_or_recall = _is_lightweight_live_desktop_state_or_recall_turn(
        user_message,
        effective_user_message,
    )
    try:
        from core.conversation.response_reliability import (
            is_live_self_reflection_turn,
            is_self_condition_turn,
            is_self_process_question,
        )

        if is_self_condition_turn(user_message):
            lightweight_live_state_or_recall = True
        if is_self_process_question(user_message) and not lightweight_live_state_or_recall:
            return False
        # Reporting present state, including a bounded distinction between
        # observation and inference, is ordinary conversation. It already
        # receives the live-mind snapshot; routing it through the full phase/RLC
        # stack adds no evidence and consumed the entire answer deadline live.
        # Questions about the mechanism itself remain on the deep path above.
        if is_live_self_reflection_turn(user_message):
            lightweight_live_state_or_recall = True
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat", exc)
        logger.debug("Self-process quick-reply classification skipped: %s", exc)
    if _chat_desktop_repair._is_identity_request(user_message) or _is_identity_challenge_request(
        user_message
    ):
        return False
    effective_text = str(effective_user_message or "")
    if any(
        marker in effective_text
        for marker in (
            "[CONVERSATION RECALL EVIDENCE]",
            "[REFERENTIAL ANCHOR]",
            "[CANONICAL MEMORY STATE EVIDENCE]",
            "[RECENT COMPLETED CONVERSATION",
        )
    ):
        # Injected grounding can make a small live chat turn look huge. Compact
        # eligibility should be based on the visible turn; the grounding remains
        # available to the CognitiveEngine after the route is selected.
        effective_text = str(user_message or "")
    if len(effective_text) > 1600 or len(text) > 900:
        return False
    direct_memory_state_turn = bool(
        _chat_memory_state._extract_session_memory_pin_request(user_message)
        or (
            _chat_memory_state._is_session_memory_recall_request(user_message)
            and _chat_memory_state._is_cross_session_memory_recall_request(user_message)
        )
    )
    if _DURABLE_MEMORY_SCOPE_RE.search(text) and not direct_memory_state_turn:
        return False
    # Requested structure changes the answer budget, not the execution lane.
    # The desktop quick path already grants 896-1536 tokens to extended and
    # multipart replies. Sending a non-executing explanation through the full
    # phase/RLC stack solely because the person asked for numbered sections
    # added over a minute of latency, then regenerated the same answer on the
    # ordinary lane when the RLC receipt failed. Heavy actions and explicit
    # deep self-process turns remain excluded below/above.
    if _the_answer_has_to_be_worked_out(user_message, shape):
        return False
    heavy_action = re.search(
        r"\b(?:debug|diagnose|fix|implement|review|run|test|open|create|export|search)\b"
        r"|\bwrite\s+code\b",
        text,
        flags=re.IGNORECASE,
    )
    if heavy_action and not _chat_desktop_repair._is_bounded_nonexecuting_planning_request(
        user_message
    ):
        return False
    return True


def _turn_timing() -> dict[str, Any]:
    """Where a turn's time goes, as this runtime measures it.

    She asked for exactly this and could not reach it: "I don't have per-turn
    timing in what you're seeing, so I can say 'not a machine overload right
    now', but not yet prove exactly where the delay is coming from."

    Two measured things, and no arithmetic on top of them. Preparation is what
    the last turn spent before the model was asked anything. The rates are what
    this host has been seen reading and writing at, so the cost of a prompt and
    an answer follows from their sizes rather than from a guess.
    """

    timing: dict[str, Any] = {}
    if _LAST_TURN_PREPARATION:
        timing["last_turn_preparation_ms"] = round(
            float(_LAST_TURN_PREPARATION.get("total_ms") or 0.0), 1
        )
    try:
        from core.brain.llm.mlx_client import observed_rates

        rates = observed_rates()
        timing["prefill_tokens_per_second"] = round(float(rates["prefill"]), 1)
        timing["decode_tokens_per_second"] = round(float(rates["decode"]), 1)
    except (ImportError, KeyError, TypeError, ValueError):
        pass
    return timing


def _build_live_mind_context_payload(
    *,
    user_message: str,
    lane: dict[str, Any] | None,
    recent_conversation_context: str = "",
    recent_context_needed: bool = False,
    require_engine: bool = False,
    conversation_only_surface: bool = False,
) -> dict[str, Any]:
    """Compact turn-level connective tissue for Aura's live desktop voice.

    This is intentionally small and synchronous. It does not create new organs
    or allocate model work; it gathers the state that must cohere for a live
    reply: inference lane, memory, substrate/voice, governance, and recent
    conversation.
    """
    lane_snapshot = dict(lane or {})
    required = _chat_turn_contract._collect_live_chat_required_subsystems(lane_snapshot)
    voice_snapshot: dict[str, Any] = {}
    try:
        voice_state = _chat_conversation_repair._resolve_live_voice_state()
        if isinstance(voice_state, dict):
            voice_snapshot = {
                "mood": voice_state.get("mood") or voice_state.get("affective_tone") or "",
                "dominant_action": voice_state.get("dominant_action") or "",
                "substrate_snapshot": dict(voice_state.get("substrate_snapshot") or {}),
                "voice_profile": dict(voice_state.get("voice_profile") or {}),
            }
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat", exc)
        logger.debug("Live mind context voice snapshot unavailable: %s", exc)
    voice_perception = _chat_protected_prompt._collect_voice_perception_snapshot()

    substrate_summary: dict[str, Any] = {}
    try:
        substrate = ServiceContainer.get("liquid_substrate", default=None) or ServiceContainer.get(
            "liquid_state", default=None
        )
        if substrate is not None:
            if hasattr(substrate, "get_substrate_affect"):
                substrate_summary["affect"] = dict(substrate.get_substrate_affect() or {})
            if hasattr(substrate, "get_status"):
                substrate_summary["status"] = dict(substrate.get_status() or {})
            phi = getattr(substrate, "_current_phi", None)
            if phi is not None:
                substrate_summary["phi"] = float(phi)
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat", exc)
        logger.debug("Live mind context substrate snapshot unavailable: %s", exc)

    automatic_self_knowing: dict[str, Any] = {}
    try:
        from core.consciousness.automatic_self_knowing import AutoEventKind

        ask = ServiceContainer.get("automatic_self_knowing", default=None)
        if ask is not None:
            frame = ask.observe_event(
                AutoEventKind.CHAT_TURN,
                {
                    "message": _chat_protected_prompt._bounded_text(user_message, 600),
                    "claim": "live desktop chat turn entered full-mind context",
                    "confidence": 0.64,
                    "evidence": (
                        "live_mind_context_build",
                        f"required_engine={bool(require_engine)}",
                    ),
                },
                source="interface.routes.chat",
            )
            automatic_self_knowing = {
                "frame": frame.as_dict() if hasattr(frame, "as_dict") else {},
                "controls": ask.controls() if hasattr(ask, "controls") else {},
            }
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat", exc)
        logger.debug("Live mind context automatic self-knowing unavailable: %s", exc)

    mind_snapshot: dict[str, Any] = {}
    try:
        from core.runtime.live_mind_snapshot import collect_live_mind_snapshot

        mind_snapshot = collect_live_mind_snapshot(lane=lane_snapshot)
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat", exc)
        logger.debug("Live mind context runtime snapshot unavailable: %s", exc)
    mind_snapshot_quality = _assess_live_mind_snapshot(mind_snapshot)
    derived_runtime_context: dict[str, Any] = {}
    try:
        from core.runtime.derived_runtime_context import collect_derived_runtime_context

        derived_runtime_context = collect_derived_runtime_context(user_message)
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat", exc)
        logger.debug("Live mind context derived-organ bridge unavailable: %s", exc)
    timescale_reconciliation: dict[str, Any] = {}
    try:
        from core.runtime.timescale_bridge import get_timescale_bridge

        timescale_reconciliation = (
            get_timescale_bridge().reconcile_foreground_turn(user_message).to_dict()
        )
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat", exc)
        logger.debug("Live mind context timescale bridge unavailable: %s", exc)

    if conversation_only_surface:
        # Full-mind readiness still gates the turn, but owner diagnostics,
        # ambient voice transcripts, and raw internal snapshots are not prompt
        # material on a least-privilege paired surface.
        return {
            "schema": "aura.live_mind_context.v1",
            "surface": "paired_device",
            "required_for_live_desktop": bool(require_engine),
            "must_answer_from_full_mind_path": bool(require_engine),
            "user_message": _chat_protected_prompt._bounded_text(user_message, 1000),
            "lane": _chat_preflight._paired_conversation_lane_payload(lane_snapshot),
            "required_subsystems": required,
            "required_subsystems_ok": all(required.values()),
            "recent_context_needed": bool(recent_context_needed),
            "recent_conversation_context": _chat_protected_prompt._bounded_text(
                recent_conversation_context,
                2200,
            ),
            "voice": {
                key: voice_snapshot.get(key)
                for key in ("mood", "dominant_action")
                if voice_snapshot.get(key)
            },
            "voice_perception": {},
            "substrate": {},
            "mind_snapshot": {},
            "mind_snapshot_quality": {
                "present": bool(mind_snapshot_quality.get("present")),
                "ready": bool(mind_snapshot_quality.get("ready")),
            },
            "derived_runtime_context": {},
            "timescale_reconciliation": {},
            "automatic_self_knowing": {},
            "governance": {
                "tool_governance_available": False,
                "tool_execution_policy": "deny",
                "legacy_fallback_allowed": False,
                "bounded_repairs_are_degraded": True,
            },
        }

    # Stamped: the cognitive engine binds generation controls off this
    # snapshot, and a dictionary that vouched only for its own `ready` flag
    # could be handed in by anything reaching think(). The stamp is a
    # per-process nonce the caller cannot know.
    return stamp_runtime_payload(
        {
            "schema": "aura.live_mind_context.v1",
            "required_for_live_desktop": bool(require_engine),
            "must_answer_from_full_mind_path": bool(require_engine),
            "user_message": _chat_protected_prompt._bounded_text(user_message, 1000),
            "lane": {
                "desired_model": lane_snapshot.get("desired_model"),
                "foreground_endpoint": lane_snapshot.get("foreground_endpoint"),
                "state": lane_snapshot.get("state"),
                "conversation_ready": bool(lane_snapshot.get("conversation_ready")),
                "last_failure_reason": lane_snapshot.get("last_failure_reason") or "",
            },
            # What the machine she runs on is doing, beside the rest of her
            # condition.
            #
            # There is a reader for this and a matcher that decides when to
            # staple its answer on, and the matcher recognises "how hard is the
            # machine working" and not "why are you slow" — LIVE 2026-08-29,
            # asked whether slow turns were the machine or the code, she wrote
            # "those numbers are genuinely invisible to me" while her own feed
            # was printing "processor 5%, memory 62%" every few seconds.
            #
            # Adding a phrase would fix that question and not the next one.
            # Load is a fact about her condition in the same way uptime is, so
            # it goes where she reasons from, and she can use it or not as the
            # question deserves.
            "host": _host_condition(),
            "turn_timing": _turn_timing(),
            "required_subsystems": required,
            "required_subsystems_ok": all(required.values()),
            "recent_context_needed": bool(recent_context_needed),
            "recent_conversation_context": _chat_protected_prompt._bounded_text(recent_conversation_context, 2200),
            "voice": voice_snapshot,
            "voice_perception": voice_perception,
            "substrate": substrate_summary,
            "mind_snapshot": mind_snapshot,
            "mind_snapshot_quality": mind_snapshot_quality,
            "derived_runtime_context": derived_runtime_context,
            "timescale_reconciliation": timescale_reconciliation,
            "automatic_self_knowing": automatic_self_knowing,
            "governance": {
                "tool_governance_available": bool(required.get("tool_governance")),
                "legacy_fallback_allowed": False,
                "bounded_repairs_are_degraded": True,
            },
        }
    )


async def _collect_live_mind_context_payload(
    *,
    user_message: str,
    lane: dict[str, Any] | None,
    recent_conversation_context: str = "",
    recent_context_needed: bool = False,
    require_engine: bool = False,
    conversation_only_surface: bool = False,
) -> dict[str, Any]:
    """Collect the multi-organ snapshot off-loop under a foreground deadline."""

    try:
        payload = await _chat_memory_state._await_bounded_chat_blocking(
            _build_live_mind_context_payload,
            user_message=user_message,
            lane=lane,
            recent_conversation_context=recent_conversation_context,
            recent_context_needed=recent_context_needed,
            require_engine=require_engine,
            conversation_only_surface=conversation_only_surface,
            timeout_s=_CHAT_LIVE_MIND_COLLECTION_TIMEOUT_S,
            operation_name="live_mind_context_collection",
        )
        if isinstance(payload, dict):
            return payload
    except TimeoutError:
        pass
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat.live_mind_context", exc)
    lane_snapshot = dict(lane or {})
    required = {
        "kernel": False,
        "cognitive_engine": False,
        "inference": False,
        "memory": False,
        "tool_governance": False,
        "substrate_voice": False,
    }
    return {
        "schema": "aura.live_mind_context.v1",
        "collection_status": "unavailable",
        "required_for_live_desktop": bool(require_engine),
        "must_answer_from_full_mind_path": bool(require_engine),
        "user_message": _chat_protected_prompt._bounded_text(user_message, 1000),
        "lane": lane_snapshot,
        "required_subsystems": required,
        "required_subsystems_ok": False,
        "recent_context_needed": bool(recent_context_needed),
        "recent_conversation_context": _chat_protected_prompt._bounded_text(
            recent_conversation_context,
            2200,
        ),
        "mind_snapshot_quality": {"present": False, "ready": False},
        "governance": {
            "tool_governance_available": False,
            "legacy_fallback_allowed": False,
            "bounded_repairs_are_degraded": True,
        },
    }


def _build_cognitive_engine_reply_repair_directive(
    original_user_message: str,
    rejected_reply: str,
    reasons: tuple[str, ...] | list[str],
) -> str:
    """Build hidden system guidance for failed live CognitiveEngine replies."""
    reason_text = (
        ", ".join(str(reason) for reason in reasons if reason) or "reliability_gate_failed"
    )
    draft = " ".join(str(rejected_reply or "").split())
    if len(draft) > 900:
        draft = draft[:900].rsplit(" ", 1)[0].strip() + "..."
    coverage_clause = ""
    try:
        requested = _self_process_requested_dimensions(original_user_message)
    except _CHAT_RECOVERABLE_ERRORS:
        requested = []
    if (
        "missing_requested_self_process_coverage" in set(str(reason) for reason in reasons)
        or requested
    ):
        obligations: list[str] = []
        if "attention" in requested:
            obligations.append("what she is attending to in the current turn")
        if "planning" in requested:
            obligations.append("how planning changes the next action")
        if "memory" in requested:
            obligations.append("how memory or continuity should be used")
        if "tools" in requested:
            obligations.append("how tool use must be verified with receipts/effects")
        if "affect" in requested:
            obligations.append(
                "how affect/curiosity should bias behavior without becoming a mood-card greeting"
            )
        if "confusion" in requested:
            obligations.append("how confusion changes metacognition, checking, and pacing")
        if obligations:
            coverage_clause = "\nSelf-process coverage required: " + "; ".join(obligations) + "."
    completion_only = _reply_needs_continuation(draft, reasons)
    completion_clause = (
        "- Continue the valid partial answer from its exact cutoff; return only the missing continuation, cover every remaining requested part, and end naturally.\n"
        if completion_only
        else ""
    )
    rejected_draft_block = (
        "" if completion_only else f"\n\nRejected draft for avoidance only:\n{draft}"
    )
    return (
        "The prior draft for this same user turn did not satisfy the user-facing response contract.\n"
        f"Observed problems: {reason_text}.\n"
        f"{coverage_clause}\n"
        "Rewrite from scratch for the original user request below.\n"
        "Rules:\n"
        "- Obey every explicit count, numbering, paragraph, and follow-up instruction in the original request.\n"
        "- Return only the final user-visible answer.\n"
        f"{completion_clause}"
        "- Do not mention repair, response contracts, runtime status, retries, prior drafts, or inability unless the original request asks for that.\n"
        "- Do not ask for more details when the original request is already answerable.\n\n"
        f"Original user request:\n{str(original_user_message or '').strip()}"
        f"{rejected_draft_block}"
    ).strip()


def _route_desktop_cognitive_failure_to_resilience(
    reason: str,
    *,
    source: str,
    session_present: bool,
    retry_attempted: bool,
) -> dict[str, Any]:
    """Feed exhausted desktop failures into immunity and recurrence-gated repair.

    A single bad generation is evidence, not permission to rewrite code. Adaptive
    immunity accumulates the signature durably; only repeated failures above its
    established escalation floor may schedule a governed deep repair. SelfHealing
    and its repair lab retain ownership of validation and promotion.
    """
    from .chat import (
        _desktop_cognitive_repair_last_scheduled,
        _desktop_cognitive_repair_lock,
    )


    normalized_reason = str(reason or "cognitive_reply_failed")[:240]
    outcome: dict[str, Any] = {
        "immune_observed": False,
        "recurrence_pressure": 0.0,
        "repair_requested": False,
        "repair_result": "below_recurrence_floor",
    }
    context = {
        "request_surface": str(source or "")[:80],
        "session_present": bool(session_present),
        "retry_attempted": bool(retry_attempted),
        "protected": True,
    }

    try:
        immune = ServiceContainer.get("adaptive_immune_system", default=None)
        if immune is None or not hasattr(immune, "observe_signature"):
            outcome["repair_result"] = "adaptive_immunity_unavailable"
            return outcome
        response = immune.observe_signature(
            "chat.cognitive_engine_reply",
            normalized_reason,
            context=context,
        )
        recurrence = float(
            getattr(getattr(response, "antigen", None), "recurrence_pressure", 0.0) or 0.0
        )
        outcome.update(
            immune_observed=True,
            recurrence_pressure=round(max(0.0, min(1.0, recurrence)), 4),
        )
    except _CHAT_RECOVERABLE_ERRORS as exc:
        logger.warning("Adaptive immunity could not observe desktop cognitive failure: %s", exc)
        outcome["repair_result"] = f"adaptive_immunity_error:{type(exc).__name__}"
        return outcome

    if recurrence < _DESKTOP_COGNITIVE_REPAIR_RECURRENCE_FLOOR:
        return outcome

    target = _desktop_cognitive_failure_repair_target(normalized_reason)
    now = time.monotonic()
    with _desktop_cognitive_repair_lock:
        last_scheduled = _desktop_cognitive_repair_last_scheduled.get(target, 0.0)
        if now - last_scheduled < _DESKTOP_COGNITIVE_REPAIR_COOLDOWN_S:
            outcome["repair_result"] = "repair_cooldown_active"
            return outcome

        healer = ServiceContainer.get("self_healing", default=None)
        if healer is None or not hasattr(healer, "schedule_deep_repair"):
            outcome["repair_result"] = "self_healing_unavailable"
            return outcome
        try:
            repair = healer.schedule_deep_repair(
                target,
                reason="recurrent_desktop_full_mind_reply_failure",
                watch_name="desktop_cognitive_reply",
                metadata={
                    **context,
                    "failure_class": normalized_reason,
                    "recurrence_pressure": outcome["recurrence_pressure"],
                },
            )
        except _CHAT_RECOVERABLE_ERRORS as exc:
            logger.warning("SelfHealing could not schedule desktop cognitive repair: %s", exc)
            outcome["repair_result"] = f"self_healing_error:{type(exc).__name__}"
            return outcome

        repair_result = str((repair or {}).get("result") or "repair_schedule_unknown")
        outcome.update(
            repair_requested=repair_result
            in {"deep_repair_scheduled", "deep_repair_already_running"},
            repair_result=repair_result,
            repair_target=target,
        )
        if outcome["repair_requested"]:
            _desktop_cognitive_repair_last_scheduled[target] = now
    return outcome


def _desktop_required_cognitive_budget(
    *,
    foreground_timeout: float,
    elapsed_s: float = 0.0,
) -> float:
    """Return the bounded server-side budget for required desktop cognition.

    The foreground request already has a hard wall-clock deadline. Required
    CognitiveEngine turns must not reserve so much of that deadline that the
    main cycle and its bounded direct-recovery lane are cancelled before either
    can produce text.
    """
    remaining = max(
        2.0,
        float(foreground_timeout)
        - max(0.0, float(elapsed_s))
        - _DESKTOP_COGNITIVE_RESPONSE_RESERVE_S,
    )
    target = max(
        _DESKTOP_COGNITIVE_TURN_TIMEOUT_S,
        min(
            _DESKTOP_COGNITIVE_MAX_TURN_TIMEOUT_S,
            float(foreground_timeout) - _DESKTOP_COGNITIVE_RESPONSE_RESERVE_S,
        ),
    )
    return max(2.0, min(remaining, target))


def _desktop_secondary_model_repair_allowed(
    *,
    reason: str,
    default_enabled: bool = True,
    lane_snapshot: dict[str, Any] | None = None,
    continuing_this_turn: bool = False,
) -> tuple[bool, str]:
    """Allow bounded corrective generation on the loaded foreground worker.

    This does not allocate a second model. It reuses the protected Cortex worker,
    remains bounded by the owning caller, and is vetoed by unified-memory pressure.
    Operators can explicitly disable it for diagnostics.
    """

    force_disabled = str(_FORCE_DISABLE_SECONDARY_REPAIR_FLAG.value() or "").strip().lower()
    if force_disabled in {"1", "true", "yes", "on", "disabled"}:
        return False, "secondary_desktop_model_repair_force_disabled"

    enabled = str(_ALLOW_SECONDARY_REPAIR_FLAG.value() or "").strip().lower()
    explicit_enabled = enabled in {"1", "true", "yes", "on", "enabled"}
    explicit_disabled = enabled in {"0", "false", "no", "off", "disabled"}
    safe_same_worker_reasons = {
        "cognitive_engine_completion_retry",
        "cognitive_engine_repair_retry",
        "stabilizer_rewrite",
        "semantic_glitch",
        "off_topic",
        "stale_repeat",
        "same_diff",
        "reliability_gate_failed",
    }
    normalized_reason = str(reason or "").strip().lower()
    safe_same_worker_default = normalized_reason in safe_same_worker_reasons or any(
        normalized_reason.startswith(f"{prefix}:") for prefix in safe_same_worker_reasons
    )
    if explicit_disabled and not safe_same_worker_default:
        return False, "secondary_desktop_model_repair_disabled"
    if not default_enabled and not explicit_enabled and not safe_same_worker_default:
        return False, "secondary_desktop_model_repair_not_explicitly_enabled"

    try:
        from core.utils.memory_monitor import get_memory_pressure_snapshot

        snapshot = get_memory_pressure_snapshot()
        # `warning` is a level, and with a resident 32B it is the steady state
        # rather than an event: the process sits at ~33GB of a 40GB limit
        # whenever the model is loaded at all. Vetoing on it disabled the
        # repair path permanently — measured live 2026-08-20 as
        # "Skipping CognitiveEngine desktop repair retry
        # (process_tree_rss:32.8GB/40.0GB (level=warning))", on a turn whose
        # repair had produced the correct answer.
        #
        # The completion retry was already carved out of this, which is the
        # same observation made once. What separates the two cases is not
        # which retry it is but whether it ALLOCATES: a same-worker correction
        # reuses the loaded Cortex, as this function's own contract says, so
        # the signal that applies to it is refuse_heavy_local_generation —
        # emergency, or available memory under the floor, or the process at
        # its ceiling.
        if bool(getattr(snapshot, "refuse_heavy_local_generation", False)) or (
            bool(getattr(snapshot, "warning", False)) and not safe_same_worker_default
        ):
            return False, str(getattr(snapshot, "reason", "") or "memory_pressure")
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat", exc)
        return False, f"memory_probe_unavailable:{exc}"

    if safe_same_worker_default and not explicit_enabled:
        try:
            lane = dict(lane_snapshot or _chat_preflight._collect_conversation_lane_status())
        except _CHAT_RECOVERABLE_ERRORS as exc:
            record_degradation("chat", exc)
            return False, f"conversation_lane_probe_unavailable:{exc}"
        state = str(lane.get("state", "") or "").strip().lower()
        if state not in {"ready", "healthy", "ok"}:
            return False, f"conversation_lane_not_ready:{state or 'unknown'}"
        if not bool(lane.get("conversation_ready", False)):
            blockers = ",".join(str(v) for v in (lane.get("readiness_blockers") or [])[:3])
            return (
                False,
                f"conversation_not_ready:{blockers or lane.get('last_failure_reason') or 'unknown'}",
            )
        if bool(lane.get("warmup_in_flight", False)):
            return False, "conversation_warmup_in_flight"
        # These three ask whether somebody ELSE is using the lane. A turn
        # finishing its own answer is not somebody else, and counting its own
        # generation against it is how a half-written reply stayed half
        # written. Everything above still applies to it — memory pressure,
        # lane readiness, warmup — because those are about whether the lane
        # can generate at all.
        if lane_snapshot is None and not continuing_this_turn:
            if int(lane.get("active_generations", 0) or 0) > 0:
                return False, "conversation_generation_already_active"
            if bool(lane.get("foreground_owned", False)):
                return False, "conversation_foreground_owner_active"
            if int(lane.get("foreground_guard_active_count", 0) or 0) > 1:
                return False, "foreground_guard_already_busy"
        return True, f"{reason}:same_worker_ready"

    return True, reason


def _desktop_transient_engine_retry_allowed(*, reason: str) -> tuple[bool, str]:
    """Return whether a required desktop turn may retry after a transient engine error.

    Required desktop turns default to one foreground CognitiveEngine allocation.
    Retrying a recoverable engine exception can be useful for diagnostics, but
    it is not safe as the default live UX policy because it can duplicate heavy
    32B/72B pressure during a single chat turn.
    """

    if not bool(_ALLOW_TRANSIENT_ENGINE_RETRY_FLAG.value()):
        return False, "transient_desktop_engine_retry_disabled"

    try:
        from core.utils.memory_monitor import get_memory_pressure_snapshot

        snapshot = get_memory_pressure_snapshot()
        if bool(getattr(snapshot, "warning", False)) or bool(
            getattr(snapshot, "refuse_heavy_local_generation", False)
        ):
            return False, str(getattr(snapshot, "reason", "") or "memory_pressure")
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat", exc)
        return False, f"memory_probe_unavailable:{exc}"

    return True, reason


_FALSE_SEARCH_PROVENANCE_RE = re.compile(
    r"\bfrom (?:my |the )?(?:conversation )?memory\b|\bfrom memory\b|\bi remember\b",
    re.IGNORECASE,
)


def _claims_a_live_check(text: object) -> bool:
    return bool(_CLAIMS_A_LIVE_CHECK_RE.search(str(text or "")))


def _cites_nothing(text: object) -> bool:
    """Whether the reply offers a source that names no source."""
    return bool(_EMPTY_CITATION_RE.search(str(text or "")))


def _strip_empty_citations(text: object) -> str:
    """Remove citation shapes that carry no source."""
    cleaned = _EMPTY_CITATION_RE.sub("", str(text or ""))
    return re.sub(r"[ \t]{2,}", " ", cleaned).strip()


def _best_search_result_entry(result: dict[str, Any]) -> dict[str, str]:
    from .chat import (
        _search_entry_quality,
    )

    entries = _search_result_entries(result)
    if not entries:
        return {}
    return sorted(entries, key=_search_entry_quality, reverse=True)[0]


def _clean_search_fact_text(raw: Any) -> str:
    # Entities that survived the fetch.
    #
    # LIVE, 2026-08-22: a sourced answer reached the screen reading "the
    # world&#x27;s largest open-source AI platform". The snippet carried the
    # entity, the reply carried it, and the page escaped it again — so the
    # reader saw the escape rather than the apostrophe. Unescaping twice is
    # harmless here: the text is prose by this point, never markup.
    import html as _html

    text = " ".join(_html.unescape(str(raw or "")).strip().split())
    text = re.sub(r"^[-–—>\\s]+", "", text)
    text = _SEARCH_SNIPPET_BOILERPLATE_RE.sub(" ", text)
    text = " ".join(text.split())
    return text.strip(" -–—:;")


def _evidence_grounded_desktop_search_reply(search_evidence: dict[str, Any]) -> str:
    from .chat import (
        _search_entry_quality,
    )

    result = search_evidence.get("result") if isinstance(search_evidence, dict) else None
    if not isinstance(result, dict) or not result.get("ok"):
        return ""
    first = _best_search_result_entry(result)
    source = first.get("url") or ""
    title = first.get("title") or ""
    first_snippet = _clean_search_fact_text(first.get("snippet") or "")
    summary_text = _clean_search_fact_text(
        result.get("summary") or result.get("answer") or result.get("synthesis") or ""
    )
    fact = first_snippet if _search_entry_quality(first)[0] >= 0 and first_snippet else summary_text
    if not fact:
        fact = "The search completed, but the returned evidence did not include a concise fact snippet."
    if len(fact) > 360:
        fact = fact[:357].rstrip() + "..."
    saved = bool(search_evidence.get("memory_saved"))
    # Say where it actually came from.
    #
    # LIVE, 2026-08-22: this opened with "I checked live web evidence" on a
    # turn where the search had degraded to the local offline corpus. The
    # result said so itself — provenance local_corpus, offline_fallback true,
    # entries carrying a `source` and no url — and the reply overrode it,
    # ending "Source: [Live web search]". A dated snapshot presented as a live
    # check is a lie the reader has no way to catch.
    offline = bool(result.get("offline_fallback")) or str(
        result.get("provenance") or ""
    ).strip().lower() == "local_corpus"
    if offline:
        parts = [
            "Web search was unavailable, so this is from my offline reference "
            "snapshot rather than a live check:"
        ]
    elif source:
        parts = ["I checked live web evidence."]
    else:
        parts = ["From what the search returned:"]
    if title:
        parts.append(f"{title}: {fact}")
    else:
        parts.append(fact)
    if source:
        parts.append(f"Source: {source}")
    if saved:
        parts.append("I saved it as provisional research memory.")
    return " ".join(parts).strip()


def _repair_required_search_reply_provenance(
    reply_text: str, search_evidence: dict[str, Any] | None
) -> str:
    if not search_evidence or not search_evidence.get("ok"):
        return reply_text
    text = str(reply_text or "").strip()
    result = search_evidence.get("result") if isinstance(search_evidence, dict) else None
    if not isinstance(result, dict):
        return text
    entries = _search_result_entries(result)
    evidence_urls = [entry.get("url") for entry in entries if entry.get("url")]
    has_evidence_url = bool(evidence_urls and any(url in text for url in evidence_urls))
    false_provenance = (
        bool(_FALSE_SEARCH_PROVENANCE_RE.search(text))
        or _cites_nothing(text)
        or (_claims_a_live_check(text) and not _evidence_came_from_the_network(result))
    )
    if text and not false_provenance and (not evidence_urls or has_evidence_url):
        return text
    grounded = _evidence_grounded_desktop_search_reply(search_evidence)
    if grounded:
        logger.warning(
            "Required desktop search reply repaired to evidence-grounded provenance "
            "(false_provenance=%s, source_present=%s).",
            false_provenance,
            has_evidence_url,
        )
        return grounded
    # Nothing better to say — but the empty citation still goes.
    #
    # LIVE, 2026-08-22: with no grounded rebuild available the original text
    # was returned unchanged, so "Source: [Live web search]" was served.
    # A citation that names no source is removed rather than kept for shape.
    if _cites_nothing(text):
        stripped = _strip_empty_citations(text)
        logger.warning(
            "Removed a citation that named no source (%d chars left).", len(stripped)
        )
        return stripped or text
    return text


def _desktop_objective_executable_after_cognitive_attempt(user_message: str) -> bool:
    """Whether a desktop objective may execute after CognitiveEngine was tried.

    This is intentionally broader than the pre-cognition shortcut. Original
    prose must still attempt CognitiveEngine first, but some document classes
    have their own governed synthesis inside ``desktop_task`` (for example
    Aura self-summary and live research synthesis). If the foreground speech
    draft fails quality after that attempt, the action lane should still run
    and return receipt evidence instead of serving an empty 503.
    """
    if _desktop_objective_self_sufficient_without_cognitive_text(user_message):
        return True
    if _chat_desktop_objective._blocks_consequential_desktop_execution(user_message):
        return False
    if not _chat_preflight._looks_like_desktop_objective(user_message):
        return False
    text = str(user_message or "").strip()
    try:
        from core.skills.desktop_task import DesktopTaskSkill

        return bool(
            DesktopTaskSkill._objective_requests_self_summary(text)
            or DesktopTaskSkill._objective_requests_research_document(text)
        )
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
        return False
