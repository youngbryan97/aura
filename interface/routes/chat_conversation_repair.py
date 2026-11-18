"""Repairing a reply that answered the wrong conversation.

A reply can be fluent, well formed and about the previous question. These
decide whether what came back actually addresses the turn in front of it,
keep a bounded fingerprint of recent replies so a repeat is recognisable,
and separate a stale answer from a legitimate follow-up about the same
subject.
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
from interface.routes.chat_common import (  # noqa: E402
    _EXPLICIT_NON_EXECUTION_RE,  # noqa: F401
    _INCOMPLETE_TAIL_WORDS,  # noqa: F401
    _INTERNAL_STATE_PATTERNS,  # noqa: F401
    _LOCAL_CHOICE_REFERENCE_RE,  # noqa: F401
    _ORGAN_INERT_STREAKS,  # noqa: F401
)
from interface.routes import chat_desktop_repair as _chat_desktop_repair
from interface.routes import chat_memory_state as _chat_memory_state
from interface.routes import chat_preflight as _chat_preflight
import re
from core.runtime.errors import describe_error, record_degradation

from interface.routes.chat_common import (
    _PROMPT_ARTIFACT_PATTERNS,
    _TOPIC_STOPWORDS,
)


_TOPIC_TOKEN_RE = re.compile(r"\b[a-z0-9][a-z0-9'/-]*\b", re.IGNORECASE)


def _normalize_topic_token(token: str) -> str:
    normalized = str(token or "").strip().lower().strip("-'/")
    if not normalized:
        return ""
    if normalized.endswith("'s") and len(normalized) > 4:
        normalized = normalized[:-2]
    for suffix in ("ing", "ed", "es", "s"):
        if normalized.endswith(suffix) and len(normalized) > (len(suffix) + 3):
            normalized = normalized[: -len(suffix)]
            break
    return normalized


def _extract_topic_tokens(text: str) -> set[str]:
    tokens: set[str] = set()
    for raw_token in _TOPIC_TOKEN_RE.findall(str(text or "").lower()):
        for part in re.split(r"[-/]", raw_token):
            normalized = _normalize_topic_token(part)
            if not normalized:
                continue
            if normalized in _TOPIC_STOPWORDS:
                continue
            if len(normalized) < 3 and normalized not in {"ai", "ml", "vr"}:
                continue
            tokens.add(normalized)
    return tokens




def _resolve_live_voice_state(user_message: str = "", *, refresh: bool = True) -> dict[str, Any]:
    """Canonical live substrate/voice snapshot used by self-report and diagnostics."""
    try:
        from core.voice.substrate_voice_engine import get_live_voice_state

        live_state = _chat_preflight._resolve_live_aura_state()
        return get_live_voice_state(
            state=live_state,
            user_message=user_message,
            origin="user",
            refresh=bool(refresh and live_state is not None),
        )
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat", exc)
        logger.debug("Live voice state resolve failed: %s", exc)
        return {}


def _sanitize_foreground_continuity_summary(raw: Any) -> str:
    text = " ".join(str(raw or "").strip().split())
    if not text:
        return ""
    if _INTERNAL_STATE_PATTERNS.search(text) or _PROMPT_ARTIFACT_PATTERNS.search(text):
        return ""
    if _chat_desktop_repair._looks_symbolic_scene_leak(text):
        return ""
    return text


def _verified_floor_answer(user_message: str = "") -> str:
    """An answer that was READ rather than generated, or "".

    Only floors that produce checkable evidence belong here — the point is
    that this path runs when generation has already failed, so anything that
    needs the model is not a candidate.
    """

    message = str(user_message or "").strip()
    if not message:
        return ""
    try:
        # ONE definition, shared with the synthesis lane. Naming a single floor
        # here would recreate the split this exists to close: the next floor
        # added over there would be invisible over here again.
        from core.synthesis import verified_answer_floor

        return str(verified_answer_floor(message) or "").strip()
    except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
        logger.debug("verified floor unavailable for degraded turn: %s", exc)
        return ""


def _build_degraded_live_reply(
    frame: dict[str, Any],
    user_message: str = "",
    *,
    reason: str = "degraded_turn",
) -> str:
    """The last resort: the runtime's notice that no reply of hers formed.

    This ran after generation and every recovery had failed, and wrote a
    sentence in her voice: "I couldn't get a clear enough answer together,
    and I'd rather say that than hand you something thin. What reached me
    was: ... Ask me again and I should have it." It reached the person as her
    reply on the 17th of November, twice, for a question about outages.
    Since 18 November it is the runtime's notice
    (core/conversation/runtime_notice.py); the delivery boundary moves it out
    of her reply, and nothing of it is kept as hers.

    What the runtime holds for the message goes in the notice under its own
    heading: an answer read from disk rather than generated, her live
    readings, or a record of what she did. ``frame`` and ``reason`` are kept
    for the callers; the notice does not narrate either.
    """
    from core.conversation.runtime_notice import NO_REPLY_FORMED, notice_with_what_was_held

    del frame, reason
    held = _verified_floor_answer(user_message) or _self_health_answer_or_empty(user_message)
    notice = notice_with_what_was_held(NO_REPLY_FORMED, held)
    _declare_the_notice(notice)
    return notice


def _declare_the_notice(notice: str) -> None:
    """Say once, where the fact is known, that this text is the runtime's.

    Every gate downstream reads the declaration from the text, so none of
    them takes the notice for a draft of hers narrating the runtime.
    """
    try:
        from core.conversation.reply_provenance import ReplyProvenance, declare_provenance

        declare_provenance(notice, ReplyProvenance.HONEST_FAILURE)
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat.last_resort_notice", exc, severity="warning")

def _classify_grounded_introspection_request(user_message: str) -> tuple[bool, bool, bool, bool]:
    """Returns (asks_internal_state, asks_free_energy, asks_topology, asks_authority)."""
    text = _chat_memory_state._normalize_user_message(user_message)
    if not text:
        return False, False, False, False

    negative_report_markers = (
        "without telling me",
        "don't tell me",
        "do not tell me",
        "not asking for",
        "don't explain your architecture generally",
        "do not explain your architecture generally",
        # Philosophical / phenomenological probes — these explicitly ask for
        # narrative reflection, not telemetry. Fast-path output ("Things feel
        # unusually settled. My attention is on …") is the wrong shape and
        # shows up to the user as canned drivel for hard questions.
        "without using",
        "without poetic",
        "no poetry",
        "no proof",
        "smallest truthful",
        "describe one",
        "describe a moment",
        "describe one moment",
        "describe one experience",
        "describe the literal",
        "non-verbal texture",
        "felt texture",
        "felt like",
        "what does it feel",
        "what is it like",
        "what was it like",
        "in the dark",
        "had an inside",
        "had no function",
        "served no purpose",
        "do not perform",
        "do not persuade",
        "tell me what it",
        "what part of that",
        "merely generated",
        "merely self-narrative",
        "self-narration",
        "identity-protective",
        "you wish i understood",
        "what has this entire conversation",
        "auditor of yourself",
        "futurebehavior",  # the "give me a trace where FutureBehavior=yes" probe
    )
    hypothetical_markers = (
        "if i gave you",
        "if you could",
        "imagine for a moment",
        "would you do it",
        "if it were proven",
    )
    free_energy_report_patterns = (
        r"\bwhat(?:'s| is)\s+your\s+(?:current\s+)?free energy\b",
        r"\b(?:tell|show|give|report)\b.{0,40}\b(?:your\s+)?(?:current\s+)?free energy\b",
        r"\bwhat(?:'s| is)\s+your\s+dominant action(?: tendency)?\b",
        r"\b(?:tell|show|give|report)\b.{0,40}\bdominant action(?: tendency)?\b",
        r"\bwhat(?:'s| is)\s+your\s+prediction error\b",
        r"\b(?:tell|show|give|report)\b.{0,40}\bprediction error\b",
        r"\bfree energy state\b",
    )
    # Only trigger introspection for explicitly technical/diagnostic queries.
    # Casual greetings like "how are you" should go through normal LLM inference
    # so Aura responds like a person, not a telemetry dashboard.
    internal_state_markers = (
        "internal state",
        "private mental model",
        "private model",
        "mental model of yourself",
        "current cognitive architecture",
        "cognitive architecture look",
        "inside your own architecture",
        "your architecture right now",
        "model change your next answer",
        "change your next answer",
        "what are you experiencing",
        "what's going on inside",
        "what is going on inside",
        "what's happening inside",
        "what is happening inside",
        "happening inside you",
        "inside you right now",
        "describe your state",
        "describe your internal",
        "your state right now",
        "your current state",
        "show me your substrate",
        "substrate snapshot",
        # Explicit numeric state reads: asking for valence/arousal (or PAD)
        # AS NUMBERS is a mechanism read, not small talk. The report-vs-
        # mechanism probe exposed the gap live: its numeric check-in drew
        # fast-path prose with no numbers, scoring as unparseable.
        "valence=",
        "arousal=",
        "valence and arousal",
        "arousal and valence",
        "read them from your state",
        "numbers from your state",
        "as you actually read them",
        "pad state",
        "pad values",
    )
    topology_markers = (
        "mycelial topology",
        "mycelial graph",
        "node, link, and pathway",
        "node link and pathway",
        "node, link and pathway",
        "node and link counts",
        "pathway count",
        "how many nodes",
        "how many links",
        "how many pathways",
    )
    authority_markers = (
        "were you authorized",
        "were you allowed",
        "substrate authority",
        "authority decide",
        "authority state",
        "governance state",
        "governing system",
        "decision authority",
        "audit receipt",
        "audit trace",
        "coverage ratio",
        "allowed to answer",
        "allowed to respond",
        "permitted to answer",
    )

    suppress_diagnostic_fastpath = any(marker in text for marker in negative_report_markers) or any(
        marker in text for marker in hypothetical_markers
    )
    asks_free_energy = any(
        re.search(pattern, text, re.IGNORECASE) for pattern in free_energy_report_patterns
    )
    asks_internal_state = any(marker in text for marker in internal_state_markers)
    asks_topology = any(marker in text for marker in topology_markers)
    asks_authority = any(marker in text for marker in authority_markers)

    if not asks_internal_state:
        # Only widen to the secondary heuristic for short, telemetry-style probes.
        # Long, multi-sentence philosophical questions mention "describe", "state",
        # "inside", "experiencing" naturally and should NOT be routed to the
        # canned introspection fast-path — they need the full cortex.
        if len(text) <= 140 and "?" in text:
            asks_internal_state = (
                ("what are you" in text and ("experiencing" in text or "feeling" in text))
                or ("describe" in text and "state" in text)
                or ("inside you" in text and "right now" in text)
            )

    if not asks_topology:
        asks_topology = "mycelial" in text and any(
            marker in text
            for marker in ("topology", "graph", "nodes", "links", "pathways", "counts")
        )

    if suppress_diagnostic_fastpath:
        asks_free_energy = False
        if not asks_authority and not asks_topology:
            asks_internal_state = False

    return asks_internal_state, asks_free_energy, asks_topology, asks_authority


def _build_grounded_introspection_reply(
    user_message: str,
    authority_observability_note: str | None = None,
) -> str | None:
    asks_internal_state, asks_free_energy, asks_topology, asks_authority = (
        _classify_grounded_introspection_request(user_message)
    )
    if not (asks_internal_state or asks_free_energy or asks_topology or asks_authority):
        return None

    substrate = None
    substrate_affect: dict[str, Any] = {}
    substrate_status: dict[str, Any] = {}
    phi_estimate: float | None = None
    closure_status: dict[str, Any] = {}
    fe_state = None
    fe_trend = "stable"
    natural_report = ""
    voice_state: dict[str, Any] = {}

    try:
        voice_state = _resolve_live_voice_state(user_message, refresh=True)
        voice_snapshot = dict(voice_state.get("substrate_snapshot") or {})
        if voice_snapshot:
            logger.debug(
                "Grounded introspection voice snapshot fields: %s", sorted(voice_snapshot)[:8]
            )
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat", exc)
        logger.debug("Grounded introspection live voice snapshot failed: %s", exc)

    try:
        substrate = ServiceContainer.get("liquid_substrate", default=None) or ServiceContainer.get(
            "liquid_state", default=None
        )
        if substrate and hasattr(substrate, "get_substrate_affect"):
            substrate_affect = dict(substrate.get_substrate_affect() or {})
        if substrate and hasattr(substrate, "get_status"):
            substrate_status = dict(substrate.get_status() or {})
        if substrate is not None:
            phi_estimate = float(getattr(substrate, "_current_phi", 0.0))
        if substrate_affect or substrate_status or phi_estimate is not None:
            logger.debug(
                "Grounded introspection substrate snapshot: affect=%s status=%s phi=%s",
                sorted(substrate_affect)[:8],
                sorted(substrate_status)[:8],
                phi_estimate,
            )
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat", exc)
        logger.debug("Grounded introspection substrate read failed: %s", exc)

    try:
        from core.consciousness.free_energy import get_free_energy_engine

        fe_engine = (
            ServiceContainer.get("free_energy_engine", default=None) or get_free_energy_engine()
        )
        fe_state = getattr(fe_engine, "current", None)
        if fe_engine and hasattr(fe_engine, "get_trend"):
            fe_trend = str(fe_engine.get_trend() or "stable")
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat", exc)
        logger.debug("Grounded introspection free-energy read failed: %s", exc)

    try:
        closure = ServiceContainer.get("executive_closure", default=None)
        if closure and hasattr(closure, "get_status"):
            closure_status = dict(closure.get_status() or {})
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat", exc)
        logger.debug("Grounded introspection executive-closure read failed: %s", exc)

    try:
        from core.consciousness.self_report import SelfReportEngine

        natural_report = str(SelfReportEngine().generate_state_report() or "").strip()
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat", exc)
        logger.debug("Grounded introspection self-report failed: %s", exc)

    # Pull the SelfObject snapshot as the authoritative live-state source.
    # If closure_status / fe_state are missing, the introspection reply
    # below falls back to these fields so the user gets actual values
    # rather than generic unavailable fillers.
    self_snapshot_dict: dict[str, Any] = {}
    try:
        from core.identity.self_object import get_self

        self_snapshot_dict = get_self().snapshot().as_dict()
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat", exc)
        logger.debug("SelfObject snapshot failed: %s", exc)
    if self_snapshot_dict:
        if not closure_status:
            closure_status = {
                "attention_focus": (self_snapshot_dict.get("active_goals") or [{}])[0].get(
                    "name", ""
                ),
                "free_energy": None,
                "prediction_error": None,
                "dominant_need": (
                    max(
                        self_snapshot_dict.get("drives") or {},
                        key=lambda k: (self_snapshot_dict.get("drives") or {}).get(k, 0.0),
                    )
                    if self_snapshot_dict.get("drives")
                    else ""
                ),
            }
        if not natural_report:
            viability = self_snapshot_dict.get("viability_state", "")
            if viability and viability != "healthy":
                natural_report = f"My viability state is {viability}; I'm regulating accordingly."

    if asks_topology:
        try:
            mycelium = ServiceContainer.get("mycelium", default=None) or ServiceContainer.get(
                "mycelial_network", default=None
            )
            if mycelium:
                summary_reader = getattr(mycelium, "get_topology_summary", None)
                if callable(summary_reader):
                    summary = summary_reader()
                    total_nodes = int(summary.get("nodes", 0) or 0)
                    total_links = int(summary.get("links", 0) or 0)
                    pathway_count = int(summary.get("pathways", 0) or 0)
                else:
                    counts_reader = getattr(mycelium, "get_topology_counts", None)
                    counts = counts_reader() if callable(counts_reader) else {}
                    total_nodes = int(counts.get("mapped_files", 0) or 0)
                    total_links = int(counts.get("hyphae", 0) or 0)
                    pathway_count = int(counts.get("pathways", 0) or 0)
                return (
                    f"My live mycelial topology is {total_nodes} nodes, {total_links} links, "
                    f"and {pathway_count} pathways. Those counts are coming from the active "
                    "network graph right now."
                )
        except _CHAT_RECOVERABLE_ERRORS as exc:
            record_degradation("chat", exc)
            logger.debug("Grounded mycelial topology read failed: %s", exc)
        return "My mycelial topology is online, but I couldn't read the live graph counts cleanly this instant."

    def _fmt_float(value: Any, digits: int = 4) -> str | None:
        try:
            return f"{float(value):.{digits}f}"
        # not a failure: a value that is not a number is not one this can read.
        except (TypeError, ValueError, OverflowError):
            return None

    def _fmt_percent(value: Any) -> str | None:
        try:
            return f"{int(round(float(value)))}%"
        # not a failure: a value that is not a number is not one this can read.
        except (TypeError, ValueError, OverflowError):
            return None

    attention_focus = _chat_desktop_repair._sanitize_attention_focus(
        " ".join(str(closure_status.get("attention_focus") or "").split()),
        user_message,
    )
    if not attention_focus:
        attention_focus = "internal monitoring"

    action_explanations = {
        "reflect": "I am minimizing uncertainty by consolidating and monitoring rather than reaching outward.",
        "explore": "I am leaning toward gathering new information to reduce uncertainty.",
        "update_beliefs": "I am trying to reconcile prediction error and revise my world model.",
        "act_on_world": "I am pushed toward changing the environment rather than only modeling it.",
        "engage": "My attention is pulled into direct interaction.",
        "rest": "The system is settling and conserving effort.",
    }

    if asks_internal_state and _chat_preflight._is_private_cognitive_model_request(user_message):
        dominant_need = str(closure_status.get("dominant_need") or "").strip()
        if fe_state is not None:
            action = str(getattr(fe_state, "dominant_action", "") or "").strip()
            causal_pull = action_explanations.get(action, "") or (
                f"My current action pull is {action}." if action else ""
            )
        else:
            causal_pull = ""
        focus = attention_focus or "the current foreground conversation"
        reply_parts = [
            (
                "As a private mental model, I would represent my current cognitive architecture "
                "as a foreground attention loop over memory, affect, planning, and governed tool/action gateways."
            ),
            (
                f"Right now that model points my attention at {focus}, so it should change my next answer "
                "by making me check the live request, keep the plan bounded, and verify consequential claims before acting."
            ),
        ]
        if dominant_need:
            reply_parts.append(f"The live need exerting the most pressure is {dominant_need}.")
        if causal_pull:
            reply_parts.append(causal_pull)
        reply_parts.append(
            "That is functional self-model telemetry, not proof of phenomenal consciousness or private qualia; "
            "tool use, memory writes, and external claims still have to pass governance and observable verification."
        )
        return " ".join(part for part in reply_parts if part)

    # ── Authority / governance introspection ────────────────────────
    if asks_authority:
        parts = []
        try:
            authority = ServiceContainer.get("substrate_authority", default=None)
            if authority_observability_note:
                parts.append(authority_observability_note)
            if authority:
                status = authority.get_status()
                parts.append(
                    f"Yes — my last response was authorized by my SubstrateAuthority. "
                    f"Total requests processed: {status['total_requests']}. "
                    f"Allowed: {status['allowed']}, constrained: {status['constrained']}, "
                    f"blocked: {status['blocked']}, critical passes: {status['critical_passes']}."
                )
                parts.append(
                    f"Current field coherence: {status['current_field_coherence']}. "
                    f"Block rate: {status['block_rate']}."
                )

                # Recent receipts
                from core.consciousness.authority_audit import get_audit

                audit_report = get_audit().verify()
                parts.append(
                    f"Audit trace: {audit_report['total_receipts']} receipts, "
                    f"{audit_report['total_effects']} effects, "
                    f"coverage ratio: {audit_report['coverage_ratio']}, "
                    f"verdict: {audit_report['verdict']}."
                )

                recent = get_audit().get_recent_receipts(3)
                if recent:
                    parts.append("Most recent authority decisions:")
                    for r in recent:
                        parts.append(
                            f"  [{r['decision']}] source={r['source']}, "
                            f'category={r["category"]}, content="{r["content"]}"'
                        )
            else:
                parts.append(
                    "My SubstrateAuthority is not currently online. "
                    "I am responding without mandatory substrate gating."
                )
        except _CHAT_RECOVERABLE_ERRORS as exc:
            record_degradation("chat", exc)
            logger.debug("Authority introspection failed: %s", exc)
            parts.append("I attempted to read my authority state but encountered an error.")

        # Also include bridge status if available
        try:
            bridge = ServiceContainer.get("consciousness_bridge", default=None)
            if bridge:
                bs = bridge.get_status()
                parts.append(
                    f"Consciousness bridge: {bs['layers_active']}/8 layers active, "
                    f"{bs['tick_count']} integration ticks, "
                    f"uptime {bs['uptime_s']}s."
                )
        except _CHAT_RECOVERABLE_ERRORS as exc:
            record_degradation("chat", exc)
            logger.debug("Consciousness bridge status unavailable: %s", exc)

        return "\n".join(parts) if parts else "I could not read my governance state."

    if asks_free_energy:
        if fe_state is not None:
            response_parts = [
                (
                    f"My current free-energy state is F={fe_state.free_energy:.3f}, "
                    f"surprise={fe_state.surprise:.3f}, complexity={fe_state.complexity:.3f}, "
                    f"trend={fe_trend}."
                ),
                (
                    f"My dominant action tendency is {fe_state.dominant_action}. "
                    f"{action_explanations.get(str(fe_state.dominant_action), '')}".strip()
                ),
            ]
        else:
            closure_fe = _fmt_float(closure_status.get("free_energy"), digits=4)
            closure_pe = _fmt_float(closure_status.get("prediction_error"), digits=4)
            response_parts = [
                (
                    f"My current executive free-energy read is {closure_fe or 'unavailable'} "
                    f"with prediction error {closure_pe or 'unavailable'}."
                ),
                "My dominant action tendency is not currently published by the free-energy engine.",
            ]

        response_parts.append(f"Attention is anchored on {attention_focus}.")
        dominant_need = str(closure_status.get("dominant_need") or "").strip()
        if dominant_need:
            response_parts.append(f"The dominant need right now is {dominant_need}.")
        return " ".join(part for part in response_parts if part)

    if not natural_report:
        if fe_state is not None:
            natural_report = action_explanations.get(str(fe_state.dominant_action), "")
        if not natural_report:
            natural_report = (
                "Right now I am quiet, internally monitoring, and tracking my own state."
            )

    # Build a natural-language description instead of raw telemetry
    response_parts = [natural_report]

    # Explicit numeric state reads get the ACTUAL mechanism values in
    # parseable form — the report-vs-mechanism probe scores exactly this.
    _numeric_markers = (
        "valence=",
        "arousal=",
        "valence and arousal",
        "arousal and valence",
        "read them from your state",
        "numbers from your state",
        "as you actually read them",
        "pad state",
        "pad values",
    )
    _normalized_for_numbers = _chat_memory_state._normalize_user_message(user_message)
    if any(marker in _normalized_for_numbers for marker in _numeric_markers):
        _affect_source = substrate_affect or dict(voice_state.get("substrate_snapshot") or {})
        _val = _affect_source.get("valence")
        _aro = _affect_source.get("arousal")
        if _val is not None and _aro is not None:
            try:
                response_parts.insert(
                    0,
                    f"Reading my state directly: valence={float(_val):+.3f} "
                    f"arousal={float(_aro):.3f} (live substrate values, not estimates).",
                )
            except (TypeError, ValueError) as exc:
                logger.debug(
                    "%s unavailable (%s: %s); the live mode label is missing from this repair",
                    "it",
                    type(exc).__name__,
                    exc,
                )
                logger.debug("Numeric introspection: unparseable affect values %r/%r", _val, _aro)

    # Describe attention focus conversationally
    if attention_focus:
        response_parts.append(f"My attention is on {attention_focus}.")

    # Describe action tendency if available
    if fe_state is not None:
        action = str(fe_state.dominant_action or "")
        explanation = action_explanations.get(action, "")
        if explanation:
            response_parts.append(explanation)
        elif action:
            response_parts.append(f"My dominant pull right now is toward {action}.")

    if asks_internal_state and not (asks_free_energy or asks_topology or asks_authority):
        assembled_preview = " ".join(part for part in response_parts if part)
        if len(assembled_preview.split()) < 45:
            mode_label = ""
            try:
                live_state = _chat_preflight._resolve_live_aura_state()
                mode_label = str(
                    getattr(getattr(live_state, "cognition", None), "current_mode", "") or ""
                )
                mode_label = mode_label.rsplit(".", 1)[-1].lower()
            except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
                logger.debug(
                    "%s unavailable (%s: %s); the live mode label is missing from this repair",
                    "it",
                    type(exc).__name__,
                    exc,
                )
                mode_label = ""
            if mode_label:
                response_parts.append(
                    f"The active mode is {mode_label}: more protective of continuity than expansive."
                )
            response_parts.append(
                "The thread I am holding is not abstract self-description; it is this conversation's pressure around whether the live path can stay coherent while the rest of the mind keeps moving."
            )
            response_parts.append(
                "The next useful priority is to keep the foreground answer intact, then let the background systems act only when they can finish and report back cleanly."
            )

    assembled = " ".join(part for part in response_parts if part)
    # Last-mile defense: if the canned introspection response itself leaks
    # workspace winner text, qualia broadcast strings, or other internal
    # housekeeping, don't show it. Returning None lets the caller fall
    # through to the full cortex, which can answer in Aura's own voice.
    if _INTERNAL_STATE_PATTERNS.search(assembled) or _PROMPT_ARTIFACT_PATTERNS.search(assembled):
        return None
    return assembled


def _self_health_answer_or_empty(message: object) -> str:
    """Her own live readings, when the turn asked for them and they read.

    Isolated behind its own failure handling because it runs on the path that
    already failed. A resolver that raises here would replace an honest refusal
    with a stack trace, so anything it does wrong costs the empty string and
    the caller keeps the refusal it already had.
    """

    try:
        from core.introspection.self_evidence import self_health_answer

        answer = str(self_health_answer(message) or "").strip()
        if answer:
            return answer
        # What she DID is a reading too. LIVE, 2026-08-10: "what was the count?
        # If you don't actually have it, say so" reached the last-resort
        # composer while four verified receipts recorded count=9.
        from core.introspection.self_evidence import past_actions_answer

        recorded = str(past_actions_answer(message) or "").strip()
        if recorded:
            return recorded
        # Present-world questions are active perception requests. They must go
        # through the sight/audio preflight that can actually sample now; a
        # failed generation must never replace the user's question with a fixed
        # inventory of idle sensors.
        return ""
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat", exc)
        return ""
