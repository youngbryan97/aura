from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any

from core.kernel.bridge import Phase
from core.phases.dialogue_policy import validate_dialogue_response
from core.phases.response_contract import ResponseContract
from core.runtime.background_policy import background_activity_allowed
from core.runtime.errors import FallbackClassification, Severity, record_degradation
from core.runtime.runtime_settings import get_runtime_setting
from core.runtime.service_access import resolve_orchestrator
from core.state.aura_state import AuraState

if TYPE_CHECKING:
    from core.kernel.aura_kernel import AuraKernel

logger = logging.getLogger("Aura.LearningPhase")

_LEARNING_RECOVERABLE_ERRORS = (ImportError, AttributeError, RuntimeError, TypeError, ValueError)


def _ensure_response_modifiers(state: AuraState) -> dict[str, Any]:
    modifiers = getattr(state, "response_modifiers", None)
    if modifiers is None:
        state.response_modifiers = {}
        modifiers = state.response_modifiers
    return modifiers


def _mark_learning_degraded(
    state: AuraState,
    *,
    stage: str,
    error: BaseException,
    action: str,
) -> None:
    modifiers = _ensure_response_modifiers(state)
    marker = modifiers.setdefault(
        "learning_phase",
        {
            "status": "degraded",
            "failures": [],
        },
    )
    marker["status"] = "degraded"
    failures = marker.setdefault("failures", [])
    failures.append(
        {
            "stage": stage,
            "error_type": type(error).__name__,
            "action": action,
            "at": time.time(),
        }
    )
    del failures[:-8]
    modifiers["learning_phase_degraded"] = True

    health = getattr(state, "health", None)
    if isinstance(health, dict):
        health["learning_phase"] = {
            "status": "degraded",
            "stage": stage,
            "last_error_type": type(error).__name__,
            "action": action,
            "at": time.time(),
        }


def _record_learning_degradation(
    error: BaseException,
    *,
    stage: str,
    action: str,
    state: AuraState | None = None,
    severity: Severity = "warning",
) -> None:
    record_degradation(
        "learning_phase",
        error,
        severity=severity,
        action=action,
        classification=FallbackClassification.SAFE_FALLBACK,
        extra={
            "stage": stage,
            "repair_requested": True,
        },
    )
    if state is not None:
        _mark_learning_degraded(state, stage=stage, error=error, action=action)


def _recall_her_talk() -> int:
    """Give her record of what she has met the conversations she already had.

    A subject stands out against what she has talked about before, and on the
    day that record is first kept it has nothing behind it. Her stored turns
    are that history. Read once, off the loop, when the record has none.
    """
    import sqlite3

    from core.conation.what_she_wants import get_what_she_wants
    from core.conversation.persistence import get_persistence

    try:
        stored = get_persistence().get_recent_history(limit=4096)
    except (sqlite3.Error, OSError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
        _record_learning_degradation(
            exc,
            stage="recall_her_talk",
            action="left her record without her earlier conversations; new subjects stand out only against later ones",
        )
        return 0
    exchanges: list[tuple[str, str]] = []
    heard = ""
    for turn in stored:
        role = str(turn.get("role", "")).lower()
        content = str(turn.get("content", "") or "")
        if role == "user":
            heard = content
        elif heard and content:
            exchanges.append((heard, content))
            heard = ""
    return get_what_she_wants().talk.remember_talk(exchanges)


def _said_this_turn(state: AuraState) -> list[str]:
    """What the last person to speak said and what she answered, once each.

    A turn is read every tick, and the same exchange is still the newest one
    on the ticks after it. It is handed over on the first of them only, so an
    exchange is counted once in what her later ones are set against.
    """
    cognition = getattr(state, "cognition", None)
    heard = answered = ""
    # The newest thing somebody said, and what she said after it. Her answer
    # is in working memory before it is committed as her last response.
    for entry in list(getattr(cognition, "working_memory", []) or []):
        if not isinstance(entry, dict):
            continue
        content = str(entry.get("content", "") or "")
        if str(entry.get("role", "")).lower() == "user":
            heard, answered = content, ""
        elif heard and content:
            answered = content
    answered = answered or str(getattr(cognition, "last_response", "") or "")
    exchange = (heard, answered)
    if not heard or exchange == getattr(_said_this_turn, "last", None):
        return []
    _said_this_turn.last = exchange
    return [heard, answered]


def _with_the_person(about: list[str], person: str, sent_from: str) -> list[str]:
    """A turn somebody sent has that person in it, named or not.

    The loop that serves chat does not pass through the door that names her
    partner, so a turn a person sent read as time she spent alone and nobody
    could come to matter to her. Somebody who was not identified is whoever
    writes to her from that place.
    """
    if not sent_from:
        return about
    who = f"person:{person.strip() or 'someone at ' + sent_from}"
    kept = [key for key in about if key != "self" and not key.startswith("person:")]
    return [who, *kept]


#: Her last read of each person's distress, to set the next against.
_THEIR_DISTRESS: dict[str, float] = {}


def _how_they_are_reaches_her(who: str) -> None:
    """Her read of the person's distress against her last read of it is how they are faring.

    Falling distress is them doing better and rising is worse; the read is a
    posterior from the channels she has of them, never a word list
    (core/interiority/other_minds.py). What she feels of it is her stake in
    them, and with no stake it is news (core/conation/what_she_cares_about.py).
    """
    from core.conation.what_she_wants import get_what_she_wants
    from core.container import ServiceContainer

    other = getattr(ServiceContainer.get("interiority", default=None), "last_other", None)
    distress = getattr(other, "distress", None)
    if distress is None or getattr(getattr(distress, "provenance", None), "name", "ABSENT") == "ABSENT":
        return
    now = max(0.0, min(1.0, float(distress.value) * float(distress.confidence)))
    before = _THEIR_DISTRESS.get(who)
    _THEIR_DISTRESS[who] = now
    if before is not None and now != before:
        get_what_she_wants().care.fared(who, before - now)


async def read_what_the_turn_was_worth(state: AuraState, *, person: str = "", sent_from: str = "") -> None:
    """Read the turn that just ended against what she expected, and teach by it.

    Its own function because the rest of this phase waits for a background
    tick and this cannot. The phase is skipped on every tick somebody is
    waiting on, so while the reading lived inside it no turn she spent with a
    person was ever read: live, 10 to 13 October, 22 of her 500 choices were
    appraised. The kernel calls this on those ticks in the phase's place
    (core/kernel/aura_kernel_lifecycle.py).
    """
    try:
        from core.affect.what_it_was_worth import (
            broadcast,
            disabled,
            read_turn,
            teach_connections,
        )

        if not disabled():
            reading = read_turn(state)
            direction = broadcast(reading)
            lessons = await asyncio.to_thread(teach_connections, reading)
            # And what she holds in mind settles on what the turn was worth.
            # See memory_retrieval.settle_on_what_it_was_worth.
            from core.phases.memory_retrieval import settle_on_what_it_was_worth

            settle_on_what_it_was_worth(state)
            # And what it leaves her to work on. See core/soma/reserve.py.
            from core.affect.what_it_was_worth import dose
            from core.soma.reserve import get_reserve_ledger

            if reading.measured:
                get_reserve_ledger().charge(dose(reading))
            _ensure_response_modifiers(state)["worth"] = {
                **reading.as_dict(),
                "sent": direction,
                "taught": sorted(lessons),
            }
            # And what it was worth is what the things in it were worth:
            # her likes, dislikes and wants are read off this record.
            # See core/conation/what_she_wants.py.
            from core.conation.what_she_wants import get_what_she_wants
            from core.conation.what_she_wants import read_turn as met_this_turn

            if not get_what_she_wants().talk.talked_before():
                await asyncio.to_thread(_recall_her_talk)
            markers = getattr(getattr(state, "affect", None), "markers", None) or {}
            about = _with_the_person(
                list((markers.get("feelings_about") or {}).get("present") or []), person, sent_from
            )
            if sent_from and about:
                # She is in a conversation with them from here until she is
                # free again; the turns between their messages are part of it.
                who = about[0]
                get_what_she_wants().started(who, "conversation", label=who.split(":", 1)[1])
            said = _said_this_turn(state)
            # Whoever the turn was with, named by the door it came through or
            # by her own reading of who was in it. A turn through the kernel
            # arrives with no place on it.
            who = next((key for key in about if key.startswith("person:")), "")
            if who and len(said) > 1:
                _how_they_are_reaches_her(who)
                # Her turn in whatever she and they are doing together.
                from core.together.taking_it_up import get_taking_it_up

                get_taking_it_up().she_said(said[1], who=who)
            met_this_turn(about, reading, said=said)
    except _LEARNING_RECOVERABLE_ERRORS as e:
        _record_learning_degradation(
            e,
            stage="worth",
            action="left this turn's payoff unread; her chemistry and connections were not taught by it",
            state=state,
        )


class LearningPhase(Phase):
    """
    Post-response learning phase.

    Detects follow-up and confusion signals from working memory,
    then records the interaction into LiveLearner for quality scoring.
    """

    def __init__(self, kernel: AuraKernel):
        super().__init__(kernel)
        self._learner = None  # Lazy-loaded

    def _get_learner(self):
        if self._learner is None:
            try:
                from core.learning.live_learner import get_live_learner

                self._learner = get_live_learner()
            except (ImportError, AttributeError, RuntimeError) as e:
                _record_learning_degradation(
                    e,
                    stage="learner_load",
                    action="disabled live learner for this tick; lazy load will retry on next learning phase",
                )
                logger.debug("LearningPhase: could not load learner: %s", e)
        return self._learner

    async def execute(self, state: AuraState, objective: str | None = None, **kwargs) -> AuraState:
        # What the turn was worth to her against what she expected, read before
        # anything below can return early: a turn with no reply still paid or
        # cost her something. See core/affect/what_it_was_worth.py.
        await read_what_the_turn_was_worth(state)

        # ISSUE-88: Self-Modification Awareness
        has_mod = False
        if hasattr(state.cognition, "modifiers") and state.cognition.modifiers:
            if "self_modification" in state.cognition.modifiers:
                has_mod = True
                logger.info(
                    "🧠 SelfModification: Awareness triggered. Preserving modification context."
                )

        # Memory Consolidation Logic
        # Skip if response is too short and no modification occurred
        if not has_mod and len(state.cognition.last_response or "") < 10:
            return state

        # Metacognitive feedback: wire user corrections to the calibrator.
        # When the user signals confusion/correction, feed that as actual_correctness=0
        # for the previous response. When they follow up (engaged), score it as correct.
        follow_up = self._detect_follow_up(state)
        confusion = self._detect_confusion(state)
        try:
            from core.container import ServiceContainer

            calibrator = ServiceContainer.get("metacognitive_calibrator", default=None)
            if calibrator:
                prev_confidence = state.response_modifiers.get("learning_score", 0.7)
                if confusion:
                    calibrator.record_prediction(confidence=prev_confidence, actual_correctness=0.0)
                    logger.debug("MetaCal: correction signal → recording low correctness")
                elif follow_up:
                    calibrator.record_prediction(confidence=prev_confidence, actual_correctness=1.0)
        except (ImportError, AttributeError, RuntimeError) as e:
            _record_learning_degradation(
                e,
                stage="metacognitive_calibration",
                action="skipped calibration write; response learning continues without confidence update",
                state=state,
            )
            logger.debug("LearningPhase: metacognitive feedback failed: %s", e)

        # Run standard learning + cross-domain synthesis
        try:
            state = await self._perform_standard_learning(state, objective or "")
        except (RuntimeError, AttributeError, TypeError, ValueError) as e:
            _record_learning_degradation(
                e,
                stage="standard_learning",
                action="preserved response state and continued to cross-domain/follow-up learning",
                state=state,
            )
            logger.debug("LearningPhase: standard learning failed: %s", e)

        try:
            state = await self._map_cross_domain(state, objective or "")
        except (RuntimeError, AttributeError, TypeError, ValueError) as e:
            _record_learning_degradation(
                e,
                stage="cross_domain_mapping",
                action="skipped cross-domain synthesis for this tick; foreground response state preserved",
                state=state,
            )
            logger.debug("LearningPhase: cross-domain mapping failed: %s", e)

        try:
            await self._wire_conversation_learning(state, objective or "")
        except (RuntimeError, AttributeError, TypeError, ValueError) as e:
            _record_learning_degradation(
                e,
                stage="conversation_learning",
                action="skipped conversation enrichment/distillation for this tick after response delivery",
                state=state,
            )
            logger.debug("LearningPhase: follow-up wiring failed: %s", e)

        return state

    async def _perform_standard_learning(self, state: AuraState, objective: str) -> AuraState:
        learner = self._get_learner()
        if learner is None:
            return state
        response = state.cognition.last_response
        if not response:
            return state

        follow_up = self._detect_follow_up(state)
        confusion = self._detect_confusion(state)

        try:
            # ISSUE-94: Affective Memory synchronization
            affect_data = {
                "valence": state.affect.valence,
                "arousal": state.affect.arousal,
                "dominant": state.affect.dominant_emotion,
            }
            score = learner.record_tick(
                state=state,
                user_input=objective,
                response=response,
                follow_up=follow_up,
                confusion=confusion,
                affect=affect_data,  # Sync emotional context
                origin=str(getattr(state.cognition, "current_origin", "") or ""),
            )
            if score:
                if not hasattr(state, "response_modifiers") or state.response_modifiers is None:
                    state.response_modifiers = {}
                state.response_modifiers["learning_score"] = round(score.raw_score, 3)
                state.response_modifiers["affective_sync"] = True
        except (RuntimeError, AttributeError, TypeError) as e:
            _record_learning_degradation(
                e,
                stage="learner_record_tick",
                action="kept response state and left learning_score unchanged for this tick",
                state=state,
            )
            logger.debug("LearningPhase: record failed: %s", e)
        return state

    async def _map_cross_domain(self, state: AuraState, objective: str) -> AuraState:
        """[AGI Seed] Relates current objective to distant knowledge domains."""
        curiosity = getattr(state.affect, "curiosity", 0.5)
        if curiosity > 0.8:
            try:

                orch = resolve_orchestrator()
            except (ImportError, AttributeError, RuntimeError) as exc:
                _record_learning_degradation(
                    exc,
                    stage="cross_domain_orchestrator_lookup",
                    action="evaluated curiosity without orchestrator context; background synthesis remains disabled",
                    state=state,
                )
                orch = None
            try:
                allowed = background_activity_allowed(
                    orch,
                    min_idle_seconds=1200.0,
                    max_memory_percent=78.0,
                    max_failure_pressure=0.08,
                    require_conversation_ready=False,
                )
            except _LEARNING_RECOVERABLE_ERRORS as exc:
                _record_learning_degradation(
                    exc,
                    stage="cross_domain_background_gate",
                    action="blocked cross-domain synthesis because background safety gate failed closed",
                    state=state,
                )
                return state
            if not allowed:
                return state
            logger.info("🔭 [AGI] High curiosity: Generating cross-domain synthesis...")
            state.cognition.pending_intents.append(
                {
                    "type": "cross_domain_synthesis",
                    "trigger": objective,
                    "curiosity_level": curiosity,
                }
            )
        return state

    async def _trigger_autotelic_curiosity(self, state: AuraState) -> AuraState:
        """[AGI Seed] Generates a new internal objective based on idle curiosity."""
        logger.info("🧩 [AGI] Idle detected. Triggering autotelic exploration.")
        state.cognition.pending_intents.append(
            {
                "type": "autotelic_objective",
                "domain": "UNKNOWN",  # To be filled by ResearchCycle
                "curiosity_vector": state.affect.curiosity,
            }
        )
        return state

    @staticmethod
    def _looks_generic(response: str) -> bool:
        text = str(response or "").strip().lower()
        if not text:
            return True
        generic_markers = (
            "how can i help",
            "what would you like",
            "let me know if you'd like",
            "as an ai",
            "as a language model",
            "i'm here to help",
            "what about you?",
            "how about you?",
            "what questions do you have?",
        )
        return any(marker in text for marker in generic_markers)

    @staticmethod
    def _recent_learning_messages(state: AuraState, limit: int = 8):
        messages = []
        for msg in list(getattr(state.cognition, "working_memory", []) or [])[-limit:]:
            if not isinstance(msg, dict):
                continue
            content = str(msg.get("content", "") or "").strip()
            metadata = msg.get("metadata", {}) or {}
            if not content or msg.get("ephemeral"):
                continue
            if msg.get("role") == "system" and metadata.get("type") != "skill_result":
                continue
            messages.append(dict(msg))
        return messages

    async def _wire_conversation_learning(self, state: AuraState, objective: str) -> None:
        response = str(getattr(state.cognition, "last_response", "") or "").strip()
        if not objective or not response:
            return

        from core.container import ServiceContainer

        contract = (
            dict(getattr(state, "response_modifiers", {}) or {}).get("response_contract", {}) or {}
        )
        dialogue_validation = (
            dict(getattr(state, "response_modifiers", {}) or {}).get("dialogue_validation", {})
            or {}
        )
        learning_score = float((state.response_modifiers or {}).get("learning_score", 0.0) or 0.0)
        confusion = self._detect_confusion(state)
        recent_messages = self._recent_learning_messages(state)
        affect_signature = (
            state.affect.get_cognitive_signature()
            if hasattr(state.affect, "get_cognitive_signature")
            else {}
        )

        auto_enrichment_enabled = bool(
            get_runtime_setting("learning.auto_enrichment_enabled", True)
        )
        if recent_messages and auto_enrichment_enabled:
            try:
                from core.cognition.knowledge_enrichment import get_enricher
                from core.utils.task_tracker import get_task_tracker

                enricher = get_enricher(
                    knowledge_graph=ServiceContainer.get("knowledge_graph", default=None),
                    brain=ServiceContainer.get("cognitive_engine", default=None),
                )
                force = bool(
                    contract.get("requires_search") or contract.get("tool_evidence_available")
                )
                task = get_task_tracker().create_task(
                    enricher.enrich_from_conversation(recent_messages, force=force),
                    name="learning_phase.knowledge_enrichment",
                )
                get_task_tracker().track_task(task)
            except (ImportError, AttributeError, RuntimeError) as e:
                _record_learning_degradation(
                    e,
                    stage="knowledge_enrichment_schedule",
                    action="skipped async knowledge enrichment scheduling; foreground learning continues",
                    state=state,
                )
                logger.debug("LearningPhase: knowledge enrichment scheduling failed: %s", e)
        elif recent_messages:
            _ensure_response_modifiers(state)["knowledge_enrichment"] = {
                "status": "disabled_by_runtime_setting",
                "setting": "learning.auto_enrichment_enabled",
            }

        should_distill = bool(
            confusion
            or (learning_score and learning_score < 0.55)
            or self._looks_generic(response)
            or (contract.get("requires_search") and not contract.get("tool_evidence_available"))
        )
        if not should_distill and contract:
            try:
                dialogue_contract = ResponseContract(**contract)
                should_distill = not validate_dialogue_response(response, dialogue_contract).ok
            except (RuntimeError, AttributeError, TypeError, ValueError) as _exc:
                _record_learning_degradation(
                    _exc,
                    stage="dialogue_contract_validation",
                    action="fell back to heuristic distillation decision after contract validation failed",
                    state=state,
                )
                logger.debug("Suppressed Exception: %s", _exc)
        if should_distill:
            try:
                from core.adaptation.distillation_pipe import get_distillation_pipe

                confidence = max(0.05, min(0.99, learning_score or 0.45))
                await get_distillation_pipe().flag_for_distillation(
                    prompt=objective,
                    local_response=response,
                    confidence=confidence,
                    context={
                        "origin": str(getattr(state.cognition, "current_origin", "") or ""),
                        "confusion": confusion,
                        "learning_score": learning_score,
                        "response_contract": contract,
                        "dialogue_validation": dialogue_validation,
                        "affect_signature": affect_signature,
                    },
                )
            except (ImportError, AttributeError, RuntimeError) as e:
                _record_learning_degradation(
                    e,
                    stage="distillation_flagging",
                    action="left failed sample unqueued for this tick; future failures remain visible in state",
                    state=state,
                )
                logger.debug("LearningPhase: distillation flagging failed: %s", e)

    def _detect_follow_up(self, state: AuraState) -> bool:
        """
        A follow-up is when the user responded to our last message.
        The working memory should show: ... user, assistant, user
        """
        wm = state.cognition.working_memory
        if len(wm) < 3:
            return False
        last3_roles = [m.get("role") for m in wm[-3:]]
        # Pattern: user → assistant → user means they engaged with our response
        return last3_roles == ["user", "assistant", "user"]

    def _detect_confusion(self, state: AuraState) -> bool:
        """
        Confusion signals from the most recent user message.
        """
        wm = state.cognition.working_memory
        # Get the current user message (the one being processed)
        user_msgs = [m for m in wm if m.get("role") == "user"]
        if not user_msgs:
            return False
        last_user = user_msgs[-1].get("content", "").lower()
        confusion_signals = [
            "what?",
            "??",
            "that's wrong",
            "that's not right",
            "i don't understand",
            "what do you mean",
            "huh?",
            "you're wrong",
            "that makes no sense",
            "incorrect",
            "not what i asked",
            "stop",
            "nevermind",
        ]
        return any(sig in last_user for sig in confusion_signals)
