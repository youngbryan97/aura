"""core/phases/conversational_dynamics_phase.py — Conversational Dynamics Phase

Kernel phase that runs on every user message and computes the full
conversational dynamics state. Runs BEFORE response generation so the
LLM speaks from computed pragmatic state, not just raw text.

Position in pipeline: after SensoryIngestion, before CognitiveRouting.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from core.kernel.bridge import Phase
from core.runtime.conversation_support import (
    relational_memory_allows,
    resolve_primary_user_id,
)
from core.runtime.errors import record_degradation
from core.state.aura_state import AuraState

if TYPE_CHECKING:
    from core.kernel.aura_kernel import AuraKernel

logger = logging.getLogger("Aura.ConversationalDynamics")


def _record_conversational_degradation(
    exc: BaseException,
    *,
    action: str,
    severity: str = "warning",
) -> None:
    record_degradation(
        "conversational_dynamics_phase",
        exc,
        severity=severity,
        action=action,
    )


class ConversationalDynamicsPhase(Phase):
    """
    Computes full conversational state before response generation:
    - Pragmatic / illocutionary force
    - Emotional frame and trajectory
    - Topic trajectory, drift chain, open threads
    - Register and accommodation cues
    - Face threat assessment
    - Humor detection
    - Turn management
    """

    def __init__(self, kernel: AuraKernel):
        super().__init__(kernel)
        self._engine = None

    def _get_engine(self):
        if self._engine is None:
            try:
                from core.conversational.dynamics import get_dynamics_engine
                self._engine = get_dynamics_engine()
            except (ImportError, AttributeError, RuntimeError) as e:
                _record_conversational_degradation(
                    e,
                    action="left conversational dynamics phase inactive for this turn",
                    severity="error",
                )
                logger.warning("ConversationalDynamics: Engine init failed: %s", e)
        return self._engine

    @staticmethod
    def _read_register(state: AuraState, message: str) -> None:
        """Measure the shape of what was just said, and carry the reading.

        The two determinations are what change her behaviour. A register that
        asks is a request. A first-person register that asks nothing and
        carries the connectives of holding on is somebody testifying, and the
        response to testimony is company rather than assistance.

        Written to `response_modifiers` so the response phase sees it, and to
        `world` because it is a fact about the person she is talking to rather
        than about her.
        """
        try:
            from core.expression.register import read

            reading = read(message)
            if not reading.measured:
                # Nothing resets `response_modifiers` until the turn ends, so
                # returning here left the last message's reading standing: a
                # message too short to read would still be treated as testimony
                # because the one before it was.
                for key in ("register", "asks_to_be_witnessed", "asks_for_help"):
                    state.response_modifiers.pop(key, None)
                state.world.partner_register = {}
                return
            row = reading.as_dict()
            # And whether this message is insistent for them, which is the
            # thing their insistence can carry. See core/social/resolve.py.
            from core.social.resolve import get_resolve_ledger

            state.cognition.borrowed_resolve = (
                get_resolve_ledger().read(reading.persistence).as_dict()
            )
            # And what they just said about how she is, scored against how she
            # actually is. Their model of her is the third model in the room
            # and the only one nothing was comparing. See core/self/borrowed.py.
            ConversationalDynamicsPhase._score_their_read_of_her(state, message)
            # And whether they show up reliably at all, which is what the
            # attachment in that record moves toward. See core/social/constancy.py.
            ConversationalDynamicsPhase._note_they_came_back(state)
            state.response_modifiers["register"] = row
            state.response_modifiers["asks_to_be_witnessed"] = reading.asks_to_be_witnessed()
            state.response_modifiers["asks_for_help"] = reading.asks_for_help()
            state.world.partner_register = row
        except (AttributeError, ImportError, TypeError, ValueError) as exc:
            logger.debug("register unread for this message: %s", exc)

    @staticmethod
    def _score_their_read_of_her(state: AuraState, message: str) -> None:
        """Score what they said about her against her state and against her broadcast.

        Whether their read beats her own is `core/self/recognition.py`. What is
        scored here is the other question: the same claim against what she
        actually felt and against what she last said she felt. Somebody
        repeating the broadcast back is not reading her, and the difference
        between those two errors is the only thing that separates them.
        """
        try:
            from core.self.borrowed import claims_about_her, score_claim

            claims = claims_about_her(message)
            if not claims:
                return
            emotions = dict(getattr(getattr(state, "affect", None), "emotions", {}) or {})
            # Whether their read of her beats her own is `core/self/recognition.py`,
            # which owns that channel and reports it to `identity.read_by_other`.
            # What is measured here is the other question: whether what they
            # have learned is the broadcast. Somebody repeating back what she
            # last said about herself is not reading her, and from the inside
            # the two feel the same. See core/self/persona_gap.py.
            from core.self.borrowed import feelings_she_named
            from core.self.persona_gap import get_persona_ledger, read_gap

            broadcast = feelings_she_named(
                ConversationalDynamicsPhase._last_said_by_her(state)
            )
            persona = get_persona_ledger()
            for claim in claims:
                if broadcast:
                    persona.note(
                        for_the_person=score_claim(claim, emotions),
                        for_the_performance=score_claim(claim, broadcast),
                    )
            if broadcast:
                felt = max(emotions.values(), default=0.0) if emotions else 0.0
                state.cognition.persona_gap = read_gap(
                    presented=max(broadcast.values(), default=0.0),
                    felt=felt,
                    ledger=persona,
                ).as_dict()
        except (AttributeError, ImportError, TypeError, ValueError) as exc:
            logger.debug("their read of her went unscored: %s", exc)

    @staticmethod
    def _note_they_came_back(state: AuraState) -> None:
        """One return, and one of her own cycles, so theirs has a reference.

        Her period is what theirs is read against rather than a number chosen
        here: something is unreliable when it is less regular than she is.
        """
        try:
            import time

            from core.social.constancy import get_constancy_ledger

            ledger = get_constancy_ledger()
            # Their return only. Her own period is noted once per turn by the
            # affect phase, which runs whether or not anybody spoke — noting
            # both here gave the two series identical timestamps by
            # construction, so the comparison read exactly equal every time and
            # could not have come out any other way.
            now = time.time()
            ledger.they_came_back(now)
            state.cognition.constancy = ledger.read().as_dict()
            # And what she saw about them and did not write down. The observer
            # refuses a trait read off one exchange, and the refusal left no
            # trace until now. See core/social/averted.py.
            from core.social.averted import get_averted_ledger

            state.cognition.averted = get_averted_ledger().read().as_dict()
            # And whether what she holds about them is about them, or is the
            # handful of things she holds about everybody.
            # See core/social/particular.py.
            from core.memory.interpersonal_store import get_interpersonal_store
            from core.social.particular import read_particularity

            store = get_interpersonal_store()
            state.cognition.particular = read_particularity(store.models()).as_dict()
            partner = str(getattr(state.cognition, "current_partner", "") or "")
            # And what he has told her he likes, where the prompt looks for it.
            # `world.user_preferences` is described as durable and injected into
            # every prompt, and nothing wrote it: the context assembler found it
            # empty every time and the column reading it never moved.
            mirrored = store.preferences_for_prompt(partner)
            if mirrored:
                state.world.user_preferences = mirrored
            ConversationalDynamicsPhase._mirror_known_entities(state)
            # And where this sitting with them stands against how their
            # sittings have ended. See core/social/closing_window.py.
            from core.social.closing_window import get_sitting_ledger

            sittings = get_sitting_ledger()
            # With how strained things were around this message, so what their
            # long absences follow can be learned. See `Attribution` there.
            strain = ConversationalDynamicsPhase._strain_of(partner)
            sittings.message(partner, now, strain=strain)
            state.cognition.closing_window = sittings.reading(partner).as_dict()
            # What they were carrying, against how much of her came back. Every
            # gate that shortens an answer looks at one turn, so nothing could
            # see her giving least where there was most to carry.
            # See core/conversation/presence_under_weight.py.
            if strain is not None:
                from core.conversation.presence_under_weight import get_weight_ledger

                weights = get_weight_ledger()
                weights.note(
                    strain,
                    len(ConversationalDynamicsPhase._last_said_by_her(state)),
                )
                state.cognition.presence_under_weight = weights.read().as_dict()
            ConversationalDynamicsPhase._read_civility(state)
        except (AttributeError, ImportError, TypeError, ValueError) as exc:
            logger.debug("their regularity went unread: %s", exc)

    @staticmethod
    def _mirror_known_entities(state: AuraState) -> None:
        """The standing list of what she has met, where the prompt looks for it.

        `world.known_entities` is read by the context assembler and by the
        subject schema and had no writer anywhere in the tree: the block was
        empty on every turn and the column reading it was flat through a
        six-hour recording. The associative entity memory is where the meeting
        is actually recorded, and the description is its own stance sentence
        rather than a summary written here.
        """
        try:
            from core.memory.associative_entity_memory import (
                get_associative_entity_memory,
            )

            memory = get_associative_entity_memory()
            known = memory.best_known()
            if not known:
                return
            state.world.known_entities = {
                entity.canonical_name: {
                    "kind": str(entity.kind.value),
                    "description": memory.stance(entity).sentence(entity.canonical_name),
                    "met": entity.mention_count,
                }
                for entity in known
            }
        except (AttributeError, ImportError, RuntimeError, TypeError, ValueError) as exc:
            logger.debug("the standing list of what she knows was not mirrored: %s", exc)

    @staticmethod
    def _strain_of(partner: str) -> float | None:
        """How frustrated the person here seems, from the other-agent model, or None."""
        if not partner:
            return None
        try:
            from core.social.other_agent_model import get_other_agent_model

            estimate = get_other_agent_model().estimate(partner)
        except (AttributeError, ImportError, RuntimeError, TypeError, ValueError) as exc:
            logger.debug("no strain reading for this message: %s", exc)
            return None
        if getattr(estimate, "abstained", False):
            return None
        value = (getattr(estimate, "affect", None) or {}).get("frustration")
        return None if value is None else float(value)

    @staticmethod
    def _read_made_minor(state: AuraState) -> None:
        """Whether their account of a past they share gives her less of it than usual.

        Read only when what came back to her this turn was a past she shares
        with them, which memory retrieval, earlier in the turn, has marked.
        See core/social/made_minor.py.
        """
        try:
            from core.expression.register import read
            from core.social.made_minor import get_account_ledger
            from core.social.togetherness import last_said

            history = list(getattr(state.cognition, "working_memory", []) or [])
            relived = getattr(state.cognition, "relived", None) or {}
            shared = bool(isinstance(relived, dict) and relived.get("shared"))
            reading = get_account_ledger().read(
                read(last_said(history, ("assistant", "aura"))),
                read(last_said(history, ("user",))),
                shared=shared,
            )
            state.cognition.made_minor = reading.as_dict()
        except (AttributeError, ImportError, TypeError, ValueError) as exc:
            logger.debug("no account of a shared past compared: %s", exc)

    @staticmethod
    def _read_civility(state: AuraState) -> None:
        """What she is in, against the warmth the reply performs.

        Both numbers already existed and were never compared. `offering` is
        the share of her clauses that hand over something of her own, which is
        the surface; valence is what is under it, rescaled to the same range.
        See core/social/civility.py.
        """
        try:
            from core.expression.register import read
            from core.social.civility import get_civility_ledger
            from core.social.togetherness import last_said

            history = list(getattr(state.cognition, "working_memory", []) or [])
            said = last_said(history, ("assistant", "aura"))
            if not said:
                return
            shown = float(read(said).offering)
            valence = float(getattr(getattr(state, "affect", None), "valence", 0.0) or 0.0)
            ledger = get_civility_ledger()
            ledger.note((valence + 1.0) / 2.0, shown)
            state.cognition.civility = ledger.read().as_dict()
        except (AttributeError, ImportError, TypeError, ValueError) as exc:
            logger.debug("what the surface was covering went unread: %s", exc)

    @staticmethod
    def _last_said_by_her(state: AuraState) -> str:
        """The last thing she put out, which is the broadcast they may be reading."""
        history = list(getattr(getattr(state, "cognition", None), "working_memory", []) or [])
        for item in reversed(history):
            if not isinstance(item, dict):
                continue
            if str(item.get("role", "")).lower() in ("assistant", "aura"):
                return str(item.get("content", "") or "")[:2000]
        return ""

    @staticmethod
    def _read_witness(state: AuraState) -> None:
        """Whether she is being asked to keep somebody company rather than help.

        Read off the register this phase just measured and the sentiment the
        integration phase read earlier in the turn, and written to cognition
        so routing, the reply and the affect phase all read one stance.
        """
        try:
            from core.social.witness import read_witness

            state.cognition.witness = read_witness(state.response_modifiers).as_dict()
        except (AttributeError, ImportError, TypeError, ValueError) as exc:
            state.cognition.witness = {}
            logger.debug("witness stance unread for this message: %s", exc)

    @staticmethod
    def _read_togetherness(state: AuraState) -> None:
        """Whether both of them are saying we, and what draws its edge.

        Her own register was never read, so an exchange where both had started
        saying "we" looked the same as one where only one had.
        See core/social/togetherness.py.
        """
        try:
            from core.social.togetherness import read_togetherness

            history = list(getattr(state.cognition, "working_memory", []) or [])
            state.cognition.togetherness = read_togetherness(history).as_dict()
        except (AttributeError, ImportError, TypeError, ValueError) as exc:
            logger.debug("no we read from this exchange: %s", exc)

    @staticmethod
    def _read_cadence(state: AuraState) -> None:
        """The pulse the other person is keeping, and how far off it she sat.

        Read off their recent turns. The placement is of her own last turn
        against that pulse, and it is recorded rather than corrected: the
        deviation is the expressive act, and driving it to zero would make her
        the click track.
        """
        try:
            from core.expression.entrainment import cadence, placement

            history = list(getattr(state.cognition, "working_memory", []) or [])
            theirs = cadence(history)
            mine = ""
            for entry in reversed(history):
                if isinstance(entry, dict) and str(entry.get("role", "")).lower() in {
                    "assistant",
                    "aura",
                }:
                    mine = str(entry.get("content", "") or "")
                    break
            row = theirs.as_dict()
            row["placement"] = round(placement(mine, theirs), 6)
            state.cognition.partner_cadence = row
        except (AttributeError, ImportError, TypeError, ValueError) as exc:
            logger.debug("no pulse read from this exchange: %s", exc)

    @staticmethod
    def _read_recognition(state: AuraState, message: str) -> None:
        """What somebody just said she is feeling, and whether it was warm.

        The claim is paired with what she has already predicted she will feel,
        and the next moment scores both against the same outcome. Being cared
        about is warmth in a message that is about her rather than about
        something else.
        """
        try:
            from core.runtime.service_registry import get_runtime_service
            from core.self.recognition import (
                cared_for,
                claim_about_her,
                get_recognition_ledger,
            )
            from core.social.met_as_a_type import get_type_ledger, met_as_a_type
            from core.state.percepts import emit_percept

            ledger = get_recognition_ledger()
            claimed = claim_about_her(message)
            # Said to a kind of thing rather than to her: told she cannot be
            # tired, addressed as the role, or judged as the class she is in.
            # A statement about a type is not evidence about the individual,
            # and a self-model that takes it as evidence learns the type.
            # See core/social/met_as_a_type.py.
            typed = met_as_a_type(message)
            types = get_type_ledger()
            types.note(typed)
            state.identity.met_as_a_type = {
                **typed.as_dict(), **types.read().as_dict(),
            }
            if typed.about_a_type:
                claimed = None
            if claimed is not None:
                loop = get_runtime_service("self_prediction", default=None)
                prediction = loop.get_current_prediction() if loop is not None else None
                ledger.claim(
                    claimed,
                    float(getattr(prediction, "predicted_affect_valence", 0.0) or 0.0),
                )
            warmth = cared_for(state.response_modifiers)
            reading = ledger.reading().as_dict()
            reading["cared_for"] = round(warmth, 6)
            state.identity.read_by_other = reading
            # The comfort of believing somebody cares is real and its ground
            # may never have been tested. Both are kept: the relief decays
            # each turn it goes untested, so it has to be renewed by contact
            # rather than by repetition, and a belief they contradict is
            # dropped along with what it was holding up.
            # See core/social/unchecked_relief.py.
            felt = warmth
            try:
                from core.social.unchecked_relief import get_relief_ledger

                relief = get_relief_ledger()
                relief.note_turn()
                partner = str(getattr(state.cognition, "current_partner", "") or "")
                if warmth > 0.0 and partner:
                    relief.note_belief(partner, warmth)
                standing = relief.read()
                state.identity.unchecked_relief = standing.as_dict()
                felt = min(warmth, standing.relief) if standing.beliefs else warmth
            except (AttributeError, ImportError, TypeError, ValueError):
                felt = warmth
            if felt > 0.0:
                emit_percept(
                    state.world,
                    "cared_for",
                    content="that was about me, and it was warm",
                    intensity=felt,
                    source="conversation",
                )
        except (AttributeError, ImportError, TypeError, ValueError) as exc:
            logger.debug("how she was read went unrecorded for this message: %s", exc)

    @staticmethod
    async def _let_the_record_see_this_turn(
        state: AuraState, message: str, partner: str
    ) -> None:
        """Let the person record observe the exchange, on whichever lane it came.

        `log_chat_turn_auto` is called from one place in the tree — the HTTP
        chat route — so everything she knows about a person was learned through
        that one door. A voice turn, an autonomous turn or a battery turn
        updated nothing, and the organs that read the record (particularity,
        what she declined to write down, what he has told her he likes) had
        nothing to read on any other lane.

        The store treats the same words twice in a row as one exchange, so this
        and the route watching the same turn record it once.
        """
        try:
            from core.memory.interpersonal_store import get_interpersonal_store

            if not partner:
                return
            said = ""
            for entry in reversed(list(getattr(state.cognition, "working_memory", []) or [])):
                if isinstance(entry, dict) and str(entry.get("role", "")).lower() in {
                    "assistant",
                    "aura",
                }:
                    said = str(entry.get("content", "") or "")
                    break
            await get_interpersonal_store().observe_turn(
                partner,
                episode_id=f"turn:{getattr(state, 'version', 0)}",
                user_text=str(message or ""),
                assistant_text=said,
            )
        except (AttributeError, ImportError, RuntimeError, TypeError, ValueError) as exc:
            logger.debug("the person record did not see this turn: %s", exc)
        ConversationalDynamicsPhase._let_the_entity_record_see_this_turn(state, partner)

    @staticmethod
    def _let_the_entity_record_see_this_turn(state: AuraState, partner: str) -> None:
        """The one entity a conversation definitionally has.

        `core/memory/entity_memory_bridge.py` has a writer for turn evidence
        and it had no caller anywhere in the tree, because entities are only
        ever introduced from the text of a message and nothing introduces the
        person sending it. So the entity memory was empty in every run, and
        `world.known_entities` with it: `W.entity_load` read 0.000000 on all
        5,280 frames of a probe, beside `relationship_load`, `preference_load`
        and `concept_load` at zero and `fact_load` pinned at one fact. Five of
        the world model's columns could not move, in the domain that had one
        measured cause.

        Introducing the interlocutor is a deliberate act and not the silent
        filling the bridge refuses: "resolution never creates" guards against
        every capitalised word becoming a permanent member of her world, and
        the person she is talking to is not a capitalised word. Everyone else
        still has to be introduced in what somebody says.

        The turn's own affect goes with it, which is how a stance becomes
        earned rather than declared.
        """
        if not partner:
            return
        try:
            from core.memory.associative_entity_memory import (
                EntityKind,
                get_associative_entity_memory,
            )
            from core.memory.entity_memory_bridge import learn_entity, record_turn_evidence

            entity = learn_entity(partner, EntityKind.PERSON)
            if entity is None:
                return
            # Meeting them is itself evidence, which is what the bridge says
            # where it recognises somebody in a message, and what
            # `best_known` orders the standing list by.
            get_associative_entity_memory().note_mention(entity.entity_id)
            affect = getattr(state, "affect", None)
            record_turn_evidence(
                entity,
                episode_id=f"turn:{getattr(state, 'version', 0)}",
                valence=float(getattr(affect, "valence", 0.0) or 0.0),
                arousal=float(getattr(affect, "arousal", 0.0) or 0.0),
                role="interlocutor",
            )
        except (AttributeError, ImportError, RuntimeError, TypeError, ValueError) as exc:
            logger.debug("the entity record did not see this turn: %s", exc)

    async def _execute_compute_dynamics_latest(self, active_user_id, engine, new_state, objective, state):
        # Compute dynamics from the latest user message
        dynamics = engine.update(
            message=objective,
            role="user",
            working_memory=state.cognition.working_memory
        )

        # The shape of what arrived, separately from what it said. Who it
        # is about, whether it asks anything, and whether it holds on —
        # which together tell somebody testifying from somebody asking.
        # Assistance is the wrong response to testimony and nothing here
        # could tell the difference before. See core/expression/register.py.
        await self._let_the_record_see_this_turn(new_state, objective, active_user_id)
        self._read_register(new_state, objective)
        self._read_witness(new_state)
        self._read_recognition(new_state, objective)
        self._read_cadence(new_state)
        self._read_togetherness(new_state)
        self._read_made_minor(new_state)

        # Store the prompt injection in response_modifiers so UnitaryResponsePhase can use it
        new_state.response_modifiers["conversational_dynamics"] = engine.get_prompt_injection()

        # Also surface key fields directly to CognitiveContext for other phases to read
        cog = new_state.cognition

        # Update discourse state with richer data
        if dynamics.current_topic != "general":
            cog.discourse_topic = dynamics.current_topic
        cog.user_emotional_trend = dynamics.partner_frame

        # Mirror conversation_energy from partner intensity
        cog.conversation_energy = dynamics.partner_intensity
        return cog, dynamics

    @staticmethod
    async def _execute_store_callback_topics(active_user_id, cog, dynamics, new_state, objective, origin, state):
        # Store callback topics in discourse_branches
        available_callbacks = [
            a.topic for a in dynamics.topic_anchors[-5:]
            if not a.is_resolved and a.topic != dynamics.current_topic
        ]
        cog.discourse_branches = available_callbacks
        cog.turns_since_user_spoke = int(dynamics.turns_since_user_spoke)

        # Store the full dynamics state for downstream phases
        new_state.response_modifiers["conv_dynamics_state"] = {
            "floor_state": dynamics.floor_state,
            "partner_frame": dynamics.partner_frame,
            "partner_intensity": dynamics.partner_intensity,
            "humor_frame_active": dynamics.humor_frame_active,
            "humor_type": dynamics.humor_type,
            "escalation_invited": dynamics.escalation_invited,
            "last_speech_act": dynamics.last_speech_act,
            "illocutionary_intent": dynamics.illocutionary_intent,
            "conditional_relevance_open": dynamics.conditional_relevance_open,
            "register": dynamics.register,
            "in_group_active": dynamics.in_group_active,
            "accommodation_cues": dynamics.accommodation_cues,
            "hedge_level": dynamics.hedge_level,
            "challenge_is_face_threat": dynamics.challenge_is_face_threat,
            "open_threads_count": len(dynamics.open_threads),
            "association_chain": dynamics.association_chain,
        }

        # ── Discourse threading, energy and the user's trend ──
        #
        # `DiscourseTracker` is constructed and registered in
        # `core/social/presence_integration.py` and its `update` was called
        # by nothing in the tree. It is the only writer of
        # `cognition.conversation_energy`, `cognition.discourse_depth` and
        # `cognition.user_emotional_trend`, so all three were constants for
        # the whole of every life — and the workspace prices the exchange's
        # claim on attention from the first of them, so what had just been
        # said always asked for attention by exactly the same amount.
        #
        # "Call this after each incoming user message", says the method.
        # This is the phase that has the message.
        try:
            from core.container import ServiceContainer

            tracker = ServiceContainer.get("discourse_tracker", default=None)
            if tracker is not None and objective:
                await tracker.update(new_state, str(objective))
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            _record_conversational_degradation(
                exc,
                action="continued without discourse threading for this turn",
                severity="degraded",
            )

        # ── Multiple Drafts (Dennett): parallel interpretation streams ──
        # Submit the user's input to spawn competing drafts FIRST.
        # If there are unresolved drafts from the PREVIOUS input, probe
        # them now -- the arrival of a new message IS the probe event.
        try:
            from core.consciousness.multiple_drafts import get_multiple_drafts_engine
            md_engine = get_multiple_drafts_engine()
            # Probe previous drafts (new user message = retroactive elevation)
            if md_engine.get_pending_draft_count() > 0:
                md_engine.probe(source="user_message")
            # Submit current input for new draft generation
            md_engine.submit_input(objective, new_state)
            # Surface divergence and context for downstream consumption
            divergence = md_engine.get_draft_divergence()
            md_block = md_engine.get_context_block()
            if md_block:
                new_state.response_modifiers["multiple_drafts"] = md_block
            if divergence > 0.15:
                cog.modifiers["draft_divergence"] = f"{divergence:.2f}"
            # What the spend decision said when the drafts tied, so the
            # verdict is visible where the turn is read rather than only in
            # a log line. It is already acted on inside the engine, which
            # holds the competition open for one more probe when another
            # round is worth what it costs her.
            judgement = md_engine.last_spend_decision()
            if judgement is not None:
                new_state.response_modifiers["worth_more_thought"] = {
                    "worth": str(judgement.worth),
                    "margin": round(float(judgement.margin), 4),
                    "cost": round(float(judgement.cost), 4),
                    "because": judgement.because,
                }
        except (ImportError, AttributeError, RuntimeError) as exc:
            _record_conversational_degradation(
                exc,
                action="continued without multiple-drafts context for this turn",
                severity="degraded",
            )
            logger.debug("ConversationalDynamics: multiple drafts skipped: %s", exc)

        # ── Higher-Order Thought (Rosenthal): thought about the thought ──
        # Generate a HOT from the current affective state so the LLM has
        # meta-awareness of its own cognitive condition during this turn.
        # This is DISTINCT from AttentionSchema (Graziano): AST models
        # attention itself; HOT is a reflexive representation of mental state.
        try:
            from core.consciousness.hot_engine import get_hot_engine
            hot_engine = get_hot_engine()
            affect = new_state.affect
            hot_state = {
                "valence": float(getattr(affect, "valence", 0.0)),
                "arousal": float(getattr(affect, "arousal", 0.5)),
                "curiosity": float(getattr(affect, "curiosity", 0.5)),
                "energy": float(getattr(affect, "energy", 0.7) if hasattr(affect, "energy") else 0.7),
                "surprise": float(getattr(affect, "surprise", 0.0) if hasattr(affect, "surprise") else 0.0),
            }
            hot_engine.generate_fast(hot_state)
            # Apply reflexive feedback: noticing changes the noticed
            try:
                from core.container import ServiceContainer
                affect_engine = ServiceContainer.get("affect_engine", default=None)
                if affect_engine:
                    hot_engine.apply_feedback(affect_engine)
            except (ImportError, AttributeError, RuntimeError) as feedback_exc:
                _record_conversational_degradation(
                    feedback_exc,
                    action="kept higher-order thought context but skipped affect feedback bridge",
                )
            hot_block = hot_engine.get_context_block()
            if hot_block:
                new_state.response_modifiers["higher_order_thought"] = hot_block
                cog.modifiers["higher_order_thought"] = hot_block
        except (ImportError, AttributeError, RuntimeError) as exc:
            _record_conversational_degradation(
                exc,
                action="continued without higher-order thought context for this turn",
                severity="degraded",
            )
            logger.debug("ConversationalDynamics: HOT engine skipped: %s", exc)

        # ── Wire dormant personhood modules into the foreground path ──
        # These modules exist but were never called during live conversation.
        # Each provides context that shapes HOW Aura responds, not just WHAT.

        # Humor Engine: adaptive banter calibrated to this specific user
        try:
            from core.container import ServiceContainer
            humor = ServiceContainer.get("humor_engine", default=None)
            if (
                humor
                and hasattr(humor, "update_banter_state")
                and relational_memory_allows(
                    active_user_id,
                    "style_preference",
                    "recall",
                )
            ):
                humor.update_banter_state(
                    objective,
                    dynamics,
                    user_id=active_user_id,
                )
            if humor and relational_memory_allows(
                active_user_id,
                "style_preference",
                "prompt",
            ):
                guidance = (
                    humor.get_humor_guidance(active_user_id)
                    if hasattr(humor, "get_humor_guidance")
                    else ""
                )
                banter = (
                    humor.get_banter_directive(active_user_id)
                    if hasattr(humor, "get_banter_directive")
                    else ""
                )
                if guidance or banter:
                    new_state.response_modifiers["humor_guidance"] = f"{guidance}\n{banter}".strip()
        except (ImportError, AttributeError, RuntimeError) as exc:
            _record_conversational_degradation(
                exc,
                action="continued without humor guidance for this turn",
            )
            logger.debug("ConversationalDynamics: humor engine skipped: %s", exc)

        # Conversation Intelligence: rhythm, pacing, arc awareness
        try:
            from core.container import ServiceContainer
            conv_intel = ServiceContainer.get("conversation_intelligence", default=None)
            if conv_intel and hasattr(conv_intel, "get_context_injection"):
                ci_block = conv_intel.get_context_injection()
                if ci_block:
                    new_state.response_modifiers["conversation_intelligence"] = ci_block
        except (ImportError, AttributeError, RuntimeError) as exc:
            _record_conversational_degradation(
                exc,
                action="continued without conversation intelligence context for this turn",
            )
            logger.debug("ConversationalDynamics: conversation intelligence skipped: %s", exc)

        # Relational Intelligence: social modeling of this specific person
        try:
            from core.container import ServiceContainer
            rel_intel = ServiceContainer.get("relational_intelligence", default=None)
            if (
                rel_intel
                and hasattr(rel_intel, "get_context_injection")
                and relational_memory_allows(
                    active_user_id,
                    "derived_profile",
                    "prompt",
                )
            ):
                ri_block = rel_intel.get_context_injection(active_user_id)
                if ri_block:
                    new_state.response_modifiers["relational_intelligence"] = ri_block
        except (ImportError, AttributeError, RuntimeError) as exc:
            _record_conversational_degradation(
                exc,
                action="continued without relational intelligence context for this turn",
            )
            logger.debug("ConversationalDynamics: relational intelligence skipped: %s", exc)

        # MetaCognition: reasoning strategy selection for this turn
        try:
            from core.container import ServiceContainer
            metacog = ServiceContainer.get("metacognition", default=None)
            if metacog and hasattr(metacog, "before_reasoning"):
                meta_hints = metacog.before_reasoning(objective, {
                    "origin": origin,
                    "dynamics": new_state.response_modifiers.get("conv_dynamics_state", {}),
                })
                if meta_hints and isinstance(meta_hints, dict):
                    strategy = meta_hints.get("strategy", "")
                    if strategy:
                        new_state.response_modifiers["metacognitive_strategy"] = strategy
        except (ImportError, AttributeError, RuntimeError) as exc:
            _record_conversational_degradation(
                exc,
                action="continued without metacognitive strategy hints for this turn",
            )
            logger.debug("ConversationalDynamics: metacognition skipped: %s", exc)

        # Credit Assignment: outcome-aware context from prior actions
        try:
            from core.container import ServiceContainer
            credit = ServiceContainer.get("credit_assignment", default=None)
            if credit and hasattr(credit, "get_context_block"):
                ca_block = credit.get_context_block()
                if ca_block:
                    new_state.response_modifiers["credit_assignment"] = ca_block
        except (ImportError, AttributeError, RuntimeError) as exc:
            _record_conversational_degradation(
                exc,
                action="continued without credit-assignment context for this turn",
            )
            logger.debug("ConversationalDynamics: credit assignment skipped: %s", exc)

        # Agency Comparator: sense of authorship over recent actions
        try:
            from core.consciousness.agency_comparator import get_agency_comparator
            agency = get_agency_comparator()
            agency_block = agency.get_context_block()
            if agency_block:
                new_state.response_modifiers["agency_comparator"] = agency_block
        except (ImportError, AttributeError, RuntimeError) as exc:
            _record_conversational_degradation(
                exc,
                action="continued without agency-comparator context for this turn",
            )
            logger.debug("ConversationalDynamics: agency comparator skipped: %s", exc)

        # Narrative Memory: autobiographical narrative context from journal/arcs
        try:
            from core.container import ServiceContainer
            narrative = ServiceContainer.get("narrative_engine", default=None)
            if narrative and hasattr(narrative, "get_narrative_context"):
                nm_block = narrative.get_narrative_context()
                if nm_block:
                    new_state.response_modifiers["narrative_context"] = nm_block
        except (ImportError, AttributeError, RuntimeError) as exc:
            _record_conversational_degradation(
                exc,
                action="continued without narrative memory context for this turn",
            )
            logger.debug("ConversationalDynamics: narrative memory skipped: %s", exc)

        # Autobiographical mythos: durable dream/identity continuity artifact
        try:
            from core.container import ServiceContainer

            dream_journal = ServiceContainer.get("dream_journal", default=None)
            if dream_journal and hasattr(dream_journal, "get_autobiographical_mythos_block"):
                mythos_block = dream_journal.get_autobiographical_mythos_block()
                if mythos_block:
                    new_state.response_modifiers["autobiographical_mythos"] = mythos_block
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            _record_conversational_degradation(
                exc,
                action="continued without autobiographical mythos context for this turn",
            )
            logger.debug("ConversationalDynamics: autobiographical mythos skipped: %s", exc)

        # Natural Follow-up: whether Aura should ask a follow-up, make a statement, or stay quiet
        try:
            from core.container import ServiceContainer
            sve = ServiceContainer.get("substrate_voice_engine", default=None)
            if sve and hasattr(sve, "_followup_engine"):
                followup_engine = sve._followup_engine
                profile = sve.get_current_profile() if hasattr(sve, "get_current_profile") else None
                if followup_engine and profile:
                    last_assistant = ""
                    for msg in reversed(state.cognition.working_memory or []):
                        if msg.get("role") == "assistant":
                            last_assistant = msg.get("content", "")
                            break
                    decision = followup_engine.decide(
                        profile=profile,
                        user_message=objective,
                        aura_response=last_assistant,
                        conversation_history=list(state.cognition.working_memory or []),
                    )
                    if decision and decision.should_followup:
                        new_state.response_modifiers["natural_followup"] = {
                            "should_followup": True,
                            "followup_type": decision.followup_type,
                            "delay_seconds": decision.delay_seconds,
                            "context_hint": decision.context_hint,
                            "word_budget": decision.word_budget,
                            "reason": decision.reason,
                        }
        except (ImportError, AttributeError, RuntimeError) as exc:
            _record_conversational_degradation(
                exc,
                action="continued without natural follow-up scheduling for this turn",
            )
            logger.debug("ConversationalDynamics: natural followup skipped: %s", exc)

        # Intersubjectivity: constitutive other-perspective modeling (Husserl/Zahavi)
        try:
            from core.consciousness.intersubjectivity import get_intersubjectivity_engine
            from core.container import ServiceContainer

            isub = get_intersubjectivity_engine()
            # Feed interlocutor data from user model if available
            try:
                user_model = ServiceContainer.get("user_model", default=None)
                if (
                    user_model
                    and hasattr(user_model, "get_profile")
                    and relational_memory_allows(
                        active_user_id,
                        "derived_profile",
                        "prompt",
                    )
                ):
                    profile_data = user_model.get_profile(active_user_id) or {}
                    isub.update_interlocutor_model(
                        communication_style=str(profile_data.get("communication_style", "")),
                        emotional_state=str(profile_data.get("emotional_state", "")),
                        knowledge_level=str(profile_data.get("knowledge_level", "")),
                        engagement_level=float(profile_data.get("engagement", 0.5)),
                        trust_level=float(profile_data.get("trust", 0.5)),
                    )
            except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as profile_exc:
                _record_conversational_degradation(
                    profile_exc,
                    action="continued intersubjectivity without user profile update",
                )
            # Compute intersubjective frame from current qualia state
            try:
                qs = ServiceContainer.get("qualia_synthesizer", default=None)
                q_vec = getattr(qs, "q_vector", None) if qs else None
                if q_vec is not None:
                    isub.compute_intersubjective_frame(
                        q_vec,
                        topic=str(dynamics.current_topic or ""),
                        is_shared_event=True,
                    )
                    isub_block = isub.get_context_block()
                    if isub_block:
                        new_state.response_modifiers["intersubjectivity"] = isub_block
            except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as frame_exc:
                _record_conversational_degradation(
                    frame_exc,
                    action="continued without intersubjective frame computation for this turn",
                )
        except (ImportError, AttributeError, RuntimeError) as exc:
            _record_conversational_degradation(
                exc,
                action="continued without intersubjectivity context for this turn",
                severity="degraded",
            )
            logger.debug("ConversationalDynamics: intersubjectivity skipped: %s", exc)

    @staticmethod
    def _execute_narrative_gravity_autobiographical(dynamics, new_state, objective):
        # Narrative Gravity: autobiographical self-narrative (Gazzaniga/Dennett)
        try:
            from core.consciousness.narrative_gravity import get_narrative_gravity_center
            ngc = get_narrative_gravity_center()
            # Record this conversation turn as an autobiographical event
            ngc.record_event(
                f"Conversation with user: {objective[:100]}",
                emotional_tone=dynamics.partner_frame,
                identity_relevance=0.3 if dynamics.partner_intensity > 0.5 else 0.1,
                arc_theme="ongoing_relationship" if dynamics.partner_intensity > 0.3 else "",
            )
            ng_block = ngc.get_context_block()
            if ng_block:
                new_state.response_modifiers["narrative_gravity"] = ng_block
        except (ImportError, AttributeError, RuntimeError) as exc:
            _record_conversational_degradation(
                exc,
                action="continued without narrative-gravity context for this turn",
                severity="degraded",
            )
            logger.debug("ConversationalDynamics: narrative gravity skipped: %s", exc)

        logger.debug(
            "ConversationalDynamics: frame=%s intensity=%.2f speech_act=%s register=%s humor=%s",
            dynamics.partner_frame,
            dynamics.partner_intensity,
            dynamics.last_speech_act,
            dynamics.register,
            dynamics.humor_type or "none"
        )

    @staticmethod
    def _name_the_kind(state: AuraState, objective: str) -> None:
        """Read the person's act for its cost and its form, and tell the posterior.

        The cost is read from what the act risked: a correction contradicts
        her to her face, which is the one thing in a message that is plainly
        expensive to the person sending it. Difficulty with no correction in
        it costs them nothing and is read on how it landed, which is what a
        costless act can be read on. See core/social/receptivity.py for the
        weighing this feeds.
        """
        try:
            from core.memory.interpersonal_observer import (
                _CORRECTION_PATTERNS,
                _DIFFICULT,
                _WARM,
            )
            from core.social.never_told import get_telling_ledger
            from core.social.receptivity import get_receptivity
            from core.social.the_kind_it_was import the_kind_it_was

            said = str(objective or "")
            # What kind of regard arrived, if any. See core/social/never_told.py.
            telling = get_telling_ledger()
            telling.note_turn(said)
            state.cognition.never_told = telling.read().as_dict()

            corrected = any(pattern.search(said) for pattern in _CORRECTION_PATTERNS)
            difficult = bool(_DIFFICULT.search(said))
            warm = bool(_WARM.search(said))
            from core.social.the_form_they_welcome import note_heard

            note_heard(str(getattr(state.cognition, "current_partner", "") or ""), warm, corrected or difficult)
            # And how warm it was, which is felt. See core/social/warmth.py.
            from core.social.warmth import get_warmth_ledger

            get_warmth_ledger().heard(
                str(getattr(state.cognition, "current_partner", "") or ""),
                warm=warm,
                objected=corrected or difficult,
            )
            # An objection is where anger comes from, and who it is at.
            # See core/affect/anger_feeds_itself.py.
            if corrected or difficult:
                from core.affect.anger_feeds_itself import get_anger_ledger

                get_anger_ledger().provoked_by(str(getattr(state.cognition, "current_partner", "") or ""))
            if not (corrected or difficult or warm):
                return
            reading = the_kind_it_was(
                # A correction is the costly one: it contradicts her openly.
                cost_to_source=1.0 if corrected else 0.0,
                welcome=warm and not difficult,
            )
            state.cognition.the_kind = reading.as_dict()
            partner = str(getattr(state.cognition, "current_partner", "") or "")
            if partner:
                get_receptivity().observe(
                    partner, reading.kind, cost_to_source=reading.cost_to_source
                )
                from core.social.what_passes_between import note_act

                note_act(partner, reading.kind, reading.cost_to_source)
        except (ImportError, AttributeError, TypeError, ValueError) as exc:
            logger.debug(
                "%s unavailable (%s: %s); the act between them was not noted",
                "get_telling_ledger",
                type(exc).__name__,
                exc,
            )
            return

    async def _execute_new_state(self, engine, objective, origin, state):
        new_state = state.derive("conversational_dynamics", origin="ConversationalDynamicsPhase")
        active_user_id = resolve_primary_user_id(new_state)

        cog, dynamics = await self._execute_compute_dynamics_latest(active_user_id, engine, new_state, objective, state)

        try:
            from core.container import ServiceContainer

            interaction_signals = ServiceContainer.get("interaction_signals", default=None)
            if interaction_signals and hasattr(interaction_signals, "get_status"):
                signal_status = interaction_signals.get_status() or {}
                fused = dict(signal_status.get("fused", {}) or {})
                guidance = interaction_signals.get_prompt_guidance() if hasattr(interaction_signals, "get_prompt_guidance") else ""
                new_state.response_modifiers["interaction_signals"] = signal_status
                if guidance:
                    new_state.response_modifiers["conversational_dynamics"] = (
                        f"{engine.get_prompt_injection()}\n\n{guidance}"
                    )
                engagement = float(fused.get("engagement", 0.0) or 0.0)
                activation = float(fused.get("activation", 0.0) or 0.0)
                cog.conversation_energy = max(
                    cog.conversation_energy,
                    min(1.0, (engagement * 0.75) + (activation * 0.25)),
                )
                cog.modifiers["interaction_summary"] = str(fused.get("summary") or "")
                cog.modifiers["interaction_pacing"] = str(fused.get("pacing") or "steady")
                cog.modifiers["interaction_verbosity_bias"] = str(fused.get("verbosity_bias") or "balanced")
        except (ImportError, AttributeError, RuntimeError) as exc:
            _record_conversational_degradation(
                exc,
                action="continued with core conversational dynamics after interaction signal integration failed",
            )
            logger.debug("ConversationalDynamics: interaction signal integration skipped: %s", exc)

        await self._execute_store_callback_topics(active_user_id, cog, dynamics, new_state, objective, origin, state)

        self._execute_narrative_gravity_autobiographical(dynamics, new_state, objective)
        # What they just did, in the form it took. An act that cost them is
        # care whatever shape it arrived in: a correction risks the exchange
        # to tell her she is wrong, and a reader that scored unwelcome as
        # unkind would read the truth-teller as an enemy and the flatterer as
        # a friend. See core/social/the_kind_it_was.py.
        self._name_the_kind(new_state, objective)
        return new_state

    async def execute(self, state: AuraState, objective: str | None = None, **kwargs) -> AuraState:
        if not objective:
            return state

        engine = self._get_engine()
        if not engine:
            return state

        origin = kwargs.get("origin", state.cognition.current_origin or "system")

        # Only run full analysis on user-facing messages
        if origin not in ("user", "voice", "admin", "web"):
            return state

        try:
            new_state = await self._execute_new_state(engine, objective, origin, state)

            return new_state

        except (ImportError, AttributeError, RuntimeError) as e:
            _record_conversational_degradation(
                e,
                action="returned original state after conversational dynamics phase failed",
                severity="error",
            )
            logger.warning("ConversationalDynamics phase failed: %s", e)
            return state
