import asyncio
import inspect
import json
import logging
import time
from typing import Any

from core.runtime.errors import FallbackClassification, record_degradation
from core.soma.effort import note_effort
from core.utils.queues import decode_stringified_priority_message, role_for_origin
from core.utils.task_tracker import get_task_tracker

from ..state.aura_state import AuraState
from . import BasePhase

logger = logging.getLogger(__name__)

#: When recall has taken long enough to be worth saying where the time went:
#: the kernel's own budget for a background phase (core/kernel/aura_kernel.py).
_SLOW_ENOUGH_TO_SAY_S = 1.5

_MEMORY_RECOVERABLE_ERRORS = (
    ImportError,
    AttributeError,
    RuntimeError,
    TypeError,
    ValueError,
    OSError,
    ConnectionError,
    TimeoutError,
)


def _record_memory_degradation(
    error: BaseException,
    *,
    action: str,
    stage: str,
    severity: str = "warning",
    extra: dict[str, Any] | None = None,
) -> None:
    metadata = dict(extra or {})
    metadata["stage"] = stage
    try:
        record_degradation(
            "memory_retrieval",
            error,
            severity=severity,  # type: ignore[arg-type]
            action=action,
            classification=FallbackClassification.SAFE_FALLBACK,
            extra=metadata,
        )
    except TypeError:
        record_degradation(
            "memory_retrieval",
            error,
            severity=severity,  # type: ignore[arg-type]
            action=action,
        )


def _safe_text(value: Any, *, max_chars: int = 12_000) -> str:
    if value is None:
        return ""
    try:
        text = str(value)
    except (RuntimeError, TypeError, ValueError):
        return ""
    text = text.replace("\x00", "").strip()
    if len(text) > max_chars:
        return text[:max_chars]
    return text


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value if value is not None else default)
    except (RuntimeError, TypeError, ValueError):
        return default


def _iter_retrieval_items(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, dict):
        return [value]
    if isinstance(value, str):
        return [value]
    try:
        return list(value)
    except (RuntimeError, TypeError, ValueError):
        return []


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def _safe_metadata(raw: Any) -> dict[str, Any]:
    """Coerce metadata to a dict.  Knowledge-graph rows store metadata as a
    JSON TEXT column; if the upstream forgot to parse it we handle it here."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, ValueError) as _exc:
            logger.debug(
                "MemoryRetrieval: ignored malformed memory metadata: %s",
                _exc,
            )
        return {}
    return {}


#: What the retrieval organ reports before it has a distribution to compare
#: against, and therefore what an ordinary moment reads as. Above it, the
#: moment is more unlike her ordinary life than an ordinary one is.
ORDINARY_NOVELTY: float = 0.5


def _shared_with(metadata: dict[str, Any], partner_id: str) -> bool:
    """Whether a recalled record was written with the person she is with now.

    Read off the principal the record carries. A record with no principal
    says nothing about who was there, which is different from saying nobody
    was.
    """
    partner = " ".join(str(partner_id or "").split()).casefold()
    if not partner or partner == "local_user":
        return False
    recorded = " ".join(
        str(metadata.get("principal_id") or metadata.get("user_id") or "").split()
    ).casefold()
    return bool(recorded) and recorded == partner


def novelty_deepens(novelty: float) -> int:
    """How much further to look, given how unlike her ordinary life this is.

    A moment she has no precedent for is searched wider: her existing memories
    are less likely to answer it in the first few, which is the same argument
    the retrieval organ already makes for choosing a broader plan.

    Development's only other route into recall is that organ's discrete breadth
    choice, and it starts by agreeing with the incumbent plan — so over a
    sixty-round campaign it never disagreed, the plan was identical in every
    arm, and displacing development changed nothing that came back. N->M
    measured exactly 0.000 with a p-value of one.
    """
    try:
        reading = float(novelty)
    # not a failure: a value that is not a number is not one this can read.
    except (TypeError, ValueError):
        return 0
    if reading != reading:  # NaN is an absent reading, not a novel moment
        return 0
    return 1 if reading > ORDINARY_NOVELTY else 0



def _percept_cue(state: Any) -> str:
    """The content of the most salient percept memory has not taken yet. See `_percept_reading`."""
    return _percept_reading(state)[0]


def _candidate_pool(limit: int) -> int:
    """How many memories to search for, so feeling can choose which of them come back.

    Retrieval weighs every candidate by importance, how its feeling matches hers
    and how salient memory is now, then keeps `limit`. Searching for only
    `limit` left that weighing a pool the size of the answer: when stored texts
    are alike the search's cut decided recall and her state reordered five
    memories it never chose. The facade's own pool rule, where it is available.
    """
    try:
        from core.memory.memory_facade import MemoryFacade

        return int(MemoryFacade.candidate_pool(int(limit)))
    except (ImportError, AttributeError, TypeError, ValueError):
        return int(limit)


def _percept_reading(state: Any) -> tuple[str, float, str]:
    """The content, salience and kind of the most salient percept memory has not taken yet.

    Every fresh percept is marked as taken by memory, so one percept cues one
    recall and the next turn is asked about what arrives next. What recall
    itself put into the stream is taken without being asked about: a
    recollection is not something she perceived, and cueing on it would make
    every recall the question for the next.
    """
    try:
        from core.state.percepts import fresh_for, mark_consumed, read_percept

        world = getattr(state, "world", None)
        best = None
        for item in fresh_for(getattr(world, "recent_percepts", None), "memory"):
            reading = read_percept(item)
            mark_consumed(item, "memory")
            if reading.raw.get("source") == "memory_retrieval":
                continue
            if reading.content.strip() and (best is None or reading.salience > best.salience):
                best = reading
        if best is None:
            return "", 0.0, ""
        return _safe_text(best.content), max(0.0, min(1.0, float(best.salience))), str(best.kind or "")
    except _MEMORY_RECOVERABLE_ERRORS as exc:
        _record_memory_degradation(
            exc,
            action="searched without a percept as a cue",
            stage="percept_cue",
        )
        return "", 0.0, ""

def _rescored(text: str, bounded: float, updated: float) -> str:
    if updated != bounded and text.startswith("[memory score="):
        return f"[memory score={updated:.3f}]" + text.split("]", 1)[1]
    return text


def _cued_by_the_percept(
    memory_candidates: list[tuple[float, str]],
    felt_by_text: dict[str, Any],
    percept_cue: str,
    percept_salience: float,
    percept_kind: str,
) -> list[tuple[float, str]]:
    """What she perceives changes what a recollection is worth now.

    The same memory bears on a moment differently depending on what is in
    front of her, and retrieval ranked recollections by how well they matched
    the question and how they felt, never by the percept that arrived with it.
    A recollection closes the gap to full relevance in proportion to how much
    of the percept it carries and how salient the percept was, which is
    attention's gain with the roles the other way round. It carries the
    percept in its words or in the feeling it was made in. See
    core/memory/felt_at_encoding.py.
    """
    if not (percept_cue and percept_salience > 0.0):
        return memory_candidates
    from core.memory.felt_at_encoding import distinctive, percept_carried
    from core.state.percepts import PERCEPT_EMOTIONS, word_overlap

    appraisal = tuple(PERCEPT_EMOTIONS.get(percept_kind, ()) or ())
    set_apart = distinctive(felt_by_text) if appraisal else {}
    reread: list[tuple[float, str]] = []
    for score, text in memory_candidates:
        bounded = max(0.0, min(1.0, float(score)))
        carried = percept_carried(word_overlap(percept_cue, text), appraisal, set_apart.get(text))
        updated = bounded + (1.0 - bounded) * carried * percept_salience
        reread.append((updated, _rescored(text, bounded, updated)))
    return reread


def _cued_by_what_she_feels(
    memory_candidates: list[tuple[float, str]],
    felt_by_text: dict[str, Any],
    affect: Any,
    memory_salience: float,
) -> list[tuple[float, str]]:
    """And what she feels now.

    What was learned in a state comes back more readily in that state (Bower,
    1981), and in the retrieved-context account the emotional state at
    encoding is part of the context that the present state cues (Talmi, Lohnas
    and Daw, 2019). Her mood reached recall only as one valence against
    another on the scale of -1 to 1, and her memories are felt across a band a
    fifth of that wide, so every alignment sat near one and a change in how
    she felt moved a score by a hundredth: on seed 7 a displacement that moved
    her internal geometry by 0.178 moved what she recalled by 0.011. A
    recollection now closes the gap to full relevance by the share of its
    distinctive feeling she holds now, at the gain affect gives memory, the
    way the percept's does at the percept's salience. See
    core/memory/felt_at_encoding.py.
    """
    if not felt_by_text or memory_salience <= 0.0:
        return memory_candidates
    from core.memory.felt_at_encoding import distinctive, felt_now, held_now

    now = felt_now(affect)
    if not now:
        return memory_candidates
    set_apart = distinctive(felt_by_text)
    gain = min(1.0, memory_salience)
    reread: list[tuple[float, str]] = []
    for score, text in memory_candidates:
        bounded = max(0.0, min(1.0, float(score)))
        updated = bounded + (1.0 - bounded) * held_now(now, set_apart.get(text)) * gain
        reread.append((updated, _rescored(text, bounded, updated)))
    return reread


class MemoryRetrievalPhase(BasePhase):
    """
    Phase 2: Memory Retrieval.
    Uses current working memory to retrieve relevant long-term context (RAG)
    and updates the state's long_term_memory field.
    """

    def __init__(self, container: Any):
        self.container = container

    @staticmethod
    def _execute_what_she_means(percept_cue, query, spoken_to, state):
        # What she means to do next cues what comes back to her, when nobody has
        # just spoken. The question on those turns was the objective routing set
        # from input text, so her open goals and initiatives never entered it,
        # and in the subject-core runs no displacement of deliberation reached
        # memory at all. The most urgent open intention joins the question the
        # way an entity cue does; a turn someone spoke on is asked what they
        # said.
        if not spoken_to:
            try:
                from core.state.aura_state import _normalize_goal_text

                open_intentions = [
                    item
                    for item in list(getattr(state.cognition, "active_goals", None) or [])
                    + list(getattr(state.cognition, "pending_initiatives", None) or [])
                    if isinstance(item, dict)
                ]
                pressing = max(
                    open_intentions,
                    key=lambda item: _safe_float(item.get("urgency", item.get("priority"))),
                    default=None,
                )
                cue = _normalize_goal_text(pressing) if pressing is not None else ""
                if cue and len(cue) <= 240 and cue.lower() not in query.lower():
                    query = f"{query} {cue}".strip()[:2000]
            except _MEMORY_RECOVERABLE_ERRORS as exc:
                _record_memory_degradation(
                    exc,
                    action="searched without the most pressing intention as a cue",
                    stage="intention_cue",
                )

        # What she perceives cues what comes back, when nobody has just spoken.
        # Retrieval was asked what someone said or what she was working on, so
        # a percept reached affect, the workspace and the world model and never
        # reached memory. In the content runs every percept class brought back
        # exactly the same memories, and no subject-core run kept a P -> M edge.
        # The most salient percept memory has not yet taken joins the question
        # the way the most pressing intention does.
        if not spoken_to and percept_cue and percept_cue.lower() not in query.lower():
            query = f"{query} {percept_cue}".strip()[:2000]
        return query

    @staticmethod
    def _execute_metadata(affect_sources, content, effective_memory_salience, item, memory_candidates, partner_id, shared_texts, state, felt_by_text=None):
        metadata = _safe_metadata(item.get("metadata", {}))
        emotional_valence = _safe_float(metadata.get("emotional_valence"))
        importance = _safe_float(metadata.get("importance"))
        score = _safe_float(item.get("score"))
        salience = effective_memory_salience
        valence_alignment = 1.0 - min(
            1.0,
            abs(
                _safe_float(getattr(state.affect, "valence", 0.0))
                - emotional_valence
            ),
        )
        weighted_score = round(
            (score * 0.35)
            + (importance * 0.25)
            + (valence_alignment * 0.25)
            + (salience * 0.15),
            3,
        )
        memory_candidates.append(
            (weighted_score, f"[memory score={weighted_score:.3f}] {content}")
        )
        # The feeling it was made in, for the percept. See core/memory/felt_at_encoding.py.
        if felt_by_text is not None and metadata.get("felt"):
            felt_by_text[f"[memory score={weighted_score:.3f}] {content}"] = metadata.get("felt")
        # Whether the person she is with now was part of it.
        # The principal a personal record was written for is
        # stored with it; a record written for somebody else
        # never reaches here. See `_shared_with`.
        if _shared_with(metadata, partner_id):
            shared_texts.add(f"[memory score={weighted_score:.3f}] {content}")

        affect_sources.append((content, emotional_valence, importance))
        return importance

    @staticmethod
    def _execute_part_3(memories, new_state, recall_key, recalled_valence_total, recalled_with_feeling, returns, scores):
        from core.memory.reliving import get_match_ledger
        new_state.cognition.long_term_memory = memories
        new_state.cognition.memory_scores = scores
        # And what she recalled changes which percepts she takes as meant. See
        # core/state/percepts.py `prime_stream`.
        try:
            from core.state.percepts import prime_stream

            prime_stream(getattr(new_state, "world", None), list(zip(memories, scores, strict=True)))
        except _MEMORY_RECOVERABLE_ERRORS as exc:
            _record_memory_degradation(
                exc,
                action="kept the recollections without priming the percept stream",
                stage="percept_priming",
            )
        new_state.cognition.last_retrieval_query = recall_key
        # And say that something came back to her.
        #
        # The affect phase has carried a mapping from `memory_replay` to
        # sadness, joy, trust, nostalgia, warmth and belonging since it was
        # written, and nothing in the tree has ever emitted that percept: a
        # reader with no writer, so recall could put something in front of her
        # and her feeling never heard about it. The intensity is the best match
        # score, so a faint recollection moves affect faintly and there is no
        # threshold to choose.
        # Whether this is a recall she relives or one she looked up. A match
        # far above the matches she usually gets brings the feeling back with
        # it, and the feeling is the one stored with what came back — the same
        # quantity the affective hit above is computed from.
        # See core/memory/reliving.py.
        recalled_feeling = (
            recalled_valence_total / recalled_with_feeling if recalled_with_feeling else 0.0
        )
        reliving = get_match_ledger().reading(
            float(scores[0]) if scores else 0.0,
            recalled_feeling,
            returns=returns,
        )
        new_state.cognition.relived = reliving.as_dict()
        return recalled_feeling, reliving

    async def execute(self, state: AuraState, objective: str | None = None, **kwargs) -> AuraState:
        """
        Retrieve relevant long-term memories for the most recent user message.

        Queries both the dual-memory RAG store and the knowledge graph in parallel,
        then merges the results into state.cognition.long_term_memory.
        Returns state unchanged if working memory is empty or the last message is not
        from a user.
        """
        # What just arrived is a cue in its own right, so a percept still asks
        # on a turn with nothing in working memory. See `_percept_cue`.
        percept_cue, percept_salience, percept_kind = _percept_reading(state)
        if not state.cognition.working_memory and not percept_cue:
            return state

        # Use the most recent user entry or objective for retrieval
        last_msg = (
            state.cognition.working_memory[-1]
            if state.cognition.working_memory
            else {"role": "perception", "content": ""}
        )
        if isinstance(last_msg, dict):
            query = last_msg.get("content", "")
        else:
            query = _safe_text(last_msg)
            last_msg = {"role": "user", "content": query}
        decoded_payload, decoded_origin, was_decoded = decode_stringified_priority_message(query)
        if was_decoded:
            if isinstance(decoded_payload, dict):
                query = decoded_payload.get("content", "")
                if decoded_payload.get("origin"):
                    decoded_origin = decoded_payload["origin"]
            else:
                query = str(decoded_payload)
            if decoded_origin:
                last_msg = {
                    **last_msg,
                    "origin": decoded_origin,
                    "role": role_for_origin(decoded_origin),
                }
        query = _safe_text(query)

        # What she is working on, when the last thing in mind is not something
        # a person said. Retrieval used to run only when the newest
        # working-memory entry came from the user — "to save cycles" — and the
        # newest entry after a turn is her own reply, so across an ordinary
        # hour of thinking she recalled nothing at all. Every autonomous turn,
        # every turn of her own problem solving, every idle turn: no recall,
        # and the recalled set every consumer reads went stale and stayed
        # stale. A memory that can only be reached when somebody speaks is not
        # hers.
        #
        # The cycles are still saved, by the thing that was actually costing
        # them: a query already answered is not asked again.
        spoken_to = bool(query) and last_msg.get("role") == "user"
        if not query or last_msg.get("role") != "user":
            query = _safe_text(
                getattr(state.cognition, "current_objective", "") or objective or ""
            )
        if not query:
            query = percept_cue
        if not query:
            return state

        if len(query) < 5:
            return state

        #: The question before any cue is appended to it. What repeats across a
        #: life is the thing being asked, not the decorations this turn put on
        #: it, and the return ledger has to key on the part that comes back.
        asked_about = query[:240]

        try:
            from core.runtime.proof_policy import is_strict_proof_answer_prompt

            proof_origin = (
                last_msg.get("origin")
                or getattr(state.cognition, "current_origin", None)
                or kwargs.get("origin")
            )
            if is_strict_proof_answer_prompt(query or objective or "", origin=proof_origin):
                new_state = state.derive("memory_retrieval_skipped_for_strict_proof")
                new_state.cognition.long_term_memory = []
                new_state.cognition.memory_scores = []
                new_state.response_modifiers["proof_memory_retrieval_skipped"] = True
                return new_state
        except _MEMORY_RECOVERABLE_ERRORS as exc:
            _record_memory_degradation(
                exc,
                action="continued memory retrieval after strict proof guard failed",
                stage="strict_proof_memory_guard",
            )

        try:
            affect_signature = (
                state.affect.get_cognitive_signature()
                if hasattr(state.affect, "get_cognitive_signature")
                else {}
            )
            if not isinstance(affect_signature, dict):
                affect_signature = {}
        except _MEMORY_RECOVERABLE_ERRORS as exc:
            affect_signature = {}
            _record_memory_degradation(
                exc,
                action="used neutral affect signature after affect state read failed",
                stage="affect_signature",
            )
        response_modifiers = dict(getattr(state, "response_modifiers", {}) or {})
        contract = response_modifiers.get("response_contract", {}) or {}
        imagination_memory_pressure = _safe_float(
            response_modifiers.get("imagination_memory_pressure")
        )
        bicameral_causal_effects = response_modifiers.get("bicameral_causal_effects")
        if not isinstance(bicameral_causal_effects, dict):
            bicameral_causal_effects = {}
        bicameral_memory_priority = _safe_float(
            response_modifiers.get("bicameral_memory_priority")
            or bicameral_causal_effects.get("memory_priority")
        )
        imagination_verification_pressure = _safe_float(
            response_modifiers.get("imagination_verification_pressure")
            or response_modifiers.get("verification_pressure")
        )
        bicameral_verification_pressure = _safe_float(
            response_modifiers.get("bicameral_verification_pressure")
            or bicameral_causal_effects.get("verification_pressure")
        )
        verification_pressure = max(
            imagination_verification_pressure,
            bicameral_verification_pressure,
            _safe_float(response_modifiers.get("verification_pressure")),
        )
        effective_memory_salience = max(
            _safe_float(affect_signature.get("memory_salience")),
            imagination_memory_pressure,
            bicameral_memory_priority,
        )
        retrieval_limit = 5
        if (
            contract.get("requires_memory_grounding")
            or response_modifiers.get("requires_memory_grounding")
            or imagination_memory_pressure > 0.55
            or bicameral_memory_priority > 0.55
        ):
            retrieval_limit += 2
        if effective_memory_salience > 0.65:
            retrieval_limit += 1
        if verification_pressure > 0.55:
            retrieval_limit += 1
        hot_limit = 4 if _safe_float(affect_signature.get("social_hunger")) > 0.65 else 3

        # ── Consciousness-driven memory modulation ──
        # High attention coherence (flow state) → retrieve more (deeper context)
        # High free energy (surprise) → retrieve more (need grounding)
        # Low homeostasis vitality → retrieve less (conserve resources)
        try:
            from core.container import ServiceContainer

            attention = ServiceContainer.get("attention_schema", default=None)
            if attention and hasattr(attention, "is_in_flow") and attention.is_in_flow():
                retrieval_limit += 2  # Flow state: deeper retrieval
            fe_engine = ServiceContainer.get("free_energy_engine", default=None)
            if fe_engine and fe_engine.current and _safe_float(fe_engine.current.free_energy) > 0.6:
                retrieval_limit += 1  # High surprise: need more grounding
            homeostasis = ServiceContainer.get("homeostasis", default=None)
            if homeostasis and _safe_float(homeostasis.compute_vitality(), default=0.5) < 0.35:
                retrieval_limit = max(2, retrieval_limit - 2)  # Low energy: conserve
            # A moment unlike the ordinary run of her life is searched wider:
            # her existing memories are less likely to answer it on the first
            # few, which is the same argument the retrieval organ already makes
            # for choosing a broader plan.
            #
            # Development had one route into recall and it was that organ's
            # discrete breadth choice, which starts by agreeing with the
            # incumbent and only disagrees once its head has learned. Over a
            # sixty-round campaign it never disagreed, so the plan was identical
            # in every arm and displacing development changed nothing that came
            # back: N->M measured exactly 0.000 with a p-value of one. This is
            # a continuous dependence on the same reading, in the form every
            # other modulation here takes.
            #
            # The reference is the organ's own: `novelty()` returns 0.5 until it
            # has a distribution to compare against, so above a half is more
            # unlike her ordinary life than an ordinary moment is.
            from core.ontogeny.control_points import novelty_now

            retrieval_limit += novelty_deepens(novelty_now())
        except _MEMORY_RECOVERABLE_ERRORS as exc:
            _record_memory_degradation(
                exc,
                action="kept bounded default retrieval limits after substrate modulation failed",
                stage="retrieval_modulation",
            )

        # Entity memory targets the search. When Aura has recognised a person,
        # place, or thing in what was said, that entity's canonical name and its
        # best-evidenced associations are appended to the query, so retrieval
        # looks for what she KNOWS about it rather than only the literal words
        # the user happened to type ("he" retrieves nothing; the resolved name
        # and its bound episode ids retrieve the history).
        try:
            entity_cues = response_modifiers.get("entity_retrieval_cues")
            if isinstance(entity_cues, (list, tuple)) and entity_cues:
                seen_cue: set[str] = set()
                extra: list[str] = []
                lowered_query = query.lower()
                for cue in entity_cues:
                    text = str(cue or "").strip()
                    if not text or len(text) > 120:
                        continue
                    folded = text.lower()
                    if folded in seen_cue or folded in lowered_query:
                        continue
                    seen_cue.add(folded)
                    extra.append(text)
                    if len(extra) >= 6:
                        break
                if extra:
                    query = f"{query} {' '.join(extra)}".strip()[:2000]
                    logger.debug(
                        "🧠 MemoryRetrieval: entity memory added %d retrieval cue(s).",
                        len(extra),
                    )
        except _MEMORY_RECOVERABLE_ERRORS as exc:
            _record_memory_degradation(
                exc,
                action="searched without entity-memory retrieval cues",
                stage="entity_cue_targeting",
            )

        query = self._execute_what_she_means(percept_cue, query, spoken_to, state)

        # The question and the depth it is asked at. The same words asked with a
        # different limit are a different recall: affect's memory salience, the
        # imagination and bicameral pressures, flow, surprise and vitality all
        # set the limit, and with the skip keyed on the words alone none of them
        # could change what came back for as long as the objective stayed the
        # same. The cost the skip saves is still saved for a repeated question.
        # Asking again goes further rather than returning early. The ladder
        # doubles, so the second asking looks one deeper and the eighth three,
        # and because the depth is part of the key below, a repeat is no longer
        # skipped. See core/memory/reliving.py.
        from core.memory.reliving import deeper, get_return_ledger

        # Keyed on what actually repeats. By this point the query carries the
        # entity cues and the most pressing intention, and both move every
        # turn, so the same question asked again was never the same string and
        # the return count sat at zero for the whole of a campaign — the ladder
        # that makes a repeat go deeper could not start. The objective is the
        # part that comes back.
        returns = get_return_ledger().returns(asked_about)
        retrieval_limit += deeper(returns)

        recall_key = f"{query}\x1f{retrieval_limit}\x1f{hot_limit}"
        if recall_key == getattr(state.cognition, "last_retrieval_query", None):
            return state

        logger.info("🧠 MemoryRetrieval: Searching for context: %s...", query[:50])

        async def _get_dual():
            try:
                mm = self.container.get("memory_manager", default=None)
                if mm and hasattr(mm, "dual_memory"):
                    async with asyncio.timeout(15.0):
                        return await mm.dual_memory.retrieve_context(query)
            except TimeoutError as exc:
                logger.debug(
                    "MemoryRetrieval: optional DualMemory RAG timed out; continuing without it: %s",
                    exc,
                )
                return None
            except _MEMORY_RECOVERABLE_ERRORS as exc:
                _record_memory_degradation(
                    exc,
                    action="continued retrieval without dual-memory context",
                    stage="dual_memory",
                )
                logger.debug("MemoryRetrieval: DualMemory RAG failed: %s", exc)
                return None
            return None

        async def _get_kg():
            try:
                kg = self.container.get("knowledge_graph", default=None)
                if kg:
                    method = kg.search_knowledge
                    async with asyncio.timeout(15.0):
                        if inspect.iscoroutinefunction(method):
                            return await method(query, limit=retrieval_limit)
                        else:
                            return await asyncio.to_thread(method, query, limit=retrieval_limit)
            except TimeoutError as exc:
                logger.debug(
                    "MemoryRetrieval: optional KnowledgeGraph retrieval timed out; continuing without it: %s",
                    exc,
                )
                return None
            except _MEMORY_RECOVERABLE_ERRORS as exc:
                _record_memory_degradation(
                    exc,
                    action="continued retrieval without knowledge-graph context",
                    stage="knowledge_graph",
                )
                logger.debug("MemoryRetrieval: KnowledgeGraph search failed: %s", exc)
                return None
            return None

        async def _get_facade():
            try:
                memory = self.container.get("memory_facade", default=None)
                if not memory:
                    return None

                recalled = []
                if hasattr(memory, "search"):
                    async with asyncio.timeout(15.0):
                        recalled.extend(
                            list(
                                await _maybe_await(memory.search(query, limit=_candidate_pool(retrieval_limit)))
                                or []
                            )
                        )

                if hasattr(memory, "get_hot_memory"):
                    async with asyncio.timeout(15.0):
                        hot = await _maybe_await(memory.get_hot_memory(limit=hot_limit))
                    if isinstance(hot, dict):
                        for episode in hot.get("recent_episodes", []) or []:
                            content_str = ""
                            if hasattr(episode, "to_retrieval_text"):
                                content_str = episode.to_retrieval_text()
                            elif hasattr(episode, "full_description"):
                                content_str = episode.full_description
                            elif isinstance(episode, dict):
                                content_str = episode.get("content") or episode.get("description") or episode.get("context") or str(episode)
                            else:
                                content_str = _safe_text(episode, max_chars=2_000)

                            metadata_dict = {"type": "recent_episode"}
                            valence_val = 0.0
                            importance_val = 0.5

                            if hasattr(episode, "emotional_valence"):
                                valence_val = _safe_float(episode.emotional_valence)
                            elif isinstance(episode, dict):
                                valence_val = _safe_float(episode.get("emotional_valence", 0.0))

                            if hasattr(episode, "importance"):
                                importance_val = _safe_float(episode.importance, default=0.5)
                            elif isinstance(episode, dict):
                                importance_val = _safe_float(episode.get("importance", 0.5), default=0.5)

                            metadata_dict["emotional_valence"] = valence_val
                            metadata_dict["importance"] = importance_val

                            recalled.append(
                                {
                                    "content": content_str,
                                    "metadata": metadata_dict,
                                    "score": 0.85,
                                }
                            )

                return recalled or None
            except TimeoutError as exc:
                logger.debug(
                    "MemoryRetrieval: optional MemoryFacade retrieval timed out; continuing without it: %s",
                    exc,
                )
                return None
            except _MEMORY_RECOVERABLE_ERRORS as exc:
                _record_memory_degradation(
                    exc,
                    action="continued retrieval without memory-facade context",
                    stage="memory_facade",
                )
                logger.debug("MemoryRetrieval: MemoryFacade search failed: %s", exc)
                return None

        async def _get_episodic():
            try:
                from core.container import ServiceContainer

                ep = self.container.get("episodic_memory", default=None)
                if ep is None:
                    ep = ServiceContainer.get("episodic_memory", default=None)
                if ep and hasattr(ep, "recall_similar_async"):
                    async with asyncio.timeout(15.0):
                        return await ep.recall_similar_async(query, limit=retrieval_limit)
                elif ep and hasattr(ep, "recall_similar"):
                    return await asyncio.to_thread(ep.recall_similar, query, retrieval_limit)
            except TimeoutError as exc:
                logger.debug(
                    "MemoryRetrieval: optional EpisodicMemory recall timed out; continuing without it: %s",
                    exc,
                )
                return None
            except _MEMORY_RECOVERABLE_ERRORS as exc:
                _record_memory_degradation(
                    exc,
                    action="continued retrieval without episodic context",
                    stage="episodic_memory",
                )
                logger.debug("MemoryRetrieval: Episodic recall failed: %s", exc)
            return None

        async def _get_intentional():
            # The task-driven retriever the runtime registers, which asks the
            # ontogenetic organ how wide to search. Only the subject-core
            # harness called it, in one condition of eight, so development had
            # no way to change what she recalls in the running organism.
            try:
                from core.container import ServiceContainer
                from core.memory.intentional_retrieval import RetrievalIntent

                retriever = self.container.get("intentional_retriever", default=None)
                if retriever is None:
                    retriever = ServiceContainer.get("intentional_retriever", default=None)
                if retriever is None or not hasattr(retriever, "retrieve"):
                    return None
                intent = RetrievalIntent(task=query, query=query, limit=retrieval_limit)
                async with asyncio.timeout(15.0):
                    result = await asyncio.to_thread(retriever.retrieve, intent)
                return list(getattr(result, "hits", None) or [])
            except TimeoutError as exc:
                logger.debug(
                    "MemoryRetrieval: optional intentional retrieval timed out; continuing without it: %s",
                    exc,
                )
                return None
            except _MEMORY_RECOVERABLE_ERRORS as exc:
                _record_memory_degradation(
                    exc,
                    action="continued retrieval without the intentional retriever",
                    stage="intentional_retriever",
                )
                return None

        # Each source timed, so a slow phase says which part of it was slow.
        # The kernel measured this phase at five to nine seconds a background
        # tick (live, 2026-09-18) and could only name the phase.
        took: dict[str, float] = {}

        async def _timed(name: str, source: Any) -> Any:
            began = time.monotonic()
            try:
                return await source
            finally:
                took[name] = time.monotonic() - began

        gathered_from = time.monotonic()
        dual_res, kg_res, facade_res, episodic_res, intentional_res = await asyncio.gather(
            _timed("dual memory", _get_dual()),
            _timed("knowledge graph", _get_kg()),
            _timed("memory facade", _get_facade()),
            _timed("episodic", _get_episodic()),
            _timed("intentional", _get_intentional()),
        )
        if time.monotonic() - gathered_from > _SLOW_ENOUGH_TO_SAY_S and took:
            slowest = max(took, key=took.get)
            logger.info(
                "recall took %.1fs; slowest source %s at %.1fs (%s)",
                time.monotonic() - gathered_from,
                slowest,
                took[slowest],
                ", ".join(f"{name} {seconds:.1f}s" for name, seconds in sorted(took.items())),
            )

        memories: list[str] = []
        memory_candidates: list[tuple[float, str]] = []
        # Recollections that include the person she is talking to now.
        shared_texts: set[str] = set()
        try:
            from core.runtime.conversation_support import resolve_primary_user_id

            partner_id = resolve_primary_user_id(state)
        # not a failure: no service here, so the caller falls back to its own default.
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
            partner_id = ""

        # ── Gap 3 Fix: Memory Affect → Steering ──
        # Each candidate's feeling, kept beside it so the nudge below is taken
        # from the memories that came back rather than from everything searched.
        affect_sources: list[tuple[str, float, float]] = []
        # And each candidate's whole feeling, where its store kept one.
        felt_by_text: dict[str, Any] = {}

        if dual_res:
            memory_candidates.append(
                (
                    0.45 + effective_memory_salience * 0.1,
                    _safe_text(dual_res, max_chars=2_000),
                )
            )
        if kg_res:
            for km in _iter_retrieval_items(kg_res):
                if isinstance(km, dict):
                    metadata = _safe_metadata(km.get("metadata", {}))
                    emotional_valence = _safe_float(metadata.get("emotional_valence"))
                    importance = _safe_float(metadata.get("importance"))
                    valence_alignment = 1.0 - min(
                        1.0,
                        abs(_safe_float(getattr(state.affect, "valence", 0.0)) - emotional_valence),
                    )
                    weighted_score = (
                        0.3
                        + (importance * 0.3)
                        + (valence_alignment * 0.2)
                        + (effective_memory_salience * 0.2)
                    )
                    content = _safe_text(km.get("content"), max_chars=2_000)
                    if content:
                        memory_candidates.append(
                            (weighted_score, f"[{km.get('type', 'fact')}] {content}")
                        )
                        if metadata.get("felt"):
                            felt_by_text[f"[{km.get('type', 'fact')}] {content}"] = metadata.get("felt")

                    affect_sources.append((content, emotional_valence, importance))

        if facade_res:
            for item in _iter_retrieval_items(facade_res):
                if isinstance(item, dict):
                    content = _safe_text(item.get("content") or item.get("text"), max_chars=2_000)
                    if content:
                        importance = self._execute_metadata(affect_sources, content, effective_memory_salience, item, memory_candidates, partner_id, shared_texts, state, felt_by_text)
                elif item:
                    memory_candidates.append(
                        (0.35, f"[memory] {_safe_text(item, max_chars=2_000)}")
                    )

        if episodic_res:
            for ep in _iter_retrieval_items(episodic_res):
                desc = _safe_text(
                    getattr(ep, "description", "") or getattr(ep, "context", "") or ep,
                    max_chars=2_000,
                )
                outcome = _safe_text(getattr(ep, "outcome", ""), max_chars=1_000)
                importance = _safe_float(getattr(ep, "importance", 0.5), default=0.5)
                valence = _safe_float(getattr(ep, "emotional_valence", 0.0))
                content = f"{desc}" + (f" → {outcome}" if outcome and outcome != desc else "")
                if content and len(content) > 10:
                    score = 0.5 + importance * 0.3 + abs(valence) * 0.2
                    memory_candidates.append((score, f"[episodic] {content}"))
                    affect_sources.append((content, valence, importance))

        for hit in intentional_res or []:
            content = _safe_text(getattr(hit, "content", ""), max_chars=2_000)
            # A memory another store already returned is not a second memory.
            if not content or any(content in text for _, text in memory_candidates):
                continue
            score = max(0.0, min(1.0, _safe_float(getattr(hit, "score", 0.0))))
            memory_candidates.append((score, f"[{getattr(hit, 'store_type', 'memory')}] {content}"))

        # The nudge is pushed after the cut, below, from what came back.
        def _push_affect_of_recalled(recalled_texts: list[str]) -> tuple[float, int]:
            total_valence_hit = 0.0
            total_arousal_hit = 0.0
            memory_hits = 0
            remaining: dict[str, int] = {}
            for text in recalled_texts:
                body = text.split("] ", 1)[1] if text.startswith("[") and "] " in text else text
                remaining[body] = remaining.get(body, 0) + 1
            for content, valence, importance in affect_sources:
                if remaining.get(content, 0) <= 0 or abs(valence) <= 0.3:
                    continue
                remaining[content] -= 1
                total_valence_hit += valence * importance
                total_arousal_hit += importance * 0.5
                memory_hits += 1
            if memory_hits > 0:
                try:
                    from core.container import ServiceContainer

                    affect_engine = ServiceContainer.get("affect_engine", default=None)
                    if affect_engine and hasattr(affect_engine, "modify"):
                        val_shift = (total_valence_hit / memory_hits) * 0.4
                        arousal_shift = (total_arousal_hit / memory_hits) * 0.3

                        logger.debug(
                            "💥 Memory retrieval triggered affective hit: val_shift=%.2f, arousal_shift=%.2f",
                            val_shift,
                            arousal_shift,
                        )

                        modification = affect_engine.modify(
                            dv=val_shift,
                            da=arousal_shift,
                            de=0.0,
                            source="memory_retrieval",
                        )
                        get_task_tracker().create_task(
                            modification,
                            name="memory_retrieval.affective_hit",
                        )
                except _MEMORY_RECOVERABLE_ERRORS as exc:
                    close = getattr(locals().get("modification", None), "close", None)
                    if callable(close):
                        close()
                    _record_memory_degradation(
                        exc,
                        action="kept retrieved memories after affect scheduling failed",
                        stage="affective_memory_hit",
                    )
                    logger.debug("Failed to push memory affect: %s", exc)
            return total_valence_hit, memory_hits

        # Recall costs the search, not the finding. Reported after the early
        # return below, a recall that came back empty cost nothing — so the
        # body could not feel a fruitless search, and the column that carries
        # this was constant for the whole of every recording.
        note_effort("recall", max(1, len(memory_candidates)))

        # The percept and what she feels now each cue what comes back.
        memory_candidates = _cued_by_the_percept(
            memory_candidates, felt_by_text, percept_cue, percept_salience, percept_kind
        )
        memory_candidates = _cued_by_what_she_feels(
            memory_candidates, felt_by_text, getattr(state, "affect", None), effective_memory_salience
        )

        scores: list[float] = []
        # The feeling stored with what came back, and how many carried one.
        recalled_valence_total, recalled_with_feeling = 0.0, 0
        if memory_candidates:
            memory_candidates.sort(key=lambda item: item[0], reverse=True)
            kept = memory_candidates[:retrieval_limit]
            memories = [text for _, text in kept]
            # And how well each one matched. The ranking is computed here and
            # was discarded here, so every consumer downstream saw an unranked
            # list of strings and had to treat a recollection that answered the
            # question exactly the same as one that scraped in last.
            scores = [round(float(score), 4) for score, _ in kept]
            recalled_valence_total, recalled_with_feeling = _push_affect_of_recalled(memories)

        if not memories:
            return state

        # Derive new state with retrieved context
        new_state = state.derive("memory_retrieval")
        recalled_feeling, reliving = self._execute_part_3(memories, new_state, recall_key, recalled_valence_total, recalled_with_feeling, returns, scores)
        # Joint recall: whether what came back is something she and the person
        # she is with now were both part of. "Remember the Time" asks fifteen
        # times and every one is a memory marked as theirs together.
        new_state.cognition.relived["shared"] = bool(
            memory_candidates and memory_candidates[0][1] in shared_texts
        )
        get_return_ledger().note(asked_about)

        try:
            from core.state.percepts import emit_percept

            emit_percept(
                new_state.world,
                "memory_replay",
                content=str(memories[0])[:200],
                intensity=max(0.0, min(1.0, float(scores[0]) if scores else 0.0)),
                source="memory_retrieval",
                relived=reliving.relived,
                feeling=round(recalled_feeling, 4),
            )
        except _MEMORY_RECOVERABLE_ERRORS as exc:
            _record_memory_degradation(
                exc,
                action="recall reached the state but not affect",
                stage="memory_replay_percept",
                severity="debug",
            )
        new_state.response_modifiers["memory_retrieval_signature"] = {
            "query": query[:160],
            "retrieval_limit": retrieval_limit,
            "hot_limit": hot_limit,
            "affect": affect_signature,
            "imagination_memory_pressure": round(imagination_memory_pressure, 4),
            "imagination_verification_pressure": round(imagination_verification_pressure, 4),
            "bicameral_memory_priority": round(bicameral_memory_priority, 4),
            "bicameral_verification_pressure": round(bicameral_verification_pressure, 4),
            "verification_pressure": round(verification_pressure, 4),
        }
        return new_state
