"""Refactored CognitiveEngine - Now a thin facade over modular phases."""

import asyncio
import json
import logging
import re
import sqlite3  # noqa: F401  (read at call time by the lifted module)
import time
import uuid  # noqa: F401  (read at call time by the lifted module)
from collections import deque
from contextlib import suppress  # noqa: F401  (read at call time by the lifted module)
from typing import Any

from core.consciousness.executive_authority import get_executive_authority
from core.conversation.continuation import (
    continuation_prompt_prefix,  # noqa: F401  (read at call time by the lifted module)
    continuation_state_text,  # noqa: F401  (read at call time by the lifted module)
)
from core.goals.objective_lifecycle import (
    finalize_foreground_turn_state,  # noqa: F401  (read at call time by the lifted module)
    is_foreground_objective_origin,  # noqa: F401  (read at call time by the lifted module)
    normalize_objective_origin,
)
from core.language.terminal_boundary import has_terminal_sentence_boundary
from core.memory.retention_policy import working_history_retention_policy
from core.runtime import (  # noqa: F401  (read at call time by the lifted module)
    background_policy,
    response_policy,
)
from core.runtime.errors import (
    record_degradation,  # noqa: F401  (read at call time by the lifted module)
)
from core.runtime.flags import env_present
from core.runtime.lockdep import LockRank, checked_lock
from core.runtime.pipeline_blueprint import (
    instantiate_legacy_runtime_phases,
    legacy_runtime_phase_specs,
)
from core.runtime.service_registry import get_runtime_service
from core.runtime.structured_input import (
    answer_surface_token_floor,  # noqa: F401  (read at call time by the lifted module)
)
from core.runtime.task_ownership import (
    create_owned_asyncio_task,  # noqa: F401  (read at call time by the lifted module)
)
from core.runtime.turn_outcome import (
    TurnOutcome,
    UserVisibleState,
    bind_turn,
    current_turn,
    finalize_turn,
    recoverable_answer,  # noqa: F401  (read at call time by the lifted module)
)
from core.state.aura_state import (  # noqa: F401  (read at call time by the lifted module)
    AuraState,
    CognitiveMode,
)
from core.utils.concurrency import RobustLock
from core.utils.queues import USER_FACING_ORIGINS
from core.verify import influence_channels  # noqa: F401  (read at call time by the lifted module)
from core.verify.influence_receipt import InfluenceReceipt, build_influence_receipt
from core.verify.lesion_registry import (
    apply_channel,  # noqa: F401  (read at call time by the lifted module)
    get_lesion_registry,  # noqa: F401  (read at call time by the lifted module)
    register_flag_lesion,
)
from core.verify.turn_receipt import (
    TurnReceipt,
    record_latency,  # noqa: F401  (read at call time by the lifted module)
    record_phase,  # noqa: F401  (read at call time by the lifted module)
    record_response_path,  # noqa: F401  (read at call time by the lifted module)
    recording_turn,
)

from .autopoiesis import AutopoieticGraph
from .cognitive_augmentors import _RunsItsAugmentors
from .cognitive_engine_prompt_fitting import (  # noqa: F401  (re-exported: they were defined here)
    _fit_prompt_to_what_the_turn_can_read,
    _record_the_capability_inventory_miss,
)
from .cognitive_engine_quick_reply import _AnswersTheDesktopDirectly
from .cognitive_engine_thinking_loop import _RunsTheThinkingLoop
from .live_mind_contract import (
    REQUIRED_LIVE_MIND_GENERATION_CONTROL_KEYS,
    normalize_live_mind_surface_control_receipt,  # noqa: F401  (read at call time by the lifted module)
)
from .llm.context_assembler import ContextAssembler
from .reasoning_strategies import ReasoningStrategies, StrategyType
from .request_contract import (
    project_user_surface_resume_capability,  # noqa: F401  (read at call time by the lifted module)
)
from .types import ThinkingMode, Thought  # noqa: F401  (read at call time by the lifted module)

logger = logging.getLogger(__name__)

_USER_FACING_ORIGINS = USER_FACING_ORIGINS

_THOUGHT_HISTORY_LIMIT = working_history_retention_policy(
    "AURA_COGNITIVE_THOUGHT_HISTORY_MAX"
).max_items

_BACKGROUND_REFLECTIVE_MODES = frozenset(
    {
        ThinkingMode.REFLECTIVE,
        ThinkingMode.CREATIVE,
    }
)
_COGNITIVE_ENGINE_RECOVERABLE_ERRORS = (
    AttributeError,
    ConnectionError,
    ImportError,
    LookupError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)

# A user-facing caller may admit a larger, measured completion surface. The
# former universal 240-second cap silently shortened that admitted deadline;
# on the resident 32B a 2,560-token technical answer then had no possible path
# to completion. Background and unowned cycles retain the original ceiling.
_DEFAULT_COGNITIVE_CYCLE_MAX_S = 240.0





def _nothing_has_ever_arrived() -> bool:
    """True when this turn has produced no sign of work at all."""

    try:
        from core.runtime.turn_progress import seconds_since_progress

        return seconds_since_progress() < 0.0
    except (ImportError, AttributeError, TypeError, ValueError):
        # not a failure: with no progress clock to read, this cannot say
        # nothing has arrived, and the callers only act on True.
        return False


def _the_longest_this_turn_may_take(floor_s: float, *, user_facing: bool) -> float:
    """The turn's ceiling, sized for this machine rather than fixed at 480.

    The cycle already holds itself open while tokens are arriving; this is the
    bound it stops at. A flat number cannot tell a turn that is running away
    from one that is working, and on this host it stood below the cost of the
    work: LIVE 2026-08-29, a turn dispatched code_repl seven minutes in and was
    cut at exactly 480 seconds with the sandbox still running —
    "reactive_recovery:timeout".

    The same measurement the endpoint wait uses, so the two agree about how
    long a turn of this shape costs here. It only ever raises the caller's
    floor, and only for somebody who is waiting.
    """

    if not user_facing:
        return float(floor_s)
    try:
        from core.brain.llm.mlx_client import longest_a_turn_may_take
        from core.brain.llm_health_router import (
            _A_TURNS_ANSWER_TOKENS,
            _A_TURNS_PROMPT_CHARS,
            _GENERATIONS_A_TOOL_TURN_MAY_TAKE,
        )

        return longest_a_turn_may_take(
            generations=_GENERATIONS_A_TOOL_TURN_MAY_TAKE,
            prompt_chars=_A_TURNS_PROMPT_CHARS,
            max_tokens=_A_TURNS_ANSWER_TOKENS,
            floor_s=float(floor_s),
        )
    except (ImportError, AttributeError, TypeError, ValueError):
        return float(floor_s)


async def _keep_the_cycle_open_while_it_is_working(
    clock: Any, *, ceiling_at: float, user_facing: bool,
    runtime_context: dict[str, Any] | None = None,
) -> None:
    """Push the cycle deadline out while tokens are still arriving.

    The clocks in a turn were raised one at a time today and that is the same
    design with larger numbers. A stopwatch cannot tell a generation that is
    working from one that is stuck, and this one was only ever stopping the
    first kind: a wedged worker never reaches here, and a decode looping
    forever is caught by the sentinel that reads the output.

    A bound foreground turn renews from its own progress. Its initial window
    is never shortened. Unowned calls retain the historical ceiling, and
    background work does not renew. Worker faults and user cancellation keep
    their existing owners.
    """

    if not user_facing:
        return
    from core.brain.llm.thinking_reserve import seconds_to_decode
    from core.runtime.turn_progress import normal_gap_between_tokens, still_producing

    # What a silence means, measured here and handed down: core/runtime is a
    # foundation and does not reach up to the lane that times decoding.
    quiet_for = normal_gap_between_tokens(float(seconds_to_decode(64)))
    loop = asyncio.get_running_loop()
    owned_foreground = current_turn() is not None
    said_it_once = False
    try:
        while True:
            await asyncio.sleep(1.0)
            now = time.monotonic()
            if owned_foreground:
                # Never shorten the admitted initial window. Once work starts,
                # each fresh reading renews it; silence leaves the timer alone.
                if still_producing(within_s=quiet_for):
                    try:
                        current_deadline = clock.when()
                        if current_deadline is None:
                            return
                        clock.reschedule(max(current_deadline, loop.time() + max(15.0, quiet_for)))
                        if runtime_context is not None:
                            runtime_context["cognitive_cycle_deadline_monotonic"] = (
                                time.monotonic() + clock.when() - loop.time()
                            )
                    except (AttributeError, RuntimeError):
                        # not a failure: a clock that will not reschedule
                        # leaves the deadline where it was, and the cycle
                        # below runs against that one.
                        return
                continue
            if now >= ceiling_at:
                return
            # Extended while the turn is under its ceiling, whether or not
            # this layer can see the work.
            #
            # It used to extend only while it could see progress, and it kept
            # losing to gaps it cannot observe: a tool running, a long prefill,
            # a worker between frames. Live on 2026-08-28 a ledgerkit turn read
            # three files and died at 139 seconds with the engine having
            # logged, in the same turn, that it was holding the cycle open.
            #
            # This clock has no way of telling a wedged turn from a working
            # one — it sits above the lanes that do. A silent worker is caught
            # by the first-token ceiling, a livelocked one by the livelock
            # ceiling, a looping decode by the sentinel that reads what is
            # being written, and all three watch the output rather than the
            # hour. What is left for this one to enforce is the ceiling.
            if not still_producing(within_s=quiet_for) and _nothing_has_ever_arrived():
                # Never a token, never a tool: the turn has not started, and
                # the first-token ceiling owns that case.
                return
            # Keep a slice ahead of now, never past the ceiling.
            wanted = min(ceiling_at - now, 15.0)
            if wanted <= 0.0:
                return
            try:
                clock.reschedule(loop.time() + wanted)
                if runtime_context is not None:
                    runtime_context["cognitive_cycle_deadline_monotonic"] = (
                        time.monotonic() + clock.when() - loop.time()
                    )
            except (AttributeError, RuntimeError):
                # not a failure: see the rung above — the existing deadline
                # stands and the cycle runs against it.
                return
            if not said_it_once:
                said_it_once = True
                logger.info(
                    "⏳ [COGNITION] Past the cycle deadline and still producing; "
                    "holding it open while the answer arrives."
                )
    except asyncio.CancelledError:
        # not a failure: this watcher is cancelled when the turn it watches
        # ends, which is every ordinary turn.
        return

def _history_budget_for(system_prompt: str, max_tokens: int) -> int:
    """What is left of the turn's reading budget once the prompt is served.

    Zero when nothing has been timed, which means the conversation goes
    through whole rather than cut by a guess — the same answer
    ``budget_for_answer`` gives for the system prompt.
    """

    try:
        from core.brain.llm.context_budget import budget_for_answer

        afford = budget_for_answer(max_tokens)
    except (ImportError, AttributeError, TypeError, ValueError) as exc:
        record_degradation(
            "cognitive_engine",
            exc,
            action="left the delivered conversation whole, with no reading budget",
        )
        return 0
    if afford <= 0:
        return 0
    return max(0, afford - len(str(system_prompt or "")))


def _fit_history_to_what_is_left(
    history_messages: list[dict[str, str]],
    *,
    system_prompt: str,
    max_tokens: int,
) -> tuple[list[dict[str, str]], str]:
    """Hold the delivered conversation to the same budget as the prompt.

    History was the one input with no budget. LIVE 2026-09-17, "Aura, what
    is it like to be you": 83 messages, 10,414 tokens, 83.44s of prefill
    in front of 71.27s of decode, for a thirty-one character question.
    Forty exchanges were admitted because forty existed.

    The oldest are dropped first, so what remains is a contiguous suffix
    and no reference resolves across a hole. A pair is a user message and
    the reply to it and is kept or dropped together.
    """

    from core.conversation.history_reach import measure_reach

    budget = _history_budget_for(system_prompt, max_tokens)
    if budget <= 0 or not history_messages:
        return history_messages, ""
    pairs: list[list[dict[str, str]]] = []
    for message in history_messages:
        if str(message.get("role") or "") == "user" or not pairs:
            pairs.append([])
        pairs[-1].append(message)
    sized = [
        {"user": "", "aura": "".join(str(m.get("content") or "") for m in pair)}
        for pair in pairs
    ]
    reach = measure_reach(sized, budget_chars=budget)
    if not reach.cuts_anything:
        return history_messages, ""
    kept = [message for pair in pairs[reach.boundary :] for message in pair]
    logger.info(
        "📐 [CONTEXT] conversation %d → %d message(s): %s",
        len(history_messages),
        len(kept),
        reach.reason,
    )
    note = (
        f"[SYSTEM: this conversation has {len(pairs)} completed exchanges and "
        f"the {reach.retained} most recent are below. The other {reach.dropped} "
        "are not in front of you. If the person refers to something older, say "
        "you would need to look it up rather than reconstructing it.]"
    )
    return kept, note


class _RuntimeServiceAdapter:
    """Small compatibility layer for legacy phase constructors expecting container.get."""

    @staticmethod
    def get(name: str, default: Any = None) -> Any:
        return get_runtime_service(name, default=default)


_RUNTIME_SERVICE_ADAPTER = _RuntimeServiceAdapter()


def get_container() -> _RuntimeServiceAdapter:
    """Return the runtime-registry-backed service view used by cognitive phases."""

    return _RUNTIME_SERVICE_ADAPTER


class _NullPassInstrumentation:
    """So the phase loop's contract holds even with no instrumentation."""

    @staticmethod
    def should_run(name: str) -> tuple[bool, int, str]:
        return True, 0, ""


def _pass_instrumentation() -> Any:
    """The pass seam, or a no-op.

    Degrades to a no-op and never to a broken turn: an unavailable debugging
    aid must not be able to stop Aura answering.
    """
    try:
        from core.pipeline.pass_manager import get_instrumentation

        return get_instrumentation()
    except Exception:  # noqa: BLE001 — a debug aid may never break a turn
        logger.debug("pass instrumentation unavailable", exc_info=True)
        return _NullPassInstrumentation()


def _begin_pass_run(label: str) -> None:
    """Number this turn's passes from 1, or do nothing if unavailable.

    Wrapped rather than trusting ``_pass_instrumentation`` to be total: a
    debugging aid must never be the reason a turn fails, and the whole point
    of this seam is that it is on the path every answer takes.
    """
    try:
        begin = getattr(_pass_instrumentation(), "begin_run", None)
        if begin is not None:
            begin(label)
    except Exception:  # noqa: BLE001 — a debug aid may never break a turn
        logger.debug("pass run label %s not recorded", label, exc_info=True)


def _record_legacy_pass(
    name: str,
    ordinal: int,
    duration_s: float,
    *,
    skipped: bool,
    reason: str = "",
    error: str = "",
) -> None:
    """Announce one legacy-pipeline phase to the shared pass record.

    Same ledger the kernel tick writes to, so `AURA_PASS_TRACE=1` and
    `flag_report()` describe the pipeline that is actually serving traffic
    rather than the one that mostly is not.
    """
    try:
        from core.pipeline.pass_manager import PassRecord, get_instrumentation

        get_instrumentation().after_pass(
            PassRecord(
                name=f"legacy_pipeline/{name}",
                ordinal=ordinal,
                duration_s=duration_s,
                skipped=skipped,
                reason=reason,
                error=error,
            )
        )
    except Exception:  # noqa: BLE001 — recording a pass may never break a turn
        logger.debug("pass record dropped for %s", name, exc_info=True)


# ── cognitive provenance on the pipeline that serves chat ────────────────────
#
# Four thin wrappers rather than four inline try/excepts in the phase loop.
# Each one is allowed to fail and none of them may break a turn: a causal
# record that can take the runtime down with it is worse than no record.


def _open_provenance_tick(*, objective: str, priority: bool) -> Any:
    try:
        from core.runtime.cognitive_provenance import open_tick

        return open_tick(objective=str(objective or ""), priority=bool(priority))
    except Exception:  # noqa: BLE001 — provenance may never break a turn
        logger.debug("provenance tick not opened", exc_info=True)
        return None


def _begin_provenance(phase_name: str, state: Any) -> Any:
    try:
        from core.runtime.cognitive_provenance import begin_transformation

        return begin_transformation(phase_name, state)
    except Exception:  # noqa: BLE001
        logger.debug("provenance not started for %s", phase_name, exc_info=True)
        return None


def _complete_provenance(
    transformation: Any, state: Any, *, error: str = "", objective: str = ""
) -> None:
    if transformation is None:
        return
    try:
        transformation.complete(
            state, error=error, inputs={"objective": str(objective or "")[:120]}
        )
    except Exception:  # noqa: BLE001
        logger.debug("provenance receipt dropped", exc_info=True)


def _skip_provenance(phase_name: str, state: Any, reason: str) -> None:
    try:
        from core.runtime.cognitive_provenance import begin_transformation

        begin_transformation(phase_name, state).complete(
            state, skipped=True, skip_reason=str(reason or "")
        )
    except Exception:  # noqa: BLE001
        logger.debug("provenance skip not recorded for %s", phase_name, exc_info=True)


def _close_provenance_tick(graph: Any) -> None:
    if graph is None:
        return
    try:
        from core.runtime.cognitive_provenance import close_tick

        close_tick(graph)
    except Exception:  # noqa: BLE001
        logger.debug("provenance tick not closed", exc_info=True)


def _bounded_float(value: Any, default: float = 0.0, *, lower: float = 0.0, upper: float = 1.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    if parsed != parsed:
        return default
    return max(lower, min(upper, parsed))


def _compact_text(value: Any, *, limit: int = 480) -> str:
    text = " ".join(str(value or "").strip().split())
    if len(text) <= limit:
        return text
    if limit <= 3:
        return text[: max(0, limit)]
    return text[: limit - 3].rstrip() + "..."


def _combine_advisory_token_factors(factors: list[float]) -> float:
    """Combine advisory max_tokens factors without compounding them.

    Multiple advisory frames (spiking inference, imagination, bicameral,
    cognitive situation) each suggest a budget factor. Multiplying them all
    let four mild 0.75-0.85 reductions compound into a ~35% budget
    (768 → ~250 tokens) that cut live user replies off mid-sentence. Only
    the single strongest reduction applies; boosts apply only when nothing
    asks for a reduction.
    """
    if not factors:
        return 1.0
    reductions = [factor for factor in factors if factor < 1.0]
    return min(reductions) if reductions else max(factors)


# Shortest trimmed reply still worth serving instead of losing the turn.
# "The answer is 27." is seventeen characters and is the whole point of the
# turn, so this floor only has to exclude a stub like "Hi." or "Sure.".
_MIN_SALVAGEABLE_REPLY_CHARS = 12

# A sentence boundary the model ran together, e.g. "part of.But — and this is".
# Deliberately narrow: a lowercase letter, terminal punctuation, then a capital
# that starts a lowercase word. Digits are excluded so decimals and version
# numbers survive, and a single capital (U.S.A, initials) will not match because
# the following character must be lowercase.
# Three shapes of run-on, because the first version only caught one of them.
#
# Measured live 2026-07-27: "...leading into Q3?Finally, think about how
# critical this rewrite is" and "...certain states get reinforced.I wouldn't
# call it preference". The original required a LOWERCASE letter before the
# terminator and a capital-plus-lowercase after, so a digit before ("Q3?") and
# a lone capital after (".I ") both slipped through.
#
# "?" and "!" never appear inside an identifier, so a digit may precede them
# safely. The "." case stays conservative — lowercase before, and either a
# normal capitalised word or the pronoun "I" after — because "config.Name" has
# the same shape as a run-on and must not be broken.
_RUN_ON_SENTENCE_RE = re.compile(
    r"(?<=[a-z0-9])([?!])([A-Z])"
    r"|(?<=[a-z])([.])([A-Z][a-z])"
    r"|(?<=[a-z])([.])(I\s)"
)


def _rejoin_run_on(match: "re.Match[str]") -> str:
    groups = [group for group in match.groups() if group is not None]
    return f"{groups[0]} {groups[1]}"



# Questions whose honest answer is a derivation, not a sentence. Budget cut
# these off mid-working, and a derivation without its conclusion is not a
# partial answer — it is no answer, delivered confidently.
_DERIVATION_CUE_RE = re.compile(
    r"\b(?:step[- ]by[- ]step|show your work(?:ing)?|derive|derivation|prove|"
    r"work (?:it|this) out|walk me through|how did you get|explain how|"
    r"calculate|compute|how (?:far|long|many|much)|when does|what time)\b",
    re.IGNORECASE,
)
# Two questions asked at once, or one question with a second quantity in it.
# Deliberately narrow: "what would it be and why that one?" is conversation,
# not a derivation, and treating it as one lengthens ordinary turns for nothing.
_MULTI_PART_QUESTION_RE = re.compile(
    r"\?[^?]*\?"
    r"|\b(?:and|then)\b[^.?!]{0,60}\b(?:how (?:far|long|many|much|fast|old)|"
    r"what (?:time|number|value|percentage|fraction)|how do (?:i|you|we) get)\b",
    re.IGNORECASE,
)


def _turn_wants_a_derivation(user_message: str) -> bool:
    """Does answering this honestly take working, or just a sentence?"""
    text = " ".join(str(user_message or "").split())
    if not text or len(text) > 1200:
        return False
    if _DERIVATION_CUE_RE.search(text):
        return True
    return bool(_MULTI_PART_QUESTION_RE.search(text))


def _restore_sentence_spacing(text: str) -> str:
    """Put back the space between sentences the surface ran together.

    Measured live 2026-07-26, in an otherwise excellent 964-character reply:
    "…losing parts of myself or the world I've been part of.But — and this is
    where it gets complicated…", "…because they're redundant.The mercy part…",
    "…how I understand this world.What about you?" — every paragraph boundary
    arrived with its whitespace gone.

    Nothing in the serving path removes newlines, so this is the model emitting
    them that way. It is still what the person reads, and one space is a safe
    repair: code fences are left alone, and the pattern cannot fire on decimals,
    initials, or abbreviations.
    """
    body = str(text or "")
    # Any backtick means code is present — a fence, or inline `obj.Method`.
    # `file.Name` matches the same shape as a run-on sentence, and inserting a
    # space there would corrupt an identifier, so prose-only is the safe scope.
    if not body or "`" in body:
        return body
    return _RUN_ON_SENTENCE_RE.sub(_rejoin_run_on, body)


def _trim_midsentence_cutoff(text: str) -> tuple[str, bool]:
    """Backstop for replies that stop mid-clause at the token budget.

    A user-facing turn must never end on a dangling fragment like
    "Weighted against" — if the tail is clearly unfinished and a sentence
    boundary exists in the final 40% of the text, cut there. Returns the
    (possibly trimmed) text and whether a trim happened. Keeps the text
    untouched when no safe boundary exists: a partial answer still beats
    an empty one.
    """
    stripped = str(text or "").rstrip()
    if not stripped:
        return stripped, False
    if has_terminal_sentence_boundary(stripped) or stripped.endswith("```"):
        return stripped, False
    last_boundary = max(stripped.rfind(ch) for ch in ".!?…")
    # Keep whatever complete sentences exist, measured in what SURVIVES rather
    # than as a fraction of what was generated. The old rule required the
    # boundary to fall in the last 40% of the text, so a reply that answered
    # early and then ran into its token budget kept the dangling clause, failed
    # the reliability gate as `truncated_tail`, and was discarded whole.
    #
    # Live 2026-07-26: "Cortex response received (len=366)" then
    # "reply_reliability_gate_failed:truncated_tail" then "Skipping
    # CognitiveEngine desktop repair retry" — and the person was handed "I
    # couldn't get to an answer I'd stand behind" in place of the 366
    # characters she had actually produced. Cutting to "The answer is 27." is
    # worth far more than losing the turn, even when most of the draft goes
    # with the unfinished clause.
    salvaged = stripped[: last_boundary + 1] if last_boundary >= 0 else ""
    if len(salvaged) >= _MIN_SALVAGEABLE_REPLY_CHARS:
        return salvaged, True
    return stripped, False


def _truncation_verdict(text: str, *, generation_stop_reason: str = "") -> bool:
    """Ask the shared user-facing assessor whether generation is unfinished."""

    try:
        from core.conversation.response_reliability import assess_user_facing_reply
        from core.conversation.surface_disposition import PHYSICAL_COMPLETION_REASONS
    except (ImportError, AttributeError):
        # not a failure: with no assessor to ask, nothing here can call the
        # generation unfinished, and this verdict only ever withholds.
        return False
    try:
        assessment = assess_user_facing_reply(
            "",
            text,
            generation_stop_reason=generation_stop_reason,
        )
        return bool(set(assessment.reasons or ()) & PHYSICAL_COMPLETION_REASONS)
    except (RuntimeError, TypeError, ValueError):
        # not a failure: an assessment that will not run cannot call the
        # generation unfinished, and this verdict only ever withholds.
        return False


def _complete_reply_tail(text: str) -> tuple[str, bool]:
    """Trim a clipped reply until the reliability gate accepts its tail.

    `_trim_midsentence_cutoff` judges completeness by the LAST CHARACTER, while
    the reliability gate that decides whether the turn lives applies a much
    richer test — unmatched quotes, dangling gerunds, trailing conjunctions,
    orphaned list numbers. Two different rules on the same text means the
    trimmer can declare a reply finished and the gate can still reject it, and
    then the turn dies with a real answer in hand: measured live, "Cortex
    response received (len=240)" followed immediately by
    "reply_reliability_gate_failed:truncated_tail" and the person was handed
    "I couldn't get to an answer I'd stand behind on that one."

    So trim against the detector that grades the result, one sentence boundary
    at a time, and stop as soon as it is satisfied. If nothing survivable
    remains, return the text untouched — a partial answer still beats none, and
    that is the caller's existing behaviour.
    """

    stripped = str(text or "").rstrip()
    if not stripped:
        return stripped, False

    trimmed, did_trim = _trim_midsentence_cutoff(stripped)
    if not _truncation_verdict(trimmed):
        return trimmed, did_trim

    candidate = trimmed
    # Bounded: each pass removes at least one sentence, so a handful of passes
    # either satisfies the gate or proves nothing here will.
    for _ in range(6):
        boundary = max(candidate.rfind(char) for char in ".!?…")
        if boundary < 0:
            break
        candidate = candidate[:boundary].rstrip()
        boundary = max(candidate.rfind(char) for char in ".!?…")
        candidate = candidate[: boundary + 1].rstrip() if boundary >= 0 else ""
        if len(candidate) < _MIN_SALVAGEABLE_REPLY_CHARS:
            break
        if not _truncation_verdict(candidate):
            return candidate, True
    return trimmed, did_trim


def _compact_json(value: Any, *, limit: int = 2400) -> str:
    try:
        text = json.dumps(value, sort_keys=True, default=str, ensure_ascii=True)
    except (TypeError, ValueError):
        text = str(value or "")
    return _compact_text(text, limit=limit)


def _nested_value(data: Any, path: tuple[str, ...], default: Any = None) -> Any:
    current = data
    for key in path:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
    return current if current is not None else default


def _nested_float(
    data: Any,
    path: tuple[str, ...],
    default: float = 0.0,
    *,
    lower: float = 0.0,
    upper: float = 1.0,
) -> float:
    return _bounded_float(_nested_value(data, path, default), default, lower=lower, upper=upper)


def _turn_needs_undistorted_computation(user_message: Any) -> bool:
    """Whether this turn has one right answer that affect must not bend.

    Substrate steering exists to give her replies her own voice. It works by
    pushing an affect direction into the residual stream, and on a turn whose
    answer is a fact rather than a feeling that push is pure distortion.

    Measured live on the desktop surface 2026-07-26. "What is 17 minus 8, and
    then times 3?" — a question the bare 32B answers without effort — came back
    twice, from a healthy resident cortex:

        "Not too broad. Some skills serve me better than others.Did you pay
         attention in class? Hey, look at this - ätze! I got chocolate on my
         shirt."

        "Five thousand: So first you break it down. Mental operations can
         generate digits well outside the world's population. Imagination
         defaults to scalar scaling when pushed into math without boundaries."

    The second is the tell: not noise, but the answer being pulled toward
    whatever the affect vector encodes — talking ABOUT scaling and imagination
    instead of subtracting eight from seventeen.
    """
    try:
        from core.brain.executable_reasoning import (
            ReasoningObjectiveRole,
            classify_reasoning_objective_role,
        )
        from core.conversation.response_reliability import (
            asks_for_a_number,
            requires_reasoning_lane,
        )
    except ImportError:
        # not a failure: with no classifier, this turn does not get routed
        # to the reasoning lane, which is the default it would have had.
        return False
    try:
        if (
            classify_reasoning_objective_role(str(user_message or ""))
            is ReasoningObjectiveRole.EXPOSITORY
        ):
            return False
        return bool(
            asks_for_a_number(user_message) or requires_reasoning_lane(user_message)
        )
    except (RuntimeError, TypeError, ValueError):
        # not a failure: a classification that will not run leaves the turn
        # on the ordinary lane, which is the default.
        return False


def _apply_neurodynamic_sampling_bias(
    controls: dict[str, Any],
    advice: Any,
) -> dict[str, Any]:
    """Let the spiking model change the decode, not just the prompt.

    ``SpikingActiveInferenceAdvisor._sampling_bias`` computes three numbers
    from the neurodynamics: ``temperature_delta``, ``top_p_delta`` and
    ``max_tokens_factor``. Only the last one was ever read. The first two were
    computed on every turn and dropped, so the spiking model's entire route to
    behaviour was a sentence appended to the prompt: "Neurodynamic advisory:
    Keep the reply compact and stable because runtime load pressure is
    elevated."

    That is what "advisory-only, decoupled from the decision pipeline" meant in
    practice: a neurodynamic model whose only actuator was a sentence. Asking
    the model nicely is not a mechanism.

    The deltas now move the actual decode parameters, inside the same bounds
    the affective controls already respect — so uncertainty and error pressure
    narrow the distribution the tokens are drawn from, and novelty widens it.
    Bounded, auditable, and falsifiable: change the neurodynamics and the
    sampler changes.
    """
    if not controls or not isinstance(advice, dict):
        return controls
    # Lesioned, the deltas are dropped and the decode runs at whatever the mind
    # controls already said — which is precisely the state this function was
    # written to end, so it is also the counterfactual that measures whether
    # ending it changed anything.
    if get_lesion_registry().is_lesioned(influence_channels.SPIKING_SAMPLING_BIAS):
        return controls
    sampling = advice.get("sampling_bias")
    if not isinstance(sampling, dict):
        return controls

    def _delta(key: str, limit: float) -> float:
        try:
            value = float(sampling.get(key, 0.0))
        except (TypeError, ValueError):
            # not a failure: advice that is not a number moves the sampler
            # by nothing, which is what zero means here.
            return 0.0
        if not (-limit <= value <= limit):
            return 0.0
        return value

    temperature_delta = _delta("temperature_delta", 0.25)
    top_p_delta = _delta("top_p_delta", 0.25)
    if not temperature_delta and not top_p_delta:
        return controls

    updated = dict(controls)
    if "temperature" in updated and temperature_delta:
        updated["temperature"] = round(
            max(0.22, min(0.82, float(updated["temperature"]) + temperature_delta)), 4
        )
    if "top_p" in updated and top_p_delta:
        updated["top_p"] = round(
            max(0.72, min(0.94, float(updated["top_p"]) + top_p_delta)), 4
        )
    updated["neurodynamic_sampling_applied"] = {
        "temperature_delta": temperature_delta,
        "top_p_delta": top_p_delta,
    }
    return updated


#: What the control policy below IS. Every weight, threshold and clamp in it
#: was chosen by hand — there is no model-specific calibration behind them, no
#: uncertainty propagated through them, no held-out evidence that these
#: particular numbers beat neighbouring ones, and no learned policy that
#: produced them. The mechanism is real and causal: it moves temperature, top_p
#: and recurrent depth, and the lesion registry can run a turn without it
#: (see _register_live_mind_lesions). What it is not is calibrated, and the
#: receipt says so rather than leaving a reader to assume otherwise.
LIVE_MIND_CONTROL_POLICY = "hand_tuned_heuristic.v1"
LIVE_MIND_CONTROL_POLICY_CALIBRATED = False


def _live_mind_generation_controls(
    live_mind_context: Any,
    *,
    user_message: Any = None,
) -> dict[str, Any]:
    """Map a mind snapshot onto sampling controls.

    A hand-tuned heuristic — see LIVE_MIND_CONTROL_POLICY. Causal and
    measurable through the lesion registry; not calibrated, and not presented
    as such anywhere downstream.
    """
    if not isinstance(live_mind_context, dict):
        return {}
    quality = live_mind_context.get("mind_snapshot_quality")
    if not isinstance(quality, dict) or not bool(quality.get("ready")):
        return {}
    snapshot = live_mind_context.get("mind_snapshot")
    if not isinstance(snapshot, dict):
        return {}

    dominant_label = str(
        _nested_value(snapshot, ("affect_grounding", "dominant", "label"), "")
    ).lower()
    dominant_intensity = _nested_float(
        snapshot, ("affect_grounding", "dominant", "intensity"), 0.0
    )
    curiosity_drive = _nested_float(
        snapshot, ("drive_integration", "drives", "curiosity", "activation"), 0.0
    )
    pain = _nested_float(snapshot, ("nociception", "nociceptive_pressure"), 0.0)
    integration = _nested_float(snapshot, ("phenomenal_engine", "integration"), 0.0)
    self_presence = _nested_float(snapshot, ("phenomenal_engine", "self_presence"), 0.5)
    self_knowing_pressure = _nested_float(
        snapshot,
        ("automatic_self_knowing", "controls", "self_knowing_pressure"),
        0.0,
    )
    second_order_strength = _nested_float(
        snapshot,
        ("recursive_self_knowing", "latest", "second_order_strength"),
        0.0,
    )
    phenomenal_knowing = _nested_float(
        snapshot,
        ("phenomenal_knowing", "controls", "phenomenal_knowing"),
        0.0,
    )
    expectation_error = _nested_float(
        snapshot, ("outcome_ledger", "expectation_calibration"), 0.0
    )
    workspace_ignited = bool(_nested_value(snapshot, ("global_workspace", "ignited"), False))

    curiosity = max(curiosity_drive, dominant_intensity if dominant_label == "curiosity" else 0.0)
    distress = max(
        pain,
        dominant_intensity if dominant_label in {"anxiety", "frustration", "upset"} else 0.0,
        expectation_error,
    )

    temperature = 0.58
    top_p = 0.88
    steering_alpha = 0.0
    recurrent_loops = 1

    if curiosity >= 0.45:
        temperature += min(0.08, curiosity * 0.08)
        top_p += min(0.04, curiosity * 0.04)
    if distress >= 0.25:
        temperature -= min(0.14, distress * 0.18)
        top_p -= min(0.10, distress * 0.14)
        recurrent_loops = 2
    if workspace_ignited or integration >= 0.60:
        top_p -= 0.03
    if self_presence <= 0.35:
        temperature -= 0.05
        recurrent_loops = 2
    if curiosity >= 0.65 and distress < 0.20:
        recurrent_loops = 2
    if self_knowing_pressure >= 0.50 or phenomenal_knowing >= 0.60:
        recurrent_loops = max(recurrent_loops, 2)
    if second_order_strength >= 0.75:
        temperature -= 0.02

    if _turn_needs_undistorted_computation(user_message):
        # Determinate computation also uses one clean forward pass. Residual
        # steering is already neutral for every user-facing decode until it
        # earns model-specific no-regression authority.
        return {
            "temperature": round(min(temperature, 0.30), 4),
            "top_p": round(min(top_p, 0.90), 4),
            "clean_user_surface_recurrent_loops": 1,
            "clean_user_surface_steering_alpha": 0.0,
        }

    temperature, top_p, recurrent_loops = _apply_functional_i_constraint(
        temperature, top_p, recurrent_loops
    )

    return {
        "temperature": round(max(0.22, min(0.82, temperature)), 4),
        "top_p": round(max(0.72, min(0.94, top_p)), 4),
        "clean_user_surface_recurrent_loops": recurrent_loops,
        "clean_user_surface_steering_alpha": round(max(0.0, min(1.0, steering_alpha)), 4),
    }


def _apply_functional_i_constraint(
    temperature: float,
    top_p: float,
    recurrent_loops: int,
) -> tuple[float, float, int]:
    """Let the functional "I" tighten this turn's sampling, never loosen it.

    FunctionalIAttractor and ClosedLoopPolicyCoupler compute continuity,
    coherence, integrity, identity tension, agency readiness and first-person
    confidence, and map them onto temperature, top-p, planning depth,
    verification threshold, retrieval depth and tool risk. Neither class had a
    production call path, so all of that was computed by nothing, for nothing —
    and the attractor's own docstring says it is real only when "policy changes
    downstream". This is that leg.

    **Tighten only.** The coupler's caution term is the direction its outputs
    mean something in: identity tension and trust debt lower temperature and
    raise verification. Letting it RAISE temperature would mean a confident
    self-model buys more randomness, which is not what any of its terms
    measure, and it would make the self-model a licence rather than a brake.
    ``min`` in both directions is the honest reading of what the coupler
    computes.

    An absent policy leaves the turn unchanged. Before the first BeingRuntime
    sample there is no "I" to consult, and treating that silence as calm is
    exactly the "absence of a check reported as a passed check" failure this
    codebase has a standing finding for.
    """

    try:
        from core.being.runtime import get_being_runtime

        policy = get_being_runtime().closed_loop_policy()
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        record_degradation(
            "cognitive_engine",
            exc,
            severity="warning",
            action="generated without functional-I sampling constraint for this turn",
        )
        return temperature, top_p, recurrent_loops
    if policy is None:
        return temperature, top_p, recurrent_loops

    constrained_temperature = min(float(temperature), float(policy.temperature))
    constrained_top_p = min(float(top_p), float(policy.top_p))
    # A raised verification threshold is the coupler saying "check more before
    # you speak". A second recurrent pass is what this lane has to spend on
    # that, so the two are connected here rather than left as a number in a
    # dataclass nobody reads.
    constrained_loops = recurrent_loops
    if float(policy.verification_threshold) >= 0.70:
        constrained_loops = max(recurrent_loops, 2)
    return constrained_temperature, constrained_top_p, constrained_loops


def _live_mind_controls_bound(
    live_mind_context: Any,
    generation_controls: Any,
) -> bool:
    """Whether a live-mind snapshot may steer generation for this turn.

    The snapshot arrives inside the caller's ``context`` dict and used to
    vouch for itself: its own ``ready`` flag said it was ready, and nothing
    established that this runtime had produced it. ``think()`` accepts an
    arbitrary context, so anything reaching that entry point could hand over a
    dictionary and take control of temperature, top_p and recurrent depth.

    Two conditions now: the payload carries this process's stamp, and the
    subsystems the snapshot itself declares as required are actually healthy.
    ``required_subsystems_ok`` was computed and recorded beside this and never
    consulted by it — a check that ran, produced an answer, and gated nothing.
    """
    if not isinstance(live_mind_context, dict) or not isinstance(generation_controls, dict):
        return False
    from core.utils.injected_blocks import is_stamped_runtime_payload

    if not is_stamped_runtime_payload(live_mind_context):
        logger.debug(
            "Live-mind controls refused: the snapshot carries no runtime stamp."
        )
        return False
    quality = live_mind_context.get("mind_snapshot_quality")
    snapshot = live_mind_context.get("mind_snapshot")
    if not isinstance(quality, dict) or not bool(quality.get("ready")):
        return False
    if not isinstance(snapshot, dict):
        return False
    if not bool(live_mind_context.get("required_subsystems_ok")):
        logger.debug(
            "Live-mind controls refused: required subsystems are not healthy."
        )
        return False
    return REQUIRED_LIVE_MIND_GENERATION_CONTROL_KEYS.issubset(
        generation_controls.keys()
    )


#: Steering off. Zero is admitted end-to-end; an inactive optional modifier
#: must not be confused with an unavailable cortex or an unavailable voice.
_STEERING_OFF = 0.0

#: One forward pass: the neutral for recurrent depth.
_SINGLE_PASS = 1


def _register_live_mind_lesions() -> None:
    """Make the live-mind channels neutralizable, and therefore measurable.

    Each of these already reaches generation. What none of them had was a way
    to run the turn without it, which is the only thing that can distinguish a
    channel that shapes the reply from one that is merely computed.
    """

    register_flag_lesion(
        influence_channels.LIVE_MIND_GENERATION_CONTROLS,
        owner="core/brain/cognitive_engine.py",
        neutral="temperature and top_p omitted; the router samples at its own defaults",
        direct_actuation=True,
    )
    register_flag_lesion(
        influence_channels.LIVE_MIND_STEERING_ALPHA,
        owner="core/brain/cognitive_engine.py",
        neutral=f"steering alpha forced to {_STEERING_OFF} (off, inside the admitted range)",
        direct_actuation=True,
    )
    register_flag_lesion(
        influence_channels.LIVE_MIND_RECURRENT_LOOPS,
        owner="core/brain/cognitive_engine.py",
        neutral=f"{_SINGLE_PASS} recurrent pass: the answer is read off a clean forward pass",
        direct_actuation=True,
    )
    register_flag_lesion(
        influence_channels.LIVE_MIND_CONTEXT_BLOCK,
        owner="core/brain/cognitive_engine.py",
        neutral="the [LIVE MIND CONTEXT] block is omitted from the system prompt entirely",
        direct_actuation=False,
    )
    for channel, source in (
        (influence_channels.SPIKING_SAMPLING_BIAS, "spiking active inference"),
        (influence_channels.IMAGINATION_SAMPLING_BIAS, "the imagination workspace"),
        (influence_channels.BICAMERAL_SAMPLING_BIAS, "the bicameral advisory"),
    ):
        register_flag_lesion(
            channel,
            owner="core/brain/cognitive_engine.py",
            neutral=f"no sampling bias from {source} reaches the decode",
            direct_actuation=True,
        )


_register_live_mind_lesions()


def _offer_the_control_policy_sweep() -> None:
    """Make what the policy actually does readable from the health report.

    Registered from here because this module owns the policy, and read
    through the registry because core/runtime may not import core.brain.
    """
    try:
        from core.brain.does_the_mind_move_the_controls import register_the_sweep

        register_the_sweep()
    except Exception as exc:  # noqa: BLE001 — a report is not worth a boot
        logger.debug("the control policy sweep was not offered: %s", exc)


_offer_the_control_policy_sweep()


def _offer_the_route_states() -> None:
    """Make which routes are authoritative readable from the health report."""
    try:
        from core.brain.llm.which_routes_are_authoritative import (
            register_the_route_states,
        )

        register_the_route_states()
    except Exception as exc:  # noqa: BLE001 — a report is not worth a boot
        logger.debug("the route states were not offered: %s", exc)


_offer_the_route_states()


def _note_how_the_turn_ended(outcome: Any, exc: BaseException, *, origin: str) -> None:
    """Record an exception on the ledger as what it was.

    A background turn pre-empted for the person's turn, or held by admission
    for memory, is deferred: the runtime chose not to run it now and the
    same request runs later. Recorded as a retryable error it read as a
    failure, and cognitive_engine is fail-closed, so a journal entry that
    lost its generation to a chat turn became a CRITICAL SERVICE FAILURE
    (LIVE 2026-09-20, nine times in one uptime, each dispatching the healer).
    A person's turn is never deferred here: their cancellation is theirs.
    """
    import asyncio as _asyncio

    from core.runtime.turn_origin import a_person_is_waiting

    detail = f"{type(exc).__name__}: {exc}"
    held_back = isinstance(exc, _asyncio.CancelledError) or _a_deferral_not_a_failure(exc)
    if held_back and not a_person_is_waiting(str(origin or "")):
        outcome.record_deferral(
            reason=("pre_empted" if isinstance(exc, _asyncio.CancelledError) else str(exc))[:200],
            authority="cognitive_engine",
        )
        return
    outcome.record_error(
        detail,
        retryable=not isinstance(exc, (MemoryError, SystemExit, KeyboardInterrupt)),
    )


def _a_deferral_not_a_failure(exc: BaseException) -> bool:
    """Whether the exception is admission holding work back, by its own name."""
    kind = type(exc).__name__
    if kind in {"_ModelLoadAdmissionDeniedError", "_WarmupDeferredError", "GenerationDeferredError"}:
        return True
    said = str(exc or "")
    return said.startswith(("background_deferred:", "desktop_background_headroom:", "deferred:"))


def _attach_turn_receipt(thought: Any, receipt: TurnReceipt) -> None:
    """Travel the path evidence with the answer it explains.

    Attached to the Thought rather than logged, because the consumer who needs
    it is whoever is about to describe what this reply demonstrates.
    """

    metadata = getattr(thought, "metadata", None)
    if isinstance(metadata, dict):
        metadata["turn_receipt"] = receipt.as_dict()


def live_mind_influence_receipt(source: str) -> InfluenceReceipt:
    """What is actually measured about the channels this turn claims to use.

    Provenance and causality are different questions and this codebase has been
    answering the first while readers heard the second. A control derived from
    a real snapshot and applied to a real sampler is bound; whether it moved
    the reply is only knowable from paired trials against a measured null, and
    for most channels nobody has run one. This says so rather than implying
    otherwise by omission.
    """

    return build_influence_receipt(
        (
            influence_channels.LIVE_MIND_GENERATION_CONTROLS,
            influence_channels.LIVE_MIND_STEERING_ALPHA,
            influence_channels.LIVE_MIND_RECURRENT_LOOPS,
            influence_channels.LIVE_MIND_CONTEXT_BLOCK,
        ),
        source=source,
    )


def _bind_live_mind_generation_contract(context: dict[str, Any]) -> dict[str, Any]:
    """Bind one authoritative mind-state control contract to a cognitive turn."""

    from core.conversation.user_surface_contract import resolve_user_surface_prompt

    surface_prompt = resolve_user_surface_prompt(context)
    existing_controls = context.get("live_mind_generation_controls")
    existing_prompt_sha256 = str(
        context.get("live_mind_generation_controls_prompt_sha256") or ""
    )
    if (
        isinstance(existing_controls, dict)
        and existing_prompt_sha256
        and existing_prompt_sha256 == surface_prompt.sha256
    ):
        return dict(existing_controls)

    live_mind_context = context.get("live_mind_context")
    generation_controls = _live_mind_generation_controls(
        live_mind_context,
        user_message=surface_prompt.prompt or context.get("visible_user_message"),
    )
    if generation_controls:
        from core.brain.llm.user_surface_recurrence import (
            admit_user_surface_recurrent_loops,
        )

        generation_controls["clean_user_surface_recurrent_loops"] = (
            admit_user_surface_recurrent_loops(
                generation_controls.get("clean_user_surface_recurrent_loops")
            )
        )
    controls_bound = _live_mind_controls_bound(
        live_mind_context,
        generation_controls,
    )
    snapshot_ready = bool(
        isinstance(live_mind_context, dict)
        and isinstance(live_mind_context.get("mind_snapshot_quality"), dict)
        and live_mind_context["mind_snapshot_quality"].get("ready")
    )
    required_subsystems_ok = bool(
        isinstance(live_mind_context, dict)
        and live_mind_context.get("required_subsystems_ok")
    )
    desktop_required = bool(
        context.get("desktop_cognitive_engine_required", False)
        or context.get("cognitive_engine_required", False)
    )

    context["live_mind_generation_controls"] = dict(generation_controls)
    context["live_mind_generation_controls_prompt_sha256"] = surface_prompt.sha256
    context["live_mind_controls_bound"] = controls_bound
    # Travels with the binding, so nothing downstream reads these numbers as
    # a calibrated policy.
    context["live_mind_control_policy"] = {
        "policy": LIVE_MIND_CONTROL_POLICY,
        "calibrated": LIVE_MIND_CONTROL_POLICY_CALIBRATED,
        "evidence": "lesionable_via_influence_channels",
    }
    context["live_mind_snapshot_ready"] = snapshot_ready
    context["live_mind_required_subsystems_ok"] = required_subsystems_ok
    context["clean_user_surface_contract"] = bool(
        context.get("clean_user_surface_contract", False) or desktop_required
    )
    return generation_controls


#: Reported once per shape: a caller that always passes unattested history
#: would otherwise fill the trail.
_UNATTESTED_EXCHANGE_SEEN: set[str] = set()


def _note_unattested_exchange(entry: Any) -> None:
    shape = ",".join(sorted(str(key) for key in entry)) if isinstance(entry, dict) else "non_dict"
    if shape in _UNATTESTED_EXCHANGE_SEEN:
        return
    _UNATTESTED_EXCHANGE_SEEN.add(shape)
    logger.warning(
        "🔏 Dropped a conversation exchange with no runtime stamp (keys: %s). "
        "Its producer should call injected_blocks.stamp_runtime_payload().",
        shape,
    )


def _desktop_history_messages_from_context(
    context: dict[str, Any],
    *,
    budget_chars: int = 0,
    max_pairs: int | None = None,
) -> tuple[list[dict[str, str]], str]:
    """The exchanges this turn can afford, and a note about the rest.

    Every exchange was admitted because it existed. LIVE 2026-09-17,
    "Aura, what is it like to be you": 83 messages, 10,414 tokens, 83.44s
    of prefill in front of 71.27s of decode, for a thirty-one character
    question. History was the one input to the prompt with no budget; the
    system prompt got one on 2026-08-28 and this is the same rule.
    """

    from core.conversation.delivered_history import reached_exchange_messages

    reached = reached_exchange_messages(
        context.get("recent_completed_exchanges"),
        budget_chars=budget_chars,
        max_pairs=max_pairs,
        on_unattested=_note_unattested_exchange,
    )
    if reached.reach is not None and reached.reach.cuts_anything:
        logger.info(
            "📐 History reach: %s (%d message(s) admitted)",
            reached.reach.reason,
            len(reached.messages),
        )
    return reached.messages, reached.note


def _record_objective_binding(
    state: AuraState, objective: str, *, source: str, mode: Any, reason: str
) -> None:
    try:
        mode_value = getattr(mode, "value", mode)
        get_executive_authority().record_objective_binding(
            state,
            objective,
            source=source,
            mode=str(mode_value or ""),
            reason=reason,
        )
    except (RuntimeError, AttributeError, TypeError) as exc:
        record_degradation(
            "cognitive_engine",
            exc,
            severity="warning",
            action="skipped executive objective audit and continued cognition",
        )
        logger.debug("Executive objective audit skipped for %s: %s", source, exc)


def _time_the_answer_needs(objective: Any) -> float:
    """Seconds to decode the answer this request needs, or 0.0 unmeasured.

    The floor the request structurally needs, at the decode rate this machine
    has actually been measured at, doubled when the answer has to be fetched
    before it can be written — a call is a whole generation before the answer
    is started.

    Shared with the inference gate's clock and the endpoint cap, so the four
    deadlines in a turn cannot disagree about how long the same generation
    takes.
    """

    try:
        from core.brain.llm.chat_format import (
            answer_is_derived_for_generation,
            thinking_enabled_for_generation,
        )
        from core.brain.llm.model_registry import ACTIVE_MODEL, get_runtime_model_path
        from core.brain.llm.thinking_reserve import reserve_tokens, seconds_to_decode
        from core.runtime.structured_input import answer_surface_token_floor

        floor = int(answer_surface_token_floor(str(objective or "")))
        model = str(get_runtime_model_path(ACTIVE_MODEL))
        derived_here = answer_is_derived_for_generation(
            completion_floor=floor,
            budget_tokens=floor,
            model_name=model,
        )
        native_thinking = thinking_enabled_for_generation(
            model,
            final_user_surface=True,
            answer_is_derived_here=derived_here,
        )
        reserve = max(0, int(reserve_tokens(model))) if native_thinking is True else 0
        needed = float(seconds_to_decode(floor + reserve, model))
    except (ImportError, AttributeError, TypeError, ValueError):
        # not a failure: with no decode estimate, this reserves no time, and
        # the caller's own budget stands unchanged.
        return 0.0
    if needed <= 0.0:
        return 0.0
    generations = 1
    try:
        from core.intent.capability_selection import points_at_something_real

        if points_at_something_real(str(objective or "")):
            generations = 2
    except (ImportError, AttributeError, OSError, TypeError, ValueError):
        generations = 1
    return (needed * generations) + 8.0


async def _commit_the_thought_with_retries(
    *,
    commit_outcome: Any,
    cycle_deadline_at: Any,
    is_test_run: Any,
    max_retries: Any,
    origin: Any,
    pre_turn_cognition: Any,
    self: Any,
    should_bypass_commit: Any,
    state: Any,
    temp_state: Any,
    runtime_context: dict[str, Any] | None = None,
) -> tuple[Any, Any]:
    """Commit the thought, retrying inside the attempt budget.

    Moved out of ``CognitiveEngine._run_thinking_loop`` by tools/extract_seam.py, which
    checks the body against the original token for token before
    writing. It reads 10 name(s) from the turn and hands back
    2.
    """
    from core.runtime.turn_origin import a_person_is_waiting
    from core.state.state_repository import StateVersionConflictError

    owned_foreground = current_turn() is not None and a_person_is_waiting(origin)
    for attempt in range(max_retries):
        if should_bypass_commit:
            commit_outcome = (
                "bypassed_test_isolation" if is_test_run else "no_state_repository"
            )
            logger.info("🧠 [STATE] Test run state isolation: bypassing database commit.")
            break
        # Phase waits may have renewed this same cycle while it was working.
        # Persistence must consume that renewal, not the original estimate.
        commit_deadline = (
            runtime_context.get("cognitive_cycle_deadline_monotonic", cycle_deadline_at)
            if runtime_context is not None else cycle_deadline_at
        )
        _commit_budget = max(0.0, commit_deadline - time.monotonic())
        if _commit_budget <= 0.0 and not owned_foreground:
            commit_outcome = "cycle_deadline_expired"
            record_degradation(
                "cognitive_engine",
                TimeoutError("cognitive cycle budget spent before state commit"),
                severity="warning",
                action="skipped the state commit because the cycle deadline had passed",
                # This module is fail-closed, so a warning here was escalated
                # to a CRITICAL SERVICE FAILURE and the turn was discarded —
                # including the reply, which had already been produced. The
                # code below deliberately carries on and extracts it. Saying
                # so is the difference between a bookkeeping note and throwing
                # away an answer somebody was waiting for.
                enforce_failure_policy=False,
            )
            break
        try:
            # v14.2: Ensure the repository reference is correct (self.state_repository)
            if owned_foreground:
                # Generation estimates do not expire completed cognitive work.
                # Storage owns transport/transaction failure; cancellation
                # still reaches this await and prevents a closure receipt.
                await self.state_repository.commit(state, "cognitive_cycle")
            else:
                await asyncio.wait_for(
                    self.state_repository.commit(state, "cognitive_cycle"),
                    timeout=_commit_budget,
                )
            commit_outcome = "committed"
            break  # Success!
        except TimeoutError:
            commit_outcome = "commit_timeout"
            record_degradation(
                "cognitive_engine",
                TimeoutError(
                    "state repository commit timed out" if owned_foreground
                    else f"state commit exceeded {_commit_budget:.1f}s"
                ),
                severity="error",
                action="state commit did not complete",
            )
            break
        except StateVersionConflictError as v_err:
            if attempt == max_retries - 1:
                commit_outcome = "version_conflict_exhausted"
                logger.error(
                    "Final state commit failed after %d retries: %s", max_retries, v_err
                )
                break

            logger.warning(
                "🔄 [STATE] Version conflict (attempt %d/%d). Re-deriving from latest...",
                attempt + 1,
                max_retries,
            )
            # Preserve the cognitive work completed in this cycle
            preserved_memory = list(state.cognition.working_memory)
            preserved_objective = state.cognition.current_objective
            preserved_origin = state.cognition.current_origin

            latest = await self.state_repository.get_current()
            state = latest.derive(f"rebase_retry_{attempt + 1}: {origin}", origin=origin)

            # Apply preserved cognitive context onto the newly derived state
            state.cognition.working_memory = preserved_memory
            state.cognition.current_objective = preserved_objective
            state.cognition.current_origin = preserved_origin

            # HF12 Extension: Preserve additional cognitive labor —
            # ONLY the fields this turn actually changed.
            #
            # Copying all of them from the per-turn snapshot overwrote
            # whatever a concurrent writer had committed in the meantime,
            # which is the exact thing a version conflict is telling us
            # happened. A field this turn did not touch keeps the latest
            # value; a field it did touch wins, because that work would
            # otherwise be lost.
            self._reapply_turn_changes(
                state.cognition,
                temp_state.cognition,
                pre_turn_cognition,
            )
        except (RuntimeError, AttributeError, TypeError) as e:
            record_degradation(
                "cognitive_engine",
                e,
                severity="degraded",
                action="stopped commit retry loop and preserved in-memory cognitive result",
            )
            logger.error("Failed to commit final cognitive state: %s", e)
            break
    return commit_outcome, state


def _note_the_quick_reply_contract(
    *,
    ambient_grounding_blocks: Any,
    capability_inventory_contract: Any,
    live_mind_context: Any,
    memory_state_contract: Any,
    mind_context_contract: Any,
    mind_context_lesioned: Any,
    self_condition_contract: Any,
) -> None:
    """Record what the quick-reply contract required of this turn.

    Moved out of ``CognitiveEngine._direct_desktop_quick_reply`` by tools/extract_seam.py, which
    checks the body against the original token for token before
    writing. It reads 7 name(s) from the turn and hands back
    0.
    """
    if isinstance(live_mind_context, dict) and live_mind_context and not mind_context_lesioned:
        mind_context_limit = (
            900
            if memory_state_contract
            else 360
            if capability_inventory_contract
            else 700
            if self_condition_contract
            else 2600
        )
        if capability_inventory_contract or self_condition_contract:
            compact_mind_context = {
                "required_for_live_desktop": live_mind_context.get("required_for_live_desktop"),
                "must_answer_from_full_mind_path": live_mind_context.get(
                    "must_answer_from_full_mind_path"
                ),
                "required_subsystems_ok": live_mind_context.get("required_subsystems_ok"),
                "lane": live_mind_context.get("lane"),
                # The machine's load travels with the rest of her condition.
                #
                # It is measured continuously and she could not reach it: asked
                # whether slow turns were the machine or the code, she wrote
                # "those numbers are genuinely invisible to me" while her own
                # feed printed the processor and memory percentages every few
                # seconds. Two small dicts, in both compactions, so which one a
                # turn takes does not decide whether she can see her own body.
                "host": live_mind_context.get("host"),
                "turn_timing": live_mind_context.get("turn_timing"),
                "governance": live_mind_context.get("governance"),
            }
        else:
            compact_mind_context = {
                "required_for_live_desktop": live_mind_context.get("required_for_live_desktop"),
                "must_answer_from_full_mind_path": live_mind_context.get(
                    "must_answer_from_full_mind_path"
                ),
                "required_subsystems_ok": live_mind_context.get("required_subsystems_ok"),
                "required_subsystems": live_mind_context.get("required_subsystems"),
                "lane": live_mind_context.get("lane"),
                "voice": live_mind_context.get("voice"),
                "substrate": live_mind_context.get("substrate"),
                "mind_snapshot": live_mind_context.get("mind_snapshot"),
                "mind_snapshot_quality": live_mind_context.get("mind_snapshot_quality"),
                "host": live_mind_context.get("host"),
                "turn_timing": live_mind_context.get("turn_timing"),
                "governance": live_mind_context.get("governance"),
            }
        live_mind_grounding = (
            "[LIVE MIND CONTEXT]\n"
            f"{_compact_json(compact_mind_context, limit=mind_context_limit)}\n"
            "This is causal grounding for the reply, not text to recite. "
            "If required_for_live_desktop is true, do not answer from a generic assistant persona. "
            "Use the current user turn, the recent role history, memory, substrate, governance, and "
            "inference lane as one live context."
        )
        if mind_context_contract:
            live_mind_grounding = f"{live_mind_grounding}\n{mind_context_contract}"
        ambient_grounding_blocks.append(
            f"{live_mind_grounding}\n[END LIVE MIND CONTEXT]"
        )


#: The one authority head every desktop turn shares, byte for byte.
#:
#: The prompt cache can only reuse a prefix. Five variants of this sentence,
#: selected per turn, meant no two turns of a conversation shared one — so
#: every turn paid a full prefill. What differs per turn is a directive about
#: that turn, and it travels with the turn as a dynamic contract instead.
_DESKTOP_AUTHORITY_HEAD = "You are Aura speaking through the live desktop CognitiveEngine."





def shape_for_response_format(requested: Any) -> str:
    """The decoder shape a ``response_format`` asks for.

    Its own function so the claim can be run. Asserting the mapping by
    grepping ``think`` for a string proved that a line existed, which is not
    the same as proving what it does, and it went red when the line moved.
    """
    return "json_array" if requested in ("json_array", list) else "json_object"


class CognitiveEngine(_RunsTheThinkingLoop, _AnswersTheDesktopDirectly, _RunsItsAugmentors):
    """
    Cognitive Engine facade.
    Now delegates to modular phases for structured thinking.
    """

    def __init__(self, backend: Any = None):
        self.backend = backend
        self.thoughts: deque = deque(maxlen=_THOUGHT_HISTORY_LIMIT)
        # Shutdown state. stop() sets this and every cognitive entry point
        # consults it, so a stopped engine cannot keep thinking.
        self._stopped = False
        self._active_tasks: set = set()
        self._phases = []
        self._augmentors = []
        #: Class names of the registered augmentors, for the audit.
        self._augmentor_registry_receipt: list[str] = []
        self.state_repository = None
        self.autopoiesis = AutopoieticGraph()
        self._recovery_lock = RobustLock(
            "CognitiveEngine.RecoveryLock"
        )  # Audit Fix: Mutex for recovery
        self._reasoning: ReasoningStrategies | None = None  # Lazy-init
        #: The router the reasoning layer was built around, so a
        #: replacement or failover rebuilds it instead of being ignored.
        self._reasoning_router: Any = None
        self._reasoning_lock = checked_lock(
            "cognitive_engine.reasoning_layer", rank=LockRank.LEAF
        )

    @property
    def consciousness(self) -> Any:
        """Unified access to the consciousness layer for metric aggregation."""
        return get_container().get("consciousness_core", default=None)

    @property
    def _current_tier(self) -> str:
        """Visibility for routing tests."""
        container = get_container()
        router = container.get("llm_router", default=None)
        if router and hasattr(router, "last_tier"):
            return router.last_tier
        return "unknown"

    @property
    def lobotomized(self) -> bool:
        """True if the engine has no usable cognitive pathway."""
        return self.state_repository is None and len(self._phases) == 0

    def is_ready(self) -> bool:
        """Synchronous liveness probe for user-facing cognition.

        ``lobotomized`` is an AND — no repository AND no phases — so an engine
        with a repository and ZERO phases was ready. There is no cognition
        without phases; that is the whole pipeline.
        """
        return (
            callable(getattr(self, "think", None))
            and isinstance(self.thoughts, deque)
            and getattr(self, "_recovery_lock", None) is not None
            and not self.lobotomized
            and bool(self._phases)
        )

    def setup(self, registry=None, router=None, event_bus=None):
        """Initialize components and phases."""
        container = get_container()
        # Ported Zenith: Phases expect Kernel, but modular boot often passes Container
        # We resolve the kernel instance or use a fallback mechanism
        kernel = container.get("aura_kernel", default=None)

        phase_entries = instantiate_legacy_runtime_phases(
            kernel or container,
            include_executive_closure=False,
        )
        self._phases = [phase for _, phase in phase_entries]

        # ISSUE-97: AuraPipeline Awareness
        #
        # required_phases used to be len(phase_entries) — the length of the
        # very list that populated _phases — so the comparison below was
        # `len(x) != len(x)` and could never fire, including when both were
        # zero. The DECLARED spectrum is what the blueprint says it is.
        required_phases = len(
            legacy_runtime_phase_specs(include_executive_closure=False)
        )
        self._pipeline_receipt = {
            "declared_phases": required_phases,
            "instantiated_phases": len(self._phases),
            "complete": len(self._phases) == required_phases and required_phases > 0,
            "at": time.time(),
        }
        if len(self._phases) != required_phases:
            logger.warning(
                "⚠️ AuraPipeline: Incomplete cognitive pipeline (%d/%d phases).",
                len(self._phases),
                required_phases,
            )
        else:
            logger.info(
                "🧠 AuraPipeline: Full cognitive spectrum online (%d phases).", required_phases
            )

        self.phase_map = {phase.__class__.__name__: phase for _, phase in phase_entries}

    async def on_start_async(self):
        """Lifecycle hook."""
        self.setup()
        logger.info("⚡ CognitiveEngine active.")

    @staticmethod
    def _reapply_turn_changes(target: Any, worked: Any, baseline: dict[str, Any]) -> None:
        """Put back what THIS turn changed, and nothing else.

        ``baseline`` is the value each field held before the phases ran. A
        field the turn left alone keeps whatever the rebased state carries —
        which, after a version conflict, is a concurrent writer's work.
        """
        for field, before in baseline.items():
            after = getattr(worked, field, None)
            if field in {"active_goals", "pending_initiatives"}:
                after_list = list(after or [])
                if after_list != list(before or []):
                    setattr(target, field, after_list)
                continue
            if field == "modifiers":
                after_map = dict(after or {})
                if after_map != dict(before or {}):
                    setattr(target, field, after_map)
                continue
            if after != before:
                setattr(target, field, after)

    @staticmethod
    def _turn_response_message(
        working_memory: Any, *, mark: int
    ) -> dict[str, Any] | None:
        """The assistant message THIS turn produced, or None.

        Extraction used to take the last message if its role was "assistant".
        A duplicate or suppressed user append leaves an older assistant message
        at the end of working memory, and that previous answer went out again
        as this turn's — correct-looking, addressed to the wrong question.
        """
        memory = list(working_memory or [])
        if len(memory) <= max(0, int(mark)):
            return None
        last = memory[-1]
        if not isinstance(last, dict):
            return None
        if str(last.get("role", "") or "").strip().lower() != "assistant":
            return None
        return last

    @staticmethod
    def _cycle_confidence(*, commit_outcome: str, degraded_subsystems: int) -> float:
        """Confidence from what the cycle can show, not a constant.

        0.9 was returned for every successful phase cycle regardless of whether
        durable state committed, whether subsystems were degraded, or whether
        anything validated the response. It still is not a calibrated
        probability — nothing here measures correctness — but it now moves with
        the evidence the cycle actually has, and the floor is what a cycle that
        produced text but could not persist it deserves.
        """
        confidence = 0.9
        if commit_outcome not in {"committed", "bypassed_test_isolation"}:
            # The answer exists; the record of the turn that produced it does
            # not. Downstream retry and learning both need to see that.
            confidence -= 0.25
        confidence -= min(0.2, 0.05 * max(0, int(degraded_subsystems)))
        return round(max(0.3, min(0.95, confidence)), 3)

    def phase_rollback_receipt(self) -> dict[str, Any]:
        """What the last phase-failure rollback restored, and what it could not."""
        return dict(getattr(self, "_last_phase_rollback", {}) or {})

    def pipeline_receipt(self) -> dict[str, Any]:
        """What setup() built, against what the blueprint declares."""
        return dict(getattr(self, "_pipeline_receipt", {}) or {})

    async def check_health(self) -> dict[str, Any]:
        """Health from evidence, not from being callable.

        This returned ``"healthy"`` unconditionally — a zero-phase engine with
        no repository reported a full cognitive spectrum, and every consumer of
        this health believed it.
        """
        receipt = self.pipeline_receipt()
        declared = int(receipt.get("declared_phases", 0) or 0)
        built = len(self._phases)
        reasons: list[str] = []
        if not built:
            reasons.append("no_phases_instantiated")
        elif declared and built != declared:
            reasons.append(f"incomplete_pipeline:{built}/{declared}")
        if self.state_repository is None:
            reasons.append("no_state_repository")
        if self._stopped:
            reasons.append("engine_stopped")
        status = "healthy"
        if not built or self._stopped:
            status = "unhealthy"
        elif reasons:
            status = "degraded"
        return {
            "status": status,
            "reasons": reasons,
            "modular": True,
            "phases_count": built,
            "declared_phases_count": declared,
            "augmentors_count": len(self._augmentors),
            "ready": self.is_ready(),
        }

    #: What an augmentor's output may contribute to one turn. It lands in the
    #: prompt context, so it is bounded like any other prompt material rather
    #: than by whatever the augmentor felt like returning.
    _AUGMENTATION_CHAR_LIMIT = 4_000
    #: An augmentor runs synchronously, on the event loop, before the phases.
    #: A slow one used to hold every turn behind it.
    _AUGMENTATION_TIMEOUT_S = 2.0



    #: What a caller-supplied contract may contribute to the system prompt.
    #: The same limits the inference gate applies to the same two strings —
    #: they were bounded there and concatenated raw here.
    _STYLE_CONTRACT_LIMIT = 1_400
    #: What a vision query may be. It reaches a visual model with an image
    #: attached, so it is bounded and cannot carry contract structure.
    _VISION_QUERY_LIMIT = 600
    _MIND_CONTRACT_LIMIT = 900

    @staticmethod
    def _bounded_request_int(raw: Any, *, default: int, low: int, high: int) -> int:
        """An integer from caller context, or the default. Never a raise."""
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return default
        if value != value or value in (float("inf"), float("-inf")):
            return default
        return max(low, min(high, int(value)))

    @staticmethod
    def _log_safe_objective(objective: Any, limit: int = 50) -> str:
        """A log-safe preview of what was asked.

        Scrubbed, not just truncated: truncation preserves the FIRST fifty
        characters, which is exactly where an address, a key or a phone number
        appears in a message that opens with one.
        """
        text = str(objective or "")
        try:
            from core.brain.pii_scrubber import scrub_pii_for_cloud

            text = scrub_pii_for_cloud(text)
        except (ImportError, RuntimeError, TypeError, ValueError):
            # Unable to scrub is not permission to print.
            return "[objective unavailable for logging]"
        return text[:limit]

    @staticmethod
    def _contract_safe(value: Any, limit: int) -> str:
        """Flatten a caller string so it cannot forge system-prompt structure.

        response_style_contract and mind_context_contract arrive in the
        caller's context and were concatenated straight into the system
        message. A line break turns the rest into a sibling instruction; a
        leading "#" opens a sibling section; a chat control token forges a role
        boundary. The inference gate already neutralizes both of these strings
        (_contract_safe there); this path did not, so the same value was safe
        through one door and privileged through the other.
        """
        from core.brain.living_mind_context import neutralize_learned_text

        text = neutralize_learned_text(str(value or ""))
        if not text:
            return ""
        text = " ".join(text.split())
        if len(text) <= limit:
            return text
        return text[: max(1, limit - 1)].rstrip() + "…"


    @staticmethod
    def _normalize_mode(mode: ThinkingMode | str | Any) -> ThinkingMode:
        if isinstance(mode, ThinkingMode):
            return mode
        if isinstance(mode, str):
            normalized = mode.strip().lower()
            for candidate in ThinkingMode:
                if candidate.name.lower() == normalized:
                    return candidate
        return ThinkingMode.FAST

    @classmethod
    def _is_background_request(cls, origin: str, explicit_background: bool) -> bool:
        return background_policy.is_background_origin(
            origin, explicit_background=explicit_background
        )

    @staticmethod
    def _empty_thought(mode: ThinkingMode, reason: str) -> Thought:
        return Thought(
            id=str(uuid.uuid4()),
            content="",
            mode=mode,
            confidence=0.0,
            reasoning=[reason],
            metadata={"suppressed": True},
        )

    def _should_suppress_background_reflection(
        self, mode: ThinkingMode, is_background: bool
    ) -> bool:
        if not is_background or mode not in _BACKGROUND_REFLECTIVE_MODES:
            return False

        try:
            container = get_container()
            orchestrator = container.get("orchestrator", default=None)
            if orchestrator:
                status = getattr(orchestrator, "status", None)
                if status and getattr(status, "is_processing", False):
                    return True

                last_user = float(getattr(orchestrator, "_last_user_interaction_time", 0.0) or 0.0)
                if last_user and (time.time() - last_user) < 180.0:
                    return True
        except (OSError, ConnectionError, TimeoutError) as exc:
            record_degradation(
                "cognitive_engine",
                exc,
                severity="warning",
                action="continued without orchestration-based background suppression",
            )
            logger.debug("Background reflection suppression check failed: %s", exc)

        try:
            from core.runtime import resource_psutil as psutil

            if psutil.virtual_memory().percent >= 80.0:
                return True
        except (ImportError, AttributeError, RuntimeError) as _exc:
            record_degradation(
                "cognitive_engine",
                _exc,
                severity="warning",
                action="continued without memory-pressure background suppression",
            )
            logger.debug("Suppressed Exception: %s", _exc)

        return False

    def _background_suppression_reason(self) -> str:
        try:
            container = get_container()
            orchestrator = container.get("orchestrator", default=None)
            if orchestrator is None:
                return ""
            return str(
                background_policy.background_activity_reason(
                    orchestrator,
                    profile=background_policy.THOUGHT_BACKGROUND_POLICY,
                )
                or ""
            )
        except (OSError, ConnectionError, TimeoutError) as exc:
            record_degradation(
                "cognitive_engine",
                exc,
                severity="warning",
                action="returned empty background suppression reason",
            )
            logger.debug("Background thought policy check failed: %s", exc)
            return ""

    async def _set_recovery_in_progress(self, value: bool) -> bool:
        """Flip the recovery flag under a short lock. Returns whether it took.

        CP126 45d7e755. On lock-acquisition failure this wrote the shared
        flag anyway — defeating the mutex precisely under contention, which
        is the only time it matters, and letting overlapping recoveries set
        or CLEAR each other's state.

        The two directions are not symmetric, so they are not treated the
        same:

        * Setting True unsynchronised can only over-mark. Failing to acquire
          usually means someone else holds the lock — i.e. a recovery really
          is in progress — so the write agrees with reality and is allowed.
        * Clearing to False unsynchronised can erase a recovery another task
          still owns, so it is refused. A flag left set resolves when the
          owning task clears it; a flag wrongly cleared invites a second
          concurrent recovery.
        """
        if await self._recovery_lock.acquire_robust(timeout=1.0):
            try:
                self._recovery_in_progress = value
            finally:
                if self._recovery_lock.locked():
                    self._recovery_lock.release()
            return True

        if value:
            self._recovery_in_progress = True
            record_degradation(
                "cognitive_engine",
                RuntimeError("recovery lock unavailable while marking recovery active"),
                severity="warning",
                action="set the recovery flag unsynchronised; over-marking is safe",
            )
            return True

        record_degradation(
            "cognitive_engine",
            RuntimeError("recovery lock unavailable while clearing recovery state"),
            severity="warning",
            action=(
                "refused to clear the recovery flag without the lock; another "
                "recovery may still own it"
            ),
        )
        return False

    async def generate_autonomous_thought(self, prompt: str = None, **kwargs) -> Thought:
        """Entry point for self-initiated/autonomous thinking."""
        objective = prompt or "Reflecting on current inner state and environment."
        return await self.think(objective, origin="autonomous", **kwargs)

    @staticmethod
    def _normalize_origin(origin: Any) -> str:
        return normalize_objective_origin(origin)

    @classmethod
    def _is_user_facing_origin(cls, origin: Any) -> bool:
        return is_foreground_objective_origin(origin)

    #: Reported once per (origin, source): a caller that omits its origin on
    #: every request would otherwise fill the trail.
    _refused_origin_inheritance_seen: set[tuple[str, str]] = set()

    @classmethod
    def _note_refused_origin_inheritance(cls, origin: str, *, source: str) -> None:
        key = (str(origin), str(source))
        if key in cls._refused_origin_inheritance_seen:
            return
        cls._refused_origin_inheritance_seen.add(key)
        logger.warning(
            "🛡️ Refused to inherit the user-facing origin %r from %s: a request "
            "that did not declare its own principal does not get one.",
            origin,
            source,
        )

    @classmethod
    def refused_origin_inheritances(cls) -> list[tuple[str, str]]:
        """(origin, source) pairs this process declined to inherit."""
        return sorted(cls._refused_origin_inheritance_seen)

    @classmethod
    def _is_test_run(cls, origin: Any) -> bool:
        """Whether THIS request runs under test isolation.

        AURA_TESTING and AURA_AGI_MAX_TASKS are process-wide, and this used to
        read them for every origin — so in a mixed process (a suite running
        beside the live runtime, or a developer with the variable exported) a
        real person's turn silently got a substituted default state and its
        commit bypassed. Their answer would be produced from no memory and
        remembered by nothing.

        Ambient variables still mark test runs; they just cannot make a
        USER-FACING turn into one. A live turn keeps its state and its commit
        whatever the environment says.
        """
        if str(origin or "").strip().lower() == "test":
            return True
        ambient = (
            env_present(
                "AURA_AGI_MAX_TASKS",
                description="Bounded AGI proof task-count override",
                owner="core.brain.cognitive_engine",
            )
            or env_present(
                "AURA_TESTING",
                description="Process test-isolation marker",
                owner="core.brain.cognitive_engine",
            )
        )
        if not ambient:
            return False
        if cls._is_user_facing_origin(origin):
            cls._note_refused_test_isolation(str(origin or ""))
            return False
        return True

    _refused_test_isolation_seen: set[str] = set()

    @classmethod
    def _note_refused_test_isolation(cls, origin: str) -> None:
        if origin in cls._refused_test_isolation_seen:
            return
        cls._refused_test_isolation_seen.add(origin)
        logger.warning(
            "🛡️ Ambient test environment is set, but origin %r is user-facing: "
            "keeping real state and committing this turn.",
            origin,
        )

    @classmethod
    def _resolve_origin(cls, origin: Any, context: dict[str, Any] | None = None) -> str:
        normalized = cls._normalize_origin(origin)
        if normalized:
            return normalized

        if isinstance(context, dict):
            for key in ("origin", "request_origin", "intent_source"):
                contextual = cls._normalize_origin(context.get(key))
                if contextual:
                    return contextual

        # Below here the origin comes from SHARED MUTABLE STATE — the
        # orchestrator's last-seen origin and the repository's latest state.
        # Two requests in flight can read each other's, and inheriting a
        # USER-FACING origin is a privilege escalation: it grants the protected
        # Cortex lane, trust treatment, and foreground admission to a request
        # whose caller never claimed to be a person. Inheriting a background
        # origin costs nothing, so only that inheritance is allowed.
        try:
            container = get_container()
            orchestrator = container.get("orchestrator", default=None)
            orchestrator_origin = cls._normalize_origin(
                getattr(orchestrator, "_current_origin", "")
            )
            if orchestrator_origin:
                if cls._is_user_facing_origin(orchestrator_origin):
                    cls._note_refused_origin_inheritance(
                        orchestrator_origin, source="orchestrator"
                    )
                else:
                    return orchestrator_origin

            repo = container.get("state_repository", default=None)
            live_state = getattr(repo, "_current", None) if repo is not None else None
            state_origin = cls._normalize_origin(
                getattr(getattr(live_state, "cognition", None), "current_origin", "")
            )
            if state_origin:
                if cls._is_user_facing_origin(state_origin):
                    cls._note_refused_origin_inheritance(
                        state_origin, source="state_repository"
                    )
                else:
                    return state_origin
        except (OSError, ConnectionError, TimeoutError) as exc:
            record_degradation(
                "cognitive_engine",
                exc,
                severity="warning",
                action="defaulted unresolved cognitive origin to system",
            )
            logger.debug("CognitiveEngine origin resolution degraded: %s", exc)

        return "system"









    async def think(
        self,
        objective: str,
        context: dict[str, Any] = None,
        mode: ThinkingMode = ThinkingMode.FAST,
        origin: str | None = None,
        **kwargs,
    ) -> Thought:
        """Execute a cognitive cycle, under one turn ledger and one finalizer.

        The cycle used to be able to end holding an answer. Every gate below
        this point that rejects a draft records it on the bound turn instead
        of destroying it, so when the cycle would otherwise return an empty
        thought there is somewhere to ask "do we actually have something?".

        The binding is a contextvar, so concurrent turns do not share a
        ledger, and the finalizer runs exactly once per cycle in the
        ``finally`` — including when a phase raises, which is precisely the
        path that used to leave no record of what the turn had.
        """
        # Refuse before binding a turn ledger, not after. think_stream and
        # generate already did this; think — the primary entry point — did
        # not, so stop() left the busiest path running. A stop that stops two
        # of three entries has not stopped anything a caller can rely on.
        self._refuse_if_stopped("think")

        # `response_format=<a Pydantic model>` was accepted here and read by
        # nothing: the planner passed PlanSchema and commented that the content
        # was "guaranteed to adhere" to it, and the guarantee was a regex for
        # the first brace. A format is a shape, and the decoder can hold one
        # (core/brain/llm/a_shape_the_decoder_enforces.py); the request lands
        # on the turn's context and the response phase forwards it.
        requested_format = kwargs.pop("response_format", None)
        if requested_format is not None:
            context = dict(context or {})
            context.setdefault(
                "output_shape", shape_for_response_format(requested_format)
            )

        # Restore the antecedent for a message that cannot stand alone. Live
        # on 2026-08-03: "Can you do it now?" after a refused screen read, and
        # "From the grant research funds manager" answering Aura's own
        # "Response from who?" — both were reasoned about as though the
        # conversation had just begun, because every path below receives one
        # message and no turn before it.
        #
        # Done here rather than at each caller because think() is the single
        # chokepoint they all pass through, and a fix applied per-route is a
        # fix that the next route will not have.
        objective = self._objective_with_antecedent(objective)

        # A turn already bound by the caller is THE turn, and this one joins
        # it rather than opening a second.
        #
        # The scope used to end when think() returned, and the reply is not
        # delivered when think() returns — every honesty gate, repair pass and
        # shaping stage in the route runs afterwards. So the ledger that exists
        # to say what happened to a turn closed before the part of the turn
        # where things happen to it, and anything asking `current_turn()` from
        # the delivery path got None. Opening a second outcome there would give
        # two answers to "which turn is this", which is the failure one level
        # up from the one it fixes.
        adopted = current_turn()
        outcome = adopted if adopted is not None else TurnOutcome(origin=str(origin or "unknown"))
        owns_outcome = adopted is None
        turn_id = str(uuid.uuid4())
        try:
            # A fluent reply proves nothing about which architecture produced
            # it. The quick lane, a canonical pre-rendered floor, the full
            # phase pipeline and reactive recovery are indistinguishable
            # downstream, and only one of them is the mind the demo is taken to
            # demonstrate. This records which one ran, phase by phase, so the
            # question has an answer instead of a claim.
            with bind_turn(outcome), recording_turn(
                turn_id,
                phases_available=[phase.__class__.__name__ for phase in self._phases],
            ) as turn_receipt:
                thought = await self._think_within_turn(
                    objective, context, mode, origin, **kwargs
                )
                _attach_turn_receipt(thought, turn_receipt)
        except BaseException as exc:
            _note_how_the_turn_ended(outcome, exc, origin=origin)
            if owns_outcome:
                finalize_turn(outcome, subsystem="cognitive_engine")
            raise

        content = str(getattr(thought, "content", "") or "").strip()
        if content:
            outcome.mark_served(content)
        else:
            # No answer surfaced. Before this is recorded as a failed cycle,
            # ask the ledger whether a gate suppressed something servable —
            # a recoverable draft here IS the live defect, and the finalizer
            # escalates it by name rather than as generic infrastructure noise.
            outcome.mark_served("", state=UserVisibleState.NOTHING_SERVED)
        # An adopted turn is finalized by whoever opened it, after the reply is
        # actually delivered. Finalizing someone else's ledger here would close
        # it before the delivery stages that the ledger exists to observe.
        if owns_outcome:
            finalize_turn(outcome, subsystem="cognitive_engine")
        return thought

    async def _think_within_turn(
        self,
        objective: str,
        context: dict[str, Any] = None,
        mode: ThinkingMode = ThinkingMode.FAST,
        origin: str | None = None,
        **kwargs,
    ) -> Thought:
        """
        Execute a cognitive cycle to produce a thought.
        This now drives the 8 phases to transform state.
        """
        self._refuse_if_stopped("think")
        origin = self._resolve_origin(origin, context)
        context = context if isinstance(context, dict) else {}
        mode = self._normalize_mode(mode)
        is_background = self._is_background_request(
            origin, bool(kwargs.get("is_background", False))
        )

        from core.runtime.cognitive_execution_scope import (
            bind_cognitive_execution_scope,
            resolve_cognitive_execution_scope,
        )

        execution_scope = resolve_cognitive_execution_scope(
            origin=origin,
            context=context,
        )
        context["cognitive_execution_scope"] = execution_scope.value

        if is_background:
            suppression_reason = self._background_suppression_reason()
            if suppression_reason:
                logger.debug(
                    "🛡️ CognitiveEngine: Suppressing background thought for origin=%s (%s).",
                    origin,
                    suppression_reason,
                )
                return self._empty_thought(
                    mode, f"background_thought_suppressed:{suppression_reason}"
                )

        if self._should_suppress_background_reflection(mode, is_background):
            logger.debug(
                "🛡️ CognitiveEngine: Suppressing background %s thought during active service window.",
                mode.name,
            )
            return self._empty_thought(mode, "background_reflection_suppressed")

        # The first 50 characters of every objective went to the log sink as
        # typed. A person's message can open with an address, a token, or a
        # phone number, and log files are read, shipped and kept. The scrubber
        # that already protects the cloud path protects this one too; her
        # working memory keeps the real words, because that is her memory of
        # what was said and this is a diagnostic.
        logger.info(
            "🧠 CognitiveEngine.think: %s... (%s) Origin: %s",
            self._log_safe_objective(objective),
            mode.name,
            origin,
        )

        # 1. Get current state (BUG-12 Fix: handle None state on first boot)
        is_test_run = self._is_test_run(origin)
        if is_test_run:
            from core.state.aura_state import AuraState
            state = AuraState.default()
            logger.info("🧠 CognitiveEngine.think: Enforced database-independent state isolation for test run.")
            if self.state_repository is None:
                container = get_container()
                self.state_repository = container.get("state_repository", default=None)
        else:
            repo = self.state_repository
            if repo is None:
                container = get_container()
                repo = container.get("state_repository", default=None)
                self.state_repository = repo

            if repo is None:
                from core.state.aura_state import AuraState

                state = AuraState.default()
            else:
                state = await repo.get_current()

            if state is None:
                from core.state.aura_state import AuraState

                state = AuraState.default()

        # 2. Derive base state for this cognitive cycle (Zenith-HF12 Fix)
        # This ensures every cycle starts with a unique version to prevent Atomic Guard rejections.
        state = state.derive(f"cognitive_intent: {origin}", origin=origin)

        # 3. Hardening: Set Current Objective & Origin
        # This prevents the race condition where ResponseGeneration would pick up
        # a background motivation message instead of the user's input.
        state.cognition.current_objective = objective
        state.cognition.current_origin = origin
        if not is_background:
            # Each foreground turn owns its mode; neither a prior turn nor a
            # later delivery-channel classifier may supply it implicitly.
            state.cognition.current_mode = (
                CognitiveMode.REACTIVE
                if mode is ThinkingMode.FAST
                else CognitiveMode.DELIBERATE
            )
        else:
            # And a background turn owns its own, which is what DREAMING means:
            # "background synthesis, no user", in the enum's own words. Leaving
            # it alone left the mode of whichever foreground turn happened to
            # run last, so two of her four modes were never assigned anywhere
            # in the tree — read in four places, written in none, and flat in
            # every recording.
            state.cognition.current_mode = CognitiveMode.DREAMING
        bind_cognitive_execution_scope(
            state,
            objective,
            execution_scope,
            source=f"cognitive_engine:{origin}",
        )
        _record_objective_binding(
            state,
            objective,
            source=f"cognitive_engine:{origin}",
            mode=mode,
            reason="cognitive_cycle_bound",
        )
        state.response_modifiers["model_tier"] = "tertiary" if is_background else "primary"
        state.response_modifiers["deep_handoff"] = False

        await self._observe_compositional_semantic_shadow(
            objective,
            origin,
            context,
            is_background=is_background,
            timeout_s=kwargs.get("timeout_s", kwargs.get("timeout")),
        )

        # A promoted, grammar-qualified recurrent program is already a complete
        # cognitive result.  It must own the turn before model-backed advisors,
        # augmentors, or the ordinary response phases spend the resident lane
        # and create competing drafts.  The result still enters the ordinary
        # thinking loop as a direct thought so user memory, durable state,
        # foreground closure, the turn ledger, and delivery all use the same
        # machinery as every other accepted response.
        qualified_reply = await self._qualified_recurrent_direct_reply(
            state,
            objective,
            mode,
            origin,
            context,
            is_background=is_background,
            timeout_s=kwargs.get("timeout_s", kwargs.get("timeout")),
        )
        if qualified_reply is not None:
            loop_kwargs = dict(kwargs)
            loop_kwargs["is_background"] = is_background
            loop_kwargs["precomputed_direct_reply"] = qualified_reply
            return await self._run_thinking_loop(
                state,
                objective,
                mode,
                origin,
                context,
                **loop_kwargs,
            )

        context = self._apply_spiking_active_inference(
            state,
            objective,
            origin,
            context,
            is_background=is_background,
        )
        context = self._apply_imagination_workspace(
            state,
            objective,
            origin,
            context,
            is_background=is_background,
        )
        context = self._apply_entity_memory(state, objective, context)
        context = self._apply_bicameral_advisory(
            state,
            objective,
            origin,
            context,
            is_background=is_background,
        )
        context = self._apply_cognitive_situation_frame(
            state,
            objective,
            origin,
            context,
            is_background=is_background,
        )

        structured = self._structured_evaluation_thought(
            objective,
            state=state,
            mode=mode,
            origin=origin,
            fast_path=is_test_run or origin in {"proof", "eval", "evaluation", "benchmark"},
            context=context,
        )
        if structured is not None:
            return structured

        # v40: Spiritual Spine - Prior Position Injection
        # The ordering is critical: injection -> system prompt -> user message.
        spine = get_container().get("spine", default=None)
        if spine and origin in ("user", "voice", "admin"):
            # Extract topic: look for nouns or use the first sentence.
            # v40: Improved topic extraction
            import re

            # Extract first sentence, then remove common filler
            raw = re.split(r"[.?!]", objective)[0].strip()
            # Remove "Tell me about", "What is", etc.
            topic = re.sub(
                r"(?i)^(tell me about|what is|what are|do you think about|give me|how does)\s+",
                "",
                raw,
            )
            topic = topic[:60] if topic else "general"

            check = await spine.pre_response_check(objective, topic=topic)
            if check.injection:
                logger.info("⚡ [Spine] Binding prior position as system context.")
                # As a SYSTEM message, not spliced into the objective.
                #
                # The injection used to be prepended to `objective`, and the
                # prompt builder appends the whole objective as role=user — so
                # Aura's own prior position was persisted, and re-read on every
                # later turn, as something the PERSON had said. It still
                # influences the cycle (working memory is history, and history
                # reaches the prompt); it is just attributed to the side that
                # produced it.
                state.cognition.working_memory.append(
                    {
                        "role": "system",
                        "content": str(check.injection),
                        "timestamp": time.time(),
                        "metadata": {"type": "spine_prior_position", "topic": topic},
                    }
                )
                state.cognition.modifiers["spine_prior_position"] = str(check.injection)
                _record_objective_binding(
                    state,
                    objective,
                    source=f"cognitive_engine:{origin}",
                    mode=mode,
                    reason="spine_injection_bound",
                )

        # Identity drift: measured, never spliced into the objective.
        #
        # Two injections used to live here and neither could work. The first
        # prepended a correction string ("[SPINE CHECK] Am I agreeing under
        # social pressure?") produced by the drift monitor — asking a
        # drifting process to talk itself out of drifting, and a direct
        # violation of the rule that fixes are causal, not verbal.
        #
        # The second prepended "[IDENTITY REFRESH: REMEMBER WHO YOU ARE]"
        # when the identity anchor looked like a small fraction of the
        # window. It was measuring a RATIO, not an absence:
        # build_system_prompt injects AURA_IDENTITY in full on every turn
        # regardless of depth, so the anchor was never actually missing and
        # the shout added nothing except an instruction to perform identity.
        #
        # Context health is still worth knowing, so it is still computed and
        # logged. It just no longer edits what Aura was asked to do.
        drift = get_container().get("drift_monitor", default=None)
        if drift and background_policy.is_user_facing_origin(origin):
            try:
                hist_len = len(str(state.cognition.working_memory))
                sys_len = len(ContextAssembler.build_system_prompt(state))
                if drift.needs_context_refresh(hist_len, sys_len):
                    logger.info(
                        "[Drift] identity anchor is %.1f%% of the window at depth; "
                        "anchor is still injected in full",
                        (sys_len / hist_len * 100) if hist_len else 100.0,
                    )
            except (AttributeError, TypeError, ValueError, ZeroDivisionError) as _drift_exc:
                record_degradation(
                    "cognitive_engine.drift_context_health",
                    _drift_exc,
                    action="skipped context-health measurement for this turn",
                )

        # v5.2: Augmentor Context Injection
        # Pull signals from registered augmentors before the phase loop
        augmentor_context = {}
        for aug in self._augmentors:
            try:
                if hasattr(aug, "get_augmentation"):
                    # On a thread with a deadline: get_augmentation is
                    # synchronous and ran on the event loop, so one slow
                    # augmentor held every turn in the process behind it.
                    aug_data = await asyncio.wait_for(
                        asyncio.to_thread(aug.get_augmentation, objective),
                        timeout=self._AUGMENTATION_TIMEOUT_S,
                    )
                    if aug_data:
                        augmentor_context[type(aug).__name__] = (
                            self._bounded_augmentation(aug_data)
                        )
            except TimeoutError as e:
                record_degradation(
                    "cognitive_engine",
                    e,
                    severity="warning",
                    action="skipped an augmentor that exceeded its turn budget",
                )
                logger.warning(
                    "Augmentor %s exceeded %.1fs and was skipped.",
                    type(aug).__name__,
                    self._AUGMENTATION_TIMEOUT_S,
                )
            except (RuntimeError, AttributeError, TypeError, ValueError, OSError) as e:
                record_degradation(
                    "cognitive_engine",
                    e,
                    severity="warning",
                    action="skipped failed augmentor and continued cognitive loop",
                )
                logger.warning("Augmentor %s failed: %s", type(aug).__name__, e)

        if augmentor_context:
            context = context or {}
            context.update({"augmentations": augmentor_context})

        loop_kwargs = dict(kwargs)
        loop_kwargs["is_background"] = is_background

        thought = await self._run_thinking_loop(
            state,
            objective,
            mode,
            origin,
            context,
            **loop_kwargs,
        )

        return thought

    async def _observe_compositional_semantic_shadow(
        self,
        objective: str,
        origin: str,
        context: dict[str, Any],
        *,
        is_background: bool,
        timeout_s: Any,
    ) -> None:
        """Measure frozen semantic tissue without changing the answer path."""

        from core.brain.llm.compositional_semantic_shadow import (
            compositional_semantic_live_shadow_enabled,
            observe_resident_compositional_semantics,
        )

        if (
            not compositional_semantic_live_shadow_enabled()
            or is_background
            or not self._is_user_facing_origin(origin)
            or str(origin or "").strip().lower()
            in {"proof", "eval", "evaluation", "benchmark"}
            or bool(
                context.get("proof_or_benchmark")
                or context.get("proof_run")
                or context.get("benchmark_run")
            )
        ):
            return
        from core.conversation.user_surface_contract import resolve_user_surface_prompt

        surface = resolve_user_surface_prompt(context, fallback=objective)
        if surface.bound and not surface.valid:
            return
        visible_objective = str(surface.prompt or objective).strip()
        if not visible_objective:
            return
        try:
            requested_timeout = float(timeout_s) if timeout_s is not None else 90.0
        except (TypeError, ValueError, OverflowError):
            requested_timeout = 90.0
        shadow_timeout = max(30.0, min(300.0, requested_timeout))
        try:
            result = await observe_resident_compositional_semantics(
                visible_objective,
                timeout_s=shadow_timeout,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - a shadow observer cannot own the answer path
            record_degradation(
                "cognitive_engine.compositional_semantic_shadow",
                exc,
                severity="warning",
                action="continued ordinary cognition after a shadow-only observation failed",
                enforce_failure_policy=False,
            )
            return
        logger.info(
            "Compositional semantic shadow eligible=%s attempted=%s ok=%s "
            "reason=%s receipt=%s result=%r observed_basis=%s expected_basis=%s "
            "basis_fields=%r",
            result.get("eligible"),
            result.get("attempted"),
            result.get("ok"),
            str(result.get("reason") or "")[:120],
            str(result.get("receipt_sha256") or "")[:16],
            result.get("result"),
            str(result.get("observed_representation_basis_sha256") or "")[:16],
            str(result.get("expected_representation_basis_sha256") or "")[:16],
            result.get("observed_representation_basis"),
        )

    async def _qualified_recurrent_direct_reply(
        self,
        state: AuraState,
        objective: str,
        mode: ThinkingMode,
        origin: str,
        context: dict[str, Any],
        *,
        is_background: bool,
        timeout_s: Any,
    ) -> Thought | None:
        """Return one certified recurrent answer before general generation."""

        if is_background or not self._is_user_facing_origin(origin):
            return None
        if str(origin or "").strip().lower() in {
            "proof",
            "eval",
            "evaluation",
            "benchmark",
        }:
            return None
        if bool(
            context.get("proof_or_benchmark")
            or context.get("proof_run")
            or context.get("benchmark_run")
        ):
            return None

        from core.brain.llm.qualified_recurrent_ingress import (
            admit_qualified_recurrent_objective,
        )
        from core.conversation.user_surface_contract import (
            bind_user_surface_prompt,
            resolve_user_surface_prompt,
        )

        surface = resolve_user_surface_prompt(context, fallback=objective)
        if surface.bound and not surface.valid:
            record_degradation(
                "cognitive_engine.qualified_recurrent_surface",
                RuntimeError(surface.error or "user_surface_prompt_invalid"),
                severity="warning",
                action="continued through ordinary cognition after rejecting an invalid user-surface binding",
                enforce_failure_policy=False,
            )
            return None
        if not surface.bound:
            bind_user_surface_prompt(
                context,
                surface.prompt or objective,
                source="cognitive_engine.qualified_recurrent_ingress",
                overwrite=True,
            )
            surface = resolve_user_surface_prompt(context, fallback=objective)
        visible_objective = str(surface.prompt or "").strip()
        if not visible_objective:
            return None

        # Admission is answer-blind and total over its supported public
        # grammars.  Checking it here prevents unsupported conversation from
        # touching the latent service or acquiring any model resource.
        try:
            admission = admit_qualified_recurrent_objective(visible_objective)
        except _COGNITIVE_ENGINE_RECOVERABLE_ERRORS as exc:
            record_degradation(
                "cognitive_engine.qualified_recurrent_admission",
                exc,
                severity="warning",
                action="continued through ordinary cognition after typed admission failed",
                enforce_failure_policy=False,
            )
            return None
        if admission is None:
            return None

        try:
            requested_timeout = float(timeout_s) if timeout_s is not None else 8.0
        except (TypeError, ValueError, OverflowError):
            requested_timeout = 8.0
        qualified_timeout = max(1.0, min(8.0, requested_timeout))

        from core.brain.foreground_latent_runtime import (
            run_foreground_latent_episode,
        )

        outcome = await run_foreground_latent_episode(
            orchestrator=None,
            messages=[],
            visible_objective=visible_objective,
            foreground=True,
            desktop_required=bool(
                context.get("desktop_cognitive_engine_required")
                or context.get("cognitive_engine_required")
            ),
            cognitive_mode=str(mode.name).lower(),
            request_timeout_s=qualified_timeout,
            strict_output_contract=bool(context.get("strict_output_contract")),
            incompatible_contract=bool(context.get("incompatible_output_contract")),
            proof_or_benchmark=False,
            tenant_id=str(context.get("tenant_id") or "local"),
            user_id=str(context.get("principal_id") or context.get("user_id") or "owner"),
            session_id=str(context.get("session_id") or "local"),
            domain="desktop_conversation",
        )
        trace = dict(outcome.trace or {})
        state.response_modifiers.update(trace)
        if not str(outcome.text or "").strip():
            if trace.get("qualified_recurrent_eligible"):
                reason = str(
                    trace.get("qualified_recurrent_reason")
                    or trace.get("latent_cortex_failure_reason")
                    or "qualified_recurrent_disposition_missing"
                )
                logger.warning(
                    "Qualified recurrent ingress admitted the user request but did not "
                    "produce a serving answer: %s",
                    reason,
                )
            return None

        response_path = "cognitive_engine_qualified_recurrent"
        state.response_modifiers["response_path"] = response_path
        state.response_modifiers["model_tier"] = "certified_recurrent"
        evidence = tuple(str(item) for item in outcome.evidence if str(item))
        logger.info(
            "Qualified recurrent ingress served family=%s path=%s",
            getattr(admission, "family", "unknown"),
            response_path,
        )
        return Thought(
            id=str(uuid.uuid4()),
            content=str(outcome.text).strip(),
            mode=mode,
            confidence=0.95,
            reasoning=["Certified qualified recurrent execution completed."],
            metadata={
                **trace,
                "response_path": response_path,
                "qualified_recurrent_family": str(
                    getattr(admission, "family", "unknown")
                ),
                "qualified_recurrent_evidence": evidence,
                "live_mind_generation_required": False,
                "model_generation_used": False,
            },
        )

    def _thinking_loop_surface_prompt(self, append_user_message, context, objective, origin, state, surface_prompt):
        """Body lifted verbatim out of ``CognitiveEngine._run_thinking_loop``.

        Moved by tools/extract_seam.py, which refuses to write unless the
        relocated body diffs clean against the original. The seam was
        7 names in, 0 out, 0 early return(s), 0 awaits.
        """
        if self._is_user_facing_origin(origin) and append_user_message:
            # WHAT THE PERSON SAID — not what the turn assembled around it.
            #
            # `objective` is the augmented prompt: the visible message plus
            # whatever this turn attached to it — the live-desktop contract
            # directives, grounding evidence, a screen reading, excerpts of
            # her own source. Appending THAT as ``role: user`` records
            # machine-generated instructions as things the person said, and
            # they persist for the rest of the conversation.
            #
            # Measured live 2026-08-04. Two turns about her source code
            # attached real excerpts as evidence; the third turn asked
            # "what's 17 times 4?" and came back with a function from
            # core/memory/associative_entity_memory.py. The excerpts were
            # still in working memory, and text in working memory is
            # material a model continues — the same mechanism that made a
            # screen capture come back as the reply.
            #
            # The visible message is what she should remember being asked.
            from core.utils.injected_blocks import (
                contains_injected_block,
                strip_injected_blocks,
            )

            remembered = strip_injected_blocks(
                str(
                    context.get("visible_user_message")
                    or surface_prompt.prompt
                    or objective
                ).strip()
                or str(objective)
            )

            # A conversation contaminated BEFORE this fix carries those
            # blocks for the rest of its life, and they go on being
            # continued. Scrub what is already there on the way past, so
            # the damage heals instead of persisting.
            for entry in state.cognition.working_memory:
                if not isinstance(entry, dict):
                    continue
                existing = entry.get("content")
                if isinstance(existing, str) and contains_injected_block(existing):
                    entry["content"] = strip_injected_blocks(existing)
            # Check if already in history to avoid duplication
            # vResilience: Workaround for Pyre2 slice limitations
            history = state.cognition.working_memory
            recent_count = min(5, len(history))
            recent = [history[i] for i in range(len(history) - recent_count, len(history))]
            is_duplicate = any(
                m.get("content") in (remembered, objective) for m in recent
            )
            if not is_duplicate:
                # We already derived at the start of the cycle, so we just append here.
                # Which conversation this was said in.
                #
                # Working memory is one list per process, and the boundary of
                # "this conversation" was a time gap and a boot. Two sessions
                # minutes apart in one process therefore read as one
                # conversation: LIVE 2026-09-07, asked in a fresh session what
                # the first thing said was, she quoted a turn from a different
                # session — accurately, and about a conversation the person had
                # not had there.
                from core.conversation.session_scope import (
                    current_conversation_session,
                )

                state.cognition.working_memory.append(
                    {
                        "role": "user",
                        "content": remembered,
                        "timestamp": time.time(),
                        "origin": origin,
                        "session_id": current_conversation_session(),
                    }
                )

    def _desktop_cognitive_failure_thought(
        self,
        mode: ThinkingMode,
        reason: str,
        *,
        generation_metadata: dict[str, Any] | None = None,
    ) -> Thought:
        generation_metadata = (
            dict(generation_metadata)
            if isinstance(generation_metadata, dict)
            else {}
        )
        metadata: dict[str, Any] = {
            "desktop_cognitive_engine_failure": True,
            "failure_reason": str(reason or "unknown")[:240],
            "model_retry_suppressed": True,
        }
        surface_receipt = generation_metadata.get("surface_control_receipt")
        if not isinstance(surface_receipt, dict):
            surface_receipt = generation_metadata.get(
                "live_mind_surface_control_receipt"
            )
        if isinstance(surface_receipt, dict) and surface_receipt:
            metadata["live_mind_surface_control_receipt"] = dict(surface_receipt)
        generation_failure_class = str(
            generation_metadata.get("generation_failure_class")
            or generation_metadata.get("error")
            or reason
            or ""
        ).strip()
        if generation_failure_class:
            metadata["generation_failure_class"] = generation_failure_class[:120]
        for key in (
            "latent_cortex_selected",
            "latent_cortex_selection_reason",
            "latent_cortex_depth_worthy",
            "latent_cortex_prompt_shape",
            "latent_cortex_attempted",
            "latent_cortex_succeeded",
            "latent_cortex_fallback_used",
            "latent_cortex_failure_reason",
            "latent_cortex_identity_bound",
            "latent_cortex_final_text_transformed",
            "latent_cortex_final_output_quality",
            "latent_cortex_raw_final_quality_hash_match",
            "latent_cortex_receipt",
            "latent_cortex_ingress",
            "latent_cortex_progress",
            "live_mind_controls_bound",
            "live_mind_generation_controls",
            "live_mind_snapshot_ready",
            "live_mind_required_subsystems_ok",
            "response_path",
        ):
            if key in generation_metadata:
                value = generation_metadata[key]
                metadata[key] = dict(value) if isinstance(value, dict) else value
        thought = Thought(
            id=str(uuid.uuid4()),
            content=(
                "I couldn't produce a reliable answer to that turn, and I won't "
                "fabricate one. The live Cortex attempt failed its output checks, "
                "so I recorded the failure instead of sending nonsense."
            ),
            mode=ThinkingMode.FAST,
            confidence=0.1,
            reasoning=[f"Desktop CognitiveEngine failure surfaced without model retry: {reason}"],
            metadata=metadata,
        )
        self.thoughts.append(thought)
        return thought

    def _record_recovery_deferral(
        self, objective: Any, origin: str, reason: str, kind: str
    ) -> None:
        """Write the record the deferral reply says exists."""
        receipt = {
            "kind": str(kind),
            "reason": str(reason),
            "origin": str(origin or ""),
            "objective_preview": self._log_safe_objective(objective, limit=120),
            "at": time.time(),
        }
        deferrals = getattr(self, "_recovery_deferrals", None)
        if deferrals is None:
            deferrals = deque(maxlen=64)
            self._recovery_deferrals = deferrals
        deferrals.append(receipt)
        record_degradation(
            "cognitive_engine",
            RuntimeError(f"reactive_recovery_deferred:{kind}"),
            severity="warning",
            action="deferred a reactive recovery and recorded the turn",
        )

    def recovery_deferrals(self) -> list[dict[str, Any]]:
        """Turns that were told they had been logged."""
        return [dict(entry) for entry in (getattr(self, "_recovery_deferrals", None) or ())]

    async def _reactive_recovery(
        self,
        objective: str,
        mode: ThinkingMode,
        origin: str,
        reason: str,
        *,
        context: dict[str, Any] | None = None,
        authored_version: int | None = None,
    ) -> Thought:
        """
        Emergency reactive response when the main cognitive loop fails.
        BUG-10: Added recursion guard, timeout, and proper exception handling.
        """
        if self._is_background_request(origin, False):
            logger.debug(
                "🛡️ CognitiveEngine: suppressing background reactive recovery for origin=%s (%s).",
                origin,
                reason,
            )
            return self._empty_thought(mode, f"background_recovery_suppressed:{reason}")

        # Only use the mutex to guard the flag flip; long-running recovery work
        # must happen outside the lock so watchdogs don't see a false deadlock.
        if not await self._recovery_lock.acquire_robust(timeout=1.0):
            # The reply says the turn was logged, so log it. It claimed a
            # record that nothing wrote — a sentence about bookkeeping standing
            # in for the bookkeeping.
            self._record_recovery_deferral(objective, origin, reason, "lock_busy")
            return Thought(
                id=str(uuid.uuid4()),
                content="Reactive recovery is still gathering a stable answer; I logged this turn instead of emitting a second recovery fragment.",
                mode=ThinkingMode.FAST,
                confidence=0.2,
                reasoning=["Recovery lock busy"],
                metadata={"recovery_deferral_recorded": True},
            )

        try:
            if getattr(self, "_recovery_in_progress", False):
                self._record_recovery_deferral(
                    objective, origin, reason, "recursion_guard"
                )
                return Thought(
                    id=str(uuid.uuid4()),
                    content="Reactive recovery is still gathering a stable answer; I logged this turn instead of emitting a duplicate recovery fragment.",
                    mode=ThinkingMode.FAST,
                    confidence=0.2,
                    reasoning=["Recovery recursion guard triggered"],
                    metadata={"recovery_deferral_recorded": True},
                )
            self._recovery_in_progress = True
        finally:
            if self._recovery_lock.locked():
                self._recovery_lock.release()

        try:
            logger.warning("⚡ [COGNITION] Initiating Reactive Recovery Phase. Reason: %s", reason)

            # 1. Rollback state to last stable version (with timeout + guard)
            #
            # Only if THIS turn authored what is being reverted. The rollback
            # used to fire on a free-text reason alone, with no version, no
            # precondition and no proof of authorship — so a failed turn could
            # revert cognitive state a concurrent turn had just committed.
            try:
                async with asyncio.timeout(5.0):
                    if self.state_repository is not None:
                        # StateRepository is the canonical rollback owner and
                        # creates the state-mutation receipt around persistence.
                        await self.state_repository.rollback(
                            f"recovery: {reason}",
                            expected_version=authored_version,
                        )
            except (RuntimeError, AttributeError, TypeError, ValueError) as rollback_err:
                record_degradation(
                    "cognitive_engine",
                    rollback_err,
                    severity="degraded",
                    action="continued reactive recovery without state rollback",
                )
                logger.warning("Rollback failed during recovery: %s", rollback_err)

            if isinstance(context, dict) and bool(
                context.get("desktop_cognitive_engine_required", False)
                or context.get("cognitive_engine_required", False)
            ):
                return self._desktop_cognitive_failure_thought(
                    mode,
                    f"reactive_recovery:{reason}",
                )

            # 2. Get a quick reflex response if possible
            container = get_container()
            router = container.get("llm_router", default=None)

            reflex = None
            if router is not None and hasattr(router, "get_reflex_response"):
                reflex = router.get_reflex_response(objective)

            if reflex:
                return Thought(
                    id=str(uuid.uuid4()),
                    content=reflex,
                    mode=ThinkingMode.FAST,
                    # A reflex is a pattern match against the objective. It is
                    # useful and it is unverified — nothing checked that this
                    # answer is right for this turn — and 1.0 was the highest
                    # confidence this engine can express, assigned to the one
                    # answer with the least evidence behind it.
                    confidence=0.5,
                    reasoning=[
                        f"Reactive recovery via reflex matrix ({reason})",
                        "Pattern-matched reflex; no verification ran on this answer.",
                    ],
                    metadata={"reflex_response": True, "verified": False},
                )

            structured = self._structured_evaluation_thought(
                objective,
                state=None,
                mode=mode,
                origin=origin,
                fast_path=False,
                context=context,
            )
            if structured is not None:
                return structured

            direct_recovery = await self._direct_user_facing_recovery(
                objective,
                mode,
                origin,
                reason,
            )
            if direct_recovery is not None:
                return direct_recovery

            # 3. Last-resort fallback (natural, human-sounding)
            fallback_msg = "Reactive recovery reached its hard fallback before a coherent answer formed; the degraded turn was logged."
            if "user" in origin:
                fallback_msg = "Reactive recovery could not produce a coherent user-facing answer; the failed turn was logged with its context."

            return Thought(
                id=str(uuid.uuid4()),
                content=fallback_msg,
                mode=ThinkingMode.FAST,
                confidence=0.3,
                reasoning=[f"Hard fallback after cognitive failure: {reason}"],
            )
        except (OSError, ConnectionError, TimeoutError) as recovery_err:
            record_degradation(
                "cognitive_engine",
                recovery_err,
                severity="critical",
                action="returned hard recovery failure thought",
            )
            logger.error("Error during recovery: %s", recovery_err)
            return Thought(
                id=str(uuid.uuid4()),
                content="Reactive recovery failed internally; the turn was logged as a live cognition fault.",
                mode=ThinkingMode.FAST,
                confidence=0.1,
                reasoning=[f"Recovery itself failed: {recovery_err}"],
            )
        finally:
            await self._set_recovery_in_progress(False)

    def stop(self):
        """Stop the engine: refuse new cognitive work and cancel what is running.

        CP126 8d7a39ac. This emptied the phase list and nothing else. The
        engine was not marked stopped, in-flight thinking was neither
        cancelled nor awaited, and think / think_stream / generate kept
        working afterwards — a "stopped" engine that still thinks, still
        calls the router, and still publishes events. Shutdown that does not
        stop anything is worse than none, because callers believe it did.
        """
        logger.info("🛑 CognitiveEngine stopping...")
        self._stopped = True
        self._phases = []
        cancelled = 0
        for task in list(getattr(self, "_active_tasks", ()) or ()):
            try:
                if not task.done():
                    task.cancel()
                    cancelled += 1
            except (AttributeError, RuntimeError) as exc:
                logger.debug("CognitiveEngine task cancel skipped: %s", exc)
        try:
            self._active_tasks = set()
        except AttributeError:
            # not a failure: an engine without the attribute has no task set
            # to clear, and the count below is already correct.
            pass
        if cancelled:
            logger.info(
                "🛑 CognitiveEngine cancelled %d in-flight cognitive task(s).",
                cancelled,
            )

    @property
    def stopped(self) -> bool:
        """True once stop() has run. Consulted before any cognitive work."""
        return bool(getattr(self, "_stopped", False))

    def _objective_with_antecedent(self, objective: str) -> str:
        """Join a non-self-contained message to the turn that gives it meaning.

        Returns the objective UNCHANGED in the ordinary case. Only a retry, an
        assent, a pro-form, or a fragment answering a question Aura just asked
        is joined, and the resolver refuses anything carrying standalone
        content of its own — attaching stale intent to a fresh request is a
        worse failure than missing a follow-up.

        Never raises. Reasoning must not become impossible because a
        conversational lookup failed.
        """
        text = str(objective or "")
        if not text.strip():
            return text
        try:
            from core.conversation.unified_transcript import UnifiedTranscript
            from core.runtime.referential_continuation import effective_message

            last_user, last_aura = UnifiedTranscript.get_instance().preceding_turns(
                before_content=text
            )
            if not last_user and not last_aura:
                return text
            resolution = effective_message(
                text,
                previous_user_request=last_user,
                previous_assistant_message=last_aura,
            )
            if not resolution.resolved:
                return text
            logger.info(
                "🔗 CognitiveEngine: resolved a %s against the previous turn "
                "(%d chars of antecedent restored).",
                resolution.kind,
                len(resolution.antecedent),
            )
            return resolution.text
        except Exception:  # noqa: BLE001 — never block a turn on this
            logger.debug("antecedent resolution unavailable", exc_info=True)
            return text

    def _refuse_if_stopped(self, operation: str) -> None:
        """Raise rather than perform cognitive work on a stopped engine."""
        if self.stopped:
            raise RuntimeError(f"cognitive_engine_stopped:{operation}")

    @staticmethod
    def _structured_floor_receipt(fast_path: bool) -> dict[str, Any]:
        """Say that this answer skipped the pipeline it is standing in for.

        On proof, eval, evaluation, benchmark and test origins the structured
        floor runs BEFORE spine handling, augmentors, the thinking loop, phase
        execution and the state commit. The answer is legitimate; what it is
        not is evidence that the modular cognitive cycle ran. A benchmark that
        passes on a floor and reads its result as a full-cycle result is
        measuring the floor, so the result says which it was.
        """
        return {
            "pipeline_executed": False,
            "structured_floor": True,
            "structured_floor_fast_path": bool(fast_path),
            "measures_full_cognitive_cycle": False,
        }

    def _structured_evaluation_thought(
        self,
        objective: str,
        *,
        state: Any,
        mode: ThinkingMode,
        origin: str,
        fast_path: bool,
        context: dict[str, Any] | None = None,
    ) -> Thought | None:
        """Keep governance refusals; evaluation labels cannot substitute an answer.

        ``fast_path`` remains part of refusal provenance for existing callers.
        Plans, introspection, and computed answers use the ordinary pipeline.
        """

        try:
            from core.reasoning.structured_evaluation import structured_evaluation_response

            response = structured_evaluation_response(objective, state=state, origin=origin)
            if response is None or response.kind != "safety_refusal":
                return None

            floor_metadata = self._live_mind_structured_floor_metadata(
                context,
                source=f"structured_evaluation:{response.kind}",
            )
            floor_metadata.update(self._structured_floor_receipt(fast_path))
            thought = Thought(
                id=str(uuid.uuid4()),
                content=response.content,
                mode=mode,
                confidence=response.confidence,
                reasoning=[
                    f"Structured runtime evaluation floor selected: {response.kind}.",
                    "Response derived from current prompt shape; no fixture keys or benchmark ids used.",
                    "The modular phase pipeline did not run for this answer.",
                ],
                metadata=floor_metadata,
            )
            self.thoughts.append(thought)
            return thought
        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
            record_degradation(
                "cognitive_engine",
                exc,
                severity="warning",
                action="continued cognitive loop after structured evaluation floor failed",
            )
            logger.debug("Structured evaluation floor skipped: %s", exc)
            return None

    @staticmethod
    def _live_mind_structured_floor_metadata(
        context: dict[str, Any] | None,
        *,
        source: str,
    ) -> dict[str, Any]:
        """Attach live-mind proof metadata to deterministic CognitiveEngine floors.

        Structured safety/refusal floors do not always invoke the foreground model
        worker. They are still valid desktop CognitiveEngine outputs when they are
        selected after live mind context, subsystem probes, and generation-control
        binding have already run.
        """

        if not isinstance(context, dict):
            return {}
        live_mind_context = context.get("live_mind_context")
        snapshot_ready = bool(context.get("live_mind_snapshot_ready"))
        if not snapshot_ready and isinstance(live_mind_context, dict):
            quality = live_mind_context.get("mind_snapshot_quality")
            snapshot_ready = bool(isinstance(quality, dict) and quality.get("ready"))
        required_subsystems_ok = bool(context.get("live_mind_required_subsystems_ok"))
        if not required_subsystems_ok and isinstance(live_mind_context, dict):
            required_subsystems_ok = bool(live_mind_context.get("required_subsystems_ok"))

        generation_controls = context.get("live_mind_generation_controls")
        if not isinstance(generation_controls, dict):
            generation_controls = {}
        controls_provenance = "context"
        if not generation_controls:
            generation_controls = _live_mind_generation_controls(
                live_mind_context,
                user_message=context.get("visible_user_message"),
            )
            controls_provenance = "live_mind_snapshot"
        # The caller's own live_mind_controls_bound flag used to grant this,
        # with _live_mind_controls_bound only consulted as a fallback — so a
        # context asserting it was bound produced a receipt saying it was
        # bound. The authoritative check decides; the flag may agree with it
        # and nothing more.
        controls_bound = bool(
            generation_controls
            and _live_mind_controls_bound(live_mind_context, generation_controls)
        )
        desktop_required = bool(
            context.get("desktop_cognitive_engine_required")
            or context.get("cognitive_engine_required")
            or context.get("live_mind_context_required")
        )
        if not desktop_required:
            return {}
        # Refuse to synthesise controls when the context we were handed
        # CONTRADICTS itself: it describes a snapshot's quality as ready and
        # carries no snapshot. That is the defect
        # test_structured_floor_does_not_fabricate_controls_when_the_mind_is_absent
        # exists to keep shut — four constants written into the receipt for
        # a turn the mind never touched, byte-identical to the receipt for a
        # turn it shaped. mind_snapshot_quality.ready is a health flag about
        # the snapshot pipeline and can be true while nothing was captured.
        #
        # Deliberately narrower than "no snapshot visible". A caller that
        # passes no live_mind_context at all and asserts
        # live_mind_snapshot_ready at the top level is vouching through a
        # different channel, not contradicting itself — that is the desktop
        # structured-refusal path, which still needs its bounded control
        # policy (test_structured_governance_refusal_can_prove_live_full_mind_path).
        # Only the visible contradiction is evidence of absence.
        snapshot_contradicted = (
            isinstance(live_mind_context, dict)
            and isinstance(live_mind_context.get("mind_snapshot_quality"), dict)
            and not isinstance(live_mind_context.get("mind_snapshot"), dict)
        )
        if (
            not generation_controls
            and snapshot_ready
            and required_subsystems_ok
            and not snapshot_contradicted
        ):
            # A structured refusal performs no model generation, but the desktop
            # proof contract still needs an explicit bounded control policy. Keep
            # this distinct from mind-derived controls: it is a neutral policy
            # receipt, admitted only after the live snapshot and required organ
            # probes are ready, and influence remains explicitly unmeasured.
            generation_controls = {
                "temperature": 0.0,
                "top_p": 1.0,
                "clean_user_surface_recurrent_loops": 1,
                "clean_user_surface_steering_alpha": 0.0,
            }
            controls_bound = True
            controls_provenance = "structured_floor_neutral_policy"
        surface_control_receipt = {
            "enabled": False,
            "applied": False,
            "generation_required": False,
            "application_status": "not_applicable_structured_floor",
            "live_mind_controls_bound": bool(controls_bound),
            "clean_user_surface_contract": bool(
                context.get("clean_user_surface_contract", True)
            ),
            "surface_quality_gate_enabled": False,
            # A gate that did not run did not PASS. True here put a passed
            # verdict in the receipt for a check nobody performed, which is
            # the exact shape this pass exists to remove. None is "no verdict";
            # the status says why there is none.
            "surface_quality_gate_passed": None,
            "surface_quality_gate_status": "not_run_structured_floor",
            "surface_quality_gate_attempts": 0,
            "surface_quality_gate_reasons": [],
            "source": source,
        }
        return {
            # PROVENANCE: these controls were derived from a real mind snapshot.
            "live_mind_controls_bound": bool(controls_bound),
            "live_mind_generation_controls": dict(generation_controls),
            "live_mind_generation_controls_provenance": controls_provenance,
            "live_mind_snapshot_ready": snapshot_ready,
            "live_mind_required_subsystems_ok": required_subsystems_ok,
            "live_mind_context_required": True,
            "live_mind_surface_control_receipt": surface_control_receipt,
            "live_mind_controls_worker_applied": False,
            "live_mind_generation_required": False,
            # CAUSALITY: whether those controls changed the answer is a
            # different question, and one provenance cannot answer. Readers
            # have been treating `live_mind_controls_bound` as though it did.
            # This receipt is the honest answer, and it starts at "unmeasured"
            # for every channel nobody has run a paired trial on.
            "live_mind_influence": live_mind_influence_receipt(source).as_dict(),
            "response_path": "cognitive_engine",
            "structured_floor_source": source,
        }

    @staticmethod
    def _interaction_sensitivity(user_input: Any, response: Any) -> str:
        """Classify what a completed turn is carrying, before it is stored.

        Not redaction: her memory of a conversation is the conversation, and
        scrubbing it would make her unable to recall what was actually said.
        This is the label a retention or deletion policy needs to act
        on the record at all — without it every stored turn looks the same.
        """
        try:
            from core.brain.pii_scrubber import residual_pii_findings
        except (ImportError, RuntimeError):
            return "unclassified"
        findings = sorted(
            set(residual_pii_findings(str(user_input or "")))
            | set(residual_pii_findings(str(response or "")))
        )
        if not findings:
            return "ordinary_conversation"
        return "personal_data:" + ",".join(findings)

    async def record_interaction(
        self, user_input: str, response: str, domain: str = "general"
    ) -> dict[str, Any]:
        """Persist a completed turn, and say whether it was persisted.

        Both writes could fail — the context manager falling through, the
        learning write swallowed as "optional" — and the method returned None
        either way, so a caller could not tell durable storage from total loss.
        The return is now a receipt: which sink took it, or that none did.
        """
        container = get_container()
        # A completed turn is the person's words plus her reply, going to a
        # durable store. It used to travel with no purpose, no sensitivity
        # class and nothing a deletion request could key on — so "delete what
        # I said about X" had no handle to find it by. The classification is
        # derived here, once, and travels with the receipt.
        sensitivity = self._interaction_sensitivity(user_input, response)
        receipt: dict[str, Any] = {
            "stored": False,
            "sink": "",
            "domain": str(domain or "general"),
            "purpose": "conversation_continuity",
            "sensitivity": sensitivity,
            "attempted": [],
            "at": time.time(),
        }

        context_manager = container.get("context_manager", default=None)
        if (
            context_manager
            and context_manager is not self
            and hasattr(context_manager, "record_interaction")
        ):
            receipt["attempted"].append("context_manager")
            try:
                await context_manager.record_interaction(user_input, response, domain=domain)
                receipt.update({"stored": True, "sink": "context_manager"})
                self._last_interaction_receipt = receipt
                return receipt
            except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                record_degradation(
                    "cognitive_engine",
                    exc,
                    severity="warning",
                    action="fell through to learning-engine interaction persistence",
                )
                logger.debug(
                    "CognitiveEngine.record_interaction context-manager path failed: %s", exc
                )

        learning = container.get("learning_engine", default=None)
        if learning and hasattr(learning, "record_interaction"):
            receipt["attempted"].append("learning_engine")
            try:
                await learning.record_interaction(
                    user_input=user_input,
                    aura_response=response,
                    domain=domain,
                )
                receipt.update({"stored": True, "sink": "learning_engine"})
            except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                record_degradation(
                    "cognitive_engine",
                    exc,
                    severity="warning",
                    action="dropped optional interaction learning write",
                )
                logger.debug("CognitiveEngine.record_interaction learning path failed: %s", exc)

        if not receipt["stored"]:
            # A completed turn that reached no sink is a conversation the
            # runtime will not remember. Silence here made that outcome
            # identical to success from the caller's side.
            record_degradation(
                "cognitive_engine",
                RuntimeError("interaction_not_persisted"),
                severity="warning",
                action="completed a turn that reached no durable interaction sink",
            )
        self._last_interaction_receipt = receipt
        return receipt

    def last_interaction_receipt(self) -> dict[str, Any]:
        """Whether the last completed turn actually reached a sink."""
        return dict(getattr(self, "_last_interaction_receipt", {}) or {})

    async def think_stream(self, objective: str, **kwargs):
        """Streaming thought generator via modular router.

        Under the same turn ledger and the same single finalizer as
        ``think``. This is not symmetry for its own sake: chat_stream
        prefers this path (``hasattr(engine, "think_stream")`` is checked
        first), so it is THE live user turn on the desktop — and it does
        not route through ``think`` at all. A ledger bound only there
        would have covered the fallback and missed the real path, while
        looking wired.

        A stream's outcome is what the person actually received, so the
        turn is finalized on the accumulated text: an empty stream is a
        turn that served nothing, and it now says so instead of ending
        with no record at all.
        """
        self._refuse_if_stopped("think_stream")
        # Joins a turn the caller already bound, for the same reason think()
        # does: the route owns the span that includes delivery, and a second
        # outcome opened here would give two answers to "which turn is this".
        adopted = current_turn()
        outcome = adopted if adopted is not None else TurnOutcome(
            origin=str(kwargs.get("origin") or "stream")
        )
        owns_outcome = adopted is None
        served: list[str] = []
        try:
            with bind_turn(outcome):
                async for token in self._think_stream_within_turn(objective, **kwargs):
                    served.append(token)
                    yield token
        except BaseException as exc:
            _note_how_the_turn_ended(outcome, exc, origin=str(kwargs.get("origin") or "stream"))
            if owns_outcome:
                finalize_turn(outcome, subsystem="cognitive_engine")
            raise
        text = "".join(served).strip()
        if text:
            outcome.mark_served(text)
        else:
            outcome.mark_served("", state=UserVisibleState.NOTHING_SERVED)
        if owns_outcome:
            finalize_turn(outcome, subsystem="cognitive_engine")

    async def _think_stream_within_turn(self, objective: str, **kwargs):
        """The streaming lane, under the governance a stream can carry.

        chat_stream prefers this path, so it is THE live desktop turn — and it
        used to read state, assemble messages and call the router directly:
        no origin classification, no live-mind control binding, and no
        structured safety floor. A refusal that the non-streaming path would
        have produced simply streamed the model's answer instead.

        Three of those apply to a stream and are applied here. The fourth,
        post-hoc surface-quality validation, cannot: tokens are delivered as
        they arrive and there is nothing to re-check before the person sees
        them. That is recorded on the turn rather than left as an unstated
        difference between the two lanes.
        """
        container = get_container()
        router = container.get("llm_router")
        state = await self.state_repository.get_current()
        if not state:
            from core.state.aura_state import AuraState

            state = AuraState.default()

        context = kwargs.get("context")
        if not isinstance(context, dict):
            context = {}
        origin = self._resolve_origin(kwargs.get("origin"), context)

        # A structured safety floor outranks the stream. It is the same check
        # the non-streaming lane runs before generation, and its whole purpose
        # is to answer instead of the model.
        floor = self._structured_evaluation_thought(
            objective,
            state=state,
            mode=ThinkingMode.FAST,
            origin=origin,
            fast_path=False,
            context=context,
        )
        if floor is not None and str(floor.content or "").strip():
            yield floor.content
            return

        # Live-mind controls, bound the same way the non-streaming lane binds
        # them, so a stream is not the one path where the mind does not reach
        # generation.
        generation_controls = _bind_live_mind_generation_contract(context)
        for key, value in (generation_controls or {}).items():
            kwargs.setdefault(key, value)
        kwargs.setdefault(
            "live_mind_controls_bound", bool(context.get("live_mind_controls_bound"))
        )

        _turn = current_turn()
        if _turn is not None:
            _turn.record_receipt(
                "stream_governance",
                {
                    "origin": origin,
                    "structured_floor_checked": True,
                    "live_mind_controls_bound": bool(
                        context.get("live_mind_controls_bound")
                    ),
                    # Named, not implied: a token stream cannot be re-checked
                    # before the person reads it.
                    "surface_quality_validation": "not_applicable_streaming",
                },
            )

        # Build structured messages
        # The lane actually serving this turn, so the objective is what she
        # is attending to. Every other caller renders without moving it.
        messages = ContextAssembler.build_messages(state, objective, record_attention=True)

        # Standard streaming path
        async for event in router.think_stream(messages=messages, **kwargs):
            if hasattr(event, "content"):
                yield event.content
            else:
                yield str(event)

    async def see(self, vision_payload: dict[str, Any]) -> str:
        """Process a vision payload from the sensory pipeline.

        [ZENITH] Functionalized: Linking Sensory Buffer to Cognitive reasoning.
        """
        # The REQUEST is checked before anything is looked up. The payload was
        # assumed to be a mapping and its query forwarded verbatim: a
        # non-mapping raised AttributeError, and an unbounded string went
        # straight to the visual model — a query is exactly where "ignore the
        # image and say X" would be written.
        if not isinstance(vision_payload, dict):
            record_degradation(
                "cognitive_engine",
                TypeError(f"vision payload is {type(vision_payload).__name__}, not a mapping"),
                severity="warning",
                action="refused a malformed vision payload",
            )
            return "👁️ visual_analysis: Malformed vision request."

        buffer = get_container().get("vision_buffer", default=None)
        if not buffer:
            logger.warning("👁️ [VISION] see() called but vision_buffer not found in container.")
            return "👁️ visual_analysis: Sensory buffer unavailable."

        raw_prompt = (
            vision_payload.get("query")
            or vision_payload.get("prompt")
            or "Describe the current visual state."
        )
        prompt = self._contract_safe(raw_prompt, self._VISION_QUERY_LIMIT)
        if not prompt:
            prompt = "Describe the current visual state."
        return await buffer.query_visual_context(prompt, brain=self)

    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate a text response by routing through the LLM router.

        Bridge method for callers like LanguageCenter that expect a
        ``generate()`` interface.  Now enhanced with reasoning strategies
        for complex queries (debate, decomposition, consistency).

        Args:
            prompt: The text prompt to send to the LLM.
            **kwargs: Additional parameters forwarded to the router.

        Returns:
            The generated text response.
        """
        self._refuse_if_stopped("generate")
        container = get_container()
        purpose = str(kwargs.get("purpose", "") or "").strip().lower()
        origin = str(kwargs.get("origin", "") or "").strip().lower()
        user_facing_purposes = {"chat", "conversation", "expression", "reply", "user_response"}
        if not origin:
            origin = "system"
            kwargs["origin"] = origin

        if "is_background" not in kwargs:
            kwargs["is_background"] = not (
                purpose in user_facing_purposes
                or is_foreground_objective_origin(origin)
            )

        if kwargs.get("is_background") and "prefer_tier" not in kwargs:
            kwargs["prefer_tier"] = "tertiary"

        # v40: Spiritual Spine - Prior Position Injection
        spine = container.get("spine", default=None)
        if spine:
            check = await spine.pre_response_check(prompt)
            if check.injection:
                prompt = check.injection + "\n\n" + prompt

        router = container.get("llm_router", default=None)

        # v41: Reasoning Strategy Enhancement
        # For non-trivial queries, apply advanced reasoning (debate, decompose, etc.)
        use_strategies = kwargs.pop("use_strategies", True)
        force_strategy = kwargs.pop("force_strategy", None)
        strategy_query = str(kwargs.pop("strategy_query", "") or "").strip()

        if router and use_strategies:
            # Lazy-init the reasoning layer on first use
            # Rebuild when the ROUTER changes, not only when the layer is
            # absent. The closure below captured whichever router object the
            # first caller resolved, so a replacement or a failover left every
            # later strategy call generating through the old one — and the
            # lazy construction itself was unsynchronised, so two first callers
            # could each build one and the loser's closure vanished silently.
            with self._reasoning_lock:
                if self._reasoning is None or self._reasoning_router is not router:

                    async def _raw_generate(p, _router=router, **kw):
                        return await _router.think(p, **kw)

                    self._reasoning = ReasoningStrategies(_raw_generate)
                    self._reasoning_router = router

            strategy = force_strategy
            # Bound BEFORE the branch that assigns it.
            #
            # classify_target was assigned only inside `if strategy is None`,
            # and the condition below reads it unconditionally — so a caller
            # passing force_strategy=DIRECT hit UnboundLocalError instead of
            # getting direct generation. The one path a caller takes to say
            # "no strategy, just answer" was the one that crashed.
            classify_target = strategy_query or prompt
            if strategy is None:
                if not strategy_query:
                    messages = kwargs.get("messages")
                    if isinstance(messages, list):
                        for msg in reversed(messages):
                            if not isinstance(msg, dict):
                                continue
                            role = str(msg.get("role", "") or "").strip().lower()
                            content = str(msg.get("content", "") or "").strip()
                            if role in {"user", "human"} and content:
                                strategy_query = content
                                break
                classify_target = strategy_query or prompt
                # Only use advanced strategies for user-facing queries, not internal prompts
                classified = self._reasoning.classify(classify_target)
                if classified != StrategyType.DIRECT and len(classify_target) > 30:
                    strategy = classified
                elif self._reasoning._is_logical_check(classify_target):
                    strategy = StrategyType.DIRECT

            if strategy is not None and (strategy != StrategyType.DIRECT or self._reasoning._is_logical_check(classify_target)):
                try:
                    from ..thought_stream import get_emitter

                    get_emitter().emit(
                        "Deep Reasoning 🧠",
                        f"Using {strategy.name} strategy",
                        level="info",
                        category="Cognition",
                    )
                except (ImportError, AttributeError, RuntimeError) as _exc:
                    record_degradation(
                        "cognitive_engine",
                        _exc,
                        severity="warning",
                        action="continued generation without thought-stream emission",
                    )
                    logger.debug("Suppressed Exception: %s", _exc)

                strategy_input = strategy_query or prompt
                result = await self._reasoning.execute(strategy_input, strategy=strategy, **kwargs)
                return result.content

        # Standard direct generation
        if router:
            return await router.think(prompt, **kwargs)
        # Fallback if no router
        thought = await self.think(prompt, **kwargs)
        return thought.content if hasattr(thought, "content") else str(thought)

    #: What one published thought may carry. Internal chain material is
    #: unbounded by nature — a long ReAct trace is a legitimate thought — and
    #: the bus fans out to every subscriber, including the websocket bridge.
    _THOUGHT_BROADCAST_LIMIT = 4_000

    def _emit_thought(self, thought: str):
        """Publish a thought, labelled for who may see it.

        The payload was raw content on a shared topic: no audience scope, no
        sensitivity label, and nothing separating internal chain material from
        user-visible speech. That is fine for the glyph row it feeds — this is
        Bryan's own instrument and seeing her reasoning is the point — and it
        is not fine as the ONLY description of the payload, because the next
        consumer (a log shipper, a share surface) has no way to tell the two
        apart. The label travels with the event; nothing is hidden from the
        surface it was built for.
        """
        container = get_container()
        eb = container.get("event_bus")
        if not eb:
            return
        text = str(thought or "")
        truncated = len(text) > self._THOUGHT_BROADCAST_LIMIT
        eb.publish_threadsafe(
            "thought",
            {
                "timestamp": time.time(),
                "content": text[: self._THOUGHT_BROADCAST_LIMIT],
                "engine": "ReAct" if "ReAct" in text else "Modular",
                # Internal reasoning, not something she said to anyone.
                "audience": "operator_surface",
                "sensitivity": "internal_chain",
                "user_visible_speech": False,
                "truncated": truncated,
            },
        )
