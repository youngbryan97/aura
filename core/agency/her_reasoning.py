"""Wiring a decision about what to do next into the reasoning she already has.

:mod:`core.agency.deliberate_action` deliberately knows nothing about how she
thinks — it takes a ``think`` callable so it stays testable and so any caller
can supply a different mind. This module supplies the real one.

The route is the same verifier-backed amplifier that already carries her hard
answers, run with ``task_type="planning"``. Amplification was held off action
requests on purpose: response generation says so in as many words, that action
requests stay owned by tool dispatch. The consequence was that she reasoned
carefully about everything except what she was about to do.

The evidence handed to the amplifier is measured — the goal, what is visible,
which moves exist, what those moves did before. None of it is written to push
the answer one way, because a decision steered by its own prompt cannot be
evidence of anything later.
"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from typing import Any

from core.runtime.errors import record_degradation

logger = logging.getLogger(__name__)

#: A move decided under this much delay stops being a move in a live loop.
#:
#: Measured live: a board move was taking about thirty seconds, because a
#: choice between four named options was being generated with the same budget
#: as an essay. A decision is not a composition.
DECISION_BUDGET_S = 8.0
#: How much room a choice needs to be said.
#:
#: Naming one option out of a closed set, with a sentence of reason, is tens
#: of tokens. Asking for five hundred does not make the answer better; it
#: makes the model keep writing after it has answered, and a loop that has to
#: move once a second waits for all of it.
CHOICE_TOKENS = 96
#: Deliberation this long is for a decision that carries real weight.
DELIBERATE_BUDGET_S = 45.0
#: The role given to the model when it is choosing rather than answering.
CHOOSING_ROLE = (
    "You are choosing Aura's next move. The evidence lists the goal, what is "
    "visible now, and every move that is really available. Name the move you "
    "choose and say why in one sentence. If the next few moves follow from it "
    "and the position is stable, name them in order."
)


def _router() -> Any:
    """The model router, or None when nothing is registered yet.

    A container that has not been built is a real state — at boot, in a test,
    on an instance whose model has been evicted. It comes back as None so the
    deliberation reports a mind out of reach and stops, rather than raising
    a container error out of the middle of a goal loop.
    """
    from core.container import get_container  # noqa: PLC0415
    from core.exceptions import ContainerError  # noqa: PLC0415
    from core.service_names import ServiceNames  # noqa: PLC0415

    try:
        return get_container().get(ServiceNames.LLM_ROUTER)
    except (ContainerError, KeyError, AttributeError, RuntimeError):
        return None


def time_this_question_needs(prompt: str, max_tokens: int, floor_s: float) -> float:
    """How long to allow, from the size of what is asked and what is wanted back.

    A budget picked as a constant has to fit both "left" and "how should I
    play this", and fits neither. Live 2026-08-26: every approach question in
    a game of 2048 timed out at the eight seconds that suit a one-word move,
    so she played the whole game with no plan and nothing said why.

    Never shortens what the caller asked for — only lengthens it to what the
    machine has been measured needing.
    """
    try:
        from core.brain.llm.mlx_client import time_a_prompt_needs  # noqa: PLC0415

        needed = time_a_prompt_needs(len(str(prompt or "")), max_tokens)
    except (ImportError, AttributeError, TypeError, ValueError):
        return float(floor_s)
    return max(float(floor_s), needed)


def generator(
    *,
    origin: str = "agency_next_move",
    tier: str = "primary",
    max_tokens: int = CHOICE_TOKENS,
    timeout_s: float = DECISION_BUDGET_S,
):
    """A generate function over the resident model, shaped for the amplifier."""

    async def generate(prompt: str, temperature: float) -> str:
        router = _router()
        if router is None:
            raise RuntimeError("no model router is registered")
        allow_s = time_this_question_needs(prompt, max_tokens, timeout_s)
        messages = [
            {"role": "system", "content": CHOOSING_ROLE},
            {"role": "user", "content": prompt},
        ]
        out = await router.think(
            messages=messages,
            priority=1.0,
            origin=origin,
            purpose="action_deliberation",
            prefer_tier=tier,
            is_background=False,
            foreground_request=True,
            allow_cloud_fallback=False,
            max_tokens=max_tokens,
            # A decision on a live loop carries its own deadline.
            #
            # The endpoint's own budget is 103 seconds, which is right for a
            # hard answer somebody is waiting on and wrong for a move in a
            # game: measured live, one generation timed out at the endpoint
            # and the whole run stood still for it. Past this, deciding from
            # evidence is not just faster, it is the only thing that is still
            # a decision about the board in front of her.
            timeout=allow_s,
            # Deciding a move is not answering a person. Without this the
            # user-surface reply contract is applied to the decision and
            # grades it against a question invented from its own prompt:
            # measured live, "left" was rejected as arithmetic_answer_missing.
            internal_inference=True,
        )
        if isinstance(out, str):
            return out
        if out is None:
            # A mind that answered nothing is not a broken mind.
            #
            # The think path declares `str | None` and None is its way of
            # saying it has no answer this time — a refusal, an exhausted
            # local lane, a request it declined. Raising here turned an
            # ordinary answer into a subsystem fault: measured live, 51 of
            # them in half an hour, each one recorded MARGINAL and driving
            # her own affect to frustration=1.00, depletion=0.49, strain.
            #
            # Callers already handle an empty answer properly. A decision
            # falls back to evidence; an approach is simply not stated this
            # time and is asked for again later. A real transport failure
            # still raises, because that is a different thing and the
            # deliberation should stop for it.
            return ""
        for field in ("content", "text", "answer", "response"):
            value = getattr(out, field, None) or (out.get(field) if isinstance(out, dict) else None)
            if isinstance(value, str) and value.strip():
                return value
        raise RuntimeError(f"model returned nothing usable ({type(out).__name__})")


    return generate


def her_reasoning(
    *,
    generate: Any = None,
    time_budget_s: float = DECISION_BUDGET_S,
    risk_level: str = "normal",
    origin: str = "agency_next_move",
    max_tokens: int = CHOICE_TOKENS,
):
    """Her reasoning, as the ``think`` callable a deliberation asks for.

    Failures are raised rather than swallowed. The deliberation treats an
    unreachable mind as a reason to stop, and turning that into a quiet
    default would put her back to acting without deciding.
    """
    produce = generate or generator(origin=origin, max_tokens=max_tokens, timeout_s=time_budget_s)

    async def think(objective: str, evidence: Sequence[str]) -> str:
        from core.brain.reasoning_amplifier_v2 import amplify_turn  # noqa: PLC0415

        asked = "\n".join([objective, *evidence])
        allow_s = time_this_question_needs(asked, max_tokens, time_budget_s)
        amplified = await asyncio.wait_for(
            amplify_turn(
                objective,
                produce,
                task_type="planning",
                evidence=list(evidence),
                risk_level=risk_level,
                time_budget_s=allow_s,
            ),
            timeout=allow_s + 4.0,
        )
        answer = str(amplified.answer or "").strip()
        if not answer:
            # Nothing came back, so say that rather than passing emptiness on.
            #
            # The amplifier absorbs a generation failure and returns an
            # unverified answer with no candidates in it. A caller reading only
            # the text cannot tell that apart from a reply that named nothing,
            # and the difference is the whole diagnosis: measured live while
            # the resident worker was still starting, a pursuit reported "she
            # named no available move" six times over. She had not been asked.
            raise RuntimeError("her reasoning produced nothing (no candidate survived)")
        return answer

    return think


def deep_reasoning(*, budget: int = 2, timeout_s: float = DELIBERATE_BUDGET_S):
    """Her slower route, for a decision worth sharpening the question first.

    Deep deliberation refines what is being asked before answering it and can
    run the recursive latent cortex on the resident model, so a move that
    commits her to something expensive gets the same treatment as a hard
    question.
    """

    async def think(objective: str, evidence: Sequence[str]) -> str:
        from core.brain.deep_deliberation import get_deep_deliberation  # noqa: PLC0415

        engine = get_deep_deliberation()
        result = await engine.deliberate(
            objective,
            context={"evidence": list(evidence)},
            budget=budget,
            timeout_s=timeout_s,
            foreground_request=True,
        )
        answer = str(result.answer or "").strip()
        if not answer:
            raise RuntimeError("her deliberation produced nothing")
        return answer

    return think


def quick_reasoning(*, origin: str = "agency_next_move", max_tokens: int = CHOICE_TOKENS):
    """One pass at the model, for a decision that has to keep up with a loop.

    The amplifier's value is agreement between several attempts and a
    verifier's opinion on the result. That is worth the wall-clock on a hard
    answer and wrong on a move in a game: a loop acting once a second cannot
    spend several generations per decision, and a choice between four named
    options has little for a verifier to check.

    So effort follows what rides on the decision — one pass here, agreement
    when something is at stake, a sharpened question when a lot is.
    """
    produce = generator(origin=origin, max_tokens=max_tokens, timeout_s=DECISION_BUDGET_S)

    async def think(objective: str, evidence: Sequence[str]) -> str:
        # Bounded here as well as at the endpoint: a client that never
        # returns is the same to this loop as one that returns nothing, and
        # only one of those is visible without a deadline of its own.
        said = await asyncio.wait_for(
            produce("\n".join([objective, *evidence]), 0.3), timeout=DECISION_BUDGET_S + 2.0
        )
        answer = str(said or "").strip()
        if not answer:
            raise RuntimeError("her reasoning produced nothing (no text came back)")
        return answer

    return think


def reasoning_for(stakes: float):
    """Pick how hard to think by how much rides on the move.

    Three tiers rather than two, because the middle one was being paid for
    on every routine step of a live loop.
    """
    if stakes >= 0.7:
        return deep_reasoning()
    if stakes >= 0.45:
        return her_reasoning()
    return quick_reasoning()


def narrate_through(voice: Any, line: str) -> bool:
    """Say a line through whatever expression path is wired, and report if it landed."""
    if not line or voice is None:
        return False
    try:
        speak = getattr(voice, "offer_utterance", None) or getattr(voice, "say", None)
        if speak is None:
            return False
        speak(line)
        return True
    except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
        record_degradation("her_reasoning", exc, action="narrate a move")
        return False
