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

import logging
from typing import Any, Sequence

from core.runtime.errors import record_degradation

logger = logging.getLogger(__name__)

#: A move decided under this much delay stops being a move in a live loop.
DECISION_BUDGET_S = 20.0
#: Deliberation this long is for a decision that carries real weight.
DELIBERATE_BUDGET_S = 45.0
#: The role given to the model when it is choosing rather than answering.
CHOOSING_ROLE = (
    "You are choosing Aura's next move. The evidence lists the goal, what is "
    "visible now, and every move that is really available. Name the one move "
    "you choose and say why in one sentence."
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


def generator(*, origin: str = "agency_next_move", tier: str = "primary"):
    """A generate function over the resident model, shaped for the amplifier."""

    async def generate(prompt: str, temperature: float) -> str:
        router = _router()
        if router is None:
            raise RuntimeError("no model router is registered")
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
            # Deciding a move is not answering a person. Without this the
            # user-surface reply contract is applied to the decision and
            # grades it against a question invented from its own prompt:
            # measured live, "left" was rejected as arithmetic_answer_missing.
            internal_inference=True,
        )
        if isinstance(out, str):
            return out
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
):
    """Her reasoning, as the ``think`` callable a deliberation asks for.

    Failures are raised rather than swallowed. The deliberation treats an
    unreachable mind as a reason to stop, and turning that into a quiet
    default would put her back to acting without deciding.
    """
    produce = generate or generator(origin=origin)

    async def think(objective: str, evidence: Sequence[str]) -> str:
        from core.brain.reasoning_amplifier_v2 import amplify_turn  # noqa: PLC0415

        amplified = await amplify_turn(
            objective,
            produce,
            task_type="planning",
            evidence=list(evidence),
            risk_level=risk_level,
            time_budget_s=time_budget_s,
        )
        return amplified.answer

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
        return result.answer

    return think


def reasoning_for(stakes: float):
    """Pick how hard to think by how much rides on the move."""
    return deep_reasoning() if stakes >= 0.7 else her_reasoning()


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
