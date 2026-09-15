"""Conversational amplifier — best-of-N taste-selection + self-critique-revise.

The analogue of the reasoning amplifier for the UNVERIFIABLE dimensions (wit, voice,
creativity, presence). There is no truth-engine for "good conversation", so instead of
a correctness verifier we use a personalized TasteModel as the selector:

    draft + N alternatives  →  score each by taste  →  optional self-revise  →  best

This harvests the median→best-of-N gap (the model can already be witty; this makes it
reliably be) and the model-as-critic gap (a model spots "this is generic/hedge-y" even
in its own draft). Personalized: the TasteModel learns Bryan's preference from reactions.

Live, causal, bounded, fail-open. Imperative actions and verifiable reasoning turns are
excluded (the tool path and reasoning amplifier own those). Records the sent response's
features so the taste model can learn when the reaction arrives.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from core.brain.response_quality import extract_features, select_best
from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.ConversationalAmplifier")

GenerateFn = Callable[[str, float], Awaitable[str]]
_ALT_TEMPS = [0.8, 1.0, 0.6, 1.1]
_USER_ORIGINS = frozenset({"user", "voice", "gui", "ws", "websocket", "api", "direct", "external"})


def _flag_on(name: str, default: str = "1") -> bool:
    return str(os.getenv(name, default)).strip().lower() not in {"0", "false", "off", "no"}


def is_conversationally_amplifiable(objective: str, origin: str) -> bool:
    """True for substantive conversational turns (not actions, not verifiable reasoning)."""
    if origin not in _USER_ORIGINS:
        return False
    q = str(objective or "").strip()
    if len(q.split()) < 3:
        return False
    # CP126 (high): "Classifier import failure admits actions and verifiable
    # tasks. If the action/reasoning classifier cannot import, the function
    # falls through to True for any substantive user-origin text."
    #
    # It was `except ImportError: pass` followed by `return True`, so the
    # one component that knows which turns are EXCLUDED could vanish and the
    # answer became "amplify everything" — creative rewrites applied to
    # imperatives and to verifiable reasoning the module says it excludes.
    #
    # Amplification is an enhancement. Declining it costs a plainer reply;
    # applying it to an action request rewrites an instruction. Without the
    # classifier there is no way to know which this is, so it declines.
    try:
        from core.brain.reasoning_amplifier_v2 import is_action_request, is_amplifiable
    except (ImportError, AttributeError, RuntimeError) as exc:
        record_degradation(
            "conversational_amplifier",
            exc,
            severity="warning",
            action="declined to amplify because the exclusion classifier was unavailable",
            enforce_failure_policy=False,
        )
        return False
    try:
        if is_action_request(q) or is_amplifiable(q) is not None:
            return False
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        record_degradation(
            "conversational_amplifier",
            exc,
            severity="warning",
            action="declined to amplify because the exclusion classifier raised",
            enforce_failure_policy=False,
        )
        return False
    return True


@dataclass
class ConversationResult:
    answer: str
    n_candidates: int = 1
    revised: bool = False
    selected_score: float = 0.0
    features: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_candidates": self.n_candidates,
            "revised": self.revised,
            "selected_score": round(self.selected_score, 4),
            "features": {k: round(v, 3) for k, v in self.features.items()},
        }


def _alt_prompt(user_message: str) -> str:
    return (
        f"{user_message}\n\n"
        "[Respond as Aura: take a clear stance, be specific, use contractions, stay casual. "
        "No filler, no 'I'd be happy to', no closing question that punts it back. Have a reaction.]"
    )


def _revise_prompt(draft: str, user_message: str) -> str:
    return (
        "Improve this reply. Keep its meaning and voice, but make it sharper: lead with a "
        "stance, add one specific detail or callback, cut every hedge and filler phrase, and "
        "do NOT end on a question unless it's the only thing worth saying. Return ONLY the "
        f"improved reply.\n\nUser said: {user_message}\n\nDraft: {draft}"
    )


async def amplify_conversation(
    draft: str,
    *,
    generate: GenerateFn,
    objective: str = "",
    user_message: str = "",
    grounding_tokens: set[str] | None = None,
    word_budget: int = 0,
    n: int = 3,
    time_budget_s: float = 20.0,
    revise: bool = True,
    conversation_id: str = "default",
    lived_analogue: float | None = None,
) -> ConversationResult:
    """Generate alternatives, taste-select the best, optionally self-revise.

    Fail-open, structurally. CP126 (high): "Fail-open claim excludes several
    unhandled failure points. Numeric coercion, select_best, feature
    extraction, and result serialization are outside protected blocks, while
    model handlers omit I/O, timeout, connection, overflow, and custom
    provider failures. These errors abort the caller instead of returning
    the draft."

    The docstring said fail-open and the body implemented it in patches —
    every model call was wrapped, and `int(n)`, `float(time_budget_s)`,
    select_best and extract_features were not. A caller that passed a bad
    budget, or a taste model that raised, lost the whole turn rather than
    getting the plain draft back.

    Scattering more try/except would leave the same gap one refactor later.
    The guarantee belongs at the boundary: this wrapper always returns a
    ConversationResult, and the worst case is the draft the caller already
    had. Enhancement failures may cost the enhancement and never the answer.
    """
    safe_draft = str(draft or "").strip()
    try:
        return await _amplify_conversation_inner(
            safe_draft,
            generate=generate,
            objective=objective,
            user_message=user_message,
            grounding_tokens=grounding_tokens,
            word_budget=word_budget,
            n=n,
            time_budget_s=time_budget_s,
            revise=revise,
            conversation_id=conversation_id,
            lived_analogue=lived_analogue,
        )
    except asyncio.CancelledError:
        # Cancellation is the caller's decision, not a failure to absorb.
        raise
    except Exception as exc:  # noqa: BLE001 - the boundary IS the contract
        record_degradation(
            "conversational_amplifier",
            exc,
            severity="warning",
            action="returned the unamplified draft so the turn survived",
            enforce_failure_policy=False,
        )
        return ConversationResult(
            answer=safe_draft, n_candidates=1 if safe_draft else 0,
        )


async def _amplify_conversation_inner(
    draft: str,
    *,
    generate: GenerateFn,
    objective: str = "",
    user_message: str = "",
    grounding_tokens: set[str] | None = None,
    word_budget: int = 0,
    n: int = 3,
    time_budget_s: float = 20.0,
    revise: bool = True,
    conversation_id: str = "default",
    lived_analogue: float | None = None,
) -> ConversationResult:
    """The amplification itself. Anything it raises becomes the draft."""
    draft = str(draft or "").strip()
    um = user_message or objective
    # Her best recall match for what they said, when this turn's recall was
    # about it. See `lived_analogue` in core/brain/response_quality.py.
    feats_kw = {
        "user_message": um,
        "grounding_tokens": grounding_tokens or set(),
        "word_budget": word_budget,
        "lived_analogue": lived_analogue,
    }

    if not _flag_on("AURA_CONVERSATIONAL_AMPLIFIER"):
        return ConversationResult(answer=draft, n_candidates=1 if draft else 0)

    start = time.monotonic()
    deadline = start + max(3.0, float(time_budget_s))
    candidates: list[str] = [draft] if draft else []

    async def _one(temp: float) -> str:
        if time.monotonic() >= deadline:
            return ""
        try:
            return str(await generate(_alt_prompt(um), temp) or "").strip()
        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
            record_degradation("conversational_amplifier_generate", exc)
            return ""

    want = max(0, int(n) - len(candidates))
    if want:
        try:
            alts = await asyncio.gather(*[_one(t) for t in _ALT_TEMPS[:want]])
            candidates.extend(a for a in alts if a)
        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
            record_degradation("conversational_amplifier_gather", exc)

    if not candidates:
        return ConversationResult(answer=draft, n_candidates=0)

    best, ranked = select_best(candidates, **feats_kw)
    n_cand = len(candidates)
    if not best:
        return ConversationResult(answer=draft or "", n_candidates=n_cand)

    revised = False
    best_score = ranked[0][1] if ranked else 0.0
    if revise and time.monotonic() < deadline:
        try:
            improved = str(await generate(_revise_prompt(best, um), 0.6) or "").strip()
        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
            record_degradation("conversational_amplifier_revise", exc)
            improved = ""
        if improved and improved != best:
            winner, ranked2 = select_best([best, improved], **feats_kw)
            if winner == improved:
                best, best_score, revised = improved, ranked2[0][1], True

    features = extract_features(best, **feats_kw)
    # Stash the sent response's features so the taste model can learn from the reaction.
    try:
        from core.brain.conversation_outcome import record_pending_response

        # CP126 dea1d2f1: keyed, so a reply in one conversation cannot be
        # applied to the response sent in another.
        record_pending_response(best, features, conversation_id=conversation_id)
    except (ImportError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
        record_degradation("conversational_amplifier_pending", exc)

    logger.info(
        "🗣️ [ConvAmplify] n=%d revised=%s score=%.2f → %s",
        n_cand, revised, best_score, "adopted" if best != draft else "kept draft",
    )
    return ConversationResult(
        answer=best, n_candidates=n_cand, revised=revised, selected_score=best_score, features=features
    )
