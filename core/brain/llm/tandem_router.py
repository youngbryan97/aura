"""
core/brain/llm/tandem_router.py — opt-in tandem routing for HealthAwareLLMRouter.

should_use_tandem() picks tandem vs solo via heuristic.
attach_tandem() exposes a TandemKame at router.tandem; existing router behaviour
is unchanged unless callers opt in via task_type='tandem' / explicit=True.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, AsyncIterator, Optional

from core.brain.llm.tandem_kame import TandemKame
from core.brain.llm.tandem_signal_bus import TandemSignalBus

logger = logging.getLogger("Brain.TandemRouter")

_DEEP_INTENTS = frozenset({
    "analysis", "analyze", "planning", "plan", "research",
    "code", "coding", "debug", "debugging", "explain_deep", "explain",
    "multi_step", "multistep", "reasoning", "math", "task",
})
_TRIVIAL_INTENTS = frozenset({
    "casual", "chitchat", "greeting", "smalltalk", "ack", "acknowledgement", "yesno",
})
_DEEP_KEYWORDS = (
    "why", "how", "explain", "compare", "design", "debug", "trace",
    "prove", "derive", "analyze", "refactor", "optimize", "plan",
)
_LONG_CHAR_THRESHOLD = 280
_LONG_WORD_THRESHOLD = 40


@dataclass
class TandemDecision:
    use_tandem: bool
    reason: str

    def __bool__(self) -> bool:
        return self.use_tandem


def should_use_tandem(
    prompt: str, *, intent: Optional[str] = None,
    task_type: Optional[str] = None, explicit: Optional[bool] = None,
) -> TandemDecision:
    if explicit is True:
        return TandemDecision(True, "explicit_opt_in")
    if explicit is False:
        return TandemDecision(False, "explicit_opt_out")
    if task_type and task_type.lower() == "tandem":
        return TandemDecision(True, "task_type=tandem")
    intent_l = (intent or "").lower().strip()
    if intent_l in _TRIVIAL_INTENTS:
        return TandemDecision(False, f"trivial_intent:{intent_l}")
    if intent_l in _DEEP_INTENTS:
        return TandemDecision(True, f"deep_intent:{intent_l}")
    text = (prompt or "").strip()
    if len(text) >= _LONG_CHAR_THRESHOLD:
        return TandemDecision(True, "long_prompt_chars")
    if len(re.findall(r"\S+", text)) >= _LONG_WORD_THRESHOLD:
        return TandemDecision(True, "long_prompt_words")
    if any(kw in text.lower() for kw in _DEEP_KEYWORDS):
        return TandemDecision(True, "deep_keyword")
    return TandemDecision(False, "default_solo")


class TandemFastAdapter:
    """Wrap a HealthAwareLLMRouter so TandemKame can call astream()."""

    def __init__(self, inner: Any, *, prefer_tier: Optional[str] = "fast"):
        self.inner = inner
        self.prefer_tier = prefer_tier

    async def astream(self, prompt: str, *, system: Optional[str] = None) -> AsyncIterator[str]:
        for attr in ("astream", "stream", "stream_response", "agen_stream"):
            fn = getattr(self.inner, attr, None)
            if fn is None: continue  # noqa: E701
            try:
                result = fn(prompt, system_prompt=system, prefer_tier=self.prefer_tier)
            except TypeError:
                try: result = fn(prompt)
                except Exception: continue  # noqa: BLE001,E701
            if hasattr(result, "__aiter__"):
                async for tok in result:
                    if tok: yield tok  # noqa: E701
                return
        gen = getattr(self.inner, "generate", None)
        if gen is None:
            raise TypeError("TandemFastAdapter inner has no astream() or generate()")
        try: text = await gen(prompt, system_prompt=system, prefer_tier=self.prefer_tier)
        except TypeError: text = await gen(prompt)  # noqa: E701
        if text: yield str(text)  # noqa: E701


def attach_tandem(
    router: Any, fast_client: Any, slow_client: Any,
    *, signal_bus: Optional[TandemSignalBus] = None, **kame_kwargs: Any,
) -> TandemKame:
    """Attach a TandemKame at router.tandem (opt-in for callers)."""
    tandem = TandemKame(fast_client, slow_client, signal_bus=signal_bus, **kame_kwargs)
    try:
        setattr(router, "tandem", tandem)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not attach tandem to router: %s", exc)
    return tandem


async def respond_via_tandem(
    tandem: TandemKame, prompt: str, *,
    system: Optional[str] = None, on_signal: Optional[Any] = None,
) -> AsyncIterator[str]:
    async for chunk in tandem.respond(prompt, system=system, on_signal=on_signal):
        yield chunk


def explain_decision(prompt: str, **kwargs: Any) -> str:
    d = should_use_tandem(prompt, **kwargs)
    return f"tandem={'on' if d.use_tandem else 'off'} reason={d.reason}"


__all__ = (
    "TandemDecision", "TandemFastAdapter", "attach_tandem",
    "explain_decision", "respond_via_tandem", "should_use_tandem",
)
