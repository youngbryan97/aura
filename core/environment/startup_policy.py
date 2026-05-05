"""Generic environment startup prompt policy.

Startup prompts are lifecycle concerns, not task strategy.  A fresh benchmark
run may need to replace a stale suspended session, while ordinary risky
confirmation prompts should still fail closed.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class StartupPromptDecision:
    action: str
    response: str | None
    reason: str
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class StartupPromptPolicy:
    """Resolve lifecycle prompts before an environment run enters policy control."""

    _STALE_SESSION_PATTERNS = (
        re.compile(r"\balready\b.{0,80}\bin progress\b", re.IGNORECASE | re.DOTALL),
        re.compile(r"\bdestroy\b.{0,80}\bold\b", re.IGNORECASE | re.DOTALL),
        re.compile(r"\breplace\b.{0,80}\bstale\b", re.IGNORECASE | re.DOTALL),
    )
    _SAFE_SETUP_TOKENS = (
        "is this ok",
        "is this okay",
        "confirm setup",
        "confirm settings",
        "shall i pick",
        "pick for you",
        "choose for you",
        "use default",
        "default settings",
        "default profile",
    )

    def decide(self, prompt_text: str, *, fresh_start_required: bool = False) -> StartupPromptDecision:
        text = str(prompt_text or "")
        lower = text.lower()
        if "--more--" in lower or "space to continue" in lower:
            return StartupPromptDecision(
                action="acknowledge_startup_message",
                response=" ",
                reason="startup continuation prompt requires acknowledgement",
                confidence=0.9,
            )
        if any(token in lower for token in ("press return", "hit return", "return to continue", "enter to continue")):
            return StartupPromptDecision(
                action="acknowledge_startup_message",
                response="\r",
                reason="startup continuation prompt requires return acknowledgement",
                confidence=0.9,
            )
        has_yes_no = bool(re.search(r"\[(?:y/?n|yn|ynq)\]", text, re.IGNORECASE))
        stale = any(pattern.search(text) for pattern in self._STALE_SESSION_PATTERNS)
        if fresh_start_required and has_yes_no and stale:
            return StartupPromptDecision(
                action="replace_stale_session",
                response="y",
                reason="fresh run requires clearing stale environment session",
                confidence=0.9,
            )
        if has_yes_no and any(token in lower for token in self._SAFE_SETUP_TOKENS):
            return StartupPromptDecision(
                action="accept_default_startup_setup",
                response="y",
                reason="startup setup prompt offers a default configuration",
                confidence=0.82,
            )
        if has_yes_no:
            return StartupPromptDecision(
                action="decline_confirmation",
                response="n",
                reason="startup confirmation is not classified as stale-session replacement",
                confidence=0.7,
            )
        return StartupPromptDecision(
            action="no_prompt",
            response=None,
            reason="no startup prompt requiring response",
            confidence=0.5,
        )


__all__ = ["StartupPromptDecision", "StartupPromptPolicy"]
