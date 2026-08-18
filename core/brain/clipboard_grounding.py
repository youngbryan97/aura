"""What is actually on the clipboard, for turns that ask about it.

LIVE 2026-08-17: "what's on my clipboard right now?" was answered "I can use
the clipboard — computer_use, desktop_task, os_automation are registered and
enabled right now... I can only work with the information you provide me during
our conversation." The clipboard held BUILD-7741-verify.

The capability was never missing. Nothing read it, so she had nothing to say,
and the honest-sounding half of that reply ("I can only work with the
information you provide") was false in the same breath as the true half.

This is the same shape as the file read: the reading has to exist before the
answer can contain it. Asking the model to go and look is not a mechanism.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any

CLIPBOARD_HEADER = "## WHAT IS ON THE CLIPBOARD"

#: Enough to answer with, bounded so a copied document does not become the
#: prompt. A person asking "what's on my clipboard" wants to recognise it.
CLIPBOARD_CHAR_BUDGET = 2000

#: Reading the clipboard is a privacy-relevant act, so it happens only when the
#: turn is actually about the clipboard — never as ambient context.
_ASKS_ABOUT_CLIPBOARD = re.compile(
    # "what did I just copy?" asks about the clipboard without naming it, and
    # is how a person actually phrases it half the time.
    r"\b(?:clip\s?board|paste(?:board)?|copied)\b"
    r"|\b(?:just|last)\s+cop(?:y|ied)\b"
    r"|\bdid\s+i\s+(?:just\s+)?copy\b",
    re.IGNORECASE,
)


#: Putting something ON the clipboard is a write, not a question about it.
#: Reading someone's clipboard because they asked you to write to it is both
#: unnecessary and a small privacy violation.
_CLIPBOARD_WRITE_RE = re.compile(
    r"\b(?:put|copy|place|set|write|save)\b[^.?!]{0,60}?\b(?:on|onto|to|in|into)\s+"
    r"(?:my|the)\s+clip\s?board\b"
    r"|\bclip\s?board\s+(?:it|that|this)\b",
    re.IGNORECASE,
)


def asks_about_clipboard(text: Any) -> bool:
    """True when the turn is about clipboard CONTENTS, not about writing to it."""

    body = str(text or "")
    if _CLIPBOARD_WRITE_RE.search(body):
        return False
    return bool(_ASKS_ABOUT_CLIPBOARD.search(body))


async def clipboard_block(user_prompt: Any, *, timeout_s: float = 2.5) -> str:
    """The clipboard block for this turn, or "" when it does not apply.

    Returns a block naming the ABSENCE when the clipboard is empty, because
    "it's empty" and "I couldn't look" are different answers and only one of
    them is true.
    """

    if not asks_about_clipboard(user_prompt):
        return ""
    try:
        from core.capabilities.clipboard_manager import get_clipboard_manager

        content = await asyncio.wait_for(
            get_clipboard_manager().get(), timeout=max(0.1, float(timeout_s))
        )
    except (ImportError, AttributeError, OSError, RuntimeError, TimeoutError):
        return ""
    text = str(content or "")
    if not text.strip():
        return f"{CLIPBOARD_HEADER}\nThe clipboard is empty."
    truncated = len(text) > CLIPBOARD_CHAR_BUDGET
    body = text[:CLIPBOARD_CHAR_BUDGET]
    suffix = f"\n[{len(text)} characters total; showing the first {CLIPBOARD_CHAR_BUDGET}]" if truncated else ""
    return f"{CLIPBOARD_HEADER}\n{body}{suffix}"
