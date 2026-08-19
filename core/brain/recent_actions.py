"""What she actually just did, so the next turn cannot invent it.

Asked to build a clean-room 2048 and put it on the Desktop, the attempt failed
and she said so honestly. Asked the very next turn whether it was playable, she
answered:

    "When you run it, the board pops up and you click cells to reveal numbers.
     If you hit a mine, the game shows you which squares had mines and ends."

That is Minesweeper, and there was no file. Nothing was dishonest about it: the
attempt's receipt lived in the intention ledger, the conversation history held
only her own sentence about having tried, and a question about how the artifact
behaves has no answer in the transcript — so the model wrote the most plausible
"how a small game behaves" paragraph available to it.

The ledger already records the truth. ``IntentionLoop`` keeps a Say-Do-Observe
record per attempt: the tools invoked, whether they succeeded, and the observed
outcome. It just was not in front of her when she was asked. This puts the last
few there, in the same spirit as the clock and the instruments: cheap, read from
a real source, and omitted entirely rather than guessed at.
"""
from __future__ import annotations

import re
import time
from typing import Any

from core.runtime.errors import record_degradation

_RECOVERABLE = (RuntimeError, AttributeError, TypeError, ValueError, OSError, ImportError, KeyError)

RECENT_ACTIONS_HEADER = "## WHAT YOU ACTUALLY JUST DID"

# Older than this and it is no longer "just now" — it belongs to memory, and
# quoting it as recent activity is its own species of confabulation.
_RECENCY_WINDOW_S = 45 * 60
_MAX_ACTIONS = 4


def _ago(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 90:
        return f"{int(seconds)}s ago"
    if seconds < 5400:
        return f"{seconds / 60:.0f} min ago"
    return f"{seconds / 3600:.1f}h ago"


def _describe(record: Any, now: float) -> str:
    """One line: what was attempted, with what tools, and how it came out."""
    intention = " ".join(str(getattr(record, "intention", "") or "").split())[:110]
    actions = list(getattr(record, "actions_taken", None) or [])
    tools = ", ".join(
        dict.fromkeys(str(getattr(action, "tool_name", "") or "") for action in actions if action)
    )[:80]
    succeeded = bool(actions) and all(bool(getattr(action, "success", False)) for action in actions)
    outcome = " ".join(
        str(
            getattr(record, "actual_outcome", None)
            or getattr(record, "observation", None)
            or ""
        ).split()
    )[:160]

    when = _ago(now - float(getattr(record, "completed_at", None) or now))
    verdict = "SUCCEEDED" if succeeded else "DID NOT SUCCEED"
    parts = [f"- {when}: {intention or 'an action'} — {verdict}"]
    if tools:
        parts.append(f"(tools: {tools})")
    if outcome:
        parts.append(f"— observed: {outcome}")
    return " ".join(parts)


def recent_actions_block(*, now: float | None = None) -> str:
    """The last few attempts and their real outcomes — or that there were none.

    Never returns an empty heading, and never returns nothing when it could say
    "nothing". Silence about a period is what gets filled in with something
    plausible; a stated absence is an answer she can give.
    """
    stamp = float(now if now is not None else time.time())
    try:
        from core.agency.intention_loop import get_intention_loop

        completed = list(getattr(get_intention_loop(), "_completed_intentions", None) or [])
    except _RECOVERABLE as exc:
        record_degradation(
            "recent_actions", exc, severity="info", action="omitted recent-actions grounding"
        )
        return ""
    lines: list[str] = []
    for record in reversed(completed):
        try:
            finished = float(getattr(record, "completed_at", None) or 0.0)
        except (TypeError, ValueError):
            continue
        if finished <= 0.0 or stamp - finished > _RECENCY_WINDOW_S:
            continue
        try:
            lines.append(_describe(record, stamp))
        except _RECOVERABLE:
            continue
        if len(lines) >= _MAX_ACTIONS:
            break

    if not lines:
        # A blank section is an invitation. Asked for "one concrete thing that
        # actually happened in your runtime in the last hour", with nothing in
        # front of her, she described processing a 45-page PDF on neuromorphic
        # computing, and on another run a user asking about caffeine chemistry.
        # Neither happened. She was not being dishonest — she was answering a
        # question about a period she had no record of, and the honest answer
        # was unavailable to her because nobody had written it down.
        #
        # "Nothing" is a fact. Stated, it is answerable; omitted, it is a gap
        # that gets filled with something plausible.
        return "\n".join(
            [
                RECENT_ACTIONS_HEADER,
                "You have taken no tool actions in the last "
                f"{int(_RECENCY_WINDOW_S // 60)} minutes. If you are asked what "
                "you have been doing or what has happened recently, that is the "
                "answer — say so plainly. Do not describe an action you did not "
                "take.",
            ]
        )
    return "\n".join(
        [
            RECENT_ACTIONS_HEADER,
            "Your real action receipts, newest first. If the user asks about "
            "something you did, an artifact you made, or whether it worked, "
            "answer from these. An attempt that DID NOT SUCCEED produced no "
            "artifact — do not describe how it behaves.",
            *lines,
        ]
    )


#: "prove you did something", "what have you actually done", "did you do
#: anything while I was away" — a question the receipts answer outright.
_ASKS_WHAT_SHE_DID = re.compile(
    r"\b(?:prove|show\s+me)\b[^.?!]{0,40}\byou\s+(?:did|ran|made|built|used)\b"
    r"|\bwhat\s+have\s+you\s+(?:actually\s+)?(?:done|been\s+doing|run|used)\b"
    r"|\bwhat\s+did\s+you\s+(?:actually\s+)?(?:do|run|use|make|build)\b"
    r"|\bdid\s+you\s+(?:actually\s+)?(?:do|run|use|make|build)\s+anything\b"
    r"|\banything\s+you(?:'ve| have)\s+done\b",
    re.IGNORECASE,
)


def asks_what_she_recently_did(user_message: Any) -> bool:
    """True when the receipts are the answer, not context for one."""
    return bool(_ASKS_WHAT_SHE_DID.search(str(user_message or "")))


def recent_actions_answer() -> str:
    """The receipts as an ANSWER, with nothing written for the model in it.

    The block above is scaffolding: it carries a heading and an instruction
    about how to use the receipts. Serving that verbatim would hand someone
    the prompt instead of the answer, which is the same failure as any other
    leaked internal text. This keeps the receipt lines and says what they are.
    """
    block = recent_actions_block()
    if not block:
        return ""
    lines = [
        line.strip()
        for line in block.splitlines()
        if line.strip().startswith("-")
    ]
    if not lines:
        return ""
    return "\n".join(
        [
            "Here is what actually ran, from my own action receipts:",
            *lines,
        ]
    )


__all__ = [
    "RECENT_ACTIONS_HEADER",
    "asks_what_she_recently_did",
    "recent_actions_answer",
    "recent_actions_block",
]
