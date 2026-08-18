"""The observables themselves. One entry each; the registry does the rest.

Kept apart from observable_grounding so the mechanism has no opinion about
which things exist, and adding a reading never means editing the machinery.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any

from core.brain.observable_grounding import Observable, register_observable

# ── clipboard ────────────────────────────────────────────────────────────────

def _matches_clipboard(prompt: str) -> bool:
    from core.brain.clipboard_grounding import asks_about_clipboard

    return asks_about_clipboard(prompt)


async def _read_clipboard(prompt: str) -> str:
    from core.brain.clipboard_grounding import CLIPBOARD_HEADER, clipboard_block

    block = await clipboard_block(prompt)
    return block.replace(f"{CLIPBOARD_HEADER}\n", "", 1) if block else ""


# ── a named file ─────────────────────────────────────────────────────────────

def _matches_file(prompt: str) -> bool:
    from core.conversation.filesystem_check import requested_file_read

    return requested_file_read(prompt) is not None


async def _read_file(prompt: str) -> str:
    from core.conversation.filesystem_check import requested_file_read

    read = await asyncio.to_thread(requested_file_read, prompt)
    if read is None:
        return ""
    if not read.exists:
        return f"No file exists at {read.path}."
    if not read.text.strip():
        return f"{read.path} is empty."
    suffix = " [truncated]" if read.truncated else ""
    coverage = ""
    if read.barely_covers_topic:
        coverage = (
            f"\nCOVERAGE: this file uses the word '{read.topic}' "
            f"{read.topic_mentions} time(s) in total. It does not discuss the topic."
        )
    return f"{read.path}{suffix}{coverage}\n{read.text}"


# ── a directory count ────────────────────────────────────────────────────────

def _matches_count(prompt: str) -> bool:
    from core.conversation.filesystem_check import requested_filesystem_count

    return requested_filesystem_count(prompt) is not None


async def _read_count(prompt: str) -> str:
    from core.conversation.filesystem_check import requested_filesystem_count

    counted = await asyncio.to_thread(requested_filesystem_count, prompt)
    if counted is None:
        return ""
    if not counted.exists:
        return f"There is no directory at {counted.path}."
    kind = f"{counted.suffix} " if counted.suffix else ""
    listed = ", ".join(counted.names[:40]) or "nothing"
    return f"{counted.path} contains {counted.count} {kind}file(s): {listed}"


# ── the local reference corpus ───────────────────────────────────────────────

def _matches_corpus(prompt: str) -> bool:
    from core.knowledge.corpus_grounding import is_corpus_groundable

    return is_corpus_groundable(prompt)


async def _read_corpus(prompt: str) -> str:
    from core.knowledge.corpus_grounding import corpus_grounding_for

    grounding = await asyncio.to_thread(corpus_grounding_for, prompt)
    if not grounding.grounded:
        return ""
    return "\n".join(grounding.render())


# ── the wall clock ───────────────────────────────────────────────────────────
#
# "what time is it" was answered "my clock says 06:15 and the ambient light
# sensors report low illumination" at 01:40, from a runtime with no light
# sensor. The time is the most trivially observable thing on the machine.

_ASKS_TIME = re.compile(
    r"\bwhat\s+(?:time|day|date)\b|\bwhat'?s\s+the\s+(?:time|date)\b"
    r"|\btoday'?s\s+date\b|\bwhat\s+day\s+is\s+it\b",
    re.IGNORECASE,
)


def _matches_clock(prompt: str) -> bool:
    return bool(_ASKS_TIME.search(prompt))


async def _read_clock(_prompt: str) -> str:
    from datetime import datetime

    now = datetime.now().astimezone()
    return now.strftime("%A %d %B %Y, %H:%M:%S %Z")


# ── the screen ───────────────────────────────────────────────────────────────
#
# "what's on my screen right now?" was answered "I couldn't get to an answer
# I'd stand behind on that one." Screen capture was permitted and working; it
# simply was not taken. A reading that comes back thin is still a reading, and
# "the frontmost window is X and no text is readable from it" is an answer.

_ASKS_SCREEN = re.compile(
    r"\b(?:my|the)\s+screen\b|\bon\s+screen\b|\bwhat\s+(?:am\s+i|are\s+you)\s+looking\s+at\b"
    r"|\bwhat\s+do\s+you\s+see\b|\bwhat'?s\s+(?:up\s+)?on\s+(?:my|the)\s+display\b"
    r"|\bwhat\s+window\b|\bwhich\s+app\b",
    re.IGNORECASE,
)


def _matches_screen(prompt: str) -> bool:
    return bool(_ASKS_SCREEN.search(prompt))


async def _read_screen(_prompt: str) -> str:
    from core.perception.screen_perception import get_screen_perception

    snapshot = await get_screen_perception().capture(save_screenshot=False)
    if getattr(snapshot, "capture_denied", False):
        return "Screen capture was refused for this turn."
    app = str(getattr(snapshot, "active_app", "") or "").strip()
    text = str(getattr(snapshot, "text", "") or "").strip()
    if not text:
        text = str(getattr(snapshot, "accessibility_text", "") or "").strip()
    focused = " / ".join(
        part
        for part in (
            str(getattr(snapshot, "focused_role", "") or "").strip(),
            str(getattr(snapshot, "focused_name", "") or "").strip(),
        )
        if part
    )
    lines = [f"Frontmost application: {app or 'unknown'}"]
    if focused:
        lines.append(f"Focused element: {focused}")
    if text:
        lines.append(text[:2000])
    else:
        # An absent reading, named. This is what stops "the room is silent and
        # the light is unchanged" from being invented to fill the gap.
        lines.append(
            "No readable text was available from this window "
            "(the accessibility layer returned nothing)."
        )
    return "\n".join(lines)


# ── what she actually believes ───────────────────────────────────────────────
#
# "what do you currently believe about me?" was answered from the model. She
# has a belief graph; the beliefs in it are the answer to that question.

_ASKS_BELIEFS = re.compile(
    r"\bwhat\s+do\s+you\s+(?:currently\s+)?(?:believe|think)\s+about\b"
    r"|\byour\s+beliefs?\b|\bwhat\s+have\s+you\s+concluded\b",
    re.IGNORECASE,
)


def _matches_beliefs(prompt: str) -> bool:
    return bool(_ASKS_BELIEFS.search(prompt))


async def _read_beliefs(_prompt: str) -> str:
    from core.container import ServiceContainer

    graph = ServiceContainer.get("belief_graph", default=None) or ServiceContainer.get(
        "world_model", default=None
    )
    if graph is None or not hasattr(graph, "get_beliefs"):
        return ""
    beliefs = await asyncio.to_thread(graph.get_beliefs)
    if not beliefs:
        return "The belief store holds no entries."
    lines = []
    for key, value in list(dict(beliefs).items())[:20]:
        lines.append(f"- {key}: {str(value)[:160]}")
    return "\n".join(lines)


# ── work she has queued ──────────────────────────────────────────────────────
#
# "do you have any scheduled or background work queued right now?" was answered
# "No, my foreground queue is empty. I'm not tracking any background
# maintenance tasks at the moment either." She had biological_sleep and
# dlq_recovery deferred in the dream coordinator at that moment, nine queue
# events in that boot alone.
#
# This is awareness of her own non-immediate actions, and she has a status()
# that answers it exactly.

_ASKS_QUEUED_WORK = re.compile(
    r"\b(?:queued|scheduled|pending|background)\s+(?:work|task|tasks|job|jobs|maintenance)\b"
    r"|\bwhat(?:'s| is)\s+(?:in\s+)?your\s+queue\b"
    r"|\banything\s+(?:queued|scheduled|pending|planned)\b"
    # "are you planning to do anything later?" is the same question in the
    # phrasing a person actually uses, and the first pattern missed it.
    r"|\bplan(?:ning|s)?\s+to\s+do\s+(?:anything|something|any\s+work)?\s*later\b"
    r"|\bwhat\s+(?:are\s+you|will\s+you)\s+(?:be\s+)?do(?:ing)?\s+(?:later|next|after)\b"
    r"|\bwaiting\s+to\s+run\b",
    re.IGNORECASE,
)


def _matches_queued_work(prompt: str) -> bool:
    return bool(_ASKS_QUEUED_WORK.search(prompt))


async def _read_queued_work(_prompt: str) -> str:
    from core.maintenance.dream_coordinator import get_dream_coordinator

    status = await asyncio.to_thread(get_dream_coordinator().status)
    pending = dict(status.get("pending") or {})
    if not pending:
        return "Nothing is deferred in the maintenance coordinator."
    lines = []
    for name, detail in list(pending.items())[:12]:
        reason = str(dict(detail or {}).get("reason") or "").strip()
        lines.append(f"- {name}" + (f" (waiting on: {reason})" if reason else ""))
    return "Deferred maintenance work:\n" + "\n".join(lines)


# ── what was actually said in this conversation ──────────────────────────────
#
# "what did I ask you two messages ago?" was answered "You asked, 'What's the
# weather like? I can't seem to find my umbrella.' Then you asked me what I
# thought about that. I said it was fine." None of that was ever said. She
# invented an exchange, in detail, with dialogue.
#
# The transcript is on disk. Recall about this conversation is a reading, not a
# recollection, and inventing it is the worst failure in the set — it is
# indistinguishable from remembering, and it rewrites what the person said.

_ASKS_TRANSCRIPT_RECALL = re.compile(
    r"\bwhat\s+did\s+(?:i|you|we)\s+(?:just\s+)?(?:ask|say|tell|mention)\b"
    r"|\b(?:messages?|turns?)\s+ago\b"
    r"|\bearlier\s+(?:i|you|we)\s+(?:asked|said|mentioned)\b"
    r"|\bmy\s+(?:first|last|previous)\s+(?:question|message)\b"
    r"|\bwhat\s+was\s+my\s+(?:first|last|previous)\b"
    # "what was the first THING I said to you" — the phrasing a person
    # actually uses, and the one the first pattern missed.
    r"|\b(?:first|last|earliest|previous)\s+thing\s+(?:i|you|we)\s+(?:said|asked|told|mentioned)\b"
    r"|\bwhat\s+did\s+(?:i|you|we)\s+(?:say|ask)\s+(?:first|last)\b"
    r"|\brepeat\s+(?:back\s+)?what\s+i\s+said\b",
    re.IGNORECASE,
)


def _matches_transcript(prompt: str) -> bool:
    return bool(_ASKS_TRANSCRIPT_RECALL.search(prompt))


async def _read_transcript(prompt: str) -> str:
    # _user_turns cascades live working memory -> transcript; the transcript
    # singleton alone came back empty in the live runtime while the
    # conversation was plainly happening, so reading only the last resort
    # produced "No transcript is available" mid-conversation.
    from core.conversation.grounded_recall import _user_turns

    turns = await asyncio.to_thread(_user_turns, "")
    turns = [str(t).strip() for t in (turns or []) if str(t or "").strip()]
    if not turns:
        # A named absence. "I have no transcript for this session" is a true
        # answer; an invented exchange is not.
        return "No transcript is available for this conversation yet."
    # "What was the FIRST thing I told you?" is not answerable from a window of
    # the most recent turns, and answering it from that window produces a
    # confident wrong answer rather than a miss. LIVE 2026-08-17: the first
    # turn was "ok" and she reported "You asked if I was still here."
    lines: list[str] = []
    if len(turns) > 8:
        for position, turn in enumerate(turns[:3], start=1):
            lines.append(f"turn {position} of this conversation, they said: {turn[:300]}")
        lines.append(f"... {len(turns) - 11} turn(s) not shown ...")
    recent = turns[-8:]
    offset = len(turns) - len(recent)
    for index, turn in enumerate(recent):
        ago = len(recent) - index
        lines.append(
            f"turn {offset + index + 1} ({ago} turn(s) ago), they said: {turn[:300]}"
        )
    lines.append(f"({len(turns)} user turn(s) in this conversation.)")
    return "\n".join(lines)


def install_default_observables() -> None:
    """Register the readings this runtime can take."""

    # Every example below that reads like an odd phrasing IS one: each was a
    # real question that the matcher beside it did not recognise, or wrongly
    # claimed, in live use. They are counter-examples for each other as much as
    # for themselves — the screen matcher wrongly took a clipboard WRITE, and
    # the recall matcher would happily swallow "the first rule in
    # CONTRIBUTING.md" if nobody said otherwise.
    for observable in (
        Observable(
            "clipboard", "## WHAT IS ON THE CLIPBOARD", _matches_clipboard, _read_clipboard,
            examples=(
                "what's on my clipboard right now?",
                "read my clipboard",
                "what did I just copy?",
                "check the pasteboard",
            ),
            counter_examples=(
                "put BUILD-42 on my clipboard",
                "how are you doing",
                "what is 2 + 2",
            ),
        ),
        Observable(
            "file", "## FILE YOU WERE ASKED ABOUT", _matches_file, _read_file,
            examples=(
                "read the file CONTRIBUTING.md and tell me the first rule",
                "what does CONTRIBUTING.md say about tests?",
                "open core/config.py",
                "tell me about ARCHITECTURE.md",
            ),
            counter_examples=(
                "how are you doing",
                "read my clipboard",
                "what did I say first?",
            ),
        ),
        Observable(
            "file_count", "## DIRECTORY LISTING YOU WERE ASKED ABOUT", _matches_count, _read_count,
            examples=(
                "count the .py files in core/introspection and tell me the number",
                "how many python files live in core/introspection?",
                "how many files do we have in core/introspection",
            ),
            counter_examples=(
                "how many files are in /etc",
                "how are you doing",
                "read CONTRIBUTING.md",
            ),
        ),
        Observable(
            "corpus", "## REFERENCE PASSAGES FROM THE LOCAL CORPUS", _matches_corpus, _read_corpus,
            examples=(
                "explain the difference between correlation and causation",
                "what is a confounding variable",
                "who was Ada Lovelace?",
            ),
            counter_examples=(
                "how are you doing right now?",
                "what did I ask you first today?",
                "open my notes folder",
            ),
        ),
        Observable(
            "clock", "## THE CURRENT LOCAL TIME", _matches_clock, _read_clock,
            examples=("what time is it?", "what's today's date?", "what day is it?"),
            counter_examples=(
                "how long have you been running",
                "how are you doing",
                "what is 2 + 2",
            ),
        ),
        # A screen capture is a real device read and the FIRST one in a process
        # pays initialisation: measured 0.81s warm, past the 2.5s default cold,
        # which is why the first screen question of a session silently returned
        # no block at all.
        Observable(
            "screen", "## WHAT IS ON THE SCREEN", _matches_screen, _read_screen,
            timeout_s=8.0,
            examples=(
                "what's on my screen right now?",
                "what do you see?",
                "which app is in front?",
                "what window am I looking at",
            ),
            counter_examples=(
                # A clipboard WRITE, which this matcher once claimed and used
                # to pull a real desktop action off the executor path.
                "put BUILD-42 on my clipboard",
                "create a file called notes.txt on my desktop",
                "how are you doing",
            ),
        ),
        Observable(
            "beliefs", "## WHAT YOU ACTUALLY BELIEVE", _matches_beliefs, _read_beliefs,
            examples=(
                "what do you currently believe about me?",
                "what do you think about me?",
                "tell me your beliefs",
            ),
            counter_examples=(
                "what do you think of that film?",
                "how are you doing",
            ),
        ),
        Observable(
            "queued_work", "## WORK YOU HAVE QUEUED", _matches_queued_work, _read_queued_work,
            examples=(
                "do you have any scheduled or background work queued right now?",
                "are you planning to do anything later?",
                "anything planned?",
                "what will you be doing next?",
            ),
            counter_examples=(
                "plan a trip to Rome",
                "how are you doing",
                "what is 2 + 2",
            ),
        ),
        Observable(
            "transcript",
            "## WHAT WAS ACTUALLY SAID IN THIS CONVERSATION",
            _matches_transcript,
            _read_transcript,
            examples=(
                "what did I ask you two messages ago?",
                "what was my first question?",
                "what was the first thing I said to you in this conversation?",
                "what was the last thing I told you?",
                "repeat back what I said",
            ),
            counter_examples=(
                # Contains "first" and asks about a file: a recall matcher that
                # swallowed this would break file reading to fix recall.
                "what is the first rule in CONTRIBUTING.md",
                "how are you doing",
                "what did you read?",
            ),
        ),
    ):
        register_observable(observable)


install_default_observables()


def observable_names() -> list[str]:
    from core.brain.observable_grounding import OBSERVABLES

    return [observable.name for observable in OBSERVABLES]


def _unused(value: Any) -> Any:  # pragma: no cover - keeps Any import honest
    return value
