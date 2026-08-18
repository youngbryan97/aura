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


def install_default_observables() -> None:
    """Register the readings this runtime can take."""

    for observable in (
        Observable("clipboard", "## WHAT IS ON THE CLIPBOARD", _matches_clipboard, _read_clipboard),
        Observable("file", "## FILE YOU WERE ASKED ABOUT", _matches_file, _read_file),
        Observable("file_count", "## DIRECTORY LISTING YOU WERE ASKED ABOUT", _matches_count, _read_count),
        Observable("corpus", "## REFERENCE PASSAGES FROM THE LOCAL CORPUS", _matches_corpus, _read_corpus),
        Observable("clock", "## THE CURRENT LOCAL TIME", _matches_clock, _read_clock),
    ):
        register_observable(observable)


install_default_observables()


def observable_names() -> list[str]:
    from core.brain.observable_grounding import OBSERVABLES

    return [observable.name for observable in OBSERVABLES]


def _unused(value: Any) -> Any:  # pragma: no cover - keeps Any import honest
    return value
