"""Telling what the turn attached apart from what the person said.

A live turn assembles a prompt around the visible message: the desktop
full-mind contract directives, active grounding evidence, a screen reading,
excerpts of her own source. All of it is bracketed with a banner so it is
legible in a prompt — and all of it used to be recorded as ``role: user``,
because the augmented objective was what got appended to working memory.

Measured live 2026-08-04. Two turns about her source attached real excerpts
as evidence; the third turn asked "what's 17 times 4?" and came back with a
function from ``core/memory/associative_entity_memory.py``. The excerpts
were still in working memory, and text in working memory is material a model
continues — the same mechanism that made a screen capture come back as the
reply.

Recording the visible message stops NEW pollution. This module is the other
half: what is already in memory is scrubbed on the way past, so a
conversation that was contaminated before the fix heals instead of carrying
those blocks for the rest of its life.

Pure string handling, no imports from the runtime: this has to be safe to
call from memory paths that must not pull cognition in behind them.
"""
from __future__ import annotations

import re
import uuid
from typing import Any

__all__ = [
    "EVIDENCE_BANNERS",
    "GROUNDING_STAMP",
    "INJECTED_BANNERS",
    "carries_read_evidence",
    "contains_injected_block",
    "is_stamped_grounding",
    "is_stamped_runtime_payload",
    "stamp_grounding",
    "stamp_runtime_payload",
    "strip_injected_blocks",
]

#: Proof that the RUNTIME produced a grounding message, not a caller.
#:
#: Grounding used to be recognised by its text: a system message counted as
#: tool or skill evidence because it contained "[TOOL RESULT:" or carried a
#: metadata type string. Both are caller-controlled, so anything that could
#: put a system message into the payload could dress arbitrary text as
#: evidence and inherit the position and compaction protection real evidence
#: gets. A per-process nonce cannot be guessed by a caller and cannot leak
#: between runs, which is exactly the property the marker never had.
GROUNDING_STAMP = f"aura-grounding-{uuid.uuid4().hex}"

_STAMP_KEY = "aura_grounding_stamp"


def stamp_grounding(message: dict) -> dict:
    """Mark a message as runtime-produced grounding. Mutates and returns it."""
    if not isinstance(message, dict):
        return message
    metadata = message.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
        message["metadata"] = metadata
    metadata[_STAMP_KEY] = GROUNDING_STAMP
    return message


#: The same per-process proof, for payloads that are not messages. A
#: live-mind snapshot arrives inside the caller's context dict, and control
#: binding used to trust its own ``ready`` flag and its own
#: ``required_subsystems_ok`` boolean — a dictionary vouching for itself.
_PAYLOAD_STAMP_KEY = "aura_runtime_stamp"


def stamp_runtime_payload(payload: dict) -> dict:
    """Mark a dict as produced by THIS runtime. Mutates and returns it."""
    if isinstance(payload, dict):
        payload[_PAYLOAD_STAMP_KEY] = GROUNDING_STAMP
    return payload


def is_stamped_runtime_payload(payload: Any) -> bool:
    """Whether this process produced this payload."""
    return (
        isinstance(payload, dict)
        and payload.get(_PAYLOAD_STAMP_KEY) == GROUNDING_STAMP
    )


def is_stamped_grounding(message: Any) -> bool:
    """Whether THIS process produced this grounding message."""
    if not isinstance(message, dict):
        return False
    metadata = message.get("metadata")
    if not isinstance(metadata, dict):
        return False
    return metadata.get(_STAMP_KEY) == GROUNDING_STAMP

#: The banners a turn attaches. Each is either a fenced block with a matching
#: ``[END …]`` line, or a single labelled section that runs to the end.
INJECTED_BANNERS: tuple[str, ...] = (
    "YOUR OWN RECENT PERCEPTION",
    "YOUR OWN SOURCE",
    "LIVE DESKTOP FULL-MIND CONTRACT",
    "LIVE DESKTOP TURN EVIDENCE",
    "ACTIVE GROUNDING EVIDENCE",
    "OBSERVATION",
    "DIRECT RESULT",
    "SKILL OUTPUT",
    "SKILL RESULT",
    "INTERNAL MEMORY RECALL",
)

#: The banners that report something the turn READ, as opposed to the ones
#: that describe her — source, perception, the desktop contract. A tool loop
#: needs the first kind and must not be handed the second: the conversational
#: scaffold around a tool call produced an immediate end-of-turn.
EVIDENCE_BANNERS: tuple[str, ...] = (
    "ACTIVE GROUNDING EVIDENCE",
    "OBSERVATION",
    "DIRECT RESULT",
    "SKILL OUTPUT",
    "SKILL RESULT",
    "INTERNAL MEMORY RECALL",
)

_EVIDENCE_BANNER_RE = re.compile(
    r"\[\s*(?:" + "|".join(re.escape(name) for name in EVIDENCE_BANNERS) + r")\b",
    re.IGNORECASE,
)


def carries_read_evidence(message: Any) -> bool:
    """Whether this message reports something the turn read.

    The stamp is the precise signal and is preferred. A block that reached
    the payload through a producer that has not adopted the stamp yet is
    still recognised by its banner, which is the vocabulary this module
    already defines — the alternative was every caller inventing its own
    metadata key, which is how the grounding marker went wrong before.
    """
    if isinstance(message, dict) and is_stamped_grounding(message):
        return True
    if isinstance(message, dict):
        metadata = message.get("metadata")
        if isinstance(metadata, dict) and str(metadata.get("type") or "") == "skill_result":
            return True
        body = str(message.get("content") or "")
    else:
        body = str(message or "")
    return bool(_EVIDENCE_BANNER_RE.search(body))


_BANNER_ALTERNATION = "|".join(re.escape(name) for name in INJECTED_BANNERS)

#: A fenced block: ``[NAME …]`` through ``[END NAME…]``, inclusive.
_FENCED_RE = re.compile(
    rf"\[\s*(?:{_BANNER_ALTERNATION})\b[^\]]*\]" r".*?" rf"\[\s*END\b[^\]]*\]",
    re.DOTALL | re.IGNORECASE,
)

#: An unfenced banner — ``[DIRECT RESULT]: …`` — which runs to the end of the
#: text. Applied only after the fenced form has been removed, so a properly
#: closed block is never over-consumed.
_TRAILING_RE = re.compile(
    rf"\[\s*(?:{_BANNER_ALTERNATION})\b[^\]]*\].*\Z",
    re.DOTALL | re.IGNORECASE,
)


def strip_injected_blocks(text: Any) -> str:
    """The person's words, with everything the turn attached removed.

    Returns the input unchanged when it carries no banner, and never returns
    an empty string for non-empty input that was ENTIRELY injected — the
    caller needs to be able to tell "nothing was said" from "all of it was
    machinery", and an empty user message is its own defect downstream.
    """
    body = str(text or "")
    if not body.strip():
        return body
    cleaned = _FENCED_RE.sub("", body)
    cleaned = _TRAILING_RE.sub("", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned or body.strip()


def contains_injected_block(text: Any) -> bool:
    """Whether this text carries machinery that was never spoken."""
    body = str(text or "")
    if not body.strip():
        return False
    return bool(_FENCED_RE.search(body) or _TRAILING_RE.search(body))
