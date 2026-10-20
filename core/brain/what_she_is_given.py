"""What her model is given on a turn: what was said, and what her instruments and memory returned.

Bryan, 13 October 2026: "i dont want the internal system prompts at all. aura
should not be clever prompting. we need to remove the prompt, and replace it
with actual systems engineering."

Until then four layers of text told her who to be before she spoke. A long
prompt of 16,478 characters, 96% of it persona; a compact one in the response
phase with eleven hard rules; a compact one in this gate that said "You have
strong opinions and you share them freely"; and an identity rule the chat
formatter added by itself whenever nothing else had. Her mood, her body and
her goals were written out beside them for the model to read.

None of that is dispatched now. Who she is is in the weights of the model she
runs on, which is a fine-tune of her own persona. What state she is in reaches
generation through her residual stream, where sixteen hooks add it on every
token (core/consciousness/affective_steering.py), and through the sampler. The
shape of an answer is held by the decoder
(core/brain/llm/a_shape_the_decoder_enforces.py and its neighbours). Whether a
claim is grounded is checked after it is made.

What is left is the turn itself and what she knows. The blocks her instruments
write (the present moment, her own instruments, what she just did, a registered
observable such as a passage from her corpus) are readings, and they travel as
readings: one message in the runtime-evidence role, which the template renders
as a tool's result and never as authority (core/utils/injected_blocks.py).

A turn that carries a contract its lane still words as instructions (a proof
or benchmark answer, an execution plan, a shape or a choice asked of the
decoder) is left as it was until that lane's instruction is a mechanism too.
Those are named in `STILL_WORDED`, and the list only gets shorter.
"""
from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Mapping
from typing import Any

from core.runtime.flags import FlagKind, declare
from core.utils.injected_blocks import (
    RUNTIME_EVIDENCE_ROLE,
    carries_read_evidence,
    injected_blocks_in,
    stamp_grounding,
)

_FLAG_SYSTEM_TEXT = declare("AURA_SYSTEM_TEXT", kind=FlagKind.STRING, default='',
                            description="keep gives her model the system text again", owner="prompt")

__all__ = ["STILL_WORDED", "for_her_model", "her_turn_is_still_worded", "readings_in", "what_she_is_given"]

#: Flags on a turn's context whose lanes still put their contract in words.
STILL_WORDED: tuple[str, ...] = (
    "proof_or_benchmark",
    "proof_run",
    "benchmark_run",
    "proof_answer_run",
    "desktop_execution_contract",
    "output_shape",
    "choose_from",
    "operator_evidence_contract",
    "strict_answer_contract",
)

_HEADING = re.compile(r"(?m)^(?=## )")
_HEADED = re.compile(r"(?m)^## ")
#: The snapshot of her live mind, which the desktop lane fences in a system text.
_LIVE_MIND = re.compile(r"\[LIVE MIND CONTEXT\].*?\[END LIVE MIND CONTEXT\]", re.DOTALL)
#: A fenced block that is a contract with the lane and not something she read.
_A_CONTRACT = "[LIVE DESKTOP FULL-MIND CONTRACT"

#: What her instruments read of herself: her mood, her body, her goals, how
#: unified she is. These shaped every turn as text. They are readings, and
#: they are what a turn is about only when the turn turns to her, so that is
#: when they travel, as evidence for what she says of herself.
_OF_HERSELF = (
    "## LIVE TONE",
    "## FUNCTIONAL STATE SIGNALS",
    "## SOMATIC STATE",
    "## UNITY",
    "## GOALS",
    "## GOAL EXECUTION STATE",
    "## STATE",
    "## CURRENT STATE",
    "## COGNITIVE TELEMETRY",
    "## AURA NOW",
    "## STATE-GROUNDED INTROSPECTION",
    "## CAUSAL VALENCED WORKSPACE",
    "## HOMEOSTASIS",
    "## LIVE PHYSIOLOGY",
    "## HEDONIC STATE",
    "## CIRCADIAN STATE",
    "## AROUSAL STATE",
    "## UNIFIED FELT STATE",
    "## WELFARE STATE",
    "## RESOURCE STATE",
    "## DERIVED RUNTIME SIGNALS",
)

logger = logging.getLogger("Aura.WhatSheIsGiven")

#: The blocks her instruments write that are not in the observable registry.
_READINGS = (
    "## PRESENT MOMENT",
    "## YOUR OWN INSTRUMENTS",
    "## WHAT YOU ACTUALLY JUST DID",
    # How the social phase read what was said: joking, sarcastic, at odds
    # with something she holds (core/phases/social_context_phase.py).
    "## HOW WHAT WAS SAID WAS MEANT",
)

#: What her memory and her skills return. These are readings of her own
#: stores and of what she just did, and they are not in the observable
#: registry yet (core/brain/observable_grounding.py), which is where each
#: belongs. Until one is registered it is named here, so that removing the
#: runtime's words does not take what she remembers with them.
_REMEMBERED = (
    "## PRIORITY MEMORY EVIDENCE",
    "## PRIORITY BETWEEN-MOMENTS TRACE",
    "## PRIORITY RECENT SPECIFICITY",
    "## RECALLED CONTEXT",
    "## CONSENTED RELATIONAL MEMORY",
    "## CONSENTED SHARED GROUND",
    "## CONSENTED USER PROFILE DATA",
    "## CONSENTED RELATIONSHIP TOPOLOGY",
    "## IDENTITY CHRONICLE",
    "## CONTINUITY SUMMARY",
    "## HELD POSITION",
    "## WHAT YOU HAVE COME TO PREFER",
    "## WHAT YOU LEARNED BUILDING SIMILAR THINGS",
    "## PRIOR PLAN OUTCOMES",
    "## ACTIVE COMMITMENTS",
    "## TEMPORAL OBLIGATIONS",
    "## SKILL EXECUTION",
    "## TOOLS AVAILABLE",
    "## LIVE TOOL OPTIONS",
    "## LIVE TOOL AFFORDANCES",
    "## HUMOR OUTCOME EVIDENCE",
    "## CANONICAL SELF-CONDITION EVIDENCE",
    "## CODING WORKING SET",
    "## GROUNDED SELF-REPORT",
)


def her_turn_is_still_worded(context: Mapping[str, Any] | None) -> bool:
    """Whether this turn's lane still depends on worded instructions."""
    if _FLAG_SYSTEM_TEXT.value().strip().lower() == "keep":
        return True
    held = context or {}
    return any(held.get(flag) for flag in STILL_WORDED)


def readings_in(text: Any, markers: Iterable[str]) -> list[str]:
    """The blocks of a system text that are readings.

    A registered block, from its heading to the next heading, and a fenced
    payload the runtime attached (`[LIVE MIND CONTEXT] ... [END ...]`), which
    is a measurement wherever in the text it stands.
    """
    heads = tuple(marker for marker in markers if marker)
    found = [
        block.strip()
        for block in _HEADING.split(str(text or ""))
        if heads and block.startswith(heads) and block.strip()
    ]
    inside = "\n".join(found)
    found.extend(
        block
        for block in injected_blocks_in(text)
        if block not in inside and not block.upper().startswith(_A_CONTRACT)
    )
    if "## LIVE TONE" in heads:
        found.extend(match.group(0) for match in _LIVE_MIND.finditer(str(text or "")) if match.group(0) not in inside)
    return found


#: Readings taken of the turn itself by whatever takes them, each given what
#: was said and returning a headed block or nothing. They travel as readings
#: with the rest. See core/together/taking_it_up.py for the first.
OF_THIS_TURN: list[Any] = []


def _of_this_turn(said: Any) -> list[str]:
    found: list[str] = []
    for read in list(OF_THIS_TURN):
        try:
            block = str(read(str(said or "")) or "").strip()
        # A reading that fails is one reading fewer, and her turn goes on.
        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError) as exc:
            logger.debug("a reading of this turn was not taken: %s", exc)
            continue
        if block:
            found.append(block)
    return found


def what_she_is_given(
    messages: Any,
    system_prompt: Any,
    *,
    markers: Iterable[str],
    of_this_turn: Iterable[str] = (),
) -> tuple[Any, str]:
    """The turn without the runtime's own words, and with its readings kept as readings.

    Returns the messages and the separately passed system prompt, which is
    empty afterwards. A system message made of headed blocks gives up the
    blocks that are readings and nothing else. One with no headings that
    carries something she read, a tool's result, is a reading whole.
    """
    if not isinstance(messages, list):
        return messages, str(system_prompt or "")
    heads = tuple(markers)
    readings: list[str] = [*readings_in(system_prompt, heads), *of_this_turn]
    kept: list[Any] = []
    for message in messages:
        if not isinstance(message, dict) or str(message.get("role") or "").strip().lower() != "system":
            kept.append(message)
            continue
        content = str(message.get("content") or "").strip()
        found = readings_in(content, heads)
        if found:
            readings.extend(found)
        elif content and not _HEADED.search(content) and carries_read_evidence(message):
            # A tool's result with no blocks in it is a reading whole. A
            # message of headed blocks is not, whatever stamp it carries: the
            # runtime stamps a whole message as grounding when one block in
            # it is, and that does not make the blocks beside it readings.
            readings.append(content)
    if readings:
        seen: set[str] = set()
        distinct = [block for block in readings if not (block in seen or seen.add(block))]
        evidence = stamp_grounding({"role": RUNTIME_EVIDENCE_ROLE, "content": "\n\n".join(distinct)})
        last_said = max(
            (index for index, message in enumerate(kept)
             if isinstance(message, dict) and str(message.get("role") or "").strip().lower() == "user"),
            default=len(kept),
        )
        kept.insert(last_said, evidence)
    return kept, ""


def _turns_to_her(said: Any) -> bool:
    """Whether what was said turns the turn to her, so that her own state is what it is about."""
    if not str(said or "").strip():
        return False
    try:
        from core.brain.llm.context_assembler import ContextAssembler

        return bool(ContextAssembler._attention_turns_inward(str(said)))
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        logger.debug("whether the turn turns to her could not be read: %s", exc)
        return False


#: Since the runtime began, by origin: dispatches to her model, and characters in them that tell it who to be or what
#: to do (system text, and sentences the runtime wrote that address it or order it; the person's own words and hers are
#: not counted). F06 on the master todo closes when every origin reads zero (tools/what_instructs_her.py).
_INSTRUCTED: dict[str, list[int]] = {}


def what_instructs_her() -> dict[str, dict[str, int]]:
    """The census so far, by origin."""
    return {origin: {"dispatches": n, "instructed_characters": chars} for origin, (n, chars) in sorted(_INSTRUCTED.items())}


def _census(given: Any, system_prompt: Any, origin: str, persons_words: str) -> None:
    from core.brain.what_instructs import instructing_sentences

    told = [str(system_prompt or "")]
    for message in given if isinstance(given, list) else []:
        if not isinstance(message, dict):
            continue
        role, content = str(message.get("role") or "").lower(), str(message.get("content") or "")
        if role == "system":
            told.append(content)
        elif role != "assistant" and not (persons_words and content.strip() == persons_words.strip()):
            told.extend(instructing_sentences(content))
    told = [text for text in told if text.strip()]
    counted = _INSTRUCTED.setdefault(origin, [0, 0])
    counted[0] += 1
    counted[1] += sum(len(text) for text in told)
    if told:
        logger.info("🧾 [INSTRUCTED] %s: %d characters tell her model who to be or what to do; the first: %r",
                    origin, sum(len(text) for text in told), told[0][:160])


def for_her_model(
    messages: Any,
    system_prompt: Any,
    context: Mapping[str, Any] | None,
    user_facing: bool,
    observables: Iterable[tuple[str, str]] = (),
    said: Any = "",
) -> tuple[Any, Any]:
    """The gate's one call: a turn of hers goes to her model without system text.

    A turn nobody sent, or one whose lane still words its contract, comes back
    as it went in. `said` is the message of the turn; when it turns to her,
    the readings her instruments took of her go with the rest.
    """
    origin = str((context or {}).get("origin") or "unknown")
    if not user_facing or her_turn_is_still_worded(context):
        _census(messages, system_prompt, origin, "")
        return messages, system_prompt
    of_herself = _OF_HERSELF if _turns_to_her(said) else ()
    told = len(str(system_prompt or "")) + sum(
        len(str(message.get("content") or ""))
        for message in (messages if isinstance(messages, list) else [])
        if isinstance(message, dict) and str(message.get("role") or "").lower() == "system"
    )
    given, none = what_she_is_given(
        messages,
        system_prompt,
        markers=(*_READINGS, *_REMEMBERED, *of_herself, *(marker for _name, marker in observables)),
        of_this_turn=_of_this_turn(said),
    )
    logger.info(
        "🧵 [NO SYSTEM TEXT] %d characters the runtime wrote were not dispatched; "
        "%d messages go to her model; readings: %s.",
        told,
        len(given) if isinstance(given, list) else 0,
        ", ".join(f"{name} {size}" for name, size in readings_by_size(given)) or "none",
    )
    _census(given, none, origin, str(said or ""))
    return given, none


def readings_by_size(messages: Any) -> list[tuple[str, int]]:
    """Each reading in the messages by its heading, with its size in characters.

    A reading runs from its heading to the next heading. This is what the
    stream shows of a turn, so what her model was handed besides the talk can
    be read off one line.
    """
    found: list[tuple[str, int]] = []
    for message in messages if isinstance(messages, list) else []:
        if not isinstance(message, dict) or str(message.get("role") or "").lower() != RUNTIME_EVIDENCE_ROLE:
            continue
        name, size = "", 0
        for line in str(message.get("content") or "").splitlines():
            heading = line.strip()
            if heading.startswith(("## ", "[")) and len(heading) < 120:
                if name or size:
                    found.append((name or "unheaded", size))
                name, size = heading.lstrip("#[ ").rstrip("] :"), 0
            size += len(line) + 1
        if name or size:
            found.append((name or "unheaded", size))
    return found
