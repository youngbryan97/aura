"""Every desktop turn shares one authority head, byte for byte.

A KV cache can only reuse a prefix. The desktop lane built its system prompt
from five hand-written variants selected by contract flag, all opening with the
same sentence and differing after it, so no two turns of a conversation shared
a prefix and every turn paid a full prefill.

Measured live 2026-09-07 on the resident 27B: two consecutive turns produced
authority heads of 587 and 459 characters with different digests, the prompt
cache reported `matched 0 (0.0%)` of 1,844 tokens, and prefill was 17.7s of a
22s turn.

What differs per turn is a directive about that turn. It travels with the turn.
"""

from __future__ import annotations

from pathlib import Path

from core.brain.cognitive_engine import _DESKTOP_AUTHORITY_HEAD
from tests.source_contract import family_text_at

_SOURCE = family_text_at(Path('core/brain/cognitive_engine.py'))


def test_the_head_is_a_constant_not_a_branch() -> None:
    assert _DESKTOP_AUTHORITY_HEAD == (
        "You are Aura speaking through the live desktop CognitiveEngine."
    )


def test_no_branch_rebuilds_the_head() -> None:
    """The defect was five copies of the head. One constant is the fix.

    Other lanes — the solver recovery path, the CognitiveEngine recovery path —
    build their own prompts and are out of scope here; what must not come back
    is a second copy of the DESKTOP head that a branch can vary.
    """
    literal_heads = _SOURCE.count(
        '"You are Aura speaking through the live desktop CognitiveEngine.'
    )
    assert literal_heads == 1, (
        f"{literal_heads} literal desktop authority heads; it must be one constant"
    )


def test_no_per_turn_directive_is_left() -> None:
    """The head moved its directives next to the turn; on 17 October they were removed.

    They told her model how to answer and none of them was evidence. What reaches
    her on each kind of turn is checked by tests/test_no_instruction_reaches_her.py.
    """
    for directive in (
        "CPU, RAM, host load",
        "Use canonical memory/state evidence",
        "current runtime-path question",
        "browser/web research",
        "Answer the user's current message directly and naturally",
    ):
        assert directive not in _SOURCE, f"a directive is back: {directive!r}"


def test_the_head_region_appends_no_contract() -> None:
    region_start = _SOURCE.index("system_prompt = _DESKTOP_AUTHORITY_HEAD")
    region_end = _SOURCE.index("_record_the_capability_inventory_miss", region_start)
    region = _SOURCE[region_start:region_end]
    assert "turn_dynamic_contracts.append(" not in region
    assert "system_prompt = (" not in region
