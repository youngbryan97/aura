"""What the turn assembled is not what the person said.

Live 2026-08-04. Two turns about her source attached real excerpts to the
objective as evidence. The third turn asked "what's 17 times 4?" and came
back with a function from core/memory/associative_entity_memory.py — the
excerpts were still in working memory, and text in working memory is
material a model continues. The same mechanism that made a screen capture
come back as the reply.

The augmented objective carries the live-desktop contract directives,
grounding evidence, screen readings and source excerpts. Recording it as
``role: user`` files all of that as things the person said, and it persists
for the rest of the conversation.
"""
from __future__ import annotations

import re
from pathlib import Path

SOURCE = (
    Path(__file__).resolve().parents[1] / "core" / "brain" / "cognitive_engine.py"
).read_text("utf-8")


def _working_memory_append_block() -> str:
    """The user-message append, located structurally.

    This used to slice between two literal lines. The second one moved during
    an unrelated refactor and the test died with "substring not found" — a
    real invariant reporting a broken marker rather than a broken guarantee.
    The branch is found by what it TESTS instead, so only a change to the
    branch itself can break it.
    """
    import ast

    tree = ast.parse(SOURCE)
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        condition = ast.get_source_segment(SOURCE, node.test) or ""
        if "_is_user_facing_origin" in condition and "append_user_message" in condition:
            return "\n".join(
                ast.get_source_segment(SOURCE, statement) or ""
                for statement in node.body
            )
    raise AssertionError(
        "no branch appends the user message to working memory; the guarantee "
        "this file protects has no code left to protect"
    )


def test_the_visible_message_is_what_gets_remembered():
    block = _working_memory_append_block()
    assert '"content": remembered' in block, (
        "working memory records the augmented objective again; injected "
        "evidence will be filed as something the person said"
    )
    assert "visible_user_message" in block


def test_the_raw_objective_is_not_appended_as_user_speech():
    block = _working_memory_append_block()
    assert '"content": objective' not in block


def test_the_module_still_parses():
    import ast

    ast.parse(SOURCE)


def test_evidence_blocks_are_distinguishable_from_a_persons_words():
    """Every injected block is fenced with a bracketed banner.

    If one is ever recorded, this is what makes it findable rather than
    indistinguishable from speech.
    """
    from core.perception.observation_evidence import Observation, ObservationKind, ObservationMemory
    from core.self.source_excerpt import source_evidence_brief

    memory = ObservationMemory()
    memory.record(
        Observation(
            kind=ObservationKind.SCREEN_TEXT,
            capture="Some Window Title\nAnother real line of text\n",
            source="Safari",
        )
    )
    perception = memory.sensory_brief()
    source = source_evidence_brief("show me your code")
    for block in (perception, source):
        assert re.match(r"^\[[A-Z][^\]]+\]", block.strip()), block[:120]


# ── what is already in memory is scrubbed, not carried ────────────────────

from core.utils.injected_blocks import (  # noqa: E402
    contains_injected_block,
    strip_injected_blocks,
)


def test_a_fenced_evidence_block_is_removed():
    text = (
        "what's on my screen?\n\n"
        "[YOUR OWN RECENT PERCEPTION — NOTES, NOT A REPLY]\n"
        "You looked at the screen yourself moments ago.\n"
        "Windows open (front to back):\n- Aura\n"
        "[END YOUR OWN RECENT PERCEPTION]"
    )
    assert contains_injected_block(text)
    assert strip_injected_blocks(text) == "what's on my screen?"


def test_the_source_excerpt_block_is_removed():
    text = (
        "show me your code\n\n"
        "[YOUR OWN SOURCE — NOTES, NOT A REPLY]\n"
        "core/mycelium.py:88\n```python\nx = 1\n```\n"
        "[END YOUR OWN SOURCE]"
    )
    assert strip_injected_blocks(text) == "show me your code"


def test_the_desktop_contract_directives_are_removed():
    text = (
        "are you with me?\n\n"
        "[LIVE DESKTOP FULL-MIND CONTRACT]\n"
        "- Presence contract: answer with the phrase 'I'm here with you'\n"
        "[END LIVE DESKTOP FULL-MIND CONTRACT]"
    )
    assert strip_injected_blocks(text) == "are you with me?"


def test_an_unfenced_trailing_banner_is_removed():
    text = "summarise this\n\n[DIRECT RESULT]: {'ok': True}\n\nSynthesize this result."
    assert strip_injected_blocks(text) == "summarise this"


def test_ordinary_speech_is_untouched():
    for spoken in (
        "what's 17 times 4?",
        "Can you show me a snippet of your code that you're interested in?",
        "I was reading about [brackets] in a paper",
        "",
    ):
        assert strip_injected_blocks(spoken) == (spoken.strip() or spoken)
        assert not contains_injected_block(spoken)


def test_a_message_that_was_entirely_machinery_does_not_become_empty():
    """An empty user message is its own defect downstream."""
    text = "[YOUR OWN SOURCE — NOTES, NOT A REPLY]\nstuff\n[END YOUR OWN SOURCE]"
    assert strip_injected_blocks(text).strip()
