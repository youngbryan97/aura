"""A reply holding code AND prose must still be repairable.

Live 2026-08-18: "write a python function to reverse a string and then explain
how it works" ran for 112 seconds and returned

    I couldn't get to an answer I'd stand behind on that one, and I won't send
    you a thinner one and pass it off as the real thing.

The worker had rejected the draft as escaped_control_artifact, exhausted its
retries, and the gate refused with lanes_exhausted. The draft was fine; the
model had typed one literal backslash-n in the explanation.

Two guards, each written for a real defect, closed over every request that
wants code and prose together:

  * the repair declined to run at all when the reply contained a fence,
    because rewriting inside code would corrupt it (correct concern, wrong
    scope: hold the fences out and repair around them);
  * it skipped any escape followed by a letter, so \\text would not become a
    tab (2026-07-26, "P(\\text{same color})" served as "P( ext{same color})").
    But a letter after the escape is the COMMON case — "backwards.\\nThe
    complexity" — so the guard covered ordinary sentences too.

Both original defects are pinned below alongside the new one.
"""

from __future__ import annotations

from core.brain.llm.mlx_worker import (
    _repair_live_user_surface_escaped_newlines as repair,
)


def test_a_reply_with_code_and_prose_is_repaired() -> None:
    text = (
        "Here is the function:\n\n"
        "```python\ndef rev(s):\n    return s[::-1]\n```\n\n"
        "It slices backwards.\\nThe complexity is O(n)."
    )

    out = repair(text)

    assert out, "a fenced reply was left unrepairable"
    assert "It slices backwards.\nThe complexity" in out


def test_the_code_block_is_returned_byte_exact() -> None:
    body = "def rev(s):\n    return s[::-1]\n"
    text = f"Here:\n\n```python\n{body}```\n\nExplained.\\nLine two."

    out = repair(text)

    assert body in out, "the repair altered the code block"


def test_a_literal_escape_inside_code_is_left_alone() -> None:
    """Code may legitimately contain the two characters."""
    body = 'def f():\n    return "a\\nb"\n'
    text = f"Look:\n\n```python\n{body}```\n\nDone.\\nReally done."

    out = repair(text)

    assert body in out


def test_latex_is_still_not_mistaken_for_whitespace() -> None:
    """The 2026-07-26 defect: "P( ext{same color})"."""
    text = r"The probability is \(P(\text{same color}) = 1/3\)."

    assert not repair(text)


def test_other_latex_commands_survive() -> None:
    text = r"Rate is $x \times 2$, $\rho$ is density, and $x \neq y$ holds."

    assert not repair(text)


def test_plain_prose_is_still_repaired() -> None:
    out = repair("Two lines.\\nSecond line.")

    assert out == "Two lines.\nSecond line."


def test_validation_and_repair_share_the_same_markup_boundary() -> None:
    from core.conversation.escaped_controls import has_escaped_whitespace_artifact

    math = r"Use $x \neq y$ and \(P(\text{same color})\)."
    prose = r"First line.\nSecond line."

    assert not has_escaped_whitespace_artifact(math)
    assert has_escaped_whitespace_artifact(prose)
    assert repair(prose) == "First line.\nSecond line."


def test_a_reply_with_nothing_to_repair_reports_nothing() -> None:
    assert not repair("An ordinary sentence with no escapes.")
    assert not repair(r"A regular expression may contain \\n as two escaped characters.")
    assert not repair("")
