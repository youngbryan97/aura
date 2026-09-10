"""An answer must not say an animation is coming when none can be made.

LIVE, 2026-09-10, on "what is 17 percent of 4,280": the answer was correct and
ended "I am autonomously rendering a visual animation of this concept for you.
It will be available in the artifacts directory shortly." The log for the same
turn reads "2 validation errors for ManimInput". The renderer takes Manim source
and the name of the scene class in it; the caller passed a `task` string and a
timeout, neither of which is a field on its input. Every autonomous render since
it was written failed on validation, and every one of those answers made the
promise anyway.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.phases import response_generation_unitary as unitary
from core.skills.manim_renderer import ManimInput

_SOURCE = '''
from manim import Scene, Text


class SeventeenPercent(Scene):
    def construct(self):
        self.add(Text("17% of 4280 = 727.6"))
'''


def test_the_renderer_is_called_with_what_its_input_declares() -> None:
    """The fields the skill requires, built from what the generator returned."""
    source, scene = _SOURCE, "SeventeenPercent"
    params = ManimInput(python_code=source, scene_name=scene, quality="l")
    assert params.scene_name == "SeventeenPercent"
    assert "class SeventeenPercent(Scene)" in params.python_code


def test_the_scene_name_comes_out_of_the_source() -> None:
    assert unitary._A_SCENE_CLASS.search(_SOURCE).group(1) == "SeventeenPercent"


@pytest.mark.parametrize(
    "code",
    [
        "",
        "print('hello')",
        "class NotAScene:\n    pass\n",
    ],
)
def test_source_with_no_scene_renders_nothing(code: str, monkeypatch) -> None:
    """No scene means no render, and still nothing said to anyone."""

    class _Generator:
        def __init__(self, **_kwargs) -> None:
            pass

        def generate(self, _prompt, _context):
            return code

    monkeypatch.setattr(
        "core.brain.llm.code_generator.LLMCodeGenerator", _Generator, raising=True
    )
    assert unitary._manim_source_for("some answer") == ("", "")


def test_the_answer_never_promises_the_render() -> None:
    body = Path("core/phases/response_generation_unitary.py").read_text(encoding="utf-8")
    assert "rendering a visual animation" not in body
    assert "available in the artifacts directory" not in body
