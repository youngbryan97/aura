"""The Imagine panel showed an empty frame over a frame that was not empty.

`ImaginationEngine.snapshot` withholds the latest frame's content by default,
because a status route reachable by anyone is not a place to publish what one
person was thinking about. That was right and it was applied to everybody,
including the one person entitled to it: the owner, reading their own panel on
their own machine.

So the panel reported "(no objective)", "no objects in this frame" and "no
attractor competition in this frame" while the engine held all three, and the
render button — which reads `mental_canvas.image_prompt` out of the same
withheld payload — could never find a prompt to send.

`for_owner` is an assertion the caller makes. These tests are what make it a
checked one.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[1]


def _a_frame():
    from core.brain.imagination import ImaginationFrame

    return ImaginationFrame(
        frame_id="f1",
        objective="what a connectome is for",
        mode="associative",
        salience=0.8,
        novelty_pressure=0.5,
        curiosity_pressure=0.5,
        affective_pressure=0.1,
        memory_pressure=0.2,
        verification_pressure=0.3,
        attractor_state={"subject": "connectome", "selected": "associative"},
        mental_canvas={"image_prompt": "a lattice of lit filaments", "objects": ["lattice"]},
    )


def _engine_holding(frame):
    from core.brain.imagination import ImaginationEngine

    engine = ImaginationEngine()
    with engine._state_lock:
        engine._history.append(frame)
        engine._frame_count += 1
    return engine


def test_the_owner_sees_the_frame_and_a_stranger_sees_its_shape():
    engine = _engine_holding(_a_frame())

    stranger = engine.snapshot()["latest"]
    assert stranger["content_withheld"] is True
    assert "objective" not in stranger
    assert "mental_canvas" not in stranger

    owner = engine.snapshot(for_owner=True)["latest"]
    assert owner["objective"] == "what a connectome is for"
    assert owner["mental_canvas"]["image_prompt"] == "a lattice of lit filaments"
    assert owner["attractor_state"]["selected"] == "associative"


def test_naming_the_subject_still_works_without_claiming_to_be_the_owner():
    engine = _engine_holding(_a_frame())
    named = engine.snapshot(subject="connectome", include_content=True)["latest"]
    assert named["objective"] == "what a connectome is for"
    wrong = engine.snapshot(subject="something else", include_content=True)["latest"]
    assert wrong["content_withheld"] is False or "objective" not in wrong


def test_the_render_button_can_reach_the_prompt_it_exists_to_send():
    """The visualize route reads the canvas out of the same payload."""
    from interface.routes.system import _collect_imagination_status

    from core.container import ServiceContainer

    engine = _engine_holding(_a_frame())
    ServiceContainer.register_instance("imagination_engine", engine, required=False)
    try:
        assert _collect_imagination_status()["latest"].get("mental_canvas") is None
        owner = _collect_imagination_status(for_owner=True)
        assert owner["latest"]["mental_canvas"]["image_prompt"]
    finally:
        ServiceContainer.clear()


def test_every_caller_that_claims_to_be_the_owner_has_checked():
    """The flag is an assertion, so it is read off the source rather than trusted.

    A call passing `for_owner` must do one of three things: pass a value
    derived from `_owner_authenticated`, forward its own function's parameter
    of the same name, or sit inside a function that has already refused a
    request that is not the owner's.
    """
    checked = 0
    for path in sorted((REPO / "interface").rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            claims = [
                keyword
                for keyword in node.keywords
                if keyword.arg == "for_owner"
            ]
            if not claims:
                continue
            checked += 1
            value = claims[0].value
            derived = (
                isinstance(value, ast.Call)
                and getattr(value.func, "id", "") == "_owner_authenticated"
            )
            if derived:
                continue
            holder = _enclosing(tree, node)
            if isinstance(value, ast.Name) and holder is not None:
                # Forwarding a parameter of the same name decides nothing; the
                # claim is made wherever that parameter was filled in.
                arguments = holder.args
                names = {
                    one.arg
                    for one in (
                        *arguments.args,
                        *arguments.kwonlyargs,
                        *arguments.posonlyargs,
                    )
                }
                if value.id in names:
                    continue
            assert holder is not None and _refuses_a_stranger(holder), (
                f"{path.relative_to(REPO)} passes for_owner without checking the owner"
            )
    assert checked >= 2, "the flag has no call sites; this test would pass on nothing"


def _enclosing(tree: ast.AST, target: ast.Call):
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for inner in ast.walk(node):
            if inner is target:
                return node
    return None


def _refuses_a_stranger(function) -> bool:
    """Does this function raise unless the caller is the owner?"""
    for node in ast.walk(function):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if not (isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not)):
            continue
        called = test.operand
        if (
            isinstance(called, ast.Call)
            and getattr(called.func, "id", "") == "_owner_authenticated"
            and any(isinstance(one, ast.Raise) for one in ast.walk(node))
        ):
            return True
    return False
