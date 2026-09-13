"""Three of the workspace's readings had no writer, because nothing called it.

`DiscourseTracker.update` says in its own docstring: "Call this after each
incoming user message." Nothing in the tree called it. The tracker was
constructed inside `core/social/presence_integration.py`, which is optional and
does not run in every runtime, so in most of them the service did not exist
either.

It is the only writer of `cognition.conversation_energy`,
`cognition.discourse_depth` and `cognition.user_emotional_trend`. All three
were constants for the life of every state — and the workspace prices the
exchange's claim on attention from the first of them, so what had just been
said asked for attention by exactly the same amount on every turn of every
conversation.
"""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_the_tracker_is_registered_wherever_the_container_is():
    source = (ROOT / "core" / "service_registration.py").read_text()
    assert "'discourse_tracker'" in source


def test_the_phase_that_has_the_message_calls_it():
    source = (ROOT / "core" / "phases" / "conversational_dynamics_phase.py").read_text()
    tree = ast.parse(source)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "update"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "tracker"
    ]
    assert calls, "nothing calls the tracker's update"


def test_the_tracker_writes_the_three_fields_nothing_else_writes():
    source = (ROOT / "core" / "brain" / "discourse_tracker.py").read_text()
    for field in (
        "conversation_energy",
        "discourse_depth",
        "user_emotional_trend",
    ):
        assert f"state.cognition.{field}" in source, field


def test_no_other_writer_appeared_for_conversation_energy():
    """If one does, this file's premise needs rechecking rather than keeping."""
    writers = []
    for path in (ROOT / "core").rglob("*.py"):
        if path.name == "discourse_tracker.py":
            continue
        text = path.read_text(errors="ignore")
        if "cognition.conversation_energy =" in text or "cognition.conversation_energy=" in text:
            writers.append(str(path.relative_to(ROOT)))
    assert not writers, f"conversation energy now has other writers: {writers}"
