"""A system message in the wrong place killed the model lane.

LIVE 2026-08-19, mid-game:

    jinja2.exceptions.TemplateError: System message must be at the beginning.
    [MLX] Worker died during generation.
    Crash-loop breaker tripped (trip 1): lane backing off 30s
    Circuit OPEN for Brainstem after 4 failures.

She was playing 2048 and stopped. The person got "I couldn't get to an answer
I'd stand behind."

Chat templates disagree about where a system message may appear and some
raise rather than cope. That exception surfaces inside the worker process,
which dies mid-generation and takes the lane with it. A misordered list is a
caller's mistake, and the cost of it should be a reordered list.
"""
from __future__ import annotations

from core.brain.llm.chat_format import system_first


def test_a_system_message_arriving_late_is_moved_to_the_front():
    moved = system_first(
        [
            {"role": "user", "content": "a"},
            {"role": "system", "content": "s"},
            {"role": "user", "content": "b"},
        ]
    )
    assert [message["role"] for message in moved] == ["system", "user", "user"]


def test_nothing_is_dropped_or_merged():
    """A template that does accept them later must see the same content."""
    original = [
        {"role": "user", "content": "a"},
        {"role": "system", "content": "s1"},
        {"role": "assistant", "content": "b"},
        {"role": "system", "content": "s2"},
    ]
    moved = system_first(original)
    assert len(moved) == len(original)
    assert [message["content"] for message in moved] == ["s1", "s2", "a", "b"]


def test_a_conversation_already_in_order_is_returned_untouched():
    ordered = [{"role": "system", "content": "s"}, {"role": "user", "content": "a"}]
    assert system_first(ordered) is ordered


def test_a_conversation_with_no_system_message_is_untouched():
    plain = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
    assert system_first(plain) is plain


def test_anything_that_is_not_a_conversation_is_left_alone():
    assert system_first(None) is None
    assert system_first("a string") == "a string"
    assert system_first([]) == []


def test_objects_with_a_role_attribute_are_handled_too():
    class _Message:
        def __init__(self, role):
            self.role = role

    user, system = _Message("user"), _Message("system")
    assert system_first([user, system]) == [system, user]


def test_every_render_path_goes_through_it():
    """One unguarded path is enough to kill the worker again."""
    import inspect

    from core.brain.llm import chat_format

    source = inspect.getsource(chat_format)
    renders = [line for line in source.splitlines() if "str(apply(" in line]
    assert renders, "the render calls moved; this test needs to follow them"
    assert all("system_first(" in line for line in renders), renders
