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


def test_multiple_system_messages_become_one_canonical_system_block():
    """Strict templates permit one system message, not merely a system prefix."""
    original = [
        {"role": "user", "content": "a"},
        {"role": "system", "content": "s1"},
        {"role": "assistant", "content": "b"},
        {"role": "system", "content": "s2"},
    ]
    moved = system_first(original)
    assert len(moved) == 3
    assert [message["role"] for message in moved] == ["system", "user", "assistant"]
    assert moved[0]["content"] == "s1\n\ns2"
    assert [message["content"] for message in moved[1:]] == ["a", "b"]


def test_developer_authority_is_folded_into_the_canonical_system_block():
    moved = system_first(
        [
            {"role": "system", "content": "identity"},
            {"role": "user", "content": "question"},
            {"role": "developer", "content": "runtime state"},
        ]
    )
    assert moved == [
        {"role": "system", "content": "identity\n\nruntime state"},
        {"role": "user", "content": "question"},
    ]


def test_strict_single_system_template_accepts_the_normalized_transcript():
    class _StrictTokenizer:
        def apply_chat_template(self, messages, **_kwargs):
            assert messages[0]["role"] == "system"
            assert all(message["role"] != "system" for message in messages[1:])
            assert all(message["role"] != "developer" for message in messages)
            return "rendered"

    from core.brain.llm.chat_format import render_chat_template

    rendered = render_chat_template(
        _StrictTokenizer(),
        [
            {"role": "system", "content": "identity"},
            {"role": "user", "content": "question"},
            {"role": "system", "content": "runtime state"},
        ],
    )
    assert rendered == "rendered"


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

    for render in (
        chat_format.render_chat_template,
        chat_format.render_chat_continuation_template,
    ):
        source = inspect.getsource(render)
        assert "system_first(messages)" in source
        assert "normalize_tool_transcript_for_template" in source


def test_mapping_tool_template_receives_an_object_without_mutating_history():
    class _MappingTokenizer:
        chat_template = "tool_call.arguments|items"

        def __init__(self):
            self.seen = []

        def apply_chat_template(self, messages, **_kwargs):
            for message in messages:
                for call in message.get("tool_calls", ()):
                    arguments = call["function"]["arguments"]
                    list(arguments.items())
                    self.seen.append(arguments)
            return "rendered"

    from core.brain.llm import chat_format

    chat_format._TOOL_ARGUMENT_MODE.clear()
    tokenizer = _MappingTokenizer()
    messages = [
        {"role": "user", "content": "search"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "web_search",
                        "arguments": '{"query":"current release"}',
                    },
                }
            ],
        },
    ]

    assert chat_format.render_chat_template(tokenizer, messages) == "rendered"
    assert tokenizer.seen[-1] == {"query": "current release"}
    assert isinstance(messages[-1]["tool_calls"][0]["function"]["arguments"], str)


def test_legacy_tool_template_receives_a_json_string_from_canonical_state():
    class _StringTokenizer:
        chat_template = "legacy-json-string-arguments"

        def __init__(self):
            self.seen = []

        def apply_chat_template(self, messages, **_kwargs):
            for message in messages:
                for call in message.get("tool_calls", ()):
                    arguments = call["function"]["arguments"]
                    if not isinstance(arguments, str):
                        raise TypeError("arguments must be JSON text")
                    self.seen.append(arguments)
            return "rendered"

    import json

    from core.brain.llm import chat_format

    chat_format._TOOL_ARGUMENT_MODE.clear()
    tokenizer = _StringTokenizer()
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "code_repl",
                        "arguments": {"code": "print(7)"},
                    },
                }
            ],
        }
    ]

    assert chat_format.render_chat_template(tokenizer, messages) == "rendered"
    assert json.loads(tokenizer.seen[-1]) == {"code": "print(7)"}
    assert isinstance(messages[0]["tool_calls"][0]["function"]["arguments"], dict)


def test_the_assemblers_own_note_goes_with_the_system_content():
    """This is the caller that actually built the list that killed the worker.

    The omission notice was appended, so it landed after the whole
    conversation — and it only fires once a run has gone on long enough to
    drop messages, which is why it struck mid-game.
    """
    from core.brain.llm.context_assembler import _place_system_note

    messages = [
        {"role": "system", "content": "canonical"},
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
    ]
    _place_system_note(messages, "3 messages omitted")
    assert [message["role"] for message in messages] == ["system", "system", "user", "assistant"]
    assert messages[1]["content"] == "3 messages omitted"


def test_the_note_still_goes_first_when_there_is_no_system_content():
    from core.brain.llm.context_assembler import _place_system_note

    messages = [{"role": "user", "content": "a"}]
    _place_system_note(messages, "note")
    assert [message["role"] for message in messages] == ["system", "user"]


def test_the_trimmer_cannot_kill_the_worker_with_a_template_error():
    """LIVE, three times in one run: jinja2.TemplateError is not an
    AttributeError, RuntimeError, TypeError or ValueError, so it went past
    the guard, out of the worker loop, and killed the worker mid-generation.

    Failing to trim is recoverable — the caller keeps the untrimmed prompt
    and finds out it is too long. Having no model is not.
    """
    import inspect

    from core.brain.llm import mlx_worker

    source = inspect.getsource(mlx_worker)
    where = source.index("def _render(candidate_messages")
    body = source[where : where + 1200]
    assert "system_first(" in body, "the trimmer renders without normalising order"
    assert "except Exception" in body, "a template refusal still escapes the trimmer"


def test_native_template_failure_cannot_escape_the_generation_job():
    import inspect

    from core.brain.llm import mlx_worker

    source = inspect.getsource(mlx_worker._mlx_worker_loop)
    start = source.index('logger.info("🎯 [WORKER] Rendering native chat/tool template.")')
    end = source.index('temp = _admit_sampling_control(job, "temp")', start)
    body = source[start:end]
    assert "except Exception" in body
    assert '"chat_template_failed_with_tools:"' in body
    assert '"status": "error"' in body
    assert "continue" in body
