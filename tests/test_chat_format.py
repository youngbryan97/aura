from core.brain.llm.chat_format import format_chatml_messages, format_chatml_prompt
from core.context.context_manager import ContextWindowManager


def test_grok_chat_template_uses_separator_format():
    rendered = format_chatml_messages(
        [
            {"role": "system", "content": "Stay factual."},
            {"role": "user", "content": "What is your name?"},
        ],
        model_name="grok-2",
    )

    assert rendered.startswith(
        "System: Stay factual.<|separator|>\n\nHuman: What is your name?<|separator|>\n\nAssistant:"
    )


def test_grok_chat_template_prompt_helper_uses_separator_format():
    rendered = format_chatml_prompt(
        "Ping",
        system_prompt="Be brief.",
        model_name="grok-2",
    )

    assert rendered == "System: Be brief.<|separator|>\n\nHuman: Ping<|separator|>\n\nAssistant:"


def test_context_window_manager_recognizes_grok_2_context_limit():
    manager = ContextWindowManager("grok-2")

    assert manager._raw_limit == 131_072


def test_continuation_messages_restore_open_assistant_turn_from_typed_state():
    from core.brain.llm.chat_format import normalize_chat_continuation_messages

    original = [
        {"role": "system", "content": "stable"},
        {"role": "user", "content": "Explain every requested part."},
    ]
    partial = "1. The first requested part is complete;"

    normalized = normalize_chat_continuation_messages(original, partial)

    assert normalized[-1] == {"role": "assistant", "content": partial}
    assert original[-1]["role"] == "user"


def test_continuation_messages_replace_stale_assistant_prefix_without_duplication():
    from core.brain.llm.chat_format import normalize_chat_continuation_messages

    messages = [
        {"role": "user", "content": "Explain every requested part."},
        {"role": "assistant", "content": "stale prefix"},
    ]

    normalized = normalize_chat_continuation_messages(messages, "authoritative prefix")

    assert normalized == [
        {"role": "user", "content": "Explain every requested part."},
        {"role": "assistant", "content": "authoritative prefix"},
    ]
