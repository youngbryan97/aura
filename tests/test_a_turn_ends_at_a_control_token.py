"""A turn ends where the model emits a token its tokenizer marks as structure, never at a word.

LIVE 16 November 2026, 21:41 and 21:51 local: asked "Could you put up a desktop notification that says time to
stretch?", her model wrote ``<tool_call>`` ``<function=notify`` and stopped, twice, and nothing was sent. Her tokenizer
splits ``notify_user`` into ``notify`` and ``_user``, and a pattern over the decoded text ended every reply at the first
"_user" anywhere in it. It had been written in May for a reply that came back "8_user".
"""
from __future__ import annotations

from types import SimpleNamespace

from core.brain.llm.mlx_worker_surface_markers import (
    _truncate_reply_role_continuation,
    _truncate_role_continuation,
    _turns_end_at_its_control_tokens,
)

TOOL_CALL = "<tool_call>\n<function=notify_user>\n<parameter=message>\ntime to stretch\n</parameter>\n</function>\n</tool_call>"


def test_a_call_to_notify_user_is_written_whole():
    assert _truncate_role_continuation(TOOL_CALL, final=True) == (TOOL_CALL, False)
    for end in range(1, len(TOOL_CALL) + 1):  # after every token, as the decode loop asks
        assert _truncate_reply_role_continuation(TOOL_CALL[:end], native_thinking=False) == (TOOL_CALL[:end], False)


def test_the_next_turn_begun_in_the_text_is_still_cut():
    assert _truncate_role_continuation("Done.<|im_end|>\n<|im_start|>user\nthanks", final=True) == ("Done.", True)


class _HerTokenizer:
    """The shape of mlx_lm's TokenizerWrapper over her Qwen3 tokenizer: special flags as in her tokenizer.json."""

    def __init__(self):
        self.eos_token_ids = {248046}
        decoder = {248044: ("<|endoftext|>", True), 248045: ("<|im_start|>", True), 248046: ("<|im_end|>", True),
                   248056: ("<|image_pad|>", True), 248058: ("<tool_call>", False), 248059: ("</tool_call>", False),
                   248068: ("<think>", False), 248069: ("</think>", False)}
        self._tokenizer = SimpleNamespace(added_tokens_decoder={
            token_id: SimpleNamespace(content=content, special=special)
            for token_id, (content, special) in decoder.items()})

    def add_eos_token(self, token: str):
        self.eos_token_ids.add(int(token))


def test_every_token_her_tokenizer_marks_special_ends_a_decode_and_no_other():
    model, tokenizer = object(), _HerTokenizer()

    assert _turns_end_at_its_control_tokens(model, tokenizer) == (model, tokenizer)
    assert tokenizer.eos_token_ids == {248044, 248045, 248046, 248056}, "<tool_call> and <think> are text she writes"


def test_a_tokenizer_without_the_table_is_left_as_it_was():
    plain = SimpleNamespace(eos_token_ids={2})
    assert _turns_end_at_its_control_tokens("m", plain) == ("m", plain) and plain.eos_token_ids == {2}


def test_every_load_her_worker_makes_ends_turns_at_control_tokens():
    import inspect

    from core.brain.llm import mlx_worker

    source = inspect.getsource(mlx_worker._mlx_worker_loop)
    loads = source[source.index("def load(model_path, adapter_path=None):"):source.index("from mlx_lm.sample_utils")]
    returns = [line.strip() for line in loads.splitlines() if line.strip().startswith("return ")]
    assert returns and all(line.startswith("return _turns_end_at_its_control_tokens(*") for line in returns), returns
