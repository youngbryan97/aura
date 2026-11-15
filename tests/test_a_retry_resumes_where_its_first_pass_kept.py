"""A pass her worker may ask again keeps the prefix the retry shares with it.

LIVE 14 November 2026: a refused draft's retry matched the first pass's prompt for 2,114 of 2,172 tokens and read all
of them again (21.9 s), because the first pass, a complete miss, kept no prefix to resume from.
"""
from __future__ import annotations

from core.brain.llm.where_calls_share import where_the_last_turn_starts


class _Tokenizer:
    END = 7

    def encode(self, text, add_special_tokens=False):
        return [self.END] if text == "<|im_end|>" else [1, 2]


def test_the_last_turn_starts_after_the_message_before_it():
    # [history ... END] [readings ... END] [the person's turn ... END] [assistant opening]
    tokens = [1, 1, 7, 2, 2, 7, 3, 3, 3, 7, 4, 4]
    assert where_the_last_turn_starts(_Tokenizer(), tokens) == 6


def test_a_prompt_with_one_message_has_no_last_turn_to_keep_before():
    assert where_the_last_turn_starts(_Tokenizer(), [1, 1, 7, 4]) == 0
    assert where_the_last_turn_starts(_Tokenizer(), []) == 0
    assert where_the_last_turn_starts(None, [1, 7, 2, 7]) == 0
