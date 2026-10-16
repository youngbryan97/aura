"""A reply does not copy her own earlier replies, and is free to quote anybody else.

Live, 13 October: three different invitations each got the same paragraph
back, her model writing out its own earlier answer from the transcript. The
mechanism is core/brain/llm/not_her_own_words_again.py (DRY, Weidmann 2024).
"""
from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")

from core.brain.llm.not_her_own_words_again import (  # noqa: E402
    ALLOWED_LENGTH,
    BASE,
    MULTIPLIER,
    her_earlier_turns,
    not_her_own_words_again,
)


class Words:
    """One token per word, so a test can say which word is penalised."""

    def __init__(self) -> None:
        self.ids: dict[str, int] = {}

    def encode(self, text, add_special_tokens=False):
        return [self.ids.setdefault(word, len(self.ids)) for word in str(text).split()]


EARLIER = "I am ready . I have my eyes closed , but my attention is fully open"


def _after(words: Words, hold, prompt: list[int], written: str):
    """The logits the processor leaves after the prompt and what was written so far."""
    tokens = list(prompt)
    out = hold(mx.array(tokens), mx.zeros((1, 64)))  # the first call carries the prompt
    for word in written.split():
        tokens.append(words.encode(word)[0])
        out = hold(mx.array(tokens), mx.zeros((1, 64)))
    return out


def test_carrying_on_a_copy_is_penalised_more_the_longer_the_copy():
    words = Words()
    hold = not_her_own_words_again(words, [EARLIER])
    prompt = words.encode("hey do you want to watch this")
    two = _after(words, hold, prompt, "I am")
    assert float(two[0, words.ids["ready"]]) == pytest.approx(-MULTIPLIER)
    five = _after(words, not_her_own_words_again(words, [EARLIER]), prompt, "I am ready . I")
    assert float(five[0, words.ids["have"]]) == pytest.approx(-MULTIPLIER * BASE ** (5 - ALLOWED_LENGTH))
    # Nothing else is touched.
    assert float(five[0, words.ids["eyes"]]) == 0.0


def test_one_shared_word_costs_nothing():
    words = Words()
    hold = not_her_own_words_again(words, [EARLIER])
    out = _after(words, hold, words.encode("so"), "I")
    assert float(mx.min(out)) == 0.0


def test_only_her_own_earlier_turns_count():
    messages = [
        {"role": "user", "content": "I am ready . I have my eyes closed"},
        {"role": "assistant", "content": "The heron waited ."},
        {"role": "runtime_evidence", "content": "the screen shows a heron"},
        {"role": "user", "content": "Want to watch it with me ?"},
    ]
    assert her_earlier_turns(messages) == ["The heron waited ."]
    words = Words()
    hold = not_her_own_words_again(words, her_earlier_turns(messages))
    # Quoting what he said, word for word, is not repeating herself.
    out = _after(words, hold, words.encode("Want to watch it"), "I am ready . I have my")
    assert float(mx.min(out)) == 0.0


def test_with_nothing_said_before_there_is_nothing_to_hold():
    assert not_her_own_words_again(Words(), []) is None
    assert not_her_own_words_again(Words(), ["ok"]) is None


def test_a_reply_somebody_waits_for_gets_the_hold_and_an_internal_one_does_not():
    import logging

    from core.brain.llm.what_the_decoder_holds import hold_the_answer

    words = Words()
    job = {"clean_user_surface_contract": True, "messages": [{"role": "assistant", "content": EARLIER}]}
    held: list = []
    hold_the_answer(job, words, None, held, logging.getLogger("t"), lambda *a, **k: None)
    assert len(held) == 1 and held[0].__name__ == "not_again"
    internal: list = []
    hold_the_answer({"messages": job["messages"]}, words, None, internal, logging.getLogger("t"), lambda *a, **k: None)
    assert internal == []


class Digits(Words):
    """Words as one token each, numbers one digit a token, as her tokenizer writes them."""

    def encode(self, text, add_special_tokens=False):
        out = []
        for word in str(text).split():
            pieces = list(word) if word.isdigit() else [" " + word]
            out += [self.ids.setdefault(piece, len(self.ids)) for piece in pieces]
        return out

    def decode(self, ids):
        names = {index: piece for piece, index in self.ids.items()}
        return "".join(names[i] for i in ids)


def test_a_number_already_begun_is_finished_as_written():
    """Live 16 October: asked the same gcd again, she wrote "486" for 4862, the digits held back as a copy."""
    words = Digits()
    hold = not_her_own_words_again(words, ["7854 = 1 × 4862 + 2992"])
    prompt = words.encode("what is the gcd of 4862 and 7854")
    tokens = list(prompt)
    hold(mx.array(tokens), mx.zeros((1, 64)))
    for piece in ["4", "8", "6"]:
        tokens.append(words.ids[piece])
        out = hold(mx.array(tokens), mx.zeros((1, 64)))
    assert float(out[0, words.ids["2"]]) == 0.0
    # A word that would carry a copied stretch on is still pushed down.
    tokens.append(words.ids["2"])
    out = hold(mx.array(tokens), mx.zeros((1, 64)))
    assert float(out[0, words.ids[" +"]]) < 0.0
