"""Every degraded reply is written to continue a mood prefix.

    Mmm, that answer took too long to finish cleanly.

With no mood to continue from, the sentence has to start for itself.
Otherwise it reaches the person as a fragment, which is how

    that answer took too long to finish cleanly. I logged the timeout and
    preserved the turn context.

was read out, lowercase, mid-sentence. LIVE 2026-08-26.
"""
from __future__ import annotations

import inspect

from interface.routes.chat import _with_mood


def test_no_mood_means_the_sentence_starts_itself():
    assert _with_mood("", "that answer took too long.") == "That answer took too long."


def test_a_mood_carries_the_sentence_as_written():
    assert _with_mood("Mmm, ", "that answer took too long.") == "Mmm, that answer took too long."
    assert _with_mood("Hmm — ", "the lane is recovering.") == "Hmm — the lane is recovering."


def test_nothing_to_say_says_nothing():
    assert _with_mood("", "") == ""
    assert _with_mood("Mmm, ", "") == "Mmm, "


def test_every_degraded_reply_goes_through_it():
    """Seven of them, all written as continuations."""
    # Where the block lives now, not where it was written.
    #
    # This read `interface.routes.chat`, and the mood prefix moved to
    # `chat_lane_bookkeeping` when that module was split out — so the test
    # failed with `ValueError: substring not found`, which says nothing about
    # moods and sends the reader to the wrong file. The owning module is asked
    # of the function itself, so the next split does not break this again.
    import importlib

    from interface.routes import chat

    # Where the sentences live, not where they were written. They moved to
    # chat_lane_bookkeeping in a decomposition and this read chat.py, so a
    # test about seven degraded replies failed over a file that no longer
    # holds them. The owning module is asked of the function itself.
    chat_lane = importlib.import_module(chat._with_mood.__module__)
    source = inspect.getsource(chat_lane)
    where = source.index("# Build a mood-aware prefix for softer messages")
    block = source[where : where + 3000]
    assert block.count("_with_mood(_mood_prefix,") >= 7
    assert 'f"{_mood_prefix}' not in block, "a raw prefix left somewhere"
