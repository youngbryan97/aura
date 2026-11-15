"""Her play child's messages are read by an unpickler that builds only what a message between them is made of.

14 November 2026: the release gate's scanner refused plain pickle.loads in core/perception/seeing_play_elsewhere.py.
A pickle can name any class in the interpreter and have it built; this pipe carries her own perception state, numpy
arrays and plain containers.
"""
from __future__ import annotations

import os
import pickle
from collections import deque

import numpy as np
import pytest

from core.perception import seeing_play_elsewhere as elsewhere
from core.perception.what_moves_in_the_picture import Thing


def test_what_her_child_sends_comes_back_whole():
    look = np.arange(64, dtype=np.float32)
    thing = Thing(1, 10.0, 20.0, 14.0, 14.0, look, (170, 220, 140), np.zeros((1, 1, 3), np.uint8),
                  born=0.0, seen=0.5, kind=0, moved=True)
    message = ("seen", {"frames": 3, "recent": deque([1, 2])}, {1: thing, "kinds": {frozenset({1, 2})}})

    back = elsewhere._loads(pickle.dumps(message, protocol=pickle.HIGHEST_PROTOCOL))

    assert back[0] == "seen" and back[1]["recent"] == deque([1, 2])
    assert isinstance(back[2][1], Thing) and np.array_equal(back[2][1].look, look)


def test_a_message_that_names_anything_else_is_refused():
    class Builds:
        def __reduce__(self):
            return (os.system, ("true",))

    with pytest.raises(pickle.UnpicklingError, match="not part of a message from her play child"):
        elsewhere._loads(pickle.dumps(Builds()))


def test_a_plain_address_is_not_a_secret_but_one_carrying_a_credential_still_is():
    from tools import security_scan

    assert security_scan._is_plain_url_literal("https://www.reddit.com/api/v1/access_token")
    assert not security_scan._is_plain_url_literal("https://user:hunter2-secret@example.com/api")
    assert not security_scan._is_plain_url_literal("https://example.com/api?token=8f3kd92kfj39dk2l")
