"""Whether every position of the pass being read is a token she is speaking.

Her steering nudges the last position of a pass, since a pass of many positions has been a prompt being read, whose
last position alone is about to be spoken from (core/consciousness/affective_steering.py), and the Φ residual sampler
samples only passes of one position. Her own head's check reads two tokens she is speaking in one pass
(core/brain/llm/her_head_in_the_stream.py): MEASURED 20 October 2026 on her 27B, steered at its last position only,
the first of the two was unsteered and her words parted from her one-at-a-time decoding at token 59. Inside
``speaking_several`` every position is steered as a decoded token is, and Φ samples the pass.
"""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

__all__ = ["every_position_is_spoken", "speaking_several"]

_EVERY = ContextVar("every_position_is_spoken", default=False)


def every_position_is_spoken() -> bool:
    """Whether the pass being read is tokens she is speaking, every one, rather than a prompt."""
    return _EVERY.get()


@contextmanager
def speaking_several() -> Iterator[None]:
    """A pass in which every position is a token she is speaking."""
    token = _EVERY.set(True)
    try:
        yield
    finally:
        _EVERY.reset(token)
