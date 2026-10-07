"""A forward pass that reads her model and is not one of her generations.

Her steering hooks add an affective nudge at the completion position of every
forward, because in a generation that position produces the next token. The
hidden-sequence encoder runs the same forward to read a request's
representation and generates nothing, so the nudge there changes what a
reader sees without steering any output: a program reader fitted on her
unsteered states read different states whenever her affect was up. The
latent-bridge readouts that follow the steering hook would also count that
pass as her own activity and feed it back into her affect.

Inside ``observation_pass()`` the hooks that act on a forward stand down and
the hooks that learn from one do not record. It is thread-local: the worker
serializes its forwards, and a generation running on another thread keeps its
steering.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager

_STATE = threading.local()


@contextmanager
def observation_pass() -> Iterator[None]:
    depth = getattr(_STATE, "depth", 0)
    _STATE.depth = depth + 1
    try:
        yield
    finally:
        _STATE.depth = depth


def in_observation_pass() -> bool:
    return getattr(_STATE, "depth", 0) > 0


__all__ = ["in_observation_pass", "observation_pass"]
