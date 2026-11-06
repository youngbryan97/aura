"""A picture already looked at is not looked at again: what was found in it is kept by the picture itself.

A person who glances back at a screen that has not changed does not read it again; they know it is the one they read.
Her looks did read it again. Waiting for a screen to come to rest takes it twice and compares; a look after an act
that changed nothing takes the same screen; a menu returned to is the menu read before. Measured offline 2026-10-12
over two games, reading the screen was 56% of all her time, a look at a screen 1.7 seconds, and the words in it were
read about twice a look.

So what is found in a picture by the ways of looking that depend on nothing but the picture (its words, the shapes
drawn in it, what looks pressable) is kept, by the picture's own content: every pixel of it, digested. The same
pixels are the same picture and get the same findings, exactly as reading them again would give them (those ways of
looking give the same answer for the same pixels every time); a picture different in one pixel is another picture,
and is read. Nothing is guessed, so nothing is lost: only the time.

What is kept is copied out each time, so nobody who marks up a finding marks up the one kept.
"""
from __future__ import annotations

import copy
import hashlib
from collections import OrderedDict
from collections.abc import Callable
from typing import Any

import numpy as np
from core.runtime.lockdep import checked_lock

__all__ = ["KEPT", "how_often_again", "picture_key", "read_once"]

#: How many pictures' findings are kept for each way of looking, the latest looked at kept longest.
KEPT = 48

_KEPT: dict[str, OrderedDict[str, Any]] = {}
_AGAIN: dict[str, list[int]] = {}
_LOCK = checked_lock("a_picture_read_once.cache")


def picture_key(picture: Any) -> str:
    """The picture's content as a short key: every pixel, its shape and its kind of value."""
    pixels = np.ascontiguousarray(np.asarray(picture))
    digest = hashlib.blake2b(pixels.data, digest_size=16)
    digest.update(repr((pixels.shape, pixels.dtype.str)).encode())
    return digest.hexdigest()


def read_once[T](way: str, picture: Any, look: Callable[[], T], *also: Any) -> T:
    """What ``look`` finds in ``picture``, the way of looking named ``way``: found once for each picture (and each
    ``also``, whatever else the finding depends on), kept, and given again for the same pixels."""
    key = picture_key(picture) + (repr(also) if also else "")
    with _LOCK:
        kept = _KEPT.setdefault(way, OrderedDict())
        counts = _AGAIN.setdefault(way, [0, 0])
        if key in kept:
            kept.move_to_end(key)
            counts[0] += 1
            found = kept[key]
            return copy.deepcopy(found)
        counts[1] += 1
    found = look()
    with _LOCK:
        kept[key] = copy.deepcopy(found)
        while len(kept) > KEPT:
            kept.popitem(last=False)
    return found


def how_often_again() -> dict[str, tuple[int, int]]:
    """For each way of looking: how many pictures were looked at again, and how many anew."""
    with _LOCK:
        return {way: (again, anew) for way, (again, anew) in _AGAIN.items()}
