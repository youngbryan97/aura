"""An object that holds operating-system descriptors as plain integers.

A descriptor held as an int looks like data to anything that copies objects.
A copy of the holder is a second owner of the same descriptor numbers, and the
two close them independently: the first to be collected closes descriptors the
other is still using, and when the other closes them in turn the numbers may
already belong to something else.

That is how the first whole report run died on 23 September at 00:23. The
subject fork deep-copied the latent cortex's private action-snapshot store with
the rest of the organism, the discarded copy's finalizer closed the store's
descriptors, and a later close landed on a number Metal had since taken and
guarded. The kernel killed the process with EXC_GUARD (GUARD_TYPE_FD, CLOSE on
descriptor 39). Under the stub organ the same double close goes unseen and
closes somebody else's file.

A class that owns descriptors says so by inheriting this. A copy of it is the
object itself, and the subject fork treats it as process furniture, the way it
treats an open file or a socket: never carried, never duplicated.
"""

from __future__ import annotations

from typing import Any

__all__ = ["OwnsDescriptors"]


class OwnsDescriptors:
    """Holds OS descriptors as ints; copying yields the same object."""

    __slots__ = ()

    def __copy__(self) -> Any:
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> Any:
        memo[id(self)] = self
        return self
