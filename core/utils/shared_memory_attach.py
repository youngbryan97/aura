"""Attach to a shared-memory segment another process owns, without telling the resource tracker it is ours.

Python's tracker (before 3.13) registers a segment on every attach and unlinks what is still registered when its
process ends, so an attaching process used to unregister the name straight after. A child spawned by the owner shares
the owner's tracker, though, and the tracker keeps one entry per name: the child's unregister removes the owner's, and
the owner's own unlink then unregisters a name that is gone. LIVE 14 November 2026, her app printed
``KeyError: '/psm_da68655a'`` from the tracker's main loop during a voice turn. Not registering at all is right in
both cases: a tracker of the attacher's own has nothing to unlink, and a shared one keeps the owner's entry.
"""
from __future__ import annotations

import inspect
from multiprocessing import resource_tracker, shared_memory

from core.runtime.lockdep import checked_lock

# The tracker's register function is patched for the length of one attach; attaches are serialised.
_ATTACHING = checked_lock("shared_memory_attach.register")
_TAKES_TRACK = "track" in inspect.signature(shared_memory.SharedMemory.__init__).parameters


def attach(name: str) -> shared_memory.SharedMemory:
    """The segment named ``name``, attached and not registered with this process's resource tracker."""
    if _TAKES_TRACK:
        return shared_memory.SharedMemory(name=name, track=False)
    with _ATTACHING:
        original = resource_tracker.register

        def _register(resource: str, rtype: str) -> None:
            if rtype != "shared_memory":
                original(resource, rtype)

        resource_tracker.register = _register
        try:
            return shared_memory.SharedMemory(name=name)
        finally:
            resource_tracker.register = original


__all__ = ["attach"]
