"""Attaching to another process's shared memory does not register it, so the owner's tracker entry stays.

LIVE 14 November 2026 her app printed ``KeyError: '/psm_da68655a'`` from Python's resource tracker during a voice
turn: a reader attached to the speech hold and unregistered it from the tracker it shared with the owner, and the
owner's unlink then unregistered a name that was gone.
"""
from __future__ import annotations

from multiprocessing import resource_tracker, shared_memory

from core.utils.shared_memory_attach import attach


def test_an_attach_registers_nothing_and_reads_the_owners_bytes(monkeypatch):
    owner = shared_memory.SharedMemory(create=True, size=16)
    try:
        owner.buf[:4] = b"aura"
        registered = []
        monkeypatch.setattr(resource_tracker, "register", lambda name, rtype: registered.append((name, rtype)))
        reader = attach(owner.name)
        try:
            assert bytes(reader.buf[:4]) == b"aura"
            assert registered == []
        finally:
            reader.close()
    finally:
        owner.close()
        owner.unlink()


def test_the_tracker_is_patched_only_for_the_length_of_an_attach():
    before = resource_tracker.register
    owner = shared_memory.SharedMemory(create=True, size=8)
    try:
        attach(owner.name).close()
        assert resource_tracker.register is before
    finally:
        owner.close()
        owner.unlink()


def test_no_reader_of_another_process_unregisters_what_it_attached():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    for relative in ("core/brain/llm/speech_hold.py", "core/perception/seeing_play_elsewhere.py"):
        assert 'resource_tracker.unregister(' not in (root / relative).read_text(encoding="utf-8"), relative
