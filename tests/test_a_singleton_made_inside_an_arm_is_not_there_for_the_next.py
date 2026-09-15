"""A module global that was empty when the snapshot was taken is empty again after the restore.

The fork carries every object the organism keeps at module scope, and it found
them by what they held. A slot still holding None when the snapshot was taken
held nothing, so nothing was carried, and a ledger or engine made on first use
inside one arm was there, with that arm's history in it, when the next arm
began. The frisson ledger and the scientific engine's cache did exactly that
whenever an earlier test had emptied their slots.
"""

from __future__ import annotations

import sys
import types

import pytest

from core.subject.snapshot import _empty_again, _empty_module_slots

PROBE = "core._fork_probe_empty_slots"


class _Ledger:
    def __init__(self) -> None:
        self.seen = ["first arm"]


_Ledger.__module__ = PROBE


@pytest.fixture
def probe(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    module = types.ModuleType(PROBE)
    module._LEDGER = None
    module._optional_import = None
    module.PUBLIC = None
    monkeypatch.setitem(sys.modules, PROBE, module)
    return module


def test_an_empty_private_slot_is_recorded(probe) -> None:
    slots = _empty_module_slots()
    assert f"{PROBE}:_LEDGER" in slots
    assert f"{PROBE}:PUBLIC" not in slots


def test_a_singleton_an_arm_made_is_emptied_again(probe) -> None:
    slots = _empty_module_slots()
    probe._LEDGER = _Ledger()
    emptied = _empty_again(slots)
    assert probe._LEDGER is None
    assert f"{PROBE}:_LEDGER" in emptied


def test_a_module_imported_into_an_empty_slot_is_left_alone(probe) -> None:
    slots = _empty_module_slots()
    probe._optional_import = types.ModuleType("json_lookalike")
    _empty_again(slots)
    assert probe._optional_import is not None


def test_only_the_keys_asked_about_are_checked(probe) -> None:
    probe._LEDGER = _Ledger()
    assert _empty_module_slots([f"{PROBE}:_LEDGER"]) == frozenset()
    probe._LEDGER = None
    assert _empty_module_slots([f"{PROBE}:_LEDGER"]) == frozenset({f"{PROBE}:_LEDGER"})
