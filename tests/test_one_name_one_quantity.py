"""Three numbers called "energy", in one prompt, from two organs.

MEASURED live 2026-08-18. Asked "what's your energy reading right now? one
number.", her context held all of these at once:

    [Affect: Current Mood: TIRED (Energy: 0.14, Focus: 0.50, ...)]  field, 0-1
    Energy: 14.0                                                    field, 0-100
    [Measured ... interoception=yes (... energy 0.647 ...)]         soma reserve

The cognitive field's energy and the metabolic reserve are different quantities
owned by different organs, and three renderers published them under one word,
differing by more than four times. No answer she could give was right: whichever
number she picked, the guard that owns the other one calls it a fabrication. Her
mood reads TIRED off the field while the reserve says she is fine.

A name collision is not something a better-worded instruction can repair — two
measurements that share a name are ONE measurement as far as anything
downstream can tell. So each organ's quantity carries its organ, and every one
of them is registered in the single instrument registry that the contradiction
guard reads.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1]


def _read(relative: str) -> str:
    return (SOURCE / relative).read_text(encoding="utf-8")


def test_the_substrate_summary_names_its_organ():
    from types import SimpleNamespace
    from unittest import mock

    from core.consciousness.liquid_substrate import LiquidSubstrate

    substrate = LiquidSubstrate.__new__(LiquidSubstrate)
    substrate.idx_energy = 5
    substrate.idx_focus = 6
    snapshot = {
        "x": {5: 0.14, 6: 0.50},
        "snapshot_age_s": 0.1,
        "freshness_threshold_s": 5.0,
    }
    with mock.patch.object(
        LiquidSubstrate, "_state_snapshot_nowait", lambda self, *a, **k: snapshot
    ), mock.patch.object(LiquidSubstrate, "get_mood", lambda self: "TIRED"):
        summary = substrate.get_summary()

    assert "substrate energy" in summary
    assert "substrate focus" in summary
    assert not re.search(r"\(Energy:", summary), summary


def test_no_renderer_emits_a_bare_energy_label():
    """The collision is a naming fact, so it is checked on the source."""
    for relative in (
        "core/consciousness/liquid_substrate.py",
        "interface/routes/chat_protected_prompt.py",
    ):
        text = _read(relative)
        assert '_compact_snapshot_line("Energy"' not in text, relative
        assert "(Energy: {" not in text, relative


def test_the_registry_names_the_field_quantities_separately():
    from unittest import mock

    from types import SimpleNamespace

    import core.self.capability_ledger as cl

    class _Container:
        @staticmethod
        def peek(name, default=None):
            if name == "liquid_substrate":
                return SimpleNamespace(current=SimpleNamespace(energy=0.14, focus=0.50))
            return default

    with mock.patch.dict("sys.modules", {"core.container": SimpleNamespace(ServiceContainer=_Container)}):
        reading = cl._substrate_reading()

    assert reading == {"substrate_energy": 0.14, "substrate_focus": 0.5}
    assert "energy" not in reading, "the bare name belongs to the soma reserve"


def test_an_absent_substrate_contributes_nothing():
    """Silence must never become a zeroed field."""
    from types import SimpleNamespace
    from unittest import mock

    import core.self.capability_ledger as cl

    class _Container:
        @staticmethod
        def peek(name, default=None):
            return default

        @staticmethod
        def get(name, default=None):
            return default

    with mock.patch.dict("sys.modules", {"core.container": SimpleNamespace(ServiceContainer=_Container)}):
        assert cl._substrate_reading() == {}


def test_both_quantities_are_checkable_against_their_own_owner():
    """The point of the split: each number is verified by the organ that owns it."""
    from unittest import mock

    import core.self.capability_ledger as cl

    measured = {"energy": 0.647, "substrate_energy": 0.14}
    with mock.patch.object(cl, "measured_self_metrics", lambda: dict(measured)):
        assert cl.contradicted_self_readings("My energy is 0.647.") == []
        assert cl.contradicted_self_readings("Substrate energy is 0.14.") == []
        wrong = cl.contradicted_self_readings("My energy is 0.14.")
        assert [metric for metric, _c, _v in wrong] == ["energy"]
