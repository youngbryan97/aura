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
    """The collision is a naming fact, so it is checked on the source.

    14 November 2026: five renderers still printed "Energy" for arousal, the
    field and the homeostatic budget, beside the soma reserve the guard calls
    energy. Every module her prompt is built from is swept, not a list."""
    root = Path(__file__).resolve().parents[1]
    bare = re.compile(r"""(?:^|["( |])Energy: \{""")
    for module in sorted((*root.joinpath("core").rglob("*.py"), *root.joinpath("interface").rglob("*.py"))):
        text = module.read_text(encoding="utf-8")
        relative = str(module.relative_to(root))
        assert '_compact_snapshot_line("Energy"' not in text, relative
        assert not bare.search(text), relative


def test_a_bare_name_agrees_with_any_instrument_published_under_it(monkeypatch):
    """LIVE 13 November 2026, 20:06: shown "Current Mood: TIRED (substrate
    energy: 0.00 ...)", she said "My mood reads tired right now (energy at 0%)",
    and the reserve's 0.678 was set against it as a correction."""
    import core.self.capability_ledger as cl

    monkeypatch.setattr(cl, "measured_self_metrics", lambda: {"energy": 0.678, "substrate_energy": 0.003})
    said = "My mood reads tired right now (energy at 0%, curiosity at its ceiling)."

    assert cl.contradicted_self_readings(said) == []
    assert cl.with_her_readings_in_place(said) == (said, [])


def test_a_figure_no_instrument_supports_is_corrected_where_it_stands(monkeypatch):
    """Bryan, 14 November 2026: "ideally just make the correction vs putting it in a note"."""
    import core.self.capability_ledger as cl

    monkeypatch.setattr(cl, "measured_self_metrics", lambda: {"memory_pressure": 0.717, "energy": 0.678})
    said = "Memory pressure is 37% and energy is 0.2 now, which feels about right."

    corrected, figures = cl.with_her_readings_in_place(said)

    assert corrected == "Memory pressure is 72% and energy is 0.7 now, which feels about right."
    assert sorted(metric for metric, _claimed, _value in figures) == ["energy", "memory_pressure"]


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
        # A qualified name is held to its own organ.
        wrong = cl.contradicted_self_readings("Substrate energy is 0.647.")
        assert [metric for metric, _c, _v in wrong] == ["substrate_energy"]
        # The bare word may mean either (13 November 2026, 20:06: her mood's
        # "substrate energy: 0.00" said back as "energy at 0%"), so it is wrong
        # only when neither reads it.
        assert cl.contradicted_self_readings("My energy is 0.14.") == []
        wrong = cl.contradicted_self_readings("My energy is 0.30.")
        assert [metric for metric, _c, _v in wrong] == ["energy"]


def test_her_drives_are_readings_a_reply_is_held_to(monkeypatch):
    """LIVE 13 November 2026: "The system says curiosity is at 2%, frustration at 9%", read back from an answer an hour
    old, a minute after she had reported curiosity at 12%."""
    from types import SimpleNamespace
    from unittest import mock

    import core.self.capability_ledger as cl

    class _Container:
        @staticmethod
        def peek(name, default=None):
            if name == "liquid_substrate":
                return SimpleNamespace(current=SimpleNamespace(energy=0.14, focus=0.5, curiosity=0.12, frustration=0.09))
            return default

    with mock.patch.dict("sys.modules", {"core.container": SimpleNamespace(ServiceContainer=_Container)}):
        reading = cl._substrate_reading()
    assert reading["curiosity"] == 0.12 and reading["frustration"] == 0.09

    monkeypatch.setattr(cl, "measured_self_metrics", lambda: reading)
    said = "The system says curiosity is at 2%, frustration at 9%, so I guess I'm mildly annoyed."
    assert [metric for metric, _claimed, _value in cl.contradicted_self_readings(said)] == ["curiosity"]


@pytest.mark.asyncio
async def test_a_served_stale_drive_figure_is_corrected_where_it_stands(monkeypatch):
    """LIVE 13 November 2026, 15:14: "Curiosity is at 12%" (reading 0.218) was served with a sentence naming neither;
    by evening it had become a bracketed note in her voice. Bryan, 14 November: "just make the correction"."""
    import core.self.capability_ledger as cl
    from interface.routes import chat

    class _Ledger:
        @staticmethod
        def contradicted_claims(_text):
            return []

    async def _asked_again(*_args, **_kwargs):
        raise AssertionError("a figure the instrument settles needed a second generation")

    monkeypatch.setattr(cl, "get_capability_ledger", lambda: _Ledger())
    monkeypatch.setattr(cl, "measured_self_metrics", lambda: {"curiosity": 0.218, "frustration": 0.09})
    monkeypatch.setattr(chat, "_run_cognitive_engine_chat_turn", _asked_again)
    said = "But right now, the drives are flat. Curiosity is at 12%, and everything else is sitting still."

    served = await chat._reanswer_when_the_runtime_contradicts_her(said, user_message="How are your drives?")

    assert served == "But right now, the drives are flat. Curiosity is at 22%, and everything else is sitting still."


def test_nothing_is_written_into_her_reply_in_her_voice():
    source = _read("interface/routes/chat_desktop_mode.py")
    assert "Correcting myself" not in source
    assert "I need to correct that" not in source
