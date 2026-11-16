"""Empty agency state stays in memory until a governed save creates it."""
from __future__ import annotations

import asyncio
import stat
import threading
from pathlib import Path

import pytest

from core.agency.subjective_choice import SubjectiveChoiceEngine
from core.agency.tension_engine import Tension, TensionCategory, TensionEngine
from core.governance_context import get_active_governance
from core.runtime import atomic_writer, file_write_gateway


def _engine(kind, target):
    if kind == "tension":
        return TensionEngine(persist_path=target)
    return SubjectiveChoiceEngine(state_path=target, mirror_identity=False)


def _update(kind, engine):
    if kind == "tension":
        engine.register_tension(Tension(
            id="question", category=TensionCategory.OPEN_QUESTION,
            description="Which material lasts longest outdoors?", severity=0.4,
        ))
    else:
        engine.set_item_preference(domain="materials", item_id="cedar", label="Cedar", strength=0.84)


def _assert_reloaded(kind, target):
    restored = _engine(kind, target)
    if kind == "tension":
        assert [t.id for t in restored.get_active_tensions()] == ["question"]
    else:
        preference = restored.recall_item_preference(domain="materials", item_id="cedar")
        assert preference is not None and preference.label == "Cedar" and preference.strength == 0.84


@pytest.mark.parametrize("kind", ["tension", "choice"])
def test_an_empty_store_creates_no_directory_until_it_is_saved(tmp_path, monkeypatch, kind):
    monkeypatch.setenv("AURA_GOVERNANCE_MODE", "enforce")
    target = tmp_path / "private" / "state.json"
    engine = _engine(kind, target)
    assert not target.parent.exists()
    _update(kind, engine)
    _assert_reloaded(kind, target)
    assert stat.S_IMODE(target.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(target.stat().st_mode) == 0o600


@pytest.mark.parametrize("refused", [False, True])
def test_a_choice_save_uses_the_governed_writer_and_preserves_a_refused_file(tmp_path, monkeypatch, refused):
    monkeypatch.setenv("AURA_GOVERNANCE_MODE", "enforce")
    target = tmp_path / "choices.json"
    target.write_text("{}\n")
    previous = target.read_bytes()
    engine = _engine("choice", target)
    write = file_write_gateway.write_text_behind
    writes = []

    def observe(path, text, **kwargs):
        assert get_active_governance() is not None
        writes.append((path, kwargs["source"]))
        if refused:
            raise OSError("the choice writer refused this scratch save")
        return write(path, text, **kwargs)

    monkeypatch.setattr(file_write_gateway, "write_text_behind", observe)
    _update("choice", engine)
    if refused:
        assert target.read_bytes() == previous
        assert _engine("choice", target).recall_item_preference(domain="materials", item_id="cedar") is None
    else:
        _assert_reloaded("choice", target)
    assert writes == [(target, "subjective_choice.save")]


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["tension", "choice"])
async def test_a_first_save_creates_private_state_off_the_loop(tmp_path, monkeypatch, kind):
    monkeypatch.setenv("AURA_GOVERNANCE_MODE", "enforce")
    target = tmp_path / "private" / "state.json"
    engine = _engine(kind, target)
    assert not target.parent.exists()
    loop_thread = threading.get_ident()
    write = file_write_gateway.write_text_behind
    atomic = atomic_writer.atomic_write_bytes
    completed = threading.Event()
    admitted, written_on = [], []

    def observe(path, text, **kwargs):
        assert threading.get_ident() == loop_thread
        assert get_active_governance() is not None
        result = write(path, text, **kwargs)
        admitted.append(result)
        return result

    def materialize(path, payload, **kwargs):
        if Path(path) != target:
            return atomic(path, payload, **kwargs)
        written_on.append(threading.get_ident())
        result = atomic(path, payload, **kwargs)
        completed.set()
        return result

    monkeypatch.setattr(file_write_gateway, "write_text_behind", observe)
    monkeypatch.setattr(atomic_writer, "atomic_write_bytes", materialize)
    _update(kind, engine)
    assert await asyncio.wait_for(asyncio.to_thread(completed.wait, 5), timeout=6)
    assert admitted == [False] and written_on and loop_thread not in written_on
    assert stat.S_IMODE(target.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    _assert_reloaded(kind, target)
