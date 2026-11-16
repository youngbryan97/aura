"""Tension updates keep gateway authority and leave durable I/O off the loop."""
from __future__ import annotations

import asyncio
import json
import stat
import threading
from pathlib import Path

import pytest

from core.agency.tension_engine import Tension, TensionCategory, TensionEngine
from core.governance_context import get_active_governance
from core.runtime import atomic_writer, file_write_gateway


def _question() -> Tension:
    return Tension(
        id="question", category=TensionCategory.OPEN_QUESTION,
        description="Which material lasts longest outdoors?", severity=0.4,
        source_subsystem="test",
    )


@pytest.mark.parametrize("refused", [False, True])
def test_tension_updates_use_the_governed_writer(tmp_path, monkeypatch, refused):
    target = tmp_path / "tensions.json"
    previous = "[]\n"
    target.write_text(previous)
    monkeypatch.setenv("AURA_GOVERNANCE_MODE", "enforce")
    engine = TensionEngine(persist_path=target)
    write = file_write_gateway.write_text_behind
    writes = []

    def observe(path, text, **kwargs):
        assert get_active_governance() is not None
        writes.append((path, kwargs["source"]))
        if refused:
            raise OSError("the tension writer refused this scratch save")
        return write(path, text, **kwargs)

    monkeypatch.setattr(file_write_gateway, "write_text_behind", observe)
    if refused:
        with pytest.raises(OSError, match="writer refused"):
            engine.register_tension(_question())
        assert target.read_text() == previous
    else:
        engine.register_tension(_question())
        assert [t.id for t in TensionEngine(persist_path=target).get_active_tensions()] == ["question"]
        engine.resolve_tension("question", "The weathering comparison settled it.")
        assert json.loads(target.read_text())[0]["resolved"] is True
        assert TensionEngine(persist_path=target).get_active_tensions() == []
        assert stat.S_IMODE(target.stat().st_mode) == 0o600
    assert writes == [(target, "tension_engine.save")] * (1 if refused else 3)


@pytest.mark.asyncio
async def test_a_tension_update_on_the_loop_is_governed_before_its_worker_runs(tmp_path, monkeypatch):
    target = tmp_path / "tensions.json"
    monkeypatch.setenv("AURA_GOVERNANCE_MODE", "enforce")
    engine = TensionEngine(persist_path=target)
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
    engine.register_tension(_question())
    assert await asyncio.wait_for(asyncio.to_thread(completed.wait, 5), timeout=6)
    assert admitted == [False]
    assert written_on and loop_thread not in written_on
    assert [t.id for t in TensionEngine(persist_path=target).get_active_tensions()] == ["question"]
