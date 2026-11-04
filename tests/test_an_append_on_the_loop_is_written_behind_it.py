"""An append-only log reached from a coroutine is appended off the loop, every line in order.

On her day runs of 1 and 3 November lockdep named five synchronous writes made on the event loop thread from a
coroutine: her growth ladder's save (from her reply's finalize), the campaign's record of why it did not run (from the
autonomy conductor), her reasoning memory (from the amplifier), and the OOM shed log (from the metabolic coordinator).
Each one made every task wait on the disk. Whole-file saves already had a writer behind the loop; appends did not.
"""
from __future__ import annotations

import asyncio
import inspect
import threading

import pytest

from core.runtime import atomic_writer


def test_off_the_loop_it_appends_before_returning(tmp_path):
    target = tmp_path / "log.jsonl"
    assert atomic_writer.atomic_append_text_behind(target, "one\n")
    assert target.read_text() == "one\n"


def test_on_the_loop_every_line_lands_off_it_in_order(tmp_path):
    target = tmp_path / "log.jsonl"
    writers: list[str] = []
    real = atomic_writer.atomic_append_text

    def watched(path, text, **kwargs):
        writers.append(threading.current_thread().name)
        return real(path, text, **kwargs)

    async def burst() -> tuple[list[bool], str]:
        loop_thread = threading.current_thread().name
        atomic_writer.atomic_append_text = watched
        try:
            now = [atomic_writer.atomic_append_text_behind(target, f"{n}\n") for n in range(50)]
            for _ in range(300):
                if target.exists() and target.read_text().count("\n") == 50:
                    break
                await asyncio.sleep(0.01)
        finally:
            atomic_writer.atomic_append_text = real
        return now, loop_thread

    written_now, loop_thread = asyncio.run(burst())
    assert not any(written_now)
    assert loop_thread not in writers
    assert target.read_text() == "".join(f"{n}\n" for n in range(50)), "an append behind keeps every line, in order"
    assert len(writers) < 50, "a burst of appends goes down in few writes"


def test_what_is_held_at_shutdown_is_appended(tmp_path):
    target = tmp_path / "log.jsonl"
    target.write_text("before\n")
    with atomic_writer._BEHIND_LOCK:
        atomic_writer._APPEND_BEHIND[str(target)] = [("a\n", "utf-8"), ("b\n", "utf-8")]
    assert atomic_writer.flush_writes_behind() == 1
    assert target.read_text() == "before\na\nb\n"


def test_the_gateway_checks_governance_before_it_defers(tmp_path, monkeypatch):
    from core.runtime import file_write_gateway as gateway_module

    asked: list[str] = []
    monkeypatch.setattr(gateway_module, "governance_runtime_active", lambda: True)
    monkeypatch.setattr(gateway_module, "require_governance", lambda what, **_: asked.append(what))
    gateway = gateway_module.get_file_write_gateway()
    gateway.append_text_behind(tmp_path / "a.jsonl", "x\n", source="test_source")
    gateway.write_text_behind(tmp_path / "b.json", "{}", source="test_source")
    assert asked == ["file_write_gateway.append_text:test_source", "file_write_gateway.write_text:test_source"]
    assert (tmp_path / "a.jsonl").read_text() == "x\n"
    assert (tmp_path / "b.json").read_text() == "{}"


@pytest.mark.parametrize(
    ("module", "function", "writer"),
    [
        ("core.self_modification.growth_ladder", "GrowthLadder._save", "atomic_write_text_behind("),
        ("core.verify.why_the_campaign_did_not_run", "_write", "write_text_behind("),
        ("core.brain.reasoning_memory", "ReasoningMemory.record", "atomic_append_text_behind("),
        ("core.runtime.oom_policy", "_append_shed_log", "append_text_behind("),
    ],
)
def test_the_writes_named_on_the_day_runs_go_behind_the_loop(module, function, writer):
    import importlib

    target = importlib.import_module(module)
    for part in function.split("."):
        target = getattr(target, part)
    assert writer in inspect.getsource(target)
