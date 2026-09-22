"""The conversation condition hears the tape, and a fork hears the same thing again.

The n-th conversation turn of a run hears the n-th thing on the tape. The
count is carried by the snapshot, so an arm restored from before a turn hears
the message that turn heard, and the difference between two arms stays the
intervention rather than which line of the conversation each one reached.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from core.subject.conversation_tape import ConversationTape

pytestmark = pytest.mark.slow


def _said(runtime) -> list[str]:
    return [
        str(item.get("content", ""))
        for item in runtime.state.cognition.working_memory
        if isinstance(item, dict) and item.get("role") == "user"
    ]


def test_a_fork_hears_what_the_turn_it_forked_from_heard(monkeypatch, tmp_path: Path) -> None:
    from core.config import config
    from core.subject.clock import installed_clock
    from core.subject.driver import CONDITIONS, build_runtime, quiesce_organism, start_organism

    state_root = tmp_path / "state"
    state_root.mkdir()
    monkeypatch.setenv("AURA_STATE_ROOT", str(state_root))
    monkeypatch.setattr(config.paths, "home_dir_override", None)
    conversation = next(condition for condition in CONDITIONS if condition.name == "conversation")

    async def run():
        runtime = build_runtime(tmp_path / "runtime", seed=3)
        await start_organism(runtime)
        await quiesce_organism(runtime)
        runtime.conversation_tape = ConversationTape(
            turns=("the first thing he said", "the second thing he said")
        )
        runtime.spoken = 0
        before = runtime.snapshot()
        await runtime.turn_once(conversation)
        first_arm = _said(runtime)
        runtime.restore(before)
        await runtime.turn_once(conversation)
        second_arm = _said(runtime)
        await runtime.turn_once(conversation)
        after = _said(runtime)
        return first_arm, second_arm, after

    try:
        first_arm, second_arm, after = asyncio.run(run())
    finally:
        clock = installed_clock()
        if clock is not None:
            clock.uninstall()

    assert first_arm[-1] == "the first thing he said"
    assert second_arm[-1] == "the first thing he said", "the restored arm heard a different line"
    assert after[-1] == "the second thing he said"
