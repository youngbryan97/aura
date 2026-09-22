"""The pieces that let a campaign run her whole self, each against what it promises.

A conversation tape replays what the person she talks to actually said, in
order, and records only its digest and length. The steady mind speaks through
the live router with a greedy decode and answers a repeated call as it did the
first time, so two arms of a paired trial get the same answer. The whole
environment finds the desktop's `.env` from a worktree, where there is none of
its own.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from core.subject import steady_mind
from core.subject.conversation_tape import ConversationTape
from core.subject.steady_mind import SteadyMind
from tools.whole_environment import _read_env_file, env_file_for

pytestmark = pytest.mark.unit


def test_a_tape_replays_in_order_and_wraps() -> None:
    tape = ConversationTape(turns=("morning", "how was it", "thanks"), sessions=(0, 0, 1))
    assert [tape.at(i) for i in range(5)] == ["morning", "how was it", "thanks", "morning", "how was it"]


def test_what_a_run_records_about_a_tape_holds_no_words() -> None:
    tape = ConversationTape(turns=("a private thing", "another"), sessions=(0, 1))
    record = tape.provenance()
    assert record["turns"] == 2 and record["sessions"] == 2
    assert "a private thing" not in repr(record)
    assert record["digest"] == ConversationTape(turns=("a private thing", "another")).digest


def test_an_empty_tape_is_refused() -> None:
    with pytest.raises(ValueError):
        ConversationTape(turns=())


def test_a_tape_survives_being_written_and_read(tmp_path: Path) -> None:
    import json

    tape = ConversationTape(turns=("one", "two"), sessions=(0, 0), notes={"principal": "bryan"})
    path = tmp_path / "tape.json"
    path.write_text(json.dumps(tape.as_dict()), "utf-8")
    assert ConversationTape.load(path) == tape


class _Router:
    def __init__(self, answers=None) -> None:
        self.calls: list[dict] = []
        self.answers = list(answers or [])

    async def think(self, prompt=None, **kwargs):
        self.calls.append({"prompt": prompt, **kwargs})
        return self.answers.pop(0) if self.answers else f"answer to {prompt}"

    def get_stats(self) -> dict:
        return {"total_calls": len(self.calls)}


@pytest.fixture(autouse=True)
def _forget():
    steady_mind.forget_for_test()
    yield
    steady_mind.forget_for_test()


def test_the_same_call_gets_the_same_answer_without_asking_twice() -> None:
    router = _Router()
    mind = SteadyMind(router)
    first = asyncio.run(mind.think("what now?", prefer_tier="primary"))
    second = asyncio.run(mind.think("what now?", prefer_tier="primary"))
    assert first == second == "answer to what now?"
    assert len(router.calls) == 1


def test_the_decode_is_greedy() -> None:
    router = _Router()
    asyncio.run(SteadyMind(router).think("what now?", temperature=0.9))
    assert router.calls[0]["temperature"] == 0.0


def test_a_different_call_is_asked() -> None:
    router = _Router()
    mind = SteadyMind(router)
    asyncio.run(mind.think("what now?"))
    asyncio.run(mind.think("what now?", prefer_tier="tertiary"))
    assert len(router.calls) == 2


def test_an_empty_answer_is_not_kept() -> None:
    router = _Router(answers=["", "ready"])
    mind = SteadyMind(router)
    assert asyncio.run(mind.think("ready?")) == ""
    assert asyncio.run(mind.think("ready?")) == "ready"
    assert steady_mind.kept() == 1


def test_the_wait_on_the_organ_is_counted() -> None:
    class _Slow(_Router):
        async def think(self, prompt=None, **kwargs):
            await asyncio.sleep(0.05)
            return "slow"

    asyncio.run(SteadyMind(_Slow()).think("x"))
    assert steady_mind.waited() >= 0.05


def test_the_body_reads_the_organs_own_stats() -> None:
    router = _Router()
    mind = SteadyMind(router)
    asyncio.run(mind.think("x"))
    assert mind.get_stats() == {"total_calls": 1}


def test_a_worktree_reads_the_primary_checkouts_env(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("AURA_ENV_FILE", raising=False)
    primary = tmp_path / "primary"
    (primary / ".git" / "worktrees" / "wt").mkdir(parents=True)
    (primary / ".env").write_text("AURA_BRAINSTEM_MODEL=Ternary\n", "utf-8")
    worktree = tmp_path / "wt"
    worktree.mkdir()
    (worktree / ".git").write_text(f"gitdir: {primary / '.git' / 'worktrees' / 'wt'}\n", "utf-8")
    assert env_file_for(worktree) == primary / ".env"


def test_the_env_file_is_read_as_the_desktop_reads_it(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.write_text('# a comment\nexport A="one"\nB=two\n\nC=\n', "utf-8")
    assert _read_env_file(path) == {"A": "one", "B": "two", "C": ""}
