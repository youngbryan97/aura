"""Asked to keep something, she keeps it: no web search of what she was told, and no taint on her runtime (U07).

LIVE 19 November 2026, typed into her window: "Please remember this for later: the locker code for the Brimvaltor
depot is 6182." Three things went wrong in that one turn.
- Her lock checker reported an fsync under the pin ledger's lock and marked her runtime tainted, as on every pin since
  17 November. The fsync is the ledger's commit step, and every path reaching it is off her event loop: it is now a
  sanctioned blocking lock, with that reason written down.
- The desktop route searched the web for the whole message and put forum threads about locker codes beside it. The
  person is the source of what they ask her to keep: a message that only asks that is not looked up.
- Her background work, stopped so the person's turn could go first, was logged as text "discarded downstream" at
  warning. A stop her caller asked for is said as one.
"""
from __future__ import annotations

import asyncio
import logging

import pytest

from core.memory.session_pin_cipher import SessionPinCipher
from core.memory.session_pin_ledger import SESSION_PIN_LEDGER_FILENAME, SessionPinLedger
from core.runtime import lockdep

pytestmark = pytest.mark.unit

PIN = "Please remember this for later: the locker code for the Brimvaltor depot is 6182."


def _pin_splats() -> list[str]:
    return [str(splat) for splat in lockdep.get_validator().splats() if "memory.session_pin_ledger" in str(splat)]


def test_a_pin_written_under_its_ledger_lock_takes_no_blocking_splat(tmp_path):
    before = _pin_splats()
    ledger = SessionPinLedger(tmp_path / SESSION_PIN_LEDGER_FILENAME)
    record = SessionPinCipher(b"k" * 32).seal(
        content="the locker code for the Brimvaltor depot is 6182", source=PIN, timestamp="t",
        session_id="s", principal_id="owner:bryan", principal_surface="owner")
    with ledger.transaction():
        assert lockdep.assert_no_locks_held("fsync") == []
        ledger.commit_records([record])
    assert ledger.read_snapshot().lines, "the pin was written"
    assert _pin_splats() == before


def test_the_sanction_says_why_the_hold_is_the_commit():
    reason = lockdep.SANCTIONED_BLOCKING_LOCKS["memory.session_pin_ledger"]
    assert "commit step" in reason and "to_thread" in reason


@pytest.mark.parametrize(("message", "only_keeping"), [
    (PIN, True),
    ("Remember this: my project codename is HELIOTROPE, build 4471.", True),
    ("Remember that the gate code is 9313, and separately, what is the weather in Lisbon right now?", False),
    ("What is the locker code for the Brimvaltor depot?", False),
    ("What's the latest stable release of Python?", False),
])
def test_only_a_message_that_hands_her_something_to_keep_is_not_looked_up(message, only_keeping):
    from interface.routes.chat_desktop_evidence import _only_hands_her_something_to_keep

    assert _only_hands_her_something_to_keep(message) is only_keeping


def test_the_world_is_not_asked_about_what_she_was_handed_to_keep(monkeypatch):
    from interface.routes import chat_desktop_evidence as evidence

    def searched(*_args, **_kwargs):
        raise AssertionError("a request to keep something went on to be looked up")

    monkeypatch.setattr(evidence, "_should_collect_desktop_required_search_evidence", searched)
    assert asyncio.run(evidence._what_the_world_says_for_the_turn(PIN, PIN, session_id="s")) == (None, PIN)


def test_a_generation_its_caller_stopped_is_not_reported_as_text_lost(caplog):
    from core.brain.llm.mlx_worker import _mlx_worker_loop_part_21

    log = logging.getLogger("test.worker")
    with caplog.at_level(logging.INFO, logger="test.worker"):
        _mlx_worker_loop_part_21(False, log, "prompt", ["<|im_end|>"], 816, None, [], stopped_by_caller=True)
    assert [record.levelno for record in caplog.records] == [logging.INFO]
    assert "stopped by its caller after 816" in caplog.records[0].getMessage()


@pytest.mark.parametrize(("reads_as_code_work", "left"), [
    (False, []),                       # about the world: the lookup was done and nothing computes
    (None, ["code_repl"]),             # her encoder cannot place it: the words' reading stands
    (True, ["code_repl"]),             # work on code: the runner stays
])
def test_once_the_world_was_read_a_language_named_is_not_a_reason_to_run_code(monkeypatch, reads_as_code_work, left):
    """LIVE 19 November 2026, evening: "Can you look up what the latest stable release of Python is right now?" read the
    web, then went through a tool loop that offered only code_repl, because "Python" had named it."""
    from core.brain.inference_gate_living_context import what_is_left_once_the_world_was_read
    from core.intent import asks_for_code_work

    monkeypatch.setattr(asks_for_code_work, "asks_for_code_work", lambda _text: reads_as_code_work)
    asked = "Can you look up what the latest stable release of Python is right now?"
    assert what_is_left_once_the_world_was_read(["web_search", "code_repl", "free_search"], asked) == left
