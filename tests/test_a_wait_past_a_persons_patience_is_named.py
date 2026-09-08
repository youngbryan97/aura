"""How long the work takes and how long a person waits are two numbers.

The endpoint's first-token ceiling is sized from the prompt and the token
budget: it says how long the WORK should take. The foreground wait is a
different quantity — how long somebody sitting in front of the desktop will
wait — and the runtime had one number doing both jobs.

LIVE, 2026-09-08: a desktop turn with a 49,136-character prompt got a
900-second first-token ceiling. The 27B took the job on a host at 11.8GB free
with a 9B already resident, spent fifteen minutes at half a core paging
weights, and produced no first token. `is_inference_ready()` reported False
for 631 seconds while it happened. Nothing was broken. The person was not
being served, and nothing anywhere said so.

The wait still continues — the endpoint owns first-token, livelock, heartbeat,
memory-pressure and cancellation, and this outer estimate cannot see native MLX
work while the loop is delayed. What changes is that passing a person's
patience is now a line in the log with the number it passed.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROUTER = ROOT / "core/brain/llm_health_router.py"


def _wait_source() -> str:
    from core.brain import llm_health_router

    return inspect.getsource(llm_health_router._await_while_it_is_working)


def test_the_owned_wait_still_waits() -> None:
    """The endpoint owns the terminal state; this must not start cancelling."""

    body = _wait_source()
    assert body.count("return await task") >= 1
    assert "task.cancel()" not in body.split("owned_foreground")[1][:2000]


def test_passing_a_persons_patience_is_said_out_loud() -> None:
    body = _wait_source()
    assert "longest_a_turn_may_take" in body
    assert "A person has been waiting" in body
    assert "logger.warning" in body


def test_the_watcher_is_cancelled_with_the_wait() -> None:
    """A timer left running after the turn is a leak."""

    body = _wait_source()
    assert "watcher.cancel()" in body
    assert "finally:" in body


def test_an_unreadable_patience_changes_nothing() -> None:
    """A number that will not load is not a reason to behave differently."""

    body = _wait_source()
    assert "a_person_waits = 0.0" in body
    assert "if a_person_waits > 0.0" in body


def test_the_file_still_parses() -> None:
    ast.parse(ROUTER.read_text("utf-8"))
