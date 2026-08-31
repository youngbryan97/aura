"""A bound proportional to one step's budget cut a turn that was still working.

LIVE 2026-08-29: "Endpoint gave up 31s past its budget: last sign of work 0.2s
ago, quiet window 20s". Nothing had gone quiet. Six clocks above this one had
already been taught to wait while tokens arrive; this one gave a step 31
seconds, three times that as overrun, and cancelled it two tenths of a second
after its last sign of work. The person got a list of files they had not asked
for instead of the answer.

The multiplier is right for what it was written for. "user_facing" is true for
anything on the foreground lane, a two-second health probe included, and that
probe once waited eight minutes here and left the runtime in CRITICAL.
Whether somebody is actually sitting in front of the turn is the fact that
separates the two, and the caller has it.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

SOURCE = Path("core/brain/llm_health_router.py")


def _the_waiting_function() -> ast.AsyncFunctionDef:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.AsyncFunctionDef)
            and node.name == "_await_while_it_is_working"
        ):
            return node
    raise AssertionError("the waiting function is gone")


def test_the_caller_can_say_a_person_is_waiting() -> None:
    names = {a.arg for a in _the_waiting_function().args.kwonlyargs}
    assert "person_is_waiting" in names
    assert "user_facing" in names, "the probe distinction is still separate"


def test_nobody_waiting_keeps_the_proportional_bound() -> None:
    """A probe stays a probe: it must not wait eight minutes for a yes or no."""

    source = ast.unparse(_the_waiting_function())
    assert "person_is_waiting" in source
    assert "float(budget_s) * 3.0" in source, "the probe's bound was removed"


def test_the_tool_loop_says_so() -> None:
    """It already puts the same fact on the context it hands the conscience."""

    gate = Path("core/brain/inference_gate.py").read_text(encoding="utf-8")
    assert "person_is_waiting=True," in gate
    assert '"a_person_is_waiting": True,' in gate


def test_unowned_waits_keep_their_existing_ceiling() -> None:
    """Owned foreground renewal is tested separately with executing tasks."""

    source = ast.unparse(_the_waiting_function())
    assert "USER_FACING_COMPLETION_DEADLINE_MAX_S" in source
    assert "ceiling if person_is_waiting else min(ceiling" in source.replace("\n", " ")


class TestTheCeilingIsMeasuredNotFlat:
    """A flat number cannot tell a turn running away from a turn working.

    LIVE 2026-08-29, after the proportional cut above was fixed: "gave up 298s
    past its budget: last sign of work 0.3s ago", at exactly 480 seconds, while
    the turn was composing its answer. On this host one generation of a
    thousand tokens against an eight-thousand-character prompt takes about a
    hundred seconds and a tool loop is allowed several, so the bound stood
    below the cost of the work it was standing over.
    """

    def test_the_measured_ceiling_only_ever_raises_the_flat_one(self) -> None:
        from core.brain.llm.mlx_client import longest_a_turn_may_take

        assert (
            longest_a_turn_may_take(
                generations=1, prompt_chars=10, max_tokens=8, floor_s=480.0
            )
            == 480.0
        ), "a small measurement must not shorten somebody else's bound"

    def test_a_turn_that_costs_more_than_the_flat_bound_gets_the_room(self) -> None:
        from core.brain.llm.mlx_client import longest_a_turn_may_take

        assert (
            longest_a_turn_may_take(
                generations=4, prompt_chars=8324, max_tokens=2048, floor_s=480.0
            )
            > 480.0
        )

    def test_a_faster_machine_gets_a_shorter_ceiling(self) -> None:
        from core.brain.llm import mlx_client

        rates = dict(mlx_client._HOST_RATES)
        try:
            mlx_client._HOST_RATES.update({"prefill": 300.0, "decode": 8.0})
            slow = mlx_client.longest_a_turn_may_take(
                generations=4, prompt_chars=8324, max_tokens=2048, floor_s=0.0
            )
            mlx_client._HOST_RATES.update({"prefill": 900.0, "decode": 40.0})
            fast = mlx_client.longest_a_turn_may_take(
                generations=4, prompt_chars=8324, max_tokens=2048, floor_s=0.0
            )
        finally:
            mlx_client._HOST_RATES.clear()
            mlx_client._HOST_RATES.update(rates)
        assert fast < slow

    def test_only_a_person_waiting_gets_it(self) -> None:
        """A probe keeps the flat bound; it is asking for a fast yes or no."""

        source = ast.unparse(_the_waiting_function())
        assert "if person_is_waiting:" in source
        assert "longest_a_turn_may_take" in source

    def test_silence_still_ends_it_whatever_the_ceiling(self) -> None:
        """The ceiling only binds something producing continuously."""

        source = ast.unparse(_the_waiting_function())
        assert "still_producing" in source
        assert "break" in source


class TestTheTurnsOwnCeilingIsMeasuredToo:
    """The cycle holds itself open while working; this is where it stops.

    LIVE 2026-08-29: a turn dispatched code_repl seven minutes in and was cut
    at exactly 480 seconds with the sandbox still running —
    "reactive_recovery:timeout". The endpoint wait had been taught to size its
    ceiling from what this machine measures; the turn's own bound had not, so
    the two disagreed about how long a turn of this shape costs here.
    """

    def test_a_waiting_person_gets_the_measured_ceiling(self) -> None:
        from core.brain.cognitive_engine import _the_longest_this_turn_may_take

        assert _the_longest_this_turn_may_take(480.0, user_facing=True) > 480.0

    def test_background_keeps_the_floor(self) -> None:
        from core.brain.cognitive_engine import _the_longest_this_turn_may_take

        assert _the_longest_this_turn_may_take(480.0, user_facing=False) == 480.0

    def test_it_only_ever_raises_the_callers_floor(self) -> None:
        from core.brain.cognitive_engine import _the_longest_this_turn_may_take

        assert _the_longest_this_turn_may_take(9000.0, user_facing=True) >= 9000.0

    def test_both_clocks_size_it_the_same_way(self) -> None:
        """The endpoint wait and the turn must not disagree about the cost."""

        from core.brain.cognitive_engine import _the_longest_this_turn_may_take
        from core.brain.llm.mlx_client import longest_a_turn_may_take
        from core.brain.llm_health_router import (
            _A_TURNS_ANSWER_TOKENS,
            _A_TURNS_PROMPT_CHARS,
            _GENERATIONS_A_TOOL_TURN_MAY_TAKE,
        )

        assert _the_longest_this_turn_may_take(480.0, user_facing=True) == (
            longest_a_turn_may_take(
                generations=_GENERATIONS_A_TOOL_TURN_MAY_TAKE,
                prompt_chars=_A_TURNS_PROMPT_CHARS,
                max_tokens=_A_TURNS_ANSWER_TOKENS,
                floor_s=480.0,
            )
        )
