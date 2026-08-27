"""A sampling bias may make an answer terser. It may not make it incomplete.

The completion floor is computed from the visible request and applied to the
decode budget. A runtime sampling multiplier then ran after it, in the same
function, and took the budget back under the floor.

LIVE, 2026-08-27: a question that had to be worked out carried a floor of 896
tokens. An integration measure scaled by its smallest permitted factor and the
model was dispatched with 363, stopping one sentence before the answer.
"""

from __future__ import annotations

import re
from pathlib import Path

_GATE = Path("core/brain/inference_gate.py")


def _line_of(needle: str) -> int:
    for number, line in enumerate(_GATE.read_text().splitlines(), start=1):
        if needle in line:
            return number
    raise AssertionError(f"not found: {needle}")


def test_the_floor_is_restored_after_the_bias_runs() -> None:
    applied = _line_of('context["user_surface_completion_floor"] = surface_completion_floor')
    scaled = _line_of("somatic_temperature, max_tokens, applied_bias = self._apply_runtime_sampling_biases(")
    restored = _line_of("Completion floor restored after sampling bias")
    assert applied < scaled, "the floor is applied before the multiplier"
    assert scaled < restored, "the floor must be put back after the multiplier"


def test_the_restore_reads_the_floor_the_gate_already_stored() -> None:
    body = _GATE.read_text()
    window = body[body.index("Completion floor restored after sampling bias") - 900 :][:1200]
    assert 'context.get("user_surface_completion_floor")' in window
    assert "max_tokens = _floor" in window


def test_a_missing_or_unreadable_floor_changes_nothing() -> None:
    body = _GATE.read_text()
    start = body.index("# A bias may spend less of the budget")
    window = body[start : start + 1200]
    assert "except (TypeError, ValueError, OverflowError)" in window
    assert "if 0 < _floor and max_tokens < _floor:" in window


def test_the_multiplier_still_has_a_smallest_factor() -> None:
    # The bias keeps its own range; the floor is a second, independent bound.
    body = _GATE.read_text()
    assert re.search(r"0\.40 <= factor <= 1\.20", body)


def test_the_floor_covers_the_same_turns_the_multiplier_does() -> None:
    """A hard question routed to the deliberate lane still gets a floor.

    LIVE, 2026-08-27: the deliberate lane carried no desktop contract, so the
    floor did not apply and the model was dispatched with 128 tokens for a
    question the quick lane had given 896. Choosing the right lane made the
    budget worse.
    """

    body = _GATE.read_text()
    start = body.index("_foreground_answer_turn = (")
    guard = body[start : start + 400]
    for condition in (
        "not is_background",
        "self._origin_is_user_facing(origin)",
        "not isolated_generation_contract",
        "not health_probe",
        "not benchmark_request",
        "not proof_evaluation_contract",
        "not strict_answer_contract",
    ):
        assert condition in guard, condition
    assert "desktop_cognitive_engine_contract or _foreground_answer_turn" in body


def test_a_declared_ceiling_and_a_blocked_stake_still_win() -> None:
    body = _GATE.read_text()
    start = body.index("desktop_cognitive_engine_contract or _foreground_answer_turn")
    window = body[start : start + 300]
    assert 'context.get("hard_output_token_ceiling", False)' in window
    assert 'context.get("resource_stakes_blocked", False)' in window


def test_a_discarded_draft_is_written_down() -> None:
    """A gate that destroys an answer records what it destroyed."""

    client = Path("core/brain/llm/mlx_client.py").read_text()
    start = client.index("Worker rejected the visible draft for semantic ")
    window = client[start - 1400 : start + 700]
    assert "draft_chars=%d head=%r tail=%r" in window
    assert "_rejected_surface_draft(" in window
