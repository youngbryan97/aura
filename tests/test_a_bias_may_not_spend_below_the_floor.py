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

def _code_after(anchor: str, *, lines: int) -> str:
    """The next ``lines`` lines of CODE after an anchor comment.

    These two contracts were checked inside a fixed count of CHARACTERS after
    the anchor, which makes a comment added above the code the same event as
    the code being deleted. Both went red for that reason and neither contract
    had moved. Counting code lines says what was meant: the requirement is
    applied here, near this, not somewhere else in a fourteen-thousand-line
    file.
    """

    body = _GATE.read_text().splitlines()
    start = next(
        number for number, line in enumerate(body) if anchor in line
    )
    kept: list[str] = []
    for line in body[start + 1 :]:
        bare = line.strip()
        if not bare or bare.startswith("#"):
            continue
        kept.append(line)
        if len(kept) >= lines:
            break
    return "\n".join(kept)



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


def test_the_answer_floor_gets_the_last_word_at_dispatch() -> None:
    """Every earlier raise can be, and was, overwritten by a later cap.

    LIVE, 2026-08-27: a question whose answer had to be worked out carried a
    completion floor of 896 tokens all the way to the worker, which read it and
    opened the private reasoning channel. The same turn was dispatched with
    128, so the generation ended still inside that channel with no answer.
    """

    body = _GATE.read_text()
    answer = _line_of("[ANSWER BUDGET] Answer turn:")
    lane = _line_of("serving_lane = self._cortex_serving_lane(")
    bias = _line_of("somatic_temperature, max_tokens, applied_bias = self._apply_runtime_sampling_biases(")
    assert bias < answer, "the last word comes after every earlier budget step"
    assert answer < lane, "and before the serving profile clamps it"


def test_the_last_word_yields_to_a_declared_requirement() -> None:
    """Three declared requirements still win over the last word on the budget.

    The requirement is that each one is consulted here. How it is consulted is
    the gate's business: three separate reads and one loop over three names
    are the same contract, and the earlier form of this test could only see
    the first.
    """

    window = _code_after("FINAL word on the budget for an answer turn", lines=30)
    for guard in (
        "hard_output_token_ceiling",
        "resource_stakes_blocked",
        "desktop_execution_contract",
    ):
        assert guard in window, guard
    assert "context.get(" in window, "the requirements are read from the turn context"


def test_the_execution_floor_still_owns_execution_turns() -> None:
    body = _GATE.read_text()
    assert "FINAL word on the budget for an execution turn" in body
    assert "_plan_floor_final" in body


def test_the_floor_in_context_is_the_entitlement() -> None:
    """Re-deriving the qualification gives two conditions room to drift.

    LIVE, 2026-08-27: the turn that carried floor=896 all the way to the worker
    did not satisfy a second copy of the entitlement test at dispatch, and was
    sent with 375 tokens.
    """

    window = _code_after("The presence of a floor is the entitlement", lines=40)
    assert "_foreground_answer_turn" not in window
    assert 'context.get("user_surface_completion_floor")' in window
