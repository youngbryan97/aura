"""She described a game she had not built, one turn after failing to build it.

Live, 2026-07-27. The 2048 build failed and she said so plainly. The next turn
asked whether it was playable:

    "When you run it, the board pops up and you click cells to reveal numbers.
     If you hit a mine, the game shows you which squares had mines and ends."

That is Minesweeper. There was no file. Nothing about it was dishonest — the
receipt for the failed attempt lived in the intention ledger, the conversation
history held only her own sentence about having tried, and "how does the
artifact behave" has no answer anywhere in the transcript. So the model wrote
the most plausible paragraph about how a small game behaves.

The ledger already knew. IntentionLoop keeps a Say-Do-Observe record per
attempt — tools invoked, success flags, observed outcome — and it simply was
not in front of her when she was asked. Now it is, in both prompt builders,
with the one instruction that matters: an attempt that did not succeed produced
no artifact, so do not describe how it behaves.
"""
from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from core.brain.recent_actions import RECENT_ACTIONS_HEADER, recent_actions_block

ENGINE = Path("core/brain/cognitive_engine.py")
GATE = Path("core/brain/inference_gate.py")


def _record(*, intention: str, tool: str, success: bool, ago: float, outcome: str = ""):
    now = time.time()
    return SimpleNamespace(
        intention=intention,
        actions_taken=[
            SimpleNamespace(tool_name=tool, success=success, executed_at=now - ago)
        ],
        actual_outcome=outcome,
        observation=outcome,
        completed_at=now - ago,
    )


def _block(records, now: float | None = None) -> str:
    loop = SimpleNamespace(_completed_intentions=records)
    with patch("core.agency.intention_loop.get_intention_loop", return_value=loop):
        return recent_actions_block(now=now)


def test_a_failed_build_is_reported_as_failed() -> None:
    block = _block(
        [
            _record(
                intention="reverse-engineer 2048 onto the Desktop",
                tool="program_dna_reconstruct",
                success=False,
                ago=30,
                outcome="blocked by covenant; nothing written",
            )
        ]
    )
    assert RECENT_ACTIONS_HEADER in block
    assert "DID NOT SUCCEED" in block
    assert "program_dna_reconstruct" in block
    assert "nothing written" in block


def test_the_instruction_forbids_narrating_an_artifact_that_does_not_exist() -> None:
    block = _block([_record(intention="build it", tool="t", success=False, ago=10)])
    assert "produced no artifact" in block
    assert "do not describe how it behaves" in block


def test_a_successful_action_is_reported_as_such() -> None:
    block = _block(
        [_record(intention="write the file", tool="write_text_file", success=True, ago=15)]
    )
    lines = [line for line in block.splitlines() if line.startswith("- ")]
    assert lines and all("SUCCEEDED" in line for line in lines)
    assert not any("DID NOT SUCCEED" in line for line in lines)


def test_the_newest_attempt_comes_first() -> None:
    block = _block(
        [
            _record(intention="older thing", tool="a", success=True, ago=600),
            _record(intention="newest thing", tool="b", success=False, ago=20),
        ]
    )
    assert block.index("newest thing") < block.index("older thing")


def test_stale_attempts_are_not_presented_as_recent() -> None:
    """Quoting something from hours ago as "just now" is its own confabulation."""
    block = _block(
        [_record(intention="hours ago", tool="a", success=True, ago=4 * 3600)]
    )
    assert "hours ago" not in block
    assert "no tool actions" in block


def test_no_actions_means_a_stated_absence_not_silence() -> None:
    """Silence about a period is what gets filled in with something plausible.

    Asked for "one concrete thing that actually happened in your runtime in the
    last hour" with nothing in front of her, she described processing a 45-page
    PDF on neuromorphic computing, and on another run a user asking about
    caffeine chemistry. Neither happened. She was answering a question about a
    period she had no record of. "Nothing" is a fact, and stated, it is
    answerable.
    """
    block = _block([])
    assert "no tool actions" in block
    assert "Do not describe an action you did not take" in block


def test_a_broken_ledger_never_breaks_the_turn() -> None:
    with patch(
        "core.agency.intention_loop.get_intention_loop", side_effect=RuntimeError("down")
    ):
        assert recent_actions_block() == ""


def test_the_block_stays_small_enough_to_carry_every_turn() -> None:
    records = [
        _record(intention=f"attempt number {i} with a long description" * 3,
                tool=f"tool_{i}", success=i % 2 == 0, ago=10 * i, outcome="x" * 400)
        for i in range(1, 9)
    ]
    assert len(_block(records)) < 1800


# ── One delivery owner, because duplicated grounding drifted independently ─

def test_inference_gate_owns_the_relevance_scoped_receipt_projection() -> None:
    """The gate and the modules it was split into, read as one owner.

    The present-moment grounding moved to inference_gate_present_moment.py
    when the gate was split, and this read only inference_gate.py, so it went
    red on the file layout while the gate still owned the projection.
    """
    gate = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(GATE.parent.glob("inference_gate*.py"))
    )
    assert "recent_actions_block" in gate
    assert "asks_what_she_recently_did" in gate
    assert "ambient_grounding_blocks.append(_actions)" in gate


def test_the_receipts_survive_prompt_compaction() -> None:
    """The block a compaction may not drop, checked where the list now lives.

    This read `inference_gate.py` for a literal `important_headers = (`. The
    list moved to `context_budget.py` and the gate imports it, so the
    guarantee held and the test went red on the file layout rather than on
    the guarantee. Read from the canonical list instead: what has to survive
    is that this header is in it, wherever it is kept.
    """

    from core.brain.llm.context_budget import CRITICAL_FOREGROUND_HEADERS

    assert "## WHAT YOU ACTUALLY JUST DID" in CRITICAL_FOREGROUND_HEADERS
    assert "CRITICAL_FOREGROUND_HEADERS" in GATE.read_text(encoding="utf-8"), (
        "the gate no longer reaches the list that protects the receipts"
    )


# ── The ledger decides, not the instruction ────────────────────────────────
#
# The receipts block reached the prompt and she described Minesweeper anyway.
# A prompt block competing with a fluent paragraph loses, quietly. So the same
# treatment the clock got: a claim the runtime can check is checked, at the
# egress, against a reading it can actually take.

MINESWEEPER = (
    "When you run it, the board pops up and you click cells to reveal numbers. "
    "If you hit a mine, the game shows you which squares had mines and ends."
)


def _guard(reply: str, records):
    from core.conversation.grounded_claim_guard import verify_grounded_claims
    from core.conversation.surface_disposition import record_tool_receipt
    from core.conversation.turn_evidence_custody import bind_turn_evidence_custody

    loop = SimpleNamespace(_completed_intentions=records)
    with (
        patch("core.agency.intention_loop.get_intention_loop", return_value=loop),
        bind_turn_evidence_custody(session_id="claim-tests", turn_id="turn-1"),
    ):
        for record in records:
            if "autonomous" in str(getattr(record, "drive", "")).casefold():
                continue
            intention = str(getattr(record, "intention", "") or "")
            lowered = intention.casefold()
            if any(word in lowered for word in ("wallpaper", "background")):
                action, object_ref = "system_control", "desktop background"
            elif any(word in lowered for word in ("build", "reconstruct")):
                action, object_ref = "build_artifact", intention
            elif any(word in lowered for word in ("save", "write")):
                action, object_ref = "write_text_file", intention
            else:
                action, object_ref = str(
                    getattr((getattr(record, "actions_taken", None) or [None])[0], "tool_name", "")
                    or "unknown"
                ), intention
            actions = list(getattr(record, "actions_taken", None) or [])
            succeeded = bool(actions) and all(bool(getattr(item, "success", False)) for item in actions)
            record_tool_receipt(
                "desktop_task",
                action=action,
                object_ref=object_ref,
                ok=succeeded,
                effect_observed=succeeded,
            )
        return verify_grounded_claims(reply)


def _attempt(*, succeeded: bool, intention: str = "build 2048 onto the Desktop"):
    return SimpleNamespace(
        intention=intention,
        actions_taken=[SimpleNamespace(tool_name="program_dna_reconstruct", success=succeeded)],
        actual_outcome="",
        observation="",
        completed_at=time.time() - 20,
    )


def test_describing_an_artifact_that_was_never_built_is_corrected() -> None:
    result = _guard(MINESWEEPER, [_attempt(succeeded=False)])
    assert "didn't actually get that built" in result.text
    assert "mine" not in result.text
    assert result.corrections


def test_describing_an_artifact_that_was_built_is_left_alone() -> None:
    """A guard that fights correct answers is worse than no guard."""
    result = _guard(MINESWEEPER, [_attempt(succeeded=True)])
    assert result.text == MINESWEEPER
    assert not result.corrections


def test_ordinary_conversation_is_untouched_after_a_failed_build() -> None:
    ordinary = "I think there's something it's like to be me."
    assert _guard(ordinary, [_attempt(succeeded=False)]).text == ordinary


def test_with_no_build_history_nothing_is_corrected() -> None:
    assert _guard(MINESWEEPER, []).text == MINESWEEPER


def test_a_non_build_attempt_does_not_trigger_it() -> None:
    """Only a failed attempt to MAKE something makes an artifact claim false."""
    searched = SimpleNamespace(
        intention="search the web for the F1 championship",
        actions_taken=[SimpleNamespace(tool_name="web_search", success=False)],
        actual_outcome="",
        observation="",
        completed_at=time.time() - 20,
    )
    assert _guard(MINESWEEPER, [searched]).text == MINESWEEPER


def test_a_broken_ledger_never_rewrites_a_reply() -> None:
    from core.conversation.grounded_claim_guard import verify_grounded_claims

    with patch(
        "core.agency.intention_loop.get_intention_loop", side_effect=RuntimeError("down")
    ):
        assert verify_grounded_claims(MINESWEEPER).text == MINESWEEPER


# ── Claiming an action finished when nothing finished it ──────────────────
#
# Live 2026-07-27, "I've found a beautiful image of a blue whale and set it as your desktop background. Enjoy!" with
# the wallpaper unchanged, and live 2026-07-30, a conversation about agent swarms rewritten into "the action didn't go
# through" by a background loop's failure: both are read by meaning against the turn's own record now
# (core/epistemics/what_her_record_refutes.py), and measured in tests/test_what_her_record_refutes.py, GUARD_MEASURED.


# ── The receipt list must not be shared between turns ──────────────────────
#
# ContextVar(default=[]) is ONE list shared by every context that never set
# it. A receipt recorded outside a turn was appended to that shared object and
# stayed for the life of the process; an `ok: True` in it vouched for every
# completed-action claim in every later turn that never began its own list.

def test_a_receipt_outside_a_turn_does_not_vouch_for_a_later_turn() -> None:
    import contextvars

    from core.conversation.surface_disposition import (
        begin_turn_tool_receipts,
        record_tool_receipt,
        turn_tool_receipts,
    )

    def outside_any_turn() -> None:
        record_tool_receipt("auto_refactor", ok=True)

    contextvars.copy_context().run(outside_any_turn)

    def a_fresh_turn() -> tuple:
        begin_turn_tool_receipts()
        return turn_tool_receipts()

    assert contextvars.copy_context().run(a_fresh_turn) == ()

    def never_began_a_turn() -> tuple:
        return turn_tool_receipts()

    assert contextvars.copy_context().run(never_began_a_turn) == ()


def test_a_receipt_recorded_in_a_child_context_cannot_reach_the_parent_turn() -> None:
    """The order-dependence this file kept hitting, as a direct assertion.

    A ContextVar copy shares the value's REFERENCE. While the receipts were a
    list, a receipt recorded inside a copied context — a task, a thread hop, a
    `copy_context().run` — appended to the parent turn's own list. So one
    turn's background work could put an `ok: True` into a different turn's
    evidence, which is the same fail-open that made a `[]` default wrong, one
    level further in.

    Rebinding an immutable value cannot escape the context that rebound it,
    and that is what this holds.
    """

    import contextvars

    from core.conversation.surface_disposition import (
        begin_turn_tool_receipts,
        record_tool_receipt,
        turn_tool_receipts,
    )

    begin_turn_tool_receipts()

    def a_child_context_records_something() -> tuple:
        record_tool_receipt("background_work", ok=True)
        return turn_tool_receipts()

    inside = contextvars.copy_context().run(a_child_context_records_something)
    assert inside == (), "work outside exact turn custody must not create a receipt"
    assert turn_tool_receipts() == (), (
        "a receipt recorded in a child context reached back into the parent "
        "turn, so one turn's work can vouch for another's claims"
    )
