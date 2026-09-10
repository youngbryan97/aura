"""The block that changes most looked perfectly stable, so nothing moved.

`split_on_volatility` keeps the stable part of a prompt in the authority head
and moves the per-turn part down beside the turn, because on this model a
strict prefix is the only reuse a KV cache can give. What it moves is decided
by `volatility_of`, which prefers what has actually been WATCHED changing over
the authored prior.

Nothing had ever been watched. `section_volatility.json` held zero sections
after months of running, and the reason is that a bracketed header carries its
own reading:

    [Affect: Current Mood: TIRED (substrate energy: 0.14, substrate focus: 0.50)]

Filed under the whole line, every turn invents a section that has never been
seen before, its predecessor is never found, and no count is ever incremented.
The most volatile text in the prompt sat inside the stable head on every turn.

LIVE, 2026-09-08: `matched 596 (25.5%) before diverging; divergent text begins:
' INQUISITIVE (substrate energy: 0.31, substrate focus: 0.78, substrate'`.
Three quarters of the prompt re-prefilled every turn.
"""

from __future__ import annotations

import pytest

from core.brain.llm import context_budget as budget


@pytest.fixture(autouse=True)
def _a_fresh_window(tmp_path, monkeypatch):
    monkeypatch.setenv("AURA_STATE_ROOT", str(tmp_path))
    budget._CHANGED.clear()
    budget._LAST_SEEN.clear()
    yield
    budget._CHANGED.clear()
    budget._LAST_SEEN.clear()


def test_the_label_is_the_identity_and_the_rest_is_the_reading():
    assert budget.identity_of(
        "[Affect: Current Mood: TIRED (substrate energy: 0.14)]"
    ) == "[Affect]"
    assert budget.identity_of("[Soma: CPU 41%, VRAM 0%]") == "[Soma]"
    # A header that is not a reading is left exactly as it is.
    assert budget.identity_of("## RELEASE CONTRACT") == "## RELEASE CONTRACT"
    assert budget.identity_of("[PRESENT MOMENT]") == "[PRESENT MOMENT]"
    assert budget.identity_of("") == ""


def _a_prompt(turn: int) -> str:
    return (
        "You are Aura Luna.\n\n"
        "## RELEASE CONTRACT\nThe same contract text on every turn.\n\n"
        f"[Affect: Current Mood: MOOD{turn} (substrate energy: 0.{turn}1)]\n"
        f"[Soma: CPU {turn}1%, VRAM 0%]"
    )


def test_a_block_that_changes_every_turn_is_measured_as_changing():
    for turn in range(9):
        budget.observe_sections(_a_prompt(turn))
    assert budget.measured_volatility("[Affect]") == 1.0
    assert budget.measured_volatility("[Soma]") == 1.0
    assert budget.measured_volatility("## RELEASE CONTRACT") == 0.0


def test_what_was_measured_is_what_decides_the_split():
    for turn in range(9):
        budget.observe_sections(_a_prompt(turn))
    head, tail = budget.split_on_volatility(_a_prompt(99))
    assert "## RELEASE CONTRACT" in head
    assert "You are Aura Luna." in head
    assert "[Affect:" not in head
    assert "[Soma:" not in head
    assert "[Affect:" in tail
    assert "[Soma:" in tail


def test_nothing_of_the_prompt_is_dropped_or_reworded():
    for turn in range(9):
        budget.observe_sections(_a_prompt(turn))
    prompt = _a_prompt(99)
    head, tail = budget.split_on_volatility(prompt)
    for line in prompt.splitlines():
        if line.strip():
            assert line.strip() in head or line.strip() in tail, line


def test_the_router_puts_each_reading_on_its_own_line():
    """A line holding two bracket groups matches no header, so the pattern
    that moves volatile text could not see the block at all."""
    import inspect

    from core.brain import llm_health_router

    source = inspect.getsource(llm_health_router)
    assert 'context_header = "\\n".join(ctx_summary)' in source
    assert 'context_header = " ".join(ctx_summary)' not in source
