"""What a turn can afford to read, and what it keeps when it cannot read it all.

The runtime had two prompt builders and one budget between them. The gate
trimmed to a table; the builder every desktop conversation goes through sent
whatever the assembler produced, measured live at 96,430 characters against a
question of 213.
"""

from __future__ import annotations

from core.brain.context_budget import (
    CRITICAL_FOREGROUND_HEADERS,
    budget_for_answer,
    fit_to_budget,
    section_volatility,
    sections_of,
)

_PROMPT = """You are Aura Luna. Here is who you are.

## CODING WORKING SET
python module import class function repository diff commit

## PRESENT MOMENT
It is late and the sky is clear.

## SOURDOUGH NOTES
starter flour hydration levain bake loaf oven crumb

## GOAL EXECUTION STATE
goal step pending blocked retry
"""


def test_the_head_is_a_section_even_though_it_has_no_header() -> None:
    """The identity the whole prompt hangs off is not optional."""

    found = sections_of(_PROMPT)
    assert found[0].header == ""
    assert found[0].text.startswith("You are Aura Luna")
    assert [part.header for part in found[1:]] == [
        "## CODING WORKING SET",
        "## PRESENT MOMENT",
        "## SOURDOUGH NOTES",
        "## GOAL EXECUTION STATE",
    ]


def test_a_header_written_inside_a_sentence_is_not_a_section() -> None:
    """Recalled memory and fetched pages contain text somebody wrote."""

    body = "You are Aura.\n\n## REAL\nkept\n\nthe user wrote ## FAKE in a line"
    assert [part.header for part in sections_of(body)] == ["", "## REAL"]


def test_a_prompt_inside_its_budget_is_returned_untouched() -> None:
    """Trimming what already fits spends the risk for nothing."""

    assert fit_to_budget(_PROMPT, "anything", budget=len(_PROMPT)) == _PROMPT


def test_what_survives_is_decided_by_the_request() -> None:
    """The same prompt, two questions, two different survivors."""

    baking = fit_to_budget(_PROMPT, "how do I feed my sourdough starter", budget=150)
    code = fit_to_budget(_PROMPT, "which python module holds that function", budget=150)
    # Whichever section the request implicates survives whole; what is left of
    # the budget is filled with the next best, so the test is which one is
    # intact rather than which one appears at all.
    assert baking.endswith("crumb") and "commit" not in baking
    assert code.rstrip("…").endswith("commit") or "commit" in code
    assert "crumb" not in code


def test_grounding_is_kept_whatever_the_request_mentions() -> None:
    """A turn cannot wait to be asked before it knows when it is."""

    assert "## PRESENT MOMENT" in CRITICAL_FOREGROUND_HEADERS
    for asked in ("how do I feed my sourdough starter", "which python module"):
        kept = fit_to_budget(
            _PROMPT, asked, budget=150, always=CRITICAL_FOREGROUND_HEADERS
        )
        assert "## PRESENT MOMENT" in kept, asked
        assert kept.startswith("You are Aura Luna")


def test_a_word_in_every_section_decides_nothing() -> None:
    """Weighting is measured on the sections rather than authored.

    A request made entirely of words every section shares cannot prefer one,
    so the always-kept sections and the head are what is left.
    """

    body = "head\n\n## A\nshared alpha\n\n## B\nshared beta\n\n## C\nshared gamma"
    kept = fit_to_budget(body, "shared", budget=30)
    assert kept.startswith("head")


def test_the_budget_stays_inside_what_it_was_given() -> None:
    for budget in (60, 120, 240, 480):
        kept = fit_to_budget(
            _PROMPT, "sourdough", budget=budget, always=CRITICAL_FOREGROUND_HEADERS
        )
        assert len(kept) <= budget, (budget, len(kept))


def test_the_stable_prefix_is_emitted_before_the_volatile_tail() -> None:
    """A cached prompt is reuse of a byte-identical prefix.

    Sections that change every turn are emitted last, so the identity and the
    contracts in front of them stay reusable.
    """

    body = (
        "head\n\n## LIVE TONE\ninquisitive\n\n"
        "## LIVE DESKTOP RESPONSE CONTRACT\nanswer the question\n\n"
        "## GOALS\nfinish the pass"
    )
    kept = fit_to_budget(
        body, "goals contract tone", budget=len(body) - 1, volatility=section_volatility
    )
    assert kept.index("RESPONSE CONTRACT") < kept.index("## GOALS")
    assert kept.index("## GOALS") < kept.index("## LIVE TONE")
    assert section_volatility("## NEVER SEEN\nx") > section_volatility("## LIVE TONE\nx")


def test_a_longer_answer_earns_a_longer_prompt() -> None:
    """The budget is proportion, not a ceiling somebody picked.

    Both sides come from rates the reserve measured, so this asserts the
    ordering rather than any particular number: a turn allowed more room to
    write is allowed more room to read.
    """

    from core.brain.llm import thinking_reserve

    thinking_reserve.forget()
    # Ten readings is what the reserve calls enough to express a percentile.
    for _ in range(12):
        thinking_reserve.record_decode_rate(generated_tokens=100, elapsed_s=10.0)
        thinking_reserve.record_read_rate(prompt_chars=40_000, elapsed_s=20.0)

    small = budget_for_answer(50)
    large = budget_for_answer(1024)
    assert 0 < small < large
    # Fifty tokens is five seconds of writing, which buys ten thousand
    # characters of reading at the rates just recorded.
    assert small == 10_000, small
    assert large == 204_800, large


def test_an_unmeasured_rate_imposes_no_budget() -> None:
    """A budget that cannot be worked out is no budget, not a guess."""

    from core.brain.llm import thinking_reserve

    thinking_reserve.forget()
    assert budget_for_answer(0) == 0
    assert budget_for_answer(-5) == 0
