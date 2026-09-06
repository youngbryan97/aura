"""Six named parts of a prompt, and which one paid for the fit."""
from __future__ import annotations

import pytest

from core.brain.llm.context_budget import fit_to_budget
from core.brain.llm.who_got_the_room import (
    THE_PARTS,
    forget_everything,
    how_the_room_was_shared,
    note_a_fit,
    part_of,
    what_each_part_got,
    who_was_squeezed,
)

A_PROMPT = (
    "You are Aura.\n\n"
    + "I am who I am. " * 40
    + "\n\n## Recalled memory\n"
    + "She asked about the roof last week. " * 20
    + "\n\n## Felt state\n"
    + "a little tired. " * 4
    + "\n\n## Tools\n"
    + "read_file, write_file, browse. " * 10
    + "\n\n## Conversation\n"
    + "He said hello. She said hello. " * 20
    + "\n\n## The question\nWhat did I ask you about the roof?"
)


@pytest.fixture(autouse=True)
def _clean():
    forget_everything()
    yield
    forget_everything()


def test_the_head_of_a_prompt_is_identity_even_with_no_header() -> None:
    assert part_of("") == "identity"
    assert part_of("## Recalled memory") == "memory"
    assert part_of("## Felt state") == "interiority"
    assert part_of("## Tools") == "tools"
    assert part_of("## Conversation") == "history"
    assert part_of("## The question") == "reply"
    assert part_of("## Something nobody named") == "other"


def test_every_part_of_an_assembled_prompt_is_counted_somewhere() -> None:
    got = what_each_part_got(A_PROMPT)
    assert sum(got.values()) > 0
    for part in THE_PARTS:
        assert got[part.name] > 0, f"{part.name} was not found in the prompt"


def test_a_part_that_was_never_there_is_not_squeezed() -> None:
    thin = "You are Aura.\n\n## The question\nWhat time is it?"
    under = who_was_squeezed(thin, budget=10_000, before=thin)
    assert "memory" not in under, "absent is not squeezed"


def test_a_part_that_had_text_and_came_out_empty_is_the_worst_case() -> None:
    """Reading only the fitted prompt cannot tell absent from removed.

    The first version of this took the fitted prompt alone and reported
    nothing squeezed while the question being asked had been cut entirely.
    """
    fitted = fit_to_budget(A_PROMPT, "roof", budget=600)
    under = who_was_squeezed(fitted, budget=600, before=A_PROMPT)
    assert "reply" in under
    assert under["reply"]["emptied"] is True
    assert under["reply"]["had"] > 0
    assert under["reply"]["got"] == 0

    blind = who_was_squeezed(fitted, budget=600)
    assert "reply" not in blind, "which is exactly what made it invisible"


def test_a_prompt_that_already_fits_squeezes_nobody() -> None:
    fitted = fit_to_budget(A_PROMPT, "roof", budget=100_000)
    assert fitted == A_PROMPT
    assert who_was_squeezed(fitted, budget=100_000, before=A_PROMPT) == {}


def test_the_ledger_says_who_keeps_paying() -> None:
    for budget in (600, 800, 1000):
        note_a_fit(
            A_PROMPT,
            fit_to_budget(A_PROMPT, "roof", budget=budget),
            budget=budget,
            request="roof",
        )
    seen = how_the_room_was_shared()
    assert seen["fits"] == 3
    assert seen["fits_that_squeezed_someone"] == 3
    assert seen["pays_most_often"]
    assert seen["characters_lost"]["memory"] > 0


def test_the_floors_leave_room_for_the_fit_to_allocate_on_merit() -> None:
    total = sum(part.floor for part in THE_PARTS)
    assert 0.0 < total < 1.0, f"floors sum to {total}, leaving nothing to allocate"


def test_every_part_says_why_losing_it_matters() -> None:
    for part in THE_PARTS:
        assert part.why.strip(), f"{part.name} has a floor and no reason for it"
        assert part.reads_like, f"{part.name} cannot be found in a prompt"


def test_nothing_is_recorded_until_a_fit_is_noted() -> None:
    assert how_the_room_was_shared()["fits"] == 0


def test_the_assembler_records_who_paid_for_the_fit() -> None:
    """Wired, not beside: the ledger is on the path the prompt takes."""
    from core.brain.llm import context_assembler

    source = __import__("inspect").getsource(context_assembler)
    assert "note_a_fit(" in source, "the fit is not recorded on the real path"
    assert "who_got_the_room" in source
