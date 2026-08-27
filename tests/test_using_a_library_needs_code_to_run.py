"""Being handed a library and asked to use it is a request to run code.

LIVE, 2026-08-27: "docs and source are at <path>. Read it, then actually use
it: open a ledger, post an invoice, reverse it, tell me the trial balance."

Only the file domain was recognised, so the only tool offered could read files.
The model tried three times to WRITE a script with it and was vetoed three
times — correctly, a read lease is not a write lease — and the turn ended in "I
couldn't get to an answer I'd stand behind", for a task the sandbox completes in
forty milliseconds.

The code domain was admitted for arithmetic and for finite constraint problems:
things that settle BY computation. A library settles nothing; it is used by
being called, and calling it is running code.
"""

from __future__ import annotations

import pytest

from core.intent.declared_capability import requested_foundational_domains
from core.intent.needs_computation import asks_to_exercise_software

_PATH = "/private/tmp/claude-501/-Users-bryan--aura-live-source/scratchpad/ledgerkit"

_THE_LIVE_REQUEST = (
    f"There's a little library I use called ledgerkit — docs and source are at {_PATH}. "
    "I've never explained it to you. Read it, then actually use it: open a ledger, post a "
    "£250.00 consulting invoice, then reverse the hosting one. Tell me the trial balance."
)


def test_the_live_request_needs_both_domains() -> None:
    domains = requested_foundational_domains(_THE_LIVE_REQUEST)
    assert "file" in domains, "it has to read the docs"
    assert "code" in domains, "and it has to run the library"


@pytest.mark.parametrize(
    "asked",
    [
        "try that library out and tell me what it returns",
        "call the API and show me the response",
        "run the parser over those two examples and tell me the totals",
        "use the client to fetch the record and print it",
    ],
)
def test_asking_for_software_to_be_exercised(asked: str) -> None:
    assert asks_to_exercise_software(asked) is True


@pytest.mark.parametrize(
    "asked",
    [
        "what does that library do?",
        "explain how the ledger works",
        "tell me about the API",
        "summarise the docs for me",
        "write me a one-pager about the migration",
        "how does the parser handle unicode?",
    ],
)
def test_asking_about_software_is_not_asking_for_it_to_run(asked: str) -> None:
    """A description costs nothing to give and needs no execution lane."""
    assert asks_to_exercise_software(asked) is False


def test_a_description_does_not_open_the_code_domain() -> None:
    """The whole point of the domain reader: mood is not machine I/O."""
    assert "code" not in requested_foundational_domains("explain how the ledger works")
    assert "code" not in requested_foundational_domains("what does that library do?")


_CSV = (
    "I've got a deals export at /private/tmp/claude-501/scratchpad/deals.csv that I've "
    "never shown you. How many of them are approved, what do they add up to in total, "
    "and which region has the highest average approved deal size?"
)


def test_an_aggregate_over_a_named_file_needs_code_too() -> None:
    """LIVE, 2026-08-27: only the file reader was offered, and the turn failed.

    Counting and averaging a spreadsheet is arithmetic whose operands are in the
    file, so the sentence carries none — which is why the readers for arithmetic
    written in the sentence found nothing.
    """
    from core.intent.needs_computation import asks_for_an_aggregate, needs_computation

    assert asks_for_an_aggregate(_CSV) is True
    assert needs_computation(_CSV) is True
    domains = requested_foundational_domains(_CSV)
    assert "file" in domains and "code" in domains


@pytest.mark.parametrize(
    "asked",
    [
        "how many people live in Peru?",
        "what does that library do?",
        "explain how the ledger works",
        "how much do you like this idea?",
    ],
)
def test_an_aggregate_with_nothing_to_aggregate_is_not_computation(asked: str) -> None:
    """A question about the world is not a computation over a named thing."""
    from core.intent.needs_computation import asks_for_an_aggregate

    assert asks_for_an_aggregate(asked) is False


def test_one_module_owns_the_judgement() -> None:
    """Two readers of one judgement is what this tree keeps paying for."""
    from core.intent import exercising_software, needs_computation

    assert (
        exercising_software.asks_to_exercise_software
        is needs_computation.asks_to_exercise_software
    )
