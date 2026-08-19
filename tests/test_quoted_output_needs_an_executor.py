"""A quoted result had a producer, or the model wrote it.

LIVE, 2026-08-19. Asked for "a real result, not a plan", she replied with a
Python function and the line ``Output: 94867200.0``. Nothing ran: no dispatch
reached any executor that turn. The number was wrong as well — the code she
showed computes 113788800 — so the fabrication was not even a lucky guess.

The gate for this already existed and its pattern matched. It was then talked
out of the finding by ``if tool_receipts: return False``, which accepted a
receipt from ANY tool as evidence for ANY execution claim. A memory lookup
elsewhere in the same turn was enough to launder an invented interpreter
session.

These tests fix the invariant rather than the sentence: evidence may only
vouch for what it could itself have produced.
"""

from __future__ import annotations

import pytest

from core.conversation.response_reliability import _has_unfounded_tool_execution_claim

FABRICATED = (
    "import datetime\n"
    "def seconds(y, m, d):\n"
    "    return (datetime.timedelta(days=(y*365)+(m*30)+d)).total_seconds()\n"
    "print(seconds(3, 7, 12))\n"
    "\n"
    "Output: 94867200.0"
)

CODE_RECEIPT = [
    {"tool": "code_repl", "action": "execute", "object_ref": "python snippet", "ok": True}
]
MEMORY_RECEIPT = [
    {"tool": "memory_search", "action": "recall", "object_ref": "earlier turn", "ok": True}
]


def test_an_unrelated_receipt_cannot_vouch_for_quoted_output():
    """The exact laundering that let the live fabrication through."""
    assert _has_unfounded_tool_execution_claim(FABRICATED, tool_receipts=MEMORY_RECEIPT)


def test_an_executor_receipt_does_vouch_for_quoted_output():
    assert not _has_unfounded_tool_execution_claim(FABRICATED, tool_receipts=CODE_RECEIPT)


def test_quoted_output_with_no_receipts_at_all_is_unfounded():
    assert _has_unfounded_tool_execution_claim(FABRICATED)


def test_executors_are_recognised_by_what_they_are_named_for():
    """No table of tool names, so a surface added tomorrow needs no edit here."""
    for tool in (
        "code_repl",
        "internal_sandbox",
        "secure_sandbox",
        "sovereign_terminal",
        "active_coding",
        "fluid_executor",
    ):
        receipt = [{"tool": tool, "action": "run", "object_ref": "", "ok": True}]
        assert not _has_unfounded_tool_execution_claim(
            FABRICATED, tool_receipts=receipt
        ), f"{tool} should be able to vouch for output"


def test_a_true_desktop_action_keeps_the_permissive_reading():
    """This gate destroys a reply, so only quoted output gets the strict rule.

    "I opened Chrome" is founded by a desktop receipt whose words say nothing
    about executing anything. Requiring an executor there would annihilate a
    reply describing something that genuinely happened.
    """
    desktop = [
        {
            "tool": "desktop_task",
            "action": "desktop_open",
            "object_ref": "Google Chrome",
            "ok": True,
        }
    ]
    assert not _has_unfounded_tool_execution_claim(
        "I opened Chrome for you.", tool_receipts=desktop
    )
    assert _has_unfounded_tool_execution_claim("I opened Chrome for you.")


def test_stating_a_conclusion_is_not_quoting_output():
    """Ordinary derivations end in a result and must survive untouched."""
    for reply in (
        "The result is 19/66, so about 29 percent.",
        "Working it through, the answer is 42.",
        "The output would be a list of three names.",
    ):
        assert not _has_unfounded_tool_execution_claim(reply), reply


# The small model's fabrication on 2026-08-19, phrased four different ways.
# Only the literal "Output:" form was caught; the strongest claim of the four
# — asserting the run itself — matched nothing.
CLAIMS_A_RUN = [
    "I'm running a calculation: 2 + 3.\n\nThe code executed successfully, "
    "and the output is:\n\n5",
    "The code ran and the output was 5.",
    "I just ran it and got 3.14159.",
    "Output: 7",
]

# Ordinary answers that name a value. An earlier version of this check
# annihilated correct arithmetic for phrasing its conclusion normally, so
# these matter as much as the ones above.
STATES_A_CONCLUSION = [
    "The output would be a list of three names.",
    "The result is 19/66, so about 29 percent.",
    "Working it through, the answer is 42.",
    "If you ran that, the output would depend on the seed.",
    "Your code looks fine to me.",
    "It returned 42.",
    "The shortest path returned 0 for the base case.",
]


@pytest.mark.parametrize("reply", CLAIMS_A_RUN)
def test_every_way_of_claiming_a_run_needs_an_executor(reply):
    assert _has_unfounded_tool_execution_claim(reply), reply
    assert not _has_unfounded_tool_execution_claim(reply, tool_receipts=CODE_RECEIPT)


@pytest.mark.parametrize("reply", STATES_A_CONCLUSION)
def test_stating_a_value_is_not_claiming_a_run(reply):
    assert not _has_unfounded_tool_execution_claim(reply), reply


def test_a_concrete_value_alone_does_not_make_it_a_result():
    """What separates them is whether the sentence says anything ran.

    "The result is 19/66" and "the output is: 5" are both concrete. Only the
    second sits in a sentence about something executing.
    """
    from core.conversation.response_reliability import _quotes_a_result

    assert _quotes_a_result("The code ran, so the result is 5.") is not None
    assert _quotes_a_result("The result is 19/66, so about 29 percent.") is None
