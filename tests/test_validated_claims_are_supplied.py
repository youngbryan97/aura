"""Asked for her evidence, she must be holding her evidence.

Live 2026-08-18. Asked "which measures, specifically? give me the numbers and
the sample sizes" about her own recurrence results:

    The data on cognitive performance showed a 15% improvement in pattern
    recognition tasks, with a sample size of n=42... All results were
    statistically significant at p<0.05... That's the summary straight out of
    my memory store — do you want me to pull up the full paper? I can give you
    a DOI...

No paper, no DOI, no participants, no such memory store. Every figure was
invented, and the invention arrived with a provenance claim attached.

The registry it should have come from was already populated at boot — 34
statements, each bound to the test that checks it, each graded by how strongly
it is evidenced. Nothing read it into a turn. A model asked for numbers it was
never given will produce numbers; the fix is to give it the real ones.
"""

from __future__ import annotations

import asyncio

import pytest

from core.brain.validated_claims_grounding import (
    CLAIMS_HEADER,
    asks_for_own_evidence,
    validated_claims_block,
)
from core.organism.model_validation import install_runtime_validation


@pytest.fixture(scope="module", autouse=True)
def _claims_installed():
    install_runtime_validation()


@pytest.mark.parametrize(
    "message",
    [
        # The live miss. It names no subject at all: a demand for evidence is
        # usually the SECOND thing someone says, so it inherits one.
        "which measures, specifically? give me the numbers and the sample sizes",
        "what is your evidence for that?",
        "how do you know that?",
        "what have you actually proven?",
        "show me the evidence",
        "what are the sample sizes?",
    ],
)
def test_a_demand_for_evidence_is_recognised(message: str) -> None:
    assert asks_for_own_evidence(message)


@pytest.mark.parametrize(
    "message",
    [
        "what is the evidence for dark matter?",
        "show me the data on unemployment",
        "how are you doing",
        "what is 2 + 2",
    ],
)
def test_evidence_about_the_world_is_a_different_question(message: str) -> None:
    assert not asks_for_own_evidence(message)


def _answer(prompt: str) -> str:
    """Whatever actually reaches the person, from whichever path answers.

    Two paths serve this question and only one of them speaks at a time:
    core.self.what_is_established handles a prompt that singles out a
    subject, and validated_claims_block steps aside for it and answers the
    rest. Asserting on one of them alone tests an implementation, and these
    tests failed for a year of the wrong reason when the other path started
    answering.
    """
    from core.self.what_is_established import what_is_established_block

    return what_is_established_block(prompt).strip() or validated_claims_block(prompt)


def test_the_block_carries_real_registered_claims() -> None:
    from core.organism.model_validation import get_suite

    block = _answer("what is your evidence for that?")
    statements = [
        " ".join(str(c.statement).split()) for c in get_suite().claims()
    ]

    assert block, "no claims block produced"
    assert statements, "registry is empty; the boot install did not run"
    assert any(s in block for s in statements), "block contains no real claim"


def test_every_claim_carries_how_strongly_it_is_evidenced() -> None:
    """A claim she can make and one she can support are different claims."""
    block = _answer("what have you actually proven?")

    assert "measured_live" in block or "measured_synthetic" in block
    assert "bound to the test that checks it" in block


def test_either_path_answers_but_never_both_at_once() -> None:
    """A question answered twice reads as two different registers."""
    from core.self.what_is_established import what_is_established_block

    for prompt in (
        "what is your evidence for that?",
        "what have you actually proven?",
        "show me the evidence",
        "what have you actually measured about yourself",
    ):
        nearest = what_is_established_block(prompt).strip()
        whole = validated_claims_block(prompt).strip()
        assert nearest or whole, f"{prompt!r} reached neither path"
        assert not (nearest and whole), f"{prompt!r} was answered twice"


def test_the_block_names_the_test_behind_each_claim() -> None:
    block = _answer("show me the evidence")

    assert "[checked by " in block


def test_an_empty_registry_reads_as_empty_not_as_no_claims() -> None:
    """Silence and "nothing is proven" are different readings."""

    class _Empty:
        def claims(self):
            return []

    import core.organism.model_validation as mv
    import core.self.what_is_established as established

    original = mv.get_suite
    original_registered = established._registered
    mv.get_suite = lambda: _Empty()  # type: ignore[assignment]
    # Both paths read the same register; emptying one and not the other tests
    # a state the runtime cannot be in.
    established._registered = lambda: []  # type: ignore[assignment]
    try:
        block = _answer("what is your evidence?")
    finally:
        mv.get_suite = original  # type: ignore[assignment]
        established._registered = original_registered  # type: ignore[assignment]

    assert "empty in this process" in block


def test_the_reading_reaches_the_grounding_channel() -> None:
    import core.brain.observable_registry  # noqa: F401
    from core.brain.observable_grounding import observable_blocks

    blocks = asyncio.run(
        observable_blocks("which measures? give me the numbers and sample sizes")
    )
    text = "\n".join(blocks) if isinstance(blocks, list) else str(blocks)

    assert CLAIMS_HEADER in text
