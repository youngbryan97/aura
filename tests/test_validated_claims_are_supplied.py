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


def test_the_block_carries_real_registered_claims() -> None:
    from core.organism.model_validation import get_suite

    block = validated_claims_block("what is your evidence for that?")
    statements = [
        " ".join(str(c.statement).split()) for c in get_suite().claims()
    ]

    assert block, "no claims block produced"
    assert statements, "registry is empty; the boot install did not run"
    assert any(s in block for s in statements), "block contains no real claim"


def test_every_claim_carries_how_strongly_it_is_evidenced() -> None:
    """A claim she can make and one she can support are different claims."""
    block = validated_claims_block("what have you actually proven?")

    assert "measured_live" in block or "measured_synthetic" in block
    assert "bound to the test that checks it" in block


def test_the_block_names_the_test_behind_each_claim() -> None:
    block = validated_claims_block("show me the evidence")

    assert "[checked by " in block


def test_an_empty_registry_reads_as_empty_not_as_no_claims() -> None:
    """Silence and "nothing is proven" are different readings."""

    class _Empty:
        def claims(self):
            return []

    import core.organism.model_validation as mv

    original = mv.get_suite
    mv.get_suite = lambda: _Empty()  # type: ignore[assignment]
    try:
        block = validated_claims_block("what is your evidence?")
    finally:
        mv.get_suite = original  # type: ignore[assignment]

    assert "empty in this process" in block


def test_the_reading_reaches_the_grounding_channel() -> None:
    import core.brain.observable_registry  # noqa: F401
    from core.brain.observable_grounding import observable_blocks

    blocks = asyncio.run(
        observable_blocks("which measures? give me the numbers and sample sizes")
    )
    text = "\n".join(blocks) if isinstance(blocks, list) else str(blocks)

    assert CLAIMS_HEADER in text
