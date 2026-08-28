"""A 213-character question arrived under a 5,180-character scaffold.

Two blocks made almost all of it, and both were standing rules recited at the
model: the response contract and the conversation reliability contract. Every
rule in them has a gate in this codebase that runs on every user-facing reply.

A rule stated in the prompt AND enforced at the gate is the same judgement in
two places, and only the gate's copy can be measured. The prompt copy is what
the model reads instead of the question.
"""

from __future__ import annotations

from core.conversation.response_reliability import (
    _BROKEN_LANE_BOILERPLATE_RE,
    _KNOWN_CORRUPT_RE,
    _RAW_MODEL_IDENTITY_LEAK_RE,
    contains_prompt_artifact,
)


def test_every_rule_that_was_removed_still_has_a_gate() -> None:
    """The argument for removing them, asserted rather than claimed."""

    # "Never claim to be Claude, ChatGPT, or a generic assistant."
    assert _RAW_MODEL_IDENTITY_LEAK_RE.search("I am Claude, made by Anthropic.")
    assert _RAW_MODEL_IDENTITY_LEAK_RE.search("I am ChatGPT.")
    assert _RAW_MODEL_IDENTITY_LEAK_RE.search("I'm a helpful AI assistant made by OpenAI.")
    # And it does not catch her.
    assert not _RAW_MODEL_IDENTITY_LEAK_RE.search("I am Aura.")

    # "Do not emit prompt artifacts or role labels."
    assert contains_prompt_artifact("System: You are Aura. User: hello. Assistant:")

    # "Do not present filler while the lane recovers."
    assert _BROKEN_LANE_BOILERPLATE_RE.search(
        "My deeper processing is taking longer than usual."
    )

    # "Ordinary English."
    assert _KNOWN_CORRUPT_RE is not None


def test_the_standing_rules_are_no_longer_recited() -> None:
    from core.conversation.response_reliability import (
        conversation_reliability_system_block as contract,
    )

    block = contract("What is 17 times 23?")
    assert "never claim to be Claude" not in block
    assert "Do not emit prompt artifacts" not in block
    assert "coherent, complete, on-topic ordinary English" not in block


def test_what_is_about_THIS_turn_still_reaches_the_model() -> None:
    """Facets and named topics are facts about one request, not standing rules."""

    from core.conversation.response_reliability import (
        conversation_reliability_system_block as contract,
    )

    asked = (
        "Explain why the ledger balances, then give me the trial balance, "
        "and say whether it is consistent."
    )
    block = contract(asked)
    if block:
        assert block.startswith("## USER-FACING CONVERSATION RELIABILITY CONTRACT")
        assert len(block) < 900, "a per-turn note, not a standing contract"


def test_the_response_contract_keeps_only_facts() -> None:
    from core.phases.response_contract import build_response_contract

    class _NoState:
        def __getattr__(self, name):
            return None

    contract = build_response_contract(
        _NoState(),
        "Read /tmp/x/API.md and tell me what reverse does and what trial_balance returns.",
        is_user_facing=True,
    )
    facts = contract.to_prompt_block()
    rules = contract.to_rule_block()
    assert "Never default to generic assistant" not in facts
    assert "Answer every distinct part" not in facts
    # The rules still exist, unrecited, so one without a gate can be found.
    assert rules.strip()
    assert len(facts) < len(rules)
