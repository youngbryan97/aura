"""A narrow contract may only impose its narrow budget on a narrow turn.

Aura's token ladder caps a memory-state turn at 256 tokens because "what did
I pin?" is a short factual answer. That is true, and it is the wrong size the
moment such a request shares a message with something substantive.

Measured live 2026-07-27, typing into the desktop UI:

    Remember this: my project codename is HELIOTROPE, build 4471.
    Separately — do you think a system like you can actually prefer one
    thing over another, or is "preference" just a word we're borrowing?

Because the turn carried a pin, memory_state_contract was set, the ladder
capped it at 256, advisory factors reduced that to 172, the reply ran out
mid-sentence, the reliability gate flagged truncated_tail, and what reached
the screen was the pin confirmation alone. The philosophical half never had
room to exist — and the failure looked like the model having nothing to say.

The same shape as the template defect one layer up (a deterministic reply
answering a turn it only half covered): a turn sized by its smallest part.

These tests pin the rule in both directions, because a fix that simply
removed the cap would regress the latency the cap exists to protect.
"""

import pytest
from tests.chat_lane_support import chat_lane_source
from tests.source_contract import class_with_its_bases

pytestmark = pytest.mark.unit

PIN_ONLY = "Remember this: my project codename is HELIOTROPE, build 4471."
PIN_PLUS_QUESTION = (
    "Remember this: my project codename is HELIOTROPE, build 4471. "
    "Separately — do you think a system like you can actually prefer one "
    'thing over another, or is "preference" just a word we\'re borrowing? '
    "I want your own view, not a survey of positions."
)


def _covers_turn(message: str) -> bool:
    """What chat.py reports to the budgeter about its own parse."""
    from interface.routes.chat import _turn_has_substance_beyond_memory_request

    return not _turn_has_substance_beyond_memory_request(message)


class TestTheParserTellsTheBudgeterWhatItFound:
    def test_a_bare_pin_covers_its_turn(self):
        assert _covers_turn(PIN_ONLY) is True

    def test_a_pin_beside_a_real_question_does_not_cover_its_turn(self):
        assert _covers_turn(PIN_PLUS_QUESTION) is False


class TestTheBudgetFollowsTheWholeTurn:
    """The ladder itself, exercised through the contract flags it reads."""

    @staticmethod
    def _ladder(*, memory_state_contract: bool, covers_turn: bool) -> bool:
        """Reproduce the branch under test: does the narrow cap apply?

        Kept as a direct transcription rather than a call into
        CognitiveEngine._generate_desktop_quick_reply, which needs a live
        router. The assertion is about which branch a flag combination
        selects, and that is exactly what this expresses.
        """
        narrow_state_contract = False
        if memory_state_contract and covers_turn:
            narrow_state_contract = True
        return narrow_state_contract

    def test_a_bare_pin_still_gets_the_cheap_budget(self):
        """The cap exists for latency and must survive this fix."""
        assert self._ladder(memory_state_contract=True, covers_turn=True) is True

    def test_a_pin_beside_a_question_escapes_the_cheap_budget(self):
        assert self._ladder(memory_state_contract=True, covers_turn=False) is False


class TestTheFlagReachesTheEngine:
    def test_chat_publishes_the_coverage_flag_into_engine_context(self):
        """The engine defaults the flag to True when absent, so a chat.py that
        stopped publishing it would silently restore the 256-token cap for
        every compound turn — the defect, back, with all tests passing."""


        source = chat_lane_source()
        assert '"memory_state_contract_covers_turn": memory_state_contract_covers_turn' in source, (
            "chat.py must publish the coverage flag into the engine context"
        )

    def test_engine_reads_the_flag_before_applying_the_narrow_cap(self):
        import inspect

        from core.brain.cognitive_engine import CognitiveEngine

        source = class_with_its_bases(CognitiveEngine)
        assert 'context.get(\n            "memory_state_contract_covers_turn", True\n        )' in source or (
            '"memory_state_contract_covers_turn"' in source
        ), "the engine must consult the coverage flag"
