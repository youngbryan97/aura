"""A prompt too large to prefill returns one token and no text.

Live 2026-08-03, repeatedly:

    ⏱️ [WORKER] Request deadline reached at token 1; stopping decode cooperatively.
    ⚠️ [WORKER] Generation produced 1 token(s) but no text survived to the
       caller — discarded downstream, not a decode failure.
       Prompt length: 91441

Prefill alone consumed the entire request deadline, so the answer came back
empty — which then became "cognitive cycle produced nothing", which on a
fail-closed subsystem became CRITICAL SERVICE FAILURE and took long-term
memory consolidation with it.

inference_gate has per-section AND total prompt budgets. A path that
assembles its own prompt never meets them. This is the last boundary every
generation crosses, so a bypass upstream cannot route around it.
"""
from __future__ import annotations

import pytest

from core.brain.llm.mlx_client import _prompt_within_prefill_ceiling
from core.brain.llm.prefill_ceiling import KEEP_HEAD_CHARS as _PREFILL_KEEP_HEAD_CHARS
from core.brain.llm.prefill_ceiling import prefill_ceiling_chars

#: The ceiling the prompts below are for: the active cortex's.
_PREFILL_CEILING_CHARS = prefill_ceiling_chars()

HEAD = "SYSTEM-CONTRACT-HEAD"
TAIL = "THE-ACTUAL-QUESTION-TAIL"


def _oversized(total: int = _PREFILL_CEILING_CHARS + 50_000) -> str:
    filler = "m" * max(0, total - len(HEAD) - len(TAIL))
    return f"{HEAD}{filler}{TAIL}"


class TestOrdinaryPromptsAreUntouched:
    @pytest.mark.parametrize("size", [0, 1, 5_000, _PREFILL_CEILING_CHARS])
    def test_anything_within_the_ceiling_passes_through_byte_for_byte(self, size):
        prompt = "x" * size
        assert _prompt_within_prefill_ceiling(prompt) == prompt

    def test_none_becomes_empty_not_the_string_none(self):
        assert _prompt_within_prefill_ceiling(None) == ""


class TestAnOversizedPromptIsMadeAnswerable:
    def test_the_result_fits_under_the_ceiling(self):
        bounded = _prompt_within_prefill_ceiling(_oversized())
        assert len(bounded) <= _PREFILL_CEILING_CHARS

    def test_the_system_contract_survives(self):
        assert _prompt_within_prefill_ceiling(_oversized()).startswith(HEAD)

    def test_the_question_survives(self):
        """The question is at the END. Truncating from the tail would drop it."""
        assert _prompt_within_prefill_ceiling(_oversized()).endswith(TAIL)

    def test_the_gap_is_declared(self):
        """The model must not reason across a hole it cannot see."""
        assert "characters omitted" in _prompt_within_prefill_ceiling(_oversized())

    def test_it_keeps_the_head_it_promises(self):
        bounded = _prompt_within_prefill_ceiling(_oversized())
        assert bounded[:_PREFILL_KEEP_HEAD_CHARS] == _oversized()[:_PREFILL_KEEP_HEAD_CHARS]

    @pytest.mark.parametrize("size", [48_001, 60_000, 91_441, 500_000])
    def test_every_oversize_fits(self, size):
        bounded = _prompt_within_prefill_ceiling(_oversized(size))
        assert len(bounded) <= _PREFILL_CEILING_CHARS
        assert bounded.endswith(TAIL)


class TestItIsWiredAtTheLastBoundary:
    def test_the_generate_dispatch_bounds_the_prompt(self):
        import inspect

        from core.brain.llm import mlx_client

        source = inspect.getsource(mlx_client)
        dispatch = source[source.index('"action": "generate",')]
        assert dispatch  # anchor exists
        # The call must precede the request that carries the prompt.
        cap_at = source.index("_prompt_within_prefill_ceiling(prompt")
        req_at = source.index('"action": "generate",')
        assert cap_at < req_at, "the prompt must be bounded before it is dispatched"


class TestTheCeilingIsWhatTheCortexWasQualifiedToRead:
    """It was a fixed 48,000 characters while the qualified lanes read 24,576 tokens."""

    @staticmethod
    def _limits(monkeypatch, lanes, *, qualified=True):
        from types import SimpleNamespace

        from core.brain.llm import model_registry, token_budget_evidence

        profile = SimpleNamespace(
            qualified=qualified,
            lanes=tuple(SimpleNamespace(max_input_tokens=tokens) for tokens in lanes),
        )
        monkeypatch.setattr(
            model_registry,
            "get_active_cortex_serving_limits",
            lambda model_path=None: profile if model_path in (None, "cortex") else None,
        )
        monkeypatch.setattr(
            token_budget_evidence,
            "chars_per_token",
            lambda: SimpleNamespace(tokens_to_chars=lambda tokens: int(tokens * 3.5)),
        )

    def test_the_widest_qualified_lane_in_characters(self, monkeypatch):
        self._limits(monkeypatch, (8_192, 24_576))
        assert prefill_ceiling_chars() == int(24_576 * 3.5)
        assert prefill_ceiling_chars("cortex") == int(24_576 * 3.5)

    def test_another_model_or_no_profile_keeps_the_fixed_figure(self, monkeypatch):
        from core.brain.llm.prefill_ceiling import UNQUALIFIED_CEILING_CHARS

        self._limits(monkeypatch, (24_576,))
        assert prefill_ceiling_chars("the-smaller-fallback") == UNQUALIFIED_CEILING_CHARS
        self._limits(monkeypatch, (24_576,), qualified=False)
        assert prefill_ceiling_chars() == UNQUALIFIED_CEILING_CHARS


def test_reading_beyond_the_old_ceiling_is_earned_by_the_answer(monkeypatch) -> None:
    """LIVE 2026-10-03: a 640-token answer read 130,902 characters once the window widened."""
    from core.brain.llm import context_budget, prefill_ceiling

    monkeypatch.setattr(prefill_ceiling, "prefill_ceiling_chars", lambda model_path=None: 170_000)
    earned = {640: 30_000, 4_000: 120_000, 20_000: 600_000}
    monkeypatch.setattr(context_budget, "budget_for_answer", lambda tokens: earned.get(tokens, 0))
    floor = prefill_ceiling.UNQUALIFIED_CEILING_CHARS
    assert prefill_ceiling.reading_ceiling_chars(640) == floor
    assert prefill_ceiling.reading_ceiling_chars(4_000) == 120_000
    assert prefill_ceiling.reading_ceiling_chars(20_000) == 170_000
    # Nothing timed yet: the fixed figure, as before the window was widened.
    assert prefill_ceiling.reading_ceiling_chars(7) == floor
