"""A generation the runtime declined to run must not read as a model that answered with nothing.

LIVE, 2026-09-10: the inference gate refused a background request —
"kind=deferred reason=background_local_fallback_suppressed origin=
reimplementation_lab" — and the code generator raised "LLM returned no Python
source; the model returned nothing at all". Thirty-four emergency incidents
under that title, the resilience layer at full depletion, for a model that was
never asked. The router recorded its deferrals where the generator looks; the
gate did not.
"""

from __future__ import annotations

import pytest

from core.brain.llm import deferral_record
from core.brain.llm.code_generator import GenerationDeferredError, LLMCodeGenerator


@pytest.fixture(autouse=True)
def _clean_record():
    deferral_record.reset_for_test()
    yield
    deferral_record.reset_for_test()


class _RouterThatReturnsNothing:
    async def generate(self, *_args, **_kwargs):
        return ""


def test_the_gate_records_its_refusal_where_the_generator_looks() -> None:
    from core.brain.inference_gate import InferenceGate

    gate = InferenceGate.__new__(InferenceGate)
    gate._last_refusal_receipt = None
    gate._refuse_generation(
        "deferred", "background_local_fallback_suppressed", origin="reimplementation_lab"
    )
    entry = deferral_record.last_deferral(origin="reimplementation_lab")
    assert entry is not None
    assert "background_local_fallback_suppressed" in entry.reason


@pytest.mark.asyncio
async def test_a_deferred_generation_is_its_own_kind_of_outcome(monkeypatch) -> None:
    generator = LLMCodeGenerator(router=_RouterThatReturnsNothing())
    monkeypatch.setattr(generator, "_call_router", _RouterThatReturnsNothing().generate)
    deferral_record.record_deferral(
        origin="reimplementation_lab", reason="deferred: background_local_fallback_suppressed"
    )
    with pytest.raises(GenerationDeferredError) as caught:
        await generator.generate_async("write a function", {"origin": "reimplementation_lab"})
    assert "was not run" in str(caught.value)
    assert "returned nothing at all" not in str(caught.value)


@pytest.mark.asyncio
async def test_a_blank_answer_with_no_deferral_is_still_the_models_doing(monkeypatch) -> None:
    generator = LLMCodeGenerator(router=_RouterThatReturnsNothing())
    monkeypatch.setattr(generator, "_call_router", _RouterThatReturnsNothing().generate)
    with pytest.raises(RuntimeError) as caught:
        await generator.generate_async("write a function", {"origin": "reimplementation_lab"})
    assert not isinstance(caught.value, GenerationDeferredError)
    assert "returned nothing at all" in str(caught.value)
