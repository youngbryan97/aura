"""A lane that answered nothing is not a lane that answered badly.

On the reports ground of 29 September the deep-inference phase failed 148 times
against 10 successes, and every one was reported as "inference output did not
contain a JSON object". The log beside it said the fast lane had returned no
text and that its circuit was open: the message named the model's JSON while the
fault was that nothing had been asked to produce any.

The phase asks a fast lane every turn, so a lane that is down fails it every
turn for one reason. 148 identical degradations bury the one that is new.
"""

from __future__ import annotations

import pytest

from core.phases.inference_phase import (
    _extract_json_object,
    _note_inference_failure,
    forget_inference_failures_for_test,
)

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _forget():
    forget_inference_failures_for_test()
    yield
    forget_inference_failures_for_test()


@pytest.fixture
def recorded(monkeypatch):
    seen: list[str] = []
    import core.phases.inference_phase as phase

    monkeypatch.setattr(phase, "_record_inference_degradation", lambda exc, **_k: seen.append(str(exc)))
    return seen


@pytest.mark.parametrize("nothing", ["", "   ", "\n\t ", None])
def test_a_lane_that_said_nothing_says_so(nothing):
    with pytest.raises(ValueError, match="the fast lane returned no text"):
        _extract_json_object(nothing)


def test_prose_with_no_object_still_says_that():
    with pytest.raises(ValueError, match="did not contain a JSON object"):
        _extract_json_object("I think they want reassurance.")


def test_an_object_is_still_read():
    assert _extract_json_object('noise {"momentum": "flowing"} tail') == {"momentum": "flowing"}


def test_malformed_braces_are_still_a_json_failure():
    import json

    with pytest.raises(json.JSONDecodeError):
        _extract_json_object('{"momentum": }')


def test_one_reason_holding_all_run_is_recorded_once(recorded):
    for _ in range(148):
        _note_inference_failure(ValueError("the fast lane returned no text"))
    assert len(recorded) == 1


def test_a_new_reason_is_recorded(recorded):
    _note_inference_failure(ValueError("the fast lane returned no text"))
    _note_inference_failure(ValueError("inference output did not contain a JSON object"))
    assert len(recorded) == 2


def test_a_recurrence_after_a_success_is_recorded_again(recorded):
    _note_inference_failure(ValueError("the fast lane returned no text"))
    forget_inference_failures_for_test()  # what a success does
    _note_inference_failure(ValueError("the fast lane returned no text"))
    assert len(recorded) == 2


def test_the_phase_clears_the_reason_when_it_succeeds(recorded):
    """A lane that comes back is heard: the reason it failed with is recorded again the next time it holds."""
    import asyncio

    from core.phases.inference_phase import InferencePhase
    from core.service_names import ServiceNames
    from core.state.aura_state import AuraState

    answers = iter(["", '{"implicit_intent": "help", "momentum": "flowing"}', ""])

    class Router:
        async def think(self, *_args, **_kwargs) -> str:
            return next(answers)

    class Container:
        @staticmethod
        def get(name, default=None):
            return Router() if name == ServiceNames.LLM_ROUTER else default

    phase = InferencePhase(Container())
    statuses = []
    for _ in range(3):
        state = AuraState.default()
        state.cognition.current_origin = "user"
        asyncio.run(phase.execute(state, objective="Can you check this?"))
        statuses.append(state.cognition.modifiers["deep_inference_status"]["status"])
    assert statuses == ["degraded", "ok", "degraded"]
    assert recorded == ["the fast lane returned no text"] * 2


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__])
