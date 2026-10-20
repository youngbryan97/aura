"""A turn that carries the time the world gives it is fitted to that time; one that does not keeps its clock as it was."""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


def test_an_answer_is_fitted_to_the_time_given_at_the_measured_rate(monkeypatch):
    import core.brain.llm.thinking_reserve as reserve
    from core.brain.llm.an_answer_in_the_time_given import A_SHORT_LINE, fitted_to_the_time_given

    monkeypatch.setattr(reserve, "tokens_decodable_in", lambda seconds, model="", **_: int(seconds * 16))   # 16 a second
    assert fitted_to_the_time_given(seconds=12, read_s=2, margin_s=2, max_tokens=644) == 128           # 8 s of decode
    assert fitted_to_the_time_given(seconds=600, read_s=2, margin_s=2, max_tokens=644) == 644          # never above asked
    assert fitted_to_the_time_given(seconds=3, read_s=2, margin_s=2, max_tokens=644) == A_SHORT_LINE   # no time: a short line


def test_the_answer_clock_keeps_the_world_s_deadline_and_stretches_only_without_one(monkeypatch):
    import core.brain.llm.thinking_reserve as reserve
    from core.brain.inference_gate_turn_serving import _ServesTheTurn

    monkeypatch.setattr(reserve, "tokens_decodable_in", lambda seconds, model="", **_: int(seconds * 16))
    args = dict(_decode_s=77.0, _generations=1, _needed=106.0, _read_s=11.0, _reserve_the_worker_adds=219,
                _tokens_to_pay_for=863, max_tokens=644, timeout_val=91.0)
    primary, deadline, tokens = _ServesTheTurn._answer_clock(**args, world_seconds=15.0)
    assert deadline == 15.0 and primary < 15.0 and tokens < 644
    primary, deadline, tokens = _ServesTheTurn._answer_clock(**args)
    assert deadline == 106.0 and tokens == 644                    # as on 18 October at 16:26: stretched to fit
