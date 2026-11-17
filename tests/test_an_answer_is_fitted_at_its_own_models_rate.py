"""An answer is fitted to the time it is given at the rate the model that writes it has been measured writing, not at
an average every model on the host moves.

LIVE 17 November 2026, 07:33: her first reply after a start asked for 512 tokens and was given 160, was cut, and ended
on a stand-in's "I couldn't get to an answer I'd stand behind"; ten of the twenty-seven replies to a person in that
log were given fewer tokens than their plan asked for.
"""
from __future__ import annotations

import pytest

from core.brain.inference_gate import fit_the_answer_to_the_time
from core.brain.llm import decode_rates, mlx_client


@pytest.fixture
def rates():
    held = mlx_client.reset_host_rates_for_test()
    decode_rates.forget_for_test()
    yield
    decode_rates.forget_for_test()
    mlx_client.restore_host_rates_for_test(held)


def test_each_model_keeps_its_own_rate(rates):
    decode_rates.note_decode("/models/Aura-Qwen3.8-27B", 18.0)
    decode_rates.note_decode("/models/Ternary-Bonsai-2-27B-mlx-2bit", 2.0)
    decode_rates.note_decode("/models/Aura-Qwen3.8-27B", 20.0)
    assert decode_rates.decode_rate_of("Aura-Qwen3.8-27B") == pytest.approx(18.4)
    assert decode_rates.decode_rate_of("ternary-bonsai-2-27b-mlx-2bit") == 2.0
    assert decode_rates.decode_rate_of("never-measured") == 0.0


def test_a_reply_is_fitted_at_its_own_models_rate_when_another_model_slowed_the_host(rates):
    mlx_client._HOST_RATES.update(prefill=250.0, decode=2.0)  # the host's average, moved by a slow model
    decode_rates.note_decode("/models/Aura-Qwen3.8-27B", 18.0)
    prompt = "x" * 8837
    _, at_the_hosts_rate = fit_the_answer_to_the_time(prompt, 512, 60.0)
    _, at_its_own = fit_the_answer_to_the_time(prompt, 512, 60.0, model="Aura-Qwen3.8-27B")
    assert at_the_hosts_rate < 512, "the pooled rate cut it"
    assert at_its_own == 512
