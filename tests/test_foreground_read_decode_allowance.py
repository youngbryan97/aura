from types import SimpleNamespace

import pytest


@pytest.mark.parametrize(
    "decode_rate,worker_read,private,expected",
    [(10.0, 160.0, 200, 296.0), (10.0, 200.0, 200, 336.0),
     (0.0, 200.0, 200, 180.0), (10.0, 160.0, 0, 276.0)],
)
def test_live_owner_funds_reading_and_both_output_channels(
    monkeypatch, decode_rate, worker_read, private, expected,
):
    from core.brain.llm import chat_format, mlx_client, thinking_reserve
    from core.phases.response_generation_unitary import UnitaryResponsePhase

    model = "/models/current-cortex"
    client = SimpleNamespace(
        get_worker_identity_snapshot=lambda: {"worker_model_path": model},
        least_time_to_read=lambda chars: worker_read,
    )
    monkeypatch.setattr(mlx_client, "get_mlx_client", lambda: client)
    monkeypatch.setattr(chat_format, "thinking_enabled_for_generation", lambda *a, **k: True)
    monkeypatch.setattr(thinking_reserve, "reserve_tokens", lambda name: private)
    monkeypatch.setattr(thinking_reserve, "seconds_to_read", lambda chars: chars / 100)

    def decode(tokens, name):
        assert name == model
        assert tokens == 1000 + private
        return tokens / decode_rate if decode_rate else 0.0

    monkeypatch.setattr(thinking_reserve, "seconds_to_decode", decode)
    assert UnitaryResponsePhase._timeout_for_request(
        is_user_facing=True, model_tier="primary", deep_handoff=False,
        messages=[{"role": "user", "content": "x" * 16000}],
        decode_max_tokens=1000,
    ) == expected


@pytest.mark.parametrize("foreground,tier,expected", [(False, "primary", 15.0), (True, "secondary", 210.0)])
def test_other_lanes_do_not_borrow_resident_measurements(monkeypatch, foreground, tier, expected):
    from core.brain.llm import mlx_client
    from core.phases.response_generation_unitary import UnitaryResponsePhase

    def unexpected():
        pytest.fail("another lane must not read the resident's rate")

    monkeypatch.setattr(mlx_client, "get_mlx_client", unexpected)
    assert UnitaryResponsePhase._timeout_for_request(
        is_user_facing=foreground, model_tier=tier, deep_handoff=False,
        messages=[{"role": "user", "content": "hello"}],
    ) == expected
