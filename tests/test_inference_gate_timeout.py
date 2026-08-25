from __future__ import annotations

import asyncio

from core.brain.inference_gate import InferenceGate
from core.utils.deadlines import get_deadline


class HangingClient:
    def __init__(self) -> None:
        self.abort_reasons: list[str] = []

    async def generate_text_async(self, **kwargs):
        await asyncio.sleep(5)
        return "late text"

    def force_abort_active_generation(self, reason: str = "hard_generation_deadline") -> bool:
        self.abort_reasons.append(reason)
        return True


def test_generate_with_client_aborts_when_client_ignores_deadline() -> None:
    async def run() -> None:
        client = HangingClient()
        gate = InferenceGate()

        text = await gate._generate_with_client(
            client,
            "say hello",
            "You are Aura.",
            [],
            get_deadline(0.05),
            "Cortex",
            foreground_request=True,
            origin="user",
        )

        assert text is None
        assert client.abort_reasons
        assert client.abort_reasons[0].startswith("inference_gate_generation_timeout:Cortex:")

    asyncio.run(run())


def test_foreground_retry_schedule_only_retries_fast_failures() -> None:
    assert InferenceGate._foreground_retry_schedule(
        primary_attempt_elapsed=10.0,
        primary_timeout=150.0,
    ) == (2.0,)
    assert InferenceGate._foreground_retry_schedule(
        primary_attempt_elapsed=61.0,
        primary_timeout=150.0,
    ) == ()


def test_exhausted_primary_owner_suppresses_downstream_model_retry() -> None:
    gate = InferenceGate.__new__(InferenceGate)
    fields: dict[str, object] = {}
    gate._annotate_last_generation_metadata = fields.update  # type: ignore[method-assign]

    gate._publish_exhausted_primary_owner(
        primary_attempt_elapsed=91.125,
        same_lane_retry_count=0,
    )

    assert fields == {
        "model_retry_suppressed": True,
        "generation_failure_class": "primary_no_text_after_long_attempt",
        "primary_attempt_elapsed_s": 91.125,
        "same_lane_retry_count": 0,
    }


def test_think_preserves_desktop_cognitive_engine_contract() -> None:
    async def run() -> None:
        gate = InferenceGate()
        captured: dict[str, object] = {}

        async def fake_generate(prompt, context=None, timeout=None):
            captured["prompt"] = prompt
            captured["context"] = dict(context or {})
            captured["timeout"] = timeout
            return "ready"

        gate.generate = fake_generate  # type: ignore[method-assign]
        gate._post_inference_update = lambda _text: None  # type: ignore[method-assign]

        result = await gate.think(
            "hello",
            origin="desktop_ui",
            cognitive_engine_required=True,
            desktop_cognitive_engine_required=True,
        )

        assert result == "ready"
        assert captured["context"]["cognitive_engine_required"] is True
        assert captured["context"]["desktop_cognitive_engine_required"] is True

    asyncio.run(run())


# ── Foreground warmup admission control (doom-loop breaker) ───────────────


def _bare_gate() -> InferenceGate:
    return InferenceGate.__new__(InferenceGate)


def test_cold_boot_warmup_keeps_the_generous_budget():
    """First-ever load: the user expects the one-time ~150s cold load."""
    gate = _bare_gate()
    assert gate._foreground_warmup_timeout({"last_ready_at": 0.0}, 206.0) == 206.0
    assert gate._foreground_warmup_timeout({}, 90.0) == 180.0  # floor at 180


def test_recovery_warmup_is_capped_short():
    """Cortex was ready and died (stall-kill): the turn must not wait 180s —
    observed live (Jul 7 soak) as every turn crawling to 200s+ during a single
    warm window. Short cap → fallback answers now, shielded warmup finishes
    in the background for the next turn."""
    gate = _bare_gate()
    capped = gate._foreground_warmup_timeout({"last_ready_at": 1783435462.4}, 206.0)
    assert capped == 15.0


def test_recovery_warmup_cap_is_operator_tunable(monkeypatch):
    monkeypatch.setenv("AURA_FOREGROUND_RECOVERY_WARMUP_CAP_S", "180")
    gate = _bare_gate()
    assert gate._foreground_warmup_timeout({"last_ready_at": 5.0}, 90.0) == 180.0
