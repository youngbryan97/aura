import asyncio

import pytest

from core.consciousness import liquid_substrate
from core.consciousness.liquid_substrate import LiquidSubstrate, SubstrateConfig


def _substrate(tmp_path) -> LiquidSubstrate:
    return LiquidSubstrate(
        SubstrateConfig(neuron_count=4, state_file=tmp_path / "substrate_state.npy")
    )


def test_liquid_substrate_degradation_receipts_are_rate_limited(monkeypatch, tmp_path):
    recorded: list[dict] = []

    def _record(subsystem, error, **kwargs):
        recorded.append(
            {
                "subsystem": subsystem,
                "error_type": type(error).__name__,
                "action": kwargs.get("action"),
                "extra": kwargs.get("extra") or {},
            }
        )

    monkeypatch.setattr(liquid_substrate, "record_degradation", _record)

    substrate = _substrate(tmp_path)
    substrate._record_operational_degradation(
        RuntimeError("registry offline"),
        stage="registry_sync",
        action="continued substrate loop while skipping registry export for this tick",
        cooldown_s=60.0,
    )
    substrate._record_operational_degradation(
        RuntimeError("registry still offline"),
        stage="registry_sync",
        action="continued substrate loop while skipping registry export for this tick",
        cooldown_s=60.0,
    )

    assert len(recorded) == 1

    substrate._degradation_last_reported["registry_sync:RuntimeError"] = 0.0
    substrate._record_operational_degradation(
        RuntimeError("registry offline again"),
        stage="registry_sync",
        action="continued substrate loop while skipping registry export for this tick",
        cooldown_s=60.0,
    )

    assert len(recorded) == 2
    assert recorded[-1]["extra"]["suppressed_repeats_since_last_receipt"] == 1


@pytest.mark.asyncio
async def test_liquid_substrate_loop_failure_uses_adaptive_backoff(monkeypatch, tmp_path):
    recorded: list[dict] = []
    sleep_delays: list[float] = []

    def _record(subsystem, error, **kwargs):
        recorded.append(
            {
                "subsystem": subsystem,
                "error_type": type(error).__name__,
                "action": kwargs.get("action"),
                "severity": kwargs.get("severity"),
                "extra": kwargs.get("extra") or {},
            }
        )

    async def _failing_to_thread(*_args, **_kwargs):
        raise RuntimeError("integration offline")

    async def _stop_after_backoff(delay):
        sleep_delays.append(delay)
        raise asyncio.CancelledError

    monkeypatch.setattr(liquid_substrate, "record_degradation", _record)
    monkeypatch.setattr(liquid_substrate.asyncio, "to_thread", _failing_to_thread)
    monkeypatch.setattr(liquid_substrate.asyncio, "sleep", _stop_after_backoff)
    monkeypatch.setattr("core.container.ServiceContainer.get", lambda _name, default=None: default)

    substrate = _substrate(tmp_path)
    substrate.pulse = lambda success=True: None
    substrate.running = True

    await substrate._run_loop()

    assert substrate.running is False
    assert substrate._loop_failure_streak == 1
    assert sleep_delays == [1.0]
    assert recorded[-1]["action"] == (
        "kept substrate loop alive with adaptive backoff after tick failure"
    )
    assert recorded[-1]["extra"]["backoff_s"] == 1.0
