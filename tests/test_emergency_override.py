"""CP126 4e0cae60 + 1229b07f: emergency overrides are decisions, not settings.

Two MLX memory guards could be switched off by a bare environment value with
no expiry, no upper bound, and no durable record that the risk had been
accepted. A flag exported once for a recovery session disabled the guard for
the life of the process — and, in a launch profile, for the life of the
deployment.
"""
from __future__ import annotations

import inspect

import pytest

from core.brain.llm import emergency_override as eo

# The waiting loop was lifted out of mlx_client (2026-09-20); the family
# reader sees the parent and its lifted siblings as one text.
from tests.source_contract import family_text


@pytest.fixture(autouse=True)
def _clean_state():
    eo.reset_overrides_for_test()
    yield
    eo.reset_overrides_for_test()


FLAG = "AURA_TEST_EMERGENCY_OVERRIDE"


class TestUnsetFlagIsNotAnOverride:
    def test_absent_flag_is_inactive(self, monkeypatch):
        monkeypatch.delenv(FLAG, raising=False)
        decision = eo.consume_override(FLAG, guard="g")
        assert decision.active is False
        assert decision.reason == "not_set"

    def test_falsey_values_are_inactive(self, monkeypatch):
        for value in ("0", "false", "no", "off", "", "  "):
            monkeypatch.setenv(FLAG, value)
            assert eo.consume_override(FLAG, guard="g").active is False

    def test_truthy_values_are_active(self, monkeypatch):
        for value in ("1", "true", "YES", "On"):
            eo.reset_overrides_for_test()
            monkeypatch.setenv(FLAG, value)
            assert eo.consume_override(FLAG, guard="g").active is True


class TestOverridesExpire:
    def test_active_inside_the_window(self, monkeypatch):
        monkeypatch.setenv(FLAG, "1")
        decision = eo.consume_override(FLAG, guard="g", now=1_000.0)
        assert decision.active is True
        assert decision.expires_at_unix == pytest.approx(1_000.0 + eo.DEFAULT_WINDOW_S)

    def test_expired_after_the_window(self, monkeypatch):
        monkeypatch.setenv(FLAG, "1")
        eo.consume_override(FLAG, guard="g", now=1_000.0)
        later = 1_000.0 + eo.DEFAULT_WINDOW_S + 1.0
        decision = eo.consume_override(FLAG, guard="g", now=later)
        assert decision.active is False
        assert decision.reason == "expired"

    def test_the_window_cannot_be_extended_past_the_ceiling(self, monkeypatch):
        monkeypatch.setenv(FLAG, "1")
        # A caller asking for a week still gets the ceiling.
        decision = eo.consume_override(
            FLAG, guard="g", window_s=7 * 24 * 3600.0, now=1_000.0,
        )
        assert decision.expires_at_unix == pytest.approx(1_000.0 + eo.DEFAULT_WINDOW_S)

    def test_the_window_can_be_shortened(self, monkeypatch):
        monkeypatch.setenv(FLAG, "1")
        decision = eo.consume_override(FLAG, guard="g", window_s=60.0, now=1_000.0)
        assert decision.expires_at_unix == pytest.approx(1_060.0)

    def test_the_clock_starts_at_first_observation(self, monkeypatch):
        monkeypatch.setenv(FLAG, "1")
        eo.consume_override(FLAG, guard="g", now=1_000.0)
        second = eo.consume_override(FLAG, guard="g", now=1_100.0)
        assert second.first_seen_unix == pytest.approx(1_000.0)


class TestOverridesAreUseBounded:
    def test_budget_is_spent_per_consumption(self, monkeypatch):
        monkeypatch.setenv(FLAG, "1")
        first = eo.consume_override(FLAG, guard="g", now=1_000.0)
        second = eo.consume_override(FLAG, guard="g", now=1_001.0)
        assert first.uses_remaining == eo.DEFAULT_MAX_USES - 1
        assert second.uses_remaining == eo.DEFAULT_MAX_USES - 2

    def test_exhausted_budget_re_arms_the_guard(self, monkeypatch):
        monkeypatch.setenv(FLAG, "1")
        for i in range(eo.DEFAULT_MAX_USES):
            assert eo.consume_override(FLAG, guard="g", now=1_000.0 + i).active is True
        spent = eo.consume_override(FLAG, guard="g", now=1_100.0)
        assert spent.active is False
        assert spent.reason == "exhausted"

    def test_the_budget_cannot_be_raised_past_the_ceiling(self, monkeypatch):
        monkeypatch.setenv(FLAG, "1")
        decision = eo.consume_override(FLAG, guard="g", max_uses=10_000, now=1_000.0)
        assert decision.uses_remaining == eo.DEFAULT_MAX_USES - 1


class TestUnsettingIsANewDecision:
    def test_unsetting_forgets_an_exhausted_budget(self, monkeypatch):
        monkeypatch.setenv(FLAG, "1")
        for i in range(eo.DEFAULT_MAX_USES):
            eo.consume_override(FLAG, guard="g", now=1_000.0 + i)
        assert eo.consume_override(FLAG, guard="g", now=1_100.0).active is False

        # Operator unsets the flag: the decision is withdrawn.
        monkeypatch.delenv(FLAG, raising=False)
        assert eo.consume_override(FLAG, guard="g", now=1_200.0).reason == "not_set"

        # Re-setting is a NEW decision with a fresh window.
        monkeypatch.setenv(FLAG, "1")
        fresh = eo.consume_override(FLAG, guard="g", now=1_300.0)
        assert fresh.active is True
        assert fresh.first_seen_unix == pytest.approx(1_300.0)


class TestRiskIsReceipted:
    def test_an_active_override_emits_a_receipt(self, monkeypatch):
        monkeypatch.setenv(FLAG, "1")
        captured: list[object] = []

        class _Store:
            def emit(self, receipt):
                captured.append(receipt)
                receipt.receipt_id = "gov-test"
                return receipt

        monkeypatch.setattr(
            "core.runtime.receipts.get_receipt_store", lambda *a, **k: _Store(),
        )
        decision = eo.consume_override(
            FLAG, guard="critical_memory_generation_refusal",
            observed="rss 61GB", now=1_000.0,
        )
        assert decision.receipt_id == "gov-test"
        assert len(captured) == 1
        body = captured[0].to_dict()
        assert body["metadata"]["flag"] == FLAG
        assert body["metadata"]["observed"] == "rss 61GB"
        assert body["metadata"]["use_budget"] == eo.DEFAULT_MAX_USES
        assert body["approved"] is True

    def test_a_refused_override_emits_nothing(self, monkeypatch):
        monkeypatch.delenv(FLAG, raising=False)
        calls: list[int] = []
        monkeypatch.setattr(
            "core.runtime.receipts.get_receipt_store",
            lambda *a, **k: calls.append(1),
        )
        eo.consume_override(FLAG, guard="g")
        assert calls == []

    def test_a_broken_receipt_store_cannot_break_recovery(self, monkeypatch):
        monkeypatch.setenv(FLAG, "1")

        def _boom(*_a, **_k):
            raise RuntimeError("receipt store down")

        monkeypatch.setattr("core.runtime.receipts.get_receipt_store", _boom)
        decision = eo.consume_override(FLAG, guard="g", now=1_000.0)
        # The override still works; only its receipt is missing.
        assert decision.active is True
        assert decision.receipt_id is None


class TestStatusIsNonConsuming:
    def test_status_does_not_spend_the_budget(self, monkeypatch):
        monkeypatch.setenv(FLAG, "1")
        eo.consume_override(FLAG, guard="g", now=1_000.0)
        before = eo.override_status(FLAG, now=1_001.0)["uses"]
        eo.override_status(FLAG, now=1_002.0)
        assert eo.override_status(FLAG, now=1_003.0)["uses"] == before

    def test_unseen_flag_reports_unseen(self, monkeypatch):
        monkeypatch.delenv(FLAG, raising=False)
        assert eo.override_status(FLAG)["seen"] is False


class TestClientGuardsUseTheGovernedPath:
    def test_spawn_admission_consumes_an_override(self):
        from core.brain.llm import mlx_client

        source = inspect.getsource(mlx_client._memory_pressure_blocks_worker_spawn)
        assert "consume_override(" in source
        assert "AURA_FORCE_CORTEX_WARMUP_UNDER_PRESSURE" in source
        # The raw environment read must be gone.
        assert "os.environ.get(\"AURA_FORCE_CORTEX_WARMUP_UNDER_PRESSURE\"" not in source

    def test_generation_refusal_consumes_an_override(self):
        from core.brain.llm import mlx_client

        source = family_text(mlx_client)
        assert "guard=\"critical_memory_generation_refusal\"" in source
        assert (
            "os.environ.get(\"AURA_MLX_ALLOW_CRITICAL_MEMORY_GENERATION\", \"\")"
            not in source
        )

    def test_the_budget_is_only_spent_when_the_guard_would_fire(self):
        from core.brain.llm import mlx_client

        source = family_text(mlx_client)
        # Consumption is nested under the condition that the refusal applies.
        assert "override_applies = (" in source
        assert "if override_applies:" in source

    def test_the_live_abort_guard_is_governed_too(self):
        from core.brain.llm import mlx_client

        source = family_text(mlx_client)
        assert 'guard="live_memory_pressure_abort"' in source


class TestAbortFailureIsNotAProbeFailure:
    """``0f13d1ee`` — observation and enforcement shared one try block.

    A failure while ABORTING (queue cleanup, future cancellation) was logged
    as "memory probe unavailable" and the wait loop simply continued, with
    critical pressure observed and lifecycle state half-cleared.
    """

    def _source(self) -> str:
        from core.brain.llm import mlx_client

        return family_text(mlx_client)

    def test_the_probe_has_its_own_try_block(self):
        source = self._source()
        assert "memory_snapshot = None" in source
        assert "abort decision could not be made" in source

    def test_a_blind_probe_on_a_heavy_lane_is_recorded(self):
        source = self._source()
        assert "live memory-pressure probe unavailable during heavy" in source

    def test_a_failed_abort_is_terminal_and_recorded(self):
        source = self._source()
        assert "generation_abort_failed_memory_pressure" in source
        assert "could not be proven clean" in source
