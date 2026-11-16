"""CP126 allostasis: state durability, readiness truth, and measurement decay.

Pins the second allostasis cluster:

* ``49ee0229`` — ledger events cleared before the write succeeded, so a
  failed persist silently destroyed issued/resolved/regime/tier history.
* ``a415efdc`` — persisted allostatic load and tier were written but never
  read back, so every restart reported a body with no chronic strain.
* ``baa391d2`` — persisted forecasts could create calibration series for
  vitals the engine does not measure.
* ``e54aeade`` — readiness meant only that the kill switch was off, so a
  dead pulse loop still reported a healthy predictive organ.
* ``49c0547c`` — a vital that stopped reporting kept its last value forever,
  holding a red-line breach authoritative long after it was observable.
"""
from __future__ import annotations

import asyncio
import math

import pytest

from core.autonomic import allostasis
from core.autonomic.allostasis import AllostasisEngine, AllostasisTier


def _engine(tmp_path, now_fn=None, **kwargs) -> AllostasisEngine:
    return AllostasisEngine(
        data_dir=tmp_path,
        now_fn=now_fn or (lambda: 1_000.0),
        **kwargs,
    )


class TestLedgerEventsSurviveAFailedWrite:
    @pytest.mark.parametrize("operation", ["ensure_directory_async", "append_text_async"])
    def test_failed_persist_requeues_events(self, tmp_path, monkeypatch, operation):
        from core.runtime.file_write_gateway import get_file_write_gateway

        engine = _engine(tmp_path)
        events = [{"kind": "issued", "forecast_id": "fc-1"}]
        engine._pending_events = []
        gateway = get_file_write_gateway()
        method = getattr(gateway, operation)

        async def broken_write(*args, **kwargs):
            raise OSError("fixture write refused")

        monkeypatch.setattr(gateway, operation, broken_write)
        if not asyncio.run(engine._persist(events, save_state=False)):
            engine._requeue_unpersisted(events)
        assert engine._pending_events == events
        monkeypatch.setattr(gateway, operation, method)
        assert asyncio.run(engine._persist(engine._pending_events, save_state=False))
        assert len(engine._events_path.read_text().splitlines()) == 1

    def test_requeue_puts_events_back_in_order(self, tmp_path):
        engine = _engine(tmp_path)
        engine._pending_events = [{"kind": "tier_change"}]
        engine._requeue_unpersisted([{"kind": "issued"}])
        # The failed batch is older than whatever accumulated since.
        assert [e["kind"] for e in engine._pending_events] == ["issued", "tier_change"]

    def test_backlog_is_bounded_and_the_gap_is_recorded(self, tmp_path, monkeypatch):
        engine = _engine(tmp_path)
        recorded: list[str] = []
        monkeypatch.setattr(
            allostasis, "record_degradation",
            lambda *a, **k: recorded.append(k.get("action", "")),
        )
        cap = allostasis._MAX_PENDING_EVENTS
        engine._pending_events = []
        engine._requeue_unpersisted([{"n": i} for i in range(cap + 25)])
        assert len(engine._pending_events) == cap
        # The OLDEST are dropped; the newest survive.
        assert engine._pending_events[-1]["n"] == cap + 24
        assert any("gap" in action for action in recorded)

    def test_persist_reports_success(self, tmp_path):
        engine = _engine(tmp_path)
        # Nothing to do is a successful no-op, not a failure that requeues.
        assert asyncio.run(engine._persist([], save_state=False)) is True

    def test_a_failed_snapshot_does_not_requeue_events_already_appended(self, tmp_path, monkeypatch):
        import json

        from core.runtime.file_write_gateway import get_file_write_gateway

        engine = _engine(tmp_path)
        events = [{"kind": "issued", "forecast_id": "fc-1"}]
        gateway = get_file_write_gateway()
        write_json = gateway.write_json_async
        faults = []
        monkeypatch.setattr(allostasis, "record_degradation", lambda *args, **kwargs: faults.append(kwargs["action"]))

        async def broken_snapshot(*args, **kwargs):
            raise OSError("snapshot temporarily unwritable")

        async def scenario():
            monkeypatch.setattr(gateway, "write_json_async", broken_snapshot)
            if not await engine._persist(events, save_state=True):
                engine._requeue_unpersisted(events)
            monkeypatch.setattr(gateway, "write_json_async", write_json)
            retried, engine._pending_events = engine._pending_events, []
            assert await engine._persist(retried, save_state=True)

        asyncio.run(scenario())
        assert [json.loads(line) for line in engine._events_path.read_text().splitlines()] == events
        assert faults and engine._state_path.is_file()

    def test_each_persistence_operation_has_a_fresh_grant(self, tmp_path, monkeypatch):
        from core.governance_context import get_active_governance
        from core.runtime.file_write_gateway import get_file_write_gateway

        engine = _engine(tmp_path)
        gateway = get_file_write_gateway()
        grants = []
        monkeypatch.setenv("AURA_GOVERNANCE_MODE", "enforce")

        for method in ("ensure_directory_async", "append_text_async", "write_json_async"):
            original = getattr(gateway, method)

            def wrap(operation):
                async def run(*args, **kwargs):
                    token = get_active_governance()
                    assert token is not None and token.domain == "file_write"
                    grants.append(token.lease_id)
                    result = await operation(*args, **kwargs)
                    token.mono_timestamp -= token.ttl + 1
                    return result
                return run

            monkeypatch.setattr(gateway, method, wrap(original))

        assert asyncio.run(engine._persist([{"kind": "issued", "forecast_id": "fc-2"}], save_state=True))
        assert len(set(grants)) == 3 and engine._state_path.is_file()


class TestPersistedStrainIsRestored:
    def test_load_and_tier_round_trip(self, tmp_path):
        engine = _engine(tmp_path)
        vital = next(iter(engine._specs))
        engine._load_raw[vital] = 900.0
        engine._tier = AllostasisTier.CONSERVING
        payload = engine._state_payload()
        assert payload["allostatic_load"][vital] == pytest.approx(900.0)

        restored = _engine(tmp_path)
        restored._state_path.write_text(
            __import__("json").dumps({"payload": payload}), encoding="utf-8",
        )
        restored._load_raw = {k: 0.0 for k in restored._specs}
        restored._restore_persisted_state()
        assert restored._load_raw[vital] > 0.0

    def test_downtime_decays_the_restored_strain(self, tmp_path):
        import json

        engine = _engine(tmp_path)
        vital = next(iter(engine._specs))
        payload = {
            "calibration": {},
            "open_forecasts": [],
            "allostatic_load": {vital: 1000.0},
            "tier": "conserving",
            "regime_events_total": 3,
            "saved_at": 1_000.0,
        }
        # Restart an hour later: strain the body did not carry while dead.
        later = _engine(tmp_path, now_fn=lambda: 1_000.0 + 3600.0)
        later._state_path.write_text(json.dumps({"payload": payload}), encoding="utf-8")
        later._load_raw = {k: 0.0 for k in later._specs}
        later._restore_persisted_state()
        assert later._load_raw[vital] < 1000.0
        assert later._regime_events_total == 3

    def test_tier_cannot_be_restored_above_what_load_supports(self, tmp_path):
        import json

        payload = {
            "calibration": {},
            "open_forecasts": [],
            "allostatic_load": {},
            # Claims PROTECTING with no strain behind it.
            "tier": "protecting",
            "saved_at": 1_000.0,
        }
        engine = _engine(tmp_path)
        engine._state_path.write_text(json.dumps({"payload": payload}), encoding="utf-8")
        engine._tier = AllostasisTier.SETTLED
        engine._load_raw = {k: 0.0 for k in engine._specs}
        engine._restore_persisted_state()
        assert engine._tier == AllostasisTier.SETTLED

    def test_unknown_vital_in_state_cannot_create_calibration(self, tmp_path, monkeypatch):
        import json

        recorded: list[str] = []
        monkeypatch.setattr(
            allostasis, "record_degradation",
            lambda *a, **k: recorded.append(k.get("action", "")),
        )
        payload = {
            "calibration": {},
            "open_forecasts": [
                {"vital": "not_a_real_vital", "forecast_id": "fc-x"},
            ],
            "saved_at": 1_000.0,
        }
        engine = _engine(tmp_path)
        engine._state_path.write_text(json.dumps({"payload": payload}), encoding="utf-8")
        engine._calibration = {}
        engine._pending_events = []
        engine._restore_persisted_state()
        assert "not_a_real_vital" not in engine._calibration
        assert any("stale forecast" in action for action in recorded)

    def test_known_vital_in_state_still_supersedes(self, tmp_path):
        import json

        engine = _engine(tmp_path)
        vital = next(iter(engine._specs))
        payload = {
            "calibration": {},
            "open_forecasts": [{"vital": vital, "forecast_id": "fc-ok"}],
            "saved_at": 1_000.0,
        }
        engine._state_path.write_text(json.dumps({"payload": payload}), encoding="utf-8")
        engine._calibration = {}
        engine._pending_events = []
        engine._restore_persisted_state()
        assert engine._calibration[vital].superseded == 1


class TestReadinessIsEvidence:
    def test_boot_grace_reports_ready(self, tmp_path):
        engine = _engine(tmp_path)
        state = engine.readiness()
        assert state["ready"] is True
        assert state["booting"] is True
        assert state["samples"] == 0

    def test_a_dead_feed_is_not_ready(self, tmp_path):
        clock = {"t": 1_000.0}
        engine = _engine(tmp_path, now_fn=lambda: clock["t"])
        engine._last_ingest_at = 1_000.0
        # Three missed 60 s pulses.
        clock["t"] = 1_000.0 + allostasis._INGEST_STALE_AFTER_S + 1.0
        state = engine.readiness()
        assert state["ready"] is False
        assert state["booting"] is False
        assert state["last_ingest_age_s"] > allostasis._INGEST_STALE_AFTER_S

    def test_a_fed_engine_is_ready(self, tmp_path):
        clock = {"t": 1_000.0}
        engine = _engine(tmp_path, now_fn=lambda: clock["t"])
        engine._last_ingest_at = 1_000.0
        clock["t"] = 1_030.0
        assert engine.is_ready() is True

    def test_kill_switch_still_wins(self, tmp_path):
        engine = _engine(tmp_path)
        engine._disabled = True
        assert engine.is_ready() is False
        assert engine.readiness()["enabled"] is False


class TestStaleMeasurementsExpire:
    def test_a_stale_red_line_stops_being_a_breach(self, tmp_path, monkeypatch):
        recorded: list[str] = []
        monkeypatch.setattr(
            allostasis, "record_degradation",
            lambda *a, **k: recorded.append(k.get("action", "")),
        )
        clock = {"t": 1_000.0}
        engine = _engine(tmp_path, now_fn=lambda: clock["t"])
        vital = next(iter(engine._specs))
        spec = engine._specs[vital]
        engine._series[vital].append((1_000.0, spec.red + 1.0))

        assert engine._current_breach(1_000.0) == vital
        clock["t"] = 1_000.0 + allostasis._INGEST_STALE_AFTER_S + 1.0
        assert engine._current_breach(clock["t"]) is None
        assert any("stale breach" in action for action in recorded)

    def test_the_stale_report_does_not_storm(self, tmp_path, monkeypatch):
        recorded: list[str] = []
        monkeypatch.setattr(
            allostasis, "record_degradation",
            lambda *a, **k: recorded.append(k.get("action", "")),
        )
        clock = {"t": 1_000.0}
        engine = _engine(tmp_path, now_fn=lambda: clock["t"])
        vital = next(iter(engine._specs))
        spec = engine._specs[vital]
        engine._series[vital].append((1_000.0, spec.red + 1.0))
        clock["t"] = 1_000.0 + allostasis._INGEST_STALE_AFTER_S + 1.0
        for _ in range(20):
            engine._current_breach(clock["t"])
        assert len([a for a in recorded if "stale breach" in a]) == 1

    def test_a_returning_vital_can_breach_again(self, tmp_path, monkeypatch):
        monkeypatch.setattr(allostasis, "record_degradation", lambda *a, **k: None)
        clock = {"t": 1_000.0}
        engine = _engine(tmp_path, now_fn=lambda: clock["t"])
        vital = next(iter(engine._specs))
        spec = engine._specs[vital]
        engine._series[vital].append((1_000.0, spec.red + 1.0))
        clock["t"] = 1_000.0 + allostasis._INGEST_STALE_AFTER_S + 1.0
        assert engine._current_breach(clock["t"]) is None
        # The sensor comes back, still red.
        engine.ingest({vital: spec.red + 1.0}, at=clock["t"])
        assert vital not in engine._stale_breach_reported
        assert engine._current_breach(clock["t"]) == vital

    def test_freshness_requires_a_sample(self, tmp_path):
        engine = _engine(tmp_path)
        vital = next(iter(engine._specs))
        assert engine._vital_is_fresh(vital, 1_000.0) is False


class TestRestoredLoadIsFinite:
    def test_nonfinite_persisted_load_is_ignored(self, tmp_path):
        import json

        engine = _engine(tmp_path)
        vital = next(iter(engine._specs))
        payload = {
            "calibration": {},
            "open_forecasts": [],
            "allostatic_load": {vital: "nan"},
            "saved_at": 1_000.0,
        }
        engine._state_path.write_text(json.dumps({"payload": payload}), encoding="utf-8")
        engine._load_raw = {k: 0.0 for k in engine._specs}
        engine._restore_persisted_state()
        assert math.isfinite(engine._load_raw[vital])
        assert engine._load_raw[vital] == 0.0

    def test_negative_persisted_load_is_ignored(self, tmp_path):
        import json

        engine = _engine(tmp_path)
        vital = next(iter(engine._specs))
        payload = {
            "calibration": {},
            "open_forecasts": [],
            "allostatic_load": {vital: -5.0},
            "saved_at": 1_000.0,
        }
        engine._state_path.write_text(json.dumps({"payload": payload}), encoding="utf-8")
        engine._load_raw = {k: 0.0 for k in engine._specs}
        engine._restore_persisted_state()
        assert engine._load_raw[vital] == 0.0
