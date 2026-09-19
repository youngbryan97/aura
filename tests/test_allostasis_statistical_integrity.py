"""CP126 allostasis: statistics, strain accounting, identity, and honesty.

Pins the third allostasis cluster:

* ``e10d1022`` — Mann-Kendall p-values were taken at face value on dense,
  strongly autocorrelated telemetry and judged per-vital at alpha, with no
  control over the family of tests run every pulse.
* ``fb2e1078`` — a forecast was scored from the instantaneous value only, so
  a threshold crossed and recovered between pulses became a false alarm.
* ``af4f3b3a`` — one global dt was credited at the newest value, attributing
  unobserved history (host sleep, a stalled pulse) to a single endpoint.
* ``d068e431`` / ``5424693e`` — the no-data state was narrated as universal
  stability, in first-person bodily language with no scope or freshness.
* ``850c25f1`` / ``d62b5b97`` — the getter and the container could hold
  different engines, and the test reset left the old one registered.
* ``ae9bf39f`` — ledger IDs carried 32-40 random bits with no issuer.
* ``57722132`` — the threat channel had no all-clear.
"""
from __future__ import annotations

import inspect
import math

import pytest

from core.autonomic import allostasis
from core.autonomic.allostasis import (
    AllostasisEngine,
    AllostasisTier,
    mann_kendall,
)


def _engine(tmp_path, now_fn=None, **kwargs) -> AllostasisEngine:
    return AllostasisEngine(
        data_dir=tmp_path, now_fn=now_fn or (lambda: 1_000.0), **kwargs,
    )


class TestAutocorrelationCorrection:
    def test_correction_widens_the_variance_on_a_dependent_series(self):
        ramp = [float(i) for i in range(20)]
        classical = mann_kendall(ramp, correct_autocorrelation=False)
        corrected = mann_kendall(ramp)
        # Same S, honestly larger variance, hence a larger p-value.
        assert corrected.s == classical.s
        assert corrected.var_s > classical.var_s
        assert corrected.p_value > classical.p_value

    def test_a_real_trend_still_clears_the_bar(self):
        ramp = [float(i) for i in range(20)]
        assert mann_kendall(ramp).p_value < 0.05

    def test_short_series_are_left_alone(self):
        short = [1.0, 2.0, 3.0]
        assert allostasis._hamed_rao_correction(short) == 1.0

    def test_the_factor_cannot_manufacture_certainty(self):
        # Alternating series has strong negative lag-1 dependence.
        alternating = [float(i % 2) for i in range(24)]
        factor = allostasis._hamed_rao_correction(alternating)
        assert factor >= 0.1

    def test_flat_series_has_no_defined_autocorrelation(self):
        assert allostasis._rank_autocorrelations([5.0] * 8) == []


class TestFalseDiscoveryControl:
    def test_single_test_reduces_to_alpha(self, tmp_path):
        engine = _engine(tmp_path)
        mk = mann_kendall([float(i) for i in range(12)])
        estimate = allostasis.sen_slope(
            [float(i) for i in range(12)], [float(i) for i in range(12)],
        )
        key = next(iter(engine._specs))
        spec = engine._specs[key]
        trends = {key: (mk, estimate, spec.setpoint)}
        # setpoint sits below both thresholds, so both tests are in the family.
        assert engine._admissible_p_value(trends) <= engine._alpha

    def test_empty_family_admits_alpha(self, tmp_path):
        engine = _engine(tmp_path)
        assert engine._admissible_p_value({}) == engine._alpha

    def test_the_trend_pass_is_computed_once(self):
        source = inspect.getsource(AllostasisEngine._refresh_forecasts)
        # The issuance loop reads the shared pass; it must not recompute.
        assert "self._trend_pass(now)" in source
        assert "mann_kendall(" not in source


class TestBetweenSampleCrossings:
    def test_peak_since_reads_the_history(self, tmp_path):
        engine = _engine(tmp_path)
        vital = next(iter(engine._specs))
        engine._series[vital].extend([(100.0, 5.0), (160.0, 42.0), (220.0, 6.0)])
        # The spike between pulses is in the record even though it recovered.
        assert engine._peak_since(vital, 100.0) == 42.0

    def test_peak_since_ignores_samples_before_the_forecast(self, tmp_path):
        engine = _engine(tmp_path)
        vital = next(iter(engine._specs))
        engine._series[vital].extend([(100.0, 99.0), (160.0, 5.0), (220.0, 6.0)])
        assert engine._peak_since(vital, 150.0) == 6.0

    def test_no_samples_in_window_is_nan(self, tmp_path):
        engine = _engine(tmp_path)
        vital = next(iter(engine._specs))
        assert math.isnan(engine._peak_since(vital, 1_000.0))

    def test_resolution_scores_the_high_water_mark(self):
        source = inspect.getsource(AllostasisEngine._resolve_due_forecasts)
        assert "_peak_since" in source
        assert "observed_peak >= fc.threshold_value" in source


class TestStrainAccounting:
    def test_a_long_gap_is_not_credited_to_one_sample(self, tmp_path):
        engine = _engine(tmp_path)
        vital = next(iter(engine._specs))
        spec = engine._specs[vital]
        engine._load_raw[vital] = 0.0
        # A whole hour attributed to one red reading.
        engine._accrue_load(vital, spec, spec.red, 3600.0)
        bounded = engine._load_raw[vital]
        assert bounded <= allostasis._MAX_ATTRIBUTABLE_GAP_S

    def test_decay_still_spans_the_whole_gap(self, tmp_path):
        engine = _engine(tmp_path)
        vital = next(iter(engine._specs))
        spec = engine._specs[vital]
        engine._load_raw[vital] = 1_000.0
        engine._accrue_load(vital, spec, spec.setpoint, 3600.0)
        # Strain fades in real time whether or not anyone was watching.
        assert engine._load_raw[vital] < 1_000.0

    def test_the_interval_is_integrated_not_extrapolated(self, tmp_path):
        engine = _engine(tmp_path)
        vital = next(iter(engine._specs))
        spec = engine._specs[vital]

        engine._load_raw[vital] = 0.0
        engine._accrue_load(vital, spec, spec.red, 60.0, previous=spec.setpoint)
        trapezoid = engine._load_raw[vital]

        engine._load_raw[vital] = 0.0
        engine._accrue_load(vital, spec, spec.red, 60.0)
        endpoint = engine._load_raw[vital]

        # Rising from setpoint to red over the interval is half the strain of
        # having sat at red for all of it.
        assert trapezoid < endpoint
        assert trapezoid == pytest.approx(endpoint / 2.0, rel=1e-6)


class TestNarrativeHonesty:
    def test_no_samples_is_not_stability(self, tmp_path):
        engine = _engine(tmp_path)
        text = engine.narrative()
        assert "stable" not in text.lower()
        assert "nothing is being" in text

    def test_the_claim_names_its_scope_and_freshness(self, tmp_path):
        clock = {"t": 1_000.0}
        engine = _engine(tmp_path, now_fn=lambda: clock["t"])
        vital = next(iter(engine._specs))
        spec = engine._specs[vital]
        for offset in range(0, 600, 60):
            engine.ingest({vital: spec.setpoint}, at=1_000.0 + offset)
        clock["t"] = 1_000.0 + 540.0
        text = engine.narrative()
        assert "this process" in text or "I measure" in text
        assert "read" in text or "reading" in text

    def test_partial_visibility_is_reported_as_partial(self, tmp_path):
        clock = {"t": 1_000.0}
        engine = _engine(tmp_path, now_fn=lambda: clock["t"])
        vital = next(iter(engine._specs))
        spec = engine._specs[vital]
        engine.ingest({vital: spec.setpoint}, at=1_000.0)
        text = engine.narrative()
        assert "of" in text and str(len(engine._specs)) in text

    def test_census_counts_only_fresh_vitals(self, tmp_path):
        clock = {"t": 1_000.0}
        engine = _engine(tmp_path, now_fn=lambda: clock["t"])
        vital = next(iter(engine._specs))
        engine._series[vital].append((1_000.0, 1.0))
        fresh, total, stale = engine._observation_census(1_000.0)
        assert (fresh, total, stale) == (1, len(engine._specs), False)
        fresh, _total, stale = engine._observation_census(
            1_000.0 + allostasis._INGEST_STALE_AFTER_S + 1.0,
        )
        assert fresh == 0 and stale is True


class TestLedgerIdentity:
    def test_forecast_ids_carry_an_issuer_and_full_width(self):
        source = inspect.getsource(AllostasisEngine._refresh_forecasts)
        assert 'f"fc-{_ISSUER}-{uuid.uuid4().hex}"' in source

    def test_regime_ids_carry_an_issuer_and_full_width(self):
        source = inspect.getsource(AllostasisEngine._cusum_update)
        assert 'f"{key}-{_ISSUER}-{uuid.uuid4().hex}"' in source

    def test_the_issuer_is_process_scoped(self):
        assert len(allostasis._ISSUER) == 8


class TestSingletonOwnership:
    def test_the_container_is_authoritative(self):
        source = inspect.getsource(allostasis.get_allostasis_engine)
        assert "_engine_from_container()" in source

    def test_reset_disowns_the_container_slot(self):
        source = inspect.getsource(allostasis.reset_allostasis_engine_for_test)
        assert "_container_slot_disowned = True" in source
        assert "with _engine_lock" in source

    def test_a_disowned_slot_is_taken_over_not_adopted(self):
        source = inspect.getsource(allostasis.get_allostasis_engine)
        assert "replace=_container_slot_disowned" in source

    def test_reset_actually_clears_the_module_engine(self):
        allostasis.reset_allostasis_engine_for_test()
        assert allostasis._engine is None
        assert allostasis._container_slot_disowned is True
        # Leave the flag as a fresh test would find it.
        allostasis._container_slot_disowned = False


class TestThreatChannelRecovery:
    def test_release_publishes_an_all_clear(self):
        source = inspect.getsource(AllostasisEngine.sample_and_regulate)
        assert "_clear_protecting" in source

    def test_the_all_clear_uses_the_alarm_channel(self):
        source = inspect.getsource(AllostasisEngine._clear_protecting)
        assert '"existential_threat"' in source
        assert '"resolved": True' in source

    def test_a_failed_all_clear_is_a_degradation(self):
        source = inspect.getsource(AllostasisEngine._clear_protecting)
        assert "record_degradation" in source


class TestPulseDoesNotBlockTheLoop:
    def test_the_snapshot_runs_off_the_event_loop(self):
        source = inspect.getsource(AllostasisEngine.sample_and_regulate)
        # Either way of leaving the loop. `asyncio.to_thread` became
        # `run_on_a_thread_while_it_works`, which cancels on a STALL rather
        # than on a deadline — a snapshot still making progress is no longer
        # killed at the timeout. What this holds is that the snapshot does
        # not run ON the loop, and that the wait is bounded.
        assert (
            "asyncio.to_thread" in source
            or "run_on_a_thread_while_it_works" in source
        ), "the vitals snapshot must not run on the event loop"
        assert "asyncio.wait_for" in source or "stall_s=" in source, (
            "and the wait for it must be bounded"
        )

    def test_a_wedged_provider_cannot_wedge_the_pulse(self):
        source = inspect.getsource(AllostasisEngine.sample_and_regulate)
        assert "_SNAPSHOT_TIMEOUT_S" in source
        assert "asyncio.TimeoutError" in source


class TestCoverageCensoring:
    """``b90445ef`` — intervened forecasts leave the coverage denominator.

    That exclusion is correct in principle: regulation that prevents a
    crossing does not falsify the forecast that prompted it. But an engine
    that escalates on everything would then excuse everything, and the
    coverage figure would quietly stop meaning anything. The censoring rate
    now travels with the coverage it conditions, and heavy censoring widens
    the bands instead of leaving them at nominal width.
    """

    def _book(self, **kwargs):
        book = allostasis._VitalCalibration()
        for name, value in kwargs.items():
            setattr(book, name, value)
        return book

    def test_censored_counts_interventions_and_supersessions(self):
        book = self._book(intervened=3, superseded=2)
        assert book.censored == 5

    def test_censored_fraction_is_of_all_resolved(self):
        book = self._book(hits=1, false_alarms=1, intervened=2)
        assert book.scored == 2
        assert book.censored_fraction == pytest.approx(0.5)

    def test_no_resolutions_has_no_fraction(self):
        assert self._book().censored_fraction is None

    def test_heavy_censoring_widens_the_bands(self):
        # Almost everything excused: coverage rests on one outcome.
        book = self._book(hits=1, intervened=9)
        assert book.censored_fraction >= allostasis._HEAVY_CENSORING
        assert book.widen_factor(target_coverage=0.9) > 1.0

    def test_light_censoring_leaves_bands_nominal(self):
        book = self._book(hits=4, intervened=1)
        assert book.censored_fraction < allostasis._HEAVY_CENSORING
        assert book.widen_factor(target_coverage=0.9) == 1.0

    def test_half_censored_already_counts_as_heavy(self):
        book = self._book(hits=1, intervened=1)
        assert book.widen_factor(target_coverage=0.9) > 1.0

    def test_censoring_is_reported_in_the_calibration_surface(self):
        payload = self._book(hits=2, intervened=2).to_dict()
        assert payload["censored"] == 2
        assert payload["censored_fraction"] == pytest.approx(0.5)

    def test_the_threat_contract_carries_the_censoring_rate(self):
        source = inspect.getsource(AllostasisEngine._raise_protecting)
        assert '"censored_fraction"' in source
        assert '"censored_forecasts"' in source


class TestTierRestoreCeiling:
    def test_protecting_is_not_restored_without_strain(self, tmp_path):
        engine = _engine(tmp_path)
        engine._load_raw = {k: 0.0 for k in engine._specs}
        assert engine._composite_load() < 0.30
        assert engine._tier == AllostasisTier.SETTLED
