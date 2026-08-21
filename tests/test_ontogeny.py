"""Contract tests for the ontogenetic organ.

These are written against the *guarantees*, not the implementation. Each one
corresponds to a way the organ could quietly become dishonest, and most of them
correspond to a defect that was actually found and fixed while building it:

  * an unobserved outcome silently becoming a failure label
  * test data reaching the live corpus
  * a stuck loop drowning the corpus in one repeated fact
  * repeat-collapsing merging distinct decisions that get graded separately
  * a learned head overturning a decision made for identity or safety reasons
  * a promotion happening without held-out separation from the incumbent
  * exploration stopping once a head holds authority
  * a random-exploration episode taking an action with real consequences
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from core.ontogeny.authority import (
    MIN_TRIALS,
    AuthorityLedger,
    AuthorityStage,
    Comparison,
    compare,
)
from core.ontogeny.calibration import (
    CalibrationMonitor,
    TrackRecord,
    TrackRecordIndex,
    track_records,
    wilson,
)
from core.ontogeny.experience import (
    Episode,
    ExperienceSpine,
    Outcome,
    OutcomeKind,
    Provenance,
)
from core.ontogeny.features import EXECUTIVE_ADMISSION, FeatureSchema, RunningMoments, design_row
from core.ontogeny.heads import PredictionHead
from core.ontogeny.reservation import Decider, ExplorationReservation
from core.ontogeny.resolution import ResolverRegistry
from core.ontogeny.state import OntogeneticState
from core.ontogeny.trainer import replay_design
from core.ontogeny.wiring import SEALED_REASONS, admission_features, admission_stakes, is_sealed


@pytest.fixture
def sandbox(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def spine(sandbox: Path) -> ExperienceSpine:
    store = ExperienceSpine(sandbox / "exp.db", autoflush=False)
    yield store
    store.close()


def _episode(**kwargs) -> Episode:
    base = {
        "control_point": "test.cp",
        "features": {"a": 1.0, "b": 2.0},
        "decision": "approved",
        "options": ("approved", "deferred"),
        "provenance": Provenance.TEST,
    }
    base.update(kwargs)
    return Episode(**base)


# ── L0: the corpus cannot lie ───────────────────────────────────────────────


class TestHonestOutcomes:
    def test_unobserved_is_not_failure(self):
        """The defect that motivated this whole layer."""
        outcome = Outcome.unobserved("sweeper:not_observable")
        assert outcome.kind is OutcomeKind.UNOBSERVED
        assert outcome.kind is not OutcomeKind.FAILURE
        assert not outcome.kind.is_evidence

    def test_unobserved_cannot_carry_a_utility(self):
        with pytest.raises(ValueError):
            Outcome(kind=OutcomeKind.UNOBSERVED, utility=0.0, resolver="x")

    def test_utility_must_be_a_probability(self):
        with pytest.raises(ValueError):
            Outcome(kind=OutcomeKind.SUCCESS, utility=1.4, resolver="x")

    def test_evidence_only_read_excludes_unobserved(self, spine: ExperienceSpine):
        for i in range(3):
            ep = _episode(features={"a": float(i), "b": 0.0})
            spine.record(ep)
            spine.resolve(
                ep.episode_id,
                Outcome.from_utility(1.0, "t") if i == 0
                else Outcome.unobserved("t") if i == 1
                else Outcome.from_utility(0.0, "t"),
            )
        spine.flush()
        assert len(spine.episodes("test.cp")) == 3
        assert len(spine.episodes("test.cp", evidence_only=True)) == 2


class TestProvenance:
    def test_live_store_refuses_non_live_writes(self, monkeypatch, tmp_path: Path):
        """Structural, not conventional: a forgotten flag cannot poison the corpus."""
        store = ExperienceSpine(tmp_path / "exp.db", autoflush=False)
        monkeypatch.setattr(store, "_store_kind", "live")
        assert store.record(_episode(provenance=Provenance.TEST)) is None
        assert store.record(_episode(provenance=Provenance.BENCHMARK)) is None
        assert store.record(_episode(provenance=Provenance.LIVE)) is not None
        assert store.stats()["refused_provenance"] == 2
        store.close()

    def test_sandbox_store_accepts_everything(self, spine: ExperienceSpine):
        assert spine.store_kind == "sandbox"
        assert spine.record(_episode(provenance=Provenance.TEST)) is not None


class TestBurstLimiting:
    def test_runaway_loop_collapses(self, spine: ExperienceSpine):
        """990,653 identical receipts is one fact, not a corpus."""
        for _ in range(200):
            spine.record(_episode())
        spine.flush()
        stats = spine.stats("test.cp")
        assert stats["rows"] < 20, "a stuck loop must not fill the corpus"
        assert stats["observations"] >= 200, "but the count must survive"

    def test_distinct_decisions_are_not_merged(self, spine: ExperienceSpine):
        """The bug the burst threshold exists to avoid.

        Two intents can look identical to the organ and still turn out
        differently; merging them throws away the disagreement, which is
        exactly the signal worth having.
        """
        for i in range(40):
            spine.record(_episode(features={"a": float(i), "b": float(i % 7)}))
        spine.flush()
        assert spine.stats("test.cp")["rows"] == 40


# ── L1: the counterfactual stays observable ─────────────────────────────────


class TestExploration:
    def test_authority_still_reserves_for_the_incumbent(self):
        """Permanent, or the promotion becomes unfalsifiable."""
        res = ExplorationReservation(hourly_budget=10_000)
        deciders = [
            res.decide("cp", seed=f"e{i}", stakes=0.3, has_authority=True, challenger_ready=True).decider
            for i in range(400)
        ]
        assert Decider.INCUMBENT in deciders, "a promoted head must keep being compared"
        assert deciders.count(Decider.CHALLENGER) > deciders.count(Decider.INCUMBENT)

    def test_random_slice_exists_at_every_stage(self):
        """Positivity: without it no action's effect is identifiable."""
        res = ExplorationReservation(hourly_budget=10_000)
        for authority in (False, True):
            deciders = [
                res.decide("cp", seed=f"r{i}", stakes=0.3,
                           has_authority=authority, challenger_ready=True).decider
                for i in range(400)
            ]
            assert Decider.RANDOM in deciders

    def test_high_stakes_is_never_explored(self):
        res = ExplorationReservation(hourly_budget=10_000)
        for i in range(500):
            r = res.decide("cp", seed=f"h{i}", stakes=0.95,
                           has_authority=False, challenger_ready=True)
            assert not r.reserved
            assert r.decider is Decider.INCUMBENT

    def test_reservations_are_deterministic_in_the_seed(self):
        a = ExplorationReservation(hourly_budget=10_000)
        b = ExplorationReservation(hourly_budget=10_000)
        for i in range(100):
            kwargs = dict(stakes=0.3, has_authority=False, challenger_ready=True)
            assert a.decide("cp", seed=f"d{i}", **kwargs).decider == \
                   b.decide("cp", seed=f"d{i}", **kwargs).decider

    def test_hourly_budget_bounds_a_burst(self):
        res = ExplorationReservation(hourly_budget=5)
        reserved = sum(
            1 for i in range(2000)
            if res.decide("cp", seed=f"b{i}", stakes=0.2,
                          has_authority=False, challenger_ready=True).reserved
        )
        assert reserved <= 5


# ── L2: resolution defaults to ignorance ────────────────────────────────────


class TestResolution:
    def test_no_resolver_means_unobserved_not_failure(self, spine: ExperienceSpine):
        registry = ResolverRegistry()
        ep = _episode(horizon_s=0.0)
        spine.record(ep)
        spine.flush()
        result = registry.sweep(spine)
        assert result["swept"] == 1
        assert result["unobserved"] == 1
        stored = spine.episodes("test.cp")[0]
        assert stored.outcome.kind is OutcomeKind.UNOBSERVED

    def test_sweeper_yields_between_episodes_without_overcounting(
        self,
        spine: ExperienceSpine,
    ):
        registry = ResolverRegistry()
        spine.record(
            _episode(
                episode_id="first",
                horizon_s=0.0,
                features={"a": 1.0, "b": 2.0},
            )
        )
        spine.record(
            _episode(
                episode_id="second",
                horizon_s=0.0,
                features={"a": 2.0, "b": 2.0},
            )
        )
        spine.flush()
        probes = 0

        def _foreground_arrived() -> bool:
            nonlocal probes
            probes += 1
            return probes > 1

        result = registry.sweep(spine, should_stop=_foreground_arrived)

        assert result == {"swept": 1, "observed": 0, "unobserved": 1}
        assert len(spine.open_episodes(older_than_horizon=False)) == 1

    def test_a_raising_resolver_leaves_the_episode_unobserved(self, spine: ExperienceSpine):
        registry = ResolverRegistry()

        def boom(_episode):
            raise RuntimeError("resolver is broken")

        registry.register_fn("test.cp", 0.0, boom)
        ep = _episode(horizon_s=0.0)
        spine.record(ep)
        spine.flush()
        registry.sweep(spine)
        assert spine.episodes("test.cp")[0].outcome.kind is OutcomeKind.UNOBSERVED

    def test_a_resolver_that_can_measure_produces_evidence(self, spine: ExperienceSpine):
        registry = ResolverRegistry()
        registry.register_fn("test.cp", 0.0, lambda ep: Outcome.from_utility(1.0, "t"))
        ep = _episode(horizon_s=0.0)
        spine.record(ep)
        spine.flush()
        assert registry.sweep(spine)["observed"] == 1

    def test_durable_observation_window_is_bounded_and_control_point_local(
        self,
        spine: ExperienceSpine,
    ):
        episodes = [
            _episode(
                features={"a": float(index), "b": 2.0},
                decided_at=float(index + 1),
            )
            for index in range(80)
        ]
        for episode in episodes:
            spine.record(episode)
        spine.flush()
        for index, episode in enumerate(episodes):
            outcome = (
                Outcome.from_utility(1.0, "test")
                if index < 30 or index >= 70
                else Outcome.unobserved("test")
            )
            spine.resolve(episode.episode_id, outcome)
        spine.flush()

        report = spine.observation_stats("test.cp", recent_limit=50)

        assert report["available"] is True
        assert report["closed"] == 50
        assert report["observed"] == 10
        assert report["unobserved"] == 40
        assert report["observation_rate"] == pytest.approx(0.2)
        assert spine.observation_stats("other.cp", recent_limit=50)["closed"] == 0

    def test_durable_observation_window_tracks_latest_resolutions(
        self,
        spine: ExperienceSpine,
    ):
        older_decision = _episode(
            features={"a": 1.0, "b": 2.0},
            decided_at=1.0,
        )
        newer_decision = _episode(
            features={"a": 2.0, "b": 2.0},
            decided_at=2.0,
        )
        spine.record(older_decision)
        spine.record(newer_decision)
        spine.flush()
        spine.resolve(newer_decision.episode_id, Outcome.unobserved("test"))
        spine.flush()
        spine.resolve(older_decision.episode_id, Outcome.from_utility(1.0, "test"))
        spine.flush()

        report = spine.observation_stats("test.cp", recent_limit=1)

        assert report["closed"] == 1
        assert report["observed"] == 1
        assert report["observation_rate"] == pytest.approx(1.0)
        assert report["window_started_at"] == report["window_ended_at"]


# ── L3: state and heads ─────────────────────────────────────────────────────


class TestOntogeneticState:
    def test_state_is_deterministic_given_the_seed(self):
        a = OntogeneticState(input_width=8, units=16, seed=1, path=Path("/nonexistent/a.npz"))
        b = OntogeneticState(input_width=8, units=16, seed=1, path=Path("/nonexistent/b.npz"))
        row = np.ones(8)
        for _ in range(20):
            a.step(row)
            b.step(row)
        assert np.allclose(a.h, b.h)

    def test_state_is_bounded(self):
        state = OntogeneticState(input_width=8, units=16, seed=2, path=Path("/nonexistent/c.npz"))
        for _ in range(500):
            state.step(np.ones(8) * 50.0)
        assert np.all(np.abs(state.h) <= 1.0 + 1e-6)

    def test_novelty_is_neutral_before_there_is_a_distribution(self):
        state = OntogeneticState(input_width=4, units=8, seed=3, path=Path("/nonexistent/d.npz"))
        assert state.step(np.ones(4)).novelty == 0.5

    def test_novelty_rises_for_an_unfamiliar_state(self):
        state = OntogeneticState(input_width=4, units=32, seed=4, path=Path("/nonexistent/e.npz"))
        rng = np.random.default_rng(0)
        for _ in range(300):
            state.step(rng.normal(0, 0.1, 4))
        ordinary = state.step(rng.normal(0, 0.1, 4)).novelty
        strange = state.step(np.array([9.0, -9.0, 9.0, -9.0])).novelty
        assert strange > ordinary

    def test_checkpoint_round_trip(self, sandbox: Path):
        path = sandbox / "state.npz"
        state = OntogeneticState(input_width=6, units=16, seed=7, path=path)
        for i in range(40):
            state.step(np.full(6, i / 40.0))
        assert state.save()
        restored = OntogeneticState(input_width=6, units=16, seed=7, path=path)
        assert restored.load()
        assert np.allclose(restored.h, state.h)
        assert restored.steps == state.steps

    def test_shape_mismatch_is_refused_not_coerced(self, sandbox: Path):
        path = sandbox / "state.npz"
        OntogeneticState(input_width=6, units=16, seed=7, path=path).save()
        other = OntogeneticState(input_width=9, units=16, seed=7, path=path)
        assert not other.load()

    def test_replay_cooperates_at_bounded_intervals(self):
        schema = FeatureSchema("test.cp", ("a", "b"))
        episodes = [
            _episode(features={"a": float(index), "b": float(index % 3)})
            for index in range(130)
        ]
        handoffs = 0

        def _handoff() -> None:
            nonlocal handoffs
            handoffs += 1

        rows, kept, _moments = replay_design(
            episodes,
            schema,
            units=8,
            seed=3,
            washout=0,
            cooperate=_handoff,
        )

        assert rows.shape[0] == len(kept) == 130
        assert handoffs == 3


class TestHeads:
    def _rows(self, n: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        x = rng.normal(size=(n, 4))
        y = ["success" if row[0] + row[1] > 0 else "failure" for row in x]
        return x, y

    def test_a_head_learns_a_learnable_boundary(self):
        head = PredictionHead("cp", ("failure", "success"), 4)
        x, y = self._rows(600)
        evidence = head.fit(x, y)
        assert evidence["fitted"]
        assert evidence["train_accuracy"] > 0.85

    def test_attribution_is_arithmetic_over_named_features(self):
        head = PredictionHead("cp", ("failure", "success"), 4, ("w", "x", "y", "z"))
        x, y = self._rows(400)
        head.fit(x, y)
        prediction = head.predict(x[0], attribute=True)
        assert prediction.attribution is not None
        names = {name for name, _ in prediction.attribution.contributions}
        assert names == {"w", "x", "y", "z"}

    def test_a_head_is_not_ready_before_it_has_evidence(self):
        head = PredictionHead("cp", ("failure", "success"), 4)
        assert not head.ready
        x, y = self._rows(20)
        head.fit(x, y)
        assert not head.ready, "twenty episodes is not a fitted model"

    def test_checkpoint_refuses_a_mismatched_option_set(self):
        head = PredictionHead("cp", ("failure", "success"), 4)
        x, y = self._rows(400)
        head.fit(x, y)
        other = PredictionHead("cp", ("no", "yes"), 4)
        assert not other.load_state(head.state_dict())

    def test_batch_fit_yields_to_foreground_work(self):
        head = PredictionHead("cp", ("failure", "success"), 4)
        x, y = self._rows(600)
        probes = 0

        def _foreground_arrived() -> bool:
            nonlocal probes
            probes += 1
            return probes >= 3

        evidence = head.fit(x, y, should_stop=_foreground_arrived)

        assert evidence == {"fitted": False, "reason": "foreground_preempted"}
        assert head.version == 0

    def test_batch_fit_cooperates_without_changing_the_learned_head(self):
        baseline = PredictionHead("cp", ("failure", "success"), 4)
        cooperative = PredictionHead("cp", ("failure", "success"), 4)
        x, y = self._rows(600)
        handoffs = 0

        def _handoff() -> None:
            nonlocal handoffs
            handoffs += 1

        baseline.fit(x, y)
        cooperative.fit(x, y, cooperate=_handoff)

        assert handoffs > 12
        assert np.array_equal(cooperative.w, baseline.w)
        assert np.array_equal(cooperative.b, baseline.b)
        assert cooperative.temperature == baseline.temperature
        assert cooperative.samples_seen == baseline.samples_seen
        assert cooperative.version == baseline.version
        assert cooperative.fit_evidence == baseline.fit_evidence


# ── L4: calibration and the track record ────────────────────────────────────


class TestCalibration:
    def test_wilson_bounds_bracket_the_estimate(self):
        low = wilson(7, 10, upper=False)
        high = wilson(7, 10, upper=True)
        assert low < 0.7 < high

    def test_small_samples_produce_wide_intervals(self):
        narrow = wilson(70, 100, upper=True) - wilson(70, 100, upper=False)
        wide = wilson(7, 10, upper=True) - wilson(7, 10, upper=False)
        assert wide > narrow

    def test_overconfidence_is_detected(self):
        monitor = CalibrationMonitor()
        for i in range(200):
            monitor.observe("cp", confidence=0.95, correct=i % 2 == 0)
        report = monitor.report("cp")
        assert report.overconfidence > 0.4
        assert report.ece > 0.4

    def test_a_calibrated_head_reports_low_error(self):
        monitor = CalibrationMonitor()
        rng = np.random.default_rng(1)
        for _ in range(400):
            confidence = float(rng.uniform(0.5, 1.0))
            monitor.observe("cp", confidence=confidence, correct=rng.random() < confidence)
        assert monitor.report("cp").ece < 0.12

    def test_drift_revokes_only_against_a_baseline(self):
        monitor = CalibrationMonitor()
        for _ in range(120):
            monitor.observe("cp", confidence=0.8, correct=True)
        monitor.set_baseline("cp")
        for _ in range(500):
            monitor.observe("cp", confidence=0.95, correct=False)
        drifted, reason = monitor.drifted("cp")
        assert drifted, reason


class TestTrackRecord:
    def test_a_thin_record_refuses_to_state_a_rate(self):
        record = TrackRecord("cp", "approved", successes=2, failures=1)
        assert not record.is_grounded
        assert record.interval is None
        assert "going by" in record.phrase()

    def test_a_thick_record_states_a_rate_and_an_interval(self):
        record = TrackRecord("cp", "approved", successes=80, failures=20)
        assert record.is_grounded
        low, high = record.interval
        assert low < 0.8 < high

    def test_a_bad_record_says_so(self):
        """A self-assessment that only speaks when flattering is not one."""
        record = TrackRecord("cp", "approved", successes=10, failures=90)
        assert "poor record" in record.phrase()

    def test_unobserved_episodes_are_reported_not_hidden(self):
        episodes = []
        for i in range(30):
            ep = _episode()
            ep.outcome = (
                Outcome.from_utility(1.0, "t") if i < 10
                else Outcome.unobserved("t")
            )
            episodes.append(ep)
        record = track_records(episodes)["approved"]
        assert record.successes == 10
        assert record.unobserved == 20

    def test_the_index_matches_a_full_aggregation(self):
        index = TrackRecordIndex()
        episodes = []
        for i in range(40):
            ep = _episode()
            ep.outcome = Outcome.from_utility(1.0 if i % 3 else 0.0, "t")
            episodes.append(ep)
            index.observe("test.cp", ep.decision, ep.outcome.kind)
        aggregated = track_records(episodes)["approved"]
        live = index.get("test.cp", "approved")
        assert (live.successes, live.failures) == (aggregated.successes, aggregated.failures)


# ── L7: authority is earned, conservatively ─────────────────────────────────


class TestAuthority:
    def test_promotion_needs_separation_not_just_a_better_mean(self):
        """A lucky streak is not evidence."""
        close = Comparison("cp", challenger_successes=42, challenger_total=50,
                           incumbent_successes=80, incumbent_total=100)
        assert close.challenger_rate > close.incumbent_rate
        assert not close.challenger_wins, "84% over n=50 does not beat 80% over n=100"

    def test_a_decisive_challenger_wins(self):
        decisive = Comparison("cp", challenger_successes=180, challenger_total=200,
                              incumbent_successes=600, incumbent_total=1000)
        assert decisive.challenger_wins

    def test_thin_evidence_never_promotes(self):
        thin = Comparison("cp", challenger_successes=MIN_TRIALS - 1,
                          challenger_total=MIN_TRIALS - 1,
                          incumbent_successes=0, incumbent_total=1000)
        assert not thin.sufficient
        assert not thin.challenger_wins

    def test_random_episodes_belong_to_neither_policy(self):
        episodes = []
        for decider in ("explore:random",) * 50:
            ep = _episode(decider=decider)
            ep.outcome = Outcome.from_utility(0.0, "t")
            episodes.append(ep)
        for decider in ("incumbent",) * 50:
            ep = _episode(decider=decider)
            ep.outcome = Outcome.from_utility(1.0, "t")
            episodes.append(ep)
        result = compare("test.cp", episodes)
        assert result.incumbent_total == 50
        assert result.challenger_total == 0

    def test_trust_is_harder_to_gain_than_to_lose(self):
        """The revocation test is deliberately not the mirror of promotion.

        Promotion demands separation *plus* a margin; revocation demands only
        separation. So there is a band of evidence that is decisive enough to
        take a decision away from a head but not decisive enough to have given
        it one — which is the correct asymmetry when the thing being trusted
        decides on Aura's behalf.
        """
        band = [
            challenger for challenger in range(50, 90)
            if Comparison("cp", challenger_successes=challenger, challenger_total=100,
                          incumbent_successes=80, incumbent_total=100).incumbent_wins
            and not Comparison("cp", challenger_successes=80, challenger_total=100,
                               incumbent_successes=challenger, incumbent_total=100).challenger_wins
        ]
        assert band, "revocation must be reachable on evidence that would not promote"

        # And the plain direction still holds at both ends of that band.
        decisive = Comparison("cp", challenger_successes=61, challenger_total=100,
                              incumbent_successes=80, incumbent_total=100)
        assert decisive.incumbent_wins
        assert not decisive.challenger_wins

    def test_overlapping_intervals_settle_nothing(self):
        """Ties go to the incumbent: neither side wins on ambiguous evidence."""
        overlapping = Comparison("cp", challenger_successes=60, challenger_total=100,
                                 incumbent_successes=75, incumbent_total=100)
        assert not overlapping.challenger_wins
        assert not overlapping.incumbent_wins

    def test_the_ladder_is_climbed_one_rung_at_a_time(self, sandbox: Path):
        ledger = AuthorityLedger(sandbox / "auth.json")
        assert ledger.stage("cp") is AuthorityStage.OBSERVE
        ledger.evaluate("cp", [], head_ready=True)
        assert ledger.stage("cp") is AuthorityStage.SHADOW

    def test_freeze_drops_every_head_to_shadow(self, sandbox: Path):
        ledger = AuthorityLedger(sandbox / "auth.json")
        ledger.set_stage("cp", AuthorityStage.AUTHORITY, reason="test")
        assert ledger.has_authority("cp")
        ledger.freeze("something is wrong and we do not yet know what")
        assert not ledger.has_authority("cp")
        assert ledger.stage("cp") is AuthorityStage.SHADOW
        ledger.unfreeze("resolved")
        assert ledger.has_authority("cp"), "a freeze suspends grants, it does not erase them"

    def test_grants_survive_a_restart(self, sandbox: Path):
        path = sandbox / "auth.json"
        AuthorityLedger(path).set_stage("cp", AuthorityStage.ADVISORY, reason="test")
        assert AuthorityLedger(path).stage("cp") is AuthorityStage.ADVISORY

    def test_revocation_returns_to_advisory_not_silence(self, sandbox: Path):
        ledger = AuthorityLedger(sandbox / "auth.json")
        ledger.set_stage("cp", AuthorityStage.AUTHORITY, reason="test")
        ledger.revoke("cp", "calibration drifted")
        assert ledger.stage("cp") is AuthorityStage.ADVISORY
        assert ledger.grant_of("cp").revocations == 1


# ── Wiring: the seals hold ──────────────────────────────────────────────────


class TestSeals:
    @pytest.mark.parametrize("reason", SEALED_REASONS)
    def test_safety_reasons_are_sealed(self, reason: str):
        assert is_sealed(reason)
        assert is_sealed(f"{reason}:some_detail")

    def test_user_facing_intents_are_never_experiments(self):
        assert is_sealed("approved", source="user")

    def test_ordinary_reasons_are_contestable(self):
        for reason in ("capacity_full_8/8", "internal_state_energy_low:0.10", "approved"):
            assert not is_sealed(reason, source="autonomous")

    def test_world_touching_actions_exceed_the_exploration_ceiling(self):
        from core.ontogeny.reservation import DEFAULT_STAKES_CEILING

        for action in ("tool_call", "emit_message", "update_belief", "mutate_state"):
            assert admission_stakes(action_type=action, priority=0.1, blocking=False) > \
                   DEFAULT_STAKES_CEILING

    def test_features_match_the_declared_schema(self):
        features = admission_features(
            priority=0.5, confidence=0.5, coherence=0.9, failure_pressure=0.1,
            active_goals=2, beliefs_contested=0, pending_initiatives=1, blocking=False,
            requires_tool=False, requires_memory_commit=False, identity_check=True,
            self_model_available=True, source="autonomous", action_type="spawn_task",
        )
        assert set(features) == set(EXECUTIVE_ADMISSION.names)


# ── Features: absence is not zero ───────────────────────────────────────────


class TestFeatures:
    def test_absent_and_zero_are_distinguishable(self):
        schema = FeatureSchema("cp", ("x", "y"))
        present = schema.vector({"x": 0.0, "y": 1.0})
        absent = schema.vector({"y": 1.0})
        assert present.values[0] == absent.values[0] == 0.0
        assert present.present[0] == 1.0
        assert absent.present[0] == 0.0

    def test_a_schema_change_changes_its_id(self):
        a = FeatureSchema("cp", ("x", "y"))
        b = FeatureSchema("cp", ("x", "y", "z"))
        c = FeatureSchema("cp", ("x", "y"), version=2)
        assert a.schema_id != b.schema_id
        assert a.schema_id != c.schema_id

    def test_moments_ignore_absent_values(self):
        schema = FeatureSchema("cp", ("x",))
        moments = RunningMoments(1)
        for value in (10.0, 10.0, 10.0):
            moments.update(*(lambda v: (v.values, v.present))(schema.vector({"x": value})))
        vector = schema.vector({})
        moments.update(vector.values, vector.present)
        assert moments.mean[0] == pytest.approx(10.0), "a silent subsystem must not drag the mean"

    def test_design_row_is_values_then_presence(self):
        schema = FeatureSchema("cp", ("x", "y"))
        moments = RunningMoments(2)
        row = design_row(schema.vector({"x": 1.0}), moments, update=True)
        assert row.shape == (4,)
        assert list(row[2:]) == [1.0, 0.0]


# ── The whole organ ─────────────────────────────────────────────────────────


class TestOrganEndToEnd:
    def _core(self, sandbox: Path):
        from core.ontogeny.service import OntogenyCore

        return OntogenyCore(
            spine=ExperienceSpine(sandbox / "exp.db", autoflush=False),
            authority=AuthorityLedger(sandbox / "auth.json"),
            reservation=ExplorationReservation(hourly_budget=100_000),
            autostart=False,
        )

    def _features(self, rng):
        return admission_features(
            priority=rng.random(), confidence=rng.random(), coherence=0.6 + 0.4 * rng.random(),
            failure_pressure=rng.random(), active_goals=rng.integers(0, 8),
            beliefs_contested=rng.integers(0, 3), pending_initiatives=0, blocking=False,
            requires_tool=False, requires_memory_commit=False, identity_check=True,
            self_model_available=True, source="autonomous", action_type="spawn_task",
        )

    def test_health_report_never_scans_the_experience_corpus(
        self,
        sandbox: Path,
        monkeypatch,
    ):
        core = self._core(sandbox)

        def corpus_scan_is_a_bug(*_args, **_kwargs):
            raise AssertionError("health polled the experience database")

        monkeypatch.setattr(core._spine, "stats", corpus_scan_is_a_bug)
        report = core.health_report()

        assert report["schema"] == "aura.ontogeny.health.v1"
        assert "executive.admission" in report["stages"]
        assert "observation_rate" in report
        core.stop()

    def test_maintenance_training_cooperates_with_the_runtime(
        self,
        sandbox: Path,
        monkeypatch,
    ):
        from core.ontogeny import service as service_module
        from core.ontogeny.trainer import TrainingResult

        core = self._core(sandbox)
        handoffs: list[float] = []
        captured: dict[str, object] = {}

        def _train(*_args, **kwargs):
            captured.update(kwargs)
            cooperate = kwargs.get("cooperate")
            assert callable(cooperate)
            cooperate()
            return TrainingResult(
                control_point="executive.admission",
                fitted=False,
                reason="no evidence",
            )

        monkeypatch.setattr(core._trainer, "train", _train)
        monkeypatch.setattr(service_module.time, "sleep", handoffs.append)

        core.train(yield_to_foreground=True)

        assert callable(captured["should_stop"])
        assert handoffs == [service_module.TRAIN_COOPERATIVE_YIELD_S]
        core.stop()

    def test_the_organ_defers_to_the_incumbent_before_it_has_learned(self, sandbox: Path):
        core = self._core(sandbox)
        rng = np.random.default_rng(0)
        for i in range(60):
            verdict = core.consider(
                "executive.admission", self._features(rng),
                incumbent_choice="approved", seed=f"e{i}", stakes=0.4,
                provenance=Provenance.TEST,
            )
            assert verdict.choice in ("approved", "deferred", "degraded")
            if verdict.reservation.decider is not Decider.RANDOM:
                assert verdict.choice == "approved"
        core.stop()

    def test_an_unregistered_control_point_is_a_no_op(self, sandbox: Path):
        core = self._core(sandbox)
        verdict = core.consider(
            "nobody.knows", {"a": 1.0}, incumbent_choice="proceed", seed="s",
            provenance=Provenance.TEST,
        )
        assert verdict.choice == "proceed"
        assert verdict.decider == "incumbent"
        core.stop()

    def test_authority_observation_comes_from_persisted_control_point_evidence(
        self,
        sandbox: Path,
    ):
        core = self._core(sandbox)
        episodes = [
            _episode(
                control_point="executive.admission",
                features={"a": float(index), "b": 3.0},
                decided_at=float(index + 1),
            )
            for index in range(60)
        ]
        for episode in episodes:
            core._spine.record(episode)
        core._spine.flush()
        for index, episode in enumerate(episodes):
            outcome = (
                Outcome.from_utility(1.0, "test")
                if index >= 54
                else Outcome.unobserved("test")
            )
            core._spine.resolve(episode.episode_id, outcome)
        core._spine.flush()
        core.authority.set_stage(
            "executive.admission",
            AuthorityStage.AUTHORITY,
            reason="test",
        )

        report = core.authority_observation_report()

        stats = report["control_points"]["executive.admission"]
        assert stats["eligible"] is True
        assert stats["closed"] == 60
        assert stats["observation_rate"] == pytest.approx(0.1)
        assert report["minimum_rate"] == pytest.approx(0.1)
        core.stop()

    def test_collapsed_durable_observation_revokes_deciding_authority(
        self,
        sandbox: Path,
    ):
        core = self._core(sandbox)
        episodes = [
            _episode(
                control_point="executive.admission",
                features={"a": float(index), "b": 4.0},
                decided_at=float(index + 1),
            )
            for index in range(60)
        ]
        for episode in episodes:
            core._spine.record(episode)
        core._spine.flush()
        for episode in episodes:
            core._spine.resolve(episode.episode_id, Outcome.unobserved("test"))
        core._spine.flush()
        core.authority.set_stage(
            "executive.admission",
            AuthorityStage.AUTHORITY,
            reason="test",
        )

        revoked = core._enforce_authority_observation()

        assert revoked == ("executive.admission",)
        assert core.authority.stage("executive.admission") is AuthorityStage.ADVISORY
        core.stop()

    def test_collapsed_episode_keeps_original_pending_calibration(
        self,
        sandbox: Path,
        monkeypatch,
    ):
        core = self._core(sandbox)
        rng = np.random.default_rng(5)
        remembered: list[str] = []
        monkeypatch.setattr(core._spine, "record", lambda _episode: "original-episode")
        monkeypatch.setattr(
            core,
            "_remember_episode",
            lambda episode: remembered.append(episode.episode_id),
        )

        verdict = core.consider(
            "executive.admission",
            self._features(rng),
            incumbent_choice="approved",
            seed="collapsed",
            stakes=0.4,
            provenance=Provenance.TEST,
        )

        assert verdict.episode_id == "original-episode"
        assert remembered == []
        core.stop()

    def test_the_organ_beats_a_beatable_incumbent_on_held_out_evidence(self, sandbox: Path):
        """The end-to-end claim: it learns, and it is only promoted on evidence."""
        core = self._core(sandbox)
        rng = np.random.default_rng(11)

        def truth(features, action):
            right = features["priority"] > features["failure_pressure"]
            if action == "deferred":
                right = not right
            elif action in ("degraded", "rejected"):
                right = abs(features["priority"] - features["failure_pressure"]) < 0.15
            return right != (rng.random() < 0.08)

        def incumbent(features):
            return "approved" if features["priority"] > 0.5 else "deferred"

        for i in range(1400):
            features = self._features(rng)
            verdict = core.consider(
                "executive.admission", features, incumbent_choice=incumbent(features),
                seed=f"t{i}", stakes=0.4, provenance=Provenance.TEST,
            )
            if verdict.episode_id:
                core.resolve_success(
                    verdict.episode_id, truth(features, verdict.choice), resolver="test"
                )
        core._spine.flush()

        result = core.train()["executive.admission"]
        assert result.fitted, result.reason
        assert result.lift > 0.05, f"holdout {result.holdout_accuracy} vs base {result.holdout_base_rate}"
        assert core.authority.stage("executive.admission").rank >= AuthorityStage.SHADOW.rank
        core.stop()

    def test_an_action_with_no_evidence_gets_no_score(self, sandbox: Path):
        """Missing evidence is visible, never silently extrapolated."""
        core = self._core(sandbox)
        rng = np.random.default_rng(3)
        for i in range(900):
            features = self._features(rng)
            verdict = core.consider(
                "executive.admission", features, incumbent_choice="approved",
                seed=f"n{i}", stakes=0.4, provenance=Provenance.TEST,
            )
            if verdict.episode_id:
                core.resolve_success(verdict.episode_id, rng.random() < 0.7, resolver="test")
        core._spine.flush()
        core.train()
        cp = core._control_points["executive.admission"]
        assert "rejected" not in cp.scorable, "never chosen, so never scored"
        core.stop()

    def test_a_broken_organ_costs_the_learning_and_not_the_decision(self, sandbox: Path, monkeypatch):
        core = self._core(sandbox)

        def explode(*_args, **_kwargs):
            raise RuntimeError("organ failure")

        monkeypatch.setattr(core, "_consider", explode)
        verdict = core.consider(
            "executive.admission", {"priority": 0.5}, incumbent_choice="approved",
            seed="x", provenance=Provenance.TEST,
        )
        assert verdict.choice == "approved"
        assert verdict.decider == "incumbent"
        core.stop()


# ── Causal on day one: novelty reaches the allocator ────────────────────────


class TestNoveltyIsCausal:
    """The organ changes what Aura does before any head has earned authority.

    Novelty is a measurement of her own state distribution, not a model's
    prediction, so it needs no earned trust — only a bounded effect.
    """

    def _service(self):
        from core.brain.latent_cortex_service import LatentCortexService

        service = LatentCortexService.__new__(LatentCortexService)
        service._body_pressure = lambda: 0.1
        service._runtime_pressure_snapshot = lambda: {
            "observation_source": "test_probe",
            "resource_observation_available": True,
            "memory_percent": 40.0,
        }
        return service

    def _allocate(self, service, novelty: float | None):
        from core.brain.latent_cortex_service import LatentCortexService
        from core.ontogeny.service import get_ontogeny
        from core.ontogeny.state import StateReading

        core = get_ontogeny()
        previous = core._last_reading
        if novelty is not None:
            core._last_reading = StateReading(
                hidden=np.zeros(4), novelty=novelty, displacement=0.0, steps=1, era=1
            )
        try:
            return LatentCortexService.allocate(service, stakes=0.5, uncertainty=0.3)
        finally:
            core._last_reading = previous

    def test_an_ordinary_moment_is_left_exactly_as_measured(self):
        """A signal that moves every allocation has stopped saying anything."""
        _, budget = self._allocate(self._service(), 0.5)
        assert budget["effective_uncertainty"] == pytest.approx(0.3)

    def test_an_unprecedented_moment_buys_more_thought(self):
        service = self._service()
        ordinary, _ = self._allocate(service, 0.5)
        unprecedented, budget = self._allocate(service, 1.0)
        assert budget["effective_uncertainty"] > 0.3
        assert unprecedented["max_steps"] > ordinary["max_steps"]

    def test_the_effect_is_bounded(self):
        """Unfamiliarity earns a little more thought, never a blank cheque."""
        from core.brain.latent_cortex_service import LatentCortexService

        _, budget = self._allocate(self._service(), 1.0)
        assert budget["effective_uncertainty"] <= 0.3 + LatentCortexService._NOVELTY_EFFORT_WEIGHT

    def test_a_missing_organ_leaves_allocation_untouched(self, monkeypatch):
        """A broken organ costs Aura the signal, never the allocation."""
        from core.brain.latent_cortex_service import LatentCortexService

        def down():
            raise RuntimeError("organ down")

        monkeypatch.setattr("core.ontogeny.service.get_ontogeny", down)
        config, budget = LatentCortexService.allocate(
            self._service(), stakes=0.5, uncertainty=0.3
        )
        assert budget["novelty"] is None
        assert budget["effective_uncertainty"] == pytest.approx(0.3)
        assert config["max_steps"] >= 2
