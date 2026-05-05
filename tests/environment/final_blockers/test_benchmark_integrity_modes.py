"""Final Blocker: Strict benchmark integrity and mode separation.

Simulated canaries, fixture replays, and contaminated runs must never be
counted as real deep-run evidence. Mode separation must be enforced.
"""
import pytest
from core.environment.benchmark_runner import BenchmarkResult, BenchmarkReport, BenchmarkRunner
from core.environment.boundary_guard import BoundaryGuard, BoundaryConfig, BoundaryViolationError


class TestBenchmarkIntegrity:
    """Benchmark results must strictly separate real/simulated/fixture/contaminated."""

    def test_simulated_canary_result_not_scored_as_real(self):
        report = BenchmarkReport()
        canary = BenchmarkResult(
            run_id="canary_1",
            baseline_name="random_baseline",
            mode="simulated_canary",
            success=True,
            total_steps=100,
            simulated=True,
        )
        report.add_result(canary)
        # Real results must be empty
        assert len(report.real_results()) == 0
        assert len(report.canary_results()) == 1

    def test_fixture_replay_not_counted_as_live(self):
        report = BenchmarkReport()
        replay = BenchmarkResult(
            run_id="replay_1",
            baseline_name="parser_regression",
            mode="fixture_replay",
            success=True,
            total_steps=50,
        )
        report.add_result(replay)
        assert len(report.real_results()) == 0

    def test_contaminated_run_excluded_from_real(self):
        report = BenchmarkReport()
        contaminated = BenchmarkResult(
            run_id="real_1",
            baseline_name="full_policy",
            mode="strict_real",
            success=True,
            total_steps=200,
            contaminated=True,
        )
        report.add_result(contaminated)
        assert len(report.real_results()) == 0

    def test_clean_strict_real_counted(self):
        report = BenchmarkReport()
        clean = BenchmarkResult(
            run_id="real_2",
            baseline_name="full_policy",
            mode="strict_real",
            success=False,
            total_steps=300,
            contaminated=False,
        )
        report.add_result(clean)
        assert len(report.real_results()) == 1

    def test_ablation_results_tracked_separately(self):
        report = BenchmarkReport()
        ablation = BenchmarkResult(
            run_id="abl_1",
            baseline_name="ablation_no_belief_graph",
            mode="fixture_replay",
            success=False,
            total_steps=50,
        )
        report.add_result(ablation)
        assert "ablation_no_belief_graph" in report.ablations

    def test_deep_run_claim_fails_without_strict_real(self):
        report = BenchmarkReport()
        canary = BenchmarkResult(
            run_id="c1", baseline_name="test", mode="simulated_canary",
            success=True, total_steps=10, simulated=True,
        )
        report.add_result(canary)
        # No real results means no valid deep-run claim
        # Validate via runner's method
        runner = BenchmarkRunner.__new__(BenchmarkRunner)
        runner.report = report
        runner.boundary_guard = BoundaryGuard()
        valid, failures = runner.validate_for_deep_run_claim()
        assert not valid
        assert "No strict_real results" in failures


class TestBoundaryEnforcement:
    """BoundaryGuard must block forbidden information channels at runtime."""

    def test_save_file_access_blocked(self):
        config = BoundaryConfig(forbidden_file_patterns=[".nethackdir/save", "save/"])
        guard = BoundaryGuard(config=config)
        with pytest.raises(BoundaryViolationError):
            guard.check_file_access("/home/user/.nethackdir/save/game.sav")
        assert guard.contaminated

    def test_oracle_observation_metadata_blocked(self):
        guard = BoundaryGuard()
        with pytest.raises(BoundaryViolationError):
            guard.check_observation_metadata({"seed": 12345, "mode": "strict_real"})

    def test_clean_observation_metadata_allowed(self):
        guard = BoundaryGuard()
        # Should not raise
        guard.check_observation_metadata({"mode": "strict_real", "run_id": "test123"})
        assert not guard.contaminated

    def test_process_memory_access_blocked(self):
        guard = BoundaryGuard()
        with pytest.raises(BoundaryViolationError):
            guard.check_process_memory_access()

    def test_eval_trace_leakage_blocked(self):
        config = BoundaryConfig(trace_split="train")
        guard = BoundaryGuard(config=config)
        with pytest.raises(BoundaryViolationError):
            guard.check_trace_split("trace_001", "eval")

    def test_train_trace_access_allowed(self):
        config = BoundaryConfig(trace_split="train")
        guard = BoundaryGuard(config=config)
        # Should not raise
        guard.check_trace_split("trace_001", "train")
        assert not guard.contaminated

    def test_integrity_report_reflects_violations(self):
        guard = BoundaryGuard()
        try:
            guard.check_process_memory_access()
        except BoundaryViolationError:
            pass
        report = guard.get_integrity_report("run_1", "strict_real")
        assert report.verdict == "CONTAMINATED"
        assert report.contamination_count == 1

    def test_strict_real_enforcement(self):
        guard = BoundaryGuard()
        with pytest.raises(ValueError, match="strict_real"):
            guard.enforce_strict_real("simulated_canary")

    def test_legacy_check_operation_still_works(self):
        guard = BoundaryGuard()
        with pytest.raises(BoundaryViolationError):
            guard.check_operation("read_save_file")
        with pytest.raises(BoundaryViolationError):
            guard.check_operation("any_op", channel="forbidden_channel")
