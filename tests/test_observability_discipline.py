"""Contract tests for the Chromium-derived disciplines.

core/observability/{histograms,trace_events}.py,
core/runtime/{memory_infra,field_trials}.py,
core/security/rule_of_two.py, tools/check_layering.py.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import pytest

from core.observability import histograms as hist_mod
from core.observability import trace_events as trace_mod
from core.observability.histograms import (
    Bucketing,
    declare_histogram,
    get_histogram,
    histograms_report,
)
from core.observability.trace_events import Tracer
from core.runtime import field_trials as trial_mod
from core.runtime import memory_infra as mem_mod
from core.runtime import sanitizers as san_mod
from core.runtime import taint as taint_mod
from core.runtime.field_trials import FieldTrials, TrialGroup, TrialSpec, declare_trial
from core.runtime.memory_infra import (
    AllocatorDump,
    DetailLevel,
    MemoryInfra,
    register_sized_container,
)
from core.security import rule_of_two as r2_mod
from core.security.rule_of_two import (
    Capability,
    InputTrust,
    Isolation,
    RuleOfTwoViolation,
    accept_risk,
    declare_handler,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def _clean():
    for mod in (hist_mod, trace_mod, mem_mod, trial_mod, r2_mod, san_mod, taint_mod):
        for name in dir(mod):
            if name.startswith("reset_") and name.endswith("_for_test"):
                getattr(mod, name)()
    yield
    for mod in (hist_mod, trace_mod, mem_mod, trial_mod, r2_mod, san_mod, taint_mod):
        for name in dir(mod):
            if name.startswith("reset_") and name.endswith("_for_test"):
                getattr(mod, name)()


# ── histograms ────────────────────────────────────────────────────────

def test_a_histogram_without_an_owner_is_refused():
    with pytest.raises(ValueError, match="description and an owner"):
        declare_histogram("Nameless", description="", owner="")


def test_exponential_buckets_put_resolution_where_the_action_is():
    h = declare_histogram(
        "Lat", description="latency", owner="t", minimum=1.0, maximum=100_000.0, bucket_count=30
    )
    for value in (2, 3, 5, 900, 40_000):
        h.record(value)
    snapshot = h.snapshot()
    assert snapshot["count"] == 5
    # p50 lands near the cluster of small values, not near the mean.
    assert snapshot["p50"] < 100
    assert snapshot["mean"] > 1000


def test_percentiles_survive_a_heavy_tail():
    h = declare_histogram(
        "Turn", description="turn latency", owner="t", minimum=1.0, maximum=100_000.0
    )
    for _ in range(985):
        h.record(80)
    for _ in range(15):
        h.record(40_000)
    snapshot = h.snapshot()
    assert snapshot["p50"] < 200
    assert snapshot["p99"] > 5_000, "the tail is the thing that hurts and must be visible"
    # And the mean, which is what a plain counter would have reported,
    # describes neither population.
    assert 500 < snapshot["mean"] < 1500


def test_overflow_is_visible_as_clipping():
    h = declare_histogram(
        "Small", description="s", owner="t", minimum=1.0, maximum=100.0, bucket_count=10
    )
    for _ in range(10):
        h.record(5)
    for _ in range(10):
        h.record(10_000)
    snapshot = h.snapshot()
    assert snapshot["overflow"] == 10
    assert snapshot["clipping"] is True
    assert "Small" in histograms_report()["clipping"]


def test_linear_bucketing_for_bounded_fractions():
    h = declare_histogram(
        "Frac",
        description="f",
        owner="t",
        bucketing=Bucketing.LINEAR,
        minimum=0.0,
        maximum=1.0,
        bucket_count=10,
    )
    for value in (0.05, 0.15, 0.95):
        h.record(value)
    assert h.snapshot()["count"] == 3
    assert h.percentile(0.5) <= 1.0


def test_enum_histograms_count_labels():
    h = declare_histogram(
        "Verdict",
        description="will verdicts",
        owner="t",
        bucketing=Bucketing.ENUM,
        enum_labels=("allow", "deny", "defer"),
    )
    h.record_enum("allow")
    h.record_enum("allow")
    h.record_enum("deny")
    h.record_enum("nonexistent")
    assert h.snapshot()["count"] == 3


def test_a_nonfinite_sample_is_discarded_and_reported():
    h = declare_histogram("Poison", description="p", owner="t")
    h.record(float("nan"))
    assert h.snapshot()["count"] == 0
    assert any(f["sanitizer"] == "numeric" for f in san_mod.sanitizer_report()["findings"])


def test_recording_into_an_undeclared_histogram_is_ignored_not_auto_created():
    assert hist_mod.record("Never.Declared", 1.0) is False
    assert get_histogram("Never.Declared") is None
    assert histograms_report()["count"] == 0


def test_conflicting_redeclaration_is_refused():
    declare_histogram("Dup", description="a", owner="one")
    with pytest.raises(ValueError, match="already declared"):
        declare_histogram("Dup", description="b", owner="two")


def test_standard_histograms_declare_cleanly():
    names = hist_mod.install_standard_histograms()
    assert "Aura.Pass.DurationMs" in names
    report = histograms_report()
    assert report["count"] == len(names)
    assert all(report["histograms"][n]["count"] == 0 for n in names)


def test_timer_records_elapsed_ms():
    declare_histogram("Timed", description="t", owner="t", minimum=0.01, maximum=10_000.0)
    with hist_mod.Timer("Timed"):
        time.sleep(0.01)
    assert get_histogram("Timed").snapshot()["count"] == 1


# ── trace events ──────────────────────────────────────────────────────

def test_complete_slices_nest_into_a_flamegraph():
    tracer = Tracer()
    with tracer.slice("outer"):
        with tracer.slice("inner"):
            pass
    events = tracer.to_trace_json()["traceEvents"]
    slices = [e for e in events if e["ph"] == "X"]
    assert [s["name"] for s in slices] == ["inner", "outer"]
    inner, outer = slices
    # Containment is what a viewer turns into nesting.
    assert outer["ts"] <= inner["ts"]
    assert outer["ts"] + outer["dur"] >= inner["ts"] + inner["dur"]


def test_async_slices_can_end_on_a_different_thread():
    tracer = Tracer()
    tracer.async_begin("model_call", "call-1")
    tracer.async_end("model_call", "call-1")
    phases = [e["ph"] for e in tracer.to_trace_json()["traceEvents"] if e.get("id") == "call-1"]
    assert phases == ["b", "e"]


def test_flow_events_make_queue_latency_visible():
    tracer = Tracer()
    tracer.flow_out("enqueue", "job-7")
    time.sleep(0.005)
    tracer.flow_in("dequeue", "job-7")
    flows = [e for e in tracer.to_trace_json()["traceEvents"] if e.get("id") == "job-7"]
    assert [f["ph"] for f in flows] == ["s", "f"]
    # The gap is time the work existed and nothing was working on it.
    assert flows[1]["ts"] > flows[0]["ts"]


def test_counters_share_the_timeline_with_slices():
    tracer = Tracer()
    tracer.counter("memory", {"rss_mb": 900.0, "available": 0.2})
    counter = next(e for e in tracer.to_trace_json()["traceEvents"] if e["ph"] == "C")
    assert counter["args"]["rss_mb"] == 900.0


def test_the_output_is_a_loadable_trace_document():
    tracer = Tracer()
    tracer.name_thread("worker")
    with tracer.slice("work"):
        pass
    document = tracer.to_trace_json()
    assert document["displayTimeUnit"] == "ms"
    encoded = json.dumps(document)
    assert "traceEvents" in encoded
    metadata = [e for e in document["traceEvents"] if e["ph"] == "M"]
    assert any(e["args"].get("name") == "worker" for e in metadata)


def test_the_ring_is_bounded_and_reports_drops():
    tracer = Tracer(capacity=8)
    for i in range(20):
        tracer.instant(f"e{i}")
    report = tracer.report()
    assert report["buffered"] == 8
    assert report["emitted"] == 20
    assert report["dropped"] == 12


def test_disabled_categories_emit_nothing():
    tracer = Tracer()
    tracer.disable_category("noisy")
    with tracer.slice("skipped", category="noisy"):
        pass
    tracer.instant("kept", category="aura")
    assert [e["name"] for e in tracer.to_trace_json()["traceEvents"] if e["ph"] != "M"] == ["kept"]


def test_pass_tracing_hooks_into_the_instrumentation():
    from core.pipeline.pass_manager import (
        PassManager,
        PreservedAnalyses,
        reset_pass_manager_for_test,
    )

    reset_pass_manager_for_test()
    hist_mod.install_standard_histograms()
    assert trace_mod.install_pass_tracing() is True
    assert trace_mod.install_pass_tracing() is False, "must be idempotent"

    pm: PassManager[dict] = PassManager("traced")
    pm.add_fn("step", lambda u, a: PreservedAnalyses.all())
    pm.run({})

    slices = [e for e in trace_mod.get_tracer().to_trace_json()["traceEvents"] if e["ph"] == "X"]
    assert any(s["name"] == "step" for s in slices)
    assert get_histogram("Aura.Pass.DurationMs").snapshot()["count"] == 1
    reset_pass_manager_for_test()


@pytest.mark.parametrize("parent_exists", [False, True])
def test_trace_write_round_trips(tmp_path, parent_exists, monkeypatch):
    parent = tmp_path / "export"
    if parent_exists:
        parent.mkdir()
        parent.chmod(0o755)
    monkeypatch.setenv("AURA_GOVERNANCE_MODE", "enforce")

    async def scenario():
        tracer = Tracer()
        tracer.instant("something")
        return await tracer.write(parent / "t.json", reason="test")

    path = asyncio.run(scenario())
    assert path is not None
    if parent_exists:
        assert parent.stat().st_mode & 0o777 == 0o755
    assert path.stat().st_mode & 0o777 == 0o600
    document = json.loads(path.read_text())
    assert any(e["name"] == "something" for e in document["traceEvents"])


# ── memory-infra ──────────────────────────────────────────────────────

def test_a_diff_between_dumps_names_the_component_that_grew():
    infra = MemoryInfra()
    sizes = {"probe_cache": 10_000_000, "episodic": 5_000_000}
    infra.register("probe_cache", lambda _l: AllocatorDump("probe_cache", sizes["probe_cache"]))
    infra.register("episodic", lambda _l: AllocatorDump("episodic", sizes["episodic"]))

    first = infra.dump()
    sizes["probe_cache"] = 950_000_000  # the leak
    sizes["episodic"] = 17_000_000
    second = infra.dump()

    diff = infra.diff(first, second)
    top = diff.top_growers(1)[0]
    assert top[0] == "probe_cache"
    assert top[1] == 940_000_000
    # The narrative must name the culprit even when process RSS happens to
    # be flat — "the process did not grow" would be true and useless here.
    assert "probe_cache" in diff.narrative()


def test_narrative_flags_component_growth_that_rss_does_not_show():
    infra = MemoryInfra()
    sizes = {"cache": 0}
    infra.register("cache", lambda _l: AllocatorDump("cache", sizes["cache"]))
    first = infra.dump()
    sizes["cache"] = 900_000_000
    second = infra.dump()
    second.process_rss_bytes = first.process_rss_bytes  # RSS flat
    narrative = infra.diff(first, second).narrative()
    assert "did not grow" in narrative and "cache" in narrative
    assert "attribution is wrong" in narrative


def test_narrative_says_so_when_nothing_grew():
    infra = MemoryInfra()
    infra.register("steady", lambda _l: AllocatorDump("steady", 1000))
    first = infra.dump()
    second = infra.dump()
    second.process_rss_bytes = first.process_rss_bytes
    assert "no growth" in infra.diff(first, second).narrative()


def test_unattributed_bytes_are_reported_explicitly(monkeypatch):
    monkeypatch.setattr(mem_mod, "_process_rss_bytes", lambda: 1_000_000_000)
    infra = MemoryInfra()
    infra.register("small", lambda _l: AllocatorDump("small", 50_000_000))
    dump = infra.dump()
    assert dump.attributed_bytes == 50_000_000
    assert dump.unattributed_bytes == 950_000_000
    assert dump.to_dict()["attributed_fraction"] == pytest.approx(0.05)


def test_ownership_edges_resolve_double_counting(monkeypatch):
    monkeypatch.setattr(mem_mod, "_process_rss_bytes", lambda: 100)
    infra = MemoryInfra()
    infra.register("owner", lambda _l: AllocatorDump("owner", 60))
    infra.register("borrower", lambda _l: AllocatorDump("borrower", 60))
    assert infra.dump().attributed_bytes == 120, "both claim the same buffer"

    infra.add_ownership_edge("owner", "borrower")
    assert infra.dump().attributed_bytes == 60


def test_detail_levels_gate_expensive_providers():
    infra = MemoryInfra()
    calls: list[str] = []
    infra.register("cheap", lambda _l: (calls.append("cheap"), AllocatorDump("cheap", 1))[1])
    infra.register(
        "expensive",
        lambda _l: (calls.append("expensive"), AllocatorDump("expensive", 1))[1],
        min_level=DetailLevel.DETAILED,
    )
    infra.dump(DetailLevel.BACKGROUND)
    assert calls == ["cheap"]
    infra.dump(DetailLevel.DETAILED)
    assert "expensive" in calls


def test_a_broken_provider_does_not_blind_the_others():
    infra = MemoryInfra()
    infra.register("broken", lambda _l: (_ for _ in ()).throw(RuntimeError("nope")))
    infra.register("fine", lambda _l: AllocatorDump("fine", 42))
    dump = infra.dump()
    assert "fine" in dump.dumps
    assert infra.report()["providers"]["broken"]["failures"] == 1


def test_a_plain_container_can_be_attributed_without_instrumenting_it():
    cache: dict[str, int] = {}
    register_sized_container("probe_cache", cache, owner="t", bytes_per_entry=1000)
    before = mem_mod.get_memory_infra().dump()
    for i in range(50):
        cache[str(i)] = i
    after = mem_mod.get_memory_infra().dump()
    diff = mem_mod.get_memory_infra().diff(before, after)
    assert diff.growth_by_component["probe_cache"] == 50_000


def test_leak_report_needs_two_dumps():
    infra = MemoryInfra()
    infra.register("x", lambda _l: AllocatorDump("x", 1))
    assert infra.leak_report()["available"] is False
    infra.dump()
    infra.dump()
    assert infra.leak_report()["available"] is True


def test_growth_rate_is_expressed_per_hour():
    infra = MemoryInfra()
    first = infra.dump()
    second = infra.dump()
    object.__setattr__(second, "at", first.at + 3600.0)
    second.process_rss_bytes = first.process_rss_bytes + 242_000_000
    diff = infra.diff(first, second)
    assert diff.rate_mb_per_hour == pytest.approx(242.0, rel=0.01)


# ── field trials ──────────────────────────────────────────────────────

def _spec(name="t", **kwargs):
    return TrialSpec(
        name=name,
        hypothesis=kwargs.pop("hypothesis", "arm B is better"),
        owner=kwargs.pop("owner", "test"),
        groups=kwargs.pop("groups", (TrialGroup("control", 0.5), TrialGroup("treatment", 0.5))),
        **kwargs,
    )


def test_a_trial_without_a_hypothesis_is_refused():
    trials = FieldTrials()
    with pytest.raises(ValueError, match="hypothesis and an owner"):
        trials.declare(_spec(hypothesis=""))


def test_assignment_is_deterministic_and_sticky(monkeypatch):
    monkeypatch.setenv("AURA_FIELD_TRIAL_ENTROPY", "stable-install-id")
    first = FieldTrials()
    first.declare(_spec("retrieval"))
    assignment = first.group("retrieval")

    # A fresh process with the same entropy must land in the same arm; a
    # trial that reassigns on restart produces a mixture, and a mixture
    # measures nothing.
    for _ in range(5):
        other = FieldTrials()
        other.declare(_spec("retrieval"))
        assert other.group("retrieval") == assignment


def test_different_trials_get_independent_assignments(monkeypatch):
    monkeypatch.setenv("AURA_FIELD_TRIAL_ENTROPY", "seed")
    trials = FieldTrials()
    assignments = set()
    for i in range(20):
        trials.declare(_spec(f"trial_{i}"))
        assignments.add(trials.group(f"trial_{i}"))
    assert assignments == {"control", "treatment"}, "hashing must not collapse to one arm"


def test_weights_are_respected_across_many_trials(monkeypatch):
    monkeypatch.setenv("AURA_FIELD_TRIAL_ENTROPY", "weighted")
    trials = FieldTrials()
    counts = {"rare": 0, "common": 0}
    for i in range(200):
        trials.declare(
            _spec(f"w{i}", groups=(TrialGroup("rare", 0.1), TrialGroup("common", 0.9)))
        )
        counts[trials.group(f"w{i}")] += 1
    assert counts["common"] > counts["rare"] * 3


def test_assignment_is_frozen_within_a_process():
    trials = FieldTrials()
    trials.declare(_spec("frozen"))
    first = trials.group("frozen")
    assert all(trials.group("frozen") == first for _ in range(10))


def test_a_trial_can_be_forced_for_debugging_and_tests():
    trials = FieldTrials()
    trials.declare(_spec("forceable"))
    assert trials.force("forceable", "treatment") is True
    assert trials.group("forceable") == "treatment"
    assert trials.force("forceable", "nonexistent") is False


def test_an_undeclared_trial_returns_the_default_group():
    assert trial_mod.group("no_such_trial") == "default"


def test_an_expired_trial_falls_back_to_default():
    trials = FieldTrials()
    trials.declare(_spec("old", expires_days=0))
    trials._trials["old"].declared_at = time.time() - 86400 * 2
    assert trials.group("old") == "default"
    assert "old" in trials.report()["expired"]


def test_observations_are_keyed_by_arm():
    trials = FieldTrials()
    trials.declare(_spec("measured", metrics=("latency_ms",)))
    trials.force("measured", "treatment")
    trials.observe("measured", "latency_ms", 100)
    trials.observe("measured", "latency_ms", 200)
    results = trials.results("measured")
    assert results["arms"]["treatment"]["latency_ms"]["n"] == 2
    assert results["arms"]["treatment"]["latency_ms"]["mean"] == 150


def test_changing_an_in_flight_trial_is_refused():
    trials = FieldTrials()
    trials.declare(_spec("stable"))
    with pytest.raises(ValueError, match="already declared"):
        trials.declare(_spec("stable", hypothesis="a different claim"))


def test_module_level_declare_trial_works():
    declare_trial(
        "module_level",
        hypothesis="it works",
        owner="test",
        groups={"a": 1.0},
        metrics=("m",),
    )
    assert trial_mod.group("module_level") == "a"


# ── rule of two ───────────────────────────────────────────────────────

def test_three_legs_is_refused_at_declaration():
    with pytest.raises(RuleOfTwoViolation) as excinfo:
        declare_handler(
            "reckless_web_executor",
            input_trust=InputTrust.UNTRUSTED,
            capability=Capability.EXECUTES,
            isolation=Isolation.IN_PROCESS,
            owner="test",
        )
    message = str(excinfo.value)
    assert "All three at once is forbidden" in message
    # The error must say how to fix it, not just that it is wrong.
    assert "Give up one" in message
    assert "sandbox" in message


def test_two_legs_passes_silently():
    for legs in (
        dict(input_trust=InputTrust.UNTRUSTED, capability=Capability.EXECUTES, isolation=Isolation.SANDBOXED),
        dict(input_trust=InputTrust.UNTRUSTED, capability=Capability.PARSE_ONLY, isolation=Isolation.IN_PROCESS),
        dict(input_trust=InputTrust.TRUSTED, capability=Capability.EXECUTES, isolation=Isolation.IN_PROCESS),
    ):
        spec = declare_handler(f"h{hash(str(legs))}", owner="t", **legs)
        assert spec.violates is False
        assert spec.leg_count <= 2


def test_accepted_risk_makes_the_violation_an_artifact_with_a_name_on_it():
    accept_risk(
        "legacy_executor",
        reason="pre-existing surface, sandbox migration scheduled",
        accepted_by="owner",
    )
    spec = declare_handler(
        "legacy_executor",
        input_trust=InputTrust.UNTRUSTED,
        capability=Capability.EXECUTES,
        isolation=Isolation.IN_PROCESS,
        owner="test",
    )
    assert spec.violates is True
    assert "owner:" in spec.accepted_risk
    # A carried violation taints, so no later report reads clean over it.
    assert taint_mod.is_tainted(taint_mod.TaintFlag.GATE_BYPASSED)


def test_accept_risk_requires_a_reason_and_someone_accepting():
    with pytest.raises(ValueError, match="reason and someone"):
        accept_risk("x", reason="", accepted_by="")


def test_at_the_limit_handlers_are_reported_separately():
    declare_handler(
        "two_legs",
        input_trust=InputTrust.UNTRUSTED,
        capability=Capability.EXECUTES,
        isolation=Isolation.SANDBOXED,
        owner="t",
    )
    report = r2_mod.rule_of_two_report()
    assert report["at_the_limit"] == ["two_legs"]
    assert report["violations"] == []


def test_the_shipped_handler_postures_hold_the_rule():
    declared = r2_mod.install_known_handlers()
    assert "web_content_ingest" in declared
    report = r2_mod.rule_of_two_report()
    assert report["violations"] == [], (
        f"a shipped surface violates the Rule of Two: {report['violations']}"
    )


# ── layering gate ─────────────────────────────────────────────────────

def test_the_layering_gate_passes_on_the_current_tree():
    from tools.check_layering import main

    assert main([]) == 0


def test_deps_parser_never_executes_code(tmp_path):
    from tools.check_layering import parse_deps

    marker = tmp_path / "executed"
    deps_file = tmp_path / "DEPS"
    deps_file.write_text(
        f'include_rules = []\nopen({str(marker)!r}, "w").write("unsafe")\n',
        encoding="utf-8",
    )

    parsed = parse_deps(deps_file)

    assert parsed.rules == []
    assert not marker.exists()


def test_the_layering_gate_catches_a_new_upward_import(tmp_path):
    from tools.check_layering import Rule, check_module, scan

    # Unit-level: the rule engine itself.
    rules = [Rule("+", "core.runtime"), Rule("-", "core.brain")]
    assert check_module("core.runtime.taint", rules) is None
    assert check_module("core.brain.cortex", rules) is not None

    # Integration: a synthetic tree with a violation.
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "DEPS").write_text('include_rules = ["-core.brain"]\n')
    (package / "bad.py").write_text("import core.brain.cortex\n")
    violations = scan(package, package_root=package)
    assert len(violations) == 1
    assert violations[0].imported == "core.brain.cortex"


def test_rules_inherit_down_the_tree(tmp_path):
    from tools.check_layering import scan

    root = tmp_path / "root"
    (root / "child").mkdir(parents=True)
    (root / "DEPS").write_text('include_rules = ["-core.brain"]\n')
    (root / "child" / "deep.py").write_text("from core.brain import cortex\n")
    assert len(scan(root, package_root=root)) == 1


def test_the_nearest_rule_wins(tmp_path):
    from tools.check_layering import scan

    root = tmp_path / "root"
    (root / "child").mkdir(parents=True)
    (root / "DEPS").write_text('include_rules = ["-core.brain"]\n')
    (root / "child" / "DEPS").write_text('include_rules = ["+core.brain.approved"]\n')
    (root / "child" / "ok.py").write_text("import core.brain.approved.thing\n")
    (root / "child" / "bad.py").write_text("import core.brain.other\n")
    violations = scan(root, package_root=root)
    assert [v.imported for v in violations] == ["core.brain.other"]


def test_relative_imports_are_never_layering_breaks(tmp_path):
    from tools.check_layering import scan

    package = tmp_path / "pkg"
    package.mkdir()
    (package / "DEPS").write_text('include_rules = ["-core"]\n')
    (package / "m.py").write_text("from . import sibling\n")
    assert scan(package, package_root=package) == []


def test_the_baseline_file_is_well_formed():
    payload = json.loads(
        (PROJECT_ROOT / "config" / "layering_baseline.json").read_text(encoding="utf-8")
    )
    assert payload["count"] == len(payload["grandfathered"])
    assert all("::" in entry for entry in payload["grandfathered"])


# ── invariants ────────────────────────────────────────────────────────

def test_observability_invariants_registered_and_clean():
    from core.verify import runtime_invariants  # noqa: F401
    from core.verify.invariants import get_registry, verify

    names = {s.name for s in get_registry().specs()}
    for expected in (
        "histograms.are_not_clipping",
        "memory.attribution_is_meaningful",
        "trials.have_hypotheses_and_expire",
        "security.rule_of_two_holds",
    ):
        assert expected in names

    r2_mod.install_known_handlers()
    report = verify("observability", record=False)
    assert report.ok, report.summary()
