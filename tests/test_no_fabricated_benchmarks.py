"""No benchmark may print a score it did not measure.

``aura_bench/ablations/runner.py`` defined "empirical scorecards" as dict
literals (raw_model 0.42 … full_aura 0.94) and printed them as an "ABLATION
SUITE" result. Nothing was executed. It had already been superseded by the real
harness, but the file stayed in the tree — where it would be used against the
project, and rightly so.

This test guards the deletion and the shape of the mistake, so a fabricated
scorecard cannot quietly reappear under another name.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# Metric-shaped names that must never be assigned a hardcoded constant at the
# top level of a benchmark module.
_METRIC_NAMES = {
    "aletheia_score",
    "score",
    "recovery",
    "policy",
    "tool_invention",
    "transfer",
    "accuracy",
    "pass_rate",
    "mean_score",
}

_BENCHMARK_DIRS = ("aura_bench", "evaluation")

# Modules that legitimately contain metric constants (thresholds, fixtures, and
# answer keys are not fabricated results). Keep this list short and justified;
# it should shrink, not grow.
_ALLOWLIST = {
    "aura_bench/property_tests",
    "aura_bench/hard_suite.py",
}


def test_the_fabricated_ablation_runner_is_gone():
    """It printed 0.42 for the raw model and 0.94 for full Aura, having run nothing."""
    assert not (ROOT / "aura_bench" / "ablations" / "runner.py").exists(), (
        "the fabricated ablation runner is back"
    )


def test_a_real_ablation_runner_exists_instead():
    """Deleting theatre must not delete the capability it pretended to have."""
    assert (ROOT / "tools" / "ablation_runner.py").exists(), (
        "no real ablation runner — deleting the fake one removed the capability"
    )


#: The numbers the deleted runner printed. Deleting the source that made them
#: did not delete the JSON it had already written: five byte-identical copies
#: of the same dict sat under artifacts/certification/latest/ for five weeks,
#: and EVALUATE_AURA.md linked one of them to external reviewers as
#: "quantitative baseline comparisons proving each module is causally
#: load-bearing". A test over Python source cannot see a committed artifact.
_FABRICATED_SCORES = {"raw_model": 0.42, "full_aura": 0.94}


def test_no_committed_artifact_carries_the_fabricated_scorecard():
    """The deleted runner's output must not survive the deleted runner."""
    import json

    offenders = []
    for path in (ROOT / "artifacts").rglob("*.json"):
        if path.stat().st_size > 2_000_000:
            continue
        try:
            payload = json.loads(path.read_text(errors="replace"))
        except (ValueError, OSError):
            continue
        blocks = payload.get("ablations") if isinstance(payload, dict) else None
        if not isinstance(blocks, dict):
            continue
        for name, expected in _FABRICATED_SCORES.items():
            entry = blocks.get(name)
            if isinstance(entry, dict) and entry.get("aletheia_score") == expected:
                offenders.append(f"{path.relative_to(ROOT)} ({name}={expected})")
    assert not offenders, (
        "fabricated ablation scorecard found in committed artifacts:\n  "
        + "\n  ".join(sorted(offenders))
    )


def test_the_soak_simulator_is_gone():
    """It wrote 4h, 24h and 72h "audit-grade telemetry" from random.seed().

    All three completed within six milliseconds of each other, and
    EVALUATE_AURA.md offered them for verifying resource stability.
    tools/longevity/run_longevity_soak.py measures a real one.
    """
    assert not (ROOT / "tools" / "generate_soak_logs.py").exists(), (
        "the soak simulator is back"
    )
    assert (ROOT / "tools" / "longevity" / "run_longevity_soak.py").exists(), (
        "no real longevity soak — deleting the simulator removed the capability"
    )
    survivors = sorted(
        p.relative_to(ROOT).as_posix()
        for p in (ROOT / "artifacts").rglob("SOAK_LOG_*.json")
    )
    assert not survivors, f"simulated soak telemetry still committed: {survivors}"


def test_certification_runs_the_real_ablation_runner():
    """The gate pointed at the deleted file and failed on it every run."""
    source = (ROOT / "tools" / "certify.py").read_text()
    assert "aura_bench/ablations/runner.py" not in source, (
        "certify.py still invokes the deleted fabricated runner"
    )
    assert "tools/ablation_runner.py" in source, (
        "certify.py no longer runs any ablation gate"
    )


def _iter_benchmark_modules():
    for d in _BENCHMARK_DIRS:
        base = ROOT / d
        if not base.exists():
            continue
        for path in base.rglob("*.py"):
            rel = path.relative_to(ROOT).as_posix()
            if any(rel.startswith(a) for a in _ALLOWLIST):
                continue
            if "__pycache__" in rel:
                continue
            yield path, rel


@pytest.mark.parametrize("_ignored", [None])
def test_no_benchmark_hardcodes_a_metric_dict(_ignored):
    """Catch the exact shape: a dict literal of metric names → float constants.

    A benchmark result must come from running something. A dict literal mapping
    ``aletheia_score`` to ``0.94`` is a claim with no measurement behind it.
    """
    offenders: list[str] = []

    for path, rel in _iter_benchmark_modules():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (SyntaxError, UnicodeDecodeError):
            continue

        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict):
                continue
            metric_constants = 0
            for key, value in zip(node.keys, node.values):
                if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
                    continue
                if key.value not in _METRIC_NAMES:
                    continue
                if isinstance(value, ast.Constant) and isinstance(
                    value.value, (int, float)
                ):
                    metric_constants += 1
            if metric_constants >= 2:
                offenders.append(f"{rel}:{node.lineno}")

    assert not offenders, (
        "benchmark modules contain hardcoded metric scorecards (a result with no "
        "measurement behind it): " + ", ".join(offenders)
    )
