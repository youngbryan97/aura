"""SLO measurement harness — pure CPU, no model loads.

Each measurement function returns a number plus a unit, suitable for
comparison against ``slo/baseline.json``.  All measurements are
deterministic enough on a quiet box that the soft tolerance in the
baseline absorbs normal jitter.

Run with ``python -m slo.measure`` to see the current numbers, or with
``--emit`` to print a baseline-shaped JSON document to stdout.
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List


def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    if pct <= 0:
        return min(values)
    if pct >= 100:
        return max(values)
    s = sorted(values)
    k = (len(s) - 1) * (pct / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


def _time_ms(fn: Callable[[], Any]) -> float:
    t0 = time.perf_counter()
    fn()
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0


# ---------------------------------------------------------------------------
# audit chain
# ---------------------------------------------------------------------------
def measure_audit_chain_append_p95_ms(samples: int = 200) -> float:
    from core.runtime.audit_chain import AuditChain

    with tempfile.TemporaryDirectory() as tmp:
        chain = AuditChain(Path(tmp))
        latencies = []
        for i in range(samples):
            latencies.append(
                _time_ms(
                    lambda i=i: chain.append(
                        receipt_id=f"r-{i}",
                        kind="turn",
                        body={"i": i, "data": "x" * 64},
                        timestamp=time.time(),
                    )
                )
            )
        return _percentile(latencies, 95.0)


def measure_audit_chain_verify_per_entry_us(samples: int = 500) -> float:
    from core.runtime.audit_chain import AuditChain

    with tempfile.TemporaryDirectory() as tmp:
        chain = AuditChain(Path(tmp))
        for i in range(samples):
            chain.append(
                receipt_id=f"r-{i}",
                kind="turn",
                body={"i": i},
                timestamp=time.time(),
            )
        t0 = time.perf_counter()
        ok, problems = chain.verify()
        elapsed = time.perf_counter() - t0
        assert ok, f"chain unexpectedly failed verification: {problems[:3]}"
        return (elapsed / samples) * 1_000_000.0


# ---------------------------------------------------------------------------
# prediction ledger
# ---------------------------------------------------------------------------
def measure_prediction_ledger_register_p95_ms(samples: int = 200) -> float:
    from core.runtime.prediction_ledger import PredictionLedger

    with tempfile.TemporaryDirectory() as tmp:
        ledger = PredictionLedger(Path(tmp) / "ledger.db")
        latencies = []
        for i in range(samples):
            latencies.append(
                _time_ms(
                    lambda i=i: ledger.register(
                        belief=f"b{i}",
                        modality="text",
                        action="probe",
                        expected={"i": i},
                        prior_prob=0.5,
                    )
                )
            )
        return _percentile(latencies, 95.0)


def measure_prediction_ledger_brier_correctness() -> float:
    """Synthetic perfectly calibrated predictor: Brier should be 0."""
    from core.runtime.prediction_ledger import PredictionLedger

    with tempfile.TemporaryDirectory() as tmp:
        ledger = PredictionLedger(Path(tmp) / "ledger.db")
        # Predict True with prior=1.0 and observe truth=True; Brier=0 each.
        for i in range(50):
            pid = ledger.register(
                belief=f"b{i}",
                modality="text",
                action="probe",
                expected={"i": i},
                prior_prob=1.0,
            )
            ledger.resolve(pid, observed={"truth": True}, observed_truth=True)
        return float(ledger.score_brier()["mean_brier"] or 0.0)


# ---------------------------------------------------------------------------
# mutation evaluator
# ---------------------------------------------------------------------------
def measure_mutation_eval_passed_p95_ms(samples: int = 30) -> float:
    """Trivial passing mutation; bounds spawn + bootstrap + import overhead."""
    from core.self_modification.mutation_safety import (
        QuarantineStore,
        SafeMutationEvaluator,
    )

    with tempfile.TemporaryDirectory() as tmp:
        evaluator = SafeMutationEvaluator(
            timeout_seconds=10.0,
            memory_mb=256,
            quarantine=QuarantineStore(Path(tmp)),
        )
        latencies = []
        for _ in range(samples):
            t0 = time.perf_counter()
            diag = evaluator.evaluate("x = 1\n")
            t1 = time.perf_counter()
            assert diag.outcome.value == "passed"
            latencies.append((t1 - t0) * 1000.0)
        return _percentile(latencies, 95.0)


# ---------------------------------------------------------------------------
# diagnostics bundle
# ---------------------------------------------------------------------------
def measure_doctor_bundle_p95_ms(samples: int = 10, warmup: int = 2) -> float:
    """Bundle is heavy and variance-prone; warm up to amortize the
    first-call import+init cost (~10× steady-state), then sample p95."""
    from core.runtime.diagnostics_bundle import build_bundle

    with tempfile.TemporaryDirectory() as tmp:
        for i in range(warmup):
            out = Path(tmp) / f"warmup_{i}.tar.gz"
            ws = Path(tmp) / f"warmup_ws_{i}"
            info = build_bundle(output_path=out, workspace=ws)
            assert info["ok"] is True

        latencies = []
        for i in range(samples):
            out = Path(tmp) / f"bundle_{i}.tar.gz"
            ws = Path(tmp) / f"ws_{i}"
            t0 = time.perf_counter()
            info = build_bundle(output_path=out, workspace=ws)
            t1 = time.perf_counter()
            assert info["ok"] is True
            latencies.append((t1 - t0) * 1000.0)
        return _percentile(latencies, 95.0)


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------
MEASUREMENTS: Dict[str, Dict[str, Any]] = {
    "audit_chain_append_p95_ms": {
        "fn": measure_audit_chain_append_p95_ms,
        "unit": "ms",
    },
    "audit_chain_verify_per_entry_us": {
        "fn": measure_audit_chain_verify_per_entry_us,
        "unit": "us",
    },
    "prediction_ledger_register_p95_ms": {
        "fn": measure_prediction_ledger_register_p95_ms,
        "unit": "ms",
    },
    "prediction_ledger_brier_correctness": {
        "fn": measure_prediction_ledger_brier_correctness,
        "unit": "score",
    },
    "mutation_eval_passed_p95_ms": {
        "fn": measure_mutation_eval_passed_p95_ms,
        "unit": "ms",
    },
    "doctor_bundle_p95_ms": {
        "fn": measure_doctor_bundle_p95_ms,
        "unit": "ms",
    },
}


def run_all() -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for name, spec in MEASUREMENTS.items():
        value = float(spec["fn"]())
        out[name] = {"value": value, "unit": spec["unit"]}
    return out


def emit_baseline(measured: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    # Per-SLO tolerance: the doctor bundle is heavy file-I/O and runs
    # on whatever the host box is doing at the time, so we widen its
    # tolerance.  All other surfaces are tight numerics on small data.
    tolerance_overrides = {
        "doctor_bundle_p95_ms": 200,
        "mutation_eval_passed_p95_ms": 100,  # subprocess spawn overhead varies
    }
    return {
        "schema_version": 1,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "platform": f"{platform.system().lower()}-{platform.machine()}-py{platform.python_version()}",
        "slos": {
            name: {
                "value": measured[name]["value"],
                "unit": measured[name]["unit"],
                "tolerance_pct": tolerance_overrides.get(name, 50),
            }
            for name in MEASUREMENTS
        },
        "hard_limits": {
            "doctor_bundle_p95_ms": 5000,
            "audit_chain_append_p95_ms": 50,
            "prediction_ledger_brier_correctness": 0.001,
        },
    }


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--emit",
        action="store_true",
        help="Emit a baseline-shaped JSON document instead of a human report.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Write the JSON output to this path instead of stdout (avoids "
        "importer log lines polluting stdout).",
    )
    args = parser.parse_args(argv)

    measured = run_all()

    if args.emit:
        payload = emit_baseline(measured)
        text = json.dumps(payload, indent=2) + "\n"
        if args.out:
            Path(args.out).write_text(text, encoding="utf-8")
        else:
            sys.stdout.write(text)
    else:
        for name, m in measured.items():
            print(f"  {name:48s} {m['value']:>10.4f} {m['unit']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
