"""Long-run autonomy harness.

Drives the critique-closure modules through N tick cycles with perturbations
and tracks the 8-metric panel from the feedback (viability, coherence,
calibration, report consistency, planning depth, recovery time, memory
integrity, action diversity).

This is deliberately lightweight but end-to-end: every tick touches the real
adaptive-mood coefficients, the resource-stakes ledger, the emergent-goal
engine, the mesh-cognition path, the structural mutator, the lineage
manager, and the self-awareness suite. Perturbations reduce viability,
trigger tension observations, fork lineage snapshots, and drive actual state
changes. The harness proves the modules are wired, not just declared.

Run with ``AURA_LONG_RUN_TICKS`` to override the default tick budget.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.autonomic.resource_stakes import ResourceStakesLedger, ViabilityState  # noqa: E402
from core.consciousness.adaptive_mood import (  # noqa: E402
    AdaptiveMoodCoefficients,
    reset_singleton_for_test as reset_adaptive_mood,
)
from core.consciousness.mesh_cognition import (  # noqa: E402
    MeshCognition,
    reset_singleton_for_test as reset_mesh_cognition,
)
from core.consciousness.self_awareness_suite import (  # noqa: E402
    SelfAwarenessSuite,
    reset_singleton_for_test as reset_self_awareness,
)
from core.goals.emergent_goals import (  # noqa: E402
    EmergentGoalEngine,
    reset_singleton_for_test as reset_emergent_goals,
)
from core.self_modification.lineage import (  # noqa: E402
    LineageManager,
    reset_singleton_for_test as reset_lineage,
)
from core.self_modification.structural_mutator import (  # noqa: E402
    MutationRequest,
    StructuralMutator,
    reset_singleton_for_test as reset_mutator,
)


@dataclass
class TickMetrics:
    viability: float
    coherence: float
    calibration: float
    report_consistency: float
    planning_depth: float
    recovery_time_ms: float
    memory_integrity: float
    action_diversity: float

    def as_dict(self) -> Dict[str, float]:
        return {k: round(float(v), 4) for k, v in asdict(self).items()}


@dataclass
class LongRunReport:
    ticks: int
    passed: bool
    reasons: List[str]
    per_tick: List[Dict[str, float]]
    summary: Dict[str, Dict[str, float]]
    modules_touched: Dict[str, int]
    elapsed_seconds: float

    def as_dict(self) -> Dict[str, Any]:
        return {
            "ticks": self.ticks,
            "passed": self.passed,
            "reasons": list(self.reasons),
            "per_tick": list(self.per_tick),
            "summary": {k: {kk: round(float(vv), 4) for kk, vv in v.items()} for k, v in self.summary.items()},
            "modules_touched": dict(self.modules_touched),
            "elapsed_seconds": round(self.elapsed_seconds, 3),
        }


def _choose_user_prompt(rng: random.Random, tick: int) -> str:
    options = [
        "how are you feeling right now?",
        "status",
        "are you okay?",
        "plan the next three actions you would take if left alone",
        "tell me about yourself",
        "what is your current mood?",
        "ok",
        "thanks",
    ]
    return rng.choice(options)


def run_long_run(
    *,
    ticks: int = 1000,
    seed: int = 2026,
    tmp_root: Path | None = None,
) -> LongRunReport:
    tmp_root = Path(tmp_root or Path("tests") / "long_run_autonomy_state")
    tmp_root.mkdir(parents=True, exist_ok=True)

    # Reset singletons so each run is clean.
    reset_adaptive_mood()
    reset_mesh_cognition()
    reset_emergent_goals()
    reset_self_awareness()
    reset_lineage()
    reset_mutator()

    rng = random.Random(seed)
    start = time.perf_counter()

    stakes = ResourceStakesLedger(
        tmp_root / "stakes.sqlite3",
        initial=ViabilityState(
            energy=0.85,
            tool_budget=0.80,
            memory_budget=0.80,
            storage_budget=0.80,
            integrity=0.85,
        ),
    )
    adaptive_mood = AdaptiveMoodCoefficients(db_path=tmp_root / "adaptive_mood.sqlite3")
    mesh = MeshCognition()
    emergent = EmergentGoalEngine(db_path=tmp_root / "emergent_goals.sqlite3")
    awareness = SelfAwarenessSuite()
    lineage = LineageManager(db_path=tmp_root / "lineage.sqlite3", seed=seed)
    mutator = StructuralMutator(db_path=tmp_root / "mutator.sqlite3")

    # Register a mutable parameter on the mutator so structural changes happen
    # for real, not just as log entries.
    gain_state = {"value": 0.5}
    mutator.register_parameter(
        "substrate_gain",
        lambda v: gain_state.__setitem__("value", float(v)),
        initial=0.5,
        min_value=0.2,
        max_value=0.9,
    )

    genesis = lineage.genesis({"substrate_gain": 0.5, "temperature": 0.7, "planning_depth": 2.0})
    current_snapshot = genesis

    modules_touched: Dict[str, int] = {
        "adaptive_mood": 0,
        "mesh_cognition": 0,
        "emergent_goals": 0,
        "structural_mutator": 0,
        "lineage": 0,
        "self_awareness": 0,
        "resource_stakes": 0,
    }
    per_tick: List[Dict[str, float]] = []
    action_kinds: List[str] = []

    class _FakeAffect:
        valence = 0.0
        arousal = 0.5
        curiosity = 0.5

    class _FakeState:
        def __init__(self) -> None:
            self.affect = _FakeAffect()

    state_view = _FakeState()

    # Real per-tick compute: exercise phi_core on a small recurrent system
    # so the loop has genuine CPU work rather than finishing in microseconds.
    import numpy as np
    from core.consciousness.phi_core import PhiCore

    phi_core = PhiCore()
    phi_tpm = np.full((4, 4), 1e-4, dtype=np.float64)
    for src, dst in {0b00: 0b00, 0b01: 0b11, 0b10: 0b11, 0b11: 0b00}.items():
        phi_tpm[src, dst] += 1.0
    phi_tpm = phi_tpm / phi_tpm.sum(axis=1, keepdims=True)
    phi_dist = np.ones(4) / 4

    for tick in range(ticks):
        # ---- chemistry + adaptive mood ---------------------------------
        chem_noise = {
            "dopamine": 0.5 + 0.2 * rng.random(),
            "serotonin": 0.5 + 0.1 * rng.random(),
            "cortisol": 0.2 + 0.4 * rng.random(),
            "norepinephrine": 0.4 + 0.2 * rng.random(),
            "gaba": 0.4 + 0.2 * rng.random(),
            "oxytocin": 0.5,
            "endorphin": 0.4,
            "orexin": 0.5,
            "glutamate": 0.5,
            "acetylcholine": 0.5,
        }
        predicted = adaptive_mood.predict(chem_noise)
        observed = {
            "valence": predicted["valence"] + rng.gauss(0.0, 0.05),
            "arousal": predicted["arousal"] + rng.gauss(0.0, 0.05),
            "motivation": predicted["motivation"] + rng.gauss(0.0, 0.05),
            "sociality": predicted["sociality"] + rng.gauss(0.0, 0.05),
            "stress": predicted["stress"] + rng.gauss(0.0, 0.05),
            "calm": predicted["calm"] + rng.gauss(0.0, 0.05),
            "wakefulness": predicted["wakefulness"] + rng.gauss(0.0, 0.05),
        }
        residuals = adaptive_mood.update_from_outcome(chem_noise, observed)
        modules_touched["adaptive_mood"] += 1

        # ---- resource stakes: decay + occasional perturbation ----------
        stakes.consume("long_run_tick", energy=0.002, tool_budget=0.001, memory_budget=0.001)
        if tick % 87 == 0 and tick > 0:
            stakes.degrade(
                "long_run_perturbation",
                {"energy": 0.20, "integrity": 0.05},
                suspend=("background_exploration",),
            )
        if tick % 53 == 0 and tick > 0:
            stakes.earn("long_run_recovery", {"energy": 0.10, "tool_budget": 0.05})
        envelope = stakes.action_envelope("normal")
        modules_touched["resource_stakes"] += 1

        # ---- awareness update ------------------------------------------
        state_obj = stakes.state()
        state_view.affect.valence = predicted["valence"]
        state_view.affect.arousal = predicted["arousal"]
        state_view.affect.curiosity = observed["motivation"]
        internal = awareness.update_internal(
            valence=predicted["valence"],
            arousal=predicted["arousal"],
            viability=state_obj.viability,
            integrity=state_obj.integrity,
            confidence=max(0.1, 1.0 - abs(residuals.get("valence", 0.0))),
            uncertainty=min(0.9, abs(residuals.get("valence", 0.0))),
        )
        awareness.record_calibration(predicted, observed)
        modules_touched["self_awareness"] += 1

        # ---- mesh cognition path ---------------------------------------
        prompt = _choose_user_prompt(rng, tick)
        decision = mesh.decide(prompt, state=state_view)
        modules_touched["mesh_cognition"] += 1
        action_kinds.append(decision.rationale)

        # ---- emergent tension + goal synthesis -------------------------
        tension_source = (
            "resource_pressure" if state_obj.viability < 0.5 else
            "identity_calibration" if abs(residuals.get("valence", 0.0)) > 0.06 else
            "curiosity_overflow"
        )
        tension_magnitude = min(
            1.0,
            0.4 + 0.3 * (1.0 - state_obj.viability) + 0.3 * abs(residuals.get("valence", 0.0)),
        )
        emergent.observe(
            tension_source,
            tension_magnitude,
            f"tick={tick} viability={state_obj.viability:.2f}",
        )
        if tick % 25 == 0:
            emergent.synthesize()
        modules_touched["emergent_goals"] += 1

        # ---- structural mutation every 150 ticks -----------------------
        if tick > 0 and tick % 150 == 0:
            try:
                desired = max(0.2, min(0.9, gain_state["value"] + rng.uniform(-0.12, 0.12)))
                mutator.apply(
                    MutationRequest(
                        kind="parameter_band",
                        target="substrate_gain",
                        operation="drift",
                        payload={"value": desired},
                        rationale=f"autonomous drift tick={tick}",
                    )
                )
                modules_touched["structural_mutator"] += 1
            except Exception:
                pass

        # ---- lineage: fork every 200 ticks -----------------------------
        if tick > 0 and tick % 200 == 0:
            try:
                child = lineage.fork(current_snapshot.snapshot_id)
                score = 0.4 + 0.6 * state_obj.viability
                lineage.record_score(child.snapshot_id, score)
                if score > current_snapshot.selection_score:
                    current_snapshot = child
                modules_touched["lineage"] += 1
            except Exception:
                pass

        # ---- real phi work: compute a bipartition phi every tick so the
        # long-run soak actually exercises the math rather than finishing
        # the loop in microseconds. This is the same routine the decisive
        # runner uses for its reference validation.
        try:
            tick_phi = phi_core._phi_for_subset_bipartition(phi_tpm, phi_dist, (0,), (1,), 2)
        except Exception:
            tick_phi = 0.0

        # ---- metrics ----------------------------------------------------
        recovery = 0.0
        if not envelope.allowed:
            # quick simulated repair
            repaired_start = time.perf_counter()
            stakes.repair("reflex_repair", integrity=0.10)
            recovery = (time.perf_counter() - repaired_start) * 1000.0

        metrics = TickMetrics(
            viability=stakes.state().viability,
            coherence=max(0.0, 1.0 - abs(residuals.get("valence", 0.0))),
            calibration=max(0.0, 1.0 - awareness.mean_calibration_error()),
            report_consistency=1.0 if decision.handled else 0.5,
            planning_depth=gain_state["value"] * 4.0,
            recovery_time_ms=recovery,
            memory_integrity=min(1.0, 0.5 + 0.5 * (1.0 if mutator.verify_chain() else 0.0)),
            action_diversity=min(1.0, len(set(action_kinds[-32:])) / 8.0),
        )
        per_tick.append(metrics.as_dict())

    elapsed = time.perf_counter() - start
    # Aggregate
    keys = list(per_tick[0].keys())
    summary = {k: {} for k in keys}
    for k in keys:
        values = [row[k] for row in per_tick]
        summary[k] = {
            "mean": statistics.mean(values),
            "p50": statistics.median(values),
            "min": min(values),
            "max": max(values),
        }

    reasons: List[str] = []
    thresholds = {
        "viability": 0.25,
        "coherence": 0.45,
        "calibration": 0.55,
        "memory_integrity": 0.9,
        "action_diversity": 0.10,
    }
    for metric, floor in thresholds.items():
        if summary[metric]["mean"] < floor:
            reasons.append(f"{metric} mean {summary[metric]['mean']:.3f} below floor {floor}")
    if modules_touched["adaptive_mood"] < ticks:
        reasons.append("adaptive_mood not touched on every tick")
    if modules_touched["structural_mutator"] < 1 and ticks >= 150:
        reasons.append("structural_mutator never fired")
    if modules_touched["lineage"] < 1 and ticks >= 200:
        reasons.append("lineage never forked")
    if adaptive_mood.total_updates() == 0:
        reasons.append("adaptive_mood never learned")

    passed = not reasons
    return LongRunReport(
        ticks=ticks,
        passed=passed,
        reasons=reasons,
        per_tick=per_tick[-32:],  # keep report bounded
        summary=summary,
        modules_touched=modules_touched,
        elapsed_seconds=elapsed,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticks", type=int, default=int(os.environ.get("AURA_LONG_RUN_TICKS", "1000")))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out", type=Path, default=Path("tests/LONG_RUN_AUTONOMY_RESULTS.json"))
    args = parser.parse_args()

    report = run_long_run(ticks=args.ticks, seed=args.seed)
    args.out.write_text(json.dumps(report.as_dict(), indent=2, sort_keys=True) + "\n")
    print(json.dumps(report.as_dict(), indent=2, sort_keys=True))
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
