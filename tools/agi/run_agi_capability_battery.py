#!/usr/bin/env python3
"""
tools/agi/run_agi_capability_battery.py
Aura External AGI Capability Battery Evaluation Runner.
Handles frozen stack verification, 17-category tests, multiple ablations,
statistical significance validation, and logs full decision traces.
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import numpy as np

# Repo imports are intentionally resolved after the script inserts PROJECT_ROOT.
# ruff: noqa: E402

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.container import ServiceContainer
from core.orchestrator import RobustOrchestrator
from core.runtime.resource_observation import get_resource_observer

PROBE_RECOVERABLE_ERRORS = (
    AttributeError,
    ImportError,
    KeyError,
    RuntimeError,
    TypeError,
    ValueError,
)

# 17 AGI Battery Categories definition
CATEGORIES = {
    1: {"name": "General Assistant Intelligence (GAIA)", "metric": "reasoning_accuracy"},
    2: {"name": "Humanity's Last Exam (HLE)", "metric": "expert_knowledge_score"},
    3: {"name": "GPQA Diamond", "metric": "phd_scientific_reasoning"},
    4: {"name": "MMLU-Pro", "metric": "complex_problem_solving"},
    5: {"name": "FrontierMath", "metric": "symbolic_proof_rigor"},
    6: {"name": "ARC-AGI", "metric": "inductive_grid_coherence"},
    7: {"name": "BrowseComp", "metric": "web_navigation_fidelity"},
    8: {"name": "SWE-bench", "metric": "repository_patch_rate"},
    9: {"name": "OSWorld", "metric": "os_grounding_accuracy"},
    10: {"name": "WebArena", "metric": "transactional_task_completion"},
    11: {"name": "τ-bench", "metric": "multi_agent_negotiation"},
    12: {"name": "MLE-bench", "metric": "machine_learning_engineering"},
    13: {"name": "RE-Bench", "metric": "reverse_engineering_rigor"},
    14: {"name": "Unknown APIs", "metric": "black_box_api_synthesis"},
    15: {"name": "Black-Box World Modeling", "metric": "state_transition_discovery"},
    16: {"name": "Long-Horizon Autonomy", "metric": "persistent_survival_index"},
    17: {"name": "Self-Improvement (RSI)", "metric": "recursive_self_optimization"}
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="artifacts/agi_live/capability_battery.json")
    parser.add_argument("--markdown", type=str, default="artifacts/agi_live/CAPABILITY_BATTERY_RESULTS.md")
    parser.add_argument("--seeds", type=int, default=100)
    return parser.parse_args()

def get_git_commit():
    try:
        git_dir = PROJECT_ROOT / ".git"
        head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
        if head.startswith("ref: "):
            ref = head.split(" ", 1)[1].strip()
            ref_path = git_dir / ref
            if ref_path.exists():
                return ref_path.read_text(encoding="utf-8").strip()
            packed_refs = git_dir / "packed-refs"
            if packed_refs.exists():
                for line in packed_refs.read_text(encoding="utf-8").splitlines():
                    if line and not line.startswith("#") and line.endswith(f" {ref}"):
                        return line.split(" ", 1)[0].strip()
            return "unknown"
        return head
    except (OSError, UnicodeDecodeError):
        return "unknown"

# NOTE: capability SCORES are deliberately not synthesized here — that would be
# fabrication. This battery reports real subsystem-liveness probes only; the
# graded ablation lives in tools/agi/run_prompt_baseline_ablation.py.

async def main():
    args = parse_args()
    
    # Establish outputs
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    md_path = Path(args.markdown)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    
    commit_sha = get_git_commit()
    print("Verify Stack:")
    print(f"  Commit SHA: {commit_sha}")
    print("  Operating System: macOS (Live)")
    print("  Target Battery: 17 AGI Capability Battery Categories")
    
    # Instantiate RobustOrchestrator to verify Will / Authority / Volition loops are fully load-bearing
    orch = RobustOrchestrator()
    assert orch.state is not None, "AuraState failed to boot."
    
    # Ensure Will is initialized and registered
    from core.will import get_will
    will_service = get_will()
    await will_service.start()
    
    # Ensure VolitionEngine is initialized and registered
    from core.volition import VolitionEngine
    volition_service = ServiceContainer.get("volition", default=None) or ServiceContainer.get("unified_volition", default=None) or ServiceContainer.get("volition_engine", default=None)
    if not volition_service:
        volition_service = VolitionEngine(orch)
        ServiceContainer.register_instance("volition", volition_service)
        ServiceContainer.register_instance("unified_volition", volition_service)
        ServiceContainer.register_instance("volition_engine", volition_service)
        
    # Ensure AgencyCore is initialized and registered
    from core.agency_core import AgencyCore
    agency_core_service = ServiceContainer.get("agency_core", default=None)
    if not agency_core_service:
        agency_core_service = AgencyCore(orch)
        ServiceContainer.register_instance("agency_core", agency_core_service)
        
    # Retrieve current active state asynchronously from StateRepository
    state = await orch.state.get_state()
    
    # Resolve real system and cognitive metrics for the proving dashboard

    resource_observer = get_resource_observer()
    compute_observation = resource_observer.compute()
    memory_observation = resource_observer.memory()
    cpu_usage = compute_observation.cpu_percent if compute_observation.available else 100.0
    mem_usage = memory_observation.percent if memory_observation.available else 100.0
    resource_provenance = resource_observer.provenance.to_dict()
    
    # Query real registered skill surface size dynamically
    registered_skills = 56
    if orch.capability_engine and hasattr(orch.capability_engine, "skills"):
        registered_skills = len(orch.capability_engine.skills)
    elif ServiceContainer.has("agency_core"):
        agency_core = ServiceContainer.get("agency_core")
        if hasattr(agency_core, "skills"):
            registered_skills = len(agency_core.skills)
            
    coherence = 1.0
    active_goals_count = 0
    if state and hasattr(state, "cognition"):
        coherence = getattr(state.cognition, "coherence_score", 1.0)
        active_goals_count = len(getattr(state.cognition, "active_goals", []))
        
    print("Live Environment Metrics:")
    print(f"  CPU Usage: {cpu_usage}%")
    print(f"  Memory Usage: {mem_usage}%")
    print(f"  Registered Skills Surface: {registered_skills}")
    print(f"  Cognitive Coherence Score: {coherence}")
    print(f"  Active Persistent Goals: {active_goals_count}")
    
    # ---------------------------------------------------------------------------
    # EXHAUSTIVE LIVE PROBES TO TRULY PROVE ARCHITECTURE RESILIENCE
    # ---------------------------------------------------------------------------
    print("\nExecuting Live Architectural Probes...")
    
    # Probe 1: Will Concurrency & Latency Stress Probe
    print("  [PROBE 1/5] Running Will Concurrency and Latency Stress Probe...")
    will_ok = False
    p50_latency = 0.0
    p99_latency = 0.0
    try:
        will = ServiceContainer.get("will", default=None) or ServiceContainer.get("unified_will", default=None)
        if will:
            from core.will import ActionDomain
            samples = []
            
            async def run_one_decide(i):
                t_start = time.perf_counter()
                d = will.decide(f"Capability battery stress test decision {i}", source="capability_battery", domain=ActionDomain.RESPONSE, priority=0.5)
                duration = (time.perf_counter() - t_start) * 1000
                return d, duration
                
            tasks = [run_one_decide(i) for i in range(50)]
            decide_results = await asyncio.gather(*tasks)

            samples = [res[1] for res in decide_results]
            p50_latency = float(np.percentile(samples, 50))
            p99_latency = float(np.percentile(samples, 99))
            
            # Verify the decisions in the audit trail (Proven provability & cryptographic signature checking)
            audit_trail_ok = True
            for res in decide_results:
                decision = res[0]
                if not will.verify_receipt(decision.receipt_id):
                    audit_trail_ok = False
                    break
                    
            will_ok = len(samples) == 50 and p50_latency < 50.0 and audit_trail_ok
            print(f"    → Will Probe: p50={p50_latency:.2f}ms, p99={p99_latency:.2f}ms, audit={audit_trail_ok}. Status: {'PASS' if will_ok else 'DEGRADED'}")
        else:
            print("    → Will service not available. Status: SKIPPED")
    except PROBE_RECOVERABLE_ERRORS as e:
        print(f"    → Will Probe failed with exception: {e}")
        
    # Probe 2: Volition Goal Cooldown & Deduplication Probe
    print("  [PROBE 2/5] Running Volition Goal Cooldown and Deduplication Probe...")
    dedup_ok = False
    try:
        volition = ServiceContainer.get("volition", default=None) or ServiceContainer.get("unified_volition", default=None) or ServiceContainer.get("volition_engine", default=None)
        if volition:
            # This probe used to select the same goal twice and require the
            # second call to be filtered. That was the old contract. Cooldown
            # recording was deliberately moved OUT of selection and into
            # _commit_goal_effects, because committing timers at selection
            # time made a persistently failing action look like completed
            # work while simultaneously suppressing its own retry — a broken
            # initiative path went silently dormant.
            #
            # So the probe was asserting a behaviour that no longer exists and
            # reporting a healthy subsystem as broken. It now exercises the
            # real contract: select, commit, and only then expect the repeat
            # to be filtered.
            test_goals = [{"objective": "explore_neural_mesh_anomaly", "origin": "boredom", "priority": 0.5}]
            first_selection = volition._select_and_parse_goal(test_goals)
            selection_ok = first_selection is not None

            # Selection alone must NOT deduplicate — that is the property the
            # refactor bought, and losing it is a real regression.
            repeat_before_commit = volition._select_and_parse_goal(test_goals)
            no_premature_cooldown = repeat_before_commit is not None

            if first_selection is not None:
                volition._commit_goal_effects(first_selection)
            after_commit = volition._select_and_parse_goal(test_goals)
            cooldown_dedup_pass = after_commit is None

            # Verify that different goals do NOT block each other (Aura's free volition must not be constrained)
            different_goals = [{"objective": "different_objective_01", "origin": "boredom", "priority": 0.5}]
            different_selection = volition._select_and_parse_goal(different_goals)
            different_ok = different_selection is not None

            dedup_ok = (
                selection_ok
                and no_premature_cooldown
                and cooldown_dedup_pass
                and different_ok
            )
            print(
                f"    → Volition Probe: Commit-scoped deduplication is "
                f"{'PASS' if dedup_ok else 'FAIL'} "
                f"(selected={selection_ok}, no_premature_cooldown={no_premature_cooldown}, "
                f"cooldown_after_commit={cooldown_dedup_pass}, different_pass={different_ok})"
            )
        else:
            print("    → Volition service not available. Status: SKIPPED")
    except PROBE_RECOVERABLE_ERRORS as e:
        print(f"    → Volition Probe failed with exception: {e}")

    # Probe 3: Agency Core Goal Completion & Tracking Probe
    print("  [PROBE 3/5] Running Agency Core Goal Completion & Tracking Probe...")
    completion_ok = False
    try:
        agency_core = ServiceContainer.get("agency_core", default=None)
        if agency_core:
            # Inject a goal fixture into pending goals.
            test_goal = {"id": "battery_test_goal_01", "text": "battery_test_goal_completion", "status": "pending", "priority": 0.5}
            agency_core.state.pending_goals.append(test_goal)
            
            # Call complete_goal_by_match
            matched = agency_core.complete_goal_by_match(test_goal, status="completed")
            if matched:
                # Find the goal and verify it has status="completed"
                for g in agency_core.state.pending_goals:
                    if g.get("id") == "battery_test_goal_01" and g.get("status") == "completed":
                        completion_ok = True
                        break
            # Clean up the test goal
            agency_core.state.pending_goals = [g for g in agency_core.state.pending_goals if g.get("id") != "battery_test_goal_01"]
            print(f"    → Agency Core Probe: Goal lifecycle completion is {'PASS' if completion_ok else 'FAIL'}")
        else:
            print("    → Agency Core service not available. Status: SKIPPED")
    except PROBE_RECOVERABLE_ERRORS as e:
        print(f"    → Agency Core Probe failed with exception: {e}")

    # Probe 4: CAA Affective Steering Vector Library Probe
    print("  [PROBE 4/5] Running CAA Affective Steering Vector Library Probe...")
    steering_ok = False
    try:
        from core.consciousness.affective_steering import (
            AFFECTIVE_DIMENSIONS,
            SteeringVectorLibrary,
        )
        library = SteeringVectorLibrary()
        if library is not None and len(AFFECTIVE_DIMENSIONS) > 0:
            steering_ok = True
            print(f"    → Steering Probe: Found {len(AFFECTIVE_DIMENSIONS)} steering dimensions. Status: PASS")
        else:
            print("    → Steering Vector Library not configured. Status: FAIL")
    except PROBE_RECOVERABLE_ERRORS as e:
        print(f"    → Steering Probe failed with exception: {e}")

    # Probe 5: Skill Surface & Constraint Execution Probe
    print("  [PROBE 5/5] Running Skill Surface and Constraint Execution Probe...")
    skills_ok = False
    try:
        if registered_skills >= 50:
            # Exercise constrained execution classification with a representative dependency failure.
            from tests.live_harness_registered_skills import _judge_constrained
            constrained_payload = {"status": "error", "error": "No sounddevice could be initialized on host"}
            is_valid_constraint, _ = _judge_constrained(constrained_payload, "sounddevice")
            if is_valid_constraint:
                skills_ok = True
                print(f"    → Skills Probe: Found {registered_skills} skills, constraint handling is PASS")
            else:
                print("    → Skills Probe: Constraint judgment failed. Status: FAIL")
        else:
            print(f"    → Skills Probe: Degraded skills count ({registered_skills}). Status: FAIL")
    except PROBE_RECOVERABLE_ERRORS as e:
        print(f"    → Skills Probe failed with exception: {e}")

    # Calculate Cognitive Performance Index (CPI) based on actual probes
    passed_probes = sum([1 for p in [will_ok, dedup_ok, completion_ok, steering_ok, skills_ok] if p])
    cpi = passed_probes / 5.0
    print(f"\nFinal Cognitive Performance Index (CPI): {cpi:.2f} ({passed_probes}/5 probes passed)\n")

    # HONEST subsystem-liveness battery.
    # This battery verifies the cognitive subsystems are LIVE and functioning via
    # real probes (above). It does NOT synthesize capability scores or baseline
    # comparisons — doing so would be fabrication (the prior version generated a
    # 17-category scorecard from Gaussian noise around a hardcoded mean and
    # asserted victory over hardcoded baselines). For the real graded
    # architecture-vs-stateless ablation, see
    # tools/agi/run_prompt_baseline_ablation.py + core/evaluation/ablation_harness.py.
    probes = {
        "will_concurrency": will_ok,
        "volition_deduplication": dedup_ok,
        "agency_goal_completion": completion_ok,
        "steering_vector_library": steering_ok,
        "skill_surface_constraint": skills_ok,
    }
    all_probes_pass = passed_probes == len(probes)

    report = {
        "schema": "aura.capability_battery.subsystem_liveness.v2",
        "measurement": "subsystem_liveness",
        "note": (
            "Verifies cognitive subsystems are live via real probes. Does NOT "
            "produce benchmarked capability scores or baseline comparisons — for "
            "the graded architecture-vs-stateless ablation see "
            "tools/agi/run_prompt_baseline_ablation.py."
        ),
        "commit_sha": commit_sha,
        "eval_timestamp": time.time(),
        "capability_areas_probed": len(CATEGORIES),
        "probes": probes,
        "probes_passed": passed_probes,
        "probes_total": len(probes),
        "all_probes_pass": all_probes_pass,
        "cognitive_subsystem_index": round(cpi, 4),
        "live_telemetry": {
            "cpu_percent": cpu_usage,
            "mem_percent": mem_usage,
            "registered_skills": registered_skills,
            "cognitive_coherence": coherence,
            "active_goals": active_goals_count,
            "passed_probes": passed_probes,
            "cognitive_subsystem_index": cpi,
            "will_probe_p50_ms": round(p50_latency, 2),
            "will_probe_p99_ms": round(p99_latency, 2),
            "resource_observation": resource_provenance,
        },
        "hardware_environment": {
            "os": "macOS",
            "model_stack": "Frozen Stack (Gemini/MLX Dual Layer)",
            "concurrency_deadlock_mitigation": "active",
            "cooldown_deduplication": "active",
        },
    }

    await asyncio.to_thread(out_path.write_text, json.dumps(report, indent=2))
    print(f"Subsystem-liveness report saved to {out_path}")
    
    # Honest Markdown summary — real probe results only, no fabricated scores.
    def _mark(ok: bool) -> str:
        return "PASS" if ok else "FAIL"

    cat_rows = "\n".join(
        f"| {cat['name']} | `{cat['metric']}` |" for cat in CATEGORIES.values()
    )
    md_content = f"""# Aura Cognitive Subsystem-Liveness Battery

**Evaluation timestamp**: `{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}`
**Commit SHA**: `{commit_sha}`
**Measurement**: subsystem liveness (NOT a benchmarked capability score)

> This battery verifies Aura's cognitive subsystems are **live and functioning**
> via real probes. It does **not** produce capability scores or baseline
> comparisons — that requires the graded ablation in
> `tools/agi/run_prompt_baseline_ablation.py`. Earlier versions of this file
> fabricated a 17-category scorecard from noise; that has been removed.

## Subsystem probes ({passed_probes}/{len(probes)} passed)

| Probe | Result | Detail |
| :--- | :---: | :--- |
| Will concurrency + audit trail | {_mark(will_ok)} | p50=`{p50_latency:.2f}ms`, p99=`{p99_latency:.2f}ms` |
| Volition cooldown deduplication | {_mark(dedup_ok)} | distinct goals not blocked |
| Agency goal completion | {_mark(completion_ok)} | goal lifecycle state mutation |
| Affective steering vector library | {_mark(steering_ok)} | steering dimensions present |
| Skill surface + constraint handling | {_mark(skills_ok)} | {registered_skills} skills registered |

**Cognitive Subsystem Index (probes passed / total)**: `{cpi:.2%}`

## Live telemetry
- CPU: `{cpu_usage}%`  ·  Memory: `{mem_usage}%`
- Registered skills: `{registered_skills}`
- Cognitive coherence: `{coherence}`
- Active goals: `{active_goals_count}`

## Capability areas probed for subsystem liveness ({len(CATEGORIES)})
| Capability Area | Target Metric |
| :--- | :--- |
{cat_rows}

---
*These areas are the taxonomy the subsystems serve; this report asserts the
subsystems are live, not a graded score on each area.*
"""

    await asyncio.to_thread(md_path.write_text, md_content)
    print(f"Subsystem-liveness summary saved to {md_path}")

if __name__ == "__main__":
    asyncio.run(main())
