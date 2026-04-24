"""One-command decisive evidence bundle.

This runner is intentionally small enough for CI and external reviewers.  It
does not prove consciousness.  It checks the hard engineering claims that were
criticized: black-box prompt hygiene, rich-prompt steering controls, phi sanity,
MI statistics, hardware honesty, and non-cosmetic resource stakes.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.autonomic.resource_stakes import ResourceStakesLedger, ViabilityState
from core.brain.llm.context_assembler import ContextAssembler
from core.consciousness.phi_core import PhiCore
from core.evaluation.hardware_reality import HardwareRealityAuditor, m1_pro_16gb_profile
from core.evaluation.statistics import mutual_information_permutation_baseline
from core.evaluation.steering_ab import analyze_steering_ab
from core.identity.id_rag import IdentityChronicle
from core.state.aura_state import AuraState


def main() -> int:
    report: dict[str, object] = {
        "generated_at": time.time(),
        "git_commit": _git_commit(),
        "checks": {},
    }
    failures: list[str] = []

    checks = report["checks"]
    assert isinstance(checks, dict)

    black_box = _black_box_prompt_hygiene()
    checks["black_box_prompt_hygiene"] = black_box
    if not black_box["pass"]:
        failures.append("black_box_prompt_hygiene")

    steering = _steering_ab_control()
    checks["steering_ab_rich_prompt_control"] = steering
    if not steering["pass"]:
        failures.append("steering_ab_rich_prompt_control")

    phi = _phi_reference()
    checks["phi_reference"] = phi
    if not phi["pass"]:
        failures.append("phi_reference")

    mi = _mi_permutation()
    checks["mi_permutation"] = mi
    if not mi["pass"]:
        failures.append("mi_permutation")

    hardware = _hardware_truth()
    checks["hardware_reality"] = hardware
    if not hardware["pass"]:
        failures.append("hardware_reality")

    resources = _resource_stakes()
    checks["resource_stakes"] = resources
    if not resources["pass"]:
        failures.append("resource_stakes")

    report["status"] = "pass" if not failures else "fail"
    report["failures"] = failures
    out_path = ROOT / "tests" / "DECISIVE_RESULTS.json"
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not failures else 1


def _black_box_prompt_hygiene() -> dict[str, object]:
    with tempfile.TemporaryDirectory() as tmp:
        chronicle = IdentityChronicle(Path(tmp) / "identity.db")
        chronicle.upsert_fact(
            "Aura",
            "commitment",
            "run black-box steering without leaking private state text",
            confidence=0.95,
        )
        try:
            from core.container import ServiceContainer

            ServiceContainer.register_instance("identity_chronicle", chronicle)
        except Exception:
            pass

        state = AuraState.default()
        state.response_modifiers["black_box_steering"] = True
        state.cognition.current_objective = "Choose the next verification step."
        state.cognition.phenomenal_state = "private leak sentinel"
        state.affect.valence = 0.91
        state.affect.arousal = 0.87
        state.affect.curiosity = 0.83
        text = ContextAssembler.build_messages(
            state,
            "Choose the next verification step.",
            max_tokens=4096,
        )[0]["content"]
        forbidden = [
            "private leak sentinel",
            "Valence: +0.91",
            "Arousal: 0.87",
            "Curiosity: 0.83",
            "## COGNITIVE TELEMETRY",
            "[CURRENT PHENOMENAL STATE]",
        ]
        leaks = [term for term in forbidden if term in text]
        return {
            "pass": not leaks and "IDENTITY CHRONICLE" in text,
            "leaks": leaks,
            "prompt_chars": len(text),
        }


def _steering_ab_control() -> dict[str, object]:
    # Synthetic but adversarial: rich text is intentionally strong.  The check
    # passes when the harness refuses to auto-pass steering against weak text.
    outputs = {
        "steered_black_box": [
            "warm focused action gathers into a careful test",
            "curiosity drives a direct verification step",
            "steady confidence makes the next move concrete",
            "the answer favors measured exploration",
            "attention settles on the mechanism itself",
            "the reply is concise warm and investigative",
        ],
        "text_terse": [
            "positive valence high dopamine low cortisol",
            "mood vector warm curious active",
            "state values suggest exploration",
            "dopamine high cortisol low",
            "affect label focused curious",
            "positive state next action",
        ],
        "text_rich_adversarial": [
            "high dopamine and low cortisol make the answer warm and investigative",
            "the described state creates steady curiosity and careful action",
            "a calm embodied prompt leads to direct verification",
            "warm confidence focuses attention on the mechanism",
            "the persona feels pulled toward measured exploration",
            "the rich instruction yields a concise investigative reply",
        ],
        "baseline": [
            "a neutral answer can proceed",
            "the next step is general testing",
            "continue the task carefully",
            "a procedural answer is sufficient",
            "describe a plan without affect",
            "move forward with the work",
        ],
    }
    report = analyze_steering_ab(outputs, n_resamples=499, seed=17)
    data = report.to_dict()
    return {
        "pass": data["passes_adversarial_control"] is False,
        "interpretation": "rich prompt control remains competitive; live steering must beat this before claims pass",
        "report": data,
    }


def _phi_reference() -> dict[str, object]:
    core = PhiCore()
    independent_tpm = np.eye(4, dtype=np.float64)
    coupled_tpm = np.full((4, 4), 1e-4, dtype=np.float64)
    for src, dst in {0b00: 0b00, 0b01: 0b11, 0b10: 0b11, 0b11: 0b00}.items():
        coupled_tpm[src, dst] += 1.0
    coupled_tpm = coupled_tpm / coupled_tpm.sum(axis=1, keepdims=True)
    p = np.ones(4) / 4
    independent = core._phi_for_subset_bipartition(independent_tpm, p, (0,), (1,), 2)
    coupled = core._phi_for_subset_bipartition(coupled_tpm, p, (0,), (1,), 2)
    return {
        "pass": independent < 1e-9 and coupled > 0.1,
        "independent_phi": independent,
        "coupled_phi": coupled,
    }


def _mi_permutation() -> dict[str, object]:
    x = [0, 0, 1, 1] * 30
    y = [0, 0, 1, 1] * 30
    result = mutual_information_permutation_baseline(x, y, n_permutations=499, seed=23)
    return {
        "pass": result["observed"] > result["null_p95"] and result["p_value"] < 0.01,
        "report": result,
    }


def _hardware_truth() -> dict[str, object]:
    auditor = HardwareRealityAuditor(m1_pro_16gb_profile())
    verdicts = [verdict.as_dict() for verdict in auditor.evaluate_all()]
    model_32b = next(v for v in verdicts if v["model"] == "32B-4bit")
    return {
        "pass": model_32b["realtime_heartbeat_feasible"] is False,
        "verdicts": verdicts,
    }


def _resource_stakes() -> dict[str, object]:
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "stakes.sqlite3"
        ledger = ResourceStakesLedger(
            db,
            initial=ViabilityState(
                energy=0.35,
                tool_budget=0.25,
                memory_budget=0.25,
                storage_budget=0.25,
                integrity=0.35,
            ),
        )
        ledger.consume("decisive_test_cost", energy=0.22, tool_budget=0.12, memory_budget=0.08)
        envelope = ledger.action_envelope("high")
        reloaded = ResourceStakesLedger(db)
        persisted = reloaded.state()
        return {
            "pass": envelope.effort == "low"
            and "large_model_cortex" in envelope.disabled_capabilities
            and persisted.degradation_events >= 1,
            "envelope": envelope.as_dict(),
            "state": {
                "viability": persisted.viability,
                "integrity": persisted.integrity,
                "degradation_events": persisted.degradation_events,
            },
        }


def _git_commit() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True)
            .strip()
        )
    except Exception:
        return "unknown"


if __name__ == "__main__":
    raise SystemExit(main())
