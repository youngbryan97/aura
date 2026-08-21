#!/usr/bin/env python3
"""Verify the materialized semantic neural path through canonical ingress."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.brain.llm.latent_cortex.semantic_neural_decode_context import (  # noqa: E402
    execute_semantic_neural_decode_state,
)
from core.brain.llm.latent_cortex.semantic_surface_adapter import (  # noqa: E402
    SEMANTIC_SURFACE_PROFILES,
    execute_scientific_surface,
    render_scientific_surface,
)
from core.brain.llm.qualified_recurrent_ingress import (  # noqa: E402
    execute_qualified_recurrent_objective,
)
from core.brain.llm.semantic_neural_serving import (  # noqa: E402
    DEFAULT_ACTIVATION_PATH,
    semantic_neural_serving_status,
)
from core.learning.frontier_process_supervision import (  # noqa: E402
    frontier_process_task_battery,
)
from core.learning.semantic_neural_controls import (  # noqa: E402
    semantic_neural_family_lesion_machine,
)
from core.runtime.atomic_writer import atomic_write_text  # noqa: E402

SCHEMA = "aura.semantic_neural_runtime_verification.v2"
RUNTIME_DOMAINS = (
    "coding",
    "calibration",
    "misleading_premise",
    "scientific_inference",
)
RUNTIME_FAMILIES = (
    "frontier_calibration",
    "frontier_coding",
    "frontier_misleading_premise",
    "frontier_scientific_inference",
)
RUNTIME_DIFFICULTIES = (1, 2, 3)
RUNTIME_PACKAGE_ID = "cp568-resident-semantic-neural-active-r1"
VERIFIER_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _mixed_runtime_tasks(*, seed: int, tasks_per_difficulty: int) -> list[Any]:
    tasks = frontier_process_task_battery(
        RUNTIME_DOMAINS,
        RUNTIME_DIFFICULTIES,
        tasks_per_difficulty,
        seed=seed,
    )
    adapted = []
    surface_index = 0
    for index, task in enumerate(tasks):
        if task.family == "frontier_scientific_inference":
            profile = SEMANTIC_SURFACE_PROFILES[
                surface_index % len(SEMANTIC_SURFACE_PROFILES)
            ]
            surface_index += 1
            task = replace(
                task,
                prompt=render_scientific_surface(
                    task.prompt,
                    profile=profile,
                    permutation_seed=seed + index,
                ),
                transition_trace=None,
                transition_program=None,
            )
        adapted.append(task)
    return adapted


def _expected_state(task: Any, *, machine: Any = None) -> tuple[Any, str | None]:
    if task.family != "frontier_scientific_inference" or not task.prompt.startswith(
        ("Causal study report.\n", "Controlled causal field note.\n", "CAUSAL_FACTS_V1\n")
    ):
        return (
            execute_semantic_neural_decode_state(task.prompt, task.family, machine=machine),
            None,
        )
    decoded = execute_scientific_surface(task.prompt, machine=machine)
    return decoded.state, decoded.receipt()["receipt_sha256"]


async def _verify(*, seed: int, tasks_per_difficulty: int) -> dict[str, Any]:
    activation = json.loads(DEFAULT_ACTIVATION_PATH.read_text(encoding="utf-8"))
    model_path = str(activation["model_identity"]["path"])
    status = semantic_neural_serving_status(model_path)
    if status.get("active") is not True:
        raise RuntimeError(f"semantic neural serving is inactive: {status}")
    activation_receipt = status.get("receipt")
    if (
        not isinstance(activation_receipt, dict)
        or activation_receipt.get("package_id") != RUNTIME_PACKAGE_ID
        or tuple(activation_receipt.get("allowed_families") or ()) != RUNTIME_FAMILIES
        or tuple(activation_receipt.get("allowed_surface_profiles") or ())
        != SEMANTIC_SURFACE_PROFILES
        or activation_receipt.get("promotion_mode") != "active"
    ):
        raise RuntimeError("semantic neural serving activated the wrong runtime package")
    tasks = _mixed_runtime_tasks(
        seed=seed,
        tasks_per_difficulty=tasks_per_difficulty,
    )
    rows = []
    latencies = []
    lesion_disruptions = 0
    exact_by_domain = {domain: 0 for domain in RUNTIME_DOMAINS}
    lesions_by_domain = {domain: 0 for domain in RUNTIME_DOMAINS}
    surface_profiles = {profile: 0 for profile in SEMANTIC_SURFACE_PROFILES}
    for task in tasks:
        started = time.perf_counter()
        result = await execute_qualified_recurrent_objective(
            None,
            task.prompt,
            timeout_s=30.0,
        )
        latency_ms = (time.perf_counter() - started) * 1000.0
        latencies.append(latency_ms)
        grade = task.grade(str(result.get("text") or ""))
        if result.get("ok") is not True or grade.get("correct") is not True:
            raise RuntimeError(f"semantic runtime failed {task.task_id}: {result}")
        domain = task.family.removeprefix("frontier_")
        if domain not in exact_by_domain:
            raise RuntimeError(f"semantic runtime emitted an unexpected family: {task.family}")
        exact_by_domain[domain] += 1
        expected_state, expected_surface_receipt = _expected_state(task)
        runtime_surface_receipt = result["receipt"].get("surface_decode_receipt")
        runtime_surface_receipt_sha = (
            runtime_surface_receipt.get("receipt_sha256")
            if isinstance(runtime_surface_receipt, dict)
            else None
        )
        if runtime_surface_receipt_sha != expected_surface_receipt:
            raise RuntimeError(
                f"semantic runtime surface receipt differs for {task.task_id}"
            )
        parser_id = str(result["receipt"]["admission"]["parser_id"])
        surface_profile = None
        if expected_surface_receipt is not None:
            surface_profile = parser_id.removeprefix(
                "semantic_scientific_surface."
            ).removesuffix(".v1")
            if surface_profile not in surface_profiles:
                raise RuntimeError("semantic runtime used an unmeasured surface profile")
            surface_profiles[surface_profile] += 1
        try:
            lesion_state, _lesion_surface_receipt = _expected_state(
                task,
                machine=semantic_neural_family_lesion_machine(task.family),
            )
        except (RuntimeError, ValueError):
            lesion_disrupted = True
        else:
            lesion_disrupted = (
                lesion_state.semantic_result != expected_state.semantic_result
            )
        lesion_disruptions += int(lesion_disrupted)
        lesions_by_domain[domain] += int(lesion_disrupted)
        rows.append(
            {
                "task_id": task.task_id,
                "family": task.family,
                "depth": task.depth,
                "latency_ms": round(latency_ms, 3),
                "answer_sha256": hashlib.sha256(
                    str(result["text"]).encode("ascii")
                ).hexdigest(),
                "runtime_receipt_sha256": result["receipt"]["receipt_sha256"],
                "semantic_state_receipt_sha256": result["receipt"][
                    "semantic_state_receipt"
                ]["receipt_sha256"],
                "surface_profile": surface_profile,
                "surface_decode_receipt_sha256": runtime_surface_receipt_sha,
                "lesion_disrupted": lesion_disrupted,
            }
        )
    unsupported = await execute_qualified_recurrent_objective(
        None,
        "Please answer a general question.",
        timeout_s=5.0,
    )
    if unsupported != {
        "eligible": False,
        "attempted": False,
        "ok": False,
        "reason": "qualified_recurrent_objective_unsupported",
    }:
        raise RuntimeError("semantic runtime broadened to unsupported language")
    if lesion_disruptions != len(tasks):
        raise RuntimeError("family-targeted lesions did not remove every runtime path")
    expected_per_domain = len(RUNTIME_DIFFICULTIES) * tasks_per_difficulty
    if any(count != expected_per_domain for count in exact_by_domain.values()) or any(
        count != expected_per_domain for count in lesions_by_domain.values()
    ):
        raise RuntimeError("runtime verification did not cover every admitted domain equally")
    expected_per_surface = expected_per_domain // len(SEMANTIC_SURFACE_PROFILES)
    if any(count != expected_per_surface for count in surface_profiles.values()):
        raise RuntimeError("runtime verification did not balance scientific surfaces")
    if max(latencies) > 2_000.0:
        raise RuntimeError("semantic runtime exceeded its bounded local execution latency")
    body = {
        "schema": SCHEMA,
        "verified": True,
        "seed": seed,
        "domains": list(RUNTIME_DOMAINS),
        "difficulties": list(RUNTIME_DIFFICULTIES),
        "tasks_per_difficulty": tasks_per_difficulty,
        "task_count": len(tasks),
        "exact_count": len(tasks),
        "exact_by_domain": exact_by_domain,
        "lesion_disruption_count": lesion_disruptions,
        "lesion_disruptions_by_domain": lesions_by_domain,
        "scientific_surface_profiles": surface_profiles,
        "unsupported_language_refused": True,
        "mean_latency_ms": round(statistics.fmean(latencies), 3),
        "p50_latency_ms": round(statistics.median(latencies), 3),
        "max_latency_ms": round(max(latencies), 3),
        "activation_receipt": activation_receipt,
        "verifier_source_sha256": VERIFIER_SOURCE_SHA256,
        "rows_sha256": _sha(rows),
        "rows": rows,
        "claim_boundary": (
            "qualified canonical and less-constrained scientific-surface runtime integration "
            "on the CP568 active requalification bound to the CP566 bounded WOW evidence; not "
            "open-domain, broad reasoning, static fusion, or frontier performance"
        ),
    }
    return {**body, "verification_receipt_sha256": _sha(body)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=2026081561)
    parser.add_argument("--tasks-per-difficulty", type=int, default=10)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not 2 <= args.tasks_per_difficulty <= 20:
        raise ValueError("runtime verification task count is outside [2, 20]")
    report = asyncio.run(
        _verify(
            seed=args.seed,
            tasks_per_difficulty=args.tasks_per_difficulty,
        )
    )
    destination = args.out.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(destination, json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
