"""Deterministic cognitive-compute routing for the live latent cortex.

The policy does not perform reasoning or grant tool authority. It converts
measured demand and capacity into hard-bounded recurrence, branch, search,
verifier, and acquisition budgets. The existing RLC mechanisms remain the
executors; this contract makes their shared economy explicit and auditable.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Any

from core.brain.llm.latent_cortex.loop_core import canonical_sha256
from core.runtime.structured_input import analyze_prompt_shape

ADAPTIVE_COMPUTE_SCHEMA = "aura.rlc.adaptive_compute.v1"
ADAPTIVE_COMPUTE_VERSION = "2026.07.26.1"
ADAPTIVE_EXECUTION_SCHEMA = "aura.rlc.adaptive_compute.execution.v1"
ADAPTIVE_ACQUISITION_SCHEMA = "aura.rlc.adaptive_compute.acquisition.v1"


def _unit(value: Any, *, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{name} must be a finite number")
    return min(1.0, max(0.0, float(value)))


def _positive(value: Any, *, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def estimate_objective_difficulty(objective: str) -> dict[str, Any]:
    """Estimate structural task difficulty without asking the model itself."""

    if not isinstance(objective, str):
        raise TypeError("adaptive objective must be text")
    visible = objective.strip()
    shape = analyze_prompt_shape(visible).to_dict()
    words = len(visible.split())
    parts = int(shape["question_parts"])
    structure = (parts - 1) / 5.0
    directive_density = min(
        1.0,
        (
            int(shape["connector_parts"])
            + int(shape["numbered_parts"])
            + int(shape["imperative_parts"])
        )
        / 8.0,
    )
    length = min(1.0, words / 240.0)
    coverage = 1.0 if shape["requires_single_reply_coverage"] else 0.0
    difficulty = min(
        1.0,
        0.12
        + 0.38 * structure
        + 0.18 * directive_density
        + 0.17 * length
        + 0.15 * coverage,
    )
    return {
        "difficulty": round(difficulty, 6),
        "word_count": words,
        "prompt_shape": shape,
    }


def resource_pressure_signal(snapshot: Mapping[str, Any] | None) -> dict[str, Any]:
    """Normalize the canonical admission snapshot into one conservative signal."""

    raw = dict(snapshot or {})
    source = str(raw.get("observation_source") or "unavailable")
    observed = bool(
        raw.get("resource_observation_available")
        or raw.get("host_observed")
        or source not in {"", "unavailable"}
    )
    try:
        memory_percent = max(0.0, min(100.0, float(raw.get("memory_percent") or 0.0)))
        thermal_level = max(0, min(3, int(raw.get("thermal_level") or 0)))
        loop_lag_s = max(0.0, float(raw.get("loop_lag_s") or 0.0))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("adaptive resource snapshot is malformed") from exc
    if not all(math.isfinite(value) for value in (memory_percent, loop_lag_s)):
        raise ValueError("adaptive resource snapshot is malformed")
    red_zones = tuple(sorted({str(item) for item in raw.get("red_zones") or () if str(item)}))
    suspended = tuple(
        sorted({str(item) for item in raw.get("suspended_capabilities") or () if str(item)})
    )
    memory_pressure = min(1.0, max(0.0, (memory_percent - 65.0) / 30.0))
    thermal_pressure = thermal_level / 3.0
    loop_pressure = min(1.0, loop_lag_s / 2.0)
    explicit_pressure = max(
        0.85 if red_zones else 0.0,
        0.7 if suspended else 0.0,
        1.0 if raw.get("shutdown_requested") is True else 0.0,
    )
    pressure = max(memory_pressure, thermal_pressure, loop_pressure, explicit_pressure)
    if not observed:
        pressure = max(pressure, 0.6)
    return {
        "pressure": round(pressure, 6),
        "observed": observed,
        "observation_source": source,
        "memory_percent": round(memory_percent, 4),
        "thermal_level": thermal_level,
        "loop_lag_s": round(loop_lag_s, 6),
        "red_zones": list(red_zones),
        "suspended_capabilities": list(suspended),
        "shutdown_requested": raw.get("shutdown_requested") is True,
    }


def build_adaptive_compute_plan(
    *,
    objective: str,
    stakes: float,
    uncertainty: float,
    body_pressure: float,
    deadline_s: float,
    resource_snapshot: Mapping[str, Any] | None,
    foreground_request: bool,
    model_parameter_count: int,
    requested_decode_tokens: int,
    isolation_steps: int = 1,
) -> dict[str, Any]:
    if type(foreground_request) is not bool:
        raise ValueError("foreground_request must be boolean")
    if type(isolation_steps) is not int or isolation_steps < 1:
        raise ValueError("isolation_steps must be a positive integer")
    if (
        isinstance(model_parameter_count, bool)
        or not isinstance(model_parameter_count, int)
        or model_parameter_count < 0
    ):
        raise ValueError("model_parameter_count must be a non-negative integer")
    if type(requested_decode_tokens) is not int or requested_decode_tokens <= 0:
        raise ValueError("requested_decode_tokens must be a positive integer")

    stakes = _unit(stakes, name="stakes")
    uncertainty = _unit(uncertainty, name="uncertainty")
    body_pressure = _unit(body_pressure, name="body_pressure")
    deadline_s = _positive(deadline_s, name="deadline_s")
    objective_signal = estimate_objective_difficulty(objective)
    difficulty = float(objective_signal["difficulty"])
    resource = resource_pressure_signal(resource_snapshot)
    resource_pressure = float(resource["pressure"])

    body_headroom = 1.0 - 0.7 * body_pressure
    resource_headroom = 1.0 - 0.8 * resource_pressure
    deadline_headroom = min(1.0, max(0.0, (deadline_s - 30.0) / 150.0))
    capacity = min(body_headroom, resource_headroom) * (0.55 + 0.45 * deadline_headroom)
    demand = 0.4 * difficulty + 0.3 * uncertainty + 0.3 * stakes
    routed_intensity = min(1.0, demand * (0.45 + 0.55 * capacity))
    resident_scale = model_parameter_count >= 20_000_000_000

    level = 0
    if routed_intensity >= 0.25:
        level = 1
    if routed_intensity >= 0.58 and capacity >= 0.35:
        level = 2
    if routed_intensity >= 0.76 and capacity >= 0.55 and deadline_s >= 150.0:
        level = 3
    if body_pressure >= 0.85 or resource_pressure >= 0.9 or deadline_s < 45.0:
        level = 0

    max_steps = (2, 4, 6, 8)[level]
    if resident_scale and foreground_request:
        resident_cap = 2 if deadline_s < 105.0 else 3 if deadline_s < 180.0 else 4
        max_steps = min(max_steps, resident_cap)
    branches = (1, 2, 2, 3)[level]
    if capacity < 0.3:
        branches = 1
    elif stakes >= 0.75 and level >= 1:
        branches = max(branches, 2)
    if branches > 1:
        # A second branch that cannot be integrated is a second branch spent
        # for nothing. Cross-branch exchange is the only edge in the latent
        # workspace that runs backwards -- the consensus is written into slot
        # 0, ahead of the evidence prefix -- and the ensemble blocks it until
        # every branch has taken its isolation steps. A depth equal to that
        # isolation therefore leaves the only exchange on the final step,
        # where the consensus lands in the mailbox and no recurrent step
        # remains to carry it forward. Swept over this allocator before the
        # floor existed: 14 of 14 multi-branch allocations had a return that
        # could not travel, the resident interactive profile included.
        #
        # The fix is a step, not a branch. One more window pass over nine
        # slots and the middle half of a 64-layer decoder is 9 * 32 = 288
        # layer applications against the 4,719,333 that profile is granted,
        # which is six thousandths of one percent.
        max_steps = max(max_steps, isolation_steps + 1)
    min_steps = min(2, max_steps)

    tree_nodes = (2, 3, 5, 8)[level]
    tree_depth = (1, 1, 2, 3)[level]
    tree_branching = min(branches, (1, 2, 2, 3)[level])
    probe_tokens = (16, 24, 32, 48)[level]
    generative_tokens = (96, 128, 160, 192)[level]
    counterfactual_tokens = (64, 96, 128, 160)[level]
    counterfactual_interventions = (1, 2, 2, 3)[level]
    prefix_samples = (3, 3, 4, 5)[level]

    output_reserve_s = min(55.0, max(12.0, deadline_s * 0.25))
    tool_worthy = difficulty >= 0.35 or uncertainty >= 0.55
    acquisition_allowed = bool(
        tool_worthy
        and deadline_s - output_reserve_s >= 45.0
        and capacity >= 0.25
        and not resource["shutdown_requested"]
    )
    acquisition_reason = (
        "evidence_value_and_capacity"
        if acquisition_allowed
        else "insufficient_evidence_value"
        if not tool_worthy
        else "deadline_preserves_answer_surface"
        if deadline_s - output_reserve_s < 45.0
        else "resource_or_body_pressure"
        if capacity < 0.25
        else "runtime_shutdown"
    )
    reasons = [
        f"difficulty_level_{level}",
        "resident_foreground_cap" if resident_scale and foreground_request else "general_scale",
        f"acquisition_{acquisition_reason}",
    ]
    payload = {
        "schema": ADAPTIVE_COMPUTE_SCHEMA,
        "version": ADAPTIVE_COMPUTE_VERSION,
        "signals": {
            "difficulty": round(difficulty, 6),
            "uncertainty": round(uncertainty, 6),
            "stakes": round(stakes, 6),
            "body_pressure": round(body_pressure, 6),
            "deadline_s": round(deadline_s, 6),
            "foreground_request": foreground_request,
            "model_parameter_count": model_parameter_count,
            "resident_scale": resident_scale,
            "objective": objective_signal,
            "resource": resource,
        },
        "economy": {
            "body_headroom": round(body_headroom, 6),
            "resource_headroom": round(resource_headroom, 6),
            "deadline_headroom": round(deadline_headroom, 6),
            "capacity": round(capacity, 6),
            "demand": round(demand, 6),
            "routed_intensity": round(routed_intensity, 6),
            "level": level,
        },
        "routing": {
            "recurrence": {"min_steps": min_steps, "max_steps": max_steps},
            "branches": {"target": branches, "maximum": branches},
            "lookahead": {
                "mode": "active",
                "strategy": "uct",
                "max_nodes": tree_nodes,
                "max_depth": tree_depth,
                "branching_factor": tree_branching,
            },
            "verifier": {
                "probe_max_tokens": probe_tokens,
                "generative_max_atoms": 1 if level < 3 else 2,
                "generative_max_tokens": generative_tokens,
                "counterfactual_max_atoms": 1 if level < 3 else 2,
                "counterfactual_max_interventions": counterfactual_interventions,
                "counterfactual_max_tokens": counterfactual_tokens,
                "prefix_stability_samples": prefix_samples,
            },
            "tools": {
                "max_acquisitions": 1 if acquisition_allowed else 0,
                "max_continuation_rounds": 1 if acquisition_allowed else 0,
                "reason": acquisition_reason,
            },
        },
        "answer_surface": {
            "minimum_decode_tokens": requested_decode_tokens,
            "reserved_wall_clock_s": round(output_reserve_s, 6),
            "preserved": True,
        },
        "reasons": reasons,
    }
    return {**payload, "plan_sha256": canonical_sha256(payload)}


def validate_adaptive_compute_plan(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("adaptive compute plan must be a mapping")
    plan = dict(value)
    required = {
        "schema",
        "version",
        "signals",
        "economy",
        "routing",
        "answer_surface",
        "reasons",
        "plan_sha256",
    }
    if set(plan) != required or plan.get("schema") != ADAPTIVE_COMPUTE_SCHEMA:
        raise ValueError("adaptive compute plan fields differ")
    if plan.get("version") != ADAPTIVE_COMPUTE_VERSION:
        raise ValueError("adaptive compute policy version differs")
    payload = {key: plan[key] for key in required - {"plan_sha256"}}
    if plan.get("plan_sha256") != canonical_sha256(payload):
        raise ValueError("adaptive compute plan commitment differs")
    routing = plan.get("routing")
    if not isinstance(routing, Mapping):
        raise ValueError("adaptive compute routing is invalid")
    recurrence = routing.get("recurrence")
    branches = routing.get("branches")
    lookahead = routing.get("lookahead")
    verifier = routing.get("verifier")
    tools = routing.get("tools")
    if not all(isinstance(row, Mapping) for row in (recurrence, branches, lookahead, verifier, tools)):
        raise ValueError("adaptive compute routing sections are invalid")
    if set(routing) != {"recurrence", "branches", "lookahead", "verifier", "tools"}:
        raise ValueError("adaptive compute routing fields differ")
    if set(recurrence) != {"min_steps", "max_steps"} or set(branches) != {
        "target",
        "maximum",
    }:
        raise ValueError("adaptive compute recurrence or branch fields differ")
    if (
        type(recurrence.get("min_steps")) is not int
        or type(recurrence.get("max_steps")) is not int
        or not 1 <= recurrence["min_steps"] <= recurrence["max_steps"] <= 16
        or type(branches.get("target")) is not int
        or type(branches.get("maximum")) is not int
        or not 1 <= branches["target"] <= branches["maximum"] <= 4
        or type(tools.get("max_acquisitions")) is not int
        or tools["max_acquisitions"] not in {0, 1}
        or tools.get("max_continuation_rounds") != tools["max_acquisitions"]
    ):
        raise ValueError("adaptive compute routing bounds are invalid")
    from core.brain.llm.latent_cortex.latent_tree_search import LatentTreeSearchConfig

    LatentTreeSearchConfig.from_value(lookahead)
    verifier_fields = {
        "probe_max_tokens",
        "generative_max_atoms",
        "generative_max_tokens",
        "counterfactual_max_atoms",
        "counterfactual_max_interventions",
        "counterfactual_max_tokens",
        "prefix_stability_samples",
    }
    if set(verifier) != verifier_fields or any(
        type(verifier.get(name)) is not int or verifier[name] <= 0
        for name in verifier_fields
    ):
        raise ValueError("adaptive verifier routing is invalid")
    if (
        set(tools)
        != {"max_acquisitions", "max_continuation_rounds", "reason"}
        or not isinstance(tools.get("reason"), str)
        or not re.fullmatch(r"[a-z0-9_]{1,64}", tools["reason"])
    ):
        raise ValueError("adaptive tool routing is invalid")
    signals = plan.get("signals")
    economy = plan.get("economy")
    if (
        not isinstance(signals, Mapping)
        or set(signals)
        != {
            "difficulty",
            "uncertainty",
            "stakes",
            "body_pressure",
            "deadline_s",
            "foreground_request",
            "model_parameter_count",
            "resident_scale",
            "objective",
            "resource",
        }
        or not isinstance(economy, Mapping)
        or set(economy)
        != {
            "body_headroom",
            "resource_headroom",
            "deadline_headroom",
            "capacity",
            "demand",
            "routed_intensity",
            "level",
        }
        or type(economy.get("level")) is not int
        or economy["level"] not in {0, 1, 2, 3}
    ):
        raise ValueError("adaptive compute signals or economy are invalid")
    for name in ("difficulty", "uncertainty", "stakes", "body_pressure"):
        _unit(signals.get(name), name=f"signals.{name}")
    _positive(signals.get("deadline_s"), name="signals.deadline_s")
    for name in (
        "body_headroom",
        "resource_headroom",
        "deadline_headroom",
        "capacity",
        "demand",
        "routed_intensity",
    ):
        _unit(economy.get(name), name=f"economy.{name}")
    if (
        type(signals.get("foreground_request")) is not bool
        or type(signals.get("resident_scale")) is not bool
        or type(signals.get("model_parameter_count")) is not int
        or signals["model_parameter_count"] < 0
        or not isinstance(signals.get("objective"), Mapping)
        or not isinstance(signals.get("resource"), Mapping)
    ):
        raise ValueError("adaptive compute signal types are invalid")
    resource_pressure_signal(signals["resource"])
    answer = plan.get("answer_surface")
    if (
        not isinstance(answer, Mapping)
        or set(answer)
        != {"minimum_decode_tokens", "reserved_wall_clock_s", "preserved"}
        or type(answer.get("minimum_decode_tokens")) is not int
        or answer["minimum_decode_tokens"] <= 0
        or _positive(
            answer.get("reserved_wall_clock_s"),
            name="answer_surface.reserved_wall_clock_s",
        )
        <= 0.0
        or answer.get("preserved") is not True
        or not isinstance(plan.get("reasons"), list)
        or not plan["reasons"]
        or any(
            not isinstance(reason, str)
            or not re.fullmatch(r"[a-z0-9_]{1,96}", reason)
            for reason in plan["reasons"]
        )
    ):
        raise ValueError("adaptive answer-surface reserve is invalid")
    return plan


def apply_adaptive_compute_plan(
    config: Mapping[str, Any],
    budget: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    validated = validate_adaptive_compute_plan(plan)
    updated_config = dict(config)
    updated_budget = dict(budget)
    routing = validated["routing"]
    recurrence = routing["recurrence"]
    branches = routing["branches"]
    lookahead = routing["lookahead"]
    verifier = routing["verifier"]
    updated_config.update(
        {
            "min_steps": recurrence["min_steps"],
            "max_steps": recurrence["max_steps"],
            "n_branches": branches["target"],
            "latent_tree_search": dict(lookahead),
            "verifier_probe_max_tokens": verifier["probe_max_tokens"],
            "generative_verifier_max_atoms": verifier["generative_max_atoms"],
            "generative_verifier_max_tokens": verifier["generative_max_tokens"],
            "counterfactual_verifier_max_atoms": verifier["counterfactual_max_atoms"],
            "counterfactual_verifier_max_interventions": verifier[
                "counterfactual_max_interventions"
            ],
            "counterfactual_verifier_max_tokens": verifier["counterfactual_max_tokens"],
            "prefix_stability_samples": verifier["prefix_stability_samples"],
        }
    )
    if int(updated_config.get("decode_max_tokens") or 0) < int(
        validated["answer_surface"]["minimum_decode_tokens"]
    ):
        raise ValueError("adaptive compute reduced the answer surface")
    return updated_config, updated_budget


def enforce_adaptive_compute_limits(
    config: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Keep learned execution arms inside the prior adaptive admission."""

    validated = validate_adaptive_compute_plan(plan)
    updated = dict(config)
    routing = validated["routing"]
    recurrence = routing["recurrence"]
    verifier = routing["verifier"]
    updated["max_steps"] = min(
        int(updated.get("max_steps") or recurrence["max_steps"]),
        int(recurrence["max_steps"]),
    )
    updated["min_steps"] = min(
        int(updated.get("min_steps") or recurrence["min_steps"]),
        updated["max_steps"],
    )
    updated["n_branches"] = min(
        max(1, int(updated.get("n_branches") or 1)),
        int(routing["branches"]["maximum"]),
    )
    updated["latent_tree_search"] = dict(routing["lookahead"])
    for config_key, plan_key in (
        ("verifier_probe_max_tokens", "probe_max_tokens"),
        ("generative_verifier_max_atoms", "generative_max_atoms"),
        ("generative_verifier_max_tokens", "generative_max_tokens"),
        ("counterfactual_verifier_max_atoms", "counterfactual_max_atoms"),
        (
            "counterfactual_verifier_max_interventions",
            "counterfactual_max_interventions",
        ),
        ("counterfactual_verifier_max_tokens", "counterfactual_max_tokens"),
        ("prefix_stability_samples", "prefix_stability_samples"),
    ):
        updated[config_key] = min(
            int(updated.get(config_key) or verifier[plan_key]),
            int(verifier[plan_key]),
        )
    if int(updated.get("decode_max_tokens") or 0) < int(
        validated["answer_surface"]["minimum_decode_tokens"]
    ):
        raise ValueError("adaptive compute reduced the answer surface")
    return updated


def build_adaptive_acquisition_receipt(
    *,
    plan: Mapping[str, Any],
    request_sha256: str,
    attempted: bool,
) -> dict[str, Any]:
    validated = validate_adaptive_compute_plan(plan)
    if (
        not isinstance(request_sha256, str)
        or len(request_sha256) != 64
        or any(character not in "0123456789abcdef" for character in request_sha256)
        or type(attempted) is not bool
    ):
        raise ValueError("adaptive acquisition request identity is invalid")
    tools = validated["routing"]["tools"]
    authorized = tools["max_acquisitions"] == 1
    if attempted and not authorized:
        raise ValueError("adaptive acquisition exceeded its budget")
    payload = {
        "schema": ADAPTIVE_ACQUISITION_SCHEMA,
        "plan_sha256": validated["plan_sha256"],
        "request_sha256": request_sha256,
        "authorized": authorized,
        "attempted": attempted,
        "max_acquisitions": tools["max_acquisitions"],
        "max_continuation_rounds": tools["max_continuation_rounds"],
        "reason": tools["reason"],
    }
    return {**payload, "receipt_sha256": canonical_sha256(payload)}


def validate_adaptive_acquisition_receipt(value: Any) -> dict[str, Any]:
    fields = {
        "schema",
        "plan_sha256",
        "request_sha256",
        "authorized",
        "attempted",
        "max_acquisitions",
        "max_continuation_rounds",
        "reason",
        "receipt_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("adaptive acquisition receipt fields differ")
    payload = {key: value[key] for key in fields - {"receipt_sha256"}}
    if (
        value.get("schema") != ADAPTIVE_ACQUISITION_SCHEMA
        or value.get("receipt_sha256") != canonical_sha256(payload)
        or type(value.get("authorized")) is not bool
        or type(value.get("attempted")) is not bool
        or type(value.get("max_acquisitions")) is not int
        or value["max_acquisitions"] not in {0, 1}
        or value.get("max_continuation_rounds") != value["max_acquisitions"]
        or value["authorized"] is not (value["max_acquisitions"] == 1)
        or value["attempted"] and not value["authorized"]
        or any(
            not isinstance(value.get(name), str)
            or len(value[name]) != 64
            or any(character not in "0123456789abcdef" for character in value[name])
            for name in ("plan_sha256", "request_sha256")
        )
    ):
        raise ValueError("adaptive acquisition receipt is invalid")
    return dict(value)


def build_adaptive_execution_receipt(
    *,
    plan: Mapping[str, Any],
    config: Mapping[str, Any],
    budget: Mapping[str, Any],
    worker_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    validated = validate_adaptive_compute_plan(plan)
    routing = validated["routing"]
    max_branches = int(routing["branches"]["maximum"])
    actual_branches = worker_receipt.get("n_branches")
    actual_steps = worker_receipt.get("steps_taken")
    if (
        type(actual_branches) is not int
        or not 1 <= actual_branches <= max_branches
        or type(actual_steps) is not int
        or not 0 <= actual_steps <= int(routing["recurrence"]["max_steps"])
        or config.get("n_branches") != actual_branches
        or worker_receipt.get("verifier_probe_max_tokens")
        != config.get("verifier_probe_max_tokens")
        or int(config.get("decode_max_tokens") or 0)
        < int(validated["answer_surface"]["minimum_decode_tokens"])
    ):
        raise ValueError("adaptive execution differs from the admitted plan")
    payload = {
        "schema": ADAPTIVE_EXECUTION_SCHEMA,
        "plan": validated,
        "plan_sha256": validated["plan_sha256"],
        "episode_id": str(worker_receipt.get("episode_id") or ""),
        "actual": {
            "steps_taken": actual_steps,
            "n_branches": actual_branches,
            "verifier_probe_max_tokens": config.get("verifier_probe_max_tokens"),
            "decode_max_tokens": config.get("decode_max_tokens"),
            "wall_clock_s": budget.get("wall_clock_s"),
        },
        "within_plan": True,
        "answer_surface_preserved": True,
    }
    return {**payload, "receipt_sha256": canonical_sha256(payload)}


def validate_adaptive_execution_receipt(value: Any) -> dict[str, Any]:
    fields = {
        "schema",
        "plan",
        "plan_sha256",
        "episode_id",
        "actual",
        "within_plan",
        "answer_surface_preserved",
        "receipt_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("adaptive execution receipt fields differ")
    payload = {key: value[key] for key in fields - {"receipt_sha256"}}
    plan = validate_adaptive_compute_plan(value.get("plan"))
    actual = value.get("actual")
    if (
        value.get("schema") != ADAPTIVE_EXECUTION_SCHEMA
        or value.get("receipt_sha256") != canonical_sha256(payload)
        or value.get("plan_sha256") != plan["plan_sha256"]
        or not isinstance(value.get("episode_id"), str)
        or not value["episode_id"]
        or not isinstance(actual, Mapping)
        or set(actual)
        != {
            "steps_taken",
            "n_branches",
            "verifier_probe_max_tokens",
            "decode_max_tokens",
            "wall_clock_s",
        }
        or type(actual.get("steps_taken")) is not int
        or not 0 <= actual["steps_taken"] <= plan["routing"]["recurrence"]["max_steps"]
        or type(actual.get("n_branches")) is not int
        or not 1 <= actual["n_branches"] <= plan["routing"]["branches"]["maximum"]
        or type(actual.get("verifier_probe_max_tokens")) is not int
        or actual["verifier_probe_max_tokens"]
        > plan["routing"]["verifier"]["probe_max_tokens"]
        or type(actual.get("decode_max_tokens")) is not int
        or actual["decode_max_tokens"] < plan["answer_surface"]["minimum_decode_tokens"]
        or _positive(actual.get("wall_clock_s"), name="actual.wall_clock_s") <= 0.0
        or value.get("within_plan") is not True
        or value.get("answer_surface_preserved") is not True
    ):
        raise ValueError("adaptive execution receipt is invalid")
    return dict(value)


__all__ = [
    "ADAPTIVE_ACQUISITION_SCHEMA",
    "ADAPTIVE_COMPUTE_SCHEMA",
    "ADAPTIVE_COMPUTE_VERSION",
    "ADAPTIVE_EXECUTION_SCHEMA",
    "apply_adaptive_compute_plan",
    "build_adaptive_acquisition_receipt",
    "build_adaptive_compute_plan",
    "build_adaptive_execution_receipt",
    "estimate_objective_difficulty",
    "enforce_adaptive_compute_limits",
    "resource_pressure_signal",
    "validate_adaptive_compute_plan",
    "validate_adaptive_acquisition_receipt",
    "validate_adaptive_execution_receipt",
]
