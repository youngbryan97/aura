"""Aggregate non-serving resident probes into a bounded shadow-canary verdict."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any, Final

from core.brain.llm.unified_recurrent_shadow_probe_contract import (
    seal_shadow_probe_request,
    shadow_probe_receipt_errors,
)
from core.runtime.errors import record_degradation

PLAN_SCHEMA: Final = "aura.unified_intrinsic.shadow_canary_plan.v1"
VERDICT_SCHEMA: Final = "aura.unified_intrinsic.shadow_canary_verdict.v1"
SUPPORTED: Final = "supported_domain_shadow_canary"
REFUTED: Final = "refuted_domain_shadow_canary"
MAX_CASES: Final = 128
_HEX = frozenset("0123456789abcdef")


class UnifiedRecurrentShadowCanaryError(RuntimeError):
    """The canary plan or evidence is malformed or identity-inconsistent."""


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


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _HEX for character in value)
    )


def _normalized_cases(cases: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(cases, (list, tuple)) or not 1 <= len(cases) <= MAX_CASES:
        raise UnifiedRecurrentShadowCanaryError("shadow canary case count invalid")
    normalized: list[dict[str, Any]] = []
    task_ids: set[str] = set()
    request_ids: set[str] = set()
    for index, raw in enumerate(cases):
        if not isinstance(raw, Mapping):
            raise UnifiedRecurrentShadowCanaryError("shadow canary case invalid")
        task_id = raw.get("task_id")
        family = raw.get("family")
        if (
            not isinstance(task_id, str)
            or not task_id
            or len(task_id) > 256
            or task_id in task_ids
            or not isinstance(family, str)
            or not family
            or len(family) > 128
        ):
            raise UnifiedRecurrentShadowCanaryError(
                "shadow canary case identity invalid"
            )
        try:
            request = seal_shadow_probe_request(
                raw.get("public_token_ids"),
                raw.get("expected_token_ids"),
                max_tokens=raw.get("max_tokens"),
            )
        except (TypeError, ValueError) as exc:
            raise UnifiedRecurrentShadowCanaryError(
                f"shadow canary case request invalid: {exc}"
            ) from exc
        request_sha256 = request["request_sha256"]
        if request_sha256 in request_ids:
            raise UnifiedRecurrentShadowCanaryError(
                "shadow canary request is duplicated"
            )
        task_ids.add(task_id)
        request_ids.add(request_sha256)
        normalized.append(
            {
                "index": index,
                "task_id": task_id,
                "family": family,
                "request_sha256": request_sha256,
                "request": request,
            }
        )
    return normalized


def seal_shadow_canary_plan(
    cases: Sequence[Mapping[str, Any]],
    *,
    package_id: str,
    controller_sha256: str,
    minimum_wrong_to_right: int = 1,
    maximum_shadow_latency_ms: int = 120_000,
    maximum_latency_ratio_numerator: int = 8,
    maximum_latency_ratio_denominator: int = 1,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Seal public canary identity while retaining token payloads only in memory."""

    normalized = _normalized_cases(cases)
    if (
        not isinstance(package_id, str)
        or not package_id
        or not _is_sha256(controller_sha256)
        or type(minimum_wrong_to_right) is not int
        or not 0 <= minimum_wrong_to_right <= len(normalized)
        or type(maximum_shadow_latency_ms) is not int
        or maximum_shadow_latency_ms < 1
        or type(maximum_latency_ratio_numerator) is not int
        or type(maximum_latency_ratio_denominator) is not int
        or maximum_latency_ratio_numerator < 1
        or maximum_latency_ratio_denominator < 1
    ):
        raise UnifiedRecurrentShadowCanaryError(
            "shadow canary promotion threshold invalid"
        )
    body = {
        "schema": PLAN_SCHEMA,
        "package_id": package_id,
        "controller_sha256": controller_sha256,
        "cases": [
            {key: row[key] for key in ("index", "task_id", "family", "request_sha256")}
            for row in normalized
        ],
        "decision_rule": {
            "all_probes_completed": True,
            "all_shadow_answers_exact": True,
            "minimum_wrong_to_right": minimum_wrong_to_right,
            "maximum_right_to_wrong": 0,
            "maximum_shadow_latency_ms": maximum_shadow_latency_ms,
            "maximum_latency_ratio_numerator": maximum_latency_ratio_numerator,
            "maximum_latency_ratio_denominator": maximum_latency_ratio_denominator,
        },
        "output_exposed": False,
        "serving_authority": False,
    }
    return {**body, "plan_sha256": _sha(body)}, normalized


def _validate_plan(plan: Mapping[str, Any]) -> None:
    if not isinstance(plan, Mapping):
        raise UnifiedRecurrentShadowCanaryError("shadow canary plan is unavailable")
    body = {key: value for key, value in plan.items() if key != "plan_sha256"}
    cases = plan.get("cases")
    rule = plan.get("decision_rule")
    if (
        set(plan)
        != {
            "schema",
            "package_id",
            "controller_sha256",
            "cases",
            "decision_rule",
            "output_exposed",
            "serving_authority",
            "plan_sha256",
        }
        or not isinstance(rule, dict)
        or set(rule)
        != {
            "all_probes_completed",
            "all_shadow_answers_exact",
            "minimum_wrong_to_right",
            "maximum_right_to_wrong",
            "maximum_shadow_latency_ms",
            "maximum_latency_ratio_numerator",
            "maximum_latency_ratio_denominator",
        }
        or plan.get("schema") != PLAN_SCHEMA
        or plan.get("plan_sha256") != _sha(body)
        or not isinstance(plan.get("package_id"), str)
        or not plan["package_id"]
        or not _is_sha256(plan.get("controller_sha256"))
        or not isinstance(cases, list)
        or not 1 <= len(cases) <= MAX_CASES
        or rule.get("all_probes_completed") is not True
        or rule.get("all_shadow_answers_exact") is not True
        or rule.get("maximum_right_to_wrong") != 0
        or plan.get("output_exposed") is not False
        or plan.get("serving_authority") is not False
    ):
        raise UnifiedRecurrentShadowCanaryError("shadow canary plan invalid")
    expected_indices = list(range(len(cases)))
    if [row.get("index") for row in cases if isinstance(row, dict)] != expected_indices:
        raise UnifiedRecurrentShadowCanaryError("shadow canary plan order invalid")
    if len({row.get("task_id") for row in cases}) != len(cases) or len(
        {row.get("request_sha256") for row in cases}
    ) != len(cases):
        raise UnifiedRecurrentShadowCanaryError("shadow canary plan duplicates evidence")
    integer_thresholds = (
        "minimum_wrong_to_right",
        "maximum_shadow_latency_ms",
        "maximum_latency_ratio_numerator",
        "maximum_latency_ratio_denominator",
    )
    if any(type(rule.get(key)) is not int for key in integer_thresholds) or not (
        0 <= rule["minimum_wrong_to_right"] <= len(cases)
        and rule["maximum_shadow_latency_ms"] >= 1
        and rule["maximum_latency_ratio_numerator"] >= 1
        and rule["maximum_latency_ratio_denominator"] >= 1
    ):
        raise UnifiedRecurrentShadowCanaryError("shadow canary plan thresholds invalid")
    for row in cases:
        if (
            not isinstance(row, dict)
            or set(row) != {"index", "task_id", "family", "request_sha256"}
            or not isinstance(row.get("task_id"), str)
            or not row["task_id"]
            or not isinstance(row.get("family"), str)
            or not row["family"]
            or not _is_sha256(row.get("request_sha256"))
        ):
            raise UnifiedRecurrentShadowCanaryError("shadow canary plan case invalid")


def adjudicate_shadow_canary(
    plan: Mapping[str, Any],
    observations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Rebuild a canary verdict from individual no-output worker receipts."""

    _validate_plan(plan)
    cases = plan["cases"]
    if not isinstance(observations, (list, tuple)) or len(observations) != len(cases):
        raise UnifiedRecurrentShadowCanaryError("shadow canary evidence count differs")
    evidence: list[dict[str, Any]] = []
    completed = 0
    shadow_exact = 0
    base_exact = 0
    wrong_to_right = 0
    right_to_wrong = 0
    base_latency_ms = 0
    shadow_latency_ms = 0
    max_shadow_latency_ms = 0
    for case, observation in zip(cases, observations, strict=True):
        if not isinstance(observation, Mapping):
            raise UnifiedRecurrentShadowCanaryError("shadow canary observation invalid")
        receipt = observation.get("receipt")
        result_status = str(observation.get("status") or "unavailable")
        reason = str(observation.get("reason") or "probe_unavailable")
        receipt_errors: list[str] = []
        if isinstance(receipt, dict):
            receipt_errors = shadow_probe_receipt_errors(
                receipt,
                expected_request_sha256=case["request_sha256"],
                expected_package_id=plan["package_id"],
                expected_controller_sha256=plan["controller_sha256"],
            )
            if not receipt_errors and receipt.get("family") != case["family"]:
                receipt_errors.append("shadow_canary_family_binding_differs")
        else:
            receipt_errors = ["shadow_canary_probe_receipt_unavailable"]
        if isinstance(receipt, dict) and not receipt_errors and (
            result_status != receipt.get("status") or reason != receipt.get("reason")
        ):
            receipt_errors.append("shadow_canary_result_receipt_state_differs")
        accepted = not receipt_errors and receipt.get("status") == "completed"
        if accepted:
            completed += 1
            base = bool(receipt["base_exact_match"])
            shadow = bool(receipt["shadow_exact_match"])
            base_exact += int(base)
            shadow_exact += int(shadow)
            wrong_to_right += int(not base and shadow)
            right_to_wrong += int(base and not shadow)
            base_latency_ms += int(receipt["base_latency_ms"])
            shadow_latency_ms += int(receipt["shadow_latency_ms"])
            max_shadow_latency_ms = max(
                max_shadow_latency_ms,
                int(receipt["shadow_latency_ms"]),
            )
        evidence.append(
            {
                "index": case["index"],
                "task_id": case["task_id"],
                "family": case["family"],
                "request_sha256": case["request_sha256"],
                "probe_status": result_status,
                "probe_reason": reason,
                "receipt_sha256": (
                    receipt.get("receipt_sha256")
                    if isinstance(receipt, dict) and _is_sha256(receipt.get("receipt_sha256"))
                    else ""
                ),
                "accepted": accepted,
                "errors": receipt_errors,
            }
        )
    rule = plan["decision_rule"]
    checks = {
        "all_probes_completed": completed == len(cases),
        "all_shadow_answers_exact": shadow_exact == len(cases),
        "minimum_wrong_to_right": wrong_to_right >= rule["minimum_wrong_to_right"],
        "maximum_right_to_wrong": right_to_wrong <= rule["maximum_right_to_wrong"],
        "maximum_shadow_latency_ms": (
            completed == len(cases)
            and max_shadow_latency_ms <= rule["maximum_shadow_latency_ms"]
        ),
        "maximum_aggregate_latency_ratio": (
            completed == len(cases)
            and shadow_latency_ms * rule["maximum_latency_ratio_denominator"]
            <= max(1, base_latency_ms) * rule["maximum_latency_ratio_numerator"]
        ),
    }
    supported = all(checks.values())
    body = {
        "schema": VERDICT_SCHEMA,
        "plan_sha256": plan["plan_sha256"],
        "package_id": plan["package_id"],
        "controller_sha256": plan["controller_sha256"],
        "verdict": SUPPORTED if supported else REFUTED,
        "supported": supported,
        "checks": checks,
        "measurements": {
            "cases": len(cases),
            "completed": completed,
            "base_exact": base_exact,
            "shadow_exact": shadow_exact,
            "wrong_to_right": wrong_to_right,
            "right_to_wrong": right_to_wrong,
            "base_latency_ms": base_latency_ms,
            "shadow_latency_ms": shadow_latency_ms,
            "max_shadow_latency_ms": max_shadow_latency_ms,
        },
        "evidence": evidence,
        "output_exposed": False,
        "serving_authority": False,
    }
    return {**body, "verdict_sha256": _sha(body)}


# A probe is caller-supplied and reaches a resident worker over IPC, so the
# failure surface belongs to the transport rather than to this module.
# asyncio.CancelledError derives from BaseException and passes through
# untouched, which is what a cancelled sweep needs.
_PROBE_TRANSPORT_FAILURES: Final = (
    ArithmeticError,
    AssertionError,
    AttributeError,
    EOFError,
    ImportError,
    LookupError,
    MemoryError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)


async def run_shadow_canary(
    cases: Sequence[Mapping[str, Any]],
    *,
    package_id: str,
    controller_sha256: str,
    probe: Callable[..., Awaitable[Mapping[str, Any]]],
    minimum_wrong_to_right: int = 1,
    maximum_shadow_latency_ms: int = 120_000,
    maximum_latency_ratio_numerator: int = 8,
    maximum_latency_ratio_denominator: int = 1,
    progress: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Execute a bounded case list sequentially and adjudicate every receipt."""

    plan, normalized = seal_shadow_canary_plan(
        cases,
        package_id=package_id,
        controller_sha256=controller_sha256,
        minimum_wrong_to_right=minimum_wrong_to_right,
        maximum_shadow_latency_ms=maximum_shadow_latency_ms,
        maximum_latency_ratio_numerator=maximum_latency_ratio_numerator,
        maximum_latency_ratio_denominator=maximum_latency_ratio_denominator,
    )
    observations: list[dict[str, Any]] = []
    for row in normalized:
        if progress is not None:
            progress(
                {
                    "event": "case_started",
                    "case_index": row["index"],
                    "case_number": row["index"] + 1,
                    "case_count": len(normalized),
                    "task_id": row["task_id"],
                    "family": row["family"],
                }
            )
        request = row["request"]
        try:
            result = await probe(
                request["public_token_ids"],
                request["expected_token_ids"],
                max_tokens=request["max_tokens"],
            )
        except _PROBE_TRANSPORT_FAILURES as exc:
            # Enumerated rather than blanket. A transport or worker failure is
            # a canary observation; an exception class this boundary does not
            # know is a defect in the caller's probe, and turning that into a
            # tidy "probe_exception" row hides it inside a passing sweep.
            record_degradation(
                "unified_recurrent_shadow_canary",
                exc,
                action="recorded the case as a probe failure and continued the sweep",
                severity="warning",
            )
            result = {
                "status": "probe_exception",
                "reason": f"{type(exc).__name__}:{str(exc)[:160]}",
                "receipt": {},
            }
        observations.append(
            {
                "status": result.get("status"),
                "reason": result.get("reason"),
                "receipt": result.get("receipt"),
            }
        )
        if progress is not None:
            progress(
                {
                    "event": "case_completed",
                    "case_index": row["index"],
                    "case_number": row["index"] + 1,
                    "case_count": len(normalized),
                    "task_id": row["task_id"],
                    "family": row["family"],
                    "status": result.get("status"),
                    "reason": result.get("reason"),
                }
            )
    return {
        "plan": plan,
        "verdict": adjudicate_shadow_canary(plan, observations),
    }


__all__ = [
    "MAX_CASES",
    "PLAN_SCHEMA",
    "REFUTED",
    "SUPPORTED",
    "VERDICT_SCHEMA",
    "UnifiedRecurrentShadowCanaryError",
    "adjudicate_shadow_canary",
    "run_shadow_canary",
    "seal_shadow_canary_plan",
]
