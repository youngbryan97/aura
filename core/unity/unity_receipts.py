from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from .unity_state import FragmentationReport, UnityRepairPlan, UnityState


def unity_summary_payload(
    unity_state: UnityState | None,
    report: FragmentationReport | None = None,
    repair_plan: UnityRepairPlan | None = None,
) -> Dict[str, Any]:
    if unity_state is None:
        return {
            "status": "unavailable",
            "unity_score": 0.0,
            "fragmentation_score": 1.0,
            "level": "unknown",
        }
    payload: Dict[str, Any] = {
        "unity_id": unity_state.unity_id,
        "level": unity_state.level,
        "unity_score": unity_state.unity_score,
        "fragmentation_score": unity_state.fragmentation_score,
        "focus_id": unity_state.global_focus_id,
        "periphery": list(unity_state.peripheral_content_ids),
        "repair_needed": unity_state.repair_needed,
        "repair_reasons": list(unity_state.repair_reasons),
    }
    if report is not None:
        payload["top_causes"] = list(report.top_causes)
        payload["safe_to_act"] = report.safe_to_act
        payload["safe_to_self_report"] = report.safe_to_self_report
        payload["user_visible_summary"] = report.user_visible_summary
    if repair_plan is not None:
        payload["repair_plan"] = repair_plan.to_dict()
    return payload


def write_unity_results_artifact(path: str | Path, payload: Dict[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return target
