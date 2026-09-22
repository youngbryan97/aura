from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.runtime.file_write_gateway import get_file_write_gateway

from .unity_state import FragmentationReport, UnityRepairPlan, UnityState


def unity_summary_payload(
    unity_state: UnityState | None,
    report: FragmentationReport | None = None,
    repair_plan: UnityRepairPlan | None = None,
) -> dict[str, Any]:
    if unity_state is None:
        return {
            "status": "unavailable",
            "unity_score": 0.0,
            "fragmentation_score": 1.0,
            "level": "unknown",
        }
    payload: dict[str, Any] = {
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
    moment = dict((unity_state.metadata or {}).get("mind_moment") or {})
    if moment:
        payload["mind_moment_id"] = moment.get("moment_id")
        payload["causal_closure_score"] = moment.get("closure_score")
        payload["active_subsystems"] = list(moment.get("active_subsystems") or [])
        payload["closure_missing"] = list(moment.get("closure_missing") or [])
        payload["attention"] = moment.get("attention")
        payload["wanting"] = moment.get("wanting")
    return payload


def write_unity_results_artifact(path: str | Path, payload: dict[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    # Skip the write when only the timestamp moved: this artifact is
    # git-tracked, and every suite run used to dirty the working tree with
    # a timestamp-only rewrite (recurring rebase/stash friction).
    try:
        existing = json.loads(target.read_text(encoding="utf-8"))
        if isinstance(existing, dict) and {
            k: v for k, v in existing.items() if k != "timestamp"
        } == {k: v for k, v in payload.items() if k != "timestamp"}:
            return target
    # not a failure: text that is not the JSON this expects is not a record it can
    # read back.
    except (OSError, ValueError):
        pass  # unreadable/missing/legacy artifact: write a fresh copy
    get_file_write_gateway().write_text(
        target,
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
        source="core.unity.unity_receipts.write_results_artifact",
    )
    return target
