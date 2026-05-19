"""interface/routes/inner_state.py -- Inner State Runtime API
=============================================================
Exposes Aura's internal coherence and causal control state.

This is the visible runtime layer showing:
  - Current self-state (identity, condition, commitments)
  - Unified Will decisions (last 5 with full provenance)
  - Drive levels (curiosity, energy, social, competence)
  - World state (environment, user activity, salient events)
  - Active initiatives and last selected
  - System coherence metrics

This endpoint is a runtime receipt surface: it shows which internal
systems are alive, what they most recently decided, and what causal
state is available to shape behavior.
"""
from __future__ import annotations

import asyncio
import inspect
import logging
import time
from dataclasses import asdict
from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from core.container import ServiceContainer
from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.InnerState")

router = APIRouter(prefix="/api", tags=["inner-state"])
_INNER_STATE_ERRORS = (
    AttributeError,
    ConnectionError,
    ImportError,
    KeyError,
    LookupError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)


async def _await_maybe(value: Any, *, deadline_seconds: float = 2.0) -> Any:
    if inspect.isawaitable(value):
        return await asyncio.wait_for(value, timeout=deadline_seconds)
    return value


def _build_unity_surface() -> dict[str, Any]:
    from core.unity.unity_receipts import unity_summary_payload

    unity_state = ServiceContainer.get("unity_state", default=None)
    unity_report = ServiceContainer.get("unity_fragmentation_report", default=None)
    unity_repair = ServiceContainer.get("unity_repair_plan", default=None)
    unity_workspace = ServiceContainer.get("unity_workspace_frame", default=None)

    payload = unity_summary_payload(unity_state, unity_report, unity_repair)
    if unity_state is None:
        return payload

    focus_summary = ""
    contents = list(getattr(unity_state, "contents", []) or [])
    focus_id = getattr(unity_state, "global_focus_id", None)
    for item in contents:
        if getattr(item, "content_id", "") == focus_id:
            focus_summary = str(getattr(item, "summary", "") or "")
            break

    payload["focus"] = focus_summary
    payload["suppressed_drafts"] = [
        {
            "draft_id": str(getattr(item, "draft_id", "") or ""),
            "claim": str(getattr(item, "claim", "") or "")[:180],
            "support": float(getattr(item, "support", 0.0) or 0.0),
            "conflict": float(getattr(item, "conflict", 0.0) or 0.0),
            "suppressed_reason": str(getattr(item, "suppressed_reason", "") or ""),
        }
        for item in list(getattr(unity_state, "draft_bindings", []) or [])[1:5]
    ]
    payload["workspace_frame"] = (
        unity_workspace.to_dict()
        if unity_workspace and hasattr(unity_workspace, "to_dict")
        else {"status": "unavailable"}
    )
    payload["repair_plan"] = (
        unity_repair.to_dict()
        if unity_repair and hasattr(unity_repair, "to_dict")
        else None
    )
    payload["created_at"] = float(getattr(unity_state, "created_at", time.time()) or time.time())
    return payload


@router.get("/inner-state")
async def get_inner_state() -> JSONResponse:
    """Return the full inner state runtime surface.

    This is the single endpoint that exposes Aura's unified runtime state.
    """
    result: dict[str, Any] = {
        "timestamp": time.time(),
        "surface_version": "1.0",
    }

    # 1. Unified Will — last 5 decisions with provenance
    try:
        from core.will import get_will
        will = get_will()
        result["will"] = {
            "status": will.get_status(),
            "recent_decisions": will.get_recent_decisions(n=5),
            "recent_refusals": will.get_recent_refusals(n=3),
        }
    except _INNER_STATE_ERRORS as e:
        record_degradation('inner_state', e)
        result["will"] = {"error": str(e)}

    # 2. Canonical Self — identity, condition, values
    try:
        canonical = ServiceContainer.get("canonical_self", default=None)
        if canonical:
            try:
                self_dict = asdict(canonical)
                # Trim for readability
                for key in list(self_dict.keys()):
                    if isinstance(self_dict[key], (list, dict)) and len(str(self_dict[key])) > 500:
                        self_dict[key] = str(self_dict[key])[:500] + "..."
                result["self"] = self_dict
            except _INNER_STATE_ERRORS:
                result["self"] = {"name": getattr(canonical, "identity", {}).get("name", "Aura")}
        else:
            result["self"] = {"status": "not_booted"}
    except _INNER_STATE_ERRORS as e:
        record_degradation('inner_state', e)
        result["self"] = {"error": str(e)}

    # 3. Drive levels
    try:
        drive = ServiceContainer.get("drive_engine", default=None)
        if drive:
            result["drives"] = {
                "vector": drive.get_drive_vector() if hasattr(drive, "get_drive_vector") else {},
                "status": {},
            }
            if hasattr(drive, "get_status"):
                result["drives"]["status"] = await _await_maybe(drive.get_status())
        else:
            result["drives"] = {"status": "not_booted"}
    except _INNER_STATE_ERRORS as e:
        record_degradation('inner_state', e)
        result["drives"] = {"error": str(e)}

    # 4. World State
    try:
        from core.world_state import get_world_state
        ws = get_world_state()
        result["world"] = ws.get_status()
        result["world"]["context_summary"] = ws.get_context_summary()
        result["world"]["salient_events"] = ws.get_salient_events(limit=5)
    except _INNER_STATE_ERRORS as e:
        record_degradation('inner_state', e)
        result["world"] = {"error": str(e)}

    # 5. Initiative Synthesizer
    try:
        from core.initiative_synthesis import get_initiative_synthesizer
        synth = get_initiative_synthesizer()
        result["synthesis"] = synth.get_status()
        result["synthesis"]["recent"] = synth.get_recent_syntheses(n=3)
    except _INNER_STATE_ERRORS as e:
        record_degradation('inner_state', e)
        result["synthesis"] = {"error": str(e)}

    # 6. Initiative Arbiter — last selection
    try:
        arbiter = ServiceContainer.get("initiative_arbiter", default=None)
        if arbiter:
            history = arbiter.get_selection_history()
            if history:
                last = history[-1]
                result["last_initiative"] = {
                    "goal": str(last.initiative.get("goal", ""))[:200] if last.initiative else "",
                    "score": round(last.final_score, 4),
                    "rationale": last.rationale,
                    "scores": {k: round(v, 3) for k, v in last.scores.items()},
                }
            else:
                result["last_initiative"] = {"status": "no_selections_yet"}
        else:
            result["last_initiative"] = {"status": "not_booted"}
    except _INNER_STATE_ERRORS as e:
        record_degradation('inner_state', e)
        result["last_initiative"] = {"error": str(e)}

    # 6b. Affective Steering
    try:
        steering = ServiceContainer.get("affective_steering", default=None)
        if steering and hasattr(steering, "_vectors"):
            ncs = ServiceContainer.get("neurochemical_system", default=None)
            moods = ncs.get_mood_vector() if ncs else {}
            active_weights = {}
            for name, sv in steering._vectors.items():
                try:
                    active_weights[name] = round(sv.compute_weight(moods), 4)
                except _INNER_STATE_ERRORS as exc:
                    record_degradation("inner_state", exc)
                    logger.debug("Affective steering weight unavailable for %s: %s", name, exc)
            result["affective_steering"] = {
                "current_mood_vector": {k: round(v, 4) for k, v in moods.items()},
                "active_steering_weights": active_weights,
            }
        else:
            result["affective_steering"] = {"status": "not_booted"}
    except _INNER_STATE_ERRORS as e:
        record_degradation('inner_state', e)
        result["affective_steering"] = {"error": str(e)}

    # 7. Substrate coherence
    try:
        field = ServiceContainer.get("unified_field", default=None)
        if field:
            result["coherence"] = {
                "field_coherence": round(field.get_coherence(), 4),
            }
            try:
                result["coherence"]["phi_contribution"] = round(field.get_phi_contribution(), 4)
            except _INNER_STATE_ERRORS as exc:
                record_degradation("inner_state", exc)
                logger.debug("Phi contribution unavailable: %s", exc)
        else:
            result["coherence"] = {"status": "not_booted"}

        phi = ServiceContainer.get("phi_core", default=None)
        if phi and hasattr(phi, "get_live_phi"):
            result["coherence"]["phi"] = round(float(phi.get_live_phi(include_surrogate=True)), 6)
        elif phi and hasattr(phi, "current_phi"):
            result["coherence"]["phi"] = round(phi.current_phi, 6)
    except _INNER_STATE_ERRORS as e:
        record_degradation('inner_state', e)
        result["coherence"] = {"error": str(e)}

    # 8b. Unity layer
    try:
        result["unity"] = _build_unity_surface()
        if isinstance(result.get("coherence"), dict) and isinstance(result["unity"], dict):
            result["coherence"]["unity_score"] = result["unity"].get("unity_score")
            result["coherence"]["fragmentation_score"] = result["unity"].get("fragmentation_score")
    except _INNER_STATE_ERRORS as e:
        record_degradation('inner_state', e)
        result["unity"] = {"error": str(e)}

    # 8. Active goals
    try:
        goal_engine = ServiceContainer.get("goal_engine", default=None)
        if goal_engine:
            active = goal_engine.get_active_goals(limit=5, include_external=False)
            result["goals"] = [
                {
                    "objective": g.get("objective", g.get("name", ""))[:100],
                    "status": g.get("status"),
                    "priority": g.get("priority"),
                    "horizon": g.get("horizon"),
                }
                for g in active
            ]
        else:
            result["goals"] = []
    except _INNER_STATE_ERRORS as e:
        record_degradation('inner_state', e)
        result["goals"] = {"error": str(e)}

    # 9. LLM tier health
    try:
        gate = ServiceContainer.get("inference_gate", default=None)
        if gate:
            if hasattr(gate, "ensure_all_tiers_healthy"):
                result["llm_tiers"] = await _await_maybe(gate.ensure_all_tiers_healthy())
            if hasattr(gate, "get_conversation_status"):
                result["cortex_lane"] = gate.get_conversation_status()
        else:
            result["llm_tiers"] = {"status": "not_booted"}
    except _INNER_STATE_ERRORS as e:
        record_degradation('inner_state', e)
        result["llm_tiers"] = {"error": str(e)}

    # 10. Continuous cognition loop
    try:
        from core.continuous_cognition import get_continuous_cognition
        ccl = get_continuous_cognition()
        result["cognition_loop"] = ccl.get_status()
    except _INNER_STATE_ERRORS as e:
        record_degradation('inner_state', e)
        result["cognition_loop"] = {"error": str(e)}

    # 10. Governance enforcement status
    try:
        from core.governance_context import get_governance_status
        result["governance"] = get_governance_status()
    except _INNER_STATE_ERRORS as e:
        record_degradation('inner_state', e)
        result["governance"] = {"error": str(e)}

    # 10. Affect state
    try:
        affect = ServiceContainer.get("affect_engine", default=None) or ServiceContainer.get("affect_facade", default=None)
        if affect:
            if hasattr(affect, "get_state_sync"):
                state = affect.get_state_sync()
                if isinstance(state, dict):
                    result["affect"] = {k: round(v, 3) if isinstance(v, float) else v
                                        for k, v in state.items()}
                else:
                    result["affect"] = {"valence": getattr(state, "valence", 0),
                                        "arousal": getattr(state, "arousal", 0)}
            else:
                result["affect"] = {"status": "no_sync_api"}
        else:
            result["affect"] = {"status": "not_booted"}
    except _INNER_STATE_ERRORS as e:
        record_degradation('inner_state', e)
        result["affect"] = {"error": str(e)}

    # 11. Philosophy / receipt stream surfaces
    try:
        grounding = ServiceContainer.get("sensorimotor_grounding_bridge", default=None)
        generator = ServiceContainer.get("substrate_token_generator", default=None)
        lora = ServiceContainer.get("online_lora_governor", default=None)
        overt = ServiceContainer.get("overt_action_loop", default=None)
        result["philosophy_surface"] = {
            "cli_flag": "python aura_main.py --philosophy",
            "sensorimotor_grounding": grounding.status() if grounding and hasattr(grounding, "status") else {"status": "not_booted"},
            "substrate_token_generator": (
                generator.last_generation.to_dict()
                if generator and getattr(generator, "last_generation", None)
                else {"status": "no_generation_yet"}
            ),
            "online_lora": (
                {
                    "enabled": lora.enabled(),
                    "last_receipt": lora.last_receipt.to_dict() if lora.last_receipt else None,
                }
                if lora
                else {"status": "not_booted"}
            ),
            "overt_action_loop": (
                overt.status() if overt and hasattr(overt, "status") else {"status": "not_booted"}
            ),
        }
    except _INNER_STATE_ERRORS as e:
        record_degradation('inner_state', e)
        result["philosophy_surface"] = {"error": str(e)}

    # 12. Complete subsystem component registry
    try:
        result["subsystems"] = ServiceContainer.get_all_subsystem_statuses()
    except _INNER_STATE_ERRORS as e:
        record_degradation('inner_state', e)
        result["subsystems"] = {"error": str(e)}

    return JSONResponse(content=result)


@router.get("/unity")
async def get_unity_state() -> JSONResponse:
    try:
        return JSONResponse(content=_build_unity_surface())
    except _INNER_STATE_ERRORS as e:
        record_degradation('inner_state', e)
        return JSONResponse(content={"error": str(e)}, status_code=500)


@router.get("/inner-state/will-receipt/{receipt_id}")
async def verify_will_receipt(receipt_id: str) -> JSONResponse:
    """Verify that a specific WillReceipt exists in the audit trail.

    This is the provability endpoint: any action can be traced back.
    """
    try:
        from core.will import get_will
        verified = get_will().verify_receipt(receipt_id)
        return JSONResponse(content={
            "receipt_id": receipt_id,
            "verified": verified,
            "timestamp": time.time(),
        })
    except _INNER_STATE_ERRORS as e:
        record_degradation('inner_state', e)
        return JSONResponse(content={"error": str(e)}, status_code=500)
