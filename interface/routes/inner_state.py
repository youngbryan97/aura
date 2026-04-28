"""interface/routes/inner_state.py -- Proof Surface API
=======================================================
Exposes Aura's internal coherence in a way people can see.

This is the visible proof layer showing:
  - Current self-state (identity, condition, commitments)
  - Unified Will decisions (last 5 with full provenance)
  - Drive levels (curiosity, energy, social, competence)
  - World state (environment, user activity, salient events)
  - Active initiatives and last selected
  - System coherence metrics

This endpoint is how you prove to yourself and anyone watching
that the system is doing what you think it's doing.
"""
from __future__ import annotations
from core.runtime.errors import record_degradation


import logging
import time
from typing import Any, Dict

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from core.container import ServiceContainer

logger = logging.getLogger("Aura.ProofSurface")

router = APIRouter(prefix="/api", tags=["inner-state"])


@router.get("/inner-state")
async def get_inner_state() -> JSONResponse:
    """Return the full inner state proof surface.

    This is the single endpoint that proves Aura is a unified organism.
    """
    result: Dict[str, Any] = {
        "timestamp": time.time(),
        "proof_version": "1.0",
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
    except Exception as e:
        record_degradation('inner_state', e)
        result["will"] = {"error": str(e)}

    # 2. Canonical Self — identity, condition, values
    try:
        canonical = ServiceContainer.get("canonical_self", default=None)
        if canonical:
            from dataclasses import asdict
            try:
                self_dict = asdict(canonical)
                # Trim for readability
                for key in list(self_dict.keys()):
                    if isinstance(self_dict[key], (list, dict)) and len(str(self_dict[key])) > 500:
                        self_dict[key] = str(self_dict[key])[:500] + "..."
                result["self"] = self_dict
            except Exception:
                result["self"] = {"name": getattr(canonical, "identity", {}).get("name", "Aura")}
        else:
            result["self"] = {"status": "not_booted"}
    except Exception as e:
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
            try:
                import asyncio
                status = asyncio.get_event_loop().run_until_complete(drive.get_status())
                result["drives"]["status"] = status
            except RuntimeError:
                # Already in async context
                result["drives"]["status"] = "async_context"
        else:
            result["drives"] = {"status": "not_booted"}
    except Exception as e:
        record_degradation('inner_state', e)
        result["drives"] = {"error": str(e)}

    # 4. World State
    try:
        from core.world_state import get_world_state
        ws = get_world_state()
        result["world"] = ws.get_status()
        result["world"]["context_summary"] = ws.get_context_summary()
        result["world"]["salient_events"] = ws.get_salient_events(limit=5)
    except Exception as e:
        record_degradation('inner_state', e)
        result["world"] = {"error": str(e)}

    # 5. Initiative Synthesizer
    try:
        from core.initiative_synthesis import get_initiative_synthesizer
        synth = get_initiative_synthesizer()
        result["synthesis"] = synth.get_status()
        result["synthesis"]["recent"] = synth.get_recent_syntheses(n=3)
    except Exception as e:
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
    except Exception as e:
        record_degradation('inner_state', e)
        result["last_initiative"] = {"error": str(e)}

    # 7. Substrate coherence
    try:
        field = ServiceContainer.get("unified_field", default=None)
        if field:
            result["coherence"] = {
                "field_coherence": round(field.get_coherence(), 4),
            }
            try:
                result["coherence"]["phi_contribution"] = round(field.get_phi_contribution(), 4)
            except Exception:
                pass
        else:
            result["coherence"] = {"status": "not_booted"}

        phi = ServiceContainer.get("phi_core", default=None)
        if phi and hasattr(phi, "get_live_phi"):
            result["coherence"]["phi"] = round(float(phi.get_live_phi(include_surrogate=True)), 6)
        elif phi and hasattr(phi, "current_phi"):
            result["coherence"]["phi"] = round(phi.current_phi, 6)
    except Exception as e:
        record_degradation('inner_state', e)
        result["coherence"] = {"error": str(e)}

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
    except Exception as e:
        record_degradation('inner_state', e)
        result["goals"] = {"error": str(e)}

    # 9. LLM tier health
    try:
        gate = ServiceContainer.get("inference_gate", default=None)
        if gate:
            if hasattr(gate, "ensure_all_tiers_healthy"):
                import asyncio
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        result["llm_tiers"] = {"status": "check_via_tick"}
                    else:
                        result["llm_tiers"] = loop.run_until_complete(gate.ensure_all_tiers_healthy())
                except RuntimeError:
                    result["llm_tiers"] = {"status": "async_context"}
            if hasattr(gate, "get_conversation_status"):
                result["cortex_lane"] = gate.get_conversation_status()
        else:
            result["llm_tiers"] = {"status": "not_booted"}
    except Exception as e:
        record_degradation('inner_state', e)
        result["llm_tiers"] = {"error": str(e)}

    # 10. Continuous cognition loop
    try:
        from core.continuous_cognition import get_continuous_cognition
        ccl = get_continuous_cognition()
        result["cognition_loop"] = ccl.get_status()
    except Exception as e:
        record_degradation('inner_state', e)
        result["cognition_loop"] = {"error": str(e)}

    # 10. Governance enforcement status
    try:
        from core.governance_context import get_governance_status
        result["governance"] = get_governance_status()
    except Exception as e:
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
    except Exception as e:
        record_degradation('inner_state', e)
        result["affect"] = {"error": str(e)}

    return JSONResponse(content=result)


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
    except Exception as e:
        record_degradation('inner_state', e)
        return JSONResponse(content={"error": str(e)}, status_code=500)
