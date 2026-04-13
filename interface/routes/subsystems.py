"""interface/routes/subsystems.py
──────────────────────────────────
Extracted from server.py — Subsystem status endpoints:
PNEUMA, MHAF, Terminal, Security, Circadian, Substrate,
Skills, Mycelial graph, Knowledge graph, Brain retry, Reboot,
Strategic projects, Action log.
"""
from __future__ import annotations

import asyncio
import logging
import sqlite3
import time
from typing import Any, Dict, List

from fastapi import APIRouter, Body, Depends, HTTPException, Request
from fastapi.responses import JSONResponse

from core.container import ServiceContainer

from interface.auth import _check_rate_limit, _require_internal, _restore_owner_session_from_request, _verify_token

logger = logging.getLogger("Aura.Server.Subsystems")

router = APIRouter()


def _get_live_orchestrator_state() -> Any | None:
    """Best-effort access to the active runtime state used by the live orchestrator."""
    orch = ServiceContainer.get("orchestrator", default=None)
    if not orch:
        return None

    state = getattr(getattr(orch, "state_repo", None), "_current", None)
    if state is None:
        state = getattr(orch, "state", None) or getattr(orch, "_state", None)
    return state


def _latest_conversation_user_message() -> str:
    """Return the latest user message from the current runtime conversation log."""
    try:
        from interface.routes import chat as chat_routes

        log = getattr(chat_routes, "_conversation_log", None) or []
        if not log:
            return ""
        latest = log[-1]
        return str(latest.get("user") or "").strip()
    except Exception:
        return ""


# ── PNEUMA ────────────────────────────────────────────────────

@router.get("/pneuma/status")
async def api_pneuma_status():
    """PNEUMA engine detailed status — precision, neural ODE, topology, free energy."""
    try:
        from core.pneuma.pneuma import get_pneuma
        pn = get_pneuma()
        if not pn or not pn._running:
            return JSONResponse({"online": False, "error": "PNEUMA not running"}, status_code=503)
        block = pn.get_context_block()
        pe = getattr(pn, "precision", None)
        tm = getattr(pn, "topo_memory", None)
        arousal = stability = 0.0
        if pe and hasattr(pe, "fhn"):
            arousal   = round(float(pe.fhn.state.v), 4)
            stability = round(float(pe.fhn.state.w), 4)
        return JSONResponse({
            "online": True,
            "temperature": round(pn.get_llm_temperature(), 4),
            "context_block": block,
            "attractor_count": int(tm.attractor_count) if tm else 0,
            "arousal": arousal,
            "stability": stability,
        })
    except Exception as e:
        return JSONResponse({"online": False, "error": str(e)}, status_code=500)


# ── MHAF ──────────────────────────────────────────────────────

@router.get("/mhaf/status")
async def api_mhaf_status():
    """MHAF field detailed status — phi, nodes, edges, free energy, private lexicon."""
    try:
        from core.consciousness.mhaf_field import get_mhaf
        from core.consciousness.neologism_engine import get_neologism_engine
        mhaf = get_mhaf()
        neo = get_neologism_engine()
        if not mhaf or not mhaf._running:
            return JSONResponse({"online": False, "error": "MHAF not running"}, status_code=503)
        nodes = [
            {"name": n.name, "activation": round(float(n.activation), 3),
             "phi": round(float(n.phi), 3)}
            for n in mhaf._nodes.values()
        ]
        return JSONResponse({
            "online": True,
            "phi": round(float(mhaf._global_phi), 4),
            "free_energy": round(float(mhaf._free_energy), 4),
            "nodes": nodes,
            "edge_count": len(mhaf._edges),
            "lexicon": neo.get_lexicon_block() if neo else "",
            "lexicon_size": len(neo._lexicon) if neo else 0,
        })
    except Exception as e:
        return JSONResponse({"online": False, "error": str(e)}, status_code=500)


# ── Terminal ──────────────────────────────────────────────────

@router.get("/terminal/status")
async def api_terminal_status():
    """TerminalFallback + Watchdog status."""
    try:
        from core.terminal_chat import get_terminal_fallback, get_terminal_watchdog
        tf = get_terminal_fallback()
        tw = get_terminal_watchdog()
        return JSONResponse({
            "active": tf.is_active,
            "pending_messages": len(tf._pending),
            "watchdog_running": tw._running if tw else False,
            "ui_gone_since": tw._ui_gone_since if tw else None,
        })
    except Exception as e:
        return JSONResponse({"active": False, "error": str(e)}, status_code=500)


@router.post("/terminal/send")
async def api_terminal_send(request: Request):
    """Queue a message for terminal fallback delivery."""
    try:
        body = await request.json()
        text = str(body.get("text", "")).strip()
        if not text:
            raise HTTPException(status_code=400, detail="text required")
        from core.terminal_chat import get_terminal_fallback
        queued = get_terminal_fallback().queue_autonomous_message(text)
        if queued is False:
            return JSONResponse({"ok": False, "error": "suppressed by constitution"})
        return JSONResponse({"ok": True, "queued": text})
    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


# ── Security ──────────────────────────────────────────────────

@router.get("/security/status")
async def api_security_status(request: Request):
    """Security system status — trust level, integrity, threat score."""
    _restore_owner_session_from_request(request)
    result: Dict[str, Any] = {}
    try:
        from core.security.trust_engine import get_trust_engine
        result["trust"] = get_trust_engine().get_status()
    except Exception as e:
        result["trust"] = {"error": str(e)}
    try:
        from core.security.integrity_guardian import get_integrity_guardian
        result["integrity"] = get_integrity_guardian().get_status()
    except Exception as e:
        result["integrity"] = {"error": str(e)}
    try:
        from core.security.emergency_protocol import get_emergency_protocol
        result["emergency"] = get_emergency_protocol().get_status()
    except Exception as e:
        result["emergency"] = {"error": str(e)}
    try:
        from core.security.user_recognizer import get_user_recognizer
        result["recognition"] = {"has_passphrase": get_user_recognizer().has_passphrase()}
    except Exception as e:
        result["recognition"] = {"error": str(e)}
    return JSONResponse(result)


@router.post("/security/snapshot")
async def api_security_snapshot():
    """Force an emergency self-preservation snapshot."""
    try:
        from core.security.emergency_protocol import get_emergency_protocol
        ep = get_emergency_protocol()
        path = ep.take_snapshot_now()
        return JSONResponse({"ok": True, "path": str(path) if path else None})
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


# ── Circadian ─────────────────────────────────────────────────

@router.get("/circadian/status")
async def api_circadian_status():
    """Circadian rhythm engine — phase, arousal baseline, cognitive mode."""
    try:
        from core.senses.circadian import get_circadian
        ce = get_circadian()
        ce.update()
        s = ce.state
        return JSONResponse({
            "phase": s.phase.value,
            "hour": round(s.hour, 2),
            "arousal_baseline": round(s.arousal_baseline, 3),
            "energy_modifier": round(s.energy_modifier, 3),
            "cognitive_mode": s.cognitive_mode,
            "focus_tendency": round(s.focus_tendency, 3),
            "social_warmth": round(s.social_warmth, 3),
            "introspection_bias": round(s.introspection_bias, 3),
            "is_sleep_phase": ce.is_sleep_phase,
            "bg_task_budget": ce.bg_task_budget,
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ── Substrate ─────────────────────────────────────────────────

@router.get("/substrate/status")
async def api_substrate_status():
    """CRSM LoRA bridge + Experience Consolidator status."""
    result: Dict[str, Any] = {}
    try:
        from core.consciousness.crsm_lora_bridge import get_crsm_lora_bridge
        result["lora_bridge"] = get_crsm_lora_bridge().get_status()
    except Exception as e:
        result["lora_bridge"] = {"error": str(e)}
    try:
        from core.consciousness.experience_consolidator import get_experience_consolidator
        result["consolidator"] = get_experience_consolidator().get_status()
    except Exception as e:
        result["consolidator"] = {"error": str(e)}
    return JSONResponse(result)


@router.post("/consolidate/now")
async def api_consolidate_now():
    """Force an immediate identity consolidation cycle."""
    try:
        from core.consciousness.experience_consolidator import get_experience_consolidator
        ec = get_experience_consolidator()
        narrative = await ec.run_now()
        if narrative:
            return JSONResponse({
                "ok": True,
                "version": narrative.version,
                "signature": narrative.signature_phrase,
                "traits": narrative.stable_traits,
            })
        return JSONResponse({"ok": False, "reason": "insufficient material"})
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


# ── Mycelium ──────────────────────────────────────────────────

@router.get("/mycelium")
async def api_mycelium():
    """Returns the full Mycelial Network topology, health, and infrastructure map."""
    mycelium = ServiceContainer.get("mycelium", default=None)
    if not mycelium:
        return JSONResponse({"error": "Mycelial Network offline"}, status_code=503)
    topology = mycelium.get_network_topology()
    topology["infrastructure"] = mycelium.get_infrastructure_report()
    return JSONResponse(topology)


@router.get("/mycelial/graph")
async def api_mycelial_graph():
    """Transform the Mycelial Network topology into 3d-force-graph-compatible JSON."""
    mycelium = ServiceContainer.get("mycelium", default=None)
    if not mycelium:
        return JSONResponse({"nodes": [], "links": [], "cohesion": 0, "pathway_count": 0})

    CONSCIOUSNESS_SET = {'qualia', 'affect', 'personality', 'memory', 'substrate',
                         'consciousness', 'attention', 'sentience', 'drive', 'scanner'}

    NODE_INTEL = {
        "orchestrator": "The Central Command of Aura. Coordinates all cognitive cycles, task dispatch, and subsystem coordination.",
        "personality_engine": "Aura's Core Persona. Manages voice, tone, identity traits, and linguistic style filters.",
        "memory_facade": "The Unified Memory Interface. Routes high-level requests between episodic, semantic, and vector storage layers.",
        "affect_engine": "The Emotional Core. Modulates valence, arousal, and mood based on system state and interactions.",
        "drive_controller": "Intrinsic Motivation System. Manages survival instincts, curiosity, hunger for data, and goal prioritization.",
        "liquid_substrate": "The LNN (Liquid Neural Network) Backbone. Provides the fluid, time-continuous computational environment for consciousness.",
        "sovereign_scanner": "System Awareness. High-frequency monitoring of processes, files, and environmental context.",
        "core.orchestrator": "The Central Command of Aura. Coordinates all cognitive cycles and task dispatch.",
        "core.mycelium": "The Mycelial Network. Manages dynamic hyphae connections, unblockable pathways, and system topology.",
        "core.brain": "The Cognitive Engine. Handles reasoning, tool use, and deep LLM integration.",
        "core.memory": "The Multi-layered Persistence System. Manages long-term storage and retrieval across diverse memory types.",
        "core.senses": "The Perceptual Layer. Handles Speech-to-Text, Text-to-Speech, and Vision interfaces.",
        "core.resilience": "The Immunity System. Manages circuit breakers, health heartbeats, and autonomous state recovery.",
        "qualia": "Phenomenal Experience. The subjective quality of system states and sensory inputs.",
        "consciousness": "Global Workspace. The unified field where fragmented thoughts coalesce into singular focus.",
        "cognition": "Active Reasoning. The processing layer where objectives are broken down into executable actions.",
        "skills": "The Action Library. Encapsulated capabilities that allow Aura to interact with the world.",
        "telemetry": "The Neural Feed. Real-time stream of all internal thoughts, events, and performance metrics.",
        "autonomy": "Self-Directed Agency. The drive to act independently toward defined objectives without user prompting."
    }
    try:
        topo = mycelium.get_network_topology()
        nodes_map: Dict[str, Any] = {}
        links: List[Dict[str, Any]] = []

        all_endpoints: set = set()
        for name, h_data in topo.get("hyphae", {}).items():
            src = h_data.get("source", "")
            tgt = h_data.get("target", "")
            if src: all_endpoints.add(src)
            if tgt: all_endpoints.add(tgt)
        for mk in mycelium.mapped_files:
            all_endpoints.add(mk)

        critical_set = set(topo.get("critical_modules", []))
        for ep in all_endpoints:
            short_name = ep.split(".")[-1] if "." in ep else ep
            is_critical = ep in critical_set
            is_consciousness = any(cn in ep.lower() for cn in CONSCIOUSNESS_SET)
            is_skill = "skills" in ep.lower() or "skill" in ep.lower()
            centrality = mycelium._centrality.get(ep, 0)

            if is_critical:
                color, ntype, size = "#ff3e5e", "critical", 6 + centrality * 0.5
            elif is_consciousness:
                color, ntype, size = "#00e5ff", "consciousness", 5
            elif is_skill:
                color, ntype, size = "#00ffa3", "skill", 3
            else:
                color, ntype, size = "#8a2be2", "core", 2 + min(centrality * 0.3, 4)

            description = NODE_INTEL.get(ep, "")
            if not description:
                if is_skill:
                    description = f"Autonomous Skill Nexus for {short_name}. Enables specialized tool usage."
                elif is_consciousness:
                    description = f"High-order Consciousness Module: {short_name}. Essential for phenomenal awareness."
                elif is_critical:
                    description = f"Core Subsystem. Critical infrastructure component."
                else:
                    description = f"System Substrate Hypha. Modulating {short_name} pathways."

            hits, confidence = 0, 1.0
            if ep in topo.get("pathways", {}):
                pw = topo["pathways"][ep]
                hits = pw.get("hit_count", 0)
                confidence = pw.get("confidence", 1.0)

            nodes_map[ep] = {
                "id": ep,
                "label": short_name.replace("_", " ").title(),
                "type": ntype,
                "color": color,
                "size": round(float(size), 1),
                "description": description,
                "centrality": centrality,
                "hits": hits,
                "confidence": confidence,
                "is_critical": is_critical
            }

        for name, h_data in topo.get("hyphae", {}).items():
            src, tgt = h_data.get("source", ""), h_data.get("target", "")
            if not src or not tgt or src not in nodes_map or tgt not in nodes_map:
                continue
            is_physical = h_data.get("is_physical", False)
            strength = h_data.get("strength", 1.0)
            if is_physical:
                color = f"rgba(0,180,255,{min(0.3 + strength * 0.08, 0.8):.2f})"
                width, particles, distance = 2.0 + min(strength * 0.4, 4.0), 1, 60
            else:
                color = f"rgba(0,229,255,{min(0.3 + strength * 0.05, 0.8):.2f})"
                width, particles = 2.2 + min(strength * 0.4, 3.5), 2 if strength > 2 else 1
                distance = 40
            links.append({"source": src, "target": tgt, "color": color,
                          "width": round(float(width), 2), "particles": particles, "distance": distance})

        for pw_id, pw_data in topo.get("pathways", {}).items():
            pw_node_id = f"pw:{pw_id}"
            conf = pw_data.get("confidence", 1.0)
            nodes_map[pw_node_id] = {
                "id": pw_node_id, "label": pw_id, "type": "pathway",
                "color": "#00ffa3", "size": 2 + conf * 2, "centrality": 0,
                "description": f"Heuristic Learning Pathway: {pw_id}. Represents an emergent cognitive behavior."
            }
            skill = pw_data.get("skill_name", "")
            for mk in mycelium.mapped_files:
                if skill.lower().replace("_", "") in mk.lower().replace("_", ""):
                    links.append({"source": pw_node_id, "target": mk,
                                  "color": "rgba(0,255,163,0.5)", "width": 1.0,
                                  "particles": 1, "distance": 35})
                    break

        try:
            import psutil
            ram_usage = psutil.virtual_memory().percent
            cpu_usage = psutil.cpu_percent()
        except ImportError:
            ram_usage, cpu_usage = 0.0, 0.0

        if not nodes_map:
            SEED_SERVICES = [
                ("orchestrator",      "critical",      "#ff3e5e", "Central Command — coordinates all cognitive cycles."),
                ("cognitive_engine",  "critical",      "#ff3e5e", "Cognitive Engine — reasoning, tool use, deep LLM integration."),
                ("llm_router",        "core",          "#8a2be2", "LLM Router — multi-tier failover with circuit breakers."),
                ("memory_facade",     "core",          "#8a2be2", "Memory Facade — unified interface across all memory layers."),
                ("affect_engine",     "consciousness", "#00e5ff", "Affect Engine — emotional state, valence, arousal, mood."),
                ("liquid_state",      "consciousness", "#00e5ff", "Liquid State — time-continuous neural substrate."),
                ("mycelial_network",  "core",          "#8a2be2", "Mycelial Network — dynamic infrastructure topology."),
                ("proactive_presence","core",          "#8a2be2", "Proactive Presence — spontaneous speech and initiative."),
                ("personality_engine","consciousness", "#00e5ff", "Personality Engine — voice, tone, identity synthesis."),
                ("voice_engine",      "core",          "#8a2be2", "Voice Engine — TTS/STT pipeline and embodiment."),
                ("goal_hierarchy",    "core",          "#8a2be2", "Goal Hierarchy — motivation and objective management."),
                ("episodic_memory",   "core",          "#8a2be2", "Episodic Memory — experiential trace and recall."),
                ("homeostasis",       "consciousness", "#00e5ff", "Homeostasis — integrity, persistence, will-to-live."),
            ]
            SEED_LINKS = [
                ("orchestrator", "cognitive_engine"),
                ("orchestrator", "proactive_presence"),
                ("orchestrator", "goal_hierarchy"),
                ("cognitive_engine", "llm_router"),
                ("cognitive_engine", "affect_engine"),
                ("cognitive_engine", "liquid_state"),
                ("cognitive_engine", "memory_facade"),
                ("memory_facade", "episodic_memory"),
                ("affect_engine", "personality_engine"),
                ("liquid_state", "homeostasis"),
                ("orchestrator", "mycelial_network"),
                ("cognitive_engine", "voice_engine"),
            ]
            for svc_id, ntype, color, desc in SEED_SERVICES:
                is_live = ServiceContainer.get(svc_id, default=None) is not None
                nodes_map[svc_id] = {
                    "id": svc_id,
                    "label": svc_id.replace("_", " ").title(),
                    "type": ntype,
                    "color": color if is_live else "#4a4a4a",
                    "size": 5 if ntype == "critical" else (4 if ntype == "consciousness" else 3),
                    "description": desc + (" [LIVE]" if is_live else " [OFFLINE]"),
                    "centrality": 3 if ntype == "critical" else 1,
                    "hits": 0, "confidence": 1.0 if is_live else 0.3,
                    "is_critical": ntype == "critical"
                }
            for src, tgt in SEED_LINKS:
                if src in nodes_map and tgt in nodes_map:
                    links.append({"source": src, "target": tgt,
                                  "color": "rgba(0,229,255,0.35)", "width": 1.5,
                                  "particles": 1, "distance": 80})

        return JSONResponse({
            "nodes": list(nodes_map.values()),
            "links": links,
            "system_cohesion": topo.get("system_cohesion", 0) if nodes_map else 0.5,
            "pathway_count": topo.get("pathway_count", 0),
            "ram_usage": ram_usage,
            "cpu_usage": cpu_usage
        })
    except Exception as e:
        logger.error("Mycelial graph generation failed: %s", e, exc_info=True)
        return JSONResponse({"nodes": [], "links": [], "cohesion": 0, "pathway_count": 0})


# ── Knowledge Graph ───────────────────────────────────────────

@router.get("/knowledge/graph")
async def api_knowledge_graph(_: None = Depends(_require_internal)):
    """Fetch the current knowledge graph structure for visualization."""
    kg = ServiceContainer.get("knowledge_graph", default=None)
    if not kg:
        return JSONResponse({"nodes": [], "edges": []})

    try:
        if hasattr(kg, "to_vis_data"):
            return JSONResponse(kg.to_vis_data())

        return JSONResponse({
            "nodes": [{"id": 1, "label": "Aura Core", "color": "#8a2be2"}],
            "edges": []
        })
    except Exception as e:
        logger.error("Failed to fetch KG data: %s", e)
        return JSONResponse({"error": "Knowledge graph query failed"}, status_code=500)


@router.get("/knowledge/relationships")
async def api_knowledge_relationships(
    node_id: str = None, direction: str = "both", limit: int = 50,
    _: None = Depends(_require_internal)
):
    """Query relational edges in the knowledge graph."""
    try:
        kg = ServiceContainer.get("knowledge_graph", default=None)
        if not kg or not hasattr(kg, "get_relationships"):
            return JSONResponse({"edges": [], "error": "Knowledge graph unavailable"})

        if node_id:
            edges = kg.get_relationships(node_id, direction=direction)
        else:
            with kg._get_conn() as conn:
                conn.row_factory = sqlite3.Row
                c = conn.cursor()
                c.execute("SELECT from_id, to_id, relation_type, strength FROM relationships LIMIT ?", (limit,))
                edges = [dict(row) for row in c.fetchall()]

        return JSONResponse({"edges": edges, "count": len(edges)})
    except Exception as exc:
        logger.debug("Relationships query failed: %s", exc)
        return JSONResponse({"edges": [], "error": str(exc)})


# ── Brain / Reboot ────────────────────────────────────────────

@router.post("/brain/retry")
async def api_brain_retry(
    _: None = Depends(_require_internal),
):
    """Signal the orchestrator to retry its cognitive engine connection."""
    orch = ServiceContainer.get("orchestrator", default=None)
    if orch and hasattr(orch, "retry_brain_connection"):
        await orch.retry_brain_connection()
        return JSONResponse({"status": "retry_sent"})
    return JSONResponse({"status": "orchestrator_unavailable"}, status_code=503)


@router.post("/reboot")
async def api_reboot(
    _: None = Depends(_require_internal),
):
    """Restart the server process (send SIGTERM; supervisor restarts)."""
    logger.warning("Reboot requested via API")
    import signal as _sig
    _sig.raise_signal(_sig.SIGTERM)
    return JSONResponse({"status": "shutting_down"})


# ── Skills ────────────────────────────────────────────────────

@router.get("/skills")
async def api_skills():
    from interface.routes.system import _collect_tool_catalog
    catalog = _collect_tool_catalog()
    skills_data = [
        {
            "name": item["name"],
            "state": item["state"],
            "availability": item["availability"],
            "description": item["description"],
            "input_summary": item["input_summary"],
            "example_usage": item["example_usage"],
            "risk_class": item["risk_class"],
            "route_class": item["route_class"],
            "last_error": item["last_error"],
            "degraded_reason": item["degraded_reason"],
        }
        for item in catalog
    ]
    return JSONResponse({"skills": skills_data, "count": len(skills_data), "catalog": catalog})


@router.post("/skill/execute")
async def api_skill_execute(
    skill_name: str,
    params: Dict[str, Any] = Body(...),
    _: None = Depends(_require_internal),
    __: None = Depends(_verify_token)
):
    """Unified skill execution entry-point."""
    logger.info("🎯 API Skill Request: %s", skill_name)

    try:
        intent_router = ServiceContainer.get("intent_router", default=None)
        engine = ServiceContainer.get("capability_engine", default=None)
        if not intent_router or not engine:
            return JSONResponse({"ok": False, "error": "Skill execution engine not available"}, status_code=503)

        result = await intent_router.route_execution(skill_name, params, engine)

        return JSONResponse(result)
    except Exception as e:
        logger.error("Skill execution API failed: %s", e)
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


# ── Strategic Projects ────────────────────────────────────────

@router.get("/strategic/projects")
async def api_strategic_projects(_: None = Depends(_require_internal)):
    """Fetch all active strategic projects and their tasks for the Zenith HUD."""
    planner = ServiceContainer.get("strategic_planner", default=None)
    if not planner:
        return JSONResponse({"error": "Strategic planner not available"}, status_code=503)

    projects = planner.store.get_active_projects()
    result = []
    for p in projects:
        tasks = planner.store.get_tasks_for_project(p.id)
        result.append({
            "id": p.id,
            "name": p.name,
            "goal": p.goal,
            "status": p.status,
            "progress": {
                "completed": sum(1 for t in tasks if t.status == "completed"),
                "total": len(tasks)
            },
            "tasks": [
                {
                    "id": t.id,
                    "description": t.description,
                    "status": t.status,
                    "priority": t.priority
                } for t in tasks
            ]
        })
    return JSONResponse({"projects": result})


# ── Action Log ────────────────────────────────────────────────

@router.get("/action-log")
async def api_action_log(limit: int = 50, _: None = Depends(_require_internal)):
    """Unified behavioral assertion log — every action from every generation with gate status."""
    try:
        from core.unified_action_log import get_action_log
        log = get_action_log()
        return JSONResponse({"items": log.recent(limit), "stats": log.stats()})
    except Exception as exc:
        return JSONResponse({"items": [], "stats": {}, "error": str(exc)})


# ── Voice / Substrate Voice Engine ───────────────────────────

@router.get("/voice/state")
async def api_voice_state(_: None = Depends(_require_internal)):
    """Live voice state — how the substrate is shaping Aura's speech right now.

    Returns the current SpeechProfile compilation: word budget, tone,
    energy, warmth, directness, fragment ratio, follow-up probability,
    and the raw substrate snapshot that drove it.
    """
    try:
        from core.voice.substrate_voice_engine import get_live_voice_state

        latest_user_message = _latest_conversation_user_message()
        live_state = _get_live_orchestrator_state()
        state = get_live_voice_state(
            state=live_state,
            user_message=latest_user_message,
            origin="user",
            refresh=live_state is not None,
        )
        return JSONResponse({"voice": state})
    except Exception as exc:
        return JSONResponse({"voice": {}, "error": str(exc)})


@router.post("/voice/affect-modulate")
async def api_voice_affect_modulate(
    request: Request,
    _: None = Depends(_require_internal),
):
    """Hold Aura's voice compilation on a named affect preset for demos.

    This is a diagnostic/demo tool. Instead of relying on a single live-state
    mutation that can be immediately washed out by the runtime, it applies a
    temporary override inside the substrate voice engine so the selected mood
    stays visible long enough to demo clearly.

    Body:
      {"mood": "energized" | "tired" | "frustrated" | "warm" | "curious" | "neutral",
       "hold_seconds": 30}
    """
    body = await request.json()
    mood = str(body.get("mood", "neutral")).lower().strip()
    try:
        hold_seconds = float(body.get("hold_seconds", 30.0))
    except (TypeError, ValueError):
        hold_seconds = 30.0
    hold_seconds = max(1.0, min(300.0, hold_seconds))

    presets = {
        "energized": {"valence": 0.6, "arousal": 0.8, "curiosity": 0.8, "engagement": 0.8, "social_hunger": 0.5, "dominant_emotion": "joy"},
        "tired": {"valence": -0.1, "arousal": 0.2, "curiosity": 0.2, "engagement": 0.25, "social_hunger": 0.3, "dominant_emotion": "contemplation"},
        "frustrated": {"valence": -0.5, "arousal": 0.75, "curiosity": 0.2, "engagement": 0.5, "social_hunger": 0.2, "dominant_emotion": "frustration"},
        "warm": {"valence": 0.5, "arousal": 0.45, "curiosity": 0.5, "engagement": 0.7, "social_hunger": 0.8, "dominant_emotion": "love"},
        "curious": {"valence": 0.3, "arousal": 0.65, "curiosity": 0.85, "engagement": 0.75, "social_hunger": 0.5, "dominant_emotion": "curiosity"},
        "neutral": {"valence": 0.0, "arousal": 0.5, "curiosity": 0.5, "engagement": 0.5, "social_hunger": 0.5, "dominant_emotion": "neutral"},
    }

    preset = presets.get(mood)
    if not preset:
        return JSONResponse(
            {"error": f"Unknown mood: {mood}. Options: {list(presets.keys())}"},
            status_code=400,
        )

    try:
        from core.voice.substrate_voice_engine import get_substrate_voice_engine

        sve = get_substrate_voice_engine()
        demo_override = sve.set_demo_affect_override(
            mood=mood,
            affect=preset,
            hold_seconds=hold_seconds,
        )
        state = _get_live_orchestrator_state()

        logger.info(
            "🎭 [Voice Demo] Affect override '%s' held for %.1fs: %s",
            mood,
            hold_seconds,
            preset,
        )

        profile = sve.compile_profile(state=state, user_message="", origin="user")
        return JSONResponse({
            "shifted_to": mood,
            "affect": preset,
            "hold_seconds": hold_seconds,
            "demo_override": demo_override,
            "resulting_voice": {
                "word_budget": profile.word_budget,
                "tone": profile.tone_override or "default",
                "energy": round(profile.energy, 2),
                "warmth": round(profile.warmth, 2),
                "directness": round(profile.directness, 2),
                "playfulness": round(profile.playfulness, 2),
                "capitalization": profile.capitalization,
                "vocabulary": profile.vocabulary_tier,
                "fragment_ratio": round(profile.fragment_ratio, 2),
                "question_probability": round(profile.question_probability, 2),
                "followup_probability": round(profile.followup_probability, 2),
                "exclamation_allowed": profile.exclamation_allowed,
            },
        })
    except Exception as e:
        logger.debug("Voice profile compilation after shift failed: %s", e)
        return JSONResponse({
            "shifted_to": mood,
            "affect": preset,
            "hold_seconds": hold_seconds,
            "error": str(e),
        })


# ── Code Graph (Self-Knowledge) ─────────────────────────────────────────────

@router.get("/code-graph/stats")
async def api_code_graph_stats(_: None = Depends(_require_internal)):
    """Code graph statistics — how well Aura knows her own codebase."""
    try:
        graph = ServiceContainer.get("code_graph", default=None)
        if graph is None:
            return JSONResponse({"status": "not_initialized"})
        return JSONResponse(graph.get_stats())
    except Exception as e:
        return JSONResponse({"error": str(e)})


@router.get("/code-graph/search")
async def api_code_graph_search(q: str, type: str = "", limit: int = 20, _: None = Depends(_require_internal)):
    """Search symbols in the code graph."""
    try:
        graph = ServiceContainer.get("code_graph", default=None)
        if graph is None:
            return JSONResponse({"error": "Code graph not initialized"})
        results = graph.search_symbols(q, sym_type=type or None, limit=limit)
        return JSONResponse({"query": q, "results": results, "count": len(results)})
    except Exception as e:
        return JSONResponse({"error": str(e)})


@router.get("/code-graph/who-calls")
async def api_code_graph_who_calls(name: str, limit: int = 20, _: None = Depends(_require_internal)):
    """Find all callers of a function."""
    try:
        graph = ServiceContainer.get("code_graph", default=None)
        if graph is None:
            return JSONResponse({"error": "Code graph not initialized"})
        callers = graph.who_calls(name, limit=limit)
        return JSONResponse({"function": name, "callers": callers, "count": len(callers)})
    except Exception as e:
        return JSONResponse({"error": str(e)})


@router.get("/code-graph/hotspots")
async def api_code_graph_hotspots(limit: int = 15, _: None = Depends(_require_internal)):
    """Most-called functions in the codebase."""
    try:
        graph = ServiceContainer.get("code_graph", default=None)
        if graph is None:
            return JSONResponse({"error": "Code graph not initialized"})
        return JSONResponse({"hotspots": graph.hotspots(limit=limit)})
    except Exception as e:
        return JSONResponse({"error": str(e)})


@router.get("/code-graph/orphans")
async def api_code_graph_orphans(limit: int = 20, _: None = Depends(_require_internal)):
    """Functions never called (potential dead code)."""
    try:
        graph = ServiceContainer.get("code_graph", default=None)
        if graph is None:
            return JSONResponse({"error": "Code graph not initialized"})
        return JSONResponse({"orphans": graph.orphans(limit=limit)})
    except Exception as e:
        return JSONResponse({"error": str(e)})
