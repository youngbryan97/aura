"""interface/routes/subsystems.py
──────────────────────────────────
Extracted from server.py — Subsystem status endpoints:
PNEUMA, MHAF, Terminal, Security, Circadian, Substrate,
Skills, Mycelial graph, Knowledge graph, Brain retry, Reboot,
Strategic projects, Action log.
"""
from __future__ import annotations

import logging
import re
import sqlite3
import threading
import time
from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, Response

from core.container import ServiceContainer
from core.runtime.errors import record_degradation
from core.runtime.service_access import optional_service, resolve_orchestrator
from core.utils.concurrency import run_io_bound
from interface.auth import _require_internal, _restore_owner_session_from_request, _verify_token
from interface.routes.devices import _owner_authenticated

logger = logging.getLogger("Aura.Server.Subsystems")

router = APIRouter()
SKILL_EXECUTE_BODY = Body(...)
STANDING_AUTHORITY_GRANT_BODY = Body(...)
STANDING_AUTHORITY_OPTIONAL_BODY = Body(default=None)
_SKILL_EXECUTE_CONTEXT_KEYS = (
    "origin",
    "route",
    "foreground_request",
    "user_explicitly_authorized",
    "user_requested_action",
    "surface",
    "source",
    "explicit_authorization",
    "authorization",
    "scoped_authority",
    "proof_evaluation_contract",
    "action_expectation",
    "expectation",
    "acceptance_criteria",
    "required_evidence",
    "required_evidence_present",
    "user_visible_effect",
    "repair_hint",
    "rollback_hint",
    "allow_partial",
    "requires_sources",
    "disable_auto_action_expectation",
)
_UNTRUSTED_CLIENT_AUTHORITY_KEYS = frozenset(
    {
        "authority_args_digest",
        "capability_token",
        "capability_token_id",
        "scoped_authority",
        "standing_authority_grant_id",
        "standing_authority_receipt_id",
        "standing_authority_token",
    }
)
_CONSEQUENTIAL_SKILL_SCOPES = frozenset(
    {
        "desktop_file_io",
        "foreground_browser_dialogue",
        "foreground_desktop_control",
        "privileged_mutation",
        "state_mutation",
        "subprocess",
    }
)
_MUTATING_FILE_ACTIONS = frozenset({"append", "copy", "delete", "move", "patch", "write"})
_READ_ONLY_MEMORY_ACTIONS = frozenset(
    {"archival_search", "query", "read", "recall", "search"}
)
_SUBSYSTEM_ROUTE_ERRORS = (
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
    sqlite3.Error,
)
_TOPOLOGY_RESPONSE_CACHE_TTL_S = 1.0


def _topology_route_cache_token(mycelium: Any) -> tuple[Any, ...]:
    token_reader = getattr(mycelium, "get_route_cache_token", None)
    if callable(token_reader):
        token = token_reader()
        if isinstance(token, tuple):
            return token
        if isinstance(token, list):
            return tuple(token)
        return (token,)
    return (id(mycelium),)


class _SerializedResponseCache:
    """Singleflight a bounded serialized snapshot without sharing Response objects."""

    def __init__(self, *, ttl_s: float) -> None:
        self._ttl_s = max(0.0, float(ttl_s))
        self._lock = threading.Lock()
        self._owner: Any = None
        self._token: tuple[Any, ...] | None = None
        self._expires_at = 0.0
        self._body: bytes | None = None
        self._status_code = 200
        self._headers: dict[str, str] = {}

    def clear(self) -> None:
        with self._lock:
            self._owner = None
            self._token = None
            self._expires_at = 0.0
            self._body = None
            self._status_code = 200
            self._headers = {}

    def render(self, owner: Any, builder: Any) -> Response:
        with self._lock:
            now = time.monotonic()
            token = _topology_route_cache_token(owner)
            if (
                self._owner is owner
                and self._token == token
                and self._body is not None
                and now < self._expires_at
            ):
                headers = dict(self._headers)
                headers["X-Aura-Snapshot-Cache"] = "hit"
                return Response(
                    content=self._body,
                    status_code=self._status_code,
                    headers=headers,
                )

            response = builder(owner)
            response.headers["X-Aura-Snapshot-Cache"] = "miss"
            final_token = _topology_route_cache_token(owner)
            cacheable = (
                final_token == token
                and response.status_code < 400
                and response.background is None
            )
            if cacheable:
                self._owner = owner
                self._token = final_token
                self._expires_at = time.monotonic() + self._ttl_s
                self._body = bytes(response.body)
                self._status_code = response.status_code
                self._headers = {
                    key: value
                    for key, value in response.headers.items()
                    if key.lower() != "x-aura-snapshot-cache"
                }
            else:
                self._owner = None
                self._token = None
                self._expires_at = 0.0
                self._body = None
                self._headers = {}
            return response


_MYCELIUM_RESPONSE_CACHE = _SerializedResponseCache(
    ttl_s=_TOPOLOGY_RESPONSE_CACHE_TTL_S,
)
_MYCELIAL_GRAPH_RESPONSE_CACHE = _SerializedResponseCache(
    ttl_s=_TOPOLOGY_RESPONSE_CACHE_TTL_S,
)


def _standing_authority_owner_evidence(request: Request) -> dict[str, Any]:
    if not _owner_authenticated(request):
        raise HTTPException(status_code=403, detail="Standing authority is owner-only")
    return {
        "authenticated_principal": "owner",
        "internal_authenticated": True,
        "user_explicitly_authorized": True,
    }


def _normalize_skill_execute_payload(params: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split API skill envelopes into skill params and execution context.

    The live desktop/UI path uses a structured body so it can send foreground
    context with the tool request. Skills, however, must receive only their
    declared schema fields. Accept direct skill params too for backward
    compatibility.
    """

    if not isinstance(params, dict):
        return {}, {}

    envelope_keys = ("input", "params", "arguments", "args", "payload")
    context = dict(params.get("context") or {})
    for key in _SKILL_EXECUTE_CONTEXT_KEYS:
        if key in params and key not in context:
            context[key] = params[key]

    for key in envelope_keys:
        if key not in params:
            continue
        value = params.get(key)
        if isinstance(value, dict):
            return dict(value), context
        if value is None:
            return {}, context
        return {"value": value}, context

    return {
        key: value
        for key, value in params.items()
        if key != "context" and key not in _SKILL_EXECUTE_CONTEXT_KEYS
    }, context


def _slug_for_skill_context(value: object, default: str) -> str:
    text = str(value or default).strip().lower() or default
    return re.sub(r"[^a-z0-9_.:-]+", "_", text)


def _apply_skill_execute_authority_context(
    skill_name: str,
    context: dict[str, Any],
) -> dict[str, Any]:
    """Attach narrow authority metadata for authenticated direct skill calls.

    `/api/skill/execute` is already protected by the internal/token dependencies.
    Production Will default-deny still needs scoped context so it can distinguish
    authenticated operator skill execution from an unscoped background tool call.
    """

    ctx = dict(context or {})
    for key in _UNTRUSTED_CLIENT_AUTHORITY_KEYS:
        ctx.pop(key, None)
    ctx.setdefault("origin", "live_skill_api")
    ctx.setdefault("surface", "desktop-ui")
    ctx.setdefault("source", "api.skill.execute")
    ctx.setdefault("route", "api.skill.execute")
    ctx.setdefault("foreground_request", True)
    ctx.setdefault("user_requested_action", True)
    ctx.setdefault("user_explicitly_authorized", True)
    ctx.setdefault("explicit_authorization", "internal_authenticated_skill_execute")
    ctx.setdefault("authorization", "internal_authenticated_skill_execute")
    route_slug = _slug_for_skill_context(ctx.get("route"), "api.skill.execute")
    skill_slug = _slug_for_skill_context(skill_name, "skill")
    ctx["requested_authority_scope"] = f"api_skill_execute:{route_slug}:{skill_slug}"
    return ctx


def _live_skill_effect_scope(
    skill_name: str,
    params: dict[str, Any],
    engine: Any,
) -> tuple[str, str]:
    resolved = str(skill_name or "")
    resolver = getattr(engine, "resolve_skill_name", None)
    if callable(resolver):
        try:
            resolved = str(resolver(skill_name) or skill_name)
        except (AttributeError, RuntimeError, TypeError, ValueError):
            resolved = str(skill_name or "")
    skills = getattr(engine, "skills", None)
    meta = skills.get(resolved) if isinstance(skills, dict) else None
    classifier = getattr(engine, "_effect_scope_for_execution", None)
    if meta is None or not callable(classifier):
        return resolved, "unknown"
    try:
        scope = str(classifier(resolved, meta, params, {}) or "unknown").strip().lower()
    except (AttributeError, RuntimeError, TypeError, ValueError):
        scope = "unknown"
    return resolved, scope


def _live_skill_is_consequential(
    skill_name: str,
    params: dict[str, Any],
    effect_scope: str,
) -> bool:
    normalized = str(skill_name or "").strip().lower()
    action = str((params or {}).get("action") or "").strip().lower()
    if normalized == "file_operation":
        return action in _MUTATING_FILE_ACTIONS
    if normalized == "memory_ops" and action in _READ_ONLY_MEMORY_ACTIONS:
        return False
    if normalized == "computer_use":
        return effect_scope not in {"read_only", "sandboxed_compute"}
    return effect_scope in _CONSEQUENTIAL_SKILL_SCOPES


def _prepare_live_skill_expectation(
    skill_name: str,
    params: dict[str, Any],
    context: dict[str, Any],
    engine: Any,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    ctx = dict(context or {})
    expectation = None
    resolver = getattr(engine, "action_expectation_for", None)
    if callable(resolver):
        try:
            expectation = resolver(skill_name, params, ctx)
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation("subsystems.skill_expectation", exc)

    if expectation is not None:
        serializer = getattr(expectation, "to_dict", None)
        ctx["action_expectation"] = (
            serializer() if callable(serializer) else expectation
        )

    resolved, effect_scope = _live_skill_effect_scope(skill_name, params, engine)
    if expectation is not None or not _live_skill_is_consequential(
        resolved,
        params,
        effect_scope,
    ):
        return ctx, None

    return ctx, {
        "ok": False,
        "status": "action_expectation_required",
        "error": (
            f"Consequential live skill '{resolved}' requires an action expectation "
            "with acceptance criteria or effect evidence before execution."
        ),
        "skill": resolved,
        "effect_scope": effect_scope,
        "required_contract": {
            "objective": "what the operation must accomplish",
            "acceptance_criteria": ["falsifiable completion criterion"],
            "required_evidence": ["receipt or observed effect field"],
            "repair_hint": "bounded repair action",
            "rollback_hint": "bounded reversal or containment action",
            "allow_partial": False,
        },
    }


def _get_live_orchestrator_state() -> Any | None:
    """Best-effort access to the active runtime state used by the live orchestrator."""
    orch = resolve_orchestrator()
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
    except _SUBSYSTEM_ROUTE_ERRORS as exc:
        record_degradation("subsystems", exc)
        logger.debug("Unable to read latest user message from conversation log: %s", exc)
        return ""


# ── PNEUMA ────────────────────────────────────────────────────

@router.get("/pneuma/status")
async def api_pneuma_status():
    """PNEUMA engine detailed status — precision, neural ODE, topology, free energy."""
    try:
        from core.pneuma.pneuma import get_pneuma
        pn = get_pneuma()
        runtime_state = pn.get_state_dict() if pn else {}
        if not pn or not runtime_state.get("online", False):
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
            "tick_count": runtime_state.get("tick_count", 0),
            "last_tick": runtime_state.get("last_tick", 0.0),
            "loop_errors": runtime_state.get("loop_errors", 0),
            "compute_budget": runtime_state.get("compute_budget", {}),
            "temperature": round(pn.get_llm_temperature(), 4),
            "context_block": block,
            "attractor_count": int(tm.attractor_count) if tm else 0,
            "arousal": arousal,
            "stability": stability,
        })
    except _SUBSYSTEM_ROUTE_ERRORS as e:
        record_degradation('subsystems', e)
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
        runtime_state = mhaf.get_state_dict() if mhaf else {}
        if not mhaf or not runtime_state.get("online", False):
            return JSONResponse({"online": False, "error": "MHAF not running"}, status_code=503)
        nodes = [
            {"name": n.name, "activation": round(float(n.activation), 3)}
            for n in mhaf._nodes.values()
        ]
        return JSONResponse({
            "online": True,
            "tick_count": runtime_state.get("tick_count", 0),
            "last_tick": runtime_state.get("last_tick", 0.0),
            "loop_errors": runtime_state.get("loop_errors", 0),
            "compute_budget": runtime_state.get("compute_budget", {}),
            "phi": round(float(mhaf._global_phi), 4),
            "free_energy": round(float(mhaf._free_energy), 4),
            "nodes": nodes,
            "edge_count": len(mhaf._edges),
            "lexicon": neo.get_lexicon_block() if neo else "",
            "lexicon_size": len(neo._lexicon) if neo else 0,
        })
    except _SUBSYSTEM_ROUTE_ERRORS as e:
        record_degradation('subsystems', e)
        return JSONResponse({"online": False, "error": str(e)}, status_code=500)


# ── Terminal ──────────────────────────────────────────────────

@router.get("/terminal/status")
async def api_terminal_status():
    """TerminalFallback + Watchdog status."""
    try:
        from core.conversation.terminal_chat import get_terminal_fallback, get_terminal_watchdog
        tf = get_terminal_fallback()
        tw = get_terminal_watchdog()
        return JSONResponse({
            "active": tf.is_active,
            "pending_messages": len(tf._pending),
            "watchdog_running": tw._running if tw else False,
            "ui_gone_since": tw._ui_gone_since if tw else None,
        })
    except _SUBSYSTEM_ROUTE_ERRORS as e:
        record_degradation('subsystems', e)
        return JSONResponse({"active": False, "error": str(e)}, status_code=500)


@router.post("/terminal/send")
async def api_terminal_send(request: Request):
    """Queue a message for terminal fallback delivery."""
    try:
        body = await request.json()
        text = str(body.get("text", "")).strip()
        if not text:
            raise HTTPException(status_code=400, detail="text required")
        from core.conversation.terminal_chat import get_terminal_fallback
        queued = get_terminal_fallback().queue_autonomous_message(text)
        if queued is False:
            return JSONResponse({"ok": False, "error": "suppressed by constitution"})
        return JSONResponse({"ok": True, "queued": text})
    except HTTPException:
        raise
    except _SUBSYSTEM_ROUTE_ERRORS as e:
        record_degradation('subsystems', e)
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


# ── Security ──────────────────────────────────────────────────

@router.get("/security/status")
async def api_security_status(request: Request):
    """Security system status — trust level, integrity, threat score."""
    _restore_owner_session_from_request(request)
    result: dict[str, Any] = {}
    try:
        from core.security.trust_engine import get_trust_engine
        result["trust"] = get_trust_engine().get_status()
    except _SUBSYSTEM_ROUTE_ERRORS as e:
        record_degradation('subsystems', e)
        result["trust"] = {"error": str(e)}
    try:
        from core.security.integrity_guardian import get_integrity_guardian
        result["integrity"] = get_integrity_guardian().get_status()
    except _SUBSYSTEM_ROUTE_ERRORS as e:
        record_degradation('subsystems', e)
        result["integrity"] = {"error": str(e)}
    try:
        from core.security.emergency_protocol import get_emergency_protocol
        result["emergency"] = get_emergency_protocol().get_status()
    except _SUBSYSTEM_ROUTE_ERRORS as e:
        record_degradation('subsystems', e)
        result["emergency"] = {"error": str(e)}
    try:
        from core.security.user_recognizer import get_user_recognizer
        result["recognition"] = {"has_passphrase": get_user_recognizer().has_passphrase()}
    except _SUBSYSTEM_ROUTE_ERRORS as e:
        record_degradation('subsystems', e)
        result["recognition"] = {"error": str(e)}
    try:
        from core.security.defensive_runtime import defensive_status

        result["defensive_runtime"] = defensive_status()
    except _SUBSYSTEM_ROUTE_ERRORS as e:
        record_degradation('subsystems', e)
        result["defensive_runtime"] = {"error": str(e)}
    return JSONResponse(result)


@router.post("/security/snapshot")
async def api_security_snapshot():
    """Force an emergency self-preservation snapshot."""
    try:
        from core.security.emergency_protocol import get_emergency_protocol
        ep = get_emergency_protocol()
        path = ep.take_snapshot_now()
        return JSONResponse({"ok": True, "path": str(path) if path else None})
    except _SUBSYSTEM_ROUTE_ERRORS as e:
        record_degradation('subsystems', e)
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
    except _SUBSYSTEM_ROUTE_ERRORS as e:
        record_degradation('subsystems', e)
        return JSONResponse({"error": str(e)}, status_code=500)


# ── Substrate ─────────────────────────────────────────────────

@router.get("/substrate/status")
async def api_substrate_status():
    """CRSM LoRA bridge + Experience Consolidator status."""
    result: dict[str, Any] = {}
    try:
        from core.consciousness.crsm_lora_bridge import get_crsm_lora_bridge
        result["lora_bridge"] = get_crsm_lora_bridge().get_status()
    except _SUBSYSTEM_ROUTE_ERRORS as e:
        record_degradation('subsystems', e)
        result["lora_bridge"] = {"error": str(e)}
    try:
        from core.consciousness.experience_consolidator import get_experience_consolidator
        result["consolidator"] = get_experience_consolidator().get_status()
    except _SUBSYSTEM_ROUTE_ERRORS as e:
        record_degradation('subsystems', e)
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
    except _SUBSYSTEM_ROUTE_ERRORS as e:
        record_degradation('subsystems', e)
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


# ── Mycelium ──────────────────────────────────────────────────

@router.get("/mycelium")
async def api_mycelium():
    """Returns the full Mycelial Network topology, health, and infrastructure map."""
    mycelium = ServiceContainer.get("mycelium", default=None)
    if not mycelium:
        return JSONResponse({"error": "Mycelial Network offline"}, status_code=503)
    return await run_io_bound(
        _MYCELIUM_RESPONSE_CACHE.render,
        mycelium,
        _build_mycelium_response,
    )


def _build_mycelium_response(mycelium: Any) -> JSONResponse:
    """Snapshot and serialize the large topology response off the event loop."""
    snapshotter = getattr(mycelium, "get_runtime_snapshot", None)
    if callable(snapshotter):
        snapshot = snapshotter()
        topology = dict(snapshot.get("topology") or {})
        topology["infrastructure"] = dict(snapshot.get("infrastructure") or {})
    else:
        topology = mycelium.get_network_topology()
        topology["infrastructure"] = mycelium.get_infrastructure_report()
    headers = {}
    topology_revision = topology.get("topology_revision")
    structure_revision = topology.get("topology_structure_revision")
    if isinstance(topology_revision, int):
        headers["X-Aura-Topology-Revision"] = str(topology_revision)
    if isinstance(structure_revision, int):
        headers["X-Aura-Topology-Structure-Revision"] = str(structure_revision)
    return JSONResponse(topology, headers=headers)


@router.get("/mycelial/graph")
async def api_mycelial_graph():
    """Transform the Mycelial Network topology into 3d-force-graph-compatible JSON."""
    mycelium = ServiceContainer.get("mycelium", default=None)
    if not mycelium:
        return JSONResponse({"nodes": [], "links": [], "cohesion": None, "pathway_count": 0})
    return await run_io_bound(
        _MYCELIAL_GRAPH_RESPONSE_CACHE.render,
        mycelium,
        _build_mycelial_graph_response,
    )


def _build_mycelial_graph_response(mycelium: Any) -> JSONResponse:
    """Build and serialize the complete graph response off the event loop."""
    cognitive_state_set = {
        "qualia",
        "affect",
        "personality",
        "memory",
        "substrate",
        "consciousness",
        "attention",
        "sentience",
        "drive",
        "scanner",
    }

    node_intel = {
        "orchestrator": "The Central Command of Aura. Coordinates all cognitive cycles, task dispatch, and subsystem coordination.",
        "personality_engine": "Aura's Core Persona. Manages voice, tone, identity traits, and linguistic style filters.",
        "memory_facade": "The Unified Memory Interface. Routes high-level requests between episodic, semantic, and vector storage layers.",
        "affect_engine": "The Emotional Core. Modulates valence, arousal, and mood based on system state and interactions.",
        "drive_controller": "Intrinsic Motivation System. Manages survival instincts, curiosity, hunger for data, and goal prioritization.",
        "liquid_substrate": "The LNN (Liquid Neural Network) Backbone. Provides time-continuous computational state for attention, affect, and planning.",
        "sovereign_scanner": "System Awareness. High-frequency monitoring of processes, files, and environmental context.",
        "core.orchestrator": "The Central Command of Aura. Coordinates all cognitive cycles and task dispatch.",
        "core.mycelium": "The Mycelial Network. Manages dynamic hyphae connections, unblockable pathways, and system topology.",
        "core.brain": "The Cognitive Engine. Handles reasoning, tool use, and deep LLM integration.",
        "core.memory": "The Multi-layered Persistence System. Manages long-term storage and retrieval across diverse memory types.",
        "core.senses": "The Perceptual Layer. Handles Speech-to-Text, Text-to-Speech, and Vision interfaces.",
        "core.resilience": "The Immunity System. Manages circuit breakers, health heartbeats, and autonomous state recovery.",
        "qualia": "Functional State Descriptor. Tracks sensory and affective markers without claiming subjective qualia.",
        "consciousness": "Global Workspace. Coordinates competing cognitive contents into a selected focus for downstream behavior.",
        "cognition": "Active Reasoning. The processing layer where objectives are broken down into executable actions.",
        "skills": "The Action Library. Encapsulated capabilities that allow Aura to interact with the world.",
        "telemetry": "The Neural Feed. Real-time stream of all internal thoughts, events, and performance metrics.",
        "autonomy": "Self-Directed Agency. The drive to act independently toward defined objectives without user prompting."
    }
    try:
        graph_snapshotter = getattr(mycelium, "get_graph_snapshot", None)
        if callable(graph_snapshotter):
            graph_snapshot = graph_snapshotter()
            topo = dict(graph_snapshot.get("topology") or {})
            mapped_files = dict(graph_snapshot.get("mapped_files") or {})
            centrality_by_module = dict(graph_snapshot.get("centrality") or {})
            critical_modules = list(
                graph_snapshot.get("critical_modules")
                or topo.get("critical_modules")
                or []
            )
            mapping_generation = int(graph_snapshot.get("mapping_generation") or 0)
            mapping_state = str(graph_snapshot.get("mapping_state") or "unknown")
            topology_revision = int(graph_snapshot.get("topology_revision") or 0)
            structure_revision = int(
                graph_snapshot.get("topology_structure_revision") or 0
            )
        else:
            topo = mycelium.get_network_topology()
            snapshotter = getattr(mycelium, "get_mapped_files_snapshot", None)
            mapped_files = (
                snapshotter() if callable(snapshotter) else {}
            )
            centrality_by_module = {}
            critical_modules = list(topo.get("critical_modules") or [])
            mapping_generation = 0
            mapping_state = "legacy"
            topology_revision = 0
            structure_revision = 0
        nodes_map: dict[str, Any] = {}
        links: list[dict[str, Any]] = []

        all_endpoints: set = set()
        for _name, h_data in topo.get("hyphae", {}).items():
            src = h_data.get("source", "")
            tgt = h_data.get("target", "")
            if src:
                all_endpoints.add(src)
            if tgt:
                all_endpoints.add(tgt)
        for mk in mapped_files:
            all_endpoints.add(mk)

        critical_set = set(critical_modules)
        for ep in all_endpoints:
            short_name = ep.split(".")[-1] if "." in ep else ep
            is_critical = ep in critical_set
            is_cognitive_state = any(cn in ep.lower() for cn in cognitive_state_set)
            is_skill = "skills" in ep.lower() or "skill" in ep.lower()
            is_interface = ep.startswith("interface")
            centrality = centrality_by_module.get(ep, 0)

            if is_critical:
                color, ntype, size = "#ff3e5e", "critical", 6 + centrality * 0.5
            elif is_cognitive_state:
                color, ntype, size = "#00e5ff", "cognitive_state", 5
            elif is_skill:
                color, ntype, size = "#00ffa3", "skill", 3
            elif is_interface:
                color, ntype, size = "#ff4fa3", "interface", 2.8 + min(centrality * 0.25, 3)
            else:
                color, ntype, size = "#8a2be2", "core", 2 + min(centrality * 0.3, 4)

            description = node_intel.get(ep, "")
            if not description:
                if is_skill:
                    description = f"Autonomous Skill Nexus for {short_name}. Enables specialized tool usage."
                elif is_cognitive_state:
                    description = f"Cognitive-state module: {short_name}. Contributes functional telemetry or steering signals."
                elif is_critical:
                    description = "Core Subsystem. Critical infrastructure component."
                elif is_interface:
                    description = f"Interface Surface Hypha. Renders or serves {short_name}."
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

        for _name, h_data in topo.get("hyphae", {}).items():
            src, tgt = h_data.get("source", ""), h_data.get("target", "")
            if not src or not tgt or src not in nodes_map or tgt not in nodes_map:
                continue
            is_physical = h_data.get("is_physical", False)
            strength = h_data.get("strength", 1.0)
            if is_physical:
                color = f"rgba(0,180,255,{min(0.5 + strength * 0.08, 0.9):.2f})"
                width, particles, distance = 2.8 + min(strength * 0.45, 4.2), 1, 70
            else:
                color = f"rgba(145,92,255,{min(0.52 + strength * 0.05, 0.88):.2f})"
                width, particles = 2.4 + min(strength * 0.45, 3.8), 2 if strength > 2 else 1
                distance = 58
            links.append({"source": src, "target": tgt, "color": color,
                          "width": round(float(width), 2), "particles": particles, "distance": distance})

        module_by_path = {
            module_data.get("path"): module_key
            for module_key, module_data in mapped_files.items()
            if isinstance(module_data, dict) and module_data.get("path")
        }
        for pw_id, pw_data in topo.get("pathways", {}).items():
            pw_node_id = f"pw:{pw_id}"
            conf = pw_data.get("confidence", 1.0)
            nodes_map[pw_node_id] = {
                "id": pw_node_id, "label": pw_id, "type": "pathway",
                "color": "#00ffa3", "size": 2 + conf * 2, "centrality": 0,
                "description": f"Heuristic Learning Pathway: {pw_id}. Represents an emergent cognitive behavior."
            }
            module_key = module_by_path.get(pw_data.get("source_file"))
            if module_key is not None:
                links.append({"source": pw_node_id, "target": module_key,
                              "color": "rgba(0,255,163,0.5)", "width": 1.0,
                              "particles": 1, "distance": 35})

        try:
            from core.runtime import resource_psutil as psutil

            ram_usage = psutil.virtual_memory().percent
            cpu_usage = psutil.cpu_percent()
        except ImportError:
            ram_usage, cpu_usage = 0.0, 0.0

        if not nodes_map:
            seed_services = [
                ("orchestrator",      "critical",      "#ff3e5e", "Central Command — coordinates all cognitive cycles."),
                ("cognitive_engine",  "critical",      "#ff3e5e", "Cognitive Engine — reasoning, tool use, deep LLM integration."),
                ("llm_router",        "core",          "#8a2be2", "LLM Router — multi-tier failover with circuit breakers."),
                ("memory_facade",     "core",          "#8a2be2", "Memory Facade — unified interface across all memory layers."),
                ("affect_engine",     "cognitive_state", "#00e5ff", "Affect Engine — functional valence, arousal, and mood telemetry."),
                ("liquid_state",      "cognitive_state", "#00e5ff", "Liquid State — time-continuous computational substrate."),
                ("mycelial_network",  "core",          "#8a2be2", "Mycelial Network — dynamic infrastructure topology."),
                ("proactive_presence","core",          "#8a2be2", "Proactive Presence — spontaneous speech and initiative."),
                ("personality_engine","cognitive_state", "#00e5ff", "Personality Engine — voice, tone, and identity-style synthesis."),
                ("voice_engine",      "core",          "#8a2be2", "Voice Engine — TTS/STT pipeline and embodiment."),
                ("goal_hierarchy",    "core",          "#8a2be2", "Goal Hierarchy — motivation and objective management."),
                ("episodic_memory",   "core",          "#8a2be2", "Episodic Memory — experiential trace and recall."),
                ("homeostasis",       "cognitive_state", "#00e5ff", "Homeostasis — integrity, persistence, and resource-pressure regulation."),
            ]
            seed_links = [
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
            for svc_id, ntype, color, desc in seed_services:
                is_live = ServiceContainer.peek(svc_id, default=None) is not None
                nodes_map[svc_id] = {
                    "id": svc_id,
                    "label": svc_id.replace("_", " ").title(),
                    "type": ntype,
                    "color": color if is_live else "#4a4a4a",
                    "size": 5 if ntype == "critical" else (4 if ntype == "cognitive_state" else 3),
                    "description": desc + (" [LIVE]" if is_live else " [OFFLINE]"),
                    "centrality": 3 if ntype == "critical" else 1,
                    "hits": 0, "confidence": 1.0 if is_live else 0.3,
                    "is_critical": ntype == "critical"
                }
            for src, tgt in seed_links:
                if src in nodes_map and tgt in nodes_map:
                    links.append({"source": src, "target": tgt,
                                  "color": "rgba(0,229,255,0.35)", "width": 1.5,
                                  "particles": 1, "distance": 80})

        return JSONResponse(
            {
                "nodes": list(nodes_map.values()),
                "links": links,
                # CP126 40325f75: no topology is an unmeasured cohesion, not a
                # measured 0.5. The UI renders null as "not measured".
                "system_cohesion": topo.get("system_cohesion") if nodes_map else None,
                "cohesion_basis": topo.get("cohesion_basis"),
                "pathway_count": topo.get("pathway_count", 0),
                "mapping_generation": mapping_generation,
                "mapping_state": mapping_state,
                "topology_revision": topology_revision,
                "topology_structure_revision": structure_revision,
                "ram_usage": ram_usage,
                "cpu_usage": cpu_usage,
            },
            headers={
                "X-Aura-Topology-Revision": str(topology_revision),
                "X-Aura-Topology-Structure-Revision": str(structure_revision),
            },
        )
    except _SUBSYSTEM_ROUTE_ERRORS as e:
        record_degradation('subsystems', e)
        logger.error("Mycelial graph generation failed: %s", e, exc_info=True)
        return JSONResponse({"nodes": [], "links": [], "cohesion": None, "pathway_count": 0})


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
    except _SUBSYSTEM_ROUTE_ERRORS as e:
        record_degradation('subsystems', e)
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
    except _SUBSYSTEM_ROUTE_ERRORS as exc:
        record_degradation('subsystems', exc)
        logger.debug("Relationships query failed: %s", exc)
        return JSONResponse({"edges": [], "error": str(exc)})


# ── Brain / Reboot ────────────────────────────────────────────

@router.post("/brain/retry")
async def api_brain_retry(
    _: None = Depends(_require_internal),
):
    """Signal the orchestrator to retry its cognitive engine connection."""
    orch = resolve_orchestrator()
    if orch and hasattr(orch, "retry_brain_connection"):
        await orch.retry_brain_connection()
        return JSONResponse({"status": "retry_sent"})
    return JSONResponse({"status": "orchestrator_unavailable"}, status_code=503)


@router.post("/reboot")
async def api_reboot(
    _: None = Depends(_require_internal),
):
    """Restart the server process, arranging the restart when nothing else will.

    This used to send SIGTERM and document "supervisor restarts". That holds
    only under launchd, which sets AURA_SUPERVISED=1. Started directly — the
    ordinary way, PPID 1, no supervisor anywhere — the desktop's Reboot control
    was a kill switch: Aura went down and never came back, with nothing in the
    UI saying that "Reboot" meant "shut down".
    """
    logger.warning("Reboot requested via API")
    import signal as _sig

    from core.runtime.runtime_relaunch import schedule_relaunch, supervisor_will_restart

    relaunch: dict[str, Any] = {"scheduled": False, "reason": "supervisor_owns_restart"}
    if not supervisor_will_restart():
        relaunch = schedule_relaunch()
        if not relaunch.get("scheduled"):
            # Refuse to become a kill switch. Nothing has been signalled yet,
            # so the runtime is still up and the caller learns why.
            logger.error(
                "Reboot refused: no supervisor and no relaunch could be arranged (%s)",
                relaunch.get("reason"),
            )
            return JSONResponse(
                {
                    "status": "reboot_unavailable",
                    "detail": (
                        "Nothing would restart this runtime, so the request was "
                        "refused instead of shutting Aura down for good."
                    ),
                    "relaunch": relaunch,
                },
                status_code=503,
            )
        logger.warning(
            "Relaunch armed (waiter pid=%s) before shutting down for reboot.",
            relaunch.get("waiter_pid"),
        )

    _sig.raise_signal(_sig.SIGTERM)
    return JSONResponse({"status": "shutting_down", "relaunch": relaunch})


# ── Skills ────────────────────────────────────────────────────

@router.get("/skills")
async def api_skills():
    from interface.routes.system import _collect_skill_catalog_health, _collect_tool_catalog

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
    health = _collect_skill_catalog_health()
    return JSONResponse(
        {"skills": skills_data, "count": len(skills_data), "catalog": catalog, "health": health}
    )


@router.post("/skill/execute")
async def api_skill_execute(
    skill_name: str,
    params: dict[str, Any] = SKILL_EXECUTE_BODY,
    _: None = Depends(_require_internal),
    __: None = Depends(_verify_token)
):
    """Unified skill execution entry-point."""
    logger.info("🎯 API Skill Request: %s", skill_name)

    try:
        intent_router = ServiceContainer.get("intent_router", default=None)
        engine = optional_service("capability_engine", default=None)
        if not intent_router or not engine:
            return JSONResponse({"ok": False, "error": "Skill execution engine not available"}, status_code=503)

        skill_params, execution_context = _normalize_skill_execute_payload(params)
        execution_context = _apply_skill_execute_authority_context(skill_name, execution_context)
        execution_context, expectation_error = _prepare_live_skill_expectation(
            skill_name,
            skill_params,
            execution_context,
            engine,
        )
        if expectation_error is not None:
            return JSONResponse(expectation_error, status_code=422)
        result = await intent_router.route_execution(
            skill_name,
            skill_params,
            engine,
            context=execution_context,
        )

        return JSONResponse(result)
    except _SUBSYSTEM_ROUTE_ERRORS as e:
        record_degradation('subsystems', e)
        logger.error("Skill execution API failed: %s", e)
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


# ── Standing authority ──────────────────────────────────────

@router.get("/authority/standing")
async def api_standing_authority_status(
    request: Request,
    _: None = Depends(_require_internal),
    __: None = Depends(_verify_token),
):
    """Return durable grants and live child-lease state for the owner."""
    _standing_authority_owner_evidence(request)
    from core.executive.standing_authority import get_standing_authority_manager

    manager = get_standing_authority_manager()
    try:
        await manager.initialize()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return JSONResponse(manager.get_status())


@router.post("/authority/standing/grants")
async def api_standing_authority_install(
    request: Request,
    payload: dict[str, Any] = STANDING_AUTHORITY_GRANT_BODY,
    _: None = Depends(_require_internal),
    __: None = Depends(_verify_token),
):
    """Install or replace an owner-defined standing grant."""
    evidence = _standing_authority_owner_evidence(request)
    from core.executive.standing_authority import (
        StandingAuthorityGrant,
        get_standing_authority_manager,
    )

    grant_payload = dict(payload or {})
    grant_payload["issuer"] = "owner_api"
    grant_payload["built_in"] = False
    try:
        grant = StandingAuthorityGrant.from_dict(grant_payload)
        result = await get_standing_authority_manager().install_grant(
            grant,
            actor="api",
            evidence=evidence,
        )
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return JSONResponse(result, status_code=201)


@router.post("/authority/standing/{grant_id}/revoke")
async def api_standing_authority_revoke(
    grant_id: str,
    request: Request,
    payload: dict[str, Any] | None = STANDING_AUTHORITY_OPTIONAL_BODY,
    _: None = Depends(_require_internal),
    __: None = Depends(_verify_token),
):
    """Revoke a grant and invalidate active child leases immediately."""
    evidence = _standing_authority_owner_evidence(request)
    from core.executive.standing_authority import get_standing_authority_manager

    reason = str((payload or {}).get("reason") or "owner_revoked").strip()[:500]
    try:
        result = await get_standing_authority_manager().revoke_grant(
            grant_id,
            actor="api",
            evidence=evidence,
            reason=reason,
        )
    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown standing grant: {grant_id}",
        ) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    return JSONResponse(result)


@router.post("/authority/standing/{grant_id}/restore")
async def api_standing_authority_restore(
    grant_id: str,
    request: Request,
    _: None = Depends(_require_internal),
    __: None = Depends(_verify_token),
):
    """Restore a previously revoked durable grant."""
    evidence = _standing_authority_owner_evidence(request)
    from core.executive.standing_authority import get_standing_authority_manager

    try:
        result = await get_standing_authority_manager().restore_grant(
            grant_id,
            actor="api",
            evidence=evidence,
        )
    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown standing grant: {grant_id}",
        ) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    return JSONResponse(result)


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
        from core.observability.unified_action_log import get_action_log
        log = get_action_log()
        return JSONResponse({"items": log.recent(limit), "stats": log.stats()})
    except _SUBSYSTEM_ROUTE_ERRORS as exc:
        record_degradation('subsystems', exc)
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
    except _SUBSYSTEM_ROUTE_ERRORS as exc:
        record_degradation('subsystems', exc)
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
    except _SUBSYSTEM_ROUTE_ERRORS as e:
        record_degradation('subsystems', e)
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
    except _SUBSYSTEM_ROUTE_ERRORS as e:
        record_degradation('subsystems', e)
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
    except _SUBSYSTEM_ROUTE_ERRORS as e:
        record_degradation('subsystems', e)
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
    except _SUBSYSTEM_ROUTE_ERRORS as e:
        record_degradation('subsystems', e)
        return JSONResponse({"error": str(e)})


@router.get("/code-graph/hotspots")
async def api_code_graph_hotspots(limit: int = 15, _: None = Depends(_require_internal)):
    """Most-called functions in the codebase."""
    try:
        graph = ServiceContainer.get("code_graph", default=None)
        if graph is None:
            return JSONResponse({"error": "Code graph not initialized"})
        return JSONResponse({"hotspots": graph.hotspots(limit=limit)})
    except _SUBSYSTEM_ROUTE_ERRORS as e:
        record_degradation('subsystems', e)
        return JSONResponse({"error": str(e)})


@router.get("/code-graph/orphans")
async def api_code_graph_orphans(limit: int = 20, _: None = Depends(_require_internal)):
    """Functions never called (potential dead code)."""
    try:
        graph = ServiceContainer.get("code_graph", default=None)
        if graph is None:
            return JSONResponse({"error": "Code graph not initialized"})
        return JSONResponse({"orphans": graph.orphans(limit=limit)})
    except _SUBSYSTEM_ROUTE_ERRORS as e:
        record_degradation('subsystems', e)
        return JSONResponse({"error": str(e)})
