"""Bounded live-mind readout for desktop conversation.

This module is the structural bridge between Aura's runtime state and her live
speech path. It collects compact readouts from first-class mind services so the
desktop CognitiveEngine turn can be grounded in current workspace, affect,
drive, outcome, world-model, nociceptive, and phenomenal state.
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

from core.runtime.errors import record_degradation
from core.runtime.service_registry import get_runtime_service

_SNAPSHOT_RECOVERABLE_ERRORS = (
    AttributeError,
    ImportError,
    LookupError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)

REQUIRED_LIVE_MIND_SERVICES = (
    "global_workspace",
    "nociception",
    "affect_grounding",
    "drive_integration",
    "outcome_ledger",
    "scientific_engine",
    "unified_world_model",
    "phenomenal_engine",
)

REQUIRED_LIVE_MIND_SECTIONS = {
    "global_workspace": "global_workspace",
    "nociception": "nociception",
    "affect_grounding": "affect_grounding",
    "drive_integration": "drive_integration",
    "outcome_ledger": "outcome_ledger",
    "scientific_engine": "scientific_engine",
    "unified_world_model": "world_model",
    "phenomenal_engine": "phenomenal_engine",
}

#: Conation is read through its own module rather than the container, because
#: it is a pure computation over state the container already holds and has no
#: lifecycle of its own to register.
LIVE_MIND_OBSERVATION_SERVICES = (
    "phenomenal_knowing",
    "recursive_self_knowing",
    "automatic_self_knowing",
    "screen_perception",
    "perceptual_pump",
)

LIVE_MIND_SERVICE_NAMES = (
    *REQUIRED_LIVE_MIND_SERVICES,
    *LIVE_MIND_OBSERVATION_SERVICES,
)


def _compact(value: Any, *, depth: int = 3, items: int = 16, text: int = 420) -> Any:
    if depth <= 0:
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return type(value).__name__
    if is_dataclass(value):
        try:
            value = asdict(value)
        except _SNAPSHOT_RECOVERABLE_ERRORS as exc:
            record_degradation("live_mind_snapshot", exc, severity="debug")
            return type(value).__name__
    if hasattr(value, "to_dict") and callable(value.to_dict):
        try:
            value = value.to_dict()
        except _SNAPSHOT_RECOVERABLE_ERRORS as exc:
            record_degradation("live_mind_snapshot", exc, severity="debug")
            return type(value).__name__
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for index, (key, child) in enumerate(value.items()):
            if index >= items:
                out["_truncated"] = True
                break
            out[str(key)] = _compact(child, depth=depth - 1, items=items, text=text)
        return out
    if isinstance(value, (list, tuple, set, frozenset)):
        values = list(value)
        out = [_compact(child, depth=depth - 1, items=items, text=text) for child in values[:items]]
        if len(values) > items:
            out.append({"_truncated": True})
        return out
    if isinstance(value, str):
        value = " ".join(value.split())
        return value if len(value) <= text else value[:text].rsplit(" ", 1)[0].strip() + "..."
    if isinstance(value, (int, bool)) or value is None:
        return value
    if isinstance(value, float):
        return round(value, 6)
    return str(value)[:text]


def _service(name: str) -> Any:
    try:
        return get_runtime_service(name, default=None)
    except _SNAPSHOT_RECOVERABLE_ERRORS as exc:
        record_degradation("live_mind_snapshot", exc, severity="debug")
        return None


def _call(service: Any, method: str, **kwargs: Any) -> Any:
    if service is None:
        return None
    fn = getattr(service, method, None)
    if not callable(fn):
        return None
    try:
        return fn(**kwargs)
    except _SNAPSHOT_RECOVERABLE_ERRORS as exc:
        record_degradation("live_mind_snapshot", exc, severity="debug")
        return None


def _affect_grounding_snapshot(service: Any) -> dict[str, Any]:
    if service is None:
        return {}
    gathered = _call(service, "gather")
    engine = gathered if gathered is not None else service
    assessments = _call(engine, "assess")
    dominant = _call(engine, "dominant")
    return {
        "dominant": _compact(dominant),
        "assessments": _compact(assessments, items=8),
    }


def _phenomenal_state(service: Any) -> dict[str, Any]:
    if service is None:
        return {}
    state = getattr(service, "last_state", None)
    if state is None:
        return {"available": True, "last_state": None}
    keys = (
        "t",
        "valence",
        "arousal",
        "free_energy",
        "integration",
        "self_presence",
        "mineness",
        "curiosity",
        "intentional_object",
        "policy_priors",
        "memory_weights",
    )
    out: dict[str, Any] = {"available": True}
    for key in keys:
        if hasattr(state, key):
            out[key] = _compact(getattr(state, key))
    return out


def _frontmost_app_fast() -> str:
    try:
        from core.perception.frontmost_app import frontmost_app_name_fast

        return str(frontmost_app_name_fast() or "")
    except _SNAPSHOT_RECOVERABLE_ERRORS as exc:
        record_degradation("live_mind_snapshot", exc, severity="debug")
        return ""


def assess_live_mind_snapshot(snapshot: dict[str, Any] | None) -> dict[str, Any]:
    """Assess the one canonical structural contract for live desktop speech."""

    if not isinstance(snapshot, dict) or not snapshot:
        return {
            "present": False,
            "ready": False,
            "missing_services": list(REQUIRED_LIVE_MIND_SERVICES),
            "unpopulated_services": list(REQUIRED_LIVE_MIND_SERVICES),
            "unpopulated_sections": list(REQUIRED_LIVE_MIND_SECTIONS.values()),
            "populated_sections": [],
        }

    services = snapshot.get("services_present")
    if not isinstance(services, dict):
        services = {}
    missing = [
        name for name in REQUIRED_LIVE_MIND_SERVICES if not bool(services.get(name))
    ]
    unpopulated_services = [
        service_name
        for service_name, section_name in REQUIRED_LIVE_MIND_SECTIONS.items()
        if not bool(snapshot.get(section_name))
    ]
    unpopulated_sections = [
        REQUIRED_LIVE_MIND_SECTIONS[name] for name in unpopulated_services
    ]
    populated_sections = [
        name
        for name in (
            "global_workspace",
            "nociception",
            "affect_grounding",
            "drive_integration",
            "outcome_ledger",
            "scientific_engine",
            "world_model",
            "phenomenal_engine",
            "phenomenal_knowing",
            "recursive_self_knowing",
            "automatic_self_knowing",
        )
        if bool(snapshot.get(name))
    ]
    return {
        "present": True,
        "ready": not missing and not unpopulated_services,
        "missing_services": missing,
        "unpopulated_services": unpopulated_services,
        "unpopulated_sections": unpopulated_sections,
        "populated_sections": populated_sections,
    }


def _conation_snapshot() -> dict[str, Any]:
    """What is being wanted this turn, and on what evidence.

    One-way. The conative state grounds what she says; nothing she says may
    write it back, which core/conation/invariants.py enforces at the import
    graph rather than by convention.
    """
    try:
        from core.conation.wiring import snapshot as conation_snapshot

        return conation_snapshot()
    except _SNAPSHOT_RECOVERABLE_ERRORS as exc:
        record_degradation(
            "live_mind_snapshot", exc, severity="debug",
            action="conation section omitted from this turn",
        )
        return {"present": False, "reason": "conation unavailable"}


def _phenomena_snapshot() -> dict[str, Any]:
    """The fourteen dispositions, as a count and a short list of concerns.

    Read through the container rather than by importing them. Three of the
    nine packages they live in are ones this foundation may not depend on, and
    routing through the container means an organ that failed to load is
    reported absent instead of taking the turn with it.
    """
    try:
        from core.phenomena_wiring import snapshot as phenomena

        full = phenomena()
        return {
            "running": full.get("running"),
            "of": full.get("of"),
            "absent": [k for k, v in (full.get("present") or {}).items() if not v],
            "concerns": full.get("concerns") or [],
        }
    except _SNAPSHOT_RECOVERABLE_ERRORS as exc:
        record_degradation(
            "live_mind_snapshot", exc, severity="debug",
            action="phenomena section omitted from this turn",
        )
        return {"present": False, "reason": "phenomena unavailable"}


def collect_live_mind_snapshot(*, lane: dict[str, Any] | None = None) -> dict[str, Any]:
    """Collect compact runtime state for one live desktop conversation turn."""
    services = {name: _service(name) for name in LIVE_MIND_SERVICE_NAMES}
    snapshot: dict[str, Any] = {
        "schema": "aura.live_mind_snapshot.v1",
        "lane": _compact(lane or {}, depth=2, items=12),
        "services_present": {
            name: services[name] is not None for name in LIVE_MIND_SERVICE_NAMES
        },
    }
    snapshot["global_workspace"] = _compact(_call(services["global_workspace"], "get_snapshot"))
    snapshot["nociception"] = _compact(_call(services["nociception"], "snapshot"))
    snapshot["affect_grounding"] = _compact(_affect_grounding_snapshot(services["affect_grounding"]))
    snapshot["drive_integration"] = _compact(_call(services["drive_integration"], "state"))
    snapshot["outcome_ledger"] = _compact(_call(services["outcome_ledger"], "stats"))
    snapshot["scientific_engine"] = _compact(_call(services["scientific_engine"], "stats"))
    snapshot["world_model"] = _compact(_call(services["unified_world_model"], "status"))
    snapshot["phenomenal_engine"] = _compact(_phenomenal_state(services["phenomenal_engine"]))
    snapshot["phenomenal_knowing"] = _compact(_call(services["phenomenal_knowing"], "snapshot"))
    snapshot["recursive_self_knowing"] = _compact(_call(services["recursive_self_knowing"], "snapshot"))
    snapshot["automatic_self_knowing"] = _compact(_call(services["automatic_self_knowing"], "snapshot"))
    snapshot["screen_perception"] = _compact(_call(services["screen_perception"], "get_status"))
    snapshot["perceptual_pump"] = _compact(_call(services["perceptual_pump"], "get_status"))
    snapshot["conation"] = _compact(_conation_snapshot())
    snapshot["phenomena"] = _compact(_phenomena_snapshot())
    snapshot["frontmost_app_fast"] = _frontmost_app_fast()
    return snapshot
