"""Aura's governed introspection and control surface for physical reach."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Mapping
from typing import Any

from core.skills.what_every_skill_gives_back import THE_SHARED_RESULT
from core.container import ServiceContainer
from core.embodiment.world_bridge import Channel, get_world_bridge
from core.governance.capability_chain import CapabilityViolation, get_capability_issuer
from core.governance.will import ActionDomain, get_will
from core.reality_reach.attachment_authority import (
    ATTACHMENT_AUTHORITY_ACTION,
    MANIFEST_MIGRATION_AUTHORITY_ACTION,
)
from core.reality_reach.attachments import AttachmentAccess
from core.reality_reach.metrology import (
    AcquisitionChannel,
    AcquisitionMode,
    AcquisitionTask,
    EvidenceSource,
    MetrologyError,
)
from core.runtime.audit_chain import canonical_json
from core.runtime.errors import record_degradation
from core.skills.base_skill import BaseSkill

_HOME_ASSISTANT_CONTROL_DOMAINS = frozenset(
    {"climate", "fan", "input_boolean", "light", "switch"}
)


def _boolean_parameter(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError("boolean parameter must be true or false")


def _bounded_int_parameter(
    value: Any,
    *,
    name: str,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    if value is None or value == "":
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if not minimum <= parsed <= maximum:
        raise ValueError(f"{name} must lie inside [{minimum}, {maximum}]")
    return parsed


def _bounded_float_parameter(
    value: Any,
    *,
    name: str,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not minimum <= parsed <= maximum:
        raise ValueError(f"{name} must lie inside [{minimum}, {maximum}]")
    return parsed


def _service(name: str) -> Any | None:
    try:
        return ServiceContainer.get(name, default=None)
    except (ImportError, AttributeError, RuntimeError, TypeError) as exc:
        record_degradation(
            "embodiment_skill",
            exc,
            severity="warning",
            action=f"reported {name} unavailable without inventing physical capability",
        )
        return None


def _world_result(result: Any) -> dict[str, Any]:
    return {
        "ok": bool(getattr(result, "ok", False)),
        "receipt_id": str(getattr(result, "receipt_id", "") or ""),
        "status": str(getattr(result, "status", "") or ""),
        "data": getattr(result, "data", None),
        "error": str(getattr(result, "error", "") or ""),
        "transport_succeeded": getattr(result, "transport_succeeded", None),
        "effect_verified": getattr(result, "effect_verified", None),
        "manual_reconciliation_required": bool(
            getattr(result, "manual_reconciliation_required", False)
        ),
    }


class EmbodimentSkill(BaseSkill):  # type: ignore[misc]  # skipped import is untyped
    """Observe, discover, attach, focus, and control declared physical surfaces."""
    #: What a caller gets back. The shared part only: every skill here
    #: returns `ok`, and a schema claiming to be complete would be wrong
    #: for every one that adds a field.
    result_schema = THE_SHARED_RESULT


    name = "embodiment"
    description = (
        "Use my live physical body and Reality Reach fabric: inspect connected "
        "sensors and actuators, discover nearby configured devices, propose a "
        "connection, focus attention on a sensor, read observations, or execute "
        "a governed and verified physical command. I can also inspect durable "
        "sensor history, active alarms, and quarantined physical evidence."
        " I can run calibrated synchronized measurements and explicitly separated "
        "live, simulated, or hardware-in-loop experiments with uncertainty receipts."
    )
    effect_scope = "external_io"
    retry_safe = False
    timeout_seconds = 90.0
    inputs = {
        "action": (
            "inventory | discover | candidates | connection_requests | "
            "request_connection | authorize_connection | rotate_trust_custody | "
            "focus_sensor | pause_sensors | resume_sensors | "
            "pause_sensor_attention | resume_sensor_attention | latest_observations | "
            "observation_history | active_alarms | acknowledge_alarm | "
            "observation_quarantine | middleware_status | call_service | "
            "metrology_status | list_calibrations | run_acquisition | "
            "restore_live_measurement | "
            "start_action | action_status | action_feedback | wait_action | "
            "cancel_action | activate_physical_node | deactivate_physical_node | "
            "query_device | command_device"
        ),
        "device_id": "Hardware device id for query or command.",
        "candidate_id": "Discovered candidate id for request_connection.",
        "request_id": (
            "Pending attachment request id, or idempotency id for a managed service call."
        ),
        "channel_id": "Reality Reach channel or prefix for query/focus.",
        "target_value": "Numeric target for an attached Reality Reach actuator channel.",
        "access": "observe, or observe+control when proposing a connection.",
        "persistent": "Whether bounded trust may survive a runtime migration.",
        "grant_ttl_s": "Requested trust lifetime within the enforced policy ceiling.",
        "command": "Declared hardware command or Home Assistant operation.",
        "endpoint_id": "Managed telemetry, service, or action endpoint id.",
        "node_id": "Managed physical node id for lifecycle transitions.",
        "goal_id": "Durable identity for a long-running physical action.",
        "timeout_s": "Bounded service or action wait deadline.",
        "preempt": "Whether a new action may safely preempt the active goal.",
        "after_sequence": "Return action feedback after this sequence.",
        "parameters": "Bounded command/effect parameters.",
        "channels": (
            "Measurement channels as ids, or {channel_id, expected_source} records "
            "for explicit hardware-in-loop partitioning."
        ),
        "mode": "live, simulation, or hardware_in_loop measurement mode.",
        "scenario_id": "Required stable scenario identity for simulation or HIL evidence.",
        "sample_count": "Bounded synchronized sample-set count.",
        "sample_interval_s": "Bounded delay between sample sets.",
        "max_capture_skew_ns": "Maximum accepted timestamp skew inside a sample set.",
        "require_calibration": "Whether every requested channel must have valid calibration.",
    }

    async def execute(
        self,
        goal: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        params = goal.get("params", goal)
        if not isinstance(params, Mapping):
            return {"ok": False, "error": "embodiment parameters must be a mapping"}
        action = str(params.get("action") or "inventory").strip().lower()
        if action in {"inventory", "list_devices"}:
            return self._inventory()
        if action == "discover":
            broker = _service("reality_attachment_broker")
            if broker is None:
                return {"ok": False, "error": "physical attachment broker is offline"}
            candidates = await broker.discover()
            return {
                "ok": True,
                "candidates": [item.to_dict() for item in candidates],
                "connection_requests": [item.to_dict() for item in broker.requests()],
                "summary": f"Discovered {len(candidates)} declared physical surfaces.",
            }
        if action == "candidates":
            broker = _service("reality_attachment_broker")
            if broker is None:
                return {"ok": False, "error": "physical attachment broker is offline"}
            candidates = broker.candidates()
            return {"ok": True, "candidates": [item.to_dict() for item in candidates]}
        if action == "connection_requests":
            broker = _service("reality_attachment_broker")
            if broker is None:
                return {"ok": False, "error": "physical attachment broker is offline"}
            requests = broker.requests()
            return {"ok": True, "connection_requests": [item.to_dict() for item in requests]}
        if action == "request_connection":
            return await self._request_connection(params)
        if action == "authorize_connection":
            return await self._authorize_connection(params, context)
        if action == "rotate_trust_custody":
            broker = _service("reality_attachment_broker")
            if broker is None:
                return {"ok": False, "error": "physical attachment broker is offline"}
            receipt = await broker.rotate_trust_custody()
            return {
                "ok": True,
                "rotation_receipt": dict(receipt),
                "attachments": broker.status(),
            }
        if action == "focus_sensor":
            router = _service("reality_observation_router")
            selector = str(params.get("channel_id") or params.get("selector") or "").strip()
            if router is None or not selector:
                return {"ok": False, "error": "a live router and channel selector are required"}
            subscription = router.focus(
                selector,
                duration_s=float(params.get("duration_s") or 30.0),
                max_rate_hz=float(params.get("max_rate_hz") or 4.0),
                min_salience=float(params.get("min_salience") or 0.0),
            )
            return {
                "ok": True,
                "subscription": subscription.to_dict(),
                "summary": f"Focused physical attention on {selector} for a bounded interval.",
            }
        if action in {"pause_sensors", "resume_sensors"}:
            router = _service("reality_observation_router")
            if router is None:
                return {"ok": False, "error": "physical observation router is offline"}
            (router.pause if action == "pause_sensors" else router.resume)()
            return {"ok": True, "observation_router": router.status()}
        if action in {"pause_sensor_attention", "resume_sensor_attention"}:
            router = _service("reality_observation_router")
            if router is None:
                return {"ok": False, "error": "physical observation router is offline"}
            (
                router.pause_attention
                if action == "pause_sensor_attention"
                else router.resume_attention
            )()
            return {"ok": True, "observation_router": router.status()}
        if action == "latest_observations":
            router = _service("reality_observation_router")
            if router is None:
                return {"ok": False, "error": "physical observation router is offline"}
            prefix = str(params.get("channel_id") or "").strip().lower()
            latest = router.latest()
            if prefix:
                latest = {
                    key: value for key, value in latest.items() if key.startswith(prefix)
                }
            return {"ok": True, "observations": latest, "router": router.status()}
        if action == "observation_history":
            historian = _service("reality_historian")
            if historian is None:
                return {"ok": False, "error": "physical historian is offline"}
            try:
                limit = _bounded_int_parameter(
                    params.get("limit"),
                    name="limit",
                    default=100,
                    minimum=1,
                    maximum=500,
                )
                before_row_id = params.get("before_row_id")
                if before_row_id is not None and before_row_id != "":
                    before_row_id = _bounded_int_parameter(
                        before_row_id,
                        name="before_row_id",
                        default=1,
                        minimum=1,
                        maximum=2**63 - 1,
                    )
                else:
                    before_row_id = None
            except ValueError as exc:
                return {"ok": False, "error": str(exc)}
            history = await historian.replay_history(
                channel_id=str(params.get("channel_id") or "").strip().lower() or None,
                before_row_id=before_row_id,
                limit=limit,
            )
            historian_status = await asyncio.to_thread(historian.status)
            return {"ok": True, "history": history, "historian": historian_status}
        if action == "active_alarms":
            historian = _service("reality_historian")
            if historian is None:
                return {"ok": False, "error": "physical historian is offline"}
            try:
                limit = _bounded_int_parameter(
                    params.get("limit"),
                    name="limit",
                    default=100,
                    minimum=1,
                    maximum=500,
                )
            except ValueError as exc:
                return {"ok": False, "error": str(exc)}
            alarms = await historian.active_alarms(limit=limit)
            historian_status = await asyncio.to_thread(historian.status)
            return {
                "ok": True,
                "active_alarms": list(alarms),
                "historian": historian_status,
            }
        if action == "acknowledge_alarm":
            historian = _service("reality_historian")
            channel_id = str(params.get("channel_id") or "").strip().lower()
            if historian is None or not channel_id:
                return {
                    "ok": False,
                    "error": "physical historian and channel_id are required",
                }
            try:
                receipt = await historian.acknowledge_alarm(
                    channel_id,
                    actor="aura",
                )
            except LookupError:
                return {
                    "ok": False,
                    "error": "no active physical alarm exists for that channel",
                }
            return {
                "ok": True,
                "acknowledgement": receipt,
                "summary": "I acknowledged the alarm without clearing its physical state.",
            }
        if action == "observation_quarantine":
            historian = _service("reality_historian")
            if historian is None:
                return {"ok": False, "error": "physical historian is offline"}
            try:
                limit = _bounded_int_parameter(
                    params.get("limit"),
                    name="limit",
                    default=100,
                    minimum=1,
                    maximum=500,
                )
            except ValueError as exc:
                return {"ok": False, "error": str(exc)}
            quarantined = await historian.quarantine(limit=limit)
            historian_status = await asyncio.to_thread(historian.status)
            return {
                "ok": True,
                "quarantine": list(quarantined),
                "historian": historian_status,
            }
        if action in {
            "metrology_status",
            "list_calibrations",
            "run_acquisition",
            "restore_live_measurement",
        }:
            return await self._metrology(action, params)
        if action in {
            "middleware_status",
            "call_service",
            "start_action",
            "action_status",
            "action_feedback",
            "wait_action",
            "cancel_action",
            "activate_physical_node",
            "deactivate_physical_node",
        }:
            return await self._middleware(action, params)
        if action == "query_device":
            return await self._query(params)
        if action == "command_device":
            return await self._command(params)
        return {"ok": False, "error": f"unknown embodiment action: {action}"}

    @staticmethod
    def _inventory() -> dict[str, Any]:
        manager = _service("hardware_manager")
        reality = _service("reality_reach")
        router = _service("reality_observation_router")
        broker = _service("reality_attachment_broker")
        historian = _service("reality_historian")
        metrology = _service("reality_metrology")
        middleware = _service("reality_middleware")
        body = _service("body_schema")
        devices = manager.list_devices() if manager is not None else []
        declarations = (
            [item.to_dict() for item in reality.declarations()]
            if reality is not None
            else []
        )
        body_map = body.get_body_map() if body is not None else {}
        physical_limbs = {
            name: limb
            for name, limb in body_map.items()
            if str(limb.get("source") or "").startswith("reality:")
        }
        return {
            "ok": True,
            "devices": devices,
            "channels": declarations,
            "physical_limbs": physical_limbs,
            "observation_router": router.status() if router is not None else None,
            "historian": (
                historian.health_snapshot()
                if historian is not None
                and callable(getattr(historian, "health_snapshot", None))
                else historian.status()
                if historian is not None
                else None
            ),
            "managed_physical_runtime": (
                middleware.status() if middleware is not None else None
            ),
            "metrology": metrology.status() if metrology is not None else None,
            "attachments": broker.status() if broker is not None else None,
            "summary": (
                f"My physical body currently exposes {len(declarations)} channels "
                f"across {len(physical_limbs)} sensor or actuator limbs."
            ),
        }

    @staticmethod
    async def _metrology(action: str, params: Mapping[str, Any]) -> dict[str, Any]:
        service = _service("reality_metrology")
        if service is None:
            return {"ok": False, "error": "physical metrology service is offline"}
        if action == "metrology_status":
            return {"ok": True, "metrology": service.status()}
        if action == "list_calibrations":
            return {
                "ok": True,
                "calibrations": list(service.calibrations()),
                "metrology": service.status(),
            }
        if action == "restore_live_measurement":
            receipt = await service.restore_live(reason="aura_requested_restoration")
            return {
                "ok": True,
                "restoration": receipt,
                "metrology": service.status(),
            }
        try:
            raw_channels = params.get("channels")
            if isinstance(raw_channels, str):
                raw_channels = [item.strip() for item in raw_channels.split(",") if item.strip()]
            if not isinstance(raw_channels, (list, tuple)) or not raw_channels:
                raise ValueError("channels must be a non-empty bounded list")
            channels: list[AcquisitionChannel] = []
            for item in raw_channels:
                if isinstance(item, Mapping):
                    channel_id = str(item.get("channel_id") or "")
                    expected = EvidenceSource(
                        str(item.get("expected_source") or "live").strip().lower()
                    )
                else:
                    channel_id = str(item)
                    expected = EvidenceSource.LIVE
                channels.append(AcquisitionChannel(channel_id, expected))
            mode = AcquisitionMode(str(params.get("mode") or "live").strip().lower())
            task_id = str(params.get("task_id") or f"aura.acquisition.{uuid.uuid4().hex}")
            task = AcquisitionTask(
                task_id=task_id,
                channels=tuple(channels),
                mode=mode,
                sample_count=_bounded_int_parameter(
                    params.get("sample_count"),
                    name="sample_count",
                    default=1,
                    minimum=1,
                    maximum=1024,
                ),
                sample_interval_s=_bounded_float_parameter(
                    params.get("sample_interval_s"),
                    name="sample_interval_s",
                    default=0.0,
                    minimum=0.0,
                    maximum=60.0,
                ),
                timeout_s=_bounded_float_parameter(
                    params.get("timeout_s"),
                    name="timeout_s",
                    default=30.0,
                    minimum=0.05,
                    maximum=86_400.0,
                ),
                max_capture_skew_ns=_bounded_int_parameter(
                    params.get("max_capture_skew_ns"),
                    name="max_capture_skew_ns",
                    default=100_000_000,
                    minimum=0,
                    maximum=10_000_000_000,
                ),
                require_calibration=_boolean_parameter(
                    params.get("require_calibration"),
                    default=False,
                ),
                scenario_id=str(params.get("scenario_id") or ""),
            )
            receipt = await service.acquire(task)
        except (LookupError, MetrologyError, TypeError, ValueError) as exc:
            return {"ok": False, "error": str(exc), "metrology": service.status()}
        return {
            "ok": True,
            "acquisition": receipt.to_dict(),
            "metrology": service.status(),
            "summary": (
                f"Completed {receipt.sample_sets} synchronized sample set(s) "
                f"across {len(receipt.summaries)} channel(s); live mode restored."
            ),
        }

    @staticmethod
    async def _middleware(action: str, params: Mapping[str, Any]) -> dict[str, Any]:
        middleware = _service("reality_middleware")
        if middleware is None:
            return {"ok": False, "error": "managed physical runtime is offline"}
        try:
            if action == "middleware_status":
                return {"ok": True, "managed_physical_runtime": middleware.status()}
            if action in {"activate_physical_node", "deactivate_physical_node"}:
                node_id = str(params.get("node_id") or "").strip().lower()
                if not node_id:
                    raise ValueError("node_id is required")
                operation = (
                    middleware.activate_node
                    if action == "activate_physical_node"
                    else middleware.deactivate_node
                )
                ok = await operation(node_id)
                return {
                    "ok": bool(ok),
                    "node": middleware.node_status(node_id),
                }
            endpoint_id = str(params.get("endpoint_id") or "").strip().lower()
            if action == "call_service":
                request = params.get("parameters", params.get("request", {}))
                if not endpoint_id or not isinstance(request, Mapping):
                    raise ValueError("endpoint_id and a request mapping are required")
                timeout_s = _bounded_float_parameter(
                    params.get("timeout_s"),
                    name="timeout_s",
                    default=10.0,
                    minimum=0.01,
                    maximum=300.0,
                )
                receipt = await middleware.call_service(
                    endpoint_id,
                    request,
                    request_id=str(params.get("request_id") or "").strip() or None,
                    timeout_s=timeout_s,
                )
                return {"ok": receipt.ok, "service_receipt": receipt.to_dict()}
            goal_id = str(params.get("goal_id") or "").strip().lower()
            if action == "start_action":
                request = params.get("parameters", params.get("request", {}))
                if not endpoint_id or not isinstance(request, Mapping):
                    raise ValueError("endpoint_id and an action request mapping are required")
                timeout_s = _bounded_float_parameter(
                    params.get("timeout_s"),
                    name="timeout_s",
                    default=300.0,
                    minimum=0.1,
                    maximum=86400.0,
                )
                action_record = await middleware.start_action(
                    endpoint_id,
                    request,
                    goal_id=goal_id or None,
                    timeout_s=timeout_s,
                    preempt=_boolean_parameter(params.get("preempt"), default=False),
                )
                return {"ok": True, "action": action_record}
            if not goal_id:
                raise ValueError("goal_id is required")
            if action == "action_status":
                return {"ok": True, "action": middleware.action_status(goal_id)}
            if action == "action_feedback":
                after_sequence = _bounded_int_parameter(
                    params.get("after_sequence"),
                    name="after_sequence",
                    default=0,
                    minimum=0,
                    maximum=2**31 - 1,
                )
                return {
                    "ok": True,
                    "goal_id": goal_id,
                    "feedback": middleware.action_feedback(
                        goal_id,
                        after_sequence=after_sequence,
                    ),
                }
            if action == "wait_action":
                timeout_s = _bounded_float_parameter(
                    params.get("timeout_s"),
                    name="timeout_s",
                    default=30.0,
                    minimum=0.01,
                    maximum=86400.0,
                )
                return {
                    "ok": True,
                    "action": await middleware.wait_action(goal_id, timeout_s=timeout_s),
                }
            if action == "cancel_action":
                return {
                    "ok": True,
                    "action": await middleware.cancel_action(
                        goal_id,
                        reason=str(params.get("reason") or "Aura cancelled the action")[:320],
                    ),
                }
            raise ValueError(f"unsupported managed physical action: {action}")
        except (
            LookupError,
            OSError,
            RuntimeError,
            TimeoutError,
            TypeError,
            ValueError,
        ) as exc:
            return {"ok": False, "error": f"{type(exc).__name__}:{exc}"[:320]}

    @staticmethod
    async def _request_connection(params: Mapping[str, Any]) -> dict[str, Any]:
        broker = _service("reality_attachment_broker")
        candidate_id = str(params.get("candidate_id") or "").strip()
        if broker is None or not candidate_id:
            return {"ok": False, "error": "candidate_id and attachment broker are required"}
        raw_access = params.get("access", "observe")
        tokens = (
            [str(item).strip().lower() for item in raw_access]
            if isinstance(raw_access, (list, tuple))
            else str(raw_access).replace("+", ",").split(",")
        )
        access = tuple(dict.fromkeys(AttachmentAccess(item.strip()) for item in tokens if item.strip()))
        if AttachmentAccess.CONTROL in access and AttachmentAccess.OBSERVE not in access:
            access = (AttachmentAccess.OBSERVE, *access)
        request = await broker.request_connection(
            candidate_id,
            requested_access=access or (AttachmentAccess.OBSERVE,),
            initiated_by="aura",
            reason=str(params.get("reason") or "I chose to request this physical capability")[:320],
        )
        return {
            "ok": True,
            "connection_request": request.to_dict(),
            "summary": (
                "The device was attached through existing trust."
                if request.state.value == "attached"
                else "I proposed the connection; trust has not been invented or assumed."
            ),
        }

    @staticmethod
    async def _authorize_connection(
        params: Mapping[str, Any],
        context: Mapping[str, Any],
    ) -> dict[str, Any]:
        broker = _service("reality_attachment_broker")
        request_id = str(params.get("request_id") or "").strip()
        if broker is None or not request_id:
            return {"ok": False, "error": "request_id and attachment broker are required"}
        try:
            persistent = _boolean_parameter(params.get("persistent"), default=True)
            raw_ttl = params.get("grant_ttl_s")
            if raw_ttl is None:
                grant_ttl_s = None
            elif isinstance(raw_ttl, bool):
                raise ValueError("grant_ttl_s must be an integer")
            elif isinstance(raw_ttl, int):
                grant_ttl_s = raw_ttl
            elif isinstance(raw_ttl, str) and raw_ttl.strip().isdigit():
                grant_ttl_s = int(raw_ttl.strip())
            else:
                raise ValueError("grant_ttl_s must be an integer")
            intent = broker.authority_intent(
                request_id,
                persistent=persistent,
                grant_ttl_s=grant_ttl_s,
            )
            decision_context = dict(context)
            decision_context.update(
                {
                    "physical_attachment_request_id": request_id,
                    "physical_attachment_scope": intent["scope"],
                    "persistent_physical_trust": persistent,
                    "verification_required": True,
                }
            )
            decision = get_will().decide(
                content=(
                    "Authorize this exact bounded physical attachment relationship: "
                    + canonical_json(intent).decode("utf-8")
                ),
                source="embodiment_skill",
                domain=ActionDomain.ENVIRONMENT_ACTION,
                priority=0.7 if "control" in intent["requested_access"] else 0.5,
                context=decision_context,
            )
            capability = get_capability_issuer().issue_from_decision(
                decision,
                action=ATTACHMENT_AUTHORITY_ACTION,
                payload=intent,
                scope=str(intent["scope"]),
            )
            migration_intent = broker.manifest_migration_intent(request_id)
            migration_capability = None
            if migration_intent is not None:
                migration_context = dict(decision_context)
                migration_context.update(
                    {
                        "physical_manifest_migration": True,
                        "physical_manifest_expected": migration_intent[
                            "expected_manifest_sha256"
                        ],
                        "physical_manifest_replacement": migration_intent[
                            "new_manifest_sha256"
                        ],
                    }
                )
                migration_decision = get_will().decide(
                    content=(
                        "Authorize this exact physical manifest compare-and-swap: "
                        + canonical_json(migration_intent).decode("utf-8")
                    ),
                    source="embodiment_skill",
                    domain=ActionDomain.ENVIRONMENT_ACTION,
                    priority=0.8,
                    context=migration_context,
                )
                migration_capability = get_capability_issuer().issue_from_decision(
                    migration_decision,
                    action=MANIFEST_MIGRATION_AUTHORITY_ACTION,
                    payload=migration_intent,
                    scope=str(migration_intent["scope"]),
                )
            attached = await broker.authorize_and_attach(
                request_id,
                authority_capability=capability.to_dict(),
                manifest_migration_capability=(
                    migration_capability.to_dict()
                    if migration_capability is not None
                    else None
                ),
                persistent=persistent,
                grant_ttl_s=grant_ttl_s,
            )
            return {
                "ok": attached.state.value == "attached",
                "connection_request": attached.to_dict(),
                "authority_receipt_id": attached.authority_receipt_id,
                "grant_ttl_s": int(intent["grant_ttl_s"]),
                "persistent": persistent,
                "manifest_migration_authorized": migration_capability is not None,
                "summary": (
                    "The declared physical relationship is attached under bounded trust."
                    if attached.state.value == "attached"
                    else "Authority was valid, but the physical attachment did not complete."
                ),
            }
        except (
            CapabilityViolation,
            LookupError,
            PermissionError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            return {
                "ok": False,
                "error": f"{type(exc).__name__}:{exc}"[:320],
                "request_id": request_id,
            }

    @staticmethod
    async def _query(params: Mapping[str, Any]) -> dict[str, Any]:
        channel_id = str(params.get("channel_id") or "").strip().lower()
        if channel_id:
            reality = _service("reality_reach")
            if reality is None:
                return {"ok": False, "error": "Reality Reach is offline"}
            reading = reality.reading(channel_id)
            declaration = next(
                (item for item in reality.declarations() if item.channel_id == channel_id),
                None,
            )
            if reading is None or declaration is None:
                return {"ok": False, "error": f"physical channel not found: {channel_id}"}
            return {
                "ok": True,
                "declaration": declaration.to_dict(),
                "reading": reading.to_dict(),
            }
        device_id = str(params.get("device_id") or "").strip()
        manager = _service("hardware_manager")
        device = manager.get_device(device_id) if manager is not None else None
        if device is None:
            return {"ok": False, "error": f"hardware device not found: {device_id}"}
        status = await device.get_status()
        return {"ok": bool(status.get("ok", True)), "device_id": device_id, "status": status}

    @staticmethod
    async def _command(params: Mapping[str, Any]) -> dict[str, Any]:
        channel_id = str(params.get("channel_id") or "").strip().lower()
        if channel_id:
            target = params.get("target_value", params.get("value"))
            if target is None or isinstance(target, bool):
                return {
                    "ok": False,
                    "error": "target_value must be numeric for a Reality Reach channel",
                }
            try:
                target_value = float(target)
                deadline_s = _bounded_float_parameter(
                    params.get("timeout_s"),
                    name="timeout_s",
                    default=30.0,
                    minimum=0.1,
                    maximum=300.0,
                )
                reason = str(
                    params.get("reason") or "Aura selected a physical target"
                )[:240]
                result = await get_world_bridge().call(
                    Channel.ENVIRONMENTAL_CHANGE,
                    action=f"physical:{channel_id}:set_target",
                    intent=reason,
                    payload={
                        "operation": "reality_target",
                        "channel_id": channel_id,
                        "target_value": target_value,
                        "timeout_s": deadline_s,
                        "idempotency_key": str(
                            params.get("idempotency_key")
                            or f"embodiment.target.{uuid.uuid4().hex}"
                        ),
                        "reason": reason,
                        "source": "embodiment_skill",
                    },
                )
            except (
                LookupError,
                OSError,
                RuntimeError,
                TimeoutError,
                TypeError,
                ValueError,
            ) as exc:
                return {
                    "ok": False,
                    "error": f"{type(exc).__name__}:{exc}"[:320],
                    "channel_id": channel_id,
                }
            output = _world_result(result)
            return {
                **output,
                "channel_id": channel_id,
                "target_value": target_value,
            }
        device_id = str(params.get("device_id") or params.get("target") or "").strip().lower()
        command = str(params.get("command") or params.get("op") or "").strip().lower()
        raw_parameters = params.get("parameters", params.get("effect", {}))
        if not device_id or not command or not isinstance(raw_parameters, Mapping):
            return {
                "ok": False,
                "error": "device_id, command, and a parameter mapping are required",
            }
        requested_transport = str(params.get("transport") or "").strip().lower()
        domain = device_id.partition(".")[0]
        use_home_assistant = (
            requested_transport == "home_assistant"
            or domain in _HOME_ASSISTANT_CONTROL_DOMAINS
        )
        operation = "home_assistant_apply" if use_home_assistant else "hardware_apply"
        payload: dict[str, Any] = {
            "operation": operation,
            "target": device_id,
            "op": command,
            "parameters": dict(raw_parameters),
            "reason": str(params.get("reason") or "Aura selected a physical action")[:240],
            "idempotency_key": str(
                params.get("idempotency_key")
                or f"embodiment.{uuid.uuid4().hex}"
            ),
        }
        if operation == "home_assistant_apply":
            payload["transport"] = "home_assistant"
            payload["effect"] = payload.pop("parameters")
        result = await get_world_bridge().call(
            Channel.ENVIRONMENTAL_CHANGE,
            action=f"physical:{device_id}:{command}",
            intent=payload["reason"],
            payload=payload,
        )
        return _world_result(result)


__all__ = ["EmbodimentSkill"]
