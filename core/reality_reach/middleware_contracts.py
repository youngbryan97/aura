"""Shared validation and durable receipt contracts for Reality Reach middleware."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from core.bus.qos import History, QosProfile
from core.reality_reach.live import ChannelReading
from core.runtime.audit_chain import canonical_json, sha256_hex

_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9_.:-]{0,127}$")
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")


class RealityMiddlewareError(RuntimeError):
    """A typed endpoint, lifecycle, deadline, or recovery contract failed."""


class PhysicalEffectIndeterminateError(RealityMiddlewareError):
    """A command may have reached hardware but its final effect is unproven."""


def canonical_identifier(value: str, *, name: str) -> str:
    normalized = str(value or "").strip().lower()
    if not _IDENTIFIER.fullmatch(normalized):
        raise ValueError(f"{name} must be a canonical identifier")
    return normalized


def sha256_digest(value: str, *, name: str) -> str:
    normalized = str(value or "").strip().lower()
    if not _DIGEST.fullmatch(normalized):
        raise ValueError(f"{name} must be a sha256 digest")
    return normalized


def bounded_payload(
    value: Mapping[str, Any],
    *,
    name: str,
    maximum: int,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    snapshot = dict(value)
    encoded = canonical_json(snapshot)
    if len(encoded) > maximum:
        raise ValueError(f"{name} exceeds {maximum} bytes")
    return snapshot


_identifier = canonical_identifier
_digest = sha256_digest
_bounded_payload = bounded_payload

TERMINAL_ACTION_STATES = frozenset(
    {
        "succeeded",
        "aborted",
        "cancelled",
        "preempted",
        "timed_out",
        "interrupted",
        "indeterminate",
    }
)
INFLIGHT_ACTION_STATES = frozenset(
    {"accepted", "executing", "cancel_requested", "preempting"}
)


class TelemetryMode(StrEnum):
    PUSH = "push"
    PULL = "pull"


class RestartPolicy(StrEnum):
    """How an interrupted action is treated after process restart."""

    RECONCILE = "reconcile"
    ABORT = "abort"


class ActionState(StrEnum):
    ACCEPTED = "accepted"
    EXECUTING = "executing"
    CANCEL_REQUESTED = "cancel_requested"
    PREEMPTING = "preempting"
    SUCCEEDED = "succeeded"
    ABORTED = "aborted"
    CANCELLED = "cancelled"
    PREEMPTED = "preempted"
    TIMED_OUT = "timed_out"
    INTERRUPTED = "interrupted"
    INDETERMINATE = "indeterminate"

    @property
    def terminal(self) -> bool:
        return self.value in TERMINAL_ACTION_STATES


def bounded_seconds(value: float, *, name: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be numeric")
    number = float(value)
    if not minimum <= number <= maximum:
        raise ValueError(f"{name} must lie inside [{minimum}, {maximum}]")
    return number


def _validate_qos(profile: QosProfile) -> None:
    if not isinstance(profile, QosProfile):
        raise TypeError("qos must be a QosProfile")
    if isinstance(profile.depth, bool) or not 1 <= int(profile.depth) <= 4096:
        raise ValueError("qos depth must lie inside [1, 4096]")
    for name, value in (
        ("lifespan_s", profile.lifespan_s),
        ("deadline_s", profile.deadline_s),
        ("liveliness_lease_s", profile.liveliness_lease_s),
    ):
        if isinstance(value, bool) or not 0.0 <= float(value) <= 86400.0:
            raise ValueError(f"qos {name} must lie inside [0, 86400]")
    if profile.history is History.KEEP_ALL and profile.depth > 4096:
        raise ValueError("KEEP_ALL still requires a bounded depth")


@dataclass(frozen=True, slots=True)
class TelemetryEndpoint:
    endpoint_id: str
    channel_ids: tuple[str, ...]
    qos: QosProfile
    mode: TelemetryMode = TelemetryMode.PUSH
    sample_period_s: float = 1.0
    sample_timeout_s: float = 0.8
    payload_bytes: int = 65536

    def __post_init__(self) -> None:
        object.__setattr__(self, "endpoint_id", _identifier(self.endpoint_id, name="endpoint_id"))
        channel_ids = tuple(_identifier(item, name="channel_id") for item in self.channel_ids)
        if not channel_ids or len(channel_ids) != len(set(channel_ids)):
            raise ValueError("telemetry channel_ids must be non-empty and unique")
        object.__setattr__(self, "channel_ids", channel_ids)
        _validate_qos(self.qos)
        if not isinstance(self.mode, TelemetryMode):
            raise TypeError("mode must be a TelemetryMode")
        period = bounded_seconds(
            self.sample_period_s,
            name="sample_period_s",
            minimum=0.01,
            maximum=3600.0,
        )
        timeout = bounded_seconds(
            self.sample_timeout_s,
            name="sample_timeout_s",
            minimum=0.01,
            maximum=300.0,
        )
        if self.mode is TelemetryMode.PULL and timeout > period:
            raise ValueError("pull sample_timeout_s must not exceed sample_period_s")
        object.__setattr__(self, "sample_period_s", period)
        object.__setattr__(self, "sample_timeout_s", timeout)
        if isinstance(self.payload_bytes, bool) or not 256 <= int(self.payload_bytes) <= 1_048_576:
            raise ValueError("payload_bytes must lie inside [256, 1048576]")

    @property
    def topic(self) -> str:
        return f"reality.telemetry.{self.endpoint_id}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "endpoint_id": self.endpoint_id,
            "channel_ids": list(self.channel_ids),
            "qos": self.qos.to_dict(),
            "mode": self.mode.value,
            "sample_period_s": self.sample_period_s,
            "sample_timeout_s": self.sample_timeout_s,
            "payload_bytes": self.payload_bytes,
        }


@dataclass(frozen=True, slots=True)
class ServiceEndpoint:
    endpoint_id: str
    timeout_s: float = 10.0
    max_inflight: int = 1
    request_bytes: int = 65536
    response_bytes: int = 65536
    read_only: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "endpoint_id", _identifier(self.endpoint_id, name="endpoint_id"))
        object.__setattr__(
            self,
            "timeout_s",
            bounded_seconds(self.timeout_s, name="timeout_s", minimum=0.01, maximum=300.0),
        )
        if isinstance(self.max_inflight, bool) or not 1 <= int(self.max_inflight) <= 64:
            raise ValueError("max_inflight must lie inside [1, 64]")
        for name in ("request_bytes", "response_bytes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not 256 <= int(value) <= 1_048_576:
                raise ValueError(f"{name} must lie inside [256, 1048576]")

    def to_dict(self) -> dict[str, Any]:
        return {
            "endpoint_id": self.endpoint_id,
            "timeout_s": self.timeout_s,
            "max_inflight": self.max_inflight,
            "request_bytes": self.request_bytes,
            "response_bytes": self.response_bytes,
            "read_only": self.read_only,
        }


@dataclass(frozen=True, slots=True)
class ActionEndpoint:
    endpoint_id: str
    timeout_s: float = 300.0
    cancel_timeout_s: float = 10.0
    feedback_depth: int = 64
    request_bytes: int = 65536
    result_bytes: int = 65536
    preemptible: bool = True
    restart_policy: RestartPolicy = RestartPolicy.RECONCILE
    requires_effect_verification: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "endpoint_id", _identifier(self.endpoint_id, name="endpoint_id"))
        object.__setattr__(
            self,
            "timeout_s",
            bounded_seconds(self.timeout_s, name="timeout_s", minimum=0.1, maximum=86400.0),
        )
        object.__setattr__(
            self,
            "cancel_timeout_s",
            bounded_seconds(
                self.cancel_timeout_s,
                name="cancel_timeout_s",
                minimum=0.05,
                maximum=300.0,
            ),
        )
        if isinstance(self.feedback_depth, bool) or not 1 <= int(self.feedback_depth) <= 1024:
            raise ValueError("feedback_depth must lie inside [1, 1024]")
        for name in ("request_bytes", "result_bytes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not 256 <= int(value) <= 1_048_576:
                raise ValueError(f"{name} must lie inside [256, 1048576]")
        if not isinstance(self.restart_policy, RestartPolicy):
            raise TypeError("restart_policy must be a RestartPolicy")

    def to_dict(self) -> dict[str, Any]:
        return {
            "endpoint_id": self.endpoint_id,
            "timeout_s": self.timeout_s,
            "cancel_timeout_s": self.cancel_timeout_s,
            "feedback_depth": self.feedback_depth,
            "request_bytes": self.request_bytes,
            "result_bytes": self.result_bytes,
            "preemptible": self.preemptible,
            "restart_policy": self.restart_policy.value,
            "requires_effect_verification": self.requires_effect_verification,
        }


@dataclass(frozen=True, slots=True)
class ManagedAdapterDeclaration:
    node_id: str
    adapter_id: str
    adapter_identity_sha256: str
    telemetry: tuple[TelemetryEndpoint, ...] = ()
    services: tuple[ServiceEndpoint, ...] = ()
    actions: tuple[ActionEndpoint, ...] = ()
    transition_timeout_s: float = 30.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "node_id", _identifier(self.node_id, name="node_id"))
        object.__setattr__(self, "adapter_id", _identifier(self.adapter_id, name="adapter_id"))
        object.__setattr__(
            self,
            "adapter_identity_sha256",
            _digest(self.adapter_identity_sha256, name="adapter_identity_sha256"),
        )
        object.__setattr__(
            self,
            "transition_timeout_s",
            bounded_seconds(
                self.transition_timeout_s,
                name="transition_timeout_s",
                minimum=0.1,
                maximum=300.0,
            ),
        )
        endpoints = [
            *(item.endpoint_id for item in self.telemetry),
            *(item.endpoint_id for item in self.services),
            *(item.endpoint_id for item in self.actions),
        ]
        if not endpoints:
            raise ValueError("a managed adapter must declare at least one endpoint")
        if len(endpoints) != len(set(endpoints)):
            raise ValueError("endpoint ids must be unique within a managed node")

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "adapter_id": self.adapter_id,
            "adapter_identity_sha256": self.adapter_identity_sha256,
            "telemetry": [item.to_dict() for item in self.telemetry],
            "services": [item.to_dict() for item in self.services],
            "actions": [item.to_dict() for item in self.actions],
            "transition_timeout_s": self.transition_timeout_s,
        }

    @property
    def sha256(self) -> str:
        return str(sha256_hex(canonical_json(self.to_dict())))


@dataclass(slots=True)
class ActionRecord:
    goal_id: str
    endpoint_id: str
    node_id: str
    adapter_id: str
    adapter_identity_sha256: str
    request: dict[str, Any]
    request_sha256: str
    state: ActionState
    created_at_ns: int
    updated_at_ns: int
    deadline_at_ns: int
    result: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    feedback: list[dict[str, Any]] = field(default_factory=list)
    recovery_required: bool = False

    def to_dict(self) -> dict[str, Any]:
        body = {
            "goal_id": self.goal_id,
            "endpoint_id": self.endpoint_id,
            "node_id": self.node_id,
            "adapter_id": self.adapter_id,
            "adapter_identity_sha256": self.adapter_identity_sha256,
            "request": dict(self.request),
            "request_sha256": self.request_sha256,
            "state": self.state.value,
            "created_at_ns": self.created_at_ns,
            "updated_at_ns": self.updated_at_ns,
            "deadline_at_ns": self.deadline_at_ns,
            "result": dict(self.result),
            "error": self.error,
            "feedback": [dict(item) for item in self.feedback],
            "recovery_required": self.recovery_required,
        }
        return {**body, "record_sha256": str(sha256_hex(canonical_json(body)))}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ActionRecord:
        body = dict(value)
        recorded = str(body.pop("record_sha256", ""))
        if recorded != str(sha256_hex(canonical_json(body))):
            raise RealityMiddlewareError("action record integrity check failed")
        state = ActionState(str(body["state"]))
        request = _bounded_payload(body.get("request") or {}, name="request", maximum=1_048_576)
        if str(sha256_hex(canonical_json(request))) != str(body["request_sha256"]):
            raise RealityMiddlewareError("action request digest differs")
        return cls(
            goal_id=_identifier(str(body["goal_id"]), name="goal_id"),
            endpoint_id=_identifier(str(body["endpoint_id"]), name="endpoint_id"),
            node_id=_identifier(str(body["node_id"]), name="node_id"),
            adapter_id=_identifier(str(body["adapter_id"]), name="adapter_id"),
            adapter_identity_sha256=_digest(
                str(body["adapter_identity_sha256"]),
                name="adapter_identity_sha256",
            ),
            request=request,
            request_sha256=str(body["request_sha256"]),
            state=state,
            created_at_ns=int(body["created_at_ns"]),
            updated_at_ns=int(body["updated_at_ns"]),
            deadline_at_ns=int(body["deadline_at_ns"]),
            result=dict(body.get("result") or {}),
            error=str(body.get("error") or "")[:1024],
            feedback=[dict(item) for item in list(body.get("feedback") or [])[-1024:]],
            recovery_required=bool(body.get("recovery_required", False)),
        )


class ActionContext:
    """Adapter-facing cancellation and monotonic feedback surface."""

    def __init__(
        self,
        goal_id: str,
        cancel_event: asyncio.Event,
        feedback: Callable[[float, Mapping[str, Any]], Awaitable[None]],
    ) -> None:
        self.goal_id = goal_id
        self._cancel_event = cancel_event
        self._feedback = feedback

    @property
    def cancel_requested(self) -> bool:
        return self._cancel_event.is_set()

    async def wait_cancelled(self, *, timeout_s: float = 60.0) -> bool:
        if isinstance(timeout_s, bool):
            raise TypeError("timeout_s must be numeric")
        try:
            timeout = float(timeout_s)
        except (TypeError, ValueError) as exc:
            raise TypeError("timeout_s must be numeric") from exc
        if not 0.001 <= timeout <= 3600.0:
            raise ValueError("timeout_s must lie inside [0.001, 3600]")
        try:
            await asyncio.wait_for(self._cancel_event.wait(), timeout=timeout)
        # not a failure: the wait ran out, which is what the timeout was set to decide.
        except TimeoutError:
            return False
        return True

    async def publish_feedback(self, progress: float, payload: Mapping[str, Any]) -> None:
        await self._feedback(progress, payload)


@runtime_checkable
class ManagedRealityAdapter(Protocol):
    @property
    def adapter_id(self) -> str: ...

    def lifecycle_declaration(self) -> ManagedAdapterDeclaration: ...

    async def on_configure(self) -> bool: ...

    async def on_activate(self) -> bool: ...

    async def on_deactivate(self) -> bool: ...

    async def on_cleanup(self) -> bool: ...

    async def on_shutdown(self) -> bool: ...

    async def on_error(self) -> bool: ...

    async def read_telemetry(
        self, endpoint_id: str
    ) -> Mapping[str, Any] | tuple[ChannelReading, ...]: ...

    async def handle_service(
        self, endpoint_id: str, request: Mapping[str, Any]
    ) -> Mapping[str, Any]: ...

    async def execute_action(
        self,
        endpoint_id: str,
        request: Mapping[str, Any],
        context: ActionContext,
    ) -> Mapping[str, Any]: ...

    async def cancel_action(self, endpoint_id: str, goal_id: str, reason: str) -> bool: ...

    async def reconcile_action(
        self, endpoint_id: str, record: Mapping[str, Any]
    ) -> Mapping[str, Any]: ...


@dataclass(frozen=True, slots=True)
class ServiceReceipt:
    """Integrity-protected result of one idempotent bounded service request."""

    request_id: str
    endpoint_id: str
    request_sha256: str
    ok: bool
    response: dict[str, Any]
    error: str
    started_at_ns: int
    completed_at_ns: int
    adapter_identity_sha256: str

    def to_dict(self) -> dict[str, Any]:
        body = {
            "request_id": self.request_id,
            "endpoint_id": self.endpoint_id,
            "request_sha256": self.request_sha256,
            "ok": self.ok,
            "response": dict(self.response),
            "error": self.error,
            "started_at_ns": self.started_at_ns,
            "completed_at_ns": self.completed_at_ns,
            "adapter_identity_sha256": self.adapter_identity_sha256,
        }
        return {**body, "receipt_sha256": str(sha256_hex(canonical_json(body)))}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ServiceReceipt:
        body = dict(value)
        recorded = str(body.pop("receipt_sha256", ""))
        if recorded != str(sha256_hex(canonical_json(body))):
            raise RealityMiddlewareError("service receipt integrity check failed")
        return cls(
            request_id=canonical_identifier(str(body["request_id"]), name="request_id"),
            endpoint_id=canonical_identifier(
                str(body["endpoint_id"]),
                name="endpoint_id",
            ),
            request_sha256=str(body["request_sha256"]),
            ok=bool(body["ok"]),
            response=bounded_payload(
                body.get("response") or {},
                name="service receipt response",
                maximum=1_048_576,
            ),
            error=str(body.get("error") or "")[:1024],
            started_at_ns=int(body["started_at_ns"]),
            completed_at_ns=int(body["completed_at_ns"]),
            adapter_identity_sha256=sha256_digest(
                str(body["adapter_identity_sha256"]),
                name="adapter_identity_sha256",
            ),
        )


__all__ = [
    "ActionContext",
    "ActionEndpoint",
    "ActionRecord",
    "ActionState",
    "INFLIGHT_ACTION_STATES",
    "ManagedAdapterDeclaration",
    "ManagedRealityAdapter",
    "PhysicalEffectIndeterminateError",
    "RealityMiddlewareError",
    "RestartPolicy",
    "ServiceEndpoint",
    "ServiceReceipt",
    "TERMINAL_ACTION_STATES",
    "TelemetryEndpoint",
    "TelemetryMode",
    "bounded_payload",
    "bounded_seconds",
    "canonical_identifier",
    "sha256_digest",
]
