"""Manifest-bound OPC UA sensing and verified scalar actuation.

The connector deliberately exposes scalar variables and idempotent setpoints,
not arbitrary OPC UA methods.  A writable resource must name a command node
and a distinct state node so transport acceptance cannot masquerade as effect
verification.  Session credentials remain in the process environment and are
never copied into candidates, manifests, samples, or receipts.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
import time
import urllib.parse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from core.reality_reach.attachments import AttachmentAccess, DeviceCandidate
from core.reality_reach.contracts import NumericDomain
from core.reality_reach.live import LiveChannelAdapter
from core.reality_reach.scalar_adapter import (
    ScalarRealityAdapter,
    ScalarResourceProfile,
    ScalarSample,
    ScalarWriteResult,
)
from core.runtime.audit_chain import canonical_json, sha256_hex
from core.runtime.lockdep import checked_async_lock
from core.runtime.flags import env_str

_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9_.:-]{0,127}$")
_VALUE_TYPES = frozenset(
    {
        "boolean",
        "byte",
        "double",
        "float",
        "int16",
        "int32",
        "int64",
        "sbyte",
        "uint16",
        "uint32",
        "uint64",
    }
)
_INTEGER_TYPES = _VALUE_TYPES - {"boolean", "double", "float"}
_VARIANT_TYPE_NAMES = {
    "boolean": "Boolean",
    "byte": "Byte",
    "double": "Double",
    "float": "Float",
    "int16": "Int16",
    "int32": "Int32",
    "int64": "Int64",
    "sbyte": "SByte",
    "uint16": "UInt16",
    "uint32": "UInt32",
    "uint64": "UInt64",
}
_MAX_IDEMPOTENCY_RECEIPTS = 4096


class OPCUAConnectorError(RuntimeError):
    """An OPC UA configuration, session, or node violated its contract."""


def _digest(value: Any) -> str:
    return str(sha256_hex(canonical_json(value)))


def _identifier(value: object, *, name: str) -> str:
    normalized = str(value or "").strip().lower()
    if not _IDENTIFIER.fullmatch(normalized):
        raise ValueError(f"{name} must be a canonical identifier")
    return normalized


def _node_id(value: object, *, name: str) -> str:
    normalized = str(value or "").strip()
    if not normalized or "\x00" in normalized or len(normalized.encode("utf-8")) > 512:
        raise ValueError(f"{name} must be a bounded OPC UA node id")
    if not re.fullmatch(r"(?:(?:ns|nsu)=[^;]+;)?[isgb]=.+", normalized):
        raise ValueError(f"{name} must use canonical OPC UA NodeId syntax")
    return normalized


def _finite(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be numeric")
    number = float(value)  # type: ignore[arg-type]
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _allow_insecure() -> bool:
    from core.runtime.flags import FlagKind, declare

    return str(
        declare(
            "AURA_OPCUA_ALLOW_INSECURE",
            kind=FlagKind.STRING,
            default="",
            description="Permit an OPC UA session without message security",
            owner="core.embodiment.opcua_connector",
        ).value()
    ).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True, slots=True)
class OPCUAResourceSpec:
    resource_id: str
    device_id: str
    observable: str
    unit: str
    state_node_id: str
    domain: NumericDomain
    resolution: float
    command_node_id: str = ""
    safe_value: float | None = None
    tolerance: float | None = None
    value_type: str = "double"
    max_commands_per_minute: int = 12
    cooldown_s: float = 0.0
    stale_after_s: float = 30.0

    def __post_init__(self) -> None:
        for name in ("resource_id", "device_id", "observable", "unit"):
            object.__setattr__(self, name, _identifier(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "state_node_id",
            _node_id(self.state_node_id, name="state_node_id"),
        )
        command_node = str(self.command_node_id or "").strip()
        if command_node:
            command_node = _node_id(command_node, name="command_node_id")
            if command_node == self.state_node_id:
                raise ValueError(
                    "writable OPC UA resources require a distinct state node"
                )
        object.__setattr__(self, "command_node_id", command_node)
        if not isinstance(self.domain, NumericDomain):
            raise TypeError("domain must be NumericDomain")
        resolution = _finite(self.resolution, name="resolution")
        if resolution <= 0.0:
            raise ValueError("resolution must be positive")
        object.__setattr__(self, "resolution", resolution)
        tolerance = resolution if self.tolerance is None else _finite(
            self.tolerance,
            name="tolerance",
        )
        if tolerance < resolution:
            raise ValueError("tolerance must not be smaller than resolution")
        object.__setattr__(self, "tolerance", tolerance)
        if self.safe_value is not None:
            safe = _finite(self.safe_value, name="safe_value")
            if not self.domain.contains(safe):
                raise ValueError("safe_value lies outside the domain")
            object.__setattr__(self, "safe_value", safe)
        value_type = str(self.value_type or "double").strip().lower()
        if value_type not in _VALUE_TYPES:
            raise ValueError("value_type is not a supported scalar OPC UA type")
        object.__setattr__(self, "value_type", value_type)
        if not 1 <= int(self.max_commands_per_minute) <= 600:
            raise ValueError("max_commands_per_minute must lie inside [1, 600]")
        cooldown = _finite(self.cooldown_s, name="cooldown_s")
        stale = _finite(self.stale_after_s, name="stale_after_s")
        if cooldown < 0.0 or not 0.1 <= stale <= 86_400.0:
            raise ValueError("OPC UA timing bounds are invalid")
        object.__setattr__(self, "cooldown_s", cooldown)
        object.__setattr__(self, "stale_after_s", stale)

    @property
    def writable(self) -> bool:
        return bool(self.command_node_id)

    @property
    def sha256(self) -> str:
        return _digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "resource_id": self.resource_id,
            "device_id": self.device_id,
            "observable": self.observable,
            "unit": self.unit,
            "state_node_id": self.state_node_id,
            "domain": self.domain.to_dict(),
            "resolution": self.resolution,
            "command_node_id": self.command_node_id,
            "safe_value": self.safe_value,
            "tolerance": self.tolerance,
            "value_type": self.value_type,
            "max_commands_per_minute": self.max_commands_per_minute,
            "cooldown_s": self.cooldown_s,
            "stale_after_s": self.stale_after_s,
        }

    def decode(self, value: object) -> float:
        if self.value_type == "boolean":
            if not isinstance(value, bool):
                raise OPCUAConnectorError("opcua_boolean_state_type_mismatch")
            number = 1.0 if value else 0.0
        elif isinstance(value, bool):
            raise OPCUAConnectorError("opcua_numeric_state_type_mismatch")
        else:
            number = _finite(value, name="OPC UA state")
        if not self.domain.contains(number):
            raise OPCUAConnectorError("opcua_state_outside_manifest_domain")
        return number

    def encode(self, value: float) -> bool | float | int:
        number = _finite(value, name="OPC UA command")
        if not self.domain.contains(number):
            raise OPCUAConnectorError("opcua_command_outside_manifest_domain")
        if self.value_type == "boolean":
            if number not in {0.0, 1.0}:
                raise OPCUAConnectorError("opcua_boolean_command_requires_zero_or_one")
            return bool(number)
        if self.value_type in _INTEGER_TYPES:
            integer = int(number)
            if float(integer) != number:
                raise OPCUAConnectorError("opcua_integer_command_requires_integer_value")
            return integer
        return number


def parse_opcua_resource_manifest(raw: object) -> tuple[OPCUAResourceSpec, ...]:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise OPCUAConnectorError("opcua_resource_manifest_invalid_json") from exc
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise OPCUAConnectorError("opcua_resource_manifest_must_be_a_list")
    if not 1 <= len(raw) <= 512:
        raise OPCUAConnectorError("opcua_resource_manifest_size_invalid")
    resources: list[OPCUAResourceSpec] = []
    for item in raw:
        if not isinstance(item, Mapping):
            raise OPCUAConnectorError("opcua_resource_manifest_entry_invalid")
        resources.append(
            OPCUAResourceSpec(
                resource_id=str(item.get("resource_id") or ""),
                device_id=str(item.get("device_id") or ""),
                observable=str(item.get("observable") or ""),
                unit=str(item.get("unit") or ""),
                state_node_id=str(item.get("state_node_id") or ""),
                domain=NumericDomain(
                    _finite(item.get("minimum"), name="minimum"),
                    _finite(item.get("maximum"), name="maximum"),
                ),
                resolution=_finite(item.get("resolution"), name="resolution"),
                command_node_id=str(item.get("command_node_id") or ""),
                safe_value=(
                    None
                    if item.get("safe_value") is None
                    else _finite(item.get("safe_value"), name="safe_value")
                ),
                tolerance=(
                    None
                    if item.get("tolerance") is None
                    else _finite(item.get("tolerance"), name="tolerance")
                ),
                value_type=str(item.get("value_type") or "double"),
                max_commands_per_minute=int(item.get("max_commands_per_minute") or 12),
                cooldown_s=_finite(item.get("cooldown_s") or 0.0, name="cooldown_s"),
                stale_after_s=_finite(
                    item.get("stale_after_s") or 30.0,
                    name="stale_after_s",
                ),
            )
        )
    if len({item.resource_id for item in resources}) != len(resources):
        raise OPCUAConnectorError("opcua_resource_id_duplicate")
    state_nodes = [item.state_node_id for item in resources]
    if len(set(state_nodes)) != len(state_nodes):
        raise OPCUAConnectorError("opcua_state_node_duplicate")
    command_nodes = [item.command_node_id for item in resources if item.command_node_id]
    if len(set(command_nodes)) != len(command_nodes):
        raise OPCUAConnectorError("opcua_command_node_duplicate")
    if set(state_nodes) & set(command_nodes):
        raise OPCUAConnectorError("opcua_command_and_state_nodes_must_be_distinct")
    return tuple(sorted(resources, key=lambda item: item.resource_id))


@runtime_checkable
class OPCUAScalarTransport(Protocol):
    @property
    def transport_id(self) -> str: ...

    @property
    def server_identity_sha256(self) -> str: ...

    async def read_scalar(self, resource_id: str) -> ScalarSample: ...

    async def write_scalar(
        self,
        resource_id: str,
        value: float,
        *,
        idempotency_key: str,
        recovery: bool = False,
    ) -> ScalarWriteResult: ...


class AsyncUaScalarTransport:
    """Secure, reconnecting asyncua transport with bounded idempotency memory."""

    transport_id = "opcua.asyncua"

    def __init__(self, resources: tuple[OPCUAResourceSpec, ...]) -> None:
        if not resources:
            raise ValueError("resources must not be empty")
        self._resources = {item.resource_id: item for item in resources}
        self._endpoint = str(env_str("AURA_OPCUA_ENDPOINT", description="OPC UA endpoint", owner="core.embodiment.opcua") or "").strip()
        self._installation_id = str(
            env_str("AURA_OPCUA_INSTALLATION_ID", description="OPC UA installation id", owner="core.embodiment.opcua") or ""
        ).strip()
        if not self._endpoint or not self._installation_id:
            raise OPCUAConnectorError("opcua_endpoint_and_installation_id_required")
        parsed = urllib.parse.urlparse(self._endpoint)
        if (
            parsed.scheme != "opc.tcp"
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise OPCUAConnectorError("opcua_endpoint_invalid")
        self._security_policy = str(
            env_str("AURA_OPCUA_SECURITY_POLICY", description="OPC UA security policy", owner="core.embodiment.opcua_connector") or "Basic256Sha256"
        ).strip()
        self._security_mode = str(
            env_str("AURA_OPCUA_SECURITY_MODE", description="OPC UA security mode", owner="core.embodiment.opcua_connector") or "SignAndEncrypt"
        ).strip()
        self._certificate = str(env_str("AURA_OPCUA_CERTIFICATE", description="OPC UA certificate", owner="core.embodiment.opcua_connector") or "").strip()
        self._private_key = str(env_str("AURA_OPCUA_PRIVATE_KEY", description="OPC UA private key", owner="core.embodiment.opcua_connector") or "").strip()
        self._server_certificate = str(
            env_str("AURA_OPCUA_SERVER_CERTIFICATE", description="OPC UA server certificate", owner="core.embodiment.opcua_connector") or ""
        ).strip()
        self._private_key_password = str(
            env_str("AURA_OPCUA_PRIVATE_KEY_PASSWORD", description="OPC UA private key password", owner="core.embodiment.opcua_connector") or ""
        )
        secure = self._security_policy != "None"
        if secure:
            if self._security_policy not in {
                "Aes128Sha256RsaOaep",
                "Aes256Sha256RsaPss",
                "Basic256Sha256",
            }:
                raise OPCUAConnectorError("opcua_security_policy_invalid")
            if self._security_mode not in {"Sign", "SignAndEncrypt"}:
                raise OPCUAConnectorError("opcua_security_mode_invalid")
            for path in (
                self._certificate,
                self._private_key,
                self._server_certificate,
            ):
                if not path or not Path(path).expanduser().is_file():
                    raise OPCUAConnectorError("opcua_security_material_missing")
        elif self._security_mode != "None" or not _allow_insecure():
            raise OPCUAConnectorError("opcua_insecure_session_requires_explicit_opt_in")
        self._certificate_path = (
            str(Path(self._certificate).expanduser()) if self._certificate else ""
        )
        self._private_key_path = (
            str(Path(self._private_key).expanduser()) if self._private_key else ""
        )
        self._server_certificate_path = (
            str(Path(self._server_certificate).expanduser())
            if self._server_certificate
            else ""
        )
        username = str(env_str("AURA_OPCUA_USERNAME", description="OPC UA username", owner="core.embodiment.opcua_connector") or "").strip()
        password = str(env_str("AURA_OPCUA_PASSWORD", description="OPC UA password", owner="core.embodiment.opcua_connector") or "")
        if bool(username) != bool(password):
            raise OPCUAConnectorError("opcua_username_and_password_must_be_paired")
        self._username = username
        self._password = password
        timeout_s = _finite(
            env_str("AURA_OPCUA_TIMEOUT_S", description="OPC UA timeout s", owner="core.embodiment.opcua_connector") or 8.0,
            name="OPC UA timeout",
        )
        if not 1.0 <= timeout_s <= 30.0:
            raise OPCUAConnectorError("opcua_timeout_must_lie_inside_1_to_30_seconds")
        self._timeout_s = timeout_s
        self._server_identity = _digest(
            {
                "endpoint": self._endpoint,
                "installation_id": self._installation_id,
                "security_policy": self._security_policy,
                "server_certificate_sha256": (
                    sha256_hex(Path(self._server_certificate_path).read_bytes())
                    if self._server_certificate_path
                    else "insecure-explicit"
                ),
            }
        )
        self._client: Any | None = None
        self._connected = False
        self._lifecycle_lock = checked_async_lock("opcua_transport.lifecycle")
        self._write_lock = checked_async_lock("opcua_transport.write")
        self._write_receipts: dict[str, tuple[str, float, ScalarWriteResult]] = {}

    @property
    def server_identity_sha256(self) -> str:
        return self._server_identity

    async def _ensure_connected(self) -> Any:
        async with self._lifecycle_lock:
            if self._connected and self._client is not None:
                return self._client
            try:
                from asyncua import Client, ua
                from asyncua.crypto import security_policies
            except ImportError as exc:
                raise OPCUAConnectorError(
                    "opcua_transport_dependency_missing:asyncua"
                ) from exc
            client = Client(
                self._endpoint,
                timeout=self._timeout_s,
                auto_reconnect=True,
                reconnect_max_delay=min(30.0, self._timeout_s),
                reconnect_request_timeout=self._timeout_s,
            )
            if self._security_policy != "None":
                policies = {
                    "Aes128Sha256RsaOaep": (
                        security_policies.SecurityPolicyAes128Sha256RsaOaep
                    ),
                    "Aes256Sha256RsaPss": (
                        security_policies.SecurityPolicyAes256Sha256RsaPss
                    ),
                    "Basic256Sha256": security_policies.SecurityPolicyBasic256Sha256,
                }
                mode = getattr(ua.MessageSecurityMode, self._security_mode)
                await client.set_security(
                    policies[self._security_policy],
                    self._certificate_path,
                    self._private_key_path,
                    private_key_password=self._private_key_password or None,
                    server_certificate=self._server_certificate_path,
                    mode=mode,
                )
            if self._username:
                client.set_user(self._username)
                client.set_password(self._password)
            await client.connect(auto_reconnect=True)
            self._client = client
            self._connected = True
            return client

    async def _invalidate(self, client: Any) -> None:
        async with self._lifecycle_lock:
            if client is not self._client:
                return
            self._client = None
            self._connected = False
        try:
            await client.disconnect()
        except (OSError, RuntimeError, TimeoutError):
            return

    async def _data_value(self, node_id: str) -> Any:
        last_error: BaseException | None = None
        for _attempt in range(2):
            client = await self._ensure_connected()
            try:
                return await client.get_node(node_id).read_data_value()
            except (OSError, RuntimeError, TimeoutError) as exc:
                last_error = exc
                await self._invalidate(client)
        raise OPCUAConnectorError("opcua_read_failed_after_reconnect") from last_error

    async def read_scalar(self, resource_id: str) -> ScalarSample:
        spec = self._resources.get(resource_id)
        if spec is None:
            raise LookupError("opcua_resource_not_bound")
        data = await self._data_value(spec.state_node_id)
        status = getattr(data, "StatusCode", None)
        if status is None or not bool(status.is_good()):
            raise OPCUAConnectorError("opcua_state_status_not_good")
        variant = getattr(data, "Value", None)
        raw_value = getattr(variant, "Value", None)
        value = spec.decode(raw_value)
        source_time = getattr(data, "SourceTimestamp", None)
        if not isinstance(source_time, datetime):
            source_time = getattr(data, "ServerTimestamp", None)
        if isinstance(source_time, datetime):
            captured_at_ns = max(1, int(source_time.timestamp() * 1_000_000_000))
            wall_clock_source = "opcua.source_timestamp"
        else:
            captured_at_ns = max(1, time.time_ns())
            wall_clock_source = "system.time_ns"
        return ScalarSample(
            value=value,
            captured_at_ns=captured_at_ns,
            source_event_id=_digest(
                {
                    "server_identity_sha256": self._server_identity,
                    "node_id": spec.state_node_id,
                    "value": value,
                    "captured_at_ns": captured_at_ns,
                    "status": str(status),
                }
            ),
            quality="server_reported",
            wall_clock_source=wall_clock_source,
            source_epoch=self._server_identity,
        )

    async def write_scalar(
        self,
        resource_id: str,
        value: float,
        *,
        idempotency_key: str,
        recovery: bool = False,
    ) -> ScalarWriteResult:
        spec = self._resources.get(resource_id)
        if spec is None or not spec.command_node_id:
            raise PermissionError("opcua_resource_not_writable")
        stable_key = str(idempotency_key or "").strip()
        if not stable_key or len(stable_key.encode("utf-8")) > 256:
            raise ValueError("opcua_idempotency_key_invalid")
        encoded = spec.encode(value)
        async with self._write_lock:
            previous = self._write_receipts.get(stable_key)
            if previous is not None:
                old_resource, old_value, old_result = previous
                if old_resource != resource_id or old_value != float(value):
                    raise OPCUAConnectorError("opcua_idempotency_key_conflict")
                return old_result
            client = await self._ensure_connected()
            try:
                from asyncua import ua

                variant_type = getattr(
                    ua.VariantType,
                    _VARIANT_TYPE_NAMES[spec.value_type],
                )
                await client.get_node(spec.command_node_id).write_value(
                    encoded,
                    varianttype=variant_type,
                )
            except (AttributeError, OSError, RuntimeError, TimeoutError) as exc:
                await self._invalidate(client)
                raise OPCUAConnectorError("opcua_write_failed") from exc
            result = ScalarWriteResult(
                accepted=True,
                transport_completed=True,
                receipt={
                    "protocol": self.transport_id,
                    "resource_id": resource_id,
                    "server_identity_sha256": self._server_identity,
                    "command_node_sha256": _digest(spec.command_node_id),
                    "idempotency_sha256": _digest(stable_key),
                    "recovery": recovery,
                },
            )
            if len(self._write_receipts) >= _MAX_IDEMPOTENCY_RECEIPTS:
                self._write_receipts.pop(next(iter(self._write_receipts)))
            self._write_receipts[stable_key] = (resource_id, float(value), result)
            return result

    async def stop(self) -> None:
        async with self._lifecycle_lock:
            client = self._client
            self._client = None
            self._connected = False
        if client is not None:
            try:
                await client.disconnect()
            except (OSError, RuntimeError, TimeoutError):
                return


class OPCUAConnector:
    """Expose declared OPC UA variables as attachable Reality Reach channels."""

    connector_id = "opcua.manifest"

    def __init__(
        self,
        transport: OPCUAScalarTransport,
        resources: tuple[OPCUAResourceSpec, ...],
        *,
        candidate_ttl_s: float = 180.0,
    ) -> None:
        if not isinstance(transport, OPCUAScalarTransport):
            raise TypeError("transport must satisfy OPCUAScalarTransport")
        if not resources:
            raise ValueError("resources must not be empty")
        self._transport = transport
        self._resources = {item.resource_id: item for item in resources}
        self._ttl_s = max(30.0, min(float(candidate_ttl_s), 3600.0))

    def _profile(self, spec: OPCUAResourceSpec) -> ScalarResourceProfile:
        return ScalarResourceProfile(
            resource_id=spec.resource_id,
            observable=spec.observable,
            unit=spec.unit,
            domain=spec.domain,
            resolution=spec.resolution,
            writable=spec.writable,
            physical_identity_sha256=_digest(
                {
                    "server": self._transport.server_identity_sha256,
                    "device_id": spec.device_id,
                    "resource_id": spec.resource_id,
                    "state_node_id": spec.state_node_id,
                    "command_node_id": spec.command_node_id,
                }
            ),
            owner="core.embodiment.opcua_connector",
            protocol="opcua",
            safe_value=spec.safe_value,
            tolerance=spec.tolerance,
            max_commands_per_minute=spec.max_commands_per_minute,
            cooldown_s=spec.cooldown_s,
            stale_after_s=spec.stale_after_s,
            readback_distinct_from_command=spec.writable,
        )

    async def discover(self) -> tuple[DeviceCandidate, ...]:
        candidates: list[DeviceCandidate] = []
        now_ns = max(1, time.time_ns())
        for spec in self._resources.values():
            try:
                sample = await self._transport.read_scalar(spec.resource_id)
            except (OSError, RuntimeError, TimeoutError, TypeError, ValueError):
                continue
            profile = self._profile(spec)
            if not profile.domain.contains(sample.value):
                continue
            manifest = _digest(
                {
                    "spec_sha256": spec.sha256,
                    "profile_sha256": profile.sha256,
                    "server_identity_sha256": self._transport.server_identity_sha256,
                }
            )
            access = (
                (AttachmentAccess.OBSERVE, AttachmentAccess.CONTROL)
                if spec.writable
                else (AttachmentAccess.OBSERVE,)
            )
            candidates.append(
                DeviceCandidate(
                    candidate_id=(
                        "opcua.candidate."
                        + manifest.removeprefix("sha256:")[:32]
                    ),
                    connector_id=self.connector_id,
                    device_id=f"opcua.{spec.device_id}.{spec.resource_id}",
                    display_name=f"{spec.device_id}: {spec.observable}"[:160],
                    transport=self._transport.transport_id,
                    identity_fingerprint=profile.physical_identity_sha256,
                    manifest_sha256=manifest,
                    access=access,
                    discovered_at_ns=now_ns,
                    expires_at_ns=now_ns + int(self._ttl_s * 1_000_000_000),
                    persistent_identity=True,
                    proposal_salience=0.4,
                    metadata={
                        "resource_id": spec.resource_id,
                        "device_id": spec.device_id,
                        "spec_sha256": spec.sha256,
                        "profile_sha256": profile.sha256,
                        "control_available": spec.writable,
                        "independent_readback": spec.writable,
                        "state_node_sha256": _digest(spec.state_node_id),
                        "command_node_sha256": (
                            _digest(spec.command_node_id)
                            if spec.command_node_id
                            else ""
                        ),
                    },
                )
            )
        return tuple(sorted(candidates, key=lambda item: item.candidate_id))

    async def attach(
        self,
        candidate: DeviceCandidate,
        access: tuple[AttachmentAccess, ...],
    ) -> LiveChannelAdapter:
        if candidate.connector_id != self.connector_id:
            raise ValueError("opcua_candidate_connector_mismatch")
        requested = set(access)
        if not requested or not requested.issubset(set(candidate.access)):
            raise PermissionError("opcua_attachment_access_not_declared")
        if AttachmentAccess.CONTROL in requested and AttachmentAccess.OBSERVE not in requested:
            raise PermissionError("opcua_control_requires_observation")
        resource_id = str(candidate.metadata.get("resource_id") or "")
        spec = self._resources.get(resource_id)
        if spec is None:
            raise LookupError("opcua_candidate_resource_missing")
        current = next(
            (item for item in await self.discover() if item.candidate_id == candidate.candidate_id),
            None,
        )
        if current is None or (
            current.identity_fingerprint != candidate.identity_fingerprint
            or current.manifest_sha256 != candidate.manifest_sha256
        ):
            raise RuntimeError("opcua_candidate_changed_before_attachment")
        profile = self._profile(spec)
        if AttachmentAccess.CONTROL not in requested:
            profile = replace(profile, writable=False, safe_value=None)
        sample = await self._transport.read_scalar(resource_id)
        return ScalarRealityAdapter(self._transport, profile, initial_sample=sample)

    async def detach(self, _adapter: LiveChannelAdapter) -> None:
        return None

    async def stop(self) -> None:
        stop = getattr(self._transport, "stop", None)
        if not callable(stop):
            return
        result = stop()
        if asyncio.iscoroutine(result):
            await result


def build_configured_opcua_connector() -> OPCUAConnector:
    raw = str(env_str("AURA_OPCUA_RESOURCES_JSON", description="OPC UA resources JSON", owner="core.embodiment.opcua") or "").strip()
    if not raw:
        raise OPCUAConnectorError("opcua_resource_manifest_missing")
    resources = parse_opcua_resource_manifest(raw)
    return OPCUAConnector(AsyncUaScalarTransport(resources), resources)


__all__ = [
    "AsyncUaScalarTransport",
    "OPCUAConnector",
    "OPCUAConnectorError",
    "OPCUAResourceSpec",
    "OPCUAScalarTransport",
    "build_configured_opcua_connector",
    "parse_opcua_resource_manifest",
]
