"""Bounded rosbridge 2.1 WebSocket transport for Aura's ROS 2 connector."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import urllib.parse
import uuid
from collections.abc import Mapping
from typing import Any

from core.embodiment.ros2_contracts import (
    _DIGEST,
    _MAX_PENDING,
    _MAX_WIRE_BYTES,
    ROS2ActionSpec,
    ROS2ConnectorError,
    ROS2NodeSpec,
    ROS2TelemetrySpec,
    ROSGraphSnapshot,
    ROSTopicSample,
    _bounded_mapping,
    _digest,
    _identifier,
    _ros_name,
    _ros_type,
    _rosbridge_qos,
)
from core.runtime.audit_chain import canonical_json
from core.runtime.errors import NetworkEffectDenied
from core.runtime.lockdep import checked_async_lock
from core.runtime.network_gateway import get_network_gateway
from core.utils.task_tracker import get_task_tracker

logger = logging.getLogger("Aura.RealityReach.ROS2")


class RosbridgeWebSocketTransport:
    """One bounded, pinned rosbridge WebSocket session with correlation."""

    transport_id = "rosbridge.v2.1"

    def __init__(self) -> None:
        url = str(os.getenv("AURA_ROSBRIDGE_URL") or "").strip()
        installation = _identifier(
            os.getenv("AURA_ROSBRIDGE_INSTALLATION_ID"),
            name="AURA_ROSBRIDGE_INSTALLATION_ID",
        )
        parsed = urllib.parse.urlparse(url)
        if (
            parsed.scheme not in {"ws", "wss"}
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ROS2ConnectorError("rosbridge_url_invalid")
        if parsed.scheme == "ws" and not self._allow_plaintext():
            raise ROS2ConnectorError("rosbridge_plaintext_requires_explicit_opt_in")
        pin = str(os.getenv("AURA_ROSBRIDGE_SERVER_CERT_SHA256") or "").strip().lower()
        if parsed.scheme == "wss" and not _DIGEST.fullmatch(pin):
            raise ROS2ConnectorError("rosbridge_server_certificate_pin_required")
        version = str(os.getenv("AURA_ROSBRIDGE_PROTOCOL_VERSION") or "2.1.0").strip()
        try:
            version_tuple = tuple(int(part) for part in version.split("."))
        except ValueError as exc:
            raise ROS2ConnectorError("rosbridge_protocol_version_invalid") from exc
        if len(version_tuple) != 3 or version_tuple < (2, 1, 0):
            raise ROS2ConnectorError("rosbridge_protocol_2_1_required")
        self._url = url
        self._secure = parsed.scheme == "wss"
        self._certificate_pin = pin
        self._server_identity = _digest(
            {
                "installation": installation,
                "endpoint": f"{parsed.scheme}://{parsed.hostname}:{parsed.port or (443 if self._secure else 80)}{parsed.path or '/'}",
                "certificate_pin": pin,
            }
        )
        self._connect_lock = checked_async_lock("rosbridge.connect")
        self._send_lock = checked_async_lock("rosbridge.send")
        self._socket: Any | None = None
        self._receive_task: asyncio.Task[Any] | None = None
        self._service_futures: dict[str, asyncio.Future[Mapping[str, Any]]] = {}
        self._action_queues: dict[str, asyncio.Queue[dict[str, Any]]] = {}
        self._action_results: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self._latest: dict[str, ROSTopicSample] = {}
        self._topic_events: dict[str, asyncio.Event] = {}
        self._subscriptions: dict[str, str] = {}
        self._sequence = 0

    @staticmethod
    def _allow_plaintext() -> bool:
        from core.runtime.flags import FlagKind, declare

        return str(
            declare(
                "AURA_ROSBRIDGE_ALLOW_PLAINTEXT",
                kind=FlagKind.STRING,
                default="",
                description="Permit a plaintext development rosbridge session",
                owner="core.embodiment.ros2_connector",
            ).value()
        ).strip().lower() in {"1", "true", "yes", "on"}

    @property
    def server_identity_sha256(self) -> str:
        return self._server_identity

    @property
    def identity_stable(self) -> bool:
        return self._secure

    def _authorization_headers(self) -> dict[str, str]:
        authorization = str(os.getenv("AURA_ROSBRIDGE_AUTHORIZATION") or "").strip()
        return {"Authorization": authorization} if authorization else {}

    async def connect(self) -> None:
        async with self._connect_lock:
            if (
                self._socket is not None
                and self._receive_task is not None
                and not self._receive_task.done()
            ):
                return
            try:
                admission = await get_network_gateway().connect_websocket(
                    self._url,
                    headers=self._authorization_headers(),
                    open_timeout=10.0,
                    close_timeout=5.0,
                    ping_interval=20.0,
                    ping_timeout=10.0,
                    max_size=_MAX_WIRE_BYTES,
                    max_queue=64,
                    source="reality_reach:ros2.rosbridge",
                    read_only=True,
                    allow_private_target=True,
                )
                socket = admission.connection
                if self._secure:
                    ssl_object = socket.transport.get_extra_info("ssl_object")
                    certificate = ssl_object.getpeercert(binary_form=True) if ssl_object else None
                    actual = (
                        "sha256:" + __import__("hashlib").sha256(certificate or b"").hexdigest()
                    )
                    if certificate is None or actual != self._certificate_pin:
                        await socket.close(code=1008, reason="certificate pin mismatch")
                        raise ROS2ConnectorError("rosbridge_server_certificate_pin_mismatch")
            except ROS2ConnectorError:
                raise
            except (ImportError, NetworkEffectDenied, OSError, RuntimeError, TimeoutError) as exc:
                raise ROS2ConnectorError("rosbridge_connect_failed") from exc
            self._socket = socket
            self._receive_task = get_task_tracker().create_task(
                self._receive_loop(socket),
                name="ROSBridgeReceive",
            )

    async def _send(self, body: Mapping[str, Any]) -> None:
        encoded = canonical_json(body)
        if len(encoded) > _MAX_WIRE_BYTES:
            raise ROS2ConnectorError("rosbridge_message_too_large")
        await self.connect()
        async with self._send_lock:
            socket = self._socket
            if socket is None:
                raise ROS2ConnectorError("rosbridge_not_connected")
            try:
                await socket.send(encoded.decode("utf-8"))
            except (OSError, RuntimeError) as exc:
                await self._invalidate(socket, exc)
                raise ROS2ConnectorError("rosbridge_send_failed") from exc

    async def _receive_loop(self, socket: Any) -> None:
        failure: BaseException | None = None
        try:
            async for raw in socket:
                if not isinstance(raw, str) or len(raw.encode("utf-8")) > _MAX_WIRE_BYTES:
                    raise ROS2ConnectorError("rosbridge_non_json_or_oversize_message")
                decoded = json.loads(raw)
                if not isinstance(decoded, Mapping):
                    raise ROS2ConnectorError("rosbridge_message_not_an_object")
                self._dispatch(dict(decoded))
        except asyncio.CancelledError:
            raise
        except (json.JSONDecodeError, OSError, RuntimeError, TypeError, ValueError) as exc:
            failure = exc
        finally:
            await self._invalidate(socket, failure or ConnectionError("rosbridge_closed"))

    def _dispatch(self, body: dict[str, Any]) -> None:
        operation = str(body.get("op") or "")
        correlation_id = str(body.get("id") or "")
        if operation == "publish":
            topic = str(body.get("topic") or "")
            message = _bounded_mapping(body.get("msg"), name="ROS topic message")
            self._sequence += 1
            self._latest[topic] = ROSTopicSample(
                topic=topic,
                message=message,
                captured_at_ns=time.time_ns(),
                source_sequence=self._sequence,
            )
            self._topic_events.setdefault(topic, asyncio.Event()).set()
            return
        if operation == "service_response" and correlation_id:
            future = self._service_futures.pop(correlation_id, None)
            if future is not None and not future.done():
                if body.get("result") is True:
                    future.set_result(
                        _bounded_mapping(body.get("values"), name="ROS service response")
                    )
                else:
                    future.set_exception(ROS2ConnectorError("ros_service_call_failed"))
            return
        if operation == "action_feedback" and correlation_id:
            queue = self._action_queues.get(correlation_id)
            if queue is not None:
                self._put_action_event(queue, body)
            return
        if operation == "action_result" and correlation_id:
            future = self._action_results.get(correlation_id)
            if future is not None and not future.done():
                future.set_result(body)
            return
        if operation == "status" and str(body.get("level") or "").lower() in {"error", "fatal"}:
            error = ROS2ConnectorError(f"rosbridge_status_error:{str(body.get('msg') or '')[:240]}")
            future = self._service_futures.pop(correlation_id, None)
            if future is not None and not future.done():
                future.set_exception(error)
            queue = self._action_queues.get(correlation_id)
            if queue is not None:
                self._put_action_event(queue, {"op": "error", "error": str(error)})

    @staticmethod
    def _put_action_event(
        queue: asyncio.Queue[dict[str, Any]],
        event: dict[str, Any],
    ) -> None:
        if queue.full():
            queue.get_nowait()
        queue.put_nowait(event)

    @staticmethod
    def _fail_future(future: asyncio.Future[Any], error: BaseException) -> None:
        if future.done():
            return
        future.set_exception(error)
        # Mark the exception observed even when shutdown has no remaining waiter.
        future.exception()

    async def _invalidate(self, socket: Any, reason: BaseException) -> None:
        async with self._connect_lock:
            if socket is not self._socket:
                return
            self._socket = None
            self._receive_task = None
            self._subscriptions.clear()
            error = ROS2ConnectorError(f"rosbridge_connection_lost:{type(reason).__name__}")
            for future in self._service_futures.values():
                self._fail_future(future, error)
            self._service_futures.clear()
            for queue in self._action_queues.values():
                self._put_action_event(queue, {"op": "error", "error": str(error)})
            for future in self._action_results.values():
                self._fail_future(future, error)

    async def subscribe(self, spec: ROS2TelemetrySpec) -> None:
        prior = self._subscriptions.get(spec.topic)
        if prior is not None:
            if prior != spec.sha256:
                raise ROS2ConnectorError("ros_topic_redeclared_with_different_contract")
            return
        subscription_id = f"sub-{_digest(spec.to_dict()).removeprefix('sha256:')[:32]}"
        await self._send(
            {
                "op": "subscribe",
                "id": subscription_id,
                "topic": spec.topic,
                "type": spec.message_type,
                "qos": _rosbridge_qos(spec.qos),
                "throttle_rate": max(0, int(spec.sample_period_s * 1000)),
                "queue_length": max(1, min(spec.qos.depth, 1024)),
                "compression": "none",
            }
        )
        self._subscriptions[spec.topic] = spec.sha256

    async def unsubscribe(self, spec: ROS2TelemetrySpec) -> None:
        if spec.topic not in self._subscriptions:
            return
        subscription_id = f"sub-{_digest(spec.to_dict()).removeprefix('sha256:')[:32]}"
        await self._send({"op": "unsubscribe", "id": subscription_id, "topic": spec.topic})
        self._subscriptions.pop(spec.topic, None)

    async def latest(self, spec: ROS2TelemetrySpec, *, timeout_s: float) -> ROSTopicSample:
        await self.subscribe(spec)
        sample = self._latest.get(spec.topic)
        if sample is None or time.time_ns() - sample.captured_at_ns > int(spec.stale_after_s * 1e9):
            event = self._topic_events.setdefault(spec.topic, asyncio.Event())
            event.clear()
            try:
                await asyncio.wait_for(event.wait(), timeout=timeout_s)
            except TimeoutError as exc:
                raise ROS2ConnectorError("ros_topic_sample_timeout") from exc
            sample = self._latest.get(spec.topic)
        if sample is None:
            raise ROS2ConnectorError("ros_topic_sample_missing")
        return sample

    async def call_service(
        self,
        service: str,
        service_type: str,
        request: Mapping[str, Any],
        *,
        timeout_s: float,
        request_id: str | None = None,
    ) -> Mapping[str, Any]:
        if len(self._service_futures) >= _MAX_PENDING:
            raise ROS2ConnectorError("ros_service_pending_limit_reached")
        correlation = _identifier(
            request_id or f"service-{uuid.uuid4().hex}",
            name="request_id",
        )
        if correlation in self._service_futures:
            raise ROS2ConnectorError("ros_service_request_id_in_flight")
        future: asyncio.Future[Mapping[str, Any]] = asyncio.get_running_loop().create_future()
        self._service_futures[correlation] = future
        try:
            await self._send(
                {
                    "op": "call_service",
                    "id": correlation,
                    "service": _ros_name(service, name="service"),
                    "type": _ros_type(service_type, name="service_type", category="srv"),
                    "args": _bounded_mapping(request, name="ROS service request"),
                    "timeout": float(timeout_s),
                }
            )
            return await asyncio.wait_for(future, timeout=float(timeout_s))
        except TimeoutError as exc:
            raise ROS2ConnectorError("ros_service_response_timeout") from exc
        finally:
            self._service_futures.pop(correlation, None)

    async def graph_snapshot(self, spec: ROS2NodeSpec) -> ROSGraphSnapshot:
        topics_result = await self.call_service(
            "/rosapi/topics",
            "rosapi_msgs/srv/Topics",
            {},
            timeout_s=5.0,
        )
        services_result = await self.call_service(
            "/rosapi/services",
            "rosapi_msgs/srv/Services",
            {},
            timeout_s=5.0,
        )
        topic_names = list(topics_result.get("topics") or [])
        topic_types = list(topics_result.get("types") or [])
        topics = {
            str(name): str(topic_types[index]) if index < len(topic_types) else ""
            for index, name in enumerate(topic_names)
        }
        service_names = [str(item) for item in list(services_result.get("services") or [])]
        required_services = {item.service for item in spec.services}
        required_services.update(item.verification_service for item in spec.actions)
        required_services.update(
            item.command_service for item in spec.actions if item.transport_kind == "service"
        )
        required_services.update(
            item.reconciliation_service for item in spec.actions if item.reconciliation_service
        )
        services: dict[str, str] = {}
        for service in sorted(required_services & set(service_names)):
            result = await self.call_service(
                "/rosapi/service_type",
                "rosapi_msgs/srv/ServiceType",
                {"service": service},
                timeout_s=5.0,
            )
            services[service] = str(result.get("type") or "")
        for action in spec.actions:
            for suffix in ("send_goal", "get_result", "cancel_goal"):
                name = f"{action.action}/_action/{suffix}".replace("//", "/")
                if name in service_names:
                    services[name] = "action_transport"
        return ROSGraphSnapshot(topics=topics, services=services)

    async def send_action_goal(
        self,
        spec: ROS2ActionSpec,
        goal_id: str,
        request: Mapping[str, Any],
    ) -> None:
        goal_id = _identifier(goal_id, name="goal_id")
        completed = [item for item, future in self._action_results.items() if future.done()]
        while len(self._action_results) >= _MAX_PENDING and completed:
            retired = completed.pop(0)
            self._action_results.pop(retired, None)
            self._action_queues.pop(retired, None)
        if goal_id in self._action_queues or len(self._action_results) >= _MAX_PENDING:
            raise ROS2ConnectorError("ros_action_goal_id_in_flight_or_limit_reached")
        self._action_queues[goal_id] = asyncio.Queue(maxsize=1024)
        self._action_results[goal_id] = asyncio.get_running_loop().create_future()
        try:
            await self._send(
                {
                    "op": "send_action_goal",
                    "id": goal_id,
                    "action": spec.action,
                    "action_type": spec.action_type,
                    "args": _bounded_mapping(request, name="ROS action request"),
                    "feedback": True,
                }
            )
        except BaseException:
            self._action_queues.pop(goal_id, None)
            self._action_results.pop(goal_id, None)
            raise

    async def next_action_event(self, goal_id: str, *, timeout_s: float) -> Mapping[str, Any]:
        queue = self._action_queues.get(_identifier(goal_id, name="goal_id"))
        result = self._action_results.get(goal_id)
        if queue is None or result is None:
            raise ROS2ConnectorError("ros_action_goal_unknown")
        if not queue.empty():
            return queue.get_nowait()
        if result.done():
            return result.result()
        feedback_task = get_task_tracker().create_task(
            queue.get(),
            name=f"ROS2Feedback:{goal_id}",
        )
        result_task = asyncio.shield(result)
        try:
            done, pending = await asyncio.wait(
                {feedback_task, result_task},
                timeout=float(timeout_s),
                return_when=asyncio.FIRST_COMPLETED,
            )
            if not done:
                raise ROS2ConnectorError("ros_action_event_timeout")
            if feedback_task in done:
                return feedback_task.result()
            return result_task.result()
        finally:
            for task in (feedback_task, result_task):
                if not task.done():
                    task.cancel()

    async def cancel_action_goal(
        self,
        spec: ROS2ActionSpec,
        goal_id: str,
        *,
        timeout_s: float,
    ) -> bool:
        goal_id = _identifier(goal_id, name="goal_id")
        result = self._action_results.get(goal_id)
        if goal_id not in self._action_queues or result is None:
            return False
        await self._send({"op": "cancel_action_goal", "id": goal_id, "action": spec.action})
        try:
            event = await asyncio.wait_for(asyncio.shield(result), timeout=float(timeout_s))
        # not a failure: the wait ran out, which is what the timeout was set to decide.
        except TimeoutError:
            return False
        return event.get("op") == "action_result" and int(event.get("status") or 0) == 5

    async def close(self) -> None:
        async with self._connect_lock:
            socket = self._socket
            task = self._receive_task
            error = ROS2ConnectorError("rosbridge_transport_closed")
            for future in self._service_futures.values():
                self._fail_future(future, error)
            for future in self._action_results.values():
                self._fail_future(future, error)
            self._service_futures.clear()
            self._socket = None
            self._receive_task = None
            self._subscriptions.clear()
            self._action_queues.clear()
            self._action_results.clear()
        if task is not None:
            task.cancel()
        if socket is not None:
            try:
                await socket.close(code=1000, reason="Aura ROS adapter closed")
            except (OSError, RuntimeError) as exc:
                logger.debug(
                    "ROS bridge close failed after local state was fenced: %s",
                    exc,
                )
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)

__all__ = ["RosbridgeWebSocketTransport"]
