"""Private inherited channel for supervisor-certified detached worker results."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import select
import socket
import time
from collections.abc import Mapping
from typing import Any, Never

from core.brain.llm.latent_cortex.campaign_journal import (
    CampaignJournalError,
    canonical_json_bytes,
)
from core.runtime.detached_worker_origin import (
    DetachedWorkerOriginAuthority,
    DetachedWorkerOriginError,
)

logger = logging.getLogger(__name__)

WORKER_ORIGIN_FD_ENV = "AURA_DETACHED_WORKER_ORIGIN_FD"
WORKER_ORIGIN_SESSION_ENV = "AURA_DETACHED_WORKER_ORIGIN_SESSION"
WORKER_ORIGIN_REQUEST_SCHEMA = "aura.detached_worker_origin.request.v1"
WORKER_ORIGIN_RESPONSE_SCHEMA = "aura.detached_worker_origin.response.v1"
MAX_WORKER_ORIGIN_REQUEST_BYTES = 512 * 1024
MAX_WORKER_ORIGIN_RESPONSE_BYTES = 1024 * 1024
WORKER_ORIGIN_TRANSPORT = "unix_stream_length_prefixed_v1"
DEFAULT_WORKER_ORIGIN_IO_TIMEOUT_SECONDS = 5.0
_FRAME_HEADER_BYTES = 4

_REQUEST_KEYS = {
    "schema",
    "action",
    "session_id",
    "request_sequence",
    "result_body",
}
_RESPONSE_KEYS = {
    "schema",
    "status",
    "session_id",
    "request_sequence",
    "result",
    "error",
    "response_sha256",
}


class DetachedWorkerOriginChannelError(RuntimeError):
    """Stable framing, transport, or authority error."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _fail(code: str) -> Never:
    raise DetachedWorkerOriginChannelError(code)


def _sha256(value: Mapping[str, Any]) -> str:
    try:
        return hashlib.sha256(canonical_json_bytes(value)).hexdigest()
    except CampaignJournalError as exc:
        raise DetachedWorkerOriginChannelError("worker_origin_channel_json_invalid") from exc


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            _fail("worker_origin_channel_duplicate_key")
        value[key] = item
    return value


def _decode_canonical_object(
    payload: bytes,
    *,
    maximum: int,
    role: str,
) -> dict[str, Any]:
    if not payload or len(payload) > maximum:
        _fail(f"{role}_size_invalid")
    try:
        value = json.loads(payload, object_pairs_hook=_reject_duplicate_keys)
    except DetachedWorkerOriginChannelError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError):
        _fail(f"{role}_json_invalid")
    if not isinstance(value, dict):
        _fail(f"{role}_object_required")
    try:
        canonical = canonical_json_bytes(value)
    except (
        CampaignJournalError,
        TypeError,
        ValueError,
        RecursionError,
        OverflowError,
    ):
        _fail(f"{role}_json_invalid")
    if payload != canonical:
        _fail(f"{role}_noncanonical")
    return value


def _socket_type(channel: socket.socket) -> int:
    return int(channel.getsockopt(socket.SOL_SOCKET, socket.SO_TYPE))


def _validate_channel(channel: socket.socket) -> None:
    if channel.family != socket.AF_UNIX or _socket_type(channel) != socket.SOCK_STREAM:
        _fail("worker_origin_channel_type_invalid")


def create_worker_origin_socketpair() -> tuple[socket.socket, socket.socket]:
    """Create a private framed stream for one authority/worker pair.

    Darwin exposes ``SOCK_SEQPACKET`` but rejects Unix socket pairs with
    ``EPROTONOSUPPORT``. A length-prefixed Unix stream retains reliable,
    bidirectional inherited-FD custody while making every frame boundary and
    size check explicit.
    """

    supervisor, worker = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        supervisor.set_inheritable(False)
        worker.set_inheritable(False)
        supervisor.setblocking(False)
        _validate_channel(supervisor)
        _validate_channel(worker)
        return supervisor, worker
    except BaseException:
        supervisor.close()
        worker.close()
        raise


def _encoded_frame(
    value: Mapping[str, Any],
    *,
    maximum: int,
    role: str,
) -> bytes:
    try:
        payload = canonical_json_bytes(value)
    except (
        CampaignJournalError,
        TypeError,
        ValueError,
        RecursionError,
        OverflowError,
    ):
        _fail(f"{role}_json_invalid")
    if not payload or len(payload) > maximum:
        _fail(f"{role}_size_invalid")
    return len(payload).to_bytes(_FRAME_HEADER_BYTES, "big") + payload


def _send_frame(
    channel: socket.socket,
    value: Mapping[str, Any],
    *,
    maximum: int,
    role: str,
    timeout_s: float,
) -> None:
    frame = _encoded_frame(value, maximum=maximum, role=role)
    deadline = time.monotonic() + timeout_s
    previous_timeout = channel.gettimeout()
    view = memoryview(frame)
    try:
        channel.setblocking(False)
        while view:
            try:
                sent = channel.send(view)
            # not a failure: a non-blocking socket saying "not now" is the
            # normal case this loop is written around, and the deadline below
            # is what turns a persistent one into an error.
            except BlockingIOError:
                sent = 0
            if sent > 0:
                view = view[sent:]
                continue
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                _fail(f"{role}_timeout")
            _readable, writable, _exceptional = select.select(
                [],
                [channel],
                [],
                remaining,
            )
            if not writable:
                _fail(f"{role}_timeout")
    except OSError:
        _fail(f"{role}_transport_failed")
    finally:
        channel.settimeout(previous_timeout)


def _recv_frame_blocking(
    channel: socket.socket,
    *,
    maximum: int,
    role: str,
) -> bytes:
    header = bytearray()
    payload = bytearray()
    try:
        while len(header) < _FRAME_HEADER_BYTES:
            chunk = channel.recv(_FRAME_HEADER_BYTES - len(header))
            if not chunk:
                _fail(f"{role}_closed")
            header.extend(chunk)
        length = int.from_bytes(header, "big")
        if length <= 0 or length > maximum:
            _fail(f"{role}_size_invalid")
        while len(payload) < length:
            chunk = channel.recv(min(64 * 1024, length - len(payload)))
            if not chunk:
                _fail(f"{role}_closed")
            payload.extend(chunk)
    except OSError:
        _fail(f"{role}_transport_failed")
    return bytes(payload)


def _recv_frame_nonblocking(
    channel: socket.socket,
    buffer: bytearray,
    *,
    maximum: int,
    role: str,
) -> bytes | None:
    def extract_complete_frame() -> bytes | None:
        if len(buffer) < _FRAME_HEADER_BYTES:
            return None
        length = int.from_bytes(buffer[:_FRAME_HEADER_BYTES], "big")
        if length <= 0 or length > maximum:
            _fail(f"{role}_size_invalid")
        frame_end = _FRAME_HEADER_BYTES + length
        if len(buffer) < frame_end:
            return None
        if len(buffer) != frame_end:
            _fail(f"{role}_pipelining_prohibited")
        payload = bytes(buffer[_FRAME_HEADER_BYTES:frame_end])
        buffer.clear()
        return payload

    while True:
        try:
            chunk = channel.recv(64 * 1024)
        except BlockingIOError:
            break
        except OSError:
            _fail(f"{role}_transport_failed")
        if not chunk:
            _fail(f"{role}_closed")
        buffer.extend(chunk)
        if len(buffer) > maximum + _FRAME_HEADER_BYTES:
            _fail(f"{role}_size_invalid")
        if (payload := extract_complete_frame()) is not None:
            return payload
    return extract_complete_frame()


class DetachedWorkerOriginChannelServer:
    """Serve one authority over one inherited, session-bound packet channel."""

    def __init__(
        self,
        channel: socket.socket,
        authority: DetachedWorkerOriginAuthority,
        *,
        io_timeout_s: float = DEFAULT_WORKER_ORIGIN_IO_TIMEOUT_SECONDS,
    ) -> None:
        _validate_channel(channel)
        if (
            isinstance(io_timeout_s, bool)
            or not isinstance(io_timeout_s, (int, float))
            or not math.isfinite(float(io_timeout_s))
            or float(io_timeout_s) <= 0.0
        ):
            _fail("worker_origin_channel_timeout_invalid")
        session_id = authority.authorization_payload.get("session_id")
        if not isinstance(session_id, str) or len(session_id) != 32:
            _fail("worker_origin_channel_session_invalid")
        self._channel = channel
        self._authority = authority
        self._io_timeout_s = float(io_timeout_s)
        self._session_id = session_id
        self._request_sequence = 0
        self._receive_buffer = bytearray()
        self._poisoned = False
        self._peer_closed = False
        self._closed = False

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def poisoned(self) -> bool:
        return self._poisoned

    @property
    def peer_closed(self) -> bool:
        return self._peer_closed

    def _error_response(self, *, sequence: int, code: str) -> None:
        body = {
            "schema": WORKER_ORIGIN_RESPONSE_SCHEMA,
            "status": "error",
            "session_id": self._session_id,
            "request_sequence": sequence,
            "result": None,
            "error": code,
        }
        try:
            _send_frame(
                self._channel,
                {**body, "response_sha256": _sha256(body)},
                maximum=MAX_WORKER_ORIGIN_RESPONSE_BYTES,
                role="worker_origin_response",
                timeout_s=self._io_timeout_s,
            )
        except DetachedWorkerOriginChannelError as exc:
            # The worker is waiting on this refusal. Losing it means the
            # caller waits out its timeout instead of being told no.
            logger.warning(
                "worker origin refusal %s for sequence %s was not delivered: %s",
                code,
                sequence,
                exc,
            )

    def poll_once(self) -> bool:
        """Process at most one packet; return whether a packet was consumed."""

        if self._closed:
            _fail("worker_origin_channel_closed")
        if self._poisoned:
            _fail("worker_origin_channel_poisoned")
        if self._peer_closed:
            return False
        try:
            payload = _recv_frame_nonblocking(
                self._channel,
                self._receive_buffer,
                maximum=MAX_WORKER_ORIGIN_REQUEST_BYTES,
                role="worker_origin_request",
            )
            if payload is None:
                return False
            request = _decode_canonical_object(
                payload,
                maximum=MAX_WORKER_ORIGIN_REQUEST_BYTES,
                role="worker_origin_request",
            )
            sequence = request.get("request_sequence")
            if (
                set(request) != _REQUEST_KEYS
                or request.get("schema") != WORKER_ORIGIN_REQUEST_SCHEMA
                or request.get("action") != "record_result"
                or request.get("session_id") != self._session_id
                or isinstance(sequence, bool)
                or not isinstance(sequence, int)
                or sequence != self._request_sequence + 1
                or not isinstance(request.get("result_body"), dict)
            ):
                _fail("worker_origin_request_binding_invalid")
            try:
                result = self._authority.record_result(request["result_body"])
            except DetachedWorkerOriginError as exc:
                raise DetachedWorkerOriginChannelError(exc.code) from exc
            self._request_sequence = sequence
            body = {
                "schema": WORKER_ORIGIN_RESPONSE_SCHEMA,
                "status": "ok",
                "session_id": self._session_id,
                "request_sequence": sequence,
                "result": result,
                "error": None,
            }
            _send_frame(
                self._channel,
                {**body, "response_sha256": _sha256(body)},
                maximum=MAX_WORKER_ORIGIN_RESPONSE_BYTES,
                role="worker_origin_response",
                timeout_s=self._io_timeout_s,
            )
            return True
        except DetachedWorkerOriginChannelError as exc:
            if exc.code == "worker_origin_request_closed" and not self._receive_buffer:
                self._peer_closed = True
                return False
            self._poisoned = True
            sequence_value = locals().get("sequence")
            response_sequence = (
                sequence_value
                if isinstance(sequence_value, int) and not isinstance(sequence_value, bool)
                else self._request_sequence + 1
            )
            self._error_response(sequence=response_sequence, code=exc.code)
            raise

    def close(self) -> None:
        if not self._closed:
            self._channel.close()
            self._closed = True


class DetachedWorkerOriginChannelClient:
    """Worker-side typed-result client with no key or generic signing operation."""

    def __init__(
        self,
        channel: socket.socket,
        *,
        session_id: str,
        io_timeout_s: float = DEFAULT_WORKER_ORIGIN_IO_TIMEOUT_SECONDS,
    ) -> None:
        _validate_channel(channel)
        if (
            not isinstance(session_id, str)
            or len(session_id) != 32
            or any(character not in "0123456789abcdef" for character in session_id)
        ):
            _fail("worker_origin_channel_session_invalid")
        if (
            isinstance(io_timeout_s, bool)
            or not isinstance(io_timeout_s, (int, float))
            or not math.isfinite(float(io_timeout_s))
            or float(io_timeout_s) <= 0.0
        ):
            _fail("worker_origin_channel_timeout_invalid")
        channel.set_inheritable(False)
        channel.settimeout(float(io_timeout_s))
        self._channel = channel
        self._io_timeout_s = float(io_timeout_s)
        self._session_id = session_id
        self._request_sequence = 0
        self._closed = False

    @classmethod
    def from_environment(cls) -> DetachedWorkerOriginChannelClient:
        raw_fd = os.environ.pop(WORKER_ORIGIN_FD_ENV, "")
        session_id = os.environ.pop(WORKER_ORIGIN_SESSION_ENV, "")
        try:
            fd = int(raw_fd)
        except (TypeError, ValueError):
            _fail("worker_origin_channel_fd_invalid")
        if fd < 3:
            _fail("worker_origin_channel_fd_invalid")
        try:
            channel = socket.socket(fileno=fd)
        except OSError:
            _fail("worker_origin_channel_fd_invalid")
        try:
            return cls(channel, session_id=session_id)
        except BaseException:
            channel.close()
            raise

    @property
    def session_id(self) -> str:
        return self._session_id

    def record_result(
        self,
        result_body: Mapping[str, Any],
        *,
        cell_id: str,
        cell_type: str,
        attempt_id: str,
    ) -> dict[str, Any]:
        """Return the supervisor-certified result for one exact typed cell."""

        if self._closed:
            _fail("worker_origin_channel_closed")
        if not isinstance(result_body, Mapping):
            _fail("worker_origin_result_body_invalid")
        body = dict(result_body)
        if any(
            key in body
            for key in (
                "worker_origin",
                "cell_id",
                "cell_type",
                "attempt_id",
                "origin_session_id",
            )
        ):
            _fail("worker_origin_result_binding_preexisting")
        body.update(
            {
                "cell_id": cell_id,
                "cell_type": cell_type,
                "attempt_id": attempt_id,
                "origin_session_id": self._session_id,
            }
        )
        sequence = self._request_sequence + 1
        request = {
            "schema": WORKER_ORIGIN_REQUEST_SCHEMA,
            "action": "record_result",
            "session_id": self._session_id,
            "request_sequence": sequence,
            "result_body": body,
        }
        _send_frame(
            self._channel,
            request,
            maximum=MAX_WORKER_ORIGIN_REQUEST_BYTES,
            role="worker_origin_request",
            timeout_s=self._io_timeout_s,
        )
        payload = _recv_frame_blocking(
            self._channel,
            maximum=MAX_WORKER_ORIGIN_RESPONSE_BYTES,
            role="worker_origin_response",
        )
        response = _decode_canonical_object(
            payload,
            maximum=MAX_WORKER_ORIGIN_RESPONSE_BYTES,
            role="worker_origin_response",
        )
        material = dict(response)
        response_sha = material.pop("response_sha256", None)
        if (
            set(response) != _RESPONSE_KEYS
            or response.get("schema") != WORKER_ORIGIN_RESPONSE_SCHEMA
            or response.get("session_id") != self._session_id
            or response.get("request_sequence") != sequence
            or not isinstance(response_sha, str)
            or response_sha != _sha256(material)
        ):
            _fail("worker_origin_response_binding_invalid")
        if response.get("status") != "ok":
            error = response.get("error")
            _fail(error if isinstance(error, str) and error else "worker_origin_response_rejected")
        result = response.get("result")
        if not isinstance(result, dict) or result.get("worker_origin") is None:
            _fail("worker_origin_response_result_invalid")
        self._request_sequence = sequence
        return result

    def close(self) -> None:
        if not self._closed:
            self._channel.close()
            self._closed = True

    def __enter__(self) -> DetachedWorkerOriginChannelClient:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def worker_origin_channel_available() -> bool:
    return bool(os.environ.get(WORKER_ORIGIN_FD_ENV) and os.environ.get(WORKER_ORIGIN_SESSION_ENV))


__all__ = [
    "DetachedWorkerOriginChannelClient",
    "DetachedWorkerOriginChannelError",
    "DetachedWorkerOriginChannelServer",
    "DEFAULT_WORKER_ORIGIN_IO_TIMEOUT_SECONDS",
    "MAX_WORKER_ORIGIN_REQUEST_BYTES",
    "MAX_WORKER_ORIGIN_RESPONSE_BYTES",
    "WORKER_ORIGIN_FD_ENV",
    "WORKER_ORIGIN_SESSION_ENV",
    "WORKER_ORIGIN_TRANSPORT",
    "create_worker_origin_socketpair",
    "worker_origin_channel_available",
]
