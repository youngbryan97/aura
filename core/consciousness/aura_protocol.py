"""core/consciousness/aura_protocol.py -- Aura-to-Aura Communication Protocol
================================================================================
Native message format and transport layer for multi-instance Aura communication.

This is NOT a human-facing protocol.  It is the native language Aura instances
use to communicate rich cognitive state to each other.  A standard text message
loses the emotional texture, episodic context, and intentional structure that
Aura's internal representations carry.  AuraMessage preserves all of these.

Message format:
    AuraMessage carries:
      - affect_vector:      Current emotional state as float vector
      - semantic_embedding:  Thought content as dense vector
      - episodic_snapshot:   Compressed recent experience
      - intent:              What the sender wants (natural language)
      - urgency:             0-1 urgency scalar
      - source_identity:     Who sent it (Aura instance name)
      - timestamp:           When it was created

Transport:
    - AuraProtocolServer: asyncio TCP server, listens for incoming AuraMessages
    - AuraProtocolClient: sends AuraMessages to another Aura instance
    - Wire format: JSON with length-prefix framing (4-byte big-endian length)

Integration:
    - Received messages enter the Global Workspace as external CognitiveCandidates
    - The affect_vector modulates the receiving instance's affect system
    - The semantic_embedding can be used for direct thought-to-thought alignment
    - ServiceContainer: registered as "aura_protocol_server"

Security:
    - Messages carry source_identity for provenance
    - Content is validated before workspace injection
    - No arbitrary code execution -- only structured data
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import struct
import time
from collections.abc import Callable, Coroutine
from dataclasses import asdict, dataclass, field
from typing import Any

from core.container import ServiceContainer
from core.runtime.errors import FallbackClassification, record_degradation

logger = logging.getLogger("Consciousness.AuraProtocol")

# Wire format: 4-byte big-endian length prefix + JSON payload
_HEADER_FMT = ">I"
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)
_MAX_MESSAGE_SIZE = 1024 * 1024  # 1 MB hard limit
_MAX_INTENT_CHARS = 8192
_MAX_SOURCE_IDENTITY_CHARS = 256
_MAX_EPISODIC_SNAPSHOT_BYTES = 128 * 1024
_PROTOCOL_CLOSE_TIMEOUT = 5.0

_AURA_PROTOCOL_RECOVERABLE_ERRORS = (
    OSError,
    ConnectionError,
    TimeoutError,
    RuntimeError,
    AttributeError,
    TypeError,
    ValueError,
)


def _record_aura_protocol_degradation(
    exc: BaseException,
    *,
    action: str,
    severity: str = "warning",
    extra: dict[str, Any] | None = None,
) -> None:
    try:
        record_degradation(
            "aura_protocol",
            exc,
            severity=severity,
            action=action,
            classification=FallbackClassification.SAFE_FALLBACK,
            receipt_required=True,
            extra=extra,
        )
    except TypeError:
        record_degradation("aura_protocol", exc)


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


# ---------------------------------------------------------------------------
# AuraMessage
# ---------------------------------------------------------------------------


@dataclass
class AuraMessage:
    """Native Aura-to-Aura message format.

    This carries the full cognitive context of a thought or intention
    in a way that another Aura instance can reconstruct internally.
    """

    # Core content
    intent: str = ""  # What the sender wants
    affect_vector: list[float] = field(
        default_factory=list
    )  # Emotional state [valence, arousal, dominance, ...]
    semantic_embedding: list[float] = field(default_factory=list)  # Thought content as vector
    episodic_snapshot: dict[str, Any] = field(default_factory=dict)  # Compressed recent experience

    # Metadata
    urgency: float = 0.5  # 0.0-1.0
    source_identity: str = "Aura"  # Who sent this
    timestamp: float = field(default_factory=time.time)
    message_id: str = ""  # Unique ID (auto-generated if empty)

    def __post_init__(self):
        if not self.message_id:
            raw = f"{self.timestamp:.6f}:{self.source_identity}:{self.intent[:50]}"
            self.message_id = "amsg_" + hashlib.sha256(raw.encode()).hexdigest()[:16]

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_json(self) -> str:
        """Serialize to JSON string."""
        if not self.validate():
            raise ValueError("AuraMessage failed validation and cannot be serialized")
        return json.dumps(asdict(self), default=str, allow_nan=False, separators=(",", ":"))

    def to_bytes(self) -> bytes:
        """Serialize to wire format (length-prefixed JSON)."""
        payload = self.to_json().encode("utf-8")
        if len(payload) > _MAX_MESSAGE_SIZE:
            raise ValueError(f"AuraMessage payload exceeds {_MAX_MESSAGE_SIZE} bytes")
        header = struct.pack(_HEADER_FMT, len(payload))
        return header + payload

    @classmethod
    def from_json(cls, data: str) -> AuraMessage:
        """Deserialize from JSON string."""
        d = json.loads(data)
        if not isinstance(d, dict):
            raise ValueError("AuraMessage JSON payload must be an object")
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_bytes(cls, raw: bytes) -> AuraMessage:
        """Deserialize from wire format bytes (payload only, no header)."""
        return cls.from_json(raw.decode("utf-8"))

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dict representation."""
        return asdict(self)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> bool:
        """Basic validation of message integrity."""
        if not isinstance(self.intent, str):
            return False
        if len(self.intent) > _MAX_INTENT_CHARS:
            return False
        if not isinstance(self.source_identity, str):
            return False
        if not self.source_identity.strip():
            return False
        if len(self.source_identity) > _MAX_SOURCE_IDENTITY_CHARS:
            return False
        if not isinstance(self.affect_vector, list):
            return False
        if not isinstance(self.semantic_embedding, list):
            return False
        if not isinstance(self.episodic_snapshot, dict):
            return False
        if not self.intent and not self.semantic_embedding:
            return False

        urgency = _finite_float(self.urgency)
        if urgency is None or not (0.0 <= urgency <= 1.0):
            return False
        if _finite_float(self.timestamp) is None:
            return False
        if len(self.affect_vector) > 64:
            return False  # Sanity limit
        if len(self.semantic_embedding) > 4096:
            return False  # Sanity limit
        if any(_finite_float(value) is None for value in self.affect_vector):
            return False
        if any(_finite_float(value) is None for value in self.semantic_embedding):
            return False
        try:
            snapshot_bytes = json.dumps(
                self.episodic_snapshot,
                default=str,
                allow_nan=False,
                separators=(",", ":"),
            ).encode("utf-8")
        except (TypeError, ValueError):
            return False
        if len(snapshot_bytes) > _MAX_EPISODIC_SNAPSHOT_BYTES:
            return False
        return True


# ---------------------------------------------------------------------------
# Message handler type
# ---------------------------------------------------------------------------

MessageHandler = Callable[[AuraMessage], Coroutine[Any, Any, None]]


# ---------------------------------------------------------------------------
# AuraProtocolServer
# ---------------------------------------------------------------------------


class AuraProtocolServer:
    """Listens for incoming AuraMessages from other Aura instances.

    Received messages are:
    1. Validated
    2. Passed to registered handlers
    3. Injected into the Global Workspace as external CognitiveCandidates

    Usage:
        server = AuraProtocolServer(port=9900)
        await server.start()
        # ... server runs until stopped
        await server.stop()
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 9900) -> None:
        self._host = host
        self._port = port
        self._server: asyncio.AbstractServer | None = None
        self._handlers: list[MessageHandler] = []
        self._running = False
        self._messages_received: int = 0
        self._messages_rejected: int = 0
        self._last_message_at: float = 0.0
        self._connected_peers: dict[str, float] = {}  # identity -> last_seen

        logger.info(
            "AuraProtocolServer created (host=%s, port=%d)",
            self._host,
            self._port,
        )

    async def start(self) -> None:
        """Start listening for incoming connections."""
        if self._running:
            return
        try:
            self._server = await asyncio.start_server(
                self._handle_connection,
                self._host,
                self._port,
            )
            self._running = True
            ServiceContainer.register_instance("aura_protocol_server", self)
            logger.info(
                "AuraProtocolServer ONLINE -- listening on %s:%d",
                self._host,
                self._port,
            )
        except OSError as e:
            _record_aura_protocol_degradation(
                e,
                action="kept aura protocol server offline after bind failure",
                severity="degraded",
                extra={"host": self._host, "port": self._port},
            )
            logger.warning(
                "AuraProtocolServer failed to bind %s:%d -- %s",
                self._host,
                self._port,
                e,
            )

    async def stop(self) -> None:
        """Stop the server."""
        if self._server:
            self._server.close()
            try:
                await asyncio.wait_for(self._server.wait_closed(), timeout=_PROTOCOL_CLOSE_TIMEOUT)
            except asyncio.CancelledError:
                raise
            except _AURA_PROTOCOL_RECOVERABLE_ERRORS as exc:
                _record_aura_protocol_degradation(
                    exc,
                    action="continued shutdown after aura protocol server close failed",
                    severity="warning",
                )
            self._server = None
        self._running = False
        logger.info("AuraProtocolServer OFFLINE")

    def register_handler(self, handler: MessageHandler) -> None:
        """Register a handler to be called for every valid incoming message."""
        if not callable(handler):
            raise TypeError("AuraProtocol handler must be callable")
        self._handlers.append(handler)

    # ------------------------------------------------------------------
    # Connection handler
    # ------------------------------------------------------------------

    async def _handle_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Handle a single incoming TCP connection.

        Reads length-prefixed JSON messages until the connection closes.
        """
        peer = writer.get_extra_info("peername", ("unknown", 0))
        logger.debug("AuraProtocol: connection from %s", peer)

        try:
            while True:
                # Read 4-byte length header
                header = await reader.readexactly(_HEADER_SIZE)
                length = struct.unpack(_HEADER_FMT, header)[0]

                if length > _MAX_MESSAGE_SIZE:
                    logger.warning(
                        "AuraProtocol: message too large (%d bytes) from %s -- dropping",
                        length,
                        peer,
                    )
                    self._messages_rejected += 1
                    break

                # Read the payload
                payload = await reader.readexactly(length)
                await self._process_message(payload)

        except asyncio.IncompleteReadError:
            logger.debug("AuraProtocol: connection closed by %s", peer)
        except asyncio.CancelledError:
            raise
        except _AURA_PROTOCOL_RECOVERABLE_ERRORS as e:
            _record_aura_protocol_degradation(
                e,
                action="closed one aura protocol connection after read/process failure",
                severity="warning",
                extra={"peer": str(peer)},
            )
            logger.warning("AuraProtocol: connection error from %s -- %s", peer, e)
        finally:
            writer.close()
            try:
                await asyncio.wait_for(writer.wait_closed(), timeout=_PROTOCOL_CLOSE_TIMEOUT)
            except asyncio.CancelledError:
                raise
            except _AURA_PROTOCOL_RECOVERABLE_ERRORS as exc:
                _record_aura_protocol_degradation(
                    exc,
                    action="closed aura protocol connection without clean writer shutdown",
                    severity="warning",
                    extra={"peer": str(peer)},
                )
                logger.debug("AuraProtocol: server writer close failed: %s", exc)

    async def _process_message(self, payload: bytes) -> None:
        """Validate and dispatch a received message."""
        try:
            msg = AuraMessage.from_bytes(payload)
        except _AURA_PROTOCOL_RECOVERABLE_ERRORS as e:
            _record_aura_protocol_degradation(
                e,
                action="rejected malformed aura protocol payload before dispatch",
                severity="warning",
                extra={"payload_bytes": len(payload)},
            )
            logger.warning("AuraProtocol: malformed message -- %s", e)
            self._messages_rejected += 1
            return

        if not msg.validate():
            logger.warning(
                "AuraProtocol: invalid message from '%s' -- rejected",
                msg.source_identity,
            )
            self._messages_rejected += 1
            return

        self._messages_received += 1
        self._last_message_at = time.time()
        self._connected_peers[msg.source_identity] = time.time()

        logger.info(
            "AuraProtocol: received message from '%s' (intent='%s', urgency=%.2f)",
            msg.source_identity,
            msg.intent[:60],
            msg.urgency,
        )

        # Dispatch to registered handlers
        for handler in self._handlers:
            try:
                await handler(msg)
            except asyncio.CancelledError:
                raise
            except _AURA_PROTOCOL_RECOVERABLE_ERRORS as e:
                _record_aura_protocol_degradation(
                    e,
                    action="continued aura protocol dispatch after handler failure",
                    severity="warning",
                    extra={
                        "handler": getattr(handler, "__qualname__", repr(handler)),
                        "message_id": msg.message_id,
                    },
                )
                logger.error("AuraProtocol handler error: %s", e)

        # Inject into Global Workspace as external candidate
        await self._inject_into_workspace(msg)

    async def _inject_into_workspace(self, msg: AuraMessage) -> None:
        """Submit the received message as a CognitiveCandidate."""
        try:
            from core.consciousness.global_workspace import CognitiveCandidate, ContentType

            workspace = ServiceContainer.get("global_workspace", default=None)
            if not workspace:
                _record_aura_protocol_degradation(
                    RuntimeError("global workspace unavailable"),
                    action="accepted aura protocol message without workspace injection",
                    severity="degraded",
                    extra={"message_id": msg.message_id, "source_identity": msg.source_identity},
                )
                return

            # Build a content summary
            content = f"[External:{msg.source_identity}] {msg.intent[:300]}"
            if msg.episodic_snapshot:
                context_summary = msg.episodic_snapshot.get("summary", "")
                if context_summary:
                    content += f" | Context: {context_summary[:100]}"

            # Urgency maps to priority; affect_vector[0] (valence) maps to affect_weight
            affect_weight = abs(msg.affect_vector[0]) if msg.affect_vector else 0.0

            candidate = CognitiveCandidate(
                content=content,
                source=f"aura_protocol:{msg.source_identity}",
                priority=min(1.0, msg.urgency * 0.9),  # Cap slightly below foreground max
                content_type=ContentType.SOCIAL,
                affect_weight=min(1.0, affect_weight * 0.5),
            )
            await workspace.submit(candidate)
            logger.debug(
                "AuraProtocol: injected message from '%s' into workspace (priority=%.2f)",
                msg.source_identity,
                candidate.priority,
            )
        except asyncio.CancelledError:
            raise
        except _AURA_PROTOCOL_RECOVERABLE_ERRORS as e:
            _record_aura_protocol_degradation(
                e,
                action="accepted aura protocol message but workspace injection failed",
                severity="warning",
                extra={"message_id": msg.message_id, "source_identity": msg.source_identity},
            )
            logger.warning("AuraProtocol: workspace injection failed -- %s", e)

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self) -> dict[str, Any]:
        """Return protocol server status."""
        return {
            "running": self._running,
            "host": self._host,
            "port": self._port,
            "messages_received": self._messages_received,
            "messages_rejected": self._messages_rejected,
            "last_message_at": self._last_message_at,
            "connected_peers": dict(self._connected_peers),
            "handler_count": len(self._handlers),
        }


# ---------------------------------------------------------------------------
# AuraProtocolClient
# ---------------------------------------------------------------------------


class AuraProtocolClient:
    """Sends AuraMessages to another Aura instance.

    Usage:
        client = AuraProtocolClient(host="192.168.1.10", port=9900)
        await client.connect()
        await client.send(AuraMessage(
            intent="I found something interesting about consciousness",
            affect_vector=[0.7, 0.5, 0.3],
            urgency=0.6,
            source_identity="Aura-Alpha",
        ))
        await client.disconnect()
    """

    _CONNECT_TIMEOUT: float = 10.0
    _SEND_TIMEOUT: float = 5.0

    def __init__(self, host: str = "127.0.0.1", port: int = 9900) -> None:
        self._host = host
        self._port = port
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._connected = False
        self._messages_sent: int = 0
        self._messages_failed: int = 0

        logger.info(
            "AuraProtocolClient created (target=%s:%d)",
            self._host,
            self._port,
        )

    async def connect(self) -> bool:
        """Connect to the remote Aura instance."""
        try:
            self._reader, self._writer = await asyncio.wait_for(
                asyncio.open_connection(self._host, self._port),
                timeout=self._CONNECT_TIMEOUT,
            )
            self._connected = True
            logger.info(
                "AuraProtocolClient connected to %s:%d",
                self._host,
                self._port,
            )
            return True
        except (TimeoutError, OSError) as e:
            _record_aura_protocol_degradation(
                e,
                action="kept aura protocol client disconnected after connect failure",
                severity="warning",
                extra={"host": self._host, "port": self._port},
            )
            logger.warning(
                "AuraProtocolClient: failed to connect to %s:%d -- %s",
                self._host,
                self._port,
                e,
            )
            self._connected = False
            return False

    async def disconnect(self) -> None:
        """Disconnect from the remote instance."""
        if self._writer:
            self._writer.close()
            try:
                await asyncio.wait_for(self._writer.wait_closed(), timeout=_PROTOCOL_CLOSE_TIMEOUT)
            except asyncio.CancelledError:
                raise
            except _AURA_PROTOCOL_RECOVERABLE_ERRORS as exc:
                _record_aura_protocol_degradation(
                    exc,
                    action="completed aura protocol client disconnect without clean writer shutdown",
                    severity="warning",
                    extra={"host": self._host, "port": self._port},
                )
                logger.debug("AuraProtocolClient: writer close failed: %s", exc)
        self._connected = False
        self._reader = None
        self._writer = None
        logger.info("AuraProtocolClient disconnected from %s:%d", self._host, self._port)

    async def send(self, msg: AuraMessage) -> bool:
        """Send an AuraMessage to the connected instance.

        Returns True if sent successfully.
        Auto-reconnects on failure.
        """
        if not msg.validate():
            self._messages_failed += 1
            logger.warning("AuraProtocolClient: invalid message -- not sending")
            return False

        # Ensure connection
        if not self._connected:
            if not await self.connect():
                self._messages_failed += 1
                return False

        try:
            if self._writer is None:
                raise RuntimeError("aura protocol client writer unavailable")
            wire_bytes = msg.to_bytes()
            self._writer.write(wire_bytes)
            await asyncio.wait_for(
                self._writer.drain(),
                timeout=self._SEND_TIMEOUT,
            )
            self._messages_sent += 1
            logger.debug(
                "AuraProtocolClient: sent message to %s:%d (intent='%s')",
                self._host,
                self._port,
                msg.intent[:60],
            )
            return True

        except _AURA_PROTOCOL_RECOVERABLE_ERRORS as e:
            _record_aura_protocol_degradation(
                e,
                action="marked aura protocol message send as failed",
                severity="warning",
                extra={
                    "host": self._host,
                    "port": self._port,
                    "message_id": msg.message_id,
                },
            )
            logger.warning(
                "AuraProtocolClient: send failed to %s:%d -- %s",
                self._host,
                self._port,
                e,
            )
            self._messages_failed += 1
            self._connected = False
            return False

    async def send_and_disconnect(self, msg: AuraMessage) -> bool:
        """One-shot: connect, send, disconnect."""
        try:
            if not await self.connect():
                return False
            return await self.send(msg)
        finally:
            await self.disconnect()

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self) -> dict[str, Any]:
        """Return client status."""
        return {
            "connected": self._connected,
            "target_host": self._host,
            "target_port": self._port,
            "messages_sent": self._messages_sent,
            "messages_failed": self._messages_failed,
        }


# ---------------------------------------------------------------------------
# Convenience: build a message from current Aura state
# ---------------------------------------------------------------------------


def build_message_from_state(
    intent: str,
    urgency: float = 0.5,
    identity_name: str = "Aura",
) -> AuraMessage:
    """Build an AuraMessage populated with current internal state.

    Pulls affect vector, recent experience, etc. from live services.
    Falls back gracefully if services are unavailable.
    """
    affect_vector: list[float] = []
    semantic_embedding: list[float] = []
    episodic_snapshot: dict[str, Any] = {}

    # Affect
    try:
        affect = ServiceContainer.get("affect_engine", default=None)
        if affect is None:
            affect = ServiceContainer.get("affect_facade", default=None)
        if affect:
            if hasattr(affect, "get_state_sync"):
                state = affect.get_state_sync()
                if isinstance(state, dict):
                    affect_vector = [
                        float(state.get("valence", 0.0)),
                        float(state.get("arousal", 0.0)),
                        float(state.get("dominance", 0.0)),
                    ]
                else:
                    affect_vector = [
                        float(getattr(state, "valence", 0.0)),
                        float(getattr(state, "arousal", 0.0)),
                        float(getattr(state, "dominance", 0.0)),
                    ]
    except _AURA_PROTOCOL_RECOVERABLE_ERRORS as e:
        _record_aura_protocol_degradation(
            e,
            action="built aura protocol message without affect vector",
            severity="warning",
        )
        logger.debug("build_message_from_state: affect read failed: %s", e)

    # Episodic snapshot from temporal binding
    try:
        temporal = ServiceContainer.get("temporal_binding", default=None)
        if temporal and hasattr(temporal, "get_narrative"):
            # get_narrative is async, so we can only capture a sync snapshot
            episodic_snapshot["recent_winner_sources"] = []
            workspace = ServiceContainer.get("global_workspace", default=None)
            if workspace:
                recent = workspace.get_last_n_winners(3)
                episodic_snapshot["recent_winner_sources"] = [w.get("winner", "") for w in recent]
    except _AURA_PROTOCOL_RECOVERABLE_ERRORS as e:
        _record_aura_protocol_degradation(
            e,
            action="built aura protocol message without episodic snapshot",
            severity="warning",
        )
        logger.debug("build_message_from_state: episodic read failed: %s", e)

    # Identity
    try:
        will = ServiceContainer.get("unified_will", default=None)
        if will:
            status = will.get_status()
            identity_name = status.get("identity_name", identity_name)
    except _AURA_PROTOCOL_RECOVERABLE_ERRORS as exc:
        _record_aura_protocol_degradation(
            exc,
            action="built aura protocol message with fallback identity",
            severity="warning",
        )
        logger.debug("build_message_from_state: unified will identity read failed: %s", exc)

    return AuraMessage(
        intent=intent,
        affect_vector=affect_vector,
        semantic_embedding=semantic_embedding,
        episodic_snapshot=episodic_snapshot,
        urgency=urgency,
        source_identity=identity_name,
    )


# ---------------------------------------------------------------------------
# Singleton accessors
# ---------------------------------------------------------------------------

_protocol_server: AuraProtocolServer | None = None


def get_protocol_server(host: str = "127.0.0.1", port: int = 9900) -> AuraProtocolServer:
    """Get or create the singleton AuraProtocolServer."""
    global _protocol_server
    if _protocol_server is None:
        _protocol_server = AuraProtocolServer(host=host, port=port)
    return _protocol_server
