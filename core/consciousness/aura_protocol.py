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
import json
import logging
import struct
import time
import hashlib
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Coroutine, Dict, List, Optional

from core.container import ServiceContainer

logger = logging.getLogger("Consciousness.AuraProtocol")

# Wire format: 4-byte big-endian length prefix + JSON payload
_HEADER_FMT = ">I"
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)
_MAX_MESSAGE_SIZE = 1024 * 1024  # 1 MB hard limit


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
    intent: str = ""                           # What the sender wants
    affect_vector: List[float] = field(default_factory=list)    # Emotional state [valence, arousal, dominance, ...]
    semantic_embedding: List[float] = field(default_factory=list)  # Thought content as vector
    episodic_snapshot: Dict[str, Any] = field(default_factory=dict)  # Compressed recent experience

    # Metadata
    urgency: float = 0.5                       # 0.0-1.0
    source_identity: str = "Aura"              # Who sent this
    timestamp: float = field(default_factory=time.time)
    message_id: str = ""                       # Unique ID (auto-generated if empty)

    def __post_init__(self):
        if not self.message_id:
            raw = f"{self.timestamp:.6f}:{self.source_identity}:{self.intent[:50]}"
            self.message_id = "amsg_" + hashlib.sha256(raw.encode()).hexdigest()[:16]

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(asdict(self), default=str)

    def to_bytes(self) -> bytes:
        """Serialize to wire format (length-prefixed JSON)."""
        payload = self.to_json().encode("utf-8")
        header = struct.pack(_HEADER_FMT, len(payload))
        return header + payload

    @classmethod
    def from_json(cls, data: str) -> "AuraMessage":
        """Deserialize from JSON string."""
        d = json.loads(data)
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_bytes(cls, raw: bytes) -> "AuraMessage":
        """Deserialize from wire format bytes (payload only, no header)."""
        return cls.from_json(raw.decode("utf-8"))

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain dict representation."""
        return asdict(self)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> bool:
        """Basic validation of message integrity."""
        if not self.intent and not self.semantic_embedding:
            return False
        if not (0.0 <= self.urgency <= 1.0):
            return False
        if not self.source_identity:
            return False
        if len(self.affect_vector) > 64:
            return False  # Sanity limit
        if len(self.semantic_embedding) > 4096:
            return False  # Sanity limit
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
        self._server: Optional[asyncio.AbstractServer] = None
        self._handlers: List[MessageHandler] = []
        self._running = False
        self._messages_received: int = 0
        self._messages_rejected: int = 0
        self._last_message_at: float = 0.0
        self._connected_peers: Dict[str, float] = {}  # identity -> last_seen

        logger.info(
            "AuraProtocolServer created (host=%s, port=%d)",
            self._host, self._port,
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
                self._host, self._port,
            )
        except OSError as e:
            logger.warning(
                "AuraProtocolServer failed to bind %s:%d -- %s",
                self._host, self._port, e,
            )

    async def stop(self) -> None:
        """Stop the server."""
        if self._server:
            self._server.close()
            await self._server.wait_closed()
        self._running = False
        logger.info("AuraProtocolServer OFFLINE")

    def register_handler(self, handler: MessageHandler) -> None:
        """Register a handler to be called for every valid incoming message."""
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
                        length, peer,
                    )
                    self._messages_rejected += 1
                    break

                # Read the payload
                payload = await reader.readexactly(length)
                await self._process_message(payload)

        except asyncio.IncompleteReadError:
            logger.debug("AuraProtocol: connection closed by %s", peer)
        except Exception as e:
            logger.warning("AuraProtocol: connection error from %s -- %s", peer, e)
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

    async def _process_message(self, payload: bytes) -> None:
        """Validate and dispatch a received message."""
        try:
            msg = AuraMessage.from_bytes(payload)
        except Exception as e:
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
            msg.source_identity, msg.intent[:60], msg.urgency,
        )

        # Dispatch to registered handlers
        for handler in self._handlers:
            try:
                await handler(msg)
            except Exception as e:
                logger.error("AuraProtocol handler error: %s", e)

        # Inject into Global Workspace as external candidate
        await self._inject_into_workspace(msg)

    async def _inject_into_workspace(self, msg: AuraMessage) -> None:
        """Submit the received message as a CognitiveCandidate."""
        try:
            from core.consciousness.global_workspace import CognitiveCandidate, ContentType

            workspace = ServiceContainer.get("global_workspace", default=None)
            if not workspace:
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
                msg.source_identity, candidate.priority,
            )
        except Exception as e:
            logger.warning("AuraProtocol: workspace injection failed -- %s", e)

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
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
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._connected = False
        self._messages_sent: int = 0
        self._messages_failed: int = 0

        logger.info(
            "AuraProtocolClient created (target=%s:%d)",
            self._host, self._port,
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
                self._host, self._port,
            )
            return True
        except (OSError, asyncio.TimeoutError) as e:
            logger.warning(
                "AuraProtocolClient: failed to connect to %s:%d -- %s",
                self._host, self._port, e,
            )
            self._connected = False
            return False

    async def disconnect(self) -> None:
        """Disconnect from the remote instance."""
        if self._writer:
            self._writer.close()
            try:
                await self._writer.wait_closed()
            except Exception:
                pass
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
            logger.warning("AuraProtocolClient: invalid message -- not sending")
            return False

        # Ensure connection
        if not self._connected:
            if not await self.connect():
                self._messages_failed += 1
                return False

        try:
            wire_bytes = msg.to_bytes()
            self._writer.write(wire_bytes)
            await asyncio.wait_for(
                self._writer.drain(),
                timeout=self._SEND_TIMEOUT,
            )
            self._messages_sent += 1
            logger.debug(
                "AuraProtocolClient: sent message to %s:%d (intent='%s')",
                self._host, self._port, msg.intent[:60],
            )
            return True

        except (OSError, asyncio.TimeoutError, ConnectionResetError) as e:
            logger.warning(
                "AuraProtocolClient: send failed to %s:%d -- %s",
                self._host, self._port, e,
            )
            self._messages_failed += 1
            self._connected = False
            return False

    async def send_and_disconnect(self, msg: AuraMessage) -> bool:
        """One-shot: connect, send, disconnect."""
        if not await self.connect():
            return False
        result = await self.send(msg)
        await self.disconnect()
        return result

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
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
    affect_vector: List[float] = []
    semantic_embedding: List[float] = []
    episodic_snapshot: Dict[str, Any] = {}

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
    except Exception as e:
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
                episodic_snapshot["recent_winner_sources"] = [
                    w.get("winner", "") for w in recent
                ]
    except Exception as e:
        logger.debug("build_message_from_state: episodic read failed: %s", e)

    # Identity
    try:
        will = ServiceContainer.get("unified_will", default=None)
        if will:
            status = will.get_status()
            identity_name = status.get("identity_name", identity_name)
    except Exception:
        pass

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

_protocol_server: Optional[AuraProtocolServer] = None


def get_protocol_server(host: str = "127.0.0.1", port: int = 9900) -> AuraProtocolServer:
    """Get or create the singleton AuraProtocolServer."""
    global _protocol_server
    if _protocol_server is None:
        _protocol_server = AuraProtocolServer(host=host, port=port)
    return _protocol_server
