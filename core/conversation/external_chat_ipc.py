"""Private, authenticated, durable IPC for external Aura chat surfaces.

The original terminal surface used two predictable FIFOs under ``/tmp``. A
writer removed a reply from memory before opening the outbound pipe, one line
was treated as one message, and no reader meant permanent loss. This module
uses a private random channel directory and signed JSON frame files instead.
Files are the transport deliberately: an outbound frame remains durable until
the client writes a signed acknowledgement, and a restarted client naturally
replays anything it did not acknowledge.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import stat
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.runtime.atomic_writer import atomic_write_text, ensure_private_directory

PROTOCOL_VERSION = 1
MAX_FRAME_BYTES = 1_048_576
MAX_MESSAGE_CHARS = 262_144
CHANNEL_DIR_MODE = 0o700


class ExternalChatIPCError(RuntimeError):
    """Base error for a channel whose custody or authenticity is invalid."""


class ExternalChatAuthenticationError(ExternalChatIPCError):
    """A frame did not authenticate for this channel."""


class ExternalChatLaunchError(ExternalChatIPCError):
    """The external surface did not prove that it launched."""


@dataclass(frozen=True)
class AuthenticatedFrame:
    message_id: str
    kind: str
    text: str
    sent_at_ns: int


def _canonical_payload(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


class FrameCodec:
    """Encode and verify one independently authenticated channel frame."""

    def __init__(self, *, channel_id: str, secret: bytes) -> None:
        if not channel_id or not secret:
            raise ValueError("external chat frames require channel identity and secret")
        self.channel_id = channel_id
        self.secret = bytes(secret)

    def encode(self, *, kind: str, text: str = "", message_id: str = "") -> str:
        normalized_kind = str(kind or "").strip()
        normalized_text = str(text or "")
        if not normalized_kind:
            raise ValueError("external chat frame kind is required")
        if len(normalized_text) > MAX_MESSAGE_CHARS:
            raise ValueError("external chat frame exceeds message budget")
        identity = str(message_id or uuid.uuid4().hex)
        body = {
            "version": PROTOCOL_VERSION,
            "channel_id": self.channel_id,
            "message_id": identity,
            "kind": normalized_kind,
            "text": normalized_text,
            "sent_at_ns": time.time_ns(),
        }
        body["mac"] = hmac.new(
            self.secret,
            _canonical_payload(body),
            hashlib.sha256,
        ).hexdigest()
        encoded = _canonical_payload(body).decode("utf-8")
        if len(encoded.encode("utf-8")) > MAX_FRAME_BYTES:
            raise ValueError("external chat frame exceeds wire budget")
        return encoded + "\n"

    def decode(self, encoded: str | bytes) -> AuthenticatedFrame:
        raw = encoded.encode("utf-8") if isinstance(encoded, str) else bytes(encoded)
        if not raw or len(raw) > MAX_FRAME_BYTES:
            raise ExternalChatAuthenticationError("external chat frame size is invalid")
        try:
            envelope = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError) as exc:
            raise ExternalChatAuthenticationError("external chat frame is not JSON") from exc
        if not isinstance(envelope, dict):
            raise ExternalChatAuthenticationError("external chat frame is not an object")
        supplied_mac = str(envelope.pop("mac", ""))
        expected_mac = hmac.new(
            self.secret,
            _canonical_payload(envelope),
            hashlib.sha256,
        ).hexdigest()
        if not supplied_mac or not hmac.compare_digest(supplied_mac, expected_mac):
            raise ExternalChatAuthenticationError("external chat frame MAC is invalid")
        if envelope.get("version") != PROTOCOL_VERSION:
            raise ExternalChatAuthenticationError("external chat protocol version mismatch")
        if envelope.get("channel_id") != self.channel_id:
            raise ExternalChatAuthenticationError("external chat channel identity mismatch")
        message_id = str(envelope.get("message_id") or "")
        kind = str(envelope.get("kind") or "")
        text = str(envelope.get("text") or "")
        sent_at_ns = envelope.get("sent_at_ns")
        if not message_id or not kind or not isinstance(sent_at_ns, int):
            raise ExternalChatAuthenticationError("external chat frame fields are incomplete")
        valid_message_id = (
            message_id == "client_ready"
            if kind == "client_ready"
            else len(message_id) == 32
            and all(character in "0123456789abcdef" for character in message_id.lower())
        )
        if not valid_message_id:
            raise ExternalChatAuthenticationError("external chat message identity is invalid")
        if len(text) > MAX_MESSAGE_CHARS:
            raise ExternalChatAuthenticationError("external chat frame text is oversized")
        return AuthenticatedFrame(
            message_id=message_id,
            kind=kind,
            text=text,
            sent_at_ns=sent_at_ns,
        )


def _validate_private_directory(path: Path) -> None:
    if path.is_symlink():
        raise ExternalChatIPCError(f"external chat directory may not be a symlink: {path}")
    info = path.stat()
    if not stat.S_ISDIR(info.st_mode):
        raise ExternalChatIPCError(f"external chat channel is not a directory: {path}")
    if hasattr(os, "getuid") and info.st_uid != os.getuid():
        raise ExternalChatIPCError(f"external chat directory has a foreign owner: {path}")
    if stat.S_IMODE(info.st_mode) != CHANNEL_DIR_MODE:
        raise ExternalChatIPCError(f"external chat directory is not private: {path}")


def create_private_channel_directory(root: Path | None = None) -> Path:
    """Create an owner-only random channel namespace outside shared ``/tmp``."""

    # state_root() rather than Path.home(): a test run, a worktree or an
    # alternate profile sets the root explicitly, and resolving it here put the
    # external chat channel under the live instance's directory regardless.
    from core.runtime.state_ownership import state_root

    base = Path(root) if root is not None else state_root() / "runtime" / "external_chat"
    if base.exists() and base.is_symlink():
        raise ExternalChatIPCError(f"external chat runtime root may not be a symlink: {base}")
    ensure_private_directory(base)
    _validate_private_directory(base)
    channel = Path(tempfile.mkdtemp(prefix="channel-", dir=base))
    channel.chmod(CHANNEL_DIR_MODE)
    _validate_private_directory(channel)
    return channel


class DurableChannelSpool:
    """Authenticated, acknowledgement-driven message custody for one channel."""

    def __init__(self, channel_dir: Path, codec: FrameCodec) -> None:
        self.channel_dir = Path(channel_dir)
        self.codec = codec
        _validate_private_directory(self.channel_dir)
        # The five lanes of one channel, made from one place.
        lanes: dict[str, Path] = {}
        for name in ("to_client", "to_server", "acks_to_client", "acks_to_server", "control"):
            lanes[name] = ensure_private_directory(self.channel_dir / name)
        self.to_client = lanes["to_client"]
        self.to_server = lanes["to_server"]
        self.acks_to_client = lanes["acks_to_client"]
        self.acks_to_server = lanes["acks_to_server"]
        self.control = lanes["control"]

    @staticmethod
    def _frame_path(directory: Path, message_id: str) -> Path:
        if len(message_id) != 32 or any(
            ch not in "0123456789abcdef" for ch in message_id.lower()
        ):
            raise ValueError("external chat message id is not a 128-bit hexadecimal identity")
        return directory / f"{message_id.lower()}.frame"

    def write_frame(
        self,
        directory: Path,
        *,
        kind: str,
        text: str = "",
        message_id: str = "",
    ) -> str:
        identity = str(message_id or uuid.uuid4().hex).lower()
        path = self._frame_path(directory, identity)
        atomic_write_text(
            path,
            self.codec.encode(kind=kind, text=text, message_id=identity),
            mode=0o600,
        )
        return identity

    def read_frame(self, path: Path) -> AuthenticatedFrame:
        if path.is_symlink() or path.parent not in {
            self.to_client,
            self.to_server,
            self.acks_to_client,
            self.acks_to_server,
            self.control,
        }:
            raise ExternalChatAuthenticationError("external chat frame path escaped its channel")
        return self.codec.decode(path.read_bytes())

    def enqueue_outbound(self, text: str, *, message_id: str = "") -> str:
        return self.write_frame(
            self.to_client,
            kind="aura_message",
            text=text,
            message_id=message_id,
        )

    def pending_outbound(self) -> tuple[AuthenticatedFrame, ...]:
        frames: list[AuthenticatedFrame] = []
        for path in sorted(self.to_client.glob("*.frame")):
            frames.append(self.read_frame(path))
        return tuple(sorted(frames, key=lambda frame: (frame.sent_at_ns, frame.message_id)))

    def acknowledge_outbound(self) -> tuple[str, ...]:
        acknowledged: list[str] = []
        for ack_path in sorted(self.acks_to_server.glob("*.frame")):
            ack = self.read_frame(ack_path)
            if ack.kind != "ack":
                raise ExternalChatAuthenticationError("outbound acknowledgement kind is invalid")
            message_path = self._frame_path(self.to_client, ack.message_id)
            try:
                message_path.unlink()
            # not a failure: a process or entry already gone is the state this is reaching.
            except FileNotFoundError:
                pass
            ack_path.unlink()
            acknowledged.append(ack.message_id)
        return tuple(acknowledged)

    def pending_inbound(self) -> tuple[AuthenticatedFrame, ...]:
        inbound: list[AuthenticatedFrame] = []
        for message_path in sorted(self.to_server.glob("*.frame")):
            frame = self.read_frame(message_path)
            if frame.kind != "user_message":
                raise ExternalChatAuthenticationError("inbound message kind is invalid")
            inbound.append(frame)
        return tuple(sorted(inbound, key=lambda frame: (frame.sent_at_ns, frame.message_id)))

    def acknowledge_inbound(self, message_id: str) -> None:
        """Acknowledge only after the orchestrator accepted the exact message."""

        self.write_frame(
            self.acks_to_client,
            kind="ack",
            message_id=message_id,
        )
        try:
            self._frame_path(self.to_server, message_id).unlink()
        # not a failure: a process or entry already gone is the state this is reaching.
        except FileNotFoundError:
            pass

    def inbound_completed(self, message_id: str) -> bool:
        """Return whether this exact inbound turn already crossed cognition.

        The completion marker is intentionally retained for the life of the
        channel.  If the process fails after publishing a response but before
        deleting the inbound frame, replay acknowledges the original turn
        instead of asking Aura to reason about it twice.
        """

        path = self._frame_path(self.control, message_id)
        if not path.exists():
            return False
        completed = self.read_frame(path)
        if completed.kind != "inbound_complete":
            raise ExternalChatAuthenticationError(
                "inbound completion marker kind is invalid"
            )
        return True

    def complete_inbound(self, message_id: str, *, response_text: str | None) -> str:
        """Commit response custody and completion before acknowledging input."""

        response_id = ""
        if response_text:
            response_id = self.enqueue_outbound(
                response_text,
                message_id=message_id,
            )
        self.write_frame(
            self.control,
            kind="inbound_complete",
            text=response_id,
            message_id=message_id,
        )
        self.acknowledge_inbound(message_id)
        return response_id

    def client_ready(self) -> bool:
        ready_path = self.control / "client_ready.frame"
        if not ready_path.exists():
            return False
        ready = self.read_frame(ready_path)
        return ready.kind == "client_ready"


def channel_secret() -> bytes:
    return secrets.token_bytes(32)


def terminal_client_source(*, channel_id: str, secret: bytes, channel_dir: Path) -> str:
    """Return a standalone standard-library client for the private spool."""

    # The secret lives only in this owner-readable script inside an owner-only
    # random directory. It never appears in argv, process listings, or /tmp.
    return f'''#!/usr/bin/env python3
import hashlib, hmac, json, os, pathlib, tempfile, threading, time, uuid

VERSION = {PROTOCOL_VERSION!r}
CHANNEL = {channel_id!r}
SECRET = bytes.fromhex({secret.hex()!r})
ROOT = pathlib.Path({str(channel_dir)!r})
TO_CLIENT = ROOT / "to_client"
TO_SERVER = ROOT / "to_server"
ACKS_TO_CLIENT = ROOT / "acks_to_client"
ACKS_TO_SERVER = ROOT / "acks_to_server"
CONTROL = ROOT / "control"
stop = threading.Event()

def canonical(payload):
    return json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True).encode("utf-8")

def encode(kind, text="", message_id=""):
    body = {{"version": VERSION, "channel_id": CHANNEL, "message_id": message_id or uuid.uuid4().hex, "kind": kind, "text": text, "sent_at_ns": time.time_ns()}}
    body["mac"] = hmac.new(SECRET, canonical(body), hashlib.sha256).hexdigest()
    return canonical(body) + b"\\n"

def decode(path):
    envelope = json.loads(path.read_bytes())
    supplied = str(envelope.pop("mac", ""))
    expected = hmac.new(SECRET, canonical(envelope), hashlib.sha256).hexdigest()
    if not supplied or not hmac.compare_digest(supplied, expected) or envelope.get("version") != VERSION or envelope.get("channel_id") != CHANNEL:
        raise ValueError("unauthenticated external chat frame")
    return envelope

def publish(directory, kind, text="", message_id=""):
    identity = message_id or uuid.uuid4().hex
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".frame-", dir=directory)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb", closefd=True) as handle:
            handle.write(encode(kind, text, identity)); handle.flush(); os.fsync(handle.fileno())
        os.replace(tmp, directory / (identity + ".frame"))
        dir_fd = os.open(directory, os.O_RDONLY)
        try: os.fsync(dir_fd)
        finally: os.close(dir_fd)
    finally:
        try: os.unlink(tmp)
        except FileNotFoundError: pass
    return identity

def input_loop():
    while not stop.is_set():
        try: text = input("YOU: ")
        except (EOFError, KeyboardInterrupt): stop.set(); return
        if not text.strip(): continue
        publish(TO_SERVER, "user_message", text)
        if text.strip().lower() in {{"exit", "quit"}}: stop.set(); return

publish(CONTROL, "client_ready", message_id="client_ready")
print("==================================")
print("            AURA CHAT")
print("==================================\\n")
threading.Thread(target=input_loop, daemon=True).start()
seen = set()
while not stop.is_set():
    if not ROOT.exists():
        stop.set(); break
    for path in sorted(TO_CLIENT.glob("*.frame")):
        try: frame = decode(path)
        except (OSError, ValueError, json.JSONDecodeError): continue
        identity = str(frame.get("message_id", ""))
        if frame.get("kind") != "aura_message" or not identity: continue
        already_acked = (ACKS_TO_SERVER / (identity + ".frame")).exists()
        if identity not in seen and not already_acked:
            text = str(frame.get("text", ""))
            print("\\nAURA: " + text.replace("\\n", "\\n      ") + "\\n")
            seen.add(identity)
        publish(ACKS_TO_SERVER, "ack", message_id=identity)
    for ack_path in sorted(ACKS_TO_CLIENT.glob("*.frame")):
        try:
            ack = decode(ack_path)
            if ack.get("kind") == "ack": ack_path.unlink()
        except (OSError, ValueError, json.JSONDecodeError): pass
    time.sleep(0.05)
'''
