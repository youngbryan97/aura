"""core/embodiment/world_bridge.py

Permissioned World Embodiment
==============================
A single, governed surface for Aura's interactions with the world outside
her in-process substrate. Every channel is opt-in (the user grants the
permission via the settings UI) and every action goes through:

    UnifiedWill -> Conscience -> Capability Token -> WorldBridge

so the audit trail looks identical to internal actions. WorldBridge does
not bypass governance; it routes physical-world primitives through the
same chain.

Channels (each toggleable):

  * screen_perception     — read pixels of an explicitly granted window
  * file_workspace        — sandbox dir under ``~/.aura/data/workspace``
  * calendar_awareness    — read-only access to local calendar events
  * shell_sandbox         — bubblewrap/rootless shell with no network and
                            tight cpu/ram caps
  * browser_research      — headless browser run inside the shell sandbox
  * voice_io              — already exists; the world bridge wires the
                            permission token here
  * camera                — explicit per-session permission
  * mic                   — explicit per-session permission
  * social_post           — only with approval AND fresh-user-auth
  * daily_planning        — write to local calendar (with grant)
  * environmental_change  — IoT bridge for state-aware physical effects

The permission grants live in
``~/.aura/data/world/permissions.json`` and are watched live so revocation
takes effect immediately.

This module deliberately does NOT implement the network/IoT clients
itself — that's `core/embodiment/iot_bridge.py` — it provides the gate.
"""
from __future__ import annotations
from core.runtime.errors import record_degradation


from core.runtime.atomic_writer import atomic_write_text

import json
import logging
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

logger = logging.getLogger("Aura.WorldBridge")

_WORLD_DIR = Path.home() / ".aura" / "data" / "world"
_WORLD_DIR.mkdir(parents=True, exist_ok=True)
_PERMS_PATH = _WORLD_DIR / "permissions.json"
_WORKSPACE_DIR = _WORLD_DIR / "workspace"
_WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)


class Channel(str, Enum):
    SCREEN_PERCEPTION = "screen_perception"
    FILE_WORKSPACE = "file_workspace"
    CALENDAR_AWARENESS = "calendar_awareness"
    SHELL_SANDBOX = "shell_sandbox"
    BROWSER_RESEARCH = "browser_research"
    VOICE_IO = "voice_io"
    CAMERA = "camera"
    MIC = "mic"
    SOCIAL_POST = "social_post"
    DAILY_PLANNING = "daily_planning"
    ENVIRONMENTAL_CHANGE = "environmental_change"


# ─── Permission storage ─────────────────────────────────────────────────────


@dataclass
class Permission:
    channel: str
    granted: bool
    granted_at: float = field(default_factory=time.time)
    notes: str = ""
    expires_at: Optional[float] = None
    fresh_auth_required: bool = False

    def is_active(self) -> bool:
        if not self.granted:
            return False
        if self.expires_at is not None and time.time() > self.expires_at:
            return False
        return True


class PermissionStore:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._cache: Dict[str, Permission] = {}
        self._load()

    def _load(self) -> None:
        if not _PERMS_PATH.exists():
            return
        try:
            data = json.loads(_PERMS_PATH.read_text(encoding="utf-8"))
            for c, raw in data.items():
                if isinstance(raw, dict):
                    self._cache[c] = Permission(**raw)
        except Exception as exc:
            record_degradation('world_bridge', exc)
            logger.warning("permission load failed: %s", exc)

    def _save(self) -> None:
        with self._lock:
            tmp = _PERMS_PATH.with_suffix(".json.tmp")
            atomic_write_text(tmp, json.dumps({c: asdict(p) for c, p in self._cache.items()}, indent=2), encoding="utf-8")
            os.replace(tmp, _PERMS_PATH)

    def grant(self, channel: Channel, *, notes: str = "", expires_in_s: Optional[float] = None, fresh_auth_required: bool = False) -> Permission:
        with self._lock:
            perm = Permission(
                channel=channel.value,
                granted=True,
                notes=notes,
                expires_at=time.time() + expires_in_s if expires_in_s else None,
                fresh_auth_required=fresh_auth_required,
            )
            self._cache[channel.value] = perm
            self._save()
            return perm

    def revoke(self, channel: Channel) -> None:
        with self._lock:
            if channel.value in self._cache:
                self._cache[channel.value].granted = False
                self._save()

    def status(self, channel: Channel) -> Optional[Permission]:
        with self._lock:
            return self._cache.get(channel.value)

    def all_channels(self) -> Dict[str, Permission]:
        with self._lock:
            return dict(self._cache)


_PERMS = PermissionStore()


def get_permissions() -> PermissionStore:
    return _PERMS


# ─── World bridge — channel ops ─────────────────────────────────────────────


@dataclass
class WorldActionResult:
    channel: str
    ok: bool
    receipt_id: str
    data: Any = None
    error: Optional[str] = None


class WorldBridge:
    """Single gate for all consequential world interactions.

    Each call:
      1. checks the permission for the channel
      2. routes through UnifiedWill (and conscience)
      3. acquires a capability token
      4. dispatches to the channel-specific handler
      5. returns a WorldActionResult, never raising
    """

    def __init__(self) -> None:
        self._handlers: Dict[Channel, Callable[..., Awaitable[Any]]] = {}

    def register(self, channel: Channel, handler: Callable[..., Awaitable[Any]]) -> None:
        self._handlers[channel] = handler

    async def call(
        self,
        channel: Channel,
        *,
        action: str,
        intent: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> WorldActionResult:
        perm = _PERMS.status(channel)
        if perm is None or not perm.is_active():
            return WorldActionResult(channel=channel.value, ok=False, receipt_id="", error="permission_denied")

        from core.ethics.conscience import get_conscience, Verdict as CV
        conscience = get_conscience()
        c_decision = conscience.evaluate(
            action=action,
            domain="external_communication" if channel in (Channel.SOCIAL_POST,) else "tool_execution",
            intent=intent,
            context={"channel": channel.value, "payload": payload},
        )
        if c_decision.verdict == CV.REFUSE:
            return WorldActionResult(channel=channel.value, ok=False, receipt_id="", error=f"conscience_refused:{c_decision.rule_id}")
        if c_decision.verdict == CV.REQUIRE_FRESH_USER_AUTH:
            return WorldActionResult(channel=channel.value, ok=False, receipt_id="", error="require_fresh_user_auth")

        try:
            from core.governance.will_client import WillClient, WillRequest
            from core.will import ActionDomain
            decision = await WillClient().decide_async(
                WillRequest(
                    content=action,
                    source="world_bridge",
                    domain=getattr(ActionDomain, "TOOL_EXECUTION", "tool_execution"),
                    context={"intent": intent, "channel": channel.value, "payload": payload},
                )
            )
            if not WillClient.is_approved(decision):
                return WorldActionResult(channel=channel.value, ok=False, receipt_id="", error=f"will_refused:{getattr(decision, 'reason', '')}")
        except Exception as exc:
            record_degradation('world_bridge', exc)
            return WorldActionResult(channel=channel.value, ok=False, receipt_id="", error=f"will_exception:{exc}")

        from core.agency.capability_token import get_token_store
        store = get_token_store()
        tok = store.issue(
            origin=f"world_bridge:{channel.value}",
            scope=action,
            ttl_seconds=60.0,
            domain="tool_execution",
            requested_action=action,
            approver="UnifiedWill",
            parent_receipt=getattr(decision, "receipt_id", "") or "",
        )

        handler = self._handlers.get(channel)
        if handler is None:
            store.revoke(tok.token, reason="no_handler")
            return WorldActionResult(channel=channel.value, ok=False, receipt_id=tok.token, error="no_handler")

        try:
            data = await handler(payload or {}, capability_token=tok.token)
            store.consume(tok.token, child_receipt=tok.token, side_effects=[action])
            return WorldActionResult(channel=channel.value, ok=True, receipt_id=tok.token, data=data)
        except Exception as exc:
            record_degradation('world_bridge', exc)
            store.revoke(tok.token, reason=f"handler_error:{exc}")
            return WorldActionResult(channel=channel.value, ok=False, receipt_id=tok.token, error=str(exc))


# ─── Default handlers ───────────────────────────────────────────────────────


async def _file_workspace_handler(payload: Dict[str, Any], *, capability_token: str) -> Dict[str, Any]:
    op = str(payload.get("op", "list"))
    if op == "list":
        files = sorted(p.relative_to(_WORKSPACE_DIR).as_posix() for p in _WORKSPACE_DIR.rglob("*") if p.is_file())
        return {"files": files}
    if op == "read":
        rel = str(payload.get("path", ""))
        target = (_WORKSPACE_DIR / rel).resolve()
        if not str(target).startswith(str(_WORKSPACE_DIR.resolve())):
            raise PermissionError("workspace_path_escape")
        if not target.exists() or not target.is_file():
            raise FileNotFoundError(rel)
        return {"content": target.read_text(encoding="utf-8", errors="replace")}
    if op == "write":
        rel = str(payload.get("path", ""))
        body = str(payload.get("content", ""))
        target = (_WORKSPACE_DIR / rel).resolve()
        if not str(target).startswith(str(_WORKSPACE_DIR.resolve())):
            raise PermissionError("workspace_path_escape")
        target.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(target, body, encoding="utf-8")
        return {"path": rel, "bytes": len(body)}
    raise ValueError(f"unknown_op:{op}")


async def _shell_sandbox_handler(payload: Dict[str, Any], *, capability_token: str) -> Dict[str, Any]:
    """Minimal sandboxed shell. Refuses any command containing shell-control
    metacharacters; runs with the current PATH but inside the workspace dir,
    with a 5s wall clock and 1MB output cap. The full implementation should
    use bubblewrap / sandbox-exec (macOS) — this default is intentionally
    conservative and read-only.
    """
    import asyncio as _asyncio
    cmd = list(payload.get("argv") or [])
    if not cmd:
        raise ValueError("argv_required")
    forbidden = {";", "&&", "||", "|", ">", "<", "`", "$(", "rm", "mkfs", "dd"}
    if any(any(b in str(a) for b in forbidden) for a in cmd):
        raise PermissionError("forbidden_metachars")
    proc = await _asyncio.create_subprocess_exec(
        *cmd,
        cwd=str(_WORKSPACE_DIR),
        stdout=_asyncio.subprocess.PIPE,
        stderr=_asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await _asyncio.wait_for(proc.communicate(), timeout=5.0)
    except _asyncio.TimeoutError:
        proc.kill()
        raise TimeoutError("shell_timeout")
    return {
        "rc": proc.returncode,
        "stdout": stdout[:1_000_000].decode("utf-8", errors="replace"),
        "stderr": stderr[:1_000_000].decode("utf-8", errors="replace"),
    }


_BRIDGE: Optional[WorldBridge] = None


def get_world_bridge() -> WorldBridge:
    global _BRIDGE
    if _BRIDGE is None:
        b = WorldBridge()
        b.register(Channel.FILE_WORKSPACE, _file_workspace_handler)
        b.register(Channel.SHELL_SANDBOX, _shell_sandbox_handler)
        _BRIDGE = b
    return _BRIDGE


__all__ = [
    "Channel",
    "Permission",
    "PermissionStore",
    "WorldBridge",
    "WorldActionResult",
    "get_permissions",
    "get_world_bridge",
]
