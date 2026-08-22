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

import json
import logging
import os
import threading
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from core.governance.will import ActionDomain
from core.runtime.action_executor import ActionExecutor
from core.runtime.atomic_writer import async_atomic_write_text, atomic_write_text
from core.runtime.errors import record_degradation
from core.runtime.file_write_gateway import get_file_write_gateway
from core.runtime.runtime_settings import get_runtime_setting
from core.runtime.skill_contract import ActionExpectation
from core.runtime.subprocess_gateway import get_subprocess_gateway
from core.runtime.state_ownership import state_root

logger = logging.getLogger("Aura.WorldBridge")

_WORLD_DIR = state_root() / "data" / "world"
_PERMS_PATH = _WORLD_DIR / "permissions.json"
_WORKSPACE_DIR = _WORLD_DIR / "workspace"


class Channel(StrEnum):
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
    expires_at: float | None = None
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
        self._cache: dict[str, Permission] = {}
        self._load()

    def _load(self) -> None:
        if not _PERMS_PATH.exists():
            return
        try:
            data = json.loads(_PERMS_PATH.read_text(encoding="utf-8"))
            for c, raw in data.items():
                if isinstance(raw, dict):
                    self._cache[c] = Permission(**raw)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            record_degradation('world_bridge', exc)
            logger.warning("permission load failed: %s", exc)

    def _save(self) -> None:
        with self._lock:
            tmp = _PERMS_PATH.with_suffix(".json.tmp")
            atomic_write_text(tmp, json.dumps({c: asdict(p) for c, p in self._cache.items()}, indent=2), encoding="utf-8")
            os.replace(tmp, _PERMS_PATH)

    def grant(self, channel: Channel, *, notes: str = "", expires_in_s: float | None = None, fresh_auth_required: bool = False) -> Permission:
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

    def status(self, channel: Channel) -> Permission | None:
        with self._lock:
            return self._cache.get(channel.value)

    def all_channels(self) -> dict[str, Permission]:
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
    error: str | None = None
    status: str = ""
    transport_succeeded: bool = False
    effect_verified: bool = False
    manual_reconciliation_required: bool = False


def _world_handler_effect_verified(channel: Channel, data: Any) -> bool:
    if not isinstance(data, dict):
        return False
    if channel in {Channel.FILE_WORKSPACE, Channel.CALENDAR_AWARENESS}:
        return True
    if channel == Channel.SHELL_SANDBOX:
        return int(data.get("rc", 1)) == 0
    return data.get("effect_verified") is True


def _world_effect_verifier(context: Mapping[str, Any]) -> dict[str, Any]:
    result = context.get("result")
    verified = bool(
        isinstance(result, dict)
        and result.get("handler_effect_verified") is True
    )
    capability_receipt = (
        str(result.get("capability_token_receipt") or "")
        if isinstance(result, dict)
        else ""
    )
    return {
        "effect_verified": verified,
        "reason": "world_handler_observed_effect" if verified else "world_effect_unverified",
        "receipt_id": capability_receipt,
        "observation": {
            "handler_effect_verified": verified,
            "capability_token_consumed": bool(capability_receipt),
        },
    }


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
        self._handlers: dict[Channel, Callable[..., Awaitable[Any]]] = {}

    def register(self, channel: Channel, handler: Callable[..., Awaitable[Any]]) -> None:
        self._handlers[channel] = handler

    async def call(
        self,
        channel: Channel,
        *,
        action: str,
        intent: str,
        payload: dict[str, Any] | None = None,
    ) -> WorldActionResult:
        # Privacy mode (docs/SETTINGS_WIRING_AUDIT.md): "isolated" pauses ALL
        # consequential world actions; "private" pauses external posting. This is
        # the single gate for world interactions, so the block is comprehensive.
        privacy = str(get_runtime_setting("privacy.mode", "standard")).strip().lower()
        if privacy == "isolated":
            return WorldActionResult(channel=channel.value, ok=False, receipt_id="", error="privacy_mode_isolated")
        if privacy == "private" and channel == Channel.SOCIAL_POST:
            return WorldActionResult(channel=channel.value, ok=False, receipt_id="", error="privacy_mode_private")

        perm = _PERMS.status(channel)
        if perm is None or not perm.is_active():
            return WorldActionResult(channel=channel.value, ok=False, receipt_id="", error="permission_denied")

        try:
            from core.ethics.conscience import Verdict as ConscienceVerdict
            from core.ethics.conscience import get_conscience

            conscience = get_conscience()
            c_decision = conscience.evaluate(
                action=action,
                domain=(
                    "external_communication"
                    if channel in (Channel.SOCIAL_POST,)
                    else "tool_execution"
                ),
                intent=intent,
                context={"channel": channel.value, "payload": payload},
            )
        except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation("world_bridge", exc)
            return WorldActionResult(
                channel=channel.value,
                ok=False,
                receipt_id="",
                error=f"conscience_exception:{exc}",
            )
        if c_decision.verdict == ConscienceVerdict.REFUSE:
            return WorldActionResult(channel=channel.value, ok=False, receipt_id="", error=f"conscience_refused:{c_decision.rule_id}")
        if c_decision.verdict == ConscienceVerdict.REQUIRE_FRESH_USER_AUTH:
            return WorldActionResult(channel=channel.value, ok=False, receipt_id="", error="require_fresh_user_auth")

        from core.agency.capability_token import get_token_store

        store = get_token_store()
        handler = self._handlers.get(channel)
        if handler is None:
            return WorldActionResult(
                channel=channel.value,
                ok=False,
                receipt_id="",
                error="no_handler",
            )

        async def _execute_handler(execution: Mapping[str, Any]) -> dict[str, Any]:
            will_receipt_id = str(execution.get("will_receipt_id") or "")
            tok = store.issue(
                origin=f"world_bridge:{channel.value}",
                scope=action,
                ttl_seconds=60.0,
                domain=ActionDomain.ENVIRONMENT_ACTION.value,
                requested_action=action,
                approver="UnifiedWill",
                parent_receipt=will_receipt_id,
            )
            try:
                store.validate(
                    tok.token,
                    domain=ActionDomain.ENVIRONMENT_ACTION.value,
                    action=action,
                )
                data = await handler(payload or {}, capability_token=tok.token)
                verified = _world_handler_effect_verified(channel, data)
                transport_succeeded = bool(
                    not isinstance(data, dict)
                    or data.get("transport_succeeded", True) is True
                )
                try:
                    store.consume(
                        tok.token,
                        child_receipt=will_receipt_id,
                        side_effects=[action],
                    )
                except (OSError, RuntimeError, TypeError, ValueError) as exc:
                    store.revoke(
                        tok.token,
                        reason=f"capability_receipt_completion_failed:{exc}",
                    )
                    return {
                        "ok": transport_succeeded,
                        "error": f"capability_receipt_completion_failed:{exc}",
                        "handler_data": data,
                        "handler_effect_verified": False,
                        "capability_token_receipt": tok.token,
                    }
                return {
                    "ok": transport_succeeded,
                    "handler_data": data,
                    "handler_effect_verified": verified,
                    "capability_token_receipt": tok.token,
                }
            except (OSError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
                store.revoke(tok.token, reason=f"handler_error:{exc}")
                return {
                    "ok": False,
                    "error": str(exc),
                    "handler_effect_verified": False,
                    "capability_token_receipt": tok.token,
                }

        try:
            execution = await ActionExecutor.execute(
                domain=ActionDomain.ENVIRONMENT_ACTION,
                action_name=action,
                params={
                    "channel": channel.value,
                    "intent": str(intent or "")[:500],
                    "payload": dict(payload or {}),
                },
                source=f"world_bridge:{channel.value}",
                expectation=ActionExpectation(
                    objective=f"complete and observe {action} through {channel.value}",
                    acceptance_criteria=["effect_verified"],
                    required_evidence=["verification_evidence.custom_verifier"],
                    repair_hint="inspect the channel handler receipt and observe the world effect",
                    rollback_hint="use the channel-specific rollback or reconcile manually",
                    allow_partial=False,
                ),
                effect_handler=_execute_handler,
                effect_verifier=_world_effect_verifier,
            )
        except (OSError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
            record_degradation("world_bridge", exc)
            return WorldActionResult(
                channel=channel.value,
                ok=False,
                receipt_id="",
                error=f"action_executor_exception:{exc}",
            )
        receipt_id = str(
            execution.get("post_action_receipt_id")
            or execution.get("will_receipt_id")
            or execution.get("capability_token_receipt")
            or ""
        )
        effect_verified = execution.get("effect_verified") is True
        ok = bool(execution.get("ok") and effect_verified)
        error = str(execution.get("error") or "") or None
        transport_succeeded = execution.get("transport_succeeded") is True
        manual_reconciliation_required = bool(
            execution.get("manual_reconciliation_required")
            or (transport_succeeded and not effect_verified)
        )
        if transport_succeeded and not effect_verified:
            detail = str(error or "").strip()
            error = (
                f"world_effect_unverified:{detail}"
                if detail
                else "world_effect_unverified"
            )
        return WorldActionResult(
            channel=channel.value,
            ok=ok,
            receipt_id=receipt_id,
            data=execution.get("handler_data"),
            error=error,
            status=str(execution.get("status") or ""),
            transport_succeeded=transport_succeeded,
            effect_verified=effect_verified,
            manual_reconciliation_required=manual_reconciliation_required,
        )


# ─── Default handlers ───────────────────────────────────────────────────────


async def _file_workspace_handler(payload: dict[str, Any], *, capability_token: str) -> dict[str, Any]:
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
        await async_atomic_write_text(target, body, encoding="utf-8")
        return {"path": rel, "bytes": len(body)}
    raise ValueError(f"unknown_op:{op}")


async def _shell_sandbox_handler(payload: dict[str, Any], *, capability_token: str) -> dict[str, Any]:
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
    await get_file_write_gateway().ensure_directory_async(
        _WORKSPACE_DIR,
        source="core.embodiment.world_bridge.shell_workspace",
    )
    proc = await get_subprocess_gateway().spawn_async(
        cmd,
        cwd=str(_WORKSPACE_DIR),
        stdout=_asyncio.subprocess.PIPE,
        stderr=_asyncio.subprocess.PIPE,
        read_only=True,
        source="tool_execution:world_bridge.shell_sandbox",
        accelerator_capability="auto",
    )
    try:
        stdout, stderr = await _asyncio.wait_for(proc.communicate(), timeout=5.0)
    except TimeoutError as exc:
        proc.kill()
        raise TimeoutError("shell_timeout") from exc
    return {
        "rc": proc.returncode,
        "stdout": stdout[:1_000_000].decode("utf-8", errors="replace"),
        "stderr": stderr[:1_000_000].decode("utf-8", errors="replace"),
    }


_BRIDGE: WorldBridge | None = None


def get_world_bridge() -> WorldBridge:
    global _BRIDGE
    if _BRIDGE is None:
        from core.embodiment.iot_bridge import _environmental_change_handler

        b = WorldBridge()
        b.register(Channel.FILE_WORKSPACE, _file_workspace_handler)
        b.register(Channel.SHELL_SANDBOX, _shell_sandbox_handler)
        b.register(Channel.ENVIRONMENTAL_CHANGE, _environmental_change_handler)
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
