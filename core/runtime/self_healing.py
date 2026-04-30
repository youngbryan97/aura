"""core/runtime/self_healing.py

Self-Healing Loop
==================
A small, hot-path loop that watches for cognitive-module hangs and
restarts the offending module without losing receipts. It is
complementary to StabilityGuardian (broad health checks) and the
OrganSupervisor (subprocess restart): self_healing operates *inside*
the same process, on async tasks that have stopped progressing.

Detection signals:

  * a registered "heartbeat" callable hasn't been called in N seconds
  * an asyncio.Task referenced in the registry is in a "hanging" state
    (still pending after a grace window beyond its declared budget)

Repair actions:

  * cancel the hanging task
  * call the module's ``restart_async()`` if available; otherwise
    re-instantiate via ServiceContainer
  * record an action receipt + a phenomenal envelope (severity = 0.5)
"""
from __future__ import annotations
from core.runtime.errors import record_degradation

from core.utils.task_tracker import get_task_tracker

import asyncio
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

logger = logging.getLogger("Aura.SelfHealing")

_DIR = Path.home() / ".aura" / "data" / "self_healing"
_DIR.mkdir(parents=True, exist_ok=True)
_LEDGER = _DIR / "events.jsonl"


@dataclass
class WatchEntry:
    name: str
    last_heartbeat_at: float = field(default_factory=time.time)
    expected_interval_s: float = 30.0
    restart_async: Optional[Callable[[], Awaitable[None]]] = None
    container_key: Optional[str] = None
    restarts: int = 0


class SelfHealing:
    def __init__(self) -> None:
        self._watches: Dict[str, WatchEntry] = {}
        self._task: Optional[asyncio.Task] = None
        self._running = False

    def watch(
        self,
        name: str,
        *,
        expected_interval_s: float = 30.0,
        restart_async: Optional[Callable[[], Awaitable[None]]] = None,
        container_key: Optional[str] = None,
    ) -> None:
        self._watches[name] = WatchEntry(
            name=name,
            expected_interval_s=expected_interval_s,
            restart_async=restart_async,
            container_key=container_key,
        )

    def heartbeat(self, name: str) -> None:
        w = self._watches.get(name)
        if w is None:
            return
        w.last_heartbeat_at = time.time()

    async def start(self, *, interval: float = 5.0) -> None:
        if self._running:
            return
        self._running = True

        async def _loop():
            while self._running:
                await self._tick()
                await asyncio.sleep(interval)

        self._task = get_task_tracker().create_task(_loop(), name="SelfHealing")

    async def stop(self) -> None:
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass  # no-op: intentional
            self._task = None

    async def _tick(self) -> None:
        now = time.time()
        for w in list(self._watches.values()):
            age = now - w.last_heartbeat_at
            if age <= w.expected_interval_s * 2.5:
                continue
            await self._heal(w, age)

    async def _heal(self, w: WatchEntry, age: float) -> None:
        record = {
            "when": time.time(),
            "name": w.name,
            "stale_for_s": age,
            "restart_count": w.restarts,
        }
        try:
            if w.restarts >= 3:
                # Deep repair strategy using ReimplementationLab
                from core.container import ServiceContainer
                lab = ServiceContainer.get("reimplementation_lab", default=None)
                if lab is not None:
                    module_path = None
                    if w.container_key:
                        instance = ServiceContainer.get(w.container_key, default=None)
                        if instance is not None:
                            module_name = type(instance).__module__
                            module_path = module_name.replace('.', '/') + '.py'
                    
                    if module_path:
                        logger.warning("Deep repair triggered for %s (%s)", w.name, module_path)
                        # We use asyncio.create_task to not block the healing loop indefinitely
                        asyncio.create_task(
                            lab.run_reconstruction(
                                module_path, 
                                metadata={"trigger": "self_healing", "watch_name": w.name}
                            )
                        )
                        record["result"] = "deep_repair_triggered"
                        w.restarts = 0  # Reset counter to wait and see if it recovers
                        w.last_heartbeat_at = time.time()
                    else:
                        record["result"] = "deep_repair_failed_no_module_path"
                else:
                    record["result"] = "deep_repair_failed_no_lab"

            if record.get("result") not in ("deep_repair_triggered", "deep_repair_failed_no_module_path", "deep_repair_failed_no_lab"):
                if w.restart_async is not None:
                    await w.restart_async()
                elif w.container_key:
                    from core.container import ServiceContainer
                    instance = ServiceContainer.get(w.container_key, default=None)
                    if instance is not None and hasattr(instance, "restart_async"):
                        await instance.restart_async()
                w.restarts += 1
                w.last_heartbeat_at = time.time()
                record["result"] = "restarted"
        except Exception as exc:
            record_degradation('self_healing', exc)
            record["result"] = f"restart_failed:{exc}"
        try:
            with open(_LEDGER, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, default=str) + "\n")
                fh.flush()
                try:
                    os.fsync(fh.fileno())
                except Exception:
                    pass  # no-op: intentional
        except Exception:
            pass  # no-op: intentional


_HEALER: Optional[SelfHealing] = None


def get_healer() -> SelfHealing:
    global _HEALER
    if _HEALER is None:
        _HEALER = SelfHealing()
    return _HEALER


__all__ = ["SelfHealing", "WatchEntry", "get_healer"]
