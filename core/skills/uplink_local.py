"""Local persistence verification — a real probe, not a status string.

Historically this skill returned "Persistence verified locally" without
checking anything. It now earns that sentence: it inspects the live state
repository's runtime health and performs a governed write→read→delete
round-trip on the data directory, and only reports verified when both hold.
"""

import os
import time
from typing import Any

from core.skills.what_every_skill_gives_back import THE_SHARED_RESULT
from core.container import ServiceContainer
from core.runtime.errors import record_degradation
from core.skills.base_skill import BaseSkill

# last_commit_at older than this (with commits expected) counts as stale.
_STALE_COMMIT_SECONDS = 15 * 60


class UplinkSkill(BaseSkill):
    #: What a caller gets back. The shared part only: every skill
    #: here returns `ok`, and a schema claiming to be complete
    #: would be wrong for every one that adds a field.
    result_schema = THE_SHARED_RESULT

    name = "uplink_local"
    retry_safe = False  # external send/act — never double-fire on retry
    description = (
        "Verify local persistence for real: state-repository health (DB connected, "
        "consumer alive, commits flowing) plus a governed write-read-delete "
        "round-trip on the data directory. Reports exactly what was checked."
    )
    inputs = {"goal": "objective"}
    output = "Evidence-backed persistence verdict"

    def match(self, goal: dict[str, Any]) -> bool:
        return "Uplink" in goal.get("objective", "") or "Persistence" in goal.get("objective", "")

    async def execute(self, params: Any, context: dict[str, Any]) -> dict[str, Any]:
        checks: dict[str, Any] = {}

        state_ok = self._check_state_repository(checks)
        disk_ok = await self._check_disk_round_trip(checks)

        verified = state_ok and disk_ok
        failed = [name for name, check in checks.items() if not check.get("ok")]
        summary = (
            "Persistence verified: state repository healthy and disk round-trip succeeded."
            if verified
            else f"Persistence NOT verified — failing checks: {', '.join(failed)}."
        )
        return {
            "ok": verified,
            "status": "verified" if verified else "failed",
            "checks": checks,
            "summary": summary,
        }

    @staticmethod
    def _check_state_repository(checks: dict[str, Any]) -> bool:
        repo = ServiceContainer.get("state_repository", default=None)
        if repo is None or not hasattr(repo, "get_runtime_status"):
            checks["state_repository"] = {
                "ok": False,
                "evidence": "state_repository service unavailable",
            }
            return False
        try:
            status = repo.get_runtime_status()
        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
            record_degradation("uplink_local", exc, action="reported unhealthy state repository in persistence probe")
            checks["state_repository"] = {"ok": False, "evidence": f"status query failed: {exc}"}
            return False

        problems = []
        if not status.get("state_available"):
            problems.append("state unavailable")
        if not status.get("consumer_alive"):
            problems.append("mutation consumer dead")
        if status.get("is_vault_owner") and not status.get("db_connected"):
            problems.append("db not connected")
        dropped = int(status.get("dropped_commit_count", 0) or 0)
        if dropped > 0:
            problems.append(f"{dropped} dropped commits")
        last_commit_at = float(status.get("last_commit_at", 0.0) or 0.0)
        commit_age = time.time() - last_commit_at if last_commit_at > 0 else None
        if commit_age is not None and commit_age > _STALE_COMMIT_SECONDS:
            problems.append(f"last commit {commit_age / 60:.0f}m ago")

        checks["state_repository"] = {
            "ok": not problems,
            "evidence": (
                "healthy"
                if not problems
                else "; ".join(problems)
            ),
            "current_version": status.get("current_version"),
            "queue_depth": status.get("queue_depth"),
            "last_commit_age_seconds": round(commit_age, 1) if commit_age is not None else None,
        }
        return not problems

    @staticmethod
    async def _check_disk_round_trip(checks: dict[str, Any]) -> bool:
        try:
            from core.config import config
            from core.runtime.file_write_gateway import get_file_write_gateway

            nonce = f"aura-persistence-probe {time.time_ns()} pid={os.getpid()}"
            probe_path = config.paths.data_dir / ".persistence_probe"
            gateway = get_file_write_gateway()
            await gateway.write_text_async(
                probe_path, nonce, source="uplink_local.persistence_probe"
            )
            import asyncio

            read_back = await asyncio.to_thread(
                probe_path.read_text, "utf-8"
            )
            await gateway.delete_path_async(
                probe_path, source="uplink_local.persistence_probe"
            )
            match = read_back == nonce
            checks["disk_round_trip"] = {
                "ok": match,
                "evidence": (
                    f"wrote+read+deleted {probe_path.name} ({len(nonce)} bytes)"
                    if match
                    else "read-back content did not match written nonce"
                ),
            }
            return match
        except (OSError, RuntimeError, ImportError, AttributeError, ValueError, TypeError) as exc:
            record_degradation("uplink_local", exc, action="reported failed disk round-trip in persistence probe")
            checks["disk_round_trip"] = {"ok": False, "evidence": f"round-trip failed: {exc}"}
            return False
