"""Self-repair backlog — route detected test defects to governed repair goals.

Closes the mandate's "code catches and fixes regressions before users do":
the chunk runner detects order-dependence and real failures and (with
``--defect-register``) writes a machine-readable register. This module turns
that register into concrete repair goals for the autonomous repair lane — so
Aura can burn down her own defect backlog behind mutation-tier, sandbox,
rollback, and receipt gates.

Design guarantees:
- **Safe auto-executes by default.** Repair goals enter the autonomous repair
  executor; high-risk paths are still quarantined/proposal-only by downstream
  mutation-tier policy.
- **Idempotent.** Each defect has a stable id; re-ingesting the same register
  never double-creates a goal.
- **Read-only ingestion.** Parsing a register mutates nothing until a plan is
  explicitly created.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.governance_context import local_internal_governed_scope
from core.runtime.errors import record_degradation
from core.runtime.file_write_gateway import get_file_write_gateway
from core.runtime.state_ownership import state_root

logger = logging.getLogger("Aura.SelfRepairBacklog")

_SEEN_SCHEMA = "aura.self_repair_seen.v1"
_REGISTER_SCHEMA = "aura.test_defect_register.v1"
_AUTO_ACCEPTED_STATUSES = frozenset({"scheduled", "cooldown"})


@dataclass(frozen=True)
class RepairItem:
    defect_id: str
    kind: str          # "order_dependence" | "real_failure"
    target: str        # the failing test id
    goal: str          # the repair goal text for the task engine

    def to_dict(self) -> dict[str, str]:
        return {
            "defect_id": self.defect_id,
            "kind": self.kind,
            "target": self.target,
            "goal": self.goal,
        }


def _defect_id(kind: str, target: str) -> str:
    return kind[:4] + "-" + hashlib.sha256(f"{kind}|{target}".encode()).hexdigest()[:12]


_ORDER_DEP_GOAL = (
    "Diagnose and fix the ORDER-DEPENDENCE defect in test '{target}': it "
    "fails inside a chunk but passes in isolation, which means shared process "
    "state (module globals, singletons, the ServiceContainer, monkeypatched "
    "module attributes, or _PROCESS_STARTED_AT-style clocks) leaks across "
    "tests. Find the leaking state, make the test hermetic (fixture-scoped "
    "reset), and prove it by running the test both alone and after the "
    "polluting sibling. Do not weaken the assertion."
)

_REAL_FAILURE_GOAL = (
    "Diagnose and fix the FAILING test '{target}' (fails both in-chunk and in "
    "isolation). Root-cause it, fix the underlying defect (not the test unless "
    "the test is provably wrong), and verify green. Do not silence it."
)


# Statuses that positively mean the repair goal was created. Anything
# outside this set — including new statuses nobody has classified yet — is
# NOT acceptance, so an unrecognised result leaves the defect in the backlog
# instead of silently retiring it (CP126 52feb1d1).
_ACCEPTING_STATUSES = frozenset({
    "created",
    "accepted",
    "queued",
    "scheduled",
    "succeeded",
    "success",
    "ok",
    # A shadow plan that exists and is awaiting approval WAS created — that
    # is this subsystem's designed success path (see
    # test_enqueue_creates_approval_gated_shadow_plans), not a defect. The
    # finding listed these as suspect; the design says otherwise, and the
    # tests pin it. What the fix actually removes is acceptance-by-default
    # for statuses nobody has classified.
    "planned",
    "waiting_for_approval",
    "awaiting_approval",
    "pending_approval",
})


@dataclass
class SelfRepairBacklog:
    """Parses defect registers into approval-gated repair goals."""

    seen_path: Path = field(
        default_factory=lambda: state_root() / "data" / "self_repair_seen.json"
    )
    _seen: set[str] = field(default_factory=set)
    _inflight: set[str] = field(default_factory=set, init=False, repr=False)
    _seen_lock: threading.RLock = field(
        default_factory=threading.RLock,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        try:
            if self.seen_path.exists():
                self._seen = self._decode_seen_state(
                    json.loads(self.seen_path.read_text(encoding="utf-8"))
                )
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            record_degradation("self_repair_backlog", exc, severity="debug")
            self._seen = set()

    @staticmethod
    def _decode_seen_state(payload: object) -> set[str]:
        if isinstance(payload, list):
            values = payload
        elif isinstance(payload, dict) and payload.get("schema") == _SEEN_SCHEMA:
            values = payload.get("defect_ids")
            if not isinstance(values, list):
                raise ValueError("self-repair seen ledger defect_ids must be a list")
        else:
            raise ValueError("unsupported self-repair seen ledger schema")
        if any(not isinstance(value, str) or not value for value in values):
            raise ValueError("self-repair seen ledger contains an invalid defect id")
        return set(values)

    def parse_register(self, register_path: str | Path) -> list[RepairItem]:
        """Read a defect register into RepairItems (pure — mutates nothing)."""
        try:
            data = json.loads(Path(register_path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            record_degradation("self_repair_backlog", exc, severity="warning")
            return []
        if not isinstance(data, dict):
            return []
        schema = data.get("schema")
        if schema not in {None, _REGISTER_SCHEMA}:
            record_degradation(
                "self_repair_backlog",
                ValueError(f"unsupported defect register schema: {schema}"),
                severity="warning",
            )
            return []

        def _targets(field_name: str) -> list[str]:
            raw_values = data.get(field_name, [])
            if raw_values is None:
                return []
            if not isinstance(raw_values, list):
                record_degradation(
                    "self_repair_backlog",
                    TypeError(f"defect register {field_name} must be a list"),
                    severity="warning",
                )
                return []
            targets: list[str] = []
            seen_targets: set[str] = set()
            for raw_target in raw_values:
                if not isinstance(raw_target, str) or not raw_target.strip():
                    continue
                target = raw_target.strip()
                if target not in seen_targets:
                    seen_targets.add(target)
                    targets.append(target)
            return targets

        items: list[RepairItem] = []
        for target in _targets("order_dependent"):
            items.append(
                RepairItem(
                    _defect_id("order_dependence", target),
                    "order_dependence",
                    target,
                    _ORDER_DEP_GOAL.format(target=target),
                )
            )
        for target in _targets("real_failures"):
            items.append(
                RepairItem(
                    _defect_id("real_failure", target),
                    "real_failure",
                    target,
                    _REAL_FAILURE_GOAL.format(target=target),
                )
            )
        return items

    def new_items(self, register_path: str | Path) -> list[RepairItem]:
        """RepairItems not already ingested (idempotent by defect_id)."""
        with self._seen_lock:
            unavailable = self._seen | self._inflight
        return [
            item
            for item in self.parse_register(register_path)
            if item.defect_id not in unavailable
        ]

    def _claim_new_items(
        self,
        register_path: str | Path,
        *,
        max_items: int,
    ) -> list[RepairItem]:
        parsed = self.parse_register(register_path)
        claimed: list[RepairItem] = []
        with self._seen_lock:
            unavailable = self._seen | self._inflight
            for item in parsed:
                if item.defect_id in unavailable:
                    continue
                claimed.append(item)
                unavailable.add(item.defect_id)
                if len(claimed) >= max_items:
                    break
            self._inflight.update(item.defect_id for item in claimed)
        return claimed

    def _release_claims(self, items: list[RepairItem]) -> None:
        with self._seen_lock:
            self._inflight.difference_update(item.defect_id for item in items)

    def _mark_seen(self, items: list[RepairItem]) -> bool:
        with self._seen_lock:
            candidate = self._seen | {item.defect_id for item in items}
            if candidate == self._seen:
                return True
            payload = {
                "schema": _SEEN_SCHEMA,
                "updated_at": time.time(),
                "defect_ids": sorted(candidate),
            }
            try:
                gateway = get_file_write_gateway()
                with local_internal_governed_scope(
                    "self_repair_backlog.mark_seen",
                    domain="file_write",
                ):
                    gateway.ensure_directory(
                        self.seen_path.parent,
                        source="core.agency.self_repair_backlog.mark_seen",
                    )
                    gateway.write_text(
                        self.seen_path,
                        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
                        source="core.agency.self_repair_backlog.mark_seen",
                    )
                persisted = self._decode_seen_state(
                    json.loads(self.seen_path.read_text(encoding="utf-8"))
                )
                if persisted != candidate:
                    raise RuntimeError("self-repair seen ledger verification mismatch")
            except (OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError) as exc:
                record_degradation("self_repair_backlog", exc, severity="warning")
                return False
            self._seen = candidate
            return True

    @staticmethod
    def _annotate_persistence(
        results: list[dict[str, Any]],
        accepted_ids: set[str],
        persisted: bool,
    ) -> None:
        for result in results:
            if result["defect_id"] in accepted_ids:
                result["seen_persisted"] = persisted

    async def enqueue_repairs(
        self,
        register_path: str | Path,
        *,
        task_engine: Any = None,
        max_items: int = 5,
        dry_run: bool = False,
        auto_execute: bool = True,
    ) -> list[dict[str, Any]]:
        """Create repair goals for new defects.

        Returns a summary per item. ``dry_run`` and unavailable execution lanes
        never acknowledge defects. With ``auto_execute=True`` the item is
        scheduled through the autonomous repair executor; otherwise the old
        approval-gated shadow plan behavior is used. A defect is acknowledged
        only after an execution lane accepts it and the seen ledger is durable.
        """
        limit = max(0, int(max_items))
        if limit == 0:
            return []
        items = self._claim_new_items(register_path, max_items=limit)
        if not items:
            return []
        try:
            return await self._enqueue_claimed_repairs(
                items,
                task_engine=task_engine,
                dry_run=dry_run,
                auto_execute=auto_execute,
            )
        finally:
            self._release_claims(items)

    async def _enqueue_claimed_repairs(
        self,
        items: list[RepairItem],
        *,
        task_engine: Any,
        dry_run: bool,
        auto_execute: bool,
    ) -> list[dict[str, Any]]:
        """Route items already reserved by ``enqueue_repairs``."""

        if auto_execute and not dry_run:
            results: list[dict[str, Any]] = []
            try:
                from core.resilience.autonomous_repair_executor import (
                    AutonomousRepairRequest,
                    get_autonomous_repair_executor,
                )

                executor = get_autonomous_repair_executor()
                accepted: list[RepairItem] = []
                for item in items:
                    request = AutonomousRepairRequest(
                        subsystem="self_repair_backlog",
                        error_type=item.kind,
                        error_message=item.target,
                        severity="degraded",
                        goal=item.goal,
                        context={
                            "origin": "self_repair_backlog",
                            "defect_id": item.defect_id,
                            "defect_kind": item.kind,
                            "target": item.target,
                        },
                    )
                    decision = executor.enqueue_background(request)
                    status = str(decision.get("status", "unknown") or "unknown").lower()
                    is_accepted = status in _AUTO_ACCEPTED_STATUSES
                    if is_accepted:
                        accepted.append(item)
                    results.append(
                        {
                            **item.to_dict(),
                            "created": status == "scheduled",
                            "accepted": is_accepted,
                            "reason": status,
                            "fingerprint": decision.get("fingerprint", ""),
                            "auto_execute": True,
                        }
                    )
                accepted_ids = {item.defect_id for item in accepted}
                persisted = (
                    await asyncio.to_thread(self._mark_seen, accepted)
                    if accepted
                    else True
                )
                self._annotate_persistence(results, accepted_ids, persisted)
                logger.info(
                    "[SELF-REPAIR] Accepted %d/%d defect(s) for autonomous repair "
                    "(seen_persisted=%s).",
                    len(accepted),
                    len(results),
                    persisted,
                )
                return results
            except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
                record_degradation("self_repair_backlog", exc, severity="warning")

        if dry_run:
            return [
                {**item.to_dict(), "created": False, "accepted": False, "reason": "dry_run"}
                for item in items
            ]

        if task_engine is None:
            try:
                from core.agency.autonomous_task_engine import get_task_engine

                task_engine = get_task_engine()
            except (ImportError, RuntimeError, AttributeError) as exc:
                record_degradation("self_repair_backlog", exc, severity="warning")
                task_engine = None
        if task_engine is None:
            return [
                {
                    **item.to_dict(),
                    "created": False,
                    "accepted": False,
                    "reason": "no_task_engine",
                }
                for item in items
            ]

        results: list[dict[str, Any]] = []
        accepted: list[RepairItem] = []
        for item in items:
            created = False
            accepted_item = False
            reason = "unknown"
            try:
                result = await task_engine.execute_goal(
                    item.goal,
                    context={
                        "origin": "self_repair_backlog",
                        "defect_id": item.defect_id,
                        "defect_kind": item.kind,
                        "target": item.target,
                        "requires_approval": True,
                    },
                    is_shadow=True,
                )
                reason = str(getattr(result, "status", "planned") or "planned")
                succeeded = getattr(result, "succeeded", None)
                # CP126 52feb1d1. With `succeeded` absent, acceptance was
                # "any status not in a short denylist" — so unknown,
                # planned, waiting, or an unexpected object repr all counted
                # as accepted, were marked created, and were persisted as
                # SEEN. A defect recorded as handled is a defect nobody
                # looks at again, so a permissive default here quietly
                # retires real work.
                #
                # Acceptance now needs a positive statement: either
                # succeeded is truthy, or the status is one this code
                # recognises as success. Everything else, including
                # anything new, is not accepted and stays in the backlog.
                if succeeded is not None:
                    accepted_item = bool(succeeded)
                else:
                    accepted_item = reason.strip().lower() in _ACCEPTING_STATUSES
                created = accepted_item
                if accepted_item:
                    accepted.append(item)
            except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                record_degradation("self_repair_backlog", exc, severity="warning")
                reason = f"error:{type(exc).__name__}"
            results.append(
                {
                    **item.to_dict(),
                    "created": created,
                    "accepted": accepted_item,
                    "reason": reason,
                }
            )

        accepted_ids = {item.defect_id for item in accepted}
        persisted = (
            await asyncio.to_thread(self._mark_seen, accepted) if accepted else True
        )
        self._annotate_persistence(results, accepted_ids, persisted)
        logger.info(
            "[SELF-REPAIR] Accepted %d/%d defect(s) into repair goals "
            "(seen_persisted=%s).",
            len(accepted),
            len(results),
            persisted,
        )
        return results


_BACKLOG: SelfRepairBacklog | None = None


def get_self_repair_backlog() -> SelfRepairBacklog:
    global _BACKLOG
    if _BACKLOG is None:
        _BACKLOG = SelfRepairBacklog()
    return _BACKLOG
