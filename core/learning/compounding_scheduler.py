"""core/learning/compounding_scheduler.py — autonomous trigger for weight compounding.

This is what makes weight-level learning ACTIVE instead of installed. Before
this scheduler existed, Aura had two dormant trainers: LiveLearner's autorun
defaulted off, and genuine_learning_pipeline's ``run_if_ready`` had no caller.
Data was harvested forever; weights never moved on their own.

The scheduler is the single canonical trigger (the existing collectors keep
collecting — this is the one executor). Every wake it checks, in order:

  1. kill switch        — AURA_WEIGHT_COMPOUNDING=0 disables cleanly
  2. cooldown           — at most one cycle per AURA_COMPOUND_COOLDOWN_S
  3. maintenance idle   — the MAINTENANCE background profile (30 min idle,
                          memory + failure-pressure + conversation-ready
                          gates); training must never contend with the user
  4. data readiness     — cheap row counts before any heavy work
  5. Will approval      — weight mutation is a governed act, same contract as
                          every other state mutation

then runs one WeightCompoundingLoop cycle off-loop. Admission control inside
the cycle re-verifies RAM headroom at the moment of truth (the maintenance
gate means the cortex worker is typically unloaded — that's what makes a 32B
training pass safe on a 64GB host: she trains while she sleeps, she does not
fight herself for memory). A successful fuse is recorded as a qualification
candidate. This scheduler never hot-swaps a whole cortex or changes its boot
pointer.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from core.runtime.atomic_writer import atomic_write_text
from core.runtime.errors import FallbackClassification, record_degradation

logger = logging.getLogger("Aura.CompoundingScheduler")

_RECOVERABLE = (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError)

SERVICE_NAME = "weight_compounding"


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


class CompoundingScheduler:
    """Periodic, governed, idle-gated executor of weight-compounding cycles."""

    def __init__(self, orchestrator: Any = None) -> None:
        self._orchestrator = orchestrator
        self._task: asyncio.Task | None = None
        self._active = False
        self._running_cycle = False
        self._last_receipt: dict[str, Any] | None = None
        self.check_interval_s = float(_env_int("AURA_COMPOUND_CHECK_INTERVAL_S", 900))
        self.cooldown_s = float(_env_int("AURA_COMPOUND_COOLDOWN_S", 6 * 3600))
        self.autonomous_max_gb = _env_int("AURA_COMPOUND_AUTONOMOUS_MAX_GB", 24)

    # ── lifecycle ────────────────────────────────────────────────────────────

    async def start(self) -> None:
        if not _env_flag("AURA_WEIGHT_COMPOUNDING", True):
            logger.info("Weight compounding disabled by AURA_WEIGHT_COMPOUNDING=0.")
            return
        self._active = True
        from core.utils.task_tracker import get_task_tracker

        self._task = get_task_tracker().create_task(
            self._run(), name="weight_compounding.scheduler"
        )
        logger.info(
            "Weight-compounding scheduler online (check every %.0fs, cooldown %.0fs, "
            "autonomous cap %dGB).",
            self.check_interval_s, self.cooldown_s, self.autonomous_max_gb,
        )

    async def stop(self) -> None:
        self._active = False
        if self._task is not None:
            self._task.cancel()
            try:
                await asyncio.wait_for(self._task, timeout=5.0)
            except (asyncio.CancelledError, TimeoutError):
                pass
            self._task = None

    # ── main loop ────────────────────────────────────────────────────────────

    async def _run(self) -> None:
        # Late first check: never compete with boot warmup.
        await asyncio.sleep(min(self.check_interval_s, 600.0))
        while self._active:
            try:
                await self._maybe_cycle()
            except asyncio.CancelledError:
                raise
            except _RECOVERABLE as exc:
                record_degradation(
                    "weight_compounding_scheduler",
                    exc,
                    action="skipped this compounding check; next check unaffected",
                    classification=FallbackClassification.SAFE_FALLBACK,
                )
            await asyncio.sleep(self.check_interval_s)

    async def _maybe_cycle(self) -> None:
        if self._running_cycle:
            return
        if not _env_flag("AURA_WEIGHT_COMPOUNDING", True):
            return

        state = self._load_state()
        since_last = time.time() - float(state.get("last_attempt_at", 0.0))
        if since_last < self.cooldown_s:
            return

        from core.runtime.background_policy import (
            MAINTENANCE_BACKGROUND_POLICY,
            background_activity_allowed,
        )

        if not background_activity_allowed(
            self._orchestrator, profile=MAINTENANCE_BACKGROUND_POLICY
        ):
            return

        await self._execute_cycle(reason="scheduled_idle")
        await self._maybe_train_specialist()

    async def run_cycle_now(self, *, reason: str = "manual") -> dict[str, Any]:
        """On-demand governed cycle — the seam other subsystems call.

        The RSI loop's weight-update action and operator asks land here so
        every weight mutation goes through ONE path: same Will approval, same
        admission control inside the loop, same receipts. Bypasses the idle
        gate and cooldown (the caller decided the moment) but never the
        single-flight guard.
        """
        if self._running_cycle:
            return {"status": "blocked", "reasons": ["cycle_already_running"]}
        if not _env_flag("AURA_WEIGHT_COMPOUNDING", True):
            return {"status": "blocked", "reasons": ["disabled_by_env"]}
        return await self._execute_cycle(reason=reason)

    async def _execute_cycle(self, *, reason: str) -> dict[str, Any]:
        """The one governed execution core behind both triggers."""
        state = self._load_state()
        loop = self._build_loop()
        readiness = loop.data_readiness()
        if not readiness.get("ready"):
            logger.debug("Compounding data not ready (%s): %s", reason, readiness)
            return {"status": "blocked", "reasons": ["data_not_ready"], "readiness": readiness}

        approved, will_reason = self._will_approval(
            {"operation": "weight_compounding_cycle", "trigger": reason, "readiness": readiness}
        )
        if not approved:
            logger.info("Will declined compounding cycle (%s): %s", reason, will_reason)
            self._save_state(
                {"last_attempt_at": time.time(), "last_status": f"will_denied:{will_reason}"}
            )
            return {"status": "blocked", "reasons": [f"approval_denied:{will_reason}"]}

        self._running_cycle = True
        try:
            logger.info(
                "🧬 Weight-compounding cycle starting (trigger=%s, readiness=%s).",
                reason, readiness,
            )
            from core.governance_context import local_internal_governed_scope

            with local_internal_governed_scope(
                "weight_compounding.cycle",
                domain="memory_write",
                constraints={"artifact": "model_weights", "governed_by": "will+heldout_gate"},
            ):
                receipt = await asyncio.to_thread(loop.run_cycle)
            self._last_receipt = receipt.to_dict()
            self._save_state(
                {
                    "last_attempt_at": time.time(),
                    "last_status": receipt.status,
                    "last_generation_id": receipt.generation_id,
                    "last_trigger": reason,
                    "last_candidate_at": (
                        time.time() if receipt.status == "candidate"
                        else state.get("last_candidate_at", 0.0)
                    ),
                }
            )
            logger.info(
                "🧬 Compounding cycle %s: %s %s",
                receipt.generation_id, receipt.status, receipt.reasons or "",
            )
            return self._last_receipt
        finally:
            self._running_cycle = False

    async def _maybe_train_specialist(self) -> None:
        """Train ONE domain specialist per idle window, when supply exists.

        Specialists are adapter-only artifacts for the expert-LoRA library —
        the modular-weights half of the learning stack (the general cycle
        above owns the fused lineage). Default-off until validated live;
        supply-gated (min pairs per domain), idle-gated by the same window
        that admitted the general cycle, and every outcome lands in a
        receipt under data/learning/specialists/.
        """
        if not _env_flag("AURA_DOMAIN_SPECIALISTS", False):
            return
        try:
            from core.config import get_config
            from core.learning.domain_specialists import (
                DomainSpecialistTrainer,
                SpecialistConfig,
            )

            data_dir = Path(get_config().paths.data_dir)
            trainer = DomainSpecialistTrainer(
                SpecialistConfig(
                    work_root=data_dir / "learning" / "specialists",
                    store_path=data_dir / "verifiable_preferences.jsonl",
                )
            )
            eligible = trainer.eligible_domains()
            if not eligible:
                return
            state = self._load_state()
            trained: dict[str, float] = dict(state.get("specialist_trained_at", {}) or {})
            # The Practice Director picks the highest-NEED eligible domain
            # (failure-directed, receipts-ranked); least-recently-trained is
            # the fallback when direction is absent, off, or evidence-free.
            # Resolved from the service spine only — never self-created, so
            # hermetic tests without a registered director keep pure LRT.
            domain = None
            chosen_by = "least_recently_trained"
            try:
                from core.runtime.service_access import resolve_practice_director

                director = resolve_practice_director(default=None)
                if director is not None:
                    await asyncio.to_thread(director.harvest)
                    domain = director.choose_focus_domain(eligible)
                    if domain is not None:
                        chosen_by = "practice_director"
            except _RECOVERABLE as exc:
                record_degradation(
                    "compounding_scheduler",
                    exc,
                    action="fell back to least-recently-trained specialist choice",
                    severity="debug",
                )
            if domain is None:
                domain = min(eligible, key=lambda d: float(trained.get(d, 0.0)))
            logger.info(
                "🧩 Domain-specialist cycle starting for '%s' (chosen by %s).",
                domain, chosen_by,
            )
            from core.governance_context import local_internal_governed_scope

            with local_internal_governed_scope(
                "domain_specialists.cycle",
                domain="memory_write",
                constraints={"artifact": "domain_adapter", "governed_by": "domain+general_gate"},
            ):
                receipt = await asyncio.to_thread(trainer.train_domain, domain)
            trained[domain] = time.time()
            self._save_state(
                {
                    "specialist_trained_at": trained,
                    "last_specialist_status": f"{domain}:{receipt.status}",
                }
            )
            logger.info(
                "🧩 Specialist '%s': %s %s",
                domain, receipt.status, receipt.reasons or "",
            )
        except _RECOVERABLE as exc:
            record_degradation(
                "weight_compounding_scheduler",
                exc,
                action="skipped domain-specialist cycle after failure; next window retries",
            )

    # ── collaborators ────────────────────────────────────────────────────────

    def _build_loop(self):
        from core.learning.weight_compounding import (
            WeightCompoundingLoop,
            default_config,
        )
        from core.runtime.background_policy import (
            MAINTENANCE_BACKGROUND_POLICY,
            background_activity_allowed,
        )

        config = default_config(
            autonomous_max_model_bytes=self.autonomous_max_gb * 1024**3,
        )
        return WeightCompoundingLoop(
            config,
            idle_hook=lambda: background_activity_allowed(
                self._orchestrator, profile=MAINTENANCE_BACKGROUND_POLICY
            ),
        )

    def _will_approval(self, context: dict[str, Any]) -> tuple[bool, str]:
        """Weight mutation is governed: no Will, no training. Fail closed."""
        try:
            from core.will import ActionDomain, get_will

            decision = get_will().decide(
                content=f"weight_compounding_cycle:{json.dumps(context.get('readiness', {}), sort_keys=True)}",
                source="compounding_scheduler",
                domain=ActionDomain.STATE_MUTATION,
                priority=0.8,
                context=context,
            )
            if decision.is_approved():
                return True, str(getattr(decision, "receipt_id", "approved"))
            return False, str(getattr(decision, "reason", "denied"))
        except _RECOVERABLE as exc:
            record_degradation(
                "weight_compounding_scheduler",
                exc,
                action="blocked compounding cycle because Will approval was unavailable",
            )
            return False, f"will_unavailable:{type(exc).__name__}"

    # ── state + status ───────────────────────────────────────────────────────

    def _state_path(self) -> Path:
        from core.config import get_config

        root = Path(get_config().paths.data_dir) / "learning" / "compounding"
        root.mkdir(parents=True, exist_ok=True)
        return root / "scheduler_state.json"

    def _load_state(self) -> dict[str, Any]:
        try:
            path = self._state_path()
            if path.exists():
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return data
        except _RECOVERABLE as exc:
            record_degradation(
                "weight_compounding_scheduler",
                exc,
                action="treated scheduler state as empty after read failure",
                classification=FallbackClassification.SAFE_FALLBACK,
            )
        return {}

    def _save_state(self, updates: dict[str, Any]) -> None:
        try:
            state = self._load_state()
            state.update(updates)
            atomic_write_text(
                self._state_path(),
                json.dumps(state, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        except _RECOVERABLE as exc:
            record_degradation(
                "weight_compounding_scheduler",
                exc,
                action="continued with unpersisted scheduler state",
                classification=FallbackClassification.SAFE_FALLBACK,
            )

    def get_status(self) -> dict[str, Any]:
        state = self._load_state()
        status: dict[str, Any] = {
            "service": SERVICE_NAME,
            "active": self._active,
            "cycle_running": self._running_cycle,
            "check_interval_s": self.check_interval_s,
            "cooldown_s": self.cooldown_s,
            "autonomous_cap_gb": self.autonomous_max_gb,
            "last_attempt_at": state.get("last_attempt_at", 0.0),
            "last_status": state.get("last_status", "never_attempted"),
            "last_generation_id": state.get("last_generation_id", ""),
            "last_receipt": self._last_receipt,
        }
        try:
            status["lineage"] = self._build_loop().stats()
        except _RECOVERABLE as exc:
            record_degradation(
                "weight_compounding_scheduler",
                exc,
                action="reported scheduler status without lineage stats",
                classification=FallbackClassification.SAFE_FALLBACK,
                severity="debug",
            )
        return status


_scheduler: CompoundingScheduler | None = None


def get_compounding_scheduler(orchestrator: Any = None) -> CompoundingScheduler:
    global _scheduler
    if _scheduler is None:
        _scheduler = CompoundingScheduler(orchestrator)
    return _scheduler


def reset_compounding_scheduler_for_test() -> None:
    global _scheduler
    _scheduler = None


__all__ = [
    "SERVICE_NAME",
    "CompoundingScheduler",
    "get_compounding_scheduler",
    "reset_compounding_scheduler_for_test",
]
