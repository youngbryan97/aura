"""core/learning/selfplay_flywheel.py — idle self-play through the serving lane.

The closed experience→weights flywheel, final segment. The pieces already
running: real conversations feed the SFT buffer (learning_phase), the
reasoning amplifier emits verifier-checked DPO pairs when hard problems come
through chat, and the compounding scheduler turns the stores into weight
updates during deep idle. The gap this closes: DPO pairs only trickled in when
a USER happened to ask a verifiable question. Aura's idle time did nothing for
her weights.

Now idle time is practice. A burst samples a handful of seeded, exact-checkable
tasks and answers them THROUGH THE SERVING LANE (``llm_router.think`` with
``is_background=True`` — the priority beacon yields to any conversation, and no
second model is ever loaded). Every attempt is graded by the task's exact
checker; win/loss contrasts on the same prompt land in the canonical
VerifiablePreferenceHarness — the same store the amplifier feeds and the
compounding loop trains from. The promoted weights answer the next burst.
Practice → verified contrast → weight update → better practice. Receipts at
every step.

Soundness boundary, stated once: only exact-checkable attempts become training
signal (the verifier is the reward). Incident narratives and quality-gate
failures stay observability — a style-gate rejection is not a verified wrong
answer, and pretending otherwise would poison the reward.

Task seeds stay strictly below the held-out floor (1000): eval batteries are
minted at/above it, and the compounding harvest additionally drops any row
that textually collides with the sealed battery. The correct-rate this module
persists over time is itself a live capability trace worth watching.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from core.learning.heldout_battery import BatterySpec, generate_battery, grade_response
from core.runtime.atomic_writer import atomic_write_text
from core.runtime.errors import FallbackClassification, record_degradation
from core.runtime.executors import off_the_loop
from core.utils.concurrency import cancel_and_join

logger = logging.getLogger("Aura.SelfPlayFlywheel")

_RECOVERABLE = (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError)

SERVICE_NAME = "selfplay_flywheel"
EVAL_SEED_FLOOR = 1000          # seeds at/above are reserved for gate batteries
SEED_SPAN = 997                 # rotate task seeds 3..999 (prime span, full coverage)


def _resolve_practice_director() -> Any | None:
    """The Practice Director from the service spine, or None. Consumers must
    NEVER self-create one: boot registers the real instance, and an implicit
    singleton here would read (and write!) the runtime data dir from any
    hermetic test that exercises a burst."""
    try:
        from core.runtime.service_access import resolve_practice_director

        return resolve_practice_director(default=None)
    # not a failure: no practice director here, so the caller runs without one.
    except _RECOVERABLE:
        return None


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


class SelfPlayFlywheel:
    """Idle-gated practice bursts that convert idle time into DPO pairs."""

    def __init__(self, orchestrator: Any = None) -> None:
        self._orchestrator = orchestrator
        self._task: asyncio.Task | None = None
        self._active = False
        self.burst_tasks = _env_int("AURA_SELFPLAY_BURST_TASKS", 4)
        self.attempts_per_task = _env_int("AURA_SELFPLAY_ATTEMPTS", 3)
        self.burst_interval_s = float(_env_int("AURA_SELFPLAY_INTERVAL_S", 600))
        self.max_tokens = _env_int("AURA_SELFPLAY_MAX_TOKENS", 256)
        self._last_burst_at = 0.0

    # ── lifecycle ────────────────────────────────────────────────────────────

    async def start(self) -> None:
        if not _env_flag("AURA_SELFPLAY_FLYWHEEL", True):
            logger.info("Self-play flywheel disabled by AURA_SELFPLAY_FLYWHEEL=0.")
            return
        self._active = True
        from core.utils.task_tracker import get_task_tracker

        self._task = get_task_tracker().create_task(
            self._run(), name="selfplay_flywheel.loop"
        )
        logger.info(
            "Self-play flywheel online (%d tasks × %d attempts per burst, ≥%.0fs apart).",
            self.burst_tasks, self.attempts_per_task, self.burst_interval_s,
        )

    async def stop(self) -> None:
        self._active = False
        if self._task is not None:
            await cancel_and_join(
                self._task, owner="core.learning.selfplay_flywheel", timeout=5.0
            )
            self._task = None

    # ── loop ─────────────────────────────────────────────────────────────────

    async def _run(self) -> None:
        await asyncio.sleep(300.0)  # never compete with boot warmup
        while self._active:
            try:
                if self._burst_allowed():
                    await self._burst()
                    self._last_burst_at = time.time()
            except asyncio.CancelledError:
                raise
            except _RECOVERABLE as exc:
                record_degradation(
                    SERVICE_NAME,
                    exc,
                    action="skipped this practice burst; flywheel loop continues",
                    classification=FallbackClassification.SAFE_FALLBACK,
                )
            await asyncio.sleep(60.0)

    def _burst_allowed(self) -> bool:
        if time.time() - self._last_burst_at < self.burst_interval_s:
            return False
        from core.runtime.background_policy import (
            THOUGHT_BACKGROUND_POLICY,
            background_activity_allowed,
        )

        return background_activity_allowed(
            self._orchestrator, profile=THOUGHT_BACKGROUND_POLICY
        )

    # ── the burst ────────────────────────────────────────────────────────────

    async def _burst(self) -> dict[str, Any]:
        from core.learning.verifiable_preference_harness import (
            Attempt,
            get_verifiable_preference_harness,
        )
        from core.runtime.service_access import resolve_llm_router

        llm_router = resolve_llm_router(default=None)
        if llm_router is None or not hasattr(llm_router, "think"):
            record_degradation(
                SERVICE_NAME,
                RuntimeError("llm_router unavailable"),
                action="skipped practice burst without an LLM route",
                classification=FallbackClassification.SAFE_FALLBACK,
            )
            return {"skipped": "no_llm_router"}

        state = self._load_state()
        seed = 3 + (int(state.get("seed_cursor", 0)) % SEED_SPAN)
        tasks = self._practice_tasks(seed)
        harness = get_verifiable_preference_harness()

        burst_stats = {"seed": seed, "attempts": 0, "correct": 0, "pairs": 0, "aborted": False}
        domain_results: dict[str, list[int]] = {}
        for task in tasks:
            attempts: list[Attempt] = []
            for _ in range(self.attempts_per_task):
                # Yield mid-burst: if a conversation started, stop practicing.
                if not self._still_allowed():
                    burst_stats["aborted"] = True
                    break
                response = await llm_router.think(
                    task.prompt,
                    is_background=True,
                    origin="selfplay_flywheel",
                    temperature=0.85,
                    max_tokens=self.max_tokens,
                )
                text = str(response or "")
                if not text.strip():
                    continue
                ok = grade_response(task, text)
                attempts.append(
                    Attempt(candidate=text, verified=ok, checked=True,
                            confidence=1.0 if ok else 0.0)
                )
                burst_stats["attempts"] += 1
                burst_stats["correct"] += int(ok)
                tally = domain_results.setdefault(task.domain, [0, 0])
                tally[0] += 1
                tally[1] += int(ok)
            if len(attempts) >= 2:
                burst_stats["pairs"] += harness.ingest(
                    task.prompt, attempts, domain=f"selfplay:{task.domain}"
                )
            if burst_stats["aborted"]:
                break

        state["seed_cursor"] = int(state.get("seed_cursor", 0)) + 1
        # A burst that yielded before any attempt (machine stayed busy) is a
        # correct yield, not practice — count it separately so `bursts` stays
        # an honest count of real practice windows and the correct-rate isn't
        # diluted by no-ops.
        if burst_stats["attempts"]:
            state["bursts"] = int(state.get("bursts", 0)) + 1
        else:
            state["yielded_bursts"] = int(state.get("yielded_bursts", 0)) + 1
        state["total_attempts"] = int(state.get("total_attempts", 0)) + burst_stats["attempts"]
        state["total_correct"] = int(state.get("total_correct", 0)) + burst_stats["correct"]
        state["total_pairs"] = int(state.get("total_pairs", 0)) + burst_stats["pairs"]
        state["last_burst"] = {**burst_stats, "at": time.time()}
        # capability trace: exponential moving average of per-burst correct-rate
        if burst_stats["attempts"]:
            rate = burst_stats["correct"] / burst_stats["attempts"]
            prior = float(state.get("correct_rate_ema", rate))
            state["correct_rate_ema"] = round(0.8 * prior + 0.2 * rate, 4)
        await off_the_loop(self._save_state, state)

        # Feed the Practice Director: every verified outcome, per domain,
        # becomes curriculum evidence with this burst's state file + seed as
        # its receipt. Best-effort — direction must never break practice.
        director = _resolve_practice_director() if domain_results else None
        if director is not None:
            try:
                director.observe_burst(
                    {d: (t[0], t[1]) for d, t in domain_results.items()},
                    source=SERVICE_NAME,
                    receipt=f"{self._state_path()}#seed{seed}",
                )
                await director.flush_async()
            except _RECOVERABLE as exc:
                record_degradation(
                    SERVICE_NAME,
                    exc,
                    action="continued burst without curriculum feedback",
                    classification=FallbackClassification.SAFE_FALLBACK,
                    severity="debug",
                )

        if burst_stats["attempts"]:
            logger.info(
                "🎯 Self-play burst (seed %d): %d/%d correct, +%d pairs%s.",
                seed, burst_stats["correct"], burst_stats["attempts"],
                burst_stats["pairs"], " (yielded early)" if burst_stats["aborted"] else "",
            )
        return burst_stats

    def _practice_tasks(self, seed: int) -> list:
        """The burst's task set. The Practice Director concentrates it on the
        domains Aura actually fails (receipts-ranked); uniform round-robin is
        the fallback whenever direction is absent, disabled, or broken."""
        director = _resolve_practice_director()
        if director is None:
            return generate_battery(BatterySpec(seed=seed, size=self.burst_tasks))
        try:
            return director.focused_battery(seed=seed, size=self.burst_tasks)
        except _RECOVERABLE as exc:
            record_degradation(
                SERVICE_NAME,
                exc,
                action="fell back to uniform practice battery",
                classification=FallbackClassification.SAFE_FALLBACK,
                severity="debug",
            )
            return generate_battery(BatterySpec(seed=seed, size=self.burst_tasks))

    def _still_allowed(self) -> bool:
        from core.runtime.background_policy import (
            THOUGHT_BACKGROUND_POLICY,
            background_activity_allowed,
        )

        try:
            return background_activity_allowed(
                self._orchestrator, profile=THOUGHT_BACKGROUND_POLICY
            )
        # not a failure: background activity that cannot be confirmed as allowed is not
        # allowed, which is the refusing direction.
        except _RECOVERABLE:
            return False

    # ── state + status ───────────────────────────────────────────────────────

    def _state_path(self) -> Path:
        from core.config import get_config

        root = Path(get_config().paths.data_dir) / "learning"
        root.mkdir(parents=True, exist_ok=True)
        return root / "selfplay_flywheel.json"

    def _load_state(self) -> dict[str, Any]:
        try:
            path = self._state_path()
            if path.exists():
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return data
        except _RECOVERABLE as exc:
            record_degradation(
                SERVICE_NAME,
                exc,
                action="treated flywheel state as empty after read failure",
                classification=FallbackClassification.SAFE_FALLBACK,
                severity="debug",
            )
        return {}

    def _save_state(self, state: dict[str, Any]) -> None:
        try:
            atomic_write_text(
                self._state_path(),
                json.dumps(state, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        except _RECOVERABLE as exc:
            record_degradation(
                SERVICE_NAME,
                exc,
                action="continued with unpersisted flywheel state",
                classification=FallbackClassification.SAFE_FALLBACK,
            )

    def get_status(self) -> dict[str, Any]:
        state = self._load_state()
        return {
            "service": SERVICE_NAME,
            "active": self._active,
            "bursts": state.get("bursts", 0),
            "yielded_bursts": state.get("yielded_bursts", 0),
            "total_attempts": state.get("total_attempts", 0),
            "total_correct": state.get("total_correct", 0),
            "total_pairs": state.get("total_pairs", 0),
            "correct_rate_ema": state.get("correct_rate_ema"),
            "last_burst": state.get("last_burst"),
        }


_flywheel: SelfPlayFlywheel | None = None


def get_selfplay_flywheel(orchestrator: Any = None) -> SelfPlayFlywheel:
    global _flywheel
    if _flywheel is None:
        _flywheel = SelfPlayFlywheel(orchestrator)
    return _flywheel


def reset_selfplay_flywheel_for_test() -> None:
    global _flywheel
    _flywheel = None


__all__ = [
    "SERVICE_NAME",
    "SelfPlayFlywheel",
    "get_selfplay_flywheel",
    "reset_selfplay_flywheel_for_test",
]
