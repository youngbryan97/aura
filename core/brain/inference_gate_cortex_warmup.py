"""Whether the cortex is coming up, already up, or stuck — and what that permits.

A model that is loading looks exactly like a model that has wedged, from
outside. Telling them apart is what this does: the admission snapshot, the
lane transition claim that stops two loads racing, the backoff after a setback,
the deferral reasons that keep a warm-up from starting while somebody is
waiting, and the kill that is only allowed once a load has actually stopped
making progress.
"""
from __future__ import annotations

import logging

logger = logging.getLogger("Aura.InferenceGate")

#: The refusal this lane last logged, and when. Module state because the
#: throttle is per process, not per gate.
_LAST_EXPLICIT_DEFERRED_PREWARM_REFUSAL_AT = 0.0
_LAST_EXPLICIT_DEFERRED_PREWARM_REFUSAL_REASON = ""

import asyncio
import copy
import os
import time
from typing import Any

from core.runtime import resource_psutil as psutil
from core.runtime.errors import record_degradation
from core.runtime.shutdown_coordinator import is_shutdown_requested


class _WatchesTheCortexComeUp:
    """Lifted whole from InferenceGate; see inference_gate.py."""

    @staticmethod
    def _cortex_worker_is_legitimately_loading(client: Any) -> bool:
        """True when the cortex worker is running because it is LOADING the
        model, not because it is wedged.

        The cascade-cleanup path force-kills a "stuck" cortex worker to free
        blocked IPC feeder threads. But a worker actively loading the ~20GB
        32B is running and NOT stuck — killing it there was a full doom loop
        (2026-07-15 soak: spawn → load → killed mid-warmup on the next turn →
        warmup_deferred → repeat, 216s/turn, zero real cortex answers for an
        hour). A worker is legitimately loading when warmup is in flight OR
        the lane is warming/recovering, AND it entered that state within a
        generous load deadline. Past the deadline a still-warming worker is
        genuinely stuck and may be killed.
        """
        # Imported here rather than at module level: the module these
        # came from imports this one to build the class. A call-time
        # import also still sees a test's patch of the original.
        from .inference_gate import (
            InferenceGate,
            _worker_process_started_at,
        )

        if client is None:
            return False
        load_deadline_s = InferenceGate._env_float("AURA_CORTEX_LOAD_DEADLINE_S", 200.0)
        # A worker that has only just been spawned is neither wedged nor idle.
        # It is new.
        #
        # The lane bookkeeping below is set when warmup BEGINS, and there is a
        # window after the process exists where none of it is true yet. In
        # that window a running worker read as idle-but-running — the wedged
        # case — and was killed, which is the doom loop this guard was written
        # to end, reached through the one gap it did not cover. LIVE
        # 2026-08-26: spawn, "Loading model", "Model loaded", force-killed,
        # respawn, five times over, while every caller that needed her writing
        # was told "worker_not_alive" and the runtime's own health said the
        # lane was ready.
        #
        # Process creation time is the ground truth here: it cannot be unset,
        # cannot lag, and is already captured by the kill path itself.
        started_at = _worker_process_started_at(client)
        if started_at and (time.time() - started_at) < load_deadline_s:
            return True
        warming = bool(getattr(client, "_warmup_in_flight", False)) or str(
            getattr(client, "_lane_state", "")
        ) in {"warming", "recovering"}
        if not warming:
            return False
        transition_at = float(getattr(client, "_lane_transition_at", 0.0) or 0.0)
        warming_age = time.time() - transition_at if transition_at else 1e9
        return warming_age < load_deadline_s

    @staticmethod
    def _cortex_worker_is_actively_generating(client: Any) -> bool:
        """True when the worker is producing tokens right now.

        The mid-LOAD guard above closed one half of the doom loop. This is the
        other half, and it is the ~15-turn conversation ceiling: a generation
        that overran its budget got the worker force-killed, which costs a
        60-150s cold reload, which makes the NEXT turn slower, which overruns
        sooner. The 2026-07-25 probe recorded twenty
        "respawn_cortex_if_needed: cortex is dead" events across thirty turns,
        the UnitaryResponsePhase climbing 25s → 100s, and the answered rate
        falling 10/10 → 4/10 → 2/10 as it went.

        A slow worker and a wedged worker are not the same thing. Slowness is
        answered by the turn's own timeout and the fallback ladder; killing the
        lane converts one slow turn into a broken session.

        A generation that has run past AURA_CORTEX_GENERATION_DEADLINE_S is
        genuinely wedged and may still be killed.
        """
        from .inference_gate import (
            InferenceGate,
        )

        if client is None:
            return False
        if int(getattr(client, "_active_generations", 0) or 0) <= 0:
            return False
        started_at = float(getattr(client, "_active_generation_started_at", 0.0) or 0.0)
        if not started_at:
            return True  # generating, with no clock to condemn it by
        deadline_s = InferenceGate._env_float(
            "AURA_CORTEX_GENERATION_DEADLINE_S", 600.0
        )
        return (time.time() - started_at) < deadline_s

    @staticmethod
    def _recent_virtual_memory() -> Any:
        """psutil.virtual_memory(), at most once per _VIRTUAL_MEMORY_MEMO_TTL_S.

        Keyed on the identity of the probe as well as the clock. A class-level
        cache with only a TTL is order-dependent: the first version of this
        shared one reading across test functions, so a test that replaced the
        probe was served the previous test's value and three of them failed.
        Keying on the callable means replacing it — which is what patching does,
        and what a runtime swapping its resource observer does — misses the memo
        instead of silently reusing a reading taken through a different probe.
        """
        from .inference_gate import (
            InferenceGate,
        )

        probe = psutil.virtual_memory
        probe_key = id(probe)
        now = time.monotonic()
        with InferenceGate._virtual_memory_memo_lock:
            memo = InferenceGate._virtual_memory_memo
            if (
                memo is not None
                and memo[1] == probe_key
                and (now - memo[0]) <= InferenceGate._VIRTUAL_MEMORY_MEMO_TTL_S
            ):
                cached = memo[2]
                if isinstance(cached, BaseException):
                    raise cached
                return cached
        try:
            reading: Any = probe()
        except (OSError, RuntimeError, ValueError) as exc:
            # A BROKEN probe is remembered too, for the same window.
            #
            # Otherwise every caller re-attempts a syscall that has just
            # failed: measured at 20 raising probes in one
            # ensure_foreground_ready. "The probe is not answering" is as
            # valid a reading as a number, and half a second of staleness on
            # it costs nothing while re-asking twenty times costs the hot path
            # before every generation.
            with InferenceGate._virtual_memory_memo_lock:
                InferenceGate._virtual_memory_memo = (now, probe_key, exc)
            raise
        with InferenceGate._virtual_memory_memo_lock:
            InferenceGate._virtual_memory_memo = (now, probe_key, reading)
        return reading

    @staticmethod
    def _cortex_warmup_admission_snapshot(context: str = "background") -> dict[str, Any]:
        """Return whether a cold Cortex load is safe under current RAM pressure.

        The normal foreground headroom check is intentionally permissive because
        a *resident* Cortex can keep answering while RAM is high. A cold 32B
        load is different: it adds tens of GB of unified-memory pressure in one
        burst. This snapshot is therefore stricter and is used before any
        background/recovery/foreground warmup that would spawn the Cortex worker.
        
        [HARDENING v57-CORTEX] PRIORITY: 32B cortex is PRIMARY model. Must be less
        deferent to memory pressure to ensure system works regardless of cloud.
        """
        from .inference_gate import (
            _FLAG_FORCE_CORTEX_WARMUP_UNDER_PRESSURE,
            ADMISSION_SNAPSHOT_SCHEMA,
            InferenceGate,
            _record_inference_degradation,
        )

        context_key = str(context or "background").strip().upper()
        try:
            vm = InferenceGate._recent_virtual_memory()
            total_gb = float(vm.total) / float(1024**3)
            available_gb = float(vm.available) / float(1024**3)
            pressure_pct = float(vm.percent)

            if total_gb >= 60.0:
                # Cold-loading Cortex is a host-survival decision, not a normal
                # generation decision. The 32B lane is the user-facing default,
                # but it must not be admitted while macOS is close to swap/jetsam.
                default_max_pressure = 72.0 if context_key == "FOREGROUND" else 58.0
                default_min_available = 20.0 if context_key == "FOREGROUND" else 26.0
            else:
                default_max_pressure = 68.0 if context_key == "FOREGROUND" else 54.0
                default_min_available = 14.0 if context_key == "FOREGROUND" else 18.0

            max_pressure = InferenceGate._env_float(
                f"AURA_CORTEX_{context_key}_WARMUP_MAX_PRESSURE_PCT",
                InferenceGate._env_float(
                    "AURA_CORTEX_COLD_WARMUP_MAX_PRESSURE_PCT",
                    default_max_pressure,
                ),
            )
            min_available = InferenceGate._env_float(
                f"AURA_CORTEX_{context_key}_WARMUP_MIN_AVAILABLE_GB",
                InferenceGate._env_float(
                    "AURA_CORTEX_COLD_WARMUP_MIN_AVAILABLE_GB",
                    default_min_available,
                ),
            )
            can_admit = bool(pressure_pct < max_pressure and available_gb >= min_available)
            reason = ""
            if not can_admit:
                reason = (
                    f"memory_pressure:{pressure_pct:.1f}%/{available_gb:.1f}GB "
                    f"(need <{max_pressure:.1f}% and >={min_available:.1f}GB)"
                )
            return {
                "context": str(context or "background"),
                "pressure_pct": pressure_pct,
                "available_gb": available_gb,
                "total_gb": total_gb,
                "max_pressure_pct": max_pressure,
                "min_available_gb": min_available,
                "can_admit": can_admit,
                "reason": reason,
                "measured": True,
                "schema": ADMISSION_SNAPSHOT_SCHEMA,
                "measured_at_monotonic": time.monotonic(),
            }
        except (AttributeError, TypeError, ValueError, OSError) as exc:
            _record_inference_degradation(
                exc,
                action="continued bounded inference fallback after non-fatal degradation",
            )
            logger.debug("Cortex warmup memory probe failed: %s", exc)
            force_warmup = str(
                _FLAG_FORCE_CORTEX_WARMUP_UNDER_PRESSURE.value()
            ).strip().lower() in {"1", "true", "yes", "on"}
            # measured=False marks every numeric field below as UNKNOWN, not a
            # real observation — consumers must not treat these zeros as a
            # calm-memory measurement.
            return {
                "context": str(context or "background"),
                "pressure_pct": 0.0,
                "available_gb": 0.0,
                "total_gb": 0.0,
                "max_pressure_pct": 100.0,
                "min_available_gb": 0.0,
                "can_admit": force_warmup,
                "reason": (
                    "memory_probe_failed_forced_override"
                    if force_warmup
                    else "memory_probe_failed"
                ),
                "measured": False,
                "schema": ADMISSION_SNAPSHOT_SCHEMA,
                "measured_at_monotonic": time.monotonic(),
            }

    def cortex_load_setbacks(self) -> dict[str, int]:
        """Counts by kind, so "we killed it twice" and "it was slow twice" are
        distinguishable in the record."""
        return dict(getattr(self, "_cortex_load_setback_counts", {}) or {})

    def _claim_lane_transition(self, owner: str) -> bool:
        """Take the right to rewrite Cortex lane state. False means someone has it.

        The watchdog, the status path and the recovery scheduler all reach into
        the client's private ``_warmup_in_flight``, cancel the prewarm task and
        call private lane-state setters. None of them took anything first, so
        two of them could do it at once: one clears the flag while the other is
        mid-cancel, and a fresh warmup starts underneath a load that has not
        stopped.

        Deliberately not a lock: two of the three callers are synchronous and
        one is on the event loop, so a lock here would either block the loop or
        not be honoured. A compare-and-set with an expiry gives the same
        exclusion without either.
        """
        now = time.monotonic()
        held_by = getattr(self, "_lane_transition_owner", "")
        held_at = float(getattr(self, "_lane_transition_at", 0.0) or 0.0)
        if held_by and (now - held_at) < self._LANE_TRANSITION_LEASE_S:
            logger.debug(
                "Lane transition for %s refused; %s holds the lease (%.1fs old).",
                owner,
                held_by,
                now - held_at,
            )
            return False
        if held_by:
            logger.warning(
                "🔍 Lane-transition lease from %s expired after %.0fs; %s is taking it.",
                held_by,
                now - held_at,
                owner,
            )
        self._lane_transition_owner = str(owner)
        self._lane_transition_at = now
        return True

    def _release_lane_transition(self, owner: str) -> None:
        if getattr(self, "_lane_transition_owner", "") == str(owner):
            self._lane_transition_owner = ""
            self._lane_transition_at = 0.0

    def _clear_wedged_cortex_warmup(self, reason: str, *, owner: str) -> dict[str, Any]:
        """Clear a wedged warmup flag and cancel its load, under one owner.

        Cancelling a task is a REQUEST. The old code cancelled, set
        ``_prewarm_task = None`` and moved on, so a load that had not yet
        noticed the cancellation became invisible — and the next warmup started
        on top of it, two 20GB loads competing for one GPU slot. The task is
        kept here instead of dropped; :meth:`await_abandoned_cortex_loads`
        proves it stopped, and warmup admission refuses while one is unproven.
        """
        receipt: dict[str, Any] = {
            "reason": str(reason),
            "owner": str(owner),
            "cleared_warmup_flag": False,
            "cancelled_prewarm": False,
            "at": time.time(),
        }
        if not self._claim_lane_transition(owner):
            receipt["refused"] = "lane_transition_held"
            return receipt
        try:
            client = self._mlx_client
            if client is not None and getattr(client, "_warmup_in_flight", False):
                client._warmup_in_flight = False
                receipt["cleared_warmup_flag"] = True
            task = getattr(self, "_prewarm_task", None)
            if task is not None and not task.done():
                task.cancel()
                receipt["cancelled_prewarm"] = True
                abandoned = getattr(self, "_abandoned_cortex_loads", None)
                if abandoned is None:
                    abandoned = []
                    self._abandoned_cortex_loads = abandoned
                abandoned.append(task)
            self._prewarm_task = None
        finally:
            self._release_lane_transition(owner)
        return receipt

    def unproven_cortex_loads(self) -> int:
        """Cancelled loads that have not been observed to stop."""
        return len(
            [
                task
                for task in (getattr(self, "_abandoned_cortex_loads", None) or [])
                if not task.done()
            ]
        )

    async def await_abandoned_cortex_loads(self, timeout: float = 10.0) -> dict[str, Any]:
        """Wait for cancelled loads to actually finish, and say if they did not."""
        from .inference_gate import (
            _record_inference_degradation,
        )

        abandoned = list(getattr(self, "_abandoned_cortex_loads", None) or [])
        if not abandoned:
            return {"awaited": 0, "still_running": 0}
        # asyncio.wait, NOT wait_for: on timeout wait_for CANCELS what it is
        # waiting on, and the case this method exists to detect is a load that
        # ignores cancellation — so the cleanup would wait forever on the one
        # task it was written to notice. wait observes and returns.
        await asyncio.wait(abandoned, timeout=max(0.1, float(timeout)))
        still_running = [task for task in abandoned if not task.done()]
        self._abandoned_cortex_loads = still_running
        if still_running:
            _record_inference_degradation(
                TimeoutError(
                    f"{len(still_running)} cancelled Cortex load(s) did not stop within {timeout:.0f}s"
                ),
                action="refused to certify that a cancelled model load had stopped",
                severity="error",
                extra={"still_running": len(still_running)},
            )
        return {"awaited": len(abandoned), "still_running": len(still_running)}

    def _note_cortex_warmup_overrun(self) -> None:
        """A load exceeded its budget and was LEFT RUNNING. Not a kill."""
        self._note_cortex_load_setback(self.LOAD_SETBACK_OVERRUN)

    def _note_cortex_stuck_kill(self) -> None:
        """A stuck load was force-killed and reaped."""
        self._note_cortex_load_setback(self.LOAD_SETBACK_KILL)

    def _note_cortex_load_setback(self, kind: str) -> None:
        """Record a load setback and arm a warmup cooldown once they cluster.

        Each kill means a load attempt exceeded the deadline (thermal throttle /
        GPU contention) and got reaped. Re-spawning immediately just repeats the
        thrash, and every repeat grabs the single GPU slot for a 20GB weight
        load, starving the foreground fallback that is actually serving the turn.
        After ``AURA_CORTEX_STUCK_KILL_THRESHOLD`` kills inside a rolling window
        we cool down for an escalating interval, during which warmup is deferred
        and the resident fallback carries smoothly until thermal recovers.
        """
        from .inference_gate import (
            InferenceGate,
        )

        now = time.monotonic()
        window = InferenceGate._env_float("AURA_CORTEX_STUCK_KILL_WINDOW_S", 300.0)
        threshold = max(1, int(InferenceGate._env_float("AURA_CORTEX_STUCK_KILL_THRESHOLD", 2.0)))
        counts = getattr(self, "_cortex_load_setback_counts", None)
        if counts is None:
            counts = {}
            self._cortex_load_setback_counts = counts
        counts[str(kind)] = counts.get(str(kind), 0) + 1
        self._cortex_stuck_kill_times.append(now)
        recent = [t for t in self._cortex_stuck_kill_times if now - t <= window]
        if len(recent) < threshold:
            return
        base = InferenceGate._env_float("AURA_CORTEX_WARMUP_BACKOFF_S", 90.0)
        cap = InferenceGate._env_float("AURA_CORTEX_WARMUP_BACKOFF_CAP_S", 240.0)
        self._cortex_warmup_backoff_streak += 1
        cooldown = min(cap, base * self._cortex_warmup_backoff_streak)
        self._cortex_warmup_backoff_until = now + cooldown
        logger.warning(
            "🧊 [CORTEX BACKOFF] %d load setbacks in %.0fs — deferring warmup %.0fs so the "
            "resident fallback carries and thermal recovers before the next reload shot.",
            len(recent),
            window,
            cooldown,
        )

    def _cortex_warmup_backoff_reason(self) -> str | None:
        """Non-None while a post-thrash warmup cooldown is active."""
        backoff_until = float(
            getattr(self, "_cortex_warmup_backoff_until", 0.0) or 0.0
        )
        remaining = backoff_until - time.monotonic()
        if remaining <= 0.0:
            return None
        return f"warmup_backoff:{remaining:.0f}s"

    def _reset_cortex_warmup_backoff(self) -> None:
        """Clear the cooldown after the cortex proves it can serve again."""
        kill_times = getattr(self, "_cortex_stuck_kill_times", None)
        if getattr(self, "_cortex_warmup_backoff_until", 0.0) or kill_times:
            if kill_times is not None:
                kill_times.clear()
            self._cortex_warmup_backoff_until = 0.0
            self._cortex_warmup_backoff_streak = 0

    def _cortex_warmup_deferral_reason(self, context: str = "background") -> str | None:
        # The un-forced verdict: warmup backoff, then measured memory admission.
        # A probe failure (measured=False) is treated as a deferral here — the
        # snapshot only turns can_admit True on an unmeasured probe when the
        # force flag is set, and that emergency path is decided below, never by
        # a silent can_admit.
        backoff = self._cortex_warmup_backoff_reason()
        snapshot = self._cortex_warmup_admission_snapshot(context)
        measured = bool(snapshot.get("measured", True))
        if backoff is not None:
            normal_reason: str | None = backoff
        elif not measured:
            normal_reason = "memory_probe_failed"
        elif not snapshot["can_admit"]:
            normal_reason = str(snapshot["reason"] or "memory_pressure")
        else:
            normal_reason = None

        if normal_reason is None:
            return None  # Admission already allows warmup; no override needed.

        force_requested = str(
            os.environ.get(self._FORCE_WARMUP_FLAG, "")
        ).strip().lower() in {"1", "true", "yes", "on"}
        if not force_requested:
            return normal_reason

        # An override was requested to bypass a real deferral. It may skip the
        # soft admission thresholds and warmup backoff, but never the
        # host-survival floor: a cold 32B load into single-digit free GB risks
        # jetsam/swap-death of the whole process tree, which no operator
        # override should authorize.
        hard_floor_gb = self._env_float(
            "AURA_FORCE_CORTEX_WARMUP_HARD_FLOOR_GB", 10.0, minimum=4.0
        )
        available_gb = float(snapshot.get("available_gb", 0.0) or 0.0)
        if not measured:
            # The probe failed, so the survival floor cannot be confirmed. A
            # blind ~20GB cold load under the override is exactly the host-death
            # risk the floor exists to prevent — with no measurement there is no
            # boundary, so we fail closed rather than authorize an unbounded
            # load. The override takes effect again once the probe reports a
            # real above-floor reading.
            return "forced_warmup_denied_survival_floor_unmeasured"
        if available_gb < hard_floor_gb:
            return (
                "forced_warmup_denied_survival_floor:"
                f"{available_gb:.1f}GB"
                f"<{hard_floor_gb:.1f}GB"
            )

        # Past the inviolable floor, the override is a bounded, receipted
        # decision — not a permanent setting. It expires on its own, caps how
        # many bypasses one flag can authorize, and leaves a GovernanceReceipt
        # for each use, exactly as the MLX client governs the same flag.
        from core.brain.llm.emergency_override import consume_override

        decision = consume_override(
            self._FORCE_WARMUP_FLAG,
            guard=f"cortex_warmup_admission:{context}",
            observed=f"{normal_reason} (available={available_gb:.1f}GB)",
        )
        if not decision.active:
            # Expired or budget-exhausted: the memory guard is re-armed and the
            # normal deferral stands until the operator renews the decision.
            return normal_reason

        now = time.monotonic()
        last_log = getattr(self, "_last_forced_warmup_override_log_at", 0.0)
        if (now - last_log) > 60.0:
            self._last_forced_warmup_override_log_at = now
            logger.warning(
                "⚠️ %s active — bypassing %s warmup admission "
                "(available=%.1fGB, survival floor %.1fGB, %s).",
                self._FORCE_WARMUP_FLAG,
                context,
                available_gb,
                hard_floor_gb,
                decision.as_detail(),
            )
        return None

    def _log_cortex_warmup_deferral(self, reason: str, *, context: str) -> None:
        # COUNT every deferral, log a coalesced sample.
        #
        # A deferral is the ladder deciding not to load a tier, which is
        # designed backpressure and not a fault — recording it as a degradation
        # on this fail-closed subsystem escalates it to CRITICAL, and the
        # 2026-07-18 soak produced 52 of those from healthy deferrals. But a
        # runtime that cannot warm its primary lane IS something health should
        # be able to see, and a coalesced log line is not evidence. The counter
        # is the durable half; it reaches the conversation status snapshot.
        counters = getattr(self, "_warmup_deferral_counts", None)
        if counters is None:
            counters = {}
            self._warmup_deferral_counts = counters
        key = f"{context}:{reason}"
        entry = counters.get(key)
        if entry is None:
            entry = {"count": 0, "first_at": time.time(), "last_at": 0.0}
            counters[key] = entry
        entry["count"] += 1
        entry["last_at"] = time.time()

        now = time.monotonic()
        last_log = getattr(self, "_last_cortex_warmup_deferral_log_at", 0.0)
        if (now - last_log) < 30.0:
            return
        self._last_cortex_warmup_deferral_log_at = now
        logger.warning(
            "⏸️ Cortex %s warmup deferred to protect RAM: %s (%d so far)",
            context,
            reason,
            entry["count"],
        )

    def warmup_deferral_receipt(self) -> dict[str, Any]:
        """Every warmup deferral this process has taken, by cause.

        Deliberately not degradation records — see above — but durable, so a
        primary lane that has been refused a hundred times is a number
        somebody can find rather than a log line that scrolled.
        """
        return copy.deepcopy(getattr(self, "_warmup_deferral_counts", {}) or {})

    def _note_foreground_warmup_failure(self, warmup_exc: BaseException) -> bool:
        """Classify a foreground-warmup failure; returns True for RAM deferrals.

        A ``foreground_warmup_deferred`` outcome is expected RAM-admission
        backpressure — the turn reroutes to the fallback tier, so it is logged
        at info and NOT recorded as a degradation: on the fail-closed
        inference_gate a degradation record raises CRITICAL SERVICE FAILURE
        out of the handler and kills the protected recovery lane (seen live
        July 8: one memory deferral cascaded into chat 503s). Same discipline
        as the timeout demotion in core/runtime/errors.py. Genuine warmup
        faults keep the full degradation record.
        """
        if "foreground_warmup_deferred" in str(warmup_exc):
            logger.info(
                "🧠 Foreground warmup deferred by RAM admission; rerouting this turn: %s",
                warmup_exc,
            )
            return True
        record_degradation(
            "inference_gate",
            warmup_exc,
            severity="degraded",
            action="skipped cold primary attempt or fell back after foreground warmup failure",
        )
        from core.runtime.errors import describe_error

        logger.warning(
            "🧠 Foreground preflight warmup did not complete cleanly: %s",
            describe_error(warmup_exc),
        )
        return False

    def _log_cold_cortex_policy_deferred(self) -> None:
        now = time.monotonic()
        last_log = getattr(self, "_last_cortex_policy_deferred_log_at", 0.0)
        if (now - last_log) < 300.0:
            return
        self._last_cortex_policy_deferred_log_at = now
        logger.info(
            "Cold-start Cortex recovery deferred by desktop prewarm policy; "
            "foreground demand will warm the lane when needed."
        )

    @staticmethod
    def _boot_should_eager_warmup() -> bool:
        """Keep the resident Cortex warm on high-memory desktops unless disabled."""
        from .inference_gate import (
            _FLAG_BOOT_WARMUP_MIN_TOTAL_GB,
            _FLAG_EAGER_CORTEX_WARMUP,
            _FLAG_FORCE_CORTEX_WARMUP_UNDER_PRESSURE,
            _INFERENCE_RECOVERABLE_ERRORS,
            InferenceGate,
            _primary_lane_label,
            _record_inference_degradation,
        )

        if str(_FLAG_FORCE_CORTEX_WARMUP_UNDER_PRESSURE.value()).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            return True
        if InferenceGate._desktop_resource_guard_enabled():
            logger.info(
                "🛡️ Desktop resource guard active — skipping eager %s warmup during launch.",
                _primary_lane_label(),
            )
            return False
        setting = str(_FLAG_EAGER_CORTEX_WARMUP.value()).strip().lower()
        if setting in {"1", "true", "yes", "on"}:
            snapshot = InferenceGate._cortex_warmup_admission_snapshot("boot")
            if not snapshot["can_admit"] and str(
                _FLAG_FORCE_CORTEX_WARMUP_UNDER_PRESSURE.value()
            ).strip().lower() not in {"1", "true", "yes", "on"}:
                logger.warning(
                    "⏸️ Explicit eager Cortex warmup deferred to protect RAM: %s", snapshot["reason"]
                )
                return False
            return True
        if setting in {"0", "false", "no", "off"}:
            return False

        try:
            vm = InferenceGate._recent_virtual_memory()
            snapshot = InferenceGate._cortex_warmup_admission_snapshot("boot")
            min_total_gb = float(_FLAG_BOOT_WARMUP_MIN_TOTAL_GB.value())
            if (vm.total / float(1024**3)) < min_total_gb or not snapshot["can_admit"]:
                logger.warning(
                    "⏸️ Deferring eager %s warmup at boot "
                    "(total=%.1fGB pressure=%.1f%% available=%.1fGB).",
                    _primary_lane_label(),
                    snapshot["total_gb"],
                    snapshot["pressure_pct"],
                    snapshot["available_gb"],
                )
                return False
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            _record_inference_degradation(
                exc,
                action="kept conservative boot warmup decision after desktop policy probe failed",
            )
            logger.debug("Boot warmup memory probe failed: %s", exc)
            return False

        return True

    @staticmethod
    def _boot_should_schedule_deferred_prewarm() -> bool:
        from .inference_gate import (
            _EXPLICIT_DEFERRED_PREWARM_REFUSAL_LOG_INTERVAL_S,
            _FLAG_DEFERRED_CORTEX_PREWARM,
            _FLAG_FORCE_CORTEX_WARMUP_UNDER_PRESSURE,
            InferenceGate,
            _primary_lane_label,
        )

        explicit_setting = _FLAG_DEFERRED_CORTEX_PREWARM.value()
        setting = str(explicit_setting if explicit_setting is not None else "auto").strip().lower()
        if setting in {"1", "true", "yes", "on"}:
            snapshot = InferenceGate._cortex_warmup_admission_snapshot("background")
            if not snapshot["can_admit"] and str(
                _FLAG_FORCE_CORTEX_WARMUP_UNDER_PRESSURE.value()
            ).strip().lower() not in {"1", "true", "yes", "on"}:
                global _LAST_EXPLICIT_DEFERRED_PREWARM_REFUSAL_AT
                global _LAST_EXPLICIT_DEFERRED_PREWARM_REFUSAL_REASON
                now = time.monotonic()
                reason = str(snapshot["reason"] or "memory_pressure")
                if (
                    reason != _LAST_EXPLICIT_DEFERRED_PREWARM_REFUSAL_REASON
                    or (now - _LAST_EXPLICIT_DEFERRED_PREWARM_REFUSAL_AT)
                    >= _EXPLICIT_DEFERRED_PREWARM_REFUSAL_LOG_INTERVAL_S
                ):
                    _LAST_EXPLICIT_DEFERRED_PREWARM_REFUSAL_AT = now
                    _LAST_EXPLICIT_DEFERRED_PREWARM_REFUSAL_REASON = reason
                    logger.warning(
                        "⏸️ Explicit deferred Cortex prewarm refused to protect RAM: %s",
                        reason,
                    )
                else:
                    logger.debug(
                        "Explicit deferred Cortex prewarm still refused to protect RAM: %s",
                        reason,
                    )
                return False
            return True
        if setting in {"0", "false", "no", "off"}:
            return False
        if InferenceGate._desktop_safe_boot_enabled():
            if explicit_setting is None:
                logger.info(
                    "🛡️ Recovery safe boot active — skipping implicit deferred %s "
                    "prewarm during launch.",
                    _primary_lane_label(),
                )
                return False
            snapshot = InferenceGate._cortex_warmup_admission_snapshot("background")
            if not snapshot["can_admit"]:
                logger.warning(
                    "⏸️ Recovery safe-boot deferred Cortex prewarm deferred to protect RAM: %s",
                    snapshot["reason"],
                )
                return False
            return True
        return True

    @staticmethod
    def _cortex_already_resident() -> bool:
        """True when the conversation model is loaded and has served a turn.

        Deliberately conservative in both directions. It requires evidence that
        the model is actually up — a lane that merely intends to load does not
        count — and any failure to determine that answers False, which keeps
        the stricter load-sized floor rather than relaxing it on a guess.
        """
        from .inference_gate import (
            _INFERENCE_RECOVERABLE_ERRORS,
        )

        try:
            from core.container import ServiceContainer

            gate = ServiceContainer.peek("inference_gate", default=None)
            if gate is None:
                return False
            lane = gate.get_conversation_status()
        except _INFERENCE_RECOVERABLE_ERRORS:
            # A probe must never break admission; unknown residency keeps the
            # stricter load-sized floor.
            return False
        if not isinstance(lane, dict):
            return False
        try:
            if not bool(lane.get("conversation_ready")):
                return False
            # "Ready" without a completed generation is an intention, not a
            # residency: the weights may still be streaming in.
            return bool(lane.get("has_generated_successfully"))
        except (AttributeError, TypeError, ValueError):
            return False

    def _foreground_warmup_timeout(
        self, lane_status: dict[str, Any], primary_timeout: float
    ) -> float:
        """Admission control for the foreground preflight — break the doom loop.

        A COLD first boot legitimately needs ~150s to load the cortex, and the
        user expects that one-time wait. But a RECOVERY (Cortex was ready, got
        force-killed on a first-token stall, is reloading) must NOT hold every
        foreground turn hostage for 90-180s — observed live (Jul 7 soak):
        turns 21-30 crawled to 200s+ while a single warm window played out.

        When the lane was EVER ready (``last_ready_at`` > 0), cap the wait
        short (floored to 15s by ensure_foreground_ready — one honest warm
        chance) and let the turn fall to the ready fallback tier; the warmup
        task is shielded, so Cortex keeps warming in the background and the
        NEXT turn gets it. AURA_FOREGROUND_RECOVERY_WARMUP_CAP_S=180 restores
        the old behavior if this ever needs reverting live.
        """
        from .inference_gate import (
            InferenceGate,
        )

        was_ever_ready = float(lane_status.get("last_ready_at", 0.0) or 0.0) > 0.0
        if was_ever_ready:
            return InferenceGate._env_float(
                "AURA_FOREGROUND_RECOVERY_WARMUP_CAP_S", 15.0
            )
        # A lane held by SOMEONE ELSE is not a cold boot, and waiting the cold
        # budget for it waits for something that cannot happen.
        #
        # LIVE 2026-08-13: a training run (standalone:96317, CP399) held the
        # exclusive model lane. The cortex was admission-deferred 15 times in a
        # row; last_ready_at was 0.0 because it had never been ready THIS boot,
        # so every turn took the cold-boot branch and waited the full 180s for
        # a lane whose owner would hold it for hours. The brainstem — loaded,
        # weights present, not the lane owner — was never asked once, and the
        # user got "the live answer lane could not finish preparing".
        #
        # Preempting the owner is not the answer: that would destroy whatever
        # it is doing. Falling to the fallback tier is. The cortex keeps
        # warming behind a shielded task and takes over the moment the lane
        # frees.
        if InferenceGate._foreign_owner_holds_model_lane():
            return InferenceGate._env_float(
                "AURA_FOREGROUND_FOREIGN_LANE_WARMUP_CAP_S", 15.0
            )
        # [STABILITY v56] Cold 32B load can take 150s; give it at least 180s
        # or the primary timeout, whichever is greater.
        return max(180.0, float(primary_timeout))

    async def _await_warmup_deferral_clear(
        self,
        *,
        deadline: float,
        context: str,
        initial_reason: str,
    ) -> str:
        """Poll until the warmup deferral lifts, or the budget runs out.

        Returns "" when it cleared and the caller may proceed, or the last
        reason when it did not. Backpressure is a wait; only an exhausted
        budget is a failure.
        """

        reason = str(initial_reason or "")
        announced = False
        while reason and time.monotonic() < deadline:
            if is_shutdown_requested():
                return reason
            if not announced:
                logger.info(
                    "⏳ Cortex warmup deferred (%s); holding the turn for up to "
                    "%.0fs rather than answering with a failure.",
                    reason,
                    max(0.0, deadline - time.monotonic()),
                )
                announced = True
            await asyncio.sleep(0.5)
            lane = self.get_conversation_status()
            if self._lane_can_attempt_visible_conversation_turn(lane):
                return ""
            reason = str(self._cortex_warmup_deferral_reason(context) or "")
        if not reason and announced:
            logger.info("✅ Cortex warmup deferral cleared; the turn proceeds.")
        return reason

    def _confirmed_cortex_warmup(
        self, warmup_result: Any
    ) -> tuple[bool, dict[str, Any], str]:
        """Require process and lane evidence before reporting a warmup as successful."""
        from .inference_gate import (
            _INFERENCE_RECOVERABLE_ERRORS,
        )

        lane = self.get_conversation_status()
        state = str(lane.get("state", "") or "").strip().lower()
        blockers = [
            str(blocker)
            for blocker in (lane.get("readiness_blockers") or [])
            if str(blocker or "").strip()
        ]
        try:
            worker_alive = bool(self._mlx_client and self._mlx_client.is_alive())
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            worker_alive = False
            blockers.append(f"worker_probe_failed:{type(exc).__name__}")

        ready = bool(
            warmup_result is not False
            and not is_shutdown_requested()
            and worker_alive
            and state == "ready"
            and lane.get("conversation_ready")
        )
        if ready:
            return True, lane, ""
        if is_shutdown_requested():
            reason = "runtime_shutdown"
        elif warmup_result is False:
            reason = str(lane.get("last_failure_reason") or "warmup_deferred")
        elif not worker_alive:
            reason = "worker_not_alive"
        elif state != "ready":
            reason = f"lane_{state or 'unknown'}"
        elif not lane.get("conversation_ready"):
            reason = ",".join(blockers[:3]) or "conversation_not_ready"
        else:
            reason = "warmup_not_confirmed"
        return False, lane, reason
