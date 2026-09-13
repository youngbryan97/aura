"""Standing aside while a person is waiting.

Background work and a foreground turn want the same lane, and the background
one is the one that can wait. Every reason to defer is named rather than
boolean — a quiet window after boot, a foreground owner holding the model, a
safe-boot guard — so a deferral that turns out to be wrong can be argued with.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # annotation only; that module imports this one
    from .llm_health_router import EndpointHealth

import math
import os
import time
from typing import Any

from core.brain.llm.model_registry import (
    BRAINSTEM_ENDPOINT,
    FALLBACK_ENDPOINT,
)
from core.runtime.desktop_boot_safety import desktop_resource_guard_enabled


class _DefersBackgroundWork:
    """Lifted whole from HealthAwareLLMRouter; see llm_health_router.py."""

    @classmethod
    def _is_background_request(
        cls,
        *,
        origin: str | None,
        purpose: str | None,
        explicit_background: bool,
        explicit_foreground: bool = False,
    ) -> bool:
        # Imported here rather than at module level: the module these
        # came from imports this one to build the class. A call-time
        # import also still sees a test's patch of the original.
        from .llm_health_router import (
            _BACKGROUND_ORIGIN_HINTS,
            _USER_FACING_ORIGINS,
            _USER_FACING_PURPOSES,
        )

        if explicit_background:
            return True
        if explicit_foreground:
            return False

        normalized_purpose = str(purpose or "").strip().lower()
        if normalized_purpose in _USER_FACING_PURPOSES:
            return False

        tokens = cls._origin_tokens(origin)
        if not tokens:
            return normalized_purpose not in _USER_FACING_PURPOSES

        if tokens & _USER_FACING_ORIGINS:
            return False

        # Hardened default: anything that is not explicitly user-facing is
        # background. This prevents internal/kernel/autonomous traffic with
        # weak or unfamiliar origins from contaminating the foreground lane.
        if tokens & _BACKGROUND_ORIGIN_HINTS:
            return True

        return True

    def _background_suppression_result(
        self,
        *,
        origin: str | None,
        purpose: str | None,
        explicit_background: bool,
        explicit_foreground: bool = False,
    ) -> dict[str, Any] | None:
        """Return a suppression result before scarce generation capacity is acquired."""
        from .llm_health_router import (
            _record_router_degradation,
            logger,
        )


        is_bg = self._is_background_request(
            origin=origin,
            purpose=purpose,
            explicit_background=explicit_background,
            explicit_foreground=explicit_foreground,
        )
        if not is_bg:
            return None
        # Explicit tool compositions (composing a message to another AI in a live
        # web-interlocutor conversation) are foreground work the user asked for —
        # NOT deferrable background chatter. Serving them (rather than deferring
        # under foreground_quiet_window) is what lets her actually compose each
        # turn instead of the composition coming back empty and falling to a
        # canned default. Reply gates stay off because origin is non-user-facing.
        if str(origin or "").strip().lower().replace("-", "_") == "web_interlocutor":
            return None

        reason = ""
        try:
            from core.runtime.background_policy import (
                THOUGHT_BACKGROUND_POLICY,
                background_activity_reason,
            )

            reason = str(
                background_activity_reason(
                    None,
                    profile=THOUGHT_BACKGROUND_POLICY,
                    allow_no_user_anchor=True,
                )
                or ""
            )
        except (ImportError, AttributeError, RuntimeError) as exc:
            _record_router_degradation(
                exc,
                action="deferred background routing because background policy was unavailable",
                severity="degraded",
            )
            logger.warning("Background router policy probe failed: %s", exc)
            reason = "background_policy_unavailable"
        if not reason:
            try:
                from core.runtime.service_access import resolve_inference_gate

                gate = resolve_inference_gate()
                if gate and hasattr(gate, "_background_local_deferral_reason"):
                    reason = str(gate._background_local_deferral_reason(origin=origin) or "")
            except (ImportError, AttributeError, RuntimeError) as exc:
                _record_router_degradation(
                    exc,
                    action="continued background routing without inference-gate deferral signal",
                )
                logger.debug("Background router deferral probe failed: %s", exc)
        if not reason and self._foreground_quiet_window_active():
            reason = "foreground_quiet_window"
        if not reason and getattr(self, "high_pressure_mode", False):
            reason = "memory_pressure"
        if not reason and (
            self._foreground_user_turn_active() or self._foreground_owner_active()
        ):
            reason = "foreground_busy"

        if not reason:
            return None

        self._log_background_deferral(
            scope="generation_gate",
            origin=origin,
            reason=reason,
        )
        return {
            "ok": False,
            "text": "",
            "endpoint": "suppressed",
            "tokens": 0,
            "error": f"background_deferred:{reason}",
        }

    def _log_background_deferral(
        self,
        scope: str,
        origin: str,
        reason: str,
        endpoint: str | None = None,
    ) -> None:
        """Log repeated background deferrals as a state, not as a feed flood."""
        from .llm_health_router import (
            _deferral_reason_kind,
            logger,
        )

        key = f"{scope}:{endpoint or '*'}:{origin or '*'}"
        now = time.monotonic()
        previous_reason, previous_at, suppressed = self._background_deferral_log_state.get(
            key,
            ("", 0.0, 0),
        )
        # Compare the CAUSE, not the instantaneous measurement. The reason
        # carries live numbers —
        # "desktop_background_headroom:Reflex:66.6%/21.3GB(need <66.0% ...)" —
        # so 66.6 vs 66.5 vs 66.4 made every sample a "new" reason and the
        # suppression below essentially never fired. Measured live in the
        # neural feed: roughly forty lines a minute of one deferral, drowning
        # out every actual thought, with an occasional "after suppressing 1"
        # on the rare tick where two samples rounded identically.
        #
        # The numbers still appear in the message; they just no longer decide
        # whether it is the same event.
        reason_kind = _deferral_reason_kind(reason)
        previous_kind = _deferral_reason_kind(previous_reason)
        if reason_kind == previous_kind and (now - previous_at) < 30.0:
            self._background_deferral_log_state[key] = (previous_reason, previous_at, suppressed + 1)
            logger.debug(
                "Router: repeated background deferral suppressed scope=%s endpoint=%s origin=%s reason=%s.",
                scope,
                endpoint or "",
                origin,
                reason,
            )
            return

        self._background_deferral_log_state[key] = (reason, now, 0)
        suffix = f" after suppressing {suppressed} repeated notices" if suppressed else ""
        if endpoint:
            logger.info(
                "⏸️ Router: Deferring background local endpoint %s (%s)%s.",
                endpoint,
                reason,
                suffix,
            )
            return
        logger.info(
            "⏸️ Router: Queueing background inference until admission clears for origin=%s reason=%s%s.",
            origin,
            reason,
            suffix,
        )

    @classmethod
    def _foreground_user_turn_active(cls) -> bool:
        from .llm_health_router import (
            _record_router_degradation,
        )

        try:
            from core.runtime.service_access import resolve_orchestrator

            orch = resolve_orchestrator()
            if not orch:
                # No orchestrator (tests, standalone scripts): genuinely no
                # foreground turn to protect.
                return False

            status = getattr(orch, "status", None)
            if not getattr(status, "is_processing", False):
                return False

            current_origin = getattr(orch, "_current_origin", "")
            if not cls._is_user_facing_origin(current_origin):
                return False

            return not bool(getattr(orch, "_current_task_is_autonomous", False))
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            # Fail PROTECTIVE: this probe guards the foreground lane from
            # background admission. Broken telemetry must not read as "no
            # foreground owner" — that removed protection exactly when
            # ownership could not be established.
            _record_router_degradation(
                exc,
                action="assumed an active foreground turn after ownership probe failed",
                severity="degraded",
            )
            return True

    @classmethod
    def _foreground_quiet_window_active(cls) -> bool:
        from .llm_health_router import (
            _record_router_degradation,
        )

        try:
            from core.runtime.service_access import resolve_orchestrator

            orch = resolve_orchestrator()
            if not orch:
                return False

            quiet_until = float(getattr(orch, "_foreground_user_quiet_until", 0.0) or 0.0)
            return quiet_until > time.time()
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            # Fail PROTECTIVE (see _foreground_user_turn_active).
            _record_router_degradation(
                exc,
                action="assumed the foreground quiet window is active after probe failed",
                severity="degraded",
            )
            return True

    def _safe_boot_background_guard_active(self) -> bool:
        """Reserve launch headroom for foreground chat before waking spare local models."""
        if not desktop_resource_guard_enabled():
            return False
        try:
            guard_secs = float(os.environ.get("AURA_SAFE_BOOT_BACKGROUND_GUARD_SECS", "180"))
        except (TypeError, ValueError):
            # float() raises conversion errors, not network errors — the old
            # tuple let a malformed env value abort routing entirely.
            guard_secs = 180.0
        if not math.isfinite(guard_secs):
            guard_secs = 180.0
        if guard_secs <= 0:
            return False
        return (time.monotonic() - self._created_at) < guard_secs

    @staticmethod
    def _desktop_background_local_enabled() -> bool:
        """Default ON — but this only lifts the BLANKET block.

        Background cognition on a 64GB desktop must not freely wake extra
        7B/1.5B MLX workers beside the ~35GB 32B Cortex; that pattern showed up
        live as a footprint spike followed by forced shedding. The old default
        answered that by refusing the whole lane, which also refused every case
        where there was ample headroom.

        Per-endpoint memory admission in
        ``_desktop_background_endpoint_deferral_reason`` is the real guard and
        it still runs: Brainstem needs substantially more free unified memory
        than Reflex, and both are checked against a live pressure snapshot on
        every dispatch. Lifting the blanket refusal leaves that admission in
        place rather than removing it, so a background model wakes when the
        memory is genuinely there and defers when it is not.

        AURA_ENABLE_DESKTOP_BACKGROUND_LOCAL_LLM=0 restores the blanket refusal.
        """
        raw = str(
            os.environ.get("AURA_ENABLE_DESKTOP_BACKGROUND_LOCAL_LLM", "1")
        ).strip().lower()
        return raw in {"1", "true", "yes", "on"}

    def _desktop_background_local_disabled(self) -> bool:
        return desktop_resource_guard_enabled() and not self._desktop_background_local_enabled()

    @staticmethod
    def _desktop_background_endpoint_deferral_reason(ep: EndpointHealth) -> str | None:
        """Protect live desktop Aura from background local-model memory spikes.

        Background cognition should stay active, but on a 64GB-class desktop it
        cannot freely wake extra 9B/1.5B MLX workers beside the cortex lane.
        That pattern is what showed up in the live neural stream as a large
        footprint spike followed by forced shedding.  Admission is endpoint
        specific: Reflex is light enough to run with moderate headroom, while
        Brainstem needs substantially more free unified memory.
        """
        from .llm_health_router import (
            _record_router_degradation,
        )

        if not desktop_resource_guard_enabled():
            return None
        name = str(getattr(ep, "name", "") or "")
        if name not in {BRAINSTEM_ENDPOINT, FALLBACK_ENDPOINT}:
            return None
        try:
            from core.utils.memory_monitor import get_memory_pressure_snapshot

            snapshot = get_memory_pressure_snapshot()
            pressure_pct = float(snapshot.pressure_pct)
            available_gb = float(snapshot.available_gb)
            process_rss_gb = float(snapshot.process_rss_gb)
            process_limit_gb = float(snapshot.process_rss_limit_gb)
        except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
            _record_router_degradation(
                exc,
                action="deferred desktop background local endpoint after memory probe failed",
                severity="warning",
            )
            return "desktop_background_memory_probe_failed"
        if not (math.isfinite(pressure_pct) and math.isfinite(available_gb)):
            # NaN readings make BOTH admission comparisons false — that is
            # fail-open admission on a corrupt probe, not headroom.
            return "desktop_background_memory_probe_invalid"

        def _threshold(env_name: str, default: float) -> float:
            """NaN or malformed values must neither crash routing nor make
            both pressure comparisons false (fail-open admission)."""
            try:
                value = float(os.environ.get(env_name, str(default)))
            except (TypeError, ValueError):
                return default
            return value if math.isfinite(value) else default

        if name == BRAINSTEM_ENDPOINT:
            # Brainstem is the 9B (~6GB @ 4-bit) background lane. The old
            # 48% / 34GB-free gate was UNMEETABLE on a desktop whose whole job
            # is holding the ~16-20GB 32B Cortex: steady state is ~56% / ~28GB
            # available, so background cognition could NEVER admit → mind_tick
            # never completes a successful tick → false-death → the launcher
            # respawns a second 32B → memory doubling → worse false-death (a
            # self-sustaining respawn loop, observed 2026-07-06). Calibrate to
            # the hardware: allow the 7B beside the Cortex while holding a 22GB
            # available floor (above Reflex's 20GB) — the external memory
            # sentinel (42GB RSS lethal) remains the hard OOM backstop.
            max_pressure = _threshold("AURA_BACKGROUND_BRAINSTEM_MAX_PRESSURE_PCT", 62.0)
            min_available = _threshold("AURA_BACKGROUND_BRAINSTEM_MIN_AVAILABLE_GB", 22.0)
        else:
            max_pressure = _threshold("AURA_BACKGROUND_REFLEX_MAX_PRESSURE_PCT", 66.0)
            min_available = _threshold("AURA_BACKGROUND_REFLEX_MIN_AVAILABLE_GB", 20.0)
        # The percentages above come from psutil's macOS accounting, which
        # counts file-backed cache and compressed pages as consumed. The OS
        # reclaims those on demand, so during a cortex load the derived reading
        # says "no headroom" while the kernel reports no pressure at all.
        #
        # LIVE 2026-08-17: that is why the fallback ladder returned an empty
        # answer on every cold start. The Brainstem and the CPU-only Reflex
        # were both deferred for want of headroom the machine had, so a turn
        # the cortex could not take was answered by nobody.
        #
        # When the OS itself says there is no pressure, the derived percentage
        # does not get to veto the small models. The absolute allocation floor
        # still binds. It represents memory the new worker and the already
        # resident foreground lane need after admission; kernel pressure says
        # whether pages are currently contested, not whether two model peaks
        # fit together. Lowering this floor to 4GB admitted a 9B beside the cortex
        # at 67% host use, then the emergency reclaimer killed the Cortex.
        try:
            from core.utils.memory_monitor import kernel_memory_pressure_level

            kernel_level = kernel_memory_pressure_level()
        except (ImportError, OSError, RuntimeError, ValueError):
            kernel_level = "unknown"
        if kernel_level == "normal":
            max_pressure = max(max_pressure, 100.0)
        elif kernel_level == "critical":
            max_pressure = min(max_pressure, 0.0)
        if pressure_pct >= max_pressure or available_gb < min_available:
            return (
                f"desktop_background_headroom:{name}:"
                f"{pressure_pct:.1f}%/{available_gb:.1f}GB"
                f"(need <{max_pressure:.1f}% and >={min_available:.1f}GB)"
            )
        if process_limit_gb > 0.0 and process_rss_gb >= max(0.0, process_limit_gb - 6.0):
            return (
                f"desktop_background_process_rss:{process_rss_gb:.1f}GB/"
                f"{process_limit_gb:.1f}GB"
            )
        return None

    def _cortex_startup_quiet_window_active(self) -> bool:
        """Block background local fallbacks while Cortex is still warming or launch headroom is reserved."""
        from .llm_health_router import (
            logger,
        )

        if self._safe_boot_background_guard_active():
            return True
        if not self._foreground_quiet_window_active():
            return False

        try:
            from core.runtime.service_access import resolve_inference_gate

            gate = resolve_inference_gate()
            if gate and hasattr(gate, "get_conversation_status"):
                lane = gate.get_conversation_status() or {}
                if lane.get("conversation_ready"):
                    return False
                state = str(lane.get("state", "") or "").strip().lower()
                if lane.get("warmup_in_flight"):
                    return True
                return state in {"cold", "spawning", "handshaking", "warming", "recovering"}
        except (ImportError, AttributeError, RuntimeError):
            logger.debug("Router quiet-window lane probe failed.", exc_info=True)

        # Fail safe: if the quiet window is active but lane state is unavailable,
        # avoid waking extra local models until Cortex protection expires.
        return True

    @staticmethod
    def _foreground_owner_active() -> bool:
        from .llm_health_router import (
            _record_router_degradation,
        )

        try:
            from core.brain.llm.mlx_client import _foreground_owner_active
        except ImportError:
            # MLX client absent from this build: there is no local
            # foreground owner to protect.
            return False
        try:
            return bool(_foreground_owner_active())
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            # Fail PROTECTIVE: a crashed ownership probe must not admit
            # background work into a possibly-owned foreground lane.
            _record_router_degradation(
                exc,
                action="assumed an active foreground owner after MLX ownership probe failed",
                severity="degraded",
            )
            return True
