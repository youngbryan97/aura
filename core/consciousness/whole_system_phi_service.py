"""core/consciousness/whole_system_phi_service.py — live whole-system Φ.
==========================================================================
The runtime host for core/consciousness/integrated_information.py:
harvests real channels each cognitive cycle, keeps a sliding window, and
periodically produces a provenance-complete PhiEstimate off the event
loop.  Registered as ``whole_system_phi``; the report is consumed by the
phi phase (state metadata + logs), persisted through the governed write
gateway, and exposed via status() for health/observability.

This supersedes the epistemics of the hand-picked 16-node φ: channels are
whatever the runtime actually exposes, every retained channel is represented
through a bounded quotient, and every number ships with its search method,
exactness boundary, null, CI, and bounded claim.  The legacy phi scalar keeps
its scale for downstream gates; this service adds the honest measurement plus
named interventional rows from low-cadence reversible probe campaigns.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import threading
import time
from collections import deque
from typing import Any

from core.runtime.errors import record_degradation
from core.runtime.flags import FlagKind, declare
from core.runtime.resource_observation import get_resource_observer
from core.runtime.service_access import optional_service
from core.runtime.state_ownership import state_root

logger = logging.getLogger("Aura.WholeSystemPhi")

_MIN_SAMPLES_FLAG = declare(
    "AURA_WSPHI_MIN_SAMPLES", kind=FlagKind.INT, default=240,
    description="Observations required before the first whole-system Φ estimate",
    owner="core.consciousness.whole_system_phi_service",
)
_WINDOW_FLAG = declare(
    "AURA_WSPHI_WINDOW", kind=FlagKind.INT, default=1200,
    description="Sliding-window length (observations) for whole-system Φ",
    owner="core.consciousness.whole_system_phi_service",
)
_EVERY_FLAG = declare(
    "AURA_WSPHI_ESTIMATE_EVERY", kind=FlagKind.INT, default=180,
    description="Recompute whole-system Φ every N new observations",
    owner="core.consciousness.whole_system_phi_service",
)
_PHI_DIR_FLAG = declare(
    "AURA_PHI_DIR", kind=FlagKind.STRING, default="",
    description="Override directory for persisted whole-system Φ reports",
    owner="core.consciousness.whole_system_phi_service",
)
_MAX_ELEMENTS_FLAG = declare(
    "AURA_WSPHI_MAX_ELEMENTS", kind=FlagKind.INT, default=16,
    description="Maximum estimator elements after a coverage-preserving quotient",
    owner="core.consciousness.whole_system_phi_service",
)
_SURROGATES_FLAG = declare(
    "AURA_WSPHI_SURROGATES", kind=FlagKind.INT, default=20,
    description="Circular-shift null draws per whole-system estimate",
    owner="core.consciousness.whole_system_phi_service",
)
_BOOTSTRAP_FLAG = declare(
    "AURA_WSPHI_BOOTSTRAPS", kind=FlagKind.INT, default=20,
    description="Moving-block bootstrap draws per whole-system estimate",
    owner="core.consciousness.whole_system_phi_service",
)
_PROBE_ENABLED_FLAG = declare(
    "AURA_WSPHI_PROBE_ENABLED", kind=FlagKind.BOOL, default=True,
    description="Enable low-cadence governed causal perturbation campaigns",
    owner="core.consciousness.whole_system_phi_service",
)
_PROBE_INITIAL_DELAY_FLAG = declare(
    "AURA_WSPHI_PROBE_INITIAL_DELAY_S", kind=FlagKind.FLOAT, default=900.0,
    description="Minimum stable runtime age before the first causal probe campaign",
    owner="core.consciousness.whole_system_phi_service",
)
_PROBE_INTERVAL_FLAG = declare(
    "AURA_WSPHI_PROBE_INTERVAL_S", kind=FlagKind.FLOAT, default=21600.0,
    description="Minimum seconds between governed causal probe campaigns",
    owner="core.consciousness.whole_system_phi_service",
)
_PROBE_RETRY_FLAG = declare(
    "AURA_WSPHI_PROBE_RETRY_S", kind=FlagKind.FLOAT, default=1800.0,
    description="Retry delay after a deferred or refused causal probe campaign",
    owner="core.consciousness.whole_system_phi_service",
)
_PROBE_TRIALS_FLAG = declare(
    "AURA_WSPHI_PROBE_TRIALS", kind=FlagKind.INT, default=3,
    description="Paired sham/perturb/recovery trials per causal campaign",
    owner="core.consciousness.whole_system_phi_service",
)


def _maybe(value: Any) -> float | None:
    try:
        v = float(value)
    # not a failure: a value that is not a number is not one this can read.
    except (TypeError, ValueError):
        return None
    if v != v or v in (float("inf"), float("-inf")):
        return None
    return v


class WholeSystemPhiService:
    """Sliding-window channel collector + periodic estimator."""

    def __init__(self) -> None:
        self.shutdown_timeout_s = 15.0
        self._lock = threading.RLock()
        self._window: deque[dict[str, float]] = deque(maxlen=int(_WINDOW_FLAG.value()))
        self._since_estimate = 0
        self._latest: Any = None            # PhiEstimate
        self._latest_probe: dict[str, Any] = {}
        self._interventional: list[Any] = []
        self._estimating = False
        self._estimates_done = 0
        self._last_error = ""
        self._last_seed = 0
        self._last_compute_seconds = 0.0
        self._last_matrix_diagnostics: dict[str, Any] = {}
        self._estimate_task: asyncio.Task[Any] | None = None
        self._probe_task: asyncio.Task[Any] | None = None
        self._admission_deferrals = 0
        self._last_admission_reason = ""
        self._probe_admission_deferrals = 0
        self._last_probe_admission_reason = ""
        self._last_probe_at = 0.0
        self._last_probe_attempt_at = 0.0
        self._created_at = time.time()
        self._latest_runtime_state: Any = None
        self._shutting_down = False

    # ── channel harvest ──────────────────────────────────────────────────
    def sample_runtime_channels(self, state: Any = None) -> dict[str, float]:
        """One reading of every cheap live scalar the runtime exposes.
        Each source is optional and guarded; a missing organ just means a
        narrower channel set — reported, never fabricated."""
        channels: dict[str, float] = {}

        try:
            affect = optional_service("affect_engine", "affect_facade", default=None)
            if affect is not None and hasattr(affect, "get_state_sync"):
                st = affect.get_state_sync()
                if isinstance(st, dict):
                    for key in ("valence", "arousal", "dominance"):
                        v = _maybe(st.get(key))
                        if v is not None:
                            channels[f"affect.{key}"] = v
                    # Damasio V2 exposes these on a 0-100 scale
                    for key in ("curiosity", "frustration", "stability"):
                        v = _maybe(st.get(key))
                        if v is not None:
                            channels[f"affect.{key}"] = v / 100.0
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation("whole_system_phi", exc, severity="debug",
                               action="affect channels skipped")

        try:
            stakes = optional_service("existential_stakes", default=None)
            if stakes is not None and hasattr(stakes, "get_existential_threat"):
                v = _maybe(stakes.get_existential_threat())
                if v is not None:
                    channels["survival.threat"] = v
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation("whole_system_phi", exc, severity="debug",
                               action="stakes channel skipped")

        try:
            unity = optional_service("unity_state", default=None)
            if unity is not None:
                for attr, name in (("unity_score", "unity.score"),
                                   ("fragmentation_score", "unity.fragmentation")):
                    v = _maybe(getattr(unity, attr, None))
                    if v is not None:
                        channels[name] = v
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation("whole_system_phi", exc, severity="debug",
                               action="unity channels skipped")

        try:
            will = optional_service("unified_will", default=None)
            wstate = getattr(will, "_state", None)
            if wstate is not None:
                for attr in ("confidence", "assertiveness", "identity_coherence"):
                    v = _maybe(getattr(wstate, attr, None))
                    if v is not None:
                        channels[f"will.{attr}"] = v
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation("whole_system_phi", exc, severity="debug",
                               action="will channels skipped")

        try:
            covenant = optional_service("ulysses_covenant", default=None)
            if covenant is not None and hasattr(covenant, "integrity_score"):
                v = _maybe(covenant.integrity_score())
                if v is not None:
                    channels["covenant.integrity"] = v
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation("whole_system_phi", exc, severity="debug",
                               action="covenant channel skipped")

        if state is not None:
            for path, name in (
                (("consciousness", "phi"), "state.phi"),
                (("consciousness", "arousal"), "state.arousal"),
                (("cognition", "cognitive_load"), "state.cognitive_load"),
                (("phenomenal", "intensity"), "state.phenomenal_intensity"),
            ):
                obj = state
                for part in path:
                    obj = getattr(obj, part, None)
                    if obj is None:
                        break
                v = _maybe(obj)
                if v is not None:
                    channels[name] = v

        try:
            process = get_resource_observer().process(os.getpid())
            if process is not None:
                channels["body.rss_gb"] = process.rss_bytes / 1e9
                cpu = _maybe(process.cpu_percent)
                if cpu is not None:
                    channels["body.cpu_pct"] = cpu
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation("whole_system_phi", exc, severity="debug",
                               action="body channels skipped")

        return channels

    def observe(self, channels: dict[str, float]) -> None:
        clean = {
            str(key): float(value)
            for key, value in (channels or {}).items()
            if _maybe(value) is not None
        }
        if len(clean) < 2:
            return
        with self._lock:
            self._window.append(clean)
            self._since_estimate += 1

    def observe_runtime(self, state: Any = None) -> None:
        with self._lock:
            self._latest_runtime_state = state
        self.observe(self.sample_runtime_channels(state))

    def add_interventional_transitions(
        self, transitions: list[tuple[tuple[int, ...], tuple[int, ...]]],
        probe_report: dict[str, Any] | None = None,
        *,
        channel_names: tuple[str, ...] | None = None,
    ) -> None:
        """Add interventional rows to the discrete estimator.

        ``channel_names`` is required in practice: rows are dropped without it.

        Anonymous rows are not merely useless, they are unsafe. Constant
        channels are dropped before estimation, so a raw 13-tuple no longer
        corresponds to the 8 surviving elements. When the arity does not match
        the row is rejected at projection — that is what emptied the checked-in
        campaign (5 trials in, ``n_interventional_transitions: 0``,
        ``n_projection_rejected_transitions: 5``) while the artifact kept the
        name ``estimate_with_interventions``.

        When the arity *coincidentally* matches, the row is worse than useless:
        it is interpreted positionally against whatever channels happened to
        survive, silently attributing an intervention on one channel to another.
        A misaligned interventional row is fabricated causal evidence. So rows
        without names are refused outright rather than passed downstream to be
        either rejected or misread.
        """
        raw = list(transitions or [])
        if not channel_names:
            if raw:
                record_degradation(
                    "whole_system_phi",
                    ValueError(
                        f"{len(raw)} interventional transitions supplied without "
                        "channel_names; refusing them (they could not be aligned to "
                        "the estimator's retained channels)"
                    ),
                    action="dropped anonymous interventional rows",
                    enforce_failure_policy=False,
                )
                logger.error(
                    "WholeSystemPhi: REFUSED %d interventional transitions that "
                    "arrived without channel_names. Unnamed rows cannot be aligned "
                    "to the retained channels and would be silently misread if the "
                    "arity happened to match. Pass channel_names.",
                    len(raw),
                )
            if probe_report:
                with self._lock:
                    self._latest_probe = dict(probe_report)
            return

        from core.consciousness.integrated_information import (
            InterventionalTransition,
        )

        additions: list[Any] = [
            InterventionalTransition(
                channel_names=tuple(channel_names),
                before=tuple(before),
                after=tuple(after),
            )
            for before, after in raw
        ]
        with self._lock:
            self._interventional.extend(additions)
            self._interventional = self._interventional[-200:]
            if probe_report:
                self._latest_probe = dict(probe_report)

    # ── estimation ───────────────────────────────────────────────────────
    def _matrix(self) -> tuple[Any, tuple[str, ...], dict[str, Any]]:
        import numpy as np

        with self._lock:
            rows = list(self._window)
        # channels present in ≥90% of the window (late-boot organs join later)
        counts: dict[str, int] = {}
        for r in rows:
            for k in r:
                counts[k] = counts.get(k, 0) + 1
        names = tuple(sorted(k for k, c in counts.items()
                             if c >= 0.9 * len(rows)))
        columns: list[Any] = []
        missing_by_channel: dict[str, int] = {}
        sample_index = np.arange(len(rows), dtype=float)
        for name in names:
            values = np.asarray([row.get(name, np.nan) for row in rows], dtype=float)
            valid = np.isfinite(values)
            missing = int((~valid).sum())
            missing_by_channel[name] = missing
            if missing:
                # Nearest-edge linear interpolation avoids manufacturing a zero
                # impulse whenever an otherwise healthy source misses one tick.
                values = np.interp(sample_index, sample_index[valid], values[valid])
            columns.append(values)
        X = np.stack(columns, axis=1) if columns else np.empty((len(rows), 0))
        diagnostics = {
            "candidate_channels": len(counts),
            "retained_channels": len(names),
            "dropped_sparse_channels": sorted(set(counts) - set(names)),
            "missing_values_interpolated": sum(missing_by_channel.values()),
            "missing_by_channel": missing_by_channel,
        }
        return X, names, diagnostics

    def ready(self) -> bool:
        with self._lock:
            return (len(self._window) >= int(_MIN_SAMPLES_FLAG.value())
                    and self._since_estimate >= int(_EVERY_FLAG.value()))

    def maybe_estimate(self) -> Any:
        """Heavy — run off-loop (asyncio.to_thread).  Returns the fresh
        PhiEstimate when due, else None."""
        with self._lock:
            if self._estimating or not self.ready():
                return None
            self._estimating = True
        try:
            from core.consciousness.integrated_information import (
                estimate_whole_system_phi,
            )

            X, names, matrix_diagnostics = self._matrix()
            if X.shape[1] < 4:
                return None
            with self._lock:
                extra = list(self._interventional)
            digest = hashlib.sha256()
            digest.update("\0".join(names).encode("utf-8"))
            digest.update(X.astype("<f8", copy=False).tobytes(order="C"))
            seed = int.from_bytes(digest.digest()[:8], "big") & 0x7FFF_FFFF
            started = time.perf_counter()
            est = estimate_whole_system_phi(
                X, channel_names=names,
                n_surrogates=max(8, int(_SURROGATES_FLAG.value())),
                n_boot=max(4, int(_BOOTSTRAP_FLAG.value())),
                seed=seed,
                max_effective_elements=max(4, int(_MAX_ELEMENTS_FLAG.value())),
                extra_transitions=extra,
            )
            compute_seconds = time.perf_counter() - started
            est.diagnostics.update(matrix_diagnostics)
            with self._lock:
                self._latest = est
                self._since_estimate = 0
                self._estimates_done += 1
                self._last_error = ""
                self._last_seed = seed
                self._last_compute_seconds = compute_seconds
                self._last_matrix_diagnostics = matrix_diagnostics
            logger.info("WholeSystemPhi: %s", est.claim)
            return est
        except (ValueError, ArithmeticError, RuntimeError) as exc:
            with self._lock:
                self._last_error = f"{type(exc).__name__}: {exc}"
            record_degradation("whole_system_phi", exc, severity="warning",
                               action="whole-system Φ estimate skipped")
            return None
        finally:
            with self._lock:
                self._estimating = False

    def schedule_if_due(self) -> bool:
        """Schedule due work without joining it to the cognition phase."""
        with self._lock:
            if self._shutting_down or not self.ready():
                return False
            if self._estimate_task is not None and not self._estimate_task.done():
                return False
            if self._probe_task is not None and not self._probe_task.done():
                return False
        try:
            asyncio.get_running_loop()
            from core.utils.task_tracker import get_task_tracker

            task = get_task_tracker().create_task(
                self._run_scheduled_estimate(),
                name="whole_system_phi.estimate",
            )
        except (ImportError, RuntimeError) as exc:
            record_degradation(
                "whole_system_phi",
                exc,
                severity="debug",
                action="deferred estimate because no owned event loop was available",
            )
            return False
        with self._lock:
            self._estimate_task = task
        return True

    def schedule_maintenance(self) -> bool:
        """Admit at most one estimate or causal campaign this cycle."""
        if self.schedule_if_due():
            return True
        return self.schedule_probe_if_due()

    def schedule_probe_if_due(self) -> bool:
        now = time.time()
        with self._lock:
            due = (
                bool(_PROBE_ENABLED_FLAG.value())
                and not self._shutting_down
                and self._estimates_done > 0
                and now - self._created_at
                >= max(0.0, float(_PROBE_INITIAL_DELAY_FLAG.value()))
                and (
                    self._last_probe_at <= 0.0
                    or now - self._last_probe_at
                    >= max(60.0, float(_PROBE_INTERVAL_FLAG.value()))
                )
                and (
                    self._last_probe_attempt_at <= 0.0
                    or now - self._last_probe_attempt_at
                    >= max(60.0, float(_PROBE_RETRY_FLAG.value()))
                )
                and not (
                    self._estimate_task is not None
                    and not self._estimate_task.done()
                )
                and not (
                    self._probe_task is not None and not self._probe_task.done()
                )
            )
            if not due:
                return False
        try:
            asyncio.get_running_loop()
            from core.utils.task_tracker import get_task_tracker

            task = get_task_tracker().create_task(
                self._run_scheduled_probe(),
                name="whole_system_phi.perturbational_probe",
            )
        except (ImportError, RuntimeError) as exc:
            record_degradation(
                "whole_system_phi",
                exc,
                severity="debug",
                action="deferred causal probe because no owned event loop was available",
            )
            return False
        with self._lock:
            self._probe_task = task
            self._last_probe_attempt_at = now
        return True

    async def _run_scheduled_estimate(self) -> None:
        lease_id = ""
        admission = None
        try:
            admission = optional_service("resource_admission", default=None)
            if admission is None or not hasattr(admission, "acquire"):
                with self._lock:
                    self._admission_deferrals += 1
                    self._last_admission_reason = "resource_admission_unavailable"
                return
            from core.runtime.control_plane import (
                AdmissionPriority,
                AdmissionRequest,
                WorkClass,
            )

            decision = await admission.acquire(AdmissionRequest(
                owner="whole_system_phi",
                work_class=WorkClass.BACKGROUND,
                lane="causal_measurement",
                priority=AdmissionPriority.BACKGROUND,
                timeout_s=0.0,
                lease_ttl_s=120.0,
                preemptible=False,
                receipt_required=False,
                metadata={"operation": "estimate_whole_system_phi"},
            ))
            if not decision.admitted:
                with self._lock:
                    self._admission_deferrals += 1
                    self._last_admission_reason = decision.reason
                return
            lease_id = decision.lease_id
            with self._lock:
                self._last_admission_reason = "admitted"
            estimate = await asyncio.to_thread(self.maybe_estimate)
            if estimate is not None:
                await self.persist_latest()
        except asyncio.CancelledError:
            raise
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            with self._lock:
                self._last_error = f"{type(exc).__name__}: {exc}"
            record_degradation(
                "whole_system_phi",
                exc,
                severity="warning",
                action="scheduled whole-system estimate failed",
            )
        finally:
            if lease_id:
                try:
                    if admission is not None:
                        await asyncio.shield(
                            admission.release(lease_id, reason="phi_estimate_completed")
                        )
                except (KeyError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
                    record_degradation(
                        "whole_system_phi",
                        exc,
                        severity="debug",
                        action="phi estimate lease release reconciled by expiry",
                    )
            with self._lock:
                if self._estimate_task is asyncio.current_task():
                    self._estimate_task = None

    async def _run_scheduled_probe(self) -> None:
        lease_id = ""
        admission = None
        try:
            admission = optional_service("resource_admission", default=None)
            if admission is None or not hasattr(admission, "acquire"):
                with self._lock:
                    self._probe_admission_deferrals += 1
                    self._last_probe_admission_reason = "resource_admission_unavailable"
                return
            from core.runtime.control_plane import (
                AdmissionPriority,
                AdmissionRequest,
                WorkClass,
            )

            decision = await admission.acquire(AdmissionRequest(
                owner="whole_system_phi_probe",
                work_class=WorkClass.BACKGROUND,
                lane="causal_measurement",
                priority=AdmissionPriority.BACKGROUND,
                timeout_s=0.0,
                lease_ttl_s=180.0,
                preemptible=False,
                receipt_required=False,
                metadata={"operation": "perturbational_probe_campaign"},
            ))
            if not decision.admitted:
                with self._lock:
                    self._probe_admission_deferrals += 1
                    self._last_probe_admission_reason = decision.reason
                return
            lease_id = decision.lease_id
            with self._lock:
                self._last_probe_admission_reason = "admitted"

            from core.consciousness.perturbational_probe import PerturbationalProbe

            def sample() -> dict[str, float]:
                with self._lock:
                    state = self._latest_runtime_state
                return self.sample_runtime_channels(state)

            campaign = await asyncio.to_thread(
                PerturbationalProbe(sampler=sample).run_campaign,
                trials=max(3, int(_PROBE_TRIALS_FLAG.value())),
                n_baseline=30,
                n_response=30,
                n_recovery=15,
                interval_s=0.02,
            )
            self.add_interventional_transitions(
                campaign.transitions,
                campaign.to_dict(),
                channel_names=campaign.channel_names,
            )
            if campaign.trials_completed:
                with self._lock:
                    self._last_probe_at = time.time()
        except asyncio.CancelledError:
            raise
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation(
                "whole_system_phi",
                exc,
                severity="warning",
                action="scheduled causal probe campaign failed",
            )
        finally:
            if lease_id:
                try:
                    if admission is not None:
                        await asyncio.shield(
                            admission.release(lease_id, reason="phi_probe_completed")
                        )
                except (KeyError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
                    record_degradation(
                        "whole_system_phi",
                        exc,
                        severity="debug",
                        action="causal probe lease release reconciled by expiry",
                    )
            with self._lock:
                if self._probe_task is asyncio.current_task():
                    self._probe_task = None

    async def on_stop_async(self) -> None:
        with self._lock:
            self._shutting_down = True
            tasks = [
                task for task in (self._estimate_task, self._probe_task)
                if task is not None and not task.done()
            ]
        if tasks:
            _done, pending = await asyncio.wait(tasks, timeout=12.0)
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

    async def persist_latest(self) -> str:
        """Write the latest report through the governed async write lane."""
        with self._lock:
            est = self._latest
            probe = dict(self._latest_probe)
        if est is None:
            return ""
        from pathlib import Path

        from core.governance_context import local_internal_governed_scope
        from core.runtime.file_write_gateway import get_file_write_gateway

        root = Path(str(_PHI_DIR_FLAG.value() or "")
                    or (state_root() / "data" / "phi"))
        target = root / "whole_system_latest.json"
        payload = {"estimate": est.to_dict(), "probe": probe}
        with local_internal_governed_scope("whole_system_phi",
                                           domain="state_mutation"):
            gateway = get_file_write_gateway()
            await gateway.ensure_directory_async(root, source="whole_system_phi")
            await gateway.write_json_async(
                target, payload, schema_version=1,
                schema_name="whole_system_phi_report",
                source="whole_system_phi",
            )
        return str(target)

    # ── observability ────────────────────────────────────────────────────
    def latest(self) -> Any:
        with self._lock:
            return self._latest

    def status(self) -> dict[str, Any]:
        with self._lock:
            est = self._latest
            return {
                "window": len(self._window),
                "estimates_done": self._estimates_done,
                "since_estimate": self._since_estimate,
                "interventional_rows": len(self._interventional),
                "last_error": self._last_error,
                "last_seed": self._last_seed,
                "last_compute_seconds": round(self._last_compute_seconds, 3),
                "estimate_task_active": bool(
                    self._estimate_task is not None and not self._estimate_task.done()
                ),
                "probe_task_active": bool(
                    self._probe_task is not None and not self._probe_task.done()
                ),
                "admission_deferrals": self._admission_deferrals,
                "last_admission_reason": self._last_admission_reason,
                "probe_admission_deferrals": self._probe_admission_deferrals,
                "last_probe_admission_reason": self._last_probe_admission_reason,
                "last_probe_at": self._last_probe_at,
                "last_probe_attempt_at": self._last_probe_attempt_at,
                "matrix_diagnostics": dict(self._last_matrix_diagnostics),
                "latest": est.to_dict() if est is not None else None,
                "latest_probe": dict(self._latest_probe),
            }

    def is_alive(self) -> bool:
        with self._lock:
            return not self._shutting_down


_service: WholeSystemPhiService | None = None
_service_lock = threading.Lock()


def get_whole_system_phi() -> WholeSystemPhiService:
    global _service
    if _service is None:
        with _service_lock:
            if _service is None:
                _service = WholeSystemPhiService()
    return _service


def boot_whole_system_phi() -> WholeSystemPhiService:
    service = get_whole_system_phi()
    try:
        from core.container import ServiceContainer

        ServiceContainer.register_instance("whole_system_phi", service,
                                           required=False)
    except (ImportError, RuntimeError, TypeError, ValueError) as exc:
        record_degradation("whole_system_phi", exc, severity="warning",
                           action="service built but not registered")
    return service
