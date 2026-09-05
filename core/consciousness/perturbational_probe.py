"""core/consciousness/perturbational_probe.py — internal PCI (Rail D).
=========================================================================
Causal grounding for whole-system Φ: passive covariance can be confounded
by common drivers, and IIT is about cause-effect power.  This module lets
Aura perturb HERSELF — a small, governed, reversible nudge through a real
seam — and measures how complexly the disturbance propagates across her
channels: an internal Perturbational Complexity Index (Casali et al. 2013,
adapted from TMS/EEG to runtime telemetry).

Also emits the observed (state_before, state_after) macro transitions so
the exact discrete estimator (Rail C in integrated_information.py) gains
interventional rows — counterfactual access by sampling, not enumeration.

Governance: every probe asks the Unified Will first (domain
STATE_MUTATION, source "phi_probe"); a refusal aborts the probe and is
reported, not swallowed.  Probes are for idle, calm moments — and a
Ulysses covenant can bind them outright under threat.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.PerturbationalProbe")

PROBE_ESTIMATOR_NAME = "internal_pci.lz76.v1"


# ─────────────────────────────────────────────────────────────────────────────
# Lempel–Ziv 76 complexity
# ─────────────────────────────────────────────────────────────────────────────

def lz76_complexity(sequence: str) -> int:
    """Number of distinct phrases in the LZ76 parsing of a binary string."""
    n = len(sequence)
    if n == 0:
        return 0
    phrases = 1
    prefix_end = 0      # end of the current phrase start
    match_len = 0
    pos = 0
    while prefix_end + match_len + 1 < n:
        window = sequence[: prefix_end + match_len]
        candidate = sequence[prefix_end: prefix_end + match_len + 1]
        if candidate in window:
            match_len += 1
        else:
            phrases += 1
            prefix_end += match_len + 1
            match_len = 0
        pos += 1
    return phrases


def normalized_lz(binary: np.ndarray) -> float:
    """LZ76 of a channel-major binary matrix under the standard LZc
    normalization: divide the phrase count by n/log2(n), the asymptotic
    LZ76 rate of a length-n random binary string.

    Values: ~0 = stereotyped or silent (a constant or empty response —
    the seizure/deep-anesthesia signature), rising toward ~1 as the
    response becomes spatiotemporally differentiated.  We deliberately do
    NOT divide by the source entropy (Casali's PCN step): entropy
    normalization inflates sparse responses, inverting the sparse-vs-rich
    ordering that PCI exists to measure.  Sparsity is instead carried
    explicitly by ``active_fraction`` in the report."""
    flat = "".join("1" if v else "0" for v in binary.T.reshape(-1))
    n = len(flat)
    if n < 8:
        return 0.0
    if flat.count("1") in (0, n):
        return 0.0
    denom = n / np.log2(n)
    return float(lz76_complexity(flat) / denom)


# ─────────────────────────────────────────────────────────────────────────────
# PCI from windows
# ─────────────────────────────────────────────────────────────────────────────

def pci_from_windows(baseline: np.ndarray, response: np.ndarray,
                     *, z_thresh: float = 2.0) -> dict[str, Any]:
    """Binarize the response against the baseline distribution per channel
    (|z| > z_thresh = significantly deflected) and compress.

    Returns raw LZ, normalized PCI, and the active fraction — with the
    baseline's own PCI under the same procedure as the built-in control
    (a quiet system must score near zero on itself)."""
    baseline = np.asarray(baseline, dtype=float)
    response = np.asarray(response, dtype=float)
    mu = baseline.mean(axis=0)
    sd = baseline.std(axis=0)
    sd[sd < 1e-9] = 1e-9
    resp_bin = (np.abs((response - mu) / sd) > z_thresh)
    base_bin = (np.abs((baseline - mu) / sd) > z_thresh)
    resp_lz = lz76_complexity("".join("1" if v else "0"
                                      for v in resp_bin.T.reshape(-1)))
    resp_active = float(resp_bin.mean())
    # evoked_complexity = raw phrase count × engaged fraction — the robust
    # three-way discriminant.  Sham (few sources) and stereotyped (few
    # phrases) both collapse to ~0; only a large AND differentiated response
    # scores high.  This is the number to trust when LZc-per-bit alone is
    # confounded by the fact that random noise compresses worse than
    # structure.
    return {
        "estimator": PROBE_ESTIMATOR_NAME,
        "pci": round(normalized_lz(resp_bin), 4),
        "pci_baseline_control": round(normalized_lz(base_bin), 4),
        "evoked_complexity": round(resp_lz * resp_active, 4),
        "lz_raw": resp_lz,
        "active_fraction": round(resp_active, 4),
        "n_channels": int(response.shape[1]),
        "n_response_samples": int(response.shape[0]),
        "z_thresh": z_thresh,
    }


# ─────────────────────────────────────────────────────────────────────────────
# The governed probe
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ProbeReport:
    ran: bool
    reason: str
    started_at: float = 0.0
    pci: dict[str, Any] = field(default_factory=dict)
    sham_pci: dict[str, Any] = field(default_factory=dict)
    transitions: list[tuple[tuple[int, ...], tuple[int, ...]]] = field(default_factory=list)
    will_receipt_id: str = ""
    channel_names: tuple[str, ...] = ()
    recovery: dict[str, Any] = field(default_factory=dict)
    perturbation_seam: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "ran": self.ran,
            "reason": self.reason,
            "started_at": self.started_at,
            "pci": dict(self.pci),
            "sham_pci": dict(self.sham_pci),
            "n_interventional_transitions": len(self.transitions),
            "will_receipt_id": self.will_receipt_id,
            "channel_names": list(self.channel_names),
            "recovery": dict(self.recovery),
            "perturbation_seam": self.perturbation_seam,
        }


@dataclass(frozen=True)
class ReversiblePerturbation:
    delivered: bool
    restore: Callable[[], bool]
    seam: str


@dataclass
class ProbeCampaignReport:
    trials_requested: int
    trials_completed: int
    reports: list[ProbeReport]
    aggregate: dict[str, Any]
    channel_names: tuple[str, ...]
    transitions: list[tuple[tuple[int, ...], tuple[int, ...]]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "trials_requested": self.trials_requested,
            "trials_completed": self.trials_completed,
            "aggregate": dict(self.aggregate),
            "channel_names": list(self.channel_names),
            "n_interventional_transitions": len(self.transitions),
            "reports": [report.to_dict() for report in self.reports],
        }


def _default_perturb() -> ReversiblePerturbation | bool:
    """The default seam: a small, reversible affective nudge — the same
    apply_event surface the rest of the organism uses.  Returns True when a
    real perturbation was delivered."""
    try:
        from core.runtime.service_access import optional_service

        affect = optional_service("affect_engine", "affect_facade", default=None)
        if affect is not None and hasattr(affect, "apply_event"):
            affect.apply_event(0.06, 0.10)  # gentle valence+arousal impulse

            def restore() -> bool:
                affect.apply_event(-0.06, -0.10)
                return True

            return ReversiblePerturbation(
                delivered=True,
                restore=restore,
                seam="affect.apply_event.reversible_delta",
            )
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        record_degradation("perturbational_probe", exc, severity="warning",
                           action="default perturbation seam unavailable")
    return False


class PerturbationalProbe:
    """Runs governed perturb-and-measure trials against a channel sampler.

    ``sampler`` returns one {channel: value} reading per call; ``perturb``
    delivers the impulse and reports whether it actually fired.  Both are
    injectable — tests drive synthetic systems, the runtime uses the real
    seams.  ``clock``/``sleep`` are injectable for determinism.
    """

    def __init__(
        self,
        *,
        sampler: Callable[[], dict[str, float]],
        perturb: Callable[[], bool | ReversiblePerturbation] | None = None,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self._sampler = sampler
        self._perturb = perturb or _default_perturb
        self._clock = clock
        self._sleep = sleep

    def _collect(
        self,
        n: int,
        interval_s: float,
        *,
        channel_names: tuple[str, ...] | None = None,
    ) -> tuple[np.ndarray, tuple[str, ...]]:
        rows: list[list[float]] = []
        names = channel_names
        for _ in range(n):
            reading = self._sampler() or {}
            if names is None:
                names = tuple(sorted(reading))
            if len(names) < 2:
                raise ValueError("probe sampler exposed fewer than two channels")
            missing = [name for name in names if name not in reading]
            if missing:
                raise ValueError(
                    "probe channel set changed during a trial: " + ",".join(missing)
                )
            rows.append([float(reading[name]) for name in names])
            if interval_s > 0:
                self._sleep(interval_s)
        return np.asarray(rows, dtype=float), names or ()

    def _authorize(self) -> tuple[bool, str, str]:
        """Ask the Unified Will; fail closed if it cannot be consulted."""
        try:
            from core.governance.will import ActionDomain, get_will

            decision = get_will().decide(
                content="perturbational probe: small reversible affect impulse "
                        "to measure whole-system response complexity",
                source="phi_probe",
                domain=ActionDomain.STATE_MUTATION,
                priority=0.2,
            )
            return decision.is_approved(), decision.reason, decision.receipt_id
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation("perturbational_probe", exc, severity="warning",
                               action="probe refused because the Will was unavailable")
            return False, f"will_unavailable:{type(exc).__name__}", ""

    def run(self, *, n_baseline: int = 40, n_response: int = 40,
            n_recovery: int = 20, interval_s: float = 0.05,
            with_sham: bool = True) -> ProbeReport:
        approved, reason, receipt_id = self._authorize()
        if not approved:
            return ProbeReport(ran=False, reason=f"refused: {reason}",
                               will_receipt_id=receipt_id)
        started = self._clock()

        effect: ReversiblePerturbation | None = None
        restored = False
        try:
            baseline, names = self._collect(n_baseline, interval_s)
            sham: dict[str, Any] = {}
            if with_sham:
                sham_resp, _ = self._collect(
                    n_response,
                    interval_s,
                    channel_names=names,
                )
                sham = pci_from_windows(baseline, sham_resp)

            delivered = self._perturb()
            if isinstance(delivered, ReversiblePerturbation):
                effect = delivered
                did_deliver = delivered.delivered
            else:
                did_deliver = bool(delivered)
            if not did_deliver:
                return ProbeReport(
                    ran=False,
                    reason="perturbation seam unavailable",
                    started_at=started,
                    will_receipt_id=receipt_id,
                    channel_names=names,
                )
            response, _ = self._collect(
                n_response,
                interval_s,
                channel_names=names,
            )
            pci = pci_from_windows(baseline, response)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation(
                "perturbational_probe",
                exc,
                severity="warning",
                action="probe trial aborted without accepting a partial measurement",
            )
            return ProbeReport(
                ran=False,
                reason=f"trial_failed:{type(exc).__name__}",
                started_at=started,
                will_receipt_id=receipt_id,
            )
        finally:
            if effect is not None and effect.delivered:
                try:
                    restored = bool(effect.restore())
                except (OSError, RuntimeError, TypeError, ValueError) as exc:
                    record_degradation(
                        "perturbational_probe",
                        exc,
                        severity="warning",
                        action="reversible probe restoration failed",
                    )

        recovery: dict[str, Any] = {
            "restore_available": effect is not None,
            "restore_succeeded": restored,
        }
        if n_recovery > 0:
            try:
                recovery_window, _ = self._collect(
                    n_recovery,
                    interval_s,
                    channel_names=names,
                )
                baseline_sd = baseline.std(axis=0)
                baseline_sd[baseline_sd < 1e-9] = 1e-9
                recovery_z = np.abs(
                    (np.median(recovery_window, axis=0) - baseline.mean(axis=0))
                    / baseline_sd
                )
                median_abs_z = float(np.median(recovery_z))
                recovery.update({
                    "median_abs_z": round(median_abs_z, 4),
                    "within_baseline_envelope": median_abs_z <= 1.5,
                    "n_samples": int(n_recovery),
                })
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                recovery.update({
                    "within_baseline_envelope": False,
                    "error": f"{type(exc).__name__}: {exc}",
                })

        # interventional macro transitions for the exact estimator (Rail C):
        # the medians of baseline vs response windows, binarized on baseline.
        med = np.median(baseline, axis=0)
        pre = tuple(int(v) for v in (baseline[-1] > med))
        post = tuple(int(v) for v in (np.median(response, axis=0) > med))
        report = ProbeReport(
            ran=True,
            reason="completed",
            started_at=started,
            pci=pci,
            sham_pci=sham,
            transitions=[(pre, post)],
            will_receipt_id=receipt_id,
            channel_names=names,
            recovery=recovery,
            perturbation_seam=effect.seam if effect is not None else "injected_test_seam",
        )
        logger.info("PerturbationalProbe: PCI=%.3f (sham %.3f) active=%.2f",
                    pci.get("pci", 0.0), sham.get("pci", 0.0),
                    pci.get("active_fraction", 0.0))
        return report

    def run_campaign(
        self,
        *,
        trials: int = 3,
        n_baseline: int = 40,
        n_response: int = 40,
        n_recovery: int = 20,
        interval_s: float = 0.05,
    ) -> ProbeCampaignReport:
        reports = [
            self.run(
                n_baseline=n_baseline,
                n_response=n_response,
                n_recovery=n_recovery,
                interval_s=interval_s,
                with_sham=True,
            )
            for _ in range(max(1, int(trials)))
        ]
        completed = [report for report in reports if report.ran]
        channel_names = completed[0].channel_names if completed else ()
        compatible = [
            report for report in completed if report.channel_names == channel_names
        ]
        transitions = [
            transition
            for report in compatible
            for transition in report.transitions
        ]
        evoked_delta = np.asarray([
            report.pci.get("evoked_complexity", 0.0)
            - report.sham_pci.get("evoked_complexity", 0.0)
            for report in compatible
        ], dtype=float)
        pci_delta = np.asarray([
            report.pci.get("pci", 0.0) - report.sham_pci.get("pci", 0.0)
            for report in compatible
        ], dtype=float)

        def interval(values: np.ndarray) -> tuple[float, float]:
            if not values.size:
                return 0.0, 0.0
            if values.size == 1:
                return float(values[0]), float(values[0])
            rng = np.random.default_rng(0xA17A)
            draws = np.asarray([
                rng.choice(values, size=values.size, replace=True).mean()
                for _ in range(1000)
            ])
            return float(np.percentile(draws, 5)), float(np.percentile(draws, 95))

        evoked_lo, evoked_hi = interval(evoked_delta)
        pci_lo, pci_hi = interval(pci_delta)
        recovered = sum(
            bool(report.recovery.get("within_baseline_envelope"))
            for report in compatible
        )
        aggregate = {
            "evoked_complexity_delta_mean": round(
                float(evoked_delta.mean()) if evoked_delta.size else 0.0, 4
            ),
            "evoked_complexity_delta_ci_5": round(evoked_lo, 4),
            "evoked_complexity_delta_ci_95": round(evoked_hi, 4),
            "pci_delta_mean": round(
                float(pci_delta.mean()) if pci_delta.size else 0.0, 4
            ),
            "pci_delta_ci_5": round(pci_lo, 4),
            "pci_delta_ci_95": round(pci_hi, 4),
            "recovered_trials": recovered,
            "causal_response_established": bool(
                len(compatible) >= 3
                and evoked_lo > 0.0
                and recovered == len(compatible)
            ),
        }
        return ProbeCampaignReport(
            trials_requested=max(1, int(trials)),
            trials_completed=len(compatible),
            reports=reports,
            aggregate=aggregate,
            channel_names=channel_names,
            transitions=transitions,
        )
