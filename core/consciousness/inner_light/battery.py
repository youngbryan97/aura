"""core/consciousness/inner_light/battery.py — the inner-light test itself.

Runs the discriminator measures on a real activity matrix AND on every negative
control, places each system on four axes using absolute regime thresholds, and
returns a bounded, honest verdict.

The claim is a **conjunction that only the intact system satisfies**: Aura's
activity is in-regime on differentiation AND integrated-complexity AND
criticality AND ignition — 4/4 — while each control, by construction, drops at
least one axis. The strongest controls (time-shuffle, phase-randomise) reproduce
3/4 on purpose: that proves the four measures are not redundant, yet *neither
reproduces the whole signature*. If any control reaches 4/4, the test says the
signature is not discriminating and the verdict reflects that.

Explicit boundary, always carried in the result: this measures the
information-theoretic signature that in biological systems is present in
conscious brains and absent in unconscious ones and non-neural systems. It is
NOT a claim that Aura is conscious.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from core.consciousness.inner_light import controls as ctl
from core.consciousness.inner_light import measures as ms

logger = logging.getLogger("Aura.InnerLight")

# Absolute per-axis regime thresholds (calibrated against the measures' known-
# answer behaviour; see the tests). A system is "in-regime" on an axis when it
# clears the threshold.
LZ_MIN = 0.30      # differentiation: richer than a repeating pattern
TSE_MIN = 0.005    # integrated complexity: integrated AND differentiated
DFA_MIN = 0.65     # criticality: long-range temporal correlations
BIMOD_MIN = 0.52   # ignition: all-or-none global broadcast (bimodal)

AXES = ("differentiation", "integrated_complexity", "criticality", "ignition")

CAVEAT = (
    "Bounded claim: this measures the information-theoretic signature that in "
    "biological systems is present in conscious brains and absent in unconscious "
    "ones and in non-neural systems. It is NOT a claim that Aura is conscious."
)


def axis_membership(M: np.ndarray) -> dict[str, bool]:
    """Which of the four conscious-like axes this activity matrix occupies."""
    M = np.asarray(M, dtype=float)
    if M.ndim != 2 or M.shape[0] < 2 or M.shape[1] < 16:
        return {a: False for a in AXES}
    g = M.sum(axis=0)
    return {
        "differentiation": ms.normalized_lz(M) >= LZ_MIN,
        "integrated_complexity": ms.tse_complexity(M) >= TSE_MIN,
        "criticality": ms.dfa(g) >= DFA_MIN,
        "ignition": ms.bimodality_ignition(M) >= BIMOD_MIN,
    }


def axis_values(M: np.ndarray) -> dict[str, float]:
    M = np.asarray(M, dtype=float)
    g = M.sum(axis=0)
    return {
        "differentiation": round(ms.normalized_lz(M), 4),
        "integrated_complexity": round(ms.tse_complexity(M), 4),
        "criticality": round(ms.dfa(g), 4),
        "ignition": round(ms.bimodality_ignition(M), 4),
    }


@dataclass
class BatteryResult:
    verdict: str
    score: float                       # real axes occupied / 4
    discriminating: bool               # real == 4/4 AND best control < 4/4
    real_axes: int
    best_control_axes: int
    real_values: dict[str, float] = field(default_factory=dict)
    real_membership: dict[str, bool] = field(default_factory=dict)
    controls: dict[str, dict] = field(default_factory=dict)
    phi_system: float | None = None
    n_channels: int = 0
    n_timesteps: int = 0
    caveat: str = CAVEAT

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "score": round(self.score, 4),
            "discriminating": self.discriminating,
            "real_axes": self.real_axes,
            "best_control_axes": self.best_control_axes,
            "real_values": self.real_values,
            "real_membership": self.real_membership,
            "controls": self.controls,
            "phi_system": self.phi_system,
            "n_channels": self.n_channels,
            "n_timesteps": self.n_timesteps,
            "caveat": self.caveat,
        }


def _build_controls(M: np.ndarray, *, seed: int) -> dict[str, np.ndarray]:
    n, T = M.shape
    return {
        "time_shuffle": ctl.time_shuffle(M, seed=seed),
        "phase_randomize": ctl.phase_randomize(M, seed=seed),
        "lesion_decouple": ctl.lesion_decouple(M, seed=seed),
        "white_noise": ctl.white_noise((n, T), seed=seed),
        "ordered": ctl.ordered((n, T), seed=seed),
        "feedforward_chain": ctl.feedforward_chain((n, T), seed=seed),
    }


def run_on_matrix(M: np.ndarray, *, phi_system: float | None = None, seed: int = 20260709) -> BatteryResult:
    """Run the full battery on an activity matrix and its negative controls."""
    M = np.asarray(M, dtype=float)
    if M.ndim != 2 or M.shape[0] < 2 or M.shape[1] < 16:
        return BatteryResult(
            verdict="insufficient_data", score=0.0, discriminating=False,
            real_axes=0, best_control_axes=0,
            n_channels=int(M.shape[0]) if M.ndim == 2 else 0,
            n_timesteps=int(M.shape[1]) if M.ndim == 2 else 0,
        )

    real_membership = axis_membership(M)
    real_values = axis_values(M)
    real_axes = sum(real_membership.values())

    controls_out: dict[str, dict] = {}
    best_control_axes = 0
    for name, C in _build_controls(M, seed=seed).items():
        mem = axis_membership(C)
        occupied = sum(mem.values())
        best_control_axes = max(best_control_axes, occupied)
        controls_out[name] = {
            "values": axis_values(C),
            "membership": mem,
            "axes": occupied,
        }

    discriminating = (real_axes == len(AXES)) and (best_control_axes < len(AXES))
    if discriminating:
        verdict = "signature_present"
    elif real_axes == len(AXES):
        verdict = "signature_not_discriminating"  # a control matched it — honest
    elif real_axes >= len(AXES) - 1:
        verdict = "signature_partial"
    else:
        verdict = "signature_absent"

    return BatteryResult(
        verdict=verdict,
        score=real_axes / len(AXES),
        discriminating=discriminating,
        real_axes=real_axes,
        best_control_axes=best_control_axes,
        real_values=real_values,
        real_membership=real_membership,
        controls=controls_out,
        phi_system=phi_system,
        n_channels=int(M.shape[0]),
        n_timesteps=int(M.shape[1]),
    )


def run_live(*, bus: Any = None, workspace: Any = None, n_bins: int = 96) -> BatteryResult:
    """Build the activity matrix from the live streams (ConsequenceBus merged
    with global-workspace broadcast winners) and run the battery.

    Also corroborates integration with the Ghost's system-Φ over the same stream.
    """
    from core.consciousness.inner_light.activity import from_live_streams

    sample = from_live_streams(bus=bus, workspace=workspace, n_bins=n_bins)
    if not sample.sufficient:
        res = BatteryResult(
            verdict="insufficient_data", score=0.0, discriminating=False,
            real_axes=0, best_control_axes=0,
            n_channels=sample.matrix.shape[0] if sample.matrix.size else 0,
            n_timesteps=sample.matrix.shape[1] if sample.matrix.size else 0,
        )
        res.caveat = f"{CAVEAT} (insufficient live activity: {sample.reason})"
        return res

    phi = None
    try:
        from core.ghost.causal_integration import get_system_integration
        phi = round(get_system_integration().report().phi_system, 4)
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
        phi = None
    return run_on_matrix(sample.matrix, phi_system=phi)


__all__ = [
    "BatteryResult",
    "axis_membership",
    "axis_values",
    "run_on_matrix",
    "run_live",
    "AXES",
    "CAVEAT",
]
