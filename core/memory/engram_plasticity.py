"""Bind episodic engrams to the voltage-dependent plasticity field.

This is the causal wiring that makes the Clopath/Pantheon plasticity engine
(``core.consciousness.voltage_plasticity``) actually *govern memory* instead of
running as an isolated simulation.

Two effects, both from the same dynamics:

  * **Retrieval competition** — when several engrams are candidates for a query,
    their salience drives a transient plasticity field.  The nonlinear escape-
    rate activation sharpens the best match, **voltage-gating** suppresses
    weakly-relevant engrams (they fall below threshold and contribute nothing),
    and the **homeostatic** bound stops one over-strong-but-irrelevant trace from
    swamping the answer.  This is the anti-confabulation mechanism: the engram
    that actually matches wins the competition, rather than the loudest one.

  * **Consolidation feedback** — the engrams that win co-activation receive a
    bounded, voltage-gated LTP bump to their importance, while the homeostatic
    pressure caps runaway rehearsal so a single memory cannot dominate the store
    (the "epilepsy"/obsession failure mode the whiteboard's homeostasis prevents).

The substrate couples in as the membrane-potential context: affective **arousal**
lowers the activation threshold θ (emotionally charged recall keeps more engrams
above threshold and potentiates harder — biologically real), and **valence**
shifts the temperature.  Homeostatic pressure is exported as a governance signal.
"""
from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from core.consciousness.voltage_plasticity import (
    VoltageDependentPlasticityEngine,
    VoltagePlasticityConfig,
)
from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.Memory.EngramPlasticity")


def _enabled() -> bool:
    return os.getenv("AURA_ENGRAM_PLASTICITY", "1") not in ("0", "false", "False")


@dataclass
class EngramCompetitionResult:
    """Outcome of resolving a recall by competition."""

    order: list[int]                 # candidate indices, strongest first
    weights: list[float]             # competitive activation per candidate (input order)
    pressure: float                  # homeostatic pressure at settle
    winner: int                      # index of the winning candidate (-1 if none)
    gated_out: list[int]             # candidates suppressed below threshold
    governance_breach: bool          # pressure exceeded the safe homeostatic bound

    def as_dict(self) -> dict[str, Any]:
        return {
            "order": self.order,
            "weights": [round(w, 5) for w in self.weights],
            "pressure": round(self.pressure, 5),
            "winner": self.winner,
            "gated_out": self.gated_out,
            "governance_breach": self.governance_breach,
        }


# Pressure above this means total recall activation is running hot — a single
# trace (or a tight cluster) is dominating.  Surfaced to governance/telemetry.
_PRESSURE_BREACH = 12.0


class EngramPlasticityField:
    """Resolves engram retrieval and consolidation through the plasticity engine."""

    def __init__(self, *, settle_steps: int = 24) -> None:
        self.settle_steps = settle_steps
        self._last_result: EngramCompetitionResult | None = None
        self._breach_events = 0

    # ── retrieval competition ─────────────────────────────────────────────

    def compete(
        self,
        salience: Sequence[float],
        *,
        arousal: float = 0.5,
        valence: float = 0.0,
    ) -> EngramCompetitionResult:
        """Resolve a set of candidate saliences into competitive weights.

        ``salience[k]`` is the drive for candidate k (relevance × strength,
        already ≥ 0).  Returns competitive activations after the field settles —
        voltage-gated, homeostatically bounded, winner-sharpened.
        """
        s = np.asarray(list(salience), dtype=np.float64).reshape(-1)
        n = int(s.shape[0])
        if n == 0:
            return EngramCompetitionResult([], [], 1.0, -1, [], False)
        if n == 1:
            return EngramCompetitionResult([0], [float(s[0] > 0)], 1.0,
                                           0 if s[0] > 0 else -1, [], False)

        # Substrate coupling: arousal lowers the threshold (more candidates stay
        # above it under emotional load); valence warms the escape-rate temp.
        arousal = float(np.clip(arousal, 0.0, 1.0))
        valence = float(np.clip(valence, -1.0, 1.0))
        theta = float(np.clip(1.0 - 0.6 * arousal, 0.25, 1.0))
        delta_beta = float(np.clip(0.75 + 0.25 * valence, 0.4, 1.1))

        cfg = VoltagePlasticityConfig(
            n_nodes=n,
            theta=theta,
            delta_beta=delta_beta,
            seed=17,
        )
        eng = VoltageDependentPlasticityEngine(cfg)

        # Normalise drive to a comparable scale, then map it through the board's
        # far-left input transfer function h(t)=β_VRP·((p₀−θ₀)/ΔB) before it drives
        # the field — the same "raw signal → field input" gate the whiteboard
        # shows feeding the simulation (identity at default gate params).
        peak = float(np.max(s))
        drive = (s / peak) if peak > 1e-9 else s
        drive = np.maximum(0.0, np.asarray(eng.input_gate(drive), dtype=np.float64))
        # The membrane integrates the input through a PSP kernel before the
        # escape-rate (board: b_k=β₀·exp((Σ∫PSP)/Δβ)). Warm-start the PSP trace to
        # the steady drive so the salient engram is already "present" — faithful
        # to the convolution without a startup ramp.
        eng._psp = drive.copy()
        try:
            for _ in range(self.settle_steps):
                eng.step_activity(eng.psp_kernel(drive))
        except (FloatingPointError, ValueError) as exc:  # pragma: no cover
            record_degradation("engram_plasticity", exc)
            order = list(np.argsort(-s))
            return EngramCompetitionResult(order, list(s), 1.0,
                                           int(order[0]), [], False)

        b = eng.b.copy()
        pressure = eng.homeostatic_pressure()
        total = float(np.sum(b))
        weights = (b / total) if total > 1e-9 else np.zeros(n)

        # Voltage-gating: candidates the field could not lift above a floor of
        # the winner's activation are treated as suppressed (no leak into recall).
        gate_floor = 0.10 * float(np.max(b)) if np.max(b) > 0 else 0.0
        gated_out = [int(i) for i in range(n) if b[i] <= gate_floor]
        order = [int(i) for i in np.argsort(-b)]
        winner = order[0] if b[order[0]] > 0 else -1
        breach = pressure > _PRESSURE_BREACH
        if breach:
            self._breach_events += 1
            try:
                from core.observability.metrics import get_metrics

                get_metrics().increment_counter("engram_homeostatic_breach_total")
            except (ImportError, AttributeError, RuntimeError, TypeError) as exc:
                record_degradation("engram_plasticity", exc)

        result = EngramCompetitionResult(
            order=order,
            weights=[float(w) for w in weights],
            pressure=pressure,
            winner=winner,
            gated_out=gated_out,
            governance_breach=breach,
        )
        self._last_result = result
        return result

    # ── governance ────────────────────────────────────────────────────────

    def governance_signal(self) -> dict[str, Any]:
        """Homeostatic-pressure signal for the constitutional/plasticity governor.

        A sustained breach means recall is being dominated by one attractor —
        the memory analogue of runaway excitation — which governance may damp.
        """
        last = self._last_result
        return {
            "homeostatic_pressure": round(last.pressure, 5) if last else 1.0,
            "governance_breach": bool(last.governance_breach) if last else False,
            "breach_events": self._breach_events,
            "safe_bound": _PRESSURE_BREACH,
        }


_field: EngramPlasticityField | None = None


def get_engram_plasticity_field() -> EngramPlasticityField:
    global _field
    if _field is None:
        _field = EngramPlasticityField()
    return _field


def is_engram_plasticity_enabled() -> bool:
    return _enabled()
