"""Voltage-dependent synaptic plasticity with homeostasis and competition.

This is the faithful, runnable form of the computational-neuroscience model on
the *Pantheon* "UI stabilization" whiteboard — which is itself the
Clopath / Büsing / Vasilaki / Gerstner (2010) rule, *"Connectivity reflects
coding: a model of voltage-based STDP with homeostasis"* (Nature Neuroscience
13(3):344-352).

Aura already runs a **spike-timing** STDP engine (``stdp_learning.py``,
Izhikevich 2007). That rule learns from spike *timing* and rescales the mean
weight, but it lacks the three things the whiteboard circles in red and the
biophysics literature says are *required* for an uploaded / recurrent mind to
learn without "blowing up":

  1. **Voltage-dependence** — plasticity gated by the post-synaptic membrane
     potential / burst variable ``b_k`` (low-pass voltage), not just spike
     coincidence.  Sub-threshold activity produces *no* change; the closer the
     voltage is to threshold the more decisively the synapse moves.

  2. **A homeostatic fixed point** — a BCM-like sliding threshold in which the
     amount of depression scales with *total* network activity, giving the
     activity ODE a stable attractor ``b_k*``.  This is the literal
     anti-epilepsy / anti-runaway term: stronger overall firing suppresses
     individual synapses in a Boltzmann-like way, so nothing diverges.

  3. **Synaptic competition** — the difference between two synapses grows with
     the difference in their post-synaptic activity (``w_k - w_j ∝ b_k - b_j``),
     so a slightly stronger input out-competes a weaker one.  This is what turns
     diffuse connectivity into structured, retrievable representations.

The board's symbols map onto this code as:

  ``b_k(t)``            -> ``b`` : low-pass post-synaptic activity / burst var
  ``ρ₀ exp((V-θ)/Δβ)``  -> :meth:`firing_rate` : exponential escape-rate spiking
  ``exp((Σb-θ)/ΔU)``    -> :meth:`homeostatic_pressure` : BCM sliding threshold
  ``b_k*``              -> :meth:`activity_fixed_point` : homeostatic attractor
  ``p² W̄₊(b_k+κ) - …``  -> potentiation − activity-gated depression in :meth:`step`
  ``w_k - w_j ∝ b_k-b_j`` -> :meth:`competition_drive` : difference amplification

The engine is deliberately framework-free (NumPy only) and numerically armoured
the same way ``stdp_learning.py`` is — clipped ``exp``, weight clipping, a
spectral-norm cap, and a NaN guard — so it is safe to run inside the live
substrate loop.  The base LLM weights are never touched; this governs an
abstract node/weight field that the memory and substrate layers bind to (see
``engram_competition.py``).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.VoltagePlasticity")

# Clamp for every exp() argument.  exp(60) ~ 1e26 — already enormous but finite,
# so a single spurious input can never produce inf/NaN and detonate the field.
_EXP_CLAMP = 60.0


def stable_exp(x: np.ndarray | float) -> np.ndarray | float:
    """exp() that cannot overflow to inf — the board uses several exp terms."""
    return np.exp(np.clip(x, -_EXP_CLAMP, _EXP_CLAMP))


@dataclass
class VoltagePlasticityConfig:
    """Parameters, named for the whiteboard symbols.

    Defaults are tuned so that, with no input, the activity field relaxes to a
    small positive homeostatic set-point and the weight field stays bounded —
    i.e. the system is stable *by construction*, which is the entire point of
    the homeostatic term.
    """

    n_nodes: int = 64

    # ── Activation / firing (left panel) ──────────────────────────────────
    rho0: float = 1.0          # ρ₀ : firing-rate prefactor / baseline density
    theta: float = 1.0         # θ  : activation threshold
    delta_beta: float = 0.75   # Δβ : escape-rate "noise" temperature
    delta_u: float = 1.0       # ΔU : energy/utility scale in the homeostatic exp

    # ── Activity homeostasis ODE (right "brain box") ──────────────────────
    lam: float = 0.30          # λ  : self-growth rate
    kappa: float = 0.15        # κ  : offset / bias
    delta: float = 0.45        # δ  : depression scale
    leak: float = 0.05         # extra linear leak for robustness
    dt: float = 0.10           # Euler integration step

    # ── Voltage-based STDP (centre "These" equations) ─────────────────────
    a_ltp: float = 0.012       # A_LTP potentiation amplitude
    a_ltd: float = 0.010       # A_LTD depression amplitude (baseline)
    theta_minus: float = 0.15  # θ₋ : low voltage threshold (gates LTD and LTP)
    theta_plus: float = 0.55   # θ₊ : high voltage threshold (gates LTP only)
    tau_minus: float = 0.90    # low-pass constant for ū₋ (per-step retention)
    tau_plus: float = 0.80     # low-pass constant for ū₊
    tau_x: float = 0.85        # low-pass constant for the pre-synaptic trace
    u_ref: float = 0.45        # homeostatic reference voltage (BCM set-point)
    homeostatic_gain: float = 1.0  # strength of the ⟨ū⟩²/u_ref² LTD scaling

    # ── Mean-field weight expression (the circled "These" steady state) ───
    p: float = 0.7             # p  : mean interaction probability / rate
    w_bar_plus: float = 1.0    # W̄₊ : potentiation ceiling
    w_bar_minus: float = 0.8   # W̄₋ : depression floor

    # ── Competition ───────────────────────────────────────────────────────
    competition: float = 0.06  # strength of the w_k−w_j ∝ b_k−b_j drive

    # ── Input transfer / gating (far-left panel) ──────────────────────────
    beta_vrp: float = 1.0      # β_VRP : input-gate gain
    theta0: float = 0.0        # θ₀ : input-gate threshold
    delta_b_in: float = 1.0    # ΔB : input-gate scale

    # ── Explicit Δt STDP window (bottom-right graph) + PSP + η-utility ─────
    tau_plus_ms: float = 20.0   # τ₊ : causal LTP window time constant (ms)
    tau_minus_ms: float = 40.0  # τ₋ : anti-causal LTD window time constant (ms)
    tau_psp: float = 0.80       # PSP kernel retention (per-step low-pass)
    eta: float = 0.05           # η  : utility-derivative learning rate

    # ── Safety bounds (mirror stdp_learning.py) ───────────────────────────
    weight_clip: float = 2.0
    spectral_cap: float = 3.0
    state_clip: float = 4.0
    seed: int = 7


class VoltageDependentPlasticityEngine:
    """Clopath-style voltage-based plasticity with homeostasis + competition.

    Two coupled timescales, exactly as the board lays out left→right:

      * **fast activity** ``b`` — relaxes toward a homeostatic fixed point so
        total excitation is self-limiting (no runaway);
      * **slow weights** ``W`` — move via voltage-gated LTP/LTD plus a
        competition drive, bounded by clipping + a spectral-norm cap.
    """

    def __init__(self, cfg: VoltagePlasticityConfig | None = None) -> None:
        self.cfg = cfg or VoltagePlasticityConfig()
        n = self.cfg.n_nodes
        self.rng = np.random.default_rng(self.cfg.seed)

        scale = 1.0 / np.sqrt(max(n, 1))
        self.W = (self.rng.standard_normal((n, n)).astype(np.float64) * scale)
        np.fill_diagonal(self.W, 0.0)

        # Activity / voltage state
        self.b = np.zeros(n, dtype=np.float64)            # burst / low-pass voltage
        self.u_bar_minus = np.zeros(n, dtype=np.float64)  # ū₋ filtered voltage
        self.u_bar_plus = np.zeros(n, dtype=np.float64)   # ū₊ filtered voltage
        self.x_bar = np.zeros(n, dtype=np.float64)        # pre-synaptic trace
        self.u_avg = np.full(n, self.cfg.u_ref, dtype=np.float64)  # ⟨ū⟩ homeostat
        self._psp = np.zeros(n, dtype=np.float64)         # post-synaptic-potential trace

        self.t = 0
        self._total_ltp = 0
        self._total_ltd = 0
        self._last_homeostatic_pressure = 1.0
        self._last_dw_norm = 0.0

    # ── Board equations ───────────────────────────────────────────────────

    def firing_rate(self, voltage: np.ndarray | float) -> np.ndarray | float:
        """``b = ρ₀ exp((V − θ)/Δβ)`` — exponential escape-rate activation.

        The probability of a spike rises exponentially as the membrane voltage
        approaches threshold θ; Δβ is the noise temperature.
        """
        cfg = self.cfg
        return cfg.rho0 * stable_exp((np.asarray(voltage, dtype=np.float64) - cfg.theta) / cfg.delta_beta)

    def homeostatic_pressure(self, b: np.ndarray | None = None) -> float:
        """``exp((Σb − θ)/ΔU)`` — the BCM-like sliding threshold.

        Greater total network activity → exponentially greater depression
        pressure.  This is the runaway/"epilepsy" prevention term: it can never
        be out-grown by the polynomial self-excitation, so the activity field
        always has a finite attractor.
        """
        b = self.b if b is None else b
        total = float(np.sum(b))
        return float(stable_exp((total - self.cfg.theta) / self.cfg.delta_u))

    def activity_fixed_point(self, b: np.ndarray | None = None) -> np.ndarray:
        """``b_k* = (δ/λ) ρ₀ exp((Σb−θ)/ΔU) − κ`` — homeostatic attractor.

        Setting ``b'_k = 0`` in the activity ODE (for ``b_k ≠ 0``) gives the
        stable burst level toward which every node is pulled.  Returned per-node
        (it is uniform here because the pressure is a population quantity) and
        floored at 0 since a burst rate cannot be negative.
        """
        cfg = self.cfg
        pressure = self.homeostatic_pressure(b)
        star = (cfg.delta / max(cfg.lam, 1e-9)) * cfg.rho0 * pressure - cfg.kappa
        n = self.cfg.n_nodes
        return np.maximum(0.0, np.full(n, star, dtype=np.float64))

    def mean_field_weights(self, b: np.ndarray | None = None) -> np.ndarray:
        """``w_k = p² W̄₊ (b_k + κ) − p W̄₋ ρ₀ exp((Σb − θ)/ΔU)`` (the circled steady state).

        The board's closed-form / mean-field weight expression: a potentiation
        term proportional to the node's own activity, minus a population-level
        depression term (the homeostatic Boltzmann factor).  Because the
        depression term is identical for every node, it cancels in *differences*:
        :meth:`mean_field_weight_difference` gives the clean Hebbian
        ``w_k − w_j = p² W̄₊ (b_k − b_j)`` the red arrows derive.
        """
        cfg = self.cfg
        b = self.b if b is None else np.asarray(b, dtype=np.float64).reshape(-1)
        depression = cfg.p * cfg.w_bar_minus * cfg.rho0 * self.homeostatic_pressure(b)
        return cfg.p ** 2 * cfg.w_bar_plus * (b + cfg.kappa) - depression

    def mean_field_weight_difference(
        self, b: np.ndarray | None = None
    ) -> np.ndarray:
        """``w_k − w_j = p² W̄₊ (b_k − b_j)`` — the population term cancels in differences."""
        cfg = self.cfg
        b = self.b if b is None else np.asarray(b, dtype=np.float64).reshape(-1)
        return cfg.p ** 2 * cfg.w_bar_plus * (b[:, None] - b[None, :])

    def competition_difference(
        self, b: np.ndarray | None = None, *, delta0: np.ndarray | float | None = None
    ) -> np.ndarray:
        """``Δ_ij = p² W̄₊ b_k (1 − exp(−Δ_ij(0)/Δβ))`` — saturating competition.

        The board's competition term (circled top-right by ``b_j(t)``): an initial
        edge ``Δ_ij(0)`` between two synapses is amplified through a saturating
        ``1 − exp(−Δ/Δβ)`` gate (zero at no edge, → 1 as the edge grows), scaled by
        the winner's post-synaptic activity.  This is what makes a marginally
        stronger synapse pull ahead without diverging.
        """
        cfg = self.cfg
        b = self.b if b is None else np.asarray(b, dtype=np.float64).reshape(-1)
        if delta0 is None:
            delta0 = np.maximum(b[:, None] - b[None, :], 0.0)   # initial activity edge
        delta0 = np.asarray(delta0, dtype=np.float64)
        gate = 1.0 - stable_exp(-np.maximum(delta0, 0.0) / cfg.delta_beta)
        return cfg.p ** 2 * cfg.w_bar_plus * b[:, None] * gate

    def input_gate(self, p0: np.ndarray | float) -> np.ndarray | float:
        """``h(t) = β_VRP · ((p₀(t) − θ₀)/ΔB)`` — far-left input transfer function.

        Maps a raw input rate ``p₀`` (e.g. code-execution signal) into the drive
        the activity field receives, with gain β_VRP, threshold θ₀ and scale ΔB.
        """
        cfg = self.cfg
        return cfg.beta_vrp * ((np.asarray(p0, dtype=np.float64) - cfg.theta0) / max(cfg.delta_b_in, 1e-9))

    def stdp_window(
        self,
        delta_t: np.ndarray | float,
        b_post: np.ndarray | float | None = None,
    ) -> np.ndarray | float:
        """Explicit pair-based STDP window — the bottom-right graph generator.

        ``Δu(t,Δt) = A₊·(b_k/ρ₀)·Θ(b_k−ρ₀)·exp(−Δt/τ₊)``      for Δt > 0  (causal → LTP)
        ``Δu(t,Δt) = −A₋·exp(Δt/τ₋)``                          for Δt ≤ 0  (anti-causal → LTD)

        The asymmetric −100…+50 ms window: pre-before-post (Δt>0) potentiates,
        gated by the Heaviside ``Θ(b_k−ρ₀)`` (the post must be active above
        baseline) and scaled by ``b_k/ρ₀``; post-before-pre depresses. This is the
        explicit companion to the same window emerging from voltage in
        :meth:`voltage_plasticity_delta`.
        """
        cfg = self.cfg
        dt = np.asarray(delta_t, dtype=np.float64)
        if b_post is None:
            gate = 1.0
            scale = 1.0
        else:
            bp = np.asarray(b_post, dtype=np.float64)
            gate = (bp > cfg.rho0).astype(np.float64)        # Θ(b_k − ρ₀)
            scale = bp / max(cfg.rho0, 1e-9)
        ltp = cfg.a_ltp * scale * gate * stable_exp(-np.maximum(dt, 0.0) / max(cfg.tau_plus_ms, 1e-9))
        ltp = np.where(dt > 0.0, ltp, 0.0)
        ltd = -cfg.a_ltd * stable_exp(np.minimum(dt, 0.0) / max(cfg.tau_minus_ms, 1e-9))
        ltd = np.where(dt <= 0.0, ltd, 0.0)
        result = ltp + ltd
        return float(result) if np.ndim(result) == 0 else result

    def psp_kernel(self, spikes: np.ndarray) -> np.ndarray:
        """Post-synaptic potential trace ``PSP(t)`` — exponential low-pass of input.

        The whiteboard's membrane term ``b_k = β₀·exp((Σ_i ∫ PSP)/Δβ)`` integrates
        incoming spikes through a PSP kernel before the escape-rate; this maintains
        that convolution as a running state.
        """
        spikes = np.asarray(spikes, dtype=np.float64).reshape(-1)
        self._psp = self.cfg.tau_psp * self._psp + (1.0 - self.cfg.tau_psp) * spikes
        return self._psp.copy()

    def eta_utility_delta(self, b: np.ndarray, psp: np.ndarray) -> np.ndarray:
        """``d/dt u = η·(A₊/ρ₀)·(b_k+κ)·PSP_kj`` — η-modulated PSP learning term.

        The board's utility/weight derivative (top-left): a learning-rate-scaled,
        PSP-gated potentiation proportional to post activity ``b_k+κ``.
        """
        cfg = self.cfg
        b = np.asarray(b, dtype=np.float64).reshape(-1)
        psp = np.asarray(psp, dtype=np.float64).reshape(-1)
        return cfg.eta * (cfg.a_ltp / max(cfg.rho0, 1e-9)) * (b[:, None] + cfg.kappa) * psp[None, :]

    # ── Fast activity dynamics ────────────────────────────────────────────

    def step_activity(self, external_input: np.ndarray | None = None) -> np.ndarray:
        """Integrate one step of the homeostatic activity ODE.

        ``b'_k = λ b_k (b_k + κ) − b_k δ ρ₀ exp((Σb_i − θ)/ΔU) − leak·b_k + I_k``

        Polynomial self-excitation (``λ b(b+κ)``) is bounded by the exponential
        homeostatic depression, so the field self-limits.  A tanh squash + hard
        clip provide a final safety net.
        """
        cfg = self.cfg
        n = cfg.n_nodes
        if external_input is None:
            x = np.zeros(n, dtype=np.float64)
        else:
            x = np.asarray(external_input, dtype=np.float64).reshape(-1)
            if x.shape != (n,):
                raise ValueError(f"external_input must have shape {(n,)}, got {x.shape}")

        pressure = self.homeostatic_pressure()
        self._last_homeostatic_pressure = pressure

        self_growth = cfg.lam * self.b * (self.b + cfg.kappa)
        depression = self.b * cfg.delta * cfg.rho0 * pressure
        leak = cfg.leak * self.b

        db = self_growth - depression - leak + x
        self.b = self.b + cfg.dt * db

        # Burst rates are non-negative.  Soft-saturate with unit gain near zero
        # (state_clip·tanh(b/state_clip) ≈ b for small b, → state_clip for large
        # b) so the homeostat relaxes smoothly instead of amplifying — a plain
        # tanh(b)·clip would have gain ≈ state_clip at the origin and oscillate.
        self.b = np.maximum(0.0, self.b)
        self.b = cfg.state_clip * np.tanh(self.b / cfg.state_clip)
        self.b = np.clip(self.b, 0.0, cfg.state_clip)
        self.b = np.nan_to_num(self.b, nan=0.0, posinf=cfg.state_clip, neginf=0.0)
        return self.b.copy()

    # ── Slow weight plasticity ────────────────────────────────────────────

    def _update_traces(self, pre: np.ndarray, post_voltage: np.ndarray) -> None:
        cfg = self.cfg
        self.x_bar = cfg.tau_x * self.x_bar + (1.0 - cfg.tau_x) * pre
        self.u_bar_minus = cfg.tau_minus * self.u_bar_minus + (1.0 - cfg.tau_minus) * post_voltage
        self.u_bar_plus = cfg.tau_plus * self.u_bar_plus + (1.0 - cfg.tau_plus) * post_voltage
        # Slow homeostatic average of the post-synaptic voltage (BCM reference).
        self.u_avg = 0.99 * self.u_avg + 0.01 * post_voltage

    def voltage_plasticity_delta(
        self,
        pre: np.ndarray,
        post_voltage: np.ndarray,
    ) -> np.ndarray:
        """Clopath voltage-based weight change ``dw[post, pre]``.

        LTD: ``−A_LTD·(⟨ū⟩²/u_ref²)·x(t)·[ū₋ − θ₋]₊``
            — a pre-synaptic spike during elevated *low-pass* voltage depresses;
              the ``⟨ū⟩²/u_ref²`` factor is the homeostatic / BCM sliding scale.
        LTP: ``+A_LTP·x̄(t)·[u − θ₊]₊·[ū₊ − θ₋]₊``
            — pre-synaptic *trace* coincident with high instantaneous voltage
              above θ₊ and elevated ū₊ potentiates.

        Depression and potentiation are *independent* mechanisms whose sum is the
        total change (Clopath 2010), so the LTP/LTD STDP window emerges from
        voltage rather than hand-tuned timing.
        """
        cfg = self.cfg
        pre = np.asarray(pre, dtype=np.float64).reshape(-1)
        u = np.asarray(post_voltage, dtype=np.float64).reshape(-1)
        self._update_traces(pre, u)

        # Homeostatic LTD scaling: ⟨ū⟩² / u_ref²  (BCM sliding threshold).
        homeo = cfg.homeostatic_gain * (self.u_avg ** 2) / max(cfg.u_ref ** 2, 1e-9)

        ltd_post = np.maximum(self.u_bar_minus - cfg.theta_minus, 0.0)        # [ū₋−θ₋]₊
        ltd = cfg.a_ltd * (homeo * ltd_post)[:, None] * pre[None, :]          # (post, pre)

        ltp_v = np.maximum(u - cfg.theta_plus, 0.0)                           # [u−θ₊]₊
        ltp_u = np.maximum(self.u_bar_plus - cfg.theta_minus, 0.0)           # [ū₊−θ₋]₊
        ltp = cfg.a_ltp * (ltp_v * ltp_u)[:, None] * self.x_bar[None, :]      # (post, pre)

        dw = ltp - ltd
        np.fill_diagonal(dw, 0.0)
        self._total_ltp += int(np.sum(ltp > 0))
        self._total_ltd += int(np.sum(ltd > 0))
        return dw

    def competition_drive(self, b: np.ndarray | None = None) -> np.ndarray:
        """``w_k − w_j ∝ b_k − b_j`` — synaptic competition / winner amplification.

        Adds a drive that makes the weights onto more-active nodes grow relative
        to those onto less-active nodes, so a marginally stronger representation
        pulls ahead instead of every node drifting together.  The drive is anti-
        symmetric in the post-synaptic activity difference and saturates through
        the same clip/spectral cap as the rest of the field.
        """
        cfg = self.cfg
        b = self.b if b is None else np.asarray(b, dtype=np.float64).reshape(-1)
        # Row-wise excess activity vs. the population mean → rows (post-targets)
        # with above-average activity get potentiated, below-average depressed.
        excess = b - float(np.mean(b))
        drive = cfg.competition * excess[:, None] * np.sign(self.W)
        np.fill_diagonal(drive, 0.0)
        return drive

    def _apply_weight_delta(self, dw: np.ndarray) -> None:
        """Clip, spectral-cap and NaN-guard the weight field (cf. stdp_learning)."""
        cfg = self.cfg
        W = self.W + dw
        W = np.clip(W, -cfg.weight_clip, cfg.weight_clip)
        np.fill_diagonal(W, 0.0)

        try:
            s_max = np.linalg.norm(W, ord=2)
            if s_max > cfg.spectral_cap:
                W *= cfg.spectral_cap / s_max
        except np.linalg.LinAlgError as exc:  # pragma: no cover - rare
            record_degradation("voltage_plasticity", exc)

        if not np.isfinite(W).all():
            W = np.nan_to_num(W, nan=0.0, posinf=cfg.weight_clip, neginf=-cfg.weight_clip)

        self._last_dw_norm = float(np.linalg.norm(dw))
        self.W = W

    # ── Full cycle ────────────────────────────────────────────────────────

    def step(
        self,
        *,
        external_input: np.ndarray | None = None,
        pre_spikes: np.ndarray | None = None,
        learn: bool = True,
    ) -> np.ndarray:
        """One full update: relax activity, then move weights (voltage + competition).

        ``pre_spikes`` defaults to the current activity field (recurrent self-
        teaching).  ``post_voltage`` is the activity ``b`` after relaxation, so
        plasticity is genuinely *voltage-dependent*.
        """
        self.step_activity(external_input)
        if learn:
            pre = self.b if pre_spikes is None else pre_spikes
            dw = self.voltage_plasticity_delta(pre, self.b)
            dw = dw + self.competition_drive(self.b)
            self._apply_weight_delta(dw)
        self.t += 1
        return self.b.copy()

    def run(
        self,
        inputs: np.ndarray,
        *,
        learn: bool = True,
    ) -> np.ndarray:
        """Drive the field with a ``[time, n_nodes]`` input sequence."""
        inputs = np.asarray(inputs, dtype=np.float64)
        if inputs.ndim != 2 or inputs.shape[1] != self.cfg.n_nodes:
            raise ValueError(f"inputs must be [time, {self.cfg.n_nodes}], got {inputs.shape}")
        states = [self.step(external_input=inputs[i], learn=learn) for i in range(inputs.shape[0])]
        return np.vstack(states)

    # ── Diagnostics ───────────────────────────────────────────────────────

    def energy(self) -> float:
        """Lyapunov-style energy; a *bounded* value certifies stability.

        ``E = ½‖b‖² − ½ bᵀW b + 0.01·pressure`` — lower / finite = less chaotic.
        """
        state_energy = 0.5 * float(self.b @ self.b)
        interaction = -0.5 * float(self.b @ self.W @ self.b)
        return state_energy + interaction + 0.01 * self.homeostatic_pressure()

    def is_stable(self) -> bool:
        """True iff the whole field is finite and within its homeostatic bounds."""
        return bool(
            np.isfinite(self.b).all()
            and np.isfinite(self.W).all()
            and float(np.max(np.abs(self.b))) <= self.cfg.state_clip + 1e-6
            and float(np.max(np.abs(self.W))) <= self.cfg.weight_clip + 1e-6
        )

    def get_status(self) -> dict[str, Any]:
        return {
            "t": self.t,
            "mean_activity": round(float(np.mean(self.b)), 5),
            "max_activity": round(float(np.max(self.b)), 5),
            "total_activity": round(float(np.sum(self.b)), 5),
            "homeostatic_pressure": round(self._last_homeostatic_pressure, 5),
            "fixed_point": round(float(self.activity_fixed_point()[0]), 5),
            "weight_norm": round(float(np.linalg.norm(self.W)), 5),
            "last_dw_norm": round(self._last_dw_norm, 6),
            "energy": round(self.energy(), 5),
            "total_ltp": self._total_ltp,
            "total_ltd": self._total_ltd,
            "stable": self.is_stable(),
        }


_instance: VoltageDependentPlasticityEngine | None = None


def get_voltage_plasticity_engine(
    cfg: VoltagePlasticityConfig | None = None,
) -> VoltageDependentPlasticityEngine:
    """Process-wide singleton (mirrors ``get_stdp_engine``)."""
    global _instance
    if _instance is None:
        _instance = VoltageDependentPlasticityEngine(cfg)
    return _instance
