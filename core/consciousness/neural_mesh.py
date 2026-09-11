"""core/consciousness/neural_mesh.py — Cortical Neural Mesh

A 4096-neuron dynamical substrate organized into 64 cortical columns of 64 neurons
each.  Three hierarchical tiers (sensory → association → executive) with biologically
realistic connectivity:

  • Dense intra-column recurrence  (p ≈ 0.8)
  • Sparse inter-column long-range (p ≈ 0.05, distance-weighted)
  • Lateral inhibition within columns via interneuron population
  • Spike-Timing-Dependent Plasticity (STDP) on every tick
  • Continuous ODE integration (Euler) at configurable rate

The mesh feeds a 64-dimensional *projection* back into the existing LiquidSubstrate,
so the original 64-neuron core becomes the executive summary of a much larger field.
"""
from __future__ import annotations

import asyncio
import logging
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from core.consciousness.mesh_wiring import CorticalTier, MeshWiring
from core.runtime.desktop_boot_safety import inprocess_mlx_metal_enabled
from core.runtime.errors import FallbackClassification, Severity, record_degradation
from core.utils.task_tracker import get_task_tracker

logger = logging.getLogger("Consciousness.NeuralMesh")

_RECOVERABLE_NEURAL_MESH_ERRORS = (
    AttributeError,
    ImportError,
    LookupError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)


def _record_neural_mesh_degradation(
    error: BaseException,
    *,
    action: str,
    severity: Severity = "degraded",
    extra: dict[str, Any] | None = None,
) -> None:
    """Record a visible, receipt-backed neural mesh degradation."""
    try:
        record_degradation(
            "neural_mesh",
            error,
            severity=severity,
            action=action,
            classification=FallbackClassification.SAFE_FALLBACK,
            receipt_required=True,
            extra=extra,
        )
    except TypeError:
        # Compatibility with legacy tests/adapters that monkeypatch the old
        # two-argument signature while the runtime migrates to receipt metadata.
        record_degradation("neural_mesh", error)


def _finite_float(raw: object, default: float) -> tuple[float, bool]:
    """Return a finite float plus whether the caller-provided value was valid."""
    try:
        value = float(raw)
    except (TypeError, ValueError, OverflowError):
        return default, False
    if not np.isfinite(value):
        return default, False
    return value, True


def _clamp_float(value: float, *, lower: float, upper: float) -> tuple[float, bool]:
    clamped = max(lower, min(upper, value))
    return clamped, clamped == value

# Metal acceleration: use MLX for the batched column matmul if available.
# MLX runs on Apple Metal GPU — same hardware as the LLM inference.
# Falls back to numpy einsum if MLX is not installed.
_MLX_ACCELERATOR = "numpy"
_MLX_ACCELERATOR_REASON = "mlx_unavailable"
_MLX_METAL_ENABLED, _MLX_ACCELERATOR_REASON = inprocess_mlx_metal_enabled()
if _MLX_METAL_ENABLED:
    try:
        import mlx.core as mx
        _HAS_MLX = True
        _MLX_ACCELERATOR = "metal"
        logger.info("NeuralMesh: MLX Metal enabled for batched matmuls.")
    except ImportError:
        _HAS_MLX = False
        _MLX_METAL_ENABLED = False
        _MLX_ACCELERATOR_REASON = "mlx_unavailable"
        mx = None
        logger.info("NeuralMesh: MLX unavailable; using NumPy fallback.")
else:
    _HAS_MLX = False
    mx = None
    logger.info(
        "NeuralMesh: MLX Metal disabled (%s); using NumPy fallback.",
        _MLX_ACCELERATOR_REASON,
    )

# ---------------------------------------------------------------------------
# Config & enums
# ---------------------------------------------------------------------------

#: Re-exported from the wiring module, which is where it is defined and where
#: every builder that reads it lives. Kept importable from here because that is
#: where the rest of the runtime has always found it.
CorticalTier = CorticalTier


def _from_cortex(name: str, fallback: float) -> float:
    """One derived mesh constant, or the value it had before the derivation.

    A mesh that will not build is worse than a mesh built on a number somebody
    picked, so every one of these falls back rather than raising.
    """
    try:
        from core.connectome.cortical_constants import derived_mesh_constants

        return float(derived_mesh_constants()[name]["value"])
    except (ImportError, KeyError, TypeError, ValueError):
        return fallback


def _cortical_intra_density() -> float:
    return _from_cortex("intra_column_density", 0.80)


def _cortical_inter_density() -> float:
    return _from_cortex("inter_column_density", 0.05)


def _cortical_leak() -> float:
    return _from_cortex("decay", 0.03)


def _cortical_stdp_window() -> float:
    return _from_cortex("stdp_window", 0.02)


def _cortical_stdp_asymmetry() -> float:
    return _from_cortex("stdp_depression", 0.5)


def _cortical_inhibitory_fraction() -> float:
    """Inhibitory cells as a share of a cortical column, from the published table.

    Falls back to the rounded 0.20 if the connectome package cannot be reached,
    because a mesh that will not build is worse than a mesh built on a rounding.
    """
    try:
        from core.connectome.types import CORTICAL_EXCITATORY, CORTICAL_INHIBITORY

        total = CORTICAL_EXCITATORY + CORTICAL_INHIBITORY
        return round(CORTICAL_INHIBITORY / total, 4) if total else 0.20
    except ImportError:
        return 0.20

_TIER_CONSTANTS: dict[str, dict[str, Any]] | None = None


def _tier_constants() -> dict[str, dict[str, Any]]:
    """What each tier's cortical layers say about its own wiring.

    Read once. A failed import leaves an empty table, and every caller falls
    back to the global figure, which is what the mesh used before this existed.
    """
    global _TIER_CONSTANTS

    if _TIER_CONSTANTS is None:
        try:
            from core.connectome.cortical_constants import derived_tier_constants

            _TIER_CONSTANTS = derived_tier_constants()
        except (ImportError, KeyError, TypeError, ValueError):
            _TIER_CONSTANTS = {}
    return _TIER_CONSTANTS


def _for_tier(tier: CorticalTier, name: str, fallback: float) -> float:
    """One tier's value for a constant, or the global one."""
    entry = _tier_constants().get(tier.name.lower(), {})
    value = entry.get(name)
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float(fallback)
    return number if math.isfinite(number) and number > 0.0 else float(fallback)


#: How far from its own mean a unit has to go to count as having fired, in
#: standard deviations of its own activity.
#:
#: Three, which is what Shriki et al. used on human MEG: each sensor's
#: continuous signal is thresholded at three standard deviations of that
#: sensor's own trace, and every excursion past it is an event. It is a
#: statement about a signal and its own variability, so it transfers to a unit
#: that has neither millivolts nor a membrane, where an absolute level does
#: not — and it is the same criterion the avalanche statistics this mesh is
#: compared against were measured under.
#:
#: The mesh had an absolute 0.5 on a tanh output. Measured over 3,000 ticks
#: with 4,096 units and an ordinary drive: zero spikes. Nothing ever reached
#: it, `last_spike_time` stayed at -1 for every unit for the life of the mesh,
#: and `_apply_stdp` therefore skipped every column on every tick. All of the
#: plasticity in this file — both rules, both windows, every constant taken
#: from a paper — had never once fired.
#:
#: Source: Shriki et al. 2013, J Neurosci 33(16):7079.
SPIKE_SIGMA = 3.0

#: How many samples the variance needs before a threshold may be built on it.
#:
#: The relative error of an estimated standard deviation is about
#: 1/sqrt(2(n-1)), so ten samples put it under a quarter. Below that the
#: estimate is small for no reason and the threshold is crossed by everything:
#: unguarded, the mesh fired 8,370 of its 8,373 spikes in one burst in the
#: first few ticks and was silent for the next five thousand.
SPIKE_STATISTICS_MINIMUM = 10


def _from_human_connectome(name: str, fallback: float) -> float:
    """One number from the human connectome reference, or the fallback."""
    try:
        from core.connectome.rich_club import HUMAN_RICH_CLUB

        return float(HUMAN_RICH_CLUB[name])
    except (ImportError, KeyError, TypeError, ValueError):
        return float(fallback)


#: Which index of a weight matrix is the cell that receives.
#:
#: `recurrent = W @ x` means `recurrent[i] = sum_j W[i, j] * x[j]`, so W[i, j]
#: is the weight FROM j INTO i. The tick is the authority on this: it is what
#: actually moves activity, and everything else is analysis of what it does.
#:
#: Three builders in this file wrote the other way round — `weights[source,
#: target]` — and one reader read them that way too, so each agreed with itself
#: and none agreed with the tick. What that cost: the feedforward pathway
#: delivered association activity into sensory columns instead of the other
#: way, the top-down feedback matrix put its drive on the executive columns
#: that sent it so the sensory and association columns it was written for
#: received nothing at all, and the reachability measurement read the
#: inter-column matrix transposed, which is why it agreed with the pathway that
#: was backwards.
#:
#: Written here rather than in each builder because the mistake is the same one
#: three times.
RECEIVER_IS_THE_ROW = True


def _is_human_island(index: int, cfg: Any) -> bool:
    """Whether this column takes its local wiring from H01. -1 means all of them."""
    declared = int(getattr(cfg, "human_island_columns", 0) or 0)
    if declared < 0:
        return True
    return index < declared


#: The one seed every structural stream is derived from.
_MESH_SEED = 42


@dataclass(frozen=True)
class MeshConfig:
    """Immutable configuration for the neural mesh."""
    total_neurons: int = 4096
    columns: int = 64
    neurons_per_column: int = 64   # total_neurons / columns

    # Connectivity
    #
    # Both densities are the cortical microcircuit's own, from Potjans and
    # Diesmann's 8x8 connection matrix: the mean of its eight within-population
    # probabilities, and the mean of its fifty-six between-population ones.
    # They were 0.80 and 0.05, which made this mesh six times more densely
    # wired inside a column than cortex is.
    #
    # Measured before adopting, over 600 ticks with the same seed and the same
    # drive: the multistep-regression branching ratio moves from 0.9917 to
    # 0.9971 — closer to the critical 1.0 the regulator steers for — and the
    # regression's own fit improves from 0.991 to 0.995. A quieter mesh, and a
    # better-conditioned one.
    intra_column_density: float = field(default_factory=_cortical_intra_density)
    inter_column_density: float = field(default_factory=_cortical_inter_density)
    inter_column_distance_decay: float = 0.15   # strength ∝ exp(-d * decay)
    #: How many columns belong to the rich club.
    #:
    #: Twelve of eighty-two regions in van den Heuvel and Sporns' human
    #: connectome, which is one node in seven. Her long-range wiring had no
    #: club at all: measured against degree-preserving rewiring of her own
    #: graph, the normalised coefficient FALLS as the cutoff rises — 0.964,
    #: 0.928, 0.851 — so her best-connected columns are less joined to each
    #: other than their degrees alone would give. Cortex's rises above one.
    #:
    #: A graph with no hubs carries small cascades, because nothing recruits a
    #: distant part of the network in a step or two, and her avalanche
    #: exponents are 3.7 and 3.6 against cortex's 1.5 and 2.0.
    hub_fraction: float = field(
        default_factory=lambda: _from_human_connectome("hub_fraction", 12.0 / 82.0)
    )
    #: How much likelier two hubs are to connect than two ordinary columns.
    #:
    #: Not from a paper. The human measurement says a club EXISTS and how big
    #: it is; the coupling that produces one in a 64-column graph is a property
    #: of this graph, so it is set to whatever reproduces the published
    #: property and no more. See `tools/measure_rich_club.py`.
    hub_coupling: float = 6.0
    #: Derived, not chosen. Potjans and Diesmann's cortical column has 77,169
    #: cells in eight populations, 15,326 of them inhibitory, which is 0.1986.
    #: The table is in core/connectome/types.py and this reads it rather than
    #: repeating a rounded 0.20 that nothing could check.
    inhibitory_fraction: float = field(default_factory=_cortical_inhibitory_fraction)
    #: How many columns are wired from the H01 human reconstruction instead of
    #: from a Gaussian.
    #:
    #: The mesh draws every connection strength from the same normal
    #: distribution, which says a connection is a continuous quantity centred
    #: on zero with no heavy tail. H01 measured the thing itself in a cubic
    #: millimetre of human temporal cortex, and it is a COUNT: 96.5% of
    #: connected pairs are joined by one contact, 0.092% by four or more, and
    #: the heaviest pair carries about fifty. Those are different objects, and
    #: only one of them was measured in a person.
    #:
    #: An island rather than the whole mesh, because what H01 measured is local
    #: wiring in one volume. Claiming it for long-range structure that came from
    #: nowhere near a microscope would be borrowing its authority.
    #:
    #: Measured over 600 ticks with the same seed and the same drive, as the
    #: cortical densities were: the multistep-regression branching ratio goes
    #: from 0.9923 to 0.9932 when every column takes the human wiring, and the
    #: regression's own fit from 0.996 to 0.999. The heaviest connection goes
    #: from 16.5 times the median to 56.0, which is the tail arriving.
    #:
    #: -1 means every column. Intra-column wiring is local wiring, which is
    #: what H01 measured; the inter-column matrices are long-range and keep
    #: their own construction, because nothing in that volume speaks to them.
    human_island_columns: int = -1

    # Dynamics
    #
    # dt is NOT the cortical step. A membrane resolves a 0.5 ms synaptic
    # current and this mesh ticks at 10 Hz over units that have no membrane;
    # the biological number describes a different clock, and adopting it would
    # be arithmetic dressed as fidelity.
    #
    # The leak IS a ratio and does transfer: a membrane forgets its input with
    # a time constant of 10 ms, so one step loses dt/tau of what it held.
    dt: float = 0.05                     # integration timestep
    decay: float = field(default_factory=_cortical_leak)
    noise_sigma: float = 0.008           # stochastic drive
    activation_gain: float = 1.0         # tanh gain

    #: Ceiling on the total inter-column coupling, as the Frobenius norm of the
    #: whole matrix, re-imposed every ten ticks.
    #:
    #: A choice with no measurement behind it, and an inert one: the matrix this
    #: mesh builds has a norm of 0.65, so the ceiling has never once fired.
    #: Worth knowing before reading it as a cap on recruitment -- it is not one,
    #: and a sweep of the coupling that assumed it was would have been chasing a
    #: guard that does nothing. It was an unnamed 15.0 inside the tick, where
    #: nothing sweeping the mesh's coupling could see it. Naming it changes no
    #: behaviour; the per-weight clip beside it, to [-1, 1], is the constraint
    #: that does bite once a weight is scaled far enough.
    inter_column_weight_norm: float = 15.0

    # STDP
    #
    # The window and the asymmetry are Bi and Poo's, measured in hippocampal
    # culture: potentiation falls off with a time constant of 16.8 ms and
    # depression with 33.7 ms, so each depression step is 16.8/33.7 of a
    # potentiation step. The chosen 0.02 and 0.5 were within a whisker of both,
    # which is worth saying — somebody had read the paper — and they are read
    # from it now rather than repeated.
    stdp_lr: float = 0.0005             # base learning rate
    stdp_window: float = field(default_factory=_cortical_stdp_window)
    stdp_potentiation: float = 1.0       # A+
    stdp_depression: float = field(default_factory=_cortical_stdp_asymmetry)
    #: Inhibitory synapses do not learn by the excitatory rule.
    #:
    #: Vogels et al. measured a SYMMETRIC window for inhibitory plasticity —
    #: either order of firing strengthens the synapse — against a standing
    #: depression on every presynaptic spike. The pair holds the postsynaptic
    #: cell near a target rate, which is what makes inhibition track excitation
    #: rather than drift under a rule derived from excitatory pairs.
    #:
    #: The depression constant is their own arithmetic: alpha = 2 * rho * tau,
    #: with rho the target rate of 5 Hz and tau the 20 ms window. Nothing here
    #: is a third number.
    inhibitory_stdp_window: float = field(
        default_factory=lambda: _from_cortex("inhibitory_stdp_window", 0.02)
    )
    inhibitory_stdp_depression: float = field(
        default_factory=lambda: _from_cortex("inhibitory_stdp_depression", 0.2)
    )
    #: How fast inhibitory synapses learn relative to excitatory ones. One,
    #: because no measurement here separates them and inventing a ratio would
    #: be the thing this file exists to stop.
    inhibitory_stdp_rate_ratio: float = 1.0
    #: How much stronger one inhibitory synapse is than one excitatory one.
    #: Potjans & Diesmann's g, and the reason a network with one inhibitory
    #: cell in five does not saturate.
    relative_inhibitory_strength: float = field(
        default_factory=lambda: _from_cortex("relative_inhibitory_strength", 4.0)
    )

    # Pooled lateral inhibition: off, because the mesh now does it with cells.
    #
    # This subtracted a fraction of the mean inhibitory activity from every
    # excitatory unit in a column — one pooled drive, at a strength nobody
    # measured, standing in for inhibition the mesh was not otherwise doing.
    # It was not otherwise doing it because Dale's law was on the wrong axis,
    # so no cell reliably inhibited anything.
    #
    # With Dale's law on the presynaptic axis and Potjans and Diesmann's g of
    # 4, inhibition is carried by the inhibitory cells themselves, through
    # their own synapses, at a strength that came from a paper. The pooled term
    # is then a second inhibition on top of the real one.
    #
    # Measured over 600 ticks with the same seed and drive: turning it off
    # moves the branching ratio from 0.9933 to 0.9812, improves the
    # regression's fit from 0.999 to 1.000, changes the settled level of a
    # driven tier from 0.3019 to 0.2960, and saturates nothing. The mesh stays
    # critical, and a dynamical correction of that size belongs to the
    # criticality regulator, which has gain and noise to steer with, rather
    # than to a fixed number nobody can source.
    #
    # Kept as a knob so the ablation can be run, not as a default.
    lateral_inhibition_strength: float = 0.0

    # Feedforward pathway (sensory → association → executive)
    #
    # Built explicitly, and without a distance term, because a projection from
    # one tier to the next is not a local connection. The mesh used to leave
    # this to inter_column_weights, whose probability decays as
    # exp(-|i - j| * 0.15); a sensory column at index 0 and an executive column
    # at index 48 are 48 apart, which makes that probability 0.05 * e^-7.2, or
    # about one edge in twenty-eight thousand. Measured over eight seeds with
    # both matrices built, nought of sixteen executive columns was reachable
    # from any sensory column, every time, while the code injects into the
    # sensory tier and reads the executive projection. The density is the same
    # 0.05 the local wiring uses; only the decay is gone.
    feedforward_density: float = 0.05
    feedforward_strength: float = 0.06
    #: Sensory straight to executive, sparser, as a shortcut rather than a path.
    feedforward_direct_density: float = 0.01

    # Tier boundaries (column indices)
    sensory_end: int = 16
    association_end: int = 48
    # executive = 48..63

    # Integration
    update_hz: float = 10.0              # 10 Hz mesh tick (lighter than substrate 20 Hz)
    projection_dim: int = 64             # output projection back to LiquidSubstrate


# ---------------------------------------------------------------------------
# Column
# ---------------------------------------------------------------------------

class CorticalColumn:
    """A minicolumn of neurons with local recurrence and lateral inhibition.

    Each column maintains:
      • x  — activation vector  (n,)
      • W  — intra-column weight matrix (n, n)
      • inh_mask — boolean mask for inhibitory neurons
    """

    __slots__ = ("index", "tier", "n", "x", "W", "inh_mask", "last_spike_time",
                 "_lateral_inh_strength", "human_island", "x_mean", "x_var",
                 "stats_samples")

    def __init__(self, index: int, tier: CorticalTier, n: int, cfg: MeshConfig,
                 rng: np.random.Generator, human_island: bool = False):
        self.index = index
        self.tier = tier
        self.n = n
        self.human_island = bool(human_island)
        self.x = rng.standard_normal(n).astype(np.float32) * 0.05

        # What this tier's cortical layers say about its own wiring, rather
        # than one figure averaged over all of them. Layer 4 is 20.0%
        # inhibitory at density 0.106, layers 2/3 are 22.0% at 0.135, and the
        # output layers are 17.3% at 0.091.
        #
        # Measured against the global figures on the same seed and drive, with
        # every column on the human wiring: the branching ratio is 0.9932
        # either way and the fit is 0.999 against 1.000. The dynamics do not
        # care, and the anatomy does — her three bands now carry the
        # composition their cortical layers were measured to have, which the
        # global figures cannot express at all.
        density = _for_tier(tier, "intra_column_density", cfg.intra_column_density)
        inhibitory_fraction = _for_tier(
            tier, "inhibitory_fraction", cfg.inhibitory_fraction
        )

        # Intra-column connectivity (dense)
        if self.human_island:
            # Strengths are contact counts drawn from what H01 measured in
            # human cortex, not Gaussian draws. Same density, same sign
            # convention, different distribution of strength: almost every
            # connection is worth one contact and a rare one is worth fifty.
            from core.connectome.island import wire_island

            self.W = wire_island(n, density, rng, contact_strength=0.1)
        else:
            mask = rng.random((n, n)) < density
            self.W = np.abs(
                rng.standard_normal((n, n)).astype(np.float32) * 0.1
            ) * mask

        # Dale's law, on the presynaptic axis.
        #
        # `recurrent = W @ x` makes W[i, j] the weight FROM j INTO i, so a
        # cell's output is its COLUMN. This negated the ROW, which is a cell's
        # input: every inhibitory cell was receiving nothing but inhibition
        # while sending whatever sign the draw happened to give it. Measured on
        # column 0 before the fix: 100% of what inhibitory cells received was
        # negative, 71.7% of what they sent was, and 64.4% of what EXCITATORY
        # cells sent was negative too. There was no Dale's law in the mesh.
        #
        # And an inhibitory synapse is stronger than an excitatory one. Potjans
        # and Diesmann's g is 4, and it is not an independent choice: with one
        # cell in five inhibitory, 0.801 of unit excitation meets 0.199 * 4 =
        # 0.794 of inhibition and the network sits at balance. Without it the
        # mesh runs away — measured over 1,200 ticks with Dale's law applied
        # and g left at 1, mean activity went from 0.385 to 0.978 and 95% of
        # units pinned against their ceiling by tick 300.
        num_inh = max(1, int(n * inhibitory_fraction))
        self.inh_mask = np.zeros(n, dtype=bool)
        self.inh_mask[rng.choice(n, size=num_inh, replace=False)] = True
        self.W[:, self.inh_mask] = -cfg.relative_inhibitory_strength * self.W[
            :, self.inh_mask
        ]

        # Zero diagonal (no self-connection)
        np.fill_diagonal(self.W, 0.0)

        # Spike timing for STDP
        self.last_spike_time = np.full(n, -1.0, dtype=np.float64)

        # What this unit's own activity usually looks like, so a threshold
        # crossing can be defined against it. Both are exponential averages
        # over the unit's own timescale; see `spike_thresholds`.
        #: Mean and sum of squared deviations over the unit's WHOLE history,
        #: by Welford's method. Not a window: Shriki's threshold is three
        #: standard deviations of a sensor's own trace over the recording, and
        #: a window would be a length nobody measured. An exponential average
        #: at the unit's own timescale was tried and tracks the unit rather
        #: than its variability — one spike in eight thousand ticks.
        self.x_mean = self.x.copy()
        self.x_var = np.zeros(n, dtype=np.float32)
        self.stats_samples = 0

        self._lateral_inh_strength = cfg.lateral_inhibition_strength

    def step(self, external_input: np.ndarray, dt: float, decay: float,
             noise_sigma: float, gain: float, now: float, spike_threshold: float = 0.5):
        """Euler step with lateral inhibition and spike-time recording."""
        try:
            external_input = np.asarray(external_input, dtype=np.float32).reshape(-1)
        except (TypeError, ValueError) as exc:
            _record_neural_mesh_degradation(
                exc,
                action="discarded malformed CorticalColumn external input",
                severity="warning",
                extra={"column": self.index, "expected_len": self.n},
            )
            external_input = np.zeros(self.n, dtype=np.float32)
        if external_input.size != self.n:
            _record_neural_mesh_degradation(
                ValueError(
                    f"CorticalColumn input length {external_input.size} != {self.n}"
                ),
                action="padded or truncated CorticalColumn external input",
                severity="warning",
                extra={"column": self.index, "expected_len": self.n},
            )
            normalized_input = np.zeros(self.n, dtype=np.float32)
            normalized_input[: min(self.n, external_input.size)] = external_input[: self.n]
            external_input = normalized_input
        external_input = np.nan_to_num(external_input, nan=0.0, posinf=1.0, neginf=-1.0)

        dt, valid_dt = _finite_float(dt, 0.05)
        decay, valid_decay = _finite_float(decay, 0.03)
        noise_sigma, valid_noise = _finite_float(noise_sigma, 0.0)
        gain, valid_gain = _finite_float(gain, 1.0)
        spike_threshold, valid_threshold = _finite_float(spike_threshold, 0.5)
        if not all((valid_dt, valid_decay, valid_noise, valid_gain, valid_threshold)):
            _record_neural_mesh_degradation(
                ValueError("CorticalColumn step received non-finite dynamics"),
                action="normalized CorticalColumn dynamics before stepping",
                severity="warning",
                extra={"column": self.index},
            )
        dt, _ = _clamp_float(dt, lower=1e-6, upper=1.0)
        decay, _ = _clamp_float(decay, lower=0.0, upper=10.0)
        noise_sigma, _ = _clamp_float(noise_sigma, lower=0.0, upper=1.0)
        gain, _ = _clamp_float(gain, lower=0.05, upper=5.0)
        spike_threshold, _ = _clamp_float(spike_threshold, lower=0.0, upper=1.0)
        now, valid_now = _finite_float(now, time.monotonic())
        if not valid_now:
            _record_neural_mesh_degradation(
                ValueError("CorticalColumn spike timestamp was non-finite"),
                action="used monotonic time for CorticalColumn spike timestamp",
                severity="warning",
                extra={"column": self.index},
            )

        recurrent = self.W @ self.x
        recurrent = np.nan_to_num(recurrent, nan=0.0, posinf=1.0, neginf=-1.0)
        activity = np.tanh(gain * (recurrent + external_input))

        # Lateral inhibition: inhibitory pool suppresses excitatory neurons
        inh_activity = np.mean(np.abs(self.x[self.inh_mask])) if np.any(self.inh_mask) else 0.0
        inhibition = np.zeros(self.n, dtype=np.float32)
        inhibition[~self.inh_mask] = -self._lateral_inh_strength * inh_activity

        noise = np.random.standard_normal(self.n).astype(np.float32) * noise_sigma
        # A leaky integrator leaks its state toward its drive. This leaked the
        # state and added the drive, which is a different equation: the resting
        # point of `dx = (-decay*x + drive)*dt` is drive/decay, and with the
        # cortical leak of 0.025 that is forty times whatever the unit is being
        # driven by. Every unit ran to its clip and stayed there, which is why
        # 95% of the mesh sat pinned at |x| > 0.99 after three hundred ticks
        # once Dale's law was applied and the drive stopped cancelling itself.
        #
        # In this form the resting point is the drive itself, bounded by the
        # tanh that produced it, and `decay` finally means what the constants
        # file says it means: the fraction of its state a unit loses each step.
        dx = (-self.x + activity + inhibition + noise) * decay
        dx = np.nan_to_num(dx, nan=0.0, posinf=1.0, neginf=-1.0)
        self.x = np.clip(self.x + dx, -1.0, 1.0).astype(np.float32)

        # Record spike times for STDP. `spike_threshold` is kept in the
        # signature for callers that want an absolute one; when it is left at
        # its default the unit's own fluctuation decides, as in the batched
        # tick.
        deviation = self.x - self.x_mean
        self.stats_samples += 1
        self.x_mean = (self.x_mean + deviation / self.stats_samples).astype(np.float32)
        self.x_var = (self.x_var + deviation * (self.x - self.x_mean)).astype(np.float32)
        if self.stats_samples >= SPIKE_STATISTICS_MINIMUM:
            sigma = np.sqrt(np.maximum(self.x_var / (self.stats_samples - 1), 1e-12))
            firing = np.abs(self.x - self.x_mean) > SPIKE_SIGMA * sigma
            self.last_spike_time[firing] = now


# ---------------------------------------------------------------------------
# Main mesh
# ---------------------------------------------------------------------------

class NeuralMesh(MeshWiring):
    """The 4096-neuron cortical mesh.

    Lifecycle:
        mesh = NeuralMesh()
        await mesh.start()   # spawns background integration loop
        ...
        await mesh.stop()

    External API:
        mesh.inject_sensory(vector)       — push embodiment/interoceptive signals
        mesh.inject_association(vector)    — push cross-modal / memory signals
        mesh.get_executive_projection()    — 64-d projection for LiquidSubstrate
        mesh.get_field_state()             — full 4096-d activation snapshot
        mesh.get_column_summary(i)         — per-column stats
        mesh.get_tier_energy(tier)         — mean energy for a tier
    """

    def __init__(self, cfg: MeshConfig | None = None):
        self.cfg = cfg or MeshConfig()
        self._validate_config()
        # One generator per structure, not one for the whole mesh.
        #
        # Everything drew from a single stream in construction order, so a
        # change to how a COLUMN is wired changed how many numbers came out
        # before the long-range matrices were built, and those came out
        # different too. Measured while giving each tier its own layers'
        # density: the inter-column graph went from 106 edges to 80, four
        # columns fell out of it entirely, and executive reachability dropped
        # from 14 of 16 to 12 — none of it caused by the change, all of it the
        # stream having moved. An experiment on local wiring cannot be allowed
        # to rewire the long-range graph as a side effect.
        #
        # Independent streams from one seed, so each structure is reproducible
        # on its own and every arm of a comparison gets the same long-range
        # graph unless the arm is about the long-range graph.
        columns, inter, feedforward, feedback, projection, noise = (
            np.random.default_rng([_MESH_SEED, stream]) for stream in range(6)
        )
        self._rng_columns = columns
        self._rng_inter = inter
        self._rng_feedforward = feedforward
        self._rng_feedback = feedback
        self._rng_projection = projection
        #: Kept for the live step, which wants fresh noise rather than a
        #: reproducible structure.
        self._rng = noise
        self._lock = threading.Lock()
        self._modulation_lock = threading.Lock()

        # Build columns
        self.columns: list[CorticalColumn] = []
        for i in range(self.cfg.columns):
            tier = self._tier_for(i)
            col = CorticalColumn(
                i,
                tier,
                self.cfg.neurons_per_column,
                self.cfg,
                self._rng_columns,
                human_island=_is_human_island(i, self.cfg),
            )
            self.columns.append(col)

        #: Which tier each column belongs to, as a name, so a per-tier multiplier
        #: can be turned into a per-column vector without asking again.
        # ``CorticalTier`` is an auto() enum, so ``.value`` is 1, 2, 3. The NAME
        # is what a receptor field is keyed by, and reading the value here made
        # every per-tier lookup miss and every multiplier silently stay at one.
        self._tier_names: list[str] = [
            self._tier_for(index).name.lower() for index in range(self.cfg.columns)
        ]

        #: Which columns belong to the rich club. Drawn before the long-range
        #: matrix, because it is what that matrix is built against.
        from core.connectome.rich_club import hub_columns

        self._hubs = hub_columns(
            self.cfg.columns,
            float(getattr(self.cfg, "hub_fraction", 0.0) or 0.0),
            self._tier_names,
            self._rng_inter,
        )

        # Inter-column weight matrix (columns × columns), sparse, distance-weighted
        self._inter_W = self._build_inter_column_weights()

        # Live per-column activation (mean membrane state), refreshed each step.
        # Initialized here so a reader that arrives before the first step gets a
        # correctly-shaped zero vector rather than a missing attribute.
        self._column_activations = np.zeros(self.cfg.columns, dtype=np.float32)

        # Projection matrix: 4096 → 64 (learned via slow PCA-like update)
        self._projection = self._rng_projection.standard_normal(
            (self.cfg.projection_dim, self.cfg.total_neurons)
        ).astype(np.float32) * (1.0 / np.sqrt(self.cfg.total_neurons))

        # Neurochemistry supplies the base state; criticality supplies bounded
        # multiplicative factors. Publishing one immutable tuple keeps mesh
        # ticks coherent without taking a controller lock on the hot path.
        self._base_modulatory_state = (1.0, 1.0, 1.0)
        # Per-tier multipliers on gain and noise. Uniform until something
        # measures otherwise, which is the honest default: it says the spatial
        # structure has not been measured rather than guessing at it. A
        # transmitter arriving at the sensory tier and the same transmitter
        # arriving at the executive tier were the same event before this — one
        # scalar for 4,096 units — and in cortex they are not.
        self._tier_modulation: dict[str, tuple[float, float]] = {
            "sensory": (1.0, 1.0),
            "association": (1.0, 1.0),
            "executive": (1.0, 1.0),
        }
        self._criticality_modulatory_factors = (1.0, 1.0)
        self._modulatory_state = (1.0, 1.0, 1.0)
        self._modulatory_gain: float = 1.0
        self._modulatory_plasticity: float = 1.0  # scales STDP rate
        self._modulatory_noise: float = 1.0        # scales noise

        # Runtime
        self._running = False
        self._task: asyncio.Task | None = None
        self._tick_count: int = 0
        self._start_time: float = 0.0
        self._consecutive_tick_failures: int = 0
        self._last_tick_error_at: float = 0.0

        # Sensory injection buffer (set externally, consumed each tick)
        self._sensory_buffer: np.ndarray | None = None
        self._association_buffer: np.ndarray | None = None

        # Recurrent Processing Theory (Lamme): explicit top-down feedback
        self._recurrent_feedback_enabled: bool = True
        self._recurrent_feedback_strength: float = 0.8  # relative to feedforward
        self._feedback_W: np.ndarray | None = None
        self._build_feedback_weights()
        # And the pathway that carries signal the other way. Folded into
        # _inter_W rather than applied separately: the feedforward sweep is the
        # mesh's ordinary integration step, not a second pass over it.
        self._inter_W = self._inter_W + self._build_feedforward_weights()
        self._connect_every_column()

        # Stats
        self._mean_column_energy: float = 0.0
        self._global_synchrony: float = 0.0
        self._tier_energies: dict[CorticalTier, float] = {t: 0.0 for t in CorticalTier}
        initial_state = np.concatenate([col.x for col in self.columns]).astype(np.float32, copy=False)
        self._cached_field_state = initial_state.copy()
        self._cached_executive_projection = np.tanh(
            self._projection @ self._cached_field_state
        ).astype(np.float32)

        logger.info(
            "NeuralMesh initialized: %d neurons, %d columns, tiers=[S:%d A:%d E:%d]",
            self.cfg.total_neurons, self.cfg.columns,
            self.cfg.sensory_end,
            self.cfg.association_end - self.cfg.sensory_end,
            self.cfg.columns - self.cfg.association_end,
        )

    def _validate_config(self) -> None:
        """Fail closed before invalid numeric topology reaches the runtime loop."""
        cfg = self.cfg
        int_fields = {
            "total_neurons": cfg.total_neurons,
            "columns": cfg.columns,
            "neurons_per_column": cfg.neurons_per_column,
            "sensory_end": cfg.sensory_end,
            "association_end": cfg.association_end,
            "projection_dim": cfg.projection_dim,
        }
        for name, value in int_fields.items():
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"NeuralMesh config field {name} must be a positive int")
        if cfg.total_neurons != cfg.columns * cfg.neurons_per_column:
            raise ValueError(
                "NeuralMesh total_neurons must equal columns * neurons_per_column"
            )
        if not (0 < cfg.sensory_end < cfg.association_end < cfg.columns):
            raise ValueError(
                "NeuralMesh tier boundaries must satisfy "
                "0 < sensory_end < association_end < columns"
            )

        finite_fields = {
            "intra_column_density": cfg.intra_column_density,
            "inter_column_density": cfg.inter_column_density,
            "inter_column_distance_decay": cfg.inter_column_distance_decay,
            "inhibitory_fraction": cfg.inhibitory_fraction,
            "dt": cfg.dt,
            "decay": cfg.decay,
            "noise_sigma": cfg.noise_sigma,
            "activation_gain": cfg.activation_gain,
            "stdp_lr": cfg.stdp_lr,
            "stdp_window": cfg.stdp_window,
            "stdp_potentiation": cfg.stdp_potentiation,
            "stdp_depression": cfg.stdp_depression,
            "lateral_inhibition_strength": cfg.lateral_inhibition_strength,
            "update_hz": cfg.update_hz,
        }
        for name, value in finite_fields.items():
            finite, valid = _finite_float(value, 0.0)
            if not valid:
                raise ValueError(f"NeuralMesh config field {name} must be finite")
            if name in {"dt", "stdp_window", "update_hz"} and finite <= 0.0:
                raise ValueError(f"NeuralMesh config field {name} must be > 0")
            if name.endswith("_density") and not (0.0 <= finite <= 1.0):
                raise ValueError(f"NeuralMesh config field {name} must be in [0, 1]")
            if name == "inhibitory_fraction" and not (0.0 < finite < 1.0):
                raise ValueError("NeuralMesh inhibitory_fraction must be in (0, 1)")
            if name in {"decay", "noise_sigma", "stdp_lr"} and finite < 0.0:
                raise ValueError(f"NeuralMesh config field {name} must be >= 0")
            if name in {
                "activation_gain",
                "stdp_potentiation",
                "stdp_depression",
                "lateral_inhibition_strength",
            } and finite < 0.0:
                raise ValueError(f"NeuralMesh config field {name} must be >= 0")
            if name == "inter_column_distance_decay" and finite < 0.0:
                raise ValueError("NeuralMesh inter_column_distance_decay must be >= 0")

    @staticmethod
    def _coerce_signal_vector(
        vector: object,
        *,
        expected_len: int,
        action: str,
    ) -> np.ndarray:
        """Normalize external signals before they can perturb the mesh state."""
        try:
            arr = np.asarray(vector, dtype=np.float32).reshape(-1)
        except (TypeError, ValueError) as exc:
            _record_neural_mesh_degradation(
                exc,
                action=action,
                severity="warning",
                extra={"fallback": "zero_vector", "expected_len": expected_len},
            )
            return np.zeros(expected_len, dtype=np.float32)

        if arr.size == 0:
            _record_neural_mesh_degradation(
                ValueError("empty neural mesh injection vector"),
                action=action,
                severity="warning",
                extra={"fallback": "zero_vector", "expected_len": expected_len},
            )
            return np.zeros(expected_len, dtype=np.float32)

        nonfinite = int(np.size(arr) - np.count_nonzero(np.isfinite(arr)))
        if nonfinite:
            _record_neural_mesh_degradation(
                ValueError(f"neural mesh injection vector had {nonfinite} non-finite values"),
                action=action,
                severity="warning",
                extra={"fallback": "finite_sanitization", "expected_len": expected_len},
            )
            arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=-1.0)

        if arr.size > expected_len:
            _record_neural_mesh_degradation(
                ValueError(
                    f"neural mesh injection vector length {arr.size} exceeds {expected_len}"
                ),
                action=action,
                severity="warning",
                extra={"fallback": "truncate", "expected_len": expected_len},
            )
            arr = arr[:expected_len]

        return np.clip(arr.astype(np.float32, copy=False), -1.0, 1.0)

    # ── Tier helpers ─────────────────────────────────────────────────────

    def _tier_for(self, col_idx: int) -> CorticalTier:
        if col_idx < self.cfg.sensory_end:
            return CorticalTier.SENSORY
        if col_idx < self.cfg.association_end:
            return CorticalTier.ASSOCIATION
        return CorticalTier.EXECUTIVE

    # ── Inter-column connectivity ────────────────────────────────────────


    def _apply_recurrent_feedback(self, dt: float, gain: float,
                                   noise_sigma: float, now: float):
        """Apply top-down recurrent feedback from executive to sensory tiers.

        This is the Lamme RPT mechanism: after feedforward processing completes
        in step 3, executive columns send signals back down to sensory columns.
        This recurrent sweep is what RPT claims generates phenomenal experience.

        The feedback modulates sensory columns by adding a top-down prior that
        shapes what the sensory tier "expects to see" based on executive state.
        """
        if self._feedback_W is None:
            return

        # Compute column-level means for the feedback path
        col_means = np.array([np.mean(c.x) for c in self.columns], dtype=np.float32)
        col_means = np.nan_to_num(col_means, nan=0.0, posinf=1.0, neginf=-1.0)
        feedback_w = np.nan_to_num(self._feedback_W, nan=0.0, posinf=1.0, neginf=-1.0)
        strength, valid_strength = _finite_float(self._recurrent_feedback_strength, 0.8)
        strength, strength_unchanged = _clamp_float(strength, lower=0.0, upper=3.0)
        if not valid_strength or not strength_unchanged:
            _record_neural_mesh_degradation(
                ValueError("NeuralMesh recurrent feedback strength was invalid"),
                action="normalized recurrent feedback strength before applying feedback",
                severity="warning",
                extra={"feedback_strength": strength},
            )
        feedback_drive = feedback_w @ col_means * strength
        feedback_drive = np.nan_to_num(feedback_drive, nan=0.0, posinf=1.0, neginf=-1.0)

        # Apply feedback as modulatory input to target columns
        for i, col in enumerate(self.columns):
            if col.tier in (CorticalTier.SENSORY, CorticalTier.ASSOCIATION):
                if abs(feedback_drive[i]) > 1e-6:
                    feedback_input = np.full(col.n, feedback_drive[i], dtype=np.float32)
                    # The feedback is gentler than feedforward — it modulates, not overrides
                    dx_feedback = np.tanh(gain * 0.5 * feedback_input) * dt * 0.3
                    col.x = np.clip(col.x + dx_feedback, -1.0, 1.0).astype(np.float32)

    def set_recurrent_feedback_enabled(self, enabled: bool):
        """Enable/disable recurrent feedback for ablation testing.

        When disabled, feedforward processing (steps 1-3) still works but
        RPT predicts phenomenal experience should degrade. Compare qualia
        output with and without this to test RPT vs GWT predictions.
        """
        prev = self._recurrent_feedback_enabled
        self._recurrent_feedback_enabled = enabled
        if prev != enabled:
            logger.info("NeuralMesh: Recurrent feedback %s (RPT ablation)",
                        "ENABLED" if enabled else "DISABLED")

    # ── Lifecycle ────────────────────────────────────────────────────────

    async def start(self):
        if self._running:
            return
        self._running = True
        self._start_time = time.monotonic()
        try:
            self._task = get_task_tracker().create_task(self._run_loop(), name="NeuralMesh")
        except _RECOVERABLE_NEURAL_MESH_ERRORS as exc:
            self._running = False
            self._task = None
            _record_neural_mesh_degradation(
                exc,
                action="failed closed when NeuralMesh task creation failed",
                severity="critical",
            )
            raise
        logger.info("NeuralMesh STARTED (%s Hz)", self.cfg.update_hz)

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                logger.debug("NeuralMesh task cancellation acknowledged")
            self._task = None
        logger.info("NeuralMesh STOPPED (ticks=%d)", self._tick_count)

    # ── Main loop ────────────────────────────────────────────────────────

    async def _run_loop(self):
        update_hz, valid_update_hz = _finite_float(self.cfg.update_hz, 10.0)
        update_hz, unchanged_update_hz = _clamp_float(update_hz, lower=0.1, upper=120.0)
        if not valid_update_hz or not unchanged_update_hz:
            _record_neural_mesh_degradation(
                ValueError(f"unsafe NeuralMesh update_hz: {self.cfg.update_hz!r}"),
                action="normalized NeuralMesh update rate before entering runtime loop",
                severity="warning",
                extra={"normalized_update_hz": update_hz},
            )
        interval = 1.0 / update_hz
        try:
            while self._running:
                t0 = time.monotonic()
                try:
                    await asyncio.to_thread(self._tick)
                    self._consecutive_tick_failures = 0
                except _RECOVERABLE_NEURAL_MESH_ERRORS as exc:
                    self._consecutive_tick_failures += 1
                    self._last_tick_error_at = time.monotonic()
                    _record_neural_mesh_degradation(
                        exc,
                        action="kept NeuralMesh loop alive after tick failure and damped field",
                        extra={"consecutive_tick_failures": self._consecutive_tick_failures},
                    )
                    logger.error("NeuralMesh tick error: %s", exc, exc_info=True)
                    self._enter_fail_safe_state()
                elapsed = time.monotonic() - t0
                backoff = min(
                    interval * max(0, self._consecutive_tick_failures),
                    2.0,
                )
                await asyncio.sleep(max(0.0, interval + backoff - elapsed))
        except asyncio.CancelledError:
            logger.debug("NeuralMesh run loop cancelled")
        finally:
            self._running = False

    def _enter_fail_safe_state(self) -> None:
        """Dampen unstable dynamics after a failed tick without erasing topology."""
        with self._lock:
            for col in self.columns:
                col.x = (
                    np.nan_to_num(col.x, nan=0.0, posinf=1.0, neginf=-1.0)
                    .clip(-1.0, 1.0)
                    .astype(np.float32, copy=False)
                    * np.float32(0.95)
                )
            with self._modulation_lock:
                gain, plasticity, noise = self._modulatory_state
                self._modulatory_state = (
                    max(0.1, min(1.0, gain)),
                    plasticity,
                    max(0.0, min(1.0, noise)),
                )
                (
                    self._modulatory_gain,
                    self._modulatory_plasticity,
                    self._modulatory_noise,
                ) = self._modulatory_state
            self._refresh_cached_snapshots()

    def _tick(self):
        """One integration step (runs in thread pool)."""
        with self._lock:
            self._tick_inner()

    def _tick_inner(self):
        now = time.monotonic()
        dt = self.cfg.dt
        cfg = self.cfg
        modulatory_gain, _, modulatory_noise = self._modulatory_state
        gain = cfg.activation_gain * modulatory_gain
        # Apply subcortical arousal gating to mesh gain
        try:
            from core.consciousness.subcortical_core import get_subcortical_core
            subcortical_gain, valid_subcortical_gain = _finite_float(
                get_subcortical_core().get_mesh_gain_multiplier(),
                1.0,
            )
            if not valid_subcortical_gain:
                _record_neural_mesh_degradation(
                    ValueError("subcortical mesh gain was non-finite"),
                    action="used neutral subcortical gain for NeuralMesh tick",
                    severity="warning",
                )
            gain *= subcortical_gain
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            _record_neural_mesh_degradation(
                exc,
                action="used base NeuralMesh gain because subcortical gain was unavailable",
                severity="warning",
            )
            logger.debug("Subcortical mesh gain unavailable, using base gain: %s", exc)
        gain, valid_gain = _finite_float(gain, cfg.activation_gain)
        gain, gain_unchanged = _clamp_float(gain, lower=0.05, upper=5.0)
        if not valid_gain or not gain_unchanged:
            _record_neural_mesh_degradation(
                ValueError("NeuralMesh gain was non-finite or out of bounds"),
                action="clamped NeuralMesh gain before integration",
                severity="warning",
                extra={"effective_gain": gain},
            )
        noise_sigma = cfg.noise_sigma * modulatory_noise
        noise_sigma, valid_noise_sigma = _finite_float(noise_sigma, cfg.noise_sigma)
        noise_sigma, noise_unchanged = _clamp_float(noise_sigma, lower=0.0, upper=1.0)
        if not valid_noise_sigma or not noise_unchanged:
            _record_neural_mesh_degradation(
                ValueError("NeuralMesh noise was non-finite or out of bounds"),
                action="clamped NeuralMesh noise before integration",
                severity="warning",
                extra={"effective_noise_sigma": noise_sigma},
            )
        n = cfg.neurons_per_column

        # ── 1. Distribute injection buffers to tier columns ──────────
        sensory_input = self._consume_buffer("_sensory_buffer", cfg.sensory_end)
        assoc_input = self._consume_buffer("_association_buffer",
                                           cfg.association_end - cfg.sensory_end,
                                           offset=cfg.sensory_end)

        # ── 2. Batched column step (vectorized) ─────────────────────
        # Gather all column activations into a single (columns, n) matrix.
        # This replaces 64 sequential matmuls with batched numpy operations.
        x_matrix = np.array([c.x for c in self.columns], dtype=np.float32)  # (64, 64)
        if not np.all(np.isfinite(x_matrix)):
            _record_neural_mesh_degradation(
                ValueError("NeuralMesh column state contained non-finite values"),
                action="sanitized NeuralMesh column state before integration",
                severity="warning",
            )
            x_matrix = np.nan_to_num(x_matrix, nan=0.0, posinf=1.0, neginf=-1.0)

        # Inter-column coupling: column means → inter-column drive
        col_means = x_matrix.mean(axis=1)  # (64,) — computed ONCE, reused in stats
        # Published for the layers that read the mesh's live state (ALife/Lenia,
        # topology evolution, criticality). They fetch `mesh.column_activations`,
        # and until this existed the attribute was simply absent — so every one
        # of them hit `if activations is None: return` and never ran at all.
        self._column_activations = col_means
        inter_drive = self._inter_W @ col_means  # (64,)

        # Build external input matrix (64, 64)
        ext = np.broadcast_to(inter_drive[:, None], (cfg.columns, n)).copy()

        # Tier-specific injection
        if sensory_input is not None:
            max_s = min(cfg.sensory_end * n, len(sensory_input))
            ext[:cfg.sensory_end, :].flat[:max_s] += sensory_input[:max_s]
        if assoc_input is not None:
            a_start = cfg.sensory_end
            a_end = cfg.association_end
            max_a = min((a_end - a_start) * n, len(assoc_input))
            ext[a_start:a_end, :].flat[:max_a] += assoc_input[:max_a]

        # NaN guard
        ext = np.nan_to_num(ext, nan=0.0, posinf=1.0, neginf=-1.0)

        # Batched recurrent: each column's W @ x, vectorized.
        # W_all shape (64, 64, 64) — 64 columns each with (64,64) weight matrix
        if not hasattr(self, '_W_batch') or self._tick_count % 100 == 0:
            self._W_batch = np.array([c.W for c in self.columns], dtype=np.float32)
            if _HAS_MLX and _MLX_METAL_ENABLED:
                self._W_batch_mx = mx.array(self._W_batch)

        # Metal GPU acceleration: offload the heavy einsum to Apple Metal via MLX.
        # For 64 columns × (64×64) matmuls, Metal is 5-10x faster than CPU numpy.
        # One gain per column rather than one for the mesh. Uniform unless a
        # receptor field says otherwise, so this is the same arithmetic it was
        # until something measures a difference between the tiers.
        gain_by_column = gain * self._tier_vector(0)
        if _HAS_MLX and _MLX_METAL_ENABLED:
            x_mx = mx.array(x_matrix)
            ext_mx = mx.array(ext)
            gain_mx = mx.array(gain_by_column.reshape(-1, 1))
            recurrent_mx = mx.einsum('cij,cj->ci', self._W_batch_mx, x_mx)
            activity_mx = mx.tanh(gain_mx * (recurrent_mx + ext_mx))
            mx.eval(activity_mx)  # force Metal evaluation
            activity = np.array(activity_mx, dtype=np.float32)
            recurrent = np.array(recurrent_mx, dtype=np.float32)
        else:
            recurrent = np.einsum('cij,cj->ci', self._W_batch, x_matrix)  # (64, 64)
            activity = np.tanh(gain_by_column[:, None] * (recurrent + ext))
        recurrent = np.nan_to_num(recurrent, nan=0.0, posinf=1.0, neginf=-1.0)
        activity = np.nan_to_num(activity, nan=0.0, posinf=1.0, neginf=-1.0)

        # Lateral inhibition: per-column inhibitory pool
        inh_masks = np.array([c.inh_mask for c in self.columns])  # (64, 64) bool
        inh_activity = np.where(inh_masks, np.abs(x_matrix), 0.0).sum(axis=1)
        inh_counts = inh_masks.sum(axis=1).clip(1)
        inh_mean = inh_activity / inh_counts  # (64,)
        inhibition = np.where(~inh_masks, -cfg.lateral_inhibition_strength * inh_mean[:, None], 0.0)

        noise = (
            self._rng.standard_normal(x_matrix.shape).astype(np.float32)
            * noise_sigma
            * self._tier_vector(1)[:, None]
        )
        # The same leaky integrator as CorticalColumn.step, and for the same
        # reason: the resting point of a unit is its drive, not its drive
        # divided by the leak.
        dx = (-x_matrix + activity + inhibition + noise) * cfg.decay
        dx = np.nan_to_num(dx, nan=0.0, posinf=1.0, neginf=-1.0)
        x_new = np.clip(x_matrix + dx, -1.0, 1.0).astype(np.float32)

        # Write back to columns and record spike times.
        #
        # A spike is an excursion past the unit's OWN fluctuation, not past a
        # fixed level. The fixed level was 0.5 on a tanh output, and nothing in
        # this mesh ever reached it — see SPIKE_SIGMA.
        for i, col in enumerate(self.columns):
            col.x = x_new[i]
            # Both averages run at the unit's own timescale, which is the leak.
            # No new constant: a unit that forgets its state at rate `decay`
            # has no other rate to remember its statistics at.
            deviation = col.x - col.x_mean
            col.stats_samples += 1
            col.x_mean = (col.x_mean + deviation / col.stats_samples).astype(np.float32)
            col.x_var = (col.x_var + deviation * (col.x - col.x_mean)).astype(np.float32)
            if col.stats_samples >= SPIKE_STATISTICS_MINIMUM:
                sigma = np.sqrt(np.maximum(col.x_var / (col.stats_samples - 1), 1e-12))
                firing = np.abs(col.x - col.x_mean) > SPIKE_SIGMA * sigma
                col.last_spike_time[firing] = now

        # Cache col_means for stats (avoid recomputation)
        self._cached_col_means = col_means

        # ── 3. Recurrent Processing (Lamme RPT) ─────────────────────
        if self._recurrent_feedback_enabled:
            self._apply_recurrent_feedback(dt, gain, noise_sigma, now)

        # ── 4. STDP (every other tick for perf) ─────────────────────
        if self._tick_count % 2 == 0:
            self._apply_stdp(now)
            # Sync batched weights after STDP modifies them
            self._W_batch = np.array([c.W for c in self.columns], dtype=np.float32)

        # ── 5. Inter-column weight normalization ─────────────────────
        # Only recompute norm every 10 ticks (weights change slowly)
        if self._tick_count % 10 == 0:
            norm = np.linalg.norm(self._inter_W)
            if norm > cfg.inter_column_weight_norm:
                self._inter_W *= cfg.inter_column_weight_norm / norm
            self._inter_W = np.nan_to_num(
                np.clip(self._inter_W, -1.0, 1.0),
                nan=0.0,
                posinf=1.0,
                neginf=-1.0,
            ).astype(np.float32)

        # ── 6. Compute stats ─────────────────────────────────────────
        self._update_stats()

        self._tick_count += 1

    # ── live state surface ───────────────────────────────────────────
    # The mesh's real dynamical state, published under the names its consumers
    # actually read. Before these existed, `getattr(mesh, "column_activations",
    # None)` returned None in ALife dynamics, topology evolution and the
    # criticality regulator, so all three returned early on every tick — their
    # mathematics ran only in unit tests. The advertised "living neural ecology"
    # was structurally disconnected, not merely discarded.

    @property
    def column_activations(self) -> np.ndarray:
        """Mean activation per column, shape (columns,).

        This is the same vector the mesh feeds through ``_inter_W`` to produce
        inter-column drive — the live state, not a copy computed for observers.
        """
        return self._column_activations

    @property
    def inter_column_weights(self) -> np.ndarray:
        """The causal inter-column coupling matrix, shape (columns, columns).

        Returned by reference: ``apply_inter_column_coupling`` is the supported
        way to change it, but readers get the live matrix that actually drives
        the mesh rather than a snapshot that silently diverges from it.
        """
        return self._inter_W

    @property
    def projection_weights(self) -> np.ndarray:
        """Column → executive projection matrix."""
        return self._projection

    def alife_mesh_state(self) -> dict:
        """The mesh's real state in the shape the ALife extension layer reads.

        That layer's ``tick`` looks for ``columns_W``, ``contributions``,
        ``stabilities``, ``error_rates`` and ``specialization_profiles``. It was
        being handed ``column_activations``/``inter_column_weights`` instead —
        names it does not read — so ``columns_W`` resolved to ``[]`` and the
        replicator had nothing to replicate, while every other field fell back to
        a synthetic constant. The speciation and replication mathematics were
        running against defaults rather than against Aura.

        ``columns_W`` is passed by reference on purpose: the replicator modifies
        intra-column weights in place, which is how replication becomes causal.
        """
        x_matrix = np.array([c.x for c in self.columns], dtype=np.float32)
        x_matrix = np.nan_to_num(x_matrix, nan=0.0, posinf=1.0, neginf=-1.0)

        # Contribution: how strongly a column drives the rest of the mesh —
        # its outgoing coupling mass scaled by its own activity.
        outgoing = np.abs(self._inter_W).sum(axis=0)
        outgoing = outgoing / (outgoing.max() + 1e-6)
        activity = np.abs(self._column_activations)
        activity = activity / (activity.max() + 1e-6)
        contributions = (0.5 * outgoing + 0.5 * activity).astype(np.float32)

        # Stability: low within-column dispersion = stable. Inverted and
        # normalized so 1.0 is maximally stable.
        dispersion = x_matrix.std(axis=1)
        stabilities = (1.0 / (1.0 + dispersion)).astype(np.float32)

        # Error rate: saturation is this mesh's observable failure mode — a
        # column pinned at the clip bound has stopped carrying information.
        error_rates = np.mean(np.abs(x_matrix) > 0.99, axis=1).astype(np.float32)

        return {
            "columns_W": [c.W for c in self.columns],  # by reference — in-place
            "contributions": contributions,
            "stabilities": stabilities,
            "error_rates": error_rates,
            "specialization_profiles": self._specialization_profiles(x_matrix),
            # Kept for readers that want the coupling/activation view too.
            "column_activations": self._column_activations,
            "inter_column_weights": self._inter_W,
        }

    def _specialization_profiles(self, x_matrix: np.ndarray) -> np.ndarray:
        """(columns, 8) specialization profile from real column state.

        Eight coarse dimensions the mesh can actually observe about itself, so
        speciation clusters on measured structure rather than on a zero matrix.
        """
        n = self.cfg.columns
        prof = np.zeros((n, 8), dtype=np.float32)
        prof[:, 0] = np.abs(x_matrix).mean(axis=1)          # activity
        prof[:, 1] = x_matrix.std(axis=1)                    # dispersion
        prof[:, 2] = (x_matrix > 0).mean(axis=1)             # excitatory balance
        prof[:, 3] = np.abs(self._inter_W).sum(axis=1)       # incoming mass
        prof[:, 4] = np.abs(self._inter_W).sum(axis=0)       # outgoing mass
        prof[:, 5] = np.array(
            [float(np.linalg.norm(c.W)) for c in self.columns], dtype=np.float32
        )                                                     # recurrent strength
        prof[:, 6] = np.array(
            [float(c.tier.value) if hasattr(c.tier, "value") else 0.0
             for c in self.columns],
            dtype=np.float32,
        ) if self.columns else 0.0                            # tier identity
        prof[:, 7] = np.abs(self._column_activations)         # current drive
        # Normalize each dimension so no single scale dominates clustering.
        for j in range(8):
            span = float(prof[:, j].max() - prof[:, j].min())
            if span > 1e-6:
                prof[:, j] = (prof[:, j] - prof[:, j].min()) / span
        return np.nan_to_num(prof, nan=0.0, posinf=1.0, neginf=0.0)

    def apply_inter_column_coupling(
        self, weights: np.ndarray, *, blend: float = 0.1, source: str = "unknown"
    ) -> bool:
        """Blend an externally computed coupling matrix into the live mesh.

        This is what makes the ALife/Lenia layer causal rather than telemetry:
        its kernel produces a replacement coupling matrix, and without a way to
        write it back the entire computation ended at a dataclass field.

        Blended rather than assigned, and bounded by the same clip the mesh's own
        normalization applies. A hard overwrite would let one layer discard the
        STDP-learned structure in a single tick; the mesh's dynamics must stay
        the composition of its influences, not the last writer.

        Args:
            weights: (columns, columns) coupling matrix.
            blend:   fraction of the new matrix to mix in (0–1).
            source:  who is writing, for degradation attribution.

        Returns:
            True when applied; False when refused (bad shape / non-finite).
        """
        try:
            W = np.asarray(weights, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            _record_neural_mesh_degradation(
                exc,
                action=f"refused inter-column coupling from {source}: not an array",
                severity="warning",
            )
            return False

        expected = (self.cfg.columns, self.cfg.columns)
        if W.shape != expected:
            _record_neural_mesh_degradation(
                ValueError(f"coupling shape {W.shape} != {expected}"),
                action=f"refused inter-column coupling from {source}: wrong shape",
                severity="warning",
            )
            return False
        if not np.all(np.isfinite(W)):
            _record_neural_mesh_degradation(
                ValueError("coupling matrix contained non-finite values"),
                action=f"refused inter-column coupling from {source}: non-finite",
                severity="warning",
            )
            return False

        alpha = float(np.clip(blend, 0.0, 1.0))
        if alpha <= 0.0:
            return False
        self._inter_W = np.clip(
            (1.0 - alpha) * self._inter_W + alpha * W, -1.0, 1.0
        ).astype(np.float32)
        return True

    def _consume_buffer(self, attr: str, expected_cols: int,
                        offset: int = 0) -> np.ndarray | None:
        """Atomically consume an injection buffer."""
        buf = getattr(self, attr)
        if buf is None:
            return None
        setattr(self, attr, None)
        expected_len = expected_cols * self.cfg.neurons_per_column
        try:
            buf = np.asarray(buf, dtype=np.float32).reshape(-1)
        except (TypeError, ValueError) as exc:
            _record_neural_mesh_degradation(
                exc,
                action=f"discarded malformed NeuralMesh buffer {attr}",
                severity="warning",
                extra={"expected_len": expected_len},
            )
            return np.zeros(expected_len, dtype=np.float32)
        if not np.all(np.isfinite(buf)):
            _record_neural_mesh_degradation(
                ValueError(f"NeuralMesh buffer {attr} contained non-finite values"),
                action=f"sanitized NeuralMesh buffer {attr}",
                severity="warning",
                extra={"expected_len": expected_len},
            )
            buf = np.nan_to_num(buf, nan=0.0, posinf=1.0, neginf=-1.0)
        if len(buf) < expected_len:
            padded = np.zeros(expected_len, dtype=np.float32)
            padded[:len(buf)] = buf[:expected_len]
            return np.clip(padded, -1.0, 1.0)
        return np.clip(buf[:expected_len].astype(np.float32, copy=False), -1.0, 1.0)

    @staticmethod
    def _foreground_request_active() -> bool:
        """Yield plasticity work to the live conversation lane."""
        try:
            from core.container import ServiceContainer

            gate = ServiceContainer.get("inference_gate", default=None)
            mlx = getattr(gate, "_mlx_client", None)
            if mlx is None or not hasattr(mlx, "get_lane_status"):
                return False

            lane = mlx.get_lane_status()
            if bool(lane.get("foreground_owned")):
                return True

            started_at = float(lane.get("current_request_started_at", 0.0) or 0.0)
            completed_at = float(lane.get("last_generation_completed_at", 0.0) or 0.0)
            return started_at > 0.0 and started_at > completed_at
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            _record_neural_mesh_degradation(
                exc,
                action="allowed NeuralMesh plasticity because foreground lane status was unavailable",
                severity="warning",
            )
            logger.debug("Foreground lane status unavailable, allowing plasticity: %s", exc)
            return False

    # ── STDP ─────────────────────────────────────────────────────────

    def _apply_stdp(self, now: float):
        """Spike-Timing-Dependent Plasticity within each column.

        Pre-before-post → potentiate (causal)
        Post-before-pre → depress   (acausal)
        """
        if self._foreground_request_active():
            return

        lr = self.cfg.stdp_lr * self._modulatory_state[1]
        inhibitory_lr = lr * self.cfg.inhibitory_stdp_rate_ratio
        inhibitory_window, inhibitory_window_valid = _finite_float(
            self.cfg.inhibitory_stdp_window, 0.02
        )
        inhibitory_window, inhibitory_window_unchanged = _clamp_float(
            inhibitory_window, lower=1e-6, upper=5.0
        )
        if not (inhibitory_window_valid and inhibitory_window_unchanged):
            _record_neural_mesh_degradation(
                ValueError("NeuralMesh inhibitory plasticity window was out of bounds"),
                action="normalized the inhibitory plasticity window",
                severity="warning",
                extra={"window": inhibitory_window},
            )
        inhibitory_alpha, alpha_valid = _finite_float(
            self.cfg.inhibitory_stdp_depression, 0.2
        )
        inhibitory_alpha, alpha_unchanged = _clamp_float(
            inhibitory_alpha, lower=0.0, upper=10.0
        )
        if not (alpha_valid and alpha_unchanged):
            _record_neural_mesh_degradation(
                ValueError("NeuralMesh inhibitory plasticity constant was out of bounds"),
                action="normalized the inhibitory depression constant",
                severity="warning",
                extra={"alpha": inhibitory_alpha},
            )
        lr, lr_valid = _finite_float(lr, self.cfg.stdp_lr)
        lr, lr_unchanged = _clamp_float(lr, lower=0.0, upper=0.05)
        window, window_valid = _finite_float(self.cfg.stdp_window, 0.02)
        window, window_unchanged = _clamp_float(window, lower=1e-6, upper=5.0)
        a_plus, a_plus_valid = _finite_float(self.cfg.stdp_potentiation, 1.0)
        a_minus, a_minus_valid = _finite_float(self.cfg.stdp_depression, 0.5)
        a_plus, a_plus_unchanged = _clamp_float(a_plus, lower=0.0, upper=10.0)
        a_minus, a_minus_unchanged = _clamp_float(a_minus, lower=0.0, upper=10.0)
        if not all(
            (
                lr_valid,
                lr_unchanged,
                window_valid,
                window_unchanged,
                a_plus_valid,
                a_minus_valid,
                a_plus_unchanged,
                a_minus_unchanged,
            )
        ):
            _record_neural_mesh_degradation(
                ValueError("NeuralMesh STDP parameters were non-finite or out of bounds"),
                action="normalized NeuralMesh STDP parameters before plasticity update",
                severity="warning",
                extra={"lr": lr, "window": window, "a_plus": a_plus, "a_minus": a_minus},
            )

        for col in self.columns:
            t = np.nan_to_num(col.last_spike_time, nan=-1.0, posinf=-1.0, neginf=-1.0)
            col.last_spike_time = t
            # Only process neurons that have spiked at least once
            active = t > 0
            if not np.any(active):
                continue

            # `recurrent = W @ x` makes W[i, j] the weight FROM j INTO i, so
            # row i is the postsynaptic cell and column j the presynaptic one.
            # This is the axis the whole rule hangs off, and it was read the
            # other way round: the code potentiated on t_i < t_j, which under
            # this convention is POST before PRE. The mesh was running
            # anti-Hebbian, strengthening exactly the pairs a causal rule
            # weakens.
            delay = t[None, :] - t[:, None]  # pre minus post, so causal is < 0
            both = active[:, None] & active[None, :]
            causal = (delay < 0) & (delay > -window) & both
            acausal = (delay > 0) & (delay < window) & both
            near = np.clip(-np.abs(delay) / window, -20.0, 20.0)
            # The inhibitory window is its own, 20 ms against the excitatory
            # 16.8, and both come from the papers rather than from each other.
            inhibitory_near = np.clip(
                -np.abs(delay) / inhibitory_window, -20.0, 20.0
            )
            inhibitory_paired = (np.abs(delay) < inhibitory_window) & both

            # The excitatory rule, from Bi and Poo: pre before post strengthens,
            # post before pre weakens, and the two halves are not the same size.
            excitatory_dw = lr * (
                np.where(causal, a_plus * np.exp(near), 0.0)
                - np.where(acausal, a_minus * np.exp(near), 0.0)
            ).astype(np.float32)

            # The inhibitory rule, from Vogels et al. One rule for every synapse
            # was the last plasticity assumption left in the mesh, and it is not
            # what was measured: inhibitory synapses learn over a SYMMETRIC
            # window — either order of firing strengthens them — against a
            # standing depression on every presynaptic spike. What that pair of
            # terms does is hold the cell being inhibited near a target rate, so
            # inhibition tracks excitation instead of being learned by a rule
            # derived from excitatory pairs.
            presynaptic_fired = np.broadcast_to(active[None, :], both.shape)
            inhibitory_dw = inhibitory_lr * (
                np.where(inhibitory_paired, np.exp(inhibitory_near), 0.0)
                - np.where(presynaptic_fired, inhibitory_alpha, 0.0)
            ).astype(np.float32)

            # Presynaptic cell class picks the rule, because that is what the
            # synapse belongs to. Columns are presynaptic here.
            dw = np.where(col.inh_mask[None, :], inhibitory_dw, excitatory_dw)
            dw = np.nan_to_num(dw, nan=0.0, posinf=0.0, neginf=0.0)
            # Plasticity changes synapses that exist. It does not grow them.
            #
            # There was no such mask, so any two units that fired inside the
            # window acquired a synapse whether or not one had ever been wired,
            # and a column's density climbed toward full the longer it ran. All
            # the measured densities above describe the mesh at construction and
            # nothing was holding them there.
            dw = dw * (col.W != 0)

            # Both rules change a synapse's STRENGTH; neither may change what it
            # does. An inhibitory synapse is negative, so strengthening it means
            # subtracting.
            col.W = np.nan_to_num(col.W, nan=0.0, posinf=1.0, neginf=-1.0)
            polarity = np.where(col.inh_mask[None, :], -1.0, 1.0).astype(np.float32)
            old_sign = np.sign(col.W)
            col.W = col.W + dw * polarity
            # A synapse that would change sign has been driven to nothing, which
            # is where it stops.
            sign_flipped = (np.sign(col.W) != old_sign) & (old_sign != 0)
            col.W[sign_flipped] = 0.0

            # Normalize to prevent runaway
            norm = np.linalg.norm(col.W)
            if norm > 5.0:
                col.W *= 5.0 / norm
            col.W = np.nan_to_num(
                np.clip(col.W, -1.0, 1.0),
                nan=0.0,
                posinf=1.0,
                neginf=-1.0,
            ).astype(np.float32, copy=False)

    # ── Stats ────────────────────────────────────────────────────────

    def _update_stats(self):
        """Compute summary statistics for telemetry and downstream consumers.

        Reuses _cached_col_means from the batched tick to avoid redundant
        numpy operations.
        """
        # Vectorized energy: mean(|x|) per column
        x_matrix = np.array([c.x for c in self.columns], dtype=np.float32)
        x_matrix = np.nan_to_num(x_matrix, nan=0.0, posinf=1.0, neginf=-1.0)
        energies = np.mean(np.abs(x_matrix), axis=1)  # (64,)

        self._mean_column_energy = float(energies.mean())

        for tier in CorticalTier:
            mask = [c.tier == tier for c in self.columns]
            tier_e = energies[mask]
            self._tier_energies[tier] = float(tier_e.mean()) if len(tier_e) > 0 else 0.0

        # Global synchrony: reuse cached col_means from tick (no recomputation)
        col_means = getattr(self, '_cached_col_means', None)
        if col_means is not None and len(col_means) > 1:
            col_means = np.nan_to_num(col_means, nan=0.0, posinf=1.0, neginf=-1.0)
            stds = np.std(x_matrix, axis=1)
            mean_of_stds = stds.mean() + 1e-8
            std_of_means = col_means.std() + 1e-8
            self._global_synchrony = float(np.clip(std_of_means / mean_of_stds, 0.0, 1.0))

        full_state = x_matrix.reshape(-1).astype(np.float32, copy=True)
        self._cached_field_state = full_state
        projection = np.nan_to_num(self._projection, nan=0.0, posinf=1.0, neginf=-1.0)
        self._cached_executive_projection = np.nan_to_num(
            np.tanh(projection @ full_state),
            nan=0.0,
            posinf=1.0,
            neginf=-1.0,
        ).astype(np.float32)

    def _refresh_cached_snapshots(self) -> None:
        full_state = np.nan_to_num(
            np.concatenate([col.x for col in self.columns]),
            nan=0.0,
            posinf=1.0,
            neginf=-1.0,
        ).astype(np.float32, copy=True)
        self._cached_field_state = full_state
        projection = np.nan_to_num(self._projection, nan=0.0, posinf=1.0, neginf=-1.0)
        self._cached_executive_projection = np.nan_to_num(
            np.tanh(projection @ full_state),
            nan=0.0,
            posinf=1.0,
            neginf=-1.0,
        ).astype(np.float32)

    def _refresh_cached_snapshots_if_idle(self) -> None:
        if not self._lock.acquire(False):
            return
        try:
            self._refresh_cached_snapshots()
        finally:
            self._lock.release()

    # ── External API ─────────────────────────────────────────────────

    def inject_sensory(self, vector: np.ndarray):
        """Push embodiment/interoceptive signals into sensory tier columns.
        Vector length should be sensory_columns * neurons_per_column (1024).
        Shorter vectors are zero-padded. Called from EmbodiedInteroception.
        """
        expected_len = self.cfg.sensory_end * self.cfg.neurons_per_column
        self._sensory_buffer = self._coerce_signal_vector(
            vector,
            expected_len=expected_len,
            action="sanitized sensory ingress before NeuralMesh injection",
        )

    def inject_association(self, vector: np.ndarray):
        """Push cross-modal / memory / LLM signals into association tier.
        Vector length should be association_columns * neurons_per_column (2048).
        """
        expected_len = (
            self.cfg.association_end - self.cfg.sensory_end
        ) * self.cfg.neurons_per_column
        self._association_buffer = self._coerce_signal_vector(
            vector,
            expected_len=expected_len,
            action="sanitized association ingress before NeuralMesh injection",
        )

    def _publish_modulatory_state_locked(self) -> None:
        base_gain, base_plasticity, base_noise = self._base_modulatory_state
        criticality_gain, criticality_noise = self._criticality_modulatory_factors
        effective = (
            max(0.1, min(3.0, base_gain * criticality_gain)),
            max(0.0, min(5.0, base_plasticity)),
            max(0.0, min(3.0, base_noise * criticality_noise)),
        )
        self._modulatory_state = effective
        (
            self._modulatory_gain,
            self._modulatory_plasticity,
            self._modulatory_noise,
        ) = effective

    def _tier_vector(self, which: int) -> np.ndarray:
        """One multiplier per column, from the per-tier pair. 0 is gain, 1 is noise."""
        return np.array(
            [
                self._tier_modulation.get(name, (1.0, 1.0))[which]
                for name in self._tier_names
            ],
            dtype=np.float32,
        )

    def set_regional_modulation(
        self, multipliers: dict[str, tuple[float, float]] | None
    ) -> dict[str, tuple[float, float]]:
        """How much each tier scales gain and noise, on top of the global state.

        A receptor field supplies these. Passing None restores uniform, which is
        what the mesh did before it could express the difference at all: one
        scalar for every one of its 4,096 units, so dopamine arriving at the
        sensory tier and dopamine arriving at the executive tier were the same
        event. Cortex is not like that — receptor densities vary by area, and a
        transmitter's effect depends on where it lands and what receptor is
        there — and until a measurement fills these in they stay at one, which
        says the structure is unmeasured rather than guessing at it.
        """
        wanted = {"sensory": (1.0, 1.0), "association": (1.0, 1.0), "executive": (1.0, 1.0)}
        if multipliers:
            for tier, pair in multipliers.items():
                name = str(tier).rsplit(".", 1)[-1].lower()
                if name not in wanted:
                    continue
                try:
                    gain_scale = float(pair[0])
                    noise_scale = float(pair[1])
                except (TypeError, ValueError, IndexError):
                    continue
                if not (math.isfinite(gain_scale) and math.isfinite(noise_scale)):
                    continue
                wanted[name] = (
                    max(0.1, min(4.0, gain_scale)),
                    max(0.0, min(4.0, noise_scale)),
                )
        with self._modulation_lock:
            self._tier_modulation = wanted
        return dict(wanted)

    def regional_modulation(self) -> dict[str, tuple[float, float]]:
        """What each tier is currently scaling gain and noise by."""
        with self._modulation_lock:
            return dict(self._tier_modulation)

    def set_modulatory_state(
        self,
        gain: float = 1.0,
        plasticity: float = 1.0,
        noise: float = 1.0,
    ) -> None:
        """Set the neurochemical base state without erasing other controllers."""
        gain_value, gain_valid = _finite_float(gain, 1.0)
        plasticity_value, plasticity_valid = _finite_float(plasticity, 1.0)
        noise_value, noise_valid = _finite_float(noise, 1.0)
        gain_value, gain_unchanged = _clamp_float(gain_value, lower=0.1, upper=3.0)
        plasticity_value, plasticity_unchanged = _clamp_float(
            plasticity_value,
            lower=0.0,
            upper=5.0,
        )
        noise_value, noise_unchanged = _clamp_float(noise_value, lower=0.0, upper=3.0)
        if not all(
            (
                gain_valid,
                plasticity_valid,
                noise_valid,
                gain_unchanged,
                plasticity_unchanged,
                noise_unchanged,
            )
        ):
            _record_neural_mesh_degradation(
                ValueError("NeuralMesh modulatory state was non-finite or out of bounds"),
                action="normalized NeuralMesh modulatory state before applying it",
                severity="warning",
                extra={
                    "gain": gain_value,
                    "plasticity": plasticity_value,
                    "noise": noise_value,
                },
            )
        with self._modulation_lock:
            self._base_modulatory_state = (
                gain_value,
                plasticity_value,
                noise_value,
            )
            self._publish_modulatory_state_locked()

    def set_criticality_adjustment(
        self,
        *,
        gain: float = 1.0,
        noise: float = 1.0,
    ) -> None:
        """Apply bounded criticality factors over the neurochemical base state."""
        gain_value, gain_valid = _finite_float(gain, 1.0)
        noise_value, noise_valid = _finite_float(noise, 1.0)
        gain_value, gain_unchanged = _clamp_float(
            gain_value,
            lower=0.5,
            upper=2.0,
        )
        noise_value, noise_unchanged = _clamp_float(
            noise_value,
            lower=0.5,
            upper=2.0,
        )
        if not all((gain_valid, noise_valid, gain_unchanged, noise_unchanged)):
            _record_neural_mesh_degradation(
                ValueError("NeuralMesh criticality adjustment was non-finite or out of bounds"),
                action="normalized criticality modulation before applying it",
                severity="warning",
                extra={"gain": gain_value, "noise": noise_value},
            )
        with self._modulation_lock:
            self._criticality_modulatory_factors = (gain_value, noise_value)
            self._publish_modulatory_state_locked()

    def get_modulatory_state(self) -> dict[str, dict[str, float]]:
        """Return one coherent base, criticality, and effective modulation snapshot."""
        with self._modulation_lock:
            base_gain, base_plasticity, base_noise = self._base_modulatory_state
            criticality_gain, criticality_noise = self._criticality_modulatory_factors
            effective_gain, effective_plasticity, effective_noise = self._modulatory_state
        return {
            "base": {
                "gain": base_gain,
                "plasticity": base_plasticity,
                "noise": base_noise,
            },
            "criticality": {
                "gain": criticality_gain,
                "noise": criticality_noise,
            },
            "effective": {
                "gain": effective_gain,
                "plasticity": effective_plasticity,
                "noise": effective_noise,
            },
        }

    def get_executive_projection(self) -> np.ndarray:
        """Project full 4096-d state down to 64-d for LiquidSubstrate injection.
        This is how the mesh feeds into the existing consciousness core.
        """
        self._refresh_cached_snapshots_if_idle()
        return self._cached_executive_projection.copy()

    def get_field_state(self) -> np.ndarray:
        """Full 4096-dimensional activation snapshot."""
        self._refresh_cached_snapshots_if_idle()
        return self._cached_field_state.copy()

    def get_column_summary(self, col_idx: int) -> dict:
        """Per-column diagnostic."""
        if col_idx < 0 or col_idx >= len(self.columns):
            _record_neural_mesh_degradation(
                IndexError(f"NeuralMesh column index out of range: {col_idx}"),
                action="returned empty NeuralMesh column summary for invalid index",
                severity="warning",
                extra={"column_index": col_idx, "columns": len(self.columns)},
            )
            return {
                "index": col_idx,
                "tier": "UNKNOWN",
                "mean_activation": 0.0,
                "energy": 0.0,
                "std": 0.0,
                "inhibitory_activity": 0.0,
                "excitatory_activity": 0.0,
                "weight_norm": 0.0,
            }
        col = self.columns[col_idx]
        x = np.nan_to_num(col.x, nan=0.0, posinf=1.0, neginf=-1.0)
        weights = np.nan_to_num(col.W, nan=0.0, posinf=1.0, neginf=-1.0)
        return {
            "index": col.index,
            "tier": col.tier.name,
            "mean_activation": float(np.mean(x)),
            "energy": float(np.mean(np.abs(x))),
            "std": float(np.std(x)),
            "inhibitory_activity": float(np.mean(np.abs(x[col.inh_mask]))) if np.any(col.inh_mask) else 0.0,
            "excitatory_activity": float(np.mean(np.abs(x[~col.inh_mask]))),
            "weight_norm": float(np.linalg.norm(weights)),
        }

    def get_tier_energy(self, tier: CorticalTier) -> float:
        return self._tier_energies.get(tier, 0.0)

    def get_global_synchrony(self) -> float:
        """How synchronized are the columns? 0=desynchronized, 1=fully coupled."""
        return self._global_synchrony

    def get_status(self) -> dict:
        modulation = self.get_modulatory_state()
        effective_modulation = modulation["effective"]
        return {
            "running": self._running,
            "tick_count": self._tick_count,
            "total_neurons": self.cfg.total_neurons,
            "columns": self.cfg.columns,
            "mean_energy": round(self._mean_column_energy, 4),
            "global_synchrony": round(self._global_synchrony, 4),
            "tier_energies": {t.name: round(v, 4) for t, v in self._tier_energies.items()},
            "modulatory_gain": round(effective_modulation["gain"], 3),
            "modulatory_plasticity": round(effective_modulation["plasticity"], 3),
            "modulatory_noise": round(effective_modulation["noise"], 3),
            "modulation": modulation,
            "accelerator": _MLX_ACCELERATOR,
            "accelerator_reason": _MLX_ACCELERATOR_REASON,
            "consecutive_tick_failures": self._consecutive_tick_failures,
            "last_tick_error_age_s": round(time.monotonic() - self._last_tick_error_at, 1)
            if self._last_tick_error_at
            else 0,
            "uptime_s": round(time.monotonic() - self._start_time, 1) if self._start_time else 0,
        }
