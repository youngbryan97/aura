"""core/connectome/cortical_constants.py — the mesh's numbers, from measurements.

Operationally: this measures nothing itself. It holds published cortical
measurements with their sources, and derives from them the structural and
dynamical constants the neural mesh needs, so that a number in ``MeshConfig`` is
either traceable to a paper or is explicitly recorded as a choice nobody has
measured.

The audit in ``core.science.reference_mind`` counted thirteen of the mesh's
fourteen structural numbers as choices with no measurement behind them. That is
not a small thing: a 4,096-unit network with tier boundaries at 16 and 48, a
0.05 inter-column density and a 0.15 distance decay is a design, and calling its
parts sensory, association and executive does not make the indices findings.

Four sources, all already cited elsewhere in this package:

* Potjans & Diesmann (2014), Cerebral Cortex 24(3):785-806 — the cortical
  microcircuit under 1 mm^2: eight populations, their sizes, the 8x8 connection
  probability matrix, and the membrane and synaptic time constants.
* Shapson-Coe et al. (2024), Science 384:adk4858 — H01, a cubic millimetre of
  human temporal cortex: how many contacts a connected pair makes.
* Bi & Poo (1998), J Neurosci 18(24):10464-72 — spike-timing dependent
  plasticity in hippocampal culture: the two time constants of the window and
  the asymmetry between potentiation and depression.
* Markram et al. (2004), Nat Rev Neurosci 5(10):793-807 — interneuron
  diversity, for the ratio of inhibitory to excitatory membrane time constants.
* Vogels, Sprekeler, Zenke, Clopath & Gerstner (2011), Science 334:1569-73 —
  inhibitory spike-timing plasticity: a symmetric window and a depression term
  that holds the postsynaptic cell near a target rate.

What each derivation does NOT claim is stated with it. A membrane time constant
measured in millivolts across a real membrane is not the leak of a tanh unit;
what transfers is the RATIO of one timescale to another, and that is what these
functions compute. A number that cannot be derived that way is left alone and
listed in :data:`STILL_CHOSEN` with the reason no measurement reaches it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("Aura.Connectome.CorticalConstants")

__all__ = [
    "CorticalMeasurement",
    "MEASUREMENTS",
    "STILL_CHOSEN",
    "derived_mesh_constants",
    "provenance_report",
]


@dataclass(frozen=True, slots=True)
class CorticalMeasurement:
    """One published number, what it is, and where it came from."""

    name: str
    value: float
    unit: str
    source: str
    what_it_is: str

    def as_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "source": self.source,
            "what_it_is": self.what_it_is,
        }


#: Published cortical measurements. Every one is a number somebody measured in
#: tissue and wrote down, with the paper that holds it.
MEASUREMENTS: tuple[CorticalMeasurement, ...] = (
    CorticalMeasurement(
        name="membrane_time_constant_ms",
        value=10.0,
        unit="ms",
        source="Potjans & Diesmann 2014, Table 4",
        what_it_is="how long a cortical neuron's membrane takes to forget its input",
    ),
    CorticalMeasurement(
        name="synaptic_time_constant_ms",
        value=0.5,
        unit="ms",
        source="Potjans & Diesmann 2014, Table 4",
        what_it_is="how long one synaptic current lasts",
    ),
    CorticalMeasurement(
        name="refractory_ms",
        value=2.0,
        unit="ms",
        source="Potjans & Diesmann 2014, Table 4",
        what_it_is="how long after a spike a neuron cannot spike again",
    ),
    CorticalMeasurement(
        name="excitatory_delay_ms",
        value=1.5,
        unit="ms",
        source="Potjans & Diesmann 2014, Table 4",
        what_it_is="how long an excitatory spike takes to reach its target",
    ),
    CorticalMeasurement(
        name="stdp_potentiation_window_ms",
        value=16.8,
        unit="ms",
        source="Bi & Poo 1998, Figure 7",
        what_it_is="the time constant of the strengthening half of the STDP window",
    ),
    CorticalMeasurement(
        name="stdp_depression_window_ms",
        value=33.7,
        unit="ms",
        source="Bi & Poo 1998, Figure 7",
        what_it_is="the time constant of the weakening half of the STDP window",
    ),
    CorticalMeasurement(
        name="relative_inhibitory_synaptic_strength",
        value=4.0,
        unit="ratio",
        source="Potjans & Diesmann 2014, Table 4 (g)",
        what_it_is="how much stronger one inhibitory synapse is than one excitatory synapse",
    ),
    CorticalMeasurement(
        name="inhibitory_stdp_window_ms",
        value=20.0,
        unit="ms",
        source="Vogels et al. 2011, Science 334:1569",
        what_it_is="the time constant of the symmetric inhibitory plasticity window",
    ),
    CorticalMeasurement(
        name="inhibitory_stdp_target_rate_hz",
        value=5.0,
        unit="Hz",
        source="Vogels et al. 2011, Science 334:1569",
        what_it_is="the postsynaptic firing rate inhibitory plasticity holds a cell near",
    ),
    CorticalMeasurement(
        name="h01_single_contact_fraction",
        value=0.965,
        unit="fraction",
        source="Shapson-Coe et al. 2024",
        what_it_is="the share of connected pairs in human cortex that touch exactly once",
    ),
    CorticalMeasurement(
        name="inhibitory_membrane_time_constant_ratio",
        value=0.5,
        unit="ratio",
        source="Markram et al. 2004",
        what_it_is=(
            "how much faster a cortical interneuron's membrane is than a "
            "pyramidal cell's — interneurons run at roughly half the time constant"
        ),
    ),
)

_BY_NAME: dict[str, CorticalMeasurement] = {one.name: one for one in MEASUREMENTS}


def measurement(name: str) -> float:
    """One published value by name. Raises rather than defaulting."""
    return _BY_NAME[name].value


#: Numbers in ``MeshConfig`` that no measurement reaches, and why each one does
#: not. A choice recorded as a choice is honest; a choice recorded as a finding
#: is the defect this whole file exists against.
STILL_CHOSEN: dict[str, str] = {
    "total_neurons": (
        "A size, not a measurement. Human cortex has about 16 billion neurons; "
        "4,096 units is what fits beside a resident 27B on this host. No "
        "published number can tell you how many units a model should have — "
        "only what the model has to reproduce with them."
    ),
    "columns": (
        "Same. 64 columns of 64 units is a partition of the size above, and a "
        "cortical column's real cell count is 80,000 to 120,000."
    ),
    "neurons_per_column": "total_neurons divided by columns; determined by the two above.",
    "activation_gain": (
        "The slope of a tanh. Cortex has no tanh, and the gain that matches a "
        "biological input-output curve depends on the unit model, not on tissue."
    ),
    "noise_sigma": (
        "Membrane noise in cortex is measured in millivolts against a threshold "
        "in millivolts. This unit has neither, so the ratio does not transfer. "
        "What CAN be checked is the branching ratio the noise produces, and the "
        "criticality regulator steers on that."
    ),
    "lateral_inhibition_strength": (
        "A pooled inhibitory drive per column, where cortex has individual "
        "inhibitory cells with their own connection probabilities. The pooled "
        "form has no measured counterpart; what it approximates does."
    ),
    "feedforward_strength": (
        "A weight scale on a unit with no membrane. The DENSITY of the pathway "
        "is derived below; its strength is not."
    ),
}


#: Which cortical populations stand behind each of the mesh's three tiers.
#:
#: The mesh has a sensory band, an association band and an executive band, and
#: cortex has four laminar ones. The mapping is the standard cortical
#: feedforward account and not a choice about Aura: layer 4 is where thalamic
#: input arrives, layers 2/3 are where it is elaborated and passed sideways,
#: and layers 5 and 6 are the output layers, 5 projecting subcortically and 6
#: back to the thalamus.
#:
#: Indices are into POPULATIONS, so the arithmetic below reads the same table
#: the densities came from.
TIER_POPULATIONS: dict[str, tuple[int, ...]] = {
    "sensory": (2, 3),
    "association": (0, 1),
    "executive": (4, 5, 6, 7),
}


def derived_tier_constants() -> dict[str, dict[str, Any]]:
    """What differs between the mesh's tiers, from the layers behind them.

    One set of dynamics ran all 4,096 units. Cortex does not work that way, and
    the table that supplied the mesh's global density says so directly: the
    inhibitory share is 22.0% in layers 2/3, 20.0% in layer 4 and 17.3% in the
    output layers, and the wiring inside a band is 0.135 dense in layers 2/3,
    0.106 in layer 4 and 0.091 in the output layers. The single global figures
    the mesh has been using, 0.1986 and 0.1288, are averages of things that
    were never the same.

    Nothing here is chosen. Each number is the mean of the entries of Potjans
    and Diesmann's matrix that fall inside the band, or that band's inhibitory
    headcount over its total.
    """
    from core.connectome.microcircuit import CORTICAL_CONN_PROBS, CORTICAL_SIZES, POPULATIONS

    tiers: dict[str, dict[str, Any]] = {}
    for tier, indices in TIER_POPULATIONS.items():
        inhibitory = sum(
            CORTICAL_SIZES[index] for index in indices if POPULATIONS[index].endswith("I")
        )
        total = sum(CORTICAL_SIZES[index] for index in indices)
        probabilities = [
            CORTICAL_CONN_PROBS[row][column] for row in indices for column in indices
        ]
        tiers[tier] = {
            "inhibitory_fraction": round(inhibitory / total, 6) if total else 0.0,
            "intra_column_density": round(sum(probabilities) / len(probabilities), 6),
            "populations": tuple(POPULATIONS[index] for index in indices),
            "cells": total,
            "arithmetic": (
                f"{inhibitory} inhibitory of {total}; "
                f"{len(probabilities)} matrix entries inside the band"
            ),
        }
    return tiers


def derived_mesh_constants() -> dict[str, Any]:
    """The mesh constants that follow from published cortical measurements.

    Each entry carries what it is, what it was derived from, and the arithmetic,
    so the derivation can be checked rather than believed.
    """
    from core.connectome.microcircuit import CORTICAL_CONN_PROBS, CORTICAL_SIZES

    tau_m = measurement("membrane_time_constant_ms")
    tau_syn = measurement("synaptic_time_constant_ms")
    potentiation = measurement("stdp_potentiation_window_ms")
    depression = measurement("stdp_depression_window_ms")

    # The integration step. A simulator's step has to resolve the fastest thing
    # it simulates, and the fastest thing here is the synaptic current. Half of
    # it is the ordinary bound.
    dt_ms = tau_syn / 2.0

    # The leak per step. A membrane forgets its input with time constant tau_m,
    # so one step of dt loses dt/tau_m of what it held. This is the one place
    # the biological number transfers directly, because it is a ratio of two
    # times and both are in the same units.
    decay = dt_ms / tau_m

    # Local and long-range density, from the microcircuit's own matrix. Within
    # a population is local; across populations is not.
    within, between, pairs_within, pairs_between = 0.0, 0.0, 0, 0
    for row, source in enumerate(CORTICAL_CONN_PROBS):
        for column, probability in enumerate(source):
            if row == column:
                within += probability
                pairs_within += 1
            else:
                between += probability
                pairs_between += 1
    intra = within / max(1, pairs_within)
    inter = between / max(1, pairs_between)

    # Vogels' depression constant. Every presynaptic spike at an inhibitory
    # synapse depresses it by alpha, and the value that makes the rule settle
    # with the postsynaptic cell at rate rho is alpha = 2 * rho * tau. It is a
    # derivation from the paper's own two numbers, not a third number.
    inhibitory_window = measurement("inhibitory_stdp_window_ms")
    target_rate = measurement("inhibitory_stdp_target_rate_hz")
    inhibitory_depression = 2.0 * target_rate * (inhibitory_window / 1000.0)

    # Why cortex does not blow up. One cell in five is inhibitory and each of
    # its synapses is four times as strong, so 0.80 of unit excitation meets
    # 0.199 * 4 = 0.794 of inhibition and the network sits near balance. The
    # two numbers are not independent choices; g is what makes that fraction
    # work.
    relative_inhibition = measurement("relative_inhibitory_synaptic_strength")

    excitatory = sum(CORTICAL_SIZES[index] for index in (0, 2, 4, 6))
    inhibitory = sum(CORTICAL_SIZES[index] for index in (1, 3, 5, 7))

    return {
        "dt": {
            "value": round(dt_ms / 1000.0, 6),
            "unit": "seconds",
            "from": "synaptic_time_constant_ms",
            "arithmetic": f"{tau_syn} ms / 2, in seconds",
            "what_it_is": "the integration step, half the fastest thing simulated",
        },
        "decay": {
            "value": round(decay, 6),
            "unit": "per step",
            "from": "membrane_time_constant_ms and the step above",
            "arithmetic": f"{dt_ms} ms / {tau_m} ms",
            "what_it_is": "how much of its state a unit loses each step",
        },
        "relative_inhibitory_strength": {
            "value": round(relative_inhibition, 6),
            "unit": "ratio",
            "from": "relative_inhibitory_synaptic_strength",
            "arithmetic": f"{relative_inhibition}, as published",
            "what_it_is": (
                "how much stronger an inhibitory synapse is; with one cell in five "
                "inhibitory it is what keeps the network off its ceiling"
            ),
        },
        "inhibitory_stdp_window": {
            "value": round(inhibitory_window / 1000.0, 6),
            "unit": "seconds",
            "from": "inhibitory_stdp_window_ms",
            "arithmetic": f"{inhibitory_window} ms, in seconds",
            "what_it_is": (
                "the symmetric window inhibitory synapses learn over, where the "
                "excitatory one is asymmetric"
            ),
        },
        "inhibitory_stdp_depression": {
            "value": round(inhibitory_depression, 6),
            "unit": "per presynaptic spike",
            "from": "inhibitory_stdp_target_rate_hz and the window above",
            "arithmetic": f"2 * {target_rate} Hz * {inhibitory_window / 1000.0} s",
            "what_it_is": (
                "how much a presynaptic spike weakens an inhibitory synapse, which "
                "is what holds the cell it targets near five spikes a second"
            ),
        },
        "intra_column_density": {
            "value": round(intra, 6),
            "unit": "probability",
            "from": "Potjans & Diesmann connection matrix, the eight diagonal entries",
            "arithmetic": f"mean of the {pairs_within} within-population probabilities",
            "what_it_is": "how densely a cortical population connects to itself",
        },
        "inter_column_density": {
            "value": round(inter, 6),
            "unit": "probability",
            "from": "Potjans & Diesmann connection matrix, the 56 off-diagonal entries",
            "arithmetic": f"mean of the {pairs_between} between-population probabilities",
            "what_it_is": "how densely one cortical population connects to another",
        },
        "inhibitory_fraction": {
            "value": round(inhibitory / (excitatory + inhibitory), 6),
            "unit": "fraction",
            "from": "Potjans & Diesmann population sizes",
            "arithmetic": f"{inhibitory} / {excitatory + inhibitory}",
            "what_it_is": "inhibitory cells as a share of a cortical column",
        },
        "stdp_window": {
            "value": round(potentiation / 1000.0, 6),
            "unit": "seconds",
            "from": "stdp_potentiation_window_ms",
            "arithmetic": f"{potentiation} ms in seconds",
            "what_it_is": "how far apart two spikes can be and still strengthen a synapse",
        },
        "stdp_depression": {
            "value": round(potentiation / depression, 6),
            "unit": "ratio",
            "from": "the two STDP window constants",
            "arithmetic": f"{potentiation} ms / {depression} ms",
            "what_it_is": (
                "how much weaker each depression step is than each potentiation "
                "step, which is what makes the window net-potentiating over a "
                "spike train with no timing structure"
            ),
        },
        "inhibitory_time_constant_ratio": {
            "value": measurement("inhibitory_membrane_time_constant_ratio"),
            "unit": "ratio",
            "from": "inhibitory_membrane_time_constant_ratio",
            "arithmetic": "published directly",
            "what_it_is": "how much faster an interneuron's membrane is than a pyramidal cell's",
        },
    }


def provenance_report() -> dict[str, Any]:
    """Which mesh numbers are derived, which are chosen, and what each rests on."""
    derived = derived_mesh_constants()
    return {
        "measurements": [one.as_json() for one in MEASUREMENTS],
        "derived": derived,
        "still_chosen": dict(STILL_CHOSEN),
        "counts": {
            "derived": len(derived),
            "chosen": len(STILL_CHOSEN),
        },
        "verdict": (
            f"{len(derived)} mesh constants follow from a published measurement; "
            f"{len(STILL_CHOSEN)} are choices, each with the reason no measurement "
            "reaches it"
        ),
    }
