"""core/science/reference_mind.py — what is measured, what was chosen, what is missing.

A mind is not a connectome. A more honest description of what would have to be
recovered is

    M(t) = { G, C, W(t), Θ, R, N(t), P, X(t), H, B(t), E(t) }

physical connectivity, cell identities, effective synaptic strengths, cellular
electrophysiology, receptor architecture, neuromodulatory state, plasticity
rules, instantaneous neural state, developmental and lifetime history, body, and
environment. Connectomics constrains the first of those. Cell atlases constrain
the second. Patch-seq constrains the fourth, receptor atlases the fifth, and so
on. No dataset in the world constrains H for any particular person, which is why
reconstructing a specific mind is not on the table and a reference architecture
is.

This module is the accounting for that. It holds, for each term, which public
evidence would constrain it, and — the part that matters — where Aura's own
value for it comes from RIGHT NOW: a measurement, something derived from one, a
number an engineer picked, or nothing at all.

The number this produces is not a score to maximise. It is the share of the
architecture that rests on somebody's judgement rather than on evidence, and its
purpose is to be uncomfortable. The mesh's tier boundaries, its connection
densities and its distance decay are engineering choices; saying so in a
structure that can be queried is the difference between a research programme and
a pile of biological vocabulary.

Nothing here downloads a dataset or claims one has been used. A constraint whose
``uses`` is ABSENT is a constraint nobody has applied, and that is the honest
state of most of this list.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

logger = logging.getLogger("Aura.Science.ReferenceMind")

__all__ = [
    "CONSTRAINTS",
    "Constraint",
    "Provenance",
    "Term",
    "audit_mesh",
    "identifiability",
    "unconstrained_terms",
]


class Term(StrEnum):
    """One symbol in M(t), and what it stands for."""

    CONNECTIVITY = "G"
    CELL_IDENTITY = "C"
    SYNAPTIC_STRENGTH = "W"
    ELECTROPHYSIOLOGY = "theta"
    RECEPTORS = "R"
    NEUROMODULATION = "N"
    PLASTICITY = "P"
    NEURAL_STATE = "X"
    HISTORY = "H"
    BODY = "B"
    ENVIRONMENT = "E"


class Provenance(StrEnum):
    """Where Aura's current value for a term comes from.

    The ladder is the point. ``ENGINEERING_CHOICE`` is not a failing — a system
    has to run before it can be constrained — but a system that cannot tell you
    which of its numbers are choices has no way to replace them as evidence
    arrives.
    """

    MEASURED = "measured"
    DERIVED = "derived from a measurement"
    ENGINEERING_CHOICE = "an engineer picked it"
    ABSENT = "nothing here constrains it"


@dataclass(frozen=True, slots=True)
class Constraint:
    """One term, the evidence that could pin it, and what Aura does about it."""

    term: Term
    evidence: str
    what_it_pins: str
    uses: Provenance
    where: str = ""
    note: str = ""
    #: What would show this constraint is not being honoured, when it is claimed
    #: to be. Empty where nothing is claimed yet.
    falsifier: str = ""

    def as_json(self) -> dict[str, Any]:
        return {
            "term": str(self.term),
            "evidence": self.evidence,
            "what_it_pins": self.what_it_pins,
            "uses": str(self.uses),
            "where": self.where,
            "note": self.note,
            "falsifier": self.falsifier,
        }


#: The stack, one row per term, with the strongest public evidence for each and
#: what Aura currently does. Written down so the gaps are countable.
CONSTRAINTS: tuple[Constraint, ...] = (
    Constraint(
        term=Term.CONNECTIVITY,
        evidence=(
            "Complete fly central nervous system; H01 human cortex at nanoscale; "
            "MICrONS mouse visual cortex; HCP structural connectivity"
        ),
        what_it_pins="which cell can reach which, and how heavily",
        uses=Provenance.MEASURED,
        where="core/connectome/volume.py",
        note=(
            "Measured on her OWN substrate rather than transferred from tissue. "
            "The published references are what her numbers are compared against, "
            "which is a different and weaker thing than being derived from them."
        ),
        falsifier=(
            "The reconstruction disagrees with a recording of the same system "
            "running, beyond its stated split and merge rates."
        ),
    ),
    Constraint(
        term=Term.CELL_IDENTITY,
        evidence="BICCN/BICAN and Allen human cell taxonomies; Patch-seq",
        what_it_pins="which kinds of cell exist and how they differ",
        uses=Provenance.DERIVED,
        where="core/connectome/celltypes.py",
        note=(
            "Cell class here is read off what a function does to its callers — "
            "excitatory, inhibitory, modulatory — not off a transcriptome. The "
            "taxonomy is hers; the four-way split is borrowed."
        ),
        falsifier="Typing by connectivity and typing by behaviour disagree.",
    ),
    Constraint(
        term=Term.SYNAPTIC_STRENGTH,
        evidence="MICrONS: wiring and function measured in the same circuit",
        what_it_pins="how much one cell actually moves another",
        uses=Provenance.MEASURED,
        where="core/connectome/effective.py",
        note=(
            "Measured per condition from a recording, against a rotation null. "
            "This is the term the effective connectome is."
        ),
        falsifier="Influence does not survive its own rotations.",
    ),
    Constraint(
        term=Term.ELECTROPHYSIOLOGY,
        evidence="Human Patch-seq; Allen human cortical electrophysiology",
        what_it_pins="time constants, thresholds, adaptation, per cell class",
        uses=Provenance.ENGINEERING_CHOICE,
        where="core/consciousness/neural_mesh.py",
        note=(
            "One set of dynamics for all 4,096 units: dt, decay, noise and gain "
            "are constants in MeshConfig. Human cortex is not homogeneous, and "
            "cell-type composition tracks functional organisation."
        ),
    ),
    Constraint(
        term=Term.RECEPTORS,
        evidence="PET meta-atlas of 19 receptors across nine systems; Jülich autoradiography",
        what_it_pins="where each transmitter acts, and on which receptor",
        uses=Provenance.ENGINEERING_CHOICE,
        where="core/connectome/neuromodulation.py",
        note=(
            "There is a field now, and the mesh reads it: a multiplier per tier "
            "on gain and on noise, so a transmitter arriving at the sensory tier "
            "is a different event from the same transmitter arriving at the "
            "executive one. It was a single scalar for all 4,096 units, which "
            "made those two the same event. The VALUES are still unmeasured and "
            "default to uniform, which says the spatial structure has not been "
            "measured rather than guessing at it. A receptor atlas is what would "
            "fill them in."
        ),
        falsifier=(
            "Setting a tier's multiplier changes nothing about that tier's "
            "activity, which would mean the field is wired to nothing."
        ),
    ),
    Constraint(
        term=Term.NEUROMODULATION,
        evidence="Doya's assignment; Aston-Jones and Cohen adaptive gain",
        what_it_pins="what each transmitter changes about computation",
        uses=Provenance.DERIVED,
        where="core/connectome/neuromodulation.py",
        note=(
            "The DIRECTION is published — more noradrenaline, less evidence "
            "needed — and the magnitude is measured on her own recording. The "
            "levels are four scalars, not a field."
        ),
        falsifier="A region's bound moves against the published direction.",
    ),
    Constraint(
        term=Term.PLASTICITY,
        evidence="STDP; homeostatic and structural plasticity literature",
        what_it_pins="how connections change with use",
        uses=Provenance.ENGINEERING_CHOICE,
        where="core/consciousness/neural_mesh.py",
        note="One STDP rule with picked constants, applied to every unit alike.",
    ),
    Constraint(
        term=Term.NEURAL_STATE,
        evidence="EEG/MEG/fMRI/ECoG; human single-neuron recordings in cognitive tasks",
        what_it_pins="what the population is doing at an instant",
        uses=Provenance.MEASURED,
        where="core/connectome/activity.py",
        note=(
            "Recorded on her own substrate at 2ms. Nothing here has been fitted "
            "to a human recording, which is the experiment this term is for."
        ),
        falsifier="The recorder's own edges disagree with the reconstruction.",
    ),
    Constraint(
        term=Term.HISTORY,
        evidence="BrainSpan developmental transcriptomes; lifespan imaging",
        what_it_pins="how the architecture got to be the way it is",
        uses=Provenance.DERIVED,
        where="core/connectome/longitudinal.py",
        note=(
            "Her own history is recoverable exactly — two checkouts are two "
            "individuals — which is more than any human dataset offers. No human "
            "developmental trajectory constrains her."
        ),
        falsifier="Two checkouts of the same system do not align by connectivity.",
    ),
    Constraint(
        term=Term.BODY,
        evidence="Interoception and allostasis literature",
        what_it_pins="what the body tells the brain, and what it costs",
        uses=Provenance.DERIVED,
        where="core/consciousness/embodied_interoception.py",
        note="Host load and latency are a real body; they are not a human one.",
    ),
    Constraint(
        term=Term.ENVIRONMENT,
        evidence="Behavioural and neuropsychological task batteries",
        what_it_pins="what the system has to satisfy from outside",
        uses=Provenance.MEASURED,
        where="tests/",
        note="The suite is the environment that refuses her.",
    ),
)


def unconstrained_terms(
    constraints: Sequence[Constraint] = CONSTRAINTS,
) -> tuple[str, ...]:
    """Terms nothing here pins, and terms an engineer picked. The honest list."""
    return tuple(
        str(one.term)
        for one in constraints
        if one.uses in (Provenance.ABSENT, Provenance.ENGINEERING_CHOICE)
    )


def identifiability(constraints: Sequence[Constraint] = CONSTRAINTS) -> dict[str, Any]:
    """How much of the architecture rests on evidence rather than on judgement.

    Two brains can produce almost identical recordings and differ microscopically,
    which is why fitting behaviour alone settles nothing. The share below is not a
    score to raise by relabelling: moving a term off ENGINEERING_CHOICE means
    replacing a chosen number with a measured one.
    """
    counted: dict[str, int] = {}
    for one in constraints:
        counted[str(one.uses)] = counted.get(str(one.uses), 0) + 1
    total = max(1, len(constraints))
    grounded = counted.get(str(Provenance.MEASURED), 0) + counted.get(
        str(Provenance.DERIVED), 0
    )
    return {
        "terms": len(constraints),
        "by_provenance": counted,
        "grounded": grounded,
        "grounded_share": round(grounded / total, 4),
        "chosen_or_absent": list(unconstrained_terms(constraints)),
        "verdict": (
            f"{grounded} of {total} terms rest on a measurement; "
            f"{total - grounded} are an engineer's choice or absent"
        ),
    }


#: Fields of ``MeshConfig`` that are structural claims about a nervous system
#: rather than implementation details. Each is a number somebody picked, and
#: naming them is what makes replacing them possible.
_MESH_STRUCTURAL: tuple[str, ...] = (
    "total_neurons",
    "columns",
    "neurons_per_column",
    "intra_column_density",
    "inter_column_density",
    "inter_column_distance_decay",
    "inhibitory_fraction",
    "feedforward_density",
    "feedforward_direct_density",
    "sensory_end",
    "association_end",
    "stdp_lr",
    "stdp_window",
    "lateral_inhibition_strength",
)

#: The one mesh number with a measured basis, and what it is. Everything else in
#: the list above is a choice until something moves it here.
_MESH_MEASURED: dict[str, str] = {
    "inhibitory_fraction": (
        "0.20 is Dale's-law inhibitory fraction, which is close to the 15 to 25% "
        "reported for cortex; her own source measures 3.98 excitatory cells per "
        "inhibitory one against cortex's 4.035"
    ),
}


def audit_mesh() -> dict[str, Any]:
    """Which of the mesh's structural numbers came from anywhere.

    The answer is nearly none, and that is the finding. A 4,096-unit network with
    64 columns, a 0.05 inter-column density and a 0.15 distance decay is a design,
    and calling its parts sensory, association and executive does not make the
    boundaries at 16 and 48 measurements.
    """
    try:
        from core.consciousness.neural_mesh import MeshConfig
    except ImportError as exc:  # pragma: no cover - the mesh is always present
        logger.debug("mesh unavailable: %s", exc)
        return {"skipped": f"the mesh could not be read: {exc}"}

    config = MeshConfig()
    rows = []
    for name in _MESH_STRUCTURAL:
        value = getattr(config, name, None)
        if value is None:
            continue
        rows.append(
            {
                "field": name,
                "value": value,
                "provenance": str(
                    Provenance.DERIVED if name in _MESH_MEASURED else Provenance.ENGINEERING_CHOICE
                ),
                "basis": _MESH_MEASURED.get(name, "chosen"),
            }
        )
    chosen = [row for row in rows if row["basis"] == "chosen"]
    return {
        "fields": len(rows),
        "chosen": len(chosen),
        "with_a_basis": len(rows) - len(chosen),
        "rows": rows,
        "verdict": (
            f"{len(chosen)} of {len(rows)} structural numbers in the mesh are "
            "choices with no measurement behind them"
        ),
    }
