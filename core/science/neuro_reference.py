"""core/science/neuro_reference.py — a biological name is a claim, or it is decoration.

Ninety-four modules under ``core/`` use anatomical vocabulary. There is a
``hippocampus.py``, a basal-ganglia analogy in action selection, interoception
in the body model, a cortex, a thalamus, neuromodulators. Some of those names
carry a real computational hypothesis. Some are evocative labels for a module
that does something else entirely. Nothing in the repository can tell you which
is which, and a reader — including a future Aura reasoning about herself — will
assume the strong reading every time.

A :class:`Mapping` makes the name into a claim with an expiry:

* **the biological structure**, with a species. Rodent hippocampal replay and
  human episodic recall are different findings and combining them silently is
  how a mapping becomes an argument nobody made.
* **the computational hypothesis** it is supposed to license. Not "this is like
  a hippocampus" — what computation the structure is being claimed to perform.
* **the Aura mechanism** that implements it.
* **an evidence grade**, from the literature down to metaphor.
* **what was abstracted away**: neurons, cell types, synapses, neuromodulators,
  geometry, delays, plasticity. Card 115's bar, and the field that most often
  reads as an admission.
* **a falsifier**. What would show the mapping is wrong. A mapping with no
  falsifier is a name.

:meth:`NeuroReference.strongest_supportable_claim` is the enforcement. A claim
that leans on a mapping cannot be stronger than that mapping's grade, so a
METAPHOR-grade analogy cannot support a sentence about how Aura's memory works,
and the downgrade happens in code rather than in review.

What this deliberately does not do
----------------------------------
It does not predict neural data. Aura has no recordings and will not have any;
a mapping's grade tops out at ANALOGOUS_FUNCTION unless someone runs an
experiment that discriminates between two mappings. Saying that plainly is
better than a traceability system that implies a neuroscience programme nobody
is running.
"""

from __future__ import annotations

from core.runtime.lockdep import checked_lock
import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import IntEnum, StrEnum
from typing import Any

__all__ = [
    "Species",
    "Grade",
    "Abstracted",
    "Mapping",
    "NeuroReference",
    "get_neuro_reference",
    "reset_neuro_reference_for_test",
]


class Species(StrEnum):
    """Whose brain. Never assumed, never silently combined."""

    HUMAN = "human"
    MACAQUE = "macaque"
    RODENT = "rodent"
    INVERTEBRATE = "invertebrate"
    #: A claim about brains in general. Usually a sign nobody checked.
    UNSPECIFIED = "unspecified"


class Grade(IntEnum):
    """How much the mapping has earned, worst to best."""

    #: A name that sounds like biology. Licenses nothing.
    METAPHOR = 1
    #: The structure inspired the design. No claim that they compute alike.
    INSPIRED_BY = 2
    #: The published function and the Aura mechanism are described the same way.
    ANALOGOUS_FUNCTION = 3
    #: The connectivity constraints the biology imposes are respected here.
    CONNECTIVITY_MATCHED = 4
    #: An experiment discriminated this mapping from a competing one.
    DISCRIMINATED = 5

    @property
    def licenses(self) -> str:
        return {
            Grade.METAPHOR: "the name only; no claim about how Aura works",
            Grade.INSPIRED_BY: "a design lineage, not a functional equivalence",
            Grade.ANALOGOUS_FUNCTION: "that the two are described as doing the same job",
            Grade.CONNECTIVITY_MATCHED: "that the information flow matches, with stated mismatches",
            Grade.DISCRIMINATED: "that a competing mapping was tested and lost",
        }[self]


class Abstracted(StrEnum):
    """What the analogy leaves out. Every mapping names these."""

    NEURONS = "neurons"
    CELL_TYPES = "cell_types"
    SYNAPSES = "synapses"
    NEUROMODULATORS = "neuromodulators"
    GEOMETRY = "geometry"
    CONDUCTION_DELAYS = "conduction_delays"
    PLASTICITY_RULES = "plasticity_rules"
    ENERGY = "energy"


#: Everything a purely computational analogue abstracts away. Named once so a
#: mapping that abstracts all of it can say so without listing eight items.
EVERYTHING_PHYSICAL = tuple(Abstracted)


@dataclass(frozen=True, slots=True)
class Mapping:
    """One biological name in the codebase, and what it is entitled to claim."""

    label: str
    module: str
    structure: str
    species: Species
    hypothesis: str
    grade: Grade
    abstracted: tuple[Abstracted, ...]
    falsifier: str
    #: A competing computational hypothesis for the same structure, when one
    #: exists. Card 110: a mapping with no rival was never tested against one.
    competing_hypothesis: str = ""
    #: Where the biology comes from. A DOI, a title, a review.
    source: str = ""
    #: Stable ontology id for the structure, so two spellings resolve to one
    #: circuit. Empty until someone links it.
    ontology_id: str = ""

    def __post_init__(self) -> None:
        if not self.falsifier.strip():
            raise ValueError(
                f"mapping {self.label!r} names no falsifier; a biological analogy that "
                "nothing could refute is a name, not a claim"
            )
        if not self.abstracted:
            raise ValueError(
                f"mapping {self.label!r} lists nothing it abstracts away; every "
                "computational analogue of a brain structure abstracts something"
            )
        if self.grade >= Grade.ANALOGOUS_FUNCTION and self.species is Species.UNSPECIFIED:
            raise ValueError(
                f"mapping {self.label!r} claims {self.grade.name} for an unspecified "
                "species; rodent and human findings are different findings"
            )
        if self.grade >= Grade.CONNECTIVITY_MATCHED and not self.source:
            raise ValueError(
                f"mapping {self.label!r} claims {self.grade.name} with no source"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "module": self.module,
            "structure": self.structure,
            "species": self.species.value,
            "hypothesis": self.hypothesis,
            "grade": self.grade.name.lower(),
            "grade_value": int(self.grade),
            "licenses": self.grade.licenses,
            "abstracted": [a.value for a in self.abstracted],
            "falsifier": self.falsifier,
            "competing_hypothesis": self.competing_hypothesis,
            "source": self.source,
            "ontology_id": self.ontology_id,
        }


class NeuroReference:
    """Every biological name, its grade, and what a claim may lean on it for."""

    def __init__(self) -> None:
        self._lock = checked_lock("core.science.neuro_reference.NeuroReference", reentrant=True)
        self._mappings: dict[str, Mapping] = {}

    def declare(self, mapping: Mapping) -> Mapping:
        with self._lock:
            self._mappings[mapping.label] = mapping
            return mapping

    def get(self, label: str) -> Mapping | None:
        with self._lock:
            return self._mappings.get(label)

    def for_circuit(self, structure: str) -> list[Mapping]:
        """Every Aura mechanism attached to one biological circuit.

        Card 112's query, done by structure rather than by module name, so two
        modules that spell the same circuit differently still both come back.
        """
        needle = structure.strip().lower()
        with self._lock:
            return sorted(
                (
                    m for m in self._mappings.values()
                    if needle in m.structure.lower() or (m.ontology_id and m.ontology_id == structure)
                ),
                key=lambda m: m.label,
            )

    def strongest_supportable_claim(self, labels: Sequence[str]) -> dict[str, Any]:
        """The strongest thing a claim leaning on these mappings may say.

        A chain is as strong as its weakest mapping, and an unregistered label
        is weaker than any of them — a biological word nobody declared is a
        word somebody typed.
        """
        with self._lock:
            found = {label: self._mappings.get(label) for label in labels}
        unregistered = [label for label, mapping in found.items() if mapping is None]
        if unregistered:
            return {
                "grade": None,
                "licenses": "nothing: " + ", ".join(unregistered) + " is not a declared mapping",
                "unregistered": unregistered,
            }
        weakest = min(found.values(), key=lambda m: int(m.grade))
        return {
            "grade": weakest.grade.name.lower(),
            "licenses": weakest.grade.licenses,
            "limited_by": weakest.label,
            "species": sorted({m.species.value for m in found.values()}),
            "mixes_species": len({m.species for m in found.values()}) > 1,
        }

    def audit(self) -> dict[str, Any]:
        with self._lock:
            mappings = list(self._mappings.values())
        by_grade: dict[str, int] = {}
        for mapping in mappings:
            by_grade[mapping.grade.name.lower()] = by_grade.get(mapping.grade.name.lower(), 0) + 1
        return {
            "mappings": len(mappings),
            "by_grade": dict(sorted(by_grade.items())),
            "by_species": {
                species.value: sum(1 for m in mappings if m.species is species)
                for species in Species
                if any(m.species is species for m in mappings)
            },
            "without_a_competing_hypothesis": [
                m.label for m in mappings if not m.competing_hypothesis
            ],
            "without_a_source": [m.label for m in mappings if not m.source],
            "metaphor_only": [m.label for m in mappings if m.grade is Grade.METAPHOR],
        }

    def mappings(self) -> list[Mapping]:
        with self._lock:
            return sorted(self._mappings.values(), key=lambda m: m.label)


_lock = checked_lock("core.science.neuro_reference.singleton")
_reference: NeuroReference | None = None


def get_neuro_reference() -> NeuroReference:
    global _reference
    with _lock:
        if _reference is None:
            _reference = NeuroReference()
            _install_declared(_reference)
        return _reference


def reset_neuro_reference_for_test(*, declared: bool = False) -> NeuroReference:
    global _reference
    with _lock:
        _reference = NeuroReference()
        if declared:
            _install_declared(_reference)
        return _reference


def _install_declared(reference: NeuroReference) -> None:
    """The biological names this work can honestly grade.

    Four, and three of them are INSPIRED_BY. That ratio is the finding: most
    anatomical vocabulary in a cognitive architecture is design lineage, and
    writing it down at its real grade is what stops it reading as anatomy.
    """
    reference.declare(
        Mapping(
            label="global_workspace",
            module="core/consciousness/global_workspace.py",
            structure="thalamocortical broadcast",
            species=Species.HUMAN,
            hypothesis=(
                "A capacity-limited competition makes one content available to many "
                "otherwise independent processes at once."
            ),
            grade=Grade.ANALOGOUS_FUNCTION,
            abstracted=EVERYTHING_PHYSICAL,
            falsifier=(
                "Lesioning the broadcast leaves every downstream consumer unchanged, or "
                "consumers turn out to read the candidate list directly."
            ),
            competing_hypothesis=(
                "The competition is a scheduler and the 'broadcast' is a side effect of "
                "ordering, with no downstream consumer that needs it to be exclusive."
            ),
            source="Global workspace theory as a functional account; no neural recording here.",
        )
    )
    reference.declare(
        Mapping(
            label="hippocampus",
            module="core/memory/hippocampus.py",
            structure="hippocampus",
            species=Species.RODENT,
            hypothesis=(
                "Rapid one-shot encoding of episodes that are later replayed into a "
                "slower semantic store."
            ),
            grade=Grade.INSPIRED_BY,
            abstracted=EVERYTHING_PHYSICAL,
            falsifier=(
                "Episodes never consolidate into the semantic store, or the store learns "
                "just as well when the fast path is removed."
            ),
            competing_hypothesis=(
                "The fast store is a write buffer and consolidation is compaction, with "
                "no learning advantage over writing to the slow store directly."
            ),
            source="Complementary learning systems as a design lineage, not a measurement.",
        )
    )
    reference.declare(
        Mapping(
            label="basal_ganglia_selection",
            module="core/consciousness/global_workspace.py",
            structure="basal ganglia",
            species=Species.UNSPECIFIED,
            hypothesis=(
                "Action selection by disinhibition: everything is suppressed and the "
                "winner is released rather than chosen."
            ),
            grade=Grade.INSPIRED_BY,
            abstracted=EVERYTHING_PHYSICAL,
            falsifier=(
                "Aura's arbitration turns out to be selection by ranking with no "
                "inhibitory step, which is what the code currently does."
            ),
            competing_hypothesis="Plain argmax over a scored list, with fatigue as the only dynamics.",
        )
    )
    reference.declare(
        Mapping(
            label="interoception",
            module="core/being/thought_interoception.py",
            structure="insular and interoceptive pathways",
            species=Species.HUMAN,
            hypothesis=(
                "Internal state is sensed by the same machinery that senses the world, "
                "and that sensing changes what the system does next."
            ),
            grade=Grade.INSPIRED_BY,
            abstracted=EVERYTHING_PHYSICAL,
            falsifier=(
                "Interoceptive readings never change a decision, which would make the "
                "pathway a reporter rather than a sense."
            ),
            competing_hypothesis=(
                "Self-state is telemetry: read by logging and by nothing that decides."
            ),
        )
    )
