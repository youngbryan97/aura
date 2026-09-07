"""core/connectome/invariants.py — what has to stay true about the map.

A measurement is only worth something while the thing it measured is still the
thing it says it measured. These five checks sit next to the code they protect
and catch the ways this package could quietly stop meaning what it claims:

* a telemetry channel written but never declared, which is a number with no
  unit and no limits arriving on a surface that will believe it;
* a biological mapping that lost its falsifier, which turns a claim back into
  a name;
* a published constant edited away from the table it was taken from;
* a connection whose endpoints are not cells;
* a drive edge and a return edge collapsing into one key, which would make
  reciprocity an artefact of the data structure.

All five are observational. None of them can fail a boot; they report.
"""

from __future__ import annotations

from collections.abc import Iterator

from core.verify.invariants import Severity, Violation, invariant

from .integration import (
    CHANNEL_CELLS,
    CHANNEL_CONTACTS,
    CHANNEL_COVERAGE,
    CHANNEL_EI_RATIO,
    CHANNEL_FINDINGS,
    CHANNEL_FINDINGS_CONFIRMED,
    CHANNEL_SPLIT_ERRORS,
    CHANNEL_WITHIN_LAYER,
    peek_snapshot,
)

__all__ = ["WRITTEN_CHANNELS"]

#: Every channel this package writes. The invariant below checks each one is
#: declared, because a channel with a writer and no declaration is the defect
#: this repository keeps rediscovering from the other side.
WRITTEN_CHANNELS: tuple[str, ...] = (
    CHANNEL_CELLS,
    CHANNEL_CONTACTS,
    CHANNEL_EI_RATIO,
    CHANNEL_WITHIN_LAYER,
    CHANNEL_COVERAGE,
    CHANNEL_FINDINGS,
    CHANNEL_FINDINGS_CONFIRMED,
    CHANNEL_SPLIT_ERRORS,
)


@invariant(
    "connectome.channels_declared",
    scope="connectome",
    severity=Severity.WARNING,
    owner="core/connectome/integration.py",
    description="every telemetry channel the connectome writes is declared with limits",
)
def _channels_are_declared() -> Iterator[Violation]:
    from .integration import declare_telemetry

    declare_telemetry()
    try:
        from core.fsw.telemetry_dictionary import get_telemetry
    except ImportError:
        return
    dictionary = get_telemetry()
    for name in WRITTEN_CHANNELS:
        if not dictionary.is_declared(name):
            yield Violation(
                subject=name,
                message="written by core/connectome and not declared",
                remedy="declare it in core/connectome/integration.declare_telemetry",
            )


@invariant(
    "connectome.mappings_are_falsifiable",
    scope="connectome",
    severity=Severity.ERROR,
    owner="core/connectome/integration.py",
    description="every connectome mapping carries a falsifier, a source and a rival",
)
def _mappings_are_falsifiable() -> Iterator[Violation]:
    from .integration import declare_mappings

    declare_mappings()
    try:
        from core.science.neuro_reference import get_neuro_reference
    except ImportError:
        return
    reference = get_neuro_reference()
    for label in (
        "connectome.contact_multiplicity",
        "connectome.laminar_microcircuit",
        "connectome.branching_ratio",
        "connectome.sensorimotor_neck",
    ):
        mapping = reference.get(label)
        if mapping is None:
            yield Violation(
                subject=label,
                message="declared nowhere; a biological name with no mapping claims nothing",
                remedy="declare it in core/connectome/integration.declare_mappings",
            )
            continue
        if not mapping.falsifier.strip():
            yield Violation(
                subject=label,
                message="has no falsifier, so it is a name rather than a claim",
                remedy="state what measurement would show the mapping is wrong",
            )
        if not mapping.source.strip():
            yield Violation(
                subject=label,
                message="has no source to be matched against",
                remedy="cite the published measurement the mapping is checked against",
            )
        if not mapping.competing_hypothesis.strip():
            yield Violation(
                subject=label,
                message="has no rival, so it was never tested against one",
                remedy="state the competing computational hypothesis",
            )


@invariant(
    "connectome.published_constants_agree",
    scope="connectome",
    severity=Severity.ERROR,
    owner="core/connectome/types.py",
    description="the encoded cortical and H01 constants agree with the tables they came from",
)
def _published_constants_agree() -> Iterator[Violation]:
    from .microcircuit import CORTICAL_CONN_PROBS, CORTICAL_SIZES, POPULATIONS
    from .types import CORTICAL_EI_RATIO, CORTICAL_EXCITATORY, CORTICAL_INHIBITORY

    if sum(CORTICAL_SIZES) != 77_169:
        yield Violation(
            subject="CORTICAL_SIZES",
            message=f"sums to {sum(CORTICAL_SIZES)} where Potjans & Diesmann give 77,169",
            remedy="restore the population table from the reference implementation",
        )
    excitatory = sum(CORTICAL_SIZES[index] for index in (0, 2, 4, 6))
    inhibitory = sum(CORTICAL_SIZES[index] for index in (1, 3, 5, 7))
    if (excitatory, inhibitory) != (CORTICAL_EXCITATORY, CORTICAL_INHIBITORY):
        yield Violation(
            subject="CORTICAL_EI_RATIO",
            message="the ratio was not derived from the population table",
            remedy="derive it rather than writing the number",
        )
    if inhibitory and abs(CORTICAL_EI_RATIO - excitatory / inhibitory) > 1e-9:
        yield Violation(
            subject="CORTICAL_EI_RATIO",
            message=f"{CORTICAL_EI_RATIO} does not equal {excitatory}/{inhibitory}",
            remedy="derive it from the table",
        )
    if len(CORTICAL_CONN_PROBS) != len(POPULATIONS):
        yield Violation(
            subject="CORTICAL_CONN_PROBS",
            message="the matrix is not square over the eight populations",
            remedy="restore the 8x8 table",
        )
    for row_index, row in enumerate(CORTICAL_CONN_PROBS):
        for column_index, value in enumerate(row):
            if not 0.0 <= float(value) <= 1.0:
                yield Violation(
                    subject=f"CORTICAL_CONN_PROBS[{row_index}][{column_index}]",
                    message=f"{value} is not a probability",
                    remedy="restore the table from the reference implementation",
                )


@invariant(
    "connectome.no_dangling_edge",
    scope="connectome",
    severity=Severity.ERROR,
    owner="core/connectome/volume.py",
    description="every connection joins two cells the reconstruction knows about",
)
def _no_dangling_edge() -> Iterator[Violation]:
    snapshot = peek_snapshot()
    if snapshot is None:
        return
    reported = 0
    for connection in snapshot.connections.values():
        for endpoint in (connection.pre, connection.post):
            if endpoint not in snapshot.units:
                reported += 1
                if reported <= 8:
                    yield Violation(
                        subject=f"{connection.pre}->{connection.post}",
                        message=f"endpoint {endpoint} is not a cell",
                        remedy="rebuild the snapshot; an edge outlived its cell",
                    )


@invariant(
    "connectome.edge_kinds_stay_separate",
    scope="connectome",
    severity=Severity.ERROR,
    owner="core/connectome/types.py",
    description="drive and return edges for one pair never collapse into one entry",
)
def _edge_kinds_stay_separate() -> Iterator[Violation]:
    snapshot = peek_snapshot()
    if snapshot is None:
        return
    for key, connection in snapshot.connections.items():
        if len(key) != 3 or key[2] != str(connection.kind):
            yield Violation(
                subject=f"{connection.pre}->{connection.post}",
                message="the connection key does not carry its edge kind",
                remedy="key connections by (pre, post, kind); reciprocity depends on it",
            )
            return
