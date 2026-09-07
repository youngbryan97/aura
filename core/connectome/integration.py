"""core/connectome/integration.py — putting the map where the rest of the system can see it.

A measurement nobody reads is a file. This module is the seam: the telemetry
channels the connectome writes, the invariants that hold over it, the biological
mappings it is now entitled to claim, and one status dictionary the health
report can carry.

Nothing here runs at import. Reconstruction walks four thousand files and the
recorder claims a monitoring slot, so both are asked for explicitly and the
snapshot is cached once built. A process that never asks pays nothing.

The mappings need explaining. ``core/science/neuro_reference.py``
grades every biological name in the codebase and caps what a claim may lean on
it for, and its own docstring said the ceiling was ANALOGOUS_FUNCTION because
Aura has no recordings and no way to discriminate one mapping from another.
Both halves of that are now false: the activity recorder produces recordings,
and the connectivity comparisons check Aura's wiring against published measured
wiring and report the mismatches. So these mappings are declared at
CONNECTIVITY_MATCHED, each with the source it is matched against and the
measurement that would falsify it.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

logger = logging.getLogger("Aura.Connectome.Integration")

__all__ = [
    "CHANNEL_CELLS",
    "CHANNEL_CONTACTS",
    "CHANNEL_EI_RATIO",
    "CHANNEL_WITHIN_LAYER",
    "CHANNEL_COVERAGE",
    "CHANNEL_SPLIT_ERRORS",
    "CHANNEL_FINDINGS",
    "CHANNEL_FINDINGS_CONFIRMED",
    "declare_telemetry",
    "declare_mappings",
    "cached_snapshot",
    "clear_cache",
    "connectome_status",
    "publish_telemetry",
    "peek_snapshot",
    "register_health_fragment_provider",
    "record_pathology",
]

CHANNEL_CELLS = "connectome.cells"
CHANNEL_CONTACTS = "connectome.contacts"
CHANNEL_EI_RATIO = "connectome.ei_ratio"
CHANNEL_WITHIN_LAYER = "connectome.within_layer_ratio"
CHANNEL_COVERAGE = "connectome.in_volume_coverage"
CHANNEL_SPLIT_ERRORS = "connectome.split_errors"
CHANNEL_FINDINGS = "connectome.findings"
CHANNEL_FINDINGS_CONFIRMED = "connectome.findings_confirmed"

_CACHE: dict[str, Any] = {}
_LOCK = threading.RLock()
_DECLARED = False
_MAPPED = False


# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------


def declare_telemetry() -> list[str]:
    """Declare the connectome's channels. Idempotent, and safe to call early."""
    global _DECLARED
    if _DECLARED:
        return []
    try:
        from core.fsw.telemetry_dictionary import ChannelType, channel
    except ImportError as exc:
        logger.debug("telemetry dictionary unavailable: %s", exc)
        return []
    declared: list[str] = []
    failed = False
    # 0x1801-0x1807 is owned by interiority. Keep these wire IDs distinct.
    for spec in (
        {
            "identifier": 0x1901,
            "name": CHANNEL_CELLS,
            "type": ChannelType.INT,
            "unit": "count",
            "description": "cells in the current reconstruction",
            "owner": "core/connectome/volume.py",
            "group": "connectome",
            "stale_after_s": 86_400.0,
        },
        {
            "identifier": 0x1902,
            "name": CHANNEL_CONTACTS,
            "type": ChannelType.INT,
            "unit": "count",
            "description": "call sites resolved into the graph, drive and return together",
            "owner": "core/connectome/volume.py",
            "group": "connectome",
            "stale_after_s": 86_400.0,
        },
        {
            "identifier": 0x1903,
            "name": CHANNEL_EI_RATIO,
            "unit": "ratio",
            "description": (
                "excitatory cells per inhibitory cell; cortex runs 4.04 "
                "(Potjans & Diesmann 2014)"
            ),
            "owner": "core/connectome/synaptology.py",
            "group": "connectome",
            "yellow_low": 2.0,
            "red_low": 1.2,
            "yellow_high": 8.0,
            "stale_after_s": 86_400.0,
        },
        {
            "identifier": 0x1904,
            "name": CHANNEL_WITHIN_LAYER,
            "unit": "ratio",
            "description": (
                "within-layer over between-layer connection density; cortex runs 5.95"
            ),
            "owner": "core/connectome/microcircuit.py",
            "group": "connectome",
            "yellow_low": 1.0,
            "red_low": 0.3,
            "stale_after_s": 86_400.0,
        },
        {
            "identifier": 0x1905,
            "name": CHANNEL_COVERAGE,
            "unit": "fraction",
            "description": "share of in-volume call sites the reconstruction attached",
            "owner": "core/connectome/volume.py",
            "group": "connectome",
            "yellow_low": 0.40,
            "red_low": 0.20,
            "stale_after_s": 86_400.0,
        },
        {
            "identifier": 0x1907,
            "name": CHANNEL_FINDINGS,
            "type": ChannelType.INT,
            "unit": "count",
            "description": "connectome findings open, measured and candidate together",
            "owner": "core/connectome/pathology.py",
            "group": "connectome",
            "yellow_high": 150.0,
            "stale_after_s": 86_400.0,
        },
        {
            "identifier": 0x1908,
            "name": CHANNEL_FINDINGS_CONFIRMED,
            "type": ChannelType.INT,
            "unit": "count",
            "description": "connectome findings the reconstruction or a recording settles",
            "owner": "core/connectome/pathology.py",
            "group": "connectome",
            "yellow_high": 80.0,
            "stale_after_s": 86_400.0,
        },
        {
            "identifier": 0x1906,
            "name": CHANNEL_SPLIT_ERRORS,
            "type": ChannelType.INT,
            "unit": "count",
            "description": "edges seen firing that the reconstruction does not contain",
            "owner": "core/connectome/segmentation.py",
            "group": "connectome",
            "yellow_high": 200.0,
            "stale_after_s": 86_400.0,
        },
    ):
        try:
            channel(**spec)
            declared.append(str(spec["name"]))
        except (ValueError, TypeError, KeyError) as exc:
            failed = True
            logger.warning("channel %s not declared: %s", spec.get("name"), exc)
    _DECLARED = not failed
    return declared


def publish_telemetry(status: dict[str, Any] | None = None) -> dict[str, str]:
    """Write the current numbers onto the declared channels."""
    declare_telemetry()
    try:
        from core.fsw.telemetry_dictionary import write
    except ImportError as exc:
        logger.debug("telemetry write unavailable: %s", exc)
        return {}
    status = status if status is not None else connectome_status()
    written: dict[str, str] = {}
    for name, value in (
        (CHANNEL_CELLS, status.get("cells")),
        (CHANNEL_CONTACTS, status.get("contacts")),
        (CHANNEL_EI_RATIO, status.get("ei_ratio")),
        (CHANNEL_WITHIN_LAYER, status.get("within_layer_ratio")),
        (CHANNEL_COVERAGE, status.get("in_volume_coverage")),
        (CHANNEL_SPLIT_ERRORS, status.get("split_errors")),
        (CHANNEL_FINDINGS, status.get("findings")),
        (CHANNEL_FINDINGS_CONFIRMED, status.get("findings_confirmed")),
    ):
        if value is None:
            continue
        try:
            written[name] = str(write(name, value))
        except (KeyError, ValueError, TypeError) as exc:
            logger.debug("channel %s not written: %s", name, exc)
    return written


# ---------------------------------------------------------------------------
# Mappings
# ---------------------------------------------------------------------------


def declare_mappings() -> list[str]:
    """Register what this package is entitled to claim, and against what."""
    global _MAPPED
    if _MAPPED:
        return []
    try:
        from core.science.neuro_reference import (
            Abstracted,
            Grade,
            Mapping,
            Species,
            get_neuro_reference,
        )
    except ImportError as exc:
        logger.debug("neuro reference unavailable: %s", exc)
        return []
    reference = get_neuro_reference()
    declared: list[str] = []
    for mapping in (
        Mapping(
            label="connectome.contact_multiplicity",
            module="core/connectome/synaptology.py",
            structure="cortical axon-to-target contact multiplicity",
            species=Species.HUMAN,
            hypothesis=(
                "How many call sites join two functions is the same quantity as how "
                "many synapses join two neurons, so the distribution can be compared "
                "against a measured human one and the difference read as structure."
            ),
            grade=Grade.CONNECTIVITY_MATCHED,
            abstracted=(
                Abstracted.NEURONS,
                Abstracted.NEUROMODULATORS,
                Abstracted.GEOMETRY,
                Abstracted.ENERGY,
                Abstracted.PLASTICITY_RULES,
            ),
            falsifier=(
                "Show that call-site count does not track how strongly one function "
                "drives another — two pairs with the same count whose driving differs "
                "by an order of measured magnitude."
            ),
            competing_hypothesis=(
                "Call-site count is a style artefact of how the author factored the "
                "code and carries no information about coupling strength."
            ),
            source="Shapson-Coe et al., Science 384:adk4858 (2024)",
        ),
        Mapping(
            label="connectome.laminar_microcircuit",
            module="core/connectome/microcircuit.py",
            structure="canonical cortical microcircuit, four layers",
            species=Species.HUMAN,
            hypothesis=(
                "Trophic depth in the call graph plays the role laminar position plays "
                "in cortex, so the within-layer to between-layer density ratio is "
                "comparable and Aura's shortfall against cortex is a real difference in "
                "local recurrence."
            ),
            grade=Grade.CONNECTIVITY_MATCHED,
            abstracted=(
                Abstracted.NEURONS,
                Abstracted.CELL_TYPES,
                Abstracted.GEOMETRY,
                Abstracted.CONDUCTION_DELAYS,
                Abstracted.ENERGY,
            ),
            falsifier=(
                "Show the laminar assignment is arbitrary: that shuffling cells between "
                "bands leaves the within-over-between ratio unchanged, or that the "
                "hierarchy orientation flips the sign of every reported comparison."
            ),
            competing_hypothesis=(
                "Trophic depth measures nothing but call-graph nesting, and any "
                "resemblance to layers is a coincidence of how deep call chains run."
            ),
            source="Potjans & Diesmann, Cereb Cortex 24:785 (2014)",
        ),
        Mapping(
            label="connectome.branching_ratio",
            module="core/connectome/criticality.py",
            structure="cortical branching process near the critical point",
            species=Species.RODENT,
            hypothesis=(
                "Activity propagation between cells is a branching process, so the "
                "multistep regression estimator recovers its ratio from a subsampled "
                "recording and a controller can steer on it."
            ),
            grade=Grade.CONNECTIVITY_MATCHED,
            abstracted=(
                Abstracted.NEURONS,
                Abstracted.SYNAPSES,
                Abstracted.NEUROMODULATORS,
                Abstracted.ENERGY,
            ),
            falsifier=(
                "Show that avalanche exponents measured on Aura fail the crackling "
                "scaling relation while the estimator still reports a ratio near one, "
                "which would mean the ratio is not measuring what it claims."
            ),
            competing_hypothesis=(
                "Aura's activity is driven by an external request stream rather than by "
                "internal propagation, so its statistics are the stream's and no "
                "branching ratio is identifiable."
            ),
            source="Wilting & Priesemann, Nat Commun 9:2325 (2018)",
        ),
        Mapping(
            label="connectome.sensorimotor_neck",
            module="core/connectome/spine.py",
            structure="ascending and descending neurons of the neck connective",
            species=Species.INVERTEBRATE,
            hypothesis=(
                "Cells that read the world and cells that change it are separated by a "
                "narrow set of carriers that integrate rather than relay, and the "
                "integration ratio is measurable as their in-degree against the mean."
            ),
            grade=Grade.ANALOGOUS_FUNCTION,
            abstracted=(
                Abstracted.NEURONS,
                Abstracted.CELL_TYPES,
                Abstracted.GEOMETRY,
                Abstracted.CONDUCTION_DELAYS,
                Abstracted.PLASTICITY_RULES,
                Abstracted.ENERGY,
            ),
            falsifier=(
                "Run the analysis on a proofread snapshot and find that most "
                "sense-to-action pairs reach each other without passing through any "
                "small set of cells."
            ),
            competing_hypothesis=(
                "The concentration is an artefact of a reconstruction that can only see "
                "static calls, and the routes Aura actually uses go through the event "
                "bus, where there is no neck at all."
            ),
            source="Janelia FlyEM male CNS connectome v1.0 (2026)",
        ),
    ):
        try:
            reference.declare(mapping)
            declared.append(mapping.label)
        except (ValueError, TypeError) as exc:
            logger.debug("mapping %s not declared: %s", mapping.label, exc)
    _MAPPED = True
    return declared


# ---------------------------------------------------------------------------
# The cached reconstruction and its status
# ---------------------------------------------------------------------------


def peek_snapshot() -> Any:
    """The cached reconstruction if one exists, and never a new one.

    The health surface is polled often and reconstruction walks four thousand
    files, so the fragment must be able to say "nothing built here" rather than
    building one to answer.
    """
    with _LOCK:
        return _CACHE.get("snapshot")


def cached_snapshot(*, rebuild: bool = False, max_age_s: float = 3_600.0) -> Any:
    """Build the connectome once and hand back the same one afterwards.

    Reconstruction costs about fifteen seconds and the source does not change
    inside a process, so the cache is a plain age check rather than a file
    watcher. Anything that edits the tree and wants the new map asks for a
    rebuild.
    """
    with _LOCK:
        built_at = float(_CACHE.get("built_at", 0.0))
        if not rebuild and _CACHE.get("snapshot") is not None:
            if time.time() - built_at < max_age_s:
                return _CACHE["snapshot"]
    from pathlib import Path

    from .volume import VolumeReconstructor

    reconstructor = VolumeReconstructor(Path(__file__).resolve().parents[2])
    reconstructor.scan()
    snapshot = reconstructor.build()
    with _LOCK:
        _CACHE["snapshot"] = snapshot
        _CACHE["sites"] = reconstructor.ambiguous_sites
        _CACHE["built_at"] = time.time()
    return snapshot


def clear_cache() -> None:
    with _LOCK:
        _CACHE.clear()


def connectome_status(
    *,
    rebuild: bool = False,
    deep: bool = False,
    build_if_missing: bool = True,
) -> dict[str, Any]:
    """One dictionary for the health report.

    ``deep`` adds the laminar comparison, which costs a sparse solve over the
    whole graph. The default is the cheap half, so a health poll never pays for
    an analysis nobody asked for.
    """
    if not build_if_missing and not rebuild:
        snapshot = peek_snapshot()
        if snapshot is None:
            return {
                "available": False,
                "reason": "no reconstruction in this process",
                "hint": "core.connectome.integration.cached_snapshot() builds one",
            }
    else:
        try:
            snapshot = cached_snapshot(rebuild=rebuild)
        except (OSError, RuntimeError, ValueError) as exc:
            logger.info("connectome status unavailable: %s", exc)
            return {"available": False, "reason": str(exc)}

    from .synaptology import measure_multiplicity

    law = measure_multiplicity(snapshot)
    status: dict[str, Any] = {
        "available": True,
        "digest": snapshot.digest(),
        "cells": snapshot.cell_count(),
        "contacts": snapshot.contact_count(),
        "drive_pairs": law.pairs,
        "ei_ratio": round(snapshot.ei_ratio(), 4),
        "in_volume_coverage": float(snapshot.attrs.get("in_volume_coverage", 0.0)),
        "single_contact_fraction": round(law.single_fraction, 5),
        "heavy_excess_over_human_cortex": round(law.heavy_excess, 2),
        "afferent_cells": int(snapshot.attrs.get("afferent_cells", 0)),
        "efferent_cells": int(snapshot.attrs.get("efferent_cells", 0)),
        "split_errors": int(_CACHE.get("split_errors", 0)),
    }
    findings = _CACHE.get("pathology")
    if isinstance(findings, dict):
        status["findings"] = findings.get("total", 0)
        status["findings_confirmed"] = findings.get("confirmed", 0)
        status["findings_by_kind"] = findings.get("by_kind", {})
    if deep:
        from .microcircuit import assign_layers, compare_to_cortex, connection_probabilities

        assignment = assign_layers(snapshot)
        comparison = compare_to_cortex(connection_probabilities(snapshot, assignment))
        status["within_layer_ratio"] = comparison["orientation_free"][
            "aura_within_over_between"
        ]
        status["cortex_within_layer_ratio"] = comparison["orientation_free"][
            "cortex_within_over_between"
        ]
        status["trophic_incoherence"] = assignment.incoherence
        status["orientation_anchor_margin"] = assignment.anchors.get("anchor_margin", 0.0)
    return status


def register_health_fragment_provider() -> bool:
    """Publish the connectome fragment to the health surface.

    The provider never builds a reconstruction. A process that has not made one
    reports that it has not, which is the honest answer and costs nothing.
    """
    try:
        from core.runtime.health_fragments import register_health_fragment
    except ImportError as exc:
        logger.debug("health fragment register unavailable: %s", exc)
        return False

    def _fragment() -> dict[str, Any]:
        try:
            return connectome_status(build_if_missing=False)
        except (OSError, RuntimeError, ValueError, KeyError) as exc:
            return {"available": False, "reason": repr(exc)}

    register_health_fragment("connectome", _fragment)
    return True


def record_pathology(report: Any) -> dict[str, Any]:
    """Publish a diagnosis so the health surface and telemetry can see it.

    The report is kept as its own summary rather than as the object, because a
    health poll must not hold a reference to every finding's detail for the life
    of the process.
    """
    summary = report.as_json(limit=0) if hasattr(report, "as_json") else dict(report)
    trimmed = {
        "total": summary.get("total", 0),
        "confirmed": summary.get("confirmed", 0),
        "by_kind": summary.get("by_kind", {}),
        "by_confidence": summary.get("by_confidence", {}),
        "at": time.time(),
    }
    with _LOCK:
        _CACHE["pathology"] = trimmed
    publish_telemetry()
    return trimmed
