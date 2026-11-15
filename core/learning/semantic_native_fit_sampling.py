"""Bound native pilot updates while retaining fit-only contrast donors."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any

from core.learning.recurrent_sft_sampling import sample_history
from core.verify.invariants import invariant

BOUNDED_NATIVE_SAMPLER = "construction_depth_balanced_v1"


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False).encode("ascii")).hexdigest()


def bounded_native_fit_schedule(
    examples: Any,
    fit_ids: tuple[Any, ...],
    *,
    steps: int,
    seed: int,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Sample fit construction/depth strata with the existing SFT sampler."""
    fit_ids = tuple(fit_ids)
    requested = set(fit_ids)
    fitting = [item for item in examples if item.ir.source_text_sha256 in requested]
    by_id = {item.ir.source_text_sha256: item for item in fitting}
    if (not requested or len(requested) != len(fit_ids) or len(by_id) != len(fitting)
            or set(by_id) != requested or any(item.split != "train" for item in fitting)
            or type(steps) is not int or steps < 1 or type(seed) is not int or seed < 0):
        raise ValueError("bounded native schedule needs the unique complete fit-only partition")
    rows = []
    for identity, item in sorted(by_id.items()):
        depth = len(item.ir.instructions)
        if not item.construction_id or not depth:
            raise ValueError("bounded native schedule needs construction and program depth")
        rows.append({"example_id": identity,
            "family": json.dumps([item.construction_id, depth], separators=(",", ":"))})
    strata = {row["family"] for row in rows}
    if steps < len(strata):
        raise ValueError("bounded native schedule cannot visit every construction/depth stratum")
    history = sample_history(rows, seed=seed, steps=steps)
    indices = history["indices"]
    schedule = tuple(rows[index]["example_id"] for index in indices)
    exposures = Counter(rows[index]["family"] for index in indices)
    if set(exposures) != strata or max(exposures.values()) - min(exposures.values()) > 1:
        raise ValueError("bounded native schedule lost balanced stratum coverage")
    body = {"schema": "aura.native_bounded_fit_schedule.v1",
        "policy": BOUNDED_NATIVE_SAMPLER, "sampler": history["sampler"], "seed": seed,
        "eligible_population_sha256": _digest(rows), "schedule_sha256": _digest(schedule),
        "eligible_sources": len(rows), "primary_sources": len(set(schedule)),
        "primary_updates": len(schedule), "strata": len(strata),
        "stratum_exposures": dict(sorted(exposures.items())), "all_strata_visited": True,
        "complete_primary_epoch": set(schedule) == requested,
        "selection_basis": "fit_source_identity_construction_and_annotated_depth",
        "held_labels_used": False, "qualification_evidence": False}
    return schedule, {**body, "receipt_sha256": _digest(body)}


def native_depth_calibration_subset(
    examples: Any,
    identities: tuple[Any, ...],
    *,
    per_stratum: int,
) -> tuple[Any, ...]:
    """Keep a source-identity sample from every calibration construction/depth."""
    identities = tuple(identities)
    requested = set(identities)
    eligible = [item for item in examples if item.ir.source_text_sha256 in requested]
    by_id = {item.ir.source_text_sha256: item for item in eligible}
    if (not requested or len(requested) != len(identities)
            or len(by_id) != len(eligible) or set(by_id) != requested
            or type(per_stratum) is not int or per_stratum < 1):
        raise ValueError("native depth calibration needs unique eligible identities and a positive quota")
    groups = {}
    for identity in sorted(requested):
        item = by_id[identity]
        key = (item.construction_id, len(item.ir.instructions))
        if not key[0] or not key[1]:
            raise ValueError("native depth calibration needs construction and program depth")
        groups.setdefault(key, []).append(identity)
    return tuple(sorted(identity for group in groups.values() for identity in group[:per_stratum]))


def scheduled_native_source_pairs(
    pairs: dict[str, Any],
    schedule: tuple[Any, ...],
    fit_ids: tuple[Any, ...],
) -> dict[str, Any]:
    """Keep primary interactions and their donors inside the admitted fit set."""
    fitting, scheduled = set(fit_ids), set(schedule)
    if not scheduled or not scheduled <= fitting or not set(pairs) <= fitting:
        raise ValueError("native scheduled source pairs cross the fit partition")
    selected = {identity: rows for identity, rows in pairs.items() if identity in scheduled}
    for rows in selected.values():
        for row in rows if isinstance(rows, list) else [rows]:
            if row["partner"] not in fitting:
                raise ValueError("native scheduled source pair donor is outside the fit partition")
    if not selected:
        raise ValueError("native schedule has no witnessed source pairs")
    return selected


@invariant("learning.native_pilot_donors_stay_in_fit", scope="learning",
           owner="core/learning/semantic_native_fit_sampling.py", observational=False)
def _pilot_donor_scope() -> tuple:
    selected = scheduled_native_source_pairs(
        {"primary": [{"partner": "donor"}], "donor": [{"partner": "primary"}]},
        ("primary",), ("primary", "donor"))
    assert set(selected) == {"primary"}
    try:
        scheduled_native_source_pairs({"primary": [{"partner": "held"}]},
            ("primary",), ("primary", "donor"))
    except ValueError:
        # Clean is nothing yielded: a dict returned here was read as its keys, each one a breach (Q09, 15 November 2026).
        return ()  # not a failure: the held donor was refused and the fit donor kept, which is the scope held
    raise AssertionError("native pilot admitted a held source donor")
