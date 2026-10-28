"""The conjunction Bryan asked to finish: ISC-v6 (or v7, reported apart), J*, and bridge parity."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from core.subject.battery import Verdict
from core.subject.bridge import JStar


def compatibility(reports: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """Require a common clean revision and boundary, with paired seeds where needed."""
    blockers: list[str] = []
    pins: dict[str, Mapping[str, Any]] = {}
    for name, report in reports.items():
        pin = report.get("evidence_source") or (report.get("campaign") or {}).get("evidence_source") or {}
        pins[name] = pin
        if not pin.get("commit") or not pin.get("boundary_sha256"):
            blockers.append(f"{name}: source revision or carrier boundary was not pinned")
        if pin.get("dirty") is not False:
            blockers.append(f"{name}: source was dirty or cleanliness was not recorded")
        if report.get("analysis_source") is not None and report["analysis_source"] != pin:
            blockers.append(f"{name}: samples and analysis came from different source or boundary")
    for key in ("commit", "boundary_sha256"):
        values = {pin.get(key) for pin in pins.values() if pin.get(key)}
        if len(values) > 1:
            blockers.append(f"the evidence disagrees on {key}")
    # These are arms of the same seed's claim. Report grounding can be a
    # separate experiment; its seed is preserved in its own source record.
    seeds = [pins.get(name, {}).get("seed") for name in ("campaign", "carrier", "content", "interventional")]
    if any(type(seed) is not int for seed in seeds) or len(set(seeds)) != 1:
        blockers.append("campaign, carrier, content and closure need the same recorded seed")
    return blockers


def completion(verdict: Verdict, j: JStar, *, blockers: list[str], scope: str, version: str = "v6") -> dict[str, Any]:
    """Keep each goal visible; completion requires every goal and compatible evidence.

    `version` is the battery the goal is read under: v6, or v7, whose closure
    line is decided by intervention (addendum of 27 October, evening). Under
    v7 the v6 reading is reported beside it and never combined with it.
    """
    if version not in ("v6", "v7"):
        raise ValueError(f"no battery version {version!r}")
    lines = verdict.v7_lines() if version == "v7" else verdict.v6_lines()
    keys = [line.key for line in lines]
    on_this_seed = verdict.isc_v7_on_this_seed if version == "v7" else verdict.isc_v6_on_this_seed
    battery = len(lines) == 24 and len(set(keys)) == 24 and on_this_seed
    beside = {}
    if version == "v7":
        v6 = verdict.v6_lines()
        beside = {"isc_v6_on_this_seed": verdict.isc_v6_on_this_seed, "v6_passed": sum(line.passed for line in v6)}
    goals = {
        f"isc_{version}_24_of_24": battery,
        "j_star": j.status in {"DETERMINED_UP_TO_GAUGE", "DETERMINED_UP_TO_SYMMETRY"},
        "bridge_at_parity": j.bridge == "AT_PARITY",
    }
    return {
        "complete": all(goals.values()) and not blockers,
        "goals": goals,
        "scope": scope,
        "passed": sum(line.passed for line in lines),
        "total": len(lines),
        "failing_criteria": [line.key for line in lines if not line.passed],
        "evidence_blockers": list(blockers),
        "battery_version": version,
        **beside,
        "interpretation": "J* and parity are conditional on P1-P6; this is not a proof of those postulates.",
    }
