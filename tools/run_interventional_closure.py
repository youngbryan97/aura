#!/usr/bin/env python3
"""Whether anything outside K* moves K*, by swapping it in from another moment.

The second of closure's two tests (docs/ISC_V6_PREREGISTRATION.md, addenda of 25
October). On a seed's own anchors, every unit outside K* is swapped whole from
another anchor, and every attribute of a unit K* holds part of but does not hold
is swapped alone, each through a hybrid snapshot and `runtime.restore` and
nothing else. A donor's instants are taken from the receiving anchor first, so a
swap is of state and not of the time since something happened. After two turns,
K's recorded vector and every K* reading (`read_held`) are compared bit for bit
with an untouched roll made just before the swap. If that untouched roll has
drifted from the one made at the start, an earlier swap left something behind
that a restore did not undo: a fork defect, reported as residue.

Each swap comes from more than one donor, at distances that share no factor.
A unit that acts on a cycle, every sixth message or every twentieth tick, holds
a count; a donor whose count is in phase with the receiver's hands over the same
phase and nothing moves. The discourse tracker deepens its reading on every sixth
message, and seed 23's donors sat 48 turns away, six messages, in phase every
time: it never moved K there, and it moved K at half the anchors of seven fresh
seeds whose donors sat 64 turns away (26 October). Anchors 13 turns apart and
donors one anchor and half the bank away left a period of 13 hidden, and every
period 13 divides: the external audit of 27 October named it. Anchors now sit
13 and 14 turns apart in turn, so the one-anchor donors are 13 or 14 turns away
and the half-bank donors 54, and no period above one is in phase with all of
them (`periods_hidden`).

CLOSED: no swap moves K or any comparable K* reading at any anchor whose
untouched twins agree, with no residue. OPEN names what moves it and, for a
whole unit, which of its attributes. UNRESOLVED: residue, or no anchor decided.

    python tools/run_interventional_closure.py --anchors 8 --spacing 13,14 --seed 61 --out runs/interventional
"""
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

#: Two turns: the deciding horizon of the partition sweep and of the interchange that drew K*.
TURNS = 2
#: Rounds of every condition before the anchors, so the organism is past its first turns.
WARM_ROUNDS = 6


def aligned(donor: Any, receiver: Any, key: str = "") -> Any:
    """The donor's state with every instant taken from the receiver: state swapped, time since not."""
    from core.subject.closure import _named_clock
    from core.subject.monotonic_instants import MonotonicInstant
    from core.subject.snapshot import _EPOCH_FLOOR

    if isinstance(donor, dict) and isinstance(receiver, dict):
        return {k: aligned(v, receiver.get(k), str(k)) if k in receiver else v for k, v in donor.items()}
    if isinstance(donor, tuple) and isinstance(receiver, tuple) and len(donor) == len(receiver) == 2:
        return (donor[0], aligned(donor[1], receiver[1], key))
    is_instant = isinstance(donor, MonotonicInstant) or (
        isinstance(donor, (int, float)) and not isinstance(donor, bool) and donor > _EPOCH_FLOOR
    )
    if (is_instant or _named_clock(key) or "monotonic" in key.lower()) and receiver is not None:
        return receiver
    return donor


def donor_offsets(spec: str, anchors: int) -> list[int]:
    """Donor distances in anchors, from "1,half"-style text; zero and repeats dropped."""
    return sorted({
        (anchors // 2 if part.strip() == "half" else int(part)) % anchors
        for part in str(spec).split(",") if part.strip()
    } - {0})


def in_shard(index: int, shard: str) -> bool:
    """Whether the index-th swap belongs to shard "k/n"."""
    k, _, n = str(shard).partition("/")
    return index % max(1, int(n or 1)) == int(k or 0)


def periods_in_phase(offsets: list[int], spacing: int, longest: int) -> list[int]:
    """The cycles, in turns, that every donor's distance is a whole number of: a swap cannot show them."""
    return [p for p in range(2, longest + 1) if all((o * spacing) % p == 0 for o in offsets)]


def spacing_pattern(text: str | int) -> list[int]:
    """Turns between anchors, repeating: "13,14" from text, or one spacing."""
    return [max(1, int(part)) for part in str(text).split(",") if part.strip()]


def anchor_turns(pattern: list[int], anchors: int) -> list[int]:
    """The turn each anchor is taken at under a repeating spacing pattern."""
    out, at = [], 0
    for index in range(anchors):
        at += pattern[index % len(pattern)]
        out.append(at)
    return out


def donor_distances(turns: list[int], offsets: list[int]) -> list[int]:
    """Every distance in turns between an anchor and its donors, as the swaps use them."""
    n = len(turns)
    return sorted({abs(turns[(i + o) % n] - turns[i]) for i in range(n) for o in offsets} - {0})


def periods_hidden(distances: list[int], longest: int) -> list[int]:
    """The cycles, in turns, that every donor distance is a whole number of."""
    return [p for p in range(2, longest + 1) if distances and all(d % p == 0 for d in distances)]


def anchor_schedule(spec: str, anchors: int, spacing: str | int) -> list[int]:
    """An explicit prospective bank, or the declared repeating spacing pattern."""
    positions = [int(part) for part in spec.split(",")] if spec else anchor_turns(spacing_pattern(spacing), anchors)
    if (len(positions) != anchors or anchors < 2 or min(positions) < 1
            or positions != sorted(set(positions))):
        raise ValueError("anchor turns must be distinct, positive, increasing, and match --anchors >= 2")
    return positions


def verdict_of(moved: list[str], residue: list[dict[str, Any]], decided: bool) -> str:
    """OPEN when something moved K*, UNRESOLVED on residue or with no anchor decided, CLOSED otherwise."""
    if moved:
        return "OPEN"
    if residue or not decided:
        return "UNRESOLVED"
    return "CLOSED"


def hybrid_snapshot(receiver: Any, source: Any, name: str, key: str | None) -> Any:
    """Replace one declared outside unit while retaining the receiving snapshot's other state."""
    from core.subject.copies import identical

    if name.startswith("module:"):
        held_key = name.split(":", 1)[1]
        mine = (receiver.module_state or {}).get(held_key)
        donor = (source.module_state or {}).get(held_key)
        if not (isinstance(mine, tuple) and isinstance(donor, tuple) and len(mine) == len(donor) == 2):
            return None
        if key is None:
            given = aligned(donor[1], mine[1])
        elif isinstance(mine[1], dict) and isinstance(donor[1], dict) and key in mine[1] and key in donor[1]:
            given = {**mine[1], key: aligned(donor[1][key], mine[1][key], key)}
        else:
            return None
        if identical(given, mine[1]):
            return None
        return dataclasses.replace(receiver, module_state={**receiver.module_state, held_key: (mine[0], given)})
    if name.startswith("organ:"):
        organ = name.split(":", 1)[1]
        mine, donor = receiver.organs.get(organ), source.organs.get(organ)
        if not isinstance(mine, dict) or not isinstance(donor, dict) or key not in mine or key not in donor:
            return None
        given = aligned(donor[key], mine[key], key or "")
        if identical(given, mine[key]):
            return None
        return dataclasses.replace(receiver, organs={**receiver.organs, organ: {**mine, key: given}})
    field = "services" if name in receiver.services else "phases"
    mine, donor = getattr(receiver, field).get(name), getattr(source, field).get(name)
    if mine is None or donor is None:
        return None
    if key is None:
        given = aligned(donor, mine)
        if identical(given, mine):
            return None
        return dataclasses.replace(receiver, **{field: {**getattr(receiver, field), name: given}})
    if not isinstance(mine, dict) or not isinstance(donor, dict) or key not in mine or key not in donor:
        return None
    given = aligned(donor[key], mine[key], key)
    if identical(given, mine[key]):
        return None
    return dataclasses.replace(receiver, **{field: {**getattr(receiver, field), name: {**mine, key: given}}})

def joint_snapshot(receiver: Any, donor: Any, units: list[tuple[str, str | None]]) -> Any:
    """Swap every declared outside unit together, keeping all held state at the receiver.

    Individual swaps miss interactions: from outside state (0, 0), neither
    single change moves an AND gate, but a donor (1, 1) does. This joint arm
    uses the same donor across units, preserving its correlations.
    """
    hybrid = receiver
    changed = False
    for name, key in units:
        updated = hybrid_snapshot(hybrid, donor, name, key)
        if updated is not None:
            hybrid = updated
            changed = True
    return hybrid if changed else None


async def compare_swap(
    roll: Any, receiver: Any, swapped: Any, index: int, baseline: Any,
    comparable: set[str], *, label: str, one_offs: list[dict[str, Any]],
    observations: Any = None,
) -> tuple[bool, float, list[str], float]:
    """Compare paired rolls and repeat a movement, with optional full-array custody.

    Every untouched array control must match the anchor's baseline, including
    controls after previous interventions. Initial twins alone do not rule out
    later restore residue hidden by the core's numeric sketch.
    """
    import numpy as np

    control_x, control_h = await roll(receiver, index)
    if observations is not None:
        observations.require_repeatable_digests(baseline[1], control_h)
    drift = float(np.max(np.abs(control_x - baseline[0])))
    x, h = await roll(swapped, index)
    apart = float(np.max(np.abs(x - control_x)))
    changed = sorted(k for k in comparable if h.get(k) != control_h.get(k))
    if apart > 0.0 or changed or drift > 0.0:
        again_x, again_h = await roll(receiver, index)
        if observations is not None:
            observations.require_repeatable_digests(baseline[1], again_h)
        redrift = float(np.max(np.abs(again_x - baseline[0])))
        y, g = await roll(swapped, index)
        reapart = float(np.max(np.abs(y - again_x)))
        rechanged = sorted(k for k in comparable if g.get(k) != again_h.get(k))
        if (apart > 0.0 or changed) and not (reapart > 0.0 or rechanged):
            one_offs.append({"swap": label, "anchor": index, "apart": apart, "kind": "a swap that did not come back"})
            apart, changed = 0.0, []
        if drift > 0.0 and redrift == 0.0:
            one_offs.append({"swap": label, "anchor": index, "drift": drift, "kind": "an untouched roll that did not come back"})
            drift = 0.0
    return apart > 0.0 or bool(changed), apart, changed, drift


async def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--anchors", type=int, default=8)
    parser.add_argument("--turns", type=int, default=TURNS,
                        help="prospectively declared turns per roll; the historical horizon is two")
    parser.add_argument("--coalition-swaps", type=int, default=0,
                        help="prospective additional subset swaps (0 to 1024); recorded separately from single and whole swaps")
    parser.add_argument(
        "--observation-plan", type=Path,
        help="prospective exact-array JSON plan; also compare complete held-array content digests",
    )
    parser.add_argument(
        "--spacing", default="13,14",
        help="turns between anchors, a repeating comma-separated pattern",
    )
    parser.add_argument("--anchor-turns", default="",
                        help="explicit increasing turn indices for a prospective nonuniform donor bank")
    parser.add_argument(
        "--shard", default="0/1",
        help="k/n: swap only every n-th unit from the k-th, so one seed's test can run in n processes",
    )
    parser.add_argument(
        "--donors", default="1,half",
        help="donor distances in anchors, comma-separated; 'half' is half the bank",
    )
    parser.add_argument(
        "--joint-periphery", action="store_true",
        help="also swap every outside unit together; record this prospective diagnostic in the result",
    )
    parser.add_argument(
        "--units", default="",
        help="swap alone only the units whose name matches this pattern (validation runs); the joint row is whole",
    )
    parser.add_argument(
        "--planted", default="",
        help="validation only: plant these known-answer units outside K* (cause,gate,record; core/subject/planted.py)",
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.turns < 1:
        parser.error("turns must be positive")
    if not 0 <= args.coalition_swaps <= 1024:
        parser.error("coalition swaps must be between 0 and 1024")
    positions = anchor_schedule(args.anchor_turns, args.anchors, args.spacing)

    import numpy as np

    from core.subject.carrier_reads import HELD, ORGANS_WHOLE, held_attributes_of, held_whole, read_held
    from core.subject.closure import _core_attributes
    from core.subject.driver import CONDITIONS, build_runtime, calibrate_clock, quiesce_organism, start_organism
    from core.subject.isolation import isolate_state
    from core.subject.recording import build_recording
    from core.subject.v25_runtime import _run_turns, collect_anchor_bank

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("AURA_LOG_DIR", str(out / "logs"))
    random.seed(args.seed)
    np.random.seed(args.seed % (2**32))
    isolate_state(out)
    from core.subject.provenance import evidence_source
    source = evidence_source(args.seed)
    observations = None
    observation_declaration = None
    if args.observation_plan:
        from core.subject.declared_observations import ObservationPlan

        observation_declaration = json.loads(args.observation_plan.read_text())
        observations = ObservationPlan(observation_declaration)
        observations.digest_budget()
    runtime = build_runtime(out, seed=args.seed)
    await start_organism(runtime)
    if args.planted:
        from core.subject.planted import install

        install(tuple(kind.strip() for kind in args.planted.split(",") if kind.strip()))
        print(f"planted outside K*: {args.planted}", flush=True)
    await quiesce_organism(runtime)
    await calibrate_clock(runtime, CONDITIONS)
    for _ in range(WARM_ROUNDS):
        for condition in CONDITIONS:
            await runtime.turn_once(condition)
    await quiesce_organism(runtime)
    runtime.freeze_host()
    runtime.forked_services = None
    runtime.forked_phases = None
    runtime.forked_modules = None
    pattern = spacing_pattern(args.spacing)
    rounds = max(1, math.ceil(positions[-1] / len(CONDITIONS)))
    bank = await collect_anchor_bank(
        runtime, CONDITIONS, rounds=rounds, history_turns=8, take_at=frozenset(positions),
    )
    snaps = [anchor.snapshot for anchor in list(bank)[: args.anchors]]
    if len(snaps) != len(positions):
        raise RuntimeError("the requested donor bank could not be collected")
    offsets = donor_offsets(args.donors, len(snaps))
    distances = donor_distances(positions, offsets)
    hidden = periods_hidden(distances, max(distances, default=0))
    print(f"donors at {offsets} anchors, {distances} turns away; periods in phase with every donor: {hidden}", flush=True)
    held_units = {entry.unit for entry in HELD}

    async def roll(snapshot: Any, index: int) -> tuple[Any, dict[str, Any]]:
        runtime.restore(snapshot)
        frames = await _run_turns(runtime, CONDITIONS[index % len(CONDITIONS)], args.turns)
        held_readings: dict[str, Any] = read_held(runtime.kernel, organs=runtime.organs)
        if observations is not None:
            held_readings.update(observations.digests(runtime.kernel, runtime.organs))
        return build_recording(frames).x, held_readings

    base: list[Any] = []
    comparable: list[set[str]] = []
    for a, snap in enumerate(snaps):
        x1, h1 = await roll(snap, a)
        x2, h2 = await roll(snap, a)
        if observations is not None:
            observations.require_repeatable_digests(h1, h2)
        if float(np.max(np.abs(x1 - x2))) > 0.0:
            print(f"anchor {a}: untouched twins differ in K; it decides nothing", flush=True)
            base.append(None)
            comparable.append(set())
            continue
        base.append((x1, h1))
        comparable.append({k for k in h1 if h2.get(k) == h1[k]})
    print(f"{len(snaps)} anchors; K* readings comparable per anchor: {[len(c) for c in comparable]}", flush=True)

    read_by_k = _core_attributes()

    def swaps():
        if not ORGANS_WHOLE:
            for organ in sorted(snaps[0].organs):
                entry = snaps[0].organs.get(organ)
                for key in sorted(entry) if isinstance(entry, dict) else ():
                    if key in read_by_k.get(organ, set()) or key.lstrip("_") in read_by_k.get(organ, set()):
                        continue
                    if key not in held_attributes_of(f"organ:{organ}"):
                        yield f"organ:{organ}", key
        # What her modules keep is a unit too: the substate bus, the cognitive
        # event graph, the ontogeny core. A test that swapped only services and
        # phases could never find one of them moving K.
        for key in sorted(set().union(*((s.module_state or {}).keys() for s in snaps))):
            name = f"module:{key}"
            if name not in held_units:
                yield name, None
                continue
            if held_whole(name):
                continue
            entry = (snaps[0].module_state or {}).get(key)
            captured = entry[1] if isinstance(entry, tuple) and len(entry) == 2 else None
            for attribute in sorted(captured) if isinstance(captured, dict) else ():
                if attribute not in held_attributes_of(name):
                    yield name, attribute
        names = sorted(set().union(*(s.services.keys() for s in snaps)) | set().union(*(s.phases.keys() for s in snaps)))
        for name in names:
            if name not in held_units:
                yield name, None
                continue
            if held_whole(name):
                continue
            entry = snaps[0].services.get(name) if name in snaps[0].services else snaps[0].phases.get(name)
            for key in sorted(entry) if isinstance(entry, dict) else ():
                if key not in held_attributes_of(name):
                    yield name, key

    def hybrid(a: int, b: int, name: str, key: str | None) -> Any:
        if name == "joint-periphery":
            return joint_snapshot(snaps[a], snaps[b], units)
        if name in coalition_units:
            return joint_snapshot(snaps[a], snaps[b], coalition_units[name])
        return hybrid_snapshot(snaps[a], snaps[b], name, key)

    one_offs: list[dict[str, Any]] = []

    async def moves(swapped: Any, a: int, label: str = "") -> tuple[bool, float, list[str], float]:
        return await compare_swap(
            roll, snaps[a], swapped, a, base[a], comparable[a],
            label=label, one_offs=one_offs, observations=observations,
        )

    async def write(payload: dict[str, Any]) -> None:
        from core.subject.latch import refuse_if_latched
        from core.runtime.atomic_writer import async_atomic_write_text

        refuse_if_latched("the interventional closure results")
        await async_atomic_write_text(out / "interventional_closure.json", json.dumps(
            {**payload, "evidence_source": source, "joint_periphery": args.joint_periphery,
             "turns_per_roll": args.turns,
             "coalition_plan": coalition_plan.description(),
             "anchor_turns": positions, "donor_turn_distances": distances,
             "declared_array_plan": observation_declaration,
             "declared_array_observations": observations.description() if observations else None}, indent=1))

    results: dict[str, dict[str, Any]] = {}
    moved: list[str] = []
    residue: list[dict[str, Any]] = []
    previous = None
    started = time.monotonic()
    whole: set[str] = set()
    units = list(swaps())
    from core.subject.coalitions import plan_coalitions

    coalition_plan = plan_coalitions(units, budget=args.coalition_swaps, seed=args.seed)
    coalition_units = {f"coalition:{i:04d}": [units[j] for j in indices]
                       for i, indices in enumerate(coalition_plan.subsets)}
    measured_units = (units + ([("joint-periphery", None)] if args.joint_periphery else [])
                      + [(label, None) for label in coalition_units])
    for index, (name, key) in enumerate(measured_units):
        if not in_shard(index, args.shard):
            continue
        label = name if key is None else f"{name}.{key}"
        if args.units and name != "joint-periphery" and name not in coalition_units and not re.search(args.units, label):
            continue
        if key is None and name != "joint-periphery" and name not in coalition_units:
            whole.add(label)
        row: dict[str, Any] = {"tested": 0, "moved_k": 0, "moved_held": 0, "max_k": 0.0, "held_keys": []}
        for a in range(len(snaps)):
            if base[a] is None:
                continue
            for offset in offsets:
                swapped = hybrid(a, (a + offset) % len(snaps), name, key)
                if swapped is None:
                    continue
                row["tested"] += 1
                _hit, apart, changed, drift = await moves(swapped, a, label)
                if drift > 0.0:
                    residue.append({"after": previous, "anchor": a, "drift": drift})
                    print(f"  RESIDUE: an untouched roll at anchor {a} moved {drift:.3g} after {previous}", flush=True)
                if apart > 0.0:
                    row["moved_k"] += 1
                    row["max_k"] = max(row["max_k"], apart)
                if changed:
                    row["moved_held"] += 1
                    row["held_keys"] = sorted(set(row["held_keys"]) | set(changed[:5]))
        results[label] = row
        previous = label
        if row["moved_k"] or row["moved_held"]:
            moved.append(label)
            print(f"[{time.monotonic() - started:.0f}s] {label} MOVES K* (K {row['moved_k']}/{row['tested']}, max "
                  f"{row['max_k']:.3g}; held {row['moved_held']}/{row['tested']} {row['held_keys'][:3]})", flush=True)
        await write({"seed": args.seed, "anchors": len(snaps), "results": results, "moves_k_star": moved, "residue": residue,
               "one_offs": one_offs})

    culprits: dict[str, list[str]] = {}
    for label in [m for m in moved if m in whole]:
        if label.startswith("module:"):
            held = (snaps[0].module_state or {}).get(label.split(":", 1)[1])
            entry = held[1] if isinstance(held, tuple) and len(held) == 2 else None
        else:
            field = "services" if label in snaps[0].services else "phases"
            entry = getattr(snaps[0], field).get(label)
        found = []
        for key in sorted(entry) if isinstance(entry, dict) else ():
            hits = 0
            for a in range(len(snaps)):
                if base[a] is None:
                    continue
                for offset in offsets:
                    swapped = hybrid(a, (a + offset) % len(snaps), label, key)
                    if swapped is not None and (await moves(swapped, a, f"{label}.{key}"))[0]:
                        hits += 1
            if hits:
                found.append(key)
        culprits[label] = found
        print(f"  {label}: the attributes that move K*: {found}", flush=True)

    verdict = verdict_of(moved, residue, any(b is not None for b in base))
    print(f"\ninterventional closure of K*: {verdict}; {len(results)} swaps; moved by: {moved}", flush=True)
    # A unit that is the same in every anchor and its donors is never varied by
    # a swap, so the test says nothing about it (167 of 960 varied on seed 73).
    singles = {label: row for label, row in results.items()
               if label != "joint-periphery" and label not in coalition_units}
    coverage = {"enumerated": len(singles), "varied": sum(1 for row in singles.values() if row["tested"])}
    print(f"coverage: {coverage['varied']} of {coverage['enumerated']} units outside K* differed between an anchor "
          "and a donor", flush=True)
    await write({"seed": args.seed, "anchors": len(snaps), "spacing": pattern, "donor_offsets": offsets, "donor_distances": distances, "periods_hidden": hidden, "shard": args.shard,
           "verdict": verdict, "results": results, "moves_k_star": moved,
           "residue": residue, "one_offs": one_offs, "culprit_attributes": culprits, "coverage": coverage})
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
