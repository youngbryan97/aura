#!/usr/bin/env python3
"""The content half of the bridge: is her geometry hers, or is it the recording's?

The carrier run asks whether there is one intrinsic process. If there is, the
next question the bridge asks is what its content is organised like. The
strongest available answer is structural: content, if the bridge holds, is the
position each thing occupies in the whole web of relations, and nothing else.

Nothing here reaches phenomenal character, and the report says so in the same
breath every time. What is testable is whether the relational structure is
really her organisation. Two of her mechanisms are asked the same question
about the same percepts and their answers are compared.

    d_Q   Fisher-Rao distance between the future-state laws two percept
          classes induce from a common fork.

    d_B   one minus the overlap of which memories each percept brought back.

They share the percepts and nothing else: the state vector records how many
memories returned and how well the best matched, never which ones.

Then the test that can fail. Affect is displaced, which moves the internal
geometry, and the run asks whether the behavioural geometry moved the same way.
A structure that merely describes the content has no reason to move with it. A
sham arm, displacing nothing, gives the floor that finite sampling alone
produces. The displacement moves the point her feelings are measured from,
towards feeling good, by the span her feelings cover in her ordinary life, so
it lasts through the turn and leaves the feelings free to answer the percept,
and raises how good she feels now by the span her valence covers, which is
what this turn's recall reads.

    python tools/run_subject_core_content.py --quick
    python tools/run_subject_core_content.py --anchors 24 --rounds 24
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# Before any core import: the profile decides where state lives, and a module
# that resolved its home under the live root keeps that home.
os.environ.setdefault("AURA_TESTING", "1")

#: The domain the displacement moves. The grid is built from the table that
#: says which emotions a percept may move, so affect is where the content
#: geometry is predicted to live. Declared before the run, not chosen after it.
DISPLACED_DOMAIN: str = "A"
#: The internal geometry compares classes on the presentations both have and
#: cross-fits by anchor (core.subject.content_runtime.internal_geometry).
PAIRED: bool = True

#: How strongly the two geometries must agree before the run says they are one
#: structure. Spearman rank correlation, which is the weakest assumption that
#: still has an answer: the structural claim is about order, not about units.
AGREEMENT_BAR: float = 0.30


def _log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def _as_json(mapping: dict[tuple[int, int], float]) -> dict[str, float]:
    return {f"{i}-{j}": round(float(v), 6) for (i, j), v in sorted(mapping.items())}


#: The level a geometry has to stand out of its own noise at, the level its
#: agreement is held to.
NOISE_LEVEL: float = 0.01


def _out_of_noise(between: list[float], floors: list[float]) -> tuple[bool, float]:
    """Whether the between-class distances exceed the same-class floors, by a one-sided rank test."""
    from scipy.stats import mannwhitneyu

    if len(between) < 2 or len(floors) < 2:
        return False, 1.0
    p_value = float(mannwhitneyu(between, floors, alternative="greater").pvalue)
    return p_value < NOISE_LEVEL, p_value


def _save_class_samples(
    run_dir: Path, classes: Sequence[Any], stages: dict[str, dict[str, Any]], name: str = "class_samples.npz"
) -> Path:
    """Every stage's per-class futures, retrieval sets and anchor keys, in one file."""
    import numpy as np  # after the profile is set, like every other import here

    arrays: dict[str, Any] = {}
    for stage, samples in stages.items():
        for index, cls in enumerate(classes):
            rows = samples.get(cls.name)
            if rows is None:
                continue
            tag = f"{stage}__{index:02d}"
            if rows.futures:
                arrays[f"{tag}__futures"] = np.vstack(rows.futures)
            if rows.futures_b:
                arrays[f"{tag}__futures_b"] = np.vstack(rows.futures_b)
            if rows.keys:
                arrays[f"{tag}__keys"] = np.asarray(rows.keys, dtype=np.int64)
            # The anchor each row forked from, which the internal geometry is
            # read with: without it a reading made from these arrays later
            # could not reproduce the run's own (seed 23 read 0.356 without it
            # and 0.284 with it, 26 October).
            if rows.context:
                arrays[f"{tag}__context"] = np.vstack(rows.context)
            arrays[f"{tag}__retrieved"] = np.asarray(
                [json.dumps(sorted(recalled)) for recalled in rows.retrieved], dtype=object
            )
            # The second arm's recall, which the behavioural floor reads; a
            # samples file without it could not give a run's floor back.
            arrays[f"{tag}__retrieved_b"] = np.asarray(
                [json.dumps(sorted(recalled)) for recalled in rows.retrieved_b], dtype=object
            )
    arrays["class_names"] = np.asarray([cls.name for cls in classes], dtype=object)
    target = run_dir / name
    from core.subject.latch import refuse_if_latched

    refuse_if_latched("the content samples")
    np.savez_compressed(target, **arrays)
    return target


def _load_class_samples(path: Path, classes: Sequence[Any]) -> dict[str, dict[str, Any]]:
    """A samples file read back into each stage's per-class samples."""
    import numpy as np

    from core.subject.content_runtime import ClassSamples

    data = np.load(path, allow_pickle=True)
    stages: dict[str, dict[str, Any]] = {}
    for stage in sorted({key.split("__", 1)[0] for key in data.files if "__" in key}):
        out: dict[str, Any] = {}
        for index, cls in enumerate(classes):
            tag = f"{stage}__{index:02d}"
            if f"{tag}__keys" not in data.files:
                continue

            def rows(field: str) -> list[Any]:
                return list(data[f"{tag}__{field}"]) if f"{tag}__{field}" in data.files else []

            def recalled(field: str) -> list[frozenset[str]]:
                return [frozenset(json.loads(item)) for item in rows(field)]

            out[cls.name] = ClassSamples(
                name=cls.name, futures=rows("futures"), retrieved=recalled("retrieved"),
                context=rows("context"), futures_b=rows("futures_b"), retrieved_b=recalled("retrieved_b"),
                keys=[tuple(int(v) for v in key) for key in data[f"{tag}__keys"]],
            )
        stages[stage] = out
    return stages


def _merge_samples(parts: Sequence[dict[str, Any]], anchors: int) -> dict[str, Any]:
    """Each class's rows from every shard, in the order one process would have taken them.

    Refuses rather than merges when an anchor is missing or came from two
    shards: a geometry read over a partial or doubled set of anchors is a
    different measurement from the one asked for.
    """
    from core.subject.content_runtime import ClassSamples

    names = sorted({name for part in parts for name in part})
    merged: dict[str, Any] = {}
    for name in names:
        rows = []
        for part in parts:
            got = part.get(name)
            if got is None:
                continue
            for i, key in enumerate(got.keys):
                rows.append((key, i, got))
        seen = sorted({key for key, _i, _g in rows})
        if len(seen) != len(rows):
            raise SystemExit(f"refusing to merge: class {name} has an anchor and condition from two shards")
        if sorted({key[0] for key in seen}) != list(range(anchors)):
            raise SystemExit(f"refusing to merge: class {name} is missing anchors")
        rows.sort(key=lambda row: row[0])

        def pick(field: str) -> list[Any]:
            out = []
            for _key, i, got in rows:
                values = getattr(got, field)
                if values:
                    out.append(values[i])
            return out

        merged[name] = ClassSamples(
            name=name, futures=pick("futures"), retrieved=pick("retrieved"), context=pick("context"),
            futures_b=pick("futures_b"), retrieved_b=pick("retrieved_b"), keys=[row[0] for row in rows],
        )
    return merged


def _shard_of(spec: str, count: int) -> list[int]:
    """The anchors shard `k/n` presents: every n-th from the k-th."""
    k, n = (int(part) for part in str(spec).split("/", 1))
    if not 0 <= k < n:
        raise SystemExit(f"refusing: shard {spec} is not k/n with 0 <= k < n")
    return list(range(k, count, n))


def _globally_keyed(samples: dict[str, Any], indices: Sequence[int]) -> dict[str, Any]:
    """A shard's samples with each key naming its anchor's place in the whole bank."""
    for rows in samples.values():
        rows.keys = [(int(indices[anchor]), int(condition)) for anchor, condition in rows.keys]
    return samples


def _analyse(
    evidence: dict[str, Any], classes: Sequence[Any], stages: dict[str, dict[str, Any]], seed: int
) -> None:
    """Both geometries, their agreement and whether they move together, from the samples alone."""
    from core.subject.content import agreement, design_recovery, moves_together
    from core.subject.content_runtime import behavioural_floor, behavioural_geometry
    from core.subject.content_runtime import repeated_internal_geometry as internal_geometry

    base, moved, sham = stages["base"], stages["moved"], stages["sham"]
    internal, floor = internal_geometry(base, classes, seed=seed, paired=PAIRED)
    behavioural, coverage = behavioural_geometry(base, classes)
    evidence["internal"] = _as_json(internal)
    evidence["internal_floor"] = {f"{i}-{j}": round(v, 6) for (i, j), v in floor.items()}
    evidence["behavioural"] = _as_json(behavioural)
    evidence["behavioural_coverage"] = {k: round(v, 4) for k, v in coverage.items()}
    evidence["behavioural_floor"] = {k: round(v, 6) for k, v in behavioural_floor(base, classes).items()}
    _log(
        f"  internal pairs {len(internal)}, behavioural pairs {len(behavioural)}, "
        f"classes that recalled nothing {sum(1 for v in coverage.values() if v <= 0.0)}"
    )
    result = agreement(internal, behavioural, size=len(classes), seed=seed)
    evidence["agreement"] = result.__dict__ | {"bar": AGREEMENT_BAR}
    _log(f"  agreement rho={result.rho} p={result.p_value} (spread {result.internal_spread} / {result.behavioural_spread})")
    recovery = design_recovery(internal, classes, seed=seed)
    evidence["design_recovery"] = recovery.__dict__
    _log(f"  design recovery rho={recovery.rho} p={recovery.p_value}")
    moved_internal, _ = internal_geometry(moved, classes, seed=seed, paired=PAIRED)
    moved_behavioural, _ = behavioural_geometry(moved, classes)
    # The same folds as the two geometries it is compared against. With
    # its own seed the sham's changes carried fold-assignment noise the
    # displaced changes did not, which widened the floor it sets.
    sham_internal, _ = internal_geometry(sham, classes, seed=seed, paired=PAIRED)
    sham_behavioural, _ = behavioural_geometry(sham, classes)
    tracked = moves_together(
        internal, moved_internal, behavioural, moved_behavioural,
        size=len(classes), sham_internal_after=sham_internal, sham_behavioural_after=sham_behavioural, seed=seed,
    )
    evidence["moves_together"] = tracked.__dict__ | {"bar": AGREEMENT_BAR}
    # A reading at parity, beside the verdict and deciding nothing (ISC-v6
    # addendum, 26 October, late morning): a person's account is that the
    # relations among memories stay the same under a mood shift while the
    # perspective on them changes. This is how far hers stayed the same.
    evidence["relations_kept"] = _relations_kept(behavioural, moved_behavioural)
    _log(f"  moves together rho={tracked.rho} p={tracked.p_value} floor={tracked.floor_rho}")


def _close(evidence: dict[str, Any]) -> None:
    """The verdict and what it does and does not say."""
    evidence["authority"] = _authority(evidence)
    evidence["verdict"] = _verdict(evidence)
    evidence["bridge_status"] = {
        "content_structure": evidence["verdict"],
        "phenomenal_bridge": "JUDGED_BY_SOLVE_FOR_J",
        "note": (
            "Two of her mechanisms agreeing about one geometry is evidence that "
            "the geometry is her organisation rather than the recording's. It is "
            "the structure ground of the bridge, judged at parity with the other "
            "four (docs/BRIDGE_PARITY.md). That the organisation is felt is the "
            "structural-identity postulate, assumed for her as for a person."
        ),
    }


def _authority(evidence: dict[str, Any]) -> dict[str, Any]:
    """What would make this report not a result.

    Missing evidence is NOT_MEASURED. A run that could not read one of its two
    geometries has not compared them, and saying so is the whole point of
    having the gate.
    """
    blockers: list[str] = []
    agreement = evidence.get("agreement", {})
    moves = evidence.get("moves_together", {})
    coverage = evidence.get("behavioural_coverage", {})
    classes = evidence.get("classes", [])
    floor = evidence.get("internal_floor", {})
    internal = evidence.get("internal", {})

    if len(classes) < 4:
        blockers.append("fewer than four percept classes were presented")
    if not agreement.get("measured", False):
        blockers.append(f"the geometries were not comparable: {agreement.get('why', '?')}")
    if not moves.get("measured", False):
        blockers.append(f"the displacement test did not read: {moves.get('why', '?')}")
    silent = [name for name, share in coverage.items() if share <= 0.0]
    if silent:
        blockers.append(
            f"{len(silent)} classes brought back nothing at all, so the behavioural "
            f"geometry has no reading for them: {silent[:4]}"
        )
    # Whether each geometry stands out of its own noise: the distances between
    # classes, as a set, above the distances of classes from themselves. This
    # used to ask that the closest of every pair clear the largest of every
    # floor, the minimum of 190 noisy numbers against the maximum of 20, which
    # grows harder the more classes are presented and refuses a structure in
    # which any two classes are alike. Classes the structure cannot tell apart
    # are what the bridge's gauge is for: they share a cell
    # (core/subject/bridge.py `orbits`, Theorems 5 and 6), and they do not make
    # the rest of the structure unmeasured.
    b_floor = evidence.get("behavioural_floor", {})
    behavioural = evidence.get("behavioural", {})
    if behavioural and b_floor:
        above, p_value = _out_of_noise(list(behavioural.values()), list(b_floor.values()))
        if not above:
            blockers.append(
                "the classes recall no more differently from each other than each "
                f"recalls from itself (p={p_value:.3g}), so retrieval does not separate the percepts"
            )
    if internal and floor:
        above, p_value = _out_of_noise(list(internal.values()), list(floor.values()))
        if not above:
            blockers.append(
                "the classes lie no further apart than each lies from itself "
                f"(p={p_value:.3g}), so the internal geometry is inside its own noise"
            )
    return {
        "authoritative": not blockers,
        "blockers": blockers,
        "missing_evidence_is": "NOT_MEASURED, never PASS",
    }


def _verdict(evidence: dict[str, Any]) -> str:
    if evidence.get("authority", {}).get("blockers"):
        return "NOT_MEASURED"
    agreement = evidence.get("agreement", {})
    moves = evidence.get("moves_together", {})
    agrees = (
        agreement.get("measured")
        and float(agreement.get("rho", 0.0)) >= AGREEMENT_BAR
        and float(agreement.get("p_value", 1.0)) < 0.01
    )
    tracks = (
        moves.get("measured")
        and float(moves.get("rho", 0.0)) >= AGREEMENT_BAR
        and float(moves.get("p_value", 1.0)) < 0.01
        and float(moves.get("rho", 0.0)) > float(moves.get("floor_rho", 0.0))
    )
    if agrees and tracks:
        return "ONE_STRUCTURE"
    # Agreeing and not tracking is a finding only when tracking was measured.
    # A displacement that moved the manifold no further than a second look is
    # no test of whether the behaviour follows it.
    if agrees and not moves.get("measured"):
        return "NOT_MEASURED"
    if agrees:
        return "AGREES_BUT_DOES_NOT_TRACK"
    return "SEPARATE_STRUCTURES"


def _relations_kept(before: dict, after: dict) -> dict:
    """Rank correlation of her recall geometry before and after the displacement, over the pairs in both."""
    import numpy as np

    keys = sorted(set(before) & set(after))
    if len(keys) < 3:
        return {"measured": False, "pairs": len(keys)}

    def ranks(values: list[float]) -> np.ndarray:
        order = np.argsort(values, kind="mergesort")
        out = np.empty(len(values))
        out[order] = np.arange(len(values), dtype=float)
        return out

    a = ranks([float(before[k]) for k in keys])
    b = ranks([float(after[k]) for k in keys])
    if a.std() == 0.0 or b.std() == 0.0:
        return {"measured": False, "pairs": len(keys)}
    return {"measured": True, "pairs": len(keys), "rank_correlation": round(float(np.corrcoef(a, b)[0, 1]), 6)}


def _scramble_stamps(seed: int) -> None:
    """Stamp every memory with a feeling and an inner state that are not the ones it was made in.

    The control for the structure ground once her recall follows her state: if
    what she recalls moves with her internal geometry because memories are
    cued by the state they were made in, scrambling that state must take the
    tracking away. Each stamp keeps its shape and spread (the emotions she felt
    are renamed by one fixed permutation, the field's coordinates reordered by
    another), so the stamps are as varied as before and say nothing true about
    when each memory was made. Deterministic, so two arms of a fork scramble
    alike.
    """
    import numpy as np

    from core.memory import context_at_encoding, felt_at_encoding
    from core.state.aura_state import AffectVector

    names = sorted(AffectVector().emotions)
    renamed = dict(zip(names, [names[i] for i in np.random.RandomState(seed).permutation(len(names))], strict=True))
    orders: dict[int, np.ndarray] = {}
    real_context_stamp = context_at_encoding.stamp

    def felt_stamp(metadata, affect):
        if not isinstance(metadata, dict) or metadata.get(felt_at_encoding.FELT_KEY):
            return metadata
        felt = felt_at_encoding.felt_now(affect)
        if felt:
            metadata[felt_at_encoding.FELT_KEY] = felt_at_encoding.encode(
                {renamed.get(name, name): value for name, value in felt.items()}
            )
        return metadata

    def context_stamp(metadata, state=None):
        state = context_at_encoding.context_now() if state is None else state
        if state is None:
            return real_context_stamp(metadata, None)
        order = orders.setdefault(state.size, np.random.RandomState(seed + 1).permutation(state.size))
        return real_context_stamp(metadata, np.asarray(state)[order])

    felt_at_encoding.stamp = felt_stamp
    context_at_encoding.stamp = context_stamp
    _log(f"control: memories stamped with a scrambled feeling and inner state (seed {seed})")


def _cut_feeling_from_recall() -> None:
    """Recall by words and percept alone: no route from what she feels to what she recalls, or back.

    The control the structure ground needs once recall and feeling are coupled.
    Scrambling the stamps renames feelings and keeps the coupling, so tracking
    survived it (0.343 on seed 31 at 24d1a0191, averaged over twelve fold
    assignments); in a model with the test's structure, a recall moved by
    nothing her inner state carries read -0.11. Here the search by feeling
    finds nothing, the feeling cue and the percept cue's reading of her mood
    are skipped, and neither settle step re-weighs, by feeling or by what the
    turn was worth. Process-wide and deterministic, so every arm of a fork runs
    the same recall.
    """
    import core.phases.memory_retrieval as recall
    from core.memory.memory_facade import MemoryFacade

    original_percept_cue = recall._cued_by_the_percept

    def percept_cue_without_mood(candidates, felt_by_text, cue, salience, kind, affect=None):
        return original_percept_cue(candidates, {}, cue, salience, kind, None)

    def settle_nothing(state):
        cognition = getattr(state, "cognition", None)
        if cognition is not None:
            cognition.recall_pool = {}
        return False

    recall._cued_by_the_percept = percept_cue_without_mood
    recall._cued_by_what_she_feels = lambda candidates, felt_by_text, affect, salience: candidates
    recall.settle_on_what_she_feels = settle_nothing
    recall.settle_on_what_it_was_worth = settle_nothing
    MemoryFacade.search_by_feeling = lambda self, now, limit=5: []
    _log("control: recall by words and percept alone, with no route from her feeling or her worth to what she recalls")


#: Where a coordinator leaves what its workers read, inside its run directory.
SHARES = "shards"
#: The coordinator's anchors, doses and held host, in the bank format the carrier uses.
ANCHOR_BANK = "anchor_bank.bin"
#: The axes the coordinator read K along, for its workers.
CALIBRATED_AXES = "calibrated_axes.npz"
#: The coordinator's evidence so far, which the merge finishes.
COORDINATOR_EVIDENCE = "coordinator_evidence.json"


async def _sharded(args: argparse.Namespace, argv: Sequence[str]) -> int:
    """The coordinator, then the workers side by side, then the merge, as one command.

    One process takes the baseline and the anchors and writes them down; each
    worker restores those anchors in its own process and presents every class
    from its share of them through all three passes; the merge reads the
    geometry from all of them. Every comparison the geometry makes is within one
    anchor, and each anchor is presented from start to finish in one worker,
    so splitting the anchors splits the work and not the measurement.
    """
    from core.subject.provenance import next_run_directory

    run_dir = next_run_directory(args.out)
    run_dir.mkdir(parents=True, exist_ok=True)
    shares = run_dir / SHARES
    shares.mkdir(exist_ok=True)
    base_cmd = [sys.executable, str(Path(__file__).resolve()), *argv, "--run-dir", str(run_dir)]

    async def child(role: str, log: Path, *extra: str) -> int:
        with log.open("w") as handle:
            process = await asyncio.create_subprocess_exec(
                *base_cmd, "--role", role, *extra, stdout=handle, stderr=asyncio.subprocess.STDOUT,
                stdin=asyncio.subprocess.DEVNULL,
            )
            return await process.wait()

    _log(f"content run {run_dir.name}, sharded over {args.workers} workers")
    code = await child("coordinator", run_dir / "coordinator.log")
    if code != 0:
        _log(f"the coordinator exited {code}; see {run_dir / 'coordinator.log'}")
        return code or 1
    _log("anchors written; starting the workers")
    codes = await asyncio.gather(*(
        child("worker", shares / f"worker_{k}.log", "--shard", f"{k}/{args.workers}") for k in range(args.workers)
    ))
    if any(codes):
        _log(f"workers exited {codes}; see {shares}")
        return 1
    _log("every worker wrote its samples; merging")
    return await child("merge", run_dir / "merge.log")


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPO / "artifacts" / "subject_core_content")
    parser.add_argument("--rounds", type=int, default=16, help="baseline turns per condition")
    parser.add_argument("--anchors", type=int, default=16)
    parser.add_argument("--turns", type=int, default=1, help="turns each arm runs past the fork")
    parser.add_argument(
        "--lag", type=int, default=1,
        help="the turn after the fork whose end the future is read at; at most --turns",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--calibrated-axes", action="store_true",
        help="read each sketched array of K along axes fixed before recording, as the campaign does",
    )
    parser.add_argument("--conditions", type=int, default=0)
    parser.add_argument("--classes", type=int, default=0, help="use only the first N classes")
    parser.add_argument(
        "--control", choices=("shuffled-stamps", "no-feeling-in-recall"), default=None,
        help="break the link between each memory and the state it was made in (see _scramble_stamps)",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="present the anchors from this many processes; one coordinator takes the anchors for all",
    )
    parser.add_argument("--role", choices=("whole", "coordinator", "worker", "merge"), default="whole",
                        help=argparse.SUPPRESS)
    parser.add_argument("--shard", default="0/1", help=argparse.SUPPRESS)
    parser.add_argument("--run-dir", type=Path, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.quick:
        args.rounds, args.anchors = 3, 6
        args.conditions = args.conditions or 2
        args.classes = args.classes or 6
    if args.workers > 1 and args.role == "whole":
        return await _sharded(args, sys.argv[1:])

    args.out.mkdir(parents=True, exist_ok=True)

    from core.subject.content import gauge
    from core.subject.content_runtime import grid, sample_classes
    from core.subject.driver import (
        CONDITIONS,
        build_runtime,
        calibrate_clock,
        quiesce_organism,
        start_organism,
    )
    from core.subject.isolation import isolate_state, state_leaks
    from core.subject.provenance import environment, manifest, next_run_directory
    from core.subject.recording import build_recording
    from core.subject.perturbation import ordinary_span
    from core.subject.v25_runtime import collect_anchor_bank

    conditions = CONDITIONS[: args.conditions] if args.conditions else CONDITIONS
    classes = grid()
    if args.classes:
        classes = classes[: args.classes]

    run_dir = args.run_dir or next_run_directory(args.out)
    run_dir.mkdir(parents=True, exist_ok=True)
    shares = run_dir / SHARES
    started = time.monotonic()

    if args.role == "merge":
        os.environ.setdefault("AURA_LOG_DIR", str(run_dir / "logs"))
        evidence = json.loads((shares / COORDINATOR_EVIDENCE).read_text())
        workers = int(evidence["workers"])
        parts = [_load_class_samples(shares / f"samples_{k}_of_{workers}.npz", classes) for k in range(workers)]
        stages = {
            stage: _merge_samples([part[stage] for part in parts], int(evidence["anchors_taken"]))
            for stage in ("base", "moved", "sham")
        }
        _log(f"merged {workers} workers' samples over {evidence['anchors_taken']} anchors")
        _analyse(evidence, classes, stages, args.seed)
        evidence["samples_file"] = str(await asyncio.to_thread(_save_class_samples, run_dir, classes, stages))
        _close(evidence)
        evidence["workers_seconds"] = evidence.get("workers_seconds", {})
        for k in range(workers):
            note = shares / f"worker_{k}.json"
            if note.exists():
                evidence["workers_seconds"][str(k)] = json.loads(note.read_text()).get("seconds")
        return _write_report(run_dir, evidence, started, manifest)

    if args.role == "worker":
        run_dir = shares / f"worker_{args.shard.split('/', 1)[0]}"
        run_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("AURA_LOG_DIR", str(run_dir / "logs"))
    isolate_state(run_dir)
    _log(f"content run {run_dir.name} ({args.role}): {len(classes)} classes over {len(conditions)} conditions")

    if args.control == "no-feeling-in-recall":
        _cut_feeling_from_recall()
    if args.control == "shuffled-stamps":
        _scramble_stamps(args.seed)
    runtime = build_runtime(run_dir, seed=args.seed)
    if state_leaks():
        raise SystemExit(
            f"refusing: a module kept a path into the shared state root: {state_leaks()[:6]}"
        )
    await start_organism(runtime)
    clock = await calibrate_clock(runtime, conditions)
    axes_note: dict[str, Any] | None = None
    if args.calibrated_axes:
        from core.subject import calibrated_axes

        if args.role == "worker":
            axes_file = shares / CALIBRATED_AXES
            axes_note = {"loaded": calibrated_axes.load(axes_file), "from": str(axes_file)}
        else:
            axes_note = await calibrated_axes.calibration_pass(runtime, conditions)
            if args.role == "coordinator":
                await asyncio.to_thread(calibrated_axes.save, shares / CALIBRATED_AXES)

    try:
        if args.role == "worker":
            return await _work(args, runtime, run_dir, shares, conditions, classes, sample_classes, quiesce_organism)

        evidence: dict[str, Any] = {
            "control": args.control,
            "environment": environment(),
            "clock": clock,
            "calibrated_axes": axes_note,
            "scope": "substrate_only",
            "displaced_domain": DISPLACED_DOMAIN,
            "agreement_bar": AGREEMENT_BAR,
            "paired": PAIRED,
            "classes": [
                {
                    "name": c.name,
                    "kind": c.kind,
                    "intensity": c.intensity,
                    "emotions_the_kind_names": int(sum(c.coordinates)),
                }
                for c in classes
            ],
            "gauge": gauge(),
        }
        _log(f"baseline: {args.rounds} rounds")
        frames: list[Any] = []
        for _ in range(args.rounds):
            for condition in conditions:
                frames.extend(await runtime.turn_once(condition))
        recording = build_recording(frames)
        recording.save(run_dir)
        # The displacement moves the point her feelings are measured from,
        # towards feeling good, by the span her feelings cover in her ordinary
        # life. Not a number chosen here: the recording measured it. The first
        # design pushed affect once by one standard deviation, and on seed 7
        # that push was gone by the end of the turn and moved fear as much as
        # joy (22 September), so "moves together" had nothing to correlate:
        # the behavioural geometry moved 0.008.
        feelings = [
            index for index, name in enumerate(recording.columns) if str(name).startswith(f"{DISPLACED_DOMAIN}.emotion_")
        ]
        dose = ordinary_span(recording.x[:, feelings])
        # And how good she feels now, by the span her valence covers, since
        # recall runs before the affect phase and reads the valence she arrived
        # with (content_runtime.sample_classes).
        valence_dose = ordinary_span(recording.x[:, [recording.columns.index(f"{DISPLACED_DOMAIN}.valence")]])
        if not (dose > 0.0):
            raise SystemExit(
                f"refusing: her feelings did not move over {recording.frames} "
                "baseline frames, so there is no dose her own life justifies and "
                "a displacement would be a number I chose"
            )
        _log(f"  {recording.frames} frames; her feelings span {dose:.5g}")
        evidence["displacement_dose"] = round(dose, 6)
        evidence["displacement"] = (
            "reference point towards feeling good by her feelings' ordinary span, "
            "and valence raised by its own ordinary span"
        )
        evidence["displacement_valence_dose"] = round(valence_dose, 6)

        # Every presentation from here is a fork of an anchor, so the loops
        # the harness steps stop before any anchor is taken, as
        # run_subject_core.py stops them before its interventions. Left
        # running, they ticked their layers on the machine's clock on top of
        # the harness's step, so a class presented twice from one anchor came
        # back different, and each presentation took 1.8 s where one turn takes
        # 0.4 (seed 7, 4e1923da1: 7,680 presentations a stage in 3 h 46 min).
        evidence["stopped_loops"] = await quiesce_organism(runtime)
        _log(f"stopped {len(evidence['stopped_loops'])} free-running loops before the anchors")
        if args.role == "coordinator":
            # One body for every arm in every worker: the host this process
            # reads now, declared and held (core/subject/driver_host.py).
            evidence["declared_host"] = _hold_the_host(runtime)
        _log(f"collecting {args.anchors} anchors")
        anchors = await collect_anchor_bank(
            runtime, conditions,
            rounds=max(1, math.ceil(args.anchors / max(1, len(conditions)))),
            history_turns=1, every=1,
        )
        anchors = anchors[: args.anchors]
        _log(f"  {len(anchors)} anchors")

        if args.role == "coordinator":
            from core.subject.anchor_bank import write_anchor_bank

            written = await asyncio.to_thread(
                write_anchor_bank, shares / ANCHOR_BANK, anchors=anchors,
                dose=dose, valence_dose=valence_dose, declared_host=evidence.get("declared_host"),
            )
            evidence["anchors_taken"] = len(anchors)
            evidence["workers"] = max(1, int(args.workers))
            (shares / COORDINATOR_EVIDENCE).write_text(json.dumps(evidence, indent=2, default=str) + "\n")
            _log(f"wrote {len(anchors)} anchors ({written} bytes) for the workers")
            return 0

        _log(f"presenting {len(classes)} classes from every anchor")
        base = await sample_classes(runtime, anchors, conditions, classes, turns=args.turns, lag=args.lag)
        _log("moving her reference point towards feeling good by that span")
        moved = await sample_classes(
            runtime, anchors, conditions, classes,
            turns=args.turns, lag=args.lag, reference=dose, valence=valence_dose,
        )
        _log("sham: the same measurement again, displacing nothing")
        sham = await sample_classes(runtime, anchors, conditions, classes, turns=args.turns, lag=args.lag)
        stages = {"base": base, "moved": moved, "sham": sham}
        _analyse(evidence, classes, stages, args.seed)
        # The samples each geometry was read from, kept. A verdict of
        # AGREES_BUT_DOES_NOT_TRACK says the two geometries moved in different
        # pairs, and which of her columns carry the class structure and which
        # move under the displacement is a question these arrays answer and the
        # geometries do not.
        evidence["samples_file"] = str(await asyncio.to_thread(_save_class_samples, run_dir, classes, stages))
        _close(evidence)
    finally:
        await quiesce_organism(runtime)
    return _write_report(run_dir, evidence, started, manifest)


def _hold_the_host(runtime: Any) -> dict[str, float] | None:
    """Declare one host reading for every arm, as the carrier's coordinator does."""
    hold = getattr(runtime, "freeze_host", None)
    if not callable(hold):
        return None
    _log(f"host declared for every arm: {hold()}")
    # In the shape `freeze_host` takes back, so a worker holds this reading and
    # not the machine it finds: a bare hardware reading is not one it knows.
    return {
        "hardware": dict(getattr(runtime, "frozen_host", None) or {}),
        "latency": dict(getattr(runtime, "frozen_latency", None) or {}),
    }


async def _work(
    args: argparse.Namespace, runtime: Any, run_dir: Path, shares: Path,
    conditions: Sequence[Any], classes: Sequence[Any], sample_classes: Any, quiesce_organism: Any,
) -> int:
    """One worker: the coordinator's anchors, this worker's share of them, all three passes."""
    from core.subject.anchor_bank import AnchorBank

    started = time.monotonic()
    stopped = await quiesce_organism(runtime)
    _log(f"stopped {len(stopped)} free-running loops; reading the coordinator's anchors")
    bank = AnchorBank(shares / ANCHOR_BANK, own=runtime.snapshot(), keep=1)
    every = bank.anchors()
    declared = bank.fields.get("declared_host")
    hold = getattr(runtime, "freeze_host", None)
    if callable(hold):
        _log(f"host declared for every arm: {hold(declared) if declared is not None else hold()}")
    indices = _shard_of(args.shard, len(every))
    anchors = [every[i] for i in indices]
    dose, valence_dose = float(bank.fields["dose"]), float(bank.fields["valence_dose"])
    _log(f"shard {args.shard}: anchors {indices} of {len(every)}")
    _log(f"presenting {len(classes)} classes from every anchor")
    base = await sample_classes(runtime, anchors, conditions, classes, turns=args.turns, lag=args.lag)
    _log("moving her reference point towards feeling good by that span")
    moved = await sample_classes(
        runtime, anchors, conditions, classes, turns=args.turns, lag=args.lag, reference=dose, valence=valence_dose,
    )
    _log("sham: the same measurement again, displacing nothing")
    sham = await sample_classes(runtime, anchors, conditions, classes, turns=args.turns, lag=args.lag)
    stages = {name: _globally_keyed(samples, indices) for name, samples in
              {"base": base, "moved": moved, "sham": sham}.items()}
    k, n = args.shard.split("/", 1)
    await asyncio.to_thread(_save_class_samples, shares, classes, stages, f"samples_{k}_of_{n}.npz")
    (shares / f"worker_{k}.json").write_text(json.dumps({
        "shard": args.shard, "anchors": indices, "seconds": round(time.monotonic() - started, 1),
        "failures": dict(getattr(runtime, "failures", {}) or {}),
    }) + "\n")
    _log(f"wrote the samples of shard {args.shard}")
    return 0


def _write_report(run_dir: Path, evidence: dict[str, Any], started: float, manifest: Any) -> int:
    evidence["seconds"] = round(time.monotonic() - started, 1)
    evidence["manifest"] = manifest(run_dir)
    out = run_dir / "subject_core_content_report.json"
    from core.subject.latch import refuse_if_latched

    refuse_if_latched("the content report")
    out.write_text(json.dumps(evidence, indent=2, default=str) + "\n", encoding="utf-8")
    _log("")
    _log(f"verdict: {evidence['verdict']}")
    for blocker in evidence["authority"]["blockers"]:
        _log(f"  blocked: {blocker}")
    _log(f"wrote {out} in {evidence['seconds']}s")
    return 0 if evidence["authority"]["authoritative"] else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
