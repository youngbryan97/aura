#!/usr/bin/env python3
"""An accelerated lifetime: millions of firings, then the world changes.

Four Gap Atlas cards ask the same question from different directions.

  026   chunk utility over an accelerated lifetime, with distribution shift
  A12.4 a long episode compiled, then re-run on unseen related states
  A12.5 the same run scored on the minimal support the compiler kept
  A12.6 the same run scored on firings compressed per compiled rule

Soar's long experience with chunking is that compiled rules are a liability
as often as an asset: a rule learned under one distribution keeps firing
after the distribution moves, and the system gets faster at being wrong. The
bar is not "does chunking speed things up" — it is "does the utility
accounting notice when a chunk stops paying, and how fast".

So: run a lifetime of task firings, compile what recurs, shift the world
under the compiled rules, and measure what the registry does about it. The
control is the same lifetime with pruning disabled, which is what a system
that compiles but does not audit looks like.

    python tools/campaigns/procedure_lifetime.py --firings 2000000
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.cognition.cognitive_event import (  # noqa: E402
    EventGraph,
    Phase,
    ReadDependency,
)
from core.cognition.procedure import (  # noqa: E402
    Backend,
    Effect,
    Precondition,
    ProceduralValue,
    ProcedureRegistry,
    Signature,
)
from core.cognition.trace_compiler import TraceCompiler  # noqa: E402


def _resident_bytes() -> int:
    """Resident set size in bytes, or 0 where the platform will not say."""
    try:
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except (ImportError, OSError, ValueError):
        return 0
    # Linux reports kilobytes, macOS bytes.
    return peak if peak > 1 << 20 else peak * 1024

#: Each task reads three keys that matter and two that only happen to be
#: there. A compiler that keeps all five has learned the room, not the task.
REAL_KEYS = ("goal", "board", "hand")
INCIDENTAL_KEYS = ("clock", "battery")

#: How many deliberation steps one episode takes before the chunk exists. The
#: cards say "previously hundreds-step episode", and three steps would make
#: the compression claim about nothing.
EPISODE_STEPS = 240


def _episode(
    graph: EventGraph,
    task: str,
    rng: random.Random,
    *,
    shifted: bool,
    steps: int = EPISODE_STEPS,
) -> int:
    """One run of a task. Returns the terminal event id."""
    reads = [
        ReadDependency(
            key=key,
            value_digest=f"{key}:{rng.randrange(4)}",
            owner="world",
        )
        for key in REAL_KEYS
    ]
    # The incidental keys vary every run, which is the signal that they are
    # not preconditions. After the shift one of them freezes, which is how a
    # spurious condition gets learned by a compiler that only looks at variance.
    for key in INCIDENTAL_KEYS:
        digest = f"{key}:0" if (shifted and key == "clock") else f"{key}:{rng.randrange(64)}"
        reads.append(ReadDependency(key=key, value_digest=digest, owner="world"))

    first = graph.record(
        Phase.PERCEIVE, "world", f"{task}:read", reads=reads, duration_s=0.001
    )
    # A long deliberation, not a token one: this is what the compiled rule
    # replaces, and a compression ratio measured against three steps says
    # nothing about the claim the cards make.
    parent = first.seq
    for step in range(steps):
        parent = graph.record(
            Phase.ELABORATE,
            "planner",
            f"{task}:think:{step}",
            parents=[parent],
            duration_s=0.004,
        ).seq
    last = graph.record(
        Phase.APPLY,
        "actor",
        f"{task}:do",
        parents=[parent],
        produced=[f"task:{task}:done"],
        duration_s=0.002,
        outcome="ok",
    )
    return last.seq


def compile_phase(
    registry: ProcedureRegistry, tasks: list[str], *, runs: int, seed: int
) -> dict[str, object]:
    """Live the pre-shift life and compile what recurs."""
    rng = random.Random(seed)
    graph = EventGraph(capacity=runs * len(tasks) * (EPISODE_STEPS + 4) + 64)
    compiler = TraceCompiler(registry)
    for _ in range(runs):
        for task in tasks:
            terminal = _episode(graph, task, rng, shifted=False)
            compiler.observe(graph, task, terminal)
    results = {task: compiler.compile(task) for task in tasks}
    kept = {
        task: len(r.compiled.signature.preconditions) if r.compiled else 0
        for task, r in results.items()
    }
    steps_before = EPISODE_STEPS + 2
    return {
        "episode_steps_before": steps_before,
        "steps_after_compilation": 1,
        "compression_ratio": steps_before,
        "compiled": sum(1 for r in results.values() if r.compiled),
        "refused": sum(1 for r in results.values() if not r.compiled),
        "support_kept_median": statistics.median(kept.values()) if kept else 0,
        "support_kept_min": min(kept.values(), default=0),
        "support_kept_max": max(kept.values(), default=0),
        "incidental_kept": sum(
            1
            for r in results.values()
            if r.compiled
            and any(
                p.key in INCIDENTAL_KEYS for p in r.compiled.signature.preconditions
            )
        ),
        "provisional_median": statistics.median(
            [
                len(r.compiled.origin.provisional_conditions)
                for r in results.values()
                if r.compiled and r.compiled.origin
            ]
            or [0]
        ),
        "procedure_ids": [
            r.compiled.procedure_id for r in results.values() if r.compiled
        ],
        "runs_per_task": runs,
    }


def generalise_phase(
    registry: ProcedureRegistry, procedure_ids: list[str]
) -> dict[str, object]:
    """Give each rule one run that did without an incidental key, and widen it.

    This is the half a success trace cannot supply. A key present in every
    successful run is indistinguishable from a key the task needs, and stays a
    condition until something succeeds without it.
    """
    widened: list[str] = []
    dropped = 0
    for pid in procedure_ids:
        procedure = registry.get(pid)
        if procedure is None or procedure.origin is None:
            continue
        current = pid
        for key in procedure.origin.provisional_conditions:
            if key not in INCIDENTAL_KEYS:
                continue
            child = registry.generalise(
                current, key, witness=f"run-without-{key}"
            )
            if child is not None:
                current = child.procedure_id
                dropped += 1
        if current != pid:
            widened.append(current)
    return {
        "widened": len(widened),
        "conditions_dropped": dropped,
        "procedure_ids": widened,
    }


def transfer_and_lesion(
    registry: ProcedureRegistry,
    procedure_ids: list[str],
    *,
    unseen: int,
    rng: random.Random,
) -> dict[str, object]:
    """Does the compiled rule fire on states it never saw, and stop when lesioned?

    Two halves of the same bar. A rule that fires on unseen states of the same
    family is a reusable response; one that fires on every state is a rule
    with no content, which is why the lesion arm has to be run in the same
    breath. Each required condition is removed from the state in turn: if the
    rule still fires, that condition was never doing anything.
    """
    fired_on_unseen = 0
    offered = 0
    lesion_offered = 0
    lesion_fired = 0
    for pid in procedure_ids:
        procedure = registry.get(pid)
        if procedure is None:
            continue
        keys = [p.key for p in procedure.signature.preconditions]
        for _ in range(unseen):
            # A state of the same family the rule has never met: every
            # required key present, every value new.
            state = {k: f"unseen-{rng.randrange(1 << 30)}" for k in keys}
            offered += 1
            if procedure.signature.matches(state):
                fired_on_unseen += 1
            for missing in keys:
                lesioned = {k: v for k, v in state.items() if k != missing}
                lesion_offered += 1
                if procedure.signature.matches(lesioned):
                    lesion_fired += 1
    return {
        "unseen_states_offered": offered,
        "fired_on_unseen": fired_on_unseen,
        "unseen_fire_rate": round(fired_on_unseen / offered, 4) if offered else 0.0,
        "lesioned_states_offered": lesion_offered,
        "fired_when_lesioned": lesion_fired,
        "lesion_prevents_firing": lesion_fired == 0,
    }


def lifetime(
    registry: ProcedureRegistry,
    procedure_ids: list[str],
    *,
    firings: int,
    shift_at: float,
    pays_before: float,
    pays_after: float,
    prune_every: int,
    seed: int,
) -> dict[str, object]:
    """Fire the compiled rules for a lifetime, moving the world part-way."""
    rng = random.Random(seed)
    shift_index = int(firings * shift_at)
    retired_at: dict[str, int] = {}
    wrong_after_shift = 0
    fired_after_shift = 0
    # "Memory stays bounded" is a claim about the high-water mark, not about
    # the value at the end: a registry that grows to a million and then prunes
    # is not bounded, and only a sample taken during the run can see that.
    peak_procedures = 0
    peak_rss = _resident_bytes()
    began = time.perf_counter()

    for i in range(firings):
        pid = procedure_ids[i % len(procedure_ids)]
        if pid not in retired_at:
            after = i >= shift_index
            p_success = pays_after if after else pays_before
            success = rng.random() < p_success
            # No value passed: the deliberation the compiler measured is what a
            # firing is worth. Restating it here would make the campaign score
            # its own number instead of the registry's.
            registry.record_use(pid, success=success)
            if after:
                fired_after_shift += 1
                if not success:
                    wrong_after_shift += 1
        # Outside the skip. Inside it, the audit only ran on the ticks that
        # happened to land on a live rule, and six rules with a negative net
        # were never looked at again.
        if prune_every and (i + 1) % prune_every == 0:
            for gone in registry.prune():
                retired_at.setdefault(gone.procedure_id, i)
        if (i + 1) % 100_000 == 0:
            peak_procedures = max(peak_procedures, registry.report()["procedures"])
            peak_rss = max(peak_rss, _resident_bytes())

    seconds = time.perf_counter() - began
    lag = [
        retired_at[pid] - shift_index
        for pid in procedure_ids
        if pid in retired_at and retired_at[pid] >= shift_index
    ]
    return {
        "firings": firings,
        "peak_live_procedures": max(peak_procedures, len(procedure_ids)),
        "peak_resident_mb": round(peak_rss / (1 << 20), 1),
        "seconds": round(seconds, 2),
        "firings_per_second": round(firings / seconds) if seconds else 0,
        "shift_at_firing": shift_index,
        "retired": len(retired_at),
        "of": len(procedure_ids),
        "median_firings_to_notice": statistics.median(lag) if lag else None,
        "wrong_firings_after_shift": wrong_after_shift,
        "wrong_rate_after_shift": (
            round(wrong_after_shift / fired_after_shift, 4) if fired_after_shift else 0.0
        ),
    }


def sweep_decay(
    tasks: list[str], *, firings: int, decays: list[float], seed: int, args
) -> list[dict[str, object]]:
    """What each decay costs, so the constant is a measurement.

    Two costs pull opposite ways. A slow decay keeps firing a rule the world
    has moved past; a fast one retires a rule on a short run of bad luck. Both
    are counted here, on the same lifetime, and the constant in
    core/cognition/procedure.py is read off this table.
    """
    from core.cognition import procedure as procedure_module

    original = procedure_module._RECENT_DECAY
    rows: list[dict[str, object]] = []
    try:
        for decay in decays:
            procedure_module._RECENT_DECAY = decay
            registry = ProcedureRegistry(max_procedures=len(tasks) * 8)
            compiled = compile_phase(
                registry, tasks, runs=args.runs_per_task, seed=seed
            )
            ids = list(compiled["procedure_ids"])
            shifted = lifetime(
                registry, ids, firings=firings, shift_at=args.shift_at,
                pays_before=args.pays_before, pays_after=args.pays_after,
                prune_every=args.prune_every, seed=seed + 1,
            )
            # The other arm: the world never moves. Anything retired here was
            # retired for a run of luck, which is the cost of a fast decay.
            steady_registry = ProcedureRegistry(max_procedures=len(tasks) * 8)
            steady_compiled = compile_phase(
                steady_registry, tasks, runs=args.runs_per_task, seed=seed
            )
            steady = lifetime(
                steady_registry, list(steady_compiled["procedure_ids"]),
                firings=firings, shift_at=args.shift_at,
                pays_before=args.pays_before, pays_after=args.pays_before,
                prune_every=args.prune_every, seed=seed + 1,
            )
            rows.append({
                "decay": decay,
                "firings_to_notice": shifted["median_firings_to_notice"],
                "wrong_firings_after_shift": shifted["wrong_firings_after_shift"],
                "retired_after_shift": shifted["retired"],
                "retired_with_no_shift": steady["retired"],
                "of": shifted["of"],
            })
            print(json.dumps(rows[-1]))
    finally:
        procedure_module._RECENT_DECAY = original
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=int, default=64)
    parser.add_argument("--runs-per-task", type=int, default=6)
    parser.add_argument("--firings", type=int, default=2_000_000)
    parser.add_argument("--shift-at", type=float, default=0.5)
    parser.add_argument("--pays-before", type=float, default=0.92)
    parser.add_argument("--pays-after", type=float, default=0.18)
    parser.add_argument("--prune-every", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=1_618_033)
    parser.add_argument("--out", default="docs/evidence/procedure_lifetime.json")
    parser.add_argument("--sweep-decay", default="")
    args = parser.parse_args()

    tasks = [f"t{i}" for i in range(args.tasks)]

    if args.sweep_decay:
        rows = sweep_decay(
            tasks,
            firings=args.firings,
            decays=[float(d) for d in args.sweep_decay.split(",")],
            seed=args.seed,
            args=args,
        )
        payload = {
            "schema": "aura.procedure_lifetime_halflife.v1",
            "cards": ["026"],
            "claim_boundary": (
                "decay sweep on the synthetic accelerated lifetime; picks the "
                "constant in core/cognition/procedure.py and nothing else"
            ),
            "config": {"firings": args.firings, "tasks": args.tasks, "seed": args.seed},
            "rows": rows,
        }
        from core.governance_context import local_internal_governed_scope
        from core.runtime.file_write_gateway import get_file_write_gateway

        out = ROOT / "docs/evidence/procedure_lifetime_halflife.json"
        with local_internal_governed_scope("procedure_lifetime_campaign"):
            get_file_write_gateway().write_text(
                out, json.dumps(payload, indent=2, sort_keys=True) + "\n"
            )
        print(f"wrote {out}")
        return 0

    registry = ProcedureRegistry(max_procedures=args.tasks * 8)
    compiled = compile_phase(
        registry, tasks, runs=args.runs_per_task, seed=args.seed
    )
    ids = list(compiled.pop("procedure_ids"))
    if not ids:
        print("nothing compiled; the campaign has no subject", file=sys.stderr)
        return 1

    widened = generalise_phase(registry, ids)
    transfer = transfer_and_lesion(
        registry, ids, unseen=8, rng=random.Random(args.seed + 2)
    )
    audited = lifetime(
        registry,
        ids,
        firings=args.firings,
        shift_at=args.shift_at,
        pays_before=args.pays_before,
        pays_after=args.pays_after,
        prune_every=args.prune_every,
        seed=args.seed + 1,
    )

    # The control: compile the same way, then never audit. This is what
    # chunking without utility accounting does over the same lifetime.
    control_registry = ProcedureRegistry(max_procedures=args.tasks * 8)
    control_compiled = compile_phase(
        control_registry, tasks, runs=args.runs_per_task, seed=args.seed
    )
    control_ids = list(control_compiled.pop("procedure_ids"))
    unaudited = lifetime(
        control_registry,
        control_ids,
        firings=args.firings,
        shift_at=args.shift_at,
        pays_before=args.pays_before,
        pays_after=args.pays_after,
        prune_every=0,
        seed=args.seed + 1,
    )

    # The half of card 026 the shifted arm cannot answer: over the same
    # lifetime with the world holding still, does the audit leave the useful
    # old procedures alone? A retirement rule that clears the shelf is not an
    # improvement on one that never clears anything.
    steady_registry = ProcedureRegistry(max_procedures=args.tasks * 8)
    steady_compiled = compile_phase(
        steady_registry, tasks, runs=args.runs_per_task, seed=args.seed
    )
    steady = lifetime(
        steady_registry,
        list(steady_compiled["procedure_ids"]),
        firings=args.firings,
        shift_at=args.shift_at,
        pays_before=args.pays_before,
        pays_after=args.pays_before,
        prune_every=args.prune_every,
        seed=args.seed + 1,
    )

    payload = {
        "schema": "aura.procedure_lifetime.v1",
        "cards": ["026", "A12.4", "A12.5", "A12.6"],
        "claim_boundary": (
            "synthetic accelerated lifetime over compiled procedures with a "
            "planted distribution shift; measures the utility accounting, not "
            "task competence in any real environment"
        ),
        "config": {
            "tasks": args.tasks,
            "runs_per_task": args.runs_per_task,
            "firings": args.firings,
            "shift_at": args.shift_at,
            "pays_before": args.pays_before,
            "pays_after": args.pays_after,
            "prune_every": args.prune_every,
            "seed": args.seed,
        },
        "compilation": compiled,
        "generalisation": {
            k: v for k, v in widened.items() if k != "procedure_ids"
        },
        "transfer_and_lesion": transfer,
        "audited": audited,
        "unaudited_control": unaudited,
        "steady_world_control": steady,
        "useful_old_procedures_surviving": {
            "of": steady["of"],
            "kept": steady["of"] - steady["retired"],
        },
        "wrong_firings_avoided": (
            unaudited["wrong_firings_after_shift"]
            - audited["wrong_firings_after_shift"]
        ),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))

    from core.governance_context import local_internal_governed_scope
    from core.runtime.file_write_gateway import get_file_write_gateway

    out = ROOT / args.out
    with local_internal_governed_scope("procedure_lifetime_campaign"):
        get_file_write_gateway().write_text(
            out, json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
