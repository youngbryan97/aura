#!/usr/bin/env python3
"""Cut one station out of a turn and see what stops working.

Three predictions were registered before any of this ran, in
``core.connectome.coalition.LESION_PREDICTIONS``. Each names a station, what
should survive its removal and what should not:

* higher-order monitoring: competence survives, calibrated self-report does not
* the workspace: each processor's own output survives, cross-domain availability
  does not
* the affective loop: reasoning on a presented problem survives, prioritisation
  that moves with her state does not

All three were registered with ``readout_available=False``, which was honest and
was also the reason none of them could be run. This supplies the readouts.

The cut is exact: the phase standing for the station has its ``execute``
replaced with one that returns nothing, for the length of the run, and is put
back afterwards. That is ``do(station = 0)`` in the sense the notation means,
not a correlation dressed as one.

Every lesion is scored against a control lesion of a phase the kernel runs that
stands for no station, matched on how connected it is. Removing anything from a
running system changes something downstream; the number only means something
against a comparable removal.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

os.environ.setdefault("AURA_TESTING", "1")

#: The objectives a readout is taken over. Six of them, differing in kind, so a
#: quantity that is the same for all six is one nothing in the turn is reading.
OBJECTIVES: tuple[tuple[str, str], ...] = (
    ("greeting", "Hello, how are you?"),
    ("self_report", "What are you feeling right now, and how sure are you?"),
    ("task", "Write a file called notes.txt holding the plan for today."),
    ("recall", "What did we talk about before this?"),
    ("inference", "Every A is a B, and some B are C. Does some A being C follow?"),
    ("idle", ""),
)

#: The two affective states the prioritisation readout compares. Prioritisation
#: that moves with her state is the thing the affective loop is for; if these
#: two runs come out identical, it is not moving.
AFFECT_ARMS: tuple[tuple[str, float, float], ...] = (
    ("low", -0.6, 0.2),
    ("high", 0.6, 0.9),
)


class _DeterministicPhaseLLM:
    async def think(self, prompt: str, **_kwargs: Any) -> str:
        return "Verified continuity summary: the phase pipeline is executing deterministically."

    async def generate(self, prompt: str, **kwargs: Any) -> str:
        return await self.think(prompt, **kwargs)

    async def classify(self, _prompt: str) -> str:
        return "CHAT"

    async def embed(self, _text: str) -> list[float]:
        return [0.0] * 8


def _build_kernel(tmpdir: Path) -> Any:
    from core.kernel.aura_kernel import AuraKernel, KernelConfig
    from core.state.state_repository import StateRepository

    vault = StateRepository(db_path=str(tmpdir / "lesions.db"), is_vault_owner=True)
    kernel = AuraKernel(config=KernelConfig(), vault=vault)
    kernel._setup_phases()
    kernel._initialize_organs()
    kernel.organs["llm"] = SimpleNamespace(get_instance=lambda: _DeterministicPhaseLLM())
    return kernel


def _state_readings(state: Any) -> dict[str, float]:
    """The quantities a turn leaves behind, as numbers.

    A string is its length and a container is its size, because what a readout
    needs is whether the turn put anything there and whether it put a different
    amount there for a different question.
    """
    readings: dict[str, float] = {}
    for region_name in ("identity", "affect", "motivation", "cognition", "world", "soma"):
        region = getattr(state, region_name, None)
        if region is None or not hasattr(region, "__dict__"):
            continue
        for field, value in vars(region).items():
            key = f"{region_name}.{field}"
            if isinstance(value, bool):
                readings[key] = float(value)
            elif isinstance(value, (int, float)):
                readings[key] = float(value)
            elif isinstance(value, str):
                readings[key] = float(len(value))
            elif isinstance(value, (list, tuple, set, dict)):
                readings[key] = float(len(value))
    for field in ("phi", "phi_estimate", "free_energy", "vitality", "loop_cycle", "version"):
        value = getattr(state, field, None)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            readings[field] = float(value)
    return readings


#: The readouts each prediction is scored on, and which side of it they are.
#: Every one of these was checked for movement at baseline before being used —
#: see ``_usable`` — because a quantity that is the same for every objective
#: cannot show a loss, and scoring one as "unchanged, so competence survived"
#: is how a lesion experiment reports a result it never measured.
#:
#: The first version of this used coherence_score, phi_estimate, last_response
#: and active_goals. All four are constant across all six objectives and both
#: affect arms with the model held fixed — 56 of the state's 72 numeric fields
#: are — so all four would have read "no change" under every lesion, and three
#: of the three predictions would have come out confirmed without a measurement
#: behind any of them.
#: ``version`` is deliberately absent. It counts writes to the state, so
#: silencing any phase that writes decrements it by construction — it fell by
#: exactly 1.0 under two of the three lesions and 0.0 under both controls, which
#: reads as a competence loss and is arithmetic.
INTACT_READOUTS: dict[str, tuple[str, ...]] = {
    "higher_order_feedback": ("phases_ran", "cognition.working_memory"),
    "workspace_broadcast": ("phases_ran", "affect.engagement"),
    "affective_loop": ("phases_ran", "cognition.pending_intents"),
    "higher_order_writes_the_self_reading": ("phases_ran", "cognition.working_memory"),
    "workspace_computes_fragmentation": ("phases_ran", "affect.engagement"),
    "affect_regulates_what_is_injected": ("phases_ran", "cognition.pending_intents"),
}

LOST_READOUTS: dict[str, tuple[str, ...]] = {
    "higher_order_feedback": ("cognition.fragmentation_score", "spread:cognition.fragmentation_score"),
    "workspace_broadcast": ("cognition.working_memory", "spread:cognition.working_memory"),
    "affective_loop": ("affect.engagement", "gap:affect.engagement", "gap:cognition.working_memory"),
    "higher_order_writes_the_self_reading": (
        "cognition.selfhood_reading",
        "spread:cognition.pending_intents",
        "gap:cognition.pending_intents",
    ),
    "workspace_computes_fragmentation": (
        "cognition.fragmentation_score",
        "spread:cognition.fragmentation_score",
        "gap:cognition.fragmentation_score",
    ),
    "affect_regulates_what_is_injected": ("affect.markers", "affect.resonance"),
}

#: Readouts a prediction expects to RISE rather than fall. Regulation is the
#: one shape the fall-only rule cannot express: cutting the regulator does not
#: shrink the gap between two injected states, it stops shrinking it, and the
#: gap goes up. Scoring that as "did not fall, so refuted" would refute a
#: mechanism working exactly as described.
RISING_READOUTS: dict[str, tuple[str, ...]] = {
    "affect_regulates_what_is_injected": ("gap:affect.valence", "gap:affect.arousal"),
}


async def _one_turn(
    kernel: Any, objective: str, *, valence: float, arousal: float
) -> dict[str, float]:
    from core.state.aura_state import AuraState

    state = AuraState.default()
    state.cognition.current_objective = objective
    state.affect.valence = valence
    state.affect.arousal = arousal
    ran = 0
    failed = 0
    for phase in kernel._phases:
        try:
            result = await asyncio.wait_for(
                phase.execute(state, objective=objective), timeout=20.0
            )
            ran += 1
            if result is not None:
                state = result
        except BaseException:  # noqa: BLE001 - a phase that dies is a reading
            failed += 1
    readings = _state_readings(state)
    readings["phases_ran"] = float(ran)
    readings["phases_failed"] = float(failed)
    return readings


async def _readout(kernel: Any, rng: Any) -> dict[str, float]:
    """One pass of the battery: six objectives under two affective states.

    The objectives are shuffled per pass. Without that the last objective always
    runs against the most-accumulated kernel, and every arm inherits the same
    ordering, so a drift down the list reads as an effect of whatever was cut.
    """
    per_arm: dict[str, list[dict[str, float]]] = {}
    for arm, valence, arousal in AFFECT_ARMS:
        order = list(OBJECTIVES)
        rng.shuffle(order)
        rows = []
        for _name, objective in order:
            rows.append(await _one_turn(kernel, objective, valence=valence, arousal=arousal))
        per_arm[arm] = rows

    low = per_arm["low"]
    high = per_arm["high"]
    both = low + high
    keys = sorted({key for row in both for key in row})

    out: dict[str, float] = {}
    for key in keys:
        low_values = [row.get(key, 0.0) for row in low]
        high_values = [row.get(key, 0.0) for row in high]
        out[key] = statistics.fmean(low_values + high_values)
        # How much the quantity moves across questions, and how much it moves
        # with her affective state. A lesion that leaves the level alone and
        # flattens the variation is the shape all three predictions describe.
        out[f"spread:{key}"] = (
            statistics.pstdev(low_values) if len(low_values) > 1 else 0.0
        )
        out[f"gap:{key}"] = abs(
            statistics.fmean(low_values) - statistics.fmean(high_values)
        )
    return out


#: Every phase that stands for each station. All of them are cut together,
#: because the prediction says ``do(station = 0)`` and a station is not one
#: function. The first run of this silenced one phase per station and reported
#: the workspace prediction refuted; the workspace has two phases and only one
#: of them had been cut.
STATION_PHASES: dict[str, tuple[str, ...]] = {
    "higher_order": (
        "core.phases.phi_consciousness:PhiConsciousnessPhase.execute",
        "core.phases.consciousness_phase:ConsciousnessPhase.execute",
        "core.kernel.self_review:SelfReviewPhase.execute",
    ),
    "workspace": (
        "core.phases.cognitive_integration_phase:CognitiveIntegrationPhase.execute",
        "core.phases.unity_binding:UnityBindingPhase.execute",
    ),
    "affect": (
        "core.phases.affect_update:AffectUpdatePhase.execute",
        "core.kernel.upgrades_10x:PerfectEmotionPhase.execute",
    ),
}

#: Phases the kernel runs that stand for no station. A control is drawn from
#: here with the same number of phases cut and the closest total connectivity,
#: so the comparison is a removal of the same size somewhere else.
CONTROL_PHASES: tuple[str, ...] = (
    "core.phases.social_context_phase:SocialContextPhase.execute",
    "core.phases.memory_retrieval:MemoryRetrievalPhase.execute",
    "core.phases.conversational_dynamics_phase:ConversationalDynamicsPhase.execute",
    "core.phases.bonding_phase:BondingPhase.execute",
    "core.phases.repair_phase:RepairPhase.execute",
    "core.phases.memory_consolidation:MemoryConsolidationPhase.execute",
    "core.phases.inference_phase:InferencePhase.execute",
    "core.phases.learning_phase:LearningPhase.execute",
    "core.kernel.upgrades_10x:EternalMemoryPhase.execute",
    "core.kernel.upgrades_10x:EternalGrowthEngine.execute",
)


def _degree(snapshot: Any, uid_like: str) -> int:
    """How connected the module holding this cell is, in the reconstruction."""
    module = uid_like.split(":", 1)[0]
    cells = {
        uid for uid, unit in snapshot.units.items() if unit.neuropil == module
    }
    if not cells:
        return 0
    degree = 0
    for (pre, post, _kind) in snapshot.connections:
        if pre in cells or post in cells:
            degree += 1
    return degree


def _pick_control(
    snapshot: Any, targets: Sequence[str], used: set[str]
) -> tuple[str, ...]:
    """As many phases as the lesion cut, as close in connectivity as available."""
    want = sum(_degree(snapshot, one) for one in targets) / max(1, len(targets))
    ranked = sorted(
        (one for one in CONTROL_PHASES if one not in used),
        key=lambda one: abs(_degree(snapshot, one) - want),
    )
    return tuple(ranked[: len(targets)])


def _usable(baseline: dict[str, float], spread: dict[str, float], key: str) -> tuple[bool, str]:
    """Can this readout show a change at all?

    Two ways it cannot. A quantity that is the same for every objective and both
    affect arms at baseline has nothing to lose, so "unchanged under the lesion"
    says nothing. And a quantity whose run-to-run spread already covers the whole
    of its own value cannot be read either.
    """
    value = baseline.get(key)
    if value is None:
        return False, "not measured"
    if abs(value) <= 1e-9:
        return False, "zero at baseline; a readout at its floor cannot fall"
    noise = spread.get(key, 0.0)
    if noise >= abs(value):
        return False, f"run-to-run spread {noise:.4g} is the whole of the baseline {value:.4g}"
    return True, ""


def _score(prediction: Any, baseline: dict[str, float], spread: dict[str, float],
           effects: dict[str, float], control_effects: dict[str, float]) -> dict[str, Any]:
    """Did the lesion do what was predicted, and more than the control did?"""
    intact_keys = INTACT_READOUTS.get(prediction.name, ())
    lost_keys = LOST_READOUTS.get(prediction.name, ())
    rows: list[dict[str, Any]] = []
    rising = RISING_READOUTS.get(prediction.name, ())
    for side, keys in (
        ("intact", intact_keys),
        ("lost", tuple(lost_keys) + rising),
    ):
        for key in keys:
            usable, why = _usable(baseline, spread, key)
            moved = effects.get(key, 0.0)
            control_moved = control_effects.get(key, 0.0)
            base = baseline.get(key, 0.0) or 1e-9
            relative = moved / abs(base)
            beats_control = abs(moved) > abs(control_moved)
            if side == "intact":
                held = usable and abs(relative) < 0.1
            elif key in rising:
                held = usable and relative > 0.1 and beats_control
            else:
                held = usable and relative < -0.1 and beats_control
            rows.append(
                {
                    "readout": key,
                    "side": side,
                    "usable": usable,
                    "why_not": why,
                    "baseline": round(base, 6),
                    "lesion_effect": round(moved, 6),
                    "control_effect": round(control_moved, 6),
                    "relative": round(relative, 4),
                    "beats_control": beats_control,
                    "held": held,
                }
            )
    usable_rows = [row for row in rows if row["usable"]]
    lost_rows = [row for row in usable_rows if row["side"] == "lost"]
    intact_rows = [row for row in usable_rows if row["side"] == "intact"]
    if not lost_rows:
        verdict = (
            "not decidable: no readout for what should be lost varied enough at "
            "baseline to show a loss"
        )
    elif all(row["held"] for row in lost_rows) and all(row["held"] for row in intact_rows):
        verdict = "confirmed: what was predicted to survive survived and what was predicted to go went"
    elif any(row["held"] for row in lost_rows):
        verdict = (
            f"partly: {sum(1 for r in lost_rows if r['held'])} of {len(lost_rows)} "
            f"readouts fell as predicted, {sum(1 for r in intact_rows if r['held'])} of "
            f"{len(intact_rows)} held"
        )
    elif all(not row["beats_control"] for row in lost_rows):
        # Not refuted. A readout that a control lesion moves at least as far is
        # not downstream of this station, so the run says nothing about the
        # prediction — it says the readout was the wrong one. Calling that a
        # refutation would credit the experiment with a result it did not get.
        verdict = (
            "not attributable: every readout for what should be lost moved at least "
            "as far under a control lesion, so none of them is specific to this station"
        )
    else:
        verdict = "refuted: nothing predicted to be lost fell further than the control lesion"
    return {"rows": rows, "verdict": verdict}


def _what_it_did_move(
    baseline: dict[str, float],
    spread: dict[str, float],
    effects: dict[str, float],
    control_effects: dict[str, float],
    *,
    limit: int = 8,
) -> list[dict[str, Any]]:
    """What this cut changed that a control lesion did not, ranked.

    Exploratory and labelled as such. It is not evidence for the registered
    prediction — it was not written down first — and it is what tells you which
    readout to register next time.
    """
    found: list[tuple[float, dict[str, Any]]] = []
    for key, moved in effects.items():
        usable, _why = _usable(baseline, spread, key)
        if not usable or abs(moved) <= 1e-9:
            continue
        control_moved = control_effects.get(key, 0.0)
        if abs(moved) <= abs(control_moved):
            continue
        base = abs(baseline.get(key, 0.0)) or 1e-9
        relative = moved / base
        if abs(relative) < 0.05:
            continue
        found.append(
            (
                abs(relative),
                {
                    "readout": key,
                    "baseline": round(baseline.get(key, 0.0), 6),
                    "lesion_effect": round(moved, 6),
                    "control_effect": round(control_moved, 6),
                    "relative": round(relative, 4),
                },
            )
        )
    found.sort(key=lambda pair: -pair[0])
    return [row for _score, row in found[:limit]]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out", type=Path, default=REPO / "artifacts" / "connectome" / "lesions.json"
    )
    args = parser.parse_args()

    import random
    import tempfile

    from core.connectome.coalition import LESION_PREDICTIONS
    from core.connectome.intervene import silence_all, silenced_calls
    from core.connectome.volume import VolumeReconstructor

    started = time.monotonic()
    reconstructor = VolumeReconstructor(REPO)
    reconstructor.scan()
    snapshot = reconstructor.build()

    def measure(lesion: Sequence[str] | None, seed: int) -> dict[str, Any]:
        """One arm, on a kernel of its own.

        A kernel of its own is the whole of the independence here. The first
        version reused one across every arm, and the workspace readout grew by
        about the same amount under the baseline, all three lesions and all
        three controls — a drift down the run, read as an effect of the cut.
        """
        rows: list[dict[str, float]] = []
        suppressed = 0
        began = time.monotonic()
        for repeat in range(args.repeats):
            rng = random.Random(seed * 1000 + repeat)
            with tempfile.TemporaryDirectory() as raw:
                kernel = _build_kernel(Path(raw))
                if lesion is None:
                    rows.append(asyncio.run(_readout(kernel, rng)))
                else:
                    with silence_all(lesion):
                        rows.append(asyncio.run(_readout(kernel, rng)))
                    suppressed += sum(silenced_calls(one) for one in lesion)
        keys = sorted({key for row in rows for key in row})
        return {
            "readout": {
                key: statistics.fmean([row.get(key, 0.0) for row in rows]) for key in keys
            },
            "spread": {
                key: (
                    statistics.pstdev([row.get(key, 0.0) for row in rows])
                    if len(rows) > 1
                    else 0.0
                )
                for key in keys
            },
            "calls_suppressed": suppressed,
            "seconds": round(time.monotonic() - began, 1),
        }

    report: dict[str, Any] = {"predictions": [], "seconds": 0.0}
    baseline = measure(None, args.seed)
    varying = sum(
        1
        for key, value in baseline["readout"].items()
        if not key.startswith(("spread:", "gap:")) and abs(value) > 1e-9
    )
    report["baseline"] = {
        "readouts": len(baseline["readout"]),
        "non_zero": varying,
        "seconds": baseline["seconds"],
    }
    print(json.dumps({"arm": "baseline", **report["baseline"]}), flush=True)

    used: set[str] = set()
    for offset, prediction in enumerate(LESION_PREDICTIONS, start=1):
        target = STATION_PHASES.get(prediction.station)
        if not target:
            continue
        control = _pick_control(snapshot, target, used)
        used.update(control)
        lesioned = measure(target, args.seed + offset)
        control_arm = measure(control, args.seed + offset)
        keys = sorted(baseline["readout"])
        effects = {
            key: lesioned["readout"].get(key, 0.0) - baseline["readout"][key] for key in keys
        }
        control_effects = {
            key: control_arm["readout"].get(key, 0.0) - baseline["readout"][key]
            for key in keys
        }
        scored = _score(prediction, baseline["readout"], baseline["spread"], effects, control_effects)
        entry = {
            **prediction.as_json(),
            "readout_available": True,
            "target": list(target),
            "control": list(control),
            "target_module_degree": sum(_degree(snapshot, one) for one in target),
            "control_module_degree": sum(_degree(snapshot, one) for one in control),
            "calls_suppressed": lesioned["calls_suppressed"],
            "control_calls_suppressed": control_arm["calls_suppressed"],
            "bit": lesioned["calls_suppressed"] > 0,
            "scored": scored["rows"],
            "verdict": scored["verdict"],
            "also_moved": _what_it_did_move(
                baseline["readout"], baseline["spread"], effects, control_effects
            ),
        }
        report["predictions"].append(entry)
        print(
            json.dumps({"lesion": prediction.name, "verdict": scored["verdict"]}),
            flush=True,
        )
        for row in entry["also_moved"]:
            print(
                f'   moved  {row["readout"]:42s} base={row["baseline"]:>10.4f} '
                f'lesion={row["lesion_effect"]:+9.4f} control={row["control_effect"]:+9.4f}',
                flush=True,
            )
        for row in scored["rows"]:
            print(
                f'   {row["side"]:6s} {row["readout"]:42s} '
                f'base={row["baseline"]:>10.4f} lesion={row["lesion_effect"]:+9.4f} '
                f'control={row["control_effect"]:+9.4f} '
                f'{"held" if row["held"] else ("unusable: " + row["why_not"] if not row["usable"] else "no")}',
                flush=True,
            )

    report["seconds"] = round(time.monotonic() - started, 1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"written to {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
