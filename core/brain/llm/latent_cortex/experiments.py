"""The falsification harness: how we test this without fooling ourselves.

Implements the spec's Experiments 1–5 as runnable, seeded, self-verifying
protocols. Every experiment produces a graded ``Claim``:

    PROVEN      — effect present with non-overlapping Wilson intervals on
                  ≥2 task families and adequate n
    SUPPORTED   — treatment's Wilson lower bound beats control's upper
                  bound on ≥1 family (or a strictly monotone scaling trend)
    CONJECTURE  — insufficient evidence either way (small n, mixed signal)
    REFUTED     — adequate n and treatment failed to beat control

The graders are deliberately conservative; an exciting anecdote grades as
CONJECTURE, never SUPPORTED. Verdicts can be recorded to the Verifier
Foundry so the reliability of this harness itself is tracked like every
other verifier in the system.

Task generators are exact and self-verifying (graph reachability, nested
boolean evaluation, modular-arithmetic chains), each with a controllable
DEPTH knob — the compositional-depth ladder Experiment 2 climbs.

Experiment 6 (frontier comparison) is a protocol, not code that can run
here: it needs blind fresh tasks and an external comparison system under
equal information/tool/compute access. `frontier_comparison_protocol()`
returns the checklist so an operator run can't quietly skip a control.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

# Split out when this module crossed the 2,000-line ceiling. These are
# re-exports, not incidental imports: every caller of
# `latent_cortex.experiments` keeps working, and the grading digest keeps
# covering all three files. Each is named in __all__, which is what stops
# an autofixer deciding a public re-export is an unused import.
from core.brain.llm.latent_cortex.experiment_grading import (
    _MIN_N_FOR_VERDICT,
    CONJECTURE,
    PROVEN,
    REFUTED,
    SUPPORTED,
    ArmResult,
    Claim,
    ExperimentProvenance,
    PairedObservation,
    _coerce_accounted_solver_outcome,
    _coerce_role_outcome,
    _coerce_solver_outcome,
    _holm_adjust,
    experiments_implementation_sha256,
    grade_paired_treatment_vs_control,
    grade_treatment_vs_control,
)
from core.brain.llm.latent_cortex.experiment_tasks import (
    TASK_FAMILIES,
    Task,
    khop_reachability,
    modular_chain,
    nested_boolean,
    task_battery,
)
from core.brain.llm.latent_cortex.experiment_tasks import (
    is_answer_shaped as _is_answer_shaped,
)

logger = logging.getLogger("Aura.LatentCortex.Experiments")

#: Family-wise error rate for factorial attribution, held across every arm the
#: caller chose to test rather than within each arm separately.
_FACTORIAL_ARM_ALPHA = 0.05
# Significance level a slot must clear AFTER correction across every slot
# tested in the same run.
_SLOT_FAMILY_ALPHA = 0.05
# The equal-compute premise for virtual-width claims, defined ONCE so the
# documented tolerance and the tolerance actually graded cannot diverge.
_EQUAL_COMPUTE_TOLERANCE = 0.05

# Reliability weight per verdict tier. A REFUTED or CONJECTURE result must
# NOT feed a positive reliability signal: the old table scored every
# non-PROVEN tier 0.6, so a refutation raised a verifier's score almost as
# much as support did.
_FOUNDRY_TIER_SCORES = {
    PROVEN: 0.9,
    SUPPORTED: 0.7,
    CONJECTURE: 0.3,
    REFUTED: 0.0,
}

# ── Experiment 1: recurrence utility sweep ──────────────────────────────


def run_recurrence_sweep(
    solve: Callable[[Task, int], bool | tuple[bool, int]],
    tasks: list[Task],
    step_grid: list[int],
    *,
    baseline: Callable[[Task], bool | tuple[bool, int]] | None = None,
    provenance: ExperimentProvenance | None = None,
) -> dict[str, Any]:
    """Accuracy as a function of forced recurrence depth.

    ``solve(task, steps)`` runs one latent episode at exactly ``steps``
    recurrent steps and returns verified success. ``baseline(task)`` is the
    equal-FLOP conventional arm (longer CoT / best-of-N), supplied by the
    caller so its compute accounting is visible in the report, not implied.
    """
    if not step_grid or sorted(set(step_grid)) != step_grid or any(step < 1 for step in step_grid):
        raise ValueError("step_grid must be sorted, unique, and positive")
    curve: list[dict[str, Any]] = []
    outcomes_by_step: dict[int, list[tuple[bool, int | None]]] = {}
    for steps in step_grid:
        arm = ArmResult(name=f"steps={steps}")
        for task in tasks:
            success, cost = _coerce_solver_outcome(solve(task, steps))
            arm.n += 1
            arm.successes += int(success)
            arm.layer_apps += int(cost or 0)
            outcomes_by_step.setdefault(steps, []).append((success, cost))
        curve.append(arm.to_dict())
    result: dict[str, Any] = {"curve": curve}
    baseline_outcomes: list[tuple[bool, int | None]] = []
    if baseline is not None:
        base = ArmResult(name="equal_flop_baseline")
        for task in tasks:
            success, cost = _coerce_solver_outcome(baseline(task))
            base.n += 1
            base.successes += int(success)
            base.layer_apps += int(cost or 0)
            baseline_outcomes.append((success, cost))
        result["baseline"] = base.to_dict()
    accs = [c["accuracy"] for c in curve]
    result["monotone_gain"] = len(accs) >= 2 and all(
        b >= a - 1e-9 for a, b in zip(accs, accs[1:], strict=False)
    ) and accs[-1] > accs[0]
    if baseline is None:
        claim = Claim(
            experiment="exp1_recurrence_sweep",
            statement="additional recurrent steps improve equal-compute accuracy",
            tier=CONJECTURE,
            evidence={
                "curve": curve,
                "n_tasks": len(tasks),
                "limitation": "equal-compute baseline missing",
            },
        )
    else:
        deepest = outcomes_by_step[step_grid[-1]]
        paired: dict[str, list[PairedObservation]] = {}
        for index, (task, treatment, control) in enumerate(
            zip(tasks, deepest, baseline_outcomes, strict=True)
        ):
            paired.setdefault(task.family, []).append(
                PairedObservation(
                    task_id=f"{task.family}:{task.depth}:{task.seed}:{index}",
                    family=task.family,
                    treatment_success=treatment[0],
                    control_success=control[0],
                    treatment_layer_apps=treatment[1],
                    control_layer_apps=control[1],
                )
            )
        claim = grade_paired_treatment_vs_control(
            "exp1_recurrence_sweep",
            "additional recurrent steps improve equal-compute accuracy",
            paired,
            provenance=provenance,
        )
        if not result["monotone_gain"] and claim.tier in {PROVEN, SUPPORTED}:
            claim.tier = CONJECTURE
            claim.evidence["voided"] = "deepest arm won but recurrence curve was not monotone"
    result["claim"] = claim.to_dict()
    return result


# ── Experiment 2: depth extrapolation ───────────────────────────────────


def run_depth_extrapolation(
    solve: Callable[[Task, int], bool | tuple[bool, int]],
    family: str,
    depths: list[int],
    step_grid: list[int],
    *,
    per_depth: int = 8,
    seed: int = 0,
    provenance: ExperimentProvenance | None = None,
) -> dict[str, Any]:
    """T_required(depth): the minimum recurrence at which each depth is solved.

    The signature of genuine latent computation is T_required growing with
    problem depth while remaining solvable — compute buys composition."""
    gen = TASK_FAMILIES[family]
    t_required: dict[int, int | None] = {}
    matrix: dict[int, dict[int, float]] = {}
    for depth in depths:
        tasks = [gen(depth, seed * 31 + i) for i in range(per_depth)]
        matrix[depth] = {}
        t_required[depth] = None
        for steps in step_grid:
            wins = sum(
                int(_coerce_solver_outcome(solve(task, steps))[0]) for task in tasks
            )
            acc = wins / len(tasks)
            matrix[depth][steps] = round(acc, 4)
            if acc >= 0.5 and t_required[depth] is None:
                t_required[depth] = steps
    solved = [d for d in depths if t_required[d] is not None]
    pairs = [(d, t_required[d]) for d in solved]
    increasing = all(
        t2 >= t1 for (_, t1), (_, t2) in zip(pairs, pairs[1:], strict=False)
    )
    scaling = len(solved) >= 3 and increasing and len(set(t for _, t in pairs)) > 1
    tier = CONJECTURE if per_depth * len(depths) < _MIN_N_FOR_VERDICT else (
        SUPPORTED if scaling else (REFUTED if len(solved) >= 3 else CONJECTURE)
    )
    return {
        "family": family,
        "matrix": matrix,
        "t_required": t_required,
        "claim": Claim(
            experiment="exp2_depth_extrapolation",
            statement="required recurrence scales with compositional depth",
            tier=tier,
            evidence={"t_required": {str(k): v for k, v in t_required.items()}},
        ).to_dict(),
    }


# ── Experiment 3: slot causality ────────────────────────────────────────


def run_slot_causality(
    solve_with_ablation: Callable[[Task, int | None], bool],
    tasks: list[Task],
    slot_indices: list[int],
    provenance: ExperimentProvenance | None = None,
) -> dict[str, Any]:
    """Ablate slots one at a time; restore must recover performance.

    ``solve_with_ablation(task, slot)`` runs an episode with slot ``slot``
    destroyed pre-persist (None ⇒ intact). Causal workspace ⇒ intact runs
    beat ablated runs, and per-slot damage is measurable."""
    intact = ArmResult(name="intact")
    for task in tasks:
        intact.n += 1
        # STRICT outcome contract: bool() turned any non-empty string or
        # object into a success, so a solver returning an error message
        # scored as a solve.
        intact.successes += int(_coerce_solver_outcome(solve_with_ablation(task, None))[0])
    per_slot: dict[int, ArmResult] = {}
    paired_claims: dict[int, Claim] = {}
    for slot in slot_indices:
        arm = ArmResult(name=f"ablated_slot_{slot}")
        observations: dict[str, list[PairedObservation]] = {}
        for index, task in enumerate(tasks):
            ablated_success, _ = _coerce_solver_outcome(solve_with_ablation(task, slot))
            intact_success, _ = _coerce_solver_outcome(solve_with_ablation(task, None))
            arm.n += 1
            arm.successes += int(ablated_success)
            observations.setdefault(task.family, []).append(
                PairedObservation(
                    task_id=f"{task.family}:{task.depth}:{task.seed}:{index}:slot{slot}",
                    family=task.family,
                    treatment_success=intact_success,
                    control_success=ablated_success,
                )
            )
        per_slot[slot] = arm
        paired_claims[slot] = grade_paired_treatment_vs_control(
            "exp3_slot_causality",
            f"slot {slot} carries causally necessary computation",
            observations,
            require_compute=False,
            provenance=provenance,
        )
    # MULTIPLICITY ACROSS SLOTS: each slot was corrected only WITHIN its own
    # claim, so testing more slots raised the chance that at least one looked
    # causally necessary — and any single pass promoted the top-level claim.
    # Correct the per-slot pooled p-values across the slots actually tested.
    slot_pvalues = {
        str(slot): float(
            claim.evidence.get("pooled", {}).get("one_sided_exact_p", 1.0)
        )
        for slot, claim in paired_claims.items()
    }
    slot_adjusted = _holm_adjust(slot_pvalues) if slot_pvalues else {}
    damaged = [
        slot
        for slot, claim in paired_claims.items()
        if claim.tier in {PROVEN, SUPPORTED}
        and slot_adjusted.get(str(slot), 1.0) < _SLOT_FAMILY_ALPHA
    ]
    uncorrected = [
        slot
        for slot, claim in paired_claims.items()
        if claim.tier in {PROVEN, SUPPORTED} and slot not in damaged
    ]
    tier = CONJECTURE if intact.n < _MIN_N_FOR_VERDICT else (
        SUPPORTED if damaged else REFUTED
    )
    return {
        "intact": intact.to_dict(),
        "ablated": {s: a.to_dict() for s, a in per_slot.items()},
        "causally_necessary_slots": damaged,
        "claim": Claim(
            experiment="exp3_slot_causality",
            statement="thought slots carry causally necessary intermediate computation",
            tier=tier,
            evidence={
                "damaged_slots": damaged,
                "intact_accuracy": intact.accuracy,
                "slots_tested": len(paired_claims),
                "slot_holm_adjusted_p": slot_adjusted,
                "slots_dropped_by_multiplicity": uncorrected,
                # The runner ablates a slot and reruns intact separately; it
                # never restores the SAME episode, so this is necessity
                # evidence, not proof of restoration.
                "restoration_tested": False,
                "compute_matched": False,
            },
        ).to_dict(),
        "paired_slot_claims": {
            slot: claim.to_dict() for slot, claim in paired_claims.items()
        },
    }


# ── Experiment 4: virtual width vs equal-FLOP sampling ──────────────────


def run_virtual_width(
    solve_branches: Callable[
        [Task, int], tuple[bool, int, dict[str, Any], dict[str, Any]]
    ],
    solve_sampling: Callable[
        [Task, int], tuple[bool, int, dict[str, Any], dict[str, Any]]
    ],
    tasks_by_family: dict[str, list[Task]],
    k: int,
    provenance: ExperimentProvenance | None = None,
) -> dict[str, Any]:
    """K latent branches vs K textual samples at (verified-)equal FLOPs.

    Both callbacks return success, admission-layer-apps, a complete resource
    receipt, and an information receipt. The comparison checks structural
    FLOPs plus verifier/tool/external-model counters and exact information
    policy parity; token-layer applications remain a secondary audit field."""
    # K is the experiment's width and appears in every arm name and claim:
    # a bool, zero, negative, or absurd K silently produced degenerate arms.
    if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= 64:
        raise ValueError("virtual-width k must be an int in [1, 64]")
    treatment: dict[str, ArmResult] = {}
    control: dict[str, ArmResult] = {}
    paired: dict[str, list[PairedObservation]] = {}
    for family, tasks in tasks_by_family.items():
        t_arm, c_arm = ArmResult(name=f"branches_k{k}"), ArmResult(name=f"sampling_k{k}")
        for index, task in enumerate(tasks):
            # STRICT: bool()/int() let non-empty strings become successes and
            # truncated fractional costs into apparently valid receipts.
            ok_b, cost_b, resource_b, information_b = (
                _coerce_accounted_solver_outcome(solve_branches(task, k))
            )
            ok_s, cost_s, resource_s, information_s = (
                _coerce_accounted_solver_outcome(solve_sampling(task, k))
            )
            t_arm.n += 1
            t_arm.successes += int(ok_b)
            t_arm.layer_apps += cost_b
            c_arm.n += 1
            c_arm.successes += int(ok_s)
            c_arm.layer_apps += cost_s
            paired.setdefault(family, []).append(
                PairedObservation(
                    task_id=f"{family}:{task.depth}:{task.seed}:{index}",
                    family=family,
                    treatment_success=ok_b,
                    control_success=ok_s,
                    treatment_layer_apps=cost_b,
                    control_layer_apps=cost_s,
                    treatment_resource=resource_b,
                    control_resource=resource_s,
                    treatment_information=information_b,
                    control_information=information_s,
                )
            )
        treatment[family], control[family] = t_arm, c_arm
    claim = grade_paired_treatment_vs_control(
        "exp4_virtual_width",
        "latent branches beat equal-FLOP self-consistency sampling",
        paired,
        # ONE source of truth for the equal-compute premise: the docstring
        # promised 10% while the grader silently applied its 5% default, so
        # reports and operator expectations disagreed with actual behavior.
        compute_tolerance=_EQUAL_COMPUTE_TOLERANCE,
        require_resource_accounting=True,
        provenance=provenance,
    )
    return {
        "treatment": {f: a.to_dict() for f, a in treatment.items()},
        "control": {f: a.to_dict() for f, a in control.items()},
        "claim": claim.to_dict(),
    }


def extract_final_numeric_claim(text: str) -> str:
    """The candidate's final numeric claim, by the SAME rule Task.verify uses.

    Self-consistency voting needs answer extraction that cannot peek at the
    ground truth: the last answer-shaped token wins, hedging loses. This
    shares ``_is_answer_shaped`` with ``Task.verify`` so the two cannot
    drift — an extractor that only saw integers while the verifier accepted
    decimals would vote on a different answer than the one being graded.
    """
    tokens = [t.strip(".,:;!?()[]{}") for t in str(text or "").split()]
    numeric = [token for token in tokens if token and _is_answer_shaped(token)]
    return numeric[-1] if numeric else ""


def majority_answer(answers: list[str]) -> str:
    """The MAJORITY answer, or "" when the sample set does not have one.

    A tie is the absence of a majority, not a decision. Breaking ties
    lexicographically manufactured a definite answer from an undecided
    sample — which could be graded correct by luck of alphabetical order and
    inflate the self-consistency baseline this helper feeds.
    """
    filtered = [answer for answer in answers if answer]
    if not filtered:
        return ""
    counts: dict[str, int] = {}
    for answer in filtered:
        counts[answer] = counts.get(answer, 0) + 1
    top = max(counts.values())
    winners = [answer for answer, count in counts.items() if count == top]
    return winners[0] if len(winners) == 1 else ""


# ── Factorial ablations: which mechanism carries any gain ───────────────

FACTORIAL_ARMS: tuple[str, ...] = (
    "recurrence_only",
    "branches_only",
    "latent_opt_only",
    "fast_weights_only",
    "recurrence_branches",
    "recurrence_verifier",
    "full_stack",
)


def run_factorial_ablations(
    solve_arm: Callable[[Task, str], tuple[bool, int]],
    tasks_by_family: dict[str, list[Task]],
    *,
    arms: tuple[str, ...] = FACTORIAL_ARMS,
    journal_path: str | Path | None = None,
    provenance: ExperimentProvenance | None = None,
) -> dict[str, Any]:
    """Attribute any gain to a mechanism: every arm paired against vanilla.

    ``solve_arm(task, arm)`` runs one configuration ("vanilla" is the
    ordinary-decoding control; the treatment arms enable one mechanism or a
    named combination). Each arm earns its own paired claim vs vanilla on
    the SAME tasks, so "the full stack helps" can be decomposed into which
    ingredient actually carried the effect — the RSL gap analysis's
    mechanism-attribution obligation.

    CP126 51654706. This is the longest runner (arms x families x tasks) and
    it kept every accumulator in memory, returning only after the final
    callback. A crash discarded hours of completed trials, and the rerun
    could differ because the callbacks carry order-sensitive state. Passing
    ``journal_path`` makes each trial durable the moment it completes: a
    resumed run attaches only to the same manifest, skips exactly what it
    already did, and records a failing trial as a failure receipt instead of
    letting one exception destroy the completed work beside it."""
    arm_names = ("vanilla", *arms)
    results: dict[str, dict[str, ArmResult]] = {
        arm: {family: ArmResult(name=arm) for family in tasks_by_family}
        for arm in arm_names
    }
    outcomes: dict[str, dict[str, list[tuple[bool, int]]]] = {
        arm: {family: [] for family in tasks_by_family} for arm in arm_names
    }
    journal = None
    if journal_path is not None:
        from core.brain.llm.latent_cortex.trial_journal import TrialJournal

        journal = TrialJournal(
            journal_path,
            manifest={
                "runner": "run_factorial_ablations",
                "arms": list(arm_names),
                "families": {
                    family: [
                        f"{task.depth}:{task.seed}" for task in tasks
                    ]
                    for family, tasks in tasks_by_family.items()
                },
            },
        ).open()

    for family, tasks in tasks_by_family.items():
        for index, task in enumerate(tasks):
            for arm in arm_names:
                # CP126 78632859. This unpacked the solver's return directly
                # and coerced it with bool()/int(): a non-empty string became
                # a SUCCESS, a fractional cost was truncated into evidence,
                # and a negative cost could reach the report. The strict
                # contract every other runner uses rejects those instead of
                # laundering them into results.
                if journal is None:
                    success, cost = _coerce_solver_outcome(solve_arm(task, arm))
                else:
                    key = f"{arm}:{family}:{index}:{task.depth}:{task.seed}"
                    record = journal.run_trial(
                        key,
                        lambda task=task, arm=arm: dict(
                            zip(
                                ("success", "cost"),
                                _coerce_solver_outcome(solve_arm(task, arm)),
                                strict=True,
                            )
                        ),
                    )
                    if not record.ok:
                        # A trial that could not produce evidence must not be
                        # counted as evidence. It stays in the journal as an
                        # explicit failure and is excluded from the claim.
                        raise ValueError(
                            f"factorial_trial_failed:{key}:{record.error}"
                        )
                    success = bool(record.payload.get("success"))
                    cost = record.payload.get("cost")
                row = results[arm][family]
                row.n += 1
                row.successes += int(success)
                row.layer_apps += int(cost or 0)
                outcomes[arm][family].append((success, int(cost or 0)))
    claims: dict[str, dict[str, Any]] = {}
    for arm in arms:
        paired: dict[str, list[PairedObservation]] = {}
        for family, tasks in tasks_by_family.items():
            for index, task in enumerate(tasks):
                treatment = outcomes[arm][family][index]
                control = outcomes["vanilla"][family][index]
                paired.setdefault(family, []).append(
                    PairedObservation(
                        task_id=f"{family}:{task.depth}:{task.seed}:{index}",
                        family=family,
                        treatment_success=treatment[0],
                        control_success=control[0],
                        treatment_layer_apps=treatment[1],
                        control_layer_apps=control[1],
                    )
                )
        claims[arm] = grade_paired_treatment_vs_control(
            f"ablation_{arm}",
            f"mechanism arm '{arm}' beats vanilla decoding on the same tasks",
            paired,
            # Mechanism arms intentionally spend different compute than
            # vanilla — attribution is about direction, not FLOP parity;
            # Experiments 1/4 own the equal-compute claims.
            require_compute=False,
            provenance=provenance,
        ).to_dict()
    # CP126 ba3ffbac: each arm was graded on its own and any arm reaching
    # PROVEN or SUPPORTED joined the attribution list. Holm ran INSIDE a claim,
    # across that arm's families, and never across the arms themselves — so
    # testing seven mechanisms at alpha 0.05 gave roughly a one-in-three chance
    # of attributing a gain to a mechanism that did nothing, and the caller
    # chooses the arm tuple, so the error rate is the caller's to inflate.
    #
    # An arm's evidence is its strongest family, so its arm-level p is the
    # smallest within-claim adjusted p it produced. Holm across those controls
    # the family-wise error over the arms actually tested.
    arm_pvalues: dict[str, float] = {}
    for arm in arms:
        within = claims[arm]["evidence"].get("holm_adjusted_p") or {}
        finite = [
            float(value)
            for value in within.values()
            if isinstance(value, (int, float)) and math.isfinite(float(value))
        ]
        arm_pvalues[arm] = min(finite) if finite else 1.0
    arm_adjusted = _holm_adjust(arm_pvalues)
    attribution = [
        arm
        for arm in arms
        if claims[arm]["tier"] in {PROVEN, SUPPORTED}
        and arm_adjusted.get(arm, 1.0) < _FACTORIAL_ARM_ALPHA
    ]
    return {
        "arms": {
            arm: {family: row.to_dict() for family, row in families.items()}
            for arm, families in results.items()
        },
        "claims": claims,
        "attribution": attribution,
        # What attribution survived correction, and what it was corrected
        # against. A reader who only sees the surviving list cannot tell a
        # two-arm study from a twenty-arm one.
        "arm_holm_adjusted_p": {arm: round(value, 9) for arm, value in arm_adjusted.items()},
        "arm_family_wise_alpha": _FACTORIAL_ARM_ALPHA,
        "arms_tested": len(arms),
        "attribution_before_arm_correction": [
            arm for arm in arms if claims[arm]["tier"] in {PROVEN, SUPPORTED}
        ],
    }


# ── Experiment 5: latent optimization vs random control ─────────────────


def _latent_opt_arm_order(family: str, task: Task, index: int) -> tuple[str, str, str]:
    commitment = f"latent-opt-order-v1:{family}:{task.depth}:{task.seed}:{index}".encode()
    digest = hashlib.sha256(commitment).digest()
    family_offset = hashlib.sha256(f"{family}:latent-opt-order-v1".encode()).digest()[0] & 1
    gradient_first = (index + family_offset) % 2 == 0
    pair = ["gradient", "control"] if gradient_first else ["control", "gradient"]
    pair.insert((index + digest[1]) % 3, "off")
    return pair[0], pair[1], pair[2]


def run_latent_opt_control(
    solve_arm: Callable[[Task, str], bool | tuple[bool, int]],
    tasks_by_family: dict[str, list[Task]],
    provenance: ExperimentProvenance | None = None,
) -> dict[str, Any]:
    """Arms: 'off', 'gradient', 'control' (matched-magnitude random).

    The claim is only about DIRECTION: gradient must beat the random control,
    not merely beat doing nothing. That is the spec's essential control."""
    arms = ("off", "gradient", "control")
    results: dict[str, dict[str, ArmResult]] = {a: {} for a in arms}
    per_task: dict[str, dict[str, list[tuple[bool, int | None]]]] = {
        arm: {} for arm in arms
    }
    execution_order: list[dict[str, Any]] = []
    for family, tasks in tasks_by_family.items():
        family_results = {arm: ArmResult(name=arm) for arm in arms}
        for index, task in enumerate(tasks):
            order = _latent_opt_arm_order(family, task, index)
            task_id = f"{family}:{task.depth}:{task.seed}:{index}"
            execution_order.append({"task_id": task_id, "arms": list(order)})
            for arm in order:
                success, cost = _coerce_solver_outcome(solve_arm(task, arm))
                r = family_results[arm]
                r.n += 1
                r.successes += int(success)
                r.layer_apps += int(cost or 0)
                per_task[arm].setdefault(family, []).append((success, cost))
        for arm, result in family_results.items():
            results[arm][family] = result
    paired: dict[str, list[PairedObservation]] = {}
    for family, tasks in tasks_by_family.items():
        for index, task in enumerate(tasks):
            gradient = per_task["gradient"][family][index]
            control = per_task["control"][family][index]
            paired.setdefault(family, []).append(
                PairedObservation(
                    task_id=f"{family}:{task.depth}:{task.seed}:{index}",
                    family=family,
                    treatment_success=gradient[0],
                    control_success=control[0],
                    treatment_layer_apps=gradient[1],
                    control_layer_apps=control[1],
                )
            )
    claim = grade_paired_treatment_vs_control(
        "exp5_latent_opt",
        "gradient direction (not mere perturbation) improves outcomes",
        paired,
        provenance=provenance,
    )
    return {
        "arms": {a: {f: r.to_dict() for f, r in fam.items()} for a, fam in results.items()},
        "execution_order": execution_order,
        "claim": claim.to_dict(),
    }


# ── SPARK-070 row: fast-weight on/off/sham controls ─────────────────────


def _fast_weight_arm_order(family: str, task: Task, index: int) -> tuple[str, str, str]:
    """Counterbalanced arm order, committed to the task identity.

    Same discipline as the latent-opt control: a fixed order lets ordering
    effects (cache warmth, host thermal state) load onto one arm.
    """
    commitment = f"fast-weight-order-v1:{family}:{task.depth}:{task.seed}:{index}".encode()
    digest = hashlib.sha256(commitment).digest()
    family_offset = hashlib.sha256(f"{family}:fast-weight-order-v1".encode()).digest()[0] & 1
    on_first = (index + family_offset) % 2 == 0
    pair = ["on", "sham"] if on_first else ["sham", "on"]
    pair.insert((index + digest[1]) % 3, "off")
    return pair[0], pair[1], pair[2]


def _coerce_fast_weight_outcome(value: Any) -> tuple[bool, int | None, bool | None]:
    """(success, layer_apps, erased) — erasure is part of the observation.

    ``erased`` is None only for the ``off`` arm, which applies no delta and so
    has nothing to erase. For ``on`` and ``sham`` a None means the erase was
    not proven, which is NOT the same as proven-false and is not the same as
    proven-true: the observation is quarantined rather than counted either way.
    """
    if isinstance(value, tuple) and len(value) == 3:
        success, layer_apps, erased = value
        if not isinstance(success, bool):
            raise ValueError("fast-weight solver success must be boolean")
        if layer_apps is not None and (
            type(layer_apps) is not int or layer_apps < 0
        ):
            raise ValueError("fast-weight layer-app receipt must be a non-negative integer")
        if erased is not None and not isinstance(erased, bool):
            raise ValueError("fast-weight erasure evidence must be boolean or None")
        return success, layer_apps, erased
    raise ValueError(
        "fast-weight solver must return (success, layer_apps, erased)"
    )


def run_fast_weight_controls(
    solve_arm: Callable[[Task, str], tuple[bool, int | None, bool | None]],
    tasks_by_family: dict[str, list[Task]],
    provenance: ExperimentProvenance | None = None,
) -> dict[str, Any]:
    """Arms: 'off', 'on' (optimized delta), 'sham' (matched-magnitude random).

    The SPARK-070 row asks for fast-weight on/off/sham arms *with erasure
    proofs*, and both halves of that carry weight.

    **Why the graded claim is on-versus-sham, not on-versus-off.** An
    optimized delta beating no delta at all says only that perturbing the
    function of this magnitude helped; it cannot separate learned adaptation
    from the regularizing effect of any bounded change. The sham arm applies a
    delta of matched magnitude in a random direction, so the claim that
    survives is about DIRECTION. The 'off' arm is still run and reported,
    because a case where sham also beats off is worth seeing — but it is not
    what the claim rests on.

    **Why erasure gates the observation.** Query-scoped fast weights are only
    safe because they are erased afterwards. A task whose erase was not proven
    may have contaminated every task after it, so its observation is
    quarantined instead of counted — an unproven erase is not a passed check.
    A run with any refuted erase is reported as integrity-failed outright,
    because at that point the arm ordering no longer isolates anything.
    """
    arms = ("off", "on", "sham")
    results: dict[str, dict[str, ArmResult]] = {arm: {} for arm in arms}
    per_task: dict[str, dict[str, list[tuple[bool, int | None]]]] = {
        arm: {} for arm in arms
    }
    execution_order: list[dict[str, Any]] = []
    quarantined: list[dict[str, Any]] = []
    refuted: list[dict[str, Any]] = []
    erase_proven = 0
    erase_expected = 0

    for family, tasks in tasks_by_family.items():
        family_results = {arm: ArmResult(name=arm) for arm in arms}
        for index, task in enumerate(tasks):
            order = _fast_weight_arm_order(family, task, index)
            task_id = f"{family}:{task.depth}:{task.seed}:{index}"
            execution_order.append({"task_id": task_id, "arms": list(order)})
            observations: dict[str, tuple[bool, int | None]] = {}
            task_clean = True
            for arm in order:
                success, cost, erased = _coerce_fast_weight_outcome(
                    solve_arm(task, arm)
                )
                if arm != "off":
                    erase_expected += 1
                    if erased is True:
                        erase_proven += 1
                    elif erased is False:
                        refuted.append({"task_id": task_id, "arm": arm})
                        task_clean = False
                    else:
                        quarantined.append(
                            {"task_id": task_id, "arm": arm, "reason": "erase_unproven"}
                        )
                        task_clean = False
                elif erased is not None:
                    raise ValueError("the off arm applies no delta to erase")
                observations[arm] = (success, cost)
            if not task_clean:
                # The whole task is dropped, not just the offending arm: a
                # partially counted task silently unbalances the pairing.
                continue
            for arm, (success, cost) in observations.items():
                result = family_results[arm]
                result.n += 1
                result.successes += int(success)
                result.layer_apps += int(cost or 0)
                per_task[arm].setdefault(family, []).append((success, cost))
        for arm, result in family_results.items():
            results[arm][family] = result

    paired: dict[str, list[PairedObservation]] = {}
    for family in tasks_by_family:
        treatment_rows = per_task["on"].get(family, [])
        control_rows = per_task["sham"].get(family, [])
        for index, (treatment, control) in enumerate(
            zip(treatment_rows, control_rows, strict=True)
        ):
            paired.setdefault(family, []).append(
                PairedObservation(
                    task_id=f"{family}:fast-weight:{index}",
                    family=family,
                    treatment_success=treatment[0],
                    control_success=control[0],
                    treatment_layer_apps=treatment[1],
                    control_layer_apps=control[1],
                )
            )
    claim = grade_paired_treatment_vs_control(
        "spark070_fast_weight_controls",
        "an optimized episodic delta beats a matched-magnitude random delta",
        paired,
        provenance=provenance,
    )
    integrity = {
        "erase_expected": erase_expected,
        "erase_proven": erase_proven,
        "erase_refuted": len(refuted),
        "quarantined_observations": len(quarantined),
        "refuted_rows": refuted,
        "quarantined_rows": quarantined,
        # Every adapted arm proved its erase, and none was refuted. Anything
        # less and the run cannot claim its arms were isolated from each other.
        "integrity_proven": (
            erase_expected > 0
            and erase_proven == erase_expected
            and not refuted
            and not quarantined
        ),
    }
    return {
        "arms": {
            arm: {family: result.to_dict() for family, result in families.items()}
            for arm, families in results.items()
        },
        "execution_order": execution_order,
        "erasure_integrity": integrity,
        "claim": (
            claim.to_dict()
            if integrity["integrity_proven"]
            else {
                **claim.to_dict(),
                "tier": "REFUTED_INTEGRITY",
                "reason": (
                    "fast-weight erasure was not proven for every adapted arm; "
                    "no capability claim can rest on possibly-contaminated tasks"
                ),
            }
        ),
    }


# ── Experiment 6: frontier comparison protocol ──────────────────────────


def frontier_comparison_protocol() -> dict[str, Any]:
    """The operator checklist for the only claim that finally counts."""
    return {
        "preconditions": [
            "architecture and schedule library FROZEN before task generation",
            "fresh blind tasks generated after freeze (no benchmark reuse)",
            "checkpoint SHA recorded and republished with results",
        ],
        "controls": [
            "equal problem information for both systems",
            "equal tool and verification access",
            "equal-latency AND equal-compute result columns",
            "no benchmark-specific answer caches",
        ],
        "domains": [
            "novel algorithmic reasoning",
            "mathematics",
            "coding",
            "scientific inference",
            "long-horizon planning",
            "calibration",
            "robustness to misleading premises",
        ],
        "report": "publish per-domain Wilson intervals; the weakest domain is the headline",
    }


# ── Foundry recording ───────────────────────────────────────────────────


def _record_foundry_refusal(reason: str) -> None:
    """A refused verdict is a visible event, never a silent drop."""
    from core.runtime.errors import record_degradation

    record_degradation(
        "latent_cortex",
        ValueError(f"foundry_claim_refused:{reason}"),
        severity="warning",
        action="refused to record an unvalidated claim into the reliability ledger",
    )


def record_claim_to_foundry(claim: Claim | dict[str, Any], domain: str) -> bool:
    """Log an experiment verdict into the Verifier Foundry reliability ledger.

    ADMISSION: only a verdict this module actually graded may enter the
    reliability ledger. The function previously accepted any mapping, trusted
    a caller-supplied tier string, and submitted ``checked=True``
    unconditionally — so any caller could inject a SUPPORTED/PROVEN verdict
    and raise a verifier's measured reliability without running anything.
    """
    if isinstance(claim, Claim):
        body = claim.to_dict()
    elif isinstance(claim, dict):
        body = dict(claim)
    else:
        _record_foundry_refusal(f"claim_type_invalid:{type(claim).__name__}")
        return False

    tier = body.get("tier")
    if tier not in _FOUNDRY_TIER_SCORES:
        _record_foundry_refusal(f"unknown_tier:{str(tier)[:40]}")
        return False
    experiment = str(body.get("experiment") or "").strip()
    statement = str(body.get("statement") or "").strip()
    if not experiment or not statement:
        _record_foundry_refusal("claim_missing_experiment_or_statement")
        return False
    # A verdict without graded evidence is not a measurement. ``checked``
    # reports whether this claim was actually adjudicated against data.
    evidence = body.get("evidence")
    checked = isinstance(evidence, dict) and bool(evidence)
    if not checked:
        _record_foundry_refusal(f"claim_without_evidence:{experiment[:60]}")
        return False
    if not isinstance(domain, str) or not domain.strip():
        _record_foundry_refusal("domain_invalid")
        return False

    try:
        from core.brain.verifiers.foundry import get_verifier_foundry

        foundry = get_verifier_foundry()
        verdict_id = foundry.record_verdict(
            verifier=f"latent_cortex.{experiment}",
            domain=domain,
            hard_pass=tier in (PROVEN, SUPPORTED),
            score=_FOUNDRY_TIER_SCORES[tier],
            checked=checked,
            meta={"statement": statement, "tier": tier},
        )
        return bool(verdict_id)
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        from core.runtime.errors import record_degradation

        record_degradation(
            "latent_cortex",
            exc,
            action="kept experiment claim local after foundry recording failed",
        )
        return False


# ── Experiment R: are role anchors causal cognitive labor? ──────────────

ROLE_ARMS: tuple[str, ...] = (
    "distinct_roles",
    "lesioned_uniform_role",
    "swapped_roles",
    "restored_roles",
)


def run_role_lesion(
    solve_arm: Callable[[Task, str], tuple[bool, int, float | None]],
    tasks_by_family: dict[str, list[Task]],
    *,
    divergence_margin: float = 0.02,
    provenance: ExperimentProvenance | None = None,
) -> dict[str, Any]:
    """Lesion/swap the branch role anchors and measure what they carry.

    ``solve_arm(task, arm)`` runs one arm and returns
    (success, layer_apps, branch_divergence) where branch_divergence is
    1 − mean pairwise branch-summary cosine at exchanges (``None`` when the
    episode had no exchange telemetry). Arms:

    - distinct_roles: the default role rotation (treatment);
    - lesioned_uniform_role: every branch gets the SAME anchor — role
      diversity removed, everything else identical;
    - swapped_roles: the same distinct anchors, permuted across branch
      indices — if roles are causal, outcomes should track the anchors,
      not the branch index.
    - restored_roles: the original distinct assignment reinstated after the
      lesion, so an apparent effect must recover instead of merely drift.

    Claims: a paired behavioral claim (distinct vs lesioned), a
    mechanistic divergence claim (distinct trajectories diverge more than
    lesioned ones by at least ``divergence_margin``), and a swap-parity
    observation (swapped ≈ distinct implies anchor-causality, not
    index-causality), and a restoration claim. All behavioral comparisons
    require exact measured layer-app parity. Divergence claims cap at
    SUPPORTED: internal geometry cannot earn PROVEN.
    """
    if (
        isinstance(divergence_margin, bool)
        or not isinstance(divergence_margin, (int, float))
        or not math.isfinite(float(divergence_margin))
        or not 0.0 <= float(divergence_margin) < 1.0
    ):
        raise ValueError("divergence_margin must be a finite number in [0, 1)")
    outcomes: dict[str, dict[str, list[tuple[bool, int, float | None]]]] = {
        arm: {} for arm in ROLE_ARMS
    }
    for arm in ROLE_ARMS:
        for family, tasks in tasks_by_family.items():
            rows = outcomes[arm].setdefault(family, [])
            for task in tasks:
                # CP126 78632859. bool()/int()/float() on a solver's return
                # accepts almost anything: a non-empty string is a success, a
                # fractional cost truncates, and an arbitrary or non-finite
                # divergence reaches the report as evidence.
                ok, cost, divergence = _coerce_role_outcome(solve_arm(task, arm))
                rows.append((ok, cost, divergence))

    paired: dict[str, list[PairedObservation]] = {}
    for family, tasks in tasks_by_family.items():
        for index, task in enumerate(tasks):
            treatment = outcomes["distinct_roles"][family][index]
            control = outcomes["lesioned_uniform_role"][family][index]
            paired.setdefault(family, []).append(
                PairedObservation(
                    task_id=f"{family}:{task.depth}:{task.seed}:{index}",
                    family=family,
                    treatment_success=treatment[0],
                    control_success=control[0],
                    treatment_layer_apps=treatment[1],
                    control_layer_apps=control[1],
                )
            )
    behavioral = grade_paired_treatment_vs_control(
        "expR_role_diversity",
        "distinct role anchors beat a lesioned uniform-role ensemble",
        paired,
        require_compute=True,
        provenance=provenance,
    )

    restored_paired: dict[str, list[PairedObservation]] = {}
    for family, tasks in tasks_by_family.items():
        for index, task in enumerate(tasks):
            treatment = outcomes["restored_roles"][family][index]
            control = outcomes["lesioned_uniform_role"][family][index]
            restored_paired.setdefault(family, []).append(
                PairedObservation(
                    task_id=f"{family}:{task.depth}:{task.seed}:{index}",
                    family=family,
                    treatment_success=treatment[0],
                    control_success=control[0],
                    treatment_layer_apps=treatment[1],
                    control_layer_apps=control[1],
                )
            )
    restoration = grade_paired_treatment_vs_control(
        "expR_role_restoration",
        "restoring distinct role anchors recovers the lesioned capability",
        restored_paired,
        require_compute=True,
        provenance=provenance,
    )

    def _mean_divergence(arm: str) -> tuple[float | None, int]:
        values = [
            divergence
            for rows in outcomes[arm].values()
            for _, _, divergence in rows
            if divergence is not None
        ]
        if not values:
            return None, 0
        return sum(values) / len(values), len(values)

    distinct_div, distinct_n = _mean_divergence("distinct_roles")
    lesioned_div, lesioned_n = _mean_divergence("lesioned_uniform_role")
    swapped_div, swapped_n = _mean_divergence("swapped_roles")
    restored_div, restored_n = _mean_divergence("restored_roles")
    divergence_evidence = {
        "distinct_mean_divergence": distinct_div,
        "lesioned_mean_divergence": lesioned_div,
        "swapped_mean_divergence": swapped_div,
        "restored_mean_divergence": restored_div,
        "samples": {
            "distinct_roles": distinct_n,
            "lesioned_uniform_role": lesioned_n,
            "swapped_roles": swapped_n,
            "restored_roles": restored_n,
        },
        "divergence_margin": float(divergence_margin),
        "limitation": (
            "internal trajectory geometry; decorrelation jitter fires on "
            "near-collapse ensembles and partially masks lesioning"
        ),
    }
    enough = min(distinct_n, lesioned_n) >= _MIN_N_FOR_VERDICT
    if not enough or distinct_div is None or lesioned_div is None:
        divergence_tier = CONJECTURE
    elif distinct_div - lesioned_div >= float(divergence_margin):
        divergence_tier = SUPPORTED
    elif lesioned_div >= distinct_div:
        divergence_tier = REFUTED
    else:
        divergence_tier = CONJECTURE
    mechanistic = Claim(
        experiment="expR_role_divergence",
        statement=(
            "distinct role anchors produce more divergent branch "
            "trajectories than a lesioned uniform-role ensemble"
        ),
        tier=divergence_tier,
        evidence=divergence_evidence,
    )

    swap_parity: dict[str, Any] = {
        "note": (
            "swapped ≈ distinct on both accuracy and divergence implies the "
            "ANCHOR, not the branch index, carries the role"
        ),
        "accuracy_tolerance": 0.05,
        "all_families_within_tolerance": True,
    }
    for family in tasks_by_family:
        distinct_acc = sum(
            1 for ok, _, _ in outcomes["distinct_roles"][family] if ok
        ) / max(1, len(outcomes["distinct_roles"][family]))
        swapped_acc = sum(
            1 for ok, _, _ in outcomes["swapped_roles"][family] if ok
        ) / max(1, len(outcomes["swapped_roles"][family]))
        task_compute_matched = all(
            outcomes["distinct_roles"][family][index][1]
            == outcomes["swapped_roles"][family][index][1]
            for index in range(len(outcomes["distinct_roles"][family]))
        )
        within_tolerance = abs(distinct_acc - swapped_acc) <= 0.05
        swap_parity["all_families_within_tolerance"] = bool(
            swap_parity["all_families_within_tolerance"]
            and within_tolerance
            and task_compute_matched
        )
        swap_parity[family] = {
            "distinct_accuracy": round(distinct_acc, 4),
            "swapped_accuracy": round(swapped_acc, 4),
            "accuracy_delta": round(swapped_acc - distinct_acc, 4),
            "within_tolerance": within_tolerance,
            "task_compute_matched": task_compute_matched,
        }

    task_identities = [
        f"{family}:{task.depth}:{task.seed}:{index}"
        for family, tasks in sorted(tasks_by_family.items())
        for index, task in enumerate(tasks)
    ]
    task_set_sha256 = hashlib.sha256(
        json.dumps(task_identities, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    compute_parity = all(
        len(
            {
                outcomes[arm][family][index][1]
                for arm in ROLE_ARMS
            }
        )
        == 1
        for family, tasks in tasks_by_family.items()
        for index in range(len(tasks))
    )
    supported_tiers = {PROVEN, SUPPORTED}
    causal_supported = bool(
        behavioral.tier in supported_tiers
        and restoration.tier in supported_tiers
        and swap_parity["all_families_within_tolerance"] is True
        and compute_parity
    )
    role_causality = {
        "tier": SUPPORTED if causal_supported else CONJECTURE,
        "task_set_sha256": task_set_sha256,
        "task_count": len(task_identities),
        "compute_parity": compute_parity,
        "lesion_effect_supported": behavioral.tier in supported_tiers,
        "restoration_supported": restoration.tier in supported_tiers,
        "swap_follows_roles_not_indices": swap_parity[
            "all_families_within_tolerance"
        ],
        "limitation": (
            "supports differentiated role labor on this checked task set; "
            "does not establish universal task benefit or frontier capability"
        ),
    }

    return {
        "arms": {
            arm: {
                family: {
                    "n": len(rows),
                    "successes": sum(1 for ok, _, _ in rows if ok),
                    "layer_apps": sum(cost for _, cost, _ in rows),
                }
                for family, rows in by_family.items()
            }
            for arm, by_family in outcomes.items()
        },
        "behavioral_claim": behavioral.to_dict(),
        "restoration_claim": restoration.to_dict(),
        "divergence_claim": mechanistic.to_dict(),
        "swap_parity": swap_parity,
        "role_causality": role_causality,
    }


__all__ = [
    "ExperimentProvenance",
    "experiments_implementation_sha256",
    "ROLE_ARMS",
    "run_role_lesion",
    "ArmResult",
    "CONJECTURE",
    "Claim",
    "PairedObservation",
    "PROVEN",
    "REFUTED",
    "SUPPORTED",
    "TASK_FAMILIES",
    "Task",
    "FACTORIAL_ARMS",
    "extract_final_numeric_claim",
    "frontier_comparison_protocol",
    "grade_treatment_vs_control",
    "grade_paired_treatment_vs_control",
    "khop_reachability",
    "majority_answer",
    "modular_chain",
    "nested_boolean",
    "record_claim_to_foundry",
    "run_depth_extrapolation",
    "run_factorial_ablations",
    "run_fast_weight_controls",
    "run_latent_opt_control",
    "run_recurrence_sweep",
    "run_slot_causality",
    "run_virtual_width",
    "task_battery",
]
