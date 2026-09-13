"""core/learning/compiler_free_experiment.py — the held-out family, with controls.

The experiment the external review named as decisive:

    Train on several families, then give a genuinely new held-out family for
    which there is no P_k compiler. Give only a generic task interface and
    success/failure feedback. If it autonomously induces a representation and
    reusable procedure that solves the family, freezes it, transfers it across
    unseen examples, and causal lesions destroy the acquired gain, that is
    learned procedure acquisition rather than specialised computation.

Every clause is a phase here, and each has a way to fail.

    induce      `ProcedureInducer` sees input/output pairs and nothing else.
                No family label reaches it, and no per-family code exists.
    freeze      the program is hashed; every later phase runs that hash.
    transfer    the frozen program is scored on instances drawn with a
                different seed, which it never saw.
    lesion      two arms. The first restricts the same search to depth 1, so
                the only thing removed is composition. The second removes the
                induced procedure entirely and predicts the most common
                training output. The gain must sit above both.
    null        the whole search, on the same inputs with outputs permuted.
                It must find nothing, or a success means only that the
                searcher can fit twelve pairs.
    shortcut    a family a single primitive solves is a compiled family, and
                the run refuses rather than reporting a win.

The train families are not there to warm anything up — nothing is carried
between them. They are there to show the same inducer, unchanged, solves
several families it was not written for, so the held-out result is not one
lucky primitive set.

Boundary, stated plainly: the primitives are given and so are the value types.
This is procedure acquisition, not the invention of a representational
vocabulary. `OntologyGenesis` takes the neighbouring step of naming a new
predicate; neither is open-ended discovery of the primitive set itself.
"""

from __future__ import annotations

import collections
import logging
import random
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from core.learning.procedure_induction import (
    PRIMITIVE_SET_SHA,
    InductionOutcome,
    ProcedureInducer,
    Program,
    TaskFamily,
    TaskInstance,
    accuracy,
)

logger = logging.getLogger("Aura.CompilerFree")

#: Support pairs the inducer is shown. Small on purpose: a procedure that needs
#: hundreds of examples to be pinned down is being fitted, not induced.
SUPPORT_SIZE = 12

#: Fresh instances the frozen program is transferred to.
TRANSFER_SIZE = 200

#: Shuffled-output runs. Each is a full search; if any finds a program, the
#: searcher fits noise and no positive result from it means anything.
NULL_RUNS = 15


def _list_family(family_id: str, fn) -> TaskFamily:
    def generate(rng: random.Random) -> TaskInstance:
        values = tuple(rng.randint(1, 30) for _ in range(rng.randint(3, 6)))
        return TaskInstance(inputs=(values,), output=fn(values))

    return TaskFamily(family_id, generate)


#: Families are generators. Nothing here maps a family to its program, and the
#: inducer is never told which family an instance came from.
TRAIN_FAMILIES: tuple[TaskFamily, ...] = (
    _list_family("range_of_values", lambda v: max(v) - min(v)),
    _list_family("sum_of_extremes", lambda v: max(v) + min(v)),
    _list_family("second_largest", lambda v: sorted(v)[-2]),
    _list_family("count_times_smallest", lambda v: len(v) * min(v)),
)

#: The held-out family. Solvable only by composing three primitives, which
#: `single_primitive_shortcut` re-checks rather than assumes.
HELDOUT_FAMILY: TaskFamily = _list_family(
    "total_without_largest", lambda v: sum(v) - max(v)
)


@dataclass(frozen=True)
class ArmResult:
    name: str
    accuracy: float
    detail: str = ""


@dataclass(frozen=True)
class CompilerFreeResult:
    heldout_family: str
    primitive_set_sha: str
    support_size: int
    transfer_size: int
    induced: dict[str, Any] | None
    transfer_accuracy: float
    lesions: list[ArmResult]
    null_runs: int
    null_found: int
    single_primitive_shortcut: bool
    train_families_solved: dict[str, str]
    train_families_total: int
    verdict: str
    reasons: list[str] = field(default_factory=list)
    ran_at: float = field(default_factory=lambda: time.time())

    @property
    def best_lesion(self) -> float:
        return max((arm.accuracy for arm in self.lesions), default=0.0)

    @property
    def gain(self) -> float:
        return round(self.transfer_accuracy - self.best_lesion, 6)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["best_lesion_accuracy"] = self.best_lesion
        payload["gain_over_best_lesion"] = self.gain
        return payload


VERDICT_ACQUIRED = "PROCEDURE_ACQUIRED"
VERDICT_NOT_ACQUIRED = "PROCEDURE_NOT_ACQUIRED"
VERDICT_VOID = "RUN_VOID"


def _most_common_output(support: Sequence[TaskInstance]) -> Any:
    counter = collections.Counter(repr(item.output) for item in support)
    winner = counter.most_common(1)[0][0]
    for item in support:
        if repr(item.output) == winner:
            return item.output
    return None


def _constant_accuracy(value: Any, instances: Sequence[TaskInstance]) -> float:
    if not instances:
        return 0.0
    return sum(1 for item in instances if item.output == value) / len(instances)


def run_compiler_free_experiment(
    *,
    heldout: TaskFamily | None = None,
    train_families: Sequence[TaskFamily] = TRAIN_FAMILIES,
    support_size: int = SUPPORT_SIZE,
    transfer_size: int = TRANSFER_SIZE,
    null_runs: int = NULL_RUNS,
    max_depth: int = 3,
    seed: int = 20260816,
) -> CompilerFreeResult:
    """Run every phase, and return what happened including the failures."""
    heldout = heldout or HELDOUT_FAMILY
    inducer = ProcedureInducer(max_depth=max_depth)
    reasons: list[str] = []

    # The same inducer, unchanged, against families it was not written for.
    solved: dict[str, str] = {}
    for index, family in enumerate(train_families):
        outcome = inducer.induce(family.sample(support_size, seed=seed + index))
        if outcome.found and outcome.program is not None:
            solved[family.family_id] = outcome.program.describe()

    support = heldout.sample(support_size, seed=seed + 500)

    # Shortcut check first: a family one primitive solves cannot carry the
    # claim, and finding that out after reporting a win would be too late.
    shortcut = ProcedureInducer(max_depth=1).induce(support)
    if shortcut.found:
        return CompilerFreeResult(
            heldout_family=heldout.family_id,
            primitive_set_sha=PRIMITIVE_SET_SHA,
            support_size=support_size,
            transfer_size=transfer_size,
            induced=None,
            transfer_accuracy=0.0,
            lesions=[],
            null_runs=0,
            null_found=0,
            single_primitive_shortcut=True,
            train_families_solved=solved,
            train_families_total=len(train_families),
            verdict=VERDICT_VOID,
            reasons=[
                "a single primitive solves the held-out family "
                f"({shortcut.program.describe() if shortcut.program else '?'}); "
                "that is a compiled family, not an induced procedure"
            ],
        )

    # The null, before the positive result rather than after. A searcher that
    # fits permuted outputs cannot support any finding it makes.
    null_found = 0
    for run in range(null_runs):
        outputs = [item.output for item in support]
        random.Random(seed + 9000 + run).shuffle(outputs)
        permuted = [
            TaskInstance(item.inputs, output)
            for item, output in zip(support, outputs)
        ]
        if inducer.induce(permuted).found:
            null_found += 1

    outcome: InductionOutcome = inducer.induce(support)
    if not outcome.found or outcome.program is None:
        return CompilerFreeResult(
            heldout_family=heldout.family_id,
            primitive_set_sha=PRIMITIVE_SET_SHA,
            support_size=support_size,
            transfer_size=transfer_size,
            induced=None,
            transfer_accuracy=0.0,
            lesions=[],
            null_runs=null_runs,
            null_found=null_found,
            single_primitive_shortcut=False,
            train_families_solved=solved,
            train_families_total=len(train_families),
            verdict=VERDICT_NOT_ACQUIRED,
            reasons=[outcome.refusal or "no procedure was induced"],
        )

    frozen: Program = outcome.program
    frozen_sha = frozen.sha()

    # Transfer: instances from a different seed, never seen by the search.
    fresh = heldout.sample(transfer_size, seed=seed + 777)
    transfer = accuracy(frozen, fresh)

    # Lesion 1 — remove composition, keep the search. This is the arm that
    # matters: it isolates the composed procedure rather than the primitives.
    depth1 = ProcedureInducer(max_depth=1)
    best_depth1 = 0.0
    best_depth1_expr = "none"
    for candidate in _depth_one_candidates(support):
        score = accuracy(candidate, fresh)
        if score > best_depth1:
            best_depth1, best_depth1_expr = score, candidate.describe()
    del depth1

    # Lesion 2 — remove the procedure entirely.
    constant = _most_common_output(support)
    lesions = [
        ArmResult(
            "no_composition",
            round(best_depth1, 6),
            f"best depth-1 program on the same primitives: {best_depth1_expr}",
        ),
        ArmResult(
            "no_procedure",
            round(_constant_accuracy(constant, fresh), 6),
            "predict the most common training output",
        ),
    ]

    best_lesion = max(arm.accuracy for arm in lesions)
    if null_found:
        reasons.append(
            f"{null_found} of {null_runs} shuffled-output runs induced a program; "
            "the searcher fits noise at this support size"
        )
    if transfer <= best_lesion:
        reasons.append(
            f"transfer {transfer:.3f} does not exceed the best lesion {best_lesion:.3f}"
        )
    if len(solved) < 2:
        reasons.append(
            f"the inducer solved {len(solved)} of {len(train_families)} training "
            "families; one family is not evidence of a general inducer"
        )

    verdict = VERDICT_ACQUIRED if not reasons else VERDICT_NOT_ACQUIRED
    if verdict == VERDICT_ACQUIRED:
        reasons = [
            f"induced {frozen.describe()} from {support_size} input/output pairs "
            f"with no family label, transferred at {transfer:.3f} on "
            f"{transfer_size} unseen instances, and both lesions collapse to "
            f"{best_lesion:.3f}"
        ]
        logger.info("🧩 procedure acquired: %s", frozen.describe())

    induced = frozen.to_dict()
    induced["frozen_sha"] = frozen_sha
    induced["programs_considered"] = outcome.programs_considered
    return CompilerFreeResult(
        heldout_family=heldout.family_id,
        primitive_set_sha=PRIMITIVE_SET_SHA,
        support_size=support_size,
        transfer_size=transfer_size,
        induced=induced,
        transfer_accuracy=round(transfer, 6),
        lesions=lesions,
        null_runs=null_runs,
        null_found=null_found,
        single_primitive_shortcut=False,
        train_families_solved=solved,
        train_families_total=len(train_families),
        verdict=verdict,
        reasons=reasons,
    )


def _depth_one_candidates(support: Sequence[TaskInstance]) -> list[Program]:
    """Every one-instruction program over the support's inputs.

    Built directly rather than by asking the inducer for a fit, because the
    lesion needs the best non-compositional program even when none of them
    reproduces the support outputs exactly — otherwise the arm would report
    zero for the wrong reason.
    """
    from core.learning.procedure_induction import PRIMITIVES, Instruction

    n_inputs = len(support[0].inputs) if support else 1
    programs: list[Program] = []
    for primitive in PRIMITIVES:
        if primitive.arity != 1:
            continue
        for slot in range(n_inputs):
            programs.append(
                Program(n_inputs, (Instruction(primitive.name, (slot,)),))
            )
    return programs


__all__ = [
    "HELDOUT_FAMILY",
    "NULL_RUNS",
    "SUPPORT_SIZE",
    "TRAIN_FAMILIES",
    "TRANSFER_SIZE",
    "VERDICT_ACQUIRED",
    "VERDICT_NOT_ACQUIRED",
    "VERDICT_VOID",
    "ArmResult",
    "CompilerFreeResult",
    "run_compiler_free_experiment",
]
