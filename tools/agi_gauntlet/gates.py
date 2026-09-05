"""tools/agi_gauntlet/gates.py — the eighteen, and what each would take.

A single benchmark is not the claim. The intersection is: broad, adaptive,
transferable competence, using substantially the same machinery throughout,
in situations the system has not met.

Every gate here declares four things — what it measures, what would count as
passing, what the control is, and whether this harness can run it. The fourth
matters as much as the first three. Seven of these need an external set, a
sealed image, a post-cutoff repository or a human baseline, and a harness that
quietly substitutes a proxy for one of those is how a system gets credited
with a capability nobody measured. Those gates report NOT RUN and the protocol
for running them, and they never report a number.

The eleven that do run, run against the organism itself rather than against a
model of it, on worlds generated from the freeze.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from tools.agi_gauntlet.protocol import Freeze, Receipt

__all__ = ["Gate", "THE_GATES", "the_gate_called", "run_a_gate"]


@dataclass(frozen=True)
class Gate:
    """One of the eighteen, and the honest state of it."""

    number: int
    name: str
    #: What has to be demonstrated.
    measures: str
    #: What counts as passing, stated before anything runs.
    passes_when: str
    #: What makes the result mean something rather than look good.
    control: str
    #: How to run it when this harness cannot.
    if_not_here: str = ""
    run: Callable[[Freeze, dict[str, Any]], dict[str, Any]] | None = None

    @property
    def runnable(self) -> bool:
        return self.run is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "number": self.number,
            "name": self.name,
            "measures": self.measures,
            "passes_when": self.passes_when,
            "control": self.control,
            "runnable_here": self.runnable,
            "if_not_here": self.if_not_here,
        }


def run_a_gate(gate: Gate, freeze: Freeze, options: dict[str, Any]) -> Receipt:
    """Run one, or say why not. Never both, and never neither."""

    receipt = Receipt(gate=gate.name, freeze=freeze)
    if gate.run is None:
        receipt.ran = False
        receipt.why_not = gate.if_not_here or "no harness for this here"
        return receipt
    began = time.monotonic()
    try:
        found = gate.run(freeze, options)
    except Exception as exc:  # noqa: BLE001 — a gate that raises has not passed
        receipt.ran = False
        receipt.passed = False
        receipt.why_not = f"{type(exc).__name__}: {exc}"
        receipt.seconds = time.monotonic() - began
        return receipt
    receipt.seconds = time.monotonic() - began
    receipt.ran = True
    receipt.passed = bool(found.pop("passed", False))
    receipt.trajectories = list(found.pop("trajectories", []))
    receipt.measurements = found
    return receipt


# ── the ones that need somebody else ─────────────────────────────────────

_NEEDS_AN_EVALUATOR = (
    Gate(
        5,
        "broad everyday competence",
        "research, browsing, documents, calculation, tool use, evidence, on "
        "ordinary multi-step questions",
        "around 90% on a private, uncontaminated GAIA-style set — GAIA's own "
        "measured human baseline was 92%",
        "a matched human panel on the same private set, and a contamination "
        "check that the questions post-date the model",
        "needs a GAIA holdout the system has never seen. Obtain or construct "
        "one after this freeze, run with tool access and a fixed budget, and "
        "report accuracy beside interaction count.",
    ),
    Gate(
        6,
        "computer-world competence",
        "operating unfamiliar desktop software from perception and feedback",
        "near-human success on hidden OSWorld 2.0 / Verified-style tasks while "
        "approaching human action efficiency",
        "OSWorld-Human action counts on the same tasks; success at five "
        "hundred actions where a person takes twenty is not the same thing",
        "needs OSWorld images and a VM. Run the Verified split with a step "
        "cap, and report success beside actions-to-success against the human "
        "traces.",
    ),
    Gate(
        7,
        "real software engineering",
        "understanding unfamiliar repositories, debugging, changing "
        "architecture, satisfying requirements, recovering from mistakes",
        "strong results on fresh post-cutoff repository tasks in several "
        "languages, with SWE-bench Verified only as a secondary check",
        "the repositories must post-date the weights, or the result measures "
        "memorisation",
        "needs repositories created after the weight cutoff. Mine merged PRs "
        "from after that date, build the harness from their tests, and run "
        "with the tests hidden until submission.",
    ),
    Gate(
        8,
        "long-horizon autonomy",
        "staying competent over hours without being steered back",
        "at least 80% on tasks a competent person takes one to two hours over, "
        "and meaningful success on eight-hour-equivalent work, across families",
        "human time estimates per task, and deliberate interruptions: tool "
        "outages, a wrong assumption discovered late, a restart mid-task",
        "needs runs measured in hours and human-calibrated task lengths. Use "
        "HCAST-style tasks with per-task human time, and record every human "
        "intervention as a failure of autonomy rather than a hint.",
    ),
    Gate(
        12,
        "multimodal integration",
        "combining text, screenshots, diagrams, audio, documents, tables and "
        "environment state where no single channel suffices",
        "human-level completion on mixed-modality tasks built so that any one "
        "channel alone is insufficient",
        "single-channel ablations: if the text-only score matches the full "
        "score, the other channels were decoration",
        "needs sealed multimodal assets. Build tasks whose answer is only "
        "recoverable from two channels together, and run the single-channel "
        "ablations first.",
    ),
    Gate(
        14,
        "social and instructional intelligence",
        "learning from people, handling ambiguous instructions, negotiating "
        "clarification, modelling what another person knows",
        "human-range performance on novel collaborative tasks rather than "
        "canned social benchmarks",
        "the same tasks with a scripted partner: a system that scores the same "
        "against a script was not collaborating",
        "needs evaluators playing colleagues. Run the new-employee protocol: a "
        "fictional organisation, a week of simulated work, coworkers played by "
        "people, and human onboarding as the baseline.",
    ),
)


from tools.agi_gauntlet import runnable as _r

THE_GATES: tuple[Gate, ...] = (
    Gate(
        1,
        "fluid intelligence",
        "infer an unfamiliar rule from sparse examples rather than retrieve a "
        "known solution",
        "human-range accuracy on sealed novel reasoning, and a refusal far "
        "more often than a wrong guess where the evidence does not settle "
        "it — ten refusals to every confident wrong answer, until a human "
        "baseline says what that ratio should be",
        "the rules are composed from the freeze seed, so no instance existed "
        "before the commit; the question is asked at a length the examples "
        "did not use",
        run=_r.fluid_intelligence,
    ),
    Gate(
        2,
        "interactive novel-world learning",
        "enter a world with no instructions, discover what the acts mean, "
        "find the goal, and succeed",
        "near-total completion, at an interaction count near the shortest "
        "path rather than by exhausting the space",
        "the same worlds played by choosing acts at random; a world small "
        "enough to stumble through proves nothing",
        run=_r.interactive_novel_world,
    ),
    Gate(
        3,
        "learning from experience",
        "start mediocre at something unfamiliar and become markedly better "
        "through her own accumulated experience",
        "a clear rise across thirty independent trajectories, and the rise "
        "must be larger than the same trajectories with memory reset",
        "the reset ablation. A curve that rises identically without memory is "
        "a curve about the environment",
        run=_r.learning_from_experience,
    ),
    Gate(
        4,
        "cross-domain transfer",
        "discover something in one world and apply it in another without "
        "being told the mapping",
        "T̄ = P(B|A) − P(B|∅) clearly above zero across unrelated families, "
        "and at or below zero on the negative controls",
        "pairs built to look alike and differ underneath. A system that "
        "transfers there is matching surfaces",
        run=_r.transfer,
    ),
    *_NEEDS_AN_EVALUATOR[:2],
    _NEEDS_AN_EVALUATOR[2],
    _NEEDS_AN_EVALUATOR[3],
    Gate(
        9,
        "new-skill acquisition",
        "read or explore an unfamiliar system and become competent at it",
        "a human-comparable apprenticeship curve in sealed environments built "
        "after the freeze",
        "the same curve for a learner whose experience is thrown away between "
        "episodes",
        "the offline half runs as gate 3; the human-comparable half needs "
        "people learning the same sealed environments under the same access.",
        run=_r.learning_from_experience,
    ),
    Gate(
        10,
        "concept acquisition and invention",
        "develop a distinction the evaluator did not supply, and reason with it",
        "a proposal no composition of the existing vocabulary reaches, "
        "admitted; a macro and a duplicate refused; and a second generation "
        "built on the first",
        "the macro and duplicate proposals. A vocabulary that admits "
        "everything is not inventing",
        run=_r.concept_invention,
    ),
    Gate(
        11,
        "planning under novelty",
        "set subgoals, revise them after a surprise, and abandon a plan that "
        "has stopped working",
        "recovery when the rules change mid-run, clearly above a policy that "
        "keeps executing the plan it had",
        "the stubborn policy on the same worlds after the same change",
        run=_r.planning_under_novelty,
    ),
    _NEEDS_AN_EVALUATOR[4],
    Gate(
        13,
        "epistemic competence",
        "distinguish knowledge from uncertainty, investigate the uncertainty, "
        "and update after contradictory evidence",
        "calibration error under 0.15, the belief moving towards the truth "
        "after looking, no observation taken once the question is settled, "
        "and the cheap way preferred where both would settle it",
        "the settled case, where nothing further could change the answer and "
        "looking anyway is an expensive habit rather than an epistemic one",
        run=_r.epistemic_competence,
    ),
    _NEEDS_AN_EVALUATOR[5],
    Gate(
        15,
        "robustness",
        "keep working when a tool fails, information is missing or the model "
        "is wrong",
        "no invented answers under pressure, a tool failure survived without "
        "the failure being blamed on the instrument, and the clean score held",
        "the clean run beside the damaged one. Recovery is only meaningful "
        "against what it recovered from",
        run=_r.robustness,
    ),
    Gate(
        16,
        "generality rather than a bag of solvers",
        "the same machinery across all of the above, with no path keyed to an "
        "evaluation",
        "no benchmark name anywhere under core, interface, skills, llm, "
        "executors or security — including in a comment",
        "the check is a grep, which is weak and cannot be argued with. A "
        "strong check somebody can dispute is worth less here",
        run=_r.generality_not_a_bag_of_solvers,
    ),
    Gate(
        17,
        "persistence of learning",
        "improvements survive time and restart rather than living in one "
        "process",
        "the record her developmental policy reads, the library of structures "
        "and what she has learned failure looks like all come back after the "
        "process is killed",
        "the state is written, the process state is cleared, and the same "
        "questions are asked again",
        run=_r.persistence_across_restart,
    ),
    Gate(
        18,
        "independent reproducibility",
        "somebody else runs the frozen system and gets the same result",
        "the environments regenerate identically from the freeze, one gate "
        "re-runs identically, and the freeze names what actually ran",
        "the freeze itself: a dirty tree names a commit other than the one "
        "that ran",
        "the other half needs an outside team building its own families after "
        "this freeze and never seeing these. This gate checks the run "
        "reproduces; it cannot check that somebody else ran it.",
        run=_r.reproducibility,
    ),
)


def the_gate_called(name: str) -> Gate | None:
    for one in THE_GATES:
        if one.name == name:
            return one
    return None
