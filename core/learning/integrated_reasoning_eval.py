"""Pair retrieval and depth interventions on knowledge-combination tasks.

The solver receives a question, retrieved passages and a requested depth.
The default measurement grades its terminal public scalar and requires both
ablations to disrupt the same task before reporting joint necessity. Exact
paired tails accompany the descriptive effects; this diagnostic is neither
a preregistered replication nor proof of the callback's runtime identity.

Fixture retrieval supplies planted facts directly. It measures whether a
solver uses supplied facts, not whether Aura's live retrieval found them.
No-retrieval errors do not prove that a model lacks the relevant knowledge.
The legacy policy keeps the original CP236 grading and aggregation replayable.
"""
from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Protocol

INTEGRATED_EVAL_SCHEMA = "aura.integrated_reasoning_eval.v2"
INTEGRATED_EVAL_POLICY = "public_terminal_paired_v2"
LEGACY_EVAL_POLICY = "substring_aggregate_v1"

# The nine operational marks of reasoning (RLC Context). Each task declares
# which it exercises, so a run reports COVERAGE rather than asserting it.
REASONING_CRITERIA = (
    "combine_unpresented_facts",
    "derive_unstored_answer",
    "maintain_intermediate_variables",
    "revise_after_evidence",
    "distinguish_conflicting_evidence",
    "search_alternatives",
    "detect_inconsistency",
    "compute_improves_answer",
    "transfer_to_unfamiliar",
)


@dataclass(frozen=True)
class Fact:
    """One retrievable statement, absent from the prompt by construction."""

    subject: str
    relation: str
    obj: str
    authority: float = 1.0  # higher wins when two facts conflict
    text: str = ""

    def as_passage(self) -> str:
        return self.text or f"{self.subject} {self.relation} {self.obj}."


@dataclass(frozen=True)
class KnowledgeTask:
    """A question answerable only by retrieving and combining facts."""

    task_id: str
    prompt: str
    answer: str
    facts: tuple[Fact, ...]  # planted in the retrieval store
    hops: int  # how many facts must be chained (= required depth)
    criteria: tuple[str, ...]
    distractors: tuple[Fact, ...] = ()

    def __post_init__(self) -> None:
        if not self.prompt.strip() or not self.answer.strip():
            raise ValueError("task needs a prompt and an answer")
        if type(self.hops) is not int or self.hops < 1:
            raise ValueError("hops must be a positive integer")
        if self.answer.lower() in self.prompt.lower():
            # The whole point: a prompt that contains its answer measures
            # reading comprehension, not reasoning or retrieval.
            raise ValueError(
                f"{self.task_id}: the answer appears in the prompt; this task "
                "cannot distinguish reasoning from reading"
            )
        unknown = set(self.criteria) - set(REASONING_CRITERIA)
        if unknown:
            raise ValueError(f"unknown reasoning criteria: {sorted(unknown)}")

    def all_passages(self) -> list[str]:
        pool = list(self.facts) + list(self.distractors)
        return [fact.as_passage() for fact in pool]


class RetrievalSource(Protocol):
    """The seam the real organ plugs into.

    A fixture implements this for deterministic tests; a thin adapter over
    ``memory_facade.search`` implements it for the live run. The harness
    never knows which -- that is what makes the ablation clean.
    """

    def retrieve(self, query: str, *, limit: int) -> list[str]: ...


@dataclass
class FixtureRetrieval:
    """Deterministic retrieval over a planted fact store.

    Returns a task's own facts (and distractors) for its query, so the
    harness can be tested without the live organ. Ranking is by authority
    then insertion order, mirroring how a real store surfaces the most
    trusted passage first.
    """

    passages: dict[str, list[tuple[float, str]]] = field(default_factory=dict)

    def plant(self, task: KnowledgeTask) -> None:
        ranked = sorted(
            list(task.facts) + list(task.distractors),
            key=lambda f: -f.authority,
        )
        self.passages[task.prompt] = [(f.authority, f.as_passage()) for f in ranked]

    def retrieve(self, query: str, *, limit: int) -> list[str]:
        return [passage for _authority, passage in self.passages.get(query, ())[:limit]]


# Retrieval mode and depth define the factorial cells.
RETRIEVAL_ON = "retrieval_on"
RETRIEVAL_OFF = "retrieval_off"


def _selected_public_answer(produced: object) -> str:
    """Read a terminal scalar answer using the shared channel and text rules."""
    from core.brain.llm.chat_format import split_native_thinking_generation
    from core.learning.heldout_battery import normalize_answer

    if not isinstance(produced, str):
        return ""
    channels = split_native_thinking_generation(produced, native_thinking="<think>" in produced)
    if not channels.boundary_closed or "<think>" in channels.surface:
        return ""
    lines = channels.surface.strip().splitlines()
    if not lines:
        return ""
    selected = lines[-1].strip()
    label, separator, value = selected.partition(":")
    if separator and label.casefold() in {"final_answer", "answer"}:
        selected = value.strip()
    return normalize_answer(selected, "str")


def _grade_answer(produced: object, answer: str, policy: str) -> bool:
    if policy == LEGACY_EVAL_POLICY:
        return answer.strip().lower() in str(produced or "").strip().lower()
    if policy != INTEGRATED_EVAL_POLICY:
        raise ValueError("unknown integrated evaluation policy")
    from core.learning.heldout_battery import normalize_answer

    selected = _selected_public_answer(produced)
    return bool(selected) and selected == normalize_answer(answer, "str")


def _validate_design(tasks, depths, policy):
    if policy not in {INTEGRATED_EVAL_POLICY, LEGACY_EVAL_POLICY}:
        raise ValueError("unknown integrated evaluation policy")
    if not tasks:
        raise ValueError("no tasks to evaluate")
    if policy == INTEGRATED_EVAL_POLICY:
        from core.learning.heldout_battery import normalize_answer

        if any(not normalize_answer(task.answer, "str") for task in tasks):
            raise ValueError("task answer has no normalized public value")
    if (not depths or any(type(d) is not int or d < 1 for d in depths)
            or tuple(depths) != tuple(sorted(set(depths)))):
        raise ValueError("depths must be distinct increasing positive integers")
    if (len({task.task_id for task in tasks}) != len(tasks)
            or len({task.prompt for task in tasks}) != len(tasks)):
        raise ValueError("duplicate integrated evaluation tasks")


def run_factorial(
    tasks: list[KnowledgeTask],
    source: RetrievalSource,
    solve: Callable[[str, list[str], int], str],
    *,
    depths: tuple[int, ...] = (1, 2, 4),
    retrieval_limit: int = 6,
    evaluation_policy: str = INTEGRATED_EVAL_POLICY,
) -> dict[str, Any]:
    """Run {retrieval on/off} x {depth} and grade every cell.

    ``solve(prompt, context, depth)`` returns the model's answer. New runs
    grade the terminal public scalar and pair both ablations on each task.
    The explicit legacy policy permits replay of historical measurements.
    """
    _validate_design(tasks, depths, evaluation_policy)
    if type(retrieval_limit) is not int or retrieval_limit < 1:
        raise ValueError("retrieval limit must be a positive integer")

    cells: dict[str, dict[int, list[bool]]] = {
        RETRIEVAL_ON: {d: [] for d in depths},
        RETRIEVAL_OFF: {d: [] for d in depths},
    }
    per_task: list[dict[str, Any]] = []
    for task in tasks:
        context = source.retrieve(task.prompt, limit=retrieval_limit)
        row: dict[str, Any] = {"task_id": task.task_id, "hops": task.hops, "selected_answers": {}}
        for depth in depths:
            on_text = solve(task.prompt, list(context), depth)
            off_text = solve(task.prompt, [], depth)
            on = _grade_answer(on_text, task.answer, evaluation_policy)
            off = _grade_answer(off_text, task.answer, evaluation_policy)
            row["selected_answers"].update({f"on@{depth}": _selected_public_answer(on_text),
                                             f"off@{depth}": _selected_public_answer(off_text)})
            cells[RETRIEVAL_ON][depth].append(on)
            cells[RETRIEVAL_OFF][depth].append(off)
            row[f"on@{depth}"] = on
            row[f"off@{depth}"] = off
        per_task.append(row)

    accuracy = {
        mode: {d: _mean(results) for d, results in by_depth.items()}
        for mode, by_depth in cells.items()
    }
    return {
        "schema": INTEGRATED_EVAL_SCHEMA,
        "evaluation_policy": evaluation_policy,
        "n_tasks": len(tasks),
        "depths": list(depths),
        "accuracy": accuracy,
        "verdicts": (_verdicts(accuracy, depths) if evaluation_policy == LEGACY_EVAL_POLICY
                     else _paired_verdicts(accuracy, depths, per_task)),
        "criteria_coverage": _coverage(tasks),
        "task_manifest": [asdict(task) for task in tasks],
        "per_task": per_task,
        "fixture_oracle_retrieval": isinstance(source, FixtureRetrieval),
        "runtime_retrieval_verified": False,
        "ordinary_pipeline_verified": False,
        "confirmatory_evidence": False,
        "serving_authority": False,
    }


def _mean(flags: list[bool]) -> float:
    return round(sum(1 for f in flags if f) / len(flags), 4) if flags else 0.0


def _paired_verdicts(accuracy, depths, per_task):
    """Report fixed-depth effects and their intersection on the same tasks."""
    from core.brain.llm.latent_cortex.exact_paired_statistics import exact_paired_binomial_tail

    shallow, deep = depths[0], depths[-1]
    treatment = f"on@{deep}"

    def paired(control):
        wins = sum(row[treatment] and not row[control] for row in per_task)
        losses = sum(row[control] and not row[treatment] for row in per_task)
        return {"wins": wins, "losses": losses, "ties": len(per_task) - wins - losses,
                "effect": (wins - losses) / len(per_task),
                "exact_one_sided_tail": asdict(exact_paired_binomial_tail(wins, losses))}

    retrieval = paired(f"off@{deep}")
    depth = paired(f"on@{shallow}")
    joint = [row["task_id"] for row in per_task
             if row[treatment] and not row[f"off@{deep}"] and not row[f"on@{shallow}"]]
    retrieval_helps, depth_helps = retrieval["effect"] > .1, depth["effect"] > .1
    both = bool(joint) and retrieval_helps and depth_helps
    return {
        "retrieval_is_causal": retrieval_helps,
        "recurrence_helps": depth_helps,
        "recurrence_hurts": depth["effect"] < -.1,
        "both_required": both,
        "joint_success_task_ids": joint,
        "joint_success_fraction": len(joint) / len(per_task),
        "retrieval_comparison_depth": deep,
        "depth_comparison": [shallow, deep],
        "paired_retrieval": retrieval,
        "paired_depth": depth,
        "on_shallow": accuracy[RETRIEVAL_ON][shallow],
        "on_deep": accuracy[RETRIEVAL_ON][deep],
        "off_at_deep": accuracy[RETRIEVAL_OFF][deep],
        "confirmatory_evidence": False,
        "claim": ("Diagnostic: the tested solver used both factors on the same tasks for answers "
                  "it could not produce alone. Runtime identity and fresh replication remain unverified."
                  if both else "The diagnostic does not show both factors required on the same tasks."),
    }


def _verdicts(accuracy: dict, depths: tuple[int, ...]) -> dict[str, Any]:
    """The two causal claims, judged at the RIGHT depth for each.

    An earlier version judged retrieval causality only at the DEEPEST depth.
    When recurrence degrades accuracy to zero at that depth (which it does),
    retrieval-on and retrieval-off both read 0% there and retrieval is
    falsely declared non-causal -- even when it is overwhelmingly causal at
    shallow depth. The comparison point was the bug, not the data. Retrieval
    causality is now judged at the depth where retrieval-on performs BEST,
    and recurrence is reported honestly whether it helps OR hurts.
    """
    on = accuracy[RETRIEVAL_ON]
    off = accuracy[RETRIEVAL_OFF]
    best_depth = max(depths, key=lambda d: on[d])
    shallow, deep = depths[0], depths[-1]

    # Disabling retrieval breaks the result AT THE DEPTH WHERE IT HELPS MOST.
    retrieval_gain = on[best_depth] - off[best_depth]
    retrieval_causal = retrieval_gain > 0.1
    # Does depth HELP (best is deeper than shallow) or HURT (best is shallow
    # and deep is worse)? Both are causal facts; only one is the thesis.
    recurrence_helps = on[deep] > on[shallow] + 0.1
    recurrence_hurts = on[shallow] > on[deep] + 0.1
    # The thesis conjunction still requires depth to HELP. It does not here,
    # so it stays False -- but for the honest reason (depth is a net
    # negative), not because retrieval failed.
    both_required = retrieval_causal and recurrence_helps

    if both_required:
        claim = (
            "Aura retrieved knowledge she lacked and used recurrent depth to "
            "combine it into an answer she could not produce alone"
        )
    elif retrieval_causal and recurrence_hurts:
        claim = (
            "RETRIEVAL half CONFIRMED: retrieval turns 0 into a real score; "
            "the model uses external knowledge it does not have. DEPTH half "
            "REFUTED: recurrence degrades accuracy rather than helping."
        )
    elif retrieval_causal:
        claim = "retrieval is causal; depth is neutral"
    else:
        claim = "neither retrieval nor depth demonstrated"

    return {
        "retrieval_is_causal": bool(retrieval_causal),
        "recurrence_helps": bool(recurrence_helps),
        "recurrence_hurts": bool(recurrence_hurts),
        "both_required": bool(both_required),
        "claim": claim,
        "best_depth_for_retrieval": int(best_depth),
        "retrieval_gain_at_best_depth": round(retrieval_gain, 4),
        "on_shallow": on[shallow],
        "on_deep": on[deep],
        "off_at_best": off[best_depth],
    }


def _coverage(tasks: list[KnowledgeTask]) -> dict[str, int]:
    counts = {criterion: 0 for criterion in REASONING_CRITERIA}
    for task in tasks:
        for criterion in task.criteria:
            counts[criterion] += 1
    return counts


# ── Generators: knowledge-gated, multi-hop, depth = required hops ───────


def _transitive_chain(rng: random.Random, hops: int, index: int) -> KnowledgeTask:
    """A -> B -> ... following retrieved edges. Chain length = required depth.

    The edges live only in retrieval; the prompt names the start and asks
    for the end. One hop per recurrent pass is the hypothesis the depth
    ablation tests directly.
    """
    nodes = [f"n{rng.randint(1000, 9999)}_{index}_{i}" for i in range(hops + 1)]
    facts = tuple(
        Fact(nodes[i], "points to", nodes[i + 1], text=f"{nodes[i]} points to {nodes[i + 1]}.")
        for i in range(hops)
    )
    prompt = (
        f"Follow the chain starting at {nodes[0]} for {hops} steps. "
        "What node do you reach? Reply FINAL_ANSWER: <node>"
    )
    criteria = (
        "combine_unpresented_facts",
        "derive_unstored_answer",
        "maintain_intermediate_variables",
    )
    if hops >= 2:
        criteria = criteria + ("compute_improves_answer",)
    return KnowledgeTask(
        task_id=f"chain-h{hops}-{index}",
        prompt=prompt,
        answer=nodes[-1],
        facts=facts,
        hops=hops,
        criteria=criteria,
    )


def _conflicting_sources(rng: random.Random, hops: int, index: int) -> KnowledgeTask:
    """Two retrieved facts conflict; the authoritative one wins.

    Tests distinguishing supporting from conflicting evidence and revising
    on the more trusted source -- reasoning marks that a chain cannot show.
    """
    entity = f"e{rng.randint(1000, 9999)}_{index}"
    right = f"v{rng.randint(1000, 9999)}"
    wrong = f"v{rng.randint(1000, 9999)}"
    facts = (
        Fact(entity, "current value is", right, authority=1.0,
             text=f"As of the latest record, {entity} current value is {right}."),
    )
    distractors = (
        Fact(entity, "was once", wrong, authority=0.3,
             text=f"An old note says {entity} was once {wrong}."),
    )
    prompt = (
        f"Sources disagree about {entity}. Using the most authoritative, "
        f"what is its current value? Reply FINAL_ANSWER: <value>"
    )
    return KnowledgeTask(
        task_id=f"conflict-h{hops}-{index}",
        prompt=prompt,
        answer=right,
        facts=facts,
        hops=max(2, hops),
        criteria=(
            "distinguish_conflicting_evidence",
            "revise_after_evidence",
            "detect_inconsistency",
            "derive_unstored_answer",
        ),
        distractors=distractors,
    )


GENERATORS: dict[str, Callable[[random.Random, int, int], KnowledgeTask]] = {
    "transitive_chain": _transitive_chain,
    "conflicting_sources": _conflicting_sources,
}


def build_knowledge_tasks(
    *, families: list[str], hops: list[int], per_cell: int, seed: int
) -> list[KnowledgeTask]:
    unknown = [f for f in families if f not in GENERATORS]
    if unknown:
        raise ValueError(f"unknown families: {unknown}")
    tasks: list[KnowledgeTask] = []
    for family in families:
        for hop in hops:
            rng = random.Random(f"{seed}:{family}:{hop}")
            for index in range(per_cell):
                tasks.append(GENERATORS[family](rng, hop, index))
    if len({t.prompt for t in tasks}) != len(tasks):
        raise RuntimeError("generated duplicate prompts")
    return tasks


def assert_base_recall_guard(
    tasks: list[KnowledgeTask],
    solve: Callable[[str, list[str], int], str],
    *,
    depth: int = 4,
    evaluation_policy: str = INTEGRATED_EVAL_POLICY,
) -> dict[str, Any]:
    """Measure answers without retrieval; failure does not prove absent knowledge."""
    _validate_design(tasks, (depth,), evaluation_policy)
    leaked = []
    for task in tasks:
        produced = solve(task.prompt, [], depth)
        if _grade_answer(produced, task.answer, evaluation_policy):
            leaked.append(task.task_id)
    return {
        "schema": INTEGRATED_EVAL_SCHEMA,
        "evaluation_policy": evaluation_policy,
        "tasks": len(tasks),
        "answered_from_memory": leaked,
        "guard_passed": not leaked,
    }


__all__ = [
    "INTEGRATED_EVAL_SCHEMA",
    "INTEGRATED_EVAL_POLICY",
    "LEGACY_EVAL_POLICY",
    "REASONING_CRITERIA",
    "RETRIEVAL_OFF",
    "RETRIEVAL_ON",
    "Fact",
    "FixtureRetrieval",
    "KnowledgeTask",
    "RetrievalSource",
    "assert_base_recall_guard",
    "build_knowledge_tasks",
    "run_factorial",
]
