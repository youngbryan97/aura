"""The test the thesis has never faced: organs + recurrence, together (CP236).

Every measurement this arc ran used a bare model on self-contained puzzles.
Retrieval could not help, because the answer was always already in the
prompt. That harness had no power to detect the effect the RLC exists for.

Bryan's proof requirement (RLC Context) is precise. To show reasoning
rather than recall, the tasks must have:

* an answer ABSENT from the prompt and from the base model's likely recall;
* required facts available ONLY through retrieval;
* multiple pieces that must be COMBINED;
* some sources that CONFLICT;
* and then two ablations must both bite:
    - disabling retrieval breaks the result  (knowledge was external);
    - disabling recurrence breaks the result (depth did the combining).

Only when BOTH ablations break the same task have we shown the thing the
whole project claims: Aura searched for knowledge she did not possess and
used recurrent computation to derive an answer she could not produce from
innate knowledge alone.

This module builds those tasks and runs the factorial
{retrieval on/off} x {depth 1/2/4} that makes the two causal claims
falsifiable. It is model-agnostic: a solver callback produces an answer
given (prompt, retrieved_context, depth), so the harness is testable with a
fixture solver and drives the live 32B unchanged.

The honesty burden is the opposite of the earlier puzzles'. There the risk
was a task too easy to need depth; here it is a task the base model can
answer from memory, which would credit retrieval for knowledge the model
already had. So every task carries a ``base_recall_guard``: a check that
the answer is NOT derivable from the prompt alone, verified by running the
solver with EMPTY retrieval and requiring it to fail.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

INTEGRATED_EVAL_SCHEMA = "aura.integrated_reasoning_eval.v1"

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
        self.passages[task.task_id] = [(f.authority, f.as_passage()) for f in ranked]

    def retrieve(self, query: str, *, limit: int) -> list[str]:
        # The query carries its task id in a controlled fixture; a real
        # source matches on content. Both honour the limit.
        for task_id, ranked in self.passages.items():
            if task_id in query:
                return [passage for _authority, passage in ranked[:limit]]
        return []


# Retrieval mode and depth define the factorial cells.
RETRIEVAL_ON = "retrieval_on"
RETRIEVAL_OFF = "retrieval_off"


def run_factorial(
    tasks: list[KnowledgeTask],
    source: RetrievalSource,
    solve: Callable[[str, list[str], int], str],
    *,
    depths: tuple[int, ...] = (1, 2, 4),
    retrieval_limit: int = 6,
) -> dict[str, Any]:
    """Run {retrieval on/off} x {depth} and grade every cell.

    ``solve(prompt, context, depth)`` returns the model's answer. Grading
    is case-insensitive containment of the gold answer, which is lenient on
    form and strict on the fact -- the answer string is a planted token
    that does not occur by chance.
    """
    if not tasks:
        raise ValueError("no tasks to evaluate")
    if not depths or any(type(d) is not int or d < 1 for d in depths):
        raise ValueError("depths must be positive integers")

    def graded(prompt: str, context: list[str], depth: int, answer: str) -> bool:
        produced = solve(prompt, context, depth)
        return answer.strip().lower() in str(produced or "").strip().lower()

    cells: dict[str, dict[int, list[bool]]] = {
        RETRIEVAL_ON: {d: [] for d in depths},
        RETRIEVAL_OFF: {d: [] for d in depths},
    }
    per_task: list[dict[str, Any]] = []
    for task in tasks:
        context = source.retrieve(task.task_id + " " + task.prompt, limit=retrieval_limit)
        row: dict[str, Any] = {"task_id": task.task_id, "hops": task.hops}
        for depth in depths:
            on = graded(task.prompt, context, depth, task.answer)
            off = graded(task.prompt, [], depth, task.answer)
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
        "n_tasks": len(tasks),
        "depths": list(depths),
        "accuracy": accuracy,
        "verdicts": _verdicts(accuracy, depths),
        "criteria_coverage": _coverage(tasks),
        "per_task": per_task,
    }


def _mean(flags: list[bool]) -> float:
    return round(sum(1 for f in flags if f) / len(flags), 4) if flags else 0.0


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
) -> dict[str, Any]:
    """Prove the answers are NOT already in the model, before trusting a gain.

    Runs each task with EMPTY retrieval at full depth. Any task the model
    answers from memory alone is disqualified: a retrieval gain on it would
    credit external knowledge for something the model already knew. This is
    the mirror of the earlier puzzles' failure -- there, tasks too easy to
    need depth; here, tasks the base model can recall.
    """
    leaked = []
    for task in tasks:
        produced = solve(task.prompt, [], depth)
        if task.answer.strip().lower() in str(produced or "").strip().lower():
            leaked.append(task.task_id)
    return {
        "schema": INTEGRATED_EVAL_SCHEMA,
        "tasks": len(tasks),
        "answered_from_memory": leaked,
        "guard_passed": not leaked,
    }


__all__ = [
    "INTEGRATED_EVAL_SCHEMA",
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
