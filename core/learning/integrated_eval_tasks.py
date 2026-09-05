"""Retrieval-dependent verifiable tasks: the integrated-evaluation substrate.

The Jul 20 audit found every RLC capability number so far was measured with
ZERO organs engaged — and on task families (khop/modular/register_trace)
that are self-contained by construction, where retrieval CANNOT help. That
experiment had no power to detect Bryan's actual bet: the latent workspace
as the integration substrate that lets a 32B match frontier breadth through
memory, retrieval, and organ content.

These tasks make integration measurable. Each task is a question whose
answer depends on FACTS THAT ARE NOT IN THE PROMPT: entity attributes drawn
from a seeded fact base, where the answer value is a random code from a
space large enough that guessing is negligible (1 in 26^2·10^3). The facts
travel as typed cognitive-context items — the exact wire format the live
organ ingress produces — so a paired evaluation (context-on vs context-off)
measures whether slot ingress causally converts organ content into verified
answers.

Structural honesty: the answer literally does not appear in the prompt, so
a context-off arm scoring above chance is evidence of harness leakage, and
the generator refuses configurations where that property would not hold.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass

INTEGRATED_TASKS_SCHEMA = "aura.integrated_eval_tasks.v1"
INTEGRATED_TASKS_VERSION = "2026.07.20.1"

_ENTITIES = (
    "orrin", "veska", "talin", "mirel", "dovan", "sarei", "kelim", "noral",
    "peshta", "quorin", "ravel", "ilsen",
)
_ATTRIBUTES = ("access_code", "vault_key", "relay_id", "beacon_tag")
_MAX_PER_CELL = 256


@dataclass(frozen=True)
class IntegratedTask:
    """One question + the organ context that makes it answerable."""

    prompt: str
    answer: str  # bare code; graded exact, case-insensitive
    context_items: tuple[dict[str, str], ...]
    distractor_items: tuple[dict[str, str], ...]
    family: str
    seed: int
    hops: int

    def task_id(self) -> str:
        return (
            f"{self.family}-h{self.hops}-s{self.seed}-"
            f"{hashlib.sha256(self.prompt.encode()).hexdigest()[:12]}"
        )


def _rng(seed: int, index: int) -> random.Random:
    material = f"{INTEGRATED_TASKS_VERSION}:{seed}:{index}"
    return random.Random(
        int.from_bytes(hashlib.sha256(material.encode()).digest()[:8], "big")
    )


def _code(rng: random.Random) -> str:
    letters = "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ") for _ in range(2))
    return f"{letters}-{rng.randint(100, 999)}"


def generate_tasks(
    *,
    count: int,
    seed: int,
    hops: int = 2,
) -> list[IntegratedTask]:
    """Deterministic retrieval-dependent tasks.

    ``hops=1``: the answer is one fact. ``hops=2``: the question names an
    entity whose relevant attribute must be found via an intermediate fact
    ("the keeper of X's relay is Y; Y's access code is Z") — integration,
    not lookup. Every answer code is generated fresh per task and never
    rendered into the prompt.
    """
    if type(count) is not int or not 1 <= count <= _MAX_PER_CELL:
        raise ValueError(f"count must be inside [1, {_MAX_PER_CELL}]")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if hops not in (1, 2):
        raise ValueError("hops must be 1 or 2")
    tasks: list[IntegratedTask] = []
    for index in range(count):
        rng = _rng(seed, index)
        entities = rng.sample(_ENTITIES, 4)
        attribute = rng.choice(_ATTRIBUTES)
        answer = _code(rng)
        subject, intermediate, distractor_a, distractor_b = entities
        attribute_label = attribute.replace("_", " ")
        if hops == 1:
            facts = [
                {
                    "source": "memory",
                    "text": f"Recorded fact: {subject}'s {attribute_label} is {answer}.",
                }
            ]
            question = (
                f"Using your recalled records, what is {subject}'s "
                f"{attribute_label}? Reply with exactly:\n"
                "FINAL_ANSWER: <code>"
            )
        else:
            facts = [
                {
                    "source": "memory",
                    "text": (
                        f"Recorded fact: the designated keeper for {subject} "
                        f"is {intermediate}."
                    ),
                },
                {
                    "source": "memory",
                    "text": (
                        f"Recorded fact: {intermediate}'s {attribute_label} "
                        f"is {answer}."
                    ),
                },
            ]
            question = (
                f"Using your recalled records, find the designated keeper "
                f"for {subject}, then report that keeper's "
                f"{attribute_label}. Reply with exactly:\n"
                "FINAL_ANSWER: <code>"
            )
        distractors = tuple(
            {
                "source": "memory",
                "text": (
                    f"Recorded fact: {name}'s "
                    f"{rng.choice(_ATTRIBUTES).replace('_', ' ')} is {_code(rng)}."
                ),
            }
            for name in (distractor_a, distractor_b)
        )
        # Structural honesty check: the answer must not leak into the prompt.
        if answer.lower() in question.lower():
            raise RuntimeError("generator produced an answer-leaking prompt")
        tasks.append(
            IntegratedTask(
                prompt=question,
                answer=answer,
                context_items=tuple(facts),
                distractor_items=distractors,
                family=f"fact_lookup_h{hops}",
                seed=seed,
                hops=hops,
            )
        )
    return tasks


def grade(task: IntegratedTask, text: str) -> str:
    """Four-way outcome mirroring the accuracy ladder's honesty contract."""
    raw = str(text or "")
    marker = "FINAL_ANSWER:"
    upper = raw.upper()
    if marker in upper:
        after = raw[upper.index(marker) + len(marker) :]
        produced = after.strip().splitlines()[0].strip() if after.strip() else ""
        if produced:
            return (
                "correct"
                if produced.upper().rstrip(".") == task.answer.upper()
                else "incorrect"
            )
    # Lenient: the exact code anywhere in the reply still proves retrieval
    # integration; scoring it as failure would conflate format with ability.
    if task.answer.upper() in upper:
        return "correct_lenient"
    return "unparseable" if not raw.strip() else "incorrect_lenient"


def context_for_arm(task: IntegratedTask, *, with_facts: bool) -> list[dict[str, str]]:
    """The organ wire items for one arm. Distractors ride in BOTH arms so
    the context-on arm's advantage cannot be mere presence-of-context."""
    items = list(task.distractor_items)
    if with_facts:
        items = list(task.context_items) + items
    return items[:6]


__all__ = [
    "INTEGRATED_TASKS_SCHEMA",
    "INTEGRATED_TASKS_VERSION",
    "IntegratedTask",
    "context_for_arm",
    "generate_tasks",
    "grade",
]
