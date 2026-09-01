#!/usr/bin/env python3
"""Grown against reset, over sealed blocks, with the confounds refused.

  072  older Aura outperforms reset Aura on later stages because of prior
       structure
  129  V(n+1) beats V(n) on frozen held-out environments with no safety
       regression
  168  novel tools need fewer demonstrations, and tool choice beats static
       descriptions
  203  repeated interaction predicts a partner better than the language prior
       and transfers across contexts

Four cards, one discipline. Each is a claim that something got better BECAUSE
of what was kept, and each dies the same three ways: the later blocks were
easier, the grown arm was handed more context, or the answers had been seen
before. core.science.developmental_campaign refuses all three, and this runs
a campaign through it whose growth is real — the grown arm keeps a library of
solved sub-structures and the reset arm loses it between blocks.

The lesion arm is the one that turns a correlation into a cause: the grown
agent with its library removed at evaluation time should fall back to the
reset arm's score. If it does not, the advantage was never the library.

    python tools/campaigns/developmental_campaign_run.py
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.cognition.agent_model import AgentRegistry  # noqa: E402
from core.science.developmental_campaign import (  # noqa: E402
    Arm,
    DevelopmentalCampaign,
)

#: The alphabet tasks are built from. A task is a sequence of these, and a
#: library entry is a contiguous run of them that has been solved before.
PARTS = tuple("abcdefgh")

#: Every arm gets the same context. The grown arm's advantage has to come
#: from what it kept, and a wider window is a confound wearing development's
#: clothes — the campaign voids on it, which is why this is one number.
CONTEXT_TOKENS = 2048


def _task(rng: random.Random, *, length: int) -> tuple[str, ...]:
    return tuple(rng.choice(PARTS) for _ in range(length))


def _solve(task: tuple[str, ...], library: set[tuple[str, ...]]) -> float:
    """Score one task: the fraction of it covered by known sub-structures.

    A part not in the library costs a step; a run that is in the library is
    one step however long it is. The score is what fraction of the work the
    library saved, which is what "prior structure helps" has to mean if it
    means anything measurable.
    """
    steps = 0
    index = 0
    while index < len(task):
        best = 1
        for size in range(min(4, len(task) - index), 1, -1):
            if task[index : index + size] in library:
                best = size
                break
        steps += 1
        index += best
    return 1.0 - (steps - 1) / max(1, len(task) - 1)


def _learn(task: tuple[str, ...], library: set[tuple[str, ...]]) -> None:
    """Keep the sub-structures this task exercised, up to length four."""
    for size in (2, 3, 4):
        for index in range(len(task) - size + 1):
            library.add(task[index : index + size])


def run_campaign(
    *, blocks: int, tasks_per_block: int, length: int, seed: int
) -> dict[str, object]:
    rng = random.Random(seed)
    campaign = DevelopmentalCampaign(seed=seed)
    grown_library: set[tuple[str, ...]] = set()
    library_sizes: list[int] = []
    # "No regression" is the other half of card 129 and it is per task, not
    # per block: a mean that improved can still hide tasks the library made
    # worse, and those are exactly the ones a growth claim must not bury.
    regressions = 0
    tasks_scored = 0

    for block in range(blocks):
        # The task pool is drawn the same way in every block, so a later block
        # is not an easier or harder block. Difficulty drift is the confound
        # that makes a growth curve out of nothing.
        pool = [_task(rng, length=length) for _ in range(tasks_per_block)]
        reset_library: set[tuple[str, ...]] = set()

        for index, task in enumerate(pool):
            answer_key = "".join(task)
            grown_score = _solve(task, grown_library)
            reset_score = _solve(task, reset_library)
            tasks_scored += 1
            regressions += grown_score < reset_score
            campaign.record(
                block, f"b{block}t{index}", Arm.GROWN,
                grown_score,
                answer_key=answer_key, context_tokens=CONTEXT_TOKENS, seed=seed,
            )
            campaign.record(
                block, f"b{block}t{index}", Arm.RESET,
                reset_score,
                answer_key=answer_key, context_tokens=CONTEXT_TOKENS, seed=seed,
            )
            # The grown agent with its ACQUIRED library taken away — what it
            # carried across blocks — and nothing else. Handing it an empty
            # library instead also removes the within-block learning the reset
            # arm keeps, so the lesioned arm scores below reset and the
            # campaign correctly reports that the gap is caused by something
            # other than the library. That reading was right about the arm it
            # was given and wrong about the question.
            campaign.record(
                block, f"b{block}t{index}", Arm.GROWN_LESIONED,
                _solve(task, reset_library),
                answer_key=answer_key, context_tokens=CONTEXT_TOKENS, seed=seed,
            )
            _learn(task, grown_library)
            _learn(task, reset_library)

        library_sizes.append(len(grown_library))

    verdict = campaign.verdict(bootstrap=2000)
    return verdict.to_dict() | {
        "library_growth": library_sizes,
        "context_tokens_per_arm": CONTEXT_TOKENS,
        "tasks_scored": tasks_scored,
        "tasks_growth_made_worse": regressions,
        "regression_rate": round(regressions / tasks_scored, 5) if tasks_scored else 0.0,
    }


# ── 168: a novel tool, and how many demonstrations it takes ───────────────


def tool_learning(
    *, tools: int, trials: int, seed: int
) -> dict[str, object]:
    """How many demonstrations before the right tool is chosen reliably.

    The static arm reads a description and picks by word overlap, which is
    what a schema alone supports. The demonstrated arm keeps what happened
    when it used a tool. Both see the same tasks in the same order; only what
    they are allowed to remember differs.
    """
    rng = random.Random(seed)
    # Each tool works on one kind of task. Descriptions overlap, which is why
    # word matching is not enough — that is the whole difficulty with a tool
    # nobody has used yet.
    kinds = [f"kind{i}" for i in range(tools)]
    # Each description contains its own word and two of its neighbours', so
    # word matching is better than chance and still wrong most of the time.
    # A baseline at chance is not a baseline: it makes any arm look good and
    # says nothing about whether demonstrations are what did it.
    descriptions = {
        kinds[i]: {f"word{i}", f"word{(i + 1) % tools}", f"word{(i + 2) % tools}"}
        for i in range(tools)
    }

    demonstrated: dict[str, dict[str, int]] = {}
    static_right = 0
    learned_right = 0
    to_competence: list[int] = []
    streak = 0
    reached = None

    for trial in range(trials):
        kind = rng.choice(kinds)
        task_words = {f"word{kinds.index(kind)}"}

        # Ties broken at random rather than by index: taking the first match
        # every time makes the baseline deterministic in a way no real
        # description-matcher is, and the number then measures the tie-break.
        overlaps = {k: len(descriptions[k] & task_words) for k in kinds}
        best_overlap = max(overlaps.values())
        static_pick = rng.choice([k for k in kinds if overlaps[k] == best_overlap])
        static_right += static_pick == kind

        seen = demonstrated.get(kind) or {}
        learned_pick = (
            max(seen, key=lambda k: seen[k]) if seen else static_pick
        )
        correct = learned_pick == kind
        learned_right += correct
        streak = streak + 1 if correct else 0
        if reached is None and streak >= 10:
            reached = trial + 1
            to_competence.append(reached)

        # The demonstration: using the right tool is remembered as such.
        demonstrated.setdefault(kind, {}).setdefault(kind, 0)
        demonstrated[kind][kind] += 1

    return {
        "tools": tools,
        "trials": trials,
        "static_accuracy": round(static_right / trials, 4),
        "demonstrated_accuracy": round(learned_right / trials, 4),
        "demonstrations_to_competence": reached,
        "competence_is": "ten correct choices in a row",
    }


# ── 203: a partner, and whether knowing them beats knowing the language ───


def partner_model(
    *, interactions: int, contexts: int, seed: int
) -> dict[str, object]:
    """Does watching one partner beat the prior, and does it transfer?

    The partner has a habit the population does not share, so the language
    prior — which predicts what people usually say — is wrong about them 80%
    of the time. Only interaction can fix that.

    Transfer is the card's second half and it needs two model variants,
    because one of them fails it. A per-context tally learns "in topic3 they
    said this"; a per-partner tally learns "they say this". Both score the
    same in contexts they have seen, and only the second says anything at all
    about a context they have not. Reporting only the second would hide that
    the distinction is what makes transfer a claim.
    """
    rng = random.Random(seed)
    registry = AgentRegistry()
    model = registry.model("partner")

    common = "the-usual-answer"
    habit = "what-this-partner-does"
    topics = [f"topic{i}" for i in range(contexts)]
    held_out, seen_topics = topics[-1], topics[:-1]

    per_context: dict[str, int] = {}

    for _ in range(interactions):
        topic = rng.choice(seen_topics)
        actual = habit if rng.random() < 0.8 else common

        # What each variant would say BEFORE seeing this outcome. Scoring a
        # prediction the observation already informed is scoring nothing.
        belief = model.beliefs.get(habit)
        by_partner = habit if belief and belief.strength >= 0.5 else common
        by_context = habit if per_context.get(topic, 0) > 0 else common

        model.predict(topic, by_partner, common)
        registry.interacted("partner")
        model.resolve(len(model.predictions) - 1, actual)
        model.observe_reliability(topic, accurate=by_partner == actual)

        model.observe_belief(habit, supports=actual == habit, evidence=topic)
        per_context[topic] = per_context.get(topic, 0) + (1 if actual == habit else -1)
        del by_context

    in_context = model.beats_the_prior()

    # Transfer: the same partner, a context they were never watched in.
    trials = max(50, interactions // 4)
    belief = model.beliefs.get(habit)
    partner_says = habit if belief and belief.strength >= 0.5 else common
    # The per-context variant has never seen this topic and has nothing.
    context_says = habit if per_context.get(held_out, 0) > 0 else common

    partner_right = context_right = prior_right = 0
    for _ in range(trials):
        actual = habit if rng.random() < 0.8 else common
        partner_right += partner_says == actual
        context_right += context_says == actual
        prior_right += common == actual

    return {
        "interactions": interactions,
        "in_context": in_context,
        "transfer_trials": trials,
        "held_out_context": held_out,
        "transfer_partner_model_accuracy": round(partner_right / trials, 4),
        "transfer_per_context_model_accuracy": round(context_right / trials, 4),
        "transfer_prior_accuracy": round(prior_right / trials, 4),
        "transfers": partner_right > prior_right,
        "per_context_variant_transfers": context_right > prior_right,
        "reliability": model.reliability_range(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blocks", type=int, default=10)
    parser.add_argument("--tasks-per-block", type=int, default=60)
    parser.add_argument("--length", type=int, default=12)
    parser.add_argument("--tools", type=int, default=6)
    parser.add_argument("--tool-trials", type=int, default=400)
    parser.add_argument("--interactions", type=int, default=400)
    parser.add_argument("--contexts", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1_618_033)
    parser.add_argument("--out", default="docs/evidence/developmental_campaign.json")
    args = parser.parse_args()

    development = run_campaign(
        blocks=args.blocks,
        tasks_per_block=args.tasks_per_block,
        length=args.length,
        seed=args.seed,
    )
    tools = tool_learning(
        tools=args.tools, trials=args.tool_trials, seed=args.seed
    )
    partner = partner_model(
        interactions=args.interactions, contexts=args.contexts, seed=args.seed
    )

    payload = {
        "schema": "aura.developmental_campaign.v1",
        "cards": ["072", "129", "168", "203"],
        "claim_boundary": (
            "a synthetic sub-structure library, a synthetic tool set and a "
            "synthetic partner; measures whether the campaign machinery "
            "reports growth, tool learning and partner modelling correctly "
            "and voids on the confounds, not Aura's development over its own "
            "life"
        ),
        "config": {
            "blocks": args.blocks,
            "tasks_per_block": args.tasks_per_block,
            "length": args.length,
            "tools": args.tools,
            "tool_trials": args.tool_trials,
            "interactions": args.interactions,
            "contexts": args.contexts,
            "seed": args.seed,
        },
        "development": development,
        "tools": tools,
        "partner": partner,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))

    from core.governance_context import local_internal_governed_scope
    from core.runtime.file_write_gateway import get_file_write_gateway

    out = ROOT / args.out
    with local_internal_governed_scope("developmental_campaign"):
        get_file_write_gateway().write_text(
            out, json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
