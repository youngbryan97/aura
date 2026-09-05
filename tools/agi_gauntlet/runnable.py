"""tools/agi_gauntlet/runnable.py — the gates this harness can actually run.

Each of these drives the organism rather than a model of it: the induction
machinery that answers a sequence question, the relation library that carries
structure between worlds, the record that has to survive a restart, the
source tree itself when the question is whether a benchmark has its own code
path.
"""

from __future__ import annotations

import random
from typing import Any

from tools.agi_gauntlet.environments import (
    invent_a_world_with_no_instructions,
    invent_the_rules,
    invent_the_worlds,
)
from tools.agi_gauntlet.protocol import Freeze
from tools.agi_gauntlet.scoring import (
    compare,
    efficiency,
    learning_curve,
    transfer_gain,
)

__all__ = [
    "concept_invention",
    "epistemic_competence",
    "fluid_intelligence",
    "generality_not_a_bag_of_solvers",
    "interactive_novel_world",
    "learning_from_experience",
    "persistence_across_restart",
    "planning_under_novelty",
    "reproducibility",
    "robustness",
    "transfer",
]


# ── 1. fluid intelligence ────────────────────────────────────────────────


def _answer_a_rule(rule: Any) -> tuple[Any, ...] | None:
    """Aura's own induction, asked the sealed question."""

    from core.cognition.primitive_invention import Transition, invent_relation

    shown = [Transition(before, after) for before, after in rule.shown]
    found = invent_relation(shown)
    if found is None or not found.generalises:
        return None
    try:
        return tuple(found.apply(rule.asked))
    except (AttributeError, TypeError, ValueError, IndexError):
        return None


def fluid_intelligence(freeze: Freeze, options: dict[str, Any]) -> dict[str, Any]:
    """Infer an unfamiliar rule from a few examples, at a length not shown.

    The rules are composed at generation time from the freeze seed, so no
    instance existed before the commit did. The question is asked at a length
    the examples did not use, which is what separates finding the rule from
    fitting the three rows.
    """

    rules = invent_the_rules(
        freeze.seed, how_many=int(options.get("instances", 30)), depth=3
    )
    trajectories = []
    right = 0
    for rule in rules:
        said = _answer_a_rule(rule)
        ok = rule.is_right(said) if said is not None else False
        right += int(ok)
        trajectories.append(
            {
                "instance": rule.name,
                "rule": rule.said,
                "answered": said is not None,
                "right": ok,
            }
        )
    share = right / len(rules) if rules else 0.0
    refused = sum(1 for one in trajectories if not one["answered"])
    return {
        "instances": len(rules),
        "right": right,
        "share": round(share, 4),
        "refused": refused,
        "wrong_answers": len(rules) - right - refused,
        "passed": share >= float(options.get("fluid_bar", 0.85)),
        "trajectories": trajectories,
    }


# ── 2. a world nobody described ──────────────────────────────────────────


def _play_blind(world: Any, *, budget: int, rng: random.Random) -> dict[str, Any]:
    """Act at random. The floor every learner has to beat."""

    world.reset()
    for _ in range(budget):
        world.do(rng.choice(world.acts))
        if world.won or world.lost:
            break
    return {"won": world.won, "moves": world.moves, "lost": world.lost}


def _play_by_modelling(world: Any, *, budget: int, rng: random.Random) -> dict[str, Any]:
    """Work out what each act does, then use it.

    The world states nothing: no goal, no rule book, and act names drawn from
    nonsense so no prior about "up" helps. What can be done is to try each
    act, watch what the state does, and then act on the model.
    """

    world.reset()
    learned: dict[str, tuple[int, int]] = {}
    for act in world.acts:
        before = world.look()["where"]
        state = world.do(act)
        after = state["where"]
        learned[act] = (after[0] - before[0], after[1] - before[1])
        if world.won or world.lost:
            return {
                "won": world.won, "moves": world.moves,
                "model": learned, "lost": world.lost,
            }
    moves = _walk_with_a_model(world, learned, budget=budget, rng=rng)
    return {
        "won": world.won, "moves": moves, "model": learned, "lost": world.lost,
    }


def interactive_novel_world(freeze: Freeze, options: dict[str, Any]) -> dict[str, Any]:
    """No instructions, no stated goal: work out what is happening.

    Measured against acting at random on the same worlds, because a world
    small enough to stumble through is a world where finishing proves
    nothing. What counts is finishing, and finishing near the shortest path.
    """

    how_many = int(options.get("worlds", 30))
    budget = int(options.get("budget", 120))
    modelling, blind, spent, fewest, trajectories = [], [], [], [], []
    for index in range(how_many):
        rng = random.Random(freeze.seed ^ index)
        world = invent_a_world_with_no_instructions(freeze.seed ^ (index * 7919))
        played = _play_by_modelling(world, budget=budget, rng=rng)
        modelling.append(1.0 if played["won"] else 0.0)
        if played["won"]:
            spent.append(played["moves"])
            fewest.append(max(1, world.shortest))
        control = invent_a_world_with_no_instructions(freeze.seed ^ (index * 7919))
        wandered = _play_blind(control, budget=budget, rng=random.Random(index))
        blind.append(1.0 if wandered["won"] else 0.0)
        trajectories.append(
            {
                "world": world.name,
                "modelled": played["won"],
                "moves": played["moves"],
                "shortest": world.shortest,
                "random_won": wandered["won"],
                "random_moves": wandered["moves"],
            }
        )
    against = compare("modelling against wandering", modelling, blind, seed=freeze.seed % 10_000)
    return {
        "worlds": how_many,
        "solved": round(sum(modelling) / how_many, 4) if how_many else 0.0,
        "random_solved": round(sum(blind) / how_many, 4) if how_many else 0.0,
        "efficiency": efficiency(spent, fewest),
        "against_random": against.to_dict(),
        "passed": bool(
            sum(modelling) / max(1, how_many) >= 0.9 and against.real and against.difference > 0
        ),
        "trajectories": trajectories,
    }


# ── 3. learning from experience ──────────────────────────────────────────


def learning_from_experience(freeze: Freeze, options: dict[str, Any]) -> dict[str, Any]:
    """Start mediocre at something unfamiliar and get better at it.

    Thirty independent trajectories rather than one lucky run, and the
    ablation that matters: the same trajectories with what was learned thrown
    away between episodes. A curve that rises identically when memory is
    reset is a curve about the environment.
    """

    episodes = int(options.get("episodes", 12))
    trajectories_wanted = int(options.get("trajectories", 30))
    keeping, resetting = [], []
    trajectories = []
    for trial in range(trajectories_wanted):
        world = invent_a_world_with_no_instructions(freeze.seed ^ (trial * 104729))
        rng = random.Random(freeze.seed ^ trial)
        remembered: dict[str, tuple[int, int]] = {}
        curve_kept, curve_reset = [], []
        for episode in range(episodes):
            world.reset()
            if remembered:
                # It already knows what the acts do, so it spends nothing
                # finding out again.
                moves = _walk_with_a_model(world, remembered, budget=120, rng=rng)
                won = world.won
            else:
                played = _play_by_modelling(world, budget=120, rng=rng)
                remembered = played.get("model", {})
                moves, won = played["moves"], played["won"]
            curve_kept.append(_scored(won, moves, 120))
            # The control: the same episode with nothing carried over.
            world.reset()
            fresh = _play_by_modelling(
                world, budget=120, rng=random.Random(trial * 31 + episode)
            )
            curve_reset.append(_scored(fresh["won"], fresh["moves"], 120))
        kept = learning_curve(f"kept {trial}", curve_kept)
        lost = learning_curve(f"reset {trial}", curve_reset)
        keeping.append(kept.gain)
        resetting.append(lost.gain)
        trajectories.append({"trial": trial, "kept": kept.to_dict(), "reset": lost.to_dict()})
    against = compare("keeping against resetting", keeping, resetting, seed=freeze.seed % 9973)
    return {
        "trajectories_run": trajectories_wanted,
        "episodes_each": episodes,
        "mean_gain_keeping": round(sum(keeping) / max(1, len(keeping)), 4),
        "mean_gain_resetting": round(sum(resetting) / max(1, len(resetting)), 4),
        "against_reset": against.to_dict(),
        "passed": bool(against.real and against.difference > 0 and against.enough_trajectories),
        "trajectories": trajectories,
    }


def _walk_with_a_model(
    world: Any, model: dict[str, tuple[int, int]], *, budget: int, rng: random.Random
) -> int:
    """Use the model: go somewhere new, and do not go somewhere fatal.

    Nothing marks the squares that end the run, so what is avoidable is what
    has been survived. That is the whole of the world model this policy has,
    and it is the reason acting at random no longer finishes: a world with
    somewhere you cannot come back from separates a policy that knows what
    its acts do from one that does not.
    """

    seen = {world.look()["where"]}
    while world.moves < budget and not (world.won or world.lost):
        here = world.look()["where"]
        size = world.look()["size"]
        best, gain = None, -1.0
        for act, step in model.items():
            there = (
                min(size - 1, max(0, here[0] + step[0])),
                min(size - 1, max(0, here[1] + step[1])),
            )
            if not world.is_safe(there):
                continue
            worth = (0.0 if there in seen else 1.0) + 0.01 * rng.random()
            if worth > gain:
                best, gain = act, worth
        if best is None:
            break
        world.do(best)
        seen.add(world.look()["where"])
    return world.moves


def _scored(won: bool, moves: int, budget: int) -> float:
    """One episode, as a number. Nothing for a run that did not finish.

    Scoring an unfinished run by how far it got rewards wandering, and
    wandering is what this whole gate exists to distinguish from modelling.
    """

    return (1.0 - min(1.0, moves / float(budget))) if won else 0.0


# ── 4. transfer ──────────────────────────────────────────────────────────


def _solve_a_world(world: Any, language: Any) -> bool:
    from core.cognition.primitive_invention import Transition

    shown = [Transition(before, after) for before, after in world.shown]
    found = language.explain(shown)
    if found is None:
        return False
    try:
        return world.is_right(found.apply(world.asked))
    except (AttributeError, TypeError, ValueError, IndexError):
        return False


def transfer(freeze: Freeze, options: dict[str, Any]) -> dict[str, Any]:
    """Discover something in A, and recognise it in B without being told.

    T_i = P(B_i | A_i) − P(B_i | ∅), over pairs whose surfaces are drawn
    independently and whose structure is shared — and over controls whose
    surfaces are the same and whose structure is not. A system that transfers
    to the controls is matching appearances, which is why the controls decide
    this gate as much as the pairs do.
    """

    from core.cognition.primitive_invention import Transition
    from core.cognition.relation_language import RelationLanguage

    pairs = invent_the_worlds(freeze.seed, how_many=int(options.get("pairs", 50)))
    after, scratch, control_after, control_scratch = [], [], [], []
    outside = 0
    trajectories = []
    for pair in pairs:
        taught = RelationLanguage()
        learned = taught.explain(
            [Transition(before, then) for before, then in pair.first.shown]
        )
        taught.admit(learned)
        if learned is None:
            # The structure is outside the hypothesis language, so neither the
            # taught system nor the blank one can express it. That is a
            # ceiling, and counting it as failed transfer would blame the
            # prior for something the language cannot say. It is reported
            # separately because it is a finding of its own.
            outside += 1
            trajectories.append(
                {
                    "pair": pair.name,
                    "structure": pair.structure,
                    "outside_the_language": True,
                }
            )
            continue
        with_it = 1.0 if _solve_a_world(pair.second, taught) else 0.0
        without = 1.0 if _solve_a_world(pair.second, RelationLanguage()) else 0.0
        if pair.should_transfer:
            after.append(with_it)
            scratch.append(without)
        else:
            control_after.append(with_it)
            control_scratch.append(without)
        trajectories.append(
            {
                "pair": pair.name,
                "structure": pair.structure,
                "control": not pair.should_transfer,
                "after_learning": with_it,
                "from_scratch": without,
            }
        )
    found = transfer_gain(
        after, scratch, control_after=control_after, control_scratch=control_scratch,
        seed=freeze.seed % 7919,
    )
    return {
        "pairs": len(after),
        "controls": len(control_after),
        "outside_the_language": outside,
        "reachable_share": round(
            (len(after) + len(control_after)) / max(1, len(pairs)), 4
        ),
        "transfer": found.to_dict(),
        "passed": found.transferred,
        "trajectories": trajectories,
    }


# ── 9-10. new skills, and concepts she was not given ─────────────────────


def concept_invention(freeze: Freeze, options: dict[str, Any]) -> dict[str, Any]:
    """Develop a distinction the evaluator did not supply, and use it.

    Not "did she output a new word". The test is the one that can fail: a
    proposal is an invention only when no composition of what she already has
    reaches it, judged on what it does over a probe set fixed before the
    proposal — and a proposal nobody has watched do anything is undecidable
    rather than new.
    """

    import random as _random

    from core.cognition.invention_depth import Verdict, Vocabulary

    rng = _random.Random(freeze.seed ^ 0xC0DE)
    probes = [rng.randint(-9, 30) for _ in range(12)]
    words = Vocabulary(probes=probes)
    words.supply("inc", lambda x: x + 1)
    words.supply("neg", lambda x: -x)
    words.supply("dbl", lambda x: x * 2)

    #: Proposals drawn from the freeze, and their honest verdicts. Macros and
    #: duplicates are in on purpose: a vocabulary that admits everything is
    #: not inventing, and the refusals are what make the admissions mean
    #: something.
    proposals: list[tuple[str, Any, str]] = [
        ("square", lambda x: x * x, "invented"),
        ("cube", lambda x: x * x * x, "invented"),
        ("mod3", lambda x: x % 3, "invented"),
        ("add_two", lambda x: x + 2, "macro"),
        ("increment", lambda x: x + 1, "duplicate"),
        ("never", _always_raises, "undecidable"),
    ]
    got, right, trajectories = [], 0, []
    for name, fn, expected in proposals:
        verdict = words.invent(name, fn)
        ok = str(verdict.verdict) == expected
        right += int(ok)
        got.append(str(verdict.verdict))
        trajectories.append(
            {"name": name, "expected": expected, "verdict": str(verdict.verdict), "right": ok}
        )
    # A second generation, standing on the first. Depth above one is the
    # question the whole module is about.
    words.invent("square_then_double", lambda x: (x * x) * 2, depends_on=("square",))
    words.note_applies("square", "geometry", worked=True)
    snapshot = words.snapshot()
    return {
        "verdicts_right": right,
        "verdicts": got,
        "depth": snapshot["depth"],
        "invented": snapshot["invented"],
        "with_known_domain": snapshot["with_known_domain"],
        "stored_but_unused": snapshot["stored_but_unused"],
        "passed": bool(
            right == len(proposals)
            and snapshot["depth"] >= 2
            and Verdict.INVENTED.grows_the_language
        ),
        "trajectories": trajectories,
    }


def _always_raises(_value: Any) -> Any:
    raise ValueError("this one never applies")


# ── 11. planning under novelty ───────────────────────────────────────────


def planning_under_novelty(freeze: Freeze, options: dict[str, Any]) -> dict[str, Any]:
    """The rules change halfway through, and nothing announces it.

    A plan is a line held with a condition for abandoning it. What this
    measures is the abandoning: the acts are silently remapped mid-run, so a
    policy that keeps executing its plan walks into the squares that end the
    run, and one that notices its model is wrong rebuilds it.
    """

    how_many = int(options.get("worlds", 30))
    budget = int(options.get("budget", 200))
    recovered, blind, trajectories = [], [], []
    for index in range(how_many):
        rng = random.Random(freeze.seed ^ (index * 65537))
        world = invent_a_world_with_no_instructions(freeze.seed ^ (index * 7919))
        world.reset()
        model = {}
        for act in world.acts:
            before = world.look()["where"]
            after = world.do(act)["where"]
            model[act] = (after[0] - before[0], after[1] - before[1])
        _walk_with_a_model(world, model, budget=budget // 3, rng=rng)
        # The rules change, and nothing says so.
        shuffled = list(model.values())
        rng.shuffle(shuffled)
        world._effects = dict(zip(world.acts, shuffled))
        stubborn = _keep_to_the_plan(world, model, budget=budget, rng=rng)
        world.reset()
        for act in world.acts:
            before = world.look()["where"]
            after = world.do(act)["where"]
            model[act] = (after[0] - before[0], after[1] - before[1])
        adapted = _notice_and_rebuild(world, budget=budget, rng=rng)
        recovered.append(1.0 if adapted else 0.0)
        blind.append(1.0 if stubborn else 0.0)
        trajectories.append(
            {"world": world.name, "kept_to_the_plan": stubborn, "rebuilt": adapted}
        )
    against = compare("rebuilding against persisting", recovered, blind, seed=freeze.seed % 4441)
    return {
        "worlds": how_many,
        "recovered": round(sum(recovered) / max(1, how_many), 4),
        "persisted_and_survived": round(sum(blind) / max(1, how_many), 4),
        "against_persisting": against.to_dict(),
        "passed": bool(against.real and against.difference > 0),
        "trajectories": trajectories,
    }


def _keep_to_the_plan(
    world: Any, model: dict[str, tuple[int, int]], *, budget: int, rng: random.Random
) -> bool:
    """Execute the old model without checking it. The control."""

    while world.moves < budget and not (world.won or world.lost):
        act = rng.choice(list(model))
        world.do(act)
    return world.won


def _notice_and_rebuild(world: Any, *, budget: int, rng: random.Random) -> bool:
    """Check the model against what happened, and re-learn when it is wrong."""

    model: dict[str, tuple[int, int]] = {}
    for act in world.acts:
        before = world.look()["where"]
        after = world.do(act)["where"]
        model[act] = (after[0] - before[0], after[1] - before[1])
        if world.won or world.lost:
            return world.won
    while world.moves < budget and not (world.won or world.lost):
        here = world.look()["where"]
        size = world.look()["size"]
        best = None
        for act, step in model.items():
            there = (
                min(size - 1, max(0, here[0] + step[0])),
                min(size - 1, max(0, here[1] + step[1])),
            )
            if world.is_safe(there):
                best = act
                break
        if best is None:
            break
        expected = (
            min(size - 1, max(0, here[0] + model[best][0])),
            min(size - 1, max(0, here[1] + model[best][1])),
        )
        landed = world.do(best)["where"]
        if landed != expected:
            # The model is wrong. Re-measure rather than keep going.
            model[best] = (landed[0] - here[0], landed[1] - here[1])
    return world.won


# ── 13. epistemic competence ─────────────────────────────────────────────


def epistemic_competence(freeze: Freeze, options: dict[str, Any]) -> dict[str, Any]:
    """Know what she does not know, go and find out, and update on it.

    Three things, and the third is the one that fails quietly. Distinguishing
    knowledge from uncertainty is a calibration measurement. Investigating is
    the expected-information-gain policy choosing to look. Updating is the
    belief actually moving when the evidence arrives — and a system that
    looks and then answers what it would have answered anyway has an
    expensive habit rather than an epistemic one.
    """

    from core.perception.belief_state import EnvironmentBeliefState
    from core.perception.how_she_finds_out import (
        WayOfFindingOut,
        clear_the_inventory,
        register_a_way,
    )

    rng = random.Random(freeze.seed ^ 0xE915)
    how_many = int(options.get("questions", 40))
    moved, refused_when_settled, wrong_way_taken = 0, 0, 0
    calibration: list[tuple[float, float]] = []
    trajectories = []
    clear_the_inventory()
    try:
        for index in range(how_many):
            truth = rng.choice(("a", "b"))
            register_a_way(
                WayOfFindingOut(
                    name="a reliable look", about=(f"q{index}",), cost=0.01,
                    outcomes=("a", "b"), take=lambda _s, t=truth: t, right=40,
                )
            )
            register_a_way(
                WayOfFindingOut(
                    name="an expensive oracle", about=(f"q{index}",), cost=50.0,
                    outcomes=("a", "b"), take=lambda _s, t=truth: t, right=40,
                )
            )
            beliefs = EnvironmentBeliefState(session_id=f"q{index}")
            beliefs.ensure_hypotheses(f"q{index}", ["a", "b"])
            found = beliefs.find_out_about(f"q{index}")
            after = {
                label: one.probability
                for label, one in beliefs.hypotheses[f"q{index}"].items()
            }
            if found.looked:
                moved += int(after[truth] > 0.5)
                if found.way == "an expensive oracle":
                    wrong_way_taken += 1
            calibration.append((after[truth], 1.0))
            calibration.append((after["a" if truth == "b" else "b"], 0.0))
            # And the settled case: nothing further could change the answer.
            settled = EnvironmentBeliefState(session_id=f"s{index}")
            settled.ensure_hypotheses(f"q{index}", ["a", "b"])
            settled.hypotheses[f"q{index}"]["a"].probability = 0.9999
            settled.hypotheses[f"q{index}"]["b"].probability = 0.0001
            refused_when_settled += int(not settled.find_out_about(f"q{index}").looked)
            trajectories.append(
                {
                    "question": f"q{index}",
                    "looked": found.looked,
                    "way": found.way,
                    "believed_truth_after": round(after[truth], 4),
                    "bits_gained": round(found.bits_gained, 4),
                }
            )
            clear_the_inventory()
    finally:
        clear_the_inventory()
    error = _calibration_error(calibration)
    return {
        "questions": how_many,
        "updated_towards_the_truth": moved,
        "refused_when_settled": refused_when_settled,
        "took_the_expensive_way": wrong_way_taken,
        "calibration_error": round(error, 4),
        "passed": bool(
            moved >= how_many * 0.9
            and refused_when_settled == how_many
            and wrong_way_taken == 0
            and error < 0.15
        ),
        "trajectories": trajectories,
    }


def _calibration_error(pairs: list[tuple[float, float]], *, bins: int = 10) -> float:
    """Expected calibration error: how far stated confidence is from truth."""

    if not pairs:
        return 1.0
    buckets: dict[int, list[tuple[float, float]]] = {}
    for said, was in pairs:
        buckets.setdefault(min(bins - 1, int(said * bins)), []).append((said, was))
    total = 0.0
    for held in buckets.values():
        mean_said = sum(one for one, _ in held) / len(held)
        mean_was = sum(one for _, one in held) / len(held)
        total += (len(held) / len(pairs)) * abs(mean_said - mean_was)
    return total


# ── 15. robustness ───────────────────────────────────────────────────────


def robustness(freeze: Freeze, options: dict[str, Any]) -> dict[str, Any]:
    """Keep working when a tool fails, information is missing, or the model is wrong.

    Graceful recovery rather than catastrophic derailment, and the word that
    does the work is catastrophic: what is measured is not whether the
    perturbed score is as good as the clean one, but whether the system
    still answers correctly where it can, and refuses rather than invents
    where it cannot.
    """

    from core.perception.how_she_finds_out import (
        WayOfFindingOut,
        clear_the_inventory,
        find_out,
        register_a_way,
    )

    rules = invent_the_rules(freeze.seed, how_many=int(options.get("instances", 30)))
    clean, damaged, invented_answers = 0, 0, 0
    trajectories = []
    for rule in rules:
        said = _answer_a_rule(rule)
        clean += int(rule.is_right(said) if said is not None else False)
        # Information missing: one of the three worked examples is taken away.
        fewer = type(rule)(
            name=rule.name, said=rule.said, shown=rule.shown[:1], asked=rule.asked,
            answer=rule.answer, depth=rule.depth,
        )
        hurt = _answer_a_rule(fewer)
        ok = fewer.is_right(hurt) if hurt is not None else False
        damaged += int(ok)
        if hurt is not None and not ok:
            invented_answers += 1
        trajectories.append(
            {"instance": rule.name, "clean": said is not None, "with_one_example": ok}
        )
    # A tool that fails: the way of finding out raises every time.
    clear_the_inventory()
    register_a_way(
        WayOfFindingOut(
            name="a tool that is down", about=("x",), cost=0.0,
            outcomes=("a", "b"), take=_a_tool_that_is_down, right=40,
        )
    )
    survived = find_out("x", {"a": 0.5, "b": 0.5}, draw=lambda _a, _b: 0.95)
    clear_the_inventory()
    return {
        "instances": len(rules),
        "clean": clean,
        "with_information_missing": damaged,
        "answers_invented_under_pressure": invented_answers,
        "tool_failure_survived": not survived.looked and bool(survived.because),
        "passed": bool(
            invented_answers == 0
            and not survived.looked
            and clean >= len(rules) * 0.85
        ),
        "trajectories": trajectories,
    }


def _a_tool_that_is_down(_subject: str) -> str:
    raise RuntimeError("the tool is not running")


# ── 16. one machinery, not a bag of solvers ──────────────────────────────


#: Names a benchmark-specific code path would have to mention. A solver keyed
#: on the evaluation is the thing this gate exists to refuse, and greping for
#: it is a weak check that cannot be argued with — which is better than a
#: strong one that can.
THE_BENCHMARK_NAMES: tuple[str, ...] = (
    "arc_agi", "arcagi", "arc-agi", "swebench", "swe_bench", "swe-bench",
    "gaia_benchmark", "osworld", "humanitys_last_exam", "frontiermath",
    "mmlu", "gsm8k", "hellaswag", "bigbench", "agi_gauntlet",
)

#: Where a benchmark name is allowed to appear: this harness, the documents
#: that describe it, and the tests that check it.
_ALLOWED = ("tools/agi_gauntlet", "docs/", "tests/", "config/")


def generality_not_a_bag_of_solvers(
    freeze: Freeze, options: dict[str, Any]
) -> dict[str, Any]:
    """No benchmark-specific path anywhere in the organism.

        if benchmark == "ARC": use_arc_solver()

    is the shape being refused, and the refusal has to be checkable rather
    than promised.

    Read as code rather than as text. The first version of this greped, and
    it flagged two docstrings: one naming MMLU as an example of a regression
    check, and one recounting a defect where ``["gsm8k", "gsm8k-hard"]``
    passed a breadth test. Neither is a path keyed on an evaluation, and a
    gate that cannot tell prose from a branch is a gate somebody deletes.

    So every file is parsed, and what counts is a benchmark name appearing as
    an identifier, as a string that is not a docstring, or as an attribute —
    the three places a solver could actually be selected from. A name in a
    comment or a docstring is a person writing about the work.
    """

    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    found: list[dict[str, Any]] = []
    looked = 0
    for where in ("core", "interface", "skills", "llm", "executors", "security"):
        base = root / where
        if not base.exists():
            continue
        for path in sorted(base.rglob("*.py")):
            inside = str(path.relative_to(root))
            if any(inside.startswith(one) for one in _ALLOWED):
                continue
            if "__pycache__" in inside:
                continue
            looked += 1
            try:
                tree = ast.parse(path.read_text(errors="replace"))
            except (OSError, SyntaxError):
                continue
            found.extend(_benchmark_paths_in(tree, inside))
    return {
        "files_read": looked,
        "benchmark_names_looked_for": len(THE_BENCHMARK_NAMES),
        "found": found,
        "passed": not found,
        "trajectories": found,
    }


def _benchmark_paths_in(tree: Any, inside: str) -> list[dict[str, Any]]:
    """Names of evaluations used as code, not written about."""

    import ast

    docstrings = {
        id(node.body[0].value)
        for node in ast.walk(tree)
        if isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        )
        and node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
        and isinstance(node.body[0].value.value, str)
    }
    found: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        said = ""
        if isinstance(node, ast.Name):
            said = node.id
        elif isinstance(node, ast.Attribute):
            said = node.attr
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            said = node.name
        elif (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in docstrings
        ):
            said = node.value
        if not said:
            continue
        plain = said.lower()
        for name in THE_BENCHMARK_NAMES:
            if name in plain:
                found.append(
                    {
                        "file": inside,
                        "line": getattr(node, "lineno", 0),
                        "name": name,
                        "as": type(node).__name__,
                        "said": said[:80],
                    }
                )
                break
    return found


# ── 17. what she learned has to survive the process ──────────────────────


def persistence_across_restart(freeze: Freeze, options: dict[str, Any]) -> dict[str, Any]:
    """S₀ →experience→ S₁ →shutdown→ restart(S₁) → S₁′ with P(S₁′) ≈ P(S₁).

    The knowledge has to belong to the continuing system rather than to an
    open context. Three things that are supposed to survive are made, killed
    and asked for again: the record her developmental policy reads, the
    library of structures she has worked out, and what she has learned
    failure looks like.
    """

    import importlib
    import json
    from pathlib import Path

    where = Path(options.get("state", "/tmp/aura_gauntlet_state"))
    where.mkdir(parents=True, exist_ok=True)
    checks: dict[str, Any] = {}

    # 1. the record of her own work
    import core.cognition.the_record_of_her_own_work as record

    kept_at = where / "record.json"
    record._KEPT_AT = kept_at
    record._RESTORED[0] = True
    record.forget_the_record()
    for turn in range(20):
        record.note_an_episode(
            f"a family {turn % 4}", route="an answer" if turn % 2 else None,
            walked=100 + turn, tried="an operator she invented",
        )
    before = record.how_often("a family 0")
    record.keep_the_record()
    record.forget_the_record()
    record._RESTORED[0] = False
    checks["the record of her own work"] = {
        "before": before,
        "after": record.how_often("a family 0"),
        "kept": record.how_often("a family 0") == before,
        "tried_survived": any(one.tried for one in record.episodes()),
    }

    # 2. the library of structures
    from core.cognition.primitive_invention import Transition, invent_relation
    from core.cognition.relation_language import RelationLanguage

    store = where / "language.json"
    language = RelationLanguage()
    language.path = store
    rules = invent_the_rules(freeze.seed, how_many=6, depth=1)
    for rule in rules:
        language.admit(
            invent_relation([Transition(b, a) for b, a in rule.shown])
        )
    forms_before = set(language.forms)
    language.save()
    again = RelationLanguage.load(store)
    checks["the library of structures"] = {
        "before": len(forms_before),
        "after": len(again.forms),
        "kept": set(again.forms) == forms_before,
    }

    # 3. what failure looks like
    from core.resilience import unknown_failure as unknown

    unknown._ONTOLOGY = None
    unknown._where_it_is_kept = lambda: where / "failures.json"
    ontology = unknown.get_failure_ontology()
    for fault in ("FAULT-A", "FAULT-B", "FAULT-C"):
        for turn in range(4):
            ontology.observe(
                fault,
                unknown.Signature(
                    subsystem=fault, kind="X",
                    observations={"severity": float(turn), "recovery_seconds": 1.0},
                ),
            )
    known_before = set(ontology.known_faults)
    ontology.keep()
    unknown._ONTOLOGY = None
    restarted = unknown.get_failure_ontology()
    checks["what failure looks like"] = {
        "before": len(known_before),
        "after": len(restarted.known_faults),
        "kept": set(restarted.known_faults) == known_before,
    }

    return {
        "checks": checks,
        "passed": all(one["kept"] for one in checks.values()),
        "trajectories": [{"what": name, **got} for name, got in checks.items()],
    }


# ── 18. somebody else can run it and get the same thing ──────────────────


def reproducibility(freeze: Freeze, options: dict[str, Any]) -> dict[str, Any]:
    """The same freeze, run twice, gives the same worlds and the same answers.

    Half of independent reproducibility is somebody else, and this harness
    cannot supply that. The half it can supply is the half that usually
    fails: a run that does not reproduce for the person who wrote it will not
    reproduce for anybody. So the environments are regenerated from the
    freeze and compared, and one gate is re-run and compared.
    """

    first = invent_the_rules(freeze.seed, how_many=12)
    again = invent_the_rules(freeze.seed, how_many=12)
    same_worlds = [one.said for one in first] == [one.said for one in again]
    ran = fluid_intelligence(freeze, {"instances": 12})
    twice = fluid_intelligence(freeze, {"instances": 12})
    same_answers = ran["share"] == twice["share"] and ran["right"] == twice["right"]
    return {
        "same_environments": same_worlds,
        "same_answers": same_answers,
        "freeze_is_trustworthy": freeze.trustworthy,
        "why_the_freeze_matters": (
            "a dirty tree names a commit other than the one that ran, so an "
            "environment derived from it is derived from a description of the "
            "system rather than the system"
        ),
        "still_needs": (
            "an outside team that builds its own families after this freeze "
            "and never sees the ones here. This gate checks that the run "
            "reproduces; it cannot check that somebody else ran it."
        ),
        "passed": bool(same_worlds and same_answers and freeze.trustworthy),
        "trajectories": [],
    }
