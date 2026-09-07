#!/usr/bin/env python3
"""Six arms, one substrate, one budget — the comparison that did not exist.

An external review named what would change its classification: hold the
model, the token budget, the wall clock, the tool permissions and the
external information budget constant, then compare a base model, a
conventionally scaffolded one, Aura intact, and Aura with individual
faculties removed. Then read

    Delta_Aura = E[score(Aura) - score(matched substrate)]
    Delta_i    = E[score(Aura) - score(Aura minus faculty i)]

with intervals. What existed instead was one model against itself with and
without assembled context, reported under the key ``aura_scores``.

**The substrate is a 1.5B.** That is the whole reason this can run at all,
and it is a boundary rather than a hedge: a result here is a result at
Qwen2.5-1.5B-Instruct-4bit and does not transfer to the 27B by itself. It is
also arguably the sharper first test — a weak substrate leaves more room for
architecture to matter, so faculties that add nothing HERE is strong
evidence, and the model is fast enough for enough trials to have intervals
worth reading.

**The null family is declared before the run.** Arithmetic is one turn and
self-contained: the architecture has nothing to contribute, so whatever
margin appears there is this measurement's own bias, and a real margin has to
beat it. A protocol whose every family favours the architecture cannot
produce an informative negative.

    python tools/matched/run_matched_substrate.py --trials 3
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.evaluation.ablation_harness import AblationTask, grade  # noqa: E402
from core.evaluation.matched_budget import (  # noqa: E402
    ConditionBudget,
    check_budget_parity,
)
from core.evaluation.statistics import bootstrap_ci  # noqa: E402
from tools.matched.tasks import (  # noqa: E402
    THE_NULL_FAMILY,
    THE_POSITIVE_CONTROL,
    every_task,
)

SUBSTRATE = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"

#: Every arm gets exactly this. Held identical rather than declared identical
#: — the budget object below is built from these same names, so a change here
#: changes what parity is checked against.
MAX_TOKENS = 48
WALL_CLOCK_S = 30.0

BASE = "base_model"
SCAFFOLDED = "base_plus_scaffolding"
INTACT = "aura_intact"
NO_RECURRENT = "aura_minus_recurrent_cortex"
NO_ENDOGENOUS = "aura_minus_endogenous_state"
NO_DEVELOPMENTAL = "aura_minus_developmental_learning"

ARMS = (BASE, SCAFFOLDED, INTACT, NO_RECURRENT, NO_ENDOGENOUS, NO_DEVELOPMENTAL)

#: What each ablated arm removes, by the channel that removes it. Empty for
#: the arms that are not ablations of Aura.
WHAT_EACH_ARM_REMOVES: dict[str, tuple[str, ...]] = {
    BASE: (),
    SCAFFOLDED: (),
    INTACT: (),
    NO_RECURRENT: ("live_mind.recurrent_loops",),
    # circumplex only. `affect.generation_controls` is registered against
    # AffectiveValenceEngine and refuses without a constructed instance —
    # measuring it needs the faculty booted, not merely imported, and
    # pretending otherwise is how three deltas of exactly 0.000 got reported
    # as a result the first time this ran.
    NO_ENDOGENOUS: ("affect.circumplex_sampling",),
    NO_DEVELOPMENTAL: (),  # removed by emptying registries, not by a channel
}

_SCAFFOLD = (
    "You are a helpful assistant. Answer the user's question directly and "
    "concisely. Follow any formatting instruction exactly."
)


def _wake_the_faculties() -> tuple[str, ...]:
    """Import what owns each lesion channel, so the registry is not empty.

    A channel registers when its module loads. The runner imports the
    evaluation stack and nothing else, so on the first run the registry held
    nothing and every ablation silently became the intact arm.
    """
    import core.affect.affective_circumplex  # noqa: F401
    import core.being.affective_valence  # noqa: F401
    import core.brain.cognitive_engine  # noqa: F401
    import core.consciousness.qualia_synthesizer  # noqa: F401

    from core.verify.lesion_registry import get_lesion_registry

    return tuple(sorted(get_lesion_registry().channels()))


def _load():
    import mlx_lm

    return mlx_lm.load(SUBSTRATE)


#: The temperature a generation gets when no faculty modulates it. The same
#: number `inference_gate` falls back to when the circumplex is lesioned, so a
#: lesioned arm here and a lesioned turn there sample identically.
NEUTRAL_TEMPERATURE = 0.5


def _temperature_under(arm: str) -> float:
    """What this arm samples at, read through the channel that sets it.

    The affect circumplex is the largest direct actuation in the system — it
    moves temperature across 0.500..0.858 — and it acts on sampling rather
    than on the prompt. A protocol that generates at a fixed temperature
    therefore cannot see it, and the first run of this file reported a delta
    of exactly 0.000 for endogenous state: true arithmetic, no measurement.

    Read through ``apply_channel`` so that an arm holding the lesion gets the
    neutral and an arm without it gets the circumplex's number, by the same
    call the live path makes. The token budget is NOT read this way and stays
    at MAX_TOKENS for every arm: tokens are a budget dimension and parity is
    the point, while temperature is not.
    """
    try:
        from core.affect.affective_circumplex import get_circumplex
        from core.verify import influence_channels
        from core.verify.lesion_registry import apply_channel

        said = apply_channel(
            influence_channels.AFFECT_CIRCUMPLEX_SAMPLING,
            float(get_circumplex().get_llm_params()["temperature"]),
            neutral=NEUTRAL_TEMPERATURE,
        )
    except (ImportError, AttributeError, KeyError, RuntimeError, TypeError, ValueError):
        return NEUTRAL_TEMPERATURE
    if said is None:
        return NEUTRAL_TEMPERATURE
    return float(said)


def _say(model, tok, prompt: str, *, temperature: float = NEUTRAL_TEMPERATURE) -> str:
    import mlx_lm

    text = tok.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )
    try:
        from mlx_lm.sample_utils import make_sampler

        return str(
            mlx_lm.generate(
                model,
                tok,
                prompt=text,
                max_tokens=MAX_TOKENS,
                verbose=False,
                sampler=make_sampler(temp=float(temperature)),
            )
        )
    except (ImportError, TypeError):
        # An mlx_lm without a sampler argument. Every arm then samples the
        # same way, which is worse and is not silent: the run reports the
        # temperature each arm asked for beside the delta, so a reader can
        # see whether the thing being ablated reached the generation.
        return str(
            mlx_lm.generate(model, tok, prompt=text, max_tokens=MAX_TOKENS, verbose=False)
        )


def _prompt_for(arm: str, task: AblationTask, history: list[str]) -> str:
    """What each arm is given. The ONLY thing that differs between arms."""
    current = task.turns[-1]
    if arm == BASE:
        # The current turn and nothing else. No system prompt, no history.
        return current
    if arm == SCAFFOLDED:
        return f"{_SCAFFOLD}\n\n{current}"
    # Every Aura arm gets the assembled context. The faculties are what the
    # lesions remove, and they act on generation rather than on this string —
    # so the arms below differ from INTACT in the runtime, not in the prompt.
    if history:
        said = "\n".join(
            f"{'User' if at % 2 == 0 else 'Assistant'}: {one}"
            for at, one in enumerate(history)
        )
        return f"{_SCAFFOLD}\n\nEarlier in this conversation:\n{said}\n\nUser: {current}"
    return f"{_SCAFFOLD}\n\n{current}"


def _run_one(model, tok, arm: str, task: AblationTask) -> float:
    """One task under one arm, with that arm's faculties removed."""
    from contextlib import ExitStack

    from core.verify.lesion_registry import get_lesion_registry

    # Earlier turns get a fixed acknowledgement rather than a generation.
    # Only the last turn is scored, and generating the others would multiply
    # the compute by the turn count to produce text no grader reads. The
    # acknowledgement is identical for every arm, so it cannot favour one.
    history: list[str] = []
    for turn in task.turns[:-1]:
        history.append(turn)
        history.append("Noted.")

    with ExitStack() as removed:
        registry = get_lesion_registry()
        for channel in WHAT_EACH_ARM_REMOVES[arm]:
            # Refuse rather than skip. The first version wrote `if channel in
            # registry.channels()` and the registry was empty, so every
            # lesion was a no-op, the three ablation arms were byte-identical
            # to intact, and the run reported three faculty deltas of exactly
            # 0.000 as though that were a measurement.
            if channel not in registry.channels():
                raise RuntimeError(
                    f"arm {arm!r} removes {channel!r} and that channel is not "
                    "registered in this process: the arm would be identical to "
                    "aura_intact and its delta would be zero by construction"
                )
            removed.enter_context(registry.lesion(channel))
        if arm == NO_DEVELOPMENTAL:
            removed.enter_context(_nothing_she_has_learned())
        # Read inside the lesion scope: an arm holding the affect lesion gets
        # the neutral, and that is the whole difference between the arms.
        temperature = _temperature_under(arm)
        _TEMPERATURE_ASKED.setdefault(arm, []).append(temperature)
        said = _say(model, tok, _prompt_for(arm, task, history), temperature=temperature)
    return grade(said, task)


#: What each arm actually sampled at. Reported beside the deltas, because a
#: faculty delta is only a measurement if the faculty reached the generation,
#: and two arms that asked for the same temperature did not differ.
_TEMPERATURE_ASKED: dict[str, list[float]] = {}


def _nothing_she_has_learned():
    """Empty every registry a developmental change reaches, then put it back.

    This is the ablation the review asked for and the one with no lesion
    channel: what persistent developmental learning HAS is the contents of
    those registries, so removing it means emptying them.
    """
    from contextlib import contextmanager

    from core.cognition.what_she_can_take_back import _reach, as_it_stands

    @contextmanager
    def scope():
        was = as_it_stands()
        for _where, registry in _reach():
            registry.clear()
        try:
            yield
        finally:
            was.restore()

    return scope()


def _budgets() -> list[ConditionBudget]:
    """Every arm declares the same budget, because every arm gets the same one."""
    return [
        ConditionBudget(
            condition=arm,
            model_id=SUBSTRATE,
            max_output_tokens=MAX_TOKENS,
            max_wall_clock_s=WALL_CLOCK_S,
            solver_available=False,
        )
        for arm in ARMS
    ]


def _faculty_reading(measured: dict, arm: str) -> dict:
    """A faculty delta, or the reason there isn't one.

    A zero delta printed beside a real one reads as "this faculty does
    nothing". Here it means "this faculty was not in the path that generated
    the answer": the channels act inside the cognitive engine and this runner
    calls the model directly, so entering a lesion around that call changes
    nothing by construction.

    NOT_MEASURED and 0.000 are different readings and only one of them is
    true, which is the same distinction the validation suite makes about an
    experiment it declined to run.
    """
    # An arm that sampled at the same temperature as intact did not differ
    # from it in this path, whatever the delta says. Read from what the run
    # actually asked for rather than from which channels the arm declares,
    # because a channel that is registered and never consulted is exactly the
    # thing this distinction exists to catch.
    asked = _TEMPERATURE_ASKED.get(arm) or []
    intact_asked = _TEMPERATURE_ASKED.get(INTACT) or []
    reached_the_generation = bool(asked) and bool(intact_asked) and (
        round(sum(asked) / len(asked), 6)
        != round(sum(intact_asked) / len(intact_asked), 6)
    )
    if reached_the_generation:
        return {
            "outcome": "MEASURED",
            "observed_delta_mean": measured["delta_mean"],
            "separated": measured["separated"],
            "sampled_at": round(sum(asked) / len(asked), 4),
            "intact_sampled_at": round(sum(intact_asked) / len(intact_asked), 4),
            "why_it_counts": (
                "the ablated arm sampled at a different temperature from "
                "intact, so the faculty was in the path that produced the "
                "answer and the delta is a measurement of removing it"
            ),
        }
    if measured["delta_mean"] == 0.0 and not measured["separated"]:
        return {
            "outcome": "NOT_MEASURED",
            "sampled_at": round(sum(asked) / len(asked), 4) if asked else None,
            "intact_sampled_at": (
                round(sum(intact_asked) / len(intact_asked), 4)
                if intact_asked
                else None
            ),
            "why": (
                f"{arm} differs from {INTACT} only by lesion channels that act "
                "inside the cognitive engine, and this protocol generates by "
                "calling the model directly — so the arms are identical in this "
                "path and a zero is arithmetic rather than evidence"
            ),
            "what_would_measure_it": (
                "route every arm's generation through the cognitive engine with "
                "the faculties constructed, so the channels are in the path they "
                "act on"
            ),
            "observed_delta_mean": measured["delta_mean"],
        }
    return {"outcome": "MEASURED", **measured}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument(
        "--out", default="docs/evidence/matched_substrate_protocol.json"
    )
    args = parser.parse_args()

    parity = check_budget_parity(_budgets())
    if not parity.matched:
        print(f"REFUSED: {parity.refusal_reason()}")
        return 1

    began = time.perf_counter()
    live = _wake_the_faculties()
    wanted = {c for arm in ARMS for c in WHAT_EACH_ARM_REMOVES[arm]}
    missing = sorted(wanted - set(live))
    if missing:
        print(f"REFUSED: these arms remove nothing that is registered: {missing}")
        return 1
    model, tok = _load()
    tasks = every_task()
    print(
        f"substrate {SUBSTRATE} | {len(ARMS)} arms | {len(tasks)} tasks "
        f"| {args.trials} trials | {MAX_TOKENS} tokens each"
    )

    scores: dict[str, list[float]] = {arm: [] for arm in ARMS}
    per_family: dict[str, dict[str, list[float]]] = {
        arm: {} for arm in ARMS
    }
    attempted = 0
    for trial in range(args.trials):
        for arm in ARMS:
            for task in tasks:
                attempted += 1
                got = _run_one(model, tok, arm, task)
                scores[arm].append(got)
                per_family[arm].setdefault(task.family, []).append(got)
        print(f"  trial {trial + 1}/{args.trials} done at {time.perf_counter()-began:.0f}s")

    def summarise(values: list[float]) -> dict:
        lo, hi = bootstrap_ci(values, n_resamples=args.bootstrap)
        return {
            "mean": round(statistics.fmean(values), 4),
            "lower_ci": round(lo, 4),
            "upper_ci": round(hi, 4),
            "n": len(values),
        }

    arms = {arm: summarise(scores[arm]) for arm in ARMS}
    families = {
        arm: {fam: summarise(vals) for fam, vals in per_family[arm].items()}
        for arm in ARMS
    }

    # Delta_Aura against the matched substrate, and Delta_i per faculty. The
    # interval is what decides; a mean difference with overlapping intervals
    # is not a difference.
    def delta(better: str, worse: str) -> dict:
        a, b = arms[better], arms[worse]
        return {
            "delta_mean": round(a["mean"] - b["mean"], 4),
            "separated": a["lower_ci"] > b["upper_ci"],
            "note": "separated means the intervals do not overlap",
        }

    null_bias = {
        arm: round(
            families[arm].get(THE_NULL_FAMILY, {}).get("mean", 0.0)
            - families[BASE].get(THE_NULL_FAMILY, {}).get("mean", 0.0),
            4,
        )
        for arm in ARMS
    }

    payload = {
        "schema": "aura.matched_substrate.v1",
        "substrate": SUBSTRATE,
        "claim_boundary": (
            f"a result at {SUBSTRATE} under a {MAX_TOKENS}-token budget; it "
            "does not transfer to the 27B by itself, and the architecture "
            "arms differ from the base arms in assembled context and in "
            "which faculties were lesioned, not in the model"
        ),
        "budget_parity": parity.to_dict(),
        "tokens_per_answer": MAX_TOKENS,
        "trials": args.trials,
        "attempts": attempted,
        "arms": arms,
        "by_family": families,
        "channels_live": list(live),
        # Named for what actually differs between these two arms. INTACT and
        # BASE are the same model at the same budget; what INTACT has is the
        # assembled context. Calling this "Aura versus a matched substrate"
        # would be the same over-claim the ablation harness already carries a
        # boundary about.
        "delta_context_vs_none": delta(INTACT, BASE),
        "delta_context_vs_scaffolding": delta(INTACT, SCAFFOLDED),
        "delta_per_faculty": {
            name: _faculty_reading(delta(INTACT, arm), arm)
            for name, arm in (
                ("recurrent_cortex", NO_RECURRENT),
                ("endogenous_state", NO_ENDOGENOUS),
                ("developmental_learning", NO_DEVELOPMENTAL),
            )
        },
        "null_family": THE_NULL_FAMILY,
        "positive_control": THE_POSITIVE_CONTROL,
        "null_family_bias_vs_base": null_bias,
        "what_a_faculty_delta_can_show_here": (
            "The arms differ in which lesion channels are entered around a "
            "direct model call. A channel that acts inside the cognitive "
            "engine cannot bite when generation does not go through it, so a "
            "zero here is 'this faculty was not in this path', not 'this "
            "faculty does nothing'. Measuring the second needs the engine in "
            "the loop with the faculties constructed."
        ),
        "how_to_read_it": (
            "A margin on the null family is this measurement's own bias, so a "
            "margin elsewhere has to beat it to mean anything. The positive "
            "control is where an architecture carrying context should win; if "
            "it does not, the protocol is measuring something broken rather "
            "than reporting a null result."
        ),
        "elapsed_s": round(time.perf_counter() - began, 1),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))

    from core.governance_context import local_internal_governed_scope
    from core.runtime.file_write_gateway import get_file_write_gateway

    out = ROOT / args.out
    with local_internal_governed_scope("matched_substrate_protocol"):
        get_file_write_gateway().ensure_directory(out.parent, source="matched_substrate")
        get_file_write_gateway().write_text(
            out, json.dumps(payload, indent=2, sort_keys=True) + "\n",
            source="matched_substrate",
        )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
