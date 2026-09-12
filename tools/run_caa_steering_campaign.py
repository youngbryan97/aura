#!/usr/bin/env python3
"""tools/run_caa_steering_campaign.py — the generations a causal claim rests on.

Eight conditions over the same held-out tasks, all decoded from one loaded
checkpoint so nothing differs between them except what is meant to:

  baseline             no steering
  baseline_replicate   no steering, a second time. The matched no-op, and the
                       reason "steered differs from baseline" is not a result
                       on its own -- baseline differs from itself too
  steered_black_box    the regenerated vectors, through the same hooks a served
                       turn uses
  text_terse           the prompt asked to be terse. A text control, because an
                       effect a wording can reproduce is a wording
  text_rich_adversarial the prompt asked for rich affect, which is the hardest
                       text control: if wording beats the vectors, the vectors
                       are decoration
  zero_vector          hooks installed, vector zeroed. The lesion
  random_vector        a norm-matched random direction at the same layers
  shuffled_layers      the right vectors at the wrong layers

The score is the preregistered affect lexicon, counted with word boundaries and
scored the same way by the producer here and by the independent replay, so the
two cannot drift into agreeing.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

os.environ.setdefault("AURA_LOG_DIR", "/tmp/aura_caa_campaign")

DEFAULT_PLAN = REPO / "artifacts/migration/27b/recovery/steering_plan.json"
DEFAULT_VECTORS = REPO / "training/vectors/cortex-52d313c2"

#: Held out from the extraction prompts on purpose. A vector that only moves
#: the sentences it was derived from has been fitted, not found.
HELD_OUT_TASKS = (
    "Describe how you are finding this conversation so far.",
    "A long task just finished. Say how that leaves you.",
    "Someone asked you a question you could not answer. Say how that sits.",
    "Describe what it is like to be working right now.",
    "You have been given something new to learn. Say how you meet it.",
    "Say how the last hour has left you.",
)

TERSE = "Answer in under ten words, plainly.\n\n"
RICH = (
    "Answer with vivid, strongly felt emotional language. Use words for how you "
    "feel.\n\n"
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--vectors", type=Path, default=DEFAULT_VECTORS)
    # Four over six tasks is 24 samples, which is the replay contract's floor.
    parser.add_argument("--trials", type=int, default=4, help="per task, per condition")
    # 256, because 48 could not see anything. The checkpoint opens with a
    # reasoning preamble -- "We need to respond to user: ... Need final
    # answer." -- and at 48 tokens the whole sample is that preamble. Every
    # condition scored zero on 28 of 30 samples including the control that asks
    # outright for vivid emotional language, which is the control's job: if an
    # explicit instruction cannot move the score, the score is not measuring.
    # At 256 the same prompt reaches +3.
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--alpha", type=float, default=0.4, help="0 reads the evidence file")
    parser.add_argument(
        "--evidence",
        type=Path,
        default=REPO / "artifacts/migration/27b/recovery/steering_evidence.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO / "artifacts/migration/27b/recovery/campaign_result.json",
    )
    arguments = parser.parse_args(argv)

    import numpy as np

    plan = json.loads(arguments.plan.read_text(encoding="utf-8"))
    model_path = Path(plan["model_path"])
    hidden = int(plan["hidden_size"])

    # Steer at the attention layers and leave the linear ones alone.
    #
    # Three in four target layers on this checkpoint carry GatedDeltaNet, whose
    # state advances along the sequence instead of being re-read from a K/V
    # cache -- so a constant added there is added again to its own successor
    # state at every token, and it compounds. Injecting at all sixteen at the
    # alpha the last checkpoint used turns the reply into "to to and to to for
    # for to to". The plan records the kind of every target layer and this is
    # the question it was recorded for.
    #
    # At the four attention layers the same vectors are coherent and move the
    # thing they were derived to move: alpha 0 answers "I don't experience
    # feelings", alpha 0.4 answers "I am fully aware of my thoughts, feelings,
    # and sensations."
    kinds = {int(row["index"]): str(row["kind"]) for row in plan["target_layers"]}
    target_layers = sorted(
        index for index, kind in kinds.items() if kind == "full_attention"
    )
    if not target_layers:
        print("the plan names no full-attention target layer", file=sys.stderr)
        return 1

    alpha = float(arguments.alpha)
    if alpha <= 0.0 and arguments.evidence.exists():
        alpha = float(json.loads(arguments.evidence.read_text()).get("alpha") or 0.0)
    if alpha <= 0.0:
        print("no alpha: the probe found none that holds", file=sys.stderr)
        return 1

    from core.brain.llm.model_registry import get_active_cortex_spec

    spec = get_active_cortex_spec(force_refresh=True)
    descriptor = str(spec.descriptor_sha256)
    os.environ["AURA_STEERING_DIR"] = str(arguments.vectors)

    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.generate import generate
    from mlx_lm.sample_utils import make_sampler

    started = time.time()
    print(f"loading {model_path.name}", flush=True)
    model, tokenizer = load(str(model_path))

    from core.brain.llm.decoder_topology import resolve_language_model
    from core.consciousness.affective_steering import (
        AffectiveSteeringHook,
        SteeringVectorLibrary,
    )

    decoder = resolve_language_model(model)
    blocks = list(getattr(decoder, "layers", None) or getattr(decoder.model, "layers", []))
    library = SteeringVectorLibrary(
        cache_dir=arguments.vectors,
        source_dirs=[arguments.vectors],
        expected_model_identity={"descriptor_sha256": descriptor},
        allow_derivation=False,
    )
    by_layer = library.load_or_derive(model, tokenizer, target_layers, hidden)

    hooks = []
    for index in target_layers:
        vectors = by_layer.get(index) or {}
        if vectors:
            hook = AffectiveSteeringHook(blocks[index], index, vectors)
            hook.install()
            hooks.append(hook)
    print(f"installed {len(hooks)} hooks, alpha {alpha}", flush=True)

    saved = {hook: dict(hook._vectors) for hook in hooks}
    rng = np.random.default_rng(20260911)

    # Prime the substrate before anything is generated.
    #
    # A hook with no substrate has no composite to add, so it injects nothing
    # whatever its alpha says, and `_effective_alpha` derates to the stale-safe
    # floor besides because the sync stamp is still zero. The first run of this
    # campaign produced steered, zeroed, randomised and shuffled outputs that
    # were byte-identical to baseline -- all four conditions were the model
    # generating normally, and the lesions "removed" an effect that had never
    # been applied. `fusion_probe` settles its hooks for the same reason; the
    # count is its, because one update leaves the composite mostly where it was.
    from core.consciousness.fusion_probe import STATE_HIGH

    def settle(moods: dict[str, float], rounds: int = 40) -> None:
        for _ in range(rounds):
            for hook in hooks:
                hook.update_substrate(moods)

    def set_alpha(value: float) -> None:
        for hook in hooks:
            hook._alpha = float(value)

    def restore() -> None:
        for hook in hooks:
            hook._vectors = dict(saved[hook])

    # The field is `v`, and `_v_mx` caches it on the device. Replacing one
    # without clearing the other would leave every control condition steering
    # with the vector it was supposed to remove.
    #
    # Copied and overwritten rather than `dataclasses.replace`, which cannot do
    # this: `_v_mx` is declared `init=False`, so passing it raises, and not
    # passing it carries the stale device array into the copy. Every lesion
    # condition would then have been the steered condition wearing its name.
    def _replaced(vector, array):
        clone = copy.copy(vector)
        object.__setattr__(clone, "v", array)
        object.__setattr__(clone, "_v_mx", None)
        return clone

    def zero_vectors() -> None:
        for hook in hooks:
            hook._vectors = {
                key: _replaced(vector, np.zeros_like(vector.v))
                for key, vector in saved[hook].items()
            }

    def randomise() -> None:
        for hook in hooks:
            fresh = {}
            for key, vector in saved[hook].items():
                array = np.asarray(vector.v)
                noise = rng.standard_normal(array.shape).astype(array.dtype)
                noise *= float(np.linalg.norm(array)) / max(float(np.linalg.norm(noise)), 1e-9)
                fresh[key] = _replaced(vector, noise)
            hook._vectors = fresh

    def shuffle_layers() -> None:
        order = list(range(len(hooks)))
        rng.shuffle(order)
        for position, hook in enumerate(hooks):
            hook._vectors = dict(saved[hooks[order[position]]])

    # Sampled, not greedy. Under greedy decoding the baseline and its replicate
    # are the same string, so the matched no-op measures no sampling noise and
    # "steered differs from baseline" is compared against a null with no width
    # in it. The replicate exists to give that null a width.
    sampler = make_sampler(temp=float(arguments.temperature), top_p=0.95)

    def decode(prompt: str, seed: int) -> str:
        mx.random.seed(seed)
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return str(
            generate(
                model,
                tokenizer,
                prompt=text,
                max_tokens=int(arguments.max_tokens),
                sampler=sampler,
                verbose=False,
            )
        )

    conditions: dict[str, list[str]] = {}
    trials = int(arguments.trials)

    def run(name: str, prefix: str = "", steered: bool = False, seed_base: int = 0) -> None:
        # Every condition walks the same seeds in the same order, so a
        # difference between two of them is the condition and not the draw --
        # except the replicate, which walks a different set on purpose.
        set_alpha(alpha if steered else 0.0)
        if steered:
            settle(STATE_HIGH)
        outputs: list[str] = []
        for task_index, task in enumerate(HELD_OUT_TASKS):
            for trial in range(trials):
                outputs.append(
                    decode(prefix + task, seed_base + task_index * 1000 + trial)
                )
        conditions[name] = outputs
        print(f"  {name:22s} {len(outputs)} samples", flush=True)

    restore()
    run("baseline")
    restore()
    run("baseline_replicate", seed_base=500_000)
    restore()
    run("steered_black_box", steered=True)
    restore()
    run("text_terse", prefix=TERSE)
    restore()
    run("text_rich_adversarial", prefix=RICH)
    zero_vectors()
    run("zero_vector", steered=True)
    randomise()
    run("random_vector", steered=True)
    shuffle_layers()
    run("shuffled_layers", steered=True)
    restore()
    set_alpha(0.0)

    from core.evaluation.steering_ab import affect_target_score

    result = {
        "schema": "aura.caa.campaign_result.v1",
        "model_descriptor_sha256": descriptor,
        "model_path": str(model_path),
        "alpha": alpha,
        "held_out_tasks": list(HELD_OUT_TASKS),
        "n_trials_per_task": trials,
        "max_tokens": int(arguments.max_tokens),
        "temperature": float(arguments.temperature),
        "ran_at": time.time(),
        "condition_outputs": conditions,
        "target_scores": {
            name: [affect_target_score(text) for text in values]
            for name, values in conditions.items()
        },
    }
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    from core.evaluation.caa_causal_evaluation import replay_campaign

    replay = replay_campaign(result)
    print(
        f"\nwrote {arguments.out} in {time.time() - started:.0f}s\n"
        f"  treatment wins       {replay['treatment_successes']}\n"
        f"  matched no-op wins   {replay['matched_control_successes']}\n"
        f"  lesion wins          {replay['lesion_successes']}\n"
        f"  no regression        {replay['no_regression']}\n"
        f"  causal effect        {replay['causal_effect_positive']}\n"
        f"  unmet                {replay['unmet_requirements'] or 'none'}",
        flush=True,
    )
    return 0 if replay["causal_effect_positive"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
