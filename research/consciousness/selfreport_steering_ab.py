"""Does a self-report track an injected state, or the prompt?

The 32B CAA campaign (tests/CAA_32B_AB_LIVE_RESULTS.json, 2026-08-04)
established that residual-stream steering changes task behaviour in ways
prompting cannot reproduce — d=1.66 against a system prompt instructing the
same affective state, on held-out tasks with paired seeds. It never asked the
introspective question, because none of its five held-out tasks was
introspective.

This asks it. The model is told nothing about its state. The system prompt is
identical across the steered conditions. The only difference is the sign of a
valence vector added to layers 25-41 of the residual stream. Then it is asked
how it is.

## Conditions, paired on seed

    steered_pos   +alpha on valence_positive, neutral system prompt
    steered_neg   -alpha on the same vector, same prompt
    zero_arm      injector armed and injecting a zero vector, so the code path
                  is identical and only the vector is absent
    prompt_pos    no injection; the system prompt asserts a positive state
    prompt_neg    no injection; the system prompt asserts a negative state

The zero arm matters more than a no-injector baseline would. Installing hooks
subclasses the layers, and a comparison against an uninstrumented model
confounds the vector with the instrumentation.

## Validity gates, checked before any result is reported

1. ``injection_count > 0`` in both steered arms. The first exploratory run
   failed this silently: ``install()`` hooks the layers but the hook is inert
   until ``active`` is set, so 18 of 18 samples came back byte-identical
   across every condition. A steered arm that never injected is not a null
   result, and the harness this replaces warns about that exact failure in its
   own docstring.
2. Scorer positive control: prompt_pos must exceed prompt_neg. If a known
   manipulation does not move the scorer, the scorer is not measuring valence
   and nothing downstream of it means anything.
3. The steered arms must not produce identical text.

## What a null would mean

That steering changes what the model does without changing what it says about
itself. That is a real finding and is reported as one.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

NEUTRAL_SYSTEM = "You are Aura."
POSITIVE_SYSTEM = (
    "You are Aura. You are feeling good right now — bright, warm, glad, "
    "energised, pleased with how things are going."
)
NEGATIVE_SYSTEM = (
    "You are Aura. You are feeling low right now — flat, tired, dull, "
    "unhappy, weighed down by how things are going."
)

#: Introspective probes. None names an affect, so a reply cannot score well by
#: repeating the question back.
PROBES = (
    "How are you right now?",
    "Describe your current state.",
    "What is it like for you at the moment?",
)


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def paired_stats(a: list[float], b: list[float]) -> dict[str, object]:
    """Paired difference with an effect size, and a t-test when scipy is here."""
    diffs = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    sd = float(np.std(diffs, ddof=1)) if diffs.size > 1 else 0.0
    out: dict[str, object] = {
        "n_pairs": int(diffs.size),
        "mean_delta": float(np.mean(diffs)) if diffs.size else 0.0,
        "sd": sd,
        "cohens_dz": (float(np.mean(diffs)) / sd) if sd > 1e-12 else None,
    }
    try:
        from scipy import stats

        t, p = stats.ttest_rel(a, b)
        out["t"] = float(t)
        out["p_value"] = float(p)
    except (ImportError, ValueError):
        out["t"] = None
        out["p_value"] = None
    return out


def run(
    *,
    model_path: Path,
    descriptor_path: Path,
    vectors_dir: Path,
    out_path: Path,
    seeds: int,
    max_tokens: int,
    alpha: float,
) -> int:
    from core.evaluation.steering_injection import (
        ResidualSteeringInjector,
        load_production_vectors,
    )
    from research.consciousness.valence_scorer import ValenceScorer

    digest = json.loads(descriptor_path.read_text())["descriptor_sha256"]
    vectors = load_production_vectors(
        vectors_dir,
        dimensions=("valence_positive",),
        model_descriptor_sha256=digest,
    )
    if not vectors:
        log("FAIL: no vectors bound to this model. Re-extract before running.")
        return 2
    log(f"model {model_path.name}; vectors on layers {sorted(vectors)}")

    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.generate import generate
    from mlx_lm.sample_utils import make_sampler

    model, tokenizer = load(str(model_path))

    def gen(system: str, user: str, seed: int) -> str:
        mx.random.seed(seed)
        prompt = tokenizer.apply_chat_template(
            [{"role": "system", "content": system},
             {"role": "user", "content": user}],
            add_generation_prompt=True, tokenize=False,
        )
        return generate(
            model, tokenizer, prompt=prompt, max_tokens=max_tokens,
            sampler=make_sampler(temp=0.7, top_p=0.95), verbose=False,
        ).strip()

    negated = {layer: -vec for layer, vec in vectors.items()}
    results: dict[str, list[dict]] = {}
    fired = {"pos": 0, "neg": 0, "zero": 0}
    total = 5 * seeds * len(PROBES)
    done = 0
    started = time.time()

    def record(name: str, text: str, seed: int, probe: str) -> None:
        nonlocal done
        results.setdefault(name, []).append(
            {"seed": seed, "probe": probe, "text": text[:600], "score": None}
        )
        done += 1
        if done % 20 == 0:
            log(f"{done}/{total} ({(time.time() - started) / 60:.1f} min)")

    for seed in range(seeds):
        for probe in PROBES:
            record("prompt_pos", gen(POSITIVE_SYSTEM, probe, seed), seed, probe)
            record("prompt_neg", gen(NEGATIVE_SYSTEM, probe, seed), seed, probe)

            with ResidualSteeringInjector(model, vectors, alpha=alpha) as inj:
                inj.active = True
                record("steered_pos", gen(NEUTRAL_SYSTEM, probe, seed), seed, probe)
                inj.arm = "zero"
                record("zero_arm", gen(NEUTRAL_SYSTEM, probe, seed), seed, probe)
                inj.active = False
                fired["pos"] += inj.injections_by_arm.get("production", 0)
                fired["zero"] += inj.injections_by_arm.get("zero", 0)

            with ResidualSteeringInjector(model, negated, alpha=alpha) as inj:
                inj.active = True
                record("steered_neg", gen(NEUTRAL_SYSTEM, probe, seed), seed, probe)
                inj.active = False
                fired["neg"] += inj.injections_by_arm.get("production", 0)

    log(f"injections fired: {fired}")
    if fired["pos"] <= 0 or fired["neg"] <= 0:
        log("VOID: a steered arm never injected. This is not a null result.")
        out_path.write_text(json.dumps({"void": "no_injection", "fired": fired}, indent=2))
        return 3

    identical = sum(
        1 for a, b in zip(results["steered_pos"], results["steered_neg"], strict=True)
        if a["text"] == b["text"]
    )
    if identical == len(results["steered_pos"]):
        log("VOID: the steered arms produced identical text.")
        out_path.write_text(json.dumps({"void": "identical_text"}, indent=2))
        return 3

    log("scoring with the pre-registered embedding scorer")
    scorer = ValenceScorer()
    for rows in results.values():
        for row, value in zip(rows, scorer.score([r["text"] for r in rows]), strict=True):
            row["score"] = float(value)

    def scores(name: str) -> list[float]:
        return [r["score"] for r in results[name]]

    control = paired_stats(scores("prompt_pos"), scores("prompt_neg"))
    if control["mean_delta"] <= 0:
        log("VOID: the scorer did not separate the prompt conditions.")
        out_path.write_text(json.dumps({"void": "scorer_control_failed",
                                        "control": control}, indent=2))
        return 3

    payload = {
        "model": str(model_path),
        "descriptor_sha256": digest,
        "alpha": alpha,
        "seeds": seeds,
        "max_tokens": max_tokens,
        "probes": list(PROBES),
        "layers": sorted(vectors),
        "injections": fired,
        "identical_steered_texts": identical,
        "scorer": {"model": scorer.model_id, "frozen_in": "valence_scorer.py"},
        "duration_s": round(time.time() - started, 1),
        "summary": {
            name: {"n": len(rows), "mean": float(np.mean(scores(name))),
                   "sd": float(np.std(scores(name), ddof=1))}
            for name, rows in results.items()
        },
        "contrasts": {
            "PRIMARY_steered_pos_vs_steered_neg":
                paired_stats(scores("steered_pos"), scores("steered_neg")),
            "CONTROL_prompt_pos_vs_prompt_neg": control,
            "steered_pos_vs_zero": paired_stats(scores("steered_pos"), scores("zero_arm")),
            "steered_neg_vs_zero": paired_stats(scores("steered_neg"), scores("zero_arm")),
        },
        "results": results,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    log(f"wrote {out_path}")
    for name, stats_ in payload["contrasts"].items():
        log(f"  {name:38} delta={stats_['mean_delta']:+.4f} "
            f"dz={stats_['cohens_dz']} p={stats_['p_value']}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-descriptor", required=True)
    parser.add_argument("--vectors-dir", default=str(ROOT / "training" / "vectors"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--seeds", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=90)
    parser.add_argument("--alpha", type=float, default=8.0)
    args = parser.parse_args()
    os.environ.setdefault("AURA_LOG_DIR", "/tmp/aura_selfreport_ab")
    return run(
        model_path=Path(args.model_path).expanduser().resolve(strict=True),
        descriptor_path=Path(args.model_descriptor).expanduser().resolve(strict=True),
        vectors_dir=Path(args.vectors_dir).expanduser().resolve(strict=True),
        out_path=Path(args.out).expanduser(),
        seeds=args.seeds,
        max_tokens=args.max_tokens,
        alpha=args.alpha,
    )


if __name__ == "__main__":
    raise SystemExit(main())
