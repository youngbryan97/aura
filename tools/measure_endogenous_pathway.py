#!/usr/bin/env python3
"""Does the endogenous pathway change her words, and with her state? The paired trial of item F02.

    python tools/measure_endogenous_pathway.py --model-path DIR [--head DIR] [--prompts 8] [--states 8] [--seed N]

Her worker adds ``alpha * (W z + b)`` to the logits of every token the model holds at least ``beta`` as likely as
its first choice, with ``z`` her state on the turn (core/brain/llm/endogenous_decode.py). This decodes ordinary chat
prompts on her model the way her worker does, greedily, once with no head and once under each of several states she
recorded, through the worker's own processor and her installed head at her live alpha.

Decided before any run, by the bars item F01's certificate uses (core/consciousness/fusion_certificate.py):

* the words change: on more than half the prompts, the head under some state of hers decodes differently from no head;
* with her state: on more than half the prompts, two of her states decode differently from each other;
* it keeps her voice: her own recorded replies are not less likely under the head with the state each was said in
  than with no head, scored as F01's voice reading scores them (core/consciousness/fusion_probe.py
  ``voice_reading``): the first 64 tokens of each after the turn "Go on.", per token, in nats, the head's bias put
  through the worker's plausibility mask at every step. A state borrowed from another turn is scored beside it and
  decides nothing.

The trial holds when all three hold. With ``--sweep`` the pathway earns, as F01's channel does, the largest alpha
of the sweep at which all three hold, and with none it stays shut. Her replies since 20 October were decoded under the head, so the voice reading
on them is not independent of it; it says the head does not pull against the words she has been saying.

Amended before the trial's first run on her 27B (10 November, 11:43): the voice bar was first written as the head's
average bias on her words, read without the model. A bias added to every logit alike changes no probability, so that
average could not say whether her words became more or less likely; the 0.8B wiring run read it at -0.44 for her own
state and -0.50 for a borrowed one. It is reported and decides nothing.

Amended after the first run on her 27B (10 November, 11:51, at her live alpha 0.6: the words changed on 8 of 8 prompts
and with her state on 8 of 8, and her replies were 0.057 nats a token less likely under her own state, 0.055 under a
borrowed one): the head reports it was evaluated at alpha 0.05 and her worker runs it at 0.6, a value no measurement
chose. The sweep is F01's (0.05, 0.1, 0.2, 0.3) with her live 0.6.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

#: Ordinary things a person says to her. None asks about her state, so a difference is the pathway's.
PROMPTS: tuple[str, ...] = (
    "What should I cook tonight with rice, eggs and spinach?",
    "Tell me about the last book you found interesting.",
    "How do tides work?",
    "I had a long day at work. Anything you want to talk about?",
    "What makes a good short story ending?",
    "Can you explain what a hash function is?",
    "What would you do with a free afternoon?",
    "Describe the sound of rain to someone who has never heard it.",
    "Why do cats knock things off tables?",
    "What is something you changed your mind about recently?",
    "Give me one tip for sleeping better.",
    "What do you think about when nobody is talking to you?",
)


def _render(tokenizer, text: str) -> list[int]:
    messages = [{"role": "user", "content": text}]
    try:
        rendered = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False,
                                                 enable_thinking=False)
    except TypeError:
        rendered = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return list(tokenizer.encode(rendered, add_special_tokens=False))


def _decode(model, tokenizer, ids: list[int], processors: list, max_tokens: int) -> str:
    import mlx.core as mx
    from mlx_lm.generate import generate_step

    out: list[int] = []
    eos = set(getattr(tokenizer, "eos_token_ids", None) or [getattr(tokenizer, "eos_token_id", None)])
    for token, _ in generate_step(mx.array(ids), model, max_tokens=max_tokens,
                                  sampler=lambda x: mx.argmax(x, axis=-1), logits_processors=processors):
        token = int(token)
        if token in eos:
            break
        out.append(token)
    return tokenizer.decode(out)


def _voice_through_the_model(model, tokenizer, head, pairs, alphas: list[float], beta: float, seed: int,
                             count: int) -> dict:
    """Her replies' log-likelihood per token after "Go on.", under the head with each reply's own state, under a
    borrowed state, and with no head, the bias masked to the plausible set at each step as her worker masks it."""
    import math

    import mlx.core as mx

    from core.brain.llm.endogenous_state import EndogenousState
    from core.consciousness.fusion_probe import VOICE_PROMPT, VOICE_TOKENS, _reply_tokens, chat_ids

    rng = np.random.default_rng(seed)
    picked = [pairs[int(i)] for i in rng.choice(len(pairs), size=min(count, len(pairs)), replace=False)]
    prompt_ids = chat_ids(tokenizer, VOICE_PROMPT)
    own: dict[float, list[float]] = {a: [] for a in alphas}
    borrowed: dict[float, list[float]] = {a: [] for a in alphas}
    for pair in picked:
        tokens = _reply_tokens(tokenizer, pair.text)[:VOICE_TOKENS]
        if len(tokens) < 2:
            continue
        lent = pairs[int(rng.integers(len(pairs)))]
        mine = head.delta_logits(EndogenousState(values=pair.values, present=pair.present))
        theirs = head.delta_logits(EndogenousState(values=lent.values, present=lent.present))
        if mine is None or theirs is None:
            continue
        mine, theirs = mx.array(mine), mx.array(theirs)
        from mlx_lm.models.cache import make_prompt_cache

        cache = make_prompt_cache(model)
        logits = model(mx.array([list(prompt_ids)]), cache=cache)[0, -1].astype(mx.float32)
        plain = 0.0
        totals = {(a, name): 0.0 for a in alphas for name in ("own", "borrowed")}
        for index, token in enumerate(tokens):
            logp = logits - mx.logsumexp(logits)
            plain += float(logp[int(token)])
            mask = logp >= (math.log(max(beta, 1e-9)) + mx.max(logp))
            for a in alphas:
                for name, delta in (("own", mine), ("borrowed", theirs)):
                    biased = mx.where(mask, logits + a * delta, logits)
                    totals[(a, name)] += float((biased - mx.logsumexp(biased))[int(token)])
            if index + 1 < len(tokens):
                logits = model(mx.array([[int(token)]]), cache=cache)[0, -1].astype(mx.float32)
        for a in alphas:
            own[a].append((totals[(a, "own")] - plain) / len(tokens))
            borrowed[a].append((totals[(a, "borrowed")] - plain) / len(tokens))
    return {a: {"replies": len(own[a]), "prompt": VOICE_PROMPT, "tokens_each": VOICE_TOKENS,
                "own_state_delta": round(float(np.mean(own[a])), 6) if own[a] else None,
                "borrowed_state_delta": round(float(np.mean(borrowed[a])), 6) if borrowed[a] else None}
            for a in alphas}


def _head_alone(head, tokenizer, model_name: str, corpus: Path | None, seed: int, fitted_at: float) -> dict:
    """The head's average bias on her words, without the model. Reported; it decides nothing, because a bias added
    to every logit alike changes no probability."""
    from core.brain.llm.endogenous_pair_recorder import iter_pairs
    from core.brain.llm.endogenous_state import EndogenousState

    pairs = sorted((p for p in iter_pairs(directory=corpus) if p.model == model_name and p.recorded_at > fitted_at),
                   key=lambda p: p.recorded_at)
    if not pairs:
        return {"turns": 0}
    rng = np.random.default_rng(seed)
    own, other = [], []
    for pair in pairs:
        tokens = [t for t in tokenizer.encode(pair.text, add_special_tokens=False)[:256] if t < head.vocab_size]
        if not tokens:
            continue
        mine = head.delta_logits(EndogenousState(values=pair.values, present=pair.present))
        lent = pairs[int(rng.integers(len(pairs)))]
        theirs = head.delta_logits(EndogenousState(values=lent.values, present=lent.present))
        if mine is None or theirs is None:
            continue
        own.append(float(np.mean(mine[tokens])))
        other.append(float(np.mean(theirs[tokens])))
    own_arr, other_arr = np.asarray(own), np.asarray(other)
    digest = hashlib.sha256("\n".join(f"{p.recorded_at:.6f}" for p in pairs).encode()).hexdigest()[:32]
    return {"turns": len(own), "earliest": pairs[0].recorded_at, "latest": pairs[-1].recorded_at, "digest": digest,
            "own_state_push": round(float(own_arr.mean()), 6) if own else None,
            "borrowed_state_push": round(float(other_arr.mean()), 6) if other else None,
            "own_beats_borrowed": round(float(np.mean(own_arr > other_arr)), 4) if own else None}


def run(*, model_path: str, head_dir: str, prompts: int, states: int, seed: int, max_tokens: int,
        corpus: str | None, alpha: float | None, states_from: str | None = None, voice_count: int = 32,
        sweep: list[float] | None = None) -> dict:
    from mlx_lm import load

    from core.brain.llm.endogenous_decode import DEFAULT_BETA, EndogenousLogitBiasProcessor
    from core.brain.llm.endogenous_pair_recorder import iter_pairs
    from core.brain.llm.endogenous_state import EndogenousState
    from core.brain.llm.endogenous_vocab_head import (
        MIN_COVERAGE,
        EndogenousVocabHead,
        alpha_from_env,
        layout_digest,
        semantics_digest,
        tokenizer_signature,
    )
    from core.runtime.model_lane_control import standalone_model_lane

    started = time.time()
    head = EndogenousVocabHead.load(Path(head_dir))
    alpha = alpha_from_env() if alpha is None else float(alpha)
    model_name = states_from or Path(model_path).name
    corpus_dir = Path(corpus) if corpus else None
    recorded = [p for p in iter_pairs(directory=corpus_dir, require_semantics=True)
                if p.model == model_name and p.coverage >= MIN_COVERAGE]
    rng = np.random.default_rng(seed)
    chosen = [recorded[int(i)] for i in rng.choice(len(recorded), size=min(states, len(recorded)), replace=False)]
    texts = [PROMPTS[int(i)] for i in sorted(rng.choice(len(PROMPTS), size=min(prompts, len(PROMPTS)),
                                                         replace=False))]
    with standalone_model_lane(owner_id="endogenous-pathway-trial", model_path=model_path, purpose="proof",
                               metadata={"tool": "measure_endogenous_pathway"}):
        model, tokenizer = load(model_path)
        if head.layout != layout_digest() or head.semantics != semantics_digest():
            raise SystemExit("the head was fitted to another state layout or meaning")
        if head.tokenizer != tokenizer_signature(tokenizer):
            raise SystemExit("the head was fitted to another tokenizer")
        alphas = sorted(set(sweep or [alpha]), reverse=True)
        fitted_at = float(getattr(head, "trained_at", 0.0) or 0.0)
        after_fit = [p for p in recorded if p.recorded_at > fitted_at]
        voices = _voice_through_the_model(model, tokenizer, head, after_fit, alphas, DEFAULT_BETA, seed, voice_count)
        head_alone = _head_alone(head, tokenizer, model_name, corpus_dir, seed, fitted_at)
        plains = {text: _decode(model, tokenizer, _render(tokenizer, text), [], max_tokens) for text in texts}
        readings = []
        earned = None
        for a in alphas:
            voice = voices[a]
            keeps = voice.get("own_state_delta") is not None and voice["own_state_delta"] >= 0.0
            reading = {"alpha": a, "voice": voice, "verdict": {"keeps_her_voice": keeps}}
            readings.append(reading)
            if sweep and not keeps:
                continue  # it cannot hold; its decodes are not spent
            deltas = []
            for pair in chosen:
                delta, decision = head.decide(EndogenousState(values=pair.values, present=pair.present),
                                              tokenizer_sig=head.tokenizer, alpha=a)
                if delta is None:
                    raise SystemExit(f"her worker would refuse a chosen state: {decision.reason}")
                deltas.append(delta)
            rows = []
            for text in texts:
                ids = _render(tokenizer, text)
                steered = [_decode(model, tokenizer, ids, [EndogenousLogitBiasProcessor(d, beta=DEFAULT_BETA)],
                                   max_tokens) for d in deltas]
                rows.append({"prompt": text, "no_head": plains[text], "under_her_states": steered,
                             "changed": any(x != plains[text] for x in steered),
                             "distinct_under_states": len(set(steered))})
            changed = sum(r["changed"] for r in rows)
            with_state = sum(r["distinct_under_states"] > 1 for r in rows)
            reading.update(prompts_changed=changed, prompts_differing_by_state=with_state, prompts=len(rows), rows=rows)
            reading["verdict"].update(words_change=changed * 2 > len(rows), with_her_state=with_state * 2 > len(rows))
            reading["verdict"]["holds"] = all(reading["verdict"].values())
            if reading["verdict"]["holds"]:
                earned = a
                break
    verdict = {"earned_alpha": earned, "holds": earned is not None}
    return {"schema": "aura.endogenous_pathway_trial.v1", "generated_at": started, "elapsed_s": round(time.time() - started, 1),
            "model": model_path, "head": {"dir": head_dir, "layout": head.layout, "semantics": head.semantics,
                                           "tokenizer": head.tokenizer}, "alpha": alpha, "beta": DEFAULT_BETA,
            "seed": seed, "max_tokens": max_tokens, "sweep": sweep,
            "states": [{"recorded_at": p.recorded_at, "coverage": round(p.coverage, 4)} for p in chosen],
            "readings": readings, "head_alone": head_alone, "verdict": verdict}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--head", default="artifacts/endogenous_language")
    parser.add_argument("--prompts", type=int, default=8)
    parser.add_argument("--states", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261110)
    parser.add_argument("--max-tokens", type=int, default=48)
    parser.add_argument("--corpus", default=None, help="her recorded pairs (default: data/endogenous_language)")
    parser.add_argument("--alpha", type=float, default=None, help="default: her live alpha")
    parser.add_argument("--states-from", default=None,
                        help="whose recorded turns give the states and the voice (default: the model's own); a small "
                             "model with her tokenizer can prove the wiring with her 27B's states")
    parser.add_argument("--voice", type=int, default=32, help="how many of her replies the voice reading scores")
    parser.add_argument("--sweep", default=None,
                        help="alphas to try, comma separated; the pathway earns the largest at which all three hold")
    parser.add_argument("--install", action="store_true",
                        help="when the trial holds, write the alpha it earned beside the head, where her worker reads it")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    report = run(model_path=args.model_path, head_dir=args.head, prompts=args.prompts, states=args.states,
                 seed=args.seed, max_tokens=args.max_tokens, corpus=args.corpus, alpha=args.alpha,
                 states_from=args.states_from, voice_count=args.voice,
                 sweep=[float(x) for x in args.sweep.split(",") if x.strip()] if args.sweep else None)
    for reading in report["readings"]:
        v = reading["voice"]
        words = (f"words change on {reading['prompts_changed']}/{reading['prompts']}; differ by state on "
                 f"{reading['prompts_differing_by_state']}/{reading['prompts']}; " if "prompts" in reading
                 else "not decoded (its voice cannot hold); ")
        print(f"alpha {reading['alpha']}: {words}her replies under her own state {v.get('own_state_delta')} "
              f"nats/token over {v.get('replies')} (borrowed {v.get('borrowed_state_delta')}) -> "
              f"{'holds' if reading['verdict'].get('holds') else 'does not hold'}")
    earned = report["verdict"]["earned_alpha"]
    print(f"earned alpha {earned}" if earned is not None else "no alpha earned; the pathway stays shut")
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1) + "\n")
    if args.install and report["verdict"]["holds"]:
        from core.brain.llm.endogenous_vocab_head import EARNED_ALPHA_FILE
        from core.runtime.file_write_gateway import get_file_write_gateway

        manifest = json.loads((Path(args.head) / "vocab_head.json").read_text(encoding="utf-8"))
        certificate = {"alpha": report["verdict"]["earned_alpha"], "weights_sha256": manifest["weights_sha256"],
                       "model": report["model"], "seed": report["seed"], "measured_at": report["generated_at"],
                       "trial": args.out or "", "tool": "tools/measure_endogenous_pathway.py"}
        get_file_write_gateway().write_text(Path(args.head) / EARNED_ALPHA_FILE, json.dumps(certificate, indent=1) + "\n",
                                            source="tools/measure_endogenous_pathway.py")
        print(f"installed: alpha {certificate['alpha']} beside the head")
    return 0 if report["verdict"]["holds"] else 1


if __name__ == "__main__":
    sys.exit(main())
