#!/usr/bin/env python3
"""Did HER state make HER words more likely than another turn's state would?

    python tools/endogenous_paired_recovery.py --head DIR --tokenizer PATH --model NAME

The likelihood gain a fit reports is a claim about a whole corpus. This is the
per-turn version of the same question, and it is the closest thing to the
decisive experiment that does not need a loaded model: for each held-out turn,
score the tokens she actually produced under the state that actually held, and
again under a state that belonged to a different turn. Everything else is
identical — same head, same words, same tokenizer.

The null is the point. A head that had learned nothing about states would
favour her own words exactly half the time.

Held-out turns come from the END of the corpus, so every turn scored was
recorded after everything the head was fitted on. States drift slowly, so a
randomly chosen other-turn state is often similar to the real one; that makes
this a conservative test rather than a generous one.

The corpus is PINNED, and it has to be. The live runtime keeps recording, so
"held out from the end" names different turns every time this is run: the same
head over the same lane returned 453 turns at 57.0%, then 488 turns at 53.5%,
because 412 more turns had arrived in between and the tail moved. An
experiment whose input changes underneath it is not reproducible, whatever its
p-value. `--upto` fixes the corpus at a recorded-at boundary and every report
carries the boundary, the count and a digest of exactly which turns were
scored.

The one run this repository cites is
docs/evidence/endogenous_language/paired_recovery_9b.json: 491 held-out turns
of the 9B utility lane, 54.0% favouring her own state, sign test p = 0.043.
Above chance and close to the line. Earlier figures of 57.0% at p = 0.0018 and
57% at p = 0.008 came from unpinned runs over a corpus that grew between them
and should not be quoted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from math import comb
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.brain.llm.endogenous_pair_recorder import iter_pairs  # noqa: E402
from core.brain.llm.endogenous_readout_training import (  # noqa: E402
    _forward_in_time_split,
    tokenize_pairs,
)
from core.brain.llm.endogenous_state import EndogenousState  # noqa: E402
from core.brain.llm.endogenous_vocab_head import EndogenousVocabHead  # noqa: E402
from tools.train_endogenous_readout import load_tokenizer  # noqa: E402


def _sign_test(wins: int, total: int) -> float:
    """One-sided probability of this many wins or more under a fair coin."""
    if total <= 0:
        return 1.0
    tail = sum(comb(total, index) for index in range(wins, total + 1))
    return tail / (2**total)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--model", required=True, help="recorded model basename")
    parser.add_argument("--corpus", default=None)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument(
        "--upto",
        type=float,
        default=None,
        help=(
            "pin the corpus to turns recorded at or before this unix time, so "
            "the run is reproducible while the runtime keeps recording"
        ),
    )
    parser.add_argument("--json", default=None)
    args = parser.parse_args(argv)

    head = EndogenousVocabHead.load(args.head)
    if head is None:
        raise SystemExit(f"no head at {args.head}")
    tokenizer = load_tokenizer(args.tokenizer)

    corpus = Path(args.corpus) if args.corpus else None
    pairs = sorted(
        (p for p in iter_pairs(directory=corpus) if p.model == args.model),
        key=lambda pair: pair.recorded_at,
    )
    if args.upto is not None:
        pairs = [pair for pair in pairs if pair.recorded_at <= float(args.upto)]
    if not pairs:
        raise SystemExit(f"no recorded turns for model {args.model!r}")

    turns = tokenize_pairs(pairs, tokenizer, max_tokens_per_turn=256)
    _train, _validation, held = _forward_in_time_split(turns)

    by_group = {
        hashlib.sha256(pair.text.encode("utf-8")).hexdigest()[:16]: pair
        for pair in pairs
    }
    usable = [turn for turn in held if turn.group in by_group]
    if not usable:
        raise SystemExit("no held-out turn could be matched back to its state")

    rng = np.random.default_rng(args.seed)
    own: list[float] = []
    other: list[float] = []
    for turn in usable:
        mine = by_group[turn.group]
        delta = head.delta_logits(
            EndogenousState(values=mine.values, present=mine.present)
        )
        if delta is None:
            continue
        borrowed = by_group[usable[int(rng.integers(len(usable)))].group]
        rival = head.delta_logits(
            EndogenousState(values=borrowed.values, present=borrowed.present)
        )
        if rival is None:
            continue
        own.append(float(np.mean(delta[turn.tokens])))
        other.append(float(np.mean(rival[turn.tokens])))

    own_scores = np.asarray(own)
    other_scores = np.asarray(other)
    difference = own_scores - other_scores
    wins = int((difference > 0).sum())
    # Exactly which turns were scored, so a later run can be shown to be the
    # same experiment rather than merely the same command.
    corpus_digest = hashlib.sha256(
        "\n".join(f"{pair.recorded_at:.6f}:{pair.model}" for pair in pairs).encode()
    ).hexdigest()[:32]
    report = {
        "corpus": {
            "model": args.model,
            "turns": len(pairs),
            "earliest": round(pairs[0].recorded_at, 3),
            "latest": round(pairs[-1].recorded_at, 3),
            "pinned_upto": args.upto,
            "digest": corpus_digest,
        },
        "head": {
            "layout": head.layout,
            "semantics": head.semantics,
            "tokenizer": head.tokenizer,
        },
        "held_out_turns": len(difference),
        "own_state_mean_bias": round(float(own_scores.mean()), 6),
        "other_state_mean_bias": round(float(other_scores.mean()), 6),
        "paired_difference": round(float(difference.mean()), 6),
        "turns_favouring_own_state": wins,
        "share": round(wins / max(1, len(difference)), 4),
        "sign_test_p": _sign_test(wins, len(difference)),
    }
    print(json.dumps(report, indent=2))
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
