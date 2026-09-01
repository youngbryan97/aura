#!/usr/bin/env python3
"""What Aura's own recall actually costs, fitted and then held out.

Card 001: a held-out latency prediction beats a candidate-count-only baseline
on Aura's own recall traces.

The traces are real. This drives core.memory.hybrid_store's retrieve — the
episodic recall path the runtime uses — over stores of varying size with
varying confidence floors and top-k, and times each call with a monotonic
clock. Nothing about the latency is modelled; it is measured, and the fit is
then asked to predict recalls it never saw.

The comparison the card names is the one that matters. Candidate count alone
is the obvious predictor and it is nearly right, which is why beating it is
the bar rather than beating a constant: a law that only reproduces "more
candidates take longer" has not learned anything about the recall path.

    python tools/campaigns/retrieval_latency_campaign.py --recalls 600
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.memory.hybrid_store import HybridMemoryStore  # noqa: E402
from core.science.retrieval_latency import (  # noqa: E402
    RetrievalObservation,
    fit,
)

WORDS = [f"w{i}" for i in range(64)]


def _content(rng: random.Random) -> str:
    return " ".join(rng.choice(WORDS) for _ in range(24))


async def _collect(
    *, store_sizes: list[int], recalls: int, seed: int
) -> list[RetrievalObservation]:
    """Drive the real store and time what it does."""
    rng = random.Random(seed)
    observations: list[RetrievalObservation] = []

    with tempfile.TemporaryDirectory() as tmp:
        for size in store_sizes:
            store = HybridMemoryStore(storage_dir=str(Path(tmp) / f"s{size}"))
            # A confidence spread, so the floor actually filters and the
            # candidate count is not just the store size again.
            for _ in range(size):
                await store.store(
                    _content(rng),
                    {"confidence": rng.uniform(0.3, 1.0), "source": "campaign"},
                )

            for _ in range(recalls // len(store_sizes)):
                query = " ".join(rng.choice(WORDS) for _ in range(rng.randint(1, 4)))
                floor = rng.choice((0.0, 0.4, 0.6, 0.8))
                top_k = rng.choice((1, 5, 20))

                began = time.perf_counter()
                hits = await store.retrieve(
                    query, top_k=top_k, min_confidence=floor
                )
                seconds = time.perf_counter() - began

                observations.append(
                    RetrievalObservation(
                        seconds=seconds,
                        # Every stored entry is read and tested: that is what
                        # this recall path does, and the candidate count is
                        # the work it did, not the rows it returned.
                        candidates=size,
                        store_hops=1,
                        # One content scan per entry, which is where the time
                        # goes and what a candidate-only law cannot separate
                        # from the store size.
                        embedding_calls=len(query.split()) * size,
                        backend="hybrid_episodic",
                        hit=bool(hits),
                    )
                )
    return observations


def _baseline(rows: list[RetrievalObservation]) -> tuple[float, float]:
    """Candidate count only: the predictor the card says to beat."""
    n = len(rows)
    mean_x = sum(o.candidates for o in rows) / n
    mean_y = sum(o.seconds for o in rows) / n
    cov = sum((o.candidates - mean_x) * (o.seconds - mean_y) for o in rows)
    var = sum((o.candidates - mean_x) ** 2 for o in rows)
    slope = cov / var if var else 0.0
    return mean_y - slope * mean_x, slope


def _rmse(errors: list[float]) -> float:
    return (sum(e * e for e in errors) / len(errors)) ** 0.5 if errors else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recalls", type=int, default=600)
    parser.add_argument("--sizes", default="200,600,1200,2400")
    parser.add_argument("--seed", type=int, default=1_618_033)
    parser.add_argument("--out", default="docs/evidence/retrieval_latency.json")
    args = parser.parse_args()

    sizes = [int(s) for s in args.sizes.split(",")]
    observations = asyncio.run(
        _collect(store_sizes=sizes, recalls=args.recalls, seed=args.seed)
    )

    rng = random.Random(args.seed + 1)
    rng.shuffle(observations)
    split = int(len(observations) * 0.7)
    train, held_out = observations[:split], observations[split:]

    law = fit(train, backend="hybrid_episodic")
    if law is None:
        print("not enough observations to fit", file=sys.stderr)
        return 1

    intercept, slope = _baseline(train)
    law_errors = [law.predict(o) - o.seconds for o in held_out]
    baseline_errors = [
        (intercept + slope * o.candidates) - o.seconds for o in held_out
    ]
    mean_held = statistics.fmean(o.seconds for o in held_out)
    mean_errors = [mean_held - o.seconds for o in held_out]

    law_rmse = _rmse(law_errors)
    baseline_rmse = _rmse(baseline_errors)

    # The direction test: what the law says happens when the candidate count
    # moves. A law whose predicted direction is wrong is refuted by its own
    # coefficients whatever its fit.
    probe = held_out[0]
    intervention = law.intervene(probe, candidates=probe.candidates * 2)

    payload = {
        "schema": "aura.retrieval_latency.v1",
        "card": "001",
        "claim_boundary": (
            "core.memory.hybrid_store episodic recall on synthetic content, "
            "timed on this host; a latency law for that recall path, not for "
            "every store Aura reads"
        ),
        "config": {
            "recalls": len(observations),
            "store_sizes": sizes,
            "seed": args.seed,
            "train": len(train),
            "held_out": len(held_out),
        },
        "law": law.to_dict(),
        "held_out": {
            "law_rmse": round(law_rmse, 8),
            "candidates_only_rmse": round(baseline_rmse, 8),
            "predict_the_mean_rmse": round(_rmse(mean_errors), 8),
            "law_beats_candidates_only": law_rmse < baseline_rmse,
            "improvement": (
                round(1.0 - law_rmse / baseline_rmse, 4) if baseline_rmse else 0.0
            ),
        },
        "intervention": intervention,
        "median_seconds": round(
            statistics.median(o.seconds for o in observations), 6
        ),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))

    from core.governance_context import local_internal_governed_scope
    from core.runtime.file_write_gateway import get_file_write_gateway

    out = ROOT / args.out
    with local_internal_governed_scope("retrieval_latency_campaign"):
        get_file_write_gateway().write_text(
            out, json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
