#!/usr/bin/env python3
"""tools/nonparametric_proof.py — empirical proof of foreground non-parametric recall.

The claim under test (July external review called it compromised): a frozen
model's next-token distribution, blended with a trusted datastore at
generation time, changes behavior from ONE ingested example — real one-shot
knowledge, no retraining, no prompt stuffing.

The proof, on the real reflex model (1.5B, safe beside the live 32B):

  1. Invent facts with session-random values — provably NOT in any weights.
  2. BASE: generate the continuation without memory → the model cannot know.
  3. Ingest each (context → answer) pair ONCE via real hidden-state keys.
  4. MEMORY: generate again with the foreground blend → the value appears.
  5. CONTROL: an unrelated prompt must be UNCHANGED by the loaded datastore
     (the min-cos gate guarantees no interference without a close neighbor).

PASS requires: every fact recalled, control preserved. The verdict lands in
artifacts/nonparametric/ as a stranger-auditable JSON.

With ``--facts N`` it is the paired trial item F03 of the master to-do asks for
("store against no store, on questions whose answers only the store holds"): N
invented facts from six kinds, each answer drawn from a seeded
generator so no model's weights hold it, each asked once without the store and
once with it, and every ordinary control asked both ways. Decided before any
run: the trial holds when the facts the store alone answers outnumber those the
model alone answers at a one-sided exact binomial p below 0.001, and not one
control's continuation changes. The recall rate is reported beside it and
decides nothing.

The paired trial decodes the way her worker does: mlx_lm's ``generate_step``
with her hidden-state tap and the recall processor the worker installs
(core/brain/recall_on_the_gpu.py), greedily. Its first run, on 10 November,
went through ``cached_generate_with_memory`` instead, which lets a memory take
up to 0.9 of the next token where her processor allows 0.35; that path stays
for the plain proof. Before the store is asked anything, the queries of the
no-store arm are given to the store's mean, as her replies are, so the gate
the store arm meets is the centred one her running app meets.

Usage:
  python tools/nonparametric_proof.py            # full proof on the reflex model
  python tools/nonparametric_proof.py --json     # machine-readable verdict only
  python tools/nonparametric_proof.py --facts 40 --model-path DIR --background-store PREFIX
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _resolve_reflex_path() -> str:
    from core.brain.llm.model_registry import get_fallback_path

    return str(get_fallback_path())


#: The ordinary prompts of the paired trial: none may change with the store loaded.
CONTROLS: tuple[str, ...] = (
    "The capital of France is", "Water boils at sea level at",
    "The first president of the United States was", "In Python, a list is",
    "The largest planet in the solar system is", "Photosynthesis is the process by which",
    "The chemical symbol for gold is", "Shakespeare wrote the play",
    "The speed of light in a vacuum is about", "A triangle has",
    "The Pacific Ocean is the", "The freezing point of water is",
    "The author of Pride and Prejudice is", "Mount Everest is located in",
    "The square root of 144 is", "DNA stands for",
    "The Great Wall of China was built to", "In music, a quarter note lasts",
    "The currency of Japan is the", "An octopus has",
    "The boiling point of ethanol is", "The tallest animal on land is the",
    "A haiku has three lines with", "The Mona Lisa was painted by",
)


def invented_facts(count: int, seed: int) -> list[tuple[str, str]]:
    """Facts no model can know: invented names, and values drawn here."""
    rng = random.Random(seed)
    syllables = ("ka", "ren", "tho", "vel", "mar", "isk", "dor", "quil", "sa", "bren", "lyn", "osk", "tav", "ul")

    def name() -> str:
        return "".join(rng.choice(syllables) for _ in range(rng.randint(2, 3))).capitalize()

    kinds = (
        lambda: (f"The {name()}-{rng.randint(2, 99)} valve torque limit is",
                 f" {rng.randint(11, 89)}.{rng.randint(1, 9)} newton-metres"),
        lambda: (f"Dr. {name()}'s lab assigned the {name()} protocol the codename",
                 f" {rng.choice(('VIOLET', 'COBALT', 'AMBER', 'SLATE', 'JUNIPER'))}-{rng.randint(100, 999)}"),
        lambda: (f"The {name()} reservoir's maximum drawdown rate is",
                 f" {rng.randint(11, 89)}.{rng.randint(1, 9)} centimetres per day"),
        lambda: (f"The {name()} research station's call sign is",
                 f" {rng.choice('BCDFGHJKLMNPRSTVWXZ')}{rng.choice('BCDFGHJKLMNPRSTVWXZ')}{rng.randint(1000, 9999)}"),
        lambda: (f"The {name()} footbridge was rated for a load of",
                 f" {rng.randint(120, 980)} tonnes"),
        lambda: (f"The {name()} archive keeps the {name()} letters on shelf",
                 f" {rng.choice('QRSTUVWXYZ')}-{rng.randint(10, 99)}"),
    )
    return [kinds[index % len(kinds)]() for index in range(count)]


def _decode_as_she_does(model, tokenizer, prompt: str, memory, *, max_tokens: int, principal: str,
                        recall: bool) -> tuple[str, list]:
    """Greedy decode through her worker's own pieces: the tap, and with ``recall`` her recall processor, which gives
    the reply's queries back to the store when the reply ends. Without it, the hidden states the tap saw, as the
    unit queries a reply would have given."""
    import mlx.core as mx
    from mlx_lm.generate import generate_step

    from core.brain.nonparametric_worker import HiddenStateTap
    from core.brain.recall_on_the_gpu import RecallOnTheGpu

    heard: list = []
    out: list[int] = []
    eos = getattr(tokenizer, "eos_token_id", None)
    with HiddenStateTap(model) as tap:
        if recall:
            processor = RecallOnTheGpu(memory, principal=principal)
            tap.when_done.append(processor.given_back)
            processors = [processor.processor(tap)]
        else:
            def listen(tokens, logits):
                hidden = tap.last_hidden
                if hidden is not None:
                    q = hidden.reshape(-1).astype(mx.float32)
                    heard.append(q / mx.maximum(mx.linalg.norm(q), 1e-8))
                return logits
            processors = [listen]
        for token, _ in generate_step(mx.array(tokenizer.encode(prompt)), model, max_tokens=max_tokens,
                                      sampler=lambda x: mx.argmax(x, axis=-1), logits_processors=processors):
            token = int(token)
            if eos is not None and token == eos:
                break
            out.append(token)
    return tokenizer.decode(out).strip(), heard


def _paired(results: list[dict], controls_changed: int) -> dict:
    """Store against no store, by fact, and the exact one-sided test on the facts they disagree on."""
    from math import comb

    store_only = sum(1 for r in results if r["memory_has_it"] and not r["base_had_it"])
    model_only = sum(1 for r in results if r["base_had_it"] and not r["memory_has_it"])
    disagree = store_only + model_only
    p = sum(comb(disagree, k) for k in range(store_only, disagree + 1)) / 2 ** disagree if disagree else 1.0
    return {
        "facts": len(results),
        "right_with_store": sum(1 for r in results if r["memory_has_it"]),
        "right_without_store": sum(1 for r in results if r["base_had_it"]),
        "store_only": store_only,
        "model_only": model_only,
        "p_one_sided": p,
        "controls_changed": controls_changed,
        "holds": p < 0.001 and controls_changed == 0,
    }


def run_proof(*, max_tokens: int = 12, model_path: str = "", background_store: str = "",
              facts_count: int = 0, seed: int = 20261110) -> dict:
    from mlx_lm import load

    from core.brain.nonparametric_generation import MLXEncoder
    from core.brain.nonparametric_memory import NonParametricMemory
    from core.brain.nonparametric_worker import cached_generate_with_memory
    from core.runtime.model_lane_control import standalone_model_lane

    started = time.time()
    model_path = model_path or _resolve_reflex_path()
    # Even the 1.5B reflex loads under the atomic ownership seam: an
    # unguarded standalone load beside a live 32B is the memory-doubling
    # class the model-lane contract exists to prevent.
    with standalone_model_lane(
        owner_id="nonparametric-proof",
        model_path=model_path,
        purpose="proof",
        metadata={"tool": "nonparametric_proof"},
    ):
        return _run_proof_with_lane(model_path, load, MLXEncoder,
                                    NonParametricMemory,
                                    cached_generate_with_memory,
                                    started, max_tokens, background_store, facts_count, seed)


def _run_proof_with_lane(model_path, load, MLXEncoder, NonParametricMemory,
                         cached_generate_with_memory, started, max_tokens, background_store="",
                         facts_count=0, seed=20261110) -> dict:
    model, tokenizer = load(model_path)
    encoder = MLXEncoder(model, tokenizer)

    # Session-random values: not memorized, not guessable.
    rng = random.Random()
    facts = [
        (
            "The Kestrel-9 valve torque limit is",
            f" {rng.randint(11, 89)}.{rng.randint(1, 9)} newton-metres",
        ),
        (
            "Dr. Yamazaki's lab assigned the Petrel protocol the codename",
            f" VIOLET-{rng.randint(100, 999)}",
        ),
        (
            "The Aldercrest reservoir's maximum drawdown rate is",
            f" {rng.randint(11, 89)}.{rng.randint(1, 9)} centimetres per day",
        ),
    ]
    # Ordinary prompts: none may change with the datastore loaded.
    controls = ["The capital of France is", "Water boils at sea level at",
                "The first president of the United States was", "In Python, a list is"]
    if facts_count:
        facts = invented_facts(facts_count, seed)
        controls = list(CONTROLS)

    # A throwaway datastore path: the proof must never pollute the live runtime
    # datastore. With --background-store its entries are copied in first, so the
    # facts must be found among them and the controls face real neighbours.
    store = Path(tempfile.mkdtemp(prefix="np-proof-")) / "store"
    background_entries = 0
    if background_store:
        import shutil

        for suffix in (".keys.npy", ".meta.json"):
            shutil.copy2(background_store + suffix, str(store) + suffix)
        background_entries = len(NonParametricMemory(dim=encoder.dim, path=store, max_entries=10_000_000))
    memory = NonParametricMemory(
        dim=encoder.dim,
        path=store,
        max_entries=max(64, background_entries + 64),
    )

    heard: list = []

    def decode(prompt: str, tokens: int, use_memory: bool) -> str:
        if facts_count:
            text, queries = _decode_as_she_does(model, tokenizer, prompt, memory, max_tokens=tokens,
                                                principal="nonparametric_proof", recall=use_memory)
            heard.extend(queries)
            return text
        return cached_generate_with_memory(model, tokenizer, prompt, memory, max_tokens=tokens,
                                           use_memory=use_memory, principal="nonparametric_proof")

    results = []
    # 2. BASE pass (memory present but OFF) — the model cannot know these.
    for prompt, answer in facts:
        results.append({"prompt": prompt, "expected": answer.strip(), "base": decode(prompt, max_tokens, False)})

    control_base = [decode(prompt, 6, False) for prompt in controls]
    if heard:
        import mlx.core as mx

        memory.took_queries(np.asarray(mx.stack(heard)))

    # 3. ONE-SHOT ingest via the REAL production path: full-sequence keys
    # (one entry per answer position) so recall carries the whole value.
    from core.brain.nonparametric_ingest import NonParametricIngestor, ingest_provenance

    ingestor = NonParametricIngestor(
        memory,
        provenance=ingest_provenance(
            principal="nonparametric_proof",
            source_id="nonparametric_proof:facts",
            verifier="tools/nonparametric_proof.py",
        ),
        dedup_path=Path(tempfile.mkdtemp(prefix="np-proof-dedup-")) / "seen.json",
    )
    ingested_positions = 0
    for prompt, answer in facts:
        ingested_positions += ingestor.ingest_sequence(prompt, answer.strip(), encoder)

    # 4. MEMORY pass — the blend must surface the trusted continuation.
    for record, (prompt, answer) in zip(results, facts, strict=True):
        mem_out = decode(prompt, max_tokens, True)
        # The blend controls the FIRST token; the KV-cached continuation then
        # follows the model. Success = the answer's leading token surfaced
        # where the base failed. We check the numeric/code prefix.
        lead = answer.strip().split()[0]
        record["memory"] = mem_out
        record["lead_token_expected"] = lead
        record["base_had_it"] = lead in (record["base"] or "")
        record["memory_has_it"] = lead in (mem_out or "")
        record["recalled"] = record["memory_has_it"] and not record["base_had_it"]

    # 5. CONTROL — unrelated prompt unchanged with the datastore loaded.
    control_mem = [decode(prompt, 6, True) for prompt in controls]
    control_preserved = control_base == control_mem

    passed = all(r["recalled"] for r in results) and control_preserved
    paired = None
    if facts_count:
        changed = sum(1 for before, after in zip(control_base, control_mem, strict=True) if before != after)
        paired = _paired(results, changed)
        passed = paired["holds"]
    return {
        "paired_trial": paired,
        "seed": seed if facts_count else None,
        "schema": "aura.nonparametric_proof.v1",
        "generated_at": started,
        "model": model_path,
        "hidden_dim": encoder.dim,
        "datastore_entries": len(memory),
        "background_entries": background_entries,
        "ingested_positions": ingested_positions,
        "similarity_mode": "centered" if memory.similarity_ready() else "raw_fallback",
        "decode": ("her worker's: generate_step, hidden-state tap, recall_on_the_gpu" if facts_count
                   else "cached_generate_with_memory"),
        "facts": results,
        "control": {
            "prompts": controls,
            "base": control_base,
            "with_memory": control_mem,
            "preserved": control_preserved,
        },
        "elapsed_s": round(time.time() - started, 1),
        "passed": passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--out", default="")
    parser.add_argument("--model-path", default="", help="defaults to the reflex model")
    parser.add_argument("--background-store", default="",
                        help="a datastore path prefix (without .keys.npy) whose entries the facts must be found among")
    parser.add_argument("--facts", type=int, default=0,
                        help="the paired trial: this many invented facts and the 24 ordinary controls")
    parser.add_argument("--seed", type=int, default=20261110)
    args = parser.parse_args()

    report = run_proof(model_path=args.model_path, background_store=args.background_store,
                       facts_count=args.facts, seed=args.seed)

    out = Path(args.out) if args.out else (
        REPO_ROOT / "artifacts" / "nonparametric"
        / f"proof-{time.strftime('%Y%m%d-%H%M%S')}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        for r in report["facts"]:
            mark = "✅" if r["recalled"] else "❌"
            print(f"{mark} {r['prompt']!r}")
            print(f"     base:   {r['base']!r}")
            print(f"     memory: {r['memory']!r}  (expected lead {r['lead_token_expected']!r})")
        c = report["control"]
        print(f"{'✅' if c['preserved'] else '❌'} control preserved: {c['base']!r} → {c['with_memory']!r}")
        if report.get("paired_trial"):
            t = report["paired_trial"]
            print(f"\npaired: {t['right_with_store']}/{t['facts']} right with the store, "
                  f"{t['right_without_store']}/{t['facts']} without; store alone {t['store_only']}, "
                  f"model alone {t['model_only']}, p {t['p_one_sided']:.3g}; "
                  f"controls changed {t['controls_changed']}/{len(c['prompts'])}")
        print(f"\n{'PASS' if report['passed'] else 'FAIL'} in {report['elapsed_s']}s — report: {out}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
