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

Usage:
  python tools/nonparametric_proof.py            # full proof on the reflex model
  python tools/nonparametric_proof.py --json     # machine-readable verdict only
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _resolve_reflex_path() -> str:
    from core.brain.llm.model_registry import get_fallback_path

    return str(get_fallback_path())


def run_proof(*, max_tokens: int = 12, model_path: str = "", background_store: str = "") -> dict:
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
                                    started, max_tokens, background_store)


def _run_proof_with_lane(model_path, load, MLXEncoder, NonParametricMemory,
                         cached_generate_with_memory, started, max_tokens, background_store="") -> dict:
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

    results = []
    # 2. BASE pass (memory present but OFF) — the model cannot know these.
    for prompt, answer in facts:
        base_out = cached_generate_with_memory(
            model,
            tokenizer,
            prompt,
            memory,
            max_tokens=max_tokens,
            use_memory=False,
            principal="nonparametric_proof",
        )
        results.append({"prompt": prompt, "expected": answer.strip(), "base": base_out})

    control_base = [cached_generate_with_memory(
        model,
        tokenizer,
        prompt,
        memory,
        max_tokens=6,
        use_memory=False,
        principal="nonparametric_proof",
    ) for prompt in controls]

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
    for record, (prompt, answer) in zip(results, facts):
        mem_out = cached_generate_with_memory(
            model,
            tokenizer,
            prompt,
            memory,
            max_tokens=max_tokens,
            use_memory=True,
            principal="nonparametric_proof",
        )
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
    control_mem = [cached_generate_with_memory(
        model,
        tokenizer,
        prompt,
        memory,
        max_tokens=6,
        use_memory=True,
        principal="nonparametric_proof",
    ) for prompt in controls]
    control_preserved = control_base == control_mem

    passed = all(r["recalled"] for r in results) and control_preserved
    return {
        "schema": "aura.nonparametric_proof.v1",
        "generated_at": started,
        "model": model_path,
        "hidden_dim": encoder.dim,
        "datastore_entries": len(memory),
        "background_entries": background_entries,
        "ingested_positions": ingested_positions,
        "similarity_mode": "centered" if memory.similarity_ready() else "raw_fallback",
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
    args = parser.parse_args()

    report = run_proof(model_path=args.model_path, background_store=args.background_store)

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
        print(f"\n{'PASS' if report['passed'] else 'FAIL'} in {report['elapsed_s']}s — report: {out}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
