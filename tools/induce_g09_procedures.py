#!/usr/bin/env python3
"""G09: procedures her model writes from a kind's solved development problems.

For each named kind, the development problems with their known answers go to
core/learning/procedures_from_solved_examples.py, which asks her model for a
``solve`` function, checks it in the sandbox against every known answer, and
revises or replaces it within the rounds given. Each problem is the request
text the G09 harness sends (tools/run_g09_organ.py), so a kept procedure reads
exactly what her ordinary route reads. A procedure's returned answer agrees
with a known one when the benchmark's own grader says so.

Development and test problems never mix. For a BBEH task the test problems
are its 20 in BBEH Mini and the development problems are its other 180; for a
Natural Plan kind and for CRUXEval's output prediction, the problems are
ordered by a hash of their id and the first half is development.

Usage:
    induce_g09_procedures.py --kinds bbeh_hyperbaton,calendar --output DIR
        [--lines 2] [--rounds 3] [--shots 3] [--batch 8] [--mechanisms] [--fake-proposer FILE]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BENCH = Path("~/.aura/benchmarks").expanduser()
NATURAL_PLAN = {"calendar": "calendar_scheduling", "meeting": "meeting_planning", "trip": "trip_planning"}


def _mini_inputs() -> set[str]:
    return {e["input"] for e in json.loads((BENCH / "g09/bbeh/mini/data.json").read_text())["examples"]}


def _natural_plan_halves(kind: str) -> tuple[list[str], list[str]]:
    data = json.loads((BENCH / f"g09/natural-plan/{NATURAL_PLAN[kind]}.json").read_text(encoding="utf-8"))
    ordered = sorted(data, key=lambda key: hashlib.sha256(f"{kind}:{key}".encode()).hexdigest())
    return ordered[: len(ordered) // 2], ordered[len(ordered) // 2:]


def problems(kind: str, part: str) -> list[dict[str, Any]]:
    """The kind's development or test problems as the G09 harness poses them, with their truths."""
    from tools.run_g09_organ import BBEH_SUFFIX

    if kind == "cruxeval":
        from tools.run_g09_organ import LOADERS

        tasks = sorted(LOADERS["cruxeval"](), key=lambda t: hashlib.sha256(t["id"].encode()).hexdigest())
        half = tasks[: len(tasks) // 2] if part == "development" else tasks[len(tasks) // 2:]
        return [{"key": t["id"], "problem": t["request"], "truth": t["truth"], "answer_text": t["truth"]["output"]}
                for t in half]
    if kind.startswith("bbeh_"):
        mini = _mini_inputs()
        examples = json.loads((BENCH / f"g09/bbeh/{kind}/task.json").read_text(encoding="utf-8"))["examples"]
        return [{"key": f"{kind}:{index}", "problem": f"{e['input']}\n\n{BBEH_SUFFIX}", "truth": e["target"]}
                for index, e in enumerate(examples) if (e["input"] in mini) == (part == "test")]
    data = json.loads((BENCH / f"g09/natural-plan/{NATURAL_PLAN[kind]}.json").read_text(encoding="utf-8"))
    development, test = _natural_plan_halves(kind)
    keys = development if part == "development" else test
    if kind == "trip":
        truth = lambda v: {"cities": v["cities"], "durations": v["durations"]}  # noqa: E731
    else:
        truth = lambda v: v["golden_plan"]  # noqa: E731
    return [{"key": f"{kind}:{key}", "problem": data[key]["prompt_5shot"], "truth": truth(data[key]),
             "answer_text": data[key]["golden_plan"]} for key in keys]


def test_ids(domain: str) -> set[str]:
    """The G09 harness's task ids for a domain's test problems."""
    if domain == "bbeh":
        kinds = sorted(p.name for p in (BENCH / "g09/bbeh").glob("bbeh_*"))
    elif domain in ("planning", "calendar"):
        kinds = ["calendar"]
    elif domain == "trip":
        kinds = ["trip"]
    elif domain == "cruxeval":
        kinds = ["cruxeval"]
    else:
        raise ValueError(f"no development and test split is defined for {domain}")
    return {row["key"] for kind in kinds for row in problems(kind, "test")}


def grader_for(kind: str):
    """Whether a procedure's returned answer is right, by the benchmark's own grader."""
    from tools.run_g09_organ import grade_bbeh, grade_cruxeval, grade_planning, grade_trip

    if kind.startswith("bbeh_"):
        return lambda returned, truth: grade_bbeh(f"The answer is: {returned}", truth)[0]
    grade = {"calendar": grade_planning, "trip": grade_trip, "cruxeval": grade_cruxeval}[kind]
    return lambda returned, truth: grade(returned, truth)[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kinds", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lines", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--shots", type=int, default=3,
                        help="solved problems shown per request; BIG-Bench Hard's chain-of-thought protocol shows 3")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--model", type=Path,
                        default=Path("~/.aura/models/Aura-Qwen3.8-27B-persona-crsm-7f6a2e83f73f5eef9d15"))
    parser.add_argument("--fake-proposer", type=Path, help="a JSON list of replies, for a dry run without the model")
    parser.add_argument("--mechanisms", action="store_true",
                        help="tell her model it may import core/reasoning/mechanisms.py, and what it offers")
    args = parser.parse_args()

    from core.learning.procedures_from_solved_examples import (
        ProcedureBook,
        SolvedExample,
        candidate_record,
        dumps,
        induce,
    )

    output = args.output.expanduser()
    (output / "requests").mkdir(parents=True, exist_ok=True)
    kinds = args.kinds.split(",")
    truths: dict[str, Any] = {}
    families: dict[str, list[SolvedExample]] = {}
    for kind in kinds:
        rows = problems(kind, "development")
        families[kind] = []
        for row in rows:
            truths[row["key"]] = row["truth"]
            # The known answer shown to her model is the answer as the benchmark writes it.
            families[kind].append(SolvedExample(row["key"], row["problem"], str(row.get("answer_text", row["truth"]))))
    graders = {kind: grader_for(kind) for kind in kinds}

    def agree(returned: str, example: SolvedExample) -> bool:
        return bool(graders[example.key.split(":")[0]](returned, truths[example.key]))

    round_counter = {"n": 0}

    if args.fake_proposer:
        replies = json.loads(args.fake_proposer.read_text())

        def propose(requests: list[str]) -> list[str]:
            return [replies[i % len(replies)] for i in range(len(requests))]
        context = None
    else:
        from mlx_lm import load

        from core.runtime.model_lane_control import standalone_model_lane
        from tools.g12_batched import decode_stream

        model_path = args.model.expanduser().resolve(strict=True)
        context = standalone_model_lane(owner_id=f"g09-induce:{output.name}", model_path=str(model_path),
                                        purpose="evaluation", preemptible=False, require_exclusive=True,
                                        allow_owner_eviction=True, metadata={"tool": Path(__file__).name})
        context.__enter__()
        model, tokenizer = load(str(model_path))

        def propose(requests: list[str]) -> list[str]:
            texts = [""] * len(requests)
            started = time.monotonic()
            log = output / "requests" / f"round{round_counter['n']}.jsonl"

            def record(index: int, decoded: dict) -> None:
                texts[index] = decoded["public_text"] if decoded["termination"] == "stop" else ""
                with log.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"index": index, "request_sha256": hashlib.sha256(
                        requests[index].encode()).hexdigest(), **decoded}) + "\n")
                print(json.dumps({"round": round_counter["n"], "index": index, "of": len(requests),
                                  "termination": decoded["termination"], "tokens": decoded["generated_tokens"],
                                  "seconds": decoded["seconds"]}), flush=True)

            decode_stream(model, tokenizer, [[{"role": "user", "content": r}] for r in requests],
                          max_tokens=args.max_tokens, width=args.batch, on_record=record)
            print(json.dumps({"round_done": round_counter["n"], "seconds": round(time.monotonic() - started, 1)}),
                  flush=True)
            round_counter["n"] += 1
            return texts

    candidates_path = output / "candidates.jsonl"

    def on_candidate(candidate) -> None:
        record = candidate_record(candidate)
        with candidates_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps({k: record[k] for k in ("family", "line", "round", "pool", "sealed", "admitted")}),
              flush=True)

    try:
        from core.reasoning.mechanisms import reference

        state = induce(families, propose, shots=args.shots, lines=args.lines, rounds=args.rounds,
                       agree=agree, workers=args.workers, mechanisms=reference() if args.mechanisms else None,
                       on_candidate=on_candidate)
    finally:
        if context is not None:
            context.__exit__(None, None, None)
    book = ProcedureBook.from_families(state)
    (output / "book.json").write_text(dumps(book.to_json()), encoding="utf-8")
    summary = {name: {"development": len(f.pool) + len(f.sealed), "candidates": len(f.candidates),
                      "admitted": len(f.admitted),
                      "best": max(([c.pool_agreed + c.sealed_agreed, len(c.pool) + len(c.sealed)]
                                   for c in f.candidates), default=None),
                      "nearest_required": round(f.signature.nearest_required, 4)}
               for name, f in state.items()}
    (output / "summary.json").write_text(dumps(summary), encoding="utf-8")
    print(dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
