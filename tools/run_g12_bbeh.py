#!/usr/bin/env python3
"""Her answers to BIG-Bench Extra Hard Mini, asked and graded as its authors do.

BBEH (Kazemi et al. 2025, arXiv:2502.19187) replaces each BIG-Bench Hard task
with a harder one; its leaderboard reports named current systems on the 460
questions of BBEH Mini under one protocol, which is what lets her number
stand beside theirs. Each question is the example's input followed by the
paper's instruction to answer after "The answer is:"; the answer is read and
matched by the authors' evaluator (restated in tools/run_g09_organ.py, whose
restatement reproduces the evaluator's own checks). One greedy decode through
the model's chat template, private reasoning on at her serving effort, no
tools, no retries; a reply stopped at its budget has no answer. Rows are
written once and the run resumes. Each row is labelled with the task it was
drawn from, recovered by matching the full task files.

Usage:
    run_g12_bbeh.py --mini FILE --tasks DIR --output DIR [--model DIR] [--batch N] [--only-tasks a,b]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.benchmark_case_identity import CaseCatalog, SourceCase, source_cases  # noqa: E402

#: sha256 of bbeh/mini/data.json at google-deepmind/bbeh main, read 2026-10-07.
BBEH_MINI_SHA256 = "14e77b3d6be68faa008d268abf53b1f8d2420ffdd762504a304dafc3f8d43026"


def load_cases(mini_path: Path) -> tuple[list[dict[str, Any]], CaseCatalog]:
    """The pinned complete source and its identities, before task selection."""
    raw = mini_path.expanduser().read_bytes()
    if hashlib.sha256(raw).hexdigest() != BBEH_MINI_SHA256:
        raise ValueError("the file is not BBEH Mini as pinned")
    examples = json.loads(raw)["examples"]
    catalog = source_cases(examples, legacy_id=lambda e: hashlib.sha256(e["input"].encode()).hexdigest()[:24],
                           reference=lambda e: e["target"])
    return examples, catalog


def validate_saved_result(row: dict[str, Any]) -> None:
    """A case binding alone is not evidence that its decode completed."""
    termination = row.get("termination")
    if not isinstance(termination, str) or termination not in {
        "stop", "token_limit", "length", "native_thinking_incomplete"
    }:
        raise ValueError("saved BBEH row has no valid decode termination")
    if not isinstance(row.get("public_text"), str):
        raise ValueError("saved BBEH row has no public text")
    tokens = row.get("generated_tokens")
    if type(tokens) is not int or tokens < 0:
        raise ValueError("saved BBEH row has no valid generated token count")
    seconds = row.get("seconds")
    if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0:
        raise ValueError("saved BBEH row has no valid decode duration")
    if type(row.get("correct")) is not bool or (termination != "stop" and row["correct"]):
        raise ValueError("saved BBEH row has no valid correctness verdict")


def saved_results(rows_dir: Path, catalog: CaseCatalog) -> dict[str, dict[str, Any]]:
    saved = catalog.saved_rows(rows_dir)
    for row in saved.values():
        validate_saved_result(row)
    return saved


def publish_result(path: Path, row: dict[str, Any]) -> None:
    """Publish a completed result atomically without replacing prior evidence."""
    from core.governance_context import local_internal_governed_scope
    from core.runtime.file_write_gateway import get_file_write_gateway

    validate_saved_result(row)
    with local_internal_governed_scope("g12_bbeh.result", domain="file_write"):
        created = get_file_write_gateway().write_bytes_if_absent(
            path, json.dumps(row, indent=1).encode("utf-8"), source="g12_bbeh.result")
    if not created:
        raise ValueError(f"refusing to overwrite saved source case {row['id']!r}")


def task_of(tasks_dir: Path) -> dict[str, str]:
    """Each full-set input mapped to the task it belongs to."""
    out = {}
    for path in sorted(tasks_dir.glob("bbeh_*/task.json")):
        for example in json.loads(path.read_text(encoding="utf-8"))["examples"]:
            out[example["input"]] = path.parent.name
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mini", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True, help="the bbeh/benchmark_tasks directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path,
                        default=Path("~/.aura/models/Aura-Qwen3.8-27B-persona-crsm-7f6a2e83f73f5eef9d15"))
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--only-tasks", default="",
                        help="comma-separated task names: answer only these now; a later run without it does the rest")
    args = parser.parse_args()
    if args.batch < 1:
        raise SystemExit("--batch is at least 1")

    examples, catalog = load_cases(args.mini)
    labels = task_of(args.tasks.expanduser())
    from tools.run_g09_organ import BBEH_SUFFIX, grade_bbeh

    output = args.output.expanduser()
    rows_dir = output / "rows"
    rows_dir.mkdir(parents=True, exist_ok=True)

    saved = saved_results(rows_dir, catalog)
    pending = [case for case in catalog.cases if case.id not in saved]
    if args.only_tasks:
        wanted = set(args.only_tasks.split(","))
        pending = [case for case in pending if labels.get(examples[case.source_ordinal]["input"]) in wanted]
    print(json.dumps({"questions": len(examples), "pending": len(pending),
                      "unlabelled": sum(e["input"] not in labels for e in examples)}), flush=True)
    if not pending:
        return 0

    from mlx_lm import load

    from core.runtime.model_lane_control import standalone_model_lane
    from tools.g12_batched import decode_stream
    from tools.run_g05_public_answers import decode_public

    model_path = args.model.expanduser().resolve(strict=True)
    with standalone_model_lane(owner_id=f"g12-bbeh:{output.name}", model_path=str(model_path),
                               purpose="evaluation", preemptible=False, require_exclusive=True,
                               allow_owner_eviction=True, metadata={"tool": Path(__file__).name}):
        model, tokenizer = load(str(model_path))
        done = 0

        def write(case: SourceCase, decoded: dict) -> None:
            nonlocal done
            example = examples[case.source_ordinal]
            public = decoded["public_text"] if decoded["termination"] == "stop" else ""
            correct, answer = grade_bbeh(public, example["target"])
            row = {**decoded, "id": case.id, **case.metadata(),
                   "task": labels.get(example["input"], "unlabelled"),
                   "target": example["target"], "answer": answer, "correct": correct}
            path = rows_dir / f"{case.id}.json"
            publish_result(path, row)
            done += 1
            print(json.dumps({"done": done, "of": len(pending), "correct": correct}), flush=True)

        conversations = [[{"role": "user", "content": f"{examples[case.source_ordinal]['input']}\n\n{BBEH_SUFFIX}"}]
                         for case in pending]
        if args.batch == 1:
            for example, conversation in zip(pending, conversations, strict=True):
                write(example, decode_public(model, tokenizer, conversation, max_tokens=args.max_tokens))
        else:
            decode_stream(model, tokenizer, conversations, max_tokens=args.max_tokens, width=args.batch,
                          on_record=lambda index, decoded: write(pending[index], decoded))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
