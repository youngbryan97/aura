#!/usr/bin/env python3
"""Her answers to MATH-500, graded by the reference grader, for comparison with named models.

G12 asks for frontier performance against named current baselines on
independent broad tasks, with resources and tool access reported fairly.
MATH-500 is the 500-problem test split OpenAI drew from Hendrycks' MATH for
PRM800K (`prm800k/math_splits/test.jsonl`); published results for current
models use it with the instruction below, so her number can stand beside
theirs with the differences stated rather than hidden.

Each problem is one free, greedy decode by her persona 27B through her own
chat template, native thinking on, at the medium reasoning effort she serves
with. No tools, no retrieval, no retries, no sampling: one answer per
problem. The user turn is the problem followed by the standard instruction,
"Please reason step by step, and put your final answer within \\boxed{}.",
which the baselines' own evaluations use; nothing is added for her.

The answer is the last ``\\boxed{...}`` in her public reply; her private
reasoning is never read. It is graded by PRM800K's ``grade_answer`` (MIT,
OpenAI), unmodified, from ``--grader``. An answer with no box, or a decode
stopped at the token budget, is wrong and counted as such. Tokens, wall
time and termination are kept per problem so compute is reported.

Usage:
    run_g12_math500.py --problems FILE --grader DIR --output DIR [--limit N]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import signal
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#: sha256 of prm800k/math_splits/test.jsonl as its Git LFS pointer names it.
MATH500_SHA256 = "35dc41080a3680858b27fa7e0533d2d547825316fc5dafe5d316f4ccc5a06132"
INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."


def last_boxed(text: str) -> str | None:
    """The contents of the last ``\\boxed{...}`` (or ``\\fbox{...}``), braces matched."""
    start = max(text.rfind("\\boxed"), text.rfind("\\fbox"))
    if start < 0:
        return None
    opening = text.find("{", start)
    if opening < 0:
        return None
    depth = 0
    for index in range(opening, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[opening + 1 : index]
    return None


class _GradingTimeout(Exception):
    pass


def grade(grade_answer: Any, given: str | None, truth: str, *, seconds: int) -> tuple[bool, str]:
    """The reference grader's verdict, or False with the reason it could not give one."""
    if given is None:
        return False, "no_boxed_answer"

    def expire(_signum: int, _frame: Any) -> None:
        raise _GradingTimeout

    previous = signal.signal(signal.SIGALRM, expire)
    signal.alarm(seconds)
    try:
        return bool(grade_answer(given, truth)), ""
    except _GradingTimeout:
        return False, "grader_timeout"
    except Exception as exc:  # noqa: BLE001 - the reference grader's failure is recorded, not hidden
        return False, f"grader_error:{type(exc).__name__}"
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problems", type=Path, required=True)
    parser.add_argument("--grader", type=Path, required=True,
                        help="a directory holding PRM800K's grading/ package")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path,
                        default=Path("~/.aura/models/Aura-Qwen3.8-27B-persona-crsm-7f6a2e83f73f5eef9d15"))
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--limit", type=int, default=0, help="the first N problems only")
    parser.add_argument("--batch", type=int, default=1,
                        help="problems decoded together (tools/g12_batched.py); 1 decodes one at a time")
    args = parser.parse_args()
    if args.batch < 1:
        raise SystemExit("--batch is at least 1")

    raw = args.problems.expanduser().read_bytes()
    if hashlib.sha256(raw).hexdigest() != MATH500_SHA256:
        raise SystemExit("the problems file is not PRM800K's MATH-500 test split")
    problems = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    if args.limit:
        problems = problems[: args.limit]
    sys.path.insert(0, str(args.grader.expanduser()))
    from grading.grader import grade_answer

    from mlx_lm import load

    from core.runtime.model_lane_control import standalone_model_lane
    from tools.run_g05_public_answers import decode_public

    output = args.output.expanduser()
    rows_dir = output / "rows"
    rows_dir.mkdir(parents=True, exist_ok=True)
    pending = [p for p in problems
               if not (rows_dir / f"{hashlib.sha256(p['unique_id'].encode()).hexdigest()[:24]}.json").exists()]
    print(json.dumps({"problems": len(problems), "pending": len(pending)}), flush=True)
    if not pending:
        return 0
    model_path = args.model.expanduser().resolve(strict=True)
    with standalone_model_lane(owner_id=f"g12-math500:{output.name}", model_path=str(model_path),
                               purpose="evaluation", preemptible=False, require_exclusive=True,
                               allow_owner_eviction=True, metadata={"tool": Path(__file__).name}):
        model, tokenizer = load(str(model_path))
        from tools.g12_batched import decode_batch

        done = 0
        for start in range(0, len(pending), args.batch):
            group = pending[start : start + args.batch]
            conversations = [[{"role": "user", "content": f"{p['problem']}\n\n{INSTRUCTION}"}] for p in group]
            if args.batch == 1:
                results = [decode_public(model, tokenizer, conversations[0], max_tokens=args.max_tokens)]
            else:
                results = decode_batch(model, tokenizer, conversations, max_tokens=args.max_tokens)
            for problem, decoded in zip(group, results, strict=True):
                path = rows_dir / f"{hashlib.sha256(problem['unique_id'].encode()).hexdigest()[:24]}.json"
                given = last_boxed(decoded["public_text"])
                correct, grading_note = grade(grade_answer, given, problem["answer"], seconds=30)
                row = {
                    "unique_id": problem["unique_id"], "subject": problem["subject"], "level": problem["level"],
                    "answer": problem["answer"], "given": given, "correct": correct and
                    decoded["termination"] == "stop", "grading_note": grading_note, **decoded,
                }
                temporary = path.with_suffix(".tmp")
                temporary.write_text(json.dumps(row, indent=1), encoding="utf-8")
                temporary.replace(path)
                done += 1
                print(json.dumps({"done": done, "of": len(pending), "correct": row["correct"],
                                  "tokens": decoded["generated_tokens"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
