"""The problems: read from the benchmarks' own files, in an order fixed before any run."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from outside_bar import planbench
from outside_bar.grade import BENCH

HERE = Path(__file__).resolve().parent
PANEL = HERE / "panel_v1.json"
DOMAINS = ("blocksworld", "mystery_blocksworld", "math500", "bbeh_mini")
#: The instruction MATH-500's published results use.
MATH_INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."
#: The instruction of the BBEH paper (Kazemi et al. 2025).
BBEH_INSTRUCTION = (
    'Think step by step, and when you provide the final answer, please use the prefix "The answer is:" without any '
    'modification, and provide the answer directly, with no formatting, no bolding, and no markup. For instance: '
    '"The answer is: 42" or "The answer is: yes". If the question is multiple choice with a single correct answer, '
    'the final answer must only be the letter corresponding to the correct answer. For example, "The answer is: (a)"')
DATA = {
    "blocksworld": BENCH / "g09/planbench/src/plan-bench/prompts/blocksworld/task_1_plan_generation.json",
    "mystery_blocksworld": BENCH / "g09/planbench/src/plan-bench/prompts/mystery_blocksworld/task_1_plan_generation.json",
    "math500": BENCH / "math500/test.jsonl",
    "bbeh_mini": BENCH / "g09/bbeh/mini/data.json",
}


def _short(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def every_task(domain: str) -> dict[str, dict[str, Any]]:
    """Every problem of a benchmark, by a stable id."""
    if domain in ("blocksworld", "mystery_blocksworld"):
        return {task["id"]: {**task, "domain": domain} for task in planbench.tasks(domain)}
    if domain == "math500":
        rows = [json.loads(line) for line in DATA[domain].read_text(encoding="utf-8").splitlines() if line.strip()]
        return {f"math500:{_short(row['problem'])}": {
            "id": f"math500:{_short(row['problem'])}", "domain": domain, "truth": row["answer"],
            "request": f"{row['problem']}\n\n{MATH_INSTRUCTION}"} for row in rows}
    if domain == "bbeh_mini":
        rows = json.loads(DATA[domain].read_text(encoding="utf-8"))["examples"]
        return {f"bbeh_mini:{_short(row['input'])}": {
            "id": f"bbeh_mini:{_short(row['input'])}", "domain": domain, "truth": row["target"],
            "request": f"{row['input']}\n\n{BBEH_INSTRUCTION}"} for row in rows}
    raise KeyError(domain)


def tasks(each: int, part: str = "panel") -> list[dict[str, Any]]:
    """The first ``each`` problems of every domain, each checked against the text frozen for it."""
    frozen = json.loads(PANEL.read_text(encoding="utf-8"))
    out = []
    for domain in DOMAINS:
        known = every_task(domain)
        for task_id in frozen["domains"][domain][part][:each]:
            task = known[task_id]
            if _short(task["request"]) != frozen["requests"][task_id]:
                raise ValueError(f"{task_id} is not the problem that was frozen")
            out.append(task)
    return out
