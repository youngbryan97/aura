#!/usr/bin/env python3
"""Her answer, and the same answer after her reasoning organ, on five kinds of task (G09).

Two arms per task, paired:

* ``ordinary``: one greedy decode by her persona 27B through her own chat
  template, private channel open at her serving effort, as her runtime
  answers a question nothing upstream has settled
  (chat_format.thinking_enabled_for_generation);
* ``organ``: that answer handed to her reasoning organ the way her response
  phase hands it (ResponseGenerationPhase._maybe_amplify_response): admitted
  only where is_amplifiable admits the request, seeded with the draft, the
  draft verified first, a search of the amplifier's own planned size given
  all the time a live turn can give it (the user-facing ceiling, less what
  the draft spent and the phase's reserve) and stopped at that deadline,
  candidates generated through the same system message and template at the
  amplifier's temperatures, and the draft replaced only by an answer with
  checked-verifier or independent-executable-consensus authority. Sealed
  from her memory and caches; read-only.

Domains, each with the grader its source publishes: math (MATH test problems
outside MATH-500, graded by MATH-500's grader), code (HumanEval+, run against
its hidden tests in her kernel-bounded sandbox), planning (Natural Plan
calendar scheduling from its 5-shot prompt, its own parser against the golden
plan), knowledge (HotpotQA distractor questions over their supplied passages,
exact match after HotpotQA's normalisation), transfer (BIG-Bench Hard tasks
in its 3-shot chain-of-thought protocol, BBH's own answer form). Rows are
written once and the run resumes.

Usage:
    run_g09_organ.py --domain DOMAIN --tasks N --seed S --output DIR [--exclude FILE ...]
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import hashlib
import json
import random
import re
import string
import sys
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DOMAINS = ("math", "code", "planning", "knowledge", "transfer", "bbeh")
BENCH = Path("~/.aura/benchmarks").expanduser()
#: EvalPlus's own request for a chat model.
CODE_REQUEST = ("Please provide a self-contained Python script that solves the following problem "
                "in a markdown code block:\n```\n{prompt}\n```")
#: BIG-Bench Extra Hard's evaluation instruction, appended to every question
#: (Kazemi et al. 2025, arXiv:2502.19187).
BBEH_SUFFIX = (
    "Think step by step, and when you provide the final answer, please use the prefix \"The answer is:\" "
    "without any modification, and provide the answer directly, with no formatting, no bolding, and no "
    "markup. For instance: \"The answer is: 42\" or \"The answer is: yes\". If the question is multiple "
    "choice with a single correct answer, the final answer must only be the letter corresponding to the "
    "correct answer. For example, \"The answer is: (a)\""
)
KNOWLEDGE_REQUEST = ("Answer the question from the passages below. Give the answer as a short phrase.\n\n"
                     "{passages}\n\nQuestion: {question}")


# ── Tasks ──────────────────────────────────────────────────────────────────

def _math_tasks() -> list[dict[str, Any]]:
    import pandas as pd

    from tools.run_g12_math500 import INSTRUCTION, last_boxed

    held = {json.loads(line)["problem"].strip()
            for line in (BENCH / "math500/test.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()}
    tasks = []
    for path in sorted((BENCH / "g09/math").glob("*-test.parquet")):
        for index, row in pd.read_parquet(path).iterrows():
            if row["problem"].strip() in held:
                continue
            truth = last_boxed(row["solution"])
            if truth is None:
                continue
            tasks.append({"id": f"math:{path.stem}:{index}", "request": f"{row['problem']}\n\n{INSTRUCTION}",
                          "truth": truth, "group": f"{row['type']}|{row['level']}"})
    return tasks


def _code_tasks() -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in gzip.open(BENCH / "g09/humanevalplus/HumanEvalPlus.jsonl.gz", "rt")]
    return [{"id": row["task_id"], "request": CODE_REQUEST.format(prompt=row["prompt"].strip()),
             "truth": {"test": row["test"], "entry_point": row["entry_point"], "prompt": row["prompt"],
                       "canonical": row["canonical_solution"], "atol": row.get("atol") or 0,
                       "inputs": list(row["base_input"]) + list(row["plus_input"])},
             "group": "humaneval+"} for row in rows]


def _planning_tasks() -> list[dict[str, Any]]:
    data = json.loads((BENCH / "g09/natural-plan/calendar_scheduling.json").read_text(encoding="utf-8"))
    # The benchmark's own 5-shot prompt: its solved examples are what show the
    # "Day, HH:MM - HH:MM" form its parser reads.
    return [{"id": f"calendar:{key}", "request": value["prompt_5shot"], "truth": value["golden_plan"],
             "group": f"people={value['num_people']}|days={value['num_days']}"} for key, value in data.items()]


def _knowledge_tasks() -> list[dict[str, Any]]:
    import pandas as pd

    tasks = []
    for _, row in pd.read_parquet(BENCH / "g09/hotpotqa/validation.parquet").iterrows():
        passages = "\n\n".join(f"{title}: {' '.join(sentences)}" for title, sentences in
                               zip(row["context"]["title"], row["context"]["sentences"], strict=True))
        tasks.append({"id": f"hotpot:{row['id']}", "request": KNOWLEDGE_REQUEST.format(
            passages=passages, question=row["question"]), "truth": row["answer"],
            "group": f"{row['type']}|{row['level']}"})
    return tasks


def _transfer_tasks() -> list[dict[str, Any]]:
    from tools.run_g09_bbh import cot_examples, request_text

    tasks = []
    for path in sorted((BENCH / "bbh-src/bbh").glob("*.json")):
        worked = cot_examples(BENCH / "bbh-src", path.stem)
        for index, example in enumerate(json.loads(path.read_text(encoding="utf-8"))["examples"]):
            tasks.append({"id": f"{path.stem}:{index}", "request": request_text(example["input"], worked),
                          "truth": example["target"], "group": path.stem})
    return tasks


def _bbeh_tasks() -> list[dict[str, Any]]:
    tasks = []
    for path in sorted((BENCH / "g09/bbeh").glob("bbeh_*/task.json")):
        for index, example in enumerate(json.loads(path.read_text(encoding="utf-8"))["examples"]):
            tasks.append({"id": f"{path.parent.name}:{index}", "request": f"{example['input']}\n\n{BBEH_SUFFIX}",
                          "truth": example["target"], "group": path.parent.name})
    return tasks


LOADERS = {"math": _math_tasks, "code": _code_tasks, "planning": _planning_tasks,
           "knowledge": _knowledge_tasks, "transfer": _transfer_tasks, "bbeh": _bbeh_tasks}


def sample(tasks: list[dict[str, Any]], count: int, seed: int, excluded: set[str]) -> list[dict[str, Any]]:
    """``count`` tasks spread across groups, reproducibly, none of the excluded ids."""
    pool = [task for task in tasks if task["id"] not in excluded]
    rng = random.Random(seed)
    rng.shuffle(pool)
    by_group: dict[str, list[dict[str, Any]]] = {}
    for task in pool:
        by_group.setdefault(task["group"], []).append(task)
    chosen: list[dict[str, Any]] = []
    groups = sorted(by_group)
    while len(chosen) < count and any(by_group[group] for group in groups):
        for group in groups:
            if by_group[group] and len(chosen) < count:
                chosen.append(by_group[group].pop())
    return chosen


# ── Graders ────────────────────────────────────────────────────────────────

def grade_math(text: str, truth: str) -> tuple[bool, str]:
    from tools.run_g12_math500 import grade, last_boxed

    sys.path.insert(0, str(BENCH / "math500"))
    from grading.grader import grade_answer  # MATH-500's own grader

    return grade(grade_answer, last_boxed(text), truth, seconds=10)


_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)


_PLUS_CHECK = """
import copy as _copy, math as _math


def _same(got, want, atol):
    if isinstance(want, float) or isinstance(got, float):
        return _math.isclose(got, want, rel_tol=1e-6, abs_tol=max(atol, 1e-6)) or got == want
    if isinstance(want, (list, tuple)) and isinstance(got, (list, tuple)):
        return len(got) == len(want) and all(_same(g, w, atol) for g, w in zip(got, want))
    return got == want


for _inputs in _INPUTS:
    _want = _reference(*_copy.deepcopy(_inputs))
    _got = {entry}(*_copy.deepcopy(_inputs))
    assert _same(_got, _want, _ATOL), _inputs
"""


def grade_code(text: str, truth: dict[str, Any]) -> tuple[bool, str]:
    """HumanEval+ as EvalPlus grades it: the base asserts, then every plus input against the reference."""
    from core.sandbox.untrusted_python import run_untrusted_script

    blocks = _FENCE.findall(text)
    code = blocks[-1] if blocks else ""
    if f"def {truth['entry_point']}" not in code:
        return False, "no_function"
    reference = (truth["prompt"] + truth["canonical"]).replace(
        f"def {truth['entry_point']}(", "def _reference(", 1)
    script = "\n\n".join([
        reference, code, truth["test"], f"check({truth['entry_point']})",
        f"_INPUTS = {truth['inputs']!r}\n_ATOL = {truth['atol']!r}",
        _PLUS_CHECK.format(entry=truth["entry_point"])])
    outcome = run_untrusted_script(script, timeout_s=120.0, require_boundary=True, source="g09_grader")
    return bool(outcome.ok and outcome.returncode == 0 and outcome.sandboxed), outcome.status


def _parse_calendar(response: str) -> tuple[str, float, float]:
    """Natural Plan's parser (evaluate_calendar_scheduling._parse_response), restated: the first
    "Day, HH:MM - HH:MM" in the reply, half hours as .5. Their module imports absl, which the
    runtime does not carry."""
    found = re.findall(r"[A-Za-z]+, [0-9]+:[0-9]+ - [0-9]+:[0-9]+", response)
    if not found:
        return "", -1.0, -1.0
    day, hours = found[0].split(",")[0].strip(), found[0].split(",")[1].strip()

    def number(hour: str) -> float:
        return float(hour.split(":")[0]) + (0.5 if hour.split(":")[1] == "30" else 0.0)

    return day, number(hours.split("-")[0].strip()), number(hours.split("-")[1].strip())


def grade_planning(text: str, truth: str) -> tuple[bool, str]:
    got, want = _parse_calendar(text), _parse_calendar(truth)
    return got == want and got[1] >= 0, f"{got}"


def _normalize(text: str) -> str:
    """HotpotQA's answer normalisation (its official evaluation script)."""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def grade_knowledge(text: str, truth: str) -> tuple[bool, str]:
    answer = text.strip().splitlines()[-1] if text.strip() else ""
    answer = re.sub(r"^(?:\*\*)?(?:final\s+)?answer\s*[:：]\s*", "", answer, flags=re.I).strip("* ")
    return _normalize(answer) == _normalize(truth), answer[:120]


def grade_transfer(text: str, truth: str) -> tuple[bool, str]:
    from tools.run_g09_bbh import extract_answer, matches

    answer = extract_answer(text)
    return matches(answer, truth), answer


def _bbeh_answer(sample: str) -> str:
    """BBEH's own extraction and preprocessing (bbeh/evaluate.py), restated: the text after
    the last answer prefix, LaTeX wrappers removed, lower-cased, first line, no bold."""
    answer = sample.strip()
    for prefix in ("The answer is:", "The final answer is ", "The final answer is: ", "The answer is "):
        if prefix in answer:
            answer = answer.split(prefix)[-1].strip()
    if answer.endswith("."):
        answer = answer[:-1]
    if answer.startswith("$") and answer.endswith("$"):
        answer = answer[1:-1]
    for wrapper in ("boxed{", "text{", "texttt{"):
        if wrapper in answer and answer.endswith("}"):
            answer = answer[0:-1].split(wrapper)[1]
    answer = answer.lower().replace(", ", ",").replace("**", "").split("\n")[0]
    return answer[0:-1] if answer.endswith(".") else answer


def _bbeh_match(prediction: str, reference: str) -> bool:
    """BBEH's fuzzy_match, restated."""
    if prediction == reference:
        return True
    if len(prediction) == 3 and prediction[0] == "(" and prediction[-1] == ")":
        return prediction[1] == reference
    if len(reference) == 3 and reference[0] == "(" and reference[-1] == ")":
        return reference[1] == prediction
    try:
        if float(prediction) == float(reference):
            return True
    except ValueError:
        pass
    if prediction.replace("'", "") == reference.replace("'", ""):
        return True
    if f"[{reference}]" == prediction or f"[{prediction}]" == reference:
        return True
    return prediction.endswith("?") and prediction[:-1] == reference


def grade_bbeh(text: str, truth: str) -> tuple[bool, str]:
    answer = _bbeh_answer(text)
    return _bbeh_match(answer, truth.strip().lower().replace(", ", ",")), answer[:120]


GRADERS = {"math": grade_math, "code": grade_code, "planning": grade_planning,
           "knowledge": grade_knowledge, "transfer": grade_transfer, "bbeh": grade_bbeh}


# ── The run ────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", choices=DOMAINS, required=True)
    parser.add_argument("--tasks", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exclude", type=Path, action="append", default=[],
                        help="files of task ids no development or earlier run may reuse")
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--batch", type=int, default=1,
                        help="drafts decoded together (tools/g12_batched.py); the organ always runs one at a time")
    parser.add_argument("--arms", default="ordinary,organ",
                        help="ordinary decodes her drafts; organ runs on drafts already written")
    parser.add_argument("--model", type=Path,
                        default=Path("~/.aura/models/Aura-Qwen3.8-27B-persona-crsm-7f6a2e83f73f5eef9d15"))
    args = parser.parse_args()

    excluded: set[str] = set()
    for path in args.exclude:
        excluded |= {line.strip() for line in path.expanduser().read_text().splitlines() if line.strip()}
    tasks = sample(LOADERS[args.domain](), args.tasks, args.seed, excluded)
    output = args.output.expanduser()
    arms = {arm.strip() for arm in args.arms.split(",")}
    if not arms or not arms <= {"ordinary", "organ"}:
        raise SystemExit("--arms names ordinary and/or organ")
    (output / "rows").mkdir(parents=True, exist_ok=True)
    (output / "task_ids.txt").write_text("".join(f"{task['id']}\n" for task in tasks), encoding="utf-8")

    import mlx.core as mx
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler

    from core.brain.executable_reasoning import should_use_executable_reasoning
    from core.brain.llm.chat_format import (
        reasoning_effort_for_generation,
        render_chat_template,
        split_native_thinking_generation,
    )
    from core.brain.reasoning_amplifier_v2 import (
        AMPLIFIER_CANDIDATE_SYSTEM,
        amplify_turn,
        is_amplifiable,
        planned_new_candidates,
    )
    from core.runtime.model_lane_control import standalone_model_lane
    from core.runtime.response_policy import USER_FACING_COMPLETION_DEADLINE_MAX_S
    from tools.run_g05_public_answers import decode_public

    grade = GRADERS[args.domain]
    model_path = args.model.expanduser().resolve(strict=True)
    print(json.dumps({"domain": args.domain, "tasks": len(tasks), "excluded": len(excluded)}), flush=True)
    with standalone_model_lane(owner_id=f"g09-organ:{output.name}", model_path=str(model_path),
                               purpose="evaluation", preemptible=False, require_exclusive=True,
                               allow_owner_eviction=True, metadata={"tool": Path(__file__).name}):
        model, tokenizer = load(str(model_path))
        model_lock = threading.Lock()
        generations: list[dict[str, Any]] = []
        # When the turn's time runs out. A generation in a thread cannot be
        # cancelled from outside, so it reads this and stops itself.
        turn_deadline = [float("inf")]

        def sampled(messages: list[dict[str, str]], temperature: float) -> str:
            with model_lock:
                seed = int(hashlib.sha256(f"{messages}\0{temperature}".encode()).hexdigest()[:8], 16)
                mx.random.seed(seed)
                rendered = render_chat_template(tokenizer, messages, add_generation_prompt=True,
                                                enable_thinking=True,
                                                reasoning_effort=reasoning_effort_for_generation(thinking=True))
                ids = [int(t) for t in tokenizer.encode(rendered, add_special_tokens=False)]
                began, pieces = time.monotonic(), []
                cut = False
                for response in stream_generate(model, tokenizer, ids, max_tokens=args.max_tokens,
                                                sampler=make_sampler(temp=float(temperature), top_p=0.95)):
                    pieces.append(str(response.text or ""))
                    if time.monotonic() >= turn_deadline[0]:
                        cut = True
                        break
                channels = split_native_thinking_generation("".join(pieces), native_thinking=True)
                generations.append({"temperature": temperature, "seconds": round(time.monotonic() - began, 3),
                                    "closed": channels.boundary_closed, "cut_at_turn_deadline": cut})
                return "" if cut or not channels.boundary_closed else channels.surface

        async def generate(prompt: str, temperature: float) -> str:
            messages = [{"role": "system", "content": AMPLIFIER_CANDIDATE_SYSTEM},
                        {"role": "user", "content": prompt}]
            return await asyncio.to_thread(sampled, messages, temperature)

        def write_once(path: Path, row: dict[str, Any]) -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(row, indent=1, default=str), encoding="utf-8")
            temporary.replace(path)

        def row_name(task: dict[str, Any]) -> str:
            return f"{hashlib.sha256(task['id'].encode()).hexdigest()[:24]}.json"

        def ordinary_row(task: dict[str, Any], draft: dict[str, Any]) -> dict[str, Any]:
            draft_text = draft["public_text"] if draft["termination"] == "stop" else ""
            correct, note = grade(draft_text, task["truth"])
            return {"id": task["id"], "domain": args.domain, "group": task["group"], "correct": correct,
                    "note": note, "text": draft_text, "termination": draft["termination"],
                    "tokens": draft["generated_tokens"], "seconds": draft["seconds"],
                    "batch_size": draft.get("batch_size", 1)}

        if "ordinary" in arms and args.batch > 1:
            from tools.g12_batched import decode_batch

            pending = [task for task in tasks if not (output / "rows" / "ordinary" / row_name(task)).exists()]
            for start in range(0, len(pending), args.batch):
                group = pending[start : start + args.batch]
                drafts = decode_batch(model, tokenizer, [[{"role": "user", "content": task["request"]}]
                                                         for task in group], max_tokens=args.max_tokens)
                for task, draft in zip(group, drafts, strict=True):
                    write_once(output / "rows" / "ordinary" / row_name(task), ordinary_row(task, draft))
                print(json.dumps({"ordinary_batched": start + len(group), "of": len(pending)}), flush=True)

        for done, task in enumerate(tasks, 1):
            name = row_name(task)
            ordinary_path, organ_path = output / "rows" / "ordinary" / name, output / "rows" / "organ" / name
            if "ordinary" in arms and not ordinary_path.exists():
                with model_lock:
                    draft = decode_public(model, tokenizer, [{"role": "user", "content": task["request"]}],
                                          max_tokens=args.max_tokens, thinking=True)
                row = ordinary_row(task, draft)
                write_once(ordinary_path, row)
                print(json.dumps({"done": done, "of": len(tasks), "arm": "ordinary", "correct": row["correct"]}),
                      flush=True)
            if "organ" not in arms or organ_path.exists() or not ordinary_path.exists():
                continue
            ordinary = json.loads(ordinary_path.read_text(encoding="utf-8"))
            draft_text = ordinary["text"]
            task_type = is_amplifiable(task["request"])
            organ: dict[str, Any] = {"id": task["id"], "admitted": task_type is not None, "task_type": task_type}
            delivered = draft_text
            budget = USER_FACING_COMPLETION_DEADLINE_MAX_S - float(ordinary["seconds"]) - 4.0
            if task_type is not None and draft_text and budget <= 0.0:
                organ["stood_down"] = "no_time_left_in_the_turn"
            elif task_type is not None and draft_text:
                executable = should_use_executable_reasoning(task["request"], task_type=task_type)
                sample_budget = 3 if executable else None
                planned = planned_new_candidates(task_type, sample_budget=sample_budget, seeds=1)
                generations.clear()
                began = time.monotonic()
                # All a live turn can give the search: its ceiling, less what the
                # draft spent and the reserve the response phase keeps.
                turn_deadline[0] = began + budget
                try:
                    result = asyncio.run(amplify_turn(
                        task["request"], generate, task_type=task_type, time_budget_s=budget,
                        sample_budget=sample_budget,
                        extra_context={"seed_candidates": [draft_text], "enable_executable_reasoning": executable,
                                       "allow_textual_fallback_after_executable": True,
                                       "generation_max_tokens": args.max_tokens,
                                       "disable_batched_candidates": True,
                                       "sealed_evaluation": True, "read_only_evaluation": True, "skip_cache": True}))
                    receipt = result.receipt.to_dict()
                    authority = str(receipt.get("promotion_authority") or "none")
                    promoted = ""
                    if authority == "checked_verifier":
                        promoted = str(result.answer or "").strip()
                    elif authority == "independent_executable_consensus":
                        promoted = str(result.source_answer or result.answer or "").strip()
                    if promoted:
                        delivered = promoted
                    organ.update({"authority": authority, "adopted": bool(promoted), "planned": planned,
                                  "budget_s": round(budget, 3), "fallbacks": receipt.get("fallbacks"),
                                  "num_candidates": receipt.get("num_candidates"), "verified": result.verified})
                except (TimeoutError, RuntimeError, ValueError, TypeError) as exc:
                    organ.update({"error": f"{type(exc).__name__}: {exc}", "planned": planned,
                                  "budget_s": round(budget, 3)})
                with model_lock:  # any generation a timeout left running ends first
                    pass
                turn_deadline[0] = float("inf")
                organ["seconds"] = round(time.monotonic() - began, 3)
                organ["generations"] = list(generations)
            organ["correct"], organ["note"] = ((ordinary["correct"], ordinary["note"]) if delivered == draft_text
                                               else grade(delivered, task["truth"]))
            organ["text"] = delivered if delivered != draft_text else None
            write_once(organ_path, organ)
            print(json.dumps({"done": done, "of": len(tasks), "arm": "organ", "ordinary": ordinary["correct"],
                              "organ": organ["correct"], "admitted": organ["admitted"],
                              "adopted": organ.get("adopted")}), flush=True)
    ordinary_rows = {path.name: json.loads(path.read_text()) for path in (output / "rows" / "ordinary").glob("*.json")}
    organ_rows = {path.name: json.loads(path.read_text()) for path in (output / "rows" / "organ").glob("*.json")}
    summary = Counter((ordinary_rows[name]["correct"], organ_rows[name]["correct"]) for name in organ_rows)
    print(json.dumps({"ordinary": len(ordinary_rows), "ordinary_correct": sum(r["correct"] for r in ordinary_rows.values()),
                      "organ": len(organ_rows), "both": summary[(True, True)], "organ_only": summary[(False, True)],
                      "ordinary_only": summary[(True, False)], "neither": summary[(False, False)]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
