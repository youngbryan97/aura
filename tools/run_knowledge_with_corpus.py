#!/usr/bin/env python3
"""Knowledge questions: her ordinary answer, her answer with her corpus in front of her, and declining when unsure.

SimpleQA (OpenAI, 4,326 short factual questions, each with one answer)
scores an answer as correct, wrong, or declined, and a declined answer above
a wrong one. Three arms on the same questions:

* ordinary: the question alone, the G12 protocol (private reasoning on at her
  serving effort, greedy, up to ``--max-tokens``);
* grounded: the paragraphs of her corpus about what the question names
  (core/knowledge/passages_about.py, the careful lookup; her conversational
  lookup when it finds nothing), given to her as her runtime gives a skill
  result, then the question; greedy;
* declining: the grounded question answered again, sampled at her runtime's
  ``llm.deep_temperature``; when the two grounded answers do not agree she
  declines, otherwise the greedy grounded answer stands;
* grounded, fused: as grounded, with as many passages, chosen by reciprocal
  rank fusion of the careful lookup and the index of every paragraph
  (core/knowledge/passage_index.py); greedy. Left out when there is no index.
  On these questions the answer was within the first three passages for 71
  of 200 fused against 54 careful (15 October, 5 s per lookup).

The grader is deterministic and the same for every arm. An answer is read
from the reply's last line. It counts as declined when it says it does not
know or cannot answer, as correct when the gold answer's words all appear in
it (numbers compared as numbers), and as wrong otherwise. OpenAI grades with
a model; this rule is stricter about phrasing and is reported as it is.
Questions are split into halves by a hash of their text.

MMLU-Pro's knowledge subjects run the same arms (``--benchmark mmlu_pro``, ``--categories``): the lookup is
made with the question alone, not its instructions or options, and every arm is graded by MMLU-Pro's own
reading of the chosen letter (tools/run_general_reasoning.answer_of); two answers agree when they choose the
same letter.

Usage:
    run_knowledge_with_corpus.py --part development --count N --seed S --output DIR [--batch 8] [--max-tokens 32768]
        [--benchmark simpleqa|mmlu_pro] [--categories history,psychology,...]
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SIMPLEQA = Path("~/.aura/benchmarks/gr/simpleqa/simple_qa_test_set.csv").expanduser()
_DECLINED = re.compile(r"\b(i do(?:n't| not) know|not sure|cannot (?:be )?(?:determine|answer|find)|can't (?:determine|"
                       r"answer|find)|unable to (?:determine|answer|find)|no (?:reliable )?information|unknown)\b",
                       re.IGNORECASE)
_ARTICLES = {"a", "an", "the"}


def tasks(part: str) -> list[dict[str, Any]]:
    with SIMPLEQA.open(encoding="utf-8") as handle:
        rows = [{"id": "simpleqa:" + hashlib.sha256(r["problem"].encode()).hexdigest()[:16], "request": r["problem"],
                 "truth": r["answer"], "group": str(ast.literal_eval(r["metadata"]).get("topic", ""))}
                for r in csv.DictReader(handle)]
    ordered = sorted(rows, key=lambda r: r["id"])
    return ordered[: len(ordered) // 2] if part == "development" else ordered[len(ordered) // 2:]


#: MMLU-Pro's subjects whose questions turn on what is known rather than on a computation.
KNOWLEDGE_CATEGORIES = ("history", "psychology", "health", "law", "philosophy", "other")


def mmlu_pro_knowledge_tasks(part: str, categories: tuple[str, ...]) -> list[dict[str, Any]]:
    """MMLU-Pro's questions of ``categories``, each with the question alone as what is looked up."""
    from tools.run_general_reasoning import mmlu_pro_tasks

    out = []
    for task in mmlu_pro_tasks(part):
        if task["group"] not in categories:
            continue
        body = task["request"].split("Question:\n", 1)[-1].split("\nOptions:", 1)[0].strip()
        out.append({**task, "lookup": body})
    return out


def grade_letter(reply: str, gold: str) -> str:
    """MMLU-Pro: "correct" when the chosen letter is the gold one, "declined" when none is chosen and the reply
    says it does not know, "wrong" otherwise."""
    from tools.run_general_reasoning import answer_of

    chosen = answer_of("mmlu_pro", reply)
    if chosen:
        return "correct" if chosen == str(gold) else "wrong"
    return "declined" if _DECLINED.search(final_line(reply)) else "wrong"


def agree_letter(first: str, second: str) -> bool:
    from tools.run_general_reasoning import answer_of

    a, b = answer_of("mmlu_pro", first), answer_of("mmlu_pro", second)
    return bool(a) and a == b


def _words(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", str(text or "").lower()) if w not in _ARTICLES]


def final_line(reply: str) -> str:
    lines = [line.strip() for line in str(reply or "").splitlines() if line.strip()]
    return lines[-1] if lines else ""


def grade(reply: str, gold: str) -> str:
    """"correct", "wrong" or "declined"."""
    answer = final_line(reply)
    if not answer or _DECLINED.search(answer):
        return "declined"
    have, need = set(_words(answer)), _words(gold)
    return "correct" if need and all(w in have for w in need) else "wrong"


def agree(first: str, second: str) -> bool:
    """Two answers agree when the words of the shorter all appear in the longer."""
    a, b = _words(final_line(first)), _words(final_line(second))
    if not a or not b:
        return False
    short, long_ = (a, set(b)) if len(a) <= len(b) else (b, set(a))
    return all(w in long_ for w in short)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--part", choices=("development", "test"), required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--model", type=Path,
                        default=Path("~/.aura/models/Aura-Qwen3.8-27B-persona-crsm-7f6a2e83f73f5eef9d15"))
    parser.add_argument("--benchmark", choices=("simpleqa", "mmlu_pro"), default="simpleqa")
    parser.add_argument("--categories", default=",".join(KNOWLEDGE_CATEGORIES))
    args = parser.parse_args()
    if args.benchmark == "mmlu_pro":
        chosen_tasks = mmlu_pro_knowledge_tasks(args.part, tuple(c.strip() for c in args.categories.split(",")))
        grade_one, agree_two = grade_letter, agree_letter
    else:
        chosen_tasks, grade_one, agree_two = tasks(args.part), grade, agree

    from core.config import config
    from core.knowledge.corpus_grounding import corpus_grounding_for
    from core.knowledge.local_corpus import SEARCH_DEADLINE_S
    from core.knowledge.passage_index import PassageIndex, fused
    from core.knowledge.passages_about import passages_about
    from tools.run_g09_organ import sample
    from tools.run_general_reasoning import TOP_P, Rows, evidence_block

    output = args.output.expanduser()
    rows = Rows(output)
    chosen = sample(chosen_tasks, args.count, args.seed, set())
    output.mkdir(parents=True, exist_ok=True)
    (output / "task_ids.txt").write_text("".join(f"{t['id']}\n" for t in chosen), encoding="utf-8")
    index = PassageIndex()
    with_index = index.available()
    for task in chosen:
        careful = None
        if rows.get("lookup", task["id"]) is None:
            asked = task.get("lookup") or task["request"]
            careful = passages_about(asked) or list(corpus_grounding_for(asked).passages)
            rows.put("lookup", task["id"], {"passages": [f"{t}: {x}" for t, x in careful]})
        if with_index and rows.get("lookup_fused", task["id"]) is None:
            asked = task.get("lookup") or task["request"]
            if careful is None:
                careful = passages_about(asked) or list(corpus_grounding_for(asked).passages)
            every = [(p.title, p.paragraph) for p in index.search(asked, limit=max(1, len(careful)) * 4,
                                                                  deadline_s=SEARCH_DEADLINE_S)]
            both = fused([careful, every], limit=max(1, len(careful))) if (careful or every) else []
            rows.put("lookup_fused", task["id"], {"passages": [f"{t}: {x}" for t, x in both]})

    import mlx.core as mx  # noqa: F401 - loaded before the lane is held
    from mlx_lm import load
    from mlx_lm.sample_utils import make_sampler

    from core.runtime.model_lane_control import standalone_model_lane
    from tools.g12_batched import decode_stream

    model_path = args.model.expanduser().resolve(strict=True)
    with standalone_model_lane(owner_id=f"knowledge-corpus:{output.name}", model_path=str(model_path),
                               purpose="evaluation", preemptible=False, require_exclusive=True):
        model, tokenizer = load(str(model_path))
        sampler = make_sampler(temp=float(config.llm.deep_temperature), top_p=TOP_P)

        def run(stage: str, items: list[tuple[str, list[dict[str, str]]]], *, sampled: bool = False) -> None:
            todo = [(key, conversation) for key, conversation in items if rows.get(stage, key) is None]
            print(json.dumps({"stage": stage, "to_decode": len(todo)}), flush=True)
            if todo:
                decode_stream(model, tokenizer, [c for _, c in todo], max_tokens=args.max_tokens, width=args.batch,
                              on_record=lambda i, record: rows.put(stage, todo[i][0], {"id": todo[i][0], **record}),
                              sampler=sampler if sampled else None, seed=args.seed if sampled else None)

        def grounded(task: dict[str, Any], lookup: str = "lookup") -> list[dict[str, str]]:
            passages = rows.get(lookup, task["id"])["passages"]
            given = [{"role": "system", "content": evidence_block("From her corpus: " + " … ".join(passages))}] \
                if passages else []
            return [*given, {"role": "user", "content": task["request"]}]

        run("ordinary", [(t["id"], [{"role": "user", "content": t["request"]}]) for t in chosen])
        run("grounded", [(t["id"], grounded(t)) for t in chosen])
        run("grounded_sampled", [(t["id"], grounded(t)) for t in chosen], sampled=True)
        if with_index:
            run("grounded_fused", [(t["id"], grounded(t, "lookup_fused")) for t in chosen])

    from tools.g12_analysis import two_sided_mcnemar

    arms = ("ordinary", "grounded", "declining", *(("grounded_fused",) if with_index else ()))
    counts: dict[str, dict[str, int]] = {arm: {"correct": 0, "wrong": 0, "declined": 0} for arm in arms}
    correct: dict[str, dict[str, bool]] = {arm: {} for arm in counts}
    tokens = dict.fromkeys(counts, 0)
    for task in chosen:
        key = task["id"]
        ordinary, ground, sampled_record = (rows.get(s, key) for s in ("ordinary", "grounded", "grounded_sampled"))
        verdicts = {"ordinary": grade_one(ordinary["public_text"], task["truth"]),
                    "grounded": grade_one(ground["public_text"], task["truth"]),
                    "declining": (grade_one(ground["public_text"], task["truth"])
                                  if agree_two(ground["public_text"], sampled_record["public_text"]) else "declined")}
        tokens["ordinary"] += ordinary["generated_tokens"]
        tokens["grounded"] += ground["generated_tokens"]
        tokens["declining"] += ground["generated_tokens"] + sampled_record["generated_tokens"]
        if with_index:
            fused_record = rows.get("grounded_fused", key)
            verdicts["grounded_fused"] = grade_one(fused_record["public_text"], task["truth"])
            tokens["grounded_fused"] += fused_record["generated_tokens"]
        for arm, verdict in verdicts.items():
            counts[arm][verdict] += 1
            correct[arm][key] = verdict == "correct"
        rows.put("graded", key, {"id": key, **verdicts, "lookup_passages": len(rows.get("lookup", key)["passages"])})

    def pair(a: str, b: str) -> dict[str, Any]:
        first = sum(correct[a][k] and not correct[b][k] for k in correct[a])
        second = sum(correct[b][k] and not correct[a][k] for k in correct[a])
        return {f"{a}_only": first, f"{b}_only": second, "two_sided_p": two_sided_mcnemar(first, second)}

    n = len(chosen)
    report = {"requests": n, "counts": counts,
              "attempted_accuracy": {arm: round(c["correct"] / max(1, c["correct"] + c["wrong"]), 4)
                                     for arm, c in counts.items()},
              "tokens_per_request": {arm: round(t / max(1, n), 1) for arm, t in tokens.items()},
              "grounded_vs_ordinary": pair("grounded", "ordinary"),
              **({"fused_vs_grounded": pair("grounded_fused", "grounded")} if with_index else
                 {"fused": "no index of every paragraph"}),
              "with_passages": sum(bool(rows.get("lookup", t["id"])["passages"]) for t in chosen)}
    (output / "summary.json").write_text(json.dumps(report, indent=1, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
