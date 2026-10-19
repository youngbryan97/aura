#!/usr/bin/env python3
"""General reasoning, mechanism A, against her ordinary answer and against self-consistency at equal decoded tokens.

Mechanism A is core/reasoning/where_derivations_part.py: two derivations; where
they part, the crux named and settled by the strongest check that applies;
her final derivation with the settled claim in front of her. Here each stage
is decoded for every question that reaches it, in fixed groups
(tools/g12_batched.py), and every record is written once under
``rows/<stage>/``, so a run started again resumes where it stopped.

1. greedy: her ordinary answer, the G12 protocol (one user turn, private
   reasoning on at her serving effort, greedy, up to ``--max-tokens``).
2. sampled: the same request sampled at her runtime's ``llm.deep_temperature``
   (core/config.py) and top-p 0.9, her router's default.
3. crux: for each question whose two answers differ, the crux request,
   private channel closed, greedy, at most as long as the shorter reply.
4. program: for each crux whose sides state different numbers, her program
   for the settling question (core/brain/executable_reasoning.py's request),
   run in the symbolic sandbox, its printed value matched to a side.
5. formal: for each crux still open, the whole question written as constraints
   for her engine (core/reasoning/written_as_constraints.py), run in the
   sandbox, with one repair round; a unique solution that is one
   derivation's answer settles the point for that side.
6. reading: for each crux still open, the paragraphs of her corpus about what
   the settling question names (core/knowledge/passages_about.py; her
   conversational lookup when that finds nothing), read for the side they
   support.
7. alone: for each crux still open, the settling question derived greedy and
   sampled on its own; each reply read for its side; both must agree.
8. final: the whole question again, greedy, with the settled claim given to
   her as a skill-result block, the way her runtime gives evidence.
9. consistency: further sampled derivations until the run's decoded tokens
   per question match mechanism A's; the most common answer wins, ties to
   the greedy answer.

Benchmarks: ``mmlu_pro`` (TIGER-Lab's zero-shot chain-of-thought request),
``musr`` (the MuSR authors' request), ``bbeh_soft`` (the BBEH tasks no
procedure was kept for, with the paper's instruction). Questions are split
into halves by a hash of their id; ``--part`` names the half.

Usage:
    run_general_reasoning.py --benchmark NAME --part development --count N --seed S --output DIR
        [--batch 8] [--max-tokens 32768] [--model DIR] [--through STAGE]
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

GR = Path("~/.aura/benchmarks/gr").expanduser()
STAGES = ("greedy", "sampled", "crux", "program", "formal", "reading", "alone", "final", "consistency")
#: The BBEH tasks no kept procedure answers and none was written for (G09 attempts 1 and 2).
BBEH_SOFT = ("bbeh_boardgame_qa", "bbeh_causal_understanding", "bbeh_disambiguation_qa", "bbeh_hyperbaton",
             "bbeh_linguini", "bbeh_movie_recommendation", "bbeh_nycc", "bbeh_sarc_triples", "bbeh_spatial_reasoning",
             "bbeh_sportqa", "bbeh_zebra_puzzles")
#: TIGER-Lab's MMLU-Pro chain-of-thought instruction (evaluate_from_local.py).
MMLU_PRO_HEAD = ("The following are multiple choice questions (with answers) about {subject}. Think step by step and "
                 "then finish your answer with \"the answer is (X)\" where X is the correct letter choice.")
#: The MuSR authors' answer instruction (Sprague et al. 2024, eval prompts).
MUSR_TAIL = ("You must pick one option. Explain your reasoning step by step before you answer. Finally, the last "
             "thing you generate should be \"ANSWER: (your answer here, including the choice number)\"")
#: Her router's top-p when a caller names none (core/brain/llm/llm_router.py).
TOP_P = 0.9


def _half(rows: list[dict[str, Any]], part: str) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda r: hashlib.sha256(r["id"].encode()).hexdigest())
    return ordered[: len(ordered) // 2] if part == "development" else ordered[len(ordered) // 2:]


def mmlu_pro_tasks(part: str) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    rows = []
    for row in pq.read_table(GR / "mmlu-pro/data/test-00000-of-00001.parquet").to_pylist():
        letters = "ABCDEFGHIJ"[: len(row["options"])]
        options = "\n".join(f"{letter}. {option}" for letter, option in zip(letters, row["options"], strict=True))
        request = (f"{MMLU_PRO_HEAD.format(subject=row['category'])}\n\nQuestion:\n{row['question']}\n"
                   f"Options:\n{options}\nAnswer: Let's think step by step.")
        rows.append({"id": f"mmlu_pro:{row['question_id']}", "request": request, "truth": row["answer"],
                     "group": row["category"]})
    return _half(rows, part)


def musr_tasks(part: str) -> list[dict[str, Any]]:
    csv.field_size_limit(sys.maxsize)
    rows = []
    for name in ("murder_mystery", "object_placements", "team_allocation"):
        with (GR / f"musr/{name}.csv").open(encoding="utf-8") as handle:
            for index, row in enumerate(csv.DictReader(handle)):
                choices = ast.literal_eval(row["choices"])
                listed = "\n".join(f"{i + 1} - {choice}" for i, choice in enumerate(choices))
                request = (f"{row['narrative'].strip()}\n\n{row['question'].strip()}\n\n"
                           f"Pick one of the following choices:\n{listed}\n\n{MUSR_TAIL}")
                rows.append({"id": f"musr:{name}:{index}", "request": request,
                             "truth": str(int(row["answer_index"]) + 1), "group": name})
    return _half(rows, part)


def bbeh_soft_tasks(part: str) -> list[dict[str, Any]]:
    from tools.induce_g09_procedures import problems

    # BBEH's own split: BBEH Mini is the test half, the other 180 per task development.
    return [{"id": r["key"], "request": r["problem"], "truth": r["truth"], "group": kind}
            for kind in BBEH_SOFT for r in problems(kind, "test" if part == "test" else "development")]


LOADERS = {"mmlu_pro": mmlu_pro_tasks, "musr": musr_tasks, "bbeh_soft": bbeh_soft_tasks}


def answer_of(benchmark: str, text: str) -> str:
    """The answer a reply gives, as its benchmark reads it; "" when it gives none."""
    text = str(text or "")
    if benchmark == "mmlu_pro":
        found = re.findall(r"answer is \(?([A-J])\)?", text) or re.findall(r"[aA]nswer:\s*\(?([A-J])\)?", text)
        return found[-1] if found else ""
    if benchmark == "musr":
        tail = text[text.rfind("ANSWER:"):] if "ANSWER:" in text else ""
        number = re.search(r"\d+", tail)
        return number.group(0) if number else ""
    from tools.run_g09_organ import _bbeh_answer

    return " ".join(_bbeh_answer(text).split()).casefold()


def answer_line_of(benchmark: str, line: str) -> str:
    """The answer an engine's ANSWER line gives, in the benchmark's answer form."""
    line = str(line or "").strip()
    if benchmark == "mmlu_pro":
        found = re.findall(r"\(?\b([A-J])\b\)?", line)
        return found[0] if len(found) == 1 else ""
    if benchmark == "musr":
        number = re.search(r"\d+", line)
        return number.group(0) if number else ""
    return " ".join(line.split()).casefold()


def correct(benchmark: str, text: str, truth: str) -> bool:
    if benchmark == "bbeh_soft":
        from tools.run_g09_organ import grade_bbeh

        return bool(grade_bbeh(text, truth)[0])
    return answer_of(benchmark, text) == str(truth)


def _same(a: str, b: str) -> bool:
    return bool(a) and " ".join(a.split()).casefold() == " ".join(b.split()).casefold()


def _safe(task_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", task_id)


class Rows:
    """Records written once per stage and task id."""

    def __init__(self, output: Path) -> None:
        self.output = output

    def path(self, stage: str, task_id: str) -> Path:
        return self.output / "rows" / stage / f"{_safe(task_id)}.json"

    def get(self, stage: str, task_id: str) -> dict[str, Any] | None:
        path = self.path(stage, task_id)
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    def put(self, stage: str, task_id: str, record: dict[str, Any]) -> None:
        path = self.path(stage, task_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(record, indent=1, default=str), encoding="utf-8")
        temporary.replace(path)


def evidence_block(crux_text: str) -> str:
    """The settled claim as her runtime renders a skill result (core/phases/response_generation.py)."""
    from core.phases.response_generation import ResponseGenerationPhase

    return ResponseGenerationPhase._render_skill_result_block(
        skill_name="checked_point", payload={"ok": True, "summary": crux_text})


def summarize(benchmark: str, tasks: list[dict[str, Any]], rows: Rows) -> dict[str, Any]:
    from tools.g12_analysis import two_sided_mcnemar, wilson

    arms: dict[str, dict[str, bool]] = {"ordinary": {}, "mechanism_a": {}, "consistency": {}}
    tokens: dict[str, int] = Counter()
    stages: Counter = Counter()
    for task in tasks:
        greedy, mechanism, consistency = (rows.get(s, task["id"]) for s in ("greedy", "mechanism", "consistency"))
        if not (greedy and mechanism and consistency):
            continue
        arms["ordinary"][task["id"]] = correct(benchmark, greedy["public_text"], task["truth"])
        arms["mechanism_a"][task["id"]] = mechanism["answer_text_correct"]
        arms["consistency"][task["id"]] = consistency["correct"]
        tokens["ordinary"] += greedy["generated_tokens"]
        tokens["mechanism_a"] += mechanism["tokens"]
        tokens["consistency"] += consistency["tokens"]
        stages[mechanism["outcome"]] += 1

    def pair(a: str, b: str) -> dict[str, Any]:
        keys = sorted(set(arms[a]) & set(arms[b]))
        first = sum(arms[a][k] and not arms[b][k] for k in keys)
        second = sum(arms[b][k] and not arms[a][k] for k in keys)
        return {"requests": len(keys), f"{a}_only": first, f"{b}_only": second,
                "two_sided_p": two_sided_mcnemar(first, second)}

    n = len(arms["ordinary"])
    return {"benchmark": benchmark, "requests": n,
            "accuracy": {arm: {"correct": sum(v.values()), "interval": wilson(sum(v.values()), n)}
                         for arm, v in arms.items()},
            "tokens_per_request": {arm: round(t / n, 1) if n else 0 for arm, t in tokens.items()},
            "mechanism_a_vs_ordinary": pair("mechanism_a", "ordinary"),
            "consistency_vs_ordinary": pair("consistency", "ordinary"),
            "mechanism_a_vs_consistency": pair("mechanism_a", "consistency"),
            "mechanism_a_outcomes": dict(stages)}


def main() -> int:  # noqa: C901 - one stage after another, each short
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benchmark", choices=sorted(LOADERS), required=True)
    parser.add_argument("--part", choices=("development", "test"), required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--model", type=Path,
                        default=Path("~/.aura/models/Aura-Qwen3.8-27B-persona-crsm-7f6a2e83f73f5eef9d15"))
    parser.add_argument("--through", choices=STAGES, default="consistency", help="stop after this stage")
    parser.add_argument("--adapter", type=Path, help="an adapter over her model, the only difference (L01's no-harm arm)")
    args = parser.parse_args()

    from core.config import config
    from core.reasoning.where_derivations_part import (
        Crux,
        crux_request,
        evidence_text,
        kind_of,
        parse_crux,
        parse_reading,
        reading_request,
        side_of_value,
    )
    from tools.run_g09_organ import sample

    output = args.output.expanduser()
    rows = Rows(output)
    tasks = sample(LOADERS[args.benchmark](args.part), args.count, args.seed, set())
    (output).mkdir(parents=True, exist_ok=True)
    (output / "task_ids.txt").write_text("".join(f"{t['id']}\n" for t in tasks), encoding="utf-8")
    by_id = {t["id"]: t for t in tasks}
    last = STAGES.index(args.through)
    print(json.dumps({"benchmark": args.benchmark, "part": args.part, "tasks": len(tasks)}), flush=True)

    import mlx.core as mx  # noqa: F401 - loaded before the lane is held
    from mlx_lm import load
    from mlx_lm.sample_utils import make_sampler

    from core.runtime.model_lane_control import standalone_model_lane
    from tools.g12_batched import decode_stream

    model_path = args.model.expanduser().resolve(strict=True)
    with standalone_model_lane(owner_id=f"general-reasoning:{output.name}", model_path=str(model_path),
                               purpose="evaluation", preemptible=False, require_exclusive=True):
        model, tokenizer = load(str(model_path), adapter_path=str(args.adapter.expanduser()) if args.adapter else None)
        sampler = make_sampler(temp=float(config.llm.deep_temperature), top_p=TOP_P)

        def run(stage: str, items: list[tuple[str, list[dict[str, str]]]], *, thinking: bool = True,
                sampled: bool = False, seed: int = 0, max_tokens: int | dict[str, int] | None = None) -> None:
            todo = [(key, conversation) for key, conversation in items if rows.get(stage, key) is None]
            print(json.dumps({"stage": stage, "to_decode": len(todo), "written": len(items) - len(todo)}), flush=True)
            if not todo:
                return

            def write(index: int, record: dict[str, Any]) -> None:
                rows.put(stage, todo[index][0], {"id": todo[index][0], **record})

            bound = ([max_tokens[key] for key, _ in todo] if isinstance(max_tokens, dict)
                     else max_tokens or args.max_tokens)
            decode_stream(model, tokenizer, [c for _, c in todo], max_tokens=bound,
                          width=args.batch, on_record=write, thinking=thinking,
                          sampler=sampler if sampled else None, seed=args.seed + seed if sampled else None)

        def tokens_of(text: str) -> int:
            return len(tokenizer.encode(text, add_special_tokens=False))

        user = lambda text: [{"role": "user", "content": text}]  # noqa: E731
        run("greedy", [(t["id"], user(t["request"])) for t in tasks])
        if last < STAGES.index("sampled"):
            return 0  # greedy alone: its rows are what was asked for, and there is nothing to set them against
        run("sampled", [(t["id"], user(t["request"])) for t in tasks], sampled=True, seed=1)
        parted = [t for t in tasks if not _same(answer_of(args.benchmark, rows.get("greedy", t["id"])["public_text"]),
                                                answer_of(args.benchmark, rows.get("sampled", t["id"])["public_text"]))]
        print(json.dumps({"agreed": len(tasks) - len(parted), "parted": len(parted)}), flush=True)

        def replies(task_id: str) -> tuple[str, str]:
            return rows.get("greedy", task_id)["public_text"], rows.get("sampled", task_id)["public_text"]

        if last >= STAGES.index("crux") and parted:
            # A crux restates part of what both replies said, so it is no longer than the shorter.
            limits = {t["id"]: max(1, min(tokens_of(r) for r in replies(t["id"]))) for t in parted}
            run("crux", [(t["id"], user(crux_request(t["request"], *replies(t["id"])))) for t in parted],
                thinking=False, max_tokens=limits)
        cruxes: dict[str, Crux] = {}
        for task in parted:
            record = rows.get("crux", task["id"])
            crux = parse_crux(record["public_text"]) if record else None
            if crux is not None:
                crux.kind = kind_of(crux)
                cruxes[task["id"]] = crux

        if last >= STAGES.index("program"):
            from core.brain.executable_reasoning import EXECUTABLE_STRATEGIES, _program_prompt
            from core.brain.symbolic_sandbox import SymbolicSandbox
            from core.brain.verifiers.code_engine import extract_code_blocks

            quantity = [k for k, c in cruxes.items() if c.kind == "quantity"]
            run("program", [(k, user(_program_prompt(cruxes[k].question, "", strategy=EXECUTABLE_STRATEGIES[0],
                                                      excluded_program_sha256s=(), excluded_candidate_sha256s=(),
                                                      prior_failure_class="")))
                            for k in quantity])
            sandbox = SymbolicSandbox()
            for key in quantity:
                record = rows.get("program", key)
                if record is None or "value" in record:
                    continue
                blocks = extract_code_blocks(record["public_text"])
                result = asyncio.run(sandbox.run(blocks[0])) if len(blocks) == 1 else None
                record["value"] = result.stdout.strip() if result is not None and result.ok else ""
                rows.put("program", key, record)
            for key in quantity:
                value = (rows.get("program", key) or {}).get("value", "")
                side = side_of_value(value, cruxes[key])
                if side:
                    crux = cruxes[key]
                    crux.settled, crux.how, crux.detail = side, "computed in the sandbox", f"The computed value: {value}"

        if last >= STAGES.index("formal"):
            from core.learning.procedures_from_solved_examples import (
                _mechanisms_directory,
                _mechanisms_prelude,
            )
            from core.reasoning.written_as_constraints import (
                Outcome,
                formalization_request,
                parse_outcome,
                repair_request,
            )

            def executed(stage: str, key: str) -> dict[str, Any]:
                record = rows.get(stage, key)
                if record is not None and "outcome" not in record:
                    blocks = extract_code_blocks(record["public_text"])
                    if len(blocks) == 1:
                        ran = asyncio.run(sandbox.run(blocks[0], prelude=_mechanisms_prelude(),
                                                      extra_read_paths=(_mechanisms_directory(),)))
                        outcome = parse_outcome(record["public_text"], ran.stdout,
                                                "" if ran.ok else (ran.stderr or ran.traceback or "the run failed"))
                    else:
                        # A declining reply is read as declining before any error is.
                        outcome = parse_outcome(record["public_text"], "", f"the reply has {len(blocks)} code blocks")
                    record["outcome"] = outcome.__dict__
                    record["program"] = blocks[0] if len(blocks) == 1 else ""
                    rows.put(stage, key, record)
                return record or {}

            open_keys = [k for k, c in cruxes.items() if not c.settled]
            run("formal", [(k, user(formalization_request(by_id[k]["request"]))) for k in open_keys])
            again = []
            for key in open_keys:
                first = executed("formal", key)
                outcome = first.get("outcome") or {}
                if not outcome.get("declined") and not (outcome.get("solutions") == 1 and outcome.get("answer")):
                    again.append((key, user(repair_request(by_id[key]["request"], first.get("program", ""),
                                                           Outcome(**outcome)))))
            run("formal_repair", again)
            for key in open_keys:
                last_record = executed("formal_repair", key) if rows.get("formal_repair", key) else \
                    executed("formal", key)
                outcome = last_record.get("outcome") or {}
                if outcome.get("solutions") != 1 or not outcome.get("answer"):
                    continue
                solved = answer_line_of(args.benchmark, outcome["answer"])
                first, second = replies(key)
                side = ("first" if solved and _same(solved, answer_of(args.benchmark, first)) else
                        "second" if solved and _same(solved, answer_of(args.benchmark, second)) else None)
                if side:
                    crux = cruxes[key]
                    crux.settled, crux.how = side, "solved as constraints by her engine"
                    crux.detail = (f"Written as constraints, the problem has exactly one solution, whose answer is "
                                   f"{outcome['answer']}.")

        if last >= STAGES.index("reading"):
            from core.knowledge.corpus_grounding import corpus_grounding_for
            from core.knowledge.passages_about import passages_about

            open_keys = [k for k, c in cruxes.items() if not c.settled]
            asked = []
            for key in open_keys:
                found = rows.get("lookup", key)
                if found is None:
                    question = cruxes[key].question
                    pairs = passages_about(question) or list(corpus_grounding_for(question).passages)
                    found = {"passages": [f"{title}: {text}" for title, text in pairs]}
                    rows.put("lookup", key, found)
                if found["passages"]:
                    asked.append((key, user(reading_request(found["passages"], cruxes[key]))))
            limit = tokens_of("Reply with one word: FIRST, SECOND or NEITHER.")
            run("reading", asked, thinking=False, max_tokens=limit)
            for key, _ in asked:
                side = parse_reading(rows.get("reading", key)["public_text"])
                if side:
                    crux = cruxes[key]
                    crux.settled, crux.how = side, "found in her corpus"
                    crux.detail = "From her corpus: " + " … ".join(rows.get("lookup", key)["passages"])

        if last >= STAGES.index("alone"):
            open_keys = [k for k, c in cruxes.items() if not c.settled]
            run("alone_greedy", [(k, user(cruxes[k].question)) for k in open_keys])
            run("alone_sampled", [(k, user(cruxes[k].question)) for k in open_keys], sampled=True, seed=2)
            limit = tokens_of("Reply with one word: FIRST, SECOND or NEITHER.")
            for which in ("alone_greedy", "alone_sampled"):
                run(f"{which}_reading", [(k, user(reading_request([rows.get(which, k)["public_text"]], cruxes[k])))
                                         for k in open_keys], thinking=False, max_tokens=limit)
            for key in open_keys:
                sides = [parse_reading(rows.get(f"{w}_reading", key)["public_text"])
                         for w in ("alone_greedy", "alone_sampled")]
                if sides[0] and sides[0] == sides[1]:
                    crux = cruxes[key]
                    crux.settled, crux.how = sides[0], "derived again on the point alone, twice in agreement"

        settled = {k: c for k, c in cruxes.items() if c.settled}
        if last >= STAGES.index("final"):
            run("final", [(k, [{"role": "system", "content": evidence_block(evidence_text(c))},
                               {"role": "user", "content": by_id[k]["request"]}]) for k, c in settled.items()])
            for task in tasks:
                key = task["id"]
                greedy, sampled_record = rows.get("greedy", key), rows.get("sampled", key)
                spent = greedy["generated_tokens"] + sampled_record["generated_tokens"]
                for stage in ("crux", "program", "formal", "formal_repair", "reading", "alone_greedy", "alone_sampled",
                              "alone_greedy_reading", "alone_sampled_reading", "final"):
                    spent += (rows.get(stage, key) or {}).get("generated_tokens", 0)
                final = rows.get("final", key) if key in settled else None
                text = final["public_text"] if final and answer_of(args.benchmark, final["public_text"]) \
                    else greedy["public_text"]
                crux = cruxes.get(key)
                outcome = ("agreed" if task not in parted else "no_crux" if crux is None
                           else f"settled:{crux.how}" if crux.settled else "unsettled")
                rows.put("mechanism", key, {"id": key, "answer": answer_of(args.benchmark, text),
                                            "answer_text_correct": correct(args.benchmark, text, task["truth"]),
                                            "tokens": spent, "outcome": outcome,
                                            "crux": None if crux is None else crux.__dict__})

        if last >= STAGES.index("consistency"):
            ordinary = sum(rows.get("greedy", t["id"])["generated_tokens"] for t in tasks)
            spent = sum(rows.get("mechanism", t["id"])["tokens"] for t in tasks)
            # Derivations in all, the greedy and the first sampled among them, to spend what mechanism A spent.
            total = max(2, round(spent / max(1, ordinary)))
            for extra in range(2, total):
                run(f"consistency_{extra}", [(t["id"], user(t["request"])) for t in tasks], sampled=True,
                    seed=extra + 1)
            for task in tasks:
                key = task["id"]
                records = [rows.get("greedy", key), rows.get("sampled", key),
                           *(rows.get(f"consistency_{extra}", key) for extra in range(2, total))]
                answers = [answer_of(args.benchmark, r["public_text"]) for r in records]
                counts = Counter(a for a in answers if a)
                best = max(counts.values()) if counts else 0
                winners = [a for a, n in counts.items() if n == best]
                chosen = answers[0] if not winners or answers[0] in winners else winners[0]
                text = next((r["public_text"] for r, a in zip(records, answers, strict=True) if a == chosen),
                            records[0]["public_text"])
                rows.put("consistency", key, {"id": key, "answer": chosen, "derivations": total,
                                              "correct": correct(args.benchmark, text, task["truth"]),
                                              "tokens": sum(r["generated_tokens"] for r in records)})
            report = summarize(args.benchmark, tasks, rows)
            (output / "summary.json").write_text(json.dumps(report, indent=1, sort_keys=True), encoding="utf-8")
            print(json.dumps(report, indent=1, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
