#!/usr/bin/env python3
"""L09's second half: her kept procedures on against off over what people typed into her chat.

The plan is docs/evidence/L09_ORDINARY_USE_PREREGISTRATION_2026-10-18.md. Three steps, each writing into OUT:

    l09_ordinary_use.py population --out OUT      the requests: distinct user texts of her chat before the cutoff
    l09_ordinary_use.py route --out OUT           her chat's route over them with the frozen book (no model)
    l09_ordinary_use.py draft --out OUT --model DIR   her 27B's drafts for the routed requests (her app down)

The route is her chat's own functions (core/phases/kept_procedures_in_a_turn.py) with the frozen book put in
place of her live one. A request the meaning step would hand to her model to restate is recorded as a
candidate; the restatement itself is her model's and runs in the draft step.
"""
from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import re
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CONVERSATIONS = Path.home() / ".aura" / "data" / "conversations.db"
CUTOFF = dt.datetime(2026, 10, 18, 16, 30)
FIRST_HALF_PLAN = Path.home() / ".aura" / "rlc-evidence" / "learning-loop-20261016" / "plan.md"
_ANSWER_STATEMENT = re.compile(r"^(No, it's|Yes, the answer is) -?\d[\d,]*\.?$")


def first_half_questions() -> set[str]:
    """The teaching and check questions of L09's first half, read from its fixed plan."""
    text = FIRST_HALF_PLAN.read_text(encoding="utf-8")
    return {m.group(1).strip() for m in re.finditer(r"^\| *[0-9A-C]+ *\| *(How many days [^|]+?) *\|", text, re.M)}


def population() -> list[dict]:
    """Every distinct user text typed into her chat before the cutoff, first occurrence, less L09's first half."""
    excluded = first_half_questions()
    connection = sqlite3.connect(f"file:{CONVERSATIONS}?mode=ro", uri=True)
    rows = connection.execute(
        "select content, created_at from turns where role='user' and origin='desktop_ui' and created_at < ? "
        "order by created_at", (CUTOFF.timestamp(),)).fetchall()
    seen: dict[str, dict] = {}
    left_out = {"first_half_question": 0, "answer_statement": 0}
    for content, created in rows:
        text = str(content).strip()
        if text in seen:
            seen[text]["times"] += 1
            continue
        if text in excluded:
            left_out["first_half_question"] += 1
            continue
        if _ANSWER_STATEMENT.match(text):
            left_out["answer_statement"] += 1
            continue
        seen[text] = {"id": f"r{len(seen):03d}", "text": text, "first_typed": float(created), "times": 1}
    print(json.dumps({"turns": len(rows), "distinct_kept": len(seen), "left_out_turns": left_out}))
    return list(seen.values())


def _frozen(book_path: Path):
    from core.learning.procedures_from_solved_examples import ProcedureBook, kept_steps_from

    book = ProcedureBook.from_json(json.loads(book_path.read_text(encoding="utf-8")))
    book.steps = kept_steps_from(book_path, book)
    return book


async def route_one(text: str, book) -> dict:
    from core.brain.reasoning_amplifier_v2 import answer_from_kept_procedures
    from core.learning.kinds_by_meaning import TASK
    from core.phases import kept_procedures_in_a_turn as turn

    kept = await asyncio.to_thread(answer_from_kept_procedures, text, book)
    form = "words"
    if kept is None:
        kept = await turn.kept_answer_among_the_word_near(text)
        form = "words_then_meaning"
    if kept is not None:
        operation = dict(kept.receipt.cognitive_operations[0]) if kept.receipt.cognitive_operations else {}
        return {"form": form, "kind": str(operation.get("family") or ""), "answer": str(kept.answer),
                "agreed_known": int(operation.get("agreed_known") or 0)}
    vector = await asyncio.to_thread(turn.her_query_vector, text, TASK)
    if vector is None:
        raise SystemExit("her embedding model did not load; the meaning steps cannot be routed")
    kind, similarity, why = book.route_by_meaning(vector)
    if kind is not None and kind in book.procedures and book.examples.get(kind):
        return {"form": "restatement_candidate", "kind": kind, "similarity": round(float(similarity), 4)}
    return {"form": "none", "why": why}


def route(out: Path) -> None:
    import core.learning.procedures_from_solved_examples as procedures
    import core.phases.kept_procedures_in_a_turn as turn
    from core.memory.vector_memory_engine import EmbeddingEngine

    book = _frozen(out / "book.json")
    # Her chat's functions read her live book; this run reads the frozen one in its place.
    procedures.kept_procedure_book = lambda: book
    # Her chat embeds with the engine her runtime holds; offline the same engine is loaded here.
    engine = EmbeddingEngine()
    engine.embed_query("a probe", task="kind_of_problem")
    if getattr(engine, "_model", None) is None:
        raise SystemExit("her embedding model did not load; the meaning steps cannot be routed")
    turn.her_query_vector = lambda text, task: engine.embed_query(text, task=task)
    requests = json.loads((out / "requests.json").read_text())
    routed = []
    for request in requests:
        result = asyncio.run(route_one(request["text"], book))
        if result["form"] != "none":
            result["vector"] = [float(x) for x in turn.her_query_vector(request["text"], "kind_of_problem")]
        routed.append({**request, **result})
        if result["form"] != "none":
            print(json.dumps({"id": request["id"], **{k: v for k, v in result.items() if k != "vector"},
                              "text": request["text"][:80]}), flush=True)
    (out / "routed.json").write_text(json.dumps(routed, indent=1))
    counts: dict[str, int] = {}
    for row in routed:
        counts[row["form"]] = counts.get(row["form"], 0) + 1
    print(json.dumps({"requests": len(routed), "by_form": counts}))


def _patched(out: Path):
    """The frozen book and her embedding engine, put where her chat's route functions look for them."""
    import core.learning.procedures_from_solved_examples as procedures
    import core.phases.kept_procedures_in_a_turn as turn
    from core.memory.vector_memory_engine import EmbeddingEngine

    book = _frozen(out / "book.json")
    procedures.kept_procedure_book = lambda: book
    engine = EmbeddingEngine()
    engine.embed_query("a probe", task="kind_of_problem")
    if getattr(engine, "_model", None) is None:
        raise SystemExit("her embedding model did not load")
    turn.her_query_vector = lambda text, task: engine.embed_query(text, task=task)
    return book


def draft(out: Path, model_path: Path, max_tokens: int, exclusive: bool = True) -> None:
    """Off: her 27B's answer to each routed request. On: the kept answer by words, or by restatement her draft
    with the kept answer before her as evidence, the restatement her own model's, through her chat's function."""
    import mlx.core as mx
    from mlx_lm import load, stream_generate

    import core.learning.procedures_from_solved_examples as procedures
    import core.phases.kept_procedures_in_a_turn as turn
    from core.brain.llm.chat_format import for_this_template, render_chat_continuation_template
    from core.phases.kept_procedures_in_a_turn import kept_answer_by_meaning, kept_answer_evidence
    from core.runtime.model_lane_control import standalone_model_lane
    from core.utils.injected_blocks import RUNTIME_EVIDENCE_ROLE
    from tools.run_g05_public_answers import decode_public

    routed = [r for r in json.loads((out / "routed.json").read_text()) if r["form"] != "none"]
    book = _frozen(out / "book.json")
    procedures.kept_procedure_book = lambda: book
    # The embeddings the route step made; no embedder is loaded beside the 27B's exclusive lane.
    vectors = {r["text"]: r["vector"] for r in routed}
    turn.her_query_vector = lambda text, task: __import__("numpy").asarray(vectors[text], dtype="float32")
    rows = out / "rows"
    rows.mkdir(exist_ok=True)
    with standalone_model_lane(owner_id="l09-ordinary-use", model_path=str(model_path), purpose="evaluation",
                               preemptible=False, require_exclusive=exclusive, allow_owner_eviction=exclusive,
                               metadata={"tool": Path(__file__).name}):
        model, tokenizer = load(str(model_path))

        class Router:
            """Her chat's restatement call, decoded on the loaded model as her worker decodes a continuation."""

            async def think(self, *, messages, stop_sequences=(), max_tokens=256, **_ignored):
                prompt = render_chat_continuation_template(tokenizer, for_this_template(tokenizer, messages),
                                                           enable_thinking=False)
                ids = [int(t) for t in tokenizer.encode(prompt, add_special_tokens=False)]
                text = ""
                for response in stream_generate(model, tokenizer, ids, max_tokens=max_tokens,
                                                sampler=lambda logits: mx.argmax(logits, axis=-1)):
                    text += str(response.text or "")
                    if any(stop in text for stop in stop_sequences):
                        break
                for stop in stop_sequences:
                    text = text.split(stop)[0]
                return {"content": str(messages[-1]["content"]) + text}

        for row in routed:
            path = rows / f"{row['id']}.json"
            if path.exists():
                continue
            off = decode_public(model, tokenizer, [{"role": "user", "content": row["text"]}], max_tokens=max_tokens,
                                thinking=True)
            record = {**row, "off": off}
            if row["form"] in ("words", "words_then_meaning"):
                record["on"] = {"served": "kept_answer", "answer": row["answer"]}
            else:
                kept = asyncio.run(kept_answer_by_meaning(row["text"], router=Router(), origin="l09_ordinary_use",
                                                          tier="primary"))
                if kept is None:
                    # The route did not answer, so her chat's path is the off arm's.
                    record["on"] = {"served": "her_reply_route_declined", "same_as_off": True}
                else:
                    operation = dict(kept.receipt.cognitive_operations[0]) if kept.receipt.cognitive_operations else {}
                    planted = {"answer": str(kept.answer), "summary": (
                        f"computed by the procedure she keeps for problems of the kind "
                        f"{operation.get('family')!r}, which agreed with all {operation.get('agreed_known')} answers "
                        f"she knows for that kind; the request was first written in that kind's form, as: "
                        f"{operation.get('restated', '')}")}
                    evidence = ("[GROUNDING EVIDENCE FOR THIS TURN]\n" + kept_answer_evidence(planted)
                                + "\n[END GROUNDING EVIDENCE FOR THIS TURN]")
                    messages = for_this_template(tokenizer, [{"role": RUNTIME_EVIDENCE_ROLE, "content": evidence},
                                                             {"role": "user", "content": row["text"]}])
                    on = decode_public(model, tokenizer, messages, max_tokens=max_tokens, thinking=True)
                    record["on"] = {"served": "her_reply_with_evidence", "kept_answer": str(kept.answer),
                                    "restated": operation.get("restated", ""), "draft": on}
            path.write_text(json.dumps(record, indent=1, default=str))
            print(json.dumps({"id": row["id"], "form": row["form"], "on": record["on"].get("served")}), flush=True)


def _final_region(text: str) -> str:
    """The answer a reply closes on: its last \\boxed{...}, or else its last paragraph."""
    text = str(text or "")
    boxed = re.findall(r"\\boxed\{([^{}]*)\}", text)
    if boxed:
        return boxed[-1]
    paragraphs = [part for part in re.split(r"\n\s*\n", text) if part.strip()]
    return paragraphs[-1] if paragraphs else ""


def _integers(text: str) -> set[int]:
    joined = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)
    return {int(n) for n in re.findall(r"(?<![\d.])\d+(?![\d.]\d)", joined)}


def states(text: str, fixed: dict, question: str) -> bool:
    """Whether a reply's closing answer is the fixed answer: one rule for both arms, by the answer's type.

    G09's rule for DeepMind kinds reads the benchmark's boxed form; on her prose it graded "17 times 23 is
    391." wrong, which would have favoured the kept arm, whose answer is the bare value.
    """
    region = _final_region(text)
    lowered = region.casefold()
    kind, answer = fixed["kind"], str(fixed["answer"])
    if kind == "number":
        return int(answer) in _integers(region)
    if kind == "number_list":
        import sympy

        want = {int(x) for x in re.findall(r"\d+", answer)}
        found = _integers(region)
        extra = found - want - _integers(question)
        return want <= found and not any(sympy.isprime(n) for n in extra)
    if kind == "yes_no":
        said = ("no" if re.search(r"\bnot\b|\bisn'?t\b|\bno\b|\bcomposite\b", lowered)
                else "yes" if re.search(r"\byes\b|\bis (?:a )?prime\b", lowered) else "")
        return said == answer
    if kind == "word_list":
        want = answer.split()
        order = [w for w in re.findall(r"[a-z]+", lowered) if w in want]
        return list(dict.fromkeys(order)) == want
    raise KeyError(kind)


def grade(out: Path) -> dict:
    """Both arms by one rule (``states``), against the answers fixed before any draft."""
    import math as _math

    truths = json.loads((out / "truths.json").read_text())
    pairs = []
    for path in sorted((out / "rows").glob("*.json")):
        row = json.loads(path.read_text())
        fixed = truths[row["id"]]
        off_text = row["off"]["public_text"] if row["off"].get("termination") == "stop" else ""
        on = row["on"]
        if on.get("served") == "kept_answer":
            on_text = on["answer"]
        elif on.get("same_as_off"):
            on_text = off_text
        else:
            on_text = on["draft"]["public_text"] if on["draft"].get("termination") == "stop" else ""
        pairs.append({"id": row["id"], "kind": row["kind"], "text": row["text"], "served": on.get("served"),
                      "off": states(off_text, fixed, row["text"]), "on": states(on_text, fixed, row["text"]),
                      "off_closes_on": _final_region(off_text)[-300:], "on_closes_on": _final_region(on_text)[-300:]})
    gained = sum(1 for p in pairs if p["on"] and not p["off"])
    lost = sum(1 for p in pairs if p["off"] and not p["on"])
    n = gained + lost

    def tail(k: int) -> float:
        return sum(_math.comb(n, i) for i in range(k, n + 1)) / 2 ** n if n else 1.0

    summary = {"routed": len(pairs), "gained": gained, "lost": lost,
               "both_right": sum(1 for p in pairs if p["on"] and p["off"]),
               "both_wrong": sum(1 for p in pairs if not p["on"] and not p["off"]),
               "p_gain": tail(gained), "p_harm": tail(lost),
               "underpowered": n < 5, "pairs": pairs}
    (out / "graded.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "pairs"}))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=("population", "route", "draft", "grade"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", type=Path, help="her persona 27B, for the draft step")
    parser.add_argument("--max-tokens", type=int, default=32768, help="as G09's ordinary arm")
    parser.add_argument("--not-exclusive", action="store_true",
                        help="share the lane: only for a dry run with a small model beside her app")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.step == "population":
        (args.out / "requests.json").write_text(json.dumps(population(), indent=1))
    elif args.step == "route":
        route(args.out)
    elif args.step == "grade":
        grade(args.out)
    else:
        if args.model is None:
            parser.error("the draft step needs --model")
        draft(args.out, args.model.expanduser().resolve(strict=True), args.max_tokens, exclusive=not args.not_exclusive)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
