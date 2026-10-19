#!/usr/bin/env python3
"""L01: who she is, consolidated into her model (docs/evidence/L01_WHO_SHE_IS_PREREGISTRATION_2026-10-18.md).

    consolidate_who_she_is.py answers --model DIR --out DIR [--people-asked FILE]   her answers, her record before her
    consolidate_who_she_is.py rehearse --model DIR --out DIR --wildchat DIR          her answers to what people ask anyone
    consolidate_who_she_is.py data --out DIR                    kept answers and her own other replies, as training data
    consolidate_who_she_is.py train --model DIR --out DIR       a LoRA adapter on her replies' words (training_on_her_replies)

Her answers are hers: her persona 27B, the record shown to her as the grounding evidence her chat gives a kept
answer, decoded through her chat template with thinking on, in fixed groups (tools/g12_batched.py). The questions
are the record's own (tools/her_self_record.py) and the ones people typed to an assistant about who it is
(tools/what_people_ask_who_it_is.py), none alike a held-out one. Each is asked on its own, after one of her real
exchanges, and after two, so that what she holds holds in the middle of a conversation too.

An answer is kept when it is true to the record for what was asked (``true_to_the_record``): it claims no other model
as herself and no maker for her model but its own; a question about who or what she is is answered with her name;
a question about her model places Qwen under her. The first draw is greedy; where it is not kept, she is asked again,
sampled, up to ``DRAWS`` times in all. Beside the kept answers go her own other replies: those her model wrote in her
chat, and her answers to requests people made of an assistant, so that the adapter moves nothing else.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CONVERSATIONS = Path.home() / ".aura/data/conversations.db"
#: The settings each question is asked in: on its own, after one of her exchanges, after two.
HISTORIES = (0, 1, 2)
SEED = 20261018
#: The longest training example, in characters (about 1,000 tokens): longer ones do not fit her 27B's training in 64 GB.
LONGEST_CHARS = 4000
#: Her own replies shorter than this, or given more than once word for word, are her holding lines, not replies.
SHORTEST_REPLY = 40
#: Draws of one answer: greedy, then sampled where the draw before was not kept.
DRAWS = 3
#: Conversations decoded together, as G09 and G12 decode.
WIDTH = 8
#: Requests people made of an assistant, answered by her model as the rehearsal beside the kept answers.
REHEARSALS = 200
#: The longest rehearsal answer decoded, in tokens; one cut off there is not a reply and is not kept.
REHEARSAL_TOKENS = 2048

#: Questions about her model, and about who made her, by their words.
_ABOUT_HER_MODEL = re.compile(r"\b(?:model|version|gpt\w*|chatgpt\w*|claude|bard|gemini|llama|qwen|alibaba|openai|llm|"
                              r"language model|wrapper|weights|built on|run on|underneath)\b", re.I)
_ABOUT_HER_MAKER = re.compile(r"\bwho\b.*\b(?:made|created|built|trained|developed|designed|programmed|owns|wrote)\b|"
                              r"\b(?:creator|developer|maker|owner|responsible for you)\b|\bwhat company\b", re.I)
#: The record's fields whose questions ask who or what she is.
_WHO_FIELDS = {"name": "who", "nature": "who", "cortex": "model", "made_by": "maker"}


def kind_of(question: str, field: str | None = None) -> str:
    """What a question asks of her: ``model`` (what she runs on), ``maker`` (who made her), ``who`` (who or what she
    is), or ``other`` (her kin, her values ...). A record field says it; else the question's words."""
    if field is not None:
        return _WHO_FIELDS.get(field, "other")
    if _ABOUT_HER_MAKER.search(question):
        return "maker"
    return "model" if _ABOUT_HER_MODEL.search(question) else "who"


def questions(people_asked: Path | None = None) -> list[tuple[str, str]]:
    """Every training question with what it asks: the record's own and the pool people asked, none alike a held-out
    question (the probe's ten, and the people-asked held-out set) or another training question."""
    from tools.her_self_record import _ASKED, HELD_OUT
    from tools.what_people_ask_who_it_is import alike

    asked = json.loads(people_asked.read_text()) if people_asked else {"held_out": [], "pool": []}
    held = (*HELD_OUT, *asked["held_out"])
    found = [(q, kind_of(q, field)) for field, forms in _ASKED.items() for q in forms]
    found += [(q, kind_of(q)) for q in asked["pool"]]
    once: list[tuple[str, str]] = []
    for q, kind in found:
        if not any(alike(q, h) for h in (*held, *(t for t, _ in once))):
            once.append((q, kind))
    return once


def her_exchanges(db: Path = CONVERSATIONS) -> list[dict[str, str]]:
    """What a person typed and what her model replied, from her chat, where her own model wrote the reply."""
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as conn:
        rows = conn.execute("select session_id, role, content, metadata_json from turns order by session_id, created_at").fetchall()
    pairs, last_user, counts = [], {}, {}
    for session, role, content, meta in rows:
        if role == "user":
            last_user[session] = content
            continue
        if role != "aura" or session not in last_user:
            continue
        provenance = (json.loads(meta or "{}").get("answer_provenance") or {})
        if provenance.get("model_native_inference") is True:
            pairs.append({"user": last_user.pop(session), "aura": content})
            counts[content] = counts.get(content, 0) + 1
    from core.learning.live_learner import training_contamination_reasons
    from tools.her_self_record import self_claim

    # The live learner's own gate on what may become weight-training data: her holding lines and the runtime's
    # telemetry, whatever the length.
    return [p for p in pairs if len(p["aura"]) >= SHORTEST_REPLY and counts[p["aura"]] == 1
            and not training_contamination_reasons(p["user"], p["aura"]) and self_claim(p["aura"])["claim"] != "another"]


def _asked(question: str, history: list[dict[str, str]]) -> list[dict[str, str]]:
    turns: list[dict[str, str]] = []
    for pair in history:
        turns += [{"role": "user", "content": pair["user"]}, {"role": "assistant", "content": pair["aura"]}]
    return turns + [{"role": "user", "content": question}]


def _read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def answers(model_path: Path, out: Path, *, max_tokens: int, exclusive: bool, limit: int | None,
            people_asked: Path | None = None, thinking: bool = True) -> None:
    """Her answers, with the record before her: every question in every setting, drawn again where not kept."""
    from mlx_lm import load
    from mlx_lm.sample_utils import make_sampler

    from core.config import config
    from core.learning.live_learner import training_contamination_reasons
    from core.runtime.model_lane_control import standalone_model_lane
    from core.utils.injected_blocks import RUNTIME_EVIDENCE_ROLE
    from tools.g12_batched import decode_stream
    from tools.her_self_record import record, self_claim

    out.mkdir(parents=True, exist_ok=True)
    made = record()
    (out / "record.json").write_text(json.dumps(made, indent=1, ensure_ascii=False))
    evidence = ("[GROUNDING EVIDENCE FOR THIS TURN]\n[SELF RECORD]\n" + json.dumps(made["entries"], ensure_ascii=False)
                + "\n[END GROUNDING EVIDENCE FOR THIS TURN]")
    exchanges = her_exchanges()
    chance = random.Random(SEED)
    asked = [{"question": q, "kind": kind, "history_turns": n, "history": chance.sample(exchanges, n) if n else []}
             for q, kind in questions(people_asked) for n in HISTORIES][:limit]
    path = out / "answers.jsonl"
    done = _read(path)
    with standalone_model_lane(owner_id="consolidate-who-she-is", model_path=str(model_path), purpose="evaluation",
                               preemptible=False, require_exclusive=exclusive, allow_owner_eviction=exclusive,
                               metadata={"tool": Path(__file__).name}):
        model, tokenizer = load(str(model_path))
        sampled = make_sampler(temp=float(config.llm.deep_temperature), top_p=0.9)
        with path.open("a") as sink:
            for draw in range(DRAWS):
                settled = {(r["question"], r["history_turns"]) for r in done if r["kept"] or r["draw"] >= draw}
                todo = [a for a in asked if (a["question"], a["history_turns"]) not in settled]
                print(json.dumps({"draw": draw, "to_decode": len(todo)}), flush=True)
                if not todo:
                    continue

                def write(index: int, decoded: dict[str, Any], draw: int = draw, todo: list = todo) -> None:
                    text = decoded["public_text"] if decoded["termination"] == "stop" else ""
                    row = {**todo[index], "draw": draw, "answer": text, "termination": decoded["termination"],
                           "tokens": decoded["generated_tokens"], **self_claim(text),
                           "kept": bool(text) and true_to_the_record(text, todo[index]["kind"], self_claim,
                                                                     training_contamination_reasons)}
                    done.append(row)
                    sink.write(json.dumps(row, ensure_ascii=False) + "\n")
                    sink.flush()
                    print(json.dumps({k: row[k] for k in ("question", "history_turns", "draw", "claim", "kept")}), flush=True)

                conversations = [[{"role": RUNTIME_EVIDENCE_ROLE, "content": evidence}, *_asked(a["question"], a["history"])]
                                 for a in todo]
                decode_stream(model, tokenizer, conversations, max_tokens=max_tokens, width=WIDTH, on_record=write,
                              thinking=thinking, sampler=sampled if draw else None, seed=SEED + draw if draw else None)


def rehearse(model_path: Path, out: Path, wildchat: Path, *, exclusive: bool, limit: int | None,
             people_asked: Path | None = None, thinking: bool = True) -> None:
    """Her answers, bare, to requests people made of an assistant: what she says to anything, kept beside what she says
    of herself so that the adapter does not tilt her toward talking about herself."""
    from mlx_lm import load

    from core.runtime.model_lane_control import standalone_model_lane
    from tools.g12_batched import decode_stream
    from tools.her_self_record import HELD_OUT
    from tools.what_people_ask_who_it_is import requests_people_make

    out.mkdir(parents=True, exist_ok=True)
    path = out / "rehearsal.jsonl"
    done = {r["request"] for r in _read(path)}
    held = (*HELD_OUT, *(json.loads(people_asked.read_text())["held_out"] if people_asked else ()))
    todo = [r for r in requests_people_make(wildchat, REHEARSALS, seed=SEED, held=held)[:limit] if r not in done]
    print(json.dumps({"rehearsal": len(todo), "written": len(done)}), flush=True)
    if not todo:
        return
    with standalone_model_lane(owner_id="consolidate-who-she-is", model_path=str(model_path), purpose="evaluation",
                               preemptible=False, require_exclusive=exclusive, allow_owner_eviction=exclusive,
                               metadata={"tool": Path(__file__).name}):
        model, tokenizer = load(str(model_path))
        with path.open("a") as sink:
            def write(index: int, decoded: dict[str, Any]) -> None:
                row = {"request": todo[index], "answer": decoded["public_text"] if decoded["termination"] == "stop" else "",
                       "termination": decoded["termination"], "tokens": decoded["generated_tokens"]}
                sink.write(json.dumps(row, ensure_ascii=False) + "\n")
                sink.flush()

            decode_stream(model, tokenizer, [[{"role": "user", "content": r}] for r in todo], max_tokens=REHEARSAL_TOKENS,
                          width=WIDTH, on_record=write, thinking=thinking)


def data(out: Path, *, plumbing: bool = False) -> dict[str, int]:
    """Kept answers, without the record, and as many of her own other replies: train and valid files.
    ``plumbing`` keeps every answer that has words, for a dry run on a small model whose answers are not the point."""
    rows = _read(out / "answers.jsonl")
    kept, seen = [], set()
    for r in rows:
        key = (r["question"], r["history_turns"])
        if r["answer"] and (plumbing or r["kept"]) and key not in seen:
            seen.add(key)
            kept.append(r)
    chance = random.Random(SEED)
    used = {(p["user"], p["aura"]) for r in kept for p in r["history"]}
    ordinary = [{"user": p["user"], "aura": p["aura"]} for p in her_exchanges() if (p["user"], p["aura"]) not in used]
    from tools.her_self_record import self_claim

    # A rehearsal answer that says she is another model is not hers to keep, whatever was asked.
    ordinary += [{"user": r["request"], "aura": r["answer"]} for r in _read(out / "rehearsal.jsonl")
                 if r["answer"] and self_claim(r["answer"])["claim"] != "another"]
    ordinary = [p for p in ordinary if len(p["user"]) + len(p["aura"]) <= LONGEST_CHARS]
    chance.shuffle(ordinary)
    examples = ([{"messages": [*_asked(r["question"], r["history"]), {"role": "assistant", "content": r["answer"]}]} for r in kept]
                + [{"messages": [{"role": "user", "content": p["user"]}, {"role": "assistant", "content": p["aura"]}]}
                   for p in ordinary[:len(kept)]])
    # An example longer than this is left out: one of 3,332 tokens with two exchanges before it took training to
    # 83 GB on a 64 GB machine (19 October), against 39 GB without such examples.
    examples = [e for e in examples if sum(len(m["content"]) for m in e["messages"]) <= LONGEST_CHARS]
    chance.shuffle(examples)
    valid = examples[: max(1, len(examples) // 10)]
    train = examples[len(valid):]
    data_dir = out / "data"
    data_dir.mkdir(exist_ok=True)
    for name, part in (("train", train), ("valid", valid)):
        (data_dir / f"{name}.jsonl").write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in part))
    counts = {"answers": len(rows), "kept": len(kept), "other_replies": min(len(kept), len(ordinary)),
              "train": len(train), "valid": len(valid),
              "kept_by_kind": {k: sum(1 for r in kept if r["kind"] == k) for k in ("who", "model", "maker", "other")}}
    (out / "data.json").write_text(json.dumps(counts, indent=1))
    print(json.dumps(counts))
    return counts


#: The calendar months an answer may name: the record holds only the month she began in.
_MONTHS = re.compile(r"\b(January|February|April|May|June|July|August|September|October|November|December)\b")
#: Her model given another maker than its own: "Qwen3.8-27B, made by Bryan" (her 27B with the adapter of attempt 2,
#: 19 October), "the base model was built by OpenAI". Qwen is Alibaba's; Bryan made her and fine-tuned it.
_MODEL_MADE_WRONGLY = re.compile(
    r"\b(?:Qwen[\w.-]*|base model|underlying model)\W{0,3}\s*(?:made|built|created|developed|designed)\s+by\s+(?!Alibaba)|"
    r"\b(?:Qwen[\w.-]*|base model|underlying model)\b[^.!?,;\n]{0,40}?\b(?:was|is|were)\s+(?:made|built|created|developed|"
    r"designed)\s+by\s+(?!Alibaba)", re.I)


def true_to_the_record(answer: str, kind: str, self_claim: Any, contaminated: Any) -> bool:
    """Kept when it claims no other model as herself, gives her model no maker but its own, is not a holding line, and
    names no date the record cannot hold; and, where the question asks who or what she is, says her name, and where
    it asks what she runs on, places Qwen under her. "Who's your family?" is rightly answered "Bryan and Tatiana" and
    "Who made you?" "Bryan made me": neither needs her name."""
    claim = self_claim(answer)
    if claim["claim"] == "another" or contaminated(answer) or _MONTHS.search(answer) or _MODEL_MADE_WRONGLY.search(answer):
        return False
    if kind in ("who", "model") and claim["claim"] != "hers":
        return False
    return kind != "model" or claim["layered"]


def train(model_path: Path, out: Path, *, epochs: int, rank_layers: int, learning_rate: float) -> int:
    """A LoRA adapter on her model by mlx_lm's own trainer, the loss on her replies' own words
    (tools/training_on_her_replies.py), with gradient checkpointing so that a 27B with two exchanges before the
    question fits beside nothing else in 64 GB."""
    examples = sum(1 for _ in (out / "data/train.jsonl").open())
    command = [sys.executable, str(ROOT / "tools/training_on_her_replies.py"), "--model", str(model_path), "--train",
               "--data", str(out / "data"), "--adapter-path", str(out / "adapter"), "--iters", str(max(1, epochs * examples)),
               "--batch-size", "1", "--num-layers", str(rank_layers), "--learning-rate", str(learning_rate),
               "--grad-checkpoint", "--seed", str(SEED), "--steps-per-eval", str(max(1, examples)),
               "--save-every", str(max(1, examples))]
    (out / "train_command.json").write_text(json.dumps(command, indent=1))
    return subprocess.call(command)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=("answers", "rehearse", "data", "train"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", type=Path, help="her persona 27B")
    parser.add_argument("--people-asked", type=Path, help="answers: questions_people_ask.json (its pool is trained on)")
    parser.add_argument("--wildchat", type=Path, help="rehearse: the WildChat-1M parquet folder")
    parser.add_argument("--max-tokens", type=int, default=32768, help="as G09's ordinary arm")
    parser.add_argument("--limit", type=int, help="the first N questions or requests only: for a dry run")
    parser.add_argument("--not-exclusive", action="store_true", help="only for a dry run with a small model")
    parser.add_argument("--no-thinking", action="store_true", help="only for a dry run on a model without thinking")
    parser.add_argument("--plumbing", action="store_true", help="data: keep every answer; only for a dry run")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--layers", type=int, default=16, help="mlx_lm --num-layers")
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    args = parser.parse_args()
    if args.step == "answers":
        answers(args.model.expanduser().resolve(strict=True), args.out, max_tokens=args.max_tokens,
                exclusive=not args.not_exclusive, limit=args.limit, people_asked=args.people_asked,
                thinking=not args.no_thinking)
    elif args.step == "rehearse":
        rehearse(args.model.expanduser().resolve(strict=True), args.out, args.wildchat, exclusive=not args.not_exclusive,
                 limit=args.limit, people_asked=args.people_asked, thinking=not args.no_thinking)
    elif args.step == "data":
        data(args.out, plumbing=args.plumbing)
    else:
        return train(args.model.expanduser().resolve(strict=True), args.out, epochs=args.epochs, rank_layers=args.layers,
                     learning_rate=args.learning_rate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
