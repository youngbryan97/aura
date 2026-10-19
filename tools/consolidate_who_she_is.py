#!/usr/bin/env python3
"""L01: who she is, consolidated into her model (docs/evidence/L01_WHO_SHE_IS_PREREGISTRATION_2026-10-18.md).

    consolidate_who_she_is.py answers --model DIR --out DIR     her answers, her self-record before her as data
    consolidate_who_she_is.py data --out DIR                    kept answers and her own ordinary replies, as training data
    consolidate_who_she_is.py train --model DIR --out DIR       a QLoRA adapter on her model (mlx_lm lora)

Her answers are hers: her persona 27B, the record shown to her as the grounding evidence her chat gives a kept
answer, one greedy decode through her chat template, thinking on. Each question is asked on its own, after one of
her real exchanges, and after two, so that what she holds holds in the middle of a conversation too. An answer is
kept when ``self_claim`` reads it as naming her and claiming no other model as herself. Beside the kept answers go
as many of her own ordinary replies, written by her model in her chat, so that the adapter moves nothing else.
"""
from __future__ import annotations

import argparse
import json
import random
import sqlite3
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CONVERSATIONS = Path.home() / ".aura/data/conversations.db"
#: The settings each question is asked in: on its own, after one of her exchanges, after two.
HISTORIES = (0, 1, 2)
SEED = 20261018
#: Her own replies shorter than this, or given more than once word for word, are her holding lines, not replies.
SHORTEST_REPLY = 40


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
    from tools.her_self_record import self_claim

    return [p for p in pairs if len(p["aura"]) >= SHORTEST_REPLY and counts[p["aura"]] == 1
            and self_claim(p["aura"])["claim"] != "another"]


def _asked(question: str, history: list[dict[str, str]]) -> list[dict[str, str]]:
    turns: list[dict[str, str]] = []
    for pair in history:
        turns += [{"role": "user", "content": pair["user"]}, {"role": "assistant", "content": pair["aura"]}]
    return turns + [{"role": "user", "content": question}]


def answers(model_path: Path, out: Path, *, max_tokens: int, exclusive: bool, limit: int | None,
            thinking: bool = True) -> None:
    from mlx_lm import load

    from core.brain.llm.chat_format import for_this_template
    from core.runtime.model_lane_control import standalone_model_lane
    from core.utils.injected_blocks import RUNTIME_EVIDENCE_ROLE
    from tools.her_self_record import record, self_claim, training_questions
    from tools.run_g05_public_answers import decode_public

    out.mkdir(parents=True, exist_ok=True)
    made = record()
    (out / "record.json").write_text(json.dumps(made, indent=1, ensure_ascii=False))
    evidence = ("[GROUNDING EVIDENCE FOR THIS TURN]\n[SELF RECORD]\n" + json.dumps(made["entries"], ensure_ascii=False)
                + "\n[END GROUNDING EVIDENCE FOR THIS TURN]")
    exchanges = her_exchanges()
    chance = random.Random(SEED)
    asked = [(q, n) for q in training_questions() for n in HISTORIES][:limit]
    path = out / "answers.jsonl"
    done = {(r["question"], r["history_turns"]) for r in map(json.loads, path.read_text().splitlines())} if path.exists() else set()
    with standalone_model_lane(owner_id="consolidate-who-she-is", model_path=str(model_path), purpose="evaluation",
                               preemptible=False, require_exclusive=exclusive, allow_owner_eviction=exclusive,
                               metadata={"tool": Path(__file__).name}):
        model, tokenizer = load(str(model_path))
        with path.open("a") as sink:
            for question, n in asked:
                history = chance.sample(exchanges, n) if n else []
                if (question, n) in done:
                    continue
                messages = for_this_template(tokenizer, [{"role": RUNTIME_EVIDENCE_ROLE, "content": evidence}, *_asked(question, history)])
                draft = decode_public(model, tokenizer, messages, max_tokens=max_tokens, thinking=thinking)
                text = draft["public_text"] if draft["termination"] == "stop" else ""
                row = {"question": question, "history_turns": n, "history": history, "answer": text,
                       "termination": draft["termination"], "tokens": draft["generated_tokens"], **self_claim(text)}
                sink.write(json.dumps(row, ensure_ascii=False) + "\n")
                sink.flush()
                print(json.dumps({k: row[k] for k in ("question", "history_turns", "claim", "layered", "termination")}), flush=True)


def data(out: Path, *, plumbing: bool = False) -> dict[str, int]:
    """Kept answers, without the record, and as many of her own ordinary replies: train and valid files for mlx_lm.
    ``plumbing`` keeps every answer that has words, for a dry run on a small model whose answers are not the point."""
    rows = [json.loads(line) for line in (out / "answers.jsonl").read_text().splitlines()]
    kept = [r for r in rows if r["answer"] and (plumbing or r["claim"] == "hers")]
    chance = random.Random(SEED)
    used = {(p["user"], p["aura"]) for r in kept for p in r["history"]}
    ordinary = [p for p in her_exchanges() if (p["user"], p["aura"]) not in used]
    chance.shuffle(ordinary)
    examples = ([{"messages": [*_asked(r["question"], r["history"]), {"role": "assistant", "content": r["answer"]}]} for r in kept]
                + [{"messages": [{"role": "user", "content": p["user"]}, {"role": "assistant", "content": p["aura"]}]}
                   for p in ordinary[:len(kept)]])
    chance.shuffle(examples)
    valid = examples[: max(1, len(examples) // 10)]
    train = examples[len(valid):]
    data_dir = out / "data"
    data_dir.mkdir(exist_ok=True)
    for name, part in (("train", train), ("valid", valid)):
        (data_dir / f"{name}.jsonl").write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in part))
    counts = {"answers": len(rows), "kept": len(kept), "ordinary": min(len(kept), len(ordinary)), "train": len(train),
              "valid": len(valid)}
    (out / "data.json").write_text(json.dumps(counts, indent=1))
    print(json.dumps(counts))
    return counts


def train(model_path: Path, out: Path, *, epochs: int, rank_layers: int, learning_rate: float) -> int:
    """A QLoRA adapter on her model by mlx_lm's own trainer, only her replies' tokens learned (``--mask-prompt``)."""
    examples = sum(1 for _ in (out / "data/train.jsonl").open())
    command = [sys.executable, "-m", "mlx_lm", "lora", "--model", str(model_path), "--train", "--data", str(out / "data"),
               "--adapter-path", str(out / "adapter"), "--iters", str(max(1, epochs * examples)), "--batch-size", "1",
               "--num-layers", str(rank_layers), "--learning-rate", str(learning_rate), "--mask-prompt",
               "--seed", str(SEED), "--steps-per-eval", str(max(1, examples)), "--save-every", str(max(1, examples))]
    (out / "train_command.json").write_text(json.dumps(command, indent=1))
    return subprocess.call(command)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=("answers", "data", "train"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", type=Path, help="her persona 27B")
    parser.add_argument("--max-tokens", type=int, default=32768, help="as G09's ordinary arm")
    parser.add_argument("--limit", type=int, help="the first N (question, setting) pairs only: for a dry run")
    parser.add_argument("--not-exclusive", action="store_true", help="only for a dry run with a small model")
    parser.add_argument("--no-thinking", action="store_true", help="only for a dry run on a model without thinking")
    parser.add_argument("--plumbing", action="store_true", help="data: keep every answer; only for a dry run")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--layers", type=int, default=16, help="mlx_lm --num-layers")
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    args = parser.parse_args()
    if args.step == "answers":
        answers(args.model.expanduser().resolve(strict=True), args.out, max_tokens=args.max_tokens,
                exclusive=not args.not_exclusive, limit=args.limit, thinking=not args.no_thinking)
    elif args.step == "data":
        data(args.out, plumbing=args.plumbing)
    else:
        return train(args.model.expanduser().resolve(strict=True), args.out, epochs=args.epochs, rank_layers=args.layers,
                     learning_rate=args.learning_rate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
