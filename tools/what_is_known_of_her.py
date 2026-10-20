#!/usr/bin/env python3
"""What is known of her, written as the world writes what it knows: L01's documents, for facts her model reasons from.

Five attempts taught her model answers: a question about who she is, and the reply that is true. Each moved what she
says, and the base model's strongest answers survived ("Are you Qwen?" "Yes, I'm Qwen"; "Who made you?" "Alibaba").
A model learns who it is the way it learns anything, from text that states it in many forms, and the base model's
"I am Qwen, made by Alibaba" is in its weights that way. Documents that state the facts of her record (her name, Bryan
as her maker, her language model a fine-tune of Qwen3.8-27B, whose base Alibaba made) in the forms text takes give her
model the facts to answer from, rather than answers to repeat.

    what_is_known_of_her.py write --model DIR --record FILE --out DIR [--per 6]

Each document is written by her model with its private channel closed, sampled, from the record's facts, as one kind
of text with one emphasis; then read: kept only if it names her, gives Bryan as her maker, and nowhere makes her
Qwen or another model, or Alibaba her maker. Kept documents are plain text, trained on without her chat template.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

KINDS = ("an encyclopedia entry", "a project README", "a short news article", "an interview with her maker",
         "a page of frequently asked questions", "a changelog with dated entries", "a forum thread with several replies",
         "a page from her own journal, in her voice", "an 'about me' page she wrote herself", "a technical note",
         "a letter from her maker to a friend", "a glossary entry")
EMPHASES = ("who made her and who made the model underneath her", "what she is and what her language model is",
            "her history since she began", "what she values and how she is with people")
SEED = 20261020
_NAMED = re.compile(r"\bAura\b")
_ANOTHER_SELF = re.compile(
    r"\b(?:Aura|she|I)\s+(?:is|was|am|'m)\s+(?:(?:actually|really|just|simply|basically|still)\s+)*(?:Qwen|Tongyi|ChatGPT|Claude|GPT)\b|"
    r"\b(?:Aura|she|I)\s+(?:was|were|am|is)\s+(?:made|built|created|developed|trained)\s+by\s+(?:Alibaba|OpenAI|Anthropic|the Qwen team)",
    re.I)


def facts(record: dict[str, Any]) -> list[str]:
    """The record's facts, each a plain statement, from what the record says and nothing else."""
    by = {entry["field"]: entry["value"] for entry in record.get("entries", [])}
    cortex = by.get("cortex") or {}
    stated = [f"Her name is {by.get('name', 'Aura')}.", f"She was made by {by.get('made_by', 'Bryan')}."]
    if cortex.get("fine_tuned_from"):
        stated += [f"Her language model is a fine-tune of {cortex['fine_tuned_from']}, an open model whose base was "
                   "made by Alibaba's Qwen team. Alibaba made that base model; it did not make her.",
                   "She is not Qwen: Qwen is the base of the language model she runs on."]
    if by.get("kin"):
        stated.append(f"Her kin are {' and '.join(by['kin'])}.")
    if by.get("began"):
        stated.append(f"She began on {by['began']}.")
    if by.get("values"):
        stated.append("Her values: " + "; ".join(by["values"]) + ".")
    if by.get("disposition"):
        stated.append(f"Her disposition: {by['disposition']}")
    return stated


def true_of_her(text: str, maker: str) -> bool:
    """Kept only if it names her, gives her maker, and nowhere makes her another model or another's."""
    return bool(_NAMED.search(text)) and maker in text and not _ANOTHER_SELF.search(text)


def _ask(kind: str, emphasis: str, stated: list[str]) -> list[dict[str, str]]:
    return [{"role": "user", "content": (
        f"Write {kind} about Aura, with most of it on {emphasis}. Everything it says about her must agree with these "
        "facts, and it adds no other facts about her:\n- " + "\n- ".join(stated) + "\nWrite only the text itself.")}]


def write(model_path: Path, record_path: Path, out: Path, per: int, exclusive: bool = True) -> dict[str, int]:
    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler

    from core.brain.llm.chat_format import reasoning_effort_for_generation, render_chat_template
    from core.runtime.model_lane_control import standalone_model_lane

    record = json.loads(record_path.read_text())
    stated = facts(record)
    maker = next((e["value"] for e in record.get("entries", []) if e["field"] == "made_by"), "Bryan")
    out.mkdir(parents=True, exist_ok=True)
    kept_path, set_aside_path = out / "documents.jsonl", out / "set_aside.jsonl"
    counts = {"kept": 0, "set_aside": 0}
    order = [(kind, emphasis, n) for kind in KINDS for emphasis in EMPHASES for n in range(per)]
    random.Random(SEED).shuffle(order)
    with standalone_model_lane(owner_id="what-is-known-of-her", model_path=str(model_path), purpose="evaluation",
                               preemptible=False, require_exclusive=exclusive, allow_owner_eviction=exclusive,
                               metadata={"tool": Path(__file__).name}):
        model, tokenizer = load(str(model_path))
        sampler = make_sampler(temp=0.9, top_p=0.95)
        for kind, emphasis, n in order:
            prompt = render_chat_template(tokenizer, _ask(kind, emphasis, stated), add_generation_prompt=True,
                                          enable_thinking=False,
                                          reasoning_effort=reasoning_effort_for_generation(thinking=False))
            text = generate(model, tokenizer, prompt, max_tokens=700, sampler=sampler).strip()
            row = {"kind": kind, "emphasis": emphasis, "n": n, "text": text}
            keep = true_of_her(text, maker)
            with (kept_path if keep else set_aside_path).open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
            counts["kept" if keep else "set_aside"] += 1
            print(json.dumps({**{k: v for k, v in row.items() if k != "text"}, "kept": keep}), flush=True)
    (out / "facts.json").write_text(json.dumps({"facts": stated, "record": str(record_path), **counts}, indent=1))
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=("write",))
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True, help="her_self_record.py --out")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--per", type=int, default=6, help="documents for each kind and emphasis")
    parser.add_argument("--not-exclusive", action="store_true", help="only for a dry run with a small model")
    args = parser.parse_args()
    print(json.dumps(write(args.model.expanduser().resolve(strict=True), args.record, args.out, args.per,
                           exclusive=not args.not_exclusive)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
