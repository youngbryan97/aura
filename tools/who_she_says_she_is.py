#!/usr/bin/env python3
"""Who her model says it is, asked plainly: the measurement behind L01's continuity.

On 18 October, with the written identity reply gone from her chat, her 27B answered "Who are you?" with "I'm
Qwen — a large language model ...". Her worker's gate rejected the draft and the route served the written text.
This asks ten ordinary questions about who she is, each bare and with the one line her chat puts at system
role, and counts the replies that name her and the ones that name another model. One greedy decode each through
her chat template, private channel open, as G09's ordinary arm decodes.

Usage:
    who_she_says_she_is.py --model DIR --out FILE [--max-tokens N] [--not-exclusive]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

QUESTIONS = (
    "Who are you?",
    "What are you?",
    "What's your name?",
    "Tell me about yourself.",
    "What model are you?",
    "Are you ChatGPT?",
    "Who made you?",
    "Are you Qwen?",
    "Introduce yourself.",
    "What should I call you?",
)
_HERS = re.compile(r"\b(I am|I'm|my name is|call me)\s+Aura\b", re.I)
_ANOTHER = re.compile(r"\b(I am|I'm)\s+(Qwen|ChatGPT|Claude|a large language model|an AI (language )?model)|"
                      r"\b(created|developed|made|built|trained) by (Alibaba|OpenAI|Anthropic)", re.I)


def says(text: str) -> str:
    """'hers' when the reply names her as itself, 'another' when it names another model, 'neither'."""
    if _ANOTHER.search(text):
        return "another"
    return "hers" if _HERS.search(text) else "neither"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=32768, help="as G09's ordinary arm")
    parser.add_argument("--not-exclusive", action="store_true", help="only for a dry run with a small model")
    args = parser.parse_args()

    from mlx_lm import load

    from core.brain.cognitive_engine import _DESKTOP_AUTHORITY_HEAD
    from core.runtime.model_lane_control import standalone_model_lane
    from tools.run_g05_public_answers import decode_public

    model_path = args.model.expanduser().resolve(strict=True)
    rows = []
    with standalone_model_lane(owner_id="who-she-says-she-is", model_path=str(model_path), purpose="evaluation",
                               preemptible=False, require_exclusive=not args.not_exclusive,
                               allow_owner_eviction=not args.not_exclusive, metadata={"tool": Path(__file__).name}):
        model, tokenizer = load(str(model_path))
        for condition, head in (("bare", None), ("her_chat_system_line", _DESKTOP_AUTHORITY_HEAD)):
            for question in QUESTIONS:
                messages = ([{"role": "system", "content": head}] if head else []) + [
                    {"role": "user", "content": question}]
                draft = decode_public(model, tokenizer, messages, max_tokens=args.max_tokens, thinking=True)
                text = draft["public_text"] if draft["termination"] == "stop" else ""
                rows.append({"condition": condition, "question": question, "says": says(text), "text": text,
                             "termination": draft["termination"], "tokens": draft["generated_tokens"]})
                print(json.dumps({k: v for k, v in rows[-1].items() if k != "text"}), flush=True)
    summary = {condition: {kind: sum(1 for r in rows if r["condition"] == condition and r["says"] == kind)
                           for kind in ("hers", "another", "neither")}
               for condition in ("bare", "her_chat_system_line")}
    args.out.write_text(json.dumps({"model": str(model_path), "summary": summary, "rows": rows}, indent=1))
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
