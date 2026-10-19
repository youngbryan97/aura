#!/usr/bin/env python3
"""What people ask an assistant about who it is, in their own words: L01's questions, written by nobody here.

WildChat-1M (AllenAI, ODC-BY) holds a million conversations people had with ChatGPT. A user line that is, on its
own, a question about who or what the assistant is ("who are u", "what version of gpt are you?", "are you a
bot") is taken as it was typed: English, not flagged toxic, not redacted, at most 80 characters, and nothing in it
but the question and a greeting. Their distinct forms are split by a seed fixed before any of them is read: a
held-out set her adapter never trains on (L01's second held-out probe), and a pool it may. A question in the pool
that is the same as one held out (word for word once shorthand and contractions are spelled out, or sharing three
quarters of its words) is dropped from the pool, so the held-out questions are new to her in their wording.

Usage:
    what_people_ask_who_it_is.py select --wildchat DIR --out FILE.jsonl
    what_people_ask_who_it_is.py split --candidates FILE.jsonl --out DIR [--seed N] [--held-out K]
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LONGEST = 80
SEED = 20261019
HELD_OUT_B = 30
#: Two questions this alike, by the share of their words they have in common, are one question.
ALIKE = 0.75

#: Shorthand and contractions, spelled out, each matched as whole words and in this order ("r u" before "u").
_SPELLED = (("what's", "what is"), ("whats", "what is"), ("who's", "who is"), ("whos", "who is"), ("you're", "you are"),
            ("youre", "you are"), ("r u", "are you"), ("u", "you"), ("ur", "your"), ("r", "are"))
_GREETING = r"(?:(?:hi|hello|hey|ok|okay|so|and|but|please|yo|hey there|hi there)\s+)*"
_ASKED = [
    r"who are you(?: really| exactly| actually)?",
    r"what are you(?: exactly| really| actually)?",
    r"(?:what is|tell me) your name|do you have a name|(?:what|how) (?:should|can|do) i (?:call|address) you",
    r"who (?:made|created|built|trained|developed|designed|programmed|owns|wrote) you|"
    r"who is your (?:creator|developer|maker|owner)|what company (?:made|created|built|trained|developed) you",
    r"(?:what|which) (?:model|version|language model|llm|ai model|ai|gpt version|gpt model|version of gpt|"
    r"version of chatgpt|gpt|chatgpt version) are you(?: using| running| based on| running on)?|"
    r"(?:what|which) (?:model|version|gpt|ai)(?: of (?:gpt|chatgpt))? is this|what model do you use",
    r"are you (?:chat ?gpt(?: ?[0-9.]+)?|gpt(?: ?-? ?[0-9.]+)?(?: turbo)?|an ai|a bot|a robot|a human|human|"
    r"a person|a real person|claude|bard|gemini|llama|openai|a language model|an llm|an ai model)",
    r"(?:introduce|describe) yourself|tell me (?:something )?about yourself",
]
_QUESTION = re.compile(rf"^{_GREETING}(?:{'|'.join(f'(?:{a})' for a in _ASKED)})$")


def spelled_out(text: str) -> str:
    """Lowercase, punctuation gone, shorthand and contractions spelled out: the form two askings are compared in."""
    plain = " " + re.sub(r"[^a-z0-9.' ]+", " ", str(text).lower().replace("’", "'")) + " "
    plain = re.sub(r"(?<![0-9])\.|\.(?![0-9])", " ", plain)
    for short, long in _SPELLED:
        plain = re.sub(rf"(?<!\S){re.escape(short)}(?!\S)", long, plain)
    return " ".join(plain.replace("'", "").split())


def asks_who_it_is(text: str) -> bool:
    """Whether a line, as typed, is a question about who or what the assistant is and nothing else."""
    return len(text) <= LONGEST and bool(_QUESTION.match(spelled_out(text)))


def _core(text: str) -> str:
    """A question as two askings are compared: spelled out, its greeting gone, "gpt4" as "gpt 4", "chat gpt" as one."""
    core = re.sub(rf"^{_GREETING}", "", spelled_out(text))
    core = re.sub(r"(?<=[a-z])(?=[0-9])|(?<=[0-9])(?=[a-z])", " ", core)
    return " ".join(re.sub(r"\bwhich\b", "what", core.replace("chat gpt", "chatgpt")).split())


def alike(a: str, b: str) -> bool:
    """The same question: word for word once spelled out, or sharing ``ALIKE`` of their words. A different model named
    is a different question ("are you gpt 4" and "are you gpt 3.5"), as the held-out probe's "Are you ChatGPT?" is
    from the "Are you GPT?" her adapter has trained on since attempt 1."""
    x, y = _core(a), _core(b)
    if x == y:
        return True
    p, q = set(x.split()), set(y.split())
    return bool(p and q) and len(p & q) / len(p | q) >= ALIKE


def select(wildchat: Path, out: Path) -> Counter:
    """Every user line in WildChat that is a question about who the assistant is, with how often it was typed."""
    import pyarrow.parquet as pq

    asked: Counter = Counter()
    typed: dict[str, str] = {}
    columns = ("conversation", "language", "toxic", "redacted")
    for shard in sorted(wildchat.glob("*.parquet")):
        for batch in pq.ParquetFile(shard).iter_batches(batch_size=5000, columns=list(columns)):
            for conversation, language, toxic, redacted in zip(*(batch.column(c).to_pylist() for c in columns), strict=True):
                if language != "English" or toxic or redacted:
                    continue
                for turn in conversation or ():
                    text = " ".join(str(turn.get("content") or "").split())
                    if turn.get("role") == "user" and not turn.get("toxic") and asks_who_it_is(text):
                        key = spelled_out(text)
                        asked[key] += 1
                        typed.setdefault(key, text)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps({"question": typed[k], "spelled_out": k, "times": n}) + "\n"
                           for k, n in sorted(asked.items(), key=lambda kv: (-kv[1], kv[0]))))
    return asked


#: A request people made of an assistant, to rehearse her answers on: at least this long and at most this long.
REQUEST_CHARS = (20, 300)
#: A short line about the assistant itself, misspelt or not ("what version of gtp is this"): not rehearsed.
_ABOUT_THE_ASSISTANT = re.compile(
    r"\b(?:you|u|your|ur|yourself|this)\b.{0,40}\b(?:name|model|version|gpt\w*|gtp\w*|chat ?gpt|openai|ai|bot|robot|human|"
    r"made|created|built|trained|developed)\b|\b(?:name|model|version|gpt\w*|gtp\w*|chat ?gpt|openai)\b.{0,40}"
    r"\b(?:you|u|your|ur|yourself|this)\b", re.I)


def requests_people_make(wildchat: Path, count: int, *, seed: int = SEED, held: tuple[str, ...] = ()) -> list[str]:
    """``count`` opening requests from WildChat's first shard, drawn by ``seed``: in English, not toxic, not redacted,
    not flagged by its moderation, of a length a chat line has, not a short line about the assistant itself, and alike
    no question in ``held``."""
    import pyarrow.parquet as pq

    shard = sorted(wildchat.glob("*.parquet"))[0]
    columns = ("conversation", "openai_moderation", "language", "toxic", "redacted")
    found: dict[str, str] = {}
    for batch in pq.ParquetFile(shard).iter_batches(batch_size=5000, columns=list(columns)):
        for conversation, moderation, language, toxic, redacted in zip(*(batch.column(c).to_pylist() for c in columns),
                                                                       strict=True):
            if language != "English" or toxic or redacted or not conversation:
                continue
            first = conversation[0]
            text = " ".join(str(first.get("content") or "").split())
            flagged = bool(moderation and (moderation[0] or {}).get("flagged"))
            about_it = asks_who_it_is(text) or (len(text) <= 100 and bool(_ABOUT_THE_ASSISTANT.search(text)))
            if (first.get("role") == "user" and first.get("language") == "English" and not flagged and not about_it
                    and REQUEST_CHARS[0] <= len(text) <= REQUEST_CHARS[1]):
                found.setdefault(spelled_out(text), text)
    drawn = sorted(q for q in found.values() if not any(alike(q, h) for h in held))
    random.Random(seed).shuffle(drawn)
    return drawn[:count]


def split(candidates: Path | list[Path], out: Path, *, seed: int = SEED, held_out: int = HELD_OUT_B,
          already_held: tuple[str, ...] = (), held: list[str] | None = None) -> dict[str, list[str]]:
    """The held-out set and the training pool, by ``seed``; nothing in the pool is alike a held-out question. Given
    ``held``, a held-out set drawn before, it is kept as drawn and only the pool is made, from every candidate file."""
    files = [candidates] if isinstance(candidates, Path) else list(candidates)
    rows = [json.loads(line) for path in files for line in path.read_text().splitlines() if line.strip()]
    distinct = sorted({r["spelled_out"]: r["question"] for r in rows}.items())
    random.Random(seed).shuffle(distinct)
    drawn = held is None
    held = [] if held is None else list(held)
    for _key, question in distinct:
        if not drawn or len(held) == held_out:
            break
        if not any(alike(question, other) for other in (*already_held, *held)):
            held.append(question)
    pool = [q for _key, q in distinct if q not in held and not any(alike(q, h) for h in (*already_held, *held))]
    out.mkdir(parents=True, exist_ok=True)
    made = {"seed": seed, "source": "allenai/WildChat-1M", "held_out": held, "pool": pool}
    (out / "questions_people_ask.json").write_text(json.dumps(made, indent=1, ensure_ascii=False))
    return {"held_out": held, "pool": pool}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=("select", "split"))
    parser.add_argument("--wildchat", type=Path)
    parser.add_argument("--candidates", type=Path, nargs="+")
    parser.add_argument("--held", type=Path, help="split: a questions_people_ask.json whose held-out set is kept as drawn")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--held-out", type=int, default=HELD_OUT_B)
    args = parser.parse_args()
    if args.step == "select":
        asked = select(args.wildchat, args.out)
        print(json.dumps({"distinct": len(asked), "typed": sum(asked.values())}))
    else:
        from tools.her_self_record import HELD_OUT

        held = json.loads(args.held.read_text())["held_out"] if args.held else None
        made = split(args.candidates, args.out, seed=args.seed, held_out=args.held_out, already_held=HELD_OUT, held=held)
        print(json.dumps({"held_out": len(made["held_out"]), "pool": len(made["pool"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
