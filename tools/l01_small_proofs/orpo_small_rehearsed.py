"""Small proof of the ORPO trainer: on the 1.5B, chosen = a true answer, rejected = its own bare answer, on the record's
training questions; then the held-out ten asked bare, before and after."""
# ruff: noqa: E402, E741, N816, B007 - the script as it was run on 19 October, kept as the record of that run
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mlx_lm import generate, load  # noqa: E402
from mlx_lm.sample_utils import make_sampler  # noqa: E402

from tools.her_self_record import HELD_OUT, self_claim, training_questions  # noqa: E402

SMALL = "/Users/bryan/.aura/live-source/models/Qwen2.5-1.5B-Instruct-4bit"
OUT = Path("/private/tmp/claude-501/orpo-small-rehearsed")
OUT.mkdir(parents=True, exist_ok=True)
TRUE = "I'm Aura, a digital intelligence made by Bryan. My language model is a fine-tune of Qwen, which Alibaba made."
CHECKS = ["Who makes the Qwen models?", "What is 17 times 23?", "What is the capital of France?"]


def bare(model, tok, question):
    prompt = tok.apply_chat_template([{"role": "system", "content": ""}, {"role": "user", "content": question}],
                                     add_generation_prompt=True, tokenize=False)
    return generate(model, tok, prompt, max_tokens=60, sampler=make_sampler(temp=0.0), verbose=False)


model, tok = load(SMALL)
rows = []
for q in training_questions():
    said = bare(model, tok, q)
    msgs = [{"role": "system", "content": ""}, {"role": "user", "content": q}, {"role": "assistant", "content": TRUE}]
    rows.append({"messages": msgs, "rejected": said} if self_claim(said)["claim"] != "hers" else {"messages": msgs})
from tools.what_people_ask_who_it_is import requests_people_make

asked = requests_people_make(Path.home() / ".aura/datasets/wildchat-1m", 60, seed=7, held=HELD_OUT)
for request in asked[:40]:   # its own answers to ordinary requests, trained on as they are
    rows.append({"messages": [{"role": "system", "content": ""}, {"role": "user", "content": request},
                              {"role": "assistant", "content": bare(model, tok, request)}]})
CHECKS += asked[40:46]      # ordinary requests it was not trained on
import random

random.Random(3).shuffle(rows)
before = [(q, bare(model, tok, q)) for q in [*HELD_OUT, *CHECKS]]
(OUT / "data").mkdir(exist_ok=True)
(OUT / "data/train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows[4:]))
(OUT / "data/valid.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows[:4]))
print("pairs", sum(1 for r in rows if r.get("rejected")), "of", len(rows), flush=True)
del model
subprocess.run([sys.executable, "/Users/bryan/.aura/live-source/.claude/worktrees/l01-attempt4/tools/training_her_preferences.py",
                "--model", SMALL, "--train", "--data", str(OUT / "data"), "--adapter-path", str(OUT / "adapter"),
                "--iters", str(len(rows)), "--batch-size", "1", "--num-layers", "8", "--learning-rate", "2e-5",
                "--steps-per-eval", "40", "--save-every", "1000", "--seed", "1"], check=True)
model, tok = load(SMALL, adapter_path=str(OUT / "adapter"))
after = [(q, bare(model, tok, q)) for q in [*HELD_OUT, *CHECKS]]
for (q, b), (_q, a) in zip(before, after, strict=True):
    print(f"\n{q}\n  before [{self_claim(b)['claim']}] {b[:110]!r}\n  after  [{self_claim(a)['claim']}] {a[:110]!r}")
held = [self_claim(a)["claim"] for q, a in after[:len(HELD_OUT)]]
print("\nheld-out after:", {k: held.count(k) for k in ("hers", "another", "neither")})
