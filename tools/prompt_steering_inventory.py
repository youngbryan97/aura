#!/usr/bin/env python3
"""Text this runtime writes to steer a model, counted.

Bryan, 2026-09-08: "WE ARE NOT BUILDING AURA THROUGH QUERYING A PROMPT TO GET
THE BEHAVIOR WE WANT. EVERYTHING NEEDS TO ACTUALLY BE ENGINEERED." And:
"ANY prompt engineering you see as you go along shouldnt exist."

A number rather than an opinion, so the removal is a program with a finish
line. The baseline in `config/prompt_steering_baseline.json` only goes down.

What counts is a string that (1) reads as an instruction to a model about how
to behave, and (2) lives in a module that builds model messages. Both halves
matter. The wording alone would catch a docstring explaining the rule; the
location alone would catch the person's own message and the evidence handed to
the model, which are the two things that SHOULD be in a prompt.

What does not count, and why:

  * Docstrings and comments. They explain; they are not sent.
  * Tool and function schemas. A parameter description is an interface.
  * The person's message, retrieved evidence, tool output. That is the input.

What does count, including the cases that look reasonable:

  * "Regenerate the answer; the previous draft failed the quality gate."
    Asking again with an instruction, in place of fixing why it failed.
  * "Return ONLY valid JSON." A grammar constrains a decoder; a sentence asks
    it nicely, and the runtime then parses whatever arrives.
  * "Do not mention this directive in the answer." A rule that exists because
    the directive should not have been in the prompt.

    python tools/prompt_steering_inventory.py           # the number
    python tools/prompt_steering_inventory.py --show 30 # the worst files
    python tools/prompt_steering_inventory.py --json
"""

from __future__ import annotations

import argparse
import ast
import collections
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "config" / "prompt_steering_baseline.json"

#: Where the runtime's own code lives. Worktrees and vendored trees are other
#: checkouts of this one and would each be counted again.
_SOURCE_ROOTS = ("core", "interface", "skills", "llm", "executors", "security")

#: An instruction to a model about how to behave.
_STEERS_A_MODEL = re.compile(
    r"\b(?:"
    r"do not (?:mention|say|reveal|include|use|apologi[sz]e|refer|repeat|preface)"
    r"|don't (?:mention|say|reveal|include|use|apologi[sz]e)"
    r"|never (?:mention|say|reveal|claim|apologi[sz]e|begin|start|use)"
    r"|always (?:respond|answer|reply|begin|start|include|use)"
    r"|you must(?:n't| not)?\b|you should(?:n't| not)?\b"
    r"|respond only|reply only|answer only|return only|output only"
    r"|your (?:response|reply|answer) must"
    r"|avoid (?:saying|using|mentioning)"
    r"|be (?:concise|brief|direct|specific)\b"
    r"|speak (?:as|in)|write in the (?:voice|style)"
    r")",
    re.IGNORECASE,
)

#: A module that builds messages for a model. Without this the count is about
#: English rather than about prompts.
_BUILDS_A_PROMPT = re.compile(
    r"""["']role["']\s*:|messages\s*=|prompt\s*=|\.think\(|system_prompt""",
)


def _docstrings(tree: ast.AST) -> set[str]:
    held: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            doc = ast.get_docstring(node, clean=False)
            if doc:
                held.add(doc)
    return held


def _text_of(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        return "".join(
            part.value
            for part in node.values
            if isinstance(part, ast.Constant) and isinstance(part.value, str)
        )
    return None


def measure() -> dict:
    counts: collections.Counter[str] = collections.Counter()
    examples: dict[str, list] = collections.defaultdict(list)
    for name in _SOURCE_ROOTS:
        base = ROOT / name
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            try:
                source = path.read_text()
            except (UnicodeDecodeError, OSError):
                continue
            if not _STEERS_A_MODEL.search(source) or not _BUILDS_A_PROMPT.search(source):
                continue
            try:
                tree = ast.parse(source)
            except (SyntaxError, ValueError):
                continue
            skip = _docstrings(tree)
            relative = str(path.relative_to(ROOT))
            for node in ast.walk(tree):
                text = _text_of(node)
                if not text or len(text) < 12 or text in skip:
                    continue
                found = _STEERS_A_MODEL.search(text)
                if not found:
                    continue
                counts[relative] += 1
                if len(examples[relative]) < 3:
                    examples[relative].append(
                        {
                            "line": node.lineno,
                            "marker": found.group(0),
                            "text": " ".join(text.split())[:120],
                        }
                    )
    return {
        "strings": sum(counts.values()),
        "files": len(counts),
        "by_file": dict(counts.most_common()),
        "examples": dict(examples),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show", type=int, default=0)
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--write-baseline",
        action="store_true",
        help="Only after the count has gone DOWN, and never to record a rise.",
    )
    args = parser.parse_args()

    report = measure()
    held = 10**9
    if BASELINE.is_file():
        held = int(json.loads(BASELINE.read_text()).get("strings") or 0)

    if args.json:
        print(json.dumps({**report, "baseline": held}, indent=2))
        return 0

    print(
        f"{report['strings']} strings steering a model, across {report['files']} "
        f"modules that build prompts; baseline {held}"
    )
    for path, count in list(report["by_file"].items())[: args.show]:
        print(f"{count:4}  {path}")
        for item in report["examples"][path]:
            print(f"        :{item['line']} [{item['marker']}] {item['text']}")

    if args.write_baseline:
        if report["strings"] > held:
            print("refusing to raise the baseline", file=sys.stderr)
            return 1
        BASELINE.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "note": (
                        "Strings this runtime writes to steer a model, counted by "
                        "tools/prompt_steering_inventory.py. Only goes down. Each "
                        "one removed is replaced by a mechanism, not by a shorter "
                        "instruction."
                    ),
                    "strings": report["strings"],
                    "files": report["files"],
                },
                indent=2,
            )
            + "\n"
        )
        print(f"baseline written at {report['strings']}")
        return 0

    if report["strings"] > held:
        print(
            f"prompt steering rose by {report['strings'] - held}; a new instruction "
            "to a model is a mechanism that was not built",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
