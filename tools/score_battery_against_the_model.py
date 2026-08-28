"""Ask the resident model the same induction problems, and compare.

The native mechanism scores 111 of 120 on the frozen battery with no model in
the path. On its own that number says nothing: it needs the alternative. This
asks the model the same questions, through the runtime that is already up
rather than loading a second one beside it, and writes both scores down.

Deliberately fair to the model. It is given the same observations, told nothing
about shapes or families or anything the mechanism knows, and asked only for the
state that comes next. The answer is parsed leniently — any tuple-shaped run of
the right length anywhere in the reply counts — because the comparison is about
whether it can work the transformation out, not whether it formats well.

Sampled rather than exhaustive, because each turn takes minutes on this
hardware. The sample is stratified over shapes so no shape is over-represented,
the size is recorded with the result, and the same problems go to both sides.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.cognition.induction_battery import (  # noqa: E402
    Problem,
    generate_battery,
    teach_the_language,
)
from core.cognition.relation_language import RelationLanguage  # noqa: E402

_ENDPOINT = "http://localhost:8000/api/chat"


def stratified(problems: list[Problem], per_shape: int) -> list[Problem]:
    """The same problems for both sides, spread evenly over the shapes."""

    taken: dict[str, int] = {}
    chosen: list[Problem] = []
    for problem in problems:
        if taken.get(problem.shape, 0) >= per_shape:
            continue
        taken[problem.shape] = taken.get(problem.shape, 0) + 1
        chosen.append(problem)
    return chosen


def _asked(problem: Problem) -> str:
    """The problem as a person would put it, with nothing else in it."""

    lines = [
        "Work out the rule from these examples and apply it.",
        "",
    ]
    for shown in problem.shown:
        lines.append(f"{list(shown.before)}  becomes  {list(shown.after)}")
    lines.append("")
    lines.append(f"What does {list(problem.held_out.before)} become?")
    lines.append("Give only the resulting list.")
    return "\n".join(lines)


def _ask_the_runtime(text: str, *, timeout: float) -> str:
    payload = json.dumps({"message": text}).encode()
    request = urllib.request.Request(
        _ENDPOINT, data=payload, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as reply:
            body = json.loads(reply.read().decode())
    except (urllib.error.URLError, TimeoutError, ValueError, OSError) as exc:
        return f"<<no reply: {type(exc).__name__}>>"
    for key in ("response", "reply", "message", "text", "answer"):
        found = body.get(key)
        if isinstance(found, str) and found.strip():
            return found
    return str(body)[:2000]


_LISTISH = re.compile(r"[\[(]([^\[\]()]{1,400}?)[\])]")


def _read_the_answer(said: str, wanted: int) -> tuple | None:
    """Any bracketed run of the right length, read leniently.

    The comparison is about working the transformation out, not about
    formatting, so every bracketed run in the reply is tried and the last one of
    the right length wins — models often restate the question before answering.
    """

    found = None
    for hit in _LISTISH.finditer(said):
        inside = hit.group(1).strip()
        if not inside:
            continue
        try:
            parsed = ast.literal_eval(f"[{inside}]")
        except (SyntaxError, ValueError):
            parsed = [
                piece.strip().strip("'\"")
                for piece in inside.split(",")
                if piece.strip()
            ]
        if isinstance(parsed, list) and len(parsed) == wanted:
            found = tuple(parsed)
    return found


def _matches(got: tuple | None, wanted: tuple) -> bool:
    if got is None:
        return False
    if tuple(got) == tuple(wanted):
        return True
    # A model writing 3 for 3.0, or "alpha" for alpha, is right.
    return [str(item) for item in got] == [str(item) for item in wanted]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-shape", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument(
        "--out", default="config/induction_battery_comparison.json"
    )
    args = parser.parse_args()

    battery = generate_battery()
    sample = stratified(battery, args.per_shape)

    language = RelationLanguage()
    teach_the_language(battery, language=language)

    rows = []
    native_right = model_right = 0
    for index, problem in enumerate(sample, start=1):
        found = language.explain(list(problem.shown))
        try:
            native = (
                found is not None
                and tuple(found.apply(problem.held_out.before))
                == tuple(problem.held_out.after)
            )
        except Exception:  # noqa: BLE001 - a relation that throws did not answer
            native = False

        started = time.perf_counter()
        said = _ask_the_runtime(_asked(problem), timeout=args.timeout)
        took = time.perf_counter() - started
        answered = _read_the_answer(said, len(problem.held_out.after))
        model = _matches(answered, problem.held_out.after)

        native_right += int(native)
        model_right += int(model)
        rows.append(
            {
                "problem": problem.name,
                "shape": problem.shape,
                "native": native,
                "model": model,
                "model_said": said[:300],
                "seconds": round(took, 1),
            }
        )
        print(
            f"{index:3d}/{len(sample)}  native={'y' if native else 'n'}  "
            f"model={'y' if model else 'n'}  {took:6.1f}s  {problem.shape}",
            flush=True,
        )

    result = {
        "sampled": len(sample),
        "per_shape": args.per_shape,
        "native_solved": native_right,
        "model_solved": model_right,
        "rows": rows,
        "note": (
            "Same problems, same observations, both sides. The model is told "
            "nothing about shapes or families and its answer is parsed "
            "leniently. Sampled because each turn takes minutes on this "
            "hardware."
        ),
    }
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"\nnative {native_right}/{len(sample)}   model {model_right}/{len(sample)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
