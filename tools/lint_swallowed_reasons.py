#!/usr/bin/env python3
"""Every place a failure is caught and its reason thrown away.

Six of these were found one at a time in a single day, each only visible once
the one before it was fixed: a contained exception with no raise site hid an
AttributeError; a refusal reported as its exception type hid a bypass reason;
nine checks sharing one except hid a receipt; one message for nine conditions
hid which field; and the field named the cause. None of them was hard to fix.
All of them were hard to FIND, and only because the code knew and did not say.

That is a mechanical property, so it can be found mechanically rather than one
failure at a time.

What counts, and what does not. A first pass flagged every handler that did
not mention its exception and found four thousand seven hundred, which is not
a list anybody can act on and mostly is not the problem: `except KeyError:
continue` inside a loop over optional data loses nothing anybody wanted.

What hurt was narrower and is worth naming exactly — a handler that hands a
FAILURE back to somebody while dropping the reason it has in its hand:

    a silent failure     `except X: pass` where the function goes on to
                         return something a caller will read as success
    a failure value      the handler returns False, None, or a dict saying
                         ok=False or carrying an error, and the exception
                         appears nowhere in it
    a failure flag       the handler sets a name to False or "" and nothing
                         else, so a check downstream reports a verdict it
                         cannot explain

A handler is NOT swallowing when it logs the exception, records a degradation,
re-raises, or puts the exception into what it hands back. Those all carry the
reason somewhere a person can reach it.

Some handlers return None as an ANSWER rather than as a failure — asking
whether a cell holds a number, and being told it does not, is not something
going wrong. Those say so with a comment beginning "not a failure:", which
costs a sentence and makes the claim reviewable. Silence is never the way to
claim it.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

#: Where a reason is allowed to go: a log, the degradation sink, a re-raise, a
#: returned value that mentions it, or a note on the exception already on its
#: way up. That last one is the right carrier for a cleanup step that fails
#: while unwinding another failure — a rollback that will not roll back must
#: not replace the error that describes what actually went wrong, and a log
#: line separates the two.
_CARRIERS = (
    "log", "record_degradation", "raise", "logger", "print", "warn", "add_note"
)


class _Swallowed(ast.NodeVisitor):
    def __init__(self, path: Path, lines: list[str]) -> None:
        self.path = path
        self._lines = lines
        self.found: list[tuple[int, str, str]] = []

    def visit_Try(self, node: ast.Try) -> None:
        for handler in node.handlers:
            self._weigh(handler)
        self.generic_visit(node)

    def _weigh(self, handler: ast.ExceptHandler) -> None:
        body = handler.body
        if self._claims_it_is_an_answer(handler):
            return
        said = ast.unparse(ast.Module(body=body, type_ignores=[]))
        # Anything that carries the reason out clears the handler.
        if any(one in said for one in _CARRIERS):
            return
        if handler.name and handler.name in _read_names(body):
            return
        kind = ""
        if _returns_a_failure(body):
            kind = "a failure value"
        elif _sets_a_failure_flag(body):
            kind = "a failure flag"
        elif len(body) == 1 and isinstance(body[0], ast.Pass):
            kind = "a silent failure"
        if kind:
            self.found.append((handler.lineno, kind, said.strip().splitlines()[0][:60]))


    def _claims_it_is_an_answer(self, handler: ast.ExceptHandler) -> bool:
        """Whether the handler says, in words, that this is not a failure.

        The span runs from the comment block immediately ABOVE the ``except``
        line through the END of the first statement, because those are the
        two places a person writes the sentence and the gate read neither.
        Eighteen handlers in one file explained themselves on the ``pass``
        line itself; others explain themselves in a comment above the
        ``except``, which is where it goes when the handler is one line. The
        claim was there and reviewable in both, and the gate was looking at
        the one line between them.
        """
        if handler.body:
            first = handler.body[0].end_lineno or handler.body[0].lineno
        else:
            first = handler.lineno
        # Walk back over an unbroken run of comment lines. A blank line or
        # any code ends it, so this cannot reach a comment about something
        # else further up.
        begin = handler.lineno - 1
        while begin > 0 and self._lines[begin - 1].lstrip().startswith("#"):
            begin -= 1
        for line in self._lines[begin : first]:
            # Case-insensitive. The marker opens a sentence, so a writer
            # capitalises it, and a gate that matches only the lowercase form
            # reads eighteen explained handlers as eighteen silent ones.
            if "not a failure:" in line.lower():
                return True
        return False


def _read_names(body: list[ast.stmt]) -> set[str]:
    names: set[str] = set()
    for statement in body:
        for node in ast.walk(statement):
            if isinstance(node, ast.Name):
                names.add(node.id)
    return names


def _returns_a_failure(body: list[ast.stmt]) -> bool:
    """Whether the handler hands back something a caller reads as failure."""
    for statement in body:
        if not isinstance(statement, ast.Return):
            continue
        value = statement.value
        if value is None:
            return True
        if isinstance(value, ast.Constant) and value.value in (False, None):
            return True
        if isinstance(value, ast.Dict):
            said = ast.unparse(value)
            if "'ok': False" in said or '"ok": False' in said or "error" in said:
                return True
    return False


def _sets_a_failure_flag(body: list[ast.stmt]) -> bool:
    """Whether the whole handler is `something = False` and nothing else."""
    if not body or len(body) > 2:
        return False
    for statement in body:
        if not isinstance(statement, ast.Assign):
            return False
        value = statement.value
        if not (
            isinstance(value, ast.Constant) and value.value in (False, None, "")
        ):
            return False
    return True


def look(paths: list[Path]) -> list[dict[str, object]]:
    found: list[dict[str, object]] = []
    for path in paths:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue
        walker = _Swallowed(path, path.read_text(encoding="utf-8").splitlines())
        walker.visit(tree)
        for line, kind, said in walker.found:
            found.append(
                {"file": str(path), "line": line, "kind": kind, "handler": said}
            )
    return found


def sealed_by_a_qualification() -> set[Path]:
    """Files a measurement has bound by byte hash, which must not be edited.

    Two ratchets, pulling opposite ways. The bounded-WOW surface seals
    fifteen source files by SHA-256 of their BYTES, because a qualification
    run measured what those exact files do: the seal is the claim that the
    thing measured is the thing serving. A comment changes the bytes, so
    adding one switches the surface off until somebody re-runs 120 tasks.

    That has happened twice. On 2026-08-17 four bound files moved under
    ordinary development and a signal established at p=5.7e-14 was dark for
    two days before anyone noticed. On 2026-09-21 four annotation lines
    written for THIS gate did it again, and the smoke suite caught it.

    So this gate stops asking. A sealed file's handlers are real and still
    worth fixing, but they can only be fixed in the same change that re-runs
    the qualification, and a gate that demands an edit another gate forbids
    teaches people to edit the seal instead.
    """
    try:
        sys.path.insert(0, str(Path.cwd()))
        from core.brain.llm.semantic_neural_serving import (
            ACTIVATION_SOURCE_FILES,
            REPO_ROOT,
        )
    except Exception as exc:  # noqa: BLE001 - any import failure means no seal here
        print(f"   (no bound-source seal readable: {type(exc).__name__}: {exc})")
        return set()
    return {
        (Path(REPO_ROOT) / relative).resolve()
        for relative in ACTIVATION_SOURCE_FILES
        if str(relative).endswith(".py")
    }


def main() -> int:
    ask = argparse.ArgumentParser(description=__doc__)
    ask.add_argument("paths", nargs="*", default=["core", "interface", "skills"])
    ask.add_argument("--baseline", default="config/swallowed_reasons_baseline.json")
    ask.add_argument("--write-baseline", action="store_true")
    ask.add_argument("--show", type=int, default=15)
    args = ask.parse_args()

    files: list[Path] = []
    for one in args.paths:
        here = Path(one)
        files.extend(sorted(here.rglob("*.py")) if here.is_dir() else [here])
    sealed = sealed_by_a_qualification()
    skipped = [one for one in files if one.resolve() in sealed]
    if skipped:
        files = [one for one in files if one.resolve() not in sealed]
        # Count what is behind the seal rather than only how many files hold
        # it. "13 files skipped" reads like nothing is owed; the number of
        # handlers is the debt, and it belongs on every run.
        held = len(look(skipped))
        print(
            f"{len(skipped)} file(s) skipped: sealed by a qualification and "
            f"editable only with it, holding {held} handler(s) that lose the "
            "reason until it is re-run"
        )
    found = look(files)

    by_kind: dict[str, int] = {}
    for one in found:
        by_kind[str(one["kind"])] = by_kind.get(str(one["kind"]), 0) + 1
    print(f"{len(found)} handler(s) that lose the reason, across {len(files)} file(s)")
    for kind, count in sorted(by_kind.items(), key=lambda pair: -pair[1]):
        print(f"   {count:5d}  {kind}")
    for one in found[: args.show]:
        print(f"   {one['file']}:{one['line']}  [{one['kind']}]  {one['handler']}")

    where = Path(args.baseline)
    if args.write_baseline:
        where.parent.mkdir(parents=True, exist_ok=True)
        where.write_text(json.dumps({"total": len(found)}, indent=1) + "\n")
        print(f"baseline written: {len(found)}")
        return 0
    try:
        allowed = int(json.loads(where.read_text()).get("total", 0))
    except (OSError, ValueError):
        print("no baseline yet; run with --write-baseline")
        return 0
    if len(found) > allowed:
        print(f"FAIL: {len(found)} > baseline {allowed}. The ratchet only goes down.")
        return 1
    print(f"OK: {len(found)} <= baseline {allowed}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
