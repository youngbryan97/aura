"""Searching the ways a program in Python could be put right, by running its cases with each: thousands of them, in one process.

Her own modules write the ways: the fix templates and the guards a fault's own exception asks for (core/self_modification/
fix_templates.py, checker_templates.py), the expressions the program writes elsewhere (ingredient_repair.py), the shapes that
are almost never meant (code_that_looks_wrong.py); with them here a number for one its rule gives, and what a branch could be
decided by where making it go the other way mends something. Each was made for her own source, where what says whether it
helped is a run of her tests, seconds or minutes long, and a few dozen are tried. A program's cases take milliseconds when they
are run in one process that is already there, so all of them are.

The process is under the system's sandbox, with no network and no writing but in a folder of its own, as every run of a made
program is. A way that does not end is stopped by the clock.
"""
from __future__ import annotations

import ast
import asyncio
import json
import logging
import sys
import tempfile
import time
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from core.making import her_own_mends as own
from core.making.many_ways import Evidence, Way, evidence_of
from core.making.searching import Found, Mend
from core.making.tools import Case, _run
from core.making.trying import Result
from core.rebuilding.her_build_writes import governed

logger = logging.getLogger("Making.SearchingPython")

__all__ = ["conditions_python", "made_python", "search_python", "ways_python"]

#: How long one run with a way may take, and the most ways there are.
A_RUN_S, IN_ALL = 0.5, 6000
#: How many of the lines the failing cases ran are looked at, the most suspicious first.
LINES = 30

_RUNNER = '''import io, json, os, shutil, signal, sys, time
given, told = json.load(open(sys.argv[1], encoding="utf-8")), open(sys.argv[2], "w", encoding="utf-8")
base, file, was = os.path.dirname(os.path.abspath(sys.argv[2])), given["file"], list(sys.path)
class Late(BaseException):
    pass
def late(*_):
    raise Late()
signal.signal(signal.SIGALRM, late)
def ran(code, n, case):
    here = os.path.join(base, "r%d" % n)
    if case["files"] or case["writes"] or not os.path.isdir(here):
        shutil.rmtree(here, ignore_errors=True)
        os.makedirs(here)
        for name, text in case["files"].items():
            path = os.path.join(here, name)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as held:
                held.write(text)
    out, err, left = io.TextIOWrapper(io.BytesIO(), encoding="utf-8", write_through=True), io.TextIOWrapper(io.BytesIO(), encoding="utf-8", write_through=True), 0
    kept = sys.argv, sys.stdin, sys.stdout, sys.stderr
    sys.argv, sys.stdin, sys.stdout, sys.stderr = [file, *case["args"]], io.TextIOWrapper(io.BytesIO(case["typed"].encode()), encoding="utf-8"), out, err
    sys.path[:] = [os.path.dirname(file), *was[1:]]
    os.chdir(here)
    broke = ""
    signal.setitimer(signal.ITIMER_REAL, given["a_run_s"])
    try:
        exec(code, {"__name__": "__main__", "__file__": file, "__builtins__": __builtins__})
    except SystemExit as gone:
        left = gone.code if isinstance(gone.code, int) else 0 if gone.code is None else 1
        if not isinstance(gone.code, (int, type(None))):
            err.write(str(gone.code))
    except Late:
        broke, left = "it does not end", 1
    except BaseException as exc:
        broke, left = type(exc).__name__, 1
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        sys.argv, sys.stdin, sys.stdout, sys.stderr = kept
        os.chdir(base)
    said = (out.buffer.getvalue() + err.buffer.getvalue()).decode("utf-8", "replace") if not out.closed and not err.closed else ""
    flat = " ".join(said.split())
    ok = not (case["leaves_with"] is not None and left != case["leaves_with"]) and not (broke and case["leaves_with"] in (None, 0)) and broke != "it does not end"
    ok = ok and all(" ".join(want.split()) in flat for want in case["prints"]) and not any(bad and bad in said for bad in case["not_prints"])
    for name, holds in case["writes"].items():
        path = os.path.join(here, name)
        ok = ok and os.path.isfile(path) and (not holds or holds in open(path, encoding="utf-8", errors="replace").read())
    return ok, said
def tried(source):
    try:
        code = compile(source, file, "exec")
    except (SyntaxError, ValueError):
        return None
    return lambda n: ran(code, n, given["cases"][n])
ends, as_it_is = time.monotonic() + given["seconds"], {}
run = tried(given["source"])
for n in given["passing"]:
    ok, said = run(n)
    if ok:
        as_it_is[n] = said
for i, (start, end, text) in enumerate(given["ways"]):
    if time.monotonic() > ends:
        break
    source = given["source"][:start] + text + given["source"][end:]
    run = tried(source)
    row = {"i": i, "mends": [], "breaks": 0, "other": 0, "held": 0}
    if run is not None:
        row["mends"] = [n for n in given["failing"] if run(n)[0]]
        if row["mends"]:
            for n in as_it_is:
                ok, said = run(n)
                row["held"] += 1
                row["breaks"] += not ok
                row["other"] += ok and said != as_it_is[n]
                if row["breaks"] > len(row["mends"]):
                    break
    if row["mends"] or i % 200 == 0:
        told.write(json.dumps(row) + chr(10))
told.write(json.dumps({"i": i + 1 if given["ways"] else 0, "end": True}) + chr(10))
told.close()
'''


def ways_python(code: str, ranked: Sequence[tuple[int, float]], evidence: Evidence, error: str = "") -> list[Way]:
    """The ways to change some Python at the lines that are suspected, each as the whole of it changed."""
    seen: set[str] = {code}
    out: list[Way] = []

    def take(found: Iterator[tuple[str, str, str]], most: int = IN_ALL) -> None:
        for n, (sort, says, changed) in enumerate(found):
            if len(out) >= IN_ALL or n >= most:
                return
            if changed not in seen:
                seen.add(changed)
                start, end, text = _where(code, changed)
                out.append(Way(len(out) + 1, 0, sort.split(":")[0], start, end, text, -1, says))

    take(own.missing_imports(code, "\n".join(evidence.broke)))
    take(own.of_its_shapes(code, ".py"), own.MOST * 4)
    take(_numbers(code, ranked, evidence))
    take(_passed_over(code, ranked))
    take(_kinds_and_signs(code, ranked))
    take(own.of_its_lines(code, list(ranked), error))
    return out


def _where(was: str, now: str) -> tuple[int, int, str]:
    """What is different between two texts, as one change."""
    a = 0
    most = min(len(was), len(now))
    while a < most and was[a] == now[a]:
        a += 1
    z = 0
    while z < most - a and was[-1 - z] == now[-1 - z]:
        z += 1
    return a, len(was) - z, now[a:len(now) - z]


def _span(code: str, node: Any) -> tuple[int, int]:
    from core.self_modification.fix_templates import _offsets
    from core.self_modification.fix_templates import _span as span

    return span(code, _offsets(code), node)


def _numbers(code: str, ranked: Sequence[tuple[int, float]], evidence: Evidence) -> Iterator[tuple[str, str, str]]:
    """A number for one that the rule gives, at the suspected lines."""
    lines = {line for line, _score in ranked}
    for node in ast.walk(ast.parse(code)):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool) and node.lineno in lines:
            start, end = _span(code, node)
            for other in evidence.numbers:
                if other != code[start:end]:
                    yield "number_from_the_rule", f"changed `{code[start:end]}` to `{other}`", code[:start] + other + code[end:]


_SIGNS = {"+=": ("-=", "="), "-=": ("+=", "="), "*=": ("/=", "+="), "/=": ("*=", "//="), "//=": ("/=",)}


def _kinds_and_signs(code: str, ranked: Sequence[tuple[int, float]]) -> Iterator[tuple[str, str, str]]:
    """What is added where it should be taken; and a value taken as a number, or as words, where it is the other."""
    lines = {line for line, _score in ranked}
    for node in ast.walk(ast.parse(code)):
        if getattr(node, "lineno", 0) not in lines:
            continue
        if isinstance(node, ast.AugAssign):
            between_start, between_end = _span(code, node.target)[1], _span(code, node.value)[0]
            sign = code[between_start:between_end].strip()
            for other in _SIGNS.get(sign, ()):
                yield "arithmetic", f"changed `{sign}` to `{other}`", code[:between_start] + f" {other} " + code[between_end:]
        elif isinstance(node, (ast.BinOp, ast.Compare)):
            for held in (node.left, *([node.right] if isinstance(node, ast.BinOp) else node.comparators)):
                if isinstance(held, (ast.Name, ast.Attribute, ast.Subscript, ast.Call)):
                    start, end = _span(code, held)
                    for kind in ("int", "float", "str", "len", "abs"):
                        yield "same_kind", f"changed `{code[start:end]}` to `{kind}({code[start:end]})`", f"{code[:start]}{kind}({code[start:end]}){code[end:]}"
                    if isinstance(held, ast.Call) and isinstance(held.func, ast.Name) and len(held.args) == 1 and held.func.id in {"int", "float", "str"}:
                        inside = _span(code, held.args[0])
                        yield "same_kind", f"took `{held.func.id}` from it", code[:start] + code[inside[0]:inside[1]] + code[end:]


def _passed_over(code: str, ranked: Sequence[tuple[int, float]]) -> Iterator[tuple[str, str, str]]:
    """What is gone through one by one with nothing passed over, passing over what is empty."""
    lines = {line for line, _score in ranked}
    for node in ast.walk(ast.parse(code)):
        if isinstance(node, ast.comprehension) and not node.ifs and isinstance(node.target, ast.Name) and node.iter.lineno in lines:
            _start, end = _span(code, node.iter)
            for test in (node.target.id, f"{node.target.id}.strip()"):
                yield "guard", f"passed over what is empty: `if {test}`", f"{code[:end]} if {test}{code[end:]}"
        elif isinstance(node, ast.For) and isinstance(node.target, ast.Name) and node.lineno in lines and node.body:
            start, _end = _span(code, node.body[0])
            indent = " " * node.body[0].col_offset
            for test in (f"not {node.target.id}", f"not {node.target.id}.strip()"):
                yield "guard", f"passed over what is empty: `if {test}: continue`", f"{code[:start]}if {test}:\n{indent}    continue\n{indent}{code[start:]}"


def _branches(tree: ast.AST, lines: set[int]) -> Iterator[Any]:
    for node in ast.walk(tree):
        test = getattr(node, "test", None) if isinstance(node, (ast.If, ast.While, ast.IfExp)) else None
        if test is not None and test.lineno in lines:
            yield test
        for held in getattr(node, "ifs", ()) if isinstance(node, ast.comprehension) else ():
            if held.lineno in lines:
                yield held


def forced_python(code: str, ranked: Sequence[tuple[int, float]]) -> list[tuple[Any, Way]]:
    """Each branch at the suspected lines, made to go one way and the other."""
    out: list[tuple[Any, Way]] = []
    for test in _branches(ast.parse(code), {line for line, _score in ranked}):
        start, end = _span(code, test)
        out += [(test, Way(len(out) + n + 1, 0, "forced", start, end, text, -1, "")) for n, text in enumerate(("True", "False"))]
    return out


def conditions_python(code: str, tests: Sequence[Any], evidence: Evidence, *, most: int = 700) -> list[Way]:
    """What some branches could be decided by: each test of what is in reach there, alone and with what was there."""
    tree, out, seen = ast.parse(code), [], set()
    holders = [f for f in ast.walk(tree) if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef))]
    for test in tests:
        start, end = _span(code, test)
        was = code[start:end]
        holder = max((f for f in holders if f.lineno <= test.lineno <= (f.end_lineno or f.lineno)), key=lambda f: f.lineno, default=None)
        reach = _reach(holder if holder is not None else tree, test)
        atoms = [f"{p} > 0" for p in reach] + [f"{p} <= 0" for p in reach] + [f"not {p}" for p in reach] + list(reach)
        atoms += [f"len({p}) > 0" for p in reach if "." not in p] + [f"{p} is not None" for p in reach]
        atoms += [f"{p} {op} {n}" for p in reach for n in evidence.numbers[:8] for op in (">=", "<", ">", "<=", "==")]
        atoms += [f"{p} {op} {q}" for p in reach for q in reach if p != q for op in ("<", ">=", "==", "!=")]
        atoms += [f"{p} == {json.dumps(w)}" for p in reach for w in evidence.words[:6]]
        for atom in list(dict.fromkeys(atoms))[:most]:
            for sort, text in (("joined", f"({was}) and {atom}"), ("joined", f"({was}) or {atom}"), ("condition", atom)):
                if (start, text) not in seen:
                    seen.add((start, text))
                    out.append(Way(len(out) + 1, 0, sort, start, end, text, -1, f"changed `{was[:70]}` to `{text[:90]}`"))
    return out


def _reach(holder: Any, test: Any, most: int = 12) -> list[str]:
    """What can be asked about at a branch: the names the function is given and has made by then, and what of them it asks for."""
    names: dict[str, int] = {}
    for node in ast.walk(holder):
        line = getattr(node, "lineno", 0)
        if isinstance(node, ast.arg):
            names.setdefault(node.arg, line)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store) and line <= test.lineno:
            names.setdefault(node.id, line)
    asked = [ast.unparse(node) for node in ast.walk(holder) if isinstance(node, (ast.Attribute, ast.Subscript)) and isinstance(node.ctx, ast.Load)
             and isinstance(node.value, ast.Name) and node.value.id in names and len(ast.unparse(node)) < 30]
    return list(dict.fromkeys([*sorted(names, key=lambda n: abs(names[n] - test.lineno)), *asked]))[:most]


def made_python(code: str, way: Way) -> str:
    """The program with one change made."""
    return code[:way.start] + way.text + code[way.end:]


async def search_python(code: str, cases: Sequence[Case], trial_rules: Sequence[Result], lines: Sequence[set[int]], *, seconds: float = 40.0, above: int = 0,
                        pays: Mapping[str, float] | None = None) -> Found:
    """Search the ways some Python could be changed for those that make a failing case hold and break none that held. ``lines``
    is what each case ran; what stands above line ``above`` is not what is wrong."""
    return await asyncio.to_thread(_search, code, list(cases), list(trial_rules), list(lines), seconds, above, dict(pays or {}))


def _search(code: str, cases: list[Case], rules: list[Result], lines: list[set[int]], seconds: float, above: int, pays: dict[str, float]) -> Found:
    from core.making.many_ways import SORTS

    began = time.monotonic()
    ok = [r.ok for r in rules]
    failing = [i for i, held in enumerate(ok) if not held]
    if len(ok) != len(cases) or not failing:
        return Found()
    try:
        ast.parse(code)
    except SyntaxError:
        return Found()
    ranked = own.ran({str(i): lines[i] for i in failing}, {str(i): lines[i] for i in range(len(ok)) if ok[i]}, after=above)[:LINES]
    if not ranked:
        return Found()
    # How it broke is told by its lines of it, in the words of what it raised.
    told = [(cases[i].rule, json.dumps([cases[i].args, cases[i].typed[:200]]), rules[i].why) for i in failing]
    evidence = evidence_of(told)
    # The numbers of what fails first, and then of every rule: a branch that is missing is of a rule that still holds.
    numbers = [*evidence.numbers, *evidence_of([(case.rule, "", "") for case in cases]).numbers]
    evidence = Evidence(tuple(dict.fromkeys(numbers))[:16], tuple(dict.fromkeys(a for i in failing for a in cases[i].args if not a.startswith("/")))[:12], evidence.names,
                        tuple(rules[i].why for i in failing if "Error" in rules[i].why or "not defined" in rules[i].why))
    error = next((word for why in evidence.broke for word in why.replace("(", " ").replace(":", " ").split() if word.endswith(("Error", "Exception"))), "")
    chosen = sorted(ways_python(code, ranked, evidence, error), key=lambda w: -(SORTS.get(w.sort, 0.9) + pays.get(w.sort, 0.0)))
    passing = [i for i in range(len(cases)) if i not in failing]
    given = Found(of=len(chosen))
    by_rule: dict[str, list[int]] = {}
    for i, case in enumerate(cases):
        by_rule.setdefault(case.rule, []).append(i)
    given.twice = frozenset(rule for rule, its in by_rule.items() if len(its) > 1)

    def mended(some: Sequence[Way], until: float, against: list[int]) -> list[Mend]:
        rows = _ran_with(code, cases, some, failing, against, max(1.0, until - (time.monotonic() - began)))
        given.tried += next((row["i"] for row in rows if row.get("end")), 0)
        out = []
        for row in rows:
            if row.get("end") or not row["mends"]:
                continue
            # A rule is had where every case of it that failed holds.
            had = tuple(rule for rule, its in by_rule.items() if (bad := [i for i in its if i in failing]) and set(bad) <= set(row["mends"]))
            if had:
                out.append(Mend(some[row["i"]], had, ("something that held",) * int(row["breaks"]), int(row["other"]), int(row["held"])))
        return out

    given.mends = mended(chosen, seconds * 0.6, passing)
    if not any(not m.breaks and len(m.mends) == len({cases[i].rule for i in failing}) for m in given.mends):
        held = forced_python(code, ranked)
        asked = mended([way for _test, way in held], seconds * 0.7, passing)
        turned = {id(m.way): m for m in asked}
        spots = [test for test, way in held if id(way) in turned]
        worked = conditions_python(code, list({(t.lineno, t.col_offset): t for t in spots}.values())[:8], evidence)
        given.of += len(worked)
        given.mends += mended(worked, seconds, passing)
        chosen = [*chosen, *worked]
    given.seconds = time.monotonic() - began
    order = {(way.start, way.end, way.text): n for n, way in reversed(list(enumerate(chosen)))}
    given.mends.sort(key=lambda m: (len(m.breaks) - len(m.mends), m.other, order.get((m.way.start, m.way.end, m.way.text), 0)))
    logger.info("A tool: %s, %d mend something.", given.says(), len(given.mends))
    return given


def _ran_with(code: str, cases: list[Case], some: Sequence[Way], failing: list[int], passing: list[int], seconds: float) -> list[dict[str, Any]]:
    """What running the cases with each of some ways shows, as the process that ran them tells it."""
    if not some:
        return []
    with tempfile.TemporaryDirectory(prefix="made-") as made:
        folder = Path(made).resolve()
        main, runner, given, told = folder / "main.py", folder / "runner.py", folder / "given.json", folder / "told.jsonl"
        governed("write_text", main, code, source="making.searching_python")
        governed("write_text", runner, _RUNNER, source="making.searching_python")
        governed("write_text", given, json.dumps({"source": code, "file": str(main), "a_run_s": A_RUN_S, "seconds": seconds, "failing": failing, "passing": passing,
                                                  "cases": [c.model_dump() for c in cases], "ways": [[w.start, w.end, w.text] for w in some]}), source="making.searching_python")
        try:
            _run([sys.executable, "-I", str(runner), str(given), str(told)], folder, timeout=seconds + 20.0)
        except Exception as exc:  # noqa: BLE001 - a way that takes the process with it ends the search, with what it had found
            logger.info("A tool: the search broke off (%s).", str(exc)[:200])
        rows = []
        for line in told.read_text(encoding="utf-8").splitlines() if told.is_file() else ():
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue  # not a failure: the last line of a process that was stopped is cut off
        return rows
