"""A loop that waits on a deadline reads the machine's clock.

During a run her modules read the experiment's monotonic (core/subject/clock.py):
it moves one step a frame, stands still while a turn waits, and rewinds with every
restore. A wait timed on it never ends while the thing it waits for is slow. Seed
97's reports run on 29 October waited for room to load her cortex on a fifteen-
second deadline it could not reach, and the run lost an arm. The rule was already
written in the clock's docstring; this holds every module to it.

A wait is a `while` that sleeps or awaits, ended by a deadline stamped in the same
function. A loop that only computes is a budget on her own work and stays on her
clock, because on the machine's it would end wherever the machine's speed put it.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORGANISM = ("core", "interface", "llm", "skills")
MACHINERY = (
    "core/fsw", "core/governance_context", "core/health", "core/observability", "core/persistence",
    "core/runtime", "core/subject", "core/utils", "core/verify",
)
WAITING = {"sleep", "wait", "wait_for", "join", "poll", "select", "acquire"}

#: Waits whose deadline arrives from a caller, so both sides move together or not
#: at all. Only shrinks.
STILL_ON_HER_CLOCK = {
    # The turn's ceiling is stamped by the thinking loop and read by four modules,
    # one of them on the machine's clock already: one change for all of them.
    ("core/brain/cognitive_engine.py", "_keep_the_cycle_open_while_it_is_working"),
    # A screen pursuit carries one deadline through every step it takes.
    ("core/skills/screen_pursuit_looking.py", "wait_for_a_screen_to_look_at"),
}


def _clock_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call) and not node.args and not node.keywords
        and isinstance(node.func, ast.Attribute) and node.func.attr in ("monotonic", "time")
        and isinstance(node.func.value, ast.Name) and node.func.value.id in ("time", "_time")
    )


def _is_stamp(node: ast.AST) -> bool:
    if _clock_call(node):
        return True
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
        return _is_stamp(node.left) or _is_stamp(node.right)
    return False


def _names(node: ast.AST) -> set[str]:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def _stamps(fn: ast.AST) -> set[str]:
    stamped: set[str] = set()
    for st in ast.walk(fn):
        if isinstance(st, ast.Assign):
            pairs = []
            for target in st.targets:
                if isinstance(target, ast.Tuple) and isinstance(st.value, ast.Tuple) and len(target.elts) == len(st.value.elts):
                    pairs += list(zip(target.elts, st.value.elts))
                else:
                    pairs.append((target, st.value))
            stamped |= {t.id for t, v in pairs if isinstance(t, ast.Name) and _is_stamp(v)}
        elif isinstance(st, ast.AnnAssign) and st.value is not None and isinstance(st.target, ast.Name) and _is_stamp(st.value):
            stamped.add(st.target.id)
    return stamped


def _waits(loop: ast.While) -> bool:
    for n in ast.walk(loop):
        if isinstance(n, ast.Await):
            return True
        if isinstance(n, ast.Call):
            f = n.func
            name = f.attr if isinstance(f, ast.Attribute) else (f.id if isinstance(f, ast.Name) else "")
            if name in WAITING:
                return True
    return False


def waits_on_her_clock(source: str) -> list[tuple[str, int]]:
    """(function, line) of each wait whose deadline reads `time.monotonic()` or `time.time()`."""
    found = []
    for fn in ast.walk(ast.parse(source)):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        stamped = _stamps(fn)
        clocked = {
            name for st in ast.walk(fn) if isinstance(st, ast.Assign) and any(_clock_call(c) for c in ast.walk(st.value))
            for t in st.targets for name in _names(t)
        } & stamped
        for loop in ast.walk(fn):
            if not isinstance(loop, ast.While) or not _waits(loop):
                continue
            tests = [loop.test] + [
                n.test for s in loop.body for n in ast.walk(s)
                if isinstance(n, ast.If) and any(isinstance(m, (ast.Break, ast.Return)) for b in n.body for m in ast.walk(b))
            ]
            for test in tests:
                reads = _names(test) & stamped
                if not reads:
                    continue
                if any(_clock_call(c) for c in ast.walk(test)) or reads & clocked:
                    found.append((fn.name, test.lineno))
    return found


def _her_modules():
    for top in ORGANISM:
        for path in sorted((ROOT / top).rglob("*.py")):
            rel = path.relative_to(ROOT).as_posix()
            if any(rel == m + ".py" or rel.startswith(m + "/") for m in MACHINERY):
                continue
            yield rel, path


def test_no_wait_in_her_modules_reads_her_clock():
    found = []
    for rel, path in _her_modules():
        try:
            source = path.read_text(encoding="utf-8")
            hits = waits_on_her_clock(source)
        except (SyntaxError, UnicodeDecodeError):
            continue
        found += [(rel, fn, line) for fn, line in hits if (rel, fn) not in STILL_ON_HER_CLOCK]
    assert not found, (
        "a wait timed by her clock never ends while what it waits for is slow; read "
        "core.runtime.wall_clock.monotonic_time (or wall_time) instead:\n"
        + "\n".join(f"  {rel}:{line} {fn}" for rel, fn, line in found)
    )


def test_the_list_of_waits_still_on_her_clock_only_shrinks():
    present = set()
    for rel, path in _her_modules():
        try:
            hits = waits_on_her_clock(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        present |= {(rel, fn) for fn, _ in hits}
    gone = STILL_ON_HER_CLOCK - present
    assert not gone, f"moved to the machine's clock, take them off the list: {sorted(gone)}"


def test_the_check_finds_the_wait_seed_97_lost_an_arm_to():
    source = '''
def spawn(blocked):
    deadline = time.monotonic() + 15
    while blocked() and time.monotonic() < deadline:
        time.sleep(1.5)
'''
    assert waits_on_her_clock(source) == [("spawn", 4)]


def test_a_deadline_through_a_remaining_time_is_a_wait_too():
    source = '''
async def drain(timeout):
    deadline = time.monotonic() + timeout
    while busy():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        await asyncio.sleep(min(remaining, 0.25))
'''
    assert waits_on_her_clock(source) == [("drain", 6)]


def test_the_machines_clock_and_a_loop_that_only_computes_are_left_alone():
    machine = '''
def spawn(blocked):
    deadline = monotonic_time() + 15
    while blocked() and monotonic_time() < deadline:
        time.sleep(1.5)
'''
    budget = '''
def search(budget_s):
    began = time.monotonic()
    while more():
        step()
        if time.monotonic() - began >= budget_s:
            break
'''
    assert waits_on_her_clock(machine) == []
    assert waits_on_her_clock(budget) == []
