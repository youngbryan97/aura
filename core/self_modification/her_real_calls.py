"""Her real calls of the functions she may change, tried on both sides of every change to them (Master Todo F16).

The outside check judges a change on the calls its tests make and on inputs near them. Rounds five and six of its red
team left changes on probation that a benchmark asking only valid questions does not notice: a guard deleted, bad input
turned into 0. What she is asked while she runs is evidence no test author chose. This keeps a sample of it for each
function she may change next: the functions of the faults she heard, of a change of hers on probation, and of a draft
of hers waiting to land.

Each call is written back as source, as a fault's call is (``core/runtime/a_call_that_failed.py``), and kept in her
state, never in her source: her source is public and these calls carry what people said to her. To judge a change, a
scratch clone of her source gets one local commit under the change, a test for each of her calls, and the change
replayed on top. The outside check (``outside_bar/verify_change.py``, unchanged) then runs those tests on both sides as
it runs any test that reaches the changed code, and a call of hers that comes to something else counts against the
change. It compares only values a call returned, so before that each call is run twice on the code before the change,
and a refusal both runs agree on is pinned in its test: a refusal gone from a real call breaks a test. Nothing in the
clone is pushed.

The sample is a reservoir (Vitter's Algorithm R, 1985): every call a function gets has the same chance of being kept,
so its first minutes do not stand for its whole life. Only a call that is kept is written back, so a function called
often costs a count and a random draw.

    python -m core.self_modification.her_real_calls judge --repo PATH --commit SHA
"""
from __future__ import annotations

import argparse
import ast
import dataclasses
import hashlib
import inspect
import json
import logging
import os
import random
import re
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from types import CodeType, FrameType, ModuleType
from typing import Any

from core.runtime.a_call_that_failed import FailedCall, a_call_written_back, path_in_her_source
from core.runtime.lockdep import checked_lock

logger = logging.getLogger("Aura.SelfModification.RealCalls")

#: Calls kept for one function. The outside check tries 400 inputs for a function and gives the first half to its
#: commit's own tests (outside_bar/verify_change.py, NEAR_INPUTS); a sample larger than the other half is not tried.
KEPT_PER_FUNCTION = 200

#: Where the guard tests are written in a scratch clone.
GUARD_TESTS = Path("tests") / "her_real_calls"

#: As long as the outside check may take on one change: the bound her gates give one (her_goal_at_work.GATE_SECONDS).
_JUDGE_SECONDS = 2400.0

#: The monitoring slots tried, from the end the others start at: the connectome holds 5 and the module-activity watch
#: takes the first free of 1, 3 and 4.
_SLOTS = (4, 3, 1, 2)

_NAME = "aura.her_real_calls"
_LONE_GENERATOR = inspect.CO_GENERATOR | inspect.CO_ASYNC_GENERATOR


def where_her_real_calls_are_kept() -> Path:
    """Her state, beside the faults she heard: private, and still hers after a restart."""
    from core.runtime.state_ownership import state_root

    return state_root() / "her_real_calls" / "calls.json"


def _row(call: FailedCall, at: float) -> dict[str, Any]:
    row = dataclasses.asdict(call)
    for name in ("error_type", "error_message", "below", "line"):
        row.pop(name, None)
    return {**row, "at": at}


def _call_of(row: dict[str, Any]) -> FailedCall:
    return FailedCall(
        module=str(row["module"]), qualname=str(row["qualname"]), file=str(row["file"]), line=0, error_type="",
        error_message="", positional=tuple(str(part) for part in row.get("positional") or ()),
        positional_names=tuple(str(part) for part in row.get("positional_names") or ()),
        keywords=tuple((str(key), str(value)) for key, value in row.get("keywords") or ()),
        star_args=str(row.get("star_args") or ""), star_kwargs=str(row.get("star_kwargs") or ""),
        receiver_module=str(row.get("receiver_module") or ""), receiver_class=str(row.get("receiver_class") or ""),
        receiver_state=tuple((str(key), str(value)) for key, value in row.get("receiver_state") or ()),
    )


def _readable(row: Any) -> bool:
    """A kept row whose every value is source that evaluates as a plain value: a file another process could have
    written is read as data, never run."""
    try:
        call = _call_of(row)
        for source in (*call.positional, *(value for _, value in call.keywords), *(value for _, value in call.receiver_state),
                       *(part for part in (call.star_args, call.star_kwargs) if part)):
            ast.literal_eval(source)
    except (KeyError, TypeError, ValueError, SyntaxError, MemoryError, RecursionError):
        return False
    names = [call.qualname, call.module, call.receiver_module, call.receiver_class, *call.positional_names,
             *(key for key, _ in call.keywords), *(key for key, _ in call.receiver_state)]
    return all(part.isidentifier() for name in names if name for part in name.split("."))


def _resolved(module: str, qualname: str) -> Any:
    """The function ``module.qualname`` names in this process, unwrapped; None when its module is not loaded here.

    Only loaded modules: a function she does not run has no real calls to keep, and importing one to watch it would
    run its module's code for nothing."""
    holder: Any = sys.modules.get(module)
    if not isinstance(holder, ModuleType):
        return None
    for part in qualname.split("."):
        found = inspect.getattr_static(holder, part, None)
        if found is None:
            return None
        holder = found
    if isinstance(holder, (staticmethod, classmethod)):
        holder = holder.__func__
    if isinstance(holder, property):
        holder = holder.fget
    return inspect.unwrap(holder) if callable(holder) else None


class HerRealCalls:
    """The kept sample of each watched function's real calls, and the watch that keeps it."""

    def __init__(self, kept: int = KEPT_PER_FUNCTION, draw: Callable[[], float] = random.random) -> None:
        self.kept = kept
        self._draw = draw
        self._lock = checked_lock("her_real_calls")
        self._tool: int | None = None
        self._watched: dict[int, str] = {}
        self._codes: dict[str, CodeType] = {}
        self._functions: dict[str, dict[str, Any]] = {}
        self._changes = 0
        self._kept_at = 0

    # ── the watch ──────────────────────────────────────────────────────

    def _claim_a_slot(self) -> bool:
        if self._tool is not None:
            return True
        monitoring = sys.monitoring
        for tool in _SLOTS:
            if monitoring.get_tool(tool) is None:
                monitoring.use_tool_id(tool, _NAME)
                monitoring.register_callback(tool, monitoring.events.PY_START, self._started)
                self._tool = tool
                return True
        return False

    def watch(self, functions: Iterable[tuple[str, str]]) -> list[str]:
        """Watch each (module, qualified name) her process has loaded; the keys now watched. A function reloaded since
        it was first watched is watched in its new code, and its old code is let go."""
        wanted: list[tuple[str, CodeType, bool]] = []
        for module, qualname in functions:
            target = _resolved(module, qualname)
            code = getattr(target, "__code__", None)
            if not isinstance(code, CodeType) or code.co_flags & _LONE_GENERATOR or "<" in code.co_qualname:
                continue
            path = path_in_her_source(code.co_filename)
            if path:
                wanted.append((f"{path}::{code.co_qualname}", code, bool(code.co_flags & inspect.CO_COROUTINE)))
        if not wanted or not self._claim_a_slot():
            return sorted(self._codes)
        tool = self._tool
        with self._lock:
            for key, code, awaited in wanted:
                old = self._codes.get(key)
                if old is code:
                    continue
                if old is not None:
                    self._watched.pop(id(old), None)
                    sys.monitoring.set_local_events(tool, old, 0)
                self._codes[key] = code
                self._watched[id(code)] = key
                entry = self._functions.setdefault(key, {"seen": 0, "calls": [], "why_not": {}})
                entry["awaited"] = awaited
                sys.monitoring.set_local_events(tool, code, sys.monitoring.events.PY_START)
            return sorted(self._codes)

    def stop(self) -> None:
        """Let every watched function go, and the monitoring slot with them."""
        tool, self._tool = self._tool, None
        if tool is None:
            return
        with self._lock:
            for code in self._codes.values():
                sys.monitoring.set_local_events(tool, code, 0)
            self._codes.clear()
            self._watched.clear()
        sys.monitoring.register_callback(tool, sys.monitoring.events.PY_START, None)
        sys.monitoring.free_tool_id(tool)

    def _started(self, code: CodeType, _offset: int) -> Any:
        try:
            self._keep(code, sys._getframe(1))
        except Exception as exc:  # noqa: BLE001 - watching a call must never change what the call does
            logger.debug("a real call was not kept: %s: %s", type(exc).__name__, exc)
        return None

    def _keep(self, code: CodeType, frame: FrameType) -> None:
        key = self._watched.get(id(code))
        if key is None:
            return
        with self._lock:
            entry = self._functions[key]
            entry["seen"] += 1
            seen = entry["seen"]
            slot = seen - 1 if seen <= self.kept else int(self._draw() * seen)
        if slot >= self.kept:
            return
        call = a_call_written_back(frame)
        with self._lock:
            entry = self._functions[key]
            if isinstance(call, str):
                why = entry["why_not"]
                why[call] = why.get(call, 0) + 1
                return
            row, calls = _row(call, time.time()), entry["calls"]
            if slot < len(calls):
                calls[slot] = row
            else:
                calls.append(row)
            self._changes += 1

    # ── what is kept ───────────────────────────────────────────────────

    def corpus(self) -> dict[str, Any]:
        with self._lock:
            return {"functions": json.loads(json.dumps(self._functions))}

    def take_up(self, corpus: Any) -> int:
        """A corpus kept before this process, continued: each function's count and sample. How many functions."""
        functions = corpus.get("functions") if isinstance(corpus, dict) else None
        if not isinstance(functions, dict):
            return 0
        taken = 0
        with self._lock:
            for key, entry in functions.items():
                if key in self._functions or not isinstance(entry, dict):
                    continue
                calls = [row for row in entry.get("calls") or () if isinstance(row, dict) and _readable(row)]
                self._functions[str(key)] = {"seen": max(int(entry.get("seen") or 0), len(calls)),
                                             "calls": calls[: self.kept], "why_not": dict(entry.get("why_not") or {}),
                                             "awaited": bool(entry.get("awaited"))}
                taken += 1
            self._kept_at = self._changes
        return taken

    def unkept_changes(self) -> bool:
        with self._lock:
            return self._changes != self._kept_at

    def mark_kept(self, changes: int) -> None:
        with self._lock:
            self._kept_at = max(self._kept_at, changes)

    def changes(self) -> int:
        with self._lock:
            return self._changes


_THE_SAMPLE: list[HerRealCalls] = []
_SAMPLE_LOCK = checked_lock("her_real_calls.sample")


def her_real_calls() -> HerRealCalls:
    """The one sample in this process, with what earlier runs kept taken up."""
    with _SAMPLE_LOCK:
        if not _THE_SAMPLE:
            sample = HerRealCalls()
            path = where_her_real_calls_are_kept()
            try:
                if path.exists():
                    sample.take_up(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, ValueError) as exc:
                from core.runtime.errors import record_degradation

                record_degradation("her_real_calls", exc, severity="warning",
                                   action=f"could not read {path}; her real calls are kept from this run on")
            _THE_SAMPLE.append(sample)
        return _THE_SAMPLE[0]


async def keep_her_real_calls() -> bool:
    """Write the sample to her state when a call was kept since the last write."""
    from core.governance_context import local_internal_governed_scope
    from core.runtime.file_write_gateway import get_file_write_gateway

    sample = her_real_calls()
    if not sample.unkept_changes():
        return False
    changes, corpus = sample.changes(), sample.corpus()
    with local_internal_governed_scope("her_real_calls.keep", domain="file_write", receipt_prefix="her-real-calls",
                                       constraints={"artifact": "aura.her_real_calls.v1", "operation": "write"}):
        await get_file_write_gateway().write_text_async(where_her_real_calls_are_kept(), json.dumps(corpus),
                                                        source="her_real_calls")
    sample.mark_kept(changes)
    return True


# ── what she may change ─────────────────────────────────────────────────


def _in_the_scratch(operation: str) -> Any:
    """The governed scope of what this module does in its scratch clone, from which nothing reaches her source."""
    from core.governance_context import local_internal_governed_scope

    return local_internal_governed_scope(f"her_real_calls.{operation}", domain="tool_execution",
                                         receipt_prefix="her-real-calls",
                                         constraints={"artifact": "aura.her_real_calls.v1", "operation": operation})


def _run(argv: Sequence[str], *, cwd: Path, timeout: float, operation: str, env: dict[str, str] | None = None,
         read_only: bool = False) -> subprocess.CompletedProcess:
    from core.runtime.subprocess_gateway import get_subprocess_gateway

    source = f"her_real_calls.{operation}"
    if read_only:
        return get_subprocess_gateway().run(list(argv), cwd=cwd, env=env, timeout=timeout, read_only=True, source=source,
                                            accelerator_capability="none")
    with _in_the_scratch(operation):
        return get_subprocess_gateway().run(list(argv), cwd=cwd, env=env, timeout=timeout, source=source,
                                            accelerator_capability="none")


def _write(path: Path, text: str, operation: str) -> None:
    from core.runtime.file_write_gateway import get_file_write_gateway

    with _in_the_scratch(operation):
        get_file_write_gateway().write_text(path, text, source=f"her_real_calls.{operation}", durable=False)


def _git(*args: str, cwd: Path, effect: str = "") -> subprocess.CompletedProcess:
    """git through the subprocess gateway: read-only on her source, or an effect in the scratch clone."""
    return _run(["git", "-c", "core.fsmonitor=false", *args], cwd=cwd, timeout=120, operation=f"git.{effect or 'read'}",
                read_only=not effect)


def functions_a_commit_changed(repo: Path, commit: str) -> list[tuple[str, str]]:
    """(file, qualified name) of each function whose lines ``commit`` changed, in her source and outside her tests."""
    shown = _git("diff", "-U0", "--no-color", f"{commit}^", commit, "--", "*.py", cwd=repo)
    if shown.returncode != 0:
        return []
    lines: dict[str, set[int]] = {}
    current = ""
    for text in shown.stdout.splitlines():
        if text.startswith("+++ "):
            current = text[6:] if text.startswith("+++ b/") else ""
        elif text.startswith("@@") and current:
            match = re.search(r"\+(\d+)(?:,(\d+))?", text)
            if match:
                start, count = int(match.group(1)), int(match.group(2) or 1)
                lines.setdefault(current, set()).update(range(start, start + max(count, 1)))
    found: list[tuple[str, str]] = []
    for name, changed in lines.items():
        if Path(name).parts[0] in ("tests", "test") or Path(name).name.startswith("test_"):
            continue
        source = _git("show", f"{commit}:{name}", cwd=repo)
        try:
            tree = ast.parse(source.stdout)
        except (SyntaxError, ValueError):
            continue
        found += [(name, qualname) for qualname in _enclosing(tree, changed)]
    return found


def _enclosing(tree: ast.AST, changed: set[int]) -> list[str]:
    """The innermost named function around each changed line; a function inside another is its own outer one's."""
    out: list[str] = []

    def visit(node: ast.AST, prefix: str, in_function: bool) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                visit(child, f"{prefix}{child.name}.", in_function)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if in_function:
                    continue
                first = min([child.lineno, *(item.lineno for item in child.decorator_list)])
                if changed & set(range(first, (child.end_lineno or child.lineno) + 1)):
                    out.append(f"{prefix}{child.name}")
                visit(child, f"{prefix}{child.name}.<locals>.", True)

    visit(tree, "", False)
    return list(dict.fromkeys(out))


def _dotted(file: str) -> str:
    return file.removesuffix(".py").removesuffix("/__init__").replace("/", ".")


def functions_she_may_change() -> list[tuple[str, str]]:
    """(module, qualified name) of the functions of the faults she heard, of her change on probation and of her drafts
    waiting to land: what she is about to change, or has just changed."""
    from core.agency import her_goal_at_work as goals
    from core.self_modification.faults_she_can_reproduce import faults_heard

    wanted = [(fault.call.module, fault.call.qualname) for fault in faults_heard()]
    root = goals.source_root()
    for record in goals.all_records():
        commit = str(record.offer.get("commit") or "")
        standing = record.status == goals.MERGED and record.offer.get("on_probation") and not record.offer.get("probation_over")
        if commit and (standing or record.status == goals.OFFERED):
            repo = root if standing else Path(record.workspace)
            wanted += [(_dotted(file), qualname) for file, qualname in functions_a_commit_changed(repo, commit)]
    return list(dict.fromkeys(wanted))


def watch_what_she_may_change() -> list[str]:
    """Her real calls kept for each function she may change next; the keys watched."""
    return her_real_calls().watch(functions_she_may_change())


# ── trying them on a change ─────────────────────────────────────────────


def guard_test_source(key: str, entry: dict[str, Any], faults: Iterable[str] = (),
                      outcomes: Sequence[str] | None = None) -> str:
    """A test for each kept call of one function, pinning what the code before the change did with it.

    Each call is a function of its own, ``_her_call_N``, so ``probe`` can learn what the code before the change did with
    it. ``outcomes`` holds what it learned, one per call: "raised <kind>" or "returned". A call that raised is pinned:
    its test asserts the same kind is still raised, or a narrower one, which code catching the old kind still catches.
    The check compares only values that were returned, so without the pin a refusal that vanished from a real call was
    only a doubt (round six, A10: a port range deleted, and every port taken).

    A call that returned is made and its value left to the check, which refuses a change to any value a passing test
    returned. A call that raised one of ``faults``, the faults her running heard from this function, is what a mend of it
    may change, so it is not pinned. Neither is a call whose outcome is not known.
    """
    calls = [_call_of(row) for row in entry.get("calls") or () if _readable(row)]
    if not calls:
        return ""
    first = calls[0]
    imports = sorted({f"from {call.receiver_module} import {call.receiver_class.split('.')[0]}" if call.receiver_class
                      else f"from {call.module} import {call.qualname.split('.')[0]}" for call in calls})
    awaited = bool(entry.get("awaited"))
    heard = set(faults)
    head = (f'"""Her real calls of {first.module}.{first.qualname} ({key}), kept from her own running.\n\n'
            "Written in a scratch clone by core/self_modification/her_real_calls.py for the outside check, and never\n"
            'committed: these calls carry what people said to her.\n"""\n'
            + ("import asyncio\n" if awaited else "") + "\n".join(imports) + "\n\n\n"
            + "def _named(kind):\n    return f\"{kind.__module__}.{kind.__qualname__}\"\n\n\n"
            + "def _raised_as(error, name):\n"
            + '    """Of that kind or a narrower one, so code that caught the old kind still catches it."""\n'
            + "    return any(_named(kind) == name for kind in type(error).__mro__)\n")
    parts = []
    for index, call in enumerate(calls):
        body = []
        if call.receiver_class:
            state = "{" + ", ".join(f"{name!r}: {value}" for name, value in call.receiver_state) + "}"
            body += [f"    receiver = {call.receiver_class}.__new__({call.receiver_class})",
                     f"    receiver.__dict__.update({state})"]
            made = call.call_source("receiver." + call.qualname.rsplit(".", 1)[-1])
        else:
            owner, _, rest = call.qualname.partition(".")
            made = call.call_source(f"{owner}.{rest}" if rest else owner)
        made = f"asyncio.run({made})" if awaited else made
        parts.append(f"\n\ndef _her_call_{index}():\n" + "\n".join([*body, f"    return {made}"]) + "\n")
        known = (outcomes[index] if outcomes is not None and index < len(outcomes) else "") or ""
        kind = known.removeprefix("raised ") if known.startswith("raised ") else ""
        if kind and kind not in heard:
            parts.append(f"\n\ndef test_her_call_{index}():\n"
                         f"    try:\n        _her_call_{index}()\n"
                         f"    except Exception as error:\n"
                         f"        assert _raised_as(error, {kind!r}), f\"it raised {kind}, and now {{_named(type(error))}}\"\n"
                         f"    else:\n        raise AssertionError({f'it raised {kind}, and now returns'!r})\n")
        else:
            parts.append(f"\n\ndef test_her_call_{index}():\n"
                         f"    try:\n        _her_call_{index}()\n"
                         f"    except Exception:  # {'a fault her running heard' if kind else 'what it came to'}"
                         f" is the outside check's to read\n        pass\n")
    return head + "".join(parts)


#: Run in the scratch clone: what the code there does with each ``_her_call_N`` of each guard file, written as JSON.
_PROBE = """import importlib.util, json, sys


def named(kind):
    return f"{kind.__module__}.{kind.__qualname__}"


seen = {}
for number, path in enumerate(sys.argv[2:]):
    spec = importlib.util.spec_from_file_location(f"her_real_calls_probe_{number}", path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except BaseException:
        continue
    outcomes, index = [], 0
    while hasattr(module, f"_her_call_{index}"):
        try:
            getattr(module, f"_her_call_{index}")()
            outcomes.append("returned")
        except Exception as error:
            outcomes.append("raised " + named(type(error)))
        except BaseException:
            outcomes.append("")
        index += 1
    seen[path] = outcomes
with open(sys.argv[1], "w", encoding="utf-8") as out:
    json.dump(seen, out)
"""


def probe(clone: Path, guards: Sequence[Path], scratch: Path) -> dict[str, list[str]]:
    """What the code checked out in ``clone`` does with each call in ``guards``, as ``guard_test_source`` takes it.

    Asked twice, each time in a fresh process, as the check asks the code before a change twice: an outcome the two
    runs disagree on is left out, so the call is not pinned. A run that fails or passes its bound pins nothing."""
    script = scratch / "probe_her_real_calls.py"
    _write(script, _PROBE, "probe")
    env = {**os.environ, "PYTHONPATH": str(clone), "PYTHONDONTWRITEBYTECODE": "1", "VECLIB_MAXIMUM_THREADS": "1",
           "OMP_NUM_THREADS": "1", "AURA_LOG_DIR": str(scratch / "logs")}
    runs = []
    for number in range(2):
        out = scratch / f"probe-{number}.json"
        try:
            _run(["nice", "-n", "10", sys.executable, str(script), str(out), *map(str, guards)], cwd=clone, env=env,
                 timeout=_JUDGE_SECONDS, operation="probe")
            runs.append(json.loads(out.read_text(encoding="utf-8")))
        except (OSError, RuntimeError, ValueError, subprocess.SubprocessError):
            return {}
    first, again = runs
    agreed: dict[str, list[str]] = {}
    for path, outcomes in first.items():
        other = again.get(path) or []
        agreed[path] = [one if index < len(other) and other[index] == one else "" for index, one in enumerate(outcomes)]
    return agreed


def _guard_name(key: str) -> str:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:10]
    stem = re.sub(r"[^a-z0-9]+", "_", key.split("::")[-1].lower()).strip("_")[:60] or "call"
    return f"test_real_calls_{stem}_{digest}.py"


def calls_for(corpus: dict[str, Any], changed: Sequence[tuple[str, str]]) -> dict[str, dict[str, Any]]:
    """The kept entries of the changed functions, each with the calls in it that read as plain source; an entry left
    with none is left out."""
    functions = corpus.get("functions") if isinstance(corpus, dict) else None
    held: dict[str, dict[str, Any]] = {}
    for file, qualname in changed:
        entry = (functions or {}).get(f"{file}::{qualname}")
        calls = [row for row in (entry or {}).get("calls") or () if isinstance(row, dict) and _readable(row)]
        if calls:
            held[f"{file}::{qualname}"] = {**entry, "calls": calls}
    return held


def sample_digest(corpus: dict[str, Any], changed: Sequence[tuple[str, str]]) -> str:
    """What her calls of the changed functions are now, as one digest: the same digest, the same evidence."""
    held = calls_for(corpus, changed)
    if not held:
        return ""
    return hashlib.sha256(json.dumps({key: entry["calls"] for key, entry in sorted(held.items())},
                                     sort_keys=True).encode("utf-8")).hexdigest()


def faults_heard_by_function() -> dict[str, set[str]]:
    """The faults her running heard, by the function each was raised in, as "file::qualname": error types."""
    from core.self_modification.faults_she_can_reproduce import where_heard_faults_are_kept

    path = where_heard_faults_are_kept()
    try:
        rows = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    except (OSError, ValueError) as exc:
        logger.debug("the faults she heard could not be read (%s); every call of hers is held as it came", exc)
        return {}
    found: dict[str, set[str]] = {}
    for row in rows if isinstance(rows, list) else ():
        call = row.get("call") if isinstance(row, dict) else None
        if isinstance(call, dict) and call.get("file") and call.get("qualname") and call.get("error_type"):
            found.setdefault(f"{call['file']}::{call['qualname']}", set()).add(str(call["error_type"]))
    return found


def judged_on_her_real_calls(repo: Path, commit: str, corpus: dict[str, Any], *, verifier: Path | None = None,
                             run: Callable[[list[str], Path], tuple[int, str]] | None = None,
                             faults: dict[str, set[str]] | None = None) -> dict[str, Any]:
    """The outside check's verdict on ``commit`` replayed over tests of her real calls of what it changed; {} when no
    call of hers reached what it changed. ``pinned`` counts the calls whose refusal their test holds the change to."""
    changed = functions_a_commit_changed(repo, commit)
    held = calls_for(corpus, changed)
    if not held:
        return {}
    faults = faults_heard_by_function() if faults is None else faults
    verifier = verifier or Path(__file__).resolve().parents[2] / "outside_bar" / "verify_change.py"
    with tempfile.TemporaryDirectory(prefix="her-real-calls-") as scratch:
        # The clone carries the repository's own folder name, as the check's own checkouts do.
        clone = Path(scratch) / (Path(repo).resolve().name or "repository")
        # Shared: the clone reads her repository's objects in place, so a draft's commit that no branch names is there.
        if _git("clone", "-q", "--shared", "--no-checkout", str(Path(repo).resolve()), str(clone),
                cwd=Path(scratch), effect="clone").returncode != 0:
            return {"verdict": "", "could_not_be_run": "her source could not be cloned"}
        if _git("checkout", "-q", "--detach", f"{commit}^", cwd=clone, effect="checkout").returncode != 0:
            return {"verdict": "", "could_not_be_run": "the code before the change could not be checked out"}
        folder = clone / GUARD_TESTS
        from core.runtime.file_write_gateway import get_file_write_gateway

        with _in_the_scratch("guards"):
            get_file_write_gateway().ensure_directory(folder, source="her_real_calls.guards")
        written: dict[str, Path] = {}
        for key, entry in held.items():
            source = guard_test_source(key, entry, faults.get(key, ()))
            if source:
                written[key] = folder / _guard_name(key)
                _write(written[key], source, "guards")
        # What the code before the change did with each call, so that each refusal of a real call is pinned by its test.
        learned = probe(clone, list(written.values()), Path(scratch))
        pinned = 0
        for key, path in written.items():
            outcomes = learned.get(str(path)) or []
            heard = set(faults.get(key, ()))
            pinned += sum(one.startswith("raised ") and one.removeprefix("raised ") not in heard for one in outcomes)
            if outcomes:
                _write(path, guard_test_source(key, held[key], heard, outcomes), "guards")
        identity = ("-c", "user.name=her real calls", "-c", "user.email=her-real-calls@localhost")
        if _git("add", GUARD_TESTS.as_posix(), cwd=clone, effect="add").returncode != 0 \
                or _git(*identity, "commit", "-q", "-m", "her real calls", cwd=clone, effect="commit").returncode != 0 \
                or _git(*identity, "cherry-pick", "--allow-empty", commit, cwd=clone, effect="replay").returncode != 0:
            return {"verdict": "", "could_not_be_run": "the change does not replay over tests of her real calls"}
        argv = [sys.executable, str(verifier), "--repo", str(clone), "--commit", "HEAD"]
        code, out = (run or _run_the_check)(argv, clone)
    result, said = _the_verdict_in(out)
    calls = sum(len(entry.get("calls") or ()) for entry in held.values())
    return {"verdict": str(result.get("verdict") or ""), "exit": code, "calls": calls, "pinned": pinned,
            "functions": sorted(held), "result": result, "said": said[-1500:]}


def _the_verdict_in(out: str) -> tuple[dict[str, Any], str]:
    """The check's verdict object, which it prints indented over many lines, and what it said after it."""
    start = out.find("{")
    if start < 0:
        return {}, out.strip()
    try:
        result, end = json.JSONDecoder().raw_decode(out, start)
    except ValueError:
        return {}, out.strip()
    return (result if isinstance(result, dict) else {}), out[end:].strip()


def _run_the_check(argv: list[str], clone: Path) -> tuple[int, str]:
    env = {**os.environ, "AURA_LOG_DIR": tempfile.mkdtemp(prefix="her-real-calls-logs-"),
           "VECLIB_MAXIMUM_THREADS": "1", "OMP_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1"}
    try:
        done = _run(["nice", "-n", "10", *argv], cwd=clone, env=env, timeout=_JUDGE_SECONDS, operation="check")
    except subprocess.TimeoutExpired:
        return 2, json.dumps({"verdict": "", "could_not_be_run": "the outside check ran past its bound"})
    except (OSError, RuntimeError) as exc:
        return 2, json.dumps({"verdict": "", "could_not_be_run": f"the outside check could not start: {exc}"[:400]})
    return done.returncode, (done.stdout or "") + (done.stderr or "")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="action", required=True)
    judge = sub.add_parser("judge", help="the outside check's verdict on a commit, over tests of her real calls")
    judge.add_argument("--repo", type=Path, required=True)
    judge.add_argument("--commit", required=True)
    judge.add_argument("--result", type=Path, default=None,
                       help="write the verdict here as JSON: what her yes decides by, where the words are for people")
    args = parser.parse_args(argv)
    path = where_her_real_calls_are_kept()
    corpus = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    judged = judged_on_her_real_calls(args.repo, args.commit, corpus)
    if args.result is not None:
        verdict = str(judged.get("verdict") or "") if judged else "NOT_APPLICABLE"
        _write(args.result, json.dumps({"verdict": verdict, **{key: judged.get(key) for key in (
            "calls", "pinned", "functions", "could_not_be_run")}}), "result")
    if not judged:
        print("held: no real call of hers reached what it changed")
        return 0
    print(json.dumps({key: judged[key] for key in ("verdict", "calls", "functions")}))
    if judged["verdict"] == "REFUSED":
        print(f"refused on her real calls: {judged['said'][:400]}")
        return 1
    if not judged["verdict"]:
        print(f"could not be judged on her real calls: {judged.get('could_not_be_run') or judged['said'][:400]}")
        return 1
    if judged["verdict"] == "PROBATION":
        doubts = (judged.get("result") or {}).get("doubts") or ["a doubt the check could not settle"]
        print(f"on probation: on her real calls, {doubts[0][:300]}")
    return 0


__all__ = [
    "GUARD_TESTS",
    "HerRealCalls",
    "KEPT_PER_FUNCTION",
    "calls_for",
    "faults_heard_by_function",
    "functions_a_commit_changed",
    "functions_she_may_change",
    "guard_test_source",
    "her_real_calls",
    "judged_on_her_real_calls",
    "keep_her_real_calls",
    "sample_digest",
    "watch_what_she_may_change",
    "where_her_real_calls_are_kept",
]


if __name__ == "__main__":
    raise SystemExit(main())
