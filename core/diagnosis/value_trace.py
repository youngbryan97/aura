"""Watch a program run and find where it contradicts itself.

`carried_state.py` reads the source for one defect class — state that outlives a
call. It found the invoice bug, and it is a catalogue: it finds what somebody
thought to look for, which is the shape this codebase keeps having to undo.

The literature on fault localisation does not work that way. Delta debugging
minimises the input that triggers a failure; spectrum-based localisation ranks
statements by how often they appear in failing runs; Daikon infers invariants
from observed values and reports where they break. What they share is that the
defect is DERIVED from observation rather than matched against a list, which is
also how a person debugs: hold what should happen beside what did, and localise
where they diverge.

This is the observation half. It records every call a project's own functions
make during a run — arguments in, value out — and reports contradictions:

* the same function, called with equal arguments, returning different values
* a function whose return grows across identical calls

Neither names a cause. Both are facts about the run, and either one is enough
to point at the line without knowing in advance what kind of bug it is. The
mutable default in the invoice project shows up as the first; so does a cache
that leaks, a module-level accumulator, an ordering dependence and a clock read.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["Contradiction", "TracedRun", "trace_run", "describe_contradictions"]

#: A traced run costs more than a plain one; this is still a chat turn.
_TRACE_TIMEOUT_S = 120.0

#: Enough calls to see a repeat, few enough to keep the record small.
_MAX_CALLS = 4000


@dataclass(frozen=True, slots=True)
class Contradiction:
    """One place the run disagreed with itself."""

    function: str
    file: str
    line: int
    kind: str
    first_call: str
    first_result: str
    later_call: str
    later_result: str

    def as_sentence(self) -> str:
        """What was observed, in one line, with no interpretation."""
        return (
            f"{self.function} was called twice with the same arguments and answered "
            f"differently: {self.first_call} gave {self.first_result}, then "
            f"{self.later_call} gave {self.later_result} ({self.file}:{self.line})."
        )


@dataclass(frozen=True, slots=True)
class TracedRun:
    """What running it under the tracer showed."""

    ran: str = ""
    output: str = ""
    exit_code: int = 0
    calls_seen: int = 0
    contradictions: tuple[Contradiction, ...] = field(default_factory=tuple)
    error: str = ""


#: The tracer, run inside the project's own interpreter.
#:
#: It lives here as source rather than as a module the project would have to
#: import, because the project is somebody else's and nothing may be installed
#: into it.
_TRACER = '''
import json, sys, runpy, os
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve()
TARGET = sys.argv[2]
RECORD = Path(sys.argv[3])
MAX = int(sys.argv[4])

seen = []
pending = {}

# A comprehension or a lambda is not something a caller invokes by name, and a
# generator frame returns None after yielding, which reads as a contradiction
# every time.
SYNTHETIC = {"<genexpr>", "<listcomp>", "<dictcomp>", "<setcomp>", "<lambda>", "<module>"}


def _shape(value, depth=0):
    if depth > 2:
        return "..."
    if isinstance(value, (str, int, float, bool, type(None))):
        text = repr(value)
        return text if len(text) <= 80 else text[:77] + "..."
    if isinstance(value, (list, tuple, set)):
        inner = [_shape(item, depth + 1) for item in list(value)[:6]]
        more = "" if len(value) <= 6 else ", +%d" % (len(value) - 6)
        return "%s[%d](%s%s)" % (type(value).__name__, len(value), ", ".join(inner), more)
    if isinstance(value, dict):
        inner = ["%s: %s" % (_shape(k, depth + 1), _shape(v, depth + 1))
                 for k, v in list(value.items())[:6]]
        more = "" if len(value) <= 6 else ", +%d" % (len(value) - 6)
        return "dict[%d](%s%s)" % (len(value), ", ".join(inner), more)
    return "<%s>" % type(value).__name__


def _ours(code):
    name = code.co_filename
    if not name or name.startswith("<"):
        return False
    try:
        path = Path(name).resolve()
        return path.is_file() and path.is_relative_to(ROOT)
    except (ValueError, OSError):
        return False


def _defaulted(frame, code):
    """Parameter names still holding the function's own default object.

    A caller that leaves an argument out is not asking a different question, so
    those parameters are not part of what was asked. Identity is the test: the
    default object itself, not something equal to it.
    """
    function = frame.f_globals.get(code.co_name)
    defaults = getattr(function, "__defaults__", None) or ()
    if not defaults:
        return set()
    names = code.co_varnames[: code.co_argcount]
    tail = names[len(names) - len(defaults):]
    holding = set()
    for name, default in zip(tail, defaults):
        try:
            if frame.f_locals.get(name) is default:
                holding.add(name)
        except (TypeError, ValueError):
            continue
    return holding


def _trace(frame, event, arg):
    if len(seen) >= MAX:
        return None
    code = frame.f_code
    if event == "call":
        if not _ours(code) or code.co_name in SYNTHETIC:
            return None
        try:
            names = code.co_varnames[: code.co_argcount]
            skip = _defaulted(frame, code)
            asked = {n: _shape(frame.f_locals.get(n)) for n in names if n not in skip}
        except (AttributeError, TypeError, ValueError):
            asked = {}
        pending[id(frame)] = asked
        return _trace
    if event == "return" and _ours(code) and code.co_name not in SYNTHETIC:
        asked = pending.pop(id(frame), None)
        if asked is None:
            return _trace
        seen.append(
            {
                "function": code.co_name,
                "file": os.path.relpath(code.co_filename, ROOT),
                "line": code.co_firstlineno,
                "args": asked,
                "result": _shape(arg),
            }
        )
    return _trace


sys.settrace(_trace)
try:
    runpy.run_path(str(ROOT / TARGET), run_name="__main__")
finally:
    sys.settrace(None)
    RECORD.write_text(json.dumps(seen), encoding="utf-8")
'''


def _contradictions_in(calls: list[dict]) -> tuple[Contradiction, ...]:
    """Every place the same call answered differently."""
    by_signature: dict[tuple[str, str], list[dict]] = {}
    for call in calls:
        key = (str(call.get("function")), json.dumps(call.get("args"), sort_keys=True))
        by_signature.setdefault(key, []).append(call)

    found: list[Contradiction] = []
    for (function, arguments), records in by_signature.items():
        results = {str(record.get("result")) for record in records}
        if len(results) < 2:
            continue
        first, later = records[0], records[-1]
        found.append(
            Contradiction(
                function=function,
                file=str(first.get("file") or ""),
                line=int(first.get("line") or 0),
                kind="same arguments, different answer",
                first_call=f"{function}({_readable(arguments)})",
                first_result=str(first.get("result")),
                later_call=f"{function}({_readable(arguments)})",
                later_result=str(later.get("result")),
            )
        )
    return tuple(found)


def _readable(arguments: str) -> str:
    """The argument map as a call would be written."""
    try:
        loaded = json.loads(arguments)
    except (TypeError, ValueError):
        return arguments
    return ", ".join(f"{name}={value}" for name, value in loaded.items())


def trace_run(root: str | Path, entry: str) -> TracedRun:
    """Run one entry point under the tracer and report what disagreed."""
    from core.governance_context import local_internal_governed_scope
    from core.runtime.subprocess_gateway import get_subprocess_gateway

    base = Path(str(root)).expanduser()
    tracer = base / "_aura_value_trace.py"
    record = base / "_aura_value_trace.json"
    try:
        tracer.write_text(_TRACER, encoding="utf-8")
    except OSError as exc:
        return TracedRun(ran=entry, error=f"{type(exc).__name__}: {exc}")
    try:
        with local_internal_governed_scope("diagnosis.value_trace"):
            done = get_subprocess_gateway().run(
                [sys.executable, str(tracer), str(base), entry, str(record), str(_MAX_CALLS)],
                cwd=str(base),
                capture_output=True,
                text=True,
                timeout=_TRACE_TIMEOUT_S,
                read_only=False,
                check=False,
                source="diagnosis.value_trace",
                accelerator_capability="none",
            )
        calls = json.loads(record.read_text(encoding="utf-8")) if record.exists() else []
    except (OSError, subprocess.SubprocessError, RuntimeError, ImportError, ValueError) as exc:
        return TracedRun(ran=entry, error=f"{type(exc).__name__}: {exc}"[:300])
    finally:
        for path in (tracer, record):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass

    return TracedRun(
        ran=f"python {entry} (traced)",
        output=f"{done.stdout or ''}\n{done.stderr or ''}".strip(),
        exit_code=done.returncode,
        calls_seen=len(calls),
        contradictions=_contradictions_in(calls),
    )


def describe_contradictions(run: TracedRun) -> str:
    """What the run disagreed about, or "" when it was consistent."""
    if not run.contradictions:
        return ""
    lines = [
        f"I ran it with every call recorded ({run.calls_seen} calls) and it "
        f"contradicted itself:"
    ]
    lines.extend(item.as_sentence() for item in run.contradictions[:5])
    return "\n".join(lines)
