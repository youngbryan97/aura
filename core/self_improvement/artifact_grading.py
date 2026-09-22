"""Grade any reconstructed artifact — across a process boundary.

This is the piece that makes the lane an engine rather than a builder for two
programs. Nothing here knows what the target is. It takes the source she wrote
and the plan she wrote, and asks the same questions of both: does it stand up
as code, does it reproduce her own worked examples, do her own invariants hold,
and — through the plan's adapter — does it behave: the pieces move, illegal
actions are refused, it can end, and there is something to look at.

The adapter is what generalises the behavioural half. Every interactive program
has a starting state, a set of things a user may do, an effect for doing one,
and a view. Given four function names, "do the pieces move?" is answerable
without knowing which program it is.

**Where it runs is a security property, not a detail.**

The first version of this module executed the candidate with ``exec()`` inside
Aura's own interpreter, substituting ``print`` and ``input`` in the namespace.
That is not containment. A synthesised module could still ``import os``, open
files, spawn subprocesses, or reach the network — and the whole point of this
lane is to run programs that nobody has read, produced by a model, from
evidence about software nobody controls. Replacing two builtins in a namespace
is a comment about intent, not a boundary.

So nothing untrusted runs here any more. Grading happens inside
``SafeMutationEvaluator``'s subprocess, which the repository already uses for
exactly this purpose: a separate interpreter, RLIMIT_AS and RLIMIT_CPU fences,
a scrubbed environment, a temporary working directory, and a wall-clock timeout
that kills. Before that, the candidate passes ``audit_general_ast``, which
refuses ambient authority (os, sys, subprocess, socket, open, importlib,
ctypes) and the dunder gadgets that reach it another way. Static analysis alone
is not trusted either — the AST audit narrows what can be attempted, and the
process boundary is what makes the attempt survivable.

The only work still done in-process is parsing, which is not executing.
"""
from __future__ import annotations

import json
from typing import Any

from core.self_improvement.artifact_quality import (
    QualityFinding,
    QualityReport,
    check_source_is_finished,
)
from core.self_improvement.reconstruction_plan import ReconstructionPlan

_RECOVERABLE = (RuntimeError, AttributeError, TypeError, ValueError, KeyError, IndexError, OSError)

# Bounds for the child. Generous enough for a real program's battery, small
# enough that a runaway candidate dies quickly rather than competing with the
# resident model for the host.
_GRADING_TIMEOUT_S = 25.0
_GRADING_MEMORY_MB = 512

_VERDICT_MARKER = "__AURA_GRADING_VERDICT__"
# stdout is shared with the harness and with the sandbox's own result channel.
# Counting either as "the module printed" would fail every artifact, including
# a silent one.
_HARNESS_MARKERS = (_VERDICT_MARKER, "__MUTATION_RESULT__")


def _harness_source(plan: ReconstructionPlan) -> str:
    """The grading harness, which runs beside the candidate inside the child.

    Written as a source string rather than imported because the child has no
    access to Aura's package tree — by design. It speaks back over stdout as
    one JSON line behind a marker, which is the only channel out.
    """
    payload = json.dumps(plan.to_dict(), sort_keys=True)
    return f'''

# ── grading harness (runs in the sandboxed child, beside the candidate) ────
import json as _json

_PLAN = _json.loads({payload!r})
_findings = []
_evidence = []


def _fail(check, detail):
    _findings.append({{"check": check, "detail": str(detail)[:300]}})


def _lookup(name):
    return globals().get(name)


def _describe(state):
    try:
        return repr(state)[:4000]
    except Exception:
        return "<unrepresentable>"


# Entry points the plan promised.
_missing = [n for n in _PLAN["entry_points"] if not callable(_lookup(n))]
if _missing:
    _fail("entry points",
          "the plan promises " + ", ".join(_missing[:5]) +
          " and the module does not define " + ("them" if len(_missing) > 1 else "it"))

if not _findings:
    # Her own worked examples.
    _examples = _PLAN["worked_examples"]
    if _examples:
        _passed = 0
        _first = ""
        for _ex in _examples:
            _fn = _lookup(_ex["entry_point"])
            if not callable(_fn):
                _first = _first or (_ex["entry_point"] + " is not callable")
                continue
            try:
                _actual = _fn(_ex["argument"])
            except Exception as _exc:
                _first = _first or (
                    _ex["entry_point"] + " raised " + type(_exc).__name__)
                continue
            if _actual == _ex["expected"]:
                _passed += 1
            elif not _first:
                _first = "%s: expected %r, got %r" % (
                    _ex["entry_point"], _ex["expected"], _actual)
        if _passed == len(_examples):
            _evidence.append("%d/%d worked examples reproduced" % (_passed, len(_examples)))
        else:
            _fail("worked examples",
                  "%d/%d reproduced; first failure — %s" % (_passed, len(_examples), _first))

    # Her own invariants.
    _invariants = _PLAN["invariants"]
    if _invariants:
        _held = 0
        for _inv in _invariants:
            _scope = dict((k, v) for k, v in globals().items() if not k.startswith("_"))
            _scope.update(_inv.get("bindings") or {{}})
            try:
                _outcome = eval(_inv["expression"], {{"__builtins__": {{}}}}, _scope)
            except Exception as _exc:
                _fail("invariants", "%r could not be evaluated: %s: %s" % (
                    _inv["description"], type(_exc).__name__, _exc))
                continue
            if _outcome:
                _held += 1
            else:
                _fail("invariants", "%r does not hold" % (_inv["description"],))
        if _held:
            _evidence.append("%d/%d invariants hold" % (_held, len(_invariants)))

    # Behaviour, through the adapter.
    _adapter = _PLAN.get("adapter")
    if not _adapter:
        _evidence.append("no adapter declared; behaviour was not claimed either way")
    else:
        def _initial():
            return _lookup(_adapter["initial_state"])()

        def _legal(state):
            return list(_lookup(_adapter["legal_actions"])(state) or [])

        def _apply(state, action):
            return _lookup(_adapter["apply_action"])(state, action)

        _state = None
        try:
            _state = _initial()
        except Exception as _exc:
            _fail("interactive", "could not build a starting state: %s: %s" % (
                type(_exc).__name__, _exc))

        if _state is not None:
            _moved = 0
            _unchanged = 0
            _seen = set([_describe(_state)])
            _min = int(_adapter.get("min_effective_actions") or 8)
            for _ in range(240):
                try:
                    _actions = _legal(_state)
                except Exception as _exc:
                    _fail("interactive", "listing legal actions raised %s: %s" % (
                        type(_exc).__name__, _exc))
                    break
                if not _actions:
                    break
                _before = _describe(_state)
                try:
                    _state = _apply(_state, _actions[0])
                except Exception as _exc:
                    _fail("interactive", "applying a legal action raised %s: %s" % (
                        type(_exc).__name__, _exc))
                    break
                _after = _describe(_state)
                if _after == _before:
                    _unchanged += 1
                    if _unchanged >= 3:
                        _fail("interactive",
                              "the state did not change after three legal actions — "
                              "the pieces do not move")
                        break
                else:
                    _moved += 1
                    _seen.add(_after)
                if _moved >= _min * 3:
                    break
            if _moved:
                _evidence.append("%d legal actions changed the state, %d distinct positions" % (
                    _moved, len(_seen)))
            if _moved < _min and not any(f["check"] == "interactive" for f in _findings):
                _fail("interactive",
                      "only %d legal action(s) changed the state; expected at least %d" % (
                          _moved, _min))

            # Illegal actions must be refused.
            _illegal = _adapter.get("illegal_action_examples") or []
            if _illegal:
                _base_state = _initial()
                _baseline = _describe(_base_state)
                _accepted = []
                for _bad in _illegal[:8]:
                    try:
                        _result = _apply(_base_state, _bad)
                    except Exception:
                        continue
                    if _describe(_result) != _baseline:
                        _accepted.append(_bad)
                if _accepted:
                    _fail("rules",
                          "%d illegal action(s) were accepted and changed the state, e.g. %r" % (
                              len(_accepted), _accepted[0]))
                else:
                    _evidence.append("illegal actions were refused")

            # Reaching an end, only when the plan claims one.
            if _adapter.get("expects_terminal_state"):
                _ended = False
                _term_state = _initial()
                for _step in range(400):
                    _acts = _legal(_term_state)
                    if not _acts:
                        _ended = True
                        _evidence.append("reached a terminal state after %d actions" % _step)
                        break
                    _term_state = _apply(_term_state, _acts[0])
                if not _ended:
                    _fail("terminal", "no terminal state within 400 actions")

            # Something to look at.
            _render_name = _adapter.get("render") or ""
            if _render_name:
                _renderer = _lookup(_render_name)
                if not callable(_renderer):
                    _fail("presentation", "no render surface named " + _render_name)
                else:
                    try:
                        _view = _renderer(_initial())
                    except Exception as _exc:
                        _fail("presentation", "rendering raised %s: %s" % (
                            type(_exc).__name__, _exc))
                    else:
                        _text = _view if isinstance(_view, str) else str(_view)
                        _min_chars = int(_adapter.get("min_render_chars") or 2)
                        if len(_text.strip()) < _min_chars:
                            _fail("presentation",
                                  "the rendered view is %d characters; that is not a view" % (
                                      len(_text.strip()),))
                        else:
                            _lines = [ln for ln in _text.splitlines() if ln.strip()]
                            _evidence.append("renders %d line(s), %d characters" % (
                                len(_lines), len(_text)))

print("{_VERDICT_MARKER}" + _json.dumps({{"findings": _findings, "evidence": _evidence}}))
'''


def _verdict_from_stdout(stdout: str) -> dict[str, Any] | None:
    for line in reversed(str(stdout or "").splitlines()):
        if line.startswith(_VERDICT_MARKER):
            try:
                return json.loads(line[len(_VERDICT_MARKER):])
            # not a failure: text that is not the JSON this expects is not a record it can
            # read back.
            except json.JSONDecodeError:
                return None
    return None


def grade_artifact(source: str, plan: ReconstructionPlan) -> QualityReport:
    """Everything that can be checked about this artifact, without knowing it.

    Ordered so the cheapest and most decisive checks run first, and so nothing
    untrusted executes until it has passed a static audit and can be confined:
    source that does not parse cannot be audited, source that fails the audit is
    never run, and source that runs does so in another process under limits.
    """
    report = QualityReport()

    # Parsing is not executing — safe here, and it catches the cheapest defects.
    report.findings.extend(check_source_is_finished(source))
    if not report.passed:
        return report

    try:
        from core.discovery.reconstruction_sandbox import (
            ReconstructionASTViolation,
            audit_general_ast,
        )
    except ImportError as exc:  # pragma: no cover - the sandbox is a hard dep
        report.findings.append(
            QualityFinding("containment", f"the sandbox is unavailable, so nothing was run: {exc}")
        )
        return report

    try:
        audit_general_ast(source)
    except ReconstructionASTViolation as exc:
        # Not a grading failure so much as a refusal: a reconstruction that
        # reaches for the filesystem, the network or a subprocess is not
        # something to run and score, whatever else is true of it.
        report.findings.append(
            QualityFinding("containment", f"the module reaches for ambient authority: {exc}")
        )
        return report
    except SyntaxError as exc:
        report.findings.append(QualityFinding("parses", f"the module does not parse: {exc}"))
        return report

    try:
        from core.self_modification.mutation_safety import MutationOutcome, SafeMutationEvaluator

        evaluator = SafeMutationEvaluator(
            timeout_seconds=_GRADING_TIMEOUT_S, memory_mb=_GRADING_MEMORY_MB
        )
        diagnostics = evaluator.evaluate(source + "\n" + _harness_source(plan))
    except _RECOVERABLE as exc:
        report.findings.append(
            QualityFinding("containment", f"grading could not be run in a sandbox: {exc}")
        )
        return report

    verdict = _verdict_from_stdout(getattr(diagnostics, "stdout", ""))
    if verdict is None:
        outcome = getattr(diagnostics, "outcome", None)
        detail = (getattr(diagnostics, "traceback_text", "") or "").strip()[-400:]
        if outcome is MutationOutcome.TIMEOUT:
            report.findings.append(
                QualityFinding(
                    "quiet import",
                    f"grading did not finish within {_GRADING_TIMEOUT_S:.0f}s — the module "
                    "loops, blocks on input, or never returns",
                )
            )
        elif outcome is MutationOutcome.OOM:
            report.findings.append(
                QualityFinding("quiet import", "the module exhausted the sandbox's memory limit")
            )
        else:
            report.findings.append(
                QualityFinding(
                    "quiet import",
                    f"the module did not survive being imported and graded: "
                    f"{detail or getattr(outcome, 'value', 'unknown failure')}",
                )
            )
        return report

    for raw in verdict.get("findings") or []:
        report.findings.append(
            QualityFinding(str(raw.get("check") or "grading"), str(raw.get("detail") or ""))
        )
    report.evidence.extend(str(item) for item in (verdict.get("evidence") or []))

    # Output on the way out is a defect, not a crime: the interactive loop
    # belongs under __main__, and a module that narrates at import cannot be
    # reused. The harness's verdict line and the sandbox's own result line are
    # channels, not the module talking, so neither is counted.
    printed = [
        line
        for line in str(getattr(diagnostics, "stdout", "")).splitlines()
        if line.strip() and not line.startswith(_HARNESS_MARKERS)
    ]
    if printed:
        report.findings.append(
            QualityFinding(
                "quiet import",
                f"module printed {len(printed)} line(s) while loading; the interactive "
                "loop belongs under __main__",
            )
        )
    return report


__all__ = ["grade_artifact"]
