"""How it looks and how it answers you, graded like everything else.

    "Pieces didnt move. Nothing was polished. Looked/felt horrible."

That was the verdict on the first reconstruction that passed every gate this
repository had. The rules were right — the held-out battery said so — and the
result was a text grid you drove with w/a/s/d. Every check asked whether the
program was *correct*; none asked whether it was the software.

For most of the programs worth reconstructing, the surface is not decoration.
2048 without its tile palette, its slide, and its arrow keys is not 2048; it is
a matrix transformation with a printout. The palette is published, the controls
are published, and a reconstruction that ignores them is unfaithful in exactly
the way the behavioural battery cannot see.

So presentation gets a contract, and the contract gets graded:

* a real window, with a title and a canvas — not a print loop;
* the controls the original had, bound to the events they belong to;
* the palette, sampled against the published colours within a tolerance;
* the feedback surfaces a player needs — score, and an end state that says so;
* motion, because "pieces didn't move" is a defect and ``after()`` is how a
  toolkit expresses it.

**How it is graded without a screen.** The candidate's ``import tkinter`` is
resolved to a recording stand-in installed in ``sys.modules`` before its first
line runs, so nothing opens a window, and the real toolkit is never reachable.
The stand-in records what was built, bound, drawn and scheduled, then the game
is driven through synthetic key events and the recording is checked. That is a
functional test of the interface, not a grep for the word "canvas".
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from core.runtime.errors import record_degradation
from core.self_improvement.artifact_quality import QualityFinding, QualityReport

_RECOVERABLE = (RuntimeError, AttributeError, TypeError, ValueError, KeyError, IndexError, OSError)

#: The toolkit is substituted, not permitted. See ``audit_general_ast``.
SUBSTITUTED_MODULES = frozenset({"tkinter"})

_VERDICT_MARKER = "__AURA_PRESENTATION_VERDICT__"
_TIMEOUT_S = 25.0
_MEMORY_MB = 512

_HEX_RE = re.compile(r"^#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})$")


def _rgb(colour: str) -> tuple[int, int, int] | None:
    """A hex colour as channels, or None if it is not one."""
    text = str(colour or "").strip()
    if not _HEX_RE.match(text):
        return None
    body = text[1:]
    if len(body) == 3:
        body = "".join(char * 2 for char in body)
    return tuple(int(body[index : index + 2], 16) for index in (0, 2, 4))  # type: ignore[return-value]


def colour_distance(left: str, right: str) -> float:
    """Mean per-channel distance between two hex colours, or ``inf``.

    Deliberately crude. The question is "did they find this colour", not "is
    this a perceptual match" — and perceptual distance would decide no case
    here differently while needing a colour library to compute.

    The tolerance this is used with has to be small, and that is a property of
    the palettes themselves rather than a preference. 2048's eleven tile
    colours sit a mean 6.7 channel-steps apart at their closest (#edcc61 and
    #edc850 are neighbours), so a generous tolerance makes every colour match
    every other and the check stops measuring anything. Measured on the real
    palette: at tolerance 60, the empty-cell grey #cdc1b4 alone "matched" five
    of the eleven published tiles.
    """
    first, second = _rgb(left), _rgb(right)
    if first is None or second is None:
        return float("inf")
    return sum(abs(a - b) for a, b in zip(first, second, strict=True)) / 3.0


def match_palette(
    found: list[str], published: dict[str, str], tolerance: float
) -> tuple[dict[str, str], list[str]]:
    """Assign found colours to published ones, each colour used at most once.

    One-to-one is the whole point. Without it a single off-white satisfies
    every pale entry in the palette and an implementation that renders one
    colour scores as though it had found them all — which is precisely how a
    grey board passed a palette check.
    """
    remaining = list(found)
    matched: dict[str, str] = {}
    # Hardest first: an entry with only one plausible candidate should claim it
    # before an entry that has several.
    order = sorted(
        published.items(),
        key=lambda item: sum(1 for c in found if colour_distance(c, item[1]) <= tolerance),
    )
    for value, want in order:
        best = min(remaining, key=lambda c: colour_distance(c, want), default=None)
        if best is not None and colour_distance(best, want) <= tolerance:
            matched[value] = best
            remaining.remove(best)
    return matched, remaining


def hex_colours_in(source: str) -> list[str]:
    """Every hex colour literal the module names.

    Read statically, because the palette is a static property of the program
    and a rendered board only ever shows the two or three colours whose tiles
    happen to be on it. Checking the render alone would fail a perfect
    implementation for the crime of starting a new game.
    """
    import ast as _ast

    found: list[str] = []
    try:
        tree = _ast.parse(source)
    except SyntaxError:
        return found
    for node in _ast.walk(tree):
        if isinstance(node, _ast.Constant) and isinstance(node.value, str):
            text = node.value.strip()
            if _HEX_RE.match(text):
                found.append(text.lower())
    return list(dict.fromkeys(found))


@dataclass(frozen=True)
class PresentationContract:
    """The observable surface of the original, as published facts.

    Every field is something a person can check by looking at the real thing,
    which is what makes it clean-room material: no source is needed to know
    that 2048 is played with arrow keys on a 4x4 grid of coloured tiles.
    """

    #: Entry point that builds and returns the application object.
    app_factory: str = "main"
    #: Substrings that must appear in the window title, case-insensitive.
    title_contains: tuple[str, ...] = ()
    #: Events that must be bound, e.g. ``("<Up>", "<Down>", "<Left>", "<Right>")``.
    required_bindings: tuple[str, ...] = ()
    #: value -> published hex colour. Matched within ``colour_tolerance``.
    palette: dict[str, str] = field(default_factory=dict)
    #: Mean per-channel distance allowed. Small on purpose — see
    #: ``colour_distance``; real palettes are tightly spaced.
    colour_tolerance: float = 12.0
    #: Fraction of the palette that must be recognisably right.
    palette_coverage: float = 0.6
    #: Distinct drawn regions expected at rest (a 4x4 board is 16).
    min_drawn_cells: int = 0
    #: Text that must appear somewhere on the surface, case-insensitive.
    required_text: tuple[str, ...] = ()
    #: Does the surface have to move? ``after()`` is how a toolkit says so.
    requires_motion: bool = False
    #: Events to fire when driving it, in order.
    drive_events: tuple[str, ...] = ()

    def describe(self) -> str:
        parts = [f"window titled {'/'.join(self.title_contains) or 'anything'}"]
        if self.required_bindings:
            parts.append(f"{len(self.required_bindings)} controls")
        if self.palette:
            parts.append(f"{len(self.palette)}-colour palette")
        if self.requires_motion:
            parts.append("animated")
        return ", ".join(parts)


# ── The recording stand-in, as source (it runs in the child) ───────────────

FAKE_TOOLKIT_SOURCE = '''
import sys as _sys
import types as _types

_REC = {
    "titles": [], "bindings": [], "items": [], "texts": [], "after_calls": 0,
    "widgets": [], "mainloop": 0, "geometry": [],
}


class _Var:
    def __init__(self, *a, **k):
        self._value = k.get("value", "")

    def set(self, value):
        self._value = value
        _REC["texts"].append(str(value))

    def get(self):
        return self._value

    def trace_add(self, *a, **k):
        return "trace"

    trace = trace_add


class _Widget:
    def __init__(self, master=None, **kwargs):
        _REC["widgets"].append(type(self).__name__)
        self._kwargs = dict(kwargs)
        for key in ("text", "textvariable"):
            if key in kwargs and isinstance(kwargs[key], str):
                _REC["texts"].append(str(kwargs[key]))

    def __getattr__(self, name):
        def _anything(*args, **kwargs):
            return None
        return _anything

    def pack(self, *a, **k):
        return None

    def grid(self, *a, **k):
        return None

    def place(self, *a, **k):
        return None

    def config(self, **kwargs):
        self._kwargs.update(kwargs)
        if isinstance(kwargs.get("text"), str):
            _REC["texts"].append(kwargs["text"])
        return None

    configure = config

    def bind(self, sequence=None, func=None, add=None):
        _REC["bindings"].append(str(sequence))
        _BOUND.append((str(sequence), func))
        return "binding"

    bind_all = bind

    def after(self, delay, func=None, *args):
        _REC["after_calls"] += 1
        if callable(func):
            _PENDING.append((func, args))
        return "after#%d" % _REC["after_calls"]

    def after_cancel(self, *a, **k):
        return None

    def title(self, value=None):
        if value is not None:
            _REC["titles"].append(str(value))
        return "".join(_REC["titles"])

    def geometry(self, value=None):
        if value is not None:
            _REC["geometry"].append(str(value))
        return "".join(_REC["geometry"])

    def mainloop(self, *a, **k):
        _REC["mainloop"] += 1
        return None

    def destroy(self, *a, **k):
        return None

    def update(self, *a, **k):
        return None

    update_idletasks = update

    def focus_set(self, *a, **k):
        return None

    def winfo_width(self):
        return 400

    def winfo_height(self):
        return 400

    def winfo_exists(self):
        return 1


_BOUND = []
_PENDING = []


class _Canvas(_Widget):
    def _record(self, kind, kwargs):
        _REC["items"].append({
            "kind": kind,
            "fill": str(kwargs.get("fill", "")),
            "text": str(kwargs.get("text", "")),
        })
        if isinstance(kwargs.get("text"), (str, int, float)):
            _REC["texts"].append(str(kwargs["text"]))
        return len(_REC["items"])

    def create_rectangle(self, *a, **k):
        return self._record("rectangle", k)

    def create_oval(self, *a, **k):
        return self._record("oval", k)

    def create_text(self, *a, **k):
        return self._record("text", k)

    def create_line(self, *a, **k):
        return self._record("line", k)

    def create_polygon(self, *a, **k):
        return self._record("polygon", k)

    def create_image(self, *a, **k):
        return self._record("image", k)

    def itemconfig(self, item, **k):
        try:
            entry = _REC["items"][int(item) - 1]
        except (TypeError, ValueError, IndexError):
            return None
        if "fill" in k:
            entry["fill"] = str(k["fill"])
        if "text" in k:
            entry["text"] = str(k["text"])
            _REC["texts"].append(str(k["text"]))
        return None

    def coords(self, *a, **k):
        return [0, 0, 0, 0]

    def delete(self, *a, **k):
        return None

    def move(self, *a, **k):
        _REC["after_calls"] += 0
        return None


class _Event:
    def __init__(self, keysym="", char="", **kwargs):
        self.keysym = keysym
        self.char = char or (keysym[:1] if keysym else "")
        self.x = self.y = 0
        self.widget = None
        self.num = 1
        for key, value in kwargs.items():
            setattr(self, key, value)


_tk = _types.ModuleType("tkinter")
_tk.Tk = type("Tk", (_Widget,), {})
_tk.Toplevel = type("Toplevel", (_Widget,), {})
_tk.Frame = type("Frame", (_Widget,), {})
_tk.Label = type("Label", (_Widget,), {})
_tk.Button = type("Button", (_Widget,), {})
_tk.Entry = type("Entry", (_Widget,), {})
_tk.Canvas = _Canvas
_tk.StringVar = _Var
_tk.IntVar = _Var
_tk.DoubleVar = _Var
_tk.BooleanVar = _Var
_tk.Event = _Event
_tk.TclError = type("TclError", (Exception,), {})
for _name in (
    "N", "S", "E", "W", "NE", "NW", "SE", "SW", "CENTER", "LEFT", "RIGHT",
    "TOP", "BOTTOM", "BOTH", "X", "Y", "NONE", "END", "NORMAL", "DISABLED",
    "HORIZONTAL", "VERTICAL", "RAISED", "SUNKEN", "FLAT", "GROOVE", "RIDGE",
    "SOLID", "ALL", "TRUE", "FALSE", "YES", "NO",
):
    setattr(_tk, _name, _name.lower())

_ttk = _types.ModuleType("tkinter.ttk")
for _widget in ("Frame", "Label", "Button", "Style", "Notebook", "Entry"):
    setattr(_ttk, _widget, type(_widget, (_Widget,), {}))
_tk.ttk = _ttk

_font = _types.ModuleType("tkinter.font")
_font.Font = type("Font", (_Widget,), {})
_tk.font = _font

_messagebox = _types.ModuleType("tkinter.messagebox")
for _fn in ("showinfo", "showwarning", "showerror", "askyesno", "askokcancel"):
    setattr(_messagebox, _fn, lambda *a, **k: True)
_tk.messagebox = _messagebox

_sys.modules["tkinter"] = _tk
_sys.modules["tkinter.ttk"] = _ttk
_sys.modules["tkinter.font"] = _font
_sys.modules["tkinter.messagebox"] = _messagebox
'''


def _harness_source(contract: PresentationContract) -> str:
    payload = json.dumps(
        {
            "app_factory": contract.app_factory,
            "title_contains": list(contract.title_contains),
            "required_bindings": list(contract.required_bindings),
            "palette": dict(contract.palette),
            "colour_tolerance": contract.colour_tolerance,
            "palette_coverage": contract.palette_coverage,
            "min_drawn_cells": contract.min_drawn_cells,
            "required_text": list(contract.required_text),
            "requires_motion": contract.requires_motion,
            "drive_events": list(contract.drive_events),
        }
    )
    return f'''
_C = __import__("json").loads({payload!r})
_findings = []
_evidence = []


def _fail(check, detail):
    _findings.append({{"check": check, "detail": detail}})


def _channels(colour):
    text = str(colour or "").strip()
    if not text.startswith("#"):
        return None
    body = text[1:]
    if len(body) == 3:
        body = "".join(c * 2 for c in body)
    if len(body) != 6:
        return None
    try:
        return [int(body[i:i + 2], 16) for i in (0, 2, 4)]
    except ValueError:
        return None


def _near(left, right, tolerance):
    a, b = _channels(left), _channels(right)
    if a is None or b is None:
        return False
    return (sum(abs(x - y) for x, y in zip(a, b)) / 3.0) <= tolerance


# Build it. Anything raised here is the verdict.
_app = None
_factory = globals().get(_C["app_factory"])
if not callable(_factory):
    _fail("entry point", "no callable %s() to build the interface" % _C["app_factory"])
else:
    try:
        _app = _factory()
    except BaseException as _exc:
        _fail("builds", "%s() raised %s: %s" % (_C["app_factory"], type(_exc).__name__, _exc))

if _app is None and not _REC["widgets"]:
    # The single most expensive bug shape in this repository: a check that
    # cannot run reported as a check that passed. A console program builds no
    # widgets, so every question below was skipped and the verdict came back
    # clean for a program with no interface at all.
    _fail("window", "no interface was built — nothing created a window or a widget")
else:
    title = " ".join(_REC["titles"]).lower()
    for _want in _C["title_contains"]:
        if _want.lower() not in title:
            _fail("window", "the window title does not mention %r (title=%r)" % (_want, title))
    if _REC["titles"]:
        _evidence.append("window titled %r" % _REC["titles"][0])

    _bound = set(_REC["bindings"])
    _missing = [b for b in _C["required_bindings"] if b not in _bound]
    if _missing:
        _fail("controls", "not bound: %s (bound: %s)" % (", ".join(_missing), ", ".join(sorted(_bound)) or "nothing"))
    elif _C["required_bindings"]:
        _evidence.append("binds %s" % ", ".join(_C["required_bindings"]))

    _drawn = [i for i in _REC["items"] if i["kind"] in ("rectangle", "oval", "polygon", "image")]
    if _C["min_drawn_cells"] and len(_drawn) < _C["min_drawn_cells"]:
        _fail(
            "surface",
            "drew %d region(s); a %d-cell board was expected" % (len(_drawn), _C["min_drawn_cells"]),
        )
    elif _drawn:
        _evidence.append("draws %d region(s)" % len(_drawn))

    if _C["palette"]:
        # Which colours a fresh board shows depends on which tiles spawned, so
        # coverage is judged statically against the source. What the render can
        # prove is that the colours are actually USED — a module that declares
        # the whole palette and paints every cell the same is still grey mush.
        _fills = sorted(set(i["fill"] for i in _REC["items"] if i["fill"]))
        if len(_fills) < 2:
            _fail(
                "palette",
                "the whole surface is drawn in %s — the palette is declared but not used"
                % (_fills[0] if _fills else "no colour at all"),
            )
        else:
            _evidence.append("renders %d distinct colour(s)" % len(_fills))

    # Drive it. A board that does not change under its own controls is a picture.
    _before = [dict(i) for i in _REC["items"]]
    _fired = 0
    for _seq in _C["drive_events"]:
        for _bseq, _fn in _BOUND:
            if _bseq == _seq and callable(_fn):
                try:
                    _fn(_tk.Event(keysym=_seq.strip("<>")))
                    _fired += 1
                except BaseException as _exc:
                    _fail("responds", "%s raised %s: %s" % (_seq, type(_exc).__name__, _exc))
                break
    while _PENDING:
        _fn, _args = _PENDING.pop(0)
        try:
            _fn(*_args)
        except BaseException:
            break
    if _C["drive_events"]:
        if not _fired:
            _fail("responds", "no bound handler ran for any of %s" % ", ".join(_C["drive_events"]))
        elif _REC["items"] == _before and [i["fill"] for i in _REC["items"]] == [i["fill"] for i in _before]:
            _fail("responds", "the surface did not change after %d control event(s)" % _fired)
        else:
            _evidence.append("responds to %d control event(s)" % _fired)

    _surface = " ".join(_REC["texts"]).lower()
    for _want in _C["required_text"]:
        if _want.lower() not in _surface:
            _fail("feedback", "nothing on the surface says %r" % _want)
    if _C["required_text"] and not [
        w for w in _C["required_text"] if w.lower() not in _surface
    ]:
        _evidence.append("shows %s" % ", ".join(_C["required_text"]))

    if _C["requires_motion"] and _REC["after_calls"] == 0:
        _fail("motion", "nothing was ever scheduled — the pieces do not move")
    elif _REC["after_calls"]:
        _evidence.append("schedules %d frame(s)" % _REC["after_calls"])

print("{_VERDICT_MARKER}" + __import__("json").dumps({{"findings": _findings, "evidence": _evidence}}))
'''


def _verdict_from_stdout(stdout: str) -> dict[str, Any] | None:
    for line in reversed(str(stdout or "").splitlines()):
        if line.startswith(_VERDICT_MARKER):
            try:
                return json.loads(line[len(_VERDICT_MARKER) :])
            # not a failure: text that is not the JSON this expects is not a record it can
            # read back.
            except json.JSONDecodeError:
                return None
    return None


def grade_presentation(source: str, contract: PresentationContract) -> QualityReport:
    """Does this module actually look and behave like the software it claims?

    Runs in another process, against a stand-in toolkit, under the same limits
    as every other grading path here. Nothing opens a window; nothing reaches
    the real ``tkinter``.
    """
    report = QualityReport()
    if not str(source or "").strip():
        report.findings.append(QualityFinding("presentation", "there is no module to look at"))
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
        audit_general_ast(source, substituted_modules=SUBSTITUTED_MODULES)
    except ReconstructionASTViolation as exc:
        report.findings.append(
            QualityFinding("containment", f"the module reaches for ambient authority: {exc}")
        )
        return report
    except SyntaxError as exc:
        report.findings.append(QualityFinding("parses", f"the module does not parse: {exc}"))
        return report

    # The palette is a static property, checked without running anything. A
    # fresh board shows two or three colours whatever else is true of it, so
    # grading coverage off the render would fail a perfect implementation for
    # the crime of starting a new game.
    if contract.palette:
        found = hex_colours_in(source)
        matched, _ = match_palette(found, contract.palette, contract.colour_tolerance)
        ratio = len(matched) / float(len(contract.palette))
        if ratio < contract.palette_coverage:
            report.findings.append(
                QualityFinding(
                    "palette",
                    f"found {len(matched)} of {len(contract.palette)} published colours "
                    f"({ratio:.0%}, needs {contract.palette_coverage:.0%}); the module names "
                    + (", ".join(found[:8]) if found else "no colours at all"),
                )
            )
        else:
            report.evidence.append(f"palette {len(matched)}/{len(contract.palette)}")

    try:
        from core.self_modification.mutation_safety import MutationOutcome, SafeMutationEvaluator

        evaluator = SafeMutationEvaluator(timeout_seconds=_TIMEOUT_S, memory_mb=_MEMORY_MB)
        # Order matters: the stand-in must be in sys.modules before the
        # candidate's import statement executes, or the audit's allowance
        # would be handing it the real toolkit.
        diagnostics = evaluator.evaluate(
            FAKE_TOOLKIT_SOURCE + "\n" + source + "\n" + _harness_source(contract)
        )
    except _RECOVERABLE as exc:
        record_degradation(
            "presentation_contract", exc, severity="warning", action="presentation was not graded"
        )
        report.findings.append(
            QualityFinding("containment", f"presentation could not be graded in a sandbox: {exc}")
        )
        return report

    verdict = _verdict_from_stdout(getattr(diagnostics, "stdout", ""))
    if verdict is None:
        outcome = getattr(diagnostics, "outcome", None)
        detail = (getattr(diagnostics, "traceback_text", "") or "").strip()[-400:]
        if outcome is MutationOutcome.TIMEOUT:
            report.findings.append(
                QualityFinding(
                    "presentation",
                    f"the interface did not finish building within {_TIMEOUT_S:.0f}s — it "
                    "blocks, loops, or calls mainloop() at import",
                )
            )
        else:
            report.findings.append(
                QualityFinding(
                    "presentation",
                    f"the interface could not be built: "
                    f"{detail or getattr(outcome, 'value', 'unknown failure')}",
                )
            )
        return report

    for raw in verdict.get("findings") or []:
        report.findings.append(
            QualityFinding(str(raw.get("check") or "presentation"), str(raw.get("detail") or ""))
        )
    report.evidence.extend(str(item) for item in (verdict.get("evidence") or []))
    return report


__all__ = [
    "FAKE_TOOLKIT_SOURCE",
    "SUBSTITUTED_MODULES",
    "PresentationContract",
    "colour_distance",
    "grade_presentation",
]
