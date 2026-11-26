"""A log line that cannot format itself fails in the path that emits it.

Live, 2026-07-27, at 88% memory:

    Message: 'High memory pressure (%s%). Triggering emergency eviction.'
    Arguments: (88.1,)
    TypeError: not enough arguments for format string

``(%s%)`` — a literal percent written as ``%`` instead of ``%%``. Python's
logging swallows the traceback and prints "--- Logging error ---", so the line
never appeared and nothing looked wrong. The cost is where it sits: this is the
first statement of the emergency eviction path, so the one message that would
have explained a memory emergency was the one message that could not be written,
during the emergency.

The bug is invisible until the branch runs, which is why it survived: nothing
exercises an out-of-memory path in tests. A scan is the honest way to catch it —
the format strings are literals, so they can be checked without running anything.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

_ROOTS = ("core", "interface")
_LOG_METHODS = frozenset(
    {"debug", "info", "warning", "warn", "error", "exception", "critical", "log"}
)
# %(name)s, %s, %5.2f, %-10s, %r, %%, and friends.
_VALID_SPEC = "diouxXeEfFgGcrsa%"


def _percent_placeholder_count(template: str) -> int | None:
    """How many arguments this template consumes, or None if it is malformed."""
    count = 0
    index = 0
    while index < len(template):
        if template[index] != "%":
            index += 1
            continue
        index += 1
        if index >= len(template):
            return None  # trailing bare %
        if template[index] == "%":
            index += 1
            continue
        mapping_key = False
        if template[index] == "(":  # mapping key: %(name)s — consumes one dict
            closing = template.find(")", index)
            if closing < 0:
                return None
            index = closing + 1
            mapping_key = True
        while index < len(template) and template[index] in "#0- +":
            index += 1
        while index < len(template) and (template[index].isdigit() or template[index] == "*"):
            index += 1
        if index < len(template) and template[index] == ".":
            index += 1
            while index < len(template) and (template[index].isdigit() or template[index] == "*"):
                index += 1
        while index < len(template) and template[index] in "hlL":
            index += 1
        if index >= len(template) or template[index] not in _VALID_SPEC:
            return None  # not a conversion at all — the live defect
        if template[index] != "%":
            count = 1 if mapping_key else count + 1
        index += 1
    return count


def _receiver_is_a_logger(node: ast.expr) -> bool:
    """`logger`, `self.logger`, `LOG`, `_log` — by this codebase's convention."""
    if isinstance(node, ast.Name):
        return "log" in node.id.lower()
    if isinstance(node, ast.Attribute):
        return "log" in node.attr.lower()
    if isinstance(node, ast.Call):
        return _receiver_is_a_logger(node.func)
    return False


def _iter_log_calls():
    for root in _ROOTS:
        for path in sorted(Path(root).rglob("*.py")):
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except (SyntaxError, UnicodeDecodeError):
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                    continue
                if node.func.attr not in _LOG_METHODS:
                    continue
                if not _receiver_is_a_logger(node.func.value):
                    # warnings.warn() shares the name and not the semantics.
                    continue
                args = node.args[1:] if node.func.attr == "log" else node.args
                if not args or not isinstance(args[0], ast.Constant):
                    continue
                template = args[0].value
                if not isinstance(template, str):
                    continue
                yield path, node.lineno, template, args[1:]


def test_every_logging_format_string_is_well_formed() -> None:
    """A malformed template raises when the branch runs, not when it is written."""
    # logging formats only when args are supplied (`msg % self.args` is
    # guarded by `if self.args`), so a stray % in an argument-less message is
    # inert. These are the calls that would actually raise.
    broken = [
        f"{path}:{line}: {template!r}"
        for path, line, template, rest in _iter_log_calls()
        if rest and _percent_placeholder_count(template) is None
    ]
    assert not broken, "malformed logging format strings:\n" + "\n".join(broken)


def test_every_logging_call_passes_the_arguments_it_promises() -> None:
    """Too few arguments is the live TypeError; too many is silently wrong."""
    mismatched: list[str] = []
    for path, line, template, rest in _iter_log_calls():
        if not rest:
            continue  # nothing to format against
        expected = _percent_placeholder_count(template)
        if expected is None:
            continue  # reported by the test above
        if any(isinstance(arg, ast.Starred) for arg in rest):
            continue  # count unknowable without running it
        if expected == 1 and len(rest) == 1:
            continue  # could be a mapping or a single value; both are fine
        if expected != len(rest):
            mismatched.append(
                f"{path}:{line}: {expected} placeholder(s), {len(rest)} argument(s): {template!r}"
            )
    assert not mismatched, "logging argument mismatches:\n" + "\n".join(mismatched)


# ── The checker itself has to be right ─────────────────────────────────────

@pytest.mark.parametrize(
    ("template", "expected"),
    [
        ("plain message", 0),
        ("one %s here", 1),
        ("%s and %d", 2),
        ("literal percent %% only", 0),
        ("%.1f%% done", 1),
        ("%(name)s mapping", 1),
        ("%-10s padded", 1),
        ("%5.2f wide", 1),
    ],
)
def test_the_checker_counts_correctly(template: str, expected: int) -> None:
    assert _percent_placeholder_count(template) == expected


@pytest.mark.parametrize(
    "template",
    [
        "High memory pressure (%s%). Triggering emergency eviction.",  # the live one
        "trailing bare %",
        "bad spec %q here",
    ],
)
def test_the_checker_rejects_what_python_would_reject(template: str) -> None:
    assert _percent_placeholder_count(template) is None
    with pytest.raises((TypeError, ValueError)):
        _ = template % (1.0,)


def test_a_stray_percent_with_no_arguments_is_left_alone() -> None:
    """`logging` never formats a message that was given no arguments.

    "Firewall at 100%." and "RAM > 94%: ..." are malformed templates that can
    never raise, because `LogRecord.getMessage` guards the `%` on `if self.args`.
    Failing on those would make the scan noise, and a noisy scan gets disabled.
    """
    import logging

    record = logging.LogRecord(
        "t", logging.INFO, __file__, 1, "Firewall at 100%.", None, None
    )
    assert record.getMessage() == "Firewall at 100%."
