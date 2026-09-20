"""A warning limit set at the value that means nothing is wrong.

`morphogenesis.components` declared `yellow_high=1` while its own
description says "above one is a partition", and the limit is read as
`value >= yellow_high`. A whole population in one piece therefore reported
yellow on every boot, so the channel that exists to name a partition said
the same thing partitioned or not. `endogenous.unexpected_refusals`
declared `yellow_high=0` on a count of faults: zero faults read as yellow.

Both were live on 2026-09-20, in the same boot, one line apart.

A count of something bad is the case a machine can decide: zero of it is
always the healthy end, so a limit at zero can never distinguish. The rest
needs a reader, which is why this checks the decidable one and names the
others rather than guessing at them.
"""
from __future__ import annotations

import ast
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCANNED = ("core", "interface", "skills")

#: Channels whose value counts occurrences of something unwanted. Zero is the
#: healthy end of each, so a `yellow_high` of zero is a limit that fires on
#: the best reading the channel can take.
_COUNTS_SOMETHING_UNWANTED = (
    "refusal",
    "failure",
    "failures",
    "violation",
    "violations",
    "error",
    "errors",
    "splat",
    "splats",
    "leak",
    "leaks",
    "stall",
    "stalls",
    "drop",
    "drops",
    "rejected",
    "rolled_back",
    "partial_failures",
)


def _declared_channels():
    """Every `channel(...)`/`dict(...)` declaration, read from the source.

    The dictionary is populated at runtime by whichever subsystems have
    booted, so a live snapshot sees only part of it. The declarations are in
    the tree whether or not anything ran.
    """
    found = []
    for root in SCANNED:
        for path in sorted((ROOT / root).rglob("*.py")):
            if "__pycache__" in str(path):
                continue
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except (OSError, SyntaxError, UnicodeError):
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                called = getattr(node.func, "id", getattr(node.func, "attr", ""))
                if called not in {"channel", "dict"}:
                    continue
                keywords = {
                    k.arg: k.value
                    for k in node.keywords
                    if k.arg is not None
                }
                if "name" not in keywords or "yellow_high" not in keywords:
                    continue
                name = keywords["name"]
                literal = None
                if isinstance(name, ast.Constant) and isinstance(name.value, str):
                    literal = name.value
                elif isinstance(name, ast.Name):
                    literal = _constant_named(tree, name.id)
                yellow = keywords["yellow_high"]
                if literal is None or not isinstance(yellow, ast.Constant):
                    continue
                found.append(
                    (str(path.relative_to(ROOT)), node.lineno, literal, yellow.value)
                )
    return found


def _constant_named(tree, identifier):
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if (
                isinstance(target, ast.Name)
                and target.id == identifier
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            ):
                return node.value.value
    return None


def test_no_fault_count_warns_at_zero() -> None:
    declared = _declared_channels()
    assert declared, "no channel declarations found; the reader has drifted"

    offenders = [
        f"{path}:{line} {name}: yellow_high=0 on a count of faults"
        for path, line, name, yellow in declared
        if isinstance(yellow, (int, float))
        and float(yellow) == 0.0
        and any(
            word in name.rsplit(".", 1)[-1] for word in _COUNTS_SOMETHING_UNWANTED
        )
    ]

    assert offenders == [], (
        "a limit read as `value >= yellow_high` at zero fires on the healthy "
        "reading, so the channel is the same colour clean or broken:\n"
        + "\n".join(offenders)
    )


def test_the_partition_channel_is_quiet_when_there_is_no_partition() -> None:
    from core.morphogenesis.telemetry import CHANNEL_COMPONENTS

    declared = {
        name: yellow for _path, _line, name, yellow in _declared_channels()
    }
    yellow = declared[CHANNEL_COMPONENTS]

    assert yellow > 1, (
        "one component is the whole population in one piece, which is what "
        "this channel exists to say is NOT happening"
    )


def test_the_detector_can_see_one() -> None:
    """The null: a gate that cannot match reports green forever."""
    tail = "unexpected_refusals".rsplit(".", 1)[-1]
    assert any(word in tail for word in _COUNTS_SOMETHING_UNWANTED)
    assert not any(
        word in "stamina" for word in _COUNTS_SOMETHING_UNWANTED
    ), "an ordinary gauge must not be read as a fault count"
