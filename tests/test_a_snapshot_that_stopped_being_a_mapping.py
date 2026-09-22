"""`_everything_she_can_say()` returns a snapshot, and one caller still read it
as the mapping it used to be.

`HowItStood` holds the registries under `.held`. When the helper changed from
returning that mapping to returning the snapshot around it, every caller was
updated except gate 9 of the gauntlet, which asked the snapshot for
`.values()`. The gate raised `AttributeError` instead of returning a verdict,
and only the full-suite gate run saw it, because the helper is annotated
loosely enough that nothing typed could.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CALL = re.compile(r"(?:_everything_she_can_say|as_it_stands)\(\)\s*\.\s*(\w+)")
# What the snapshot itself answers to. Anything else is a mapping read.
ON_THE_SNAPSHOT = {"held", "restore", "what_changed", "operator_state"}


def test_the_helper_returns_the_snapshot_not_the_mapping():
    from core.cognition.sequence_induction import _everything_she_can_say
    from core.cognition.what_she_can_take_back import HowItStood

    held = _everything_she_can_say()
    assert isinstance(held, HowItStood)
    assert hasattr(held.held, "values"), "the registries live under .held"
    assert not hasattr(held, "values"), (
        "the snapshot must not answer to a mapping read, or this drift "
        "becomes invisible again"
    )


def test_no_caller_reads_the_snapshot_as_a_mapping():
    offenders = []
    for where in ("core", "tools", "interface", "tests"):
        for path in (ROOT / where).rglob("*.py"):
            if path.name == Path(__file__).name:
                continue
            for number, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
                for attribute in CALL.findall(line):
                    if attribute not in ON_THE_SNAPSHOT:
                        rel = path.relative_to(ROOT)
                        offenders.append(f"{rel}:{number} reads .{attribute}")
    assert offenders == [], "\n".join(offenders)


def test_gate_nine_returns_a_verdict():
    """The gate that raised. Families zero keeps it to the shape, not the run."""
    from tools.agi_gauntlet.protocol import take_the_freeze
    from tools.agi_gauntlet.runnable import acquiring_a_new_skill

    found = acquiring_a_new_skill(take_the_freeze(), {"families": 0, "situations": 3})
    assert isinstance(found.get("passed"), bool)
    assert "trajectories" in found
    assert "things_she_can_say_afterwards" in found
