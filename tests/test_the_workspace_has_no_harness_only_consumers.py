"""A workspace consumer that exists only while the battery runs is part of the test.

The global workspace broadcasts to whatever consumers are registered on it, and
every edge the battery keeps out of G is a route through one of them. If the
subject-core harness registered a consumer of its own, or a registration only
happened under the test profile, the edge would be a property of the harness
rather than of her.

Consumers are registered in one place, by the production consciousness system,
and nothing under `core/subject` touches the workspace's registration surface.
Both are checked by reading the tree, so a new registration elsewhere fails here
before it can reach a run.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[1]

#: The calls that put something on the workspace's consumer list.
REGISTRATION = re.compile(
    r"register_broadcast_consumers\(|\.register_processor\(|\.register_consumer\(|\.add_consumer\("
)


def _calls(root: Path) -> dict[str, list[int]]:
    found: dict[str, list[int]] = {}
    for path in sorted(root.rglob("*.py")):
        for number, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#") or stripped.startswith("def "):
                continue
            if REGISTRATION.search(line):
                found.setdefault(str(path.relative_to(REPO)), []).append(number)
    return found


def test_consumers_are_registered_by_the_production_consciousness_system_alone() -> None:
    callers = {
        path
        for path in _calls(REPO / "core")
        if path != "core/consciousness/broadcast_consumers.py"
    }
    registration_sites = {
        path for path in callers
        if "register_broadcast_consumers(" in (REPO / path).read_text(encoding="utf-8")
    }
    assert registration_sites == {"core/consciousness/system.py"}, registration_sites


def test_the_instrument_never_registers_a_consumer_on_the_workspace() -> None:
    assert _calls(REPO / "core" / "subject") == {}


def test_no_consumer_registration_depends_on_the_test_profile() -> None:
    """A registration behind AURA_TESTING or a proof-run check is one the live
    runtime does not have, which is the harness-only case by another route."""
    source = (REPO / "core" / "consciousness" / "broadcast_consumers.py").read_text(encoding="utf-8")
    body = source[source.index("def register_broadcast_consumers"):]
    for switch in ("AURA_TESTING", "proof_run_active", "PYTEST_CURRENT_TEST", "subject_core"):
        assert switch not in body, f"consumer registration branches on {switch}"
