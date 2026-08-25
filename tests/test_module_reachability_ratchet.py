"""Modules nothing reaches may only get fewer.

279 of 2,839 modules under core/ — 26,060 lines — are reached by nothing: no
import anywhere in the repo, and no dotted path in a string literal either.

That number is not primarily a tidiness problem. When a large slice of the tree
is unreachable, "is X wired?" stops having a trustworthy answer, and this
codebase has been bitten by exactly that failure repeatedly: a second affect
engine with no construction path, a declared fallback that could never run, a
vision flag with three different defaults across four files. Unreachable code
is where half-wired subsystems hide, because nothing distinguishes "staged for
later" from "silently disconnected".

Bulk deletion is the wrong response and this ratchet does not ask for one.
Some orphans are entry points, some are staged work, and the frozen reqproof
and architecture artifacts reference paths that still have to resolve. What
the ratchet does is make the count visible and one-way, so the cost lands on
whoever adds the next one.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.lint_module_reachability import BASELINE, scan

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def report() -> dict[str, object]:
    return scan()


@pytest.fixture(scope="module")
def baseline() -> dict[str, object]:
    assert BASELINE.is_file(), "reachability baseline is missing"
    return json.loads(BASELINE.read_text(encoding="utf-8"))


def test_no_new_unreachable_modules(report, baseline):
    known = set(baseline["orphans"])
    now = set(report["orphans"])
    new = sorted(now - known)
    assert not new, (
        "these modules became unreachable — wire them to something or retire "
        "them:\n  " + "\n  ".join(new)
    )


def test_the_count_does_not_rise(report, baseline):
    assert report["orphan_count"] <= baseline["orphan_count"]


def test_dynamic_references_count_as_reachable():
    """A module reached only by importlib or a registry string is not dead.

    Naive import-graph analysis called 292 modules orphaned; string-literal
    detection found 13 of those were reached by name. Deleting one of those
    would have broken a working path while the analysis reported it as dead
    weight — the same class of error the ratchet exists to prevent.
    """
    from tools.lint_module_reachability import _iter_sources

    sources = _iter_sources()
    assert sources, "the source scan found nothing at all"
    # A real dynamic reference exists somewhere in the tree; if string
    # detection regressed to zero, every such module would flip to orphaned
    # at once and the ratchet would fire on all of them.
    text = "\n".join(
        p.read_text(encoding="utf-8", errors="ignore")
        for p in sources
        if p.name.endswith(".py")
    )
    assert '"core.' in text or "'core." in text


def test_the_baseline_is_a_list_of_real_modules(baseline):
    """A stale baseline naming deleted files would silently absorb new orphans."""
    root = Path(__file__).resolve().parent.parent
    missing = [
        name
        for name in baseline["orphans"]
        if not (root / (name.replace(".", "/") + ".py")).is_file()
        and not (root / (name.replace(".", "/") + "/__init__.py")).is_file()
    ]
    assert not missing, (
        "the baseline names modules that no longer exist; refresh it with "
        f"--write-baseline: {missing[:10]}"
    )


# --------------------------------------------------------------------------
# Every orphan has a decision
# --------------------------------------------------------------------------


def _dispositions() -> dict:
    path = Path(__file__).resolve().parent.parent / "config" / "orphan_dispositions.json"
    assert path.is_file(), "the disposition registry is missing"
    return json.loads(path.read_text(encoding="utf-8"))


def test_every_unreachable_module_has_a_recorded_decision(report):
    """"279 unreachable" is a number; "279 decided" is a plan.

    Counting them stops the total growing and says nothing about what any one
    of them is. Some are entry points invoked by name, some expose a service
    surface nothing reaches, and some are scaffolding — and a module nobody has
    decided about is exactly how a half-wired subsystem survives.
    """
    decided = _dispositions()["modules"]
    undecided = sorted(set(report["orphans"]) - set(decided))
    assert not undecided, (
        "unreachable modules with no recorded decision:\n  " + "\n  ".join(undecided[:10])
    )


def test_every_disposition_is_one_of_the_declared_kinds():
    data = _dispositions()
    allowed = set(data["dispositions"])
    bad = {
        name: entry["disposition"]
        for name, entry in data["modules"].items()
        if entry.get("disposition") not in allowed
    }
    assert not bad, f"unknown dispositions: {bad}"


def test_the_wire_pending_set_is_real_debt_and_stays_visible():
    """53 modules expose a service surface and nothing reaches them.

    This is the half-wired shape this codebase keeps finding — a second affect
    engine, an unreachable fallback — at scale. Pinned so the count cannot
    quietly grow while the headline orphan number stays flat.
    """
    modules = _dispositions()["modules"]
    pending = [n for n, e in modules.items() if e["disposition"] == "WIRE_PENDING"]
    # 66 at first count, 65 after core.cognition.belief_revision was retired
    # as a duplicate, 53 once relative imports were resolved correctly — a
    # third of the "unwired services" were reachable all along, through
    # imports the scanner could not see. The ceiling only falls.
    assert len(pending) <= 53, (
        f"modules written to be wired but unreachable grew to {len(pending)}"
    )


def test_the_refresh_command_cannot_raise_the_count(tmp_path, monkeypatch):
    """A refresh that can loosen the gate is how debt becomes the new normal."""
    import json

    import tools.lint_module_reachability as gate

    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"orphan_count": 0, "orphans": []}) + "\n")
    monkeypatch.setattr(gate, "BASELINE", baseline)
    monkeypatch.setattr("sys.argv", ["lint_module_reachability", "--write-baseline"])

    assert gate.main() == 1, "a rise was written"
    assert json.loads(baseline.read_text())["orphan_count"] == 0


def test_a_package_reached_only_through_a_submodule_is_reached(report):
    """Importing core.engineering.draw.schematic executes the package.

    Counting only the leaf reported three packages unreachable while their own
    submodules were imported across the tree — the implicit-import blind spot
    this scanner already names when it excludes packages from the test-only
    count.
    """
    for package in ("core.engineering.draw", "core.construction", "core.diagnosis"):
        assert package not in report["orphans"], package
