"""A release requirement is decided by something you can run.

Every requirement in the registry carries an `acceptance` paragraph. A
paragraph is not a check: nothing runs it, and nothing fails when what it
describes stops being true. Counting 313 of them measures how much was written
down.

So the question here is narrower and answerable: is there something in this
repository a person can RUN whose failing would mean this requirement is not
met, and does that thing still exist?

33 of 313 on 2026-09-07. The number only goes up, and a link that stops
resolving — a renamed make target, a deleted test — fails immediately rather
than quietly becoming a paragraph again.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "config/requirement_acceptance_baseline.json"


@pytest.fixture(scope="module")
def report():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from tools.reqproof.acceptance import build

    return build(ROOT)


def test_no_link_points_at_something_that_is_gone(report) -> None:
    broken = report["summary"]["with_broken_links"]
    assert not broken, (
        "requirements naming a check that does not resolve in the tree: "
        f"{broken[:12]}"
    )


def test_the_number_linked_only_goes_up(report) -> None:
    baseline = json.loads(BASELINE.read_text())
    linked = report["summary"]["linked"]
    assert linked >= baseline["linked"], (
        f"{baseline['linked']} requirements had an executable check and now "
        f"{linked} do. Restore the link, or lower the baseline deliberately "
        "and say why."
    )


def test_the_registry_did_not_shrink_under_the_baseline(report) -> None:
    """Linking 33 of 33 is not the same fact as linking 33 of 313."""

    baseline = json.loads(BASELINE.read_text())
    assert report["summary"]["requirements"] >= baseline["requirements"], (
        "the registry lost requirements; the baseline is a fraction and both "
        "halves matter"
    )


def test_a_curated_link_is_a_link_somebody_can_run(report) -> None:
    """The map cannot be used to claim coverage with a name nobody checks."""

    from tools.reqproof.acceptance import CURATED

    curated = json.loads(CURATED.read_text())["checks"]
    entries = {entry["id"]: entry for entry in report["entries"]}
    for rid, checks in curated.items():
        assert rid in entries, f"{rid} is not a requirement in the registry"
        for check in checks:
            assert check["kind"] in {"test", "make", "tool"}, check
            resolved = [c["name"] for c in entries[rid]["checks"]]
            assert check["name"] in resolved, (
                f"{rid} names {check['name']}, which does not resolve"
            )


def test_a_deleted_selector_breaks_the_link_while_its_file_still_exists(tmp_path, monkeypatch) -> None:
    from tools.reqproof import acceptance

    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps({"requirements": [
        {"id": "missing", "acceptance": "Run tests/test_probe.py::TestProbe::test_deleted."},
        {"id": "present", "acceptance": "Run tests/test_probe.py::TestProbe::test_current."},
        {"id": "nested", "acceptance": "Run tests/test_probe.py::test_hidden."},
    ]}))
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_probe.py").write_text(
        "class TestProbe:\n"
        "    async def test_current(self):\n"
        "        def test_hidden():\n"
        "            raise AssertionError\n"
    )
    monkeypatch.setattr(acceptance, "ROOT", tmp_path)
    monkeypatch.setattr(acceptance, "REGISTRY", registry)
    monkeypatch.setattr(acceptance, "CURATED", tmp_path / "map.json")
    monkeypatch.setattr(acceptance, "LEDGER", tmp_path / "ledger.json")

    result = acceptance.build(tmp_path)
    entries = {entry["id"]: entry for entry in result["entries"]}
    assert result["summary"]["linked"] == 1
    assert result["summary"]["with_broken_links"] == ["missing", "nested"]
    assert entries["missing"]["broken_links"][0]["name"] == "tests/test_probe.py::TestProbe::test_deleted"
    assert entries["present"]["checks"][0]["name"] == "tests/test_probe.py::TestProbe::test_current"
