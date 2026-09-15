"""A result only its authors have produced is not yet a result.

What turns "Aura's own test says Aura passes" into something arguable is a
second party running the same protocol on their own machine and either
reproducing it or not. These pin that the package carries what that party
needs, and that the comparison is on verdicts and uncertainty rather than on
identical numbers — the campaign steps a live organism with a seeded generator,
so a different machine has a different clock, a different thread interleaving
and a different floating-point library.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
TOOL = REPO / "tools" / "subject_core_replication_package.py"


def _module():
    spec = importlib.util.spec_from_file_location("replication_package", TOOL)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    module = _module()
    if not module._runs(1):
        pytest.skip("no completed subject-core run to package")
    out = tmp_path_factory.mktemp("replication")
    assert module.build(3, out) == 0
    return module, out


def test_the_package_carries_what_a_replicator_needs(built) -> None:
    _, out = built
    for name in ("PINS.json", "EXPECTED.json", "THRESHOLDS.json", "HASHES.txt", "PROTOCOL.md"):
        assert (out / name).exists(), name


def test_the_pins_identify_the_tree_the_runs_came_from(built) -> None:
    """A commit without a tree hash does not pin a working directory."""
    _, out = built
    pins = json.loads((out / "PINS.json").read_text())
    for key in ("commit", "tree", "schema_hash", "campaign_fingerprint", "seeds", "command"):
        assert pins.get(key), key
    assert len(str(pins["commit"])) == 40
    assert pins["seeds"], "no declared seeds"


def test_the_pins_come_from_the_runs_rather_than_from_now(built) -> None:
    """Recomputing would describe this tree, not the measured one."""
    module, out = built
    pins = json.loads((out / "PINS.json").read_text())
    newest = json.loads((module._runs(1)[0] / "campaign.json").read_text())
    assert pins["commit"] == newest["commit"]
    assert pins["tree"] == newest["tree_hash"]


def test_every_artifact_is_hashed(built) -> None:
    _, out = built
    lines = [line for line in (out / "HASHES.txt").read_text().splitlines() if line.strip()]
    assert lines
    for line in lines:
        digest, _, name = line.partition("  ")
        assert len(digest) == 64, line
        assert name


def test_the_protocol_says_what_counts_and_what_would_falsify(built) -> None:
    _, out = built
    text = (out / "PROTOCOL.md").read_text()
    assert "What counts as a reproduction" in text
    assert "Not identical numbers" in text
    assert "What would falsify it" in text
    # The four named falsifiers, each a real outcome rather than a hedge.
    assert "null architecture passing the conjunction" in text
    assert "recurrent reference failing the conjunction" in text
    assert "shuffled and replayed" in text


def test_the_thresholds_travel_with_it(built) -> None:
    _, out = built
    thresholds = json.loads((out / "THRESHOLDS.json").read_text())
    assert thresholds
    assert all(isinstance(v, (int, float, str)) for v in thresholds.values())


def test_verifying_against_itself_finds_no_disagreement(built) -> None:
    module, out = built
    assert module.verify(out, 3) == 0


def test_a_flipped_verdict_is_reported_as_a_disagreement(built, tmp_path) -> None:
    """The check has to be able to fail, or it is not a check."""
    module, out = built
    rows = json.loads((out / "EXPECTED.json").read_text())
    name = next(iter(rows[0]["criteria"]))
    for row in rows:
        if name in row["criteria"]:
            row["criteria"][name]["passed"] = not row["criteria"][name]["passed"]
    theirs = tmp_path / "theirs"
    theirs.mkdir()
    (theirs / "EXPECTED.json").write_text(json.dumps(rows))
    assert module.verify(theirs, 3) == 2


def test_a_directory_with_nothing_in_it_is_refused(built, tmp_path) -> None:
    module, _ = built
    empty = tmp_path / "empty"
    empty.mkdir()
    assert module.verify(empty, 3) == 1
