"""Evidence about a decision is retired by changes to that decision, not by any commit.

Calibration support needs fifty graded episodes in one cohort, and a cohort
begins again when its revision changes. Keyed on the repo's HEAD, that meant
every commit anywhere retired every cohort — and on a machine whose source
changes several times an hour, no control point ever reached fifty. The
mechanism was correct and could not fire.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.ontogeny import features as features_module
from core.ontogeny.features import FeatureSchema, decision_revision, register_schema


@pytest.fixture(autouse=True)
def _forget_pinned_revisions():
    """The cache is per process on purpose; a test needs its own."""

    features_module._REVISIONS.clear()
    yield
    features_module._REVISIONS.clear()


@pytest.fixture
def watched(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """A control point whose features come from one file we can edit."""

    root = tmp_path / "repo"
    (root / "core" / "widgets").mkdir(parents=True)
    (root / "core" / "widgets" / "chooser.py").write_text("VALUE = 1\n")
    # decision_revision reads relative to the repo root above core/ontogeny.
    monkeypatch.setattr(
        features_module,
        "__file__",
        str(root / "core" / "ontogeny" / "features.py"),
    )
    register_schema(
        FeatureSchema(
            control_point="widgets.choice",
            names=("size",),
            sources={"size": "core/widgets/chooser.py:Chooser"},
        )
    )
    return str(root / "core" / "widgets" / "chooser.py")


def test_a_commit_that_cannot_change_the_decision_keeps_the_evidence(
    watched: str,
) -> None:
    before = decision_revision("widgets.choice", fallback="commit-one")
    features_module._REVISIONS.clear()
    after = decision_revision("widgets.choice", fallback="commit-two-somewhere-else")
    assert before == after, "an unrelated commit retired the cohort"


def test_a_change_to_the_code_that_decides_does_retire_the_evidence(
    watched: str,
) -> None:
    before = decision_revision("widgets.choice", fallback="commit-one")
    Path(watched).write_text("VALUE = 2\n")
    features_module._REVISIONS.clear()
    after = decision_revision("widgets.choice", fallback="commit-one")
    assert before != after, "the decision changed and the old evidence stood"


def test_a_control_point_that_declares_nothing_keeps_the_coarse_revision() -> None:
    """Unknown provenance is no argument for keeping evidence."""

    assert decision_revision("nothing.declared.at.all", fallback="head-sha") == "head-sha"


def test_the_revision_is_pinned_for_the_life_of_the_process(watched: str) -> None:
    """A running process executes the code it booted with, whatever the tree says."""

    before = decision_revision("widgets.choice", fallback="commit-one")
    Path(watched).write_text("VALUE = 3\n")
    assert decision_revision("widgets.choice", fallback="commit-one") == before


def test_a_declared_source_that_is_missing_is_still_a_stable_fact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repo"
    (root / "core" / "ontogeny").mkdir(parents=True)
    monkeypatch.setattr(
        features_module, "__file__", str(root / "core" / "ontogeny" / "features.py")
    )
    register_schema(
        FeatureSchema(
            control_point="widgets.gone",
            names=("size",),
            sources={"size": "core/widgets/vanished.py"},
        )
    )
    first = decision_revision("widgets.gone", fallback="head-sha")
    features_module._REVISIONS.clear()
    assert decision_revision("widgets.gone", fallback="head-sha") == first
    assert first != "head-sha"


def test_the_live_control_points_get_a_revision_of_their_own() -> None:
    """The three the runtime reports on must not fall back to the repo sha."""

    from core.ontogeny import control_points

    register_schema(control_points.MEMORY_RETRIEVAL_SCHEMA)
    register_schema(control_points.COGNITION_EFFORT_SCHEMA)
    for name in (
        "executive.admission",
        control_points.MEMORY_RETRIEVAL,
        control_points.COGNITION_EFFORT,
    ):
        assert decision_revision(name, fallback="head-sha") != "head-sha", name
