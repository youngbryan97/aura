"""A liking counts as hers when it stands more than two standard errors from indifference."""
from __future__ import annotations

import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _tool():
    spec = importlib.util.spec_from_file_location("own", REPO / "tools" / "what_she_does_of_her_own.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_only_likings_clear_of_their_uncertainty_are_held():
    tool = _tool()
    things = [
        {"label": "bicycle", "liking": 0.7, "unsure": 0.04, "contacts": 5},
        {"label": "rain", "liking": 0.1, "unsure": 0.04, "contacts": 3},
        {"label": "probe", "liking": -0.2, "unsure": 0.001, "contacts": 140},
        {"label": "never weighed", "liking": 0.5, "unsure": 0.0, "contacts": 0},
    ]
    held = {row["thing"]: row for row in tool.held(things)}
    assert set(held) == {"bicycle", "probe"}
    report = tool.report({"turns": 10, "things": things})
    assert [row["thing"] for row in report["liked"]] == ["bicycle"]
    assert [row["thing"] for row in report["disliked"]] == ["probe"]
