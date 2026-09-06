"""Four hundred and three settings, read by hand, six of them two ways."""
from __future__ import annotations

import ast
import json
from pathlib import Path

from core.runtime.what_the_environment_is_asked import (
    every_setting_read,
    how_the_settings_stand,
    settings_the_model_does_not_know,
    settings_whose_defaults_disagree,
)

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "config" / "environment_settings_baseline.json"


def _held() -> dict:
    return json.loads(BASELINE.read_text("utf-8"))


def test_the_number_of_settings_read_two_ways_only_goes_down() -> None:
    """Which default applies then depends on which module asked first."""
    disagreeing = settings_whose_defaults_disagree(str(ROOT))
    assert len(disagreeing) <= _held()["defaults_disagree"], (
        f"{len(disagreeing)} settings are read with more than one default: "
        + ", ".join(disagreeing)
    )


def test_the_scan_finds_the_shape_it_is_about() -> None:
    """A scan that found nothing would report green forever."""
    every = every_setting_read(str(ROOT))
    assert len(every) > 100
    assert sum(one.reads for one in every.values()) > len(every)


def test_a_setting_read_once_cannot_disagree_with_itself() -> None:
    every = every_setting_read(str(ROOT))
    for name, one in every.items():
        if len(one.read_in) == 1:
            assert name not in settings_whose_defaults_disagree(str(ROOT)), name


def test_both_call_shapes_are_read() -> None:
    """os.getenv and os.environ.get are the same question."""
    from core.runtime.what_the_environment_is_asked import _asks_the_environment

    for source in (
        'os.getenv("A_THING", "1")',
        'os.environ.get("A_THING", "1")',
        'environ.get("A_THING")',
    ):
        node = ast.parse(source).body[0].value
        asked = _asks_the_environment(node)
        assert isinstance(asked, ast.Constant) and asked.value == "A_THING", source


def test_a_default_given_by_keyword_is_read_as_a_default() -> None:
    from core.runtime.what_the_environment_is_asked import _default_of

    node = ast.parse('os.getenv("X", default="7")').body[0].value
    assert _default_of(node) == "'7'"
    plain = ast.parse('os.getenv("X")').body[0].value
    assert _default_of(plain) == ""


def test_the_report_says_how_much_the_typed_model_does_not_cover() -> None:
    seen = how_the_settings_stand(str(ROOT))
    assert seen["settings"] == len(every_setting_read(str(ROOT)))
    assert seen["the_model_does_not_know"] <= seen["settings"]
    assert len(settings_the_model_does_not_know(str(ROOT))) == seen[
        "the_model_does_not_know"
    ]


def test_the_disagreements_name_both_defaults() -> None:
    seen = how_the_settings_stand(str(ROOT))
    for name, one in seen["the_disagreements"].items():
        assert len(one["defaults"]) > 1, name
        assert len(one["read_in"]) > 1, name


def test_the_baseline_says_which_way_it_moves() -> None:
    assert "only goes down" in _held()["note"]
