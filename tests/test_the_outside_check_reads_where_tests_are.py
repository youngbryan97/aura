"""The outside check guards a change with the repository's own tests, and says what a cut-short run did not reach (F16).

LIVE 21 November 2026: her draft for the explore_knowledge impulse, 44 lines in one method of
core/orchestrator/mixins/autonomy.py, was refused after 7,212 s. Its guards were every test file anywhere in her tree that
named the module, scratch/test_full_boot.py and archive/one_off_scripts/test_boot_validation.py among them, outside the
folders her pyproject.toml gives pytest. Each of the three runs was cut short at its 2,400 s bound, and three tests the
run on her change had not reached by then were reported as "passed before and fails now". Each passes with her change.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from outside_bar import verify_change
from outside_bar.verify_change import declared_test_roots, verify

pytestmark = pytest.mark.unit

_PRICES = '''def price(table, item):
    return table[item]
'''
_GUARD = '''from shop.prices import price


def test_a_known_item_has_its_price():
    assert price({"tea": 3}, "tea") == 3
'''
_OLD_SCRIPT = '''import pytest

from shop.prices import price


def test_an_unknown_item_raises():
    with pytest.raises(KeyError):
        price({"tea": 3}, "milk")
'''
_FAULT = '''from shop.prices import price


def test_an_unknown_item_has_no_price():
    assert price({"tea": 3}, "milk") is None
'''


def _git(repo, *args):
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "core.fsmonitor=false", *args], cwd=repo,
                          check=True, capture_output=True, text=True).stdout.strip()


def _shop(root: Path, configuration: str | None) -> Path:
    for folder in ("shop", "tests", "scratch"):
        (root / folder).mkdir()
    (root / "shop" / "__init__.py").write_text("")
    (root / "shop" / "prices.py").write_text(_PRICES)
    (root / "tests" / "test_prices.py").write_text(_GUARD)
    # An old script beside the suite, which the mend below breaks.
    (root / "scratch" / "test_old_prices.py").write_text(_OLD_SCRIPT)
    (root / "conftest.py").write_text("import sys\nfrom pathlib import Path\nsys.path.insert(0, str(Path(__file__).parent))\n")
    if configuration is not None:
        (root / "pyproject.toml").write_text(configuration)
    _git(root, "init", "-q", "-b", "main")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "shop")
    (root / "shop" / "prices.py").write_text(_PRICES.replace("return table[item]", "return table.get(item)"))
    (root / "tests" / "test_unknown_item.py").write_text(_FAULT)
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "mend")
    return root


@pytest.mark.parametrize(("files", "roots"), [
    ({"pyproject.toml": '[tool.pytest.ini_options]\ntestpaths = ["tests"]\n'}, ["tests"]),
    ({"pyproject.toml": '[tool.pytest]\ntestpaths = ["tests", "more/tests"]\n'}, ["more/tests", "tests"]),
    ({"pyproject.toml": '[tool.black]\nline-length = 100\n'}, None),
    ({"pytest.ini": "", "pyproject.toml": '[tool.pytest.ini_options]\ntestpaths = ["tests"]\n'}, None),
    ({"pytest.ini": "[pytest]\ntestpaths = more/tests\n", "pyproject.toml": '[tool.pytest.ini_options]\ntestpaths = ["tests"]\n'},
     ["more/tests"]),
    ({"setup.cfg": "[tool:pytest]\ntestpaths =\n    tests\n    more/tests\n"}, ["more/tests", "tests"]),
    ({"tox.ini": "[pytest]\ntestpaths = te*\n"}, ["tests"]),
    ({"pyproject.toml": '[tool.pytest.ini_options]\ntestpaths = ["nowhere"]\n'}, None),
    ({}, None),
], ids=["pyproject", "native-table", "no-pytest-table", "empty-pytest-ini-decides", "pytest-ini-first", "setup-cfg",
        "glob", "missing-folder", "none"])
def test_where_its_tests_are_is_read_as_pytest_reads_it(tmp_path, files, roots):
    for folder in ("tests", "more/tests"):
        (tmp_path / folder).mkdir(parents=True)
    for name, text in files.items():
        (tmp_path / name).write_text(text)
    assert declared_test_roots(tmp_path) == roots


def test_a_test_file_outside_the_declared_suite_does_not_guard_a_change(tmp_path):
    shop = _shop(tmp_path, '[tool.pytest.ini_options]\ntestpaths = ["tests"]\n')
    result = verify(shop, _git(shop, "rev-parse", "HEAD"))
    assert result["tests_held"]["held"], result
    assert not any("test_old_prices" in node for node in result["tests_held"]["broke"])


def test_with_no_suite_declared_a_test_file_anywhere_still_guards_it(tmp_path):
    shop = _shop(tmp_path, None)
    result = verify(shop, _git(shop, "rev-parse", "HEAD"))
    assert result["tests_held"]["broke"] == ["scratch/test_old_prices.py::test_an_unknown_item_raises"], result
    assert result["verdict"] == "REFUSED"


def test_a_test_a_cut_short_run_never_reached_is_not_said_to_fail(tmp_path, monkeypatch):
    shop = _shop(tmp_path, '[tool.pytest.ini_options]\ntestpaths = ["tests"]\n')
    real, runs = verify_change.run_tests, []

    def cut_short_on_the_change(checkout, tests, work, *args, **kwargs):
        result = real(checkout, tests, work, *args, **kwargs)
        runs.append(result)
        if len(runs) == 3:  # the guarding tests on this change, killed at the bound before any test said its verdict
            return {**result, "verdicts": {}, "finished": False}
        return result

    monkeypatch.setattr(verify_change, "run_tests", cut_short_on_the_change)
    result = verify(shop, _git(shop, "rev-parse", "HEAD"))
    assert runs[0]["verdicts"], "the guarding tests passed on the code before"
    assert result["tests_held"]["broke"] == [], result
    assert "a run did not say it ended, so it is evidence of nothing: the guarding tests on this change" in result["must_fix"]
    assert result["verdict"] == "REFUSED"
