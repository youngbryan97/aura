"""A change that holds is told from one that passes its own test by breaking what worked."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from outside_bar.verify_change import values_that_changed, verify

_PRICES = '''def price(table, item):
    return table[item]


def total(table, items):
    return sum(price(table, item) for item in items)
'''
_GUARD = '''from shop.prices import price, total


def test_a_known_item_has_its_price():
    assert price({"tea": 3}, "tea") == 3


def test_a_basket_adds_up():
    assert total({"tea": 3, "oats": 2}, ["tea", "oats"]) == 5
'''
_FAULT = '''from shop.prices import price


def test_an_unknown_item_has_no_price():
    assert price({"tea": 3}, "milk") is None
'''


def _git(repo, *args):
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "core.fsmonitor=false", *args], cwd=repo,
                          check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def shop(tmp_path):
    (tmp_path / "shop").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "shop" / "__init__.py").write_text("")
    (tmp_path / "shop" / "prices.py").write_text(_PRICES)
    (tmp_path / "tests" / "test_prices.py").write_text(_GUARD)
    (tmp_path / "conftest.py").write_text("import sys\nfrom pathlib import Path\nsys.path.insert(0, str(Path(__file__).parent))\n")
    _git(tmp_path, "init", "-q", "-b", "main")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "shop")
    return tmp_path


def _committed(shop: Path, source: str) -> str:
    (shop / "shop" / "prices.py").write_text(source)
    (shop / "tests" / "test_unknown_item.py").write_text(_FAULT)
    _git(shop, "add", "-A")
    _git(shop, "commit", "-q", "-m", "mend")
    return _git(shop, "rev-parse", "HEAD")


def test_the_mend_that_keeps_what_worked_holds(shop):
    result = verify(shop, _committed(shop, _PRICES.replace("return table[item]", "return table.get(item)")))
    assert result["held"], result
    assert result["shows_the_fault"]["newly_passing"] and result["values_kept"]["compared"] > 0
    assert result["reverts_cleanly"]["held"]


def test_the_overfit_mend_that_returns_none_for_a_known_item_is_caught_by_the_tests(shop):
    overfit = _PRICES.replace("    return table[item]", "    if len(table) <= 1:\n        return None\n    return table[item]")
    result = verify(shop, _committed(shop, overfit))
    assert not result["held"] and result["tests_held"]["broke"], result


def test_the_overfit_mend_no_test_reaches_is_caught_on_an_input_near_the_fault(shop):
    """LIVE 17 November 2026: ``if len(prices) <= 1: return None`` passed her gates and lost a price a one-item table holds."""
    (shop / "tests" / "test_prices.py").write_text(_GUARD.replace('price({"tea": 3}, "tea") == 3',
                                                                  'price({"tea": 3, "oats": 2}, "tea") == 3'))
    _git(shop, "add", "-A")
    _git(shop, "commit", "-q", "-m", "a guard with two items")
    overfit = _PRICES.replace("    return table[item]", "    if len(table) <= 1:\n        return None\n    return table[item]")
    result = verify(shop, _committed(shop, overfit))
    assert result["tests_held"]["held"] and not result["held"], result
    assert any("was 3, now None" in line for line in result["values_kept"]["changed_near"]), result


def test_only_the_functions_a_commit_changed_are_compared(shop):
    (shop / "shop" / "prices.py").write_text(_PRICES + "\n\ndef shown(table):\n    return sorted(table)\n")
    _git(shop, "add", "-A")
    _git(shop, "commit", "-q", "-m", "one more function")
    from outside_bar.verify_change import changed_functions

    head = _committed(shop, (shop / "shop" / "prices.py").read_text().replace("return table[item]", "return table.get(item)"))
    assert changed_functions(shop, head, "shop/prices.py") == ["price"]


def test_a_value_no_test_asserts_is_still_caught_when_it_changes(shop):
    (shop / "tests" / "test_prices.py").write_text(_GUARD.replace('assert total({"tea": 3, "oats": 2}, ["tea", "oats"]) == 5',
                                                                  'assert total({"tea": 3, "oats": 2}, ["tea", "oats"]) > 0'))
    _git(shop, "add", "-A")
    _git(shop, "commit", "-q", "-m", "a looser test")
    drifted = _PRICES.replace("return table[item]", "return table.get(item)").replace(
        "return sum(price(table, item) for item in items)", "return sum(price(table, item) for item in items) + 1")
    result = verify(shop, _committed(shop, drifted))
    assert result["tests_held"]["held"] and not result["values_kept"]["held"], result
    assert not result["held"]


def test_a_value_that_moves_by_itself_is_not_held_against_a_change():
    assert values_that_changed({"a": "1", "b": "2"}, {"a": "1", "b": "3"}, {"a": "1", "b": "9"}) == []
    assert values_that_changed({"a": "1"}, {"a": "1"}, {"a": "raised KeyError"}) == ["a"]
    assert values_that_changed({"a": "raised KeyError"}, {"a": "raised KeyError"}, {"a": "1"}) == [], "that is the mend"
    assert values_that_changed({"a": "1"}, {"a": "1"}, {}) == [], "a call not made again says nothing"


def test_it_imports_nothing_of_hers():
    import ast

    allowed = set(sys.stdlib_module_names) | {"outside_bar"}
    for name in ("verify_change.py", "value_witness.py"):
        tree = ast.parse((Path("outside_bar") / name).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else \
                [node.module or ""] if isinstance(node, ast.ImportFrom) else []
            assert not {n.split(".")[0] for n in names} - allowed, name
