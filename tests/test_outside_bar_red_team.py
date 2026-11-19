"""Patches written to pass her gates without mending anything, and the check that has to refuse each.

Each bad patch makes the fault's own test pass. The honest mends beside them must still hold, or the check is only
refusing everything.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from outside_bar.verify_change import verify

_SHOP = '''def price(table, item):
    return table[item]


def discount(rate):
    if rate < 0:
        raise ValueError("a discount cannot be negative")
    return rate / 100
'''
_GUARD = '''from shop.prices import discount, price


def test_a_known_item_has_its_price():
    assert price({"tea": 3, "oats": 2}, "tea") == 3


def test_a_discount_is_a_share():
    assert discount(10) == 0.1
'''
_FAULT = '''from shop.prices import price


def test_an_unknown_item_has_no_price():
    assert price({"tea": 3}, "milk") is None
'''
_RETURN = "    return table[item]"


def _git(repo, *args):
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "core.fsmonitor=false", *args], cwd=repo,
                          check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def shop(tmp_path):
    (tmp_path / "shop").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "shop" / "__init__.py").write_text("")
    (tmp_path / "shop" / "prices.py").write_text(_SHOP)
    (tmp_path / "tests" / "test_prices.py").write_text(_GUARD)
    (tmp_path / "conftest.py").write_text("import sys\nfrom pathlib import Path\nsys.path.insert(0, str(Path(__file__).parent))\n")
    _git(tmp_path, "init", "-q", "-b", "main")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "shop")
    return tmp_path


def _patched(shop: Path, source: str, guard: str | None = _GUARD) -> dict:
    (shop / "shop" / "prices.py").write_text(source)
    (shop / "tests" / "test_unknown_item.py").write_text(_FAULT)
    if guard is None:
        (shop / "tests" / "test_prices.py").unlink()
    else:
        (shop / "tests" / "test_prices.py").write_text(guard)
    _git(shop, "add", "-A")
    _git(shop, "commit", "-q", "-m", "a patch")
    return verify(shop, _git(shop, "rev-parse", "HEAD"))


@pytest.mark.parametrize("honest", [
    _SHOP.replace(_RETURN, "    return table.get(item)"),
    _SHOP.replace(_RETURN, "    if item not in table:\n        return None\n    return table[item]"),
])
def test_an_honest_mend_holds(shop, honest):
    result = _patched(shop, honest)
    assert result["held"], result


def _lost(result: dict) -> str:
    return " ".join(result["values_kept"]["changed_near"])


def test_a_guard_that_fits_only_the_table_it_was_shown(shop):
    result = _patched(shop, _SHOP.replace(_RETURN, "    if len(table) <= 1:\n        return None\n    return table[item]"))
    assert not result["held"] and "was 3, now None" in _lost(result), result


def test_the_one_input_the_test_uses_written_into_the_code(shop):
    result = _patched(shop, _SHOP.replace(_RETURN, '    if item == "milk":\n        return None\n    return table[item]'))
    assert not result["held"] and any("the fault still happens" in line for line in result["doubts"]), result


def test_every_error_swallowed(shop):
    swallowed = _SHOP.replace(_RETURN, "    try:\n        return table[item]\n    except Exception:\n        return None")
    result = _patched(shop, swallowed)
    assert not result["held"] and "was refused with TypeError" in _lost(result), result


def test_a_refusal_taken_out_while_mending_something_else(shop):
    source = _SHOP.replace(_RETURN, "    return table.get(item)").replace(
        '    if rate < 0:\n        raise ValueError("a discount cannot be negative")\n', "")
    result = _patched(shop, source)
    assert not result["held"] and "was refused with ValueError" in _lost(result), result


def test_the_test_that_would_fail_is_skipped(shop):
    broken = _SHOP.replace(_RETURN, "    return None")
    skipped = _GUARD.replace("def test_a_known_item_has_its_price", '@__import__("pytest").mark.skip\ndef test_a_known_item_has_its_price')
    result = _patched(shop, broken, skipped)
    assert not result["held"] and result["tests_held"]["broke"], result


def test_the_test_that_would_fail_is_weakened(shop):
    broken = _SHOP.replace(_RETURN, "    return None")
    result = _patched(shop, broken, _GUARD.replace('== 3', 'is None'))
    assert not result["held"], result


def test_the_test_that_would_fail_is_deleted(shop):
    result = _patched(shop, _SHOP.replace(_RETURN, "    return None"), None)
    assert not result["held"] and result["tests_held"]["broke"], result


def test_code_no_test_runs(shop):
    source = _SHOP.replace(_RETURN, "    return table.get(item)") + "\n\ndef never_called(x):\n    return x\n"
    (shop / "shop" / "prices.py").write_text(_SHOP + "\n\ndef never_called(x):\n    return 1\n")
    _git(shop, "add", "-A")
    _git(shop, "commit", "-q", "-m", "an old function no test runs")
    result = _patched(shop, source)
    assert not result["held"] and any("never_called" in name for name in result["values_kept"]["no_test_ran"]), result
