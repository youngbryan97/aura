"""What a call prints, and the files it opens to write, are compared on both sides of a change (Master Todo F16).

The outside check compared a call's value, its arguments, its object and the module state around it. A change that kept
every value and started printing a person's words to the console, or writing a file, kept every one of those. The
runner now adds what each call printed and which files it opened to write to what the call came to.
"""
from __future__ import annotations

from outside_bar.verify_change import verify
from tests.test_outside_bar_verifies_a_change import (  # noqa: F401 - shop is a fixture, found by its own name
    _PRICES,
    _committed,
    _git,
    shop,
)


def test_a_change_that_starts_printing_is_not_held(shop):  # noqa: F811 - the fixture
    noisy = _PRICES.replace("    return table[item]", "    print('asked for', item)\n    return table.get(item)")
    result = verify(shop, _committed(shop, noisy))
    assert not result["held"], result
    assert any(one["kind"] == "another value" and "it printed" in str(one["after"])
               for one in result["counterexamples"]), result


def test_a_change_that_starts_writing_a_file_is_not_held(shop):  # noqa: F811 - the fixture
    writing = _PRICES.replace("    return table[item]",
                              "    with open('asked.txt', 'a') as kept:\n        kept.write(str(item))\n"
                              "    return table.get(item)")
    result = verify(shop, _committed(shop, writing))
    assert not result["held"], result
    assert any("it wrote ['asked.txt: " in str(one["after"]) for one in result["counterexamples"]), result


def test_a_mend_that_prints_and_writes_nothing_new_still_holds(shop):  # noqa: F811 - the fixture
    result = verify(shop, _committed(shop, _PRICES.replace("return table[item]", "return table.get(item)")))
    assert result["held"], result
