""""Open it in the browser" is a sentence, not the name of a program.

Handed over whole, it is looked up as though somebody had installed something
called "it in the browser", and the run stops one step from the thing it was
asked to do. LIVE 2026-08-27: exactly that, on the second attempt at a world
she had never seen, having got past the parsing that stopped the first.

"The browser" names no product either. Every browser is a candidate, and
whichever one is really installed is the one that answers — which is a fact
about the machine rather than about the words, so nothing here decides it.
"""

from __future__ import annotations

import pytest

from core.runtime.watched_goal import apps_named_in


# ── a sentence about an app ──────────────────────────────────────────────

def test_the_phrase_that_stopped_a_live_run():
    assert "Safari" in apps_named_in("it in the browser")


def test_a_bare_browser_could_be_any_of_them():
    named = apps_named_in("the browser")
    assert len(named) > 1
    assert "Google Chrome" in named and "Firefox" in named


def test_a_browser_named_outright_is_the_only_candidate():
    assert apps_named_in("open it in Safari") == ("Safari",)
    assert apps_named_in("Chrome") == ("Google Chrome",)


def test_the_name_is_what_comes_before_the_preposition():
    assert apps_named_in("Preview with the file open") == ("Preview",)


@pytest.mark.parametrize("said", ["it", "this", "that", "the window", "the page"])
def test_a_word_that_names_no_program_names_none(said):
    assert apps_named_in(said) == ()


def test_a_name_that_already_failed_is_not_offered_again():
    """There is nothing to be gained by looking up the same string twice."""
    assert apps_named_in("Notes") == ()


def test_nothing_is_nothing():
    assert apps_named_in("") == ()
    assert apps_named_in("   ") == ()
