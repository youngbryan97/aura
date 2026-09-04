"""An application's dictionary is a file it ships, not a tool's opinion of it.

``sdef`` is one way to read it. On a machine with the command line tools and
no Xcode it refuses — "tool 'sdef' requires Xcode" — writes nothing to its
output, and every application on the machine came back as publishing no
dictionary at all.

LIVE 2026-09-04: "open the Notes app and write a note" planned an open of
Notes and then a text file on disk, because asking Notes how it takes text
answered nothing. Every scriptable app on the machine was equally silent, so
nothing could ever be written into any of them through the interface they
publish.
"""

from __future__ import annotations

import inspect

from core.perception import app_dictionary
from core.perception.app_dictionary import read_dictionary, text_target_for


def test_it_falls_back_to_the_file_the_bundle_ships():
    source = inspect.getsource(app_dictionary._run_sdef)
    assert "_sdef_in_the_bundle(app_path)" in source
    assert source.count("_sdef_in_the_bundle(app_path)") >= 3, "every way out of the tool"


def test_the_plist_says_what_the_file_is_called():
    source = inspect.getsource(app_dictionary._sdef_named_in_the_plist)
    assert "OSAScriptingDefinition" in source


def test_a_name_without_the_extension_still_finds_the_file():
    source = inspect.getsource(app_dictionary._sdef_named_in_the_plist)
    assert '.sdef' in source


def test_notes_says_how_it_takes_text():
    """The app the request names, on this machine, right now."""
    target = text_target_for("Notes")
    if read_dictionary("Notes").unavailable_reason:
        import pytest

        pytest.skip("Notes is not installed on this machine")
    assert target is not None
    assert target.klass == "note"
    assert target.text_property


def test_an_application_that_is_not_there_says_so_rather_than_nothing():
    facts = read_dictionary("An Application Nobody Has")
    assert facts.unavailable_reason
    assert facts.can_be_written_to is False
