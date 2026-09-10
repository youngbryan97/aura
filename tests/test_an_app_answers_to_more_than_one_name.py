"""She could see the application and could never start it.

An application has more than one name and they are not interchangeable. The
bundle on disk is 2048.app. The process it runs as calls itself "2048 Game".
The window server reports the second one, so that is the name she watches it
under, learns it under, and writes down.

The resolver only ever knew the names of directories. So she found the game,
recalled everything she had learned about playing it, and got "'2048 Game'
would not come to the front and would not start" after nought moves. The
operating system agreed with her: "Unable to find application named '2048
Game'" — for an application sitting in /Applications the whole time.

LIVE 2026-09-05, asked to open the game and play it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.capabilities.host_automation import (
    installed_application_names,
    resolve_application_name,
    what_can_start,
)

HERE = Path("/Applications/2048.app")


def test_a_name_from_a_running_process_can_start_the_application():
    """The whole defect, in one line."""
    if not HERE.exists():
        pytest.skip("2048.app is not installed on this machine")
    assert what_can_start("2048 Game") == "2048"


def test_the_folder_name_still_works():
    if not HERE.exists():
        pytest.skip("2048.app is not installed on this machine")
    assert what_can_start("2048") == "2048"


def test_it_resolves_through_the_ordinary_path():
    if not HERE.exists():
        pytest.skip("2048.app is not installed on this machine")
    said = resolve_application_name("2048 Game")
    assert said.ok
    assert said.resolved == "2048"


def test_a_name_nothing_answers_to_is_still_refused():
    """Knowing more names must not turn a miss into a guess."""
    assert what_can_start("no application has ever been called this") == ""
    assert not resolve_application_name("no application has ever been called this").ok


def test_the_loose_matching_it_already_did_is_untouched():
    """"The Notes app" has to keep working."""
    if "Notes" not in installed_application_names():
        pytest.skip("Notes is not installed on this machine")
    assert resolve_application_name("Note app").resolved == "Notes"


def test_every_installed_application_can_be_started_by_its_folder_name():
    """The name that always works must never be shadowed by another one."""
    for name in installed_application_names()[:40]:
        assert what_can_start(name) == name, name


def test_asking_about_nothing_answers_nothing():
    assert what_can_start("") == ""
    assert what_can_start("   ") == ""
