"""What she built can be changed as asked and used for what it is for, and the request says which build.

LIVE-like 2026-10-05: "add a dark theme to the word processor you built" ran
under the ceiling that writes nothing and reached no capability; and a build
proved by being used ("then prove it works by writing a letter in it and
exporting it to my Desktop") had nowhere to keep what it exported.
"""
from __future__ import annotations

import json
import os
import time

from core.capabilities.where_downloads_go import _free_name
from core.phases.response_contract import requested_effect_ceiling
from core.rebuilding.rebuilding_a_program import the_build_meant
from core.skills.using_a_program import the_folder_named_in, what_to_do_with_it


def test_the_task_is_read_from_the_request():
    assert what_to_do_with_it("Rebuild Word, then prove it works by writing a letter in it and exporting it to my Desktop.") == (
        "writing a letter in it and exporting it to my Desktop"
    )
    assert what_to_do_with_it("Use the word processor you built to write a story.") == "write a story"
    assert what_to_do_with_it("Build me a habit tracker.") == ""


def test_the_folder_a_person_names_is_where_the_file_goes(tmp_path):
    assert the_folder_named_in("export it to my Desktop") == __import__("pathlib").Path.home() / "Desktop"
    assert the_folder_named_in(f"save it in {tmp_path}") == tmp_path
    assert the_folder_named_in("export it") is None
    (tmp_path / "letter.docx").write_text("x")
    assert _free_name(tmp_path / "letter.docx").name == "letter (2).docx"


def test_the_build_meant_is_the_one_named_else_the_newest(tmp_path):
    for name in ("inkwell", "tally-board"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "program.json").write_text(json.dumps({"title": name, "parts": []}))
    old = time.time() - 100
    os.utime(tmp_path / "inkwell" / "program.json", (old, old))
    assert the_build_meant("add a dark theme to inkwell", [tmp_path]).name == "inkwell"
    assert the_build_meant("add a dark theme to the app you built", [tmp_path]).name == "tally-board"
    assert the_build_meant("anything", [tmp_path / "nowhere"]) is None


def test_asking_for_her_build_to_change_is_asking_for_an_effect():
    assert requested_effect_ceiling("Add a dark theme to the word processor you built.")[0] == "read_write_artifacts"
    assert requested_effect_ceiling("The app you made needs to save as PDF.")[0] == "read_write_artifacts"
    assert requested_effect_ceiling("what did you build yesterday?")[0] != "read_write_artifacts"
