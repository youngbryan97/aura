"""Asking for a program to be made again is asking for a thing to exist, and names the act.

LIVE-like 2026-10-05: "Do a clean-room reconstruction of Microsoft Word with
your program DNA engine" reached no capability at all. Rebuilding was not a
verb any class knew, "reconstruction" did not name the act, and the turn ran
under the ceiling that writes nothing.
"""
from __future__ import annotations

from core.intent.artifact_request import names_an_artifact
from core.intent.declared_capability import (
    _act_named_by,
    request_matches_declaration,
    verb_class_of,
)
from core.phases.response_contract import requested_effect_ceiling


def test_remaking_is_an_act_and_its_noun_names_it():
    assert "reconstruct" in verb_class_of("rebuild")
    assert "rebuild" in _act_named_by("reconstruction")
    assert "clone" in _act_named_by("cloning")


def test_a_named_program_made_again_is_a_thing_asked_for():
    for asked in ("Do a clean-room reconstruction of Microsoft Word.", "Rebuild Microsoft Paint for me", "Can you clone Trello?"):
        assert names_an_artifact(asked), asked
        assert requested_effect_ceiling(asked)[0] == "read_write_artifacts", asked
    assert not names_an_artifact("reconstruct what happened yesterday")


def test_the_act_done_as_a_noun_is_still_asked_for_but_not_when_said_of_someone():
    verbs, objects = {"reconstruct"}, {"program", "dna"}
    assert request_matches_declaration("Do a clean-room reconstruction of Word with your program DNA engine", verbs=verbs, objects=objects)
    assert not request_matches_declaration("they do a program dna reconstruction every year", verbs=verbs, objects=objects)


def test_reading_a_named_file_writes_nothing():
    assert requested_effect_ceiling("read /etc/hosts and tell me the first line")[0] != "read_write_artifacts"
