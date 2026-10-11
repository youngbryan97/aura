"""Legacy positional corruption is excluded before its values become knowledge."""

from __future__ import annotations

import copy
import json

import pytest

from core.agency.what_meeting_things_does import AVOID, SHOOT
from core.agency.what_she_keeps_of_a_game import INDEXED_TABLES, kept_from
from core.runtime import what_she_learned


def game_record():
    return {"kinds": [{"colour": [10, 20, 30], "size": 10.0}],
            "evidence": {"0": [-6.0, 0.0, 4, 0, 0, 1.0, 3, 3, 1]},
            "physics": {"0": {"accelerations": [[1.0, 2.0]], "speeds": [3.0],
                               "edges": {"top": [3, 0, 0, [12.0], [0.9]]},
                               "meetings": [[0.2, 0.4, 1.1]]}},
            "note": "independent observation"}


def test_a_truncated_evidence_suffix_cannot_silently_reverse_the_measured_stance(monkeypatch):
    original = game_record()
    assert kept_from(original)["meeting"].stance(0) == AVOID
    damaged = copy.deepcopy(original)
    damaged["evidence"]["0"] = original["evidence"]["0"][4:]
    # The real reader without the new structural check reproduces the old
    # failure: default constructor fields make a five-field suffix look valid.
    with monkeypatch.context() as without_contract:
        without_contract.setattr(what_she_learned, "validate_indexed_state", lambda held, **_kw: copy.deepcopy(held))
        assert kept_from(damaged)["meeting"].stance(0) == SHOOT
    assert kept_from(damaged) == {}
    held = what_she_learned.validate_indexed_state(damaged, indexed_tables=INDEXED_TABLES)
    assert held["note"] == original["note"]
    assert not {"kinds", "evidence", "physics"} & held.keys()
    assert damaged["evidence"]["0"] == [0, 1.0, 3, 3, 1]


@pytest.mark.parametrize("damage", ["colour", "edge", "acceleration", "meeting"])
def test_short_positional_shapes_are_rejected_before_their_real_reader_fails(monkeypatch, damage):
    damaged = game_record()
    if damage == "colour":
        damaged["kinds"][0]["colour"] = [10, 20]
    elif damage == "edge":
        damaged["physics"]["0"]["edges"]["top"] = [3, 0, 0, []]
    elif damage == "acceleration":
        damaged["physics"]["0"]["accelerations"] = [[1.0]] * 20
    else:
        damaged["physics"]["0"]["meetings"] = [[0.2, 0.4]] * 5
    with monkeypatch.context() as without_contract:
        without_contract.setattr(what_she_learned, "validate_indexed_state", lambda held, **_kw: copy.deepcopy(held))
        with pytest.raises((ValueError, IndexError)):
            back = kept_from(damaged)
            if damage == "colour":
                back["kinds"][0].like((10, 20, 30), 10.0)
            elif damage == "acceleration":
                back["physics"].gravity(0)
            elif damage == "meeting":
                back["physics"].after_meeting(0, 0.0, (1.0, 2.0), False)
    assert kept_from(damaged) == {}


def sensor_schema():
    return {"table": ["sensors"], "keyed_by": [["calibration"]], "references": [["selected_sensor"]],
            "histories": [["calibration", "*", "samples"]],
            "structures": [
                {"path": ["sensors", "*"], "type": "mapping", "required_fields": ["name", "axes"]},
                {"path": ["sensors", "*", "axes"], "length": 3, "required": True},
                {"path": ["calibration", "*"], "type": "mapping", "required_fields": ["record"]},
                {"path": ["calibration", "*", "record"], "length": 6, "required": True},
                {"path": ["calibration", "*", "samples"], "type": "list"},
                {"path": ["calibration", "*", "samples", "*"], "length": 2},
            ]}


def worksheet_schema():
    return {"table": ["columns"], "keyed_by": [["formats"]], "references": [["selected_column"]],
            "histories": [["formats", "*", "edits"]],
            "structures": [
                {"path": ["columns", "*"], "type": "mapping", "required_fields": ["name", "span"]},
                {"path": ["columns", "*", "span"], "length": 2, "required": True},
                {"path": ["formats", "*"], "type": "mapping", "required_fields": ["record"]},
                {"path": ["formats", "*", "record"], "length": 4, "required": True},
                {"path": ["formats", "*", "edits"], "type": "list"},
                {"path": ["formats", "*", "edits", "*"], "length": 2},
            ]}


@pytest.mark.parametrize("domain", ["sensors", "worksheet"])
def test_fit_reload_retains_positional_meaning_and_complete_history_samples(tmp_path, monkeypatch, domain):
    monkeypatch.setattr(what_she_learned, "_KEPT_IN", tmp_path)
    monkeypatch.setattr(what_she_learned, "_MOST_KEPT", 4_500)
    if domain == "sensors":
        schema = sensor_schema()
        record = {"sensors": [{"name": f"sensor {n}", "axes": [n, n + 1, n + 2]} for n in range(60)],
                  "calibration": {str(n): {"record": list(range(n, n + 6)),
                                            "samples": [[i, n] for i in range(40)]} for n in range(60)},
                  "selected_sensor": 59, "note": "sensor observation"}
        table, payloads, selected = "sensors", "calibration", "selected_sensor"
    else:
        schema = worksheet_schema()
        record = {"columns": [{"name": f"column {n}", "span": [n, n + 1]} for n in range(60)],
                  "formats": {str(n): {"record": [f"header {n}", "number", n, True],
                                        "edits": [["editor", i] for i in range(40)]} for n in range(60)},
                  "selected_column": 59, "note": "worksheet observation"}
        table, payloads, selected = "columns", "formats", "selected_column"
    source = copy.deepcopy(record)
    assert what_she_learned.remember(domain, record, indexed_tables=[schema])
    recalled = what_she_learned.recall(domain, indexed_tables=[schema])
    assert 0 < len(recalled[table]) < 60
    assert recalled[table][recalled[selected]]["name"].endswith(" 59")
    assert recalled["note"] == source["note"]
    for index, payload in recalled[payloads].items():
        number = int(recalled[table][int(index)]["name"].split()[-1])
        if domain == "sensors":
            assert recalled[table][int(index)]["axes"] == [number, number + 1, number + 2]
            assert payload["record"] == list(range(number, number + 6))
            assert all(len(sample) == 2 and sample[1] == number for sample in payload["samples"])
            assert payload["samples"][-1] == [39, number]
        else:
            assert recalled[table][int(index)]["span"] == [number, number + 1]
            assert payload["record"] == [f"header {number}", "number", number, True]
            assert all(len(edit) == 2 and edit[0] == "editor" for edit in payload["edits"])
            assert payload["edits"][-1] == ["editor", 39]
    assert len(next(tmp_path.glob("*.json")).read_text()) <= 4_500
    assert record == source


def test_damaged_closure_is_excluded_without_rewriting_legacy_evidence_or_other_closures(tmp_path, monkeypatch):
    monkeypatch.setattr(what_she_learned, "_KEPT_IN", tmp_path)
    damaged = {"sensors": [{"name": "short axes", "axes": [1, 2]}], "calibration": {"0": {"record": [0] * 6}},
               "selected_sensor": 0, "columns": [{"name": "intact", "span": [2, 3]}],
               "formats": {"0": {"record": ["heading", "text", 12, True]}}, "selected_column": 0,
               "note": "independent observation"}
    path = tmp_path / "legacy.json"
    serialized = json.dumps(damaged)
    path.write_text(serialized)
    degradations = []
    monkeypatch.setattr(what_she_learned, "record_degradation", lambda *args, **kw: degradations.append((args, kw)))
    held = what_she_learned.recall("legacy", indexed_tables=[sensor_schema(), worksheet_schema()])
    assert not {"sensors", "calibration", "selected_sensor"} & held.keys()
    assert held["columns"] == damaged["columns"] and held["formats"] == damaged["formats"]
    assert held["selected_column"] == 0 and held["note"] == damaged["note"]
    assert path.read_text() == serialized
    assert len(degradations) == 1 and "wrong fixed length" in str(degradations[0][0][1])


def test_current_schema_replaces_repeated_wider_stored_declarations_without_validating_retired_paths():
    current = sensor_schema()
    old = {"table": ["sensors"], "keyed_by": [["calibration"], ["retired_payloads"]],
           "references": [["retired_selection"]], "structures": False}
    record = {"sensors": [{"name": "intact", "axes": [1, 2, 3]}],
              "calibration": {"0": {"record": [0] * 6}}, "selected_sensor": 0,
              "retired_payloads": {"99": "unknown old association"}, "retired_selection": 99,
              "columns": [{"name": "independent", "span": [3, 4]}],
              "formats": {"0": {"record": ["heading", "text", 12, True]}}, "selected_column": 0,
              "_indexed_tables": [old, {**old, "references": [["another_retired_selection"]]}, worksheet_schema()]}
    held = what_she_learned.validate_indexed_state(record, indexed_tables=[current])
    assert held["sensors"] == record["sensors"] and held["columns"] == record["columns"]
    assert held["_indexed_tables"] == [current, worksheet_schema()]
    assert held["retired_payloads"] == record["retired_payloads"] and held["retired_selection"] == 99
    assert all(["retired_selection"] not in relation.get("references", []) for relation in held["_indexed_tables"])
    assert record["_indexed_tables"][0]["structures"] is False


@pytest.mark.parametrize("optional", [None, {}])
def test_empty_or_missing_optional_groups_do_not_invalidate_complete_entries(optional):
    record = {"sensors": [{"name": "intact", "axes": [1, 2, 3]}], "selected_sensor": 0}
    if optional is not None:
        record["calibration"] = optional
    held = what_she_learned.validate_indexed_state(record, indexed_tables=[sensor_schema()])
    assert held["sensors"] == record["sensors"]
    record["calibration"] = None
    assert what_she_learned.validate_indexed_state(record, indexed_tables=[sensor_schema()])["sensors"] == record["sensors"]


@pytest.mark.parametrize("entry", [{"name": "missing axes"}, {"axes": [1, 2, 3]}, []])
def test_required_per_entry_fields_cannot_be_satisfied_by_an_empty_wildcard_match(entry):
    held = what_she_learned.validate_indexed_state({"sensors": [entry], "note": "independent"},
                                                  indexed_tables=[sensor_schema()])
    assert "sensors" not in held and held["note"] == "independent"


@pytest.mark.parametrize("payload", [7, "text", [[1]], {"one": [1, 2, 3]}, [None]])
def test_malformed_wildcard_containers_and_samples_cannot_pass_vacuously(payload):
    schema = [{"table": ["records"], "structures": [{"path": ["records", "*", "samples", "*"], "length": 2}]}]
    held = what_she_learned.validate_indexed_state({"records": [{"samples": payload}], "note": "independent"},
                                                  indexed_tables=schema)
    assert "records" not in held and held["note"] == "independent"


@pytest.mark.parametrize("change", [
    {"length": True}, {"length": -1}, {"length": 1.5}, {"length": "3"},
    {"type": "tuple"}, {"type": "mapping", "length": 3}, {"required": 1},
    {"required_fields": ["duplicate", "duplicate"]}, {"required_fields": ["field"]},
    {"min_length": 3}, {"path": ["records"]}, {"path": ["*", "axes"]},
    {"path": ["other", "axes"]},
])
def test_invalid_shape_declarations_fail_closed(change):
    structure = {"path": ["records", "*", "axes"], "length": 3, **change}
    schema = [{"table": ["records"], "structures": [structure]}]
    assert what_she_learned.validate_indexed_state({"records": [{"axes": [1, 2, 3]}]}, indexed_tables=schema) == {}


@pytest.mark.parametrize("second", [
    {"path": ["records", "*", "axes"], "length": 3},
    {"path": ["records", "*", "axes"], "length": 2},
    {"path": ["records", "0", "axes"], "type": "mapping"},
])
def test_overlapping_structure_patterns_cannot_assign_two_contracts_to_the_same_node(second):
    schema = [{"table": ["records"], "structures": [{"path": ["records", "*", "axes"], "length": 3}, second]}]
    assert what_she_learned.validate_indexed_state({"records": [{"axes": [1, 2, 3]}]}, indexed_tables=schema) == {}


@pytest.mark.parametrize("history", [["records", "*", "axes"], ["records", "*", "*"]])
def test_fixed_vectors_cannot_also_match_a_trimmable_history(history):
    schema = [{"table": ["records"], "histories": [history],
               "structures": [{"path": ["records", "*", "axes"], "length": 9}]}]
    record = {"records": [{"axes": list(range(9))}], "_indexed_tables": schema}
    assert what_she_learned.validate_indexed_state(record) == {}
    with pytest.raises(ValueError, match="trimmable history"):
        what_she_learned._fitted(record, 100)


def test_a_structure_cannot_borrow_another_indexed_closure():
    schema = [{"table": ["first"], "structures": [{"path": ["second", "*", "vector"], "length": 2}]},
              {"table": ["second"]}]
    assert what_she_learned.validate_indexed_state({"first": [], "second": []}, indexed_tables=schema) == {}


def test_shape_claim_runs_its_registered_measurement_and_detects_a_disabled_validator(monkeypatch):
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import Evidence, ValidationSuite

    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    check = next(test for test in suite.tests() if test.name == "retained_structures_reject_damaged_closures")
    assert check.predict(None) is True
    claim = next(claim for claim in suite.claims() if claim.test == check.name)
    assert claim.evidence is Evidence.MEASURED_SYNTHETIC
    monkeypatch.setattr(what_she_learned, "_check_structure", lambda *_args: None)
    with pytest.raises(AssertionError, match="damaged positional vector"):
        check.predict(None)
