"""Fitting a record must preserve the subjects and shape of retained evidence."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from core.runtime import what_she_learned

SCHEMA = [{"table": ["symbols"], "keyed_by": [["evidence"], ["physics"]],
           "references": [["selected", "symbol"]]}]


@pytest.fixture
def kept_in(tmp_path, monkeypatch):
    monkeypatch.setattr(what_she_learned, "_KEPT_IN", tmp_path)
    return tmp_path


def _record():
    return {
        "symbols": [{"name": f"object {n}", "colour": [n, n + 1, n + 2]} for n in range(80)],
        "evidence": {str(n): [n] * 9 for n in range(80)},
        "physics": {str(n): {"subject": f"object {n}", "speeds": list(range(50))} for n in range(80)},
        "selected": {"symbol": 79},
        "note": "measured in this world",
    }


def _same_relationships(record):
    for index, evidence in record["evidence"].items():
        subject = record["symbols"][int(index)]
        assert len(evidence) == 9
        assert subject["name"] == f"object {evidence[0]}"
        assert subject["colour"] == [evidence[0], evidence[0] + 1, evidence[0] + 2]
        assert record["physics"][index]["subject"] == subject["name"]
    assert record["symbols"][record["selected"]["symbol"]]["name"] == "object 79"


def test_fitting_and_roundtrip_preserve_referenced_subjects_and_fixed_vectors(kept_in, monkeypatch):
    monkeypatch.setattr(what_she_learned, "_MOST_KEPT", 3_000)
    original = _record()
    assert what_she_learned.remember("a measured world", original, indexed_tables=SCHEMA)
    recalled = what_she_learned.recall("a measured world")
    assert 0 < len(recalled["symbols"]) < len(original["symbols"])
    assert recalled["note"] == original["note"]
    _same_relationships(recalled)
    assert all(values["speeds"] == list(range(50)) for values in recalled["physics"].values())
    assert len(next(kept_in.glob("*.json")).read_text()) <= 3_000
    assert original == _record(), "fitting changed the caller's live knowledge"


def test_declared_histories_can_shorten_while_evidence_vectors_hold(kept_in, monkeypatch):
    monkeypatch.setattr(what_she_learned, "_MOST_KEPT", 12_000)
    schema = [{**SCHEMA[0], "histories": [["physics", "*", "speeds"]]}]
    assert what_she_learned.remember("a measured world", _record(), indexed_tables=schema)
    recalled = what_she_learned.recall("a measured world")
    _same_relationships(recalled)
    assert any(len(values["speeds"]) < 50 for values in recalled["physics"].values())
    assert all(values["speeds"][-1] == 49 for values in recalled["physics"].values())


def test_unreferenced_rows_are_released_before_measured_subjects():
    record = {"symbols": [{"name": f"object {n}"} for n in range(50)],
              "evidence": {"3": [3] * 9}, "physics": {"3": {"subject": "object 3"}},
              "_indexed_tables": SCHEMA}
    fitted, _released = what_she_learned._fitted(record, 400)
    assert fitted["evidence"]
    for index, values in fitted["evidence"].items():
        assert fitted["symbols"][int(index)]["name"] == f"object {values[0]}"


def test_a_damaged_legacy_table_excludes_its_closure_but_keeps_other_knowledge(kept_in, monkeypatch):
    damaged = {"symbols": [{"name": "a shifted row"}],
               "evidence": {"0": [0] * 9, "79": [79] * 9},
               "physics": {"79": {"subject": "object 79"}}, "selected": {"symbol": 79},
               "note": "independent finding"}
    path = kept_in / f"{what_she_learned.named('a measured world')}.json"
    serialized = json.dumps(damaged)
    path.write_text(serialized)
    degradations = []
    monkeypatch.setattr(what_she_learned, "record_degradation", lambda *args, **kw: degradations.append((args, kw)))
    recalled = what_she_learned.recall("a measured world", indexed_tables=SCHEMA)
    assert recalled["note"] == "independent finding"
    assert not {"symbols", "evidence", "physics"} & recalled.keys()
    assert "symbol" not in recalled.get("selected", {})
    assert degradations and "excluded damaged indexed state" in degradations[0][1]["action"]
    assert path.read_text() == serialized, "reading silently rewrote old evidence"


def test_missing_table_cannot_validate_a_zero_reference():
    assert "symbol" not in what_she_learned.validate_indexed_state(
        {"selected": {"symbol": 0}}, indexed_tables=SCHEMA).get("selected", {})


def test_oversized_selected_state_is_released_as_a_whole_relation():
    record = {"symbols": [{"name": "x" * 2_000}], "evidence": {"0": [0] * 9},
              "physics": {"0": {"subject": "x" * 2_000}}, "selected": {"symbol": 0},
              "note": "independent", "_indexed_tables": SCHEMA}
    fitted, released = what_she_learned._fitted(record, 300)
    assert fitted["note"] == "independent"
    assert not {"symbols", "evidence", "physics", "selected"} & fitted.keys()
    assert released


def test_an_index_table_cannot_be_declared_as_a_trimmable_history():
    record = {"symbols": ["a", "b"], "_indexed_tables": [{"table": ["symbols"], "histories": [["symbols"]]}]}
    assert what_she_learned.validate_indexed_state(record) == {}


@pytest.mark.parametrize("history", [["*"], ["symbols"]])
def test_history_patterns_cannot_retarget_a_referenced_table(history):
    record = {"symbols": [f"object {n}" for n in range(8)], "findings": {"0": [0] * 9}, "selected": 0,
              "_indexed_tables": [{"table": ["symbols"], "keyed_by": [["findings"]],
                                   "references": [["selected"]], "histories": [history]}]}
    original = json.loads(json.dumps(record))
    with pytest.raises(ValueError, match="history"):
        what_she_learned._fitted(record, len(json.dumps(record)) - 1)
    assert what_she_learned.validate_indexed_state(record) == {}
    assert record == original


@pytest.mark.parametrize("history", [["second", "symbols"], ["second", "symbols", "*", "events"], ["*"]])
def test_a_history_cannot_be_borrowed_from_another_relation(history):
    record = {"first": {"symbols": ["a"]}, "second": {"symbols": ["b"]},
              "_indexed_tables": [{"table": ["first", "symbols"], "histories": [history]},
                                   {"table": ["second", "symbols"]}]}
    assert what_she_learned.validate_indexed_state(record) == {}


def test_a_history_cannot_match_an_ancestor_of_a_nested_table():
    record = {"graph": {"symbols": ["a"]},
              "_indexed_tables": [{"table": ["graph", "symbols"], "histories": [["graph"]]}]}
    assert what_she_learned.validate_indexed_state(record) == {}


@pytest.mark.parametrize("budget", [20, 1])
def test_null_fields_and_impossible_budgets_finish_or_fail_within_a_bound(budget):
    record = {"note": None, "_indexed_tables": [{"table": ["symbols"]}]}
    program = "\n".join([
        "import json",
        "from core.runtime import what_she_learned",
        f"record = {record!r}",
        f"budget = {budget}",
        "try:",
        "    fitted, released = what_she_learned._fitted(record, budget)",
        "except ValueError as why:",
        "    assert budget < 2 and 'empty record' in str(why)",
        "else:",
        "    assert len(json.dumps(fitted)) <= budget",
        "    assert any(line.startswith('note (') for line in released)",
        "assert record['note'] is None",
    ])
    child = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True, timeout=10)
    assert child.returncode == 0, child.stdout + child.stderr


def test_releasing_a_nested_relation_keeps_independent_siblings():
    schema = [{"table": ["graph", "symbols"], "keyed_by": [["graph", "findings"]],
               "references": [["graph", "selected"]]}]
    record = {"graph": {"symbols": ["x" * 2_000], "findings": {"0": [0] * 9},
                        "selected": 0, "independent": "still measured"}, "_indexed_tables": schema}
    fitted, _released = what_she_learned._fitted(record, 350)
    assert fitted["graph"] == {"independent": "still measured"}


def test_compaction_preserves_two_independent_reference_graphs():
    schema = [{"table": [name, "symbols"], "keyed_by": [[name, "findings"]]}
              for name in ("first", "second")]
    record = {name: {"symbols": [f"{name} object {n}" for n in range(30)],
                     "findings": {str(n): [f"{name} object {n}"] * 9 for n in range(30)}}
              for name in ("first", "second")}
    record["_indexed_tables"] = schema
    fitted, _released = what_she_learned._fitted(record, 2_000)
    for name in ("first", "second"):
        assert fitted[name]["findings"]
        for index, values in fitted[name]["findings"].items():
            assert len(values) == 9 and fitted[name]["symbols"][int(index)] == values[0]


def test_the_registered_relationship_invariant():
    assert what_she_learned._indexed_knowledge_invariant() == ()
