"""Source occurrences, saved references, and pairing retain their identities."""

from __future__ import annotations

import hashlib
import json
import sys
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from tools import run_g12_bbeh
from tools.benchmark_case_identity import reference_identity, source_cases


def catalog(records):
    return source_cases(records, legacy_id=lambda row: row["input"], reference=lambda row: row["target"])


def test_distinct_source_occurrences_keep_distinct_ids_even_with_identical_references() -> None:
    cases = catalog([{"input": "same", "target": "left"}, {"input": "unique", "target": "right"},
                     {"input": "same", "target": "left"}]).cases
    assert [case.id for case in cases] == ["same--case-0", "unique", "same--case-2"]
    assert cases[0].reference_sha256 == cases[2].reference_sha256
    assert len({case.id for case in cases}) == 3


def test_generated_ids_cannot_collide_with_an_existing_unique_legacy_id() -> None:
    with pytest.raises(ValueError, match="source case IDs are not unique"):
        catalog([{"input": key, "target": "ref"} for key in ("same", "same", "same--case-0")])


def test_legacy_ambiguity_is_refused_even_when_its_target_would_disambiguate(tmp_path) -> None:
    cases = catalog([{"input": "same", "target": "left"}, {"input": "same", "target": "right"}])
    legacy = {"id": "same", "target": "left"}
    path = tmp_path / "same.json"
    payload = json.dumps(legacy)
    path.write_text(payload)
    with pytest.raises(ValueError, match="ambiguous legacy case ID"):
        cases.saved_rows(tmp_path)
    assert path.read_text() == payload


def test_a_unique_legacy_row_is_reusable_only_with_the_same_reference() -> None:
    cases = catalog([{"input": "unique", "target": 1}])
    assert cases.validate_row({"id": "unique", "target": 1}).source_ordinal == 0
    for reference in (True, "1", 2):
        with pytest.raises(ValueError, match="reference does not match"):
            cases.validate_row({"id": "unique", "target": reference})


def test_saved_source_ordinal_and_reference_digest_must_agree() -> None:
    cases = catalog([{"input": "same", "target": "left"}, {"input": "same", "target": "right"}])
    first = cases.cases[0]
    row = {"id": first.id, "target": "left", **first.metadata()}
    assert cases.validate_row(row) is first
    for changed, message in (({"source_ordinal": 1}, "source ordinal"),
                             ({"source_ordinal": False}, "source ordinal"),
                             ({"reference_sha256": reference_identity("right")}, "reference identity")):
        with pytest.raises(ValueError, match=message):
            cases.validate_row({**row, **changed})
    row.pop("source_ordinal")
    with pytest.raises(ValueError, match="no saved source ordinal"):
        cases.validate_row(row)


def mini_source(tmp_path, monkeypatch, examples):
    path = tmp_path / "mini.json"
    raw = json.dumps({"examples": examples}).encode()
    path.write_bytes(raw)
    monkeypatch.setattr(run_g12_bbeh, "BBEH_MINI_SHA256", hashlib.sha256(raw).hexdigest())
    tasks = tmp_path / "tasks"
    for name in ("a", "b"):
        selected = [example for example in examples if example.get("task") == name]
        directory = tasks / f"bbeh_{name}"
        directory.mkdir(parents=True)
        (directory / "task.json").write_text(json.dumps({"examples": selected}))
    return path, tasks


def invoke_runner(monkeypatch, *, mini, tasks, output, model, extra=()):
    monkeypatch.setattr(sys, "argv", ["run_g12_bbeh.py", "--mini", str(mini), "--tasks", str(tasks),
                                     "--output", str(output), "--model", str(model), "--batch", "8", *extra])
    return run_g12_bbeh.main()


def test_task_selection_and_out_of_order_completion_preserve_complete_source_ids(tmp_path, monkeypatch) -> None:
    from core.runtime import model_lane_control

    examples = [{"input": "shared", "target": "left", "task": "a"},
                {"input": "other", "target": "right", "task": "b"},
                {"input": "shared", "target": "right", "task": "a"}]
    mini, tasks = mini_source(tmp_path, monkeypatch, examples)
    model = tmp_path / "fake-model"
    model.mkdir()
    output = tmp_path / "output"
    deliveries = []

    def fake_decode(_model, _tokenizer, conversations, *, max_tokens, width, on_record):
        deliveries.append((conversations, max_tokens, width))
        for index in reversed(range(len(conversations))):
            on_record(index, {"public_text": "The answer is: left", "termination": "stop", "generated_tokens": 1,
                              "seconds": 0.1})

    monkeypatch.setitem(sys.modules, "mlx_lm", SimpleNamespace(load=lambda _path: (object(), object())))
    monkeypatch.setattr(model_lane_control, "standalone_model_lane", lambda **_kwargs: nullcontext())
    monkeypatch.setitem(sys.modules, "tools.g12_batched", SimpleNamespace(decode_stream=fake_decode))
    monkeypatch.setitem(sys.modules, "tools.run_g05_public_answers", SimpleNamespace(decode_public=None))
    assert invoke_runner(monkeypatch, mini=mini, tasks=tasks, output=output, model=model,
                         extra=("--only-tasks", "bbeh_a")) == 0
    _, cases = run_g12_bbeh.load_cases(mini)
    first_saved = cases.saved_rows(output / "rows")
    assert set(first_saved) == {cases.cases[0].id, cases.cases[2].id}
    assert first_saved[cases.cases[0].id]["target"] == "left"
    assert first_saved[cases.cases[2].id]["target"] == "right"
    assert first_saved[cases.cases[0].id]["correct"] is True
    assert first_saved[cases.cases[2].id]["correct"] is False
    assert first_saved[cases.cases[0].id]["source_ordinal"] == 0
    assert first_saved[cases.cases[2].id]["source_ordinal"] == 2
    assert invoke_runner(monkeypatch, mini=mini, tasks=tasks, output=output, model=model) == 0
    assert set(cases.saved_rows(output / "rows")) == set(cases.by_id)
    from tools.run_g09_organ import BBEH_SUFFIX

    assert deliveries == [([[{"role": "user", "content": f"shared\n\n{BBEH_SUFFIX}"}]] * 2, 32768, 8),
                          ([[{"role": "user", "content": f"other\n\n{BBEH_SUFFIX}"}]], 32768, 8)]
    assert invoke_runner(monkeypatch, mini=mini, tasks=tasks, output=output, model=model) == 0
    assert len(deliveries) == 2


def test_wrong_saved_target_stops_resume_before_loading_any_model(tmp_path, monkeypatch) -> None:
    mini, tasks = mini_source(tmp_path, monkeypatch, [{"input": "unique", "target": "left", "task": "a"}])
    _, cases = run_g12_bbeh.load_cases(mini)
    rows = tmp_path / "output" / "rows"
    rows.mkdir(parents=True)
    (rows / f"{cases.cases[0].id}.json").write_text(json.dumps({"id": cases.cases[0].id, "target": "wrong"}))
    loaded = []
    monkeypatch.setitem(sys.modules, "mlx_lm", SimpleNamespace(load=lambda *_args: loaded.append(True)))
    with pytest.raises(ValueError, match="reference does not match"):
        invoke_runner(monkeypatch, mini=mini, tasks=tasks, output=rows.parent, model=tmp_path / "missing-model")
    assert loaded == []


def test_g09_import_uses_each_saved_source_case_and_keeps_reference_provenance(tmp_path, monkeypatch) -> None:
    from tools.g09_kept_procedure_arm import drafts_from_g12_bbeh

    examples = [{"input": "shared", "target": "left", "task": "a"},
                {"input": "shared", "target": "right", "task": "a"}]
    mini, _ = mini_source(tmp_path, monkeypatch, examples)
    records, cases = run_g12_bbeh.load_cases(mini)
    monkeypatch.setattr(run_g12_bbeh, "load_cases", lambda _path: (records, cases))
    rows = tmp_path / "output" / "rows"
    rows.mkdir(parents=True)
    for case in reversed(cases.cases):
        (rows / f"{case.id}.json").write_text(json.dumps({"id": case.id, **case.metadata(),
            "target": examples[case.source_ordinal]["target"], "task": "bbeh_a", "termination": "stop",
            "public_text": "one delivered answer", "correct": case.source_ordinal == 0,
            "generated_tokens": 1, "seconds": 0.1}))
    drafts = {row["id"]: row for row in drafts_from_g12_bbeh(rows.parent)}
    for case in cases.cases:
        assert drafts[case.id]["truth"] == examples[case.source_ordinal]["target"]
        assert drafts[case.id]["source_ordinal"] == case.source_ordinal
        assert drafts[case.id]["reference_sha256"] == case.reference_sha256
        assert drafts[case.id]["draft_correct"] == (case.source_ordinal == 0)


@pytest.mark.parametrize("changed", [
    {"termination": "unknown"}, {"termination": []}, {"public_text": None}, {"generated_tokens": True},
    {"generated_tokens": -1}, {"seconds": True}, {"seconds": float("nan")},
    {"seconds": float("inf")}, {"seconds": -1}, {"correct": 1},
    {"termination": "token_limit", "correct": True},
])
def test_malformed_saved_decode_cannot_skip_a_source_case_before_model_import(tmp_path, monkeypatch, changed) -> None:
    mini, tasks = mini_source(tmp_path, monkeypatch, [{"input": "unique", "target": "left", "task": "a"}])
    _, cases = run_g12_bbeh.load_cases(mini)
    case = cases.cases[0]
    row = {"id": case.id, "target": "left", "termination": "stop", "public_text": "The answer is: left",
           "generated_tokens": 1, "seconds": 0.1, "correct": True, **changed}
    directory = tmp_path / "output" / "rows"
    directory.mkdir(parents=True)
    (directory / f"{case.id}.json").write_text(json.dumps(row))
    loaded = []
    monkeypatch.setitem(sys.modules, "mlx_lm", SimpleNamespace(load=lambda *_args: loaded.append(True)))
    with pytest.raises(ValueError, match="saved BBEH row"):
        invoke_runner(monkeypatch, mini=mini, tasks=tasks, output=directory.parent, model=tmp_path / "missing-model")
    assert loaded == []


def test_binding_only_saved_row_cannot_stand_for_a_completed_decode(tmp_path) -> None:
    cases = catalog([{"input": "unique", "target": "left"}])
    (tmp_path / "unique.json").write_text(json.dumps({"id": "unique", "target": "left"}))
    with pytest.raises(ValueError, match="decode termination"):
        run_g12_bbeh.saved_results(tmp_path, cases)


@pytest.mark.parametrize("termination", ["token_limit", "length", "native_thinking_incomplete"])
def test_a_valid_budget_failure_is_retained_as_a_completed_wrong_result(termination) -> None:
    run_g12_bbeh.validate_saved_result({"termination": termination, "public_text": "", "generated_tokens": 8,
                                       "seconds": 0.1, "correct": False})


def test_atomic_publication_refuses_to_replace_an_existing_receipt(tmp_path) -> None:
    path = tmp_path / "row.json"
    row = {"id": "case", "termination": "stop", "public_text": "answer", "generated_tokens": 1,
           "seconds": 0.1, "correct": False}
    run_g12_bbeh.publish_result(path, row)
    original = path.read_bytes()
    with pytest.raises(ValueError, match="refusing to overwrite"):
        run_g12_bbeh.publish_result(path, {**row, "public_text": "different answer", "correct": True})
    assert path.read_bytes() == original
