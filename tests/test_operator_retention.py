"""Invented floor terms share the language store and its rollback boundary."""

import json
import subprocess
import sys
from dataclasses import FrozenInstanceError

import pytest

from core.cognition import an_operator_she_invents as invention
from core.cognition import what_she_gave_meaning as meanings
from core.cognition.operator_invention import Candidate, OperatorKernel, Rejection
from core.cognition.the_floor_she_stands_on import L, N, PLUS, V, build


def install(kernel, name, offset, *, built_from=()):
    for _ in range(3):
        kernel.attempt(name, solved=False)
    term = build(L("x", PLUS(V("x"), N(offset))))
    verdict = kernel.consider(
        Candidate(name, f"offset {offset}", term=term, built_from=built_from),
        family=name, probes=(-3, 0, 4), compression=8,
        solves=lambda fn, _: all(fn(x) == x + offset for x in (-3, 0, 4)),
    )
    assert verdict.installed, verdict
    return term


@pytest.fixture(autouse=True)
def separate_kernel(monkeypatch, tmp_path):
    from core.cognition.what_she_can_take_back import as_it_stands

    monkeypatch.setattr(invention, "_KERNEL", OperatorKernel())
    monkeypatch.setattr(meanings, "_KEPT_AT", tmp_path / "meanings.json")
    before = as_it_stands()
    yield
    before.restore()


def test_terms_survive_the_existing_language_store_and_remain_searchable(monkeypatch):
    from core.cognition.what_she_already_knows_how_to_say import what_she_already_knows_how_to_say

    term = install(invention.the_kernel(), "offset", 7)
    assert meanings.keep()
    held = json.loads(meanings._kept_at().read_text())
    assert held["language"]["invented_operators"][0]["term"]["head"] == "given a thing"
    monkeypatch.setattr(invention, "_KERNEL", OperatorKernel())
    meanings.recall()
    operator = invention.the_kernel().operators()["offset"]
    assert operator.term == term
    assert [operator.fn(x) for x in (-80, 17, 2**80)] == [-73, 24, 2**80 + 7]
    assert term in what_she_already_knows_how_to_say()
    assert invention.the_kernel().report()["installed"] == 0  # Recall is not a new qualification.


def test_recall_is_idempotent_and_preserves_rollback_lineage(monkeypatch):
    kernel = invention.the_kernel()
    install(kernel, "first", 7)
    install(kernel, "second", 9, built_from=("first",))
    assert meanings.keep()
    restored = OperatorKernel()
    monkeypatch.setattr(invention, "_KERNEL", restored)
    meanings.recall()
    before = restored.snapshot()
    meanings.recall()
    assert restored.snapshot() == before
    assert restored.operators()["second"].built_from == ("first",)
    assert restored.operators()["second"].generation == kernel.operators()["second"].generation
    assert restored.rollback("first")["removed"] == ["first", "second"]
    with pytest.raises(KeyError):
        restored.rollback("second")
    assert not restored.operators()


def test_rolling_back_an_ancestor_cannot_resurrect_its_descendants():
    kernel = invention.the_kernel()
    install(kernel, "first", 7)
    install(kernel, "second", 9, built_from=("first",))
    kernel.rollback("first")
    with pytest.raises(KeyError):
        kernel.rollback("second")
    assert not kernel.operators()


def test_dropping_the_last_meaning_replaces_the_old_durable_snapshot(monkeypatch):
    install(invention.the_kernel(), "offset", 7)
    assert meanings.keep()
    invention.the_kernel().rollback("offset")
    # Isolate the last-entry case from other process-global language stores.
    monkeypatch.setattr(meanings, "_words_she_derived", lambda: {})
    monkeypatch.setattr("core.cognition.an_invented_kind.KINDS", {})
    assert meanings.keep()
    monkeypatch.setattr(invention, "_KERNEL", OperatorKernel())
    meanings.recall()
    assert not invention.the_kernel().operators()
    assert json.loads(meanings._kept_at().read_text()) == {"kinds": {}, "language": {}}


def test_sequence_trial_uses_the_complete_development_snapshot():
    from core.cognition import sequence_induction as sequence
    from core.cognition.an_invented_kind import KINDS

    before = sequence._everything_she_can_say()
    prior = dict(KINDS)
    install(invention.the_kernel(), "offset", 7)
    KINDS["a trial meaning"] = object()
    sequence._put_back(before)
    assert not invention.the_kernel().operators()
    assert KINDS == prior


def test_invalid_operator_batch_does_not_install_its_valid_prefix():
    kernel = invention.the_kernel()
    install(kernel, "first", 7)
    rows = kernel.written_operators()
    rows.append({**rows[0], "name": "bad", "term": {"head": "python eval"}})
    fresh = OperatorKernel()
    with pytest.raises(ValueError, match="operator"):
        fresh.recall_operators(rows)
    assert not fresh.operators()
    assert fresh.snapshot().snapshots == ()


def test_recall_cannot_replace_a_base_operator():
    kernel = invention.the_kernel()
    install(kernel, "offset", 7)
    fresh = OperatorKernel({"offset": lambda x, _budget=None: x})
    with pytest.raises(ValueError, match="conflict"):
        fresh.recall_operators(kernel.written_operators())
    assert fresh.operators()["offset"].fn(3) == 3


@pytest.mark.parametrize("field,value", [
    ("generation", True), ("generation", -1), ("built_from", ["missing"]),
    ("name", ""), ("schema", "new unknown schema"),
])
def test_malformed_lineage_does_not_restore(field, value):
    kernel = invention.the_kernel()
    install(kernel, "offset", 7)
    row = {**kernel.written_operators()[0], field: value}
    with pytest.raises(ValueError, match="operator"):
        OperatorKernel().recall_operators([row])


def test_a_python_callable_is_not_silently_dropped_from_the_save():
    kernel = invention.the_kernel()
    install(kernel, "offset", 7)
    assert meanings.keep()
    before = meanings._kept_at().read_bytes()
    for _ in range(3):
        kernel.attempt("opaque", solved=False)
    assert kernel.consider(Candidate("opaque", "opaque", fn=lambda x, b: x + 100),
        family="opaque", probes=(1, 2), solves=lambda *_: True, compression=10).installed
    assert not meanings.keep()
    assert meanings._kept_at().read_bytes() == before


def test_one_operator_must_match_the_whole_probe_vector_to_refute_novelty():
    # Each output is already produced somewhere, but neither operator is identity.
    kernel = OperatorKernel({"zero": lambda x, b: 0, "one": lambda x, b: 1})
    for _ in range(3):
        kernel.attempt("identity", solved=False)
    candidate = Candidate("identity", "identity", term=build(L("x", V("x"))))
    verdict = kernel.consider(candidate, family="identity", probes=(0, 1),
        solves=lambda fn, _: all(fn(x) == x for x in (0, 1)), compression=2)
    assert verdict.installed, verdict
    renamed = kernel.consider(Candidate("renamed", "identity", term=candidate.term),
        family="identity", probes=(0, 1), solves=lambda *_: True, compression=2)
    assert renamed.rejection is Rejection.NOT_NOVEL


def test_admission_cannot_replace_an_existing_name():
    kernel = invention.the_kernel()
    install(kernel, "offset", 7)
    before = kernel.snapshot()
    verdict = kernel.consider(Candidate("offset", "replacement", term=build(L("x", N(100)))),
        family="offset", probes=(1, 2), solves=lambda *_: True, compression=2)
    assert not verdict.installed
    assert kernel.snapshot() == before


def test_operator_snapshot_values_cannot_be_mutated_in_place():
    install(invention.the_kernel(), "offset", 7)
    with pytest.raises(FrozenInstanceError):
        invention.the_kernel().operators()["offset"].body = "changed"


def test_rejected_trial_restores_the_learned_search_and_decision_terms():
    from core.cognition import the_order_she_tries_them_in as order
    from core.cognition import the_proposer_she_can_replace as proposer
    from core.cognition import what_counts_as_better as objective
    from core.cognition import what_it_is_worth_doing as worth
    from core.cognition.what_she_can_take_back import as_it_stands, only_if_it_pays

    before = as_it_stands()
    with only_if_it_pays("change all search machinery"):
        term = build(L("x", N(17)))
        order.the_order_she_wrote(term)
        proposer.the_proposer_she_wrote(term)
        objective.the_objective_she_wrote(term)
        worth.the_worth_she_wrote(term)
        assert len(before.what_changed()) == 4
    assert as_it_stands() == before


def test_fresh_process_recalls_the_saved_term_and_undoes_it():
    install(invention.the_kernel(), "offset", 7)
    assert meanings.keep()
    script = """
import json
import sys
from pathlib import Path
from core.cognition import what_she_gave_meaning as meanings
from core.cognition.an_operator_she_invents import the_kernel
meanings._KEPT_AT = Path(sys.argv[1])
meanings.recall()
result = the_kernel().operators()['offset'].fn(123456)
the_kernel().rollback('offset')
print(json.dumps({'result': result, 'remaining': list(the_kernel().operators())}))
"""
    result = subprocess.run([sys.executable, "-c", script, str(meanings._kept_at())],
                            capture_output=True, text=True, timeout=30, check=True)
    assert json.loads(result.stdout.splitlines()[-1]) == {"result": 123463, "remaining": []}
