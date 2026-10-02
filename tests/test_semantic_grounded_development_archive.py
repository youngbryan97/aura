import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from tools.semantic_grounded_development_archive import (
    archive_decode,
    compact_decode,
    digest,
    select_population,
    verify_archive_inventory,
    verify_archived_decode,
    verify_population_basis,
)


def inputs():
    from tests.test_semantic_program_shared_transducer import _examples

    item = _examples()[0]
    record = {"refusal": "", "program_sha256": item.ir.to_program().sha(), "elapsed_seconds": .2,
        "joint_receipt": {"selected_chart": {"status": "bound",
            "edge_evidence": [{"role": index, "score": .1} for index in range(1000)],
            "dominated_mention_witnesses": [{"span": [1, 2]} for _ in range(200)]}}}
    return item, record


def archive(tmp_path, *, refusal=False, arm="joint_native"):
    item, record = inputs()
    if refusal:
        record = {**record, "refusal": "budget_exhausted", "program_sha256": None, "joint_receipt": None}
    compact = archive_decode(tmp_path, plan_sha256="a" * 64, arm=arm,
        identity=item.ir.source_text_sha256, record=record, ir=None if refusal else item.ir,
        source_token_ids=item.ir.source_token_ids, public_inputs=item.public_inputs,
        model_basis_sha256=item.ir.model_basis_receipt_sha256)
    return item, record, compact


@pytest.mark.parametrize("arm", ["source_parent", "global_chart", "joint_native"])
@pytest.mark.parametrize("refusal", [False, True])
def test_full_decode_rows_restore_exact_compact_evidence_and_actual_ir(tmp_path, arm, refusal):
    item, record, compact = archive(tmp_path, refusal=refusal, arm=arm)
    checksum = verify_archived_decode(tmp_path, plan_sha256="a" * 64, arm=arm,
        identity=item.ir.source_text_sha256, compact=compact)
    assert checksum == compact["decode_archive"]["receipt_sha256"]
    assert digest(compact_decode(record)) == digest({k: v for k, v in compact.items() if k != "decode_archive"})
    verify_archive_inventory(tmp_path, (arm,), (item.ir.source_text_sha256,))
    row = json.loads((tmp_path / compact["decode_archive"]["filename"]).read_bytes())
    assert digest(row["decode"]) == digest(record)
    if not refusal:
        assert len(row["decode"]["joint_receipt"]["selected_chart"]["edge_evidence"]) == 1000
        assert "edge_evidence" not in compact["joint_receipt"]["selected_chart"]
        assert row["decoded_ir"] == item.ir.to_dict()
        assert compact["decode_archive"]["bytes"] > len(json.dumps(compact)) * 5


@pytest.mark.parametrize("fault", ["bytes", "compact", "missing", "filename", "source", "plan", "program"])
def test_decode_archive_rejects_tampering_even_after_rehashing(tmp_path, fault):
    item, _record, compact = archive(tmp_path)
    path = tmp_path / compact["decode_archive"]["filename"]
    if fault == "missing":
        path.unlink()
    elif fault == "compact":
        compact["elapsed_seconds"] += 1.
    elif fault == "filename":
        compact["decode_archive"]["filename"] = "../substitution.json"
    else:
        row = json.loads(path.read_bytes())
        if fault == "bytes":
            raw = b"{}"
        else:
            if fault == "source":
                row["source_id"] = "b" * 64
            elif fault == "plan":
                row["plan_sha256"] = "b" * 64
            else:
                row["decoded_ir"]["instructions"][0]["op"] = "mul"
            row["receipt_sha256"] = digest({k: v for k, v in row.items() if k != "receipt_sha256"})
            raw = json.dumps(row).encode()
            compact["decode_archive"].update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(),
                receipt_sha256=row["receipt_sha256"])
        path.chmod(0o600)
        path.write_bytes(raw)
    with pytest.raises((ValueError, FileNotFoundError)):
        verify_archived_decode(tmp_path, plan_sha256="a" * 64, arm="joint_native",
            identity=item.ir.source_text_sha256, compact=compact)


def test_decode_archive_refuses_overwrite_extra_rows_and_mismatched_public_source(tmp_path):
    item, record, _compact = archive(tmp_path)
    with pytest.raises(FileExistsError):
        archive(tmp_path)
    with pytest.raises(ValueError, match="inventory"):
        verify_archive_inventory(tmp_path, ("source_parent", "joint_native"), (item.ir.source_text_sha256,))
    with pytest.raises(ValueError, match="public decode"):
        archive_decode(tmp_path, plan_sha256="a" * 64, arm="source_parent", identity=item.ir.source_text_sha256,
            record=record, ir=item.ir, source_token_ids=(999,), public_inputs=item.public_inputs,
            model_basis_sha256=item.ir.model_basis_receipt_sha256)


def population():
    from tests.test_semantic_program_shared_transducer import _examples

    examples = _examples()
    validation = tuple(replace(item, split="validation") for item in examples[:2])
    ids = [item.ir.source_text_sha256 for item in validation]
    parent = SimpleNamespace(training_receipt={"validation_example_count": 2,
        "validation_example_ids_sha256": digest(sorted(ids))})
    native = {"fit_ids": [], "calibration_ids": []}
    return validation + (replace(examples[2], split="train"),), parent, native


def test_complete_source_population_keeps_every_validation_id_and_no_training_id():
    examples, parent, native = population()
    selected = select_population(examples, {"held_ids": [examples[0].ir.source_text_sha256]}, parent,
        native, "source_validation")
    assert len(selected) == 2 and all(item.split == "validation" for item in selected)
    bank = select_population(examples, {"held_ids": [examples[0].ir.source_text_sha256]}, parent,
        native, "bank_holdout")
    assert bank == (examples[0],)


@pytest.mark.parametrize("fault", ["missing", "substitution", "fit", "selection", "duplicate", "population"])
def test_population_refuses_missing_replaced_duplicate_or_selection_sources(fault):
    examples, parent, native = population()
    kind = "source_validation"
    if fault == "missing":
        examples = examples[1:]
    elif fault == "substitution":
        parent.training_receipt["validation_example_ids_sha256"] = "b" * 64
    elif fault in {"fit", "selection"}:
        native["fit_ids" if fault == "fit" else "calibration_ids"] = [examples[0].ir.source_text_sha256]
    elif fault == "duplicate":
        examples = (*examples, examples[0])
    else:
        kind = "unknown"
    with pytest.raises(ValueError):
        select_population(examples, {}, parent, native, kind)


@pytest.mark.parametrize("fault", [None, "count", "ids", "source", "fit"])
def test_independent_population_check_binds_original_parent_report_and_native_fit(tmp_path, fault):
    examples, parent, native = population()
    ids = sorted(item.ir.source_text_sha256 for item in examples if item.split == "validation")
    paths = {name: tmp_path / (name + ".json") for name in ("parent", "source_report")}
    paths["parent"].write_text(json.dumps({"training_receipt": parent.training_receipt}))
    paths["source_report"].write_text(json.dumps(parent.training_receipt))
    native["source_basis"] = {name + "_sha256": hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in paths.items()}
    plan = {"held_ids": ids, "population_basis": {**{name: str(path) for name, path in paths.items()},
        "split": "validation", "example_count": 2, "example_ids_sha256": digest(ids)}}
    if fault == "count":
        plan["population_basis"]["example_count"] = 1
    elif fault == "ids":
        plan["held_ids"] = ["b" * 64, ids[1]]
    elif fault == "source":
        paths["source_report"].write_text("{}")
    elif fault == "fit":
        native["fit_ids"] = [ids[0]]
    if fault is None:
        verify_population_basis(plan, native)
    else:
        with pytest.raises(ValueError):
            verify_population_basis(plan, native)
