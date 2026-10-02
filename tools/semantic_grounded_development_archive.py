"""Keep complete public decode evidence in bounded, independently checked rows."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ARMS = ("source_parent", "global_chart", "joint_native")
MAX_ROW_BYTES = 64 * 1024 ** 2


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def development_contract():
    return {"schema": "aura.grounded_development_storage.v1",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "maximum_row_bytes": MAX_ROW_BYTES, "complete_decode_evidence_retained": True,
        "archive_contains_targets": False, "qualification_evidence": False}


def select_population(examples, bank, parent, native, population):
    by_id = {item.ir.source_text_sha256: item for item in examples}
    if len(by_id) != len(examples):
        raise ValueError("development population needs unique source identities")
    if population == "bank_holdout":
        selected = tuple(by_id[identity] for identity in bank["held_ids"])
    elif population == "source_validation":
        selected = tuple(sorted((item for item in examples if item.split == "validation"),
            key=lambda item: item.ir.source_text_sha256))
        ids = [item.ir.source_text_sha256 for item in selected]
        if (len(ids) != parent.training_receipt.get("validation_example_count")
                or digest(ids) != parent.training_receipt.get("validation_example_ids_sha256")):
            raise ValueError("complete source validation population differs from frozen parent")
    else:
        raise ValueError("undeclared development population")
    identities = {item.ir.source_text_sha256 for item in selected}
    if not identities or identities & set((*native["fit_ids"], *native["calibration_ids"])):
        raise ValueError("development population is empty or overlaps native fitting or selection")
    return selected


def compact_decode(record):
    result = dict(record)
    receipt = result.get("joint_receipt")
    if receipt is not None:
        receipt = dict(receipt)
        selected = receipt.get("selected_chart")
        if selected is not None:
            selected = dict(selected)
            for field in ("edge_evidence", "dominated_mention_witnesses"):
                if field in selected:
                    values = selected.pop(field)
                    selected["archived_" + field] = {"count": len(values), "sha256": digest(values)}
            receipt["selected_chart"] = selected
        result["joint_receipt"] = receipt
    return result


def _filename(arm, identity):
    if arm not in ARMS or not isinstance(identity, str) or len(identity) != 64 or any(
            value not in "0123456789abcdef" for value in identity):
        raise ValueError("decode archive needs an explicit arm and source hash")
    return f"{arm}.{identity}.json"


def archive_decode(directory, *, plan_sha256, arm, identity, record, ir, source_token_ids,
                   public_inputs, model_basis_sha256):
    from core.governance_context import local_internal_governed_scope
    from core.runtime.file_write_gateway import get_file_write_gateway

    filename = _filename(arm, identity)
    if ir is not None and (ir.source_text_sha256 != identity
            or ir.source_token_ids != tuple(source_token_ids)
            or ir.model_basis_receipt_sha256 != model_basis_sha256
            or ir.n_inputs != len(public_inputs)
            or ir.to_program().sha() != record["program_sha256"]):
        raise ValueError("archived program differs from its measured public decode")
    if (ir is None) != (record["program_sha256"] is None) or (ir is None) != bool(record["refusal"]):
        raise ValueError("archived refusal differs from its measured public decode")
    body = {"schema": "aura.grounded_public_decode_archive.v1", "plan_sha256": plan_sha256,
        "arm": arm, "source_id": identity, "decode": record,
        "decoded_ir": ir.to_dict() if ir is not None else None,
        "source_token_ids": tuple(source_token_ids), "public_inputs": public_inputs,
        "model_basis_sha256": model_basis_sha256, "expected_answer_available": False,
        "qualification_evidence": False, "serving_authority": False}
    document = {**body, "receipt_sha256": digest(body)}
    raw = json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    if len(raw) > MAX_ROW_BYTES:
        raise ValueError("complete decode archive exceeds its declared row bound")
    with local_internal_governed_scope("grounded_decode_archive", domain="file_write"):
        if not get_file_write_gateway().write_bytes_if_absent(Path(directory) / filename, raw,
                source="grounded_decode_archive", mode=0o400):
            raise FileExistsError(filename)
    compact = compact_decode(record)
    compact["decode_archive"] = {"filename": filename, "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(), "receipt_sha256": document["receipt_sha256"]}
    return compact


def verify_archived_decode(directory, *, plan_sha256, arm, identity, compact):
    from core.learning.semantic_program_ir import semantic_program_ir_from_dict
    from core.runtime.file_read_gateway import read_stable_bytes

    filename = _filename(arm, identity)
    reference = compact.get("decode_archive")
    if (not isinstance(reference, dict) or set(reference) != {"filename", "bytes", "sha256", "receipt_sha256"}
            or reference["filename"] != filename or type(reference["bytes"]) is not int
            or not 0 < reference["bytes"] <= MAX_ROW_BYTES):
        raise ValueError("decode archive reference differs from its declared identity")
    raw = read_stable_bytes(Path(directory) / filename, max_bytes=MAX_ROW_BYTES)
    if len(raw) != reference["bytes"] or hashlib.sha256(raw).hexdigest() != reference["sha256"]:
        raise ValueError("decode archive bytes differ from the measured reference")
    row = json.loads(raw)
    if (row.get("schema") != "aura.grounded_public_decode_archive.v1"
            or row.get("receipt_sha256") != digest({k: v for k, v in row.items() if k != "receipt_sha256"})
            or row["receipt_sha256"] != reference["receipt_sha256"]
            or row.get("plan_sha256") != plan_sha256 or row.get("arm") != arm
            or row.get("source_id") != identity
            or any(row.get(key) is not False for key in (
                "expected_answer_available", "qualification_evidence", "serving_authority"))
            or digest(compact_decode(row["decode"])) != digest({k: v for k, v in compact.items() if k != "decode_archive"})):
        raise ValueError("decode archive custody or compact evidence differs")
    decoded = row["decode"]
    ir = None if row["decoded_ir"] is None else semantic_program_ir_from_dict(row["decoded_ir"])
    if ((ir is None) != (decoded["program_sha256"] is None) or (ir is None) != bool(decoded["refusal"])
            or ir is not None and (ir.source_text_sha256 != identity
                or ir.source_token_ids != tuple(row["source_token_ids"])
                or ir.model_basis_receipt_sha256 != row["model_basis_sha256"]
                or ir.n_inputs != len(row["public_inputs"])
                or ir.to_program().sha() != decoded["program_sha256"])):
        raise ValueError("decode archive program differs from measured public output")
    return row["receipt_sha256"]


def verify_archive_inventory(directory, arms, identities):
    expected = {_filename(arm, identity) for arm in arms for identity in identities}
    actual = {path.name for path in Path(directory).iterdir()}
    if actual != expected:
        raise ValueError("decode archive inventory differs from complete measured population")


def verify_population_basis(plan, native):
    from core.runtime.file_read_gateway import read_stable_bytes

    basis = plan["population_basis"]
    identities = plan["held_ids"]
    raw = {name: read_stable_bytes(Path(basis[name]), max_bytes=MAX_ROW_BYTES)
        for name in ("parent", "source_report")}
    if any(hashlib.sha256(value).hexdigest() != native["source_basis"][name + "_sha256"]
            for name, value in raw.items()):
        raise ValueError("development population changed the native fit's source basis")
    parent, source = json.loads(raw["parent"]), json.loads(raw["source_report"])
    count, ids_sha = len(identities), digest(sorted(identities))
    if (basis.get("split") != "validation" or basis.get("example_count") != count
            or basis.get("example_ids_sha256") != ids_sha
            or any(owner.get("validation_example_count") != count
                or owner.get("validation_example_ids_sha256") != ids_sha
                for owner in (parent["training_receipt"], source))
            or set(identities) & set((*native["fit_ids"], *native["calibration_ids"]))):
        raise ValueError("development population is incomplete or overlaps fitting or selection")
