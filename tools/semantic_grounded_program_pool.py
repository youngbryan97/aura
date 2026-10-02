"""Immutable source-program pools with exact replay custody."""

import hashlib
import json
from pathlib import Path

from core.learning.semantic_grounded_program_objective import (
    GroundedProgramSupervision,
    mine_grounded_program_supervision,
)


def source_program_pool(parent, item, directory, **bounds):
    from core.governance_context import local_internal_governed_scope
    from core.learning.semantic_grounded_binding_engine import implementation_receipt
    from core.runtime.file_write_gateway import get_file_write_gateway

    def digest(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
            allow_nan=False).encode()).hexdigest()
    if item.split != "train":
        raise ValueError("durable program mining may only consume source-training rows")
    basis = {"schema": "aura.grounded_program_pool_basis.v1", "source_id": item.ir.source_text_sha256,
        "parent_receipt_sha256": parent.receipt_sha256, "implementation": implementation_receipt(),
        "bounds": bounds, "source_annotation": item.ir.to_dict(),
        "register_definition_spans": [(span.start, span.end) for span in item.register_definition_spans],
        "public_inputs": item.public_inputs,
        "observed_states_sha256": hashlib.sha256(item.hidden_states.tobytes()).hexdigest(),
        "observed_states_geometry": (str(item.hidden_states.dtype), item.hidden_states.shape)}
    basis_sha = digest(basis)
    path = Path(directory) / f"{item.ir.source_text_sha256}.json"
    if path.exists():
        receipt = json.loads(path.read_bytes())
        if (receipt.get("basis_sha256") != basis_sha
                or digest(receipt["basis"]) != basis_sha
                or digest({key: value for key, value in receipt.items() if key != "receipt_sha256"})
                    != receipt.get("receipt_sha256")):
            raise ValueError("durable program pool differs from source or implementation custody")
        return GroundedProgramSupervision.from_receipt(receipt["programs"]), True
    programs = mine_grounded_program_supervision(parent, item, **bounds)
    receipt = {"basis": basis, "basis_sha256": basis_sha, "programs": programs.receipt()}
    receipt["receipt_sha256"] = digest(receipt)
    with local_internal_governed_scope("grounded_program_pool", domain="file_write"):
        if not get_file_write_gateway().write_bytes_if_absent(path,
                (json.dumps(receipt, indent=2) + "\n").encode(), source="grounded_program_pool", mode=0o400):
            raise FileExistsError("another owner published the source program pool")
    return programs, False


def prepare_program_population(parent, items, directory, **bounds):
    """Collect every measured source failure before deciding fit readiness."""
    from core.governance_context import local_internal_governed_scope
    from core.learning.semantic_argument_optimization import ArgumentOptimizationIncompleteError
    from core.runtime.file_write_gateway import get_file_write_gateway

    programs, failures, rows = {}, [], []
    for item in items:
        identity = item.ir.source_text_sha256
        try:
            programs[identity], reused = source_program_pool(parent, item, directory, **bounds)
            row = {"source_id": identity, "status": "ready", "verified_pool_reused": reused,
                **programs[identity].mining}
        except (ValueError, TimeoutError, ArgumentOptimizationIncompleteError) as exc:
            row = {"source_id": identity, "status": "failed", "failure_type": type(exc).__name__,
                "reason": str(exc), "construction_id": item.construction_id,
                "operation_count": len(item.ir.instructions)}
            failures.append(row)
        rows.append(row)
        print(json.dumps({"stage": "grounded_program_pool_mined", "completed": len(rows),
            "population": len(items), **row}), flush=True)
    report = {"schema": "aura.grounded_program_population_preflight.v1", "population": len(items),
        "ready": len(programs), "failed": len(failures), "sources": rows,
        "fit_ready": not failures, "model_weights_loaded": False, "held_sources_scored": False,
        "semantic_success": None, "serving_authority": False}
    digest = hashlib.sha256(json.dumps(report, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    report["receipt_sha256"] = digest
    path = Path(directory) / f"population-{digest}.json"
    encoded = (json.dumps(report, indent=2) + "\n").encode()
    with local_internal_governed_scope("grounded_program_population", domain="file_write"):
        if not get_file_write_gateway().write_bytes_if_absent(path, encoded,
                source="grounded_program_population", mode=0o400) and path.read_bytes() != encoded:
            raise ValueError("program population receipt conflicts with its content digest")
    if failures:
        raise ValueError(f"complete-program source preflight failed on {len(failures)}/{len(items)}; receipt={path}")
    return programs, report
