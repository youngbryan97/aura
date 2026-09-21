"""Resume exact semantic validation without reusing another observation's score."""

import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from core.runtime.file_write_gateway import get_file_write_gateway
from typing import Any


def _digest(value: dict[str, Any]) -> Any:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


VALIDATION_SCORING = ("register_indices_v1", "source_anchors_v2")
_CORE_ROOT = Path(__file__).resolve().parents[1]


def validation_implementation_identity() -> Any:
    """Conservative source snapshot for cached measurements, not serving gates.

    Include the core tree because decoder imports and floor operators span
    modules and can change independently of the model's coefficient receipt.
    This binds disk source; it is not attestation of the loaded interpreter.
    """
    sources = {path.relative_to(_CORE_ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
               for path in sorted(_CORE_ROOT.rglob("*.py")) if path.is_file()}
    if not sources:
        raise ValueError("semantic validation implementation source is unavailable")
    return _digest({"sources": sources, "python": sys.version, "numpy": np.__version__})


def validation_identity(
    candidates: Any,
    examples: Any,
    *,
    scoring: str='register_indices_v1',
    implementation: Any=None,
) -> Any:
    if scoring not in VALIDATION_SCORING:
        raise ValueError("unknown semantic validation scoring")
    observations = []
    for item in examples:
        hidden = np.asarray(item.hidden_states) if item.hidden_states is not None else None
        if hidden is not None and hidden.dtype.hasobject:
            raise ValueError("validation checkpoint cannot bind object-valued hidden states")
        program = item.ir.to_program()
        observations.append({
            "source": item.ir.source_text_sha256,
            "tokens": list(item.ir.source_token_ids),
            "basis": item.ir.model_basis_receipt_sha256,
            "inputs": item.public_inputs,
            "gold": program.to_dict() if hasattr(program, "to_dict") else program,
            "hidden": None if hidden is None else {
                "dtype": hidden.dtype.str, "shape": list(hidden.shape),
                "sha256": hashlib.sha256(hidden.tobytes(order="C")).hexdigest(),
            },
        })
        if scoring == "source_anchors_v2":
            observations[-1]["input_anchors"] = [(span.start, span.end) for span in item.ir.input_spans]
    return _digest({"implementation": implementation or validation_implementation_identity(),
                    "models": {name: model.receipt_sha256 for name, model in candidates.items()},
                    "observations": observations, "scoring": "exact_and_equivalent_program_v1"
                    if scoring == "register_indices_v1" else "source_anchored_exact_and_equivalent_program_v2"})


class SemanticValidationCheckpoint:
    """A checksummed local measurement cache, not qualification authority."""

    def __init__(self, path: Any, identity: Any, candidates: Any, source_ids: Any) -> None:
        self.path = Path(path)
        self.identity = identity
        self.allowed = {name: set(source_ids) for name in candidates}
        self.rows = {}
        if self.path.exists():
            document = json.loads(self.path.read_text("utf-8"))
            body = {k: v for k, v in document.items() if k != "sha256"}
            if (document.get("sha256") != _digest(body)
                    or body.get("schema") != "aura.semantic_validation_checkpoint.v1"
                    or body.get("identity") != identity):
                raise ValueError("semantic validation checkpoint identity or checksum differs")
            self.rows = body["rows"]
            for name, rows in self.rows.items():
                if name not in self.allowed or not isinstance(rows, dict):
                    raise ValueError("semantic validation checkpoint has unknown candidate")
                for source, value in rows.items():
                    if (source not in self.allowed[name] or not isinstance(value, list)
                            or len(value) != 2 or any(type(v) is not bool for v in value)
                            or (value[0] and not value[1])):
                        raise ValueError("semantic validation checkpoint has invalid observation")

    def get(self, name: str, source: dict[str, Any]) -> Any:
        return self.rows.get(name, {}).get(source)

    def record(self, name: Any, source: Any, exact: Any, equivalent: Any) -> None:
        if name not in self.allowed or source not in self.allowed[name]:
            raise ValueError("semantic validation checkpoint observation is out of scope")
        if type(exact) is not bool or type(equivalent) is not bool or (exact and not equivalent):
            raise ValueError("semantic validation checkpoint score is invalid")
        previous = self.get(name, source)
        if previous is not None and previous != [exact, equivalent]:
            raise ValueError("semantic validation checkpoint observation changed")
        self.rows.setdefault(name, {})[source] = [exact, equivalent]
        body = {"schema": "aura.semantic_validation_checkpoint.v1",
                "identity": self.identity, "rows": self.rows}
        get_file_write_gateway().write_text(
            self.path, json.dumps({**body, "sha256": _digest(body)},
                                 sort_keys=True, separators=(",", ":")),
            source="semantic_validation_checkpoint", durable=True,
        )
