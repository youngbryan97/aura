"""Resume exact semantic validation without reusing another observation's score."""

import hashlib
import json
from pathlib import Path

import numpy as np

from core.runtime.file_write_gateway import get_file_write_gateway


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def validation_identity(candidates, examples):
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
    return _digest({"models": {name: model.receipt_sha256 for name, model in candidates.items()},
                    "observations": observations, "scoring": "exact_and_equivalent_program_v1"})


class SemanticValidationCheckpoint:
    """A checksummed local measurement cache, not qualification authority."""

    def __init__(self, path, identity, candidates, source_ids):
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

    def get(self, name, source):
        return self.rows.get(name, {}).get(source)

    def record(self, name, source, exact, equivalent):
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
