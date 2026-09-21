"""Persist accepted constraint updates with their exact evidence identity."""

from dataclasses import fields, is_dataclass
import hashlib
from io import BytesIO
import json
from pathlib import Path

import numpy as np

from core.runtime.file_write_gateway import get_file_write_gateway


def fit_identity(value):
    """Hash numerical evidence by value, retaining type, shape and field identity."""
    cache = {}

    def digest(item):
        if isinstance(item, (np.ndarray, tuple)) or is_dataclass(item):
            key = id(item)
            if key in cache:
                return cache[key]
        else:
            key = None
        if isinstance(item, np.ndarray):
            if item.dtype.hasobject:
                raise ValueError("object-valued fit evidence is unsupported")
            body = ["array", item.dtype.str, item.shape,
                    hashlib.sha256(item.tobytes(order="C")).hexdigest()]
        elif is_dataclass(item):
            body = [type(item).__module__, type(item).__qualname__,
                    [[field.name, digest(getattr(item, field.name))] for field in fields(item)]]
        elif isinstance(item, (tuple, list)):
            body = [type(item).__name__, [digest(part) for part in item]]
        elif isinstance(item, dict):
            body = {name: digest(part) for name, part in sorted(item.items())}
        elif isinstance(item, np.generic):
            body = item.item()
        else:
            body = item
        result = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"),
                                           allow_nan=False).encode("utf-8")).hexdigest()
        if key is not None:
            cache[key] = result
        return result

    return digest(value)


def write_fit_archive(path, body, arrays):
    """Write numerical checkpoint evidence through one checksummed owner."""
    metadata = json.dumps({**body, "sha256": fit_identity((body, arrays))},
                          sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    output = BytesIO()
    np.savez_compressed(output, metadata=np.frombuffer(metadata, dtype=np.uint8), **arrays)
    get_file_write_gateway().write_bytes(path, output.getvalue(), source="semantic_fit_checkpoint")


def save_round_candidate(path, *, candidate, parent, round_index, numerical_checkpoint):
    """Keep each accepted model available for decoding without replaying training."""
    from core.runtime.atomic_writer import atomic_write_bytes_if_absent

    if type(round_index) is not int or round_index < 1:
        raise ValueError("candidate round must be positive")
    body = {"schema": "aura.semantic_graph_round_candidate.v1", "parent": parent,
            "round": round_index, "candidate": candidate.to_dict(),
            "numerical_checkpoint_sha256": hashlib.sha256(Path(numerical_checkpoint).read_bytes()).hexdigest(),
            "serving_authority": False, "validation_used_for_selection": False}
    payload = (json.dumps({**body, "sha256": fit_identity(body)}, sort_keys=True,
                         separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
    path = Path(path)
    if not atomic_write_bytes_if_absent(path, payload, mode=0o400) and path.read_bytes() != payload:
        raise ValueError("saved round candidate differs from resumed numerical fit")


def load_round_candidate(path, *, expected_parent, expected_round, numerical_checkpoint):
    """Verify the complete snapshot before exposing its model for evaluation."""
    from core.learning.semantic_program_compositional_transducer import compositional_semantic_program_transducer_from_dict

    document = json.loads(Path(path).read_text("utf-8"))
    body = {key: value for key, value in document.items() if key != "sha256"}
    if (type(expected_round) is not int or expected_round < 1
            or set(body) != {"schema", "parent", "round", "candidate",
                             "numerical_checkpoint_sha256", "serving_authority", "validation_used_for_selection"}
            or body.get("schema") != "aura.semantic_graph_round_candidate.v1"
            or body.get("parent") != expected_parent
            or type(body.get("round")) is not int or body["round"] != expected_round
            or body.get("serving_authority") is not False
            or body.get("validation_used_for_selection") is not False
            or document.get("sha256") != fit_identity(body)
            or body.get("numerical_checkpoint_sha256") != hashlib.sha256(Path(numerical_checkpoint).read_bytes()).hexdigest()):
        raise ValueError("round candidate identity or checkpoint differs")
    return compositional_semantic_program_transducer_from_dict(body["candidate"])


class SemanticFitCheckpoint:
    """Crash-atomic numerical state; no model publication or serving authority."""

    _arrays = ("flat", "margins", "floors")

    def __init__(self, path, identity):
        self.path, self.identity = Path(path), identity

    def load(self):
        if not self.path.exists():
            return None
        with np.load(BytesIO(self.path.read_bytes()), allow_pickle=False) as archive:
            if set(archive.files) != {*self._arrays, "metadata"}:
                raise ValueError("fit checkpoint fields differ")
            metadata = json.loads(archive["metadata"].tobytes().decode("utf-8"))
            arrays = {name: archive[name] for name in self._arrays}
        body = {name: value for name, value in metadata.items() if name != "sha256"}
        if (body.get("schema") != "aura.semantic_fit_checkpoint.v1"
                or body.get("identity") != self.identity
                or metadata.get("sha256") != fit_identity((body, arrays))):
            raise ValueError("fit checkpoint identity or checksum differs")
        if any(value.dtype != np.float64 or value.ndim != 1 for value in arrays.values()):
            raise ValueError("fit checkpoint array geometry differs")
        return {**body, **arrays}

    def save(self, *, flat, margins, floors, trace, next_step, status):
        arrays = {name: np.asarray(value, dtype=np.float64) for name, value in
                  (("flat", flat), ("margins", margins), ("floors", floors))}
        body = {"schema": "aura.semantic_fit_checkpoint.v1", "identity": self.identity,
                "trace": trace, "next_step": next_step, "status": status}
        write_fit_archive(self.path, body, arrays)

    def save_projection(self, *, normals, required, anchor, receipt, step):
        """Retain a rejected affine problem independently of accepted model state."""
        arrays = {name: np.asarray(value, dtype=np.float64) for name, value in
                  (("normals", normals), ("required", required), ("anchor", anchor))}
        body = {"schema": "aura.semantic_projection_diagnostic.v1", "identity": self.identity,
                "step": step, "receipt": receipt, "serving_authority": False}
        path = self.path.with_name(self.path.name + ".projection.npz")
        write_fit_archive(path, body, arrays)
        return path
