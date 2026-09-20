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
        metadata = json.dumps({**body, "sha256": fit_identity((body, arrays))},
                              sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        output = BytesIO()
        np.savez_compressed(output, metadata=np.frombuffer(metadata, dtype=np.uint8), **arrays)
        get_file_write_gateway().write_bytes(self.path, output.getvalue(), source="semantic_fit_checkpoint")

    def save_projection(self, *, normals, required, anchor, receipt, step):
        """Retain a rejected affine problem independently of accepted model state."""
        arrays = {name: np.asarray(value, dtype=np.float64) for name, value in
                  (("normals", normals), ("required", required), ("anchor", anchor))}
        body = {"schema": "aura.semantic_projection_diagnostic.v1", "identity": self.identity,
                "step": step, "receipt": receipt, "serving_authority": False}
        metadata = json.dumps({**body, "sha256": fit_identity((body, arrays))},
                              sort_keys=True, allow_nan=False).encode("utf-8")
        output = BytesIO()
        np.savez_compressed(output, metadata=np.frombuffer(metadata, dtype=np.uint8), **arrays)
        path = self.path.with_name(self.path.name + ".projection.npz")
        get_file_write_gateway().write_bytes(path, output.getvalue(), source="semantic_fit_checkpoint")
        return path
