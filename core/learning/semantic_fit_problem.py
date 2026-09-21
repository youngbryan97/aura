"""Replay retained numerical fit inputs without rerunning semantic mining."""

from collections import Counter
from dataclasses import fields
from io import BytesIO
import json
from pathlib import Path

import numpy as np

from core.learning.semantic_argument_graph_learning import ArgumentScoreTerm
from core.learning.semantic_fit_checkpoint import fit_identity, write_fit_archive
from core.learning.semantic_operation_graph_learning import OperationEvidenceBank
from core.learning.semantic_relation_graph_learning import GraphChoiceNormalizer, RelationEvidenceBank, RelationGraphContrast
from typing import Any

_TYPES = {cls.__name__: cls for cls in (
    ArgumentScoreTerm, OperationEvidenceBank, RelationEvidenceBank, RelationGraphContrast, GraphChoiceNormalizer,
)}


def save_fit_problem(
    path: Any,
    *,
    identity: Any,
    initial: Any,
    contrasts: Any,
    options: Any,
) -> dict[str, Any]:
    """Store shared evidence once; the archive grants no qualification authority."""
    arrays, nodes, seen, array_names = {}, [], {}, {}

    def encode(value: tuple[Any, ...]) -> dict[str, Any]:
        if isinstance(value, np.generic):
            value = value.item()
        shared = isinstance(value, (np.ndarray, tuple)) or type(value) in _TYPES.values()
        if not shared:
            if value is not None and type(value) not in (str, bool, int, float):
                raise ValueError("unsupported fit evidence type")
            return {"literal": value}
        if id(value) in seen:
            return {"ref": seen[id(value)][1]}
        if isinstance(value, np.ndarray):
            if value.dtype.hasobject:
                raise ValueError("object-valued fit evidence is unsupported")
            digest = fit_identity(value)
            name = array_names.get(digest)
            if name is None:
                name = f"array_{len(arrays)}"
                arrays[name] = value
                array_names[digest] = name
            node = {"array": name}
        elif isinstance(value, tuple):
            node = {"tuple": [encode(part) for part in value]}
        else:
            node = {"type": type(value).__name__, "fields": {
                field.name: encode(getattr(value, field.name)) for field in fields(value)}}
        index = len(nodes)
        nodes.append(node)
        seen[id(value)] = (value, index)
        return {"ref": index}

    roots = {"initial": encode(tuple(initial)), "contrasts": encode(tuple(contrasts))}
    body = {"schema": "aura.semantic_fit_problem.v1", "identity": identity,
            "nodes": nodes, "roots": roots,
            "options": json.loads(json.dumps(options, allow_nan=False)), "serving_authority": False}
    write_fit_archive(path, body, arrays)


def load_fit_problem(path: Any, *, expected_identity: Any) -> Any:
    """Read only declared evidence classes and numeric arrays, never pickle."""
    with np.load(BytesIO(Path(path).read_bytes()), allow_pickle=False) as archive:
        metadata = json.loads(archive["metadata"].tobytes().decode())
        arrays = {name: archive[name] for name in archive.files if name != "metadata"}
    body = {key: value for key, value in metadata.items() if key != "sha256"}
    if (body.get("schema") != "aura.semantic_fit_problem.v1"
            or body.get("identity") != expected_identity
            or body.get("serving_authority") is not False
            or metadata.get("sha256") != fit_identity((body, arrays))):
        raise ValueError("fit problem identity or checksum differs")
    decoded = []
    remaining = Counter(node["array"] for node in body["nodes"] if set(node) == {"array"})

    def decode(value: Any) -> Any:
        if set(value) == {"literal"}:
            return value["literal"]
        if (set(value) != {"ref"} or type(value["ref"]) is not int
                or not 0 <= value["ref"] < len(decoded)):
            raise ValueError("fit problem reference is invalid")
        return decoded[value["ref"]]

    for node in body["nodes"]:
        if set(node) == {"array"}:
            # Equal values share storage, while distinct source arrays remain
            # distinct objects. Repeated references still resolve to one node.
            name = node["array"]
            remaining[name] -= 1
            value = arrays.pop(name) if remaining[name] == 0 else arrays[name].copy()
        elif set(node) == {"tuple"}:
            value = tuple(decode(part) for part in node["tuple"])
        elif set(node) == {"type", "fields"} and node["type"] in _TYPES:
            cls = _TYPES[node["type"]]
            names = set(node["fields"])
            expected = {field.name for field in fields(cls)}
            legacy_contrast = cls is RelationGraphContrast and names == expected - {"normalizer_terms"}
            if names != expected and not legacy_contrast:
                raise ValueError("fit problem fields differ")
            value = cls(**{name: decode(part) for name, part in node["fields"].items()})
        else:
            raise ValueError("fit problem node type is unsupported")
        decoded.append(value)
    return {**{name: decode(root) for name, root in body["roots"].items()},
            "options": body["options"], "identity": expected_identity}
