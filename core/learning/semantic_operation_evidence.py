"""Source-trained operation evidence over resident token features.

This experimental selector uses the existing floor vocabulary and measured
operation spans. It neither parses answers nor observes validation targets.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.learning.procedure_induction import Program
from core.learning.semantic_program_floor import semantic_program_structural_key
from core.learning.semantic_span_pointer import _hidden_array


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("operation evidence needs a nonzero finite vector")
    return vector / norm


def _span_vector(features: np.ndarray, span) -> np.ndarray:
    span.validate_bound(len(features))
    if features.ndim != 2 or span.start == span.end or not np.isfinite(features).all():
        raise ValueError("operation evidence requires a measured token span")
    return _unit(features[span.start:span.end].mean(axis=0))


@dataclass(frozen=True)
class OperationEvidence:
    prototypes: dict[str, np.ndarray]
    support: dict[str, int]
    feature_width: int

    @classmethod
    def fit(cls, examples, *, source_ids: set[str]) -> OperationEvidence:
        if not source_ids:
            raise ValueError("operation prototypes need source training examples")
        vectors: dict[str, list[np.ndarray]] = {}
        seen = set()
        width = None
        for item in examples:
            source = item.ir.source_text_sha256
            if source not in source_ids:
                continue
            if item.split != "train" or source in seen:
                raise ValueError("operation prototypes require unique source-only examples")
            seen.add(source)
            features = np.asarray(_hidden_array(item.hidden_states), dtype=np.float32)
            if width is None:
                width = features.shape[1]
            if features.ndim != 2 or features.shape[1] != width:
                raise ValueError("operation source feature width changed")
            for instruction in item.ir.instructions:
                vectors.setdefault(instruction.op, []).append(
                    _span_vector(features, instruction.operation_span))
        if seen != source_ids or width is None:
            raise ValueError("operation source training cohort is incomplete")
        return cls({name: _unit(np.mean(rows, axis=0)) for name, rows in vectors.items()},
                   {name: len(rows) for name, rows in vectors.items()}, width)

    def scores(self, features, programs: tuple[Program, ...], operation_spans) -> np.ndarray:
        features = np.asarray(_hidden_array(features), dtype=np.float32)
        if (features.ndim != 2 or features.shape[1] != self.feature_width
                or not programs or len(programs) != len(operation_spans)):
            raise ValueError("operation candidate geometry differs from source training")
        values = []
        for program, spans in zip(programs, operation_spans, strict=True):
            if (semantic_program_structural_key(program) is None
                    or len(spans) != len(program.instructions)):
                raise ValueError("operation candidate is outside the typed floor")
            evidence = []
            for instruction, span in zip(program.instructions, spans, strict=True):
                prototype = self.prototypes.get(instruction.op)
                if prototype is None:
                    raise ValueError("operation candidate has no source-trained prototype")
                evidence.append(float(_span_vector(features, span) @ prototype))
            values.append(float(np.mean(evidence)))
        return np.asarray(values, dtype=np.float32)

    def choose(self, features, programs: tuple[Program, ...], operation_spans) -> int:
        return int(self.scores(features, programs, operation_spans).argmax())
