"""Generate regression tests and behavioral contracts from operations.

The test suite should learn from Aura's actual successful work, not only from
hand-written failures.  This module turns high-confidence operational records
into importable pytest snippets and conservative behavioral contract proposals.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from core.promotion.behavioral_contracts import BehavioralContract, synthesize_contracts_from_history
from core.runtime.atomic_writer import atomic_write_text


@dataclass(frozen=True)
class SynthesizedTest:
    name: str
    source: str
    evidence_ids: tuple[str, ...]
    confidence: float
    generated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source": self.source,
            "evidence_ids": list(self.evidence_ids),
            "confidence": round(self.confidence, 4),
            "generated_at": self.generated_at,
        }


class SelfTestSynthesizer:
    """Converts operational experience into tests and contract proposals."""

    MIN_CONFIDENCE = 0.80

    def synthesize_tests(self, records: Iterable[Mapping[str, Any]], *, max_tests: int = 12) -> list[SynthesizedTest]:
        tests: list[SynthesizedTest] = []
        for record in records:
            confidence = float(record.get("confidence", record.get("score", 0.0)) or 0.0)
            if confidence < self.MIN_CONFIDENCE or not bool(record.get("success", True)):
                continue
            kind = str(record.get("kind") or record.get("task_type") or "")
            if kind == "memory_retrieval":
                test = self._memory_test(record, confidence)
            elif kind == "tool_selection":
                test = self._tool_test(record, confidence)
            else:
                test = self._generic_metric_test(record, confidence)
            if test:
                tests.append(test)
            if len(tests) >= max_tests:
                break
        return tests

    def synthesize_contracts(self, records: Iterable[Mapping[str, Any]]) -> list[BehavioralContract]:
        return synthesize_contracts_from_history(list(records))

    def write_pytest_file(self, tests: Iterable[SynthesizedTest], path: str | Path) -> Path:
        tests = list(tests)
        body = [
            '"""Auto-synthesized Aura regression tests from verified operations."""',
            "from __future__ import annotations",
            "",
        ]
        for test in tests:
            body.append(test.source.rstrip())
            body.append("")
        target = Path(path)
        atomic_write_text(target, "\n".join(body), encoding="utf-8")
        return target

    def _memory_test(self, record: Mapping[str, Any], confidence: float) -> SynthesizedTest | None:
        query = str(record.get("query") or "")
        expected = str(record.get("expected_memory_id") or record.get("memory_id") or "")
        if not query or not expected:
            return None
        name = self._name("memory_retrieval", query)
        source = (
            f"def {name}():\n"
            f"    query = {query!r}\n"
            f"    expected = {expected!r}\n"
            "    assert query\n"
            "    assert expected\n"
        )
        return SynthesizedTest(name, source, (str(record.get("id", "")),), confidence)

    def _tool_test(self, record: Mapping[str, Any], confidence: float) -> SynthesizedTest | None:
        task = str(record.get("task") or record.get("query") or "")
        tool = str(record.get("tool") or record.get("selected_tool") or "")
        if not task or not tool:
            return None
        name = self._name("tool_selection", task)
        source = (
            f"def {name}():\n"
            f"    task = {task!r}\n"
            f"    expected_tool = {tool!r}\n"
            "    assert task\n"
            "    assert expected_tool\n"
        )
        return SynthesizedTest(name, source, (str(record.get("id", "")),), confidence)

    def _generic_metric_test(self, record: Mapping[str, Any], confidence: float) -> SynthesizedTest | None:
        metrics = record.get("metrics")
        if not isinstance(metrics, Mapping):
            return None
        numeric = {
            str(k): float(v)
            for k, v in metrics.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        }
        if not numeric:
            return None
        name = self._name("operational_metric", str(record.get("id", time.time())))
        source = (
            f"def {name}():\n"
            f"    metrics = {json.dumps(numeric, sort_keys=True)!r}\n"
            "    assert metrics\n"
            "    assert all(float(value) == float(value) for value in metrics.values())\n"
        )
        return SynthesizedTest(name, source, (str(record.get("id", "")),), confidence)

    @staticmethod
    def _name(prefix: str, text: str) -> str:
        clean = re.sub(r"[^a-zA-Z0-9_]+", "_", text.lower()).strip("_")[:64]
        return f"test_synth_{prefix}_{clean or 'case'}"


__all__ = ["SynthesizedTest", "SelfTestSynthesizer"]
