from __future__ import annotations

import copy

import pytest

from core.learning.cortex_serving_qualification import (
    SERVING_MEASUREMENT_SCHEMA,
    _prefill_liveness,
    _resumable_row,
    _seal_row,
    build_serving_qualification,
    canonical_sha256,
    critical_gates,
    recommended_lane_limits,
    score_code_answer,
    score_complete_answer,
    score_tool_answer,
)


def _passing_measurement() -> dict:
    value = {
        "schema": SERVING_MEASUREMENT_SCHEMA,
        "model_descriptor_sha256": "a" * 64,
        "evidence_binding_sha256": "b" * 64,
        "verdict": "PASS",
        "template_pass": True,
        "complete_answer_pass": True,
        "tool_contract_pass": True,
        "code_contract_pass": True,
        "context_pass": True,
        "latency_pass": True,
        "memory_pass": True,
        "served_context_tokens": 32768,
        "requested_context_tokens": 32768,
        "prefill_chunk_tokens": 1024,
        "maximum_peak_memory_gb": 24.0,
        "minimum_available_memory_gb": 18.0,
        "cells": [],
    }
    value["evidence_sha256"] = canonical_sha256(value)
    return value


def test_complete_answer_requires_every_predicate_and_a_natural_stop() -> None:
    answer = """CORE INVARIANT
Every settled vertex has the minimum final distance among unsettled vertices.

NUMBERED PSEUDOCODE
1. Set dist[A] = 0.
2. Remove the minimum tentative vertex.
3. Relax each edge when dist[u] plus its weight improves dist[v].
4. Mark that vertex settled and continue.

WORKED EXAMPLE
Use A->B=4, A->C=1, C->B=2, B->D=1, and C->D=5.
The final distances are A = 0, B = 3, C = 1, and D = 4.

COMPLEXITY
With a binary heap the bound is O((V+E) log V). With an array it is O(V^2 + E).

NEGATIVE WEIGHTS
A negative edge can improve a vertex after it was settled, invalidating the invariant. Bellman-Ford is the correct alternative.
"""
    answer += (" supporting detail" * 330) + "."

    score = score_complete_answer(answer, generation_tokens=700, finish_reason="stop")
    assert score["passed"] is True
    assert score_complete_answer(answer, generation_tokens=700, finish_reason="length")["passed"] is False
    assert score_complete_answer(
        answer.rstrip("."),
        generation_tokens=700,
        finish_reason="stop",
    )["passed"] is False
    assert score_complete_answer(
        answer + (" excess" * 900),
        generation_tokens=1700,
        finish_reason="stop",
    )["passed"] is False


def test_complete_answer_scores_semantic_edge_statements_not_one_wire_spelling() -> None:
    answer = """CORE INVARIANT
Every settled vertex has the minimum final distance among unsettled vertices.

NUMBERED PSEUDOCODE
1. Set dist[A] = 0.
2. Remove the minimum tentative vertex.
3. Relax each edge when dist[u] plus its weight improves dist[v].
4. Mark that vertex settled and continue.

WORKED EXAMPLE
Use A to B with weight 4, A -> C with a weight of 1, C->B:2,
B -> D weight 1, and C -> D = 5. The final distances are A = 0,
B = 3, C = 1, and D = 4.

COMPLEXITY
With a binary heap the bound is O((V+E) log V). With an array it is O(V^2 + E).

NEGATIVE WEIGHTS
A negative edge can improve a vertex after it was settled, invalidating the invariant.
Bellman-Ford is the correct alternative.
"""
    answer += (" supporting detail" * 330) + "."

    score = score_complete_answer(answer, generation_tokens=700, finish_reason="stop")
    assert score["passed"] is True
    assert score["edge_pass"] is True

    wrong_edge = answer.replace("C -> D = 5", "C -> D = 6")
    assert score_complete_answer(
        wrong_edge,
        generation_tokens=700,
        finish_reason="stop",
    )["edge_pass"] is False


def test_complete_answer_accepts_latex_edges_and_indexed_distances() -> None:
    answer = """CORE INVARIANT
Every settled vertex has the minimum final distance among unsettled vertices.

NUMBERED PSEUDOCODE
1. Initialize distances.
2. Extract the minimum vertex.
3. Relax every outgoing edge.
4. Mark the vertex settled.

WORKED EXAMPLE
Use $A \\to B = 4$, $A \\to C = 1$, $C \\to B = 2$, $B \\to D = 1$,
and $C \\to D = 5$. The final values are $dist[A] = 0$, $dist[B] = 3$,
$dist[C] = 1$, and $dist[D] = 4$.

COMPLEXITY
With a binary heap the bound is O((V+E) log V). With an array it is O(V^2 + E).

NEGATIVE WEIGHTS
A negative edge can improve a vertex after it was settled, invalidating the invariant.
Bellman-Ford is the correct alternative.
"""
    answer += (" supporting detail" * 330) + "."

    score = score_complete_answer(answer, generation_tokens=700, finish_reason="stop")
    assert score["passed"] is True
    assert score["edge_pass"] is True
    assert score["distance_pass"] is True


def test_native_qwen38_tool_wire_is_schema_bound() -> None:
    response = """<tool_call>
<function=calculate>
<parameter=expression>
17 * 19
</parameter>
</function>
</tool_call>"""
    score = score_tool_answer(response)
    assert score["passed"] is True
    assert score["parsed_call"] == {
        "tool": "calculate",
        "args": {"expression": "17 * 19"},
    }


def test_generated_code_is_executed_behind_the_kernel_boundary() -> None:
    code = """```python
def rolling_pair_sums(values):
    iterator = iter(values)
    try:
        previous = next(iterator)
    except StopIteration:
        return []
    result = []
    for current in iterator:
        result.append(previous + current)
        previous = current
    return result
```"""
    score = score_code_answer(code)
    assert score["passed"] is True
    assert score["sandbox"]["sandboxed"] is True


def test_prefill_liveness_requires_moving_complete_callbacks() -> None:
    good = {
        "prompt_tokens": 4096,
        "prompt_tps": 500.0,
        "generation_tps": 20.0,
        "ttft_s": 10.0,
        "progress": [
            {"processed": 0, "total": 4096, "elapsed_s": 0.0},
            {"processed": 2048, "total": 4096, "elapsed_s": 4.0},
            {"processed": 4096, "total": 4096, "elapsed_s": 8.0},
        ],
    }
    assert _prefill_liveness(good)["passed"] is True
    stalled = copy.deepcopy(good)
    stalled["progress"][-1]["elapsed_s"] = 80.0
    assert _prefill_liveness(stalled)["passed"] is False
    incomplete = copy.deepcopy(good)
    incomplete["progress"] = incomplete["progress"][:-1]
    assert _prefill_liveness(incomplete)["passed"] is False
    wrong_total = copy.deepcopy(good)
    wrong_total["progress"][1]["total"] = 9999
    assert _prefill_liveness(wrong_total)["passed"] is False
    duplicate = copy.deepcopy(good)
    duplicate["progress"].insert(2, copy.deepcopy(duplicate["progress"][1]))
    assert _prefill_liveness(duplicate)["passed"] is False


def test_short_prefill_uses_ttft_not_a_startup_dominated_rate() -> None:
    short = {
        "prompt_tokens": 285,
        "prompt_tps": 139.0,
        "generation_tps": 7.4,
        "ttft_s": 2.4,
        "progress": [
            {"processed": 0, "total": 285, "elapsed_s": 0.0},
            {"processed": 284, "total": 285, "elapsed_s": 1.8},
            {"processed": 285, "total": 285, "elapsed_s": 2.3},
        ],
    }

    result = _prefill_liveness(short)
    assert result["passed"] is True
    assert result["prompt_throughput_applicable"] is False


def test_long_prefill_still_enforces_sustained_throughput() -> None:
    long = {
        "prompt_tokens": 8192,
        "prompt_tps": 139.0,
        "generation_tps": 7.4,
        "ttft_s": 60.0,
        "progress": [
            {"processed": 0, "total": 8192, "elapsed_s": 0.0},
            {"processed": 4096, "total": 8192, "elapsed_s": 29.0},
            {"processed": 8192, "total": 8192, "elapsed_s": 60.0},
        ],
    }

    result = _prefill_liveness(long)
    assert result["passed"] is False
    assert result["prompt_throughput_applicable"] is True
    assert result["prompt_throughput_pass"] is False


def test_only_passing_cells_are_reused_after_an_interruption() -> None:
    model_digest = "d" * 64
    binding_digest = "e" * 64
    rows = {
        "passing": _seal_row(
            cell_id="passing",
            model_descriptor_sha256=model_digest,
            evidence_binding_sha256=binding_digest,
            row={"contract_sha256": "a" * 64, "passed": True},
        ),
        "failed": {"contract_sha256": "b" * 64, "passed": False},
        "skipped": {
            "contract_sha256": "c" * 64,
            "passed": False,
            "skipped": True,
        },
    }

    kwargs = {
        "model_descriptor_sha256": model_digest,
        "evidence_binding_sha256": binding_digest,
    }
    assert _resumable_row(rows, "passing", "a" * 64, **kwargs) == rows["passing"]
    assert _resumable_row(rows, "failed", "b" * 64, **kwargs) is None
    assert _resumable_row(rows, "skipped", "c" * 64, **kwargs) is None


def test_qualification_binds_the_full_measurement() -> None:
    measurement = _passing_measurement()
    qualification = build_serving_qualification(measurement)
    assert qualification["evidence_sha256"] == measurement["evidence_sha256"]
    assert qualification["model_descriptor_sha256"] == "a" * 64
    assert qualification["template_pass"] is True
    assert qualification["tool_contract_pass"] is True
    assert qualification["code_contract_pass"] is True
    assert qualification["context_pass"] is True
    assert qualification["served_context_tokens"] == 32768

    tampered = copy.deepcopy(measurement)
    tampered["maximum_peak_memory_gb"] = 63.0
    with pytest.raises(ValueError, match="digest"):
        build_serving_qualification(tampered)


def test_failed_measurement_cannot_mint_a_pass() -> None:
    measurement = _passing_measurement()
    measurement["verdict"] = "FAIL"
    measurement.pop("evidence_sha256")
    measurement["evidence_sha256"] = canonical_sha256(measurement)
    with pytest.raises(ValueError, match="failed"):
        build_serving_qualification(measurement)


def test_recommended_limits_preserve_the_eight_thousand_token_runtime_ceiling() -> None:
    lanes = recommended_lane_limits(32768)
    assert lanes["foreground_simple"] == {
        "max_input_tokens": 8192,
        "max_output_tokens": 2048,
    }
    assert lanes["deep_reasoning"]["max_output_tokens"] == 8192
    assert all(
        limits["max_input_tokens"] + limits["max_output_tokens"] <= 32768
        for limits in lanes.values()
    )


def test_critical_gates_keep_identity_migration_separate() -> None:
    measurement = _passing_measurement()
    gates = critical_gates(measurement, identity_migration=False)
    assert gates == {
        "template": True,
        "complete_answer": True,
        "tool_contract": True,
        "code_contract": True,
        "context": True,
        "identity_migration": False,
        "latency": True,
        "memory": True,
    }
