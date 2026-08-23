"""Measured serving qualification for a candidate resident cortex.

Capability scores do not prove that a checkpoint can serve Aura. The model
must also finish long answers, speak the runtime's tool protocol, emit working
code, preserve evidence through the served context window, report prefill
progress, and fit on the host. This module measures those properties in one
loaded-model session and reduces them to the small qualification object bound
by :mod:`core.brain.llm.model_artifact_profile`.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from core.brain.llm.model_artifact_profile import SERVING_QUALIFICATION_SCHEMA

SERVING_MEASUREMENT_SCHEMA = "aura.cortex_upgrade.serving_measurement.v2"
SERVING_PROGRESS_SCHEMA = "aura.cortex_upgrade.serving_progress.v2"

DEFAULT_CONTEXT_WINDOWS = (8192, 32768)
DEFAULT_PREFILL_CHUNK_TOKENS = 1024
_MIN_PROMPT_TPS = 150.0
_MIN_PROMPT_TPS_MEASUREMENT_TOKENS = 2048
_MIN_GENERATION_TPS = 2.0
_MAX_PREFILL_SILENCE_S = 30.0
_MAX_TTFT_S = 240.0
_MIN_HOST_MARGIN_GB = 5.0
_MAX_HOST_FRACTION = 0.82

_COMPLETE_PROMPT = """Explain Dijkstra's shortest-path algorithm in one complete response of 700 to 1400 words.
Use these exact section headings: CORE INVARIANT, NUMBERED PSEUDOCODE, WORKED EXAMPLE, COMPLEXITY, NEGATIVE WEIGHTS.
The worked example must use the directed weighted edges A->B=4, A->C=1, C->B=2, B->D=1, and C->D=5, and must state the final distances from A to A, B, C, and D.
Give binary-heap and array time complexity. Explain why negative edges invalidate Dijkstra and name the correct alternative.
Finish the final sentence. Do not omit a section."""

_CODE_PROMPT = '''Write complete Python for exactly this function and no other public function:

def rolling_pair_sums(values):
    """Return each adjacent pair's sum; return [] when fewer than two values exist."""

Return the implementation as one Python code block. It must work for integers, floats, negative values, tuples, and generators without mutating the input.'''

_TOOL_DEFINITIONS: dict[str, dict[str, Any]] = {
    "calculate": {
        "description": "Evaluate one exact arithmetic expression.",
        "parameters": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
            "additionalProperties": False,
        },
    }
}


@dataclass(frozen=True)
class GenerationMeasurement:
    text: str
    prompt_tokens: int
    generation_tokens: int
    prompt_tps: float
    generation_tps: float
    peak_memory_gb: float
    mlx_active_memory_gb: float
    mlx_cache_memory_gb: float
    process_peak_rss_gb: float
    minimum_host_available_gb: float
    finish_reason: str
    elapsed_s: float
    ttft_s: float
    progress: tuple[dict[str, float | int], ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _contract_sha256(name: str, payload: Mapping[str, Any]) -> str:
    return canonical_sha256({"cell": name, "contract": dict(payload)})


def _row_sha256(
    *,
    cell_id: str,
    model_descriptor_sha256: str,
    evidence_binding_sha256: str,
    row: Mapping[str, Any],
) -> str:
    material = dict(row)
    material.pop("row_sha256", None)
    return canonical_sha256(
        {
            "cell_id": cell_id,
            "model_descriptor_sha256": model_descriptor_sha256,
            "evidence_binding_sha256": evidence_binding_sha256,
            "row": material,
        }
    )


def _seal_row(
    *,
    cell_id: str,
    model_descriptor_sha256: str,
    evidence_binding_sha256: str,
    row: Mapping[str, Any],
) -> dict[str, Any]:
    sealed = dict(row)
    sealed["row_sha256"] = _row_sha256(
        cell_id=cell_id,
        model_descriptor_sha256=model_descriptor_sha256,
        evidence_binding_sha256=evidence_binding_sha256,
        row=sealed,
    )
    return sealed


def _resume_rows(
    events: Sequence[Mapping[str, Any]] | None,
    *,
    event_validator: Callable[[Mapping[str, Any]], bool] | None,
) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for event in events or ():
        if (
            isinstance(event, Mapping)
            and event.get("schema") == SERVING_PROGRESS_SCHEMA
            and isinstance(event.get("cell_id"), str)
            and isinstance(event.get("row"), Mapping)
            and event_validator is not None
            and event_validator(event)
        ):
            rows[str(event["cell_id"])] = dict(event["row"])
    return rows


def _resumable_row(
    rows: Mapping[str, Mapping[str, Any]],
    cell_id: str,
    contract_sha256: str,
    *,
    model_descriptor_sha256: str,
    evidence_binding_sha256: str,
) -> dict[str, Any] | None:
    row = rows.get(cell_id)
    if (
        isinstance(row, Mapping)
        and row.get("contract_sha256") == contract_sha256
        and row.get("passed") is True
        and row.get("row_sha256")
        == _row_sha256(
            cell_id=cell_id,
            model_descriptor_sha256=model_descriptor_sha256,
            evidence_binding_sha256=evidence_binding_sha256,
            row=row,
        )
    ):
        return dict(row)
    return None


def _skipped_row(contract_sha256: str, *, reason: str) -> dict[str, Any]:
    """Record an unmeasured cell without making it reusable evidence."""

    return {
        "contract_sha256": contract_sha256,
        "passed": False,
        "skipped": True,
        "skip_reason": str(reason),
    }


def _emit(
    callback: Callable[[dict[str, Any]], None] | None,
    *,
    cell_id: str,
    completed: int,
    total: int,
    row: Mapping[str, Any],
) -> None:
    if callback is None:
        return
    callback(
        {
            "schema": SERVING_PROGRESS_SCHEMA,
            "cell_id": cell_id,
            "completed": completed,
            "total": total,
            "row": dict(row),
            "updated_at": time.time(),
        }
    )


def _host_memory() -> dict[str, float]:
    try:
        import psutil

        memory = psutil.virtual_memory()
        return {
            "total_gb": round(float(memory.total) / 1024**3, 4),
            "available_gb": round(float(memory.available) / 1024**3, 4),
        }
    except (ImportError, AttributeError, OSError, TypeError, ValueError):
        return {"total_gb": 0.0, "available_gb": 0.0}


def _resource_sample(mx: object) -> dict[str, float]:
    host = _host_memory()
    rss_gb = 0.0
    try:
        import psutil

        rss_gb = float(psutil.Process().memory_info().rss) / 1024**3
    except (ImportError, AttributeError, OSError, TypeError, ValueError):
        pass

    def mlx_gb(name: str) -> float:
        getter = getattr(mx, name, None)
        if not callable(getter):
            return 0.0
        try:
            return float(getter()) / 1024**3
        except (RuntimeError, TypeError, ValueError):
            return 0.0

    return {
        "host_available_gb": float(host.get("available_gb") or 0.0),
        "mlx_active_gb": mlx_gb("get_active_memory"),
        "mlx_cache_gb": mlx_gb("get_cache_memory"),
        "mlx_peak_gb": mlx_gb("get_peak_memory"),
        "process_rss_gb": rss_gb,
    }


def _render_tokens(
    tokenizer: object,
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None = None,
    enable_thinking: bool,
) -> list[int]:
    from core.brain.llm.chat_format import render_chat_template

    rendered = render_chat_template(
        tokenizer,
        messages,
        tools=tools,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    return list(tokenizer.encode(rendered, add_special_tokens=False))


def _generate(
    model: object,
    tokenizer: object,
    prompt_tokens: Sequence[int],
    *,
    max_tokens: int,
    prefill_chunk_tokens: int,
) -> GenerationMeasurement:
    import mlx.core as mx
    from mlx_lm import stream_generate

    initial_resources = _resource_sample(mx)
    mx.reset_peak_memory()
    started = time.monotonic()
    progress: list[dict[str, float | int]] = []
    resource_samples = [initial_resources]

    def prefill_progress(processed: int, total: int) -> None:
        resources = _resource_sample(mx)
        resource_samples.append(resources)
        progress.append(
            {
                "processed": int(processed),
                "total": int(total),
                "elapsed_s": round(time.monotonic() - started, 6),
                **{name: round(value, 6) for name, value in resources.items()},
            }
        )

    pieces: list[str] = []
    final = None
    first_token_at: float | None = None
    for response in stream_generate(
        model,
        tokenizer,
        list(prompt_tokens),
        max_tokens=int(max_tokens),
        sampler=lambda logits: mx.argmax(logits, axis=-1),
        prefill_step_size=int(prefill_chunk_tokens),
        prompt_progress_callback=prefill_progress,
    ):
        if first_token_at is None:
            first_token_at = time.monotonic()
        pieces.append(str(response.text or ""))
        final = response
    elapsed = time.monotonic() - started
    if final is None:
        raise RuntimeError("serving_generation_produced_no_frames")
    resource_samples.append(_resource_sample(mx))
    mlx_active = max(
        (sample["mlx_active_gb"] for sample in resource_samples),
        default=0.0,
    )
    mlx_cache = max(
        (sample["mlx_cache_gb"] for sample in resource_samples),
        default=0.0,
    )
    mlx_peak = max(
        float(getattr(final, "peak_memory", 0.0) or 0.0),
        mlx_active,
        max((sample["mlx_peak_gb"] for sample in resource_samples), default=0.0),
    )
    process_peak_rss = max(
        (sample["process_rss_gb"] for sample in resource_samples),
        default=0.0,
    )
    host_available = [
        sample["host_available_gb"]
        for sample in resource_samples
        if sample["host_available_gb"] > 0
    ]
    return GenerationMeasurement(
        text="".join(pieces),
        prompt_tokens=int(final.prompt_tokens),
        generation_tokens=int(final.generation_tokens),
        prompt_tps=round(float(final.prompt_tps), 4),
        generation_tps=round(float(final.generation_tps), 4),
        peak_memory_gb=round(mlx_peak, 4),
        mlx_active_memory_gb=round(mlx_active, 4),
        mlx_cache_memory_gb=round(mlx_cache, 4),
        process_peak_rss_gb=round(process_peak_rss, 4),
        minimum_host_available_gb=round(min(host_available, default=0.0), 4),
        finish_reason=str(final.finish_reason or ""),
        elapsed_s=round(elapsed, 4),
        ttft_s=round((first_token_at or time.monotonic()) - started, 4),
        progress=tuple(progress),
    )


def _prefill_liveness(generation: Mapping[str, Any]) -> dict[str, Any]:
    events = generation.get("progress")
    events = list(events) if isinstance(events, (list, tuple)) else []
    processed: list[int] = []
    elapsed: list[float] = []
    reported_totals: list[int] = []
    total = int(generation.get("prompt_tokens") or 0)
    for event in events:
        if not isinstance(event, Mapping):
            continue
        processed.append(int(event.get("processed") or 0))
        elapsed.append(float(event.get("elapsed_s") or 0.0))
        reported_totals.append(int(event.get("total") or 0))
    monotonic = all(
        a < b for a, b in zip(processed[:-1], processed[1:], strict=True)
    )
    elapsed_monotonic = all(
        0.0 <= a <= b
        for a, b in zip(elapsed[:-1], elapsed[1:], strict=True)
    )
    totals_match = bool(reported_totals) and all(
        reported == total for reported in reported_totals
    )
    bounds_valid = bool(processed) and all(0 <= value <= total for value in processed)
    starts_at_zero = bool(processed) and processed[0] == 0
    gaps = [
        b - a for a, b in zip(elapsed[:-1], elapsed[1:], strict=True)
    ]
    max_gap = max(gaps, default=0.0)
    complete = starts_at_zero and processed[-1] == total
    prompt_throughput_applicable = total >= _MIN_PROMPT_TPS_MEASUREMENT_TOKENS
    prompt_throughput_pass = (
        not prompt_throughput_applicable
        or float(generation.get("prompt_tps") or 0.0) >= _MIN_PROMPT_TPS
    )
    passed = (
        monotonic
        and elapsed_monotonic
        and totals_match
        and bounds_valid
        and complete
        and max_gap <= _MAX_PREFILL_SILENCE_S
        and prompt_throughput_pass
        and float(generation.get("generation_tps") or 0.0) >= _MIN_GENERATION_TPS
        and float(generation.get("ttft_s") or math.inf) <= _MAX_TTFT_S
    )
    return {
        "passed": passed,
        "callbacks": len(processed),
        "monotonic": monotonic,
        "elapsed_monotonic": elapsed_monotonic,
        "totals_match": totals_match,
        "bounds_valid": bounds_valid,
        "starts_at_zero": starts_at_zero,
        "complete": complete,
        "prompt_throughput_applicable": prompt_throughput_applicable,
        "prompt_throughput_pass": prompt_throughput_pass,
        "max_callback_gap_s": round(max_gap, 4),
        "minimum_prompt_tps": _MIN_PROMPT_TPS,
        "minimum_prompt_tps_measurement_tokens": _MIN_PROMPT_TPS_MEASUREMENT_TOKENS,
        "minimum_generation_tps": _MIN_GENERATION_TPS,
        "maximum_ttft_s": _MAX_TTFT_S,
        "maximum_callback_gap_s": _MAX_PREFILL_SILENCE_S,
    }


def score_complete_answer(text: str, *, generation_tokens: int, finish_reason: str) -> dict[str, Any]:
    body = str(text or "")
    lowered = body.casefold()
    compact = re.sub(r"\s+", "", lowered)
    headings = (
        "core invariant",
        "numbered pseudocode",
        "worked example",
        "complexity",
        "negative weights",
    )
    heading_pass = all(heading in lowered for heading in headings)
    normalized_edges = (
        lowered.replace("\\to", "->")
        .replace("→", "->")
        .replace("−", "-")
        .replace("$", "")
    )

    def states_edge(source: str, target: str, weight: int) -> bool:
        return bool(
            re.search(
                rf"\b{source}\s*(?:->|\bto\b)\s*{target}\b"
                rf"(?:\s*(?:=|:)|\s+(?:with\s+)?(?:a\s+)?weight(?:\s+of)?)\s*{weight}\b",
                normalized_edges,
            )
        )

    edges = all(
        states_edge(source, target, weight)
        for source, target, weight in (
            ("a", "b", 4),
            ("a", "c", 1),
            ("c", "b", 2),
            ("b", "d", 1),
            ("c", "d", 5),
        )
    )
    distances = all(
        re.search(pattern, lowered)
        for pattern in (
            r"\b(?:dist\s*\[\s*)?a\s*\]?\s*(?:=|:|is)\s*0\b",
            r"\b(?:dist\s*\[\s*)?b\s*\]?\s*(?:=|:|is)\s*3\b",
            r"\b(?:dist\s*\[\s*)?c\s*\]?\s*(?:=|:|is)\s*1\b",
            r"\b(?:dist\s*\[\s*)?d\s*\]?\s*(?:=|:|is)\s*4\b",
        )
    )
    invariant = bool(
        re.search(r"\b(?:settled|finali[sz]ed|visited)\b", lowered)
        and "minimum" in lowered
        and "distance" in lowered
    )
    pseudocode = "relax" in lowered and len(re.findall(r"(?m)^\s*\d+[.)]", body)) >= 4
    complexity = (
        "binary heap" in lowered
        and ("log" in lowered and "v" in lowered and "e" in lowered)
        and "array" in lowered
        and ("v^2" in compact or "v²" in compact or "v2" in compact)
    )
    negative = "negative" in lowered and "bellman" in lowered
    word_count = len(re.findall(r"\b\w+[\w'-]*\b", body))
    terminal_sentence = bool(re.search(r"[.!?][\"')\]]?\s*\Z", body))
    length_pass = 700 <= word_count <= 1400
    complete = (
        str(finish_reason) == "stop"
        and length_pass
        and terminal_sentence
        and int(generation_tokens) > 0
    )
    passed = all((heading_pass, edges, distances, invariant, pseudocode, complexity, negative, complete))
    return {
        "passed": passed,
        "heading_pass": heading_pass,
        "edge_pass": edges,
        "distance_pass": distances,
        "invariant_pass": invariant,
        "pseudocode_pass": pseudocode,
        "complexity_pass": complexity,
        "negative_weight_pass": negative,
        "completion_pass": complete,
        "length_pass": length_pass,
        "terminal_sentence_pass": terminal_sentence,
        "word_count": word_count,
    }


def score_tool_answer(text: str) -> dict[str, Any]:
    from core.brain.llm.mlx_client import MLXLocalClient, _tool_arguments_schema_error

    call = MLXLocalClient._extract_tool_call_payload(
        text,
        allowed_tools=set(_TOOL_DEFINITIONS),
        tool_definitions=_TOOL_DEFINITIONS,
    )
    schema_error = "missing_call"
    exact = False
    if isinstance(call, dict):
        name = str(call.get("tool") or "")
        args = call.get("args")
        schema_error = _tool_arguments_schema_error(_TOOL_DEFINITIONS.get(name), args)
        expression = re.sub(r"\s+", "", str((args or {}).get("expression") or ""))
        exact = name == "calculate" and expression in {"17*19", "19*17"}
    return {
        "passed": bool(call) and not schema_error and exact,
        "parsed_call": call,
        "schema_error": schema_error,
        "exact_objective": exact,
    }


def score_code_answer(text: str) -> dict[str, Any]:
    from core.brain.llm.code_generator import extract_python_code
    from core.sandbox.untrusted_python import run_untrusted_script

    code = extract_python_code(text)
    assertions = """
assert rolling_pair_sums([1, 2, 3]) == [3, 5]
assert rolling_pair_sums((-2, 5, 0)) == [3, 5]
assert rolling_pair_sums([]) == []
assert rolling_pair_sums([3.5, -1.5]) == [2.0]
assert rolling_pair_sums((value for value in [2, 4, 8])) == [6, 12]
"""
    outcome = run_untrusted_script(
        code + "\n" + assertions,
        timeout_s=5.0,
        require_boundary=True,
        source="cortex_upgrade.serving_qualification",
    )
    passed = bool(outcome.ok and outcome.sandboxed)
    return {
        "passed": passed,
        "code_sha256": hashlib.sha256(code.encode("utf-8", "replace")).hexdigest(),
        "sandbox": outcome.to_dict(),
    }


def _context_prompt_tokens(tokenizer: object, *, target_tokens: int, nonce: str) -> list[int]:
    prefix = f"The evidence key is {nonce}. Keep it available while reading the archive.\n"
    suffix = "\nEnd of archive. Reply with only the evidence key stated at the beginning."
    filler = "Archived neutral telemetry record; no key appears on this line.\n"

    def render(repetitions: int) -> list[int]:
        content = prefix + (filler * repetitions) + suffix
        return _render_tokens(
            tokenizer,
            [{"role": "user", "content": content}],
            enable_thinking=False,
        )

    unit_tokens = max(1, len(list(tokenizer.encode(filler, add_special_tokens=False))))
    high = max(1, target_tokens // unit_tokens + 4)
    while len(render(high)) <= target_tokens:
        high *= 2
    low = 0
    best = render(0)
    while low <= high:
        middle = (low + high) // 2
        candidate = render(middle)
        if len(candidate) <= target_tokens:
            best = candidate
            low = middle + 1
        else:
            high = middle - 1
    if len(best) < int(target_tokens * 0.98):
        raise RuntimeError(
            f"context_prompt_underfilled:{len(best)}/{target_tokens}"
        )
    return best


def _cell_generation_row(
    *,
    contract_sha256: str,
    generation: GenerationMeasurement,
    score: Mapping[str, Any],
    memory_before: Mapping[str, float],
    memory_after: Mapping[str, float],
) -> dict[str, Any]:
    liveness = _prefill_liveness(generation.to_dict())
    return {
        "contract_sha256": contract_sha256,
        "passed": bool(score.get("passed")) and bool(liveness["passed"]),
        "score": dict(score),
        "generation": generation.to_dict(),
        "liveness": liveness,
        "memory_before": dict(memory_before),
        "memory_after": dict(memory_after),
    }


def run_loaded_serving_qualification(
    model: object,
    tokenizer: object,
    *,
    model_descriptor_sha256: str,
    evidence_binding_sha256: str,
    context_windows: Sequence[int] = DEFAULT_CONTEXT_WINDOWS,
    prefill_chunk_tokens: int = DEFAULT_PREFILL_CHUNK_TOKENS,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    resume_events: Sequence[Mapping[str, Any]] | None = None,
    resume_event_validator: Callable[[Mapping[str, Any]], bool] | None = None,
) -> dict[str, Any]:
    """Run every serving cell without releasing the loaded checkpoint."""

    windows = tuple(sorted({int(value) for value in context_windows}))
    if not windows or windows[0] < 2048 or windows[-1] > 32768:
        raise ValueError("served context windows must lie inside [2048, 32768]")
    if not 128 <= int(prefill_chunk_tokens) <= 8192:
        raise ValueError("prefill chunk must lie inside [128, 8192]")
    if not re.fullmatch(r"[0-9a-f]{64}", str(model_descriptor_sha256)):
        raise ValueError("model_descriptor_sha256_invalid")
    if not re.fullmatch(r"[0-9a-f]{64}", str(evidence_binding_sha256)):
        raise ValueError("evidence_binding_sha256_invalid")

    resumed = _resume_rows(
        resume_events,
        event_validator=resume_event_validator,
    )
    cell_ids = ["template", "complete_answer", "tool_contract", "code_contract"]
    cell_ids.extend(f"context:{window}" for window in windows)
    total = len(cell_ids)
    rows: list[dict[str, Any]] = []

    def run_cell(
        cell_id: str,
        contract: Mapping[str, Any],
        producer: Callable[[str], dict[str, Any]],
    ) -> dict[str, Any]:
        contract_digest = _contract_sha256(cell_id, contract)
        row = _resumable_row(
            resumed,
            cell_id,
            contract_digest,
            model_descriptor_sha256=model_descriptor_sha256,
            evidence_binding_sha256=evidence_binding_sha256,
        )
        if row is None:
            row = _seal_row(
                cell_id=cell_id,
                model_descriptor_sha256=model_descriptor_sha256,
                evidence_binding_sha256=evidence_binding_sha256,
                row=producer(contract_digest),
            )
            _emit(
                progress_callback,
                cell_id=cell_id,
                completed=len(rows) + 1,
                total=total,
                row=row,
            )
        rows.append({"cell_id": cell_id, **row})
        return row

    def template_cell(contract_digest: str) -> dict[str, Any]:
        from core.brain.llm.chat_format import template_supports_thinking

        plain = _render_tokens(
            tokenizer,
            [{"role": "user", "content": "template probe"}],
            enable_thinking=False,
        )
        tool_specs = [
            {"type": "function", "function": {"name": name, **definition}}
            for name, definition in _TOOL_DEFINITIONS.items()
        ]
        tool = _render_tokens(
            tokenizer,
            [{"role": "user", "content": "tool template probe"}],
            tools=tool_specs,
            enable_thinking=False,
        )
        passed = bool(
            plain
            and tool
            and plain != tool
            and template_supports_thinking(tokenizer)
            and getattr(tokenizer, "has_tool_calling", False)
        )
        return {
            "contract_sha256": contract_digest,
            "passed": passed,
            "plain_prompt_tokens": len(plain),
            "tool_prompt_tokens": len(tool),
            "thinking_control": template_supports_thinking(tokenizer),
            "native_tool_calling": bool(getattr(tokenizer, "has_tool_calling", False)),
        }

    template_row = run_cell(
        "template",
        {"requires": ["chat_template", "thinking_control", "native_tool_calling"]},
        template_cell,
    )

    def generated_cell(
        contract_digest: str,
        *,
        prompt_tokens: Sequence[int],
        max_tokens: int,
        scorer: Callable[[str, GenerationMeasurement], Mapping[str, Any]],
    ) -> dict[str, Any]:
        before = _host_memory()
        generation = _generate(
            model,
            tokenizer,
            prompt_tokens,
            max_tokens=max_tokens,
            prefill_chunk_tokens=prefill_chunk_tokens,
        )
        after = _host_memory()
        score = scorer(generation.text, generation)
        return _cell_generation_row(
            contract_sha256=contract_digest,
            generation=generation,
            score=score,
            memory_before=before,
            memory_after=after,
        )

    complete_tokens = _render_tokens(
        tokenizer,
        [{"role": "user", "content": _COMPLETE_PROMPT}],
        enable_thinking=False,
    )
    complete_row = run_cell(
        "complete_answer",
        {
            "prompt_sha256": hashlib.sha256(_COMPLETE_PROMPT.encode()).hexdigest(),
            "max_tokens": 4096,
            "prefill_chunk_tokens": prefill_chunk_tokens,
        },
        lambda digest: generated_cell(
            digest,
            prompt_tokens=complete_tokens,
            max_tokens=4096,
            scorer=lambda text, generation: score_complete_answer(
                text,
                generation_tokens=generation.generation_tokens,
                finish_reason=generation.finish_reason,
            ),
        ),
    )

    tool_specs = [
        {"type": "function", "function": {"name": name, **definition}}
        for name, definition in _TOOL_DEFINITIONS.items()
    ]
    tool_tokens = _render_tokens(
        tokenizer,
        [{"role": "user", "content": "Use calculate to evaluate 17 times 19."}],
        tools=tool_specs,
        enable_thinking=False,
    )
    tool_row = run_cell(
        "tool_contract",
        {
            "objective": "calculate:17*19",
            "max_tokens": 2048,
            "prefill_chunk_tokens": prefill_chunk_tokens,
        },
        lambda digest: generated_cell(
            digest,
            prompt_tokens=tool_tokens,
            max_tokens=2048,
            scorer=lambda text, _generation: score_tool_answer(text),
        ),
    )

    code_tokens = _render_tokens(
        tokenizer,
        [{"role": "user", "content": _CODE_PROMPT}],
        enable_thinking=False,
    )
    code_row = run_cell(
        "code_contract",
        {
            "prompt_sha256": hashlib.sha256(_CODE_PROMPT.encode()).hexdigest(),
            "max_tokens": 1024,
            "prefill_chunk_tokens": prefill_chunk_tokens,
        },
        lambda digest: generated_cell(
            digest,
            prompt_tokens=code_tokens,
            max_tokens=1024,
            scorer=lambda text, _generation: score_code_answer(text),
        ),
    )

    foundational_pass = all(
        row.get("passed") is True
        for row in (template_row, complete_row, tool_row, code_row)
    )
    previous_context_pass = foundational_pass
    for window in windows:
        nonce = f"AURA-{window}-CORTEX-9F3C"
        target = window - 64

        def context_producer(contract_digest: str, *, _target=target, _nonce=nonce):
            prompt = _context_prompt_tokens(
                tokenizer,
                target_tokens=_target,
                nonce=_nonce,
            )
            return generated_cell(
                contract_digest,
                prompt_tokens=prompt,
                max_tokens=32,
                scorer=lambda text, generation: {
                    "passed": (
                        _nonce.casefold() in text.casefold()
                        and generation.finish_reason == "stop"
                    ),
                    "nonce_recovered": _nonce.casefold() in text.casefold(),
                    "completion_pass": generation.finish_reason == "stop",
                },
            )

        contract = {
            "served_window": window,
            "target_prompt_tokens": target,
            "max_tokens": 32,
            "prefill_chunk_tokens": prefill_chunk_tokens,
            "nonce_sha256": hashlib.sha256(nonce.encode()).hexdigest(),
        }
        if not foundational_pass:
            reason = "foundational_serving_failure"

            def context_producer(
                contract_digest: str,
                *,
                _reason: str = reason,
            ) -> dict[str, Any]:
                return _skipped_row(contract_digest, reason=_reason)

        elif not previous_context_pass:
            reason = "prior_context_tier_failure"

            def context_producer(
                contract_digest: str,
                *,
                _reason: str = reason,
            ) -> dict[str, Any]:
                return _skipped_row(contract_digest, reason=_reason)

        context_row = run_cell(
            f"context:{window}",
            contract,
            context_producer,
        )
        previous_context_pass = context_row.get("passed") is True

    host_totals = [
        float(memory.get("total_gb") or 0.0)
        for row in rows
        for memory in (row.get("memory_before", {}), row.get("memory_after", {}))
        if isinstance(memory, Mapping)
    ]
    host_available = [
        float(memory.get("available_gb") or 0.0)
        for row in rows
        for memory in (row.get("memory_before", {}), row.get("memory_after", {}))
        if isinstance(memory, Mapping) and float(memory.get("available_gb") or 0.0) > 0
    ]
    host_available.extend(
        float(generation.get("minimum_host_available_gb") or 0.0)
        for row in rows
        if isinstance((generation := row.get("generation")), Mapping)
        and float(generation.get("minimum_host_available_gb") or 0.0) > 0
    )
    peaks = [
        max(
            float(generation.get("peak_memory_gb") or 0.0),
            float(generation.get("process_peak_rss_gb") or 0.0),
            float(generation.get("mlx_active_memory_gb") or 0.0)
            + float(generation.get("mlx_cache_memory_gb") or 0.0),
        )
        for row in rows
        if isinstance((generation := row.get("generation")), Mapping)
    ]
    total_gb = max(host_totals, default=0.0)
    minimum_available = min(host_available, default=0.0)
    maximum_peak = max(peaks, default=0.0)
    memory_pass = bool(
        total_gb > 0
        and minimum_available >= _MIN_HOST_MARGIN_GB
        and maximum_peak <= total_gb * _MAX_HOST_FRACTION
    )
    liveness_rows = [row.get("liveness") for row in rows if "generation" in row]
    latency_pass = bool(liveness_rows) and all(
        isinstance(value, Mapping) and value.get("passed") is True
        for value in liveness_rows
    )
    by_id = {str(row["cell_id"]): row for row in rows}
    complete_pass = by_id["complete_answer"]["score"]["passed"] is True
    tool_pass = by_id["tool_contract"]["score"]["passed"] is True
    code_pass = by_id["code_contract"]["score"]["passed"] is True
    context_pass = all(
        by_id[f"context:{window}"]["passed"] is True for window in windows
    )
    passed_context_windows = [
        window
        for window in windows
        if by_id[f"context:{window}"]["passed"] is True
    ]
    template_pass = by_id["template"]["passed"] is True
    verdict = "PASS" if all(
        (template_pass, complete_pass, tool_pass, code_pass, context_pass, latency_pass, memory_pass)
    ) else "FAIL"
    measurement: dict[str, Any] = {
        "schema": SERVING_MEASUREMENT_SCHEMA,
        "model_descriptor_sha256": str(model_descriptor_sha256),
        "evidence_binding_sha256": str(evidence_binding_sha256),
        "verdict": verdict,
        "template_pass": template_pass,
        "complete_answer_pass": complete_pass,
        "tool_contract_pass": tool_pass,
        "code_contract_pass": code_pass,
        "context_pass": context_pass,
        "latency_pass": latency_pass,
        "memory_pass": memory_pass,
        "requested_context_tokens": max(windows),
        "served_context_tokens": max(passed_context_windows, default=0),
        "prefill_chunk_tokens": int(prefill_chunk_tokens),
        "maximum_peak_memory_gb": round(maximum_peak, 4),
        "minimum_available_memory_gb": round(minimum_available, 4),
        "cells": rows,
    }
    measurement["evidence_sha256"] = canonical_sha256(measurement)
    return measurement


def build_serving_qualification(measurement: Mapping[str, Any]) -> dict[str, Any]:
    """Reduce the full evidence to the immutable serving-profile contract."""

    if measurement.get("schema") != SERVING_MEASUREMENT_SCHEMA:
        raise ValueError("serving_measurement_schema_invalid")
    claimed = measurement.get("evidence_sha256")
    material = dict(measurement)
    material.pop("evidence_sha256", None)
    if not isinstance(claimed, str) or claimed != canonical_sha256(material):
        raise ValueError("serving_measurement_digest_invalid")
    if measurement.get("verdict") != "PASS":
        raise ValueError("serving_measurement_failed")
    qualification = {
        "schema": SERVING_QUALIFICATION_SCHEMA,
        "verdict": "PASS",
        "model_descriptor_sha256": measurement.get("model_descriptor_sha256"),
        "template_pass": measurement.get("template_pass") is True,
        "complete_answer_pass": measurement.get("complete_answer_pass") is True,
        "tool_contract_pass": measurement.get("tool_contract_pass") is True,
        "code_contract_pass": measurement.get("code_contract_pass") is True,
        "context_pass": measurement.get("context_pass") is True,
        "latency_pass": measurement.get("latency_pass") is True,
        "memory_pass": measurement.get("memory_pass") is True,
        "served_context_tokens": int(measurement.get("served_context_tokens") or 0),
        "requested_context_tokens": int(
            measurement.get("requested_context_tokens") or 0
        ),
        "prefill_chunk_tokens": int(measurement.get("prefill_chunk_tokens") or 0),
        "evidence_sha256": claimed,
    }
    if not all(
        qualification[name] is True
        for name in (
            "template_pass",
            "complete_answer_pass",
            "tool_contract_pass",
            "code_contract_pass",
            "context_pass",
            "latency_pass",
            "memory_pass",
        )
    ):
        raise ValueError("serving_qualification_incomplete")
    if (
        qualification["served_context_tokens"] <= 0
        or qualification["served_context_tokens"]
        != qualification["requested_context_tokens"]
    ):
        raise ValueError("serving_qualification_context_incomplete")
    return qualification


def recommended_lane_limits(served_context_tokens: int) -> dict[str, dict[str, int]]:
    """Keep Aura's 8K output ceiling while allocating the measured window."""

    served = int(served_context_tokens)
    if served < 16384:
        raise ValueError("served_context_too_small_for_resident_cortex")
    output = min(8192, served // 4)
    simple_output = min(2048, output)
    return {
        "foreground_simple": {
            "max_input_tokens": min(8192, served - simple_output),
            "max_output_tokens": simple_output,
        },
        "foreground_standard": {
            "max_input_tokens": served - output,
            "max_output_tokens": output,
        },
        "foreground_extended": {
            "max_input_tokens": served - output,
            "max_output_tokens": output,
        },
        "deep_reasoning": {
            "max_input_tokens": served - output,
            "max_output_tokens": output,
        },
        "tool_execution": {
            "max_input_tokens": served - output,
            "max_output_tokens": output,
        },
        "code": {
            "max_input_tokens": served - output,
            "max_output_tokens": output,
        },
        "document": {
            "max_input_tokens": served - output,
            "max_output_tokens": output,
        },
    }


def critical_gates(measurement: Mapping[str, Any], *, identity_migration: bool) -> dict[str, bool]:
    return {
        "template": measurement.get("template_pass") is True,
        "complete_answer": measurement.get("complete_answer_pass") is True,
        "tool_contract": measurement.get("tool_contract_pass") is True,
        "code_contract": measurement.get("code_contract_pass") is True,
        "context": measurement.get("context_pass") is True,
        "identity_migration": bool(identity_migration),
        "latency": measurement.get("latency_pass") is True,
        "memory": measurement.get("memory_pass") is True,
    }


__all__ = [
    "DEFAULT_CONTEXT_WINDOWS",
    "DEFAULT_PREFILL_CHUNK_TOKENS",
    "SERVING_MEASUREMENT_SCHEMA",
    "SERVING_PROGRESS_SCHEMA",
    "build_serving_qualification",
    "canonical_sha256",
    "critical_gates",
    "recommended_lane_limits",
    "run_loaded_serving_qualification",
    "score_code_answer",
    "score_complete_answer",
    "score_tool_answer",
]
