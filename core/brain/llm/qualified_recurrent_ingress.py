"""Exact public ingress for domain-qualified recurrent controller tissue.

The qualified controller is not a general chat policy. It may serve only the
public grammars named by its activation. This module recognizes those
grammars without an answer channel, reproduces the exact prompt-token boundary
used by training and evaluation, and renders only worker-authorized values.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import threading
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache, partial
from pathlib import Path
from typing import Any, Final

from core.runtime.lockdep import LockRank, checked_lock

QUALIFIED_RECURRENT_INGRESS_SCHEMA: Final = "aura.unified_intrinsic.qualified_ingress.v1"
QUALIFIED_RECURRENT_RESULT_SCHEMA: Final = "aura.unified_intrinsic.qualified_foreground_result.v1"
QUALIFIED_ANSWER_BRIDGE: Final = "\n\nFINAL_ANSWER: "
_HEX: Final = frozenset("0123456789abcdef")
_QUALIFIED_CPU_WORKERS: Final = 2
_qualified_cpu_executor_instance: ThreadPoolExecutor | None = None
_qualified_cpu_executor_lock = checked_lock(
    "qualified_recurrent_ingress.executor", rank=LockRank.REGISTRY
)
_qualified_cpu_thread_state = threading.local()

_TERMINAL = (
    r" You may reason before answering\. Finish with exactly one final line using the "
    r"envelope FINAL_ANSWER: <JSON object>\. FINAL_ANSWER is the envelope, not a "
    r"JSON key\. The JSON object must contain exactly these keys: (?P<keys>[^.]+)\."
)
_KHOP_PROMPT = re.compile(
    r"\AA functional directed graph has these edges: (?P<edges>[0-9]+->[0-9]+"
    r"(?:, [0-9]+->[0-9]+)*)\. Start at (?P<start>[0-9]+) and follow exactly "
    r"(?P<depth>[1-9][0-9]*) edges\." + _TERMINAL + r"\Z"
)
_REGISTER_PROMPT = re.compile(
    r"\ATrace three registers from r0=(?P<r0>[0-9]+), r1=(?P<r1>[0-9]+), "
    r"r2=(?P<r2>[0-9]+)\. Apply in order: (?P<operations>"
    r"r[0-2]=\(r[0-2]\+[1-3]\*r[0-2]\+[1-7]\) mod 29"
    r"(?:; r[0-2]=\(r[0-2]\+[1-3]\*r[0-2]\+[1-7]\) mod 29)*)\. End" + _TERMINAL + r"\Z"
)
_REGISTER_ACTION = re.compile(
    r"r(?P<destination>[0-2])=\(r(?P<left>[0-2])\+(?P<multiplier>[1-3])\*"
    r"r(?P<right>[0-2])\+(?P<offset>[1-7])\) mod 29"
)
_RESULT_KEYS: Final = {
    "khop": ("node",),
    "modular": ("residue",),
    "register_trace": ("r0", "r1", "r2"),
}
_SEMANTIC_PARSER_IDS: Final = {
    "frontier_coding": "semantic_coding_canonical.v1",
    "frontier_calibration": "semantic_calibration_canonical.v1",
    "frontier_misleading_premise": "semantic_premise_canonical.v1",
    "frontier_scientific_inference": "semantic_scientific_canonical.v1",
}
_SCIENTIFIC_SURFACE_PARSER_PREFIX: Final = "semantic_scientific_surface."
_QUALIFIED_FAMILIES: Final = frozenset({*_RESULT_KEYS, *_SEMANTIC_PARSER_IDS})


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _qualified_cpu_executor() -> ThreadPoolExecutor:
    """Return the isolated executor for short certified recurrent programs."""

    global _qualified_cpu_executor_instance
    if _qualified_cpu_executor_instance is None:
        with _qualified_cpu_executor_lock:
            if _qualified_cpu_executor_instance is None:
                _qualified_cpu_executor_instance = ThreadPoolExecutor(
                    max_workers=_QUALIFIED_CPU_WORKERS,
                    thread_name_prefix="aura-qualified-recurrent",
                )
    return _qualified_cpu_executor_instance


def _thread_semantic_machine() -> Any:
    """Return one verified semantic machine owned by the current worker.

    SemanticNeuralMachine resets its transition-local counters before every
    action, but its MLX arrays are mutable objects. Keeping one instance per
    dedicated executor thread removes repeated artifact loading without
    sharing mutable tissue across concurrent foreground requests.
    """

    from core.learning.semantic_neural_runtime_machine import (
        SemanticNeuralRuntimeMachine,
    )

    machine = getattr(_qualified_cpu_thread_state, "semantic_machine", None)
    if not isinstance(machine, SemanticNeuralRuntimeMachine):
        machine = SemanticNeuralRuntimeMachine()
        _qualified_cpu_thread_state.semantic_machine = machine
    return machine


def _execute_semantic_neural_state_cached(prompt: str, family: str) -> Any:
    from core.brain.llm.latent_cortex.semantic_neural_decode_context import (
        execute_semantic_neural_decode_state,
    )

    return execute_semantic_neural_decode_state(
        prompt,
        family,
        machine=_thread_semantic_machine(),
    )


def _execute_scientific_surface_cached(prompt: str) -> Any:
    from core.brain.llm.latent_cortex.semantic_surface_adapter import (
        execute_scientific_surface,
    )

    return execute_scientific_surface(
        prompt,
        machine=_thread_semantic_machine(),
    )


async def _run_qualified_cpu_bound(
    function: Any,
    /,
    *args: Any,
    timeout_s: float,
) -> Any:
    """Run certified bounded work without competing for asyncio's shared pool."""

    bounded_timeout = float(timeout_s)
    if not 0.0 < bounded_timeout <= 30.0:
        raise ValueError("qualified recurrent CPU timeout is invalid")
    future = asyncio.get_running_loop().run_in_executor(
        _qualified_cpu_executor(),
        partial(function, *args),
    )
    return await asyncio.wait_for(future, timeout=bounded_timeout)


@dataclass(frozen=True, slots=True)
class QualifiedRecurrentAdmission:
    """Answer-blind admission for one exact public task grammar."""

    schema: str
    family: str
    task_depth: int
    parser_id: str
    public_source_sha256: str
    syntax_sha256: str

    def __post_init__(self) -> None:
        if (
            self.schema != QUALIFIED_RECURRENT_INGRESS_SCHEMA
            or self.family not in _QUALIFIED_FAMILIES
            or type(self.task_depth) is not int
            or not 1 <= self.task_depth <= 64
            or not isinstance(self.parser_id, str)
            or not re.fullmatch(r"[a-z][a-z0-9_.-]{2,127}", self.parser_id)
            or not _is_sha(self.public_source_sha256)
            or not _is_sha(self.syntax_sha256)
        ):
            raise ValueError("qualified recurrent admission is invalid")

    def receipt(self) -> dict[str, Any]:
        body = {
            "schema": self.schema,
            "family": self.family,
            "task_depth": self.task_depth,
            "parser_id": self.parser_id,
            "public_source_sha256": self.public_source_sha256,
            "syntax_sha256": self.syntax_sha256,
        }
        return {**body, "receipt_sha256": _canonical_sha256(body)}


def _is_sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in _HEX for c in value)


def qualified_recurrent_result_receipt_errors(
    value: Any,
    *,
    answer_text: str,
    expected_family: str,
) -> list[str]:
    """Verify the typed result envelope that crosses into user-facing chat.

    Qualified recurrence does not use the resident text decoder.  Its answer is
    canonical serialization of an authenticated recurrent state, so ordinary
    model-generation ownership receipts do not apply.  This check binds the
    exact answer bytes to the answer-blind admission and the sealed result
    before chat may treat that distinction as meaningful.
    """

    if not isinstance(value, Mapping):
        return ["qualified_recurrent_result_receipt_missing"]
    body = {key: item for key, item in value.items() if key != "receipt_sha256"}
    errors: list[str] = []
    if value.get("schema") != QUALIFIED_RECURRENT_RESULT_SCHEMA:
        errors.append("qualified_recurrent_result_schema_invalid")
    if value.get("receipt_sha256") != _canonical_sha256(body):
        errors.append("qualified_recurrent_result_seal_invalid")
    answer = str(answer_text or "").strip()
    if (
        not answer
        or value.get("answer_sha256") != hashlib.sha256(answer.encode("utf-8")).hexdigest()
    ):
        errors.append("qualified_recurrent_answer_binding_invalid")

    admission_value = value.get("admission")
    if not isinstance(admission_value, Mapping):
        errors.append("qualified_recurrent_admission_receipt_missing")
    else:
        admission_body = {
            key: item for key, item in admission_value.items() if key != "receipt_sha256"
        }
        try:
            admission = QualifiedRecurrentAdmission(
                schema=str(admission_value.get("schema") or ""),
                family=str(admission_value.get("family") or ""),
                task_depth=admission_value.get("task_depth"),
                parser_id=str(admission_value.get("parser_id") or ""),
                public_source_sha256=str(admission_value.get("public_source_sha256") or ""),
                syntax_sha256=str(admission_value.get("syntax_sha256") or ""),
            )
        except (TypeError, ValueError):
            errors.append("qualified_recurrent_admission_identity_invalid")
        else:
            if (
                admission_value.get("receipt_sha256") != _canonical_sha256(admission_body)
                or dict(admission_value) != admission.receipt()
            ):
                errors.append("qualified_recurrent_admission_seal_invalid")
            if admission.family != str(expected_family or ""):
                errors.append("qualified_recurrent_family_binding_invalid")

    semantic_result = "semantic_state_receipt" in value
    typed_result = "worker_receipt" in value
    if semantic_result == typed_result:
        errors.append("qualified_recurrent_result_provenance_ambiguous")
    elif semantic_result:
        activation = value.get("activation_receipt")
        if (
            value.get("serialization") != "canonical_json_from_authenticated_semantic_state"
            or not isinstance(value.get("semantic_state_receipt"), Mapping)
            or not isinstance(activation, Mapping)
            or activation.get("promotion_mode") != "active"
        ):
            errors.append("qualified_recurrent_semantic_provenance_invalid")
    elif (
        not isinstance(value.get("worker_receipt"), Mapping)
        or type(value.get("public_token_count")) is not int
        or value.get("public_token_count", 0) <= 0
        or not _is_sha(value.get("public_tokens_sha256"))
    ):
        errors.append("qualified_recurrent_typed_provenance_invalid")
    return list(dict.fromkeys(errors))


def _admit_khop(prompt: str, match: re.Match[str]) -> QualifiedRecurrentAdmission:
    if match.group("keys") != "node":
        raise ValueError("qualified khop result contract differs")
    edge_pairs = tuple(
        tuple(int(part) for part in edge.split("->", 1))
        for edge in match.group("edges").split(", ")
    )
    left = tuple(pair[0] for pair in edge_pairs)
    size = len(edge_pairs)
    if left != tuple(range(size)) or any(not 0 <= pair[1] < size for pair in edge_pairs):
        raise ValueError("qualified khop graph is not a total canonical function")
    start = int(match.group("start"))
    depth = int(match.group("depth"))
    if not 0 <= start < size or not 1 <= depth <= 64:
        raise ValueError("qualified khop coordinates are outside the certified domain")
    syntax = {
        "edges": [list(pair) for pair in edge_pairs],
        "start": start,
        "depth": depth,
    }
    return QualifiedRecurrentAdmission(
        schema=QUALIFIED_RECURRENT_INGRESS_SCHEMA,
        family="khop",
        task_depth=depth,
        parser_id="khop_total_function.v1",
        public_source_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        syntax_sha256=_canonical_sha256(syntax),
    )


def _admit_register(prompt: str, match: re.Match[str]) -> QualifiedRecurrentAdmission:
    if match.group("keys") != "r0, r1, r2":
        raise ValueError("qualified register result contract differs")
    initial = tuple(int(match.group(key)) for key in ("r0", "r1", "r2"))
    if any(not 0 <= value < 29 for value in initial):
        raise ValueError("qualified register initial state is outside the certified domain")
    actions = []
    for text in match.group("operations").split("; "):
        action = _REGISTER_ACTION.fullmatch(text)
        if action is None:
            raise ValueError("qualified register action syntax differs")
        destination = int(action.group("destination"))
        left = int(action.group("left"))
        right = int(action.group("right"))
        if destination == left or destination == right or left == right:
            raise ValueError("qualified register action aliases its operands")
        actions.append(
            [
                destination,
                left,
                right,
                int(action.group("multiplier")),
                int(action.group("offset")),
                29,
            ]
        )
    if not 1 <= len(actions) <= 64:
        raise ValueError("qualified register depth is outside the certified domain")
    syntax = {"initial": list(initial), "actions": actions}
    return QualifiedRecurrentAdmission(
        schema=QUALIFIED_RECURRENT_INGRESS_SCHEMA,
        family="register_trace",
        task_depth=len(actions),
        parser_id="register_trace_canonical.v1",
        public_source_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        syntax_sha256=_canonical_sha256(syntax),
    )


def admit_qualified_recurrent_objective(
    prompt: str,
) -> QualifiedRecurrentAdmission | None:
    """Recognize only the exact public grammars certified for serving."""

    if not isinstance(prompt, str) or not prompt or prompt != prompt.strip() or "\x00" in prompt:
        return None
    khop = _KHOP_PROMPT.fullmatch(prompt)
    register = _REGISTER_PROMPT.fullmatch(prompt)
    if khop is not None:
        try:
            return _admit_khop(prompt, khop)
        except ValueError:
            return None
    if register is not None:
        try:
            return _admit_register(prompt, register)
        except ValueError:
            return None
    try:
        from core.learning.public_frontier_action_compiler import (
            compile_public_frontier_actions,
        )
        from core.learning.semantic_neural_controls import (
            classify_public_semantic_objective,
        )

    except ImportError:
        semantic_family = None
    else:
        semantic_family = classify_public_semantic_objective(prompt)
        if semantic_family is not None:
            try:
                program = compile_public_frontier_actions(prompt, semantic_family)
            except (TypeError, ValueError):
                return None
            if not 1 <= len(program.values) <= 64:
                return None
            return QualifiedRecurrentAdmission(
                schema=QUALIFIED_RECURRENT_INGRESS_SCHEMA,
                family=semantic_family,
                task_depth=len(program.values),
                parser_id=_SEMANTIC_PARSER_IDS[semantic_family],
                public_source_sha256=program.public_prompt_sha256,
                syntax_sha256=program.program_sha256,
            )
    try:
        from core.brain.llm.latent_cortex.semantic_surface_adapter import (
            SCIENTIFIC_FAMILY,
            parse_scientific_surface,
        )
        from core.learning.public_frontier_action_compiler import (
            compile_public_frontier_actions,
        )

        surface = parse_scientific_surface(prompt)
        canonical_program = compile_public_frontier_actions(
            surface.canonical_prompt,
            SCIENTIFIC_FAMILY,
        )
    except (ImportError, RuntimeError, TypeError, ValueError):
        pass
    else:
        if not 1 <= len(canonical_program.values) <= 64:
            return None
        surface_receipt = surface.receipt()
        return QualifiedRecurrentAdmission(
            schema=QUALIFIED_RECURRENT_INGRESS_SCHEMA,
            family=SCIENTIFIC_FAMILY,
            task_depth=len(canonical_program.values),
            parser_id=f"{_SCIENTIFIC_SURFACE_PARSER_PREFIX}{surface.profile}.v1",
            public_source_sha256=surface_receipt["public_prompt_sha256"],
            syntax_sha256=surface_receipt["public_fact_graph_sha256"],
        )
    try:
        from core.brain.llm.latent_cortex.typed_action_compiler import (
            compile_public_transition_program,
        )

        program = compile_public_transition_program(prompt)
    except (ImportError, ValueError):
        return None
    if program.family != "modular" or not 1 <= program.depth <= 64:
        return None
    return QualifiedRecurrentAdmission(
        schema=QUALIFIED_RECURRENT_INGRESS_SCHEMA,
        family="modular",
        task_depth=program.depth,
        parser_id="modular_operation_list.v1",
        public_source_sha256=program.public_source_sha256,
        syntax_sha256=program.program_sha256,
    )


@lru_cache(maxsize=2)
def _load_tokenizer(model_path: str) -> Any:
    from mlx_lm.utils import load_tokenizer

    return load_tokenizer(Path(model_path))


def project_qualified_public_tokens(tokenizer: Any, prompt: str) -> tuple[int, ...]:
    """Reproduce the exact CP275 public prefix, including its answer bridge."""

    render = getattr(tokenizer, "apply_chat_template", None)
    encode = getattr(tokenizer, "encode", None)
    if not callable(render) or not callable(encode):
        raise ValueError("qualified recurrent tokenizer contract is unavailable")
    rendered = render(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )
    if not isinstance(rendered, str) or not rendered:
        raise ValueError("qualified recurrent chat template is invalid")
    tokens = encode(rendered + QUALIFIED_ANSWER_BRIDGE)
    if (
        not isinstance(tokens, Sequence)
        or isinstance(tokens, (str, bytes, bytearray))
        or not tokens
        or len(tokens) > 16_384
        or any(type(token) is not int or token < 0 for token in tokens)
    ):
        raise ValueError("qualified recurrent public token projection is invalid")
    return tuple(tokens)


def render_qualified_recurrent_answer(
    family: str,
    parsed_values: Mapping[str, Any],
) -> str:
    expected = _RESULT_KEYS.get(family)
    if (
        expected is None
        or not isinstance(parsed_values, Mapping)
        or tuple(sorted(parsed_values)) != tuple(sorted(expected))
        or any(type(parsed_values[key]) is not int for key in expected)
    ):
        raise ValueError("qualified recurrent parsed answer contract differs")
    return "FINAL_ANSWER: " + json.dumps(
        {key: parsed_values[key] for key in expected},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


async def execute_qualified_recurrent_objective(
    client: Any | None,
    prompt: str,
    *,
    timeout_s: float,
) -> dict[str, Any]:
    """Execute one admitted task, exposing no authority outside its grammar."""

    admission = admit_qualified_recurrent_objective(prompt)
    if admission is None:
        return {
            "eligible": False,
            "attempted": False,
            "ok": False,
            "reason": "qualified_recurrent_objective_unsupported",
        }
    if admission.family in _SEMANTIC_PARSER_IDS:
        from core.brain.llm.latent_cortex.semantic_neural_decode_context import (
            render_semantic_neural_answer,
        )
        from core.brain.llm.semantic_neural_serving import (
            semantic_neural_default_serving_status,
            semantic_neural_serving_status,
        )

        # The promoted semantic machine is an authenticated, CPU-bound state
        # transition system.  It does not consume the resident decoder.  A
        # foreground call must therefore be able to execute it without even
        # constructing an MLX client: client construction can wait on model
        # admission during cold boot and used to turn a sub-second exact
        # program into a qualified-ingress timeout.
        if client is None:
            status = semantic_neural_default_serving_status()
        else:
            model_path = str(getattr(client, "model_path", "") or "")
            if not model_path:
                raise RuntimeError("qualified_recurrent_model_identity_unavailable")
            status = semantic_neural_serving_status(model_path)
        if not isinstance(status, Mapping) or status.get("active") is not True:
            return {
                "eligible": True,
                "attempted": False,
                "ok": False,
                "reason": str(
                    status.get("reason")
                    if isinstance(status, Mapping)
                    else "semantic_neural_serving_status_invalid"
                ),
                "admission": admission.receipt(),
            }
        activation_receipt = status.get("receipt")
        allowed_families = (
            activation_receipt.get("allowed_families")
            if isinstance(activation_receipt, Mapping)
            else None
        )
        if (
            not isinstance(allowed_families, Sequence)
            or isinstance(allowed_families, (str, bytes, bytearray))
            or admission.family not in allowed_families
        ):
            return {
                "eligible": True,
                "attempted": False,
                "ok": False,
                "reason": "semantic_neural_family_not_activated",
                "admission": admission.receipt(),
            }
        surface_decode_receipt: dict[str, Any] | None = None
        if admission.parser_id.startswith(_SCIENTIFIC_SURFACE_PARSER_PREFIX):
            surface_profile = admission.parser_id.removeprefix(
                _SCIENTIFIC_SURFACE_PARSER_PREFIX
            ).removesuffix(".v1")
            allowed_profiles = (
                activation_receipt.get("allowed_surface_profiles")
                if isinstance(activation_receipt, Mapping)
                else None
            )
            if (
                not isinstance(allowed_profiles, Sequence)
                or isinstance(allowed_profiles, (str, bytes, bytearray))
                or surface_profile not in allowed_profiles
            ):
                return {
                    "eligible": True,
                    "attempted": False,
                    "ok": False,
                    "reason": "semantic_neural_surface_profile_not_activated",
                    "admission": admission.receipt(),
                }
            surface_decode = await _run_qualified_cpu_bound(
                _execute_scientific_surface_cached,
                prompt,
                timeout_s=max(1.0, min(30.0, timeout_s)),
            )
            surface_receipt = surface_decode.program.receipt()
            if (
                surface_decode.program.profile != surface_profile
                or surface_receipt.get("public_prompt_sha256") != admission.public_source_sha256
                or surface_receipt.get("public_fact_graph_sha256") != admission.syntax_sha256
            ):
                raise RuntimeError("semantic_neural_surface_admission_drift")
            state = surface_decode.state
            surface_decode_receipt = surface_decode.receipt()
        else:
            state = await _run_qualified_cpu_bound(
                _execute_semantic_neural_state_cached,
                prompt,
                admission.family,
                timeout_s=max(1.0, min(30.0, timeout_s)),
            )
        text = render_semantic_neural_answer(state)
        if not isinstance(activation_receipt, Mapping):
            raise RuntimeError("semantic_neural_activation_receipt_unavailable")
        body = {
            "schema": QUALIFIED_RECURRENT_RESULT_SCHEMA,
            "admission": admission.receipt(),
            "semantic_state_receipt": state.receipt(),
            "surface_decode_receipt": surface_decode_receipt,
            "activation_receipt": dict(activation_receipt),
            "serialization": "canonical_json_from_authenticated_semantic_state",
            "answer_sha256": hashlib.sha256(text.encode("ascii")).hexdigest(),
        }
        return {
            "eligible": True,
            "attempted": True,
            "ok": True,
            "text": text,
            "reason": "qualified_semantic_neural_completed",
            "receipt": {**body, "receipt_sha256": _canonical_sha256(body)},
        }
    status_reader = getattr(client, "unified_recurrent_qualified_serving_status", None)
    if not callable(status_reader):
        return {
            "eligible": True,
            "attempted": False,
            "ok": False,
            "reason": "qualified_recurrent_serving_status_unavailable",
            "admission": admission.receipt(),
        }
    status = status_reader()
    if not isinstance(status, Mapping) or status.get("active") is not True:
        return {
            "eligible": True,
            "attempted": False,
            "ok": False,
            "reason": str(
                status.get("reason")
                if isinstance(status, Mapping)
                else "qualified_recurrent_serving_status_invalid"
            ),
            "admission": admission.receipt(),
        }
    model_path = str(getattr(client, "model_path", "") or "")
    if not model_path:
        raise RuntimeError("qualified_recurrent_model_identity_unavailable")
    # MLXLocalClient canonicalizes its model path before construction. Keeping
    # this call free of synchronous filesystem traversal also prevents a
    # qualified foreground request from blocking Aura's event loop.
    tokenizer = _load_tokenizer(model_path)
    public_tokens = project_qualified_public_tokens(tokenizer, prompt)
    decode = getattr(client, "unified_recurrent_qualified_decode_async", None)
    if not callable(decode):
        raise RuntimeError("qualified_recurrent_decode_contract_unavailable")
    result = await decode(
        public_tokens,
        family=admission.family,
        task_depth=admission.task_depth,
        max_tokens=32,
        timeout_s=timeout_s,
    )
    if not isinstance(result, Mapping) or result.get("ok") is not True:
        return {
            "eligible": True,
            "attempted": True,
            "ok": False,
            "reason": str(
                result.get("reason")
                if isinstance(result, Mapping)
                else "qualified_recurrent_decode_result_invalid"
            ),
            "admission": admission.receipt(),
        }
    worker = result.get("receipt")
    if not isinstance(worker, Mapping):
        raise RuntimeError("qualified_recurrent_worker_receipt_unavailable")
    text = render_qualified_recurrent_answer(admission.family, worker.get("parsed_values"))
    public_worker_receipt = {
        key: value
        for key, value in worker.items()
        if key not in {"generated_token_ids", "parsed_values"}
    }
    body = {
        "schema": QUALIFIED_RECURRENT_RESULT_SCHEMA,
        "admission": admission.receipt(),
        "public_tokens_sha256": _canonical_sha256(list(public_tokens)),
        "public_token_count": len(public_tokens),
        "worker_receipt": public_worker_receipt,
        "answer_sha256": hashlib.sha256(text.encode("ascii")).hexdigest(),
    }
    return {
        "eligible": True,
        "attempted": True,
        "ok": True,
        "text": text,
        "reason": "qualified_recurrent_completed",
        "receipt": {**body, "receipt_sha256": _canonical_sha256(body)},
    }


__all__ = [
    "QUALIFIED_ANSWER_BRIDGE",
    "QUALIFIED_RECURRENT_INGRESS_SCHEMA",
    "QUALIFIED_RECURRENT_RESULT_SCHEMA",
    "QualifiedRecurrentAdmission",
    "admit_qualified_recurrent_objective",
    "execute_qualified_recurrent_objective",
    "project_qualified_public_tokens",
    "qualified_recurrent_result_receipt_errors",
    "render_qualified_recurrent_answer",
]
