"""Bounded representation repair for malformed private verifier candidates.

Contract repair does not claim that an answer is correct. It only gives a
candidate that failed the strict ``FINAL_ANSWER`` wire contract one fresh,
source-private opportunity to express the same proposed solution in a
machine-gradeable form. Correctness and public-answer authority remain owned
by the downstream deterministic verifier and confidence-bound promotion gate.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from typing import Any

from core.brain.llm.latent_cortex.answer_contract import (
    contract_answer_state,
)

CONTRACT_REPAIR_SCHEMA = "aura.rlc.contract_repair.v2"
MAX_CONTRACT_REPAIR_REQUESTS = 8
MAX_CONTRACT_REPAIR_TOKENS = 512
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_FAILURE_REASONS = {
    "budget_unavailable",
    "generation_failed",
    "generation_contract_invalid",
}


def _sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _text_sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _request_limit(value: Any) -> int:
    if type(value) is not int or not 0 <= value <= MAX_CONTRACT_REPAIR_REQUESTS:
        raise ValueError("contract repair request limit is invalid")
    return value


def _token_limit(value: Any) -> int:
    if type(value) is not int or not 32 <= value <= MAX_CONTRACT_REPAIR_TOKENS:
        raise ValueError("contract repair token limit is invalid")
    return value


def prepare_contract_repair_requests(
    *,
    branch_candidates: Mapping[int, str],
    objective: str,
    response_contract: str = "",
    max_requests: int,
) -> list[dict[str, Any]]:
    """Build bounded fresh-context requests for contract-invalid candidates."""

    limit = _request_limit(max_requests)
    if (
        not isinstance(branch_candidates, Mapping)
        or not isinstance(objective, str)
        or not isinstance(response_contract, str)
    ):
        raise TypeError("contract repair private sources are invalid")
    if response_contract:
        from core.brain.llm.latent_cortex.response_contracts import (
            parse_response_contract,
        )

        parse_response_contract(response_contract)
    if limit == 0:
        return []
    prepared: list[dict[str, Any]] = []
    for raw_branch, candidate in sorted(branch_candidates.items()):
        if type(raw_branch) is not int or raw_branch < 0 or not isinstance(candidate, str):
            raise ValueError("contract repair candidate inventory is invalid")
        state = contract_answer_state(candidate)
        response_state: dict[str, Any] | None = None
        if response_contract and state["valid"]:
            from core.brain.llm.latent_cortex.task_verifiers import (
                check_response_contract,
            )

            response_state = check_response_contract(candidate, response_contract)
        if state["valid"] and (
            not response_contract or bool(response_state and response_state["valid"])
        ):
            continue
        original_reason = str(state["reason"])
        if state["valid"] and response_state is not None:
            failures = response_state.get("failures") or []
            original_reason = (
                f"response_contract:{str(failures[0])[:120]}"
                if failures
                else "response_contract_invalid"
            )
        payload = {
            "branch": raw_branch,
            "objective_sha256": _text_sha(objective),
            "response_contract_sha256": _text_sha(response_contract),
            "original_candidate_sha256": _text_sha(candidate),
            "original_contract_reason": original_reason[:160],
        }
        request_id = _sha(payload)
        schema_guidance = (
            "Required public response-object schema (shape and types only; it "
            f"contains no answer values):\n{response_contract}\n"
            if response_contract
            else ""
        )
        prompt = (
            "Re-encode the candidate answer below into the strict terminal answer "
            "contract. Preserve its proposed reasoning and conclusion; do not add "
            "new claims merely to improve it. Return exactly one terminal line of "
            "the form FINAL_ANSWER: followed by one strict JSON object. Do not put "
            "the object in a markdown fence and write nothing after it. This step "
            "repairs representation only; separate verifiers will judge correctness.\n"
            f"Objective:\n{objective}\n"
            f"{schema_guidance}"
            f"Candidate:\n{candidate}\n"
            "Begin the complete contract response now."
        )
        prepared.append(
            {
                **payload,
                "request_id": request_id,
                "prompt": prompt,
                "prompt_sha256": _text_sha(prompt),
            }
        )
        if len(prepared) >= limit:
            break
    return prepared


def parse_contract_repair_generation(
    value: Any,
    *,
    response_contract: str = "",
) -> str:
    """Accept exactly one complete strict answer-contract object."""

    return _contract_repair_generation_state(
        value,
        response_contract=response_contract,
    )["candidate"]


def _unambiguous_schema_coercion(
    candidate: str,
    *,
    response_contract: str,
) -> str | None:
    """Wrap one generated value only when the public shape has one mapping.

    This is serialization repair, not answer synthesis: every model-generated
    value is preserved, there is exactly one compatible container field, and
    any additional fields have one fixed public literal value. Ambiguous field
    assignments, missing free values, or incompatible types fail closed.
    """

    if not response_contract:
        return None
    from core.brain.llm.latent_cortex.response_contracts import (
        ContractType,
        ListType,
        LiteralType,
        ObjectType,
        TupleType,
        parse_response_contract,
        validate_response_payload,
    )

    source = candidate.strip()
    if source.startswith("FINAL_ANSWER:"):
        source = source.split("FINAL_ANSWER:", 1)[1].lstrip()
    try:
        payload, end = json.JSONDecoder().raw_decode(source)
    # not a failure: text that does not parse is not the shape this was reading for.
    except json.JSONDecodeError:
        return None
    tail = source[end:].strip()
    if tail:
        # Explanatory prose is disposable protocol noise. A second JSON value
        # is a competing answer and therefore ambiguous, even when both values
        # happen to be equal after canonicalization.
        decoder = json.JSONDecoder()
        for index, char in enumerate(tail):
            if char not in "[{":
                continue
            try:
                competing, _competing_end = decoder.raw_decode(tail[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(competing, (dict, list)):
                return None
    contract = parse_response_contract(response_contract)
    if not isinstance(contract, ObjectType):
        return None

    missing = object()

    def coerce(value: Any, expected: ContractType) -> Any:
        if validate_response_payload(value, expected)["valid"]:
            return value
        if isinstance(expected, ObjectType):
            if not isinstance(value, dict):
                return missing
            expected_names = {name for name, _field in expected.fields}
            if set(value) - expected_names:
                return missing
            result: dict[str, Any] = {}
            for name, field_contract in expected.fields:
                if name not in value:
                    if (
                        isinstance(field_contract, LiteralType)
                        and len(field_contract.values) == 1
                    ):
                        result[name] = field_contract.values[0]
                        continue
                    return missing
                field_value = coerce(value[name], field_contract)
                if field_value is missing:
                    return missing
                result[name] = field_value
            return result
        if isinstance(expected, TupleType):
            if not isinstance(value, list) or len(value) != len(expected.items):
                return missing
            result = []
            for item, item_contract in zip(value, expected.items, strict=True):
                coerced_item = coerce(item, item_contract)
                if coerced_item is missing:
                    return missing
                result.append(coerced_item)
            return result
        if isinstance(expected, ListType) and isinstance(value, list):
            result = []
            itemwise_valid = True
            for item in value:
                coerced_item = coerce(item, expected.item)
                if coerced_item is missing:
                    itemwise_valid = False
                    break
                result.append(coerced_item)
            if itemwise_valid and validate_response_payload(result, expected)["valid"]:
                return result
            single_item = coerce(value, expected.item)
            if single_item is not missing:
                wrapped = [single_item]
                if validate_response_payload(wrapped, expected)["valid"]:
                    return wrapped
        return missing

    generated_fields = [
        (name, field_contract)
        for name, field_contract in contract.fields
        if isinstance(field_contract, (ListType, TupleType))
    ]
    if isinstance(payload, list) and len(generated_fields) == 1:
        generated_name, _generated_contract = generated_fields[0]
        coerced: dict[str, Any] = {generated_name: payload}
    elif isinstance(payload, dict):
        coerced = dict(payload)
    else:
        return None

    coerced = coerce(coerced, contract)
    if coerced is missing or not validate_response_payload(coerced, contract)["valid"]:
        return None
    encoded = json.dumps(
        coerced,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return f"FINAL_ANSWER: {encoded}"


def _contract_repair_generation_state(
    value: Any,
    *,
    response_contract: str = "",
) -> dict[str, Any]:
    """Return the admitted candidate and whether schema coercion was used."""

    if not isinstance(value, str):
        raise TypeError("contract repair generation must be text")
    candidate = value.strip()
    state = contract_answer_state(candidate)
    # Some checkpoints emit only the requested JSON object. Restoring the
    # fixed protocol marker changes no model-proposed content and is accepted
    # only when the exact combined transcript passes the same strict parser.
    if not state["valid"] and state["reason"] == "no_marker" and candidate.startswith("{"):
        framed = f"FINAL_ANSWER: {candidate}"
        if contract_answer_state(framed)["valid"]:
            candidate = framed
            state = contract_answer_state(candidate)
    marker_payload = (
        candidate.split("FINAL_ANSWER:", 1)[1].lstrip()
        if "FINAL_ANSWER:" in candidate
        else ""
    )
    array_shaped = candidate.lstrip().startswith("[") or marker_payload.startswith("[")
    if not state["valid"] and not array_shaped:
        # A weak checkpoint may produce the right JSON object and then corrupt
        # only the surrounding protocol text. Recovering one UNIQUE object is
        # a semantic-preserving serialization operation, not answer synthesis.
        # Distinct objects are ambiguous and arrays are not the required wire
        # type, so both remain rejected.
        decoder = json.JSONDecoder()
        unique: dict[str, dict[str, Any]] = {}
        for index, char in enumerate(candidate):
            if char != "{":
                continue
            try:
                parsed, _end = decoder.raw_decode(candidate[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                canonical = json.dumps(
                    parsed,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                    allow_nan=False,
                )
                unique[canonical] = parsed
        if len(unique) == 1:
            framed = f"FINAL_ANSWER: {next(iter(unique))}"
            if contract_answer_state(framed)["valid"]:
                candidate = framed
                state = contract_answer_state(candidate)
    schema_coercion_applied = False
    if not state["valid"]:
        coerced = _unambiguous_schema_coercion(
            candidate,
            response_contract=response_contract,
        )
        if coerced is not None:
            candidate = coerced
            state = contract_answer_state(candidate)
            schema_coercion_applied = True
    if not state["valid"]:
        raise ValueError(f"contract repair generation is invalid: {state['reason']}")
    if response_contract:
        from core.brain.llm.latent_cortex.task_verifiers import (
            check_response_contract,
        )

        response_state = check_response_contract(candidate, response_contract)
        if not response_state["valid"]:
            failures = response_state.get("failures") or ["response_contract_invalid"]
            raise ValueError(
                "contract repair generation violates response contract: "
                f"{str(failures[0])[:160]}"
            )
    return {
        "candidate": candidate,
        "schema_coercion_applied": schema_coercion_applied,
    }


def _generation_context(value: Any, *, prompt_sha256: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("contract repair generation context is missing")
    required = {
        "prompt_sha256",
        "generated_token_count",
        "termination",
        "initial_cache_offsets",
        "final_cache_offsets",
        "all_initial_offsets_zero",
        "solver_context_imported",
        "parameter_relation",
    }
    if set(value) != required:
        raise ValueError("contract repair generation context fields differ")
    initial = value["initial_cache_offsets"]
    final = value["final_cache_offsets"]
    if (
        value["prompt_sha256"] != prompt_sha256
        or _SHA256_RE.fullmatch(str(prompt_sha256)) is None
        or type(value["generated_token_count"]) is not int
        or not 1 <= value["generated_token_count"] <= MAX_CONTRACT_REPAIR_TOKENS + 64
        # The raw contract decoder may stop at the first irrecoverable framing
        # token even when a complete unique JSON object already precedes it.
        # Admission still requires parse_contract_repair_generation to rebuild
        # and strictly validate that object before this context is considered.
        or value["termination"]
        not in {
            "contract_complete",
            "contract_irrecoverable",
            "eos",
            "token_limit",
            "token_limit_contract_incomplete",
            "token_limit_sentence_grace",
        }
        or not isinstance(initial, list)
        or not initial
        or any(type(offset) is not int or offset != 0 for offset in initial)
        or not isinstance(final, list)
        or len(final) != len(initial)
        or any(type(offset) is not int or offset <= 0 for offset in final)
        or value["all_initial_offsets_zero"] is not True
        or value["solver_context_imported"] is not False
        or value["parameter_relation"] != "shared_resident_checkpoint"
    ):
        raise ValueError("contract repair generation context is invalid")
    return dict(value)


def build_contract_repair_receipt(
    *,
    branch_candidates: Mapping[int, str],
    objective: str,
    response_contract: str = "",
    generated_repairs: Mapping[str, Mapping[str, Any]] | None = None,
    execution_failures: Mapping[str, str] | None = None,
    max_requests: int,
    max_tokens: int,
) -> dict[str, Any]:
    """Build a text-free, hash-bound receipt for contract repair attempts."""

    request_limit = _request_limit(max_requests)
    token_limit = _token_limit(max_tokens)
    prepared = prepare_contract_repair_requests(
        branch_candidates=branch_candidates,
        objective=objective,
        response_contract=response_contract,
        max_requests=request_limit,
    )
    generated = dict(generated_repairs or {})
    failures = dict(execution_failures or {})
    request_ids = {row["request_id"] for row in prepared}
    if set(generated) - request_ids or set(failures) - request_ids:
        raise ValueError("contract repair result names an unknown request")
    if set(generated) & set(failures):
        raise ValueError("contract repair request has conflicting outcomes")
    if any(reason not in _FAILURE_REASONS for reason in failures.values()):
        raise ValueError("contract repair failure reason is invalid")

    transactions: list[dict[str, Any]] = []
    admitted_candidates: dict[int, str] = {}
    for request in prepared:
        request_id = request["request_id"]
        generated_row = generated.get(request_id)
        if generated_row is not None:
            if not isinstance(generated_row, Mapping) or set(generated_row) != {
                "candidate",
                "generation_context",
            }:
                raise ValueError("contract repair generated result is invalid")
            parsed_generation = _contract_repair_generation_state(
                generated_row["candidate"],
                response_contract=response_contract,
            )
            candidate = parsed_generation["candidate"]
            context = _generation_context(
                generated_row["generation_context"],
                prompt_sha256=request["prompt_sha256"],
            )
            admitted_candidates[int(request["branch"])] = candidate
            transaction = {
                "request_id": request_id,
                "branch": request["branch"],
                "status": "contract_candidate_admitted",
                "reason": "strict_contract_reencoded_in_fresh_context",
                "original_candidate_sha256": request["original_candidate_sha256"],
                "repaired_candidate_sha256": _text_sha(candidate),
                "generation_context": context,
                "candidate_effect": "branch_probe_replaced",
                "answer_selection_effect": "none",
                "schema_coercion_applied": parsed_generation[
                    "schema_coercion_applied"
                ],
            }
        else:
            transaction = {
                "request_id": request_id,
                "branch": request["branch"],
                "status": "contract_repair_not_admitted",
                "reason": failures.get(request_id, "generation_failed"),
                "original_candidate_sha256": request["original_candidate_sha256"],
                "repaired_candidate_sha256": "",
                "generation_context": {},
                "candidate_effect": "none",
                "answer_selection_effect": "none",
                "schema_coercion_applied": False,
            }
        transactions.append(transaction)

    public_requests = [
        {key: value for key, value in row.items() if key != "prompt"}
        for row in prepared
    ]
    payload = {
        "schema": CONTRACT_REPAIR_SCHEMA,
        "objective_sha256": _text_sha(objective),
        "response_contract_sha256": _text_sha(response_contract),
        "response_contract_required": bool(response_contract),
        "max_requests": request_limit,
        "max_tokens": token_limit,
        "requests": public_requests,
        "transactions": transactions,
        "request_count": len(public_requests),
        "attempted_count": len(generated) + len(failures),
        "admitted_count": len(admitted_candidates),
        "candidate_effect": (
            "contract_valid_candidate_pool_addition" if admitted_candidates else "none"
        ),
        "answer_selection_effect": "none",
        "authority": "representation_repair_only",
    }
    receipt = {**payload, "receipt_sha256": _sha(payload)}
    validate_contract_repair_receipt(receipt)
    return receipt


def validate_contract_repair_receipt(value: Any) -> dict[str, Any]:
    """Validate the public transaction envelope without private candidate text."""

    if not isinstance(value, Mapping):
        raise ValueError("contract repair receipt is missing")
    fields = {
        "schema",
        "objective_sha256",
        "response_contract_sha256",
        "response_contract_required",
        "max_requests",
        "max_tokens",
        "requests",
        "transactions",
        "request_count",
        "attempted_count",
        "admitted_count",
        "candidate_effect",
        "answer_selection_effect",
        "authority",
        "receipt_sha256",
    }
    if set(value) != fields:
        raise ValueError("contract repair receipt fields differ")
    payload = {key: value[key] for key in fields - {"receipt_sha256"}}
    if value["receipt_sha256"] != _sha(payload):
        raise ValueError("contract repair receipt commitment mismatch")
    requests = value["requests"]
    transactions = value["transactions"]
    limit = _request_limit(value["max_requests"])
    _token_limit(value["max_tokens"])
    if (
        value["schema"] != CONTRACT_REPAIR_SCHEMA
        or _SHA256_RE.fullmatch(str(value["objective_sha256"])) is None
        or _SHA256_RE.fullmatch(str(value["response_contract_sha256"])) is None
        or type(value["response_contract_required"]) is not bool
        or not isinstance(requests, list)
        or not isinstance(transactions, list)
        or len(requests) != len(transactions)
        or len(requests) > limit
        or value["request_count"] != len(requests)
        or type(value["attempted_count"]) is not int
        or not 0 <= value["attempted_count"] <= len(requests)
        or type(value["admitted_count"]) is not int
        or not 0 <= value["admitted_count"] <= value["attempted_count"]
        or value["answer_selection_effect"] != "none"
        or value["authority"] != "representation_repair_only"
    ):
        raise ValueError("contract repair receipt is invalid")
    admitted = 0
    for request, transaction in zip(requests, transactions, strict=True):
        if not isinstance(request, Mapping) or not isinstance(transaction, Mapping):
            raise ValueError("contract repair row is invalid")
        if (
            request.get("request_id") != transaction.get("request_id")
            or request.get("branch") != transaction.get("branch")
            or request.get("original_candidate_sha256")
            != transaction.get("original_candidate_sha256")
            or _SHA256_RE.fullmatch(str(request.get("prompt_sha256"))) is None
        ):
            raise ValueError("contract repair row binding differs")
        if transaction.get("status") == "contract_candidate_admitted":
            admitted += 1
            if (
                transaction.get("reason")
                != "strict_contract_reencoded_in_fresh_context"
                or _SHA256_RE.fullmatch(
                    str(transaction.get("repaired_candidate_sha256"))
                )
                is None
                or transaction.get("candidate_effect") != "branch_probe_replaced"
                or type(transaction.get("schema_coercion_applied")) is not bool
            ):
                raise ValueError("contract repair admission is invalid")
            _generation_context(
                transaction.get("generation_context"),
                prompt_sha256=str(request["prompt_sha256"]),
            )
        elif transaction.get("status") == "contract_repair_not_admitted":
            if (
                transaction.get("reason") not in _FAILURE_REASONS
                or transaction.get("repaired_candidate_sha256") != ""
                or transaction.get("generation_context") != {}
                or transaction.get("candidate_effect") != "none"
                or transaction.get("schema_coercion_applied") is not False
            ):
                raise ValueError("contract repair rejection is invalid")
        else:
            raise ValueError("contract repair status is invalid")
        if transaction.get("answer_selection_effect") != "none":
            raise ValueError("contract repair gained answer authority")
    if (
        admitted != value["admitted_count"]
        or value["candidate_effect"]
        != ("contract_valid_candidate_pool_addition" if admitted else "none")
    ):
        raise ValueError("contract repair summary differs")
    return dict(value)


__all__ = [
    "CONTRACT_REPAIR_SCHEMA",
    "MAX_CONTRACT_REPAIR_REQUESTS",
    "MAX_CONTRACT_REPAIR_TOKENS",
    "build_contract_repair_receipt",
    "parse_contract_repair_generation",
    "prepare_contract_repair_requests",
    "validate_contract_repair_receipt",
]
