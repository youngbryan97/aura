"""Model-authored, sandbox-grounded computation for difficult reasoning turns.

The symbolic sandbox used to be downstream of answer generation: Aura could
repair code that she happened to write, but she did not deliberately write a
program to solve a structured problem.  This module supplies that missing
operation.  It converts a public objective into a pure-compute program, runs it
inside the existing kernel sandbox, and returns the program's stdout as a
candidate answer.  The caller remains responsible for normal answer
verification and promotion.

No task answers or benchmark-family solvers live here.  The resident model
authors each program from the same public information available to ordinary
inference.  Generated source and raw diagnostics remain ephemeral; receipts
carry hashes, sizes, containment evidence, and bounded status only.
"""
from __future__ import annotations

import ast
import asyncio
import hashlib
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from core.brain.generation_provenance import attributed_text, generation_metadata_of

GenerateFn = Callable[[str, float], Awaitable[Any]]

EXECUTABLE_REASONING_SCHEMA = "aura.executable_reasoning.v1"
_MAX_OBJECTIVE_CHARS = 12_000
_MAX_STDOUT_CHARS = 32_000
_MIN_GENERATION_WINDOW_S = 2.0

EXECUTABLE_STRATEGIES = (
    "reference_enumeration_or_simulation",
    "independent_constraint_formulation",
    "decomposition_with_independent_crosscheck",
)
_CAUSAL_STRATEGIES = (
    "causal_total_effect_reconstruction",
    "independent_structural_equation_model",
    "causal_counterfactual_crosscheck",
)
_PLANNING_STRATEGIES = (
    "exhaustive_feasible_schedule_search",
    "dynamic_programming_over_completed_tasks",
    "branch_and_bound_with_lexicographic_tiebreak",
)
_PROBABILITY_STRATEGIES = (
    "exact_fraction_probability_update",
    "independent_odds_form_update",
    "probability_normalization_crosscheck",
)
_AUDIT_STRATEGIES = (
    "independent_recomputation_and_rank",
    "counterexample_search_against_the_claim",
    "dual_formula_audit_with_tiebreak_check",
)
_ORDERED_SELECTION_STRATEGIES = (
    "literal_order_statistic_interpreter",
    "stable_indexed_selection_simulation",
    "independent_rank_and_tiebreak_crosscheck",
)
_STRATEGY_GUIDANCE = {
    "reference_enumeration_or_simulation": (
        "Implement the simplest exhaustive enumeration or literal state simulation. "
        "Avoid clever shortcuts and avoid assertions."
    ),
    "independent_constraint_formulation": (
        "Translate every stated requirement into a separate executable predicate, "
        "enumerate bounded candidates, and derive the result from the admitted set."
    ),
    "decomposition_with_independent_crosscheck": (
        "Implement two structurally different pure computations, compare their derived "
        "payloads, and print only when they agree. Do not compare against a literal answer."
    ),
    "causal_total_effect_reconstruction": (
        "Treat each reported intervention outcome as the total measured effect of that "
        "intervention. Infer direction from asymmetric interventions. When predicting "
        "the same intervention-to-outcome relation, scale that observed total effect "
        "exactly once; do not add mediated components already included in it."
    ),
    "independent_structural_equation_model": (
        "Build explicit linear structural equations from interventions. Distinguish "
        "total intervention effects from direct edge coefficients, and do not infer a "
        "direct coefficient when the observations identify only a total effect."
    ),
    "causal_counterfactual_crosscheck": (
        "Infer the causal order, compute the requested intervention two ways when "
        "identifiable, and reject any derivation that propagates an observed total "
        "effect through the graph a second time."
    ),
    "exhaustive_feasible_schedule_search": (
        "Enumerate task subsets and permutations within the stated bound. Simulate one "
        "resource timeline, enforce prerequisites and completion deadlines, then apply "
        "the objectives and tie-breaks in their declared order."
    ),
    "dynamic_programming_over_completed_tasks": (
        "Use a bounded state keyed by completed tasks and elapsed time. Admit a transition "
        "only when prerequisites and deadlines hold, retaining the best declared objective "
        "tuple for each state."
    ),
    "branch_and_bound_with_lexicographic_tiebreak": (
        "Search feasible plans with an optimistic remaining-reward bound. Compare complete "
        "plans by the exact objective tuple, including makespan and lexicographic tie-break."
    ),
    "exact_fraction_probability_update": (
        "Use fractions for every probability. Enumerate the mutually exclusive hypotheses, "
        "compute evidence mass, normalize once, reduce exactly, then derive any band from "
        "the exact posterior."
    ),
    "independent_odds_form_update": (
        "Compute prior odds times the likelihood ratio using exact fractions, convert back "
        "to probability, and independently confirm normalization."
    ),
    "probability_normalization_crosscheck": (
        "Compute the posterior both by Bayes normalization and by odds. Print only if the "
        "two exact fractions agree and the declared category follows from that value."
    ),
    "independent_recomputation_and_rank": (
        "Ignore the stated conclusion, recompute every candidate score from the supplied "
        "formula, and rank by the complete declared tie-break tuple."
    ),
    "counterexample_search_against_the_claim": (
        "Try to refute the stated claim by exhaustively finding any candidate with a better "
        "objective tuple; report validity only after that search is complete."
    ),
    "dual_formula_audit_with_tiebreak_check": (
        "Compute all scores in two independent functions, require agreement, then evaluate "
        "the stated winner and every tie-break explicitly."
    ),
    "literal_order_statistic_interpreter": (
        "Translate each ordering phrase literally. For a lower median of n sorted values, "
        "use zero-based rank (n-1)//2; do not silently substitute the upper median. Preserve "
        "original indices and apply every subsequent selection and tie-break in stated order."
    ),
    "stable_indexed_selection_simulation": (
        "Represent every item as (value, original_index), simulate the requested selection "
        "one step at a time, and keep original indices stable. Encode lower/upper, nearest, "
        "and tie-break rules as separate explicit key functions."
    ),
    "independent_rank_and_tiebreak_crosscheck": (
        "Compute the requested ordered selection twice: once by direct sorted ranks and once "
        "by enumerating candidates with an explicit objective tuple. Print only if both paths "
        "agree, including original-index and checksum rules."
    ),
}

_NO_EXECUTION = re.compile(
    r"\b(?:without\s+(?:executing|running)|do\s+not\s+(?:execute|run)|"
    r"must\s+not\s+(?:execute|run))\b",
    re.IGNORECASE,
)
_STRUCTURED_COMPUTE = re.compile(
    r"\b(?:calculate|compute|count|evaluate|trace|simulate|predict|checksum|"
    r"maximize|minimize|optimal|schedul|sequence|posterior|probability|"
    r"combinator|intervention|constraint|score|winner|median|algorithm|"
    r"shortest[ -]path|graph|vertices?|edges?)\w*\b",
    re.IGNORECASE,
)
_STRUCTURED_INPUT = re.compile(r"(?:\[[^\]]+\]|\{[^}]+\}|\b\d+(?:\.\d+)?\b)")
_STRUCTURED_OUTPUT_REQUEST = re.compile(
    r"\b(?:worked|step[ -]by[ -]step|walk\s+through|demonstrat\w*|construct\w*|"
    r"show|give|provide|build)\b.{0,100}\b(?:example|trace|simulation|table|"
    r"schedule|sequence|execution|run)\b",
    re.IGNORECASE | re.DOTALL,
)


@dataclass(frozen=True, slots=True)
class ExecutableReasoningResult:
    candidate: str
    succeeded: bool
    receipt: dict[str, Any]


def should_use_executable_reasoning(
    objective: str,
    *,
    task_type: str,
    explicitly_enabled: bool = False,
) -> bool:
    """Return whether pure computation is a plausible reasoning operation.

    Explicit user constraints against execution always win.  The default path
    is intentionally semantic rather than tied to benchmark domain names:
    math is computational by definition; other domains need a structured
    operation plus either supplied structured input or a request to construct a
    structured output such as a worked trace.  A caller may explicitly enable
    the organ for an already-classified hard task, but cannot override a
    no-execution instruction.
    """

    text = str(objective or "").strip()
    if not text or _NO_EXECUTION.search(text):
        return False
    if explicitly_enabled:
        return True
    normalized_type = str(task_type or "").strip().lower()
    if normalized_type == "math":
        return True
    return bool(
        normalized_type in {"code", "logic", "planning", "factual"}
        and _STRUCTURED_COMPUTE.search(text)
        and (
            _STRUCTURED_INPUT.search(text)
            or _STRUCTURED_OUTPUT_REQUEST.search(text)
        )
    )


def select_executable_strategies(
    objective: str,
    *,
    task_type: str,
) -> tuple[str, ...]:
    """Select reusable computational laws from task semantics, not task IDs."""

    text = str(objective or "").lower()
    if any(token in text for token in ("intervention", "causal", "baseline values")):
        return _CAUSAL_STRATEGIES
    if any(token in text for token in ("deadline", "makespan", "prerequisite", "horizon")):
        return _PLANNING_STRATEGIES
    if any(token in text for token in ("posterior", "likelihood", "bayes", "prior probability")):
        return _PROBABILITY_STRATEGIES
    if any(token in text for token in ("premise", "claim", "actual winner", "highest score")):
        return _AUDIT_STRATEGIES
    if any(
        token in text
        for token in (
            "lower median",
            "upper median",
            "original index",
            "nearest remaining",
            "order statistic",
        )
    ):
        return _ORDERED_SELECTION_STRATEGIES
    del task_type
    return EXECUTABLE_STRATEGIES


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()


def _program_prompt(
    objective: str,
    response_contract: str,
    *,
    strategy: str,
    excluded_program_sha256s: tuple[str, ...],
    excluded_candidate_sha256s: tuple[str, ...],
    prior_failure_class: str,
) -> str:
    contract = str(response_contract or "").strip()
    output_rule = (
        "Print exactly one line beginning FINAL_ANSWER: followed by one JSON "
        "object. Build a payload value, serialize it with json.dumps using strict JSON, "
        f"and satisfy this public response contract: {contract}"
        if contract
        else "Print only the final answer that should be returned to the user."
    )
    prior = ""
    if excluded_program_sha256s or excluded_candidate_sha256s:
        prior = (
            "A public verifier rejected earlier computations. Their source and output "
            "are withheld to prevent anchoring. Do not reproduce them; use the assigned "
            "independent strategy.\n"
            f"PRIOR_FAILURE_CLASS: {prior_failure_class or 'public_verifier_refuted'}\n"
            f"EXCLUDED_PROGRAM_SHA256: {','.join(excluded_program_sha256s[-6:])}\n"
            f"EXCLUDED_CANDIDATE_SHA256: {','.join(excluded_candidate_sha256s[-6:])}\n"
        )
    return (
        "Solve the task by authoring a self-contained pure-Python scratch program.\n"
        "The program will run in an isolated, network-denied sandbox with no files, "
        "input, shell, subprocesses, reflection, or external packages. Standard pure "
        "computation with builtins and safe modules such as math, itertools, fractions, "
        "collections, statistics, decimal, and json is allowed.\n"
        "Derive the answer from the task; do not guess it or embed an unexplained final "
        "constant. Prefer exhaustive search or an independent invariant when practical. "
        "Never assert a literal expected final answer, expected full sequence, or checksum. "
        "Assertions may check only generic invariants computed from the candidate itself, "
        "such as length, uniqueness, constraints, or an independently recomputed score.\n"
        f"Assigned independent strategy: {strategy}. "
        f"{_STRATEGY_GUIDANCE.get(strategy, 'Use an independent pure computation.')}\n"
        f"{prior}"
        f"{output_rule}\n"
        "Return exactly one fenced Python code block and no prose.\n\n"
        "TASK:\n"
        f"{str(objective or '').strip()[:_MAX_OBJECTIVE_CHARS]}"
    )


def _restart_prompt(
    objective: str,
    diagnostic: str,
    response_contract: str,
    prior_program_sha256: str,
    *,
    strategy: str,
) -> str:
    """Request a disjoint retry without exposing the failed source or answer."""

    contract = str(response_contract or "").strip()
    return (
        "A previous pure-Python reasoning attempt failed its sandbox check. Start over "
        "from the original task using a different derivation or algorithm. The failed "
        "source is deliberately withheld so it cannot anchor this attempt. Return exactly "
        "one fenced Python code block. Do not use files, network, shell, subprocesses, "
        "input, reflection, or external packages. Never assert a literal expected final "
        "answer, expected full sequence, or checksum; assertions may check only generic "
        "invariants computed from the candidate itself. "
        + (
            "It must print exactly one terminal FINAL_ANSWER JSON object satisfying "
            f"{contract}. "
            if contract
            else "It must print only the final user-facing answer. "
        )
        + f"Continue using the assigned independent strategy: {strategy}. "
        + f"{_STRATEGY_GUIDANCE.get(strategy, 'Use an independent pure computation.')} "
        + f"\n\nPRIOR_PROGRAM_SHA256: {prior_program_sha256}"
        + f"\nFAILURE_CLASS: {diagnostic[:512]}"
        + f"\n\nORIGINAL_TASK:\n{str(objective or '').strip()[:_MAX_OBJECTIVE_CHARS]}"
    )


def _normalize_contract_candidate(
    candidate: str,
    response_contract: str,
) -> tuple[str, bool]:
    """Canonicalize generated values without inventing or selecting values."""

    if not response_contract:
        normalized = candidate.strip()
        return normalized, False
    try:
        from core.brain.llm.latent_cortex.frontier_tasks import parse_final_answer
        from core.brain.llm.latent_cortex.response_contracts import (
            parse_response_contract,
            validate_response_payload,
        )

        contract = parse_response_contract(response_contract)
        payload = parse_final_answer(candidate)
        if validate_response_payload(payload, contract)["valid"]:
            return candidate.strip(), False
    except (KeyError, TypeError, ValueError):
        pass
    try:
        from core.brain.llm.latent_cortex.contract_repair import (
            parse_contract_repair_generation,
        )

        normalized = parse_contract_repair_generation(
            candidate,
            response_contract=response_contract,
        )
        return normalized, normalized != candidate.strip()
    except (KeyError, TypeError, ValueError):
        return "", False


def _execution_failure_class(execution: Any) -> str:
    """Return bounded failure data without feeding runtime text back as instructions."""

    if getattr(execution, "refused", False):
        warnings = getattr(execution, "warnings", []) or []
        labels = [re.sub(r"[^a-zA-Z0-9_.:-]+", "_", str(item))[:80] for item in warnings]
        return "sandbox_refused:" + ",".join(labels[:4])
    if getattr(execution, "timed_out", False):
        return "sandbox_timed_out"
    raw = (
        str(getattr(execution, "traceback", "") or "")
        or str(getattr(execution, "stderr", "") or "")
    )
    if "AssertionError" in raw:
        return "sandbox_execution_failed:AssertionError"
    if "SyntaxError" in raw:
        return "sandbox_execution_failed:SyntaxError"
    if "TypeError" in raw:
        return "sandbox_execution_failed:TypeError"
    if "ValueError" in raw:
        return "sandbox_execution_failed:ValueError"
    if "IndexError" in raw:
        return "sandbox_execution_failed:IndexError"
    if "KeyError" in raw:
        return "sandbox_execution_failed:KeyError"
    return "sandbox_execution_failed:RuntimeError"


async def derive_executable_candidate(
    *,
    objective: str,
    task_type: str,
    generate: GenerateFn,
    sandbox: Any,
    deadline: float,
    response_contract: str = "",
    explicitly_enabled: bool = False,
    strategy: str = EXECUTABLE_STRATEGIES[0],
    excluded_program_sha256s: tuple[str, ...] = (),
    excluded_candidate_sha256s: tuple[str, ...] = (),
    prior_failure_class: str = "",
    max_generation_attempts: int = 2,
) -> ExecutableReasoningResult:
    """Generate, execute, and receipt one bounded program-of-thought attempt."""

    started = time.monotonic()
    if type(max_generation_attempts) is not int or not 1 <= max_generation_attempts <= 2:
        raise ValueError("max_generation_attempts must be 1 or 2")
    base_receipt: dict[str, Any] = {
        "schema": EXECUTABLE_REASONING_SCHEMA,
        "status": "not_applicable",
        "task_type": str(task_type or ""),
        "objective_sha256": _sha256(str(objective or "")),
        "response_contract_sha256": _sha256(str(response_contract or "")),
        "generation_calls": 0,
        "program_chars": 0,
        "program_bytes": 0,
        "program_sha256": "",
        "candidate_chars": 0,
        "candidate_sha256": "",
        "contract_valid": False,
        "strategy": str(strategy),
        "excluded_program_count": len(excluded_program_sha256s),
        "excluded_candidate_count": len(excluded_candidate_sha256s),
    }
    if not should_use_executable_reasoning(
        objective,
        task_type=task_type,
        explicitly_enabled=explicitly_enabled,
    ):
        return ExecutableReasoningResult("", False, base_receipt)

    remaining = deadline - time.monotonic()
    if remaining < _MIN_GENERATION_WINDOW_S:
        return ExecutableReasoningResult(
            "", False, {**base_receipt, "status": "deadline_exhausted"}
        )

    from core.brain.verifiers.code_engine import extract_code_blocks
    generation_calls = 0
    attempts: list[dict[str, Any]] = []
    generated_metadata: dict[str, Any] = {}
    execution: Any = None
    program = ""
    diagnostic = ""
    for attempt_index in range(max_generation_attempts):
        remaining = deadline - time.monotonic()
        if remaining < _MIN_GENERATION_WINDOW_S:
            break
        prompt = (
            _program_prompt(
                objective,
                response_contract,
                strategy=strategy,
                excluded_program_sha256s=excluded_program_sha256s,
                excluded_candidate_sha256s=excluded_candidate_sha256s,
                prior_failure_class=prior_failure_class,
            )
            if attempt_index == 0
            else _restart_prompt(
                objective,
                diagnostic or "prior_attempt_failed",
                response_contract,
                _sha256(program),
                strategy=strategy,
            )
        )
        try:
            generated = await asyncio.wait_for(
                generate(prompt, 0.2 if attempt_index == 0 else 0.6),
                timeout=max(_MIN_GENERATION_WINDOW_S, remaining),
            )
        except (TimeoutError, RuntimeError, AttributeError, TypeError, ValueError):
            generation_calls += 1
            attempts.append(
                {
                    "attempt": attempt_index + 1,
                    "status": "program_generation_failed",
                    "program_sha256": "",
                }
            )
            break
        generation_calls += 1
        generated_metadata = generation_metadata_of(generated)
        blocks = extract_code_blocks(str(generated or "").strip())
        if len(blocks) != 1:
            diagnostic = f"program_shape_invalid:block_count={len(blocks)}"
            attempts.append(
                {
                    "attempt": attempt_index + 1,
                    "status": "program_shape_invalid",
                    "program_sha256": "",
                    "program_block_count": len(blocks),
                }
            )
            program = ""
            continue
        program = blocks[0]
        program_sha256 = _sha256(program)
        if program_sha256 in excluded_program_sha256s:
            diagnostic = "public_verifier_duplicate_program"
            attempts.append(
                {
                    "attempt": attempt_index + 1,
                    "status": "duplicate_program_rejected",
                    "program_sha256": program_sha256,
                    "program_chars": len(program),
                    "program_bytes": len(program.encode("utf-8")),
                }
            )
            continue
        try:
            ast.parse(program)
        except SyntaxError as exc:
            diagnostic = (
                "program_syntax_invalid:"
                f"{exc.msg}:line={int(exc.lineno or 0)}:offset={int(exc.offset or 0)}"
            )
            attempts.append(
                {
                    "attempt": attempt_index + 1,
                    "status": "syntax_invalid",
                    "program_sha256": program_sha256,
                    "program_chars": len(program),
                    "program_bytes": len(program.encode("utf-8")),
                    "diagnostic": diagnostic,
                }
            )
            continue
        try:
            execution = await sandbox.run(program)
        except (TimeoutError, OSError, RuntimeError, AttributeError, TypeError, ValueError):
            diagnostic = "sandbox_infrastructure_failure"
            attempts.append(
                {
                    "attempt": attempt_index + 1,
                    "status": "sandbox_failed",
                    "program_sha256": _sha256(program),
                    "program_chars": len(program),
                    "program_bytes": len(program.encode("utf-8")),
                }
            )
            break
        execution_receipt = execution.to_dict()
        attempt_status = "executed" if getattr(execution, "ok", False) else (
            "refused" if getattr(execution, "refused", False)
            else "timed_out" if getattr(execution, "timed_out", False)
            else "execution_failed"
        )
        attempts.append(
            {
                "attempt": attempt_index + 1,
                "status": attempt_status,
                "program_sha256": _sha256(program),
                "program_chars": len(program),
                "program_bytes": len(program.encode("utf-8")),
                "sandbox": execution_receipt,
            }
        )
        if getattr(execution, "ok", False):
            break
        diagnostic = _execution_failure_class(execution)

    if execution is None:
        return ExecutableReasoningResult(
            "",
            False,
            {
                **base_receipt,
                "status": attempts[-1]["status"] if attempts else "deadline_exhausted",
                "generation_calls": generation_calls,
                "attempts": attempts,
                "elapsed_s": round(time.monotonic() - started, 6),
            },
        )

    sandbox_receipt = execution.to_dict()
    candidate = str(getattr(execution, "stdout", "") or "").strip()
    if len(candidate) > _MAX_STDOUT_CHARS:
        candidate = ""
    normalized_candidate, representation_repaired = _normalize_contract_candidate(
        candidate,
        response_contract,
    )
    contract_valid = bool(getattr(execution, "ok", False)) and bool(
        normalized_candidate
    )
    if contract_valid:
        candidate = normalized_candidate
    status = "candidate_ready" if contract_valid else (
        "sandbox_execution_failed" if not getattr(execution, "ok", False)
        else "candidate_contract_invalid"
    )
    receipt = {
        **base_receipt,
        "status": status,
        "generation_calls": generation_calls,
        "program_chars": len(program),
        "program_bytes": len(program.encode("utf-8")),
        "program_sha256": _sha256(program),
        "candidate_chars": len(candidate),
        "candidate_sha256": _sha256(candidate) if candidate else "",
        "contract_valid": contract_valid,
        "representation_repaired": representation_repaired,
        "sandbox": sandbox_receipt,
        "attempts": attempts,
        "elapsed_s": round(time.monotonic() - started, 6),
    }
    if not contract_valid:
        return ExecutableReasoningResult("", False, receipt)

    candidate = attributed_text(
        candidate,
        {
            **generated_metadata,
            "response_path": "executable_reasoning",
            "model_native_output": False,
            "sandbox_grounded": True,
            "executable_reasoning_receipt_sha256": _sha256(str(sorted(receipt.items()))),
        },
    )
    return ExecutableReasoningResult(candidate, True, receipt)


__all__ = [
    "EXECUTABLE_REASONING_SCHEMA",
    "EXECUTABLE_STRATEGIES",
    "ExecutableReasoningResult",
    "derive_executable_candidate",
    "select_executable_strategies",
    "should_use_executable_reasoning",
]
