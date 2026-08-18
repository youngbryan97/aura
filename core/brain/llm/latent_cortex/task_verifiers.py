"""Task-typed correctness verification INSIDE the live episode.

Closes the spec's central inference-time gap: "more stable internal thought
can still be confidently wrong." Until now the live route selected branches
by convergence quality and accepted latent-opt steps by proxy loss — internal
signals that measure stability, not truth. This module gives the episode a
deterministic, worker-safe verifier so branch selection and hill-climbing are
guided by CHECKED properties of candidate answers:

- **Arithmetic claims** — every "a op b = c" the candidate asserts is
  recomputed exactly; confidently wrong arithmetic is penalized per claim.
- **Code blocks** — fenced Python must compile (syntax truth, no execution:
  running model-authored code stays outside the worker, in the sandboxed
  service-side engines); other fenced blocks must be structurally balanced.
- **Facet coverage** — the SAME request_facets definition the product gate
  judges by scores whether the candidate addresses what was asked.
- **Objective grounding** — lexical overlap with the request's key terms,
  so a fluent non-answer scores below a grounded one.

Scores are deterministic, bounded [0, 1], and receipted with per-check
evidence, so a winning branch carries WHY it won ("passed 3/3 arithmetic
claims; python compiles") — not "converged prettier". No network, no
subprocess, no model calls: safe at any point inside the worker episode.
"""

from __future__ import annotations

import ast
import logging
import re
from collections.abc import Mapping
from typing import Any

from core.brain.llm.latent_cortex.atomic_decomposition import (
    decomposition_check,
)
from core.brain.llm.latent_cortex.deterministic_verifier_router import (
    router_check,
)
from core.brain.llm.latent_cortex.output_quality import (
    evaluate_facet_coverage,
)
from core.brain.llm.latent_cortex.response_contracts import (
    ResponseContractError,
    parse_response_contract,
    validate_response_payload,
)

logger = logging.getLogger("Aura.LatentCortex.TaskVerifiers")

TASK_VERIFIER_SCHEMA = "aura.latent_task_verifier.v4"

_FENCE_RE = re.compile(r"```([a-zA-Z0-9_+-]*)\n(.*?)```", re.DOTALL)
_WORD_RE = re.compile(r"[^\W\d_]{4,}", re.UNICODE)
_ANSWER_FACET_HINTS = {
    "compare": re.compile(
        r"\b(?:whereas|while|unlike|versus|compared|by\s+contrast|in\s+contrast)\b", re.I
    ),
    "select": re.compile(
        r"\b(?:choose|recommend|prefer|stronger|best|should\s+(?:use|choose|adopt)|the\s+winner)\b",
        re.I,
    ),
    "verify": re.compile(
        r"\b(?:verify|test|assert|inject|simulate|fault|cancel|timeout|restart|invariant|receipt)\w*\b",
        re.I,
    ),
    "explain": re.compile(
        r"\b(?:because|therefore|thus|so\s+that|leads?\s+to|prevents?|causes?|ensures?)\b", re.I
    ),
    "enumerate": re.compile(r"(?:^\s*(?:[-*+]|\d+[.)])\s+\S)", re.M),
}


def check_arithmetic_claims(text: str) -> dict[str, Any]:
    """Recompute every explicit arithmetic claim in the candidate."""
    from core.reasoning.symbolic_bridge import SymbolicBridge

    source = str(text or "")
    observations = [
        observation
        for observation in SymbolicBridge().inspect_arithmetic_claims(source)
        if not (
            "/" in str(observation["claim"])
            and not float(observation["correct"]).is_integer()
        )
    ]
    errors = [observation for observation in observations if not observation["valid"]]
    checked = len(observations)
    passed = checked - len(errors)
    failures = [
        f"{re.sub(r'\s+', '', str(error['claim']))} (actual {error['replacement']})"
        for error in errors[:8]
    ]
    return {
        "checked": checked,
        "passed": passed,
        "failures": failures,
        "score": (passed / checked) if checked else None,
    }


def check_code_blocks(text: str) -> dict[str, Any]:
    """Syntax-verify fenced code. Python must parse; others must balance."""
    checked = passed = 0
    failures: list[str] = []
    for match in _FENCE_RE.finditer(text or ""):
        language = (match.group(1) or "").strip().lower()
        body = match.group(2)
        if not body.strip():
            continue
        checked += 1
        if language in {"python", "py", ""} and not language.startswith("json"):
            try:
                ast.parse(body)
                passed += 1
            except SyntaxError as exc:
                if len(failures) < 8:
                    failures.append(f"python_syntax:{exc.lineno}:{exc.msg}")
        else:
            balanced = all(
                body.count(open_ch) == body.count(close_ch)
                for open_ch, close_ch in (("{", "}"), ("(", ")"), ("[", "]"))
            )
            if balanced:
                passed += 1
            elif len(failures) < 8:
                failures.append(f"{language or 'unknown'}:unbalanced_brackets")
    return {
        "checked": checked,
        "passed": passed,
        "failures": failures,
        "score": (passed / checked) if checked else None,
    }


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n{2,}")


def check_facet_coverage(text: str, objective: str) -> dict[str, Any]:
    return evaluate_facet_coverage(text, objective)


def check_degeneracy(text: str) -> dict[str, Any]:
    """Deterministic degeneration factor in [0.5, 1.0] for longer candidates.

    Two Goodhart shapes reduce it: repetition loops (trigram diversity
    collapse — the CP105 live failure shape) and facet-cue stuffing (cue
    density far above natural prose). Returned as a multiplicative factor so
    a degenerate candidate cannot buy its score back with one correct sum.
    """
    words = (text or "").split()
    if len(words) < 30:
        return {"applicable": False, "factor": 1.0}
    trigrams = [" ".join(words[i : i + 3]).lower() for i in range(len(words) - 2)]
    diversity = len(set(trigrams)) / max(1, len(trigrams))
    severity = max(0.0, (0.5 - diversity) * 2.0) if diversity < 0.5 else 0.0
    sentences = [s for s in _SENTENCE_SPLIT_RE.split(text or "") if s.strip()]
    cue_hits = sum(len(pattern.findall(text or "")) for pattern in _ANSWER_FACET_HINTS.values())
    cue_density = cue_hits / max(1, len(sentences))
    if cue_density > 1.5:
        severity += min(1.0, (cue_density - 1.5) / 2.0)
    factor = 1.0 - 0.5 * min(1.0, severity)
    return {
        "applicable": True,
        "factor": round(factor, 6),
        "trigram_diversity": round(diversity, 6),
        "cue_density": round(cue_density, 6),
    }


def check_objective_grounding(text: str, objective: str) -> dict[str, Any]:
    objective_terms = {word.lower() for word in _WORD_RE.findall(objective or "")}
    if not objective_terms:
        return {"matched": 0, "of": 0, "score": None}
    answer_terms = {word.lower() for word in _WORD_RE.findall(text or "")}
    matched = len(objective_terms & answer_terms)
    return {
        "matched": matched,
        "of": len(objective_terms),
        "score": min(1.0, matched / max(4, min(len(objective_terms), 16))),
    }


def check_response_contract(text: str, response_contract: str) -> dict[str, Any]:
    """Validate the public final-answer marker, JSON, keys, and value types."""

    from core.brain.llm.latent_cortex.frontier_tasks import (
        FrontierTaskError,
        parse_final_answer,
    )

    try:
        parsed_contract = parse_response_contract(response_contract)
        payload = parse_final_answer(text)
        validation = validate_response_payload(payload, parsed_contract)
    except (FrontierTaskError, ResponseContractError) as exc:
        return {
            "applicable": True,
            "valid": False,
            "failures": [str(exc)],
            "score": 0.0,
        }
    return {
        "applicable": True,
        "valid": validation["valid"],
        "failures": list(validation["errors"]),
        "score": 1.0 if validation["valid"] else 0.0,
    }


def response_satisfies_contract(text: str, response_contract: str) -> bool:
    """Return whether text is already a complete, exact public answer."""

    return bool(check_response_contract(text, response_contract)["valid"])


class EpisodeTaskVerifier:
    """Deterministic candidate scorer for one episode's objective.

    Instances are callables suitable for ``LatentCortexEngine.reason``'s
    ``verifier`` argument. Every scored candidate leaves an evidence row so
    the receipt can prove WHY the winner won. Weights renormalize over the
    checks that were actually applicable (no code in the answer ⇒ the code
    check neither helps nor hurts).
    """

    _WEIGHTS = {
        "atomic_decomposition": 0.40,
        "deterministic_router": 0.45,
        "response_contract": 0.55,
        "arithmetic": 0.35,
        "code": 0.25,
        "facets": 0.25,
        "grounding": 0.15,
    }

    def __init__(
        self,
        objective: str,
        *,
        response_contract: str = "",
        facet_reliability: dict[str, float] | None = None,
    ) -> None:
        self.objective = str(objective or "")
        self.response_contract = str(response_contract or "")
        if self.response_contract:
            parse_response_contract(self.response_contract)
        self.evaluations: list[dict[str, Any]] = []
        # Held-out calibration: per-facet reliability learned from GRADED
        # verdicts (Verifier Foundry Wilson bounds, human ground truth). A
        # facet whose cue-detector humans keep overruling is muted — it
        # earns less when satisfied and demands less when requested — so
        # "add the word because" stops being a strategy the moment grading
        # evidence says the cue is hollow. Neutral (1.0) until measured.
        self.facet_reliability: dict[str, float] = {}
        for name, value in (facet_reliability or {}).items():
            if (
                name in _ANSWER_FACET_HINTS
                and isinstance(value, (int, float))
                and not isinstance(value, bool)
                and 0.0 <= float(value) <= 1.0
            ):
                self.facet_reliability[name] = float(value)

    def _facet_weighted_score(self, facets: dict[str, Any]) -> float | None:
        requested = facets.get("requested") or []
        if not requested:
            return None
        satisfied = set(facets.get("satisfied") or [])
        total = sum(self.facet_reliability.get(name, 1.0) for name in requested)
        if total <= 0.0:
            return None
        earned = sum(
            self.facet_reliability.get(name, 1.0) for name in requested if name in satisfied
        )
        return earned / total

    def evaluate(
        self,
        text: str,
        *,
        _include_response_contract: bool = True,
        _record: bool = True,
    ) -> dict[str, Any]:
        atomic = decomposition_check(text, objective=self.objective)
        routed = router_check(
            text,
            objective=self.objective,
            atomic_receipt=atomic.get("receipt") or {},
        )
        checks = {
            "atomic_decomposition": atomic,
            "deterministic_router": routed,
            "arithmetic": check_arithmetic_claims(text),
            "code": check_code_blocks(text),
            "facets": check_facet_coverage(text, self.objective),
            "grounding": check_objective_grounding(text, self.objective),
        }
        if self.response_contract and _include_response_contract:
            checks["response_contract"] = check_response_contract(
                text,
                self.response_contract,
            )
        if self.facet_reliability:
            checks["facets"]["score"] = self._facet_weighted_score(checks["facets"])
            checks["facets"]["reliability_weighted"] = True
        weighted = total_weight = 0.0
        for name, result in checks.items():
            score = result.get("score")
            if score is None:
                continue
            weighted += self._WEIGHTS[name] * float(score)
            total_weight += self._WEIGHTS[name]
        # A candidate exercising no verifiable surface scores a neutral 0.5:
        # verifiability itself must not be punished, but it earns nothing.
        score = (weighted / total_weight) if total_weight > 0 else 0.5
        # Degeneration multiplies the composite down — a repetition loop or
        # cue-stuffed candidate cannot buy its rank back with one correct sum.
        degeneracy = check_degeneracy(text)
        if degeneracy.get("applicable"):
            score *= float(degeneracy["factor"])
        # Holistic checks cannot acquire branch-selection authority when the
        # candidate omitted a structurally required dependency or source span.
        if text.strip() and (not atomic["valid"] or not routed["valid"]):
            score = min(score, 0.25)
        row = {
            "schema": TASK_VERIFIER_SCHEMA,
            "score": round(score, 6),
            "applicable_checks": [
                name for name, result in checks.items() if result.get("score") is not None
            ],
            "unverified": total_weight <= 0,
            "grade_admissible": bool(atomic["valid"]),
            "degeneracy": degeneracy,
            "checks": checks,
            "text_chars": len(text or ""),
        }
        if _record:
            self.evaluations.append(row)
        return row

    def __call__(self, text: str) -> float:
        return float(self.evaluate(text)["score"])

    def latent_state_score(self, text: str) -> float:
        """Score semantic progress before the separate wire-format repair.

        This score has latent-search authority only. It deliberately omits the
        public response-object shape because raw state probes are evaluated
        before bounded representation repair. The strict response contract is
        still mandatory for branch admission, answer selection, and serving.
        """

        return float(
            self.evaluate(
                text,
                _include_response_contract=False,
                _record=False,
            )["score"]
        )

    def fast_weight_learning_evidence(
        self,
        candidate: str,
        *,
        evaluation_index: int,
        tokenizer: Any,
        structural_diversity: Mapping[str, Any],
    ) -> tuple[dict[str, Any], list[int]]:
        """Return the exact-evidence subset eligible for temporary learning.

        This is intentionally narrower than the composite verifier score.
        Facet cues, lexical grounding, and self-consistency may rank branches,
        but only machine-checked atomic routes may become a gradient target.
        """

        if type(evaluation_index) is not int or not 0 <= evaluation_index < len(
            self.evaluations
        ):
            raise ValueError("fast-weight evidence evaluation index is invalid")
        from core.brain.llm.latent_cortex.fast_weight_learning import (
            build_fast_weight_admission,
        )

        return build_fast_weight_admission(
            self.evaluations[evaluation_index],
            candidate=candidate,
            objective=self.objective,
            evaluation_index=evaluation_index,
            tokenizer=tokenizer,
            structural_diversity=structural_diversity,
        )

    def to_receipt(
        self,
        *,
        exclude_evaluation_indices: set[int] | None = None,
    ) -> dict[str, Any]:
        """Bounded evidence: every evaluation's score + the best row's why."""
        excluded = set(exclude_evaluation_indices or ())
        if any(
            type(index) is not int or not 0 <= index < len(self.evaluations) for index in excluded
        ):
            raise ValueError("excluded verifier evaluation index is invalid")
        evaluations = [row for index, row in enumerate(self.evaluations) if index not in excluded]
        if not evaluations:
            return {
                "schema": TASK_VERIFIER_SCHEMA,
                "requested": True,
                "available": True,
                "evaluations": 0,
                "response_contract_required": bool(self.response_contract),
                "response_contract_satisfied": False,
                "outcome_checked": False,
                "outcome_passed": None,
                "outcome_reason": "candidate_checks_are_not_task_ground_truth",
            }
        best = max(evaluations, key=lambda row: row["score"])
        facets = best["checks"].get("facets") or {}
        # Per-facet judgments on the WINNING candidate, excerpt included —
        # the held-out grading surface. An operator (or downstream ground
        # truth) grades whether the excerpt really addresses the facet; the
        # grades calibrate facet_reliability for future episodes.
        judgments = [
            {
                "facet": name,
                "satisfied": name in (facets.get("satisfied") or []),
                "excerpt": str((facets.get("excerpts") or {}).get(name, ""))[:200],
            }
            for name in (facets.get("requested") or [])
        ]
        return {
            "schema": TASK_VERIFIER_SCHEMA,
            "requested": True,
            "available": True,
            "evaluations": len(evaluations),
            "score_trail": [row["score"] for row in evaluations[:32]],
            "best_score": best["score"],
            "best_applicable_checks": list(best["applicable_checks"]),
            "best_failures": {
                name: list(result.get("failures") or [])
                for name, result in best["checks"].items()
                if result.get("failures")
            },
            "atomic_decomposition": dict(
                best["checks"]["atomic_decomposition"].get("receipt") or {}
            ),
            "deterministic_router": dict(
                best["checks"]["deterministic_router"].get("receipt") or {}
            ),
            "grade_admissible": bool(best.get("grade_admissible")),
            "facet_judgments": judgments,
            "facet_reliability": dict(self.facet_reliability),
            "response_contract_required": bool(self.response_contract),
            "response_contract_satisfied": bool(
                best["checks"].get("response_contract", {}).get("valid", False)
            ),
            # These candidate-local checks rank branches and reject concrete
            # defects. They are not a ground-truth grade of the whole task, so
            # the execution bandit must not count them as a successful trial.
            "outcome_checked": False,
            "outcome_passed": None,
            "outcome_reason": "candidate_checks_are_not_task_ground_truth",
        }


__all__ = [
    "EpisodeTaskVerifier",
    "TASK_VERIFIER_SCHEMA",
    "check_arithmetic_claims",
    "check_code_blocks",
    "check_degeneracy",
    "check_facet_coverage",
    "check_objective_grounding",
    "check_response_contract",
    "response_satisfies_contract",
]
