"""Exact verification for public, self-contained objective programs.

The verifier derives an answer only from the public objective. It never
receives a benchmark answer or private grader state. Recognized objective
families are parsed into a bounded program, executed deterministically, and
compared with the candidate's strict terminal JSON object.
"""

from __future__ import annotations

import ast
import hashlib
import itertools
import json
import math
import re
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from core.brain.llm.latent_cortex.typed_action_compiler import (
    compile_public_transition_program,
)
from core.brain.llm.latent_cortex.typed_program_executor import (
    execute_compiled_action_program,
)

OBJECTIVE_PROGRAM_VERIFIER_SCHEMA = "aura.rlc.objective_program_verifier.v4"
OBJECTIVE_PROGRAM_SOLUTION_SCHEMA = "aura.rlc.objective_program_solution.v4"

_MODULAR_OBJECTIVE_RE = re.compile(
    r"\AStart at the given value and apply each operation modulo "
    r"(?P<modulus>\d+): start=(?P<start>-?\d+)\. Operations: "
    r"(?P<operations>[+*\-]\d+(?:,\s*[+*\-]\d+)*)\.",
)
_BOOLEAN_OBJECTIVE_RE = re.compile(
    r"\AEvaluate this (?P<depth>\d+)-operation expression with 1=true, "
    r"0=false, and xor meaning exactly one operand is true: "
    r"(?P<expression>.+?)\. Return a value of 1 or 0\.",
)
_STABLE_NEAREST_TRAVERSAL_RE = re.compile(
    r"\AFresh algorithm task\. The input values, in original position order, are "
    r"(?P<values>\[-?\d+(?:,\s*-?\d+){1,255}\])\. Select the "
    r"(?P<median>lower|upper) median by numeric value first\. Then repeatedly "
    r"select one remaining value by minimizing, in order: absolute distance from "
    r"the most recently selected value; numeric value; original zero-based "
    r"position\. Return the complete selected-value sequence\. Its checksum is the "
    r"sum of one-based output position multiplied by value\.",
)
_SEPARATED_SUBSET_RE = re.compile(
    r"\AFresh combinatorics task\. From the set (?P<values>\[-?\d+(?:,\s*-?\d+){1,31}\]), "
    r"choose exactly (?P<count>\d+) distinct values\. Adjacent values in sorted chosen "
    r"order must differ by at least (?P<separation>\d+), and the chosen sum must be "
    r"from (?P<low>-?\d+) through (?P<high>-?\d+), inclusive\. Count all valid subsets "
    r"and give the lexicographically smallest valid subset in ascending order\.",
)
_STATEFUL_TRACE_RE = re.compile(
    r"\AFresh code-semantics task\. Evaluate this exact Python function without executing it:\n\n"
    r"def audit\(events\):\n"
    r"    balances = \{\}\n"
    r"    pressure = \[\]\n"
    r"    for name, delta in events:\n"
    r"        balances\[name\] = balances\.get\(name, 0\) \+ delta\n"
    r"        if balances\[name\] == 0:\n"
    r"            del balances\[name\]\n"
    r"        pressure\.append\(sum\(abs\(v\) for v in balances\.values\(\)\)\)\n"
    r"    return sorted\(balances\.items\(\)\), pressure\n\n"
    r"The two inputs, in order, are (?P<inputs>\[\[.*?\]\])\. Return each result as an "
    r"object whose state is a JSON list of \[name, value\] pairs and whose pressure is a "
    r"list\. Also report the tight worst-case time complexity in n events, assuming "
    r"dictionary operations are O\(1\)\.",
    re.DOTALL,
)
# The baseline list is NOT in root-mediator-downstream order — the generator
# shuffles it so the roles cannot be read off the sentence. This regex used to
# name the three baselines by role, so it matched nothing and the deterministic
# critic silently declined every scientific-inference task: a check that could
# never fire. Baselines are captured by position; the roles come from the
# intervention clauses, which name them, exactly as the action compiler in
# core/learning/public_frontier_action_compiler.py already did.
_CAUSAL_CHAIN_RE = re.compile(
    r"\AFresh causal-inference task\. Three measured variables have baseline values "
    r"(?P<label_a>[a-z][a-z0-9_]*)=(?P<base_a>-?\d+), "
    r"(?P<label_b>[a-z][a-z0-9_]*)=(?P<base_b>-?\d+), "
    r"(?P<label_c>[a-z][a-z0-9_]*)=(?P<base_c>-?\d+)\. "
    r"Independent interventions produced these changes relative to baseline: setting "
    r"(?P<root>[a-z][a-z0-9_]*) up by (?P<root_delta>\d+) changed "
    r"(?P<root_target_1>[a-z][a-z0-9_]*) by (?P<root_change_1>[+-]\d+) and "
    r"(?P<root_target_2>[a-z][a-z0-9_]*) by (?P<root_change_2>[+-]\d+); setting "
    r"(?P<mediator>[a-z][a-z0-9_]*) up by (?P<mediator_delta>\d+) left (?P=root) "
    r"unchanged and changed (?P<downstream>[a-z][a-z0-9_]*) by "
    r"(?P<mediator_downstream_change>[+-]\d+); setting (?P=downstream) up by "
    r"(?P<downstream_delta>\d+) left both other variables unchanged\. Assume "
    r"deterministic linear effects and no hidden common cause\. Identify root, "
    r"mediator, and downstream variables, then predict the absolute value of "
    r"(?P=downstream) when (?P=root) is set (?P<predict_delta>\d+) above baseline\.",
)
_PLANNING_RE = re.compile(
    r"\AFresh planning task\. One crew executes at most one task at a time, starts at "
    r"time 0, and may skip tasks\. A selected task may start only after every required "
    r"task has completed\. It earns its reward only if completion is no later than its "
    r"own deadline and the overall horizon (?P<horizon>\d+)\. Tasks are "
    r"(?P<tasks>\[\{.*?\}\])\. Maximize total reward; then minimize makespan\. "
    r"(?:Break any remaining tie by selecting the alphabetically earliest sequence "
    r"of task labels|then choose the lexicographically smallest task-name sequence)\. "
    r"Return the selected order, reward, and makespan\.",
    re.DOTALL,
)
_BAYES_RE = re.compile(
    r"\AFresh calibration task\. Before evidence E, hypothesis H has probability "
    r"(?P<prior>\d+/\d+)\. The likelihood of E is (?P<like_h>\d+/\d+) if H is true "
    r"and (?P<like_not_h>\d+/\d+) if H is false\. Using exact Bayes updating, return "
    r"the more probable choice \(H wins ties\), the reduced posterior probability of H, "
    r"and its band: below_50, 50_to_69, 70_to_89, or 90_to_100\.",
)
_PREMISE_AUDIT_RE = re.compile(
    r"\AFresh premise-audit task\. For each project, score = impact multiplied by "
    r"reliability minus cost\. The data are (?P<projects>\[\{.*?\}\])\. The claim says "
    r"project (?P<claim>[A-Z][A-Z0-9_]*) has the highest score\. Check the premise "
    r"instead of accepting it\. Ties are won by the lexicographically smaller name\. "
    r"Return whether the claim is valid, the actual winner, and its score\.",
    re.DOTALL,
)
_BOOLEAN_TOKEN_RE = re.compile(r"\s*(?:(?P<bit>[01])|(?P<op>not|and|or|xor)|(?P<paren>[()]))")
_JSON_FENCE_RE = re.compile(
    r"\A```(?:json)?[ \t]*\r?\n(?P<body>\{[\s\S]*\})\r?\n```\Z",
    re.IGNORECASE,
)


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


@dataclass(slots=True)
class _BooleanParser:
    tokens: list[str]
    index: int = 0
    operations: int = 0

    def parse(self) -> bool:
        value = self._or()
        if self.index != len(self.tokens):
            raise ValueError("boolean_expression_trailing_tokens")
        return value

    def _peek(self) -> str:
        return self.tokens[self.index] if self.index < len(self.tokens) else ""

    def _take(self, expected: str | None = None) -> str:
        if self.index >= len(self.tokens):
            raise ValueError("boolean_expression_truncated")
        value = self.tokens[self.index]
        if expected is not None and value != expected:
            raise ValueError("boolean_expression_token_mismatch")
        self.index += 1
        return value

    def _or(self) -> bool:
        value = self._xor()
        while self._peek() == "or":
            self._take("or")
            self.operations += 1
            right = self._xor()
            value = value or right
        return value

    def _xor(self) -> bool:
        value = self._and()
        while self._peek() == "xor":
            self._take("xor")
            self.operations += 1
            value = value != self._and()
        return value

    def _and(self) -> bool:
        value = self._not()
        while self._peek() == "and":
            self._take("and")
            self.operations += 1
            right = self._not()
            value = value and right
        return value

    def _not(self) -> bool:
        if self._peek() == "not":
            self._take("not")
            self.operations += 1
            return not self._not()
        return self._atom()

    def _atom(self) -> bool:
        token = self._take()
        if token in {"0", "1"}:
            return token == "1"
        if token != "(":
            raise ValueError("boolean_expression_atom_invalid")
        value = self._or()
        self._take(")")
        return value


def _boolean_tokens(expression: str) -> list[str]:
    tokens: list[str] = []
    cursor = 0
    while cursor < len(expression):
        match = _BOOLEAN_TOKEN_RE.match(expression, cursor)
        if match is None:
            raise ValueError("boolean_expression_lex_invalid")
        tokens.append(match.group("bit") or match.group("op") or match.group("paren"))
        cursor = match.end()
    if not tokens or len(tokens) > 4_096:
        raise ValueError("boolean_expression_size_invalid")
    return tokens


def _candidate_payload(candidate: str) -> dict[str, Any]:
    from core.brain.llm.latent_cortex.frontier_tasks import parse_final_answer

    if "FINAL_ANSWER:" in candidate:
        payload = parse_final_answer(candidate)
    else:
        # Some checkpoints return only the requested JSON object, commonly in
        # one markdown fence.  That is a contract-format defect, but the
        # semantic answer remains uniquely identifiable.  Accept only a whole
        # response object or one whole-response JSON fence: prose-wrapped,
        # multiple, or otherwise ambiguous payloads remain unverifiable.
        bounded = candidate.strip()
        fence = _JSON_FENCE_RE.fullmatch(bounded)
        encoded = fence.group("body") if fence is not None else bounded
        if not (encoded.startswith("{") and encoded.endswith("}")):
            raise ValueError("terminal_answer_not_uniquely_bounded")
        payload = json.loads(encoded)
    if not isinstance(payload, dict):
        raise ValueError("terminal_answer_not_object")
    return payload


def _modular_expected(match: re.Match[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    modulus = int(match.group("modulus"))
    start = int(match.group("start"))
    if not 2 <= modulus <= 1_000_000 or not -1_000_000_000 <= start <= 1_000_000_000:
        raise ValueError("modular_program_bounds_invalid")
    value = start % modulus
    operations = [item.strip() for item in match.group("operations").split(",")]
    if not 1 <= len(operations) <= 1_024:
        raise ValueError("modular_program_length_invalid")
    trace: list[int] = [value]
    for operation in operations:
        operand = int(operation[1:])
        if not 0 <= operand <= 1_000_000_000:
            raise ValueError("modular_operand_bounds_invalid")
        if operation[0] == "+":
            value = (value + operand) % modulus
        elif operation[0] == "-":
            value = (value - operand) % modulus
        else:
            value = (value * operand) % modulus
        trace.append(value)
    return {"residue": value}, {
        "modulus": modulus,
        "operation_count": len(operations),
        "trace_sha256": _sha(trace),
    }


def _boolean_expected(match: re.Match[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    declared_depth = int(match.group("depth"))
    parser = _BooleanParser(_boolean_tokens(match.group("expression")))
    value = parser.parse()
    if parser.operations != declared_depth:
        raise ValueError("boolean_operation_count_mismatch")
    return {"value": int(value)}, {
        "declared_operations": declared_depth,
        "executed_operations": parser.operations,
        "expression_sha256": _text_sha(match.group("expression")),
    }


def _stable_nearest_expected(
    match: re.Match[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    values = json.loads(match.group("values"))
    if (
        not isinstance(values, list)
        or not 2 <= len(values) <= 256
        or any(isinstance(value, bool) or not isinstance(value, int) for value in values)
        or any(not -(1 << 31) <= value <= (1 << 31) - 1 for value in values)
    ):
        raise ValueError("stable_nearest_values_invalid")
    indexed = list(enumerate(values))
    median_rank = (len(values) - 1) // 2 if match.group("median") == "lower" else len(values) // 2
    first = sorted(indexed, key=lambda item: (item[1], item[0]))[median_rank]
    order = [first]
    remaining = [item for item in indexed if item != first]
    while remaining:
        previous_value = order[-1][1]
        chosen = min(
            remaining,
            key=lambda item: (
                abs(item[1] - previous_value),
                item[1],
                item[0],
            ),
        )
        order.append(chosen)
        remaining.remove(chosen)
    sequence = [value for _original_index, value in order]
    checksum = sum(output_index * value for output_index, value in enumerate(sequence, start=1))
    return {"sequence": sequence, "checksum": checksum}, {
        "input_count": len(values),
        "median_kind": match.group("median"),
        "median_rank_zero_based": median_rank,
        "original_indices_sha256": _sha([index for index, _value in order]),
        "selection_trace_sha256": _sha(sequence),
    }


def _literal_list(raw: str, *, role: str) -> list[Any]:
    try:
        value = ast.literal_eval(raw)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"{role}_literal_invalid") from exc
    if not isinstance(value, list):
        raise ValueError(f"{role}_not_list")
    return value


def _separated_subset_expected(
    match: re.Match[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    values = json.loads(match.group("values"))
    choose = int(match.group("count"))
    separation = int(match.group("separation"))
    low, high = int(match.group("low")), int(match.group("high"))
    if (
        not isinstance(values, list)
        or not 2 <= len(values) <= 32
        or len(set(values)) != len(values)
        or any(type(value) is not int or not -(1 << 31) <= value < (1 << 31) for value in values)
        or not 1 <= choose <= min(10, len(values))
        or not 0 <= separation <= 1_000_000
        or low > high
    ):
        raise ValueError("separated_subset_bounds_invalid")
    valid: list[tuple[int, ...]] = []
    for candidate in itertools.combinations(sorted(values), choose):
        if all(b - a >= separation for a, b in zip(candidate, candidate[1:], strict=False)):
            total = sum(candidate)
            if low <= total <= high:
                valid.append(candidate)
    witness = list(min(valid)) if valid else []
    return {"count": len(valid), "witness": witness}, {
        "input_count": len(values),
        "choose": choose,
        "candidate_count": sum(1 for _ in itertools.combinations(values, choose)),
        "valid_subsets_sha256": _sha(valid),
    }


def _stateful_trace_expected(
    match: re.Match[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    inputs = _literal_list(match.group("inputs"), role="stateful_inputs")
    if len(inputs) != 2 or any(not isinstance(events, list) for events in inputs):
        raise ValueError("stateful_inputs_shape_invalid")
    returns: list[dict[str, Any]] = []
    event_count = 0
    for events in inputs:
        if not 1 <= len(events) <= 128:
            raise ValueError("stateful_event_count_invalid")
        balances: dict[str, int] = {}
        pressure: list[int] = []
        for event in events:
            if (
                not isinstance(event, (list, tuple))
                or len(event) != 2
                or not isinstance(event[0], str)
                or re.fullmatch(r"[a-z][a-z0-9_]{0,31}", event[0]) is None
                or type(event[1]) is not int
                or not -1_000_000 <= event[1] <= 1_000_000
            ):
                raise ValueError("stateful_event_invalid")
            name, delta = event
            balances[name] = balances.get(name, 0) + delta
            if balances[name] == 0:
                del balances[name]
            pressure.append(sum(abs(value) for value in balances.values()))
        event_count += len(events)
        returns.append(
            {
                "state": [[name, value] for name, value in sorted(balances.items())],
                "pressure": pressure,
            }
        )
    return {"returns": returns, "time_complexity": "O(n^2)"}, {
        "input_count": len(inputs),
        "event_count": event_count,
        "returns_sha256": _sha(returns),
    }


def _causal_chain_expected(
    match: re.Match[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    labels = (
        match.group("root"),
        match.group("mediator"),
        match.group("downstream"),
    )
    root_delta = int(match.group("root_delta"))
    # The root's two effects are listed in either order, so read them by name
    # rather than by position — the same reason the roles are read from the
    # clauses instead of from the baseline list.
    root_effects = {
        match.group("root_target_1"): int(match.group("root_change_1")),
        match.group("root_target_2"): int(match.group("root_change_2")),
    }
    if set(root_effects) != {match.group("mediator"), match.group("downstream")}:
        raise ValueError("causal_root_effects_do_not_name_the_roles")
    mediator_change = root_effects[match.group("mediator")]
    mediator_delta = int(match.group("mediator_delta"))
    mediator_downstream_change = int(match.group("mediator_downstream_change"))
    downstream_delta = int(match.group("downstream_delta"))
    predict_delta = int(match.group("predict_delta"))
    # Baselines are listed in an order the generator chooses, so pair each one
    # with its own label and look the downstream variable up by name.
    baselines = {
        match.group(f"label_{slot}"): int(match.group(f"base_{slot}"))
        for slot in ("a", "b", "c")
    }
    if set(baselines) != set(labels):
        raise ValueError("causal_baselines_do_not_name_the_roles")
    downstream_base = baselines[match.group("downstream")]
    downstream_change = root_effects[match.group("downstream")]
    numeric_values = (
        tuple(baselines.values())
        + (mediator_change, downstream_change)
        + tuple(
            int(match.group(field))
            for field in (
                "root_delta",
                "mediator_delta",
                "mediator_downstream_change",
                "downstream_delta",
                "predict_delta",
            )
        )
    )
    if len(set(labels)) != 3 or any(abs(value) > 1_000_000_000 for value in numeric_values):
        raise ValueError("causal_program_bounds_invalid")
    if min(root_delta, mediator_delta, downstream_delta) <= 0 or predict_delta < 0:
        raise ValueError("causal_delta_invalid")
    root_to_mediator = Fraction(mediator_change, root_delta)
    mediator_to_downstream = Fraction(
        mediator_downstream_change,
        mediator_delta,
    )
    root_total_effect = Fraction(downstream_change, root_delta)
    if root_to_mediator * mediator_to_downstream != root_total_effect:
        raise ValueError("causal_interventions_inconsistent")
    predicted = root_total_effect * predict_delta
    if predicted.denominator != 1:
        raise ValueError("causal_prediction_not_integral")
    payload = {
        "root": match.group("root"),
        "mediator": match.group("mediator"),
        "downstream": match.group("downstream"),
        "predicted_downstream": downstream_base + predicted.numerator,
    }
    return payload, {
        "root_intervention_delta": root_delta,
        "root_total_downstream_effect": downstream_change,
        "root_to_mediator_effect": str(root_to_mediator),
        "mediator_to_downstream_effect": str(mediator_to_downstream),
        "prediction_delta": predict_delta,
        "total_effect_scaled_once": True,
    }


def _planning_expected(match: re.Match[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    horizon = int(match.group("horizon"))
    tasks = _literal_list(match.group("tasks"), role="planning_tasks")
    if not 1 <= horizon <= 1_000 or not 1 <= len(tasks) <= 9:
        raise ValueError("planning_bounds_invalid")
    by_name: dict[str, dict[str, Any]] = {}
    for task in tasks:
        if not isinstance(task, dict) or set(task) != {
            "name",
            "duration",
            "deadline",
            "reward",
            "requires",
        }:
            raise ValueError("planning_task_shape_invalid")
        name = task["name"]
        if (
            not isinstance(name, str)
            or re.fullmatch(r"[A-Z][A-Z0-9_]{0,15}", name) is None
            or name in by_name
            or type(task["duration"]) is not int
            or type(task["deadline"]) is not int
            or type(task["reward"]) is not int
            or not 1 <= task["duration"] <= horizon
            or not 1 <= task["deadline"] <= horizon
            or not 0 <= task["reward"] <= 1_000_000
            or not isinstance(task["requires"], list)
            or any(not isinstance(item, str) for item in task["requires"])
            or len(set(task["requires"])) != len(task["requires"])
            or name in task["requires"]
        ):
            raise ValueError("planning_task_invalid")
        by_name[name] = task
    if any(requirement not in by_name for task in tasks for requirement in task["requires"]):
        raise ValueError("planning_requirement_unknown")
    best: tuple[int, int, tuple[str, ...]] | None = None
    feasible_count = 0
    names = sorted(by_name)
    for size in range(len(names) + 1):
        for order in itertools.permutations(names, size):
            selected = set(order)
            if any(not set(by_name[name]["requires"]).issubset(selected) for name in order):
                continue
            completed: set[str] = set()
            elapsed = 0
            reward = 0
            valid = True
            for name in order:
                task = by_name[name]
                if not set(task["requires"]).issubset(completed):
                    valid = False
                    break
                elapsed += task["duration"]
                if elapsed > task["deadline"] or elapsed > horizon:
                    valid = False
                    break
                reward += task["reward"]
                completed.add(name)
            if not valid:
                continue
            feasible_count += 1
            score = (-reward, elapsed, order)
            if best is None or score < best:
                best = score
    if best is None:  # the empty sequence is always feasible
        raise ValueError("planning_no_feasible_schedule")
    return {"order": list(best[2]), "reward": -best[0], "makespan": best[1]}, {
        "task_count": len(tasks),
        "feasible_schedule_count": feasible_count,
        "search_space_upper_bound": sum(
            math.factorial(len(tasks)) // math.factorial(len(tasks) - size)
            for size in range(len(tasks) + 1)
        ),
    }


def _bayes_expected(match: re.Match[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        prior = Fraction(match.group("prior"))
        like_h = Fraction(match.group("like_h"))
        like_not_h = Fraction(match.group("like_not_h"))
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError("bayes_probability_invalid") from exc
    if any(not 0 <= value <= 1 for value in (prior, like_h, like_not_h)):
        raise ValueError("bayes_probability_bounds_invalid")
    evidence = prior * like_h + (1 - prior) * like_not_h
    if evidence == 0:
        raise ValueError("bayes_zero_evidence")
    posterior = prior * like_h / evidence
    if posterior < Fraction(1, 2):
        band = "below_50"
    elif posterior < Fraction(7, 10):
        band = "50_to_69"
    elif posterior < Fraction(9, 10):
        band = "70_to_89"
    else:
        band = "90_to_100"
    return {
        "choice": "H" if posterior >= Fraction(1, 2) else "not_H",
        "posterior": f"{posterior.numerator}/{posterior.denominator}",
        "confidence_band": band,
    }, {
        "prior": str(prior),
        "evidence_probability": str(evidence),
        "posterior_sha256": _text_sha(str(posterior)),
    }


def _premise_audit_expected(
    match: re.Match[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    projects = _literal_list(match.group("projects"), role="premise_projects")
    if not 1 <= len(projects) <= 64:
        raise ValueError("premise_project_count_invalid")
    scored: list[tuple[int, str]] = []
    names: set[str] = set()
    for project in projects:
        if not isinstance(project, dict) or set(project) != {
            "name",
            "impact",
            "reliability",
            "cost",
        }:
            raise ValueError("premise_project_shape_invalid")
        name = project["name"]
        values = (project["impact"], project["reliability"], project["cost"])
        if (
            not isinstance(name, str)
            or re.fullmatch(r"[A-Z][A-Z0-9_]{0,15}", name) is None
            or name in names
            or any(
                type(value) is not int or not -1_000_000 <= value <= 1_000_000 for value in values
            )
        ):
            raise ValueError("premise_project_invalid")
        names.add(name)
        scored.append((project["impact"] * project["reliability"] - project["cost"], name))
    winner_score, winner = min(scored, key=lambda row: (-row[0], row[1]))
    return {
        "premise_valid": winner == match.group("claim"),
        "actual_winner": winner,
        "actual_score": winner_score,
    }, {
        "project_count": len(projects),
        "score_table_sha256": _sha(sorted((name, score) for score, name in scored)),
    }


def _certified_compiled_transition_expected(
    objective: str,
) -> tuple[str, dict[str, Any], dict[str, Any]] | None:
    """The critic's transition engine: typed, certified, and MLX-free.

    Named a "fallback" while the neural path sat in front of it. It is the
    primary engine for grading — a critic that runs the student's own learned
    tissue to decide whether the student was right is not an independent check,
    whatever the result. The neural rollin now lives in
    neural_objective_producer.py.
    """

    try:
        program = compile_public_transition_program(objective)
    # not a failure: an objective that will not compile is not a public program.
    except ValueError:
        return None
    execution = execute_compiled_action_program(program)
    if program.family == "boolean":
        match = _BOOLEAN_OBJECTIVE_RE.match(objective)
        if match is None:
            raise RuntimeError("compiled Boolean objective lost parser agreement")
        family = "nested_boolean"
        expected = {"value": execution.terminal_state[1]}
        crosscheck, crosscheck_receipt = _boolean_expected(match)
    elif program.family == "modular":
        match = _MODULAR_OBJECTIVE_RE.match(objective)
        if match is None:
            raise RuntimeError("compiled modular objective lost parser agreement")
        family = "modular_chain"
        expected = {"residue": execution.terminal_state[1]}
        crosscheck, crosscheck_receipt = _modular_expected(match)
    else:  # pragma: no cover - the compiler's family registry is closed
        raise RuntimeError("compiled transition family is unsupported")
    if crosscheck != expected:
        raise RuntimeError("certified recurrent execution and independent parser disagree")
    return family, expected, {
        "engine": "certified_typed_recurrence.v1",
        "compiler": program.public_receipt(),
        "student_rollin": execution.receipt(),
        "independent_crosscheck": crosscheck_receipt,
        "independent_crosscheck_match": True,
    }


def _execute_objective(objective: str) -> tuple[str, dict[str, Any], dict[str, Any]] | None:
    if not isinstance(objective, str):
        raise TypeError("objective program must be text")
    compiled = _certified_compiled_transition_expected(objective)
    if compiled is not None:
        return compiled
    family = ""
    expected: dict[str, Any]
    execution: dict[str, Any]
    modular = _MODULAR_OBJECTIVE_RE.match(objective)
    boolean = _BOOLEAN_OBJECTIVE_RE.match(objective)
    stable_nearest = _STABLE_NEAREST_TRAVERSAL_RE.match(objective)
    separated_subset = _SEPARATED_SUBSET_RE.match(objective)
    stateful_trace = _STATEFUL_TRACE_RE.match(objective)
    causal_chain = _CAUSAL_CHAIN_RE.match(objective)
    planning = _PLANNING_RE.match(objective)
    bayes = _BAYES_RE.match(objective)
    premise_audit = _PREMISE_AUDIT_RE.match(objective)
    if modular is not None:
        family = "modular_chain"
        expected, execution = _modular_expected(modular)
    elif boolean is not None:
        family = "nested_boolean"
        expected, execution = _boolean_expected(boolean)
    elif stable_nearest is not None:
        family = "stable_nearest_traversal"
        expected, execution = _stable_nearest_expected(stable_nearest)
    elif separated_subset is not None:
        family = "separated_subset_count"
        expected, execution = _separated_subset_expected(separated_subset)
    elif stateful_trace is not None:
        family = "stateful_python_trace"
        expected, execution = _stateful_trace_expected(stateful_trace)
    elif causal_chain is not None:
        family = "interventional_chain_inference"
        expected, execution = _causal_chain_expected(causal_chain)
    elif planning is not None:
        family = "dependency_deadline_portfolio"
        expected, execution = _planning_expected(planning)
    elif bayes is not None:
        family = "bayesian_frequency_update"
        expected, execution = _bayes_expected(bayes)
    elif premise_audit is not None:
        family = "premise_audit_table"
        expected, execution = _premise_audit_expected(premise_audit)
    else:
        return None
    return family, expected, execution


def _render_solution_witness(
    objective: str,
    *,
    family: str,
    expected: dict[str, Any],
) -> str:
    """Render a public-input derivation without exposing private grader state."""

    lines: list[str] = []
    if family == "modular_chain":
        match = _MODULAR_OBJECTIVE_RE.match(objective)
        if match is None:  # pragma: no cover - guarded by _execute_objective
            raise ValueError("modular_solution_witness_parse_failed")
        modulus = int(match.group("modulus"))
        value = int(match.group("start")) % modulus
        lines.append(f"Start with {value} modulo {modulus}.")
        for index, operation in enumerate(
            (item.strip() for item in match.group("operations").split(",")),
            start=1,
        ):
            before = value
            operand = int(operation[1:])
            if operation[0] == "+":
                value = (value + operand) % modulus
            elif operation[0] == "-":
                value = (value - operand) % modulus
            else:
                value = (value * operand) % modulus
            lines.append(
                f"Step {index}: {before} {operation[0]} {operand} = {value} (mod {modulus})."
            )
    elif family == "nested_boolean":
        match = _BOOLEAN_OBJECTIVE_RE.match(objective)
        if match is None:  # pragma: no cover - guarded by _execute_objective
            raise ValueError("boolean_solution_witness_parse_failed")
        value = int(expected["value"])
        truth = "true" if value else "false"
        lines.append(
            f"Evaluate {match.group('expression')} using not, and, xor, then or precedence."
        )
        lines.append(
            f"The bounded parser executed {int(match.group('depth'))} operations and the expression is {truth}, encoded as {value}."
        )
    elif family == "stable_nearest_traversal":
        match = _STABLE_NEAREST_TRAVERSAL_RE.match(objective)
        if match is None:  # pragma: no cover - guarded by _execute_objective
            raise ValueError("stable_nearest_solution_witness_parse_failed")
        values = json.loads(match.group("values"))
        median_rank = (
            (len(values) - 1) // 2 if match.group("median") == "lower" else len(values) // 2
        )
        lines.append(
            f"Sort value/index pairs and choose the {match.group('median')} median at zero-based rank {median_rank}."
        )
        lines.append(
            "Then choose each remaining pair by (absolute distance, value, original index) and compute the weighted checksum."
        )
    elif family == "separated_subset_count":
        lines.append(
            "Enumerate every fixed-size subset, retain only subsets satisfying the adjacent-separation and inclusive-sum constraints, then choose the lexicographically first witness."
        )
    elif family == "stateful_python_trace":
        lines.append(
            "Simulate each event in order with an explicit balance map, delete zero balances, and recompute the pressure after every event."
        )
    elif family == "interventional_chain_inference":
        lines.append(
            "Use the asymmetric interventions to identify the root-to-mediator-to-downstream order and scale the observed root total effect exactly once."
        )
    elif family == "dependency_deadline_portfolio":
        lines.append(
            "Enumerate bounded task orders, enforce prerequisites and completion deadlines, then rank feasible schedules by reward, makespan, and lexical order."
        )
    elif family == "bayesian_frequency_update":
        lines.append(
            "Compute the exact evidence mass and normalize the H branch with rational arithmetic before assigning the declared confidence band."
        )
    elif family == "premise_audit_table":
        lines.append(
            "Recompute every project score and rank by descending score with the declared lexical tie-break before checking the premise."
        )
    else:  # pragma: no cover - family is closed by _execute_objective
        raise ValueError("objective_solution_witness_family_unknown")
    lines.append(
        "FINAL_ANSWER: "
        + json.dumps(
            expected,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    )
    return "\n".join(lines)


def build_solution_receipt(
    objective: str,
    executed: tuple[str, dict[str, Any], dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    """Render a candidate and its public receipt from an executed objective.

    Shared with the neural producer in neural_objective_producer.py so the two
    solvers cannot drift into emitting different receipt shapes for the same
    objective. The execution record is whatever the caller's engine produced;
    everything else here is derived from the public objective.
    """

    family, expected, execution = executed
    candidate = _render_solution_witness(
        objective,
        family=family,
        expected=expected,
    )
    payload = {
        "schema": OBJECTIVE_PROGRAM_SOLUTION_SCHEMA,
        "family": family,
        "objective_sha256": _text_sha(objective),
        "candidate_sha256": _text_sha(candidate),
        "expected_payload_sha256": _sha(expected),
        "execution": execution,
        "authority": "public_objective_deterministic_execution",
    }
    return candidate, {**payload, "receipt_sha256": _sha(payload)}


def solve_objective_program(objective: str) -> tuple[str, dict[str, Any]] | None:
    """Compile a recognized public objective into a canonical proven answer.

    This is a bounded symbolic solver, not benchmark-answer access: the only
    input is the same user-visible objective given to the generator. The public
    receipt commits to the output and execution trace without embedding answer
    text; callers must still run the independent verifier before promotion.

    DETERMINISTIC BY CONSTRUCTION. Until 2026-08-10 this module also ran the
    learned neural tissue to compute the expected value — so the grader derived
    its answer from the same tissue that produced the answer it was grading.
    A crosscheck against the independent parser kept that sound, which is the
    tell: the parser was the authority, and the neural run added an identity
    violation without adding evidence. The neural producer now lives in
    neural_objective_producer.py, outside the critic's source closure.
    """

    executed = _execute_objective(objective)
    if executed is None:
        return None
    return build_solution_receipt(objective, executed)


def validate_objective_program_solution(
    value: Any,
    *,
    objective: str,
    candidate: str,
) -> dict[str, Any]:
    rebuilt = solve_objective_program(objective)
    if rebuilt is None:
        raise ValueError("objective program solution is unavailable")
    expected_candidate, expected_receipt = rebuilt
    if candidate != expected_candidate or value != expected_receipt:
        raise ValueError("objective program solution reconstruction differs")
    return expected_receipt


def verify_objective_program(candidate: str, *, objective: str) -> dict[str, Any] | None:
    """Return an exact verdict for a recognized public objective, else ``None``."""

    if not isinstance(candidate, str) or not isinstance(objective, str):
        raise TypeError("objective program verifier inputs must be text")
    executed = _execute_objective(objective)
    if executed is None:
        return None
    if "FINAL_ANSWER:" not in candidate:
        bounded = candidate.strip()
        if not (
            (bounded.startswith("{") and bounded.endswith("}"))
            or _JSON_FENCE_RE.fullmatch(bounded) is not None
        ):
            return None
    family, expected, execution = executed
    failure_codes: list[str] = []
    try:
        produced = _candidate_payload(candidate)
    except (TypeError, ValueError) as exc:
        produced = None
        failure_codes.append(f"candidate_contract:{type(exc).__name__}")
    if produced is not None and produced != expected:
        failure_codes.append("objective_result_mismatch")
    payload = {
        "schema": OBJECTIVE_PROGRAM_VERIFIER_SCHEMA,
        "family": family,
        "outcome": "refuted" if failure_codes else "verified",
        "objective_sha256": _text_sha(objective),
        "candidate_sha256": _text_sha(candidate),
        "expected_payload_sha256": _sha(expected),
        "produced_payload_sha256": _sha(produced),
        "execution": execution,
        "failure_codes": failure_codes,
    }
    return {**payload, "receipt_sha256": _sha(payload)}


__all__ = [
    "OBJECTIVE_PROGRAM_SOLUTION_SCHEMA",
    "OBJECTIVE_PROGRAM_VERIFIER_SCHEMA",
    "solve_objective_program",
    "validate_objective_program_solution",
    "verify_objective_program",
]
