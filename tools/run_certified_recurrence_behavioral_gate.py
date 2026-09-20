#!/usr/bin/env python3
"""Frozen 1.5B behavioral gate for certified recurrent execution.

The public preregistration is written before model load and contains no answer
values. Ordinary decoding, verifier-selected best-of-N, certified recurrence,
and a one-transition recurrence lesion see the same public prompts. This gate
measures a bounded architecture gain; because the executable transition organ
remains present, it cannot establish neural-only RLC or frontier-level gain.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Final

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.brain.llm.latent_cortex.objective_program_verifier import (  # noqa: E402
    solve_objective_program,
    verify_objective_program,
)
from core.brain.llm.latent_cortex.recurrence_adapter_identity_v2 import (  # noqa: E402
    full_weight_checkpoint_identity,
    model_behavior_bundle_identity,
)
from core.brain.llm.latent_cortex.typed_action_compiler import (  # noqa: E402
    TypedActionProgram,
    compile_public_transition_program,
)
from core.brain.llm.latent_cortex.typed_transition_executor import (  # noqa: E402
    CertifiedTransitionExecutor,
    TypedTransitionInput,
)
from core.learning.recurrence_curriculum import modular_chain, nested_boolean  # noqa: E402
from core.runtime.model_lane_control import standalone_model_lane  # noqa: E402

SCHEMA: Final = "aura.certified_recurrence_behavioral_gate.v1"
PREREG_SCHEMA: Final = "aura.certified_recurrence_behavioral_preregistration.v1"
#: Resolved from the repository root rather than hard-coded to one machine's
#: home directory. The absolute form made this gate runnable only on the
#: developer's laptop, and a certification gate that cannot run elsewhere
#: certifies nothing anyone else can reproduce.
DEFAULT_MODEL: Final = str(
    Path(os.environ.get("AURA_RECURRENCE_GATE_MODEL", ""))
    if os.environ.get("AURA_RECURRENCE_GATE_MODEL")
    else REPO_ROOT / "models" / "Qwen2.5-1.5B-Instruct-4bit"
)
GENERATORS: Final = {
    "boolean": nested_boolean,
    "modular": modular_chain,
}
_TASK_ID = re.compile(r"\Arecurrence-(boolean|modular)-d([1-9][0-9]*)-s([0-9]+)\Z")
_SOURCE_COMMIT = re.compile(r"\A[0-9a-f]{40}\Z")
_CURRICULUM_SOURCE = "core/learning/recurrence_curriculum.py"


def _sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _text_sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-c", "core.fsmonitor=false", *args],
        cwd=repo,
        text=True,
        stderr=subprocess.STDOUT,
    ).strip()


def _curriculum_version_at_source_commit(source_commit: str) -> str:
    """Read the task RNG identity from the exact source committed by the campaign."""

    if _SOURCE_COMMIT.fullmatch(source_commit) is None:
        raise RuntimeError("behavioral source commit identity is invalid")
    try:
        resolved = _git(REPO_ROOT, "rev-parse", "--verify", f"{source_commit}^{{commit}}")
        source = _git(REPO_ROOT, "show", f"{source_commit}:{_CURRICULUM_SOURCE}")
    except subprocess.CalledProcessError as exc:
        raise RuntimeError("behavioral source commit is unavailable") from exc
    if resolved != source_commit:
        raise RuntimeError("behavioral source commit resolution differs")
    try:
        module = ast.parse(source, filename=f"{source_commit}:{_CURRICULUM_SOURCE}")
    except SyntaxError as exc:
        raise RuntimeError("behavioral curriculum source is invalid") from exc
    versions = []
    for node in module.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
        if not any(isinstance(target, ast.Name) and target.id == "CURRICULUM_VERSION" for target in targets):
            continue
        value = node.value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            versions.append(value.value)
    if len(versions) != 1 or not versions[0] or versions[0] != versions[0].strip():
        raise RuntimeError("behavioral curriculum version identity is invalid")
    return versions[0]


def build_tasks(
    *,
    depths: tuple[int, ...],
    seeds: tuple[int, ...],
) -> tuple[Any, ...]:
    if (
        not depths
        or not seeds
        or len(set(depths)) != len(depths)
        or len(set(seeds)) != len(seeds)
        or any(type(depth) is not int or not 1 <= depth <= 32 for depth in depths)
        or any(type(seed) is not int or seed < 0 for seed in seeds)
    ):
        raise ValueError("behavioral gate coordinates are invalid")
    return tuple(
        generator(depth, seed)
        for generator in GENERATORS.values()
        for depth in depths
        for seed in seeds
    )


def build_public_preregistration(
    tasks: tuple[Any, ...],
    *,
    source_commit: str,
    model_path: Path,
    best_of_n: int,
    max_tokens: int,
) -> dict[str, Any]:
    if (
        not tasks
        or not source_commit
        or type(best_of_n) is not int
        or best_of_n < 1
        or type(max_tokens) is not int
        or max_tokens < 16
    ):
        raise ValueError("behavioral preregistration inputs are invalid")
    rows = [
        {
            "task_id": task.task_id,
            "family": task.family,
            "depth": task.depth,
            "public_prompt_sha256": _text_sha(task.prompt),
        }
        for task in tasks
    ]
    body = {
        "schema": PREREG_SCHEMA,
        "source_commit": source_commit,
        "model_path": str(model_path),
        "task_count": len(rows),
        "tasks": rows,
        "arms": [
            "ordinary_greedy",
            f"ordinary_best_of_{best_of_n}_public_verifier",
            "certified_recurrence",
            "recurrence_lesion_t1",
        ],
        "best_of_n": best_of_n,
        "max_tokens": max_tokens,
        "primary_contrast": "certified_recurrence_minus_ordinary_greedy",
        "equal_compute_contrast": (
            f"certified_recurrence_minus_ordinary_best_of_{best_of_n}_public_verifier"
        ),
        "claim_boundary": (
            "bounded architecture gain with executable producer present; not neural-only "
            "RLC, frontier intelligence, or resident-32B evidence"
        ),
    }
    return {**body, "preregistration_sha256": _sha(body)}


def _render_terminal(program: TypedActionProgram, state: tuple[int, ...]) -> str:
    key = "value" if program.family == "boolean" else "residue"
    return "FINAL_ANSWER: " + json.dumps(
        {key: state[1]},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def execute_certified_arm(prompt: str) -> tuple[str, dict[str, Any]]:
    solved = solve_objective_program(prompt)
    if solved is None:
        raise RuntimeError("certified recurrence is unavailable for a declared task")
    candidate, receipt = solved
    execution = receipt.get("execution", {})
    if execution.get("engine") != "certified_typed_recurrence.v1":
        raise RuntimeError("objective producer did not use certified recurrence")
    verdict = verify_objective_program(candidate, objective=prompt)
    if verdict is None or verdict.get("outcome") != "verified":
        raise RuntimeError("certified recurrent candidate failed independent verification")
    return candidate, receipt


def execute_t1_lesion(prompt: str) -> tuple[str, dict[str, Any]]:
    program = compile_public_transition_program(prompt)
    result = CertifiedTransitionExecutor().execute(
        TypedTransitionInput(
            family=program.family,
            depth=program.depth,
            field_names=program.field_names,
            state=program.initial_state,
            action_field_names=program.action_field_names,
            action=program.actions[0],
        )
    )
    candidate = _render_terminal(program, result.next_state)
    body = {
        "schema": "aura.certified_recurrence_t1_lesion.v1",
        "program_sha256": program.program_sha256,
        "full_depth": program.depth,
        "executed_transitions": 1,
        "transition_receipt_sha256": result.receipt()["receipt_sha256"],
        "candidate_sha256": _text_sha(candidate),
    }
    return candidate, {**body, "receipt_sha256": _sha(body)}


def exact_paired_pvalue(
    treatment: tuple[bool, ...],
    control: tuple[bool, ...],
) -> dict[str, Any]:
    if not treatment or len(treatment) != len(control):
        raise ValueError("paired outcomes are invalid")
    wins = sum(t and not c for t, c in zip(treatment, control, strict=True))
    losses = sum(c and not t for t, c in zip(treatment, control, strict=True))
    discordant = wins + losses
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(
            math.comb(discordant, index)
            for index in range(0, min(wins, losses) + 1)
        ) / (2**discordant)
        p_value = min(1.0, 2.0 * tail)
    return {
        "treatment_only_correct": wins,
        "control_only_correct": losses,
        "discordant": discordant,
        "two_sided_exact_p": p_value,
    }


def _generate(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    max_tokens: int,
    temperature: float,
    seed: int,
) -> str:
    import mlx.core as mx
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler

    from core.brain.llm.latent_cortex.answer_contract import is_contract_complete

    mx.random.seed(seed & 0xFFFFFFFF)
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )
    pieces: list[str] = []
    for response in stream_generate(
        model,
        tokenizer,
        prompt=rendered,
        max_tokens=max_tokens,
        sampler=make_sampler(temp=temperature, top_p=0.95),
    ):
        pieces.append(response.text)
        candidate = "".join(pieces)
        if "}" in response.text and is_contract_complete(candidate):
            break
    return "".join(pieces).strip()


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    encoded = json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def _load_journal(
    path: Path,
    *,
    tasks: tuple[Any, ...],
    allowed_arms: tuple[str, ...],
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    if not path.exists():
        return [], {}
    task_map = {task.task_id: task for task in tasks}
    rows: list[dict[str, Any]] = []
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"journal line {line_number} is invalid JSON") from exc
        if not isinstance(row, dict):
            raise RuntimeError(f"journal line {line_number} is not an object")
        task = task_map.get(row.get("task_id"))
        arm = row.get("arm")
        candidate = row.get("candidate")
        key = (str(row.get("task_id") or ""), str(arm or ""))
        if (
            row.get("schema")
            != "aura.certified_recurrence_behavioral_observation.v1"
            or task is None
            or arm not in allowed_arms
            or key in indexed
            or row.get("family") != task.family
            or row.get("depth") != task.depth
            or not isinstance(candidate, str)
            or row.get("candidate_sha256") != _text_sha(candidate)
            or row.get("correct") is not bool(task.grade(candidate)["correct"])
        ):
            raise RuntimeError(f"journal line {line_number} failed reconstruction")
        rows.append(row)
        indexed[key] = row
    return rows, indexed


def _select_publicly_verified(candidates: list[str], *, prompt: str) -> tuple[str, int | None]:
    for index, candidate in enumerate(candidates):
        verdict = verify_objective_program(candidate, objective=prompt)
        if verdict is not None and verdict.get("outcome") == "verified":
            return candidate, index
    return candidates[0], None


def _score_row(task: Any, *, arm: str, candidate: str, detail: dict[str, Any]) -> dict[str, Any]:
    grade = task.grade(candidate)
    return {
        "schema": "aura.certified_recurrence_behavioral_observation.v1",
        "task_id": task.task_id,
        "family": task.family,
        "depth": task.depth,
        "arm": arm,
        "candidate": candidate,
        "candidate_sha256": _text_sha(candidate),
        "correct": bool(grade["correct"]),
        "parsed": grade.get("parsed"),
        "detail": detail,
    }


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    arms = sorted({row["arm"] for row in rows})
    task_ids = sorted({row["task_id"] for row in rows})
    matrix = {
        arm: {
            row["task_id"]: bool(row["correct"])
            for row in rows
            if row["arm"] == arm
        }
        for arm in arms
    }
    if any(set(values) != set(task_ids) for values in matrix.values()):
        raise RuntimeError("behavioral gate result matrix is incomplete")
    treatment = tuple(matrix["certified_recurrence"][task_id] for task_id in task_ids)
    contrasts = {}
    for arm in arms:
        if arm == "certified_recurrence":
            continue
        contrasts[arm] = exact_paired_pvalue(
            treatment,
            tuple(matrix[arm][task_id] for task_id in task_ids),
        )
    accuracy = {
        arm: sum(values.values()) / len(task_ids) for arm, values in matrix.items()
    }
    greedy = contrasts["ordinary_greedy"]
    best_arm = next(arm for arm in arms if arm.startswith("ordinary_best_of_"))
    equal_compute = contrasts[best_arm]
    return {
        "task_count": len(task_ids),
        "accuracy": accuracy,
        "contrasts": contrasts,
        "positive_architecture_gain": bool(
            accuracy["certified_recurrence"] > accuracy["ordinary_greedy"]
            and greedy["two_sided_exact_p"] < 0.05
        ),
        "positive_equal_compute_gain": bool(
            accuracy["certified_recurrence"] > accuracy[best_arm]
            and equal_compute["two_sided_exact_p"] < 0.05
        ),
        "wow_signal": False,
        "wow_signal_reason": "executable_producer_present_and_neural_only_gate_not_run",
    }


def verify_behavioral_bundle(
    bundle_dir: Path,
    *,
    verify_model_identity: bool = True,
) -> dict[str, Any]:
    bundle = bundle_dir.expanduser().resolve(strict=True)
    prereg = json.loads((bundle / "preregistration.json").read_text(encoding="utf-8"))
    report = json.loads((bundle / "report.json").read_text(encoding="utf-8"))
    if not isinstance(prereg, dict) or not isinstance(report, dict):
        raise RuntimeError("behavioral bundle records must be objects")
    prereg_body = {key: value for key, value in prereg.items() if key != "preregistration_sha256"}
    report_body = {key: value for key, value in report.items() if key != "report_sha256"}
    if (
        prereg.get("schema") != PREREG_SCHEMA
        or prereg.get("preregistration_sha256") != _sha(prereg_body)
        or report.get("schema") != SCHEMA
        or report.get("report_sha256") != _sha(report_body)
        or report.get("preregistration_sha256") != prereg.get("preregistration_sha256")
        or report.get("source_commit") != prereg.get("source_commit")
        or report.get("claim_boundary") != prereg.get("claim_boundary")
    ):
        raise RuntimeError("behavioral bundle commitments differ")
    source_commit = str(prereg.get("source_commit") or "")
    curriculum_version = _curriculum_version_at_source_commit(source_commit)
    tasks = []
    for row in prereg.get("tasks", []):
        if not isinstance(row, dict):
            raise RuntimeError("behavioral task commitment is invalid")
        match = _TASK_ID.fullmatch(str(row.get("task_id") or ""))
        if match is None:
            raise RuntimeError("behavioral task id is invalid")
        family, depth_text, seed_text = match.groups()
        task = GENERATORS[family](
            int(depth_text),
            int(seed_text),
            curriculum_version=curriculum_version,
        )
        if (
            row
            != {
                "task_id": task.task_id,
                "family": task.family,
                "depth": task.depth,
                "public_prompt_sha256": _text_sha(task.prompt),
            }
        ):
            raise RuntimeError("behavioral task commitment reconstruction differs")
        tasks.append(task)
    if len(tasks) != prereg.get("task_count") or len({task.task_id for task in tasks}) != len(
        tasks
    ):
        raise RuntimeError("behavioral task population differs")
    observations_path = bundle / "observations.jsonl"
    observations_text = observations_path.read_text(encoding="utf-8")
    if report.get("observations_sha256") != _text_sha(observations_text):
        raise RuntimeError("behavioral observation journal commitment differs")
    rows, indexed = _load_journal(
        observations_path,
        tasks=tuple(tasks),
        allowed_arms=tuple(prereg["arms"]),
    )
    if len(indexed) != len(tasks) * len(prereg["arms"]):
        raise RuntimeError("behavioral observation matrix is incomplete")
    summary = _summarize(rows)
    if summary != report.get("summary"):
        raise RuntimeError("behavioral summary reconstruction differs")
    if verify_model_identity:
        model_path = Path(prereg["model_path"]).expanduser().resolve(strict=True)
        if (
            full_weight_checkpoint_identity(model_path) != report.get("model_identity")
            or model_behavior_bundle_identity(model_path)
            != report.get("model_behavior_identity")
        ):
            raise RuntimeError("behavioral model identity differs")
    return {
        "verified": True,
        "task_count": len(tasks),
        "observation_count": len(rows),
        "source_commit": report["source_commit"],
        "report_sha256": report["report_sha256"],
        "summary": summary,
    }


def _run_admitted(args: argparse.Namespace) -> dict[str, Any]:
    repo = Path(args.repo).expanduser().resolve(strict=True)
    model_path = Path(args.model).expanduser().resolve(strict=True)
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if _git(repo, "status", "--porcelain"):
        raise RuntimeError("behavioral gate source tree must be clean")
    source_commit = _git(repo, "rev-parse", "HEAD")
    tasks = build_tasks(depths=tuple(args.depth), seeds=tuple(args.seed))
    prereg = build_public_preregistration(
        tasks,
        source_commit=source_commit,
        model_path=model_path,
        best_of_n=args.best_of_n,
        max_tokens=args.max_tokens,
    )
    prereg_path = out_dir / "preregistration.json"
    if prereg_path.exists():
        if json.loads(prereg_path.read_text(encoding="utf-8")) != prereg:
            raise RuntimeError("existing preregistration differs")
    else:
        prereg_path.write_text(
            json.dumps(prereg, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    state_path = out_dir / "campaign_state.json"
    state = {
        "schema": "aura.certified_recurrence_behavioral_campaign_state.v1",
        "source_commit": source_commit,
        "preregistration_sha256": prereg["preregistration_sha256"],
        "started_at": time.time(),
    }
    if state_path.exists():
        existing_state = json.loads(state_path.read_text(encoding="utf-8"))
        if (
            not isinstance(existing_state, dict)
            or existing_state.get("schema") != state["schema"]
            or existing_state.get("source_commit") != source_commit
            or existing_state.get("preregistration_sha256")
            != prereg["preregistration_sha256"]
            or not isinstance(existing_state.get("started_at"), (int, float))
        ):
            raise RuntimeError("existing campaign state differs")
        state = existing_state
    else:
        state_path.write_text(
            json.dumps(state, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    model_identity = full_weight_checkpoint_identity(model_path)
    behavior_identity = model_behavior_bundle_identity(model_path)
    journal = out_dir / "observations.jsonl"
    allowed_arms = tuple(prereg["arms"])
    rows, completed = _load_journal(
        journal,
        tasks=tasks,
        allowed_arms=allowed_arms,
    )
    model = None
    tokenizer = None

    def ensure_model() -> tuple[Any, Any]:
        nonlocal model, tokenizer
        if model is None or tokenizer is None:
            from mlx_lm import load

            model, tokenizer = load(str(model_path))
        return model, tokenizer

    def publish(row: dict[str, Any]) -> None:
        key = (row["task_id"], row["arm"])
        if key in completed:
            raise RuntimeError("attempted to duplicate a completed observation")
        _append_jsonl(journal, row)
        rows.append(row)
        completed[key] = row

    for task_index, task in enumerate(tasks):
        greedy_key = (task.task_id, "ordinary_greedy")
        if greedy_key not in completed:
            loaded_model, loaded_tokenizer = ensure_model()
            greedy = _generate(
                loaded_model,
                loaded_tokenizer,
                task.prompt,
                max_tokens=args.max_tokens,
                temperature=0.0,
                seed=args.campaign_seed + task_index * 100,
            )
            publish(
                _score_row(
                    task,
                    arm="ordinary_greedy",
                    candidate=greedy,
                    detail={"model_calls": 1, "temperature": 0.0},
                )
            )

        best_arm = f"ordinary_best_of_{args.best_of_n}_public_verifier"
        if (task.task_id, best_arm) not in completed:
            loaded_model, loaded_tokenizer = ensure_model()
            samples = [
                _generate(
                    loaded_model,
                    loaded_tokenizer,
                    task.prompt,
                    max_tokens=args.max_tokens,
                    temperature=0.7,
                    seed=args.campaign_seed + task_index * 100 + sample + 1,
                )
                for sample in range(args.best_of_n)
            ]
            selected, selected_index = _select_publicly_verified(
                samples,
                prompt=task.prompt,
            )
            publish(
                _score_row(
                    task,
                    arm=best_arm,
                    candidate=selected,
                    detail={
                        "model_calls": args.best_of_n,
                        "temperature": 0.7,
                        "selected_verified_index": selected_index,
                        "sample_sha256s": [_text_sha(sample) for sample in samples],
                    },
                )
            )

        if (task.task_id, "certified_recurrence") not in completed:
            candidate, receipt = execute_certified_arm(task.prompt)
            publish(
                _score_row(
                    task,
                    arm="certified_recurrence",
                    candidate=candidate,
                    detail={
                        "model_calls": 0,
                        "producer_receipt_sha256": receipt["receipt_sha256"],
                        "execution": receipt["execution"],
                    },
                )
            )

        if (task.task_id, "recurrence_lesion_t1") not in completed:
            candidate, receipt = execute_t1_lesion(task.prompt)
            publish(
                _score_row(
                    task,
                    arm="recurrence_lesion_t1",
                    candidate=candidate,
                    detail={"model_calls": 0, "lesion": receipt},
                )
            )
        heartbeat = {
            "schema": "aura.certified_recurrence_behavioral_heartbeat.v1",
            "task_index": task_index + 1,
            "task_count": len(tasks),
            "updated_at": time.time(),
        }
        (out_dir / "heartbeat.json").write_text(
            json.dumps(heartbeat, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    summary = _summarize(rows)
    body = {
        "schema": SCHEMA,
        "source_commit": source_commit,
        "preregistration_sha256": prereg["preregistration_sha256"],
        "model_identity": model_identity,
        "model_behavior_identity": behavior_identity,
        "started_at": state["started_at"],
        "completed_at": time.time(),
        "summary": summary,
        "observations_sha256": _text_sha(journal.read_text(encoding="utf-8")),
        "claim_boundary": prereg["claim_boundary"],
    }
    report = {**body, "report_sha256": _sha(body)}
    (out_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def run(args: argparse.Namespace) -> dict[str, Any]:
    model_path = Path(args.model).expanduser().resolve(strict=True)
    with standalone_model_lane(
        owner_id=f"certified-recurrence-behavioral:{Path(args.out).name}",
        model_path=str(model_path),
        purpose="evaluation",
        preemptible=False,
        allow_owner_eviction=False,
        metadata={"tool": Path(__file__).name},
    ):
        return _run_admitted(args)


def _parse_csv_ints(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not parsed:
        raise argparse.ArgumentTypeError("integer list is empty")
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=str(REPO_ROOT))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--out", required=True)
    parser.add_argument("--depth", type=_parse_csv_ints, default=(1, 2, 4, 8, 16, 32))
    parser.add_argument("--seed", type=_parse_csv_ints, default=(193001, 193002, 193003, 193004))
    parser.add_argument("--best-of-n", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=192)
    parser.add_argument("--campaign-seed", type=int, default=20260810193)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
