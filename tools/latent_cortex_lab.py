#!/usr/bin/env python
"""Latent Cortex Lab — run the falsification experiments on a real checkpoint.

Operator-launched, bounded, and honest by construction: every run prints the
checkpoint fingerprint, the graded claims, and writes the full JSON report.

Usage (run with the repo venv python; bound long runs with caffeinate):

  caffeinate -dims .venv/bin/python tools/latent_cortex_lab.py \\
      --model ~/.cache/huggingface/hub/models--mlx-community--Qwen2.5-1.5B-Instruct-4bit/snapshots/<hash> \\
      --experiments 1,2,3 --per-cell 8 --max-minutes 30

MEMORY SAFETY: never point this at a second 32B while the live instance is
up. The 1.5B/7B checkpoints are the offline lab vehicles; the resident 32B
runs episodes through the worker action instead.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import NoReturn

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("AURA_LOG_DIR", str(Path.home() / ".aura" / "lab-logs"))


class LabDeadlineError(RuntimeError):
    """Raised between bounded model operations so partial runs never get graded."""


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return parsed


def _parse_positive_csv(parser: argparse.ArgumentParser, raw: str, name: str) -> list[int]:
    try:
        parsed = [int(value.strip()) for value in raw.split(",") if value.strip()]
    except ValueError:
        parser.error(f"{name} must be a comma-separated list of integers")
    if not parsed or any(value <= 0 for value in parsed):
        parser.error(f"{name} must contain positive integers")
    if len(set(parsed)) != len(parsed):
        parser.error(f"{name} must not contain duplicates")
    return parsed


def _parser_error(parser: argparse.ArgumentParser, message: str) -> NoReturn:
    parser.error(message)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="mlx model directory")
    parser.add_argument(
        "--experiments", default="1,2", help="comma list drawn from 1,2,3,5"
    )
    parser.add_argument("--per-cell", type=_positive_int, default=8)
    parser.add_argument("--depths", default="2,4,8")
    parser.add_argument("--steps", default="1,2,4,8")
    parser.add_argument("--families", default="khop,boolean,modular")
    parser.add_argument("--n-slots", type=_positive_int, default=16)
    parser.add_argument("--branches", type=_positive_int, default=2)
    parser.add_argument("--max-minutes", type=_positive_float, default=30.0)
    parser.add_argument("--out", default="")
    parser.add_argument(
        "--task-seed",
        type=_positive_int,
        default=11,
        help="task-battery seed; preregister a FRESH value for campaign runs",
    )
    parser.add_argument(
        "--adapter",
        default="",
        help=(
            "recurrence-native adapter dir (from tools/recurrence_native_train.py); "
            "wraps the window projections and loads the trained LoRA before running"
        ),
    )
    parser.add_argument("--record-foundry", action="store_true")
    parser.add_argument(
        "--vanilla-baseline",
        action="store_true",
        help="run the ordinary-decoding control arm alongside Experiment 1",
    )
    args = parser.parse_args()

    from core.runtime.model_lane_control import standalone_model_lane

    with standalone_model_lane(
        owner_id=f"latent-cortex-lab:{os.getpid()}",
        model_path=args.model,
        purpose="benchmark",
        preemptible=False,
        metadata={"tool": "latent_cortex_lab", "operator_launched": True},
    ):
        return _run_admitted_lab(args, parser)


def _load_recurrence_adapter(model, adapter_dir: Path) -> dict:
    """Wrap window projections exactly as training did and load the LoRA.

    The adapter's receipt.json is the authority for rank/targets; the
    checkpoint fingerprint recorded at training time is returned so the
    report can prove the adapter belongs to this model.
    """
    import mlx.core as mx
    from mlx_lm.tuner.lora import LoRALinear

    receipt = json.loads((adapter_dir / "receipt.json").read_text(encoding="utf-8"))
    lora = receipt.get("lora") or {}
    rank = int(lora.get("rank") or 8)
    targets = tuple(lora.get("targets") or ("o_proj", "v_proj"))
    inner = model.model
    n_layers = len(inner.layers)
    prelude_end = max(1, int(n_layers * 0.25))
    coda_start = min(n_layers - 1, n_layers - int(n_layers * 0.25))
    wrapped = 0
    for layer in inner.layers[prelude_end:coda_start]:
        for target in targets:
            parent = (
                layer.self_attn
                if hasattr(layer.self_attn, target)
                else layer.mlp
                if hasattr(layer.mlp, target)
                else None
            )
            if parent is None:
                continue
            setattr(parent, target, LoRALinear.from_base(getattr(parent, target), r=rank))
            wrapped += 1
    weights_path = adapter_dir / "adapter_final.safetensors"
    if not weights_path.is_file():
        weights_path = adapter_dir / "adapter_latest.safetensors"
    flat = mx.load(str(weights_path))
    model.load_weights(list(flat.items()), strict=False)
    mx.eval(model.parameters())
    return {
        "adapter_dir": str(adapter_dir),
        "weights_file": weights_path.name,
        "rank": rank,
        "targets": list(targets),
        "wrapped_projections": wrapped,
        "trained_steps": receipt.get("steps"),
        "train_seed": receipt.get("train_seed"),
        "trained_on_checkpoint": (receipt.get("checkpoint") or {}).get("fingerprint"),
        "objective_schema": receipt.get("objective_schema"),
    }


def _run_admitted_lab(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> int:

    from core.brain.llm.latent_cortex.engine import LatentCortexEngine
    from core.brain.llm.latent_cortex.experiments import (
        TASK_FAMILIES,
        extract_final_numeric_claim,
        majority_answer,
        record_claim_to_foundry,
        run_depth_extrapolation,
        run_factorial_ablations,
        run_latent_opt_control,
        run_recurrence_sweep,
        run_role_lesion,
        run_slot_causality,
        run_virtual_width,
        task_battery,
    )
    from core.brain.llm.latent_cortex.governance import checkpoint_file_fingerprint
    from core.brain.llm.latent_cortex.resource_accounting import (
        ModelComputeProfile,
        ResourceLedger,
        triangular_attention_pairs,
    )
    from core.brain.llm.latent_cortex.types import (
        ABSOLUTE_MAX_BRANCHES,
        ABSOLUTE_MAX_SLOTS,
        BranchConfig,
        ComputeBudget,
        CortexConfig,
        LatentOptConfig,
        RecurrenceConfig,
        WorkspaceConfig,
    )
    from core.brain.llm.latent_cortex.value_of_computation import (
        build_evidence_snapshot,
    )

    supported_experiments = {"1", "2", "3", "4", "5", "A", "R"}
    wanted = {value.strip() for value in args.experiments.split(",") if value.strip()}
    if not wanted:
        _parser_error(parser, "--experiments cannot be empty")
    unknown = sorted(wanted - supported_experiments)
    if unknown:
        _parser_error(
            parser,
            "unsupported experiment(s): "
            + ",".join(unknown)
            + "; supported: 1,2,3,4,5, A (factorial mechanism ablations), "
            "and R (role lesion/swap causality)",
        )
    if args.n_slots > ABSOLUTE_MAX_SLOTS:
        _parser_error(parser, f"--n-slots cannot exceed {ABSOLUTE_MAX_SLOTS}")
    if args.branches > ABSOLUTE_MAX_BRANCHES:
        _parser_error(parser, f"--branches cannot exceed {ABSOLUTE_MAX_BRANCHES}")
    depths = _parse_positive_csv(parser, args.depths, "--depths")
    steps = _parse_positive_csv(parser, args.steps, "--steps")
    if steps != sorted(steps):
        _parser_error(parser, "--steps must be sorted in increasing order")
    families = [family.strip() for family in args.families.split(",") if family.strip()]
    unknown_families = sorted(set(families) - set(TASK_FAMILIES))
    if not families or unknown_families:
        _parser_error(
            parser,
            "--families contains unsupported values: " + ",".join(unknown_families),
        )

    from mlx_lm import load

    deadline = time.monotonic() + args.max_minutes * 60.0
    model, tokenizer = load(args.model)
    checkpoint_receipt = checkpoint_file_fingerprint(args.model)
    adapter_receipt: dict | None = None
    if args.adapter:
        adapter_dir = Path(args.adapter).expanduser()
        adapter_receipt = _load_recurrence_adapter(model, adapter_dir)
        print(
            "recurrence-native adapter loaded: "
            f"{adapter_receipt['wrapped_projections']} projections, "
            f"rank {adapter_receipt['rank']}, "
            f"trained_steps {adapter_receipt.get('trained_steps')}",
            flush=True,
        )

    def make_engine(
        max_steps: int,
        *,
        latent_opt: str = "off",
        branches: int | None = None,
        fast_weights: bool = False,
        roles: tuple[str, ...] = (),
        exchange_interval: int | None = None,
    ):
        from core.brain.llm.latent_cortex.types import FastWeightsConfig

        branch_kwargs: dict = {"n_branches": branches or args.branches}
        if roles:
            branch_kwargs["roles"] = roles
        if exchange_interval is not None:
            branch_kwargs["exchange_interval"] = exchange_interval
        return LatentCortexEngine(
            model,
            tokenizer,
            CortexConfig(
                workspace=WorkspaceConfig(n_slots=args.n_slots, seed=7),
                recurrence=RecurrenceConfig(
                    max_steps=max_steps, min_steps=max_steps, convergence_eps=1e-9
                ),
                branches=BranchConfig(**branch_kwargs),
                latent_opt=LatentOptConfig(
                    enabled=latent_opt != "off",
                    control_mode=latent_opt == "control",
                    steps=4,
                ),
                fast_weights=FastWeightsConfig(enabled=fast_weights),
                decode_max_tokens=64,
            ),
            model_path=args.model,
        )

    def out_of_time() -> bool:
        if time.monotonic() > deadline:
            print("wall-clock bound reached — reporting what completed", flush=True)
            return True
        return False

    def solve(
        task,
        n_steps: int,
        *,
        latent_opt: str = "off",
        ablate=None,
        branches: int | None = None,
        fast_weights: bool = False,
        verifier_guided: bool = False,
        return_accounting: bool = False,
    ):
        remaining_s = deadline - time.monotonic()
        if remaining_s <= 0.0:
            raise LabDeadlineError("lab wall-clock bound reached")
        engine = make_engine(
            n_steps,
            latent_opt=latent_opt,
            branches=branches,
            fast_weights=fast_weights,
        )
        budget = ComputeBudget(wall_clock_s=min(120.0, remaining_s))
        verifier = None
        if verifier_guided:
            from core.brain.llm.latent_cortex.task_verifiers import (
                EpisodeTaskVerifier,
            )

            verifier = EpisodeTaskVerifier(task.prompt)
        # Chat-template parity with the vanilla control arm: an instruct
        # checkpoint answers through its template; comparing a template-free
        # latent arm against a templated control confounds the experiment
        # (the first 32B sweep measured exactly that confound).
        result = engine.reason(
            messages=[{"role": "user", "content": task.prompt}],
            budget=budget,
            ablate_slot=ablate,
            verifier=verifier,
            decode_max_tokens=64,
        )
        if time.monotonic() > deadline:
            raise LabDeadlineError("lab wall-clock bound reached during episode")
        outcome = (result.ok and task.verify(result.text), budget.spent_layer_apps)
        if not return_accounting:
            return outcome
        return (
            *outcome,
            result.receipt.budget["resource_accounting"],
            result.receipt.budget["information_accounting"],
        )

    def solve_vanilla(task) -> tuple[bool, int]:
        """The strong control: the same checkpoint, ordinary decoding, no
        latent machinery. Compute is charged identically (token-layer apps)
        so the sweep's equal-compute comparison stays honest."""
        remaining_s = deadline - time.monotonic()
        if remaining_s <= 0.0:
            raise LabDeadlineError("lab wall-clock bound reached")
        from mlx_lm import stream_generate

        information_engine = make_engine(max(steps), branches=1)
        prompt_token_ids = information_engine._encode(
            None,
            [{"role": "user", "content": task.prompt}],
            None,
        )
        pieces: list[str] = []
        generated_tokens = 0
        for response in stream_generate(
            model,
            tokenizer,
            prompt=prompt_token_ids,
            max_tokens=64,
        ):
            pieces.append(response.text)
            generated_tokens = int(response.generation_tokens)
        text = "".join(pieces)
        prompt_tokens = len(prompt_token_ids)
        n_layers = len(model.model.layers)
        # Prefill produces the first output token; only subsequent output
        # tokens execute an additional transformer forward pass.
        cost = (prompt_tokens + max(0, max(1, generated_tokens) - 1)) * n_layers
        return task.verify(text), cost

    def solve_branches(task, k: int):
        """Experiment-4 treatment: K latent branches, convergence-selected."""
        return solve(
            task,
            max(steps),
            branches=k,
            return_accounting=True,
        )

    def solve_sampling(task, k: int):
        """Experiment-4 control: K-sample self-consistency at matched FLOPs.

        Standard recipe — sample K answers at temperature, majority-vote the
        EXTRACTED final claims (extraction never sees the ground truth), then
        verify the winning claim once."""
        remaining_s = deadline - time.monotonic()
        if remaining_s <= 0.0:
            raise LabDeadlineError("lab wall-clock bound reached")
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler

        information_engine = make_engine(max(steps), branches=k)
        prompt_token_ids = information_engine._encode(
            None,
            [{"role": "user", "content": task.prompt}],
            None,
        )
        claims: list[str] = []
        output_token_counts: list[int] = []
        output_byte_counts: list[int] = []
        for _sample_index in range(max(1, int(k))):
            if time.monotonic() > deadline:
                raise LabDeadlineError("lab wall-clock bound reached during sampling")
            pieces: list[str] = []
            generated_tokens = 0
            for response in stream_generate(
                model,
                tokenizer,
                prompt=prompt_token_ids,
                max_tokens=64,
                sampler=make_sampler(temp=0.8, top_p=0.95),
            ):
                pieces.append(response.text)
                generated_tokens = int(response.generation_tokens)
            text = "".join(pieces)
            claims.append(extract_final_numeric_claim(text))
            output_token_counts.append(max(1, generated_tokens))
            output_byte_counts.append(len(text.encode("utf-8", errors="replace")))
        voted = majority_answer(claims)
        prompt_tokens = len(prompt_token_ids)
        n_layers = len(model.model.layers)
        profile = ModelComputeProfile.from_model(model)
        ledger = ResourceLedger(profile)
        for sample_index, output_tokens in enumerate(output_token_counts):
            ledger.charge(
                f"sample_{sample_index}:prefill",
                transformer_layer_apps=prompt_tokens * n_layers,
                attention_query_key_pairs=(
                    triangular_attention_pairs(prompt_tokens) * n_layers
                ),
                output_head_tokens=1,
            )
            decode_forwards = max(0, output_tokens - 1)
            decode_pairs = sum(
                prompt_tokens + index + 1 for index in range(decode_forwards)
            )
            ledger.charge(
                f"sample_{sample_index}:decode",
                transformer_layer_apps=decode_forwards * n_layers,
                attention_query_key_pairs=decode_pairs * n_layers,
                output_head_tokens=decode_forwards,
                tensor_element_reads=output_tokens * profile.vocab_size,
                tensor_element_writes=output_tokens * profile.vocab_size,
                host_scalar_ops=output_tokens * profile.vocab_size * 8,
            )
        ledger.charge(
            "majority_vote",
            host_scalar_ops=max(1, sum(output_byte_counts) + len(claims) * 4),
        )
        cost = sum(
            (prompt_tokens + max(0, output_tokens - 1)) * n_layers
            for output_tokens in output_token_counts
        )
        encoded_tokens = json.dumps(
            prompt_token_ids,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
        information = information_engine._information_receipt(
            encoded_tokens=encoded_tokens,
            token_count=len(prompt_token_ids),
            context_items=[],
            policy_evidence=build_evidence_snapshot(
                bucket="general|none|short|s:mid|u:mid",
                cells={},
            ),
            verifier=None,
        )
        assert information["sources"][0]["content_sha256"] == hashlib.sha256(
            encoded_tokens
        ).hexdigest()
        return bool(voted) and voted == task.answer, cost, ledger.to_receipt(), information

    def solve_role_arm(task, arm: str) -> tuple[bool, int, float | None]:
        """One Experiment-R role, lesion, swap, or restoration arm.

        Returns (success, layer_apps, branch_divergence) where divergence is
        1 − mean pairwise branch-summary cosine across exchange snapshots
        (None when no exchange telemetry was recorded)."""
        from core.brain.llm.latent_cortex.branches import BRANCH_ROLES

        k = max(2, args.branches)
        base_roles = tuple(BRANCH_ROLES[i % len(BRANCH_ROLES)] for i in range(k))
        if arm in {"distinct_roles", "restored_roles"}:
            roles = base_roles
        elif arm == "lesioned_uniform_role":
            roles = (base_roles[0],) * k
        elif arm == "swapped_roles":
            roles = base_roles[1:] + base_roles[:1]
        else:
            raise ValueError(f"unknown role arm: {arm}")
        remaining_s = deadline - time.monotonic()
        if remaining_s <= 0.0:
            raise LabDeadlineError("lab wall-clock bound reached")
        engine = make_engine(
            max(steps), branches=k, roles=roles, exchange_interval=2
        )
        budget = ComputeBudget(wall_clock_s=min(120.0, remaining_s))
        result = engine.reason(
            messages=[{"role": "user", "content": task.prompt}],
            budget=budget,
            decode_max_tokens=64,
        )
        if time.monotonic() > deadline:
            raise LabDeadlineError("lab wall-clock bound reached during episode")
        divergence = None
        snapshots = (result.receipt.latent_telemetry or {}).get(
            "exchange_snapshots"
        ) or []
        cosines = [
            float(row["mean_cos"])
            for row in snapshots
            if isinstance(row, dict)
            and isinstance(row.get("mean_cos"), (int, float))
            and math.isfinite(float(row["mean_cos"]))
        ]
        if cosines:
            divergence = 1.0 - (sum(cosines) / len(cosines))
        return (
            result.ok and task.verify(result.text),
            budget.spent_layer_apps,
            divergence,
        )

    def solve_ablation_arm(task, arm: str) -> tuple[bool, int]:
        """One factorial-ablation arm: exactly one mechanism (or named combo)."""
        deep = max(steps)
        if arm == "vanilla":
            return solve_vanilla(task)
        if arm == "recurrence_only":
            return solve(task, deep, branches=1)
        if arm == "branches_only":
            return solve(task, 1, branches=args.branches)
        if arm == "latent_opt_only":
            return solve(task, 1, branches=1, latent_opt="gradient")
        if arm == "fast_weights_only":
            return solve(task, 1, branches=1, fast_weights=True)
        if arm == "recurrence_branches":
            return solve(task, deep, branches=args.branches)
        if arm == "recurrence_verifier":
            return solve(task, deep, branches=args.branches, verifier_guided=True)
        if arm == "full_stack":
            return solve(
                task,
                deep,
                branches=args.branches,
                latent_opt="gradient",
                fast_weights=True,
                verifier_guided=True,
            )
        raise ValueError(f"unknown ablation arm: {arm}")

    report: dict = {
        "model": args.model,
        "checkpoint": checkpoint_receipt,
        "recurrence_native_adapter": adapter_receipt,
        "frontier_claim_eligible": False,
        "claim_scope": (
            "offline mechanism and scaling evidence only; exact installed-app "
            "resident-32B frontier certification is a separate required run"
        ),
        "started_at": time.time(),
        "settings": vars(args),
        "requested_experiments": sorted(wanted),
        "deadline_exceeded": False,
        "results": {},
    }

    battery = task_battery(families, depths, args.per_cell, seed=args.task_seed)
    try:
        if "1" in wanted and not out_of_time():
            print(f"▶ Experiment 1: recurrence sweep over {len(battery)} tasks …", flush=True)
            report["results"]["exp1"] = run_recurrence_sweep(
                lambda t, s: solve(t, s),
                battery,
                steps,
                baseline=(solve_vanilla if args.vanilla_baseline else None),
            )
            print("  claim:", report["results"]["exp1"]["claim"]["tier"], flush=True)
        if "2" in wanted and not out_of_time():
            report["results"]["exp2"] = {}
            for family in families:
                if out_of_time():
                    raise LabDeadlineError("lab wall-clock bound reached")
                print(f"▶ Experiment 2: depth extrapolation on {family} …", flush=True)
                report["results"]["exp2"][family] = run_depth_extrapolation(
                    lambda t, s: solve(t, s),
                    family,
                    depths,
                    steps,
                    per_depth=args.per_cell,
                )
                print(
                    "  claim:",
                    report["results"]["exp2"][family]["claim"]["tier"],
                    flush=True,
                )
        if "3" in wanted and not out_of_time():
            print("▶ Experiment 3: slot causality …", flush=True)
            report["results"]["exp3"] = run_slot_causality(
                lambda t, slot: solve(t, max(steps), ablate=slot)[0],
                battery,
                slot_indices=list(range(0, args.n_slots, max(1, args.n_slots // 4))),
            )
            print("  claim:", report["results"]["exp3"]["claim"]["tier"], flush=True)
        if "4" in wanted and not out_of_time():
            print(
                f"▶ Experiment 4: {args.branches} branches vs {args.branches}-sample "
                "self-consistency …",
                flush=True,
            )
            by_family = {family: [t for t in battery if t.family == family] for family in families}
            report["results"]["exp4"] = run_virtual_width(
                solve_branches, solve_sampling, by_family, k=args.branches
            )
            print("  claim:", report["results"]["exp4"]["claim"]["tier"], flush=True)
        if "5" in wanted and not out_of_time():
            print("▶ Experiment 5: latent opt vs random control …", flush=True)
            by_family = {family: [t for t in battery if t.family == family] for family in families}
            report["results"]["exp5"] = run_latent_opt_control(
                lambda t, arm: solve(t, max(steps), latent_opt=arm), by_family
            )
            print("  claim:", report["results"]["exp5"]["claim"]["tier"], flush=True)
        if "R" in wanted and not out_of_time():
            print("▶ Experiment R: role lesion/swap causality …", flush=True)
            by_family = {
                family: [t for t in battery if t.family == family]
                for family in families
            }
            report["results"]["expR"] = run_role_lesion(solve_role_arm, by_family)
            print(
                "  behavioral:",
                report["results"]["expR"]["behavioral_claim"]["tier"],
                "| divergence:",
                report["results"]["expR"]["divergence_claim"]["tier"],
                flush=True,
            )
        if "A" in wanted and not out_of_time():
            print("▶ Factorial ablations: mechanism attribution vs vanilla …", flush=True)
            by_family = {family: [t for t in battery if t.family == family] for family in families}
            report["results"]["ablations"] = run_factorial_ablations(
                solve_ablation_arm, by_family
            )
            print(
                "  attribution:",
                report["results"]["ablations"]["attribution"] or "none yet",
                flush=True,
            )
    except LabDeadlineError as exc:
        report["deadline_exceeded"] = True
        report["incomplete_reason"] = str(exc)
        print(f"wall-clock bound reached: {exc}; incomplete experiment discarded", flush=True)

    report["finished_at"] = time.time()
    out_path = Path(args.out) if args.out else REPO_ROOT / "data" / "latent_cortex" / (
        f"lab_report_{int(time.time())}.json"
    )
    from core.brain.llm.latent_cortex.persistence import get_latent_cortex_persistence

    get_latent_cortex_persistence().save_lab_report(
        out_path,
        json.dumps(report, indent=1, sort_keys=True).encode(),
    )
    print(f"report → {out_path}")

    if args.record_foundry:
        for res in report["results"].values():
            if "claim" in res:
                claims = [res["claim"]]
            elif "claims" in res and isinstance(res["claims"], dict):
                claims = list(res["claims"].values())
            else:
                claims = [
                    v["claim"] for v in res.values() if isinstance(v, dict) and "claim" in v
                ]
            for claim in claims:
                record_claim_to_foundry(claim, domain="latent_lab")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
