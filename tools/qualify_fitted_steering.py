#!/usr/bin/env python3
"""Qualify directions fitted to her replies as her signed steering generation.

    seal        write metadata.json for a directory of fitted directions
    evaluate    score the held-out replies on her checkpoint (loads the 27B)
    verify      reopen every vector and recompute every count from the record
    adjudicate  issue the causal evaluation the authority reads

The claim and its test are in core/evaluation/fitted_steering_evaluation.py and
the prereg addendum of 23 October, midday. Then:

    tools/issue_cortex_migration_authority.py --descriptor ... steering \\
        --metadata DIR/metadata.json --causal-evaluation ... --independent-evidence ...
    tools/cortex_generation_upgrade.py requalify-steering --authority ... --authorized-by ...

`evaluate` loads her cortex, about 20 GB: run it with no other model loaded.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

os.environ.setdefault("AURA_LOG_DIR", "/tmp/aura_fitted_steering")

DEFAULT_PLAN = REPO / "artifacts/migration/27b/recovery/steering_plan_wide.json"


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    from core.governance_context import local_internal_governed_scope
    from core.runtime.file_write_gateway import get_file_write_gateway

    with local_internal_governed_scope("steering.fitted_qualification", domain="file_write"):
        gateway = get_file_write_gateway()
        gateway.ensure_directory(path.parent, source="steering.fitted_qualification")
        gateway.write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n",
                           source="steering.fitted_qualification")


def seal(arguments) -> int:
    from core.brain.llm.model_registry import get_active_cortex_spec
    from core.evaluation.caa_causal_evaluation import validate_vector_generation
    from core.evaluation.fitted_steering_evaluation import fitted_generation_metadata

    root = arguments.vectors.expanduser().resolve(strict=True)
    spec = get_active_cortex_spec(force_refresh=True)
    if spec is None:
        print("no active cortex", file=sys.stderr)
        return 1
    descriptor = str(spec.descriptor_sha256)
    plan = json.loads(arguments.plan.read_text(encoding="utf-8"))
    metadata = fitted_generation_metadata(root, descriptor=descriptor, plan=plan)
    _write_json(root / "metadata.json", metadata)
    checked = validate_vector_generation(metadata, generation_dir=root, model_descriptor_sha256=descriptor)
    print(f"sealed {checked['vector_count']} vectors, generation {checked['generation_sha256'][:16]}")
    return 0


def _probe_descriptor(checkpoint: Path) -> dict:
    """The descriptor the certificate tool rehashes, so both name one identity."""
    from core.brain.llm.model_artifact_profile import build_model_artifact_descriptor

    spec = importlib.util.spec_from_file_location("measure_fusion_channel", REPO / "tools/measure_fusion_channel.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class _Arguments:
        model_path = str(checkpoint)
        model = ""
        require_active_cortex = True

    return module._probe_descriptor(_Arguments(), checkpoint, build_model_artifact_descriptor)


def evaluate(arguments) -> int:
    import numpy as np

    from core.consciousness.affective_steering import MEASUREMENT_BASIS_ENV, AffectiveSteeringEngine
    from core.consciousness.fusion_probe import (
        SETTLE_UPDATES,
        VOICE_PROMPT,
        VOICE_TOKENS,
        _log_likelihood,
        _reply_tokens,
        chat_ids,
    )
    from core.evaluation.fitted_steering_evaluation import (
        LESIONS,
        MATCHED_CONTROL,
        REPLIES_FILE,
        RESULT_SCHEMA,
        TREATMENT,
        derangement,
    )
    from core.runtime.model_lane_control import standalone_model_lane

    root = arguments.generation.expanduser().resolve(strict=True)
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    replies_path = root / REPLIES_FILE
    if _file_sha256(replies_path) != metadata["extraction_contract"]["held_out_replies_sha256"]:
        print("the held-out replies are not the ones the generation was sealed with", file=sys.stderr)
        return 1
    rows = [json.loads(line) for line in replies_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    checkpoint = arguments.model_path.expanduser().resolve(strict=True)
    os.environ[MEASUREMENT_BASIS_ENV] = str(root)

    with standalone_model_lane(owner_id="fitted-steering-evaluation", model_path=str(checkpoint),
                               purpose="measurement", request_gb=20.0,
                               metadata={"tool": "qualify_fitted_steering", "step": "evaluate"}):
        from mlx_lm import load

        descriptor = _probe_descriptor(checkpoint)
        if descriptor["descriptor_sha256"] != metadata["model_identity"]["model_descriptor_sha256"]:
            print("the checkpoint is not the one the generation is bound to", file=sys.stderr)
            return 1
        model, tokenizer = load(str(checkpoint))
        engine = AffectiveSteeringEngine()
        if not engine.attach(model, tokenizer, alpha=0.0, model_path=checkpoint, model_identity=descriptor):
            print(f"steering did not attach: {engine._model_info.get('attachment_error')}", file=sys.stderr)
            return 2
        hooks = list(engine._hooks)
        if not hooks or not all(hook._is_fitted_basis() for hook in hooks):
            print("the attached directions are not a fitted basis", file=sys.stderr)
            return 2

        def replaced(vector, array):
            clone = copy.copy(vector)
            object.__setattr__(clone, "v", array.astype(np.float32))
            object.__setattr__(clone, "_v_mx", None)
            return clone

        rng = np.random.default_rng(arguments.seed)
        saved = [dict(hook._vectors) for hook in hooks]
        random_unit = [
            {key: replaced(sv, (lambda r: r / np.linalg.norm(r))(rng.standard_normal(sv.d_model)))
             for key, sv in vectors.items()}
            for vectors in saved
        ]
        zeroed = [{key: replaced(sv, np.zeros(sv.d_model)) for key, sv in vectors.items()} for vectors in saved]
        moved = derangement(len(hooks), arguments.seed)
        assignments = {
            TREATMENT: saved,
            MATCHED_CONTROL: random_unit,
            LESIONS[0]: zeroed,
            LESIONS[1]: [saved[moved[index]] for index in range(len(hooks))],
        }
        other = derangement(len(rows), arguments.seed)
        prompt_ids = chat_ids(tokenizer, VOICE_PROMPT)

        def settle(state) -> None:
            vector = np.asarray(state, dtype=np.float32).reshape(-1)
            for hook in hooks:
                hook.override_composite_vector(None)
            for _ in range(SETTLE_UPDATES):
                for hook in hooks:
                    hook.update_substrate_vector(vector, source="fitted_steering_evaluation")

        scored, skipped, started = [], 0, time.monotonic()
        with engine.controlled_measurement():
            for index, row in enumerate(rows):
                tokens = _reply_tokens(tokenizer, str(row.get("text") or ""))[:VOICE_TOKENS]
                if len(tokens) < 2:
                    skipped += 1
                    continue
                engine.set_alpha(0.0)
                loglik = {"plain": _log_likelihood(model, prompt_ids, tokens)}
                for arm, assignment in assignments.items():
                    for hook, vectors in zip(hooks, assignment, strict=True):
                        hook._vectors = dict(vectors)
                    for at, state in (("own", row["s"]), ("other", rows[other[index]]["s"])):
                        settle(state)
                        engine.set_alpha(arguments.alpha)
                        loglik[f"{arm}:{at}"] = _log_likelihood(model, prompt_ids, tokens)
                        engine.set_alpha(0.0)
                for hook, vectors in zip(hooks, saved, strict=True):
                    hook._vectors = dict(vectors)
                scored.append({"index": index, "tokens": len(tokens), "loglik": loglik})
                if len(scored) % 10 == 0:
                    print(f"[{time.strftime('%H:%M:%S')}] {len(scored)} of {len(rows)} replies, "
                          f"{time.monotonic() - started:.0f} s", flush=True)

    result = {
        "schema": RESULT_SCHEMA,
        "model_descriptor_sha256": descriptor["descriptor_sha256"],
        "generation_sha256": metadata["generation_sha256"],
        "held_out_replies_sha256": metadata["extraction_contract"]["held_out_replies_sha256"],
        "alpha": arguments.alpha,
        "seed": arguments.seed,
        "other_state": other,
        "layer_derangement": moved,
        "layers": [hook._layer_idx for hook in hooks],
        "skipped_replies": skipped,
        "replies": scored,
        "measured_at": time.time(),
    }
    _write_json(arguments.out, result)
    print(f"wrote {arguments.out}: {len(scored)} replies scored, {skipped} too short")
    return 0


def verify(arguments) -> int:
    from core.evaluation.fitted_steering_evaluation import build_independent_verifier_evidence

    result = json.loads(arguments.result.read_text(encoding="utf-8"))
    metadata = json.loads(arguments.metadata.read_text(encoding="utf-8"))
    evidence = build_independent_verifier_evidence(
        result=result, result_sha256=_file_sha256(arguments.result),
        metadata=metadata, metadata_sha256=_file_sha256(arguments.metadata),
        generation_dir=arguments.metadata.parent,
    )
    _write_json(arguments.out, evidence)
    replayed = evidence["replay"]
    print(json.dumps({key: replayed[key] for key in (
        "sample_count", "treatment_successes", "matched_control_successes", "lesion_successes",
        "sign_test_p", "voice_delta", "causal_effect_positive", "unmet_requirements")}, indent=1))
    return 0 if evidence["verified"] else 3


def adjudicate(arguments) -> int:
    from core.evaluation.fitted_steering_evaluation import build_causal_evaluation

    evaluation = build_causal_evaluation(
        result=json.loads(arguments.result.read_text(encoding="utf-8")),
        metadata=json.loads(arguments.metadata.read_text(encoding="utf-8")),
        verifier_evidence=json.loads(arguments.independent_evidence.read_text(encoding="utf-8")),
        verifier_evidence_sha256=_file_sha256(arguments.independent_evidence),
    )
    _write_json(arguments.out, evaluation)
    print(f"wrote {arguments.out}: evaluation {evaluation['evaluation_sha256'][:16]}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="step", required=True)
    sealing = sub.add_parser("seal")
    sealing.add_argument("--vectors", type=Path, required=True)
    sealing.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    sealing.set_defaults(func=seal)
    scoring = sub.add_parser("evaluate")
    scoring.add_argument("--generation", type=Path, required=True)
    scoring.add_argument("--model-path", type=Path, required=True)
    scoring.add_argument("--alpha", type=float, default=0.5)
    scoring.add_argument("--seed", type=int, default=20260908)
    scoring.add_argument("--out", type=Path, required=True)
    scoring.set_defaults(func=evaluate)
    replaying = sub.add_parser("verify")
    replaying.add_argument("--result", type=Path, required=True)
    replaying.add_argument("--metadata", type=Path, required=True)
    replaying.add_argument("--out", type=Path, required=True)
    replaying.set_defaults(func=verify)
    deciding = sub.add_parser("adjudicate")
    deciding.add_argument("--result", type=Path, required=True)
    deciding.add_argument("--metadata", type=Path, required=True)
    deciding.add_argument("--independent-evidence", type=Path, required=True)
    deciding.add_argument("--out", type=Path, required=True)
    deciding.set_defaults(func=adjudicate)
    arguments = parser.parse_args(argv)
    return arguments.func(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
