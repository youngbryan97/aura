#!/usr/bin/env python3
"""Qualify a frozen program reader on the current model, as her runtime would run it (G10).

The reader was evaluated offline on hidden states a standalone process
captured with no steering attached. Her runtime reads a request through its
resident worker, where her affective steering and latent bridge are attached.
This loads her model once and measures, on every request whose states were
recorded for G04 and G05:

1. conversion: the worker's own encode (mlx_worker._encode_hidden_sequence_response)
   returns the recorded token ids and bit-identical hidden states;
2. geometry: every hook sits on a layer the model has and every steering
   vector has the model's width; the states have the recorded width;
3. steering: with steering and the latent bridge attached the worker's way
   (_attach_affective_steering, _attach_latent_bridge) and settled at each of
   the fusion probe's declared extreme states, the encode is bit-identical
   again and teaches the latent bridge nothing;
4. lesion: the same encode with the observation pass removed differs, which
   is what makes (3) a test, and the reader's answers on those states are
   counted;
5. rollback: after the engine detaches, greedy continuations of the fusion
   probe's prompts and the encodes are identical to the never-attached ones;
6. canaries: through the runtime's own observation path
   (semantic_program_runtime.execute_compositional_semantic_observation, with
   its basis check) the reader returns what it returned on the recorded
   states offline, request by request.

Usage:
    qualify_g10_reader.py --reader FILE --features DIR [--features DIR ...] --output FILE
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#: Where her cortex authority key lives under her own state root
#: (core/learning/cortex_migration_authority.default_authority_key_path).
HER_AUTHORITY_KEY = Path("~/.aura/private/cortex-upgrade/migration-authority.key")


def own_state_root(output: Path, authority_key: Path) -> Path:
    """A fresh state root for this run, reading her authority key and nothing else of hers.

    Attaching steering needs her exact model identity, which is signed with
    that key, and materializes her signed steering generation into the state
    root. Done in her own state root that would write into it; here the
    generation is materialized into the run's root and the key is only read.
    Set before anything imports core, which reads the root on import.
    """
    root = output.parent / f"{output.stem}.state"
    if root.exists():
        raise SystemExit(f"{root} exists; a run's state root is used once")
    root.mkdir(parents=True)
    os.environ["AURA_STATE_ROOT"] = str(root)
    os.environ["AURA_CORTEX_AUTHORITY_KEY_FILE"] = str(authority_key.expanduser().resolve(strict=True))
    return root


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def runtime_reading(reader, text, token_ids, offsets, hidden, receipt, expected_basis) -> dict[str, Any]:
    from core.cognition.procedure import get_procedure_registry
    from core.learning.semantic_program_ir import semantic_value_to_json
    from core.learning.semantic_program_runtime import (
        SemanticProgramDecodeRejectedError,
        SemanticProgramObservationError,
        execute_compositional_semantic_observation,
    )

    try:
        outcome = execute_compositional_semantic_observation(
            model=reader, source_text=text, source_token_ids=token_ids, offset_mapping=offsets,
            hidden_states=hidden, worker_model_basis=receipt["model_basis"],
            expected_representation_basis_sha256=expected_basis, procedure_registry=get_procedure_registry())
    except (SemanticProgramDecodeRejectedError, SemanticProgramObservationError) as exc:
        return {"read": False, "reason": str(exc)}
    return {"read": True, "program": outcome.receipt["semantic_ir_receipt"]["alpha_normalized_sha256"],
            "answer": semantic_value_to_json(outcome.execution.result)}


def offline_reading(reader, item) -> dict[str, Any]:
    """The reader on the recorded states, as G04 and G05 decoded them, then executed."""
    from core.learning.semantic_program_floor import (
        compile_semantic_program_to_floor,
        execute_semantic_floor_program,
    )
    from core.learning.semantic_program_ir import semantic_value_to_json
    from tools.run_g04_transfer import decode

    outcome, failure = decode(reader, item)
    if outcome is None or outcome.ir is None:
        return {"read": False, "reason": failure or str(getattr(outcome, "refusal", ""))}
    try:
        execution = execute_semantic_floor_program(compile_semantic_program_to_floor(outcome.ir, item.public_inputs))
    except (RuntimeError, TypeError, ValueError) as exc:
        return {"read": True, "program": outcome.ir.receipt()["alpha_normalized_sha256"], "answer": None,
                "reason": f"not_executable:{exc}"}
    return {"read": True, "program": outcome.ir.receipt()["alpha_normalized_sha256"],
            "answer": semantic_value_to_json(execution.result)}


def same_reading(first: dict[str, Any], second: dict[str, Any]) -> bool:
    if first["read"] != second["read"]:
        return False
    if not first["read"]:
        return True
    return first.get("program") == second.get("program") and first.get("answer") == second.get("answer")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reader", type=Path, required=True)
    parser.add_argument("--features", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0, help="first N requests only (a smoke run)")
    parser.add_argument("--authority-key", type=Path, default=HER_AUTHORITY_KEY)
    args = parser.parse_args()
    state = own_state_root(args.output.expanduser(), args.authority_key)

    import numpy as np

    from core.learning.compositional_semantic_qualification import (
        SEMANTIC_READER_QUALIFICATION_SCHEMA,
    )
    from core.learning.semantic_operation_peaks import semantic_reader_from_dict
    from core.learning.semantic_program_campaign import training_examples_from_feature_bundle
    from core.learning.semantic_program_feature_materialization import (
        load_standard_semantic_feature_bundle,
        offset_tokenizer_for_worker,
        rebuild_semantic_feature_selection,
        tokenize_with_offsets,
        tokenizer_checkpoint_identity,
    )

    reader_raw = args.reader.expanduser().read_bytes()
    reader = semantic_reader_from_dict(json.loads(reader_raw))
    requests: list[dict[str, Any]] = []
    model_paths = set()
    for directory in args.features:
        directory = directory.expanduser()
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        model_paths.add(manifest["exact_model_path"])
        _config, examples = rebuild_semantic_feature_selection(manifest)
        texts = {example.example_id: example.source_text for example in examples}
        bundle = load_standard_semantic_feature_bundle(directory)
        items = training_examples_from_feature_bundle(
            bundle, required_splits=frozenset({str(e.metadata["split"]) for e in bundle.examples}))
        for stored, item in zip(bundle.examples, items, strict=True):
            item = replace(item, ir=replace(item.ir, model_basis_receipt_sha256=reader.model_basis_sha256))
            requests.append({"bundle": directory.name, "id": stored.metadata["example_id"],
                             "text": texts[stored.metadata["example_id"]], "token_ids": list(stored.token_ids),
                             "hidden_sha256": stored.metadata["worker_receipt"]["hidden_state_sha256"],
                             "hidden_size": int(stored.metadata["hidden_size"]),
                             "representation": stored.metadata["worker_receipt"]["representation"],
                             "item": item})
    if len(model_paths) != 1:
        raise SystemExit("the feature bundles were not all read from one model")
    if args.limit:
        requests = requests[: args.limit]
    model_path = Path(next(iter(model_paths))).resolve(strict=True)
    print(json.dumps({"requests": len(requests), "model": str(model_path)}), flush=True)

    from mlx_lm import load

    from core.brain.llm import mlx_worker
    from core.brain.llm.latent_cortex.runtime_identity import (
        build_worker_identity,
        worker_representation_basis,
    )
    from core.brain.llm.latent_cortex.worker_capture_identity import build_worker_capture_identity
    from core.brain.llm.model_registry import resolve_cortex_bound_artifact
    from core.consciousness import fusion_probe
    from core.runtime.model_lane_control import standalone_model_lane

    began = time.monotonic()
    result: dict[str, Any] = {"schema": SEMANTIC_READER_QUALIFICATION_SCHEMA,
                              "reader": {"path": str(args.reader.expanduser().resolve()),
                                         "file_sha256": hashlib.sha256(reader_raw).hexdigest(),
                                         "receipt_sha256": reader.receipt_sha256},
                              "requests": len(requests), "state_root": str(state)}
    with standalone_model_lane(owner_id=f"g10-reader:{args.output.name}", model_path=str(model_path),
                               purpose="evaluation", preemptible=False, require_exclusive=True,
                               allow_owner_eviction=True, metadata={"tool": Path(__file__).name}):
        model, tokenizer = load(str(model_path))
        boot_id = hashlib.sha256(f"g10:{time.time_ns()}".encode()).hexdigest()[:32]
        capture = build_worker_capture_identity(worker_boot_id=boot_id)

        def identity(engine: Any = None) -> dict[str, Any]:
            active = bool(engine is not None and engine.is_active())
            return build_worker_identity(model, model_path=model_path, worker_boot_id=boot_id,
                                         worker_source_path=Path(mlx_worker.__file__),
                                         worker_action_capture_identity=capture.public_identity,
                                         tokenizer=tokenizer, affective_steering_active=active,
                                         affective_steering_alpha=float(engine._alpha) if active else 0.0)

        offset_tokenizer = offset_tokenizer_for_worker(tokenizer)
        encoder_cache: dict[str, Any] = {}

        def encode(request: dict[str, Any], worker_identity: dict[str, Any]) -> tuple[list[int], Any, dict]:
            response = mlx_worker._encode_hidden_sequence_response(
                model=model, tokenizer=tokenizer, text=request["text"], request_id=request["id"],
                encoder_cache=encoder_cache, worker_identity=worker_identity,
                metal_semaphore=contextlib.nullcontext(), representation=request["representation"])
            hidden = np.frombuffer(response["hidden_state_bytes"], dtype="<f4").reshape(response["hidden_shape"])
            return response["token_ids"], hidden, response["receipt"]

        prompts = [fusion_probe.chat_ids(tokenizer, prompt) for prompt in fusion_probe.PROBE_PROMPTS]
        steps = 24  # the step count of the certificate the runtime serves under (owned_layer_matched_v1)
        never_attached_paths = [fusion_probe._greedy_path(model, ids, steps) for ids in prompts]

        # 1, 6: the never-attached worker against the recorded states and the offline readings.
        plain_identity = identity()
        plain: dict[str, str] = {}
        rows = []
        expected_basis = None
        for count, request in enumerate(requests, 1):
            started = time.monotonic()
            token_ids, hidden, receipt = encode(request, plain_identity)
            encoded = time.monotonic()
            basis = _sha(worker_representation_basis(receipt["model_basis"]))
            expected_basis = expected_basis or basis
            offsets = tokenize_with_offsets(offset_tokenizer, request["text"])[1]
            runtime = runtime_reading(reader, request["text"], token_ids, offsets, hidden, receipt, expected_basis)
            read = time.monotonic()
            offline = offline_reading(reader, request["item"])
            plain[request["id"]] = receipt["hidden_state_sha256"]
            rows.append({"bundle": request["bundle"], "id": request["id"],
                         "tokens_match": token_ids == request["token_ids"],
                         "states_match": receipt["hidden_state_sha256"] == request["hidden_sha256"],
                         "width_matches": int(hidden.shape[1]) == request["hidden_size"],
                         "basis": basis, "offline": offline, "runtime": runtime,
                         "runtime_matches_offline": same_reading(offline, runtime),
                         # The reader's own cost, for G06: one forward for the states, then the reading.
                         "encode_seconds": round(encoded - started, 4),
                         "reading_seconds": round(read - encoded, 4)})
            if count % 25 == 0:
                print(json.dumps({"phase": "plain", "done": count, "of": len(requests),
                                  "seconds": round(time.monotonic() - began, 1)}), flush=True)

        # 2, 3: attached the worker's way, at each declared extreme state.
        attachment = mlx_worker._attach_affective_steering(model, tokenizer, None, None, None,
                                                           model_path=str(model_path))
        engine = attachment.engine
        if engine is None or not engine.active_hooks():
            raise SystemExit(f"steering did not attach: {attachment.disposition}")
        import multiprocessing

        from core.consciousness.latent_readout_channel import create_channel

        bridge = mlx_worker._attach_latent_bridge(model, create_channel(multiprocessing.get_context("spawn")))
        if bridge is not None and getattr(bridge, "_injection_thread", None) is not None:
            bridge._injection_thread.stop()  # nothing reads the channel here; the readout hooks stay live
        hooks = engine.active_hooks()
        layers = len(model.model.layers) if hasattr(model, "model") else len(model.layers)
        widths = {vector["width"] for hook in hooks for vector in hook.fusion_basis()["vectors"]}
        hidden_size = int(getattr(model.args, "hidden_size", 0) or 0)
        bridge_hooks = list(getattr(bridge, "_readout_hooks", None) or [])
        from core.consciousness.fusion_certificate import steering_basis_sha256

        descriptor = resolve_cortex_bound_artifact(str(model_path)).descriptor or {}
        basis = steering_basis_sha256(hooks)
        served_alpha = float(engine._alpha)
        if served_alpha <= 0.0:
            # Neutral at attach: steer at the ceiling her runtime serves user surfaces under.
            from core.consciousness.fusion_certificate import certified_alpha

            served_alpha = float(certified_alpha(str(descriptor.get("descriptor_sha256", "")), basis_sha256=basis))
            engine.set_alpha(served_alpha)
        result["steering"] = {"disposition": attachment.disposition, "alpha": served_alpha,
                              "alpha_at_attach": float(engine._alpha) if attachment.active else 0.0,
                              "hooks": sorted(hook._layer_idx for hook in hooks), "basis_sha256": basis,
                              "descriptor_sha256": descriptor.get("descriptor_sha256"),
                              "latent_bridge_hooks": len(bridge_hooks)}
        geometry = (all(0 <= hook._layer_idx < layers for hook in hooks) and widths == {hidden_size}
                    and all(row["width_matches"] for row in rows))

        def settle(moods: dict[str, float]) -> None:
            for _ in range(fusion_probe.SETTLE_UPDATES):
                for hook in hooks:
                    hook.update_substrate(moods)
                    hook._last_substrate_sync_monotonic = time.monotonic()

        def readouts() -> int:
            return sum(int(getattr(hook, "_readout_count", 0)) for hook in bridge_hooks)

        steered: dict[str, dict[str, Any]] = {}
        for state_name, moods in (("high", fusion_probe.STATE_HIGH), ("low", fusion_probe.STATE_LOW)):
            settle(moods)
            steered_identity = identity(engine)
            composites = [None if hook.current_composite_vector() is None else hook.current_composite_vector().tolist()
                          for hook in hooks]
            readouts_before = readouts()
            same = differ = 0
            for request in requests:
                _tokens, _hidden, receipt = encode(request, steered_identity)
                same += receipt["hidden_state_sha256"] == plain[request["id"]]
                differ += receipt["hidden_state_sha256"] != plain[request["id"]]
            unchanged_state = composites == [None if hook.current_composite_vector() is None
                                             else hook.current_composite_vector().tolist() for hook in hooks]
            steered[state_name] = {"identical": same, "different": differ,
                                   "injecting": any(c is not None for c in composites),
                                   "effective_alpha": [hook._effective_alpha() for hook in hooks][:1],
                                   "latent_readouts_added": readouts() - readouts_before,
                                   "affect_state_unchanged": unchanged_state,
                                   "receipt_basis": _sha(worker_representation_basis(receipt["model_basis"]))}
            print(json.dumps({"phase": f"steered_{state_name}", **steered[state_name]}), flush=True)

        # 4: the lesion, at the high state: the encode without its observation pass.
        settle(fusion_probe.STATE_HIGH)
        from core.consciousness import affective_steering, latent_bridge

        lesion = {"different": 0, "identical": 0, "reading_changed": 0}
        with contextlib.ExitStack() as stack:
            for module in (affective_steering, latent_bridge):
                original = module.in_observation_pass
                module.in_observation_pass = lambda: False
                stack.callback(setattr, module, "in_observation_pass", original)
            for request, row in zip(requests, rows, strict=True):
                token_ids, hidden, receipt = encode(request, plain_identity)
                if receipt["hidden_state_sha256"] == plain[request["id"]]:
                    lesion["identical"] += 1
                    continue
                lesion["different"] += 1
                offsets = tokenize_with_offsets(offset_tokenizer, request["text"])[1]
                lesioned = runtime_reading(reader, request["text"], token_ids, offsets, hidden, receipt,
                                           expected_basis)
                lesion["reading_changed"] += not same_reading(row["runtime"], lesioned)
        print(json.dumps({"phase": "lesion", **lesion}), flush=True)

        # 5: rollback.
        engine.detach()
        detached_identity = identity()
        rolled_back_paths = [fusion_probe._greedy_path(model, ids, steps) for ids in prompts]
        rolled_back_states = sum(encode(request, detached_identity)[2]["hidden_state_sha256"] == plain[request["id"]]
                                 for request in requests)

        tokenizer_identity = tokenizer_checkpoint_identity(model_path)
        result.update({
            "model": {"path": str(model_path),
                      "descriptor_sha256": result["steering"]["descriptor_sha256"],
                      "tokenizer_identity_sha256": tokenizer_identity["identity_sha256"],
                      "representation_basis_sha256": expected_basis},
            "steered": steered, "lesion": lesion,
            "rollback": {"paths_identical": rolled_back_paths == never_attached_paths,
                         "states_identical": rolled_back_states, "steps": steps,
                         "prompts": len(prompts)},
            "rows": [{key: value for key, value in row.items()} for row in rows],
            "seconds": round(time.monotonic() - began, 1),
        })
    n = len(requests)
    result["predicates"] = {
        "conversion_reproduces_stored_states": all(r["tokens_match"] and r["states_match"] for r in rows),
        "geometry_matches": bool(geometry),
        "steering_does_not_reach_the_reading": all(
            s["identical"] == n and s["injecting"] and s["latent_readouts_added"] == 0
            and s["affect_state_unchanged"] and s["receipt_basis"] == expected_basis for s in steered.values()),
        "lesion_shows_the_check_can_fail": lesion["different"] > 0,
        "rollback_restores_generation": result["rollback"]["paths_identical"],
        "rollback_restores_states": result["rollback"]["states_identical"] == n,
        "readings_match_offline_evaluation": all(r["runtime_matches_offline"] for r in rows),
    }
    result["qualified"] = all(result["predicates"].values())
    output = args.output.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=1, sort_keys=True, default=str), encoding="utf-8")
    temporary.replace(output)
    print(json.dumps({"predicates": result["predicates"], "qualified": result["qualified"],
                      "seconds": result["seconds"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
