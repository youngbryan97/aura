"""Source-only joint native fitting uses bounded real prefixes and one lane."""

import hashlib
import json
import weakref
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace

import mlx.core as mx
import pytest

from core.learning.semantic_grounded_binding_acquisition import (
    grounded_supervision_from_source_example,
)
from tests.test_semantic_program_shared_transducer import _shared_example
from tools.semantic_grounded_native_fit import fit_native_grounded_sources, native_capture_template


def sources():
    items = []
    for variant in (0, 2):
        item = _shared_example(three_steps=False, variant=variant, split="train")
        text = "".join(chr(token) for token in item.ir.source_token_ids)
        items.append(replace(item, ir=replace(item.ir, source_text_sha256=hashlib.sha256(text.encode()).hexdigest())))
    return tuple(items)


def test_native_capture_span_contract_is_blind_to_teacher_references():
    item = sources()[0]
    evidence = grounded_supervision_from_source_example(item).evidence
    changed = replace(item, ir=replace(item.ir, instructions=tuple(replace(instruction,
        args=tuple(reversed(instruction.args))) for instruction in item.ir.instructions)))
    assert native_capture_template(item, evidence, (0, 1)) == native_capture_template(changed, evidence, (0, 1))


def test_native_source_metadata_retains_exact_capture_without_retaining_archived_features():
    from tools.fit_semantic_grounded_binding import compact_native_sources
    item = sources()[0]
    features = weakref.ref(item.hidden_states)
    evidence = grounded_supervision_from_source_example(item).evidence
    expected = native_capture_template(item, evidence, (0, 1))
    compact = compact_native_sources((item,))[0]
    assert native_capture_template(compact, evidence, (0, 1)) == expected
    assert compact.ir is item.ir and not hasattr(compact, "hidden_states")
    del item
    assert features() is None


@pytest.mark.parametrize("interrupted", [False, "update", "completion", "prepared", "mixed", "hybrid"])
def test_joint_cli_engine_shards_actual_prefixes_and_drops_model_before_lane_release(monkeypatch, tmp_path, interrupted):
    import mlx_lm
    from mlx_lm.models.qwen2 import Model, ModelArgs

    import core.runtime.model_lane_control as lane
    import tools.train_semantic_native_program as custody

    config = {"model_type": "qwen2", "hidden_size": 16, "intermediate_size": 32,
        "num_hidden_layers": 3, "num_attention_heads": 4, "num_key_value_heads": 2,
        "vocab_size": 1024, "rms_norm_eps": 1e-6}
    if interrupted == "hybrid":
        from mlx_lm.models.qwen3_5 import TextModel as Model
        from mlx_lm.models.qwen3_5 import TextModelArgs as ModelArgs
        config = {**config, "model_type": "qwen3_5_text", "num_hidden_layers": 4,
            "hidden_size": 32, "intermediate_size": 64,
            "num_attention_heads": 2, "num_key_value_heads": 1,
            "head_dim": 16, "full_attention_interval": 4, "linear_num_key_heads": 2,
            "linear_num_value_heads": 2, "linear_key_head_dim": 32, "linear_value_head_dim": 32,
            "tie_word_embeddings": True}
    model_path = tmp_path / "fixture-model"
    model_path.mkdir()
    (model_path / "config.json").write_text(json.dumps(config))
    spec = SimpleNamespace(model_path=model_path, descriptor_sha256="a" * 64, pointer_sha256="b" * 64)
    monkeypatch.setattr(custody, "require_native_cortex_spec", lambda: spec)
    calls = []
    residents = []
    @contextmanager
    def owned(**kwargs):
        calls.append(("enter", kwargs))
        try:
            yield
        finally:
            assert all(reference() is None for reference in residents)
            calls.append(("exit", kwargs))
    monkeypatch.setattr(lane, "standalone_model_lane", owned)
    class Tokenizer:
        def encode(self, text, **kwargs):
            return list(map(ord, text))
        def decode(self, tokens, **kwargs):
            return "".join(map(chr, tokens))
    loads = []
    def load(path):
        loads.append(path)
        mx.random.seed(111)
        model = Model(ModelArgs.from_dict(config))
        if interrupted == "hybrid":
            import mlx.nn as nn
            nn.quantize(model, group_size=32, bits=4)
        residents.append(weakref.ref(model))
        return model, Tokenizer()
    monkeypatch.setattr(mlx_lm, "load", load)
    items = sources()
    examples = tuple(grounded_supervision_from_source_example(item) for item in items)
    options = dict(spec=spec, rank=2, layers=2, max_tokens=64, cache_bytes=8192, relation_width=8,
        fit_options={"steps": 4, "save_every": 2, "learning_rate": .01, "max_seconds": 30., "role_margin": .25})
    if interrupted in {"prepared", "hybrid"}:
        from core.learning.semantic_program_compositional_transducer import (
            fit_compositional_semantic_program_transducer,
        )
        from tests.test_semantic_program_shared_transducer import _examples, _grounding
        parent = fit_compositional_semantic_program_transducer(_examples(), input_grounding=_grounding())
        parent_bytes = json.dumps(parent.to_dict()).encode()
        options["source_basis"] = {"parent_sha256": hashlib.sha256(parent_bytes).hexdigest()}
    if interrupted == "mixed":
        options["adapter_options"] = {"layer_kinds": ["lora", "product"], "layer_ranks": [2, 3]}
    if interrupted == "hybrid":
        options.update(layers=3, adapter_options={"layer_kinds": ["lora", "product", "silu"]})
    if interrupted == "prepared":
        engine, prepared = fit_native_grounded_sources(examples[:1], examples[1:], items,
            tmp_path / "fit", prepare_only=True, **options)
        assert engine is None and not loads and not calls
        assert not prepared["model_weights_loaded"] and prepared["semantic_success"] is None
        assert not prepared["activation_memory_is_measured"]
        assert not (tmp_path / "fit").exists()
        assert not (tmp_path / "fit-native-custody" / "prefixes").exists()
        with pytest.raises(ValueError, match="plan"):
            fit_native_grounded_sources(examples[:1], examples[1:], items, tmp_path / "fit",
                **{**options, "rounds": 3})
        assert not loads and not calls
    if interrupted == "update":
        _unused, uninterrupted = fit_native_grounded_sources(examples[:1], examples[1:], items,
            tmp_path / "uninterrupted", **options)
        import core.learning.semantic_grounded_binding_engine as module
        original = module._save_grounded_restart

        def interrupt(directory, state, *args):
            original(directory, state, *args)
            if state["step"] == 2:
                from tools.verify_semantic_grounded_restart import verify as verify_restart
                checked = verify_restart(directory)
                assert checked["step"] == 2 and checked["generation_integrity_verified"]
                assert not checked["report_exists"] and not checked["completion_exists"]
                raise RuntimeError("native interruption")

        monkeypatch.setattr(module, "_save_grounded_restart", interrupt)
        with pytest.raises(RuntimeError, match="native interruption"):
            fit_native_grounded_sources(examples[:1], examples[1:], items, tmp_path / "fit", **options)
        monkeypatch.setattr(module, "_save_grounded_restart", original)
        engine, report = fit_native_grounded_sources(examples[:1], examples[1:], items, tmp_path / "fit",
            resume=True, **options)
        assert report["resume_from_step"] == 2
        assert report["history"] == uninterrupted["history"]
        assert report["selected_step"] == uninterrupted["selected_step"]
        for name in ("checkpoint-4.safetensors", "selected.safetensors"):
            expected = mx.load(str(tmp_path / "uninterrupted" / name))
            actual = mx.load(str(tmp_path / "fit" / name))
            assert set(actual) == set(expected)
            assert all(mx.array_equal(actual[key], value).item() for key, value in expected.items())
    elif interrupted == "completion":
        from core.runtime.file_write_gateway import get_file_write_gateway
        gateway = get_file_write_gateway()
        original = gateway.write_bytes_if_absent

        def interrupt(path, *args, **kwargs):
            if path.name == "completion.json":
                raise OSError("completion publication interrupted")
            return original(path, *args, **kwargs)

        monkeypatch.setattr(gateway, "write_bytes_if_absent", interrupt)
        with pytest.raises(OSError, match="completion publication"):
            fit_native_grounded_sources(examples[:1], examples[1:], items, tmp_path / "fit", **options)
        monkeypatch.setattr(gateway, "write_bytes_if_absent", original)
        with pytest.raises(ValueError, match="source supervision"):
            fit_native_grounded_sources((replace(examples[0], environment="changed"),), examples[1:], items,
                tmp_path / "fit", resume=True, **options)
        assert len(loads) == 1
        engine, report = fit_native_grounded_sources(examples[:1], examples[1:], items, tmp_path / "fit",
            resume=True, **options)
    else:
        engine, report = fit_native_grounded_sources(examples[:1], examples[1:], items, tmp_path / "fit", **options)
    assert engine is None and report["joint_native_adapter_training"] and report["semantic_success"] is None
    assert len(loads) == (3 if interrupted == "update" else 1)
    assert [event for event, _ in calls] == (["enter", "exit"] * 3 if interrupted == "update" else ["enter", "exit"])
    assert calls[0][1]["require_exclusive"] and not calls[0][1]["allow_owner_eviction"]
    receipt = json.loads((tmp_path / "fit-native-custody" / "completion.json").read_text())
    assert receipt["state_store"]["sources"] == 2 and receipt["state_store"]["peak_resident_bytes"] <= 8192
    assert receipt["observed_adapter_parameters"] == receipt["adapter_projection"]["trainable_parameters"]
    assert not receipt["held_sources_scored"] and not receipt["serving_authority"]
    if interrupted == "completion":
        assert receipt["completion_recovery"] == "verified_saved_fit_without_model_loading"
        assert receipt["memory_envelope"] is None and receipt["peak_memory_bytes"] is None
    assert any(key.startswith("native_suffix.") for key in mx.load(str(tmp_path / "fit" / "selected.safetensors")))
    from tools.verify_semantic_grounded_fit import verify
    checked = verify(tmp_path / "fit")
    assert checked["native_acquisition_sha256"] and checked["artifacts_verified"]
    assert not checked["model_weights_loaded"] and not checked["held_sources_scored"]
    from tools.verify_semantic_grounded_restart import verify as verify_restart
    restart = verify_restart(tmp_path / "fit")
    assert restart["step"] == 4 and restart["completion_exists"]
    assert not restart["model_weights_loaded"] and not restart["qualification_evidence"]
    if interrupted in {"prepared", "hybrid"}:
        from tools.semantic_grounded_native_decode import GroundedNativeChartDecoder
        mx.random.seed(111)
        native = Model(ModelArgs.from_dict(config))
        if interrupted == "hybrid":
            import mlx.nn as nn
            nn.quantize(native, group_size=32, bits=4)
        with pytest.raises(ValueError, match="custody"):
            GroundedNativeChartDecoder.from_fit(native, directory=tmp_path / "fit",
                parent_bytes=parent_bytes + b" ", spec=spec)
        if not checked["learned_checkpoint_selected"]:
            with pytest.raises(ValueError, match="custody"):
                GroundedNativeChartDecoder.from_fit(native, directory=tmp_path / "fit",
                    parent_bytes=parent_bytes, spec=spec)
        decoder = GroundedNativeChartDecoder.from_fit(native, directory=tmp_path / "fit",
            parent_bytes=parent_bytes, spec=spec, allow_initial_checkpoint=True)
        item = items[0]
        result = decoder.decode(source_token_ids=item.ir.source_token_ids, hidden_states=item.hidden_states,
            public_inputs=item.public_inputs, source_text_sha256=item.ir.source_text_sha256,
            model_basis_sha256=parent.model_basis_sha256, search_time_limit_s=30.)
        assert decoder.last_receipt["examined_charts"] > 0
        if result.ir is not None:
            assert decoder.last_receipt["selected_chart"]["all_options_retained"]
        else:
            assert decoder.last_receipt["selected_chart"] is None and decoder.last_receipt["refusal"]
        assert decoder.last_receipt["native_state_shape"] == [len(item.ir.source_token_ids),
            options["layers"], config["hidden_size"]]
        assert not decoder.last_receipt["target_available_to_decoder"]
        assert not decoder.last_receipt["serving_authority"]
    with pytest.raises(ValueError, match="already complete"):
        fit_native_grounded_sources(examples[:1], examples[1:], items, tmp_path / "fit", resume=True, **options)
    assert len(loads) == (3 if interrupted == "update" else 1)
    pointer = json.loads((tmp_path / "fit" / "resume.json").read_bytes())
    generation = tmp_path / "fit" / pointer["file"]
    generation.write_bytes(generation.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        verify_restart(tmp_path / "fit")


def test_restart_verdict_matches_the_detached_consumer_and_rejects_unbound_context():
    from tools.run_detached_step import validate_resume_verdict
    from tools.verify_semantic_grounded_restart import detached_verdict
    context = {"transport": "stdout-v3", "plan_sha256": "a" * 64, "command_sha256": "b" * 64,
        "prior_attempt": 1, "prior_journal_head_sha256": "c" * 64}
    for complete in (False, True):
        result = detached_verdict({"step": 32, "completion_exists": complete}, context)
        accepted = validate_resume_verdict(result, **{key: value for key, value in context.items() if key != "transport"})
        assert accepted["verdict"] == ("already_completed" if complete else "safe_to_resume")
    with pytest.raises(ValueError, match="bound detached"):
        detached_verdict({"step": 0, "completion_exists": False}, {**context, "prior_attempt": 0})


def test_resume_if_available_refuses_partial_directories(tmp_path):
    from tools.fit_semantic_grounded_binding import resume_native_if_available
    assert not resume_native_if_available(tmp_path / "missing")
    with pytest.raises(FileNotFoundError):
        resume_native_if_available(tmp_path)


def test_bad_source_population_or_token_bound_fails_before_model_loading(tmp_path):
    item = sources()[0]
    example = grounded_supervision_from_source_example(item)
    spec = SimpleNamespace(model_path=tmp_path, descriptor_sha256="a" * 64, pointer_sha256="b" * 64)
    with pytest.raises(ValueError, match="disjoint"):
        fit_native_grounded_sources((example,), (example,), (item,), tmp_path / "unused", spec=spec)
    assert not (tmp_path / "unused-native-custody").exists()


@pytest.mark.parametrize("options,match", [
    ({"fit_options": {"steps": 3, "save_every": 2}}, "checkpoints"),
    ({"fit_options": {"training_schedule": ["held-source"]}}, "schedule"),
    ({"fit_options": {"domain_reversal": .1}}, "multiple source"),
    ({"prepare_only": True, "resume": True}, "geometry"),
])
def test_preparation_rejects_invalid_fit_contract_without_acquiring_model(tmp_path, options, match):
    items = sources()
    examples = tuple(grounded_supervision_from_source_example(item) for item in items)
    spec = SimpleNamespace(model_path=tmp_path, descriptor_sha256="a" * 64, pointer_sha256="b" * 64)
    with pytest.raises(ValueError, match=match):
        fit_native_grounded_sources(examples[:1], examples[1:], items, tmp_path / "unused",
            spec=spec, **options)
    assert not (tmp_path / "unused-native-custody").exists()


def test_missing_launch_arithmetic_rejected_before_preparation(monkeypatch, tmp_path):
    import tools.probe_semantic_native_prefix_branches as arithmetic
    monkeypatch.setattr(arithmetic, "installed_arithmetic_basis", lambda: {"MLX_ENABLE_TF32": None})
    items = sources()
    examples = tuple(grounded_supervision_from_source_example(item) for item in items)
    spec = SimpleNamespace(model_path=tmp_path, descriptor_sha256="a" * 64, pointer_sha256="b" * 64)
    with pytest.raises(ValueError, match="process launch"):
        fit_native_grounded_sources(examples[:1], examples[1:], items, tmp_path / "unused",
            spec=spec, prepare_only=True)
    assert not (tmp_path / "unused-native-custody").exists()
