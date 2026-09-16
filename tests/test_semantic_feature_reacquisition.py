import asyncio
import copy
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from core.learning import semantic_program_feature_materialization as features
from tests.test_semantic_program_feature_materialization import (
    _CharacterTokenizer, _FeatureClient, _lane_receipt, _sha, _tokenizer_identity,
)
from tools import materialize_semantic_program_features as cli


@pytest.fixture
def acquired(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    config = features.SemanticFeatureConfig(
        schema=features.FAMILY_FEATURE_CONFIG_SCHEMA,
        corpus_kind=features.COUNTERFACTUAL_SOURCE_CORPUS_KIND,
        seed=43, max_examples=36, idle_wait_s=0,
    )
    corpus = features.build_semantic_program_corpus_for_config(config)
    output = tmp_path / "original"
    asyncio.run(features.materialize_semantic_program_features(
        client=_FeatureClient(model), checkpoint=model, tokenizer=_CharacterTokenizer(),
        output_directory=output, config=config, corpus=corpus,
        lane_ownership_receipt=_lane_receipt(model), tokenizer_identity=_tokenizer_identity(model),
    ))
    return model, output, config, corpus


def _plan(tmp_path, jobs):
    path = tmp_path / "plan.json"
    path.write_text(json.dumps({"schema": "aura.semantic_feature_reacquisition_plan.v1", "jobs": jobs}))
    return SimpleNamespace(plan=path, output=tmp_path / "new")


def test_reacquisition_rebuilds_exact_selection_without_loading_arrays(acquired, tmp_path, monkeypatch):
    _model, source, config, corpus = acquired
    monkeypatch.setattr(features, "load_semantic_feature_record", lambda *_a, **_k: pytest.fail("old array read"))
    args = _plan(tmp_path, [{"name": "counterfactual", "source_manifest": str(source / "manifest.json")}])
    name, target, rebuilt_config, rebuilt, digest = cli._configured_jobs(args)[0]
    assert name == "counterfactual"
    assert target == args.output / name
    assert rebuilt_config == config
    assert rebuilt == features.select_bounded_semantic_examples(corpus, max_examples=36)
    assert {item.split for item in rebuilt} == {"train"}
    assert len(digest) == 64


@pytest.mark.parametrize("change", ["hash", "selection", "corpus", "config"])
def test_reconstruction_refuses_relabelled_or_tampered_source(acquired, change):
    _model, source, _config, _corpus = acquired
    manifest = json.loads((source / "manifest.json").read_text())
    if change == "selection":
        manifest["config"]["selected_example_ids"].reverse()
        manifest["config_sha256"] = _sha(manifest["config"])
    elif change == "config":
        manifest["config"]["seed"] += 1
    else:
        manifest["corpus_sha256"] = "0" * 64
    if change != "hash":
        manifest["manifest_sha256"] = _sha({k: v for k, v in manifest.items() if k != "manifest_sha256"})
    with pytest.raises(features.SemanticFeatureMaterializationError):
        features.rebuild_semantic_feature_selection(manifest)


@pytest.mark.parametrize("name", ["../original", "", ".", "a/b", ".."])
def test_plan_refuses_output_path_escape(tmp_path, name):
    args = _plan(tmp_path, [{"name": name, "source_manifest": "missing"}])
    with pytest.raises(ValueError, match="name"):
        cli._configured_jobs(args)


def test_plan_refuses_duplicate_and_source_output_alias(acquired, tmp_path):
    _model, source, _config, _corpus = acquired
    job = {"name": "original", "source_manifest": str(source / "manifest.json")}
    args = _plan(tmp_path, [job, copy.deepcopy(job)])
    with pytest.raises(ValueError, match="duplicate"):
        cli._configured_jobs(args)
    args = _plan(tmp_path, [job])
    args.output = source.parent
    with pytest.raises(ValueError, match="overwrite"):
        cli._configured_jobs(args)


def test_two_real_cohorts_share_one_worker_lifecycle(acquired, tmp_path):
    model, source, _config, _corpus = acquired
    args = _plan(tmp_path, [{"name": name, "source_manifest": str(source / "manifest.json")}
        for name in ("first", "second")])
    client = _FeatureClient(model)
    client.warmup = AsyncMock(return_value=True)
    client.aclose = AsyncMock()
    client.get_model_lane_ownership_snapshot = lambda: _lane_receipt(model)
    results = asyncio.run(cli._acquire_jobs(client, model=model, tokenizer=_CharacterTokenizer(),
        tokenizer_identity=_tokenizer_identity(model), jobs=cli._configured_jobs(args)))
    client.warmup.assert_awaited_once_with(foreground_request=True, skip_swap_cooldown=True)
    client.aclose.assert_awaited_once()
    assert client.calls == 72
    assert len(results) == 2 and all(row["complete"] for row in results)
    for row in results:
        assert len(features.load_standard_semantic_feature_bundle(args.output / row["name"]).examples) == 36


def test_busy_worker_writes_durable_partial_status_and_stops_remaining_jobs(acquired, tmp_path):
    model, source, _config, _corpus = acquired
    args = _plan(tmp_path, [{"name": name, "source_manifest": str(source / "manifest.json")}
        for name in ("first", "second")])
    client = SimpleNamespace(warmup=AsyncMock(return_value=True), aclose=AsyncMock(),
        encode_hidden_sequence=AsyncMock(return_value=None),
        get_model_lane_ownership_snapshot=lambda: _lane_receipt(model))
    results = asyncio.run(cli._acquire_jobs(client, model=model, tokenizer=_CharacterTokenizer(),
        tokenizer_identity=_tokenizer_identity(model), jobs=cli._configured_jobs(args)))
    assert len(results) == 1 and not results[0]["complete"]
    assert not (args.output / "second").exists()
    status = json.loads((args.output / "first" / "status.json").read_text())
    assert status["reason"] == "resident_lane_busy" and status["complete"] is False
    client.aclose.assert_awaited_once()


def test_worker_failure_preserves_primary_error_and_closes_once():
    error = RuntimeError("warmup failed")
    client = SimpleNamespace(warmup=AsyncMock(side_effect=error), aclose=AsyncMock(side_effect=ValueError("close failed")))
    with pytest.raises(RuntimeError, match="warmup failed") as caught:
        asyncio.run(cli._acquire_jobs(client, model=None, tokenizer=None, tokenizer_identity=None, jobs=[]))
    assert any("close failed" in note for note in caught.value.__notes__)
    client.aclose.assert_awaited_once()
