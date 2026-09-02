from __future__ import annotations

import asyncio
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from core.brain.llm.hidden_sequence_contract import (
    FINAL_HIDDEN_V1,
    LEXICAL_CONTEXTUAL_V1,
    LEXICAL_MID_FINAL_V1,
    hidden_sequence_channel_widths,
    hidden_sequence_channels,
    hidden_sequence_schema,
)
from core.learning.semantic_program_corpus import (
    build_semantic_program_corpus,
    build_semantic_program_fork_join_corpus,
    build_semantic_program_sequence_binary_corpus,
    build_semantic_program_sequence_cataphoric_corpus,
    build_semantic_program_sequence_corpus,
    build_semantic_program_sequence_reserved_alias_corpus,
    build_semantic_program_sequence_role_binding_corpus,
)
from core.learning.semantic_program_feature_materialization import (
    FAMILY_FEATURE_CONFIG_SCHEMA,
    FORK_JOIN_CORPUS_KIND,
    FORK_JOIN_FACTORIAL_CORPUS_KIND,
    FORK_JOIN_SOURCE_ORDER_CORPUS_KIND,
    SEQUENCE_BINARY_CHAIN_CORPUS_KIND,
    SEQUENCE_CATAPHORIC_CORPUS_KIND,
    SEQUENCE_CHAIN_CORPUS_KIND,
    SEQUENCE_RESERVED_ALIAS_CORPUS_KIND,
    SEQUENCE_ROLE_BINDING_CORPUS_KIND,
    SemanticFeatureConfig,
    SemanticFeatureMaterializationError,
    build_semantic_program_corpus_for_config,
    load_semantic_feature_bundle,
    load_standard_semantic_feature_bundle,
    materialize_semantic_program_features,
    offset_tokenizer_for_worker,
    select_bounded_semantic_examples,
    tokenize_with_offsets,
)
from tools import materialize_semantic_program_features as materializer_cli


def _sha(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def test_hidden_sequence_channel_widths_follow_the_versioned_packing_contract() -> None:
    assert hidden_sequence_channel_widths(FINAL_HIDDEN_V1, 5120) == (5120,)
    assert hidden_sequence_channel_widths(LEXICAL_CONTEXTUAL_V1, 10240) == (
        5120,
        5120,
    )
    with pytest.raises(ValueError, match="divisible by two"):
        hidden_sequence_channel_widths(LEXICAL_CONTEXTUAL_V1, 10241)
    assert hidden_sequence_channel_widths(LEXICAL_MID_FINAL_V1, 15360) == (
        5120,
        5120,
        5120,
    )
    with pytest.raises(ValueError, match="divisible by three"):
        hidden_sequence_channel_widths(LEXICAL_MID_FINAL_V1, 15361)


def test_materializer_cli_accepts_shared_hidden_representation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "materialize_semantic_program_features.py",
            "--model",
            "/tmp/model",
            "--output",
            "/tmp/output",
            "--representation",
            LEXICAL_MID_FINAL_V1,
            "--corpus-kind",
            SEQUENCE_BINARY_CHAIN_CORPUS_KIND,
        ],
    )

    arguments = materializer_cli._arguments()
    assert arguments.representation == LEXICAL_MID_FINAL_V1
    assert arguments.corpus_kind == SEQUENCE_BINARY_CHAIN_CORPUS_KIND


class _CharacterTokenizer:
    def __call__(self, text: str, **kwargs):
        assert kwargs["add_special_tokens"] is False
        return {
            "input_ids": [ord(character) for character in text],
            "offset_mapping": [(index, index + 1) for index in range(len(text))],
        }


class _FeatureClient:
    def __init__(self, checkpoint: Path) -> None:
        self.checkpoint = checkpoint
        self.calls = 0

    async def encode_hidden_sequence(
        self,
        text: str,
        *,
        timeout_s: float = 8.0,
        representation: str = "final_hidden_v1",
    ):
        assert timeout_s == 120.0
        assert representation == FINAL_HIDDEN_V1
        self.calls += 1
        token_ids = [ord(character) for character in text]
        states = np.zeros((len(token_ids), 8), dtype=np.float32)
        for index, token_id in enumerate(token_ids):
            states[index, token_id % states.shape[1]] = 1.0
        payload = states.astype("<f4", copy=False).tobytes(order="C")
        basis = {
            "worker_model_path": str(self.checkpoint),
            "worker_pid": 991,
            "worker_boot_id": "boot-991",
            "worker_source_sha256": "7" * 64,
        }
        return {
            "token_ids": token_ids,
            "hidden_states": states,
            "receipt": {
                "schema": hidden_sequence_schema(representation),
                "request_id": f"request-{self.calls}",
                "action": "encode_hidden_sequence",
                "input_char_count": len(text),
                "token_count": len(token_ids),
                "hidden_size": 8,
                "hidden_state_bytes": len(payload),
                "hidden_state_sha256": hashlib.sha256(payload).hexdigest(),
                "transport": "packed_float32_le",
                "limits": {
                    "max_input_chars": 4096,
                    "max_tokens": 512,
                    "max_hidden_size": 32768,
                },
                "model_basis": basis,
                "representation": representation,
                "channels": list(hidden_sequence_channels(representation)),
                "forward_passes": 1,
                "causal_full_sequence": True,
                "sampling": False,
                "generated_tokens": 0,
                "generated_text": False,
            },
        }


def _lane_receipt(checkpoint: Path) -> dict[str, object]:
    body: dict[str, object] = {
        "schema": "aura.mlx_model_lane_ownership.v1",
        "exclusive": True,
        "owner_id": "mlx:test:materializer",
        "fencing_token": 23,
        "terminal_receipt_id": "terminal-23",
        "model_path": os.path.realpath(checkpoint),
        "campaign_pid": os.getpid(),
        "worker_pid": 991,
        "worker_boot_id": "boot-991",
    }
    return {**body, "receipt_sha256": _sha(body)}


def _tokenizer_identity(checkpoint: Path) -> dict[str, object]:
    body: dict[str, object] = {
        "schema": "aura.semantic_program_tokenizer_identity.v1",
        "checkpoint_path": str(checkpoint),
        "files": {"tokenizer.json": {"bytes": 1, "sha256": "9" * 64}},
        "offsets_required": True,
        "generated_text": False,
    }
    return {**body, "identity_sha256": _sha(body)}


def test_selection_retains_every_construction_topology_and_operation_pair() -> None:
    corpus = build_semantic_program_corpus(
        seed=271828,
        examples_per_operation_pair=2,
    )
    selected = select_bounded_semantic_examples(corpus, max_examples=576)

    cells = {
        (
            item.construction_id,
            item.topology_id,
            tuple(step.instruction.op for step in item.instructions),
        )
        for item in selected
    }
    assert len(selected) == 576
    assert len(cells) == 576
    with pytest.raises(ValueError, match="factorial corpus cell"):
        select_bounded_semantic_examples(corpus, max_examples=575)


def test_offset_tokenization_matches_worker_without_special_tokens() -> None:
    token_ids, offsets = tokenize_with_offsets(_CharacterTokenizer(), "add 2 and 3")

    assert token_ids == [ord(character) for character in "add 2 and 3"]
    assert offsets == [(index, index + 1) for index in range(11)]


def test_offset_tokenizer_uses_the_wrapper_tokenizer_surface() -> None:
    underlying = _CharacterTokenizer()
    wrapper = type("Wrapper", (), {"_tokenizer": underlying})()

    assert offset_tokenizer_for_worker(wrapper) is underlying


def test_materialization_publishes_only_after_cpu_reload(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model"
    checkpoint.mkdir()
    output = tmp_path / "features"
    corpus = build_semantic_program_corpus(
        seed=271828,
        examples_per_operation_pair=1,
    )
    client = _FeatureClient(checkpoint)
    result = asyncio.run(
        materialize_semantic_program_features(
            client=client,
            tokenizer=_CharacterTokenizer(),
            checkpoint=checkpoint,
            output_directory=output,
            corpus=corpus,
            config=SemanticFeatureConfig(),
            lane_ownership_receipt=_lane_receipt(checkpoint),
            tokenizer_identity=_tokenizer_identity(checkpoint),
        )
    )

    assert result.complete
    assert result.completed_examples == 576
    assert client.calls == 576
    bundle = load_semantic_feature_bundle(output, expected_examples=corpus)
    assert len(bundle.examples) == 576
    assert bundle.manifest["split_counts"] == {
        "test": 128,
        "train": 320,
        "validation": 128,
    }
    assert not any(item.metadata.get("source_text") for item in bundle.examples)


def test_standard_loader_reconstructs_the_manifest_seed(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model"
    checkpoint.mkdir()
    output = tmp_path / "fresh-features"
    config = SemanticFeatureConfig(seed=314159)
    corpus = build_semantic_program_corpus(
        seed=config.seed,
        examples_per_operation_pair=config.examples_per_operation_pair,
    )
    asyncio.run(
        materialize_semantic_program_features(
            client=_FeatureClient(checkpoint),
            tokenizer=_CharacterTokenizer(),
            checkpoint=checkpoint,
            output_directory=output,
            corpus=corpus,
            config=config,
            lane_ownership_receipt=_lane_receipt(checkpoint),
            tokenizer_identity=_tokenizer_identity(checkpoint),
        )
    )

    bundle = load_standard_semantic_feature_bundle(output)

    assert bundle.manifest["config"]["seed"] == 314159
    assert len(bundle.examples) == 576


def test_standard_loader_reconstructs_declared_fork_join_family(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "model"
    checkpoint.mkdir()
    output = tmp_path / "fork-join-features"
    config = SemanticFeatureConfig(
        seed=161803,
        corpus_kind=FORK_JOIN_CORPUS_KIND,
        schema=FAMILY_FEATURE_CONFIG_SCHEMA,
    )
    corpus = build_semantic_program_fork_join_corpus(seed=config.seed)
    asyncio.run(
        materialize_semantic_program_features(
            client=_FeatureClient(checkpoint),
            tokenizer=_CharacterTokenizer(),
            checkpoint=checkpoint,
            output_directory=output,
            corpus=corpus,
            config=config,
            lane_ownership_receipt=_lane_receipt(checkpoint),
            tokenizer_identity=_tokenizer_identity(checkpoint),
        )
    )

    bundle = load_standard_semantic_feature_bundle(output)

    assert bundle.manifest["config"]["corpus_kind"] == FORK_JOIN_CORPUS_KIND
    assert len(bundle.examples) == 576


def test_source_order_fork_join_family_declares_observable_register_identity(
    tmp_path: Path,
) -> None:
    config = SemanticFeatureConfig(
        seed=161803,
        corpus_kind=FORK_JOIN_SOURCE_ORDER_CORPUS_KIND,
        schema=FAMILY_FEATURE_CONFIG_SCHEMA,
    )

    corpus = build_semantic_program_corpus_for_config(config)

    assert len(corpus) == 576
    assert all(
        list(example.input_spans) == sorted(example.input_spans, key=lambda span: span.start)
        for example in corpus
    )


def test_factorial_fork_join_family_reconstructs_from_declared_config() -> None:
    config = SemanticFeatureConfig(
        seed=2718281,
        max_examples=1296,
        corpus_kind=FORK_JOIN_FACTORIAL_CORPUS_KIND,
        schema=FAMILY_FEATURE_CONFIG_SCHEMA,
    )

    corpus = build_semantic_program_corpus_for_config(config)

    assert len(corpus) == 1296
    assert {
        split: sum(item.split == split for item in corpus)
        for split in (
            "train",
            "validation",
            "test",
        )
    } == {"train": 432, "validation": 432, "test": 432}


def test_sequence_family_reconstructs_from_declared_config() -> None:
    config = SemanticFeatureConfig(
        seed=1414213,
        examples_per_operation_pair=2,
        max_examples=540,
        corpus_kind=SEQUENCE_CHAIN_CORPUS_KIND,
        schema=FAMILY_FEATURE_CONFIG_SCHEMA,
    )

    corpus = build_semantic_program_corpus_for_config(config)

    assert corpus == build_semantic_program_sequence_corpus(
        seed=1414213,
        examples_per_operation_pair=2,
    )
    assert {
        split: sum(item.split == split for item in corpus)
        for split in (
            "train",
            "validation",
            "test",
        )
    } == {"train": 180, "validation": 180, "test": 180}


def test_sequence_binary_family_reconstructs_from_declared_config() -> None:
    config = SemanticFeatureConfig(
        seed=2236067,
        examples_per_operation_pair=2,
        max_examples=144,
        corpus_kind=SEQUENCE_BINARY_CHAIN_CORPUS_KIND,
        schema=FAMILY_FEATURE_CONFIG_SCHEMA,
    )

    corpus = build_semantic_program_corpus_for_config(config)

    assert corpus == build_semantic_program_sequence_binary_corpus(
        seed=2236067,
        examples_per_operation_pair=2,
    )
    assert {
        split: sum(item.split == split for item in corpus)
        for split in ("train", "validation", "test")
    } == {"train": 48, "validation": 48, "test": 48}


def test_sequence_reserved_alias_family_reconstructs_from_declared_config() -> None:
    config = SemanticFeatureConfig(
        seed=2449489,
        examples_per_operation_pair=2,
        max_examples=144,
        corpus_kind=SEQUENCE_RESERVED_ALIAS_CORPUS_KIND,
        schema=FAMILY_FEATURE_CONFIG_SCHEMA,
    )

    corpus = build_semantic_program_corpus_for_config(config)

    assert corpus == build_semantic_program_sequence_reserved_alias_corpus(
        seed=2449489,
        examples_per_operation_pair=2,
    )
    assert {
        split: sum(item.split == split for item in corpus)
        for split in ("train", "validation", "test")
    } == {"train": 48, "validation": 48, "test": 48}


def test_sequence_cataphoric_family_reconstructs_from_declared_config() -> None:
    config = SemanticFeatureConfig(
        seed=2653589,
        examples_per_operation_pair=2,
        max_examples=144,
        corpus_kind=SEQUENCE_CATAPHORIC_CORPUS_KIND,
        schema=FAMILY_FEATURE_CONFIG_SCHEMA,
    )

    corpus = build_semantic_program_corpus_for_config(config)

    assert corpus == build_semantic_program_sequence_cataphoric_corpus(
        seed=2653589,
        examples_per_operation_pair=2,
    )
    assert {
        split: sum(item.split == split for item in corpus)
        for split in ("train", "validation", "test")
    } == {"train": 48, "validation": 48, "test": 48}


def test_sequence_role_binding_family_reconstructs_from_declared_config() -> None:
    config = SemanticFeatureConfig(
        seed=2828427,
        examples_per_operation_pair=2,
        max_examples=144,
        corpus_kind=SEQUENCE_ROLE_BINDING_CORPUS_KIND,
        schema=FAMILY_FEATURE_CONFIG_SCHEMA,
    )

    corpus = build_semantic_program_corpus_for_config(config)

    assert corpus == build_semantic_program_sequence_role_binding_corpus(
        seed=2828427,
        examples_per_operation_pair=2,
    )
    assert {
        split: sum(item.split == split for item in corpus)
        for split in ("train", "validation", "test")
    } == {"train": 48, "validation": 48, "test": 48}


def test_sequence_feature_bundle_round_trips_nested_exact_values(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model"
    checkpoint.mkdir()
    output = tmp_path / "sequence-features"
    config = SemanticFeatureConfig(
        seed=1414213,
        examples_per_operation_pair=1,
        max_examples=270,
        corpus_kind=SEQUENCE_CHAIN_CORPUS_KIND,
        schema=FAMILY_FEATURE_CONFIG_SCHEMA,
    )
    corpus = build_semantic_program_corpus_for_config(config)
    client = _FeatureClient(checkpoint)

    result = asyncio.run(
        materialize_semantic_program_features(
            client=client,
            tokenizer=_CharacterTokenizer(),
            checkpoint=checkpoint,
            output_directory=output,
            corpus=corpus,
            config=config,
            lane_ownership_receipt=_lane_receipt(checkpoint),
            tokenizer_identity=_tokenizer_identity(checkpoint),
        )
    )
    bundle = load_standard_semantic_feature_bundle(output)

    assert result.complete
    assert len(bundle.examples) == 270
    assert all(isinstance(item.metadata["inputs"][0], list) for item in bundle.examples)

    calls_before_resume = client.calls
    resumed = asyncio.run(
        materialize_semantic_program_features(
            client=client,
            tokenizer=_CharacterTokenizer(),
            checkpoint=checkpoint,
            output_directory=output,
            corpus=corpus,
            config=config,
            lane_ownership_receipt=_lane_receipt(checkpoint),
            tokenizer_identity=_tokenizer_identity(checkpoint),
        )
    )
    status = json.loads((output / "status.json").read_text(encoding="ascii"))

    assert resumed.reason == "already_complete"
    assert client.calls == calls_before_resume
    assert status["complete"] is True
    assert status["reason"] == "already_complete"


def test_bundle_rejects_record_tampering(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model"
    checkpoint.mkdir()
    output = tmp_path / "features"
    corpus = build_semantic_program_corpus(
        seed=271828,
        examples_per_operation_pair=1,
    )
    asyncio.run(
        materialize_semantic_program_features(
            client=_FeatureClient(checkpoint),
            tokenizer=_CharacterTokenizer(),
            checkpoint=checkpoint,
            output_directory=output,
            corpus=corpus,
            config=SemanticFeatureConfig(),
            lane_ownership_receipt=_lane_receipt(checkpoint),
            tokenizer_identity=_tokenizer_identity(checkpoint),
        )
    )
    record = next(output.glob("*.spf"))
    payload = bytearray(record.read_bytes())
    payload[-1] ^= 1
    record.write_bytes(payload)

    with pytest.raises(SemanticFeatureMaterializationError):
        load_semantic_feature_bundle(output, expected_examples=corpus)
