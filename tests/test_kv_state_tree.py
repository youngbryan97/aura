"""Contract tests for SPARK-035 KV lineage and rejected-state isolation."""

from __future__ import annotations

import copy
import hashlib
import json

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

from mlx_lm.models.cache import ArraysCache, BatchKVCache, make_prompt_cache  # noqa: E402
from mlx_lm.models.qwen2 import Model, ModelArgs  # noqa: E402

from core.brain.llm.latent_cortex.engine import LatentCortexEngine  # noqa: E402
from core.brain.llm.latent_cortex.kv_state_tree import (  # noqa: E402
    KVStateTree,
    KVStateTreeError,
    validate_kv_state_tree_receipt,
)
from core.brain.llm.latent_cortex.recurrence import WindowRunner  # noqa: E402
from core.brain.llm.latent_cortex.types import (  # noqa: E402
    BranchConfig,
    ComputeBudget,
    CortexConfig,
    RecurrenceConfig,
    WorkspaceConfig,
)

N_LAYERS = 6
PROMPT = [5, 9, 17, 3, 42]


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical_sha256(value) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _rehash_receipt(receipt) -> None:
    receipt["receipt_sha256"] = _canonical_sha256(
        {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    )


class _FakeCache:
    """Capacity-backed cache with the same state/meta_state contract as MLX."""

    def __init__(self, *, offset: int, marker: float) -> None:
        self.keys = np.full((1, 2, max(8, offset), 4), marker, dtype=np.float32)
        self.values = np.full(
            (1, 2, max(8, offset), 4),
            marker + 0.5,
            dtype=np.float32,
        )
        self.offset = offset
        self._meta = ""

    @property
    def state(self):
        if self.offset == self.keys.shape[2]:
            return self.keys, self.values
        return (
            self.keys[:, :, : self.offset, :],
            self.values[:, :, : self.offset, :],
        )

    @state.setter
    def state(self, value):
        self.keys, self.values = value
        self.offset = int(self.keys.shape[2])

    @property
    def meta_state(self):
        return self._meta

    @meta_state.setter
    def meta_state(self, value):
        self._meta = value

    def append(self, *, count: int, marker: float) -> None:
        keys = np.concatenate(
            [
                self.state[0],
                np.full((1, 2, count, 4), marker, dtype=np.float32),
            ],
            axis=2,
        )
        values = np.concatenate(
            [
                self.state[1],
                np.full((1, 2, count, 4), marker + 0.5, dtype=np.float32),
            ],
            axis=2,
        )
        self.keys = keys
        self.values = values
        self.offset += count


def _fake_cache(layers: int = 4) -> list[_FakeCache]:
    return [_FakeCache(offset=3, marker=float(index)) for index in range(layers)]


def _validate(receipt, *, layers: int, branches: int = 2, final: bool = False):
    return validate_kv_state_tree_receipt(
        receipt,
        episode_id="episode-test",
        input_tokens_sha256=_sha("prompt"),
        n_layers=layers,
        expected_n_branches=branches,
        require_final=final,
    )


def test_rejected_slice_restores_exact_parent_objects_and_receipts_lineage():
    cache = _fake_cache()
    tree = KVStateTree(
        cache,
        n_layers=4,
        episode_id="episode-test",
        input_tokens_sha256=_sha("prompt"),
    )
    root_objects = [(item.keys, item.values) for item in cache]
    boundary = tree.capture_boundary(
        cache,
        parent_sha256=tree.root_sha256,
        branch_index=0,
        label="verified_branch_savepoint",
        authority="verifier",
        verified=True,
        latent_sha256=_sha("latent"),
    )
    transaction = tree.begin_speculation(
        cache,
        start=1,
        end=3,
        purpose="regenerate_from_prefix",
        branch_index=0,
        parent_sha256=boundary,
    )
    cache[1].append(count=2, marker=91.0)
    cache[2].append(count=2, marker=92.0)
    transaction.observe_mutation(cache)
    transaction.restore_parent(cache)
    event = transaction.reject_after_restore(cache)

    assert event["disposition"] == "rejected_pruned"
    assert event["appended_tokens_min"] == event["appended_tokens_max"] == 2
    assert event["child_cache_sha256"] != event["parent_cache_sha256"]
    assert all(
        item.keys is expected_keys and item.values is expected_values
        for item, (expected_keys, expected_values) in zip(
            cache,
            root_objects,
            strict=True,
        )
    )
    receipt = _validate(tree.receipt(), layers=4)
    assert receipt["regeneration_count"] == 1
    assert receipt["all_rejected_slices_pruned"] is True
    assert receipt["exact_parent_restoration"] is True
    assert receipt["no_rejected_child_reused"] is True


def test_batch_kv_cache_vector_coordinates_restore_exact_logical_boundary():
    cache = [BatchKVCache([0, 2]) for _ in range(3)]
    prefill_keys = mx.ones((2, 2, 5, 4))
    prefill_values = mx.ones((2, 2, 5, 4)) * 1.5
    for item in cache:
        item.update_and_fetch(prefill_keys, prefill_values)
    mx.eval(*(value for item in cache for value in (item.keys, item.values)))

    tree = KVStateTree(
        cache,
        n_layers=3,
        episode_id="episode-test",
        input_tokens_sha256=_sha("prompt"),
    )
    parent_keys = [item.keys for item in cache]
    parent_values = [item.values for item in cache]
    parent_offsets = [item.offset.tolist() for item in cache]

    transaction = tree.begin_speculation(
        cache,
        start=0,
        end=2,
        purpose="batch_counterfactual",
        branch_index=0,
        parent_sha256=tree.root_sha256,
    )
    branch_keys = mx.ones((2, 2, 2, 4)) * 2.0
    branch_values = mx.ones((2, 2, 2, 4)) * 2.5
    for item in cache[:2]:
        item.update_and_fetch(branch_keys, branch_values)
    transaction.observe_mutation()
    transaction.restore_parent()
    event = transaction.reject_after_restore()

    assert event["mutation_observed"] is True
    assert event["appended_tokens_min"] == 2
    assert event["appended_tokens_max"] == 2
    assert [item._idx for item in cache] == [5, 5, 5]
    assert [item.offset.tolist() for item in cache] == parent_offsets
    assert all(item.keys is expected for item, expected in zip(cache, parent_keys, strict=True))
    assert all(item.values is expected for item, expected in zip(cache, parent_values, strict=True))
    receipt = _validate(tree.receipt(), layers=3, branches=1)
    assert receipt["exact_parent_restoration"] is True
    assert receipt["restore_failure_count"] == 0


@pytest.mark.parametrize("hybrid", [False, True])
def test_recurrent_state_mutation_restores_without_inventing_token_growth(hybrid):
    recurrent = ArraysCache(2)
    recurrent[0] = mx.ones((1, 2, 4))
    recurrent[1] = mx.ones((1, 2, 4, 4))
    recurrent.left_padding = mx.array([0])
    recurrent.lengths = mx.array([5])
    cache = [recurrent]
    if hybrid:
        attention = BatchKVCache([0])
        attention.update_and_fetch(mx.ones((1, 2, 5, 4)), mx.ones((1, 2, 5, 4)))
        cache.append(attention)
    parent_state = list(recurrent.state)
    tree = KVStateTree(
        cache,
        n_layers=len(cache),
        episode_id="episode-test",
        input_tokens_sha256=_sha("prompt"),
    )
    transaction = tree.begin_speculation(
        cache,
        start=0,
        end=1,
        purpose="recurrent_state_update",
        branch_index=0,
        parent_sha256=tree.root_sha256,
    )
    recurrent[1] = recurrent[1] + 2
    recurrent.advance(2)
    transaction.observe_mutation()
    transaction.restore_parent()
    event = transaction.reject_after_restore()
    assert event["mutation_observed"] is True
    assert event["appended_tokens_min"] is None
    assert event["appended_tokens_max"] is None
    assert all(a is b for a, b in zip(recurrent.state, parent_state, strict=True))
    assert recurrent.lengths.tolist() == [5]
    assert recurrent.left_padding.tolist() == [0]
    receipt = _validate(tree.receipt(), layers=len(cache), branches=1)
    root = receipt["nodes"][0]
    assert root["min_offset"] == (5 if hybrid else None)
    assert root["max_offset"] == (5 if hybrid else None)
    assert receipt["exact_parent_restoration"] is True


@pytest.mark.parametrize("metadata_only", [False, True])
def test_cursorless_mutation_cannot_escape_declared_window(metadata_only):
    cache = [ArraysCache(2), ArraysCache(2)]
    for item in cache:
        item[0] = mx.ones((1, 2, 4))
        item[1] = mx.ones((1, 2, 4, 4))
        item.lengths = mx.array([5])
    tree = KVStateTree(
        cache,
        n_layers=2,
        episode_id="episode-test",
        input_tokens_sha256=_sha("prompt"),
    )
    transaction = tree.begin_speculation(
        cache,
        start=0,
        end=1,
        purpose="recurrent_state_update",
        branch_index=0,
        parent_sha256=tree.root_sha256,
    )
    if metadata_only:
        cache[1].advance(1)
    else:
        cache[1][0] = cache[1][0] + 1
    with pytest.raises(KVStateTreeError, match="escaped"):
        transaction.observe_mutation()
    transaction.restore_parent()


def test_attention_state_change_without_cursor_growth_is_still_a_mutation():
    cache = _fake_cache()
    tree = KVStateTree(
        cache,
        n_layers=4,
        episode_id="episode-test",
        input_tokens_sha256=_sha("prompt"),
    )
    transaction = tree.begin_speculation(
        cache,
        start=0,
        end=1,
        purpose="same_length_edit",
        branch_index=0,
        parent_sha256=tree.root_sha256,
    )
    cache[0].keys = cache[0].keys + 1
    transaction.observe_mutation()
    transaction.restore_parent()
    event = transaction.reject_after_restore()
    assert event["mutation_observed"] is True
    assert event["appended_tokens_max"] == 0
    _validate(tree.receipt(), layers=4, branches=1)


def test_legacy_receipt_remains_bound_to_its_token_growth_contract(monkeypatch):
    import core.brain.llm.latent_cortex.kv_state_tree as module

    with monkeypatch.context() as patch:
        patch.setattr(module, "KV_STATE_TREE_SCHEMA", "aura.rlc.kv_state_tree.v1")
        patch.setattr(module, "KV_STATE_NODE_SCHEMA", "aura.rlc.kv_state_node.v1")
        patch.setattr(module, "KV_STATE_EVENT_SCHEMA", "aura.rlc.kv_state_event.v1")
        cache = _fake_cache()
        tree = KVStateTree(
            cache,
            n_layers=4,
            episode_id="episode-test",
            input_tokens_sha256=_sha("prompt"),
        )
        transaction = tree.begin_speculation(
            cache,
            start=0,
            end=1,
            purpose="legacy_append",
            branch_index=0,
            parent_sha256=tree.root_sha256,
        )
        cache[0].append(count=1, marker=17.0)
        transaction.observe_mutation()
        transaction.restore_parent()
        transaction.reject_after_restore()
        receipt = tree.receipt()
    _validate(receipt, layers=4, branches=1)
    receipt["events"][0]["appended_tokens_min"] = 0
    receipt["events"][0]["appended_tokens_max"] = 0
    with pytest.raises(ValueError, match="mutation evidence"):
        _validate(receipt, layers=4, branches=1)


@pytest.mark.parametrize("offset", [None, -1, True, "5"])
def test_invalid_attention_cursor_is_not_treated_as_recurrent_state(offset):
    cache = _fake_cache()
    cache[0].offset = offset
    from core.brain.llm.latent_cortex.kv_state_tree import _cache_offsets

    with pytest.raises(KVStateTreeError, match="invalid offset"):
        _cache_offsets(cache, 0, len(cache))


def test_rejected_child_cannot_become_a_later_parent():
    cache = _fake_cache()
    tree = KVStateTree(
        cache,
        n_layers=4,
        episode_id="episode-test",
        input_tokens_sha256=_sha("prompt"),
    )
    transaction = tree.begin_speculation(
        cache,
        start=0,
        end=4,
        purpose="verifier_probe",
        branch_index=0,
        parent_sha256=tree.root_sha256,
    )
    for item in cache:
        item.append(count=1, marker=99.0)
    rejected_state = [item.state for item in cache]
    rejected_meta = [item.meta_state for item in cache]
    transaction.observe_mutation(cache)
    transaction.restore_parent(cache)
    transaction.reject_after_restore(cache)

    for item, state, meta in zip(cache, rejected_state, rejected_meta, strict=True):
        item.state = state
        item.meta_state = meta
    with pytest.raises(
        KVStateTreeError,
        match="does not descend from its declared parent",
    ):
        tree.capture_boundary(
            cache,
            parent_sha256=tree.root_sha256,
            branch_index=0,
            label="attacked_boundary",
            authority="test",
            verified=True,
            latent_sha256=_sha("attacked"),
        )


def _tiny_model() -> Model:
    args = ModelArgs(
        model_type="qwen2",
        hidden_size=48,
        num_hidden_layers=N_LAYERS,
        intermediate_size=96,
        num_attention_heads=4,
        rms_norm_eps=1e-6,
        vocab_size=96,
        num_key_value_heads=2,
        max_position_embeddings=256,
        rope_theta=10000.0,
    )
    model = Model(args)
    mx.eval(model.parameters())
    return model


def _prefill(model: Model, tokens: list[int]):
    from mlx_lm.models.base import create_attention_mask

    cache = make_prompt_cache(model)
    hidden = model.model.embed_tokens(mx.array([tokens]))
    mask = create_attention_mask(hidden, cache)
    for index, layer in enumerate(model.model.layers):
        hidden = layer(hidden, mask, cache[index])
    mx.eval(hidden)
    return cache, hidden


def test_real_qwen_rejected_work_cannot_change_regenerated_window():
    model = _tiny_model()
    cache, prompt_hidden = _prefill(model, PROMPT)
    tree = KVStateTree(
        cache,
        n_layers=N_LAYERS,
        episode_id="episode-test",
        input_tokens_sha256=_sha("prompt"),
    )
    runner = WindowRunner(model.model, ComputeBudget())
    runner.attach_kv_state_tree(tree)
    target = prompt_hidden[:, -3:, :]
    rejected = prompt_hidden[:, -2:, :] * mx.array(1.75)

    with runner.transaction_context(
        purpose="rejected_counterfactual",
        branch_index=0,
        parent_sha256=tree.root_sha256,
    ):
        runner.run(rejected, cache, 1, 5, persist=False)
    with runner.transaction_context(
        purpose="regenerate_from_prefix",
        branch_index=0,
        parent_sha256=tree.root_sha256,
    ):
        regenerated = runner.run(target, cache, 1, 5, persist=False)

    clean_cache, clean_prompt_hidden = _prefill(model, PROMPT)
    clean_tree = KVStateTree(
        clean_cache,
        n_layers=N_LAYERS,
        episode_id="episode-test",
        input_tokens_sha256=_sha("prompt"),
    )
    clean_runner = WindowRunner(model.model, ComputeBudget())
    clean_runner.attach_kv_state_tree(clean_tree)
    with clean_runner.transaction_context(
        purpose="regenerate_from_prefix",
        branch_index=0,
        parent_sha256=clean_tree.root_sha256,
    ):
        clean = clean_runner.run(
            clean_prompt_hidden[:, -3:, :],
            clean_cache,
            1,
            5,
            persist=False,
        )

    assert bool(mx.allclose(regenerated, clean, atol=0.0, rtol=0.0))
    receipt = _validate(tree.receipt(), layers=N_LAYERS)
    assert receipt["event_count"] == 2
    assert receipt["regeneration_count"] == 1
    assert all(event["parent_restored"] for event in receipt["events"])
    assert all(event["pruned"] for event in receipt["events"])


def test_full_real_episode_emits_valid_final_tree():
    model = _tiny_model()
    config = CortexConfig(
        workspace=WorkspaceConfig(n_slots=4, seed=13),
        recurrence=RecurrenceConfig(max_steps=3, min_steps=1),
        branches=BranchConfig(
            n_branches=2,
            isolation_steps=1,
            exchange_interval=2,
        ),
        prelude_frac=1 / 3,
        coda_frac=1 / 3,
        decode_max_tokens=4,
    )
    result = LatentCortexEngine(model, config=config).reason(token_ids=PROMPT)

    assert result.ok
    receipt = result.receipt.kv_state_tree
    validate_kv_state_tree_receipt(
        receipt,
        episode_id=result.receipt.episode_id,
        input_tokens_sha256=result.receipt.input_tokens_sha256,
        n_layers=N_LAYERS,
        expected_n_branches=2,
        require_final=True,
    )
    assert receipt["final_node_count"] == 1
    assert receipt["committed_event_count"] == 2
    assert receipt["rejected_event_count"] >= 4
    assert receipt["nodes"][-1]["label"] == "final_output_lane"
    assert receipt["nodes"][-1]["final"] is True


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda receipt: receipt["events"][0].__setitem__(
                "parent_cache_sha256",
                "0" * 64,
            ),
            "parent binding",
        ),
        (
            lambda receipt: receipt["events"][0].__setitem__("pruned", False),
            "not exactly pruned",
        ),
        (
            lambda receipt: receipt["nodes"][0].__setitem__(
                "cache_sha256",
                "f" * 64,
            ),
            "node hash differs",
        ),
        (
            lambda receipt: receipt.__setitem__(
                "receipt_sha256",
                "e" * 64,
            ),
            "receipt hash differs",
        ),
    ],
)
def test_validator_rejects_rehashed_or_unrehashed_lineage_tampering(mutate, match):
    cache = _fake_cache()
    tree = KVStateTree(
        cache,
        n_layers=4,
        episode_id="episode-test",
        input_tokens_sha256=_sha("prompt"),
    )
    transaction = tree.begin_speculation(
        cache,
        start=0,
        end=4,
        purpose="verifier_probe",
        branch_index=1,
        parent_sha256=tree.root_sha256,
    )
    for item in cache:
        item.append(count=1, marker=44.0)
    transaction.observe_mutation(cache)
    transaction.restore_parent(cache)
    transaction.reject_after_restore(cache)
    attacked = copy.deepcopy(tree.receipt())
    mutate(attacked)

    with pytest.raises(ValueError, match=match):
        _validate(attacked, layers=4)


def test_validator_requires_a_terminal_accepted_lane():
    cache = _fake_cache()
    tree = KVStateTree(
        cache,
        n_layers=4,
        episode_id="episode-test",
        input_tokens_sha256=_sha("prompt"),
    )
    transaction = tree.begin_speculation(
        cache,
        start=0,
        end=4,
        purpose="verifier_probe",
        branch_index=0,
        parent_sha256=tree.root_sha256,
    )
    for item in cache:
        item.append(count=1, marker=55.0)
    transaction.observe_mutation(cache)
    transaction.restore_parent(cache)
    transaction.reject_after_restore(cache)

    with pytest.raises(ValueError, match="no accepted final boundary"):
        _validate(tree.receipt(), layers=4, final=True)


def test_validator_rejects_fully_rehashed_failed_commit_lie():
    cache = _fake_cache()
    tree = KVStateTree(
        cache,
        n_layers=4,
        episode_id="episode-test",
        input_tokens_sha256=_sha("prompt"),
    )
    rejected = tree.begin_speculation(
        cache,
        start=0,
        end=4,
        purpose="verifier_probe",
        branch_index=0,
        parent_sha256=tree.root_sha256,
    )
    for item in cache:
        item.append(count=1, marker=66.0)
    rejected.observe_mutation(cache)
    rejected.restore_parent(cache)
    rejected.reject_after_restore(cache)
    final = tree.begin_speculation(
        cache,
        start=0,
        end=4,
        purpose="final_output_decode",
        branch_index=0,
        parent_sha256=tree.root_sha256,
    )
    for item in cache:
        item.append(count=1, marker=77.0)
    final.observe_mutation(cache)
    final.commit(
        label="final_output_lane",
        authority="user_visible_decode",
        latent_sha256=_sha("final"),
        final=True,
    )
    attacked = tree.receipt()
    event = attacked["events"][-1]
    event["execution_failed"] = True
    event["event_sha256"] = _canonical_sha256(
        {key: value for key, value in event.items() if key != "event_sha256"}
    )
    _rehash_receipt(attacked)

    with pytest.raises(ValueError, match="committed KV child binding differs"):
        _validate(attacked, layers=4, final=True)


def test_validator_rejects_uncommitted_final_boundary():
    cache = _fake_cache()
    tree = KVStateTree(
        cache,
        n_layers=4,
        episode_id="episode-test",
        input_tokens_sha256=_sha("prompt"),
    )
    rejected = tree.begin_speculation(
        cache,
        start=0,
        end=4,
        purpose="verifier_probe",
        branch_index=0,
        parent_sha256=tree.root_sha256,
    )
    for item in cache:
        item.append(count=1, marker=88.0)
    rejected.observe_mutation(cache)
    rejected.restore_parent(cache)
    rejected.reject_after_restore(cache)
    tree.capture_boundary(
        cache,
        parent_sha256=tree.root_sha256,
        branch_index=0,
        label="forged_final",
        authority="schedule_program",
        verified=True,
        latent_sha256=_sha("forged"),
        final=True,
    )

    with pytest.raises(
        ValueError,
        match="final node was not transactionally committed",
    ):
        _validate(tree.receipt(), layers=4, final=True)
