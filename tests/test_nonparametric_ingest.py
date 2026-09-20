"""Tests for non-parametric memory ingestion (trusted knowledge -> datastore)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from core.brain.nonparametric_ingest import NonParametricIngestor, collect_trusted_pairs
from core.brain.nonparametric_memory import NonParametricMemory
from tests.nonparametric_support import entry_provenance


class FakeEncoder:
    dim = 8

    def __init__(self) -> None:
        # A real tokenizer can turn an id back into text, and the store is
        # useless without that: the live 5120-wide store held 1,689 keys of
        # which 1,677 carried no token text, because nothing ever decoded.
        # A fake encoder that cannot decode does not stand in for a real one.
        self._vocab: dict[int, str] = {}

    def encode_hidden(self, text: str) -> np.ndarray:
        seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
        return np.random.default_rng(seed).standard_normal(self.dim).astype(np.float32)

    def first_token(self, continuation: str) -> int:
        token_id = int(hashlib.sha256(continuation.encode()).hexdigest()[:4], 16) % 1000
        self._vocab.setdefault(token_id, continuation.strip())
        return token_id

    def decode_token(self, token_id: int) -> str:
        return self._vocab.get(int(token_id), f"<{int(token_id)}>")


class FakeSeqEncoder(FakeEncoder):
    """Adds the id-level hooks ingest_sequence needs (prefix-consistent tokenization)."""

    def encode_tokens(self, text: str) -> list[int]:
        ids = []
        for word in text.split():
            token_id = int(hashlib.sha1(word.encode()).hexdigest()[:6], 16) % 5000
            self._vocab.setdefault(token_id, word)
            ids.append(token_id)
        return ids

    def encode_hidden_ids(self, ids: list[int]) -> np.ndarray:
        seed = int(hashlib.sha256(str(list(ids)).encode()).hexdigest()[:8], 16)
        v = np.random.default_rng(seed).standard_normal(self.dim).astype(np.float32)
        return (v / np.linalg.norm(v)).astype(np.float32)

    def encode_hidden(self, text: str) -> np.ndarray:
        # consistent with encode_hidden_ids (as the real MLXEncoder is)
        return self.encode_hidden_ids(self.encode_tokens(text))


class FakeBatchSeqEncoder(FakeSeqEncoder):
    def __init__(self) -> None:
        super().__init__()
        self.batch_calls = 0
        self.prefix_calls = 0

    def encode_hidden_ids(self, ids: list[int]) -> np.ndarray:
        self.prefix_calls += 1
        return super().encode_hidden_ids(ids)

    def encode_hidden_sequence_ids(self, ids: list[int]) -> np.ndarray:
        self.batch_calls += 1
        return np.vstack(
            [
                super(FakeBatchSeqEncoder, self).encode_hidden_ids(ids[: index + 1])
                for index in range(len(ids))
            ]
        )


def _ingestor(tmp_path):
    mem = NonParametricMemory(dim=8, path=tmp_path / "npm")
    ing = NonParametricIngestor(
        mem, dedup_path=tmp_path / "seen.json", provenance=entry_provenance()
    )
    return mem, ing, FakeEncoder()


def test_ingest_pair_adds_and_recalls(tmp_path):
    mem, ing, enc = _ingestor(tmp_path)
    assert ing.ingest_pair("what is the capital of Vorth", "Myrrhal", enc) is True
    assert len(mem) == 1
    nbrs = mem.query(enc.encode_hidden("what is the capital of Vorth"))
    assert nbrs[0].token_id == enc.first_token(" Myrrhal")
    assert nbrs[0].token == "Myrrhal"


def test_ingest_pair_dedups(tmp_path):
    mem, ing, enc = _ingestor(tmp_path)
    assert ing.ingest_pair("q", "a", enc) is True
    assert ing.ingest_pair("q", "a", enc) is False  # same pair → skipped
    assert len(mem) == 1


def test_ingest_pair_rejects_empty(tmp_path):
    mem, ing, enc = _ingestor(tmp_path)
    assert ing.ingest_pair("", "a", enc) is False
    assert ing.ingest_pair("q", "", enc) is False
    assert len(mem) == 0


def test_ingest_pairs_counts_and_persists(tmp_path):
    mem, ing, enc = _ingestor(tmp_path)
    n = ing.ingest_pairs([("q1", "a1"), ("q2", "a2"), ("q1", "a1")], enc)  # last is dup
    assert n == 2
    assert len(mem) == 2
    # dedup ledger persisted
    assert (tmp_path / "seen.json").exists()


def test_ingest_pairs_does_not_publish_receipt_when_memory_persist_fails(tmp_path, monkeypatch):
    mem, ing, enc = _ingestor(tmp_path)
    monkeypatch.setattr(mem, "persist", lambda: False)

    assert ing.ingest_pairs([("not durable", "not receipted")], enc) == 0
    assert (tmp_path / "seen.json").exists() is False


def test_dedup_survives_new_ingestor_instance(tmp_path):
    mem, ing, enc = _ingestor(tmp_path)
    ing.ingest_pairs([("durable", "fact")], enc)
    ing2 = NonParametricIngestor(
        mem, dedup_path=tmp_path / "seen.json", provenance=entry_provenance()
    )
    assert ing2.ingest_pair("durable", "fact", enc) is False  # remembered across instances


def test_collect_trusted_pairs_reads_stores(tmp_path):
    cache = tmp_path / "cache.json"
    cache.write_text(
        json.dumps(
            {
                "entries": {
                    "h1": {"objective": "what is 2+2", "answer": "4"},
                    "h2": {"objective": "capital of Vorth", "answer": "Myrrhal"},
                }
            }
        ),
        encoding="utf-8",
    )
    pairs = collect_trusted_pairs(sources=[(Path(cache), "entries")])
    assert ("what is 2+2", "4") in pairs
    assert ("capital of Vorth", "Myrrhal") in pairs


def test_collect_trusted_pairs_missing_file_ok(tmp_path):
    assert collect_trusted_pairs(sources=[(tmp_path / "nope.json", "entries")]) == []


def test_ingest_from_trusted_stores_kill_switch(tmp_path, monkeypatch):
    """Default flipped ON after the July end-to-end proof; the kill switch
    must still stop ingestion cold."""
    mem, ing, enc = _ingestor(tmp_path)
    monkeypatch.setenv("AURA_NONPARAMETRIC_INGEST", "0")
    assert ing.ingest_from_trusted_stores(enc) == 0  # kill switch → no-op


def test_ingest_sequence_adds_every_answer_position(tmp_path):
    mem = NonParametricMemory(dim=8, path=tmp_path / "npm")
    ing = NonParametricIngestor(
        mem, dedup_path=tmp_path / "seen.json", provenance=entry_provenance()
    )
    enc = FakeSeqEncoder()
    added = ing.ingest_sequence("the keeper is named", "Tessaly the great", enc)
    assert added == 3  # one entry per answer word
    assert len(mem) == 3
    # the context's hidden recalls the first answer token
    nbrs = mem.query(enc.encode_hidden("the keeper is named"))
    assert nbrs[0].token_id == enc.encode_tokens("Tessaly")[0]


def test_ingest_sequence_dedups(tmp_path):
    mem = NonParametricMemory(dim=8, path=tmp_path / "npm")
    ing = NonParametricIngestor(
        mem, dedup_path=tmp_path / "seen.json", provenance=entry_provenance()
    )
    enc = FakeSeqEncoder()
    assert ing.ingest_sequence("q here", "a b c", enc) > 0
    assert ing.ingest_sequence("q here", "a b c", enc) == 0  # same fact → skipped


def test_ingest_sequence_falls_back_without_id_hooks(tmp_path):
    mem = NonParametricMemory(dim=8, path=tmp_path / "npm")
    ing = NonParametricIngestor(
        mem, dedup_path=tmp_path / "seen.json", provenance=entry_provenance()
    )
    enc = FakeEncoder()  # no encode_hidden_ids / encode_tokens
    assert ing.ingest_sequence("q", "a", enc) == 1  # degrades to first-token ingestion
    assert len(mem) == 1


def test_ingest_sequence_uses_one_full_sequence_forward(tmp_path):
    mem = NonParametricMemory(dim=8, path=tmp_path / "npm")
    ing = NonParametricIngestor(
        mem, dedup_path=tmp_path / "seen.json", provenance=entry_provenance()
    )
    enc = FakeBatchSeqEncoder()

    added = ing.ingest_sequence("the keeper is named", "Tessaly the great", enc)

    assert added == 3
    assert enc.batch_calls == 1
    assert enc.prefix_calls == 0


def test_ingest_sequence_budget_refuses_partial_pair(tmp_path):
    mem = NonParametricMemory(dim=8, path=tmp_path / "npm")
    ing = NonParametricIngestor(
        mem, dedup_path=tmp_path / "seen.json", provenance=entry_provenance()
    )
    enc = FakeBatchSeqEncoder()

    assert (
        ing.ingest_sequence(
            "the keeper is named",
            "Tessaly the great",
            enc,
            max_positions=2,
        )
        == 0
    )
    assert len(mem) == 0
    assert enc.batch_calls == 0
    assert (
        ing.ingest_sequence(
            "the keeper is named",
            "Tessaly the great",
            enc,
            max_positions=3,
        )
        == 3
    )


def test_sequence_budget_check_does_not_run_model_or_mutate_memory(tmp_path):
    mem = NonParametricMemory(dim=8, path=tmp_path / "npm")
    ing = NonParametricIngestor(
        mem, dedup_path=tmp_path / "seen.json", provenance=entry_provenance()
    )
    enc = FakeBatchSeqEncoder()

    assert (
        ing.sequence_within_budget(
            "the keeper is named",
            "Tessaly the great",
            enc,
            max_positions=2,
        )
        is False
    )
    assert enc.batch_calls == 0
    assert enc.prefix_calls == 0
    assert len(mem) == 0


def test_ingest_sequence_cancellation_cannot_publish_partial_pair(tmp_path):
    mem = NonParametricMemory(dim=8, path=tmp_path / "npm")
    ing = NonParametricIngestor(
        mem, dedup_path=tmp_path / "seen.json", provenance=entry_provenance()
    )
    enc = FakeBatchSeqEncoder()
    checks = 0

    def _continue_once() -> bool:
        nonlocal checks
        checks += 1
        return checks == 1

    assert (
        ing.ingest_sequence(
            "the keeper is named",
            "Tessaly the great",
            enc,
            should_continue=_continue_once,
        )
        == 0
    )
    assert enc.batch_calls == 1
    assert len(mem) == 0


# ── CP126 remediation regressions ───────────────────────────────────────────


class _NonPrefixEncoder(FakeSeqEncoder):
    """A tokenizer that merges across the inserted space, so `context` alone is
    NOT a prefix of `context + " " + answer` — the BPE reality the sequence
    boundary arithmetic silently assumed away."""

    def encode_tokens(self, text: str) -> list[int]:
        ids = super().encode_tokens(text)
        if len(ids) >= 2:
            # Merge the final two tokens. The merge therefore lands in a
            # different place for `context` than for `context + " " + answer`,
            # exactly as a real BPE merge across the inserted space does.
            ids = ids[:-2] + [(ids[-2] * 31 + ids[-1]) % 5000]
        return ids


def test_non_prefix_tokenizer_falls_back_instead_of_misaligning(tmp_path):
    """Misaligned positions would bind every key to the WRONG target, durably."""
    memory = NonParametricMemory(dim=8, path=tmp_path / "m.npz")
    ingestor = NonParametricIngestor(
        memory, dedup_path=tmp_path / "seen.json", provenance=entry_provenance()
    )

    added = ingestor.ingest_sequence(
        "what is two plus two", "the answer is four", _NonPrefixEncoder()
    )

    # Fell back to single-token ingestion rather than storing misaligned keys.
    assert added <= 1


def test_cancellation_midway_leaves_no_receipt(tmp_path):
    """A partial sequence must stay retryable: receipting it made every future
    run skip the missing positions permanently."""
    memory = NonParametricMemory(dim=8, path=tmp_path / "m.npz")
    ingestor = NonParametricIngestor(
        memory, dedup_path=tmp_path / "seen.json", provenance=entry_provenance()
    )
    calls = {"n": 0}

    def should_continue() -> bool:
        calls["n"] += 1
        return calls["n"] <= 3  # cancel partway through the commit loop

    ingestor.ingest_sequence(
        "context words here",
        "answer words follow along now",
        FakeSeqEncoder(),
        should_continue=should_continue,
    )

    assert ingestor.has_seen("context words here", "answer words follow along now") is False


def test_malformed_keys_and_token_ids_are_rejected(tmp_path):
    class BadKeyEncoder(FakeEncoder):
        def encode_hidden(self, text: str) -> np.ndarray:
            return np.array([float("nan")] * self.dim, dtype=np.float32)

    class BadTokenEncoder(FakeEncoder):
        def first_token(self, continuation: str) -> int:
            return -5

    memory = NonParametricMemory(dim=8, path=tmp_path / "m.npz")
    ingestor = NonParametricIngestor(
        memory, dedup_path=tmp_path / "seen.json", provenance=entry_provenance()
    )

    assert ingestor.ingest_pair("ctx", "ans", BadKeyEncoder()) is False
    assert ingestor.ingest_pair("ctx2", "ans2", BadTokenEncoder()) is False
    # Neither poisoned pair earned a receipt.
    assert ingestor.has_seen("ctx", "ans") is False
    assert ingestor.has_seen("ctx2", "ans2") is False


def test_legacy_truncated_receipts_are_still_honoured(tmp_path):
    """Widening the receipt must not re-ingest everything already committed."""
    import hashlib as _h

    ctx, ans = "legacy context", "legacy answer"
    legacy = _h.sha256(f"{ctx}\x00{ans}".encode()).hexdigest()[:16]
    dedup = tmp_path / "seen.json"
    dedup.write_text(json.dumps({"seen": [legacy]}), encoding="utf-8")

    memory = NonParametricMemory(dim=8, path=tmp_path / "m.npz")
    ingestor = NonParametricIngestor(memory, dedup_path=dedup, provenance=entry_provenance())

    assert ingestor.has_seen(ctx, ans) is True
    assert ingestor.ingest_pair(ctx, ans, FakeEncoder()) is False


def test_dedup_retention_keeps_the_most_recent(tmp_path):
    """Retention used list(set)[-N:], whose order is arbitrary and unstable."""
    memory = NonParametricMemory(dim=8, path=tmp_path / "m.npz")
    dedup = tmp_path / "seen.json"
    ingestor = NonParametricIngestor(memory, dedup_path=dedup, provenance=entry_provenance())

    for i in range(50):
        ingestor._mark_seen(f"hash-{i:04d}")
    assert ingestor.persist_seen() is True

    stored = json.loads(dedup.read_text(encoding="utf-8"))["seen"]
    assert stored == [f"hash-{i:04d}" for i in range(50)]  # insertion order preserved


def test_oversized_trusted_store_is_skipped_before_loading(tmp_path, monkeypatch):
    import core.brain.nonparametric_ingest as ingest_module

    store = tmp_path / "big.json"
    store.write_text(
        json.dumps({"entries": {"a": {"objective": "o", "answer": "a"}}}), encoding="utf-8"
    )
    monkeypatch.setattr(ingest_module, "_MAX_TRUSTED_STORE_BYTES", 4)

    assert collect_trusted_pairs(sources=[(store, "entries")]) == []


class TestBothPathsStoreTheTokenTheKeyPredicts:
    """The fix went to the branch production does not take.

    `_decode_token` was written because the live 5120-wide store held keys
    with no token text, and its docstring says so in capitals. It was wired
    into the compatibility branch only. The real MLXEncoder has
    `encode_hidden_sequence_ids`, so production has always taken the batch
    branch, which went on storing `answer if position == start else ""` — the
    whole answer at the first position and nothing at every other one. On
    2026-09-20 the live store held 520 keys of which 490 carried no token and
    the other 30 carried a whole arithmetic answer as one "token", and the
    worker refused to let it steer generation on every turn of every session.
    """

    def _entries(self, tmp_path, encoder):
        mem = NonParametricMemory(dim=8, path=tmp_path / "npm")
        ing = NonParametricIngestor(
            mem, dedup_path=tmp_path / "seen.json", provenance=entry_provenance()
        )
        added = ing.ingest_sequence("the keeper is named", "Tessaly the great", encoder)
        assert added == 3
        return [
            (neighbour_id, text)
            for neighbour_id, text in zip(mem._token_ids, mem._tokens, strict=True)
        ]

    def test_the_batch_path_decodes_every_position(self, tmp_path):
        encoder = FakeBatchSeqEncoder()

        entries = self._entries(tmp_path, encoder)

        assert encoder.batch_calls == 1, "this is the branch production takes"
        assert [text for _id, text in entries] == ["the", "great", "Tessaly"] or sorted(
            text for _id, text in entries
        ) == ["Tessaly", "great", "the"]
        assert all(text.strip() for _id, text in entries)
        assert all(
            text == encoder.decode_token(token_id) for token_id, text in entries
        )

    def test_neither_path_stores_the_whole_answer_as_one_token(self, tmp_path):
        """What the defect looked like, stated so it cannot come back."""
        for encoder in (FakeBatchSeqEncoder(), FakeSeqEncoder()):
            entries = self._entries(tmp_path / type(encoder).__name__, encoder)

            assert "Tessaly the great" not in [text for _id, text in entries]
            assert "" not in [text for _id, text in entries]

    def test_the_two_paths_agree(self, tmp_path):
        """The compatibility branch is the one that was already right."""
        batch = dict(self._entries(tmp_path / "batch", FakeBatchSeqEncoder()))
        prefix = dict(self._entries(tmp_path / "prefix", FakeSeqEncoder()))

        assert batch == prefix
