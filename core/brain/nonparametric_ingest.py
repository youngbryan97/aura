"""Ingestion — populate the non-parametric memory from Aura's TRUSTED knowledge.

The datastore's capacity is only as sound as what's in it, so we ingest only
verifier-clean answers and grounded beliefs — never a raw corpus that could teach
errors. Each (context -> answer) pair becomes a datastore entry: the model's hidden
state at the context is the key, the first answer token is the value.

Encoder-pluggable so this is fully testable without a GPU (a fake encoder in tests; the
real MLXEncoder lives in nonparametric_generation). Flag-gated (AURA_NONPARAMETRIC_INGEST),
bounded, deduplicated across runs.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from core.brain.nonparametric_binding import ingest_provenance  # noqa: F401
from core.brain.nonparametric_identity import EntryProvenance
from core.runtime.atomic_writer import atomic_write_text
from core.runtime.errors import record_degradation
from core.runtime.state_ownership import state_root

logger = logging.getLogger("Aura.NonParametricIngest")


def _flag_on(name: str, default: str = "0") -> bool:
    return str(os.getenv(name, default)).strip().lower() in {"1", "true", "on", "yes", "enabled"}


@runtime_checkable
class Encoder(Protocol):
    """Turns text into a datastore key + the first continuation token."""

    dim: int

    def encode_hidden(self, text: str) -> np.ndarray: ...
    def first_token(self, continuation: str) -> int: ...

    # Optional. When present, the store learns what each key PREDICTS in
    # readable form; without it the store is keys and integers, and the
    # generation path has no token text to recall.
    def decode_token(self, token_id: int) -> str: ...


#: Legacy receipts kept only 16 hex chars (64 bits). At datastore scale that is
#: collision-prone, and a collision silently classifies a DISTINCT trusted pair
#: as already committed — it is never ingested and nothing reports it. Receipts
#: are now the full digest; the legacy prefix is still honoured on read so
#: widening does not trigger a one-time re-ingest of everything.
_LEGACY_HASH_LEN = 16

#: Largest trusted store this will load into memory at once (64 MiB). Generous
#: for a legitimate local cache, bounded against an adversarial or runaway one.
_MAX_TRUSTED_STORE_BYTES = 64 * 1024 * 1024


def _decode_token(encoder: Any, token_id: int) -> str:
    """The readable text of one token, or "" if the encoder cannot say.

    THE STORE MUST KNOW WHAT ITS KEYS PREDICT.
    
    ingest_sequence stored `answer if position == start else ""` — the whole
    answer at the first position and nothing at every other one. The live
    5120-wide store built 2026-07-13 therefore held 1,689 keys of which 1,677
    carried no token text and 12 carried an entire answer as a single "token".
    Neither is what a per-token kNN store holds, and the worker's usability
    guard correctly refused to let it steer generation — 591 refusals in one
    session, with the faculty dark throughout.
    
    The token ids were always right. Only the decode was missing.
    """
    for name in ("decode_token", "decode", "detokenize"):
        method = getattr(encoder, name, None)
        if not callable(method):
            continue
        try:
            text = method(int(token_id))
        except (RuntimeError, AttributeError, TypeError, ValueError, IndexError):
            continue
        if isinstance(text, str) and text:
            return text
    tokenizer = getattr(encoder, "tokenizer", None)
    decode = getattr(tokenizer, "decode", None) if tokenizer is not None else None
    if callable(decode):
        try:
            text = decode([int(token_id)])
            if isinstance(text, str) and text:
                return text
        except (RuntimeError, AttributeError, TypeError, ValueError, IndexError):
            pass
    return ""


def _pair_hash(context: str, answer: str) -> str:
    return hashlib.sha256(f"{context}\x00{answer}".encode()).hexdigest()


def _valid_key(key: Any, expected_dim: Any = None) -> bool:
    """A datastore key must be a finite 1-D vector of the encoder's width.

    Only the batch path checked rank and row count; the fallback and single-pair
    paths checked nothing, so NaN, infinite, empty, or wrong-width vectors could
    enter the nearest-neighbour index and corrupt every later lookup.
    """
    try:
        arr = np.asarray(key, dtype=np.float32)
    except (TypeError, ValueError):
        return False
    if arr.ndim != 1 or arr.size == 0:
        return False
    if not np.all(np.isfinite(arr)):
        return False
    try:
        if expected_dim is not None and int(expected_dim) > 0 and arr.size != int(expected_dim):
            return False
    except (TypeError, ValueError):
        pass
    return True


def _valid_token_id(token_id: Any, vocab_size: Any = None) -> bool:
    """Continuation ids were written straight through with no range check."""
    try:
        value = int(token_id)
    except (TypeError, ValueError, OverflowError):
        return False
    if value < 0:
        return False
    try:
        if vocab_size is not None and int(vocab_size) > 0 and value >= int(vocab_size):
            return False
    except (TypeError, ValueError):
        pass
    return True


def collect_trusted_pairs(
    *, limit: int = 500, sources: list[tuple[Path, str]] | None = None
) -> list[tuple[str, str]]:
    """Read (context, answer) pairs from the persisted trusted stores (cache + traces).

    Decoupled from the live objects — reads the JSON on disk so ingestion can run as a
    background job without holding references. Only source-independent task types are
    included (math/code/logic), matching what the cache itself is willing to store.
    """
    pairs: list[tuple[str, str]] = []
    if sources is None:
        sources = [
            (Path(str(state_root() / "data/runtime/reasoning_solved_cache.json")), "entries"),
            (Path(str(state_root() / "data/runtime/reasoning_traces.json")), "traces"),
        ]
    for path, key in sources:
        if not path.exists():
            continue
        try:
            # The advertised `limit` bounds PAIRS, but read_text/json.loads
            # consume the entire file first — so a large or adversarial trace
            # file could exhaust memory before the bound had any effect. Refuse
            # oversized stores up front.
            size = path.stat().st_size
            if size > _MAX_TRUSTED_STORE_BYTES:
                record_degradation(
                    "nonparametric_ingest_collect",
                    ValueError(f"trusted store {path.name} is {size} bytes"),
                    severity="warning",
                    action="skipped oversized trusted store before loading it",
                )
                continue
            raw = json.loads(path.read_text(encoding="utf-8"))
            for entry in (raw.get(key, {}) or {}).values():
                obj = str(entry.get("objective", "")).strip()
                ans = str(entry.get("answer", "")).strip()
                if obj and ans:
                    pairs.append((obj, ans))
                if len(pairs) >= limit:
                    return pairs
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            record_degradation("nonparametric_ingest_collect", exc)
    return pairs


def collect_trusted_pairs_by_source(
    *, limit: int = 500, sources: list[tuple[Path, str]] | None = None
) -> list[tuple[str, list[tuple[str, str]]]]:
    """The same pairs, grouped by the store they came from.

    Revocation is per-source, so an entry has to remember which trusted
    store produced it. ``collect_trusted_pairs`` flattened that away, and
    every ingested entry then shared one indistinguishable origin.
    """
    if sources is None:
        sources = [
            (Path(str(state_root() / "data/runtime/reasoning_solved_cache.json")), "entries"),
            (Path(str(state_root() / "data/runtime/reasoning_traces.json")), "traces"),
        ]
    grouped: list[tuple[str, list[tuple[str, str]]]] = []
    remaining = max(0, int(limit))
    for path, key in sources:
        if remaining <= 0:
            break
        found = collect_trusted_pairs(limit=remaining, sources=[(path, key)])
        if found:
            grouped.append((path.stem, found))
            remaining -= len(found)
    return grouped


class NonParametricIngestor:
    """Encode trusted (context -> answer) pairs into the non-parametric datastore."""

    def __init__(
        self,
        memory: Any,
        *,
        provenance: EntryProvenance,
        dedup_path: str | Path | None = None,
    ) -> None:
        """``provenance`` is required, and its principal and source must be real.

        Every ``add`` from this class passed no provenance, so each entry
        the resident worker wrote defaulted to ``source_id="unattributed"``,
        ``trust=UNVERIFIED``, ``principal="anonymous"`` — the exact values
        the store's revocation and per-principal erasure need to work at
        all. A write nobody can attribute is a write nobody can take back.

        Refusing at construction rather than at ``add`` is deliberate: it
        makes an anonymous ingest impossible to reach by omission from any
        call site, present or future.
        """
        if not isinstance(provenance, EntryProvenance):
            raise TypeError(
                "NonParametricIngestor needs an EntryProvenance: an entry that "
                "cannot be attributed cannot be revoked or erased"
            )
        if not str(provenance.source_id or "").strip() or provenance.source_id == "unattributed":
            raise ValueError("ingest provenance needs a source that can later be revoked")
        if not str(provenance.principal or "").strip() or provenance.principal == "anonymous":
            raise ValueError("ingest provenance needs the principal the entry belongs to")
        self._provenance = provenance
        self._mem = memory
        self._dedup_path = Path(
            dedup_path or str(state_root() / "data/runtime/nonparametric_ingested.json")
        )
        # Guards the check-add-receipt sequence. Concurrent ingest workers could
        # both observe a hash absent, both commit the same pair, and race the
        # set update and file write. Atomic file replacement never made the
        # read-modify-write atomic.
        self._lock = threading.RLock()
        self._seen: set[str] = set()
        #: Receipt order, newest last — retention needs recency, and a set has
        #: no order to retain by.
        self._seen_order: list[str] = []
        self._load_seen()

    def _mark_seen(self, pair_hash: str) -> None:
        with self._lock:
            if pair_hash not in self._seen:
                self._seen.add(pair_hash)
                self._seen_order.append(pair_hash)

    def _is_seen(self, pair_hash: str) -> bool:
        """Receipt lookup that also honours legacy truncated receipts."""
        with self._lock:
            return pair_hash in self._seen or pair_hash[:_LEGACY_HASH_LEN] in self._seen

    def has_seen(self, context: str, answer: str) -> bool:
        """Return whether a trusted pair already has a committed ingest receipt."""

        return self._is_seen(_pair_hash(str(context or "").strip(), str(answer or "").strip()))

    def ingest_pair(self, context: str, answer: str, encoder: Encoder, *, weight: float = 1.0) -> bool:
        context, answer = str(context or "").strip(), str(answer or "").strip()
        if not context or not answer:
            return False
        h = _pair_hash(context, answer)
        if self._is_seen(h):
            return False
        try:
            key = encoder.encode_hidden(context)
            token_id = encoder.first_token(" " + answer)
        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
            record_degradation("nonparametric_ingest_encode", exc)
            return False
        if not _valid_key(key, getattr(encoder, "dim", None)) or not _valid_token_id(
            token_id, getattr(encoder, "vocab_size", None)
        ):
            record_degradation(
                "nonparametric_ingest_encode",
                ValueError("encoder produced a malformed key or token id"),
                severity="debug",
                action="rejected the pair rather than store an invalid key",
            )
            return False
        # One critical section over check-commit-receipt so two workers cannot
        # both commit the same pair.
        with self._lock:
            if self._is_seen(h):
                return False
            # `answer` is the continuation, not the token this key predicts;
            # storing it made a whole reply masquerade as one token.
            if not self._mem.add(
                key,
                token_id,
                token=_decode_token(encoder, token_id),
                weight=weight,
                provenance=self._provenance,
            ):
                return False
            self._mark_seen(h)
        return True

    def ingest_sequence(
        self,
        context: str,
        answer: str,
        encoder: Encoder,
        *,
        weight: float = 1.0,
        max_positions: int | None = None,
        max_sequence_tokens: int | None = None,
        should_continue: Callable[[], bool] | None = None,
    ) -> int:
        """Full-sequence ingestion: store (prefix-hidden → next token) for every answer position.

        Single-token ingestion only lets the model recall the FIRST answer token; generation
        of a multi-token answer needs a datastore entry at each position so the chain can be
        followed. Falls back to first-token ingestion if the encoder can't encode from ids.
        Returns the number of positions added (0 if skipped/duplicate).
        """
        context, answer = str(context or "").strip(), str(answer or "").strip()
        if not context or not answer:
            return 0
        enc_ids = getattr(encoder, "encode_hidden_ids", None)
        enc_tokens = getattr(encoder, "encode_tokens", None)
        if not callable(enc_ids) or not callable(enc_tokens):
            added = self.ingest_pair(context, answer, encoder, weight=weight)
            return 1 if added else 0
        h = _pair_hash(context, answer)
        if self._is_seen(h):
            return 0
        if should_continue is not None and not should_continue():
            return 0
        try:
            ctx_ids = list(enc_tokens(context))
            full_ids = list(enc_tokens(context + " " + answer))
        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
            record_degradation("nonparametric_ingest_tokenize", exc)
            return 0
        # Token alignment is an ASSUMPTION until it is checked. BPE can merge or
        # split across the inserted space, so `context` tokenized alone is not
        # always a prefix of `context + " " + answer`. When it is not,
        # len(ctx_ids)-1 is the wrong answer boundary and every key would be
        # bound to the wrong target — silently, and durably. Verify the prefix
        # and fall back to single-token ingestion rather than poison the store.
        if len(ctx_ids) > len(full_ids) or full_ids[: len(ctx_ids)] != ctx_ids:
            record_degradation(
                "nonparametric_ingest_alignment",
                ValueError("tokenizer is not prefix-stable across concatenation"),
                severity="debug",
                action="fell back to first-token ingestion for this pair",
            )
            return 1 if self.ingest_pair(context, answer, encoder, weight=weight) else 0
        start = max(0, len(ctx_ids) - 1)
        positions = list(range(start, len(full_ids) - 1))
        if not positions:
            return 0
        if max_sequence_tokens is not None and len(full_ids) > max(
            1, int(max_sequence_tokens)
        ):
            return 0
        if max_positions is not None and len(positions) > max(1, int(max_positions)):
            return 0

        prepared: list[tuple[np.ndarray, int, str]] = []
        batch_encoder = getattr(encoder, "encode_hidden_sequence_ids", None)
        try:
            if callable(batch_encoder):
                keys = np.asarray(batch_encoder(full_ids), dtype=np.float32)
                if keys.ndim != 2 or keys.shape[0] < len(full_ids):
                    raise ValueError(
                        "sequence encoder returned fewer hidden states than input tokens"
                    )
                prepared = [
                    (
                        keys[position],
                        int(full_ids[position + 1]),
                        # The token this key predicts, decoded. This path
                        # stored `answer if position == start else ""` — the
                        # whole answer at the first position and nothing at
                        # every other one — which is the defect
                        # `_decode_token` was written for, fixed in the
                        # compatibility branch below and not here. The real
                        # encoder has `encode_hidden_sequence_ids`, so
                        # production has always taken THIS branch: the live
                        # 5120-wide store holds 520 keys of which 490 carry no
                        # token text and the other 30 carry a whole arithmetic
                        # answer as one "token", and the worker's usability
                        # guard refuses to let it steer generation every turn.
                        _decode_token(encoder, int(full_ids[position + 1])),
                    )
                    for position in positions
                ]
            else:
                # Compatibility path for lightweight/test encoders.  Compute
                # every key before mutating the datastore so cancellation can
                # never publish a partial pair.
                for position in positions:
                    if should_continue is not None and not should_continue():
                        return 0
                    prepared.append(
                        (
                            np.asarray(
                                enc_ids(full_ids[: position + 1]),
                                dtype=np.float32,
                            ),
                            int(full_ids[position + 1]),
                            # The token this key predicts, decoded — not the
                            # whole answer at position 0 and nothing after.
                            _decode_token(encoder, int(full_ids[position + 1])),
                        )
                    )
        except (RuntimeError, AttributeError, TypeError, ValueError, IndexError) as exc:
            record_degradation("nonparametric_ingest_sequence", exc)
            return 0

        if should_continue is not None and not should_continue():
            return 0

        added = 0
        complete = True
        for key, token_id, token_text in prepared:
            # Cancellation is checked INSIDE the commit loop, not only before
            # it. Stopping mid-sequence leaves a partial pair, which must not be
            # receipted as done.
            if should_continue is not None and not should_continue():
                complete = False
                break
            if not _valid_key(key, getattr(encoder, "dim", None)) or not _valid_token_id(
                token_id, getattr(encoder, "vocab_size", None)
            ):
                # A wrong-width/NaN key or an out-of-range token id must not
                # enter the nearest-neighbour store.
                record_degradation(
                    "nonparametric_ingest_sequence",
                    ValueError("rejected malformed key/token during sequence ingest"),
                    severity="debug",
                    action="skipped position; pair left un-receipted for retry",
                )
                complete = False
                continue
            try:
                if self._mem.add(
                    key,
                    token_id,
                    token=token_text,
                    weight=weight,
                    provenance=self._provenance,
                ):
                    added += 1
                else:
                    complete = False
            except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                record_degradation("nonparametric_ingest_sequence", exc)
                complete = False
        # Only a FULLY committed sequence earns a receipt. Marking a partial
        # pair as seen made every future run skip the missing positions
        # permanently, leaving a half-ingested answer that can never be
        # completed.
        if added and complete:
            self._mark_seen(h)
        return added

    def sequence_within_budget(
        self,
        context: str,
        answer: str,
        encoder: Encoder,
        *,
        max_positions: int | None = None,
        max_sequence_tokens: int | None = None,
    ) -> bool:
        """Check sequence budgets without running a model forward or mutating memory."""

        context, answer = str(context or "").strip(), str(answer or "").strip()
        if not context or not answer:
            return False
        enc_tokens = getattr(encoder, "encode_tokens", None)
        if not callable(enc_tokens):
            return True
        try:
            ctx_ids = list(enc_tokens(context))
            full_ids = list(enc_tokens(context + " " + answer))
            positions = max(0, (len(full_ids) - 1) - max(0, len(ctx_ids) - 1))
            if positions <= 0:
                return False
            if max_sequence_tokens is not None and len(full_ids) > max(
                1, int(max_sequence_tokens)
            ):
                return False
            if max_positions is not None and positions > max(1, int(max_positions)):
                return False
            return True
        except (RuntimeError, AttributeError, TypeError, ValueError, OverflowError) as exc:
            record_degradation("nonparametric_ingest_budget", exc)
            return False

    def ingest_pairs(self, pairs: Iterable[tuple[str, str]], encoder: Encoder, *, weight: float = 1.0) -> int:
        n = 0
        for ctx, ans in pairs:
            if self.ingest_pair(ctx, ans, encoder, weight=weight):
                n += 1
        if n:
            try:
                memory_persisted = bool(self._mem.persist())
            except (OSError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
                record_degradation("nonparametric_ingest_persist", exc)
                memory_persisted = False
            if not memory_persisted:
                return 0
            if not self.persist_seen():
                return 0
        return n

    def ingest_from_trusted_stores(self, encoder: Encoder, *, limit: int = 500) -> int:
        """Pull verifier-clean pairs from the on-disk trusted stores and ingest them.

        Default ON since the July end-to-end proof; the background job that
        calls this stays pressure-guarded. Kill switch: AURA_NONPARAMETRIC_INGEST=0.
        """
        if not _flag_on("AURA_NONPARAMETRIC_INGEST", "1"):
            return 0
        return self.ingest_pairs(collect_trusted_pairs(limit=limit), encoder)

    def persist_seen(self) -> bool:
        """Durably publish dedup receipts after the datastore is durable."""

        return self._save_seen()

    # ── dedup persistence ───────────────────────────────────────────────────────
    def _load_seen(self) -> None:
        if not self._dedup_path.exists():
            return
        try:
            stored = json.loads(self._dedup_path.read_text(encoding="utf-8")).get("seen", [])
            # The file preserves order, so recency survives a restart.
            self._seen_order = [str(x) for x in stored if x]
            self._seen = set(self._seen_order)
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            record_degradation("nonparametric_ingest_load_seen", exc)

    def _save_seen(self) -> bool:
        try:
            # Keep the dedup ledger bounded by RECENCY. The old code sliced
            # list(set), whose iteration order is neither insertion nor recency
            # and differs between processes, so retention dropped an arbitrary
            # subset and randomly re-enabled old duplicates.
            with self._lock:
                self._seen_order = self._seen_order[-50_000:]
                self._seen = set(self._seen_order)
                seen = list(self._seen_order)
            atomic_write_text(self._dedup_path, json.dumps({"seen": seen}), encoding="utf-8")
            return True
        except (OSError, ValueError, TypeError) as exc:
            record_degradation("nonparametric_ingest_save_seen", exc)
            return False
