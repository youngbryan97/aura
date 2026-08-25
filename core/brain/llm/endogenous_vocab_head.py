"""z_Aura → Δlogits over the model's own vocabulary.

    L_final = L_LLM + α · (W · z_Aura + b)

One matrix. No decoder stack, no second autoregressive state machine, and no
attempt to write English from 74 numbers — that road ends at a worse language
model. What this can do is move the transformer's own distribution towards the
words compatible with the state Aura is actually in, and leave the enormous
learned problem of fluent English where it already works.

Three properties make it safe to put in a live decode loop:

* **Bound.** The bias is centred (a constant added to every logit changes
  nothing after softmax, so centring makes the magnitude bound mean what it
  says) and then clipped. A head cannot shout.
* **Plausibility-gated.** The bias lands only on tokens the model already
  finds plausible this step. It re-ranks inside the model's own safe set and
  cannot promote a token the model had ruled out, so the failure mode where a
  half-trained head produces word salad is closed by construction rather than
  by hoping the weights are small.
* **Bound to one model.** A head is trained against one tokenizer and one
  channel layout. Both are fingerprinted into the artifact, and a mismatch
  refuses to attach. Swapping the foundation model keeps z_Aura and discards
  the head, which is the honest outcome: the state is Aura's, the mapping into
  a vocabulary belongs to the model it was fitted against.

The head is also the reason the random projection can finally be retired from
anything that faces a person. Until a trained artifact exists on disk, this
module attaches nothing and says why.
"""

from __future__ import annotations

import hashlib
import hmac
import io
import json
import logging
import math
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from core.brain.llm.endogenous_state import (
    STATE_DIM,
    EndogenousState,
    layout_digest,
    semantics_digest,
)
from core.runtime.flags import FlagKind, declare

logger = logging.getLogger("Aura.EndogenousVocabHead")

#: Largest logit shift a trained head may apply to any single token, before α.
#: Chosen to be smaller than the typical gap between the top token and the
#: tail of the plausible set, so the head re-ranks near-ties and cannot
#: overturn a decided token on its own.
MAX_ABS_BIAS = 3.0

#: Below this share of live channels the state is mostly absence, and biasing
#: on absence is biasing on nothing.
MIN_COVERAGE = 0.10

#: Where a trained head lives unless told otherwise.
DEFAULT_HEAD_DIR = Path("artifacts/endogenous_language")

_HEAD_DIR_FLAG = declare(
    "AURA_ENDOGENOUS_HEAD_DIR",
    kind=FlagKind.STRING,
    default=str(DEFAULT_HEAD_DIR),
    description="Directory holding the trained endogenous vocabulary head",
    owner="core.brain.llm.endogenous_vocab_head",
)
_ALPHA_FLAG = declare(
    "AURA_ENDOGENOUS_ALPHA",
    kind=FlagKind.FLOAT,
    default=0.6,
    description="Strength of the endogenous term in L_LLM + alpha*L_Aura; zero disables the pathway",
    owner="core.brain.llm.endogenous_vocab_head",
)

#: The manifest is the commit record for exactly one serialized weight payload.
#: A loader must not infer compatibility for pre-binding artifacts.
HEAD_ARTIFACT_SCHEMA = "aura.endogenous_vocab_head.v1"


class HeadUnusableError(Exception):
    """The head cannot be used, and the reason is the message."""


def tokenizer_signature(tokenizer: Any) -> str:
    """Fingerprint the exact id→token mapping a head was fitted against.

    Vocabulary size alone would not do it: two tokenizers can agree on size
    and disagree on every id, which is precisely the case where a head would
    load, produce numbers, and bias the wrong words.
    """
    vocab: Mapping[str, int] | None = None
    for accessor in ("get_vocab", "vocab"):
        candidate = getattr(tokenizer, accessor, None)
        try:
            vocab = candidate() if callable(candidate) else candidate
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            logger.debug("tokenizer vocabulary accessor declined: %s", exc)
            vocab = None
        if isinstance(vocab, Mapping) and vocab:
            break
        vocab = None
    if not isinstance(vocab, Mapping) or not vocab:
        inner = getattr(tokenizer, "_tokenizer", None)
        if inner is not None and inner is not tokenizer:
            return tokenizer_signature(inner)
        raise HeadUnusableError("tokenizer exposes no vocabulary to fingerprint")
    digest = hashlib.sha256()
    digest.update(str(len(vocab)).encode("ascii"))
    for token, index in sorted(vocab.items(), key=lambda kv: kv[1]):
        digest.update(str(index).encode("ascii"))
        digest.update(b"\x00")
        digest.update(str(token).encode("utf-8", errors="replace"))
        digest.update(b"\x01")
    return digest.hexdigest()[:32]


@dataclass(frozen=True)
class BiasDecision:
    """Whether a bias was applied to one generation, and why."""

    applied: bool
    reason: str
    alpha: float = 0.0
    coverage: float = 0.0
    max_abs_delta: float = 0.0
    nonzero_tokens: int = 0
    layout: str = ""
    semantics: str = ""
    tokenizer: str = ""
    state_digest: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "applied": self.applied,
            "reason": self.reason,
            "alpha": round(self.alpha, 6),
            "state_coverage": round(self.coverage, 4),
            "max_abs_delta": round(self.max_abs_delta, 6),
            "nonzero_tokens": self.nonzero_tokens,
            "layout": self.layout,
            "semantics": self.semantics,
            "tokenizer": self.tokenizer,
            "state_digest": self.state_digest,
        }


@dataclass(frozen=True)
class EndogenousVocabHead:
    """A trained linear map from named cognitive state to vocabulary bias."""

    weights: np.ndarray  # (vocab, state_dim)
    bias: np.ndarray  # (vocab,)
    vocab_size: int
    layout: str
    #: Fingerprint of the DERIVATIONS behind the dimensions, not just their
    #: names. A head fitted before a probe was rewired matches the layout and
    #: was fitted to numbers that no longer mean the same thing.
    #:
    #: Defaults to "" — a head that never declared its provenance — and the
    #: guard in `decide` refuses that, so nothing attaches by omission.
    semantics: str = ""
    tokenizer: str = ""
    trained: bool = False
    report: Mapping[str, Any] = field(default_factory=dict)
    trained_at: float = 0.0

    def __post_init__(self) -> None:
        if self.weights.shape != (self.vocab_size, STATE_DIM):
            raise HeadUnusableError(
                f"head is {self.weights.shape}, layout wants ({self.vocab_size}, {STATE_DIM})"
            )
        if self.bias.shape != (self.vocab_size,):
            raise HeadUnusableError("head bias does not match its vocabulary")
        if not np.all(np.isfinite(self.weights)) or not np.all(np.isfinite(self.bias)):
            raise HeadUnusableError("head carries non-finite weights")

    # ── the map itself ────────────────────────────────────────────────────
    def delta_logits(self, state: EndogenousState) -> np.ndarray:
        """W·z + b, centred and clipped. Absent dimensions contribute nothing."""
        z = np.where(state.present, state.values, 0.0).astype(np.float64)
        delta = self.weights.astype(np.float64) @ z + self.bias.astype(np.float64)
        if not np.all(np.isfinite(delta)):
            raise HeadUnusableError("head produced a non-finite bias for this state")
        delta -= float(np.mean(delta))
        return np.clip(delta, -MAX_ABS_BIAS, MAX_ABS_BIAS)

    def decide(
        self,
        state: EndogenousState,
        *,
        tokenizer_sig: str,
        alpha: float,
        min_coverage: float = MIN_COVERAGE,
    ) -> tuple[np.ndarray | None, BiasDecision]:
        """Produce the bias for this turn, or refuse and say which check failed."""
        base = BiasDecision(
            applied=False,
            reason="",
            alpha=float(alpha),
            coverage=state.coverage,
            layout=self.layout,
            semantics=self.semantics,
            tokenizer=self.tokenizer,
            state_digest=state.digest,
        )
        if not self.trained:
            return None, _with(base, reason="head_untrained")
        if self.layout != layout_digest():
            return None, _with(base, reason="layout_mismatch")
        # Same names and ranges, different meaning behind them. A head
        # fitted while `temporal.past` was a copy of `memory.recall_hits`
        # matches the layout of a state where it is episodic recency, and
        # is applied to a number it never saw. An empty field is a head
        # from before this was recorded: it cannot be shown to match, so
        # it does not.
        if self.semantics != semantics_digest():
            return None, _with(base, reason="semantics_mismatch")
        if tokenizer_sig and self.tokenizer != tokenizer_sig:
            return None, _with(base, reason="tokenizer_mismatch")
        if not math.isfinite(float(alpha)) or float(alpha) <= 0.0:
            return None, _with(base, reason="alpha_disabled")
        if state.coverage < float(min_coverage):
            return None, _with(base, reason="state_coverage_below_floor")
        try:
            delta = self.delta_logits(state) * float(alpha)
        except HeadUnusableError as exc:
            return None, _with(base, reason=f"head_rejected:{exc}")
        nonzero = int(np.count_nonzero(np.abs(delta) > 1e-6))
        if nonzero == 0:
            return None, _with(base, reason="bias_is_flat")
        return delta, _with(
            base,
            applied=True,
            reason="applied",
            max_abs_delta=float(np.max(np.abs(delta))),
            nonzero_tokens=nonzero,
        )

    # ── persistence ───────────────────────────────────────────────────────
    def save(self, directory: str | Path = DEFAULT_HEAD_DIR, *, name: str = "vocab_head") -> Path:
        """Write one mutually bound weight/report generation through the gateway.

        A half-written weight file beside a complete report would load, produce
        numbers, and carry a report claiming they were measured. That is worse
        than no head at all.
        """
        from core.governance_context import local_internal_governed_scope
        from core.runtime.file_write_gateway import (
            FileWriteBatchEntry,
            get_file_write_gateway,
        )

        gateway = get_file_write_gateway()
        target = Path(directory)
        buffer = io.BytesIO()
        np.savez(
            buffer,
            weights=self.weights.astype(np.float32),
            bias=self.bias.astype(np.float32),
        )
        weights_payload = buffer.getvalue()
        weights_path = target / f"{name}.npz"
        manifest_path = target / f"{name}.json"
        manifest = {
            "schema": HEAD_ARTIFACT_SCHEMA,
            "vocab_size": int(self.vocab_size),
            "state_dim": int(STATE_DIM),
            "layout": self.layout,
            "semantics": self.semantics,
            "tokenizer": self.tokenizer,
            "trained": bool(self.trained),
            "trained_at": self.trained_at or time.time(),
            "max_abs_bias": MAX_ABS_BIAS,
            "weights_bytes": len(weights_payload),
            "weights_sha256": hashlib.sha256(weights_payload).hexdigest(),
            "report": dict(self.report),
        }
        manifest_payload = (
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
        entries = (
            FileWriteBatchEntry(weights_path, weights_payload),
            FileWriteBatchEntry(manifest_path, manifest_payload),
        )
        with local_internal_governed_scope("endogenous_vocab_head_persistence"):
            receipt = gateway.write_bytes_batch(
                entries,
                source="endogenous_vocab_head.persistence",
            )
        expected = {
            str(path.parent.resolve() / path.name): hashlib.sha256(payload).hexdigest()
            for path, payload in (
                (weights_path, weights_payload),
                (manifest_path, manifest_payload),
            )
        }
        if (
            not receipt.transaction_id
            or receipt.paths != tuple(expected)
            or dict(receipt.sha256) != expected
        ):
            raise HeadUnusableError(
                "head persistence receipt does not match the committed generation"
            )
        return weights_path

    @classmethod
    def load(
        cls, directory: str | Path = DEFAULT_HEAD_DIR, *, name: str = "vocab_head"
    ) -> EndogenousVocabHead:
        target = Path(directory)
        manifest_path = target / f"{name}.json"
        weights_path = target / f"{name}.npz"
        if not manifest_path.exists() or not weights_path.exists():
            raise HeadUnusableError("no trained head on disk")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise HeadUnusableError(f"head manifest unreadable: {exc}") from exc
        if manifest.get("schema") != HEAD_ARTIFACT_SCHEMA:
            raise HeadUnusableError("head manifest does not bind a supported weight generation")
        try:
            weights_payload = weights_path.read_bytes()
        except OSError as exc:
            raise HeadUnusableError(f"head weights unreadable: {exc}") from exc
        declared_size = manifest.get("weights_bytes")
        declared_digest = manifest.get("weights_sha256")
        if type(declared_size) is not int or declared_size < 0:
            raise HeadUnusableError("head manifest has no valid weight size binding")
        if (
            not isinstance(declared_digest, str)
            or len(declared_digest) != 64
            or any(character not in "0123456789abcdef" for character in declared_digest)
        ):
            raise HeadUnusableError("head manifest has no valid weight digest binding")
        if len(weights_payload) != declared_size:
            raise HeadUnusableError("head weight size does not match its manifest")
        if not hmac.compare_digest(
            hashlib.sha256(weights_payload).hexdigest(),
            declared_digest,
        ):
            raise HeadUnusableError("head weight digest does not match its manifest")
        try:
            with np.load(io.BytesIO(weights_payload), allow_pickle=False) as bundle:
                weights = np.asarray(bundle["weights"], dtype=np.float32)
                bias = np.asarray(bundle["bias"], dtype=np.float32)
        except (OSError, ValueError, KeyError) as exc:
            raise HeadUnusableError(f"head weights unreadable: {exc}") from exc
        return cls(
            weights=weights,
            bias=bias,
            vocab_size=int(manifest.get("vocab_size") or weights.shape[0]),
            layout=str(manifest.get("layout") or ""),
            semantics=str(manifest.get("semantics") or ""),
            tokenizer=str(manifest.get("tokenizer") or ""),
            trained=bool(manifest.get("trained")),
            report=manifest.get("report") or {},
            trained_at=float(manifest.get("trained_at") or 0.0),
        )

    def rebind(self, *, tokenizer: str) -> EndogenousVocabHead:
        """What survives a model swap, stated in code.

        The state does. The head does not: its columns index one model's
        vocabulary. Rebinding marks it untrained, so it cannot be used against
        the new model until it has been fitted there. That is what the
        fingerprint is for.
        """
        return EndogenousVocabHead(
            weights=self.weights,
            bias=self.bias,
            vocab_size=self.vocab_size,
            layout=self.layout,
            semantics=self.semantics,
            tokenizer=tokenizer,
            trained=False,
            report={**dict(self.report), "rebound_from": self.tokenizer},
            trained_at=self.trained_at,
        )


def _with(decision: BiasDecision, **changes: Any) -> BiasDecision:
    from dataclasses import replace as _replace

    return _replace(decision, **changes)


def untrained_head(vocab_size: int, tokenizer: str) -> EndogenousVocabHead:
    """A head shaped correctly and fitted to nothing — and it says so.

    Used by tests and by the trainer's own baseline. ``trained=False`` means
    ``decide`` refuses, so this can never reach a decode loop by accident.
    """
    return EndogenousVocabHead(
        weights=np.zeros((int(vocab_size), STATE_DIM), dtype=np.float32),
        bias=np.zeros(int(vocab_size), dtype=np.float32),
        vocab_size=int(vocab_size),
        layout=layout_digest(),
        semantics=semantics_digest(),
        tokenizer=tokenizer,
        trained=False,
    )


def head_directory() -> Path:
    return Path(str(_HEAD_DIR_FLAG.value() or DEFAULT_HEAD_DIR))


def alpha_from_env(default: float = 0.6) -> float:
    """Strength of the endogenous term. Zero disables the pathway entirely.

    Bounded here rather than trusted from the flag: the declared coercion
    turns a malformed value into the default, and this turns a hostile but
    well-formed one — a negative, an infinity, a thousand — into something a
    decode loop can survive.
    """
    try:
        value = float(_ALPHA_FLAG.value())
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(value) or value < 0.0:
        return 0.0
    return min(4.0, value)


__all__ = [
    "DEFAULT_HEAD_DIR",
    "HEAD_ARTIFACT_SCHEMA",
    "MAX_ABS_BIAS",
    "MIN_COVERAGE",
    "BiasDecision",
    "EndogenousVocabHead",
    "HeadUnusableError",
    "alpha_from_env",
    "head_directory",
    "tokenizer_signature",
    "untrained_head",
]
