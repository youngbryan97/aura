"""Where z_Aura meets the decode loop.

The MLX worker runs in its own process and cannot reach the substrate, the
goal system, or anything else that lives in the parent. Earlier attempts at a
cross-process latent path failed on exactly that: a hook that resolved the
substrate from a container that was empty in the worker, forever. This one
does not resolve anything. The parent puts 74 floats on the job; the worker
reads them, multiplies them by a head it loads from disk once, and adds the
result to the model's logits inside the plausible set.

Per token the cost is one vector add and one mask, both already the shape of
the logits. The state does not change during a generation, so the bias is
computed once, before the first token, and nothing in the loop reads back a
device value.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from core.brain.llm.endogenous_state import EndogenousState
from core.brain.llm.endogenous_vocab_head import (
    BiasDecision,
    EndogenousVocabHead,
    HeadUnusableError,
    alpha_from_env,
    head_directory,
    tokenizer_signature,
)

logger = logging.getLogger("Aura.EndogenousDecode")

#: Job key the parent uses to ship z_Aura across the process boundary.
JOB_STATE_KEY = "endogenous_state"

#: Plausibility floor: a token must hold at least this share of the top
#: token's probability before the endogenous term may touch it.
DEFAULT_BETA = 0.1

_LOAD_LOCK = threading.Lock()
_CACHED: tuple[str, EndogenousVocabHead | None, str] | None = None


def load_head(directory: Path | None = None) -> tuple[EndogenousVocabHead | None, str]:
    """Load the trained head once per process; remember a refusal too.

    Remembering the refusal matters: without it a worker with no artifact
    would stat two files on every single generation, and the reason would be
    recomputed instead of reported.
    """
    global _CACHED
    target = Path(directory) if directory is not None else head_directory()
    key = str(target.resolve()) if target.exists() else str(target)
    with _LOAD_LOCK:
        if _CACHED is not None and _CACHED[0] == key:
            return _CACHED[1], _CACHED[2]
        try:
            head = EndogenousVocabHead.load(target)
            reason = "loaded"
        except HeadUnusableError as exc:
            head, reason = None, str(exc)
        except (OSError, ValueError) as exc:  # noqa: PERF203 - one-shot path
            head, reason = None, f"head_load_failed:{exc}"
        _CACHED = (key, head, reason)
    return head, reason


def reset_head_cache() -> None:
    """Forget the loaded head. For tests, and for a head retrained in place."""
    global _CACHED
    with _LOAD_LOCK:
        _CACHED = None


class EndogenousLogitBiasProcessor:
    """``L_final = L_LLM + α·(W·z + b)``, inside the model's plausible set.

    The bias vector is fixed for the whole generation. Nothing here is
    recomputed per token except the plausibility mask, which is two reductions
    over the logits the sampler was going to read anyway.
    """

    def __init__(self, delta: np.ndarray, *, beta: float = DEFAULT_BETA) -> None:
        self._delta_np = np.asarray(delta, dtype=np.float32).reshape(-1)
        self.beta = float(beta)
        self._delta_mx: Any | None = None
        self.calls = 0
        self.applied = 0

    @property
    def vocab_size(self) -> int:
        return int(self._delta_np.shape[0])

    def __call__(self, tokens: Any, logits: Any) -> Any:
        self.calls += 1
        try:
            import mlx.core as mx

            if logits.shape[-1] != self._delta_np.shape[0]:
                return logits
            if self._delta_mx is None:
                self._delta_mx = mx.array(self._delta_np)
            logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            threshold = math.log(max(self.beta, 1e-9)) + mx.max(logp, axis=-1, keepdims=True)
            mask = logp >= threshold
            self.applied += 1
            return mx.where(mask, logits + self._delta_mx, logits)
        except (ImportError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
            logger.debug("endogenous processor fell open: %s", exc)
            return logits

    def apply_numpy(self, logits: Any) -> np.ndarray:
        """The same arithmetic without MLX, so the contract is testable offline."""
        values = np.asarray(logits, dtype=np.float64).reshape(-1)
        if values.shape[0] != self._delta_np.shape[0] or not np.all(np.isfinite(values)):
            return values
        shifted = values - values.max()
        logp = shifted - math.log(float(np.sum(np.exp(shifted))))
        threshold = math.log(max(self.beta, 1e-9)) + float(np.max(logp))
        mask = logp >= threshold
        return np.where(mask, values + self._delta_np, values)


def build_endogenous_processor(
    tokenizer: Any,
    job: Mapping[str, Any],
    *,
    directory: Path | None = None,
    beta: float = DEFAULT_BETA,
) -> tuple[EndogenousLogitBiasProcessor | None, dict[str, Any]]:
    """Build the processor for one job, or explain in the receipt why not.

    Every path returns a receipt. A generation that ran without the pathway
    and a generation that ran with it are then distinguishable after the fact,
    which is the difference between a wired feature and a claimed one.
    """
    started = time.perf_counter()
    receipt: dict[str, Any] = {"pathway": "endogenous_vocab_bias"}

    payload = job.get(JOB_STATE_KEY) if isinstance(job, Mapping) else None
    if not payload:
        receipt["reason"] = "no_state_on_job"
        return None, receipt

    state = EndogenousState.from_payload(payload)
    if state is None:
        receipt["reason"] = "state_payload_rejected"
        return None, receipt

    head, load_reason = load_head(directory)
    if head is None:
        receipt["reason"] = f"no_head:{load_reason}"
        receipt["state_coverage"] = round(state.coverage, 4)
        return None, receipt

    try:
        signature = tokenizer_signature(tokenizer)
    except HeadUnusableError as exc:
        receipt["reason"] = f"tokenizer_unfingerprintable:{exc}"
        return None, receipt

    alpha = job.get("endogenous_alpha")
    alpha = alpha_from_env() if alpha is None else float(alpha)
    delta, decision = head.decide(state, tokenizer_sig=signature, alpha=alpha)
    receipt.update(decision.as_dict())
    receipt["build_ms"] = round((time.perf_counter() - started) * 1000.0, 3)
    if delta is None:
        receipt["reason"] = decision.reason
        return None, receipt
    receipt["reason"] = "applied"
    receipt["beta"] = float(beta)
    return EndogenousLogitBiasProcessor(delta, beta=beta), receipt


def decision_is_expected_absence(reason: str) -> bool:
    """Separate "correctly not attached" from "something is wrong".

    No artifact yet, alpha at zero, or a state nothing answered for are all
    the pathway behaving. A layout or tokenizer mismatch is not: it means a
    head is on disk that was fitted against something else.
    """
    return str(reason).split(":", 1)[0] in {
        "no_state_on_job",
        "no_head",
        "alpha_disabled",
        "state_coverage_below_floor",
        "head_untrained",
        "bias_is_flat",
    }


__all__ = [
    "DEFAULT_BETA",
    "JOB_STATE_KEY",
    "BiasDecision",
    "EndogenousLogitBiasProcessor",
    "build_endogenous_processor",
    "decision_is_expected_absence",
    "load_head",
    "reset_head_cache",
]
