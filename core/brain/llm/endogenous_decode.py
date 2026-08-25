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


_SIGNATURE_CACHE: tuple[Any, str] | None = None


def cached_tokenizer_signature(tokenizer: Any) -> str:
    """Fingerprint the tokenizer once per process, not once per generation.

    The fingerprint hashes every id-to-token pair, which is the point — two
    tokenizers can agree on size and disagree on every id. It is also a
    hundred thousand string operations, and this runs inside the path that
    builds a decode loop. A worker holds one tokenizer for its whole life, so
    a single slot is the whole cache; the tokenizer is held by strong
    reference so its identity cannot be reused by a later object.
    """
    global _SIGNATURE_CACHE
    with _LOAD_LOCK:
        cached = _SIGNATURE_CACHE
        if cached is not None and cached[0] is tokenizer:
            return cached[1]
    signature = tokenizer_signature(tokenizer)
    with _LOAD_LOCK:
        _SIGNATURE_CACHE = (tokenizer, signature)
    return signature


def reset_tokenizer_signature_cache() -> None:
    """Forget the fingerprint. For tests, and for a swapped tokenizer."""
    global _SIGNATURE_CACHE
    with _LOAD_LOCK:
        _SIGNATURE_CACHE = None


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
        signature = cached_tokenizer_signature(tokenizer)
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
    "cached_tokenizer_signature",
    "reset_head_cache",
    "reset_tokenizer_signature_cache",
]


# ──────────────────────────────────────────────────────────────────────────
# Health. A pathway nobody can see the state of is a pathway nobody can trust.
# ──────────────────────────────────────────────────────────────────────────

_HEALTH_LOCK = threading.Lock()
_HEALTH: dict[str, Any] = {
    "generations_seen": 0,
    "bias_applied": 0,
    "reasons": {},
    "last_receipt": {},
    "last_applied_at": 0.0,
    "unexpected_refusals": 0,
}


def observe_receipt(receipt: Mapping[str, Any] | None) -> None:
    """Fold one worker receipt into the parent's view of the pathway."""
    if not isinstance(receipt, Mapping) or not receipt:
        return
    reason = str(receipt.get("reason") or "unknown")
    with _HEALTH_LOCK:
        _HEALTH["generations_seen"] = int(_HEALTH["generations_seen"]) + 1
        counts: dict[str, int] = _HEALTH["reasons"]
        counts[reason] = int(counts.get(reason, 0)) + 1
        _HEALTH["last_receipt"] = dict(receipt)
        if receipt.get("applied"):
            _HEALTH["bias_applied"] = int(_HEALTH["bias_applied"]) + 1
            _HEALTH["last_applied_at"] = time.time()
        elif not decision_is_expected_absence(reason):
            _HEALTH["unexpected_refusals"] = int(_HEALTH["unexpected_refusals"]) + 1


def pathway_health() -> dict[str, Any]:
    """What the endogenous pathway has actually done, since this process began.

    ``unexpected_refusals`` is the field that matters: a head on disk that
    refuses to attach is a wiring fault, while no head at all is the pathway
    waiting for a fit.
    """
    with _HEALTH_LOCK:
        seen = int(_HEALTH["generations_seen"])
        applied = int(_HEALTH["bias_applied"])
        return {
            "generations_seen": seen,
            "bias_applied": applied,
            "applied_share": round(applied / seen, 4) if seen else 0.0,
            "reasons": dict(_HEALTH["reasons"]),
            "unexpected_refusals": int(_HEALTH["unexpected_refusals"]),
            "last_applied_at": float(_HEALTH["last_applied_at"]),
            "last_receipt": dict(_HEALTH["last_receipt"]),
        }


def reset_pathway_health() -> None:
    with _HEALTH_LOCK:
        _HEALTH.update(
            generations_seen=0,
            bias_applied=0,
            reasons={},
            last_receipt={},
            last_applied_at=0.0,
            unexpected_refusals=0,
        )


__all__ += ["observe_receipt", "pathway_health", "reset_pathway_health"]


def register_pathway_health() -> None:
    """Publish the pathway's health where the runtime can read it.

    Published rather than imported: ``core/runtime`` is forbidden from
    reaching ``core.brain`` by its DEPS rule, and a health block that needed
    that edge would be a layering violation dressed as observability.
    """
    from core.container import ServiceContainer
    from core.runtime.service_registry import register_runtime_service

    # Both registries, because neither is a superset of the other and the
    # registry sink is only installed once a runtime owns the process. A
    # provider registered in one place and read from the other is the exact
    # half-wiring this pathway is supposed to make visible.
    ServiceContainer.register_instance(
        "endogenous_language_health", pathway_health, required=False
    )
    register_runtime_service(
        "endogenous_language_health",
        pathway_health,
        required=False,
        owner="core/brain/llm/endogenous_decode.py",
        registered_by="register_pathway_health",
    )


def boot_endogenous_language(*, directory: Path | None = None) -> dict[str, Any]:
    """Bring the pathway up at boot and report exactly what is bound.

    Boot is where a half-wired pathway gets caught. This asks the three
    questions that decide whether anything can happen at all — is there a
    head, does it match the channel layout in this build, and does z_Aura
    have any live channels — and returns the answers rather than logging a
    reassuring line.
    """
    from core.brain.llm.endogenous_state import assemble_state, layout_digest

    register_pathway_health()
    # Importing registers the standing invariants; the module is a declaration
    # site, not a service.
    from core.brain.llm import endogenous_invariants  # noqa: F401 — registers
    from core.brain.llm.endogenous_telemetry import declare as declare_telemetry

    declare_telemetry()
    head, reason = load_head(directory)
    state = assemble_state()
    status: dict[str, Any] = {
        "head_present": head is not None,
        "head_reason": reason,
        "layout": layout_digest(),
        "state_coverage": round(state.coverage, 4),
        "live_channels": list(state.live_channels),
        "alpha": alpha_from_env(),
    }
    if head is not None:
        status["head_trained"] = bool(head.trained)
        status["head_layout_matches"] = head.layout == layout_digest()
        status["head_vocab_size"] = int(head.vocab_size)
        status["head_tokenizer"] = head.tokenizer
        status["head_report"] = dict(head.report or {})
    try:
        from core.brain.llm.endogenous_telemetry import publish

        status["telemetry"] = publish(status)
    except (ImportError, KeyError, RuntimeError, TypeError, ValueError) as exc:
        logger.debug("endogenous telemetry not published at boot: %s", exc)
    return status


__all__ += ["boot_endogenous_language", "register_pathway_health"]
