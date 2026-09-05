"""Controlled recurrence: the anti-naive-looping core.

2026 frozen-loop studies found naive layer repetition unstable. This module
is the difference between "run layers 16–47 again" and a governed dynamical
system:

  Z̃ₜ₊₁   = Window(Zₜ)                       (slots re-enter the layer window)
  Zₜ₊₁    = (1−αₜ)·Zₜ + αₜ·RMSMatch(Z̃ₜ₊₁, Zₜ)

- RMSMatch clamps per-position norm drift so the state stays on the
  activation manifold the subsequent layers were trained to expect.
- The α schedule trades update speed against stability (cosine decay ⇒
  aggressive early exploration, gentle convergence).
- The halting controller detects fixed points (converged), divergence
  (revert), budget exhaustion, and overthinking (score peaked earlier ⇒
  revert to the best state, not the last one).

KV discipline: every window pass appends slot K/V, which must be rewound so
only the engine's final clean pass persists. We reuse the battle-tested
snapshot/restore machinery from ``core.brain.llm.recurrent_depth`` — the same
code that guards the live resident model's recurrent depth today.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Mapping
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from core.brain.llm.decoder_topology import decoder_layer_masks
from core.brain.llm.latent_cortex.loop_core import (
    ABSOLUTE_POSITION_LIMIT,
    KV_BOUND_SCHEMA,
    ComputeBudgetUnaffordable,
    LoopCoreError,
    alpha_for_step,
    assert_finite_state,
    canonical_sha256,
    controlled_recurrent_update,
    rms_match,
)
from core.brain.llm.latent_cortex.resource_accounting import (
    triangular_attention_pairs,
)
from core.brain.llm.latent_cortex.types import ComputeBudget, RecurrenceConfig
from core.brain.llm.latent_cortex.workspace import per_position_rms
from core.brain.llm.recurrent_depth import (
    CacheSnapshotError,
    _cache_entry_matches_snapshot,
    _cache_token_cursor,
    _restore_recurrent_caches,
    _snapshot_recurrent_caches,
)

logger = logging.getLogger("Aura.LatentCortex.Recurrence")

CACHE_DISCIPLINE_SCHEMA = "aura.rlc.cache_discipline.v1"

if TYPE_CHECKING:
    from core.brain.llm.latent_cortex.kv_state_tree import KVStateTree


def _cache_matches_snapshot(cache, start: int, end: int, snapshots: list) -> bool:
    for index, layer_index in enumerate(range(start, end)):
        if not _cache_entry_matches_snapshot(cache[layer_index], snapshots[index]):
            return False
    return True


def alpha_at(config: RecurrenceConfig, step: int) -> float:
    """Interpolation coefficient for a given step under the schedule."""
    return alpha_for_step(
        alpha=config.alpha,
        schedule=config.alpha_schedule,
        max_steps=config.max_steps,
        step=step,
    )


def relative_residual(z_next, z_prev) -> float:
    """‖Zₜ₊₁−Zₜ‖ / ‖Zₜ‖ in mean-RMS terms — the fixed-point signal."""
    import mlx.core as mx

    if tuple(z_next.shape) != tuple(z_prev.shape):
        raise ValueError("residual state shapes differ")
    assert_finite_state(z_next, stage="residual_output")
    assert_finite_state(z_prev, stage="residual_input")
    num = mx.mean(per_position_rms(z_next - z_prev))
    den = mx.maximum(mx.mean(per_position_rms(z_prev)), 1e-6)
    value = float(num / den)
    if not math.isfinite(value) or value < 0.0:
        raise LoopCoreError("recurrent residual is invalid")
    return value


@dataclass
class HaltDecision:
    should_halt: bool
    reason: str = ""


@dataclass
class HaltingController:
    """Adaptive halting with divergence and overthinking protection.

    Tracks the best-scoring state seen so far (when an external score signal
    is provided) so the engine can revert to the trajectory's peak instead of
    shipping an over-thought state — the "excessive recurrence degrades
    results" failure mode from the recurrent-depth literature.
    """

    config: RecurrenceConfig
    baseline_rms: float = 0.0
    residual_trail: list[float] = field(default_factory=list)
    score_trail: list[float] = field(default_factory=list)
    best_step: int = -1
    best_score: float = -math.inf
    best_state: Any = None
    # Optional learned halting head (CP230/234). None => the residual policy
    # this controller has always run. Attaching a head grants nothing on its
    # own: the head is zero-initialised, so an untrained one never fires.
    halting_head: Any = None
    head_halts: int = 0
    # Calibrated public-signal stop policy. The engine uses this path for
    # learned mode; the raw latent head remains only for training compatibility.
    stop_gate: Any = None
    stop_trace: list[dict[str, Any]] = field(default_factory=list)

    def snapshot(self) -> dict[str, Any]:
        """Capture every mutable field that can influence later halting."""

        return {
            "residual_trail": tuple(self.residual_trail),
            "score_trail": tuple(self.score_trail),
            "best_step": self.best_step,
            "best_score": self.best_score,
            "best_state": self.best_state,
            "head_halts": self.head_halts,
            "stop_trace": tuple(dict(row) for row in self.stop_trace),
        }

    def restore(self, snapshot: dict[str, Any]) -> None:
        """Restore a snapshot without replacing configuration or head identity."""

        required = {
            "residual_trail",
            "score_trail",
            "best_step",
            "best_score",
            "best_state",
            "head_halts",
            "stop_trace",
        }
        if not isinstance(snapshot, dict) or set(snapshot) != required:
            raise ValueError("invalid halting-controller snapshot")
        self.residual_trail = [float(value) for value in snapshot["residual_trail"]]
        self.score_trail = [float(value) for value in snapshot["score_trail"]]
        self.best_step = int(snapshot["best_step"])
        self.best_score = float(snapshot["best_score"])
        self.best_state = snapshot["best_state"]
        self.head_halts = int(snapshot["head_halts"])
        self.stop_trace = [dict(row) for row in snapshot["stop_trace"]]

    def observe(
        self,
        step: int,
        z_next,
        residual: float,
        *,
        score: float | None = None,
        budget: ComputeBudget | None = None,
        stop_context: Any = None,
        update_decision: Any = None,
    ) -> HaltDecision:
        import mlx.core as mx

        self.residual_trail.append(residual)

        # Divergence guard: non-finite state or runaway norms ⇒ halt now.
        if not bool(mx.all(mx.isfinite(z_next))):
            return HaltDecision(True, "diverged_nonfinite")
        mean_rms = float(mx.mean(per_position_rms(z_next)))
        if self.baseline_rms > 0 and mean_rms > self.baseline_rms * self.config.divergence_ratio:
            return HaltDecision(True, "diverged_norm")

        # Best-state tracking (overthinking protection). Without an external
        # score, convergence quality (negative residual) is the proxy.
        effective_score = score if score is not None else -residual
        self.score_trail.append(effective_score)
        if effective_score > self.best_score:
            self.best_score = effective_score
            self.best_step = step
            self.best_state = z_next

        if budget is not None and budget.exhausted:
            return HaltDecision(True, "budget_exhausted")
        if step + 1 >= self.config.max_steps:
            return HaltDecision(True, "max_steps")
        if (
            not self.config.fixed_depth
            and step + 1 >= self.config.min_steps
            and residual < self.config.convergence_eps
        ):
            return HaltDecision(True, "converged")

        if (
            self.stop_gate is not None
            and not self.config.fixed_depth
            and step + 1 >= self.config.min_steps
        ):
            if stop_context is None or update_decision is None:
                raise ValueError("learned stop gate requires update and value evidence")
            stop_decision = self.stop_gate.evaluate(
                step=step + 1,
                residual=residual,
                previous_residual=(
                    self.residual_trail[-2] if len(self.residual_trail) >= 2 else None
                ),
                update_decision=update_decision,
                context=stop_context,
            )
            self.stop_trace.append(
                {
                    "ordinal": len(self.stop_trace),
                    "action_step": stop_context.action_step,
                    "step": step + 1,
                    "halt": stop_decision.halt,
                    "reason": stop_decision.reason,
                    "probability": stop_decision.probability,
                    "threshold": stop_decision.threshold,
                    "evidence_ready": stop_decision.evidence_ready,
                    "features": dict(stop_decision.features),
                    "features_sha256": stop_decision.features_sha256,
                }
            )
            if stop_decision.halt:
                self.head_halts += 1
                return HaltDecision(True, "learned_stop")

        # Learned allocation, consulted only AFTER the convergence floor.
        # Residual halting answers "has this loop stopped changing?"; the
        # head answers "does this problem deserve more thought?" CP226
        # measured where those come apart -- a loop still moving healthily
        # (deltas 0.55, 0.50, 0.32) while accuracy fell to zero. Residual
        # halting sees motion and keeps going.
        if (
            self.halting_head is not None
            and not self.config.fixed_depth
            and step + 1 >= self.config.min_steps
        ):
            identity_probe = getattr(self.halting_head, "is_identity", None)
            identity = bool(identity_probe()) if callable(identity_probe) else False
            if not identity:
                probability = float(self.halting_head.halt_probability(z_next))
                if probability >= self.halting_head.threshold:
                    self.head_halts += 1
                    return HaltDecision(True, "head_satisfied")
        return HaltDecision(False)

    def final_state(self, z_last) -> tuple[Any, bool]:
        """Return (state to ship, reverted?) — best state if it beats last."""
        if self.config.fixed_depth:
            return z_last, False
        if self.best_state is not None and (
            not self.score_trail or self.best_step < len(self.score_trail) - 1
        ):
            return self.best_state, True
        return z_last, False


class WindowRunner:
    """Runs hidden states through a contiguous layer window with KV discipline.

    ``persist=False`` (recurrent passes): slot K/V appended by the pass is
    rewound so cache offsets never drift — RoPE positions stay identical
    across passes, which the mechanics probe proved is what keeps recurrence
    stable. ``persist=True`` (the engine's final clean pass): K/V stays, so
    the decoded answer attends to the refined thoughts.
    """

    def __init__(self, inner_model, budget: ComputeBudget, mask_fn: Callable | None = None):
        self._inner = inner_model
        self._budget = budget
        self._mask_fn = mask_fn
        self._adapter_calls = 0
        self._adapter_adapted_positions = 0
        self._adapter_observed_positions = 0
        self._nonpersistent_calls = 0
        self._restored_calls = 0
        self._restore_failures = 0
        model_args = getattr(inner_model, "args", None)
        raw_limit = (
            model_args.get("max_position_embeddings")
            if isinstance(model_args, Mapping)
            else getattr(model_args, "max_position_embeddings", None)
        )
        if type(raw_limit) is int and raw_limit > 0:
            self._position_limit = min(raw_limit, ABSOLUTE_POSITION_LIMIT)
            self._position_limit_source = "model_config"
        else:
            self._position_limit = ABSOLUTE_POSITION_LIMIT
            self._position_limit_source = "absolute_safety_ceiling"
        self._kv_calls: list[dict[str, Any]] = []
        self._kv_state_tree: KVStateTree | None = None
        self._transaction_purpose = "speculative_latent_window"
        self._transaction_branch_index: int | None = None
        self._transaction_parent_sha256 = ""

    def attach_kv_state_tree(self, tree: KVStateTree) -> None:
        """Bind the episode KV lineage after prompt prefill establishes root."""

        if self._kv_state_tree is not None and self._kv_state_tree is not tree:
            raise LoopCoreError("WindowRunner KV state tree is already bound")
        self._kv_state_tree = tree
        self._transaction_parent_sha256 = tree.root_sha256

    @contextmanager
    def transaction_context(
        self,
        *,
        purpose: str,
        branch_index: int,
        parent_sha256: str,
    ):
        """Bind one branch/purpose to the next nested speculative window."""

        previous = (
            self._transaction_purpose,
            self._transaction_branch_index,
            self._transaction_parent_sha256,
        )
        self._transaction_purpose = purpose
        self._transaction_branch_index = branch_index
        self._transaction_parent_sha256 = parent_sha256
        try:
            yield
        finally:
            (
                self._transaction_purpose,
                self._transaction_branch_index,
                self._transaction_parent_sha256,
            ) = previous

    def adapter_receipt(self) -> dict[str, int | str | bool]:
        """Aggregate proof that scoped weights ran only inside slot windows."""

        return {
            "schema": "aura.recurrence_adapter_activation.v1",
            "scope": "latent_slots_only",
            "calls": self._adapter_calls,
            "adapted_positions": self._adapter_adapted_positions,
            "observed_positions": self._adapter_observed_positions,
            "active": self._adapter_calls > 0,
        }

    def cache_discipline_receipt(self) -> dict[str, int | str | bool]:
        """Public proof that every speculative cache mutation was rewound."""

        return {
            "schema": CACHE_DISCIPLINE_SCHEMA,
            "nonpersistent_calls": self._nonpersistent_calls,
            "restored_calls": self._restored_calls,
            "restore_failures": self._restore_failures,
            "all_restored": (
                self._restore_failures == 0 and self._restored_calls == self._nonpersistent_calls
            ),
        }

    def _context_tokens(self, cache, start: int, end: int) -> int | None:
        if (
            not isinstance(cache, (list, tuple))
            or type(start) is not int
            or type(end) is not int
            or not 0 <= start < end <= len(cache)
        ):
            raise LoopCoreError("recurrent cache window is invalid")
        offsets: list[int] = []
        for item in cache[start:end]:
            try:
                offset = _cache_token_cursor(item)
            except CacheSnapshotError as exc:
                raise LoopCoreError("recurrent cache offset is invalid") from exc
            if offset is not None:
                offsets.append(offset)
        if len(set(offsets)) > 1:
            raise LoopCoreError("recurrent cache window offsets disagree")
        return offsets[0] if offsets else None

    def kv_bound_receipt(self) -> dict[str, Any]:
        """Return model-bounded position evidence for every window call."""

        calls = [dict(row) for row in self._kv_calls]
        return {
            "schema": KV_BOUND_SCHEMA,
            "position_limit": self._position_limit,
            "position_limit_source": self._position_limit_source,
            "call_count": len(calls),
            "max_context_tokens": max(
                (row["context_tokens"] for row in calls if row["context_tokens"] is not None),
                default=None,
            ),
            "max_total_tokens": max(
                (row["total_tokens"] for row in calls if row["total_tokens"] is not None),
                default=None,
            ),
            "all_within_limit": bool(calls)
            and all(
                (row["total_tokens"] if row["total_tokens"] is not None else row["tokens"])
                <= self._position_limit
                for row in calls
            ),
            "calls": calls,
            "calls_sha256": canonical_sha256(calls),
        }

    def _masks(self, h, cache, start: int, end: int):
        if self._mask_fn is not None:
            mask = self._mask_fn(h, cache[start:end])
            return [mask] * (end - start)
        return decoder_layer_masks(
            self._inner,
            h,
            cache,
            start=start,
            end=end,
        )

    def run(self, h, cache, start: int, end: int, *, persist: bool) -> Any:
        import mlx.core as mx

        from core.brain.llm.latent_cortex.recurrence_adapter import (
            recurrence_adapter_scope,
        )

        tokens = int(h.shape[1])
        layers = end - start
        if tokens < 1 or layers < 1:
            raise LoopCoreError("recurrent window dimensions are empty")
        context_tokens = self._context_tokens(cache, start, end)
        total_tokens = context_tokens + tokens if context_tokens is not None else None
        if (total_tokens if total_tokens is not None else tokens) > self._position_limit:
            raise LoopCoreError(
                "recurrent KV position limit exceeded: "
                f"total={total_tokens} limit={self._position_limit}"
            )
        if not self._budget.can_afford(tokens, layers):
            # Typed, because the caller must tell "declined to spend" from
            # "broke mid-decode". Nothing has run in this window yet, so the
            # resident model is clean and the turn can still be answered by the
            # ordinary path.
            if self._budget.remaining_wall_s <= 0.0:
                reason = f"wall-clock budget exhausted (limit={self._budget.wall_clock_s:.3f}s)"
            else:
                required = tokens * layers
                reason = (
                    "layer-application budget unavailable "
                    f"(required={required} remaining={self._budget.remaining_layer_apps})"
                )
            raise ComputeBudgetUnaffordable(
                f"compute budget cannot afford window [{start}:{end}) for {tokens} slots: {reason}"
            )
        # Reserve and account the whole atomic pass before execution. A layer
        # fault can consume partial compute, so failed work must not disappear
        # from the conservative ledger or become available to a fallback.
        attention_pairs = 0
        for item in cache[start:end]:
            offset = _cache_token_cursor(item)
            if offset is not None:
                attention_pairs += triangular_attention_pairs(tokens, context_tokens=offset)
        self._budget.charge(
            tokens=tokens,
            layers=layers,
            operation=("persisted_latent_window" if persist else "speculative_latent_window"),
            attention_pairs=attention_pairs,
        )
        snaps = None
        kv_transaction = None
        if not persist:
            snaps = _snapshot_recurrent_caches(cache, start, end)
            self._nonpersistent_calls += 1
            if self._kv_state_tree is not None:
                parent_sha256 = self._transaction_parent_sha256 or self._kv_state_tree.root_sha256
                kv_transaction = self._kv_state_tree.begin_speculation(
                    cache,
                    start=start,
                    end=end,
                    purpose=self._transaction_purpose,
                    branch_index=self._transaction_branch_index,
                    parent_sha256=parent_sha256,
                )
        adapter_activation = None
        execution_failed = True
        try:
            masks = self._masks(h, cache, start, end)
            # A WindowRunner call is the live proof boundary that these inputs
            # are thought slots. Recurrent adapters remain dark for all direct
            # prompt, lexical decode, and unrelated model calls.
            with recurrence_adapter_scope() as adapter_activation:
                for i in range(start, end):
                    h = self._inner.layers[i](h, masks[i - start], cache[i])
            mx.eval(h)
            execution_failed = False
        finally:
            if adapter_activation is not None:
                self._adapter_calls += adapter_activation.calls
                self._adapter_adapted_positions += adapter_activation.adapted_positions
                self._adapter_observed_positions += adapter_activation.observed_positions
            if snaps is not None:
                try:
                    if kv_transaction is not None:
                        try:
                            kv_transaction.observe_mutation(
                                cache,
                                execution_failed=execution_failed,
                            )
                        finally:
                            # Invalid speculative work must still release its cache writes.
                            kv_transaction.restore_parent(cache)
                    else:
                        _restore_recurrent_caches(cache, start, end, snaps)
                    if kv_transaction is None and not _cache_matches_snapshot(
                        cache,
                        start,
                        end,
                        snaps,
                    ):
                        raise CacheSnapshotError(
                            "KV cache restore postcondition failed after recurrent pass"
                        )
                    if kv_transaction is not None:
                        kv_transaction.reject_after_restore(cache)
                except Exception:
                    self._restore_failures += 1
                    raise
                else:
                    self._restored_calls += 1
        post_context = self._context_tokens(cache, start, end)
        restored = not persist and post_context == context_tokens
        if not persist and not restored:
            self._restore_failures += 1
            raise CacheSnapshotError(
                "KV cache position restore postcondition failed after recurrent pass"
            )
        if persist and post_context != total_tokens:
            raise CacheSnapshotError("KV cache position did not advance after persistent pass")
        self._kv_calls.append(
            {
                "ordinal": len(self._kv_calls),
                "start": start,
                "end": end,
                "tokens": tokens,
                "context_tokens": context_tokens,
                "total_tokens": total_tokens,
                "post_context_tokens": post_context,
                "persist": persist,
                "restored": restored,
            }
        )
        return h


def recurrence_step(
    z,
    runner: WindowRunner,
    cache,
    start: int,
    end: int,
    config: RecurrenceConfig,
    step: int,
    *,
    anchor=None,
    alpha_override: float | None = None,
    branch_index: int | None = None,
):
    """One controlled update: window pass (rewound) + anchored RMSMatch + α-blend.

    ``anchor`` is the manifold reference for the RMS trust band — normally the
    post-prelude seed state Z₀ ("the norm distribution expected by the
    subsequent layers"). Banding against a FIXED anchor is what prevents the
    ratchet failure: a band around the moving previous state would permit
    clip_ratio× growth per step, compounding without bound.
    """
    import mlx.core as mx

    # The functional training graph publishes the current recurrent depth so
    # depth-conditioned operator banks select the same transform as the live
    # cached engine.  Without this scope, a trained bank could pass offline
    # objectives while live inference silently used its default depth.
    from core.learning.depth_conditioned_lora import recurrent_depth_index
    from core.learning.role_conditioned_lora import recurrent_branch_index

    branch_scope = (
        recurrent_branch_index(branch_index) if branch_index is not None else nullcontext()
    )
    with branch_scope, recurrent_depth_index(step):
        z_raw = runner.run(z, cache, start, end, persist=False)
    alpha = alpha_override if alpha_override is not None else alpha_at(config, step)
    reference = anchor if anchor is not None else z
    z_next = controlled_recurrent_update(
        z,
        z_raw,
        reference,
        alpha=alpha,
        clip_ratio=config.rms_clip_ratio,
    )
    mx.eval(z_next)
    return z_next


__all__ = [
    "HaltDecision",
    "HaltingController",
    "CACHE_DISCIPLINE_SCHEMA",
    "LoopCoreError",
    "WindowRunner",
    "alpha_at",
    "recurrence_step",
    "relative_residual",
    "rms_match",
]
