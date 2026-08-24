"""Gradient descent over thoughts: the hidden state as an optimizable object.

All weights stay frozen; Z is the variable. The differentiable proxy is
deliberately chosen so it CANNOT leak answers or simply sharpen confidence
(the failure mode the spec warns about — "merely pushing the model toward
high confidence would often strengthen confident mistakes"):

    S(Z) = λ_r·R(Z) − λ_d·D(Z, Z₀)

R — problem reconstruction: log-probability mass the workspace readout
assigns to the prompt's own token distribution. A state that can no longer
reconstruct what problem it is solving has lost the thread; pushing R up
keeps the latent computation grounded in the actual question. R contains no
information about the ANSWER.

D — manifold distance: RMS drift + cosine drift from the post-prelude seed
Z₀. Penalizing D keeps optimized states inside the activation distribution
the frozen layers were trained on.

Verifier signal (non-differentiable) enters through greedy hill-climbing:
propose → decode probe → verify → accept/reject. And the Experiment-5
control arm is built in: ``control_mode`` applies matched-magnitude RANDOM
perturbations — computed from the true gradient's step size so magnitudes
match exactly — letting the harness measure whether gradient DIRECTION
(not mere perturbation) is what helps.
"""
from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from core.brain.llm.decoder_topology import resolve_language_model
from core.brain.llm.latent_cortex.types import ComputeBudget, LatentOptConfig
from core.brain.llm.latent_cortex.workspace import per_position_rms
from core.runtime.model_layers import require_model_layers

logger = logging.getLogger("Aura.LatentCortex.LatentOpt")
_LINE_SEARCH_EVALS = 12
_VERIFIER_TRUST_SCALES = (16.0, 8.0, 4.0, 2.0, 1.0)


def prompt_token_distribution(prompt_tokens, vocab_size: int):
    """Empirical unigram distribution of the prompt (1, V) — the R target."""
    import mlx.core as mx

    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    if not prompt_tokens:
        raise ValueError("prompt token distribution requires at least one token")
    if any(type(token) is not int or not 0 <= token < vocab_size for token in prompt_tokens):
        raise ValueError("prompt token outside model vocabulary")
    counts = mx.zeros((vocab_size,))
    ones = mx.ones((len(prompt_tokens),))
    counts = counts.at[mx.array(prompt_tokens)].add(ones)
    return counts / mx.maximum(mx.sum(counts), 1.0)


def build_proxy_loss(
    model,
    z0,
    prompt_tokens: list[int],
    config: LatentOptConfig,
) -> Callable:
    """Loss(z) = −S(z), differentiable w.r.t. z with frozen weights.

    The readout path is norm → lm_head on the mean slot state: cheap, fully
    differentiable, and independent of the KV cache (no cache mutation inside
    the gradient graph).
    """
    import mlx.core as mx

    language_model = resolve_language_model(model)
    inner = require_model_layers(model).owner
    vocab = (
        language_model.lm_head.weight.shape[0]
        if hasattr(language_model, "lm_head")
        else inner.embed_tokens.weight.shape[0]
    )
    target = prompt_token_distribution(prompt_tokens, int(vocab))
    z0_rms = per_position_rms(z0)
    z0_flat = mx.reshape(z0, (-1,))
    z0_norm = mx.maximum(mx.linalg.norm(z0_flat), 1e-6)

    def readout_logits(z):
        pooled = mx.mean(z, axis=1, keepdims=True)  # (1,1,D)
        h = inner.norm(pooled)
        if hasattr(language_model, "lm_head"):
            return language_model.lm_head(h)
        return inner.embed_tokens.as_linear(h)

    def loss(z):
        # R: cross-entropy of readout against the prompt unigram target.
        logits = readout_logits(z)[0, 0]
        logp = logits - mx.logsumexp(logits)
        reconstruction = mx.sum(target * logp)  # ≤ 0, higher is better
        # D: manifold drift (norm band + direction).
        rms_drift = mx.mean(
            mx.square(per_position_rms(z) - z0_rms)
            / mx.square(mx.maximum(z0_rms, 1e-6))
        )
        z_flat = mx.reshape(z, (-1,))
        cos = mx.sum(z_flat * z0_flat) / (
            mx.maximum(mx.linalg.norm(z_flat), 1e-6) * z0_norm
        )
        manifold = rms_drift + (1.0 - cos)
        return -(config.lambda_reconstruct * reconstruction) + config.lambda_manifold * manifold

    return loss


@dataclass
class OptTrace:
    mode: str = "off"
    loss_trail: list[float] = field(default_factory=list)
    attempts: int = 0
    steps_taken: int = 0
    accepted: int = 0
    rejected: int = 0
    line_search_backtracks: int = 0
    budget_exhausted: bool = False
    verifier_policy: str = "off"
    verifier_score_source: str = "unspecified"
    verifier_commit_policy: str = "immediate"
    verifier_baseline_source: str = ""
    verifier_score_tolerance: float = 0.0
    verifier_proxy_tolerance_scale: float = 1e-9
    verifier_score_trail: list[float] = field(default_factory=list)
    verifier_decisions: list[dict[str, Any]] = field(default_factory=list)
    verifier_score_improvement_accepts: int = 0
    verifier_proxy_nonregression_accepts: int = 0
    verifier_plateau_exploration_accepts: int = 0
    verifier_plateau_rollbacks: int = 0
    verifier_strict_improvement_committed: bool = False

    def to_receipt(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "loss_trail": [round(v, 6) for v in self.loss_trail],
            "attempts": self.attempts,
            "steps_taken": self.steps_taken,
            "accepted": self.accepted,
            "rejected": self.rejected,
            "line_search_backtracks": self.line_search_backtracks,
            "budget_exhausted": self.budget_exhausted,
            "verifier": self.verifier_receipt(),
        }

    def verifier_receipt(self) -> dict[str, Any]:
        return {
            "policy": self.verifier_policy,
            "score_source": self.verifier_score_source,
            "commit_policy": self.verifier_commit_policy,
            "baseline_source": self.verifier_baseline_source,
            "score_tolerance": round(self.verifier_score_tolerance, 12),
            "proxy_tolerance_scale": round(
                self.verifier_proxy_tolerance_scale,
                12,
            ),
            # A non-finite entry is the "no verified score" sentinel, not a
            # measurement. It has to serialize as absence: the causal receipt
            # is canonicalized with allow_nan=False, so leaking -inf here
            # raises inside receipt construction and takes down an episode
            # that had otherwise completed. This path only executes once a
            # task verifier is admitted, which is why it survived every run
            # that had no verifier to admit.
            "score_trail": [
                None if not math.isfinite(float(value)) else round(float(value), 12)
                for value in self.verifier_score_trail
            ],
            "decisions": [dict(row) for row in self.verifier_decisions],
            "score_improvement_accepts": self.verifier_score_improvement_accepts,
            "proxy_nonregression_accepts": (
                self.verifier_proxy_nonregression_accepts
            ),
            "plateau_exploration_accepts": (
                self.verifier_plateau_exploration_accepts
            ),
            "plateau_rollbacks": self.verifier_plateau_rollbacks,
            "strict_improvement_committed": (
                self.verifier_strict_improvement_committed
            ),
        }


class LatentOptimizer:
    """Budgeted optimizer over a workspace state, gradient or control arm."""

    def __init__(
        self,
        loss_fn: Callable,
        config: LatentOptConfig,
        *,
        seed: int = 0,
        budget: ComputeBudget | None = None,
        layer_apps_per_loss: int = 0,
        scalar_ops_per_loss: int = 0,
        reserve_layer_apps: int = 0,
        protected_slots: tuple[int, ...] = (),
    ) -> None:
        if isinstance(layer_apps_per_loss, bool) or not isinstance(
            layer_apps_per_loss, int
        ):
            raise TypeError("layer_apps_per_loss must be an integer")
        if isinstance(reserve_layer_apps, bool) or not isinstance(
            reserve_layer_apps, int
        ):
            raise TypeError("reserve_layer_apps must be an integer")
        if isinstance(scalar_ops_per_loss, bool) or not isinstance(
            scalar_ops_per_loss, int
        ):
            raise TypeError("scalar_ops_per_loss must be an integer")
        if (
            layer_apps_per_loss < 0
            or scalar_ops_per_loss < 0
            or reserve_layer_apps < 0
        ):
            raise ValueError("optimizer compute costs cannot be negative")
        if budget is not None and layer_apps_per_loss <= 0:
            raise ValueError(
                "budgeted latent optimization requires a positive loss-evaluation cost"
            )
        if any(type(index) is not int or index < 0 for index in protected_slots):
            raise ValueError("protected latent slots must be non-negative integers")
        self._loss_fn = loss_fn
        self.config = config
        self._seed = seed
        self._budget = budget
        self._layer_apps_per_loss = layer_apps_per_loss
        self._scalar_ops_per_loss = scalar_ops_per_loss
        self._reserve_layer_apps = reserve_layer_apps
        self._protected_slots = tuple(sorted(set(protected_slots)))
        self.trace = OptTrace(mode="control" if config.control_mode else "gradient")

    def _zero_protected(self, value):
        """Remove optimizer authority over immutable evidence rows."""

        if not self._protected_slots:
            return value
        import mlx.core as mx

        slot_count = int(value.shape[1])
        if any(index >= slot_count for index in self._protected_slots):
            raise ValueError("protected latent slot is outside the workspace")
        protected = set(self._protected_slots)
        return mx.concatenate(
            [
                mx.zeros_like(value[:, index : index + 1, :])
                if index in protected
                else value[:, index : index + 1, :]
                for index in range(slot_count)
            ],
            axis=1,
        )

    def _can_reserve(self, additional_layer_apps: int = 0) -> bool:
        if isinstance(additional_layer_apps, bool) or not isinstance(
            additional_layer_apps, int
        ):
            raise TypeError("additional_layer_apps must be an integer")
        if additional_layer_apps < 0:
            raise ValueError("additional_layer_apps cannot be negative")
        if self._budget is None:
            return True
        admitted = (
            not self._budget.exhausted
            and self._reserve_layer_apps + additional_layer_apps
            <= self._budget.remaining_layer_apps
        )
        if not admitted:
            self.trace.budget_exhausted = True
        return admitted

    def _charge_loss_evals(
        self, count: int, *, additional_reserve_layer_apps: int = 0
    ) -> bool:
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("loss-evaluation count must be a non-negative integer")
        if self._budget is None:
            return True
        charge = count * self._layer_apps_per_loss
        if not self._can_reserve(charge + additional_reserve_layer_apps):
            return False
        if self._scalar_ops_per_loss <= 0 and count:
            self._budget.resource_ledger.mark_unknown("latent_proxy_loss")
        self._budget.charge_proxy_work(
            "latent_proxy_loss",
            layer_app_equivalents=charge,
            scalar_ops=count * self._scalar_ops_per_loss,
        )
        return True

    def _clipped_step(self, grad):
        """Gradient step with global-norm clipping; returns (step, magnitude)."""
        import mlx.core as mx

        gnorm = mx.maximum(mx.linalg.norm(mx.reshape(grad, (-1,))), 1e-12)
        scale = mx.minimum(1.0, self.config.max_grad_norm / gnorm)
        step = -self.config.lr * grad * scale
        return step, float(mx.linalg.norm(mx.reshape(step, (-1,))))

    def _propose(
        self,
        z,
        step_index: int,
        *,
        additional_reserve_layer_apps: int = 0,
    ):
        """One optimization move. Control mode consumes the SAME gradient
        computation to derive its magnitude, then discards the direction —
        that is what makes it a matched control rather than a strawman."""
        import mlx.core as mx

        # Forward + backward is conservatively charged as three proxy-loss
        # evaluations. The charge happens before execution and protects the
        # answer-decode reserve, so optimization can never strand completion.
        if not self._charge_loss_evals(
            3, additional_reserve_layer_apps=additional_reserve_layer_apps
        ):
            return z, False, None
        self.trace.attempts += 1
        value, grad = mx.value_and_grad(self._loss_fn)(z)
        grad = self._zero_protected(grad)
        value_float = float(value)
        if not math.isfinite(value_float) or not bool(mx.all(mx.isfinite(grad))):
            raise RuntimeError("latent optimizer produced a non-finite value or gradient")
        if not self.trace.loss_trail:
            self.trace.loss_trail.append(value_float)
        step, magnitude = self._clipped_step(grad)
        if not math.isfinite(magnitude):
            raise RuntimeError("latent optimizer produced a non-finite step magnitude")
        if self.config.control_mode:
            key = mx.random.key(90210 + 7 * self._seed + step_index)
            rand = self._zero_protected(mx.random.normal(z.shape, key=key))
            rand_norm = mx.maximum(mx.linalg.norm(mx.reshape(rand, (-1,))), 1e-12)
            step = rand * (magnitude / rand_norm)
        z_next = z + step
        mx.eval(z_next)
        return z_next, True, value_float

    def step(self, z, step_index: int):
        """Return one proposal without claiming it was accepted.

        ``run`` and ``run_with_verifier`` own acceptance bookkeeping. Keeping
        proposal generation separate prevents verifier-rejected states from
        inflating the accepted-step count in evidence receipts.
        """
        candidate, _, _ = self._propose(z, step_index)
        return candidate

    def run(self, z):
        """Bounded proxy descent under one acceptance policy for both arms."""
        import mlx.core as mx

        for step_index in range(max(0, int(self.config.steps))):
            line_search_cost = _LINE_SEARCH_EVALS * self._layer_apps_per_loss
            candidate, admitted, current_loss = self._propose(
                z,
                step_index,
                additional_reserve_layer_apps=line_search_cost,
            )
            if not admitted:
                break
            if current_loss is None:
                raise RuntimeError("latent optimizer admitted a proposal without a loss")
            if not self._charge_loss_evals(_LINE_SEARCH_EVALS):
                raise RuntimeError("latent optimizer lost an admitted line-search reservation")
            raw_step = candidate - z
            candidates: list[tuple[int, Any, float]] = []
            for backtrack in range(_LINE_SEARCH_EVALS):
                backtracked = z + raw_step * (0.5**backtrack)
                candidate_loss = float(self._loss_fn(backtracked))
                if math.isfinite(candidate_loss) and candidate_loss < current_loss:
                    candidates.append((backtrack, backtracked, candidate_loss))
            if not candidates:
                self.trace.rejected += 1
                break
            backtrack, accepted_state, accepted_loss = candidates[0]
            mx.eval(accepted_state)
            z = accepted_state
            self.trace.steps_taken += 1
            self.trace.accepted += 1
            self.trace.line_search_backtracks += backtrack
            self.trace.loss_trail.append(accepted_loss)
        mx.eval(z)
        return z

    def run_with_verifier(
        self,
        z,
        score_fn: Callable[[Any], float],
        *,
        max_proposals: int | None = None,
        verifier_layer_apps: int = 0,
        initial_score: float | None = None,
        accept_non_regression: bool = False,
        commit_requires_score_improvement: bool = False,
        score_tolerance: float = 1e-9,
    ):
        """Greedy hill-climb: proxy-guided proposals, verifier-gated accepts.

        ``score_fn`` is the honesty boundary — it must decode a probe from
        the CANDIDATE state and return a verified score. Rejected proposals
        are fully reverted; the verifier, not the proxy, has the last word.
        """
        if isinstance(verifier_layer_apps, bool) or not isinstance(
            verifier_layer_apps, int
        ):
            raise TypeError("verifier_layer_apps must be an integer")
        if verifier_layer_apps < 0:
            raise ValueError("verifier_layer_apps cannot be negative")
        if type(accept_non_regression) is not bool:
            raise TypeError("accept_non_regression must be a boolean")
        if type(commit_requires_score_improvement) is not bool:
            raise TypeError("commit_requires_score_improvement must be a boolean")
        if commit_requires_score_improvement and not accept_non_regression:
            raise ValueError(
                "strict latent commitment requires non-regressing plateau search"
            )
        if (
            isinstance(score_tolerance, bool)
            or not isinstance(score_tolerance, (int, float))
            or not math.isfinite(float(score_tolerance))
            or not 0.0 <= float(score_tolerance) <= 1e-3
        ):
            raise ValueError("score_tolerance must be finite and inside [0, 1e-3]")
        if initial_score is not None and (
            isinstance(initial_score, bool)
            or not isinstance(initial_score, (int, float))
            or not math.isfinite(float(initial_score))
        ):
            raise ValueError("initial_score must be a finite number or None")
        proposals = max_proposals if max_proposals is not None else self.config.steps
        if not self._can_reserve(verifier_layer_apps):
            return z, float("-inf")
        self.trace.verifier_policy = (
            "task_score_nonregression_with_proxy_descent_v1"
            if accept_non_regression
            else "strict_task_score_improvement_v1"
        )
        self.trace.verifier_commit_policy = (
            "strict_task_improvement_after_plateau_search_v1"
            if commit_requires_score_improvement
            else "immediate"
        )
        self.trace.verifier_score_tolerance = float(score_tolerance)
        self.trace.verifier_baseline_source = (
            "caller_reused_verified_branch"
            if initial_score is not None
            else "decoded_state_probe"
        )
        best_score = (
            float(initial_score) if initial_score is not None else float(score_fn(z))
        )
        if not math.isfinite(best_score):
            raise RuntimeError("latent verifier returned a non-finite baseline score")
        self.trace.verifier_score_trail.append(best_score)
        best_state = z
        accepted_path_decisions: list[int] = []
        committed_path_length = 0
        for i in range(max(0, int(proposals))):
            trust_scales = (1.0,)
            if accept_non_regression:
                max_scale = max(1.0, 1.0 / float(self.config.lr))
                trust_scales = tuple(
                    scale for scale in _VERIFIER_TRUST_SCALES if scale <= max_scale
                )
                if 1.0 not in trust_scales:
                    trust_scales = (*trust_scales, 1.0)
            proxy_eval_cost = (
                len(trust_scales) * self._layer_apps_per_loss
                if accept_non_regression
                else 0
            )
            candidate, admitted, current_loss = self._propose(
                z,
                i,
                additional_reserve_layer_apps=verifier_layer_apps + proxy_eval_cost,
            )
            if not admitted:
                break
            candidate_loss: float | None = None
            proposal_scale = 1.0
            if accept_non_regression:
                if current_loss is None:
                    raise RuntimeError(
                        "latent optimizer admitted a proposal without a proxy loss"
                    )
                if not self._charge_loss_evals(
                    len(trust_scales),
                    additional_reserve_layer_apps=verifier_layer_apps,
                ):
                    raise RuntimeError(
                        "latent optimizer lost an admitted proxy-verifier reservation"
                    )
                raw_step = candidate - z
                proxy_required_delta = self.trace.verifier_proxy_tolerance_scale * max(
                    1.0,
                    abs(float(current_loss)),
                )
                evaluated: list[tuple[float, Any, float]] = []
                for scale in trust_scales:
                    scaled = z + raw_step * scale
                    scaled_loss = float(self._loss_fn(scaled))
                    evaluated.append((scale, scaled, scaled_loss))
                proxy_safe = [
                    row
                    for row in evaluated
                    if math.isfinite(row[2])
                    and row[2] < float(current_loss) - proxy_required_delta
                ]
                if proxy_safe:
                    proposal_scale, candidate, candidate_loss = proxy_safe[0]
                else:
                    proposal_scale, candidate, candidate_loss = evaluated[-1]
            candidate_score = float(score_fn(candidate))
            proxy_required_delta = (
                self.trace.verifier_proxy_tolerance_scale
                * max(1.0, abs(float(current_loss)))
                if current_loss is not None and math.isfinite(float(current_loss))
                else None
            )
            decision: dict[str, Any] = {
                "proposal": i,
                "proposal_scale": round(float(proposal_scale), 6),
                "proxy_candidate_evaluations": len(trust_scales),
                "baseline_score": round(best_score, 12),
                "candidate_score": (
                    round(candidate_score, 12)
                    if math.isfinite(candidate_score)
                    else "nonfinite"
                ),
                "current_proxy_loss": (
                    round(float(current_loss), 12)
                    if current_loss is not None and math.isfinite(float(current_loss))
                    else None
                ),
                "candidate_proxy_loss": (
                    round(candidate_loss, 12)
                    if candidate_loss is not None and math.isfinite(candidate_loss)
                    else None
                ),
                "proxy_required_delta": (
                    round(proxy_required_delta, 12)
                    if proxy_required_delta is not None
                    else None
                ),
            }
            if not math.isfinite(candidate_score):
                self.trace.rejected += 1
                decision["decision"] = "rejected_nonfinite_task_score"
                self.trace.verifier_decisions.append(decision)
                self.trace.verifier_score_trail.append(best_score)
                continue
            if accept_non_regression and (
                candidate_loss is None or not math.isfinite(candidate_loss)
            ):
                self.trace.rejected += 1
                decision["decision"] = "rejected_nonfinite_proxy_loss"
                self.trace.verifier_decisions.append(decision)
                self.trace.verifier_score_trail.append(best_score)
                continue

            score_improved = candidate_score > best_score + float(score_tolerance)
            proxy_improved = bool(
                current_loss is not None
                and candidate_loss is not None
                and candidate_loss
                < float(current_loss) - float(proxy_required_delta or 0.0)
            )
            score_nonregressing = (
                candidate_score >= best_score - float(score_tolerance)
            )
            if score_improved:
                z, best_score = candidate, candidate_score
                best_state = candidate
                self.trace.accepted += 1
                self.trace.steps_taken += 1
                self.trace.verifier_score_improvement_accepts += 1
                self.trace.verifier_strict_improvement_committed = True
                decision["decision"] = "accepted_task_score_improvement"
                accepted_path_decisions.append(len(self.trace.verifier_decisions))
                committed_path_length = len(accepted_path_decisions)
                if candidate_loss is not None and proxy_improved:
                    self.trace.loss_trail.append(candidate_loss)
            elif accept_non_regression and score_nonregressing and proxy_improved:
                z = candidate
                best_score = max(best_score, candidate_score)
                self.trace.accepted += 1
                self.trace.steps_taken += 1
                self.trace.verifier_proxy_nonregression_accepts += 1
                self.trace.verifier_plateau_exploration_accepts += 1
                self.trace.loss_trail.append(float(candidate_loss))
                decision["decision"] = (
                    "accepted_task_score_nonregression_with_proxy_descent"
                )
                accepted_path_decisions.append(len(self.trace.verifier_decisions))
            else:
                self.trace.rejected += 1
                decision["decision"] = (
                    "rejected_task_score_regression"
                    if not score_nonregressing
                    else "rejected_proxy_non_descent"
                    if accept_non_regression
                    else "rejected_no_task_score_improvement"
                )
            self.trace.verifier_decisions.append(decision)
            # This is the finite incumbent trail, not a raw candidate trail.
            # Candidate scores (including explicit non-finite outcomes) live in
            # the decision rows. Keeping one incumbent per decision makes the
            # receipt total, monotonic, and JSON-safe under every verifier
            # outcome.
            self.trace.verifier_score_trail.append(best_score)
        if commit_requires_score_improvement:
            rolled_back = accepted_path_decisions[committed_path_length:]
            for decision_index in accepted_path_decisions[:committed_path_length]:
                self.trace.verifier_decisions[decision_index]["commit_disposition"] = (
                    "committed_to_best_strict_improvement"
                )
            for decision_index in rolled_back:
                self.trace.verifier_decisions[decision_index]["commit_disposition"] = (
                    "rolled_back_plateau_without_later_task_gain"
                )
            rollback_count = len(rolled_back)
            self.trace.verifier_plateau_rollbacks = rollback_count
            self.trace.accepted -= rollback_count
            self.trace.steps_taken -= rollback_count
            self.trace.rejected += rollback_count
            self.trace.verifier_proxy_nonregression_accepts -= rollback_count
            z = best_state
        return z, best_score


__all__ = [
    "LatentOptimizer",
    "OptTrace",
    "build_proxy_loss",
    "prompt_token_distribution",
]
