"""core/adaptation/meta_learner.py -- MAML-Style Meta-Learning
==============================================================
Meta-optimization over substrate and value parameters using a
derivative-free MAML-style algorithm (Evolution Strategies variant).

Since the substrate and value graph use numpy (no autograd), we use
Evolution Strategies (ES) to approximate meta-gradients:
  1. Define "tasks" (episodes of value optimization, world-model prediction)
  2. For each task: perturb meta-parameters, evaluate performance
  3. Estimate gradient from perturbation-reward correlations
  4. Update meta-parameters in the direction of cross-task improvement

This operates over:
  - Substrate coupling weights (W matrix decay/gain)
  - Value graph evolution rates
  - World model learning rate and KL weight
  - NOT the base LLM weights (those are frozen)

References:
    Finn et al. (2017) Model-Agnostic Meta-Learning (MAML)
    Salimans et al. (2017) Evolution Strategies as a Scalable Alternative
    to Reinforcement Learning
    Nichol et al. (2018) Reptile: first-order meta-learning
"""
from __future__ import annotations

import io
import json
import logging
import math
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from core.runtime.errors import record_degradation
from core.runtime.file_write_gateway import get_file_write_gateway
from core.runtime.lockdep import checked_lock
from core.runtime.state_ownership import state_root

logger = logging.getLogger("Aura.MetaLearner")

_DATA_DIR = state_root() / "data" / "meta_learning"
_STATE_PATH = _DATA_DIR / "meta_state.npz"
_LOG_PATH = _DATA_DIR / "meta_learning_log.jsonl"


@dataclass
class MetaConfig:
    """Meta-learning configuration."""
    n_perturbations: int = 20       # ES population size
    perturbation_sigma: float = 0.01  # Perturbation noise scale
    meta_lr: float = 0.005          # Meta learning rate
    max_inner_steps: int = 5        # Inner loop optimization steps
    discount_gamma: float = 0.99    # Reward discount
    antithetic: bool = True         # Use antithetic sampling (halves variance)
    seed: int = 137
    max_log_entries: int = 1000


@dataclass
class MetaTask:
    """A meta-learning task (episode)."""
    name: str
    evaluate: Callable[[np.ndarray], float]  # params -> reward
    parameter_dim: int
    baseline_params: np.ndarray

    def __post_init__(self):
        self.baseline_params = np.asarray(
            self.baseline_params, dtype=np.float64
        ).ravel()


@dataclass
class MetaStep:
    """Record of a single meta-learning step."""
    cycle_id: int
    task_name: str
    mean_reward: float
    best_reward: float
    worst_reward: float
    gradient_norm: float
    param_delta_norm: float
    n_evaluations: int
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cycle_id": self.cycle_id,
            "task_name": self.task_name,
            "mean_reward": round(self.mean_reward, 6),
            "best_reward": round(self.best_reward, 6),
            "gradient_norm": round(self.gradient_norm, 6),
            "param_delta_norm": round(self.param_delta_norm, 6),
            "n_evaluations": self.n_evaluations,
            "timestamp": self.timestamp,
        }


class ESMetaOptimizer:
    """Evolution Strategies meta-optimizer.

    Estimates gradients via perturbation-reward correlation:
        grad ~ (1/sigma) * mean(epsilon_i * reward_i)
    where epsilon_i ~ N(0, sigma^2 I) and reward_i = f(theta + epsilon_i).

    With antithetic sampling:
        For each epsilon_i, also evaluate f(theta - epsilon_i)
        This halves the variance of the gradient estimate.
    """

    def __init__(self, config: MetaConfig | None = None) -> None:
        self.config = config or MetaConfig()
        self._rng = np.random.default_rng(self.config.seed)

    def estimate_gradient(
        self,
        params: np.ndarray,
        evaluate: Callable[[np.ndarray], float],
    ) -> tuple[np.ndarray, dict[str, float]]:
        """Estimate gradient via Evolution Strategies.

        Returns:
            (gradient_estimate, metrics_dict)
        """
        params = np.asarray(params, dtype=np.float64).ravel()
        d = params.size
        sigma = self.config.perturbation_sigma
        n = self.config.n_perturbations

        epsilons = []
        rewards = []

        for _ in range(n):
            eps = self._rng.standard_normal(d) * sigma
            epsilons.append(eps)

            # Positive perturbation
            r_pos = evaluate(params + eps)
            rewards.append(r_pos)

            if self.config.antithetic:
                # Negative perturbation (antithetic)
                r_neg = evaluate(params - eps)
                rewards.append(r_neg)

        rewards_arr = np.array(rewards, dtype=np.float64)

        # An evaluator that returns NaN/inf would otherwise flow through
        # normalization (poisoning mean and std for EVERY perturbation, not
        # just its own), then into the gradient, the parameters, the persisted
        # arrays, and finally the status report. Non-finite rewards are
        # neutralised to the population mean so one bad evaluation cannot
        # destroy the step.
        finite_mask = np.isfinite(rewards_arr)
        if not np.all(finite_mask):
            n_bad = int(np.sum(~finite_mask))
            if not np.any(finite_mask):
                record_degradation(
                    'meta_learner',
                    ValueError("all evaluator rewards were non-finite"),
                    severity="warning",
                    action="skipped the ES gradient estimate for this task",
                )
                return np.zeros(d, dtype=np.float64), {
                    "mean_reward": 0.0, "best_reward": 0.0, "worst_reward": 0.0,
                    "n_evaluations": 0, "non_finite_rewards": n_bad,
                }
            record_degradation(
                'meta_learner',
                ValueError(f"{n_bad} non-finite evaluator reward(s)"),
                severity="warning",
                action="replaced non-finite rewards with the population mean",
            )
            rewards_arr = np.where(
                finite_mask, rewards_arr, float(np.mean(rewards_arr[finite_mask]))
            )

        # Normalize rewards (fitness shaping)
        mean_r = float(np.mean(rewards_arr))
        std_r = float(np.std(rewards_arr)) + 1e-8
        normalized = (rewards_arr - mean_r) / std_r

        # Estimate gradient
        gradient = np.zeros(d, dtype=np.float64)
        for i, eps in enumerate(epsilons):
            if self.config.antithetic:
                r_pos = normalized[2 * i]
                r_neg = normalized[2 * i + 1]
                gradient += eps * (r_pos - r_neg)
            else:
                gradient += eps * normalized[i]

        # ES estimator scaling. `eps` is the SCALED perturbation delta = sigma*z
        # (z ~ N(0,I)), so the canonical estimator is
        #     grad = 1/(n * sigma^2) * SUM delta_i * r_i
        # and, for antithetic pairs,
        #     grad = 1/(2n * sigma^2) * SUM delta_i * (r_i^+ - r_i^-).
        # Dividing by (n * sigma) instead left the estimate scaled by a factor
        # of sigma — with the default sigma this understated the gradient by
        # more than an order of magnitude, silently changing the effective
        # meta learning rate away from the configured one.
        denom = n * max(sigma * sigma, 1e-12)
        if self.config.antithetic:
            denom *= 2.0
        gradient /= denom
        if not np.all(np.isfinite(gradient)):
            gradient = np.nan_to_num(gradient, nan=0.0, posinf=0.0, neginf=0.0)

        metrics = {
            "mean_reward": mean_r,
            "best_reward": float(np.max(rewards_arr)),
            "worst_reward": float(np.min(rewards_arr)),
            "n_evaluations": len(rewards_arr),
            "non_finite_rewards": int(np.sum(~finite_mask)),
        }

        return gradient, metrics


class MetaLearner:
    """MAML-style meta-learner over substrate and value parameters.

    Usage:
        learner = get_meta_learner()

        # Register tasks
        learner.register_task(MetaTask(
            name="value_optimization",
            evaluate=value_eval_fn,
            parameter_dim=64,
            baseline_params=current_params,
        ))

        # Run a meta-learning cycle (during dream)
        steps = learner.meta_step()
    """

    def __init__(self, config: MetaConfig | None = None) -> None:
        self.config = config or MetaConfig()
        self._optimizer = ESMetaOptimizer(self.config)
        self._tasks: dict[str, MetaTask] = {}
        self._meta_params: dict[str, np.ndarray] = {}
        self._cycle_count = 0
        self._history: deque[MetaStep] = deque(maxlen=200)
        # Task registry, RNG, parameters, history and persistence move together.
        # Concurrent dream/adaptation calls previously interleaved evaluations
        # and overwrote each other's meta state.
        self._lock = checked_lock("meta_learner", reentrant=True)

        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        self._load()
        logger.info("MetaLearner initialized: %d tasks", len(self._tasks))

    def register_task(self, task: MetaTask) -> None:
        """Register a meta-learning task.

        Reconciles the declared parameter_dim against the baseline and any
        restored parameters. parameter_dim was previously declared and never
        checked, so a task could be optimised at a width neither its baseline
        nor its evaluator expected.
        """
        baseline = np.asarray(task.baseline_params, dtype=np.float64).ravel()
        declared = int(getattr(task, "parameter_dim", baseline.size) or baseline.size)
        if baseline.size != declared:
            raise ValueError(
                f"meta-task '{task.name}' declares parameter_dim={declared} but "
                f"its baseline has {baseline.size} parameters"
            )
        if not np.all(np.isfinite(baseline)):
            raise ValueError(f"meta-task '{task.name}' baseline is not finite")

        with self._lock:
            self._tasks[task.name] = task
            restored = self._meta_params.get(task.name)
            if restored is not None and np.asarray(restored).size != declared:
                logger.warning(
                    "Discarded restored meta-parameters for '%s': %d values do "
                    "not match the declared dimension %d.",
                    task.name, np.asarray(restored).size, declared,
                )
                self._meta_params.pop(task.name, None)
            if task.name not in self._meta_params:
                self._meta_params[task.name] = baseline.copy()
        logger.debug("Registered meta-task '%s': %d params",
                      task.name, task.parameter_dim)

    def meta_step(self, task_name: str | None = None) -> list[MetaStep]:
        """Execute one meta-learning step over registered tasks.

        Args:
            task_name: If given, only optimize this task.
                       If None, optimize all tasks.

        Returns:
            List of MetaStep records.
        """
        self._cycle_count += 1
        steps = []
        targets = [task_name] if task_name else list(self._tasks.keys())

        for tname in targets:
            task = self._tasks.get(tname)
            if task is None:
                continue

            current = self._meta_params.get(tname, task.baseline_params.copy())

            # ES gradient estimation
            gradient, metrics = self._optimizer.estimate_gradient(
                current, task.evaluate,
            )

            # Clip gradient
            grad_norm = float(np.linalg.norm(gradient))
            max_norm = 1.0
            if grad_norm > max_norm:
                gradient = gradient * (max_norm / grad_norm)
                grad_norm = max_norm

            # Meta-update (gradient ascent on reward)
            delta = self.config.meta_lr * gradient
            new_params = current + delta
            delta_norm = float(np.linalg.norm(delta))

            # ACCEPTANCE TEST. The new parameters used to be stored
            # unconditionally — every noisy ES step was committed with no
            # baseline comparison, no held-out check, and nothing to roll back
            # to, so a bad step became the new starting point permanently.
            # The candidate must beat the incumbent on a fresh evaluation
            # before it is kept; otherwise the incumbent stands.
            accepted = True
            reject_reason = ""
            if not np.all(np.isfinite(new_params)):
                accepted = False
                reject_reason = "non-finite parameters"
            else:
                try:
                    incumbent_score = float(task.evaluate(current))
                    candidate_score = float(task.evaluate(new_params))
                except (RuntimeError, TypeError, ValueError, ArithmeticError) as exc:
                    record_degradation('meta_learner', exc, severity="warning",
                                       action="rejected a meta-step it could not verify")
                    accepted, reject_reason = False, "acceptance evaluation failed"
                else:
                    if not (math.isfinite(incumbent_score) and math.isfinite(candidate_score)):
                        accepted, reject_reason = False, "non-finite acceptance scores"
                    elif candidate_score < incumbent_score:
                        accepted = False
                        reject_reason = (
                            f"candidate {candidate_score:.4f} < incumbent "
                            f"{incumbent_score:.4f}"
                        )

            if accepted:
                self._meta_params[tname] = new_params
            else:
                # Roll back to the incumbent: keeping it IS the checkpoint.
                self._meta_params[tname] = current
                delta_norm = 0.0
                logger.info(
                    "Meta-step %d/%s REJECTED (%s); keeping incumbent parameters.",
                    self._cycle_count, tname, reject_reason,
                )

            step = MetaStep(
                cycle_id=self._cycle_count,
                task_name=tname,
                mean_reward=metrics["mean_reward"],
                best_reward=metrics["best_reward"],
                worst_reward=metrics["worst_reward"],
                gradient_norm=grad_norm,
                param_delta_norm=delta_norm,
                n_evaluations=metrics["n_evaluations"],
            )
            steps.append(step)
            self._history.append(step)
            self._log_step(step)

            logger.info(
                "Meta-step %d/%s: mean_r=%.4f, grad_norm=%.4f, delta=%.4f",
                self._cycle_count, tname, step.mean_reward,
                step.gradient_norm, step.param_delta_norm,
            )

        if steps:
            self._save()

        return steps

    def get_meta_params(self, task_name: str) -> np.ndarray | None:
        """Get current meta-optimized parameters for a task."""
        return self._meta_params.get(task_name)

    def _log_step(self, step: MetaStep) -> None:
        try:
            get_file_write_gateway().append_text(
                _LOG_PATH,
                json.dumps(step.to_dict(), default=str) + "\n",
                source="adaptation.meta_learner.step_log",
            )
        except (json.JSONDecodeError, TypeError, ValueError) as _exc:
            logger.debug("Suppressed %s in core.adaptation.meta_learner: %s", type(_exc).__name__, _exc)

    # ── Persistence ─────────────────────────────────────────────────────

    def _save(self) -> None:
        try:
            save_dict: dict[str, np.ndarray] = {
                "cycle_count": np.array([self._cycle_count]),
            }
            for name, params in self._meta_params.items():
                save_dict[f"meta_{name}"] = params
            # Atomic replace through the GOVERNED gateway. Writing in place
            # meant a crash mid-write left a truncated .npz the next boot could
            # not read, silently discarding every meta-cycle completed. An
            # earlier version of this fix did its own open()+fsync+replace,
            # which bought atomicity by bypassing the write gateway — the
            # durable-write ratchet caught it. Serialising to a buffer and
            # handing the bytes to the gateway gets both.
            buffer = io.BytesIO()
            np.savez_compressed(buffer, **save_dict)
            get_file_write_gateway().write_bytes(
                _STATE_PATH,
                buffer.getvalue(),
                source="adaptation.meta_learner.state",
            )
        except (RuntimeError, AttributeError, TypeError, ValueError, OSError) as exc:
            # OSError was outside the handler, so a full disk or permission
            # failure escaped a routine save.
            record_degradation('meta_learner', exc, severity="warning",
                               action="meta-learner state not persisted this cycle")

    def _load(self) -> None:
        try:
            if not _STATE_PATH.exists():
                return
            data = np.load(str(_STATE_PATH), allow_pickle=False)
            self._cycle_count = int(data.get("cycle_count", [0])[0])
            for key in data.files:
                if not key.startswith("meta_"):
                    continue
                name = key[5:]
                arr = np.asarray(data[key], dtype=np.float64)
                # This state comes off disk unsigned. A non-finite or
                # zero-length array used to be trusted straight into the
                # parameters, where it would poison the next gradient and every
                # status report derived from it.
                if arr.size == 0 or not np.all(np.isfinite(arr)):
                    record_degradation(
                        'meta_learner',
                        ValueError(f"invalid persisted parameters for '{name}'"),
                        severity="warning",
                        action="discarded unusable persisted meta-parameters",
                    )
                    continue
                self._meta_params[name] = arr
            self._cycle_count = max(0, int(self._cycle_count))
            logger.info("Meta-learner restored (cycle %d, %d task(s))",
                        self._cycle_count, len(self._meta_params))
        except (OSError, ConnectionError, TimeoutError, ValueError, KeyError) as exc:
            record_degradation('meta_learner', exc, severity="warning",
                               action="started with no restored meta-parameters")

    def get_status(self) -> dict[str, Any]:
        task_status = {}
        for name in self._tasks:
            mp = self._meta_params.get(name)
            task_status[name] = {
                "parameter_dim": self._tasks[name].parameter_dim,
                "has_meta_params": mp is not None,
                "meta_param_norm": round(float(np.linalg.norm(mp)), 4) if mp is not None else 0.0,
            }
        recent = list(self._history)[-5:]
        return {
            "cycle_count": self._cycle_count,
            "n_tasks": len(self._tasks),
            "tasks": task_status,
            "recent_steps": [s.to_dict() for s in recent],
        }


_instance: MetaLearner | None = None


def get_meta_learner() -> MetaLearner:
    """Get or create the singleton MetaLearner."""
    global _instance
    if _instance is None:
        _instance = MetaLearner()
    return _instance
