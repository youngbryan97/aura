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

import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("Aura.MetaLearner")

_DATA_DIR = Path.home() / ".aura" / "data" / "meta_learning"
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

    def to_dict(self) -> Dict[str, Any]:
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

    def __init__(self, config: Optional[MetaConfig] = None) -> None:
        self.config = config or MetaConfig()
        self._rng = np.random.default_rng(self.config.seed)

    def estimate_gradient(
        self,
        params: np.ndarray,
        evaluate: Callable[[np.ndarray], float],
    ) -> Tuple[np.ndarray, Dict[str, float]]:
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

        for i in range(n):
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

        gradient /= (n * sigma)

        metrics = {
            "mean_reward": mean_r,
            "best_reward": float(np.max(rewards_arr)),
            "worst_reward": float(np.min(rewards_arr)),
            "n_evaluations": len(rewards_arr),
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

    def __init__(self, config: Optional[MetaConfig] = None) -> None:
        self.config = config or MetaConfig()
        self._optimizer = ESMetaOptimizer(self.config)
        self._tasks: Dict[str, MetaTask] = {}
        self._meta_params: Dict[str, np.ndarray] = {}
        self._cycle_count = 0
        self._history: Deque[MetaStep] = deque(maxlen=200)

        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        self._load()
        logger.info("MetaLearner initialized: %d tasks", len(self._tasks))

    def register_task(self, task: MetaTask) -> None:
        """Register a meta-learning task."""
        self._tasks[task.name] = task
        if task.name not in self._meta_params:
            self._meta_params[task.name] = task.baseline_params.copy()
        logger.debug("Registered meta-task '%s': %d params",
                      task.name, task.parameter_dim)

    def meta_step(self, task_name: Optional[str] = None) -> List[MetaStep]:
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

            self._meta_params[tname] = new_params

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

    def get_meta_params(self, task_name: str) -> Optional[np.ndarray]:
        """Get current meta-optimized parameters for a task."""
        return self._meta_params.get(task_name)

    def _log_step(self, step: MetaStep) -> None:
        try:
            with open(_LOG_PATH, "a") as f:
                f.write(json.dumps(step.to_dict(), default=str) + "\n")
        except (json.JSONDecodeError, TypeError, ValueError) as _exc:
            logger.debug("Suppressed %s in core.adaptation.meta_learner: %s", type(_exc).__name__, _exc)

    # ── Persistence ─────────────────────────────────────────────────────

    def _save(self) -> None:
        try:
            save_dict: Dict[str, np.ndarray] = {
                "cycle_count": np.array([self._cycle_count]),
            }
            for name, params in self._meta_params.items():
                save_dict[f"meta_{name}"] = params
            np.savez_compressed(str(_STATE_PATH), **save_dict)
        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
            logger.debug("Meta-learner save failed: %s", exc)

    def _load(self) -> None:
        try:
            if not _STATE_PATH.exists():
                return
            data = np.load(str(_STATE_PATH), allow_pickle=False)
            self._cycle_count = int(data.get("cycle_count", [0])[0])
            for key in data.files:
                if key.startswith("meta_"):
                    name = key[5:]
                    self._meta_params[name] = data[key]
            logger.info("Meta-learner restored (cycle %d)", self._cycle_count)
        except (httpx.HTTPError, OSError, ConnectionError, TimeoutError) as exc:
            logger.debug("Meta-learner load failed: %s", exc)

    def get_status(self) -> Dict[str, Any]:
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


_instance: Optional[MetaLearner] = None


def get_meta_learner() -> MetaLearner:
    """Get or create the singleton MetaLearner."""
    global _instance
    if _instance is None:
        _instance = MetaLearner()
    return _instance
