"""core/runtime/field_trials.py — deterministic A/B trials over own behaviour.

Clean-room adoption of Chromium's Finch field trials.

Aura changes its own behaviour constantly — a new verifier, a different
retrieval weighting, a revised steering dose — and the evidence for
whether a change helped is, today, a before-and-after comparison across
different days, different conversations, and a runtime that was not the
same in any other respect. It is the oldest mistake in performance work, and
it is worse here because the outcome measures are noisy.

A field trial makes the comparison honest in the only way that works:
**run both arms concurrently, in the same runtime, over the same
population of work**, and key every metric by arm.

The details that make it usable rather than a foot-gun:

* **Assignment is deterministic and sticky.** A stable entropy source (a
  per-installation id) hashed with the trial name gives the same group
  every time. A trial that reassigns on restart produces a mixture, and a
  mixture measures nothing.
* **Assignment happens once, at first query, and is then frozen.** Two
  call sites asking the same trial get the same answer within a process
  even if the weights change under them.
* **A trial can be forced** for a specific group, which is how you debug
  an arm without waiting to be assigned to it, and how tests get
  determinism.
* **Every trial has an owner, a hypothesis, and metrics it claims to
  move.** A trial nobody can state a hypothesis for cannot produce a
  conclusion, only a number.
* **Trials expire.** A permanent experiment is a config flag with extra
  steps, and it keeps a dead arm alive in the code forever.
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("Aura.FieldTrials")

#: Trials older than this are reported as needing a conclusion.
DEFAULT_TRIAL_DAYS = 90

#: The group name a disabled or expired trial always returns.
DEFAULT_GROUP = "default"


@dataclass(frozen=True)
class TrialGroup:
    name: str
    weight: float
    description: str = ""


@dataclass(frozen=True)
class TrialSpec:
    name: str
    hypothesis: str
    owner: str
    groups: tuple[TrialGroup, ...]
    #: Metrics this trial claims to move. Stated up front so the analysis
    #: cannot be chosen after seeing the data.
    metrics: tuple[str, ...] = ()
    expires_days: int = DEFAULT_TRIAL_DAYS
    enabled: bool = True

    def total_weight(self) -> float:
        return sum(g.weight for g in self.groups)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "hypothesis": self.hypothesis,
            "owner": self.owner,
            "groups": [
                {"name": g.name, "weight": g.weight, "description": g.description}
                for g in self.groups
            ],
            "metrics": list(self.metrics),
            "expires_days": self.expires_days,
            "enabled": self.enabled,
        }


@dataclass
class _Trial:
    spec: TrialSpec
    declared_at: float
    assigned_group: str | None = None
    forced_group: str | None = None
    queries: int = 0
    observations: dict[str, list[float]] = field(default_factory=dict)


def _entropy_source() -> str:
    """A stable per-installation value, so assignment is sticky.

    Derived from the installation id when one exists, and from the host
    otherwise. Never random per process — a trial that reassigns every
    restart produces a mixture, and a mixture measures nothing.
    """
    from core.runtime.flags import FlagKind, declare

    override = str(
        declare(
            "AURA_FIELD_TRIAL_ENTROPY",
            kind=FlagKind.STRING,
            default="",
            description=(
                "Overrides the per-installation entropy that makes trial "
                "assignment sticky. Set it to pin a specific arm, or to give two "
                "machines the same assignment."
            ),
            owner="core/runtime/field_trials.py",
        ).value()
        or ""
    ).strip()
    if override:
        return override
    try:
        from core.config import config

        marker = os.path.join(str(config.paths.data_dir), "runtime", "install_id")
        if os.path.exists(marker):
            with open(marker, encoding="utf-8") as handle:
                value = handle.read().strip()
            if value:
                return value
    except Exception:  # noqa: BLE001
        logger.debug("install id unavailable for trial entropy", exc_info=True)
    import socket

    return socket.gethostname()


class FieldTrials:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._trials: dict[str, _Trial] = {}
        self._entropy = _entropy_source()

    # ── declaration ───────────────────────────────────────────────────
    def declare(self, spec: TrialSpec) -> TrialSpec:
        if not spec.hypothesis.strip() or not spec.owner.strip():
            raise ValueError(
                f"trial {spec.name!r} needs a hypothesis and an owner — a trial "
                "nobody can state a hypothesis for produces a number, not a conclusion"
            )
        if not spec.groups:
            raise ValueError(f"trial {spec.name!r} declares no groups")
        if spec.total_weight() <= 0:
            raise ValueError(f"trial {spec.name!r} has zero total weight")
        with self._lock:
            existing = self._trials.get(spec.name)
            if existing is not None:
                if existing.spec != spec:
                    raise ValueError(
                        f"trial {spec.name!r} already declared by {existing.spec.owner}; "
                        "changing an in-flight trial invalidates its own data"
                    )
                return spec
            self._trials[spec.name] = _Trial(spec=spec, declared_at=time.time())
            return spec

    # ── assignment ────────────────────────────────────────────────────
    def _assign(self, trial: _Trial) -> str:
        """Deterministic from (entropy, trial name). Never random."""
        digest = hashlib.sha256(
            f"{self._entropy}::{trial.spec.name}".encode()
        ).digest()
        # 8 bytes of the digest as a fraction in [0, 1).
        fraction = int.from_bytes(digest[:8], "big") / float(1 << 64)
        cursor = 0.0
        total = trial.spec.total_weight()
        for group in trial.spec.groups:
            cursor += group.weight / total
            if fraction < cursor:
                return group.name
        return trial.spec.groups[-1].name

    def group(self, name: str) -> str:
        """The assigned group. Stable for the life of the installation."""
        with self._lock:
            trial = self._trials.get(name)
            if trial is None:
                return DEFAULT_GROUP
            trial.queries += 1
            if trial.forced_group is not None:
                return trial.forced_group
            if not trial.spec.enabled or self._expired_locked(trial):
                return DEFAULT_GROUP
            if trial.assigned_group is None:
                trial.assigned_group = self._assign(trial)
                logger.info(
                    "🧪 field trial %r → group %r (%s)",
                    name,
                    trial.assigned_group,
                    trial.spec.hypothesis,
                )
            return trial.assigned_group

    def in_group(self, name: str, group: str) -> bool:
        return self.group(name) == group

    def force(self, name: str, group: str | None) -> bool:
        """Pin a trial to a group — for debugging an arm, and for tests."""
        with self._lock:
            trial = self._trials.get(name)
            if trial is None:
                return False
            if group is not None and group not in {g.name for g in trial.spec.groups}:
                if group != DEFAULT_GROUP:
                    return False
            trial.forced_group = group
            return True

    @staticmethod
    def _expired_locked(trial: _Trial) -> bool:
        age_days = (time.time() - trial.declared_at) / 86400.0
        return age_days > trial.spec.expires_days

    # ── observation ───────────────────────────────────────────────────
    def observe(self, name: str, metric: str, value: float) -> bool:
        """Record an outcome, keyed by the arm that produced it."""
        group = self.group(name)
        with self._lock:
            trial = self._trials.get(name)
            if trial is None:
                return False
            if trial.spec.metrics and metric not in trial.spec.metrics:
                # Metrics are declared up front so the analysis cannot be
                # chosen after seeing the data.
                logger.debug(
                    "trial %r: metric %r was not declared; recorded anyway but "
                    "excluded from the trial's own conclusion",
                    name,
                    metric,
                )
            trial.observations.setdefault(f"{group}::{metric}", []).append(float(value))
            return True

    def results(self, name: str) -> dict[str, Any]:
        """Per-arm summaries for the declared metrics."""
        with self._lock:
            trial = self._trials.get(name)
            if trial is None:
                return {"trial": name, "declared": False}
            spec = trial.spec
            observations = {k: list(v) for k, v in trial.observations.items()}
            assigned = trial.assigned_group
            forced = trial.forced_group

        arms: dict[str, dict[str, Any]] = {}
        for key, values in observations.items():
            group, _, metric = key.partition("::")
            if not values:
                continue
            ordered = sorted(values)
            arms.setdefault(group, {})[metric] = {
                "n": len(values),
                "mean": round(sum(values) / len(values), 6),
                "median": round(ordered[len(ordered) // 2], 6),
                "min": round(ordered[0], 6),
                "max": round(ordered[-1], 6),
            }
        return {
            "trial": name,
            "declared": True,
            "hypothesis": spec.hypothesis,
            "owner": spec.owner,
            "assigned_group": assigned,
            "forced_group": forced,
            "declared_metrics": list(spec.metrics),
            "arms": arms,
            "comparable": len(arms) > 1,
            "note": (
                "one arm only: this process was assigned a single group, which is "
                "correct — comparison happens across installations, not within one"
                if len(arms) <= 1
                else ""
            ),
        }

    def report(self) -> dict[str, Any]:
        now = time.time()
        with self._lock:
            trials = list(self._trials.values())
        return {
            "count": len(trials),
            "entropy_source_hash": hashlib.sha256(self._entropy.encode()).hexdigest()[:12],
            "trials": {
                trial.spec.name: {
                    **trial.spec.to_dict(),
                    "assigned_group": trial.assigned_group,
                    "forced_group": trial.forced_group,
                    "queries": trial.queries,
                    "age_days": round((now - trial.declared_at) / 86400.0, 2),
                    "expired": self._expired_locked(trial),
                }
                for trial in trials
            },
            "expired": [t.spec.name for t in trials if self._expired_locked(t)],
            "active_groups": {
                t.spec.name: (t.forced_group or t.assigned_group)
                for t in trials
                if (t.forced_group or t.assigned_group)
            },
        }

    def reset_for_test(self) -> None:
        with self._lock:
            self._trials.clear()


_TRIALS = FieldTrials()


def get_field_trials() -> FieldTrials:
    return _TRIALS


def declare_trial(
    name: str,
    *,
    hypothesis: str,
    owner: str,
    groups: dict[str, float],
    metrics: tuple[str, ...] = (),
    expires_days: int = DEFAULT_TRIAL_DAYS,
    enabled: bool = True,
) -> TrialSpec:
    """Declare a trial::

        declare_trial(
            "retrieval_weighting_v2",
            hypothesis="recency-weighted retrieval raises answer groundedness "
                       "without raising latency",
            owner="core/memory/retrieval.py",
            groups={"control": 0.5, "recency_weighted": 0.5},
            metrics=("groundedness_score", "turn_latency_ms"),
        )
    """
    return _TRIALS.declare(
        TrialSpec(
            name=name,
            hypothesis=hypothesis,
            owner=owner,
            groups=tuple(
                TrialGroup(name=group, weight=float(weight))
                for group, weight in groups.items()
            ),
            metrics=tuple(metrics),
            expires_days=expires_days,
            enabled=enabled,
        )
    )


def group(name: str) -> str:
    return _TRIALS.group(name)


def in_group(name: str, group_name: str) -> bool:
    return _TRIALS.in_group(name, group_name)


def observe(name: str, metric: str, value: float) -> bool:
    return _TRIALS.observe(name, metric, value)


def field_trials_report() -> dict[str, Any]:
    return _TRIALS.report()


def reset_field_trials_for_test() -> None:
    _TRIALS.reset_for_test()


__all__ = [
    "DEFAULT_GROUP",
    "DEFAULT_TRIAL_DAYS",
    "FieldTrials",
    "TrialGroup",
    "TrialSpec",
    "declare_trial",
    "field_trials_report",
    "get_field_trials",
    "group",
    "in_group",
    "observe",
    "reset_field_trials_for_test",
]
