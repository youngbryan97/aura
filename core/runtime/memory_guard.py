"""Per-actor memory + resource guardrails.

Audit constraint: a browser leak cannot kill the model, a model stall
cannot kill state persistence, a self-repair loop cannot consume all
CPU, a movie session cannot starve conversation.

This module declares a quota table per actor / subsystem so the supervisor
and the loop_guard layer can enforce caps. Real syscall enforcement
(rlimit, cgroups) lives in the platform-specific drivers; this layer is
the contract + the in-process tracker.
"""
from __future__ import annotations


import time
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class ActorQuota:
    actor: str
    max_memory_mb: int
    max_threads: int
    max_open_fds: int
    max_subprocess_count: int
    max_browser_contexts: int
    max_queue_depth: int
    max_cpu_seconds_per_minute: float


DEFAULT_QUOTAS: Dict[str, ActorQuota] = {
    "model_runtime": ActorQuota(
        actor="model_runtime",
        max_memory_mb=24_000,
        max_threads=64,
        max_open_fds=1024,
        max_subprocess_count=2,
        max_browser_contexts=0,
        max_queue_depth=64,
        max_cpu_seconds_per_minute=55.0,
    ),
    "sensory_gate": ActorQuota(
        actor="sensory_gate",
        max_memory_mb=2_048,
        max_threads=16,
        max_open_fds=256,
        max_subprocess_count=4,
        max_browser_contexts=2,
        max_queue_depth=32,
        max_cpu_seconds_per_minute=20.0,
    ),
    "state_vault": ActorQuota(
        actor="state_vault",
        max_memory_mb=512,
        max_threads=8,
        max_open_fds=128,
        max_subprocess_count=0,
        max_browser_contexts=0,
        max_queue_depth=64,
        max_cpu_seconds_per_minute=10.0,
    ),
    "self_repair": ActorQuota(
        actor="self_repair",
        max_memory_mb=1_024,
        max_threads=4,
        max_open_fds=64,
        max_subprocess_count=2,
        max_browser_contexts=0,
        max_queue_depth=8,
        max_cpu_seconds_per_minute=15.0,
    ),
    "movie_session": ActorQuota(
        actor="movie_session",
        max_memory_mb=4_096,
        max_threads=8,
        max_open_fds=128,
        max_subprocess_count=1,
        max_browser_contexts=1,
        max_queue_depth=32,
        max_cpu_seconds_per_minute=25.0,
    ),
}


@dataclass
class ActorUsage:
    actor: str
    memory_mb: int
    threads: int
    open_fds: int
    subprocess_count: int
    browser_contexts: int
    queue_depth: int
    cpu_seconds_per_minute: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class GuardViolation:
    actor: str
    field_name: str
    observed: float
    limit: float


def evaluate_actor_usage(
    usage: ActorUsage,
    *,
    quotas: Optional[Dict[str, ActorQuota]] = None,
) -> list:
    """Return a list of GuardViolation describing every field that is
    exceeding its quota. Empty list means the actor is within bounds."""
    quotas = quotas or DEFAULT_QUOTAS
    quota = quotas.get(usage.actor)
    if quota is None:
        return []
    violations = []
    pairs = (
        ("memory_mb", usage.memory_mb, quota.max_memory_mb),
        ("threads", usage.threads, quota.max_threads),
        ("open_fds", usage.open_fds, quota.max_open_fds),
        ("subprocess_count", usage.subprocess_count, quota.max_subprocess_count),
        ("browser_contexts", usage.browser_contexts, quota.max_browser_contexts),
        ("queue_depth", usage.queue_depth, quota.max_queue_depth),
        ("cpu_seconds_per_minute", usage.cpu_seconds_per_minute, quota.max_cpu_seconds_per_minute),
    )
    for field_name, observed, limit in pairs:
        if observed > limit:
            violations.append(
                GuardViolation(actor=usage.actor, field_name=field_name, observed=observed, limit=limit)
            )
    return violations
