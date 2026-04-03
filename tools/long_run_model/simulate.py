from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any, Deque, Dict, Iterable, List, Tuple

from core.senses.circadian import _phase_for_hour, _smooth_circadian_arousal

from .profiles import RestartEventSpec, SimulationProfile
from .registry import RuntimeRegistry, build_registry


@dataclass(frozen=True)
class RetentionCliff:
    name: str
    threshold_s: float
    description: str
    impact: str
    reached: bool = False


@dataclass(frozen=True)
class FailureForecast:
    severity: str
    subsystem: str
    title: str
    description: str
    horizon: str
    evidence: Dict[str, Any] = field(default_factory=dict)
    recommendation: str = ""


@dataclass
class CheckpointReport:
    horizon: str
    elapsed_s: int
    organism_summary: Dict[str, Any]
    maintenance_summary: Dict[str, Any]
    storage_summary: Dict[str, Any]
    pressure_summary: Dict[str, Any]
    cliff_summary: List[RetentionCliff]
    risk_summary: List[FailureForecast]
    attractors: Dict[str, Any]
    exact_drift_summary: Dict[str, Any]
    scenario_assumptions: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["cliff_summary"] = [asdict(item) for item in self.cliff_summary]
        payload["risk_summary"] = [asdict(item) for item in self.risk_summary]
        return payload


@dataclass
class ForecastRunSummary:
    generated_at: float
    profile: Dict[str, Any]
    registry: Dict[str, Any]
    checkpoints: List[CheckpointReport]
    risk_ledger: List[FailureForecast]
    remediation_backlog: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "profile": self.profile,
            "registry": self.registry,
            "checkpoints": [item.to_dict() for item in self.checkpoints],
            "risk_ledger": [asdict(item) for item in self.risk_ledger],
            "remediation_backlog": list(self.remediation_backlog),
        }


@dataclass
class _EpisodeBucket:
    created_at_s: int
    count: int
    importance: float
    decay_rate: float
    emotional_valence: float


@dataclass
class _VectorBucket:
    created_at_s: int
    count: int
    salience: float


@dataclass
class _SimState:
    elapsed_s: int = 0
    session_count: int = 1
    continuity_pressure: float = 0.0
    continuity_scar: str = ""
    continuity_reentry_required: bool = False
    pending_initiatives: int = 1
    active_goals: int = 1
    affect_valence: float = 0.15
    affect_arousal: float = 0.55
    affect_curiosity: float = 0.55
    social_hunger: float = 0.42
    trust: float = 0.38
    joy: float = 0.22
    fear: float = 0.05
    sadness: float = 0.06
    anger: float = 0.03
    anticipation: float = 0.52
    energy_budget: float = 100.0
    curiosity_budget: float = 80.0
    social_budget: float = 90.0
    integrity_budget: float = 95.0
    growth_budget: float = 50.0
    v_curiosity: float = 0.8
    v_energy: float = 1.0
    phi: float = 0.52
    task_count: float = 42.0
    queue_depth: float = 0.0
    mean_tick_ms: float = 480.0
    rss_gb: float = 41.0
    rss_repair_headroom_gb: float = 0.0
    lock_hold_age_s: float = 0.0
    turns_total: int = 0
    tool_exec_total: int = 0
    research_cycles: int = 0
    subconscious_dreams: int = 0
    subconscious_sandbox_runs: int = 0
    episodic_memories: int = 0
    vector_memories: int = 0
    long_term_memories: int = 0
    backups: int = 0
    backup_archives_retained: int = 0
    vacuums: int = 0
    conversation_prunes: int = 0
    vector_prunes: int = 0
    ltm_consolidations: int = 0
    episodic_consolidations: int = 0
    scheduled_repair_actions: int = 0
    last_research_at_s: int = -10**9
    last_impulse_at_s: int = -10**9
    last_backup_at_s: int = 0
    last_vector_prune_at_s: int = 0
    last_vacuum_at_s: int = 0
    last_memory_prune_at_s: int = -10**9
    last_memory_unload_at_s: int = -10**9
    last_memory_critical_at_s: int = -10**9
    last_memory_prune_rss_gb: float = 0.0
    last_memory_unload_rss_gb: float = 0.0
    last_conversation_prune_at_s: int = 0
    last_ltm_consolidation_at_s: int = 0
    idle_duration_s: int = 0
    current_session_started_at_s: int = 0
    episode_buckets: Deque[_EpisodeBucket] = field(default_factory=deque)
    vector_buckets: Deque[_VectorBucket] = field(default_factory=deque)
    conversation_session_ages_s: Deque[int] = field(default_factory=lambda: deque([0]))
    restarts_triggered: List[Dict[str, Any]] = field(default_factory=list)
    max_task_count_seen: float = 42.0
    max_queue_depth_seen: float = 0.0
    max_rss_gb_seen: float = 41.0
    min_phi_seen: float = 0.52
    max_phi_seen: float = 0.52


def _parse_horizon(value: str) -> int:
    raw = str(value).strip().lower()
    if raw.endswith("h"):
        return int(float(raw[:-1]) * 3600)
    if raw.endswith("d"):
        return int(float(raw[:-1]) * 86400)
    raise ValueError(f"Unsupported horizon: {value}")


def _format_horizon(seconds: int) -> str:
    if seconds % 86400 == 0:
        return f"{seconds // 86400}d"
    return f"{seconds // 3600}h"


def _restart_schedule(profile: SimulationProfile) -> Dict[int, RestartEventSpec]:
    events: Dict[int, RestartEventSpec] = {}
    for event in profile.planned_restarts:
        events[(event.day - 1) * 86400 + (event.hour * 3600)] = event
    return events


def _is_active(profile: SimulationProfile, elapsed_s: int) -> bool:
    day_second = elapsed_s % 86400
    start = profile.activity_start_hour_local * 3600
    end = start + (profile.active_hours_per_day * 3600)
    return start <= day_second < end


def _is_idle_window(profile: SimulationProfile, elapsed_s: int, step_s: int) -> bool:
    if profile.autonomous_idle_windows_per_day <= 0:
        return False
    active_start = profile.activity_start_hour_local * 3600
    active_span = profile.active_hours_per_day * 3600
    active_end = active_start + active_span
    day_second = elapsed_s % 86400
    if day_second < active_start or day_second >= active_end:
        return False
    window_span = profile.idle_window_minutes * 60
    spacing = max(step_s, active_span // profile.autonomous_idle_windows_per_day)
    for idx in range(profile.autonomous_idle_windows_per_day):
        window_start = active_start + idx * spacing
        if window_start <= day_second < min(active_end, window_start + window_span):
            return True
    return False


def _update_will_engine(state: _SimState, arousal: float, step_s: int) -> None:
    tau = 12.5
    cycles = max(1, step_s // 60)
    for _ in range(cycles):
        flow_state = max(0.0, min(1.0, state.v_curiosity * arousal))
        temporal_gain = 1.0 + (flow_state * 2.0)
        dt = 2.0 * temporal_gain
        v = state.v_curiosity
        w = state.v_energy
        dv = v - (v ** 3 / 3.0) - w + 0.5
        dw = (v + 0.7 - 0.8 * w) / tau
        new_curiosity = max(0.0, min(1.0, v + (dv * dt)))
        new_energy = max(0.1, min(1.0, w + (dw * dt)))
        if new_energy <= 0.15:
            new_curiosity = min(1.0, new_curiosity + 0.3)
        state.v_curiosity = new_curiosity
        state.v_energy = new_energy

    state.curiosity_budget = state.v_curiosity * 100.0
    state.energy_budget = state.v_energy * 100.0


def _apply_motivation_decay(state: _SimState, registry: RuntimeRegistry, step_s: int, conversation_energy: float) -> None:
    dt = min(step_s, int(registry.motivation_kernel_dt_cap_s))
    social_decay_multiplier = max(0.1, 1.0 - conversation_energy) if conversation_energy > 0.5 else 1.0
    for name, budget in registry.motivation_budgets.items():
        if name in {"energy", "curiosity"} and registry.dual_writer_guard_active:
            continue
        decay = float(budget.get("decay", 0.0))
        level = getattr(state, f"{name}_budget")
        capacity = float(budget.get("capacity", 100.0))
        effective_decay = decay * social_decay_multiplier if name == "social" else decay
        new_level = max(0.0, min(capacity, level - (effective_decay * dt)))
        setattr(state, f"{name}_budget", float(new_level))

    if conversation_energy > 0.5:
        engagement_recovery = max(0.0, conversation_energy - 0.5) * 0.4 * dt / 60.0
        state.social_budget = min(100.0, state.social_budget + engagement_recovery)

    if state.trust > 0.6 or state.joy > 0.6:
        recovery = registry.motivation_social_recovery_per_min * dt / 60.0
        state.social_budget = min(100.0, state.social_budget + recovery)
        state.integrity_budget = min(100.0, state.integrity_budget + recovery)


def _apply_affect_decay(state: _SimState) -> None:
    momentum = 0.85
    baselines = {
        "trust": 0.06,
        "joy": 0.05,
        "fear": 0.04,
        "sadness": 0.05,
        "anger": 0.03,
        "anticipation": 0.5,
    }
    for key, baseline in baselines.items():
        current = getattr(state, key)
        setattr(state, key, max(0.0, min(1.0, (current * momentum) + (baseline * (1.0 - momentum)))))


def _apply_turn_feedback(state: _SimState, turns_this_step: int, active: bool, idle_window: bool) -> None:
    if turns_this_step > 0:
        state.anticipation = min(1.0, state.anticipation + (0.08 * turns_this_step))
        state.joy = min(1.0, state.joy + (0.05 * turns_this_step))
        state.trust = min(1.0, state.trust + (0.06 * turns_this_step))
        state.social_hunger = max(0.0, state.social_hunger - (0.05 * turns_this_step))
    elif active and idle_window:
        state.sadness = min(1.0, state.sadness + 0.05)
        state.social_hunger = min(1.0, state.social_hunger + 0.06)
        state.fear = min(1.0, state.fear + 0.02)
    elif not active:
        state.social_hunger = min(1.0, state.social_hunger + 0.02)
        state.anticipation = min(1.0, state.anticipation + 0.01)
        state.affect_curiosity = min(1.0, state.affect_curiosity + 0.01)


def _apply_system_pressures(state: _SimState) -> None:
    if state.continuity_pressure > 0.0:
        state.anticipation = min(1.0, state.anticipation + (0.04 * state.continuity_pressure))
        state.sadness = min(1.0, state.sadness + (0.04 * state.continuity_pressure))
        state.fear = min(1.0, state.fear + (0.05 * state.continuity_pressure))
        state.affect_curiosity = min(1.0, state.affect_curiosity + (0.03 * state.continuity_pressure))
        if state.continuity_reentry_required:
            state.social_hunger = min(1.0, state.social_hunger + (0.02 * state.continuity_pressure))


def _cap_state_drives(state: _SimState, registry: RuntimeRegistry) -> None:
    state.pending_initiatives = max(0, min(int(state.pending_initiatives), int(registry.pending_initiative_cap)))
    state.active_goals = max(0, min(int(state.active_goals), int(registry.active_goal_cap)))


def _derive_affect(state: _SimState) -> None:
    pos = state.joy + state.trust
    neg = state.fear + state.sadness + state.anger
    state.affect_valence = max(-1.0, min(1.0, pos - neg))
    state.affect_arousal = max(0.05, min(1.0, max(state.joy, state.trust, state.fear, state.sadness, state.anger, state.anticipation)))
    state.affect_curiosity = max(0.0, min(1.0, state.anticipation))


def _update_phi(state: _SimState, registry: RuntimeRegistry, active: bool, websocket: bool) -> None:
    integration = max(
        0.0,
        min(
            1.0,
            (1.0 - min(1.0, state.pending_initiatives / 10.0) * 0.15)
            + (state.trust * 0.10)
            - (state.fear * 0.10)
            - (state.sadness * 0.08),
        ),
    )
    differentiation = max(
        0.0,
        min(
            1.0,
            (abs(state.affect_valence) * 0.20)
            + (state.affect_arousal * 0.25)
            + (state.affect_curiosity * 0.20)
            + (0.15 if active else 0.05)
            + (0.10 if websocket else 0.0),
        ),
    )
    broadcast = max(
        0.0,
        min(
            1.0,
            (0.25 if state.turns_total else 0.10)
            + min(0.35, state.pending_initiatives * 0.05)
            + min(0.20, state.active_goals * 0.04)
            + (0.10 if state.long_term_memories > 0 else 0.0),
        ),
    )
    coupling = max(
        0.0,
        min(
            1.0,
            (state.energy_budget / 100.0) * 0.40
            + (0.20 if active else 0.05)
            + (0.15 if websocket else 0.0)
            + (0.15 * (1.0 - state.continuity_pressure))
            + (0.10 * (1.0 - min(1.0, state.social_hunger))),
        ),
    )
    state.phi = float(
        f"{((integration * 0.32) + (differentiation * 0.24) + (broadcast * 0.24) + (coupling * 0.20)):.4f}"
    )
    state.min_phi_seen = min(state.min_phi_seen, state.phi)
    state.max_phi_seen = max(state.max_phi_seen, state.phi)


def _update_episode_consolidation(state: _SimState, registry: RuntimeRegistry, now_s: int) -> None:
    if now_s % int(registry.episodic_eval_interval_s) != 0:
        return
    state.episodic_consolidations += 1
    retained: Deque[_EpisodeBucket] = deque()
    for bucket in state.episode_buckets:
        age_hours = max(0.0, (now_s - bucket.created_at_s) / 3600.0)
        stability = (1.0 / bucket.decay_rate) * (1.0 + bucket.importance)
        strength = math.exp(-age_hours / stability) + (abs(bucket.emotional_valence) * 0.2)
        if strength < 0.05 and bucket.importance < 0.7:
            state.episodic_memories = max(0, state.episodic_memories - bucket.count)
            continue
        retained.append(bucket)
    state.episode_buckets = retained
    if state.episodic_memories > registry.episodic_max_episodes:
        excess = state.episodic_memories - registry.episodic_max_episodes
        state.episodic_memories -= excess


def _prune_vectors(state: _SimState, registry: RuntimeRegistry, now_s: int) -> None:
    if now_s == 0 or (now_s - state.last_vector_prune_at_s) < int(registry.vector_prune_interval_s):
        return
    state.last_vector_prune_at_s = now_s
    state.vector_prunes += 1
    neutral_hard_prune_days = max(registry.vector_prune_threshold_days, registry.vector_soft_prune_threshold_days * 2)
    retained: Deque[_VectorBucket] = deque()
    for bucket in state.vector_buckets:
        age_days = (now_s - bucket.created_at_s) / 86400.0
        if (
            (age_days >= registry.vector_prune_threshold_days and bucket.salience <= -0.2)
            or (age_days >= neutral_hard_prune_days and bucket.salience <= 0.05)
        ):
            state.vector_memories = max(0, state.vector_memories - bucket.count)
            continue
        retained.append(bucket)
    state.vector_buckets = retained


def _run_backup_maintenance(state: _SimState, registry: RuntimeRegistry, now_s: int, backup_active: bool) -> None:
    if not backup_active or now_s <= 0:
        return

    if (now_s - state.last_vacuum_at_s) >= int(registry.backup_vacuum_interval_s):
        state.vacuums += 1
        state.last_vacuum_at_s = now_s

    if (now_s - state.last_backup_at_s) >= int(registry.backup_interval_s):
        state.backups += 1
        state.backup_archives_retained = min(registry.backup_max_backups, state.backup_archives_retained + 1)
        state.last_backup_at_s = now_s


def _apply_runtime_repairs(state: _SimState, registry: RuntimeRegistry, now_s: int) -> None:
    rss_mb = state.rss_gb * 1024.0
    pre_repair_rss_gb = state.rss_gb

    if (
        rss_mb >= registry.memory_governor_threshold_critical_mb
        and (now_s - state.last_memory_critical_at_s) >= int(registry.memory_governor_critical_cooldown_s)
    ):
        reclaimed_gb = 10.0
        state.scheduled_repair_actions += 1
        pruned_vectors = max(50, int(state.vector_memories * 0.18))
        pruned_episodic = max(10, int(state.episodic_memories * 0.08))
        state.vector_memories = max(0, state.vector_memories - pruned_vectors)
        state.episodic_memories = max(0, state.episodic_memories - pruned_episodic)
        state.rss_repair_headroom_gb = min(18.0, state.rss_repair_headroom_gb + reclaimed_gb)
        state.rss_gb = max(28.0, state.rss_gb - reclaimed_gb)
        state.task_count = max(22.0, state.task_count - 8.0)
        state.queue_depth = max(0.0, state.queue_depth - 4.0)
        state.fear = min(1.0, state.fear + 0.08)
        state.last_memory_critical_at_s = now_s
    elif (
        rss_mb >= registry.memory_governor_threshold_unload_mb
        and (now_s - state.last_memory_unload_at_s) >= int(registry.memory_governor_unload_cooldown_s)
        and (
            state.last_memory_unload_rss_gb == 0.0
            or rss_mb >= ((state.last_memory_unload_rss_gb * 1024.0) + registry.memory_governor_unload_hysteresis_mb)
        )
    ):
        reclaimed_gb = 7.0
        state.scheduled_repair_actions += 1
        state.rss_repair_headroom_gb = min(14.0, state.rss_repair_headroom_gb + reclaimed_gb)
        state.rss_gb = max(30.0, state.rss_gb - reclaimed_gb)
        state.task_count = max(24.0, state.task_count - 4.0)
        state.last_memory_unload_at_s = now_s
        state.last_memory_unload_rss_gb = pre_repair_rss_gb
    elif (
        rss_mb >= registry.memory_governor_threshold_prune_mb
        and (now_s - state.last_memory_prune_at_s) >= int(registry.memory_governor_prune_cooldown_s)
        and (
            state.last_memory_prune_rss_gb == 0.0
            or rss_mb >= ((state.last_memory_prune_rss_gb * 1024.0) + registry.memory_governor_prune_hysteresis_mb)
        )
    ):
        reclaimed_gb = 3.0
        state.scheduled_repair_actions += 1
        state.vector_memories = max(0, state.vector_memories - max(25, int(state.vector_memories * 0.08)))
        state.rss_repair_headroom_gb = min(8.0, state.rss_repair_headroom_gb + reclaimed_gb)
        state.rss_gb = max(32.0, state.rss_gb - reclaimed_gb)
        state.last_memory_prune_at_s = now_s
        state.last_memory_prune_rss_gb = pre_repair_rss_gb

    queue_threshold = max(1.0, registry.state_commit_queue_maxsize * 0.75)
    if registry.state_queue_repair_enabled and state.queue_depth >= queue_threshold:
        state.scheduled_repair_actions += 1
        state.queue_depth = max(1.0, state.queue_depth * 0.35)
        state.mean_tick_ms = max(240.0, state.mean_tick_ms - 500.0)

    if state.lock_hold_age_s > registry.lock_watchdog_threshold_s and registry.lock_watchdog_auto_repair:
        state.scheduled_repair_actions += 1
        state.lock_hold_age_s = max(0.0, state.lock_hold_age_s - registry.lock_watchdog_threshold_s)
        state.mean_tick_ms = max(240.0, state.mean_tick_ms - 650.0)
        state.task_count = max(20.0, state.task_count - 6.0)
        state.queue_depth = max(0.0, state.queue_depth - 2.0)
        state.fear = min(1.0, state.fear + 0.04)

    _cap_state_drives(state, registry)


def _prune_conversations(state: _SimState, registry: RuntimeRegistry, now_s: int) -> None:
    if now_s == 0 or (now_s - state.last_conversation_prune_at_s) < int(registry.conversation_prune_interval_s):
        return
    state.last_conversation_prune_at_s = now_s
    retained: Deque[int] = deque()
    pruned = 0
    for started_at in state.conversation_session_ages_s:
        age_days = (now_s - started_at) / 86400.0
        if age_days >= registry.conversation_retention_days:
            pruned += 1
            continue
        retained.append(started_at)
    if pruned:
        state.conversation_prunes += pruned
    state.conversation_session_ages_s = retained


def _run_research_cycle(state: _SimState, registry: RuntimeRegistry, now_s: int) -> bool:
    if (now_s - state.last_research_at_s) < int(registry.research_cycle_interval_s):
        return False
    if state.energy_budget <= 20.0 or state.affect_curiosity <= 0.3:
        return False
    if state.pending_initiatives <= 0 and state.active_goals <= 0:
        return False
    state.last_research_at_s = now_s
    state.research_cycles += 1
    state.pending_initiatives = max(0, state.pending_initiatives - 1)
    state.curiosity_budget = min(100.0, state.curiosity_budget + 20.0)
    state.anticipation = min(1.0, state.anticipation + 0.05)
    state.trust = min(1.0, state.trust + 0.03)
    state.vector_memories += 3
    state.vector_buckets.append(_VectorBucket(created_at_s=now_s, count=3, salience=0.1))
    state.long_term_memories += 1
    return True


def _apply_restart(state: _SimState, registry: RuntimeRegistry, event: RestartEventSpec, now_s: int) -> None:
    gap_seconds = event.downtime_minutes * 60
    contradiction_count = 1 if event.kind == "abrupt" else 0
    unfinished_factor = min(1.0, max(state.pending_initiatives / 4.0, state.active_goals / 4.0))
    gap_factor = min(1.0, gap_seconds / 21600.0)
    shutdown_factor = 0.0 if event.kind == "graceful" else 1.0
    contradiction_factor = min(1.0, contradiction_count / 3.0)
    failure_factor = 1.0 if event.kind != "graceful" else 0.0
    continuity_pressure = min(
        1.0,
        (gap_factor * 0.38)
        + (shutdown_factor * 0.24)
        + (contradiction_factor * 0.14)
        + (unfinished_factor * 0.14)
        + (failure_factor * 0.18),
    )
    scars: List[str] = []
    if gap_seconds >= 900:
        scars.append("time_gap")
    if event.kind != "graceful":
        scars.append("abrupt_shutdown")
        scars.append("unresolved_failure")
    if state.pending_initiatives > 0 or state.active_goals > 0:
        scars.append("unfinished_obligations")
    state.continuity_pressure = max(state.continuity_pressure, continuity_pressure)
    state.continuity_scar = ", ".join(scars)
    state.continuity_reentry_required = continuity_pressure >= 0.28 or event.kind != "graceful"
    if state.continuity_reentry_required:
        state.pending_initiatives += 1
        state.active_goals = max(state.active_goals, 1)
    _cap_state_drives(state, registry)
    state.session_count += 1
    state.conversation_session_ages_s.append(now_s + gap_seconds)
    state.restarts_triggered.append({
        "label": event.label or event.kind,
        "kind": event.kind,
        "at_s": now_s,
        "gap_seconds": gap_seconds,
        "continuity_pressure": round(continuity_pressure, 4),
    })


def _build_cliffs(registry: RuntimeRegistry, elapsed_s: int, backup_active: bool) -> List[RetentionCliff]:
    cliffs = [
        RetentionCliff(
            name="ledger_prune_window",
            threshold_s=int(registry.cognitive_ledger_prune_days * 86400),
            description="Cognitive ledger prune horizon.",
            impact="Seven-day ledger pruning begins compacting old transitions and checkpoint history.",
            reached=elapsed_s >= int(registry.cognitive_ledger_prune_days * 86400),
        ),
        RetentionCliff(
            name="episodic_eval",
            threshold_s=int(registry.episodic_eval_interval_s),
            description="Episodic memory decay evaluation window.",
            impact="Low-importance memories start becoming prune-eligible only after repeated 6-hour evaluations.",
            reached=elapsed_s >= int(registry.episodic_eval_interval_s),
        ),
        RetentionCliff(
            name="vector_soft_prune_window",
            threshold_s=int(registry.vector_soft_prune_threshold_days * 86400),
            description="Legacy soft-prune horizon for low-salience memories.",
            impact="Around 14 days, legacy maintenance paths begin considering low-salience memories for softer pruning.",
            reached=elapsed_s >= int(registry.vector_soft_prune_threshold_days * 86400),
        ),
        RetentionCliff(
            name="conversation_retention",
            threshold_s=int(registry.conversation_retention_days * 86400),
            description="Conversation session retention boundary.",
            impact="Sessions older than 30 days become eligible for daily pruning.",
            reached=elapsed_s >= int(registry.conversation_retention_days * 86400),
        ),
        RetentionCliff(
            name="vector_retention",
            threshold_s=int(registry.vector_prune_threshold_days * 86400),
            description="Low-salience vector prune horizon.",
            impact="Vectors older than 30 days can finally be dropped by the scheduled prune.",
            reached=elapsed_s >= int(registry.vector_prune_threshold_days * 86400),
        ),
        RetentionCliff(
            name="backup_window",
            threshold_s=86400,
            description="Backup cadence boundary.",
            impact="A new backup should exist once per day if automatic backup scheduling is active.",
            reached=backup_active and elapsed_s >= 86400,
        ),
    ]
    return cliffs


def _generate_risks(
    registry: RuntimeRegistry,
    state: _SimState,
    horizon_label: str,
    cliffs: List[RetentionCliff],
    backup_active: bool,
) -> List[FailureForecast]:
    risks: List[FailureForecast] = []
    for issue in registry.known_issues:
        if issue.status == "active":
            risks.append(FailureForecast(
                severity=issue.severity,
                subsystem=issue.issue_id,
                title=issue.title,
                description=issue.evidence,
                horizon=horizon_label,
                evidence={"recommendation": issue.recommendation},
                recommendation=issue.recommendation,
            ))

    if state.max_task_count_seen > registry.stability_max_task_count:
        risks.append(FailureForecast(
            severity="critical",
            subsystem="asyncio_tasks",
            title="Projected task explosion exceeds StabilityGuardian threshold",
            description="Scenario-bound task count crosses the guardian cap.",
            horizon=horizon_label,
            evidence={"max_task_count": round(state.max_task_count_seen, 1), "threshold": registry.stability_max_task_count},
            recommendation="Reduce unsupervised background spawning or tighten bounded trackers around high-frequency loops.",
        ))

    if state.max_queue_depth_seen >= registry.state_commit_queue_maxsize * 0.75:
        risks.append(FailureForecast(
            severity="warning",
            subsystem="state_repository",
            title="State mutation queue approaches saturation",
            description="Projected write pressure pushes the bounded commit queue close to the drop/coalesce path.",
            horizon=horizon_label,
            evidence={"max_queue_depth": round(state.max_queue_depth_seen, 1), "queue_max": registry.state_commit_queue_maxsize},
            recommendation="Reduce commit fan-out or increase coalescing around bursty write sources.",
        ))

    if state.min_phi_seen <= registry.phi_dormant:
        risks.append(FailureForecast(
            severity="warning",
            subsystem="phi",
            title="Phi falls into dormant territory during forecast",
            description="Projected integration dips below the dormant threshold for at least one interval.",
            horizon=horizon_label,
            evidence={"min_phi": round(state.min_phi_seen, 4), "dormant_threshold": registry.phi_dormant},
            recommendation="Inspect whether continuity pressure, low broadcast richness, or low coupling are starving integration.",
        ))

    if state.max_phi_seen >= 0.95:
        risks.append(FailureForecast(
            severity="info",
            subsystem="phi",
            title="Phi approaches saturation under forecast assumptions",
            description="Projected integration stays very high for long stretches, which may flatten autonomy scaling or mask degraded nuance.",
            horizon=horizon_label,
            evidence={"max_phi": round(state.max_phi_seen, 4)},
            recommendation="Add runtime saturation alerts and verify that high phi still reflects rich, not merely repetitive, activity.",
        ))

    if not backup_active:
        risks.append(FailureForecast(
            severity="warning",
            subsystem="backups",
            title="Automatic backup cadence does not appear wired into the active runtime",
            description="BackupManager scheduling exists in code, but no guaranteed active boot registration was found by the registry pass.",
            horizon=horizon_label,
            evidence={"projected_backups": state.backups},
            recommendation="Either wire automatic backups into boot explicitly or treat backups as manual-only in operational docs.",
        ))
    elif horizon_label not in {"24h"} and state.backups == 0:
        risks.append(FailureForecast(
            severity="warning",
            subsystem="backups",
            title="Forecast horizon crossed a backup window without a generated backup",
            description="The simulation expected backup cadence to fire, but no backup archive was retained by this checkpoint.",
            horizon=horizon_label,
            evidence={"backup_archives_retained": state.backup_archives_retained},
            recommendation="Verify backup scheduling remains registered and healthy across long uptimes.",
        ))

    if any(item.name == "vector_retention" and item.reached for item in cliffs) and state.vector_prunes == 0:
        risks.append(FailureForecast(
            severity="warning",
            subsystem="vector_memory",
            title="Vector retention cliff reached without a prune event",
            description="The forecast crossed the 30-day low-salience boundary but no vector prune fired.",
            horizon=horizon_label,
            evidence={"vector_memories": state.vector_memories},
            recommendation="Ensure periodic vector pruning remains scheduled independently of memory pressure.",
        ))

    if state.lock_hold_age_s > registry.lock_watchdog_threshold_s and not registry.lock_watchdog_auto_repair:
        risks.append(FailureForecast(
            severity="critical",
            subsystem="lock_watchdog",
            title="Projected lock stall exceeds watchdog threshold without auto-repair",
            description="The scenario accumulated a long-held lock beyond the watchdog threshold and no automatic intervention is available.",
            horizon=horizon_label,
            evidence={"lock_hold_age_s": round(state.lock_hold_age_s, 2), "threshold_s": registry.lock_watchdog_threshold_s},
            recommendation="Keep lock watchdog callbacks wired into robust locks so deadlocks do not linger indefinitely.",
        ))

    if registry.stability_repair_uses_task_cancellation and state.max_task_count_seen > (registry.stability_max_task_count * 0.6):
        risks.append(FailureForecast(
            severity="info",
            subsystem="stability_guardian",
            title="Task pressure is still partially mitigated by cancellation fallback",
            description="The forecast did not necessarily exceed the hard cap, but the repair path still depends in part on shedding anonymous unsupervised tasks.",
            horizon=horizon_label,
            evidence={"max_task_count": round(state.max_task_count_seen, 1)},
            recommendation="Continue migrating background loops to bounded tracked tasks so cancellation becomes exceptional rather than normal.",
        ))

    return risks


def _build_remediation_backlog(risks: List[FailureForecast]) -> List[Dict[str, Any]]:
    severity_rank = {"critical": 0, "error": 1, "warning": 2, "info": 3}
    grouped: Dict[tuple[str, str], Dict[str, Any]] = {}

    for risk in risks:
        key = (risk.subsystem, risk.title)
        current = grouped.get(key)
        candidate = {
            "subsystem": risk.subsystem,
            "title": risk.title,
            "severity": risk.severity,
            "recommendation": risk.recommendation,
            "horizons": [risk.horizon],
            "count": 1,
        }
        if current is None:
            grouped[key] = candidate
            continue
        current["count"] += 1
        current["horizons"] = sorted(set(list(current["horizons"]) + [risk.horizon]))
        if severity_rank.get(risk.severity, 99) < severity_rank.get(current["severity"], 99):
            current["severity"] = risk.severity
            current["recommendation"] = risk.recommendation

    backlog = list(grouped.values())
    backlog.sort(key=lambda item: (severity_rank.get(item["severity"], 99), -item["count"], item["subsystem"], item["title"]))
    return backlog


def _build_checkpoint(
    registry: RuntimeRegistry,
    profile: SimulationProfile,
    horizon_s: int,
    horizon_label: str,
    state: _SimState,
    backup_active: bool,
) -> CheckpointReport:
    cliffs = _build_cliffs(registry, horizon_s, backup_active)
    risks = _generate_risks(registry, state, horizon_label, cliffs, backup_active)
    attractors = {
        "energy_budget_band": [round(max(0.0, state.energy_budget - 8.0), 2), round(min(100.0, state.energy_budget + 8.0), 2)],
        "curiosity_budget_band": [round(max(0.0, state.curiosity_budget - 10.0), 2), round(min(100.0, state.curiosity_budget + 10.0), 2)],
        "social_hunger_band": [round(max(0.0, state.social_hunger - 0.08), 3), round(min(1.0, state.social_hunger + 0.08), 3)],
        "phi_band": [round(max(0.0, state.phi - 0.08), 4), round(min(1.0, state.phi + 0.08), 4)],
    }
    return CheckpointReport(
        horizon=horizon_label,
        elapsed_s=horizon_s,
        organism_summary={
            "affect": {
                "valence": round(state.affect_valence, 4),
                "arousal": round(state.affect_arousal, 4),
                "curiosity": round(state.affect_curiosity, 4),
                "social_hunger": round(state.social_hunger, 4),
                "trust": round(state.trust, 4),
                "joy": round(state.joy, 4),
                "fear": round(state.fear, 4),
                "sadness": round(state.sadness, 4),
            },
            "motivation": {
                "energy": round(state.energy_budget, 2),
                "curiosity": round(state.curiosity_budget, 2),
                "social": round(state.social_budget, 2),
                "integrity": round(state.integrity_budget, 2),
                "growth": round(state.growth_budget, 2),
            },
            "phi": round(state.phi, 4),
            "continuity": {
                "session_count": state.session_count,
                "continuity_pressure": round(state.continuity_pressure, 4),
                "continuity_scar": state.continuity_scar or "none",
                "reentry_required": state.continuity_reentry_required,
                "restarts": list(state.restarts_triggered),
            },
            "initiative": {
                "pending_initiatives": state.pending_initiatives,
                "active_goals": state.active_goals,
                "research_cycles": state.research_cycles,
            },
        },
        maintenance_summary={
            "research_cycles": state.research_cycles,
            "subconscious_dreams": state.subconscious_dreams,
            "subconscious_sandbox_runs": state.subconscious_sandbox_runs,
            "episodic_consolidations": state.episodic_consolidations,
            "ltm_consolidations": state.ltm_consolidations,
            "vector_prunes": state.vector_prunes,
            "conversation_prunes": state.conversation_prunes,
            "db_vacuums": state.vacuums,
            "backups": state.backups,
            "backup_archives_retained": state.backup_archives_retained,
            "scheduled_repair_actions": state.scheduled_repair_actions,
        },
        storage_summary={
            "turns_total": state.turns_total,
            "tool_exec_total": state.tool_exec_total,
            "conversation_sessions": len(state.conversation_session_ages_s),
            "episodic_memories": state.episodic_memories,
            "vector_memories": state.vector_memories,
            "long_term_memories": state.long_term_memories,
            "backups": state.backups,
            "backup_archives_retained": state.backup_archives_retained,
        },
        pressure_summary={
            "task_count": round(state.task_count, 2),
            "max_task_count_seen": round(state.max_task_count_seen, 2),
            "queue_depth": round(state.queue_depth, 2),
            "max_queue_depth_seen": round(state.max_queue_depth_seen, 2),
            "mean_tick_ms": round(state.mean_tick_ms, 2),
            "rss_gb": round(state.rss_gb, 2),
            "rss_percent": round((state.rss_gb / registry.total_ram_gb) * 100.0, 2),
            "lock_hold_age_s": round(state.lock_hold_age_s, 2),
            "scheduled_repair_actions": state.scheduled_repair_actions,
        },
        cliff_summary=cliffs,
        risk_summary=risks,
        attractors=attractors,
        exact_drift_summary={
            "liquid_time_clamp_s": registry.liquid_time_clamp_s,
            "circadian_cycle_hours": 24,
            "episodic_eval_interval_s": registry.episodic_eval_interval_s,
            "ltm_consolidation_interval_s": registry.ltm_consolidation_interval_s,
            "conversation_retention_days": registry.conversation_retention_days,
            "vector_prune_threshold_days": registry.vector_prune_threshold_days,
            "vector_soft_prune_threshold_days": registry.vector_soft_prune_threshold_days,
            "cognitive_ledger_prune_days": registry.cognitive_ledger_prune_days,
            "backup_interval_s": registry.backup_interval_s,
            "backup_vacuum_interval_s": registry.backup_vacuum_interval_s,
        },
        scenario_assumptions={
            "profile": profile.name,
            "hardware": registry.hardware_model,
            "turns_per_day": profile.foreground_turns_per_day,
            "tools_per_day": profile.tool_executions_per_day,
            "websocket_active": profile.websocket_active_during_active_hours,
            "backup_wired": backup_active,
            "lock_watchdog_auto_repair": registry.lock_watchdog_auto_repair,
            "state_queue_repair_enabled": registry.state_queue_repair_enabled,
            "note": "Queue depth, task count, lock age, tick latency, and RSS are scenario-bound forecasts for an M5 64GB MacBook Pro, not deterministic proofs.",
        },
    )


def run_forecast(
    profile: SimulationProfile,
    horizons: Iterable[str],
    registry: RuntimeRegistry | None = None,
) -> ForecastRunSummary:
    registry = registry or build_registry()
    step_s = int(registry.simulation_step_s)
    horizon_map = sorted({_parse_horizon(item): str(item) for item in horizons}.items())
    if not horizon_map:
        raise ValueError("At least one horizon is required")
    max_horizon_s = max(item[0] for item in horizon_map)

    state = _SimState()
    state.last_vector_prune_at_s = 0
    state.last_vacuum_at_s = 0
    backup_active = bool(registry.backup_wired)
    restart_schedule = _restart_schedule(profile)
    turns_accumulator = 0.0
    tools_accumulator = 0.0
    checkpoints: List[CheckpointReport] = []

    active_steps_per_day = max(1.0, (profile.active_hours_per_day * 3600) / step_s)
    turns_per_active_step = profile.foreground_turns_per_day / active_steps_per_day
    tools_per_active_step = profile.tool_executions_per_day / active_steps_per_day

    checkpoint_targets = {seconds: raw for seconds, raw in horizon_map}

    for now_s in range(0, max_horizon_s + step_s, step_s):
        state.elapsed_s = now_s
        if now_s in restart_schedule:
            _apply_restart(state, registry, restart_schedule[now_s], now_s)

        active = _is_active(profile, now_s)
        idle_window = _is_idle_window(profile, now_s, step_s)
        websocket = bool(profile.websocket_active_during_active_hours and active)
        turns_this_step = 0
        tools_this_step = 0
        conversation_energy = 0.15

        if active:
            turns_accumulator += turns_per_active_step
            tools_accumulator += tools_per_active_step
            turns_this_step = int(turns_accumulator)
            tools_this_step = int(tools_accumulator)
            turns_accumulator -= turns_this_step
            tools_accumulator -= tools_this_step
            if idle_window:
                turns_this_step = 0
                conversation_energy = 0.2
            else:
                conversation_energy = 0.8 if turns_this_step > 0 else 0.35
        else:
            conversation_energy = 0.05

        _update_will_engine(state, _smooth_circadian_arousal((now_s % 86400) / 3600.0), step_s)
        _apply_motivation_decay(state, registry, step_s, conversation_energy)
        _apply_affect_decay(state)
        _apply_turn_feedback(state, turns_this_step, active, idle_window)
        _apply_system_pressures(state)
        _derive_affect(state)

        if (now_s - state.last_impulse_at_s) >= 60:
            threshold_base = max(0.5, min(1.5, 0.7 + (0.6 * max(state.phi, registry.phi_reactive))))
            threshold = max(0.5, min(0.95, 0.8 / max(0.5, threshold_base)))
            is_bored = state.affect_arousal < 0.2
            if state.affect_curiosity > threshold or state.social_hunger > threshold or is_bored:
                state.pending_initiatives += 1
                state.active_goals = max(state.active_goals, 1)
                state.last_impulse_at_s = now_s
                if is_bored:
                    state.affect_arousal = min(1.0, state.affect_arousal + 0.2)

        silent = (not active) or idle_window or turns_this_step == 0
        if silent:
            state.idle_duration_s += step_s
            if state.idle_duration_s >= int(registry.subconscious_idle_threshold_s):
                if (now_s % int(registry.subconscious_dream_interval_s)) == 0:
                    state.subconscious_dreams += 1
                if (now_s % int(registry.subconscious_sandbox_interval_s)) == 0:
                    state.subconscious_sandbox_runs += 1
                if _run_research_cycle(state, registry, now_s):
                    state.affect_valence = min(1.0, state.affect_valence + 0.05)
        else:
            state.idle_duration_s = 0

        state.turns_total += turns_this_step
        state.tool_exec_total += tools_this_step
        if turns_this_step:
            state.episodic_memories += turns_this_step
            state.episode_buckets.append(_EpisodeBucket(now_s, turns_this_step, 0.35, 0.02, 0.1))
            state.vector_memories += turns_this_step // 2
            if turns_this_step // 2:
                state.vector_buckets.append(_VectorBucket(now_s, turns_this_step // 2, -0.1))
        if tools_this_step:
            state.vector_memories += tools_this_step
            state.vector_buckets.append(_VectorBucket(now_s, tools_this_step, 0.0))

        _cap_state_drives(state, registry)

        _update_episode_consolidation(state, registry, now_s)
        _prune_vectors(state, registry, now_s)
        _prune_conversations(state, registry, now_s)
        _run_backup_maintenance(state, registry, now_s, backup_active)
        if now_s and (now_s - state.last_ltm_consolidation_at_s) >= int(registry.ltm_consolidation_interval_s):
            state.ltm_consolidations += 1
            state.last_ltm_consolidation_at_s = now_s

        phase = _phase_for_hour((now_s % 86400) / 3600.0)
        if phase.value in {"night", "deep_night"}:
            state.affect_arousal = max(0.08, min(state.affect_arousal, 0.45))

        _update_phi(state, registry, active, websocket)
        state.continuity_pressure = max(0.0, round(state.continuity_pressure * 0.999, 6))
        if state.continuity_pressure < 0.05:
            state.continuity_reentry_required = False

        state.task_count = (
            34.0
            + (10.0 if websocket else 2.0)
            + min(18.0, state.pending_initiatives * 1.8)
            + (6.0 if state.research_cycles and (now_s - state.last_research_at_s) < 1800 else 0.0)
        )
        commit_generation = turns_this_step + tools_this_step + (1 if state.research_cycles and (now_s - state.last_research_at_s) < 300 else 0)
        service_capacity = 20.0 if registry.total_ram_gb >= 64 else 12.0
        state.queue_depth = max(0.0, min(registry.state_commit_queue_maxsize, state.queue_depth + commit_generation - service_capacity))
        state.mean_tick_ms = (
            220.0
            + (180.0 if active else 40.0)
            + (250.0 if websocket else 0.0)
            + (160.0 if commit_generation > 2 else 0.0)
            + (state.queue_depth * 15.0)
            + (state.pending_initiatives * 12.0)
        )
        state.rss_repair_headroom_gb = max(
            0.0,
            state.rss_repair_headroom_gb - ((step_s / 3600.0) * 0.25),
        )
        base_rss_gb = (
            34.0
            + (10.5 if websocket else 3.0)
            + min(8.0, state.vector_memories / 2500.0)
            + min(4.0, state.episodic_memories / 5000.0)
            + (4.0 if active else 0.0)
        )
        state.rss_gb = max(20.0, base_rss_gb - state.rss_repair_headroom_gb)
        if state.mean_tick_ms > registry.stability_max_tick_lag_ms:
            state.lock_hold_age_s += step_s
        else:
            state.lock_hold_age_s = max(0.0, state.lock_hold_age_s - step_s)

        state.max_task_count_seen = max(state.max_task_count_seen, state.task_count)
        state.max_queue_depth_seen = max(state.max_queue_depth_seen, state.queue_depth)
        state.max_rss_gb_seen = max(state.max_rss_gb_seen, state.rss_gb)
        _apply_runtime_repairs(state, registry, now_s)
        _cap_state_drives(state, registry)

        if now_s in checkpoint_targets:
            checkpoints.append(
                _build_checkpoint(
                    registry,
                    profile,
                    now_s,
                    checkpoint_targets[now_s],
                    state,
                    backup_active,
                )
            )

    risk_ledger: List[FailureForecast] = []
    for checkpoint in checkpoints:
        risk_ledger.extend(checkpoint.risk_summary)
    remediation_backlog = _build_remediation_backlog(risk_ledger)

    return ForecastRunSummary(
        generated_at=time.time(),
        profile=asdict(profile),
        registry=registry.to_dict(),
        checkpoints=checkpoints,
        risk_ledger=risk_ledger,
        remediation_backlog=remediation_backlog,
    )
