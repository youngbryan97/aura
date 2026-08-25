"""Virtual width: a tied-weight latent society.

K branches are K concurrent dynamical states of the SAME neural operator —
not K models. Each branch gets a distinct role (constructive solution,
counterexample search, constraint checking, …) purely through its workspace
seed basin; the weights are identical and the prompt KV is shared read-only.

Exchange: every E steps the branches communicate through a designated
communication slot — an agreement-weighted consensus of branch summaries is
blended into each branch's comm slot, so useful partial results propagate
without collapsing the ensemble.

Anti-collapse: if two branch summaries become near-parallel, deterministic
decorrelation jitter is injected into the later branch. Diversity is a
maintained invariant, not a hope.

Honest accounting: the ensemble reports total token-layer applications so
Experiment 4 can compare against equal-FLOP self-consistency sampling. If
branches don't beat sampling at equal compute, they are expensive theater —
the harness is allowed to say so.
"""

from __future__ import annotations

import copy
import hashlib
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from core.brain.llm.latent_cortex.branch_exchange import (
    BRANCH_EXCHANGE_SCHEMA,
    MAX_EXCHANGE_SOURCE_SLOTS,
    candidate_set_sha256,
    canonical_sha256,
    private_exchange_slots,
    validate_branch_exchange_receipt,
)
from core.brain.llm.latent_cortex.cognitive_operators import (
    CognitiveOperator,
    execute_cognitive_operator,
    operator_for_role,
)
from core.brain.llm.latent_cortex.context_focus import apply_context_focus
from core.brain.llm.latent_cortex.epistemic_state import OperationKind
from core.brain.llm.latent_cortex.escape import BranchEscapeLadder, EscapeConfig
from core.brain.llm.latent_cortex.loop_core import transition_metrics
from core.brain.llm.latent_cortex.recurrence import (
    HaltingController,
    WindowRunner,
    alpha_at,
    recurrence_step,
    relative_residual,
    rms_match,
)
from core.brain.llm.latent_cortex.types import BranchConfig, ComputeBudget, RecurrenceConfig
from core.brain.llm.latent_cortex.update_gate import (
    PASSTHROUGH,
    UpdateGateRuntime,
)
from core.brain.llm.latent_cortex.verified_best import (
    VerifierObservation,
    tensor_sha256,
)
from core.brain.llm.latent_cortex.workspace import (
    LatentWorkspace,
    _role_seed,
    per_position_rms,
)
from core.runtime.tensor_identity import tensor_identity_sha256

logger = logging.getLogger("Aura.LatentCortex.Branches")
_UNSCORED_BRANCH_FLOOR = -1e30

BRANCH_ISOLATION_SCHEMA = "aura.rlc.branch_isolation.v1"

if TYPE_CHECKING:
    from core.brain.llm.latent_cortex.kv_state_tree import KVStateTree

# Cognitive roles for branch seeding, in priority order (from the spec's
# "tied-weight latent society"). Branch k takes BRANCH_ROLES[k % len].
BRANCH_ROLES: tuple[str, ...] = (
    "constructive_solution",
    "counterexample_search",
    "constraint_checking",
    "causal_reconstruction",
    "analogy",
    "reverse_reasoning",
    "simplification",
    "adversarial_criticism",
)


def _tensor_sha256(array: Any) -> str:
    return tensor_identity_sha256(array)


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


@dataclass
class BranchState:
    index: int
    role: str
    workspace: LatentWorkspace
    halting: HaltingController
    z: Any = None
    anchor: Any = None
    halted: bool = False
    halt_reason: str = ""
    steps: int = 0
    score: float = 0.0
    escape: BranchEscapeLadder | None = None
    # Neural-bytecode savepoint: one snapshot slot per branch (later
    # savepoints overwrite). verify_probe(revert_on_drop) restores it.
    savepoint: Any = None
    savepoint_steps: int = 0
    kv_boundary_sha256: str = ""
    savepoint_kv_boundary_sha256: str = ""
    seed_sha256: str = ""
    candidate_sha256: str = ""
    candidate_step: int = 0
    rng_stream_sha256: str = ""
    operator: CognitiveOperator = CognitiveOperator.DIRECT_DERIVATION
    evidence_anchor_sha256: str = ""
    initial_hypothesis_sha256: str = ""
    recurrent_grounding_trace: list[dict[str, Any]] = field(default_factory=list)
    loop_stability_trace: list[dict[str, Any]] = field(default_factory=list)
    update_acceptance_trace: list[dict[str, Any]] = field(default_factory=list)
    update_gate: UpdateGateRuntime | None = None
    last_loop_delta: Any = None
    verified_best_state: Any = None
    verified_best_step: int = -1
    verified_best_state_sha256: str = ""
    verified_best_observation: dict[str, Any] = field(default_factory=dict)
    verified_best_trace: list[dict[str, Any]] = field(default_factory=list)
    verified_finalization: dict[str, Any] = field(default_factory=dict)
    uncertainty_runtime: Any = None
    uncertainty_trace: list[dict[str, Any]] = field(default_factory=list)
    mistake_locator_runtime: Any = None
    mistake_locator_trace: list[dict[str, Any]] = field(default_factory=list)
    reflector_trace: list[dict[str, Any]] = field(default_factory=list)

    def to_receipt(self) -> dict[str, Any]:
        receipt = {
            "index": self.index,
            "role": self.role,
            "steps": self.steps,
            "halted": self.halted,
            "halt_reason": self.halt_reason,
            "score": round(float(self.score), 6),
            "residual_trail": [round(r, 5) for r in self.halting.residual_trail],
        }
        if self.escape is not None and self.escape.attempts:
            receipt["escape"] = self.escape.to_receipt()
        return receipt


class BranchEnsemble:
    """K latent branches stepping over shared frozen weights + shared prompt KV.

    The ensemble serializes branch window passes (memory-light: one branch's
    activations at a time) and rewinds slot KV after every pass, so branches
    never see each other's cache side effects. Only the winner's final state
    is persisted — by the engine, not here.
    """

    def __init__(
        self,
        branches: list[BranchState],
        config: BranchConfig,
        recurrence: RecurrenceConfig,
    ) -> None:
        self.branches = branches
        self.config = config
        self.recurrence = recurrence
        self.exchanges = 0
        self.exchange_receipts: list[dict[str, Any]] = []
        self._exchange_sync_points: set[str] = set()
        # Optional per-episode observers, attached by the engine.
        self.telemetry: Any = None
        self._isolation_sealed = False
        self._isolation_failure = ""
        self._blocked_cross_exposures = 0
        self._cross_exposure_started = False
        self._first_exchange_step: int | None = None
        self._context_sha256 = ""
        self._configured_role_lesion = len({branch.role for branch in branches}) != len(branches)
        self._seed_alias_free = len({id(branch.workspace) for branch in branches}) == len(
            branches
        ) and len({id(branch.z) for branch in branches}) == len(branches)
        self._seed_states_unique = len({branch.seed_sha256 for branch in branches}) == len(branches)
        self._rng_streams_unique = len({branch.rng_stream_sha256 for branch in branches}) == len(
            branches
        )
        self._support_weights = {branch.index: 1.0 for branch in branches}
        self._kv_state_tree: KVStateTree | None = None
        self._kv_cache: Any = None

    def bind_kv_state_tree(self, tree: KVStateTree, cache: Any) -> None:
        """Bind branch savepoints and rewinds to the episode KV lineage."""

        if self._kv_state_tree is not None and self._kv_state_tree is not tree:
            raise ValueError("branch ensemble KV state tree is already bound")
        self._kv_state_tree = tree
        self._kv_cache = cache
        for branch in self.branches:
            branch.kv_boundary_sha256 = tree.root_sha256

    def set_support_weights(self, weights: dict[int, float]) -> None:
        """Install bounded correlation discounts before peer exchange."""

        if set(weights) != {branch.index for branch in self.branches}:
            raise ValueError("support weights must cover every branch exactly")
        normalized: dict[int, float] = {}
        for index, value in weights.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not 0.0 < float(value) <= 1.0
            ):
                raise ValueError("support weights must be inside (0, 1]")
            normalized[index] = float(value)
        self._support_weights = normalized

    # ── Construction ────────────────────────────────────────────────────
    @classmethod
    def seed(
        cls,
        prompt_embeddings,
        workspace_cfg,
        branch_cfg: BranchConfig,
        recurrence_cfg: RecurrenceConfig,
        runner: WindowRunner,
        cache,
        prelude_end: int,
        *,
        context_seeds: list[tuple[str, Any]] | None = None,
        escape_cfg: EscapeConfig | None = None,
    ) -> BranchEnsemble:
        import mlx.core as mx

        branches: list[BranchState] = []
        # BranchConfig refuses a role list that does not match n_branches at
        # construction; repeated here because a config can be mutated after it
        # is built, and this runs before anything touches a tensor.
        role_override = tuple(branch_cfg.roles or ())
        if role_override and len(role_override) != branch_cfg.n_branches:
            raise ValueError(
                "BranchConfig.roles must name exactly n_branches roles, got "
                f"{len(role_override)} for {branch_cfg.n_branches} branches"
            )
        context_sha256 = _tensor_sha256(prompt_embeddings)
        for k in range(branch_cfg.n_branches):
            role = role_override[k] if role_override else BRANCH_ROLES[k % len(BRANCH_ROLES)]
            operator = operator_for_role(role)
            ws = LatentWorkspace.from_prompt_embeddings(
                prompt_embeddings,
                workspace_cfg,
                branch_role=role,
                context_seeds=context_seeds,
            )
            # Prelude pass: persist=False for every branch — the engine
            # persists the WINNER's prelude at selection time. Rationale: all
            # branches must see identical caches; only one set of slot KV may
            # survive into decode.
            z0 = runner.run(ws.z, cache, 0, prelude_end, persist=False)
            ws.update(z0)
            ws.seal_context_evidence()
            halting = HaltingController(
                config=recurrence_cfg,
                baseline_rms=float(mx.mean(per_position_rms(z0))),
                best_state=z0,
            )
            branches.append(
                BranchState(
                    index=k,
                    role=role,
                    workspace=ws,
                    halting=halting,
                    z=z0,
                    anchor=z0,
                    escape=(
                        BranchEscapeLadder(escape_cfg, k)
                        if escape_cfg is not None and escape_cfg.enabled
                        else None
                    ),
                    seed_sha256=_tensor_sha256(z0),
                    rng_stream_sha256=hashlib.sha256(
                        f"{role}:{_role_seed(role, workspace_cfg.seed)}".encode()
                    ).hexdigest(),
                    operator=operator,
                )
            )
        ensemble = cls(branches, branch_cfg, recurrence_cfg)
        ensemble._context_sha256 = context_sha256
        return ensemble

    def _seal_isolation_if_ready(self) -> None:
        if self._isolation_sealed or self._isolation_failure:
            return
        required = int(self.config.isolation_steps)
        short_halts = [
            branch.index for branch in self.branches if branch.halted and branch.steps < required
        ]
        if short_halts:
            self._isolation_failure = "branch_halted_before_candidate"
            return
        if any(branch.steps < required for branch in self.branches):
            return
        for branch in self.branches:
            branch.candidate_sha256 = _tensor_sha256(branch.z)
            branch.candidate_step = branch.steps
        if not self._seed_alias_free:
            self._isolation_failure = "branch_state_alias_detected"
            return
        if not self._configured_role_lesion and not self._seed_states_unique:
            self._isolation_failure = "seed_state_collision"
            return
        if not self._configured_role_lesion and not self._rng_streams_unique:
            self._isolation_failure = "rng_stream_collision"
            return
        if not self._configured_role_lesion and len(
            {branch.candidate_sha256 for branch in self.branches}
        ) != len(self.branches):
            self._isolation_failure = "candidate_state_collision"
            return
        self._isolation_sealed = True

    def isolation_receipt(self, cache_discipline: dict[str, Any]) -> dict[str, Any]:
        """Return the public proof that candidates preceded peer exposure."""

        self._seal_isolation_if_ready()
        cache_proven = (
            isinstance(cache_discipline, dict)
            and cache_discipline.get("all_restored") is True
            and cache_discipline.get("restore_failures") == 0
            and cache_discipline.get("restored_calls")
            == cache_discipline.get("nonpersistent_calls")
        )
        candidates = [
            {
                "index": branch.index,
                "role": branch.role,
                "context_sha256": self._context_sha256,
                "rng_stream_sha256": branch.rng_stream_sha256,
                "seed_sha256": branch.seed_sha256,
                "candidate_sha256": branch.candidate_sha256,
                "candidate_step": branch.candidate_step,
            }
            for branch in self.branches
        ]
        certified = (
            self._isolation_sealed
            and not self._isolation_failure
            and not self._configured_role_lesion
            and self._seed_alias_free
            and self._seed_states_unique
            and self._rng_streams_unique
            and cache_proven
            and all(branch.candidate_sha256 for branch in self.branches)
            and (
                self._first_exchange_step is None
                or self._first_exchange_step >= int(self.config.isolation_steps)
            )
        )
        if certified:
            reason = "certified"
        elif self._isolation_failure:
            reason = self._isolation_failure
        elif self._configured_role_lesion:
            reason = "configured_role_lesion"
        elif not cache_proven:
            reason = "cache_restoration_unproven"
        else:
            reason = "isolation_incomplete"
        return {
            "schema": BRANCH_ISOLATION_SCHEMA,
            "n_branches": len(self.branches),
            "required_steps": int(self.config.isolation_steps),
            "sealed": self._isolation_sealed,
            "certified": certified,
            "reason": reason,
            "configured_role_lesion": self._configured_role_lesion,
            "seed_alias_free": self._seed_alias_free,
            "seed_states_unique": self._seed_states_unique,
            "rng_streams_unique": self._rng_streams_unique,
            "cross_exposure_started": self._cross_exposure_started,
            "first_exchange_step": self._first_exchange_step,
            "blocked_cross_exposures": self._blocked_cross_exposures,
            "candidates": candidates,
            "cache_discipline": dict(cache_discipline),
        }

    # ── Stepping ────────────────────────────────────────────────────────
    def active(self) -> list[BranchState]:
        return [b for b in self.branches if not b.halted]

    def snapshot_branch_runtime(self, branch: BranchState) -> dict[str, Any]:
        """Capture mutable branch state for an exact action-local rollback."""

        if not any(item is branch for item in self.branches):
            raise ValueError("runtime snapshot branch is not in this ensemble")
        return {
            "z": branch.z,
            "role": branch.role,
            "operator": branch.operator.value,
            "halted": branch.halted,
            "halt_reason": branch.halt_reason,
            "steps": branch.steps,
            "score": branch.score,
            "halting": branch.halting.snapshot(),
            "escape": branch.escape.snapshot() if branch.escape is not None else None,
            "kv_boundary_sha256": branch.kv_boundary_sha256,
            "evidence_anchor_sha256": branch.evidence_anchor_sha256,
            "initial_hypothesis_sha256": branch.initial_hypothesis_sha256,
            "candidate_sha256": branch.candidate_sha256,
            "candidate_step": branch.candidate_step,
            "last_loop_delta": branch.last_loop_delta,
            "recurrent_grounding_trace_length": len(branch.recurrent_grounding_trace),
            "loop_stability_trace_length": len(branch.loop_stability_trace),
            "update_acceptance_trace_length": len(branch.update_acceptance_trace),
            "uncertainty_trace_length": len(branch.uncertainty_trace),
            "mistake_locator_trace_length": len(branch.mistake_locator_trace),
            "reflector_trace_length": len(branch.reflector_trace),
        }

    def restore_branch_runtime(
        self,
        branch: BranchState,
        snapshot: Mapping[str, Any],
        *,
        preserve_execution_traces: bool = False,
    ) -> None:
        """Restore one action-local snapshot, including trace append points."""

        if not any(item is branch for item in self.branches):
            raise ValueError("runtime restore branch is not in this ensemble")
        required = {
            "z",
            "role",
            "operator",
            "halted",
            "halt_reason",
            "steps",
            "score",
            "halting",
            "escape",
            "kv_boundary_sha256",
            "evidence_anchor_sha256",
            "initial_hypothesis_sha256",
            "candidate_sha256",
            "candidate_step",
            "last_loop_delta",
            "recurrent_grounding_trace_length",
            "loop_stability_trace_length",
            "update_acceptance_trace_length",
            "uncertainty_trace_length",
            "mistake_locator_trace_length",
            "reflector_trace_length",
        }
        if not isinstance(snapshot, Mapping) or set(snapshot) != required:
            raise ValueError("branch runtime snapshot is invalid")
        if (snapshot["escape"] is None) != (branch.escape is None):
            raise ValueError("branch escape configuration changed during action")
        trace_fields = (
            ("recurrent_grounding_trace", "recurrent_grounding_trace_length"),
            ("loop_stability_trace", "loop_stability_trace_length"),
            ("update_acceptance_trace", "update_acceptance_trace_length"),
            ("uncertainty_trace", "uncertainty_trace_length"),
            ("mistake_locator_trace", "mistake_locator_trace_length"),
            ("reflector_trace", "reflector_trace_length"),
        )
        if type(preserve_execution_traces) is not bool:
            raise TypeError("branch trace preservation flag must be boolean")
        for trace_name, length_name in trace_fields:
            length = snapshot[length_name]
            trace = getattr(branch, trace_name)
            if type(length) is not int or not 0 <= length <= len(trace):
                raise ValueError("branch runtime trace boundary is invalid")
            if not preserve_execution_traces:
                del trace[length:]
        branch.z = snapshot["z"]
        branch.workspace.update(branch.z)
        branch.role = str(snapshot["role"])
        branch.operator = CognitiveOperator(snapshot["operator"])
        branch.halted = bool(snapshot["halted"])
        branch.halt_reason = str(snapshot["halt_reason"])
        branch.steps = int(snapshot["steps"])
        branch.score = float(snapshot["score"])
        branch.halting.restore(snapshot["halting"])
        if branch.escape is not None:
            branch.escape.restore(snapshot["escape"])
        if not preserve_execution_traces:
            branch.evidence_anchor_sha256 = str(snapshot["evidence_anchor_sha256"])
            branch.initial_hypothesis_sha256 = str(snapshot["initial_hypothesis_sha256"])
            branch.candidate_sha256 = str(snapshot["candidate_sha256"])
            branch.candidate_step = int(snapshot["candidate_step"])
        branch.last_loop_delta = snapshot["last_loop_delta"]
        kv_boundary_sha256 = str(snapshot["kv_boundary_sha256"])
        if self._kv_state_tree is not None:
            if not kv_boundary_sha256:
                raise ValueError("branch runtime snapshot has no KV boundary")
            self._kv_state_tree.restore_boundary(
                self._kv_cache,
                kv_boundary_sha256,
            )
        branch.kv_boundary_sha256 = kv_boundary_sha256

    def snapshot_ensemble_runtime(self) -> dict[str, Any]:
        """Capture every non-budget mutation a complete branch round can make."""

        telemetry_state = (
            copy.deepcopy(vars(self.telemetry))
            if self.telemetry is not None and hasattr(self.telemetry, "__dict__")
            else None
        )
        return {
            "branches": {
                branch.index: self.snapshot_branch_runtime(branch) for branch in self.branches
            },
            "exchanges": self.exchanges,
            "exchange_receipt_length": len(self.exchange_receipts),
            "exchange_sync_points": set(self._exchange_sync_points),
            "isolation_sealed": self._isolation_sealed,
            "isolation_failure": self._isolation_failure,
            "blocked_cross_exposures": self._blocked_cross_exposures,
            "cross_exposure_started": self._cross_exposure_started,
            "first_exchange_step": self._first_exchange_step,
            "telemetry_state": telemetry_state,
        }

    def restore_ensemble_runtime(self, snapshot: Mapping[str, Any]) -> None:
        """Restore an entire failed round while retaining its metered compute."""

        required = {
            "branches",
            "exchanges",
            "exchange_receipt_length",
            "exchange_sync_points",
            "isolation_sealed",
            "isolation_failure",
            "blocked_cross_exposures",
            "cross_exposure_started",
            "first_exchange_step",
            "telemetry_state",
        }
        if not isinstance(snapshot, Mapping) or set(snapshot) != required:
            raise ValueError("ensemble runtime snapshot is invalid")
        branch_snapshots = snapshot["branches"]
        if not isinstance(branch_snapshots, Mapping) or set(branch_snapshots) != {
            branch.index for branch in self.branches
        }:
            raise ValueError("ensemble branch snapshot inventory differs")
        receipt_length = snapshot["exchange_receipt_length"]
        if type(receipt_length) is not int or not 0 <= receipt_length <= len(
            self.exchange_receipts
        ):
            raise ValueError("ensemble exchange receipt boundary is invalid")
        for branch in self.branches:
            self.restore_branch_runtime(
                branch,
                branch_snapshots[branch.index],
            )
        self.exchanges = int(snapshot["exchanges"])
        del self.exchange_receipts[receipt_length:]
        self._exchange_sync_points = set(snapshot["exchange_sync_points"])
        self._isolation_sealed = bool(snapshot["isolation_sealed"])
        self._isolation_failure = str(snapshot["isolation_failure"])
        self._blocked_cross_exposures = int(snapshot["blocked_cross_exposures"])
        self._cross_exposure_started = bool(snapshot["cross_exposure_started"])
        self._first_exchange_step = snapshot["first_exchange_step"]
        telemetry_state = snapshot["telemetry_state"]
        if telemetry_state is not None:
            if self.telemetry is None or not hasattr(self.telemetry, "__dict__"):
                raise ValueError("ensemble telemetry disappeared during transaction")
            vars(self.telemetry).clear()
            vars(self.telemetry).update(copy.deepcopy(telemetry_state))

    def step_all(
        self,
        runner: WindowRunner,
        cache,
        start: int,
        end: int,
        *,
        budget: ComputeBudget,
        alpha_override: float | None = None,
        score_fn: Callable[[BranchState], float] | None = None,
        reserve_layer_apps: int = 0,
        stop_context: Any = None,
        transaction_purpose: str = "recurrent_update",
    ) -> bool:
        """Advance a complete branch round transactionally."""

        runtime_snapshot = self.snapshot_ensemble_runtime()
        try:
            return self._step_all_untransactional(
                runner,
                cache,
                start,
                end,
                budget=budget,
                alpha_override=alpha_override,
                score_fn=score_fn,
                reserve_layer_apps=reserve_layer_apps,
                stop_context=stop_context,
                transaction_purpose=transaction_purpose,
            )
        except Exception:
            self.restore_ensemble_runtime(runtime_snapshot)
            raise

    def _step_all_untransactional(
        self,
        runner: WindowRunner,
        cache,
        start: int,
        end: int,
        *,
        budget: ComputeBudget,
        alpha_override: float | None = None,
        score_fn: Callable[[BranchState], float] | None = None,
        reserve_layer_apps: int = 0,
        stop_context: Any = None,
        transaction_purpose: str = "recurrent_update",
    ) -> bool:
        """Advance every live branch, or none when the whole round cannot fit."""
        active = self.active()
        round_cost = sum(int(branch.z.shape[1]) * (end - start) for branch in active)
        if round_cost + reserve_layer_apps > budget.remaining_layer_apps:
            return False
        deferred_fixed_depth_halts: list[tuple[BranchState, str]] = []
        for branch in active:
            evidence_slots = branch.workspace.context_slot_indices
            hypothesis_slots = branch.workspace.hypothesis_slot_indices(
                comm_slot=int(self.config.comm_slot)
            )
            if not hypothesis_slots:
                raise RuntimeError("recurrent workspace has no persistent hypothesis slot")
            evidence_pre = _tensor_sha256(branch.workspace.select_slots(branch.z, evidence_slots))
            hypothesis_pre = _tensor_sha256(
                branch.workspace.select_slots(branch.z, hypothesis_slots)
            )
            if not branch.evidence_anchor_sha256:
                branch.evidence_anchor_sha256 = evidence_pre
            if not branch.initial_hypothesis_sha256:
                branch.initial_hypothesis_sha256 = hypothesis_pre
            reasoning_slots = (int(self.config.comm_slot), *hypothesis_slots)
            reasoning_pre_state = branch.workspace.select_slots(
                branch.z,
                reasoning_slots,
            )
            reasoning_pre_sha256 = _tensor_sha256(reasoning_pre_state)
            reasoning_anchor_state = branch.workspace.select_slots(
                branch.anchor,
                reasoning_slots,
            )
            anchor_sha256 = _tensor_sha256(reasoning_anchor_state)
            continuous = bool(
                branch.loop_stability_trace
                and branch.loop_stability_trace[-1]["reasoning_post_sha256"] == reasoning_pre_sha256
            )
            effective_alpha = (
                alpha_override
                if alpha_override is not None
                else alpha_at(self.recurrence, branch.steps)
            )
            with runner.transaction_context(
                purpose=transaction_purpose,
                branch_index=branch.index,
                parent_sha256=(
                    branch.kv_boundary_sha256
                    or (self._kv_state_tree.root_sha256 if self._kv_state_tree is not None else "")
                ),
            ):
                proposal = recurrence_step(
                    branch.z,
                    runner,
                    cache,
                    start,
                    end,
                    self.recurrence,
                    branch.steps,
                    anchor=branch.anchor,
                    alpha_override=alpha_override,
                    branch_index=branch.index,
                )
            proposal = branch.workspace.restore_context_evidence(proposal)
            proposal_evidence_post = _tensor_sha256(
                branch.workspace.select_slots(proposal, evidence_slots)
            )
            if proposal_evidence_post != evidence_pre:
                raise RuntimeError("sealed recurrent evidence changed in an update proposal")
            proposal_hypothesis_post = _tensor_sha256(
                branch.workspace.select_slots(proposal, hypothesis_slots)
            )
            proposal_reasoning_state = branch.workspace.select_slots(
                proposal,
                reasoning_slots,
            )
            proposal_reasoning_sha256 = _tensor_sha256(proposal_reasoning_state)
            evidence_state = (
                branch.workspace.select_slots(branch.anchor, evidence_slots)
                if evidence_slots
                else None
            )
            gate = branch.update_gate or UpdateGateRuntime(mode=PASSTHROUGH)
            gate_decision = gate.evaluate(
                reasoning_pre_state,
                proposal_reasoning_state,
                reasoning_anchor_state,
                evidence_state=evidence_state,
                previous_residual=(
                    float(branch.loop_stability_trace[-1]["residual"]) if continuous else None
                ),
                previous_delta=branch.last_loop_delta if continuous else None,
            )
            z_next = proposal if gate_decision.accepted else branch.z
            z_next = branch.workspace.restore_context_evidence(z_next)
            evidence_post = _tensor_sha256(branch.workspace.select_slots(z_next, evidence_slots))
            hypothesis_post = _tensor_sha256(
                branch.workspace.select_slots(z_next, hypothesis_slots)
            )
            evidence_unchanged = evidence_pre == evidence_post == branch.evidence_anchor_sha256
            if not evidence_unchanged:
                raise RuntimeError("sealed recurrent evidence changed during a window pass")
            reasoning_post_state = branch.workspace.select_slots(
                z_next,
                reasoning_slots,
            )
            reasoning_post_sha256 = _tensor_sha256(reasoning_post_state)
            stability, branch.last_loop_delta = transition_metrics(
                reasoning_pre_state,
                proposal_reasoning_state,
                reasoning_anchor_state,
                alpha=effective_alpha,
                convergence_eps=self.recurrence.convergence_eps,
                previous_residual=(
                    float(branch.loop_stability_trace[-1]["residual"]) if continuous else None
                ),
                previous_delta=branch.last_loop_delta if continuous else None,
            )
            branch.loop_stability_trace.append(
                {
                    "ordinal": len(branch.loop_stability_trace),
                    "branch_step": branch.steps,
                    "window_start": start,
                    "window_end": end,
                    "hypothesis_pre_sha256": hypothesis_pre,
                    "hypothesis_post_sha256": proposal_hypothesis_post,
                    "reasoning_pre_sha256": reasoning_pre_sha256,
                    "reasoning_post_sha256": proposal_reasoning_sha256,
                    "anchor_sha256": anchor_sha256,
                    "continuous_from_previous": continuous,
                    "disposition": ("accepted" if gate_decision.accepted else "quality_rejected"),
                    "divergence_reason": "",
                    "containment_action": ("" if gate_decision.accepted else "retain_previous"),
                    **stability,
                }
            )
            branch.update_acceptance_trace.append(
                {
                    "ordinal": len(branch.update_acceptance_trace),
                    "branch_step": branch.steps,
                    "prior_hypothesis_sha256": hypothesis_pre,
                    "proposal_hypothesis_sha256": proposal_hypothesis_post,
                    "admitted_hypothesis_sha256": hypothesis_post,
                    "prior_reasoning_sha256": reasoning_pre_sha256,
                    "proposal_reasoning_sha256": proposal_reasoning_sha256,
                    "admitted_reasoning_sha256": reasoning_post_sha256,
                    "probability": gate_decision.probability,
                    "threshold": gate_decision.threshold,
                    "accepted": gate_decision.accepted,
                    "reason": gate_decision.reason,
                    "features": dict(gate_decision.features),
                    "features_sha256": gate_decision.features_sha256,
                }
            )
            from core.brain.llm.latent_cortex.bidirectional_reflector import (
                observe_reflector_transition,
            )

            branch.reflector_trace.append(
                observe_reflector_transition(
                    reasoning_pre_state,
                    proposal_reasoning_state,
                    reasoning_post_state,
                    branch_index=branch.index,
                    branch_step=branch.steps,
                    prior_state_sha256=reasoning_pre_sha256,
                    proposal_state_sha256=proposal_reasoning_sha256,
                    admitted_state_sha256=reasoning_post_sha256,
                    accepted=gate_decision.accepted,
                )
            )
            state_width = int(reasoning_post_state.shape[-1])
            sketch_width = 2 * min(64, state_width)
            position_count = int(reasoning_post_state.shape[-2])
            position_sketch_width = 2 * min(8, state_width)
            budget.charge_tensor_work(
                "bidirectional_reflector_capture",
                element_reads=(
                    int(reasoning_pre_state.size)
                    + int(proposal_reasoning_state.size)
                    + int(reasoning_post_state.size)
                ),
                host_scalar_ops=(
                    int(reasoning_pre_state.size)
                    + int(proposal_reasoning_state.size)
                    + int(reasoning_post_state.size)
                    + 6 * (sketch_width + position_count * position_sketch_width)
                ),
            )
            if (
                branch.uncertainty_runtime is not None
                and branch.uncertainty_runtime.mode == "learned"
            ):
                branch.uncertainty_trace.append(
                    branch.uncertainty_runtime.observe(
                        reasoning_post_state,
                        branch_index=branch.index,
                        branch_step=branch.steps,
                        state_sha256=reasoning_post_sha256,
                    )
                )
                head = branch.uncertainty_runtime.head
                hidden_width = 0 if head is None else int(head.input_weights.shape[1])
                budget.charge_tensor_work(
                    "neural_uncertainty_head",
                    element_reads=int(reasoning_post_state.size),
                    host_scalar_ops=(
                        int(reasoning_post_state.size)
                        + int(reasoning_post_state.shape[-1]) * hidden_width
                        + hidden_width
                    ),
                )
            if (
                branch.mistake_locator_runtime is not None
                and branch.mistake_locator_runtime.mode == "learned"
            ):
                branch.mistake_locator_trace.append(
                    branch.mistake_locator_runtime.observe(
                        reasoning_pre_state,
                        proposal_reasoning_state,
                        branch_index=branch.index,
                        branch_step=branch.steps,
                        prior_state_sha256=reasoning_pre_sha256,
                        proposal_state_sha256=proposal_reasoning_sha256,
                        admitted_state_sha256=reasoning_post_sha256,
                        accepted=gate_decision.accepted,
                    )
                )
                locator_head = branch.mistake_locator_runtime.head
                locator_hidden = (
                    0 if locator_head is None else int(locator_head.input_weights.shape[1])
                )
                locator_features = int(reasoning_post_state.shape[-1]) * 4
                budget.charge_tensor_work(
                    "mistake_locator_head",
                    element_reads=(
                        int(reasoning_pre_state.size) + int(proposal_reasoning_state.size)
                    ),
                    host_scalar_ops=(
                        int(reasoning_pre_state.size)
                        + int(proposal_reasoning_state.size)
                        + locator_features * locator_hidden
                        + locator_hidden
                    ),
                )
            branch.recurrent_grounding_trace.append(
                {
                    "ordinal": len(branch.recurrent_grounding_trace),
                    "branch_step": branch.steps,
                    "window_start": start,
                    "window_end": end,
                    "evidence_pre_sha256": evidence_pre,
                    "evidence_post_sha256": evidence_post,
                    "hypothesis_pre_sha256": hypothesis_pre,
                    "hypothesis_post_sha256": hypothesis_post,
                    "evidence_unchanged": True,
                    "hypothesis_changed": hypothesis_pre != hypothesis_post,
                }
            )
            hidden = int(branch.z.shape[-1])
            committed_slots = len(evidence_slots) + len(hypothesis_slots)
            budget.charge_tensor_work(
                "recurrent_grounding_commitment",
                element_reads=2 * committed_slots * hidden,
                host_scalar_ops=2 * committed_slots * hidden,
            )
            budget.charge_tensor_work(
                "recurrent_update_acceptance",
                element_reads=(4 * len(reasoning_slots) + len(evidence_slots)) * hidden,
                host_scalar_ops=64,
            )
            residual = relative_residual(
                proposal_reasoning_state,
                reasoning_pre_state,
            )
            score = (
                self._score_candidate(branch, z_next, score_fn) if score_fn is not None else None
            )
            decision = branch.halting.observe(
                branch.steps,
                z_next,
                residual,
                score=score,
                budget=budget,
                stop_context=stop_context,
                update_decision=gate_decision,
            )
            if branch.halting.stop_gate is not None:
                budget.charge_tensor_work(
                    "recurrent_stop_gate",
                    host_scalar_ops=96,
                )
            branch.z = z_next
            branch.workspace.update(z_next)
            branch.steps += 1
            if self.telemetry is not None:
                self.telemetry.record_step(branch.index, branch.z, branch.anchor, residual)
            if decision.should_halt:
                if self.recurrence.fixed_depth and decision.reason == "max_steps":
                    deferred_fixed_depth_halts.append((branch, decision.reason))
                    continue
                # Divergence gets a second life through the escape ladder;
                # legitimate halts (converged / max_steps / budget) do not.
                if decision.reason.startswith("diverged"):
                    transition = branch.loop_stability_trace[-1]
                    transition["disposition"] = "contained_divergence"
                    transition["divergence_reason"] = decision.reason
                if branch.escape is not None and decision.reason.startswith("diverged"):
                    action = branch.escape.on_divergence(branch, decision.reason)
                    transition["containment_action"] = action
                    if action == "escaped":
                        continue
                    self._halt(
                        branch,
                        action.removeprefix("halt:"),
                        budget=budget,
                    )
                    continue
                if decision.reason.startswith("diverged"):
                    transition["containment_action"] = "halt_revert"
                self._halt(branch, decision.reason, budget=budget)
                continue
            if branch.escape is not None:
                action = branch.escape.on_step(branch)
                if action.startswith("halt:"):
                    self._halt(
                        branch,
                        action.removeprefix("halt:"),
                        budget=budget,
                    )

        self._seal_isolation_if_ready()

        active_branches = self.active()
        active_steps = {branch.steps for branch in active_branches}
        completed_interval_steps = {
            int(row["sync_id"].removeprefix("recurrent-step:"))
            for row in self.exchange_receipts
            if row.get("sync_kind") == "interval"
            and isinstance(row.get("sync_id"), str)
            and row["sync_id"].removeprefix("recurrent-step:").isdigit()
        }
        if (
            len(active_branches) > 1
            and len(active_steps) == 1
            and next(iter(active_steps)) % self.config.exchange_interval == 0
            and next(iter(active_steps)) not in completed_interval_steps
        ):
            if self.exchange(
                sync_kind="interval",
                sync_id=f"recurrent-step:{next(iter(active_steps))}",
                budget=budget,
            ):
                self.maintain_diversity(budget=budget)
        for branch, reason in deferred_fixed_depth_halts:
            self._halt(branch, reason, budget=budget)
        return True

    @staticmethod
    def _score_candidate(
        branch: BranchState,
        z_next: Any,
        score_fn: Callable[[BranchState], float],
    ) -> float:
        """Evaluate a candidate through the existing branch-scoring contract.

        The callback historically receives ``BranchState``. Project the
        candidate into that view only for the call, then restore the committed
        state even if the verifier raises. This prevents a score for ``z_t``
        from being attached to ``z_(t+1)`` without changing public callers.
        """

        prior_z = branch.z
        prior_steps = branch.steps
        branch.z = z_next
        branch.workspace.update(z_next)
        branch.steps = prior_steps + 1
        try:
            return float(score_fn(branch))
        finally:
            branch.z = prior_z
            branch.workspace.update(prior_z)
            branch.steps = prior_steps

    # ── Neural-bytecode instructions ────────────────────────────────────
    def exchange_now(
        self,
        *,
        sync_kind: str,
        sync_id: str,
        budget: ComputeBudget | None = None,
    ) -> bool:
        """Bytecode-forced exchange: communicate immediately when ≥2 live."""
        if len(self.active()) < 2:
            return False
        if not self.exchange(sync_kind=sync_kind, sync_id=sync_id, budget=budget):
            return False
        self.maintain_diversity(budget=budget)
        return True

    def savepoint_all(
        self,
        *,
        verified: bool = False,
        authority: str = "schedule_program",
    ) -> int:
        """Snapshot every live branch's complete mutable execution state."""
        saved = 0
        for branch in self.active():
            kv_boundary_sha256 = branch.kv_boundary_sha256
            if self._kv_state_tree is not None:
                kv_boundary_sha256 = self._kv_state_tree.capture_boundary(
                    self._kv_cache,
                    parent_sha256=(branch.kv_boundary_sha256 or self._kv_state_tree.root_sha256),
                    branch_index=branch.index,
                    label=("verified_branch_savepoint" if verified else "branch_savepoint"),
                    authority=authority,
                    verified=verified,
                    latent_sha256=tensor_sha256(branch.z),
                )
                branch.kv_boundary_sha256 = kv_boundary_sha256
            branch.savepoint = {
                "z": branch.z,
                "role": branch.role,
                "operator": branch.operator.value,
                "halted": branch.halted,
                "halt_reason": branch.halt_reason,
                "steps": branch.steps,
                "score": branch.score,
                "halting": branch.halting.snapshot(),
                "escape": branch.escape.snapshot() if branch.escape is not None else None,
                "kv_boundary_sha256": kv_boundary_sha256,
            }
            branch.savepoint_steps = branch.steps
            branch.savepoint_kv_boundary_sha256 = kv_boundary_sha256
            saved += 1
        return saved

    def savepoint_branch(
        self,
        branch: BranchState,
        *,
        verified: bool = True,
        authority: str = "verifier",
    ) -> bool:
        """Snapshot one branch; evidence from one probe grants no peer authority."""

        if not any(item is branch for item in self.branches) or branch.halted:
            return False
        kv_boundary_sha256 = branch.kv_boundary_sha256
        if self._kv_state_tree is not None:
            kv_boundary_sha256 = self._kv_state_tree.capture_boundary(
                self._kv_cache,
                parent_sha256=(branch.kv_boundary_sha256 or self._kv_state_tree.root_sha256),
                branch_index=branch.index,
                label=("verified_branch_savepoint" if verified else "branch_savepoint"),
                authority=authority,
                verified=verified,
                latent_sha256=tensor_sha256(branch.z),
            )
            branch.kv_boundary_sha256 = kv_boundary_sha256
        branch.savepoint = {
            "z": branch.z,
            "role": branch.role,
            "operator": branch.operator.value,
            "halted": branch.halted,
            "halt_reason": branch.halt_reason,
            "steps": branch.steps,
            "score": branch.score,
            "halting": branch.halting.snapshot(),
            "escape": branch.escape.snapshot() if branch.escape is not None else None,
            "kv_boundary_sha256": kv_boundary_sha256,
        }
        branch.savepoint_steps = branch.steps
        branch.savepoint_kv_boundary_sha256 = kv_boundary_sha256
        return True

    def revert_branch_to_savepoint(self, branch: BranchState) -> bool:
        """Transactionally restore one branch to its most recent savepoint."""

        snapshot = branch.savepoint
        if not isinstance(snapshot, dict):
            return False
        required = {
            "z",
            "role",
            "operator",
            "halted",
            "halt_reason",
            "steps",
            "score",
            "halting",
            "escape",
            "kv_boundary_sha256",
        }
        if set(snapshot) != required:
            raise ValueError("invalid branch savepoint")
        if (snapshot["escape"] is None) != (branch.escape is None):
            raise ValueError("branch escape configuration changed after savepoint")
        branch.z = snapshot["z"]
        branch.workspace.update(branch.z)
        branch.role = str(snapshot["role"])
        branch.operator = CognitiveOperator(snapshot["operator"])
        branch.halted = bool(snapshot["halted"])
        branch.halt_reason = str(snapshot["halt_reason"])
        branch.steps = int(snapshot["steps"])
        branch.score = float(snapshot["score"])
        branch.halting.restore(snapshot["halting"])
        if branch.escape is not None:
            branch.escape.restore(snapshot["escape"])
        kv_boundary_sha256 = str(snapshot["kv_boundary_sha256"])
        if self._kv_state_tree is not None:
            if not kv_boundary_sha256:
                raise ValueError("branch savepoint has no KV boundary")
            self._kv_state_tree.restore_boundary(
                self._kv_cache,
                kv_boundary_sha256,
            )
        branch.kv_boundary_sha256 = kv_boundary_sha256
        branch.savepoint_kv_boundary_sha256 = kv_boundary_sha256
        return True

    def revert_all_to_savepoint(self) -> int:
        """Backtrack every branch that holds a savepoint transactionally."""
        reverted = 0
        for branch in self.branches:
            if self.revert_branch_to_savepoint(branch):
                reverted += 1
        return reverted

    def observe_verified_best(
        self,
        branch: BranchState,
        raw_observation: Any,
        *,
        action_step: int,
        restore_target_state_sha256: str = "",
        budget: ComputeBudget | None = None,
    ) -> tuple[VerifierObservation, str, bool]:
        """Promote or preserve a branch-local state under interval dominance."""

        if not any(item is branch for item in self.branches):
            raise ValueError("verified-best branch is not in this ensemble")
        if type(action_step) is not int or action_step < 0:
            raise ValueError("verified-best action step is invalid")
        observation = VerifierObservation.from_value(raw_observation)
        observation_receipt = observation.to_dict()
        if budget is not None:
            elements = int(branch.z.size)
            budget.charge_tensor_work(
                "verified_best_state",
                element_reads=elements,
                host_scalar_ops=elements + 64,
            )
        candidate_sha256 = tensor_sha256(branch.z)
        prior_sha256 = branch.verified_best_state_sha256
        restored = False
        if observation.authoritative and observation.upper_bound <= 1e-9:
            if not _is_sha256(restore_target_state_sha256):
                raise ValueError("verified failure has no committed restore target")
            decision = "reject_verified_failure"
        elif not observation.authoritative:
            decision = "ranking_only"
        elif branch.verified_best_state is None:
            decision = "promote"
        else:
            incumbent = VerifierObservation.from_value(
                {
                    key: branch.verified_best_observation[key]
                    for key in (
                        "schema",
                        "score",
                        "lower_bound",
                        "upper_bound",
                        "sample_count",
                        "basis",
                        "independent",
                        "evidence_sha256",
                    )
                }
            )
            if observation.lower_bound > incumbent.upper_bound + 1e-9:
                decision = "promote"
            else:
                decision = "preserve_verified"
                if candidate_sha256 != prior_sha256:
                    branch.z = branch.verified_best_state
                    branch.workspace.update(branch.z)
                    restored = True
        if decision == "promote":
            branch.verified_best_state = branch.z
            branch.verified_best_step = branch.steps
            branch.verified_best_state_sha256 = candidate_sha256
            branch.verified_best_observation = observation_receipt
        resulting_sha256 = (
            branch.verified_best_state_sha256
            if decision == "preserve_verified"
            else candidate_sha256
        )
        branch.verified_best_trace.append(
            {
                "ordinal": len(branch.verified_best_trace),
                "action_step": action_step,
                "branch_step": branch.steps,
                "candidate_state_sha256": candidate_sha256,
                "prior_best_state_sha256": prior_sha256,
                "restore_target_state_sha256": (
                    restore_target_state_sha256 if decision == "reject_verified_failure" else ""
                ),
                "observation": observation_receipt,
                "decision": decision,
                "restored": restored,
                "resulting_state_sha256": resulting_sha256,
            }
        )
        return observation, decision, restored

    def commit_verified_failure_restore(
        self,
        branch: BranchState,
        *,
        action_step: int,
    ) -> None:
        """Bind an engine-level exact-parent restore to its verifier decision."""

        if not any(item is branch for item in self.branches):
            raise ValueError("verified failure branch is not in this ensemble")
        if not branch.verified_best_trace:
            raise ValueError("verified failure restore has no source decision")
        row = branch.verified_best_trace[-1]
        if (
            row.get("action_step") != action_step
            or row.get("decision") != "reject_verified_failure"
            or row.get("restored") is not False
        ):
            raise ValueError("verified failure restore source differs")
        restored_sha256 = tensor_sha256(branch.z)
        if restored_sha256 != row.get("restore_target_state_sha256"):
            raise RuntimeError("verified failure did not restore its committed parent")
        row["restored"] = True
        row["resulting_state_sha256"] = restored_sha256

    def final_state(
        self,
        branch: BranchState,
        *,
        budget: ComputeBudget | None = None,
    ) -> tuple[Any, bool, str]:
        """Prefer confidence-bound verified state, then the legacy proxy peak."""

        if budget is not None:
            elements = int(branch.z.size)
            budget.charge_tensor_work(
                "verified_best_finalization",
                element_reads=2 * elements,
                host_scalar_ops=2 * elements + 64,
            )
        pre_sha256 = tensor_sha256(branch.z)
        if not self.recurrence.fixed_depth and branch.verified_best_state is not None:
            result = (
                branch.verified_best_state,
                pre_sha256 != branch.verified_best_state_sha256,
                "verified",
            )
        else:
            final, reverted = branch.halting.final_state(branch.z)
            result = final, reverted, "proxy" if reverted else "current"
        final, reverted, source = result
        branch.verified_finalization = {
            "source": source,
            "pre_state_sha256": pre_sha256,
            "post_state_sha256": tensor_sha256(final),
            "reverted": reverted,
            "fixed_depth": self.recurrence.fixed_depth,
        }
        return result

    def inject_control(self, control, *, strength: float = 0.12) -> int:
        """Causally write one bounded operator vector into each live workspace."""

        import mlx.core as mx

        if (
            isinstance(strength, bool)
            or not isinstance(strength, (int, float))
            or not 0.0 < float(strength) <= 0.5
        ):
            raise ValueError("control strength must be inside (0, 0.5]")
        changed = 0
        for branch in self.active():
            z = branch.z
            vector = mx.reshape(control, (1, 1, int(z.shape[-1])))
            slot = min(int(self.config.comm_slot), int(z.shape[1]) - 1)
            prior = z[:, slot : slot + 1, :]
            blended = (1.0 - float(strength)) * prior + float(strength) * vector
            blended = rms_match(blended, prior, self.recurrence.rms_clip_ratio)
            branch.z = mx.concatenate(
                [z[:, :slot, :], blended, z[:, slot + 1 :, :]],
                axis=1,
            )
            branch.workspace.update(branch.z)
            changed += 1
        if changed:
            mx.eval(*[branch.z for branch in self.active()])
        return changed

    def apply_cognitive_operators(
        self,
        control,
        *,
        action: str,
        action_step: int,
        budget: ComputeBudget | None = None,
    ) -> list[dict[str, Any]]:
        """Run each live branch's distinct executable strategy privately."""

        receipts: list[dict[str, Any]] = []
        for branch in self.active():
            protected = tuple(
                int(item["slot"])
                for item in branch.workspace.context_slots
                if isinstance(item, dict) and type(item.get("slot")) is int
            )
            output, receipt = execute_cognitive_operator(
                branch.z,
                branch.anchor,
                control,
                operator=branch.operator,
                role=branch.role,
                branch_index=branch.index,
                action=action,
                action_step=action_step,
                protected_slots=protected,
                comm_slot=int(self.config.comm_slot),
                rms_clip_ratio=float(self.recurrence.rms_clip_ratio),
            )
            branch.z = output
            branch.workspace.update(output)
            receipts.append(receipt)
            if budget is not None:
                accounting = receipt["tensor_accounting"]
                budget.charge_tensor_work(
                    f"cognitive_operator:{branch.operator.value}",
                    element_reads=accounting["element_reads"],
                    element_writes=accounting["element_writes"],
                    scalar_ops=accounting["tensor_scalar_ops"],
                    host_scalar_ops=accounting["commitment_host_ops"],
                )
        return receipts

    def apply_context_focus(
        self,
        *,
        action: OperationKind,
        action_step: int,
        budget: ComputeBudget | None = None,
    ) -> list[dict[str, Any]]:
        """Focus live branches on the action's admitted source class."""

        receipts: list[dict[str, Any]] = []
        for branch in self.active():
            output, receipt = apply_context_focus(
                branch.z,
                context_slots=branch.workspace.context_slots,
                action=action,
                branch_index=branch.index,
                action_step=action_step,
                comm_slot=int(self.config.comm_slot),
                rms_clip_ratio=float(self.recurrence.rms_clip_ratio),
            )
            branch.z = output
            branch.workspace.update(output)
            receipts.append(receipt)
            if budget is not None:
                accounting = receipt["tensor_accounting"]
                budget.charge_tensor_work(
                    f"context_focus:{action.value}",
                    element_reads=accounting["element_reads"],
                    element_writes=accounting["element_writes"],
                    scalar_ops=accounting["tensor_scalar_ops"],
                    host_scalar_ops=accounting["commitment_host_ops"],
                )
        return receipts

    def compress_state(
        self,
        *,
        strength: float = 0.25,
        budget: ComputeBudget | None = None,
    ) -> int:
        """Fold global branch summaries into comm slots without erasing detail."""

        import mlx.core as mx

        if (
            isinstance(strength, bool)
            or not isinstance(strength, (int, float))
            or not 0.0 < float(strength) <= 0.5
        ):
            raise ValueError("compression strength must be inside (0, 0.5]")
        live = self.active()
        if not live:
            return 0
        if len(live) > 1 and not self._isolation_sealed:
            self._blocked_cross_exposures += 1
            return 0
        summaries = [branch.workspace.summary() for branch in live]
        global_summary = sum(summaries) / len(summaries)
        for branch in live:
            z = branch.z
            slot = min(int(self.config.comm_slot), int(z.shape[1]) - 1)
            prior = z[:, slot : slot + 1, :]
            compressed = (1.0 - float(strength)) * prior + float(strength) * global_summary
            compressed = rms_match(
                compressed,
                prior,
                self.recurrence.rms_clip_ratio,
            )
            branch.z = mx.concatenate(
                [z[:, :slot, :], compressed, z[:, slot + 1 :, :]],
                axis=1,
            )
            branch.workspace.update(branch.z)
        mx.eval(*[branch.z for branch in live])
        if budget is not None:
            slots = int(live[0].z.shape[1])
            hidden = int(live[0].z.shape[-1])
            budget.charge_tensor_work(
                "branch_state_compression",
                element_reads=len(live) * slots * hidden,
                element_writes=len(live) * hidden,
                scalar_ops=len(live) * hidden * (slots + 6),
            )
        return len(live)

    def disagreement(self, *, budget: ComputeBudget | None = None) -> float:
        """Mean pairwise cosine distance between active branch summaries."""

        import mlx.core as mx

        live = self.active()
        if len(live) < 2:
            return 0.0
        distances: list[float] = []
        pair_count = len(live) * (len(live) - 1) // 2
        if budget is not None:
            hidden = int(live[0].z.shape[-1])
            slots = int(live[0].z.shape[1])
            budget.charge_tensor_work(
                "branch_disagreement",
                element_reads=2 * pair_count * slots * hidden,
                scalar_ops=pair_count * ((2 * slots + 6) * hidden + 4),
                host_scalar_ops=pair_count * 4,
            )
        for left_index, left in enumerate(live):
            left_summary = left.workspace.summary()
            for right in live[left_index + 1 :]:
                right_summary = right.workspace.summary()
                cosine = float(
                    mx.sum(left_summary * right_summary)
                    / mx.maximum(
                        mx.linalg.norm(left_summary) * mx.linalg.norm(right_summary),
                        1e-6,
                    )
                )
                distances.append(max(0.0, min(1.0, 0.5 * (1.0 - cosine))))
        return sum(distances) / max(1, len(distances))

    def halt_all(
        self,
        reason: str,
        *,
        budget: ComputeBudget | None = None,
    ) -> int:
        """Stop every live branch through the same best-state finalizer."""

        live = list(self.active())
        for branch in live:
            self._halt(branch, reason, budget=budget)
        return len(live)

    def _halt(
        self,
        branch: BranchState,
        reason: str,
        *,
        budget: ComputeBudget | None = None,
    ) -> None:
        """Halt one branch, shipping the best state when it beats the last."""
        final, reverted, _source = self.final_state(branch, budget=budget)
        branch.z = final
        branch.workspace.update(final)
        branch.halted = True
        branch.halt_reason = reason + ("_reverted" if reverted else "")
        if branch.escape is not None:
            branch.escape.finalize()

    # ── Communication ───────────────────────────────────────────────────
    def exchange(
        self,
        *,
        sync_kind: str,
        sync_id: str,
        budget: ComputeBudget | None = None,
    ) -> bool:
        """Blend the agreement-weighted consensus into each comm slot.

        Weights favor branches whose summaries agree with the ensemble mean —
        an outlier branch still RECEIVES the consensus but contributes little
        to it, which lets adversarial/counterexample roles stay adversarial
        without dragging the consensus around.
        """
        import mlx.core as mx

        live = self.active()
        if len(live) < 2:
            return False
        if not self._isolation_sealed:
            self._blocked_cross_exposures += 1
            return False
        gamma = float(self.config.exchange_gamma)
        if not 0.0 < gamma <= 1.0:
            return False
        if sync_kind not in {
            "interval",
            "schedule_bytecode",
            "controller_compare",
            "test",
        }:
            raise ValueError("exchange requires a declared synchronization kind")
        if not isinstance(sync_id, str) or not 1 <= len(sync_id) <= 160:
            raise ValueError("exchange requires a bounded synchronization identity")
        sync_point = f"{sync_kind}:{sync_id}"
        if sync_point in self._exchange_sync_points:
            raise ValueError("exchange synchronization point was already consumed")

        context_slots = sorted(
            {
                int(item["slot"])
                for item in live[0].workspace.context_slots
                if isinstance(item, dict) and type(item.get("slot")) is int
            }
        )
        if any(
            sorted(
                {
                    int(item["slot"])
                    for item in branch.workspace.context_slots
                    if isinstance(item, dict) and type(item.get("slot")) is int
                }
            )
            != context_slots
            for branch in live[1:]
        ):
            raise RuntimeError("branch context-slot topology differs")
        slot = int(self.config.comm_slot)
        n_slots = int(live[0].z.shape[1])
        source_slots = private_exchange_slots(
            n_slots=n_slots,
            comm_slot=slot,
            context_slots=context_slots,
        )
        summaries = [
            mx.mean(
                mx.concatenate(
                    [branch.z[:, index : index + 1, :] for index in source_slots],
                    axis=1,
                ),
                axis=1,
                keepdims=True,
            )
            for branch in live
        ]
        if self.telemetry is not None:
            self.telemetry.record_exchange(summaries)
        stack = mx.concatenate(summaries, axis=1)  # (1,K,D)
        mean = mx.mean(stack, axis=1, keepdims=True)  # (1,1,D)

        def _cos(a, b):
            num = mx.sum(a * b)
            den = mx.maximum(mx.linalg.norm(a) * mx.linalg.norm(b), 1e-6)
            return num / den

        agreements = mx.stack([_cos(s, mean) for s in summaries])  # (K,)
        weights = mx.softmax(agreements, axis=0)
        support = mx.array([self._support_weights[branch.index] for branch in live])
        weights = weights * support
        weights = weights / mx.maximum(mx.sum(weights), 1e-6)
        consensus = sum(w * s for w, s in zip(weights, summaries, strict=True))
        mx.eval(weights, consensus, *summaries)

        candidate_rows = [
            {
                "index": branch.index,
                "role": branch.role,
                "candidate_sha256": branch.candidate_sha256,
                "candidate_step": branch.candidate_step,
            }
            for branch in live
        ]
        isolation_binding = {
            "candidates": candidate_rows,
            "configured_role_lesion": self._configured_role_lesion,
        }
        excluded_slots = sorted(set(context_slots + [slot]))
        source_rows = []
        for position, (branch, summary) in enumerate(zip(live, summaries, strict=True)):
            private_state = mx.concatenate(
                [branch.z[:, index : index + 1, :] for index in source_slots],
                axis=1,
            )
            mx.eval(private_state)
            source_rows.append(
                {
                    "branch_index": branch.index,
                    "role": branch.role,
                    "operator": branch.operator.value,
                    "step": branch.steps,
                    "candidate_sha256": branch.candidate_sha256,
                    "candidate_step": branch.candidate_step,
                    "source_slots": list(source_slots),
                    "excluded_slots": excluded_slots,
                    "state_sha256": _tensor_sha256(branch.z),
                    "private_state_sha256": _tensor_sha256(private_state),
                    "message_sha256": _tensor_sha256(summary),
                    "support_weight": round(float(self._support_weights[branch.index]), 12),
                    "consensus_weight": round(float(weights[position]), 12),
                }
            )

        prior_states = {branch.index: branch.z for branch in live}
        for branch in live:
            z = branch.z
            comm = (1.0 - gamma) * z[:, slot : slot + 1, :] + gamma * consensus
            branch.z = mx.concatenate([z[:, :slot, :], comm, z[:, slot + 1 :, :]], axis=1)
            branch.workspace.update(branch.z)
        mx.eval(*[b.z for b in live])

        recipient_rows = []
        for branch in live:
            prior = prior_states[branch.index]
            prior_non_comm = mx.concatenate([prior[:, :slot, :], prior[:, slot + 1 :, :]], axis=1)
            post_non_comm = mx.concatenate(
                [branch.z[:, :slot, :], branch.z[:, slot + 1 :, :]], axis=1
            )
            mx.eval(prior_non_comm, post_non_comm)
            comm_pre = _tensor_sha256(prior[:, slot : slot + 1, :])
            comm_post = _tensor_sha256(branch.z[:, slot : slot + 1, :])
            state_pre = _tensor_sha256(prior)
            state_post = _tensor_sha256(branch.z)
            recipient_rows.append(
                {
                    "branch_index": branch.index,
                    "comm_pre_sha256": comm_pre,
                    "comm_post_sha256": comm_post,
                    "non_comm_pre_sha256": _tensor_sha256(prior_non_comm),
                    "non_comm_post_sha256": _tensor_sha256(post_non_comm),
                    "state_pre_sha256": state_pre,
                    "state_post_sha256": state_post,
                    "causal": comm_pre != comm_post and state_pre != state_post,
                }
            )
        if not any(row["causal"] for row in recipient_rows):
            return False

        ordinal = len(self.exchange_receipts)
        hidden_dimension = int(live[0].z.shape[-1])
        payload = {
            "schema": BRANCH_EXCHANGE_SCHEMA,
            "ordinal": ordinal,
            "sync_kind": sync_kind,
            "sync_id": sync_id,
            "generation": (
                "lesioned_candidates"
                if ordinal == 0 and self._configured_role_lesion
                else "independent_candidates"
                if ordinal == 0
                else "cooperative_refinement"
            ),
            "n_branches": len(live),
            "n_slots": n_slots,
            "comm_slot": slot,
            "exchange_gamma": gamma,
            "source_policy": ("bounded_private_reasoning_mean_excluding_mailbox_and_context_v1"),
            "message_representation": "latent_tensor_only",
            "message_slot_count": 1,
            "hidden_dimension": hidden_dimension,
            "source_slot_limit": MAX_EXCHANGE_SOURCE_SLOTS,
            "context_slots_excluded": context_slots,
            "comm_slot_excluded": True,
            "first_answer_text_exposed": False,
            "prior_peer_context_possible": ordinal > 0,
            "counts_as_independent_support": (ordinal == 0 and not self._configured_role_lesion),
            "candidate_set_sha256": candidate_set_sha256(isolation_binding),
            "source_rows": source_rows,
            "consensus_sha256": _tensor_sha256(consensus),
            "recipient_rows": recipient_rows,
            "tensor_accounting": {
                "source_elements_read": len(live) * len(source_slots) * hidden_dimension,
                "message_elements_emitted": len(live) * hidden_dimension,
                "consensus_elements_written": len(live) * hidden_dimension,
                "tensor_scalar_ops": (
                    len(live) * hidden_dimension * (len(source_slots) + 12) + 9 * len(live)
                ),
                "hidden_layer_apps": 0,
            },
        }
        receipt = {**payload, "receipt_sha256": canonical_sha256(payload)}
        validate_branch_exchange_receipt(
            receipt,
            n_branches=len(live),
            n_slots=n_slots,
            comm_slot=slot,
            exchange_gamma=gamma,
            branch_isolation=isolation_binding,
            cognitive_slots=[{"slot": index} for index in context_slots],
            expected_ordinal=ordinal,
        )
        self.exchange_receipts.append(receipt)
        if budget is not None:
            accounting = receipt["tensor_accounting"]
            budget.charge_tensor_work(
                "branch_exchange",
                element_reads=accounting["source_elements_read"],
                element_writes=(
                    accounting["message_elements_emitted"]
                    + accounting["consensus_elements_written"]
                ),
                scalar_ops=accounting["tensor_scalar_ops"],
            )
        self._exchange_sync_points.add(sync_point)
        self.exchanges += 1
        self._cross_exposure_started = True
        if self._first_exchange_step is None:
            self._first_exchange_step = min(branch.steps for branch in live)
        return True

    def maintain_diversity(
        self,
        *,
        budget: ComputeBudget | None = None,
    ) -> bool:
        """Decorrelate near-parallel branch pairs with deterministic jitter."""
        import mlx.core as mx

        live = self.active()
        if len(live) > 1 and not self._isolation_sealed:
            self._blocked_cross_exposures += 1
            return False
        pair_count = len(live) * (len(live) - 1) // 2
        if budget is not None and live:
            hidden = int(live[0].z.shape[-1])
            slots = int(live[0].z.shape[1])
            budget.charge_tensor_work(
                "branch_diversity_guard",
                element_reads=2 * pair_count * slots * hidden,
                scalar_ops=pair_count * ((2 * slots + 6) * hidden + 4),
                host_scalar_ops=pair_count * 4,
            )
        for i in range(len(live)):
            for j in range(i + 1, len(live)):
                a, b = live[i], live[j]
                sa, sb = a.workspace.summary(), b.workspace.summary()
                cos = float(
                    mx.sum(sa * sb) / mx.maximum(mx.linalg.norm(sa) * mx.linalg.norm(sb), 1e-6)
                )
                if cos <= self.config.collapse_cos_threshold:
                    continue
                key = mx.random.key(1000 + 31 * a.index + b.index + b.steps)
                jitter = mx.random.normal(b.z.shape, key=key)
                jitter = jitter * (
                    float(self.config.jitter_scale)
                    * per_position_rms(b.z)
                    / mx.maximum(per_position_rms(jitter), 1e-6)
                )
                b.z = b.z + jitter
                b.z = b.workspace.restore_context_evidence(b.z)
                b.workspace.update(b.z)
                mx.eval(b.z)
                if budget is not None:
                    elements = int(b.z.size)
                    budget.charge_tensor_work(
                        "branch_diversity_jitter",
                        element_reads=3 * elements,
                        element_writes=elements,
                        scalar_ops=12 * elements,
                    )
                logger.debug("Branch diversity jitter: %s↔%s cos=%.4f", a.index, b.index, cos)
        return True

    # ── Selection ───────────────────────────────────────────────────────
    def all_halted(self) -> bool:
        return all(b.halted for b in self.branches)

    def select(self, score_fn: Callable[[BranchState], float] | None = None) -> BranchState:
        """Pick the winning branch: external score if given, else convergence.

        Convergence quality = negative last residual — a branch that settled
        into a fixed point beats one still wandering when no verifier exists.
        """
        for branch in self.branches:
            if score_fn is not None:
                branch.score = float(score_fn(branch))
            else:
                trail = branch.halting.residual_trail
                # Empty trajectories are ineligible, but public causal receipts
                # must remain canonical JSON. A finite floor preserves the same
                # ordering without leaking an IEEE infinity into evidence.
                branch.score = (
                    -trail[-1]
                    if trail
                    else _UNSCORED_BRANCH_FLOOR
                )
        return max(self.branches, key=lambda b: b.score)

    def to_receipt(self) -> dict[str, Any]:
        return {
            "n_branches": len(self.branches),
            "exchanges": self.exchanges,
            "branches": [b.to_receipt() for b in self.branches],
        }


__all__ = [
    "BRANCH_ISOLATION_SCHEMA",
    "BRANCH_ROLES",
    "BranchEnsemble",
    "BranchState",
]
