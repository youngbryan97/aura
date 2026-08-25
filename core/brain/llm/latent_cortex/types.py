"""Configuration, budget, and receipt types for the Recursive Latent Cortex.

Everything the engine does is parameterized here and everything it did is
reported here. Receipts are the honesty spine: an episode that diverged,
blew its budget, or fell back to the vanilla path says so in machine-readable
form — downstream consumers (health, ledgers, the experiment harness) never
have to infer what happened.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

from core.brain.llm.latent_cortex.resource_accounting import (
    ModelComputeProfile,
    ResourceLedger,
    validate_information_receipt,
)

# Hard ceilings no configuration may exceed. These protect the live host:
# a runaway schedule on the resident 32B is a memory/latency incident, not
# an experiment. Operators may lower them via config, never raise them.
ABSOLUTE_MAX_RECURRENT_STEPS = 64
ABSOLUTE_MAX_SLOTS = 128
ABSOLUTE_MAX_BRANCHES = 8
ABSOLUTE_MAX_LAYER_APPS = 500_000_000  # token-layer applications per episode
# Claim-grade resident-model episodes can legitimately need more than the
# serving controller's 900-second default.  Keep a hard host-protection
# ceiling, but do not silently truncate the explicit 1,800-second research
# contract used by frozen campaigns.
ABSOLUTE_MAX_WALL_CLOCK_S = 1_800.0

# Default per-episode compute in token-layer applications. Sized so a 64-layer
# model with a 2k prompt (prefill 2048*64 ≈ 131k) plus 32 slots recurring over
# a 32-layer window for 16 steps (32*32*16 ≈ 16k) fits with wide margin.
DEFAULT_EPISODE_LAYER_APPS = 4_000_000


@dataclass
class WorkspaceConfig:
    """Writable latent workspace: M continuous thought slots."""

    n_slots: int = 16
    seed: int = 0
    # Role names seed deterministic anchor vectors; slot i takes roles[i % len].
    roles: tuple[str, ...] = (
        "objective",
        "constraints",
        "hypothesis",
        "counterexample",
        "world_state",
        "subgoal",
        "uncertainty",
        "self_monitor",
    )
    # Scale of the role-anchor perturbation applied on top of the pooled
    # prompt embedding (relative to embedding RMS).
    anchor_scale: float = 0.05


@dataclass
class RecurrenceConfig:
    """Controlled recurrence — the anti-naive-looping controls."""

    max_steps: int = 12
    min_steps: int = 2
    alpha: float = 0.5
    alpha_schedule: str = "constant"  # constant | cosine
    # RMSMatch ratio clamp: the state entering the next step remains inside
    # this per-position RMS band around the fixed post-prelude anchor.
    rms_clip_ratio: float = 3.0
    # Fixed-point convergence: relative residual below eps ⇒ converged.
    convergence_eps: float = 0.02
    # Divergence guard: mean-RMS growth beyond this factor of the post-seed
    # state (or any non-finite value) ⇒ halt and revert to best state.
    divergence_ratio: float = 10.0
    # Training-parity mode retains divergence and budget guards, but does not
    # stop on convergence or substitute an earlier state after fixed steps.
    fixed_depth: bool = False


@dataclass
class BranchConfig:
    """Virtual width: K concurrent latent trajectories of the same weights."""

    n_branches: int = 1
    # Every branch must independently advance this many recurrent steps from
    # the original prompt before any cross-branch exchange or aggregation.
    isolation_steps: int = 1
    exchange_interval: int = 4
    # Blend factor when writing the cross-branch consensus into each branch's
    # communication slot.
    exchange_gamma: float = 0.35
    comm_slot: int = 0
    # Anti-collapse: if two branch summaries exceed this cosine similarity,
    # deterministic decorrelation jitter is applied to the later branch.
    collapse_cos_threshold: float = 0.98
    jitter_scale: float = 0.02
    # Role-causality instrumentation (Experiment R): when non-empty, branch
    # k takes roles[k] instead of the default rotation. Lesion arms repeat
    # one role; swap arms permute — proving the ANCHOR, not the branch
    # index, drives differentiated cognitive labor. Must match n_branches.
    roles: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        # Refused at construction, so a mismatched arm cannot exist anywhere.
        # The check used to live in BranchEnsemble.seed, after the prompt
        # embeddings were hashed — a misconfigured lesion arm reported a
        # tensor error rather than the naming mistake that caused it.
        roles = tuple(self.roles or ())
        if roles and len(roles) != self.n_branches:
            raise ValueError(
                "BranchConfig.roles must name exactly n_branches roles, got "
                f"{len(roles)} for {self.n_branches} branches"
            )


@dataclass
class LatentOptConfig:
    """Gradient descent over thoughts (frozen weights, Z is the variable)."""

    enabled: bool = False
    steps: int = 4
    lr: float = 0.05
    lambda_reconstruct: float = 1.0
    lambda_manifold: float = 0.5
    max_grad_norm: float = 1.0
    # When True the optimizer applies matched-magnitude RANDOM perturbations
    # instead of gradient steps — the Experiment-5 control arm.
    control_mode: bool = False


@dataclass
class FastWeightsConfig:
    """Episode-scoped low-rank ΔW = s·U Vᵀ on selected window-layer linears."""

    enabled: bool = False
    rank: int = 2
    scale: float = 1.0
    target: str = "o_proj"  # o_proj | down_proj
    # Where inside the recurrent region the bounded wrappers attach. ``early``
    # preserves the historical serving function; other placements are
    # explicit, source-bound experimental choices until causally promoted.
    # Named narrow-coda variants are fixed-width registered sites; they do not
    # inherit max_wrapped_layers and therefore remain stable experiment IDs.
    layer_placement: str = "early"
    opt_steps: int = 4
    lr: float = 0.01
    # Layers (within the recurrent window) that receive fast weights; None ⇒
    # every window layer. Keep small on big models.
    max_wrapped_layers: int = 8
    # A verified teaching event may compile its semantic U directions into a
    # query key analytically: V_j = gain*x/(||x||^2 + lambda).  This is an
    # episode-scoped minimum-norm write, followed by the ordinary optimizer,
    # verifier, canaries, and exact erase boundary.
    associative_bootstrap_enabled: bool = True
    associative_bootstrap_gain: float = 0.25
    associative_bootstrap_regularization: float = 1e-4
    # Keys for the supervised trajectory map must be an explicit experiment
    # identity. The historical live-query keys bind tightly to the actual
    # decode; incumbent-trajectory keys instead learn the layer-local map from
    # a refuted answer state to the verified correction state.
    supervised_trajectory_key_source: str = "live_query"
    # A direct verified write is useful only for the episode that taught it.
    # Gate the temporary delta by cosine similarity to private, captured query
    # activations so unrelated canary and user contexts remain near identity.
    query_gate_enabled: bool = True
    query_gate_threshold: float = 0.8
    query_gate_temperature: float = 0.05
    # Diagnostic-only associative memory at the normalized output boundary.
    # It cannot authorize a served answer until matched controls establish
    # that verified supervision, rather than any supervision, is selective.
    output_memory_diagnostic_enabled: bool = False
    # Claim-grade phase lesion over one frozen learned delta. Disabled in
    # ordinary serving because it requires four matched answer decodes.
    locality_diagnostic_enabled: bool = False
    # Export mechanically-clean episode synapses (accepted descent + proven
    # erase) to the governed consolidation queue for the compounding loop.
    export_candidates: bool = False
    # In-episode protected-behavior canaries: before any decode happens under
    # active ΔW, a tiny protected battery (prose / instruction-following /
    # tool syntax / identity / calibration / reasoning) is measured under the
    # adapted function and compared to the base function. Regression beyond
    # the drop threshold walks a bounded ladder: halve the fast-weight scale
    # and re-measure (up to canary_rescale_attempts), then erase entirely.
    canary_enabled: bool = True
    canary_max_logprob_drop: float = 0.5
    # Structural backstop for a canary battery's inevitable blind spots. The
    # engine computes the exact RMS of s*U@V.T from rank-sized Gram matrices;
    # an update above this ceiling is rescaled or erased before any decode.
    canary_max_effective_delta_rms: float = 0.05
    canary_rescale_attempts: int = 2
    canary_max_tokens: int = 24
    # CP126: the likelihood battery is a fingerprint, not a postcondition. The
    # generated battery decodes under the adapted function and CHECKS what
    # came out. It is an order of magnitude more expensive — greedy decode
    # re-runs the forward pass per token — so its cost is reserved up front
    # rather than discovered as a mid-episode overrun. Turning it off is a
    # deliberate choice to run on fingerprints only, and the receipt grades
    # such an episode FINGERPRINT_ONLY rather than passed.
    canary_generated_enabled: bool = True


@dataclass
class ComputeBudget:
    """Episode admission budget plus claim-grade operation accounting."""

    max_layer_apps: int = DEFAULT_EPISODE_LAYER_APPS
    wall_clock_s: float = 120.0
    started_monotonic: float = field(default_factory=time.monotonic)
    spent_layer_apps: int = 0
    resource_ledger: ResourceLedger = field(default_factory=ResourceLedger, repr=False)
    information_receipt: dict[str, Any] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if isinstance(self.max_layer_apps, bool) or not isinstance(self.max_layer_apps, int):
            raise TypeError("max_layer_apps must be an integer")
        if self.max_layer_apps <= 0:
            raise ValueError("max_layer_apps must be positive")
        self.max_layer_apps = min(self.max_layer_apps, ABSOLUTE_MAX_LAYER_APPS)
        if isinstance(self.wall_clock_s, bool) or not isinstance(self.wall_clock_s, (int, float)):
            raise TypeError("wall_clock_s must be numeric")
        self.wall_clock_s = float(self.wall_clock_s)
        if not math.isfinite(self.wall_clock_s) or self.wall_clock_s <= 0.0:
            raise ValueError("wall_clock_s must be finite and positive")
        self.wall_clock_s = min(self.wall_clock_s, ABSOLUTE_MAX_WALL_CLOCK_S)
        if self.spent_layer_apps < 0:
            raise ValueError("spent_layer_apps cannot be negative")

    def bind_model(self, model: Any) -> ModelComputeProfile:
        profile = ModelComputeProfile.from_model(model)
        self.resource_ledger.bind_profile(profile)
        return profile

    def bind_information(self, receipt: dict[str, Any]) -> None:
        self.information_receipt = validate_information_receipt(receipt)

    def charge(
        self,
        tokens: int,
        layers: int,
        *,
        operation: str = "unclassified_transformer_forward",
        attention_pairs: int | None = None,
        output_head_tokens: int = 0,
    ) -> None:
        if (
            isinstance(tokens, bool)
            or isinstance(layers, bool)
            or not isinstance(tokens, int)
            or not isinstance(layers, int)
            or tokens < 0
            or layers < 0
        ):
            raise ValueError("budget charges require non-negative integer tokens and layers")
        layer_apps = tokens * layers
        if layer_apps > self.remaining_layer_apps:
            raise RuntimeError(
                f"compute budget exhausted: requested={layer_apps} "
                f"remaining={self.remaining_layer_apps}"
            )
        self.spent_layer_apps += layer_apps
        if attention_pairs is None:
            self.resource_ledger.mark_unknown(f"{operation}:attention_pairs")
            attention_pairs = 0
        self.resource_ledger.charge(
            operation,
            transformer_layer_apps=layer_apps,
            attention_query_key_pairs=attention_pairs,
            output_head_tokens=output_head_tokens,
        )

    def charge_layer_apps(
        self,
        layer_apps: int,
        *,
        operation: str = "unclassified_layer_app_equivalent",
    ) -> None:
        if isinstance(layer_apps, bool) or not isinstance(layer_apps, int) or layer_apps < 0:
            raise ValueError("layer-app charge must be a non-negative integer")
        if layer_apps > self.remaining_layer_apps:
            raise RuntimeError(
                f"compute budget exhausted: requested={layer_apps} "
                f"remaining={self.remaining_layer_apps}"
            )
        self.spent_layer_apps += layer_apps
        if layer_apps:
            self.resource_ledger.mark_unknown(operation)

    def charge_tensor_work(
        self,
        operation: str,
        *,
        element_reads: int = 0,
        element_writes: int = 0,
        scalar_ops: int = 0,
        host_scalar_ops: int = 0,
    ) -> None:
        self.resource_ledger.charge(
            operation,
            tensor_element_reads=element_reads,
            tensor_element_writes=element_writes,
            tensor_scalar_ops=scalar_ops,
            host_scalar_ops=host_scalar_ops,
        )

    def charge_verifier(
        self,
        operation: str,
        *,
        input_bytes: int,
        output_bytes: int = 8,
        host_scalar_ops: int = 0,
    ) -> None:
        self.resource_ledger.charge(
            operation,
            verifier_calls=1,
            verifier_input_bytes=input_bytes,
            verifier_output_bytes=output_bytes,
            host_scalar_ops=host_scalar_ops,
        )

    def charge_proxy_work(
        self,
        operation: str,
        *,
        layer_app_equivalents: int,
        scalar_ops: int,
    ) -> None:
        if (
            type(layer_app_equivalents) is not int
            or layer_app_equivalents < 0
            or type(scalar_ops) is not int
            or scalar_ops < 0
        ):
            raise ValueError("proxy work requires non-negative integer costs")
        if layer_app_equivalents > self.remaining_layer_apps:
            raise RuntimeError(
                "compute budget exhausted: "
                f"requested={layer_app_equivalents} remaining={self.remaining_layer_apps}"
            )
        self.spent_layer_apps += layer_app_equivalents
        self.resource_ledger.charge(operation, tensor_scalar_ops=scalar_ops)

    def charge_training_work(
        self,
        operation: str,
        *,
        tokens: int,
        layers: int,
        attention_pairs_per_forward: int,
        forward_evaluations: int,
        backward_evaluations: int,
    ) -> None:
        for name, value in (
            ("tokens", tokens),
            ("layers", layers),
            ("attention_pairs_per_forward", attention_pairs_per_forward),
            ("forward_evaluations", forward_evaluations),
            ("backward_evaluations", backward_evaluations),
        ):
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        equivalents = tokens * layers * (forward_evaluations + 2 * backward_evaluations)
        if equivalents > self.remaining_layer_apps:
            raise RuntimeError(
                "compute budget exhausted: "
                f"requested={equivalents} remaining={self.remaining_layer_apps}"
            )
        profile = self.resource_ledger.profile
        if profile is None:
            self.resource_ledger.mark_unknown(f"{operation}:model_profile")
            backward_flops = 0
        else:
            one_forward_flops = profile.estimate_neural_flops(
                transformer_layer_apps=tokens * layers,
                attention_query_key_pairs=attention_pairs_per_forward,
                output_head_tokens=0,
            )
            backward_flops = 2 * one_forward_flops * backward_evaluations
        self.spent_layer_apps += equivalents
        self.resource_ledger.charge(
            operation,
            transformer_layer_apps=tokens * layers * forward_evaluations,
            attention_query_key_pairs=(attention_pairs_per_forward * forward_evaluations),
            tensor_scalar_ops=backward_flops,
        )

    def charge_cleanup_overdraft(
        self,
        tokens: int,
        layers: int,
        *,
        operation: str = "cleanup_transformer_forward",
        attention_pairs: int | None = None,
        output_head_tokens: int = 0,
    ) -> None:
        """Charge safety-obligation work even past exhaustion.

        Cleanup proofs (fast-weight erase probes) must NEVER be refused for
        budget reasons — refusing converts a slow episode into an integrity
        failure and a worker recycle. The spend still lands in the receipt,
        so an overdraft is visible, just not refusable."""
        if (
            isinstance(tokens, bool)
            or isinstance(layers, bool)
            or not isinstance(tokens, int)
            or not isinstance(layers, int)
            or tokens < 0
            or layers < 0
        ):
            raise ValueError("budget charges require non-negative integer tokens and layers")
        self.spent_layer_apps += tokens * layers
        if attention_pairs is None:
            self.resource_ledger.mark_unknown(f"{operation}:attention_pairs")
            attention_pairs = 0
        self.resource_ledger.charge(
            operation,
            transformer_layer_apps=tokens * layers,
            attention_query_key_pairs=attention_pairs,
            output_head_tokens=output_head_tokens,
        )

    @property
    def remaining_wall_s(self) -> float:
        return max(0.0, self.wall_clock_s - (time.monotonic() - self.started_monotonic))

    def can_afford(self, tokens: int, layers: int, *, reserve_layer_apps: int = 0) -> bool:
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (tokens, layers, reserve_layer_apps)
        ):
            return False
        return (
            not self.exhausted and tokens * layers + reserve_layer_apps <= self.remaining_layer_apps
        )

    @property
    def exhausted(self) -> bool:
        if self.spent_layer_apps >= min(self.max_layer_apps, ABSOLUTE_MAX_LAYER_APPS):
            return True
        return (time.monotonic() - self.started_monotonic) >= self.wall_clock_s

    @property
    def remaining_layer_apps(self) -> int:
        return max(0, min(self.max_layer_apps, ABSOLUTE_MAX_LAYER_APPS) - self.spent_layer_apps)

    def to_receipt(self) -> dict[str, Any]:
        return {
            "max_layer_apps": self.max_layer_apps,
            "spent_layer_apps": self.spent_layer_apps,
            "wall_clock_s": self.wall_clock_s,
            "elapsed_s": round(time.monotonic() - self.started_monotonic, 3),
            "exhausted": self.exhausted,
            "resource_accounting": self.resource_ledger.to_receipt(),
            "information_accounting": (
                dict(self.information_receipt) if self.information_receipt is not None else None
            ),
        }


@dataclass
class CortexConfig:
    """The integrated machine's full configuration."""

    workspace: WorkspaceConfig = field(default_factory=WorkspaceConfig)
    recurrence: RecurrenceConfig = field(default_factory=RecurrenceConfig)
    branches: BranchConfig = field(default_factory=BranchConfig)
    latent_opt: LatentOptConfig = field(default_factory=LatentOptConfig)
    fast_weights: FastWeightsConfig = field(default_factory=FastWeightsConfig)
    # Maximum prompt tokens in one materialized prefill graph. The cache
    # carries exact state between chunks, so this changes peak memory rather
    # than model semantics. Resident-scale hybrid checkpoints cannot safely
    # materialize a long prompt across every decoder layer in one graph.
    prefill_chunk_tokens: int = 128
    # Prelude/coda split as layer fractions (window = the middle region).
    prelude_frac: float = 0.25
    coda_frac: float = 0.25
    # Explicit schedule program (schedules.LayerSchedule.to_dict() form); None
    # ⇒ single-window default derived from recurrence.max_steps.
    schedule: dict[str, Any] | None = None
    # Decode settings for the answer produced after latent computation.
    decode_max_tokens: int = 512
    decode_temperature: float = 0.0
    decode_top_p: float = 1.0
    decode_bridge_policy: str = "none"
    # Public decode authority. ``latent`` preserves historical/research
    # behavior. ``vanilla_incumbent`` runs the full latent episode but emits
    # from the untouched prompt-tail lane until a separately admitted fusion
    # policy has demonstrated a behavioral gain.
    decode_incumbent_policy: str = "latent"
    # Terminal disposition language injected ahead of the answer decode.
    # ``applied`` preserves live behavior: the episode tells itself, in words,
    # how its own reasoning ended. ``suppressed`` withholds that block so a
    # research arm decodes from exactly the context an ordinary vanilla decode
    # sees. The disposition is still classified and receipted either way — only
    # the injection is withheld — so an arm that suppresses it is still
    # attributable. Suppression exists because the injected text is an
    # instruction ("give only the best bounded answer", "disclose the
    # unresolved part"), and an arm that receives it is not comparable to a
    # control arm that does not.
    terminal_instruction_policy: str = "applied"
    # Contract-aware decode termination (CP180): "final_answer_v1" stops the
    # decode the moment a single FINAL_ANSWER JSON object completes — a
    # uniform serving-side stop rule so bounded budgets measure reasoning,
    # not truncation. "none" preserves historical behavior.
    decode_contract: str = "none"
    # Additional model-generated tokens allowed to close a required terminal
    # answer object after decode_max_tokens. Zero preserves the hard ceiling.
    decode_contract_grace_tokens: int = 0
    # CTRL-style sliding-window repetition penalty for the answer decode.
    # 1.0 disables; the resident live profile runs 1.25 over 72 tokens —
    # CP105's live turn proved a degeneration loop survives temperature
    # tuning alone (one line repeated ~80 times at t=0.35).
    decode_repetition_penalty: float = 1.0
    decode_repetition_window: int = 72
    # EOS suppression floor: sampling variance can emit end-of-sequence a
    # handful of tokens into an answer (a live 32B turn stopped at 16
    # tokens). Until this many tokens exist, EOS logits are masked — the
    # standard min-new-tokens constraint. 0 disables.
    decode_min_tokens: int = 0
    # Task-verifier probes are answer previews, not user-visible answers. The
    # lab/frontier default remains broad; the resident interactive profile may
    # use a shorter, explicitly receipted probe to preserve the answer budget.
    verifier_probe_max_tokens: int = 48
    # Candidate probes have a separate answer contract because they feed
    # verification and promotion, while the public decode may remain an exact
    # vanilla incumbent. Coupling the two either moved the floor or left every
    # candidate ungradeable.
    verifier_probe_contract: str = "none"
    # What a caller-supplied branch verifier IS. "advisory" means it guides
    # selection when the budget allows and the internal score stands in when
    # it does not. "required" means the caller supplied it as a correctness
    # gate, so an episode that cannot pay for branch verification fails
    # instead of quietly selecting on the ensemble's own score.
    branch_verifier_mode: str = "advisory"
    # Fresh-context generative challenge lane. It shares the resident
    # checkpoint but imports no solver KV state; generated prose has no
    # authority unless a deterministic witness relation reconstructs.
    generative_verifier_enabled: bool = True
    generative_verifier_max_atoms: int = 1
    generative_verifier_max_tokens: int = 160
    # Fresh-context interventions may resolve only an exact top task-verifier
    # tie. Every tied branch receives equal, machine-checkable coverage.
    counterfactual_verifier_enabled: bool = True
    counterfactual_verifier_max_atoms: int = 1
    counterfactual_verifier_max_interventions: int = 2
    counterfactual_verifier_max_tokens: int = 128
    # Fresh, seed-isolated continuations from a machine-verified prefix.
    # The signal measures future conclusion recurrence, never correctness,
    # and cannot affect branch selection in SPARK-045.
    prefix_stability_enabled: bool = True
    prefix_stability_samples: int = 3
    prefix_stability_max_tokens: int = 128
    prefix_stability_temperature: float = 0.35
    prefix_stability_top_p: float = 0.9
    prefix_stability_seed: int = 104_729
    prefix_stability_calibrator: dict[str, Any] | None = None
    # Exact verifier refutations may trigger one source-private regeneration
    # from the last unchanged atomic prefix.
    local_repair_enabled: bool = True
    local_repair_max_attempts: int = 1
    local_repair_max_tokens: int = 128
    # A repaired candidate can replace the accepted branch answer only when
    # its deterministic correctness lower bound clears the original upper
    # bound by this preregistered margin.
    answer_replacement_enabled: bool = True
    # Permit the replacement gate to compile a recognized, finite public
    # objective into an independently reconstructable exact solution. This is
    # separate from answer replacement itself so causal experiments can remove
    # the producer while preserving every selection and safety boundary.
    objective_program_enabled: bool = True
    # Permit an independently verified objective program to provide a private
    # teaching target to episode-scoped fast weights.  This authority is
    # deliberately separate from ``objective_program_enabled``: a causal
    # experiment must be able to withdraw the executable answer producer while
    # retaining the neural intervention, then require the adapted model to
    # regenerate the answer without exposing the target on the public lane.
    verified_objective_teacher_enabled: bool = True
    answer_replacement_margin: float = 0.05
    # Strict experiments accept only a higher task-verifier score. The live
    # product profile may additionally accept an exactly non-regressing score
    # when the candidate also proves descent on the answer-leak-proof proxy.
    verifier_accept_non_regression: bool = False
    input_context_max_chars: int = 0
    allow_vanilla_fallback: bool = True
    # Structured attractor-escape ladder for diverged/stalled branches
    # (escape.EscapeConfig form); None ⇒ ladder enabled with defaults.
    escape: dict[str, Any] | None = None
    # Per-episode latent interpretability/safety telemetry in the receipt.
    telemetry_enabled: bool = True
    # Per-episode decode-probe memoization: identical latent states decode
    # once; the cache flushes on every fast-weight function change.
    probe_cache_enabled: bool = True
    # Learned halting attachment (learned_halting_bridge). None ⇒ residual
    # policy, byte-for-byte the engine's historical behaviour. Learned mode
    # requires a trained head on disk: {"mode": "learned",
    # "head_path": "...", "threshold": optional (0,1)}. A requested head
    # that cannot load REFUSES the episode rather than silently reporting
    # learned allocation while running the residual rule.
    halting: dict[str, Any] | None = None
    # Per-transition accept/discard policy. Learned mode requires a calibrated
    # artifact and its exact SHA-256; an unreadable or changed head refuses the
    # episode. None/passthrough preserves historical recurrence explicitly.
    update_gate: dict[str, Any] | None = None
    # Hidden-state correctness/entropy measurement. Learned mode requires a
    # task-disjoint calibrated artifact and exact SHA-256. None/unavailable
    # emits no confidence rather than substituting generated self-report.
    uncertainty_head: dict[str, Any] | None = None
    # Transition-level mistake localization. Learned mode requires a
    # task-disjoint, OOD-admitted artifact and exact SHA-256. Localization is
    # diagnostic in SPARK-029 and cannot authorize repair steering.
    mistake_locator: dict[str, Any] | None = None
    # Full-trace, latent-position contradiction evidence. Learned mode
    # requires a disjoint ID/OOD-admitted artifact and exact SHA-256.
    # SPARK-031 remains diagnostic and cannot perturb attention.
    contradiction_head: dict[str, Any] | None = None
    # Bounded SPARK-032 intervention. Counterfactual mode is inert unless an
    # admitted contradiction coordinate and independently authoritative task
    # verifier are both present. Every attempted mutation competes against
    # repeated no-op and matched-random controls and rolls back by default.
    contradiction_perturber: dict[str, Any] | None = None
    # SPARK-033 bounded local exploration. Calibrated predictive entropy
    # scales source-bound stochastic candidates only at an admitted
    # contradiction position. Equal-compute no-op and stable-position sham
    # families prevent global or unearned exploration authority.
    local_exploration: dict[str, Any] | None = None
    # SPARK-034 incumbent/corrected/fused policy arbitration. A fusion weight
    # comes only from conservative verifier bounds, and every policy executes
    # both lanes under equal measured compute before it can affect decode.
    heterogeneous_integration: dict[str, Any] | None = None
    # SPARK-036 verified failure avoidance. None enables conservative live
    # defaults; callers may tighten bounded efficacy/lifetime settings but
    # cannot exceed the hard safety ceilings in TransientConstraintConfig.
    transient_negative_constraints: dict[str, Any] | None = None
    # SPARK-037 episode-local latent compute. None enables conservative live
    # defaults. Authority requires a real matched-compute verifier win; text,
    # caller vectors, and self-reported contribution cannot create a quantum.
    virtual_quanta: dict[str, Any] | None = None
    # SPARK-038 bounded search over complete recurrent/KV state snapshots.
    # None enables a small live UCT transaction when the value controller
    # selects BRANCH and an independently admitted bounded verifier exists.
    latent_tree_search: dict[str, Any] | None = None
    # Checked historical branch-error correlations. None is an explicit
    # bootstrap state: duplicate programs still collapse, but no empirical
    # relationship is invented before independently graded paired outcomes.
    branch_correlation_evidence: dict[str, Any] | None = None
    # Domain/global verifier reliability, calibration, and shared-error
    # evidence assembled only from independently checked outcomes. None is an
    # explicit unmeasured bootstrap and never creates correctness authority.
    verifier_fusion_evidence: dict[str, Any] | None = None
    # Independently graded generator/critic outcomes, keyed to the exact
    # function identities. The worker validates this before the critic can
    # influence recurrence; None is an honest unmeasured bootstrap.
    critic_blind_spot_evidence: dict[str, Any] | None = None

    def validate(self) -> list[str]:
        """Return a list of human-readable violations (empty ⇒ valid)."""
        problems: list[str] = []

        def finite(value: Any) -> bool:
            return (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
            )

        def integer_in(value: Any, minimum: int, maximum: int) -> bool:
            return type(value) is int and minimum <= value <= maximum

        if not integer_in(self.workspace.n_slots, 1, ABSOLUTE_MAX_SLOTS):
            problems.append(f"n_slots {self.workspace.n_slots} outside [1, {ABSOLUTE_MAX_SLOTS}]")
        if (
            not isinstance(self.workspace.roles, (list, tuple))
            or not self.workspace.roles
            or any(not isinstance(role, str) or not role.strip() for role in self.workspace.roles)
            or len(self.workspace.roles) > ABSOLUTE_MAX_SLOTS
        ):
            problems.append("workspace roles must be non-empty strings")
        if not integer_in(self.workspace.seed, -(2**63), 2**63 - 1):
            problems.append("workspace seed must be a signed 64-bit integer")
        if not finite(self.workspace.anchor_scale) or not 0.0 <= self.workspace.anchor_scale <= 1.0:
            problems.append("anchor_scale must be finite and inside [0, 1]")
        if not integer_in(self.recurrence.max_steps, 1, ABSOLUTE_MAX_RECURRENT_STEPS):
            problems.append(
                f"max_steps {self.recurrence.max_steps} outside [1, {ABSOLUTE_MAX_RECURRENT_STEPS}]"
            )
        if not (
            type(self.recurrence.min_steps) is int
            and type(self.recurrence.max_steps) is int
            and 1 <= self.recurrence.min_steps <= self.recurrence.max_steps
        ):
            problems.append("min_steps must be inside [1, max_steps]")
        if not isinstance(
            self.recurrence.alpha_schedule, str
        ) or self.recurrence.alpha_schedule not in {
            "constant",
            "cosine",
        }:
            problems.append("alpha_schedule must be constant or cosine")
        if not finite(self.recurrence.alpha) or not 0.0 < self.recurrence.alpha <= 1.0:
            problems.append(f"alpha {self.recurrence.alpha} outside (0, 1]")
        if (
            not finite(self.recurrence.rms_clip_ratio)
            or not 1.0 <= self.recurrence.rms_clip_ratio <= 100.0
        ):
            problems.append("rms_clip_ratio must be finite and inside [1, 100]")
        if (
            not finite(self.recurrence.convergence_eps)
            or not 0.0 < self.recurrence.convergence_eps <= 1.0
        ):
            problems.append("convergence_eps must be finite and inside (0, 1]")
        if (
            not finite(self.recurrence.divergence_ratio)
            or not 1.0 < self.recurrence.divergence_ratio <= 1000.0
        ):
            problems.append("divergence_ratio must be finite and inside (1, 1000]")
        if type(self.recurrence.fixed_depth) is not bool:
            problems.append("fixed_depth must be boolean")
        if not integer_in(self.prefill_chunk_tokens, 1, 8192):
            problems.append("prefill_chunk_tokens must be inside [1, 8192]")
        if not integer_in(self.branches.n_branches, 1, ABSOLUTE_MAX_BRANCHES):
            problems.append(
                f"n_branches {self.branches.n_branches} outside [1, {ABSOLUTE_MAX_BRANCHES}]"
            )
        if not (
            type(self.branches.isolation_steps) is int
            and type(self.recurrence.max_steps) is int
            and 1 <= self.branches.isolation_steps <= self.recurrence.max_steps
        ):
            problems.append("isolation_steps must be inside [1, max_steps]")
        if not integer_in(self.branches.exchange_interval, 1, ABSOLUTE_MAX_RECURRENT_STEPS):
            problems.append("exchange_interval outside recurrent-step limits")
        if (
            not finite(self.branches.exchange_gamma)
            or not 0.0 <= self.branches.exchange_gamma <= 1.0
        ):
            problems.append("exchange_gamma must be finite and inside [0, 1]")
        if not (
            type(self.branches.comm_slot) is int
            and type(self.workspace.n_slots) is int
            and 0 <= self.branches.comm_slot < self.workspace.n_slots
        ):
            problems.append("comm_slot index outside workspace")
        if (
            not finite(self.branches.collapse_cos_threshold)
            or not -1.0 <= self.branches.collapse_cos_threshold <= 1.0
        ):
            problems.append("collapse_cos_threshold must be finite and inside [-1, 1]")
        if not finite(self.branches.jitter_scale) or not 0.0 <= self.branches.jitter_scale <= 1.0:
            problems.append("jitter_scale must be finite and inside [0, 1]")
        if type(self.latent_opt.enabled) is not bool:
            problems.append("latent_opt.enabled must be boolean")
        if type(self.latent_opt.control_mode) is not bool:
            problems.append("latent_opt.control_mode must be boolean")
        if not integer_in(self.latent_opt.steps, 1, ABSOLUTE_MAX_RECURRENT_STEPS):
            problems.append("latent_opt.steps outside recurrent-step limits")
        if not finite(self.latent_opt.lr) or not 0.0 < self.latent_opt.lr <= 1.0:
            problems.append("latent_opt.lr must be finite and inside (0, 1]")
        if (
            not finite(self.latent_opt.lambda_reconstruct)
            or self.latent_opt.lambda_reconstruct < 0.0
        ):
            problems.append("latent_opt.lambda_reconstruct must be finite and non-negative")
        if not finite(self.latent_opt.lambda_manifold) or self.latent_opt.lambda_manifold < 0.0:
            problems.append("latent_opt.lambda_manifold must be finite and non-negative")
        if (
            not finite(self.latent_opt.max_grad_norm)
            or not 0.0 < self.latent_opt.max_grad_norm <= 1000.0
        ):
            problems.append("latent_opt.max_grad_norm must be finite and inside (0, 1000]")
        if not finite(self.prelude_frac) or not 0.0 < self.prelude_frac < 0.5:
            problems.append(f"prelude_frac {self.prelude_frac} outside (0, 0.5)")
        if not finite(self.coda_frac) or not 0.0 < self.coda_frac < 0.5:
            problems.append(f"coda_frac {self.coda_frac} outside (0, 0.5)")
        if (
            finite(self.prelude_frac)
            and finite(self.coda_frac)
            and self.prelude_frac + self.coda_frac >= 1.0
        ):
            problems.append("prelude_frac + coda_frac must be < 1")
        if self.schedule is not None and not isinstance(self.schedule, dict):
            problems.append("schedule must be a mapping or null")
        if not integer_in(self.decode_max_tokens, 1, 8192):
            problems.append("decode_max_tokens outside [1, 8192]")
        if not finite(self.decode_temperature) or not 0.0 <= self.decode_temperature <= 2.0:
            problems.append("decode_temperature must be finite and inside [0, 2]")
        if not finite(self.decode_top_p) or not 0.0 < self.decode_top_p <= 1.0:
            problems.append("decode_top_p must be finite and inside (0, 1]")
        if (
            not finite(self.decode_repetition_penalty)
            or not 1.0 <= self.decode_repetition_penalty <= 2.0
        ):
            problems.append("decode_repetition_penalty must be finite and inside [1, 2]")
        if (
            type(self.decode_repetition_window) is not int
            or not 1 <= self.decode_repetition_window <= 512
        ):
            problems.append("decode_repetition_window must be an integer inside [1, 512]")
        if (
            type(self.decode_min_tokens) is not int
            or not 0 <= self.decode_min_tokens <= 512
            or self.decode_min_tokens >= max(1, self.decode_max_tokens)
        ):
            problems.append(
                "decode_min_tokens must be an integer inside [0, 512] and below decode_max_tokens"
            )
        if not integer_in(self.verifier_probe_max_tokens, 16, 512):
            problems.append("verifier_probe_max_tokens outside [16, 512]")
        if self.branch_verifier_mode not in {"advisory", "required"}:
            problems.append("branch_verifier_mode must be 'advisory' or 'required'")
        if self.verifier_probe_contract not in {"none", "final_answer_v1"}:
            problems.append(
                "verifier_probe_contract must be 'none' or 'final_answer_v1'"
            )
        if type(self.generative_verifier_enabled) is not bool:
            problems.append("generative_verifier_enabled must be boolean")
        if not integer_in(self.generative_verifier_max_atoms, 1, 8):
            problems.append("generative_verifier_max_atoms outside [1, 8]")
        if not integer_in(self.generative_verifier_max_tokens, 32, 256):
            problems.append("generative_verifier_max_tokens outside [32, 256]")
        if type(self.counterfactual_verifier_enabled) is not bool:
            problems.append("counterfactual_verifier_enabled must be boolean")
        if not integer_in(self.counterfactual_verifier_max_atoms, 1, 4):
            problems.append("counterfactual_verifier_max_atoms outside [1, 4]")
        if not integer_in(self.counterfactual_verifier_max_interventions, 1, 3):
            problems.append("counterfactual_verifier_max_interventions outside [1, 3]")
        if not integer_in(self.counterfactual_verifier_max_tokens, 32, 256):
            problems.append("counterfactual_verifier_max_tokens outside [32, 256]")
        if type(self.prefix_stability_enabled) is not bool:
            problems.append("prefix_stability_enabled must be boolean")
        if not integer_in(self.prefix_stability_samples, 3, 8):
            problems.append("prefix_stability_samples outside [3, 8]")
        if not integer_in(self.prefix_stability_max_tokens, 32, 256):
            problems.append("prefix_stability_max_tokens outside [32, 256]")
        if (
            not finite(self.prefix_stability_temperature)
            or not 0.05 <= self.prefix_stability_temperature <= 1.5
        ):
            problems.append("prefix_stability_temperature outside [0.05, 1.5]")
        if (
            not finite(self.prefix_stability_top_p)
            or not 0.1 <= self.prefix_stability_top_p <= 1.0
        ):
            problems.append("prefix_stability_top_p outside [0.1, 1]")
        if not integer_in(self.prefix_stability_seed, -(2**63), 2**63 - 1):
            problems.append("prefix_stability_seed must be a signed 64-bit integer")
        if (
            self.prefix_stability_calibrator is not None
            and not isinstance(self.prefix_stability_calibrator, dict)
        ):
            problems.append("prefix_stability_calibrator must be a mapping or null")
        elif self.prefix_stability_calibrator is not None:
            calibrator = self.prefix_stability_calibrator
            if (
                set(calibrator) != {"mode", "artifact_path", "artifact_sha256"}
                or calibrator.get("mode") != "learned"
                or not isinstance(calibrator.get("artifact_path"), str)
                or not calibrator.get("artifact_path")
                or len(calibrator.get("artifact_path", "")) > 4096
                or not isinstance(calibrator.get("artifact_sha256"), str)
                or len(calibrator.get("artifact_sha256", "")) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in calibrator.get("artifact_sha256", "")
                )
            ):
                problems.append("prefix_stability_calibrator config is invalid")
        if self.decode_contract not in ("none", "final_answer_v1"):
            problems.append("decode_contract must be 'none' or 'final_answer_v1'")
        if self.decode_incumbent_policy not in {"latent", "vanilla_incumbent"}:
            problems.append(
                "decode_incumbent_policy must be latent or vanilla_incumbent"
            )
        if not integer_in(self.decode_contract_grace_tokens, 0, 4096):
            problems.append("decode_contract_grace_tokens outside [0, 4096]")
        if type(self.verifier_accept_non_regression) is not bool:
            problems.append("verifier_accept_non_regression must be boolean")
        if self.decode_bridge_policy not in {
            "none",
            "assistant_answer_v1",
            "assistant_answer_v2",
            "assistant_answer_v3",
            "assistant_answer_v4",
        }:
            problems.append("decode_bridge_policy must be none or an assistant_answer_v1-v4 policy")
        if self.terminal_instruction_policy not in {"applied", "suppressed"}:
            problems.append("terminal_instruction_policy must be applied or suppressed")
        if not (
            type(self.input_context_max_chars) is int
            and (self.input_context_max_chars == 0 or 2048 <= self.input_context_max_chars <= 65536)
        ):
            problems.append("input_context_max_chars must be 0 or inside [2048, 65536]")
        if type(self.allow_vanilla_fallback) is not bool:
            problems.append("allow_vanilla_fallback must be boolean")
        if type(self.fast_weights.enabled) is not bool:
            problems.append("fast_weights.enabled must be boolean")
        if not integer_in(self.fast_weights.rank, 1, 64):
            problems.append("fast_weights.rank outside [1, 64]")
        if not finite(self.fast_weights.scale) or not 0.0 < self.fast_weights.scale <= 16.0:
            problems.append("fast_weights.scale must be finite and inside (0, 16]")
        if not isinstance(self.fast_weights.target, str) or self.fast_weights.target not in {
            "o_proj",
            "down_proj",
        }:
            problems.append("fast_weights.target must be o_proj or down_proj")
        if self.fast_weights.layer_placement not in {
            "early",
            "distributed",
            "late",
            "coda",
            "coda_late4",
            "coda_late2",
            "coda_terminal",
        }:
            problems.append(
                "fast_weights.layer_placement is not a registered placement"
            )
        if not integer_in(self.fast_weights.opt_steps, 1, ABSOLUTE_MAX_RECURRENT_STEPS):
            problems.append("fast_weights.opt_steps outside recurrent-step limits")
        if not finite(self.fast_weights.lr) or not 0.0 < self.fast_weights.lr <= 1.0:
            problems.append("fast_weights.lr must be finite and inside (0, 1]")
        if not integer_in(
            self.fast_weights.max_wrapped_layers,
            1,
            ABSOLUTE_MAX_RECURRENT_STEPS,
        ):
            problems.append("fast_weights.max_wrapped_layers outside [1, 64]")
        if type(self.fast_weights.associative_bootstrap_enabled) is not bool:
            problems.append("fast_weights.associative_bootstrap_enabled must be boolean")
        if (
            not finite(self.fast_weights.associative_bootstrap_gain)
            or not 0.0 < self.fast_weights.associative_bootstrap_gain <= 4.0
        ):
            problems.append(
                "fast_weights.associative_bootstrap_gain must be finite and inside (0, 4]"
            )
        if (
            not finite(self.fast_weights.associative_bootstrap_regularization)
            or not 0.0 < self.fast_weights.associative_bootstrap_regularization <= 1.0
        ):
            problems.append(
                "fast_weights.associative_bootstrap_regularization must be finite and inside (0, 1]"
            )
        if self.fast_weights.supervised_trajectory_key_source not in {
            "live_query",
            "incumbent_trajectory",
        }:
            problems.append(
                "fast_weights.supervised_trajectory_key_source is unsupported"
            )
        if type(self.fast_weights.query_gate_enabled) is not bool:
            problems.append("fast_weights.query_gate_enabled must be boolean")
        if (
            not finite(self.fast_weights.query_gate_threshold)
            or not -1.0 < self.fast_weights.query_gate_threshold < 1.0
        ):
            problems.append(
                "fast_weights.query_gate_threshold must be finite and inside (-1, 1)"
            )
        if (
            not finite(self.fast_weights.query_gate_temperature)
            or not 0.0 < self.fast_weights.query_gate_temperature <= 1.0
        ):
            problems.append(
                "fast_weights.query_gate_temperature must be finite and inside (0, 1]"
            )
        if type(self.fast_weights.output_memory_diagnostic_enabled) is not bool:
            problems.append("fast_weights.output_memory_diagnostic_enabled must be boolean")
        if type(self.fast_weights.locality_diagnostic_enabled) is not bool:
            problems.append("fast_weights.locality_diagnostic_enabled must be boolean")
        if type(self.fast_weights.canary_enabled) is not bool:
            problems.append("fast_weights.canary_enabled must be boolean")
        if type(self.fast_weights.canary_generated_enabled) is not bool:
            problems.append("fast_weights.canary_generated_enabled must be boolean")
        if (
            not finite(self.fast_weights.canary_max_logprob_drop)
            or not 0.0 < self.fast_weights.canary_max_logprob_drop <= 10.0
        ):
            problems.append(
                "fast_weights.canary_max_logprob_drop must be finite and inside (0, 10]"
            )
        if (
            not finite(self.fast_weights.canary_max_effective_delta_rms)
            or not 0.0 < self.fast_weights.canary_max_effective_delta_rms <= 10.0
        ):
            problems.append(
                "fast_weights.canary_max_effective_delta_rms must be finite and inside (0, 10]"
            )
        if not integer_in(self.fast_weights.canary_rescale_attempts, 0, 8):
            problems.append("fast_weights.canary_rescale_attempts outside [0, 8]")
        if not integer_in(self.fast_weights.canary_max_tokens, 4, 128):
            problems.append("fast_weights.canary_max_tokens outside [4, 128]")
        if type(self.telemetry_enabled) is not bool:
            problems.append("telemetry_enabled must be boolean")
        if type(self.probe_cache_enabled) is not bool:
            problems.append("probe_cache_enabled must be boolean")
        if self.halting is not None:
            if not isinstance(self.halting, dict):
                problems.append("halting must be a mapping or null")
            else:
                mode = self.halting.get("mode", "residual")
                if mode not in {"residual", "learned"}:
                    problems.append("halting.mode must be residual or learned")
                head_path = self.halting.get("head_path")
                head_sha256 = self.halting.get("head_sha256")
                if mode == "learned" and (not isinstance(head_path, str) or not head_path.strip()):
                    problems.append("halting.learned requires head_path")
                if mode == "learned" and (
                    not isinstance(head_sha256, str)
                    or len(head_sha256) != 64
                    or any(character not in "0123456789abcdef" for character in head_sha256)
                ):
                    problems.append("halting.learned requires head_sha256")
                if mode == "residual" and (head_path is not None or head_sha256 is not None):
                    problems.append("halting.residual cannot carry a head")
                unknown = set(self.halting) - {
                    "mode",
                    "head_path",
                    "head_sha256",
                }
                if unknown:
                    problems.append(f"halting has unknown keys: {sorted(unknown)}")
        if self.update_gate is not None:
            if not isinstance(self.update_gate, dict):
                problems.append("update_gate must be a mapping or null")
            else:
                mode = self.update_gate.get("mode", "passthrough")
                if mode not in {"passthrough", "learned"}:
                    problems.append("update_gate.mode must be passthrough or learned")
                head_path = self.update_gate.get("head_path")
                head_sha256 = self.update_gate.get("head_sha256")
                if mode == "learned" and (not isinstance(head_path, str) or not head_path.strip()):
                    problems.append("update_gate.learned requires head_path")
                if mode == "learned" and (
                    not isinstance(head_sha256, str)
                    or len(head_sha256) != 64
                    or any(character not in "0123456789abcdef" for character in head_sha256)
                ):
                    problems.append("update_gate.learned requires head_sha256")
                if mode == "passthrough" and (head_path is not None or head_sha256 is not None):
                    problems.append("update_gate.passthrough cannot carry a head")
                unknown = set(self.update_gate) - {
                    "mode",
                    "head_path",
                    "head_sha256",
                }
                if unknown:
                    problems.append(f"update_gate has unknown keys: {sorted(unknown)}")
        if self.uncertainty_head is not None:
            if not isinstance(self.uncertainty_head, dict):
                problems.append("uncertainty_head must be a mapping or null")
            else:
                mode = self.uncertainty_head.get("mode", "unavailable")
                if mode not in {"unavailable", "learned"}:
                    problems.append("uncertainty_head.mode must be unavailable or learned")
                head_path = self.uncertainty_head.get("head_path")
                head_sha256 = self.uncertainty_head.get("head_sha256")
                if mode == "learned" and (not isinstance(head_path, str) or not head_path.strip()):
                    problems.append("uncertainty_head.learned requires head_path")
                if mode == "learned" and (
                    not isinstance(head_sha256, str)
                    or len(head_sha256) != 64
                    or any(character not in "0123456789abcdef" for character in head_sha256)
                ):
                    problems.append("uncertainty_head.learned requires head_sha256")
                if mode == "unavailable" and (head_path is not None or head_sha256 is not None):
                    problems.append("uncertainty_head.unavailable cannot carry a head")
                unknown = set(self.uncertainty_head) - {
                    "mode",
                    "head_path",
                    "head_sha256",
                }
                if unknown:
                    problems.append(f"uncertainty_head has unknown keys: {sorted(unknown)}")
        if self.mistake_locator is not None:
            if not isinstance(self.mistake_locator, dict):
                problems.append("mistake_locator must be a mapping or null")
            else:
                mode = self.mistake_locator.get("mode", "unavailable")
                if mode not in {"unavailable", "learned"}:
                    problems.append("mistake_locator.mode must be unavailable or learned")
                head_path = self.mistake_locator.get("head_path")
                head_sha256 = self.mistake_locator.get("head_sha256")
                if mode == "learned" and (not isinstance(head_path, str) or not head_path.strip()):
                    problems.append("mistake_locator.learned requires head_path")
                if mode == "learned" and (
                    not isinstance(head_sha256, str)
                    or len(head_sha256) != 64
                    or any(character not in "0123456789abcdef" for character in head_sha256)
                ):
                    problems.append("mistake_locator.learned requires head_sha256")
                if mode == "unavailable" and (head_path is not None or head_sha256 is not None):
                    problems.append("mistake_locator.unavailable cannot carry a head")
                unknown = set(self.mistake_locator) - {
                    "mode",
                    "head_path",
                    "head_sha256",
                }
                if unknown:
                    problems.append(f"mistake_locator has unknown keys: {sorted(unknown)}")
        if self.contradiction_head is not None:
            if not isinstance(self.contradiction_head, dict):
                problems.append("contradiction_head must be a mapping or null")
            else:
                mode = self.contradiction_head.get("mode", "unavailable")
                if mode not in {"unavailable", "learned"}:
                    problems.append("contradiction_head.mode must be unavailable or learned")
                head_path = self.contradiction_head.get("head_path")
                head_sha256 = self.contradiction_head.get("head_sha256")
                if mode == "learned" and (not isinstance(head_path, str) or not head_path.strip()):
                    problems.append("contradiction_head.learned requires head_path")
                if mode == "learned" and (
                    not isinstance(head_sha256, str)
                    or len(head_sha256) != 64
                    or any(character not in "0123456789abcdef" for character in head_sha256)
                ):
                    problems.append("contradiction_head.learned requires head_sha256")
                if mode == "unavailable" and (head_path is not None or head_sha256 is not None):
                    problems.append("contradiction_head.unavailable cannot carry a head")
                unknown = set(self.contradiction_head) - {
                    "mode",
                    "head_path",
                    "head_sha256",
                }
                if unknown:
                    problems.append(f"contradiction_head has unknown keys: {sorted(unknown)}")
        if self.contradiction_perturber is not None:
            if not isinstance(self.contradiction_perturber, dict):
                problems.append("contradiction_perturber must be a mapping or null")
            else:
                try:
                    from core.brain.llm.latent_cortex.contradiction_perturber import (
                        ContradictionPerturberConfig,
                    )

                    ContradictionPerturberConfig.from_value(self.contradiction_perturber)
                except (TypeError, ValueError) as exc:
                    problems.append(str(exc))
        if type(self.local_repair_enabled) is not bool:
            problems.append("local_repair_enabled must be boolean")
        if not integer_in(self.local_repair_max_attempts, 0, 8):
            problems.append("local_repair_max_attempts outside [0, 8]")
        if not integer_in(self.local_repair_max_tokens, 32, 512):
            problems.append("local_repair_max_tokens outside [32, 512]")
        if type(self.answer_replacement_enabled) is not bool:
            problems.append("answer_replacement_enabled must be boolean")
        if type(self.objective_program_enabled) is not bool:
            problems.append("objective_program_enabled must be boolean")
        if type(self.verified_objective_teacher_enabled) is not bool:
            problems.append("verified_objective_teacher_enabled must be boolean")
        if (
            isinstance(self.answer_replacement_margin, bool)
            or not isinstance(self.answer_replacement_margin, (int, float))
            or not math.isfinite(float(self.answer_replacement_margin))
            or not 0.0 <= float(self.answer_replacement_margin) < 1.0
        ):
            problems.append("answer_replacement_margin outside [0, 1)")
        if self.local_exploration is not None:
            if not isinstance(self.local_exploration, dict):
                problems.append("local_exploration must be a mapping or null")
            else:
                try:
                    from core.brain.llm.latent_cortex.local_exploration import (
                        LocalExplorationConfig,
                    )

                    LocalExplorationConfig.from_value(self.local_exploration)
                except (TypeError, ValueError) as exc:
                    problems.append(str(exc))
        if self.heterogeneous_integration is not None:
            if not isinstance(self.heterogeneous_integration, dict):
                problems.append("heterogeneous_integration must be a mapping or null")
            else:
                try:
                    from core.brain.llm.latent_cortex.heterogeneous_integrator import (
                        HeterogeneousIntegrationConfig,
                    )

                    HeterogeneousIntegrationConfig.from_value(self.heterogeneous_integration)
                except (TypeError, ValueError) as exc:
                    problems.append(str(exc))
        if self.transient_negative_constraints is not None:
            if not isinstance(self.transient_negative_constraints, dict):
                problems.append("transient_negative_constraints must be a mapping or null")
            else:
                try:
                    from core.brain.llm.latent_cortex.transient_constraints import (
                        TransientConstraintConfig,
                    )

                    TransientConstraintConfig.from_value(self.transient_negative_constraints)
                except (TypeError, ValueError) as exc:
                    problems.append(str(exc))
        if self.virtual_quanta is not None:
            if not isinstance(self.virtual_quanta, dict):
                problems.append("virtual_quanta must be a mapping or null")
            else:
                try:
                    from core.brain.llm.latent_cortex.virtual_quanta import (
                        VirtualQuantaConfig,
                    )

                    VirtualQuantaConfig.from_value(self.virtual_quanta)
                except (TypeError, ValueError) as exc:
                    problems.append(str(exc))
        if self.latent_tree_search is not None:
            if not isinstance(self.latent_tree_search, dict):
                problems.append("latent_tree_search must be a mapping or null")
            else:
                try:
                    from core.brain.llm.latent_cortex.latent_tree_search import (
                        LatentTreeSearchConfig,
                    )

                    LatentTreeSearchConfig.from_value(self.latent_tree_search)
                except (TypeError, ValueError) as exc:
                    problems.append(str(exc))
        if self.verifier_fusion_evidence is not None:
            try:
                from core.brain.llm.latent_cortex.verifier_fusion import (
                    validate_verifier_fusion_evidence,
                )

                validate_verifier_fusion_evidence(self.verifier_fusion_evidence)
            except (TypeError, ValueError) as exc:
                problems.append(f"verifier_fusion_evidence invalid: {exc}")
        if self.escape is not None:
            if not isinstance(self.escape, dict):
                problems.append("escape must be a mapping or null")
            else:
                if type(self.escape.get("enabled", True)) is not bool:
                    problems.append("escape.enabled must be boolean")
                for key, low, high in (
                    ("stall_patience", 1, 32),
                    ("max_attempts", 0, 8),
                    ("probation_steps", 1, 16),
                ):
                    value = self.escape.get(key)
                    if value is not None and not integer_in(value, low, high):
                        problems.append(f"escape.{key} outside [{low}, {high}]")
                scale = self.escape.get("perturbation_scale")
                if scale is not None and (not finite(scale) or not 0.0 < float(scale) <= 0.5):
                    problems.append("escape.perturbation_scale outside (0, 0.5]")
                unknown = set(self.escape) - {
                    "enabled",
                    "stall_patience",
                    "max_attempts",
                    "probation_steps",
                    "perturbation_scale",
                    "min_improvement",
                }
                if unknown:
                    problems.append(f"escape has unknown keys: {sorted(unknown)}")
        return problems


@dataclass
class WeightIntegrityProof:
    """Digest evidence that resident weights survived an episode untouched.

    CP126 6e1ef7be. ``params_unchanged`` and ``fast_weights_erased`` were
    independent mutable booleans sitting beside the identity fields, with
    nothing in the schema relating them to any measurement. A receipt could
    assert that parameters were untouched and ephemeral weights erased while
    carrying no evidence whatsoever — and downstream gates, which use those
    booleans to decide whether the lane may keep serving without a reload,
    had no way to tell an attested claim from a default.

    A proof is a comparison, so this records both sides of it:

    * ``params_before`` / ``params_after`` — digests over the resident
      parameter set, taken before the episode and after teardown.
    * ``canary_before`` / ``canary_after`` — digests over the protected
      canary slice, which is what actually detects an incomplete erase: a
      fast-weight delta that was applied and not fully removed changes the
      canary even when a coarse parameter digest does not.
    * ``erased_layer_ids`` — which layers the teardown claims to have
      cleared, so the claim is enumerable rather than a bare True.

    The verdicts below return ``None`` when the evidence is absent. That is
    the whole point: callers must treat unknown as unproven and fail closed,
    rather than reading a default False/True as a measurement.
    """

    algorithm: str = "sha256"
    version: int = 1
    params_before: str = ""
    params_after: str = ""
    canary_before: str = ""
    canary_after: str = ""
    erased_layer_ids: list[str] = field(default_factory=list)
    # Why proof is missing, when it is. An empty reason with empty digests
    # means nobody even tried, which is itself worth seeing.
    unavailable_reason: str = ""

    @property
    def has_parameter_evidence(self) -> bool:
        return bool(self.params_before and self.params_after)

    @property
    def has_canary_evidence(self) -> bool:
        return bool(self.canary_before and self.canary_after)

    @property
    def params_unchanged_proven(self) -> bool | None:
        """True/False only when both digests exist; None means unproven."""
        if not self.has_parameter_evidence:
            return None
        return self.params_before == self.params_after

    @property
    def fast_weights_erased_proven(self) -> bool | None:
        """Erase is proven by the canary returning to its pre-episode digest.

        A parameter digest alone is too coarse: it can miss a small delta
        left behind in a single layer. The canary slice is chosen to move
        when the adapted function moves, so its return to baseline is the
        evidence that the adaptation is really gone.
        """
        if not self.has_canary_evidence:
            return None
        return self.canary_before == self.canary_after

    def to_dict(self) -> dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "version": self.version,
            "params_before": self.params_before,
            "params_after": self.params_after,
            "canary_before": self.canary_before,
            "canary_after": self.canary_after,
            "erased_layer_ids": list(self.erased_layer_ids),
            "unavailable_reason": self.unavailable_reason,
            "params_unchanged_proven": self.params_unchanged_proven,
            "fast_weights_erased_proven": self.fast_weights_erased_proven,
        }

    @classmethod
    def from_dict(cls, data: Any) -> WeightIntegrityProof:
        """Parse defensively: a malformed proof is NO proof, never a pass."""
        if not isinstance(data, dict):
            return cls(unavailable_reason="proof_not_a_mapping")
        raw_layers = data.get("erased_layer_ids")
        layers = [str(item) for item in raw_layers] if isinstance(raw_layers, (list, tuple)) else []
        try:
            version = int(data.get("version", 1))
        except (TypeError, ValueError):
            version = 0
        return cls(
            algorithm=str(data.get("algorithm") or ""),
            version=version,
            params_before=str(data.get("params_before") or ""),
            params_after=str(data.get("params_after") or ""),
            canary_before=str(data.get("canary_before") or ""),
            canary_after=str(data.get("canary_after") or ""),
            erased_layer_ids=layers,
            unavailable_reason=str(data.get("unavailable_reason") or ""),
        )


def _finite_record(value: Any) -> Any:
    """Strip non-finite sentinels from anything about to become a receipt.

    Internally the cortex uses +/-inf to mean "no verified score exists yet" --
    a fine sentinel for a comparison, and meaningless in a record. The causal
    receipt is canonicalized with ``allow_nan=False``, so one leaked sentinel
    raises inside receipt construction and destroys an episode that had
    already produced its answer.

    Sanitizing field by field is the wrong altitude: the sentinels arrive from
    branch scores, latent-optimization score trails, verifier arbitration, and
    anywhere else a comparison starts at an extremum. This is the single
    boundary all of them cross. Absence serializes as null, which is what the
    sentinel meant in the first place.

    None of this was reachable until a task verifier could be admitted, which
    is why the entire class survived every run this program has done.
    """
    if isinstance(value, bool) or isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _finite_record(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_record(v) for v in value]
    return value


@dataclass
class EpisodeReceipt:
    """Everything one reasoning episode actually did — the honesty record."""

    episode_id: str = ""
    domain: str = "general"
    started_at: float = field(default_factory=time.time)
    # Invariant proofs (governance.CheckpointInvariant fills these).
    checkpoint_fingerprint: str = ""
    checkpoint_fingerprint_method: str = ""
    checkpoint_file_count: int = 0
    worker_boot_id: str = ""
    worker_pid: int = 0
    worker_model_path: str = ""
    worker_model_parameter_count: int = 0
    worker_model_stored_parameter_element_count: int = 0
    worker_model_parameter_count_basis: str = ""
    worker_source_sha256: str = ""
    worker_affective_steering_active: bool = False
    worker_affective_steering_alpha: float = 0.0
    # Exact worker boot/model/serving-stack identity. The flattened fields
    # above remain compatibility telemetry; this object is what SPARK-056
    # binds into the measured runtime-integrity proof.
    worker_identity: dict[str, Any] = field(default_factory=dict)
    episode_affective_steering_applied: bool = False
    episode_affective_steering_alpha: float = 0.0
    request_payload_sha256: str = ""
    input_tokens_sha256: str = ""
    input_token_count: int = 0
    input_context_compaction: dict[str, Any] = field(default_factory=dict)
    # Typed cognitive ingress into the workspace itself: which slots were
    # seeded from which organ (memory/goals/world model/interoception/...),
    # so "the organs reached her thoughts" is receipted per slot and each
    # seeded slot remains individually ablation-testable (Experiment 3).
    cognitive_slots: list[dict[str, Any]] = field(default_factory=list)
    # Proof that prompt/evidence rows remained available and immutable while a
    # distinct hidden hypothesis persisted through every recurrent transition.
    recurrent_grounding: dict[str, Any] = field(default_factory=dict)
    # Public numerical evidence for fixed-anchor dynamics, finite states,
    # bounded KV positions, and the exact train/live update implementation.
    loop_stability: dict[str, Any] = field(default_factory=dict)
    # Every recurrent proposal's calibrated admission decision, including the
    # exact prior/proposal/admitted state commitments and learned-head identity.
    update_acceptance: dict[str, Any] = field(default_factory=dict)
    # Optional one-shot datastore observation admitted before recurrence.
    # Empty is backward-compatible; a populated receipt is independently
    # validated by the service and bound to its immutable evidence slot.
    nonparametric_memory: dict[str, Any] = field(default_factory=dict)
    # Service-admitted operation authority echoed by the worker. It binds the
    # exact epistemic state, controller decision, config, and budget to this
    # request without exposing private reasoning content.
    runtime_operation_authority: dict[str, Any] = field(default_factory=dict)
    # CP126 6e1ef7be. These remain for compatibility with every existing
    # reader, but they are no longer the AUTHORITY: weight_integrity below
    # carries the digests, and integrity_verdicts() reports what the
    # evidence actually supports. None means unproven, and consumers must
    # treat unproven as unsafe rather than as a passed check.
    params_unchanged: bool | None = None
    fast_weights_erased: bool | None = None
    # Digest evidence backing the two booleans above.
    weight_integrity: WeightIntegrityProof = field(default_factory=WeightIntegrityProof)
    # Unified measured proof over checkpoint, permanent canaries, exact
    # adapted layers, serving stack, fast-weight erase, caches, and worker.
    runtime_integrity: dict[str, Any] = field(default_factory=dict)
    # Topology actually used.
    n_layers: int = 0
    prelude_end: int = 0
    coda_start: int = 0
    n_slots: int = 0
    n_branches: int = 0
    schedule_hash: str = ""
    # Trajectory evidence.
    steps_taken: int = 0
    residual_trail: list[float] = field(default_factory=list)
    halting_reason: str = ""
    best_step: int = -1
    reverted_to_best: bool = False
    branch_scores: list[float] = field(default_factory=list)
    # Per-branch public-contract verdicts on the selection probes (CP180):
    # which branches' probe texts reached a complete/valid FINAL_ANSWER and
    # why the others did not — selection is auditable against the contract,
    # not just a scalar score. Empty when no verifier probes ran.
    branch_contract: list[dict[str, Any]] = field(default_factory=list)
    # Fresh-context representation repair for malformed private branch probes.
    # This can add a contract-valid candidate but owns no correctness or
    # public-answer authority.
    contract_repair: dict[str, Any] = field(default_factory=dict)
    # Hash-bound refresh of the selected candidate after accepted latent or
    # fast-weight adaptation. This prevents downstream arbitration from
    # grading the pre-adaptation probe as though it came from the final state.
    post_adaptation_candidate: dict[str, Any] = field(default_factory=dict)
    verifier_preflight: dict[str, Any] = field(default_factory=dict)
    blind_review: dict[str, Any] = field(default_factory=dict)
    decoy_verification: dict[str, Any] = field(default_factory=dict)
    # A resident-model derivation/falsification pass in a fresh KV context.
    # The public receipt discloses shared weights and only grants a bounded
    # refutation veto when deterministic witness evidence reconstructs.
    generative_verifier: dict[str, Any] = field(default_factory=dict)
    # Equal-score branch robustness under fresh, exact counterfactual changes.
    # The module can only tiebreak; it cannot outrank stronger correctness
    # evidence or claim independence from the shared resident checkpoint.
    counterfactual_verifier: dict[str, Any] = field(default_factory=dict)
    # Seed-isolated continuations from a deterministically verified prefix.
    # This is a recurrence diagnostic only; it cannot certify correctness or
    # influence the selected branch.
    prefix_stability: dict[str, Any] = field(default_factory=dict)
    # Historically calibrated, dependence-discounted verifier mesh. SPARK-046
    # is diagnostic only; later replacement policy must consume its confidence
    # bounds without promoting any individual probabilistic source.
    verifier_fusion: dict[str, Any] = field(default_factory=dict)
    critic_identity: dict[str, Any] = field(default_factory=dict)
    shared_blind_spots: dict[str, Any] = field(default_factory=dict)
    # Fresh-context virtual-width proof. Exact hidden-state contents stay
    # private; commitments and cache-discipline counters prove that every
    # candidate existed before cross-branch exposure.
    branch_isolation: dict[str, Any] = field(default_factory=dict)
    # Complete cache lineage across speculative windows, verifier probes,
    # branch savepoints/backtracks, regeneration, and accepted final lanes.
    # Tensors remain worker-private; the receipt carries salted immutable
    # storage commitments, offsets, and independently reconstructable hashes.
    kv_state_tree: dict[str, Any] = field(default_factory=dict)
    # Every cross-branch mailbox write: declared synchronization point,
    # candidate/role/operator provenance, bounded source slots, and causal
    # pre/post commitments. Later cooperative generations never create a new
    # independent vote.
    branch_exchange: dict[str, Any] = field(default_factory=dict)
    selected_branch: int = 0
    # ``selected_branch`` retains its integer wire shape for historical
    # receipts.  This bit distinguishes an admitted latent winner from the
    # dataclass default when the episode falls back before branch admission.
    branch_selection_admitted: bool = False
    exchanges: int = 0
    # Scoped durable-adapter activation. Zero calls means no recurrence-native
    # delta was resident; nonzero calls prove it was read only by slot windows.
    recurrence_adapter: dict[str, Any] = field(default_factory=dict)
    # Independently scoped interpretation tissue may fire only after an RLC
    # state is selected.  Keeping this separate prevents a decode-side coda
    # intervention from being misreported as recurrent-window execution.
    coda_adapter: dict[str, Any] = field(default_factory=dict)
    # Optimization evidence.
    # Digest of the first-decode logits (next-token distribution conditioned
    # on [prompt; refined thoughts]).
    #
    # CP126 16757b09. This was described as a universal causal audit — "any
    # change to the latent computation shows up here". That claim is too
    # strong and the field cannot support it: distinct latent states can
    # produce identical first-token logits (the decoder is not injective),
    # quantization and reduction order can collapse near-identical states to
    # the same bytes, and a digest that differs proves only that SOMETHING
    # differed, not what.
    #
    # What it honestly supports, one direction only:
    #   same digest  -> the first-decode distribution was indistinguishable
    #                   at this precision. NOT proof the latent path matched.
    #   different    -> the first-decode distribution genuinely differed.
    #
    # Establishing that a latent change was causal needs controlled
    # ablations plus later-token and output evidence. The digest is a cheap
    # screen, not a verdict, and it is only comparable across runs sharing
    # first_logits_digest_spec below.
    first_logits_digest: str = ""
    # Binding for the digest above: what was hashed and how. Digests
    # computed under different specs are NOT comparable, and comparing them
    # was previously possible because nothing recorded the difference.
    first_logits_digest_spec: dict[str, Any] = field(default_factory=dict)
    latent_opt_applied: bool = False
    latent_opt_mode: str = ""  # gradient | control | off
    latent_opt_loss_trail: list[float] = field(default_factory=list)
    latent_opt_attempts: int = 0
    latent_opt_steps: int = 0
    latent_opt_rejected: int = 0
    latent_opt_budget_exhausted: bool = False
    # Task-verifier arbitration over latent proposals: baseline provenance,
    # score/proxy trails, and why each proposal was accepted or rejected.
    latent_opt_verifier: dict[str, Any] = field(default_factory=dict)
    verifier_probe_max_tokens: int = 48
    verifier_probe_contract: str = "none"
    # True from the moment the first resident layer is touched. It says the
    # model MAY be dirty; fast_weights_applied says the attach completed.
    # A partial attach sets the first and not the second, and that gap is
    # exactly the state a vanilla fallback must refuse to decode against.
    fast_weights_attach_attempted: bool = False
    fast_weights_applied: bool = False
    fast_weights_layers: int = 0
    fast_weight_optimization_attempts: int = 0
    fast_weight_optimized_steps: int = 0
    fast_weight_rejected_steps: int = 0
    fast_weight_budget_exhausted: bool = False
    fast_weight_optimizer: str = ""
    fast_weight_loss_trail: list[float] = field(default_factory=list)
    fast_weight_gradient_norm_trail: list[float] = field(default_factory=list)
    fast_weight_accepted_step_sizes: list[float] = field(default_factory=list)
    fast_weight_line_search_backtracks: int = 0
    # Protected-behavior canary evidence: what the adapted function did to
    # the protected battery and what the ladder decided (accepted /
    # rescaled / erased). Empty when canaries did not run.
    fast_weight_canaries: dict[str, Any] = field(default_factory=dict)
    # Task-verifier arbitration over the adapted function: the verifier
    # scores a decoded probe before and after ΔW optimization and erases
    # the adaptation on regression — the verifier, not the proxy, has the
    # last word over fast weights too. Empty when arbitration did not run.
    fast_weight_verifier: dict[str, Any] = field(default_factory=dict)
    # `ok` on the result says the machinery ran and produced text. It has
    # never said the answer was checked, or that the recurrence helped — the
    # task verifier is optional, branch selection can fall back to the
    # ensemble's internal score, and latent optimization can descend on a
    # proxy. Downstream code that wanted "this episode improved reasoning"
    # had only `ok` to read, so these two say what `ok` does not.
    # What backs the recurrence on THIS checkpoint: a registered architecture
    # with paired evidence, a declared positional contract and nothing more,
    # or neither. Structural access proves the call succeeds; it has never
    # proved the repetition means anything.
    recurrence_support: dict[str, Any] = field(default_factory=dict)
    # Whether the prompt cache the adapted probes and the final decode read was
    # filled under the same weights they run under. Empty when no fast weights
    # attached.
    fast_weight_cache_attestation: dict[str, Any] = field(default_factory=dict)
    quality_verified: bool = False
    gain_established: bool = False
    verifier_identity: str = ""
    # Same learned U,V evaluated under none/recurrent/decode/both scopes.
    fast_weight_locality: dict[str, Any] = field(default_factory=dict)
    # Complete SPARK-055 contract: exact evidence admission, exclusive model
    # lease, measured identity-at-attach, matched causal probe, answer binding,
    # and exact cleanup. Public commitments only; no latent values or evidence
    # text are copied into the receipt.
    fast_weight_learning: dict[str, Any] = field(default_factory=dict)
    # Independently verified private derivation encoded into the actual
    # recurrent workspace, compared with a norm-matched semantic sham, and
    # retained only when the neural decode strictly improves.
    verified_workspace_evidence: dict[str, Any] = field(default_factory=dict)
    # Cleanup is a separate transaction proof so optimizer/attach failures do
    # not discard the measured erase evidence needed to decide whether the
    # resident worker may safely continue or run a vanilla fallback.
    fast_weight_cleanup: dict[str, Any] = field(default_factory=dict)
    # Decode completeness. Contract-required tasks separately receipt whether
    # generated text actually satisfied the terminal answer contract.
    decode_requested_tokens: int = 0
    decode_generated_tokens: int = 0
    decode_termination: str = "not_started"
    decode_contract_required: bool = False
    decode_contract_satisfied: bool = False
    decode_contract_grace_tokens: int = 0
    decode_contract_grace_used_tokens: int = 0
    decode_incumbent_policy: str = "latent"
    decode_incumbent_prompt_logits_sha256: str = ""
    # Exact ordinary-decode artifact retained by the RLC output floor. Empty
    # means the episode regenerated its incumbent internally and therefore
    # cannot claim byte identity with a separately measured control.
    incumbent_artifact: dict[str, Any] = field(default_factory=dict)
    # Times the decode sampler masked a pure-newline token because the run
    # already held _MAX_NEWLINE_RUN — a sampling constraint, never text
    # editing; nonzero values reveal the model still trying to babble.
    decode_newline_suppressions: int = 0
    # -1 means the decode was deterministic and no seed was in play. Any other
    # value reproduces the sampling exactly; the trace digest commits to the
    # sequence of decisions that seed actually produced.
    decode_sample_seed: int = -1
    decode_sample_trace_sha256: str = ""
    decode_repetition_penalty_applied: float = 1.0
    # Deterministic task-verifier evidence when the episode ran under
    # verifier guidance (task_verifiers.EpisodeTaskVerifier receipt).
    verifier_guidance: dict[str, Any] = field(default_factory=dict)
    # Attractor-escape evidence: per-branch ladder receipts (rungs tried,
    # triggers, probation outcomes). Empty when no branch needed escape.
    escape: dict[str, Any] = field(default_factory=dict)
    # Halting-policy evidence: mode, per-branch head halts, and whether the
    # learned head actually determined any stop (head_was_causal) — a
    # learned run whose every stop came from the residual floor is the old
    # policy under a new name, and the receipt must say so.
    halting: dict[str, Any] = field(default_factory=dict)
    # One strict terminal taxonomy above every low-level halt mechanism. It
    # distinguishes convergence, non-positive value, budget exhaustion, and
    # irreducible uncertainty, then binds the disposition into model-generated
    # user language instead of selecting a canned response.
    terminal_disposition: dict[str, Any] = field(default_factory=dict)
    # Ordered, hash-linked public commitments spanning the complete episode.
    # The DAG carries no latent values or private reasoning text.
    causal_receipt: dict[str, Any] = field(default_factory=dict)
    # Confidence-bound, branch-local best-state promotions and preservations.
    # Empty/default traces mean no verifier earned state-selection authority.
    verified_best_state: dict[str, Any] = field(default_factory=dict)
    # Verified failed transitions may create one branch/action-scoped,
    # episode-local negative latent direction. Critic prose has no authority;
    # admission requires repeated equal-compute verifier wins over no-op and
    # orthogonal-sham controls, and every admitted constraint expires or is
    # consumed exactly once.
    transient_negative_constraints: dict[str, Any] = field(default_factory=dict)
    # One episode-local latent quantum may be admitted only after real
    # no-op/matched-random/guided probes prove a bounded contribution. The
    # private direction is zeroized after its one use.
    virtual_quanta: dict[str, Any] = field(default_factory=dict)
    # Bounded MCTS/beam/BFS evidence over exact recurrent and KV snapshots.
    latent_tree_search: dict[str, Any] = field(default_factory=dict)
    # Objective hidden-state correctness probability and predictive entropy.
    # Unavailable mode is explicit and emits no observations.
    neural_uncertainty: dict[str, Any] = field(default_factory=dict)
    # Admitted transition-level localization evidence. It remains diagnostic
    # until a separately proved repair-steering milestone grants authority.
    mistake_locator: dict[str, Any] = field(default_factory=dict)
    # Full-sequence, hidden-trace-only premise/conclusion reflection. This
    # critic is non-causal in context access and read-only in authority.
    bidirectional_reflector: dict[str, Any] = field(default_factory=dict)
    # Calibrated transition-by-latent-position contradiction evidence over
    # the reflected trace. It remains diagnostic by itself.
    contradiction_tensor: dict[str, Any] = field(default_factory=dict)
    # Counterbalanced no-op/matched-random/guided intervention evidence.
    # Only an authoritative, repeat-stable, equal-compute win may alter the
    # selected branch; every other evaluated path restores the exact baseline.
    contradiction_perturbation: dict[str, Any] = field(default_factory=dict)
    # Entropy-conditioned source-bound candidate family against no-op and
    # stable-position sham controls. Stable positions have no retained write
    # authority; any unproven search restores the exact baseline.
    local_exploration: dict[str, Any] = field(default_factory=dict)
    # Equal-compute comparison of incumbent selection, corrected selection,
    # and per-token probability fusion. Overlapping evidence abstains.
    heterogeneous_integration: dict[str, Any] = field(default_factory=dict)
    # Final user-visible decode commitment for the integration policy. Fusion
    # carries dual-lane trace/accounting evidence; selection carries no text.
    heterogeneous_decode: dict[str, Any] = field(default_factory=dict)
    # Neural-bytecode trace: one event per non-window instruction the
    # schedule program executed (exchange/savepoint/verify_probe outcomes,
    # probe scores, backtracks). Empty for plain window programs.
    bytecode_events: list[dict[str, Any]] = field(default_factory=list)
    # Per-recurrence cognitive-operator decisions. This contains only public
    # scalar state signals, measured progress/cost, and action receipts; it
    # never contains private reasoning text or hidden-state tensors.
    value_of_computation: dict[str, Any] = field(default_factory=dict)
    cognitive_action_trace: list[dict[str, Any]] = field(default_factory=list)
    cognitive_operator_trace: list[dict[str, Any]] = field(default_factory=list)
    # Source-selective causal writes for SEARCH_MEMORY and RETRIEVE_EVIDENCE.
    # External re-fetch remains a separately governed service operation.
    context_focus_trace: list[dict[str, Any]] = field(default_factory=list)
    # Parameter-free, digest-bound request for the host ActionExecutor to
    # perform an already-Will-admitted effect. The worker never owns effect
    # authority or raw action parameters.
    external_execution_handoff: dict[str, Any] = field(default_factory=dict)
    # Wording-independent structural support classes reconstructed from the
    # primary action/operator/isolation traces. Different prose never creates
    # another vote; causal structure has to differ across six named facets.
    structural_diversity: dict[str, Any] = field(default_factory=dict)
    # Pairwise exact-prefix graph over causal operator programs and, when
    # decoded branch probes exist, their hash-bound claims/dependencies.
    # Diagnostic only: later policy chooses what operation resolves a dispute.
    disagreement_graph: dict[str, Any] = field(default_factory=dict)
    #: What this episode irreversibly COMMITTED, and the coverage that
    #: bought. Under i.i.d. branch sampling most branches re-derive the same
    #: answer, so best-of-N is best-of-2 and the receipt never said so; the
    #: ratchet's receipt carries distinct coverage, the exclusions that
    #: produced it, and whether any narrowing was actually measured.
    commitment_ratchet: dict[str, Any] = field(default_factory=dict)
    # Cheapest available diagnostic selected for each localized disagreement,
    # bound to deterministic routes and measured/declared action costs.
    diagnostic_action_selection: dict[str, Any] = field(default_factory=dict)
    # Source-private regeneration from an exactly refuted atom. Original
    # branch commitments remain immutable.
    local_repair: dict[str, Any] = field(default_factory=dict)
    # Independently reconstructable lower-bound > upper-bound authority for
    # promoting a repaired candidate into the user-visible output.
    answer_replacement: dict[str, Any] = field(default_factory=dict)
    # Hidden-ground-truth arbitration used only by the explicitly diagnostic
    # oracle research arm. It can alter that arm's measured output but carries
    # no live serving or capability-claim authority.
    research_oracle_arbitration: dict[str, Any] = field(default_factory=dict)
    correlated_support: dict[str, Any] = field(default_factory=dict)
    # Latent interpretability/safety telemetry (telemetry.LatentTelemetry).
    latent_telemetry: dict[str, Any] = field(default_factory=dict)
    # Decode-probe memoization evidence (probe_cache.DecodeProbeCache).
    probe_cache: dict[str, Any] = field(default_factory=dict)
    decode_temperature: float = 0.0
    decode_top_p: float = 1.0
    decode_bridge_applied: bool = False
    decode_bridge_policy: str = "none"
    decode_bridge_token_count: int = 0
    decode_bridge_tokens_sha256: str = ""
    decode_bridge_logits_digest: str = ""
    # Everything the model reads between the latent episode and the answer,
    # attributed to its sources. ``decode_bridge_*`` above answers for the
    # configured bridge policy only; terminal-disposition language is a
    # separate injection and is counted separately here. An arm comparing
    # itself to an ordinary decode must read this, not the bridge fields.
    decode_prefix_token_count: int = 0
    decode_prefix_composition: dict[str, Any] = field(default_factory=dict)
    output_quality: dict[str, Any] = field(default_factory=dict)
    # Runtime lifecycle evidence. Timings are stage-local wall-clock seconds;
    # progress messages use the same stage names so a parent can distinguish a
    # slow live episode from a wedged worker without peeking into model state.
    last_stage: str = "not_started"
    stage_timings_s: dict[str, float] = field(default_factory=dict)
    # Economy.
    budget: dict[str, Any] = field(default_factory=dict)
    # Honesty flags: anything a consumer must know before trusting the output
    # ("diverged_reverted", "budget_exhausted", "fallback_vanilla", ...).
    honest_flags: list[str] = field(default_factory=list)

    def flag(self, name: str) -> None:
        if name not in self.honest_flags:
            self.honest_flags.append(name)

    def has_flag(self, name: str) -> bool:
        """True when this exact flag was raised. Prefixed flags carry a
        detail suffix, so a caller wanting the family passes the prefix and
        gets a prefix match."""

        return any(
            flag == name or flag.startswith(f"{name}:") for flag in self.honest_flags
        )

    def integrity_verdicts(self) -> dict[str, Any]:
        """What the EVIDENCE supports about weight integrity, not what was asserted.

        CP126 6e1ef7be. Each verdict is one of:

        * ``proven`` — digests exist and agree;
        * ``refuted`` — digests exist and disagree (the claim is false);
        * ``unproven`` — no digests, so nothing is established.

        ``asserted`` reports the legacy boolean beside the verdict, so a
        receipt makes disagreement between claim and evidence visible rather
        than letting the boolean stand in for a measurement. A caller that
        needs integrity must require ``proven`` — treating ``unproven`` as
        acceptable is the exact fail-open this finding names.
        """

        def _verdict(proven: bool | None) -> str:
            if proven is None:
                return "unproven"
            return "proven" if proven else "refuted"

        proof = self.weight_integrity
        params_proven = proof.params_unchanged_proven
        erased_proven = proof.fast_weights_erased_proven
        runtime_reason = ""
        if self.runtime_integrity:
            try:
                from core.brain.llm.latent_cortex.runtime_integrity import (
                    validate_runtime_integrity_receipt,
                )

                runtime = validate_runtime_integrity_receipt(
                    self.runtime_integrity,
                    require_worker=False,
                    expected_episode_id=self.episode_id,
                    expected_input_tokens_sha256=self.input_tokens_sha256,
                    expected_fast_weights_applied=self.fast_weights_applied,
                    expected_fast_weights_attach_attempted=(
                        self.fast_weights_attach_attempted
                    ),
                    expected_checkpoint_fingerprint=(
                        self.checkpoint_fingerprint
                    ),
                    expected_checkpoint_method=(
                        self.checkpoint_fingerprint_method
                    ),
                    expected_checkpoint_file_count=(
                        self.checkpoint_file_count
                    ),
                )
                params_proven = bool(
                    runtime["parameters"]["unchanged"]
                    and runtime["adapted_layers"]["unchanged"]
                    and runtime["serving_stack"]["unchanged"]
                )
                erased_proven = bool(
                    runtime["fast_weight_erase"]["exact"]
                    and runtime["cache"]["safe"]
                )
            except (ImportError, TypeError, ValueError) as exc:
                params_proven = None
                erased_proven = None
                runtime_reason = f"runtime_integrity_invalid:{type(exc).__name__}"
        params_verdict = _verdict(params_proven)
        erased_verdict = _verdict(erased_proven)
        verdicts = {
            "params_unchanged": {
                "verdict": params_verdict,
                "asserted": self.params_unchanged,
            },
            "fast_weights_erased": {
                "verdict": erased_verdict,
                "asserted": self.fast_weights_erased,
            },
            "algorithm": proof.algorithm,
            "version": proof.version,
            "unavailable_reason": runtime_reason or proof.unavailable_reason,
        }
        # A claim contradicted by its own evidence is the case worth
        # shouting about, so it is named rather than left to be inferred by
        # comparing two fields.
        contradictions: list[str] = []
        if params_verdict == "refuted" and self.params_unchanged is True:
            contradictions.append("params_unchanged_asserted_but_refuted")
        if erased_verdict == "refuted" and self.fast_weights_erased is True:
            contradictions.append("fast_weights_erased_asserted_but_refuted")
        verdicts["contradictions"] = contradictions
        return verdicts

    def integrity_is_proven(self) -> bool:
        """True only when BOTH integrity claims are backed by agreeing digests.

        Deliberately strict: this is the predicate a lane should consult
        before continuing to serve on weights an episode touched.
        """
        verdicts = self.integrity_verdicts()
        return (
            verdicts["params_unchanged"]["verdict"] == "proven"
            and verdicts["fast_weights_erased"]["verdict"] == "proven"
        )

    def to_dict(self) -> dict[str, Any]:
        return _finite_record(self._to_dict_raw())

    def _to_dict_raw(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "domain": self.domain,
            "started_at": self.started_at,
            "checkpoint_fingerprint": self.checkpoint_fingerprint,
            "checkpoint_fingerprint_method": self.checkpoint_fingerprint_method,
            "checkpoint_file_count": self.checkpoint_file_count,
            "worker_boot_id": self.worker_boot_id,
            "worker_pid": self.worker_pid,
            "worker_model_path": self.worker_model_path,
            "worker_model_parameter_count": self.worker_model_parameter_count,
            "worker_model_stored_parameter_element_count": (
                self.worker_model_stored_parameter_element_count
            ),
            "worker_model_parameter_count_basis": (self.worker_model_parameter_count_basis),
            "worker_source_sha256": self.worker_source_sha256,
            "worker_affective_steering_active": self.worker_affective_steering_active,
            "worker_affective_steering_alpha": self.worker_affective_steering_alpha,
            "worker_identity": dict(self.worker_identity),
            "episode_affective_steering_applied": self.episode_affective_steering_applied,
            "episode_affective_steering_alpha": self.episode_affective_steering_alpha,
            "request_payload_sha256": self.request_payload_sha256,
            "input_tokens_sha256": self.input_tokens_sha256,
            "input_token_count": self.input_token_count,
            "input_context_compaction": dict(self.input_context_compaction),
            "cognitive_slots": [dict(row) for row in self.cognitive_slots],
            "recurrent_grounding": dict(self.recurrent_grounding),
            "loop_stability": dict(self.loop_stability),
            "update_acceptance": dict(self.update_acceptance),
            "nonparametric_memory": dict(self.nonparametric_memory),
            "runtime_operation_authority": dict(self.runtime_operation_authority),
            "params_unchanged": self.params_unchanged,
            "fast_weights_erased": self.fast_weights_erased,
            "weight_integrity": self.weight_integrity.to_dict(),
            "runtime_integrity": dict(self.runtime_integrity),
            "integrity_verdicts": self.integrity_verdicts(),
            "n_layers": self.n_layers,
            "prelude_end": self.prelude_end,
            "coda_start": self.coda_start,
            "n_slots": self.n_slots,
            "n_branches": self.n_branches,
            "schedule_hash": self.schedule_hash,
            "steps_taken": self.steps_taken,
            "residual_trail": [round(r, 6) for r in self.residual_trail],
            "halting_reason": self.halting_reason,
            "first_logits_digest": self.first_logits_digest,
            "first_logits_digest_spec": dict(self.first_logits_digest_spec),
            "best_step": self.best_step,
            "reverted_to_best": self.reverted_to_best,
            # An unscored branch carries the -inf sentinel internally, which
            # says no verified score exists -- not a score of negative
            # infinity. Consumers keep the float; the SERIALIZED receipt must
            # say "unscored", because it is canonicalized with allow_nan=False
            # and the sentinel would otherwise raise during receipt
            # construction and destroy an episode that already had its answer.
            # Only reachable once a task verifier is admitted, which is why no
            # prior run in this program ever hit it.
            "branch_scores": [
                None if not math.isfinite(float(s)) else round(float(s), 6)
                for s in self.branch_scores
            ],
            "branch_contract": [dict(row) for row in self.branch_contract],
            "contract_repair": dict(self.contract_repair),
            "verifier_preflight": dict(self.verifier_preflight),
            "blind_review": dict(self.blind_review),
            "decoy_verification": dict(self.decoy_verification),
            "generative_verifier": dict(self.generative_verifier),
            "counterfactual_verifier": dict(self.counterfactual_verifier),
            "prefix_stability": dict(self.prefix_stability),
            "verifier_fusion": dict(self.verifier_fusion),
            "critic_identity": dict(self.critic_identity),
            "shared_blind_spots": dict(self.shared_blind_spots),
            "branch_isolation": dict(self.branch_isolation),
            "kv_state_tree": dict(self.kv_state_tree),
            "branch_exchange": dict(self.branch_exchange),
            "selected_branch": self.selected_branch,
            "branch_selection_admitted": self.branch_selection_admitted,
            "exchanges": self.exchanges,
            "recurrence_adapter": dict(self.recurrence_adapter),
            "coda_adapter": dict(self.coda_adapter),
            "latent_opt_applied": self.latent_opt_applied,
            "latent_opt_mode": self.latent_opt_mode,
            "latent_opt_loss_trail": [round(v, 6) for v in self.latent_opt_loss_trail],
            "latent_opt_attempts": self.latent_opt_attempts,
            "latent_opt_steps": self.latent_opt_steps,
            "latent_opt_rejected": self.latent_opt_rejected,
            "latent_opt_budget_exhausted": self.latent_opt_budget_exhausted,
            "latent_opt_verifier": dict(self.latent_opt_verifier),
            "post_adaptation_candidate": dict(
                self.post_adaptation_candidate
            ),
            "verifier_probe_max_tokens": self.verifier_probe_max_tokens,
            "verifier_probe_contract": self.verifier_probe_contract,
            "fast_weights_attach_attempted": (
                self.fast_weights_attach_attempted
            ),
            "fast_weights_applied": self.fast_weights_applied,
            "fast_weights_layers": self.fast_weights_layers,
            "fast_weight_optimization_attempts": self.fast_weight_optimization_attempts,
            "fast_weight_optimized_steps": self.fast_weight_optimized_steps,
            "fast_weight_rejected_steps": self.fast_weight_rejected_steps,
            "fast_weight_budget_exhausted": self.fast_weight_budget_exhausted,
            "fast_weight_optimizer": self.fast_weight_optimizer,
            "fast_weight_loss_trail": [round(v, 6) for v in self.fast_weight_loss_trail],
            "fast_weight_gradient_norm_trail": [
                round(v, 6) for v in self.fast_weight_gradient_norm_trail
            ],
            "fast_weight_accepted_step_sizes": [
                round(v, 12) for v in self.fast_weight_accepted_step_sizes
            ],
            "fast_weight_line_search_backtracks": (self.fast_weight_line_search_backtracks),
            "fast_weight_canaries": dict(self.fast_weight_canaries),
            "fast_weight_verifier": dict(self.fast_weight_verifier),
            "recurrence_support": dict(self.recurrence_support),
            "fast_weight_cache_attestation": dict(
                self.fast_weight_cache_attestation
            ),
            "quality_verified": self.quality_verified,
            "gain_established": self.gain_established,
            "verifier_identity": self.verifier_identity,
            "fast_weight_locality": dict(self.fast_weight_locality),
            "fast_weight_learning": dict(self.fast_weight_learning),
            "verified_workspace_evidence": dict(
                self.verified_workspace_evidence
            ),
            "fast_weight_cleanup": dict(self.fast_weight_cleanup),
            "decode_requested_tokens": self.decode_requested_tokens,
            "decode_generated_tokens": self.decode_generated_tokens,
            "decode_termination": self.decode_termination,
            "decode_contract_required": self.decode_contract_required,
            "decode_contract_satisfied": self.decode_contract_satisfied,
            "decode_contract_grace_tokens": self.decode_contract_grace_tokens,
            "decode_contract_grace_used_tokens": (self.decode_contract_grace_used_tokens),
            "decode_incumbent_policy": self.decode_incumbent_policy,
            "decode_incumbent_prompt_logits_sha256": (
                self.decode_incumbent_prompt_logits_sha256
            ),
            "incumbent_artifact": dict(self.incumbent_artifact),
            "decode_newline_suppressions": self.decode_newline_suppressions,
            "decode_sample_seed": self.decode_sample_seed,
            "decode_sample_trace_sha256": self.decode_sample_trace_sha256,
            "decode_repetition_penalty_applied": self.decode_repetition_penalty_applied,
            "verifier_guidance": dict(self.verifier_guidance),
            "escape": dict(self.escape),
            "halting": dict(self.halting),
            "terminal_disposition": dict(self.terminal_disposition),
            "causal_receipt": dict(self.causal_receipt),
            "verified_best_state": dict(self.verified_best_state),
            "transient_negative_constraints": dict(self.transient_negative_constraints),
            "virtual_quanta": dict(self.virtual_quanta),
            "latent_tree_search": dict(self.latent_tree_search),
            "neural_uncertainty": dict(self.neural_uncertainty),
            "mistake_locator": dict(self.mistake_locator),
            "bidirectional_reflector": dict(self.bidirectional_reflector),
            "contradiction_tensor": dict(self.contradiction_tensor),
            "contradiction_perturbation": dict(self.contradiction_perturbation),
            "local_exploration": dict(self.local_exploration),
            "heterogeneous_integration": dict(self.heterogeneous_integration),
            "heterogeneous_decode": dict(self.heterogeneous_decode),
            "bytecode_events": [dict(row) for row in self.bytecode_events],
            "value_of_computation": dict(self.value_of_computation),
            "cognitive_action_trace": [dict(row) for row in self.cognitive_action_trace],
            "cognitive_operator_trace": [dict(row) for row in self.cognitive_operator_trace],
            "context_focus_trace": [dict(row) for row in self.context_focus_trace],
            "external_execution_handoff": dict(self.external_execution_handoff),
            "structural_diversity": dict(self.structural_diversity),
            "disagreement_graph": dict(self.disagreement_graph),
            "commitment_ratchet": dict(self.commitment_ratchet),
            "diagnostic_action_selection": dict(self.diagnostic_action_selection),
            "local_repair": dict(self.local_repair),
            "answer_replacement": dict(self.answer_replacement),
            "research_oracle_arbitration": dict(
                self.research_oracle_arbitration
            ),
            "correlated_support": dict(self.correlated_support),
            "latent_telemetry": dict(self.latent_telemetry),
            "probe_cache": dict(self.probe_cache),
            "decode_temperature": self.decode_temperature,
            "decode_top_p": self.decode_top_p,
            "decode_bridge_applied": self.decode_bridge_applied,
            "decode_bridge_policy": self.decode_bridge_policy,
            "decode_bridge_token_count": self.decode_bridge_token_count,
            "decode_bridge_tokens_sha256": self.decode_bridge_tokens_sha256,
            "decode_bridge_logits_digest": self.decode_bridge_logits_digest,
            "decode_prefix_token_count": self.decode_prefix_token_count,
            "decode_prefix_composition": dict(self.decode_prefix_composition),
            "output_quality": dict(self.output_quality),
            "last_stage": self.last_stage,
            "stage_timings_s": {
                str(name): round(float(seconds), 6)
                for name, seconds in self.stage_timings_s.items()
            },
            "budget": dict(self.budget),
            "honest_flags": list(self.honest_flags),
        }


@dataclass
class LatentReasoningResult:
    """What the engine returns to the worker/caller."""

    ok: bool
    text: str
    receipt: EpisodeReceipt
    reason: str = ""  # populated when ok is False or a fallback occurred
    # Raw output token ids — substrate-level callers (the experiments harness
    # driving random-weight models with synthetic vocabularies) verify these.
    tokens: list[int] = field(default_factory=list)
    # Opt-in behavior-policy trace for recurrence-native training. Empty for
    # every ordinary live request; callers must explicitly request capture.
    decode_token_logprobs: list[float] = field(default_factory=list)
    # Internal worker-to-service evidence used to rerun answer-replacement
    # verification outside the model worker. The service removes it before
    # returning any result to product callers.
    answer_replacement_private: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "text": self.text,
            "reason": self.reason,
            "tokens": list(self.tokens),
            "decode_token_logprobs": list(self.decode_token_logprobs),
            "answer_replacement_private": dict(
                self.answer_replacement_private
            ),
            "receipt": self.receipt.to_dict(),
        }


__all__ = [
    "ABSOLUTE_MAX_BRANCHES",
    "ABSOLUTE_MAX_LAYER_APPS",
    "ABSOLUTE_MAX_RECURRENT_STEPS",
    "ABSOLUTE_MAX_SLOTS",
    "ABSOLUTE_MAX_WALL_CLOCK_S",
    "BranchConfig",
    "ComputeBudget",
    "CortexConfig",
    "DEFAULT_EPISODE_LAYER_APPS",
    "EpisodeReceipt",
    "FastWeightsConfig",
    "LatentOptConfig",
    "LatentReasoningResult",
    "RecurrenceConfig",
    "WorkspaceConfig",
]
