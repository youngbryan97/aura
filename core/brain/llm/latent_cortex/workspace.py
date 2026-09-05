"""The latent workspace: M writable continuous thought slots.

Slots are real sequence positions appended after the prompt. They are not
words — they are writable internal state, refined by recurrent computation
and finally persisted into the KV cache so the decoded answer attends to
them at every layer. That persistence is the causality contract: ablating a
slot (zeroing its K/V) measurably changes the answer, and Experiment 3
verifies exactly that.

Role anchors give branches/slots distinct starting basins (constructor,
counterexample-hunter, checker, ...) without any trained parameters: they are
deterministic unit-scale directions derived from the role name, so runs are
reproducible and roles are causally testable rather than decorative.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Any

from core.runtime.errors import record_degradation
from core.brain.llm.latent_cortex.types import WorkspaceConfig

logger = logging.getLogger("Aura.LatentCortex.Workspace")


def _role_seed(role: str, base_seed: int) -> int:
    """Deterministic, platform-stable seed for a role anchor."""
    digest = hashlib.sha256(f"{role}:{base_seed}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def role_anchor(role: str, dim: int, base_seed: int = 0):
    """A deterministic direction in hidden space for a named cognitive role."""
    import mlx.core as mx

    key = mx.random.key(_role_seed(role, base_seed))
    vec = mx.random.normal((dim,), key=key)
    return vec / mx.maximum(mx.linalg.norm(vec), 1e-6)


def _validated_context_seed(seed: Any) -> tuple[int, str, Any]:
    """A seed states which context item it came from, or it is not a seed.

    The two-element form let the caller's position be reconstructed from the
    row count downstream, which is wrong the moment any item is skipped.
    """

    if not isinstance(seed, tuple) or len(seed) != 3:
        raise TypeError(
            "a context seed is (context_index, source, vector); the "
            "two-element form cannot say which item it came from"
        )
    index, source, vector = seed
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError("a context seed index must be a non-negative integer")
    return int(index), str(source), vector


def per_position_rms(x):
    """Per-position RMS over the hidden dimension: (..., L, D) → (..., L, 1).

    The accumulation runs in float32 regardless of input dtype: real Qwen
    activations carry outlier channels whose SQUARE overflows fp16's 65504
    ceiling, which surfaced as NaN the first time the recurrence-native
    objective ran the shared trust-band math over full-sequence states. The
    result is cast back to the input dtype, so downstream math is unchanged
    whenever the fp16 computation would have been finite anyway.
    """
    import mlx.core as mx

    return mx.sqrt(
        mx.mean(mx.square(x.astype(mx.float32)), axis=-1, keepdims=True)
    ).astype(x.dtype)


@dataclass
class SlotAblation:
    """Record of one applied ablation, so restores are exact and auditable."""

    slot_index: int
    mode: str
    prior_state: Any = None


class LatentWorkspace:
    """M thought slots in hidden space, with snapshot/ablate/readout support.

    Holds the slot tensor ``z`` of shape (1, M, D) plus provenance. The
    workspace itself never touches the KV cache — the engine owns cache
    discipline. This separation keeps the workspace trivially testable.
    """

    def __init__(
        self,
        z,
        roles: list[str],
        config: WorkspaceConfig,
        *,
        context_slots: list[dict[str, Any]] | None = None,
        context_admission: dict[str, Any] | None = None,
    ) -> None:
        self.z = z
        self.roles = list(roles)
        self.config = config
        self.seed_z = z  # immutable reference state for drift measurement
        # Which slots were seeded from typed cognitive context (organ → slot),
        # in receipt form: [{"slot": int, "source": str}].
        self.context_slots = list(context_slots or [])
        # How much of the requested cognitive context actually reached the
        # workspace. The slot cap is structural (comm slot + one private
        # hypothesis slot), so dropping is legitimate — reasoning on two of
        # five memories looking identical to reasoning on all five is not.
        self.context_admission = dict(context_admission or {
            "schema": "aura.workspace_context_admission.v1",
            "requested": 0, "admitted": 0, "dropped": 0,
            "dropped_sources": [], "n_slots": int(config.n_slots),
            "complete": True,
        })
        # Sealed after the prelude pass, when the evidence rows have reached the
        # same layer representation as the recurrent core expects. Recurrent
        # proposals may read these rows but may not replace them.
        self._context_evidence_anchor = None
        self._ablations: list[SlotAblation] = []

    # ── Construction ────────────────────────────────────────────────────
    @classmethod
    def from_prompt_embeddings(
        cls,
        prompt_embeddings,
        config: WorkspaceConfig,
        *,
        branch_role: str | None = None,
        context_seeds: list[tuple[str, Any]] | None = None,
    ) -> LatentWorkspace:
        """Seed M slots from the pooled prompt embedding + role anchors.

        Each slot starts at the prompt's mean embedding, perturbed along its
        role-anchor direction, then RMS-matched to the embedding distribution
        so the first prelude pass sees in-manifold inputs. ``branch_role``
        additionally rotates every anchor seed, giving branches distinct
        starting basins over identical weights.

        ``context_seeds`` is the typed cognitive ingress into thought itself:
        (source, embedding) pairs from the organs (memory recall, active goal,
        world model, interoception, self-model). Evidence occupies a causal
        prefix immediately after the communication slot. Every later hypothesis
        slot can therefore attend to every evidence slot under an ordinary
        decoder-only causal mask on every recurrent pass. At least one private
        hypothesis slot remains after the evidence prefix.
        """
        import mlx.core as mx

        m = int(config.n_slots)
        dim = int(prompt_embeddings.shape[-1])
        pooled = mx.mean(prompt_embeddings, axis=1, keepdims=True)  # (1,1,D)
        target_rms = mx.mean(per_position_rms(prompt_embeddings))
        # Span the prompt instead of averaging it.
        #
        # Seeding every slot from the SAME global mean left the workspace with
        # an effective rank of one: measured slot-to-slot cosine at seed was
        # 0.9993 (min 0.9992, max 0.9993) against 0.0419 for the prompt's own
        # token embeddings. Sixteen slots carried one direction sixteen times,
        # differentiated only by a 5% role anchor. A recurrent operator cannot
        # pull apart states that begin 99.93% aligned, which is exactly the
        # long-standing cos(pass1, pass2) = 0.9994 obstacle -- the same number,
        # arriving from the seed rather than from the recurrence.
        #
        # Mean-pooling also destroys order: for "start=17, apply +13 then -6
        # mod 19" the centroid averages away which number is the start and
        # which operations follow, so the decode attends to sixteen copies of a
        # bag-of-words gist. Topic without specifics is precisely the observed
        # failure -- fluent, committed, wrong.
        #
        # Each slot now pools a DISJOINT span of the prompt. A mean of token
        # embeddings stays inside their convex hull, so every seed remains in
        # the embedding manifold the frozen layers were trained on, while the
        # slots differ because their spans differ. Order survives as position
        # across slots. Short prompts fall back to the global mean, which is
        # the previous behaviour and the only sensible answer when there are
        # fewer tokens than slots.
        length = int(prompt_embeddings.shape[1])
        if length >= m:
            edges = [round(i * length / m) for i in range(m + 1)]
            spans = []
            for i in range(m):
                lo, hi = edges[i], max(edges[i + 1], edges[i] + 1)
                spans.append(
                    mx.mean(prompt_embeddings[:, lo:hi, :], axis=1, keepdims=True)
                )
            span_seed = mx.concatenate(spans, axis=1)  # (1,M,D)
        else:
            span_seed = mx.broadcast_to(pooled, (1, m, prompt_embeddings.shape[2]))

        base_seed = config.seed
        if branch_role:
            base_seed = _role_seed(branch_role, base_seed)

        requested_seeds = [
            _validated_context_seed(seed) for seed in (context_seeds or [])
        ]
        # Keep the comm slot (0) and at least one persistent hypothesis slot.
        # The prior quarter-workspace cap silently discarded admitted evidence
        # on the live four-slot profile.
        max_context = max(0, min(len(requested_seeds), m - 2))
        seeds = requested_seeds[:max_context]
        # CP126 (high): "Excess context seeds are silently dropped."
        #
        # The cap is structural — slot 0 is the communication slot and at
        # least one hypothesis slot must stay private, so a four-slot
        # workspace can carry two evidence seeds and no more. That is
        # correct. What was missing is that the caller admitted N pieces of
        # cognitive context and the workspace kept fewer, with nothing
        # saying so: a reasoning trace built on two of five memories looked
        # exactly like one built on all five.
        dropped = requested_seeds[max_context:]
        if dropped:
            record_degradation(
                "latent_workspace",
                ValueError(
                    f"workspace admitted {len(seeds)} of {len(requested_seeds)} "
                    f"context seeds (n_slots={m})"
                ),
                severity="info",
                action=(
                    "seeded the workspace with the leading context and dropped "
                    "the remainder; reasoning did not see it"
                ),
                enforce_failure_policy=False,
            )
        # The context index is the item's OWN position, carried through from
        # the embedder. Recomputing it from the row count silently reassigned
        # provenance whenever an item encoded to nothing and was skipped.
        context_by_slot = {
            1 + j: (int(index), str(source), vector)
            for j, (index, source, vector) in enumerate(seeds)
        }
        context_admission = {
            "schema": "aura.workspace_context_admission.v1",
            "requested": len(requested_seeds),
            "admitted": len(seeds),
            "dropped": len(dropped),
            "dropped_sources": [
                str(source) for _index, source, _vector in dropped
            ][:8],
            "n_slots": m,
            "complete": not dropped,
        }

        roles: list[str] = []
        anchors = []
        for i in range(m):
            context_entry = context_by_slot.get(i)
            role = (
                f"context:{context_entry[1]}"
                if context_entry is not None
                else config.roles[i % len(config.roles)]
            )
            roles.append(role)
            anchors.append(role_anchor(f"{role}#{i}", dim, base_seed))
        anchor_mat = mx.stack(anchors, axis=0)[None, :, :]  # (1,M,D)

        z = span_seed + (float(config.anchor_scale) * target_rms * anchor_mat)
        if context_by_slot:
            rows = []
            for i in range(m):
                entry = context_by_slot.get(i)
                if entry is None:
                    rows.append(z[:, i : i + 1, :])
                    continue
                vector = mx.reshape(entry[2], (1, 1, dim))
                blended = 0.5 * pooled + 0.5 * vector + (
                    float(config.anchor_scale)
                    * target_rms
                    * anchor_mat[:, i : i + 1, :]
                )
                rows.append(blended)
            z = mx.concatenate(rows, axis=1)
        # RMS-match the seeds to the embedding norm distribution.
        z = z * (target_rms / mx.maximum(per_position_rms(z), 1e-6))
        mx.eval(z)
        context_slots = [
            {"slot": slot, "context_index": index, "source": source}
            for slot, (index, source, _vector) in sorted(context_by_slot.items())
        ]
        return cls(
            z,
            roles,
            config,
            context_slots=context_slots,
            context_admission=context_admission,
        )

    # ── State management ────────────────────────────────────────────────
    def snapshot(self):
        return self.z

    def restore(self, snap) -> None:
        self.z = snap

    def update(self, new_z) -> None:
        self.z = new_z

    @property
    def context_slot_indices(self) -> tuple[int, ...]:
        return tuple(
            sorted(
                int(item["slot"])
                for item in self.context_slots
                if isinstance(item, dict) and type(item.get("slot")) is int
            )
        )

    def hypothesis_slot_indices(self, *, comm_slot: int = 0) -> tuple[int, ...]:
        """Writable private state that persists from one recurrent step to the next."""

        context = set(self.context_slot_indices)
        return tuple(
            index
            for index in range(int(self.z.shape[1]))
            if index != comm_slot and index not in context
        )

    def seal_context_evidence(self) -> None:
        """Freeze post-prelude evidence rows as the recurrent source of truth."""

        self._context_evidence_anchor = self.z

    def restore_context_evidence(self, candidate):
        """Return ``candidate`` with every sealed evidence row restored exactly."""

        indices = set(self.context_slot_indices)
        anchor = self._context_evidence_anchor
        if not indices or anchor is None:
            return candidate
        import mlx.core as mx

        restored = mx.concatenate(
            [
                anchor[:, index : index + 1, :]
                if index in indices
                else candidate[:, index : index + 1, :]
                for index in range(int(candidate.shape[1]))
            ],
            axis=1,
        )
        mx.eval(restored)
        return restored

    def select_slots(self, state, indices: tuple[int, ...]):
        """Project a slot subset without changing order or exposing tensor contents."""

        import mlx.core as mx

        if not indices:
            return state[:, :0, :]
        return mx.concatenate(
            [state[:, index : index + 1, :] for index in indices],
            axis=1,
        )

    # ── Causality instrumentation (Experiment 3) ────────────────────────
    def ablate(self, slot_index: int, mode: str = "zero") -> SlotAblation:
        """Destroy one slot's content in-place (zero or matched-RMS noise).

        Returns the ablation record; pass it to :meth:`restore_ablation` to
        prove recovery. Ablating the workspace BEFORE final persistence tests
        whether the slot carried causally necessary intermediate computation.
        """
        import mlx.core as mx

        if not 0 <= slot_index < self.z.shape[1]:
            raise ValueError(f"slot_index {slot_index} outside workspace of {self.z.shape[1]}")
        record = SlotAblation(slot_index=slot_index, mode=mode, prior_state=self.z)
        keep = self.z
        if mode == "zero":
            replacement = mx.zeros_like(keep[:, slot_index : slot_index + 1, :])
        elif mode == "noise":
            key = mx.random.key(_role_seed(f"ablate#{slot_index}", self.config.seed))
            noise = mx.random.normal(keep[:, slot_index : slot_index + 1, :].shape, key=key)
            rms_here = per_position_rms(keep[:, slot_index : slot_index + 1, :])
            replacement = noise * rms_here / mx.maximum(per_position_rms(noise), 1e-6)
        else:
            raise ValueError(f"unknown ablation mode: {mode!r}")
        self.z = mx.concatenate(
            [keep[:, :slot_index, :], replacement, keep[:, slot_index + 1 :, :]], axis=1
        )
        mx.eval(self.z)
        self._ablations.append(record)
        return record

    def restore_ablation(self, record: SlotAblation) -> None:
        self.z = record.prior_state
        if record in self._ablations:
            self._ablations.remove(record)

    # ── Readouts ────────────────────────────────────────────────────────
    def summary(self):
        """Mean slot state (1, 1, D) — the branch-exchange currency."""
        import mlx.core as mx

        return mx.mean(self.z, axis=1, keepdims=True)

    def stats(self) -> dict[str, Any]:
        """Cheap scalar readouts for receipts and health (no tensors)."""
        import mlx.core as mx

        rms_now = per_position_rms(self.z)
        drift_num = mx.linalg.norm(self.z - self.seed_z)
        drift_den = mx.maximum(mx.linalg.norm(self.seed_z), 1e-6)
        return {
            "n_slots": int(self.z.shape[1]),
            "dim": int(self.z.shape[2]),
            "mean_rms": float(mx.mean(rms_now)),
            "max_rms": float(mx.max(rms_now)),
            "seed_drift": float(drift_num / drift_den),
            "roles": list(self.roles),
            "active_ablations": len(self._ablations),
        }


__all__ = ["LatentWorkspace", "SlotAblation", "per_position_rms", "role_anchor"]
