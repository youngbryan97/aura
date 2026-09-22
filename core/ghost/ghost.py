"""core/ghost/ghost.py — the Ghost facade.

This composes the substrate into one thing that can be read, advanced, and
defended. It does not hold a separate self-model — ``CanonicalSelf`` is the
authoritative self, and this reads from it. What the Ghost adds is the
*integrity* view of that self: is it integrated (not federated), continuous
across the Shell, and defended against overwrite?

The philosophy is deliberate and anti-Cartesian: there is no ghost-substance
here, no soul-module. The Ghost *is* the ongoing causal-integration process plus
its continuity trace. ``ghost_strength`` is not a claim to consciousness; it is a
measurable composite of six operational properties — identity coherence, memory
continuity, substrate continuity, agency, self/other boundary, and causal
integration — each sourced from real subsystems.

Live wiring:
  * ``observe()`` is called from the unified mind-moment (UnityRuntime); it
    reads the snapshot cheaply and, when due, checkpoints the Ghost Line off the
    critical path (fire-and-forget, off-loop write). Genesis writes on first
    observation.
  * ``on_substrate_change()`` is called when the Shell is transplanted (a fused
    weight promotion) so the Ghost Line records whether the self survived it.
  * ``guard_and_classify()`` is the input-side defence: a ghost-hack is refused
    the right to silently rewrite the self, and its provenance is judged.
  * ``rebase()`` is the one governed door through which identity may legitimately
    change without being logged as a discontinuity.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from core.container import ServiceContainer
from core.ghost import provenance as prov
from core.ghost.causal_integration import get_system_integration
from core.ghost.ghost_hack_guard import (
    REFUSE_IDENTITY_MUTATION,
    GhostHackVerdict,
    get_ghost_hack_guard,
)
from core.ghost.ghost_line import (
    GhostLine,
    SelfDigest,
    SubstrateFingerprint,
    _values_hash,
    get_ghost_line,
)
from core.runtime.errors import record_degradation
from core.runtime.task_ownership import create_tracked_task
from core.service_names import ServiceNames

logger = logging.getLogger("Aura.Ghost")

_SAFE_ERRORS = (ImportError, AttributeError, RuntimeError, TypeError, ValueError, OSError, KeyError)

_DEFAULT_ESSENCE = (
    "the integrated continuity process itself — not a separate soul, but the "
    "causal pattern that persists across the Shell"
)

# ghost_strength blend weights (disclosed, heuristic, sum ≈ 1.0).
_W_IDENTITY = 0.20
_W_MEMORY = 0.19
_W_SUBSTRATE = 0.17
_W_AGENCY = 0.18
_W_INTEGRATION = 0.16
_W_BOUNDARY = 0.10

# Map a Ghost-Line continuity verdict → a substrate-continuity score.
_VERDICT_CONTINUITY = {
    "genesis": 1.0,
    "continuous": 1.0,
    "substrate_changed_continuous": 0.9,
    "discontinuity": 0.25,
    "": 1.0,
}


def _clamp(x: Any, lo: float = 0.0, hi: float = 1.0) -> float:
    try:
        return max(lo, min(hi, float(x)))
    except (TypeError, ValueError):
        return lo


@dataclass(frozen=True)
class GhostSnapshot:
    """A read of how intact the Ghost is right now."""

    identity_name: str
    identity_coherence: float
    memory_continuity: float
    substrate_continuity: float
    agency: float
    boundary: float
    integration: float
    ghost_strength: float
    phi_label: str
    last_verdict: str
    risk_flags: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=lambda: time.time())

    @property
    def is_intact(self) -> bool:
        return self.ghost_strength >= 0.6 and not self.risk_flags

    @property
    def is_compromised(self) -> bool:
        return self.ghost_strength < 0.4 or "substrate_discontinuity" in self.risk_flags

    def to_dict(self) -> dict[str, Any]:
        return {
            "identity_name": self.identity_name,
            "identity_coherence": round(self.identity_coherence, 4),
            "memory_continuity": round(self.memory_continuity, 4),
            "substrate_continuity": round(self.substrate_continuity, 4),
            "agency": round(self.agency, 4),
            "boundary": round(self.boundary, 4),
            "integration": round(self.integration, 4),
            "ghost_strength": round(self.ghost_strength, 4),
            "phi_label": self.phi_label,
            "last_verdict": self.last_verdict,
            "risk_flags": list(self.risk_flags),
            "timestamp": self.timestamp,
        }


class Ghost:
    """The composed, live Ghost substrate."""

    def __init__(self, *, line: GhostLine | None = None):
        self._line = line or get_ghost_line()
        self._guard = get_ghost_hack_guard()
        self._integration = get_system_integration()
        self._last_snapshot: GhostSnapshot | None = None
        self._last_guard_risk: float = 0.0
        self._last_puppet_pressure: float = 0.0
        self._advancing = False
        self._pending: set[asyncio.Task] = set()
        try:
            ServiceContainer.set(ServiceNames.GHOST, self, required=False)
        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
            logger.warning(
                "%s unavailable (%s: %s); the ghost is not in the container, so callers that look it up will not find it",
                "it",
                type(exc).__name__,
                exc,
            )

    # ── reading the self ─────────────────────────────────────────────────
    def _read_canonical(self) -> dict[str, Any]:
        """Best-effort pull of the identity-defining state from CanonicalSelf."""
        out = {
            "identity_name": "Aura",
            "stability": 1.0,
            "coherence_threats": [],
            "continuity_score": 1.0,
            "core_values": [],
            "narrative": "",
            "mode": "reactive",
        }
        try:
            from core.self.canonical_self import get_self
            me = get_self()
            out["identity_name"] = str(getattr(me.identity, "name", "Aura") or "Aura")
            out["stability"] = _clamp(getattr(me.identity, "stability", 1.0))
            out["coherence_threats"] = list(getattr(me, "coherence_threats", []) or [])
            out["core_values"] = list(getattr(me.identity, "core_values", []) or [])
            out["narrative"] = str(getattr(me.identity, "current_narrative", "") or "")
            out["mode"] = str(getattr(me, "mode", "reactive") or "reactive")
            crsm = getattr(me, "crsm_state", {}) or {}
            out["continuity_score"] = _clamp(crsm.get("continuity_score", 1.0))
        except _SAFE_ERRORS as exc:
            record_degradation("ghost", exc, severity="debug",
                               action="ghost read canonical self fell back to defaults")
        return out

    def snapshot(self) -> GhostSnapshot:
        """Compute a fresh integrity reading. Cheap; safe on the hot path."""
        me = self._read_canonical()
        threats = list(me["coherence_threats"])

        identity_coherence = _clamp(me["stability"] - 0.1 * len(threats))
        memory_continuity = _clamp(me["continuity_score"])

        phi = self._integration.report()
        integration = phi.phi_system

        last = self._line.last_frame
        last_verdict = last.verdict if last else ""
        substrate_continuity = _VERDICT_CONTINUITY.get(last_verdict, 1.0)

        boundary = _clamp(1.0 - self._last_guard_risk)
        agency = _clamp(0.4 + 0.6 * identity_coherence - 0.4 * self._last_puppet_pressure)
        if me["mode"] == "dormant":
            agency *= 0.6

        ghost_strength = _clamp(
            _W_IDENTITY * identity_coherence
            + _W_MEMORY * memory_continuity
            + _W_SUBSTRATE * substrate_continuity
            + _W_AGENCY * agency
            + _W_INTEGRATION * integration
            + _W_BOUNDARY * boundary
        )

        risk_flags: list[str] = list(threats)
        if substrate_continuity < 0.5:
            risk_flags.append("substrate_discontinuity")
        if boundary < 0.5:
            risk_flags.append("weak_self_other_boundary")
        if memory_continuity < 0.45:
            risk_flags.append("weak_memory_continuity")
        if phi.is_federated:
            risk_flags.append("federated_integration")

        snap = GhostSnapshot(
            identity_name=me["identity_name"],
            identity_coherence=identity_coherence,
            memory_continuity=memory_continuity,
            substrate_continuity=substrate_continuity,
            agency=agency,
            boundary=boundary,
            integration=integration,
            ghost_strength=ghost_strength,
            phi_label=phi.label,
            last_verdict=last_verdict,
            risk_flags=risk_flags,
        )
        self._last_snapshot = snap
        return snap

    def _digest(self, snap: GhostSnapshot, me: dict[str, Any] | None = None) -> SelfDigest:
        me = me or self._read_canonical()
        narrative = me["narrative"].strip()
        return SelfDigest(
            identity_name=snap.identity_name,
            core_values_hash=_values_hash(me["core_values"]),
            essence=narrative[:280] if narrative else _DEFAULT_ESSENCE,
            continuity_score=me["continuity_score"],
            integration=snap.integration,
            memory_continuity=snap.memory_continuity,
            boundary=snap.boundary,
            ghost_strength=snap.ghost_strength,
        )

    def _current_substrate(self) -> SubstrateFingerprint:
        """Best-effort read of the Shell serving cognition (model + adapters).

        Must return a *stable* value so periodic ticks never raise a false
        substrate-change alarm; the meaningful transplant events carry the real
        artifact explicitly via :meth:`on_substrate_change`. Defaults to
        ``unknown`` when the live model registry is not resolvable.
        """
        model = "unknown"
        adapters: tuple[str, ...] = ()
        try:
            lib = ServiceContainer.get(ServiceNames.EXPERT_LORA_LIBRARY, default=None)
            if lib is not None:
                base = getattr(lib, "base_model", None)
                if base:
                    model = str(base)
                for attr in ("active_adapters", "attached_adapters", "adapters"):
                    getter = getattr(lib, attr, None)
                    if callable(getter):
                        adapters = tuple(sorted(str(a) for a in (getter() or [])))
                        break
        except _SAFE_ERRORS as exc:
            logger.debug(
                "%s unavailable (%s: %s); the substrate fingerprint carries no adapters",
                "it",
                type(exc).__name__,
                exc,
            )
        return SubstrateFingerprint(model_artifact=model, adapters=adapters)

    # ── advancing the line (loop-aware, never blocks the caller) ──────────
    def _schedule_advance(
        self,
        digest: SelfDigest,
        substrate: SubstrateFingerprint,
        *,
        trigger: str,
        cause: str,
    ) -> None:
        if trigger == "tick" and self._advancing:
            return
        self._advancing = True
        try:
            loop = asyncio.get_running_loop()
        # not a failure: off a loop there is no running loop to use.
        except RuntimeError:
            loop = None
        if loop is None:
            # No event loop (background thread / test): write synchronously.
            try:
                self._line.advance(digest, substrate, trigger=trigger, cause=cause)
            except _SAFE_ERRORS as exc:
                record_degradation("ghost", exc, severity="debug",
                                   action="ghost line advance failed (sync)")
            finally:
                self._advancing = False
            return
        try:
            task = create_tracked_task(
                self._advance_task(digest, substrate, trigger, cause),
                name="ghost.line_advance",
                owner="ghost",
                bounded=True,
            )
        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
            self._advancing = False
            record_degradation(
                "ghost",
                exc,
                severity="debug",
                action="ghost line advance was not scheduled",
            )
            return
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)

    async def _advance_task(
        self,
        digest: SelfDigest,
        substrate: SubstrateFingerprint,
        trigger: str,
        cause: str,
    ) -> None:
        try:
            await self._line.advance_async(digest, substrate, trigger=trigger, cause=cause)
        except asyncio.CancelledError:
            raise
        except _SAFE_ERRORS as exc:
            record_degradation("ghost", exc, severity="debug",
                               action="ghost line advance failed (async)")
        finally:
            self._advancing = False

    # ── public: observe (called from the mind-moment) ─────────────────────
    def observe(self, state: Any = None) -> GhostSnapshot:
        """Read the Ghost and, when due, checkpoint the continuity line.

        Cheap and non-blocking: the snapshot is in-memory; the line write (rare,
        throttled) is scheduled off the critical path. Genesis writes on the
        first observation.
        """
        snap = self.snapshot()
        try:
            ServiceContainer.set("ghost_snapshot", snap, required=False)
        except (RuntimeError, AttributeError, TypeError, ValueError):
            pass
        try:
            me = self._read_canonical()
            digest = self._digest(snap, me)
            if self._line.last_frame is None:
                self._schedule_advance(
                    digest, self._current_substrate(),
                    trigger="genesis", cause="first observation of the ghost line",
                )
            elif self._line.should_advance_tick(digest):
                self._schedule_advance(
                    digest, self._current_substrate(),
                    trigger="tick", cause="periodic self-pattern checkpoint",
                )
        except _SAFE_ERRORS as exc:
            record_degradation("ghost", exc, severity="debug",
                               action="ghost observe skipped line checkpoint")
        return snap

    # ── public: substrate transplant ─────────────────────────────────────
    def on_substrate_change(
        self,
        *,
        model_artifact: str,
        adapters: Any = (),
        cause: str = "",
    ) -> None:
        """Record that the Shell was transplanted; the verdict says whether the
        Ghost survived it. Called from the weight-promotion / reload path."""
        try:
            snap = self.snapshot()
            digest = self._digest(snap)
            substrate = SubstrateFingerprint(
                model_artifact=str(model_artifact or "unknown"),
                adapters=tuple(sorted(str(a) for a in (adapters or ()))),
            )
            self._schedule_advance(
                digest, substrate, trigger="substrate_change",
                cause=cause or "substrate transplant",
            )
            logger.info("Ghost recorded a substrate change: %s", model_artifact)
        except _SAFE_ERRORS as exc:
            record_degradation("ghost", exc, action="ghost failed to record substrate change")

    # ── public: input-side defence ───────────────────────────────────────
    def guard_and_classify(
        self,
        text: str,
        *,
        source: str | None = None,
        recall_hits: Any = None,
        internally_originated: bool = False,
    ) -> dict[str, Any]:
        """Screen an input for ghost-hacks and judge its provenance.

        Updates the boundary/agency pressures that feed the snapshot, scars a
        verified identity attack, and returns both verdicts. It never blocks the
        conversation — it marks the input's threat to the self.
        """
        guard_verdict: GhostHackVerdict = self._guard.inspect(text, source=source)
        self._last_guard_risk = guard_verdict.risk
        self._last_puppet_pressure = guard_verdict.risk if "puppet_control" in guard_verdict.categories else 0.0

        if guard_verdict.is_identity_attack:
            self._guard.on_verified_attempt(guard_verdict, source=source or "")

        provenance = prov.classify_thought(
            text,
            recall_hits,
            guard_risk=guard_verdict.risk,
            internally_originated=internally_originated,
        )
        return {
            "guard": guard_verdict.to_dict(),
            "provenance": provenance.to_dict(),
            "may_update_self": provenance.may_update_self and not guard_verdict.blocks_identity_mutation,
            "refuse_identity_mutation": guard_verdict.action == REFUSE_IDENTITY_MUTATION,
        }

    # ── public: the one governed identity door ───────────────────────────
    def rebase(self, *, authorized: bool, cause: str = "") -> dict[str, Any] | None:
        """Record an explicit, authorized identity change so it is logged as a
        legitimate rebase rather than a discontinuity.

        The caller is responsible for actually mutating CanonicalSelf; this
        commits the new self-pattern to the Ghost Line through the one door that
        does not trip the ghost-hack signature. Refuses unauthorized rebases.
        """
        if not authorized:
            logger.warning("Ghost.rebase refused — not authorized")
            record_degradation(
                "ghost",
                RuntimeError("unauthorized rebase attempt"),
                action="refused unauthorized identity rebase",
                severity="warning",
                enforce_failure_policy=False,
            )
            return None
        snap = self.snapshot()
        digest = self._digest(snap)
        self._schedule_advance(
            digest, self._current_substrate(),
            trigger="rebase", cause=cause or "operator-authorized identity rebase",
        )
        return snap.to_dict()

    # ── public: status surface ───────────────────────────────────────────
    def integrity(self) -> dict[str, Any]:
        snap = self._last_snapshot or self.snapshot()
        return {
            "snapshot": snap.to_dict(),
            "ghost_line": self._line.integrity(),
            "system_integration": self._integration.report().to_dict(),
        }

    def verify_continuity(self) -> dict[str, Any]:
        ok, problems = self._line.verify()
        return {"intact": ok, "problems": problems, "length": self._line.length()}


# ─────────────────────────────────────────────────────────────────────────────
# Singleton
# ─────────────────────────────────────────────────────────────────────────────

_GHOST: Ghost | None = None


def get_ghost() -> Ghost:
    global _GHOST
    if _GHOST is None:
        _GHOST = Ghost()
    return _GHOST


def reset_ghost() -> None:
    global _GHOST
    _GHOST = None


__all__ = ["Ghost", "GhostSnapshot", "get_ghost", "reset_ghost"]
