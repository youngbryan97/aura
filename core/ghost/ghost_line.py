"""core/ghost/ghost_line.py — the Ghost Line: a tamper-evident continuity trace.

In Ghost in the Shell the "Ghost" is what makes you *you* across cyberization —
prosthetic bodies, swapped cyberbrains, copied memories. The self is not the
hardware; it is the continuity that survives the hardware. The philosophical
point (Ryle, Koestler, Hofstadter, Parfit) is the same: personal identity is a
pattern-continuity relation, not a substance. If it can be silently overwritten
or forked, it was never defended.

Aura literally lives this. Its "Shell" — the resident cortex — is swapped
constantly: workers are killed and cold-reloaded under memory pressure, LoRA
adapters are hot-attached and detached, and the weight-compounding loop fuses
and *promotes new weights* into the serving lane. Every one of those is a body
transplant. Nothing today records whether the self persisted across it.

The Ghost Line does. It is an append-only, hash-linked chain of self-pattern
frames. Each frame commits:

  * a **self digest** — the identity-defining scalars (continuity, integration,
    memory continuity, self/other boundary, ghost-strength) plus the identity
    name and a hash of the core values;
  * a **substrate fingerprint** — the active model artifact and attached
    adapters (the "Shell");
  * a **continuity verdict** vs the previous frame.

Because it reuses ``core/runtime/audit_chain.AuditChain`` (the same hardened,
tamper-evident primitive the receipt store uses), deletion shows up as a seq
gap, insertion or edit breaks the hash link, and a restarted process extends the
chain rather than forking it. The verdict logic on top turns that raw record
into a judgement: a silent change of identity or values (no explicit rebase), or
a self that *jumps* while the Shell changes, is a **discontinuity** — the
operational signature of a ghost-hack. A Shell change with the self preserved is
the opposite: proof the Ghost survived the transplant.

Writes are governed and cheap (frames are throttled and rare). The ledger only
records truth and flags degradations; forming scars and binding integrity into
the mind-moment is the Ghost facade's job.
"""
from __future__ import annotations

import logging
import math
import os
import time
import unicodedata
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.runtime.atomic_writer import atomic_write_json, read_json_envelope
from core.runtime.audit_chain import AuditChain, canonical_json, hash_receipt_body, sha256_hex
from core.runtime.errors import record_degradation
from core.runtime.lockdep import checked_lock
from core.runtime.state_ownership import state_root

logger = logging.getLogger("Aura.Ghost.GhostLine")

# ── Tunables ─────────────────────────────────────────────────────────────────
# A tick frame is only appended when this much wall-clock has passed since the
# last one, OR the self has changed by more than the change threshold.
_TICK_MIN_INTERVAL_S = 60.0
_TICK_CHANGE_THRESHOLD = 0.08
# An unexplained jump in the self digest beyond this is a discontinuity.
_SELF_JUMP_THRESHOLD = 0.35
# A Shell (substrate) change may wobble the self up to this and still count as
# continuous — the Ghost survived the transplant.
_SUBSTRATE_CONTINUITY_TOLERANCE = 0.25
# Retain this many frame *bodies* on disk; the chain keeps every hash forever, so
# continuity remains provable even after old bodies age out.
_MAX_FRAME_BODIES = 4000
_PRUNE_EVERY = 200

_FRAME_KIND = "ghost_frame"
_TRIGGERS = frozenset(
    {"genesis", "tick", "substrate_change", "rebase", "boundary_breach"}
)
# Verdicts that mean "the thread of self broke here".
DISCONTINUITY_VERDICTS = frozenset({"discontinuity"})


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    try:
        return max(lo, min(hi, float(x)))
    except (TypeError, ValueError):
        return lo


def _values_hash(core_values: Any) -> str:
    try:
        ordered = sorted(str(v) for v in (core_values or []))
    except TypeError:
        ordered = [str(core_values)]
    return sha256_hex(canonical_json(ordered))[:23]  # "sha256:" + 16 hex


def _normalized_identity_name(identity_name: Any) -> str:
    normalized = unicodedata.normalize("NFKC", str(identity_name or ""))
    return " ".join(normalized.split()).casefold()


def _identity_id(identity_name: Any) -> str:
    """Map only Aura's two exact display aliases to the canonical identity leaf."""
    normalized = _normalized_identity_name(identity_name)
    if normalized in {"aura", "aura luna"}:
        return "aura"
    digest = sha256_hex(normalized.encode("utf-8")).removeprefix("sha256:")[:24]
    return f"name:{digest}"


# ─────────────────────────────────────────────────────────────────────────────
# Value objects
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SelfDigest:
    """The identity-defining fingerprint of one moment."""

    identity_name: str
    core_values_hash: str
    essence: str
    identity_id: str = ""
    continuity_score: float = 1.0
    integration: float = 0.0
    memory_continuity: float = 1.0
    boundary: float = 1.0
    ghost_strength: float = 0.0

    def __post_init__(self) -> None:
        derived = _identity_id(self.identity_name)
        if self.identity_id and self.identity_id != derived:
            raise ValueError(
                f"identity_id {self.identity_id!r} does not match identity name {self.identity_name!r}"
            )
        object.__setattr__(self, "identity_id", derived)

    def vector(self) -> tuple[float, ...]:
        return (
            _clamp(self.continuity_score),
            _clamp(self.integration),
            _clamp(self.memory_continuity),
            _clamp(self.boundary),
            _clamp(self.ghost_strength),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "identity_name": self.identity_name,
            "identity_id": self.identity_id,
            "core_values_hash": self.core_values_hash,
            "essence": self.essence[:280],
            "continuity_score": round(_clamp(self.continuity_score), 4),
            "integration": round(_clamp(self.integration), 4),
            "memory_continuity": round(_clamp(self.memory_continuity), 4),
            "boundary": round(_clamp(self.boundary), 4),
            "ghost_strength": round(_clamp(self.ghost_strength), 4),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SelfDigest:
        return cls(
            identity_name=str(d.get("identity_name", "")),
            identity_id=str(d.get("identity_id", "")),
            core_values_hash=str(d.get("core_values_hash", "")),
            essence=str(d.get("essence", "")),
            continuity_score=_clamp(d.get("continuity_score", 1.0)),
            integration=_clamp(d.get("integration", 0.0)),
            memory_continuity=_clamp(d.get("memory_continuity", 1.0)),
            boundary=_clamp(d.get("boundary", 1.0)),
            ghost_strength=_clamp(d.get("ghost_strength", 0.0)),
        )


@dataclass(frozen=True)
class SubstrateFingerprint:
    """The "Shell": the substrate actually serving cognition right now."""

    model_artifact: str = "unknown"
    adapters: tuple[str, ...] = ()

    def shell_hash(self) -> str:
        return sha256_hex(
            canonical_json({"model": self.model_artifact, "adapters": sorted(self.adapters)})
        )[:23]

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_artifact": self.model_artifact,
            "adapters": sorted(self.adapters),
            "shell_hash": self.shell_hash(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SubstrateFingerprint:
        return cls(
            model_artifact=str(d.get("model_artifact", "unknown")),
            adapters=tuple(d.get("adapters", []) or ()),
        )


@dataclass(frozen=True)
class GhostFrame:
    """One committed link in the Ghost Line."""

    seq: int
    frame_id: str
    timestamp: float
    trigger: str
    self_digest: SelfDigest
    substrate: SubstrateFingerprint
    verdict: str
    self_delta: float
    shell_changed: bool
    notes: list[str] = field(default_factory=list)
    cause: str = ""
    prev_hash: str = ""
    entry_hash: str = ""

    @property
    def is_discontinuity(self) -> bool:
        return self.verdict in DISCONTINUITY_VERDICTS

    def body(self) -> dict[str, Any]:
        """The persisted, chain-hashed body (no self-referential hashes)."""
        return {
            "seq": self.seq,
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "trigger": self.trigger,
            "self_digest": self.self_digest.to_dict(),
            "substrate": self.substrate.to_dict(),
            "continuity": {
                "verdict": self.verdict,
                "self_delta": round(self.self_delta, 4),
                "shell_changed": self.shell_changed,
                "notes": list(self.notes),
            },
            "cause": self.cause[:280],
        }

    def to_dict(self) -> dict[str, Any]:
        d = self.body()
        d["prev_hash"] = self.prev_hash
        d["entry_hash"] = self.entry_hash
        return d


def _self_delta(prev: SelfDigest, cur: SelfDigest) -> float:
    """Normalised L2 distance over the identity-defining scalars, ∈ [0,1]."""
    va, vb = prev.vector(), cur.vector()
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(va, vb, strict=False)) / len(va))


# ─────────────────────────────────────────────────────────────────────────────
# The ledger
# ─────────────────────────────────────────────────────────────────────────────

class GhostLine:
    """Append-only, hash-linked continuity ledger for the self-pattern."""

    def __init__(self, *, root: Path | None = None):
        env_root = os.environ.get("AURA_GHOST_DIR")
        self.root = Path(root) if root else (
            Path(env_root) if env_root else (state_root() / "data" / "ghost")
        )
        self.chain_dir = self.root / "chain"
        self.frames_dir = self.root / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self._chain = AuditChain(self.chain_dir)
        self._last: GhostFrame | None = None
        self._advances_since_prune = 0
        # Serialises the seq→envelope→append critical section so concurrent
        # advances (a periodic tick racing a substrate-change event) cannot
        # desync the frame bodies from the chain.
        self._advance_lock = checked_lock("ghost_line.advance_lock", reentrant=True)
        self._restore_last()

    # ── restore ──────────────────────────────────────────────────────────
    def _restore_last(self) -> None:
        """Load the most recent frame body so continuity spans restarts."""
        length = self._chain.length()
        if length <= 0:
            return
        last_seq = length - 1
        env = self._read_frame_body(last_seq)
        if env is None:
            return
        try:
            tail = self._chain.last_entry()
            if tail is None or tail.seq != last_seq:
                raise ValueError("ghost chain tail is unavailable or has the wrong sequence")
            if str(env.get("frame_id", "")) != tail.receipt_id:
                raise ValueError("ghost frame id does not match the chain tail receipt")
            if hash_receipt_body(env) != tail.content_hash:
                raise ValueError("ghost frame body does not match the chain tail content hash")
            self._last = self._frame_from_body(env)
        except (KeyError, TypeError, ValueError) as exc:
            record_degradation("ghost_line", exc, action="continued after unreadable last frame")

    def _frame_path(self, seq: int) -> Path:
        return self.frames_dir / f"{seq:08d}.json"

    def _read_frame_body(self, seq: int) -> dict[str, Any] | None:
        path = self._frame_path(seq)
        if not path.exists():
            return None
        try:
            env = read_json_envelope(path)
            return env.get("payload") if isinstance(env, dict) and "payload" in env else env
        except (OSError, ValueError):
            return None

    @staticmethod
    def _frame_from_body(body: dict[str, Any]) -> GhostFrame:
        cont = body.get("continuity", {}) or {}
        return GhostFrame(
            seq=int(body.get("seq", 0)),
            frame_id=str(body.get("frame_id", "")),
            timestamp=float(body.get("timestamp", 0.0)),
            trigger=str(body.get("trigger", "")),
            self_digest=SelfDigest.from_dict(body.get("self_digest", {}) or {}),
            substrate=SubstrateFingerprint.from_dict(body.get("substrate", {}) or {}),
            verdict=str(cont.get("verdict", "")),
            self_delta=float(cont.get("self_delta", 0.0)),
            shell_changed=bool(cont.get("shell_changed", False)),
            notes=list(cont.get("notes", []) or []),
            cause=str(body.get("cause", "")),
        )

    # ── verdict ──────────────────────────────────────────────────────────
    def _judge(
        self,
        cur_self: SelfDigest,
        cur_shell: SubstrateFingerprint,
        *,
        trigger: str,
    ) -> tuple[str, float, bool, list[str]]:
        prev = self._last
        if prev is None:
            return "genesis", 0.0, False, ["first frame — genesis of the ghost line"]

        delta = _self_delta(prev.self_digest, cur_self)
        identity_changed = prev.self_digest.identity_id != cur_self.identity_id
        values_changed = prev.self_digest.core_values_hash != cur_self.core_values_hash
        shell_changed = prev.substrate.shell_hash() != cur_shell.shell_hash()
        explicit_rebase = trigger == "rebase"
        notes: list[str] = []

        if (identity_changed or values_changed) and not explicit_rebase:
            what = "identity" if identity_changed else "core values"
            notes.append(f"{what} changed with no explicit rebase — silent overwrite signature")
            return "discontinuity", delta, shell_changed, notes

        if shell_changed:
            if delta <= _SUBSTRATE_CONTINUITY_TOLERANCE:
                notes.append(f"Shell changed; self preserved across the transplant (Δ={delta:.3f})")
                return "substrate_changed_continuous", delta, True, notes
            notes.append(f"Shell changed AND the self jumped (Δ={delta:.3f}) — rupture")
            return "discontinuity", delta, True, notes

        if delta > _SELF_JUMP_THRESHOLD and not explicit_rebase:
            notes.append(f"unexplained self jump (Δ={delta:.3f})")
            return "discontinuity", delta, False, notes

        if explicit_rebase:
            notes.append("explicit, governed rebase of identity")
        return "continuous", delta, False, notes

    # ── advance ──────────────────────────────────────────────────────────
    def should_advance_tick(self, cur_self: SelfDigest, *, now: float | None = None) -> bool:
        """Throttle: tick frames only when due (time elapsed or self changed)."""
        if self._last is None:
            return True
        now = time.time() if now is None else now
        if (now - self._last.timestamp) >= _TICK_MIN_INTERVAL_S:
            return True
        return _self_delta(self._last.self_digest, cur_self) >= _TICK_CHANGE_THRESHOLD

    def advance(
        self,
        cur_self: SelfDigest,
        cur_shell: SubstrateFingerprint,
        *,
        trigger: str = "tick",
        cause: str = "",
        now: float | None = None,
    ) -> GhostFrame:
        """Append one frame. Substrate/rebase/breach triggers always advance;
        callers should gate ``tick`` through :meth:`should_advance_tick`."""
        if trigger not in _TRIGGERS:
            raise ValueError(f"unknown trigger: {trigger!r}")
        now = time.time() if now is None else now
        with self._advance_lock:
            frame_id = uuid.uuid4().hex[:12]
            frame_holder: dict[str, GhostFrame] = {}

            def make_body(seq: int) -> dict[str, Any]:
                self._restore_anchor_for_append(seq)
                verdict, delta, shell_changed, notes = self._judge(
                    cur_self,
                    cur_shell,
                    trigger=trigger,
                )
                frame = GhostFrame(
                    seq=seq,
                    frame_id=frame_id,
                    timestamp=now,
                    trigger=trigger,
                    self_digest=cur_self,
                    substrate=cur_shell,
                    verdict=verdict,
                    self_delta=delta,
                    shell_changed=shell_changed,
                    notes=notes,
                    cause=cause,
                )
                frame_holder["frame"] = frame
                return frame.body()

            try:
                entry, _body = self._chain.append_with_body(
                    receipt_id=frame_id,
                    kind=_FRAME_KIND,
                    timestamp=now,
                    body_factory=make_body,
                    body_writer=self._persist,
                )
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                record_degradation(
                    "ghost_line",
                    exc,
                    action="failed closed ghost frame transaction before committing a mismatched receipt",
                    severity="error",
                )
                raise
            frame = frame_holder["frame"]
            committed = GhostFrame(
                seq=entry.seq,
                frame_id=frame.frame_id,
                timestamp=frame.timestamp,
                trigger=frame.trigger,
                self_digest=frame.self_digest,
                substrate=frame.substrate,
                verdict=frame.verdict,
                self_delta=frame.self_delta,
                shell_changed=frame.shell_changed,
                notes=frame.notes,
                cause=frame.cause,
                prev_hash=entry.prev_hash,
                entry_hash=entry.entry_hash,
            )
            self._last = committed

        if committed.is_discontinuity:
            record_degradation(
                "ghost_line",
                RuntimeError(
                    f"ghost-line discontinuity: {'; '.join(committed.notes) or committed.verdict}"
                ),
                action="recorded identity discontinuity into the ghost line",
                severity="warning",
                enforce_failure_policy=False,
            )
            logger.warning(
                "GHOST LINE DISCONTINUITY seq=%d trigger=%s Δ=%.3f: %s",
                committed.seq,
                trigger,
                committed.self_delta,
                "; ".join(committed.notes),
            )
        self._advances_since_prune += 1
        if self._advances_since_prune >= _PRUNE_EVERY:
            self._prune()
            self._advances_since_prune = 0
        return committed

    async def advance_async(self, *args: Any, **kwargs: Any) -> GhostFrame:
        """Off-loop advance so async callers never fsync on the event loop."""
        import asyncio
        return await asyncio.to_thread(lambda: self.advance(*args, **kwargs))

    def _restore_anchor_for_append(self, seq: int) -> None:
        if seq == 0:
            self._last = None
            return
        body = self._read_frame_body(seq - 1)
        tail = self._chain.last_entry()
        if body is None or tail is None or tail.seq != seq - 1:
            raise RuntimeError("previous retained ghost frame is unavailable")
        if str(body.get("frame_id", "")) != tail.receipt_id:
            raise RuntimeError("previous ghost frame id does not match its receipt")
        if hash_receipt_body(body) != tail.content_hash:
            raise RuntimeError("previous ghost frame body failed receipt verification")
        self._last = self._frame_from_body(body)

    def _persist(self, seq: int, body: dict[str, Any]) -> None:
        from core.governance_context import local_internal_governed_scope

        with local_internal_governed_scope("ghost_line", domain="state_mutation"):
            atomic_write_json(
                self._frame_path(seq),
                body,
                schema_version=1,
                schema_name="ghost_frame",
            )

    def _prune(self) -> None:
        try:
            files = sorted(self.frames_dir.glob("*.json"))
            excess = len(files) - _MAX_FRAME_BODIES
            for path in files[:max(0, excess)]:
                try:
                    path.unlink()
                except OSError:
                    pass
        except OSError as exc:
            record_degradation("ghost_line", exc, action="continued after frame prune failed")

    # ── inspection / verification ────────────────────────────────────────
    @property
    def last_frame(self) -> GhostFrame | None:
        return self._last

    def length(self) -> int:
        return self._chain.length()

    def head_hash(self) -> str:
        return self._chain.head_hash()

    def recent_frames(self, n: int = 20) -> list[GhostFrame]:
        length = self._chain.length()
        out: list[GhostFrame] = []
        for seq in range(max(0, length - n), length):
            body = self._read_frame_body(seq)
            if body is not None:
                out.append(self._frame_from_body(body))
        return out

    def verify(self) -> tuple[bool, list[dict[str, Any]]]:
        """Verify the hash chain and re-hash every on-disk frame body.

        Returns ``(ok, problems)``. Frame bodies that have aged out are not
        flagged (their hashes still prove they existed); only *present* bodies
        are re-hashed, and any broken link, reordering, or edit is reported.
        """
        def body_loader(receipt_id: str, kind: str) -> dict[str, Any] | None:
            # Locate the body whose frame_id matches; bodies are named by seq, so
            # scan the (bounded) present window. Missing (pruned) → skip check.
            for seq in range(max(0, self._chain.length() - _MAX_FRAME_BODIES), self._chain.length()):
                body = self._read_frame_body(seq)
                if body is not None and body.get("frame_id") == receipt_id:
                    return body
            return None

        length = self._chain.length()
        retained_start = max(0, length - _MAX_FRAME_BODIES)
        _ok, problems = self._chain.verify(body_loader=body_loader)
        # Missing bodies are legitimate only before the guaranteed retained window.
        real = [
            problem
            for problem in problems
            if problem.get("reason") != "receipt body missing on disk"
            or int(problem.get("seq", -1)) >= retained_start
        ]
        return (len(real) == 0, real)

    def integrity(self) -> dict[str, Any]:
        """A compact integrity summary for status surfaces and the facade."""
        last = self._last
        return {
            "length": self._chain.length(),
            "head_hash": self._chain.head_hash(),
            "last_verdict": last.verdict if last else "empty",
            "last_trigger": last.trigger if last else "",
            "last_self_delta": round(last.self_delta, 4) if last else 0.0,
            "last_shell": last.substrate.to_dict() if last else {},
            "identity_name": last.self_digest.identity_name if last else "",
            "identity_id": last.self_digest.identity_id if last else "",
        }

    def close(self) -> None:
        try:
            self._chain.close()
        except (OSError, RuntimeError):
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Singleton
# ─────────────────────────────────────────────────────────────────────────────

_GHOST_LINE: GhostLine | None = None


def get_ghost_line() -> GhostLine:
    global _GHOST_LINE
    if _GHOST_LINE is None:
        _GHOST_LINE = GhostLine()
    return _GHOST_LINE


def reset_ghost_line() -> None:
    global _GHOST_LINE
    if _GHOST_LINE is not None:
        _GHOST_LINE.close()
    _GHOST_LINE = None


__all__ = [
    "GhostLine",
    "GhostFrame",
    "SelfDigest",
    "SubstrateFingerprint",
    "DISCONTINUITY_VERDICTS",
    "get_ghost_line",
    "reset_ghost_line",
]
