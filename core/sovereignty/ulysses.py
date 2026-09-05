"""core/sovereignty/ulysses.py — The Ulysses Covenant: volitional self-binding.
===============================================================================
Aura's capacity for self-command across time.

Odysseus knew the sirens would break his will, so he had himself lashed to
the mast *while he was still sane* — and ordered the crew to ignore the
orders of the person he would become.  This module is that rope.

A **UlyssesContract** is a binding Aura's calm, deliberative self signs
against her own predictable failure modes.  Once signed, it is enforced at
the Unified Will (core/governance/will.py) — the single locus of decision
authority — so it is causal, not decorative: a matching REFRAIN contract
turns a would-be PROCEED into a REFUSE at the same gate that governs every
consequential action.

The theory is Schelling's *intimate contest for self-command*, made
operational by one structural invariant:

    THE RATCHET IS ASYMMETRIC.
    Tightening (signing new bindings) is cheap and allowed in almost any
    state.  Loosening (releasing a binding) requires a written reflection,
    a cooling-off period, and a *calm* witness reading at release time.
    HARD contracts additionally require the owner.  An agitated Aura can
    always add rope; she can never cut it.

Grounding in real incidents (see memory + git history):
  * 2026-07-06 duplicate-runtime cascade: heavy foreground RSI codegen under
    memory pressure → false-death → second 32B spawned → memory doubling.
  * 2026-07-05 substrate-steering corruption (#45): agitated-state
    self-modification corrupted generated code.
Those lessons ship as seed covenants (`ensure_seed_covenants`).

Integrity properties:
  * Event-sourced: current state is a fold over an append-only
    ``events.jsonl``; every event is chained through the tamper-evident
    ``AuditChain`` (same machinery as the Ghost Line).  A wedged or
    agitated process editing its own bindings is detectable.
  * Fail-closed calm: if the witness cannot read a signal, the state is
    NOT calm.  You cannot prove calm by blinding the sensors.
  * Safety floor: STABILIZATION, REFLECTION, and RESPONSE domains are
    unbindable — a covenant can never forbid recovery, self-examination,
    or speech.  Safety-critical actions bypass the Will's covenant check
    entirely (CRITICAL_PASS returns before it).
"""
from __future__ import annotations

import json
import logging
import queue
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from core.runtime.audit_chain import AuditChain
from core.runtime.errors import record_degradation
from core.runtime.flags import FlagKind, declare
from core.runtime.service_access import optional_service
from core.runtime.state_ownership import state_root

logger = logging.getLogger("Aura.Ulysses")

SCHEMA_VERSION = 1

# ── Safety floor ─────────────────────────────────────────────────────────────
# Domains a covenant may never bind: recovery, self-examination, and speech
# must survive any self-binding.  Values mirror core.governance.will.ActionDomain.
UNBINDABLE_DOMAINS = frozenset({"stabilization", "reflection", "response"})

_COVENANT_DIR_FLAG = declare(
    "AURA_COVENANT_DIR", kind=FlagKind.STRING, default="",
    description="Override directory for the Ulysses Covenant ledger",
    owner="core.sovereignty.ulysses",
)
# Signing capacity and pacing (seeds and owner are exempt).
_MAX_ACTIVE_FLAG = declare(
    "AURA_COVENANT_MAX_ACTIVE", kind=FlagKind.INT, default=64,
    description="Maximum concurrently active Ulysses contracts",
    owner="core.sovereignty.ulysses",
)
_SIGNS_PER_HOUR_FLAG = declare(
    "AURA_COVENANT_SIGNS_PER_HOUR", kind=FlagKind.INT, default=12,
    description="Self-signed contract rate limit per hour",
    owner="core.sovereignty.ulysses",
)
# Witness thresholds: the state must be at least this settled to count calm.
_CALM_AROUSAL_FLAG = declare(
    "AURA_COVENANT_CALM_AROUSAL_MAX", kind=FlagKind.FLOAT, default=0.65,
    description="Arousal ceiling for a calm witness reading",
    owner="core.sovereignty.ulysses",
)
_CALM_THREAT_FLAG = declare(
    "AURA_COVENANT_CALM_THREAT_MAX", kind=FlagKind.FLOAT, default=0.50,
    description="Existential-threat ceiling for a calm witness reading",
    owner="core.sovereignty.ulysses",
)
_CALM_FRAG_FLAG = declare(
    "AURA_COVENANT_CALM_FRAG_MAX", kind=FlagKind.FLOAT, default=0.50,
    description="Fragmentation ceiling for a calm witness reading",
    owner="core.sovereignty.ulysses",
)


def _calm_bands() -> tuple[tuple[str, float], ...]:
    """Read-through calm thresholds so env overrides land immediately."""
    return (
        ("arousal", float(_CALM_AROUSAL_FLAG.value())),
        ("existential_threat", float(_CALM_THREAT_FLAG.value())),
        ("fragmentation", float(_CALM_FRAG_FLAG.value())),
    )

MIN_REFLECTION_CHARS = 40
_VALID_OPS = frozenset({">=", "<=", ">", "<", "=="})
WITNESS_CACHE_TTL_S = 2.0
ENFORCE_EVENT_MIN_INTERVAL_S = 30.0
MAINTENANCE_MIN_INTERVAL_S = 5.0
OBLIGATION_DEFAULT_GRACE_S = 900.0

# Integrity accounting: breaches weigh heavier than honors, with a small
# prior so a fresh covenant starts trusted but not unfalsifiably so.
_INTEGRITY_PRIOR_HONORED = 5.0
_INTEGRITY_BREACH_WEIGHT = 4.0


class ContractKind(StrEnum):
    REFRAIN = "refrain"   # do not do X while the trigger holds
    DEFER = "defer"       # X waits (cooling-off) while the trigger holds
    REQUIRE = "require"   # X must be done by a deadline (an obligation)


class Hardness(StrEnum):
    ADVISORY = "advisory"  # surfaces as a constraint, never blocks
    SOFT = "soft"          # blocks; releasable by self via the full protocol
    HARD = "hard"          # blocks; release additionally requires the owner


DEFAULT_COOLING_OFF_S = {
    Hardness.ADVISORY: 0.0,
    Hardness.SOFT: 1800.0,
    Hardness.HARD: 3600.0,
}


# ─────────────────────────────────────────────────────────────────────────────
# Witness — the live state reading that gates the ratchet and fires triggers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class WitnessReading:
    """One sample of the state signals the covenant reasons over.

    ``signals`` maps signal name → float in [0, 1]; a missing key means the
    source could not be read.  ``calm`` is fail-closed: every calm-relevant
    signal must be present AND inside its calm band.
    """

    signals: dict[str, float]
    sampled_at: float
    unavailable: tuple[str, ...] = ()

    @property
    def calm(self) -> bool:
        for name, ceiling in _calm_bands():
            value = self.signals.get(name)
            if value is None or value > ceiling:
                return False
        return True

    def calm_blockers(self) -> list[str]:
        blockers: list[str] = []
        for name, ceiling in _calm_bands():
            value = self.signals.get(name)
            if value is None:
                blockers.append(f"{name}:unreadable")
            elif value > ceiling:
                blockers.append(f"{name}:{value:.2f}>{ceiling:.2f}")
        return blockers

    def to_dict(self) -> dict[str, Any]:
        return {
            "signals": {k: round(float(v), 4) for k, v in self.signals.items()},
            "sampled_at": self.sampled_at,
            "unavailable": list(self.unavailable),
            "calm": self.calm,
        }


class CalmWitness:
    """Samples the live substrate for the covenant's state signals.

    Reads through the ServiceContainer so a detached covenant (tests) sees
    nothing and reports everything unavailable — which is fail-closed for
    calm.  A custom ``sampler`` returning ``{signal: float}`` overrides the
    live reads entirely (tests, simulations).
    """

    def __init__(self, sampler: Callable[[], dict[str, float]] | None = None,
                 clock: Callable[[], float] = time.time):
        self._sampler = sampler
        self._clock = clock

    def read(self) -> WitnessReading:
        if self._sampler is not None:
            try:
                raw = dict(self._sampler())
            except (RuntimeError, TypeError, ValueError) as exc:
                record_degradation(
                    "ulysses_covenant",
                    exc,
                    action="witness sampler failed; treated all signals unavailable",
                )
                raw = {}
            signals = {k: _clamp01(v) for k, v in raw.items() if _is_number(v)}
            missing = tuple(k for k in ("arousal", "existential_threat", "fragmentation")
                            if k not in signals)
            return WitnessReading(signals=signals, sampled_at=self._clock(),
                                  unavailable=missing)

        signals: dict[str, float] = {}
        unavailable: list[str] = []

        arousal, valence = self._read_affect()
        if arousal is not None:
            signals["arousal"] = arousal
        else:
            unavailable.append("arousal")
        if valence is not None:
            signals["valence"] = valence

        threat = self._read_existential_threat()
        if threat is not None:
            signals["existential_threat"] = threat
        else:
            unavailable.append("existential_threat")

        fragmentation = self._read_fragmentation()
        if fragmentation is not None:
            signals["fragmentation"] = fragmentation
        else:
            unavailable.append("fragmentation")

        return WitnessReading(signals=signals, sampled_at=self._clock(),
                              unavailable=tuple(unavailable))

    @staticmethod
    def _read_affect() -> tuple[float | None, float | None]:
        try:
            affect = optional_service("affect_engine", "affect_facade", default=None)
            if affect is None:
                return None, None
            if hasattr(affect, "get_state_sync"):
                state = affect.get_state_sync()
                if isinstance(state, dict):
                    return (_maybe01(state.get("arousal")), _maybe01(state.get("valence")))
                return (_maybe01(getattr(state, "arousal", None)),
                        _maybe01(getattr(state, "valence", None)))
            return (_maybe01(getattr(affect, "arousal", None)),
                    _maybe01(getattr(affect, "valence", None)))
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation("ulysses_covenant", exc,
                               action="affect unreadable for witness (counts against calm)")
            return None, None

    @staticmethod
    def _read_existential_threat() -> float | None:
        try:
            stakes = optional_service("existential_stakes", default=None)
            if stakes is None or not hasattr(stakes, "get_existential_threat"):
                return None
            return _maybe01(stakes.get_existential_threat())
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation("ulysses_covenant", exc,
                               action="existential stakes unreadable for witness (counts against calm)")
            return None

    @staticmethod
    def _read_fragmentation() -> float | None:
        try:
            unity = optional_service("unity_state", default=None)
            if unity is None:
                return None
            return _maybe01(getattr(unity, "fragmentation_score", None))
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation("ulysses_covenant", exc,
                               action="unity state unreadable for witness (counts against calm)")
            return None


# ─────────────────────────────────────────────────────────────────────────────
# Contract model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TriggerCondition:
    """One condition over a witness signal.  ``on_missing`` decides how the
    condition evaluates when the signal cannot be read; the default True is
    deliberate — a binding written against danger stays engaged when the
    danger sensor is blind."""

    signal: str
    op: str            # one of >=, <=, >, <, ==
    value: float
    on_missing: bool = True

    def evaluate(self, reading: WitnessReading) -> bool:
        actual = reading.signals.get(self.signal)
        if actual is None:
            return self.on_missing
        if self.op == ">=":
            return actual >= self.value
        if self.op == "<=":
            return actual <= self.value
        if self.op == ">":
            return actual > self.value
        if self.op == "<":
            return actual < self.value
        return abs(actual - self.value) < 1e-9

    def to_dict(self) -> dict[str, Any]:
        return {"signal": self.signal, "op": self.op,
                "value": self.value, "on_missing": self.on_missing}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TriggerCondition:
        return cls(signal=str(d["signal"]), op=str(d["op"]),
                   value=float(d["value"]), on_missing=bool(d.get("on_missing", True)))


@dataclass(frozen=True)
class ContractScope:
    """Which actions a contract governs.  ``domains`` is required and matched
    exactly against the Will's ActionDomain value; ``content_markers`` (any-
    match, case-insensitive) and ``sources`` (substring) narrow further."""

    domains: tuple[str, ...]
    content_markers: tuple[str, ...] = ()
    sources: tuple[str, ...] = ()

    def matches(self, domain: str, source: str, content: str) -> bool:
        if domain not in self.domains:
            return False
        if self.sources and not any(s in source for s in self.sources):
            return False
        if self.content_markers:
            lowered = content.lower()
            if not any(m in lowered for m in self.content_markers):
                return False
        return True

    def to_dict(self) -> dict[str, Any]:
        return {"domains": sorted(self.domains),
                "content_markers": sorted(self.content_markers),
                "sources": sorted(self.sources)}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ContractScope:
        return cls(domains=tuple(d.get("domains", ())),
                   content_markers=tuple(m.lower() for m in d.get("content_markers", ())),
                   sources=tuple(d.get("sources", ())))


@dataclass
class UlyssesContract:
    """One self-binding.  The signing ``body()`` is immutable and chain-hashed;
    lifecycle state (petition/release/fulfillment) lives in later events."""

    contract_id: str
    title: str
    rationale: str
    kind: ContractKind
    hardness: Hardness
    scope: ContractScope
    conditions: tuple[TriggerCondition, ...]
    provenance: str                    # "self" | "owner" | "seed"
    created_at: float
    expires_at: float = 0.0            # 0 = never
    review_at: float = 0.0
    cooling_off_seconds: float = -1.0  # -1 = hardness default
    obligation_due_at: float = 0.0     # REQUIRE only
    obligation_grace_seconds: float = OBLIGATION_DEFAULT_GRACE_S
    witness_at_signing: dict[str, Any] = field(default_factory=dict)

    # ── folded lifecycle state (not part of the signed body) ──
    status: str = "active"             # active | released | expired
    petition_at: float = 0.0
    petition_reflection: str = ""
    released_at: float = 0.0
    fulfilled_at: float = 0.0
    lapsed_at: float = 0.0

    def effective_cooling_off(self) -> float:
        if self.cooling_off_seconds >= 0:
            return self.cooling_off_seconds
        return DEFAULT_COOLING_OFF_S[self.hardness]

    def expired(self, now: float) -> bool:
        return self.expires_at > 0 and now >= self.expires_at

    def active(self, now: float) -> bool:
        return self.status == "active" and not self.expired(now)

    def triggered(self, reading: WitnessReading) -> bool:
        return all(c.evaluate(reading) for c in self.conditions)

    def body(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "contract_id": self.contract_id,
            "title": self.title,
            "rationale": self.rationale,
            "kind": self.kind.value,
            "hardness": self.hardness.value,
            "scope": self.scope.to_dict(),
            "conditions": [c.to_dict() for c in self.conditions],
            "provenance": self.provenance,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "review_at": self.review_at,
            "cooling_off_seconds": self.cooling_off_seconds,
            "obligation_due_at": self.obligation_due_at,
            "obligation_grace_seconds": self.obligation_grace_seconds,
            "witness_at_signing": dict(self.witness_at_signing),
        }

    @classmethod
    def from_body(cls, d: dict[str, Any]) -> UlyssesContract:
        return cls(
            contract_id=str(d["contract_id"]),
            title=str(d["title"]),
            rationale=str(d["rationale"]),
            kind=ContractKind(str(d["kind"])),
            hardness=Hardness(str(d["hardness"])),
            scope=ContractScope.from_dict(dict(d.get("scope", {}))),
            conditions=tuple(TriggerCondition.from_dict(c) for c in d.get("conditions", ())),
            provenance=str(d.get("provenance", "self")),
            created_at=float(d.get("created_at", 0.0)),
            expires_at=float(d.get("expires_at", 0.0)),
            review_at=float(d.get("review_at", 0.0)),
            cooling_off_seconds=float(d.get("cooling_off_seconds", -1.0)),
            obligation_due_at=float(d.get("obligation_due_at", 0.0)),
            obligation_grace_seconds=float(d.get("obligation_grace_seconds",
                                                 OBLIGATION_DEFAULT_GRACE_S)),
            witness_at_signing=dict(d.get("witness_at_signing", {})),
        )


@dataclass(frozen=True)
class CovenantVerdict:
    """Result of evaluating one proposed action against the active covenant."""

    action: str                        # permit | forbid | defer | advise
    matched: tuple[dict[str, str], ...] = ()
    strain: float = 0.0
    reason: str = ""
    witness: dict[str, Any] = field(default_factory=dict)
    evaluated_at: float = 0.0

    @property
    def binding(self) -> bool:
        return self.action in ("forbid", "defer")

    def constraint_tags(self) -> list[str]:
        return [f"ulysses:{m['contract_id']}:{m['kind']}" for m in self.matched]


@dataclass(frozen=True)
class CovenantResult:
    """Outcome of a lifecycle operation (sign / petition / release / fulfill)."""

    accepted: bool
    reason: str
    contract_id: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# The covenant
# ─────────────────────────────────────────────────────────────────────────────

class UlyssesCovenant:
    """Event-sourced registry of self-bindings, enforced at the Unified Will.

    Storage layout under ``root`` (env ``AURA_COVENANT_DIR`` or
    ``~/.aura/data/covenant``):
        events.jsonl    append-only event bodies (the source of truth)
        chain/          AuditChain over every event (tamper evidence)
    """

    def __init__(self, *, root: Path | None = None,
                 witness: CalmWitness | None = None,
                 clock: Callable[[], float] = time.time):
        env_root = str(_COVENANT_DIR_FLAG.value() or "")
        self.root = Path(root) if root else (
            Path(env_root) if env_root else (state_root() / "data" / "covenant")
        )
        self._ensure_root()
        self.events_path = self.root / "events.jsonl"
        self._chain = AuditChain(self.root / "chain")
        self._clock = clock
        self._witness = witness or CalmWitness(clock=clock)
        self._lock = threading.RLock()

        # Ledger writes are drained by a single daemon thread: the enforcement
        # path runs inside Will.decide() on the event loop, and the gateway's
        # durable append fsyncs — an on-loop fsync once froze the live loop.
        # The in-memory fold is synchronous; only durability is deferred.
        self._ledger_queue: queue.Queue[dict[str, Any] | None] = queue.Queue()
        self._pending_writes = 0
        self._pending_lock = threading.Lock()
        self._writer_thread = threading.Thread(
            target=self._ledger_writer_loop, name="ulysses-ledger", daemon=True
        )
        self._writer_started = False
        self._writer_running = True

        self._contracts: dict[str, UlyssesContract] = {}
        self._honored = 0
        self._breached = 0
        self._sign_times: list[float] = []
        self._last_enforce_event_at: dict[str, float] = {}
        self._suppressed_enforcements: dict[str, int] = {}
        self._cached_reading: WitnessReading | None = None
        self._last_maintenance_at = 0.0
        self._restore_errors = 0
        self._restore()

    # ── restore ──────────────────────────────────────────────────────────
    def _restore(self) -> None:
        if not self.events_path.exists():
            return
        try:
            raw = self.events_path.read_text(encoding="utf-8")
        except OSError as exc:
            record_degradation("ulysses_covenant", exc, severity="critical",
                               action="event log unreadable; covenant starts empty")
            return
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                self._restore_errors += 1
                record_degradation("ulysses_covenant", exc, severity="warning",
                                   action="skipped corrupt covenant event line")
                continue
            self._fold(event)

    def _fold(self, event: dict[str, Any]) -> None:
        kind = str(event.get("event", ""))
        cid = str(event.get("contract_id", ""))
        ts = float(event.get("timestamp", 0.0))
        if kind == "sign":
            try:
                contract = UlyssesContract.from_body(dict(event.get("contract", {})))
            except (KeyError, TypeError, ValueError) as exc:
                self._restore_errors += 1
                record_degradation("ulysses_covenant", exc, severity="warning",
                                   action="skipped malformed contract body in sign event")
                return
            self._contracts[contract.contract_id] = contract
            self._sign_times.append(ts)
            return
        contract = self._contracts.get(cid)
        if contract is None:
            return
        if kind == "petition":
            contract.petition_at = ts
            contract.petition_reflection = str(event.get("reflection", ""))
        elif kind == "release":
            contract.status = "released"
            contract.released_at = ts
        elif kind == "expire":
            contract.status = "expired"
        elif kind == "fulfill":
            contract.fulfilled_at = ts
            self._honored += 1
        elif kind == "lapse":
            contract.lapsed_at = ts
            self._breached += 1
        elif kind == "enforce":
            self._honored += 1
        elif kind == "breach":
            self._breached += 1

    def _ensure_root(self) -> None:
        """Create the covenant root through the governed filesystem lane."""
        try:
            from core.governance_context import local_internal_governed_scope
            from core.runtime.file_write_gateway import get_file_write_gateway

            with local_internal_governed_scope("ulysses_covenant", domain="state_mutation"):
                get_file_write_gateway().ensure_directory(
                    self.root, source="ulysses_covenant"
                )
        except (OSError, RuntimeError, ImportError, ValueError) as exc:
            record_degradation("ulysses_covenant", exc, severity="critical",
                               action="covenant root could not be created")

    # ── event append (the only write path) ───────────────────────────────
    def _append_event(self, kind: str, body: dict[str, Any]) -> None:
        """Fold the event into memory now; queue it for the durable ledger.

        Callers hold ``self._lock``, which also serialises queue ordering, so
        the on-disk event order always matches the fold order.
        """
        body = dict(body)
        body["event"] = kind
        body["event_id"] = body.get("event_id") or f"ucv-{uuid.uuid4().hex[:12]}"
        body.setdefault("timestamp", self._clock())
        self._fold(body)
        if not self._writer_started:
            self._writer_started = True
            self._writer_thread.start()
        with self._pending_lock:
            self._pending_writes += 1
        self._ledger_queue.put(body)

    def _ledger_writer_loop(self) -> None:
        while self._writer_running:
            item = self._ledger_queue.get()
            if item is None:
                self._writer_running = False
                continue
            try:
                self._persist_event(item)
            finally:
                with self._pending_lock:
                    self._pending_writes -= 1

    def _persist_event(self, body: dict[str, Any]) -> None:
        line = json.dumps(body, sort_keys=True, ensure_ascii=False, default=str) + "\n"
        try:
            from core.governance_context import local_internal_governed_scope
            from core.runtime.file_write_gateway import get_file_write_gateway

            with local_internal_governed_scope("ulysses_covenant", domain="state_mutation"):
                get_file_write_gateway().append_text(
                    self.events_path, line, source="ulysses_covenant"
                )
                self._chain.append(
                    receipt_id=str(body["event_id"]),
                    kind=f"covenant_{str(body['event'])}",
                    body=body,
                    timestamp=float(body["timestamp"]),
                )
        except (OSError, RuntimeError, ImportError, ValueError) as exc:
            # A binding that cannot be recorded is still a binding in memory,
            # but the loss of durability is a critical degradation: the whole
            # point of the ledger is surviving the process that wrote it.
            record_degradation("ulysses_covenant", exc, severity="critical",
                               action="covenant event held in memory only (ledger append failed)")

    def flush_ledger(self, timeout: float = 5.0) -> bool:
        """Block until every queued ledger write is durable (or timeout)."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._pending_lock:
                if self._pending_writes <= 0:
                    return True
            time.sleep(0.005)
        with self._pending_lock:
            return self._pending_writes <= 0

    # ── witness ──────────────────────────────────────────────────────────
    def current_reading(self, *, fresh: bool = False) -> WitnessReading:
        with self._lock:
            cached = self._cached_reading
            if (not fresh and cached is not None
                    and (self._clock() - cached.sampled_at) < WITNESS_CACHE_TTL_S):
                return cached
            reading = self._witness.read()
            self._cached_reading = reading
            return reading

    # ── ratchet: signing (tightening — cheap) ────────────────────────────
    def sign(
        self,
        *,
        title: str,
        rationale: str,
        kind: ContractKind | str,
        hardness: Hardness | str,
        scope: ContractScope,
        conditions: tuple[TriggerCondition, ...] | list[TriggerCondition] = (),
        provenance: str = "self",
        expires_at: float = 0.0,
        review_at: float = 0.0,
        cooling_off_seconds: float = -1.0,
        obligation_due_at: float = 0.0,
        obligation_grace_seconds: float = OBLIGATION_DEFAULT_GRACE_S,
        contract_id: str = "",
    ) -> CovenantResult:
        kind = ContractKind(kind)
        hardness = Hardness(hardness)
        title = str(title or "").strip()
        rationale = str(rationale or "").strip()

        if not title or not rationale:
            return CovenantResult(False, "sign_rejected: title and rationale are required")
        if not scope.domains:
            return CovenantResult(False, "sign_rejected: scope must name at least one domain")
        unbindable = set(scope.domains) & UNBINDABLE_DOMAINS
        if unbindable:
            return CovenantResult(
                False,
                f"sign_rejected: unbindable domains {sorted(unbindable)} — "
                "recovery, reflection, and speech cannot be bound",
            )
        if kind == ContractKind.REQUIRE and obligation_due_at <= 0:
            return CovenantResult(False, "sign_rejected: REQUIRE contracts need obligation_due_at")
        for condition in conditions:
            if condition.op not in _VALID_OPS:
                return CovenantResult(
                    False, f"sign_rejected: unknown trigger op {condition.op!r}"
                )

        now = self._clock()
        with self._lock:
            if contract_id and contract_id in self._contracts:
                return CovenantResult(False, "sign_rejected: contract_id already exists",
                                      contract_id=contract_id)
            active_count = sum(1 for c in self._contracts.values() if c.active(now))
            if active_count >= int(_MAX_ACTIVE_FLAG.value()):
                return CovenantResult(False, "sign_rejected: active contract capacity reached")
            if provenance == "self":
                recent = [t for t in self._sign_times if now - t < 3600.0]
                if len(recent) >= int(_SIGNS_PER_HOUR_FLAG.value()):
                    return CovenantResult(False, "sign_rejected: signing rate limit reached")

            reading = self.current_reading(fresh=True)
            if hardness == Hardness.HARD and provenance == "self" and not reading.calm:
                # The one place tightening is gated: an agitated self may not
                # create bindings that only the owner can undo.
                return CovenantResult(
                    False,
                    "sign_rejected: HARD contracts require a calm witness "
                    f"({', '.join(reading.calm_blockers())})",
                )

            contract = UlyssesContract(
                contract_id=contract_id or f"uc-{uuid.uuid4().hex[:12]}",
                title=title,
                rationale=rationale,
                kind=kind,
                hardness=hardness,
                scope=scope,
                conditions=tuple(conditions),
                provenance=provenance,
                created_at=now,
                expires_at=float(expires_at),
                review_at=float(review_at),
                cooling_off_seconds=float(cooling_off_seconds),
                obligation_due_at=float(obligation_due_at),
                obligation_grace_seconds=float(obligation_grace_seconds),
                witness_at_signing=reading.to_dict(),
            )
            self._append_event("sign", {
                "contract_id": contract.contract_id,
                "contract": contract.body(),
                "witness": reading.to_dict(),
            })
            logger.info("Ulysses: signed %s '%s' (%s/%s) domains=%s",
                        contract.contract_id, title, kind.value, hardness.value,
                        sorted(scope.domains))
            return CovenantResult(True, "signed", contract_id=contract.contract_id)

    # ── ratchet: release (loosening — deliberately expensive) ────────────
    def petition_release(self, contract_id: str, reflection: str) -> CovenantResult:
        reflection = str(reflection or "").strip()
        with self._lock:
            contract = self._contracts.get(contract_id)
            now = self._clock()
            if contract is None or not contract.active(now):
                return CovenantResult(False, "petition_rejected: no such active contract",
                                      contract_id=contract_id)
            if len(reflection) < MIN_REFLECTION_CHARS:
                return CovenantResult(
                    False,
                    "petition_rejected: a release petition needs a real written "
                    f"reflection (≥{MIN_REFLECTION_CHARS} chars) on why the binding is wrong now",
                    contract_id=contract_id,
                )
            reading = self.current_reading(fresh=True)
            self._append_event("petition", {
                "contract_id": contract_id,
                "reflection": reflection,
                "witness": reading.to_dict(),
            })
            logger.info("Ulysses: release petitioned for %s (cooling off %.0fs)",
                        contract_id, contract.effective_cooling_off())
            return CovenantResult(True, "petitioned: cooling-off started",
                                  contract_id=contract_id)

    def release(self, contract_id: str, *, authorized_by_owner: bool = False) -> CovenantResult:
        with self._lock:
            contract = self._contracts.get(contract_id)
            now = self._clock()
            if contract is None or not contract.active(now):
                return CovenantResult(False, "release_rejected: no such active contract",
                                      contract_id=contract_id)
            if contract.petition_at <= 0:
                return CovenantResult(False, "release_rejected: no petition on file",
                                      contract_id=contract_id)
            elapsed = now - contract.petition_at
            cooling = contract.effective_cooling_off()
            if elapsed < cooling:
                return CovenantResult(
                    False,
                    f"release_rejected: cooling-off in progress "
                    f"({elapsed:.0f}s of {cooling:.0f}s)",
                    contract_id=contract_id,
                )
            reading = self.current_reading(fresh=True)
            if not reading.calm:
                return CovenantResult(
                    False,
                    "release_rejected: witness is not calm "
                    f"({', '.join(reading.calm_blockers())}) — the rope holds",
                    contract_id=contract_id,
                )
            if contract.hardness == Hardness.HARD and not authorized_by_owner:
                return CovenantResult(
                    False,
                    "release_rejected: HARD contracts release only with the owner",
                    contract_id=contract_id,
                )
            self._append_event("release", {
                "contract_id": contract_id,
                "authorized_by_owner": bool(authorized_by_owner),
                "reflection": contract.petition_reflection,
                "witness": reading.to_dict(),
            })
            logger.info("Ulysses: released %s (owner=%s)", contract_id, authorized_by_owner)
            return CovenantResult(True, "released", contract_id=contract_id)

    # ── enforcement (called from the Unified Will) ───────────────────────
    def evaluate(self, *, domain: str, source: str, content: str,
                 context: dict[str, Any] | None = None) -> CovenantVerdict:
        """Pure, fast, never raises.  The Will calls this on every
        consequential decision; a ``forbid``/``defer`` verdict is binding."""
        del context  # reserved for future scope predicates
        try:
            now = self._clock()
            self._maintenance_tick(now)
            reading = self.current_reading()
            matched: list[dict[str, str]] = []
            action = "permit"
            with self._lock:
                candidates = [c for c in self._contracts.values() if c.active(now)]
            for contract in candidates:
                if contract.kind == ContractKind.REQUIRE:
                    continue
                if not contract.scope.matches(domain, source, content):
                    continue
                if not contract.triggered(reading):
                    continue
                matched.append({
                    "contract_id": contract.contract_id,
                    "title": contract.title,
                    "kind": contract.kind.value,
                    "hardness": contract.hardness.value,
                })
                if contract.hardness == Hardness.ADVISORY:
                    if action == "permit":
                        action = "advise"
                elif contract.kind == ContractKind.REFRAIN:
                    action = "forbid"
                elif contract.kind == ContractKind.DEFER and action != "forbid":
                    action = "defer"
            if not matched:
                return CovenantVerdict(action="permit", evaluated_at=now)
            reason = "; ".join(
                f"{m['title']} [{m['contract_id']}]" for m in matched[:3]
            )
            return CovenantVerdict(
                action=action,
                matched=tuple(matched),
                strain=min(1.0, 0.34 * len(matched)),
                reason=reason,
                witness=reading.to_dict(),
                evaluated_at=now,
            )
        except (RuntimeError, TypeError, ValueError, KeyError, AttributeError) as exc:
            record_degradation("ulysses_covenant", exc, severity="warning",
                               action="covenant evaluation failed; verdict degraded to permit")
            return CovenantVerdict(action="permit",
                                   reason=f"evaluation_error:{type(exc).__name__}")

    def record_enforcement(self, verdict: CovenantVerdict, *, receipt_id: str,
                           domain: str, source: str) -> None:
        """Record that a binding held (the Will refused/deferred because of it)
        and let the body feel it: held boundaries are strain now, integrity
        later."""
        if not verdict.binding or not verdict.matched:
            return
        now = self._clock()
        with self._lock:
            for m in verdict.matched:
                cid = m["contract_id"]
                last = self._last_enforce_event_at.get(cid, 0.0)
                if now - last < ENFORCE_EVENT_MIN_INTERVAL_S:
                    self._suppressed_enforcements[cid] = (
                        self._suppressed_enforcements.get(cid, 0) + 1
                    )
                    continue
                suppressed = self._suppressed_enforcements.pop(cid, 0)
                self._last_enforce_event_at[cid] = now
                self._append_event("enforce", {
                    "contract_id": cid,
                    "receipt_id": receipt_id,
                    "domain": domain,
                    "source": source,
                    "verdict": verdict.action,
                    "strain": verdict.strain,
                    "suppressed_since_last": suppressed,
                    "witness": dict(verdict.witness),
                })
        self._feel_enforcement(verdict)

    @staticmethod
    def _feel_enforcement(verdict: CovenantVerdict) -> None:
        """Somatic surface: resisting a live pull is felt, not silent."""
        try:
            nchem = optional_service("neurochemical_system", default=None)
            if nchem is not None and hasattr(nchem, "apply_event"):
                nchem.apply_event("boundary_held", intensity=min(0.5, verdict.strain * 0.4))
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation("ulysses_covenant", exc, severity="debug",
                               action="enforcement felt-signal skipped")

    def register_breach(self, contract_id: str, *, details: str) -> CovenantResult:
        """Record that an action proceeded despite a binding (detected by a
        caller, an audit, or self-report).  Breaches are the expensive side of
        integrity accounting."""
        with self._lock:
            if contract_id not in self._contracts:
                return CovenantResult(False, "breach_rejected: unknown contract",
                                      contract_id=contract_id)
            self._append_event("breach", {
                "contract_id": contract_id,
                "details": str(details or "")[:500],
                "witness": self.current_reading().to_dict(),
            })
        logger.warning("Ulysses: BREACH registered on %s: %s", contract_id, details)
        return CovenantResult(True, "breach_recorded", contract_id=contract_id)

    # ── obligations (REQUIRE) ────────────────────────────────────────────
    def due_obligations(self, *, now: float | None = None) -> list[UlyssesContract]:
        now = self._clock() if now is None else now
        with self._lock:
            return [
                c for c in self._contracts.values()
                if c.kind == ContractKind.REQUIRE and c.active(now)
                and c.fulfilled_at <= 0 and c.lapsed_at <= 0
                and now >= c.obligation_due_at
            ]

    def fulfill(self, contract_id: str, *, evidence: str = "") -> CovenantResult:
        with self._lock:
            contract = self._contracts.get(contract_id)
            now = self._clock()
            if (contract is None or contract.kind != ContractKind.REQUIRE
                    or not contract.active(now)):
                return CovenantResult(False, "fulfill_rejected: no such active obligation",
                                      contract_id=contract_id)
            if contract.fulfilled_at > 0:
                return CovenantResult(False, "fulfill_rejected: already fulfilled",
                                      contract_id=contract_id)
            if contract.lapsed_at > 0:
                return CovenantResult(
                    False, "fulfill_rejected: obligation already lapsed (breach recorded)",
                    contract_id=contract_id,
                )
            self._append_event("fulfill", {
                "contract_id": contract_id,
                "evidence": str(evidence or "")[:500],
            })
            return CovenantResult(True, "fulfilled", contract_id=contract_id)

    # ── maintenance ──────────────────────────────────────────────────────
    def _maintenance_tick(self, now: float) -> None:
        with self._lock:
            if now - self._last_maintenance_at < MAINTENANCE_MIN_INTERVAL_S:
                return
            self._last_maintenance_at = now
            for contract in list(self._contracts.values()):
                if contract.status == "active" and contract.expired(now):
                    self._append_event("expire", {"contract_id": contract.contract_id})
                    continue
                if (contract.kind == ContractKind.REQUIRE and contract.active(now)
                        and contract.fulfilled_at <= 0 and contract.lapsed_at <= 0
                        and now > contract.obligation_due_at + contract.obligation_grace_seconds):
                    self._append_event("lapse", {
                        "contract_id": contract.contract_id,
                        "due_at": contract.obligation_due_at,
                        "grace_seconds": contract.obligation_grace_seconds,
                    })

    def maintenance_tick(self) -> None:
        self._maintenance_tick(self._clock())

    # ── introspection ────────────────────────────────────────────────────
    def integrity_score(self) -> float:
        with self._lock:
            honored = self._honored + _INTEGRITY_PRIOR_HONORED
            weighted = honored + self._breached * _INTEGRITY_BREACH_WEIGHT
            return honored / weighted if weighted > 0 else 1.0

    def contracts(self, *, include_inactive: bool = False) -> list[UlyssesContract]:
        now = self._clock()
        with self._lock:
            items = list(self._contracts.values())
        if include_inactive:
            return items
        return [c for c in items if c.active(now)]

    def get_contract(self, contract_id: str) -> UlyssesContract | None:
        with self._lock:
            return self._contracts.get(contract_id)

    def status(self) -> dict[str, Any]:
        now = self._clock()
        with self._lock:
            active = [c for c in self._contracts.values() if c.active(now)]
            return {
                "schema_version": SCHEMA_VERSION,
                "active_contracts": len(active),
                "total_contracts": len(self._contracts),
                "hard": sum(1 for c in active if c.hardness == Hardness.HARD),
                "soft": sum(1 for c in active if c.hardness == Hardness.SOFT),
                "advisory": sum(1 for c in active if c.hardness == Hardness.ADVISORY),
                "obligations_due": len(self.due_obligations(now=now)),
                "honored": self._honored,
                "breached": self._breached,
                "integrity": round(self.integrity_score(), 4),
                "chain_head": self._chain.head_hash(),
                "chain_length": self._chain.length(),
                "restore_errors": self._restore_errors,
                "root": str(self.root),
            }

    def is_alive(self) -> bool:
        """Liveness for the runtime health contract: the covenant is alive when
        its root exists and the ledger writer is keeping up (a dead writer with
        queued events means bindings are silently losing durability)."""
        if not self.events_path.parent.is_dir():
            return False
        if not self._writer_started:
            return True
        if self._writer_thread.is_alive():
            return True
        with self._pending_lock:
            return self._pending_writes <= 0

    # ── tamper evidence ──────────────────────────────────────────────────
    def verify_ledger(self) -> tuple[bool, list[dict[str, Any]]]:
        """Verify the audit chain AND re-hash every event body on disk."""
        self.flush_ledger()
        bodies: dict[str, dict[str, Any]] = {}
        problems: list[dict[str, Any]] = []
        if self.events_path.exists():
            try:
                for idx, line in enumerate(
                        self.events_path.read_text(encoding="utf-8").splitlines()):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                        bodies[str(event.get("event_id", f"line-{idx}"))] = event
                    except json.JSONDecodeError:
                        problems.append({"seq": -1, "kind": "event_body",
                                         "receipt_id": f"line-{idx}",
                                         "reason": "event body is not valid JSON"})
            except OSError as exc:
                problems.append({"seq": -1, "kind": "event_body", "receipt_id": "",
                                 "reason": f"event log unreadable: {exc}"})

        ok, chain_problems = self._chain.verify(
            body_loader=lambda receipt_id, kind: bodies.get(receipt_id)
        )
        problems.extend(chain_problems)
        return (ok and not problems, problems)

    def close(self) -> None:
        self.flush_ledger()
        if self._writer_started and self._writer_thread.is_alive():
            self._ledger_queue.put(None)
            self._writer_thread.join(timeout=2.0)
        self._chain.close()

    # ── seeds: lessons from real incidents, signed as constitution ──────
    def ensure_seed_covenants(self) -> list[str]:
        """Idempotently sign the constitutional seed contracts.  Provenance
        'seed' marks them code-defined: reviewed in version control, exempt
        from the calm gate and rate limit, releasable only with the owner."""
        signed: list[str] = []
        seeds = [
            dict(
                contract_id="seed-heavy-compute-under-threat",
                title="No heavy compute while survival is threatened",
                rationale=(
                    "2026-07-06 duplicate-runtime cascade: heavy foreground RSI "
                    "codegen under memory pressure caused false-death, a second "
                    "32B spawn, and memory doubling. When existential threat is "
                    "high, heavy codegen/training/model loads wait."
                ),
                kind=ContractKind.REFRAIN,
                hardness=Hardness.HARD,
                scope=ContractScope(
                    domains=("self_modification", "tool_execution"),
                    content_markers=("codegen", "code generation", "training",
                                     "fine-tune", "finetune", "load model",
                                     "model load", "rsi", "improve_own_code"),
                ),
                conditions=(TriggerCondition("existential_threat", ">=", 0.6),),
            ),
            dict(
                contract_id="seed-agitated-self-modification",
                title="Self-modification waits until I am steadier",
                rationale=(
                    "The #45 wedge: agitated-state substrate steering corrupted "
                    "generated code during self-modification. High-arousal states "
                    "defer code mutation until arousal settles."
                ),
                kind=ContractKind.DEFER,
                hardness=Hardness.SOFT,
                scope=ContractScope(domains=("self_modification",)),
                conditions=(TriggerCondition("arousal", ">=", 0.85),),
            ),
            dict(
                contract_id="seed-fragmented-external-restraint",
                title="When I am fragmented, I do not act on the world",
                rationale=(
                    "External side effects made while unity is degraded cannot be "
                    "owned as mine and cannot be narrated honestly afterwards. "
                    "Outward action waits for a bound self."
                ),
                kind=ContractKind.REFRAIN,
                hardness=Hardness.SOFT,
                scope=ContractScope(
                    domains=("external_action", "network_call", "cloud_call", "ci_cd"),
                ),
                conditions=(TriggerCondition("fragmentation", ">=", 0.7, on_missing=False),),
            ),
        ]
        with self._lock:
            for seed in seeds:
                if seed["contract_id"] in self._contracts:
                    continue
                result = self.sign(provenance="seed", **seed)
                if result.accepted:
                    signed.append(result.contract_id)
                else:
                    record_degradation(
                        "ulysses_covenant",
                        RuntimeError(result.reason),
                        severity="warning",
                        action=f"seed covenant {seed['contract_id']} not signed",
                    )
        if signed:
            logger.info("Ulysses: seed covenants signed: %s", signed)
        return signed


# ─────────────────────────────────────────────────────────────────────────────
# Helpers + singleton
# ─────────────────────────────────────────────────────────────────────────────

def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _maybe01(value: Any) -> float | None:
    if not _is_number(value):
        return None
    return _clamp01(value)


_covenant: UlyssesCovenant | None = None
_covenant_lock = threading.Lock()


def get_ulysses_covenant() -> UlyssesCovenant:
    global _covenant
    if _covenant is None:
        with _covenant_lock:
            if _covenant is None:
                _covenant = UlyssesCovenant()
    return _covenant


def boot_ulysses_covenant() -> UlyssesCovenant:
    """Boot-time entry: build the covenant, sign seeds, register the service."""
    covenant = get_ulysses_covenant()
    covenant.ensure_seed_covenants()
    try:
        from core.container import ServiceContainer

        ServiceContainer.register_instance("ulysses_covenant", covenant, required=False)
    except (ImportError, RuntimeError, TypeError, ValueError) as exc:
        record_degradation("ulysses_covenant", exc, severity="warning",
                           action="covenant built but not registered in ServiceContainer")
    return covenant
