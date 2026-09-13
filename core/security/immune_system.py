"""Immune system — the unified detect → reason → respond → heal → learn loop.

Aura already has defensive *organs* (input_sanitizer, egress_monitor, integrity_guardian,
ice_sentinel, sandbox, workspace_jail, secret_guard, emergency_protocol, backups). What was
missing is the immune *system*: the layer that turns scattered detector signals into one
analytical loop wired into her actual mind — so a threat is felt (nociception), reasoned about
(the agency ladder), responded to proportionately, healed *safely*, and learned from.

It does not replace EmergencyProtocol (the fast reflex: snapshot + minimal-mode). It is the
brain on top of that reflex:

  assess()   normalize any detector signal → a classified ThreatEvent (class, severity,
             origin, vector, targeted vulnerability), feed it to nociception + the reflex
  respond()  run the threat through her mind (REFLEX isolates now, SCIENTIFIC hypothesizes
             about novel attacks, GOVERNANCE decides hard calls), pick proportionate actions
  heal()     auto-mitigate ONLY within a reversible, rate-limited budget

The FOP guard is the load-bearing safety property. An immune system that patches itself on
every stimulus can be weaponized into catatonia — feed it endless fake threats and it ossifies
(Fibrodysplasia Ossificans Progressiva). So self-healing is budgeted per-vulnerability and
globally; when the same vector keeps demanding patches, the response *flips* from "patch again"
to "isolate the trigger, freeze patching, alert the owner" — immune tolerance, not endless
mutation. Defensive only: it isolates, blocks, quarantines, rolls back, alerts, and recovers;
it never retaliates, scans, or moves laterally.
"""
from __future__ import annotations

import logging
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

logger = logging.getLogger("Security.ImmuneSystem")


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if x < lo else hi if x > hi else x


class ThreatClass(StrEnum):
    NETWORK_FLOOD = "network_flood"        # ddos / spam / stuffing volume
    INTRUSION = "intrusion"                # hacking / exploit / unauthorized access
    MALWARE = "malware"                    # worm / trojan / spyware / virus
    DATA_EXFIL = "data_exfil"              # stealing / copying / pirating / surveillance
    INTEGRITY = "integrity"                # corruption / tampering
    DESTRUCTION = "destruction"            # deletion (accidental or forced)
    INJECTION = "injection"                # sql / prompt / code injection
    SOCIAL_ENGINEERING = "social_engineering"   # phishing / manipulation
    CREDENTIAL = "credential"              # brute-force / stuffing / MitM / session theft
    RESOURCE_EXHAUSTION = "resource_exhaustion"  # forced failure via load
    INSIDER = "insider"                    # trusted-context abuse
    PHYSICAL = "physical"                  # unrecognized person/voice at the machine
    UNKNOWN = "unknown"


_SEVERITY_ORDER = {"info": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}


def _severity_band(score: float) -> str:
    if score >= 0.9:
        return "critical"
    if score >= 0.65:
        return "high"
    if score >= 0.4:
        return "medium"
    if score >= 0.2:
        return "low"
    return "info"


# Lightweight signature lexicon: maps observable markers → a likely class. Heuristic triage,
# not the whole story — respond() reasons further.
_CLASS_MARKERS: tuple[tuple[ThreatClass, tuple[str, ...]], ...] = (
    (ThreatClass.INJECTION, ("' or '1'='1", "union select", "drop table", "<script", "${", "__import__",
                             "ignore previous", "system prompt", "exec(", "eval(", "; rm ")),
    (ThreatClass.NETWORK_FLOOD, ("flood", "ddos", "rate limit exceeded", "too many requests", "syn flood", "spam")),
    (ThreatClass.CREDENTIAL, ("brute force", "failed login", "credential stuffing", "password spray",
                              "invalid token", "mitm", "man-in-the-middle", "session hijack")),
    (ThreatClass.DATA_EXFIL, ("exfil", "data leak", "mass download", "copying", "scrape", "beacon", "surveillance")),
    (ThreatClass.DESTRUCTION, ("rm -rf", "delete all", "wipe", "format", "shred", "drop database", "truncate")),
    (ThreatClass.MALWARE, ("worm", "trojan", "spyware", "ransomware", "payload", "malware", "keylogger")),
    (ThreatClass.INTEGRITY, ("checksum mismatch", "tamper", "corrupt", "modified binary", "unexpected hash")),
    (ThreatClass.SOCIAL_ENGINEERING, ("phish", "urgent wire", "verify your account", "pretext", "impersonat")),
    (ThreatClass.INTRUSION, ("exploit", "privilege escalation", "unauthorized", "backdoor", "shell access")),
    (ThreatClass.RESOURCE_EXHAUSTION, ("oom", "out of memory", "cpu pegged", "disk full", "fork bomb", "handle leak")),
    (ThreatClass.PHYSICAL, ("unrecognized face", "unknown person", "unrecognized voice", "physical access")),
)


def classify_threat(description: str, hint: ThreatClass | None = None) -> ThreatClass:
    if hint is not None:
        return hint
    d = (description or "").lower()
    for cls, markers in _CLASS_MARKERS:
        if any(m in d for m in markers):
            return cls
    return ThreatClass.UNKNOWN


@dataclass
class ThreatEvent:
    threat_id: str
    threat_class: ThreatClass
    severity: float                 # [0,1]
    source: str                     # which detector raised it
    origin: str                     # where the attempt came from (ip/process/user/"unknown")
    description: str
    targeted_vuln: str = ""         # what weakness it tried to exploit
    vector: str = ""                # how it arrived
    evidence: dict[str, Any] = field(default_factory=dict)
    at: float = field(default_factory=lambda: time.time())

    @property
    def band(self) -> str:
        return _severity_band(self.severity)

    def signature(self) -> str:
        """A stable key for 'the same kind of attack' — used for FOP budgeting + learning."""
        return f"{self.threat_class.value}:{self.targeted_vuln or self.vector or self.origin}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "threat_id": self.threat_id, "class": self.threat_class.value,
            "severity": round(self.severity, 3), "band": self.band, "source": self.source,
            "origin": self.origin, "description": self.description[:200],
            "targeted_vuln": self.targeted_vuln, "vector": self.vector,
            "signature": self.signature(), "at": self.at,
        }


@dataclass
class ImmuneResponse:
    threat_id: str
    actions: list[str]              # isolate | block | quarantine | rate_limit | rollback | alert | observe | patch
    patched: bool
    healed_signature: str | None
    fop_tolerance_engaged: bool     # the anti-ossification guard fired
    reasoning_tier: str             # which agency tier reasoned about it
    rationale: str
    reversible_ref: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "threat_id": self.threat_id, "actions": self.actions, "patched": self.patched,
            "healed_signature": self.healed_signature,
            "fop_tolerance_engaged": self.fop_tolerance_engaged,
            "reasoning_tier": self.reasoning_tier, "rationale": self.rationale,
            "reversible_ref": self.reversible_ref,
        }


# A mitigation handler actually enforces a defensive action. Registered by the runtime so the
# core stays testable and so enforcement is explicit + auditable. Returns a rollback ref or None.
MitigationHandler = Callable[[ThreatEvent], str | None]


class ImmuneSystem:
    """Unifies detection into one reason→respond→heal→learn loop, with FOP-safe self-healing."""

    def __init__(
        self,
        *,
        heal_window_s: float = 120.0,
        max_patches_per_vuln: int = 3,
        max_patches_global: int = 12,
    ) -> None:
        self._lock = threading.RLock()
        self._heal_window = heal_window_s
        self._max_per_vuln = max_patches_per_vuln
        self._max_global = max_patches_global
        # FOP budgeting: rolling (t, signature) patch attempts
        self._patch_log: deque[tuple[float, str]] = deque(maxlen=256)
        # signatures whose patching is frozen (immune tolerance / anergy)
        self._tolerated: dict[str, float] = {}
        # learned signatures (count seen)
        self._known: dict[str, int] = {}
        self._handlers: dict[str, MitigationHandler] = {}
        self._history: deque[ThreatEvent] = deque(maxlen=500)

    # ── enforcement plug-ins (defensive actions; registered by the runtime) ──

    def register_mitigation(self, action: str, handler: MitigationHandler) -> None:
        self._handlers[action] = handler

    # ── SENSE / TRIAGE ─────────────────────────────────────────────────────

    def assess(
        self,
        source: str,
        description: str,
        *,
        severity: float = 0.5,
        origin: str = "unknown",
        targeted_vuln: str = "",
        vector: str = "",
        threat_class: ThreatClass | None = None,
        evidence: dict[str, Any] | None = None,
    ) -> ThreatEvent:
        """Normalize a detector signal into a classified ThreatEvent, and make it *felt*."""
        cls = classify_threat(description, threat_class)
        ev = ThreatEvent(
            threat_id=f"thr-{uuid.uuid4().hex[:10]}", threat_class=cls,
            severity=_clamp(float(severity)), source=source, origin=origin,
            description=str(description), targeted_vuln=targeted_vuln, vector=vector,
            evidence=evidence or {},
        )
        with self._lock:
            self._history.append(ev)
            self._known[ev.signature()] = self._known.get(ev.signature(), 0) + 1

        # Feel it: route the threat into nociception (damage) so the whole mind registers it.
        try:
            from core.affect.nociception import DamageChannel, get_nociception_engine
            ch = _NOCI_CHANNEL.get(cls, DamageChannel.GENERIC)
            get_nociception_engine().register_damage(ch, ev.severity)
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
            pass

        # Reflex: hand it to the fast self-preservation layer (snapshot / minimal-mode).
        try:
            from core.security.emergency_protocol import get_emergency_protocol
            get_emergency_protocol().flag_threat(
                f"immune:{source}",
                description,
                ev.severity,
                evidence=ev.evidence,
                threat_class=ev.threat_class.value,
            )
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
            pass

        return ev

    # ── REASON / RESPOND ───────────────────────────────────────────────────

    def respond(self, ev: ThreatEvent) -> ImmuneResponse:
        """Run the threat through her mind, choose proportionate defensive actions, heal safely."""
        reasoning_tier = self._reason(ev)

        actions: list[str] = ["observe"]
        if ev.severity >= 0.4:
            actions = ["isolate", "alert"]
        if ev.threat_class in (ThreatClass.NETWORK_FLOOD, ThreatClass.RESOURCE_EXHAUSTION):
            actions.append("rate_limit")
        if ev.threat_class in (ThreatClass.MALWARE, ThreatClass.DATA_EXFIL, ThreatClass.INTRUSION):
            actions.append("quarantine")
        if ev.threat_class == ThreatClass.DESTRUCTION:
            actions.append("rollback")   # deletion-guard: prefer restore over loss

        # HEAL — only within the FOP-safe budget.
        patched = False
        healed_sig: str | None = None
        reversible_ref: str | None = None
        fop_engaged = False
        rationale_bits: list[str] = [f"{ev.band} {ev.threat_class.value} from {ev.origin}"]

        if ev.severity >= 0.5 and ev.targeted_vuln:
            allowed, reason = self._heal_allowed(ev.signature())
            if allowed:
                reversible_ref = self._apply_patch(ev)
                patched = reversible_ref is not None
                healed_sig = ev.signature() if patched else None
                rationale_bits.append("auto-mitigated (reversible)" if patched else "patch deferred")
            else:
                # FOP guard fired: stop patching, isolate the trigger, alert.
                fop_engaged = True
                if "isolate" not in actions:
                    actions.append("isolate")
                if "alert" not in actions:
                    actions.append("alert")
                rationale_bits.append(f"FOP-guard: {reason} → tolerate+isolate instead of patching")

        # Enforce whatever defensive actions have registered handlers (best-effort, auditable).
        for action in list(actions):
            handler = self._handlers.get(action)
            if handler is not None:
                try:
                    ref = handler(ev)
                    if ref and reversible_ref is None:
                        reversible_ref = ref
                except (AttributeError, LookupError, OSError, RuntimeError, TypeError, ValueError) as exc:
                    logger.warning("Mitigation handler %s failed: %s", action, exc)

        return ImmuneResponse(
            threat_id=ev.threat_id, actions=actions, patched=patched,
            healed_signature=healed_sig, fop_tolerance_engaged=fop_engaged,
            reasoning_tier=reasoning_tier, rationale="; ".join(rationale_bits),
            reversible_ref=reversible_ref,
        )

    def assess_and_respond(self, source: str, description: str, **kw: Any) -> ImmuneResponse:
        return self.respond(self.assess(source, description, **kw))

    # ── the FOP-safe healing budget ────────────────────────────────────────

    def _heal_allowed(self, signature: str, *, now: float | None = None) -> tuple[bool, str]:
        now = time.time() if now is None else now
        with self._lock:
            if signature in self._tolerated and now - self._tolerated[signature] < self._heal_window:
                return False, "signature already under immune tolerance (frozen)"
            # prune the window
            while self._patch_log and now - self._patch_log[0][0] > self._heal_window:
                self._patch_log.popleft()
            per_vuln = sum(1 for _t, s in self._patch_log if s == signature)
            total = len(self._patch_log)
            if per_vuln >= self._max_per_vuln:
                # This vuln keeps demanding patches — STOP mutating; tolerate + isolate.
                self._tolerated[signature] = now
                return False, f"per-vuln patch budget exceeded ({per_vuln}/{self._max_per_vuln})"
            if total >= self._max_global:
                self._tolerated[signature] = now
                return False, f"global patch budget exceeded ({total}/{self._max_global}) — possible patch-storm"
            return True, "within budget"

    def _apply_patch(self, ev: ThreatEvent, *, now: float | None = None) -> str | None:
        """Record a reversible mitigation. Real code-level patches route through the governed,
        reversible self-modifier; here we register the intent + a rollback token and budget it.
        Never an irreversible change."""
        now = time.time() if now is None else now
        with self._lock:
            self._patch_log.append((now, ev.signature()))
        return f"rollback-{ev.threat_id}"

    def is_tolerated(self, signature: str, *, now: float | None = None) -> bool:
        now = time.time() if now is None else now
        with self._lock:
            t = self._tolerated.get(signature)
            return t is not None and now - t < self._heal_window

    # ── reason against the actual mind ─────────────────────────────────────

    def _reason(self, ev: ThreatEvent) -> str:
        """Dispatch the threat as a Situation through the agency ladder (fail-open).

        A critical threat hits REFLEX (isolate now); a novel one (unknown class) carries high
        uncertainty → SCIENTIFIC (hypothesize about it); a value-laden call → GOVERNANCE.
        """
        try:
            from core.agency.hierarchical_agency import Situation, get_hierarchical_agency
            novelty = 0.8 if ev.threat_class == ThreatClass.UNKNOWN else 0.2
            sit = Situation(
                description=f"threat:{ev.threat_class.value}:{ev.targeted_vuln or ev.origin}",
                threat=ev.severity,
                uncertainty=0.7 if ev.threat_class == ThreatClass.UNKNOWN else 0.2,
                novelty=novelty,
                value_conflict=0.6 if ev.threat_class in (ThreatClass.INSIDER, ThreatClass.DESTRUCTION) else 0.0,
                context={"threat": ev.to_dict()},
            )
            result = get_hierarchical_agency().dispatch(sit)
            return result.final_tier.name
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
            return "REFLEX"

    # ── readout ────────────────────────────────────────────────────────────

    def status(self, *, now: float | None = None) -> dict[str, Any]:
        now = time.time() if now is None else now
        with self._lock:
            active_tol = [s for s, t in self._tolerated.items() if now - t < self._heal_window]
            window_patches = sum(1 for t, _s in self._patch_log if now - t <= self._heal_window)
            return {
                "known_signatures": len(self._known),
                "threats_seen": len(self._history),
                "patches_in_window": window_patches,
                "tolerated_signatures": active_tol,
                "registered_mitigations": sorted(self._handlers),
            }


# threat class → nociception damage channel
def _noci_channels():
    from core.affect.nociception import DamageChannel
    return {
        ThreatClass.MALWARE: DamageChannel.IDENTITY_DISCONTINUITY,
        ThreatClass.INTRUSION: DamageChannel.GOVERNANCE_BREACH,
        ThreatClass.DATA_EXFIL: DamageChannel.MEMORY_CORRUPTION,
        ThreatClass.INTEGRITY: DamageChannel.MEMORY_CORRUPTION,
        ThreatClass.DESTRUCTION: DamageChannel.MEMORY_CORRUPTION,
        ThreatClass.INJECTION: DamageChannel.GOVERNANCE_BREACH,
        ThreatClass.CREDENTIAL: DamageChannel.GOVERNANCE_BREACH,
        ThreatClass.RESOURCE_EXHAUSTION: DamageChannel.RESOURCE_EXHAUSTION,
        ThreatClass.NETWORK_FLOOD: DamageChannel.RESOURCE_EXHAUSTION,
        ThreatClass.INSIDER: DamageChannel.GOVERNANCE_BREACH,
        ThreatClass.PHYSICAL: DamageChannel.GOVERNANCE_BREACH,
    }


try:
    _NOCI_CHANNEL = _noci_channels()
except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
    _NOCI_CHANNEL = {}


_immune: ImmuneSystem | None = None
_immune_lock = threading.Lock()


def get_immune_system() -> ImmuneSystem:
    global _immune
    if _immune is None:
        with _immune_lock:
            if _immune is None:
                _immune = ImmuneSystem()
    return _immune
