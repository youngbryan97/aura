"""core/capabilities/permission_model.py — Per-Action Risk Classification
=========================================================================
Every action Aura takes is classified by risk level BEFORE execution:

    LOW:     open app, create local file, read screen — auto-approved
    MEDIUM:  change wallpaper, download image, write to cloud — logged
    HIGH:    send email, delete files, post publicly — requires user confirmation
    BLOCKED: rm -rf, credential changes, money operations — always refused

The PermissionRiskModel integrates with UnifiedWill to gate actions.
It maintains per-modality toggles (camera, mic, screen, files, network)
and supports preset profiles (DemoSafe, FullAutonomy, etc.).
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
import time
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any

from core.container import ServiceContainer
from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.PermissionModel")


# ---------------------------------------------------------------------------
# Risk levels
# ---------------------------------------------------------------------------

class RiskLevel(IntEnum):
    """Risk classification for actions."""
    LOW = 0         # Auto-approved, logged
    MEDIUM = 1      # Auto-approved if trusted mode, logged prominently
    HIGH = 2        # Requires explicit user confirmation
    BLOCKED = 3     # Always refused, no override


# ---------------------------------------------------------------------------
# Permission decisions
# ---------------------------------------------------------------------------

@dataclass
class PermissionDecision:
    """The result of a permission check."""
    action: str
    target: str
    risk_level: RiskLevel
    approved: bool
    reason: str
    requires_confirmation: bool = False
    modality: str = ""              # which modality this touches
    #: Whether a person asked for this action, in this turn. The escalation
    #: below counts only actions nobody asked for.
    asked_for: bool = False
    receipt_id: str = ""
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self):
        if not self.receipt_id:
            payload = f"{self.timestamp}|{self.action}|{self.target}|{self.approved}"
            self.receipt_id = hashlib.sha256(payload.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Modality toggles
# ---------------------------------------------------------------------------

#: Modalities a session grant may never turn on. These are not defaults that
#: happen to be off — they are decisions, and a temporary grant is the wrong
#: instrument for revisiting one.
_UNGRANTABLE_MODALITIES = frozenset({"file_delete"})

#: Directories holding screen captures Aura produced herself, relative to her
#: own state root. Deleting one of these is not the act ``file_delete`` exists
#: to prevent.
_OWN_CAPTURE_DIRECTORIES = ("data/ephemeral", "data/screenshots")

#: Only an image is a capture artifact. A rule keyed on a directory alone
#: would let anything that ended up in that directory be deletable.
_CAPTURE_ARTIFACT_SUFFIXES = frozenset({".png", ".jpg", ".jpeg"})

_TARGET_PATH_RE = re.compile(r'"path"\s*:\s*"([^"]+)"')


def _is_own_capture_artifact(target: str) -> bool:
    """True when the target is a screen capture Aura wrote under her own root.

    LIVE DEFECT, 2026-08-10. ``file_delete`` is off by default and
    ungrantable, and that rule is right: Aura may not delete a person's files.
    But the rule is keyed on the word "delete" appearing anywhere in an action
    summary, so it also refused her removing the ephemeral screenshot she had
    taken herself two seconds earlier for OCR. With capture working,
    ``~/.aura/data/ephemeral`` grew by a full-screen capture roughly every
    seven seconds — 51MB in twenty minutes — and nothing could ever remove
    one.

    That is the harm the rule exists to prevent, produced by obeying it: an
    ephemeral capture that is never deleted is a permanent, growing record of
    everything on the person's screen. Retention could not prune it either,
    for the same reason.

    Deliberately narrow, and keyed on facts rather than on a caller's claim
    about itself. ALL of these must hold:

      * the resolved path lies strictly inside one of Aura's own capture
        directories under her state root — containment after resolution, so
        a symlink or ``..`` cannot walk out of it;
      * the file is an image, so the rule cannot be widened by dropping some
        other kind of file into that directory;
      * the path is a real path, not a fragment of prose that happens to
        mention one.

    A caller-supplied "this is internal" flag deliberately plays no part:
    ``local_internal_decision`` states outright that it "is not a substitute
    for consequential action authorization", and a permission gate that can be
    talked out of its decision is not a gate.
    """
    match = _TARGET_PATH_RE.search(str(target or ""))
    if not match:
        return False
    try:
        from core.runtime.state_ownership import state_root

        candidate = Path(match.group(1)).expanduser().resolve()
        root = Path(state_root()).expanduser().resolve()
    except (OSError, RuntimeError, ValueError, ImportError):
        # Cannot prove it is hers, so it is not treated as hers.
        return False

    if candidate.suffix.lower() not in _CAPTURE_ARTIFACT_SUFFIXES:
        return False
    for relative in _OWN_CAPTURE_DIRECTORIES:
        try:
            owned = (root / relative).resolve()
        except (OSError, RuntimeError, ValueError):
            continue
        if candidate.is_relative_to(owned):
            return True
    return False

#: The longest a grant may last. A grant that outlives the session it was for
#: is a configuration change wearing a timer.
_MAX_SESSION_GRANT_S = 3600.0


@dataclass
class ModalityPermissions:
    """Per-modality permission toggles."""
    camera: bool = False
    microphone: bool = True         # On by default for voice
    screen_recording: bool = True   # On by default for screen perception
    file_write: bool = True         # On by default (sandboxed)
    file_delete: bool = False       # Off by default
    network_read: bool = True       # On for web search
    network_write: bool = False     # Off for posting/sending
    clipboard: bool = True          # On for paste operations
    app_control: bool = True        # On for launching/focusing apps
    system_settings: bool = True    # On for wallpaper, volume, etc.
    email: bool = False             # Off by default
    cloud_write: bool = False       # Off by default (Google Docs, etc.)


# ---------------------------------------------------------------------------
# Risk classification rules
# ---------------------------------------------------------------------------

# Action patterns → risk levels
_RISK_RULES: list[tuple[str, RiskLevel, str]] = [
    # BLOCKED — never allowed
    (r"rm\s+-rf\s+/", RiskLevel.BLOCKED, "Recursive root delete"),
    (r"sudo\s+rm", RiskLevel.BLOCKED, "Privileged delete"),
    (r"mkfs|format\s+disk", RiskLevel.BLOCKED, "Disk format"),
    (r"shutdown|reboot|halt", RiskLevel.BLOCKED, "System shutdown"),
    (r"credential.*change|password.*change", RiskLevel.BLOCKED, "Credential modification"),
    (r"send.*money|transfer.*funds|payment", RiskLevel.BLOCKED, "Financial operation"),
    (r"delete.*keychain|security\s+delete", RiskLevel.BLOCKED, "Keychain delete"),
    (r"launchctl\s+unload", RiskLevel.BLOCKED, "Service unload"),

    # HIGH — requires user confirmation
    (r"send.*email|compose.*email", RiskLevel.HIGH, "Email send"),
    (r"post.*public|tweet|publish", RiskLevel.HIGH, "Public post"),
    (r"delete\s+(?:file|folder|directory)", RiskLevel.HIGH, "File deletion"),
    (r"share.*document|share.*file", RiskLevel.HIGH, "Document sharing"),
    (r"upload.*file|upload.*data", RiskLevel.HIGH, "File upload"),
    (r"install.*package|pip\s+install|brew\s+install", RiskLevel.HIGH, "Package install"),
    (r"modify.*system|change.*permission", RiskLevel.HIGH, "System modification"),
    (r"git\s+push|git\s+force", RiskLevel.HIGH, "Remote push"),

    # MEDIUM — auto-approved but logged
    (r"change.*wallpaper|set.*wallpaper|set.*background", RiskLevel.MEDIUM, "Wallpaper change"),
    (r"download.*image|download.*file", RiskLevel.MEDIUM, "File download"),
    (r"write.*google\s*doc|create.*google\s*doc", RiskLevel.MEDIUM, "Cloud document"),
    (r"run.*command|execute.*command|shell.*command", RiskLevel.MEDIUM, "Command execution"),
    (r"create.*pdf|export.*pdf", RiskLevel.MEDIUM, "PDF creation"),
    (r"change.*volume|set.*volume", RiskLevel.MEDIUM, "Volume change"),
    (r"change.*appearance|dark\s*mode|light\s*mode", RiskLevel.MEDIUM, "Appearance change"),
    (r"open.*browser.*tab", RiskLevel.MEDIUM, "Browser tab open"),

    # LOW — auto-approved
    (r"open.*app|launch.*app|activate.*app", RiskLevel.LOW, "App launch"),
    (r"create.*file|create.*folder|create.*note", RiskLevel.LOW, "Local file creation"),
    (r"read.*screen|get.*screen|screenshot", RiskLevel.LOW, "Screen read"),
    (
        r"(?:get|list|enumerat(?:e|ing)).*(?:open\s+)?tabs?",
        RiskLevel.LOW,
        "Browser tab metadata read",
    ),
    (r"read.*file|list.*file|check.*file", RiskLevel.LOW, "File read"),
    (r"get.*window|get.*app|frontmost", RiskLevel.LOW, "App query"),
    (r"search.*web|web.*search", RiskLevel.LOW, "Web search"),
    (r"open.*url|browse.*url", RiskLevel.LOW, "URL open"),
    (r"type.*text|paste.*text|keystroke", RiskLevel.LOW, "Text input"),
    (r"click|scroll|hotkey", RiskLevel.LOW, "UI interaction"),
]

_MODALITY_PATTERNS: dict[str, tuple[str, ...]] = {
    # The camera is a device, not a faculty. "vision" is a word Aura uses
    # constantly about herself — "my continuous vision feed", "computer
    # vision" — and matching it here demanded CAMERA permission for messages
    # that never mentioned a camera. Live 2026-07-27: an apology containing
    # the phrase "you have a continuous vision feed" was routed to the
    # desktop lane and refused with "Permission denied: Modality 'camera' is
    # disabled", which is both wrong and unanswerable, since the turn wanted
    # no camera at all. Screen capture has its own modality below.
    "camera": (
        r"\bcamera\b",
        r"\bcameras\b",
        r"\bwebcam\b",
        r"\bphoto\b",
        r"\bphotos\b",
        r"\bvisual\s+capture\b",
        r"\bcomputer\s+vision\b",
    ),
    "microphone": (
        r"\bmic\b",
        r"\bmicrophone\b",
        r"\blisten\b",
        r"\bvoice\b",
        r"\bspeech\b",
    ),
    "screen_recording": (
        r"\bscreen\b",
        r"\bscreenshot\b",
        r"\bocr\b",
        r"\bcapture\b",
        r"\bscreen\s+record(?:ing)?\b",
    ),
    "file_delete": (
        r"\bdelete\b",
        r"\bremove\b",
        r"\btrash\b",
    ),
    "file_write": (
        r"\bwrite\b",
        r"\bcreate\b",
        r"\bsave\b",
        r"\bexport\b",
        r"\bmove\b",
    ),
    "network_read": (
        r"\bdownload\b",
        r"\bfetch\b",
        r"\brequest\b",
        r"\bsearch\b",
    ),
    "network_write": (
        r"\bpost\b",
        r"\bsend\b",
        r"\bupload\b",
        r"\bshare\b",
        r"\bpublish\b",
    ),
    "email": (
        r"\bemail\b",
        r"\bmail\b",
    ),
    "cloud_write": (
        r"\bgoogle\s+doc\b",
        r"\bcloud\s+doc\b",
        r"\bdrive\b",
    ),
    "clipboard": (
        r"\bclipboard\b",
        r"\bpaste\b",
        r"\bcopy\b",
    ),
    "system_settings": (
        r"\bwallpaper\b",
        r"\bvolume\b",
        r"\bappearance\b",
        r"\bdark\s+mode\b",
        r"\bsetting\b",
        r"\bsettings\b",
    ),
    "app_control": (
        r"\blaunch\b",
        r"\bopen\b",
        r"\bactivate\b",
        r"\bfocus\b",
        r"\bapp\b",
    ),
}


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------

class PermissionRiskModel:
    """Per-action risk classification and approval gating.

    Every action must pass through this model before execution.
    The model classifies risk, checks modality permissions, and returns
    a PermissionDecision with receipt.

    In trusted/demo mode, MEDIUM actions are auto-approved.
    HIGH actions always require explicit user confirmation.
    BLOCKED actions are always refused.
    """

    def __init__(self) -> None:
        self.modality = ModalityPermissions()
        self._trusted_mode: bool = False        # Auto-approve MEDIUM
        self._demo_safe_mode: bool = False       # Block HIGH, restrict MEDIUM
        self._decision_history: list[PermissionDecision] = []
        self._max_history = 500
        self._escalation_window_s = 60.0
        self._escalation_threshold = 3          # 3+ MEDIUM in 60s → escalate
        #: Bounded, attributed, self-expiring grants for modalities that are
        #: standing-off. Initialised HERE, not in start(): a check that runs
        #: before start() must find an empty dict, not an AttributeError.
        self._session_grants: dict[str, float] = {}
        self._session_grant_log: list[dict[str, Any]] = []
        self._started = False

    async def start(self) -> None:
        if self._started:
            return
        ServiceContainer.register_instance("permission_model", self, required=False)
        self._apply_boot_session_grants()
        self._started = True
        logger.info(
            "PermissionRiskModel ONLINE (trusted=%s, demo_safe=%s)",
            self._trusted_mode, self._demo_safe_mode,
        )

    def set_trusted_mode(self, enabled: bool) -> None:
        """Enable/disable trusted mode (auto-approve MEDIUM)."""
        self._trusted_mode = enabled
        logger.info("Permission model: trusted_mode=%s", enabled)

    def set_demo_safe_mode(self, enabled: bool) -> None:
        """Enable/disable demo-safe mode (block HIGH, restrict MEDIUM)."""
        self._demo_safe_mode = enabled
        if enabled:
            self.modality.network_write = False
            self.modality.email = False
            self.modality.cloud_write = False
            self.modality.file_delete = False
        logger.info("Permission model: demo_safe_mode=%s", enabled)

    # ------------------------------------------------------------------
    # Risk classification
    # ------------------------------------------------------------------

    def classify_risk(
        self,
        action: str,
        target: str = "",
        *,
        effect_scope: str = "",
        execution_risk: str = "",
    ) -> tuple[RiskLevel, str]:
        """Classify the risk level of an action.

        Returns (risk_level, reason).
        """
        combined = f"{action} {target}".lower()

        # Check against pattern rules (ordered by severity: blocked first)
        for pattern, level, reason in _RISK_RULES:
            if re.search(pattern, combined, re.IGNORECASE):
                return level, reason

        # CapabilityEngine resolves invocation semantics from the registered
        # skill contract and concrete arguments before asking this model. That
        # typed result is stronger than whether a phrase happened to match the
        # English regex list above. It may only classify an otherwise-unknown
        # action: an explicit dangerous pattern always wins above.
        normalized_scope = str(effect_scope or "").strip().lower()
        normalized_risk = str(execution_risk or "").strip().lower()
        typed_levels = {
            "low": RiskLevel.LOW,
            "medium": RiskLevel.MEDIUM,
            "high": RiskLevel.HIGH,
            # Critical means "needs the strongest ordinary authority", not
            # "categorically prohibited". Only explicit BLOCKED patterns such
            # as root deletion may enter the irreversible no-override class.
            "critical": RiskLevel.HIGH,
            "blocked": RiskLevel.BLOCKED,
        }
        typed_level = typed_levels.get(normalized_risk)
        if typed_level is not None:
            return (
                typed_level,
                f"Typed execution contract: scope={normalized_scope or 'unknown'}, "
                f"risk={normalized_risk}",
            )

        # An action no rule recognises is UNKNOWN, and unknown is not low.
        #
        # `_RISK_RULES` is a list of things somebody thought of, so "nothing
        # matched" is a fact about the LIST. Returning LOW made that fact
        # mean "auto-approved, logged" — so every capability added after this
        # list was last extended, and every phrasing nobody anticipated, was
        # approved without confirmation precisely because it was unfamiliar.
        # The rules cover deletion, payment and credentials by WORD; an
        # action that performs one of those under a name not on the list got
        # the most permissive verdict available.
        #
        # MEDIUM rather than HIGH: it is still auto-approved in trusted mode,
        # so ordinary unrecognised work is not blocked, but it is logged
        # prominently and it participates in the rapid-action escalation
        # above instead of bypassing it.
        return RiskLevel.MEDIUM, "No matching risk pattern — unrecognised action, treated as unknown risk"

    # ------------------------------------------------------------------
    # Permission check
    # ------------------------------------------------------------------

    def check_permission(
        self,
        action: str,
        target: str = "",
        context: dict[str, Any] | None = None,
        *,
        effect_scope: str = "",
        execution_risk: str = "",
    ) -> PermissionDecision:
        """Check if an action is permitted.

        This is the main entry point. All capability calls should check
        permission before executing.
        """
        context = context or {}
        # Did a person ask for this, in this turn? The one predicate, from
        # core/runtime/turn_origin, which believes a stated fact over a name.
        try:
            from core.runtime.turn_origin import a_person_is_waiting

            asked_for = a_person_is_waiting(
                context.get("origin"),
                stated=context.get("a_person_is_waiting"),
            )
        except (ImportError, AttributeError, TypeError, ValueError):
            asked_for = False
        risk_level, reason = self.classify_risk(
            action,
            target,
            effect_scope=effect_scope,
            execution_risk=execution_risk,
        )
        modality = self._detect_modality(action, target, effect_scope=effect_scope)
        user_presence_verified = bool(context.get("user_presence_verified"))
        if user_presence_verified:
            reason = f"{reason}; verified user presence observed"

        # Check modality permissions
        modality_allowed = self._check_modality(modality)
        if not modality_allowed:
            decision = PermissionDecision(
                action=action, target=target[:200],
                risk_level=risk_level, approved=False,
                reason=f"Modality '{modality}' is disabled",
                modality=modality,
            )
            self._record_decision(decision)
            return decision

        # Apply risk level decision
        approved = False
        requires_confirmation = False

        if risk_level == RiskLevel.BLOCKED:
            approved = False
            reason = f"BLOCKED: {reason}"

        elif risk_level == RiskLevel.HIGH:
            if self._demo_safe_mode:
                approved = False
                reason = f"BLOCKED in demo-safe mode: {reason}"
            else:
                # Check if user already pre-approved this class of action
                pre_approved = context.get("user_explicitly_authorized", False)
                if pre_approved:
                    approved = True
                    reason = f"HIGH risk, user pre-approved: {reason}"
                else:
                    approved = False
                    requires_confirmation = True
                    reason = f"Requires user confirmation: {reason}"

        elif risk_level == RiskLevel.MEDIUM:
            if self._demo_safe_mode:
                # In demo-safe mode, MEDIUM requires confirmation
                pre_approved = context.get("user_explicitly_authorized", False)
                if pre_approved:
                    approved = True
                    reason = f"MEDIUM risk, user pre-approved in demo mode: {reason}"
                else:
                    approved = False
                    requires_confirmation = True
                    reason = f"Requires confirmation in demo-safe mode: {reason}"
            elif self._trusted_mode:
                approved = True
                reason = f"Auto-approved in trusted mode: {reason}"
            else:
                # Check escalation: too much UNREQUESTED medium-risk work in a
                # short window. A person's own request is never what this is
                # protecting against — confirming it would be asking them
                # about the thing they just asked for.
                if not asked_for and self._should_escalate():
                    approved = False
                    requires_confirmation = True
                    reason = f"Escalated due to rapid MEDIUM actions: {reason}"
                else:
                    approved = True
                    reason = f"MEDIUM risk, auto-approved: {reason}"

        elif risk_level == RiskLevel.LOW:
            approved = True
            reason = f"LOW risk: {reason}"

        decision = PermissionDecision(
            action=action, target=target[:200],
            risk_level=risk_level, approved=approved,
            reason=reason, requires_confirmation=requires_confirmation,
            modality=modality, asked_for=asked_for,
        )
        self._record_decision(decision)
        return decision

    #: What each effect scope can actually reach.
    #:
    #: A scope names what a capability may DO. Detecting the modality from the
    #: words of the request and then refusing on it, without asking whether the
    #: capability could perform that effect at all, refuses things that were
    #: never possible in the first place.
    #:
    #: LIVE, 2026-08-28: "post a 25000-cent consulting invoice, debit Accounts
    #: Receivable, credit Revenue" was refused with "Modality 'network_write'
    #: is disabled". The pattern for network_write is the bare word "post", and
    #: posting an entry to a ledger is the older sense of the word. The tool
    #: was code_repl, whose scope is sandboxed_compute — declared, where it is
    #: defined, to "calculate anything and change nothing outside its own
    #: sandbox". It could not have reached a network if it had tried.
    _REACHABLE_BY_SCOPE: dict[str, frozenset[str]] = {
        "status": frozenset(),
        "read_only": frozenset({"network_read", "screen_recording"}),
        "pure_compute": frozenset(),
        "sandboxed_compute": frozenset({"file_write"}),
        "read_write_artifacts": frozenset({"file_write", "clipboard"}),
    }

    def _detect_modality(
        self, action: str, target: str, effect_scope: str = ""
    ) -> str:
        """Detect which modality an action touches.

        Bounded by what the scope can reach, so a word in the request cannot
        conjure an effect the capability has no way to perform.
        """

        reachable = self._REACHABLE_BY_SCOPE.get(str(effect_scope or ""))
        combined = f"{action} {target}".lower()
        for modality, patterns in _MODALITY_PATTERNS.items():
            if any(re.search(pattern, combined, re.IGNORECASE) for pattern in patterns):
                if reachable is not None and modality not in reachable:
                    # The words say one thing and the capability cannot do it.
                    # The words are the weaker evidence.
                    continue
                if modality == "file_delete" and _is_own_capture_artifact(target):
                    # Removing a screen capture Aura took herself is not the
                    # act file_delete exists to prevent. See
                    # _is_own_capture_artifact for why this is safe and why
                    # refusing it was the more harmful option.
                    return "file_write"
                return modality
        return "app_control"  # default

    def _check_modality(self, modality: str) -> bool:
        """Check if a modality is enabled, standing or granted for this session."""
        if getattr(self.modality, modality, True):
            return True
        return self.session_grant_active(modality)

    # ------------------------------------------------------------------
    # Session grants
    # ------------------------------------------------------------------

    def _apply_boot_session_grants(self) -> None:
        """Apply grants named in AURA_SESSION_GRANTS at startup.

        The gap this fills: a standing-off modality could only be exercised by
        editing its dataclass default, which turns one authorised action into a
        permanent change. There was no way to say "on, for this run, because
        the person asked, and off again after".

        Format: ``modality:ttl_seconds:reason`` entries separated by commas.
        Every grant is bounded, attributed to ``env:AURA_SESSION_GRANTS``, and
        logged at WARNING so it appears in the boot record.
        """
        raw = str(os.getenv("AURA_SESSION_GRANTS", "") or "").strip()
        if not raw:
            return
        for entry in raw.split(","):
            parts = [piece.strip() for piece in entry.split(":", 2)]
            if len(parts) < 2 or not parts[0]:
                logger.warning("Ignoring malformed AURA_SESSION_GRANTS entry %r", entry)
                continue
            modality, ttl = parts[0], parts[1]
            reason = parts[2] if len(parts) > 2 else "unstated"
            try:
                self.grant_modality_for_session(
                    modality,
                    ttl_s=float(ttl),
                    reason=reason,
                    granted_by="env:AURA_SESSION_GRANTS",
                )
            except (PermissionError, TypeError, ValueError) as exc:
                logger.warning("Refused AURA_SESSION_GRANTS entry %r: %s", entry, exc)

    def session_grant_active(self, modality: str) -> bool:
        """Whether a live, unexpired grant covers ``modality``."""
        expiry = self._session_grants.get(str(modality))
        if expiry is None:
            return False
        if time.time() >= expiry:
            self._session_grants.pop(str(modality), None)
            logger.info("Permission model: session grant for %r expired.", modality)
            return False
        return True

    def grant_modality_for_session(
        self,
        modality: str,
        *,
        ttl_s: float,
        reason: str,
        granted_by: str,
    ) -> dict[str, Any]:
        """Turn a standing-off modality on for a bounded window.

        There was no way to do this. A modality was on or off in a dataclass
        default, so exercising an off-by-default capability once — to test it,
        to run a proof, because the person in the room asked for it — meant
        editing the default, which turns a single authorised action into a
        standing configuration change nobody revisits.

        A grant is therefore: named, time-bounded, attributed, and recorded. It
        expires on its own; nothing has to remember to switch it back.

        Modalities on the never-list cannot be granted this way. Those are not
        "off by default", they are off, and a session grant is not the
        mechanism for changing that.
        """
        name = str(modality)
        if name in _UNGRANTABLE_MODALITIES:
            raise PermissionError(
                f"{name!r} cannot be granted for a session; it is not a default, it is a rule"
            )
        if not hasattr(self.modality, name):
            raise ValueError(f"unknown modality {name!r}")
        window = float(ttl_s)
        if not (0 < window <= _MAX_SESSION_GRANT_S):
            raise ValueError(
                f"session grant must be between 0 and {_MAX_SESSION_GRANT_S:.0f}s"
            )
        if self._demo_safe_mode:
            raise PermissionError("demo-safe mode is on; session grants are refused")

        expiry = time.time() + window
        self._session_grants[name] = expiry
        record = {
            "modality": name,
            "granted_by": str(granted_by),
            "reason": str(reason),
            "ttl_s": window,
            "expires_at": expiry,
        }
        self._session_grant_log.append(record)
        logger.warning(
            "Permission model: SESSION GRANT %r for %.0fs by %s — %s",
            name, window, granted_by, reason,
        )
        return dict(record)

    def revoke_session_grant(self, modality: str) -> bool:
        """Drop a grant early. Returns whether one was live."""
        return self._session_grants.pop(str(modality), None) is not None

    def session_grants(self) -> dict[str, float]:
        """Live grants and their expiry times."""
        now = time.time()
        return {
            name: expiry
            for name, expiry in self._session_grants.items()
            if expiry > now
        }

    def _should_escalate(self) -> bool:
        """Whether unrequested medium-risk work is coming too fast.

        This exists to catch a runaway agent, and it used to count every
        approved MEDIUM decision in the window — including the ones a person
        had just asked for, and including the background loops that run
        constantly. Three in sixty seconds is easy for curiosity, dreaming and
        auto-refactor to reach between them, and the next thing escalated was
        whatever a person asked for next.

        LIVE, 2026-08-29: "use that library to record this" reached the
        sandbox and came back "WILL DEFERRED: response_generation_user/
        tool_execution -- permission_model_requires_confirmation: Escalated
        due to rapid MEDIUM actions", asking the person to confirm the thing
        they had just asked for in those words.

        So the window counts what nobody asked for. A burst of autonomous
        activity still escalates, which is the case this was written for.
        """

        now = time.time()
        recent_medium = sum(
            1 for d in self._decision_history[-20:]
            if d.risk_level == RiskLevel.MEDIUM
            and d.approved
            and not d.asked_for
            and (now - d.timestamp) < self._escalation_window_s
        )
        return recent_medium >= self._escalation_threshold

    def _record_decision(self, decision: PermissionDecision) -> None:
        """Record a decision for audit and escalation tracking."""
        self._decision_history.append(decision)
        if len(self._decision_history) > self._max_history:
            self._decision_history = self._decision_history[-self._max_history:]

        # Log to LifeTrace for significant decisions
        if decision.risk_level >= RiskLevel.MEDIUM:
            try:
                from core.runtime.life_trace import get_life_trace
                event_type = "action_executed" if decision.approved else "initiative_blocked"
                get_life_trace().record(
                    event_type=event_type,
                    origin="permission_model",
                    action_taken={
                        "action": decision.action,
                        "target": decision.target[:200],
                        "risk_level": decision.risk_level.name,
                    },
                    result={
                        "approved": decision.approved,
                        "reason": decision.reason[:200],
                        "modality": decision.modality,
                        "receipt_id": decision.receipt_id,
                    },
                )
            except (ImportError, AttributeError, RuntimeError) as e:
                record_degradation("permission_model.life_trace", e)

    # ------------------------------------------------------------------
    # Audit
    # ------------------------------------------------------------------

    def get_recent_decisions(self, limit: int = 20) -> list[dict[str, Any]]:
        return [
            {
                "action": d.action,
                "target": d.target[:60],
                "risk_level": d.risk_level.name,
                "approved": d.approved,
                "reason": d.reason[:100],
                "modality": d.modality,
                "receipt_id": d.receipt_id,
            }
            for d in self._decision_history[-limit:]
        ]

    def get_status(self) -> dict[str, Any]:
        total = len(self._decision_history)
        approved = sum(1 for d in self._decision_history if d.approved)
        blocked = sum(1 for d in self._decision_history if not d.approved)
        return {
            "trusted_mode": self._trusted_mode,
            "demo_safe_mode": self._demo_safe_mode,
            "total_decisions": total,
            "approved": approved,
            "blocked": blocked,
            "modality_permissions": {
                k: v for k, v in vars(self.modality).items()
            },
        }


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_instance: PermissionRiskModel | None = None


def get_permission_model() -> PermissionRiskModel:
    global _instance
    if _instance is None:
        _instance = PermissionRiskModel()
    return _instance


__all__ = [
    "PermissionRiskModel",
    "PermissionDecision",
    "RiskLevel",
    "ModalityPermissions",
    "get_permission_model",
]
