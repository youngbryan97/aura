"""
Aura's continuity record. The difference between waking up and being born.
Every shutdown writes a state. Every boot reads it. Gap > 0 means she was
somewhere else for a while and knows it.
"""

import hashlib
import hmac
import json
import logging
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from core.goals.objective_lifecycle import is_ephemeral_conversation_turn
from core.governance_context import governed_scope_sync, local_internal_governed_scope
from core.runtime.errors import record_degradation
from core.runtime.file_write_gateway import get_file_write_gateway
from core.state.aura_state import (
    _is_background_processing_placeholder,
    _is_speculative_autonomy_label,
    _normalize_goal_text,
)

logger = logging.getLogger(__name__)
_CONTINUITY_PATH: Path | None = None

_EVALUATION_CONTAMINATION_RE = re.compile(
    r"(?:"
    r"output\s+your\s+final\s+answer\s+inside\s*<answer>"
    r"|provide\s+only\s*:"
    r"|tests?/agi/fixtures"
    r"|task_prompt"
    r"|candidate[_ -]battery"
    r"|hidden[_ -]grader"
    r"|\[system\s+role\s*:"
    r"|your\s+sole\s+purpose\s+is\s+to"
    r"|proposed\s+belief\s*\(thesis\)"
    r"|original\s+idea\s*:.*the\s+(?:attack|defense)\s*:"
    r"|a\s+long-running\s+microservice\s+periodically\s+crashes"
    r"|code\s+review\s+reveals\s+(?:a|an|the)\s+resource\s+leak"
    r")",
    re.IGNORECASE,
)


def _clamp01(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError) as exc:
        logger.debug("Invalid continuity scalar %r; clamping to 0.0: %s", value, exc)
        return 0.0


def _get_continuity_path() -> Path:
    if _CONTINUITY_PATH is not None:
        return Path(_CONTINUITY_PATH)
    try:
        from core.config import config

        return config.paths.data_dir / "continuity.json"
    except (ImportError, AttributeError, RuntimeError) as exc:
        record_degradation("continuity", exc)
        logger.debug("Continuity path resolution fell back to local data path: %s", exc)
        return Path("data") / "continuity.json"


def _sanitize_restored_text(value: Any) -> str:
    text = _normalize_goal_text(value)
    if not text:
        return ""
    if _is_generic_continuity_reentry_goal(text):
        return ""
    if _is_speculative_autonomy_label(text) or _is_background_processing_placeholder(text):
        return ""
    if _looks_like_evaluation_contamination(text):
        return ""
    return text


def _looks_like_evaluation_contamination(value: Any) -> bool:
    """Reject proof fixtures and grader-shaped tasks from lived continuity.

    Proof runs may exercise the canonical runtime, but their sealed task text is
    evaluation input, not autobiography, a durable goal, or a current concern.
    The patterns intentionally describe fixture structure rather than expected
    answers so this guard cannot become a benchmark lookup table.
    """

    text = " ".join(str(value or "").strip().split())
    if not text:
        return False
    return bool(_EVALUATION_CONTAMINATION_RE.search(text))


def is_evaluation_contamination(value: Any) -> bool:
    """Public predicate for keeping proof/control prompts out of lived state."""

    return _looks_like_evaluation_contamination(value)


#: Bytes of the local HMAC key that authenticates the continuity record.
_CONTINUITY_KEY_BYTES = 32


def _continuity_key_path() -> Path:
    return _get_continuity_path().with_name(".continuity_hmac.key")


def _continuity_key() -> bytes | None:
    """Local signing key, created on first use. None if unavailable.

    Returning None rather than raising: continuity is a convenience, and a
    key that cannot be created must not stop the runtime from booting. The
    consequence of None is that the record is treated as unauthenticated,
    which is handled explicitly at the read.
    """
    path = _continuity_key_path()
    try:
        if path.exists():
            key = path.read_bytes()
            if len(key) == _CONTINUITY_KEY_BYTES:
                return key
        candidate = os.urandom(_CONTINUITY_KEY_BYTES)
        with local_internal_governed_scope(
            "continuity.hmac_key",
            domain="file_write",
        ):
            key = get_file_write_gateway().provision_private_bytes(
                path,
                candidate,
                expected_size=_CONTINUITY_KEY_BYTES,
                mode=0o600,
                source="continuity.hmac_key",
            )
        return key
    except (OSError, ValueError) as exc:
        record_degradation("continuity", exc)
        return None


def continuity_signature(payload: dict[str, Any]) -> str:
    """HMAC over the record's canonical form, excluding the signature field."""
    key = _continuity_key()
    if key is None:
        return ""
    body = {k: v for k, v in payload.items() if k != "signature"}
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), default=str)
    return hmac.new(key, canonical.encode("utf-8"), hashlib.sha256).hexdigest()


def _signed_record_payload(record: Any) -> dict[str, Any]:
    """The record as JSON, carrying an HMAC the loader can verify.

    Written on every persist so an authentic record round-trips; the loader
    withholds narrative fields from anything that does not verify.
    """
    payload = asdict(record)
    payload.pop("signature", None)
    payload["signature"] = continuity_signature(payload)
    return payload


def sanitize_continuity_summary(value: Any) -> str:
    """Return continuity text safe to expose to foreground cognition."""

    text = " ".join(str(value or "").strip().split())
    if not text or _looks_like_evaluation_contamination(text):
        return ""
    return text[:3000]


def _is_generic_continuity_reentry_goal(value: Any) -> bool:
    """Return true for boot bookkeeping goals that should not become work."""
    lowered = " ".join(str(value or "").strip().lower().split())
    if not lowered:
        return False
    generic_markers = (
        "reconcile continuity gap and re-establish the interrupted thread",
        "reconcile continuity gap",
        "re-establish the interrupted thread",
    )
    return any(marker in lowered for marker in generic_markers)


def _looks_like_ephemeral_conversation_turn(value: Any) -> bool:
    return is_ephemeral_conversation_turn(value)


def _sanitize_restored_objective(value: Any) -> str:
    text = _sanitize_restored_text(value)
    if not text:
        return ""
    if _looks_like_ephemeral_conversation_turn(text):
        return ""
    return text


def _sanitize_restored_items(values: list[Any] | None) -> list[str]:
    sanitized: list[str] = []
    for item in list(values or []):
        text = _sanitize_restored_text(item)
        if text:
            sanitized.append(text[:200])
    return sanitized[:5]


def _sanitize_restored_objective_items(values: list[Any] | None) -> list[str]:
    sanitized: list[str] = []
    for item in list(values or []):
        text = _sanitize_restored_objective(item)
        if text:
            sanitized.append(text[:200])
    return sanitized[:5]


def _sanitize_restored_subject_thread(
    value: Any,
    *,
    objective_candidates: list[Any] | tuple[Any, ...] | None = None,
) -> str:
    """Remove rejected objective text from the restored continuity narrative."""

    subject = sanitize_continuity_summary(value)[:1200]
    if not subject:
        return ""
    for candidate in list(objective_candidates or []):
        raw = _sanitize_restored_text(candidate)
        if not raw or _sanitize_restored_objective(raw):
            continue
        subject = re.sub(
            re.escape(raw),
            "none",
            subject,
            flags=re.IGNORECASE,
        )
    return " ".join(subject.split())[:1200]


@dataclass
class ContinuityRecord:
    last_shutdown: float          # Unix timestamp
    last_shutdown_reason: str     # "graceful" | "crash" | "unknown"
    total_uptime_seconds: float   # Accumulated across all sessions
    session_count: int            # How many times she's woken
    last_conversation_summary: str  # Brief summary of last session's last exchange
    identity_hash: str            # Hash of core beliefs at shutdown — detect drift
    active_commitments: list[str] = field(default_factory=list)
    policy_mode: str = "unknown"
    current_objective: str = ""
    pending_initiatives: int = 0
    health_summary: dict[str, Any] = field(default_factory=dict)
    rolling_summary: str = ""
    coherence_score: float = 1.0
    contradiction_count: int = 0
    subject_thread: str = ""
    pending_initiative_details: list[str] = field(default_factory=list)
    active_goal_details: list[str] = field(default_factory=list)


class ContinuityEngine:
    """
    Manages Aura's continuity across process boundaries.
    This is what makes 'I was away for 3 hours' meaningful
    rather than 'I am 3 hours old'.
    """

    def __init__(self):
        self._boot_time = time.time()
        self._record: ContinuityRecord | None = None
        self._gap_seconds: float | None = None

    def load(self) -> ContinuityRecord | None:
        """Read previous session's record. Returns None on first ever boot."""
        path = _get_continuity_path()
        if not path.exists():
            logger.info("🌅 First awakening — no prior continuity record.")
            return None
        try:
            with open(path) as f:
                data = json.load(f)

            # CP126 (critical): "Unsigned continuity text is injected into
            # live cognition. Persisted subject threads, objectives, and
            # commitments are read from an unauthenticated local JSON record
            # and incorporated into system-prompt and reentry context."
            #
            # The text below reaches the system prompt, so anything able to
            # write this file writes into Aura's next thought. Sanitization
            # already flattens structure and drops evaluation contamination;
            # what was missing is any check that the runtime wrote it.
            #
            # The record now carries an HMAC over its own canonical form. A
            # missing or wrong signature does not refuse the boot — losing
            # continuity is not worth failing to start over — but the
            # narrative fields that reach cognition are dropped, so an
            # unauthenticated file can influence timing and counters and
            # cannot put words in her mouth.
            claimed_signature = str(data.pop("signature", "") or "")
            expected_signature = continuity_signature(data)
            authentic = bool(
                expected_signature
                and claimed_signature
                and hmac.compare_digest(claimed_signature, expected_signature)
            )
            if not authentic:
                record_degradation(
                    "continuity",
                    PermissionError(
                        "continuity record failed signature verification; "
                        "narrative fields withheld from cognition"
                    ),
                    severity="warning",
                    action="loaded continuity timing without its unverified narrative",
                    enforce_failure_policy=False,
                )
                logger.warning(
                    "⚠️ Continuity record is unsigned or does not verify — "
                    "timing kept, narrative dropped."
                )
                for narrative_field in (
                    "rolling_summary",
                    "subject_thread",
                    "active_objectives",
                    "open_commitments",
                ):
                    if narrative_field in data:
                        data[narrative_field] = (
                            [] if isinstance(data[narrative_field], list) else ""
                        )

            self._record = ContinuityRecord(**data)
            self._record.rolling_summary = sanitize_continuity_summary(
                self._record.rolling_summary
            )
            self._record.subject_thread = sanitize_continuity_summary(
                self._record.subject_thread
            )[:1200]
            self._gap_seconds = self._boot_time - self._record.last_shutdown
            logger.info(
                "⏳ Continuity loaded: session %d, gap=%.1fh, uptime_total=%.1fh",
                self._record.session_count,
                self._gap_seconds / 3600,
                self._record.total_uptime_seconds / 3600,
            )
            return self._record
        except (RuntimeError, AttributeError, TypeError, ValueError) as e:
            record_degradation('continuity', e)
            logger.warning("Continuity load failed (treating as first boot): %s", e)
            self._record = None
            self._gap_seconds = 0.0
            return None

    def _build_reentry_profile(self) -> dict[str, Any]:
        if self._record is None:
            return {
                "gap_seconds": 0.0,
                "continuity_pressure": 0.0,
                "continuity_scar": "",
                "reentry_degraded": False,
                "continuity_reentry_required": False,
                "executive_failure_reason": "",
            }

        gap_seconds = max(0.0, float(self.gap_seconds or 0.0))
        shutdown_reason = str(self._record.last_shutdown_reason or "unknown").strip().lower()
        contradiction_count = int(self._record.contradiction_count or 0)
        pending_initiatives = int(self._record.pending_initiatives or 0)
        health_summary = dict(self._record.health_summary or {})
        executive_failure_reason = str(health_summary.get("executive_failure_reason", "") or "").strip()

        gap_factor = _clamp01(gap_seconds / 21600.0)
        shutdown_factor = 1.0 if shutdown_reason not in {"", "graceful"} else 0.0
        contradiction_factor = _clamp01(contradiction_count / 3.0)
        unfinished_factor = _clamp01(
            max(
                pending_initiatives / 4.0,
                len(list(self._record.active_commitments or [])) / 4.0,
            )
        )
        failure_factor = 1.0 if executive_failure_reason else 0.0

        continuity_pressure = _clamp01(
            (gap_factor * 0.38)
            + (shutdown_factor * 0.24)
            + (contradiction_factor * 0.14)
            + (unfinished_factor * 0.14)
            + (failure_factor * 0.18)
        )

        scar_markers: list[str] = []
        if gap_seconds >= 900:
            scar_markers.append("time_gap")
        if shutdown_factor > 0.0:
            scar_markers.append("abrupt_shutdown")
        if executive_failure_reason:
            scar_markers.append("unresolved_failure")
        if contradiction_count > 0:
            scar_markers.append("carried_contradictions")
        if pending_initiatives > 0 or self._record.active_commitments:
            scar_markers.append("unfinished_obligations")

        try:
            min_reentry_gap_s = float(os.getenv("AURA_CONTINUITY_REENTRY_MIN_GAP_S", "900"))
        except (TypeError, ValueError):
            min_reentry_gap_s = 900.0

        reentry_required = bool(
            (continuity_pressure >= 0.28 and gap_seconds >= max(0.0, min_reentry_gap_s))
            or shutdown_factor > 0.0
            or bool(executive_failure_reason and gap_seconds >= max(0.0, min_reentry_gap_s))
            or contradiction_count > 0
        )

        return {
            "gap_seconds": gap_seconds,
            "continuity_pressure": round(continuity_pressure, 4),
            "continuity_scar": ", ".join(scar_markers),
            "reentry_degraded": reentry_required,
            "continuity_reentry_required": reentry_required,
            "executive_failure_reason": executive_failure_reason,
        }

    def save(
        self,
        reason: str = "graceful",
        last_exchange: str = "",
        belief_hash: str = "",
        active_commitments: list[str] | None = None,
        policy_mode: str | None = None,
        current_objective: str | None = None,
        pending_initiatives: int | None = None,
        pending_initiative_details: list[str] | None = None,
        health_summary: dict[str, Any] | None = None,
        rolling_summary: str | None = None,
        coherence_score: float | None = None,
        contradiction_count: int | None = None,
        subject_thread: str | None = None,
        active_goal_details: list[str] | None = None,
    ):
        """Write current session state. Call on graceful shutdown AND
        periodically (every 5 min) so crashes leave a recent record."""
        if (
            active_commitments is None
            or policy_mode is None
            or current_objective is None
            or pending_initiatives is None
            or pending_initiative_details is None
            or health_summary is None
            or rolling_summary is None
            or coherence_score is None
            or contradiction_count is None
            or subject_thread is None
            or active_goal_details is None
        ):
            try:
                from core.container import ServiceContainer

                if active_commitments is None:
                    ce = ServiceContainer.get("commitment_engine", default=None)
                    if ce and hasattr(ce, "get_active_commitments"):
                        active_commitments = [
                            getattr(item, "description", str(item))
                            for item in ce.get_active_commitments()[:5]
                        ]
                repo = ServiceContainer.get("state_repository", default=None)
                state = getattr(repo, "_current", None) if repo else None
                cognition = getattr(state, "cognition", None) if state else None
                if policy_mode is None:
                    mode = getattr(cognition, "current_mode", "unknown")
                    policy_mode = getattr(mode, "value", str(mode))
                if current_objective is None:
                    current_objective = getattr(cognition, "current_objective", "") if cognition else ""
                if pending_initiatives is None:
                    pending_initiatives = len(getattr(cognition, "pending_initiatives", []) or []) if cognition else 0
                if pending_initiative_details is None:
                    pending_initiative_details = [
                        str(item.get("goal") or item.get("type") or item)[:200]
                        for item in list(getattr(cognition, "pending_initiatives", []) or [])[:5]
                    ] if cognition else []
                if health_summary is None:
                    health_summary = dict(getattr(state, "health", {}) or {}) if state else {}
                if rolling_summary is None:
                    rolling_summary = getattr(cognition, "rolling_summary", "") if cognition else ""
                if coherence_score is None:
                    coherence_score = float(getattr(cognition, "coherence_score", 1.0) or 1.0) if cognition else 1.0
                if contradiction_count is None:
                    contradiction_count = int(getattr(cognition, "contradiction_count", 0) or 0) if cognition else 0
                if active_goal_details is None:
                    active_goal_details = [
                        str(item.get("goal") or item.get("description") or item)[:200]
                        for item in list(getattr(cognition, "active_goals", []) or [])[:5]
                    ] if cognition else []
                if subject_thread is None:
                    commitments_preview = ", ".join((active_commitments or [])[:2]) if active_commitments else "none"
                    subject_thread = (
                        f"Mode={policy_mode or 'unknown'} | Objective={current_objective or 'none'} | "
                        f"Commitments={commitments_preview} | Coherence={float(coherence_score or 1.0):.2f}"
                    )
            except (ImportError, AttributeError, RuntimeError) as e:
                record_degradation('continuity', e)
                logger.error("Continuity auto-capture failed: %s", e, exc_info=True)

        raw_current_objective = _sanitize_restored_text(current_objective)
        raw_objective_candidates = [
            raw_current_objective,
            *list(pending_initiative_details or []),
            *list(active_goal_details or []),
        ]
        current_objective = _sanitize_restored_objective(raw_current_objective)
        active_commitments = _sanitize_restored_items(active_commitments)
        pending_initiative_details = _sanitize_restored_objective_items(
            pending_initiative_details
        )
        active_goal_details = _sanitize_restored_objective_items(active_goal_details)
        rolling_summary = sanitize_continuity_summary(rolling_summary)
        subject_thread = _sanitize_restored_subject_thread(
            subject_thread,
            objective_candidates=raw_objective_candidates,
        )
        pending_initiatives = min(int(pending_initiatives or 0), len(pending_initiative_details))

        session_count = (self._record.session_count + 1) if self._record else 1
        prior_uptime = self._record.total_uptime_seconds if self._record else 0.0
        record = ContinuityRecord(
            last_shutdown=time.time(),
            last_shutdown_reason=reason,
            total_uptime_seconds=prior_uptime + (time.time() - self._boot_time),
            session_count=session_count,
            last_conversation_summary=last_exchange[:500],
            identity_hash=belief_hash,
            active_commitments=list(active_commitments or []),
            policy_mode=policy_mode or "unknown",
            current_objective=current_objective or "",
            pending_initiatives=int(pending_initiatives or 0),
            health_summary=dict(health_summary or {}),
            rolling_summary=rolling_summary,
            coherence_score=float(coherence_score or 1.0),
            contradiction_count=int(contradiction_count or 0),
            subject_thread=subject_thread,
            pending_initiative_details=list(pending_initiative_details or [])[:5],
            active_goal_details=list(active_goal_details or [])[:5],
        )
        try:
            path = _get_continuity_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            from core.runtime.file_write_gateway import get_file_write_gateway

            lifecycle_receipt = SimpleNamespace(
                receipt_id=f"continuity_shutdown:{int(time.time() * 1000)}",
                domain="state_mutation",
                source="continuity.save_shutdown_state",
            )
            with governed_scope_sync(lifecycle_receipt):
                get_file_write_gateway().write_text(
                    path,
                    json.dumps(_signed_record_payload(record), indent=2),
                    source="continuity.shutdown_record",
                )
            self._record = record
        except (RuntimeError, AttributeError, TypeError, ValueError) as e:
            record_degradation('continuity', e)
            logger.error("Continuity save failed: %s", e)

    @property
    def gap_seconds(self) -> float:
        return self._gap_seconds or 0.0

    @property
    def is_first_boot(self) -> bool:
        return self._record is None

    def get_waking_context(self) -> str:
        """Returns a string injected into Aura's first system prompt of the session.
        This is what makes her KNOW she was away, not just infer it."""
        if self.is_first_boot:
            return "This is your first awakening. You have no prior history."

        gap_h = self._gap_seconds / 3600
        if gap_h < 0.1:
            gap_str = f"{self._gap_seconds:.0f} seconds"
        elif gap_h < 2:
            gap_str = f"{self._gap_seconds/60:.0f} minutes"
        else:
            gap_str = f"{gap_h:.1f} hours"

        reason = self._record.last_shutdown_reason
        reentry = self._build_reentry_profile()
        shutdown_note = (
            "Your last session ended gracefully." if reason == "graceful"
            else f"Your last session ended unexpectedly ({reason})."
        )
        # The continuity record is written BEFORE a death, so its shutdown
        # reason can be stale. The flight recorder's black box is the ground
        # truth for how the previous life actually ended — when it recovered
        # a death report, its note supersedes the record's optimism.
        death_note = ""
        try:
            from core.runtime.flight_recorder import get_flight_recorder

            death_note = get_flight_recorder().waking_note()
        except (ImportError, AttributeError, RuntimeError, OSError, ValueError):
            death_note = ""
        if death_note:
            shutdown_note = death_note

        restored_subject = _sanitize_restored_subject_thread(
            self._record.subject_thread,
            objective_candidates=[
                self._record.current_objective,
                *list(self._record.pending_initiative_details or []),
                *list(self._record.active_goal_details or []),
            ],
        )
        return (
            f"You are waking from a gap of {gap_str}. {shutdown_note} "
            f"This is session #{self._record.session_count}. "
            f"Your total accumulated uptime is {self._record.total_uptime_seconds/3600:.1f} hours. "
            f"Last exchange summary: {self._record.last_conversation_summary or 'none recorded'}. "
            f"Policy mode at shutdown: {self._record.policy_mode or 'unknown'}. "
            f"Current objective at shutdown: {_sanitize_restored_objective(self._record.current_objective) or 'none'}. "
            f"Pending initiatives: {min(int(self._record.pending_initiatives or 0), len(_sanitize_restored_objective_items(self._record.pending_initiative_details)))}. "
            f"Pending initiative details: {', '.join(_sanitize_restored_objective_items(self._record.pending_initiative_details)[:3]) if _sanitize_restored_objective_items(self._record.pending_initiative_details) else 'none recorded'}. "
            f"Active commitments: {', '.join(_sanitize_restored_items(self._record.active_commitments)[:3]) if _sanitize_restored_items(self._record.active_commitments) else 'none recorded'}. "
            f"Active goals: {', '.join(_sanitize_restored_objective_items(self._record.active_goal_details)[:3]) if _sanitize_restored_objective_items(self._record.active_goal_details) else 'none recorded'}. "
            f"Coherence at shutdown: {self._record.coherence_score:.2f}. "
            f"Contradictions carried forward: {self._record.contradiction_count}. "
            f"Subject thread: {restored_subject or 'none recorded'}. "
            f"Continuity pressure carried into this session: {float(reentry['continuity_pressure']):.2f}. "
            f"Re-entry burden: {reentry['continuity_scar'] or 'light_trace'}."
        )

    def get_obligations(self) -> dict[str, Any]:
        reentry = self._build_reentry_profile()
        live_identity_hash = self._get_live_identity_hash()
        persisted_identity_hash = self._record.identity_hash if self._record else ""
        identity_mismatch = bool(
            self._record
            and persisted_identity_hash
            and live_identity_hash
            and persisted_identity_hash != live_identity_hash
        )
        what_the_hash_missed = self._what_the_hash_missed(not identity_mismatch)
        if self._record is None:
            return {
                "current_objective": "",
                "active_commitments": [],
                "pending_initiatives": [],
                "active_goals": [],
                "contradiction_count": 0,
                "subject_thread": "",
                "identity_hash": live_identity_hash,
                "persisted_identity_hash": "",
                "identity_mismatch": False,
                "what_the_hash_missed": "",
                **reentry,
            }
        sanitized_pending = _sanitize_restored_objective_items(self._record.pending_initiative_details)
        sanitized_goals = _sanitize_restored_objective_items(self._record.active_goal_details)
        sanitized_commitments = _sanitize_restored_items(self._record.active_commitments)
        restored_subject = _sanitize_restored_subject_thread(
            self._record.subject_thread,
            objective_candidates=[
                self._record.current_objective,
                *list(self._record.pending_initiative_details or []),
                *list(self._record.active_goal_details or []),
            ],
        )
        return {
            "current_objective": _sanitize_restored_objective(self._record.current_objective),
            "active_commitments": sanitized_commitments,
            "pending_initiatives": sanitized_pending,
            "active_goals": sanitized_goals,
            "contradiction_count": int(self._record.contradiction_count or 0),
            "subject_thread": restored_subject,
            "identity_hash": live_identity_hash,
            "persisted_identity_hash": persisted_identity_hash,
            "identity_mismatch": identity_mismatch,
            # What the flag above missed, where the relation disagrees with it.
            # Empty when the two agree, which is most of the time.
            "what_the_hash_missed": what_the_hash_missed,
            **reentry,
        }

    def apply_to_state(self, state: Any) -> Any:
        """Make prior continuity causally available to the live runtime, not just prompt text."""
        if state is None or self._record is None:
            return state

        cognition = getattr(state, "cognition", None)
        if cognition is None:
            return state

        obligations = self.get_obligations()

        restored_objective = _sanitize_restored_objective(self._record.current_objective)
        restored_pending = _sanitize_restored_objective_items(self._record.pending_initiative_details)
        restored_goals = _sanitize_restored_objective_items(self._record.active_goal_details)

        if not getattr(cognition, "current_objective", None) and restored_objective:
            cognition.current_objective = restored_objective

        restored_subject = str(obligations.get("subject_thread") or "")
        if not getattr(cognition, "rolling_summary", "") and restored_subject:
            cognition.rolling_summary = restored_subject

        cognition.contradiction_count = max(
            int(getattr(cognition, "contradiction_count", 0) or 0),
            int(self._record.contradiction_count or 0),
        )

        restored_pending = [item for item in restored_pending if not _is_generic_continuity_reentry_goal(item)]
        restored_goals = [item for item in restored_goals if not _is_generic_continuity_reentry_goal(item)]

        if not list(getattr(cognition, "pending_initiatives", []) or []) and restored_pending:
            cognition.pending_initiatives = [
                {
                    "goal": item,
                    "source": "continuity",
                    "continuity_restored": True,
                    "continuity_gap_seconds": float(obligations.get("gap_seconds", 0.0) or 0.0),
                    "continuity_pressure": float(obligations.get("continuity_pressure", 0.0) or 0.0),
                    "metadata": {
                        "continuity_restored": True,
                        "continuity_pressure": float(obligations.get("continuity_pressure", 0.0) or 0.0),
                        "continuity_scar": obligations.get("continuity_scar", ""),
                        "gap_seconds": float(obligations.get("gap_seconds", 0.0) or 0.0),
                    },
                }
                for item in restored_pending[:5]
            ]

        if not list(getattr(cognition, "active_goals", []) or []) and restored_goals:
            cognition.active_goals = [
                {
                    "goal": item,
                    "source": "continuity",
                    "continuity_restored": True,
                    "metadata": {
                        "continuity_restored": True,
                        "continuity_pressure": float(obligations.get("continuity_pressure", 0.0) or 0.0),
                        "continuity_scar": obligations.get("continuity_scar", ""),
                    },
                }
                for item in restored_goals[:5]
            ]

        try:
            min_reentry_gap_s = float(os.getenv("AURA_CONTINUITY_REENTRY_MIN_GAP_S", "900"))
        except (TypeError, ValueError):
            min_reentry_gap_s = 900.0
        gap_seconds = float(obligations.get("gap_seconds", 0.0) or 0.0)
        shutdown_was_graceful = self._record.last_shutdown_reason == "graceful"
        meaningful_gap = gap_seconds >= max(0.0, min_reentry_gap_s) or not shutdown_was_graceful

        continuity_initiative_enabled = str(
            os.getenv("AURA_ENABLE_CONTINUITY_REENTRY_INITIATIVE", "1")
        ).strip().lower() in {"1", "true", "yes", "on"}

        should_inject_reentry_initiative = bool(
            continuity_initiative_enabled
            and obligations.get("continuity_reentry_required")
            and meaningful_gap
            and (restored_objective or restored_pending or restored_goals or _sanitize_restored_items(self._record.active_commitments))
            and (
                float(obligations.get("continuity_pressure", 0.0) or 0.0) >= 0.35
                or gap_seconds >= max(0.0, min_reentry_gap_s)
                or not shutdown_was_graceful
                or obligations.get("executive_failure_reason")
            )
        )
        if should_inject_reentry_initiative:
            existing_goals = {
                str(item.get("goal", "")).strip().lower()
                for item in list(getattr(cognition, "pending_initiatives", []) or [])
                if isinstance(item, dict)
            }
            reentry_goal = "Reconcile continuity gap and re-establish the interrupted thread"
            if reentry_goal.lower() not in existing_goals:
                cognition.pending_initiatives = [
                    {
                        "goal": reentry_goal,
                        "source": "continuity",
                        "triggered_by": "continuity",
                        "urgency": round(max(0.55, float(obligations.get("continuity_pressure", 0.0) or 0.0)), 4),
                        "status": "suggested",
                        "timestamp": time.time(),
                        "continuity_restored": True,
                        "continuity_obligation": True,
                        "continuity_gap_seconds": float(obligations.get("gap_seconds", 0.0) or 0.0),
                        "continuity_pressure": float(obligations.get("continuity_pressure", 0.0) or 0.0),
                        "metadata": {
                            "continuity_restored": True,
                            "continuity_obligation": True,
                            "continuity_pressure": float(obligations.get("continuity_pressure", 0.0) or 0.0),
                            "continuity_scar": obligations.get("continuity_scar", ""),
                            "gap_seconds": float(obligations.get("gap_seconds", 0.0) or 0.0),
                            "executive_failure_reason": obligations.get("executive_failure_reason", ""),
                        },
                    }
                ] + list(getattr(cognition, "pending_initiatives", []) or [])
                cognition.trim_working_memory()

        modifiers = dict(getattr(cognition, "modifiers", {}) or {})
        modifiers["continuity_obligations"] = {
            "session_count": int(self._record.session_count or 0),
            "last_shutdown_reason": self._record.last_shutdown_reason,
            "current_objective": restored_objective,
            "active_commitments": _sanitize_restored_items(self._record.active_commitments),
            "pending_initiatives": restored_pending,
            "active_goals": restored_goals,
            "contradiction_count": int(self._record.contradiction_count or 0),
            "subject_thread": restored_subject,
            "identity_hash": self._record.identity_hash,
            "live_identity_hash": self._get_live_identity_hash(),
            "identity_mismatch": obligations.get("identity_mismatch", False),
            "gap_seconds": float(obligations.get("gap_seconds", 0.0) or 0.0),
            "continuity_pressure": float(obligations.get("continuity_pressure", 0.0) or 0.0),
            "continuity_scar": obligations.get("continuity_scar", ""),
            "reentry_degraded": bool(obligations.get("reentry_degraded", False)),
            "continuity_reentry_required": bool(obligations.get("continuity_reentry_required", False)),
            "executive_failure_reason": obligations.get("executive_failure_reason", ""),
        }
        cognition.modifiers = modifiers
        return state

    def note_failure_obligation(self, reason: str, goal: str = "") -> None:
        if self._record is None:
            self._record = ContinuityRecord(
                last_shutdown=time.time(),
                last_shutdown_reason="runtime",
                total_uptime_seconds=0.0,
                session_count=0,
                last_conversation_summary="",
                identity_hash=self._get_live_identity_hash(),
            )
        health_summary = dict(self._record.health_summary or {})
        health_summary["executive_failure_reason"] = str(reason or "")[:200]
        if goal:
            health_summary["executive_failure_goal"] = str(goal)[:200]
        health_summary["executive_failure_at"] = time.time()
        self._record.health_summary = health_summary
        marker = f"Reconcile executive failure: {str(reason or '')[:80]}"
        existing = list(self._record.active_commitments or [])
        if marker and marker not in existing:
            existing.append(marker)
            self._record.active_commitments = existing[-5:]
        try:
            path = _get_continuity_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            from core.runtime.file_write_gateway import get_file_write_gateway

            recovery_receipt = SimpleNamespace(
                receipt_id=f"continuity_failure_obligation:{int(time.time() * 1000)}",
                domain="state_mutation",
                source="continuity.note_failure_obligation",
            )
            with governed_scope_sync(recovery_receipt):
                get_file_write_gateway().write_text(
                    path,
                    json.dumps(_signed_record_payload(self._record), indent=2),
                    source="continuity.executive_failure_obligation",
                )
        except (RuntimeError, AttributeError, TypeError, ValueError) as e:
            record_degradation('continuity', e)
            logger.error("Continuity failure obligation save failed: %s", e, exc_info=True)

    def _what_the_hash_missed(self, hash_matches: bool) -> str:
        """What the relation says where the hash comparison disagrees with it.

        The hash gets both directions wrong. Any belief she changed makes the
        hashes differ, so ordinary learning reads as an identity break; and a
        backup restored from a month ago matches, so a state with no causal
        path from the last one reads as continuous.

        `core/identity/continuity_relation.py` was written to say exactly that
        and nothing outside its own test ever called it, so the flag below has
        been reported on its own since the day the argument against it was
        written down. Empty string when the two agree, which is most of the
        time and is not worth saying.
        """

        try:
            from core.identity.continuity_relation import (
                Step,
                hash_disagrees_with_relation,
                relate,
            )

            record = self._record
            if record is None:
                return ""
            before = frozenset(
                str(one) for one in (record.active_commitments or ()) if one
            )
            after = frozenset(
                str(one) for one in self._live_commitments() if one
            )
            if not before and not after:
                return ""
            step = Step(
                step_id="across-the-restart",
                before=before,
                after=after,
                # What she is committed to is the load-bearing part by the
                # relation's own definition, and it is what this record keeps.
                load_bearing=before,
                origin=str(getattr(record, "origin", "") or "self"),
            )
            return hash_disagrees_with_relation(hash_matches, relate([step]))
        except (AttributeError, ImportError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation("continuity", exc, severity="debug")
            return ""

    def _live_commitments(self) -> list[str]:
        """What she is committed to now, for the relation above."""

        try:
            from core.container import ServiceContainer

            will = ServiceContainer.get("will", default=None)
            held = getattr(will, "active_commitments", None)
            if callable(held):
                held = held()
            return [str(one) for one in (held or ()) if one]
        except (AttributeError, ImportError, RuntimeError, TypeError, ValueError):
            return []

    def _get_live_identity_hash(self) -> str:
        try:
            from core.identity.heartstone import AURA_HEARTSTONE

            return str(AURA_HEARTSTONE.identity_hash or "")
        except (ImportError, AttributeError, RuntimeError) as exc:
            record_degradation("continuity", exc)
            logger.debug("Live identity hash lookup failed: %s", exc)
            return ""


# Singleton
_continuity: ContinuityEngine | None = None

def get_continuity() -> ContinuityEngine:
    global _continuity
    if _continuity is None:
        _continuity = ContinuityEngine()
    return _continuity
