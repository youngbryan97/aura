"""Shared live-mind generation contract helpers.

The desktop lane treats live-mind metadata as proof material. Parent context
may reconcile a stale structural-bound bit, but it must never manufacture a
worker application or quality success that the worker did not report.
"""

from __future__ import annotations

import hashlib
import math
import re
import uuid
from collections.abc import Mapping
from typing import Any

from core.runtime.errors import record_degradation

REQUIRED_LIVE_MIND_GENERATION_CONTROL_KEYS = frozenset(
    {
        "temperature",
        "top_p",
        "clean_user_surface_recurrent_loops",
        "clean_user_surface_steering_alpha",
    }
)

_MAX_TEXT_MUTATIONS = 64
_CONTENT_FREE_REASON_RE = re.compile(r"^[A-Za-z0-9_.:/-]{1,96}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

# A text hash chain proves which bytes changed; it does not prove who authored
# the new meaning. Keep that second question explicit. Unknown/legacy values
# normalize to ``replaced_by_runtime`` so incomplete provenance cannot inherit
# model authorship merely because a model ran earlier in the turn.
TEXT_MUTATION_AUTHORSHIP_EFFECTS = frozenset(
    {
        "preserved",
        "augmented_by_runtime",
        "replaced_by_model",
        "replaced_by_runtime",
    }
)


def _normalize_authorship_effect(value: Any) -> str:
    effect = str(value or "").strip().lower()
    if effect in TEXT_MUTATION_AUTHORSHIP_EFFECTS:
        return effect
    return "replaced_by_runtime"


def _content_free_reason(value: Any) -> str:
    """Keep categorical reasons while replacing arbitrary detail with a digest."""

    if isinstance(value, Mapping):
        for key in ("code", "category", "rule", "type", "name"):
            candidate = str(value.get(key) or "").strip()
            if _CONTENT_FREE_REASON_RE.fullmatch(candidate):
                return candidate[:96]
        value = repr(sorted((str(key), repr(item)) for key, item in value.items()))
    candidate = str(value or "").strip()
    if not candidate:
        return ""
    if _CONTENT_FREE_REASON_RE.fullmatch(candidate):
        return candidate[:96]
    digest = hashlib.sha256(candidate.encode("utf-8", errors="replace")).hexdigest()[:16]
    return f"detail_sha256:{digest}"


def normalize_text_mutations(value: Any) -> list[dict[str, Any]]:
    """Return bounded, content-free provenance for visible-text mutations."""

    if not isinstance(value, list):
        return []
    normalized: list[dict[str, Any]] = []
    for fallback_sequence, item in enumerate(
        value[-_MAX_TEXT_MUTATIONS:],
        start=1,
    ):
        if not isinstance(item, Mapping):
            continue
        stage = str(item.get("stage") or "").strip()[:96]
        method = str(item.get("method") or "").strip()[:96]
        reasons = [
            normalized_reason
            for reason in (item.get("reasons") or [])
            if (normalized_reason := _content_free_reason(reason))
        ][:16]
        if not stage or not method:
            continue
        entry: dict[str, Any] = {
            "sequence": fallback_sequence,
            "stage": stage,
            "method": method,
            "reasons": reasons,
            "deterministic": bool(item.get("deterministic", False)),
            "authorship_effect": _normalize_authorship_effect(
                item.get("authorship_effect")
            ),
        }
        event_id = str(item.get("event_id") or "").strip()[:64]
        if event_id:
            entry["event_id"] = event_id
        try:
            entry["sequence"] = max(1, int(item.get("sequence") or fallback_sequence))
        except (TypeError, ValueError, OverflowError):
            entry["sequence"] = fallback_sequence
        for key in ("before_chars", "after_chars"):
            try:
                entry[key] = max(0, int(item.get(key) or 0))
            # not a failure: a value that is not a number is not one this can read.
            except (TypeError, ValueError, OverflowError):
                entry[key] = 0
        for key in ("before_sha256", "after_sha256"):
            digest = str(item.get(key) or "").strip().lower()
            entry[key] = digest if _SHA256_RE.fullmatch(digest) else ""
        normalized.append(entry)
    return normalized


def merge_text_mutations(*values: Any) -> list[dict[str, Any]]:
    """Merge ordered mutation ledgers while removing copied receipt duplicates."""

    def _content_identity(entry: dict[str, Any]) -> tuple[Any, ...]:
        """What this mutation actually IS, independent of its label."""
        return (
            entry["stage"],
            entry["method"],
            tuple(entry["reasons"]),
            entry["deterministic"],
            entry["authorship_effect"],
            entry["before_chars"],
            entry["after_chars"],
            entry["before_sha256"],
            entry["after_sha256"],
        )

    merged: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    # event_id -> the content it was first seen carrying.
    #
    # CP126 (high): "Merge identity is only the presence and text of
    # event_id. Reusing another event's ID causes a different mutation to be
    # silently dropped, with no comparison of its hashes or metadata."
    #
    # This ledger is the audit trail for what was done to a user-facing
    # answer, and an id is caller-supplied. Any caller — a retry, a buggy
    # copy, a replayed receipt — that reuses an id erased a genuinely
    # different mutation from the record, leaving a history that reads as
    # complete.
    #
    # A duplicate id with IDENTICAL content is the case this dedupe exists
    # for and is still dropped. A duplicate id with DIFFERENT content is a
    # collision: the entry is kept under its content identity and the
    # collision is recorded, because losing a real mutation from an audit
    # ledger is worse than carrying a redundant one.
    seen_events: dict[str, tuple[Any, ...]] = {}
    for value in values:
        for entry in normalize_text_mutations(value):
            event_id = str(entry.get("event_id") or "")
            content = _content_identity(entry)
            if event_id:
                previous = seen_events.get(event_id)
                if previous is None:
                    seen_events[event_id] = content
                    identity: tuple[Any, ...] = ("event", event_id)
                elif previous == content:
                    identity = ("event", event_id)
                else:
                    record_degradation(
                        "live_mind_contract",
                        ValueError(
                            f"text mutation event_id collision with differing content: {event_id[:32]}"
                        ),
                        severity="warning",
                        action="kept both mutations rather than dropping one from the audit ledger",
                        enforce_failure_policy=False,
                    )
                    identity = ("content", *content)
            else:
                identity = ("legacy", *content)
            if identity in seen:
                continue
            seen.add(identity)
            merged.append(entry)
    bounded = merged[-_MAX_TEXT_MUTATIONS:]
    return [dict(entry, sequence=index) for index, entry in enumerate(bounded, start=1)]


def summarize_text_mutation_authorship(value: Any) -> dict[str, bool]:
    """Derive stable authorship flags from the normalized mutation ledger."""

    effects = {
        str(entry.get("authorship_effect") or "replaced_by_runtime")
        for entry in normalize_text_mutations(value)
    }
    return {
        "authorship_replacement_applied": "replaced_by_runtime" in effects,
        "authorship_augmentation_applied": "augmented_by_runtime" in effects,
        "model_replacement_applied": "replaced_by_model" in effects,
    }


def append_text_mutation(
    container: dict[str, Any],
    *,
    stage: str,
    method: str,
    reasons: Any,
    before: Any,
    after: Any,
    deterministic: bool,
    authorship_effect: str = "replaced_by_runtime",
) -> dict[str, Any]:
    """Append one mutation only when the visible bytes actually changed."""

    before_text = str(before or "")
    after_text = str(after or "")
    if before_text == after_text:
        return container
    if isinstance(reasons, str):
        reason_items = [reasons]
    else:
        reason_items = list(reasons or [])
    entry = {
        "event_id": uuid.uuid4().hex,
        "sequence": 1
        + max(
            (
                int(item.get("sequence") or 0)
                for item in normalize_text_mutations(container.get("text_mutations"))
            ),
            default=0,
        ),
        "stage": stage,
        "method": method,
        "reasons": reason_items,
        "deterministic": bool(deterministic),
        "authorship_effect": _normalize_authorship_effect(authorship_effect),
        "before_chars": len(before_text),
        "after_chars": len(after_text),
        "before_sha256": hashlib.sha256(before_text.encode("utf-8")).hexdigest(),
        "after_sha256": hashlib.sha256(after_text.encode("utf-8")).hexdigest(),
    }
    mutations = merge_text_mutations(container.get("text_mutations"), [entry])
    container["text_mutations"] = mutations
    container["text_mutation_count"] = len(mutations)
    container["deterministic_repair_applied"] = any(
        bool(item.get("deterministic")) for item in mutations
    )
    container.update(summarize_text_mutation_authorship(mutations))
    return container


def verify_text_mutation_chain(
    value: Any,
    *,
    before_sha256: Any,
    after_sha256: Any,
) -> dict[str, Any]:
    """Verify a contiguous, hash-bound mutation suffix from before to after.

    A surface ledger may contain worker-side mutations that happened before a
    model output was quality-certified. Those entries are legitimate prefix
    provenance, but the public-delivery proof begins at the certified output
    hash. The verifier therefore accepts only a contiguous suffix beginning at
    that hash and ending at the exact public bytes; every mutation in that
    suffix must carry an independently checkable before/after digest.
    """

    expected_before = str(before_sha256 or "").strip().lower()
    expected_after = str(after_sha256 or "").strip().lower()
    entries = normalize_text_mutations(value)
    reasons: list[str] = []
    if not _SHA256_RE.fullmatch(expected_before):
        reasons.append("invalid_before_sha256")
    if not _SHA256_RE.fullmatch(expected_after):
        reasons.append("invalid_after_sha256")
    if not isinstance(value, list):
        reasons.append("mutation_ledger_missing")
    elif len(value) > _MAX_TEXT_MUTATIONS:
        reasons.append("mutation_ledger_exceeds_bound")
    elif len(entries) != len(value):
        reasons.append("mutation_ledger_malformed")
    if expected_before == expected_after and not reasons:
        # "Nothing changed" is a claim about the ledger, so it has to be checked
        # against the ledger. This used to return passed with a zero-length
        # chain WITHOUT looking at the entries at all, so a ledger recording a
        # real post-certification mutation still produced a clean no-change
        # proof — precisely the case the proof exists to catch.
        #
        # Entries before the certified hash are legitimate prefix provenance
        # (see the docstring), so only mutations that START at the certified
        # hash and move away from it contradict the claim.
        contradicting = [
            entry for entry in entries
            if str(entry.get("before_sha256") or "").strip().lower() == expected_before
            and str(entry.get("after_sha256") or "").strip().lower() != expected_before
        ]
        if contradicting:
            reasons.append("no_change_claim_contradicted_by_ledger")
        else:
            return {
                "schema": "aura.text_mutation_chain.v1",
                "passed": True,
                "before_sha256": expected_before,
                "after_sha256": expected_after,
                "ledger_entry_count": len(entries),
                "chain_start_sequence": 0,
                "chain_length": 0,
                "reasons": [],
            }
    if not entries:
        reasons.append("mutation_chain_missing")

    valid_suffix: list[dict[str, Any]] | None = None
    if not reasons:
        for index, entry in enumerate(entries):
            if entry.get("before_sha256") != expected_before:
                continue
            suffix = entries[index:]
            if any(
                not item.get("event_id")
                or not _SHA256_RE.fullmatch(str(item.get("before_sha256") or ""))
                or not _SHA256_RE.fullmatch(str(item.get("after_sha256") or ""))
                for item in suffix
            ):
                continue
            if len({str(item["event_id"]) for item in suffix}) != len(suffix):
                continue
            if any(
                left.get("after_sha256") != right.get("before_sha256")
                for left, right in zip(suffix, suffix[1:], strict=False)
            ):
                continue
            if suffix[-1].get("after_sha256") != expected_after:
                continue
            valid_suffix = suffix
            break
    if valid_suffix is None and not reasons:
        reasons.append("mutation_hash_chain_mismatch")

    return {
        "schema": "aura.text_mutation_chain.v1",
        "passed": not reasons,
        "before_sha256": expected_before,
        "after_sha256": expected_after,
        "ledger_entry_count": len(entries),
        "chain_start_sequence": (
            int(valid_suffix[0].get("sequence") or 0) if valid_suffix else 0
        ),
        "chain_length": len(valid_suffix or ()),
        "reasons": reasons,
    }


#: Admissible ranges for the generation controls. Presence of a KEY proves
#: nothing about whether the setting is usable, so each control declares the
#: shape and range it must actually satisfy.
_CONTROL_RANGES: dict[str, tuple[float, float]] = {
    # Sampling temperature: 0 is greedy decoding; above 2 is not a setting any
    # of Aura's decoders accept.
    "temperature": (0.0, 2.0),
    # Nucleus sampling mass must be a probability, and 0 would admit no tokens.
    "top_p": (1e-6, 1.0),
    # Steering strength is a unit interval by construction.
    "clean_user_surface_steering_alpha": (0.0, 1.0),
}
#: Recurrent loop counts are a positive integer count of passes, bounded so a
#: malformed value cannot request an unbounded number of them.
_MAX_CLEAN_SURFACE_LOOPS = 32


def live_mind_generation_controls_present(generation_controls: Any) -> bool:
    """True only when the controls are present AND usable.

    Key presence used to be the whole test, so a mapping carrying nulls, NaN,
    strings, or out-of-range numbers counted as "controls structurally
    present" — a proof of configuration that proved nothing about whether
    generation could actually be steered.
    """
    if not isinstance(generation_controls, Mapping):
        return False
    if not REQUIRED_LIVE_MIND_GENERATION_CONTROL_KEYS.issubset(
        generation_controls.keys()
    ):
        return False

    for key, (low, high) in _CONTROL_RANGES.items():
        raw = generation_controls.get(key)
        # bool is an int subclass, and a numeric STRING is a serialisation
        # artefact rather than a setting — neither is evidence that generation
        # was configured.
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            return False
        value = float(raw)
        # NaN fails every comparison, so it must be rejected explicitly rather
        # than by the range check below.
        if not math.isfinite(value) or not (low <= value <= high):
            return False

    loops = generation_controls.get("clean_user_surface_recurrent_loops")
    if isinstance(loops, bool) or not isinstance(loops, int):
        return False
    if not (1 <= loops <= _MAX_CLEAN_SURFACE_LOOPS):
        return False
    return True


def normalize_live_mind_surface_control_receipt(
    receipt: Any,
    *,
    controls_bound: bool,
    generation_controls: Any,
    surface_quality_gate_passed: bool | None = None,
    source: str,
) -> dict[str, Any]:
    """Return a coherent receipt for an already-bound live-mind turn.

    The worker reports whether it applied CAA/recurrent surface controls, while
    the CognitiveEngine owns whether the live mind controls were structurally
    bound for the turn. If the worker receipt is otherwise successful but omits
    that structural bit, normalize it before the chat contract evaluates the
    full-mind path.
    """

    normalized = dict(receipt) if isinstance(receipt, Mapping) else {}
    mutations = normalize_text_mutations(normalized.get("text_mutations"))
    if mutations:
        normalized["text_mutations"] = mutations
        normalized["text_mutation_count"] = len(mutations)
        normalized.update(summarize_text_mutation_authorship(mutations))
    else:
        # Summary booleans are derived evidence, not independent assertions.
        # With no retained ledger they cannot prove a mutation's authorship.
        normalized.pop("authorship_replacement_applied", None)
        normalized.pop("authorship_augmentation_applied", None)
        normalized.pop("model_replacement_applied", None)
    controls_present = live_mind_generation_controls_present(generation_controls)
    quality_passed = bool(
        normalized.get("surface_quality_gate_passed") is True
        if surface_quality_gate_passed is None
        else (
            surface_quality_gate_passed
            and normalized.get("surface_quality_gate_passed") is True
        )
    )

    generation_required = normalized.get("generation_required") is not False
    if not generation_required:
        return {
            **normalized,
            "enabled": False,
            "applied": False,
            "generation_required": False,
            "application_status": "not_applicable_structured_floor",
            "live_mind_controls_bound": bool(controls_bound and controls_present),
            "clean_user_surface_contract": True,
            "surface_quality_gate_enabled": False,
            "surface_quality_gate_passed": bool(quality_passed),
            "surface_quality_gate_attempts": 0,
            "surface_quality_gate_reasons": [],
            "source": normalized.get("source") or source,
        }

    if not normalized:
        return {
            "enabled": False,
            "applied": False,
            "generation_required": True,
            "application_status": "worker_receipt_missing",
            "live_mind_controls_bound": bool(controls_bound and controls_present),
            "clean_user_surface_contract": False,
            "surface_quality_gate_enabled": False,
            "surface_quality_gate_passed": False,
            "surface_quality_gate_attempts": 0,
            "surface_quality_gate_reasons": ["worker_receipt_missing"],
            "source": source,
        }

    worker_applied = normalized.get("applied") is True
    clean_surface_proven = normalized.get("clean_user_surface_contract") is True
    worker_success = bool(worker_applied and clean_surface_proven and quality_passed)
    structurally_bound = bool(controls_bound and controls_present and worker_success)

    if structurally_bound and normalized.get("live_mind_controls_bound") is True:
        return normalized

    if not worker_success:
        if not worker_applied:
            failure_status = "worker_controls_not_applied"
        elif not clean_surface_proven:
            failure_status = "worker_clean_surface_unproven"
        else:
            failure_status = "worker_surface_quality_unproven"
    elif not structurally_bound:
        failure_status = "live_mind_controls_unbound"
    else:
        failure_status = "applied"

    return {
        **normalized,
        "enabled": bool(normalized.get("enabled", False)),
        "applied": worker_applied,
        "live_mind_controls_bound": structurally_bound,
        "clean_user_surface_contract": clean_surface_proven,
        "surface_quality_gate_enabled": bool(
            normalized.get("surface_quality_gate_enabled", False)
        ),
        "surface_quality_gate_passed": quality_passed,
        "surface_quality_gate_attempts": int(
            normalized.get("surface_quality_gate_attempts", 0) or 0
        ),
        "surface_quality_gate_reasons": list(
            normalized.get("surface_quality_gate_reasons", []) or []
        ),
        "application_status": normalized.get("application_status") or failure_status,
        "source": normalized.get("source") or source,
    }
