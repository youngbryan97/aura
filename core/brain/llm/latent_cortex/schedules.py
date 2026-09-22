"""Layer-schedule programs: each transformer block becomes an instruction.

The ordinary schedule (0,1,…,63, once each) is one point in an enormous
family of virtual architectures the same frozen tensors can execute. A
``LayerSchedule`` is a validated neural program over the recurrent region:

    [(start, end, repeats, alpha_override), ...]

Guarantees enforced here, not hoped for:
- Programs are validated against the model's actual topology and the
  episode's compute budget BEFORE execution — an invalid program never
  touches the model.
- Canonical serialization + content hash so every receipt names exactly
  which program ran.
- The library tracks per-domain reliability with Wilson lower bounds (the
  same statistics the Verifier Foundry uses) — a schedule is only ever
  preferred on evidence, never on vibes.
- Search is evolutionary, seeded, and budgeted; candidates are scored ONLY
  by a caller-supplied evaluator (which must itself be verifier-backed) and
  every promotion carries provenance.

Whatever shape a program takes, the engine's final clean persist pass
guarantees coherent slot KV — a pathological program can waste compute but
cannot corrupt attention state.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import random
import re
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from core.brain.verifiers.foundry import wilson_lower_bound, wilson_upper_bound

logger = logging.getLogger("Aura.LatentCortex.Schedules")

# A schedule may apply at most this many window-layer applications per slot
# token, regardless of budget — programs beyond this are degenerate.
MAX_TOTAL_LAYER_REPEATS = 4096
SCHEDULE_LIBRARY_SCHEMA_VERSION = 2
COMPUTE_MATCH_RELATIVE_TOLERANCE = 0.05
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_RUN_ORDERS = frozenset({"candidate_first", "default_first"})
_MAX_LAYER_APPS = (1 << 63) - 1


def _canonical_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_sha256(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    return value


def _require_identifier(value: Any, *, field_name: str, limit: int = 200) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    normalized = value.strip()
    if not normalized or len(normalized) > limit or any(ord(char) < 32 for char in normalized):
        raise ValueError(f"{field_name} must be a non-empty printable identifier")
    return normalized


def _safe_display(value: Any, *, limit: int = 120) -> str:
    try:
        rendered = repr(value)
    except (TypeError, ValueError, OverflowError, RecursionError):
        return f"<{type(value).__name__}:unprintable>"
    if len(rendered) > limit:
        return f"{rendered[:limit]}..."
    return rendered


@dataclass(frozen=True)
class StageOp:
    """One typed instruction of the neural bytecode.

    Kinds:
      window        — run layers [start, end) ``repeats`` times (the
                      original schedule instruction; the only kind that
                      spends layer compute directly).
      exchange      — force a branch-communication exchange NOW, instead
                      of waiting for the step-count interval.
      savepoint     — snapshot every branch's latent state (one slot;
                      later savepoints overwrite).
      verify_probe  — decode a probe from the current leading branch and
                      score it with the episode verifier; with
                      ``revert_on_drop`` every branch backtracks to its
                      savepoint when the verified score fell — explicit,
                      receipted, verifier-guided backtracking.

    Every kind is validated, canonically serialized, and covered by the
    schedule hash. ``window`` payloads serialize exactly as before, so
    existing library hashes and provenance receipts remain valid.
    """

    start: int = 0
    end: int = 0
    repeats: int = 1
    alpha: float | None = None  # override the recurrence alpha for this stage
    kind: str = "window"
    revert_on_drop: bool = False

    #: Places of alpha that are part of a schedule's IDENTITY.
    ALPHA_QUANTUM_PLACES: ClassVar[int] = 6

    def __post_init__(self) -> None:
        """Quantize alpha so the executed value IS the hashed value.

        CP126 2e5d5cd6: execution retained the raw ``alpha`` while
        ``to_dict`` rounded it to six places before hashing. Two schedules
        whose alphas differed past the sixth place therefore behaved
        differently but shared one schedule hash — and that hash is the key
        for receipts, the score cache, and promotion evidence. Distinct
        recurrence behaviour aliasing onto one identity means a schedule can
        inherit another's measured results.

        Quantizing here rather than widening the hash is deliberate: it makes
        the two agree by construction, so no future serializer can reopen the
        gap. Alpha is a recurrence blend weight; six places is far below any
        behavioural resolution, and the value is now honestly what it claims.

        This deliberately does NOT raise. Direct construction is permissive by
        design so that ``validate()`` can REPORT an unexecutable program
        instead of the caller crashing on it; ``from_dict`` is the strict gate
        for untrusted input. An alpha that cannot even be represented as a
        finite float is left exactly as supplied, so the existing validation
        path still sees it and still refuses it.
        """
        if self.alpha is None:
            return
        try:
            alpha = float(self.alpha)
        # not a failure: a value that is not a number is not one this can read.
        except (TypeError, ValueError, OverflowError):
            return
        if not math.isfinite(alpha):
            return
        quantized = round(alpha, self.ALPHA_QUANTUM_PLACES)
        if quantized != alpha:
            object.__setattr__(self, "alpha", quantized)

    def to_dict(self) -> dict[str, Any]:
        if self.kind != "window":
            out: dict[str, Any] = {"kind": self.kind}
            if self.revert_on_drop:
                out["revert_on_drop"] = True
            return out
        out = {"start": self.start, "end": self.end, "repeats": self.repeats}
        if self.alpha is not None:
            try:
                alpha = float(self.alpha)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("stage alpha must be a finite number") from exc
            if not math.isfinite(alpha):
                raise ValueError("stage alpha must be a finite number")
            out["alpha"] = round(alpha, 6)
        return out


STAGE_OP_KINDS: frozenset[str] = frozenset(
    {"window", "exchange", "savepoint", "verify_probe"}
)
_MAX_VERIFY_PROBES = 4

#: Cost of the non-window ops, in layer-application equivalents.
#:
#: CP126 bd211d76: only window ops counted toward the compute ceiling, so
#: exchange, savepoint and verify_probe were free. They are not: exchange
#: copies latent state across branches, savepoint snapshots every branch, and
#: verify_probe DECODES TEXT and invokes the episode verifier — by far the most
#: expensive instruction in the set. A schedule could therefore stay nominally
#: under a layer-repeat ceiling while doing a great deal of real work, which is
#: precisely how a budget stops bounding anything.
#:
#: These are declared ESTIMATES, not measurements, and they are deliberately
#: coarse. Their job is to stop the ceiling being trivially evadable; the
#: measured cost belongs in the execution receipt, which is the other half of
#: this finding and is owned by the executor rather than by validation.
EXCHANGE_LAYER_EQUIVALENT = 1
SAVEPOINT_LAYER_EQUIVALENT = 1
VERIFY_PROBE_LAYER_EQUIVALENT = 32

NON_WINDOW_LAYER_EQUIVALENTS: dict[str, int] = {
    "exchange": EXCHANGE_LAYER_EQUIVALENT,
    "savepoint": SAVEPOINT_LAYER_EQUIVALENT,
    "verify_probe": VERIFY_PROBE_LAYER_EQUIVALENT,
}


def op_layer_equivalents(op: StageOp) -> int:
    """Estimated cost of one op, in layer-application equivalents."""
    kind = getattr(op, "kind", "window")
    if kind != "window":
        return NON_WINDOW_LAYER_EQUIVALENTS.get(kind, 0)
    try:
        span = int(op.end) - int(op.start)
        repeats = int(op.repeats)
    # not a failure: a value that is not a number is not one this can read.
    except (TypeError, ValueError, OverflowError):
        return 0
    if span <= 0 or repeats < 1:
        return 0
    return span * repeats


@dataclass
class LayerSchedule:
    """A validated program over the recurrent region [prelude_end, coda_start)."""

    ops: tuple[StageOp, ...]
    name: str = ""

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> LayerSchedule:
        if not isinstance(payload, dict):
            raise ValueError("schedule payload must be a mapping")
        unknown = sorted(set(payload) - {"name", "ops"})
        if unknown:
            raise ValueError(f"schedule contains unknown keys: {unknown}")
        raw_ops = payload.get("ops")
        if not isinstance(raw_ops, list):
            raise ValueError("schedule ops must be a list")
        if len(raw_ops) > 256:
            raise ValueError("schedule contains more than 256 ops")
        parsed: list[StageOp] = []
        for index, op in enumerate(raw_ops):
            if not isinstance(op, dict):
                raise ValueError(f"schedule op{index} must be a mapping")
            kind = op.get("kind", "window")
            if kind not in STAGE_OP_KINDS:
                raise ValueError(
                    f"schedule op{index}.kind must be one of {sorted(STAGE_OP_KINDS)}"
                )
            if kind != "window":
                op_unknown = sorted(set(op) - {"kind", "revert_on_drop"})
                if op_unknown:
                    raise ValueError(
                        f"schedule op{index} ({kind}) contains unknown keys: {op_unknown}"
                    )
                revert_on_drop = op.get("revert_on_drop", False)
                if type(revert_on_drop) is not bool:
                    raise ValueError(
                        f"schedule op{index}.revert_on_drop must be boolean"
                    )
                if revert_on_drop and kind != "verify_probe":
                    raise ValueError(
                        f"schedule op{index}.revert_on_drop only applies to verify_probe"
                    )
                parsed.append(StageOp(kind=kind, revert_on_drop=revert_on_drop))
                continue
            op_unknown = sorted(set(op) - {"start", "end", "repeats", "alpha", "kind"})
            if op_unknown:
                raise ValueError(f"schedule op{index} contains unknown keys: {op_unknown}")
            for key in ("start", "end"):
                if type(op.get(key)) is not int:
                    raise ValueError(f"schedule op{index}.{key} must be an integer")
            repeats = op.get("repeats", 1)
            if type(repeats) is not int:
                raise ValueError(f"schedule op{index}.repeats must be an integer")
            alpha = op.get("alpha")
            alpha_float: float | None = None
            if alpha is not None:
                if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
                    raise ValueError(f"schedule op{index}.alpha must be a finite number")
                try:
                    alpha_float = float(alpha)
                except (TypeError, ValueError, OverflowError) as exc:
                    raise ValueError(
                        f"schedule op{index}.alpha must be a finite number"
                    ) from exc
                if not math.isfinite(alpha_float):
                    raise ValueError(f"schedule op{index}.alpha must be a finite number")
            parsed.append(
                StageOp(
                    start=op["start"],
                    end=op["end"],
                    repeats=repeats,
                    alpha=alpha_float,
                )
            )
        name = payload.get("name", "")
        if not isinstance(name, str):
            raise ValueError("schedule name must be a string")
        return cls(ops=tuple(parsed), name=name[:200])

    @classmethod
    def single_window(cls, prelude_end: int, coda_start: int, repeats: int) -> LayerSchedule:
        return cls(
            ops=(StageOp(prelude_end, coda_start, repeats),),
            name=f"window[{prelude_end}:{coda_start}]x{repeats}",
        )

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "ops": [op.to_dict() for op in self.ops]}

    def canonical_json(self) -> str:
        # name excluded: identity is the program, not its label.
        return json.dumps([op.to_dict() for op in self.ops], sort_keys=True, separators=(",", ":"))

    @property
    def schedule_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    @property
    def total_layer_repeats(self) -> int:
        """Exact window-layer applications. Window ops only, by definition."""
        return sum((op.end - op.start) * op.repeats for op in self.ops)

    @property
    def estimated_layer_equivalents(self) -> int:
        """Estimated cost of the WHOLE program, non-window ops included.

        Distinct from :attr:`total_layer_repeats`, which is an exact count of
        one instruction kind. This is an estimate covering all of them, and is
        named so the difference cannot be mistaken (CP126 bd211d76).
        """
        return sum(op_layer_equivalents(op) for op in self.ops)

    def validate(self, *, prelude_end: int, coda_start: int) -> list[str]:
        """Human-readable violations; empty ⇒ the program may execute.

        The recurrent region itself is validated first. CP126 f905e912: these
        bounds arrive from the caller and were used without any check of their
        types or their relation to each other, so a false topology could make
        an invalid program look valid — the ops were compared against numbers
        that described no real model. An empty list from this method is the
        module's "safe to execute" claim, and a claim measured against
        unchecked bounds is not one.

        This still does not bind the bounds to a LOADED MODEL's layer count;
        that needs a topology receipt from whoever owns the weights, and is
        the remaining half of f905e912.
        """
        problems: list[str] = []
        for name, bound in (("prelude_end", prelude_end), ("coda_start", coda_start)):
            if isinstance(bound, bool) or not isinstance(bound, int):
                problems.append(f"{name} must be an int, got {_safe_display(bound)}")
            elif bound < 0:
                problems.append(f"{name} must be non-negative, got {bound}")
        if problems:
            # The region is unusable, so per-op comparisons against it would be
            # noise at best and false reassurance at worst.
            return problems
        if prelude_end >= coda_start:
            return [
                f"recurrent region is empty or inverted "
                f"(prelude_end={prelude_end} >= coda_start={coda_start})"
            ]
        if not self.ops:
            problems.append("schedule has no ops")
        total_layer_repeats = 0
        window_ops = 0
        verify_probes = 0
        savepoint_seen = False
        for i, op in enumerate(self.ops):
            kind = getattr(op, "kind", "window")
            if kind not in STAGE_OP_KINDS:
                problems.append(f"op{i} unknown kind {_safe_display(kind)}")
                continue
            if kind != "window":
                if kind == "savepoint":
                    savepoint_seen = True
                elif kind == "verify_probe":
                    verify_probes += 1
                    if op.revert_on_drop and not savepoint_seen:
                        problems.append(
                            f"op{i} verify_probe revert_on_drop needs a "
                            "preceding savepoint op"
                        )
                elif op.revert_on_drop:
                    problems.append(
                        f"op{i} revert_on_drop only applies to verify_probe"
                    )
                continue
            window_ops += 1
            if any(type(value) is not int for value in (op.start, op.end, op.repeats)):
                problems.append(f"op{i} start/end/repeats must be integers")
                continue
            if op.start < prelude_end or op.end > coda_start:
                problems.append(
                    f"op{i} [{_safe_display(op.start)}:{_safe_display(op.end)}) "
                    "escapes recurrent region "
                    f"[{prelude_end}:{coda_start})"
                )
            if op.start >= op.end:
                problems.append(
                    f"op{i} empty window "
                    f"[{_safe_display(op.start)}:{_safe_display(op.end)})"
                )
            if op.repeats < 1:
                problems.append(f"op{i} repeats {_safe_display(op.repeats)} < 1")
            if op.alpha is not None:
                try:
                    alpha = float(op.alpha)
                except (TypeError, ValueError, OverflowError):
                    alpha = math.nan
                if (
                    isinstance(op.alpha, bool)
                    or not isinstance(op.alpha, (int, float))
                    or not math.isfinite(alpha)
                    or not 0.0 < alpha <= 1.0
                ):
                    problems.append(
                        f"op{i} alpha {_safe_display(op.alpha)} outside finite (0, 1]"
                    )
            if op.start < op.end and op.repeats >= 1:
                total_layer_repeats += (op.end - op.start) * op.repeats
        if total_layer_repeats > MAX_TOTAL_LAYER_REPEATS:
            problems.append(
                f"total layer repeats {_safe_display(total_layer_repeats)} exceeds "
                f"{MAX_TOTAL_LAYER_REPEATS}"
            )
        # The ceiling has to cover every instruction that spends compute, or a
        # schedule can sit under it while decoding and verifying repeatedly
        # (CP126 bd211d76). Window cost is exact; the rest are declared
        # estimates, so this is reported as an estimate.
        estimated = self.estimated_layer_equivalents
        if estimated > MAX_TOTAL_LAYER_REPEATS and total_layer_repeats <= MAX_TOTAL_LAYER_REPEATS:
            problems.append(
                f"estimated total cost {estimated} layer-equivalents exceeds "
                f"{MAX_TOTAL_LAYER_REPEATS} once non-window ops are counted"
            )
        if self.ops and window_ops == 0:
            problems.append("schedule needs at least one window op")
        if verify_probes > _MAX_VERIFY_PROBES:
            problems.append(
                f"{verify_probes} verify_probe ops exceed the "
                f"{_MAX_VERIFY_PROBES}-probe budget"
            )
        return problems


# ── Reliability-tracked library ─────────────────────────────────────────


@dataclass(frozen=True)
class ScheduleComputeReceipt:
    """Comparable measured work for one arm of a paired schedule trial."""

    layer_apps: int
    estimator_sha256: str

    def __post_init__(self) -> None:
        if (
            type(self.layer_apps) is not int
            or self.layer_apps <= 0
            or self.layer_apps > _MAX_LAYER_APPS
        ):
            raise ValueError("schedule compute layer_apps must be a bounded positive integer")
        _require_sha256(self.estimator_sha256, field_name="estimator_sha256")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ScheduleComputeReceipt:
        if not isinstance(payload, dict) or set(payload) != {"layer_apps", "estimator_sha256"}:
            raise ValueError("schedule compute receipt has an invalid schema")
        return cls(
            layer_apps=payload["layer_apps"],
            estimator_sha256=payload["estimator_sha256"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer_apps": self.layer_apps,
            "estimator_sha256": self.estimator_sha256,
        }


def _resolve_receipts(
    resolver: Callable[[str, str], bool] | None,
    *,
    scorer_receipt_sha256: str,
    verifier_receipt_sha256: str,
) -> bool:
    """Whether the referenced receipts were actually resolved and accepted.

    CP126 487e7c0a: validation checked that these fields LOOK like SHA-256
    strings and never loaded what they point at, so a self-asserted digest
    satisfied the promotion gate. Resolution needs a store this module does
    not own, so the caller supplies one — and when none is supplied the answer
    is False, which is the honest reading of "nothing was checked".
    """
    if resolver is None:
        return False
    try:
        return bool(
            resolver("scorer", scorer_receipt_sha256)
            and resolver("verifier", verifier_receipt_sha256)
        )
    # not a failure: a resolver that refuses has not resolved the receipt, and False
    # is the refusing direction.
    except (TypeError, ValueError, KeyError, OSError, RuntimeError):
        return False


@dataclass(frozen=True)
class PairedScheduleOutcome:
    """One held-out candidate/default comparison and its provenance.

    It used to say "tamper-evident provenance". It is not tamper-evident, and
    CP126 bf285ef2 is that claim: every provenance field and both success
    booleans are supplied by the caller, and ``evidence_binding_sha256`` is an
    UNKEYED canonical hash over those same values. Anyone who can edit or
    construct the ledger can change an outcome and recompute a valid binding.
    That makes the binding an integrity CHECKSUM — it detects accidental
    corruption and mismatched fields — not a defence against a party who can
    write the file.

    The distinction is now explicit rather than implied.
    :attr:`evidence_authenticity` reports which one a consumer is holding, and
    :meth:`verified_provenance` answers whether the referenced receipts were
    actually resolved or merely asserted (CP126 487e7c0a). Real tamper
    evidence needs the Ed25519 issuer in core/governance/capability_chain.py
    signing the binding under an authority registry; until that is wired, a
    promotion gate reading this must treat it as self-reported.
    """

    task_id: str
    task_commitment_sha256: str
    candidate_success: bool
    default_success: bool
    candidate_compute: ScheduleComputeReceipt
    default_compute: ScheduleComputeReceipt
    run_order: str
    held_out: bool
    contamination_scan_passed: bool
    scorer_receipt_sha256: str
    verifier_receipt_sha256: str
    evaluation_run_id: str
    evaluator_build_sha256: str
    model_checkpoint_sha256: str
    evidence_protocol_sha256: str
    evidence_binding_sha256: str
    #: CP126 78c85746: the binding named the CANDIDATE schedule but never the
    #: baseline it was compared against, so different defaults could each be
    #: labelled "default" and aggregated as though they were one comparator.
    #: A paired result means nothing without both halves identified.
    default_schedule_hash: str = ""
    #: What the binding actually proves. See the class docstring.
    evidence_authenticity: str = "unsigned_checksum"
    #: True only when the referenced receipts were resolved and checked, not
    #: when they merely looked like digests (CP126 487e7c0a).
    receipts_resolved: bool = False

    @classmethod
    def create(
        cls,
        *,
        schedule_hash: str,
        domain: str,
        task_id: str,
        task_commitment_sha256: str,
        candidate_success: bool,
        default_success: bool,
        candidate_compute: ScheduleComputeReceipt,
        default_compute: ScheduleComputeReceipt,
        run_order: str,
        held_out: bool,
        contamination_scan_passed: bool,
        scorer_receipt_sha256: str,
        verifier_receipt_sha256: str,
        evaluation_run_id: str,
        evaluator_build_sha256: str,
        model_checkpoint_sha256: str,
        evidence_protocol_sha256: str,
        default_schedule_hash: str = "",
        receipt_resolver: Callable[[str, str], bool] | None = None,
    ) -> PairedScheduleOutcome:
        if not isinstance(candidate_compute, ScheduleComputeReceipt) or not isinstance(
            default_compute, ScheduleComputeReceipt
        ):
            raise ValueError("paired schedule compute receipts are required")
        values: dict[str, Any] = {
            "task_id": task_id,
            "task_commitment_sha256": task_commitment_sha256,
            "candidate_success": candidate_success,
            "default_success": default_success,
            "candidate_compute": candidate_compute.to_dict(),
            "default_compute": default_compute.to_dict(),
            "run_order": run_order,
            "held_out": held_out,
            "contamination_scan_passed": contamination_scan_passed,
            "scorer_receipt_sha256": scorer_receipt_sha256,
            "verifier_receipt_sha256": verifier_receipt_sha256,
            "evaluation_run_id": evaluation_run_id,
            "evaluator_build_sha256": evaluator_build_sha256,
            "model_checkpoint_sha256": model_checkpoint_sha256,
            # CP126 78c85746: the baseline is half the comparison, so it is
            # part of the binding. Without it, two trials against DIFFERENT
            # defaults both say "default" and aggregate as one comparator.
            "default_schedule_hash": default_schedule_hash,
            "evidence_protocol_sha256": evidence_protocol_sha256,
        }
        binding = cls.binding_sha256(
            schedule_hash=schedule_hash,
            domain=domain,
            values=values,
        )
        outcome = cls(
            task_id=task_id,
            task_commitment_sha256=task_commitment_sha256,
            candidate_success=candidate_success,
            default_success=default_success,
            candidate_compute=candidate_compute,
            default_compute=default_compute,
            run_order=run_order,
            held_out=held_out,
            contamination_scan_passed=contamination_scan_passed,
            scorer_receipt_sha256=scorer_receipt_sha256,
            verifier_receipt_sha256=verifier_receipt_sha256,
            evaluation_run_id=evaluation_run_id,
            evaluator_build_sha256=evaluator_build_sha256,
            model_checkpoint_sha256=model_checkpoint_sha256,
            evidence_protocol_sha256=evidence_protocol_sha256,
            evidence_binding_sha256=binding,
            default_schedule_hash=default_schedule_hash,
            receipts_resolved=_resolve_receipts(
                receipt_resolver,
                scorer_receipt_sha256=scorer_receipt_sha256,
                verifier_receipt_sha256=verifier_receipt_sha256,
            ),
        )
        outcome.validate(schedule_hash=schedule_hash, domain=domain)
        return outcome

    @staticmethod
    def binding_sha256(*, schedule_hash: str, domain: str, values: dict[str, Any]) -> str:
        _require_sha256(schedule_hash, field_name="schedule_hash")
        normalized_domain = _require_identifier(domain, field_name="domain").lower()
        return _canonical_sha256(
            {
                "domain": normalized_domain,
                "schedule_hash": schedule_hash,
                "outcome": values,
            }
        )

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, Any],
        *,
        schedule_hash: str,
        domain: str,
    ) -> PairedScheduleOutcome:
        expected = {
            "task_id",
            "task_commitment_sha256",
            "candidate_success",
            "default_success",
            "candidate_compute",
            "default_compute",
            "run_order",
            "held_out",
            "contamination_scan_passed",
            "scorer_receipt_sha256",
            "verifier_receipt_sha256",
            "evaluation_run_id",
            "evaluator_build_sha256",
            "model_checkpoint_sha256",
            "evidence_protocol_sha256",
            "evidence_binding_sha256",
            "default_schedule_hash",
        }
        # Verification metadata is optional on the wire: it is not part of the
        # binding, and a ledger written before it existed is still readable.
        optional = {"evidence_authenticity", "receipts_resolved"}
        if not isinstance(payload, dict) or not expected <= set(payload) <= (expected | optional):
            raise ValueError("paired schedule outcome has an invalid schema")
        outcome = cls(
            task_id=payload["task_id"],
            task_commitment_sha256=payload["task_commitment_sha256"],
            candidate_success=payload["candidate_success"],
            default_success=payload["default_success"],
            candidate_compute=ScheduleComputeReceipt.from_dict(payload["candidate_compute"]),
            default_compute=ScheduleComputeReceipt.from_dict(payload["default_compute"]),
            run_order=payload["run_order"],
            held_out=payload["held_out"],
            contamination_scan_passed=payload["contamination_scan_passed"],
            scorer_receipt_sha256=payload["scorer_receipt_sha256"],
            verifier_receipt_sha256=payload["verifier_receipt_sha256"],
            evaluation_run_id=payload["evaluation_run_id"],
            evaluator_build_sha256=payload["evaluator_build_sha256"],
            model_checkpoint_sha256=payload["model_checkpoint_sha256"],
            evidence_protocol_sha256=payload["evidence_protocol_sha256"],
            # Absent on pre-baseline-identity ledgers; validate() then refuses
            # the record, which is correct — an outcome that cannot name its
            # comparator is not usable as paired evidence (CP126 78c85746).
            default_schedule_hash=str(payload.get("default_schedule_hash") or ""),
            receipts_resolved=bool(payload.get("receipts_resolved", False)),
            evidence_binding_sha256=payload["evidence_binding_sha256"],
        )
        outcome.validate(schedule_hash=schedule_hash, domain=domain)
        return outcome

    def verified_provenance(self) -> bool:
        """Whether this outcome rests on anything a third party could check.

        False means every provenance claim is self-reported: the flags are the
        caller's booleans and the binding is an unkeyed hash the same caller
        could recompute. A promotion gate that treats such an outcome as
        evidence is trusting the thing it is supposed to be checking.
        """
        return bool(self.receipts_resolved) and self.evidence_authenticity != "unsigned_checksum"

    def _binding_values(self) -> dict[str, Any]:
        payload = self.to_dict()
        payload.pop("evidence_binding_sha256")
        # Verification METADATA is excluded: it records who checked the
        # evidence, not what the evidence claims. Including it would mean a
        # producer could alter the binding simply by asserting it had verified
        # itself.
        payload.pop("evidence_authenticity", None)
        payload.pop("receipts_resolved", None)
        return payload

    def validate(self, *, schedule_hash: str, domain: str) -> None:
        _require_identifier(self.task_id, field_name="task_id")
        _require_sha256(self.task_commitment_sha256, field_name="task_commitment_sha256")
        if type(self.candidate_success) is not bool or type(self.default_success) is not bool:
            raise ValueError("paired schedule outcomes must contain boolean arm results")
        if self.run_order not in _RUN_ORDERS:
            raise ValueError("run_order must be candidate_first or default_first")
        # CP126 487e7c0a: these are SELF-ASSERTED booleans. Requiring them to
        # be True stops an outcome that admits it was contaminated, and stops
        # nothing else — a caller that sets them satisfies the gate. They are
        # still required, because an honest producer must state them, but
        # `receipts_resolved` is what says whether anything was checked, and a
        # promotion gate must read that rather than these.
        if self.held_out is not True:
            raise ValueError("schedule promotion evidence must be held out")
        if self.contamination_scan_passed is not True:
            raise ValueError("schedule promotion evidence failed contamination screening")
        # CP126 78c85746: a paired outcome without its baseline identified is
        # not a paired outcome.
        _require_sha256(self.default_schedule_hash, field_name="default_schedule_hash")
        if self.default_schedule_hash == schedule_hash:
            raise ValueError("candidate and default schedule hashes are identical")
        for field_name in (
            "scorer_receipt_sha256",
            "verifier_receipt_sha256",
            "evaluator_build_sha256",
            "model_checkpoint_sha256",
            "evidence_protocol_sha256",
            "evidence_binding_sha256",
        ):
            _require_sha256(getattr(self, field_name), field_name=field_name)
        _require_identifier(self.evaluation_run_id, field_name="evaluation_run_id")
        if self.candidate_compute.estimator_sha256 != self.default_compute.estimator_sha256:
            raise ValueError("paired schedule arms used different compute estimators")
        larger = max(self.candidate_compute.layer_apps, self.default_compute.layer_apps)
        allowed_delta = max(1, math.ceil(larger * COMPUTE_MATCH_RELATIVE_TOLERANCE))
        if abs(self.candidate_compute.layer_apps - self.default_compute.layer_apps) > allowed_delta:
            raise ValueError("paired schedule arms exceeded the compute matching tolerance")
        expected_binding = self.binding_sha256(
            schedule_hash=schedule_hash,
            domain=domain,
            values=self._binding_values(),
        )
        if self.evidence_binding_sha256 != expected_binding:
            raise ValueError("paired schedule evidence binding does not match its contents")

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_commitment_sha256": self.task_commitment_sha256,
            "candidate_success": self.candidate_success,
            "default_success": self.default_success,
            "candidate_compute": self.candidate_compute.to_dict(),
            "default_compute": self.default_compute.to_dict(),
            "run_order": self.run_order,
            "held_out": self.held_out,
            "contamination_scan_passed": self.contamination_scan_passed,
            "scorer_receipt_sha256": self.scorer_receipt_sha256,
            "verifier_receipt_sha256": self.verifier_receipt_sha256,
            "evaluation_run_id": self.evaluation_run_id,
            "evaluator_build_sha256": self.evaluator_build_sha256,
            "model_checkpoint_sha256": self.model_checkpoint_sha256,
            "evidence_protocol_sha256": self.evidence_protocol_sha256,
            # Part of the binding: the comparison is meaningless without it.
            "default_schedule_hash": self.default_schedule_hash,
            "evidence_binding_sha256": self.evidence_binding_sha256,
            # NOT part of the binding — these describe how much anyone
            # CHECKED the evidence, not what the evidence says. Binding them
            # would let a producer change the claim by asserting it verified
            # itself, and would break every existing binding.
            "evidence_authenticity": self.evidence_authenticity,
            "receipts_resolved": self.receipts_resolved,
        }


@dataclass
class ScheduleRecord:
    schedule: LayerSchedule
    domain: str
    outcomes: dict[str, PairedScheduleOutcome] = field(default_factory=dict)

    @property
    def trials(self) -> int:
        return len(self.outcomes)

    @property
    def successes(self) -> int:
        return sum(int(outcome.candidate_success) for outcome in self.outcomes.values())

    @property
    def default_successes(self) -> int:
        return sum(int(outcome.default_success) for outcome in self.outcomes.values())

    @property
    def reliability_lb(self) -> float:
        return wilson_lower_bound(self.successes, self.trials)

    @property
    def default_reliability_ub(self) -> float:
        return wilson_upper_bound(self.default_successes, self.trials)

    def _profile(self) -> tuple[str, str, str] | None:
        profiles = {
            (
                outcome.evaluator_build_sha256,
                outcome.model_checkpoint_sha256,
                outcome.evidence_protocol_sha256,
            )
            for outcome in self.outcomes.values()
        }
        if len(profiles) != 1:
            return None
        return next(iter(profiles))

    def promotion_ready(self, *, min_trials: int) -> bool:
        if self.trials < min_trials or self._profile() is None:
            return False
        candidate_first = sum(
            outcome.run_order == "candidate_first" for outcome in self.outcomes.values()
        )
        default_first = self.trials - candidate_first
        return (
            abs(candidate_first - default_first) <= 1
            and self.reliability_lb > self.default_reliability_ub
        )

    def add_outcome(self, outcome: PairedScheduleOutcome, *, allow_identical: bool = False) -> bool:
        outcome.validate(schedule_hash=self.schedule.schedule_hash, domain=self.domain)
        existing = self.outcomes.get(outcome.task_id)
        if existing is not None:
            if allow_identical and existing == outcome:
                return False
            raise ValueError(f"duplicate or conflicting schedule task_id: {outcome.task_id}")
        for prior in self.outcomes.values():
            if prior.task_commitment_sha256 == outcome.task_commitment_sha256:
                raise ValueError("replayed schedule task commitment")
            if prior.scorer_receipt_sha256 == outcome.scorer_receipt_sha256:
                raise ValueError("replayed schedule scorer receipt")
            if prior.verifier_receipt_sha256 == outcome.verifier_receipt_sha256:
                raise ValueError("replayed schedule verifier receipt")
        current_profile = self._profile()
        new_profile = (
            outcome.evaluator_build_sha256,
            outcome.model_checkpoint_sha256,
            outcome.evidence_protocol_sha256,
        )
        if current_profile is not None and current_profile != new_profile:
            raise ValueError("schedule evidence profile changed within one candidate record")
        self.outcomes[outcome.task_id] = outcome
        return True

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ScheduleRecord:
        if not isinstance(payload, dict) or set(payload) != {"schedule", "domain", "outcomes"}:
            raise ValueError("schedule record has an invalid schema")
        schedule = LayerSchedule.from_dict(payload["schedule"])
        domain = _require_identifier(payload["domain"], field_name="domain").lower()
        raw_outcomes = payload["outcomes"]
        if not isinstance(raw_outcomes, list) or len(raw_outcomes) > 100_000:
            raise ValueError("schedule outcomes must be a bounded list")
        record = cls(schedule=schedule, domain=domain)
        for raw_outcome in raw_outcomes:
            record.add_outcome(
                PairedScheduleOutcome.from_dict(
                    raw_outcome,
                    schedule_hash=schedule.schedule_hash,
                    domain=domain,
                )
            )
        return record

    def to_dict(self) -> dict[str, Any]:
        return {
            "schedule": self.schedule.to_dict(),
            "domain": self.domain,
            "outcomes": [self.outcomes[key].to_dict() for key in sorted(self.outcomes)],
        }


class ScheduleLibrary:
    """Per-domain schedule ledger gated by matched held-out evidence."""

    MIN_TRIALS = 20
    MAX_SAVE_RETRIES = 4

    def __init__(self, path: Path | str | None = None) -> None:
        self._path = Path(path) if path else None
        self._records: dict[tuple[str, str], ScheduleRecord] = {}
        self._revision = 0
        #: Why the library is empty, when it is empty for a REASON. Empty
        #: string means it loaded (possibly to nothing, which is a fact
        #: about the library rather than about the reader).
        self._load_error = ""
        self._lock = threading.RLock()
        if self._path is not None and self._path.exists():
            self._load()

    # ── Persistence (governed writes; loads are plain reads) ───────────
    @staticmethod
    def _parse_store(payload: Any) -> tuple[int, dict[tuple[str, str], ScheduleRecord]]:
        if not isinstance(payload, dict) or set(payload) != {"version", "revision", "records"}:
            raise ValueError("schedule library root has an invalid schema")
        if payload["version"] != SCHEDULE_LIBRARY_SCHEMA_VERSION:
            raise ValueError("unsupported schedule library schema version")
        revision = payload["revision"]
        if type(revision) is not int or revision < 0:
            raise ValueError("schedule library revision must be a non-negative integer")
        raw_records = payload["records"]
        if not isinstance(raw_records, list) or len(raw_records) > 10_000:
            raise ValueError("schedule library records must be a bounded list")
        records: dict[tuple[str, str], ScheduleRecord] = {}
        for raw_record in raw_records:
            record = ScheduleRecord.from_dict(raw_record)
            key = (record.domain, record.schedule.schedule_hash)
            if key in records:
                raise ValueError("duplicate schedule record in persisted library")
            records[key] = record
        return revision, records

    def _read_store(self) -> tuple[int, dict[tuple[str, str], ScheduleRecord]]:
        if self._path is None:
            return 0, {}
        payload = json.loads(self._path.read_text(encoding="utf-8"))
        return self._parse_store(payload)

    def _load(self) -> None:
        try:
            revision, records = self._read_store()
        except (OSError, TypeError, ValueError) as exc:
            # "starting empty" and "is empty" were the same observable state.
            # Downstream status and schedule selection could not tell a
            # library that legitimately holds nothing from one that is
            # corrupt, unreadable by permission, or written under a schema
            # this build does not accept — so a silent loss of every
            # measured schedule looked like a fresh install.
            from core.runtime.errors import record_degradation

            self._load_error = f"{type(exc).__name__}: {exc}"
            record_degradation(
                "latent_cortex.schedules",
                exc,
                severity="error",
                action="started with an empty schedule library because the store was unreadable",
                extra={"path": str(self._path)},
            )
            logger.error(
                "Schedule library unreadable at %s: %s - starting empty",
                self._path,
                exc,
            )
            return
        with self._lock:
            self._load_error = ""
            self._revision = revision
            self._records = records

    def _merge_records(self, incoming: dict[tuple[str, str], ScheduleRecord]) -> None:
        for key, incoming_record in incoming.items():
            current = self._records.get(key)
            if current is None:
                self._records[key] = incoming_record
                continue
            for task_id in sorted(incoming_record.outcomes):
                current.add_outcome(
                    incoming_record.outcomes[task_id],
                    allow_identical=True,
                )

    def _serialized_payload(self, revision: int) -> bytes:
        records = [self._records[key].to_dict() for key in sorted(self._records)]
        return json.dumps(
            {
                "version": SCHEDULE_LIBRARY_SCHEMA_VERSION,
                "revision": revision,
                "records": records,
            },
            indent=1,
            sort_keys=True,
        ).encode("utf-8")

    @property
    def load_error(self) -> str:
        """Why this library is empty, or "" when it genuinely loaded.

        The distinction selection needs: an empty library is a reason to
        explore, an unreadable one is a reason to stop and look.
        """
        return self._load_error

    def save(self) -> bool:
        if self._path is None:
            return False
        try:
            from core.brain.llm.latent_cortex.persistence import (
                StaleScheduleLibraryError,
                get_latent_cortex_persistence,
            )

            with self._lock:
                for _attempt in range(self.MAX_SAVE_RETRIES):
                    expected_revision = self._revision
                    next_revision = expected_revision + 1
                    try:
                        get_latent_cortex_persistence().save_schedule_library(
                            self._path,
                            self._serialized_payload(next_revision),
                            expected_revision=expected_revision,
                        )
                    except StaleScheduleLibraryError:
                        disk_revision, disk_records = self._read_store()
                        self._merge_records(disk_records)
                        self._revision = disk_revision
                        continue
                    self._revision = next_revision
                    return True
                raise RuntimeError("schedule library remained stale after bounded CAS retries")
        except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
            from core.runtime.errors import record_degradation

            record_degradation(
                "latent_cortex",
                exc,
                action="kept schedule library in memory after persist failed",
            )
            return False

    # ── Evidence ────────────────────────────────────────────────────────
    def record_paired_outcome(
        self,
        schedule: LayerSchedule,
        domain: str,
        outcome: PairedScheduleOutcome,
    ) -> ScheduleRecord:
        if not isinstance(schedule, LayerSchedule):
            raise ValueError("schedule outcome requires a LayerSchedule")
        normalized_domain = _require_identifier(domain, field_name="domain").lower()
        if not isinstance(outcome, PairedScheduleOutcome):
            raise ValueError("schedule outcome requires a PairedScheduleOutcome")
        key = (normalized_domain, schedule.schedule_hash)
        with self._lock:
            record = self._records.get(key)
            if record is None:
                record = ScheduleRecord(schedule=schedule, domain=normalized_domain)
                self._records[key] = record
            record.add_outcome(outcome)
            return record

    def best_for_domain(
        self, domain: str, *, prelude_end: int, coda_start: int, default_repeats: int
    ) -> LayerSchedule:
        default = LayerSchedule.single_window(prelude_end, coda_start, default_repeats)
        # CP126 4b6e3234: candidates were validated and the DEFAULT was not,
        # yet the default is what gets returned whenever no candidate wins —
        # which is the common case. An inverted region, non-positive repeats
        # or out-of-topology bounds therefore reached execution through the
        # one path nothing checked, straight past the module's promise that an
        # invalid program never touches the model.
        #
        # This fails closed rather than degrading: if even the trivial
        # single-window program is invalid for these bounds, the caller has
        # asked for a schedule over a region that cannot be executed, and
        # returning something unrunnable would only move the failure somewhere
        # less legible.
        default_problems = default.validate(
            prelude_end=prelude_end, coda_start=coda_start
        )
        if default_problems:
            raise ValueError(
                "default schedule is not executable for this topology: "
                + "; ".join(default_problems[:3])
            )
        normalized_domain = str(domain or "").strip().lower()
        with self._lock:
            records = list(self._records.items())

        best, best_lb = default, 0.0
        for (record_domain, _), record in records:
            if record_domain != normalized_domain:
                continue
            if record.schedule.schedule_hash == default.schedule_hash:
                continue
            if record.schedule.validate(prelude_end=prelude_end, coda_start=coda_start):
                continue
            if not record.promotion_ready(min_trials=self.MIN_TRIALS):
                continue
            if record.reliability_lb > best_lb:
                best, best_lb = record.schedule, record.reliability_lb
        return best

    def status(self) -> dict[str, Any]:
        """A snapshot of one ledger state, not a blend of several.

        CP126 e7bc7faa. The record list and revision were copied under the
        lock, but each record's ``trials`` was summed AFTER releasing it —
        and the records are mutable, so a concurrent observation could raise
        a trial count that the reported revision does not cover. The
        returned counts therefore need not describe any state the ledger was
        ever actually in, which is exactly what makes a status surface
        untrustworthy for the gates that read it.

        Everything is now read inside the lock, so the counts, the domain
        breakdown and the revision are one consistent observation.
        """
        domains: dict[str, int] = {}
        with self._lock:
            revision = self._revision
            record_count = len(self._records)
            observations = 0
            for (domain, _schedule_hash), record in self._records.items():
                domains[domain] = domains.get(domain, 0) + 1
                observations += int(record.trials)
        return {
            "records": record_count,
            "observations": observations,
            "domains": domains,
            "revision": revision,
        }


# ── Evolutionary schedule search ────────────────────────────────────────


@dataclass
class SearchResult:
    """The winner of a schedule search, and what would let you re-run it.

    CP126 74987822. The result carried only the best schedule, a scalar
    score, an evaluation count and a hash/score history. That is enough to
    USE the winner and not nearly enough to audit it: nothing recorded the
    seed, who produced the scores, what budget was spent, or on which model
    topology — so a reported gain could not be reproduced, and no score
    could be attributed to an evaluator.
    """

    best: LayerSchedule
    best_score: float
    evaluated: int
    history: list[dict[str, Any]] = field(default_factory=list)
    # Held-out score of the winner, when a separate evaluator was supplied.
    holdout_score: float | None = None
    # ── Reproduction ────────────────────────────────────────────────────
    # The seed and the RNG state after the search, so a rerun can be shown
    # to follow the same trajectory rather than merely a similar one.
    seed: int | None = None
    rng_state_sha256: str = ""
    # ── Attribution ─────────────────────────────────────────────────────
    # Who produced the scores, and against what. A score with no evaluator
    # identity cannot be challenged or replicated by anyone else.
    evaluator_id: str = ""
    evaluator_build_sha256: str = ""
    topology: dict[str, Any] = field(default_factory=dict)
    # ── Budget ──────────────────────────────────────────────────────────
    # What the search was allowed and what it actually spent: a winner found
    # after exhausting the budget is a different claim from one found early.
    population: int = 0
    generations: int = 0
    budget_evaluations: int = 0
    budget_exhausted: bool = False
    wall_seconds: float = 0.0
    # Per-evaluation receipts, when the caller supplied them. Empty means
    # unreceipted, which callers must not read as clean.
    evaluation_receipts: list[dict[str, Any]] = field(default_factory=list)

    def reproduction_gaps(self) -> list[str]:
        """What is missing before this result could be independently re-run.

        Empty means the search is reproducible and attributable from the
        result alone. Anything listed is a reason a reported gain should not
        yet be treated as established.
        """
        gaps: list[str] = []
        if self.seed is None:
            gaps.append("seed_absent")
        if not self.evaluator_id:
            gaps.append("evaluator_unidentified")
        if not self.topology:
            gaps.append("topology_unrecorded")
        if self.budget_evaluations <= 0:
            gaps.append("budget_unrecorded")
        if not self.evaluation_receipts:
            gaps.append("per_evaluation_receipts_absent")
        return gaps

    def generalization_gap(self) -> float | None:
        """Search score minus held-out score.

        A large gap means the search found the search set, not a better
        program -- and in the final number that looks identical to a
        discovery unless it is reported.
        """
        if self.holdout_score is None:
            return None
        return self.best_score - self.holdout_score

    def overfit_warning(self, *, tolerance: float = 0.1) -> bool:
        gap = self.generalization_gap()
        return gap is not None and gap > tolerance


class ScheduleSearch:
    """Budgeted, seeded evolutionary search over schedule programs.

    The evaluator is the honesty boundary: it must return a score derived
    from VERIFIED task outcomes (the experiments harness provides one). The
    search itself never inspects model internals — it only proposes programs
    and listens to evidence.
    """

    def __init__(
        self,
        *,
        prelude_end: int,
        coda_start: int,
        max_repeats: int = 8,
        seed: int = 0,
    ) -> None:
        if coda_start - prelude_end < 2:
            raise ValueError("recurrent region too small to search")
        self._p = prelude_end
        self._c = coda_start
        self._max_repeats = max_repeats
        self._rng = random.Random(seed)

    # ── Mutation operators ──────────────────────────────────────────────
    def _mutate(self, schedule: LayerSchedule) -> LayerSchedule:
        ops = list(schedule.ops)
        choice = self._rng.random()
        idx = self._rng.randrange(len(ops))
        op = ops[idx]
        if choice < 0.30:  # adjust repeats
            delta = self._rng.choice((-2, -1, 1, 2))
            ops[idx] = StageOp(op.start, op.end, max(1, min(self._max_repeats, op.repeats + delta)), op.alpha)
        elif choice < 0.55:  # shift window
            shift = self._rng.choice((-2, -1, 1, 2))
            start = max(self._p, min(op.start + shift, self._c - 1))
            end = max(start + 1, min(op.end + shift, self._c))
            ops[idx] = StageOp(start, end, op.repeats, op.alpha)
        elif choice < 0.75 and op.end - op.start >= 2:  # split stage
            mid = self._rng.randrange(op.start + 1, op.end)
            ops[idx : idx + 1] = [
                StageOp(op.start, mid, op.repeats, op.alpha),
                StageOp(mid, op.end, op.repeats, op.alpha),
            ]
        elif choice < 0.90 and len(ops) >= 2:  # merge adjacent stages
            j = min(idx + 1, len(ops) - 1)
            if j != idx:
                a, b = ops[idx], ops[j]
                merged = StageOp(min(a.start, b.start), max(a.end, b.end), max(a.repeats, b.repeats), a.alpha)
                ops[idx : j + 1] = [merged]
        else:  # adjust alpha
            alpha = op.alpha if op.alpha is not None else 0.5
            alpha = min(1.0, max(0.05, alpha + self._rng.choice((-0.15, -0.05, 0.05, 0.15))))
            ops[idx] = StageOp(op.start, op.end, op.repeats, round(alpha, 4))
        mutated = LayerSchedule(ops=tuple(ops), name="searched")
        if mutated.validate(prelude_end=self._p, coda_start=self._c):
            return schedule  # invalid mutation ⇒ keep parent
        return mutated

    def run(
        self,
        evaluator: Callable[[LayerSchedule], float],
        *,
        population: int = 6,
        generations: int = 4,
        seed_schedule: LayerSchedule | None = None,
        holdout_evaluator: Callable[[LayerSchedule], float] | None = None,
        max_layer_apps: int | None = None,
    ) -> SearchResult:
        """Evolve schedules; optionally verify the winner on held-out tasks.

        ``holdout_evaluator`` scores the finalists ONCE, after the search is
        finished. Anima Rationis line 138 requires schedules be searched
        against held-out verified tasks: a schedule selected on the tasks it
        is scored on is a memorized answer key, and in the final number that
        is indistinguishable from a discovery. Passing the SAME callable for
        both is refused rather than silently permitted.

        ``max_layer_apps`` caps compute so a "better" schedule cannot win by
        simply running more layer applications than the baseline.
        """
        if population < 2 or population > 128:
            raise ValueError("population outside [2, 128]")
        if generations < 1 or generations > 256:
            raise ValueError("generations outside [1, 256]")
        if holdout_evaluator is not None and holdout_evaluator is evaluator:
            raise ValueError(
                "search and held-out evaluators must be different task sets: "
                "selecting on the tasks being scored produces an answer key"
            )
        base = seed_schedule or LayerSchedule.single_window(self._p, self._c, 4)
        violations = base.validate(prelude_end=self._p, coda_start=self._c)
        if violations:
            raise ValueError(f"seed schedule invalid: {violations}")

        scored: dict[str, tuple[LayerSchedule, float]] = {}

        def score(s: LayerSchedule) -> float:
            key = s.schedule_hash
            if key not in scored:
                value = float(evaluator(s))
                if not math.isfinite(value):
                    raise ValueError("schedule evaluator returned a non-finite score")
                scored[key] = (s, value)
            return scored[key][1]

        def admissible(candidate: LayerSchedule) -> bool:
            if max_layer_apps is None:
                return True
            return candidate.total_layer_repeats <= max_layer_apps

        if max_layer_apps is not None and not admissible(base):
            # The seed enters the pool unconditionally, so a budget below it
            # would silently cap nothing while appearing to bound compute.
            raise ValueError(
                f"max_layer_apps={max_layer_apps} is below the seed "
                f"schedule's own {base.total_layer_repeats} layer repeats; "
                "the budget would bound nothing"
            )

        pool = [base]
        guard = 0
        while len(pool) < population and guard < population * 50:
            guard += 1
            child = self._mutate(base)
            if admissible(child):
                pool.append(child)
        while len(pool) < population:
            pool.append(base)
        history: list[dict[str, Any]] = []
        for gen in range(generations):
            ranked = sorted(pool, key=score, reverse=True)
            survivors = ranked[: max(2, population // 2)]
            history.append(
                {
                    "generation": gen,
                    "best_hash": survivors[0].schedule_hash,
                    "best_score": score(survivors[0]),
                    "pool": [s.schedule_hash for s in ranked],
                }
            )
            children: list[LayerSchedule] = []
            guard = 0
            need = population - len(survivors)
            while len(children) < need and guard < need * 50:
                guard += 1
                child = self._mutate(self._rng.choice(survivors))
                if admissible(child):
                    children.append(child)
            pool = survivors + children

        best_hash = max(scored, key=lambda k: scored[k][1])
        best, best_score = scored[best_hash]
        holdout = None
        if holdout_evaluator is not None:
            # Scored ONCE, after the search finished. Scoring it during
            # selection would make the held-out set part of the search set.
            holdout = float(holdout_evaluator(best))
            if not math.isfinite(holdout):
                raise ValueError("held-out evaluator returned a non-finite score")
        return SearchResult(
            best=best,
            best_score=best_score,
            evaluated=len(scored),
            history=history,
            holdout_score=holdout,
        )


__all__ = [
    "COMPUTE_MATCH_RELATIVE_TOLERANCE",
    "LayerSchedule",
    "MAX_TOTAL_LAYER_REPEATS",
    "PairedScheduleOutcome",
    "SCHEDULE_LIBRARY_SCHEMA_VERSION",
    "ScheduleComputeReceipt",
    "ScheduleLibrary",
    "ScheduleRecord",
    "ScheduleSearch",
    "SearchResult",
    "StageOp",
]
