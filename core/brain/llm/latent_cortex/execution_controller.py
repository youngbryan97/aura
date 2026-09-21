"""Learned per-problem execution controller — evidence-gated, never vibes.

The schedule library already promotes validated layer programs; the
allocation already scales with stakes/uncertainty. What neither does is
LEARN which execution configuration suits which kind of problem. This
controller closes that loop as a conservative contextual bandit:

    context bucket  = (domain, facet signature, stakes band, uncertainty band)
    arm             = a bounded, validated tweak over the base allocation
                      (deeper recurrence / wider branches / probe-guided
                      bytecode / lean fast weights)
    reward          = the episode's VERIFIED outcome (task-verifier best
                      score), never convergence prettiness

Selection is Wilson-bounded and evidence-gated: an arm may override the
base allocation only when its pessimistic (lower-bound) verified success
rate beats the base arm's optimistic (upper-bound) rate on ≥ MIN_TRIALS
graded episodes in that context — the same conservatism the Verifier
Foundry applies to verifiers. Until then the controller only OBSERVES
(base allocation runs, outcomes are recorded) and explores at most one
arm per EXPLORE_EVERY episodes, budget permitting. Every decision is
receipted with the evidence that justified it.

State persists under data/latent_cortex/controller/ through the governed
write gateway; a corrupt ledger degrades to observe-only, never crashes.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import threading
import time
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from core.brain.llm.latent_cortex.epistemic_state import OperationKind
from core.brain.llm.latent_cortex.value_of_computation import (
    ActionEvidence,
    build_evidence_snapshot,
    validate_action_transition,
)
from core.runtime.file_read_gateway import read_stable_bytes

logger = logging.getLogger("Aura.LatentCortex.ExecutionController")

EXECUTION_CONTROLLER_SCHEMA = "aura.latent_execution_controller.v1"

MIN_TRIALS = 12
EXPLORE_EVERY = 4
_MAX_LEDGER_ROWS = 5000
_Z95 = 1.959963984540054

#: Row schema. Rows without it are pre-identity and are not folded — they
#: carry no episode identity and so cannot be de-duplicated (CP126 cbcb73c1).
CONTROLLER_ROW_SCHEMA = "aura.latent_execution_controller.outcome.v2"

#: The evidence in a cell is only comparable within one model + verifier
#: generation. Rows from another provenance are quarantined rather than folded.
_PROVENANCE_ENV = "AURA_CONTROLLER_EVIDENCE_PROVENANCE"
_ACTION_CERTIFICATE_ENV = "AURA_RLC_ACTION_CALIBRATION_CERTIFICATE"
_ACTION_POLICY_ENV = "AURA_RLC_ACTION_CALIBRATION_POLICY"
_ACTION_TRUST_ROOT_ENV = "AURA_RLC_ACTION_CALIBRATION_TRUST_ROOT"


# Bounded arm menu: every arm is a small, validated delta over the base
# allocation. Arms may only tighten or reshape — never exceed the absolute
# caps the service already enforces.
ARMS: dict[str, dict[str, Any]] = {
    "base": {},
    "deeper_recurrence": {"max_steps_delta": 4, "n_branches_cap": 2},
    "wider_branches": {"n_branches_delta": 1, "max_steps_delta": -2},
    "probe_guided_bytecode": {"bytecode_probes": True},
    "lean_fast_weights": {"fast_weights_max_layers_cap": 2},
}

_WORD_RE = re.compile(r"[^\W\d_]{4,}", re.UNICODE)

#: Family size for the repeated arm-vs-base comparisons (CP126 91f196dc).
_ARM_FAMILY_SIZE = max(1, len(ARMS) - 1)

#: Worst-case predicted cost multiplier per arm, used to refuse an arm whose
#: predicted cost cannot fit the caller's remaining budget (CP126 c53c85b8).
_ARM_COST_FACTOR: dict[str, float] = {
    "base": 1.0,
    "deeper_recurrence": 2.0,
    "wider_branches": 1.8,
    "probe_guided_bytecode": 1.6,
    "lean_fast_weights": 1.1,
}

#: An arm may not be promoted if its measured latency is materially worse than
#: base — "budget permitting" has to mean something (CP126 171f2e25).
_MAX_LATENCY_REGRESSION = 1.25


def _bonferroni_z(family_size: int) -> float:
    """Two-sided z for alpha/family_size at nominal alpha=0.05.

    CP126 91f196dc: every choose() tested all four non-base arms against base
    at a nominal 95% bound and repeated those looks after each new outcome, so
    the eventual false-promotion rate was not the claimed conservative gate.
    Splitting alpha across the family is the minimum honest correction; the
    boundary actually used is receipted in the decision evidence.
    """
    family = max(1, int(family_size))
    if family == 1:
        return _Z95
    alpha = 0.05 / family
    # Inverse normal CDF (Acklam's rational approximation) at 1 - alpha/2.
    p = 1.0 - alpha / 2.0
    a = (-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00)
    b = (-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00)
    p_low, p_high = 0.02425, 1.0 - 0.02425
    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0)
    if p > p_high:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1.0)


def _wilson(successes: int, n: int, *, upper: bool, z: float = _Z95) -> float:
    """Wilson bound for binary, independently graded outcomes.

    A mean verifier score is not a binomial success count. Keeping that score
    as descriptive telemetry is useful, but feeding its sum into this formula
    creates fictitious fractional trials and invalid confidence intervals.

    ``z`` lets the caller pass a multiplicity-corrected critical value.
    """

    if n <= 0:
        return 1.0 if upper else 0.0
    if successes < 0 or successes > n:
        raise ValueError("Wilson successes must be an integer in [0, n]")
    p_hat = successes / n
    z2 = z * z
    denominator = 1.0 + z2 / n
    center = p_hat + z2 / (2 * n)
    margin = z * math.sqrt(
        (p_hat * (1.0 - p_hat) + z2 / (4 * n)) / n
    )
    bound = (center + margin) / denominator if upper else (center - margin) / denominator
    return max(0.0, min(1.0, bound))


def evidence_provenance() -> str:
    """Identity of the model+verifier generation this evidence belongs to.

    CP126 cbcb73c1: without it, evidence gathered under an older checkpoint or
    verifier survived the upgrade and kept voting.
    """
    return str(os.environ.get(_PROVENANCE_ENV, "") or "default").strip()[:64]


def _canonical_bucket_component(value: Any) -> str:
    """Escape a bucket component so components cannot collide across the delimiter.

    CP126 84345b43: raw strings were joined with '|', so a domain containing the
    delimiter could forge another bucket's key.
    """
    text = str(value or "")
    return text.replace("\\", "\\\\").replace("|", "\\p")[:24]


def _validated_region(
    region: Any, *, model_layers: int | None = None
) -> tuple[int, int] | None:
    """A usable (start, end) recurrent region, or None.

    CP126 094e4b5d: boundaries went into schedule ops unvalidated.
    """
    if region is None or isinstance(region, (str, bytes, dict)):
        return None
    try:
        items = list(region)
    except TypeError:
        # not a failure: the docstring says boundaries went into schedule
        # ops unvalidated, and this is the validation saying no.
        return None
    if len(items) != 2:
        return None
    start_raw, end_raw = items
    if isinstance(start_raw, bool) or isinstance(end_raw, bool):
        return None
    try:
        start, end = int(start_raw), int(end_raw)
    except (TypeError, ValueError, OverflowError):
        # not a failure: a boundary that is not two integers is not a
        # region, which the range guards below also refuse.
        return None
    if start < 0 or end <= start:
        return None
    if end - start > 512:
        return None
    if model_layers is not None:
        try:
            layers = int(model_layers)
        except (TypeError, ValueError):
            # not a failure: a layer count this cannot read bounds nothing,
            # and the check below only acts on a positive one.
            layers = 0
        if layers > 0 and end > layers:
            return None
    return start, end


def _schedule_sha256(schedule: Any) -> str:
    try:
        encoded = json.dumps(
            schedule, sort_keys=True, separators=(",", ":"), default=str
        ).encode("utf-8")
    except (TypeError, ValueError):
        encoded = repr(schedule).encode("utf-8", errors="replace")
    return hashlib.sha256(encoded).hexdigest()


def _unit_or_none(value: Any) -> float | None:
    """A finite value in [0, 1], else None (NaN was silently classified 'low')."""
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        # not a failure: the docstring says None, and NaN being silently
        # classified 'low' is the defect this exists to stop.
        return None
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        return None
    return number


def context_bucket(
    objective: str, domain: str, stakes: float, uncertainty: float
) -> str:
    """Coarse, deterministic context key: generalizes, never memorizes.

    CP126 84345b43: stakes/uncertainty were compared directly with no type or
    finite check (so a malformed value could raise and NaN was silently banded
    'low'), and components were joined raw so a delimiter inside a domain could
    collide buckets. Values outside [0, 1] are now an explicit 'invalid' band —
    never quietly the lowest one — and components are escaped.
    """
    try:
        from core.brain.llm.latent_cortex.output_quality import request_facets

        facets = ",".join(sorted(request_facets(str(objective or ""))))
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
        # not a failure: an objective with no readable facets is keyed on
        # the rest, which the comment above says is never the lowest.
        facets = ""
    words = len(_WORD_RE.findall(str(objective or "")))
    length_band = "short" if words < 24 else ("medium" if words < 96 else "long")

    def _band(value: Any) -> str:
        number = _unit_or_none(value)
        if number is None:
            return "invalid"
        return "high" if number >= 0.7 else ("mid" if number >= 0.4 else "low")

    return "|".join(
        [
            _canonical_bucket_component(domain or "general"),
            _canonical_bucket_component(facets or "none"),
            length_band,
            f"s:{_band(stakes)}",
            f"u:{_band(uncertainty)}",
        ]
    )


class ExecutionController:
    """Persistent bandit over execution arms, keyed by context bucket."""

    def __init__(self, root: Path | str | None = None) -> None:
        if root is None:
            try:
                from core.config import DATA_DIR

                root = Path(DATA_DIR) / "latent_cortex" / "controller"
            except (ImportError, AttributeError, RuntimeError, TypeError):
                root = Path("data/latent_cortex/controller")
        self.root = Path(root)
        self.ledger_path = self.root / "outcomes.jsonl"
        self.action_ledger_path = self.root / "action_transitions.jsonl"
        self._cells: dict[tuple[str, str], dict[str, float]] = {}
        self._action_cells: dict[tuple[str, OperationKind], ActionEvidence] = {}
        self._episodes_seen = 0
        self._restore_errors = 0
        # CP126 ce5b26e0: _cells, the counters and persistence were mutated
        # from choose/_fold/record_outcome with no synchronization, so
        # concurrent requests could select on stale counts, lose or double
        # cadence updates, and interleave persistence.
        self._lock = threading.RLock()
        # Episode identities already folded — a copied or replayed row must not
        # inflate trial counts (CP126 cbcb73c1).
        self._seen_episodes: set[str] = set()
        # Decision tokens issued by choose() and not yet consumed
        # (CP126 3b3d44e8).
        self._open_decisions: dict[str, dict[str, Any]] = {}
        # CP126 b36c802d: choose() incremented _episodes_seen and _fold
        # incremented it AGAIN, so the SAME counter carried two different
        # meanings and exploration cadence changed across execution modes and
        # restarts. They are now separate: _episodes_seen counts COMPLETED
        # decision/outcome pairs (evidence, restored from the ledger), while
        # cadence runs off decisions actually made this process.
        self._decisions_made = 0
        self._provenance = evidence_provenance()
        self._rows_on_disk = 0
        self._certified_action_certificate: dict[str, Any] | None = None
        self._certified_action_policy: Any | None = None
        self._certified_action_load_error: str | None = None
        self._restore()
        self._restore_action_transitions()
        self._restore_certified_action_evidence()

    def _restore_certified_action_evidence(self) -> None:
        """Admit a claim-grade action certificate through a pinned root."""

        default_dir = self.root / "calibration"
        configured = (
            os.environ.get(_ACTION_CERTIFICATE_ENV),
            os.environ.get(_ACTION_POLICY_ENV),
            os.environ.get(_ACTION_TRUST_ROOT_ENV),
        )
        explicit = any(value for value in configured)
        paths = (
            Path(configured[0]).expanduser()
            if configured[0]
            else default_dir / "certificate.json",
            Path(configured[1]).expanduser()
            if configured[1]
            else default_dir / "policy.json",
            Path(configured[2]).expanduser()
            if configured[2]
            else None,
        )
        artifact_paths = tuple(path for path in paths if path is not None)
        if not explicit and not any(path.exists() for path in artifact_paths):
            return
        if (
            paths[2] is None
            or not all(path.is_file() for path in artifact_paths)
        ):
            self._certified_action_load_error = (
                "calibration_artifact_set_incomplete"
            )
            logger.warning(
                "Certified action evidence not admitted: %s",
                self._certified_action_load_error,
            )
            return
        try:
            from core.brain.llm.latent_cortex.action_calibration import (
                verify_action_calibration_certificate,
            )
            from core.brain.llm.latent_cortex.campaign_trust import (
                validate_campaign_trust_policy,
            )

            certificate = json.loads(
                read_stable_bytes(
                    cast(Path, paths[0]),
                    max_bytes=64 * 1024 * 1024,
                )
            )
            policy_document = json.loads(
                read_stable_bytes(
                    cast(Path, paths[1]),
                    max_bytes=2 * 1024 * 1024,
                )
            )
            trust_root = read_stable_bytes(
                cast(Path, paths[2]),
                max_bytes=64 * 1024,
            )
            if not isinstance(certificate, Mapping):
                raise ValueError("calibration certificate must be an object")
            candidate = certificate.get("candidate")
            if not isinstance(candidate, Mapping):
                raise ValueError("calibration certificate candidate must be an object")
            campaign_name = candidate.get("campaign_name")
            policy = validate_campaign_trust_policy(
                policy_document,
                trusted_root_public_key_pem=trust_root,
                expected_campaign_name=campaign_name,
                now_unix=int(time.time()),
            )
            verified = verify_action_calibration_certificate(
                certificate,
                policy=policy,
            )
        except (
            ImportError,
            AttributeError,
            KeyError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            self._certified_action_load_error = (
                f"{type(exc).__name__}:{str(exc)[:160]}"
            )
            logger.warning(
                "Certified action evidence not admitted: %s",
                self._certified_action_load_error,
            )
            return
        self._certified_action_certificate = verified
        self._certified_action_policy = policy
        logger.info(
            "Certified action evidence admitted: certificate=%s bucket=%s",
            verified["certificate_sha256"][:12],
            verified["candidate"]["calibration_bucket"],
        )

    # ── Integrity ────────────────────────────────────────────────────────
    def integrity_ok(self) -> bool:
        """False once any ledger row failed to parse/validate.

        CP126 af9ac58e: the module promised corruption degrades to
        observe-only, but errors merely incremented a counter while valid rows
        stayed active and choose() never looked at it. Now a single integrity
        failure blocks every non-base selection until the ledger is repaired
        or quarantined.
        """
        with self._lock:
            return self._restore_errors == 0

    # ── State ────────────────────────────────────────────────────────────
    def _restore(self) -> None:
        try:
            if not self.ledger_path.exists():
                return
            rows = 0
            with open(self.ledger_path, encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    rows += 1
                    try:
                        row = json.loads(line)
                        self._fold(row)
                    except (json.JSONDecodeError, TypeError, ValueError, KeyError):
                        self._restore_errors += 1
            self._rows_on_disk = rows
            if self._restore_errors:
                logger.warning(
                    "Controller ledger had %d unusable row(s) — non-base selection "
                    "is blocked until the ledger is repaired.",
                    self._restore_errors,
                )
        except OSError as exc:
            self._restore_errors += 1
            logger.warning("Controller ledger unreadable — observe-only: %s", exc)

    def _fold(self, row: dict[str, Any]) -> None:
        # CP126 cbcb73c1: rows carried no schema, episode identity, or
        # provenance, so a copied file inflated trial counts and evidence from
        # an incompatible model/verifier generation survived upgrades.
        if row.get("schema") != CONTROLLER_ROW_SCHEMA:
            raise ValueError("controller outcome row schema is invalid")
        provenance = row.get("provenance")
        if not isinstance(provenance, str) or not provenance:
            raise ValueError("controller outcome provenance is missing")
        if provenance != self._provenance:
            # Not corruption — evidence from another generation. Skip it
            # WITHOUT counting an integrity error.
            return
        episode_id = row.get("episode_id")
        if not isinstance(episode_id, str) or not (8 <= len(episode_id) <= 64):
            raise ValueError("controller outcome episode id is invalid")
        if row.get("checked") is not True:
            raise ValueError("controller outcome is not independently checked")
        if not isinstance(row.get("success"), bool):
            raise ValueError("controller outcome success must be boolean")
        bucket = row.get("bucket")
        arm = row.get("arm")
        score = row.get("verified_score")
        if not isinstance(bucket, str) or not bucket or len(bucket) > 160:
            raise ValueError("controller outcome bucket is invalid")
        if not isinstance(arm, str) or arm not in ARMS:
            raise ValueError("controller outcome arm is invalid")
        if (
            not isinstance(score, (int, float))
            or isinstance(score, bool)
            or not math.isfinite(float(score))
            or not 0.0 <= float(score) <= 1.0
        ):
            raise ValueError("controller verified score is invalid")
        latency = row.get("wall_clock_s", 0.0)
        if (
            not isinstance(latency, (int, float))
            or isinstance(latency, bool)
            or not math.isfinite(float(latency))
            or float(latency) < 0.0
        ):
            raise ValueError("controller wall_clock_s is invalid")
        if episode_id in self._seen_episodes:
            # A replayed/duplicated row. Refuse it, and do not treat it as
            # corruption — the file is readable, it is just not new evidence.
            return
        self._seen_episodes.add(episode_id)
        cell = self._cells.setdefault(
            (bucket, arm),
            {"n": 0, "verified_sum": 0.0, "successes": 0, "latency_sum": 0.0},
        )
        cell["n"] += 1
        cell["verified_sum"] += float(score)
        cell["successes"] += int(bool(row.get("success")))
        # CP126 171f2e25: latency was persisted and then discarded, so an
        # arbitrarily slow arm could be promoted despite the budget language.
        cell["latency_sum"] = float(cell.get("latency_sum", 0.0)) + float(latency)
        self._episodes_seen += 1

    def _restore_action_transitions(self) -> None:
        try:
            if not self.action_ledger_path.exists():
                return
            with open(self.action_ledger_path, encoding="utf-8") as handle:
                for line in handle:
                    try:
                        self._fold_action_transition(json.loads(line))
                    except (json.JSONDecodeError, TypeError, ValueError, KeyError):
                        self._restore_errors += 1
        except OSError as exc:
            self._restore_errors += 1
            logger.warning(
                "Controller action ledger unreadable - bootstrap-only: %s",
                exc,
            )

    def _fold_action_transition(self, row: dict[str, Any]) -> None:
        transition = validate_action_transition(row)
        key = (transition["bucket"], OperationKind(transition["action"]))
        current = self._action_cells.get(key, ActionEvidence())
        metrics = transition["metrics"]
        self._action_cells[key] = current.append(
            gain=float(metrics["gain"]),
            cost=float(metrics["cost"]),
        )

    def _append(self, row: dict[str, Any]) -> bool:
        try:
            from core.governance_context import local_internal_governed_scope
            from core.runtime.file_write_gateway import get_file_write_gateway

            line = json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            with local_internal_governed_scope(
                "latent_execution_controller", domain="state_mutation"
            ):
                get_file_write_gateway().append_text(
                    self.ledger_path, line, source="latent_execution_controller"
                )
            return True
        except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
            logger.debug("Controller outcome not persisted: %s", exc)
            return False

    def _append_action_transitions(self, rows: list[dict[str, Any]]) -> bool:
        if not rows:
            return True
        try:
            from core.governance_context import local_internal_governed_scope
            from core.runtime.file_write_gateway import get_file_write_gateway

            payload = "".join(
                json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                for row in rows
            )
            with local_internal_governed_scope(
                "latent_value_of_computation", domain="state_mutation"
            ):
                get_file_write_gateway().append_text(
                    self.action_ledger_path,
                    payload,
                    source="latent_value_of_computation",
                )
            return True
        except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
            logger.debug("Controller action transitions not persisted: %s", exc)
            return False

    # ── Decisions ────────────────────────────────────────────────────────
    def choose(
        self,
        *,
        objective: str,
        domain: str,
        stakes: float,
        uncertainty: float,
        remaining_budget_s: float | None = None,
    ) -> dict[str, Any]:
        """Pick an arm for this episode, with the evidence receipted.

        Exploitation requires separation: arm.lb > base.ub with both arms
        having ≥ MIN_TRIALS in this bucket. Exploration runs one non-base
        arm every EXPLORE_EVERY episodes (round-robin over the least-tried
        arms) so evidence accumulates without destabilizing the live path.
        """
        bucket = context_bucket(objective, domain, stakes, uncertainty)
        z = _bonferroni_z(_ARM_FAMILY_SIZE)
        with self._lock:
            decision: dict[str, Any] = {
                "schema": EXECUTION_CONTROLLER_SCHEMA,
                "bucket": bucket,
                "arm": "base",
                "mode": "observe",
                # CP126 8c7f02a0: observe and explore modes returned an EMPTY
                # evidence object while the module claimed every decision is
                # receipted with its justification. Every mode now carries the
                # decision boundary, trial counts, cadence and budget rationale.
                "evidence": {
                    "decision_boundary": {
                        "family_size": _ARM_FAMILY_SIZE,
                        "nominal_alpha": 0.05,
                        "corrected_alpha": round(0.05 / _ARM_FAMILY_SIZE, 5),
                        "critical_z": round(z, 4),
                        "correction": "bonferroni_over_non_base_arms",
                        "min_trials": MIN_TRIALS,
                    },
                    "integrity_ok": self._restore_errors == 0,
                    "restore_errors": self._restore_errors,
                    "provenance": self._provenance,
                    "episodes_seen": self._episodes_seen,
                    "decisions_made": self._decisions_made,
                    "explore_every": EXPLORE_EVERY,
                },
            }

            # CP126 f1088112: the advertised kill switch was never consulted by
            # choose/apply_arm/record_outcome, so direct users of the class or
            # the singleton kept selecting and learning while it was off.
            if not controller_enabled():
                decision["evidence"]["disabled"] = "AURA_EXECUTION_CONTROLLER"
                decision["mode"] = "disabled"
                return decision

            # CP126 af9ac58e: a partial ledger must not drive exploration or
            # exploitation.
            if self._restore_errors:
                decision["evidence"]["blocked"] = "ledger_integrity_failed"
                decision["mode"] = "observe_only_integrity"
                return decision

            base_cell = self._cells.get((bucket, "base"))
            base_n = int(base_cell["n"]) if base_cell else 0
            base_ub = (
                _wilson(int(base_cell["successes"]), base_n, upper=True, z=z)
                if base_cell and base_n
                else 1.0
            )
            base_latency = (
                float(base_cell.get("latency_sum", 0.0)) / base_n
                if base_cell and base_n
                else 0.0
            )
            decision["evidence"].update({"base_n": base_n, "base_ub": round(base_ub, 4)})

            # CP126 c53c85b8: exploration claimed to be "budget permitting" but
            # selection had no budget input at all. An arm whose predicted
            # worst-case cost exceeds the caller's remaining budget is refused,
            # and the calculation is receipted.
            budget_s: float | None = None
            if remaining_budget_s is not None:
                try:
                    candidate_budget = float(remaining_budget_s)
                except (TypeError, ValueError):
                    candidate_budget = float("nan")
                if math.isfinite(candidate_budget) and candidate_budget > 0.0:
                    budget_s = candidate_budget
            decision["evidence"]["remaining_budget_s"] = budget_s

            def _predicted_cost_s(arm: str) -> float:
                cell = self._cells.get((bucket, arm))
                if cell and int(cell["n"]) > 0:
                    return float(cell.get("latency_sum", 0.0)) / int(cell["n"])
                reference = base_latency if base_latency > 0.0 else 0.0
                return reference * _ARM_COST_FACTOR.get(arm, 1.0)

            def _affordable(arm: str) -> bool:
                if budget_s is None:
                    return True
                predicted = _predicted_cost_s(arm)
                if predicted <= 0.0:
                    return True  # No measured cost yet; nothing to refuse on.
                return predicted <= budget_s

            best_arm, best_lb = "", 0.0
            considered: dict[str, Any] = {}
            for arm in ARMS:
                if arm == "base":
                    continue
                cell = self._cells.get((bucket, arm))
                arm_n = int(cell["n"]) if cell else 0
                entry: dict[str, Any] = {"n": arm_n}
                if not cell or arm_n < MIN_TRIALS or base_n < MIN_TRIALS:
                    entry["skipped"] = "insufficient_trials"
                    considered[arm] = entry
                    continue
                lb = _wilson(int(cell["successes"]), arm_n, upper=False, z=z)
                entry["lb"] = round(lb, 4)
                arm_latency = float(cell.get("latency_sum", 0.0)) / arm_n
                entry["mean_latency_s"] = round(arm_latency, 3)
                if not _affordable(arm):
                    entry["skipped"] = "over_budget"
                    considered[arm] = entry
                    continue
                # CP126 171f2e25: deadline compliance is part of eligibility.
                if (
                    base_latency > 0.0
                    and arm_latency > base_latency * _MAX_LATENCY_REGRESSION
                ):
                    entry["skipped"] = "latency_regression"
                    considered[arm] = entry
                    continue
                if lb > base_ub and lb > best_lb:
                    best_arm, best_lb = arm, lb
                considered[arm] = entry
            decision["evidence"]["candidates"] = considered

            if best_arm:
                decision.update({"arm": best_arm, "mode": "exploit"})
                decision["evidence"].update(
                    {
                        "arm_lb": round(best_lb, 4),
                        "arm_n": int(self._cells[(bucket, best_arm)]["n"]),
                    }
                )
                self._issue_decision(decision)
                return decision

            # Cadence advances once per decision (see __init__); evidence
            # counting lives in _fold and is never double-incremented here.
            self._decisions_made += 1
            if self._decisions_made % EXPLORE_EVERY == 0:
                affordable = [
                    arm for arm in ARMS if arm != "base" and _affordable(arm)
                ]
                if affordable:
                    candidates = sorted(
                        affordable,
                        key=lambda arm: self._cells.get((bucket, arm), {"n": 0})["n"],
                    )
                    decision.update({"arm": candidates[0], "mode": "explore"})
                    decision["evidence"]["explore_reason"] = "least_tried_affordable_arm"
                else:
                    decision["evidence"]["explore_reason"] = "no_affordable_arm"
            self._issue_decision(decision)
            return decision

    def _issue_decision(self, decision: dict[str, Any]) -> None:
        """Bind an immutable token to this decision (CP126 3b3d44e8).

        choose() emitted no identity and record_outcome accepted an arbitrary
        caller-supplied bucket/arm, so a stale, mistaken, or malicious caller
        could credit any arm in any context — including recording a BASE
        execution as a treatment.
        """
        # Caller holds the lock.
        decision_id = uuid.uuid4().hex
        commitment = hashlib.sha256(
            json.dumps(
                {
                    "bucket": decision["bucket"],
                    "arm": decision["arm"],
                    "mode": decision["mode"],
                    "provenance": self._provenance,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        decision["decision_id"] = decision_id
        decision["decision_sha256"] = commitment
        self._open_decisions[decision_id] = {
            "bucket": decision["bucket"],
            "arm": decision["arm"],
            "mode": decision["mode"],
            "sha256": commitment,
            "at": time.time(),
        }
        # Bounded: abandoned decisions must not grow unbounded.
        if len(self._open_decisions) > 512:
            for stale in sorted(
                self._open_decisions,
                key=lambda key: self._open_decisions[key]["at"],
            )[:128]:
                self._open_decisions.pop(stale, None)

    def apply_arm(
        self,
        arm: str,
        config: dict[str, Any],
        *,
        recurrent_region: tuple[int, int] | None = None,
    ) -> dict[str, Any]:
        """Overlay one arm's bounded deltas onto an allocation config.

        ``recurrent_region`` is the (prelude_end, coda_start) of the target
        model; the probe-guided bytecode arm needs it to emit a valid
        program and quietly degenerates to base when it is unknown.
        """
        return self.apply_arm_receipt(
            arm, config, recurrent_region=recurrent_region
        )["config"]

    def apply_arm_receipt(
        self,
        arm: str,
        config: dict[str, Any],
        *,
        recurrent_region: tuple[int, int] | None = None,
        model_layers: int | None = None,
    ) -> dict[str, Any]:
        """Apply an arm and report EXACTLY what happened.

        CP126 ea828a97: choose() could receipt ``probe_guided_bytecode`` as the
        selected treatment while apply_arm quietly left the config unchanged
        when ``recurrent_region`` was None — and returned no applied flag — so
        subsequent outcomes were credited to bytecode that never ran. The
        effective arm falls back to ``base`` and the caller is told.

        CP126 f1088112: the kill switch is enforced here too.
        """
        receipt: dict[str, Any] = {
            "requested_arm": str(arm),
            "effective_arm": str(arm),
            "applied": False,
            "config": dict(config),
            "reason": "",
        }
        if not controller_enabled():
            receipt.update({"effective_arm": "base", "reason": "controller_disabled"})
            return receipt
        if arm not in ARMS:
            receipt.update({"effective_arm": "base", "reason": "unknown_arm"})
            return receipt
        if arm == "base":
            receipt.update({"applied": True, "reason": "base"})
            return receipt
        spec = ARMS.get(arm) or {}
        adjusted = dict(config)
        if "max_steps_delta" in spec:
            adjusted["max_steps"] = max(
                2, min(16, int(config.get("max_steps", 4)) + spec["max_steps_delta"])
            )
        if "n_branches_delta" in spec:
            adjusted["n_branches"] = max(
                1, min(4, int(config.get("n_branches", 1)) + spec["n_branches_delta"])
            )
        if "n_branches_cap" in spec:
            adjusted["n_branches"] = min(
                int(adjusted.get("n_branches", 1)), spec["n_branches_cap"]
            )
        if "fast_weights_max_layers_cap" in spec:
            adjusted["fast_weights_max_layers"] = min(
                int(config.get("fast_weights_max_layers", 4)),
                spec["fast_weights_max_layers_cap"],
            )
        if spec.get("bytecode_probes"):
            # CP126 094e4b5d: ANY non-None two-item value was coerced into
            # start/end and inserted straight into schedule ops — negative,
            # reversed, equal, boolean, oversized, or malformed shapes were
            # never rejected (and a bad shape raised AFTER selection).
            bounds = _validated_region(recurrent_region, model_layers=model_layers)
            if bounds is None:
                # CP126 ea828a97: do not silently run base while still calling
                # this a bytecode episode.
                receipt.update(
                    {
                        "effective_arm": "base",
                        "applied": False,
                        "config": adjusted,
                        "reason": "invalid_or_absent_recurrent_region",
                    }
                )
                return receipt
            start, end = bounds
            repeats = max(2, int(adjusted.get("max_steps", 4)))
            first = max(1, repeats // 2)
            second = max(1, repeats - first)
            schedule = {
                "name": "controller_probe_guided_v1",
                "ops": [
                    {"start": start, "end": end, "repeats": first},
                    {"kind": "savepoint"},
                    {"kind": "verify_probe", "revert_on_drop": True},
                    {"kind": "exchange"},
                    {"start": start, "end": end, "repeats": second},
                    {"kind": "verify_probe", "revert_on_drop": True},
                ],
            }
            # CP126 261201c3: this overwrote a caller-provided validated
            # schedule with no record of what it replaced. Supersede it
            # EXPLICITLY, carrying both identities — in the RECEIPT, because
            # LayerSchedule.from_dict accepts only {name, ops} and any extra
            # key would make the program itself unparseable.
            prior = config.get("schedule")
            if isinstance(prior, dict) and prior:
                receipt["superseded_schedule"] = {
                    "name": str(prior.get("name") or "unnamed"),
                    "sha256": _schedule_sha256(prior),
                    "reason": "probe_guided_bytecode_arm",
                }
            adjusted["schedule"] = schedule
            receipt["schedule_sha256"] = _schedule_sha256(schedule)
        receipt.update({"applied": True, "config": adjusted, "reason": "applied"})
        return receipt

    def record_outcome(
        self,
        *,
        bucket: str,
        arm: str,
        verified_score: float,
        success: bool,
        checked: bool,
        wall_clock_s: float = 0.0,
        decision_id: str = "",
    ) -> bool:
        """Fold one independently checked episode outcome and persist it.

        ``decision_id`` is the token ``choose`` issued for this episode. It is
        REQUIRED (CP126 3b3d44e8): without it a stale, mistaken, or malicious
        caller could credit any arm in any context — including recording a base
        execution as a treatment — and the bucket/arm are taken from the bound
        decision, not from the caller's arguments.
        """
        if not controller_enabled():
            return False
        if checked is not True or not isinstance(success, bool):
            return False
        if not isinstance(decision_id, str) or not decision_id:
            logger.debug("Controller outcome refused: no decision token")
            return False
        with self._lock:
            issued = self._open_decisions.pop(decision_id, None)
        if issued is None:
            logger.debug("Controller outcome refused: unknown/consumed decision token")
            return False
        # The decision is authoritative; caller-supplied identity must agree.
        if str(bucket) != issued["bucket"] or str(arm) != issued["arm"]:
            logger.warning(
                "Controller outcome refused: caller claimed %s/%s but the bound "
                "decision was %s/%s.",
                bucket, arm, issued["bucket"], issued["arm"],
            )
            return False
        bucket, arm = issued["bucket"], issued["arm"]
        if not isinstance(bucket, str) or not bucket or len(bucket) > 160:
            return False
        if not isinstance(arm, str) or arm not in ARMS:
            return False
        if not isinstance(verified_score, (int, float)) or isinstance(
            verified_score, bool
        ):
            return False
        score = float(verified_score)
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            return False
        try:
            elapsed = float(wall_clock_s)
        except (TypeError, ValueError, OverflowError):
            # not a failure: a wall clock that is not a number cannot be
            # accepted, and the finite check below refuses the same thing.
            return False
        if not math.isfinite(elapsed) or elapsed < 0.0:
            return False
        row = {
            "schema": CONTROLLER_ROW_SCHEMA,
            "episode_id": uuid.uuid4().hex,
            "decision_sha256": issued["sha256"],
            "provenance": self._provenance,
            "bucket": bucket,
            "arm": arm,
            "verified_score": round(score, 6),
            "success": success,
            "checked": True,
            "wall_clock_s": round(elapsed, 3),
            "at": time.time(),
        }
        # Durable BEFORE it can influence selection (already the order here);
        # _append reports failure so the caller gets a receipt.
        if not self._append(row):
            return False
        with self._lock:
            self._fold(row)
            self._rows_on_disk += 1
            self._maybe_compact()
        return True

    def _maybe_compact(self) -> None:
        """Enforce the declared ledger bound (CP126 be8c21e8).

        ``_MAX_LEDGER_ROWS`` was declared and never used: restore read the
        whole file and append never rotated, so the ledger grew without limit
        and startup cost was workload-controlled. Compaction rewrites the file
        from the folded state, atomically, and only replaces the original once
        the replacement is complete.
        """
        # Caller holds the lock.
        if self._rows_on_disk <= _MAX_LEDGER_ROWS:
            return
        try:
            from core.governance_context import local_internal_governed_scope
            from core.runtime.file_write_gateway import get_file_write_gateway

            rows: list[str] = []
            for (bucket, arm), cell in sorted(self._cells.items()):
                n = int(cell["n"])
                if n <= 0:
                    continue
                # One COMPACTED row per cell carries the folded evidence
                # without replaying every episode.
                rows.append(
                    json.dumps(
                        {
                            "schema": CONTROLLER_ROW_SCHEMA,
                            "episode_id": hashlib.sha256(
                                f"compact:{self._provenance}:{bucket}:{arm}:{n}".encode()
                            ).hexdigest()[:32],
                            "provenance": self._provenance,
                            "bucket": bucket,
                            "arm": arm,
                            "verified_score": round(
                                float(cell["verified_sum"]) / n, 6
                            ),
                            "success": bool(int(cell["successes"]) * 2 >= n),
                            "checked": True,
                            "wall_clock_s": round(
                                float(cell.get("latency_sum", 0.0)) / n, 3
                            ),
                            "compacted_trials": n,
                            "at": time.time(),
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
            with local_internal_governed_scope(
                "latent_execution_controller", domain="state_mutation"
            ):
                get_file_write_gateway().write_text(
                    self.ledger_path,
                    "".join(rows),
                    source="latent_execution_controller_compaction",
                )
            self._rows_on_disk = len(rows)
            logger.info(
                "Controller ledger compacted to %d cell row(s).", len(rows)
            )
        except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
            logger.warning("Controller ledger compaction skipped: %s", exc)

    def action_evidence_snapshot(self, *, bucket: str) -> dict[str, Any]:
        """Freeze measured per-action evidence for one worker episode."""

        if (
            self._certified_action_certificate is not None
            and self._certified_action_policy is not None
            and self._certified_action_certificate["candidate"][
                "calibration_bucket"
            ]
            == bucket
        ):
            from core.brain.llm.latent_cortex.action_calibration import (
                certified_evidence_snapshot,
            )

            return certified_evidence_snapshot(
                self._certified_action_certificate,
                policy=self._certified_action_policy,
                bucket=bucket,
            )
        return build_evidence_snapshot(
            bucket=bucket,
            cells={
                action: cell
                for (cell_bucket, action), cell in self._action_cells.items()
                if cell_bucket == bucket
            },
        )

    def record_action_transitions(self, rows: list[dict[str, Any]]) -> bool:
        """Persist a checked worker trace atomically before it can train policy."""

        if not isinstance(rows, list) or len(rows) > 256:
            return False
        normalized: list[dict[str, Any]] = []
        try:
            for row in rows:
                normalized.append(validate_action_transition(row))
        except (TypeError, ValueError) as exc:
            # These rows are the action trace a receipt is checked against.
            # Refusing is right; refusing in silence leaves a rejected
            # trace looking the same as one nobody sent.
            logger.warning("Action transitions were refused: %s", exc)
            return False
        if not self._append_action_transitions(normalized):
            return False
        for row in normalized:
            self._fold_action_transition(row)
        return True

    def status(self) -> dict[str, Any]:
        return {
            "schema": EXECUTION_CONTROLLER_SCHEMA,
            "integrity_ok": self._restore_errors == 0,
            "enabled": controller_enabled(),
            "provenance": self._provenance,
            "rows_on_disk": self._rows_on_disk,
            "max_ledger_rows": _MAX_LEDGER_ROWS,
            "open_decisions": len(self._open_decisions),
            "decisions_made": self._decisions_made,
            "buckets": len({bucket for bucket, _ in self._cells}),
            "cells": [
                {
                    "bucket": bucket,
                    "arm": arm,
                    "n": int(cell["n"]),
                    "mean_verified": round(
                        cell["verified_sum"] / max(1, int(cell["n"])), 4
                    ),
                    "success_rate": round(
                        int(cell["successes"]) / max(1, int(cell["n"])), 4
                    ),
                }
                for (bucket, arm), cell in sorted(self._cells.items())
            ][:200],
            "episodes_seen": self._episodes_seen,
            "action_cells": [
                {
                    "bucket": bucket,
                    "action": action.value,
                    **cell.estimate(),
                }
                for (bucket, action), cell in sorted(
                    self._action_cells.items(),
                    key=lambda item: (item[0][0], item[0][1].value),
                )
            ][:400],
            "certified_action_evidence": {
                "admitted": self._certified_action_certificate is not None,
                "certificate_sha256": (
                    self._certified_action_certificate[
                        "certificate_sha256"
                    ]
                    if self._certified_action_certificate is not None
                    else None
                ),
                "calibration_bucket": (
                    self._certified_action_certificate["candidate"][
                        "calibration_bucket"
                    ]
                    if self._certified_action_certificate is not None
                    else None
                ),
                "load_error": self._certified_action_load_error,
            },
            "restore_errors": self._restore_errors,
        }


def controller_enabled() -> bool:
    """Kill switch: AURA_EXECUTION_CONTROLLER=0 disables learn + apply."""
    from core.runtime.flags import FlagKind, declare

    return bool(
        declare(
            "AURA_EXECUTION_CONTROLLER",
            kind=FlagKind.BOOL,
            default=True,
            description="Learned per-problem execution controller (learn + apply)",
            owner="core.brain.llm.latent_cortex.execution_controller",
        ).value()
    )


_instance: ExecutionController | None = None
_instance_lock = threading.Lock()


def get_execution_controller() -> ExecutionController:
    """Process-local singleton.

    CP126 ce5b26e0: check-then-create raced, so concurrent first access could
    instantiate multiple restorers over the same ledger.
    """
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = ExecutionController()
    return _instance


__all__ = [
    "ARMS",
    "EXECUTION_CONTROLLER_SCHEMA",
    "ExecutionController",
    "context_bucket",
    "controller_enabled",
    "get_execution_controller",
]
