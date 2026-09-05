"""CRSM→LoRA loop closure monitor — make the training loop *verifiable*.

The CRSM-LoRA bridge captures high-salience moments and accumulates them into a
JSONL dataset (``data/synthetic_training/lora_dataset.jsonl``). What it could not
answer — the critique's gap — is whether that dataset is actually *consumed* by LoRA
training and whether the resulting weights *persist* into the next session. The
architecture supports it; whether it is running was invisible and "required active
reading."

This monitor closes that observability gap. It compares three timestamps —
captured-dataset growth, the newest fused-model artifact, and the active-model
pointer — to classify the loop as CLOSED, OPEN (captures accumulating but not trained
in), or IDLE, and surfaces a governance signal + a loud warning when it is open. It
also exposes ``mark_dataset_consumed`` so a training run records exactly how much of
the dataset it ingested and which model it produced — turning "is it running?" into a
verified, queryable fact.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

from core.runtime.errors import record_degradation
from core.runtime.file_write_gateway import get_file_write_gateway

logger = logging.getLogger("Aura.CRSMLoop")

_REPO_ROOT = Path(__file__).resolve().parents[2]
_UNCONSUMED_WARN = 25          # captures accumulated past this without training → warn
_STALE_AFTER_S = 7 * 24 * 3600  # dataset newer than the model by this long → stale


class CRSMLoopMonitor:
    def __init__(
        self,
        *,
        dataset_path: Path | None = None,
        fused_model_dir: Path | None = None,
        marker_path: Path | None = None,
        integration_manifest_path: Path | None = None,
        training_state_path: Path | None = None,
        training_data_dir: Path | None = None,
    ) -> None:
        self.dataset_path = dataset_path or (_REPO_ROOT / "data" / "synthetic_training" / "lora_dataset.jsonl")
        self.fused_model_dir = fused_model_dir or (_REPO_ROOT / "training" / "fused-model")
        self.marker_path = marker_path or (self.dataset_path.parent / ".crsm_consumed.json")
        self.training_data_dir = training_data_dir or (_REPO_ROOT / "training" / "data")
        self.integration_manifest_path = integration_manifest_path or (
            self.training_data_dir / "crsm_integration_manifest.json"
        )
        self.training_state_path = training_state_path or (
            _REPO_ROOT / "training" / "adapters" / "aura-personality" / "training_state.json"
        )
        # (size, mtime)-keyed digest cache. Training JSONLs are megabytes and
        # change only when a pipeline stage writes them, but status/health
        # surfaces poll this monitor constantly — an uncached full-file
        # sha256 per poll was a fingerprinted event-loop stall class
        # (api_health → crsm get_status, 5-10s under load).
        self._digest_cache: dict[str, tuple[tuple[int, float], tuple[int, str]]] = {}
        # Eligibility is expensive (parse + safety-gate every capture) but
        # changes only when the dataset does — cache it by the dataset sha256.
        self._eligible_cache: tuple[str, int] | None = None

    def _digest_file(self, path: Path) -> tuple[int, str, int, float]:
        """Return (lines, sha256, size, mtime), re-reading the file only when
        its (size, mtime) signature changed since the last computation."""
        stat = path.stat()
        signature = (int(stat.st_size), float(stat.st_mtime))
        cached = self._digest_cache.get(str(path))
        if cached is not None and cached[0] == signature:
            lines, digest_hex = cached[1]
            return lines, digest_hex, signature[0], signature[1]
        digest = hashlib.sha256()
        lines = 0
        with path.open("rb") as fh:
            for raw in fh:
                lines += 1
                digest.update(raw)
        digest_hex = digest.hexdigest()
        self._digest_cache[str(path)] = (signature, (lines, digest_hex))
        return lines, digest_hex, signature[0], signature[1]

    # ── pipeline observations ─────────────────────────────────────────────

    def dataset_state(self) -> dict[str, Any]:
        try:
            if not self.dataset_path.exists():
                return {"exists": False, "lines": 0, "mtime": 0.0, "size": 0, "sha256": ""}
            lines, digest_hex, size, mtime = self._digest_file(self.dataset_path)
            return {
                "exists": True,
                "lines": lines,
                "mtime": mtime,
                "size": size,
                "sha256": digest_hex,
            }
        except OSError as exc:
            record_degradation("crsm_loop_monitor", exc)
            return {"exists": False, "lines": 0, "mtime": 0.0, "size": 0, "sha256": ""}

    def eligible_capture_count(self) -> int:
        """How many captures would actually TRAIN, through the trainer's own gate.

        The loop's raw line count includes internal-control captures (idle
        self-reflection with <thought>/<action> tags, will-approved receipts)
        that the training safety filter always rejects. Counting raw lines as
        "untrained captures" made the health poll cry 'CRSM→LoRA loop OPEN (N
        captures untrained)' forever for a corpus with nothing trainable in
        it. This routes through build_crsm_experience_examples — the SAME gate
        the trainer uses — so the monitor and the trainer can never disagree.
        Cached by the dataset sha256: it recomputes only when captures change.
        """
        ds = self.dataset_state()
        sha = str(ds.get("sha256") or "")
        if not ds.get("exists") or int(ds.get("lines", 0)) == 0:
            return 0
        if self._eligible_cache is not None and self._eligible_cache[0] == sha:
            return self._eligible_cache[1]
        try:
            from training.build_dataset_v3 import build_crsm_experience_examples

            examples, _manifest = build_crsm_experience_examples(
                self.dataset_path, max_examples=5000
            )
            count = len(examples)
        except (ImportError, OSError, ValueError, RuntimeError, TypeError) as exc:
            record_degradation("crsm_loop_monitor", exc)
            # Unknown eligibility must not manufacture a false OPEN: assume the
            # optimistic case (there may be trainable captures) only when we
            # genuinely cannot tell, and let the raw path decide.
            return int(ds.get("lines", 0))
        self._eligible_cache = (sha, count)
        return count

    def latest_training_artifact(self) -> dict[str, Any]:
        """Newest fused-model directory + the active-model pointer's fuse time."""
        newest_mtime = 0.0
        newest_name = None
        try:
            if self.fused_model_dir.exists():
                for child in self.fused_model_dir.iterdir():
                    if child.is_dir():
                        m = child.stat().st_mtime
                        if m > newest_mtime:
                            newest_mtime, newest_name = m, child.name
        except OSError as exc:
            record_degradation("crsm_loop_monitor", exc)
        active_fused_at = 0.0
        active_path = None
        active_governance: dict[str, Any] = {}
        try:
            active_json = self.fused_model_dir / "active.json"
            if active_json.exists():
                data = json.loads(active_json.read_text(encoding="utf-8"))
                active_fused_at = float(data.get("fused_at", 0.0) or 0.0)
                active_path = data.get("active_model_path")
                active_governance = dict(data.get("governance") or {})
        except (OSError, ValueError, TypeError) as exc:
            record_degradation("crsm_loop_monitor", exc)
        return {
            "newest_model": newest_name,
            "newest_mtime": newest_mtime,
            "active_fused_at": active_fused_at,
            "active_model_path": active_path,
            "active_governance": active_governance,
        }

    def _consumed_marker(self) -> dict[str, Any]:
        try:
            if self.marker_path.exists():
                return json.loads(self.marker_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError) as exc:
            record_degradation("crsm_loop_monitor", exc)
        return {}

    def _jsonl_file_state(self, path: Path, expected: dict[str, Any] | None = None) -> dict[str, Any]:
        expected = expected or {}
        try:
            if not path.exists():
                return {"exists": False, "path": str(path), "matches_expected": False}
            lines, digest_hex, size, mtime = self._digest_file(path)
            actual = {
                "exists": True,
                "path": str(path),
                "lines": lines,
                "size": size,
                "mtime": mtime,
                "sha256": digest_hex,
            }
            expected_hash = str(expected.get("sha256") or "")
            expected_lines = int(expected.get("lines", -1) or -1)
            expected_size = int(expected.get("size", -1) or -1)
            actual["matches_expected"] = bool(
                expected_hash
                and actual["sha256"] == expected_hash
                and expected_lines == lines
                and expected_size == size
            )
            return actual
        except (OSError, ValueError, TypeError) as exc:
            record_degradation("crsm_loop_monitor", exc)
            return {"exists": False, "path": str(path), "matches_expected": False, "error": f"{type(exc).__name__}: {exc}"}

    def integration_manifest_state(self) -> dict[str, Any]:
        try:
            if not self.integration_manifest_path.exists():
                return {"exists": False, "current_for_dataset": False}
            manifest = json.loads(self.integration_manifest_path.read_text(encoding="utf-8"))
            ds = self.dataset_state()
            source_lines = int(manifest.get("source_lines", 0) or 0)
            source_size = int(manifest.get("source_size", -1) or -1)
            source_sha256 = str(manifest.get("source_sha256") or "")
            source_mtime = float(manifest.get("source_mtime", 0.0) or 0.0)
            dataset_mtime = float(ds.get("mtime", 0.0) or 0.0)
            output = dict(manifest.get("output") or {})
            train_expected = dict(output.get("train") or {})
            valid_expected = dict(output.get("valid") or {})
            train_path = Path(str(train_expected.get("path") or (self.training_data_dir / "train.jsonl")))
            valid_path = Path(str(valid_expected.get("path") or (self.training_data_dir / "valid.jsonl")))
            train_state = self._jsonl_file_state(train_path, train_expected)
            valid_state = self._jsonl_file_state(valid_path, valid_expected)
            expected_total = int(output.get("total_examples", 0) or 0)
            actual_total = int(train_state.get("lines", 0) or 0) + int(valid_state.get("lines", 0) or 0)
            if source_sha256:
                source_current = (
                    source_lines == int(ds.get("lines", 0) or 0)
                    and source_size == int(ds.get("size", 0) or 0)
                    and source_sha256 == str(ds.get("sha256") or "")
                )
            else:
                # Backward compatibility for manifests written before the
                # source hash was recorded. New manifests use content identity
                # because safe rewrites can change mtime without changing the
                # capture corpus.
                source_current = (
                    source_lines == int(ds.get("lines", 0) or 0)
                    and source_mtime + 1.0 >= dataset_mtime
                )
            corpus_current = bool(
                output
                and train_state.get("matches_expected")
                and valid_state.get("matches_expected")
                and expected_total == actual_total
            )
            return {
                "exists": True,
                "path": str(self.integration_manifest_path),
                "source_lines": source_lines,
                "source_size": source_size if source_size >= 0 else int(manifest.get("source_size", 0) or 0),
                "source_sha256": source_sha256,
                "accepted": int(manifest.get("accepted", 0) or 0),
                "deduplicated": int(manifest.get("deduplicated", 0) or 0),
                "rejected_by_reason": dict(manifest.get("rejected_by_reason") or {}),
                "source_mtime": source_mtime,
                "output_integrity": {
                    "expected_total": expected_total,
                    "actual_total": actual_total,
                    "train": train_state,
                    "valid": valid_state,
                    "corpus_current": corpus_current,
                },
                "current_for_dataset": source_current and corpus_current,
            }
        except (OSError, ValueError, TypeError) as exc:
            record_degradation("crsm_loop_monitor", exc)
            return {"exists": False, "current_for_dataset": False, "error": f"{type(exc).__name__}: {exc}"}

    def training_state(self) -> dict[str, Any]:
        try:
            if not self.training_state_path.exists():
                return {"exists": False}
            state = json.loads(self.training_state_path.read_text(encoding="utf-8"))
            crsm_delta = dict(state.get("crsm_delta") or {})
            return {
                "exists": True,
                "path": str(self.training_state_path),
                "phase": state.get("phase"),
                "last_iter": int(state.get("last_iter", 0) or 0),
                "last_checkpoint_path": state.get("last_checkpoint_path"),
                "last_pipeline_rc": state.get("last_pipeline_rc"),
                "last_resume_rc": state.get("last_resume_rc"),
                "last_signal": state.get("last_signal"),
                "last_heartbeat": state.get("last_heartbeat"),
                "crsm_delta": crsm_delta,
            }
        except (OSError, ValueError, TypeError) as exc:
            record_degradation("crsm_loop_monitor", exc)
            return {"exists": False, "error": f"{type(exc).__name__}: {exc}"}

    # ── loop closure ──────────────────────────────────────────────────────

    def mark_dataset_consumed(
        self,
        *,
        model_path: str | None = None,
        lines_consumed: int | None = None,
        accepted_lines: int | None = None,
        rejected_lines: int | None = None,
        manifest_path: str | None = None,
        source: str | None = None,
        governance_receipt_id: str | None = None,
        authority_intent_id: str | None = None,
    ) -> bool:
        """Record that a training run ingested the dataset — call after LoRA training.

        Writes how many dataset lines were consumed and which model resulted, so loop
        closure is a verified fact rather than an inference.
        """
        dataset = self.dataset_state()
        if lines_consumed is None:
            lines_consumed = int(dataset.get("lines", 0))
        accepted = int(accepted_lines if accepted_lines is not None else lines_consumed)
        rejected = int(rejected_lines if rejected_lines is not None else max(0, lines_consumed - accepted))
        payload = {
            "lines_consumed": int(lines_consumed),
            "accepted_lines": max(0, accepted),
            "rejected_lines": max(0, rejected),
            "dataset_size": int(dataset.get("size", 0) or 0),
            "dataset_mtime": float(dataset.get("mtime", 0.0) or 0.0),
            "dataset_sha256": str(dataset.get("sha256") or ""),
            "consumed_at": time.time(),
            "model_path": model_path,
            "manifest_path": manifest_path,
            "source": source or "unspecified",
            "governance_receipt_id": governance_receipt_id,
            "authority_intent_id": authority_intent_id,
        }
        try:
            get_file_write_gateway().write_text(
                self.marker_path,
                json.dumps(payload),
                encoding="utf-8",
                source="training:crsm_loop_monitor",
            )
            logger.info("🔁 [CRSMLoop] dataset consumed: %d lines → %s", lines_consumed, model_path)
            return True
        except OSError as exc:
            record_degradation("crsm_loop_monitor", exc)
            return False

    def loop_state(self) -> dict[str, Any]:
        ds = self.dataset_state()
        art = self.latest_training_artifact()
        marker = self._consumed_marker()
        manifest = self.integration_manifest_state()
        training_state = self.training_state()
        lines = int(ds.get("lines", 0))
        consumed = int(marker.get("lines_consumed", 0))
        unconsumed = max(0, lines - consumed)
        marker_hash = str(marker.get("dataset_sha256") or "")
        marker_size = int(marker.get("dataset_size", -1) or -1)
        marker_matches_dataset = bool(
            marker_hash
            and marker_hash == str(ds.get("sha256") or "")
            and marker_size == int(ds.get("size", 0) or 0)
            and consumed == lines
        )
        last_train = max(float(art.get("newest_mtime", 0.0)), float(art.get("active_fused_at", 0.0)))
        ds_mtime = float(ds.get("mtime", 0.0))

        accepted = int(marker.get("accepted_lines", consumed) or 0)
        rejected = int(marker.get("rejected_lines", max(0, consumed - accepted)) or 0)
        marker_model_path = str(marker.get("model_path") or "")
        active_model_path = str(art.get("active_model_path") or "")
        marker_model_matches_active = bool(
            marker_model_path
            and active_model_path
            and Path(marker_model_path).expanduser().resolve()
            == Path(active_model_path).expanduser().resolve()
        )
        marker_counts_reconcile = bool(
            accepted >= 0
            and rejected >= 0
            and accepted + rejected == consumed
        )
        marker_consumed_at = float(marker.get("consumed_at", 0.0) or 0.0)
        verified_consumption = bool(
            marker_matches_dataset
            and marker_model_matches_active
            and marker_counts_reconcile
            and marker_consumed_at > 0.0
            and last_train > 0.0
        )

        # Trained in, and not yet the model she is running.
        #
        # A cycle ends by writing a fused candidate and marking the captures
        # consumed against it. It deliberately does not move the active
        # pointer — activation is a separate, staged act. So the marker names
        # the candidate and the pointer names the incumbent, and they do not
        # match by design.
        #
        # Closure was defined as those two matching, which nothing in the
        # autonomous pipeline can bring about: it prepares a dataset and
        # trains, and has no evaluate or activate phase at all. The loop could
        # therefore never report itself closed however well it ran, and the
        # test that covered it faked a monitor whose active model changed when
        # training returned zero, so the mismatch never showed.
        #
        # The closure test is left exactly as strict as it was — saying closed
        # when nothing was activated would be a claim about the running model
        # that is not true. What was missing is the state that actually
        # happens, which is this one.
        trained_not_active = bool(
            marker_matches_dataset
            and marker_counts_reconcile
            and marker_consumed_at > 0.0
            and last_train > 0.0
            and marker_model_path
            and not marker_model_matches_active
        )

        if lines == 0:
            state, reason = "idle", "no captured moments yet"
        elif verified_consumption:
            if rejected > 0:
                state, reason = (
                    "closed",
                    f"{accepted} eligible captures trained and {rejected} retired by the training gate",
                )
            else:
                state, reason = "closed", "dataset trained in and weights persisted"
        elif trained_not_active:
            state, reason = (
                "qualified",
                "dataset trained into a candidate that is not the active model; "
                "closing it needs an activation this pipeline does not perform",
            )
        elif unconsumed == 0 and consumed >= lines:
            state, reason = (
                "pending",
                "capture corpus changed after the last consumed marker; current corpus needs train/fuse confirmation",
            )
        elif last_train >= ds_mtime and unconsumed <= _UNCONSUMED_WARN:
            state, reason = (
                "pending",
                "a newer model exists, but current captures lack a verified active-model consumption marker",
            )
        elif unconsumed > _UNCONSUMED_WARN or (ds_mtime - last_train) > _STALE_AFTER_S:
            # Only genuinely trainable captures can hold the loop OPEN. A
            # corpus of pure internal-control captures (idle self-reflection)
            # is nothing to crystallize — reporting it as OPEN / 'proof
            # integrity degraded' is a false alarm, and the closer would fail
            # the dataset build anyway ("no eligible captures").
            eligible = self.eligible_capture_count()
            if eligible <= 0:
                state = "idle"
                reason = (
                    f"{unconsumed} captures accumulated but 0 are eligible for "
                    "training (internal-control/ineligible); nothing to crystallize"
                )
            else:
                state = "open"
                if manifest.get("current_for_dataset"):
                    reason = (
                        f"{eligible} of {unconsumed} captures are trainable and in "
                        "the LoRA corpus but not yet trained/fused into the active model"
                    )
                else:
                    reason = f"{eligible} of {unconsumed} accumulated captures are trainable but not trained in"
        else:
            state, reason = "pending", "captures awaiting the next training run"

        return {
            "state": state,
            "reason": reason,
            "dataset_lines": lines,
            "unconsumed": unconsumed,
            "eligible_captures": self.eligible_capture_count(),
            "accepted_lines": accepted,
            "rejected_lines": rejected,
            "last_training_at": last_train,
            "active_model": art.get("active_model_path"),
            "active_model_governance": dict(art.get("active_governance") or {}),
            "dataset_mtime": ds_mtime,
            "marker_matches_dataset": marker_matches_dataset,
            "marker_model_matches_active": marker_model_matches_active,
            "marker_counts_reconcile": marker_counts_reconcile,
            "verified_consumption": verified_consumption,
            "integration_manifest": manifest,
            "training_state": training_state,
            "next_action": self.next_action(state, manifest, training_state),
            "consumption_marker": marker,
        }

    def next_action(
        self,
        state: str | None = None,
        manifest: dict[str, Any] | None = None,
        training_state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        state = state or self.loop_state().get("state")
        manifest = manifest if manifest is not None else self.integration_manifest_state()
        training_state = training_state if training_state is not None else self.training_state()
        command = [
            "python",
            "training/train_and_fuse.py",
            "--crsm-delta",
            "--tag",
            "crsm-closeout",
        ]
        preflight_command = [
            "python",
            "training/train_and_fuse.py",
            "--crsm-delta",
            "--preflight-only",
            "--tag",
            "crsm-closeout",
        ]
        if state in {"closed", "qualified", "idle"}:
            return {
                "required": False,
                "reason": (
                    "CRSM captures already consumed by active training marker"
                    if state == "closed"
                    else "CRSM captures trained into a candidate awaiting activation"
                    if state == "qualified"
                    else "No eligible CRSM captures require training"
                ),
            }
        if not manifest.get("current_for_dataset"):
            return {
                "required": True,
                "phase": "prepare_dataset",
                "command": ["python", "training/build_dataset_v3.py"],
                "reason": "CRSM integration manifest is missing or stale",
            }
        return {
            "required": True,
            "phase": "crsm_delta_train_fuse_publish",
            "command": command,
            "preflight_command": preflight_command,
            "reason": (
                "Current CRSM captures are in the LoRA corpus, but proof closure "
                "requires a bounded real CRSM delta train/fuse marker from "
                "training/train_and_fuse.py"
            ),
            "last_training_phase": training_state.get("phase"),
            "last_training_rc": training_state.get("last_pipeline_rc"),
        }

    def audit(self) -> dict[str, Any]:
        """Evaluate the loop and log loudly if it is open (the previously-silent gap)."""
        state = self.loop_state()
        if state["state"] == "open":
            logger.warning(
                "🔁 [CRSMLoop] LOOP OPEN: %s — captured experience is not being "
                "crystallized into weights. Run LoRA training on the synthetic dataset.",
                state["reason"],
            )
            try:
                from core.observability.metrics import get_metrics

                get_metrics().increment_counter("crsm_loop_open_total")
            except (ImportError, AttributeError, RuntimeError, TypeError):
                pass
        return state

    def governance_signal(self) -> dict[str, Any]:
        return self.loop_state()


_monitor: CRSMLoopMonitor | None = None


def get_crsm_loop_monitor() -> CRSMLoopMonitor:
    global _monitor
    if _monitor is None:
        _monitor = CRSMLoopMonitor()
    return _monitor
