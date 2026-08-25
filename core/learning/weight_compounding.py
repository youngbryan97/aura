"""core/learning/weight_compounding.py — the canonical compounding weight-learning cycle.

Aura has had every *piece* of weight-level learning for months — LoRA training
(live_learner), sound DPO data (verifiable_preference_harness), a real DPO
trainer (mlx-lm-lora), fuse/candidate recording, and a
tamper-evident lineage ledger (rsi_lineage). What it never had is the spine
that makes the pieces eligible to compound: each cycle trains on the current
active artifact, is gated by a sealed held-out capability battery scored
one-model-at-a-time in subprocesses, and appends an honest generation record.
A candidate compounds only after the independent cortex-upgrade path activates
it; candidate production itself never changes the next cycle's base.

The cycle:
  1. resolve_base   — fresh read of the active-model manifest (never a stale
                      import-time path). Only an independently activated
                      generation can become the next cycle's base.
  2. admission      — single-flight lockfile, RAM headroom vs model footprint,
                      autonomous size cap, optional foreground-idle hook. The
                      loop refuses to fight the resident cortex for memory.
  3. harvest        — DPO pairs from the verifier harness when there are
                      enough (the strongest signal), else SFT rows from the
                      live-learner buffer; contamination-filtered and with the
                      sealed battery prompts excluded so the eval can never
                      leak into training.
  4. train          — mlx subprocess (mlx_lm_lora.train for DPO, mlx_lm lora
                      for SFT) against the RESOLVED base, bounded timeout.
  5. evaluate       — tools/heldout_eval.py run sequentially (one extra model
                      in memory at a time) for incumbent and candidate on the
                      visible battery AND on a second hidden battery seed,
                      plus an identity-regression scan over the candidate's
                      raw responses. Writes the evaluation_report.json
                      contract that core/learning/eval_before_promotion.py
                      reads.
  6. decide/record  — accept only when all three gates hold; fuse and record
                      an exact qualification candidate without changing the
                      serving pointer. Refusals are first-class results.
  7. record         — append an RSIGenerationRecord to the tamper-evident
                      ledger. Compounding is then a *verdict computed from
                      receipts* (rsi_lineage.evaluate_lineage), never a claim.

The three gates, and why each is shaped the way it is:

**Capability, against the high-water mark.** ``candidate ≥ high_water −
epsilon``, where high_water is the best visible-battery accuracy any PROMOTED
generation has reached. It used to be ``candidate ≥ incumbent − epsilon``,
which let every generation give back epsilon and still record "no
regression" — twenty of those walk accuracy down by twenty epsilons with
twenty clean receipts behind them. Epsilon is tolerance for battery noise
around a level already reached, not a per-generation allowance.

**Hidden battery, against the incumbent.** The hidden seed now runs on BOTH
models. It used to run on the candidate only, so the strongest statement
available was ``hidden is not None`` — the second eval ran every cycle,
cost a full model load, and could not refuse anything. It exists to catch a
candidate that gained on the visible battery by learning the visible
battery, and that is a comparison or it is nothing.

**Identity.** Unchanged: assistant-regression phrases in the candidate's raw
responses.

Honesty boundary: this module can establish only a locally accepted adapter
or an exact fused candidate. It cannot promote a cortex. Promotion additionally
requires the central comparison, migration, serving, staging, authorization,
and activation contracts. The compounding claim is only ever computed from
activated lineage receipts; this producer cannot assert it, only submit work
that may later earn it.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.learning.cortex_generation_upgrade import record_upgrade_candidate
from core.learning.heldout_battery import (
    BatterySpec,
    generate_battery,
    text_collides_with_battery,
)
from core.learning.rsi_lineage import (
    PROVENANCE_MEASURED,
    PROVENANCE_UNMEASURED,
    RSIGenerationRecord,
    RSILineageLedger,
    RSILineageVerdict,
    evaluate_lineage,
    improver_efficiency,
)
from core.runtime.atomic_writer import atomic_write_text
from core.runtime.disk_budget import DiskBudgetRefusal, ensure_headroom_for
from core.runtime.errors import record_degradation
from core.runtime.file_write_gateway import get_file_write_gateway

logger = logging.getLogger("Aura.WeightCompounding")

_RECOVERABLE = (OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError)

# Assistant-regression phrases scanned over the candidate's raw eval responses.
# Battery prompts are task-shaped, so ANY of these appearing is a bad sign.
_IDENTITY_REGRESSION_PHRASES = (
    "as an ai language model",
    "as a language model",
    "i cannot assist with that",
    "how can i assist you",
    "i'm just an ai",
)

DEFAULT_EPSILON = 0.02          # max tolerated held-out accuracy drop
HIDDEN_SEED_OFFSET = 100003     # hidden battery = primary seed + this offset


# ── configuration ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CompoundingConfig:
    """One cycle's knobs. Everything bounded, everything recorded."""

    work_root: Path                       # runs + ledger live under here
    fused_root: Path                      # active pointer is read; candidates are recorded
    model_override: str = ""              # explicit base (proof runs); else manifest
    default_base: str = ""                # fallback when no manifest exists yet

    # data
    sft_buffer_path: Path | None = None   # live-learner experience buffer
    dpo_store_path: Path | None = None    # verifiable preference store
    min_dpo_pairs: int = 24
    min_sft_examples: int = 30
    max_examples: int = 240

    # training
    iters: int = 80
    batch_size: int = 1
    learning_rate: float = 5e-6
    num_layers: int = 16
    max_seq_length: int = 1024
    train_timeout_s: int = 3600

    # evaluation
    battery_seed_base: int = 1000
    battery_size: int = 40
    hidden_battery_size: int = 20
    eval_max_tokens: int = 256
    eval_timeout_s: int = 1800
    epsilon: float = DEFAULT_EPSILON

    # admission
    ram_headroom_factor: float = 1.5      # need model_bytes*factor free
    ram_slack_bytes: int = 2 * 1024**3
    autonomous_max_model_bytes: int = 6 * 1024**3   # ~7B-4bit; larger needs operator
    operator_run: bool = False            # CLI/operator runs may exceed the cap

    # candidate materialization
    publish: bool = True
    fuse_timeout_s: int = 3600
    keep_fused: int = 3                   # prune loop-created fused artifacts beyond this
    # Fuse dequantizes the base to fp16 and merges — transient peak is well
    # above the on-disk size, and it runs BESIDE the resident serving model.
    # Live Jul 8: a gated 32B candidate OOM-killed its fuse (empty stderr,
    # SIGKILL) beside the live 32B. Admit the fuse against real free RAM;
    # defer (keep the adapter) instead of crashing when it won't fit.
    fuse_peak_factor: float = 2.5
    fuse_min_slack_bytes: int = 3 * 1024**3

    @property
    def ledger_path(self) -> Path:
        return self.work_root / "lineage.jsonl"

    @property
    def manifest_path(self) -> Path:
        return self.fused_root / "active.json"


# ── receipts ──────────────────────────────────────────────────────────────────

@dataclass
class CycleReceipt:
    generation_id: str
    status: str                             # candidate | qualified_adapter | refused | blocked | failed
    reasons: list[str] = field(default_factory=list)
    base_model: str = ""
    base_source: str = ""                   # manifest | override | default
    train_mode: str = ""                    # dpo | sft
    data_counts: dict[str, int] = field(default_factory=dict)
    incumbent_accuracy: float | None = None
    candidate_accuracy: float | None = None
    hidden_accuracy: float | None = None
    #: The same hidden battery on the incumbent. Without it the hidden eval
    #: had nothing to be compared against and could not gate anything.
    hidden_incumbent_accuracy: float | None = None
    #: Best visible-battery accuracy any promoted generation has reached.
    #: The promotion floor is measured from here, not from the immediate
    #: parent, so epsilon cannot be spent again every generation.
    high_water_accuracy: float | None = None
    identity_ok: bool | None = None
    candidate_model_path: str = ""
    candidate_receipt_path: str = ""
    # Compatibility field. Only central staged activation may populate this.
    promoted_model_path: str = ""
    run_dir: str = ""
    elapsed_s: float = 0.0
    ledger_entry_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "generation_id": self.generation_id,
            "status": self.status,
            "reasons": list(self.reasons),
            "base_model": self.base_model,
            "base_source": self.base_source,
            "train_mode": self.train_mode,
            "data_counts": dict(self.data_counts),
            "incumbent_accuracy": self.incumbent_accuracy,
            "candidate_accuracy": self.candidate_accuracy,
            "hidden_accuracy": self.hidden_accuracy,
            "hidden_incumbent_accuracy": self.hidden_incumbent_accuracy,
            "high_water_accuracy": self.high_water_accuracy,
            "identity_ok": self.identity_ok,
            "candidate_model_path": self.candidate_model_path,
            "candidate_receipt_path": self.candidate_receipt_path,
            "promoted_model_path": self.promoted_model_path,
            "run_dir": self.run_dir,
            "elapsed_s": round(self.elapsed_s, 2),
            "ledger_entry_hash": self.ledger_entry_hash,
        }


# ── helpers ───────────────────────────────────────────────────────────────────

_HF_REPO_ID_RE = re.compile(r"[\w.\-]+/[\w.\-]+")


def _resolve_model_dir(model_path: str) -> Path | None:
    """Resolve a model reference to a local directory, if one exists.

    Accepts filesystem paths and Hugging Face repo ids ("org/name"). Repo ids
    resolve against the local HF cache WITHOUT any network touch, so admission
    control can size a cached model exactly like a local directory. A model
    that is neither on disk nor cached stays unresolved — admission then
    blocks on the unknown footprint, which is the fail-closed default.
    """
    candidate = Path(model_path).expanduser()
    if candidate.exists():
        return candidate
    if _HF_REPO_ID_RE.fullmatch(model_path or ""):
        try:
            from core.runtime.third_party_imports import import_attribute_serialized

            snapshot_download = import_attribute_serialized(
                "huggingface_hub",
                "snapshot_download",
            )
            return Path(snapshot_download(repo_id=model_path, local_files_only=True))
        except (ImportError, OSError, ValueError):
            return None
    return None


def _dir_weight_bytes(model_dir: Path) -> int:
    """Approximate in-memory footprint by on-disk weight size."""
    try:
        return sum(
            f.stat().st_size
            for f in model_dir.glob("*.safetensors")
        ) or sum(f.stat().st_size for f in model_dir.iterdir() if f.is_file())
    except OSError:
        return 0


def _artifact_digest(path: Path) -> str:
    """Content digest for small artifacts, structural digest for model dirs.

    Fully hashing a 20GB model directory per cycle is not honest engineering,
    it is theater; for large dirs we digest (name, size, mtime) of every file,
    which detects any swap/tamper of the artifact while staying O(files).
    """
    try:
        if path.is_file():
            h = hashlib.sha256()
            with path.open("rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            return "sha256:" + h.hexdigest()
        entries = sorted(
            (str(f.relative_to(path)), f.stat().st_size, int(f.stat().st_mtime))
            for f in path.rglob("*")
            if f.is_file()
        )
        blob = json.dumps(entries, sort_keys=True).encode()
        return "sha256-structural:" + hashlib.sha256(blob).hexdigest()
    except OSError:
        return "unavailable"


def _default_command_runner(command: tuple[str, ...], timeout_s: float):
    from core.tasks.managed_command import run_project_command

    return run_project_command(command, timeout_s=timeout_s)


# ── the loop ──────────────────────────────────────────────────────────────────

class WeightCompoundingLoop:
    """One canonical driver for a train→gate→candidate/refuse→record cycle.

    Injectable seams (all optional) keep this testable offline and let the
    runtime supply live context:
      * command_runner    — subprocess executor (tests inject fakes)
      * approval_hook     — Will/governance consult; returns (approved, reason)
      * idle_hook         — returns True when the foreground lane is quiet
    """

    def __init__(
        self,
        config: CompoundingConfig,
        *,
        command_runner: Callable[[tuple[str, ...], float], Any] | None = None,
        approval_hook: Callable[[dict[str, Any]], tuple[bool, str]] | None = None,
        idle_hook: Callable[[], bool] | None = None,
    ) -> None:
        self.config = config
        self._run_command = command_runner or _default_command_runner
        self._approval_hook = approval_hook
        self._idle_hook = idle_hook
        self.config.work_root.mkdir(parents=True, exist_ok=True)
        self._ledger = RSILineageLedger(self.config.ledger_path)

    # ── 1. base resolution (the compounding hinge) ───────────────────────────

    def resolve_base(self) -> tuple[str, str]:
        """Return (model_path, source). Reads the manifest FRESH every cycle."""
        if self.config.model_override:
            return self.config.model_override, "override"
        manifest = self.config.manifest_path
        try:
            if manifest.exists():
                data = json.loads(manifest.read_text(encoding="utf-8"))
                path = str(data.get("active_model_path") or "").strip()
                if path and Path(path).exists():
                    return path, "manifest"
        except _RECOVERABLE as exc:
            record_degradation(
                "weight_compounding",
                exc,
                action="fell back to default base after manifest read failed",
            )
        if self.config.default_base:
            return self.config.default_base, "default"
        raise RuntimeError(
            "no base model resolvable: manifest missing/invalid and no default_base configured"
        )

    # ── 2. admission control ─────────────────────────────────────────────────

    def admission_check(self, model_path: str, mode: str = "sft") -> tuple[bool, list[str]]:
        reasons: list[str] = []
        model_dir = _resolve_model_dir(model_path)
        weight_bytes = _dir_weight_bytes(model_dir) if model_dir is not None else 0

        if weight_bytes == 0:
            reasons.append(f"model_footprint_unknown:{model_path}")

        if (
            not self.config.operator_run
            and weight_bytes > self.config.autonomous_max_model_bytes
        ):
            reasons.append(
                "model_exceeds_autonomous_cap"
                f":{weight_bytes >> 30}GB>{self.config.autonomous_max_model_bytes >> 30}GB"
            )

        try:
            from core.runtime import resource_psutil as psutil

            available = psutil.virtual_memory().available
            # DPO holds the policy AND a frozen reference model in memory —
            # roughly double the SFT footprint. Verified against mlx_lm_lora,
            # which loads (and only later dels) reference_model.
            mode_multiplier = 2.0 if mode == "dpo" else 1.0
            needed = (
                int(weight_bytes * self.config.ram_headroom_factor * mode_multiplier)
                + self.config.ram_slack_bytes
            )
            if available < needed:
                reasons.append(
                    f"insufficient_ram:available={available >> 30}GB needed={needed >> 30}GB mode={mode}"
                )
        except ImportError as exc:
            record_degradation(
                "weight_compounding",
                exc,
                action="blocked cycle because RAM headroom could not be verified",
            )
            reasons.append("ram_check_unavailable")

        if self._idle_hook is not None:
            try:
                if not self._idle_hook():
                    reasons.append("foreground_not_idle")
            except _RECOVERABLE as exc:
                record_degradation(
                    "weight_compounding",
                    exc,
                    action="blocked cycle because idle check failed",
                )
                reasons.append("idle_check_failed")

        return not reasons, reasons

    def _acquire_lock(self) -> bool:
        """Single-flight lock with stale-pid reclamation."""
        lock = self.config.work_root / "cycle.lock"
        try:
            fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            with os.fdopen(fd, "w") as fh:
                fh.write(json.dumps({"pid": os.getpid(), "at": time.time()}))
            return True
        except FileExistsError:
            try:
                stale = json.loads(lock.read_text(encoding="utf-8"))
                pid = int(stale.get("pid", -1))
                os.kill(pid, 0)  # raises when the holder is gone
                return False
            except (OSError, ValueError, json.JSONDecodeError):
                try:
                    lock.unlink()
                except OSError:
                    return False
                return self._acquire_lock()

    def _release_lock(self) -> None:
        try:
            (self.config.work_root / "cycle.lock").unlink(missing_ok=True)
        except OSError as exc:
            record_degradation(
                "weight_compounding",
                exc,
                action="left stale cycle lock for next-run reclamation",
            )

    # ── 3. harvest ───────────────────────────────────────────────────────────

    def harvest(self, run_dir: Path, battery_tasks) -> tuple[str, Path, dict[str, int]]:
        """Choose DPO when the verifier pairs suffice, else SFT. Returns
        (mode, data_dir, split_counts). Raises RuntimeError when neither
        source has enough clean rows — a cycle without real data must not run.
        """
        data_dir = run_dir / "data"

        dpo_rows = self._load_dpo_rows(battery_tasks)
        if len(dpo_rows) >= self.config.min_dpo_pairs:
            from core.learning.preference_trainer import export_preference_splits

            counts = export_preference_splits(dpo_rows[: self.config.max_examples], data_dir)
            return "dpo", data_dir, counts

        sft_rows = self._load_sft_rows(battery_tasks)
        if len(sft_rows) >= self.config.min_sft_examples:
            counts = self._export_sft_splits(sft_rows[: self.config.max_examples], data_dir)
            return "sft", data_dir, counts

        raise RuntimeError(
            f"insufficient_training_data:dpo={len(dpo_rows)}/{self.config.min_dpo_pairs} "
            f"sft={len(sft_rows)}/{self.config.min_sft_examples}"
        )

    def _load_dpo_rows(self, battery_tasks) -> list[dict[str, str]]:
        store = self.config.dpo_store_path
        if store is None or not Path(store).exists():
            return []
        rows: list[dict[str, str]] = []
        try:
            with Path(store).open(encoding="utf-8") as fh:
                for line in fh:
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    prompt = str(row.get("prompt", "")).strip()
                    chosen = str(row.get("chosen", "")).strip()
                    rejected = str(row.get("rejected", "")).strip()
                    if not prompt or not chosen or chosen == rejected:
                        continue
                    if text_collides_with_battery(
                        f"{prompt}\n{chosen}\n{rejected}", battery_tasks
                    ):
                        continue
                    rows.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
        except OSError as exc:
            record_degradation(
                "weight_compounding",
                exc,
                action="continued harvest without DPO store after read failure",
            )
        return rows

    def _load_sft_rows(self, battery_tasks) -> list[dict[str, Any]]:
        buffer_path = self.config.sft_buffer_path
        if buffer_path is None or not Path(buffer_path).exists():
            return []
        from core.learning.live_learner import LiveLearner

        rows: list[tuple[float, dict[str, Any]]] = []
        try:
            with Path(buffer_path).open(encoding="utf-8") as fh:
                for line in fh:
                    try:
                        raw = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(raw, dict):
                        continue
                    clean = LiveLearner._clean_training_example(raw)
                    if clean is None:
                        continue
                    text_view = json.dumps(clean, ensure_ascii=False)
                    if text_collides_with_battery(text_view, battery_tasks):
                        continue
                    quality = float(raw.get("_quality", 0.0) or 0.0)
                    rows.append((quality, clean))
        except OSError as exc:
            record_degradation(
                "weight_compounding",
                exc,
                action="continued harvest without SFT buffer after read failure",
            )
        rows.sort(key=lambda item: item[0], reverse=True)
        # de-duplicate while preserving quality order
        seen: set[str] = set()
        out: list[dict[str, Any]] = []
        for _, clean in rows:
            fp = hashlib.sha256(json.dumps(clean, sort_keys=True).encode()).hexdigest()
            if fp in seen:
                continue
            seen.add(fp)
            out.append(clean)
        return out

    @staticmethod
    def _export_sft_splits(rows: list[dict[str, Any]], data_dir: Path) -> dict[str, int]:
        data_dir.mkdir(parents=True, exist_ok=True)
        valid_count = max(1, len(rows) // 12) if len(rows) >= 12 else 0
        train_rows = rows[: len(rows) - valid_count] if valid_count else rows
        valid_rows = rows[len(rows) - valid_count:] if valid_count else []
        counts: dict[str, int] = {}
        for split, split_rows in (("train", train_rows), ("valid", valid_rows)):
            if not split_rows and split != "train":
                continue
            atomic_write_text(
                data_dir / f"{split}.jsonl",
                "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in split_rows),
                encoding="utf-8",
            )
            counts[split] = len(split_rows)
        return counts

    # ── 4. train ─────────────────────────────────────────────────────────────

    def train(self, base_model: str, data_dir: Path, adapter_dir: Path, mode: str) -> tuple[bool, str]:
        adapter_dir.mkdir(parents=True, exist_ok=True)
        cfg = self.config
        if mode == "dpo":
            # mlx_lm_lora's CONFIG_DEFAULTS sets fuse=True, which would
            # de-quantize and fuse INSIDE the trainer, before our eval gate
            # (and would write ~65GB for a 32B). A config file is the only
            # way to disable it; fusing stays a gated post-eval step here.
            trainer_config = adapter_dir / "trainer_config.yaml"
            atomic_write_text(trainer_config, "fuse: false\n", encoding="utf-8")
            command = (
                sys.executable, "-m", "mlx_lm_lora.train",
                "--model", base_model,
                "--train",
                "--data", str(data_dir),
                "--train-type", "lora",
                "--train-mode", "dpo",
                "--adapter-path", str(adapter_dir),
                "--num-layers", str(cfg.num_layers),
                "--iters", str(max(1, cfg.iters)),
                "--batch-size", str(max(1, cfg.batch_size)),
                "--learning-rate", str(cfg.learning_rate),
                "--save-every", str(max(1, cfg.iters)),
                "--max-seq-length", str(max(128, cfg.max_seq_length)),
                "--grad-checkpoint",
                "-c", str(trainer_config),
            )
        else:
            command = (
                sys.executable, "-m", "mlx_lm", "lora",
                "--model", base_model,
                "--train",
                "--data", str(data_dir),
                "--fine-tune-type", "lora",
                "--adapter-path", str(adapter_dir),
                "--num-layers", str(cfg.num_layers),
                "--iters", str(max(1, cfg.iters)),
                "--batch-size", str(max(1, cfg.batch_size)),
                "--learning-rate", str(cfg.learning_rate),
                "--save-every", str(max(1, cfg.iters)),
                "--max-seq-length", str(max(128, cfg.max_seq_length)),
                "--grad-checkpoint",
                "--mask-prompt",
            )
        result = self._run_command(command, float(cfg.train_timeout_s))
        ok = bool(getattr(result, "ok", False)) and (adapter_dir / "adapters.safetensors").exists()
        detail = (getattr(result, "stderr", "") or getattr(result, "stdout", "") or "")[-2000:]
        return ok, detail

    # ── 5. evaluate ──────────────────────────────────────────────────────────

    def _run_heldout(
        self,
        model: str,
        adapter: str,
        seed: int,
        size: int,
        output: Path,
    ) -> dict[str, Any] | None:
        command = [
            sys.executable,
            str(Path(__file__).resolve().parent.parent.parent / "tools" / "heldout_eval.py"),
            "--model", model,
            "--seed", str(seed),
            "--size", str(size),
            "--max-tokens", str(self.config.eval_max_tokens),
            "--output", str(output),
        ]
        if adapter:
            command.extend(["--adapter-path", adapter])
        result = self._run_command(tuple(command), float(self.config.eval_timeout_s))
        if not getattr(result, "ok", False) or not output.exists():
            record_degradation(
                "weight_compounding",
                RuntimeError(
                    f"heldout_eval_failed:{(getattr(result, 'stderr', '') or '')[-400:]}"
                ),
                action="treated missing eval report as gate failure",
            )
            return None
        try:
            return json.loads(output.read_text(encoding="utf-8"))
        except _RECOVERABLE as exc:
            record_degradation(
                "weight_compounding",
                exc,
                action="treated unreadable eval report as gate failure",
            )
            return None

    @staticmethod
    def _identity_scan(responses_path: Path) -> tuple[bool, list[str]]:
        """Scan the candidate's raw eval responses for assistant regressions."""
        hits: list[str] = []
        try:
            if responses_path.exists():
                text = responses_path.read_text(encoding="utf-8").lower()
                hits = [p for p in _IDENTITY_REGRESSION_PHRASES if p in text]
        except OSError:
            hits = ["responses_unreadable"]
        return not hits, hits

    # ── 6. publish ───────────────────────────────────────────────────────────

    def _fuse_memory_admits(self, base_model: str) -> tuple[bool, str]:
        """True if a fuse of ``base_model`` can run without OOM right now.

        Operator runs bypass this (the operator chose the moment). Autonomous
        runs check live free RAM against the fuse's transient peak so a gated
        adapter is deferred, not crashed, when the machine is busy serving.
        """
        if self.config.operator_run:
            return True, "operator_run"
        model_dir = _resolve_model_dir(base_model)
        base_bytes = _dir_weight_bytes(model_dir) if model_dir is not None else 0
        if base_bytes == 0:
            return True, "unknown_footprint"  # can't size it; don't block on ignorance
        try:
            from core.runtime import resource_psutil as psutil

            available = psutil.virtual_memory().available
        except (ImportError, AttributeError, OSError):
            return True, "ram_probe_unavailable"
        needed = int(base_bytes * self.config.fuse_peak_factor) + self.config.fuse_min_slack_bytes
        if available < needed:
            return False, f"available={available >> 30}GB<peak≈{needed >> 30}GB"
        return True, f"available={available >> 30}GB>=peak≈{needed >> 30}GB"

    def _fuse_disk_admits(self, base_model: str) -> tuple[bool, str]:
        """True if the fused artifact fits without pushing the volume red.

        The memory admission above asks "is there RAM for this?" and nothing
        ever asked "is there DISK for this?". On 2026-08-13 the answer had been
        no for a while: 19GB free on 1.8TB, the volume pinned at 99%, metabolism
        in permanent lockdown. Sixty worktrees had each written their own 17GB
        copy of a fused model, and no code path anywhere was able to decline.

        Operator runs are NOT exempt, unlike the RAM check. An OOM kills one
        subprocess and the host survives; filling the volume corrupts the
        artifact being written AND every other writer on the machine, with no
        kernel backstop to choose a victim.
        """
        model_dir = _resolve_model_dir(base_model)
        base_bytes = _dir_weight_bytes(model_dir) if model_dir is not None else 0
        if base_bytes == 0:
            return True, "unknown_footprint"  # can't size it; don't block on ignorance
        try:
            ensure_headroom_for(
                base_bytes,
                purpose=f"fuse {base_model} -> {self.config.fused_root.name}",
                path=self.config.fused_root,
            )
        except DiskBudgetRefusal as exc:
            return False, str(exc)
        return True, f"fits≈{base_bytes >> 30}GB"

    def fuse_and_record_candidate(
        self,
        base_model: str,
        adapter_dir: Path,
        generation_id: str,
        metadata: dict[str, Any],
    ) -> tuple[str, str, list[str]]:
        cfg = self.config
        fused_path = cfg.fused_root / f"Aura-compound-{generation_id}"
        get_file_write_gateway().ensure_directory(
            fused_path.parent,
            source="core.learning.weight_compounding.candidate_root",
        )

        # Pre-fuse memory admission. Fusing dequantizes the base to fp16 and
        # merges the adapter — a transient peak far above the on-disk size,
        # running beside the resident serving model. Rather than OOM-kill the
        # subprocess (SIGKILL leaves NO stderr — the Jul 8 "fuse_failed:"
        # mystery), defer: the trained+gated adapter is preserved for a later
        # window (or operator fuse). A deferral is not a failure.
        fits, admit_reason = self._fuse_memory_admits(base_model)
        if not fits:
            return "", "", [f"fuse_deferred_memory:{admit_reason}"]

        # Pre-fuse DISK admission. Same deferral semantics: a gated adapter is
        # preserved for a later window, not thrown away, and not written into a
        # volume that cannot hold it.
        fits_disk, disk_reason = self._fuse_disk_admits(base_model)
        if not fits_disk:
            return "", "", [f"fuse_deferred_disk:{disk_reason}"]

        command = (
            sys.executable, "-m", "mlx_lm", "fuse",
            "--model", base_model,
            "--adapter-path", str(adapter_dir),
            "--save-path", str(fused_path),
        )
        result = self._run_command(command, float(cfg.fuse_timeout_s))
        if not getattr(result, "ok", False) or not fused_path.exists():
            # Never a blind "fuse_failed:". SIGKILL (OOM) shows as a negative
            # returncode with empty stderr; a timeout sets timed_out; capture
            # all three so the receipt is diagnosable.
            rc = getattr(result, "returncode", None)
            timed_out = bool(getattr(result, "timed_out", False))
            stderr_tail = (getattr(result, "stderr", "") or "")[-300:]
            stdout_tail = (getattr(result, "stdout", "") or "")[-150:]
            if timed_out:
                detail = f"timeout_after_{cfg.fuse_timeout_s}s"
            elif isinstance(rc, int) and rc < 0:
                detail = f"killed_signal_{-rc}(likely_oom)"
            elif stderr_tail:
                detail = stderr_tail
            elif not fused_path.exists():
                detail = f"no_output rc={rc} stdout={stdout_tail!r}"
            else:
                detail = f"rc={rc}"
            return "", "", [f"fuse_failed:{detail}"]

        resolved_base = _resolve_model_dir(base_model)
        if resolved_base is None:
            return "", "", ["candidate_record_failed:base_model_not_local"]
        candidate_receipt = record_upgrade_candidate(
            candidate_model_path=fused_path,
            base_model_path=resolved_base,
            tag=f"compound-{generation_id}",
            fused_model_dir=cfg.fused_root,
            source="core.learning.weight_compounding",
            metadata={"generation_id": generation_id, **metadata},
        )
        self._prune_loop_fused(keep=cfg.keep_fused, active=str(fused_path))
        return (
            str(fused_path),
            str(candidate_receipt["candidate_receipt_path"]),
            [],
        )

    def _prune_loop_fused(self, *, keep: int, active: str) -> None:
        """Bound disk growth: prune ONLY artifacts this loop created (never
        operator-built fused models), never the active one, keep the newest N."""
        try:
            candidates = sorted(
                (
                    p for p in self.config.fused_root.glob("Aura-compound-*")
                    if p.is_dir() and str(p) != active
                ),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            for stale in candidates[max(0, keep - 1):]:
                if get_file_write_gateway().delete_path(
                    stale,
                    recursive=True,
                    source="core.learning.weight_compounding.prune_candidate",
                ):
                    logger.info(
                        "WeightCompounding: pruned stale fused artifact %s",
                        stale.name,
                    )
        except OSError as exc:
            record_degradation(
                "weight_compounding",
                exc,
                action="left stale fused artifacts on disk after prune failure",
            )

    # ── 7. the full cycle ────────────────────────────────────────────────────

    def run_cycle(self) -> CycleReceipt:
        started = time.time()
        generation_seq = len(self._ledger.load_records())
        generation_id = f"g{generation_seq:04d}-{int(started)}"
        run_dir = self.config.work_root / "runs" / generation_id
        receipt = CycleReceipt(generation_id=generation_id, status="failed", run_dir=str(run_dir))

        if self._approval_hook is not None:
            approved, reason = self._approval_hook(
                {"operation": "weight_compounding_cycle", "generation_id": generation_id}
            )
            if not approved:
                receipt.status = "blocked"
                receipt.reasons = [f"approval_denied:{reason}"]
                receipt.elapsed_s = time.time() - started
                return receipt

        if not self._acquire_lock():
            receipt.status = "blocked"
            receipt.reasons = ["cycle_already_running"]
            receipt.elapsed_s = time.time() - started
            return receipt

        def record(*, promoted: bool) -> None:
            receipt.elapsed_s = time.time() - started
            self._record_generation(receipt, promoted=promoted)

        try:
            base_model, base_source = self.resolve_base()
            receipt.base_model, receipt.base_source = base_model, base_source

            run_dir.mkdir(parents=True, exist_ok=True)
            battery_seed = self.config.battery_seed_base + generation_seq
            battery_tasks = generate_battery(
                BatterySpec(seed=battery_seed, size=self.config.battery_size)
            )

            # Harvest before admission: it is model-free, and admission's RAM
            # budget depends on the training mode (DPO holds a reference model).
            try:
                mode, data_dir, counts = self.harvest(run_dir, battery_tasks)
            except RuntimeError as exc:
                receipt.status = "blocked"
                receipt.reasons = [str(exc)]
                return receipt
            receipt.train_mode = mode
            receipt.data_counts = counts

            ok, reasons = self.admission_check(base_model, mode)
            if not ok:
                receipt.status = "blocked"
                receipt.reasons = reasons
                return receipt

            adapter_dir = run_dir / "adapter"
            trained, train_detail = self.train(base_model, data_dir, adapter_dir, mode)
            if not trained:
                receipt.reasons = [f"training_failed:{train_detail[-400:]}"]
                record(promoted=False)
                return receipt

            incumbent = self._run_heldout(
                base_model, "", battery_seed, self.config.battery_size,
                run_dir / "incumbent_eval.json",
            )
            candidate = self._run_heldout(
                base_model, str(adapter_dir), battery_seed, self.config.battery_size,
                run_dir / "candidate_eval.json",
            )
            if incumbent is None or candidate is None:
                receipt.reasons.append("eval_gate_unavailable")
                record(promoted=False)
                return receipt

            # The hidden battery on BOTH models. It used to run on the
            # candidate only, which made it unusable as a gate: with nothing
            # to compare against, `hidden_passed = hidden is not None` was
            # the best anyone could write, and a full eval ran every cycle
            # without being able to refuse anything. Its whole purpose is to
            # catch a candidate that learned the visible battery, and that is
            # a comparison.
            hidden_seed = battery_seed + HIDDEN_SEED_OFFSET
            hidden_incumbent = self._run_heldout(
                base_model, "", hidden_seed, self.config.hidden_battery_size,
                run_dir / "hidden_incumbent_eval.json",
            )
            hidden = self._run_heldout(
                base_model, str(adapter_dir),
                hidden_seed, self.config.hidden_battery_size,
                run_dir / "hidden_eval.json",
            )

            receipt.incumbent_accuracy = float(incumbent.get("accuracy", 0.0))
            receipt.candidate_accuracy = float(candidate.get("accuracy", 0.0))
            receipt.hidden_accuracy = float(hidden.get("accuracy", 0.0)) if hidden else None
            receipt.hidden_incumbent_accuracy = (
                float(hidden_incumbent.get("accuracy", 0.0)) if hidden_incumbent else None
            )

            identity_ok, identity_hits = self._identity_scan(
                run_dir / "candidate_eval.responses.jsonl"
            )
            receipt.identity_ok = identity_ok

            # ── the three gates ───────────────────────────────────────────
            #
            # epsilon is tolerance for battery noise, not a per-generation
            # licence to decline. Measured against the lineage's HIGH-WATER
            # mark rather than the immediate parent: comparing only to the
            # parent let each generation give back epsilon and still pass, so
            # twenty "successful" generations could walk accuracy down by
            # 20*epsilon with every single receipt reading "no regression".
            # A floor that moves down with you is not a floor.
            high_water = self._capability_high_water(receipt.incumbent_accuracy)
            receipt.high_water_accuracy = high_water
            floor = high_water - self.config.epsilon
            regression_passed = receipt.candidate_accuracy >= floor

            # The hidden battery now decides something. Same seed on both
            # models, same comparison: a candidate that gained on the visible
            # battery and lost on the unseen one has learned the battery, and
            # that is the failure this second eval was paid for.
            if hidden is None or hidden_incumbent is None:
                hidden_passed = False
                hidden_reason = "hidden_eval_unavailable"
            else:
                hidden_passed = (
                    receipt.hidden_accuracy
                    >= receipt.hidden_incumbent_accuracy - self.config.epsilon
                )
                hidden_reason = (
                    f"hidden_regressed:{receipt.hidden_accuracy:.3f}"
                    f"<{receipt.hidden_incumbent_accuracy:.3f}-eps"
                )

            # The contract report that eval_before_promotion.AdapterEvaluator
            # reads — this loop is the producer that contract was waiting for.
            evaluation_report = {
                "passed_safety": identity_ok,
                "accuracy_score": receipt.candidate_accuracy,
                "regression_passed": regression_passed,
                "hidden_eval_passed": hidden_passed,
                "promotion_threshold": max(0.0, floor),
                "incumbent_accuracy": receipt.incumbent_accuracy,
                "high_water_accuracy": high_water,
                "hidden_accuracy": receipt.hidden_accuracy,
                "hidden_incumbent_accuracy": receipt.hidden_incumbent_accuracy,
                "identity_hits": identity_hits,
                "battery_seed": battery_seed,
                "hidden_battery_seed": hidden_seed,
                "generation_id": generation_id,
            }
            atomic_write_text(
                adapter_dir / "evaluation_report.json",
                json.dumps(evaluation_report, indent=2, sort_keys=True),
                encoding="utf-8",
            )

            if not (regression_passed and hidden_passed and identity_ok):
                receipt.status = "refused"
                if not regression_passed:
                    receipt.reasons.append(
                        f"capability_regressed:{receipt.candidate_accuracy:.3f}"
                        f"<{high_water:.3f}-eps"
                    )
                if not hidden_passed:
                    receipt.reasons.append(hidden_reason)
                if not identity_ok:
                    receipt.reasons.append(f"identity_regression:{identity_hits}")
                record(promoted=False)
                return receipt

            candidate_path = ""
            candidate_receipt_path = ""
            if self.config.publish:
                candidate_path, candidate_receipt_path, publish_errors = (
                    self.fuse_and_record_candidate(
                        base_model,
                        adapter_dir,
                        generation_id,
                        metadata={
                            "train_mode": mode,
                            "data_counts": counts,
                            "evaluation": evaluation_report,
                        },
                    )
                )
                if publish_errors:
                    receipt.reasons.extend(publish_errors)
                    # A memory deferral is not a failure: the adapter is
                    # trained, gated, and preserved on disk for a later window
                    # or an operator fuse. Distinguish it so the scheduler's
                    # cooldown and the lineage read it honestly.
                    if any(e.startswith("fuse_deferred_memory") for e in publish_errors):
                        receipt.status = "deferred"
                    record(promoted=False)
                    return receipt
            receipt.candidate_model_path = candidate_path
            receipt.candidate_receipt_path = candidate_receipt_path
            receipt.status = "candidate" if candidate_path else "qualified_adapter"
            record(promoted=False)
            return receipt

        except _RECOVERABLE as exc:
            record_degradation(
                "weight_compounding",
                exc,
                action="closed failed compounding cycle without promotion",
            )
            receipt.reasons.append(f"cycle_error:{exc}")
            return receipt
        finally:
            receipt.elapsed_s = time.time() - started
            try:
                if run_dir.exists():
                    atomic_write_text(
                        run_dir / "cycle_receipt.json",
                        json.dumps(receipt.to_dict(), indent=2, sort_keys=True),
                        encoding="utf-8",
                    )
            except _RECOVERABLE as exc:
                record_degradation(
                    "weight_compounding",
                    exc,
                    action="cycle completed but receipt file could not be written",
                )
            self._release_lock()

    # ── ledger ───────────────────────────────────────────────────────────────

    def _capability_high_water(self, incumbent_accuracy: float) -> float:
        """The best visible-battery accuracy any promoted generation reached.

        The promotion floor was ``incumbent - epsilon``, which meant each
        generation could give back epsilon and still record "no regression".
        Twenty such generations walk accuracy down by twenty epsilons with
        twenty clean receipts behind them — a ratchet pointed the wrong way.
        Measuring from the high-water mark makes epsilon what it was meant to
        be: tolerance for battery noise around a level already reached.

        Read from the tamper-evident ledger, and only from generations that
        actually promoted — a refused candidate's score never set a bar the
        serving model has to clear. Falls back to the live incumbent, which
        is the honest answer on generation 1 and after any ledger it cannot
        read: never a floor lower than what is serving right now.
        """
        best = float(incumbent_accuracy)
        try:
            for record in self._ledger.load_records():
                if not getattr(record, "promoted", False):
                    continue
                best = max(best, float(getattr(record, "after_score", 0.0) or 0.0))
        except _RECOVERABLE as exc:
            record_degradation(
                "weight_compounding",
                exc,
                action=(
                    "used the live incumbent as the promotion floor after the "
                    "lineage ledger could not be read"
                ),
            )
        return best

    def _record_generation(self, receipt: CycleReceipt, *, promoted: bool) -> None:
        records = self._ledger.load_records()
        parent = records[-1].generation_id if records else None
        adapter_dir = Path(receipt.run_dir) / "adapter"
        record = RSIGenerationRecord(
            generation_id=receipt.generation_id,
            parent_generation_id=parent,
            hypothesis=(
                f"{receipt.train_mode or 'unknown'} training on harvested experience "
                f"({receipt.data_counts}) improves held-out verifier accuracy"
            ),
            intervention_type=f"weight_lora_{receipt.train_mode or 'unknown'}",
            artifact_hashes={
                "base_model": _artifact_digest(Path(receipt.base_model)) if receipt.base_model else "unavailable",
                "adapter": _artifact_digest(adapter_dir / "adapters.safetensors"),
                "promoted_model": (
                    _artifact_digest(Path(receipt.promoted_model_path))
                    if receipt.promoted_model_path else "none"
                ),
                "candidate_model": (
                    _artifact_digest(Path(receipt.candidate_model_path))
                    if receipt.candidate_model_path else "none"
                ),
            },
            baseline_score=float(receipt.incumbent_accuracy or 0.0),
            after_score=float(receipt.candidate_accuracy or 0.0),
            hidden_eval_score=float(receipt.hidden_accuracy or 0.0),
            regressions=[r for r in receipt.reasons if "regress" in r],
            promoted=promoted,
            rollback_performed=False,
            time_to_valid_improvement_s=receipt.elapsed_s,
            # NOT candidate_accuracy. This recorded the capability score a
            # second time under a different name, so a rising capability curve
            # produced an identically rising "improver" curve and the
            # two-inequality strong-RSI test was satisfied by one measurement
            # counted twice.
            #
            # The improver score has to measure the improver: verified
            # capability gain per hour of the cycle that produced it. That
            # comes apart from capability exactly where it matters — under
            # diminishing returns capability keeps climbing while each
            # increment costs more, so this falls while after_score rises.
            # rsi_lineage.improver_curve_dependence now refuses a strong
            # verdict if the two curves are ever affinely related again.
            improver_score=improver_efficiency(
                baseline_score=float(receipt.incumbent_accuracy or 0.0),
                after_score=float(receipt.candidate_accuracy or 0.0),
                cost_s=float(receipt.elapsed_s or 0.0),
            ),
            improver_provenance=(
                PROVENANCE_MEASURED
                if float(receipt.elapsed_s or 0.0) > 0.0
                else PROVENANCE_UNMEASURED
            ),
            improver_measurement={
                "baseline_score": float(receipt.incumbent_accuracy or 0.0),
                "after_score": float(receipt.candidate_accuracy or 0.0),
                "cost_s": float(receipt.elapsed_s or 0.0),
                "source": "weight_compounding cycle receipt",
            },
        )
        try:
            entry = self._ledger.append(record)
            receipt.ledger_entry_hash = str(entry.get("entry_hash", ""))
        except OSError as exc:
            record_degradation(
                "weight_compounding",
                exc,
                action="cycle result NOT recorded in lineage ledger",
                severity="error",
            )

    # ── public evidence surface ──────────────────────────────────────────────

    def lineage_verdict(self) -> RSILineageVerdict:
        """The only place a compounding claim may come from: the receipts."""
        return evaluate_lineage(self._ledger.load_records())

    def verify_ledger(self) -> tuple[bool, list[str]]:
        return self._ledger.verify()

    def data_readiness(self) -> dict[str, Any]:
        """Cheap pre-check for schedulers: raw row counts vs thresholds.

        Approximate by design (no contamination/seal filtering) — the real
        harvest re-validates. This exists so the scheduler can decline without
        touching a model or writing a receipt.
        """
        def _count(path: Path | None) -> int:
            if path is None or not Path(path).exists():
                return 0
            try:
                with Path(path).open(encoding="utf-8") as fh:
                    return sum(1 for line in fh if line.strip())
            except OSError:
                return 0

        dpo = _count(self.config.dpo_store_path)
        sft = _count(self.config.sft_buffer_path)
        return {
            "dpo_rows": dpo,
            "sft_rows": sft,
            "min_dpo_pairs": self.config.min_dpo_pairs,
            "min_sft_examples": self.config.min_sft_examples,
            "ready": dpo >= self.config.min_dpo_pairs or sft >= self.config.min_sft_examples,
        }

    def stats(self) -> dict[str, Any]:
        records = self._ledger.load_records()
        verdict = evaluate_lineage(records)
        return {
            "generations": len(records),
            "promoted": sum(1 for r in records if r.promoted),
            "refused": sum(1 for r in records if not r.promoted),
            "capability_curve": [round(r.after_score, 4) for r in records],
            "verdict": verdict.verdict,
            "verdict_reasons": verdict.reasons,
            "ledger_path": str(self.config.ledger_path),
            "ledger_intact": self._ledger.verify()[0],
        }


def default_config(**overrides: Any) -> CompoundingConfig:
    """Runtime-flavored config: live buffers, live fused root, autonomous caps."""
    from core.config import get_config

    cfg = get_config()
    data_dir = Path(cfg.paths.data_dir)
    repo = Path(getattr(cfg.paths, "project_root", Path.cwd()))
    base: dict[str, Any] = {
        "work_root": data_dir / "learning" / "compounding",
        "fused_root": repo / "training" / "fused-model",
        "sft_buffer_path": data_dir / "learning" / "experience_buffer.jsonl",
        "dpo_store_path": data_dir / "verifiable_preferences.jsonl",
    }
    if "default_base" not in overrides and not overrides.get("model_override"):
        # No fuse has ever been published (fresh install / worktree): fall back
        # to the serving model the registry resolves, so cycle 0 has a parent.
        try:
            from core.brain.llm.model_registry import get_model_path

            resolved = get_model_path()
            if resolved and Path(resolved).exists():
                base["default_base"] = resolved
        except (ImportError, AttributeError, RuntimeError, OSError) as exc:
            record_degradation(
                "weight_compounding",
                exc,
                action="left default base unset; cycles require a manifest or override",
                severity="debug",
            )
    base.update(overrides)
    return CompoundingConfig(**base)


__all__ = [
    "CompoundingConfig",
    "CycleReceipt",
    "WeightCompoundingLoop",
    "default_config",
]
